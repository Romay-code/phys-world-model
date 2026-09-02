"""L2 状态转移 F : (z_t, a_t, d_t) -> z_{t+1}

    z_{t+1} = z_t + delta * G(z_t, a_t, d_t)

三条设计要点：
    1. 残差/欧拉形式 —— 小步长下天然稳定，不必让网络从零学「保持不变」
    2. 谱归一化压 Lipschitz 常数 —— L<=1 时 H 步误差线性增长而非指数发散
    3. **无时间注意力、无因果掩码** —— z 已经没有 W 维了，转移只看当前隐状态

alpha（软分组权重）从 G 最后一层的空间注意力现取，不是从编码器冻结下来的：
QK^T 依赖 z，z 每步在变。冻结 alpha 会让反事实推演（改塔的启停）失效。
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm

from ..data import schema as S
from .layers import FeedForward, MultiHeadAttention


class ActionEmbed(nn.Module):
    """动作 a 与外生量 d 按同一套 token 划分投影，与 z 逐 token 相加。

    a 落在对应设备的 token 上（塔动作 -> tower token）；
    d 只进 plant token，再经空间注意力扩散到全体。
    """

    def __init__(self, d: int, field_dims: dict[str, int]):
        super().__init__()
        self.role_idx = {fam: S.role_index(fam) for fam in S.FAMILIES}
        self.proj = nn.ModuleDict()
        for fam in S.FAMILIES:
            n_a = len(self.role_idx[fam]["a"]) + len(self.role_idx[fam]["d"])
            if n_a > 0:
                self.proj[fam] = nn.Linear(2 * n_a, d)

    def forward(self, ad: torch.Tensor, avail: torch.Tensor,
                type_id: torch.Tensor) -> torch.Tensor:
        """ad, avail [B,N,F] -> [B,N,d]（观测位已被调用方置零）"""
        B, N, _ = ad.shape
        out = None
        for t, fam in enumerate(S.FAMILIES):
            if fam not in self.proj:
                continue
            sel = type_id == t
            if not bool(sel.any()):
                continue
            k = self.role_idx[fam]["a"] + self.role_idx[fam]["d"]
            xs, av = ad[:, sel][..., k], avail[:, sel][..., k]
            h = self.proj[fam](torch.cat([xs * av, av], dim=-1))
            if out is None:
                out = ad.new_zeros(B, N, h.shape[-1])
            out[:, sel] = h
        return out if out is not None else ad.new_zeros(B, N, 1)


class TransitionBlock(nn.Module):
    def __init__(self, d: int, n_head: int, sn: bool = True):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = MultiHeadAttention(d, n_head)
        self.ff = FeedForward(d, 4)
        if sn:
            for m in (self.attn.q, self.attn.k, self.attn.v, self.attn.o):
                spectral_norm(m)
            self.ff.net[0] = spectral_norm(self.ff.net[0])
            self.ff.net[3] = spectral_norm(self.ff.net[3])

    def forward(self, h, *, bias, key_mask, need_attn=False):
        a, attn = self.attn(self.n1(h), bias=bias, key_mask=key_mask,
                            need_weights=need_attn)
        h = h + a
        h = h + self.ff(self.n2(h))
        return h, attn


class Transition(nn.Module):
    def __init__(self, *, d: int = 192, n_head: int = 6, n_layer: int = 2,
                 field_dims: dict[str, int], delta: float = 1.0,
                 spectral: bool = True):
        super().__init__()
        self.delta = delta
        self.act = ActionEmbed(d, field_dims)
        self.blocks = nn.ModuleList(
            [TransitionBlock(d, n_head, spectral) for _ in range(n_layer)])
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, d)
        # 零初始化输出层 -> 起点 G ≡ 0，转移退化为恒等，误差不会一开始就爆
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, z: torch.Tensor, ad: torch.Tensor, ad_avail: torch.Tensor,
                *, type_id, bias=None, key_mask=None, need_attn: bool = False):
        """z [B,N,d] -> z' [B,N,d], attn [B,h,N,N]"""
        h = z + self.act(ad, ad_avail, type_id)
        attn = None
        for i, blk in enumerate(self.blocks):
            last = i == len(self.blocks) - 1
            h, a = blk(h, bias=bias, key_mask=key_mask,
                       need_attn=need_attn and last)
            if a is not None:
                attn = a
        g = self.out(self.norm(h))
        return z + self.delta * g, attn

    @torch.no_grad()
    def lipschitz_estimate(self) -> float:
        """整个转移映射 z -> z + delta*G(z) 的 Lipschitz 上界估计。

        注意两点：
        1. **恒 >= 1**。残差形式下 L = 1 + delta*Lip(G)，所以「L <= 1」不可达。
           有意义的量是 eps = delta*Lip(G)：H 步误差约 exp(eps*H)，
           要 H=48 不发散需 eps <~ 0.02。P3 要扫的是 delta。
        2. 必须把输出投影 `self.out` 算进去 —— 它零初始化且**不做**谱归一化，
           训练初期 sigma(out)=0 使 Lip(G)=0、L=1。漏掉它会得到虚高的估计
           （之前报的 2.32 就是这么来的，实际起点是 1.00）。

        这是上界不是精确值，逐层奇异值连乘会明显高估。
        """
        prod = 1.0
        for blk in self.blocks:
            for m in (blk.attn.q, blk.attn.k, blk.attn.v, blk.attn.o,
                      blk.ff.net[0], blk.ff.net[3]):
                w = getattr(m, "weight", None)
                if w is not None:
                    prod *= float(torch.linalg.matrix_norm(w.detach(), ord=2))
        prod *= float(torch.linalg.matrix_norm(self.out.weight.detach(), ord=2))
        return 1.0 + self.delta * prod * (1.1 ** len(self.blocks))
