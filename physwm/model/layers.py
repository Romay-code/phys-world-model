"""注意力与基础层。

两条与常规 Transformer 不同的地方，都是刻意的：

1. **空间轴不加任何位置编码**。token 的身份完全由 Emb_type + W_desc·desc 承载，
   两者都与索引无关。一旦加入按索引的 PE，置换等变立即破坏，跨站迁移失效。
2. **时间轴用 RoPE**。窗口起点在推理时会漂移，需要外推能力，可学习绝对 PE 不行。

attn_bias 的输入含 type 的 one-hot —— 只有 stat_rel 的「同族」标志不足以区分
「冷机看塔」与「冷冻泵看塔」，那两者应当有不同的偏置。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

NEG_INF = -1e9


# --------------------------------------------------------------------------
# RoPE
# --------------------------------------------------------------------------

def build_rope_cache(seq_len: int, d_head: int, device, dtype,
                     base: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    """返回 cos/sin，形状 [seq_len, d_head]。pos 取以 t0 为原点的负偏移。"""
    assert d_head % 2 == 0
    half = d_head // 2
    inv = base ** (-torch.arange(half, device=device, dtype=torch.float32) / half)
    pos = torch.arange(seq_len, device=device, dtype=torch.float32) - (seq_len - 1)
    ang = pos[:, None] * inv[None, :]                 # [L, half]
    cos = torch.cat([ang.cos(), ang.cos()], dim=-1)
    sin = torch.cat([ang.sin(), ang.sin()], dim=-1)
    return cos.to(dtype), sin.to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x [..., L, d_head]，只作用于 Q/K，不作用于 V。"""
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    rot = torch.cat([-x2, x1], dim=-1)
    return x * cos + rot * sin


# --------------------------------------------------------------------------
# attn_bias
# --------------------------------------------------------------------------

class AttnBiasMLP(nn.Module):
    """attn_bias(i,j) = MLP(desc_i, desc_j, type_i, type_j, stat_rel_ij) -> [n_head]

    desc 与 stat_rel 都是时不变的，所以每个站点只需算一次并缓存；
    rollout 期间直接复用，不重复计算。
    """

    def __init__(self, d_desc: int, d_rel: int, n_type: int, n_head: int,
                 hidden: int = 64):
        super().__init__()
        self.n_type = n_type
        d_in = 2 * d_desc + 2 * n_type + d_rel
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.GELU(), nn.Linear(hidden, n_head))
        # 零初始化末层 -> 起点 bias ≡ 0，退化为纯 QK^T，结构先验从零生长
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, desc: torch.Tensor, stat_rel: torch.Tensor,
                type_id: torch.Tensor) -> torch.Tensor:
        """desc [N,d_desc]  stat_rel [N,N,d_rel]  type_id [N] -> [n_head,N,N]"""
        N = desc.shape[0]
        th = F.one_hot(type_id, self.n_type).to(desc.dtype)      # [N, n_type]
        di = torch.cat([desc, th], -1)                            # [N, d_desc+n_type]
        a = di[:, None, :].expand(N, N, -1)
        b = di[None, :, :].expand(N, N, -1)
        z = torch.cat([a, b, stat_rel], dim=-1)                   # [N,N,d_in]
        return self.net(z).permute(2, 0, 1).contiguous()          # [n_head,N,N]


# --------------------------------------------------------------------------
# 注意力
# --------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    """标准 MHA，额外支持加性偏置与 key padding 掩码。"""

    def __init__(self, d: int, n_head: int, dropout: float = 0.0):
        super().__init__()
        assert d % n_head == 0
        self.d, self.h, self.dh = d, n_head, d // n_head
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.o = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, *,
                bias: torch.Tensor | None = None,
                key_mask: torch.Tensor | None = None,
                causal: bool = False,
                rope: tuple[torch.Tensor, torch.Tensor] | None = None,
                need_weights: bool = False):
        """x [B,L,d]
        bias      [h,L,L] 或 [B,h,L,L]，加在 softmax 之前
        key_mask  [B,L]，True=有效
        """
        B, L, _ = x.shape
        q = self.q(x).view(B, L, self.h, self.dh).transpose(1, 2)   # [B,h,L,dh]
        k = self.k(x).view(B, L, self.h, self.dh).transpose(1, 2)
        v = self.v(x).view(B, L, self.h, self.dh).transpose(1, 2)

        if rope is not None:
            cos, sin = rope
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

        # 加性掩码：pad 位置置 -inf。允许 pad token attend 自身，避免整行 -inf
        # 让 softmax 出 NaN —— 它的输出在最后被乘 0 丢弃。
        add_mask = None
        if bias is not None:
            add_mask = bias if bias.dim() == 4 else bias.unsqueeze(0)
        if key_mask is not None:
            eye = torch.eye(L, device=x.device, dtype=torch.bool)[None, None]
            km = (~key_mask[:, None, None, :] & ~eye).to(q.dtype) * NEG_INF
            add_mask = km if add_mask is None else add_mask + km

        if not need_weights:
            # 融合核（flash / mem-efficient）。不显式构造 [B,h,L,L] 打分矩阵 ——
            # 空间注意力上那是 [1024,6,43,43]=45MB/层，前向加反向是主要开销。
            if causal and add_mask is None:
                out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            else:
                m = add_mask
                if causal:
                    tri = torch.ones(L, L, device=x.device, dtype=torch.bool).tril()
                    cm = torch.zeros(L, L, device=x.device, dtype=q.dtype)
                    m = cm.masked_fill(~tri, NEG_INF) if m is None else \
                        m + cm.masked_fill(~tri, NEG_INF)
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=m)
            a = None
        else:
            # 只有需要读注意力矩阵时（末层，供软分组 alpha）才走手写路径
            s = (q @ k.transpose(-1, -2)) / math.sqrt(self.dh)      # [B,h,L,L]
            if add_mask is not None:
                s = s + add_mask
            if causal:
                tri = torch.ones(L, L, device=x.device, dtype=torch.bool).tril()
                s = s.masked_fill(~tri, NEG_INF)
            a = self.drop(s.softmax(-1))
            out = a @ v

        out = self.o(out.transpose(1, 2).reshape(B, L, self.d))
        if key_mask is not None:
            out = out * key_mask[..., None].to(out.dtype)
        return out, a


class FeedForward(nn.Module):
    def __init__(self, d: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, d * mult), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d * mult, d))

    def forward(self, x):
        return self.net(x)


class FiLM(nn.Module):
    """站点调制。gamma/beta 末层零初始化 -> 起点是恒等映射。

    site_ctx 由站级统计量算出，不是可学习查表 —— 新站点接入不需要新参数。
    """

    def __init__(self, d_ctx: int, d: int, hidden: int = 32):
        super().__init__()
        self.g = nn.Sequential(nn.Linear(d_ctx, hidden), nn.GELU(), nn.Linear(hidden, d))
        self.b = nn.Sequential(nn.Linear(d_ctx, hidden), nn.GELU(), nn.Linear(hidden, d))
        for m in (self.g, self.b):
            nn.init.zeros_(m[-1].weight)
            nn.init.zeros_(m[-1].bias)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        """x [B,...,d]，ctx [B,d_ctx]"""
        shape = (ctx.shape[0],) + (1,) * (x.dim() - 2) + (x.shape[-1],)
        gamma = 1.0 + self.g(ctx).view(shape)
        beta = self.b(ctx).view(shape)
        return gamma * x + beta
