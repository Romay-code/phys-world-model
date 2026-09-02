"""L0 token 化 + L1 状态编码器。

L0: u_t[n] = Proj_type([x⊙avail ; avail]) + Emb_type(type[n]) + W_desc·desc[n]
    - avail 与数值一起拼进投影，不是只做乘法：模型必须能区分「缺测」与「值为 0」
    - 历史观测 y 入窗（继承 use_hist_y），当前帧的观测位由调用方置 0 并标记

L1: [B,W,N,d] -> [B,N,d]
    每层：PreLN -> 空间注意力 -> 残差 -> PreLN -> 时间注意力 -> 残差
          -> PreLN -> FFN -> 残差 -> FiLM
    末尾：末帧 ⊕ 均值池化 -> 线性降维，把 W 维吃掉
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..data import schema as S
from .layers import AttnBiasMLP, FeedForward, FiLM, MultiHeadAttention, build_rope_cache


class TokenEmbed(nn.Module):
    """L0。每个设备族一套独立的输入投影，族内所有设备共享。"""

    def __init__(self, d: int, d_desc: int, n_type: int,
                 field_dims: dict[str, int], f_max: int):
        super().__init__()
        self.f_max = f_max
        self.n_type = n_type
        self.proj = nn.ModuleList([
            nn.Linear(2 * field_dims[fam], d) for fam in S.FAMILIES])
        self.field_dims = [field_dims[fam] for fam in S.FAMILIES]
        self.type_emb = nn.Embedding(n_type, d)
        self.desc_proj = nn.Linear(d_desc, d, bias=False)
        nn.init.normal_(self.type_emb.weight, std=0.02)

    def forward(self, x: torch.Tensor, avail: torch.Tensor,
                type_id: torch.Tensor, desc: torch.Tensor) -> torch.Tensor:
        """x, avail [B,W,N,F] ; type_id [N] ; desc [N,d_desc] -> [B,W,N,d]"""
        B, W, N, _ = x.shape
        out = None
        for t, nf in enumerate(self.field_dims):
            sel = type_id == t
            if not bool(sel.any()):
                continue
            xs = x[:, :, sel, :nf]
            av = avail[:, :, sel, :nf]
            h = self.proj[t](torch.cat([xs * av, av], dim=-1))
            if out is None:
                out = x.new_zeros(B, W, N, h.shape[-1])
            out[:, :, sel] = h
        assert out is not None
        return out + self.type_emb(type_id) + self.desc_proj(desc)


class EncoderBlock(nn.Module):
    def __init__(self, d: int, n_head: int, d_ctx: int, dropout: float = 0.0):
        super().__init__()
        self.n1, self.n2, self.n3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.spatial = MultiHeadAttention(d, n_head, dropout)
        self.temporal = MultiHeadAttention(d, n_head, dropout)
        self.ff = FeedForward(d, 4, dropout)
        self.film = FiLM(d_ctx, d)

    def forward(self, h: torch.Tensor, *, bias, key_mask, rope, site_ctx,
                need_attn: bool = False):
        """h [B,W,N,d]"""
        B, W, N, d = h.shape

        # 空间：把 W 并进 batch。
        # bias 形状 [n_head,N,N]，与 batch 和帧都无关，直接靠广播加到 [B*W,h,N,N]
        # 的打分上 —— 不要 expand/repeat_interleave 成实体张量，那是 45MB/层的
        # 无谓拷贝（前向+反向 4 层共 360MB），实测是这一层的主要开销。
        z = self.n1(h).reshape(B * W, N, d)
        km = (key_mask[:, None, :].expand(B, W, N).reshape(B * W, N)
              if key_mask is not None else None)
        a_out, attn = self.spatial(z, bias=bias, key_mask=km, need_weights=need_attn)
        h = h + a_out.reshape(B, W, N, d)

        # 时间：把 N 并进 batch
        z = self.n2(h).permute(0, 2, 1, 3).reshape(B * N, W, d)
        t_out, _ = self.temporal(z, causal=True, rope=rope)
        h = h + t_out.reshape(B, N, W, d).permute(0, 2, 1, 3)

        h = h + self.ff(self.n3(h))
        h = self.film(h, site_ctx)
        return h, attn


class Encoder(nn.Module):
    def __init__(self, *, d: int = 192, n_head: int = 6, n_layer: int = 4,
                 d_desc: int = 32, d_rel: int = 5, d_ctx: int = 8,
                 field_dims: dict[str, int], f_max: int, dropout: float = 0.0):
        super().__init__()
        self.d, self.n_head = d, n_head
        self.embed = TokenEmbed(d, d_desc, len(S.FAMILIES), field_dims, f_max)
        self.bias_mlp = AttnBiasMLP(d_desc, d_rel, len(S.FAMILIES), n_head)
        self.blocks = nn.ModuleList(
            [EncoderBlock(d, n_head, d_ctx, dropout) for _ in range(n_layer)])
        self.norm = nn.LayerNorm(d)
        self.pool = nn.Linear(2 * d, d)
        # 编码器输出必须归一化。`pool` 是裸 Linear，其权重可以自由增长，而
        # 归一化后的 lat 项对 z 的尺度不变 —— 于是没有任何东西约束 |z| 的量级。
        # 实测：|z| 从 1.22 指数增长到 10.09（每 10 epoch 约 3 倍），随后撞上
        # 数值边界暴力回弹到 1.86，train 从 0.046 跳到 1.68。
        #
        # 对 P3 更要紧：H=48 的推演里，转移网络的 Lipschitz 分析默认状态有界；
        # z_0 自己在漂就无从谈起。归一化 z_true 之后，lam_lat 还能通过
        # 「向 z_true 靠拢」间接约束整段 rollout 的尺度漂移。
        self.out_norm = nn.LayerNorm(d)

    def compute_bias(self, desc, stat_rel, type_id) -> torch.Tensor:
        """[n_head,N,N]。desc/stat_rel 时不变，整段 rollout 只需算一次。"""
        return self.bias_mlp(desc, stat_rel, type_id)

    def forward(self, x, avail, *, desc, stat_rel, type_id, site_ctx,
                key_mask=None, bias=None, need_attn: bool = False):
        """x, avail [B,W,N,F] -> z [B,N,d], attn（最后一层的空间注意力，可选）"""
        B, W, N, _ = x.shape
        if bias is None:
            bias = self.compute_bias(desc, stat_rel, type_id)
        h = self.embed(x, avail, type_id, desc)
        rope = build_rope_cache(W, self.d // self.n_head, x.device, x.dtype)

        attn = None
        for i, blk in enumerate(self.blocks):
            last = i == len(self.blocks) - 1
            h, a = blk(h, bias=bias, key_mask=key_mask, rope=rope,
                       site_ctx=site_ctx, need_attn=need_attn and last)
            if a is not None:
                attn = a

        h = self.norm(h)
        z = self.out_norm(self.pool(torch.cat([h[:, -1], h.mean(1)], dim=-1)))
        if attn is not None:
            # [B*W,h,N,N] -> 取末帧 -> [B,h,N,N]
            attn = attn.reshape(B, W, self.n_head, N, N)[:, -1]
        return z, attn
