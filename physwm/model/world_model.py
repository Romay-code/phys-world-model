"""世界模型总装：E -> F^H -> D，以及两种推演模式。

推演循环（设计目标「在隐状态空间内按控制周期逐步推进」）：

    z <- E(真实窗口_{t0})                      整段只调用一次
    for h in 1..H:
        z <- F(z, a_{t0+h}, d_{t0+h})
        o_hat <- D(z, a_{t0+h}, d_{t0+h})

rollout 期间**不重新编码窗口**，解码出的观测是输出，不回灌编码器 —— 这是单步
推理 <= 50ms 的前提（1 次 F + 1 次 D，无 W 帧重编码）。

训练时的隐状态重锚定：
    z <- (1-m) * F(z,a,d) + m * E(真实窗口_{t0+h})     m ~ Bernoulli(p)
真值编码分支带 stop-gradient，避免编码器为迎合转移网络而退化。
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..data import schema as S
from .decoder import PhysicsDecoder
from .encoder import Encoder
from .transition import Transition


@dataclass
class ModelConfig:
    d: int = 192
    n_head: int = 6
    n_enc: int = 4
    n_trans: int = 2
    d_desc: int = 32
    d_rel: int = 5
    d_ctx: int = 8
    dropout: float = 0.0
    delta: float = 1.0
    spectral: bool = True
    q_mode: str = "from_w"   # "free" 为消融用（不可辨识，仅作对照）
    # "free" 复现 §13 #24 的病灶（dt_evap 独立预测 -> eta 无梯度），仅作对照臂
    dt_evap_mode: str = "soft"
    # 冷凝侧接线（cool_dt = q_cond/mcp_cool，不动点解）。**默认关闭。**
    # 数学上它能辨识 eta 尺度，但 yb3 实测不可行：W = Q_cond - Q_evap 是两个
    # 高度相关大量之间 15% 的差（Q_cond/Q_evap = 1+1/COP ~ 1.15），
    # 回归量相关性 0.932、条件数 24.2，解出 k_cold < 0。信噪比不够。
    # 保留为消融臂 —— 在有流量的站（pb1）信噪比高得多，可能可用。
    cool_dt_mode: str = "free"
    # m·cp 的来源。"flow" = 泵频驱动（默认）；"const" = 每机常数，已被实测否掉，仅消融
    mcp_mode: str = "flow"


class WorldModel(nn.Module):
    def __init__(self, cfg: ModelConfig, n_dev: dict[str, int], f_max: int):
        super().__init__()
        self.cfg = cfg
        self.n_dev = dict(n_dev)
        field_dims = {fam: len(S.CANON_FIELDS[fam]) for fam in S.FAMILIES}
        self.f_max = f_max

        self.encoder = Encoder(
            d=cfg.d, n_head=cfg.n_head, n_layer=cfg.n_enc, d_desc=cfg.d_desc,
            d_rel=cfg.d_rel, d_ctx=cfg.d_ctx, field_dims=field_dims,
            f_max=f_max, dropout=cfg.dropout)
        self.transition = Transition(
            d=cfg.d, n_head=cfg.n_head, n_layer=cfg.n_trans,
            field_dims=field_dims, delta=cfg.delta, spectral=cfg.spectral)
        self.decoder = PhysicsDecoder(cfg.d, n_dev, q_mode=cfg.q_mode,
                                      dt_evap_mode=cfg.dt_evap_mode,
                                      cool_dt_mode=cfg.cool_dt_mode,
                                      mcp_mode=cfg.mcp_mode,
                                      d_desc=cfg.d_desc)

        # 观测位在「当前帧」要被置零 —— 继承 use_hist_y：历史帧给真值，当前帧不给
        obs_mask = torch.zeros(len(S.FAMILIES), f_max)
        for t, fam in enumerate(S.FAMILIES):
            for k in S.role_index(fam)["o"]:
                obs_mask[t, k] = 1.0
        self.register_buffer("obs_field_mask", obs_mask)

    # -- 辅助 --------------------------------------------------------------

    def split_by_family(self, z: torch.Tensor, type_id: torch.Tensor
                        ) -> dict[str, torch.Tensor]:
        return {fam: z[:, type_id == t] for t, fam in enumerate(S.FAMILIES)}

    def strip_obs(self, x: torch.Tensor, avail: torch.Tensor, type_id: torch.Tensor
                  ) -> tuple[torch.Tensor, torch.Tensor]:
        """把观测位置零并标记不可用，只留 a 与 d。用于转移网络的输入。"""
        m = self.obs_field_mask[type_id]                    # [N,F]
        keep = 1.0 - m
        return x * keep, avail * keep

    def alpha_from_attn(self, attn: torch.Tensor, type_id: torch.Tensor
                        ) -> torch.Tensor:
        """从注意力矩阵切出 chiller 行 x tower 列，行内重归一 -> [B, n_ch, n_tw]。"""
        ci = type_id == S.FAMILIES.index("chiller")
        ti = type_id == S.FAMILIES.index("tower")
        a = attn.mean(1)                                    # 头维平均 [B,N,N]
        a = a[:, ci][:, :, ti]
        return a / a.sum(-1, keepdim=True).clamp_min(1e-8)

    def physical_inputs(self, raw: torch.Tensor, type_id: torch.Tensor
                        ) -> dict[str, torch.Tensor]:
        """从原始量纲的一帧里取出解码器需要的物理量。raw [B,N,F]"""
        idx = {fam: (type_id == t) for t, fam in enumerate(S.FAMILIES)}

        def fld(fam: str, name: str) -> torch.Tensor:
            k = S.CANON_FIELDS[fam].index(name)
            return raw[:, idx[fam], k]

        phys = {
            "wet_bulb": fld("plant", "wet_bulb").squeeze(-1),
            "load": fld("plant", "load").squeeze(-1),   # 蒸发侧能量平衡的锚
            "cold_out": fld("chiller", "cold_out_temp"),
            "on_ch": fld("chiller", "on"),
        }
        for fam in ("tower", "coolpump", "coldpump"):
            if self.n_dev.get(fam, 0) > 0:
                phys[f"on_{fam}"] = fld(fam, "on")
                phys[f"freq_{fam}"] = fld(fam, "frequency")
        return phys

    # -- 主流程 ------------------------------------------------------------

    def encode(self, hist_x, hist_avail, *, desc, stat_rel, type_id, site_ctx,
               key_mask=None, bias=None, need_attn=False):
        return self.encoder(hist_x, hist_avail, desc=desc, stat_rel=stat_rel,
                            type_id=type_id, site_ctx=site_ctx,
                            key_mask=key_mask, bias=bias, need_attn=need_attn)

    def rollout(self, batch: dict[str, torch.Tensor], *, desc, stat_rel, type_id,
                site_ctx, H: int, W: int, reanchor_p: float = 0.0,
                key_mask=None, scales: dict | None = None, need_z_true: bool = False, lat_k: int = 0):
        """返回 preds（长度 H 的 dict 列表）与诊断量。

        reanchor_p > 0 时，每步以概率 p 用**该时刻**的真值窗口重新编码：
            z <- (1-m)*F(z,a,d) + m*E(seq[h+1 : h+1+W])     m ~ Bernoulli(p)
        注意重锚的是 t0+h 处的窗口，不是 t0 处的 —— 后者是恒定的，起不到重锚作用。
        真值分支带 stop-gradient，避免编码器为迎合转移网络而退化。

        **顺序要紧**：先用转移出的 z_pred 解码，再把重锚结果作为下一步的输入。
        若反过来（先重锚再解码），p=1 时解码的是纯真值编码，转移网络拿不到任何
        梯度，损失退化成编码器-解码器的自编码。p=1 的语义是「每步都从真值起跳
        推一步」≡ 单步预测，不是「跳过转移」。

        need_z_true=True 时额外返回逐步的真值隐状态，供隐一致性损失使用。
        """
        seq_x, seq_av, seq_raw = batch["seq_x"], batch["seq_avail"], batch["seq_raw"]
        bias = self.encoder.compute_bias(desc, stat_rel, type_id)

        enc_kw = dict(desc=desc, stat_rel=stat_rel, type_id=type_id,
                      site_ctx=site_ctx, key_mask=key_mask, bias=bias)
        z, enc_attn = self.encode(seq_x[:, :W], seq_av[:, :W], need_attn=True, **enc_kw)

        B = z.shape[0]
        preds, z_traj, z_true_traj, lat_steps = [], [], [], []
        alpha = self.alpha_from_attn(enc_attn, type_id)

        # 哪些步需要真值编码。这是 H=48 下的主要开销 —— 每步一次编码器前向，
        # 实测占单步耗时的 68%（1248ms / 1828ms）。
        #   - 重锚（p>0）：每步都要，因为 m 是逐样本的，B=64 时几乎必有样本命中
        #   - 隐一致性：只是正则项，抽样若干 h 即可（设计文档 §4.5.2 对 mono 同此处理）
        # t3 阶段 p=0，于是只剩 lat_k 次编码，H=48 的单步从 1828ms 降到约 700ms。
        if reanchor_p > 0.0:
            need_at = set(range(H))
        elif need_z_true:
            k = min(lat_k if lat_k > 0 else H, H)
            need_at = set(torch.randperm(H)[:k].tolist())
        else:
            need_at = set()

        for h in range(H):
            i = W + h                                  # seq 中当前帧的下标
            ad, ad_av = self.strip_obs(seq_x[:, i], seq_av[:, i], type_id)
            z, tr_attn = self.transition(z, ad, ad_av, type_id=type_id, bias=bias,
                                         key_mask=key_mask, need_attn=True)

            # alpha 每步从转移网络的空间注意力重取 —— QK^T 依赖 z，z 在变。
            # 冻结 alpha 会让反事实推演（改塔启停）失效。
            if tr_attn is not None:
                alpha = self.alpha_from_attn(tr_attn, type_id)

            # 先用转移出的状态解码 —— 转移网络必须在梯度路径上
            preds.append(self.decoder(self.split_by_family(z, type_id), alpha,
                                      self.physical_inputs(seq_raw[:, i], type_id),
                                      desc_ch=desc[type_id == S.FAMILIES.index("chiller")],
                                      scales=scales))
            z_traj.append(z)

            if h in need_at:
                # 以 t0+h+1 结尾的真值窗口 = seq[h+1 : h+1+W]
                with torch.no_grad():
                    z_true, _ = self.encode(seq_x[:, h + 1:i + 1],
                                            seq_av[:, h + 1:i + 1], **enc_kw)
                z_true = z_true.detach()               # stop-gradient
                if need_z_true:
                    z_true_traj.append(z_true)
                    lat_steps.append(h)
                # 重锚只改「往下一步传的状态」，不改刚才解码用的状态
                if reanchor_p > 0.0:
                    m = (torch.rand(B, 1, 1, device=z.device) < reanchor_p).to(z.dtype)
                    z = (1 - m) * z + m * z_true

        out = {"preds": preds, "z": torch.stack(z_traj, 1), "alpha": alpha, "bias": bias}
        if need_z_true and z_true_traj:
            out["z_true"] = torch.stack(z_true_traj, 1)
            # z 只取被抽到的那些步，与 z_true 逐一对齐
            out["z_lat"] = torch.stack([z_traj[h] for h in lat_steps], 1)
        return out

    @torch.no_grad()
    def imagine(self, batch, *, desc, stat_rel, type_id, site_ctx, H: int, W: int, **kw):
        """a 取历史回放 -> 用于测有效推演步数 h*。"""
        return self.rollout(batch, desc=desc, stat_rel=stat_rel, type_id=type_id,
                            site_ctx=site_ctx, H=H, W=W, reanchor_p=0.0, **kw)

    @torch.no_grad()
    def counterfactual(self, batch, intervene, *, desc, stat_rel, type_id,
                       site_ctx, H: int, W: int, **kw):
        """固定 z 与 d，换 a -> 干预响应。

        intervene(seq_x, seq_raw, W) 就地返回修改后的两个张量。只应改推演段
        (下标 >= W) 的动作维，历史窗必须原样保留 —— 否则 z_{t0} 变了，
        比较的就不是「同一 z 下换动作」。
        """
        b = dict(batch)
        b["seq_x"], b["seq_raw"] = intervene(batch["seq_x"].clone(),
                                             batch["seq_raw"].clone(), W)
        return self.rollout(b, desc=desc, stat_rel=stat_rel, type_id=type_id,
                            site_ctx=site_ctx, H=H, W=W, reanchor_p=0.0, **kw)

    def n_params(self) -> dict[str, int]:
        def cnt(m):
            return sum(p.numel() for p in m.parameters())
        return {
            "encoder": cnt(self.encoder),
            "  embed": cnt(self.encoder.embed),
            "  attn_bias": cnt(self.encoder.bias_mlp),
            "  blocks": cnt(self.encoder.blocks),
            "transition": cnt(self.transition),
            "decoder": cnt(self.decoder),
            "total": cnt(self),
        }
