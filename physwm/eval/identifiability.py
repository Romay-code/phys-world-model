"""可辨识性诊断：报出来的物理量，到底是数据定的还是先验定的。

**为什么必须有这一条。** 本项目已经两次被同一类问题咬：

  #24  `eta` 梯度恒为 0，报出的 0.387 是区间中点初始化值 ——
       硬约束零违例、物理量落在合理范围，两类检查都查不出来。
  #29  接回梯度后 `eta` 沿平坦方向滑到下界 0.100，`m·cp` 精确反补 ——
       梯度检查这次也过了（确实有梯度），但那个值仍然无信息。

梯度非零只说明「有通路」，不说明「被定住」。要判断后者，得直接问：
**把这个量的水平整体推开，监督损失变不变？** 不变就是没被定住。

这就是 profile-likelihood 的思路，代价是几次前向，不需要重训。
"""
from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def eta_level_profile(model, ds, ctx, obs_loss, wts, H: int, device,
                      shifts=(-1.0, -0.5, 0.0, 0.5, 1.0),
                      batch_size: int = 32, max_batches: int = 8,
                      scales: dict | None = None) -> dict:
    """把 eta 的 logit 整体平移若干量，看监督损失怎么变。

    返回 {"shifts": [...], "loss": [...], "eta": [...], "rel_rise": [...],
          "identified": bool, "curvature": float}

    判据：把水平推到 ±0.5 logit（eta 约 0.40 -> 0.29/0.52，覆盖离心机的
    整个合理区间）时，若监督损失相对上升 < 1%，即判为**未被数据定住**。
    """
    from ..train.losses import compute_loss

    dec = model.decoder
    old = float(dec.eta_logit_shift)
    model.eval()
    ds.set_H(H)
    rows = []
    try:
        for sh in shifts:
            dec.eta_logit_shift.fill_(float(sh))
            tot, n, eta_acc = 0.0, 0, []
            for i, b in enumerate(ds.epoch(batch_size, shuffle=False, drop_last=False)):
                if i >= max_batches:
                    break
                out = model.rollout(
                    b, desc=ctx.desc, stat_rel=ctx.stat_rel, type_id=ctx.type_id,
                    site_ctx=ctx.ctx(b["P_plant"].shape[0], device),
                    H=H, W=ctx.W, reanchor_p=0.0, key_mask=ctx.key_mask,
                    scales=scales)
                # 只看观测项：先验项本身当然随平移变化，那是循环论证
                w0 = type(wts)(**{**wts.__dict__, "lam_eta_level": 0.0,
                                  "lam_lat": 0.0, "lam_lip": 0.0})
                l, _ = compute_loss(model, out, b, obs_loss, w0, ctx.W)
                tot += float(l)
                n += 1
                eta_acc.append(float(out["preds"][0]["eta"].median()))
            rows.append((sh, tot / max(n, 1), float(np.median(eta_acc)) if eta_acc else float("nan")))
    finally:
        dec.eta_logit_shift.fill_(old)

    sh = [r[0] for r in rows]
    ls = [r[1] for r in rows]
    et = [r[2] for r in rows]
    i0 = sh.index(0.0) if 0.0 in sh else int(np.argmin(ls))
    base = ls[i0]
    rel = [(x - base) / max(abs(base), 1e-12) for x in ls]
    probe = [rel[i] for i, v in enumerate(sh) if abs(abs(v) - 0.5) < 1e-9]
    rise = max(probe) if probe else max(rel)
    return {"shifts": sh, "loss": ls, "eta": et, "rel_rise": rel,
            "rise_at_half": rise, "identified": bool(rise > 0.01)}


def format_identifiability(rep: dict) -> str:
    lines = [f"{'logit 平移':>10}{'eta 中位':>10}{'监督损失':>12}{'相对变化':>10}"]
    for sh, et, ls, rl in zip(rep["shifts"], rep["eta"], rep["loss"], rep["rel_rise"]):
        lines.append(f"{sh:>10.1f}{et:>10.3f}{ls:>12.5f}{rl:>+10.2%}")
    v = "**被数据定住**" if rep["identified"] else "**未被定住 —— 先验主导，绝对值不可作为物理证据**"
    lines.append(f"±0.5 处相对上升 {rep['rise_at_half']:.2%}  ->  {v}")
    return "\n".join(lines)
