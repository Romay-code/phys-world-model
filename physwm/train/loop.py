"""训练循环（设计文档 §4.5.3）。

    Adam lr=2e-3, CosineAnnealing -> eta_min=0.05*lr, wd=1e-4, grad clip 5.0
    选模：验证集 **H 步纯想象（p=0）** 的复合损失 —— 不是 teacher-forcing 口径。
          这是参考项目 v2/v3 的实测修复项之一，换成 TF 选模会选出「靠真值撑着」的权重。

必须分 seed 看，不能只看均值。参考项目 B2 v3 分 seed MAE 为
640.8/205.8/167.6/367.3/251.3，均值 327 完全由 seed 0 拉起来，去掉后是 248。
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from ..data import schema as S
from ..data.dataset import GPUWindows, covered_rows
from ..eval.metrics import (all_metrics, effective_horizon, effective_horizon_multi,
                            error_amplification,
                            per_step_metrics)
from ..model.decoder import check_hard_constraints
from .curriculum import Curriculum
from .losses import LossWeights, ObsLoss, TargetScales, compute_loss


def setup_backend(tf32: bool = True, seed: int | None = None) -> None:
    """打开 Ampere 的 TF32 张量核。

    PyTorch 默认对 matmul 关闭 TF32，于是 A6000 只跑到 fp32 的 38 TFLOPS，
    而 TF32 有约 150 TFLOPS。本模型是纯 matmul 密集型（注意力投影 + FFN），
    这一项是单步耗时的主要来源。TF32 的尾数只有 10 位，但这里的量级都在 O(1)
    附近（输入已归一化、物理量另有 scale 承载），精度足够。

    需要逐位复现时把 tf32 关掉。
    """
    if tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)


@dataclass
class TrainConfig:
    epochs: int = 300
    batch_size: int = 0        # 0 = 按 H 自动选，见 batch_for()
    lr: float = 2e-3
    eta_min_frac: float = 0.05
    weight_decay: float = 1e-4
    grad_clip: float = 5.0
    patience: int = 60
    eval_every: int = 5
    eval_H: int = 0            # 0 = 用课程当前的 H
    num_workers: int = 0
    seed: int = 0
    log_every: int = 10
    max_eval_batches: int = 40
    hard_check_every: int = 25
    # 每个 epoch 的优化步数。0 = 走完整个训练集。
    # 窗口是 stride-1 采的，相邻窗口 98% 重叠，走完整集是大量冗余；
    # H=48 时一个完整 epoch 要 7.7 分钟，限步是主要提速手段。
    steps_per_epoch: int = 0
    # 选模评估用的固定 H。0 = 用课程末期的 H（即 H_max）。
    #
    # **绝不能用课程的当前 H**：H=1 时 val 约 0.05、H=24 时约 0.27、H=48 时约 0.15，
    # 不同 epoch 的 val 根本不可比，取最小值必然选中 t1 阶段（H=1）的权重，
    # 整个 t2/t3 的多步训练全部作废。实测过一次：200 epoch 的课程，
    # 「最优 val @ epoch 55」落在 t1 里，最终 h*=1、误差放大 +4715%，
    # 与纯单步训练无异 —— 而且不报任何错。
    # 设计文档 §4.5.3 的「验证集 H 步纯想象复合损失」中的 H 指的是**目标 H**。
    select_H: int = 0
    # 按 H 调整 batch。H 大时前向是 H 步串行的 Python 循环，每步张量只有
    # [B,43,192]，完全由 kernel 启动开销主导，GPU 大量空转 —— 实测 H=48 上
    # batch 64->128 吞吐提升 1.50x；而 H=1 上 GPU 已喂饱，64->256 只提升 1.21x。
    # 所以是「H 越大 batch 越要大」，不是统一调大。
    # 两档都卡在约 14.5 GiB：不是 48G 显存不够，是别人的 VLLM 常驻 26.5G 且
    # 用量浮动，分配器在 14-15 GiB 就失败。
    batch_small_H: int = 256      # H <= h_switch 时用
    batch_large_H: int = 128      # H >  h_switch 时用
    h_switch: int = 8

    def batch_for(self, H: int) -> int:
        if self.batch_size:
            return self.batch_size
        return self.batch_small_H if H <= self.h_switch else self.batch_large_H


@dataclass
class Context:
    """一次训练需要的所有站点侧常量（张量已在目标设备上）。"""

    desc: torch.Tensor
    stat_rel: torch.Tensor
    type_id: torch.Tensor
    W: int
    site_ctx_dim: int
    key_mask: torch.Tensor | None = None
    # 少样本适配学到的站点调制向量。None = 全零 = 预训练时的口径。
    # FiLM 末层零初始化，故 ctx_vec 为 0 时 gamma=1/beta=0 是恒等映射 ——
    # **「零样本」与「ctx_vec=0 的适配模型」逐位相同**，零样本守门因此有
    # 一个干净的基线（physwm/train/fewshot.py）。
    ctx_vec: torch.Tensor | None = None

    def ctx(self, B: int, device) -> torch.Tensor:
        if self.ctx_vec is None:
            return torch.zeros(B, self.site_ctx_dim, device=device)
        return self.ctx_vec.to(device).view(1, -1).expand(B, -1)


def _to(batch: dict, device) -> dict:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


@torch.no_grad()
def evaluate(model, source, ctx: Context, obs_loss: ObsLoss, wts: LossWeights,
             H: int, device, max_batches: int = 0, collect: bool = False,
             batch_size: int = 64, scales: dict | None = None):
    """纯想象（p=0）口径的验证。collect=True 时额外回收逐步预测供 h* 计算。

    `scales` 为该站的量纲。**多站模型必须传** —— 它的解码器 buffer 从未被
    `set_scales` 写过（量纲是站点属性，见 decoder._s），不传会用默认值评测，
    而 14 站的 w_scale 差 6.9 倍、k_mcp 差 38.5 倍，结果会系统性偏移且不报错。
    """
    model.eval()
    tot, n = 0.0, 0
    parts: dict[str, float] = {}
    yt, yp = [], []
    hard = None
    res_phys: dict[str, float] = {}
    source.set_H(H)
    for i, b in enumerate(source.epoch(batch_size, shuffle=False, drop_last=False)):
        if max_batches and i >= max_batches:
            break
        out = model.rollout(b, desc=ctx.desc, stat_rel=ctx.stat_rel,
                            type_id=ctx.type_id, site_ctx=ctx.ctx(b["P_plant"].shape[0], device),
                            H=H, W=ctx.W, reanchor_p=0.0,
                            need_z_true=wts.lam_lat > 0, key_mask=ctx.key_mask,
                            lat_k=wts.lat_k, scales=scales)
        loss, log = compute_loss(model, out, b, obs_loss, wts, ctx.W)
        tot += float(loss)
        n += 1
        for k, v in log.items():
            parts[k] = parts.get(k, 0.0) + v
        if collect:
            yt.append(b["P_plant"][:, :H].cpu().numpy())
            yp.append(torch.stack([p["P_plant"] for p in out["preds"]], 1).cpu().numpy())
        if hard is None:
            on_ch = model.physical_inputs(b["seq_raw"][:, ctx.W], ctx.type_id)["on_ch"]
            p0 = out["preds"][0]
            hard = check_hard_constraints(p0, on=on_ch)
            # 物理量回读：硬约束全 0 不代表物理合理 —— 松 eta 区间那一版六项全 0
            # 但 COP 中位只有 0.549。必须把实际物理量也报出来才能判断。
            m = on_ch > 0.5
            if bool(m.any()):
                phys = {k: float(p0[k][m].median()) for k in
                        ("cop_actual", "eta", "cop_carnot", "q_evap", "w_chiller",
                         "dt_evap", "implied_mcp")
                        if k in p0}
                phys["cool_dt"] = float(p0["cool_dt"][m].median())
                # 塌缩检测：head 死掉时输出变常数，指标却只表现为「不再改善」，
                # 很容易被误读成收敛。实测过一次 head_w 塌缩 -> k3 恒为目标方差。
                for k in ("w_chiller", "eta", "approx"):
                    v = p0[k]
                    v = v[m] if v.shape == m.shape else v
                    if v.numel() > 1:
                        rsd = float(v.std() / v.abs().mean().clamp_min(1e-9))
                        phys[f"rsd_{k}"] = rsd
                res_phys = phys
    model.train()
    res = {"loss": tot / max(n, 1), **{k: v / max(n, 1) for k, v in parts.items()}}
    res["hard"] = hard or {}
    res["phys"] = res_phys
    if collect and yt:
        res["y_true"] = np.concatenate(yt, 0)
        res["y_pred"] = np.concatenate(yp, 0)
    return res


def train(model, sd, splits, spec, norm, ctx: Context, cfg: TrainConfig,
          cur: Curriculum, wts: LossWeights, device,
          out_dir: str | Path | None = None) -> dict:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    tr_w, va_w = splits["train"], splits["val"]
    train_rows = covered_rows(tr_w, spec, sd.T)
    scales = TargetScales.fit(sd, train_rows)
    obs_loss = ObsLoss(sd, scales, wts, device)

    # 方向损失要把原始量纲的扰动换算到归一化通路上（Normalizer 是逐 (token,field)
    # 的线性变换）。只在启用时才建，避免无谓占显存。
    dir_norm_scale = (torch.as_tensor(norm.scale, dtype=torch.float32, device=device)
                      if wts.lam_dir > 0 else None)

    ds_tr = GPUWindows(sd, tr_w, spec, norm, device)
    ds_va = GPUWindows(sd, va_w, spec, norm, device)

    # 一维参数（LayerNorm 权重、各种 bias）不做 weight decay。
    no_decay, decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 else decay).append(p)
    opt = torch.optim.Adam(
        [{"params": decay, "weight_decay": cfg.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}], lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg.epochs, eta_min=cfg.lr * cfg.eta_min_frac)

    H_sel = cfg.select_H or cur.H(cfg.epochs - 1)
    print(f"  选模口径：固定 H={H_sel} 纯想象（p=0），与课程当前 H 无关", flush=True)

    best = {"loss": math.inf, "epoch": -1}
    best_state = None
    bad = 0
    hist: list[dict] = []
    t_start = time.time()

    for epoch in range(cfg.epochs):
        H = cur.H(epoch)
        p = cur.reanchor_p(epoch)
        ds_tr.set_H(H)

        bs = cfg.batch_for(H)
        run, nb = 0.0, 0
        dir_run: dict[str, float] = {}
        for bi, b in enumerate(ds_tr.epoch(bs, shuffle=True, drop_last=True)):
            if cfg.steps_per_epoch and bi >= cfg.steps_per_epoch:
                break
            opt.zero_grad(set_to_none=True)
            out = model.rollout(
                b, desc=ctx.desc, stat_rel=ctx.stat_rel, type_id=ctx.type_id,
                site_ctx=ctx.ctx(b["P_plant"].shape[0], device), H=H, W=ctx.W,
                reanchor_p=p, need_z_true=wts.lam_lat > 0, key_mask=ctx.key_mask,
                lat_k=wts.lat_k)
            loss, log = compute_loss(model, out, b, obs_loss, wts, ctx.W)
            loss.backward()
            # 方向损失（P4）。**必须在主图 backward 之后单独反向。**
            #
            # 梯度是累加的，拆开与合并在数学上完全等价，但峰值显存从
            # 「主图 + 方向图」降为「两者取大」。合并那版实测在 A6000 上 OOM：
            # H=48/batch=128 的主图已占 21.5 GB，而卡上常驻的 VLLM 涨到 25.7 GB，
            # 只剩约 22 GB —— 再挂 3 个额外的编码器前向就爆了。
            # （方向损失虽只推 1 步，但每次仍要跑完整的 W=16 历史窗编码器，
            #   那才是开销大头，不是 H。）
            if wts.lam_dir > 0:
                from .direction_loss import direction_penalty
                del out
                # h_sample 不能超过当前课程的 H
                hs = tuple(h for h in wts.dir_h if h <= H) or (1,)
                dp, dlog = direction_penalty(model, b, ctx, ctx.W,
                                             sd.sch, dir_norm_scale, sub=0.25,
                                             h_sample=hs, margin=wts.dir_margin)
                (wts.lam_dir * dp).backward()
                log["dir"] = float(dp.detach())
                log.update(dlog)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            if wts.lam_dir > 0:
                for k, v in dlog.items():
                    dir_run[k] = dir_run.get(k, 0.0) + v
            run += float(loss.detach())
            nb += 1
        sched.step()
        tr_loss = run / max(nb, 1)

        if epoch % cfg.eval_every == 0 or epoch == cfg.epochs - 1:
            # 选模一律在固定的 H_sel 上评，与课程当前的 H 无关
            ev = evaluate(model, ds_va, ctx, obs_loss, wts, H_sel, device,
                          max_batches=cfg.max_eval_batches,
                          batch_size=cfg.batch_for(H_sel))
            rec = {"epoch": epoch, "stage": cur.stage(epoch), "H": H, "H_sel": H_sel, "p": p,
                   "lr": sched.get_last_lr()[0], "train": tr_loss, "val": ev["loss"],
                   **{f"val_{k}": v for k, v in ev.items()
                      if k not in ("loss", "hard", "y_true", "y_pred")}}
            # 方向项的逐目标违例率必须进 history。
            # P4-C/D 四个臂跑完才发现它没被记录 —— 于是「损失到底有没有把
            # 自己的指标压下去」全程不可见，只能靠事后加载 checkpoint 诊断。
            # 加一个损失项就必须同时能看到它自己的指标，否则等于盲跑。
            if dir_run and nb:
                rec.update({f"tr_{k}": v / nb for k, v in dir_run.items()})
            hist.append(rec)

            if ev["loss"] < best["loss"] - 1e-6:
                best = {"loss": ev["loss"], "epoch": epoch, "H": H}
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += cfg.eval_every

            if epoch % cfg.log_every == 0 or epoch == cfg.epochs - 1:
                hv = ev["hard"]
                bad_hard = {k: v for k, v in hv.items() if v > 1e-5}
                # 阈值 1e-4 而非 1e-3：初始化时各 head 权重很小，输出天然接近
                # 常数（eta 在 ep0 的 rsd 约 7e-4），1e-3 会在第一轮误报。
                # 真实塌缩时 rsd 会掉到 1e-6 量级（输出恒为常数）。
                dead = [k[4:] for k, v in ev.get("phys", {}).items()
                        if k.startswith("rsd_") and v < 1e-4]
                print(f"  ep{epoch:4d} {cur.stage(epoch)} H={H:2d} B={bs} p={p:.2f} "
                      f"lr={sched.get_last_lr()[0]:.2e} | train {tr_loss:.4f} "
                      f"val@H{H_sel} {ev['loss']:.4f} | k1 {ev.get('k1',0):.4f} "
                      f"k3 {ev.get('k3',0):.4f} lat {ev.get('lat',0):.4f} "
                      f"|z| {ev.get('z_scale',0):.3f}"
                      + (f" | 硬约束违例 {bad_hard}" if bad_hard else "")
                      + (f" | **head 塌缩 {dead}**" if dead else ""), flush=True)

            if bad >= cfg.patience:
                print(f"  early stop @ epoch {epoch}（{cfg.patience} epoch 无改善）")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    res = {"best": best, "history": hist, "seed": cfg.seed,
           "minutes": (time.time() - t_start) / 60.0,
           "scales": {"p_plant": scales.p_plant,
                      "p_fam": scales.p_fam.tolist(),
                      "w_chiller": scales.w_chiller}}

    if out_dir:
        d = Path(out_dir)
        d.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), d / f"model_seed{cfg.seed}.pt")
        (d / f"history_seed{cfg.seed}.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    return res


@torch.no_grad()
def final_report(model, sd, splits, spec, norm, ctx: Context, cfg: TrainConfig,
                 wts: LossWeights, device, H_eval: int, split: str = "test",
                 max_batches: int = 0, dir_vr: bool = False,
                 dir_max_batches: int = 12, scales: dict | None = None) -> dict:
    """出 §5.2 要求的标准输出：逐 h 指标、h*、误差放大、硬约束。

    **`max_batches` 必须用 0（不限）来报正式指标。**
    窗口由 `enumerate_windows` 按时间升序产出、`split_windows` 保序，
    而 `evaluate` 用 `shuffle=False` —— 于是截断 `max_batches` 取到的是
    **最早的那一批窗口**，不是随机样本。实测截断到 1920 个窗口时，
    14 站的 test 覆盖率只有 41.5%~67.9%，且系统性地**只覆盖前半段日历**，
    把盛夏整段排除在外。而模型在盛夏恰恰最差（hx 的 8-9 月：真值均值 5106、
    预测偏低 736，`corr` 仍有 0.882 但 R² 为负）。
    截断评测因此是**系统性偏乐观**，不是采样噪声。
    """
    # 注意与形参 `scales`（解码器的物理量纲）区分：这里是损失用的目标尺度，
    # 两者完全不同。第一版重名把形参遮住了，评测直接崩在 decoder._s 里。
    tsc = TargetScales.fit(sd, covered_rows(splits["train"], spec, sd.T))
    obs_loss = ObsLoss(sd, tsc, wts, device)
    ds = GPUWindows(sd, splits[split], spec, norm, device)

    eb = cfg.batch_for(H_eval)
    ev = evaluate(model, ds, ctx, obs_loss, wts, H_eval, device,
                  max_batches=max_batches, collect=True, batch_size=eb, scales=scales)
    yt, yp = ev["y_true"], ev["y_pred"]
    per_h = per_step_metrics(yt, yp)
    hs = effective_horizon(per_h)
    # G9 未拍板，三口径一起报（§13 #27）。绝对档取实测全站功率中位的 10%，
    # 物理含义是「推演误差不超过全站功率的 10%」，与模型自身表现无关。
    p_med = float(np.median(np.abs(yt[yt > 0]))) if (yt > 0).any() else float("nan")
    mae_abs = 0.10 * p_med if np.isfinite(p_med) else None
    hs_multi = effective_horizon_multi(per_h, mae_abs_max=mae_abs)

    # 误差放大：TF（每步都从真值起跳）vs FR（纯想象）
    tf_yt, tf_yp = [], []
    ds.set_H(H_eval)
    for i, b in enumerate(ds.epoch(eb, shuffle=False, drop_last=False)):
        # 与 `evaluate` 保持同一语义：0 = 不限。原先写成 `i >= max_batches`，
        # max_batches=0 时会**立刻 break**，误差放大栏直接变空。
        if max_batches and i >= max_batches:
            break
        o = model.rollout(b, desc=ctx.desc, stat_rel=ctx.stat_rel, type_id=ctx.type_id,
                          site_ctx=ctx.ctx(b["P_plant"].shape[0], device),
                          H=H_eval, W=ctx.W, reanchor_p=1.0, key_mask=ctx.key_mask,
                          scales=scales)
        tf_yt.append(b["P_plant"][:, :H_eval].cpu().numpy())
        tf_yp.append(torch.stack([q["P_plant"] for q in o["preds"]], 1).cpu().numpy())
    tf_yt, tf_yp = np.concatenate(tf_yt), np.concatenate(tf_yp)

    out = {
        "split": split, "H": H_eval, "n_samples": int(yt.shape[0]),
        "per_h": per_h,
        "overall": all_metrics(yt.ravel(), yp.ravel()),
        "step1": per_h[0],
        "h_star": hs,
        "h_star_multi": hs_multi,
        "p_plant_median": p_med,
        "amp": error_amplification(all_metrics(tf_yt.ravel(), tf_yp.ravel())["MAE"],
                                   all_metrics(yt.ravel(), yp.ravel())["MAE"]),
        "hard": ev["hard"],
        "phys": ev.get("phys", {}),
    }
    # 方向一致性（P4 交付件，§5.3）。每个动作维要多跑一遍推演，
    # 故默认关闭、且批数单独限流 —— 4 个动作维等于 5 倍评测开销。
    # eta 水平是否被数据定住（§13 #24/#29 的通用化检查）
    try:
        from ..eval.identifiability import eta_level_profile
        out["eta_ident"] = eta_level_profile(model, ds, ctx, obs_loss, wts,
                                             H_eval, device, batch_size=eb,
                                             scales=scales)
    except Exception as e:                      # 诊断失败不该拖垮整轮评测
        out["eta_ident"] = {"error": repr(e)}

    if dir_vr:
        from ..eval.direction import rollout_direction_check
        out["dir"] = rollout_direction_check(
            model, ds, ctx, H=H_eval, device=device, sch=sd.sch,
            norm_scale=norm.scale, batch_size=eb, max_batches=dir_max_batches,
            scales=scales)
    return out
