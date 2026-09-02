"""多站联合预训练（设计文档 §6 P5）。

一个模型 + N 套站点侧状态。**站点侧状态包含五样东西，缺一不可**：

    desc / stat_rel / type_id   设备描述符与拓扑统计（逐站，形状随台数变）
    scales                      量纲（逐站，跨 14 站差 6.9~38.5 倍）
    norm                        归一化中心与尺度（逐站逐 token 逐字段）
    windows                     切好的连续段窗口
    obs_loss                    观测损失（含该站的逐设备标签掩码）

**一个 batch 只能来自一个站点。** 不同站的 token 数不同（实测 21~50）、
desc 不同、量纲不同，无法在一个张量里拼起来。故「多站」体现在**逐 batch 换站**，
不是在 batch 内混合。

站点采样默认**按训练窗口数成比例**：14 站窗口数从 10919 到 17653，差异不大
（1.6 倍），比例采样与均匀采样接近，但对将来引入小站更稳。
可用 `sample="uniform"` 切成等权 —— 若目标是「每个站都好」而非
「总体样本上好」，等权更合适。这个选择会影响标度曲线的解释，**须在报告中写明**。

选模用**跨站未加权平均**的验证损失：基础模型关心的是逐站质量，
不是样本量加权后的总体质量，否则大站会主导选模。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from ..data.dataset import GPUWindows, WindowSpec, build_site_bundle, covered_rows
from ..data.descriptors import DescNormalizer, DescriptorBundle, compute_descriptors
from ..data.registry import (DUP_FILES, capability_tier, expand_holdout,
                             holdout_kind, plant_of)
from ..data.scales import infer_site_scales
from .avail_dropout import AvailDropout
from .loop import Context
from .losses import LossWeights, ObsLoss, TargetScales

@dataclass
class Site:
    """一个站点的全部训练侧状态。"""

    name: str
    sd: object
    splits: dict
    spec: WindowSpec
    norm: object
    ctx: Context
    scales: dict
    obs_loss: ObsLoss
    ds_tr: GPUWindows
    ds_va: GPUWindows
    n_train: int = 0
    held_out: bool = False
    plant: str = ""          # 所属冷站（中温/低温两回路同属一个）
    tier: str = ""           # 能力档 A/B/C/D，见 data.registry

    def __repr__(self) -> str:
        return (f"<Site {self.name} N={self.sd.N} 窗口={self.n_train}"
                f" 冷站={self.plant} 档={self.tier}"
                f"{' 留出' if self.held_out else ''}>")


def build_sites(data_dir: str | Path, device, *, spec: WindowSpec | None = None,
                wts: LossWeights | None = None, hold_out: tuple[str, ...] = (),
                only: tuple[str, ...] = (), d_ctx: int = 8,
                artifacts: str | Path | None = None,
                season_months: tuple[int, ...] | None = None) -> list[Site]:
    """装配全部站点。`hold_out` 里的站点仍会被装配（供零样本评测），但不参与训练。

    `season_months`（如 `(7, 8)`）打开设计文档 §6 P5 的**轴 3 时段留出**：
    每个站的 test 换成该月份段，train/val 只从其余月份取，且归一化/描述符/
    量纲全部在切完之后重算（`build_site_bundle` 负责）。**这是站点留出之外的
    另一条正交轴**，两者可以同时开，但报数时必须分开说。

    描述符归一化 `DescNormalizer` **必须在所有参与训练的站点上联合拟合**：
    它的作用就是把各站的描述符放到同一尺度上，逐站各拟合一份等于没做。
    留出站点不参与拟合（否则是泄漏），但用训练站拟合出的参数来变换 ——
    这正是零样本要考验的：新站的描述符能否被已有的归一化正确处理。
    """
    spec = spec or WindowSpec()
    wts = wts or LossWeights()
    data_dir = Path(data_dir)
    art = Path(artifacts) if artifacts else None

    raw = []
    for f in sorted(data_dir.glob("*.csv")):
        if f.stem in DUP_FILES:
            continue
        if only and f.stem not in only:
            continue
        b = build_site_bundle(f, f.stem, spec, out_dir=art,
                              season_months=season_months)
        sd, trm = b["data"], b["train_rows"]
        db = compute_descriptors(sd.x, sd.avail, sd.sch, trm)
        raw.append((f.stem, b, sd, trm, db))

    # 留出必须**冷站完整**：只留一个温区回路而把兄弟回路留在训练里，
    # 同楼同天气同排程，「零样本」会偏乐观且不报任何错（见 data.registry）
    hold_out = expand_holdout(hold_out, [n for n, *_ in raw])

    train_db = [db for name, _, _, _, db in raw if name not in hold_out]
    if not train_db:
        raise ValueError("没有任何参与训练的站点，无法拟合描述符归一化")
    dn = DescNormalizer.fit([d.desc for d in train_db])

    sites: list[Site] = []
    for name, b, sd, trm, db in raw:
        ctx = Context(
            desc=torch.from_numpy(dn.apply(db.desc)).float().to(device),
            stat_rel=torch.from_numpy(db.stat_rel).float().to(device),
            type_id=torch.from_numpy(db.type_id).to(device),
            W=spec.W, site_ctx_dim=d_ctx)
        sc = infer_site_scales(sd, trm)
        tsc = TargetScales.fit(sd, covered_rows(b["splits"]["train"], spec, sd.T))
        sites.append(Site(
            name=name, sd=sd, splits=b["splits"], spec=spec, norm=b["norm"],
            ctx=ctx, scales=sc, obs_loss=ObsLoss(sd, tsc, wts, device),
            ds_tr=GPUWindows(sd, b["splits"]["train"], spec, b["norm"], device),
            ds_va=GPUWindows(sd, b["splits"]["val"], spec, b["norm"], device),
            n_train=len(b["splits"]["train"]), held_out=name in hold_out,
            plant=plant_of(name), tier=capability_tier(b["info"]["power_src"])))
    return sites


def site_sampler(sites: list[Site], mode: str = "proportional", seed: int = 0):
    """产出训练站点的下标序列。留出站点永不出现。"""
    idx = [i for i, s in enumerate(sites) if not s.held_out]
    if not idx:
        raise ValueError("全部站点都被留出了")
    if mode == "uniform":
        p = np.ones(len(idx)) / len(idx)
    elif mode == "proportional":
        w = np.array([sites[i].n_train for i in idx], dtype=float)
        p = w / w.sum()
    else:
        raise ValueError(f"未知的站点采样方式: {mode}")
    rng = np.random.default_rng(seed)
    while True:
        yield int(rng.choice(idx, p=p))


def describe(sites: list[Site]) -> str:
    tr = [s for s in sites if not s.held_out]
    ho = [s for s in sites if s.held_out]
    train_tiers = {s.tier for s in tr}
    lines = [f"{'站点':<22}{'冷站':<10}{'档':>3}{'N':>4}{'训练窗口':>10}"
             f"{'验证':>8}  {'角色'}"]
    lines.append("-" * 82)
    for s in sites:
        role = (f"留出/{holdout_kind(s.tier, train_tiers)}" if s.held_out else "训练")
        lines.append(f"{s.name:<22}{s.plant:<10}{s.tier:>3}{s.sd.N:>4}"
                     f"{s.n_train:>10}{len(s.splits['val']):>8}  {role}")
    lines.append("-" * 82)
    lines.append(f"训练 {len(tr)} 个文件 / {len({s.plant for s in tr})} 个独立冷站；"
                 f"留出 {len(ho)} 个文件 / {len({s.plant for s in ho})} 个独立冷站")
    lines.append(f"训练窗口合计 {sum(s.n_train for s in tr):,}；"
                 f"训练集覆盖能力档 {sorted(train_tiers)}")
    for s in ho:
        if s.tier not in train_tiers:
            lines.append(f"⚠ {s.name} 属 {s.tier} 档，训练集里**一个成员都没有** —— "
                         f"这是跨档外推，不可与同档跨站混报")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 训练循环
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_sites(model, sites: list[Site], wts: LossWeights, H: int, device,
                   max_batches: int = 8, batch_size: int = 64,
                   which: str = "train") -> dict:
    """逐站验证，返回 {站名: loss} 与跨站未加权均值。

    `which="train"` 只评参与训练的站；`"held"` 只评留出站（零样本）；`"all"` 全评。
    """
    from .losses import compute_loss
    model.eval()
    out: dict[str, float] = {}
    for s in sites:
        if which == "train" and s.held_out:
            continue
        if which == "held" and not s.held_out:
            continue
        s.ds_va.set_H(H)
        tot, n = 0.0, 0
        for i, b in enumerate(s.ds_va.epoch(batch_size, shuffle=False, drop_last=False)):
            if i >= max_batches:
                break
            o = model.rollout(b, desc=s.ctx.desc, stat_rel=s.ctx.stat_rel,
                              type_id=s.ctx.type_id,
                              site_ctx=s.ctx.ctx(b["P_plant"].shape[0], device),
                              H=H, W=s.ctx.W, reanchor_p=0.0,
                              key_mask=s.ctx.key_mask, scales=s.scales)
            # 验证只看观测项：lat/dir/evap_bal 是训练期正则，混进来会让
            # 不同站因正则项量级不同而不可比
            w0 = LossWeights(**{**wts.__dict__, "lam_lat": 0.0, "lam_lip": 0.0,
                                "lam_dir": 0.0, "lam_evap_bal": 0.0,
                                "lam_eta_level": 0.0, "lam_soft_b": 0.0})
            l, _ = compute_loss(model, o, b, s.obs_loss, w0, s.ctx.W)
            tot += float(l); n += 1
        if n:
            out[s.name] = tot / n
    model.train()
    vals = list(out.values())
    return {"per_site": out, "mean": float(np.mean(vals)) if vals else float("nan")}


def train_multisite(model, sites: list[Site], cfg, cur, wts: LossWeights, device,
                    *, sample: str = "proportional", seed: int = 0,
                    out_dir: str | Path | None = None, log_every: int = 5,
                    avail_drop: AvailDropout | None = None) -> dict:
    """多站联合预训练。

    与单站 `loop.train` 的差别只有四处，其余（课程、选模口径、梯度裁剪）完全一致：
      1. 每个 batch 先抽站，再用**该站的** ctx / scales / obs_loss
      2. 选模用跨站未加权平均的验证损失
      3. 留出站点每次评测也算一遍，但**不进选模**，只作零样本曲线
      4. `avail_drop` 随机把整条输入通道掩成缺测（只训练期，评测绝不掩）

    第 4 条是 P5-B 的修复项，动机见 `avail_dropout` 模块文档：P5-A 里
    pc3 零样本崩到 R²=−1.64，根因是它缺的五条负荷类通道模型从没见过缺。
    """
    import time
    from .direction_loss import direction_penalty
    from .losses import compute_loss

    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    tr_sites = [s for s in sites if not s.held_out]
    # 优化器与调度器逐项对齐单站 `loop.train`，否则多站/单站结果不可比
    no_decay, decay = [], []
    for _, prm in model.named_parameters():
        if prm.requires_grad:
            (no_decay if prm.ndim <= 1 else decay).append(prm)
    opt = torch.optim.Adam(
        [{"params": decay, "weight_decay": cfg.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}], lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg.epochs, eta_min=cfg.lr * cfg.eta_min_frac)
    gen = site_sampler(sites, sample, seed)
    # 掩码用独立 Generator：与站点采样/参数初始化解耦，
    # 开关 avail_drop 不会连带改变抽站序列，两条臂才可比
    drop_gen = torch.Generator().manual_seed(seed + 10_000)
    H_sel = cfg.select_H or cur.H(cfg.epochs - 1)

    best = {"loss": float("inf"), "epoch": -1}
    hist: list[dict] = []
    t0 = time.time()
    for epoch in range(cfg.epochs):
        H, p = cur.H(epoch), cur.reanchor_p(epoch)
        bs = cfg.batch_for(H)
        run, nb, seen = 0.0, 0, {}
        drop_hits: dict[str, int] = {}
        n_drop_steps = 0
        for step in range(cfg.steps_per_epoch or 100):
            s = sites[next(gen)]
            s.ds_tr.set_H(H)
            idx = torch.randint(0, len(s.ds_tr), (bs,), device=device)
            b = s.ds_tr.batch(idx)
            if avail_drop is not None and avail_drop.enabled:
                b, hit = avail_drop.apply(b, s.sd.sch, gen=drop_gen)
                if hit:
                    n_drop_steps += 1
                    for fam, fld in hit:
                        k = f"{fam}.{fld}"
                        drop_hits[k] = drop_hits.get(k, 0) + 1
            opt.zero_grad(set_to_none=True)
            o = model.rollout(b, desc=s.ctx.desc, stat_rel=s.ctx.stat_rel,
                              type_id=s.ctx.type_id, site_ctx=s.ctx.ctx(bs, device),
                              H=H, W=s.ctx.W, reanchor_p=p,
                              need_z_true=wts.lam_lat > 0, key_mask=s.ctx.key_mask,
                              lat_k=wts.lat_k, scales=s.scales)
            loss, log = compute_loss(model, o, b, s.obs_loss, wts, s.ctx.W)
            loss.backward()
            if wts.lam_dir > 0:
                del o
                hs = tuple(h for h in wts.dir_h if h <= H) or (1,)
                dp, _ = direction_penalty(model, b, s.ctx, s.ctx.W, s.sd.sch,
                                          torch.as_tensor(s.norm.scale, dtype=torch.float32,
                                                          device=device),
                                          sub=0.25, h_sample=hs, margin=wts.dir_margin)
                (wts.lam_dir * dp).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            run += float(loss.detach()); nb += 1
            seen[s.name] = seen.get(s.name, 0) + 1
        sched.step()

        if epoch % log_every == 0 or epoch == cfg.epochs - 1:
            ev = evaluate_sites(model, sites, wts, H_sel, device, which="train")
            zs = evaluate_sites(model, sites, wts, H_sel, device, which="held")
            rec = {"epoch": epoch, "H": H, "p": p, "lr": sched.get_last_lr()[0],
                   "train": run / max(nb, 1), "val_mean": ev["mean"],
                   "val_per_site": ev["per_site"], "zeroshot_per_site": zs["per_site"],
                   "zeroshot_mean": zs["mean"], "site_counts": seen,
                   # §13 #44：加了机制却不记录它自己的指标 == 盲跑
                   "avail_drop_steps": n_drop_steps,
                   "avail_drop_hits": drop_hits}
            hist.append(rec)
            if ev["mean"] < best["loss"]:
                best = {"loss": ev["mean"], "epoch": epoch}
                if out_dir:
                    Path(out_dir).mkdir(parents=True, exist_ok=True)
                    torch.save(model.state_dict(), Path(out_dir) / f"model_seed{cfg.seed}.pt")
            zt = "  ".join(f"{k}={v:.4f}" for k, v in zs["per_site"].items())
            print(f"  ep {epoch:3d} H={H:2d} p={p:.2f} | train {rec['train']:.4f} "
                  f"val跨站均值 {ev['mean']:.4f} | 零样本 {zt}")
    return {"best": best, "history": hist, "minutes": (time.time() - t0) / 60,
            "sites": [s.name for s in tr_sites],
            "held_out": [s.name for s in sites if s.held_out]}
