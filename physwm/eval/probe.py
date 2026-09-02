"""线性探针：z 里到底有没有物理量（设计文档 §5.2「表征通用性」、§6 P5 门限）。

门限是「探针可回读 ≥3 类物理量（R² ≥ 0.70）」。要让这个数有意义，
必须先处理一个**本项目已经踩过一次的坑**。

## 为什么必须带对照臂

§13 #36 的教训：`dir_vr` 有三项恒为 0.000，因为解码器里 `w = on·(fr/50)³·k`
是**结构恒等式** —— 那三项「通过」与模型学没学到物理毫无关系。

线性探针有**同型**的陷阱，而且更隐蔽：`plr` 本身就是一条**输入通道**，
`approach = tower_out − wet_bulb` 是两条输入通道的**线性组合**。
拿它们做探针目标，高 R² 只证明「编码器没把输入丢掉」，不证明学到了物理。

因此每个目标都同时报两个数：

    R²(z)     从编码器隐状态线性回读
    R²(raw)   从**同一个窗口的原始输入**按同样方式池化后线性回读（对照臂）

判据是 `gain = R²(z) − R²(raw)`：

  · R²(raw) 本来就高          -> 该目标**平凡**，不构成表征证据，报告里必须单列
  · R²(z) 显著高于 R²(raw)    -> 编码器确实算出了输入里没有的东西

每个目标的平凡度是**先验可判**的（见 `TARGETS` 的 `trivial` 字段），
但仍然实测 —— 先验说不平凡而对照臂打脸的情况，正是最该看到的。

## 目标

全部由**实测通道**算出，不碰任何模型输出 —— 否则是拿模型验自己。

    approach    塔逼近度 = tower_out − wet_bulb    与解码器 `tower_out = approx_eff + wb` 同定义
    plr         部分负荷率，开机冷机均值
    cop_carnot  T_ev/(T_cd − T_ev)，由实测两端温度算
    eps_tower   冷却塔换热效能 = (cool_out − cool_back)/(cool_out − wb)
    ntu         −ln(1 − eps_tower)
    dt_evap     蒸发侧温差 = cold_back − cold_out

## 口径

拟合用 train 窗口、报数用 val 窗口。两者之间已有 purge/embargo（§12 G7），
不额外处理。岭回归闭式解，`alpha` 由 train 内部再切一刀选，不看 val。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..data import schema as S

T0_K = 273.15


@dataclass(frozen=True)
class Target:
    name: str
    trivial: bool      # 先验判断：是否只是输入通道的平凡函数
    note: str


TARGETS: tuple[Target, ...] = (
    Target("approach", True, "= tower_out − wet_bulb，两条输入通道的线性组合"),
    Target("plr", True, "本身就是一条输入通道"),
    Target("cop_carnot", False, "两条温度通道的非线性比值"),
    Target("eps_tower", False, "三条温度通道的非线性比值"),
    Target("ntu", False, "eps 的对数变换"),
    Target("dt_evap", True, "两条输入通道的线性组合"),
)


def _ch_rows(sch, fam: str) -> list[int]:
    return [i for i, (f, _) in enumerate(sch.token_index) if f == fam]


def compute_targets(sd) -> dict[str, np.ndarray]:
    """逐行算出全部探针目标，[T]。不可算的位置为 NaN，由调用方按行丢弃。"""
    K, sch = S.CANON_FIELDS, sd.sch
    ch = _ch_rows(sch, "chiller")
    x, av = sd.x, sd.avail.astype(bool)

    def plant(fld):
        k = K["plant"].index(fld)
        v = x[:, 0, k].astype(np.float64).copy()
        v[~av[:, 0, k]] = np.nan
        return v

    def chill(fld):
        k = K["chiller"].index(fld)
        v = x[:, ch, k].astype(np.float64).copy()
        v[~av[:, ch, k]] = np.nan
        return v

    on = chill("on") > 0.5

    def on_mean(v):
        w = np.where(on & np.isfinite(v), v, np.nan)
        if w.shape[1] == 0:
            return np.full(len(w), np.nan)
        with np.errstate(invalid="ignore"):
            m = np.nanmean(np.where(np.isnan(w), np.nan, w), axis=1)
        return m

    wb = plant("wet_bulb")
    tower_out = plant("tower_out")
    cold_out = on_mean(chill("cold_out_temp"))
    cold_back = on_mean(chill("cold_back_temp"))
    cool_out = on_mean(chill("cool_out_temp"))
    cool_back = on_mean(chill("cool_back_temp"))

    with np.errstate(invalid="ignore", divide="ignore"):
        t_ev = cold_out + T0_K
        t_cd = np.maximum(cool_out + T0_K, t_ev + 2.0)
        cop_c = t_ev / (t_cd - t_ev)
        eps = (cool_out - cool_back) / (cool_out - wb)
        eps_c = np.clip(eps, 1e-3, 1 - 1e-3)
        ntu = -np.log(1.0 - eps_c)

    ok_eps = np.isfinite(eps) & (eps > 0) & (eps < 1)
    out = {
        "approach": tower_out - wb,
        "plr": on_mean(chill("plr")),
        "cop_carnot": np.where(np.isfinite(cop_c) & (cop_c > 0), cop_c, np.nan),
        "eps_tower": np.where(ok_eps, eps, np.nan),
        "ntu": np.where(ok_eps, ntu, np.nan),
        "dt_evap": cold_back - cold_out,
    }
    return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}


def pool_by_family(v: torch.Tensor, type_id: torch.Tensor) -> torch.Tensor:
    """[B,N,C] -> [B, 5*C]。逐族均值拼接，**与站点台数无关** —— 否则跨站不可比。"""
    parts = []
    for t in range(len(S.FAMILIES)):
        sel = type_id == t
        if bool(sel.any()):
            parts.append(v[:, sel].mean(1))
        else:
            parts.append(v.new_zeros(v.shape[0], v.shape[-1]))
    return torch.cat(parts, dim=-1)


@torch.no_grad()
def collect(model, site, device, split: str = "train", max_batches: int = 60,
            batch_size: int = 64):
    """返回 (Z, RAW, t0)：编码器隐状态、同窗末帧原始输入、锚点行号。

    两个特征都按 `pool_by_family` 池化，维度处理一致 —— 对照臂必须与主臂
    只差「特征来自哪里」这一件事，别的都不能差。
    """
    from ..data.dataset import GPUWindows
    if split == "train":
        ds = site.ds_tr
    elif split == "val":
        ds = site.ds_va
    else:
        ds = GPUWindows(site.sd, site.splits[split], site.spec, site.norm, device)
    ds.set_H(1)
    W = site.ctx.W
    Z, R, T = [], [], []
    for i, b in enumerate(ds.epoch(batch_size, shuffle=False, drop_last=False)):
        if i >= max_batches:
            break
        z, _ = model.encode(b["seq_x"][:, :W], b["seq_avail"][:, :W],
                            desc=site.ctx.desc, stat_rel=site.ctx.stat_rel,
                            type_id=site.ctx.type_id,
                            site_ctx=site.ctx.ctx(b["seq_x"].shape[0], device))
        Z.append(pool_by_family(z, site.ctx.type_id).cpu().numpy())
        # 对照臂：同一个窗口的**归一化输入末帧**，同样逐族池化
        R.append(pool_by_family(b["seq_x"][:, W - 1], site.ctx.type_id).cpu().numpy())
        T.append(b["t0"].cpu().numpy())
    return np.concatenate(Z), np.concatenate(R), np.concatenate(T)


def _ridge_fit(X, y, alpha):
    Xc, yc = X.mean(0), y.mean()
    Xs = X - Xc
    A = Xs.T @ Xs + alpha * np.eye(X.shape[1])
    w = np.linalg.solve(A, Xs.T @ (y - yc))
    return w, Xc, yc


def _r2(y, p):
    ss = float(((y - y.mean()) ** 2).sum())
    if ss <= 0:
        return float("nan")
    return 1.0 - float(((y - p) ** 2).sum()) / ss


def ridge_probe(Xtr, ytr, Xva, yva,
                alphas=(1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3, 1e4)) -> float:
    """岭回归线性探针，返回 val 上的 R²。

    `alpha` 由 train 内部再切 80/20 选，**绝不看 val** —— 否则探针的
    超参在报数的那批数据上被调过，报出来的 R² 是乐观的。
    """
    if len(Xtr) < 50 or len(Xva) < 20:
        return float("nan")
    cut = int(len(Xtr) * 0.8)          # 时间序，内部切一刀也按时间切，不打乱
    best, best_a = -np.inf, alphas[0]
    for a in alphas:
        w, Xc, yc = _ridge_fit(Xtr[:cut], ytr[:cut], a)
        r = _r2(ytr[cut:], (Xtr[cut:] - Xc) @ w + yc)
        if np.isfinite(r) and r > best:
            best, best_a = r, a
    w, Xc, yc = _ridge_fit(Xtr, ytr, best_a)
    return _r2(yva, (Xva - Xc) @ w + yc)


def is_readable(r2_raw: float) -> bool:
    """对照臂为负 -> 该 (站, 目标) 的探针数**不可读**。

    实测踩到：yb3 的 `eps_tower` 对照臂 R² 是 **−6.32**、tx 的 `cop_carnot`
    是 −3.89 —— 连原始输入都推不出来的目标，`R²(z)` 反映的是 train/val 划分
    而不是表征质量。把它算成「模型没学到」是错误归因，算成「过线」更糟。
    """
    return bool(np.isfinite(r2_raw) and r2_raw >= 0.0)


def summarize(rows: list[dict], gate: float = 0.70) -> dict:
    """门限统计：只数**非平凡、可读、且过线**的目标。"""
    def ok(r):
        return r["readable"] and np.isfinite(r["r2_z"]) and r["r2_z"] >= gate

    nt = [r for r in rows if not r["trivial"]]
    return {"n_pass_nontrivial": sum(1 for r in nt if ok(r)),
            "n_pass_all": sum(1 for r in rows if ok(r)),
            "n_nontrivial_unreadable": sum(1 for r in nt if not r["readable"]),
            "n_nontrivial": len(nt)}


def probe_site(model, site, device, *, max_batches: int = 60,
               gate: float = 0.70) -> dict:
    """一个站点的全部探针目标。返回逐目标的 R²(z) / R²(raw) / gain。"""
    Ztr, Rtr, ttr = collect(model, site, device, "train", max_batches)
    Zva, Rva, tva = collect(model, site, device, "val", max_batches)
    tgt = compute_targets(site.sd)

    rows = []
    for T in TARGETS:
        ytr_all, yva_all = tgt[T.name][ttr], tgt[T.name][tva]
        mtr, mva = np.isfinite(ytr_all), np.isfinite(yva_all)
        r2z = ridge_probe(Ztr[mtr], ytr_all[mtr], Zva[mva], yva_all[mva])
        r2r = ridge_probe(Rtr[mtr], ytr_all[mtr], Rva[mva], yva_all[mva])
        gain = (r2z - r2r) if (np.isfinite(r2z) and np.isfinite(r2r)) else float("nan")
        rows.append({"target": T.name, "trivial": T.trivial, "note": T.note,
                     "n_train": int(mtr.sum()), "n_val": int(mva.sum()),
                     "r2_z": r2z, "r2_raw": r2r, "gain": gain,
                     "readable": is_readable(r2r)})

    return {"site": site.name, "held_out": site.held_out, "rows": rows,
            **summarize(rows, gate)}
