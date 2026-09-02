"""设备描述符 desc[n] 与成对关系统计量 stat_rel[i,j]。

这两样东西取代了 B2 的 `nn.Embedding(chiller_id)` 与人工拓扑掩码：
    desc      -> token 的身份向量，跨站同义，从数据算出，无可学习查表
    stat_rel  -> attn_bias MLP 的成对输入，编码「谁和谁有物理联系」

硬性要求（设计文档 §4.2）：
    1. 只用**训练集时段**的行统计，否则测试集信息经描述符泄漏进训练
    2. 标准化参数一并冻结存盘，新站接入时复用，不重新拟合
    3. 全程不出现按站点/按设备索引的可学习参数

维度（合计 32）:
    [0]      容量份额
    [1:4]    频率工作区间 f_p5/p50/p95 / 50Hz
    [4:7]    运行节律：开机率、平均连续运行时长(log)、启停频次/日(log)
    [7:9]    功率-频率弹性：dlogW/dlogf 斜率、拟合 R^2
    [9:21]   联锁谱：对 4 个族各取 phi 的 max/mean/熵
    [21:30]  冷机专有：PLR p25/p50/p95、效率代理 p25/p50/p95（非真 COP，见下）、
             ΔT_evap p50、ΔT_cond p50、卡诺 COP 倒数 p50
    [30]     valid_interlock  联锁谱是否算得出
    [31]     valid_chiller    冷机专有段是否适用
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import schema as S

D_DESC = 32
D_REL = 5   # phi, 温度残差偏相关, 频率-响应互信息, 同族, log容量比

CP_WATER = 4.186 / 3600.0  # kJ/(kg*K) -> kWh，此处只用于量纲说明，未直接使用


# --------------------------------------------------------------------------
# 小工具
# --------------------------------------------------------------------------

def _safe_q(v: np.ndarray, qs: list[float], default: float = 0.0) -> np.ndarray:
    if v.size < 8:
        return np.full(len(qs), default, dtype=np.float64)
    return np.percentile(v, qs)


def _run_lengths(on: np.ndarray) -> tuple[float, float]:
    """平均连续运行时长（步）与启停次数。"""
    on = (on > 0.5).astype(np.int8)
    if on.size == 0 or on.sum() == 0:
        return 0.0, 0.0
    d = np.diff(on)
    starts = int((d == 1).sum()) + int(on[0] == 1)
    mean_run = float(on.sum() / max(starts, 1))
    return mean_run, float(starts)


def _loglog_slope(w: np.ndarray, f: np.ndarray) -> tuple[float, float]:
    """log W 对 log f 的 OLS 斜率与 R^2。泵/塔理论值约 3。"""
    m = np.isfinite(w) & np.isfinite(f) & (w > 1e-3) & (f > 1.0)
    if m.sum() < 32:
        return 0.0, 0.0
    lw, lf = np.log(w[m]), np.log(f[m])
    if lf.std() < 1e-6:
        return 0.0, 0.0
    A = np.vstack([lf, np.ones_like(lf)]).T
    coef, *_ = np.linalg.lstsq(A, lw, rcond=None)
    pred = A @ coef
    ss_res = float(((lw - pred) ** 2).sum())
    ss_tot = float(((lw - lw.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-9 else 0.0
    return float(coef[0]), float(np.clip(r2, 0.0, 1.0))


def _phi(a: np.ndarray, b: np.ndarray) -> float:
    """两个 0/1 序列的 phi 系数（等价于二值皮尔逊相关）。"""
    a = (a > 0.5).astype(np.float64)
    b = (b > 0.5).astype(np.float64)
    if a.size < 32:
        return 0.0
    sa, sb = a.std(), b.std()
    if sa < 1e-9 or sb < 1e-9:
        return 0.0
    return float(np.clip(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb), -1, 1))


def _entropy(v: np.ndarray) -> float:
    """归一化后的谱熵，度量「联锁是集中在少数设备还是弥散」。"""
    p = np.abs(v)
    s = p.sum()
    if s < 1e-9 or p.size < 2:
        return 0.0
    p = p / s
    p = p[p > 1e-12]
    return float(-(p * np.log(p)).sum() / np.log(len(v)))


def _mutual_info_binned(x: np.ndarray, y: np.ndarray, bins: int = 8) -> float:
    """分箱互信息，归一到 [0,1]。用于频率-响应关系。"""
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 128:
        return 0.0
    x, y = x[m], y[m]
    if x.std() < 1e-9 or y.std() < 1e-9:
        return 0.0
    hx = np.histogram2d(x, y, bins=bins)[0]
    p = hx / hx.sum()
    px, py = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
    nz = p > 0
    mi = float((p[nz] * np.log(p[nz] / (px @ py)[nz])).sum())
    hxe = -float((px[px > 0] * np.log(px[px > 0])).sum())
    hye = -float((py[py > 0] * np.log(py[py > 0])).sum())
    denom = min(hxe, hye)
    return float(np.clip(mi / denom, 0.0, 1.0)) if denom > 1e-9 else 0.0


def _partial_corr_resid(a: np.ndarray, b: np.ndarray,
                        drivers: np.ndarray) -> float:
    """扣掉公共驱动后的残差相关。

    公共驱动（湿球/负荷/站级塔温）会把所有温度拉成同向变化，
    直接算电平相关会全是 ~1.0，毫无分辨力（yb3 实测）。这里先差分再回归扣除。
    """
    m = np.isfinite(a) & np.isfinite(b) & np.isfinite(drivers).all(axis=1)
    if m.sum() < 256:
        return 0.0
    A, B, D = a[m], b[m], drivers[m]
    X = np.hstack([D, np.ones((len(D), 1))])
    try:
        ra = A - X @ np.linalg.lstsq(X, A, rcond=None)[0]
        rb = B - X @ np.linalg.lstsq(X, B, rcond=None)[0]
    except np.linalg.LinAlgError:
        return 0.0
    if ra.std() < 1e-9 or rb.std() < 1e-9:
        return 0.0
    return float(np.clip(np.corrcoef(ra, rb)[0, 1], -1, 1))


# --------------------------------------------------------------------------
# 主体
# --------------------------------------------------------------------------

@dataclass
class DescriptorBundle:
    desc: np.ndarray        # [N, 32]
    stat_rel: np.ndarray    # [N, N, 5]
    type_id: np.ndarray     # [N]
    names: list[str]

    def save(self, path: str | Path) -> None:
        np.savez(path, desc=self.desc, stat_rel=self.stat_rel,
                 type_id=self.type_id, names=np.array(self.names, dtype=object))

    @staticmethod
    def load(path: str | Path) -> "DescriptorBundle":
        z = np.load(path, allow_pickle=True)
        return DescriptorBundle(desc=z["desc"], stat_rel=z["stat_rel"],
                                type_id=z["type_id"], names=list(z["names"]))


def _field(x: np.ndarray, a: np.ndarray, sch: S.SiteSchema,
           n: int, fam: str, fld: str) -> tuple[np.ndarray, np.ndarray]:
    """取 token n 的某个字段序列及其有效掩码。"""
    if fld not in S.CANON_FIELDS[fam]:
        z = np.zeros(x.shape[0])
        return z, np.zeros(x.shape[0], dtype=bool)
    k = S.CANON_FIELDS[fam].index(fld)
    return x[:, n, k].astype(np.float64), a[:, n, k].astype(bool)


def compute_descriptors(x: np.ndarray, avail: np.ndarray, sch: S.SiteSchema,
                        row_mask: np.ndarray) -> DescriptorBundle:
    """只用 `row_mask` 选中的行（必须是训练窗口覆盖的行）计算描述符。

    用布尔掩码而非 slice：分块划分下训练行不连续，slice(min,max) 会把
    val/test 的行也统计进来，造成不体现在样本边界上的隐性泄漏。
    """
    row_mask = np.asarray(row_mask)
    if row_mask.dtype != bool:
        raise TypeError("row_mask 必须是布尔掩码，不能传 slice")
    xs, as_ = x[row_mask], avail[row_mask]
    T, N, _ = xs.shape
    tokens = sch.token_index
    desc = np.zeros((N, D_DESC), dtype=np.float64)
    names = [f"{f}_{d}" if f != "plant" else "plant" for f, d in tokens]

    # ---- 预取每个 token 的 on / freq / power ----
    on_seq, freq_seq, pw_seq = [], [], []
    for n, (fam, _) in enumerate(tokens):
        o, om = _field(xs, as_, sch, n, fam, "on")
        f, fm = _field(xs, as_, sch, n, fam, "frequency")
        w, wm = _field(xs, as_, sch, n, fam, "consumption")
        on_seq.append(np.where(om, o, 0.0))
        freq_seq.append(np.where(fm, f, np.nan))
        pw_seq.append(np.where(wm, w, np.nan))

    # ---- 族内容量份额分母 ----
    fam_cap: dict[str, float] = {}
    for fam in S.DEVICE_FAMILIES:
        caps = []
        for n, (f, _) in enumerate(tokens):
            if f == fam:
                w = pw_seq[n]
                caps.append(np.nanpercentile(w, 95) if np.isfinite(w).any() else 0.0)
        fam_cap[fam] = float(np.nansum(caps)) if caps else 0.0

    # ---- 公共驱动（用于偏相关）----
    k_wb = S.CANON_FIELDS["plant"].index("wet_bulb")
    k_ld = S.CANON_FIELDS["plant"].index("load")
    k_to = S.CANON_FIELDS["plant"].index("tower_out")
    drivers = np.stack([np.diff(xs[:, 0, k], prepend=xs[0, 0, k])
                        for k in (k_wb, k_ld, k_to)], axis=1)

    # ---- 逐 token 的一元描述量 ----
    for n, (fam, dev) in enumerate(tokens):
        if fam == "plant":
            desc[n, 31] = 0.0
            continue
        on, freq, pw = on_seq[n], freq_seq[n], pw_seq[n]
        run = on > 0.5

        # ① 容量份额
        cap = np.nanpercentile(pw, 95) if np.isfinite(pw).any() else 0.0
        desc[n, 0] = cap / fam_cap[fam] if fam_cap[fam] > 1e-6 else 0.0

        # ② 频率工作区间（仅开机时）
        fr = freq[run & np.isfinite(freq)]
        desc[n, 1:4] = _safe_q(fr, [5, 50, 95]) / 50.0

        # ③ 运行节律
        mean_run, n_start = _run_lengths(on)
        days = max(T * S.CTRL_PERIOD / 86400.0, 1e-6)
        desc[n, 4] = float(run.mean())
        desc[n, 5] = np.log1p(mean_run)
        desc[n, 6] = np.log1p(n_start / days)

        # ④ 功率-频率弹性
        if fam != "chiller":
            slope, r2 = _loglog_slope(pw[run], freq[run])
            desc[n, 7:9] = (np.clip(slope, -6, 6), r2)

        # ⑥ 冷机专有
        if fam == "chiller":
            plr, _ = _field(xs, as_, sch, n, fam, "plr")
            co, _ = _field(xs, as_, sch, n, fam, "cold_out_temp")
            cb, _ = _field(xs, as_, sch, n, fam, "cold_back_temp")
            kb, _ = _field(xs, as_, sch, n, fam, "cool_back_temp")
            ko, _ = _field(xs, as_, sch, n, fam, "cool_out_temp")
            m = run & np.isfinite(pw) & (pw > 1e-3)
            desc[n, 21:24] = _safe_q(plr[m], [25, 50, 95])
            # 效率代理，**不是真 COP**：无铭牌容量（G8）也无冷冻流量，拿不到 Q_evap，
            # 故用「负荷率 / 该机自身归一功率」这个无量纲相对量。
            # 它的绝对值无物理意义，作用只是在 desc 里区分设备间的效率差异。
            wn = pw[m] / max(np.nanpercentile(pw[m], 50), 1e-6) if m.sum() > 8 else np.array([])
            eff_proxy = (plr[m] / np.clip(wn, 1e-3, None)) if wn.size else np.array([])
            desc[n, 24:27] = _safe_q(eff_proxy, [25, 50, 95])
            desc[n, 27] = float(np.median((cb - co)[m])) if m.sum() > 8 else 0.0
            desc[n, 28] = float(np.median((ko - kb)[m])) if m.sum() > 8 else 0.0
            # 卡诺 COP 的倒数（= 1/COP_carnot = (T_cd-T_ev)/T_ev）。
            # 只用温度算，是纯热力学量，与效率代理无关，不受上面 proxy 的量纲问题影响。
            if m.sum() > 8:
                t_ev = co[m] + 273.15
                t_cd = np.maximum(ko[m] + 273.15, t_ev + 2.0)
                desc[n, 29] = float(np.median((t_cd - t_ev) / t_ev))
            desc[n, 31] = 1.0

    # ---- ⑤ 联锁谱 + stat_rel ----
    phi_mat = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(i + 1, N):
            v = _phi(on_seq[i], on_seq[j])
            phi_mat[i, j] = phi_mat[j, i] = v

    fam_of = [f for f, _ in tokens]
    for n in range(N):
        if fam_of[n] == "plant":
            continue
        seg = []
        for fam in S.DEVICE_FAMILIES:
            idx = [k for k in range(N) if fam_of[k] == fam and k != n]
            if not idx:
                seg.extend([0.0, 0.0, 0.0])
                continue
            v = phi_mat[n, idx]
            seg.extend([float(np.max(np.abs(v))), float(np.mean(v)), _entropy(v)])
        desc[n, 9:21] = seg
        desc[n, 30] = 1.0

    # 成对关系统计量
    stat_rel = np.zeros((N, N, D_REL), dtype=np.float64)
    resid_cache: dict[int, np.ndarray] = {}
    for n, (fam, _) in enumerate(tokens):
        fld = {"chiller": "cool_back_temp"}.get(fam)
        if fld:
            v, m = _field(xs, as_, sch, n, fam, fld)
            v = np.where(m, v, np.nan)
            resid_cache[n] = np.diff(v, prepend=v[0])

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            stat_rel[i, j, 0] = phi_mat[i, j]
            if i in resid_cache and j in resid_cache:
                stat_rel[i, j, 1] = _partial_corr_resid(
                    resid_cache[i], resid_cache[j], drivers)
            fi, fj = freq_seq[i], freq_seq[j]
            if np.isfinite(fi).any() and np.isfinite(fj).any():
                stat_rel[i, j, 2] = _mutual_info_binned(fi, fj)
            stat_rel[i, j, 3] = float(fam_of[i] == fam_of[j])
            # 容量份额比。任一方无功率标签（份额=0）时该维无意义，置 0 而非给极值 ——
            # yb3 的 tower 0..15 无逐台功率，若不处理会产生 |log| ~ 7.8 的野值主导 MLP 输入
            ci, cj = desc[i, 0], desc[j, 0]
            if ci > 1e-6 and cj > 1e-6:
                stat_rel[i, j, 4] = float(np.clip(np.log(ci / cj), -3.0, 3.0))

    desc = np.nan_to_num(desc, nan=0.0, posinf=0.0, neginf=0.0)
    stat_rel = np.nan_to_num(stat_rel, nan=0.0, posinf=0.0, neginf=0.0)
    return DescriptorBundle(desc=desc.astype(np.float32),
                            stat_rel=stat_rel.astype(np.float32),
                            type_id=sch.type_id, names=names)


@dataclass
class DescNormalizer:
    """描述符的跨站标准化。必须在多站预训练时用同一套参数。"""

    center: np.ndarray
    scale: np.ndarray

    @staticmethod
    def fit(descs: list[np.ndarray]) -> "DescNormalizer":
        allv = np.concatenate(descs, axis=0)
        q25, q50, q75 = np.percentile(allv, [25, 50, 75], axis=0)
        iqr = (q75 - q25) / 1.349
        scale = np.where(iqr > 1e-6, iqr, np.maximum(allv.std(0), 1e-6))
        return DescNormalizer(center=q50.astype(np.float32),
                              scale=scale.astype(np.float32))

    def apply(self, d: np.ndarray) -> np.ndarray:
        return np.clip((d - self.center) / self.scale, -5.0, 5.0).astype(np.float32)

    def save(self, p: str | Path) -> None:
        np.savez(p, center=self.center, scale=self.scale)

    @staticmethod
    def load(p: str | Path) -> "DescNormalizer":
        z = np.load(p)
        return DescNormalizer(center=z["center"], scale=z["scale"])
