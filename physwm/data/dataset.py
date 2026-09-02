"""窗口采样、时间划分、归一化。

防泄漏（G7）的实现要点：
    一个样本占用的行区间是 [t0-W+1, t0+H]，共 W+H 步。
    划分时要求**整个区间落在同一个 split 内**（full containment），
    这本身就保证了任意训练样本与验证样本的行集合不相交 —— 不需要额外隔离带。
    在此之上再叠加 embargo 步的空档，用来削弱边界附近的自相关（默认 64 = W+H_max）。
    `assert_no_leak()` 会显式验证这一点。
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import json
import numpy as np
import pandas as pd
import torch

from . import schema as S


@dataclass(frozen=True)
class WindowSpec:
    W: int = 16          # 历史窗长
    H: int = 48          # 最大推演步数
    embargo: int = 64    # 额外空档，见模块 docstring
    ratios: tuple[float, float, float] = (0.70, 0.10, 0.20)
    # "blocked"       : 按块交错划分，每个 split 覆盖全部季节 —— 开发期主口径
    # "chronological" : 按时间顺序前后切 —— 跨季节外推评测，单独报
    split_mode: str = "blocked"
    # 块长（步）。480 步 = 5 天。实测的取舍（yb3, 13 个月）：
    #   1440 -> 窗口留存 86.5%，但 val 只覆盖 2 个月
    #    480 -> 留存 59.9%，val 覆盖 8 个月，load 中位 tr/va/te = 1467/1465/1622
    #    320 -> 留存 40.4%，三个 split 都覆盖全年，但丢掉六成数据
    # 取 480：分布对齐已经足够，再缩块换来的季节覆盖抵不过数据损失。
    block_len: int = 480

    @property
    def span(self) -> int:
        return self.W + self.H


@dataclass
class SiteData:
    """一个站点的全部张量。行索引与原 CSV 一致。"""

    site: str
    sch: S.SiteSchema
    x: np.ndarray          # [T, N, F]
    avail: np.ndarray      # [T, N, F]
    extra: dict            # 站级派生量
    segments: list[tuple[int, int]]

    @property
    def T(self) -> int:
        return self.x.shape[0]

    @property
    def N(self) -> int:
        return self.x.shape[1]

    @property
    def F(self) -> int:
        return self.x.shape[2]


def load_site(csv_path: str | Path, site: str | None = None,
              spec: WindowSpec = WindowSpec()) -> SiteData:
    p = Path(csv_path)
    site = site or p.stem
    df = pd.read_csv(p)
    sch = S.parse_site(df, site)
    x, avail, extra = S.build_arrays(df, sch)
    segs = S.split_segments(df, sch, min_len=spec.span)
    return SiteData(site=site, sch=sch, x=x, avail=avail, extra=extra, segments=segs)


def enumerate_windows(sd: SiteData, spec: WindowSpec) -> np.ndarray:
    """所有合法的 t0。返回 [M, 3] = (seg_idx, t0, span_start)。

    约束：[t0-W+1, t0+H] 必须完整落在同一连续段内。
    """
    rows = []
    for si, (s, e) in enumerate(sd.segments):
        lo = s + spec.W - 1          # 最早的 t0
        hi = e - spec.H - 1          # 最晚的 t0（含）
        for t0 in range(lo, hi + 1):
            rows.append((si, t0, t0 - spec.W + 1))
    return np.asarray(rows, dtype=np.int64).reshape(-1, 3)


def split_windows(sd: SiteData, windows: np.ndarray, spec: WindowSpec
                  ) -> dict[str, np.ndarray]:
    """把窗口分到 train/val/test。两种模式见 WindowSpec.split_mode。

    两者都要求**整个 [t0-W+1, t0+H] 区间落在同一 split 内**（full containment），
    这本身就保证任意两个 split 的行集合不相交；embargo 是额外的自相关缓冲。

    为什么默认 blocked：yb3 只有 13 个月数据，时序切法会把秋冬全给 train、
    盛夏全给 val —— 实测 val 的 load 最小值（1950）比 train 的 p95（1928）还高，
    整个验证集落在训练分布之外。对暖通站季节是第一主导因素，那样切出来的指标
    只在测外推，无法用于选模。跨季节外推是**单独一项评测**，用 chronological 报。
    """
    T = sd.T
    start = windows[:, 2]                    # 区间左端
    end = windows[:, 1] + spec.H             # 区间右端（含）
    g = spec.embargo

    if spec.split_mode == "chronological":
        r_tr, r_va, _ = spec.ratios
        b1, b2 = int(T * r_tr), int(T * (r_tr + r_va))
        out = {
            "train": windows[end < b1 - g],
            "val": windows[(start >= b1) & (end < b2 - g)],
            "test": windows[start >= b2],
        }
        out["_bounds"] = np.array([b1, b2], dtype=np.int64)
        return out

    if spec.split_mode != "blocked":
        raise ValueError(f"未知 split_mode: {spec.split_mode}")

    # 10 块一个循环，7 train / 1 val / 2 test，恰好是 70/10/20
    pattern = np.array([0] * 7 + [1] + [2] * 2, dtype=np.int64)
    L = spec.block_len
    blk_s, blk_e = start // L, end // L
    same_block = blk_s == blk_e                      # 整段必须在同一块内
    # 距块边界 >= embargo，削弱边界处的自相关
    off_s, off_e = start % L, end % L
    inner = (off_s >= g) & (off_e <= L - 1 - g)
    role = pattern[blk_s % len(pattern)]

    ok = same_block & inner
    out = {name: windows[ok & (role == i)] for i, name in
           enumerate(("train", "val", "test"))}
    out["_bounds"] = np.array([L, len(pattern)], dtype=np.int64)
    out["_block_role"] = role
    return out


def covered_rows(windows: np.ndarray, spec: WindowSpec, T: int) -> np.ndarray:
    """这批窗口实际用到哪些行。返回 [T] 的布尔掩码。

    **所有从数据算的统计量都必须用这个掩码，不能用 `slice(lo, hi)`** ——
    分块划分下 min..max 区间会横跨 val/test 块，那样归一化参数、设备描述符、
    目标 scale 全都会泄漏测试集信息。这类泄漏不体现在样本边界上，很隐蔽。
    """
    m = np.zeros(T, dtype=bool)
    if len(windows) == 0:
        return m
    for _, t0, hs in windows:
        m[hs:t0 + spec.H + 1] = True
    return m


def assert_no_leak(splits: dict[str, np.ndarray], spec: WindowSpec, T: int) -> dict:
    """显式验证：任意两个 split 用到的行集合不相交。

    直接做布尔覆盖再取交集，对时序和分块两种模式都成立。
    """
    names = ["train", "val", "test"]
    cov = {n: covered_rows(splits[n], spec, T) for n in names}
    stats = {}
    for i, a in enumerate(names):
        stats[f"rows_{a}"] = int(cov[a].sum())
        for b in names[i + 1:]:
            n_inter = int((cov[a] & cov[b]).sum())
            assert n_inter == 0, f"泄漏: {a} 与 {b} 有 {n_inter} 行重叠"
            stats[f"overlap_{a}_{b}"] = 0
    return stats


# --------------------------------------------------------------------------
# 归一化：只用训练集、只在 avail=1 的位置上统计，参数冻结存盘
# --------------------------------------------------------------------------

@dataclass
class Normalizer:
    center: np.ndarray   # [N, F]
    scale: np.ndarray    # [N, F]

    def apply(self, x: np.ndarray) -> np.ndarray:
        return (x - self.center) / self.scale

    def save(self, path: str | Path) -> None:
        np.savez(path, center=self.center, scale=self.scale)

    @staticmethod
    def load(path: str | Path) -> "Normalizer":
        z = np.load(path)
        return Normalizer(center=z["center"], scale=z["scale"])


def fit_normalizer(sd: SiteData, train_windows: np.ndarray,
                   spec: WindowSpec) -> Normalizer:
    """robust 标准化：中位数 + IQR。只看训练窗口**实际覆盖**的行。"""
    if len(train_windows) == 0:
        raise ValueError("训练集为空，无法拟合归一化")
    m = covered_rows(train_windows, spec, sd.T)
    x = sd.x[m]                           # [t, N, F]
    a = sd.avail[m].astype(bool)

    center = np.zeros((sd.N, sd.F), dtype=np.float32)
    scale = np.ones((sd.N, sd.F), dtype=np.float32)
    for n in range(sd.N):
        for k in range(sd.F):
            v = x[:, n, k][a[:, n, k]]
            if v.size < 32:
                continue
            q25, q50, q75 = np.percentile(v, [25, 50, 75])
            iqr = float(q75 - q25)
            center[n, k] = float(q50)
            # IQR 退化（常量/二值列）时退回 std，再退回 1.0
            if iqr > 1e-6:
                scale[n, k] = iqr / 1.349
            else:
                sd_ = float(v.std())
                scale[n, k] = sd_ if sd_ > 1e-6 else 1.0
    return Normalizer(center=center, scale=scale)


# --------------------------------------------------------------------------
# torch Dataset
# --------------------------------------------------------------------------

class RolloutWindows(torch.utils.data.Dataset):
    """一个样本 = 连续的 W+H 步：历史窗 [t0-W+1, t0] + 推演段 [t0+1, t0+H]。

    返回**整段** `seq_*`（长度 W+H）而不是切好的 hist/fut，因为两处需要任意
    时刻的历史窗：
        - 隐状态重锚定要 E(真实窗口_{t0+h})，即 seq[h+1 : h+1+W]
        - 隐一致性损失 lambda_lat 同上
    切成 hist/fut 就取不到这些窗口了。调用方用 `hist_view` / `fut_view` 取切片。

    返回 dict:
        seq_x     [W+H, N, F]  归一化值
        seq_avail [W+H, N, F]
        seq_raw   [W+H, N, F]  原始量纲，物理解码器与指标用
        P_plant   [H]          主口径总功率 (kW)
        P_fam     [H, 4]       四类分项功率 (kW)，列序同 S.DEVICE_FAMILIES
        P_ok      [H]
        steady    [H]
        t0        标量
    """

    def __init__(self, sd: SiteData, windows: np.ndarray, spec: WindowSpec,
                 norm: Normalizer, H: int | None = None):
        self.sd = sd
        self.w = windows
        self.spec = spec
        self.H = spec.H if H is None else H
        self.norm = norm
        self.xn = norm.apply(sd.x).astype(np.float32)

        k_steady = S.CANON_FIELDS["plant"].index("steady")
        self.steady = sd.x[:, 0, k_steady].astype(np.float32)
        self.P_fam = np.stack([sd.extra[f"power_{f}"] for f in S.DEVICE_FAMILIES],
                              axis=-1).astype(np.float32)

    def __len__(self) -> int:
        return len(self.w)

    def set_H(self, H: int) -> None:
        """课程学习中动态改变推演步数。"""
        assert 1 <= H <= self.spec.H
        self.H = H

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        _, t0, hs = self.w[i]
        W, H = self.spec.W, self.H
        se, fs, fe = hs + W + H, t0 + 1, t0 + 1 + H
        e = self.sd.extra
        return {
            "seq_x": torch.from_numpy(self.xn[hs:se]),
            "seq_avail": torch.from_numpy(self.sd.avail[hs:se]),
            "seq_raw": torch.from_numpy(self.sd.x[hs:se]),
            "P_plant": torch.from_numpy(e["P_plant"][fs:fe]),
            "P_fam": torch.from_numpy(self.P_fam[fs:fe]),
            "P_ok": torch.from_numpy(e["P_plant_ok"][fs:fe]),
            "steady": torch.from_numpy(self.steady[fs:fe]),
            "t0": torch.tensor(t0, dtype=torch.long),
        }


class GPUWindows:
    """整站数据常驻显存，按下标算术取 batch。

    替代 torch DataLoader：H=1 时每个 epoch 有 400+ 个 batch，每个 batch 要搬
    ~19MB，`__getitem__` 逐样本 numpy 切片 + from_numpy 会把单核 CPU 打满，
    GPU 反而在等。整站三个数组一共 175MB（yb3: 37616 x 43 x 9 x 4B x 3），
    直接放显存，取 batch 变成一次 gather。

    与 RolloutWindows 返回完全相同的键，两者可互换。
    """

    def __init__(self, sd: SiteData, windows: np.ndarray, spec: WindowSpec,
                 norm: Normalizer, device, H: int | None = None):
        self.spec = spec
        self.device = device
        self.H = spec.H if H is None else H

        t = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt, device=device)  # noqa: E731
        self.xn = t(norm.apply(sd.x))
        self.avail = t(sd.avail)
        self.raw = t(sd.x)
        e = sd.extra
        self.P = t(e["P_plant"])
        self.P_ok = t(e["P_plant_ok"])
        self.P_fam = t(np.stack([e[f"power_{f}"] for f in S.DEVICE_FAMILIES], -1))
        k_steady = S.CANON_FIELDS["plant"].index("steady")
        self.steady = t(sd.x[:, 0, k_steady])

        self.hs = t(windows[:, 2], torch.long)     # 区间左端
        self.t0 = t(windows[:, 1], torch.long)
        self.n = len(windows)

    def __len__(self) -> int:
        return self.n

    def set_H(self, H: int) -> None:
        assert 1 <= H <= self.spec.H
        self.H = H

    def batch(self, idx: torch.Tensor) -> dict[str, torch.Tensor]:
        W, H = self.spec.W, self.H
        hs, t0 = self.hs[idx], self.t0[idx]
        seq_rows = hs[:, None] + torch.arange(W + H, device=self.device)[None, :]
        fut_rows = t0[:, None] + 1 + torch.arange(H, device=self.device)[None, :]
        return {
            "seq_x": self.xn[seq_rows],
            "seq_avail": self.avail[seq_rows],
            "seq_raw": self.raw[seq_rows],
            "P_plant": self.P[fut_rows],
            "P_fam": self.P_fam[fut_rows],
            "P_ok": self.P_ok[fut_rows],
            "steady": self.steady[fut_rows],
            "t0": t0,
        }

    def epoch(self, batch_size: int, shuffle: bool = True, drop_last: bool = True):
        order = (torch.randperm(self.n, device=self.device) if shuffle
                 else torch.arange(self.n, device=self.device))
        end = (self.n // batch_size) * batch_size if drop_last else self.n
        for i in range(0, end, batch_size):
            yield self.batch(order[i:i + batch_size])


def hist_view(batch: dict[str, torch.Tensor], W: int, h: int = 0
              ) -> tuple[torch.Tensor, torch.Tensor]:
    """取以 t0+h 结尾的历史窗（归一化值与 avail）。h=0 即编码器的输入窗。"""
    return batch["seq_x"][:, h:h + W], batch["seq_avail"][:, h:h + W]


def fut_view(batch: dict[str, torch.Tensor], W: int, h: int
             ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """取推演段第 h 步（h 从 0 起）的一帧：归一化值、avail、原始量纲。"""
    i = W + h
    return (batch["seq_x"][:, i], batch["seq_avail"][:, i], batch["seq_raw"][:, i])


def build_site_bundle(csv_path: str | Path, site: str | None = None,
                      spec: WindowSpec = WindowSpec(),
                      out_dir: str | Path | None = None,
                      hold_rows: np.ndarray | None = None,
                      season_months: tuple[int, ...] | None = None) -> dict:
    """一站式：加载 -> 切段 -> 采窗 -> 划分 -> 验证无泄漏 -> 拟合归一化。

    `hold_rows`（[T] bool，True = 时段留出期）用于设计文档 §6 P5 的**轴 3**。
    给了它之后：

        test        = 整段落在留出期内的窗口
        train/val   = 留出期外的窗口，再按 blocked 花色切 train/val

    **归一化、描述符、量纲必须在切完之后才拟合** —— 它们都只看
    `splits["train"]` 覆盖的行。若沿用留出前的那一份，统计量里就含有
    留出期的数据，季节外推的结论直接作废，而且不会有任何报错
    （与 §13 #8 同型：那次是分块划分下用 `slice(lo,hi)` 取统计量）。
    本函数把三者一并重算，调用方无须关心。
    """
    sd = load_site(csv_path, site, spec)
    if season_months:
        # 按**月份序数**留出，跨全部年份。绝对时间窗留不掉 bh 的夏天
        # （它的日历与其余 13 站完全不重叠），见 temporal_holdout 模块文档。
        from .temporal_holdout import month_mask
        col = pd.read_csv(csv_path, usecols=[sd.sch.time_col])[sd.sch.time_col]
        hold_rows = month_mask(col, season_months)
    windows = enumerate_windows(sd, spec)
    if hold_rows is None:
        splits = split_windows(sd, windows, spec)
    else:
        from .temporal_holdout import split_by_period
        if len(hold_rows) != sd.T:
            raise ValueError(f"hold_rows 长度 {len(hold_rows)} 与站点行数 {sd.T} 不符")
        parts = split_by_period(windows, np.asarray(hold_rows, dtype=bool), spec)
        inner = split_windows(sd, parts["rest"], spec)
        # inner 的 test 块与 val 之间已有 blocked 的 purge/embargo，
        # 这里把它并进 train —— 时段留出下 test 另有来源，丢掉它是白扔 20% 数据
        splits = {
            "train": np.concatenate([inner["train"], inner["test"]])
            if len(inner["test"]) else inner["train"],
            "val": inner["val"],
            "test": parts["held"],
            "_bounds": inner["_bounds"],
            "_period_dropped": parts["dropped"],
        }
    leak = assert_no_leak(splits, spec, sd.T)
    norm = fit_normalizer(sd, splits["train"], spec)
    train_rows = covered_rows(splits["train"], spec, sd.T)

    info = {
        "site": sd.site,
        "N": sd.N, "F": sd.F, "T": sd.T,
        "n_dev": sd.sch.n_dev,
        "spec": asdict(spec),
        "n_segments": len(sd.segments),
        "seg_lens": [e - s for s, e in sd.segments],
        "n_windows": {k: int(len(v)) for k, v in splits.items() if not k.startswith("_")},
        "leak_check": leak,
        "bounds": splits["_bounds"].tolist(),
        "power_src": {f: sd.extra[f"power_{f}_src"] for f in S.DEVICE_FAMILIES},
        "per_device_label_n": {f: int(sd.extra[f"per_device_mask_{f}"].sum())
                               for f in S.DEVICE_FAMILIES},
        "phantom_n": {f: int(sd.extra[f"power_{f}_n_phantom"])
                      for f in S.DEVICE_FAMILIES},
    }

    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        norm.save(out / f"norm_{sd.site}.npz")
        (out / f"bundle_{sd.site}.json").write_text(
            json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    return {"data": sd, "windows": windows, "splits": splits, "norm": norm,
            "spec": spec, "info": info, "train_rows": train_rows}
