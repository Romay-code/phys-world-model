"""时段留出（设计文档 §6 P5 轴 3）：换个季节 / 换到未来，模型还行吗。

## 为什么现有的划分回答不了这个问题

站内划分因 §13 #7 用的是 `blocked`（分块交错），每个 split 都覆盖全年 ——
这是**故意**的，为的是让选模指标不被季节主导。代价是：
**当前没有任何一处在考验季节外推**。而部署时「上线后撞进一个没见过的季节」
是必然发生的事。

`chronological` 模式（切最后 20%）只答了一半：它是「未来时段」，
但各站的最后 20% 落在不同月份，跨站不可比；而且**只在留出站上能用** ——
训练站的后段本来就在训练集里。

## 两种留出，回答两个不同的问题

    season   按**月份序数**留出（如 7、8 月），跨全部年份、全部站点
             -> 考「季节外推」：训练集里根本没有盛夏
    window   按**绝对时间窗**留出（如 2025-11-01 之后）
             -> 考「未来外推」：训练集里没有那段日历

**季节外推必须用 `season` 而不是 `window`。** 实测日历（`tools/diag_time_coverage.py`）：
bh 覆盖 2023-12~2024-09，其余 13 站从 2024-10 起，**两段完全不重叠**。
若按绝对窗留出「2025-07~08」，bh 的 2024-07~08 仍在训练集里，
模型照样见过盛夏 —— 这个留出就是假的。按月份序数留则自动把 bh 的夏天一并留出。

## 划分纪律

与 `split_windows` 同一条：**整段 `[t0-W+1, t0+H]` 必须完整落在一侧**
（full containment），这本身保证行集合不相交；`embargo` 是额外的自相关缓冲。
跨在边界上的窗口**两边都不要**，直接丢弃 —— 宁可少样本，不可污染。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .dataset import WindowSpec


def month_mask(timestamps: pd.Series, months) -> np.ndarray:
    """按**月份序数**取行掩码。`months` 如 (7, 8) 表示七、八月，跨所有年份。"""
    want = {int(m) for m in months}
    if not want <= set(range(1, 13)):
        raise ValueError(f"月份必须在 1..12，收到 {sorted(want)}")
    return pd.to_datetime(timestamps).dt.month.isin(want).to_numpy()


def window_mask(timestamps: pd.Series, start=None, end=None) -> np.ndarray:
    """按**绝对时间窗** `[start, end)` 取行掩码。两端可各自为 None 表示不限。"""
    if start is None and end is None:
        raise ValueError("start 与 end 不能都为空，那样留出的是整个数据集")
    ts = pd.to_datetime(timestamps)
    m = np.ones(len(ts), dtype=bool)
    if start is not None:
        m &= (ts >= pd.Timestamp(start)).to_numpy()
    if end is not None:
        m &= (ts < pd.Timestamp(end)).to_numpy()
    return m


def split_by_period(windows: np.ndarray, hold_rows: np.ndarray,
                    spec: WindowSpec) -> dict[str, np.ndarray]:
    """按行掩码把窗口切成「留出期内 / 留出期外」，跨界的丢弃。

    `hold_rows` 是 [T] 的 bool，True = 留出期。返回

        held    整段落在留出期内的窗口          -> 时段留出的评测集
        rest    整段落在留出期外、且距边界 >= embargo 的窗口
        dropped 跨界或落在缓冲区里的窗口数（只作统计）

    **不允许「部分重叠算作训练」**：一个窗口只要有一步踩进留出期，
    它的历史窗或推演段就见过留出期的数据，留出即失效。
    """
    if windows.size == 0:
        return {"held": windows, "rest": windows, "dropped": 0}
    T = len(hold_rows)
    lo = windows[:, 2]                       # 区间左端 t0-W+1
    hi = np.minimum(windows[:, 1] + spec.H, T - 1)   # 区间右端（含）

    # 前缀和 -> O(1) 判「区间内有几行是留出期」
    csum = np.concatenate([[0], np.cumsum(hold_rows.astype(np.int64))])
    n_hold = csum[hi + 1] - csum[lo]
    span = hi - lo + 1

    all_in = n_hold == span
    all_out = n_hold == 0

    # embargo：区间两侧再各扩 g 行，仍不得碰到留出期
    g = spec.embargo
    lo_g = np.maximum(lo - g, 0)
    hi_g = np.minimum(hi + g, T - 1)
    clean = (csum[hi_g + 1] - csum[lo_g]) == 0

    held = windows[all_in]
    rest = windows[all_out & clean]
    return {"held": held, "rest": rest,
            "dropped": int(len(windows) - len(held) - len(rest))}


def describe_holdout(name: str, timestamps: pd.Series, hold_rows: np.ndarray,
                     parts: dict) -> str:
    ts = pd.to_datetime(timestamps)
    mons = sorted(set(ts[hold_rows].dt.to_period("M").astype(str))) if hold_rows.any() else []
    return (f"{name:<20}留出行 {int(hold_rows.sum()):>6}/{len(hold_rows):<6} "
            f"留出窗口 {len(parts['held']):>6}  可训窗口 {len(parts['rest']):>6}  "
            f"丢弃 {parts['dropped']:>6}  留出月份 "
            + (",".join(m[2:] for m in mons)[:44] if mons else "（无）"))
