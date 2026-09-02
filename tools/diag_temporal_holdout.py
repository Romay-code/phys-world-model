"""轴 3 时段留出的可行性预演：留出后还剩多少可训数据？

在花 2.2 h 机时之前先回答三件事：

  1. 留出七八月后，各站还剩多少训练窗口（掉太多就没法比）
  2. 留出集本身够不够大（太小的话指标方差没法看）
  3. **bh 的夏天有没有被一并留出** —— 它的日历与其余 13 站完全不重叠
     （bh 2023-12~2024-09，其余 2024-10 起），按绝对时间窗留是留不掉它的

    python tools/diag_temporal_holdout.py --months 7,8
    python tools/diag_temporal_holdout.py --start 2025-11-01
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from physwm.data import schema as S  # noqa: E402
from physwm.data.dataset import WindowSpec, enumerate_windows, load_site  # noqa: E402
from physwm.data.registry import DUP_FILES, plant_of  # noqa: E402
from physwm.data.temporal_holdout import (month_mask, split_by_period,  # noqa: E402
                                          window_mask)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", default="7,8", help="按月份序数留出，如 7,8")
    ap.add_argument("--start", default="", help="改用绝对时间窗留出的起点")
    ap.add_argument("--end", default="")
    a = ap.parse_args()
    spec = WindowSpec()
    use_window = bool(a.start or a.end)
    months = tuple(int(x) for x in a.months.split(",") if x.strip())

    print("留出方式：" + (f"绝对时间窗 [{a.start or '-inf'}, {a.end or '+inf'})"
                          if use_window else f"月份序数 {months}（跨全部年份）"))
    if use_window:
        print("⚠ 绝对时间窗**不能用于季节外推**：bh 的日历与其余 13 站不重叠，")
        print("  留 2025 年的夏天留不掉 bh 的 2024 年夏天，模型照样见过盛夏。")
    print()
    print(f"{'站点':<18}{'总窗口':>8}{'留出窗口':>9}{'可训窗口':>9}{'丢弃':>8}"
          f"{'可训占比':>9}   留出月份")
    print("-" * 96)

    tot_h = tot_r = 0
    rows = []
    for f in sorted((ROOT / "data").glob("*.csv")):
        if f.stem in DUP_FILES:
            continue
        sd = load_site(f, f.stem, spec)
        df = pd.read_csv(f, low_memory=False)
        ts = pd.to_datetime(df[sd.sch.time_col], errors="coerce")
        hold = (window_mask(ts, a.start or None, a.end or None) if use_window
                else month_mask(ts, months))
        w = enumerate_windows(sd, spec)
        p = split_by_period(w, hold, spec)
        frac = len(p["rest"]) / max(len(w), 1)
        mons = sorted(set(ts[hold].dt.to_period("M").astype(str))) if hold.any() else []
        ms = ",".join(m[2:] for m in mons)
        print(f"{f.stem:<18}{len(w):>8}{len(p['held']):>9}{len(p['rest']):>9}"
              f"{p['dropped']:>8}{frac:>8.1%}   {ms[:34]}")
        tot_h += len(p["held"]); tot_r += len(p["rest"])
        rows.append((f.stem, len(p["held"]), len(p["rest"])))

    print("-" * 96)
    print(f"合计：留出 {tot_h:,} 窗口 / 可训 {tot_r:,} 窗口")
    print(f"独立冷站 {len({plant_of(n) for n, _, _ in rows})} 个")

    no_hold = [n for n, h, _ in rows if h == 0]
    no_train = [n for n, _, r in rows if r == 0]
    print()
    if no_hold:
        print(f"⚠ 这些站**没有任何留出窗口**（该时段没数据），"
              f"它们不参与轴 3 的评测：{no_hold}")
    if no_train:
        print(f"⚠ 这些站**没有任何可训窗口**，会被挤出训练集：{no_train}")
    if not use_window:
        bh = next((r for r in rows if r[0] == "bh"), None)
        if bh and bh[1] > 0:
            print(f"✓ bh 的夏天已被一并留出（{bh[1]:,} 个窗口）—— "
                  f"这是按月份序数留出而非绝对时间窗的关键收益")
        elif bh:
            print("⚠ bh 没有留出窗口，季节外推的结论会被它的夏天污染")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
