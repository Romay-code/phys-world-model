"""各站的日历覆盖：起止、跨月、季节分布。

决定「留出时段」怎么设计的第一手事实。设计文档 §6 P5 要求「站点留出 + 时段留出
双重留出」，但时段留出到底可行不可行，取决于各站在日历上是否对齐：

  · 各站覆盖同一段月份  -> 可以做**全局**时段留出（例如所有站一起留出盛夏），
                          留出的是「季节」这个共同维度，跨站可比
  · 各站覆盖互不相同    -> 「留出 7 月」在各站含义不同，标度曲线不可比，
                          只能退回逐站按比例切

另：§13 #7 已定死**站内**不可按时序切（yb3 只有 13 个月，时序切等于
秋冬训练盛夏验证，test R² −74.9），现用分块交错。这意味着当前
**没有任何一处在考验季节外推** —— 本表是判断该缺口有多大的依据。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from physwm.data import schema as S  # noqa: E402

DUP = {"yb3_topology_clean_v2", "yb3_topology_complete", "yb3test", "zxtest"}


def main() -> int:
    rows = []
    for f in sorted((ROOT / "data").glob("*.csv")):
        if f.stem in DUP:
            continue
        df = pd.read_csv(f, low_memory=False)
        sch = S.parse_site(df, f.stem)
        ts = pd.to_datetime(df[sch.time_col], errors="coerce").dropna()
        if ts.empty:
            rows.append((f.stem, None, None, 0, set(), 0.0))
            continue
        months = set(ts.dt.to_period("M").astype(str))
        # 盛夏占比：6-9 月
        summer = float(ts.dt.month.isin([6, 7, 8, 9]).mean())
        rows.append((f.stem, ts.min(), ts.max(), len(months), months, summer))

    print(f"{'站点':<18}{'起':<12}{'止':<12}{'跨月':>5}{'盛夏占比':>9}   月份")
    print("-" * 108)
    allm = set()
    for name, lo, hi, nm, months, summer in rows:
        allm |= months
        ms = ",".join(sorted(m[2:] for m in months))       # 去掉世纪，省宽度
        print(f"{name:<18}{str(lo)[:10]:<12}{str(hi)[:10]:<12}{nm:>5}"
              f"{summer:>9.2f}   {ms[:52]}")

    print()
    common = set.intersection(*[m for *_, m, _ in rows if m]) if rows else set()
    print(f"全部站点共同覆盖的月份：{sorted(common) if common else '（空）'}")
    print(f"并集共 {len(allm)} 个月：{sorted(allm)}")

    # 每个月被多少个站覆盖 —— 决定「全局留出某月」还剩多少训练站
    print()
    print("逐月被覆盖的站点数（全局时段留出的可行性）：")
    for m in sorted(allm):
        n = sum(1 for *_, months, _ in rows if m in months)
        bar = "#" * n
        print(f"  {m}  {n:>2} {bar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
