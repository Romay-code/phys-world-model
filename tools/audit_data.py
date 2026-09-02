"""数据体检 — P-1 任务 3。

对每个站点量化：设备清单、时间连续性、逐列有效率、功率口径一致性、量程越界。
产出 reports/audit_<site>.md 与 reports/audit_summary.csv。

用法:
    python tools/audit_data.py                    # 全部站点
    python tools/audit_data.py yb3_topology_complete
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"

CTRL_PERIOD = 900  # 秒，控制周期

# 五类设备的列名模式。value = (正则, 该族的字段名列表)
FAMILY_PATTERNS = {
    "chiller": r"chiller_group/chiller/(\d+)/(.+)$",
    "tower": r"tower_group/tower/(\d+)/(.+)$",
    "coolpump": r"cooling_group/cooling_pump/(\d+)/(.+)$",
    "coldpump": r"colding_group/colding_pump/(\d+)/(.+)$",
}

# 聚合功率列 -> 对应族，用于 G2 口径核查
AGG_POWER = {
    "chiller": "chiller_consumption",
    "tower": "tower_consumption",
    "coolpump": "cool_pump_consumption",
    "coldpump": "cold_pump_consumption",
}

# 量程合理区间，超出即计为越界（不修改数据，只统计）
RANGE_RULES = {
    "frequency": (0.0, 60.0),
    "plr": (0.0, 1.5),
    "on": (0.0, 1.0),
    "cold_out_temp": (0.0, 30.0),
    "cold_back_temp": (0.0, 35.0),
    "cool_back_temp": (5.0, 50.0),
    "cool_out_temp": (5.0, 55.0),
    "tower_out_temp": (5.0, 50.0),
    "wet_bulb_temp": (-20.0, 40.0),
    "consumption": (0.0, 1e5),
}


@dataclass
class FamilyInfo:
    name: str
    ids: list[int] = field(default_factory=list)
    fields: list[str] = field(default_factory=list)


def parse_columns(cols: list[str]) -> tuple[dict[str, FamilyInfo], list[str]]:
    """把列名分成五类设备字段 + 站级列。"""
    fams = {k: FamilyInfo(k) for k in FAMILY_PATTERNS}
    matched: set[str] = set()
    for fam, pat in FAMILY_PATTERNS.items():
        ids, flds = set(), set()
        for c in cols:
            m = re.match(pat, c)
            if m:
                ids.add(int(m.group(1)))
                flds.add(m.group(2))
                matched.add(c)
        fams[fam].ids = sorted(ids)
        fams[fam].fields = sorted(flds)
    plant_cols = [c for c in cols if c not in matched]
    return fams, plant_cols


def col_of(fam: str, dev_id: int, fld: str) -> str:
    prefix = {
        "chiller": "chiller_group/chiller",
        "tower": "tower_group/tower",
        "coolpump": "cooling_group/cooling_pump",
        "coldpump": "colding_group/colding_pump",
    }[fam]
    return f"{prefix}/{dev_id}/{fld}"


def find_time_col(cols: list[str]) -> str | None:
    for cand in ("time_format", "time", "timestamp", "datetime"):
        if cand in cols:
            return cand
    return None


def audit_time(df: pd.DataFrame, tcol: str) -> dict:
    """时间连续性：采样间隔分布、连续段长度分布。"""
    ts = pd.to_datetime(df[tcol], errors="coerce")
    n_bad = int(ts.isna().sum())
    ts_sorted = ts.sort_values()
    monotonic = bool(ts.is_monotonic_increasing)
    n_dup = int(ts.duplicated().sum())

    dt = ts_sorted.diff().dt.total_seconds().dropna()
    gaps = dt.value_counts().sort_values(ascending=False)

    # 以 CTRL_PERIOD 为准切段：间隔 != 900s 即断点
    is_break = (dt != CTRL_PERIOD).to_numpy()
    seg_id = np.concatenate([[0], np.cumsum(is_break)])
    seg_len = pd.Series(seg_id).value_counts().sort_index()

    return {
        "n_rows": len(df),
        "n_bad_ts": n_bad,
        "monotonic": monotonic,
        "n_dup_ts": n_dup,
        "span": f"{ts.min()} .. {ts.max()}",
        "dt_top": gaps.head(5).to_dict(),
        "dt_eq_900_frac": float((dt == CTRL_PERIOD).mean()),
        "n_segments": int(seg_len.size),
        "seg_len_median": float(seg_len.median()),
        "seg_len_max": int(seg_len.max()),
        "seg_ge_64": int((seg_len >= 64).sum()),
        "rows_in_seg_ge_64": int(seg_len[seg_len >= 64].sum()),
    }


def audit_coverage(df: pd.DataFrame, fams: dict[str, FamilyInfo]) -> pd.DataFrame:
    """逐设备逐字段的有效率。有效 = 非 NaN 且非全零常量。"""
    rows = []
    n = len(df)
    for fam, info in fams.items():
        for dev in info.ids:
            for fld in info.fields:
                c = col_of(fam, dev, fld)
                if c not in df.columns:
                    rows.append(dict(family=fam, dev=dev, field=fld, status="MISSING_COL",
                                     valid_frac=0.0, nunique=0, vmin=np.nan, vmax=np.nan))
                    continue
                s = pd.to_numeric(df[c], errors="coerce")
                nn = s.notna()
                nun = int(s.nunique(dropna=True))
                if nn.sum() == 0:
                    status = "ALL_NAN"
                elif nun <= 1:
                    status = "CONSTANT"
                else:
                    status = "OK"
                rows.append(dict(
                    family=fam, dev=dev, field=fld, status=status,
                    valid_frac=float(nn.mean()), nunique=nun,
                    vmin=float(s.min()) if nn.any() else np.nan,
                    vmax=float(s.max()) if nn.any() else np.nan,
                ))
    return pd.DataFrame(rows)


def audit_power_caliber(df: pd.DataFrame, fams: dict[str, FamilyInfo]) -> pd.DataFrame:
    """G2：聚合列 vs 逐台求和 的一致性。

    **只有「非幽灵设备全部有逐台标签」时才做比较。** 上一轮就是漏了这一条：
    直接把有值的几台求和去比全族的聚合列，yb3 的 tower 只有 4/20 台有数据，
    算出来「差 75%」，其实两边根本不是一回事（实测比值 0.199 对开机台数占比
    0.200，完全吻合）。不可比的情形一律标 `not_comparable` 并给出原因，
    不给一个会被误读的数字。

    幽灵设备（`on` 恒为 0）视为不存在，不影响可比性判定。
    """
    rows = []
    T = len(df)
    for fam, agg_col in AGG_POWER.items():
        info = fams[fam]
        n_dev = len(info.ids)
        has_label, is_phantom, per = [], [], []
        for d in info.ids:
            c = col_of(fam, d, "consumption")
            v = (pd.to_numeric(df[c], errors="coerce").to_numpy()
                 if c in df.columns else np.full(T, np.nan))
            has_label.append(bool(np.isfinite(v).any()))
            per.append(np.nan_to_num(v))
            oc = col_of(fam, d, "on")
            on = (pd.to_numeric(df[oc], errors="coerce").to_numpy()
                  if oc in df.columns else np.zeros(T))
            is_phantom.append(not bool(np.nansum(on) > 0))
        has_label = np.array(has_label, dtype=bool)
        is_phantom = np.array(is_phantom, dtype=bool)
        real = ~is_phantom
        per = np.stack(per, 1) if per else np.zeros((T, 0))

        agg = (pd.to_numeric(df[agg_col], errors="coerce")
               if agg_col in df.columns else None)

        rec = dict(family=fam, agg_col=agg_col, n_dev=n_dev,
                   n_label=int(has_label.sum()), n_phantom=int(is_phantom.sum()),
                   has_agg=agg is not None)
        comparable = bool(n_dev and (has_label | is_phantom).all() and agg is not None)
        rec["comparable"] = comparable
        if agg is not None:
            rec["agg_median"] = float(agg.median())
            rec["agg_nonzero_frac"] = float((agg.fillna(0) != 0).mean())
        if real.any():
            ps = per[:, real].sum(1)
            rec["per_sum_median"] = float(np.median(ps))

        # 采信规则与 schema._plant_extras 保持一致：逐台齐全优先，其次聚合列。
        # 「不可比」不等于「不可用」—— bh 没有任何聚合列但逐台标签齐全，
        # 该走 per_device，不是 missing。
        per_full = bool(n_dev and (has_label | is_phantom).all())
        rec["采信"] = ("per_device" if per_full
                       else "aggregate" if agg is not None else "missing")

        if not comparable:
            if not n_dev:
                rec["reason"] = "无该族设备"
            elif per_full and agg is None:
                rec["reason"] = "无聚合列可对照（逐台齐全，直接采信逐台）"
            elif agg is None:
                rec["reason"] = "无聚合列且逐台不全"
            else:
                miss = [int(i) for i in np.flatnonzero(~(has_label | is_phantom))]
                rec["reason"] = (f"仅 {int(has_label.sum())}/{n_dev} 台有逐台标签，"
                                 f"缺 {miss[:8]}{'...' if len(miss) > 8 else ''}")
            rows.append(rec)
            continue

        ps = per[:, real].sum(1)
        a = agg.to_numpy()
        both = np.isfinite(a) & (np.abs(a) > 1e-6)
        if both.sum() > 0:
            rel = np.abs((ps[both] - a[both]) / a[both])
            rec["n_comparable"] = int(both.sum())
            rec["rel_diff_median"] = float(np.median(rel))
            rec["rel_diff_p90"] = float(np.percentile(rel, 90))
            rec["frac_gt_10pct"] = float((rel > 0.10).mean())
        rec["采信"] = "per_device"
        rows.append(rec)
    return pd.DataFrame(rows)


def audit_range(df: pd.DataFrame, fams: dict[str, FamilyInfo],
                plant_cols: list[str]) -> pd.DataFrame:
    """量程越界统计。"""
    rows = []

    def check(col: str, fld: str, tag: str):
        if col not in df.columns:
            return
        lo, hi = RANGE_RULES[fld]
        s = pd.to_numeric(df[col], errors="coerce")
        nn = s.notna()
        if nn.sum() == 0:
            return
        oob = ((s < lo) | (s > hi)) & nn
        if oob.sum() > 0:
            rows.append(dict(scope=tag, col=col, rule=f"[{lo},{hi}]",
                             n_oob=int(oob.sum()), frac_oob=float(oob.sum() / nn.sum()),
                             vmin=float(s.min()), vmax=float(s.max())))

    for fam, info in fams.items():
        for dev in info.ids:
            for fld in info.fields:
                if fld in RANGE_RULES:
                    check(col_of(fam, dev, fld), fld, f"{fam}/{dev}")
    for c in plant_cols:
        if c in RANGE_RULES:
            check(c, c, "plant")
    return pd.DataFrame(rows)


def audit_site(path: Path) -> dict:
    site = path.stem
    df = pd.read_csv(path)
    cols = df.columns.tolist()
    fams, plant_cols = parse_columns(cols)
    tcol = find_time_col(cols)

    n_tokens = 1 + sum(len(f.ids) for f in fams.values())

    res = {
        "site": site,
        "n_cols": len(cols),
        "n_rows": len(df),
        "n_tokens": n_tokens,
        "counts": {k: len(v.ids) for k, v in fams.items()},
        "fields": {k: v.fields for k, v in fams.items()},
        "plant_cols": plant_cols,
        "time_col": tcol,
        "time": audit_time(df, tcol) if tcol else None,
        "coverage": audit_coverage(df, fams),
        "caliber": audit_power_caliber(df, fams),
        "range": audit_range(df, fams, plant_cols),
    }
    return res


def write_report(res: dict) -> Path:
    REPORTS.mkdir(exist_ok=True)
    site = res["site"]
    out = REPORTS / f"audit_{site}.md"
    L: list[str] = []
    A = L.append

    A(f"# 数据体检 — {site}\n")
    A(f"- 列数 {res['n_cols']} / 行数 {res['n_rows']}")
    A(f"- token 数 N = {res['n_tokens']} = plant 1 + " +
      " + ".join(f"{k} {v}" for k, v in res["counts"].items()))
    A(f"- 站级列: {', '.join(res['plant_cols'])}")
    for fam, flds in res["fields"].items():
        A(f"- {fam} 字段: {', '.join(flds)}")
    A("")

    t = res["time"]
    if t:
        A("## 时间连续性\n")
        A(f"- 时间列 `{res['time_col']}`，跨度 {t['span']}")
        A(f"- 单调递增 {t['monotonic']}，重复时间戳 {t['n_dup_ts']}，无法解析 {t['n_bad_ts']}")
        A(f"- 采样间隔 == {CTRL_PERIOD}s 的比例 **{t['dt_eq_900_frac']:.4f}**")
        A(f"- 间隔 top5（秒: 次数）: {t['dt_top']}")
        A(f"- 连续段数 **{t['n_segments']}**，中位长度 {t['seg_len_median']:.0f} 步，"
          f"最长 {t['seg_len_max']} 步")
        A(f"- 长度 >= 64 步（W+H_max）的段: **{t['seg_ge_64']}** 段，"
          f"覆盖 **{t['rows_in_seg_ge_64']}** 行 "
          f"({t['rows_in_seg_ge_64'] / res['n_rows'] * 100:.1f}%)")
        A("")

    cov = res["coverage"]
    A("## 逐设备字段有效性\n")
    bad = cov[cov.status != "OK"]
    A(f"- 总计 {len(cov)} 个 (设备,字段) 组合，异常 **{len(bad)}** 个")
    if len(bad):
        A("")
        A("| 族 | 设备 | 字段 | 状态 | 有效率 | 取值数 |")
        A("|---|---:|---|---|---:|---:|")
        for _, r in bad.iterrows():
            A(f"| {r.family} | {r.dev} | {r.field} | **{r.status}** | "
              f"{r.valid_frac:.3f} | {r['nunique']} |")
    A("")
    A("按族的最低有效率：")
    A("")
    A("| 族 | 字段 | 最低有效率 | 出现设备 |")
    A("|---|---|---:|---|")
    for (fam, fld), g in cov.groupby(["family", "field"]):
        mn = g.valid_frac.min()
        who = ",".join(str(int(d)) for d in g[g.valid_frac == mn].dev.tolist()[:6])
        A(f"| {fam} | {fld} | {mn:.3f} | {who} |")
    A("")

    cal = res["caliber"]
    A("## G2 功率口径一致性\n")
    A("只有「非幽灵设备全部有逐台标签」才做比较；不可比的给原因，不给会被误读的数字。")
    A("")
    A("| 族 | 台数 | 有标签 | 幽灵 | 可比 | 采信 | 聚合中位 | 逐台和中位 | "
      "中位相对差 | >10% 占比 | 不可比原因 |")
    A("|---|---:|---:|---:|---|---|---:|---:|---:|---:|---|")
    for _, r in cal.iterrows():
        def g(k, f="{:.3f}"):
            v = r.get(k, np.nan)
            return "-" if (v is None or (isinstance(v, float) and pd.isna(v))) else f.format(v)
        A(f"| {r.family} | {r.n_dev} | {r.n_label} | {r.n_phantom} | "
          f"{'是' if r.comparable else '**否**'} | {r.get('采信','-')} | "
          f"{g('agg_median','{:.1f}')} | {g('per_sum_median','{:.1f}')} | "
          f"**{g('rel_diff_median')}** | {g('frac_gt_10pct')} | "
          f"{r.get('reason','') or ''} |")
    A("")

    rng = res["range"]
    A("## 量程越界\n")
    if len(rng) == 0:
        A("无越界。")
    else:
        rng = rng.sort_values("frac_oob", ascending=False)
        A(f"共 {len(rng)} 列有越界值，按越界比例降序前 25 条：")
        A("")
        A("| 位置 | 列 | 规则 | 越界数 | 越界率 | min | max |")
        A("|---|---|---|---:|---:|---:|---:|")
        for _, r in rng.head(25).iterrows():
            A(f"| {r.scope} | `{r.col}` | {r.rule} | {r.n_oob} | "
              f"{r.frac_oob:.4f} | {r.vmin:.2f} | {r.vmax:.2f} |")
    A("")

    out.write_text("\n".join(L), encoding="utf-8")
    return out


def main(argv: list[str]) -> int:
    if argv:
        paths = [DATA / f"{a}.csv" if not a.endswith(".csv") else DATA / a for a in argv]
    else:
        paths = sorted(DATA.glob("*.csv"))

    summary = []
    for p in paths:
        if not p.exists():
            print(f"[skip] {p} 不存在")
            continue
        print(f"[audit] {p.name} ...", flush=True)
        try:
            res = audit_site(p)
        except Exception as e:  # noqa: BLE001
            print(f"  FAILED: {type(e).__name__}: {e}")
            continue
        out = write_report(res)
        t = res["time"] or {}
        cal = res["caliber"]
        summary.append(dict(
            site=res["site"], n_rows=res["n_rows"], n_tokens=res["n_tokens"],
            **{f"n_{k}": v for k, v in res["counts"].items()},
            dt900_frac=t.get("dt_eq_900_frac"),
            n_seg=t.get("n_segments"), seg_median=t.get("seg_len_median"),
            rows_in_seg_ge_64=t.get("rows_in_seg_ge_64"),
            n_field_issues=int((res["coverage"].status != "OK").sum()),
            n_oob_cols=len(res["range"]),
            **{f"采信_{r.family}": r.get("采信") for _, r in cal.iterrows()},
            **{f"标签_{r.family}": f"{r.n_label}/{r.n_dev}" for _, r in cal.iterrows()},
            **{f"reldiff_{r.family}": (r.get("rel_diff_median") if r.comparable else None)
               for _, r in cal.iterrows()},
        ))
        print(f"  -> {out.relative_to(ROOT)}")

    if summary:
        REPORTS.mkdir(exist_ok=True)
        sdf = pd.DataFrame(summary)
        sp = REPORTS / "audit_summary.csv"
        sdf.to_csv(sp, index=False, encoding="utf-8-sig")
        print(f"\n汇总 -> {sp.relative_to(ROOT)}")
        with pd.option_context("display.width", 200, "display.max_columns", 40):
            print(sdf.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
