"""时段留出（轴 3）。

留出错了不会报错，只会让「季节外推」这个数偷偷变成「季节内插」。
下面每条都钉一个这样的失效形态。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from physwm.data.dataset import WindowSpec  # noqa: E402
from physwm.data.temporal_holdout import (  # noqa: E402
    month_mask, split_by_period, window_mask)


def _ts(n, start="2025-01-01", freq="15min"):
    return pd.Series(pd.date_range(start, periods=n, freq=freq))


def _windows(t0s, W=16, H=48):
    t0s = np.asarray(t0s)
    return np.stack([np.zeros(len(t0s), int), t0s, t0s - W + 1], 1)


# --- 掩码 -----------------------------------------------------------------

def test_month_mask_spans_all_years():
    """按月份序数留出必须跨所有年份。

    实测日历：bh 覆盖 2023-12~2024-09，其余 13 站从 2024-10 起，两段不重叠。
    若按绝对时间窗留「2025 年 7-8 月」，bh 的 2024 年 7-8 月还在训练集里，
    模型照样见过盛夏 —— 那个留出是假的。
    """
    ts = pd.Series(pd.date_range("2024-06-01", periods=800, freq="D"))
    m = month_mask(ts, (7, 8))
    hit = pd.to_datetime(ts[m])
    # 数据横跨 2024-06 ~ 2026-08，故七八月应在**每个**出现过的年份里都被选中
    assert set(hit.dt.year) == {2024, 2025, 2026}, f"只覆盖了 {set(hit.dt.year)}"
    assert set(hit.dt.month) == {7, 8}
    # 反证：绝对时间窗只会命中一年，这正是它不能用来考季节外推的原因
    w = window_mask(ts, start="2025-07-01", end="2025-09-01")
    assert set(pd.to_datetime(ts[w]).dt.year) == {2025}


@pytest.mark.parametrize("bad", [(0,), (13,), (7, 99)])
def test_month_mask_rejects_bad_months(bad):
    with pytest.raises(ValueError, match="月份"):
        month_mask(_ts(10), bad)


def test_window_mask_is_half_open():
    ts = pd.Series(pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"]))
    m = window_mask(ts, start="2025-01-02", end="2025-01-03")
    assert m.tolist() == [False, True, False], "区间应为左闭右开"


def test_window_mask_refuses_empty_bounds():
    with pytest.raises(ValueError):
        window_mask(_ts(10))


# --- 划分 -----------------------------------------------------------------

def test_window_touching_holdout_goes_to_neither():
    """只要有一步踩进留出期，该窗口两边都不要。

    「部分重叠算训练」会让训练窗口的历史窗或推演段见过留出期数据，
    留出立刻失效，而且不会有任何报错。
    """
    spec = WindowSpec(W=16, H=48, embargo=0)
    T = 400
    hold = np.zeros(T, bool)
    hold[200:260] = True
    # t0=210 的区间 [195, 258] 与留出期部分重叠
    parts = split_by_period(_windows([210]), hold, spec)
    assert len(parts["held"]) == 0
    assert len(parts["rest"]) == 0
    assert parts["dropped"] == 1


def test_fully_inside_goes_to_held():
    spec = WindowSpec(W=16, H=48, embargo=0)
    T = 400
    hold = np.zeros(T, bool)
    hold[100:300] = True
    parts = split_by_period(_windows([160]), hold, spec)   # [145, 208] 全在内
    assert len(parts["held"]) == 1 and len(parts["rest"]) == 0


def test_fully_outside_goes_to_rest():
    spec = WindowSpec(W=16, H=48, embargo=0)
    hold = np.zeros(400, bool)
    hold[300:] = True
    parts = split_by_period(_windows([100]), hold, spec)   # [85, 148] 全在外
    assert len(parts["rest"]) == 1 and len(parts["held"]) == 0


def test_embargo_pushes_near_boundary_windows_out():
    """距留出期不足 embargo 的窗口不能进训练 —— 自相关会把信息漏过去。"""
    spec_no = WindowSpec(W=16, H=48, embargo=0)
    spec_g = WindowSpec(W=16, H=48, embargo=64)
    hold = np.zeros(600, bool)
    hold[300:400] = True
    w = _windows([230])                     # 区间 [215, 278]，右端距 300 只有 22
    assert len(split_by_period(w, hold, spec_no)["rest"]) == 1
    assert len(split_by_period(w, hold, spec_g)["rest"]) == 0, "embargo 没生效"


def test_held_and_rest_share_no_rows():
    """两侧的行集合必须完全不相交 —— 这是留出成立的最低要求。"""
    spec = WindowSpec(W=16, H=48, embargo=64)
    T = 3000
    ts = _ts(T)
    hold = month_mask(ts, (2,))
    w = _windows(np.arange(16, T - 49))
    parts = split_by_period(w, hold, spec)

    def rows(ws):
        out = set()
        for _, t0, lo in ws:
            out |= set(range(int(lo), int(t0) + spec.H + 1))
        return out

    assert not (rows(parts["held"]) & rows(parts["rest"]))


def test_counts_add_up():
    spec = WindowSpec(W=16, H=48, embargo=32)
    T = 2000
    hold = np.zeros(T, bool)
    hold[800:1200] = True
    w = _windows(np.arange(16, T - 49))
    p = split_by_period(w, hold, spec)
    assert len(p["held"]) + len(p["rest"]) + p["dropped"] == len(w)


def test_seasonal_holdout_leaves_no_summer_in_training():
    """轴 3 的核心性质：留出七八月后，可训窗口里一步夏天都不能有。"""
    spec = WindowSpec(W=16, H=48, embargo=64)
    T = 40000                                     # 约 417 天 @15min
    ts = _ts(T, start="2024-10-01")
    hold = month_mask(ts, (7, 8))
    w = _windows(np.arange(16, T - 49))
    parts = split_by_period(w, hold, spec)
    assert len(parts["held"]) > 0 and len(parts["rest"]) > 0
    months = set()
    for _, t0, lo in parts["rest"]:
        months |= set(pd.to_datetime(ts[int(lo):int(t0) + spec.H + 1]).dt.month)
    assert not (months & {7, 8}), f"训练侧仍含夏季月份：{sorted(months & {7, 8})}"


def test_empty_windows_is_handled():
    spec = WindowSpec()
    p = split_by_period(np.empty((0, 3), dtype=np.int64), np.zeros(10, bool), spec)
    assert len(p["held"]) == 0 and len(p["rest"]) == 0 and p["dropped"] == 0


# --- 端到端：接进 build_site_bundle 之后仍然无泄漏（慢，读真实 CSV）--------

@pytest.mark.slow
def test_bundle_season_holdout_is_leak_free_on_real_site():
    """真站点上端到端验证：train 一步夏天都不能有，test 全是夏天。

    `assert_no_leak` 已经保证三个 split 行集合不相交，但它**不检查
    「训练侧是否含留出月份」** —— 那才是季节外推成立与否的关键，
    且错了不会有任何报错，只会让「季节外推」悄悄变成「季节内插」。
    """
    from physwm.data.dataset import build_site_bundle, covered_rows

    site = ROOT / "data" / "yb3.csv"
    spec = WindowSpec()
    b = build_site_bundle(site, "yb3", spec, out_dir=None, season_months=(7, 8))
    sd, sp = b["data"], b["splits"]
    ts = pd.to_datetime(pd.read_csv(site, usecols=[sd.sch.time_col],
                                    low_memory=False)[sd.sch.time_col])
    mon = ts.dt.month.to_numpy()

    for name in ("train", "val"):
        rows = covered_rows(sp[name], spec, sd.T)
        hit = set(mon[rows]) & {7, 8}
        assert not hit, f"{name} 侧仍含夏季月份 {sorted(hit)}"
    test_rows = covered_rows(sp["test"], spec, sd.T)
    assert set(mon[test_rows]) <= {7, 8}, "test 侧混进了非夏季月份"
    assert len(sp["train"]) > 0 and len(sp["test"]) > 0


@pytest.mark.slow
def test_bundle_normalizer_is_refit_without_the_holdout_period():
    """归一化必须在切完之后重算 —— 沿用留出前那份就把夏天的统计量带了进去。

    与 §13 #8 同型：那次是分块划分下用 `slice(lo,hi)` 取统计量，
    统计量横跨了 val/test。两者都不报错，只让结论失真。
    """
    from physwm.data.dataset import build_site_bundle

    site = ROOT / "data" / "yb3.csv"
    spec = WindowSpec()
    plain = build_site_bundle(site, "yb3", spec, out_dir=None)
    season = build_site_bundle(site, "yb3", spec, out_dir=None, season_months=(7, 8))
    a, b = plain["norm"], season["norm"]
    same = np.allclose(np.asarray(a.center), np.asarray(b.center)) and \
        np.allclose(np.asarray(a.scale), np.asarray(b.scale))
    assert not same, "两种划分给出了完全相同的归一化参数，说明没有重算"
