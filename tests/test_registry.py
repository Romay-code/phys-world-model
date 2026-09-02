"""站点登记表：冷站分组与能力分档（P5 留出设计的结构约束）。

这份表是**留出设计的唯一事实来源**。它错了不会有任何报错，只会让
「零样本」这个数偏乐观 —— 所以每一条都要能从数据重算核对。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from physwm.data.registry import (  # noqa: E402
    DUP_FILES, PLANT_OF, TIER_DESC, capability_tier, expand_holdout,
    holdout_kind, plant_of, siblings)

PD, AG = "per_device", "aggregate"


def _src(chiller=PD, tower=PD, coolpump=PD, coldpump=PD):
    return {"chiller": chiller, "tower": tower,
            "coolpump": coolpump, "coldpump": coldpump}


# --- 分档 -----------------------------------------------------------------

@pytest.mark.parametrize("src,want", [
    (_src(), "A"),                                             # bh / tx
    (_src(tower=AG), "B"),                                     # hx / yb3 / yb / yc / pa2
    (_src(tower=AG, coolpump=AG, coldpump=AG), "C"),           # zx / pb1 / pc2
    (_src(chiller=AG, tower=AG, coolpump=AG, coldpump=AG), "D"),  # pc3
])
def test_capability_tier(src, want):
    assert capability_tier(src) == want


def test_chiller_aggregate_always_lands_in_D():
    """冷机逐台功率是分档的**决定性**通道。

    §13 #47：pc3 崩掉的直接原因就是这条输入通道全空，而 12 个训练站都有。
    只要它是聚合的，无论别的族多齐，都必须落 D 档。
    """
    for tower in (PD, AG):
        for pump in (PD, AG):
            assert capability_tier(_src(AG, tower, pump, pump)) == "D"


def test_every_tier_has_a_description():
    for t in "ABCD":
        assert TIER_DESC.get(t)


# --- 冷站分组 -------------------------------------------------------------

def test_singleton_site_is_its_own_plant():
    assert plant_of("hx") == "hx"
    assert siblings("hx") == ["hx"]


@pytest.mark.parametrize("a,b", [("yb_中温", "yb_低温"),
                                 ("yc_中温", "yc_低温"),
                                 ("pa2_中温", "pa2_低温")])
def test_loop_pairs_share_a_plant(a, b):
    assert plant_of(a) == plant_of(b)
    assert siblings(a) == siblings(b) == sorted([a, b])


def test_holdout_expands_to_whole_plant():
    """点一个回路必须连兄弟回路一起留出。

    漏掉兄弟回路 -> 同楼、同天气、同排程的数据还在训练集里，
    「零样本」偏乐观，**而且不报任何错**。
    """
    assert expand_holdout(["yb_低温"]) == ("yb_中温", "yb_低温")
    assert expand_holdout(["yc_中温"]) == ("yc_中温", "yc_低温")


def test_holdout_of_singleton_is_unchanged():
    assert expand_holdout(["hx"]) == ("hx",)
    assert expand_holdout(["hx", "pc3"]) == ("hx", "pc3")


def test_holdout_is_intersected_with_available_sites():
    """`only=` 子集下不能凭空造出不存在的站。"""
    assert expand_holdout(["yb_低温"], all_sites=["yb_低温", "yb3"]) == ("yb_低温",)


def test_holdout_is_idempotent():
    once = expand_holdout(["yb_低温"])
    assert expand_holdout(once) == once


# --- 留出类型 -------------------------------------------------------------

def test_holdout_kind_distinguishes_the_two_questions():
    """同档跨站与跨档外推是两个难度完全不同的问题，混报即失真。

    P5-A 把 hx（B 档，训练集有 5 个 B 档冷站）与 pc3（D 档，训练集 0 个）
    并排当成「留出站零样本」报，两者一个 0.9335 一个 −1.64。
    """
    assert holdout_kind("B", {"A", "B", "C"}) == "同档跨站"
    assert holdout_kind("D", {"A", "B", "C"}) == "跨档外推"


# --- 与数据核对（慢，需读全部 CSV）----------------------------------------

@pytest.mark.slow
def test_plant_grouping_matches_data():
    """从数据重算冷站分组，核对 `PLANT_OF`。

    判据与 `tools/diag_site_independence.py` 一致：**时间轴逐位一致 +
    湿球相关 > 0.99**。不能用「湿球逐位相同」——实测 yc 两回路中位差
    0.24 K、pa2 是 0.43 K（同址的两个传感器），只有 yb 是 0.000。
    """
    import itertools

    import pandas as pd

    from physwm.data import schema as S
    from physwm.data.dataset import load_site

    k = S.CANON_FIELDS["plant"].index("wet_bulb")
    info = {}
    for f in sorted((ROOT / "data").glob("*.csv")):
        if f.stem in DUP_FILES:
            continue
        sd = load_site(f, f.stem)
        df = pd.read_csv(f, low_memory=False)
        info[f.stem] = (pd.to_datetime(df[sd.sch.time_col], errors="coerce").to_numpy(),
                        sd.x[:, 0, k].astype(np.float64),
                        sd.avail[:, 0, k].astype(bool))

    found = set()
    for a, b in itertools.combinations(info, 2):
        (ta, wa, va), (tb, wb, vb) = info[a], info[b]
        if len(ta) != len(tb) or not bool((ta == tb).all()):
            continue
        m = va & vb
        if m.sum() > 100 and float(np.corrcoef(wa[m], wb[m])[0, 1]) > 0.99:
            found.add(frozenset((a, b)))

    declared = {frozenset(siblings(s)) for s in PLANT_OF}
    assert found == declared, (
        f"登记表与数据不符。数据说是同一冷站的：{sorted(map(sorted, found))}；"
        f"表里写的：{sorted(map(sorted, declared))}")


@pytest.mark.slow
def test_tier_assignment_matches_data():
    """从数据重算每个站的能力档，确认 D 档确实只有 pc3 一个成员。

    「D 档 n=1」是 P5-A 失败的结构性原因（留出它 = 训练集里没有任何一个站
    有那个模式）。哪天新数据让 D 档多了一个成员，这条会红 —— 那时
    pc3 就可以做真正的同档跨站零样本了。
    """
    from physwm.data.dataset import WindowSpec, build_site_bundle

    spec = WindowSpec()
    tiers = {}
    for f in sorted((ROOT / "data").glob("*.csv")):
        if f.stem in DUP_FILES:
            continue
        b = build_site_bundle(f, f.stem, spec, out_dir=None)
        tiers[f.stem] = capability_tier(b["info"]["power_src"])

    assert tiers["bh"] == tiers["tx"] == "A"
    assert tiers["hx"] == tiers["yb3"] == "B"
    assert tiers["zx"] == tiers["pb1"] == tiers["pc2"] == "C"
    assert tiers["pc3"] == "D"

    d = [s for s, t in tiers.items() if t == "D"]
    assert d == ["pc3"], f"D 档成员变了：{d}"

    plants = {plant_of(s) for s in tiers}
    assert len(plants) == 11, f"独立冷站数应为 11，实测 {len(plants)}：{sorted(plants)}"
