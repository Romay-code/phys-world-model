"""线性探针（设计文档 §5.2 表征通用性）。

每条测试对应一个「错了不会报错、只会让结论失真」的失效形态。
最要紧的是 `test_probe_rejects_noise` 与 `test_trivial_targets_are_flagged`：
前者防探针把噪声也拟合出高 R²，后者防重蹈 §13 #36 —— 拿结构上必然成立的项
当成物理正确性证据。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from physwm.data import schema as S  # noqa: E402
from physwm.eval.probe import (  # noqa: E402
    TARGETS, compute_targets, pool_by_family, ridge_probe)


# --- 探针本身 -------------------------------------------------------------

def test_probe_recovers_a_linear_target():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 20))
    y = X @ rng.normal(size=20) + 0.05 * rng.normal(size=600)
    assert ridge_probe(X[:400], y[:400], X[400:], y[400:]) > 0.95


def test_probe_rejects_noise():
    """目标与特征无关时 R² 必须 ~0。

    岭回归有 20~960 维特征、样本可能只有几百 —— 若 alpha 选得不对
    （比如在 val 上选），纯噪声也能拟出漂亮的 R²，整套探针结论就是假的。
    """
    rng = np.random.default_rng(1)
    X = rng.normal(size=(600, 20))
    y = rng.normal(size=600)
    assert ridge_probe(X[:400], y[:400], X[400:], y[400:]) < 0.15


def test_alpha_is_not_selected_on_val():
    """alpha 只能在 train 内部切出来的那一刀上选。

    直接验行为：把 val 换成一份与 train 无关的噪声，若实现偷看了 val，
    它会挑一个在噪声上碰巧好的 alpha，R² 会明显高于 0。
    """
    rng = np.random.default_rng(2)
    Xtr, Xva = rng.normal(size=(500, 30)), rng.normal(size=(300, 30))
    ytr = Xtr @ rng.normal(size=30)
    yva = rng.normal(size=300)
    assert ridge_probe(Xtr, ytr, Xva, yva) < 0.15


def test_short_input_returns_nan_not_garbage():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(10, 5))
    assert np.isnan(ridge_probe(X, rng.normal(size=10), X, rng.normal(size=10)))


# --- 平凡性标注 -----------------------------------------------------------

def test_trivial_targets_are_flagged():
    """`plr` / `approach` / `dt_evap` 必须被标成平凡。

    它们是输入通道本身或其线性组合，高 R² 只证明编码器没丢输入。
    与 §13 #36 的 `w_*_dev` 同类：**结构上必然成立的项不构成物理证据。**
    """
    triv = {t.name for t in TARGETS if t.trivial}
    assert {"plr", "approach", "dt_evap"} <= triv


def test_has_at_least_three_nontrivial_targets():
    """门限要「≥3 类」，而只有非平凡目标算数 —— 少于 3 个就永远不可能达标。"""
    assert sum(1 for t in TARGETS if not t.trivial) >= 3


# --- 目标计算 -------------------------------------------------------------

class _Sch:
    n_dev = {"chiller": 2, "tower": 2, "coolpump": 1, "coldpump": 1}
    site = "fake"

    @property
    def token_index(self):
        idx = [("plant", 0)]
        for fam in S.DEVICE_FAMILIES:
            idx.extend((fam, d) for d in range(self.n_dev[fam]))
        return idx


class _SD:
    def __init__(self, x, avail):
        self.x, self.avail, self.sch = x, avail, _Sch()


def _mk(T=50):
    N = 1 + sum(_Sch.n_dev.values())
    F = max(len(v) for v in S.CANON_FIELDS.values())
    x = np.zeros((T, N, F), dtype=np.float32)
    av = np.ones((T, N, F), dtype=np.float32)
    P, C = S.CANON_FIELDS["plant"], S.CANON_FIELDS["chiller"]
    ch = [1, 2]
    x[:, 0, P.index("wet_bulb")] = 25.0
    x[:, 0, P.index("tower_out")] = 31.0            # approach = 6.0
    for i in ch:
        x[:, i, C.index("on")] = 1.0
        x[:, i, C.index("plr")] = 0.6
        x[:, i, C.index("cold_out_temp")] = 7.0
        x[:, i, C.index("cold_back_temp")] = 12.0   # dt_evap = 5.0
        x[:, i, C.index("cool_out_temp")] = 37.0
        x[:, i, C.index("cool_back_temp")] = 31.0   # eps = 6/12 = 0.5
    return _SD(x, av)


def test_targets_match_hand_computed_values():
    t = compute_targets(_mk())
    assert np.allclose(t["approach"], 6.0)
    assert np.allclose(t["plr"], 0.6)
    assert np.allclose(t["dt_evap"], 5.0)
    assert np.allclose(t["eps_tower"], 0.5)
    assert np.allclose(t["ntu"], -np.log(0.5))
    # COP_carnot = T_ev/(T_cd - T_ev) = 280.15 / 30
    assert np.allclose(t["cop_carnot"], 280.15 / 30.0)


def test_approach_matches_decoder_definition():
    """探针的 approach 必须与解码器的 `tower_out = approx_eff + wb` 同定义。

    两处若漂移，探针会去回读一个模型压根没在建模的量，而这不会有任何报错。
    """
    src = (ROOT / "physwm" / "model" / "decoder.py").read_text(encoding="utf-8")
    assert "tower_out = approx_eff + wb" in src


def test_unavailable_rows_become_nan_not_zero():
    """`avail=0` 的行必须变 NaN 被丢掉，不能当成 0 参与拟合。

    当成 0 会把「停机无读数」读成「逼近度 = −湿球」，是 §13 #2 同一个坑。
    """
    sd = _mk()
    sd.avail[:10, 0, S.CANON_FIELDS["plant"].index("tower_out")] = 0.0
    t = compute_targets(sd)
    assert np.isnan(t["approach"][:10]).all()
    assert np.isfinite(t["approach"][10:]).all()


def test_off_chillers_excluded_from_means():
    """停机冷机不得进逐台均值 —— 它的温度读数恒为 0（§13 #2）。"""
    sd = _mk()
    sd.x[:, 2, S.CANON_FIELDS["chiller"].index("on")] = 0.0
    sd.x[:, 2, S.CANON_FIELDS["chiller"].index("plr")] = 0.0
    t = compute_targets(sd)
    assert np.allclose(t["plr"], 0.6), "停机机组把 plr 均值拉低了"


# --- 池化 -----------------------------------------------------------------

def test_pool_is_independent_of_device_count():
    """池化后的维度必须只由 d 决定，与台数无关 —— 否则跨站不可比。"""
    d = 8
    for n_ch in (2, 7, 11):
        n_dev = {"chiller": n_ch, "tower": 3, "coolpump": 2, "coldpump": 2}
        n = 1 + sum(n_dev.values())
        order = {f: i for i, f in enumerate(S.FAMILIES)}
        tid = torch.tensor([order["plant"]]
                           + sum(([order[f]] * n_dev[f] for f in S.DEVICE_FAMILIES), []))
        out = pool_by_family(torch.randn(4, n, d), tid)
        assert out.shape == (4, len(S.FAMILIES) * d)


def test_pool_averages_within_family():
    d, n_dev = 3, {"chiller": 2, "tower": 1, "coolpump": 1, "coldpump": 1}
    order = {f: i for i, f in enumerate(S.FAMILIES)}
    tid = torch.tensor([order["plant"]]
                       + sum(([order[f]] * n_dev[f] for f in S.DEVICE_FAMILIES), []))
    v = torch.zeros(1, 1 + sum(n_dev.values()), d)
    v[0, 1] = 1.0
    v[0, 2] = 3.0                       # 两台冷机，均值应为 2.0
    out = pool_by_family(v, tid)
    ci = S.FAMILIES.index("chiller")
    assert torch.allclose(out[0, ci * d:(ci + 1) * d], torch.full((d,), 2.0))


# --- 有效性守卫 -----------------------------------------------------------

def _row(target, trivial, r2_z, r2_raw):
    from physwm.eval.probe import is_readable
    return {"target": target, "trivial": trivial, "r2_z": r2_z,
            "r2_raw": r2_raw, "readable": is_readable(r2_raw)}


@pytest.mark.parametrize("r2_raw,want", [(-6.32, False), (-0.01, False),
                                         (0.0, True), (0.8, True),
                                         (float("nan"), False)])
def test_negative_control_is_unreadable(r2_raw, want):
    """对照臂为负 -> 不可读。

    实测踩到：yb3 的 `eps_tower` 对照臂 R² 是 **−6.32** —— 连原始输入都
    推不出来的目标，`R²(z)` 反映的是 train/val 划分而不是表征质量。
    """
    from physwm.eval.probe import is_readable
    assert is_readable(r2_raw) is want


def test_unreadable_target_counts_as_neither_pass_nor_fail():
    """不可读的目标既不能算过线，也不能算模型失败 —— 必须单列。"""
    from physwm.eval.probe import summarize
    rows = [_row("cop_carnot", False, 0.95, -3.89),    # z 很高，但对照臂为负
            _row("eps_tower", False, 0.88, 0.70),      # 真过线
            _row("ntu", False, 0.20, 0.30)]            # 真未过线
    s = summarize(rows, gate=0.70)
    assert s["n_pass_nontrivial"] == 1, "不可读的那项被算成过线了"
    assert s["n_nontrivial_unreadable"] == 1
    assert s["n_nontrivial"] == 3


def test_trivial_target_never_counts_toward_gate():
    """平凡目标即使 R²=1.0 也不进门限计数（§13 #36 同类）。"""
    from physwm.eval.probe import summarize
    rows = [_row("approach", True, 1.0, 1.0), _row("plr", True, 0.99, 0.94)]
    s = summarize(rows, gate=0.70)
    assert s["n_pass_nontrivial"] == 0
    assert s["n_nontrivial"] == 0
    assert s["n_pass_all"] == 2, "全部计数里平凡项仍应出现，只是不进门限"
