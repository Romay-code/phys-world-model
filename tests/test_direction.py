"""方向一致性评测的构造正确性（physwm/eval/direction.py）。

这里测的不是「模型方向对不对」——那是训练结果。这里测的是
**评测装置本身有没有把实验做对**：改错了时间段、改漏了归一化通路、
或者把停机设备算进分母，都会让 dir_vr 变成一个好看但无意义的数字。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from physwm.data import schema as S  # noqa: E402
from physwm.eval.direction import (DirSpec, default_specs,  # noqa: E402
                                   make_intervene, rollout_direction_check)

N_DEV = {"chiller": 7, "tower": 20, "coolpump": 7, "coldpump": 8}
N = 1 + sum(N_DEV.values())
F, W, H, B = 9, 16, 5, 3


class _Sch:
    """最小 schema 替身：只需要 token_index。"""
    token_index = ([("plant", 0)]
                   + [("chiller", i) for i in range(7)]
                   + [("tower", i) for i in range(20)]
                   + [("coolpump", i) for i in range(7)]
                   + [("coldpump", i) for i in range(8)])


def _rows(fam):
    return [i for i, (f, _) in enumerate(_Sch.token_index) if f == fam]


def _mk(scale_val=2.0):
    raw = torch.rand(B, W + H, N, F) * 10 + 5
    xn = torch.randn(B, W + H, N, F)
    ns = torch.full((N, F), scale_val)
    return raw, xn, ns


def test_history_window_is_bit_preserved():
    """历史窗被改动 = z_{t0} 变了 = 比较的不再是「同一 z 下换动作」。"""
    raw, xn, ns = _mk()
    spec = DirSpec("t", "tower", "frequency", 1.0, {"approx": -1}, why="t",
                   on_only=False)
    fn = make_intervene(_Sch, ns, spec, None)
    x2, r2 = fn(xn.clone(), raw.clone(), W)
    assert torch.equal(r2[:, :W], raw[:, :W]), "seq_raw 历史窗被改动"
    assert torch.equal(x2[:, :W], xn[:, :W]), "seq_x 历史窗被改动"


def test_only_target_family_and_field_change():
    raw, xn, ns = _mk()
    spec = DirSpec("t", "tower", "frequency", 1.0, {"approx": -1}, why="t",
                   on_only=False)
    fn = make_intervene(_Sch, ns, spec, None)
    _, r2 = fn(xn.clone(), raw.clone(), W)
    k = S.CANON_FIELDS["tower"].index("frequency")
    tw = _rows("tower")
    other = [i for i in range(N) if i not in tw]
    assert torch.equal(r2[:, :, other], raw[:, :, other]), "改到了别的族"
    kk = [j for j in range(F) if j != k]
    assert torch.equal(r2[:, :, tw][:, :, :, kk], raw[:, :, tw][:, :, :, kk]), "改到了别的字段"
    assert torch.allclose(r2[:, W:, tw, k], raw[:, W:, tw, k] + 1.0)


def test_normalized_and_raw_stay_consistent():
    """只改一个通路，编码器与物理解码器会看到不同的世界。"""
    raw, xn, ns = _mk(scale_val=4.0)
    spec = DirSpec("t", "tower", "frequency", 2.0, {"approx": -1}, why="t",
                   on_only=False)
    fn = make_intervene(_Sch, ns, spec, None)
    x2, r2 = fn(xn.clone(), raw.clone(), W)
    k = S.CANON_FIELDS["tower"].index("frequency")
    tw = _rows("tower")
    d_raw = (r2 - raw)[:, W:, tw, k]
    d_xn = (x2 - xn)[:, W:, tw, k]
    assert torch.allclose(d_xn, d_raw / 4.0, atol=1e-5), "归一化增量与原始增量不自洽"


def test_on_only_leaves_stopped_devices_alone():
    raw, xn, ns = _mk()
    k_on = S.CANON_FIELDS["tower"].index("on")
    tw = _rows("tower")
    raw[:, :, tw, k_on] = 0.0
    raw[:, :, tw[:5], k_on] = 1.0          # 只有前 5 台开机
    spec = DirSpec("t", "tower", "frequency", 1.0, {"approx": -1}, why="t")
    fn = make_intervene(_Sch, ns, spec, None)
    _, r2 = fn(xn.clone(), raw.clone(), W)
    k = S.CANON_FIELDS["tower"].index("frequency")
    d = (r2 - raw)[:, W:, tw, k]
    assert torch.allclose(d[:, :, :5], torch.ones_like(d[:, :, :5]))
    assert torch.allclose(d[:, :, 5:], torch.zeros_like(d[:, :, 5:])), "停机设备被加了频率"


def test_clip_keeps_quasi_action_inside_measured_range():
    """G3：cold_out_temp 按准动作处理，扰动幅度必须留在实测分布内。"""
    raw, xn, ns = _mk()
    ch = _rows("chiller")
    k = S.CANON_FIELDS["chiller"].index("cold_out_temp")
    k_on = S.CANON_FIELDS["chiller"].index("on")
    raw[:, :, ch, k_on] = 1.0
    raw[:, :, ch, k] = 29.9                      # 贴着上界 30.0
    spec = [s for s in default_specs() if s.fld == "cold_out_temp"][0]
    fn = make_intervene(_Sch, ns, spec, None)
    _, r2 = fn(xn.clone(), raw.clone(), W)
    assert float(r2[:, W:, ch, k].max()) <= S.RANGE_RULES["cold_out_temp"][1] + 1e-6


def test_priors_do_not_claim_a_sign_for_plant_power_under_fan_speed():
    """塔频/泵频对 P_plant 的净效应非单调，给它安单调先验是物理错误。

    这条守的是一个**认知**错误而非代码错误：设计文档 §5.3 原文写的是拿 P_plant
    比先验符号。若有人照抄回来，本测试会红。
    """
    for s in default_specs():
        if s.fld == "frequency":
            assert "P_plant" not in s.targets, (
                f"{s.name} 给 P_plant 安了单调先验。塔/泵频↑ 使自身功率↑ 但冷机功率↓，"
                f"净效应在最优点两侧变号 —— 这正是冷站优化非平凡的原因")
        assert s.why, f"{s.name} 缺物理依据说明"


def test_direction_check_runs_end_to_end():
    """整条链路能跑通，且返回结构完整。"""
    from physwm.model.world_model import ModelConfig, WorldModel
    from physwm.train.loop import Context

    torch.manual_seed(0)
    model = WorldModel(ModelConfig(), N_DEV, F)
    type_id = torch.tensor([0] + [1] * 7 + [2] * 20 + [3] * 7 + [4] * 8)
    ctx = Context(desc=torch.randn(N, 32), stat_rel=torch.randn(N, N, 5),
                  type_id=type_id, W=W, site_ctx_dim=8)

    class _DS:
        spec = type("sp", (), {"W": W, "H": H})()
        def set_H(self, h): pass
        def epoch(self, bs, shuffle=True, drop_last=True):
            raw = torch.rand(B, W + H, N, F) * 10 + 5
            k_on = S.CANON_FIELDS["tower"].index("on")
            raw[:, :, _rows("tower"), k_on] = 1.0
            raw[:, :, _rows("chiller"), S.CANON_FIELDS["chiller"].index("on")] = 1.0
            yield {"seq_x": torch.randn(B, W + H, N, F),
                   "seq_avail": torch.ones(B, W + H, N, F),
                   "seq_raw": raw,
                   "P_plant": torch.rand(B, H) * 1000}

    rep = rollout_direction_check(model, _DS(), ctx, H=H, device=torch.device("cpu"),
                                  sch=_Sch, norm_scale=torch.ones(N, F),
                                  batch_size=B, max_batches=1)
    assert set(rep) == {s.name for s in default_specs()}
    for name, d in rep.items():
        assert d["why"], name
        for tgt, t in d["targets"].items():
            assert len(t["per_h"]) == H, f"{name}/{tgt} 逐 h 长度不对"
            assert all(0.0 <= v <= 1.0 for v in t["per_h"]), f"{name}/{tgt} 违例率越界"
            assert t["n_eval"] > 0, f"{name}/{tgt} 没有任何有效样本（分母被停机掩码清空？）"
