"""少样本适配 + 零样本守门（M2）。

守门是验收口径明文要求的一条能力：「任意样本规模下精度不低于零样本水平」。
它错了不会报错 —— 只会在报告里给出一条比零样本还差的曲线，或者更糟，
一条**用 test 选出来的**好看曲线。下面每条测试各钉一个这样的失效形态。
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from physwm.train.fewshot import (  # noqa: E402
    STEPS_PER_DAY, FewShotConfig, gate_holds, split_adapt_gate)
from physwm.train.loop import Context  # noqa: E402


def _windows(n=3000, t0=0):
    """[seg_idx, t0, span_start] —— 与 `enumerate_windows` 同布局。"""
    t = np.arange(t0, t0 + n)
    return np.stack([np.zeros(n, int), t, t], 1)


# --- 适配段的取法 ---------------------------------------------------------

def test_adapt_block_is_the_earliest_not_random():
    """必须取**最早**的一段。

    随机抽等于偷看未来（真实上线是「先采一段再上线」），而且会把
    「冬季适配、夏季部署」这个真实失败模式平均掉、看不见。
    """
    w = _windows(3000, t0=500)
    a, _ = split_adapt_gate(w, FewShotConfig(n_days=7))
    assert int(a[:, 1].min()) == 500, "没有从最早的窗口开始"
    assert int(a[:, 1].max()) < 500 + 7 * STEPS_PER_DAY


def test_gate_is_strictly_after_adapt():
    """守门段必须在适配段**之后**，否则守门在用适配期见过的分布判卷。"""
    a, g = split_adapt_gate(_windows(), FewShotConfig(n_days=7))
    assert len(a) and len(g)
    assert int(g[:, 1].min()) > int(a[:, 1].max())


def test_adapt_and_gate_do_not_overlap():
    a, g = split_adapt_gate(_windows(), FewShotConfig(n_days=7))
    assert not (set(a[:, 1].tolist()) & set(g[:, 1].tolist()))


@pytest.mark.parametrize("days,want_rows", [(1, 96), (3, 288), (7, 672), (30, 2880)])
def test_days_map_to_control_periods(days, want_rows):
    """900 s 控制周期 -> 96 步/天。换算错了，整条曲线的横轴就是错的。"""
    assert FewShotConfig(n_days=days).n_rows == want_rows


def test_tiny_budget_yields_empty_gate_not_a_bogus_split():
    """数据少到切不出守门段时，必须给出空 gate，让上层走「保留预训练权重」。"""
    a, g = split_adapt_gate(_windows(50), FewShotConfig(n_days=0.05))
    assert len(g) == 0


# --- 守门判据 -------------------------------------------------------------

def _row(zero, final):
    return {"test_zero": {"R2": zero}, "test_final": {"R2": final}}


def test_gate_holds_when_never_worse():
    assert gate_holds([_row(0.5, 0.5), _row(0.5, 0.7), _row(0.5, 0.9)])


def test_gate_catches_a_dip_at_any_budget():
    """守门失效的形态是曲线上某一点低于零样本 —— 哪怕别的点都更好。"""
    assert not gate_holds([_row(0.5, 0.7), _row(0.5, 0.42), _row(0.5, 0.9)])


def test_gate_ignores_nan_rows():
    assert gate_holds([_row(float("nan"), float("nan")), _row(0.5, 0.6)])


# --- ctx_vec 的恒等性（守门基线的前提）------------------------------------

def test_zero_ctx_is_bitwise_identical_to_no_ctx():
    """`ctx_vec=0` 必须与「不传 ctx」逐位相同。

    这是守门能成立的前提：零样本基线与适配模型的起点必须是**同一个模型**，
    否则「适配后不劣于零样本」比的是两个不同的东西。
    """
    c0 = Context(desc=None, stat_rel=None, type_id=None, W=16, site_ctx_dim=8)
    c1 = dataclasses.replace(c0, ctx_vec=torch.zeros(8))
    assert torch.equal(c0.ctx(5, "cpu"), c1.ctx(5, "cpu"))


def test_ctx_vec_broadcasts_to_batch():
    c = Context(desc=None, stat_rel=None, type_id=None, W=16, site_ctx_dim=8,
                ctx_vec=torch.arange(8.0))
    out = c.ctx(4, "cpu")
    assert out.shape == (4, 8)
    assert torch.equal(out[0], out[3])


def test_film_is_identity_at_zero_ctx():
    """FiLM 末层零初始化 -> ctx=0 时 gamma=1、beta=0。

    这条塌了，`ctx_vec=0 ≡ 零样本` 就不成立，守门的基线也就不成立。
    """
    from physwm.model.layers import FiLM
    f = FiLM(8, 16)
    x = torch.randn(3, 5, 16)
    assert torch.allclose(f(x, torch.zeros(3, 8)), x, atol=1e-7)


# --- 适配机制与梯度存活守卫 ----------------------------------------------

def _fake_site():
    """能走到冻结之后、在 `GPUWindows` 处抛错的最小 site。

    关键是它必须**通过** `site.obs_loss.w` 那一步 —— 否则 `adapt` 在冻结
    之前就抛错，下面两条测试会**空过**（第一版正是这么假过一次）。
    """
    from physwm.train.losses import LossWeights
    s = type("S", (), {})()
    s.ctx = Context(desc=None, stat_rel=None, type_id=None, W=16, site_ctx_dim=8)
    s.obs_loss = type("O", (), {"w": LossWeights()})()
    s.sd = s.spec = s.norm = s.scales = None
    return s


def _tiny_model():
    from physwm.model.world_model import ModelConfig, WorldModel
    return WorldModel(ModelConfig(), {"chiller": 2, "tower": 2,
                                      "coolpump": 1, "coldpump": 1}, 9)


def test_fake_site_really_reaches_the_freeze():
    """守住上面那条注释：假 site 必须能走过 `obs_loss.w`。

    若哪天 `adapt` 在冻结前多读一个 site 字段，这条会红 —— 提醒去补
    `_fake_site`，而不是让另外两条测试悄悄变成空过。
    """
    from physwm.train.losses import LossWeights
    assert isinstance(_fake_site().obs_loss.w, LossWeights)


def test_adapt_freezes_the_backbone():
    """主干必须一位不动（`film_bias` 那几个偏置除外，且它们会被还原）。"""
    from physwm.train.fewshot import adapt
    m = _tiny_model()
    before = {k: v.clone() for k, v in m.state_dict().items()}
    with pytest.raises(Exception):
        adapt(m, _fake_site(), FewShotConfig(steps=1), torch.device("cpu"),
              windows=None)
    after = m.state_dict()
    bad = [k for k in before if not torch.equal(before[k], after[k])]
    assert not bad, f"这些权重在适配中被改了：{bad[:5]}"


def test_requires_grad_restored_even_when_adapt_raises():
    """适配中途抛错也必须恢复 `requires_grad`。

    不恢复的话模型停在「全部冻结」，之后的训练**静默地什么都不更新** ——
    损失照样在降，只是主干不动，极难查。
    """
    from physwm.train.fewshot import adapt
    m = _tiny_model()
    with pytest.raises(Exception):
        adapt(m, _fake_site(), FewShotConfig(steps=1), torch.device("cpu"),
              windows=None)
    off = [n for n, p in m.named_parameters() if not p.requires_grad]
    assert not off, f"这些参数的 requires_grad 没恢复：{off[:5]}"


def test_rejects_unknown_mechanism():
    with pytest.raises(ValueError, match="未知的适配机制"):
        FewShotConfig(mechanism="lora_everything")


def test_film_bias_names_are_the_live_ones():
    """只取 FiLM 的输出偏置 —— `.0.weight` 已 denormal，`.2.weight` 太大。"""
    from physwm.train.fewshot import film_bias_names
    names = film_bias_names(_tiny_model())
    assert names, "一个都没找到"
    assert all(n.endswith(".2.bias") and ".film." in n for n in names)
    assert not any(".0.weight" in n for n in names)


# --- 梯度存活守卫（§13 #54 的绊索）----------------------------------------

def test_gradient_guard_fires_when_all_params_are_dead():
    """全部被调参数梯度为零 -> 必须报错，而不是空转 300 步。

    实测踩过：`site_ctx` 机制训 30 步 `ctx_norm` 恒为 0.000，两臂 R² 逐位
    相同（−64.4886），而损失照样随 batch 抖动 —— 训练日志、损失曲线、指标
    三者都看不出来。这条守卫把它变成第一步就报错。
    """
    from physwm.train.fewshot import _assert_gradient_alive
    p = torch.zeros(3, requires_grad=True)
    p.grad = torch.zeros(3)
    with pytest.raises(RuntimeError, match="梯度为零"):
        _assert_gradient_alive(["site_ctx"], [p], "site_ctx")


def test_gradient_guard_names_the_film_cause_for_site_ctx():
    """报错信息要直接给出根因与出路，不能只说「梯度为零」。"""
    from physwm.train.fewshot import _assert_gradient_alive
    p = torch.zeros(3, requires_grad=True)
    p.grad = torch.zeros(3)
    with pytest.raises(RuntimeError, match="film_bias"):
        _assert_gradient_alive(["site_ctx"], [p], "site_ctx")


def test_gradient_guard_passes_when_any_param_is_alive():
    """部分参数没梯度是正常的（比如某族不存在），全死才是空转。"""
    from physwm.train.fewshot import _assert_gradient_alive
    dead = torch.zeros(3, requires_grad=True); dead.grad = torch.zeros(3)
    live = torch.zeros(3, requires_grad=True); live.grad = torch.ones(3)
    _assert_gradient_alive(["a", "b"], [dead, live], "film_bias")


def test_gradient_guard_treats_none_grad_as_dead():
    from physwm.train.fewshot import _assert_gradient_alive
    p = torch.zeros(3, requires_grad=True)          # grad is None
    with pytest.raises(RuntimeError):
        _assert_gradient_alive(["x"], [p], "film_bias")


# --- 适配的安装 / 还原 ----------------------------------------------------

def test_adaptation_install_restores_pretrained_values():
    """守门判负时必须把权重原样还原 —— 否则「保留预训练权重」是句空话。"""
    from physwm.train.fewshot import Adaptation
    m = _tiny_model()
    name = next(n for n, _ in m.named_parameters() if n.endswith(".2.bias"))
    base = dict(m.named_parameters())[name].detach().clone()
    ad = Adaptation(base={name: base}, params={name: base + 1.0})

    ad.install(m, use_adapted=True)
    assert torch.allclose(dict(m.named_parameters())[name], base + 1.0)
    ad.install(m, use_adapted=False)
    assert torch.equal(dict(m.named_parameters())[name], base)


def test_ctx_for_returns_none_vec_when_not_adopted():
    from physwm.train.fewshot import Adaptation
    c = Context(desc=None, stat_rel=None, type_id=None, W=16, site_ctx_dim=8)
    ad = Adaptation(ctx_vec=torch.ones(8))
    assert ad.ctx_for(c, False).ctx_vec is None
    assert ad.ctx_for(c, True).ctx_vec is not None


# --- 守门判据用 MAE 而非 R²（实测踩过）------------------------------------

def test_gate_uses_mae_not_r2():
    """守门段真值方差近似为 0 时，R² 会变成天文数字，不能用来比大小。

    实测：hx 1 天那点的守门段只有 20 个连续窗口、跨几小时，
    零样本 R² = **−1,625,593.77**、适配 R² = −142,874.17。
    两个数都没有意义，而按 R² 比大小会判「适配更好」并采纳 ——
    test 随之从 −2.30 掉到 −5.11。MAE 与真值方差无关，是这里唯一站得住的统计量。
    """
    src = (ROOT / "physwm" / "train" / "fewshot.py").read_text(encoding="utf-8")
    body = src.split("def gated_adapt")[1]
    decide = body.split("better = bool(")[1].split(")\n")[0]
    assert "MAE" in decide, "守门判据没在用 MAE"
    assert '["R2"]' not in decide, "守门判据还在用 R²"


def test_small_gate_never_adopts():
    """守门段太短就一律保留预训练权重 —— 这正是 M2 说的「数据不足」。"""
    cfg = FewShotConfig(min_gate_windows=64)
    assert cfg.min_gate_windows == 64
    # 判据本身：窗口数不足时 enough 为假，无论 MAE 多好都不采纳
    for n_gate, mae_z, mae_a in [(20, 1000.0, 1.0), (63, 1000.0, 1.0)]:
        enough = n_gate >= cfg.min_gate_windows
        better = enough and mae_a < mae_z * (1 - cfg.gate_margin)
        assert not better, f"{n_gate} 个窗口就采纳了"


def test_margin_rejects_marginal_wins():
    """守门段短、自相关强，小幅优势多半是噪声，必须留余量。"""
    cfg = FewShotConfig(min_gate_windows=8, gate_margin=0.05)
    enough = True
    for mae_z, mae_a, want in [(100.0, 99.0, False),    # 只赢 1%
                               (100.0, 96.0, False),    # 赢 4%，仍不够
                               (100.0, 90.0, True)]:    # 赢 10%
        better = enough and mae_a < mae_z * (1 - cfg.gate_margin)
        assert better is want, f"MAE {mae_z}->{mae_a} 判定错了"
