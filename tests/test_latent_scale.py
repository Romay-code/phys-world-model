"""隐状态尺度的不变量。

单独成文件是因为这一组测的是**训练动力学的前提条件**，不是单次前向的正确性：
z 的尺度无界会让 head 输入漂进 softplus/sigmoid 的饱和区，表现为「指标停止改善」
而非报错，只有看 |z| 曲线才发现。实测过一次：|z| 1.22 -> 10.09 -> 撞墙回弹 1.86。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from physwm.data import schema as S  # noqa: E402
from physwm.model.world_model import ModelConfig, WorldModel  # noqa: E402
from physwm.train.losses import LossWeights, latent_consistency  # noqa: E402

N_DEV = {"chiller": 7, "tower": 20, "coolpump": 7, "coldpump": 8}
N = 1 + sum(N_DEV.values())
F = 9
W = 16


def _model():
    torch.manual_seed(0)
    return WorldModel(ModelConfig(), N_DEV, f_max=F)


def _ctx(B):
    return dict(desc=torch.randn(N, 32), stat_rel=torch.randn(N, N, 5),
                type_id=torch.from_numpy(
                    __import__("numpy").array(
                        [0] + [1] * 7 + [2] * 20 + [3] * 7 + [4] * 8, dtype="int64")),
                site_ctx=torch.zeros(B, 8))


@pytest.mark.parametrize("input_scale", [0.1, 1.0, 50.0])
def test_encoder_output_scale_is_bounded(input_scale):
    """输入放大 500 倍，编码器输出的尺度必须基本不变。

    没有输出归一化时 |z| 会跟着输入一起漂，训练中再叠加 `pool` 权重的自由增长，
    就是指数发散。
    """
    m = _model().eval()
    B = 4
    x = torch.randn(B, W, N, F) * input_scale
    av = torch.ones(B, W, N, F)
    with torch.no_grad():
        z, _ = m.encode(x, av, **_ctx(B))
    rms = float((z ** 2).mean().sqrt())
    assert 0.5 < rms < 2.0, f"|z| RMS = {rms:.3f}（input_scale={input_scale}）超出受控范围"


def test_latent_loss_is_scale_invariant():
    """lat 项必须对 z 的整体缩放不变 —— 否则「把 z 缩小」是降 lat 的免费午餐。

    这正是塌缩的成因：lat 先涨到 38.76 再塌到 0.0024，head_w 跟着死掉，
    逐台冷机功率变成常数。
    """
    torch.manual_seed(0)
    z = torch.randn(4, 3, N, 192)
    zt = torch.randn(4, 3, N, 192)

    def lat_of(scale: float) -> float:
        # 必须调真实实现，不能在测试里重抄一遍公式 —— 抄一遍的话改坏了
        # losses.py 测试也不会红（变异测试实测漏检过）。
        return float(latent_consistency(z * scale, zt * scale)[0])

    base = lat_of(1.0)
    for s in (0.01, 0.1, 10.0, 100.0):
        assert abs(lat_of(s) - base) < 1e-4 * max(base, 1.0), \
            f"lat 随尺度变化（scale={s}）—— 缩小 z 就能降低损失"


def test_rollout_latent_does_not_blow_up():
    """H 步纯想象推演后 |z| 不得爆炸。

    转移是 z' = z + delta*G(z)，|z_H| 最多线性增长；若出现数量级跳变，
    说明 G 的输出没有被有效约束，H=48 必然发散。
    """
    m = _model().eval()
    B, H = 2, 24
    batch = {
        "seq_x": torch.randn(B, W + H, N, F),
        "seq_avail": torch.ones(B, W + H, N, F),
        "seq_raw": torch.rand(B, W + H, N, F) * 10 + 5,
    }
    c = _ctx(B)
    with torch.no_grad():
        out = m.rollout(batch, H=H, W=W, **c)
    zt = out["z"]                                   # [B,H,N,d]
    rms = (zt ** 2).mean(dim=(0, 2, 3)).sqrt()      # 每步的 RMS
    assert float(rms.max()) < 10.0 * float(rms[0]), \
        f"rollout 中 |z| 放大了 {float(rms.max()/rms[0]):.1f} 倍"
    assert torch.isfinite(zt).all()


def test_heads_stay_out_of_saturation_at_realistic_z():
    """z 在受控尺度下，各 head 的输出不应贴死在边界上。

    贴边界 = 梯度消失 = 静默塌缩。判据用相对标准差，恒定输出即为塌缩。
    """
    m = _model().eval()
    B = 16
    z = {f: torch.randn(B, N_DEV.get(f, 1), 192) for f in S.FAMILIES}
    z["plant"] = torch.randn(B, 1, 192)
    a = torch.rand(B, 7, 20)
    a = a / a.sum(-1, keepdim=True)
    phys = {"wet_bulb": torch.rand(B) * 30,
            "cold_out": 5 + torch.rand(B, 7) * 8,
            "on_ch": torch.ones(B, 7)}
    for f in ("tower", "coolpump", "coldpump"):
        phys[f"on_{f}"] = torch.ones(B, N_DEV[f])
        phys[f"freq_{f}"] = 20 + torch.rand(B, N_DEV[f]) * 35
    with torch.no_grad():
        o = m.decoder(z, a, phys)
    for k in ("w_chiller", "eta", "approx", "cool_dt"):
        v = o[k]
        rsd = float(v.std() / v.abs().mean().clamp_min(1e-9))
        assert rsd > 1e-3, f"{k} 输出近乎常数（rsd={rsd:.2e}）—— head 已塌缩"


def test_default_loss_weights_enable_lat_normalisation():
    """回归保护：lam_lat 默认开启，且默认关闭昂贵且恒为 0 的 lam_lip。"""
    w = LossWeights()
    assert w.lam_lat > 0
    assert w.lam_lip == 0.0


def test_lipschitz_penalty_actually_detects_violation():
    """惩罚项必须**真的**检出超界，否则调大 lam_lip 等于什么都没做。

    实测踩过（§13 #34）：真实谱范数乘积 1.12（超界 12%）时，
    n_iter=2 的幂迭代仍返回 0.000e+00，n_iter=20 才报出来。
    一个恒返回 0 的惩罚项，与「没有惩罚项」在训练日志里完全一样。
    """
    import torch
    from physwm.model.world_model import ModelConfig, WorldModel
    from physwm.train.losses import lipschitz_penalty

    torch.manual_seed(0)
    m = WorldModel(ModelConfig(), {"chiller": 7, "tower": 20,
                                   "coolpump": 7, "coldpump": 8}, 9)
    with torch.no_grad():
        for p in m.transition.parameters():
            if p.dim() > 1:
                p.add_(torch.randn_like(p) * 0.05)
    v = float(lipschitz_penalty(m.transition, 1.0))
    assert v > 1e-6, (
        f"惩罚项对一个确实超界的 transition 返回 {v:.3e} —— "
        f"幂迭代步数不足，该项在训练中恒为 0，调大 lam_lip 不会有任何效果")
