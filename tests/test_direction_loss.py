"""训练期方向损失的构造正确性（physwm/train/direction_loss.py）。

测的是这个惩罚项**有没有惩罚对东西**，不是模型学得好不好：
  - 反向响应必须被罚，正向响应不能被罚
  - 惩罚必须能回传到模型参数（否则加了等于没加）
  - 归一化分母必须 detach（否则「把预测整体放大」是降低本项的免费路径）
  - 结构恒等的三个功率目标不该进训练损失（浪费算力）
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from physwm.train.direction_loss import direction_penalty, train_specs  # noqa: E402
from physwm.data import schema as S  # noqa: E402

N_DEV = {"chiller": 7, "tower": 20, "coolpump": 7, "coldpump": 8}
N = 1 + sum(N_DEV.values())
F, W, B = 9, 16, 8


class _Sch:
    token_index = ([("plant", 0)] + [("chiller", i) for i in range(7)]
                   + [("tower", i) for i in range(20)]
                   + [("coolpump", i) for i in range(7)]
                   + [("coldpump", i) for i in range(8)])


def _ctx():
    from physwm.train.loop import Context
    return Context(desc=torch.randn(N, 32), stat_rel=torch.randn(N, N, 5),
                   type_id=torch.tensor([0] + [1] * 7 + [2] * 20 + [3] * 7 + [4] * 8),
                   W=W, site_ctx_dim=8)


def _batch():
    raw = torch.rand(B, W + 1, N, F) * 10 + 5
    for fam in ("tower", "chiller", "coolpump", "coldpump"):
        rows = [i for i, (f, _) in enumerate(_Sch.token_index) if f == fam]
        raw[:, :, rows, S.CANON_FIELDS[fam].index("on")] = 1.0
    return {"seq_x": torch.randn(B, W + 1, N, F),
            "seq_avail": torch.ones(B, W + 1, N, F),
            "seq_raw": raw,
            "P_plant": torch.rand(B, 1) * 1000}


class _StubModel(torch.nn.Module):
    """响应符号可控的替身模型。

    **必须走真实的 `direction_penalty`**，不能在测试里重抄一遍 hinge 公式 ——
    P2 记录 §6 第 3 条已经踩过：`lat` 的测试重抄公式，改坏 losses.py 也不红，
    测试逻辑正确、也能通过，但等于什么都没测。变异测试是唯一能发现它的手段。
    （本文件第一版就又犯了一次：把符号取反，5 条测试全绿。）
    """

    def __init__(self, gain: float):
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor(float(gain)))
        rows = [i for i, (f, _) in enumerate(_Sch.token_index) if f == "tower"]
        self.rows = rows
        self.k = S.CANON_FIELDS["tower"].index("frequency")

    def rollout(self, batch, **kw):
        W_ = kw["W"]
        fr = batch["seq_raw"][:, W_, self.rows, self.k].mean(-1, keepdim=True)
        # 基础离散度取自**未被扰动**的 plant token，保证 base/perturbed 一致；
        # 响应项是 w·fr。于是 gain 改变的是「响应相对于离散度」的比值 ——
        # 归一化后的违例确实随 gain 增长。
        # （若写成 approx = w·fr，std 也跟着 w 缩放，归一化违例与 w 无关 ——
        #   那正是本项对整体缩放不变的设计性质，不是 bug。）
        spread = batch["seq_raw"][:, W_, :1, :1].reshape(-1, 1) * 3.0
        approx = spread.expand(-1, 20) + self.w * fr.expand(-1, 20)
        cool = spread.expand(-1, 7) + self.w * fr.expand(-1, 7)
        return {"preds": [{"approx": approx, "cool_out": cool}]}


def _tower_only_spec():
    from physwm.eval.direction import DirSpec
    return [DirSpec("塔频+1Hz", "tower", "frequency", 1.0,
                    {"approx": -1}, why="测试用", on_only=False)]


def test_correct_direction_gives_zero_penalty():
    """gain<0：塔频↑ -> approx↓，与先验 -1 一致，惩罚必须为 0。"""
    pen, _ = direction_penalty(_StubModel(-1.0), _batch(), _ctx(), W, _Sch,
                               torch.ones(N, F), specs=_tower_only_spec(), sub=1.0)
    assert float(pen) == 0.0, f"正确方向被罚了 {float(pen)}"


def test_wrong_direction_is_penalised():
    """gain>0：塔频↑ -> approx↑，与先验相反，惩罚必须 > 0。

    这条是整个模块的核心断言。第一版测试重抄公式，把实现里的符号取反
    竟然全绿 —— 现在走真实函数，取反必红。
    """
    pen, logs = direction_penalty(_StubModel(+1.0), _batch(), _ctx(), W, _Sch,
                                  torch.ones(N, F), specs=_tower_only_spec(), sub=1.0)
    assert float(pen) > 0.0, "反向响应没有被惩罚 —— hinge 的符号可能反了"
    assert logs.get("dir_approx@1", 0.0) > 0.0, "逐目标违例率日志没记上"


def test_zero_response_is_penalised():
    """**没有响应**必须被惩罚 —— 这是加正 margin 的全部目的。

    无 margin 的平方 hinge 其最优解是 d=0（惩罚与梯度同时为零），
    而 dir_vr 在 d=0 附近数符号近乎掷硬币。实测确认过：
    塔频->逼近度 h=36 的违例 |d| 已被压到合规值的 1/12、只占平方和 0.1%，
    dir_vr 仍有 0.212（tools/diag_direction_gap.py）。
    """
    pen, _ = direction_penalty(_StubModel(0.0), _batch(), _ctx(), W, _Sch,
                               torch.ones(N, F), specs=_tower_only_spec(), sub=1.0)
    assert float(pen) > 0.0, "零响应没有被惩罚 —— margin 没生效，驻点仍在 d=0"


def test_margin_zero_reproduces_the_stationary_point_bug():
    """margin=0 时零响应不被罚 —— 保留这条以证明 margin 确实是那个修复点。

    若哪天有人把 margin 默认改回 0，上一条测试会红，而这条会绿，
    两条一起指明「改动改的正是这里」。
    """
    pen, _ = direction_penalty(_StubModel(0.0), _batch(), _ctx(), W, _Sch,
                               torch.ones(N, F), specs=_tower_only_spec(),
                               sub=1.0, margin=0.0)
    assert float(pen) == 0.0


def test_penalty_backpropagates_to_model():
    """惩罚必须真的能改模型 —— 否则这一项加了等于没加。"""
    from physwm.model.world_model import ModelConfig, WorldModel
    torch.manual_seed(0)
    m = WorldModel(ModelConfig(), N_DEV, F)
    pen, logs = direction_penalty(m, _batch(), _ctx(), W, _Sch,
                                  torch.ones(N, F), sub=1.0)
    assert pen.requires_grad, "惩罚项与计算图断开了"
    pen.backward()
    g = sum(float(p.grad.abs().sum()) for p in m.parameters() if p.grad is not None)
    assert g > 0, "方向惩罚没有回传到任何参数"
    assert logs, "没有产出任何逐目标日志"


def test_structural_power_targets_excluded_from_training_specs():
    """`w_*_dev` 在解码器里是 `on·(fr/50)^3·k` 的恒等式，dir_vr 实测恒为 0。

    把它们放进训练损失是纯浪费 —— 而且会用一个必然为 0 的项稀释
    真正在学的那几项的权重。
    """
    tgts = {t for s in train_specs() for t in s.targets}
    for k in ("w_tower_dev", "w_coolpump_dev", "w_coldpump_dev"):
        assert k not in tgts, f"{k} 是结构恒等式，不该进训练损失"
    assert "approx" in tgts and "P_plant" in tgts


def test_normaliser_is_detached():
    """归一化分母必须 detach。

    否则「把该目标的预测整体放大」会同时放大分母、缩小归一化违例 ——
    降低本项的免费路径，与 §4.5.2 隐一致性塌缩是同一个陷阱。
    """
    src = (ROOT / "physwm" / "train" / "direction_loss.py").read_text(encoding="utf-8")
    assert "raw_d.detach().abs().median()" in src, "归一化分母没有 detach"


def test_correct_but_undersized_response_is_penalised_less_than_wrong_sign():
    """方向正确但幅度不足 -> 罚一点；方向错误 -> 罚更多。

    margin 版的判据从「幅度」转为「是否达到 margin 倍的典型正向响应」，
    故排序必须是：足量正向 0 < 不足量正向 < 反向。
    """
    torch.manual_seed(0); bt = _batch(); cx = _ctx()
    def pen(gain):
        return float(direction_penalty(_StubModel(gain), bt, cx, W, _Sch,
                                       torch.ones(N, F), specs=_tower_only_spec(),
                                       sub=1.0)[0])
    # gain<0 = 正确方向（塔频↑ -> approx↓）；gain>0 = 反向
    assert pen(-1.0) == 0.0, "足量正向响应被罚了"
    assert pen(0.0) > 0.0, "零响应未被罚"
    assert pen(+1.0) > pen(0.0), "反向响应的惩罚应大于零响应"
