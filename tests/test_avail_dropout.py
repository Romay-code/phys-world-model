"""输入可用性丢弃（P5-B）。

这条机制修的是 P5-A 的零样本失败：pc3 缺五条负荷类通道，
模型训练时从没见过它们缺，于是推不出量级、输出塌成近似常数
（实测 R²=−1.64、corr 0.883、预测 std 只有真值的 0.26）。

下面每条测试都对应一个「错了不会报错、只会让结论失真」的失效形态。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from physwm.data import schema as S  # noqa: E402
from physwm.train.avail_dropout import (  # noqa: E402
    DROPPABLE, PHYSICAL_INPUTS, AvailDropout)


class _Sch:
    """最小 schema 替身：只需要 token_index。yb3 规模。"""

    n_dev = {"chiller": 7, "tower": 16, "coolpump": 7, "coldpump": 7}

    @property
    def token_index(self):
        idx = [("plant", 0)]
        for fam in S.DEVICE_FAMILIES:
            idx.extend((fam, d) for d in range(self.n_dev[fam]))
        return idx


def _batch(N=38, F=9, B=4, W=19):
    return {"seq_x": torch.randn(B, W, N, F),
            "seq_avail": torch.ones(B, W, N, F),
            "seq_raw": torch.rand(B, W, N, F) * 10 + 5}


def test_droppable_excludes_physical_inputs():
    """可掩清单绝不能碰物理解码器要读的通道。

    掩掉 `cold_out_temp` 会让卡诺 COP 拿 0 K 去算 —— 不报错，只出坏数。
    `WorldModel.physical_inputs` 若哪天多读一个字段，这里要跟着更新，
    否则这条测试就是这个改动的第一个绊索。
    """
    assert not (set(DROPPABLE) & PHYSICAL_INPUTS)


def test_physical_inputs_list_matches_model():
    """`PHYSICAL_INPUTS` 必须与 `WorldModel.physical_inputs` 实际读的字段一致。

    两份清单分处两个文件，漂移了不会有任何报错 —— 与 §13 #32 同类。
    """
    src = (ROOT / "physwm" / "model" / "world_model.py").read_text(encoding="utf-8")
    # 按**方法级**缩进切，不能用裸 "def " —— physical_inputs 内部还有
    # 一个嵌套的 def fld，会把 body 提前截断成空壳，让这条测试永远通过
    body = src.split("def physical_inputs")[1].split("\n    def ")[0]
    for fam, fld in PHYSICAL_INPUTS:
        assert f'"{fld}"' in body, f"{fam}.{fld} 已不在 physical_inputs 里，清单需更新"


def test_rejects_channel_that_feeds_decoder():
    with pytest.raises(ValueError, match="物理解码器"):
        AvailDropout(channels=(("chiller", "cold_out_temp"),))


@pytest.mark.parametrize("bad", [-0.1, 1.5])
def test_rejects_out_of_range_probability(bad):
    with pytest.raises(ValueError):
        AvailDropout(p_joint=bad)


def test_disabled_is_a_true_noop():
    """两个概率都为 0 时必须逐位不动 —— 否则 P5-A 与后续臂不可比。"""
    d = AvailDropout(p_joint=0.0, p_each=0.0)
    assert not d.enabled
    b = _batch()
    out, hit = d.apply(b, _Sch())
    assert hit == []
    assert out is b


def test_masked_channel_is_zero_in_all_three_tensors():
    """x / raw / avail 必须一起置零。

    只改其中一个，编码器与物理解码器会看到不同的世界 —— 与 §5.3
    `counterfactual` 的第 2 条约束同一个坑。
    """
    d = AvailDropout(p_joint=1.0, p_each=0.0)
    sch = _Sch()
    b = _batch()
    out, hit = d.apply(b, sch)
    assert set(hit) == set(DROPPABLE)
    for fam, fld in DROPPABLE:
        rows = [i for i, (f, _) in enumerate(sch.token_index) if f == fam]
        k = S.CANON_FIELDS[fam].index(fld)
        for key in ("seq_x", "seq_avail", "seq_raw"):
            assert torch.all(out[key][:, :, rows, k] == 0), f"{key} {fam}.{fld} 未置零"


def test_untouched_channels_are_bit_identical():
    """没被抽中的通道一位都不许动。"""
    d = AvailDropout(p_joint=1.0, p_each=0.0)
    sch = _Sch()
    b = _batch()
    out, _ = d.apply(b, sch)
    drop = {(fam, S.CANON_FIELDS[fam].index(fld)) for fam, fld in DROPPABLE}
    for fam in S.FAMILIES:
        rows = [i for i, (f, _) in enumerate(sch.token_index) if f == fam]
        for k, fld in enumerate(S.CANON_FIELDS[fam]):
            if (fam, k) in drop:
                continue
            for key in ("seq_x", "seq_avail", "seq_raw"):
                assert torch.equal(out[key][:, :, rows, k], b[key][:, :, rows, k]), \
                    f"{key} {fam}.{fld} 被误伤"


def test_input_batch_is_not_mutated():
    """必须返回副本。就地改会污染 `GPUWindows` 的缓存张量。"""
    d = AvailDropout(p_joint=1.0, p_each=0.0)
    b = _batch()
    ref = {k: v.clone() for k, v in b.items()}
    d.apply(b, _Sch())
    for k in ref:
        assert torch.equal(b[k], ref[k]), f"{k} 被就地修改"


def test_joint_case_is_actually_sampled():
    """`p_joint` 存在的理由：五条同时缺的概率在独立采样下是 p⁵，
    几乎永远采不到，而 pc3 恰恰就是那个站。"""
    d = AvailDropout(p_joint=0.15, p_each=0.20)
    g = torch.Generator().manual_seed(0)
    n_all = sum(len(d.pick(g)) == len(DROPPABLE) for _ in range(4000))
    assert 0.10 < n_all / 4000 < 0.22, f"整组掩比例 {n_all/4000:.3f} 偏离 p_joint"


def test_each_channel_gets_dropped_sometimes():
    """任何一条都不能因为索引写错而永远掩不到。"""
    d = AvailDropout(p_joint=0.0, p_each=0.5)
    g = torch.Generator().manual_seed(1)
    seen = set()
    for _ in range(500):
        seen.update(d.pick(g))
    assert seen == set(DROPPABLE), f"这些通道从未被抽中：{set(DROPPABLE) - seen}"
