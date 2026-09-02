"""M2 核心性质：一套权重吃任意台数的站点。

P0–P4 全程只用 yb3 一站，这条性质从未被验证过。实测（2026-08-25）：
整个 3.58e6 参数里只有 `log_mcp_raw` / `log_mcp_cool_raw` 两个 `(n_chiller,)`
张量阻塞迁移 —— 其余全部形状无关，置换等变的 token 架构按设计工作。
改为由设备描述符生成后，形状差异归零。

**这两条测试必须一直绿**，否则 P5 的跨站预训练在结构上就不成立。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from physwm.model.world_model import ModelConfig, WorldModel  # noqa: E402

# 取自实测的两个真实站点规模（tools/check_sites.py）：yb3 N=43、tx N=21
YB3 = {"chiller": 7, "tower": 20, "coolpump": 7, "coldpump": 8}
TX = {"chiller": 4, "tower": 8, "coolpump": 4, "coldpump": 4}
ZX = {"chiller": 11, "tower": 16, "coolpump": 11, "coldpump": 11}   # N=50，最大


def _mk(n_dev):
    return WorldModel(ModelConfig(), n_dev, 9)


def _tid(n_dev):
    return torch.tensor([0] + [1] * n_dev["chiller"] + [2] * n_dev["tower"]
                        + [3] * n_dev["coolpump"] + [4] * n_dev["coldpump"])


@pytest.mark.parametrize("dst", [TX, ZX])
def test_weights_transfer_across_device_counts(dst):
    """yb3 训出的权重必须能原样装进别的站点的模型。

    若哪天有人再引入 per-device 的可训练参数，这里会以
    「size mismatch」直接报出来 —— 那正是 M2 失效的形态。
    """
    src = _mk(YB3).state_dict()
    m = _mk(dst)
    m.load_state_dict(src)          # 不该抛异常


def test_no_parameter_depends_on_device_count():
    """更严的版本：逐张量比形状，任何差异都点名。"""
    a, b = _mk(YB3).state_dict(), _mk(ZX).state_dict()
    bad = [(k, tuple(a[k].shape), tuple(b[k].shape))
           for k in a if k in b and a[k].shape != b[k].shape]
    missing = [k for k in a if k not in b] + [k for k in b if k not in a]
    assert not bad, f"这些张量依赖设备台数，阻塞跨站迁移：{bad}"
    assert not missing, f"两站的键不一致：{missing}"


@pytest.mark.parametrize("n_dev", [YB3, TX, ZX])
def test_rollout_runs_at_every_site_scale(n_dev):
    """装完权重要真能跑，不只是形状对得上。"""
    n = 1 + sum(n_dev.values())
    m = _mk(n_dev)
    m.load_state_dict(_mk(YB3).state_dict())
    W, H, B = 16, 3, 2
    out = m.rollout({"seq_x": torch.randn(B, W + H, n, 9),
                     "seq_avail": torch.ones(B, W + H, n, 9),
                     "seq_raw": torch.rand(B, W + H, n, 9) * 10 + 5},
                    desc=torch.randn(n, 32), stat_rel=torch.randn(n, n, 5),
                    type_id=_tid(n_dev), site_ctx=torch.zeros(B, 8), H=H, W=W)
    p = out["preds"][0]
    assert p["P_plant"].shape == (B,)
    assert p["mcp"].shape == (B, n_dev["chiller"])
    assert torch.isfinite(p["P_plant"]).all()


def test_mcp_comes_from_descriptor_not_free_parameter():
    """m·cp 必须由 desc 生成。

    自由参数版本不可迁移；更隐蔽的是，它会让「未见过的站点」没有
    任何办法得到 mcp —— 而 P5 的零样本正是要处理这种情况。
    """
    m = _mk(YB3)
    names = {n for n, _ in m.decoder.named_parameters()}
    assert "log_mcp_raw" not in names, "per-chiller 的 mcp 参数回来了，M2 迁移失效"
    assert "log_mcp_cool_raw" not in names
    assert any(n.startswith("mcp_net") for n in names), "mcp_net 不存在"

    # 不同的 desc 必须给出不同的 mcp，否则这个网络是摆设
    z = {f: torch.randn(2, YB3.get(f, 1), m.cfg.d) for f in ("chiller", "tower",
                                                             "coolpump", "coldpump")}
    z["plant"] = torch.randn(2, 1, m.cfg.d)
    with torch.no_grad():
        for lin in m.decoder.mcp_net:
            if isinstance(lin, torch.nn.Linear):
                lin.weight.normal_(0, 0.5)
                lin.bias.normal_(0, 0.5)
    a = m.decoder._mcp_raw(torch.randn(7, 32), z["chiller"])[0]
    b = m.decoder._mcp_raw(torch.randn(7, 32), z["chiller"])[0]
    assert not torch.allclose(a, b), "不同 desc 给出相同 mcp —— mcp_net 没在用 desc"


# ---------------------------------------------------------------------------
# 量纲必须是**站点属性**，不能是模型状态。
# 实测 14 站：w_scale 跨 6.9 倍（178.5 tx ~ 1229.0 yb_低温）、k_mcp 跨 38.5 倍、
# q_scale 跨 22.2 倍。单一全局 buffer 服务不了多站。
# ---------------------------------------------------------------------------

def _run(m, n_dev, scales=None):
    n = 1 + sum(n_dev.values())
    return m.rollout({"seq_x": torch.zeros(2, 19, n, 9),
                      "seq_avail": torch.ones(2, 19, n, 9),
                      "seq_raw": torch.rand(2, 19, n, 9) * 10 + 5},
                     desc=torch.zeros(n, 32), stat_rel=torch.zeros(n, n, 5),
                     type_id=_tid(n_dev), site_ctx=torch.zeros(2, 8),
                     H=3, W=16, scales=scales)["preds"][0]


def test_scales_override_changes_output_magnitude():
    """传入别站的量纲，输出量级必须跟着变。

    若 override 没生效，多站训练会用 yb3 的量纲去解释所有站点 ——
    而 tx 的单台冷机功率只有 yb3 的 1/5，这个错误不会报任何异常，
    只会让 tx 的预测系统性偏大 5 倍。
    """
    torch.manual_seed(0)
    m = _mk(YB3)
    m.decoder.set_scales(w_scale=952.0)
    a = _run(m, YB3)["w_chiller"].median()
    b = _run(m, YB3, scales={"w_scale": 178.5})["w_chiller"].median()
    assert float(a) > float(b) * 3, (
        f"override 未生效：w_scale 952 -> 178.5 时输出仅从 {float(a):.1f} 变到 {float(b):.1f}")


def test_scales_are_buffers_not_parameters():
    """量纲不可训练。

    它是量纲换算，不该被梯度改；更要紧的是它一旦可训就变成了模型状态，
    跨站迁移时会把源站的量纲带过去。
    """
    m = _mk(YB3)
    names = {n for n, _ in m.named_parameters()}
    for k in ("w_scale", "q_scale", "dt_scale", "dt_evap_scale", "approx_scale",
              "mcp_scale", "mcp_cool_scale", "k_mcp",
              "p_scale_tower", "p_scale_coolpump", "p_scale_coldpump"):
        assert f"decoder.{k}" not in names, f"{k} 变成了可训练参数"


def test_omitting_scales_keeps_single_site_behaviour():
    """不传 scales 时必须与传 buffer 里那套值完全一致。

    单站路径（P0–P4）依赖这条 —— 若默认路径悄悄变了，
    之前所有实验结果就不可复现，而这件事不会有任何报错。
    """
    torch.manual_seed(0)
    m = _mk(YB3)
    m.decoder.set_scales(w_scale=952.0, dt_scale=5.0, k_mcp=33.5)
    torch.manual_seed(1)
    a = _run(m, YB3)
    torch.manual_seed(1)
    b = _run(m, YB3, scales={"w_scale": 952.0, "dt_scale": 5.0, "k_mcp": 33.5})
    for k in ("P_plant", "w_chiller", "cool_out", "dt_evap"):
        assert torch.allclose(a[k], b[k], atol=1e-6), f"{k} 在两条路径下不一致"
