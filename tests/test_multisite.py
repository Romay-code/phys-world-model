"""多站联合预训练的装配与调度正确性（physwm/train/multisite.py）。

测的是**基础设施有没有把站点侧状态配对正确**，不是模型学得好不好。
配错站的 desc / 量纲 / 损失掩码，训练照样跑、不报任何错，
只会让某些站的预测系统性偏移 —— 与本项目反复遇到的静默失效同类。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from physwm.data.dataset import WindowSpec  # noqa: E402
from physwm.train.multisite import (DUP_FILES, Site, build_sites,  # noqa: E402
                                    describe, site_sampler)

pytestmark = pytest.mark.skipif(not (ROOT / "data").exists(), reason="无数据目录")


def _fake(name, n_train, held=False):
    class _SD:
        N = 40
    return Site(name=name, sd=_SD(), splits={"train": [], "val": []},
                spec=WindowSpec(), norm=None, ctx=None, scales={}, obs_loss=None,
                ds_tr=None, ds_va=None, n_train=n_train, held_out=held)


def test_held_out_sites_are_never_sampled():
    """留出站点被采到 = 零样本评测失效，而且结果会好得可疑。"""
    sites = [_fake("a", 100), _fake("b", 100), _fake("hold", 100, held=True)]
    g = site_sampler(sites, "proportional", seed=0)
    seen = {sites[next(g)].name for _ in range(3000)}
    assert "hold" not in seen, "留出站点被采样了 —— 零样本口径已污染"
    assert seen == {"a", "b"}


def test_all_held_out_raises():
    """全部留出必须显式报错，不能静默训练 0 个站。"""
    with pytest.raises(ValueError, match="留出"):
        list(site_sampler([_fake("a", 10, held=True)], "proportional"))


def test_proportional_follows_window_counts():
    sites = [_fake("big", 9000), _fake("small", 1000)]
    g = site_sampler(sites, "proportional", seed=0)
    n = sum(sites[next(g)].name == "big" for _ in range(4000))
    assert 0.85 < n / 4000 < 0.95, f"比例采样偏离窗口数：big 占 {n/4000:.2%}，期望 ~90%"


def test_uniform_ignores_window_counts():
    sites = [_fake("big", 9000), _fake("small", 1000)]
    g = site_sampler(sites, "uniform", seed=0)
    n = sum(sites[next(g)].name == "big" for _ in range(4000))
    assert 0.45 < n / 4000 < 0.55, f"均匀采样不该受窗口数影响，实得 {n/4000:.2%}"


def test_unknown_sampler_raises():
    with pytest.raises(ValueError, match="未知的站点采样"):
        list(site_sampler([_fake("a", 10)], "round_robin"))


@pytest.mark.slow
def test_build_sites_pairs_state_correctly():
    """真数据装配：每站的 desc / type_id / 量纲必须与该站的 token 数配套。

    配错了不会报错 —— 只会让模型用别站的描述符解释这个站。
    """
    sites = build_sites(ROOT / "data", torch.device("cpu"),
                        hold_out=("hx",), only=("yb3", "tx", "hx"))
    assert {s.name for s in sites} == {"yb3", "tx", "hx"}
    for s in sites:
        n = s.sd.N
        assert s.ctx.desc.shape[0] == n, f"{s.name} 的 desc 行数与 token 数不符"
        assert s.ctx.type_id.shape[0] == n
        assert s.ctx.stat_rel.shape[:2] == (n, n)
        assert s.scales["w_scale"] > 0
    assert [s.held_out for s in sites if s.name == "hx"] == [True]
    assert "留出" in describe(sites)


def test_dup_files_excluded():
    """同站的多个文件必须去重，否则同一站会被当成多个站重复采样。"""
    assert {"yb3test", "zxtest", "yb3_topology_complete",
            "yb3_topology_clean_v2"} == set(DUP_FILES)
