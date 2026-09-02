"""少样本曲线 + 零样本守门（设计文档 §6 P5 第 2、3 条，验收项 M2）。

    python tools/run_fewshot.py --ckpt experiments/results/p5b_availdrop/model_seed0.pt \
        --sites pc3,hx --tag p5b_availdrop

## 两条口径，必须一起读

**一、描述符与归一化沿用预训练口径，只换窗口划分。**
`build_sites` 在**训练站**上联合拟合 `DescNormalizer`，且每站的 `desc` /
`scales` 由该站的 `train_rows` 算出。若为了做时序评测而整体改成
`chronological`，`train_rows` 会变 -> `desc` 跟着变 -> 喂给模型的就不是它
训练时见过的那套（§13 #49 的同型陷阱）。
故本工具**先按预训练的 blocked 口径装配全部 14 站**，再单独为目标站重算一份
`chronological` 的窗口划分，只替换窗口，不动 desc / norm / scales。

**二、归一化用了该站全年的数据。**
这是项目既定口径（`physwm/data/scales.py`：「零样本」指不微调，不是没有该站
数据）。零样本臂与适配臂用的是同一份 norm，故**两臂之间的比较是公平的**；
但绝对值相对「连归一化都只能用前 N 天」的严格口径**偏乐观**。报数时须写明。
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from physwm.data.dataset import (WindowSpec, enumerate_windows,  # noqa: E402
                                 split_windows)
from physwm.model.world_model import ModelConfig, WorldModel  # noqa: E402
from physwm.train.fewshot import FewShotConfig, gate_holds, gated_adapt  # noqa: E402
from physwm.train.multisite import build_sites  # noqa: E402


def chronological_splits(site) -> dict:
    """只重算窗口划分，desc / norm / scales 一律沿用预训练口径。"""
    spec_c = dataclasses.replace(site.spec, split_mode="chronological")
    windows = enumerate_windows(site.sd, site.spec)
    return split_windows(site.sd, windows, spec_c)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sites", default="pc3,hx")
    ap.add_argument("--hold-out", default="hx,pc3",
                    help="必须与训练时一致，否则描述符归一化对不上（§13 #49）")
    ap.add_argument("--days", default="0,1,3,7,14,30")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=3e-2)
    ap.add_argument("--H", type=int, default=8)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--tag", default="fewshot")
    a = ap.parse_args()
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")

    spec = WindowSpec(W=16, H=48)
    want = tuple(x for x in a.sites.split(",") if x.strip())
    hold = tuple(x for x in a.hold_out.split(",") if x.strip())
    days = [float(d) for d in a.days.split(",") if d.strip()]

    allsites = build_sites(ROOT / "data", dev, spec=spec, hold_out=hold,
                           artifacts=ROOT / "artifacts" / "multisite")
    sites = [s for s in allsites if s.name in want]
    big = max((s for s in allsites if not s.held_out), key=lambda s: s.sd.N)
    model = WorldModel(ModelConfig(delta=0.5), big.sd.sch.n_dev, big.sd.F).to(dev)
    st = torch.load(a.ckpt, map_location=dev, weights_only=False)
    model.load_state_dict(st.get("model", st) if isinstance(st, dict) else st)
    model.eval()

    out = {}
    for s in sites:
        # 换成时序划分：适配取最早的一段、评测取最晚的一段
        sp = chronological_splits(s)
        s = dataclasses.replace(s, splits=sp)
        print(f"\n=== {s.name}  档 {s.tier}  冷站 {s.plant}"
              f"{'  留出/零样本' if s.held_out else '  训练站'} ===")
        print(f"时序划分窗口数  train {len(sp['train'])} / val {len(sp['val'])}"
              f" / test {len(sp['test'])}")
        print(f"{'适配天数':>8}{'适配窗口':>9}{'守门-零样本':>12}{'守门-适配':>11}"
              f"{'采纳?':>7}{'test 零样本 R²':>15}{'test 最终 R²':>14}{'MAE':>10}")
        print("-" * 98)

        rows = []
        for seed in a.seeds:
            for d in days:
                cfg = FewShotConfig(n_days=d, steps=a.steps, lr=a.lr, H=a.H,
                                    seed=seed)
                r = gated_adapt(model, s, cfg, dev)
                r["seed"] = seed
                rows.append(r)
                g = lambda m, k: ("     n/a" if m is None or not np.isfinite(m[k])
                                  else f"{m[k]:8.4f}")  # noqa: E731
                print(f"{d:>8.0f}{r['n_adapt_windows']:>9}"
                      f"{g(r['gate_zero'], 'R2'):>12}{g(r['gate_adapt'], 'R2'):>11}"
                      f"{'是' if r['adopted'] else '否':>7}"
                      f"{r['test_zero']['R2']:>15.4f}{r['test_final']['R2']:>14.4f}"
                      f"{r['test_final']['MAE']:>10.1f}")
        held = gate_holds(rows)
        print(f"  零样本守门（任意样本规模下不低于零样本）：{'成立' if held else '**失效**'}")
        out[s.name] = {"tier": s.tier, "plant": s.plant, "held_out": s.held_out,
                       "gate_holds": bool(held), "rows": rows,
                       "n_windows": {k: int(len(v)) for k, v in sp.items()
                                     if not k.startswith("_")}}

    print("\n" + "=" * 98)
    print("口径：目标站按**时序**划分（适配取最早、评测取最晚），"
          "考的是「先采一段再上线」下的季节漂移。")
    print("      归一化沿用该站全年数据（项目既定口径），故绝对值偏乐观；"
          "两臂共用同一份 norm，比较公平。")
    for n, r in out.items():
        print(f"  {n:<20}守门 {'成立' if r['gate_holds'] else '**失效**'}")

    d = ROOT / "experiments" / "results" / a.tag
    d.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(d / "fewshot.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=float)
    print(f"\n结果 -> experiments/results/{a.tag}/fewshot.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
