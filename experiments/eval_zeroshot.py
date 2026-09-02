"""P5 零样本 / 全站正式评测：把「验证损失」换成验收口径的 MAE / R² / h*。

    python experiments/eval_zeroshot.py --ckpt experiments/results/p5a_scout/model_seed0.pt \
        --hold-out hx,pc3 --eval-H 48 --tag p5a_scout

**训练损失不是指标。** 指标⑥要的是留出站点零样本 R² >= 0.80，
而训练损失是 k1–k5 的加权和，两者不可换算。P5-A 侦察轮报的都是前者。

同时评全部 14 站（训练站 + 留出站），因为「零样本好不好」只有跟
「见过的站好到什么程度」并排才有意义。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from physwm.data.dataset import WindowSpec  # noqa: E402
from physwm.model.world_model import ModelConfig, WorldModel  # noqa: E402
from physwm.train.loop import TrainConfig, final_report  # noqa: E402
from physwm.train.losses import LossWeights  # noqa: E402
from physwm.train.multisite import build_sites  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--W", type=int, default=16)
    ap.add_argument("--H", type=int, default=48)
    ap.add_argument("--eval-H", type=int, default=48)
    ap.add_argument("--delta", type=float, default=0.5)
    ap.add_argument("--hold-out", default="hx,pc3")
    # 0 = 评完整个 test。**正式指标必须用 0** —— 截断取到的是最早的一批
    # 窗口（窗口按时间升序、evaluate 不打乱），实测只覆盖 41.5%~67.9%
    # 的 test 且系统性排除盛夏，偏乐观。
    ap.add_argument("--max-batches", type=int, default=0)
    # 轴 3 时段留出：按**月份序数**留出（如 7,8），跨全部年份、全部站点。
    # 绝对时间窗留不掉 bh 的夏天（它的日历与其余 13 站不重叠），
    # 见 physwm/data/temporal_holdout.py。空 = 不做时段留出。
    ap.add_argument("--season-holdout", default="",
                    help="留出的月份序数，逗号分隔，如 7,8")
    ap.add_argument("--dir-vr", action="store_true")
    ap.add_argument("--tag", default="p5a_scout")
    a = ap.parse_args()

    dev = torch.device(a.device if torch.cuda.is_available() or a.device == "cpu" else "cpu")
    spec = WindowSpec(W=a.W, H=a.H)
    wts = LossWeights(lam_evap_bal=0.03)
    hold = tuple(x for x in a.hold_out.split(",") if x.strip())
    season = tuple(int(x) for x in a.season_holdout.split(",") if x.strip()) or None
    sites = build_sites(ROOT / "data", dev, spec=spec, wts=wts, hold_out=hold,
                        artifacts=ROOT / "artifacts" / "multisite",
                        season_months=season)

    big = max((s for s in sites if not s.held_out), key=lambda s: s.sd.N)
    model = WorldModel(ModelConfig(delta=a.delta), big.sd.sch.n_dev, big.sd.F).to(dev)
    st = torch.load(a.ckpt, map_location=dev, weights_only=False)
    model.load_state_dict(st.get("model", st) if isinstance(st, dict) else st)
    model.eval()
    cfg = TrainConfig(epochs=1, batch_size=64, lr=1e-3)

    rows, out = [], {}
    print(f"{'站点':<22}{'角色':<10}{'MAE':>9}{'R²':>9}{'h*(验收)':>11}"
          f"{'h*(绝对阈)':>11}{'硬约束':>8}")
    print("-" * 82)
    for s in sites:
        rep = final_report(model, s.sd, s.splits, s.spec, s.norm, s.ctx, cfg, wts,
                           dev, H_eval=a.eval_H, split="test",
                           max_batches=a.max_batches, dir_vr=a.dir_vr,
                           scales=s.scales)
        hm = rep.get("h_star_multi") or {}
        bad = {k: v for k, v in rep["hard"].items() if v > 1e-5}
        r = {"site": s.name, "held_out": s.held_out,
             "MAE": rep["step1"]["MAE"], "R2": rep["step1"]["R2"],
             "h_r2": (hm.get("r2_only") or {}).get("h_star"),
             "h_abs": (hm.get("abs_bar") or {}).get("h_star"),
             "hard_ok": not bad, "per_h": rep["per_h"], "phys": rep.get("phys", {}),
             "dir": rep.get("dir")}
        rows.append(r); out[s.name] = r
        print(f"{s.name:<22}{'留出(零样本)' if s.held_out else '训练':<10}"
              f"{r['MAE']:>9.1f}{r['R2']:>9.4f}{str(r['h_r2']):>11}"
              f"{str(r['h_abs']):>11}{'零违例' if r['hard_ok'] else '有违例':>8}")

    tr = [r for r in rows if not r["held_out"]]
    ho = [r for r in rows if r["held_out"]]
    print("-" * 82)
    import numpy as np
    print(f"训练站 {len(tr)} 个：R² 中位 {np.median([r['R2'] for r in tr]):.4f}   "
          f"最低 {min(r['R2'] for r in tr):.4f}")
    print(f"留出站 {len(ho)} 个：" + "  ".join(
        f"{r['site']} R²={r['R2']:.4f}{' ✅' if r['R2'] >= 0.80 else ' ❌'}" for r in ho))
    print(f"\n验收指标⑥门限：留出站点零样本 R² >= 0.80")

    d = ROOT / "experiments" / "results" / a.tag
    d.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(d / "zeroshot_eval.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=float)
    print(f"结果 -> experiments/results/{a.tag}/zeroshot_eval.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
