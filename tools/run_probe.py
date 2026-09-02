"""跑线性探针（设计文档 §6 P5 门限：可回读 ≥3 类物理量，R² ≥ 0.70）。

    python tools/run_probe.py --ckpt experiments/results/p5a_scout/model_seed0.pt \
        --sites yb3,hx,tx,pb1,pc3 --tag p5a_scout

**报数纪律**：每个目标都并排给 `R²(z)` 与对照臂 `R²(raw)`。
`plr` / `approach` / `dt_evap` 是输入通道的平凡函数，即使 R² 很高也
**不构成表征证据**（与 §13 #36 同类），门限只数非平凡的那几个。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from physwm.data.dataset import WindowSpec  # noqa: E402
from physwm.eval.probe import probe_site  # noqa: E402
from physwm.model.world_model import ModelConfig, WorldModel  # noqa: E402
from physwm.train.multisite import build_sites  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sites", default="yb3,tx,pb1,hx,pc3")
    ap.add_argument("--hold-out", default="hx,pc3",
                    help="必须与训练时一致，否则描述符归一化对不上（§13 #49）")
    ap.add_argument("--max-batches", type=int, default=60)
    ap.add_argument("--gate", type=float, default=0.70)
    ap.add_argument("--tag", default="probe")
    a = ap.parse_args()
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")

    spec = WindowSpec(W=16, H=48)
    want = tuple(x for x in a.sites.split(",") if x.strip())
    hold = tuple(x for x in a.hold_out.split(",") if x.strip())
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
        r = probe_site(model, s, dev, max_batches=a.max_batches, gate=a.gate)
        out[s.name] = r
        print(f"\n=== {s.name}{'（留出/零样本）' if s.held_out else '（训练站）'} ===")
        print(f"{'目标':<14}{'平凡?':>7}{'R²(z)':>10}{'R²(raw)':>10}"
              f"{'增益':>9}{'可读?':>7}{'n_val':>8}   说明")
        print("-" * 96)
        for row in r["rows"]:
            f = lambda v: "  n/a" if not np.isfinite(v) else f"{v:>9.3f}"  # noqa: E731
            print(f"{row['target']:<14}{'平凡' if row['trivial'] else '非平凡':>7}"
                  f"{f(row['r2_z'])}{f(row['r2_raw'])}{f(row['gain'])}"
                  f"{'是' if row['readable'] else '不可读':>7}"
                  f"{row['n_val']:>8}   {row['note']}")
        print(f"  过线(≥{a.gate})：非平凡 {r['n_pass_nontrivial']}/{r['n_nontrivial']}"
              f"（其中 {r['n_nontrivial_unreadable']} 项对照臂为负、不可读）   "
              f"全部 {r['n_pass_all']}/{len(r['rows'])}")

    print("\n" + "=" * 96)
    print(f"门限：可回读 ≥3 类物理量（R² ≥ {a.gate}）。"
          f"**只数非平凡目标** —— 平凡目标高分只说明编码器没丢输入。")
    for n, r in out.items():
        ok = "达标" if r["n_pass_nontrivial"] >= 3 else "未达标"
        print(f"  {n:<20}非平凡过线 {r['n_pass_nontrivial']}/{r['n_nontrivial']}   {ok}"
              + (f"   （{r['n_nontrivial_unreadable']} 项不可读）"
                 if r["n_nontrivial_unreadable"] else ""))

    d = ROOT / "experiments" / "results" / a.tag
    d.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(d / "probe.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=float)
    print(f"\n结果 -> experiments/results/{a.tag}/probe.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
