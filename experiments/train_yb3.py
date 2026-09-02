"""yb3 训练入口。

P2（自回归内核 v0）：H=1 单步，与基线对齐口径。
    python experiments/train_yb3.py --device cuda --mode single --epochs 300 --seeds 0 1 2 3 4

P3（多步推演）：三阶段课程。
    python experiments/train_yb3.py --device cuda --mode curriculum --epochs 1200 --H 48
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

from physwm.data import schema as S  # noqa: E402
from physwm.data.dataset import WindowSpec, build_site_bundle  # noqa: E402
from physwm.data.descriptors import (DescNormalizer, DescriptorBundle,  # noqa: E402
                                     compute_descriptors)
from physwm.eval.metrics import format_metrics  # noqa: E402
from physwm.model.world_model import ModelConfig, WorldModel  # noqa: E402
from physwm.train.curriculum import Curriculum, SingleStep  # noqa: E402
from physwm.train.loop import (Context, TrainConfig, final_report,  # noqa: E402
                               setup_backend, train)
from physwm.train.losses import LossWeights  # noqa: E402

CSV = ROOT / "data" / "yb3_topology_complete.csv"
sys.path.insert(0, str(ROOT / "experiments"))
from smoke_yb3 import (infer_dt_evap_scale, infer_k_mcp, infer_mcp_cool_scale, infer_mcp_scale, infer_scales,  # noqa: E402
                       infer_w_scale)


def build(args, device):
    spec = WindowSpec(W=args.W, H=args.H, split_mode=args.split_mode)
    b = build_site_bundle(CSV, "yb3", spec, out_dir=ROOT / "artifacts")
    sd, splits = b["data"], b["splits"]
    trm = b["train_rows"]

    cache = ROOT / "artifacts" / f"desc_yb3_W{spec.W}H{spec.H}_{spec.split_mode}.npz"
    if cache.exists() and not args.refresh_desc:
        db = DescriptorBundle.load(cache)
    else:
        db = compute_descriptors(sd.x, sd.avail, sd.sch, trm)
        db.save(cache)
    dn = DescNormalizer.fit([db.desc])

    ctx = Context(
        desc=torch.from_numpy(dn.apply(db.desc)).to(device),
        stat_rel=torch.from_numpy(db.stat_rel).to(device),
        type_id=torch.from_numpy(db.type_id).to(device),
        W=spec.W, site_ctx_dim=ModelConfig().d_ctx)
    return b, sd, splits, spec, ctx, trm


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--mode", choices=["single", "curriculum"], default="single")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=0,
                    help="0=按 H 自动选（H<=8 用 256，否则 128）")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--W", type=int, default=16)
    ap.add_argument("--H", type=int, default=48,
                    help="数据窗口的最大推演步数（决定 seq 长度与可采窗口数）")
    ap.add_argument("--train-H", type=int, default=0,
                    help="课程的 H_max，0=同 --H。与 --H 解耦是为了"
                         "「训练到 48 步、评测到 96 步」这种超视野外推评测")
    ap.add_argument("--eval-H", type=int, default=0, help="0=用训练末期的 H")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--delta", type=float, default=1.0)
    ap.add_argument("--q-mode", choices=["from_w", "free"], default="from_w")
    ap.add_argument("--cool-dt-mode", choices=["tied", "free"], default="tied",
                    help="tied = cool_dt 由 q_cond 导出（eta 尺度可辨识）；"
                         "free = P2/P3 的自由预测写法，作对照臂")
    ap.add_argument("--dt-evap-mode", choices=["soft", "derived", "free"], default="soft",
                    help="free = 复现 #24 病灶的消融臂（dt_evap 独立预测，eta 无梯度）")
    ap.add_argument("--lam-lat", type=float, default=0.1)
    ap.add_argument("--dir-margin", type=float, default=0.25,
                    help="方向损失的正 margin（归一化响应的倍数）。"
                         "0 = 退回无 margin 版，其驻点在「零响应」处，dir_vr 压不下去")
    ap.add_argument("--dir-h", default="1",
                    help="方向损失施加在哪几个推演步，逗号分隔。"
                         "P4-C 实测：只压 h=1 压不住 h=36，正式配置用 1,12,36")
    ap.add_argument("--lam-dir", type=float, default=0.0,
                    help="方向损失（P4）。单步、平方 hinge，代价约 +5%")
    ap.add_argument("--lam-evap-bal", type=float, default=0.1,
                    help="L-soft-B 蒸发侧能量平衡软一致项。0 = 退化为 #24 的病灶")
    ap.add_argument("--lam-soft-b", type=float, default=0.0,
                    help="L-soft-B：同机 Q_evap/PLR 容量守恒（P4，默认关）")
    ap.add_argument("--lam-lip", type=float, default=0.0,
                    help="Lipschitz 显式惩罚。§4.4 的决策就是 0（谱归一化已压住）。"
                         "P2/P3 的 CLI 默认误写成 0.01，所幸 n_iter=2 下该项恒为 0，"
                         "历史结果不受影响。见 §13 #34")
    ap.add_argument("--lipschitz-max", type=float, default=1.0)
    ap.add_argument("--patience", type=int, default=60)
    ap.add_argument("--steps-per-epoch", type=int, default=0,
                    help="0=走完整训练集；H 大时限步可大幅提速")
    ap.add_argument("--select-H", type=int, default=0,
                    help="选模评估的固定 H，0=用 H_max。不可用课程当前 H，"
                         "否则必然选中 t1 权重、多步训练全部作废")
    ap.add_argument("--lat-k", type=int, default=4,
                    help="隐一致性抽样的 h 个数，0=每步（H 大时很贵）")
    ap.add_argument("--tag", default="p2_single")
    ap.add_argument("--split-mode", choices=["blocked", "chronological"],
                    default="blocked",
                    help="blocked=按块交错(开发主口径); chronological=跨季节外推")
    ap.add_argument("--dir-vr", action="store_true",
                    help="额外跑方向一致性评测（§5.3，P4 门限 dir_vr(36)<=0.10）。"
                         "每个动作维多一遍推演，约 5 倍评测开销")
    ap.add_argument("--refresh-desc", action="store_true")
    ap.add_argument("--no-tf32", action="store_true",
                    help="关闭 TF32（需要逐位复现时用，会慢 2-3 倍）")
    a = ap.parse_args()
    setup_backend(tf32=not a.no_tf32)

    device = torch.device(a.device if torch.cuda.is_available() or a.device == "cpu"
                          else "cpu")
    if a.device == "cuda" and device.type == "cpu":
        print("[warn] 请求 cuda 但不可用，回退 cpu")

    out_dir = ROOT / "experiments" / "results" / a.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    b, sd, splits, spec, ctx, trm = build(a, device)
    print(f"站点 yb3  N={sd.N} F={sd.F}  窗口 "
          f"train/val/test={len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])}")
    print(f"设备 {device}  模式 {a.mode}  epochs {a.epochs}  seeds {a.seeds}")

    wts = LossWeights(lam_lat=a.lam_lat, lam_lip=a.lam_lip,
                      lipschitz_max=a.lipschitz_max, lat_k=a.lat_k,
                      lam_soft_b=a.lam_soft_b, lam_evap_bal=a.lam_evap_bal,
                      lam_dir=a.lam_dir,
                      dir_h=tuple(int(x) for x in str(a.dir_h).split(",") if x.strip()),
                      dir_margin=a.dir_margin)
    sc = infer_scales(sd, trm)
    sc["w_scale"] = infer_w_scale(sd, trm)
    sc["dt_evap_scale"] = infer_dt_evap_scale(sd, trm)
    sc["mcp_scale"] = infer_mcp_scale(sd, trm, sc["w_scale"], sc["dt_evap_scale"])
    sc["mcp_cool_scale"] = infer_mcp_cool_scale(sd, trm, sc["w_scale"])
    sc["k_mcp"] = infer_k_mcp(sd, trm, sc["w_scale"], sc["dt_evap_scale"])

    runs = []
    for seed in a.seeds:
        print(f"\n{'='*70}\nseed {seed}")
        torch.manual_seed(seed)
        model = WorldModel(ModelConfig(delta=a.delta, q_mode=a.q_mode,
                                       dt_evap_mode=a.dt_evap_mode,
                                       cool_dt_mode=a.cool_dt_mode),
                           sd.sch.n_dev, sd.F).to(device)
        model.decoder.set_scales(**sc)
        model._type_id = ctx.type_id      # L-soft-B 要按族取 plr，见 losses.soft_b_capacity

        cur_cls = SingleStep if a.mode == "single" else Curriculum
        cur = cur_cls(total_epochs=a.epochs, H_max=(a.train_H or a.H))
        if a.mode == "curriculum":
            print("课程:\n" + cur.describe())

        cfg = TrainConfig(epochs=a.epochs, batch_size=a.batch_size, lr=a.lr,
                          patience=a.patience, seed=seed,
                          steps_per_epoch=a.steps_per_epoch,
                          select_H=(a.select_H or a.train_H or a.H))
        res = train(model, sd, splits, spec, b["norm"], ctx, cfg, cur, wts,
                    device, out_dir=out_dir)
        print(f"  最优 val {res['best']['loss']:.4f} @ epoch {res['best']['epoch']}"
              f"  用时 {res['minutes']:.1f} min")

        H_eval = a.eval_H or cur.H(a.epochs - 1)
        rep = final_report(model, sd, splits, spec, b["norm"], ctx, cfg, wts,
                           device, H_eval=H_eval, split="test", dir_vr=a.dir_vr)
        rep["seed"] = seed
        rep["best"] = res["best"]
        runs.append(rep)

        s1 = rep["step1"]
        print(f"  test H={H_eval}  step1: MAE {s1['MAE']:.1f} R2 {s1['R2']:.4f} "
              f"dyMAE {s1['dyMAE']:.1f}")
        print(f"  h*={rep['h_star']['h_star']:.0f} (slope {rep['h_star']['slope']:.2f})"
              f"  误差放大 TF {rep['amp']['tf_MAE']:.1f} -> FR {rep['amp']['fr_MAE']:.1f} "
              f"(+{rep['amp']['rel_amp']*100:.1f}%)")
        hm = rep.get("h_star_multi") or {}
        if hm:
            # G9 未拍板，三口径并列（§13 #27）。方括号内是卡住它的判据，
            # "顶格" 表示 h* 顶到评测视野本身，那个数字只是下界（§13 #26）。
            lbl = {"r2_only": "R2>=.80(验收)", "self_ratio": "+3xMAE_1(§5.1)",
                   "abs_bar": "+绝对阈(10%P)"}
            parts = []
            for k in ("r2_only", "self_ratio", "abs_bar"):
                d = hm.get(k)
                if not d:
                    continue
                tag = "顶格" if d["censored"] else d["binding"]
                parts.append(f"{lbl[k]}={d['h_star']:.0f}[{tag}]")
            print("  h* 三口径: " + "  ".join(parts))
        if rep.get("dir"):
            from physwm.eval.direction import format_direction
            at = 36 if H_eval >= 36 else H_eval
            print("  方向一致性 dir_vr（门限 h=36 时 <=0.10）：")
            for ln in format_direction(rep["dir"], at=at).splitlines():
                print("    " + ln)
        ei = rep.get("eta_ident") or {}
        if "shifts" in ei:
            from physwm.eval.identifiability import format_identifiability
            print("  eta 可辨识性（平移 logit 看监督损失是否变化）：")
            for ln in format_identifiability(ei).splitlines():
                print("    " + ln)
        elif ei.get("error"):
            print(f"  [warn] eta 可辨识性诊断失败: {ei['error']}")
        bad = {k: v for k, v in rep["hard"].items() if v > 1e-5}
        print(f"  硬约束: {'零违例' if not bad else bad}")
        ph = rep.get("phys", {})
        if ph:
            print(f"  物理量(开机机中位): COP={ph.get('cop_actual',0):.2f} "
                  f"eta={ph.get('eta',0):.3f} COP_carnot={ph.get('cop_carnot',0):.2f} "
                  f"Q={ph.get('q_evap',0):.0f}kW W={ph.get('w_chiller',0):.0f}kW "
                  f"cool_dt={ph.get('cool_dt',0):.2f}K dT_evap={ph.get('dt_evap',0):.2f}K")
            print(f"    参照: pb1流量实测 COP 中位 8.11, eta 中位 0.563")
        print(f"  w_scale = {float(model.decoder.w_scale):.1f} kW/台")

    # ---- 跨 seed 汇总。必须分 seed 看，均值会把稳定性崩坏误报成精度下滑 ----
    print(f"\n{'='*70}\n跨 seed 汇总（tag={a.tag}）")
    rows = {}
    for r in runs:
        rows[f"seed{r['seed']}"] = r["step1"]
    print(format_metrics(rows))

    mae = np.array([r["step1"]["MAE"] for r in runs])
    r2 = np.array([r["step1"]["R2"] for r in runs])
    hstar = np.array([r["h_star"]["h_star"] for r in runs])
    print(f"\n  step1 MAE  {mae.mean():.1f} +- {mae.std():.1f}   "
          f"(cv {mae.std()/max(mae.mean(),1e-9)*100:.1f}%)  分 seed {np.round(mae,1).tolist()}")
    print(f"  step1 R2   {r2.mean():.4f} +- {r2.std():.4f}")
    print(f"  h*         {hstar.mean():.1f} +- {hstar.std():.1f}  分 seed {hstar.tolist()}")
    if mae.std() > 0.30 * mae.mean():
        print("  [警告] 跨 seed 标准差 > 均值的 30%，未达 P3 稳定性门限")

    (out_dir / "summary.json").write_text(
        json.dumps({"tag": a.tag, "mode": a.mode, "args": vars(a), "runs": runs},
                   ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(f"\n结果 -> {out_dir.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
