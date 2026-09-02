"""P5 多站联合预训练。

    python experiments/train_multisite.py --device cuda --epochs 50 --steps 200 \
        --hold-out hx,pc3 --tag p5a_scout

留出站点默认取 **hx** 与 **pc3**：
  hx        B 系历史崩溃站（B0/B1 在此 R² 为 −5.94 / −7.64）。设计文档 §5.2 明确
            要求 P5 回答「B2 是不是只在 yb3 好使」，它是最该被问的那个站。
  pc3    能力最差的站：冷机逐台功率标签 0/7，全站只能用聚合列。
            零样本若在它上面成立，说明能力矩阵掩码确实生效。
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
from physwm.train.curriculum import Curriculum  # noqa: E402
from physwm.train.loop import TrainConfig  # noqa: E402
from physwm.train.losses import LossWeights  # noqa: E402
from physwm.train.avail_dropout import AvailDropout  # noqa: E402
from physwm.train.multisite import build_sites, describe, train_multisite  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--W", type=int, default=16)
    ap.add_argument("--H", type=int, default=48)
    ap.add_argument("--select-H", type=int, default=48)
    ap.add_argument("--delta", type=float, default=0.5)
    ap.add_argument("--lam-lat", type=float, default=0.1)
    ap.add_argument("--lam-evap-bal", type=float, default=0.03)
    ap.add_argument("--lam-dir", type=float, default=0.0)
    ap.add_argument("--dir-h", default="1,12,36")
    ap.add_argument("--dir-margin", type=float, default=0.25)
    ap.add_argument("--sample", choices=["proportional", "uniform"],
                    default="proportional")
    ap.add_argument("--hold-out", default="hx,pc3")
    ap.add_argument("--only", default="")
    # 轴 3 时段留出：按**月份序数**留出（如 7,8），跨全部年份、全部站点。
    # 绝对时间窗留不掉 bh 的夏天（它的日历与其余 13 站不重叠），
    # 见 physwm/data/temporal_holdout.py。空 = 不做时段留出。
    ap.add_argument("--season-holdout", default="",
                    help="留出的月份序数，逗号分隔，如 7,8")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--tag", default="p5a_scout")
    ap.add_argument("--log-every", type=int, default=2)
    # 输入可用性丢弃（P5-B 修复项）。默认关闭 —— 开关必须显式，
    # 否则 P5-A 的结果与之后的臂不可比。
    ap.add_argument("--avail-drop-joint", type=float, default=0.0,
                    help="整组通道一起掩的概率，模拟 pc3 那种全聚合站")
    ap.add_argument("--avail-drop-each", type=float, default=0.0,
                    help="逐条独立掩的概率，模拟 zx/pb1/pc2 那种部分缺失")
    a = ap.parse_args()

    dev = torch.device(a.device if torch.cuda.is_available() or a.device == "cpu" else "cpu")
    if a.device == "cuda" and dev.type == "cpu":
        print("[warn] 请求 cuda 但不可用，回退 cpu")
    out_dir = ROOT / "experiments" / "results" / a.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    spec = WindowSpec(W=a.W, H=a.H)
    wts = LossWeights(lam_lat=a.lam_lat, lam_evap_bal=a.lam_evap_bal,
                      lam_dir=a.lam_dir, dir_margin=a.dir_margin,
                      dir_h=tuple(int(x) for x in a.dir_h.split(",") if x.strip()))
    hold = tuple(x for x in a.hold_out.split(",") if x.strip())
    season = tuple(int(x) for x in a.season_holdout.split(",") if x.strip()) or None
    only = tuple(x for x in a.only.split(",") if x.strip())
    adrop = AvailDropout(p_joint=a.avail_drop_joint, p_each=a.avail_drop_each)

    sites = build_sites(ROOT / "data", dev, spec=spec, wts=wts,
                        hold_out=hold, only=only,
                        artifacts=ROOT / "artifacts" / "multisite",
                        season_months=season)
    print(describe(sites))
    print(f"\n站点采样 {a.sample}   delta {a.delta}   "
          f"lam_lat {a.lam_lat}  lam_evap_bal {a.lam_evap_bal}  lam_dir {a.lam_dir}")
    print("输入可用性丢弃: " + (
        f"整组 {adrop.p_joint}  逐条 {adrop.p_each}  可掩 {len(adrop.channels)} 条"
        if adrop.enabled else "关闭"))

    # 模型按 token 数最多的**训练**站构建。权重形状与台数无关
    # （tests/test_cross_site.py 保证），这里只是取一份 n_dev 供
    # 解码器判断哪些族存在 —— 14 站四族齐全，故取最大者安全。
    big = max((s for s in sites if not s.held_out), key=lambda s: s.sd.N)
    print(f"模型按 {big.name} 构建（N={big.sd.N}）")

    runs = []
    for seed in a.seeds:
        print(f"\n{'='*70}\nseed {seed}")
        torch.manual_seed(seed)
        model = WorldModel(ModelConfig(delta=a.delta), big.sd.sch.n_dev, big.sd.F).to(dev)
        model._type_id = big.ctx.type_id
        cfg = TrainConfig(epochs=a.epochs, batch_size=a.batch_size, lr=a.lr,
                          steps_per_epoch=a.steps, seed=seed, select_H=a.select_H)
        cur = Curriculum(total_epochs=a.epochs, H_max=a.H)
        print("课程:\n" + cur.describe())
        r = train_multisite(model, sites, cfg, cur, wts, dev, sample=a.sample,
                            seed=seed, out_dir=out_dir, log_every=a.log_every,
                            avail_drop=adrop)
        r["seed"] = seed
        runs.append(r)
        print(f"  最优 val跨站均值 {r['best']['loss']:.4f} @ epoch {r['best']['epoch']}"
              f"  用时 {r['minutes']:.1f} min")
        h = r["history"][-1]
        print(f"  末轮零样本: " + "  ".join(f"{k}={v:.4f}"
                                          for k, v in h["zeroshot_per_site"].items()))
        json.dump(r, open(out_dir / f"history_seed{seed}.json", "w",
                          encoding="utf-8"), ensure_ascii=False, indent=1)

    json.dump({"tag": a.tag, "args": vars(a), "runs": runs},
              open(out_dir / "summary.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"\n结果 -> experiments/results/{a.tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
