"""诊断：零样本失败是不是「输入可用性模式」的分布外。

P5-A 实测：留出站 hx 零样本 R²=0.9335（通过），pc3 R²=−1.6377（崩）。
排除了两个解释：
  · 不是标签缺失的假象 —— 零样本评的是 P_plant，pc3 的 P_plant 有效率 1.000
  · 不是数值分布外 —— pc3 的 P_plant 范围 [1855,7619] 落在训练站 [217,9251] 内

剩下的假设：**pc3 的冷机逐台功率通道整体为空（avail=0.000），
而 12 个训练站都有（最差的 zx 也有 0.646）。模型从没见过这种缺失模式。**

验法：拿模型熟悉的站，人为把历史窗里同一个通道掩掉，看 R² 是否同样崩。
崩 -> 假设成立，问题在缺失模式而非站点本身；
不崩 -> 假设不成立，得另找原因。

    python tools/probe_avail_shift.py --ckpt ... --site yb3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from physwm.data import schema as S  # noqa: E402
from physwm.data.dataset import WindowSpec  # noqa: E402
from physwm.eval.metrics import all_metrics  # noqa: E402
from physwm.model.world_model import ModelConfig, WorldModel  # noqa: E402
from physwm.train.multisite import build_sites  # noqa: E402


@torch.no_grad()
def run(model, s, dev, H, batches, mask_fam=None, mask_fld=None):
    s.ds_va.set_H(H)
    k = S.CANON_FIELDS[mask_fld[0]].index(mask_fld[1]) if mask_fld else None
    rows = ([i for i, (f, _) in enumerate(s.sd.sch.token_index) if f == mask_fam]
            if mask_fam else [])
    yt, yp = [], []
    for i, b in enumerate(s.ds_va.epoch(64, shuffle=False, drop_last=False)):
        # 0 = 不限。窗口按时间升序、epoch 不打乱，截断取到的是**最早**
        # 的一批窗口而非随机样本（§13 #56）—— 诊断工具同样会被它带偏。
        if batches and i >= batches:
            break
        if rows:
            b = dict(b)
            b["seq_avail"] = b["seq_avail"].clone()
            b["seq_x"] = b["seq_x"].clone()
            b["seq_avail"][:, :, rows, k] = 0.0
            b["seq_x"][:, :, rows, k] = 0.0
        o = model.rollout(b, desc=s.ctx.desc, stat_rel=s.ctx.stat_rel,
                          type_id=s.ctx.type_id,
                          site_ctx=s.ctx.ctx(b["P_plant"].shape[0], dev),
                          H=H, W=s.ctx.W, reanchor_p=0.0, scales=s.scales)
        yt.append(b["P_plant"][:, 0].cpu().numpy())
        yp.append(o["preds"][0]["P_plant"].cpu().numpy())
    return all_metrics(np.concatenate(yt), np.concatenate(yp))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sites", default="yb3,tx,pb1,zx,pc3")
    ap.add_argument("--H", type=int, default=8)
    ap.add_argument("--batches", type=int, default=0,
                    help="0 = 评完整个 split。正式判读必须用 0（§13 #56）")
    ap.add_argument("--hold-out", default="hx,pc3",
                    help="必须与训练时一致，否则描述符归一化对不上")
    a = ap.parse_args()
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")

    spec = WindowSpec(W=16, H=48)
    want = tuple(x for x in a.sites.split(",") if x.strip())
    hold = tuple(x for x in a.hold_out.split(",") if x.strip())
    # **必须装配全部 14 站、并用与训练同一份 hold_out**：`build_sites` 会在
    # 参与训练的站上联合拟合 `DescNormalizer`。只装子集会重新拟合出另一套
    # 归一化参数，喂给模型的 desc 就不是它训练时见过的那套 —— 「原始 R²」
    # 会对不上正式评测，且留出站还会因参与拟合而泄漏。装全站慢几分钟，
    # 但这是唯一能与 P5-A 对齐的口径。
    allsites = build_sites(ROOT / "data", dev, spec=spec, hold_out=hold,
                           artifacts=ROOT / "artifacts" / "multisite")
    sites = [s for s in allsites if s.name in want]
    missing = [w for w in want if w not in {s.name for s in sites}]
    if missing:
        raise SystemExit(f"这些站点不存在: {missing}")
    # 模型按 token 数最多的**训练**站构建，与 train_multisite 逐字一致
    big = max((s for s in allsites if not s.held_out), key=lambda s: s.sd.N)
    model = WorldModel(ModelConfig(delta=0.5), big.sd.sch.n_dev, big.sd.F).to(dev)
    st = torch.load(a.ckpt, map_location=dev, weights_only=False)
    model.load_state_dict(st.get("model", st) if isinstance(st, dict) else st)
    model.eval()

    k_cons = S.CANON_FIELDS["chiller"].index("consumption")
    print("前提实测：历史窗输入里 chiller.consumption 通道的可用率\n")
    print(f"{'站点':<20}{'chiller 台数':>12}{'avail 均值':>12}{'P_plant 有效率':>14}")
    print("-" * 60)
    for s_ in sites:
        rows_ = [i for i, (f, _) in enumerate(s_.sd.sch.token_index) if f == "chiller"]
        av = float(s_.sd.avail[:, rows_, k_cons].mean()) if rows_ else float("nan")
        pk = float(s_.sd.extra["P_plant_ok"].mean())
        print(f"{s_.name:<20}{len(rows_):>12}{av:>12.3f}{pk:>14.3f}")
    print()

    print("把历史窗里的**冷机逐台功率**通道整体掩掉，模拟pc3 的缺失模式\n")
    print(f"{'站点':<20}{'原始 R²':>10}{'掩掉后 R²':>12}{'原始 MAE':>11}{'掩掉后 MAE':>12}")
    print("-" * 66)
    for s in sites:
        a0 = run(model, s, dev, a.H, a.batches)
        a1 = run(model, s, dev, a.H, a.batches,
                 mask_fam="chiller", mask_fld=("chiller", "consumption"))
        print(f"{s.name:<20}{a0['R2']:>10.4f}{a1['R2']:>12.4f}"
              f"{a0['MAE']:>11.1f}{a1['MAE']:>12.1f}")
    print()
    print("读法：若训练站被掩掉后 R² 同样崩到负值，则 pc3 的失败源于**缺失模式分布外**，")
    print("      而非站点本身；若不崩，则该假设被否掉，须另找原因。")
    print("参照：P5-A 正式评测里 pc3 零样本 R²=−1.6377 / MAE 2662.9")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
