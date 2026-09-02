"""累积消融：把「pc3 缺哪些通道」逐条加回到一个模型熟悉的站上。

`probe_avail_shift.py` 只掩了 `chiller.consumption` 一条，结论是**否定**的 ——
hx 同为零样本站，掩掉它只从 0.938 掉到 0.851，远到不了 pc3 的 −1.82。
但那次测试选错了范围：`diag_avail_matrix.py` 摊开全通道后，pc3 相对 12 个
训练站中位实际缺**五条**：

    chiller.consumption   0.000 vs 1.000     逐台冷机功率
    plant.tower_out       0.000 vs 1.000     冷却塔出水温 —— **全 14 站里只有 pc3 没有**
    chiller.plr           0.000 vs 0.983     部分负荷率 —— 冷机负荷的直接指示量
    coolpump.consumption  0.000 vs 0.979     （zx / pb1 / pc2 也没有，模型见过）
    coldpump.consumption  0.000 vs 0.940     （同上）

前三条共同的性质：**它们都是负荷水平的直接证据**。若假设成立，模型在 pc3 上
应当仍能靠温度/频率/开机位跟住动态（实测 corr 0.883 ✔），但不知道「量级」——
于是输出被压成近似常数（实测 pred std 500 vs 真值 1890 ✔）。

本工具在**熟悉的站**上按顺序逐条掩掉，看 R² 沿哪一条掉下去，以及全掩之后
是否复现 pc3 的 −1.8。掩到位仍不崩 -> 假设再次被否，须另找。

    python tools/diag_avail_ablate.py --ckpt ... --sites hx,yb3
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

# 与 pc3 的缺失顺序一致：先它独有的，再它与别站共有的
STEPS = [
    ("chiller", "consumption"),
    ("chiller", "plr"),
    ("plant", "tower_out"),
    ("coolpump", "consumption"),
    ("coldpump", "consumption"),
]


def _rows(sd, fam):
    return [i for i, (f, _) in enumerate(sd.sch.token_index) if f == fam]


@torch.no_grad()
def run(model, s, dev, H, batches, masks=()):
    """`masks` 为 [(family, field)]，在**历史窗与推演段整段**上掩掉。

    三个张量都要改：`seq_x` 进编码器、`seq_raw` 进物理解码器、`seq_avail`
    是掩码本身。只改一个会让两条通路看到不同的世界（§5.3 的同一条约束）。
    """
    s.ds_va.set_H(H)
    plan = [(_rows(s.sd, fam), S.CANON_FIELDS[fam].index(fld)) for fam, fld in masks]
    yt, yp = [], []
    for i, b in enumerate(s.ds_va.epoch(64, shuffle=False, drop_last=False)):
        # 0 = 不限。窗口按时间升序、epoch 不打乱，截断取到的是**最早**
        # 的一批窗口而非随机样本（§13 #56）—— 诊断工具同样会被它带偏。
        if batches and i >= batches:
            break
        if plan:
            b = dict(b)
            for k in ("seq_avail", "seq_x", "seq_raw"):
                b[k] = b[k].clone()
            for rows, k in plan:
                if not rows:
                    continue
                for key in ("seq_avail", "seq_x", "seq_raw"):
                    b[key][:, :, rows, k] = 0.0
        o = model.rollout(b, desc=s.ctx.desc, stat_rel=s.ctx.stat_rel,
                          type_id=s.ctx.type_id,
                          site_ctx=s.ctx.ctx(b["P_plant"].shape[0], dev),
                          H=H, W=s.ctx.W, reanchor_p=0.0, scales=s.scales)
        yt.append(b["P_plant"][:, 0].cpu().numpy())
        yp.append(o["preds"][0]["P_plant"].cpu().numpy())
    yt, yp = np.concatenate(yt), np.concatenate(yp)
    m = all_metrics(yt, yp)
    m["p_std"] = float(yp.std()); m["t_std"] = float(yt.std())
    m["p_mean"] = float(yp.mean()); m["t_mean"] = float(yt.mean())
    m["corr"] = float(np.corrcoef(yt, yp)[0, 1])
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sites", default="hx,yb3,pc2")
    ap.add_argument("--hold-out", default="hx,pc3")
    ap.add_argument("--H", type=int, default=8)
    ap.add_argument("--batches", type=int, default=0,
                    help="0 = 评完整个 split。正式判读必须用 0（§13 #56）")
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

    for s in sites:
        print(f"\n=== {s.name}{'（留出/零样本）' if s.held_out else '（训练站）'} ===")
        print(f"{'累积掩掉':<44}{'R²':>9}{'MAE':>9}{'corr':>8}"
              f"{'预测std':>10}{'真值std':>9}")
        print("-" * 90)
        for n in range(len(STEPS) + 1):
            masks = STEPS[:n]
            m = run(model, s, dev, a.H, a.batches, masks)
            lbl = "（原样）" if n == 0 else f"+{STEPS[n-1][0]}.{STEPS[n-1][1]}"
            print(f"{lbl:<44}{m['R2']:>9.3f}{m['MAE']:>9.1f}{m['corr']:>8.3f}"
                  f"{m['p_std']:>10.0f}{m['t_std']:>9.0f}")
    print()
    print("参照 pc3 零样本实测：R²=−1.82  MAE 2600  corr 0.883  "
          "预测std 500  真值std 1890")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
