"""诊断：零样本失败是「标定」问题还是「表征」问题。

`tools/probe_avail_shift.py` 已否掉「缺失通道分布外」这一解释：
hx 同样是零样本站，把它的冷机功率通道整体掩掉也只从 0.938 掉到 0.851，
远到不了 pc3 的 −1.82。**掩掉通道的杀伤力跟着「该站训练时有多依赖它」走，
与零样本与否无关。**

于是问题变成：pc3 的预测究竟长什么样？

    corr 高、斜率/截距不对   -> 标定问题。模型看懂了这个站，只是量纲/偏置错了，
                              一个逐站的仿射改正就能救，属于「零样本 + 现场估量纲」。
    corr 低                  -> 表征问题。模型根本没看懂这个站，得靠少样本适配。

两者的处置完全不同，所以必须先分开。报三个数：

    R²            原样
    R²_scaled     只允许一个比例因子（过原点）后的上界
    R²_affine     允许比例 + 偏置（即 corr²）后的上界 —— 标定能达到的天花板

    python tools/diag_zeroshot_calib.py --ckpt ... --sites hx,pc3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from physwm.data.dataset import WindowSpec  # noqa: E402
from physwm.model.world_model import ModelConfig, WorldModel  # noqa: E402
from physwm.train.multisite import build_sites  # noqa: E402


@torch.no_grad()
def collect(model, s, dev, H, batches):
    s.ds_va.set_H(H)
    yt, yp = [], []
    for i, b in enumerate(s.ds_va.epoch(64, shuffle=False, drop_last=False)):
        # 0 = 不限。窗口按时间升序、epoch 不打乱，截断取到的是**最早**
        # 的一批窗口而非随机样本（§13 #56）—— 诊断工具同样会被它带偏。
        if batches and i >= batches:
            break
        o = model.rollout(b, desc=s.ctx.desc, stat_rel=s.ctx.stat_rel,
                          type_id=s.ctx.type_id,
                          site_ctx=s.ctx.ctx(b["P_plant"].shape[0], dev),
                          H=H, W=s.ctx.W, reanchor_p=0.0, scales=s.scales)
        yt.append(b["P_plant"][:, 0].cpu().numpy())
        yp.append(o["preds"][0]["P_plant"].cpu().numpy())
    return np.concatenate(yt), np.concatenate(yp)


def stats(yt, yp):
    yt = yt.astype(np.float64); yp = yp.astype(np.float64)
    ss_tot = float(((yt - yt.mean()) ** 2).sum())
    r2 = 1.0 - float(((yt - yp) ** 2).sum()) / ss_tot
    # 只允许一个比例因子（过原点）
    k = float((yt * yp).sum() / max((yp * yp).sum(), 1e-12))
    r2_k = 1.0 - float(((yt - k * yp) ** 2).sum()) / ss_tot
    # 允许比例 + 偏置 == corr²
    c = float(np.corrcoef(yt, yp)[0, 1])
    return {"R2": r2, "R2_scaled": r2_k, "R2_affine": c * c, "corr": c, "k": k,
            "t_mean": float(yt.mean()), "t_std": float(yt.std()),
            "p_mean": float(yp.mean()), "p_std": float(yp.std())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sites", default="yb3,hx,pc2,pc3")
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

    print(f"{'站点':<18}{'角色':<8}{'真值均值':>10}{'真值std':>9}"
          f"{'预测均值':>10}{'预测std':>9}{'corr':>8}{'R²':>9}"
          f"{'R²_比例':>9}{'R²_仿射':>9}{'最优k':>8}")
    print("-" * 110)
    for s in sites:
        yt, yp = collect(model, s, dev, a.H, a.batches)
        r = stats(yt, yp)
        print(f"{s.name:<18}{'留出' if s.held_out else '训练':<8}"
              f"{r['t_mean']:>10.0f}{r['t_std']:>9.0f}{r['p_mean']:>10.0f}"
              f"{r['p_std']:>9.0f}{r['corr']:>8.3f}{r['R2']:>9.3f}"
              f"{r['R2_scaled']:>9.3f}{r['R2_affine']:>9.3f}{r['k']:>8.3f}")
    print()
    print("读法：R² 远低于 R²_仿射 -> 标定问题（模型看懂了，量纲/偏置错）；")
    print("      R²_仿射 本身就低 -> 表征问题（模型没看懂这个站）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
