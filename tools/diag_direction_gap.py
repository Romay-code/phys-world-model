"""诊断：训练用的平方 hinge 与评测用的 dir_vr 是不是在测同一件事。

**怀疑**：`dir_vr` 数的是违例**个数**（任意幅度都算一次），训练损失罚的是
违例**幅度的平方**。若 h=36 处响应被推演漂移淹没、幅度普遍很小，则平方后
惩罚可忽略，而个数照样计满 —— 损失下降了，指标却纹丝不动。

这能解释 P4-D 那个说不通的现象：**在 h=36 上直接施加惩罚，
h=36 的 dir_vr 仍是 0.145 / 0.625。**

判据：若「违例样本的 |d| 中位」显著小于「合规样本的 |d| 中位」，假设成立 ——
说明违例集中在幅度极小处，正是平方 hinge 看不见的地方。

    python tools/diag_direction_gap.py --ckpt experiments/results/p4d_dirh_0.1/model_seed0.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from physwm.data.dataset import GPUWindows  # noqa: E402
from physwm.eval.direction import default_specs, make_intervene  # noqa: E402
from physwm.model.world_model import ModelConfig, WorldModel  # noqa: E402


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--H", type=int, default=48)
    ap.add_argument("--hs", default="1,12,36")
    ap.add_argument("--batches", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=32)
    a = ap.parse_args()

    import train_yb3 as T
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")

    class _A:
        W, H, split_mode, refresh_desc = 16, a.H, "blocked", False
    b, sd, splits, spec, ctx, trm = T.build(_A(), dev)

    model = WorldModel(ModelConfig(), sd.sch.n_dev, sd.F).to(dev)
    sc = T.infer_scales(sd, trm)
    sc["w_scale"] = T.infer_w_scale(sd, trm)
    sc["dt_evap_scale"] = T.infer_dt_evap_scale(sd, trm)
    sc["mcp_scale"] = T.infer_mcp_scale(sd, trm, sc["w_scale"], sc["dt_evap_scale"])
    sc["mcp_cool_scale"] = T.infer_mcp_cool_scale(sd, trm, sc["w_scale"])
    sc["k_mcp"] = T.infer_k_mcp(sd, trm, sc["w_scale"], sc["dt_evap_scale"])
    model.decoder.set_scales(**sc)
    st = torch.load(a.ckpt, map_location=dev, weights_only=False)
    model.load_state_dict(st.get("model", st) if isinstance(st, dict) else st)
    model.eval()

    ds = GPUWindows(sd, splits["test"], spec, b["norm"], dev)
    hs = [int(x) for x in a.hs.split(",")]
    ds.set_H(max(hs))
    ns = torch.as_tensor(b["norm"].scale, dtype=torch.float32, device=dev)
    kw = dict(desc=ctx.desc, stat_rel=ctx.stat_rel, type_id=ctx.type_id,
              W=ctx.W, H=max(hs), reanchor_p=0.0, key_mask=ctx.key_mask)

    acc: dict = {}
    for i, bt in enumerate(ds.epoch(a.batch_size, shuffle=False, drop_last=False)):
        if i >= a.batches:
            break
        B = bt["P_plant"].shape[0]
        sctx = ctx.ctx(B, dev)
        base = model.rollout(bt, site_ctx=sctx, **kw)["preds"]
        for s in default_specs():
            fn = make_intervene(sd.sch, ns, s, ctx.type_id)
            b2 = dict(bt)
            b2["seq_x"], b2["seq_raw"] = fn(bt["seq_x"].clone(), bt["seq_raw"].clone(), ctx.W)
            pert = model.rollout(b2, site_ctx=sctx, **kw)["preds"]
            for h in hs:
                for tgt, sign in s.targets.items():
                    if tgt not in base[h - 1]:
                        continue
                    y0 = base[h - 1][tgt]
                    d = ((pert[h - 1][tgt] - y0) / y0.std().clamp_min(1e-6)).reshape(-1)
                    live = (y0.reshape(-1).abs() > 1e-6)
                    d = d[live]
                    bad = (-float(sign) * d) > 0
                    acc.setdefault((s.name, tgt, h), []).append(
                        (d[bad].abs().cpu().numpy(), d[~bad].abs().cpu().numpy()))

    print(f"{'动作维':<16}{'目标':<15}{'h':>4}{'dir_vr':>9}"
          f"{'违例|d|中位':>12}{'合规|d|中位':>12}{'比值':>8}{'违例占平方和':>13}")
    print("-" * 92)
    for (name, tgt, h), chunks in acc.items():
        bad = np.concatenate([c[0] for c in chunks])
        ok = np.concatenate([c[1] for c in chunks])
        n = len(bad) + len(ok)
        if n == 0:
            continue
        vr = len(bad) / n
        mb = float(np.median(bad)) if bad.size else 0.0
        mo = float(np.median(ok)) if ok.size else 0.0
        share = float((bad ** 2).sum() / max((bad ** 2).sum() + (ok ** 2).sum(), 1e-12))
        print(f"{name:<16}{tgt:<15}{h:>4}{vr:>9.3f}{mb:>12.4f}{mo:>12.4f}"
              f"{(mb / mo if mo > 0 else float('nan')):>8.2f}{share:>13.3f}")
    print()
    print("若「违例|d|中位」远小于「合规|d|中位」（比值 << 1），说明违例集中在")
    print("幅度极小处 —— 平方 hinge 对它们几乎没有梯度，而 dir_vr 照样计满。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
