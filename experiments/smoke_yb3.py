"""yb3 单站 smoke：形状、硬约束、置换等变、梯度连通、时延。

不训练，只验证结构正确。跑通了才进 P2。
    python experiments/smoke_yb3.py [--device cuda] [--train-steps 50]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from physwm.data.dataset import RolloutWindows, WindowSpec, build_site_bundle  # noqa: E402
from physwm.data.descriptors import DescNormalizer, compute_descriptors  # noqa: E402
from physwm.data import schema as S  # noqa: E402
# 量纲推断已移入库（多站训练需要，见 physwm/data/scales.py）；此处仅再导出
from physwm.data.scales import (infer_dt_evap_scale, infer_k_mcp,  # noqa: E402,F401
                                infer_mcp_cool_scale, infer_mcp_scale,
                                infer_scales, infer_site_scales, infer_w_scale)
from physwm.model.decoder import ETA_MAX, ETA_MIN, check_hard_constraints  # noqa: E402
from physwm.model.world_model import ModelConfig, WorldModel  # noqa: E402

CSV = ROOT / "data" / "yb3_topology_complete.csv"


ETA_TYP = 0.30   # 典型卡诺效率，仅用于初始化标定，训练会自己调


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--H", type=int, default=8)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--train-steps", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    dev = torch.device(a.device)

    print("=" * 72)
    print("1) 数据底座")
    spec = WindowSpec()
    b = build_site_bundle(CSV, "yb3", spec, out_dir=ROOT / "artifacts")
    sd, tr = b["data"], b["splits"]["train"]
    trm = b["train_rows"]
    info = b["info"]
    print(f"   N={info['N']} F={info['F']} T={info['T']}  段={info['n_segments']}")
    print(f"   窗口 train/val/test = {info['n_windows']['train']}/"
          f"{info['n_windows']['val']}/{info['n_windows']['test']}")
    print(f"   功率口径 {info['power_src']}")

    print("\n2) 设备描述符")
    t = time.time()
    db = compute_descriptors(sd.x, sd.avail, sd.sch, trm)
    dn = DescNormalizer.fit([db.desc])
    desc = torch.from_numpy(dn.apply(db.desc)).to(dev)
    stat_rel = torch.from_numpy(db.stat_rel).to(dev)
    type_id = torch.from_numpy(db.type_id).to(dev)
    print(f"   desc {tuple(desc.shape)} stat_rel {tuple(stat_rel.shape)}  {time.time()-t:.1f}s")
    ci = [i for i, n in enumerate(db.names) if n.startswith("chiller_")]
    cp = [i for i, n in enumerate(db.names) if n.startswith("coolpump_")]
    M = db.stat_rel[np.ix_(ci, cp, [0])][:, :, 0]
    d_, o_ = np.diag(M), M[~np.eye(len(ci), dtype=bool)]
    print(f"   联锁 phi 冷机x冷却泵: 对角中位 {np.median(d_):.3f} / 非对角中位 {np.median(o_):.3f}")

    print("\n3) 模型")
    cfg = ModelConfig()
    model = WorldModel(cfg, sd.sch.n_dev, sd.F).to(dev)
    for k, v in model.n_params().items():
        print(f"   {k:14s} {v:>10,}")
    sc = infer_scales(sd, trm)
    sc["w_scale"] = infer_w_scale(sd, trm)
    sc["dt_evap_scale"] = infer_dt_evap_scale(sd, trm)
    sc["mcp_scale"] = infer_mcp_scale(sd, trm, sc["w_scale"], sc["dt_evap_scale"])
    sc["mcp_cool_scale"] = infer_mcp_cool_scale(sd, trm, sc["w_scale"])
    sc["k_mcp"] = infer_k_mcp(sd, trm, sc["w_scale"], sc["dt_evap_scale"])
    model.decoder.set_scales(**sc)
    print("   量纲: " + "  ".join(f"{k}={v:.1f}" for k, v in sc.items()))

    ds = RolloutWindows(sd, tr, spec, b["norm"])
    ds.set_H(a.H)
    dl = torch.utils.data.DataLoader(ds, batch_size=a.batch, shuffle=True)
    batch = {k: v.to(dev) for k, v in next(iter(dl)).items()}
    ctx = torch.zeros(a.batch, cfg.d_ctx, device=dev)

    print(f"\n4) rollout H={a.H}")
    t = time.time()
    out = model.rollout(batch, desc=desc, stat_rel=stat_rel, type_id=type_id,
                        site_ctx=ctx, H=a.H, W=spec.W, need_z_true=True)
    print(f"   前向 {time.time()-t:.2f}s | z {tuple(out['z'].shape)} "
          f"alpha {tuple(out['alpha'].shape)} bias {tuple(out['bias'].shape)}")
    p0 = out["preds"][0]
    for k in ["approx", "tower_out", "cool_dt", "cool_out", "q_evap", "eta",
              "cop_carnot", "cop_actual", "w_chiller", "q_cond", "P_plant"]:
        v = p0[k]
        print(f"   {k:11s} {str(tuple(v.shape)):10s} med={v.median():9.2f} "
              f"min={v.min():9.2f} max={v.max():9.2f}")
    print(f"   真值 P_plant med={batch['P_plant'].median():.1f} kW | "
          f"预测 med={p0['P_plant'].median():.1f} kW")

    print("\n5) 硬约束自检（必须全为 0）")
    ok = True
    with torch.no_grad():
        on_ch = model.physical_inputs(batch["seq_raw"][:, spec.W], type_id)["on_ch"]
        for k, v in check_hard_constraints(
                {kk: vv.detach() for kk, vv in p0.items()}, on=on_ch).items():
            print(f"   {k:18s} {v:.3e}")
            ok &= v < 1e-5
    print(f"   -> {'全部结构性满足' if ok else '**有违例**'}")

    print("\n6) 梯度连通与 Lipschitz")
    loss = sum(((pp["P_plant"] - batch["P_plant"][:, h]) ** 2).mean()
               for h, pp in enumerate(out["preds"]))
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9)
    nog = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    print(f"   loss={loss.item():.4e} grad_norm={gn:.3e} 无梯度参数={len(nog)}")
    print(f"   Lipschitz 上界 L={model.transition.lipschitz_estimate():.3f}  "
          f"(delta={cfg.delta}; L<=1 才能保证误差线性增长)")

    print("\n7) 置换等变性（交换 chiller_0 / chiller_1）")
    perm = torch.arange(sd.N, device=dev)
    cidx = (type_id == S.FAMILIES.index("chiller")).nonzero().squeeze(-1)
    i0, i1 = int(cidx[0]), int(cidx[1])
    perm[i0], perm[i1] = i1, i0
    model.eval()
    with torch.no_grad():
        hx, ha = batch["seq_x"][:, :spec.W], batch["seq_avail"][:, :spec.W]
        z1, _ = model.encode(hx, ha, desc=desc, stat_rel=stat_rel,
                             type_id=type_id, site_ctx=ctx)
        z2, _ = model.encode(hx[:, :, perm], ha[:, :, perm],
                             desc=desc[perm], stat_rel=stat_rel[perm][:, perm],
                             type_id=type_id[perm], site_ctx=ctx)
    rel = float((z2 - z1[:, perm]).abs().max() / z1.abs().max())
    print(f"   ||E(Pz)-P.E(z)||_inf / ||z||_inf = {rel:.3e}  "
          f"{'通过' if rel < 1e-4 else '**破坏**'}")

    print("\n8) 推理时延（batch=1）")
    ds1 = RolloutWindows(sd, tr, spec, b["norm"])
    ds1.set_H(36)
    b1 = {k: v[None].to(dev) for k, v in ds1[0].items()}
    with torch.no_grad():
        model.rollout(b1, desc=desc, stat_rel=stat_rel, type_id=type_id,
                      site_ctx=ctx[:1], H=36, W=spec.W)
        t = time.time()
        for _ in range(5):
            model.rollout(b1, desc=desc, stat_rel=stat_rel, type_id=type_id,
                          site_ctx=ctx[:1], H=36, W=spec.W)
        dt = (time.time() - t) / 5
    print(f"   H=36 整段 {dt*1000:.0f} ms -> 单步 {dt/36*1000:.2f} ms（目标 <=50ms）")

    if a.train_steps > 0:
        print(f"\n9) 过拟合单 batch {a.train_steps} 步（检验可学）")
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=2e-3)
        ds.set_H(a.H)
        for i in range(a.train_steps):
            opt.zero_grad()
            o = model.rollout(batch, desc=desc, stat_rel=stat_rel, type_id=type_id,
                              site_ctx=ctx, H=a.H, W=spec.W)
            tgt = batch["P_plant"][:, :a.H]
            pred = torch.stack([pp["P_plant"] for pp in o["preds"]], 1)
            l = ((pred - tgt) ** 2).mean() / (tgt.std() ** 2 + 1e-6)
            l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            if i % max(a.train_steps // 8, 1) == 0 or i == a.train_steps - 1:
                mae = (pred - tgt).abs().mean().item()
                print(f"   step {i:4d}  归一化MSE={l.item():.4f}  MAE={mae:8.1f} kW")
        with torch.no_grad():
            hc = check_hard_constraints({k: v.detach() for k, v in o["preds"][0].items()})
        print(f"   训练后硬约束仍满足: {all(v < 1e-5 for v in hc.values())}")

    print("\n" + "=" * 72)
    print("smoke 完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
