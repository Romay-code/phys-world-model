"""单步训练的分段计时，定位瓶颈。

    python tools/profile_step.py --device cuda --H 1
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from physwm.data.dataset import GPUWindows, WindowSpec, build_site_bundle  # noqa: E402
from physwm.data.descriptors import DescNormalizer, compute_descriptors  # noqa: E402
from physwm.model.world_model import ModelConfig, WorldModel  # noqa: E402
from physwm.train.loop import setup_backend  # noqa: E402
from physwm.train.losses import (LossWeights, ObsLoss, TargetScales,  # noqa: E402
                                 compute_loss, lipschitz_penalty)
from smoke_yb3 import (infer_dt_evap_scale, infer_k_mcp, infer_mcp_cool_scale, infer_mcp_scale, infer_scales,  # noqa: E402
                       infer_w_scale)


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def timeit(fn, dev, n=10, warm=3):
    for _ in range(warm):
        fn()
    sync(dev)
    t = time.time()
    for _ in range(n):
        fn()
    sync(dev)
    return (time.time() - t) / n * 1000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--H", type=int, default=1)
    ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args()
    dev = torch.device(a.device)
    setup_backend()

    spec = WindowSpec()
    b = build_site_bundle(ROOT / "data" / "yb3_topology_complete.csv", "yb3", spec)
    sd, tr = b["data"], b["splits"]["train"]
    trm = b["train_rows"]
    db = compute_descriptors(sd.x, sd.avail, sd.sch, trm)
    dn = DescNormalizer.fit([db.desc])
    desc = torch.from_numpy(dn.apply(db.desc)).to(dev)
    stat_rel = torch.from_numpy(db.stat_rel).to(dev)
    type_id = torch.from_numpy(db.type_id).to(dev)

    model = WorldModel(ModelConfig(), sd.sch.n_dev, sd.F).to(dev)
    _sc = infer_scales(sd, trm); _sc["w_scale"] = infer_w_scale(sd, trm); _sc["dt_evap_scale"] = infer_dt_evap_scale(sd, trm); _sc["mcp_scale"] = infer_mcp_scale(sd, trm, _sc["w_scale"], _sc["dt_evap_scale"]); _sc["mcp_cool_scale"] = infer_mcp_cool_scale(sd, trm, _sc["w_scale"]); _sc["k_mcp"] = infer_k_mcp(sd, trm, _sc["w_scale"], _sc["dt_evap_scale"])
    model.decoder.set_scales(**_sc)
    wts = LossWeights()
    obs = ObsLoss(sd, TargetScales.fit(sd, trm), wts, dev)

    gw = GPUWindows(sd, tr, spec, b["norm"], dev)
    gw.set_H(a.H)
    idx = torch.arange(a.batch, device=dev)
    batch = gw.batch(idx)
    ctx = torch.zeros(a.batch, 8, device=dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)

    print(f"device={dev} H={a.H} batch={a.batch} N={sd.N}")
    print(f"{'阶段':28s} {'ms':>9s}")
    print("-" * 40)

    print(f"{'取 batch (GPU gather)':28s} {timeit(lambda: gw.batch(idx), dev, 20):9.2f}")

    def enc():
        with torch.no_grad():
            model.encode(batch["seq_x"][:, :spec.W], batch["seq_avail"][:, :spec.W],
                         desc=desc, stat_rel=stat_rel, type_id=type_id, site_ctx=ctx)
    print(f"{'编码器 (no_grad)':28s} {timeit(enc, dev):9.2f}")

    def fwd():
        with torch.no_grad():
            model.rollout(batch, desc=desc, stat_rel=stat_rel, type_id=type_id,
                          site_ctx=ctx, H=a.H, W=spec.W)
    print(f"{'整 rollout 前向 (no_grad)':28s} {timeit(fwd, dev):9.2f}")

    def lip():
        lipschitz_penalty(model.transition, 1.0).backward()
        model.zero_grad(set_to_none=True)
    print(f"{'Lipschitz 惩罚 (含反向)':28s} {timeit(lip, dev):9.2f}")

    def lip_nograd():
        with torch.no_grad():
            lipschitz_penalty(model.transition, 1.0)
    print(f"{'Lipschitz 惩罚 (仅前向)':28s} {timeit(lip_nograd, dev):9.2f}")

    for name, lam_lip, lam_lat in (("完整训练步", 0.01, 0.1),
                                   ("去掉 Lipschitz", 0.0, 0.1),
                                   ("去掉 Lip+隐一致", 0.0, 0.0)):
        w = LossWeights(lam_lip=lam_lip, lam_lat=lam_lat)

        def step(w=w):
            opt.zero_grad(set_to_none=True)
            out = model.rollout(batch, desc=desc, stat_rel=stat_rel, type_id=type_id,
                                site_ctx=ctx, H=a.H, W=spec.W,
                                need_z_true=w.lam_lat > 0)
            loss, _ = compute_loss(model, out, batch, obs, w, spec.W)
            loss.backward()
            opt.step()
        print(f"{name:28s} {timeit(step, dev, 8):9.2f}")

    n_batch = len(tr) // a.batch
    print(f"\n训练集 {len(tr)} 窗口 -> {n_batch} batch/epoch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
