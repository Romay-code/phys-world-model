"""少样本适配 + 零样本守门（设计文档 §6 P5 第 2、3 条，验收项 M2）。

## 为什么必须有这条路

P5-A 的 pc3 零样本崩掉（R² −1.64），根因是它属于**能力 D 档，而 D 档
只有它一个成员**（`data.registry`）—— 留出它等于训练集里没有任何一个站有
那个模式，考的不是「零样本泛化」而是「外推到一个空档」。

D 档 n=1 时只剩两条路：合成该档（`avail_dropout`），或者**给它一点自己的
数据**。后者才是部署时的真问题：真上线一个全聚合的冷站，采一周数据不难。

## FiLM 的 ctx 通路是死的（§13 #54）—— 必须先知道这件事

`layers.FiLM` 的设计意图写在它自己的 docstring 里：「site_ctx 由站级统计量
算出」。但 `Context.ctx()` 一直返回**全零**，站级统计量从来没实现过。
后果是 `film.*.0.weight`（乘 ctx 的那一层）**恒收不到梯度**，而它 ndim=2、
落在优化器的 weight_decay 组里，于是被 1e-4 的衰减在 10000 步里推成 denormal：

    encoder.blocks.0.film.g.0.weight   |w| 均值 = 5.1e-41
    encoder.blocks.3.film.b.0.weight   |w| 均值 = 5.1e-41

**于是 `g(ctx) = W2·GELU(b1) + b2` 与 ctx 无关，`dg/dctx ≈ 0`。**
实测：`site_ctx` 机制训 30 步，`ctx_norm` 恒为 **0.000**，
适配臂与零样本臂的 R² 逐位相同（−64.4886）。

这与 §13 #24（eta 零梯度）、#6（Lipschitz 惩罚恒为 0）是同一类：
**代码里有这个机制、却从没被真正激活，然后静默地退化成没有。**

所以本模块做两件事：

  1. `mechanism="film_bias"` 作为**默认** —— 调 FiLM 的输出偏置，
     那些参数是活的（|w| 1e-2 量级）。
  2. **梯度存活守卫**：第一次反向后立刻检查被调参数的梯度范数，
     全为零就**直接报错**并点名。有了它，上面那个坑第一次跑就会喊出来，
     而不是靠人盯着「两臂 R² 怎么一模一样」。

`mechanism="site_ctx"` 保留，供 FiLM 通路修好后使用（修法见 §13 #54：
`site_ctx` 接真实站级统计量，且把 `film.*.0.weight` 移出 weight_decay 组）。

## 口径：三条纪律，少一条这个数就不能信

1. **目标站改用 `chronological` 划分。** 适配数据取**最早**的一段、
   评测取**最晚**的一段 —— 模拟真实上线（先采一段再上线）。
   随机抽等于偷看未来，而且会把「冬季适配、夏季部署」这个真实失败模式
   平均掉、看不见。§13 #7 反对 chronological 是针对**选模**说的；
   这里要测的恰恰就是那个分布漂移。

2. **守门只能看部署时拿得到的数据。** 适配段再切 80/20，用后 20%（`gate`）
   比较零样本与适配，**绝不看 test**。拿 test 选就是把门限做成了后见之明。

3. **守门是「不劣于」而非「更好」。** M2 要求「任意样本规模下精度不低于
   零样本水平」，所以平局要判给零样本 —— 适配必须**赢**才被采纳。
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np
import torch

from ..data.dataset import GPUWindows
from ..eval.metrics import all_metrics

# 一天的步数：900 s 控制周期 -> 96 步/天
STEPS_PER_DAY = 96

MECHANISMS = ("film_bias", "site_ctx")


@dataclass
class FewShotConfig:
    n_days: float = 7.0        # 适配数据天数。0 = 纯零样本（不训练）
    lr: float = 3e-2
    steps: int = 300
    H: int = 8                 # 适配用的推演步数，短一些以省时
    batch_size: int = 32
    gate_frac: float = 0.20    # 适配段末尾留作守门的比例
    seed: int = 0
    # 守门段少于这么多窗口就**一律不采纳** —— M2 明文要求「少样本数据不足以
    # 支撑可靠适配时自动保留预训练权重」，这就是「不足」的判据。
    min_gate_windows: int = 64
    # 适配必须把守门段 MAE 压低这个比例才算赢。留余量是因为守门段短、
    # 自相关强，小幅优势多半是噪声。
    gate_margin: float = 0.05
    # 默认 film_bias：`site_ctx` 在现有权重上是死通路（§13 #54）
    mechanism: str = "film_bias"

    def __post_init__(self) -> None:
        if self.mechanism not in MECHANISMS:
            raise ValueError(f"未知的适配机制 {self.mechanism}，可选 {MECHANISMS}")

    @property
    def n_rows(self) -> int:
        return int(round(self.n_days * STEPS_PER_DAY))


@dataclass
class Adaptation:
    """一次适配的结果。`ctx_vec` 与 `params` 至多一个非空。

    `base` 保留预训练值，供守门判负时原样还原 —— 这不是可选项：
    守门的语义就是「适配没赢就当没发生过」。
    """

    ctx_vec: torch.Tensor | None = None
    params: dict[str, torch.Tensor] | None = None
    base: dict[str, torch.Tensor] = field(default_factory=dict)
    n_params: int = 0

    def install(self, model, use_adapted: bool) -> None:
        """把适配值（或预训练值）写回模型。`ctx_vec` 机制无需写回。"""
        if not self.params:
            return
        src = self.params if use_adapted else self.base
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in src:
                    p.copy_(src[n])

    def ctx_for(self, ctx, use_adapted: bool):
        vec = self.ctx_vec if (use_adapted and self.ctx_vec is not None) else None
        return dataclasses.replace(ctx, ctx_vec=vec)


def film_bias_names(model) -> list[str]:
    """FiLM 的**输出偏置** —— 在现有权重上是活的那部分。

    只取 `.2.bias`（gamma/beta 末层的偏置），不碰 `.0.weight`（已 denormal）
    也不碰 `.2.weight`（d×hidden，参数量大一个量级）。
    """
    return [n for n, _ in model.named_parameters()
            if ".film." in n and n.endswith(".2.bias")]


def split_adapt_gate(windows: np.ndarray, cfg: FewShotConfig
                     ) -> tuple[np.ndarray, np.ndarray]:
    """从**最早**的 `n_days` 天里切出 adapt / gate 两段。

    取「最早的一段」而不是「随机 n 个」，见模块文档纪律 1。
    """
    if len(windows) == 0:
        return windows, windows
    w = windows[np.argsort(windows[:, 1])]     # 按锚点 t0 排序
    keep = w[w[:, 1] < int(w[0, 1]) + cfg.n_rows]
    if len(keep) < 8:
        return keep, keep[:0]
    cut = int(len(keep) * (1.0 - cfg.gate_frac))
    return keep[:cut], keep[cut:]


@torch.no_grad()
def _eval_windows(model, site, ctx, windows, device, H: int, batch_size: int = 64,
                  max_batches: int = 40) -> dict:
    if len(windows) == 0:
        return {"R2": float("nan"), "MAE": float("nan"), "n": 0}
    ds = GPUWindows(site.sd, windows, site.spec, site.norm, device, H=H)
    ds.set_H(H)
    yt, yp = [], []
    for i, b in enumerate(ds.epoch(batch_size, shuffle=False, drop_last=False)):
        if i >= max_batches:
            break
        o = model.rollout(b, desc=site.ctx.desc, stat_rel=site.ctx.stat_rel,
                          type_id=site.ctx.type_id,
                          site_ctx=ctx.ctx(b["P_plant"].shape[0], device),
                          H=H, W=site.ctx.W, reanchor_p=0.0, scales=site.scales)
        yt.append(b["P_plant"][:, 0].cpu().numpy())
        yp.append(o["preds"][0]["P_plant"].cpu().numpy())
    m = all_metrics(np.concatenate(yt), np.concatenate(yp))
    m["n"] = int(sum(len(a) for a in yt))
    return m


def _delta_norm(ad: Adaptation, names, tuned) -> float:
    if ad.ctx_vec is not None:
        return float(ad.ctx_vec.detach().norm())
    with torch.no_grad():
        return float(torch.sqrt(sum(((p - ad.base[n]) ** 2).sum()
                                    for n, p in zip(names, tuned))))


def _assert_gradient_alive(names, tuned, mechanism: str, tol: float = 1e-12) -> None:
    """被调参数必须真的收到梯度 —— 否则「适配」是个空转。

    §13 #24 / #6 / #54 全是同一类：机制在代码里、却没被激活，
    而训练日志、损失曲线、指标**三者都看不出来**（损失照样随 batch 抖动）。
    这条守卫把它变成第一步就报错。
    """
    dead = [n for n, p in zip(names, tuned)
            if p.grad is None or float(p.grad.abs().sum()) <= tol]
    if len(dead) < len(names):
        return
    extra = ("\n`site_ctx` 在现有权重上确实是死通路（§13 #54）："
             "FiLM 的 `film.*.0.weight` 已被 weight_decay 推成 denormal (~5e-41)，"
             "因为预训练全程 `site_ctx ≡ 0`、它收不到任何梯度。"
             "改用 mechanism='film_bias'，或先修好 FiLM 通路再重训。"
             if mechanism == "site_ctx" else f"\n梯度为零的参数：{dead[:5]}")
    raise RuntimeError(
        f"适配机制 '{mechanism}' 的全部 {len(names)} 个参数梯度为零，适配是空转。" + extra)


def adapt(model, site, cfg: FewShotConfig, device, windows=None):
    """冻结主干，只调很少的参数。返回 (Adaptation, 日志)。

    主干**必须**冻结：目标站只有几百个窗口，放开 3.6e6 个参数一定过拟合，
    而且会毁掉预训练学到的跨站结构 —— 那正是 M2 要保住的东西。
    """
    from .losses import LossWeights, compute_loss

    # 适配期关掉训练正则，但 **lam_evap_bal 必须保留**：
    # §13 #35 —— soft 模式配 lam_evap_bal=0 会静默复现 #24 的 eta 零梯度，
    # `compute_loss` 在 grad 打开时对此直接报错。
    # lam_lat 关掉是因为它要 `need_z_true=True`，适配用不上。
    w0 = LossWeights(**{**site.obs_loss.w.__dict__,
                        "lam_lat": 0.0, "lam_lip": 0.0, "lam_dir": 0.0})

    torch.manual_seed(cfg.seed)
    was_training = model.training
    model.eval()
    for prm in model.parameters():
        prm.requires_grad_(False)

    ad = Adaptation()
    hist: list[dict] = []
    names: list[str] = []
    # **必须 try/finally**：中途抛错（OOM、数据坏、梯度守卫触发）若不恢复
    # requires_grad，模型会停在「全部冻结」，之后的训练**静默地什么都不更新**。
    try:
        if cfg.mechanism == "site_ctx":
            ctx_vec = torch.zeros(site.ctx.site_ctx_dim, device=device,
                                  requires_grad=True)
            tuned, names = [ctx_vec], ["site_ctx"]
            ad.ctx_vec, ad.n_params = ctx_vec, int(ctx_vec.numel())
        else:
            names = film_bias_names(model)
            if not names:
                raise RuntimeError("模型里找不到 FiLM 输出偏置，无法用 film_bias 机制")
            byname = dict(model.named_parameters())
            ad.base = {n: byname[n].detach().clone() for n in names}
            tuned = []
            for n in names:
                byname[n].requires_grad_(True)
                tuned.append(byname[n])
            ad.n_params = int(sum(p.numel() for p in tuned))

        opt = torch.optim.Adam(tuned, lr=cfg.lr)
        ds = GPUWindows(site.sd, windows, site.spec, site.norm, device, H=cfg.H)
        ds.set_H(cfg.H)

        checked = False
        for step in range(cfg.steps):
            idx = torch.randint(0, len(ds), (min(cfg.batch_size, len(ds)),),
                                device=device)
            b = ds.batch(idx)
            opt.zero_grad(set_to_none=True)
            B = b["P_plant"].shape[0]
            sctx = (ad.ctx_vec.view(1, -1).expand(B, -1) if ad.ctx_vec is not None
                    else torch.zeros(B, site.ctx.site_ctx_dim, device=device))
            o = model.rollout(b, desc=site.ctx.desc, stat_rel=site.ctx.stat_rel,
                              type_id=site.ctx.type_id, site_ctx=sctx,
                              H=cfg.H, W=site.ctx.W, reanchor_p=0.0,
                              scales=site.scales)
            loss, _ = compute_loss(model, o, b, site.obs_loss, w0, site.ctx.W)
            loss.backward()
            if not checked:
                checked = True
                _assert_gradient_alive(names, tuned, cfg.mechanism)
            opt.step()
            if step % 50 == 0 or step == cfg.steps - 1:
                hist.append({"step": step, "loss": float(loss.detach()),
                             "delta": _delta_norm(ad, names, tuned)})

        if cfg.mechanism == "film_bias":
            byname = dict(model.named_parameters())
            ad.params = {n: byname[n].detach().clone() for n in names}
    finally:
        for prm in model.parameters():
            prm.requires_grad_(True)
        if was_training:
            model.train()

    return ad, {"history": hist, "n_params": ad.n_params,
                "mechanism": cfg.mechanism,
                "n_windows": int(len(windows)) if windows is not None else 0}


def gated_adapt(model, site, cfg: FewShotConfig, device, eval_windows=None) -> dict:
    """少样本适配 + 零样本守门。

    `eval_windows` 是最终报数用的段（目标站的 test），**只在最后用一次**，
    不参与任何选择。
    """
    ev = eval_windows if eval_windows is not None else site.splits["test"]
    zero_ctx = dataclasses.replace(site.ctx, ctx_vec=None)

    def _bail(reason, n_adapt=0):
        m0 = _eval_windows(model, site, zero_ctx, ev, device, cfg.H)
        return {"n_days": cfg.n_days, "adopted": False, "reason": reason,
                "gate_zero": None, "gate_adapt": None, "test_zero": m0,
                "test_final": m0, "n_adapt_windows": n_adapt, "n_params": 0,
                "delta": 0.0}

    if cfg.n_days <= 0:
        return _bail("纯零样本，未适配")

    a_w, g_w = split_adapt_gate(site.splits["train"], cfg)
    if len(a_w) < 8 or len(g_w) < 4:
        return _bail(f"适配窗口不足（adapt {len(a_w)} / gate {len(g_w)}），"
                     f"保留预训练权重", int(len(a_w)))

    ad, log = adapt(model, site, cfg, device, windows=a_w)

    # --- 守门：只看 gate 段，绝不看 test ---
    ad.install(model, use_adapted=False)
    gz = _eval_windows(model, site, ad.ctx_for(site.ctx, False), g_w, device, cfg.H)
    ad.install(model, use_adapted=True)
    ga = _eval_windows(model, site, ad.ctx_for(site.ctx, True), g_w, device, cfg.H)

    # **判据用 MAE，不用 R²。** 守门段是几小时的连续窗口，真值几乎不变，
    # `SS_tot ≈ 0` 让 R² 失去意义 —— 实测 hx 1 天那点的守门 R² 是
    # **−1,625,593**，拿它比大小纯属胡来。MAE 与真值方差无关，是这里唯一
    # 站得住的统计量。平局判给零样本：M2 要的是「不低于」，适配必须赢。
    enough = len(g_w) >= cfg.min_gate_windows
    better = bool(enough and np.isfinite(ga["MAE"]) and np.isfinite(gz["MAE"])
                  and ga["MAE"] < gz["MAE"] * (1.0 - cfg.gate_margin))

    ad.install(model, use_adapted=False)
    tz = _eval_windows(model, site, ad.ctx_for(site.ctx, False), ev, device, cfg.H)
    if better:
        ad.install(model, use_adapted=True)
        tf = _eval_windows(model, site, ad.ctx_for(site.ctx, True), ev, device, cfg.H)
    else:
        tf = tz
    # 评完一律还原，不把本预算点的适配带到下一个点
    ad.install(model, use_adapted=False)

    return {"n_days": cfg.n_days, "adopted": better,
            "reason": ("适配在 gate 段 MAE 更低，采纳" if better
                       else (f"守门段仅 {len(g_w)} 个窗口 < {cfg.min_gate_windows}，"
                             f"数据不足以支撑可靠适配，保留预训练权重（M2 守门）"
                             if not enough else
                             "适配未在 gate 段以 MAE 胜出，保留预训练权重（零样本守门生效）")),
            "gate_zero": gz, "gate_adapt": ga, "test_zero": tz, "test_final": tf,
            "n_adapt_windows": int(len(a_w)), "n_params": ad.n_params,
            "delta": log["history"][-1]["delta"] if log["history"] else 0.0,
            "adapt_log": log}


def curve(model, site, device, days=(0.0, 1.0, 3.0, 7.0, 14.0, 30.0),
          **cfg_kw) -> list[dict]:
    """少样本曲线。`days=0` 那点即零样本，是守门的参照线。"""
    return [gated_adapt(model, site, FewShotConfig(n_days=d, **cfg_kw), device)
            for d in days]


def gate_holds(rows: list[dict], tol: float = 1e-6) -> bool:
    """M2 守门判据：**任意**样本规模下都不低于零样本。

    守门失效的形态是曲线上某一点低于零样本 —— 哪怕别的点都更好。
    """
    for r in rows:
        z, f = r["test_zero"]["R2"], r["test_final"]["R2"]
        if np.isfinite(z) and np.isfinite(f) and f < z - tol:
            return False
    return True
