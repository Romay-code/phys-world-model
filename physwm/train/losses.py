"""损失函数（设计文档 §4.5.2）。

    L = sum_h gamma^(h-1) * sum_k w_k * L_obs^k(o_hat, o; M_loss, w_steady)
      + lam_lat    * sum_h || z_rollout - sg[E(真值窗口)] ||^2
      + lam_lip    * L_Lipschitz(G)
      + lam_mono   * L_mono
      + lam_dir    * L_dir
      + lam_anchor * L_slope_anchor

物理硬约束**不在这里** —— 它们是解码器前向计算的函数形式，结构性满足，无权重。
这里只有软约束。

目标分解 k（各自除以训练集 std 归一）:
    k=1 全站总功率 P_plant        w=3.0   全部站点
    k=2 四类分项功率              w=1.0   按口径裁决
    k=3 逐台冷机功率              w=2.0   on_i=1 且该站有逐台标签
    k=4 中间温度量                w=1.0   按能力矩阵
    k=5 逐塔/逐泵功率             w=0.5   仅有逐设备标签的设备，作三次方律的验证锚
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from ..data import schema as S

FAM_ORDER = list(S.DEVICE_FAMILIES)


@dataclass
class LossWeights:
    k1_total: float = 3.0
    k2_family: float = 1.0
    k3_chiller: float = 2.0
    k4_temp: float = 1.0
    k5_device: float = 0.5
    lam_lat: float = 0.1
    # 默认 0：spectral_norm 已把每层压到 sigma=1，这个惩罚恒为 0 却要花 30% 的
    # 单步耗时。Lipschitz 由 ModelConfig.delta 结构性控制，P3 扫的是 delta。
    lam_lip: float = 0.0
    lam_soft_b: float = 0.0    # L-soft-B：Q_evap 与 PLR 成比例（P4，默认关）
    # eta **水平**的弱先验（只罚 batch 均值，形状不受影响）。
    # eta 的绝对尺度在无流量的站不可观测（§13 #29/#30）：它与 m·cp 只以比值出现。
    # 不加约束时该方向是平的，优化器随机漂移直到撞上区间端点，
    # sigmoid 饱和后连形状一起死掉。这一项给那条平坦方向一点曲率，
    # 含义是「**数据说不出来时，停在物理容许区间的中心**」。
    # 注意它罚的是 logit 偏离 0（即 eta 偏离区间中点 0.40），
    # **不是任何一个站的实测值** —— 不构成跨站标定。
    # 有流量的站走 L-hard-B，真实数据会直接压过这个弱项。
    lam_eta_level: float = 0.01
    # L-soft-B 蒸发侧能量平衡的**软**一致项：head 预测的 dt_evap 与
    # 能量平衡算出的 q_evap/(m·cp) 应当一致。
    #
    # §4.3 明确：硬等式（L-hard-B）只适用于有实测流量的站；其余站走软约束。
    # 实测验证过这条界线：在 yb3（无流量）把它做成硬结构，k4 被压垮并经共享
    # 编码器漏进 P_plant，step1 MAE 60.9 -> 127.8（§13 #33）。
    #
    # 软化之后 eta 仍经 q_evap 拿到梯度（治 #24），但不再硬性支配 cold_back。
    # lam 是**精度 / 物理一致性的旋钮**，P4 扫它出权衡曲线。
    lam_evap_bal: float = 0.1     # 与 train_yb3.py 的 CLI 默认保持一致
    lam_mono: float = 0.0      # P4 接入，先留 0
    lam_dir: float = 0.0       # P4
    # 方向损失施加在哪几个推演步。P4-C 实测：只压 h=1 压不住 h=36
    # （h=1 被压到 0.024-0.192，h=36 仍 0.089-0.482，且两个动作维互相挤兑）。
    dir_h: tuple[int, ...] = (1,)
    # 方向损失的正 margin。0 会让惩罚的驻点落在「零响应」处 —— 那正是
    # dir_vr 数符号最含糊的地方，实测 dir_vr 因此压不下去（§13 #41）。
    dir_margin: float = 0.25
    lam_anchor: float = 0.0    # P4
    gamma: float = 0.97        # 多步折扣：36 步后仍保留 0.33
    w_steady: float = 0.3      # 瞬态行的权重
    lipschitz_max: float = 1.0
    lat_k: int = 4             # 隐一致性抽样多少个 h（0=每步）。H=48 时每步一次
                               # 编码器前向占单步耗时 68%，抽样是主要提速手段。


@dataclass
class TargetScales:
    """各目标的归一化尺度，从训练集统计。除以它之后各项才可比。"""

    p_plant: float = 1.0
    p_fam: np.ndarray = field(default_factory=lambda: np.ones(4, dtype=np.float32))
    w_chiller: float = 1.0
    temps: dict[str, float] = field(default_factory=dict)
    w_dev: dict[str, float] = field(default_factory=dict)

    @staticmethod
    def fit(sd, row_mask) -> "TargetScales":
        """row_mask: 训练窗口覆盖的行的布尔掩码（见 dataset.covered_rows）。"""
        e = sd.extra
        x, av = sd.x[row_mask], sd.avail[row_mask].astype(bool)
        ti = sd.sch.token_index

        def std_of(fam: str, fld: str) -> float:
            if fld not in S.CANON_FIELDS[fam]:
                return 1.0
            k = S.CANON_FIELDS[fam].index(fld)
            idx = [i for i, (f, _) in enumerate(ti) if f == fam]
            v = x[:, idx, k][av[:, idx, k]]
            return float(v.std()) if v.size > 32 else 1.0

        return TargetScales(
            p_plant=max(float(e["P_plant"][row_mask].std()), 1e-3),
            p_fam=np.array([max(float(e[f"power_{f}"][row_mask].std()), 1e-3)
                            for f in FAM_ORDER], dtype=np.float32),
            w_chiller=max(std_of("chiller", "consumption"), 1e-3),
            temps={n: max(std_of("chiller", n), 1e-3) for n in
                   ("cold_back_temp", "cool_back_temp", "cool_out_temp")}
            | {"tower_out": max(std_of("plant", "tower_out"), 1e-3)},
            w_dev={f: max(std_of(f, "consumption"), 1e-3)
                   for f in ("tower", "coolpump", "coldpump")},
        )


def _masked_mse(pred: torch.Tensor, tgt: torch.Tensor, mask: torch.Tensor,
                scale: float, wrow: torch.Tensor | None = None) -> torch.Tensor:
    """按 mask 加权的归一化 MSE。mask 全 0 时返回 0（不是 NaN）。"""
    if wrow is not None:
        mask = mask * wrow
    d = ((pred - tgt) / scale) ** 2 * mask
    n = mask.sum()
    return d.sum() / n.clamp_min(1.0) if n > 0 else pred.sum() * 0.0


class ObsLoss:
    """观测损失，五类目标。字段下标在构造时算好，前向里不再查表。"""

    def __init__(self, sd, scales: TargetScales, wts: LossWeights, device):
        self.sc = scales
        self.w = wts
        ti = sd.sch.token_index
        self.idx = {f: torch.tensor([i for i, (ff, _) in enumerate(ti) if ff == f],
                                    device=device) for f in S.FAMILIES}
        self.K = {(f, n): S.CANON_FIELDS[f].index(n)
                  for f in S.FAMILIES for n in S.CANON_FIELDS[f]}
        # 逐设备功率标签掩码（哪几台真有标签）
        self.dev_mask = {
            f: torch.from_numpy(sd.extra[f"per_device_mask_{f}"]).to(device)
            for f in ("tower", "coolpump", "coldpump")}
        self.p_fam = torch.tensor(scales.p_fam, device=device)

    def __call__(self, pred: dict[str, torch.Tensor], raw: torch.Tensor,
                 avail: torch.Tensor, tgt_P: torch.Tensor, tgt_fam: torch.Tensor,
                 wrow: torch.Tensor) -> dict[str, torch.Tensor]:
        """raw/avail [B,N,F] 为该步真值帧；tgt_P [B]；tgt_fam [B,4]；wrow [B] 行权重。"""
        out: dict[str, torch.Tensor] = {}
        ch, pl = self.idx["chiller"], self.idx["plant"]

        def fld(fam, name):
            i, k = self.idx[fam], self.K[(fam, name)]
            return raw[:, i, k], avail[:, i, k]

        # k=1 全站总功率
        out["k1"] = (((pred["P_plant"] - tgt_P) / self.sc.p_plant) ** 2 * wrow).mean()

        # k=2 四类分项功率
        pf = []
        for f in FAM_ORDER:
            key = "w_chiller_on" if f == "chiller" else f"w_{f}_dev"
            pf.append(pred[key].sum(-1) if key in pred else pred["P_plant"] * 0)
        out["k2"] = ((((torch.stack(pf, -1) - tgt_fam) / self.p_fam) ** 2)
                     .mean(-1) * wrow).mean()

        # k=3 逐台冷机功率，仅 on=1
        y, m = fld("chiller", "consumption")
        on = raw[:, ch, self.K[("chiller", "on")]]
        out["k3"] = _masked_mse(pred["w_chiller"], y, m * on,
                                self.sc.w_chiller, wrow[:, None])

        # k=4 中间温度量
        t = pred["P_plant"].sum() * 0.0
        for name, pk in (("cold_back_temp", "cold_back"),
                         ("cool_back_temp", "cool_back"),
                         ("cool_out_temp", "cool_out")):
            y, m = fld("chiller", name)
            t = t + _masked_mse(pred[pk], y, m * on, self.sc.temps[name], wrow[:, None])
        # 站级出塔温只有一路，用开机加权平均对齐（yb3 的 tower_out_temp 主要来自制冷站1）
        y, m = fld("plant", "tower_out")
        w_on = on / on.sum(-1, keepdim=True).clamp_min(1e-6)
        t = t + _masked_mse((pred["tower_out"] * w_on).sum(-1, keepdim=True), y, m,
                            self.sc.temps["tower_out"], wrow[:, None])
        out["k4"] = t / 4.0

        # k=5 逐塔/逐泵功率，只在真有标签的设备上
        t = pred["P_plant"].sum() * 0.0
        n_used = 0
        for f in ("tower", "coolpump", "coldpump"):
            if f"w_{f}_dev" not in pred:
                continue
            y, m = fld(f, "consumption")
            t = t + _masked_mse(pred[f"w_{f}_dev"], y, m * self.dev_mask[f][None],
                                self.sc.w_dev[f], wrow[:, None])
            n_used += 1
        out["k5"] = t / max(n_used, 1)
        return out


def compute_loss(model, out: dict, batch: dict, obs_loss: ObsLoss,
                 wts: LossWeights, W: int) -> tuple[torch.Tensor, dict[str, float]]:
    """总损失。返回 (标量 loss, 各项明细)。"""
    preds = out["preds"]
    H = len(preds)
    dev = preds[0]["P_plant"].device
    seq_raw, seq_av = batch["seq_raw"], batch["seq_avail"]

    # 瞬态行降权（steady=0 -> w_steady）
    st = batch["steady"]
    wrow_all = torch.where(st > 0.5, torch.ones_like(st),
                           torch.full_like(st, wts.w_steady))

    terms = {k: torch.zeros((), device=dev) for k in ("k1", "k2", "k3", "k4", "k5")}
    gsum = 0.0
    for h, p in enumerate(preds):
        g = wts.gamma ** h
        gsum += g
        d = obs_loss(p, seq_raw[:, W + h], seq_av[:, W + h],
                     batch["P_plant"][:, h], batch["P_fam"][:, h], wrow_all[:, h])
        for k, v in d.items():
            terms[k] = terms[k] + g * v
    for k in terms:
        terms[k] = terms[k] / gsum

    loss = (wts.k1_total * terms["k1"] + wts.k2_family * terms["k2"]
            + wts.k3_chiller * terms["k3"] + wts.k4_temp * terms["k4"]
            + wts.k5_device * terms["k5"])

    log = {k: float(v.detach()) for k, v in terms.items()}

    # 隐一致性 —— 对抗长程漂移的主力。
    #
    # **必须除以 z_true 的尺度**。z 的绝对尺度是自由的（编码器末端的 pool 是裸
    # Linear），未归一化时优化器会发现「把 z 整体缩小」是降低 lat 最省力的路径，
    # 于是塌缩到平凡解。实测过一次：lat 先涨到 38.76 再塌到 0.0024，同时
    # head_w 的输入跟着塌进 softplus 饱和区、梯度消失，逐台冷机功率变成常数
    # （k3 恒为 3.3685 = 目标方差），val 从 0.070 崩到 26。
    #
    # 分母取 detach，本身不回传梯度，于是该项对 z 的整体缩放不变 —— 缩小 z
    # 不再有收益，只能靠真正对齐轨迹来降低。
    if wts.lam_lat > 0 and "z_true" in out:
        lat, z_scale = latent_consistency(out.get("z_lat", out["z"]), out["z_true"])
        loss = loss + wts.lam_lat * lat
        log["lat"] = float(lat.detach())
        log["z_scale"] = float(z_scale)

    # L-soft-B：同机容量守恒（P4）。type_id 从 model 上取，避免改 compute_loss 签名。
    if wts.lam_soft_b > 0:
        tid = getattr(model, "_type_id", None)
        # 缺 type_id 时**必须报错而不是跳过**：静默跳过会让整个 P4 臂在
        # 「约束已开」的假象下跑完，日志里什么都看不出来。
        if tid is None:
            raise RuntimeError(
                "lam_soft_b > 0 但 model._type_id 未设置。"
                "请在构建模型后写 `model._type_id = ctx.type_id`")
        sb, sbd = soft_b_capacity(preds, seq_raw, seq_av, W, tid, wts.gamma)
        loss = loss + wts.lam_soft_b * sb
        log["soft_b"] = float(sb.detach())
        log.update(sbd)

    # L-soft-B：蒸发侧能量平衡的软一致项
    #
    # **这个组合必须显式拒绝**：dt_evap_mode="soft" 时 eta 的**唯一**梯度通路
    # 就是本项。lam_evap_bal=0 会让 eta 退回 #24 的零梯度状态，而训练日志、
    # 硬约束违例率、物理量回读三者**都看不出来**（这是第三次了：#24 零梯度、
    # #29 有梯度但滑到界、这里是有开关但没打开）。要跑那个消融请显式写
    # dt_evap_mode="free"。
    # **只在梯度启用时检查。** 这条守卫讲的是「eta 没有梯度通路」，
    # 评测期本就没有梯度，那里把正则权重清零是为了让各站的验证损失可比
    # （不同站的正则项量级不同，混进来就没法横比）。
    # 第一版漏了这个条件，多站评测直接被自己的守卫拦下 —— 守卫也要有适用边界。
    _mode = getattr(getattr(model, "decoder", None), "dt_evap_mode", None)
    if torch.is_grad_enabled() and _mode == "soft" and wts.lam_evap_bal <= 0:
        raise RuntimeError(
            "dt_evap_mode='soft' 但 lam_evap_bal=0：eta 将没有任何梯度通路，"
            "等于静默复现 §13 #24。要跑该消融请显式用 dt_evap_mode='free'")

    if wts.lam_evap_bal > 0 and "dt_evap_bal" in preds[0]:
        num = torch.zeros((), device=dev)
        gs = 0.0
        for h, p in enumerate(preds):
            if "dt_evap_bal" not in p:
                continue
            g = wts.gamma ** h
            # 取对数比：无量纲、对 (eta, k_mcp) 的整体缩放表现为常数偏置，
            # 不会因量纲选择而隐性加权某一段工况。两者都严格为正。
            r = (torch.log(p["dt_evap"].clamp_min(1e-6))
                 - torch.log(p["dt_evap_bal"].clamp_min(1e-6)))
            num = num + g * (r ** 2).mean()
            gs += g
        if gs > 0:
            bal = num / gs
            loss = loss + wts.lam_evap_bal * bal
            log["evap_bal"] = float(bal.detach())

    # eta 水平先验：只罚 batch 均值，逐样本的形状完全自由
    if wts.lam_eta_level > 0 and "eta_logit" in preds[0]:
        lg = torch.stack([p["eta_logit"].mean() for p in preds]).mean()
        eta_pen = lg ** 2
        loss = loss + wts.lam_eta_level * eta_pen
        log["eta_lvl"] = float(lg.detach())

    # Lipschitz：超过 L_max 的部分才罚
    if wts.lam_lip > 0:
        lip = lipschitz_penalty(model.transition, wts.lipschitz_max)
        loss = loss + wts.lam_lip * lip
        log["lip"] = float(lip.detach())

    log["total"] = float(loss.detach())
    return loss, log


def soft_b_capacity(preds: list[dict], seq_raw: torch.Tensor, seq_avail: torch.Tensor,
                    W: int, type_id: torch.Tensor, gamma: float = 1.0
                    ) -> tuple[torch.Tensor, dict[str, float]]:
    """L-soft-B：同一台冷机的额定容量是常数，故 `Q_evap / PLR` 应当不随工况漂移。

    §4.3 原文写的是比值式 `Q_i/Q_i^ref ≈ PLR_i/PLR_i^ref`，需要一个「训练集中位
    工况」的参照 `Q_i^ref`。但 `Q_evap` 不可观测，那个参照只能取模型自己的预测，
    是个**训练中不断移动的靶子**。

    改写成等价但无参照的形式：`Q_i/PLR_i = C_i`（额定容量）对同一台机是常数，
    于是直接罚

        k6 = mean_i  Var_t[ log Q_{i,t} - log PLR_{i,t} ]

    取对数使它对 C_i 的取值免疫（容量约掉，正是 §4.3 要的），取方差使它不需要
    任何外部参照。**这是无量纲约束，不引入任何铭牌参数**（§12 G8：数据里没有
    铭牌，用 consumption 回归标定容量再拿容量约束 consumption 是循环论证）。

    ⚠ 它**不能**打掉 (eta, mcp) 的尺度共线（decoder §6 记的那条）：对数差里
    整体尺度被方差吃掉了，这是设计使然。要打掉共线需接冷凝侧。

    可证伪的诊断：`Q/plr` 的 cv 应当**低于** `W/plr` 的 cv —— 后者含 COP 随工况
    的波动，前者不含。若反过来，说明模型没学到蒸发侧的容量守恒。
    yb3 实测 `W/plr` cv：0.085 / 0.295 / 0.085 / 0.313 / 0.474 / 0.266 / 0.200。
    """
    k_plr = S.CANON_FIELDS["chiller"].index("plr")
    k_on = S.CANON_FIELDS["chiller"].index("on")
    ch = (type_id == S.FAMILIES.index("chiller"))
    tot = preds[0]["P_plant"].sum() * 0.0
    gsum, n_h = 0.0, 0
    diag = {}
    for h, p in enumerate(preds):
        if "q_evap" not in p:
            continue
        raw = seq_raw[:, W + h]
        av = seq_avail[:, W + h]
        plr = raw[:, ch, k_plr]
        on = raw[:, ch, k_on] > 0.5
        m = on & (plr > 1e-2) & (av[:, ch, k_plr] > 0.5)
        if not bool(m.any()):
            continue
        r = torch.log(p["q_evap"].clamp_min(1e-6)) - torch.log(plr.clamp_min(1e-2))
        # 逐台求方差：mask 下的 E[r^2] - E[r]^2
        cnt = m.sum(0).clamp_min(1.0)                     # [n_ch]
        mu = (r * m).sum(0) / cnt
        var = ((r - mu[None, :]) ** 2 * m).sum(0) / cnt
        used = m.sum(0) >= 8                              # 样本太少的机不计入
        if not bool(used.any()):
            continue
        g = gamma ** h
        tot = tot + g * var[used].mean()
        gsum += g
        n_h += 1
        if h == 0:
            diag["soft_b_cv0"] = float(var[used].mean().detach().sqrt())
    if n_h == 0:
        return tot, diag
    return tot / max(gsum, 1e-8), diag


def latent_consistency(z: torch.Tensor, z_true: torch.Tensor
                       ) -> tuple[torch.Tensor, torch.Tensor]:
    """隐一致性项。返回 (归一化后的距离, z_true 的 RMS)。

    **必须除以 z_true 的尺度。** z 的绝对尺度是自由的，未归一化时优化器会发现
    「把 z 整体缩小」是降低本项最省力的路径，于是塌缩到平凡解。实测过一次：
    lat 先涨到 38.76 再塌到 0.0024，同时 head_w 的输入跟着塌进 softplus 饱和区、
    梯度消失，逐台冷机功率退化成常数（k3 恒为目标方差），val 从 0.070 崩到 26。

    分母取 detach，不构成梯度路径，因此本项对 z 的整体缩放严格不变 ——
    缩小 z 不再有收益，只能靠真正对齐轨迹来降低。
    """
    denom = (z_true ** 2).mean().detach().clamp_min(1e-6)
    return ((z - z_true) ** 2).mean() / denom, denom.sqrt()


def lipschitz_penalty(transition, l_max: float, n_iter: int = 20) -> torch.Tensor:
    """对各线性层谱范数乘积超出 l_max 的部分做 relu^2 惩罚。

    用幂迭代估最大奇异值，**不用 `torch.linalg.matrix_norm(ord=2)`** ——
    后者要做完整 SVD，12 个矩阵每 batch 一次、还要对 SVD 求导，实测占单步耗时
    的 30%（74ms / 249ms），而 spectral_norm 已把每层压到 sigma=1，
    这个惩罚算出来恒等于 0。花的全是冤枉钱。

    默认 lam_lip=0，Lipschitz 由 `delta` 结构性控制（见下），本函数仅在需要
    显式惩罚时启用。

    **n_iter 不能取 2（2026-08-20 实测，§13 #34）。** 幂迭代 2 步的低估极其严重：
    构造一个真实谱范数乘积为 **1.12**（超界 12%）的 transition，n_iter=2 下本函数
    仍然返回 **0.000e+00**，n_iter=20 才报出 1.12。也就是说谁把 lam_lip 调上去，
    这个惩罚会**静默地什么都不做** —— 与 #24 同一类静默失效。
    默认改为 20；它只是矩阵-向量乘，代价远低于当年那版 SVD。

    关于 L_max：`z' = z + delta*G(z)` 的 Lipschitz 常数是 `1 + delta*Lip(G)`，
    **恒 >= 1**，所以设计文档 §4.5.2 写的「L <= 1」对残差映射不可达。真正的旋钮
    是 `eps = delta*Lip(G)`：H 步误差约 `(1+eps)^H ~ exp(eps*H)`，要 H=48 时
    不发散需要 `eps <~ 0.02`。P3 扫的应当是 delta，不是这个惩罚权重。
    """
    total = torch.zeros((), device=next(transition.parameters()).device)
    for blk in transition.blocks:
        for m in (blk.attn.q, blk.attn.k, blk.attn.v, blk.attn.o,
                  blk.ff.net[0], blk.ff.net[3]):
            w = getattr(m, "weight", None)
            if w is None:
                continue
            u = torch.randn(w.shape[0], device=w.device, dtype=w.dtype)
            u = u / u.norm().clamp_min(1e-9)
            for _ in range(n_iter):
                v = F.normalize(w.t() @ u, dim=0, eps=1e-9)
                u = F.normalize(w @ v, dim=0, eps=1e-9)
            total = total + torch.log((u @ (w @ v)).abs().clamp_min(1e-6))
    excess = F.relu(total - float(np.log(max(l_max, 1e-3))))
    return excess ** 2
