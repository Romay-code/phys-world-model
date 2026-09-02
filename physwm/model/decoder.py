"""L3 物理解码器 —— 八级级联，硬约束靠重参数化。

约束不是损失项，是前向计算的函数形式，因此训练中、推理中、外推区一律 100% 满足：

    (1) approx >= 0                softplus
    (2) cool_dt > 0                softplus
    (3) W > Q_evap / COP_carnot    Q = W*COP_carnot*eta 且 eta <= ETA_MAX < 1
    (4) Q_cond = Q_evap + W        恒等式，不是拟合

物理量的绝对量纲由 scale buffer 承载，网络内部保持 O(1)。scale 从训练集统计设定，
不是可学习参数 —— 它是量纲换算，不该被梯度改动。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

T0 = 273.15
DT_MIN = 2.0        # T_cond - T_evap 的下限，防 COP_carnot 发散
DELTA_CB = 2.0      # 每机冷凝进水的管路微调幅度 ±2 C

# 卡诺效率的物理区间。**不能用裸 sigmoid**：
#   1) fp32 下 sigmoid 会饱和到恰好 0.0 / 1.0，数学上的开区间在浮点里不成立；
#   2) 更要紧的是 eta -> 0 会让 W = Q/(COP*eta) 爆炸。实测 dry-run 就是这样发散的
#      （loss 1.77 -> 41.7），当时靠 clamp_min(1e-3) 兜底，而那个 clamp 一旦生效
#      就切断了 eta 的梯度屏障，反而让它一路滑到底。
# 改为把 sigmoid 仿射到闭区间内，约束在 fp32 下也真正成立，且无需任何 clamp。
#
# **区间必须收到物理范围，松区间等于不放信息。** 第一版取 [0.02, 0.98]，
# seed 0 训完实测：eta 双峰贴两端（p50=0.033，2.97% 贴下界、13.42% 贴上界），
# 反推实际 COP = Q/W 中位只有 **0.549** —— 热力学上不存在这样的冷机。
# 此时六项硬约束仍全部为 0，但它们是**空洞地满足**的：Q_evap 没被钉住，
# 把 Q 做小，卡诺下界 W > Q/COP_carnot 就自动成立，等于没约束住任何东西。
#
# 真实冷机 eta = COP/COP_carnot：离心机满载约 0.35-0.55，部分负荷降到 0.25 左右，
# 螺杆机约 0.3。取 [0.10, 0.70] —— 足够宽不至于和真实数据打架，
# 又足够窄能把 Q 钉住。pb1（唯一有冷冻流量的站）实测 eta 的
# p1..p99 = [0.446, 0.627]，落在该区间内的比例 99.97% —— 区间是实测支持的，
# 不是拍的。对应实测 COP 中位 8.11。
ETA_MIN, ETA_MAX = 0.10, 0.70

# 严格正的量要加下限，理由同 eta：softplus 在 fp32 下 x < -88 时下溢到**恰好 0**，
# 于是「> 0」这个约束在浮点里不成立。实测 dry-run 里 cool_dt 有 71% 的位置为 0，
# 且 71% 恰好等于冷机停机比例 —— 停机机的 cool_dt 不进任何损失（k4 按 on 掩码、
# P_plant 里 w_chiller 乘 on），完全不受约束，head 就漂到很负去了。
# 数据侧确认约束本身是对的：yb3 实测 cool_dt 中位 4.7 K，<=0 占比 0.0000。
# 1e-3 K 在物理上可忽略，但让约束在 fp32 下真正成立。
EPS_POS = 1e-3

# mcp（蒸发侧 m·cp）的对数活动范围：mcp ∈ mcp_scale · [e^-3, e^3] ≈ [0.05×, 20×]。
# 用 exp(RANGE·tanh(·)) 而非 softplus/clamp：严格正、有界、处处光滑，
# 且不会像 softplus 那样在 fp32 下underflow 到恰好 0（那会让 dt_evap 爆炸）。
# 与 eta 的 `ETA_MIN + (ETA_MAX-ETA_MIN)·sigmoid` 是同一个套路。
MCP_LOG_RANGE = 3.0

# 冷凝侧不动点迭代次数。该映射是负反馈，yb3 量级下收缩因子实测 0.267，
# 6 次后残差 ~7e-5 K，远小于温度测量分辨率。全是逐元素运算，代价可忽略。
N_FIXPOINT = 6


def _head(d: int, hidden: int = 64) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, 1))


class PhysicsDecoder(nn.Module):
    """D : (z, a, d) -> o_hat

    forward 需要的物理输入（原始量纲，不是归一化值）：
        wet_bulb  [B]
        cold_out  [B, n_ch]      冷冻供水温（准动作）
        on_ch     [B, n_ch]
        on/freq   每个泵塔族 [B, n_dev]
    """

    def __init__(self, d: int, n_dev: dict[str, int], *,
                 q_scale: float = 1000.0, dt_scale: float = 5.0,
                 approx_scale: float = 5.0,
                 p_scale: dict[str, float] | None = None,
                 q_mode: str = "from_w", w_scale: float = 500.0,
                 dt_evap_scale: float = 5.0, mcp_scale: float = 700.0,
                 dt_evap_mode: str = "soft",
                 mcp_cool_scale: float = 1500.0, cool_dt_mode: str = "free",
                 mcp_mode: str = "flow", k_mcp: float = 15.0,
                 d_desc: int = 32):
        super().__init__()
        self.n_dev = dict(n_dev)
        assert q_mode in ("from_w", "free")
        self.q_mode = q_mode
        # "soft"（默认，L-soft-B）：cold_back 由 head 拟合，能量平衡走软一致项。
        # "derived"：硬等式（L-hard-B）—— **只该用在有实测流量的站**。
        # "free"    ：dt_evap 由独立 head 预测 —— 这**就是 §13 #24 的病灶本身**，
        #             保留为消融臂，用来量化「把 dt_evap 接回 q_evap」的精度代价，
        #             以及在回归测试里持续证明该写法确实会切断 eta 的梯度。
        assert dt_evap_mode in ("soft", "derived", "free")
        self.dt_evap_mode = dt_evap_mode
        # "free"（默认）：cool_dt 由 head_d 自由预测。
        # "tied"        ：cool_dt = q_cond/mcp_cool，不动点解。数学上能辨识 eta 尺度，
        #                 但 yb3 信噪比不够（见 world_model.ModelConfig 注释），
        #                 保留为消融臂，待有流量的站再用。
        assert cool_dt_mode in ("tied", "free")
        self.cool_dt_mode = cool_dt_mode
        # "flow"（默认）：m·cp = k_mcp · Σ(开机泵频) / 开机冷机数，k_mcp 不可训。
        # "const"       ：每机学习常数 —— **已被实测否掉**，保留为消融臂。
        assert mcp_mode in ("flow", "const")
        self.mcp_mode = mcp_mode
        self.register_buffer("w_scale", torch.tensor(float(w_scale)))
        self.register_buffer("dt_evap_scale", torch.tensor(float(dt_evap_scale)))
        self.register_buffer("mcp_scale", torch.tensor(float(mcp_scale)))
        self.register_buffer("mcp_cool_scale", torch.tensor(float(mcp_cool_scale)))
        # 不可训练：见 forward 里 mcp_mode=="flow" 分支的说明
        self.register_buffer("k_mcp", torch.tensor(float(k_mcp)))
        self.register_buffer("eta_logit_shift", torch.zeros(()))
        self.head_a = _head(d)      # 塔逼近度
        self.head_d = _head(d)      # 冷却水温差
        self.head_delta = _head(d)  # 每机管路微调
        self.head_q = _head(d)      # 制冷量（仅 q_mode="free"）
        self.head_w = _head(d)      # 冷机功率（q_mode="from_w"，被 k1/k3 直接监督）
        self.head_eta = _head(d)    # 卡诺效率

        # 蒸发侧 m·cp：**每台冷机一个学习标量**，不是逐样本 head。
        #
        # 这是本模块最容易写错的一处。若让 mcp 逐样本自由预测，
        # `dt_evap = q_evap/mcp = W·COP_c·eta/mcp` 里 (eta, mcp) 可同比缩放，
        # 立刻退化成与 §4.3 那个 (Q, eta) 一模一样的不可辨识对 —— 换个地方重犯老病。
        #
        # 取常数的物理依据：冷冻侧多为定流量一次泵，同一台机的设计流量近似恒定。
        # 这样 W 与 COP_carnot 逐样本变化且都由实测温度/功率给定，
        # **eta 随工况的变化被数据完全钉住**，只剩每台机一个标定标量的自由度
        # —— 正是 §4.3「全站只留一个标定标量」所要的形态。
        #
        # ⚠ P5 遗留：per-chiller 参数不可跨站迁移，与 M2「一套权重吃任意台数」冲突。
        #   横向铺开前须改为由设备描述符 desc 生成（desc 本就是跨站可比的逐设备向量）。
        # **由设备描述符生成，不是 per-chiller 参数。**
        #
        # 实测（tools/check_sites.py + 权重互装测试）：整个 3.58e6 参数里，
        # 只有 log_mcp_raw / log_mcp_cool_raw 这两个 (n_chiller,) 张量
        # 阻塞跨站迁移 —— 其余全部形状无关，置换等变的 token 架构按设计工作。
        # 而 M2 要求「一套权重吃任意台数站点」，故必须改掉。
        #
        # desc 是逐设备的、由数据统计算出的**跨站可比**向量（§4.2.1 第二层），
        # 正是为这种场景设计的。未见过的站点直接由它的 desc 得到 mcp，
        # 不需要任何站点专属参数。
        self.mcp_net = nn.Sequential(
            nn.Linear(d_desc, 64), nn.GELU(), nn.Linear(64, 2))
        nn.init.zeros_(self.mcp_net[-1].weight)
        nn.init.zeros_(self.mcp_net[-1].bias)     # 零初始化 -> 起点即 mcp_scale
        self.head_dt_ev = _head(d)  # 仅 dt_evap_mode="free" 消融臂使用
        # 冷凝侧 m·cp 与蒸发侧共用 mcp_net 的第 2 个输出通道（同样 desc 驱动）
        self.head_p = nn.ModuleDict({f: _head(d) for f in ("tower", "coolpump", "coldpump")})

        p_scale = p_scale or {}
        self.register_buffer("q_scale", torch.tensor(float(q_scale)))
        self.register_buffer("dt_scale", torch.tensor(float(dt_scale)))
        self.register_buffer("approx_scale", torch.tensor(float(approx_scale)))
        for f in ("tower", "coolpump", "coldpump"):
            self.register_buffer(f"p_scale_{f}", torch.tensor(float(p_scale.get(f, 50.0))))

    def set_scales(self, **kw) -> None:
        """从训练集统计写入量纲。调用方负责算，这里只存。"""
        for k, v in kw.items():
            buf = getattr(self, k, None)
            if isinstance(buf, torch.Tensor):
                buf.fill_(float(v))

    def _s(self, name: str, override: dict | None):
        """取量纲。override 优先于 buffer。

        **多站训练必须逐 batch 传 override。** 量纲是站点属性不是模型属性：
        实测 14 站的 `w_scale` 跨 6.9 倍（178.5 tx ~ 1229.0 yb_低温）、
        `k_mcp` 跨 38.5 倍、`q_scale` 跨 22.2 倍 —— 单一 buffer 服务不了。
        它们全部由该站自己的历史数据算出，部署新站时可现场估，
        因此不进模型权重、不参与迁移（M2 要迁移的是无量纲的结构）。

        单站路径（override=None）走 buffer，与 P0–P4 行为逐位一致。
        """
        if override is not None and name in override:
            v = override[name]
            return v if torch.is_tensor(v) else torch.as_tensor(
                float(v), device=self.w_scale.device, dtype=self.w_scale.dtype)
        return getattr(self, name)

    def _mcp_raw(self, desc_ch, z_ch):
        """由描述符给出 (蒸发侧, 冷凝侧) 的 log-mcp 偏离量，各 [n_ch]。"""
        if desc_ch is None:
            n = z_ch.shape[1]
            z = z_ch.new_zeros(n)
            return z, z
        o = self.mcp_net(desc_ch)          # [n_ch, 2]
        return o[:, 0], o[:, 1]

    def forward(self, z_by_fam: dict[str, torch.Tensor], alpha: torch.Tensor,
                phys: dict[str, torch.Tensor],
                desc_ch: torch.Tensor | None = None,
                scales: dict | None = None) -> dict[str, torch.Tensor]:
        """z_by_fam[fam] = [B, n_dev, d]；alpha [B, n_ch, n_tw]（行归一）

        desc_ch [n_ch, d_desc]：冷机的设备描述符，用于生成 m·cp。
        为 None 时退化为零偏离（即 mcp = mcp_scale），仅供不带 desc 的单元测试用。
        """
        out: dict[str, torch.Tensor] = {}
        z_ch = z_by_fam["chiller"]
        z_tw = z_by_fam["tower"]
        wb = phys["wet_bulb"][:, None]                       # [B,1]
        cold_out = phys["cold_out"]                          # [B,n_ch]
        on_ch = phys["on_ch"]

        # --- 1 塔逼近度：>= 0 恒成立 (1) ---
        approx = F.softplus(self.head_a(z_tw).squeeze(-1)) * self._s('approx_scale', scales)
        out["approx"] = approx                               # [B,n_tw]

        # --- 2 软分组：alpha 由注意力给出，取代组级代理标签 ---
        approx_eff = torch.einsum("bij,bj->bi", alpha, approx)
        out["approx_eff"] = approx_eff                       # [B,n_ch]

        # --- 3 代数节点，无参数 ---
        tower_out = approx_eff + wb
        out["tower_out"] = tower_out

        # --- 4/5 冷却侧 + 冷机双物理头（tied 模式下二者互相依赖，一起解） ---
        delta = torch.tanh(self.head_delta(z_ch).squeeze(-1)) * DELTA_CB
        cool_back = tower_out + delta
        t_evap = cold_out + T0
        # 仿射到 [ETA_MIN, ETA_MAX]，fp32 下也严格落在区间内，无需 clamp
        # eta_logit_shift 平时恒为 0；只有可辨识性诊断会临时改它，
        # 用来测「把 eta 的水平整体推开，监督损失变不变」（profile-likelihood）。
        eta_logit = self.head_eta(z_ch).squeeze(-1) + self.eta_logit_shift
        eta = ETA_MIN + (ETA_MAX - ETA_MIN) * torch.sigmoid(eta_logit)

        # 参数化方向：**预测被监督的 W，导出潜变量 Q**，而不是反过来。
        #
        # 原设计（q_mode="free"）自由预测 Q 再由 Q/(COP*eta) 得 W。无流量的站
        # Q 没有任何标签，(Q, eta) 一起缩放 W 不变 —— 不可辨识，卡诺下界形同虚设。
        # seed 0 实测：eta 双峰贴两端(p50=0.033)，反推 COP=Q/W 中位 0.549，
        # 六项硬约束却全部为 0，即**空洞地满足**。
        #
        # 中间试过 q_mode="load_share"：Q = unit_k * load * share，用蒸发侧能量
        # 平衡锚定。但 14 站横向核查发现 **yb3 的 `load` 列不可用** ——
        # 隐含 COP (load/W_chiller) 只有 0.59，而其余 12 站在 5.7-9.3、
        # pb1实测 8.1。锚点本身不可信，这条路对 yb3 不成立。
        #
        # 现方案 from_w：W 由 head 直接预测（k1/k3 直接监督），
        # COP_carnot 由实测温度算，eta 落在物理区间（pb1流量实测 0.45-0.63），
        # 三者把 Q 钉在物理范围内：Q = W * COP_carnot * eta。
        #
        # 四条硬约束全部保留，且卡诺下界降为恒等推论：
        #     Q / COP_carnot = W * eta <= W * ETA_MAX < W   恒成立
        if self.q_mode == "free":
            w_ch = None                                   # 见下，free 臂在环内解
        else:
            w_ch = F.softplus(self.head_w(z_ch).squeeze(-1)) * self._s('w_scale', scales) + EPS_POS

        def _thermo(cool_dt):
            """给定冷却侧温差，算出这一层的全部热力学量。"""
            cool_out = cool_back + cool_dt
            # 防 COP_carnot 发散：冷凝温度至少高于蒸发温度 DT_MIN
            t_cond = torch.maximum(cool_out + T0, t_evap + DT_MIN)
            cop_carnot = t_evap / (t_cond - t_evap)
            if self.q_mode == "free":
                q_evap = F.softplus(self.head_q(z_ch).squeeze(-1)) * self._s('q_scale', scales)
                w = q_evap / (cop_carnot * eta)
            else:
                w = w_ch
                q_evap = w * cop_carnot * eta
            return cool_out, t_cond, cop_carnot, w, q_evap, q_evap + w

        if self.cool_dt_mode == "tied":
            # 冷凝侧能量平衡：Q_cond = m_cool*cp * (cool_out - cool_back)。
            #
            # **这条是 eta 尺度可辨识的唯一来源。** 蒸发侧只约束 q_evap/mcp_cold
            #   = W*COP_c*eta/mcp_cold，(eta, mcp_cold) 沿一条平坦方向共线，
            # 实测确认过：eta 一路滑到下界 0.100，mcp_cold 精确反补到 318.6
            # （若 eta=0.40 应为 1273.8），dt_evap 仍准确复现实测 5.06K。
            # 接上冷凝侧后
            #     cool_dt * mcp_cool - dt_evap * mcp_cold = W
            # 是一条**纯实测量之间**的线性关系（cool_dt、dt_evap 由两端温度实测，
            # W 实测），两个常数被它定死，eta 随之严格可辨识。
            #
            # 代价是引入循环：cool_dt -> cool_out -> t_cond -> COP_c -> q_evap
            #                 -> q_cond -> cool_dt。用不动点迭代解。
            # 该映射是**负反馈**（cool_dt↑ -> COP_c↓ -> q_evap↓ -> cool_dt↓），
            # yb3 量级下收缩因子实测 0.267，6 次迭代残差 ~7e-5 K。
            #
            # 初值取实测中位 dt_scale 这个**常数**，不用 head_d ——
            # 不动点与初值无关，用可学初值会造出又一个零梯度的死 head（#24 的教训）。
            mcp_cool = (self._s('mcp_cool_scale', scales)
                        * torch.exp(MCP_LOG_RANGE
                                    * torch.tanh(self._mcp_raw(desc_ch, z_ch)[1])))[None, :]
            cool_dt = self._s('dt_scale', scales).expand_as(cold_out).clone()
            for _ in range(N_FIXPOINT):
                *_, q_cond_i = _thermo(cool_dt)
                cool_dt = q_cond_i / mcp_cool
            out["mcp_cool"] = mcp_cool.expand_as(cool_dt)
        else:
            # 消融臂：cool_dt 自由预测（P2/P3 的写法）。eta 尺度不可辨识。
            cool_dt = F.softplus(self.head_d(z_ch).squeeze(-1)) * self._s('dt_scale', scales) + EPS_POS

        cool_out, t_cond, cop_carnot, w_ch, q_evap, q_cond = _thermo(cool_dt)
        out.update(cool_dt=cool_dt, cool_back=cool_back, cool_out=cool_out)
        out["eta_logit"] = eta_logit
        out.update(q_evap=q_evap, eta=eta, cop_carnot=cop_carnot,
                   w_chiller=w_ch, q_cond=q_cond, t_cond=t_cond, t_evap=t_evap,
                   cop_actual=q_evap / w_ch.clamp_min(1e-6))

        # --- 6 冷冻侧回水 ---
        # 蒸发侧能量平衡：Q_evap = m_dot * cp * (cold_back - cold_out)，**始终走这条等式**。
        #
        # 历史（§13 #6 与 #24 是同一处的两次翻车，必须一起读）：
        #   #6  最初写成 `dt_evap = q_evap / q_scale * dt_scale`，隐含假设 q_evap ~ q_scale。
        #       改成 from_w 后 q_evap 从 ~500 变 ~3700，算出 dT = 29 K（真值 5 K），
        #       k4 被搞坏（val 0.178 -> 1.97）。**根因是把 m*cp 写成了硬编码常数。**
        #   #24 当时的修法是改用独立 head 预测 dt_evap —— 量纲对了，
        #       但 q_evap 就此不进任何损失，`eta -> q_evap -> (无)`，
        #       **eta 梯度恒为 0**，报出的 0.387 是区间中点初始化值。修一个 bug 造出另一个。
        #
        # 现在的修法把两者一起解决：m*cp 既不硬编码（治 #6），
        # 也不让 dt_evap 绕开 q_evap（治 #24），而是把 m*cp 本身作为**每机学习常数**，
        # 由 mcp_scale 承载量纲、log_mcp_raw 承载每台机的偏离。
        #
        # 于是梯度通路恢复为   eta -> q_evap -> dt_evap -> cold_back -> k4
        #
        # ⚠ 残余自由度（如实记录，不要声称 eta 已完全可辨识）：
        #   cold_back 只约束 q_evap/mcp = W·COP_c·eta/mcp。W 与 COP_carnot 由实测
        #   功率与温度逐样本给定，故 **eta 随工况的形状被完全钉住**；
        #   但 (eta, mcp) 的整体尺度仍共线 —— 每台机剩一个标定标量。
        #   要彻底打掉它需把冷凝侧也接上 Q_cond = Q_evap + W（`cool_dt = q_cond/mcp_cool`），
        #   届时 cold_back 给斜率、cool_out 给截距，eta 严格可辨识。**那是 P4 L-soft-B 的事**，
        #   不在本次修复范围内 —— 冷凝侧当前拟合正常（cool_dt 5.0-5.3K vs 实测 4.7K），
        #   不在同一次改动里动它。
        if "m_cold_cp" in phys:
            mcp = phys["m_cold_cp"].clamp_min(1e-3)           # L-hard-B：有流量的站用实测
        elif self.mcp_mode == "flow":
            # **m·cp 由冷冻泵频驱动，不是每机常数。**
            #
            # 实测否掉了「定流量」这个假设（2026-08-19，全站 33463 样本）：
            #     W 的 cv        0.15 - 0.46      <- 负荷变化很大
            #     dt_evap 的 cv  0.073 - 0.091    <- ΔT 几乎不动
            # yb3 是**定 ΔT、变流量**运行的，ΔT 几乎不含负荷信息。用常数 mcp 时
            # `dt_evap = W·COP_c·eta/mcp` 逼着 W·COP_c·eta 保持恒定，模型只能把
            # eta 压到下界 0.100 去抵消，精度从 60.9 掉到 123.6 —— 约束在和数据打架。
            #
            # 预测全站冷机功率的 R²：仅 ΔT 0.080 / 仅泵频 0.845 / **ΔT×泵频 0.904**。
            # 能量平衡 Q = m·cp·ΔT 是对的，错的是把 m·cp 当常数。
            #
            # k_mcp 是**不可训练的 buffer**：若它与 eta 同时可训，
            # `eta/k` 又是一条平坦方向，eta 会再次滑到界上（#29 的教训）。
            # 尺度这一维由数据决定不了，就不要给模型一个假装能决定它的参数。
            #
            # ⚠ v1 简化：总流量在开机冷机间**等分**。yb3 是母管制、七台机
            #   `W·COP_c/PLR` 水平只差 1.17×（机组一致），等分是合理近似。
            #   逐机分配份额留到 P5 由 desc/注意力给出。
            flow = (phys["on_coldpump"] * phys["freq_coldpump"]).sum(-1, keepdim=True)
            n_on = on_ch.sum(-1, keepdim=True).clamp_min(1.0)
            mcp = (self._s('k_mcp', scales) * flow / n_on).clamp_min(EPS_POS)
        else:
            mcp = (self._s('mcp_scale', scales)
                   * torch.exp(MCP_LOG_RANGE * torch.tanh(self._mcp_raw(desc_ch, z_ch)[0])))[None, :]
        # 蒸发侧能量平衡给出的温差（无论走哪个分支都算出来，供 L-soft-B 用）
        dt_bal = q_evap / mcp
        dt_head = (F.softplus(self.head_dt_ev(z_ch).squeeze(-1))
                   * self._s('dt_evap_scale', scales) + EPS_POS)
        if self.dt_evap_mode == "derived":
            # **硬等式臂。§4.3 明确只适用于有实测流量的站（L-hard-B）。**
            # 在无流量的站把它做成硬结构，实测代价惨重：k4 被压垮并经共享
            # 编码器漏进 P_plant —— step1 MAE 60.9 -> 127.8，train 损失涨 9 倍
            # （k1 只涨 2.8×、k3 涨 1.6×，大头在 k4）。见 §13 #33。
            dt_evap = dt_bal
        elif self.dt_evap_mode == "free":
            # 纯消融臂：head 预测且**不加**软约束 —— 这就是 #24 的病灶本身
            # （q_evap 不进任何损失，eta 无梯度）。保留用于对照。
            dt_evap = dt_head
        else:
            # "soft"（默认，L-soft-B）：cold_back 仍由 head 拟合（保住精度），
            # 能量平衡改为损失里的软一致项 `lam_evap_bal`（见 losses.py）。
            # eta 经 `dt_bal` 进入那一项，梯度通路成立，但不再硬性支配 cold_back。
            dt_evap = dt_head
        out["cold_back"] = cold_out + dt_evap
        out["dt_evap"] = dt_evap
        out["dt_evap_bal"] = dt_bal
        out["mcp"] = mcp.expand_as(dt_bal)
        # 保留旧键名：此处已是恒等式，留作下游兼容与「等式确实闭合」的自检
        out["implied_mcp"] = q_evap / dt_evap.clamp_min(1e-6)

        # --- 7 塔/泵功率：三次方律在设备级表达 ---
        for fam in ("tower", "coolpump", "coldpump"):
            if self.n_dev.get(fam, 0) == 0:
                continue
            on, fr = phys[f"on_{fam}"], phys[f"freq_{fam}"]
            k = F.softplus(self.head_p[fam](z_by_fam[fam]).squeeze(-1))
            w_dev = (on * (fr / 50.0).clamp_min(0.0) ** 3 * k
                     * self._s(f"p_scale_{fam}", scales))
            out[f"w_{fam}_dev"] = w_dev
            out[f"w_{fam}"] = w_dev.sum(-1)

        # --- 8 主口径 ---
        p = (w_ch * on_ch).sum(-1)
        for fam in ("tower", "coolpump", "coldpump"):
            if f"w_{fam}" in out:
                p = p + out[f"w_{fam}"]
        out["P_plant"] = p
        out["w_chiller_on"] = w_ch * on_ch
        return out


# --------------------------------------------------------------------------
# 硬约束的运行时自检。训练里每隔若干步跑一次，断言而非惩罚。
# --------------------------------------------------------------------------

def check_hard_constraints(o: dict[str, torch.Tensor], tol: float = 1e-4,
                           on: torch.Tensor | None = None) -> dict[str, float]:
    """返回各约束的违例率。全为 0 才算重参数化生效。

    `on` 给出冷机启停掩码时，冷机侧的约束额外报一份只看开机机的违例率
    （后缀 `_on`）。停机机的中间量不进任何损失、也无物理意义，全量口径会把
    它们的漂移计成违例，读起来是误导 —— 但两个口径都报，不藏。
    """
    r = {}
    r["approx_nonneg"] = float((o["approx"] < -tol).float().mean())
    r["cool_dt_pos"] = float((o["cool_dt"] <= 0).float().mean())
    # 两条都要：区间检查会随 ETA_MIN/MAX 一起退化（若有人把区间设成 [0,1]，
    # `eta < 0` 永远为假，就检测不到 sigmoid 饱和到恰好 0/1 —— 变异测试实测漏检）。
    # 严格开区间这条与区间取值无关，是卡诺下界 W > Q/COP 成立的真正前提。
    r["eta_strict_open"] = float(((o["eta"] <= 0.0) | (o["eta"] >= 1.0)).float().mean())
    r["eta_in_band"] = float(((o["eta"] < ETA_MIN - tol)
                              | (o["eta"] > ETA_MAX + tol)).float().mean())
    r["t_cond_gap"] = float((o["t_cond"] - o["t_evap"] < DT_MIN - tol).float().mean())

    # 能量守恒是恒等式，相对误差应到浮点精度
    lhs, rhs = o["q_cond"], o["q_evap"] + o["w_chiller"]
    r["energy_rel_err"] = float(((lhs - rhs).abs() / rhs.abs().clamp_min(1e-6)).max())

    # 卡诺下界：W >= Q / COP_carnot
    w_min = o["q_evap"] / o["cop_carnot"].clamp_min(1e-6)
    r["carnot_violation"] = float((o["w_chiller"] < w_min * (1 - tol)).float().mean())

    if on is not None:
        m = on > 0.5
        n = float(m.sum())
        if n > 0:
            r["cool_dt_pos_on"] = float(((o["cool_dt"] <= 0) & m).float().sum() / n)
            r["eta_in_band_on"] = float(
                (((o["eta"] < ETA_MIN - tol) | (o["eta"] > ETA_MAX + tol)) & m
                 ).float().sum() / n)
            r["eta_strict_open_on"] = float(
                (((o["eta"] <= 0.0) | (o["eta"] >= 1.0)) & m).float().sum() / n)
            r["carnot_violation_on"] = float(
                ((o["w_chiller"] < w_min * (1 - tol)) & m).float().sum() / n)
    return r
