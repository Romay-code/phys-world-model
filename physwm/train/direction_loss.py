"""训练期方向损失（设计文档 §6 P4 第 2 项：方向损失分箱配对）。

**配对是天然的。** 计划原文说「分箱配对」，是因为参考项目只能在观测样本之间
找可比的对子。本项目有推演器，可以对**同一个 z_t** 换动作再推一步 ——
两条轨迹除了那一维动作以外逐位相同，这是比任何分箱都干净的配对。

**「分箱」真正要解决的问题是另一件事：先验并非处处适用。** 实测（yb3，
n=32622）按冷机功率分箱看 `corr(塔频, 逼近度)`：

    172-996 kW   -0.667      1932-2812   -0.213      3620-6496   -0.040
    996-1932     -0.682      2812-3620   **+0.519**  <- 符号翻转

裸相关 -0.486、控制负荷/湿球/开机塔数后的偏相关 -0.346，**总体支持先验**
（不存在「逼近度高->控制提塔频」那种反因果陷阱），但高负荷段塔频接近满频、
逼近度由湿球与塔容量饱和主导，边际效应趋于 0。在那里强求严格负响应就是
过约束。

处理方式不是硬性排除分箱，而是**把响应按目标自身的尺度归一化后取平方
hinge**：只罚确实反向的部分，且惩罚随违例幅度平方衰减，于是效应≈0 的
饱和区自动被降到可忽略。这比人为切分箱边界更稳，也少一个超参。

代价：每步额外跑 1 个基线 + N_spec 个扰动的**单步**推演（H=1）。
H=48 时约 +10%；只取半个 batch 时约 +5%。
"""
from __future__ import annotations

import torch

from ..eval.direction import DirSpec, make_intervene

# 训练用的先验表：只保留**学出来的**关系。
# w_tower_dev / w_coolpump_dev / w_coldpump_dev 三项在解码器里是
# `w = on·(fr/50)^3·k` 的结构恒等式，dir_vr 实测恒为 0.000 ——
# 那不是学到的物理，把它们放进损失纯属浪费算力。
def train_specs() -> list[DirSpec]:
    return [
        DirSpec("塔频+1Hz", "tower", "frequency", 1.0,
                {"approx": -1, "cool_out": -1},
                why="风量↑ -> 逼近度↓、冷凝进水温↓"),
        DirSpec("冷冻供水温+0.5K", "chiller", "cold_out_temp", 0.5,
                {"w_chiller": -1, "P_plant": -1},
                why="蒸发温度↑ -> COP_carnot↑ -> 同负荷冷机功率↓",
                clip=(0.0, 30.0)),
    ]


def direction_penalty(model, batch, ctx, W: int, sch, norm_scale,
                      specs=None, sub: float = 0.5, margin: float = 0.25,
                      h_sample: tuple[int, ...] = (1,)
                      ) -> tuple[torch.Tensor, dict[str, float]]:
    """方向损失。返回 (标量, 日志)。**不可放在 no_grad 下。**

    `h_sample` 指定在推演的哪几步上施加。基线与扰动各推一次到
    `max(h_sample)`，在指定的 h 处取惩罚 —— 不是每个 h 各推一遍。

    **⚠ 只做单步（h_sample=(1,)）是不够的，这一条是实测推翻的。**

    第一版的理由是：无干预时 h=1 与 h=36 的违例率强相关
    （P4-B seed0 0.491/0.372、seed1 0.124/0.114），故压住单步即可。
    P4-C 实测否掉了它：加了单步方向损失后 h=1 被压到 0.024-0.192，
    但 h=36 该崩照崩，且**两个动作维互相挤兑** ——

        lam_dir=0.1  塔频->逼近度 h36 0.299 ✗   供水温->P_plant h36 0.021 ✓
        lam_dir=0.5  塔频->逼近度 h36 0.089 ✓   供水温->P_plant h36 0.208 ✗

    没有哪个 lam 能同时管住两边。根因是那个「强相关」观察来自**没有干预**
    的模型；一旦显式压平 h=1，相关性即不再成立。
    **拿无干预下的相关性论证有干预时的行为，是推理漏洞。**

    故 P4 正式配置应取 h_sample=(1, 12, 36) 之类跨视野的采样。
    保留 (1,) 作为对照臂。

    **⚠ margin 的代价：饱和区过约束。**
    第一版用「无 margin 的平方 hinge」正是为了让高负荷饱和段（真实边际效应
    趋于 0，实测该负荷箱 corr 翻正到 +0.519）自动降权。加了正 margin 之后
    这个自动降权失效了 —— 那里也会被要求给出一个正向响应。
    缓解手段是分母取 **batch 内响应幅度的中位**而非目标 std：整批都饱和时
    中位随之变小，要求也跟着变小。但混合 batch 里饱和样本仍会被非饱和样本
    抬高的中位数拖着走。**这是个真实的权衡，须由 dir_vr 与精度一起判定，
    不要假定它一定更好。**
    """
    specs = specs or train_specs()
    B = batch["P_plant"].shape[0]
    n = max(int(B * sub), 1)
    sl = {k: (v[:n] if torch.is_tensor(v) else v) for k, v in batch.items()}
    dev = batch["P_plant"].device
    sctx = ctx.ctx(n, dev)
    kw = dict(desc=ctx.desc, stat_rel=ctx.stat_rel, type_id=ctx.type_id,
              site_ctx=sctx, W=W, reanchor_p=0.0, key_mask=ctx.key_mask)

    hs = tuple(sorted({max(1, int(h)) for h in h_sample}))
    kw["H"] = hs[-1]
    base_all = model.rollout(sl, **kw)["preds"]
    total = torch.zeros((), device=dev)
    logs: dict[str, float] = {}
    cnt = 0
    for s in specs:
        fn = make_intervene(sch, norm_scale, s, ctx.type_id)
        b2 = dict(sl)
        b2["seq_x"], b2["seq_raw"] = fn(sl["seq_x"].clone(), sl["seq_raw"].clone(), W)
        pert_all = model.rollout(b2, **kw)["preds"]
        for h in hs:
            base, pert = base_all[h - 1], pert_all[h - 1]
            for tgt, sign in s.targets.items():
                if tgt not in base or tgt not in pert:
                    continue
                y0 = base[tgt]
            # 按目标自身的尺度归一化：approx 是 K、P_plant 是 kW，
            # 不归一化会让 kW 量级的项吃掉全部权重。分母 detach，
            # 否则「把预测整体放大」会成为降低本项的免费路径（§4.5.2 的教训）。
                # 归一化到「典型响应幅度」而非目标的 std：
                #   前者让 d 的中位量级恒为 1，margin 因此在不同目标、
                #   不同 h 上含义一致（「至少要有典型幅度的 margin 倍」）。
                #   后者在各目标间差 200 倍（approx 0.003 vs P_plant 0.8），
                #   同一个 margin 没法通用。
                # 分母 detach —— 否则整体缩放响应是降低本项的免费路径。
                raw_d = pert[tgt] - y0
                sc = raw_d.detach().abs().median().clamp_min(1e-9)
                d = raw_d / sc
                # **margin 必须为正。** deadband/无 margin 的平方 hinge 其最优解是
                # 「没有响应」而不是「响应方向正确」：反向响应的惩罚是 d²、梯度 2d，
                # 优化器把它推到 d=0 就停 —— 那里惩罚与梯度同时为零，而 dir_vr
                # 在 d=0 附近数的是符号，落在哪一侧近乎掷硬币。
                # 实测（tools/diag_direction_gap.py，p4d_dirh_0.1 checkpoint）：
                #   塔频->逼近度 h=36  违例 |d| 中位 0.0018 vs 合规 0.0216，
                #                      违例只占平方和的 0.1% —— 损失早已"满意"，
                #                      dir_vr 仍有 0.212。
                #   供水温->P_plant h=12  违例占平方和 90.7%，损失在拼命压，
                #                      但它压的方向是把响应压到零，不是压过零点。
                # 加正 margin 后 d=0 处梯度不再为零，最优解移到 d >= margin。
                viol = torch.relu(margin - float(sign) * d)
                total = total + (viol ** 2).mean()
                cnt += 1
                logs[f"dir_{tgt}@{h}"] = float((viol > 0).float().mean().detach())
    if cnt:
        total = total / cnt
    return total, logs
