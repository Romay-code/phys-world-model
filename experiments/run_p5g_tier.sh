#!/usr/bin/env bash
# P5 阶段 G：轴 2 跨档外推 —— 换个能力更差的档，模型还行吗。
#
# ## 为什么要两臂
#
# 直接「整留 C 档」得到的数**不可解释**：不知道多少是跨档造成的、
# 多少只是那几个站本来就难。C 档的三个站（zx / pb1 / pc2）
# 在轴 1 的所有臂里都是训练站，没有同档跨站的基线可比。
#
# 故用**同一个站（pb1）**做前后对照，只改「它所属的档在
# 训练集里还有没有别的成员」这一件事：
#
#   臂 1  留出 pb1 + pc3           -> 训练集**仍有** zx / pc2 两个 C 档站
#                                      = 同档跨站（轴 1 口径，C 档版）
#   臂 2  留出 zx + pb1 + pc2 + pc3 -> 训练集**一个 C 档站都没有**
#                                      = 跨档外推（轴 2）
#
# 两臂在pb1上的差值，就是「该档在不在训练集里」的净效应 ——
# 这正是 §13 #53 说 pc3 的 −1.64「不是零样本泛化、是外推到一个空档」时
# 缺的那个对照。pc3 顺带得到第 4、第 5 次跨档测量。
#
# ## 口径
#
# **pc3 两臂都留出**，与轴 1 的五臂一致，八个数才在同一条件下。
# 配方为交付配方（不开可用性丢弃，§13 #62）。
# 臂 2 的训练集只剩 A 档 2 站 + B 档 5 冷站，数据量明显更少 ——
# 这是跨档留出的固有代价，报数时必须写明，不能假装与臂 1 同条件。
set -uo pipefail
ROOT="${PHYSWM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PY="${PHYSWM_PY:-python}"
cd "$ROOT"; mkdir -p experiments/results
export PYTHONIOENCODING=utf-8 PYTHONPATH="$ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

pc3=pc3
run_arm () {
  local tag="$1"; shift
  local hold="$1"; shift
  echo "=== $(date +%m-%d' '%H:%M) $tag : 留出 $hold ==="
  $PY -u experiments/train_multisite.py \
    --device cuda --epochs 50 --steps 200 --batch-size 64 \
    --H 48 --select-H 48 --delta 0.5 \
    --lam-lat 0.1 --lam-evap-bal 0.03 --lam-dir 0.0 \
    --avail-drop-joint 0 --avail-drop-each 0 \
    --sample proportional --hold-out "$hold" \
    --seeds 0 --tag "$tag" --log-every 5 \
    > "experiments/results/${tag}.log" 2>&1
  local rc=$?
  echo "  train rc=$rc"
  if [[ $rc -ne 0 ]]; then tail -20 "experiments/results/${tag}.log"; return; fi
  $PY -u experiments/eval_zeroshot.py \
    --ckpt "experiments/results/${tag}/model_seed0.pt" \
    --device cuda --eval-H 48 --hold-out "$hold" --tag "$tag" \
    > "experiments/results/${tag}_eval.log" 2>&1
  echo "  eval rc=$?"
  tail -8 "experiments/results/${tag}_eval.log"
}

run_arm p5g_hold_jinshi  "pb1,${pc3}"
run_arm p5h_holdtier_C   "zx,pb1,pc2,${pc3}"

echo "=== $(date +%m-%d' '%H:%M) 轴 2 完成 ==="
