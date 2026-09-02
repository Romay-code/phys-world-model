#!/usr/bin/env bash
# P5 阶段 A：第一轮多站联合预训练（侦察轮）。
#
# 目的**不是**拿指标，是回答三个问题：
#   1. 12 站联合训练能不能稳定跑（各站量纲差 6.9~38.5 倍，掩码能力各异）
#   2. 留出站的零样本损失是什么量级 —— 决定后续要不要少样本适配
#   3. hx 会不会重演 B 系的崩溃（B0/B1 在此 R² −5.94 / −7.64，设计文档 §5.2 点名）
#
# 故意不开方向损失：它给跨站路径引入未测过的代码，且 +37% 时间。
# 这轮要少变量。L-soft-B 必须开（否则 eta 无梯度，且有守卫拦着）。
#
# 12 站训练 / 2 站留出，200 步/epoch × 50 epoch = 10000 步
# （单站 P4 是 100×200=20000 步，这里每站约 830 步 —— 侦察轮够用）。
set -uo pipefail
ROOT="${PHYSWM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PY="${PHYSWM_PY:-python}"
cd "$ROOT"; mkdir -p experiments/results
export PYTHONIOENCODING=utf-8 PYTHONPATH="$ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

tag="p5a_scout"
echo "=== $(date +%m-%d' '%H:%M) $tag : 12 站联合预训练，留出 hx / pc3 ==="
$PY -u experiments/train_multisite.py \
  --device cuda --epochs 50 --steps 200 --batch-size 64 \
  --H 48 --select-H 48 --delta 0.5 \
  --lam-lat 0.1 --lam-evap-bal 0.03 --lam-dir 0.0 \
  --sample proportional --hold-out "hx,pc3" \
  --seeds 0 --tag "$tag" --log-every 2 \
  > "experiments/results/${tag}.log" 2>&1
echo "rc=$?"
tail -20 "experiments/results/${tag}.log"
