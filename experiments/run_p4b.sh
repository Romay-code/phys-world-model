#!/usr/bin/env bash
# P4 阶段 B：L-soft-B 定档配置跑满 5 seed，并首次测 dir_vr。
#
# 配置与 P3-B 的 main_d0.5 **逐项一致**（200 epoch / 训练H=48 / 评测H=96 /
# delta=0.5 / patience=100 / seed 0-4），只多开 L-soft-B。
# 于是 P3-B 那一轮天然就是对照臂，不必重跑：
#     P3-B main_d0.5:  step1 MAE 71.1 ± 7.9   h* 88.6 ± 6.2
# P4 门限里最关键的一条是「h* 不退化超过 10%」，即 h* >= 79.7。
#
# 阶段 A（60 epoch 单 seed 扫描）定出 lam=0.03 是拐点：
#     lam    0     0.03    0.1    0.3   硬等式
#     MAE   60.9   74.5    97.1  103.6  127.8
#     曲率   无    +6.1%  +11.6% +17.2% +95.6%
# 但单 seed 的 60.9 vs 74.5 只有约 2σ（P3-B 实测 cv 11.2%），
# **精度代价必须在这一轮用 5 seed 重新量**（§13 #25 的教训）。
set -uo pipefail
EPOCHS="${EPOCHS:-200}"; STEPS="${STEPS:-100}"; LAM="${LAM:-0.03}"
ROOT="${PHYSWM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PY="${PHYSWM_PY:-python}"
cd "$ROOT"; mkdir -p experiments/results
export PYTHONIOENCODING=utf-8 PYTHONPATH="$ROOT"

tag="p4b_softb_${LAM}"
echo "=== $(date +%m-%d' '%H:%M) $tag : lam_evap_bal=$LAM  5 seed  含 dir_vr ==="
$PY -u experiments/train_yb3.py \
  --device cuda --mode curriculum --split-mode blocked \
  --epochs "$EPOCHS" --steps-per-epoch "$STEPS" \
  --H 96 --train-H 48 --eval-H 96 \
  --delta 0.5 --patience 100 --seeds 0 1 2 3 4 \
  --dt-evap-mode soft --lam-evap-bal "$LAM" --dir-vr \
  --tag "$tag" > "experiments/results/${tag}.log" 2>&1
rc=$?
echo "rc=$rc"
grep -E "step1 MAE|step1 R2|^  h\*|dir_vr|被数据定住|未被定住" "experiments/results/${tag}.log" | tail -20
