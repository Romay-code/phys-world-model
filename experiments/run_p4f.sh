#!/usr/bin/env bash
# P4 阶段 F：正式验收 —— 全部物理约束齐开，生产口径 5 seed。
#
# 配置与 P3-B main_d0.5 / P4-B 逐项一致（200 epoch / 训练H=48 / 评测H=96 /
# delta=0.5 / patience=100 / seed 0-4），于是三者可直接对标：
#
#   P3-B  无 L-soft-B、无方向损失   step1 MAE 71.1±7.9   h* 88.6±6.2
#   P4-B  +L-soft-B(0.03)          step1 MAE 78.8±4.1   h* 85.2±21.1   dir_vr 全线超标
#   P4-F  +方向损失(margin=0.25)    <- 本轮
#
# 短课程实测（60 epoch/评测H=48/单 seed）：
#   margin  MAE     逼近度h36  冷机功率h36  P_plant_h36
#   0       81.6    0.371      0.083        0.145
#   0.25    77.8    0.000      0.001        0.001     <- 精度还更好
#   1.0     132.7   0.000      0.000        0.000     <- 过约束，精度崩
#
# 验收门限（设计文档 §6 P4）：
#   dir_vr(36) <= 0.10、符号一致率 >= 95%、硬约束零违例、
#   **且 h* 不退化超过 10%**（对 P3-B 的 88.6 即 >= 79.7）
set -uo pipefail
ROOT="${PHYSWM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PY="${PHYSWM_PY:-python}"
cd "$ROOT"; mkdir -p experiments/results
export PYTHONIOENCODING=utf-8 PYTHONPATH="$ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

tag="p4f_full"
echo "=== $(date +%m-%d' '%H:%M) $tag : L-soft-B 0.03 + 方向损失 0.1/margin0.25/h=1,12,36  5 seed ==="
$PY -u experiments/train_yb3.py \
  --device cuda --mode curriculum --split-mode blocked \
  --epochs 200 --steps-per-epoch 100 \
  --H 96 --train-H 48 --eval-H 96 \
  --delta 0.5 --patience 100 --seeds 0 1 2 3 4 \
  --dt-evap-mode soft --lam-evap-bal 0.03 \
  --lam-dir 0.1 --dir-h 1,12,36 --dir-margin 0.25 --dir-vr \
  --tag "$tag" > "experiments/results/${tag}.log" 2>&1
echo "rc=$?"
grep -E "step1 MAE|step1 R2|^  h\*|超门限" "experiments/results/${tag}.log" | tail -20
