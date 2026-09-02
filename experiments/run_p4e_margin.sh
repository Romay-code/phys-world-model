#!/usr/bin/env bash
# P4 阶段 E：带正 margin 的方向损失。
#
# P4-C/D 五臂全部失败，诊断（tools/diag_direction_gap.py）指出根因不是权重、
# 不是视野，而是**损失的驻点位置错了**：
#   无 margin 的平方 hinge 对反向响应的惩罚是 d²、梯度 2d，优化器把违例推到
#   d=0 就停 —— 那里惩罚与梯度同时为零，而 dir_vr 在 d=0 附近数符号近乎掷硬币。
#   实测：塔频->逼近度 h=36 违例 |d| 已压到合规值的 1/12、只占平方和 0.1%，
#         dir_vr 仍有 0.212。供水温->P_plant h=12 违例占平方和 90.7%，
#         损失在拼命压，但压的方向是把响应推到零而非推过零点。
#
# 本轮：margin>0，最优解移到「至少 margin 倍典型幅度的正向响应」。
# 同时扫 margin，因为它有明确代价 —— 饱和区（真实边际效应趋于 0）会被过约束。
#
# 对照（同口径 60 epoch / eval-H 48 / seed 0）：
#   无方向损失      MAE 74.5   逼近度 ?      冷机功率 ?     P_plant ?
#   单步 lam0.1     MAE 79.2   0.299        0.107         0.021
#   跨视野 lam0.1   MAE 81.6   0.371        0.083         0.145
set -uo pipefail
EPOCHS="${EPOCHS:-60}"; STEPS="${STEPS:-100}"; DIRH="${DIRH:-1,12,36}"; LAM="${LAM:-0.1}"
ROOT="${PHYSWM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PY="${PHYSWM_PY:-python}"
cd "$ROOT"; mkdir -p experiments/results
export PYTHONIOENCODING=utf-8 PYTHONPATH="$ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for mg in ${MARGINS:-0.25 1.0}; do
  tag="p4e_mg_${mg}"
  echo "=== $(date +%m-%d' '%H:%M) $tag : margin=$mg lam_dir=$LAM dir_h=$DIRH ==="
  $PY -u experiments/train_yb3.py \
    --device cuda --mode curriculum --split-mode blocked \
    --epochs "$EPOCHS" --steps-per-epoch "$STEPS" \
    --H 48 --eval-H 48 --delta 0.5 --patience 100 --seeds 0 \
    --dt-evap-mode soft --lam-evap-bal 0.03 \
    --lam-dir "$LAM" --dir-h "$DIRH" --dir-margin "$mg" --dir-vr --tag "$tag" \
    > "experiments/results/${tag}.log" 2>&1
  rc=$?
  [ $rc -ne 0 ] && { echo "  !! 失败 rc=$rc"; tail -6 "experiments/results/${tag}.log"|sed 's/^/     /'; continue; }
  grep -E "step1:|h\*=|approx |w_chiller |P_plant " "experiments/results/${tag}.log" | tail -6 | sed 's/^/  /'
done

echo
echo "======== margin 扫描（门限：三项 dir_vr(36) 全 <=0.10）========"
printf '%-8s %-10s %-6s %-12s %-14s %-12s\n' margin MAE h\* 逼近度h36 冷机功率h36 P_plant_h36
for mg in ${MARGINS:-0.25 1.0}; do
  f="experiments/results/p4e_mg_${mg}.log"; [ -f "$f" ] || continue
  printf '%-8s %-10s %-6s %-12s %-14s %-12s\n' "$mg" \
    "$(grep -oP 'step1: MAE \K[0-9.]+' "$f"|tail -1)" \
    "$(grep -oP 'h\*=\K[0-9]+' "$f"|tail -1)" \
    "$(grep -oP 'approx\s+-1\s+[0-9.]+\s+\K[0-9.]+' "$f"|tail -1)" \
    "$(grep -oP 'w_chiller\s+-1\s+[0-9.]+\s+\K[0-9.]+' "$f"|tail -1)" \
    "$(grep -oP 'P_plant\s+-1\s+[0-9.]+\s+\K[0-9.]+' "$f"|tail -1)"
done
echo "对照 margin=0（P4-D lam0.1）: MAE 81.6  0.371  0.083  0.145"
echo "=============================================================="
