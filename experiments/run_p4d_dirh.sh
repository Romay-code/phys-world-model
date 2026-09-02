#!/usr/bin/env bash
# P4 阶段 D：跨视野方向损失（dir-h = 1,12,36）。
#
# P4-C（单步版，dir-h=1）三臂实测，供水温那一维两个方向都单调：
#     lam    h=1(w_chiller)  h=36(w_chiller)   h=1(P_plant)  h=36(P_plant)  MAE
#     0.1        0.031           0.107            0.001          0.021      79.2
#     0.5        0.024           0.482            0.038          0.208      80.0
#     2.0        0.011           0.826            0.000          0.286      86.1
#
# **lam 越大 h=1 越完美、h=36 越糟** —— 模型用只在单步成立的局部技巧满足约束，
# 而那个技巧在自回归展开中放大。这不是权重问题，是视野问题（§13 #39）。
#
# 本轮在 h ∈ {1, 12, 36} 上同时施加。代价：基线与扰动各推一次到 36 步
# （非每个 h 各推一遍），sub=0.25，实测约 +37%，即 ~3.3 h/臂。
set -uo pipefail
EPOCHS="${EPOCHS:-60}"; STEPS="${STEPS:-100}"; DIRH="${DIRH:-1,12,36}"
ROOT="${PHYSWM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PY="${PHYSWM_PY:-python}"
cd "$ROOT"; mkdir -p experiments/results
export PYTHONIOENCODING=utf-8 PYTHONPATH="$ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for lam in ${LAMS:-0.1 0.5}; do
  tag="p4d_dirh_${lam}"
  echo "=== $(date +%m-%d' '%H:%M) $tag : lam_dir=$lam  dir_h=$DIRH ==="
  $PY -u experiments/train_yb3.py \
    --device cuda --mode curriculum --split-mode blocked \
    --epochs "$EPOCHS" --steps-per-epoch "$STEPS" \
    --H 48 --eval-H 48 --delta 0.5 --patience 100 --seeds 0 \
    --dt-evap-mode soft --lam-evap-bal 0.03 \
    --lam-dir "$lam" --dir-h "$DIRH" --dir-vr --tag "$tag" \
    > "experiments/results/${tag}.log" 2>&1
  rc=$?
  [ $rc -ne 0 ] && { echo "  !! 失败 rc=$rc"; tail -6 "experiments/results/${tag}.log"|sed 's/^/     /'; continue; }
  grep -E "step1:|h\*=|approx |w_chiller |P_plant " "experiments/results/${tag}.log" | tail -6 | sed 's/^/  /'
done

echo
echo "==== 跨视野方向损失（对比 P4-C 单步版）===="
printf '%-8s %-10s %-6s %-14s %-16s %-14s\n' lam MAE h\* 逼近度h36 冷机功率h36 P_plant_h36
for lam in ${LAMS:-0.1 0.5}; do
  f="experiments/results/p4d_dirh_${lam}.log"; [ -f "$f" ] || continue
  mae=$(grep -oP 'step1: MAE \K[0-9.]+' "$f"|tail -1)
  hs=$(grep -oP 'h\*=\K[0-9]+' "$f"|tail -1)
  a=$(grep -oP 'approx\s+-1\s+[0-9.]+\s+\K[0-9.]+' "$f"|tail -1)
  w=$(grep -oP 'w_chiller\s+-1\s+[0-9.]+\s+\K[0-9.]+' "$f"|tail -1)
  pp=$(grep -oP 'P_plant\s+-1\s+[0-9.]+\s+\K[0-9.]+' "$f"|tail -1)
  printf '%-8s %-10s %-6s %-14s %-16s %-14s\n' "$lam" "${mae:-?}" "${hs:-?}" "${a:-?}" "${w:-?}" "${pp:-?}"
done
echo "P4-C 单步 0.1: MAE 79.2  逼近度 0.299  冷机功率 0.107  P_plant 0.021"
echo "P4-C 单步 0.5: MAE 80.0  逼近度 0.089  冷机功率 0.482  P_plant 0.208"
echo "门限：三项 dir_vr(36) 全部 <=0.10，且 h* 不退化 >10%"
echo "==========================================="
