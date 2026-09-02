#!/usr/bin/env bash
# P4 阶段 C：方向损失权重扫描（短课程单 seed 定量级）。
#
# 为什么先扫再定档：lam_dir 没有任何先验，而 P4 门限里最难的一条是
# 「dir_vr(36) <= 0.10 **且 h* 不退化超过 10%**」—— v3 的教训正是
# 加硬 mono 会伤稳定性。必须先看清「压低 dir_vr」与「保住 h*」的权衡。
#
# 基线（P4-B，lam_dir=0，5 seed 中已完成 3 个）：
#     step1 MAE  84.7 / 79.1 / 77.5      h* 全部顶格 96
#     塔频->逼近度   h=1 0.491/0.124/0.158   h=36 0.372/0.114/0.145
#     供水温->P_plant h=36 0.143/0.126/0.457
#
# 用 60 epoch / eval-H 48 快扫（与 L-soft-B 那轮同口径，2.4h/臂），
# 只为定量级；选出的 lam 再按 P4-B 的完整配置跑满 5 seed。
set -uo pipefail
EPOCHS="${EPOCHS:-60}"; STEPS="${STEPS:-100}"
ROOT="${PHYSWM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PY="${PHYSWM_PY:-python}"
cd "$ROOT"; mkdir -p experiments/results
export PYTHONIOENCODING=utf-8 PYTHONPATH="$ROOT"
# 卡上常驻别人的 VLLM（实测已涨到 25.7 GB），碎片化会直接导致 OOM
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LAMS="${LAMS:-0.1 0.5 2.0}"
for lam in $LAMS; do
  tag="p4c_dir_${lam}"
  echo "=== $(date +%m-%d' '%H:%M) $tag : lam_dir=$lam ==="
  $PY -u experiments/train_yb3.py \
    --device cuda --mode curriculum --split-mode blocked \
    --epochs "$EPOCHS" --steps-per-epoch "$STEPS" \
    --H 48 --eval-H 48 --delta 0.5 --patience 100 --seeds 0 \
    --dt-evap-mode soft --lam-evap-bal 0.03 \
    --lam-dir "$lam" --dir-vr --tag "$tag" \
    > "experiments/results/${tag}.log" 2>&1
  rc=$?
  [ $rc -ne 0 ] && { echo "  !! 失败 rc=$rc"; tail -5 "experiments/results/${tag}.log"|sed 's/^/     /'; continue; }
  grep -E "step1:|h\*=|approx |P_plant " "experiments/results/${tag}.log" | tail -6 | sed 's/^/  /'
done

echo
echo "======== 方向损失权衡（门限 dir_vr(36)<=0.10 且 h* 不退化>10%）========"
printf '%-8s %-10s %-7s %-14s %-16s\n' lam step1_MAE h\* 塔频->逼近度h36 供水温->P_plant_h36
for lam in $LAMS; do
  f="experiments/results/p4c_dir_${lam}.log"; [ -f "$f" ] || continue
  mae=$(grep -oP 'step1: MAE \K[0-9.]+' "$f"|tail -1)
  hs=$(grep -oP 'h\*=\K[0-9]+' "$f"|tail -1)
  a=$(grep -oP 'approx\s+-1\s+[0-9.]+\s+\K[0-9.]+' "$f"|tail -1)
  pp=$(grep -oP 'P_plant\s+-1\s+[0-9.]+\s+\K[0-9.]+' "$f"|tail -1)
  printf '%-8s %-10s %-7s %-14s %-16s\n' "$lam" "${mae:-?}" "${hs:-?}" "${a:-?}" "${pp:-?}"
done
echo "参考 lam_dir=0（P4-B seed0-2）: MAE 84.7/79.1/77.5  h*=96  逼近度 .372/.114/.145  P_plant .143/.126/.457"
echo "======================================================================="
