#!/usr/bin/env bash
# P4 阶段 A：L-soft-B 的权重扫描 —— 精度 vs 物理一致性的权衡曲线。
#
# 两端已经测过（同样 60 epoch / delta=0.5 / seed 0，可直接比）：
#   lam=0   等价于 dt_evap_mode=free（#24 的病灶，eta 无梯度）  step1 MAE  60.9
#   硬等式   dt_evap_mode=derived（L-hard-B，只该用在有流量的站） step1 MAE 127.8
#
# §4.3 早就写明：无流量的站必须走软约束。这里扫中间三点把曲线补出来。
# 每轮同时报 eta 可辨识性（profile-likelihood），所以能看到
# 「花多少精度换到多少物理约束力」。
set -uo pipefail
EPOCHS="${EPOCHS:-60}"; STEPS="${STEPS:-100}"
ROOT="${PHYSWM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PY="${PHYSWM_PY:-python}"
cd "$ROOT"; mkdir -p experiments/results
export PYTHONIOENCODING=utf-8 PYTHONPATH="$ROOT"

LAMS="${LAMS:-0.03 0.1 0.3}"
for lam in $LAMS; do
  tag="p4a_softb_${lam}"
  echo "=== $(date +%m-%d' '%H:%M) $tag : lam_evap_bal=$lam ==="
  $PY -u experiments/train_yb3.py \
    --device cuda --mode curriculum --split-mode blocked \
    --epochs "$EPOCHS" --steps-per-epoch "$STEPS" \
    --H 48 --eval-H 48 --delta 0.5 --patience 100 --seeds 0 \
    --dt-evap-mode soft --lam-evap-bal "$lam" --tag "$tag" \
    > "experiments/results/${tag}.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "  !! 失败 rc=$rc"; tail -5 "experiments/results/${tag}.log" | sed 's/^/     /'; continue
  fi
  grep -E "step1:|被数据定住|未被定住|物理量" "experiments/results/${tag}.log" | sed 's/^/  /'
done

echo
echo "============ L-soft-B 权衡曲线 ============"
printf '%-10s %-11s %-9s %-12s %s\n' lam step1_MAE R2 eta可辨识 物理量
echo "0(free)    60.9        0.9972    否(无梯度)   COP=6.81 eta=0.395"
for lam in $LAMS; do
  f="experiments/results/p4a_softb_${lam}.log"; [ -f "$f" ] || continue
  mae=$(grep -oP 'step1: MAE \K[0-9.]+' "$f" | tail -1)
  r2=$(grep -oP 'step1: MAE [0-9.]+ R2 \K[0-9.]+' "$f" | tail -1)
  idt=$(grep -oP -e '\*\*\K(被数据定住|未被定住)' "$f" | tail -1)
  phys=$(grep -oP '物理量\(开机机中位\): \K.*' "$f" | tail -1 | cut -c1-40)
  printf '%-10s %-11s %-9s %-12s %s\n' "$lam" "${mae:-?}" "${r2:-?}" "${idt:-?}" "${phys:-?}"
done
echo "硬等式     127.8       0.9864    是(+95.6%)   COP=6.34 eta=0.366"
echo "=========================================="
