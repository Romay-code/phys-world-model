#!/usr/bin/env bash
# P3 阶段 A：delta 扫描 + lam_lat 消融。
#
# delta 是转移映射 z' = z + delta*G(z) 的步长，也是**唯一真正的 Lipschitz 旋钮**
# （设计文档 §4.4：L = 1 + delta*Lip(G) >= 1，原文写的 "L <= 1" 对残差映射不可达）。
# H 步误差约 exp(delta*Lip(G)*H)。
#
# 但 P3 smoke（H_max=8, 24 epoch）实测误差放大只有 +13.6%，远好于预期，
# 说明多步稳定性未必主要由 delta 决定。故额外加一臂 lam_lat=0 的消融：
#   - 若 lam_lat=0 后 h* 大幅下降 -> 隐一致性是主力，delta 次要
#   - 若基本不变            -> Lipschitz/delta 是主力
# 这个结论会反过来修正 §4.4 的判断，比单纯调参有价值。
#
# 扫描用短课程 + 单 seed 定档；选出的配置再跑满 5 seed（阶段 B）。
# 完整课程约 7.5 h/seed，5 组 x 5 seed = 187 h，不可接受。
#
#   bash experiments/run_p3_sweep.sh
#   EPOCHS=120 bash experiments/run_p3_sweep.sh          # 更快的粗扫
set -uo pipefail

EPOCHS="${EPOCHS:-200}"
STEPS="${STEPS:-100}"
HMAX="${HMAX:-48}"
SEEDS="${SEEDS:-0}"
ROOT="${PHYSWM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PY="${PHYSWM_PY:-python}"

cd "$ROOT"
mkdir -p experiments/results
export PYTHONIOENCODING=utf-8 PYTHONPATH="$ROOT"

# 每行： tag  delta  lam_lat
ARMS="${ARMS:-
d1.0    1.0   0.1
d0.5    0.5   0.1
d0.2    0.2   0.1
d0.05   0.05  0.1
d1.0_nolat 1.0 0.0
}"

echo "$ARMS" | while read -r name d lat; do
  [ -z "${name:-}" ] && continue
  tag="p3_${name}"
  echo "=== $(date +%H:%M) $tag : delta=$d lam_lat=$lat ==="
  $PY -u experiments/train_yb3.py \
    --device cuda --mode curriculum --split-mode blocked \
    --epochs "$EPOCHS" --steps-per-epoch "$STEPS" \
    --H "$HMAX" --eval-H "$HMAX" \
    --delta "$d" --lam-lat "$lat" \
    --patience 100 --seeds $SEEDS --tag "$tag" \
    > "experiments/results/${tag}.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "  !! 失败 rc=$rc，末尾输出："
    tail -5 "experiments/results/${tag}.log" | sed 's/^/     /'
    continue     # 一臂失败不影响其余
  fi
  grep -E "h\*=|物理量|step1: MAE" "experiments/results/${tag}.log" | sed 's/^/  /'
done

echo
echo "================= 扫描汇总 ================="
printf '%-12s %-7s %-8s %-7s %-11s %-12s %s\n' \
       arm delta lam_lat h\* step1_MAE 误差放大 硬约束
echo "$ARMS" | while read -r name d lat; do
  [ -z "${name:-}" ] && continue
  f="experiments/results/p3_${name}.log"
  [ -f "$f" ] || continue
  hs=$(grep -oP 'h\*=\K[0-9.]+' "$f" | tail -1)
  mae=$(grep -oP 'step1: MAE \K[0-9.]+' "$f" | tail -1)
  amp=$(grep -oP '\(\+\K[0-9.]+(?=%\))' "$f" | tail -1)
  hard=$(grep -oP '硬约束: \K.*' "$f" | tail -1)
  printf '%-12s %-7s %-8s %-7s %-11s %-12s %s\n' \
         "$name" "$d" "$lat" "${hs:-?}" "${mae:-?}" "+${amp:-?}%" "${hard:-?}"
done
echo "============================================"
