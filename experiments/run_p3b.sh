#!/usr/bin/env bash
# P3 阶段 B：最优配置跑满多 seed。
#
# 阶段 A（单 seed 扫描）的结论：
#   - delta=0.5 在目标视野 h=36 最优（MAE 106.7 / R2 0.9921），且 36->48 步
#     误差几乎不再增长（106.7 -> 106.9），已进入稳态
#   - delta=1.0 在短程最优（h1 53.8），两者在 h≈24-30 交叉
#   - L 与长程误差**非单调**：L 最小的 delta=0.05 长程最差 —— 推翻 §4.4
#     「Lipschitz 是 h* 决定因素」的论断
#
# 但**每臂只有 1 个 seed**，而 P2 实测多步指标跨 seed 波动极大（误差放大跨 9 倍）。
# 所以这里同时跑两个竞争配置并带上 seed 方差，不拿单 seed 的排序直接定档。
#
# 另外把评测视野从 48 放到 96：阶段 A 有 4/5 臂 h* 顶格在 48，那个数字只说明
# 「>=48」。训练仍到 48 步，评测到 96 步是**超训练视野的外推**，恰恰是要测的。
set -uo pipefail

EPOCHS="${EPOCHS:-200}"
STEPS="${STEPS:-100}"
EVAL_H="${EVAL_H:-96}"     # 数据窗口与评测视野
TRAIN_H="${TRAIN_H:-48}"   # 课程的 H_max
ROOT="${PHYSWM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PY="${PHYSWM_PY:-python}"

cd "$ROOT"
mkdir -p experiments/results
export PYTHONIOENCODING=utf-8 PYTHONPATH="$ROOT"

# tag  delta  seeds
ARMS="${ARMS:-
main_d0.5   0.5   0 1 2 3 4
alt_d1.0    1.0   0 1 2
}"

echo "$ARMS" | while read -r name d seeds; do
  [ -z "${name:-}" ] && continue
  tag="p3b_${name}"
  echo "=== $(date +%m-%d' '%H:%M) $tag : delta=$d seeds=[$seeds] 训练H=$TRAIN_H 评测H=$EVAL_H ==="
  $PY -u experiments/train_yb3.py \
    --device cuda --mode curriculum --split-mode blocked \
    --epochs "$EPOCHS" --steps-per-epoch "$STEPS" \
    --H "$EVAL_H" --train-H "$TRAIN_H" --eval-H "$EVAL_H" \
    --delta "$d" --patience 100 --seeds $seeds --tag "$tag" \
    > "experiments/results/${tag}.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "  !! 失败 rc=$rc"; tail -5 "experiments/results/${tag}.log" | sed 's/^/     /'
    continue
  fi
  grep -E "step1 MAE|step1 R2|^  h\*" "experiments/results/${tag}.log" | sed 's/^/  /'
done

echo
echo "================= P3-B 汇总 ================="
echo "$ARMS" | while read -r name d seeds; do
  [ -z "${name:-}" ] && continue
  f="experiments/results/p3b_${name}.log"
  [ -f "$f" ] || continue
  echo "--- $name (delta=$d) ---"
  grep -E "step1 MAE|step1 R2|^  h\*|警告" "$f" | tail -4
done
echo "============================================="
