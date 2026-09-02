#!/usr/bin/env bash
# P5 阶段 F：轴 3 时段留出 —— 换个没见过的季节，模型还行吗。
#
# 这是三条留出轴里唯一**没有任何测量**的一条（设计文档 §6 P5）。
# 现有划分全部是 blocked（分块交错，每个 split 覆盖全年，§13 #7 的决定），
# 于是至今没有一处在考验季节外推 —— 而「上线后撞进没见过的季节」是必然发生的。
#
# 口径三条，缺一条这个数就不是季节外推：
#
#   1. **按月份序数留出（7、8 月），跨全部年份**。绝对时间窗留不掉 bh 的夏天：
#      bh 覆盖 2023-12~2024-09，其余 13 站从 2024-10 起，两段完全不重叠，
#      留「2025 年夏天」时 bh 的 2024 年夏天还在训练集里。
#   2. **不做站点留出**（--hold-out ""）。轴 3 与轴 1 是正交的两条轴，
#      混在一起就说不清失败该记在哪一轴上。14 站全部参与训练，
#      test 换成各站的七八月。
#   3. **归一化/描述符/量纲在切完之后重算**（build_site_bundle 负责）。
#      沿用留出前那份就把夏天的统计量带了进去 —— 与 §13 #8 同型的静默失效。
#
# 配方与轴 1 的交付配方逐项一致（不开可用性丢弃，§13 #62），可直接对照。
#
# 实测预演（tools/diag_temporal_holdout.py）：留出七八月后各站仍剩
# 77%~89% 的可训窗口，留出集合计 75,872 窗口，bh 的夏天确实被一并留出。
set -uo pipefail
ROOT="${PHYSWM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PY="${PHYSWM_PY:-python}"
cd "$ROOT"; mkdir -p experiments/results
export PYTHONIOENCODING=utf-8 PYTHONPATH="$ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

tag="p5f_season78"
echo "=== $(date +%m-%d' '%H:%M) $tag : 14 站全训，留出七八月 ==="
$PY -u experiments/train_multisite.py \
  --device cuda --epochs 50 --steps 200 --batch-size 64 \
  --H 48 --select-H 48 --delta 0.5 \
  --lam-lat 0.1 --lam-evap-bal 0.03 --lam-dir 0.0 \
  --avail-drop-joint 0 --avail-drop-each 0 \
  --sample proportional --hold-out "" --season-holdout "7,8" \
  --seeds 0 --tag "$tag" --log-every 5 \
  > "experiments/results/${tag}.log" 2>&1
rc=$?
echo "train rc=$rc"
tail -6 "experiments/results/${tag}.log"

if [[ $rc -eq 0 ]]; then
  echo "=== $(date +%m-%d' '%H:%M) $tag : 14 站在**七八月**上评测 ==="
  $PY -u experiments/eval_zeroshot.py \
    --ckpt "experiments/results/${tag}/model_seed0.pt" \
    --device cuda --eval-H 48 --hold-out "" --season-holdout "7,8" \
    --tag "$tag" > "experiments/results/${tag}_eval.log" 2>&1
  echo "eval rc=$?"
  tail -22 "experiments/results/${tag}_eval.log"
fi
echo "=== $(date +%m-%d' '%H:%M) 完成 ==="
