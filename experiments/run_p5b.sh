#!/usr/bin/env bash
# P5 阶段 B：输入可用性丢弃（对照 P5-A 的唯一变量）。
#
# P5-A 的结论与遗留：
#   hx      零样本 R² 0.9335 ✅ —— B 系历史崩溃站（B0/B1 为 −5.94 / −7.64）
#                                   在新架构上不崩，设计文档 §5.2 点名的问题已答
#   pc3 零样本 R² −1.6377 ❌ —— 门限⑥（≥0.80）未过
#
# 三轮诊断把根因定死（详见 内部实施记录 §2）：
#   diag_zeroshot_calib  corr 0.883 但 R² 为负，一个比例因子 k=2.303 就能拉回
#                        0.699 —— 是**标定**问题不是表征问题
#   diag_avail_matrix    pc3 相对训练站中位缺五条，且全是负荷/功率类软测量
#   diag_avail_ablate    在 yb3 上按同样组合掩掉，复现 pc3 签名：
#                        R² 0.994 -> −0.387、corr 0.997 -> 0.047、
#                        预测 std/真值 std 0.97 -> 0.23（pc3 实测 0.26）
#
# 修法：训练期随机把整条输入通道掩成缺测，逼模型不敢依赖任何一条
# 「不是每个站都有」的通道。这是把设计文档 §2.2 的能力矩阵纪律从**输出**扩到**输入**。
#
# 除 --avail-drop-* 外，逐项与 run_p5a.sh 一致 —— 这一轮要能直接归因。
set -uo pipefail
ROOT="${PHYSWM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PY="${PHYSWM_PY:-python}"
cd "$ROOT"; mkdir -p experiments/results
export PYTHONIOENCODING=utf-8 PYTHONPATH="$ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

tag="p5b_availdrop"
echo "=== $(date +%m-%d' '%H:%M) $tag : 12 站联合预训练 + 输入可用性丢弃 ==="
$PY -u experiments/train_multisite.py \
  --device cuda --epochs 50 --steps 200 --batch-size 64 \
  --H 48 --select-H 48 --delta 0.5 \
  --lam-lat 0.1 --lam-evap-bal 0.03 --lam-dir 0.0 \
  --avail-drop-joint 0.15 --avail-drop-each 0.20 \
  --sample proportional --hold-out "hx,pc3" \
  --seeds 0 --tag "$tag" --log-every 2 \
  > "experiments/results/${tag}.log" 2>&1
rc=$?
echo "train rc=$rc"
tail -6 "experiments/results/${tag}.log"

# 训练损失不是指标（eval_zeroshot.py 模块文档）。门限⑥要的是 R²，
# 必须单独评一遍，且与 P5-A 用同一条命令，否则两轮不可比。
if [[ $rc -eq 0 ]]; then
  echo "=== $(date +%m-%d' '%H:%M) $tag : 14 站正式评测 ==="
  $PY -u experiments/eval_zeroshot.py \
    --ckpt "experiments/results/${tag}/model_seed0.pt" \
    --device cuda --eval-H 48 --hold-out "hx,pc3" \
    --tag "$tag" \
    > "experiments/results/${tag}_eval.log" 2>&1
  echo "eval rc=$?"
  tail -25 "experiments/results/${tag}_eval.log"
fi
echo "=== $(date +%m-%d' '%H:%M) 完成 ==="
