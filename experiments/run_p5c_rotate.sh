#!/usr/bin/env bash
# P5 阶段 C：轴 1「同档跨站」留出轮换 —— 把门限⑥的样本量从 n=1 做到 n=5。
#
# 为什么必须做：门限⑥要对外报的那个数（P5-A 的 hx 零样本 R²=0.9335）
# 目前建立在**一个站、一个 seed**上。项目自己的 §13 #25/#37 已经两次证明
# 单 seed 不可用于判定（dir_vr 跨 seed 差 4 倍、h* 差 2 倍）。
# 拿 n=1 的数当对外门限，低于本项目自己的标准。
#
# 做法：B 档有 5 个独立冷站，轮着留出 —— 相当于对 B 档做 5 折交叉验证。
#
#   ① hx                （P5-A / P5-B 已做）
#   ② yb3
#   ③ pa2（中温+低温）
#   ④ yb    （中温+低温）
#   ⑤ yc    （中温+低温）
#
# 三条口径纪律：
#
#   1. **pc3 每一轮都留出**。它是 D 档唯一成员，留着不训练，训练集才
#      始终不含 D 档，五轮的轴 1 数字彼此可比、也与 P5-A/P5-B 可比。
#      副产品：顺带得到 5 个独立的「跨档外推」测量。
#   2. **成对的温区回路必须同进同出**。pa2/yb/yc 各是同一冷站的两个回路
#      （§13 #52），只留一个等于把同楼同天气的数据留在训练集里，
#      「零样本」偏乐观。`registry.expand_holdout` 会强制补全，
#      这里仍写全名，让脚本自己就能读懂。
#   3. **配方与 P5-A/P5-B 逐项一致**（50 epoch × 200 步、单 seed 0）。
#      五个数来自同一配方才能合起来算均值与标准差。
#
# 五个臂的可比性（实测，报数时必须一起写）：
#
#   · **独立冷站数恒为 9** —— 这是训练多样性的正确度量，五臂完全一致
#   · 文件数 12/12/11/11/11（成对回路的站贡献 2 个文件）
#   · 训练窗口 169696 / 169540 / 161267 / 163819 / 153289
#     即 hx 0.0% / yb3 −0.1% / pa2 −5.0% / yb −3.5% / **yc −9.7%**
#
# yc 臂少 9.7% 的训练数据（它自己的窗口最多，留出后损失也最大）。
# 这不足以推翻跨臂比较，但 yc 若明显偏低，要先想到数据量而不是站点难度。
set -uo pipefail
ROOT="${PHYSWM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PY="${PHYSWM_PY:-python}"
cd "$ROOT"; mkdir -p experiments/results
export PYTHONIOENCODING=utf-8 PYTHONPATH="$ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 由 P5-B 的结果定档。跑之前必须显式确认，别让它默默用错配置。
AVAIL_JOINT="${AVAIL_JOINT:?请显式设置 AVAIL_JOINT（P5-B 定档后的值）}"
AVAIL_EACH="${AVAIL_EACH:?请显式设置 AVAIL_EACH}"
# 臂名前缀：不同配方的轮换要分开存，否则互相覆盖、事后分不清哪个是哪个
PREFIX="${PREFIX:-p5c}"
# 只跑指定的臂（逗号分隔，如 ARMS=pa2,yb,yc）。空 = 全跑。
ARMS="${ARMS:-}"
want () { [[ -z "$ARMS" ]] && return 0; [[ ",$ARMS," == *",$1,"* ]]; }

pc3="pc3"
run_arm () {
  local tag="$1"; shift
  local hold="$1"; shift
  echo "=== $(date +%m-%d' '%H:%M) $tag : 留出 $hold ==="
  $PY -u experiments/train_multisite.py \
    --device cuda --epochs 50 --steps 200 --batch-size 64 \
    --H 48 --select-H 48 --delta 0.5 \
    --lam-lat 0.1 --lam-evap-bal 0.03 --lam-dir 0.0 \
    --avail-drop-joint "$AVAIL_JOINT" --avail-drop-each "$AVAIL_EACH" \
    --sample proportional --hold-out "$hold" \
    --seeds 0 --tag "$tag" --log-every 5 \
    > "experiments/results/${tag}.log" 2>&1
  local rc=$?
  echo "  train rc=$rc"
  if [[ $rc -ne 0 ]]; then
    tail -20 "experiments/results/${tag}.log"
    return
  fi
  # 训练损失不是指标（eval_zeroshot.py 模块文档）。门限⑥要的是 R²。
  $PY -u experiments/eval_zeroshot.py \
    --ckpt "experiments/results/${tag}/model_seed0.pt" \
    --device cuda --eval-H 48 --hold-out "$hold" --tag "$tag" \
    > "experiments/results/${tag}_eval.log" 2>&1
  echo "  eval rc=$?"
  tail -6 "experiments/results/${tag}_eval.log"
}

want yb3     && run_arm "${PREFIX}_hold_yb3"    "yb3,${pc3}"
want pa2  && run_arm "${PREFIX}_hold_pa2" "pa2_中温,pa2_低温,${pc3}"
want yb      && run_arm "${PREFIX}_hold_yb"     "yb_中温,yb_低温,${pc3}"
want yc      && run_arm "${PREFIX}_hold_yc"     "yc_中温,yc_低温,${pc3}"

echo "=== $(date +%m-%d' '%H:%M) 轮换完成 ==="
