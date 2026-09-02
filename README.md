# 自回归物理世界模型（工业冷站）

面向工业冷站的自回归物理世界模型推理内核。模型在潜空间上自回归推进，
解码出的物理量受**硬约束**结构性保证（逼近度非负、能量守恒、卡诺下界等 11 项），
而不是靠损失项软性惩罚。

本仓库公开**源码、实验结果与训练权重**。原始运行数据（各冷站的时序 CSV）
属于第三方生产数据，不包含在内；站点一律以代号出现。

---

## 结果

### 单站自回归（P2 / P3）

站点 `yb3`，`d=192 / L_enc=4 / L_trans=2`，**3.59×10⁶ 参数**，窗口 `W=16`，预测域 `H=48`。

| 项 | 结果 | 来源 |
|---|---|---|
| 单步 MAE（H=1，5 seed，blocked split） | **58.1 ± 4.9 kW** | `experiments/results/p2_single/summary.json` |
| 单步 R²（同上） | **0.9972 ± 0.0004** | 同上 |
| 11 项硬约束违例（5 seed 合计） | **全部 0.000** | 同上 `runs[].hard` |
| 有效自回归步长 `h*`（δ=0.5，5 seed） | **81 / 84 / 86 / 96 / 96** | `experiments/results/p3b_main_d0.5/` |

`h*` 定义为 R² 仍 ≥ 0.8 且 MAE 不超过单步 3 倍的最大自回归步数。

**一个负面结果一并公开**：把随机 blocked split 换成严格按时间切分（`p2_chrono`），
同一配置下 MAE 从 58.1 劣化到 **2229.6**，`h*` 归零。分布漂移是这套数据上的真问题，
不是被调参掩盖掉的细节。

### 多站预训练与零样本迁移（P5-A）

12 站联合预训练，留出 2 站从未参与训练，全量 test 口径评测：

| 站点 | 是否留出 | R² | MAE (kW) |
|---|---|---:|---:|
| `yc_中温` | 否 | 0.9955 | 90.7 |
| `yb_中温` | 否 | 0.9947 | 102.5 |
| `yb3` | 否 | 0.9932 | 112.8 |
| `zx` | 否 | 0.9930 | 70.9 |
| `pa2_中温` | 否 | 0.9886 | 63.5 |
| `tx` | 否 | 0.9838 | 9.9 |
| `pc2` | 否 | 0.9819 | 271.3 |
| `pb1` | 否 | 0.9802 | 28.1 |
| `bh` | 否 | 0.9537 | 22.5 |
| **`hx`** | **是** | **0.9011** | 447.4 |
| **`pc3`** | **是** | **−1.6956** | 2912.8 |

零样本一过一崩，原因是结构性的：站点按逐台功率标签的齐备程度分 A/B/C/D 四档，
**D 档只有 `pc3` 一个成员**。留出它等于训练集里没有任何站具备那个能力模式，
考的不是泛化而是外推到一个空档。见 `physwm/data/registry.py` 的说明。

`p5b_availdrop`（训练期随机丢弃可用性标签）把 `pc3` 从 −1.70 抬到 **−0.52**，
但代价是训练站普遍退化（`bh` 0.9537 → 0.8863）。净收益为负，未采纳。

---

## 仓库内容

```
physwm/            模型与训练库
  model/           encoder / transition / decoder / layers
  train/           训练循环、损失、课程、可用性丢弃、多站
  eval/            指标、方向性、可辨识性、探针
  data/            数据 schema、描述子、站点登记表、时序留出
experiments/       训练与评测入口 + 运行脚本
  results/         各阶段结果 JSON + 训练权重 (.pt)
tools/             诊断脚本（梯度流、站点独立性、可用性矩阵、标定…）
tests/             14 个测试
artifacts/         归一化参数与描述子 (.npz / bundle .json)
```

**不包含**：原始数据 CSV、内部规划文档、运行日志。

## 站点代号

`bh` `tx` `hx` `yb3` `yb` `yc` `zx` `pa2` `pb1` `pc2` `pc3` 均为代号，
不对应可识别的实际站点。`_中温` / `_低温` 是同一冷站的两个温区回路
（独立设备，但同楼、同天气、同负荷排程，评测时必须一起留出）。

14 个数据文件只对应 **11 个独立冷站**，对应关系在 `physwm/data/registry.py`。

## 环境

```bash
pip install torch numpy "pandas>=2.3,<3.0" pytest
```

`pandas` 刻意钉在 2.x：pandas 3 的 `to_numpy()` 默认返回只读数组，与本项目行为不一致。

## 使用权重

`experiments/results/*/model_seed*.pt` 是纯 `state_dict`（`OrderedDict[str, Tensor]`），
不含任何元数据：

```python
import torch
from physwm.model.world_model import WorldModel

sd = torch.load("experiments/results/p2_single/model_seed0.pt", map_location="cpu")
model = WorldModel(...)          # 超参见同目录 summary.json 的 args
model.load_state_dict(sd)
model.eval()
```

每份权重同目录下的 `summary.json` 里 `args` 字段记录了完整超参。

## 复现说明

训练数据不公开，因此**训练过程无法在本仓库内复现**。
公开的是模型实现、完整实验配置、逐 seed 的结果 JSON 与训练好的权重，
可用于核对指标口径、复用模型结构，或在自有数据上重跑
（数据 schema 见 `physwm/data/schema.py`）。

## 许可

保留所有权利。本仓库公开供查阅与交流，尚未授予任何使用许可。
如需使用请先联系。
