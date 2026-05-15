# 标准工作流程

每一轮迭代实验的完整工作流程。

---

## 流程总览

1. **用户反馈** — 用户提供最新实验路径/截图/日志
2. **分析** — 诊断问题（指标分析 + 行为诊断 + 环境检查）
3. **设计** — 更新 spec 文档
4. **实现** — 修改代码 + 测试
5. **记录** — 更新 changelog
6. **给出脚本** — 输出下一轮实验的运行命令，由用户执行
7. → 用户运行实验后回到 1

> **注意**：训练实验由用户在本地运行，AI 不运行训练。AI 的角色是分析结果、修改代码、给出下一轮脚本。

---

## 1. 用户反馈

用户提供以下信息之一（或组合）：
- 最新实验目录路径（如 `experiments/round20_hit_reward/`）
- 训练图表截图（`training_metrics_ep<N>.png`）
- 终端日志片段
- evaluate 运行后的观察（如"红球彩球顺序不对"、"开球总碰彩球"）

---

## 2. 分析：诊断问题

按照 `spec/experiment_analysis.md` 中的流程执行：

### 2.1 指标分析
- 查看训练图表（reward / pots / fouls / angle / power）
- 数值分析（5段趋势、犯规类型占比、精确进球率）
- 判断是否有异常或停滞

### 2.2 行为诊断
```bash
# 修改 MODEL_PATH 为最新检查点
python diagnose_trajectory.py

# 可视化运行
python evaluate.py --load_model experiments/<run_name>/sac_snooker_ep<N>.pt --num_episodes 1
```

### 2.3 环境检查（如有异常行为）
- 阶段切换是否正确
- 犯规判定是否合理
- 奖励计算是否正确

### 2.4 形成结论
- 明确问题根因（缺少特征？reward 不合理？代码 bug？）
- 确定下一步修改方向

---

## 3. 设计：更新 spec 文档

**根据分析结论，先更新设计文档再改代码**。

### 如果修改特征 → 更新 `spec/features.md`
- 新增/删除了哪些维度
- 归一化方式
- 与其他特征的关系

### 如果修改奖励 → 更新 `spec/rewards.md`
- 新增/修改了哪些奖励项
- 奖励层级是否变化
- 设计理由（为什么这个值）

### 如果修改流程 → 更新 `spec/experiment_analysis.md`
- 新增了什么诊断步骤
- 新增了什么指标

---

## 4. 实现：修改代码

### 代码修改范围

| 修改类型 | 涉及文件 |
|---------|----------|
| 特征/观测空间 | `environment/pooltool_env.py` (_get_obs), `test.py` (维度断言) |
| 奖励结构 | `environment/pooltool_env.py` (RewardConfig, _compute_reward) |
| 动作空间 | `environment/pooltool_env.py` (step) |
| 训练追踪指标 | `train.py` (TrainingMetrics, 训练循环, plot, 日志) |
| 诊断工具 | `diagnose_trajectory.py` |
| 游戏规则 bug | `environment/pooltool_env.py` (阶段切换, 犯规判定等) |
| 渲染 bug | `environment/pooltool_env.py` (pygame 渲染代码) |

### 修改后必须做

1. **跑测试**: `python test.py` — 全部通过
2. **快速验证**: 跑几步随机 action 确认不 crash，输出合理
3. **如果改了观测空间维度**: 更新 `test.py` 中的维度断言

---

## 5. 记录：更新 changelog

修改完成后，在 `docs/changelog.md` 中添加新的 Round 记录：

### 记录模板

```markdown
## Round <N>：<标题>

**目标：** <一句话说明这轮要解决什么问题>

### 问题分析

<诊断发现了什么问题，数据支撑>

### 修改方案

<做了什么改动，为什么>

### 代码改动

- `environment/pooltool_env.py`: <具体改了什么>
- `train.py`: <具体改了什么>
- ...

### 验证

<测试结果、快速验证结果>
```

### 记录原则

- 每个独立的改动点是一个 Round（或 Round X.5 如果是 bugfix）
- Bug 修复要说明影响范围（从第几轮开始存在）
- 记录"为什么"比"做了什么"更重要
- 实验结果出来后补充结果数据

---

## 6. 给出下一轮实验脚本

所有修改完成后，输出下一轮实验的运行命令供用户执行：

```bash
python train.py --algo sac --num_episodes 5000 --run_name round<N+1>_<描述>
```

用户运行后，等待结果，然后回到步骤 1 反馈新的实验数据。

---

## 文件结构约定

```
snooker_rl/
├── spec/                          # 设计规范（先更新这里再改代码）
│   ├── workflow.md                # 本文件：工作流程
│   ├── features.md               # 特征/观测空间设计
│   ├── rewards.md                 # 奖励结构设计
│   └── experiment_analysis.md     # 实验分析流程
├── docs/
│   └── changelog.md              # 所有 Round 的改动记录
├── environment/
│   └── pooltool_env.py           # 核心环境（特征+奖励+规则）
├── train.py                      # 训练脚本（指标追踪+可视化）
├── diagnose_trajectory.py        # 轨迹诊断工具
├── evaluate.py                   # 可视化评估工具
├── test.py                       # 自动化测试
└── experiments/                   # 实验产出（模型+指标+图表）
    └── round<N>_<name>/
```
