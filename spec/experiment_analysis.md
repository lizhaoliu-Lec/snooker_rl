# 实验分析流程

每轮实验完成（或中间检查点）后，按以下流程分析和诊断。

---

## 1. 指标分析

### 1.1 查看训练图表

```bash
# 查看最新的训练指标图
open experiments/<run_name>/training_metrics_ep<N>.png
```

重点关注：
- **Episode Reward**: MA50 是否稳定上升
- **Avg |Angle Offset|**: 是否收敛（越小越精准，弹库边打法时会偏大）
- **Avg Power (V0)**: 是否稳定在合理范围（2-4 m/s 为宜）
- **Pots per Episode**: Intentional(绿) vs Lucky(橙) 的比例变化
- **Foul Type Breakdown**: miss_ball / wrong_ball / white_pocket 各自趋势
- **Foul Reward / Pot Reward**: 分别的趋势，不能被另一个淹没
- **Win Rate**: 是否趋近 50%（self-play 平衡点）
- **Policy/Critic Loss**: 是否收敛

### 1.2 数值分析

```bash
# 加载 metrics JSON 做详细分析
python train.py --analyse experiments/<run_name>/metrics_ep<N>.json
```

或写脚本分 5 段看趋势：
- Reward / Pots / Fouls / Angle / Power / WinRate
- 犯规类型细分：miss_ball / wrong_ball / white_pocket 各段占比
- 精确进球 vs 运气进球比例

### 1.3 关键判断指标

| 指标 | 健康信号 | 警报信号 |
|------|---------|---------|
| Reward | 稳定上升 | 持平或下降 |
| Foul Rate | 逐段下降 | 卡住不动 |
| Intentional Pots | 逐渐增加 | 始终为 0 |
| Angle Offset | 逐渐收敛 | 不收敛或发散 |
| Win Rate | 趋近 50% | 远低于 30% 或远高于 70% |
| Policy Loss | 收敛稳定 | 震荡或发散 |

---

## 2. 行为诊断

### 2.1 运行轨迹诊断

```bash
python diagnose_trajectory.py
```

逐步检查：
- **目标球选择**：是否选了近处/可达的球？还是盲选远处球？
- **出杆方向**：角度偏移是否合理？有没有朝反方向打？
- **进球质量**：精确进球(★PRECISE) vs 运气进球(lucky) 比例
- **犯规模式**：wrong_ball 的 first_contact 是什么？是固定碰到某颗彩球？

### 2.2 可视化运行

```bash
python evaluate.py --load_model experiments/<run_name>/sac_snooker_ep<N>.pt --num_episodes 1 --delay 500
```

肉眼观察：
- 白球放置位置是否合理（D zone 内）
- 出杆方向是否朝目标球
- 有没有尝试弹库边
- 红球/彩球阶段切换是否正确
- D 弧形渲染是否正确

---

## 3. 环境检查

当行为明显不合理时，需要检查游戏环境逻辑：

### 3.1 阶段切换检查

- 进红球后 → phase 应该变为 "color"
- color 阶段无论犯规/没进/进球后 → 都应该回到 "red"（只要还有红球）
- 所有红球清完 → 进入 "final_colors"

### 3.2 犯规判定检查

- `wrong_ball`: first_contact 是否真的是非法球？
- `miss_ball`: 白球是否真的没碰到任何球？（V0 太小？方向完全偏？）
- 犯规后 phase 是否正确回退？

### 3.3 奖励合理性检查

- 精确进球条件：`first_contact == chosen_target` AND `chosen_target in pocketed` AND `pocketed_into[target] == chosen_pocket`
- 命中目标球：`first_contact == chosen_target` 但没进袋
- 运气进球：合法进球但不满足精确条件

### 3.4 观测空间检查

```python
state, _ = env.reset()
print(f'Obs shape: {state.shape}')  # 应该是 (75,)
clearance = state[-21:]  # 最后21维是可达性
```

---

## 4. 决策：继续训练 or 修改

| 情况 | 决策 |
|------|------|
| 指标稳定上升，行为合理 | 继续训练，增加 episodes |
| 指标停滞但无 bug | 调参（lr, alpha, reward 权重） |
| 发现 reward 不合理 | 修改 reward 结构，重新训练 |
| 发现环境 bug | 修复 bug，之前结果作废，重新训练 |
| 模型完全学不会某件事 | 检查是否缺少特征/reward 信号 |
