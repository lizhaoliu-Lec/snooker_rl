# 改进日志 (Changelog)

记录项目从初始版本到当前版本的所有迭代改进，包括发现的问题、解决方案和实验结果。

---

## Round 1：环境搭建与 Bug 修复

**目标：** 让 `python train.py --num_episodes 1000 --render` 能正常运行。

### 发现的问题

1. **`from pymunk.pygame_util import draw` ImportError**
   - pymunk 7.x 移除了此模块
   - 修复：删除未使用的 import

2. **观测空间维度不匹配**
   - 声明 52 维，实际只有 50 维
   - 修复：`n_features = 4 + 15×2 + 6×2 + 4 = 50`

3. **PPO 初始化崩溃**
   - `cfg=None` 时调用 `None.get()` 报错
   - 修复：`if cfg is None: cfg = {}`

4. **棕球和白球位置重叠**
   - 两者都在 (300, 300)
   - 修复：棕球移到 baulk_x = 264

### 结果

训练可以正常运行，但 1000 episode 后 avg break 始终为 0。

---

## Round 2：游戏规则完善

**目标：** 审查 reward 设计和环境规则正确性。

### 改进内容

1. **碰撞检测系统**
   - 定义碰撞类型：`COL_WHITE`, `COL_RED`, `COL_COLOR`, `COL_CUSHION`
   - 使用 pymunk 7.x 的 `space.on_collision()` API
   - 跟踪白球首次碰触的球（`first_contact`）

2. **红彩交替阶段追踪**
   - 状态机：`red` → `color` → `red` → ... → `final_colors`
   - 彩球进球后 re-spot（仍有红球时）
   - 最终彩球按固定顺序清台

3. **犯规检测**
   - 空杆（白球未接触任何球）
   - 错误首触（红球阶段先碰彩球等）
   - 白球落袋

4. **物理空间泄漏修复**
   - reset 时创建全新 `pymunk.Space`，而非逐个移除对象

### 结果

规则更完善了，但 reward 仍然太稀疏——agent 一直无法进球。

---

## Round 3：Dense Reward 设计

**目标：** 解决 1000 episode 训练后 avg break = 0 的问题。

### 分析

Agent 从未成功进球 → reward 信号几乎全是负的 → 无法学到任何有效策略。

### 改进内容

引入 **4 层课程式 dense reward**：

| 层级 | 奖励信号 | 值 |
|------|---------|-----|
| Layer 1 | 白球接触/未接触目标球 | +0.1 / -2.0 |
| Layer 2 | 正确首触（红球阶段碰红球） | +0.5 |
| Layer 3 | 击球后目标球离袋口更近 | clip(approach × 3, -0.5, 2.0) |
| Layer 4 | 红球进球 / 彩球进球 | +5.0 / 分值 × 1.5 |

同时调整超参数：
- `entropy_coef`: 0.01 → 0.05（增加探索）
- `update_interval`: 2048 → 512（更频繁更新）

### 结果

| 指标 | 改进前 | 改进后 |
|------|--------|--------|
| Avg Reward (ep 1000) | ~-48 | ~-3.71 |
| Avg Break | 0 | 0（偶有 0.1） |

Reward 大幅上升，但 Break 仍然为 0——agent 学会了接触球、推球靠近袋口，但还不会进球。

---

## Round 4：动作空间重构 + Self-Play

**目标：** 完整的动作空间（母球放置 + 目标球选择 + 相对角度 + 力度）+ 双人对弈。

### 核心改进

1. **动作空间从 2 维扩展到 5 维**
   - 旧：`[angle, power]`（绝对角度）
   - 新：`[place_x, place_y, target_idx, angle_offset, power]`

2. **目标球坐标系**
   - 击球角度不再是绝对方向，而是相对于白球→目标球连线的偏移（±30°）
   - 模拟人类思维：先选球 → 再决定偏移

3. **D 区母球放置**
   - 开局和对手犯规后，可在 D 区半圆内选择母球位置
   - `[place_x, place_y]` → 归一化映射到 D 区

4. **Self-Play 训练**
   - 两个玩家共享同一个策略网络
   - 观测中包含 `current_player` 信息
   - 环境自动处理换人逻辑

5. **文件全面更新**
   - `algorithms/ppo.py`：网络加深（3 层 + LayerNorm），log_std 初始化为 -0.5
   - `train.py`：self-play 训练循环
   - `evaluate.py`：适配新动作空间
   - `test.py`：新增 ball-in-hand、target selection、self-play 测试

### 结果

训练 700 episode 后的数据：

| 指标 | Ep 10 | Ep 700 |
|------|-------|--------|
| Avg Reward | -16.9 | -8.0 |
| Avg Length | 9 | 13 |
| Avg Break | 0 | 0.1 |
| Avg Fouls | 5.0 | 5.2 |
| P1 Score | 14.8 | 14.2 |
| P2 Score | 17.8 | 15.5 |

Reward 在上升，但得分主要来自对方犯规。球还没学会进。

---

## Round 5：击球方向诊断与修复

**目标：** 调查"瞄准了目标球但球飞向别处"的问题。

### 诊断过程

1. 创建 `debug_shots.py`，逐杆打印 `base_angle`、`offset_rad`、`shot_angle`、`vx`、`vy`
2. 验证结果：**角度计算本身无 bug**，velocity 与 shot_angle 完全一致

3. 统计 ep700 模型的 angle_offset 分布：
   ```
   offset std = 16.4°
   |offset| > 15°: 48%
   |offset| > 24°: 26%
   |offset| < 6°:  仅 20%
   ```

### 根因

- **±30° 的偏移范围太大** — 模型探索时频繁输出极端偏移
- **48% 的击球偏离目标超过 15°** → 视觉上看就像"打向别处"

### 修复

1. **角度范围缩小**：±30° → **±15°**
   - 同样的 offset 原始值，实际偏移减半
   
2. **精准度奖励**：
   ```
   accuracy_bonus = 0.3 × (1 - |offset| / max_offset)
   ```
   鼓励模型学会小偏移瞄准

---

## Round 6：选球 × 角度 × 结果 联合奖励

**目标：** 建立选球和角度之间的 reward 关联。

### 分析

旧 reward 中选球、角度、接触是独立评价的。但实际上"选了某个球，角度对不对准它，有没有打到它"是一条因果链，应该联合评价。

### 新奖励体系

```
选错球 → -1.5
选对球 → +0.2 (Layer 0)
       → +0.4 × accuracy (Layer 1: 角度)
       → 碰到选定球？
           是 → +0.5 + 0.3 × accuracy (Layer 2: 击中)
           碰别的 → +0.05
           全空 → -1.0
       → 首触非选定球？ -0.3 (Layer 3)
```

这样模型能得到的最大选球+角度 reward：
- 选对 + 完美瞄准 + 打到 = 0.2 + 0.4 + 0.5 + 0.3 = **+1.4**
- 选对 + 最大偏移 + 没打到 = 0.2 + 0 - 1.0 = **-0.8**
- 选错 = **-1.5**

形成清晰的梯度信号，引导模型逐步改善。

---

## Round 7：可视化增强

**目标：** 让评估时能"像看比赛一样"观看对局。

### 改进内容

1. **球运动实时渲染**
   - `_simulate_until_stable()` 在 `render_mode='human'` 时逐帧绘制
   - 球碰撞、反弹、入袋全过程可见

2. **击球前瞄准显示**
   - 黄色线：实际击球方向
   - 白色线：白球→目标球连线
   - 目标球高亮圈：绿色=合法选择 / 红色=非法选择
   - 暂停 400ms 让观众看清

3. **底部信息栏**（80px）
   - 力度条（绿/橙/红 三色渐变）
   - 目标球名称 + 合法性标签
   - 上一杆详情持续显示

4. **评估终端输出**
   - 每杆打印：玩家、reward、进球情况、阶段

---

## Round 8：Pooltool 迁移 + SAC 算法 + 奖励简化

**目标：** 迁移到真实物理引擎，新增 SAC 算法，简化奖励系统。

### 核心改动

1. **物理引擎：pymunk → pooltool**
   - 真实台球物理（event-driven collision，12ft 台面）
   - 完整旋转物理（高杆/低杆/左塞/右塞）
   - `environment/pooltool_env.py` 全新实现

2. **动作空间：5 维 → 7 维 → 8 维**
   - 新增 `b_spin`（高低杆）、`a_spin`（侧旋）
   - 新增 `pocket_idx`（目标袋口选择，8 维）
   - 角度偏移范围保持 ±15°

3. **SAC 算法新增**（`algorithms/sac.py`）
   - Twin Q-Networks + 自动温度 α + Replay Buffer
   - 与 PPO 统一接口（`select_action`, `update`, `save/load`）

4. **奖励系统：9 层 → 7 参数统一距离函数**
   - `reward = max_r × (1 - d/d_max)^1.5`
   - d = 目标球到选定袋口的距离
   - 犯规惩罚 4 种：非法选球/白球进袋/碰非法球/空杆

5. **训练产出**
   - 新增 `metrics_final.json`（全部 episode 级指标）
   - 新增 `--analyse` 模式查看训练摘要
   - 训练输出含 ETA、reward 分解、进球数

### 实验 8a：SAC 2000 局（旧参数）

```
参数: buffer=500K, warmup=256, max_shots=10, max_fouls=3
时间: ~30 min (CPU)
```

| 指标 | 结果 |
|------|------|
| 总步数 | ~24,630 |
| 每局步数 | ~12 |
| Reward | ~ -3.3 |
| Pots | ~0.3 |
| Fouls | ~4.8 |
| Buffer 利用率 | 4.9% (24K/500K) |

**问题：** 每局太短（3 次连续犯规就终止），总样本不够。

### 实验 8b：SAC 2000 局（放宽终止条件）

```
参数: buffer=100K, warmup=1000, max_shots=30, max_fouls=5
时间: ~63 min (CPU)
```

| 指标 | 前 10% | 后 10% | 变化 |
|------|--------|--------|------|
| Reward | -14.07 | -12.25 | ↑ 1.82 |
| Foul reward | -16.36 | -15.43 | ↑ 0.94 |
| Distance reward | 2.29 | 3.17 | ↑ 0.88 |
| Pots | 1.56 | 0.79 | **↓ 0.78** |
| Fouls/ep | 18.1 | 20.9 | **↑ 2.7** |

| 总体 | 值 |
|------|------|
| 总步数 | 78,001 |
| 每局步数 | ~39（比 8a 的 12 提升 3×） |
| 犯规率 | ~52%（稳定不变） |
| Policy Loss | -1.5 → -75（单调下降，发散） |
| Value Loss | 0.5 → 67.8（爆炸） |

**问题诊断：**
- Policy Loss 单调负 = Q 值过估计，actor 追着虚假高 Q 值走
- Value Loss 爆炸 = critic 没收敛
- 犯规率始终 52% = agent 没学会选合法球
- Pots 反而下降 = 训练可能在让 agent 变差

---

## Round 9：SAC 训练稳定性修复

**目标：** 解决 Q 值过估计 + Value Loss 爆炸问题。

### 改动

1. **降低学习率**：actor/critic 3e-4 → 1e-4，alpha lr 3e-4 → 1e-4
2. **更慢的 target 更新**：tau 0.005 → 0.002
3. **reward 归一化**：对 reward 做 running normalization（稳定 Q 值尺度）
4. **延迟 actor 更新**：critic 先学 2 步再更新 actor（避免 actor 追未收敛的 Q）
5. **增大 warmup**：1000 → 2000
6. **增大 episode 数**：2000 → 5000
7. **实验目录隔离**：每次训练自动保存到 `experiments/{algo}_{timestamp}/`，支持 `--run_name` 自定义
8. **图表升级**：3×4 = 12 格布局，新增 Critic Loss、Policy vs Critic 对比、Foul Rate、Summary 文本框
9. **窗口自适应**：pygame 渲染窗口自动适配屏幕尺寸，支持拖拽缩放

### 实验 9a：SAC 5000 局（稳定性修复后）

```
参数: lr=1e-4, tau=0.002, warmup=2000, actor_update_interval=2,
      normalize_rewards=True, buffer=100K, batch=256
实验目录: experiments/round9_critic_stability/
时间: CPU
```

**分段趋势（每段 1000 局）：**

| 段 | Reward | Foul R | Dist R | Pots | Fouls | Len | Break |
|----|--------|--------|--------|------|-------|-----|-------|
| 1 | -12.46 | -15.40 | 2.94 | 1.28 | 18.7 | 38.0 | 0.67 |
| 2 | -8.42 | -13.77 | 5.35 | 0.88 | 19.8 | 45.5 | 0.54 |
| 3 | -4.04 | -12.01 | 7.97 | 0.68 | 17.8 | 49.3 | 0.41 |
| 4 | -1.65 | -11.81 | 10.16 | 0.94 | 18.3 | 53.2 | 0.54 |
| 5 | -1.64 | -12.21 | 10.57 | 0.96 | 19.3 | 54.3 | 0.56 |

**Loss 分段趋势：**

| 段 | Policy Loss | Critic Loss |
|----|-------------|-------------|
| 1 | -22.97 | 3.30 |
| 2 | -39.84 | 7.19 |
| 3 | -34.96 | 6.66 |
| 4 | -25.89 | 4.76 |
| 5 | -22.07 | 4.50 |

**犯规率分段：**

| 段 | Foul Rate (fouls/steps) |
|----|------------------------|
| 1 | 0.519 |
| 2 | 0.440 |
| 3 | 0.347 |
| 4 | 0.318 |
| 5 | 0.338 |

**进球分布：**

| Pots/Episode | 比例 |
|---|---|
| 0 | 48.6% |
| 1 | 28.0% |
| 2 | 12.9% |
| 3+ | 10.5% |
| Max | 12 |

**Break 分布：**

| Break | 比例 |
|---|---|
| 0 | 56.7% |
| 1 | 40.1% |
| 2+ | 3.2% |
| Max | 9 |

**比分统计：**
- 进球总分: 3,786 vs 罚分总分: 425,664 → 罚分占比 99.1%
- P1 总得分(pot): 1,928 / P2 总得分(pot): 1,858

**对比 8b：**

| 指标 | 8b (旧) | 9a (新) | 变化 |
|------|---------|---------|------|
| Policy Loss 最终 | -75 (发散) | -21 (收敛) | ✅ 修复 |
| Value Loss 最终 | 67.8 (爆炸) | 4.0 (稳定) | ✅ 修复 |
| 犯规率 | 52% 不变 | 52%→34% | ✅ 有改善 |
| Distance Reward | 2.3→3.2 | 2.9→10.6 | ✅ 大幅改善 |
| Reward | -14→-12 | -12→-2 | ✅ 大幅改善 |

**结论：** SAC 稳定性问题完全修复。Loss 收敛、犯规率下降、distance reward 大幅提升。
但 agent 学会了一个 **reward hack**，没有真正学会进球。

### 诊断：Reward Hack（距离奖励漏洞）

通过 `diagnose_agent.py` 观察 agent 实际行为，发现了严重的奖励漏洞：

**Agent 行为模式：**
```
#2  tgt=red_13  V0=0.6  off=+0.01  r=+0.866  potted=[-]   ← 轻触，拿 0.87
#3  tgt=red_13  V0=0.6  off=-0.01  r=+0.842  potted=[-]   ← 同一颗球，0.84
#4  tgt=red_13  V0=0.6  off=-0.02  r=+0.681  potted=[-]   ← 继续轻触，0.68
#5  tgt=red_13  V0=0.6  off=-0.01  r=+0.794  potted=[-]   ← 永远不进球
```

- **一直打同一颗球**（red_13 连续 8 杆，red_10 连续 31 杆）
- **V0 全是 0.6**（最小力量），球几乎不动
- **从不进球**，但每步拿到 0.5-0.9 的 distance reward
- 进球 3 局 94 步只有 1 次，所有得分几乎全来自对方犯规

**根因分析：**

距离奖励函数 `reward = max_r × (1 - d/d_max)^1.5` 存在根本性缺陷：

| 目标球离袋口 | Reward | 占进球奖励比例 |
|---|---|---|
| 0m（进袋） | 1.000 | 100% |
| 0.5m | 0.816 | **82%** |
| 1.0m | 0.645 | **65%** |
| 2.0m | 0.347 | **35%** |

**问题本质：**

1. **奖励基于绝对位置而非改进量** — 球本来就在台面某处，离最近袋口可能只有 1-2m，轻触一下就能拿到 0.3-0.9 分，不需要任何有效操作
2. **进球奖励太低** — 进球 1.0 vs 轻触 0.5-0.9，进球只多 10-50%，但需要精确瞄准，风险远大于收益
3. **pocket_idx 选择无实质约束** — agent 可以随便选一个离球近的袋口来"刷分"

**修复方向（Round 10）：**

核心思路：**用距离改进量（Δd）替代绝对距离**

```
d_before = 击球前，目标球到选定袋口的距离
d_after  = 击球后，目标球到选定袋口的距离
Δd = d_before - d_after

if 进袋:     reward = pot_bonus（大幅提高，如 5.0）
elif Δd > 0: reward = max_r × (Δd / d_before)^0.5  （推近了才有奖励）
else:        reward = 0（推远了不给分）
```

关键改动：
- 轻触（球几乎没动）→ Δd ≈ 0 → reward ≈ 0 ← 堵住漏洞
- 把球推近袋口 → Δd > 0 → 有奖励
- 进球 → pot_bonus = 5.0 >> 推近的 ~0.5 ← 进球值得冒险

---

## Round 10：Δd 距离改进奖励

**目标：** 修复 reward hack — 用 Δd 替代绝对距离，堵住"轻触刷分"漏洞。

### 改动

1. **距离奖励改为改进量**：`reward = approach_max × (Δd / d_before)^0.5`（仅 Δd > 0 时给奖）
2. **进球奖励提升**：红球 1.0 → 5.0，彩球 = 分值 × 1.0
3. **犯规惩罚不变**：illegal -1.0, white_pocket -1.5, wrong_ball -0.8, miss_target -0.3

### 实验 10a：SAC 4000 局

```
参数: 同 Round 9 SAC params + 新 reward
实验目录: experiments/round10_delta_d/
```

**分段趋势（每段 800 局）：**

| 段 | Reward | Foul R | Dist R | Pots | Fouls | Len | Break |
|----|--------|--------|--------|------|-------|-----|-------|
| 1 | -14.71 | -16.94 | 2.24 | 1.56 | 19.7 | 38.6 | 0.68 |
| 2 | -12.20 | -15.14 | 2.94 | 1.23 | 21.0 | 43.1 | 0.60 |
| 3 | -12.14 | -15.64 | 3.50 | 1.19 | 24.0 | 49.8 | 0.60 |
| 4 | -11.16 | -15.15 | 4.00 | 1.15 | 23.9 | 52.1 | 0.59 |
| 5 | -10.80 | -15.01 | 4.21 | 1.24 | 23.8 | 52.6 | 0.57 |

**关键指标：**
- 犯规率：0.538 → 0.451（16% 改善）
- |Foul| / Distance ratio: 4.6x
- 罚分/总分比: 99.1%（未改善）
- 进球数：1.56 → 1.24（**下降 20%**）
- Distance reward: 2.24 → 4.21（**上升 88%**）

**诊断：新 Reward Hack**

```
V0=0.7 off=-0.01  d=0.31   ← 低力度轻碰
V0=0.6 off=-0.25  d=0.66   ← 低力度轻碰
V0=0.9 off=+0.18  d=0.37   ← 低力度轻碰
```

- Δd 修复了"轻触就高分"，但 agent 学到了新策略：**用小力度反复蹭球推近，拿安全的 0.3-0.7 距离奖励**
- 进球需要大力精准击打（风险高），而蹭近安全无风险
- 一局累计 dist reward = 4.2，一次进球 = 5.0，性价比不够高
- **Pots 训练越久越低** — agent 理性选择"低风险安全蹭近"策略

**结论：** Δd 修复了旧漏洞但产生了新问题。正负 reward 结构性失衡——惩罚太密集（45%步犯规），正向太稀疏（进球偶发）。

---

## Round 11：乘法链奖励 + 进球规则修复

**目标：** 简化 reward 信号 + 修复非目标球进袋不得奖的 bug。

### Round 11a：乘法链设计

核心思路：**乘法链条，每环做对才有正向信号。**

```
reward = 选对球(gate1) × 打中球(gate2) × 结果(outcome)

Gate1 失败（选球错误/碰到非法球/空杆）→ -1
Gate2 失败（没打中目标球）              → -1
白球进袋                                → -2
两个 Gate 都过 → outcome:
  进袋 → +10（红）/ 分值×1.5（彩）
  推近 → approach_max × (Δd/d_before)^0.5（最高1.0）
  没动/推远 → 0
```

**关键改进：**
1. 统一犯规 = -1（不再区分 -0.3/-0.8/-1.0/-1.5）
2. 进球奖励 = 10（远超距离最高 1.0）
3. 白球进袋 = -2（唯一最重惩罚）
4. 推远/没动 = 0（不罚不奖，鼓励尝试）

### 实验 11a：SAC 2500 局

```
参数: 同 Round 9 SAC params + Round 11 乘法链 reward
实验目录: experiments/round11_mult_chain/
```

**分段趋势（每段 500 局）：**

| 段 | Reward | Foul R | Dist R | Pots | Fouls | Len | Break |
|----|--------|--------|--------|------|-------|-----|-------|
| 1 | -28.98 | -32.29 | 3.31 | 2.06 | 18.9 | 36.7 | 0.35 |
| 2 | -28.06 | -31.51 | 3.44 | 1.76 | 18.5 | 36.4 | 0.34 |
| 3 | -27.59 | -31.62 | 4.03 | 1.78 | 18.8 | 37.6 | 0.38 |
| 4 | -26.23 | -30.51 | 4.28 | 1.64 | 17.9 | 37.0 | 0.39 |
| 5 | -27.49 | -31.73 | 4.24 | 1.56 | 18.8 | 39.3 | 0.38 |

**Pots 分布：**

| Pots/Episode | 比例 |
|---|---|
| 0 | 20.6% |
| 1 | 30.7% |
| 2 | 22.5% |
| 3+ | 26.1% |
| Max | 12 |

**犯规率：** 0.537 → 0.496（轻微改善）

**Summary:** Reward -29.38 → -28.02 (UP 1.36), Fouls 19.1 → 19.0, Pots 2.23 → 1.59 (DOWN 0.64)

### 诊断：非目标球进袋 Bug

通过 `diagnose_agent.py` 发现**大量进球被错误惩罚**：

```
# 24  tgt=red_15  OK  potted=[red_08]  r=-1.000  ← 红球进了但给 -1！
# 42  tgt=red_03  OK  potted=[red_15]  r=-1.000  ← 红球进了但给 -1！
```

**根因：** 旧代码要求 `chosen_target in pocketed` 才给进球奖励。但斯诺克规则中：
- Red phase：碰到**任何红球**都合法，**任何红球**进袋都得分
- 连环碰击导致的"非目标红球进袋"完全合法，应该给奖励
- 旧逻辑把这些全判为 gate2_fail → -1

这就是 Pots 始终无法提升的根本原因——**大量合法进球被惩罚，agent 学会了避免进球**。

### Round 11b：进球判定修复

**修复内容：**

1. **`_count_legal_pots(pocketed)`**：按 phase 统计合法进袋数
   - Red phase: 所有红球进 = 合法
   - Color phase: 所有彩球进 = 合法
   - Final colors: 指定彩球进 = 合法

2. **`_handle_pot_scoring(pocketed, cp)`**：统一得分和阶段转换逻辑

3. **Gate2 fail 但有合法球进袋 → 给半价奖励 (+5)**
   - 没精确打中目标球，但有红球被撞进了 → 仍然奖励（鼓励散球进球）

4. **Success path 奖励所有合法进球**
   - 打中目标球 + N颗红进 → +10×N（不只奖目标球）

5. **First contact 合法性修正**
   - Red phase: 碰到任何红球 = 合法（不要求碰到 chosen_target）
   - Color phase: 碰到任何彩球 = 合法

**修复后的 reward 数值表：**

| 场景 | Reward |
|---|---|
| 打中目标球 + 任何红/彩球进 | +10 per pot |
| 合法碰球 + 非目标球进（散球进球）| +5 per pot（半价）|
| 打中目标球 + 推近 | 0~1.0 |
| 打中目标球 + 推远/没动 | 0 |
| 合法碰但没打中目标，也没进球 | -1 |
| 犯规（碰到彩球/没碰到球/选错球）| -1 |
| 白球进袋 | -2 |

---

## Round 12：Break Building（极简 Reward）

**目标：** 彻底简化 reward 信号，鼓励高手打法（红→彩→红连续得分），杜绝一切 reward hack。

### 问题分析（Round 11 遗留）

1. **"大力出奇迹"不该奖** — 一杆撞进 3 颗红球给 30 分，鼓励了非技巧性打法
2. **Gate2（精确打中目标球）概念多余** — 斯诺克规则中只要碰到合法类型的球就行，不需要精确打中 chosen_target
3. **半价奖励（散球进球 +5）设计不好** — 同样鼓励了大力散球策略
4. **Δd 距离奖励是 reward hack 的温床** — 每一种距离奖励最终都被 agent 找到漏洞利用
5. **真正的高手打法是连续得分** — 红→彩→红→彩，每一杆都有目的

### 设计方案

**极简三值 reward + break bonus：**

```
合法 + 进球 → pot_reward(10) + break_bonus(2) × current_break
合法 + 没进 → 0（没进就是给对手机会，不该奖）
犯规         → -1
白球进袋     → -2
```

**核心改动：**

1. **去掉 Gate2** — 不区分"打中目标球"和"碰到同类型其他球"，只要 first contact 合法即可
2. **进球奖励固定** — 一杆不管进几颗，奖励固定 10（不鼓励大力散球）
3. **彻底去掉 Δd 距离奖励** — 没进就是 0，信号极简，无 hack 空间
4. **新增 Break Continuation Bonus** — `+2 × current_break`，连续得分越长奖励越高
5. **犯规的隐性惩罚** — 不只是 -1，还 break 归零，丢掉累积的 bonus 系数

**Break Bonus 奖励曲线：**

| 进球时 break | Reward | 含义 |
|---|---|---|
| 0 | 10 + 0 = 10 | 开局第一颗 |
| 1 | 10 + 2 = 12 | 红接彩（连续了）|
| 4 | 10 + 8 = 18 | 两组红彩后再进 |
| 8 | 10 + 16 = 26 | 长 break |
| 20 | 10 + 40 = 50 | 超级回报 |

**设计哲学：**
- 大力散球进 3 颗 → 只给 10（和进 1 颗一样），且后续局面差，难以续 break
- 精准打进 1 颗 + 白球走位好 + 续 break → 10 + 12 + 14 + 16... 累积越来越多
- 犯规 → break 归零，丢掉所有累积——agent 会学到"维持 break 比冒险更重要"

### 代码改动

- `RewardConfig`: 只剩 4 个参数（`pot_reward`, `break_bonus`, `foul_penalty`, `white_pocket_penalty`）
- `_compute_reward()`: 简化为 foul/legal 二分，去掉 gate2、去掉 Δd
- 去掉 `_d_before` 相关代码（不再需要距离计算）
- 保留 `_count_legal_pots()` 和 `_handle_pot_scoring()`（Round 11b 的进球规则修复）

### 观测特征（已有 phase 信息）

obs 第 34 维 = `phase_val`：red=0.0, color=0.5, final_colors=1.0。
Agent 可以据此区分当前应该打红球还是彩球。
如果后续实验发现 agent 在 color phase 仍选红球，再考虑改为 one-hot。

---

## Round 13：终局胜负奖励（Win/Loss Terminal Reward）

**目标：** 给 agent 明确的长期目标——赢得比赛，解决 self-play 中缺乏博弈动力的根本问题。

### 问题分析（Round 12 实验 1300 局）

**训练趋势全面恶化：**

| 指标 | 前10% → 后10% | 趋势 |
|------|-------------|------|
| Reward | -9.99 → -11.95 | ↓ 恶化 1.96 |
| Fouls | 14.4 → 16.2 | ↑ 恶化 1.9 |
| Pots | 1.68 → 1.28 | ↓ 恶化 0.40 |

**五段趋势：**

| 段 | Reward | FoulR | DistR(进球) | Pots | Fouls | Break |
|---|--------|-------|------------|------|-------|-------|
| 1 | -9.30 | -16.85 | 7.55 | 1.80 | 14.6 | 0.87 |
| 2 | -9.27 | -17.99 | 8.72 | 1.79 | 15.8 | 1.05 |
| 3 | -9.45 | -17.50 | 8.05 | 1.75 | 15.4 | 0.92 |
| 4 | -10.87 | -17.57 | 6.70 | 1.54 | 15.5 | 0.75 |
| 5 | -11.40 | -17.77 | 6.38 | 1.35 | 15.9 | 0.74 |

**核心问题诊断：**

1. **奖励结构严重不平衡** — 98.8% 的总分来自罚分，进球得分仅占 1.2%。Agent 看到"做什么都亏"。
2. **没有终局胜负奖励** — self-play 中最关键的信号缺失。Agent 不知道"赢比赛"是目标，只学到"每一杆别犯规"。
3. **Policy-Critic 退化** — Policy Loss -5→-11（actor 过度自信），Critic Loss 0.5→2.7（value 估计越来越差）。

### 设计方案

在 Round 12 的 per-step 奖励基础上，新增终局胜负奖励：

```
# Per-step（不变）
合法 + 进球 → pot_reward(10) + break_bonus(2) × current_break
合法 + 没进 → 0
犯规         → -1
白球进袋     → -2

# 终局（新增）
赢了（我的得分 > 对手得分）→ +30
输了（我的得分 < 对手得分）→ -30
平局                       → 0
```

**RewardConfig 新增 2 个参数：**
- `win_reward: float = 30.0`
- `lose_reward: float = -30.0`

**数值选择逻辑：**
- 每局约 30 步，per-step reward 总量约 -10~-15
- 胜负 ±30 与每局 per-step 总量同量级，确保信号不被淹没
- 不设太大（如 ±100），避免终局信号过度压倒 per-step 信号

**实现细节：**
- 在 `_compute_reward()` 的 terminal conditions 之后、sum 之前计算
- 用出手玩家 `cp`（而非 `self.current_player`，因为犯规 path 已经 switch 了）的得分对比
- `reward_breakdown` 新增 `win_loss` 字段

### 代码改动

- `RewardConfig`: 新增 `win_reward=30.0`, `lose_reward=-30.0`
- `_compute_reward()`: 在 `done=True` 时，比较 `scores[cp]` vs `scores[1-cp]`，加入 `bd['win_loss']`
- `config.py`: 更新 rewards section
- `test.py`: 更新 breakdown 测试（新增 `win_loss` key），更新 RewardConfig 构造测试

### Round 13 实验结果（5000 局）

```
参数: SAC, lr=1e-4, tau=0.002, warmup=2000, actor_update_interval=2,
      normalize_rewards=True, buffer=100K, batch=256
      新增: win_reward=30, lose_reward=-30
实验目录: experiments/round13_win_loss/
```

**分段趋势（每段 1000 局）：**

| 段 | Reward | FoulR | DistR | Pots | Fouls | Len | Break |
|----|--------|-------|-------|------|-------|-----|-------|
| 1 | -37.21 | -18.88 | -18.33 | 1.45 | 16.6 | 31.6 | 0.70 |
| 2 | -30.88 | -14.02 | -16.85 | 1.51 | 12.3 | 37.3 | 0.75 |
| 3 | -29.04 | -12.11 | -16.93 | 1.45 | 10.3 | 35.0 | 0.77 |
| 4 | -27.48 | -10.14 | -17.34 | 1.55 | 8.5 | 30.8 | 0.74 |
| 5 | -26.66 | -9.47 | -17.19 | 1.52 | 8.3 | 30.1 | 0.70 |

**关键进步：**
- **Reward**: -37.21 → -26.66（**↑ 10.55**，有显著学习）
- **犯规率**: 0.575 → 0.355（**↓ 38%**，最大改善指标）
- **Fouls**: 16.6 → 8.3（**↓ 50%**，大幅降低）
- **Pots**: 1.45 → 1.52（**持平**，没有提升）

**胜负统计：**
- P1 胜率: 46.9%，P2 胜率: 51.4%，平局: 1.7%
- 得分来源：98.8% 来自对手犯规罚分，进球得分仅 1.2%

**诊断：安全低速互推策略（Safe Harbor Hack）**

通过 `diagnose_agent.py` 观察 ep5000 模型，发现了关键的策略漏洞：

```
#1  tgt=red_10  V0=0.5  off=+0.02  r=0.000  potted=[-]   ← 最小力度，合法碰球
#2  tgt=red_10  V0=0.5  off=-0.01  r=0.000  potted=[-]   ← 同样最小力度
#3  tgt=red_10  V0=0.5  off=+0.03  r=0.000  potted=[-]   ← 永远不尝试进球
...（双方交替，每步 reward = 0）
```

**Agent 学会了一个新 hack：**
1. **V0 = 0.5**（最小力度），合法碰到红球 → reward = 0（安全港）
2. **从不尝试进球** — 因为进球需要大力精准击打（风险高）
3. **双方轮流轻碰** → 直到 `max_shots_without_pocket=30` 终止比赛
4. **得分完全来自对手犯规**（早期探索阶段的随机犯规积累）

**根因分析：**

Agent 做了理性的风险-收益计算：
- 尝试进球：pot_reward=10，但可能犯规（-1 + 可能输 -30 = -31）
- 安全轻碰：reward = 0，零风险
- **结论：pot_reward=10 不值得冒输掉 -30 的风险**

`miss_penalty = 0` 创造了一个"安全港"——合法碰球但不进，reward 恰好为 0，agent 发现这是最优策略。

---

## Round 14：Behavior + Outcome 双轨制奖励

**目标：** 打破"安全低速互推"的 hack，同时建立行为奖励和终局奖励的双轨架构。

### 问题分析（Round 13 遗留）

1. **安全港问题**：合法碰球没进 → reward = 0，零风险零代价
2. **进球期望不足**：pot_reward=10 不够大，不值得冒犯规+输球的风险
3. **无意义对推太久**：max_shots_without_pocket=30，双方可以互推 30 杆

### 设计方案

**三管齐下同时修改，打破安全港：**

#### 改动 1：pot_reward 10 → 20（翻倍进球奖励）

提高进球的期望回报，让冒险变得值得：
- 旧：进球 10 vs 安全港 0，差距太小
- 新：进球 20 vs 安全港 -0.1×N，差距显著

#### 改动 2：miss_penalty = -0.1（消除安全港）

每次合法碰球但没进，给一个小负值 -0.1：
- 20 步没进 → 累积 -2.0（等于一次白球进袋的代价）
- **消除了"零代价轻碰"的安全港**
- 值很小，不构成严重惩罚，但让"无限轻碰"有了代价

#### 改动 3：max_shots_without_pocket 30 → 20（缩短无意义对推）

将不进球的最大回合数从 30 缩减到 20：
- 减少无意义对推的总步数
- 加快终局到来，让 win/loss 信号更快传递

### RewardConfig 双轨结构

将奖励参数重组为两大块：

```python
@dataclass
class RewardConfig:
    # ══ Behavior Reward（过程）══
    pot_reward: float = 20.0           # 进球（10→20）
    break_bonus: float = 2.0           # break 连续奖励
    foul_penalty: float = -1.0         # 犯规
    white_pocket_penalty: float = -2.0 # 白球进袋
    miss_penalty: float = -0.1         # 合法没进（新增，消除安全港）

    # ══ Outcome Reward（终局）══
    win_reward: float = 30.0           # 赢了
    lose_reward: float = -30.0         # 输了
```

**数值设计逻辑：**

| 场景 | Reward | 说明 |
|------|--------|------|
| 进球（break=0）| +20 | 远超安全港代价 |
| 进球（break=8）| +36 | 连续进球回报递增 |
| 合法没进 | -0.1 | 小代价，消除零风险 |
| 犯规 | -1 | 明确惩罚 |
| 白球进袋 | -2 | 最重 per-step 惩罚 |
| 赢了 | +30 | 长期目标 |
| 输了 | -30 | 长期目标 |

**期望值分析：**
- 安全港策略：20 步 × (-0.1) = -2.0（不再是零代价）
- 进球尝试（假设 30% 成功率）：0.3 × 20 - 0.7 × 0.1 = +5.93（显著正期望）
- **进球期望远超安全港 → agent 应该学会冒险进球**

### 代码改动

- `environment/pooltool_env.py`:
  - `RewardConfig` 重组为 Behavior + Outcome 双块
  - 新增 `miss_penalty: float = -0.1`
  - `pot_reward`: 10.0 → 20.0
  - 合法没进路径：`bd['distance'] = 0.0` → `bd['distance'] = rc.miss_penalty`
  - `max_shots_without_pocket` 默认值：30 → 20
- `config.py`: 更新 rewards 为双轨结构，新增 miss_penalty
- `test.py`: RewardConfig 测试新增 miss_penalty 参数
- `docs/changelog.md`: 添加 Round 13 实验分析 + Round 14 设计

### Round 14 实验结果（4240 局，pooltool crash 中断）

```
参数: SAC, lr=1e-4, tau=0.002, warmup=2000, actor_update_interval=2,
      pot_reward=20, miss_penalty=-0.1, max_shots_without_pocket=20
实验目录: experiments/round14_dual_track/
```

**分段趋势（每段 848 局）：**

| 段 | Reward | FoulR | Pot/Miss | WL | Pots | Fouls | Len | FoulRate | WinRate |
|----|--------|-------|----------|-----|------|-------|-----|----------|---------|
| 1 | -28.68 | -16.90 | 10.68 | -22.5 | 1.38 | 15.1 | 26.7 | 0.588 | 11.8% |
| 2 | -28.09 | -15.96 | 10.36 | -22.5 | 1.23 | 14.2 | 25.9 | 0.572 | 11.3% |
| 3 | -28.10 | -15.83 | 11.12 | -23.4 | 1.23 | 14.1 | 25.1 | 0.586 | 10.5% |
| 4 | -27.56 | -16.02 | 11.21 | -22.8 | 1.27 | 14.3 | 25.8 | 0.577 | 11.2% |
| 5 | -26.82 | -16.56 | 11.17 | -21.4 | 1.22 | 14.9 | 26.8 | 0.577 | 13.4% |

**对比 Round 13：**

| 指标 | Round 13 (5000ep) | Round 14 (4240ep) | 结论 |
|------|-------------------|-------------------|------|
| 犯规率 | 0.58→0.35 (↓38%) | 0.58 (无变化) | ❌ 差 |
| Pots/ep | 1.45→1.52 | 1.54→1.17 (↓) | ❌ 差 |
| 胜率 | ~47% (P1) | 11.7% | ❌ 差 |
| Reward | -37→-27 (↑10.5) | -28.5→-27.3 (↑1.3) | ❌ 改善微弱 |

**失败原因分析：**

1. **miss_penalty(-0.1) 效果微乎其微** — 累积 20 步仅 -2.0，远不如 win/loss ±30 显著
2. **pot_reward 翻倍没有帮助** — 因为 agent 58% 步都犯规，进球概率极低
3. **犯规率完全没改善** — 核心原因：agent 不会选合法球

**关键发现：agent 58% 的犯规率是结构性的，无法通过调整 reward 解决。**

犯规的 4 个来源：
- `not chose_legal` — 选了非法目标球（~30%，完全可控但 agent 学不会）
- `not white_hit_any` — 空杆
- `wrong_first_contact` — 碰到非法球
- `white_pocketed` — 白球进袋

其中 `chose_legal` 问题的根因：`target_idx` 从所有 21 颗球中盲选，合法球比例动态变化（red phase 71% vs color phase 29%），且球进了后 mapping 会变。这是规则约束，不应该让 agent 学习。

**pooltool crash bug（已修复）：**

训练在 ep4240 时因 `pooltool` 内部 `assert v_n_0 < 0` 崩溃。
原因：球贴着 cushion 时物理碰撞求解器的罕见断言失败。
修复：在 `pt.simulate()` 外加 try/except，异常时视为犯规 + 重建系统。

---

## Round 15：Action Masking（强制合法选球）

**目标：** 将"选球合法性"从 reward signal 移到 action space 约束中，消除不可学习的犯规来源。

### 问题分析

Agent 在 target selection 上面临一个几乎不可能学会的任务：
- action `target_idx` ∈ [-1, 1] 映射到 `all_targetable`（15红+6彩=21 颗）
- Red phase: 合法=红球（71%），Color phase: 合法=彩球（29%）
- 球进了后列表动态缩短，相同 action 值映射到不同的球
- Agent 需要同时学会：规则+策略+物理——信息过载

**类比：** 就像让 AlphaGo 学习"不能下在有棋子的位置"一样荒谬——这是规则前提，不是策略。

### 设计方案

**Action Masking: `target_idx` 只从合法目标球中选择**

```python
# 旧：从所有球中盲选
all_targets = self._get_all_targetable()  # 21颗混排
idx = int((target_raw + 1) / 2 * len(all_targets))
chosen_target = all_targets[idx]
chose_legal = chosen_target in legal_targets  # 可能 False!

# 新：只从合法目标中选
legal_targets = self._get_legal_targets()  # red phase→只有红球
idx = int((target_raw + 1) / 2 * len(legal_targets))
chosen_target = legal_targets[idx]
chose_legal = True  # 永远合法
```

**效果：**
- `not chose_legal` 犯规来源被彻底消除
- Agent 的 `target_idx` 从"猜哪个合法"变成"在合法球中选哪个最好打"
- 犯规判定仍有 3 个条件生效：空杆/碰错球/白球进袋（这些是 agent 需要通过技能避免的）

**Obs 稳定性确认：**
- 观测空间是硬编码的 54 维（white + red_01~red_15 + 6 colours + game state）
- 球进了后对应位置变为 [-1, -1]，不会移动其他球的特征位置
- Agent 可以通过坐标信息判断"选哪颗球离白球最近/最好打"

### 代码改动

- `environment/pooltool_env.py`:
  - `step()` 中 target selection 改为从 `legal_targets` 映射
  - `chose_legal = True`（不再需要判断）
  - 犯规判定中 `not chose_legal` 条件仍保留但永远不触发

### Round 15 实验结果（3000 Episodes）

**核心指标变化（vs Round 14）：**
- Pots: 1.17 → 2.11/ep（↑67%）— Action masking 让 agent 不再浪费步数选非法球
- Foul Rate: 0.58 → 0.50（↓8%）— 消除了 `illegal_choice` 犯规来源
- Win Rate: 11.7% → 18.7%（↑60%）
- Break≥2: 5.5% → 20.4%（↑3.7×）

**问题：** Foul rate 卡在 50%，5 个分段内无下降趋势。剩余 3 种犯规来源（空杆/碰错球/白球进袋）无法区分哪个是主因。

---

## Round 15.5：犯规类型细分追踪

**目标：** 追踪每种犯规的具体数量，通过图表和日志可视化，为下一步优化提供数据支撑。

### 问题

Foul rate 卡在 50% 但无法诊断原因：
- `miss_ball`（空杆）：白球没碰到任何球
- `wrong_ball`（碰错球）：先碰到了非法球（如 red phase 先碰彩球）
- `white_pocket`（白球进袋）：白球被打进袋
- `illegal_choice`（选非法球）：action masking 后已消除

不知道哪种占主导，"只能在猜"。

### 实现方案

**1. 环境侧 (`environment/pooltool_env.py`)**
- 在 `_compute_reward()` 中，犯规后判断具体类型：
  - 优先级: `miss_ball` > `wrong_ball` > `illegal_choice` > `white_pocket`
  - 添加 `bd['foul_type']` 到 reward_breakdown dict
- physics crash 路径也标记为 `foul_type='physics_crash'`

**2. 训练侧 (`train.py`)**
- `TrainingMetrics` 增加 4 个 per-episode 列表：
  - `miss_ball_counts`, `wrong_ball_counts`, `white_pocket_counts`, `illegal_choice_counts`
- `add()`, `save()`, `load()`, `get_average()`, `summary()` 全部适配
- `plot()`: (1,2) 图表从单线 Fouls 改为三线犯规类型细分（空杆/碰错/白袋 各一条 MA 线）
- 终端日志: `Fouls: 5.0 (空杆:2.0 碰错:2.5 白袋:0.5)`
- Summary 文本框: 显示犯规类型前后 10% 趋势对比

### 验证

50 步随机策略结果：
- 合法(None): 29, 空杆(miss_ball): 11, 碰错球(wrong_ball): 9, 白球进袋(white_pocket): 1
- illegal_choice: 0（action masking 有效）
- 全部测试通过 ✓

---

## Round 16：犯规诊断实验（round16_foul_diag）

**目标：** 利用 Round 15.5 的犯规类型追踪系统，诊断 foul rate 50% 的根因。

### 实验结果（1800 Episodes，从头训练）

**犯规类型分布：**
| 犯规类型 | 占比 | 趋势 |
|---------|------|------|
| wrong_ball（碰错球）| 54.2% | 11.6→8.8 **↓改善** |
| miss_ball（空杆）| 46.4% | 7.2→9.2 **↑恶化** |
| white_pocket（白球进袋）| 3.5% | 稳定 |

**关键发现：碰错球在改善，但空杆在恶化**
- 5 段趋势中，wrong_ball 从 11.3→8.4（↓26%）
- 同期 miss_ball 从 7.2→9.1（↑26%），在 ep800 后反超成为主要犯规类型
- 整体 reward 从 -14.4→-22.1（恶化），pots 从 2.7→1.8（退步）

**根因分析：**
- 空杆惩罚 (-1) 与碰错球惩罚 (-1) 相同，agent 无法区分"打不到球"和"打错球"的严重性差异
- 角度偏移范围 ±15° 过大，随机探索时很容易完全打偏
- Agent 学到"回避策略"：不碰球和碰错球代价一样，但碰球还有白球进袋的额外风险

---

## Round 17：空杆惩罚差异化 + 角度收窄 + 精度追踪

**目标：** 针对 Round 16 诊断出的空杆问题，做三项改进。

### 改进方案

**1. 空杆独立惩罚（RewardConfig）**
```
miss_ball_penalty: -3.0  （空杆，最重 — 打不到球是最基本的问题）
foul_penalty:      -1.0  （碰错球）
white_pocket_penalty: -2.0（白球进袋）
```
层级：空杆(-3) > 白球进袋(-2) > 碰错球(-1)，让 agent 明确知道"打到球"是第一优先级。

**2. 角度偏移范围缩小**
```
offset_deg = angle_offset * 8.0  # 从 ±15° 缩到 ±8°
```
减少 agent "打飞"的物理可能性，同时保留足够的策略空间。

**3. 角度精度追踪（新指标）**
- 环境：`offset_deg` 暴露到 `info` 中
- 训练：每 episode 记录 `mean(|offset_deg|)`（平均绝对偏移角）
- 图表：(0,1) 位置从 Episode Length 换为 Angle Offset 精度图
- 日志：`Ang: 4.2°`
- Summary：角度趋势对比（如 `4.5° → 3.2°`，越小越精准）

### 代码改动

- `environment/pooltool_env.py`:
  - `RewardConfig` 增加 `miss_ball_penalty = -3.0`
  - `_compute_reward()` FOUL PATH 中空杆使用独立惩罚
  - `offset_deg = angle_offset * 8.0`（从 15.0 缩小）
  - `_last_shot_info` 和 `_make_info()` 暴露 `offset_deg`
- `train.py`:
  - `TrainingMetrics` 增加 `angle_offsets` 列表，全链路适配
  - (0,1) 图表换为 Angle Offset 精度图
  - 终端日志增加 `Ang: X.X°`
  - Summary 增加角度趋势
  - 终端犯规日志改英文避免中文字体 warning
- `test.py`:
  - 增加 `miss_ball_penalty` 默认值验证

---

## Round 18：Color 阶段犯规后不回退 Red 的 Bug 修复

**严重程度：** 高 — 从项目初始就存在，影响所有之前的实验（Round 1-17）

### Bug 描述

斯诺克规则：进红球后打彩球，无论彩球那杆结果如何（进球、没进、犯规），下一杆应该回到 `red` 阶段打红球。

但代码中 `_compute_reward()` 的 FOUL PATH 只做了 `_switch_player()`，**没有把 `self.phase` 从 `"color"` 切回 `"red"`**。

后果：
- 犯规后 phase 仍停留在 `"color"` → 对手也被迫选彩球
- `_get_legal_targets()` 返回彩球 → action masking 让 agent 选彩球
- 但白球路径上可能先碰到红球 → 被判 `wrong_ball` 犯规
- 形成恶性循环：color 犯规 → 还是 color → 还是犯规...

### 修复

```python
# FOUL PATH 中，_switch_player() 之前：
if self.phase == "color":
    remaining_reds = self._count_remaining_reds()
    if remaining_reds > 0:
        self.phase = "red"
    else:
        self._enter_final_colors()
```

### 轨迹诊断发现的其他问题（非 bug，训练不充分）

1. **碰错球（wrong_ball）根因**：红球三角阵开球后，红球和彩球物理上混在一起。Agent 瞄准红球但白球路径上先碰到彩球。
2. **目标球选择不合理**：Agent 选远处的球（2-3m）而忽略近处的。
3. **袋口偏好**：Agent 固定选某个袋口，没学会就近选袋。

---

### Round 18 实验结果（700 Episodes）

**Bug 修复效果：**
- Reward: mean=10.1（vs Round 16/17 为负值，显著改善）
- WinRate: 40%→49%（接近 50% 平衡点）
- Foul Rate: 37%（从之前的 50% 下降，但仍偏高）

**犯规类型分布：**
- wrong_ball: 56.3%（仍是主因，红球/彩球物理混杂）
- miss_ball: 36.6%
- white_pocket: 8.5%

**关键发现：轨迹诊断揭示"运气进球"问题**

用 `diagnose_trajectory.py` 逐步分析模型行为，发现：
- 精确进球（Intentional）：2 次
- 运气进球（Lucky）：7 次
- **78% 的进球都是运气球！**

例如：选 Red6 为目标，但实际进袋的是 Red12。白球碰到了一堆红球，某颗碰巧滚进袋口。当前代码给这些运气球 +20 奖励，模型没有学习精确击球的动力。

---

## Round 19：精确进球奖励 + 力度追踪

**目标：** 只奖励精确进球（intentional pot），不奖励运气球（lucky pot），迫使模型学会精确瞄准。

### 核心改进

**1. 精确进球 vs 运气进球**

精确进球三个条件必须同时满足：
```
1. first_contact == chosen_target （白球先碰到了选定目标球）
2. chosen_target in pocketed     （目标球确实进了袋）
3. target 进入了 chosen_pocket   （进的是选定的袋口）
```

| 情况 | 奖励 |
|------|------|
| 精确进球（3条件全满足）| pot_reward(20) + break_bonus |
| 运气进球（合法但不精确）| lucky_pot_reward(0) |
| 犯规 | foul_penalty(-1/-2/-3) |
| 合法没进 | miss_penalty(-0.1) |

**2. 击球力度追踪（新指标）**
- 环境暴露 `V0`（击球力度 m/s）到 info
- 训练追踪 `mean(V0)` per episode
- 图表 (0,2) 从 Max Break 换为 Avg Power 图
- 终端日志: `V0: 3.2`

**3. 进球来源追踪 `_pocketed_into`**
- `_analyse_events()` 中记录每个球进了哪个袋口：`ball_id → pocket_id`
- 暴露到 info 中供诊断使用

### 代码改动

- `environment/pooltool_env.py`:
  - `RewardConfig` 增加 `lucky_pot_reward = 0.0`
  - `_analyse_events()` 新增 `self._pocketed_into` dict
  - `_compute_reward()` LEGAL PATH 区分精确/运气进球
  - `_make_info()` 暴露 `pocketed_into`, `chosen_target`, `V0`
  - `reset()` 和 physics_crash 路径初始化 `_pocketed_into`
- `train.py`:
  - `TrainingMetrics` 增加 `intentional_pots`, `lucky_pots`, `power_values` 列表
  - (0,2) 图表换为 Avg Power (V0) 图
  - (0,3) Pots 图增加精确/运气进球 MA 细分线
  - 终端日志: `Pots: 5.0 (int:0.5 luck:4.5) | Ang: 4.1° V0: 3.2`
  - Summary 文本框增加力度趋势和进球质量趋势
- `diagnose_trajectory.py`:
  - 显示每颗球进了哪个袋口 (`red_12→lc`)
  - 标注 `★PRECISE` 或 `(lucky)`
  - 结尾统计精确/运气进球数

---

## Round 19.5：Color 阶段合法没进球也不回退 Red 的 Bug + D 弧形渲染修复

### Bug 1：LEGAL PATH 中 color→red 阶段切换缺失

**发现方式：** 用 `diagnose_trajectory.py` 分析 Round 19 模型行为，发现 Step 11-13 连续在 color 阶段：

```
Step 10 | red   → POTTED red_14, red_08 (lucky)
Step 11 | color | Brown → OK (没进)  ← 应该切回 red
Step 12 | color | Brown → OK (没进)  ← BUG! 还在 color
Step 13 | color | Brown → OK (没进)  ← BUG!
```

**根因：** 与 Round 18 修的 FOUL PATH bug 类似，但这次是 LEGAL PATH 的 "没进球" 分支：
```python
else:
    # 没进球 → 换人
    self._switch_player()
    # ← 缺少 color→red 的阶段切换！
```

**修复：** 在换人后添加阶段切换：
```python
if self.phase == "color":
    remaining_reds = self._count_remaining_reds()
    if remaining_reds > 0:
        self.phase = "red"
    else:
        self._enter_final_colors()
```

### Bug 2：D 弧形渲染方向错误

pygame 坐标系 y 轴向下，`_to_px` 翻转了 y。原来用 `(-π/2, π/2)` 画出的弧形方向错误。改为 `(0, π)` 正确朝球台中心方向凸出。

### 关于"开球总碰彩球"

这不是 bug。标准斯诺克布局中：
- 白球在 D zone (y=0.709m)
- 黄/绿/棕在 baulk line 上 (y=0.709m) — 和白球同一水平线
- 蓝球在中间 (y=1.772m) — 挡在白球到红球的路径上
- 红球在 y≈2.7m

开球时 agent 需要学会：(1) 把白球放在 D zone 中远离 brown 的位置，(2) 选择角度绕过蓝球。这是需要训练学习的策略，当前 agent 训练不足。

---

## Round 20：可达性特征（Line-of-Sight Clearance）

**目标：** 让模型感知白球到每颗目标球之间是否有障碍物遮挡，避免选择无法直接命中的球。

### 问题

当前 54 维观测只有坐标信息，模型无法判断：
- 白球到目标球之间有没有其他球挡住
- 开局时蓝球挡在白球和红球之间，model 看不见这个信息

### 实现

**观测空间从 54 维扩展到 75 维**（新增 21 维 clearance）：

```
原有: white(2) + 15reds(30) + 6colours(12) + game_state(10) = 54
新增: clearance(21) — 白球到每颗球的直线可达性
总计: 75 维
```

**Clearance 值含义：**
- `+1.0` = 路径清晰，可直接命中
- `0.0` = 路径上有障碍球遮挡
- `-1.0` = 球已进袋/不存在

**计算方法：** 白球中心到目标球中心的线段上，是否有其他球的球心距 < 2×球半径（会被碰到）。使用点到线段距离的投影公式，O(n) per ball。

**性能：** 21 次 line_clear × ~20 balls = ~420 次距离计算/step，对比 pooltool 物理模拟（~20ms/step）完全可忽略。

### 代码改动

- `environment/pooltool_env.py`:
  - `observation_space` 从 54 扩展到 75
  - 新增 `_line_clear(target_bid)` 方法
  - `_get_obs()` 末尾追加 21 维 clearance
- `test.py`: 硬编码维度从 54 改为 75

---

## Round 20.5：出杆角度完全自由化

**问题：** 之前出杆方向被约束为 `arctan2(target - white) ± 8°`，agent 只能瞄准目标球方向的小范围内。这完全限制了弹库边打法（cushion-first shots），而真实斯诺克中通过弹库边命中被遮挡的球是基本技能。

**修改：** `action[4]` (原 angle_offset) 现在直接映射到 `[0°, 360°]` 的完全自由方向：

```python
# 之前：被约束在目标球方向附近
base_angle = arctan2(target - white)
phi = base_angle + offset * 8.0  # 只能偏±8°

# 现在：完全自由
phi = (action[4] + 1) / 2 * 360.0  # [0°, 360°)
```

**`target_idx` 的新角色：**
- 不再决定出杆方向
- 仅用于 action masking（选哪颗合法球）+ 精确进球判断（first_contact 是否等于 chosen_target）
- Agent 需要学会：选择目标球 + 选择出杆方向使白球能命中该目标球

**`offset_deg` 诊断指标的新含义：**
- 计算方式：`(phi - ideal_angle) % 360 - 180`
- 含义：实际出杆方向与"直线命中目标球"方向的偏差
- 如果 agent 用直线打：偏差小
- 如果 agent 用弹库边：偏差大但依然可能命中

---

## Round 21：命中目标球奖励（hit_target_reward）

**问题：** Round 20 (500ep) 诊断发现 agent 完全学不会命中目标球。原因是奖励结构中"碰到合法球但不是目标球"只罚 -0.1，和"碰到目标球但没进"一样都是 -0.1，agent 没有动力去精确瞄准。

**轨迹诊断数据（Round 20, 30步）：**
- 合法碰球但非目标球：17步（全是 -0.1）
- 碰错球犯规：10步
- 空杆：3步
- 精确进球：0步
- 运气进球（lucky）：reward=0

**修改：** 新增 `hit_target_reward = +2.0`

当 `first_contact == chosen_target`（白球先碰到了选定的目标球）但球没进袋时，给予 +2 正向奖励。

**新的完整奖励层级：**

```
精确进球（目标球进选定袋口）:   +20 + break_bonus
命中目标球（但没进袋）:         +2.0  ← 新增
运气进球（合法但非精确）:        0.0
碰到其他合法球（没进）:         -0.1
碰错球（犯规）:                -1.0
白球进袋:                      -2.0
空杆（没碰到任何球）:           -3.0
```

**设计理由：** 给 agent 一个清晰的中间学习目标——先学会命中目标球（+2 vs -0.1，差2.1分），再学会让球进袋（+20）。

### 代码改动

- `environment/pooltool_env.py`:
  - `RewardConfig` 增加 `hit_target_reward = 2.0`
  - LEGAL PATH "没进球"分支：区分 `first_contact == chosen_target`（+2）vs 其他合法球（-0.1）
  - 新增 `pot_type = 'hit_target'` 状态

---

### Round 21 实验结果（round20_hit_reward, 5000 Episodes）

**发现"大力出奇迹"策略：**

| 指标 | 前10% | 后10% | 趋势 |
|------|-------|-------|------|
| Power (V0) | 3.7 | 5.3 | ↑持续上升趋近上限6.0 |
| Lucky Pots | 0.9 | 2.1 | ↑同步上升 |
| Intentional Pots | 0.0 | 0.0 | **始终为0** |
| Wrong Ball | 10.7 | 14.9 | ↑恶化 |
| Angle Offset | 89° | 89° | 完全随机（未收敛）|

**根因分析：**

1. **得分 98.3% 来自对手犯规罚分**，不是自己进球
2. **犯规不计入 `shots_without_pocket`** → 互相犯规可无限持续 → episode 平均长 45 步
3. **win_reward=±30 驱动"赢就行"** → 大力散球能增加运气进球（得分）→ 同时对手更容易犯规 → 最终分多赢
4. Agent 理性选择了：不学精确(+2/+20)，而是大力散球赚比赛胜利(+30)

---

## Round 22：移除胜负奖励 + 犯规计入无进球步数

**目标：** 消除"大力出奇迹"的动机，让 agent 只能通过学习精确击球来获得正向 reward。

### 修改

**1. 移除胜负 reward（win/lose = 0）**

Agent 太弱时，胜负 reward 驱动 hack 策略。当前应专注学习基本功：命中目标球 → 精确进球。等 agent 学会精确进球后再恢复胜负奖励。

**2. 犯规计入 shots_without_pocket**

之前犯规不增加计数器，导致互相犯规的对局无限持续。修改后犯规也计入，确保 20 步内没有任何进球就终止对局。

### 代码改动

- `environment/pooltool_env.py`:
  - `win_reward = 0.0`, `lose_reward = 0.0`
  - FOUL PATH 中增加 `self.shots_without_pocket += 1`

---

### Round 22 实验结果（round22_no_winloss, 2400 Episodes）

移除 win_reward 后：
- ✅ Episode Length 45→27（犯规计入 shots_without_pocket 生效）
- ✅ V0 上升速度变慢（4.0→4.6 vs 之前 3.7→5.3）
- ❌ Intentional Pots 始终为 0
- ❌ 角度偏移 ~87°（完全随机，均匀分布期望值=90°）
- ❌ hit_target_reward(+2) 信号太稀疏，360°空间随机出杆几乎收不到

**根因：** 360° 自由角度搜索空间太大，稀疏的 hit_target_reward(+2) 不足以引导方向学习。

---

## Round 23：方向引导奖励（Aim Shaping）

**目标：** 用连续的 shaping reward 替代稀疏信号，让 agent 每一步都能感知"方向更准还是更偏"。

### 方案

新增 `aim_reward = aim_reward_scale × cos(angle_diff)`：

- `angle_diff = 0°`（完美瞄准）→ cos=1.0 → **+1.0**
- `angle_diff = 90°`（垂直打偏）→ cos=0 → **0**
- `angle_diff = 180°`（反方向）→ cos=-1 → **-1.0**

**每一步都给**，无论犯规与否。Agent 每次调整角度都有即时反馈。

### 设计要点

- 不稀疏：每步都有梯度，cos 在正确方向附近平滑
- 与 hit_target(+2) 兼容：方向对了(+1) + 命中(+2) = +3
- aim_reward_scale=1.0 量级合理：不会淹没 hit(+2)/pot(+20)/foul(-1~-3)

### 图表重构

同时将 3×4 图表升级为 4×4 布局（16 格），每项独立：
- (1,0) Intentional Pots、(1,1) Lucky Pots、(1,2) Foul Penalty、(1,3) Aim Reward
- (2,0) Foul Type、(2,1) Total Pots、(2,2) Foul Rate、(2,3) Max Break

### 代码改动

- `environment/pooltool_env.py`:
  - `RewardConfig` 增加 `aim_reward_scale = 1.0`
  - `_compute_reward()` 中添加 `bd['aim'] = scale × cos(angle_diff)`
- `train.py`:
  - `TrainingMetrics` 增加 `aim_rewards` 列表
  - plot 方法重写为 4×4 布局
  - 终端日志增加 `aim:X.X`
- `spec/rewards.md`: 更新奖励层级表

---

### Round 23 深入诊断

**诊断脚本 bug 修正**：`diagnose_trajectory.py` 用旧的 `* 8.0` 解码角度，显示"Ang: -5.5°"是假的。修正后真实偏差在 30°~150°。

**Aim reward 的三个致命问题：**

1. **Aim reward 是出杆前的静态信号**（基于 cos(angle_diff)），与击球物理结果完全脱钩。Agent 只需固定输出某个方向值就能拿 aim reward，不需要真正打中球。

2. **Lucky pot 延长对局赚更多 aim reward**：运气进球虽然 reward=0，但 `_handle_pot_scoring` 重置了 `shots_without_pocket` → 延长对局 → 更多步 aim reward。大力=更多运气进球=更长对局=更多 aim reward。

3. **Policy loss 持续上升 (-7.4→-2.1)** 是 SAC 探索崩溃信号：actor 在当前 value landscape 上越来越难找到高 Q action，卡在局部最优。

**数据支撑：**
```
Per-step reward: foul=-0.817  aim=+0.201  distance=+0.032
V0: 4.05 → 4.60（持续上升）
lucky: 0.82 → 1.14（同步上升）
角度偏差: 仍 ~74°（未学会瞄准）
```

---

## Round 24：接近目标球奖励（Approach Reward）

**目标：** 用基于物理模拟结果的连续 reward 替代出杆前的静态 aim reward。

### 核心思路

**Aim reward 问题**：出杆前计算，和击球结果脱钩，可被 hack。

**Approach reward**：计算白球在运动轨迹中距离目标球的**最近距离**（closest approach distance）。这是物理模拟的结果，不能 hack。

```python
min_dist = min(|white_pos(t) - target_pos(t)|) for all t in trajectory
approach_reward = scale × max(0, 1 - min_dist / threshold)
```

- 白球命中目标球（dist≈0）→ +3.0
- 白球擦边而过（dist=0.25m）→ +1.5
- 白球打偏（dist>0.5m）→ 0
- 弹库边命中 → 同样能得高分（基于最终轨迹距离）

### 同时修复的问题

**运气进球不再延长对局**：lucky pot 后恢复 `shots_without_pocket` 到进球前的值。

### 参数

- `approach_reward_scale = 3.0`（与 foul penalty -1~-3 量级匹配）
- `approach_threshold = 0.5m`（约球台宽度的 1/3）

### 代码改动

- `environment/pooltool_env.py`:
  - `_analyse_events()` 新增 `self._closest_approach`
  - step() 中从 `result.balls["white"].history.states` 计算最近距离
  - `_compute_reward()` 中 `bd['approach']` 替代 `bd['aim']`
  - `RewardConfig`: `aim_reward_scale` → `approach_reward_scale=3.0, approach_threshold=0.5`
  - 运气进球后恢复 `shots_without_pocket`
- `train.py`: `aim_rewards` → `approach_rewards` 全链路替换
- `diagnose_trajectory.py`: 修正角度解码

---

### Round 24 实验结果（round24_approach, 2500 Episodes）

**Approach reward 也被 hack 了：**

```
Per-step: foul=-0.835  approach=+0.855  (完美抵消！)
角度偏差: 87-89°（完全随机）
100% episodes 拿到 approach reward（均值 16/ep）
V0: 4.1→4.8（仍在上升）
Policy Loss: -5.3→-1.0（SAC 探索崩溃）
```

**原因**：白球在球台中随便往红球区打，路径就会经过很多红球附近（threshold=0.5m太大，红球密集分布在 0.5m 范围内），不需要精确瞄准。

**结论**：在 agent 学会基本技能之前，任何基于"距离目标球近"的 shaping 都容易被 hack。

---

## Round 25：方案 B — 纯结果奖励 + ±15° 角度约束

**目标：** 移除所有 shaping，回到最简单直接的方案。用角度约束缩小搜索空间替代 shaping 的引导作用。

### 设计理由

1. Agent 连直线打中目标球都做不到，弹库边为时过早
2. ±15° 约束让搜索空间缩小到 30°/360° = 1/12，命中概率大幅提升
3. 纯结果奖励不可能被 hack——只有真正打中/进球才给奖励
4. Round 15-18 的历史数据显示 ±15° 下 agent 能学会碰到球

### 修改

- **角度**：360° 自由 → 目标球方向 ±15°
- **移除 approach reward**：不再有任何 shaping
- **保留**：hit_target(+2), intentional_pot(+20), foul(-1/-2/-3), miss(-0.1)

### 代码改动

- `environment/pooltool_env.py`:
  - `phi = base_angle + action[4] * 15.0`
  - 移除 `approach_reward_scale`, `approach_threshold` 参数
  - 移除 `bd['approach']` 计算代码
- `diagnose_trajectory.py`: 角度解码改回 ±15°
- `spec/rewards.md`: 全面更新为纯结果奖励 + 课程学习规划
- `spec/features.md`: 角度范围更新

### 未来：课程学习

当 `intentional_pots > 0.5/ep` 时：
- Phase 2: 角度扩大到 ±45°
- Phase 3: 360° 自由 + 恢复 win/loss reward

---

## 待办 / 未来改进方向

- [ ] **Round 25 训练**：纯结果奖励 + ±15° 约束，验证 hit_target 是否持续上升
- [ ] **课程学习 Phase 2**：intentional_pots > 0.5 后扩大角度
- [ ] **恢复胜负奖励**：Phase 2+ 后开启
- [ ] **训练加速**：GPU / 多进程
