# 技术细节

本文档详细记录 Snooker RL 项目中各模块的技术设计与实现细节。

---

## 1. 环境 (`environment/snooker_env.py`)

### 1.1 物理引擎

- 使用 **pymunk 7.x**（基于 Chipmunk 2D 物理引擎）
- 零重力 2D 空间，`damping=0.99` 模拟球台摩擦衰减
- 碰撞检测通过 `space.on_collision()` 注册回调（pymunk 7.x API）
- 碰撞类型：`COL_WHITE=1`, `COL_RED=2`, `COL_COLOR=3`, `COL_CUSHION=4`

### 1.2 球台布局

| 参数 | 值 | 说明 |
|------|-----|------|
| 球台尺寸 | 1200×600 px | 2:1 比例，模拟标准斯诺克桌面 |
| 球半径 | 12 px | |
| 袋口半径 | 22 px | |
| 库边宽度 | 25 px | |
| 袋口数量 | 6 个 | 四角 + 两侧中袋 |

**D 区（半圆区域）：**
- 圆心 x = `table_width × 0.22`（baulk 线位置）
- 半径 = `table_height × 0.15`
- 开球和犯规后 ball-in-hand 时，母球只能放置在 D 区内

**彩球位点：**
| 球 | 位置 |
|----|------|
| 黄 | (baulk_x, table_height × 0.35) |
| 绿 | (baulk_x, table_height × 0.65) |
| 棕 | (baulk_x, table_height / 2) |
| 蓝 | (table_width / 2, table_height / 2) |
| 粉 | (table_width × 0.65, table_height / 2) |
| 黑 | (table_width × 0.85, table_height / 2) |

### 1.3 游戏阶段状态机

```
     ┌─── 有红球 ───┐
     ↓               ↑
   [red] ──进红──→ [color] ──进彩──→ 有红球？→ 回到 [red]
     │                                    ↓ 无红球
     └── 红球清完 ──→ [final_colors] ──→ 按顺序清彩 ──→ 结束
```

- `red` 阶段：只能击打红球
- `color` 阶段：可击打任意彩球（进球后 re-spot）
- `final_colors` 阶段：按 黄→绿→棕→蓝→粉→黑 顺序击打（进球不 re-spot）

### 1.4 犯规检测

以下情况判定犯规：
1. 白球落袋
2. 母球未接触任何目标球（空杆）
3. 选择了非法的目标球
4. 母球第一个碰到的球不符合当前阶段要求

犯规后果：
- 对手获得至少 4 分（若涉及高分彩球则按该球分值）
- 犯规中被打进的彩球 re-spot
- 白球落袋 → 对手获得 ball-in-hand
- 连续 3 次犯规 → 丢掉该局（对手 +7 分）

### 1.5 模拟流程

```
step(action) 内部流程：
  1. [ball-in-hand] 在 D 区放置母球
  2. 从 action 解析出目标球、角度偏移、力度
  3. 计算 base_angle = arctan2(target_y - white_y, target_x - white_x)
  4. shot_angle = base_angle + offset (±15°)
  5. 记录击球前各球到袋口的距离
  6. 施加速度：v = (power × 2000 + 200) × [cos(shot_angle), sin(shot_angle)]
  7. 物理模拟直到所有球停止（最多 500 帧）
  8. 记录击球后各球到袋口的距离
  9. 计算奖励 + 处理换人/犯规
```

---

## 2. 动作空间设计

### 2.1 5 维连续动作

所有维度均输出 [-1, 1]，环境内部负责映射：

```
action = [place_x, place_y, target_idx_raw, angle_offset, power_raw]
           dim 0     dim 1       dim 2          dim 3       dim 4
```

| 维度 | 映射逻辑 | 使用条件 |
|------|---------|---------|
| `place_x`, `place_y` | 归一化坐标 → D 区半圆内的 (px, py) | 仅 ball_in_hand=True |
| `target_idx_raw` | `idx = (raw + 1) / 2 × N` → 从全部未落袋球中选第 idx 个 | 每杆必用 |
| `angle_offset` | `offset_rad = raw × π/12` → ±15° 偏转 | 每杆必用 |
| `power_raw` | `power = clip((raw + 1) / 2, 0.1, 1.0)` | 每杆必用 |

### 2.2 以目标球为基准的坐标系

击球角度不是绝对角度，而是相对于**白球→目标球连线**的偏移：

```
                    offset = +15°
                   ╱
                  ╱
    白球 ●━━━━━━━━━━━━━━● 目标球    ← base_angle（offset=0 时完美直线）
                  ╲
                   ╲
                    offset = -15°
```

这样设计的好处：
- 模型不需要学习绝对方向感（从极坐标学到球台几何）
- offset=0 就是"直接打向目标球"，大幅降低了学习难度
- ±15° 的小范围足够覆盖斯诺克中的大部分有效击球

### 2.3 设计考量：为什么是单步而非两步

当前选球和击球在同一个 action 中完成。5 个维度通过高斯分布独立采样，这意味着 `angle_offset` 不会因为 `target_idx` 的值不同而自动调整。

未来可能改为两步决策（step1: 选球 → step2: 角度+力度），使角度能条件依赖于选球结果。

---

## 3. 观测空间

### 3.1 54 维向量结构

所有值归一化到 [-1, 1]：

```
[0:2]    白球 (x, y)
[2:32]   15 颗红球 (x, y) × 15 = 30    ← 已落袋的球填 (-1, -1)
[32:44]  6 颗彩球 (x, y) × 6 = 12      ← 已落袋的球填 (-1, -1)
[44:54]  游戏状态：
         [44] current_break / 147
         [45] phase: red=0, color=0.5, final_colors=1
         [46] next_color_index / 6
         [47] remaining_balls / 21
         [48] ball_in_hand: 0 or 1
         [49] current_player: 0 or 1
         [50] score_p1 / 147
         [51] score_p2 / 147
         [52] consecutive_fouls / 3
         [53] shots_without_pocket / 10
```

位置归一化方式：`norm_x = (x / table_width) × 2 - 1`

---

## 4. 模型架构 (`algorithms/ppo.py`)

### 4.1 Actor-Critic 网络

```
输入 (54-dim state)
       │
  ┌────┴────────────────────────────┐
  │  Shared Feature Extractor       │
  │  Linear(54 → 256) + LN + Tanh  │
  │  Linear(256 → 256) + LN + Tanh │
  │  Linear(256 → 256) + LN + Tanh │
  └────┬────────────────────────────┘
       │ 256-dim features
  ┌────┴────┐     ┌─────────┐
  │  Actor  │     │ Critic  │
  │  Head   │     │  Head   │
  │ 256→128 │     │ 256→128 │
  │  Tanh   │     │  Tanh   │
  │ 128→5   │     │ 128→1   │
  │  Tanh   │     └────┬────┘
  └────┬────┘          │
       │               V(s)
  mean ∈ [-1, 1] (5-dim)
       +
  log_std (5-dim, learnable, init=-0.5)
       │
  Normal(mean, exp(log_std))
       │
  sample → clamp(-1, 1) → action
```

**关键设计：**
- **LayerNorm**：每个隐层后加入，稳定训练
- **共享特征层**：Actor 和 Critic 共用前 3 层，减少参数
- **可学习 log_std**：初始 -0.5（std ≈ 0.6），clamp 在 [-3, 0.5] 内
- **Tanh 输出**：actor 输出天然在 [-1, 1]，加 clamp 做双重保险

### 4.2 PPO 算法细节

- **GAE (Generalized Advantage Estimation)**：`γ=0.99`, `λ=0.95`
- **Clipped Surrogate Objective**：`ε=0.2`
- **K Epochs**：每次 update 对同一批数据训练 10 轮
- **Entropy Bonus**：`coef=0.05`（鼓励探索）
- **Gradient Clipping**：`max_norm=0.5`
- **Update Interval**：每 1024 步收集一批数据后更新

### 4.3 Self-Play 训练

两个玩家共享同一个策略网络。训练循环：

```
每个 episode:
  reset() → P1 先手, ball_in_hand=True
  while not done:
    obs = env._get_obs()                    # 包含 current_player 信息
    action = shared_policy(obs)             # 同一个网络
    obs', reward, done, info = env.step(action)
    memory.add(obs, action, reward, ...)    # 存入共享 buffer
    # 环境内部自动处理换人
  每 1024 步: PPO update
```

---

## 5. 奖励系统

### 5.1 选球 × 角度 × 结果 联合奖励

核心理念：**选球和角度之间有可评价的关系**。选了某个球，角度偏移是否对准它，物理上是否碰到了它——这三者构成层级 reward：

```
选错球（非法目标）
  └── -1.5

选对球（合法目标）
  ├── Layer 0: +0.2                      ← 基础奖励
  ├── Layer 1: +0.4 × accuracy           ← 角度越准越好
  ├── Layer 2: 碰到选定球？
  │   ├── 碰到: +0.5 + 0.3 × accuracy   ← 精准击中
  │   ├── 碰别的: +0.05                  ← 有接触
  │   └── 全空: -1.0                     ← 空杆
  └── Layer 3: 首触非选定球？ -0.3       ← 擦到别的球
```

其中 `accuracy = 1 - |offset| / (π/12)`，范围 [0, 1]。

### 5.2 犯规惩罚

| 犯规类型 | 惩罚 |
|---------|------|
| 白球落袋 | -4.0 |
| 其他犯规 | -1.5 |
| 连续 3 犯规 | -5.0 + 丢局 |

### 5.3 进球奖励

| 球类 | 奖励 |
|------|------|
| 红球 | +5.0 / 颗 |
| 彩球（红球阶段后） | 分值 × 1.5 |
| 最终彩球（清彩阶段） | 分值 × 2.0 |
| 全部清台 | +30 |

### 5.4 趋近奖励 (Approach Shaping)

每杆击球后，计算所有目标球离最近袋口距离的变化量：

```
approach = Σ (pre_dist - post_dist) / table_diagonal
reward += clip(approach × 3.0, -0.5, 2.0)
```

球被推向袋口 → 正奖励；球被推远 → 负奖励。

### 5.5 其他

- **步惩罚**：每步 -0.02（防止磨时间）
- **终局条件**：全部清完 / 连续 10 杆无进球 / 连续 3 犯规

---

## 6. 可视化系统

### 6.1 渲染模式

- `render_mode='human'`：Pygame 实时窗口，窗口尺寸 1200×680（球台 + 80px 信息栏）
- `render_mode='rgb_array'`：返回 numpy 数组用于录制

### 6.2 实时击球观看

开启渲染后，每一杆的流程：

| 阶段 | 显示内容 | 持续时间 |
|------|---------|---------|
| 瞄准 | 黄色瞄准线 + 白色辅助线（白球→目标球）+ 目标球高亮圈（绿=合法/红=非法）+ 力度条 | 400ms |
| 球运动 | 物理模拟逐帧渲染（60fps），球碰撞/反弹/入袋全过程可见 | 直到球停 |
| 结果 | 底部信息栏显示上一杆详情 | 下一杆开始前 |

### 6.3 HUD 信息

- **球台上方**：当前玩家、阶段、双方比分、break、ball-in-hand 状态
- **底部信息栏**：上一杆的击球者、目标球、合法性、力度、角度偏移

### 6.4 调试工具 (`debug_shots.py`)

逐杆打印所有中间值，用于验证角度计算正确性：

```
  #  Player  Target  Legal   base°  offset°  shot°    vx      vy    |v|  Power
  1  P   1   red_9    ✓       8.5    -12.3    -3.8    786    -183    807   30%
```

---

## 7. 文件清单

| 文件 | 职责 |
|------|------|
| `environment/snooker_env.py` | Gymnasium 环境：物理、规则、奖励、渲染 |
| `algorithms/ppo.py` | PPO：ActorCritic 网络、Memory、GAE、训练更新 |
| `train.py` | Self-Play 训练循环 + 指标记录 + 绘图 |
| `evaluate.py` | 模型评估 + 可视化对局 + 报告生成 |
| `debug_shots.py` | 击球调试：打印每杆中间值 |
| `test.py` | 测试套件（环境、PPO、集成、self-play） |
| `config.py` | 超参数配置字典 |
