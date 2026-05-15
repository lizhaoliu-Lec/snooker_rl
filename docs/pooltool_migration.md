# Pooltool 物理引擎迁移方案

本文档记录将斯诺克 RL 项目的物理后端从 pymunk 迁移到 [pooltool](https://github.com/ekiefl/pooltool) 的技术调研与迁移计划。

---

## 1. Pooltool 简介

[Pooltool](https://github.com/ekiefl/pooltool)（v0.6.0）是一个专为科学与工程设计的台球物理模拟器，发表在 [JOSS](https://joss.theoj.org/papers/10.21105/joss.07301)（Journal of Open Source Software）并通过 [pyOpenSci](https://www.pyopensci.org/) 同行评审。

- **许可证**：Apache-2.0
- **主要语言**：Python（93.6%）+ C 加速核心
- **安装**：`pip install pooltool-billiards`

---

## 2. 核心特性

### 2.1 完整的旋转物理

pooltool 实现了完整的台球旋转力学，包括：

| 参数 | 含义 | 实际效果 |
|------|------|---------|
| `V0` | 击球速度 (m/s) | 力度控制 |
| `phi` | 球杆瞄准方向 (度) | 击球方向 |
| `theta` | 球杆仰角 (度) | 扎杆/跳球（0°=平打，>30°=massé） |
| `a` | 侧旋偏移 | 左塞 (a<0) / 右塞 (a>0)，影响库边反弹角度 |
| `b` | 上下旋偏移 | 高杆 (b>0) → 白球跟进；低杆 (b<0) → 白球缩回 |

**已验证的旋转效果示例（直球打蓝球入右中袋）：**

| 参数 | 结果 |
|------|------|
| `b=0.0`（无旋转） | 蓝球 ✓ 进，白球跟进落袋 (in-off) |
| `b=+0.5`（高杆） | 蓝球 ✓ 进，白球跟进落袋 (in-off) |
| `b=-0.5`（低杆） | 蓝球 ✓ 进，**白球缩回不跟进** ✓ |
| `b=-0.8`（强低杆） | 白球弹回太远，蓝球速度不够没进 |

### 2.2 事件驱动物理引擎

不同于 pymunk 的离散时间步进（需要 sub-stepping hack），pooltool 使用**基于事件的碰撞求解算法**：

- 精确计算下一次碰撞发生的时间点，不依赖时间步长
- 不存在"隧道效应"（小步长下球穿过另一个球）
- 自动处理球-球、球-库边、球-落袋等所有事件
- 每次模拟返回完整的事件列表 (`system.events`)

**事件类型：**
- `STICK_BALL` — 球杆击球
- `BALL_BALL` — 球-球碰撞
- `BALL_LINEAR_CUSHION` — 球-直线库边碰撞
- `BALL_CIRCULAR_CUSHION` — 球-弧形库边碰撞
- `BALL_POCKET` — 球落袋
- `SLIDING_ROLLING` / `ROLLING_SPINNING` / `ROLLING_STATIONARY` — 运动状态转换

### 2.3 内置斯诺克支持

```python
import pooltool as pt

# 一行创建标准斯诺克桌
table = pt.Table.from_game_type(pt.GameType.SNOOKER)

# 标准开球布局（22 颗球）
balls = pt.get_rack(pt.GameType.SNOOKER, table)
```

**斯诺克桌参数（真实比例）：**

| 参数 | 值 |
|------|-----|
| 桌面尺寸 | 1.746m × 3.545m（标准 12 英尺桌） |
| 球半径 | 26.2mm |
| 球质量 | 0.14 kg |
| 角袋半径 | 43mm |
| 中袋半径 | 42.7mm |
| 6 个袋口 | lb, lc, lt, rb, rc, rt |
| 球体摩擦系数 | u_s=0.5, u_r=0.01 |
| 球-球弹性 | e_b=0.95 |
| 球-库边弹性 | e_c=0.85 |

### 2.4 3D 可视化

pooltool 内置基于 **Panda3D** 的 3D 交互式渲染：

```python
pt.show(system)  # 打开 3D 窗口查看模拟结果
```

支持旋转视角、回放、慢动作等。（训练时可关闭以提速）

### 2.5 完整的 Python API

```python
import pooltool as pt

# 创建系统
table = pt.Table.from_game_type(pt.GameType.SNOOKER)
balls = pt.get_rack(pt.GameType.SNOOKER, table)
cue = pt.Cue(cue_ball_id='white')
system = pt.System(table=table, balls=balls, cue=cue)

# 瞄准与击球
phi = pt.aim.at_ball(system, 'red_14')  # 计算瞄准角度
cue.set_state(V0=3.0, phi=phi, theta=0.0, a=0.0, b=-0.3)

# 物理模拟
result = pt.simulate(system, inplace=False)

# 获取结果
for bid, ball in result.balls.items():
    pos = ball.state.rvw[0][:2]  # (x, y) 坐标
    is_pocketed = (ball.state.s == 4)
```

**辅助函数：**
- `pt.aim.at_ball(system, ball_id)` — 计算白球瞄准目标球中心的角度
- `pt.aim.at_pos(system, (x, y))` — 瞄准任意位置
- `pt.pot.calc_potting_angle(...)` — 计算进球角度（需 numpy 兼容）
- `pt.pot.pick_easiest_pot(...)` — 自动选择最易进的球/袋组合

---

## 3. 性能基准（本机测试）

| 场景 | 速度 |
|------|------|
| 双球模拟（白球 + 1 目标球） | **~90 shots/sec** |
| 满场模拟（22 球开球） | **~3 shots/sec** |
| 单次满场模拟延迟 | ~0.3s |

> 测试环境：macOS, Python 3.12, pooltool 0.6.0

RL 训练中多数杆次只涉及 2-5 颗球移动，预计平均速度在 **20-50 shots/sec** 之间，足以支撑训练。

---

## 4. 新 Action Space 设计（7 维）

迁移后 action space 从 5 维扩展到 **7 维**，完整覆盖斯诺克击球技术：

| 维度 | 参数 | 映射范围 | 说明 |
|------|------|---------|------|
| 0 | `place_x` | D-zone 坐标 | ball-in-hand 时母球放置 |
| 1 | `place_y` | D-zone 坐标 | ball-in-hand 时母球放置 |
| 2 | `target_idx` | 离散球索引 | 选择目标球 |
| 3 | `phi_offset` | ±15° | 基于白球→目标球连线的角度偏移 |
| 4 | `V0` | 0.5 ~ 6.0 m/s | 击球力度 |
| 5 | `b` (topspin/backspin) | -1.0 ~ +1.0 | 高杆/低杆 |
| 6 | `a` (sidespin) | -1.0 ~ +1.0 | 左塞/右塞 |

> `theta`（球杆仰角/扎杆）暂时固定为 0°，后续可考虑加入。

---

## 5. 架构设计

```
┌──────────────────────────────────────┐
│       SnookerEnv (Gymnasium)         │  ← 我们实现
│  - reset() / step() / render()       │
│  - Observation: 球位+旋转+游戏状态    │
│  - Action: 7-dim continuous          │
│  - Reward: 复用 + 增强               │
│  - 规则引擎: 犯规/换人/阶段转换       │
├──────────────────────────────────────┤
│         pooltool (Physics)           │  ← 现成的
│  - 真实旋转物理 (spin/english/massé) │
│  - 事件驱动碰撞检测 (无隧道效应)      │
│  - 标准斯诺克桌/球参数                │
│  - C 加速计算核心                     │
└──────────────────────────────────────┘
```

我们只需要实现 Gymnasium wrapper 层，物理模拟完全交给 pooltool。

---

## 6. 迁移计划

### Phase 1: 物理测试验证
- [x] 安装 pooltool，验证 API
- [x] 直球进袋、角度球进袋
- [x] 旋转效果验证（高杆/低杆/左塞/右塞）
- [x] 模拟速度基准测试
- [ ] 可视化测试脚本（对标 `test_physics.py`）

### Phase 2: Gymnasium Wrapper
- [ ] 实现 `SnookerEnv(gym.Env)` — 对接 pooltool 物理
- [ ] 观测空间设计（球位 + 游戏状态）
- [ ] 7 维 action space 映射
- [ ] 斯诺克规则引擎（犯规/阶段/换人）
- [ ] Reward 系统（复用现有 + 旋转控制奖励）

### Phase 3: 训练适配
- [ ] 更新 PPO 网络（适配 7-dim action）
- [ ] 更新 `train.py` / `evaluate.py`
- [ ] 训练并验证
