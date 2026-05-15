# Reward 构建说明

## 设计哲学

**纯结果奖励 + 角度约束**

- 所有 reward 基于击球的物理结果，不加任何 shaping
- 用 ±15° 角度约束缩小搜索空间，替代 shaping 的引导作用
- 精确进球分层：精确进球 / 命中目标 / 运气进球 / 犯规

### 历史教训：Shaping reward 容易被 hack

| 轮次 | Shaping 方式 | 被 hack 的方式 |
|------|------------|---------------|
| R23 | aim = cos(angle_diff)，出杆前静态 | 固定朝一个方向打就能拿 reward |
| R24 | approach = 白球最近距离，物理结果 | 随便往红球区打就接近很多球 |

---

## 奖励层级总览

纯结果奖励，无 shaping。从高到低：

| 行为 | Reward | 条件 |
|------|--------|------|
| 精确进球 | +20 + break_bonus | first_contact==target AND target进了chosen_pocket |
| 命中目标球 | +2.0 | first_contact==target 但球没进袋 |
| 运气进球 | 0.0 | 合法进球但不满足精确条件（不重置对局计时器）|
| 碰到其他合法球 | -0.1 | 合法击球，碰到了某颗合法球，但不是目标球 |
| 碰错球(wrong_ball) | -1.0 | 白球先碰到了非法球 |
| 白球进袋 | -2.0 | 白球落袋 |
| 空杆(miss_ball) | -3.0 | 白球没碰到任何球 |

---

## 详细参数 (RewardConfig)

```python
@dataclass
class RewardConfig:
    # ══ Behavior Reward ══
    pot_reward: float = 20.0           # 精确进球
    break_bonus: float = 2.0           # 连续进球递增
    lucky_pot_reward: float = 0.0      # 运气进球（不鼓励，不重置对局计时器）
    hit_target_reward: float = 2.0     # 命中目标球

    foul_penalty: float = -1.0         # 碰错球
    miss_ball_penalty: float = -3.0    # 空杆
    white_pocket_penalty: float = -2.0 # 白球进袋
    miss_penalty: float = -0.1         # 合法碰球但非目标球且没进

    # ══ Outcome Reward ══
    win_reward: float = 0.0            # 暂时关闭
    lose_reward: float = 0.0
```

---

## 精确进球的三个条件

必须**同时满足**才给 `pot_reward`：

1. **first_contact == chosen_target** — 白球先碰到了选定目标球
2. **chosen_target in pocketed** — 目标球进了袋
3. **pocketed_into[chosen_target] == chosen_pocket** — 进的是选定袋口

不满足全部三条但合法进球 → `lucky_pot_reward` (0)

---

## 犯规判定逻辑

```python
is_foul = (
    not white_hit_any          # 空杆
    or wrong_first_contact     # 碰错球
    or not chose_legal         # 选了非法球（action masking 后不会触发）
    or white_pocketed          # 白球进袋
)
```

犯规类型优先级：miss_ball > wrong_ball > illegal_choice > white_pocket

---

## 犯规后的阶段切换

无论犯规还是合法没进球，只要在 color 阶段：
- 有红球剩余 → phase 回到 "red"
- 无红球 → 进入 "final_colors"

---

## Reward 分解追踪

```python
bd = {
    'foul': float,       # 犯规惩罚（0 如果合法）
    'distance': float,   # 进球/命中/miss 奖励
    'win_loss': float,   # 终局奖励（暂时=0）
    'total': float,      # sum of above
    'foul_type': str,    # 'miss_ball'|'wrong_ball'|'white_pocket'|None
    'pot_type': str,     # 'intentional'|'lucky'|'hit_target'|None
}
```

---

## 未来：课程学习（Curriculum Learning）

当 agent 掌握当前阶段后，逐步放宽约束：

| Phase | 角度范围 | 目标 | 转换条件 |
|-------|---------|------|---------|
| 1（当前）| ±15° | 直线命中 + 精确进球 | intentional_pots > 0.5/ep |
| 2 | ±45° | 学习更复杂角度 | intentional_pots > 1.0/ep |
| 3 | 360° 自由 | 弹库边打法 | — |

可配合恢复 win/loss reward（Phase 2+）。
