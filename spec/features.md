# 特征构建说明

## 观测空间总览

**总维度：75 维**，全部归一化到 [-1, 1] 范围。

```
obs[0:2]    白球坐标 (x, y)
obs[2:32]   15 颗红球坐标 (red_01 ~ red_15，各 2 维)
obs[32:44]  6 颗彩球坐标 (yellow, green, brown, blue, pink, black，各 2 维)
obs[44:54]  游戏状态 (10 维)
obs[54:75]  可达性特征 (21 维)
```

---

## 1. 球坐标特征 (44 维)

### 归一化方式

```python
def _norm_pos(x, y):
    return ((x / table_w) * 2 - 1, (y / table_l) * 2 - 1)
```

- 球桌左下角 (0, 0) → (-1, -1)
- 球桌右上角 (table_w, table_l) → (+1, +1)
- 球已进袋 → (-1, -1) 作为特殊标记

### 固定位置编码

每颗球在 obs 中的位置是硬编码的，**不会因为其他球进袋而移位**：

| 球 | obs 索引 |
|----|---------|
| white | [0:2] |
| red_01 | [2:4] |
| red_02 | [4:6] |
| ... | ... |
| red_15 | [30:32] |
| yellow | [32:34] |
| green | [34:36] |
| brown | [36:38] |
| blue | [38:40] |
| pink | [40:42] |
| black | [42:44] |

---

## 2. 游戏状态特征 (10 维)

| 索引 | 含义 | 归一化 |
|------|------|--------|
| [44] | current_break | / 147 |
| [45] | phase | red=0, color=0.5, final_colors=1.0 |
| [46] | next_color_index | / 6 |
| [47] | remaining balls | / 21 |
| [48] | ball_in_hand | 0 or 1 |
| [49] | current_player | 0 or 1 |
| [50] | P1 score | / 147 |
| [51] | P2 score | / 147 |
| [52] | consecutive_fouls | / max_consecutive_fouls |
| [53] | shots_without_pocket | / max_shots_without_pocket |

---

## 3. 可达性特征 (21 维)

**Line-of-Sight Clearance**: 白球到每颗目标球之间是否有障碍物遮挡。

| 索引 | 含义 |
|------|------|
| [54:69] | 15 颗红球的可达性 (red_01 ~ red_15) |
| [69:75] | 6 颗彩球的可达性 (yellow ~ black) |

**取值：**
- `+1.0` = 路径清晰，白球可直接命中该球
- `0.0` = 路径上有障碍球遮挡
- `-1.0` = 球已进袋/不存在

### 计算方法

对于白球 W 和目标球 T，检查线段 W→T 上是否有其他球 B 的中心距线段 < 2×球半径：

```python
# 投影 B 到线段 W→T 上
t = dot(B-W, T-W) / |T-W|²
t = clamp(t, 0, 1)
closest = W + t * (T-W)
dist = |B - closest|
if dist < 2 * ball_radius:
    blocked = True
```

---

## 4. 动作空间 (8 维)

所有 action 值在 [-1, 1]，映射关系如下：

| 索引 | 含义 | 映射 |
|------|------|------|
| [0] | place_x | D-zone 白球 x 位置（仅 ball_in_hand 时有效）|
| [1] | place_y | D-zone 白球 y 位置（仅 ball_in_hand 时有效）|
| [2] | target_idx | 映射到合法目标球列表中的索引（action masking）|
| [3] | pocket_idx | 映射到 6 个袋口之一 |
| [4] | shot_angle | 目标球方向 ±15°（Phase 1，后续课程学习逐步放宽）|
| [5] | power | 击球力度 → [0.5, 6.0] m/s |
| [6] | b_spin | 纵向旋转（前旋/回旋）→ [-0.8, +0.8] |
| [7] | a_spin | 侧旋 → [-0.5, +0.5] |

### 关键设计决策

1. **Action Masking (target_idx)**：agent 只能从当前阶段的合法球中选择，不需要学习规则
2. **出杆方向完全自由**：不绑定目标球方向，允许弹库边打法
3. **target_idx 的角色**：仅声明"意图"——用于精确进球奖励判断（first_contact == chosen_target）

---

## 5. 球桌物理参数

| 参数 | 值 |
|------|-----|
| 球桌尺寸 | 1.746m × 3.545m (12ft 标准斯诺克) |
| 球半径 | 0.0262m |
| Baulk line | y = 0.2 × table_l = 0.709m |
| D-zone 半径 | 0.292m (11.5 inches) |
| 袋口数 | 6 (lb, lc, lt, rb, rc, rt) |

### 标准开局布局

- 白球：D-zone 中心 (0.873, 0.709)
- 黄/绿/棕：baulk line 上 (y=0.709)
- 蓝：球台中心 (0.873, 1.772)
- 粉：(0.873, 2.658)
- 黑：(0.873, 3.222)
- 红球三角阵：粉球后方 (y≈2.71~2.85)
