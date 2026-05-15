# Snooker RL

基于强化学习的斯诺克 AI，支持 **PPO** 和 **SAC** 两种算法进行**双人 Self-Play** 训练。

Agent 模拟人类决策过程：先选球 → 选袋口 → 再以白球-目标球连线为基准确定击球角度、力度和旋转。

## 项目结构

```
snooker_rl/
├── environment/              # 游戏环境
│   ├── pooltool_env.py       # 斯诺克 Gymnasium 环境（pooltool 物理引擎）
│   └── __init__.py
├── algorithms/               # 强化学习算法
│   ├── ppo.py                # PPO (Actor-Critic + GAE, on-policy)
│   └── sac.py                # SAC (Replay Buffer + 自动温度, off-policy)
├── utils/                    # 工具函数
│   └── __init__.py
├── docs/                     # 项目文档
│   ├── technical_details.md  # 技术细节文档
│   ├── pooltool_migration.md # Pooltool 迁移方案
│   └── changelog.md          # 改进日志
├── experiments/              # 训练产出（每次实验独立子目录）
│   ├── sac_20260512_143025/  # 自动命名：{algo}_{timestamp}
│   └── round9_stability/     # 或自定义名称（--run_name）
├── train.py                  # Self-Play 训练脚本（支持 --algo ppo/sac）
├── evaluate.py               # 可视化评估脚本（自动检测算法）
├── test_pooltool.py          # 物理引擎可视化测试（15 个场景：旋转/进球/开球/速度）
├── test.py                   # 自动化测试套件（12 个测试）
├── config.py                 # 超参数配置
└── requirements.txt          # Python 依赖
```

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行测试

```bash
python test.py
```

### 训练模型

```bash
# PPO 训练（on-policy，默认）
python train.py --algo ppo --num_episodes 2000

# SAC 训练（off-policy + Replay Buffer，样本效率更高，推荐）
python train.py --algo sac --num_episodes 5000

# 自定义实验名
python train.py --algo sac --num_episodes 5000 --run_name round9_stability

# 带可视化训练
python train.py --algo sac --num_episodes 500 --render

# 从已有模型继续训练
python train.py --algo sac --num_episodes 1000 --load_model experiments/sac_20260512_143025/sac_snooker_ep500.pt

# 分析已有训练数据（不启动训练）
python train.py --analyse experiments/sac_20260512_143025/metrics_final.json

# 分析并重新生成图表
python train.py --analyse experiments/sac_20260512_143025/metrics_final.json --replot new_plot.png
```

每次训练会自动创建独立的实验目录（按算法+时间戳），方便对比不同实验：

```
experiments/
├── sac_20260512_143025/    ← 第一次实验
│   ├── sac_snooker_final.pt
│   ├── metrics_final.json
│   ├── training_metrics_final.png
│   └── ...
├── sac_20260512_180000/    ← 第二次实验
│   └── ...
└── round9_stability/       ← 自定义名称（--run_name）
    └── ...
```

**训练输出示例：**

```
Experiment directory: experiments/sac_20260512_143025
[0:02:15 ETA 4m30s] Ep 100/500 (20%) | R: -3.12 (foul:-4.50 dist:1.38) | Pots: 0.8 | Len: 10 | Break: 1.2 | Fouls: 4.0 | α=0.180 Buf=1024
```

| 字段 | 含义 |
|------|------|
| `0:02:15` | 已用时间 |
| `ETA 4m30s` | 预估剩余时间 |
| `Ep 100/500 (20%)` | 当前进度 |
| `R: -3.12` | 平均总 reward |
| `foul:-4.50` | 平均犯规惩罚（越接近 0 越好） |
| `dist:1.38` | 平均距离奖励（越大越好 = 球离袋口更近） |
| `Pots: 0.8` | 平均每局进球数 |
| `Len: 10` | 平均每局步数 |
| `Break: 1.2` | 平均最大连续得分 |
| `Fouls: 4.0` | 平均每局犯规次数 |
| `α=0.180` | SAC 温度值（仅 SAC） |
| `Buf=1024` | Replay Buffer 大小（仅 SAC） |

**判断学习进展：**
- `foul` 在减小 → agent 在学会避免犯规
- `dist` 在增大 → agent 在学会把球推向袋口
- `Pots` 在增大 → agent 在学会进球

**训练产出文件（每个实验目录）：**

| 文件 | 说明 |
|------|------|
| `sac_snooker_ep100.pt` | 每 100 局的模型 checkpoint |
| `sac_snooker_final.pt` | 训练结束/中断时的最终模型 |
| `metrics_ep100.json` | 每 100 局的完整指标数据 |
| `metrics_final.json` | 最终完整指标（含训练配置 meta） |
| `training_metrics_*.png` | 训练曲线图 |

> 用 `python train.py --analyse experiments/<run_name>/metrics_final.json` 可以随时查看训练结果摘要。

**通用训练参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--algo` | `ppo` | 算法选择：`ppo` 或 `sac` |
| `--num_episodes` | 2000 | 训练总局数 |
| `--render` | 关闭 | 是否渲染画面 |
| `--lr` | 1e-4 | 学习率（SAC actor/critic 共用） |
| `--gamma` | 0.99 | 折扣因子 |
| `--batch_size` | 64/256 | Mini-batch 大小（PPO 64, SAC 256） |
| `--save_dir` | `experiments` | 实验输出根目录 |
| `--run_name` | 自动生成 | 实验名（默认 `{algo}_{timestamp}`） |

**PPO 特有参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--update_interval` | 1024 | PPO 更新间隔（步数） |
| `--k_epochs` | 10 | 每次更新的 SGD epoch 数 |
| `--entropy_coef` | 0.05 | 熵正则化系数 |

**SAC 特有参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--buffer_size` | 100000 | Replay Buffer 容量 |
| `--tau` | 0.002 | Target network 软更新系数（越小越稳定） |
| `--warmup_steps` | 2000 | 随机探索步数（填充 buffer） |
| `--updates_per_step` | 1 | 每个环境步的梯度更新次数 |
| `--init_alpha` | 0.2 | 初始温度值（自动调节） |
| `--lr_alpha` | 1e-4 | 温度参数学习率 |
| `--actor_update_interval` | 2 | Critic 先更新几步再更新 Actor |

### 评估 / 可视化

```bash
# 可视化对局（自动检测 PPO/SAC）
python evaluate.py --load_model experiments/sac_20260512_143025/sac_snooker_final.pt --num_episodes 3 --delay 300

# 指定算法
python evaluate.py --algo ppo --load_model experiments/ppo_20260512_143025/ppo_snooker_final.pt

# 保存评估报告
python evaluate.py --load_model experiments/sac_20260512_143025/sac_snooker_ep500.pt \
    --save_plot eval_plot.png --save_report eval_report.txt
```

### 物理测试

`test_pooltool.py` 基于 [pooltool](https://github.com/ekiefl/pooltool) 物理引擎，提供 15 个可视化测试场景，覆盖旋转物理（高杆/低杆/左塞/右塞）、进球、力度、专业开球策略和速度基准。

```bash
# 全部 15 个测试
python test_pooltool.py

# 旋转测试（高杆/低杆/侧旋/缩杆）
python test_pooltool.py -g spin

# 开球策略（安全开球 + 进攻开球）
python test_pooltool.py -g break

# 角度球
python test_pooltool.py -g angle

# 力度对比
python test_pooltool.py -g power

# 速度基准
python test_pooltool.py -g bench

# 指定编号
python test_pooltool.py -t 12 13

# 用 pooltool 3D 查看器
python test_pooltool.py -t 2 --3d

# 列出所有测试
python test_pooltool.py -l
```

**可用分组：**

| 分组 | 测试编号 | 说明 |
|------|----------|------|
| `straight` | #1-3 | 直球进袋（无旋转/低杆/高杆） |
| `spin` | #2-5, 14 | 旋转效果验证 |
| `sidespin` | #4-5 | 左塞/右塞 |
| `angle` | #6-8, 14 | 角度球 ghost-ball 瞄准 |
| `pot` | #1-3, 6-8, 14 | 所有进球测试 |
| `power` | #9-11 | 轻力/中力/重力 |
| `break` | #12-13 | 开球策略（安全/进攻） |
| `bench` | #15 | 模拟速度基准测试 |

**开球策略说明：**

| # | 策略 | 描述 |
|---|------|------|
| 12 | 安全开球 | 白球从 D 区右侧出发，薄切末排边角红球（cut=30°），白球弹到顶库后沉回底库 |
| 13 | 进攻开球 | V0=6.0 正面大力冲散红球堆（三角尖），散开 10-13 颗红球 |

操作：**SPACE** 下一个测试，**ESC** 退出。窗口支持拖拽调整大小。

## 核心设计

### 动作空间（8 维连续）

模型一次输出 8 个 [-1, 1] 的值，模拟人类决策：

| 维度 | 含义 | 映射 |
|------|------|------|
| `place_x` | D 区母球放置 x | 仅在 ball-in-hand 时生效 |
| `place_y` | D 区母球放置 y | 仅在 ball-in-hand 时生效 |
| `target_idx` | 目标球选择 | [-1,1] → 离散球索引 |
| `pocket_idx` | 目标袋口选择 | [-1,1] → 6 个袋口之一 |
| `angle_offset` | 击球角度偏移 | [-1,1] → ±15°（基于白球→目标球连线） |
| `power` | 击球力度 | [-1,1] → [0.5, 6.0] m/s |
| `b_spin` | 高杆/低杆 | [-1,1] → [-0.8, +0.8]（topspin/backspin） |
| `a_spin` | 左塞/右塞 | [-1,1] → [-0.5, +0.5]（sidespin） |

> Agent 的完整决策链：**选球 → 选袋口 → 瞄准角度 → 力度 → 旋转**，与真实斯诺克选手的思考过程一致。

### 观测空间（54 维）

- 白球位置 (2)
- 15 颗红球位置 (30)
- 6 颗彩球位置 (12)
- 游戏状态 (10)：当前 break、阶段、双方比分、ball-in-hand、犯规计数等

### 奖励系统（统一距离函数）

简洁的奖励设计：**一条主线 + 犯规惩罚**，总共 7 个参数。

> 权威数据源：`environment/pooltool_env.py` 中的 `RewardConfig` dataclass。
> 可通过 `SnookerEnv(reward_cfg=RewardConfig(pot_reward_red=2.0, ...))` 自定义。

#### 主线：递进链条 + 统一距离函数

```
选球合法 → 选定袋口 → 击中目标球 → 球离选定袋口多近 → 进袋 = 最大奖励
                                     ↓
                          reward = max_r × (1 - d/d_max)^1.5
                          d = 目标球到选定袋口的距离（击球后）
                          d = 0（进袋）→ reward = max_r
```

| 结果 | reward | 说明 |
|------|--------|------|
| 红球进袋 | +1.0 | 距离为 0，拿满 `pot_reward_red` |
| 彩球进袋 | 分值 × 0.3 | 蓝球 = 1.5，黑球 = 2.1 |
| 碰到球，推近选定袋口 | ~0.1 - 0.8 | 距离越近奖励越高（连续曲线） |
| 碰到球，但球离选定袋口远 | ~0.01 - 0.1 | 至少打到了，有微小正奖励 |

#### 犯规惩罚

| 事件 | reward | 说明 |
|------|--------|------|
| 选了非法球 | -1.0 | 直接犯规 |
| 白球进袋 | -1.5 | 最严重犯规 |
| 碰到非法球 | -0.8 | first contact 不合法 |
| 没碰到目标球 | -0.3 | 空杆或碰到别的球 |

### 训练模式

**Self-Play（自我对弈）**：两个玩家共享同一个策略网络，轮流出手。环境自动处理换人、犯规判罚、ball-in-hand 等逻辑。

**算法选择：**

| 算法 | 类型 | 核心机制 | 适用场景 |
|------|------|----------|---------|
| **PPO** | On-policy | Rollout Buffer + 比率裁剪 | 基础训练，GPU 资源充足时 |
| **SAC** | Off-policy | Replay Buffer + 最大熵 + 双 Q 网络 | **推荐**，物理模拟较慢时尤其适合 |

**SAC 技术细节：**

- **Replay Buffer**（10 万条经验）— 历史经验复用，样本效率比 PPO 高 ~10×
- **Twin Q-Networks**（双 Q 网络）— 取较小 Q 值，防止过高估计
- **自动温度 α 调节** — 目标熵 = -dim(A) = -8，自动平衡探索与利用
- **Soft Target Update**（Polyak τ=0.002）— 平滑更新目标网络（更慢 = 更稳定）
- **Warmup**（2000 步随机探索）— 先填充 buffer 再开始学习
- **Squashed Gaussian Policy** — 重参数化 + tanh 压缩，输出精确落在 [-1, 1]
- **Reward Normalization** — running mean/std 归一化，稳定 Q 值尺度
- **Delayed Actor Update** — Critic 先更新 2 步再更新 Actor，防止 Actor 追未收敛的 Q
- **Gradient Clipping** — Actor/Critic 梯度范数裁剪到 1.0

**PPO 技术细节：**

- **Rollout Buffer** — 收集 1024 步后更新，数据用完即弃
- **GAE（λ=0.95）** — 广义优势估计，平衡偏差与方差
- **比率裁剪（ε=0.2）** — 隐式重要性采样，限制策略更新幅度
- **熵正则化（0.05）** — 鼓励探索，防止策略过早收敛

## 技术文档

详见 [`docs/technical_details.md`](docs/technical_details.md)。

## 改进日志

详见 [`docs/changelog.md`](docs/changelog.md)。

## 硬件要求

- Python 3.8+
- PyTorch 2.0+
- pooltool 0.6+ (物理引擎)
- pygame 2.x (渲染)
- 建议使用 GPU 加速训练（支持 CPU）
