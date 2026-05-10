# Snooker RL

基于强化学习的斯诺克游戏AI，使用PPO算法训练。

## 项目结构

```
snooker_rl/
├── environment/          # 环境定义
│   ├── snooker_env.py   # 斯诺克环境实现
│   └── __init__.py
├── algorithms/           # 强化学习算法
│   └── ppo.py           # PPO算法实现
├── utils/               # 工具函数
│   └── __init__.py
├── train.py             # 训练脚本
├── evaluate.py          # 推理脚本
├── test.py             # 测试脚本
├── config.py           # 配置文件
└── requirements.txt    # 依赖列表
```

## 安装

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 测试环境

```bash
python test.py
```

### 2. 训练模型

```bash
python train.py --num_episodes 1000 --render
```

训练参数:
- `--num_episodes`: 训练回合数
- `--render`: 是否渲染游戏画面
- `--lr`: 学习率 (默认 3e-4)
- `--gamma`: 折扣因子 (默认 0.99)
- `--save_dir`: 模型保存目录

### 3. 评估模型

```bash
python evaluate.py --load_model saved_models/ppo_snooker_final.pt
```

评估参数:
- `--load_model`: 加载模型路径
- `--num_episodes`: 评估回合数
- `--delay`: 每步延迟(毫秒)

## 环境说明

### 动作空间
- 角度: [0, 2π] - 球杆击球角度
- 力度: [0, 1] - 击球力度

### 状态空间
- 白球位置和速度
- 红球位置 (15个)
- 彩球位置 (6个)
- 当前Break和犯规状态

### 奖励设计
- 进球: 2-7分 (根据球的颜色)
- 犯规: -15分
- 清台: +100分
- 白球落袋: -10分

## 算法

使用 PPO (Proximal Policy Optimization) 算法:

- Actor-Critic 架构
- GAE (Generalized Advantage Estimation)
- 剪切代理目标
- 熵正则化

## 硬件要求

- Python 3.8+
- PyTorch 2.0+
- 建议使用GPU进行训练
