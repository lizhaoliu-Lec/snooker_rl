"""
Configuration for Snooker RL Training (Self-Play)
"""

TRAINING_CONFIG = {
    'algorithm': 'PPO',  # or 'SAC'
    'mode': 'self-play',
    'environment': {
        'table_width': 1200,
        'table_height': 600,
        'ball_radius': 12,
        'pocket_radius': 22,
        'cushion_width': 25,
        'friction': 0.02,
        'restitution': 0.9,
        'max_shots_without_pocket': 20,   # 连续不进球终止（从30缩短到20，逼迫进攻）
        'max_consecutive_fouls': 5,        # 连续犯规终止（训练初期放宽）
    },
    'action_space': {
        'description': '8-dim continuous [-1, 1]',
        'dims': [
            'place_x      – D-zone cue ball x (only when ball_in_hand)',
            'place_y      – D-zone cue ball y (only when ball_in_hand)',
            'target_idx   – mapped to discrete ball index via all_targetable list',
            'pocket_idx   – mapped to one of 6 pockets (target pocket for reward)',
            'angle_offset – deviation from white→target axis (±15°)',
            'power        – shot speed V0 [0.5, 6.0] m/s',
            'b_spin       – topspin(+) / backspin(-) [-0.8, +0.8]',
            'a_spin       – sidespin left(-) / right(+) [-0.5, +0.5]',
        ],
    },
    'observation_space': {
        'description': '54-dim normalised [-1, 1]',
        'layout': [
            'white ball (2)',
            '15 reds × 2 (30)',
            '6 colours × 2 (12)',
            'game state (10): break/147, phase, next_color_idx/6, '
            'remaining/21, ball_in_hand, current_player, '
            'score_p1/147, score_p2/147, consec_fouls/5, shots_no_pot/30',
        ],
    },
    'ppo': {
        'gamma': 0.99,
        'lam': 0.95,
        'lr': 3e-4,
        'eps_clip': 0.2,
        'k_epochs': 10,
        'entropy_coef': 0.05,
        'value_coef': 0.5,
        'batch_size': 64,
        'hidden_dim': 256,
        'max_grad_norm': 0.5,
    },
    'sac': {
        'gamma': 0.99,
        'tau': 0.002,              # 更慢的 target 更新（稳定性）
        'lr_actor': 1e-4,          # 降低学习率（防止 Q 值爆炸）
        'lr_critic': 1e-4,
        'lr_alpha': 1e-4,
        'batch_size': 256,
        'hidden_dim': 256,
        'buffer_size': 100_000,    # 10万足够（2000局约产出6万样本）
        'warmup_steps': 2000,      # 先积累再学习（比 1000 更稳定）
        'updates_per_step': 1,
        'target_update_interval': 1,
        'init_alpha': 0.2,
        'actor_update_interval': 2,   # 延迟 actor 更新（critic 先学 2 步）
        'normalize_rewards': True,    # reward 归一化（稳定 Q 值尺度）
    },
    'training': {
        'num_episodes': 2000,
        'update_interval': 1024,  # PPO only
        'log_interval': 10,
        'save_interval': 100,
        'plot_interval': 100,
    },
    # ═════════════════════════════════════════════════════════════
    # 奖励系统（Round 14 — Behavior + Outcome 双轨制）
    # ═════════════════════════════════════════════════════════════
    #
    # 权威数据源: environment/pooltool_env.py 中的 RewardConfig dataclass
    #
    # 【Behavior（过程）】
    #   合法 + 进球 → pot_reward + break_bonus × current_break
    #   合法 + 没进 → miss_penalty（小负值，消除安全港）
    #   犯规         → foul_penalty
    #   白球进袋     → white_pocket_penalty
    #
    # 【Outcome（终局）】
    #   赢了 → +win_reward
    #   输了 → +lose_reward
    #   平局 → 0
    #
    # ── 数值排序 ──────────────────────────────────────────────
    #   赢 30 >> break=8 进球 36 >> 首次进球 20 >> miss -0.1
    #   >> 犯规 -1 >> 白进 -2 >> 输 -30
    # ═════════════════════════════════════════════════════════════
    'rewards': {
        'description': 'Behavior + Outcome 双轨制',

        # ── Behavior: 进球（翻倍鼓励冒险）─────────────────────
        'pot_reward': 20.0,              # 合法进球奖励（10→20）
        'break_bonus': 2.0,             # 每 1 点 break 额外奖励

        # ── Behavior: 犯规 ────────────────────────────────────
        'foul_penalty': -1.0,            # 犯规 -1
        'white_pocket_penalty': -2.0,    # 白球进袋 -2

        # ── Behavior: 合法没进 ────────────────────────────────
        'miss_penalty': -0.1,            # 合法碰球没进（消除安全港）

        # ── Outcome: 终局胜负 ─────────────────────────────────
        'win_reward': 30.0,              # 赢了 +30
        'lose_reward': -30.0,            # 输了 -30
    },
}

EVALUATION_CONFIG = {
    'num_episodes': 5,
    'render': True,
    'delay_ms': 300,
    'save_trajectories': True,
    'compute_statistics': True,
}
