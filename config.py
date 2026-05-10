"""
Configuration for Snooker RL Training
"""

TRAINING_CONFIG = {
    'algorithm': 'PPO',
    'environment': {
        'table_width': 1200,
        'table_height': 600,
        'ball_radius': 12,
        'pocket_radius': 22,
        'cushion_width': 25,
        'friction': 0.02,
        'restitution': 0.9,
        'max_shots_without_pocket': 3,
    },
    'ppo': {
        'gamma': 0.99,
        'lam': 0.95,
        'lr': 3e-4,
        'eps_clip': 0.2,
        'k_epochs': 10,
        'entropy_coef': 0.01,
        'value_coef': 0.5,
        'batch_size': 64,
        'hidden_dim': 256,
        'max_grad_norm': 0.5,
    },
    'training': {
        'num_episodes': 1000,
        'update_interval': 2048,
        'log_interval': 10,
        'save_interval': 100,
        'plot_interval': 100,
    },
    'rewards': {
        'red_ball': 2,
        'yellow_ball': 2,
        'green_ball': 3,
        'brown_ball': 4,
        'blue_ball': 5,
        'pink_ball': 6,
        'black_ball': 7,
        'foul_penalty': -15,
        'white_pocketed_penalty': -10,
        'step_penalty': -0.1,
        'all_cleared_bonus': 100,
    },
}

EVALUATION_CONFIG = {
    'num_episodes': 10,
    'render': True,
    'delay_ms': 500,
    'save_trajectories': True,
    'compute_statistics': True,
}
