"""
Snooker RL Environment Package
"""

# Pooltool-based environment (recommended — full spin physics)
from environment.pooltool_env import SnookerEnv

# Legacy pymunk-based environment (no spin support)
from environment.snooker_env import SnookerEnv as SnookerEnvLegacy

__all__ = ['SnookerEnv', 'SnookerEnvLegacy']
