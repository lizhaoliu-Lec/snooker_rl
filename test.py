"""
Test script to verify the Snooker RL environment works correctly
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pygame

def test_environment():
    """Test basic environment functionality"""
    print("Testing Snooker Environment...")
    
    from environment.snooker_env import SnookerEnv
    
    env = SnookerEnv(render_mode=None)
    
    print(f"Action space: {env.action_space}")
    print(f"Observation space shape: {env.observation_space.shape}")
    
    state, info = env.reset()
    print(f"Initial state shape: {state.shape}")
    print(f"Initial info: {info}")
    
    for i in range(3):
        action = env.action_space.sample()
        next_state, reward, done, truncated, info = env.step(action)
        print(f"Step {i+1}: reward={reward:.2f}, done={done}, break={info.get('break', 0)}")
        
        if done or truncated:
            state, info = env.reset()
            
    env.close()
    print("Environment test passed!")


def test_ppo_agent():
    """Test PPO agent initialization"""
    print("\nTesting PPO Agent...")
    
    from algorithms.ppo import PPO
    
    state_dim = 50
    action_dim = 2
    
    ppo = PPO(state_dim, action_dim)
    
    dummy_state = np.random.randn(state_dim)
    action, log_prob, value = ppo.select_action(dummy_state)
    
    print(f"Action shape: {action.shape}")
    print(f"Log prob: {log_prob:.4f}")
    print(f"Value: {value:.4f}")
    
    print("PPO agent test passed!")


def test_utils():
    """Test utility functions"""
    print("\nTesting Utilities...")
    
    from utils import set_seed, count_parameters, get_device_info
    
    set_seed(42)
    
    info = get_device_info()
    print(f"Device info: {info}")
    
    import torch.nn as nn
    
    class DummyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(100, 50)
            self.fc2 = nn.Linear(50, 10)
            
    net = DummyNet()
    params = count_parameters(net)
    print(f"Network parameters: {params}")
    
    print("Utils test passed!")


def main():
    print("=" * 60)
    print("Snooker RL Test Suite")
    print("=" * 60)
    
    try:
        test_environment()
        test_ppo_agent()
        test_utils()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
        print("\nYou can now train the agent using:")
        print("  python train.py --num_episodes 100 --render")
        print("\nOr evaluate using:")
        print("  python evaluate.py --load_model saved_models/ppo_snooker_final.pt")
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    return 0


if __name__ == '__main__':
    exit(main())
