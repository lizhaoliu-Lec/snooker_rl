"""
Snooker RL Training Script
Trains a PPO agent on the Snooker environment
"""

import os
import sys
import argparse
import numpy as np
import torch
from datetime import datetime
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.snooker_env import SnookerEnv
from algorithms.ppo import PPO


class TrainingMetrics:
    def __init__(self):
        self.episode_rewards = []
        self.episode_lengths = []
        self.policy_losses = []
        self.value_losses = []
        self.break_scores = []
        self.foul_counts = []
        
    def add(self, reward, length, policy_loss=None, value_loss=None, 
            break_score=0, fouls=0):
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        if policy_loss is not None:
            self.policy_losses.append(policy_loss)
        if value_loss is not None:
            self.value_losses.append(value_loss)
        self.break_scores.append(break_score)
        self.foul_counts.append(fouls)
        
    def get_average(self, n=10):
        if len(self.episode_rewards) < n:
            return {
                'reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
                'length': np.mean(self.episode_lengths) if self.episode_lengths else 0,
                'break': np.mean(self.break_scores) if self.break_scores else 0,
                'fouls': np.mean(self.foul_counts) if self.foul_counts else 0
            }
        return {
            'reward': np.mean(self.episode_rewards[-n:]),
            'length': np.mean(self.episode_lengths[-n:]),
            'policy_loss': np.mean(self.policy_losses[-n:]) if self.policy_losses else 0,
            'value_loss': np.mean(self.value_losses[-n:]) if self.value_losses else 0,
            'break': np.mean(self.break_scores[-n:]),
            'fouls': np.mean(self.foul_counts[-n:])
        }
    
    def plot(self, save_path=None):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        episodes = range(1, len(self.episode_rewards) + 1)
        
        axes[0, 0].plot(episodes, self.episode_rewards)
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True)
        
        axes[0, 1].plot(episodes, self.episode_lengths)
        axes[0, 1].set_title('Episode Lengths')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Steps')
        axes[0, 1].grid(True)
        
        if self.policy_losses:
            axes[1, 0].plot(range(1, len(self.policy_losses) + 1), self.policy_losses)
            axes[1, 0].set_title('Policy Loss')
            axes[1, 0].set_xlabel('Update')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].grid(True)
            
        axes[1, 1].plot(episodes, self.break_scores)
        axes[1, 1].set_title('Break Scores')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.close()


def train(args):
    print("=" * 60)
    print("Snooker RL Training")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    env = SnookerEnv(render_mode='human' if args.render else None)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    
    ppo_cfg = {
        'gamma': args.gamma,
        'lam': args.lam,
        'lr': args.lr,
        'eps_clip': args.eps_clip,
        'k_epochs': args.k_epochs,
        'entropy_coef': args.entropy_coef,
        'value_coef': args.value_coef,
        'batch_size': args.batch_size,
        'hidden_dim': args.hidden_dim
    }
    
    agent = PPO(state_dim, action_dim, ppo_cfg)
    
    if args.load_model:
        if agent.load(args.load_model):
            print(f"Loaded model from {args.load_model}")
        else:
            print(f"Could not load model from {args.load_model}")
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    metrics = TrainingMetrics()
    
    print("\nStarting training...")
    print("-" * 60)
    
    timestep = 0
    update_interval = args.update_interval
    
    try:
        for episode in range(1, args.num_episodes + 1):
            state, _ = env.reset()
            episode_reward = 0
            episode_length = 0
            
            while True:
                action, log_prob, value = agent.select_action(state)
                
                next_state, reward, done, truncated, info = env.step(action)
                
                agent.memory.add(state, action, log_prob, value, reward, done)
                
                episode_reward += reward
                episode_length += 1
                timestep += 1
                
                if args.render:
                    env.render()
                
                if timestep % update_interval == 0:
                    agent.update()
                    
                    stats = agent.get_stats()
                    if stats['policy_loss']:
                        policy_loss = stats['policy_loss'][-1]
                        value_loss = stats['value_loss'][-1]
                    else:
                        policy_loss = None
                        value_loss = None
                        
                    metrics.add(
                        episode_reward, episode_length,
                        policy_loss, value_loss,
                        info.get('break', 0), info.get('foul', 0)
                    )
                    
                if done or truncated:
                    break
                    
                state = next_state
                
            if episode % args.log_interval == 0:
                avg_metrics = metrics.get_average(args.log_interval)
                print(f"Episode {episode}/{args.num_episodes} | "
                      f"Avg Reward: {avg_metrics['reward']:.2f} | "
                      f"Avg Length: {avg_metrics['length']:.1f} | "
                      f"Avg Break: {avg_metrics['break']:.1f} | "
                      f"Avg Fouls: {avg_metrics['fouls']:.1f}")
                
            if episode % args.save_interval == 0:
                save_path = os.path.join(args.save_dir, f'ppo_snooker_ep{episode}.pt')
                agent.save(save_path)
                print(f"Model saved to {save_path}")
                
            if episode % args.plot_interval == 0:
                plot_path = os.path.join(args.save_dir, f'training_metrics_ep{episode}.png')
                metrics.plot(plot_path)
                
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        
    finally:
        env.close()
        
        final_save_path = os.path.join(args.save_dir, 'ppo_snooker_final.pt')
        agent.save(final_save_path)
        print(f"\nFinal model saved to {final_save_path}")
        
        final_plot_path = os.path.join(args.save_dir, 'training_metrics_final.png')
        metrics.plot(final_plot_path)
        print(f"Final metrics plot saved to {final_plot_path}")
        
    print("\nTraining completed!")


def main():
    parser = argparse.ArgumentParser(description='Train PPO agent for Snooker')
    
    parser.add_argument('--num_episodes', type=int, default=1000,
                       help='Number of training episodes')
    parser.add_argument('--render', action='store_true',
                       help='Render environment during training')
    parser.add_argument('--load_model', type=str, default=None,
                       help='Path to model to load')
    parser.add_argument('--save_dir', type=str, default='saved_models',
                       help='Directory to save models')
    
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor')
    parser.add_argument('--lam', type=float, default=0.95,
                       help='GAE lambda')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--eps_clip', type=float, default=0.2,
                       help='PPO epsilon clipping')
    parser.add_argument('--k_epochs', type=int, default=10,
                       help='Number of epochs per update')
    parser.add_argument('--entropy_coef', type=float, default=0.01,
                       help='Entropy coefficient')
    parser.add_argument('--value_coef', type=float, default=0.5,
                       help='Value loss coefficient')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for training')
    parser.add_argument('--hidden_dim', type=int, default=256,
                       help='Hidden layer dimension')
    
    parser.add_argument('--update_interval', type=int, default=2048,
                       help='Steps between updates')
    parser.add_argument('--log_interval', type=int, default=10,
                       help='Episodes between logging')
    parser.add_argument('--save_interval', type=int, default=100,
                       help='Episodes between saving')
    parser.add_argument('--plot_interval', type=int, default=100,
                       help='Episodes between plotting')
    
    args = parser.parse_args()
    
    train(args)


if __name__ == '__main__':
    main()
