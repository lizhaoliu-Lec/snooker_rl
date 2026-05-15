"""
Snooker RL Inference / Evaluation Script
Evaluates and visualizes trained agent (PPO or SAC) in two-player self-play.
"""

import os
import sys
import argparse
import numpy as np
import torch
import pygame
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.pooltool_env import SnookerEnv


def _detect_algo(filepath):
    """Auto-detect algorithm from checkpoint file."""
    if os.path.exists(filepath):
        ckpt = torch.load(filepath, map_location='cpu')
        if ckpt.get('algorithm') == 'SAC' or 'actor_state_dict' in ckpt:
            return 'sac'
    return 'ppo'


def _make_agent(algo, state_dim, action_dim):
    """Create agent by algorithm name."""
    if algo == 'sac':
        from algorithms.sac import SAC
        return SAC(state_dim, action_dim)
    else:
        from algorithms.ppo import PPO
        return PPO(state_dim, action_dim)


class InferenceVisualizer:
    def __init__(self, agent, env):
        self.agent = agent
        self.env = env
        self.episode_history = []

    def run_episode(self, render=True, delay_ms=500, record=True):
        """Run a single episode with the trained agent (self-play)."""
        state, _ = self.env.reset()
        episode_reward = 0
        episode_length = 0
        episode_data = {
            'states': [],
            'actions': [],
            'rewards': [],
            'breaks': [],
            'pocketed': [],
            'players': [],
        }

        print("\n" + "=" * 60)
        print("Starting Episode (Self-Play)")
        print("=" * 60)

        while True:
            if render:
                self.env.render()
                self._handle_events()

            action, log_prob, value = self.agent.select_action(state)

            if record:
                episode_data['states'].append(state.copy())
                episode_data['actions'].append(action.copy())

            # step() internally renders aiming line + ball motion in human mode
            next_state, reward, done, truncated, info = self.env.step(action)

            if record:
                episode_data['rewards'].append(reward)
                episode_data['breaks'].append(info.get('break', 0))
                episode_data['pocketed'].append(info.get('pocketed', []))
                episode_data['players'].append(info.get('player', 0))

            episode_reward += reward
            episode_length += 1

            # Print per-shot info to terminal
            pocketed_str = ', '.join(info.get('pocketed', [])) or 'none'
            pocket_str = info.get('chosen_pocket', '?')
            print(f"  Shot {episode_length}: P{info.get('player',0)+1} | "
                  f"reward={reward:+.2f} | pocket={pocket_str} | "
                  f"pocketed=[{pocketed_str}] | "
                  f"phase={info.get('phase','?')}")

            if render:
                # Short pause between shots so the viewer can follow
                self.env.render()
                pygame.time.wait(delay_ms)

            if done or truncated:
                break

            state = next_state

        if record:
            self.episode_history.append(episode_data)

        print("\n" + "-" * 60)
        print(f"Episode Summary:")
        print(f"  Total Reward: {episode_reward:.2f}")
        print(f"  Episode Length: {episode_length} steps")
        print(f"  P1 Score: {info.get('score_p1', 0)} "
              f"(Pot: {info.get('pot_p1', 0)}, FoulRcv: {info.get('foul_rcv_p1', 0)})")
        print(f"  P2 Score: {info.get('score_p2', 0)} "
              f"(Pot: {info.get('pot_p2', 0)}, FoulRcv: {info.get('foul_rcv_p2', 0)})")
        print(f"  Phase: {info.get('phase', '?')}")
        print("-" * 60)

        return episode_reward, episode_length, info

    def _handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return True
                elif event.key == pygame.K_SPACE:
                    pygame.event.post(pygame.event.Event(pygame.KEYUP))
        return False

    def _draw_action_info(self, action, info):
        """Draw action information overlay on screen"""
        if self.env.screen is None:
            return

        place_x, place_y, target_raw, pocket_raw = action[:4]
        angle_off = action[4] if len(action) > 4 else 0.0
        power = action[5] if len(action) > 5 else 0.0
        b_spin = action[6] if len(action) > 6 else 0.0
        a_spin = action[7] if len(action) > 7 else 0.0

        font = pygame.font.Font(None, 24)
        text_y = 100

        lines = [
            f"Player: {info.get('player', 0) + 1}  Phase: {info.get('phase', '?')}",
            f"Target raw: {target_raw:.2f}  Angle offset: {angle_off:.2f}  Power: {power:.2f}",
            f"Spin: b={b_spin:+.2f} a={a_spin:+.2f}  Break: {info.get('break', 0)}  Pocketed: {info.get('pocketed', [])}",
        ]

        for i, txt in enumerate(lines):
            surf = font.render(txt, True, (255, 255, 255))
            self.env.screen.blit(surf, (10, text_y + i * 22))

        pygame.display.flip()

    def plot_episode_history(self, save_path=None):
        """Plot episode history"""
        if not self.episode_history:
            print("No episode history to plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        rewards = [sum(ep['rewards']) for ep in self.episode_history]
        lengths = [len(ep['rewards']) for ep in self.episode_history]
        max_breaks = [max(ep['breaks']) if ep['breaks'] else 0
                      for ep in self.episode_history]

        episodes = range(1, len(self.episode_history) + 1)

        axes[0, 0].plot(episodes, rewards, 'b-', linewidth=2)
        axes[0, 0].set_title('Episode Rewards', fontsize=14)
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(episodes, lengths, 'g-', linewidth=2)
        axes[0, 1].set_title('Episode Lengths', fontsize=14)
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Steps')
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].plot(episodes, max_breaks, 'r-', linewidth=2)
        axes[1, 0].set_title('Max Break per Episode', fontsize=14)
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Break Score')
        axes[1, 0].grid(True, alpha=0.3)

        all_rewards = []
        for ep in self.episode_history:
            all_rewards.extend(ep['rewards'])
        axes[1, 1].hist(all_rewards, bins=50, alpha=0.7, edgecolor='black')
        axes[1, 1].set_title('Reward Distribution', fontsize=14)
        axes[1, 1].set_xlabel('Reward')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        plt.close()

    def generate_report(self, save_path='evaluation_report.txt'):
        """Generate a text report of evaluation results"""
        if not self.episode_history:
            print("No evaluation data available")
            return

        total_rewards = [sum(ep['rewards']) for ep in self.episode_history]
        total_steps = [len(ep['rewards']) for ep in self.episode_history]
        max_breaks = [max(ep['breaks']) if ep['breaks'] else 0
                      for ep in self.episode_history]

        with open(save_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("SNOOKER RL EVALUATION REPORT (Self-Play)\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Episodes: {len(self.episode_history)}\n\n")

            f.write("-" * 60 + "\n")
            f.write("REWARD STATISTICS\n")
            f.write("-" * 60 + "\n")
            f.write(f"Mean Reward: {np.mean(total_rewards):.2f}\n")
            f.write(f"Std Reward: {np.std(total_rewards):.2f}\n")
            f.write(f"Min Reward: {np.min(total_rewards):.2f}\n")
            f.write(f"Max Reward: {np.max(total_rewards):.2f}\n\n")

            f.write("-" * 60 + "\n")
            f.write("EPISODE LENGTH STATISTICS\n")
            f.write("-" * 60 + "\n")
            f.write(f"Mean Length: {np.mean(total_steps):.1f} steps\n")
            f.write(f"Std Length: {np.std(total_steps):.1f} steps\n")
            f.write(f"Min Length: {np.min(total_steps)} steps\n")
            f.write(f"Max Length: {np.max(total_steps)} steps\n\n")

            f.write("-" * 60 + "\n")
            f.write("BREAK STATISTICS\n")
            f.write("-" * 60 + "\n")
            f.write(f"Mean Max Break: {np.mean(max_breaks):.1f}\n")
            f.write(f"Max Break: {np.max(max_breaks)}\n")

        print(f"Report saved to {save_path}")


def evaluate(args):
    print("=" * 60)
    print("Snooker RL – Self-Play Evaluation")
    print("=" * 60)

    env = SnookerEnv(render_mode='human')

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    print(f"State dim: {state_dim}, Action dim: {action_dim}")

    # Detect or use specified algorithm
    algo = args.algo
    if algo == 'auto' and args.load_model:
        algo = _detect_algo(args.load_model)
        print(f"Auto-detected algorithm: {algo.upper()}")
    elif algo == 'auto':
        algo = 'ppo'
    print(f"Using algorithm: {algo.upper()}")

    agent = _make_agent(algo, state_dim, action_dim)

    # For SAC, use deterministic actions during evaluation
    _orig_select = agent.select_action
    if algo == 'sac':
        agent.select_action = lambda s: _orig_select(s, deterministic=True)

    if args.load_model:
        if agent.load(args.load_model):
            print(f"Successfully loaded model from {args.load_model}")
        else:
            print(f"Could not load model from {args.load_model}")
            print("Using randomly initialized agent")
    else:
        print("No model specified, using randomly initialized agent")

    visualizer = InferenceVisualizer(agent, env)

    try:
        for episode in range(1, args.num_episodes + 1):
            reward, length, info = visualizer.run_episode(
                render=True,
                delay_ms=args.delay,
                record=True
            )

            if episode % args.log_interval == 0:
                print(f"\nProgress: {episode}/{args.num_episodes} episodes completed")

    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")

    finally:
        env.close()

        if args.save_plot:
            visualizer.plot_episode_history(args.save_plot)

        if args.save_report:
            visualizer.generate_report(args.save_report)

        if visualizer.episode_history:
            total_rewards = [sum(ep['rewards']) for ep in visualizer.episode_history]
            print("\n" + "=" * 60)
            print("EVALUATION COMPLETE")
            print("=" * 60)
            print(f"Episodes Run: {len(visualizer.episode_history)}")
            print(f"Mean Reward: {np.mean(total_rewards):.2f}")
            print(f"Std Reward: {np.std(total_rewards):.2f}")
            print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Evaluate RL agent for Snooker (Self-Play)')

    parser.add_argument('--algo', type=str, default='auto',
                        choices=['auto', 'ppo', 'sac'],
                        help='Algorithm (auto-detect from checkpoint by default)')
    parser.add_argument('--load_model', type=str, required=True,
                        help='Path to trained model')
    parser.add_argument('--num_episodes', type=int, default=5,
                        help='Number of evaluation episodes')
    parser.add_argument('--delay', type=int, default=300,
                        help='Delay between steps (ms)')
    parser.add_argument('--log_interval', type=int, default=5,
                        help='Episodes between progress logs')
    parser.add_argument('--save_plot', type=str, default=None,
                        help='Path to save evaluation plot')
    parser.add_argument('--save_report', type=str, default=None,
                        help='Path to save evaluation report')

    args = parser.parse_args()

    evaluate(args)


if __name__ == '__main__':
    main()
