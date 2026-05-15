"""
Snooker RL Training Script – Self-Play (PPO / SAC)

Both players share a single policy network.  The training loop alternates
between the two players, collecting transitions from each player's
perspective and feeding them all into the same update buffer.

Usage:
  python train.py --algo ppo --num_episodes 2000
  python train.py --algo sac --num_episodes 2000
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
from datetime import datetime
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.pooltool_env import SnookerEnv


def make_agent(algo, state_dim, action_dim, args):
    """Create the appropriate agent based on --algo flag."""
    if algo == 'ppo':
        from algorithms.ppo import PPO
        cfg = {
            'gamma': args.gamma,
            'lam': args.lam,
            'lr': args.lr,
            'eps_clip': args.eps_clip,
            'k_epochs': args.k_epochs,
            'entropy_coef': args.entropy_coef,
            'value_coef': args.value_coef,
            'batch_size': args.batch_size,
            'hidden_dim': args.hidden_dim,
        }
        return PPO(state_dim, action_dim, cfg)

    elif algo == 'sac':
        from algorithms.sac import SAC
        cfg = {
            'gamma': args.gamma,
            'tau': args.tau,
            'lr_actor': args.lr,
            'lr_critic': args.lr,
            'lr_alpha': args.lr_alpha,
            'batch_size': args.batch_size,
            'hidden_dim': args.hidden_dim,
            'buffer_size': args.buffer_size,
            'warmup_steps': args.warmup_steps,
            'updates_per_step': args.updates_per_step,
            'target_update_interval': args.target_update_interval,
            'init_alpha': args.init_alpha,
            'actor_update_interval': args.actor_update_interval,
            'normalize_rewards': True,
        }
        return SAC(state_dim, action_dim, cfg)

    else:
        raise ValueError(f"Unknown algorithm: {algo}. Use 'ppo' or 'sac'.")


class TrainingMetrics:
    def __init__(self):
        self.episode_rewards = []
        self.episode_lengths = []
        self.policy_losses = []
        self.value_losses = []
        self.break_scores = []
        self.foul_counts = []
        self.score_p1 = []
        self.score_p2 = []
        self.pot_p1 = []
        self.pot_p2 = []
        self.foul_rcv_p1 = []
        self.foul_rcv_p2 = []
        # Reward breakdown (对应 reward_breakdown 中的三个字段)
        self.foul_rewards = []      # 累计 bd['foul'] per episode（犯规惩罚）
        self.distance_rewards = []  # 累计 bd['distance'] per episode（进球奖励 or miss惩罚）
        self.pot_counts = []        # 进球数 per episode
        self.win_loss_rewards = []  # 累计 bd['win_loss'] per episode（终局胜负 +30/-30/0）
        self.approach_rewards = []       # 累计 bd['approach'] per episode（方向引导奖励）
        # ── 犯规类型细分 ──────────────────────────────────────
        self.miss_ball_counts = []     # 空杆次数 per episode
        self.wrong_ball_counts = []    # 碰错球次数 per episode
        self.white_pocket_counts = []  # 白球进袋次数 per episode
        self.illegal_choice_counts = [] # 选非法球次数 per episode（action masking 后应为0）
        # ── 精度指标 ──────────────────────────────────────────
        self.angle_offsets = []        # 平均|offset_deg| per episode（越小越精准）
        self.power_values = []         # 平均 V0 per episode（击球力度）
        # ── 进球质量 ──────────────────────────────────────────
        self.intentional_pots = []     # 精确进球次数 per episode
        self.lucky_pots = []           # 运气进球次数 per episode

    def add(self, reward, length, policy_loss=None, value_loss=None,
            break_score=0, fouls=0, s_p1=0, s_p2=0,
            pot_p1=0, pot_p2=0, foul_rcv_p1=0, foul_rcv_p2=0,
            foul_reward=0.0, distance_reward=0.0, pot_count=0,
            win_loss_reward=0.0, approach_reward=0.0,
            miss_ball=0, wrong_ball=0, white_pocket=0, illegal_choice=0,
            angle_offset=0.0, power=0.0, intentional_pots=0, lucky_pots=0):
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        if policy_loss is not None:
            self.policy_losses.append(policy_loss)
        if value_loss is not None:
            self.value_losses.append(value_loss)
        self.break_scores.append(break_score)
        self.foul_counts.append(fouls)
        self.score_p1.append(s_p1)
        self.score_p2.append(s_p2)
        self.pot_p1.append(pot_p1)
        self.pot_p2.append(pot_p2)
        self.foul_rcv_p1.append(foul_rcv_p1)
        self.foul_rcv_p2.append(foul_rcv_p2)
        self.foul_rewards.append(foul_reward)
        self.distance_rewards.append(distance_reward)
        self.pot_counts.append(pot_count)
        self.win_loss_rewards.append(win_loss_reward)
        self.approach_rewards.append(approach_reward)
        self.miss_ball_counts.append(miss_ball)
        self.wrong_ball_counts.append(wrong_ball)
        self.white_pocket_counts.append(white_pocket)
        self.illegal_choice_counts.append(illegal_choice)
        self.angle_offsets.append(angle_offset)
        self.power_values.append(power)
        self.intentional_pots.append(intentional_pots)
        self.lucky_pots.append(lucky_pots)

    def get_average(self, n=10):
        if len(self.episode_rewards) < n:
            n = max(len(self.episode_rewards), 1)
        return {
            'reward': np.mean(self.episode_rewards[-n:]),
            'length': np.mean(self.episode_lengths[-n:]),
            'policy_loss': np.mean(self.policy_losses[-n:]) if self.policy_losses else 0,
            'value_loss': np.mean(self.value_losses[-n:]) if self.value_losses else 0,
            'break': np.mean(self.break_scores[-n:]),
            'fouls': np.mean(self.foul_counts[-n:]),
            'score_p1': np.mean(self.score_p1[-n:]),
            'score_p2': np.mean(self.score_p2[-n:]),
            'pot_p1': np.mean(self.pot_p1[-n:]),
            'pot_p2': np.mean(self.pot_p2[-n:]),
            'foul_rcv_p1': np.mean(self.foul_rcv_p1[-n:]),
            'foul_rcv_p2': np.mean(self.foul_rcv_p2[-n:]),
            'foul_reward': np.mean(self.foul_rewards[-n:]),
            'distance_reward': np.mean(self.distance_rewards[-n:]),
            'pot_count': np.mean(self.pot_counts[-n:]),
            'win_loss_reward': np.mean(self.win_loss_rewards[-n:]) if self.win_loss_rewards else 0,
            'approach_reward': np.mean(self.approach_rewards[-n:]) if self.approach_rewards else 0,
            'miss_ball': np.mean(self.miss_ball_counts[-n:]) if self.miss_ball_counts else 0,
            'wrong_ball': np.mean(self.wrong_ball_counts[-n:]) if self.wrong_ball_counts else 0,
            'white_pocket': np.mean(self.white_pocket_counts[-n:]) if self.white_pocket_counts else 0,
            'illegal_choice': np.mean(self.illegal_choice_counts[-n:]) if self.illegal_choice_counts else 0,
            'angle_offset': np.mean(self.angle_offsets[-n:]) if self.angle_offsets else 0,
            'power': np.mean(self.power_values[-n:]) if self.power_values else 0,
            'intentional_pots': np.mean(self.intentional_pots[-n:]) if self.intentional_pots else 0,
            'lucky_pots': np.mean(self.lucky_pots[-n:]) if self.lucky_pots else 0,
        }

    def save(self, path, meta=None):
        """Save all metrics to a JSON file for later analysis.
        
        Args:
            path: output JSON file path
            meta: optional dict with training config (algo, hyperparams, etc.)
        """
        data = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'policy_losses': self.policy_losses,
            'value_losses': self.value_losses,
            'break_scores': self.break_scores,
            'foul_counts': self.foul_counts,
            'score_p1': self.score_p1,
            'score_p2': self.score_p2,
            'pot_p1': self.pot_p1,
            'pot_p2': self.pot_p2,
            'foul_rcv_p1': self.foul_rcv_p1,
            'foul_rcv_p2': self.foul_rcv_p2,
            'foul_rewards': self.foul_rewards,
            'distance_rewards': self.distance_rewards,
            'pot_counts': self.pot_counts,
            'win_loss_rewards': self.win_loss_rewards,
            'approach_rewards': self.approach_rewards,
            'miss_ball_counts': self.miss_ball_counts,
            'wrong_ball_counts': self.wrong_ball_counts,
            'white_pocket_counts': self.white_pocket_counts,
            'illegal_choice_counts': self.illegal_choice_counts,
            'angle_offsets': self.angle_offsets,
            'power_values': self.power_values,
            'intentional_pots': self.intentional_pots,
            'lucky_pots': self.lucky_pots,
            'total_episodes': len(self.episode_rewards),
        }
        if meta:
            data['meta'] = meta
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path):
        """Load metrics from a JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        m = cls()
        m.episode_rewards = data.get('episode_rewards', [])
        m.episode_lengths = data.get('episode_lengths', [])
        m.policy_losses = data.get('policy_losses', [])
        m.value_losses = data.get('value_losses', [])
        m.break_scores = data.get('break_scores', [])
        m.foul_counts = data.get('foul_counts', [])
        m.score_p1 = data.get('score_p1', [])
        m.score_p2 = data.get('score_p2', [])
        m.pot_p1 = data.get('pot_p1', [])
        m.pot_p2 = data.get('pot_p2', [])
        m.foul_rcv_p1 = data.get('foul_rcv_p1', [])
        m.foul_rcv_p2 = data.get('foul_rcv_p2', [])
        m.foul_rewards = data.get('foul_rewards', [])
        m.distance_rewards = data.get('distance_rewards', [])
        m.pot_counts = data.get('pot_counts', [])
        m.win_loss_rewards = data.get('win_loss_rewards', [])
        m.approach_rewards = data.get('approach_rewards', [])
        m.miss_ball_counts = data.get('miss_ball_counts', [])
        m.wrong_ball_counts = data.get('wrong_ball_counts', [])
        m.white_pocket_counts = data.get('white_pocket_counts', [])
        m.illegal_choice_counts = data.get('illegal_choice_counts', [])
        m.angle_offsets = data.get('angle_offsets', [])
        m.power_values = data.get('power_values', [])
        m.intentional_pots = data.get('intentional_pots', [])
        m.lucky_pots = data.get('lucky_pots', [])
        return m

    def summary(self):
        """Generate a text summary of training metrics."""
        n = len(self.episode_rewards)
        if n == 0:
            return "No data."

        lines = []
        lines.append(f"总 Episode 数: {n}")
        lines.append("")

        # Overall stats
        lines.append("── 整体指标 ──────────────────────────────────")
        lines.append(f"  Reward:   mean={np.mean(self.episode_rewards):.3f}  "
                     f"std={np.std(self.episode_rewards):.3f}  "
                     f"min={np.min(self.episode_rewards):.3f}  "
                     f"max={np.max(self.episode_rewards):.3f}")
        lines.append(f"  Length:   mean={np.mean(self.episode_lengths):.1f}  "
                     f"max={np.max(self.episode_lengths)}")
        lines.append(f"  Break:    mean={np.mean(self.break_scores):.2f}  "
                     f"max={np.max(self.break_scores)}")
        lines.append(f"  Fouls:    mean={np.mean(self.foul_counts):.2f}  "
                     f"total={int(np.sum(self.foul_counts))}")
        # 犯规类型细分
        if self.miss_ball_counts:
            total_fouls = int(np.sum(self.foul_counts)) if self.foul_counts else 1
            miss_total = int(np.sum(self.miss_ball_counts))
            wrong_total = int(np.sum(self.wrong_ball_counts))
            wp_total = int(np.sum(self.white_pocket_counts))
            ilg_total = int(np.sum(self.illegal_choice_counts))
            lines.append(f"    空杆(miss_ball):   {miss_total}  "
                         f"({miss_total/max(total_fouls,1)*100:.1f}%)")
            lines.append(f"    碰错球(wrong_ball): {wrong_total}  "
                         f"({wrong_total/max(total_fouls,1)*100:.1f}%)")
            lines.append(f"    白球进袋(white_pocket): {wp_total}  "
                         f"({wp_total/max(total_fouls,1)*100:.1f}%)")
            if ilg_total > 0:
                lines.append(f"    非法选球(illegal_choice): {ilg_total}  "
                             f"({ilg_total/max(total_fouls,1)*100:.1f}%)")
        lines.append(f"  Pots:     mean={np.mean(self.pot_counts):.2f}  "
                     f"total={int(np.sum(self.pot_counts))}")
        if self.angle_offsets:
            lines.append(f"  Angle:    mean={np.mean(self.angle_offsets):.2f}°  "
                         f"(mean |offset|, 越小越精准)")
        lines.append("")

        # Reward breakdown
        lines.append("── Reward 分解 ────────────────────────────────")
        lines.append(f"  Foul reward:     mean={np.mean(self.foul_rewards):.3f}  "
                     f"(越接近0越好)")
        lines.append(f"  Pot/Miss reward: mean={np.mean(self.distance_rewards):.3f}  "
                     f"(越大越好，含进球奖励+miss惩罚)")
        if self.win_loss_rewards:
            lines.append(f"  Win/Loss reward: mean={np.mean(self.win_loss_rewards):.3f}  "
                         f"(越大越好)")
        lines.append("")

        # Trend: compare first 10% vs last 10%
        seg = max(n // 10, 1)
        lines.append("── 趋势（前10% vs 后10%）─────────────────────")
        r_first = np.mean(self.episode_rewards[:seg])
        r_last = np.mean(self.episode_rewards[-seg:])
        lines.append(f"  Reward:    {r_first:.3f} → {r_last:.3f}  "
                     f"({'↑' if r_last > r_first else '↓'} {abs(r_last-r_first):.3f})")

        f_first = np.mean(self.foul_rewards[:seg])
        f_last = np.mean(self.foul_rewards[-seg:])
        lines.append(f"  Foul:      {f_first:.3f} → {f_last:.3f}  "
                     f"({'↑好' if f_last > f_first else '↓差'} {abs(f_last-f_first):.3f})")

        d_first = np.mean(self.distance_rewards[:seg])
        d_last = np.mean(self.distance_rewards[-seg:])
        lines.append(f"  Pot/Miss: {d_first:.3f} → {d_last:.3f}  "
                     f"({'↑好' if d_last > d_first else '↓差'} {abs(d_last-d_first):.3f})")

        p_first = np.mean(self.pot_counts[:seg])
        p_last = np.mean(self.pot_counts[-seg:])
        lines.append(f"  Pots:      {p_first:.2f} → {p_last:.2f}  "
                     f"({'↑好' if p_last > p_first else '↓差'} {abs(p_last-p_first):.2f})")

        fc_first = np.mean(self.foul_counts[:seg])
        fc_last = np.mean(self.foul_counts[-seg:])
        lines.append(f"  Fouls:     {fc_first:.2f} → {fc_last:.2f}  "
                     f"({'↓好' if fc_last < fc_first else '↑差'} {abs(fc_last-fc_first):.2f})")

        if self.angle_offsets:
            a_first = np.mean(self.angle_offsets[:seg])
            a_last = np.mean(self.angle_offsets[-seg:])
            lines.append(f"  Angle:     {a_first:.2f}° → {a_last:.2f}°  "
                         f"({'↓好' if a_last < a_first else '↑差'} {abs(a_last-a_first):.2f}°)")

        lines.append("")

        # Score stats
        lines.append("── 比分 ───────────────────────────────────────")
        lines.append(f"  P1 总得分(pot): {int(np.sum(self.pot_p1))}  "
                     f"P2 总得分(pot): {int(np.sum(self.pot_p2))}")
        lines.append(f"  P1 总罚分(rcv): {int(np.sum(self.foul_rcv_p1))}  "
                     f"P2 总罚分(rcv): {int(np.sum(self.foul_rcv_p2))}")

        # Win/Loss stats
        if self.win_loss_rewards:
            wl = np.array(self.win_loss_rewards)
            wins = int(np.sum(wl > 0))
            losses = int(np.sum(wl < 0))
            draws = int(np.sum(wl == 0))
            lines.append("")
            lines.append("── 胜负（出手者视角）─────────────────────────")
            lines.append(f"  赢: {wins}  输: {losses}  平: {draws}  "
                         f"胜率: {wins/n*100:.1f}%")
            # 趋势
            wl_first = wl[:seg]
            wl_last = wl[-seg:]
            wr_first = np.sum(wl_first > 0) / len(wl_first) * 100
            wr_last = np.sum(wl_last > 0) / len(wl_last) * 100
            lines.append(f"  胜率趋势: {wr_first:.1f}% → {wr_last:.1f}%  "
                         f"({'↑好' if wr_last > wr_first else '↓差'} "
                         f"{abs(wr_last-wr_first):.1f}%)")

        if self.policy_losses:
            lines.append("")
            lines.append("── Loss ───────────────────────────────────────")
            lines.append(f"  Policy loss: last={self.policy_losses[-1]:.4f}  "
                         f"mean={np.mean(self.policy_losses):.4f}")
            if self.value_losses:
                lines.append(f"  Value loss:  last={self.value_losses[-1]:.4f}  "
                             f"mean={np.mean(self.value_losses):.4f}")

        return '\n'.join(lines)

    def plot(self, save_path=None):
        fig, axes = plt.subplots(4, 4, figsize=(24, 20))
        episodes = range(1, len(self.episode_rewards) + 1)
        n_ep = len(self.episode_rewards)
        window = min(50, n_ep) if n_ep >= 10 else n_ep

        def _plot_ma(ax, data, color, title, ylabel=''):
            ax.plot(episodes, data, color=color, alpha=0.25)
            if n_ep >= 10:
                ma = np.convolve(data, np.ones(window)/window, mode='valid')
                ax.plot(range(window, n_ep+1), ma, color=color, linewidth=2)
            ax.set_title(title, fontsize=10)
            if ylabel:
                ax.set_ylabel(ylabel, fontsize=8)
            ax.grid(True, alpha=0.3)

        # ══ Row 0: Behavior metrics ══
        _plot_ma(axes[0,0], self.episode_rewards, 'C0', 'Episode Reward')
        if self.angle_offsets and len(self.angle_offsets) == n_ep:
            _plot_ma(axes[0,1], self.angle_offsets, 'C1', 'Angle Offset', 'deg')
        else:
            _plot_ma(axes[0,1], self.episode_lengths, 'C1', 'Episode Length')
        if self.power_values and len(self.power_values) == n_ep:
            _plot_ma(axes[0,2], self.power_values, 'C2', 'Avg Power (V0)', 'm/s')
        else:
            axes[0,2].set_title('Power (no data)')
        _plot_ma(axes[0,3], self.episode_lengths, '#607D8B', 'Episode Length')

        # ══ Row 1: Pot breakdown ══
        if self.intentional_pots and len(self.intentional_pots) == n_ep:
            _plot_ma(axes[1,0], self.intentional_pots, '#2E7D32', 'Intentional Pots / Ep')
        else:
            axes[1,0].set_title('Intentional (no data)')
        if self.lucky_pots and len(self.lucky_pots) == n_ep:
            _plot_ma(axes[1,1], self.lucky_pots, '#FF8F00', 'Lucky Pots / Ep')
        else:
            axes[1,1].set_title('Lucky (no data)')
        _plot_ma(axes[1,2], self.foul_rewards, '#E53935', 'Foul Penalty / Ep')
        axes[1,2].axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
        _plot_ma(axes[1,3], self.approach_rewards if self.approach_rewards and len(self.approach_rewards)==n_ep
                 else self.distance_rewards, '#1565C0', 'Approach Reward / Ep')
        axes[1,3].axhline(y=0, color='gray', linestyle='--', linewidth=0.5)

        # ══ Row 2: Foul breakdown + diagnostics ══
        if self.miss_ball_counts and len(self.miss_ball_counts) == n_ep:
            m_a = np.array(self.miss_ball_counts, dtype=float)
            w_a = np.array(self.wrong_ball_counts, dtype=float)
            wp_a = np.array(self.white_pocket_counts, dtype=float)
            if n_ep >= window:
                x_ma = range(window, n_ep+1)
                axes[2,0].plot(x_ma, np.convolve(m_a, np.ones(window)/window, mode='valid'),
                               color='#E53935', linewidth=2, label='Miss')
                axes[2,0].plot(x_ma, np.convolve(w_a, np.ones(window)/window, mode='valid'),
                               color='#FB8C00', linewidth=2, label='Wrong')
                axes[2,0].plot(x_ma, np.convolve(wp_a, np.ones(window)/window, mode='valid'),
                               color='#8E24AA', linewidth=2, label='WhtPk')
            axes[2,0].legend(fontsize=7); axes[2,0].set_title('Foul Type', fontsize=10)
            axes[2,0].grid(True, alpha=0.3)
        else:
            axes[2,0].set_title('Foul Type (no data)')
        _plot_ma(axes[2,1], self.pot_counts, '#4CAF50', 'Total Pots / Ep')
        if self.episode_lengths and self.foul_counts:
            fr = [f/max(l,1) for f,l in zip(self.foul_counts, self.episode_lengths)]
            _plot_ma(axes[2,2], fr, '#FF5722', 'Foul Rate')
            axes[2,2].set_ylim(0, 1)
        else:
            axes[2,2].set_title('Foul Rate (no data)')
        _plot_ma(axes[2,3], self.break_scores, '#795548', 'Max Break / Ep')

        # ══ Row 3: Losses + Summary ══
        if self.policy_losses:
            pl_x = range(1, len(self.policy_losses)+1)
            axes[3,0].plot(pl_x, self.policy_losses, alpha=0.3, color='C0')
            if len(self.policy_losses) >= 10:
                w = min(50, len(self.policy_losses))
                axes[3,0].plot(range(w, len(self.policy_losses)+1),
                    np.convolve(self.policy_losses, np.ones(w)/w, mode='valid'),
                    color='C0', linewidth=2)
            axes[3,0].set_title('Policy Loss', fontsize=10); axes[3,0].grid(True, alpha=0.3)
        else:
            axes[3,0].set_title('Policy Loss (no data)')
        if self.value_losses:
            vl_x = range(1, len(self.value_losses)+1)
            axes[3,1].plot(vl_x, self.value_losses, alpha=0.3, color='#E91E63')
            if len(self.value_losses) >= 10:
                w = min(50, len(self.value_losses))
                axes[3,1].plot(range(w, len(self.value_losses)+1),
                    np.convolve(self.value_losses, np.ones(w)/w, mode='valid'),
                    color='#E91E63', linewidth=2)
            axes[3,1].set_title('Critic Loss', fontsize=10); axes[3,1].grid(True, alpha=0.3)
        else:
            axes[3,1].set_title('Critic Loss (no data)')
        if self.policy_losses and self.value_losses:
            ax_l = axes[3,2]; ax_r = ax_l.twinx()
            l1, = ax_l.plot(range(1,len(self.policy_losses)+1), self.policy_losses, color='C0', alpha=0.5)
            l2, = ax_r.plot(range(1,len(self.value_losses)+1), self.value_losses, color='#E91E63', alpha=0.5)
            ax_l.set_ylabel('Policy', color='C0', fontsize=7)
            ax_r.set_ylabel('Critic', color='#E91E63', fontsize=7)
            ax_l.legend([l1,l2], ['Policy','Critic'], fontsize=7, loc='upper left')
            axes[3,2].set_title('P vs C', fontsize=10); axes[3,2].grid(True, alpha=0.3)
        else:
            axes[3,2].set_title('P vs C (no data)')

        # Summary text
        axes[3,3].axis('off')
        if n_ep > 0:
            seg = max(n_ep//10, 1)
            def _t(a): return np.mean(a[:seg]), np.mean(a[-seg:])
            r1,r2 = _t(self.episode_rewards); f1,f2 = _t(self.foul_counts)
            p1,p2 = _t(self.pot_counts)
            s = [f"Eps: {n_ep}", f"Reward: {r1:.1f}->{r2:.1f}",
                 f"Fouls:  {f1:.1f}->{f2:.1f}", f"Pots:   {p1:.1f}->{p2:.1f}"]
            if self.miss_ball_counts and len(self.miss_ball_counts)==n_ep:
                m1,m2=_t(self.miss_ball_counts); w1,w2=_t(self.wrong_ball_counts)
                s += [f"Miss: {m1:.1f}->{m2:.1f}", f"Wrong:{w1:.1f}->{w2:.1f}"]
            if self.power_values and len(self.power_values)==n_ep:
                pw1,pw2=_t(self.power_values); s.append(f"V0: {pw1:.1f}->{pw2:.1f}")
            if self.intentional_pots and len(self.intentional_pots)==n_ep:
                i1,i2=_t(self.intentional_pots); l1,l2=_t(self.lucky_pots)
                s += [f"IntPot:{i1:.1f}->{i2:.1f}", f"Lucky: {l1:.1f}->{l2:.1f}"]
            if self.policy_losses:
                s.append(f"PLoss: {self.policy_losses[-1]:.3f}")
            axes[3,3].text(0.05, 0.95, '\n'.join(s), transform=axes[3,3].transAxes,
                fontsize=10, va='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#F5F5F5', edgecolor='#BDBDBD'))
            axes[3,3].set_title('Summary', fontsize=10)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=120)
        plt.close()



def train(args):
    algo = args.algo.lower()
    print("=" * 60)
    print(f"Snooker RL – Self-Play Training ({algo.upper()})")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ── Build experiment directory ────────────────────────────
    # Each training run gets its own subdirectory under save_dir:
    #   experiments/sac_20260512_143025/
    #   experiments/my_custom_name/
    if args.run_name:
        run_dir = os.path.join(args.save_dir, args.run_name)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_dir = os.path.join(args.save_dir, f'{algo}_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)
    print(f"Experiment directory: {run_dir}")

    env = SnookerEnv(render_mode='human' if args.render else None)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")

    agent = make_agent(algo, state_dim, action_dim, args)

    if args.load_model:
        if agent.load(args.load_model):
            print(f"Loaded model from {args.load_model}")
        else:
            print(f"Could not load model from {args.load_model}")

    # Build training meta (saved into metrics JSON)
    train_meta = {
        'algo': algo,
        'run_name': args.run_name or os.path.basename(run_dir),
        'run_dir': run_dir,
        'num_episodes': args.num_episodes,
        'state_dim': state_dim,
        'action_dim': action_dim,
        'lr': args.lr,
        'gamma': args.gamma,
        'batch_size': args.batch_size,
        'hidden_dim': args.hidden_dim,
        'start_time': datetime.now().isoformat(),
        'device': str(device),
    }
    if algo == 'sac':
        train_meta.update({
            'tau': args.tau,
            'lr_alpha': args.lr_alpha,
            'buffer_size': args.buffer_size,
            'warmup_steps': args.warmup_steps,
            'init_alpha': args.init_alpha,
            'updates_per_step': args.updates_per_step,
            'actor_update_interval': args.actor_update_interval,
            'normalize_rewards': True,
        })
    elif algo == 'ppo':
        train_meta.update({
            'lam': args.lam,
            'eps_clip': args.eps_clip,
            'k_epochs': args.k_epochs,
            'entropy_coef': args.entropy_coef,
            'update_interval': args.update_interval,
        })

    metrics = TrainingMetrics()

    # Algorithm-specific info
    if algo == 'sac':
        print(f"Replay buffer capacity: {args.buffer_size:,}")
        print(f"Warmup steps: {args.warmup_steps}")
        print(f"Updates per step: {args.updates_per_step}")
    elif algo == 'ppo':
        print(f"Update interval: {args.update_interval} steps")
        print(f"K epochs: {args.k_epochs}")

    print("\nStarting self-play training...")
    print("-" * 60)

    timestep = 0
    update_interval = args.update_interval
    train_start_time = datetime.now()
    last_log_time = train_start_time

    try:
        for episode in range(1, args.num_episodes + 1):
            state, _ = env.reset()
            episode_reward = 0.0
            episode_length = 0
            max_break = 0
            total_fouls = 0
            ep_foul_reward = 0.0
            ep_distance_reward = 0.0
            ep_win_loss_reward = 0.0
            ep_approach_reward = 0.0
            ep_pot_count = 0
            # 犯规类型计数
            ep_miss_ball = 0
            ep_wrong_ball = 0
            ep_white_pocket = 0
            ep_illegal_choice = 0
            # 精确进球 vs 运气进球
            ep_intentional_pots = 0
            ep_lucky_pots = 0
            # 角度精度
            ep_angle_offsets = []
            # 力度
            ep_power_values = []

            while True:
                # Current player takes action via shared policy
                action, log_prob, value = agent.select_action(state)

                next_state, reward, done, truncated, info = env.step(action)

                # Store transition
                if algo == 'sac':
                    agent.store_transition(
                        state, action, reward, next_state,
                        done or truncated)
                else:
                    # PPO uses its own memory
                    agent.memory.add(
                        state, action, log_prob, value, reward, done)

                episode_reward += reward
                episode_length += 1
                timestep += 1

                # Track max break
                current_brk = info.get('break', 0)
                if current_brk > max_break:
                    max_break = current_brk

                if info.get('foul', 0) > 0:
                    total_fouls += 1

                # Track reward breakdown
                bd = info.get('reward_breakdown', {})
                ep_foul_reward += bd.get('foul', 0.0)
                ep_distance_reward += bd.get('distance', 0.0)
                ep_win_loss_reward += bd.get('win_loss', 0.0)
                ep_approach_reward += bd.get('approach', 0.0)

                # Track foul type breakdown
                foul_type = bd.get('foul_type', None)
                if foul_type == 'miss_ball':
                    ep_miss_ball += 1
                elif foul_type == 'wrong_ball':
                    ep_wrong_ball += 1
                elif foul_type == 'white_pocket':
                    ep_white_pocket += 1
                elif foul_type == 'illegal_choice':
                    ep_illegal_choice += 1
                elif foul_type == 'physics_crash':
                    ep_miss_ball += 1  # 归入空杆类

                # Track angle offset
                offset = info.get('offset_deg', None)
                if offset is not None:
                    ep_angle_offsets.append(abs(offset))

                # Track power (V0)
                v0 = info.get('V0', None)
                if v0 is not None:
                    ep_power_values.append(v0)

                # Track pots
                pocketed = info.get('pocketed', [])
                ep_pot_count += sum(1 for b in pocketed if b != 'white')

                # Track pot type (intentional vs lucky)
                pot_type = bd.get('pot_type', None)
                if pot_type == 'intentional':
                    ep_intentional_pots += 1
                elif pot_type == 'lucky':
                    ep_lucky_pots += 1

                if args.render:
                    env.render()

                # Algorithm-specific update schedule
                if algo == 'sac':
                    # SAC updates every step (after warmup)
                    agent.update()
                elif algo == 'ppo':
                    if timestep % update_interval == 0:
                        agent.update()

                if done or truncated:
                    break

                state = next_state

            # Record episode metrics
            stats = agent.get_stats()
            policy_loss = stats['policy_loss'][-1] if stats['policy_loss'] else None
            value_loss = stats['value_loss'][-1] if stats['value_loss'] else None

            metrics.add(
                episode_reward, episode_length,
                policy_loss, value_loss,
                max_break, total_fouls,
                info.get('score_p1', 0), info.get('score_p2', 0),
                info.get('pot_p1', 0), info.get('pot_p2', 0),
                info.get('foul_rcv_p1', 0), info.get('foul_rcv_p2', 0),
                foul_reward=ep_foul_reward,
                distance_reward=ep_distance_reward,
                pot_count=ep_pot_count,
                win_loss_reward=ep_win_loss_reward,
                approach_reward=ep_approach_reward,
                miss_ball=ep_miss_ball,
                wrong_ball=ep_wrong_ball,
                white_pocket=ep_white_pocket,
                illegal_choice=ep_illegal_choice,
                angle_offset=float(np.mean(ep_angle_offsets)) if ep_angle_offsets else 0.0,
                power=float(np.mean(ep_power_values)) if ep_power_values else 0.0,
                intentional_pots=ep_intentional_pots,
                lucky_pots=ep_lucky_pots,
            )

            if episode % args.log_interval == 0:
                avg = metrics.get_average(args.log_interval)
                now = datetime.now()
                elapsed = now - train_start_time
                interval_sec = (now - last_log_time).total_seconds()
                last_log_time = now
                elapsed_sec = elapsed.total_seconds()

                # ETA calculation
                eps_done = episode
                eps_left = args.num_episodes - eps_done
                if eps_done > 0:
                    sec_per_ep = elapsed_sec / eps_done
                    eta_sec = int(sec_per_ep * eps_left)
                    eta_h, eta_rem = divmod(eta_sec, 3600)
                    eta_m, eta_s = divmod(eta_rem, 60)
                    if eta_h > 0:
                        eta_str = f"{eta_h}h{eta_m:02d}m"
                    else:
                        eta_str = f"{eta_m}m{eta_s:02d}s"
                else:
                    eta_str = "?"

                total_sec = int(elapsed_sec)
                h, rem = divmod(total_sec, 3600)
                m, s = divmod(rem, 60)
                elapsed_str = f"{h}:{m:02d}:{s:02d}"

                pct = eps_done / args.num_episodes * 100

                extra = ""
                if algo == 'sac':
                    alpha = getattr(agent, 'alpha', 0)
                    buf_sz = len(agent.replay_buffer)
                    extra = f"α={alpha:.3f} Buf={buf_sz}"
                elif algo == 'ppo':
                    extra = f"Step={timestep}"

                print(f"[{elapsed_str} ETA {eta_str}] "
                      f"Ep {episode}/{args.num_episodes} ({pct:.0f}%) | "
                      f"R: {avg['reward']:.2f} "
                      f"(foul:{avg['foul_reward']:.1f} pot:{avg['distance_reward']:.1f} "
                      f"appr:{avg['approach_reward']:.1f}) | "
                      f"Pots: {avg['pot_count']:.1f} "
                      f"(int:{avg['intentional_pots']:.1f} "
                      f"luck:{avg['lucky_pots']:.1f}) | "
                      f"Len: {avg['length']:.0f} | "
                      f"Break: {avg['break']:.1f} | "
                      f"Fouls: {avg['fouls']:.1f} "
                      f"(miss:{avg['miss_ball']:.1f} "
                      f"wrong:{avg['wrong_ball']:.1f} "
                      f"wp:{avg['white_pocket']:.1f}) | "
                      f"Ang: {avg['angle_offset']:.1f}° "
                      f"V0: {avg['power']:.1f} | "
                      f"{extra}")

            if episode % args.save_interval == 0:
                save_path = os.path.join(
                    run_dir, f'{algo}_snooker_ep{episode}.pt')
                agent.save(save_path)
                # Save metrics alongside checkpoint
                metrics_path = os.path.join(
                    run_dir, f'metrics_ep{episode}.json')
                metrics.save(metrics_path, meta=train_meta)
                print(f"  → Model saved: {save_path}")

            if episode % args.plot_interval == 0:
                plot_path = os.path.join(
                    run_dir, f'training_metrics_ep{episode}.png')
                metrics.plot(plot_path)

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")

    finally:
        env.close()

        final_save_path = os.path.join(
            run_dir, f'{algo}_snooker_final.pt')
        agent.save(final_save_path)
        print(f"\nFinal model saved to {final_save_path}")

        final_plot_path = os.path.join(
            run_dir, 'training_metrics_final.png')
        metrics.plot(final_plot_path)
        print(f"Final metrics plot saved to {final_plot_path}")

        # Save metrics JSON (all episode-level data)
        train_meta['end_time'] = datetime.now().isoformat()
        final_metrics_path = os.path.join(
            run_dir, 'metrics_final.json')
        metrics.save(final_metrics_path, meta=train_meta)
        print(f"Final metrics data saved to {final_metrics_path}")

        # Print summary
        print("\n" + "=" * 60)
        print("训练结果摘要")
        print("=" * 60)
        print(metrics.summary())
        print("=" * 60)

    print("\nTraining completed!")


def main():
    parser = argparse.ArgumentParser(
        description='Train RL agent for Snooker (Self-Play)')

    # ── Algorithm selection ───────────────────────────────────
    parser.add_argument('--algo', type=str, default='ppo',
                        choices=['ppo', 'sac'],
                        help='RL algorithm: ppo or sac (default: ppo)')

    # ── General ──────────────────────────────────────────────
    parser.add_argument('--num_episodes', type=int, default=2000,
                        help='Number of training episodes')
    parser.add_argument('--render', action='store_true',
                        help='Render environment during training')
    parser.add_argument('--load_model', type=str, default=None,
                        help='Path to model to load')
    parser.add_argument('--save_dir', type=str, default='experiments',
                        help='Base directory for experiment outputs')
    parser.add_argument('--run_name', type=str, default=None,
                        help='Experiment run name (default: {algo}_{timestamp})')

    # ── Analysis mode ────────────────────────────────────────
    parser.add_argument('--analyse', type=str, default=None,
                        help='Path to metrics JSON file to analyse (skip training)')
    parser.add_argument('--replot', type=str, default=None,
                        help='When used with --analyse, save new plot to this path')

    # ── Shared hyperparams ───────────────────────────────────
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (actor & critic for SAC)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Mini-batch size (64 for PPO, 256 for SAC)')
    parser.add_argument('--hidden_dim', type=int, default=256)

    # ── PPO-specific ─────────────────────────────────────────
    parser.add_argument('--lam', type=float, default=0.95,
                        help='[PPO] GAE lambda')
    parser.add_argument('--eps_clip', type=float, default=0.2,
                        help='[PPO] Clipping epsilon')
    parser.add_argument('--k_epochs', type=int, default=10,
                        help='[PPO] SGD epochs per update')
    parser.add_argument('--entropy_coef', type=float, default=0.05,
                        help='[PPO] Entropy bonus coefficient')
    parser.add_argument('--value_coef', type=float, default=0.5,
                        help='[PPO] Value loss coefficient')
    parser.add_argument('--update_interval', type=int, default=1024,
                        help='[PPO] Steps between updates')

    # ── SAC-specific ─────────────────────────────────────────
    parser.add_argument('--tau', type=float, default=0.002,
                        help='[SAC] Soft update coefficient (slower = more stable)')
    parser.add_argument('--lr_alpha', type=float, default=1e-4,
                        help='[SAC] Alpha (temperature) learning rate')
    parser.add_argument('--buffer_size', type=int, default=100000,
                        help='[SAC] Replay buffer capacity')
    parser.add_argument('--warmup_steps', type=int, default=2000,
                        help='[SAC] Random exploration steps before learning')
    parser.add_argument('--updates_per_step', type=int, default=1,
                        help='[SAC] Gradient updates per env step')
    parser.add_argument('--target_update_interval', type=int, default=1,
                        help='[SAC] Steps between target network soft-updates')
    parser.add_argument('--init_alpha', type=float, default=0.2,
                        help='[SAC] Initial temperature value')
    parser.add_argument('--actor_update_interval', type=int, default=2,
                        help='[SAC] Critic steps between actor updates (delayed actor)')

    # ── Training schedule ────────────────────────────────────
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Episodes between logging')
    parser.add_argument('--save_interval', type=int, default=100,
                        help='Episodes between saving')
    parser.add_argument('--plot_interval', type=int, default=100,
                        help='Episodes between plotting')

    args = parser.parse_args()

    # ── Analysis mode: load & display existing metrics ───────
    if args.analyse:
        if not os.path.exists(args.analyse):
            print(f"Error: metrics file not found: {args.analyse}")
            return
        metrics = TrainingMetrics.load(args.analyse)
        print(f"Loaded metrics from: {args.analyse}")
        print(f"Total episodes: {len(metrics.episode_rewards)}")
        print()
        print(metrics.summary())
        if args.replot:
            metrics.plot(args.replot)
            print(f"\nPlot saved to: {args.replot}")
        return

    # Auto-adjust batch_size for SAC if user didn't explicitly set it
    if args.algo == 'sac' and '--batch_size' not in sys.argv:
        args.batch_size = 256

    train(args)


if __name__ == '__main__':
    main()
