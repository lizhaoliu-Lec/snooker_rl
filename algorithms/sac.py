"""
SAC (Soft Actor-Critic) Algorithm Implementation for Snooker RL

Off-policy, maximum-entropy RL with replay buffer.
Key advantages over PPO for this environment:
  - Replay buffer enables massive sample reuse (~10x more sample-efficient)
  - Off-policy: can learn from stale data collected by old policies
  - Automatic temperature (alpha) tuning balances exploration vs exploitation
  - Per-step updates instead of batch rollout updates

Action space (8-dim continuous, all in [-1, 1]):
  [0] place_x      – D-zone cue ball placement x (only used when ball_in_hand)
  [1] place_y      – D-zone cue ball placement y (only used when ball_in_hand)
  [2] target_idx   – mapped to discrete target ball index
  [3] pocket_idx   – mapped to one of 6 pockets (target pocket for reward)
  [4] angle_offset – deviation from white→target centre line (±15°)
  [5] power        – shot speed V0 mapped to [0.5, 6.0] m/s
  [6] b_spin       – topspin (+) / backspin (-) mapped to [-0.8, +0.8]
  [7] a_spin       – sidespin left(-) / right(+) mapped to [-0.5, +0.5]

Unified interface (same as PPO):
  - select_action(state) -> (action, log_prob, value)
  - store_transition(state, action, reward, next_state, done)
  - update() -> None
  - save(filepath) / load(filepath)
  - get_stats() -> dict
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import os
import random


# ════════════════════════════════════════════════════════════════
# Running Reward Normalizer
# ════════════════════════════════════════════════════════════════

class RewardNormalizer:
    """
    Running mean/std normalizer for rewards.
    Keeps Q-value targets in a stable range, preventing critic explosion.
    """

    def __init__(self, clip=5.0):
        self.clip = clip
        self._mean = 0.0
        self._var = 1.0
        self._count = 0

    def update(self, reward):
        """Update running stats with a single reward."""
        self._count += 1
        if self._count == 1:
            self._mean = reward
            self._var = 0.0
        else:
            old_mean = self._mean
            self._mean += (reward - old_mean) / self._count
            self._var += (reward - old_mean) * (reward - self._mean)

    def normalize(self, reward):
        """Normalize a reward value."""
        std = max(np.sqrt(self._var / max(self._count, 1)), 1e-6)
        return np.clip((reward - self._mean) / std, -self.clip, self.clip)


# ════════════════════════════════════════════════════════════════
# Replay Buffer
# ════════════════════════════════════════════════════════════════

class ReplayBuffer:
    """
    Fixed-size circular replay buffer for off-policy learning.

    Stores transitions as numpy arrays for memory efficiency;
    converts to tensors only when sampling a batch.
    """

    def __init__(self, capacity, state_dim, action_dim):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0

        # Pre-allocate numpy arrays
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        """Store a single transition."""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        """Sample a random mini-batch of transitions."""
        indices = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.states[indices]),
            torch.FloatTensor(self.actions[indices]),
            torch.FloatTensor(self.rewards[indices]).unsqueeze(1),
            torch.FloatTensor(self.next_states[indices]),
            torch.FloatTensor(self.dones[indices]).unsqueeze(1),
        )

    def __len__(self):
        return self.size


# ════════════════════════════════════════════════════════════════
# Networks
# ════════════════════════════════════════════════════════════════

class SoftQNetwork(nn.Module):
    """
    Twin Q-networks (Q1, Q2) for SAC.
    Takes (state, action) and outputs a scalar Q-value.
    """

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()

        # Q1
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Q2
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)

    def q1_forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa)


class GaussianPolicy(nn.Module):
    """
    Squashed Gaussian policy for SAC.
    Outputs actions in [-1, 1] via Tanh squashing.
    """

    LOG_STD_MIN = -5
    LOG_STD_MAX = 2

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()

        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        features = self.feature_net(state)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std

    def sample(self, state):
        """
        Sample action using the reparametrisation trick.
        Returns: (squashed_action, log_prob, mean)
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mean, std)

        # Reparametrisation trick: z = mean + std * eps
        z = dist.rsample()

        # Squash through tanh → action in [-1, 1]
        action = torch.tanh(z)

        # Log probability with Jacobian correction for tanh squashing
        # log π(a|s) = log μ(z|s) - Σ log(1 - tanh²(zᵢ))
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob, mean

    def get_deterministic(self, state):
        """Get deterministic action (mean) for evaluation."""
        mean, _ = self.forward(state)
        return torch.tanh(mean)


# ════════════════════════════════════════════════════════════════
# SAC Agent
# ════════════════════════════════════════════════════════════════

class SAC:
    """
    Soft Actor-Critic with automatic temperature tuning.

    Provides the same interface as PPO:
      - select_action(state) -> (action, log_prob, value)
      - update()
      - save() / load()
      - get_stats()

    Additionally provides:
      - store_transition(state, action, reward, next_state, done)
    """

    def __init__(self, state_dim, action_dim, cfg=None):
        self.state_dim = state_dim
        self.action_dim = action_dim

        if cfg is None:
            cfg = {}

        # ── Hyperparameters ──────────────────────────────────
        self.gamma = cfg.get('gamma', 0.99)
        self.tau = cfg.get('tau', 0.002)        # target network soft update (slower)
        self.lr_actor = cfg.get('lr_actor', 1e-4)
        self.lr_critic = cfg.get('lr_critic', 1e-4)
        self.lr_alpha = cfg.get('lr_alpha', 1e-4)
        self.batch_size = cfg.get('batch_size', 256)
        self.hidden_dim = cfg.get('hidden_dim', 256)
        self.buffer_size = cfg.get('buffer_size', 100_000)
        self.warmup_steps = cfg.get('warmup_steps', 2000)
        self.updates_per_step = cfg.get('updates_per_step', 1)
        self.target_update_interval = cfg.get('target_update_interval', 1)
        self.init_alpha = cfg.get('init_alpha', 0.2)
        self.actor_update_interval = cfg.get('actor_update_interval', 2)
        self.normalize_rewards = cfg.get('normalize_rewards', True)

        # ── Device ───────────────────────────────────────────
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        # ── Networks ─────────────────────────────────────────
        self.actor = GaussianPolicy(
            state_dim, action_dim, self.hidden_dim).to(self.device)
        self.critic = SoftQNetwork(
            state_dim, action_dim, self.hidden_dim).to(self.device)
        self.critic_target = SoftQNetwork(
            state_dim, action_dim, self.hidden_dim).to(self.device)

        # Initialise target network as a copy
        self.critic_target.load_state_dict(self.critic.state_dict())

        # ── Optimisers ───────────────────────────────────────
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=self.lr_actor)
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=self.lr_critic)

        # ── Automatic temperature (alpha) tuning ─────────────
        # Target entropy = -dim(A)  (heuristic from the SAC paper)
        self.target_entropy = -float(action_dim)
        self.log_alpha = torch.tensor(
            np.log(self.init_alpha), dtype=torch.float32,
            requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam(
            [self.log_alpha], lr=self.lr_alpha)

        # ── Replay buffer ────────────────────────────────────
        self.replay_buffer = ReplayBuffer(
            self.buffer_size, state_dim, action_dim)

        # ── Reward normalizer ────────────────────────────────
        self.reward_normalizer = RewardNormalizer() if self.normalize_rewards else None

        # Compatibility shim: PPO training loop calls agent.memory.add()
        # We provide a lightweight adapter so both code paths work.
        self.memory = self._MemoryShim(self)

        # ── Stats ────────────────────────────────────────────
        self._update_count = 0
        self.training_stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'alpha': [],
            'q1_value': [],
            'q2_value': [],
        }

    # ── Memory adapter (for PPO-compatible train loop) ───────
    class _MemoryShim:
        """
        Allows the training loop to call `agent.memory.add(...)` with the
        same PPO signature.  We just store the transition in the replay
        buffer instead (next_state is deferred to store_transition).
        """
        def __init__(self, parent):
            self._parent = parent
            self._pending_state = None
            self._pending_action = None
            self.states = []  # compatibility: len(agent.memory.states)

        def add(self, state, action, log_prob, value, reward, done):
            # For SAC we need (s, a, r, s', d).  The training loop calls
            # add() *before* advancing to next_state, so we buffer one step.
            if self._pending_state is not None:
                # Now we have (s, a, r, s'=state, d=done-of-previous)
                self._parent.replay_buffer.add(
                    self._pending_state, self._pending_action,
                    self._pending_reward, state, self._pending_done)
                self.states.append(0)  # just to track count
            self._pending_state = state.copy()
            self._pending_action = action.copy()
            self._pending_reward = reward
            self._pending_done = done

            # If done, flush the pending transition with a dummy next_state
            if done:
                self._parent.replay_buffer.add(
                    self._pending_state, self._pending_action,
                    self._pending_reward, state, True)
                self.states.append(0)
                self._pending_state = None

        def clear(self):
            # No-op for replay buffer (we never discard data)
            pass

    # ── Public API ───────────────────────────────────────────

    def store_transition(self, state, action, reward, next_state, done):
        """Direct API to store a transition (preferred over memory shim)."""
        if self.reward_normalizer is not None:
            self.reward_normalizer.update(reward)
            norm_reward = self.reward_normalizer.normalize(reward)
        else:
            norm_reward = reward
        self.replay_buffer.add(state, action, norm_reward, next_state, done)

    @property
    def alpha(self):
        return self.log_alpha.exp().item()

    def select_action(self, state, deterministic=False):
        """
        Select action given state.

        Returns (action, log_prob, q_value) to match PPO interface.
        During warmup (buffer too small), returns random actions.
        """
        if (not deterministic
                and len(self.replay_buffer) < self.warmup_steps):
            # Random exploration during warmup
            action = np.random.uniform(-1, 1, self.action_dim).astype(
                np.float32)
            return action, 0.0, 0.0

        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if deterministic:
                action = self.actor.get_deterministic(state_t)
                log_prob = 0.0
            else:
                action, log_prob_t, _ = self.actor.sample(state_t)
                log_prob = log_prob_t.cpu().numpy()[0][0]

            # Use Q1 as a value estimate (for logging compatibility)
            q_val = self.critic.q1_forward(
                state_t, action).cpu().numpy()[0][0]

        action_np = action.cpu().numpy()[0].astype(np.float32)
        return action_np, float(log_prob), float(q_val)

    def update(self):
        """
        Perform one (or more) gradient updates from the replay buffer.

        Called by the training loop.  For SAC, each call performs
        `updates_per_step` gradient steps on critic + actor + alpha.
        """
        if len(self.replay_buffer) < self.warmup_steps:
            return
        if len(self.replay_buffer) < self.batch_size:
            return

        for _ in range(self.updates_per_step):
            self._update_step()

    def _update_step(self):
        """Single SAC update step."""
        # ── Sample batch ─────────────────────────────────────
        (states, actions, rewards,
         next_states, dones) = self.replay_buffer.sample(self.batch_size)

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        alpha = self.log_alpha.exp().detach()

        # ── Critic update (every step) ───────────────────────
        with torch.no_grad():
            next_actions, next_log_probs, _ = self.actor.sample(next_states)
            q1_next, q2_next = self.critic_target(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next) - alpha * next_log_probs
            q_target = rewards + self.gamma * (1 - dones) * q_next

        q1_pred, q2_pred = self.critic(states, actions)
        critic_loss = (F.mse_loss(q1_pred, q_target)
                       + F.mse_loss(q2_pred, q_target))

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        self._update_count += 1

        # ── Actor + Alpha update (delayed) ───────────────────
        # Only update actor every `actor_update_interval` critic steps
        # This lets the critic converge before the actor chases it
        if self._update_count % self.actor_update_interval == 0:
            new_actions, log_probs, _ = self.actor.sample(states)
            q1_new, q2_new = self.critic(states, new_actions)
            q_new = torch.min(q1_new, q2_new)

            actor_loss = (alpha * log_probs - q_new).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()

            # ── Alpha (temperature) update ───────────────────
            alpha_loss = -(self.log_alpha.exp() * (
                log_probs.detach() + self.target_entropy)).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self.training_stats['policy_loss'].append(actor_loss.item())
            self.training_stats['entropy'].append(-log_probs.mean().item())
            self.training_stats['alpha'].append(self.log_alpha.exp().item())

        # ── Soft-update target network ───────────────────────
        if self._update_count % self.target_update_interval == 0:
            self._soft_update(self.critic_target, self.critic, self.tau)

        # ── Record stats ─────────────────────────────────────
        self.training_stats['value_loss'].append(critic_loss.item())
        self.training_stats['q1_value'].append(q1_pred.mean().item())
        self.training_stats['q2_value'].append(q2_pred.mean().item())

    @staticmethod
    def _soft_update(target_net, source_net, tau):
        """Polyak averaging: θ_target ← τ·θ_source + (1-τ)·θ_target"""
        for tp, sp in zip(target_net.parameters(), source_net.parameters()):
            tp.data.copy_(tau * sp.data + (1 - tau) * tp.data)

    # ── Save / Load ──────────────────────────────────────────

    def save(self, filepath):
        """Save model checkpoint."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'log_alpha': self.log_alpha.detach().cpu(),
            'alpha_optimizer': self.alpha_optimizer.state_dict(),
            'training_stats': self.training_stats,
            'update_count': self._update_count,
            'algorithm': 'SAC',
        }, filepath)

    def load(self, filepath):
        """Load model checkpoint."""
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=self.device)
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(
                checkpoint['critic_target_state_dict'])
            self.actor_optimizer.load_state_dict(
                checkpoint['actor_optimizer'])
            self.critic_optimizer.load_state_dict(
                checkpoint['critic_optimizer'])
            self.log_alpha = torch.tensor(
                checkpoint['log_alpha'].item(), dtype=torch.float32,
                requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam(
                [self.log_alpha], lr=self.lr_alpha)
            self.alpha_optimizer.load_state_dict(
                checkpoint['alpha_optimizer'])
            self.training_stats = checkpoint.get(
                'training_stats', self.training_stats)
            self._update_count = checkpoint.get('update_count', 0)
            return True
        return False

    def get_stats(self):
        """Get training statistics (compatible with PPO interface)."""
        return self.training_stats
