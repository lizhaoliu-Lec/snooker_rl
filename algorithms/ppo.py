"""
PPO (Proximal Policy Optimization) Algorithm Implementation for Snooker RL
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from collections import deque
import os
from datetime import datetime


class ActorCritic(nn.Module):
    """
    Actor-Critic Network for PPO
    - Actor: Outputs action mean and standard deviation
    - Critic: Estimates state value
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(ActorCritic, self).__init__()
        
        # Shared feature extractor
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh()
        )
        
        # Actor (policy) network
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Tanh()  # Bound mean to [-1, 1]
        )
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Critic (value) network
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x):
        """Forward pass returning both action distribution and value"""
        features = self.feature_net(x)
        
        mean = self.actor_mean(features)
        std = torch.exp(self.actor_log_std)
        
        value = self.critic(features)
        
        return mean, std, value
    
    def get_action(self, x):
        """Sample action from the policy"""
        mean, std, value = self.forward(x)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action, log_prob, value
    
    def evaluate(self, x, action):
        """Evaluate actions according to the current policy"""
        mean, std, value = self.forward(x)
        dist = Normal(mean, std)
        
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return log_prob, entropy, value


class PPOMemory:
    """Experience buffer for PPO"""
    def __init__(self, batch_size):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        self.batch_size = batch_size
        
    def add(self, state, action, log_prob, value, reward, done):
        """Add experience to memory"""
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
        
    def clear(self):
        """Clear memory"""
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        
    def generate_batches(self):
        """Generate random batches from memory"""
        states = torch.FloatTensor(np.array(self.states))
        actions = torch.FloatTensor(np.array(self.actions))
        old_log_probs = torch.FloatTensor(np.array(self.log_probs))
        values = torch.FloatTensor(np.array(self.values))
        rewards = torch.FloatTensor(np.array(self.rewards))
        dones = torch.FloatTensor(np.array(self.dones))
        
        # Compute returns and advantages
        returns, advantages = self._compute_gae(rewards, values, dones)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Generate batches
        n_states = len(states)
        indices = np.arange(n_states)
        np.random.shuffle(indices)
        
        for start in range(0, n_states, self.batch_size):
            end = min(start + self.batch_size, n_states)
            batch_indices = indices[start:end]
            
            yield (
                states[batch_indices],
                actions[batch_indices],
                old_log_probs[batch_indices],
                values[batch_indices],
                returns[batch_indices],
                advantages[batch_indices]
            )
            
    def _compute_gae(self, rewards, values, dones, gamma=0.99, lam=0.95):
        """Compute Generalized Advantage Estimation"""
        n_steps = len(rewards)
        advantages = torch.zeros(n_steps)
        returns = torch.zeros(n_steps)
        
        gae = 0
        next_value = 0
        
        for t in reversed(range(n_steps)):
            mask = 1 - dones[t]
            delta = rewards[t] + gamma * next_value * mask - values[t]
            gae = delta + gamma * lam * mask * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
            next_value = values[t]
            
        return returns, advantages


class PPO:
    """
    Proximal Policy Optimization Agent
    """
    def __init__(self, state_dim, action_dim, cfg=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Hyperparameters
        self.gamma = cfg.get('gamma', 0.99)
        self.lam = cfg.get('lam', 0.95)
        self.lr = cfg.get('lr', 3e-4)
        self.eps_clip = cfg.get('eps_clip', 0.2)
        self.k_epochs = cfg.get('k_epochs', 10)
        self.entropy_coef = cfg.get('entropy_coef', 0.01)
        self.value_coef = cfg.get('value_coef', 0.5)
        self.max_grad_norm = cfg.get('max_grad_norm', 0.5)
        self.batch_size = cfg.get('batch_size', 64)
        self.hidden_dim = cfg.get('hidden_dim', 256)
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Networks
        self.policy = ActorCritic(state_dim, action_dim, self.hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
        
        # Memory
        self.memory = PPOMemory(self.batch_size)
        
        # Training stats
        self.training_stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'approx_kl': [],
            'clip_fraction': []
        }
        
    def select_action(self, state):
        """Select action given state"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, log_prob, value = self.policy.get_action(state)
            
        return action.cpu().numpy()[0], log_prob.cpu().numpy()[0], value.cpu().numpy()[0][0]
    
    def update(self):
        """Update policy using PPO algorithm"""
        if len(self.memory.states) < self.batch_size:
            return
            
        # Generate batches
        batch_count = 0
        policy_losses = []
        value_losses = []
        entropies = []
        approx_kls = []
        clip_fractions = []
        
        for batch in self.memory.generate_batches():
            states, actions, old_log_probs, values, returns, advantages = batch
            states = states.to(self.device)
            actions = actions.to(self.device)
            old_log_probs = old_log_probs.to(self.device)
            values = values.to(self.device)
            returns = returns.to(self.device)
            advantages = advantages.to(self.device)
            
            # Evaluate actions
            new_log_probs, entropy, state_values = self.policy.evaluate(states, actions)
            
            # Compute ratios
            ratios = torch.exp(new_log_probs - old_log_probs)
            
            # Clipped surrogate objective
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = nn.functional.mse_loss(state_values.squeeze(), returns)
            
            # Entropy bonus
            entropy_loss = -entropy.mean()
            
            # Total loss
            loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
            
            # Update
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            # Compute metrics
            with torch.no_grad():
                approx_kl = (old_log_probs - new_log_probs).mean()
                clip_fraction = ((ratios - 1).abs() > self.eps_clip).float().mean()
                
            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropies.append(entropy.mean().item())
            approx_kls.append(approx_kl.item())
            clip_fractions.append(clip_fraction.item())
            batch_count += 1
            
        # Update stats
        if batch_count > 0:
            self.training_stats['policy_loss'].append(np.mean(policy_losses))
            self.training_stats['value_loss'].append(np.mean(value_losses))
            self.training_stats['entropy'].append(np.mean(entropies))
            self.training_stats['approx_kl'].append(np.mean(approx_kls))
            self.training_stats['clip_fraction'].append(np.mean(clip_fractions))
            
        self.memory.clear()
        
    def save(self, filepath):
        """Save model checkpoint"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_stats': self.training_stats
        }, filepath)
        
    def load(self, filepath):
        """Load model checkpoint"""
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=self.device)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.training_stats = checkpoint.get('training_stats', self.training_stats)
            return True
        return False
    
    def get_stats(self):
        """Get training statistics"""
        return self.training_stats
