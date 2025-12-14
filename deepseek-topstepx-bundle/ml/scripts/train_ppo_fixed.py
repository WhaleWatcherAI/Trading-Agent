#!/usr/bin/env python3
"""
Fixed PPO training with correct architecture and simplified approach.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import deque
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "ml" / "data"
MODELS_DIR = ROOT / "ml" / "models"

# Configuration
class PPOConfig:
    # Fixed dimensions
    state_dim = 15  # Number of market features
    action_dim = 3  # Buy, Sell, Hold
    hidden_dim = 128

    # Environment
    initial_balance = 10000
    max_position_size = 5
    transaction_cost = 2.0

    # PPO hyperparameters
    learning_rate = 3e-4
    gamma = 0.99
    lambda_gae = 0.95
    clip_epsilon = 0.2
    value_loss_coef = 0.5
    entropy_coef = 0.01
    max_grad_norm = 0.5

    # Training
    batch_size = 32
    n_epochs = 4
    n_steps = 256
    total_timesteps = 10000  # Reduced for faster training


class SimpleActorCritic(nn.Module):
    """Fixed Actor-Critic network with correct dimensions."""

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(SimpleActorCritic, self).__init__()

        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Actor head (policy)
        self.actor = nn.Linear(hidden_dim, action_dim)

        # Critic head (value)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        # Ensure state is tensor
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state)

        # Ensure correct shape
        if state.dim() == 1:
            state = state.unsqueeze(0)

        # Get shared features
        features = self.shared(state)

        # Get action logits and value
        action_logits = self.actor(features)
        value = self.critic(features)

        return action_logits.squeeze(0), value.squeeze()


class SimpleTradingEnv:
    """Simplified trading environment."""

    def __init__(self, data, config):
        self.data = data
        self.config = config
        self.reset()

    def reset(self):
        """Reset to initial state."""
        self.current_idx = 100  # Start after some history
        self.balance = self.config.initial_balance
        self.position = 0
        self.entry_price = 0

        return self._get_state()

    def _get_state(self):
        """Get current state as numpy array."""
        if self.current_idx >= len(self.data):
            self.current_idx = len(self.data) - 1

        row = self.data.iloc[self.current_idx]

        # Create state vector with exactly 15 features
        state = np.array([
            row.get('close', 100) / 1000,  # Normalized price
            row.get('volume', 1000) / 10000,  # Normalized volume
            row.get('rsi', 50) / 100,  # RSI (0-100)
            row.get('macd', 0) / 10,  # MACD
            row.get('volatility', 0.01),  # Volatility
            self.position / 5,  # Normalized position
            self.balance / 10000,  # Normalized balance
            (self.balance - self.config.initial_balance) / 1000,  # P&L
            row.get('sma_20', 100) / 1000,  # SMA 20
            row.get('sma_50', 100) / 1000,  # SMA 50
            row.get('bb_upper', 100) / 1000,  # BB Upper
            row.get('bb_lower', 100) / 1000,  # BB Lower
            row.get('atr', 1) / 10,  # ATR
            row.get('volume_ratio', 1),  # Volume ratio
            self.current_idx / len(self.data)  # Time progress
        ], dtype=np.float32)

        # Ensure exactly 15 features
        if len(state) < 15:
            state = np.pad(state, (0, 15 - len(state)), 'constant')
        elif len(state) > 15:
            state = state[:15]

        return state

    def step(self, action):
        """Execute action and return reward."""
        current_price = self.data.iloc[self.current_idx].get('close', 100)
        reward = 0

        # Execute action
        if action == 1:  # Buy
            if self.position < self.config.max_position_size:
                self.position += 1
                self.balance -= (current_price + self.config.transaction_cost)
                if self.position == 1:
                    self.entry_price = current_price

        elif action == 2:  # Sell
            if self.position > 0:
                profit = (current_price - self.entry_price) * self.position
                self.balance += current_price - self.config.transaction_cost
                reward = profit / 100  # Scale reward
                self.position = 0

        # Move to next timestep
        self.current_idx += 1

        # Calculate reward based on position
        if self.position > 0 and self.current_idx < len(self.data):
            next_price = self.data.iloc[self.current_idx].get('close', current_price)
            unrealized = (next_price - self.entry_price) * self.position
            reward += unrealized / 1000  # Small reward for unrealized gains

        # Check if done
        done = (self.current_idx >= len(self.data) - 1) or (self.balance < 1000)

        # Get new state
        new_state = self._get_state()

        return new_state, reward, done, {}


class PPOAgent:
    """Fixed PPO agent."""

    def __init__(self, config):
        self.config = config
        self.network = SimpleActorCritic(
            config.state_dim,
            config.action_dim,
            config.hidden_dim
        )
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=config.learning_rate
        )

        # Buffers
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def select_action(self, state):
        """Select action using current policy."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state)
            action_logits, value = self.network(state_tensor)

            dist = Categorical(logits=action_logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            return action.item(), log_prob.item(), value.item()

    def update(self):
        """Simplified PPO update."""
        if len(self.states) < self.config.batch_size:
            return {}

        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states))
        actions = torch.LongTensor(self.actions)
        rewards = torch.FloatTensor(self.rewards)
        old_log_probs = torch.FloatTensor(self.log_probs)
        old_values = torch.FloatTensor(self.values)
        dones = torch.FloatTensor(self.dones)

        # Calculate returns and advantages
        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)

        running_return = 0
        for t in reversed(range(len(rewards))):
            if dones[t]:
                running_return = 0
            running_return = rewards[t] + self.config.gamma * running_return
            returns[t] = running_return

        advantages = returns - old_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update
        total_loss = 0
        for _ in range(self.config.n_epochs):
            # Forward pass
            action_logits, values = self.network(states)
            dist = Categorical(logits=action_logits)
            log_probs = dist.log_prob(actions)

            # Calculate ratio
            ratio = torch.exp(log_probs - old_log_probs)

            # Clipped objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = F.mse_loss(values.squeeze(), returns)

            # Entropy
            entropy = dist.entropy().mean()

            # Total loss
            loss = policy_loss + self.config.value_loss_coef * value_loss - self.config.entropy_coef * entropy

            # Backprop
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.config.max_grad_norm)
            self.optimizer.step()

            total_loss += loss.item()

        # Clear buffers
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()

        return {'loss': total_loss / self.config.n_epochs}


def train_ppo():
    """Train PPO agent with fixed architecture."""
    print("\n🚀 Training Fixed PPO Agent...")

    # Load data
    spy_path = DATA_DIR / 'spy_real_data.parquet'
    regular_path = DATA_DIR / 'market_data.parquet'

    if spy_path.exists():
        data = pd.read_parquet(spy_path)
        print(f"Using SPY data: {len(data)} rows")
    elif regular_path.exists():
        data = pd.read_parquet(regular_path)
        print(f"Using market data: {len(data)} rows")
    else:
        print("No data found!")
        return None

    # Ensure required columns exist
    required_cols = ['close', 'volume', 'rsi', 'macd', 'volatility', 'sma_20', 'sma_50',
                    'bb_upper', 'bb_lower', 'atr', 'volume_ratio']

    for col in required_cols:
        if col not in data.columns:
            # Add default values
            if col == 'close':
                data['close'] = data.get('Close', 100)
            else:
                data[col] = 0

    # Clean data
    data = data.fillna(0)

    # Initialize
    config = PPOConfig()
    env = SimpleTradingEnv(data, config)
    agent = PPOAgent(config)

    # Training metrics
    episode_rewards = []
    episode_lengths = []
    losses = []

    # Progress bar
    pbar = tqdm(total=config.total_timesteps)
    total_steps = 0

    while total_steps < config.total_timesteps:
        state = env.reset()
        episode_reward = 0
        episode_length = 0

        while True:
            # Select action
            action, log_prob, value = agent.select_action(state)

            # Step environment
            next_state, reward, done, _ = env.step(action)

            # Store transition
            agent.states.append(state)
            agent.actions.append(action)
            agent.rewards.append(reward)
            agent.values.append(value)
            agent.log_probs.append(log_prob)
            agent.dones.append(done)

            episode_reward += reward
            episode_length += 1
            total_steps += 1

            pbar.update(1)

            # Update if buffer full
            if len(agent.states) >= config.n_steps:
                metrics = agent.update()
                if 'loss' in metrics:
                    losses.append(metrics['loss'])

            if done:
                break

            state = next_state

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        # Update progress
        if len(episode_rewards) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            pbar.set_description(f"Avg Reward: {avg_reward:.3f}")

    pbar.close()

    # Save model
    torch.save(agent.network.state_dict(), MODELS_DIR / 'ppo_fixed.pth')

    # Calculate final metrics
    metrics = {
        'model_type': 'PPO_Fixed',
        'total_timesteps': total_steps,
        'total_episodes': len(episode_rewards),
        'avg_episode_reward': float(np.mean(episode_rewards)),
        'std_episode_reward': float(np.std(episode_rewards)),
        'avg_episode_length': float(np.mean(episode_lengths)),
        'final_balance': env.balance,
        'total_return': (env.balance - config.initial_balance) / config.initial_balance,
        'avg_loss': float(np.mean(losses)) if losses else 0,
        'trained_at': datetime.utcnow().isoformat()
    }

    # Save metrics
    with open(MODELS_DIR / 'ppo_metrics_fixed.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "="*50)
    print("PPO TRAINING COMPLETE!")
    print("="*50)
    print(f"Episodes: {len(episode_rewards)}")
    print(f"Avg Reward: {metrics['avg_episode_reward']:.3f}")
    print(f"Total Return: {metrics['total_return']:.2%}")
    print(f"\nModel saved to: {MODELS_DIR / 'ppo_fixed.pth'}")

    return agent, metrics


if __name__ == "__main__":
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    agent, metrics = train_ppo()

    if agent is not None:
        print("\n✅ PPO successfully trained and ready for position management!")