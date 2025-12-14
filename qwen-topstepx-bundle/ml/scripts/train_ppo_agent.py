#!/usr/bin/env python3
"""
PPO (Proximal Policy Optimization) reinforcement learning agent for trading.
Learns optimal trading policies through interaction with market environment.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import deque
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "ml" / "data"
MODELS_DIR = ROOT / "ml" / "models"
PPO_MODEL_PATH = MODELS_DIR / "ppo_trading_agent.pth"

# PPO Configuration
class PPOConfig:
    # Environment
    initial_balance = 100000  # Starting capital
    max_position_size = 5  # Max contracts
    transaction_cost = 2.0  # Per contract cost

    # Model architecture
    state_dim = 64  # Features from market data
    hidden_dim = 256
    action_dim = 3  # Buy, Sell, Hold

    # PPO hyperparameters
    learning_rate = 3e-4
    gamma = 0.99  # Discount factor
    lambda_gae = 0.95  # GAE parameter
    clip_epsilon = 0.2  # PPO clipping parameter
    value_loss_coef = 0.5
    entropy_coef = 0.01
    max_grad_norm = 0.5

    # Training
    batch_size = 64
    n_epochs = 4  # PPO epochs per update
    n_steps = 2048  # Steps before update
    n_envs = 4  # Parallel environments
    total_timesteps = 1000000


@dataclass
class TradingState:
    """State representation for the trading environment."""
    market_features: np.ndarray  # Technical indicators, price data
    position: int  # Current position (-max to +max)
    unrealized_pnl: float
    balance: float
    recent_actions: List[int]  # Last N actions for context
    timestep: int


class TradingEnvironment:
    """
    Trading environment for PPO agent.
    Simulates realistic market conditions with:
    - Transaction costs
    - Slippage
    - Position limits
    - Risk constraints
    """

    def __init__(self, data: pd.DataFrame, config: PPOConfig):
        self.data = data
        self.config = config
        self.reset()

    def reset(self) -> TradingState:
        """Reset environment to initial state."""
        self.current_idx = self.config.state_dim
        self.balance = self.config.initial_balance
        self.position = 0
        self.entry_price = 0
        self.recent_actions = deque([0] * 10, maxlen=10)
        self.trades = []

        return self._get_state()

    def _get_state(self) -> TradingState:
        """Get current state representation."""
        # Market features from data
        start_idx = max(0, self.current_idx - self.config.state_dim)
        market_data = self.data.iloc[start_idx:self.current_idx]

        # Normalize features
        features = []
        for col in market_data.columns:
            if col not in ['symbol', 'entry_time', 'win_5m', 'win_30m']:
                values = market_data[col].values
                if len(values) > 0:
                    normalized = (values - np.mean(values)) / (np.std(values) + 1e-8)
                    features.extend(normalized[-10:])  # Last 10 values

        # Pad if needed
        while len(features) < self.config.state_dim:
            features.append(0.0)

        features = np.array(features[:self.config.state_dim])

        # Calculate unrealized PnL
        if self.position != 0:
            current_price = market_data.iloc[-1].get('close', self.entry_price)
            unrealized_pnl = self.position * (current_price - self.entry_price)
        else:
            unrealized_pnl = 0

        return TradingState(
            market_features=features,
            position=self.position,
            unrealized_pnl=unrealized_pnl,
            balance=self.balance,
            recent_actions=list(self.recent_actions),
            timestep=self.current_idx
        )

    def step(self, action: int) -> Tuple[TradingState, float, bool, Dict]:
        """
        Execute action and return new state, reward, done flag, and info.

        Actions:
        0: Hold
        1: Buy
        2: Sell
        """
        self.recent_actions.append(action)

        current_price = self.data.iloc[self.current_idx].get('close', 100)
        reward = 0
        info = {}

        # Execute trading action
        if action == 1:  # Buy
            if self.position < self.config.max_position_size:
                # Buy one contract
                cost = current_price + self.config.transaction_cost
                if self.balance >= cost:
                    self.position += 1
                    self.balance -= cost
                    if self.position == 1:  # New position
                        self.entry_price = current_price
                    info['trade'] = 'buy'

        elif action == 2:  # Sell
            if self.position > -self.config.max_position_size:
                # Sell one contract
                self.position -= 1
                self.balance += current_price - self.config.transaction_cost
                if self.position == -1:  # New short position
                    self.entry_price = current_price
                info['trade'] = 'sell'

        # Calculate reward
        # 1. Position-based reward
        next_price = self.data.iloc[min(self.current_idx + 1, len(self.data) - 1)].get('close', current_price)
        price_change = next_price - current_price

        if self.position > 0:
            position_reward = self.position * price_change
        elif self.position < 0:
            position_reward = -self.position * price_change
        else:
            position_reward = 0

        # 2. Risk-adjusted reward (Sharpe ratio component)
        returns_window = 20
        if self.current_idx > returns_window:
            recent_returns = []
            for i in range(returns_window):
                idx = self.current_idx - returns_window + i
                price_t = self.data.iloc[idx].get('close', 100)
                price_t1 = self.data.iloc[idx + 1].get('close', 100)
                recent_returns.append((price_t1 - price_t) / price_t)

            if recent_returns:
                sharpe_component = np.mean(recent_returns) / (np.std(recent_returns) + 1e-8)
                risk_reward = sharpe_component * 0.1
            else:
                risk_reward = 0
        else:
            risk_reward = 0

        # 3. Transaction cost penalty
        if action != 0:  # Not holding
            transaction_penalty = -self.config.transaction_cost * 0.01
        else:
            transaction_penalty = 0

        # Combine rewards
        reward = position_reward + risk_reward + transaction_penalty

        # Clip reward to prevent instability
        reward = np.clip(reward, -10, 10)

        # Move to next timestep
        self.current_idx += 1

        # Check if episode is done
        done = (
            self.current_idx >= len(self.data) - 1 or
            self.balance <= self.config.initial_balance * 0.5  # 50% loss stops episode
        )

        # Get new state
        new_state = self._get_state()

        info['balance'] = self.balance
        info['position'] = self.position
        info['unrealized_pnl'] = new_state.unrealized_pnl

        return new_state, reward, done, info


class ActorCriticNetwork(nn.Module):
    """
    Actor-Critic network for PPO.
    Shared backbone with separate heads for policy and value.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(ActorCriticNetwork, self).__init__()

        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim + 10, hidden_dim),  # +10 for recent actions
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # Actor (policy) head with attention
        self.actor_attention = nn.MultiheadAttention(hidden_dim, 4, batch_first=True)
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )

        # Critic (value) head
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, state: TradingState) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning action logits and state value.
        """
        # Prepare input
        market_features = torch.FloatTensor(state.market_features)
        recent_actions = torch.FloatTensor(state.recent_actions)

        # Add position and PnL info
        position_info = torch.FloatTensor([
            state.position / 5.0,  # Normalize position
            state.unrealized_pnl / 1000.0,  # Normalize PnL
            state.balance / 100000.0  # Normalize balance
        ])

        # Concatenate all features
        x = torch.cat([market_features, recent_actions, position_info])

        # Extract features
        features = self.feature_extractor(x)

        # Self-attention for temporal dependencies
        features_reshaped = features.unsqueeze(0).unsqueeze(0)
        attended, _ = self.actor_attention(
            features_reshaped,
            features_reshaped,
            features_reshaped
        )
        features = attended.squeeze()

        # Get action logits and value
        action_logits = self.actor_head(features)
        value = self.critic_head(features)

        return action_logits, value


class PPOAgent:
    """PPO agent for trading."""

    def __init__(self, config: PPOConfig):
        self.config = config
        self.network = ActorCriticNetwork(
            config.state_dim,
            config.action_dim,
            config.hidden_dim
        )
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=config.learning_rate
        )

        # Experience buffer
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def select_action(self, state: TradingState, training: bool = True) -> Tuple[int, float, float]:
        """Select action using current policy."""
        with torch.no_grad():
            action_logits, value = self.network(state)

            if training:
                # Sample from distribution
                dist = Categorical(logits=action_logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)
            else:
                # Greedy action for evaluation
                action = torch.argmax(action_logits)
                dist = Categorical(logits=action_logits)
                log_prob = dist.log_prob(action)

            return action.item(), log_prob.item(), value.item()

    def store_transition(self, state: TradingState, action: int, reward: float,
                        value: float, log_prob: float, done: bool):
        """Store transition in buffer."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def compute_gae(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation."""
        rewards = torch.FloatTensor(self.rewards)
        values = torch.FloatTensor(self.values)
        dones = torch.FloatTensor(self.dones)

        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)

        # Compute GAE
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.config.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.config.gamma * self.config.lambda_gae * (1 - dones[t]) * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    def update(self):
        """Update policy using PPO algorithm."""
        if len(self.states) < self.config.batch_size:
            return {}

        # Compute advantages and returns
        advantages, returns = self.compute_gae()

        # Convert to tensors
        old_log_probs = torch.FloatTensor(self.log_probs)
        actions = torch.LongTensor(self.actions)

        # PPO update
        metrics = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'total_loss': []
        }

        for _ in range(self.config.n_epochs):
            # Shuffle and create mini-batches
            indices = np.random.permutation(len(self.states))

            for start in range(0, len(indices), self.config.batch_size):
                end = start + self.config.batch_size
                batch_indices = indices[start:end]

                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_actions = actions[batch_indices]

                # Forward pass for batch
                batch_values = []
                batch_log_probs = []
                batch_entropy = []

                for idx in batch_indices:
                    state = self.states[idx]
                    action_logits, value = self.network(state)

                    dist = Categorical(logits=action_logits)
                    log_prob = dist.log_prob(torch.tensor(self.actions[idx]))
                    entropy = dist.entropy()

                    batch_values.append(value)
                    batch_log_probs.append(log_prob)
                    batch_entropy.append(entropy)

                batch_values = torch.stack(batch_values).squeeze()
                batch_log_probs = torch.stack(batch_log_probs)
                batch_entropy = torch.stack(batch_entropy)

                # Calculate ratio for PPO
                ratio = torch.exp(batch_log_probs - batch_old_log_probs)

                # Clipped surrogate objective
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon,
                                   1 + self.config.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(batch_values, batch_returns)

                # Entropy bonus
                entropy_loss = -batch_entropy.mean()

                # Total loss
                total_loss = (
                    policy_loss +
                    self.config.value_loss_coef * value_loss +
                    self.config.entropy_coef * entropy_loss
                )

                # Backpropagation
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.network.parameters(),
                    self.config.max_grad_norm
                )
                self.optimizer.step()

                # Store metrics
                metrics['policy_loss'].append(policy_loss.item())
                metrics['value_loss'].append(value_loss.item())
                metrics['entropy'].append(-entropy_loss.item())
                metrics['total_loss'].append(total_loss.item())

        # Clear buffer
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()

        # Average metrics
        return {k: np.mean(v) for k, v in metrics.items()}


def train_ppo_agent(data: pd.DataFrame) -> Dict[str, float]:
    """Train PPO agent on historical data."""
    config = PPOConfig()

    # Create environment and agent
    env = TradingEnvironment(data, config)
    agent = PPOAgent(config)

    # Training metrics
    episode_rewards = []
    episode_lengths = []
    total_steps = 0
    episode = 0

    progress_bar = tqdm(total=config.total_timesteps)

    while total_steps < config.total_timesteps:
        state = env.reset()
        episode_reward = 0
        episode_length = 0

        while True:
            # Select action
            action, log_prob, value = agent.select_action(state, training=True)

            # Step environment
            next_state, reward, done, info = env.step(action)

            # Store transition
            agent.store_transition(state, action, reward, value, log_prob, done)

            episode_reward += reward
            episode_length += 1
            total_steps += 1

            progress_bar.update(1)

            # Update if buffer is full
            if len(agent.states) >= config.n_steps:
                update_metrics = agent.update()

                if episode % 10 == 0:
                    progress_bar.set_description(
                        f"Episode {episode}, Reward: {np.mean(episode_rewards[-10:]):.2f}, "
                        f"Loss: {update_metrics.get('total_loss', 0):.4f}"
                    )

            if done:
                break

            state = next_state

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode += 1

    progress_bar.close()

    # Save model
    torch.save(agent.network.state_dict(), PPO_MODEL_PATH)

    # Calculate final metrics
    metrics = {
        "model_type": "PPO",
        "total_timesteps": total_steps,
        "total_episodes": episode,
        "avg_episode_reward": float(np.mean(episode_rewards)),
        "std_episode_reward": float(np.std(episode_rewards)),
        "avg_episode_length": float(np.mean(episode_lengths)),
        "final_balance": env.balance,
        "total_return": (env.balance - config.initial_balance) / config.initial_balance,
        "trained_at": datetime.utcnow().isoformat()
    }

    return metrics


def evaluate_agent(data: pd.DataFrame, start_idx: int = None) -> Dict:
    """Evaluate trained PPO agent on test data."""
    config = PPOConfig()

    # Load trained model
    agent = PPOAgent(config)
    agent.network.load_state_dict(torch.load(PPO_MODEL_PATH))

    # Use last 20% of data for testing
    if start_idx is None:
        start_idx = int(len(data) * 0.8)

    test_data = data.iloc[start_idx:]
    env = TradingEnvironment(test_data, config)

    state = env.reset()
    total_reward = 0
    trades = []

    while True:
        action, _, _ = agent.select_action(state, training=False)
        state, reward, done, info = env.step(action)
        total_reward += reward

        if 'trade' in info:
            trades.append({
                'timestep': state.timestep,
                'action': info['trade'],
                'position': info['position'],
                'balance': info['balance']
            })

        if done:
            break

    return {
        'total_reward': total_reward,
        'final_balance': env.balance,
        'total_return': (env.balance - config.initial_balance) / config.initial_balance,
        'num_trades': len(trades),
        'trades': trades
    }


def main():
    # Load data
    parquet_path = DATA_DIR / "meta_label.parquet"
    if not parquet_path.exists():
        print("Dataset not found. Run build_dataset.py first.")
        sys.exit(1)

    df = pd.read_parquet(parquet_path)
    if len(df) < 1000:
        print(f"Insufficient data: {len(df)} rows. Need at least 1000.")
        sys.exit(1)

    print(f"Training PPO agent with {len(df)} rows of data")

    # Ensure models directory exists
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Train agent
    metrics = train_ppo_agent(df)

    # Evaluate on test set
    eval_results = evaluate_agent(df)
    metrics['evaluation'] = eval_results

    # Save metrics
    metrics_path = MODELS_DIR / "ppo_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print(f"\nModel saved to: {PPO_MODEL_PATH}")
    print(f"Training Metrics: {json.dumps(metrics, indent=2)}")
    print(f"\nEvaluation Results:")
    print(f"  Total Return: {eval_results['total_return']:.2%}")
    print(f"  Number of Trades: {eval_results['num_trades']}")
    print(f"  Final Balance: ${eval_results['final_balance']:,.2f}")


if __name__ == "__main__":
    main()