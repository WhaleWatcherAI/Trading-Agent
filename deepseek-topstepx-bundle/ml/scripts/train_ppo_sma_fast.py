#!/usr/bin/env python3
"""
Simplified and faster PPO training on 9-SMA crossover strategy.
Reduced complexity for quicker training and debugging.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "ml" / "data"
MODELS_DIR = ROOT / "ml" / "models"

class SimpleConfig:
    """Simplified config for faster training."""
    # Reduced dimensions
    state_dim = 15  # Fewer features
    action_dim = 3  # Just: Ignore, Buy, Sell
    hidden_dim = 128  # Smaller network

    # Environment
    initial_balance = 10000
    transaction_cost = 0.0

    # PPO hyperparameters
    learning_rate = 1e-4
    gamma = 0.99
    lambda_gae = 0.95
    clip_epsilon = 0.2
    value_loss_coef = 0.5
    entropy_coef = 0.01
    max_grad_norm = 1.0

    # Training - much less for faster completion
    batch_size = 256
    n_epochs = 5
    n_steps = 512
    total_timesteps = 50000  # 10x less than before

class SimpleNetwork(nn.Module):
    """Simplified network architecture."""

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state)
        if state.dim() == 1:
            state = state.unsqueeze(0)

        features = self.fc(state)
        action_logits = self.actor(features)
        value = self.critic(features)

        return action_logits.squeeze(0), value.squeeze()

class FastSMAEnv:
    """Simplified environment for faster execution."""

    def __init__(self, data, config):
        self.data = data
        self.config = config

        # Pre-calculate all SMAs
        self.data['sma_9'] = self.data['close'].rolling(window=9, min_periods=1).mean()
        self.data['sma_20'] = self.data['close'].rolling(window=20, min_periods=1).mean()

        # Pre-calculate crossovers
        self.data['signal_buy'] = (
            (self.data['close'] > self.data['sma_9']) &
            (self.data['close'].shift(1) <= self.data['sma_9'].shift(1))
        ).astype(float)

        self.data['signal_sell'] = (
            (self.data['close'] < self.data['sma_9']) &
            (self.data['close'].shift(1) >= self.data['sma_9'].shift(1))
        ).astype(float)

        self.reset()

    def reset(self):
        """Reset environment."""
        self.current_idx = 30
        self.balance = self.config.initial_balance
        self.position = 0  # 1 = long, -1 = short, 0 = neutral
        self.entry_price = 0
        self.trades = []
        self.returns = []

        return self._get_state()

    def _get_state(self):
        """Simplified state with fewer features."""
        idx = min(self.current_idx, len(self.data) - 1)
        row = self.data.iloc[idx]

        price = row['close']
        sma_9 = row['sma_9']
        sma_20 = row['sma_20']

        # Core features
        features = [
            # Price relative to SMAs
            (price - sma_9) / price,
            (price - sma_20) / price,
            (sma_9 - sma_20) / sma_9,

            # Signals
            row.get('signal_buy', 0),
            row.get('signal_sell', 0),

            # Recent returns
            self.data['close'].iloc[max(0, idx-5):idx+1].pct_change().mean(),
            self.data['close'].iloc[max(0, idx-10):idx+1].pct_change().mean(),
            self.data['close'].iloc[max(0, idx-20):idx+1].pct_change().mean(),

            # Volatility
            self.data['close'].iloc[max(0, idx-20):idx+1].pct_change().std(),

            # RSI approximation
            row.get('rsi', 50) / 100 if 'rsi' in row else 0.5,

            # Volume ratio
            row.get('volume', 1) / self.data['volume'].iloc[max(0, idx-20):idx+1].mean() if 'volume' in self.data.columns else 1.0,

            # Position info
            self.position,
            (price - self.entry_price) / self.entry_price if self.position != 0 and self.entry_price > 0 else 0,

            # Market timing
            idx / len(self.data),

            # Win rate so far
            sum(1 for r in self.returns if r > 0) / max(1, len(self.returns))
        ]

        # Ensure correct size
        features = features[:self.config.state_dim]
        while len(features) < self.config.state_dim:
            features.append(0)

        return np.array(features, dtype=np.float32)

    def step(self, action):
        """
        Execute action.
        0: Ignore signal
        1: Take buy signal (go long)
        2: Take sell signal (go short)
        """
        idx = min(self.current_idx, len(self.data) - 1)
        row = self.data.iloc[idx]
        price = row['close']

        has_buy_signal = row.get('signal_buy', 0) > 0.5
        has_sell_signal = row.get('signal_sell', 0) > 0.5

        reward = 0

        # Execute action
        if action == 1 and has_buy_signal:  # Take buy signal
            if self.position <= 0:  # Not long
                # Close short if exists
                if self.position < 0:
                    returns = (self.entry_price - price) / self.entry_price
                    self.returns.append(returns)
                    reward = returns * 100
                    self.balance *= (1 + returns)

                # Go long
                self.position = 1
                self.entry_price = price
                self.trades.append({'idx': idx, 'action': 'buy', 'price': price})

        elif action == 2 and has_sell_signal:  # Take sell signal
            if self.position >= 0:  # Not short
                # Close long if exists
                if self.position > 0:
                    returns = (price - self.entry_price) / self.entry_price
                    self.returns.append(returns)
                    reward = returns * 100
                    self.balance *= (1 + returns)

                # Go short
                self.position = -1
                self.entry_price = price
                self.trades.append({'idx': idx, 'action': 'sell', 'price': price})

        # Position P&L
        if self.position != 0:
            if self.position > 0:
                unrealized = (price - self.entry_price) / self.entry_price
            else:
                unrealized = (self.entry_price - price) / self.entry_price
            reward += unrealized * 10  # Encourage good positions

        # Move to next bar
        self.current_idx += 1

        # Check if done
        done = self.current_idx >= len(self.data) - 1

        if done:
            # Close final position
            if self.position != 0:
                if self.position > 0:
                    returns = (price - self.entry_price) / self.entry_price
                else:
                    returns = (self.entry_price - price) / self.entry_price
                self.returns.append(returns)
                self.balance *= (1 + returns)

            # Final reward based on total return
            total_return = (self.balance - self.config.initial_balance) / self.config.initial_balance
            reward += total_return * 200

        # Get new state
        new_state = self._get_state()

        info = {
            'balance': self.balance,
            'total_return': (self.balance - self.config.initial_balance) / self.config.initial_balance,
            'n_trades': len(self.trades),
            'win_rate': sum(1 for r in self.returns if r > 0) / max(1, len(self.returns))
        }

        return new_state, reward, done, info

class SimplePPOAgent:
    """Simplified PPO agent."""

    def __init__(self, config):
        self.config = config
        self.network = SimpleNetwork(config.state_dim, config.action_dim, config.hidden_dim)
        self.optimizer = optim.Adam(self.network.parameters(), lr=config.learning_rate)
        self.clear_buffers()

    def clear_buffers(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def select_action(self, state, deterministic=False):
        with torch.no_grad():
            action_logits, value = self.network(state)

            if deterministic:
                action = torch.argmax(action_logits).item()
                return action, 0, value.item()

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

        # Normalize rewards
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        # Calculate returns
        returns = torch.zeros_like(rewards)
        running_return = 0
        for t in reversed(range(len(rewards))):
            if dones[t]:
                running_return = 0
            running_return = rewards[t] + self.config.gamma * running_return
            returns[t] = running_return

        # Calculate advantages
        advantages = returns - old_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO epochs
        for _ in range(self.config.n_epochs):
            # Forward pass
            action_logits, values = self.network(states)
            dist = Categorical(logits=action_logits)
            log_probs = dist.log_prob(actions)

            # PPO loss
            ratio = torch.exp(log_probs - old_log_probs)
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

        self.clear_buffers()
        return {'loss': loss.item()}

def train_fast():
    """Fast training of simplified SMA strategy."""

    print("\n" + "="*70)
    print("🚀 FAST PPO TRAINING ON 9-SMA CROSSOVER")
    print("="*70)

    # Load data
    data_path = DATA_DIR / 'spy_5min_data.parquet'
    if not data_path.exists():
        print("❌ Data not found!")
        return None

    df = pd.read_parquet(data_path)
    print(f"📊 Loaded {len(df)} 5-minute bars")

    # Split data
    train_size = int(len(df) * 0.8)
    train_data = df.iloc[:train_size].copy()
    test_data = df.iloc[train_size:].copy()

    print(f"  Train: {len(train_data)} bars")
    print(f"  Test: {len(test_data)} bars")

    # Initialize
    config = SimpleConfig()
    env = FastSMAEnv(train_data, config)
    agent = SimplePPOAgent(config)

    print(f"\n⚡ Fast Configuration:")
    print(f"  Total steps: {config.total_timesteps:,}")
    print(f"  State dim: {config.state_dim}")
    print(f"  Actions: Ignore, Buy, Sell")

    # Training
    episode_returns = []
    episode_trades = []

    pbar = tqdm(total=config.total_timesteps, desc="Training")
    total_steps = 0

    while total_steps < config.total_timesteps:
        state = env.reset()
        episode_reward = 0

        while True:
            action, log_prob, value = agent.select_action(state)
            next_state, reward, done, info = env.step(action)

            agent.states.append(state)
            agent.actions.append(action)
            agent.rewards.append(reward)
            agent.values.append(value)
            agent.log_probs.append(log_prob)
            agent.dones.append(done)

            episode_reward += reward
            total_steps += 1
            pbar.update(1)

            if len(agent.states) >= config.n_steps:
                agent.update()

            if done:
                episode_returns.append(info['total_return'])
                episode_trades.append(info['n_trades'])

                if len(episode_returns) % 10 == 0:
                    avg_return = np.mean(episode_returns[-10:])
                    avg_trades = np.mean(episode_trades[-10:])
                    pbar.set_description(f"Return: {avg_return:.2%} | Trades: {avg_trades:.0f}")

                break

            state = next_state

    pbar.close()

    # Test
    print("\n📊 TESTING ON HOLDOUT DATA")

    test_env = FastSMAEnv(test_data, config)
    state = test_env.reset()

    for _ in range(len(test_data) - 31):
        action, _, _ = agent.select_action(state, deterministic=True)
        state, _, done, info = test_env.step(action)
        if done:
            break

    test_return = info['total_return']

    print(f"\n🎯 Test Results:")
    print(f"  Return: {test_return:.2%}")
    print(f"  Final Balance: ${info['balance']:.2f}")
    print(f"  Trades: {info['n_trades']}")
    print(f"  Win Rate: {info['win_rate']:.1%}")

    # Save model
    torch.save(agent.network.state_dict(), MODELS_DIR / 'ppo_sma_fast.pth')

    # Save metrics
    metrics = {
        'model': 'PPO_SMA_Fast',
        'test_return': float(test_return),
        'test_trades': info['n_trades'],
        'test_win_rate': float(info['win_rate']),
        'trained_at': datetime.utcnow().isoformat()
    }

    with open(MODELS_DIR / 'ppo_sma_fast_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "="*70)
    if test_return > 0:
        print(f"✅ SUCCESS! Achieved {test_return:.2%} return!")
    else:
        print(f"📊 Result: {test_return:.2%} return")
    print("="*70)

    return agent, metrics

if __name__ == "__main__":
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    agent, metrics = train_fast()