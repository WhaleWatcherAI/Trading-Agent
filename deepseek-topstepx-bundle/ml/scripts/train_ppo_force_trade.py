#!/usr/bin/env python3
"""
PPO that's forced to trade more actively through reward engineering.
Penalizes inactivity and rewards position changes.
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

class ForcedTradeConfig:
    """Config that encourages trading."""
    state_dim = 25
    action_dim = 5  # Buy Strong, Buy, Hold, Sell, Sell Strong
    hidden_dim = 128

    # Environment
    initial_balance = 10000
    transaction_cost = 0.0001  # Small cost to be realistic
    min_trades_per_episode = 20  # Force minimum trades

    # Rewards that force trading
    inactivity_penalty = -0.5  # Penalty for not changing position
    trade_bonus = 1.0  # Bonus for making a trade
    diversity_bonus = 2.0  # Bonus for using different actions

    # PPO hyperparameters
    learning_rate = 1e-4
    gamma = 0.99
    lambda_gae = 0.95
    clip_epsilon = 0.2
    value_loss_coef = 0.5
    entropy_coef = 0.05  # Higher entropy for diversity
    max_grad_norm = 0.5

    # Training
    batch_size = 256
    n_epochs = 10
    n_steps = 512
    total_timesteps = 75000

class ActiveTradingNetwork(nn.Module):
    """Network for active trading."""

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

        # Initialize to encourage diverse actions
        nn.init.normal_(self.actor.weight, std=0.1)
        nn.init.zeros_(self.actor.bias)

    def forward(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state)
        if state.dim() == 1:
            state = state.unsqueeze(0)

        features = self.fc(state)
        action_logits = self.actor(features)
        value = self.critic(features)

        return action_logits.squeeze(0), value.squeeze()

class ForcedTradeEnv:
    """Environment that rewards trading activity."""

    def __init__(self, data, config):
        self.data = data
        self.config = config
        self.reset()

    def reset(self):
        """Reset environment."""
        self.current_idx = 50
        self.balance = self.config.initial_balance
        self.position = 0  # -2 to 2
        self.entry_price = 0
        self.trades = []
        self.returns = []

        # Track activity
        self.bars_since_trade = 0
        self.action_counts = {i: 0 for i in range(5)}
        self.last_action = 2  # Start with Hold

        # Calculate indicators
        self._calculate_indicators()

        return self._get_state()

    def _calculate_indicators(self):
        """Calculate basic indicators."""
        df = self.data

        # Basic indicators
        df['returns'] = df['close'].pct_change()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['rsi'] = self._calculate_rsi(df['close'], 14)
        df['volatility'] = df['returns'].rolling(20).std()

        # MACD
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

        # Bollinger Bands
        df['bb_mid'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_mid'] + (bb_std * 2)
        df['bb_lower'] = df['bb_mid'] - (bb_std * 2)

        self.data = df.fillna(method='ffill').fillna(0)

    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _get_state(self):
        """Get current state."""
        idx = min(self.current_idx, len(self.data) - 1)
        row = self.data.iloc[idx]

        price = row['close']

        features = [
            # Price action
            row.get('returns', 0) * 100,
            (price - row.get('sma_20', price)) / price * 100,
            (price - row.get('sma_50', price)) / price * 100,

            # Indicators
            row.get('rsi', 50) / 100,
            row.get('macd', 0) / price * 100 if price > 0 else 0,
            (row.get('macd', 0) - row.get('macd_signal', 0)) / price * 100 if price > 0 else 0,

            # Bollinger position
            (price - row.get('bb_lower', price)) / (row.get('bb_upper', price) - row.get('bb_lower', price) + 0.0001),

            # Volatility
            row.get('volatility', 0.01) * 100,

            # Position info
            self.position / 2,
            (price - self.entry_price) / self.entry_price * 100 if self.position != 0 and self.entry_price > 0 else 0,

            # Activity tracking (important!)
            min(self.bars_since_trade / 20, 1.0),  # Normalized bars since last trade
            len(self.trades) / self.config.min_trades_per_episode,  # Trade progress

            # Action diversity
            self.action_counts[0] / max(1, sum(self.action_counts.values())),
            self.action_counts[1] / max(1, sum(self.action_counts.values())),
            self.action_counts[2] / max(1, sum(self.action_counts.values())),
            self.action_counts[3] / max(1, sum(self.action_counts.values())),
            self.action_counts[4] / max(1, sum(self.action_counts.values())),

            # Performance
            sum(1 for r in self.returns if r > 0) / max(1, len(self.returns)),
            (self.balance - self.config.initial_balance) / self.config.initial_balance * 100,

            # Market timing
            self.current_idx / len(self.data),

            # Recent momentum
            self.data.iloc[max(0, idx-5):idx+1]['close'].pct_change().mean() * 100 if idx > 5 else 0,
            self.data.iloc[max(0, idx-10):idx+1]['close'].pct_change().mean() * 100 if idx > 10 else 0,

            # Trend
            1.0 if row.get('sma_20', 0) > row.get('sma_50', 0) else -1.0,

            # Volume (if available)
            row.get('volume', 1) / self.data['volume'].rolling(20).mean().iloc[idx] if 'volume' in self.data.columns else 1.0,

            # Last action taken
            self.last_action / 4.0,
        ]

        # Ensure correct size
        features = features[:self.config.state_dim]
        while len(features) < self.config.state_dim:
            features.append(0)

        return np.array(features, dtype=np.float32)

    def step(self, action):
        """Execute action with forced trading mechanics."""
        idx = min(self.current_idx, len(self.data) - 1)
        price = self.data.iloc[idx]['close']

        old_position = self.position
        target_position = action - 2  # Maps 0-4 to -2 to 2

        reward = 0

        # Track action
        self.action_counts[action] += 1
        self.last_action = action

        # Check if position changed
        position_changed = (target_position != old_position)

        if position_changed:
            # TRADE EXECUTED - Reset inactivity counter
            self.bars_since_trade = 0

            # Trade bonus for activity
            reward += self.config.trade_bonus

            # Close old position
            if old_position != 0:
                if old_position > 0:  # Was long
                    returns = (price - self.entry_price) / self.entry_price * abs(old_position)
                else:  # Was short
                    returns = (self.entry_price - price) / self.entry_price * abs(old_position)

                # Transaction cost
                returns -= self.config.transaction_cost

                self.returns.append(returns)
                self.balance *= (1 + returns)

                # Profit/loss reward
                reward += returns * 100

            # Open new position
            if target_position != 0:
                self.position = target_position
                self.entry_price = price
                self.trades.append({
                    'idx': idx,
                    'action': action,
                    'price': price,
                    'position': target_position
                })
        else:
            # NO TRADE - Increment inactivity counter
            self.bars_since_trade += 1

            # Inactivity penalty (gets worse over time)
            if self.bars_since_trade > 10:
                reward += self.config.inactivity_penalty * (self.bars_since_trade / 10)

        # Continuous position reward/penalty
        if self.position != 0:
            if self.position > 0:
                unrealized = (price - self.entry_price) / self.entry_price * abs(self.position)
            else:
                unrealized = (self.entry_price - price) / self.entry_price * abs(self.position)
            reward += unrealized * 20

        # Diversity bonus (reward using different actions)
        action_diversity = len([a for a in self.action_counts.values() if a > 0])
        if action_diversity >= 4:
            reward += self.config.diversity_bonus

        # Move to next bar
        self.current_idx += 1

        # Check if done
        done = self.current_idx >= len(self.data) - 1

        if done:
            # Close final position
            if self.position != 0:
                if self.position > 0:
                    returns = (price - self.entry_price) / self.entry_price * abs(self.position)
                else:
                    returns = (self.entry_price - price) / self.entry_price * abs(self.position)
                self.returns.append(returns)
                self.balance *= (1 + returns)

            # Final rewards/penalties
            total_return = (self.balance - self.config.initial_balance) / self.config.initial_balance
            reward += total_return * 200  # Profit reward

            # Penalty if didn't meet minimum trades
            if len(self.trades) < self.config.min_trades_per_episode:
                penalty = (self.config.min_trades_per_episode - len(self.trades)) * 5
                reward -= penalty

            # Sharpe ratio bonus
            if len(self.returns) > 0:
                sharpe = np.mean(self.returns) / (np.std(self.returns) + 1e-6)
                reward += sharpe * 30

        # Get new state
        new_state = self._get_state()

        info = {
            'balance': self.balance,
            'total_return': (self.balance - self.config.initial_balance) / self.config.initial_balance,
            'n_trades': len(self.trades),
            'win_rate': sum(1 for r in self.returns if r > 0) / max(1, len(self.returns)),
            'position': self.position,
            'bars_since_trade': self.bars_since_trade
        }

        return new_state, reward, done, info

class ForcedTradeAgent:
    """PPO agent forced to trade."""

    def __init__(self, config):
        self.config = config
        self.network = ActiveTradingNetwork(config.state_dim, config.action_dim, config.hidden_dim)
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

            # Add temperature for more exploration
            temperature = 1.2
            action_logits = action_logits / temperature

            dist = Categorical(logits=action_logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            return action.item(), log_prob.item(), value.item()

    def update(self):
        """PPO update."""
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

        # Calculate returns and advantages
        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)

        running_return = 0
        running_advantage = 0

        for t in reversed(range(len(rewards))):
            if dones[t]:
                running_return = 0
                running_advantage = 0

            running_return = rewards[t] + self.config.gamma * running_return
            returns[t] = running_return

            if t < len(rewards) - 1:
                td_error = rewards[t] + self.config.gamma * old_values[t + 1] * (1 - dones[t]) - old_values[t]
            else:
                td_error = rewards[t] - old_values[t]

            running_advantage = td_error + self.config.gamma * self.config.lambda_gae * running_advantage * (1 - dones[t])
            advantages[t] = running_advantage

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

            # Entropy (important for diversity)
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

def train_forced():
    """Train PPO with forced trading."""

    print("\n" + "="*70)
    print("🚀 PPO WITH FORCED TRADING ACTIVITY")
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
    config = ForcedTradeConfig()
    env = ForcedTradeEnv(train_data, config)
    agent = ForcedTradeAgent(config)

    print(f"\n⚡ Forced Trading Configuration:")
    print(f"  Minimum trades per episode: {config.min_trades_per_episode}")
    print(f"  Inactivity penalty: {config.inactivity_penalty}")
    print(f"  Trade bonus: {config.trade_bonus}")
    print(f"  Training steps: {config.total_timesteps:,}")

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

    test_env = ForcedTradeEnv(test_data, config)
    state = test_env.reset()

    actions_taken = []

    for _ in range(len(test_data) - 51):
        action, _, _ = agent.select_action(state, deterministic=True)
        actions_taken.append(action)
        state, _, done, info = test_env.step(action)
        if done:
            break

    test_return = info['total_return']

    # Analyze actions
    action_counts = {i: actions_taken.count(i) for i in range(5)}
    action_names = ['Buy Strong', 'Buy', 'Hold', 'Sell', 'Sell Strong']

    print(f"\n🎯 Test Results:")
    print(f"  Return: {test_return:.2%}")
    print(f"  Final Balance: ${info['balance']:.2f}")
    print(f"  Trades: {info['n_trades']} ({'✅ Met minimum' if info['n_trades'] >= config.min_trades_per_episode else '❌ Below minimum'})")
    print(f"  Win Rate: {info['win_rate']:.1%}")

    print(f"\n📊 Action Distribution:")
    for i, name in enumerate(action_names):
        pct = action_counts[i] / len(actions_taken) * 100 if actions_taken else 0
        print(f"  {name:12s}: {pct:5.1f}% ({action_counts[i]:4d} times)")

    # Save model
    torch.save(agent.network.state_dict(), MODELS_DIR / 'ppo_forced_trade.pth')

    # Save metrics
    metrics = {
        'model': 'PPO_Forced_Trading',
        'test_return': float(test_return),
        'test_trades': info['n_trades'],
        'test_win_rate': float(info['win_rate']),
        'test_balance': float(info['balance']),
        'min_trades_required': config.min_trades_per_episode,
        'action_distribution': {name: action_counts[i]/len(actions_taken) for i, name in enumerate(action_names)} if actions_taken else {},
        'trained_at': datetime.utcnow().isoformat()
    }

    with open(MODELS_DIR / 'ppo_forced_trade_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "="*70)
    if test_return > 0:
        print(f"✅ SUCCESS! Achieved {test_return:.2%} return with {info['n_trades']} trades!")
    else:
        print(f"📊 Result: {test_return:.2%} return with {info['n_trades']} trades")
    print("="*70)

    return agent, metrics

if __name__ == "__main__":
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    agent, metrics = train_forced()