#!/usr/bin/env python3
"""
Improved PPO training with 5-minute bars and short selling capability.
Better reward shaping, proper position management, no transaction costs.
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
import warnings
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "ml" / "data"
MODELS_DIR = ROOT / "ml" / "models"

class PPOConfig:
    """Configuration for improved PPO."""
    # State and action space
    state_dim = 20  # More features for better decisions
    action_dim = 5  # Hold, Buy, Sell, Short, Cover
    hidden_dim = 256  # Larger network

    # Environment
    initial_balance = 10000
    max_position = 1.0  # Max 100% long or short
    transaction_cost = 0.0  # No transaction costs

    # PPO hyperparameters
    learning_rate = 1e-4
    gamma = 0.99
    lambda_gae = 0.95
    clip_epsilon = 0.2
    value_loss_coef = 0.5
    entropy_coef = 0.02
    max_grad_norm = 0.5

    # Training
    batch_size = 128
    n_epochs = 4
    n_steps = 2048
    total_timesteps = 100000

class ImprovedActorCritic(nn.Module):
    """Improved Actor-Critic with better architecture."""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(ImprovedActorCritic, self).__init__()

        # Shared network with more capacity
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )

        # Actor head
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, action_dim)
        )

        # Critic head
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)
        )

    def forward(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state)

        if state.dim() == 1:
            state = state.unsqueeze(0)

        features = self.shared(state)
        action_logits = self.actor(features)
        value = self.critic(features)

        return action_logits.squeeze(0), value.squeeze()

class TradingEnv5Min:
    """Trading environment with 5-minute bars and short selling."""

    def __init__(self, data, config, lookback=20):
        """
        Args:
            data: DataFrame with 5-minute bars and indicators
            config: PPOConfig object
            lookback: Number of bars to use for state features
        """
        self.data = data
        self.config = config
        self.lookback = lookback
        self.reset()

    def reset(self):
        """Reset environment to initial state."""
        # Start after enough history
        self.current_idx = self.lookback + 50
        self.balance = self.config.initial_balance
        self.position = 0.0  # Can be negative (short)
        self.entry_price = 0
        self.trades = []
        self.equity_curve = [self.config.initial_balance]

        return self._get_state()

    def _get_state(self):
        """Create state vector with market features and position info."""
        if self.current_idx >= len(self.data):
            self.current_idx = len(self.data) - 1

        # Get recent price data
        recent_data = self.data.iloc[self.current_idx - self.lookback:self.current_idx]
        current_bar = self.data.iloc[self.current_idx]

        # Market features (normalized)
        features = [
            # Price features
            current_bar.get('returns', 0) * 100,
            (current_bar['close'] - recent_data['close'].mean()) / recent_data['close'].std(),

            # Technical indicators
            current_bar.get('rsi', 50) / 100,
            current_bar.get('macd', 0),
            current_bar.get('macd_signal', 0),
            current_bar.get('bb_position', 0.5),

            # Volume
            current_bar.get('volume_ratio', 1),

            # Volatility
            current_bar.get('volatility', 0.01) * 100,
            current_bar.get('atr', 1) / current_bar['close'],

            # Trend (using recent prices)
            (current_bar['close'] - recent_data['close'].iloc[0]) / recent_data['close'].iloc[0],
            (current_bar['sma_20'] - current_bar['sma_50']) / current_bar['close'] if 'sma_20' in current_bar else 0,

            # Position features
            self.position,  # Current position (-1 to 1)
            (self.balance - self.config.initial_balance) / self.config.initial_balance,  # P&L ratio

            # Risk features
            self._get_unrealized_pnl() / self.config.initial_balance if self.position != 0 else 0,
            abs(self.position),  # Exposure

            # Market regime (volatility percentile)
            recent_data['volatility'].rank(pct=True).iloc[-1] if 'volatility' in recent_data else 0.5,

            # Time features
            self.current_idx / len(self.data),  # Progress through dataset
            len(self.trades) / 100,  # Trade frequency (normalized)

            # Recent performance
            (self.equity_curve[-1] - self.equity_curve[max(0, len(self.equity_curve)-20)]) / self.config.initial_balance if len(self.equity_curve) > 1 else 0,

            # Momentum
            recent_data['returns'].mean() * 100 if 'returns' in recent_data else 0
        ]

        # Ensure exactly state_dim features
        if len(features) < self.config.state_dim:
            features.extend([0] * (self.config.state_dim - len(features)))
        elif len(features) > self.config.state_dim:
            features = features[:self.config.state_dim]

        return np.array(features, dtype=np.float32)

    def _get_unrealized_pnl(self):
        """Calculate unrealized P&L for current position."""
        if self.position == 0:
            return 0

        current_price = self.data.iloc[self.current_idx]['close']

        if self.position > 0:  # Long
            return (current_price - self.entry_price) * self.position * self.balance
        else:  # Short
            return (self.entry_price - current_price) * abs(self.position) * self.balance

    def _calculate_portfolio_value(self):
        """Calculate total portfolio value including positions."""
        unrealized_pnl = self._get_unrealized_pnl()
        return self.balance + unrealized_pnl

    def step(self, action):
        """
        Execute action and return next state, reward, done, info.

        Actions:
        0: Hold
        1: Buy (go long)
        2: Sell (close long)
        3: Short (go short)
        4: Cover (close short)
        """
        current_price = self.data.iloc[self.current_idx]['close']
        old_portfolio_value = self._calculate_portfolio_value()

        reward = 0
        trade_made = False

        # Execute action based on current position
        if action == 0:  # HOLD
            # Small reward for holding in volatile markets
            if self.position == 0:
                volatility = self.data.iloc[self.current_idx].get('volatility', 0.01)
                if volatility > 0.02:  # High volatility
                    reward = 0.001  # Reward for staying out

        elif action == 1:  # BUY (go long)
            if self.position <= 0:  # Can buy if flat or short
                if self.position < 0:  # Cover short first
                    pnl = (self.entry_price - current_price) * abs(self.position) * self.balance
                    self.balance += pnl
                    reward = pnl / self.config.initial_balance * 10

                # Go long
                self.position = 0.5  # 50% position
                self.entry_price = current_price
                trade_made = True

        elif action == 2:  # SELL (close long)
            if self.position > 0:
                pnl = (current_price - self.entry_price) * self.position * self.balance
                self.balance += pnl
                reward = pnl / self.config.initial_balance * 10
                self.position = 0
                trade_made = True

        elif action == 3:  # SHORT (go short)
            if self.position >= 0:  # Can short if flat or long
                if self.position > 0:  # Close long first
                    pnl = (current_price - self.entry_price) * self.position * self.balance
                    self.balance += pnl
                    reward = pnl / self.config.initial_balance * 10

                # Go short
                self.position = -0.5  # 50% short position
                self.entry_price = current_price
                trade_made = True

        elif action == 4:  # COVER (close short)
            if self.position < 0:
                pnl = (self.entry_price - current_price) * abs(self.position) * self.balance
                self.balance += pnl
                reward = pnl / self.config.initial_balance * 10
                self.position = 0
                trade_made = True

        # Record trade
        if trade_made:
            self.trades.append({
                'bar': self.current_idx,
                'action': action,
                'price': current_price,
                'position': self.position
            })

        # Move to next bar
        self.current_idx += 1

        # Additional reward shaping based on portfolio change
        new_portfolio_value = self._calculate_portfolio_value()
        portfolio_return = (new_portfolio_value - old_portfolio_value) / self.config.initial_balance
        reward += portfolio_return * 5

        # Update equity curve
        self.equity_curve.append(new_portfolio_value)

        # Penalty for drawdown
        if new_portfolio_value < self.config.initial_balance * 0.95:
            reward -= 0.01

        # Check if done
        done = (self.current_idx >= len(self.data) - 1) or (self.balance < 1000)

        # Final reward based on total return
        if done:
            total_return = (new_portfolio_value - self.config.initial_balance) / self.config.initial_balance
            reward += total_return * 50  # Big reward/penalty at end

        # Get new state
        new_state = self._get_state()

        # Info for debugging
        info = {
            'balance': self.balance,
            'position': self.position,
            'portfolio_value': new_portfolio_value,
            'total_trades': len(self.trades),
            'current_price': current_price
        }

        return new_state, reward, done, info

class PPOAgent:
    """PPO agent for improved trading."""

    def __init__(self, config):
        self.config = config
        self.network = ImprovedActorCritic(
            config.state_dim,
            config.action_dim,
            config.hidden_dim
        )
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=config.learning_rate
        )

        # Experience buffers
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def select_action(self, state, deterministic=False):
        """Select action using current policy."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state)
            action_logits, value = self.network(state_tensor)

            if deterministic:
                action = torch.argmax(action_logits).item()
                return action, 0, value.item()
            else:
                dist = Categorical(logits=action_logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                return action.item(), log_prob.item(), value.item()

    def update(self):
        """PPO update with GAE."""
        if len(self.states) < self.config.batch_size:
            return {}

        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states))
        actions = torch.LongTensor(self.actions)
        rewards = torch.FloatTensor(self.rewards)
        old_log_probs = torch.FloatTensor(self.log_probs)
        old_values = torch.FloatTensor(self.values)
        dones = torch.FloatTensor(self.dones)

        # Calculate returns and advantages using GAE
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

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update epochs
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

def train_ppo_improved():
    """Train improved PPO with 5-minute data and short selling."""

    print("\n" + "="*70)
    print("🚀 TRAINING IMPROVED PPO WITH 5-MIN DATA & SHORTING")
    print("="*70)

    # Load 5-minute data
    data_path = DATA_DIR / 'spy_5min_data.parquet'
    if not data_path.exists():
        print("❌ 5-minute data not found! Run prepare_5min_data.py first.")
        return None

    df = pd.read_parquet(data_path)
    print(f"📊 Loaded {len(df)} 5-minute bars")

    # Split data (80% train, 20% test)
    train_size = int(len(df) * 0.8)
    train_data = df.iloc[:train_size]
    test_data = df.iloc[train_size:]

    print(f"  Training on {len(train_data)} bars")
    print(f"  Testing on {len(test_data)} bars")

    # Initialize
    config = PPOConfig()
    env = TradingEnv5Min(train_data, config)
    agent = PPOAgent(config)

    # Training metrics
    episode_rewards = []
    episode_returns = []
    episode_trades = []
    losses = []

    print(f"\n📈 Configuration:")
    print(f"  State dim: {config.state_dim}")
    print(f"  Actions: Hold, Buy, Sell, Short, Cover")
    print(f"  Transaction cost: ${config.transaction_cost}")
    print(f"  Max position: {config.max_position * 100}%")

    # Training loop
    pbar = tqdm(total=config.total_timesteps, desc="Training PPO")
    total_steps = 0

    while total_steps < config.total_timesteps:
        state = env.reset()
        episode_reward = 0
        episode_length = 0

        while True:
            # Select action
            action, log_prob, value = agent.select_action(state)

            # Step environment
            next_state, reward, done, info = env.step(action)

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
                portfolio_value = info['portfolio_value']
                episode_return = (portfolio_value - config.initial_balance) / config.initial_balance
                episode_returns.append(episode_return)
                episode_trades.append(info['total_trades'])
                break

            state = next_state

        episode_rewards.append(episode_reward)

        # Update progress
        if len(episode_returns) > 0 and len(episode_returns) % 10 == 0:
            avg_return = np.mean(episode_returns[-10:])
            avg_trades = np.mean(episode_trades[-10:])
            pbar.set_description(f"Avg Return: {avg_return:.2%} | Avg Trades: {avg_trades:.0f}")

    pbar.close()

    # Test the model
    print("\n" + "-"*50)
    print("📊 TESTING ON HOLDOUT DATA")
    print("-"*50)

    test_env = TradingEnv5Min(test_data, config)
    state = test_env.reset()
    test_reward = 0
    action_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}

    for _ in range(len(test_data) - test_env.lookback - 51):
        action, _, _ = agent.select_action(state, deterministic=True)
        state, reward, done, info = test_env.step(action)
        test_reward += reward
        action_counts[action] += 1

        if done:
            break

    test_return = (info['portfolio_value'] - config.initial_balance) / config.initial_balance
    test_trades = info['total_trades']

    # Save model
    torch.save(agent.network.state_dict(), MODELS_DIR / 'ppo_improved.pth')

    # Calculate metrics
    action_map = {0: 'Hold', 1: 'Buy', 2: 'Sell', 3: 'Short', 4: 'Cover'}
    total_actions = sum(action_counts.values())

    print(f"\n📊 Test Results:")
    print(f"  Portfolio Return: {test_return:.2%}")
    print(f"  Total Trades: {test_trades}")
    print(f"  Final Balance: ${info['portfolio_value']:.2f}")

    print(f"\n📈 Action Distribution:")
    for action_id, count in action_counts.items():
        pct = count / total_actions * 100 if total_actions > 0 else 0
        print(f"  {action_map[action_id]:6s}: {pct:.1f}%")

    # Save metrics
    metrics = {
        'model_type': 'PPO_Improved_5min',
        'data_config': {
            'timeframe': '5min',
            'train_bars': len(train_data),
            'test_bars': len(test_data)
        },
        'architecture': {
            'state_dim': config.state_dim,
            'action_dim': config.action_dim,
            'hidden_dim': config.hidden_dim
        },
        'training': {
            'total_timesteps': total_steps,
            'episodes': len(episode_returns),
            'avg_train_return': float(np.mean(episode_returns)),
            'best_train_return': float(np.max(episode_returns)) if episode_returns else 0
        },
        'test_performance': {
            'return': float(test_return),
            'trades': test_trades,
            'final_balance': float(info['portfolio_value']),
            'action_distribution': {action_map[k]: v/total_actions for k, v in action_counts.items()}
        },
        'capabilities': {
            'short_selling': True,
            'transaction_costs': config.transaction_cost
        },
        'trained_at': datetime.utcnow().isoformat()
    }

    with open(MODELS_DIR / 'ppo_improved_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "="*70)
    print("✅ PPO TRAINING COMPLETE!")
    print("="*70)
    print(f"  Test Return: {test_return:.2%}")
    print(f"  Model saved to: ppo_improved.pth")

    return agent, metrics

if __name__ == "__main__":
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    agent, metrics = train_ppo_improved()

    if agent and metrics:
        if metrics['test_performance']['return'] > 0:
            print(f"\n🎉 SUCCESS! PPO achieved {metrics['test_performance']['return']:.2%} return!")
        else:
            print(f"\n⚠️ PPO needs more tuning - achieved {metrics['test_performance']['return']:.2%}")