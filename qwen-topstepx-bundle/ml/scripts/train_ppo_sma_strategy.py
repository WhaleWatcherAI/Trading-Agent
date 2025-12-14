#!/usr/bin/env python3
"""
PPO trained on 9-SMA crossover strategy.
Learns WHEN the crossover works and maximizes profit.
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
    """Config for SMA strategy learning."""
    # State and action
    state_dim = 25  # More features for context
    action_dim = 4  # Ignore Signal, Follow Buy, Follow Sell, Scale Position
    hidden_dim = 256

    # Environment
    initial_balance = 10000
    max_position = 1.0  # Can use full capital
    transaction_cost = 0.0  # Start without costs

    # PPO hyperparameters - tuned for profit maximization
    learning_rate = 5e-5
    gamma = 0.995
    lambda_gae = 0.98
    clip_epsilon = 0.2
    value_loss_coef = 0.5
    entropy_coef = 0.03  # Some exploration
    max_grad_norm = 0.5

    # Training
    batch_size = 512
    n_epochs = 10
    n_steps = 2048
    total_timesteps = 500000  # More training for strategy learning

class SMAStrategyNetwork(nn.Module):
    """Network that learns when to follow SMA signals."""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()

        # Feature extraction layers
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Separate pathways for signal evaluation
        self.signal_evaluator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU()
        )

        # Actor and critic heads
        self.actor = nn.Linear(hidden_dim // 4, action_dim)
        self.critic = nn.Linear(hidden_dim // 4, 1)

        # Initialize actor to slightly favor following signals initially
        nn.init.xavier_uniform_(self.actor.weight, gain=0.01)
        self.actor.bias.data[0] = -0.5  # Slight bias against ignoring

    def forward(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state)
        if state.dim() == 1:
            state = state.unsqueeze(0)

        features = self.feature_net(state)
        signal_features = self.signal_evaluator(features)

        action_logits = self.actor(signal_features)
        value = self.critic(signal_features)

        return action_logits.squeeze(0), value.squeeze()

class SMAStrategyEnv:
    """Environment for learning SMA crossover strategy."""

    def __init__(self, data, config):
        self.data = data
        self.config = config
        self.reset()

    def reset(self):
        """Reset to initial state."""
        self.current_idx = 30  # Need history for SMA
        self.balance = self.config.initial_balance
        self.position = 0.0  # Fraction of capital in position
        self.entry_price = 0
        self.trades = []
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_profit = 0
        self.signals_followed = 0
        self.signals_ignored = 0
        self.false_signals = 0

        # Calculate SMA for entire dataset upfront
        self.data['sma_9'] = self.data['close'].rolling(window=9).mean()
        self.data['sma_20'] = self.data['close'].rolling(window=20).mean()
        self.data['sma_50'] = self.data['close'].rolling(window=50).mean()

        # Fill NaN values with price
        self.data['sma_9'] = self.data['sma_9'].fillna(self.data['close'])
        self.data['sma_20'] = self.data['sma_20'].fillna(self.data['close'])
        self.data['sma_50'] = self.data['sma_50'].fillna(self.data['close'])

        # Identify crossover points
        self.data['cross_above'] = (
            (self.data['close'] > self.data['sma_9']) &
            (self.data['close'].shift(1) <= self.data['sma_9'].shift(1))
        )
        self.data['cross_below'] = (
            (self.data['close'] < self.data['sma_9']) &
            (self.data['close'].shift(1) >= self.data['sma_9'].shift(1))
        )

        # Fill NaN in crossover signals
        self.data['cross_above'] = self.data['cross_above'].fillna(False)
        self.data['cross_below'] = self.data['cross_below'].fillna(False)

        return self._get_state()

    def _get_state(self):
        """Get current state with SMA context."""
        if self.current_idx >= len(self.data):
            self.current_idx = len(self.data) - 1

        current = self.data.iloc[self.current_idx]
        recent = self.data.iloc[max(0, self.current_idx-20):self.current_idx]

        # Current price and SMAs
        price = current['close']
        sma_9 = current.get('sma_9', price)
        sma_20 = current.get('sma_20', price)
        sma_50 = current.get('sma_50', price)

        # Distance from SMAs (key features)
        dist_sma9 = (price - sma_9) / price
        dist_sma20 = (price - sma_20) / price
        dist_sma50 = (price - sma_50) / price

        # SMA slopes (trend strength)
        if len(recent) > 0 and not recent['sma_9'].empty:
            sma9_base = recent['sma_9'].iloc[0]
            sma9_slope = (sma_9 - sma9_base) / sma9_base if sma9_base != 0 else 0
        else:
            sma9_slope = 0

        if len(recent) > 0 and not recent['sma_20'].empty:
            sma20_base = recent['sma_20'].iloc[0]
            sma20_slope = (sma_20 - sma20_base) / sma20_base if sma20_base != 0 else 0
        else:
            sma20_slope = 0

        # Crossover signals
        is_cross_above = 1.0 if current.get('cross_above', False) else 0.0
        is_cross_below = 1.0 if current.get('cross_below', False) else 0.0
        bars_since_cross = self._bars_since_last_cross()

        # Market context
        volatility = recent['close'].pct_change().std() if len(recent) > 1 else 0.01
        if pd.isna(volatility) or np.isinf(volatility):
            volatility = 0.01

        if len(recent) > 0 and 'volume' in recent.columns:
            vol_mean = recent['volume'].mean()
            volume_ratio = current.get('volume', 1) / vol_mean if vol_mean > 0 else 1.0
        else:
            volume_ratio = 1.0

        # Momentum indicators
        rsi = current.get('rsi', 50) / 100
        macd = current.get('macd', 0) / price if price > 0 else 0

        # Price action
        high_low_range = (current.get('high', price) - current.get('low', price)) / price
        close_range = (price - current.get('open', price)) / current.get('open', price)

        # Trend alignment
        trend_alignment = 1.0 if (dist_sma9 > 0 and dist_sma20 > 0 and dist_sma50 > 0) else (
            -1.0 if (dist_sma9 < 0 and dist_sma20 < 0 and dist_sma50 < 0) else 0.0
        )

        # Position info
        position_pnl = self._get_unrealized_pnl() if self.position != 0 else 0
        time_in_position = min((self.current_idx - self.entry_idx) / 100, 1.0) if hasattr(self, 'entry_idx') and self.position != 0 else 0

        # Strategy performance
        win_rate = self.winning_trades / max(1, self.winning_trades + self.losing_trades)
        follow_rate = self.signals_followed / max(1, self.signals_followed + self.signals_ignored)

        # Recent price changes
        returns_5 = (price - self.data.iloc[max(0, self.current_idx-5)]['close']) / self.data.iloc[max(0, self.current_idx-5)]['close']
        returns_10 = (price - self.data.iloc[max(0, self.current_idx-10)]['close']) / self.data.iloc[max(0, self.current_idx-10)]['close']
        returns_20 = (price - self.data.iloc[max(0, self.current_idx-20)]['close']) / self.data.iloc[max(0, self.current_idx-20)]['close']

        features = [
            # SMA distances (critical for strategy)
            dist_sma9 * 100,
            dist_sma20 * 100,
            dist_sma50 * 100,

            # SMA slopes
            sma9_slope * 100,
            sma20_slope * 100,

            # Crossover signals
            is_cross_above,
            is_cross_below,
            bars_since_cross / 100,

            # Market context
            volatility * 100,
            np.log(volume_ratio + 1),
            rsi - 0.5,
            macd * 100,

            # Price action
            high_low_range * 100,
            close_range * 100,

            # Trend
            trend_alignment,

            # Returns
            returns_5 * 100,
            returns_10 * 100,
            returns_20 * 100,

            # Position
            self.position,
            position_pnl * 100,
            time_in_position,

            # Performance
            win_rate,
            follow_rate,
            self.total_profit / self.config.initial_balance * 100,

            # Time
            self.current_idx / len(self.data)
        ]

        # Ensure correct dimension
        while len(features) < self.config.state_dim:
            features.append(0)
        features = features[:self.config.state_dim]

        # Clean any NaN or inf values
        features = np.array(features, dtype=np.float32)
        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)

        return features

    def _bars_since_last_cross(self):
        """Count bars since last crossover."""
        for i in range(min(100, self.current_idx)):
            idx = self.current_idx - i
            if self.data.iloc[idx].get('cross_above', False) or self.data.iloc[idx].get('cross_below', False):
                return i
        return 100

    def _get_unrealized_pnl(self):
        """Calculate unrealized P&L as percentage."""
        if self.position == 0:
            return 0

        current_price = self.data.iloc[self.current_idx]['close']

        if self.position > 0:  # Long
            return ((current_price - self.entry_price) / self.entry_price) * self.position
        else:  # Short
            return ((self.entry_price - current_price) / self.entry_price) * abs(self.position)

    def step(self, action):
        """
        Execute action based on SMA signal context.
        Actions:
        0: Ignore current signal
        1: Follow buy signal (if present) with standard position
        2: Follow sell signal (if present) with standard position
        3: Scale position based on confidence
        """
        current = self.data.iloc[self.current_idx]
        current_price = current['close']
        old_balance = self.balance

        # Check for crossover signals
        has_buy_signal = current.get('cross_above', False)
        has_sell_signal = current.get('cross_below', False)

        reward = 0
        action_taken = False

        # Process action based on signal context
        if action == 0:  # Ignore signal
            if has_buy_signal or has_sell_signal:
                self.signals_ignored += 1
                # Small reward for ignoring in high volatility
                recent_vol = self.data.iloc[max(0, self.current_idx-10):self.current_idx]['close'].pct_change().std()
                if recent_vol > 0.02:  # High volatility
                    reward += 0.01

        elif action == 1:  # Follow buy signal
            if has_buy_signal:
                self.signals_followed += 1

                # Close short if exists
                if self.position < 0:
                    pnl_pct = (self.entry_price - current_price) / self.entry_price * abs(self.position)
                    pnl = pnl_pct * self.balance
                    self.balance += pnl
                    self.total_profit += pnl
                    reward += pnl_pct * 100  # Scale reward

                    if pnl > 0:
                        self.winning_trades += 1
                    else:
                        self.losing_trades += 1

                # Enter long position
                self.position = 0.5  # 50% position for SMA strategy
                self.entry_price = current_price
                self.entry_idx = self.current_idx
                action_taken = True

                # Immediate reward based on SMA alignment
                if current['sma_9'] > current.get('sma_20', current_price):
                    reward += 0.05  # Trend aligned

        elif action == 2:  # Follow sell signal
            if has_sell_signal:
                self.signals_followed += 1

                # Close long if exists
                if self.position > 0:
                    pnl_pct = (current_price - self.entry_price) / self.entry_price * self.position
                    pnl = pnl_pct * self.balance
                    self.balance += pnl
                    self.total_profit += pnl
                    reward += pnl_pct * 100  # Scale reward

                    if pnl > 0:
                        self.winning_trades += 1
                    else:
                        self.losing_trades += 1

                # Enter short position
                self.position = -0.5  # 50% short position
                self.entry_price = current_price
                self.entry_idx = self.current_idx
                action_taken = True

                # Immediate reward based on SMA alignment
                if current['sma_9'] < current.get('sma_20', current_price):
                    reward += 0.05  # Trend aligned

        elif action == 3:  # Scale position based on confidence
            # Adjust position size based on signal strength
            if has_buy_signal:
                # Calculate confidence based on multiple SMAs alignment
                confidence = 0.3  # Base confidence
                if current_price > current.get('sma_20', current_price):
                    confidence += 0.3
                if current_price > current.get('sma_50', current_price):
                    confidence += 0.4

                self.signals_followed += 1

                # Close short if exists
                if self.position < 0:
                    pnl_pct = (self.entry_price - current_price) / self.entry_price * abs(self.position)
                    pnl = pnl_pct * self.balance
                    self.balance += pnl
                    self.total_profit += pnl
                    reward += pnl_pct * 100

                    if pnl > 0:
                        self.winning_trades += 1
                    else:
                        self.losing_trades += 1

                # Scale position based on confidence
                self.position = confidence  # Use confidence as position size
                self.entry_price = current_price
                self.entry_idx = self.current_idx
                action_taken = True
                reward += confidence * 0.1  # Reward scaling

            elif has_sell_signal:
                # Calculate confidence for short
                confidence = 0.3  # Base confidence
                if current_price < current.get('sma_20', current_price):
                    confidence += 0.3
                if current_price < current.get('sma_50', current_price):
                    confidence += 0.4

                self.signals_followed += 1

                # Close long if exists
                if self.position > 0:
                    pnl_pct = (current_price - self.entry_price) / self.entry_price * self.position
                    pnl = pnl_pct * self.balance
                    self.balance += pnl
                    self.total_profit += pnl
                    reward += pnl_pct * 100

                    if pnl > 0:
                        self.winning_trades += 1
                    else:
                        self.losing_trades += 1

                # Scale short position
                self.position = -confidence
                self.entry_price = current_price
                self.entry_idx = self.current_idx
                action_taken = True
                reward += confidence * 0.1

        # Record trade
        if action_taken:
            self.trades.append({
                'bar': self.current_idx,
                'action': action,
                'price': current_price,
                'position': self.position
            })

        # Move to next bar
        self.current_idx += 1

        # Calculate position P&L for continuous reward
        if self.position != 0:
            position_pnl = self._get_unrealized_pnl()
            reward += position_pnl * 5  # Continuous position reward

            # Check for stop loss
            if position_pnl < -0.02:  # 2% stop loss
                # Close position
                pnl = position_pnl * self.balance
                self.balance += pnl
                self.total_profit += pnl
                self.losing_trades += 1
                self.position = 0
                reward -= 1  # Penalty for stop loss
                self.false_signals += 1

        # Profit maximization reward
        profit_change = (self.balance - old_balance) / self.config.initial_balance
        reward += profit_change * 200  # Heavy weight on actual profit

        # Check if done
        done = (self.current_idx >= len(self.data) - 1) or (self.balance < 1000)

        # Final reward based on total profit
        if done:
            total_return = self.total_profit / self.config.initial_balance
            reward += total_return * 500  # Massive reward for profit

            # Bonus for good signal filtering
            if self.signals_ignored > 0 and self.signals_followed > 0:
                signal_quality = self.winning_trades / max(1, self.signals_followed)
                if signal_quality > 0.5:  # More than 50% win rate
                    reward += signal_quality * 100

        # Get new state
        new_state = self._get_state()

        info = {
            'balance': self.balance,
            'position': self.position,
            'total_profit': self.total_profit,
            'total_trades': len(self.trades),
            'win_rate': self.winning_trades / max(1, self.winning_trades + self.losing_trades),
            'signals_followed': self.signals_followed,
            'signals_ignored': self.signals_ignored,
            'signal_follow_rate': self.signals_followed / max(1, self.signals_followed + self.signals_ignored)
        }

        return new_state, reward, done, info

class SMAStrategyAgent:
    """PPO agent for SMA strategy learning."""

    def __init__(self, config):
        self.config = config
        self.network = SMAStrategyNetwork(
            config.state_dim,
            config.action_dim,
            config.hidden_dim
        )
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=config.learning_rate
        )

        self.clear_buffers()
        self.exploration_rate = 1.0
        self.exploration_decay = 0.999
        self.min_exploration = 0.05

    def clear_buffers(self):
        """Clear experience buffers."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def select_action(self, state, deterministic=False):
        """Select action with exploration."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state)
            action_logits, value = self.network(state_tensor)

            if deterministic:
                action = torch.argmax(action_logits).item()
                return action, 0, value.item()
            else:
                # Temperature for exploration
                temperature = max(self.min_exploration, self.exploration_rate)
                action_logits = action_logits / temperature

                dist = Categorical(logits=action_logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)

                return action.item(), log_prob.item(), value.item()

    def update(self):
        """PPO update."""
        if len(self.states) < self.config.batch_size:
            return {}

        self.exploration_rate *= self.exploration_decay

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
        total_loss = 0
        for _ in range(self.config.n_epochs):
            perm = torch.randperm(len(states))

            for start in range(0, len(states), self.config.batch_size):
                end = min(start + self.config.batch_size, len(states))
                batch_indices = perm[start:end]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]

                # Forward pass
                action_logits, values = self.network(batch_states)
                dist = Categorical(logits=action_logits)
                log_probs = dist.log_prob(batch_actions)

                # PPO objective
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(values.squeeze(), batch_returns)

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

        self.clear_buffers()

        return {'loss': total_loss / (self.config.n_epochs * max(1, len(states) // self.config.batch_size))}

def train_sma_strategy():
    """Train PPO to learn when SMA crossover strategy works."""

    print("\n" + "="*70)
    print("🚀 TRAINING PPO ON 9-SMA CROSSOVER STRATEGY")
    print("="*70)
    print("Goal: Learn WHEN crossover signals work and maximize profit")

    # Load data
    data_path = DATA_DIR / 'spy_5min_data.parquet'
    if not data_path.exists():
        print("❌ 5-minute data not found!")
        return None

    df = pd.read_parquet(data_path)
    print(f"📊 Loaded {len(df)} 5-minute bars")

    # Split data
    train_size = int(len(df) * 0.8)
    train_data = df.iloc[:train_size].copy()
    test_data = df.iloc[train_size:].copy()

    print(f"  Training: {len(train_data)} bars")
    print(f"  Testing: {len(test_data)} bars")

    # Initialize
    config = PPOConfig()
    env = SMAStrategyEnv(train_data, config)
    agent = SMAStrategyAgent(config)

    print(f"\n📈 Strategy Configuration:")
    print(f"  Base Signal: 9-SMA Crossover")
    print(f"  Actions: Ignore, Follow Buy, Follow Sell, Scale Position")
    print(f"  Goal: Maximize Total Profit")
    print(f"  Training Steps: {config.total_timesteps}")

    # Training metrics
    episode_profits = []
    episode_trades = []
    episode_win_rates = []
    signal_follow_rates = []

    # Training loop
    pbar = tqdm(total=config.total_timesteps, desc="Training SMA Strategy")
    total_steps = 0
    best_profit = -float('inf')

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

            # Update if buffer full
            if len(agent.states) >= config.n_steps:
                agent.update()

            if done:
                episode_profits.append(info['total_profit'])
                episode_trades.append(info['total_trades'])
                episode_win_rates.append(info['win_rate'])
                signal_follow_rates.append(info['signal_follow_rate'])

                # Save best model
                if info['total_profit'] > best_profit:
                    best_profit = info['total_profit']
                    torch.save(agent.network.state_dict(), MODELS_DIR / 'ppo_sma_best.pth')

                break

            state = next_state

        # Update progress
        if len(episode_profits) > 0 and len(episode_profits) % 10 == 0:
            avg_profit = np.mean(episode_profits[-10:])
            avg_trades = np.mean(episode_trades[-10:])
            avg_win_rate = np.mean(episode_win_rates[-10:])
            avg_follow_rate = np.mean(signal_follow_rates[-10:])

            pbar.set_description(
                f"Profit: ${avg_profit:.0f} | Trades: {avg_trades:.0f} | "
                f"WR: {avg_win_rate:.1%} | Follow: {avg_follow_rate:.1%}"
            )

    pbar.close()

    # Load best model for testing
    agent.network.load_state_dict(torch.load(MODELS_DIR / 'ppo_sma_best.pth'))

    # Test
    print("\n" + "-"*50)
    print("📊 TESTING ON HOLDOUT DATA")
    print("-"*50)

    test_env = SMAStrategyEnv(test_data, config)
    state = test_env.reset()
    action_counts = {i: 0 for i in range(4)}

    for _ in range(len(test_data) - 31):
        action, _, _ = agent.select_action(state, deterministic=True)
        state, reward, done, info = test_env.step(action)
        action_counts[action] += 1

        if done:
            break

    test_profit = info['total_profit']
    test_return = test_profit / config.initial_balance

    # Save final model
    torch.save(agent.network.state_dict(), MODELS_DIR / 'ppo_sma_strategy.pth')

    # Results
    action_map = {0: 'Ignore Signal', 1: 'Follow Buy', 2: 'Follow Sell', 3: 'Scale Position'}
    total_actions = sum(action_counts.values())

    print(f"\n📊 Test Results:")
    print(f"  Total Profit: ${test_profit:.2f}")
    print(f"  Return: {test_return:.2%}")
    print(f"  Total Trades: {info['total_trades']}")
    print(f"  Win Rate: {info['win_rate']:.1%}")
    print(f"  Signals Followed: {info['signals_followed']}")
    print(f"  Signals Ignored: {info['signals_ignored']}")
    print(f"  Signal Follow Rate: {info['signal_follow_rate']:.1%}")
    print(f"  Final Balance: ${test_env.balance:.2f}")

    print(f"\n📈 Action Distribution:")
    for action_id, count in action_counts.items():
        pct = count / total_actions * 100 if total_actions > 0 else 0
        print(f"  {action_map[action_id]:15s}: {pct:.1f}%")

    # Save metrics
    metrics = {
        'strategy': '9-SMA Crossover',
        'model_type': 'PPO_SMA_Strategy',
        'training': {
            'best_profit': float(best_profit),
            'avg_final_profit': float(np.mean(episode_profits[-100:])) if episode_profits else 0,
            'avg_final_trades': float(np.mean(episode_trades[-100:])) if episode_trades else 0,
            'avg_final_win_rate': float(np.mean(episode_win_rates[-100:])) if episode_win_rates else 0
        },
        'test': {
            'profit': float(test_profit),
            'return': float(test_return),
            'trades': info['total_trades'],
            'win_rate': float(info['win_rate']),
            'signals_followed': info['signals_followed'],
            'signals_ignored': info['signals_ignored'],
            'signal_follow_rate': float(info['signal_follow_rate']),
            'final_balance': float(test_env.balance),
            'action_distribution': {action_map[k]: v/total_actions for k, v in action_counts.items() if total_actions > 0}
        },
        'trained_at': datetime.utcnow().isoformat()
    }

    with open(MODELS_DIR / 'ppo_sma_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "="*70)
    print("✅ SMA STRATEGY TRAINING COMPLETE!")
    print("="*70)

    if test_return > 0:
        print(f"🎉 SUCCESS! Strategy achieved {test_return:.2%} return!")
        print(f"   Total Profit: ${test_profit:.2f}")
        print(f"   Learned to filter {info['signals_ignored']} bad signals")
    else:
        print(f"📊 Strategy needs tuning - achieved {test_return:.2%}")

    return agent, metrics

if __name__ == "__main__":
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    agent, metrics = train_sma_strategy()