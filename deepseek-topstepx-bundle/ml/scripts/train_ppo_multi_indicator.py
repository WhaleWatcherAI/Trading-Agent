#!/usr/bin/env python3
"""
PPO with multiple technical indicators for better trading decisions.
Uses a comprehensive set of indicators to provide rich context.
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

class MultiIndicatorConfig:
    """Config with more features for better learning."""
    # Increased dimensions for all indicators
    state_dim = 50  # Much more features
    action_dim = 5  # Buy Strong, Buy, Hold, Sell, Sell Strong
    hidden_dim = 256  # Bigger network for complexity

    # Environment
    initial_balance = 10000
    transaction_cost = 0.0

    # PPO hyperparameters
    learning_rate = 5e-5
    gamma = 0.99
    lambda_gae = 0.95
    clip_epsilon = 0.2
    value_loss_coef = 0.5
    entropy_coef = 0.02
    max_grad_norm = 0.5

    # Training
    batch_size = 512
    n_epochs = 10
    n_steps = 1024
    total_timesteps = 100000  # More training

class IndicatorNetwork(nn.Module):
    """Network to process multiple indicators."""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()

        # Deeper network for complex patterns
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )

        self.actor = nn.Linear(hidden_dim // 2, action_dim)
        self.critic = nn.Linear(hidden_dim // 2, 1)

    def forward(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state)
        if state.dim() == 1:
            state = state.unsqueeze(0)

        features = self.feature_extractor(state)
        action_logits = self.actor(features)
        value = self.critic(features)

        return action_logits.squeeze(0), value.squeeze()

class MultiIndicatorEnv:
    """Environment with comprehensive technical indicators."""

    def __init__(self, data, config):
        self.config = config
        self.original_data = data.copy()

        # Calculate ALL indicators upfront
        print("📊 Calculating technical indicators...")
        self.data = self.calculate_all_indicators(data)
        print(f"   Added {len(self.data.columns) - len(data.columns)} indicators")

        self.reset()

    def calculate_all_indicators(self, df):
        """Calculate comprehensive set of indicators manually."""
        df = df.copy()

        # Price and Volume basics
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()

        # Moving Averages
        for period in [5, 9, 20, 50]:
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
            df[f'dist_sma_{period}'] = (df['close'] - df[f'sma_{period}']) / df['close']
            df[f'dist_ema_{period}'] = (df['close'] - df[f'ema_{period}']) / df['close']

        # MACD
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_diff'] = df['macd'] - df['macd_signal']

        # RSI
        for period in [7, 14, 21]:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))

        # Stochastic
        low_14 = df['low'].rolling(window=14).min()
        high_14 = df['high'].rolling(window=14).max()
        df['stoch_k'] = 100 * ((df['close'] - low_14) / (high_14 - low_14))
        df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()

        # Bollinger Bands
        for period in [10, 20]:
            df[f'bb_mid_{period}'] = df['close'].rolling(window=period).mean()
            bb_std = df['close'].rolling(window=period).std()
            df[f'bb_upper_{period}'] = df[f'bb_mid_{period}'] + (bb_std * 2)
            df[f'bb_lower_{period}'] = df[f'bb_mid_{period}'] - (bb_std * 2)
            df[f'bb_width_{period}'] = (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']) / df[f'bb_mid_{period}']
            df[f'bb_position_{period}'] = (df['close'] - df[f'bb_lower_{period}']) / (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}'])

        # ATR (Volatility)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)

        for period in [7, 14]:
            df[f'atr_{period}'] = true_range.rolling(window=period).mean()
            df[f'atr_ratio_{period}'] = df[f'atr_{period}'] / df['close']

        # ADX (Trend Strength) - simplified
        plus_dm = df['high'].diff()
        minus_dm = -df['low'].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        atr_14 = true_range.rolling(window=14).mean()
        plus_di = 100 * (plus_dm.rolling(window=14).mean() / atr_14)
        minus_di = 100 * (minus_dm.rolling(window=14).mean() / atr_14)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        df['adx'] = dx.rolling(window=14).mean()
        df['adx_pos'] = plus_di
        df['adx_neg'] = minus_di

        # OBV (On-Balance Volume)
        obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        df['obv'] = obv
        df['obv_sma'] = df['obv'].rolling(20).mean()
        df['obv_trend'] = (df['obv'] - df['obv_sma']) / df['obv_sma'].abs()

        # Williams %R
        highest_high = df['high'].rolling(window=14).max()
        lowest_low = df['low'].rolling(window=14).min()
        df['willr'] = -100 * (highest_high - df['close']) / (highest_high - lowest_low)

        # CCI (Commodity Channel Index)
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        sma_tp = typical_price.rolling(window=20).mean()
        mean_deviation = np.abs(typical_price - sma_tp).rolling(window=20).mean()
        df['cci'] = (typical_price - sma_tp) / (0.015 * mean_deviation)

        # MFI (Money Flow Index) - simplified
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']
        positive_flow = money_flow.where(typical_price > typical_price.shift(), 0).rolling(14).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(), 0).rolling(14).sum()
        mfi_ratio = positive_flow / negative_flow
        df['mfi'] = 100 - (100 / (1 + mfi_ratio))

        # VWAP
        cumulative_volume = df['volume'].cumsum()
        cumulative_pv = (typical_price * df['volume']).cumsum()
        df['vwap'] = cumulative_pv / cumulative_volume
        df['price_vs_vwap'] = (df['close'] - df['vwap']) / df['close']

        # Volatility measures
        df['volatility_20'] = df['returns'].rolling(20).std()
        df['volatility_50'] = df['returns'].rolling(50).std()

        # Support/Resistance (simplified)
        df['resistance'] = df['high'].rolling(20).max()
        df['support'] = df['low'].rolling(20).min()
        df['price_vs_resistance'] = (df['close'] - df['resistance']) / df['close']
        df['price_vs_support'] = (df['close'] - df['support']) / df['close']

        # Momentum
        for period in [5, 10, 20]:
            df[f'momentum_{period}'] = df['close'].pct_change(periods=period)

        # Pattern Recognition (simplified)
        # Higher high / Lower low
        df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(float)
        df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(float)

        # Fill NaN with forward fill then 0
        df = df.fillna(method='ffill').fillna(0)

        return df

    def reset(self):
        """Reset environment."""
        self.current_idx = 100  # Need history for indicators
        self.balance = self.config.initial_balance
        self.position = 0  # -2 to 2 scale
        self.entry_price = 0
        self.trades = []
        self.returns = []

        return self._get_state()

    def _get_state(self):
        """Get state with all indicators."""
        idx = min(self.current_idx, len(self.data) - 1)
        row = self.data.iloc[idx]

        # Select most important features
        features = [
            # Price action
            row.get('returns', 0),
            row.get('momentum_5', 0),
            row.get('momentum_10', 0),
            row.get('momentum_20', 0),

            # Moving averages
            row.get('dist_sma_5', 0),
            row.get('dist_sma_9', 0),
            row.get('dist_sma_20', 0),
            row.get('dist_sma_50', 0),
            row.get('dist_ema_9', 0),
            row.get('dist_ema_20', 0),

            # Oscillators
            row.get('rsi_7', 50) / 100,
            row.get('rsi_14', 50) / 100,
            row.get('rsi_21', 50) / 100,
            row.get('stoch_k', 50) / 100,
            row.get('stoch_d', 50) / 100,
            row.get('willr', 0) / 100,
            row.get('cci', 0) / 200,  # Normalize
            row.get('mfi', 50) / 100,

            # MACD
            row.get('macd', 0) / row['close'] if row['close'] > 0 else 0,
            row.get('macd_signal', 0) / row['close'] if row['close'] > 0 else 0,
            row.get('macd_diff', 0) / row['close'] if row['close'] > 0 else 0,

            # Bollinger Bands
            row.get('bb_position_10', 0.5),
            row.get('bb_position_20', 0.5),
            row.get('bb_width_10', 0),
            row.get('bb_width_20', 0),

            # Trend
            row.get('adx', 0) / 100,
            row.get('adx_pos', 0) / 100,
            row.get('adx_neg', 0) / 100,

            # Volume
            row.get('volume_ratio', 1),
            row.get('obv_trend', 0),

            # Volatility
            row.get('atr_ratio_7', 0),
            row.get('atr_ratio_14', 0),
            row.get('volatility_20', 0),
            row.get('volatility_50', 0),

            # Support/Resistance
            row.get('price_vs_resistance', 0),
            row.get('price_vs_support', 0),

            # VWAP
            row.get('price_vs_vwap', 0),

            # Pattern
            row.get('higher_high', 0),
            row.get('lower_low', 0),

            # Position info
            self.position / 2,  # Normalize
            (row['close'] - self.entry_price) / self.entry_price if self.position != 0 and self.entry_price > 0 else 0,

            # Performance
            len(self.trades) / 100,  # Trade count normalized
            sum(1 for r in self.returns if r > 0) / max(1, len(self.returns)),  # Win rate

            # Market regime
            1.0 if row.get('sma_20', 0) > row.get('sma_50', 0) else -1.0,  # Trend

            # Time
            self.current_idx / len(self.data),
        ]

        # Ensure correct size
        features = features[:self.config.state_dim]
        while len(features) < self.config.state_dim:
            features.append(0)

        # Clean NaN and inf
        features = np.array(features, dtype=np.float32)
        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)

        return features

    def step(self, action):
        """
        Execute action.
        0: Buy Strong (position = 2)
        1: Buy (position = 1)
        2: Hold (position = 0)
        3: Sell (position = -1)
        4: Sell Strong (position = -2)
        """
        idx = min(self.current_idx, len(self.data) - 1)
        price = self.data.iloc[idx]['close']

        old_position = self.position
        target_position = action - 2  # Maps 0-4 to -2 to 2

        reward = 0

        # Change position
        if target_position != old_position:
            # Close old position
            if old_position != 0:
                if old_position > 0:  # Was long
                    returns = (price - self.entry_price) / self.entry_price * abs(old_position)
                else:  # Was short
                    returns = (self.entry_price - price) / self.entry_price * abs(old_position)

                self.returns.append(returns)
                self.balance *= (1 + returns)
                reward = returns * 100  # Scale reward

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

        # Continuous position reward
        if self.position != 0:
            if self.position > 0:
                unrealized = (price - self.entry_price) / self.entry_price * abs(self.position)
            else:
                unrealized = (self.entry_price - price) / self.entry_price * abs(self.position)
            reward += unrealized * 20  # Encourage good positions

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

            # Final reward
            total_return = (self.balance - self.config.initial_balance) / self.config.initial_balance
            reward += total_return * 500

            # Bonus for good Sharpe ratio
            if len(self.returns) > 0:
                sharpe = np.mean(self.returns) / (np.std(self.returns) + 1e-6)
                reward += sharpe * 50

        # Get new state
        new_state = self._get_state()

        info = {
            'balance': self.balance,
            'total_return': (self.balance - self.config.initial_balance) / self.config.initial_balance,
            'n_trades': len(self.trades),
            'win_rate': sum(1 for r in self.returns if r > 0) / max(1, len(self.returns)),
            'position': self.position
        }

        return new_state, reward, done, info

class MultiIndicatorAgent:
    """PPO agent for multi-indicator trading."""

    def __init__(self, config):
        self.config = config
        self.network = IndicatorNetwork(config.state_dim, config.action_dim, config.hidden_dim)
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

def train_multi_indicator():
    """Train PPO with multiple indicators."""

    print("\n" + "="*70)
    print("🚀 PPO TRAINING WITH MULTIPLE INDICATORS")
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
    config = MultiIndicatorConfig()
    env = MultiIndicatorEnv(train_data, config)
    agent = MultiIndicatorAgent(config)

    print(f"\n⚡ Configuration:")
    print(f"  State dimension: {config.state_dim} features")
    print(f"  Actions: Buy Strong, Buy, Hold, Sell, Sell Strong")
    print(f"  Training steps: {config.total_timesteps:,}")

    # Training
    episode_returns = []
    episode_trades = []
    episode_positions = []

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
                episode_positions.append(info['position'])

                if len(episode_returns) % 10 == 0:
                    avg_return = np.mean(episode_returns[-10:])
                    avg_trades = np.mean(episode_trades[-10:])
                    pbar.set_description(f"Return: {avg_return:.2%} | Trades: {avg_trades:.0f}")

                break

            state = next_state

    pbar.close()

    # Test
    print("\n📊 TESTING ON HOLDOUT DATA")

    test_env = MultiIndicatorEnv(test_data, config)
    state = test_env.reset()

    actions_taken = []

    for _ in range(len(test_data) - 101):
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
    print(f"  Trades: {info['n_trades']}")
    print(f"  Win Rate: {info['win_rate']:.1%}")

    print(f"\n📊 Action Distribution:")
    for i, name in enumerate(action_names):
        pct = action_counts[i] / len(actions_taken) * 100 if actions_taken else 0
        print(f"  {name:12s}: {pct:5.1f}% ({action_counts[i]:4d} times)")

    # Save model
    torch.save(agent.network.state_dict(), MODELS_DIR / 'ppo_multi_indicator.pth')

    # Save metrics
    metrics = {
        'model': 'PPO_Multi_Indicator',
        'indicators': 50,
        'test_return': float(test_return),
        'test_trades': info['n_trades'],
        'test_win_rate': float(info['win_rate']),
        'test_balance': float(info['balance']),
        'action_distribution': {name: action_counts[i]/len(actions_taken) for i, name in enumerate(action_names)} if actions_taken else {},
        'trained_at': datetime.utcnow().isoformat()
    }

    with open(MODELS_DIR / 'ppo_multi_indicator_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "="*70)
    if test_return > 0:
        print(f"✅ SUCCESS! Achieved {test_return:.2%} return with {len(action_counts)} indicators!")
    else:
        print(f"📊 Result: {test_return:.2%} return")
    print("="*70)

    return agent, metrics

if __name__ == "__main__":
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    agent, metrics = train_multi_indicator()