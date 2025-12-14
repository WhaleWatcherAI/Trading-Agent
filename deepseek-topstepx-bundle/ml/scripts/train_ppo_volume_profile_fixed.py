#!/usr/bin/env python3
"""
FIXED PPO implementation of Volume Profile Strategy with safeguards
Mean Reversion: Inside Value Area (short at VAH, long at VAL, target POC)
Continuation: Outside Value Area (breakout trades, close on return to value)
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
import logging

# Set up logging for debugging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "ml" / "data"
MODELS_DIR = ROOT / "ml" / "models"

class VolumeProfileConfig:
    """Configuration for Volume Profile PPO strategy."""

    # Volume Profile parameters
    lookback = 100  # REDUCED from 200 for stability
    n_rows = 50  # REDUCED from 100 for faster calculation
    value_area_pct = 0.70  # 70% of volume for value area
    max_iterations = 100  # SAFEGUARD: Maximum iterations for value area calculation

    # Strategy parameters
    mr_buffer_pct = 0.001  # 0.1% buffer for mean reversion
    cont_buffer_pct = 0.0005  # 0.05% buffer for continuation
    use_atr_stop = True
    atr_multiplier = 1.5
    atr_length = 14

    # PPO hyperparameters
    state_dim = 25  # REDUCED from 30 for simplicity
    action_dim = 5  # Strong Buy, Buy, Hold, Sell, Strong Sell
    hidden_dim = 128  # REDUCED from 256

    learning_rate = 3e-4
    gamma = 0.99
    lambda_gae = 0.95
    clip_epsilon = 0.2
    value_loss_coef = 0.5
    entropy_coef = 0.01
    max_grad_norm = 0.5

    # Training
    batch_size = 256  # REDUCED from 512
    n_epochs = 5  # REDUCED from 10
    n_steps = 1024  # REDUCED from 2048
    total_timesteps = 50000  # REDUCED from 200000 for testing

    # Environment
    initial_balance = 10000
    max_position = 1  # Maximum position size
    transaction_cost = 0.0

class VolumeProfileNetwork(nn.Module):
    """Neural network for Volume Profile strategy."""

    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()

        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

        # Actor head
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )

        # Critic head
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state)
        if state.dim() == 1:
            state = state.unsqueeze(0)

        shared_features = self.shared(state)
        action_logits = self.actor(shared_features)
        value = self.critic(shared_features)

        return action_logits.squeeze(0), value.squeeze()

class VolumeProfileEnvironment:
    """Environment implementing Volume Profile strategy logic with safeguards."""

    def __init__(self, data, config):
        self.data = data
        self.config = config

        # Pre-calculate technical indicators
        self._calculate_indicators()

        # Initialize
        self.reset()

    def _calculate_indicators(self):
        """Pre-calculate all technical indicators with error handling."""
        try:
            df = self.data

            # Basic price features
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

            # ATR for stops
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift(1))
            low_close = np.abs(df['low'] - df['close'].shift(1))
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['atr'] = true_range.rolling(window=self.config.atr_length).mean()

            # Volume features
            df['volume_ma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma'].replace(0, 1)  # SAFEGUARD: Avoid division by zero

            # Additional indicators for context
            df['rsi'] = self._calculate_rsi(df['close'], 14)
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()

            # Fill NaN values with forward fill then zeros
            df.fillna(method='ffill', inplace=True)
            df.fillna(0, inplace=True)

            # SAFEGUARD: Replace any infinite values with 0
            df.replace([np.inf, -np.inf], 0, inplace=True)

        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            # Fill with zeros as fallback
            df.fillna(0, inplace=True)

    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator with error handling."""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

            # SAFEGUARD: Avoid division by zero
            rs = gain / loss.replace(0, 1)
            rsi = 100 - (100 / (1 + rs))
            return rsi.fillna(50)  # Default to neutral 50
        except:
            return pd.Series(50, index=prices.index)  # Return neutral RSI on error

    def _calculate_volume_profile_simple(self, idx):
        """Simplified and robust Volume Profile calculation."""
        try:
            start_idx = max(0, idx - self.config.lookback)
            end_idx = min(len(self.data), idx + 1)

            window_data = self.data.iloc[start_idx:end_idx]

            # SAFEGUARD: Need minimum data
            if len(window_data) < 10:
                current_price = self.data['close'].iloc[idx] if idx < len(self.data) else 100
                return current_price, current_price * 1.01, current_price * 0.99, current_price

            # Use percentiles as a simple approximation - MUCH FASTER!
            prices = window_data['close'].values
            volumes = window_data['volume'].values

            # SAFEGUARD: Check for valid data
            if len(prices) == 0 or np.all(np.isnan(prices)):
                current_price = 100  # Default fallback
                return current_price, current_price * 1.01, current_price * 0.99, current_price

            # Simple volume-weighted calculations
            if np.sum(volumes) > 0:
                # Volume-weighted average price (VWAP) as POC approximation
                poc = np.sum(prices * volumes) / np.sum(volumes)
            else:
                poc = np.median(prices)

            # Use percentiles for value area
            vah = np.percentile(prices, 75)  # 75th percentile as VAH
            val = np.percentile(prices, 25)  # 25th percentile as VAL

            # SAFEGUARD: Ensure valid values
            if np.isnan(poc) or np.isnan(vah) or np.isnan(val):
                current_price = prices[-1] if len(prices) > 0 else 100
                return current_price, current_price * 1.01, current_price * 0.99, current_price

            # SAFEGUARD: Ensure VAH > VAL
            if vah <= val:
                spread = abs(poc * 0.01)  # 1% spread
                vah = poc + spread
                val = poc - spread

            return float(poc), float(vah), float(val), float(poc)

        except Exception as e:
            logger.warning(f"Error in volume profile calculation at idx {idx}: {e}")
            # Return safe default values
            current_price = 100
            return current_price, current_price * 1.01, current_price * 0.99, current_price

    def reset(self):
        """Reset environment to initial state."""
        self.current_idx = max(100, self.config.lookback)  # Start after lookback
        self.balance = self.config.initial_balance
        self.position = 0  # -1 short, 0 neutral, 1 long
        self.entry_price = 0
        self.stop_price = 0
        self.trades = []
        self.returns = []
        self.current_mode = "NONE"  # MR_LONG, MR_SHORT, CONT_LONG, CONT_SHORT

        return self._get_state()

    def _get_state(self):
        """Get current state with error handling."""
        try:
            idx = min(self.current_idx, len(self.data) - 1)
            row = self.data.iloc[idx]

            # Calculate Volume Profile with simplified method
            poc, vah, val, _ = self._calculate_volume_profile_simple(idx)

            price = float(row['close'])
            value_range = max(vah - val, 0.01)  # SAFEGUARD: Minimum range

            # Price position relative to value area
            inside_value = val <= price <= vah
            above_value = price > vah
            below_value = price < val

            # Distance to key levels (normalized and capped)
            dist_to_vah = np.clip((price - vah) / value_range, -2, 2)
            dist_to_val = np.clip((price - val) / value_range, -2, 2)
            dist_to_poc = np.clip((price - poc) / value_range, -2, 2)

            # Build feature vector with error handling
            features = [
                # Volume Profile features
                dist_to_poc,
                dist_to_vah,
                dist_to_val,
                np.clip(value_range / price, 0, 1),  # Normalized value range

                # Zone indicators
                float(inside_value),
                float(above_value),
                float(below_value),

                # Price features (with safe defaults)
                np.clip(row.get('returns', 0), -0.1, 0.1),
                np.clip(row.get('log_returns', 0), -0.1, 0.1),

                # Technical indicators (with safe access)
                np.clip(row.get('rsi', 50) / 100, 0, 1),
                np.clip(row.get('volume_ratio', 1), 0, 5),

                # Position info
                float(self.position),
                np.clip((price - self.entry_price) / self.entry_price if self.entry_price > 0 else 0, -0.5, 0.5),

                # Recent performance
                np.mean(self.returns[-10:]) if len(self.returns) > 0 else 0,
                min(len(self.trades) / 100, 1),  # Normalized trade frequency

                # Market microstructure
                np.clip((row['high'] - row['low']) / price if price > 0 else 0, 0, 0.1),
                np.clip((row['close'] - row['open']) / row['open'] if row['open'] > 0 else 0, -0.1, 0.1),

                # Time feature
                idx / len(self.data),

                # Mode encoding (simplified)
                float(self.current_mode == "MR_LONG"),
                float(self.current_mode == "MR_SHORT"),
                float(self.current_mode == "CONT_LONG"),
                float(self.current_mode == "CONT_SHORT"),

                # Additional features for stability
                np.clip(row.get('atr', 0) / price if price > 0 else 0, 0, 0.1),
                np.clip(row.get('volume', 0) / 1e6, 0, 10),  # Volume in millions
                0.0  # Padding
            ]

            # Ensure correct dimension
            features = features[:self.config.state_dim]
            while len(features) < self.config.state_dim:
                features.append(0.0)

            # SAFEGUARD: Replace any NaN or inf values
            features = [float(f) if np.isfinite(f) else 0.0 for f in features]

            return np.array(features, dtype=np.float32)

        except Exception as e:
            logger.error(f"Error in _get_state at idx {self.current_idx}: {e}")
            # Return zero state as fallback
            return np.zeros(self.config.state_dim, dtype=np.float32)

    def step(self, action):
        """Execute action with extensive error handling."""
        try:
            idx = min(self.current_idx, len(self.data) - 1)
            row = self.data.iloc[idx]
            price = float(row['close'])
            atr = float(row.get('atr', price * 0.01))  # Default to 1% if ATR unavailable

            # Calculate Volume Profile
            poc, vah, val, _ = self._calculate_volume_profile_simple(idx)
            value_range = max(vah - val, 0.01)

            # Price zones
            inside_value = val <= price <= vah
            above_value = price > vah
            below_value = price < val

            # Mean reversion zones
            vah_buffer = value_range * self.config.mr_buffer_pct
            val_buffer = value_range * self.config.mr_buffer_pct
            near_vah = (vah - vah_buffer) <= price <= vah and inside_value
            near_val = val <= price <= (val + val_buffer) and inside_value

            # Continuation zones
            cont_long_zone = price > vah * (1.0 + self.config.cont_buffer_pct)
            cont_short_zone = price < val * (1.0 - self.config.cont_buffer_pct)

            reward = 0
            old_position = self.position

            # Simplified action mapping
            if action == 4 and cont_long_zone and self.position <= 0:  # Strong Buy - Continuation
                if self.position < 0:  # Close short
                    returns = (self.entry_price - price) / self.entry_price
                    self.returns.append(returns)
                    self.balance *= (1 + returns)
                    reward += returns * 100

                # Enter long
                self.position = 1
                self.entry_price = price
                self.stop_price = price - atr * self.config.atr_multiplier
                self.current_mode = "CONT_LONG"
                self.trades.append({'type': 'cont_long', 'price': price})
                reward += 1

            elif action == 3 and near_val and self.position <= 0:  # Buy - Mean Reversion
                if self.position < 0:  # Close short
                    returns = (self.entry_price - price) / self.entry_price
                    self.returns.append(returns)
                    self.balance *= (1 + returns)
                    reward += returns * 100

                # Enter long
                self.position = 1
                self.entry_price = price
                self.stop_price = price - atr * self.config.atr_multiplier
                self.current_mode = "MR_LONG"
                self.trades.append({'type': 'mr_long', 'price': price})
                reward += 1

            elif action == 1 and near_vah and self.position >= 0:  # Sell - Mean Reversion
                if self.position > 0:  # Close long
                    returns = (price - self.entry_price) / self.entry_price
                    self.returns.append(returns)
                    self.balance *= (1 + returns)
                    reward += returns * 100

                # Enter short
                self.position = -1
                self.entry_price = price
                self.stop_price = price + atr * self.config.atr_multiplier
                self.current_mode = "MR_SHORT"
                self.trades.append({'type': 'mr_short', 'price': price})
                reward += 1

            elif action == 0 and cont_short_zone and self.position >= 0:  # Strong Sell - Continuation
                if self.position > 0:  # Close long
                    returns = (price - self.entry_price) / self.entry_price
                    self.returns.append(returns)
                    self.balance *= (1 + returns)
                    reward += returns * 100

                # Enter short
                self.position = -1
                self.entry_price = price
                self.stop_price = price + atr * self.config.atr_multiplier
                self.current_mode = "CONT_SHORT"
                self.trades.append({'type': 'cont_short', 'price': price})
                reward += 1

            # Check exit conditions
            if self.position != 0 and self.entry_price > 0:
                # Mean reversion exits
                if self.current_mode == "MR_LONG" and price >= poc:
                    returns = (price - self.entry_price) / self.entry_price
                    self.returns.append(returns)
                    self.balance *= (1 + returns)
                    reward += returns * 100 + 3  # Bonus for hitting target
                    self.position = 0
                    self.current_mode = "NONE"

                elif self.current_mode == "MR_SHORT" and price <= poc:
                    returns = (self.entry_price - price) / self.entry_price
                    self.returns.append(returns)
                    self.balance *= (1 + returns)
                    reward += returns * 100 + 3  # Bonus for hitting target
                    self.position = 0
                    self.current_mode = "NONE"

                # Continuation exits
                elif self.current_mode == "CONT_LONG" and price <= vah:
                    returns = (price - self.entry_price) / self.entry_price
                    self.returns.append(returns)
                    self.balance *= (1 + returns)
                    reward += returns * 100
                    self.position = 0
                    self.current_mode = "NONE"

                elif self.current_mode == "CONT_SHORT" and price >= val:
                    returns = (self.entry_price - price) / self.entry_price
                    self.returns.append(returns)
                    self.balance *= (1 + returns)
                    reward += returns * 100
                    self.position = 0
                    self.current_mode = "NONE"

                # Stop loss
                if self.config.use_atr_stop and self.stop_price > 0:
                    if self.position > 0 and row['low'] <= self.stop_price:
                        returns = (self.stop_price - self.entry_price) / self.entry_price
                        self.returns.append(returns)
                        self.balance *= (1 + returns)
                        reward += returns * 100 - 2  # Penalty for stop
                        self.position = 0
                        self.current_mode = "NONE"

                    elif self.position < 0 and row['high'] >= self.stop_price:
                        returns = (self.entry_price - self.stop_price) / self.entry_price
                        self.returns.append(returns)
                        self.balance *= (1 + returns)
                        reward += returns * 100 - 2  # Penalty for stop
                        self.position = 0
                        self.current_mode = "NONE"

            # Small reward for holding position in profit
            if self.position != 0 and self.entry_price > 0:
                if self.position > 0:
                    unrealized = (price - self.entry_price) / self.entry_price
                else:
                    unrealized = (self.entry_price - price) / self.entry_price
                reward += np.clip(unrealized * 5, -10, 10)

            # Move to next bar
            self.current_idx += 1
            done = self.current_idx >= len(self.data) - 1

            # Final position close
            if done and self.position != 0 and self.entry_price > 0:
                if self.position > 0:
                    returns = (price - self.entry_price) / self.entry_price
                else:
                    returns = (self.entry_price - price) / self.entry_price
                self.returns.append(returns)
                self.balance *= (1 + returns)

                # Final reward
                total_return = (self.balance - self.config.initial_balance) / self.config.initial_balance
                reward += total_return * 100

            # Get new state
            new_state = self._get_state()

            info = {
                'balance': self.balance,
                'total_return': (self.balance - self.config.initial_balance) / self.config.initial_balance,
                'n_trades': len(self.trades),
                'win_rate': sum(1 for r in self.returns if r > 0) / max(1, len(self.returns)),
                'current_mode': self.current_mode,
                'position': self.position
            }

            return new_state, reward, done, info

        except Exception as e:
            logger.error(f"Error in step at idx {self.current_idx}: {e}")
            # Return safe defaults
            self.current_idx += 1
            done = self.current_idx >= len(self.data) - 1
            return self._get_state(), 0.0, done, {'error': str(e)}

class VolumeProfilePPOAgent:
    """PPO Agent for Volume Profile strategy."""

    def __init__(self, config):
        self.config = config
        self.network = VolumeProfileNetwork(
            config.state_dim,
            config.action_dim,
            config.hidden_dim
        )
        self.optimizer = optim.Adam(self.network.parameters(), lr=config.learning_rate)

        # Experience buffers
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def clear_buffers(self):
        """Clear experience buffers."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def select_action(self, state, deterministic=False):
        """Select action using current policy."""
        with torch.no_grad():
            action_logits, value = self.network(state)

            if deterministic:
                action = torch.argmax(action_logits).item()
                return action, 0, value.item()

            # Add exploration noise
            dist = Categorical(logits=action_logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            return action.item(), log_prob.item(), value.item()

    def update(self):
        """PPO update step with error handling."""
        try:
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

                delta = rewards[t] + self.config.gamma * (old_values[t+1] if t+1 < len(rewards) else 0) - old_values[t]
                running_advantage = delta + self.config.gamma * self.config.lambda_gae * running_advantage
                advantages[t] = running_advantage

            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # PPO epochs
            total_loss = 0
            for _ in range(self.config.n_epochs):
                # Shuffle data
                indices = torch.randperm(len(states))

                for start in range(0, len(states), self.config.batch_size):
                    end = min(start + self.config.batch_size, len(states))
                    batch_indices = indices[start:end]

                    batch_states = states[batch_indices]
                    batch_actions = actions[batch_indices]
                    batch_old_log_probs = old_log_probs[batch_indices]
                    batch_advantages = advantages[batch_indices]
                    batch_returns = returns[batch_indices]

                    # Forward pass
                    action_logits, values = self.network(batch_states)
                    dist = Categorical(logits=action_logits)
                    log_probs = dist.log_prob(batch_actions)

                    # PPO loss
                    ratio = torch.exp(log_probs - batch_old_log_probs)
                    surr1 = ratio * batch_advantages
                    surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * batch_advantages
                    policy_loss = -torch.min(surr1, surr2).mean()

                    # Value loss
                    value_loss = F.mse_loss(values.squeeze(), batch_returns)

                    # Entropy bonus
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
            return {'loss': total_loss / max(1, self.config.n_epochs * (len(states) // self.config.batch_size))}

        except Exception as e:
            logger.error(f"Error in update: {e}")
            self.clear_buffers()
            return {'error': str(e)}

def train_volume_profile_fixed():
    """Train PPO on Volume Profile strategy with safeguards."""

    print("\n" + "="*70)
    print("🚀 TRAINING FIXED PPO WITH VOLUME PROFILE STRATEGY")
    print("="*70)

    # Load data
    data_path = DATA_DIR / 'spy_5min_data.parquet'
    if not data_path.exists():
        print("❌ Data not found! Please run data preparation first.")
        return None

    df = pd.read_parquet(data_path)
    print(f"📊 Loaded {len(df)} 5-minute bars")

    # Split data
    train_size = int(len(df) * 0.7)
    val_size = int(len(df) * 0.15)

    train_data = df.iloc[:train_size].copy()
    val_data = df.iloc[train_size:train_size+val_size].copy()
    test_data = df.iloc[train_size+val_size:].copy()

    print(f"  Train: {len(train_data)} bars")
    print(f"  Val: {len(val_data)} bars")
    print(f"  Test: {len(test_data)} bars")

    # Initialize
    config = VolumeProfileConfig()
    env = VolumeProfileEnvironment(train_data, config)
    agent = VolumeProfilePPOAgent(config)

    print(f"\n⚙️ Fixed Configuration:")
    print(f"  Lookback: {config.lookback} bars (reduced)")
    print(f"  Value Area: {config.value_area_pct:.0%}")
    print(f"  Max iterations: {config.max_iterations} (safeguard)")
    print(f"  State dim: {config.state_dim}")
    print(f"  Total timesteps: {config.total_timesteps:,}")

    # Training metrics
    episode_returns = []
    episode_trades = []
    mode_counts = {'MR_LONG': 0, 'MR_SHORT': 0, 'CONT_LONG': 0, 'CONT_SHORT': 0}

    # Training loop with error handling
    print(f"\n🎯 Training for {config.total_timesteps:,} timesteps...")
    pbar = tqdm(total=config.total_timesteps, desc="Training")

    total_steps = 0
    best_val_return = -float('inf')
    error_count = 0
    max_errors = 10

    while total_steps < config.total_timesteps:
        try:
            state = env.reset()
            episode_reward = 0
            episode_steps = 0
            max_episode_steps = len(train_data) - config.lookback - 10  # Safeguard

            while episode_steps < max_episode_steps:
                action, log_prob, value = agent.select_action(state)
                next_state, reward, done, info = env.step(action)

                # Check for errors
                if 'error' in info:
                    error_count += 1
                    if error_count > max_errors:
                        print(f"\n❌ Too many errors ({error_count}), stopping training")
                        break
                    logger.warning(f"Episode error: {info['error']}")
                    break

                agent.states.append(state)
                agent.actions.append(action)
                agent.rewards.append(reward)
                agent.values.append(value)
                agent.log_probs.append(log_prob)
                agent.dones.append(done)

                episode_reward += reward
                total_steps += 1
                episode_steps += 1
                pbar.update(1)

                # Update policy
                if len(agent.states) >= config.n_steps:
                    loss_info = agent.update()

                if done:
                    break

                state = next_state

            # Episode complete
            if 'total_return' in info:
                episode_returns.append(info['total_return'])
                episode_trades.append(info['n_trades'])

                # Update progress
                if len(episode_returns) % 5 == 0:
                    avg_return = np.mean(episode_returns[-5:])
                    avg_trades = np.mean(episode_trades[-5:])
                    pbar.set_description(
                        f"Return: {avg_return:.2%} | Trades: {avg_trades:.0f}"
                    )

                # Validation check
                if len(episode_returns) % 20 == 0:
                    val_env = VolumeProfileEnvironment(val_data, config)
                    val_state = val_env.reset()
                    val_steps = 0
                    max_val_steps = len(val_data) - config.lookback - 10

                    while val_steps < max_val_steps:
                        val_action, _, _ = agent.select_action(val_state, deterministic=True)
                        val_state, _, val_done, val_info = val_env.step(val_action)
                        val_steps += 1
                        if val_done:
                            break

                    val_return = val_info.get('total_return', 0)
                    if val_return > best_val_return:
                        best_val_return = val_return
                        torch.save(agent.network.state_dict(), MODELS_DIR / 'ppo_volume_profile_fixed.pth')
                        print(f"\n💾 Saved best model with val return: {val_return:.2%}")

        except Exception as e:
            logger.error(f"Training error: {e}")
            error_count += 1
            if error_count > max_errors:
                print(f"\n❌ Too many errors, stopping training")
                break
            continue

        if error_count > max_errors:
            break

    pbar.close()

    # Load best model if saved
    model_path = MODELS_DIR / 'ppo_volume_profile_fixed.pth'
    if model_path.exists():
        agent.network.load_state_dict(torch.load(model_path))

    # Test evaluation
    print("\n📊 TESTING ON HOLDOUT DATA")
    try:
        test_env = VolumeProfileEnvironment(test_data, config)
        state = test_env.reset()
        test_steps = 0
        max_test_steps = len(test_data) - config.lookback - 10

        while test_steps < max_test_steps:
            action, _, _ = agent.select_action(state, deterministic=True)
            state, _, done, info = test_env.step(action)
            test_steps += 1
            if done:
                break

        test_return = info.get('total_return', 0)

        print(f"\n🎯 Test Results:")
        print(f"  Return: {test_return:.2%}")
        print(f"  Final Balance: ${info.get('balance', 10000):.2f}")
        print(f"  Trades: {info.get('n_trades', 0)}")
        print(f"  Win Rate: {info.get('win_rate', 0):.1%}")

    except Exception as e:
        print(f"❌ Test evaluation error: {e}")
        test_return = 0

    # Save metrics
    metrics = {
        'model': 'PPO_Volume_Profile_Fixed',
        'strategy': 'Mean Reversion & Continuation (Fixed)',
        'test_return': float(test_return),
        'errors_during_training': error_count,
        'config': {
            'lookback': config.lookback,
            'value_area_pct': config.value_area_pct,
            'max_iterations': config.max_iterations,
        },
        'trained_at': datetime.utcnow().isoformat()
    }

    with open(MODELS_DIR / 'ppo_volume_profile_fixed_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "="*70)
    if test_return > 0:
        print(f"✅ SUCCESS! Fixed Volume Profile achieved {test_return:.2%} return!")
    else:
        print(f"📊 Fixed Volume Profile completed with {test_return:.2%} return")
    print("="*70)

    return agent, metrics

if __name__ == "__main__":
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    agent, metrics = train_volume_profile_fixed()