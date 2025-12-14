#!/usr/bin/env python3
"""
ROBUST PPO implementation of Volume Profile Strategy with proper safeguards
Keeps full complexity but prevents infinite loops
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
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "ml" / "data"
MODELS_DIR = ROOT / "ml" / "models"

class VolumeProfileConfig:
    """Configuration for Volume Profile PPO strategy - FULL COMPLEXITY."""

    # Volume Profile parameters - KEEPING ORIGINAL VALUES
    lookback = 200  # Full lookback period
    n_rows = 100  # Full price level resolution
    value_area_pct = 0.70  # 70% of volume for value area

    # SAFEGUARD: Maximum iterations for value area calculation
    max_value_area_iterations = 200  # Prevent infinite loop but allow enough iterations

    # Strategy parameters
    mr_buffer_pct = 0.001  # 0.1% buffer for mean reversion
    cont_buffer_pct = 0.0005  # 0.05% buffer for continuation
    use_atr_stop = True
    atr_multiplier = 1.5
    atr_length = 14

    # PPO hyperparameters - KEEPING ORIGINAL DIMENSIONS
    state_dim = 30  # Full state dimension
    action_dim = 5  # Strong Buy, Buy, Hold, Sell, Strong Sell
    hidden_dim = 256  # Full network size

    learning_rate = 3e-4
    gamma = 0.99
    lambda_gae = 0.95
    clip_epsilon = 0.2
    value_loss_coef = 0.5
    entropy_coef = 0.01
    max_grad_norm = 0.5

    # Training
    batch_size = 512  # Full batch size
    n_epochs = 10  # Full epochs
    n_steps = 2048  # Full steps
    total_timesteps = 100000  # Reduced slightly for faster testing

    # Environment
    initial_balance = 10000
    max_position = 1
    transaction_cost = 0.0

class VolumeProfileNetwork(nn.Module):
    """Neural network for Volume Profile strategy - FULL ARCHITECTURE."""

    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()

        # Full architecture
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
    """Environment implementing FULL Volume Profile strategy logic with safeguards."""

    def __init__(self, data, config):
        self.data = data
        self.config = config

        # Add debug counters
        self.calculation_times = []
        self.iteration_counts = []

        # Pre-calculate technical indicators
        self._calculate_indicators()

        # Initialize
        self.reset()

    def _calculate_indicators(self):
        """Pre-calculate all technical indicators."""
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
        df['volume_ratio'] = df['volume'] / df['volume_ma'].replace(0, 1)

        # Additional indicators for context
        df['rsi'] = self._calculate_rsi(df['close'], 14)
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()

        # Fill NaN values
        df.fillna(method='ffill', inplace=True)
        df.fillna(0, inplace=True)

        # Replace infinite values
        df.replace([np.inf, -np.inf], 0, inplace=True)

    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        # Avoid division by zero
        rs = gain / loss.replace(0, 1)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)

    def _calculate_volume_profile(self, idx):
        """FULL Volume Profile calculation with safeguards."""
        try:
            start_idx = max(0, idx - self.config.lookback)
            end_idx = min(len(self.data), idx + 1)

            window_data = self.data.iloc[start_idx:end_idx]

            if len(window_data) < 50:  # Need minimum data
                # Return simple approximations
                mid = window_data['close'].mean()
                std = window_data['close'].std()
                return mid, mid + std, mid - std, mid

            # Price range
            highest_price = window_data['high'].max()
            lowest_price = window_data['low'].min()
            price_range = highest_price - lowest_price

            if price_range <= 0:
                current_price = window_data['close'].iloc[-1]
                return current_price, current_price, current_price, current_price

            row_height = price_range / self.config.n_rows

            # Build volume profile - FULL CALCULATION
            vol_at_level = np.zeros(self.config.n_rows)

            for i in range(self.config.n_rows):
                level_low = lowest_price + i * row_height
                level_high = lowest_price + (i + 1) * row_height

                # Find bars that touched this price level
                mask = (window_data['high'] > level_low) & (window_data['low'] < level_high)
                vol_at_level[i] = window_data.loc[mask, 'volume'].sum()

            # Find POC (Point of Control - max volume level)
            poc_idx = np.argmax(vol_at_level)
            poc = lowest_price + (poc_idx + 0.5) * row_height

            # Calculate Value Area (70% of volume)
            total_vol = vol_at_level.sum()
            if total_vol == 0:
                current_price = window_data['close'].iloc[-1]
                return current_price, current_price, current_price, current_price

            target_vol = total_vol * self.config.value_area_pct
            captured_vol = vol_at_level[poc_idx]

            upper_idx = poc_idx
            lower_idx = poc_idx

            # SAFEGUARD: Track iterations
            iterations = 0

            # Expand from POC to capture value area - WITH ITERATION LIMIT
            while captured_vol < target_vol and iterations < self.config.max_value_area_iterations:
                iterations += 1

                upper_vol = 0
                lower_vol = 0

                # Check if we can expand up
                if upper_idx < self.config.n_rows - 1:
                    upper_vol = vol_at_level[upper_idx + 1]

                # Check if we can expand down
                if lower_idx > 0:
                    lower_vol = vol_at_level[lower_idx - 1]

                # If no volume in either direction, break
                if upper_vol == 0 and lower_vol == 0:
                    logger.debug(f"No volume to expand at iteration {iterations}")
                    break

                # If we can't expand anymore, break
                if upper_idx >= self.config.n_rows - 1 and lower_idx <= 0:
                    logger.debug(f"Reached boundaries at iteration {iterations}")
                    break

                # Expand in direction with more volume
                if upper_vol >= lower_vol and upper_idx < self.config.n_rows - 1:
                    upper_idx += 1
                    captured_vol += upper_vol
                elif lower_idx > 0:
                    lower_idx -= 1
                    captured_vol += lower_vol
                else:
                    # Can't expand anymore
                    logger.debug(f"Cannot expand further at iteration {iterations}")
                    break

                # Additional safeguard: Check if we're making progress
                if iterations > 50 and captured_vol < target_vol * 0.3:
                    logger.warning(f"Slow progress in value area calculation at idx {idx}")
                    break

            # Log if we hit the iteration limit
            if iterations >= self.config.max_value_area_iterations:
                logger.warning(f"Hit max iterations ({iterations}) in value area calculation at idx {idx}")

            # Store iteration count for debugging
            self.iteration_counts.append(iterations)

            vah = lowest_price + (upper_idx + 1) * row_height
            val = lowest_price + lower_idx * row_height

            # KEEPING the rolling approximation as backup
            rolling_high = window_data['high'].iloc[-50:].max()
            rolling_low = window_data['low'].iloc[-50:].min()
            rolling_mid = (rolling_high + rolling_low) / 2
            rolling_range = rolling_high - rolling_low

            approx_vah = rolling_mid + rolling_range * 0.25
            approx_val = rolling_mid - rolling_range * 0.25
            approx_poc = rolling_mid

            # Use calculated values if they make sense, otherwise use approximations
            if vah > val and vah > poc and poc > val:
                return poc, vah, val, poc
            else:
                logger.warning(f"Invalid VP values at idx {idx}, using approximations")
                return approx_poc, approx_vah, approx_val, approx_poc

        except Exception as e:
            logger.error(f"Error in volume profile calculation at idx {idx}: {e}")
            # Return safe default values
            current_price = self.data['close'].iloc[min(idx, len(self.data)-1)]
            return current_price, current_price * 1.01, current_price * 0.99, current_price

    def reset(self):
        """Reset environment to initial state."""
        self.current_idx = max(200, self.config.lookback)  # Start after lookback
        self.balance = self.config.initial_balance
        self.position = 0  # -1 short, 0 neutral, 1 long
        self.entry_price = 0
        self.stop_price = 0
        self.trades = []
        self.returns = []
        self.current_mode = "NONE"  # MR_LONG, MR_SHORT, CONT_LONG, CONT_SHORT

        # Reset debug counters
        self.iteration_counts = []

        return self._get_state()

    def _get_state(self):
        """Get current state including FULL Volume Profile features."""
        try:
            idx = min(self.current_idx, len(self.data) - 1)
            row = self.data.iloc[idx]

            # Calculate FULL Volume Profile
            poc, vah, val, actual_poc = self._calculate_volume_profile(idx)

            price = row['close']
            value_range = vah - val if vah > val else 0.01

            # Price position relative to value area
            inside_value = price >= val and price <= vah
            above_value = price > vah
            below_value = price < val

            # Distance to key levels (normalized)
            dist_to_vah = (price - vah) / value_range if value_range > 0 else 0
            dist_to_val = (price - val) / value_range if value_range > 0 else 0
            dist_to_poc = (price - poc) / value_range if value_range > 0 else 0

            # Mean reversion zones
            vah_buffer = value_range * self.config.mr_buffer_pct
            val_buffer = value_range * self.config.mr_buffer_pct
            near_vah = price >= (vah - vah_buffer) and price <= vah
            near_val = price <= (val + val_buffer) and price >= val

            # Continuation zones
            cont_long_zone = price > vah * (1.0 + self.config.cont_buffer_pct)
            cont_short_zone = price < val * (1.0 - self.config.cont_buffer_pct)

            # Build FULL feature vector
            features = [
                # Volume Profile features
                dist_to_poc,
                dist_to_vah,
                dist_to_val,
                value_range / price if price > 0 else 0,  # Normalized value range

                # Zone indicators
                float(inside_value),
                float(above_value),
                float(below_value),
                float(near_vah),
                float(near_val),
                float(cont_long_zone),
                float(cont_short_zone),

                # Price features
                row['returns'] if not np.isnan(row['returns']) else 0,
                row['log_returns'] if not np.isnan(row['log_returns']) else 0,
                (price - row['sma_20']) / row['sma_20'] if row['sma_20'] > 0 else 0,
                (price - row['sma_50']) / row['sma_50'] if row['sma_50'] > 0 else 0,

                # Volume features
                row['volume_ratio'] if not np.isnan(row['volume_ratio']) else 1,
                np.log1p(row['volume'] / 1e6) if row['volume'] > 0 else 0,  # Log scaled volume

                # Technical indicators
                row['rsi'] / 100 if not np.isnan(row['rsi']) else 0.5,
                row['atr'] / price if price > 0 and not np.isnan(row['atr']) else 0,

                # Position info
                self.position,
                (price - self.entry_price) / self.entry_price if self.entry_price > 0 else 0,

                # Recent performance
                np.mean(self.returns[-10:]) if len(self.returns) > 0 else 0,
                len(self.trades) / max(1, self.current_idx - 200),  # Trade frequency

                # Market microstructure
                (row['high'] - row['low']) / price if price > 0 else 0,  # Range
                (row['close'] - row['open']) / row['open'] if row['open'] > 0 else 0,

                # Time features
                idx / len(self.data),  # Progress through dataset

                # Mode encoding
                float(self.current_mode == "MR_LONG"),
                float(self.current_mode == "MR_SHORT"),
                float(self.current_mode == "CONT_LONG"),
                float(self.current_mode == "CONT_SHORT"),
            ]

            # Ensure correct dimension
            features = features[:self.config.state_dim]
            while len(features) < self.config.state_dim:
                features.append(0)

            # Replace NaN or inf values
            features = [float(f) if np.isfinite(f) else 0.0 for f in features]

            return np.array(features, dtype=np.float32)

        except Exception as e:
            logger.error(f"Error in _get_state at idx {self.current_idx}: {e}")
            return np.zeros(self.config.state_dim, dtype=np.float32)

    def step(self, action):
        """Execute action - FULL STRATEGY LOGIC."""
        try:
            idx = min(self.current_idx, len(self.data) - 1)
            row = self.data.iloc[idx]
            price = row['close']
            atr = row.get('atr', price * 0.01)

            # Calculate FULL Volume Profile
            poc, vah, val, _ = self._calculate_volume_profile(idx)
            value_range = vah - val if vah > val else 0.01

            # Price zones
            inside_value = price >= val and price <= vah
            above_value = price > vah
            below_value = price < val

            # Mean reversion zones
            vah_buffer = value_range * self.config.mr_buffer_pct
            val_buffer = value_range * self.config.mr_buffer_pct
            near_vah = price >= (vah - vah_buffer) and price <= vah and inside_value
            near_val = price <= (val + val_buffer) and price >= val and inside_value

            # Continuation zones
            cont_long_zone = price > vah * (1.0 + self.config.cont_buffer_pct)
            cont_short_zone = price < val * (1.0 - self.config.cont_buffer_pct)

            reward = 0
            old_position = self.position

            # FULL STRATEGY LOGIC
            if action == 4:  # Strong Buy
                if cont_long_zone and self.position <= 0:  # Continuation long
                    # Close short if exists
                    if self.position < 0:
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
                    reward += 2  # Bonus for continuation trade

            elif action == 3:  # Buy
                if near_val and self.position <= 0:  # Mean reversion long
                    # Close short if exists
                    if self.position < 0:
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
                    reward += 1  # Bonus for mean reversion trade

            elif action == 1:  # Sell
                if near_vah and self.position >= 0:  # Mean reversion short
                    # Close long if exists
                    if self.position > 0:
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
                    reward += 1  # Bonus for mean reversion trade

            elif action == 0:  # Strong Sell
                if cont_short_zone and self.position >= 0:  # Continuation short
                    # Close long if exists
                    if self.position > 0:
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
                    reward += 2  # Bonus for continuation trade

            # Check exit conditions - FULL LOGIC
            if self.position != 0:
                # Mean reversion exits (target POC)
                if self.current_mode == "MR_LONG" and price >= poc:
                    returns = (price - self.entry_price) / self.entry_price
                    self.returns.append(returns)
                    self.balance *= (1 + returns)
                    reward += returns * 100 + 5  # Bonus for hitting target
                    self.position = 0
                    self.current_mode = "NONE"

                elif self.current_mode == "MR_SHORT" and price <= poc:
                    returns = (self.entry_price - price) / self.entry_price
                    self.returns.append(returns)
                    self.balance *= (1 + returns)
                    reward += returns * 100 + 5  # Bonus for hitting target
                    self.position = 0
                    self.current_mode = "NONE"

                # Continuation exits (return to value area)
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

                # Stop loss check
                if self.config.use_atr_stop and self.stop_price > 0:
                    if self.position > 0 and row['low'] <= self.stop_price:
                        returns = (self.stop_price - self.entry_price) / self.entry_price
                        self.returns.append(returns)
                        self.balance *= (1 + returns)
                        reward += returns * 100 - 5  # Penalty for stop loss
                        self.position = 0
                        self.current_mode = "NONE"

                    elif self.position < 0 and row['high'] >= self.stop_price:
                        returns = (self.entry_price - self.stop_price) / self.entry_price
                        self.returns.append(returns)
                        self.balance *= (1 + returns)
                        reward += returns * 100 - 5  # Penalty for stop loss
                        self.position = 0
                        self.current_mode = "NONE"

            # Holding reward/penalty
            if self.position != 0 and self.entry_price > 0:
                # Unrealized P&L
                if self.position > 0:
                    unrealized = (price - self.entry_price) / self.entry_price
                else:
                    unrealized = (self.entry_price - price) / self.entry_price
                reward += unrealized * 10

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

                # Final reward based on total performance
                total_return = (self.balance - self.config.initial_balance) / self.config.initial_balance
                reward += total_return * 500

            # Get new state
            new_state = self._get_state()

            # Log iteration statistics periodically
            if len(self.iteration_counts) > 0 and len(self.iteration_counts) % 100 == 0:
                avg_iterations = np.mean(self.iteration_counts[-100:])
                max_iterations = np.max(self.iteration_counts[-100:])
                logger.info(f"VP iterations - Avg: {avg_iterations:.1f}, Max: {max_iterations}")

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
            self.current_idx += 1
            done = self.current_idx >= len(self.data) - 1
            return self._get_state(), 0.0, done, {}

class VolumeProfilePPOAgent:
    """PPO Agent for Volume Profile strategy - FULL IMPLEMENTATION."""

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
        """FULL PPO update step."""
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
                end = start + self.config.batch_size
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
        return {'loss': total_loss / (self.config.n_epochs * (len(states) // self.config.batch_size + 1))}

def train_volume_profile_robust():
    """Train ROBUST PPO on Volume Profile strategy."""

    print("\n" + "="*70)
    print("🚀 TRAINING ROBUST PPO WITH FULL VOLUME PROFILE STRATEGY")
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

    print(f"\n⚙️ ROBUST Configuration:")
    print(f"  Lookback: {config.lookback} bars (FULL)")
    print(f"  Price Levels: {config.n_rows} (FULL)")
    print(f"  Value Area: {config.value_area_pct:.0%}")
    print(f"  Max VP iterations: {config.max_value_area_iterations}")
    print(f"  State dim: {config.state_dim} (FULL)")
    print(f"  Hidden dim: {config.hidden_dim} (FULL)")
    print(f"  Total timesteps: {config.total_timesteps:,}")

    # Training metrics
    episode_returns = []
    episode_trades = []
    mode_counts = {'MR_LONG': 0, 'MR_SHORT': 0, 'CONT_LONG': 0, 'CONT_SHORT': 0}

    # Training loop
    print(f"\n🎯 Training for {config.total_timesteps:,} timesteps...")
    pbar = tqdm(total=config.total_timesteps, desc="Training")

    total_steps = 0
    best_val_return = -float('inf')

    while total_steps < config.total_timesteps:
        state = env.reset()
        episode_reward = 0

        while True:
            action, log_prob, value = agent.select_action(state)
            next_state, reward, done, info = env.step(action)

            # Store experience
            agent.states.append(state)
            agent.actions.append(action)
            agent.rewards.append(reward)
            agent.values.append(value)
            agent.log_probs.append(log_prob)
            agent.dones.append(done)

            episode_reward += reward
            total_steps += 1
            pbar.update(1)

            # Update policy
            if len(agent.states) >= config.n_steps:
                loss_info = agent.update()

            if done:
                episode_returns.append(info.get('total_return', 0))
                episode_trades.append(info.get('n_trades', 0))

                # Track modes
                for trade in env.trades:
                    trade_type = trade['type']
                    if 'mr_long' in trade_type:
                        mode_counts['MR_LONG'] += 1
                    elif 'mr_short' in trade_type:
                        mode_counts['MR_SHORT'] += 1
                    elif 'cont_long' in trade_type:
                        mode_counts['CONT_LONG'] += 1
                    elif 'cont_short' in trade_type:
                        mode_counts['CONT_SHORT'] += 1

                # Update progress
                if len(episode_returns) % 10 == 0:
                    avg_return = np.mean(episode_returns[-10:])
                    avg_trades = np.mean(episode_trades[-10:])
                    pbar.set_description(
                        f"Return: {avg_return:.2%} | Trades: {avg_trades:.0f} | "
                        f"MR: {mode_counts['MR_LONG']+mode_counts['MR_SHORT']} | "
                        f"Cont: {mode_counts['CONT_LONG']+mode_counts['CONT_SHORT']}"
                    )

                    # Validation check
                    if len(episode_returns) % 50 == 0:
                        val_env = VolumeProfileEnvironment(val_data, config)
                        val_state = val_env.reset()

                        for _ in range(len(val_data) - config.lookback - 1):
                            val_action, _, _ = agent.select_action(val_state, deterministic=True)
                            val_state, _, val_done, val_info = val_env.step(val_action)
                            if val_done:
                                break

                        val_return = val_info.get('total_return', 0)
                        if val_return > best_val_return:
                            best_val_return = val_return
                            torch.save(agent.network.state_dict(), MODELS_DIR / 'ppo_volume_profile_robust.pth')

                break

            state = next_state

    pbar.close()

    # Load best model
    if (MODELS_DIR / 'ppo_volume_profile_robust.pth').exists():
        agent.network.load_state_dict(torch.load(MODELS_DIR / 'ppo_volume_profile_robust.pth'))

    # Test evaluation
    print("\n📊 TESTING ON HOLDOUT DATA")
    test_env = VolumeProfileEnvironment(test_data, config)
    state = test_env.reset()

    for _ in range(len(test_data) - config.lookback - 1):
        action, _, _ = agent.select_action(state, deterministic=True)
        state, _, done, info = test_env.step(action)
        if done:
            break

    # Count trade types
    test_mode_counts = {'MR_LONG': 0, 'MR_SHORT': 0, 'CONT_LONG': 0, 'CONT_SHORT': 0}
    for trade in test_env.trades:
        trade_type = trade['type']
        if 'mr_long' in trade_type:
            test_mode_counts['MR_LONG'] += 1
        elif 'mr_short' in trade_type:
            test_mode_counts['MR_SHORT'] += 1
        elif 'cont_long' in trade_type:
            test_mode_counts['CONT_LONG'] += 1
        elif 'cont_short' in trade_type:
            test_mode_counts['CONT_SHORT'] += 1

    test_return = info.get('total_return', 0)

    print(f"\n🎯 Test Results:")
    print(f"  Return: {test_return:.2%}")
    print(f"  Final Balance: ${info.get('balance', 10000):.2f}")
    print(f"  Total Trades: {info.get('n_trades', 0)}")
    print(f"  Win Rate: {info.get('win_rate', 0):.1%}")
    print(f"\n  Trade Breakdown:")
    print(f"    Mean Reversion Long: {test_mode_counts['MR_LONG']}")
    print(f"    Mean Reversion Short: {test_mode_counts['MR_SHORT']}")
    print(f"    Continuation Long: {test_mode_counts['CONT_LONG']}")
    print(f"    Continuation Short: {test_mode_counts['CONT_SHORT']}")

    # Save metrics
    metrics = {
        'model': 'PPO_Volume_Profile_Robust',
        'strategy': 'Mean Reversion & Continuation (ROBUST)',
        'test_return': float(test_return),
        'test_trades': info.get('n_trades', 0),
        'test_win_rate': float(info.get('win_rate', 0)),
        'trade_breakdown': test_mode_counts,
        'config': {
            'lookback': config.lookback,
            'n_rows': config.n_rows,
            'value_area_pct': config.value_area_pct,
            'max_iterations': config.max_value_area_iterations,
            'mr_buffer': config.mr_buffer_pct,
            'cont_buffer': config.cont_buffer_pct,
            'atr_stop': config.use_atr_stop,
            'atr_mult': config.atr_multiplier
        },
        'trained_at': datetime.utcnow().isoformat()
    }

    with open(MODELS_DIR / 'ppo_volume_profile_robust_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "="*70)
    if test_return > 0.02:  # 2% return threshold
        print(f"✅ SUCCESS! ROBUST Volume Profile strategy achieved {test_return:.2%} return!")
    else:
        print(f"📊 ROBUST Volume Profile strategy completed with {test_return:.2%} return")
    print("="*70)

    return agent, metrics

if __name__ == "__main__":
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    agent, metrics = train_volume_profile_robust()