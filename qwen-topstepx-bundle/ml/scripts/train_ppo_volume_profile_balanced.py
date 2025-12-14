#!/usr/bin/env python3
"""
BALANCED Volume Profile PPO - Good complexity with stability
Maintains meaningful volume profile calculations while avoiding computational bottlenecks
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

class BalancedVolumeProfileConfig:
    """Balanced configuration - good complexity with stability"""
    # Volume Profile parameters - balanced for performance
    lookback_bars = 100  # Reasonable lookback (8+ hours of 5-min data)
    num_price_levels = 50  # Good granularity without overdoing it
    value_area_pct = 0.70
    max_value_area_iterations = 100  # Safety limit

    # State and action spaces
    state_dim = 25  # Good feature set
    action_dim = 3  # Buy, Sell, Hold
    hidden_dim = 192  # Moderate network size

    # PPO hyperparameters
    learning_rate = 3e-4
    gamma = 0.99
    lambda_gae = 0.95
    clip_epsilon = 0.2
    value_loss_coef = 0.5
    entropy_coef = 0.01
    max_grad_norm = 1.0

    # Training configuration
    batch_size = 128
    n_epochs = 4
    n_steps = 512
    total_timesteps = 100000

    # Trading environment
    initial_balance = 10000
    transaction_cost = 0.0
    max_position_size = 1.0
    atr_multiplier = 1.5
    min_confidence = 0.3

class BalancedNetwork(nn.Module):
    """Balanced neural network architecture"""

    def __init__(self, state_dim, action_dim, hidden_dim=192):
        super().__init__()

        # Shared feature extractor with dropout
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # Separate heads for actor and critic
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )

        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state)
        if state.dim() == 1:
            state = state.unsqueeze(0)

        features = self.feature_extractor(state)
        action_logits = self.actor_head(features)
        value = self.critic_head(features)

        return action_logits.squeeze(0), value.squeeze()

    def get_action(self, state, deterministic=False):
        action_logits, value = self.forward(state)

        if deterministic:
            action = torch.argmax(action_logits).item()
            return action, None, value.item()

        dist = Categorical(logits=action_logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob.item(), value.item()

class BalancedVolumeProfileEnv:
    """Balanced Volume Profile environment with optimized calculations"""

    def __init__(self, data, config):
        self.data = data
        self.config = config

        # Pre-calculate technical indicators
        self._precompute_indicators()
        self.reset()

    def _precompute_indicators(self):
        """Pre-compute indicators to speed up training"""
        # Simple moving averages
        self.data['sma_20'] = self.data['close'].rolling(20, min_periods=1).mean()
        self.data['sma_50'] = self.data['close'].rolling(50, min_periods=1).mean()

        # RSI
        delta = self.data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.data['rsi'] = 100 - (100 / (1 + rs))

        # ATR
        high_low = self.data['high'] - self.data['low']
        high_close = np.abs(self.data['high'] - self.data['close'].shift())
        low_close = np.abs(self.data['low'] - self.data['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        self.data['atr'] = true_range.rolling(14).mean()

    def reset(self):
        """Reset environment to initial state"""
        self.current_idx = self.config.lookback_bars + 10
        self.balance = self.config.initial_balance
        self.position = 0  # 1 = long, -1 = short, 0 = neutral
        self.entry_price = 0
        self.trades = []
        self.returns = []

        return self._get_state()

    def _calculate_volume_profile_optimized(self, idx):
        """Optimized volume profile calculation"""
        try:
            # Get lookback period
            start_idx = max(0, idx - self.config.lookback_bars)
            lookback_data = self.data.iloc[start_idx:idx]

            if len(lookback_data) < 20:
                return 0.5, 0.6, 0.4, 0.0  # Default values

            # Get price range
            high = lookback_data['high'].max()
            low = lookback_data['low'].min()
            price_range = high - low

            if price_range <= 0:
                return 0.5, 0.6, 0.4, 0.0

            # Create price levels
            price_levels = np.linspace(low, high, self.config.num_price_levels)
            volume_profile = np.zeros(self.config.num_price_levels)

            # Calculate volume at each price level - optimized version
            for _, bar in lookback_data.iterrows():
                bar_high = bar['high']
                bar_low = bar['low']
                bar_volume = bar['volume']

                # Find price levels touched by this bar
                touched_levels = (price_levels >= bar_low) & (price_levels <= bar_high)
                n_touched = touched_levels.sum()

                if n_touched > 0:
                    # Distribute volume evenly across touched levels
                    volume_per_level = bar_volume / n_touched
                    volume_profile[touched_levels] += volume_per_level

            # Find POC (Point of Control)
            poc_idx = np.argmax(volume_profile)
            poc_price = price_levels[poc_idx]

            # Calculate Value Area - simplified version
            total_volume = volume_profile.sum()
            if total_volume == 0:
                return 0.5, 0.6, 0.4, 0.0

            target_volume = total_volume * self.config.value_area_pct

            # Start from POC and expand
            value_area_indices = {poc_idx}
            captured_volume = volume_profile[poc_idx]

            # Expand value area - with iteration limit
            iterations = 0
            while captured_volume < target_volume and iterations < self.config.max_value_area_iterations:
                iterations += 1

                # Check expansion candidates
                min_idx = min(value_area_indices)
                max_idx = max(value_area_indices)

                expand_up = expand_down = 0

                if max_idx < len(price_levels) - 1:
                    expand_up = volume_profile[max_idx + 1]
                if min_idx > 0:
                    expand_down = volume_profile[min_idx - 1]

                # Expand in direction with more volume
                if expand_up >= expand_down and max_idx < len(price_levels) - 1:
                    value_area_indices.add(max_idx + 1)
                    captured_volume += expand_up
                elif min_idx > 0:
                    value_area_indices.add(min_idx - 1)
                    captured_volume += expand_down
                else:
                    break  # Can't expand further

                # Safety check
                if len(value_area_indices) >= self.config.num_price_levels * 0.8:
                    break  # Value area is too wide

            # Get VAH and VAL
            vah_price = price_levels[max(value_area_indices)]
            val_price = price_levels[min(value_area_indices)]

            # Normalize to 0-1 range for features
            current_price = self.data.iloc[idx]['close']
            poc_normalized = (poc_price - low) / price_range
            vah_normalized = (vah_price - low) / price_range
            val_normalized = (val_price - low) / price_range
            price_position = (current_price - val_price) / (vah_price - val_price + 1e-8)

            return poc_normalized, vah_normalized, val_normalized, price_position

        except Exception as e:
            print(f"Warning: VP calculation error at idx {idx}: {e}")
            return 0.5, 0.6, 0.4, 0.0

    def _get_state(self):
        """Get current state with balanced features"""
        idx = min(self.current_idx, len(self.data) - 1)
        row = self.data.iloc[idx]

        # Volume Profile features
        poc, vah, val, price_position = self._calculate_volume_profile_optimized(idx)

        # Price features
        price = row['close']
        price_norm = (price - row['low']) / (row['high'] - row['low'] + 1e-8)

        # Technical indicators
        sma_20 = row.get('sma_20', price)
        sma_50 = row.get('sma_50', price)
        rsi = row.get('rsi', 50) / 100
        atr = row.get('atr', 0.01)

        # Returns over different periods
        returns_5 = (price / self.data.iloc[max(0, idx-5)]['close'] - 1) if idx > 5 else 0
        returns_20 = (price / self.data.iloc[max(0, idx-20)]['close'] - 1) if idx > 20 else 0
        returns_50 = (price / self.data.iloc[max(0, idx-50)]['close'] - 1) if idx > 50 else 0

        # Volume features
        volume = row['volume']
        volume_mean = self.data['volume'].iloc[max(0, idx-20):idx+1].mean()
        volume_ratio = volume / volume_mean if volume_mean > 0 else 1.0

        # Position information
        position_encoded = float(self.position)
        position_pnl = 0
        if self.position != 0 and self.entry_price > 0:
            if self.position > 0:
                position_pnl = (price - self.entry_price) / self.entry_price
            else:
                position_pnl = (self.entry_price - price) / self.entry_price

        # Market structure
        high_20 = self.data['high'].iloc[max(0, idx-20):idx+1].max()
        low_20 = self.data['low'].iloc[max(0, idx-20):idx+1].min()
        price_range_pos = (price - low_20) / (high_20 - low_20 + 1e-8)

        # Momentum
        momentum_5 = returns_5
        momentum_20 = returns_20

        # Combine all features
        features = [
            # Volume Profile (4)
            poc, vah, val, price_position,

            # Price features (3)
            price_norm, (price - sma_20) / price, (price - sma_50) / price,

            # Technical (2)
            rsi, atr / price,

            # Returns (3)
            returns_5, returns_20, returns_50,

            # Volume (2)
            volume_ratio, np.log1p(volume / 1e6),

            # Position (2)
            position_encoded, position_pnl,

            # Market structure (2)
            price_range_pos, (high_20 - low_20) / price,

            # Momentum (2)
            momentum_5, momentum_20,

            # Time features (3)
            idx / len(self.data),  # Progress through dataset
            float(idx % 78) / 78,  # Time of day (78 5-min bars per day)
            len(self.trades) / 100,  # Trading activity

            # Padding to reach state_dim
            0, 0
        ]

        return np.array(features[:self.config.state_dim], dtype=np.float32)

    def step(self, action):
        """Execute action and return results"""
        idx = min(self.current_idx, len(self.data) - 1)
        row = self.data.iloc[idx]
        price = row['close']

        reward = 0

        # Volume Profile signals
        poc, vah, val, price_position = self._calculate_volume_profile_optimized(idx)

        # Determine if we're in value area
        in_value_area = 0.2 < price_position < 0.8

        # Execute actions based on strategy
        if action == 0:  # Buy
            if self.position <= 0:
                # Close short if exists
                if self.position < 0:
                    returns = (self.entry_price - price) / self.entry_price
                    self.returns.append(returns)
                    reward = returns * 100
                    self.balance *= (1 + returns)

                # Open long
                self.position = 1
                self.entry_price = price
                self.trades.append({
                    'idx': idx,
                    'action': 'buy',
                    'price': price,
                    'in_value': in_value_area
                })

                # Reward based on strategy alignment
                if in_value_area:  # Mean reversion buy
                    reward += 0.5
                elif price_position < 0.1:  # Breakout support buy
                    reward += 0.3

        elif action == 1:  # Sell
            if self.position >= 0:
                # Close long if exists
                if self.position > 0:
                    returns = (price - self.entry_price) / self.entry_price
                    self.returns.append(returns)
                    reward = returns * 100
                    self.balance *= (1 + returns)

                # Open short
                self.position = -1
                self.entry_price = price
                self.trades.append({
                    'idx': idx,
                    'action': 'sell',
                    'price': price,
                    'in_value': in_value_area
                })

                # Reward based on strategy alignment
                if in_value_area:  # Mean reversion sell
                    reward += 0.5
                elif price_position > 0.9:  # Breakout resistance sell
                    reward += 0.3

        elif action == 2:  # Hold
            # Small penalty for holding with no position
            if self.position == 0:
                reward -= 0.01

        # Position-based rewards
        if self.position != 0:
            # Unrealized P&L
            if self.position > 0:
                unrealized_pnl = (price - self.entry_price) / self.entry_price
            else:
                unrealized_pnl = (self.entry_price - price) / self.entry_price

            reward += unrealized_pnl * 10  # Scale unrealized P&L

            # Risk management reward
            atr = row.get('atr', price * 0.01)
            risk_level = abs(price - self.entry_price) / atr
            if risk_level > 2:  # Position at risk
                reward -= 0.5

        # Move to next bar
        self.current_idx += 1

        # Check if done
        done = self.current_idx >= len(self.data) - 1

        if done:
            # Close final position
            if self.position != 0:
                if self.position > 0:
                    final_returns = (price - self.entry_price) / self.entry_price
                else:
                    final_returns = (self.entry_price - price) / self.entry_price
                self.returns.append(final_returns)
                self.balance *= (1 + final_returns)

            # Final reward based on performance
            total_return = (self.balance - self.config.initial_balance) / self.config.initial_balance
            reward += total_return * 200

            # Bonus for good win rate
            if len(self.returns) > 0:
                win_rate = sum(1 for r in self.returns if r > 0) / len(self.returns)
                if win_rate > 0.5:
                    reward += 50 * (win_rate - 0.5)

        # Get new state
        new_state = self._get_state()

        # Info for monitoring
        info = {
            'balance': self.balance,
            'total_return': (self.balance - self.config.initial_balance) / self.config.initial_balance,
            'n_trades': len(self.trades),
            'win_rate': sum(1 for r in self.returns if r > 0) / max(1, len(self.returns)),
            'position': self.position,
            'idx': self.current_idx
        }

        return new_state, reward, done, info

class BalancedPPOAgent:
    """PPO agent with balanced complexity"""

    def __init__(self, config):
        self.config = config
        self.network = BalancedNetwork(
            config.state_dim,
            config.action_dim,
            config.hidden_dim
        )
        self.optimizer = optim.Adam(self.network.parameters(), lr=config.learning_rate)
        self.clear_buffers()

    def clear_buffers(self):
        """Clear experience buffers"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def select_action(self, state, deterministic=False):
        """Select action using current policy"""
        return self.network.get_action(state, deterministic)

    def store_transition(self, state, action, reward, value, log_prob, done):
        """Store transition in buffer"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def update(self):
        """Update policy using PPO"""
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

            td_error = rewards[t] + self.config.gamma * (
                old_values[t + 1] if t < len(rewards) - 1 and not dones[t] else 0
            ) - old_values[t]

            running_advantage = td_error + self.config.gamma * self.config.lambda_gae * running_advantage
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

            # Clipped surrogate loss
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_pred_clipped = old_values + torch.clamp(
                values.squeeze() - old_values,
                -self.config.clip_epsilon,
                self.config.clip_epsilon
            )
            value_losses = (values.squeeze() - returns) ** 2
            value_losses_clipped = (value_pred_clipped - returns) ** 2
            value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()

            # Entropy bonus
            entropy = dist.entropy().mean()

            # Total loss
            loss = (
                policy_loss +
                self.config.value_loss_coef * value_loss -
                self.config.entropy_coef * entropy
            )

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.config.max_grad_norm)
            self.optimizer.step()

            total_loss = loss.item()

        self.clear_buffers()

        return {
            'loss': total_loss,
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item()
        }

def train_balanced():
    """Train balanced Volume Profile PPO"""

    print("\n" + "="*70)
    print("⚖️ BALANCED VOLUME PROFILE PPO TRAINING")
    print("="*70)

    # Load data
    data_path = DATA_DIR / 'spy_5min_data.parquet'
    if not data_path.exists():
        print("❌ Data file not found!")
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
    config = BalancedVolumeProfileConfig()
    env = BalancedVolumeProfileEnv(train_data, config)
    agent = BalancedPPOAgent(config)

    print(f"\n⚡ Balanced Configuration:")
    print(f"  Lookback: {config.lookback_bars} bars")
    print(f"  Price levels: {config.num_price_levels}")
    print(f"  State dim: {config.state_dim}")
    print(f"  Hidden dim: {config.hidden_dim}")
    print(f"  Total timesteps: {config.total_timesteps:,}")

    # Training loop
    print(f"\n🎯 Training for {config.total_timesteps:,} timesteps...")

    episode_returns = []
    episode_trades = []
    episode_lengths = []

    pbar = tqdm(total=config.total_timesteps, desc="Training")
    total_steps = 0

    while total_steps < config.total_timesteps:
        state = env.reset()
        episode_reward = 0
        episode_length = 0

        while True:
            # Select action
            action, log_prob, value = agent.select_action(state)

            # Take step
            next_state, reward, done, info = env.step(action)

            # Store transition
            agent.store_transition(state, action, reward, value, log_prob, done)

            episode_reward += reward
            episode_length += 1
            total_steps += 1
            pbar.update(1)

            # Update if buffer is full
            if len(agent.states) >= config.n_steps:
                metrics = agent.update()
                if metrics:
                    pbar.set_postfix(
                        loss=f"{metrics['loss']:.4f}",
                        ret=f"{episode_returns[-1] if episode_returns else 0:.2%}"
                    )

            if done:
                episode_returns.append(info['total_return'])
                episode_trades.append(info['n_trades'])
                episode_lengths.append(episode_length)

                # Log progress
                if len(episode_returns) % 10 == 0:
                    recent_returns = episode_returns[-10:]
                    recent_trades = episode_trades[-10:]
                    print(f"\n📈 Last 10 episodes:")
                    print(f"  Avg return: {np.mean(recent_returns):.2%}")
                    print(f"  Avg trades: {np.mean(recent_trades):.1f}")
                    print(f"  Best return: {max(recent_returns):.2%}")

                break

            state = next_state

    pbar.close()

    # Testing
    print("\n" + "="*70)
    print("📊 TESTING ON HOLDOUT DATA")
    print("="*70)

    test_env = BalancedVolumeProfileEnv(test_data, config)
    state = test_env.reset()

    test_trades = []
    for step in range(len(test_data) - config.lookback_bars - 11):
        action, _, _ = agent.select_action(state, deterministic=True)
        state, _, done, info = test_env.step(action)

        if len(test_env.trades) > len(test_trades):
            test_trades = test_env.trades.copy()

        if done:
            break

    # Results
    test_return = info['total_return']
    test_win_rate = info['win_rate']

    print(f"\n🎯 Test Results:")
    print(f"  Total Return: {test_return:.2%}")
    print(f"  Final Balance: ${info['balance']:.2f}")
    print(f"  Number of Trades: {info['n_trades']}")
    print(f"  Win Rate: {test_win_rate:.1%}")

    # Analyze trades
    if test_trades:
        value_area_trades = sum(1 for t in test_trades if t.get('in_value', False))
        breakout_trades = len(test_trades) - value_area_trades
        print(f"\n📊 Trade Analysis:")
        print(f"  Value Area Trades: {value_area_trades}")
        print(f"  Breakout Trades: {breakout_trades}")

    # Save model
    torch.save(agent.network.state_dict(), MODELS_DIR / 'ppo_volume_profile_balanced.pth')

    # Save metrics
    metrics = {
        'model': 'PPO_VolumeProfile_Balanced',
        'config': {
            'lookback_bars': config.lookback_bars,
            'num_price_levels': config.num_price_levels,
            'state_dim': config.state_dim,
            'hidden_dim': config.hidden_dim
        },
        'test_results': {
            'return': float(test_return),
            'trades': info['n_trades'],
            'win_rate': float(test_win_rate),
            'final_balance': float(info['balance'])
        },
        'training': {
            'total_steps': total_steps,
            'episodes': len(episode_returns),
            'avg_return': float(np.mean(episode_returns)),
            'best_return': float(max(episode_returns)) if episode_returns else 0
        },
        'trained_at': datetime.utcnow().isoformat()
    }

    with open(MODELS_DIR / 'ppo_volume_profile_balanced_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "="*70)
    if test_return > 0:
        print(f"✅ SUCCESS! Balanced VP Strategy achieved {test_return:.2%} return!")
    else:
        print(f"📊 Balanced VP Strategy completed with {test_return:.2%} return")
    print("="*70)

    return agent, metrics

if __name__ == "__main__":
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    agent, metrics = train_balanced()