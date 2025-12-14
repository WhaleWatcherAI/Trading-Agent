#!/usr/bin/env python3
"""
Volume Profile PPO with 200-bar lookback and proper mean reversion/continuation logic
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

class VolumeProfileConfig:
    """Configuration for 200-bar Volume Profile strategy"""
    # Volume Profile parameters
    lookback_bars = 200  # Back to 200 bars as requested
    num_price_levels = 100  # Keep good resolution
    value_area_pct = 0.70
    max_value_area_iterations = 100  # Increased for 200 bars

    # PPO parameters
    state_dim = 30  # Increased for more features
    action_dim = 3  # Buy, Sell, Hold
    hidden_dim = 256  # Larger network for complex patterns

    learning_rate = 3e-4
    gamma = 0.99
    lambda_gae = 0.95
    clip_epsilon = 0.2
    value_loss_coef = 0.5
    entropy_coef = 0.01
    max_grad_norm = 1.0

    # Training
    batch_size = 64
    n_epochs = 4
    n_steps = 256
    total_timesteps = 100000

    # Environment
    initial_balance = 10000
    transaction_cost = 0.0  # No transaction costs as requested
    max_position_size = 1.0
    atr_multiplier = 2.0  # For stop loss

class VPEnvironment:
    """Volume Profile environment with correct strategy logic"""

    def __init__(self, data, config):
        self.data = data
        self.config = config
        self.reset()

    def reset(self):
        """Reset environment"""
        self.current_idx = self.config.lookback_bars + 10
        self.balance = self.config.initial_balance
        self.position = 0  # 1 = long, -1 = short, 0 = flat
        self.entry_price = 0
        self.trades = []
        self.returns = []
        self.value_area_trades = 0
        self.breakout_trades = 0

        return self._get_state()

    def _calculate_volume_profile(self, idx):
        """Calculate Volume Profile with POC, VAH, VAL"""
        try:
            # Get lookback data (200 bars)
            start_idx = max(0, idx - self.config.lookback_bars)
            lookback_data = self.data.iloc[start_idx:idx]

            if len(lookback_data) < 50:  # Need minimum data
                return 0.5, 0.6, 0.4, 0.5

            # Find price range
            high = lookback_data['high'].max()
            low = lookback_data['low'].min()
            price_range = high - low

            if price_range <= 0:
                return 0.5, 0.6, 0.4, 0.5

            # Create price levels
            price_levels = np.linspace(low, high, self.config.num_price_levels)
            volume_profile = np.zeros(self.config.num_price_levels)

            # Calculate volume at each price level
            for _, bar in lookback_data.iterrows():
                bar_high = bar['high']
                bar_low = bar['low']
                bar_volume = bar['volume']

                # Find price levels this bar touches
                touched_levels = np.where((price_levels >= bar_low) & (price_levels <= bar_high))[0]
                if len(touched_levels) > 0:
                    # Distribute volume equally among touched levels
                    volume_per_level = bar_volume / len(touched_levels)
                    volume_profile[touched_levels] += volume_per_level

            # Find POC (Point of Control)
            poc_idx = np.argmax(volume_profile)
            poc_price = price_levels[poc_idx]

            # Calculate Value Area (70% of volume)
            total_volume = volume_profile.sum()
            if total_volume == 0:
                return 0.5, 0.6, 0.4, 0.5

            target_volume = total_volume * self.config.value_area_pct

            # Start from POC and expand
            value_area_indices = {poc_idx}
            captured_volume = volume_profile[poc_idx]

            iterations = 0
            while captured_volume < target_volume and iterations < self.config.max_value_area_iterations:
                iterations += 1

                min_idx = min(value_area_indices)
                max_idx = max(value_area_indices)

                # Check expansion options
                expand_up = volume_profile[max_idx + 1] if max_idx < len(volume_profile) - 1 else 0
                expand_down = volume_profile[min_idx - 1] if min_idx > 0 else 0

                # Expand in direction with more volume
                if expand_up >= expand_down and max_idx < len(volume_profile) - 1:
                    value_area_indices.add(max_idx + 1)
                    captured_volume += expand_up
                elif min_idx > 0:
                    value_area_indices.add(min_idx - 1)
                    captured_volume += expand_down
                else:
                    break

            # Get VAH and VAL
            vah_price = price_levels[max(value_area_indices)]
            val_price = price_levels[min(value_area_indices)]

            # Calculate current price position
            current_price = self.data.iloc[idx]['close']

            # Price position relative to value area
            # < 0: Below VAL, 0-1: Inside value area, > 1: Above VAH
            if current_price < val_price:
                price_position = (current_price - val_price) / (vah_price - val_price + 1e-8)
            elif current_price > vah_price:
                price_position = (current_price - val_price) / (vah_price - val_price + 1e-8)
            else:
                price_position = (current_price - val_price) / (vah_price - val_price + 1e-8)

            # Normalize for features
            poc_normalized = (poc_price - low) / price_range
            vah_normalized = (vah_price - low) / price_range
            val_normalized = (val_price - low) / price_range

            return poc_normalized, vah_normalized, val_normalized, price_position

        except Exception as e:
            print(f"Warning: VP calculation error at idx {idx}: {e}")
            return 0.5, 0.6, 0.4, 0.5

    def _get_state(self):
        """Get current state with all features"""
        idx = min(self.current_idx, len(self.data) - 1)
        row = self.data.iloc[idx]

        # Volume Profile features
        poc, vah, val, price_position = self._calculate_volume_profile(idx)

        # Price features
        price = row['close']
        price_norm = (price - row['low']) / (row['high'] - row['low'] + 1e-8)

        # Distance to VP levels
        distance_to_poc = abs(poc - 0.5)  # How far POC is from midpoint
        distance_to_vah = abs(price_norm - vah)
        distance_to_val = abs(price_norm - val)

        # Technical indicators
        sma_20 = row.get('sma_20', price)
        sma_50 = row.get('sma_50', price)
        sma_200 = row.get('sma_200', price)  # Add 200 SMA
        rsi = row.get('rsi', 50) / 100
        atr = row.get('atr', 0.01)

        # Returns over different periods
        returns_5 = (price / self.data.iloc[max(0, idx-5)]['close'] - 1) if idx > 5 else 0
        returns_20 = (price / self.data.iloc[max(0, idx-20)]['close'] - 1) if idx > 20 else 0
        returns_50 = (price / self.data.iloc[max(0, idx-50)]['close'] - 1) if idx > 50 else 0
        returns_200 = (price / self.data.iloc[max(0, idx-200)]['close'] - 1) if idx > 200 else 0

        # Volume features
        volume = row['volume']
        volume_mean_20 = self.data['volume'].iloc[max(0, idx-20):idx+1].mean()
        volume_mean_50 = self.data['volume'].iloc[max(0, idx-50):idx+1].mean()
        volume_ratio_20 = volume / volume_mean_20 if volume_mean_20 > 0 else 1.0
        volume_ratio_50 = volume / volume_mean_50 if volume_mean_50 > 0 else 1.0

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

        # Combine features
        features = [
            # Volume Profile (7)
            poc, vah, val, price_position,
            distance_to_poc, distance_to_vah, distance_to_val,

            # Price features (4)
            price_norm,
            (price - sma_20) / price,
            (price - sma_50) / price,
            (price - sma_200) / price,

            # Returns (4)
            returns_5, returns_20, returns_50, returns_200,

            # Volume (2)
            volume_ratio_20, volume_ratio_50,

            # Technical (3)
            rsi, atr / price, price_range_pos,

            # Position (2)
            position_encoded, position_pnl,

            # Market info (2)
            float(idx % 78) / 78,  # Time of day
            len(self.trades) / 100,  # Trading activity

            # Padding
            0, 0, 0, 0, 0
        ]

        return np.array(features[:self.config.state_dim], dtype=np.float32)

    def step(self, action):
        """Execute action with proper mean reversion and continuation logic"""
        idx = min(self.current_idx, len(self.data) - 1)
        row = self.data.iloc[idx]
        price = row['close']

        reward = 0

        # Get Volume Profile levels
        poc, vah, val, price_position = self._calculate_volume_profile(idx)

        # Determine trading zone
        # price_position < 0: Below VAL (breakout short zone)
        # 0 <= price_position <= 0.2: Near VAL (mean reversion buy zone)
        # 0.2 < price_position < 0.8: Middle of value area (neutral)
        # 0.8 <= price_position <= 1.0: Near VAH (mean reversion sell zone)
        # price_position > 1: Above VAH (breakout long zone)

        near_val = 0 <= price_position <= 0.2  # Near lower edge
        near_vah = 0.8 <= price_position <= 1.0  # Near upper edge
        below_val = price_position < 0  # Below value area
        above_vah = price_position > 1  # Above value area
        in_middle = 0.2 < price_position < 0.8  # Middle of value area

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

                # Determine trade type and reward
                if near_val:  # Mean reversion buy near VAL
                    self.value_area_trades += 1
                    reward += 1.0  # Strong reward for mean reversion
                    trade_type = 'mean_reversion_val'
                elif above_vah:  # Breakout continuation buy above VAH
                    self.breakout_trades += 1
                    reward += 0.8  # Good reward for breakout
                    trade_type = 'breakout_long'
                elif below_val:  # Wrong direction below VAL
                    reward -= 0.5  # Penalty
                    trade_type = 'wrong_below_val'
                else:
                    trade_type = 'other'

                self.trades.append({
                    'idx': idx,
                    'action': 'buy',
                    'price': price,
                    'type': trade_type,
                    'price_position': price_position
                })

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

                # Determine trade type and reward
                if near_vah:  # Mean reversion sell near VAH
                    self.value_area_trades += 1
                    reward += 1.0  # Strong reward for mean reversion
                    trade_type = 'mean_reversion_vah'
                elif below_val:  # Breakout continuation sell below VAL
                    self.breakout_trades += 1
                    reward += 0.8  # Good reward for breakout
                    trade_type = 'breakout_short'
                elif above_vah:  # Wrong direction above VAH
                    reward -= 0.5  # Penalty
                    trade_type = 'wrong_above_vah'
                else:
                    trade_type = 'other'

                self.trades.append({
                    'idx': idx,
                    'action': 'sell',
                    'price': price,
                    'type': trade_type,
                    'price_position': price_position
                })

        elif action == 2:  # Hold
            # Small reward for holding in middle of value area
            if in_middle and self.position == 0:
                reward += 0.1
            # Penalty for holding with no position in good zones
            elif (near_val or near_vah or below_val or above_vah) and self.position == 0:
                reward -= 0.2

        # Position management rewards
        if self.position != 0:
            # Unrealized P&L
            if self.position > 0:
                unrealized_pnl = (price - self.entry_price) / self.entry_price
            else:
                unrealized_pnl = (self.entry_price - price) / self.entry_price

            reward += unrealized_pnl * 20  # Strong weight on P&L

            # Risk management
            atr = row.get('atr', price * 0.01)
            risk_level = abs(price - self.entry_price) / atr
            if risk_level > self.config.atr_multiplier:
                reward -= 1.0  # Penalty for excessive risk

        # Move to next bar
        self.current_idx += 1

        # Check if done
        done = self.current_idx >= len(self.data) - 1

        if done:
            # Final position cleanup
            if self.position != 0:
                if self.position > 0:
                    returns = (price - self.entry_price) / self.entry_price
                else:
                    returns = (self.entry_price - price) / self.entry_price
                self.returns.append(returns)
                self.balance *= (1 + returns)

        new_state = self._get_state()

        info = {
            'balance': self.balance,
            'position': self.position,
            'n_trades': len(self.trades),
            'value_area_trades': self.value_area_trades,
            'breakout_trades': self.breakout_trades
        }

        return new_state, reward, done, info


class PPOAgent:
    """PPO agent for Volume Profile trading"""

    def __init__(self, config):
        self.config = config

        # Networks
        self.network = nn.Sequential(
            nn.Linear(config.state_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU()
        )

        self.actor = nn.Linear(config.hidden_dim, config.action_dim)
        self.critic = nn.Linear(config.hidden_dim, 1)

        self.optimizer = optim.Adam(
            list(self.network.parameters()) +
            list(self.actor.parameters()) +
            list(self.critic.parameters()),
            lr=config.learning_rate
        )

        self.clear_buffers()

    def clear_buffers(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def select_action(self, state):
        """Select action using policy"""
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            features = self.network(state_t)

            action_logits = self.actor(features)
            value = self.critic(features)

            dist = Categorical(logits=action_logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            return action.item(), log_prob.item(), value.item()

    def update(self):
        """Update policy using PPO"""
        if len(self.states) < self.config.batch_size:
            return {}

        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states))
        actions = torch.LongTensor(self.actions)
        rewards = torch.FloatTensor(self.rewards)
        values = torch.FloatTensor(self.values)
        log_probs = torch.FloatTensor(self.log_probs)
        dones = torch.FloatTensor(self.dones)

        # Calculate advantages using GAE
        advantages = []
        returns = []
        advantage = 0
        next_value = 0

        for t in reversed(range(len(rewards))):
            if dones[t]:
                advantage = 0
                next_value = 0

            delta = rewards[t] + self.config.gamma * next_value - values[t]
            advantage = delta + self.config.gamma * self.config.lambda_gae * advantage
            next_value = values[t]

            advantages.insert(0, advantage)
            returns.insert(0, advantage + values[t])

        advantages = torch.FloatTensor(advantages)
        returns = torch.FloatTensor(returns)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update
        for _ in range(self.config.n_epochs):
            # Forward pass
            features = self.network(states)
            action_logits = self.actor(features)
            values_pred = self.critic(features)

            dist = Categorical(logits=action_logits)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            # Calculate losses
            ratio = torch.exp(new_log_probs - log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * advantages

            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(values_pred.squeeze(), returns)

            loss = policy_loss + self.config.value_loss_coef * value_loss - self.config.entropy_coef * entropy

            # Update
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.network.parameters()) +
                list(self.actor.parameters()) +
                list(self.critic.parameters()),
                self.config.max_grad_norm
            )
            self.optimizer.step()

        self.clear_buffers()

        return {
            'loss': loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item()
        }


def train():
    """Train Volume Profile PPO with 200-bar lookback"""

    print("\n" + "="*70)
    print("📊 VOLUME PROFILE PPO WITH 200-BAR LOOKBACK")
    print("="*70)

    # Load data
    data_path = DATA_DIR / 'spy_5min_data.parquet'
    df = pd.read_parquet(data_path)

    # Split data
    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    train_data = df.iloc[:train_end].copy()
    val_data = df.iloc[train_end:val_end].copy()
    test_data = df.iloc[val_end:].copy()

    print(f"📊 Loaded {len(df)} 5-minute bars")
    print(f"  Train: {len(train_data)} bars")
    print(f"  Val: {len(val_data)} bars")
    print(f"  Test: {len(test_data)} bars")

    # Initialize
    config = VolumeProfileConfig()
    env = VPEnvironment(train_data, config)
    agent = PPOAgent(config)

    print(f"\n⚡ Configuration:")
    print(f"  Lookback: {config.lookback_bars} bars (200 bars)")
    print(f"  Price levels: {config.num_price_levels}")
    print(f"  State dim: {config.state_dim}")
    print(f"  Hidden dim: {config.hidden_dim}")
    print(f"  Total timesteps: {config.total_timesteps}")

    print("\n📍 Strategy Logic:")
    print("  - Mean Reversion: Near VAL (buy) or VAH (sell)")
    print("  - Continuation: Below VAL (short) or Above VAH (long)")

    # Training loop
    print("\n🎯 Training for {config.total_timesteps} timesteps...")

    total_steps = 0
    episode = 0
    episode_returns = []

    pbar = tqdm(total=config.total_timesteps, desc="Training")

    while total_steps < config.total_timesteps:
        state = env.reset()
        episode_reward = 0
        episode_steps = 0

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
            episode_steps += 1
            total_steps += 1
            pbar.update(1)

            # Update policy
            if len(agent.states) >= config.n_steps:
                metrics = agent.update()
                if metrics:
                    pbar.set_postfix(
                        loss=f"{metrics['loss']:.4f}",
                        ret=f"{(env.balance/config.initial_balance - 1)*100:.2f}%"
                    )

            if done:
                episode_return = (env.balance / config.initial_balance - 1) * 100
                episode_returns.append(episode_return)

                if episode % 10 == 0 and len(episode_returns) >= 10:
                    avg_return = np.mean(episode_returns[-10:])
                    print(f"\n📈 Last 10 episodes:")
                    print(f"  Avg return: {avg_return:.2f}%")
                    print(f"  Avg trades: {info['n_trades'] / max(1, episode % 10 + 1):.1f}")
                    print(f"  Value area trades: {info['value_area_trades']}")
                    print(f"  Breakout trades: {info['breakout_trades']}")

                break

            state = next_state

        episode += 1

    pbar.close()

    # Test on holdout data
    print("\n" + "="*70)
    print("📊 TESTING ON HOLDOUT DATA")
    print("="*70)

    env_test = VPEnvironment(test_data, config)
    state = env_test.reset()

    while True:
        action, _, _ = agent.select_action(state)
        state, reward, done, info = env_test.step(action)

        if done:
            break

    # Calculate metrics
    total_return = (env_test.balance / config.initial_balance - 1) * 100
    n_trades = len(env_test.trades)
    win_rate = sum([1 for r in env_test.returns if r > 0]) / max(1, len(env_test.returns)) * 100

    print(f"\n🎯 Test Results:")
    print(f"  Total Return: {total_return:.2f}%")
    print(f"  Final Balance: ${env_test.balance:.2f}")
    print(f"  Number of Trades: {n_trades}")
    print(f"  Win Rate: {win_rate:.1f}%")

    print(f"\n📊 Trade Breakdown:")
    print(f"  Value Area Trades (Mean Reversion): {info['value_area_trades']}")
    print(f"  Breakout Trades (Continuation): {info['breakout_trades']}")

    # Analyze trade types
    if env_test.trades:
        trade_types = {}
        for trade in env_test.trades:
            trade_type = trade.get('type', 'other')
            trade_types[trade_type] = trade_types.get(trade_type, 0) + 1

        print(f"\n📈 Trade Types:")
        for t_type, count in sorted(trade_types.items(), key=lambda x: x[1], reverse=True):
            print(f"  {t_type}: {count}")

    # Save model
    model_path = MODELS_DIR / f"ppo_vp_200bar_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
    torch.save({
        'network': agent.network.state_dict(),
        'actor': agent.actor.state_dict(),
        'critic': agent.critic.state_dict(),
        'config': config.__dict__
    }, model_path)

    print(f"\n💾 Model saved to {model_path}")

    print("\n" + "="*70)
    print(f"📊 Volume Profile 200-bar completed with {total_return:.2f}% return")
    print("="*70)

    return agent


if __name__ == "__main__":
    agent = train()