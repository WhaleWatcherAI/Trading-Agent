#!/usr/bin/env python3
"""
DIAGNOSTIC version of Volume Profile PPO with extensive logging
"""

import json
import logging
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
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout,
    force=True
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "ml" / "data"
MODELS_DIR = ROOT / "ml" / "models"

class VolumeProfileConfig:
    """Configuration with diagnostic output"""
    # Key VP parameters
    lookback_bars = 50  # REDUCED for debugging
    num_price_levels = 20  # REDUCED for debugging
    value_area_pct = 0.70
    max_value_area_iterations = 50  # Lower limit for testing

    # PPO parameters
    state_dim = 15  # Simplified for debugging
    action_dim = 3  # Buy, Sell, Hold
    hidden_dim = 128  # Smaller for debugging

    learning_rate = 3e-4
    gamma = 0.99
    lambda_gae = 0.95
    clip_epsilon = 0.2
    value_loss_coef = 0.5
    entropy_coef = 0.01
    max_grad_norm = 1.0

    # Training
    batch_size = 32  # Smaller batch
    n_epochs = 3
    n_steps = 128  # Smaller buffer
    total_timesteps = 5000  # Much shorter for testing

    # Environment
    initial_balance = 10000
    transaction_cost = 0.0
    max_position_size = 1.0
    atr_multiplier = 1.5

class SimplifiedVPEnv:
    """Simplified Volume Profile environment with extensive logging"""

    def __init__(self, data, config):
        self.data = data
        self.config = config
        self.reset()
        logger.info(f"Initialized environment with {len(data)} bars")

    def reset(self):
        """Reset with logging"""
        self.current_idx = self.config.lookback_bars + 10
        self.balance = self.config.initial_balance
        self.position = 0
        self.entry_price = 0
        self.trades = []
        self.returns = []

        logger.info(f"Environment reset at index {self.current_idx}")
        return self._get_state()

    def _calculate_volume_profile_simple(self, idx):
        """Simplified VP calculation with logging"""
        logger.debug(f"Calculating VP at idx {idx}")

        try:
            # Get lookback data
            start_idx = max(0, idx - self.config.lookback_bars)
            lookback_data = self.data.iloc[start_idx:idx]

            if len(lookback_data) < 10:
                logger.warning(f"Not enough data at idx {idx}")
                return 0.5, 0.6, 0.4  # Default values

            # Simple calculations
            high = lookback_data['high'].max()
            low = lookback_data['low'].min()
            current_price = self.data.iloc[idx]['close']

            # Normalize
            if high > low:
                poc = (current_price - low) / (high - low)
            else:
                poc = 0.5

            vah = min(poc + 0.1, 1.0)
            val = max(poc - 0.1, 0.0)

            return poc, vah, val

        except Exception as e:
            logger.error(f"Error in VP calculation at idx {idx}: {e}")
            return 0.5, 0.6, 0.4

    def _get_state(self):
        """Get state with logging"""
        idx = min(self.current_idx, len(self.data) - 1)
        logger.debug(f"Getting state at idx {idx}")

        try:
            row = self.data.iloc[idx]

            # Calculate simple VP
            poc, vah, val = self._calculate_volume_profile_simple(idx)

            # Basic features
            price = row['close']
            returns_5 = (price / self.data.iloc[max(0, idx-5)]['close'] - 1) if idx > 5 else 0
            volume_ratio = row['volume'] / self.data['volume'].iloc[max(0, idx-20):idx].mean() if idx > 20 else 1

            features = [
                poc, vah, val,
                (price - row['low']) / (row['high'] - row['low'] + 1e-8),
                returns_5,
                volume_ratio,
                float(self.position),
                (price - self.entry_price) / price if self.position != 0 else 0,
                idx / len(self.data),
                len(self.trades) / 100,
                0, 0, 0, 0, 0  # Padding
            ]

            return np.array(features[:self.config.state_dim], dtype=np.float32)

        except Exception as e:
            logger.error(f"Error getting state at idx {idx}: {e}")
            return np.zeros(self.config.state_dim, dtype=np.float32)

    def step(self, action):
        """Step with extensive logging"""
        idx = min(self.current_idx, len(self.data) - 1)

        # Log every 100 steps or at critical step
        if self.current_idx % 100 == 0 or self.current_idx == 1683:
            logger.info(f"Step {self.current_idx}: action={action}, position={self.position}")

        try:
            row = self.data.iloc[idx]
            price = row['close']
            reward = 0

            # Execute action
            if action == 0:  # Buy
                if self.position <= 0:
                    if self.position < 0:  # Close short
                        returns = (self.entry_price - price) / self.entry_price
                        reward = returns * 100
                        self.returns.append(returns)
                    self.position = 1
                    self.entry_price = price
                    self.trades.append({'idx': idx, 'action': 'buy', 'price': price})
                    logger.debug(f"Buy at {price}")

            elif action == 1:  # Sell
                if self.position >= 0:
                    if self.position > 0:  # Close long
                        returns = (price - self.entry_price) / self.entry_price
                        reward = returns * 100
                        self.returns.append(returns)
                    self.position = -1
                    self.entry_price = price
                    self.trades.append({'idx': idx, 'action': 'sell', 'price': price})
                    logger.debug(f"Sell at {price}")

            # Move to next
            self.current_idx += 1
            done = self.current_idx >= len(self.data) - 1

            if done:
                logger.info(f"Episode done. Trades: {len(self.trades)}")

            new_state = self._get_state()

            info = {
                'idx': self.current_idx,
                'balance': self.balance,
                'n_trades': len(self.trades)
            }

            return new_state, reward, done, info

        except Exception as e:
            logger.error(f"Error in step at idx {idx}: {e}")
            # Return safe values
            return self._get_state(), 0, True, {'error': str(e)}

class DiagnosticPPOAgent:
    """PPO agent with diagnostic output"""

    def __init__(self, config):
        self.config = config

        # Simple network
        self.network = nn.Sequential(
            nn.Linear(config.state_dim, config.hidden_dim),
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
        logger.info("Agent initialized")

    def clear_buffers(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def select_action(self, state):
        """Select action with logging"""
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
        """Update with logging"""
        if len(self.states) < self.config.batch_size:
            return {}

        logger.info(f"Updating with {len(self.states)} samples")

        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states))
        actions = torch.LongTensor(self.actions)
        rewards = torch.FloatTensor(self.rewards)

        # Simple update
        for _ in range(self.config.n_epochs):
            features = self.network(states)
            action_logits = self.actor(features)
            values = self.critic(features)

            dist = Categorical(logits=action_logits)
            log_probs = dist.log_prob(actions)

            # Simple losses
            advantages = rewards - values.detach().squeeze()
            policy_loss = -(log_probs * advantages).mean()
            value_loss = F.mse_loss(values.squeeze(), rewards)
            entropy = dist.entropy().mean()

            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

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
        return {'loss': loss.item()}

def train_diagnostic():
    """Diagnostic training with extensive logging"""

    print("\n" + "="*70)
    print("🔍 DIAGNOSTIC VOLUME PROFILE TRAINING")
    print("="*70)

    # Load data
    data_path = DATA_DIR / 'spy_5min_data.parquet'
    df = pd.read_parquet(data_path)

    # Use smaller subset for debugging
    df = df.iloc[:2000].copy()
    print(f"📊 Using {len(df)} bars for diagnostic")

    # Initialize
    config = VolumeProfileConfig()
    env = SimplifiedVPEnv(df, config)
    agent = DiagnosticPPOAgent(config)

    print(f"\n⚡ Diagnostic Configuration:")
    print(f"  Total steps: {config.total_timesteps}")
    print(f"  Lookback: {config.lookback_bars} bars")
    print(f"  Price levels: {config.num_price_levels}")

    # Training loop with detailed logging
    total_steps = 0
    episode = 0

    pbar = tqdm(total=config.total_timesteps, desc="Diagnostic")

    while total_steps < config.total_timesteps:
        state = env.reset()
        episode_reward = 0
        episode_steps = 0

        logger.info(f"Starting episode {episode}")

        while True:
            # Critical logging at problematic step
            if total_steps == 1683 or total_steps % 500 == 0:
                logger.info(f"=== STEP {total_steps} ===")
                logger.info(f"  Env idx: {env.current_idx}")
                logger.info(f"  State shape: {state.shape}")
                logger.info(f"  State values: {state[:5]}")  # First 5 values

            try:
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

                # Update periodically
                if len(agent.states) >= config.n_steps:
                    metrics = agent.update()
                    if metrics:
                        pbar.set_postfix(loss=f"{metrics.get('loss', 0):.4f}")

                if done:
                    logger.info(f"Episode {episode} done: reward={episode_reward:.2f}, steps={episode_steps}")
                    break

                state = next_state

            except Exception as e:
                logger.error(f"Error at step {total_steps}: {e}")
                logger.error(f"Breaking from episode")
                break

        episode += 1

    pbar.close()

    print("\n" + "="*70)
    print("✅ DIAGNOSTIC COMPLETE")
    print("="*70)
    print(f"  Total episodes: {episode}")
    print(f"  Total steps: {total_steps}")

    return agent

if __name__ == "__main__":
    agent = train_diagnostic()