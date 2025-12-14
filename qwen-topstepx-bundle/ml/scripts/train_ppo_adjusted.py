#!/usr/bin/env python3
"""
PPO adjusted for 5-minute bars and 100-minute predictions.
Better reward shaping, exploration, and thresholds.
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
    """Configuration adjusted for 5-min bars."""
    # State and action space
    state_dim = 20
    action_dim = 5  # Hold, Buy, Sell, Short, Cover
    hidden_dim = 256

    # Environment - adjusted for 5-min timeframe
    initial_balance = 10000
    max_position = 1.0
    transaction_cost = 0.0  # Start with no costs

    # Thresholds adjusted for 5-min bars (100-min predictions)
    profit_threshold = 0.003  # 0.3% for 100-min is reasonable
    stop_loss = -0.005  # -0.5% stop loss

    # PPO hyperparameters - adjusted for better exploration
    learning_rate = 3e-5  # Lower for stability
    gamma = 0.995  # Higher for longer timeframe
    lambda_gae = 0.98
    clip_epsilon = 0.2
    value_loss_coef = 0.5
    entropy_coef = 0.05  # Higher for more exploration
    max_grad_norm = 0.5

    # Training
    batch_size = 256  # Larger batches for stability
    n_epochs = 10  # More epochs per update
    n_steps = 2048
    total_timesteps = 200000  # More training

class ImprovedActorCritic(nn.Module):
    """Actor-Critic with better initialization."""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(ImprovedActorCritic, self).__init__()

        # Shared network
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # Actor head with better initialization
        self.actor = nn.Linear(hidden_dim, action_dim)
        # Initialize to favor HOLD initially
        nn.init.xavier_uniform_(self.actor.weight, gain=0.01)
        self.actor.bias.data[0] = 1.0  # Bias toward HOLD

        # Critic head
        self.critic = nn.Linear(hidden_dim, 1)
        nn.init.xavier_uniform_(self.critic.weight)

    def forward(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state)

        if state.dim() == 1:
            state = state.unsqueeze(0)

        features = self.shared(state)
        action_logits = self.actor(features)
        value = self.critic(features)

        return action_logits.squeeze(0), value.squeeze()

class TradingEnv5MinAdjusted:
    """Trading environment adjusted for 5-min bars."""

    def __init__(self, data, config, lookback=20):
        self.data = data
        self.config = config
        self.lookback = lookback
        self.reset()

    def reset(self):
        """Reset to initial state."""
        self.current_idx = self.lookback + 50
        self.balance = self.config.initial_balance
        self.position = 0.0
        self.entry_price = 0
        self.trades = []
        self.winning_trades = 0
        self.losing_trades = 0
        self.consecutive_losses = 0

        return self._get_state()

    def _get_state(self):
        """Enhanced state with better features for 5-min bars."""
        if self.current_idx >= len(self.data):
            self.current_idx = len(self.data) - 1

        recent_data = self.data.iloc[self.current_idx - self.lookback:self.current_idx]
        current_bar = self.data.iloc[self.current_idx]

        # Calculate returns over different periods (5min, 25min, 100min)
        returns_5m = current_bar.get('returns', 0)
        returns_25m = (current_bar['close'] - self.data.iloc[max(0, self.current_idx-5)]['close']) / self.data.iloc[max(0, self.current_idx-5)]['close']
        returns_100m = (current_bar['close'] - self.data.iloc[max(0, self.current_idx-20)]['close']) / self.data.iloc[max(0, self.current_idx-20)]['close']

        # Market regime detection
        recent_volatility = recent_data['volatility'].mean() if 'volatility' in recent_data else 0.01
        volatility_percentile = recent_data['volatility'].rank(pct=True).iloc[-1] if 'volatility' in recent_data else 0.5

        # Trend strength
        if 'sma_20' in current_bar and 'sma_50' in current_bar:
            trend = (current_bar['sma_20'] - current_bar['sma_50']) / current_bar['close']
        else:
            trend = 0

        # Position-related features
        if self.position != 0:
            time_in_position = min((self.current_idx - self.entry_idx) / 20, 1.0) if hasattr(self, 'entry_idx') else 0
            unrealized_pnl = self._get_unrealized_pnl() / self.config.initial_balance
        else:
            time_in_position = 0
            unrealized_pnl = 0

        features = [
            # Price momentum (multi-timeframe)
            returns_5m * 100,
            returns_25m * 100,
            returns_100m * 100,

            # Technical indicators
            (current_bar.get('rsi', 50) - 50) / 50,  # Normalized RSI
            current_bar.get('macd', 0) / current_bar['close'] * 100,
            current_bar.get('bb_position', 0.5) - 0.5,  # Centered BB position

            # Volume
            np.log(current_bar.get('volume_ratio', 1) + 1),

            # Volatility
            recent_volatility * 100,
            volatility_percentile - 0.5,

            # Trend
            trend * 100,

            # Position info
            self.position,
            unrealized_pnl * 100,
            time_in_position,

            # Portfolio state
            (self.balance - self.config.initial_balance) / self.config.initial_balance,
            len(self.trades) / 100,  # Trade frequency

            # Risk metrics
            self.consecutive_losses / 5,  # Normalized consecutive losses
            self.winning_trades / max(1, self.winning_trades + self.losing_trades),  # Win rate

            # Market microstructure
            (current_bar['high'] - current_bar['low']) / current_bar['close'],  # Bar range
            (current_bar['close'] - current_bar['open']) / current_bar['open'],  # Bar return

            # Time
            self.current_idx / len(self.data)
        ]

        # Pad or truncate to state_dim
        if len(features) < self.config.state_dim:
            features.extend([0] * (self.config.state_dim - len(features)))
        else:
            features = features[:self.config.state_dim]

        return np.array(features, dtype=np.float32)

    def _get_unrealized_pnl(self):
        """Calculate unrealized P&L."""
        if self.position == 0:
            return 0

        current_price = self.data.iloc[self.current_idx]['close']

        if self.position > 0:  # Long
            return (current_price - self.entry_price) / self.entry_price * self.position
        else:  # Short
            return (self.entry_price - current_price) / self.entry_price * abs(self.position)

    def step(self, action):
        """Execute action with better reward shaping for 5-min bars."""
        current_price = self.data.iloc[self.current_idx]['close']
        old_portfolio_value = self.balance + self._get_unrealized_pnl() * self.balance

        reward = 0
        action_taken = False

        # Get next price for immediate feedback (5 minutes ahead)
        if self.current_idx < len(self.data) - 1:
            next_price = self.data.iloc[self.current_idx + 1]['close']
            immediate_return = (next_price - current_price) / current_price
        else:
            immediate_return = 0

        # Execute actions
        if action == 0:  # HOLD
            if self.position == 0:
                # Reward for avoiding bad trades
                if abs(immediate_return) < 0.001:  # Less than 0.1% move
                    reward += 0.01
            else:
                # Check if holding position is good
                unrealized = self._get_unrealized_pnl()
                if unrealized > 0:
                    reward += unrealized * 2  # Reward holding winners
                elif unrealized < self.config.stop_loss:
                    reward -= 0.1  # Penalty for not stopping loss

        elif action == 1:  # BUY
            if self.position <= 0:
                if self.position < 0:  # Cover short
                    pnl = (self.entry_price - current_price) / self.entry_price * abs(self.position)
                    self.balance *= (1 + pnl)
                    reward += pnl * 50  # Scaled reward
                    if pnl > 0:
                        self.winning_trades += 1
                        self.consecutive_losses = 0
                    else:
                        self.losing_trades += 1
                        self.consecutive_losses += 1

                # Enter long
                self.position = 0.3  # 30% position for 5-min bars
                self.entry_price = current_price
                self.entry_idx = self.current_idx
                action_taken = True

                # Immediate feedback
                if immediate_return > 0:
                    reward += 0.05  # Good direction

        elif action == 2:  # SELL (close long)
            if self.position > 0:
                pnl = (current_price - self.entry_price) / self.entry_price * self.position
                self.balance *= (1 + pnl)
                reward += pnl * 50  # Scaled reward

                if pnl > 0:
                    self.winning_trades += 1
                    self.consecutive_losses = 0
                    reward += 0.1  # Bonus for profitable trade
                else:
                    self.losing_trades += 1
                    self.consecutive_losses += 1

                self.position = 0
                action_taken = True

        elif action == 3:  # SHORT
            if self.position >= 0:
                if self.position > 0:  # Close long
                    pnl = (current_price - self.entry_price) / self.entry_price * self.position
                    self.balance *= (1 + pnl)
                    reward += pnl * 50
                    if pnl > 0:
                        self.winning_trades += 1
                        self.consecutive_losses = 0
                    else:
                        self.losing_trades += 1
                        self.consecutive_losses += 1

                # Enter short
                self.position = -0.3  # 30% short position
                self.entry_price = current_price
                self.entry_idx = self.current_idx
                action_taken = True

                # Immediate feedback
                if immediate_return < 0:
                    reward += 0.05  # Good direction

        elif action == 4:  # COVER (close short)
            if self.position < 0:
                pnl = (self.entry_price - current_price) / self.entry_price * abs(self.position)
                self.balance *= (1 + pnl)
                reward += pnl * 50

                if pnl > 0:
                    self.winning_trades += 1
                    self.consecutive_losses = 0
                    reward += 0.1  # Bonus for profitable trade
                else:
                    self.losing_trades += 1
                    self.consecutive_losses += 1

                self.position = 0
                action_taken = True

        # Record trade
        if action_taken:
            self.trades.append({
                'bar': self.current_idx,
                'action': action,
                'price': current_price
            })

            # Penalty for overtrading (adjusted for 5-min bars)
            if len(self.trades) > self.current_idx / 20:  # More than 1 trade per 100 minutes
                reward -= 0.02

        # Move to next bar
        self.current_idx += 1

        # Portfolio value change reward
        new_portfolio_value = self.balance + self._get_unrealized_pnl() * self.balance
        portfolio_return = (new_portfolio_value - old_portfolio_value) / self.config.initial_balance
        reward += portfolio_return * 10

        # Risk management rewards
        if self.consecutive_losses > 3:
            reward -= 0.05  # Penalty for consecutive losses

        if new_portfolio_value < self.config.initial_balance * 0.9:
            reward -= 0.1  # Penalty for large drawdown

        # Check if done
        done = (self.current_idx >= len(self.data) - 1) or (self.balance < 1000)

        # Final reward
        if done:
            final_return = (new_portfolio_value - self.config.initial_balance) / self.config.initial_balance
            if final_return > 0:
                reward += final_return * 100  # Big reward for profit
            else:
                reward += final_return * 50  # Less penalty for loss

            # Win rate bonus
            if self.winning_trades + self.losing_trades > 0:
                win_rate = self.winning_trades / (self.winning_trades + self.losing_trades)
                if win_rate > 0.4:  # Good win rate for 100-min predictions
                    reward += win_rate * 10

        # Get new state
        new_state = self._get_state()

        info = {
            'balance': self.balance,
            'position': self.position,
            'portfolio_value': new_portfolio_value,
            'total_trades': len(self.trades),
            'win_rate': self.winning_trades / max(1, self.winning_trades + self.losing_trades),
            'consecutive_losses': self.consecutive_losses
        }

        return new_state, reward, done, info

class PPOAgent:
    """PPO agent with better exploration."""

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

        # Buffers
        self.clear_buffers()

        # Exploration parameters
        self.exploration_rate = 1.0
        self.exploration_decay = 0.995
        self.min_exploration = 0.1

    def clear_buffers(self):
        """Clear experience buffers."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def select_action(self, state, deterministic=False):
        """Select action with temperature-based exploration."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state)
            action_logits, value = self.network(state_tensor)

            if deterministic:
                action = torch.argmax(action_logits).item()
                return action, 0, value.item()
            else:
                # Add temperature for exploration
                temperature = max(self.min_exploration, self.exploration_rate)
                action_logits = action_logits / temperature

                # Add small noise for exploration
                if np.random.random() < 0.1:  # 10% random actions
                    action = np.random.randint(0, self.config.action_dim)
                    dist = Categorical(logits=action_logits)
                    log_prob = dist.log_prob(torch.tensor(action))
                else:
                    dist = Categorical(logits=action_logits)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)

                return action.item() if torch.is_tensor(action) else action, log_prob.item(), value.item()

    def update(self):
        """PPO update with gradient clipping and better normalization."""
        if len(self.states) < self.config.batch_size:
            return {}

        # Decay exploration
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

        # PPO epochs
        total_loss = 0
        for _ in range(self.config.n_epochs):
            # Shuffle data
            perm = torch.randperm(len(states))

            for start in range(0, len(states), self.config.batch_size):
                end = start + self.config.batch_size
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

                # Calculate ratio
                ratio = torch.exp(log_probs - batch_old_log_probs)

                # Clipped objective
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

        # Clear buffers
        self.clear_buffers()

        return {
            'loss': total_loss / (self.config.n_epochs * (len(states) // self.config.batch_size)),
            'exploration_rate': self.exploration_rate
        }

def train_ppo_adjusted():
    """Train PPO adjusted for 5-minute timeframe."""

    print("\n" + "="*70)
    print("🚀 TRAINING PPO ADJUSTED FOR 5-MIN BARS & 100-MIN PREDICTIONS")
    print("="*70)

    # Load data
    data_path = DATA_DIR / 'spy_5min_data.parquet'
    if not data_path.exists():
        print("❌ 5-minute data not found!")
        return None

    df = pd.read_parquet(data_path)
    print(f"📊 Loaded {len(df)} 5-minute bars")

    # Split data
    train_size = int(len(df) * 0.8)
    train_data = df.iloc[:train_size]
    test_data = df.iloc[train_size:]

    print(f"  Training: {len(train_data)} bars")
    print(f"  Testing: {len(test_data)} bars")

    # Initialize
    config = PPOConfig()
    env = TradingEnv5MinAdjusted(train_data, config)
    agent = PPOAgent(config)

    print(f"\n📈 Configuration:")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Entropy coefficient: {config.entropy_coef}")
    print(f"  Profit threshold: {config.profit_threshold:.1%}")
    print(f"  Position size: 30%")
    print(f"  Batch size: {config.batch_size}")

    # Training metrics
    episode_rewards = []
    episode_returns = []
    episode_trades = []
    episode_win_rates = []

    # Training loop
    pbar = tqdm(total=config.total_timesteps, desc="Training PPO")
    total_steps = 0

    best_return = -float('inf')

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
                metrics = agent.update()

            if done:
                portfolio_return = (info['portfolio_value'] - config.initial_balance) / config.initial_balance
                episode_returns.append(portfolio_return)
                episode_trades.append(info['total_trades'])
                episode_win_rates.append(info.get('win_rate', 0))

                # Save best model
                if portfolio_return > best_return:
                    best_return = portfolio_return
                    torch.save(agent.network.state_dict(), MODELS_DIR / 'ppo_adjusted_best.pth')

                break

            state = next_state

        episode_rewards.append(episode_reward)

        # Update progress
        if len(episode_returns) > 0 and len(episode_returns) % 10 == 0:
            avg_return = np.mean(episode_returns[-10:])
            avg_trades = np.mean(episode_trades[-10:])
            avg_win_rate = np.mean(episode_win_rates[-10:])
            pbar.set_description(
                f"Return: {avg_return:.2%} | Trades: {avg_trades:.0f} | WR: {avg_win_rate:.1%}"
            )

    pbar.close()

    # Load best model for testing
    agent.network.load_state_dict(torch.load(MODELS_DIR / 'ppo_adjusted_best.pth'))

    # Test
    print("\n" + "-"*50)
    print("📊 TESTING ON HOLDOUT DATA")
    print("-"*50)

    test_env = TradingEnv5MinAdjusted(test_data, config)
    state = test_env.reset()
    action_counts = {i: 0 for i in range(5)}

    for _ in range(len(test_data) - test_env.lookback - 51):
        action, _, _ = agent.select_action(state, deterministic=True)
        state, reward, done, info = test_env.step(action)
        action_counts[action] += 1

        if done:
            break

    test_return = (info['portfolio_value'] - config.initial_balance) / config.initial_balance

    # Save final model
    torch.save(agent.network.state_dict(), MODELS_DIR / 'ppo_adjusted.pth')

    # Results
    action_map = {0: 'Hold', 1: 'Buy', 2: 'Sell', 3: 'Short', 4: 'Cover'}
    total_actions = sum(action_counts.values())

    print(f"\n📊 Test Results:")
    print(f"  Portfolio Return: {test_return:.2%}")
    print(f"  Total Trades: {info['total_trades']}")
    print(f"  Win Rate: {info.get('win_rate', 0):.1%}")
    print(f"  Final Balance: ${info['portfolio_value']:.2f}")

    print(f"\n📈 Action Distribution:")
    for action_id, count in action_counts.items():
        pct = count / total_actions * 100 if total_actions > 0 else 0
        print(f"  {action_map[action_id]:6s}: {pct:.1f}%")

    # Save metrics
    metrics = {
        'model_type': 'PPO_Adjusted_5min',
        'timeframe': '5min bars, 100min predictions',
        'training': {
            'best_train_return': float(best_return),
            'avg_final_return': float(np.mean(episode_returns[-100:])) if episode_returns else 0,
            'avg_final_trades': float(np.mean(episode_trades[-100:])) if episode_trades else 0,
            'avg_final_win_rate': float(np.mean(episode_win_rates[-100:])) if episode_win_rates else 0
        },
        'test': {
            'return': float(test_return),
            'trades': info['total_trades'],
            'win_rate': float(info.get('win_rate', 0)),
            'final_balance': float(info['portfolio_value']),
            'action_distribution': {action_map[k]: v/total_actions for k, v in action_counts.items() if total_actions > 0}
        },
        'config': {
            'profit_threshold': config.profit_threshold,
            'stop_loss': config.stop_loss,
            'learning_rate': config.learning_rate,
            'entropy_coef': config.entropy_coef
        },
        'trained_at': datetime.utcnow().isoformat()
    }

    with open(MODELS_DIR / 'ppo_adjusted_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "="*70)
    print("✅ PPO TRAINING COMPLETE!")
    print("="*70)

    if test_return > 0:
        print(f"🎉 SUCCESS! Achieved {test_return:.2%} return on test data!")
    else:
        print(f"📊 Model achieved {test_return:.2%} - may need further tuning")

    return agent, metrics

if __name__ == "__main__":
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    agent, metrics = train_ppo_adjusted()