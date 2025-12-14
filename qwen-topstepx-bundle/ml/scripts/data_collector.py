#!/usr/bin/env python3
"""
Data Collection Script for Training LSTM and PPO Models
Collects and formats trading data from various sources.
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import yfinance as yf  # For historical data
import random

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "ml" / "data"
TRADING_DB = ROOT / "trading-db"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
TRADING_DB.mkdir(parents=True, exist_ok=True)


class TradingDataCollector:
    """Collects and formats data for ML training."""

    def __init__(self):
        self.decisions_path = TRADING_DB / "decisions.jsonl"
        self.outcomes_path = TRADING_DB / "outcomes.jsonl"
        self.snapshots_path = DATA_DIR / "snapshots.jsonl"
        self.market_data_path = DATA_DIR / "market_data.parquet"

    def collect_market_data(self, symbol: str = "NQ=F", period: str = "3mo",
                          interval: str = "5m") -> pd.DataFrame:
        """
        Collect historical market data from Yahoo Finance or other sources.

        Args:
            symbol: Trading symbol (NQ=F for Nasdaq futures)
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y)
            interval: Data interval (1m, 5m, 15m, 30m, 1h, 1d)
        """
        print(f"Collecting market data for {symbol}...")

        try:
            # Download from Yahoo Finance
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)

            if data.empty:
                print("Warning: No data from Yahoo Finance, generating synthetic data...")
                data = self.generate_synthetic_market_data()
        except Exception as e:
            print(f"Error fetching data: {e}")
            print("Generating synthetic data for demonstration...")
            data = self.generate_synthetic_market_data()

        # Add technical indicators
        data = self.add_technical_indicators(data)

        # Save to parquet
        data.to_parquet(self.market_data_path)
        print(f"Saved market data to {self.market_data_path}")

        return data

    def generate_synthetic_market_data(self, days: int = 90) -> pd.DataFrame:
        """
        Generate synthetic market data for testing when real data isn't available.
        """
        np.random.seed(42)

        # Generate time index
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        time_index = pd.date_range(start=start_time, end=end_time, freq='5min')

        # Generate price data with realistic patterns
        n_points = len(time_index)

        # Base trend
        trend = np.linspace(16000, 16500, n_points)

        # Add daily patterns (market hours volatility)
        daily_pattern = np.array([
            10 * np.sin(2 * np.pi * (i.hour + i.minute/60) / 24)
            for i in time_index
        ])

        # Add noise and volatility
        noise = np.random.normal(0, 20, n_points)

        # Combine
        close_prices = trend + daily_pattern + np.cumsum(noise)

        # Generate OHLCV data
        data = pd.DataFrame(index=time_index)
        data['Open'] = close_prices + np.random.uniform(-5, 5, n_points)
        data['High'] = np.maximum(data['Open'], close_prices) + np.abs(np.random.normal(0, 10, n_points))
        data['Low'] = np.minimum(data['Open'], close_prices) - np.abs(np.random.normal(0, 10, n_points))
        data['Close'] = close_prices
        data['Volume'] = np.random.randint(500, 5000, n_points)

        return data

    def add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators required for ML models."""

        # Simple Moving Averages
        data['SMA_20'] = data['Close'].rolling(window=20, min_periods=1).mean()
        data['SMA_50'] = data['Close'].rolling(window=50, min_periods=1).mean()

        # RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        rs = gain / (loss + 1e-8)
        data['RSI'] = 100 - (100 / (1 + rs))

        # MACD
        exp1 = data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = exp1 - exp2
        data['MACD_signal'] = data['MACD'].ewm(span=9, adjust=False).mean()

        # Bollinger Bands
        rolling_mean = data['Close'].rolling(window=20, min_periods=1).mean()
        rolling_std = data['Close'].rolling(window=20, min_periods=1).std()
        data['BB_upper'] = rolling_mean + (rolling_std * 2)
        data['BB_lower'] = rolling_mean - (rolling_std * 2)
        data['BB_width'] = data['BB_upper'] - data['BB_lower']

        # Volume indicators
        data['Volume_SMA'] = data['Volume'].rolling(window=20, min_periods=1).mean()
        data['Volume_ratio'] = data['Volume'] / (data['Volume_SMA'] + 1e-8)

        # Price features
        data['Price_change'] = data['Close'].pct_change()
        data['High_Low_ratio'] = (data['High'] - data['Low']) / (data['Close'] + 1e-8)

        # Volatility
        data['Volatility'] = data['Price_change'].rolling(window=20, min_periods=1).std()

        # Support/Resistance (simplified POC)
        data['POC'] = data['Close'].rolling(window=100, min_periods=1).median()
        data['Dist_to_POC'] = data['Close'] - data['POC']
        data['Dist_to_POC_ticks'] = data['Dist_to_POC'] / 0.25  # Assuming 0.25 tick size

        # VWAP
        data['VWAP'] = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()

        # Clean NaN values
        data.fillna(method='ffill', inplace=True)
        data.fillna(0, inplace=True)

        return data

    def generate_trading_decisions(self, market_data: pd.DataFrame,
                                  n_trades: int = 1000) -> List[Dict]:
        """
        Generate trading decisions based on market data.
        This simulates a trading strategy for training data.
        """
        decisions = []

        # Simple strategy for generating decisions
        for i in range(100, min(len(market_data), 100 + n_trades)):
            row = market_data.iloc[i]
            prev_row = market_data.iloc[i-1]

            # Decision logic (example strategy)
            decision_id = f"decision_{i}_{datetime.now().timestamp()}"

            # RSI-based signals
            if row['RSI'] < 30:
                signal = 'BUY'
                confidence = (30 - row['RSI']) / 30
            elif row['RSI'] > 70:
                signal = 'SELL'
                confidence = (row['RSI'] - 70) / 30
            else:
                # MACD crossover
                if row['MACD'] > row['MACD_signal'] and prev_row['MACD'] <= prev_row['MACD_signal']:
                    signal = 'BUY'
                    confidence = min(abs(row['MACD'] - row['MACD_signal']) / 10, 1.0)
                elif row['MACD'] < row['MACD_signal'] and prev_row['MACD'] >= prev_row['MACD_signal']:
                    signal = 'SELL'
                    confidence = min(abs(row['MACD'] - row['MACD_signal']) / 10, 1.0)
                else:
                    signal = 'HOLD'
                    confidence = 0.5

            decision = {
                'id': decision_id,
                'timestamp': row.name.isoformat() if hasattr(row.name, 'isoformat') else str(row.name),
                'symbol': 'NQ',
                'signal': signal,
                'confidence': confidence,
                'price': float(row['Close']),
                'rsi': float(row['RSI']),
                'macd': float(row['MACD']),
                'volume': float(row['Volume']),
                'features': {
                    'sma_20': float(row['SMA_20']),
                    'sma_50': float(row['SMA_50']),
                    'bb_upper': float(row['BB_upper']),
                    'bb_lower': float(row['BB_lower']),
                    'vwap': float(row['VWAP']),
                    'volatility': float(row['Volatility']),
                    'dist_to_poc_ticks': float(row['Dist_to_POC_ticks'])
                }
            }

            if signal != 'HOLD' and random.random() < 0.3:  # 30% of signals result in trades
                decisions.append(decision)

        return decisions

    def generate_trading_outcomes(self, decisions: List[Dict],
                                 market_data: pd.DataFrame) -> List[Dict]:
        """
        Generate trading outcomes for decisions.
        Simulates trade execution and results.
        """
        outcomes = []

        for decision in decisions:
            # Find entry time in market data
            entry_time = pd.Timestamp(decision['timestamp'])

            try:
                entry_idx = market_data.index.get_loc(entry_time, method='nearest')
            except:
                continue

            if entry_idx >= len(market_data) - 30:
                continue

            entry_price = decision['price']
            signal = decision['signal']

            # Simulate trade outcome
            # Look ahead 5 and 30 minutes
            price_5m = market_data.iloc[min(entry_idx + 1, len(market_data)-1)]['Close']
            price_30m = market_data.iloc[min(entry_idx + 6, len(market_data)-1)]['Close']

            # Calculate P&L
            if signal == 'BUY':
                pnl_5m = price_5m - entry_price
                pnl_30m = price_30m - entry_price
            else:  # SELL
                pnl_5m = entry_price - price_5m
                pnl_30m = entry_price - price_30m

            # Determine win/loss (simplified)
            win_5m = pnl_5m > 2  # $2 profit target
            win_30m = pnl_30m > 5  # $5 profit target

            outcome = {
                'decisionId': decision['id'],
                'timestamp': decision['timestamp'],
                'executedTime': decision['timestamp'],
                'closedTime': (entry_time + timedelta(minutes=30)).isoformat(),
                'entryPrice': entry_price,
                'exitPrice': float(price_30m),
                'profitLoss': float(pnl_30m),
                'win_5m': win_5m,
                'win_30m': win_30m,
                'signal': signal
            }

            outcomes.append(outcome)

        return outcomes

    def generate_market_snapshots(self, market_data: pd.DataFrame,
                                 decisions: List[Dict]) -> List[Dict]:
        """
        Generate market snapshots at decision times.
        These provide features for ML models.
        """
        snapshots = []

        for decision in decisions:
            timestamp = decision['timestamp']

            snapshot = {
                'symbol': decision['symbol'],
                'timestamp': timestamp,
                'features': {
                    **decision['features'],
                    'price': decision['price'],
                    'rsi': decision['rsi'],
                    'macd': decision['macd'],
                    'volume': decision['volume']
                }
            }

            snapshots.append(snapshot)

        return snapshots

    def save_data(self, decisions: List[Dict], outcomes: List[Dict],
                  snapshots: List[Dict]):
        """Save all data to appropriate files."""

        # Save decisions
        with open(self.decisions_path, 'w') as f:
            for decision in decisions:
                f.write(json.dumps(decision) + '\n')
        print(f"Saved {len(decisions)} decisions to {self.decisions_path}")

        # Save outcomes
        with open(self.outcomes_path, 'w') as f:
            for outcome in outcomes:
                f.write(json.dumps(outcome) + '\n')
        print(f"Saved {len(outcomes)} outcomes to {self.outcomes_path}")

        # Save snapshots
        with open(self.snapshots_path, 'w') as f:
            for snapshot in snapshots:
                f.write(json.dumps(snapshot) + '\n')
        print(f"Saved {len(snapshots)} snapshots to {self.snapshots_path}")

    def prepare_training_data(self):
        """
        Complete pipeline to prepare all training data.
        """
        print("="*60)
        print("PREPARING TRAINING DATA FOR ML MODELS")
        print("="*60)

        # 1. Collect market data
        print("\n1. Collecting market data...")
        market_data = self.collect_market_data()
        print(f"   Collected {len(market_data)} data points")

        # 2. Generate trading decisions
        print("\n2. Generating trading decisions...")
        decisions = self.generate_trading_decisions(market_data)
        print(f"   Generated {len(decisions)} decisions")

        # 3. Generate outcomes
        print("\n3. Generating trading outcomes...")
        outcomes = self.generate_trading_outcomes(decisions, market_data)
        print(f"   Generated {len(outcomes)} outcomes")

        # 4. Generate snapshots
        print("\n4. Generating market snapshots...")
        snapshots = self.generate_market_snapshots(market_data, decisions)
        print(f"   Generated {len(snapshots)} snapshots")

        # 5. Save all data
        print("\n5. Saving data...")
        self.save_data(decisions, outcomes, snapshots)

        print("\n" + "="*60)
        print("DATA PREPARATION COMPLETE!")
        print("="*60)
        print("\nNext steps:")
        print("1. Run: python3 ml/scripts/build_dataset.py")
        print("2. Run: python3 ml/scripts/train_lstm_model.py")
        print("3. Run: python3 ml/scripts/train_ppo_agent.py")

        return {
            'market_data_points': len(market_data),
            'decisions': len(decisions),
            'outcomes': len(outcomes),
            'snapshots': len(snapshots)
        }


def main():
    collector = TradingDataCollector()
    stats = collector.prepare_training_data()

    print("\nData Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()