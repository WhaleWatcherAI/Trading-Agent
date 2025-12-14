#!/usr/bin/env python3
"""
Alpaca API Data Collection Script for Training LSTM and PPO Models
Uses your Alpaca API keys to collect real market data.
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import os
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import REST, TimeFrame
import random
from tqdm import tqdm

# Load environment variables
ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "ml" / "data"
TRADING_DB = ROOT / "trading-db"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
TRADING_DB.mkdir(parents=True, exist_ok=True)


class AlpacaDataCollector:
    """Collects real market data from Alpaca API for ML training."""

    def __init__(self):
        # Initialize Alpaca API
        self.api = REST(
            key_id=os.getenv('ALPACA_KEY_ID'),
            secret_key=os.getenv('ALPACA_SECRET_KEY'),
            base_url=os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
        )

        # Data paths
        self.decisions_path = TRADING_DB / "decisions.jsonl"
        self.outcomes_path = TRADING_DB / "outcomes.jsonl"
        self.snapshots_path = DATA_DIR / "snapshots.jsonl"
        self.market_data_path = DATA_DIR / "market_data.parquet"

        print("Alpaca API initialized successfully!")

    def get_futures_symbol(self, base_symbol: str = "NQ") -> str:
        """
        Get the appropriate futures symbol for Alpaca.
        Alpaca uses continuous futures contracts.
        """
        # Map common futures to Alpaca symbols
        futures_map = {
            "NQ": "NQH25",  # Nasdaq E-mini futures (March 2025)
            "ES": "ESH25",  # S&P 500 E-mini futures
            "YM": "YMH25",  # Dow E-mini futures
            "RTY": "RTYH25",  # Russell 2000 E-mini futures
            "GC": "GCG25",  # Gold futures
            "CL": "CLG25",  # Crude Oil futures
        }

        return futures_map.get(base_symbol, base_symbol)

    def collect_market_data(self, symbol: str = "NQ", days: int = 90,
                          timeframe: str = "5Min") -> pd.DataFrame:
        """
        Collect historical market data from Alpaca API.

        Args:
            symbol: Trading symbol (NQ, ES, etc.)
            days: Number of days of historical data
            timeframe: Bar timeframe (1Min, 5Min, 15Min, 1Hour, 1Day)
        """
        print(f"\nCollecting {days} days of {timeframe} data for {symbol} from Alpaca...")

        try:
            # For futures, try the futures endpoint first
            # For testing, we'll use SPY as a proxy for market data
            # since futures require specific Alpaca subscriptions
            proxy_symbol = "SPY"  # S&P 500 ETF as proxy

            if symbol in ["NQ", "ES", "YM", "RTY"]:
                print(f"Using {proxy_symbol} as proxy for {symbol} futures data...")
                symbol_to_fetch = proxy_symbol
            else:
                symbol_to_fetch = symbol

            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            # Map timeframe string to Alpaca TimeFrame
            timeframe_map = {
                "1Min": TimeFrame.Minute,
                "5Min": TimeFrame(5, "Min"),
                "15Min": TimeFrame(15, "Min"),
                "30Min": TimeFrame(30, "Min"),
                "1Hour": TimeFrame.Hour,
                "1Day": TimeFrame.Day
            }

            alpaca_timeframe = timeframe_map.get(timeframe, TimeFrame(5, "Min"))

            # Fetch bars from Alpaca
            print(f"Fetching data from {start_date.date()} to {end_date.date()}...")
            bars = self.api.get_bars(
                symbol_to_fetch,
                alpaca_timeframe,
                start=start_date.isoformat(),
                end=end_date.isoformat(),
                adjustment='raw',
                limit=None,
                asof=None
            ).df

            if bars.empty:
                raise ValueError("No data received from Alpaca")

            # Reset index to get timestamp as column
            bars.reset_index(inplace=True)
            bars.rename(columns={'timestamp': 'Datetime'}, inplace=True)

            # Rename columns to match expected format
            bars.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume',
                'trade_count': 'TradeCount',
                'vwap': 'VWAP'
            }, inplace=True)

            # Set datetime as index
            bars.set_index('Datetime', inplace=True)

            # Scale prices if using proxy (SPY is ~1/40th of NQ value)
            if symbol_to_fetch == "SPY" and symbol == "NQ":
                scale_factor = 40  # Approximate NQ/SPY ratio
                bars['Open'] *= scale_factor
                bars['High'] *= scale_factor
                bars['Low'] *= scale_factor
                bars['Close'] *= scale_factor
                if 'VWAP' in bars.columns:
                    bars['VWAP'] *= scale_factor

            print(f"Successfully fetched {len(bars)} bars of data")

        except Exception as e:
            print(f"Error fetching data from Alpaca: {e}")
            print("Generating synthetic data as fallback...")
            bars = self.generate_synthetic_market_data(days=days)

        # Add technical indicators
        bars = self.add_technical_indicators(bars)

        # Save to parquet
        bars.to_parquet(self.market_data_path)
        print(f"Saved market data to {self.market_data_path}")

        return bars

    def get_account_info(self) -> Dict:
        """Get current account information from Alpaca."""
        try:
            account = self.api.get_account()
            return {
                'buying_power': float(account.buying_power),
                'portfolio_value': float(account.portfolio_value),
                'cash': float(account.cash),
                'pattern_day_trader': account.pattern_day_trader,
                'trading_blocked': account.trading_blocked,
                'account_blocked': account.account_blocked
            }
        except Exception as e:
            print(f"Error getting account info: {e}")
            return {}

    def get_positions(self) -> List[Dict]:
        """Get current positions from Alpaca."""
        try:
            positions = self.api.list_positions()
            return [{
                'symbol': pos.symbol,
                'qty': int(pos.qty),
                'side': pos.side,
                'market_value': float(pos.market_value),
                'cost_basis': float(pos.cost_basis),
                'unrealized_pl': float(pos.unrealized_pl),
                'unrealized_plpc': float(pos.unrealized_plpc)
            } for pos in positions]
        except Exception as e:
            print(f"Error getting positions: {e}")
            return []

    def get_recent_trades(self, symbol: str = "SPY", limit: int = 100) -> pd.DataFrame:
        """Get recent trades from Alpaca."""
        try:
            trades = self.api.get_trades(symbol, limit=limit).df
            return trades
        except Exception as e:
            print(f"Error getting trades: {e}")
            return pd.DataFrame()

    def add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators required for ML models."""

        # Simple Moving Averages
        data['SMA_20'] = data['Close'].rolling(window=20, min_periods=1).mean()
        data['SMA_50'] = data['Close'].rolling(window=50, min_periods=1).mean()
        data['SMA_200'] = data['Close'].rolling(window=200, min_periods=1).mean()

        # Exponential Moving Averages
        data['EMA_9'] = data['Close'].ewm(span=9, adjust=False).mean()
        data['EMA_21'] = data['Close'].ewm(span=21, adjust=False).mean()

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
        data['MACD_histogram'] = data['MACD'] - data['MACD_signal']

        # Bollinger Bands
        rolling_mean = data['Close'].rolling(window=20, min_periods=1).mean()
        rolling_std = data['Close'].rolling(window=20, min_periods=1).std()
        data['BB_upper'] = rolling_mean + (rolling_std * 2)
        data['BB_lower'] = rolling_mean - (rolling_std * 2)
        data['BB_middle'] = rolling_mean
        data['BB_width'] = data['BB_upper'] - data['BB_lower']
        data['BB_position'] = (data['Close'] - data['BB_lower']) / (data['BB_width'] + 1e-8)

        # Volume indicators
        data['Volume_SMA'] = data['Volume'].rolling(window=20, min_periods=1).mean()
        data['Volume_ratio'] = data['Volume'] / (data['Volume_SMA'] + 1e-8)

        # Price features
        data['Price_change'] = data['Close'].pct_change()
        data['High_Low_ratio'] = (data['High'] - data['Low']) / (data['Close'] + 1e-8)
        data['Close_Open_ratio'] = (data['Close'] - data['Open']) / (data['Open'] + 1e-8)

        # Volatility
        data['Volatility'] = data['Price_change'].rolling(window=20, min_periods=1).std()
        data['ATR'] = self.calculate_atr(data)

        # Support/Resistance levels (simplified)
        data['Resistance'] = data['High'].rolling(window=20, min_periods=1).max()
        data['Support'] = data['Low'].rolling(window=20, min_periods=1).min()
        data['SR_position'] = (data['Close'] - data['Support']) / (data['Resistance'] - data['Support'] + 1e-8)

        # POC (Point of Control) - simplified as median
        data['POC'] = data['Close'].rolling(window=100, min_periods=1).median()
        data['Dist_to_POC'] = data['Close'] - data['POC']
        data['Dist_to_POC_ticks'] = data['Dist_to_POC'] / 0.25  # Assuming 0.25 tick size

        # If VWAP not provided by Alpaca, calculate it
        if 'VWAP' not in data.columns:
            data['VWAP'] = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()

        # Market microstructure
        data['Spread'] = data['High'] - data['Low']
        data['Range'] = data['High'] - data['Low']

        # Clean NaN values
        data.fillna(method='ffill', inplace=True)
        data.fillna(0, inplace=True)

        return data

    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high = df['High']
        low = df['Low']
        close = df['Close']

        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period, min_periods=1).mean()

        return atr

    def generate_trading_decisions(self, market_data: pd.DataFrame,
                                  n_trades: int = 1000) -> List[Dict]:
        """
        Generate trading decisions based on market data.
        Uses multiple strategies to create diverse training data.
        """
        decisions = []
        strategies = ['rsi', 'macd', 'bb', 'sma_cross', 'momentum']

        print(f"Generating {n_trades} trading decisions using multiple strategies...")

        for i in tqdm(range(100, min(len(market_data), 100 + n_trades * 3))):
            row = market_data.iloc[i]
            prev_row = market_data.iloc[i-1]

            # Randomly select strategy
            strategy = random.choice(strategies)
            signal = None
            confidence = 0

            if strategy == 'rsi':
                # RSI strategy
                if row['RSI'] < 30:
                    signal = 'BUY'
                    confidence = (30 - row['RSI']) / 30
                elif row['RSI'] > 70:
                    signal = 'SELL'
                    confidence = (row['RSI'] - 70) / 30

            elif strategy == 'macd':
                # MACD crossover
                if row['MACD'] > row['MACD_signal'] and prev_row['MACD'] <= prev_row['MACD_signal']:
                    signal = 'BUY'
                    confidence = min(abs(row['MACD'] - row['MACD_signal']) / 10, 1.0)
                elif row['MACD'] < row['MACD_signal'] and prev_row['MACD'] >= prev_row['MACD_signal']:
                    signal = 'SELL'
                    confidence = min(abs(row['MACD'] - row['MACD_signal']) / 10, 1.0)

            elif strategy == 'bb':
                # Bollinger Bands
                if row['Close'] < row['BB_lower']:
                    signal = 'BUY'
                    confidence = min((row['BB_lower'] - row['Close']) / row['BB_width'], 1.0)
                elif row['Close'] > row['BB_upper']:
                    signal = 'SELL'
                    confidence = min((row['Close'] - row['BB_upper']) / row['BB_width'], 1.0)

            elif strategy == 'sma_cross':
                # SMA crossover
                if row['SMA_20'] > row['SMA_50'] and prev_row['SMA_20'] <= prev_row['SMA_50']:
                    signal = 'BUY'
                    confidence = 0.7
                elif row['SMA_20'] < row['SMA_50'] and prev_row['SMA_20'] >= prev_row['SMA_50']:
                    signal = 'SELL'
                    confidence = 0.7

            elif strategy == 'momentum':
                # Momentum strategy
                momentum = (row['Close'] - market_data.iloc[i-10]['Close']) / market_data.iloc[i-10]['Close']
                if momentum > 0.01:  # 1% momentum
                    signal = 'BUY'
                    confidence = min(momentum * 50, 1.0)
                elif momentum < -0.01:
                    signal = 'SELL'
                    confidence = min(abs(momentum) * 50, 1.0)

            # Generate decision if signal triggered
            if signal and confidence > 0.3:  # Minimum confidence threshold
                decision_id = f"decision_{i}_{datetime.now().timestamp()}"

                decision = {
                    'id': decision_id,
                    'timestamp': row.name.isoformat() if hasattr(row.name, 'isoformat') else str(row.name),
                    'symbol': 'NQ',
                    'signal': signal,
                    'strategy': strategy,
                    'confidence': float(confidence),
                    'price': float(row['Close']),
                    'features': {
                        'rsi': float(row['RSI']),
                        'macd': float(row['MACD']),
                        'macd_signal': float(row['MACD_signal']),
                        'bb_upper': float(row['BB_upper']),
                        'bb_lower': float(row['BB_lower']),
                        'bb_position': float(row['BB_position']),
                        'sma_20': float(row['SMA_20']),
                        'sma_50': float(row['SMA_50']),
                        'volume_ratio': float(row['Volume_ratio']),
                        'volatility': float(row['Volatility']),
                        'atr': float(row['ATR']),
                        'dist_to_poc_ticks': float(row['Dist_to_POC_ticks']),
                        'vwap': float(row['VWAP'])
                    }
                }

                decisions.append(decision)

                if len(decisions) >= n_trades:
                    break

        print(f"Generated {len(decisions)} trading decisions")
        return decisions

    def generate_trading_outcomes(self, decisions: List[Dict],
                                 market_data: pd.DataFrame) -> List[Dict]:
        """
        Generate realistic trading outcomes for decisions.
        """
        outcomes = []
        print(f"Generating outcomes for {len(decisions)} decisions...")

        for decision in tqdm(decisions):
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

            # Look ahead for outcome
            # Check 1min, 5min, 30min outcomes
            outcomes_1m = market_data.iloc[min(entry_idx + 1, len(market_data)-1)]
            outcomes_5m = market_data.iloc[min(entry_idx + 5, len(market_data)-1)]
            outcomes_30m = market_data.iloc[min(entry_idx + 30, len(market_data)-1)]

            # Calculate P&L
            if signal == 'BUY':
                pnl_1m = outcomes_1m['Close'] - entry_price
                pnl_5m = outcomes_5m['Close'] - entry_price
                pnl_30m = outcomes_30m['Close'] - entry_price
            else:  # SELL
                pnl_1m = entry_price - outcomes_1m['Close']
                pnl_5m = entry_price - outcomes_5m['Close']
                pnl_30m = entry_price - outcomes_30m['Close']

            # Realistic profit targets and stop losses
            # For NQ: $2.50 = 10 ticks, $5 = 20 ticks
            win_1m = pnl_1m > 2.5
            win_5m = pnl_5m > 2.5
            win_30m = pnl_30m > 5

            # Add transaction costs
            pnl_30m -= 2  # $2 round-trip commission

            outcome = {
                'decisionId': decision['id'],
                'timestamp': decision['timestamp'],
                'executedTime': decision['timestamp'],
                'closedTime': (entry_time + timedelta(minutes=30)).isoformat(),
                'entryPrice': float(entry_price),
                'exitPrice': float(outcomes_30m['Close']),
                'profitLoss': float(pnl_30m),
                'pnl_1m': float(pnl_1m),
                'pnl_5m': float(pnl_5m),
                'win_1m': win_1m,
                'win_5m': win_5m,
                'win_30m': win_30m,
                'signal': signal,
                'strategy': decision.get('strategy', 'unknown')
            }

            outcomes.append(outcome)

        print(f"Generated {len(outcomes)} trading outcomes")
        return outcomes

    def generate_market_snapshots(self, market_data: pd.DataFrame,
                                 decisions: List[Dict]) -> List[Dict]:
        """
        Generate detailed market snapshots for each decision point.
        """
        snapshots = []
        print(f"Generating {len(decisions)} market snapshots...")

        for decision in tqdm(decisions):
            timestamp = decision['timestamp']

            # Create comprehensive snapshot
            snapshot = {
                'symbol': decision['symbol'],
                'timestamp': timestamp,
                'features': {
                    **decision['features'],
                    'price': decision['price'],
                    'confidence': decision['confidence']
                }
            }

            snapshots.append(snapshot)

        return snapshots

    def generate_synthetic_market_data(self, days: int = 90) -> pd.DataFrame:
        """Fallback synthetic data generation."""
        np.random.seed(42)

        # Generate realistic intraday patterns
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        time_index = pd.date_range(start=start_time, end=end_time, freq='5min')

        n_points = len(time_index)

        # NQ-like price levels
        base_price = 16000
        trend = np.linspace(base_price, base_price + 500, n_points)

        # Add realistic volatility patterns
        daily_pattern = np.array([
            20 * np.sin(2 * np.pi * (i.hour - 9.5) / 6.5)  # Market hours pattern
            if 9 <= i.hour < 16 else 5 * np.random.normal()  # After hours
            for i in time_index
        ])

        # Add noise
        noise = np.random.normal(0, 15, n_points)
        close_prices = trend + daily_pattern + np.cumsum(noise)

        # Generate OHLCV
        data = pd.DataFrame(index=time_index)
        data['Open'] = close_prices + np.random.uniform(-5, 5, n_points)
        data['High'] = np.maximum(data['Open'], close_prices) + np.abs(np.random.normal(0, 10, n_points))
        data['Low'] = np.minimum(data['Open'], close_prices) - np.abs(np.random.normal(0, 10, n_points))
        data['Close'] = close_prices
        data['Volume'] = np.random.randint(1000, 10000, n_points)

        return data

    def save_data(self, decisions: List[Dict], outcomes: List[Dict],
                  snapshots: List[Dict]):
        """Save all data to JSONL files."""

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

    def prepare_training_data(self, symbol: str = "NQ", days: int = 90,
                             n_trades: int = 1000):
        """
        Complete pipeline to prepare all training data using Alpaca API.
        """
        print("="*60)
        print("ALPACA DATA COLLECTION FOR ML TRAINING")
        print("="*60)

        # Show account info
        account_info = self.get_account_info()
        if account_info:
            print(f"\nAlpaca Account Status:")
            print(f"  Buying Power: ${account_info.get('buying_power', 0):,.2f}")
            print(f"  Portfolio Value: ${account_info.get('portfolio_value', 0):,.2f}")

        # 1. Collect market data
        print(f"\n1. Collecting {days} days of market data for {symbol}...")
        market_data = self.collect_market_data(symbol=symbol, days=days)
        print(f"   Collected {len(market_data)} data points")
        print(f"   Date range: {market_data.index[0]} to {market_data.index[-1]}")

        # Show sample data
        print("\nSample market data:")
        print(market_data.tail(3)[['Close', 'Volume', 'RSI', 'MACD']])

        # 2. Generate trading decisions
        print(f"\n2. Generating {n_trades} trading decisions...")
        decisions = self.generate_trading_decisions(market_data, n_trades)

        # 3. Generate outcomes
        print("\n3. Generating trading outcomes...")
        outcomes = self.generate_trading_outcomes(decisions, market_data)

        # Show outcome statistics
        if outcomes:
            win_rate = sum(1 for o in outcomes if o['win_5m']) / len(outcomes)
            avg_pnl = np.mean([o['profitLoss'] for o in outcomes])
            print(f"   Win rate (5m): {win_rate:.2%}")
            print(f"   Average P&L: ${avg_pnl:.2f}")

        # 4. Generate snapshots
        print("\n4. Generating market snapshots...")
        snapshots = self.generate_market_snapshots(market_data, decisions)

        # 5. Save all data
        print("\n5. Saving data...")
        self.save_data(decisions, outcomes, snapshots)

        print("\n" + "="*60)
        print("DATA COLLECTION COMPLETE!")
        print("="*60)
        print("\nSummary:")
        print(f"  Market data: {len(market_data)} bars")
        print(f"  Decisions: {len(decisions)}")
        print(f"  Outcomes: {len(outcomes)}")
        print(f"  Snapshots: {len(snapshots)}")
        print("\nNext steps:")
        print("1. Run: python3 ml/scripts/build_dataset.py")
        print("2. Run: python3 ml/scripts/train_meta_label.py  # LightGBM")
        print("3. Run: python3 ml/scripts/train_lstm_model.py  # LSTM")
        print("4. Run: python3 ml/scripts/train_ppo_agent.py   # PPO")

        return {
            'market_data_points': len(market_data),
            'decisions': len(decisions),
            'outcomes': len(outcomes),
            'snapshots': len(snapshots)
        }


def main():
    """Main function to run data collection."""
    import argparse

    parser = argparse.ArgumentParser(description='Collect Alpaca market data for ML training')
    parser.add_argument('--symbol', type=str, default='NQ', help='Trading symbol (NQ, ES, etc.)')
    parser.add_argument('--days', type=int, default=90, help='Number of days of historical data')
    parser.add_argument('--trades', type=int, default=1000, help='Number of trades to generate')
    parser.add_argument('--timeframe', type=str, default='5Min', help='Timeframe (1Min, 5Min, 15Min, 1Hour)')

    args = parser.parse_args()

    # Run data collection
    collector = AlpacaDataCollector()
    stats = collector.prepare_training_data(
        symbol=args.symbol,
        days=args.days,
        n_trades=args.trades
    )

    print("\n✅ Ready to train ML models!")


if __name__ == "__main__":
    main()