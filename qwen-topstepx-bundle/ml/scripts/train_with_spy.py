#!/usr/bin/env python3
"""
Train models with real SPY data from Alpaca and calculate actual accuracy.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import TimeFrame
from dotenv import load_dotenv
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Load environment variables
ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "ml" / "data"
TRADING_DB = ROOT / "trading-db"
MODELS_DIR = ROOT / "ml" / "models"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
TRADING_DB.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


class SPYTrainingPipeline:
    """Complete pipeline for training with real SPY data."""

    def __init__(self):
        # Initialize Alpaca API
        self.api = tradeapi.REST(
            key_id=os.getenv('ALPACA_KEY_ID'),
            secret_key=os.getenv('ALPACA_SECRET_KEY'),
            base_url='https://paper-api.alpaca.markets',
            api_version='v2'
        )

        self.symbol = 'SPY'
        self.results = {}

    def fetch_real_spy_data(self, days=180):
        """Fetch real SPY data from Alpaca."""
        print(f"\n📊 Fetching real {self.symbol} data from Alpaca...")

        end = datetime.now()
        start = end - timedelta(days=days)

        try:
            # Fetch 5-minute bars with IEX feed (free)
            bars = self.api.get_bars(
                self.symbol,
                TimeFrame.Minute,  # Use pre-defined TimeFrame
                start=start.strftime('%Y-%m-%d'),  # Use date only format
                end=end.strftime('%Y-%m-%d'),
                adjustment='raw',
                feed='iex'  # Use IEX feed which is free
            ).df

            if bars.empty:
                raise ValueError("No data received")

            print(f"✅ Fetched {len(bars)} real market bars")
            print(f"Date range: {bars.index[0]} to {bars.index[-1]}")

            # Add technical indicators
            bars = self.calculate_indicators(bars)

            # Save
            bars.to_parquet(DATA_DIR / 'spy_real_data.parquet')

            return bars

        except Exception as e:
            print(f"Error fetching SPY data: {e}")
            # Try with 1-day bars as fallback
            try:
                bars = self.api.get_bars(
                    self.symbol,
                    TimeFrame.Day,
                    start=start.strftime('%Y-%m-%d'),
                    end=end.strftime('%Y-%m-%d'),
                    feed='iex'  # Use IEX feed
                ).df

                if not bars.empty:
                    print(f"✅ Using daily bars: {len(bars)} bars")
                    bars = self.calculate_indicators(bars)
                    bars.to_parquet(DATA_DIR / 'spy_real_data.parquet')
                    return bars

            except Exception as e2:
                print(f"Fallback also failed: {e2}")
                return None

    def calculate_indicators(self, df):
        """Calculate all technical indicators."""

        # Price features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

        # Moving averages
        df['sma_5'] = df['close'].rolling(5).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()

        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # Bollinger Bands
        bb_sma = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = bb_sma + (bb_std * 2)
        df['bb_lower'] = bb_sma - (bb_std * 2)
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_width'] + 1e-8)

        # Volume features
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']

        # Volatility
        df['volatility'] = df['returns'].rolling(20).std()

        # ATR
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = true_range.rolling(14).mean()

        # Support/Resistance
        df['resistance'] = df['high'].rolling(20).max()
        df['support'] = df['low'].rolling(20).min()

        # Clean up
        df.fillna(method='ffill', inplace=True)
        df.fillna(0, inplace=True)

        return df

    def generate_trading_signals(self, df):
        """Generate realistic trading signals."""
        print("\n🎯 Generating trading signals...")

        signals = []

        for i in range(100, len(df) - 30):
            row = df.iloc[i]

            # Multiple strategy signals
            signal = None
            confidence = 0

            # Strategy 1: RSI Oversold/Overbought
            if row['rsi'] < 30:
                signal = 'BUY'
                confidence = (30 - row['rsi']) / 30
            elif row['rsi'] > 70:
                signal = 'SELL'
                confidence = (row['rsi'] - 70) / 30

            # Strategy 2: MACD Cross
            elif i > 0:
                prev = df.iloc[i-1]
                if row['macd'] > row['macd_signal'] and prev['macd'] <= prev['macd_signal']:
                    signal = 'BUY'
                    confidence = 0.65
                elif row['macd'] < row['macd_signal'] and prev['macd'] >= prev['macd_signal']:
                    signal = 'SELL'
                    confidence = 0.65

            # Strategy 3: Bollinger Band Bounce
            elif row['close'] < row['bb_lower']:
                signal = 'BUY'
                confidence = 0.7
            elif row['close'] > row['bb_upper']:
                signal = 'SELL'
                confidence = 0.7

            if signal and confidence > 0.4:
                # Calculate actual outcome
                future_5m = df.iloc[min(i+1, len(df)-1)]['close']
                future_30m = df.iloc[min(i+6, len(df)-1)]['close']

                if signal == 'BUY':
                    return_5m = (future_5m - row['close']) / row['close']
                    return_30m = (future_30m - row['close']) / row['close']
                else:  # SELL
                    return_5m = (row['close'] - future_5m) / row['close']
                    return_30m = (row['close'] - future_30m) / row['close']

                signals.append({
                    'timestamp': df.index[i],
                    'signal': signal,
                    'confidence': confidence,
                    'price': row['close'],
                    'return_5m': return_5m,
                    'return_30m': return_30m,
                    'win_5m': return_5m > 0.001,  # 0.1% profit
                    'win_30m': return_30m > 0.002,  # 0.2% profit
                    'features': {
                        'rsi': row['rsi'],
                        'macd': row['macd'],
                        'bb_position': row['bb_position'],
                        'volume_ratio': row['volume_ratio'],
                        'volatility': row['volatility']
                    }
                })

        print(f"✅ Generated {len(signals)} trading signals")
        return pd.DataFrame(signals)

    def prepare_ml_dataset(self, df, signals):
        """Prepare dataset for ML training."""
        print("\n📦 Preparing ML dataset...")

        # Merge signals with full feature set
        features = []
        labels_5m = []
        labels_30m = []

        for _, signal in signals.iterrows():
            # Get market state at signal time
            try:
                # Signal timestamp might be a column value, not index
                if hasattr(signal['timestamp'], 'to_pydatetime'):
                    signal_time = signal['timestamp']
                else:
                    signal_time = pd.Timestamp(signal['timestamp'])

                # Find nearest index
                if signal_time in df.index:
                    market_state = df.loc[signal_time]
                else:
                    # Find closest timestamp
                    time_diff = abs(df.index - signal_time)
                    closest_idx = time_diff.argmin()
                    market_state = df.iloc[closest_idx]

                feature_vector = [
                    market_state['rsi'],
                    market_state['macd'],
                    market_state['macd_signal'],
                    market_state['bb_upper'],
                    market_state['bb_lower'],
                    market_state['bb_position'],
                    market_state['sma_20'],
                    market_state['sma_50'],
                    market_state['volume_ratio'],
                    market_state['volatility'],
                    market_state['atr'],
                    signal['confidence']
                ]

                features.append(feature_vector)
                labels_5m.append(signal['win_5m'])
                labels_30m.append(signal['win_30m'])

            except Exception as e:
                # Debug: print first error to understand the issue
                if len(features) == 0:
                    print(f"Debug - Error processing signal: {e}")
                continue

        X = np.array(features)
        y_5m = np.array(labels_5m)
        y_30m = np.array(labels_30m)

        print(f"✅ Dataset shape: {X.shape}")
        print(f"✅ 5m win rate: {y_5m.mean():.2%}")
        print(f"✅ 30m win rate: {y_30m.mean():.2%}")

        return X, y_5m, y_30m

    def train_and_backtest(self, X, y_5m, y_30m):
        """Train models and perform backtesting."""
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
        import lightgbm as lgb

        print("\n🚀 Training and backtesting models...")

        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)

        results_5m = []
        results_30m = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            print(f"\n📈 Fold {fold+1}/5...")

            X_train, X_test = X[train_idx], X[test_idx]
            y_train_5m, y_test_5m = y_5m[train_idx], y_5m[test_idx]
            y_train_30m, y_test_30m = y_30m[train_idx], y_30m[test_idx]

            # Train 5-minute model
            model_5m = lgb.LGBMClassifier(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=5,
                random_state=42,
                verbosity=-1
            )
            model_5m.fit(X_train, y_train_5m)

            # Train 30-minute model
            model_30m = lgb.LGBMClassifier(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=5,
                random_state=42,
                verbosity=-1
            )
            model_30m.fit(X_train, y_train_30m)

            # Predictions
            pred_5m = model_5m.predict(X_test)
            pred_30m = model_30m.predict(X_test)
            prob_5m = model_5m.predict_proba(X_test)[:, 1]
            prob_30m = model_30m.predict_proba(X_test)[:, 1]

            # Calculate metrics
            results_5m.append({
                'accuracy': accuracy_score(y_test_5m, pred_5m),
                'precision': precision_score(y_test_5m, pred_5m, zero_division=0),
                'recall': recall_score(y_test_5m, pred_5m, zero_division=0),
                'auc': roc_auc_score(y_test_5m, prob_5m) if len(np.unique(y_test_5m)) > 1 else 0.5
            })

            results_30m.append({
                'accuracy': accuracy_score(y_test_30m, pred_30m),
                'precision': precision_score(y_test_30m, pred_30m, zero_division=0),
                'recall': recall_score(y_test_30m, pred_30m, zero_division=0),
                'auc': roc_auc_score(y_test_30m, prob_30m) if len(np.unique(y_test_30m)) > 1 else 0.5
            })

        # Calculate average metrics
        avg_5m = {k: np.mean([r[k] for r in results_5m]) for k in results_5m[0].keys()}
        avg_30m = {k: np.mean([r[k] for r in results_30m]) for k in results_30m[0].keys()}

        self.results['5m_model'] = avg_5m
        self.results['30m_model'] = avg_30m

        return model_5m, model_30m

    def simulate_trading(self, df, signals, model_5m, model_30m):
        """Simulate trading with trained models."""
        print("\n💰 Simulating trading strategy...")

        initial_capital = 10000
        capital = initial_capital
        trades = []

        for _, signal in signals.iterrows():
            try:
                idx = df.index.get_loc(signal['timestamp'], method='nearest')
                market_state = df.iloc[idx]

                # Prepare features
                features = np.array([[
                    market_state['rsi'],
                    market_state['macd'],
                    market_state['macd_signal'],
                    market_state['bb_upper'],
                    market_state['bb_lower'],
                    market_state['bb_position'],
                    market_state['sma_20'],
                    market_state['sma_50'],
                    market_state['volume_ratio'],
                    market_state['volatility'],
                    market_state['atr'],
                    signal['confidence']
                ]])

                # Get predictions
                prob_5m = model_5m.predict_proba(features)[0, 1]
                prob_30m = model_30m.predict_proba(features)[0, 1]

                # Trading logic
                if prob_5m > 0.6 and prob_30m > 0.55:
                    # Take position
                    position_size = capital * 0.1  # Risk 10% per trade

                    # Calculate actual return
                    actual_return = signal['return_5m'] if prob_5m > prob_30m else signal['return_30m']
                    pnl = position_size * actual_return
                    capital += pnl

                    trades.append({
                        'timestamp': signal['timestamp'],
                        'position_size': position_size,
                        'return': actual_return,
                        'pnl': pnl,
                        'capital': capital
                    })

            except Exception as e:
                continue

        if trades:
            total_return = (capital - initial_capital) / initial_capital
            win_rate = sum(1 for t in trades if t['pnl'] > 0) / len(trades)

            self.results['backtest'] = {
                'total_return': total_return,
                'win_rate': win_rate,
                'num_trades': len(trades),
                'final_capital': capital
            }
        else:
            self.results['backtest'] = {
                'total_return': 0,
                'win_rate': 0,
                'num_trades': 0,
                'final_capital': initial_capital
            }

        return trades

    def run_complete_pipeline(self):
        """Run the complete training and backtesting pipeline."""
        print("\n" + "="*60)
        print("🎯 SPY TRAINING PIPELINE WITH REAL DATA")
        print("="*60)

        # 1. Fetch real data
        df = self.fetch_real_spy_data(days=180)
        if df is None:
            print("❌ Failed to fetch data")
            return None

        # 2. Generate signals
        signals = self.generate_trading_signals(df)

        # 3. Prepare dataset
        X, y_5m, y_30m = self.prepare_ml_dataset(df, signals)

        # 4. Train and backtest
        model_5m, model_30m = self.train_and_backtest(X, y_5m, y_30m)

        # 5. Simulate trading
        trades = self.simulate_trading(df, signals, model_5m, model_30m)

        # Print results
        self.print_results()

        return self.results

    def print_results(self):
        """Print comprehensive results."""
        print("\n" + "="*60)
        print("📊 FINAL RESULTS WITH REAL SPY DATA")
        print("="*60)

        print("\n🎯 5-Minute Model Performance:")
        for metric, value in self.results['5m_model'].items():
            print(f"  {metric.capitalize()}: {value:.3f} ({value*100:.1f}%)")

        print("\n🎯 30-Minute Model Performance:")
        for metric, value in self.results['30m_model'].items():
            print(f"  {metric.capitalize()}: {value:.3f} ({value*100:.1f}%)")

        print("\n💰 Backtesting Results:")
        bt = self.results['backtest']
        print(f"  Total Return: {bt['total_return']:.2%}")
        print(f"  Win Rate: {bt['win_rate']:.2%}")
        print(f"  Number of Trades: {bt['num_trades']}")
        print(f"  Final Capital: ${bt['final_capital']:.2f}")

        print("\n" + "="*60)
        print("✅ TRAINING COMPLETE!")
        print("="*60)


if __name__ == "__main__":
    pipeline = SPYTrainingPipeline()
    results = pipeline.run_complete_pipeline()

    # Save results
    if results:
        with open(MODELS_DIR / 'spy_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n💾 Results saved to {MODELS_DIR / 'spy_results.json'}")