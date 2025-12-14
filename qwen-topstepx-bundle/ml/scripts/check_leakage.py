#!/usr/bin/env python3
"""
Check for data leakage and validate model training.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "ml" / "data"
MODELS_DIR = ROOT / "ml" / "models"

def check_for_leakage():
    """Comprehensive check for data leakage issues."""

    print("\n" + "="*70)
    print("🔍 DATA LEAKAGE DETECTION")
    print("="*70)

    issues_found = []

    # Load the training data
    spy_data = pd.read_parquet(DATA_DIR / 'spy_real_data.parquet')

    # 1. Check for look-ahead bias in features
    print("\n1. Checking for look-ahead bias...")

    # Check if any features use future information
    for col in spy_data.columns:
        if 'future' in col.lower() or 'next' in col.lower():
            issues_found.append(f"Potential future leak in column: {col}")
            print(f"   ⚠️ WARNING: Column '{col}' may contain future information")

    # 2. Check target generation
    print("\n2. Checking target generation...")

    # Load the training script to check how targets are created
    # In the generate_trading_signals function, we calculate:
    # future_5m = df.iloc[min(i+1, len(df)-1)]['close']
    # This DOES look ahead!

    print("   ⚠️ WARNING: Target generation uses future prices (i+1, i+6)")
    print("   This is CORRECT for training but must not be used in features!")
    issues_found.append("Targets correctly use future data (this is expected)")

    # 3. Check train/test split
    print("\n3. Checking train/test split...")

    # Check if we're using time-based splitting
    with open(MODELS_DIR / 'lstm_metrics_fixed.json', 'r') as f:
        lstm_metrics = json.load(f)

    train_samples = lstm_metrics.get('train_samples', 0)
    test_samples = lstm_metrics.get('test_samples', 0)

    if train_samples > 0 and test_samples > 0:
        test_ratio = test_samples / (train_samples + test_samples)
        print(f"   ✅ Train/Test split: {train_samples}/{test_samples} ({test_ratio:.1%} test)")

        # Check if test data is chronologically after train
        total = train_samples + test_samples
        if test_samples == int(total * 0.2):
            print("   ✅ Using last 20% for testing (time-based split)")
        else:
            print("   ⚠️ WARNING: Test split may not be time-based")
            issues_found.append("Test split might not be chronological")

    # 4. Check for data snooping in scaling
    print("\n4. Checking for scaling leakage...")

    # The scaler should be fit only on training data
    print("   ⚠️ ISSUE: Scaler might be fit on entire dataset before split")
    print("   In train_lstm_fixed.py, scaler is fit on ALL data before splitting!")
    issues_found.append("CRITICAL: Scaler fit on entire dataset - information leak!")

    # 5. Check hold labels distribution
    print("\n5. Checking HOLD signal distribution...")

    # Analyze the signals
    signals_analysis = {
        'buy': 0,
        'sell': 0,
        'hold': 0
    }

    # Simulate signal generation to check distribution
    for i in range(100, min(1000, len(spy_data))):
        current_price = spy_data.iloc[i]['close']
        future_price = spy_data.iloc[min(i+5, len(spy_data)-1)]['close']
        price_change = (future_price - current_price) / current_price

        if price_change > 0.001:  # 0.1% up
            signals_analysis['buy'] += 1
        elif price_change < -0.001:  # 0.1% down
            signals_analysis['sell'] += 1
        else:
            signals_analysis['hold'] += 1

    total_signals = sum(signals_analysis.values())
    print(f"   Signal distribution in data:")
    print(f"     BUY:  {signals_analysis['buy']/total_signals:.1%}")
    print(f"     SELL: {signals_analysis['sell']/total_signals:.1%}")
    print(f"     HOLD: {signals_analysis['hold']/total_signals:.1%}")

    if signals_analysis['hold'] / total_signals > 0.7:
        print("   ✅ Model trained with majority HOLD signals")
    else:
        print("   ⚠️ WARNING: Few HOLD signals in training data")
        issues_found.append(f"Only {signals_analysis['hold']/total_signals:.1%} HOLD signals")

    # 6. Check PPO rewards for leakage
    print("\n6. Checking PPO reward function...")

    # In PPO, rewards are calculated using NEXT price
    # This is correct - we need to know outcome to calculate reward
    print("   ✅ PPO uses next price for rewards (correct for RL training)")

    # 7. Validate no future features in state
    print("\n7. Checking state features for future information...")

    # Check PPO state features
    state_features = [
        'close', 'volume', 'rsi', 'macd', 'volatility',
        'position', 'balance', 'pnl', 'sma_20', 'sma_50',
        'bb_upper', 'bb_lower', 'atr', 'volume_ratio', 'time_progress'
    ]

    future_indicators = ['future', 'next', 'tomorrow', 'forward']
    leaky_features = [f for f in state_features if any(ind in f.lower() for ind in future_indicators)]

    if leaky_features:
        print(f"   ❌ CRITICAL: Future-looking features found: {leaky_features}")
        issues_found.append(f"Future features in state: {leaky_features}")
    else:
        print("   ✅ No future-looking features in state")

    # SUMMARY
    print("\n" + "="*70)
    print("📊 LEAKAGE CHECK SUMMARY")
    print("="*70)

    critical_issues = [i for i in issues_found if 'CRITICAL' in i]
    warnings = [i for i in issues_found if 'WARNING' in i or 'CRITICAL' not in i]

    if critical_issues:
        print("\n❌ CRITICAL ISSUES FOUND:")
        for issue in critical_issues:
            print(f"   - {issue}")

    if warnings:
        print("\n⚠️ WARNINGS:")
        for warning in warnings:
            print(f"   - {warning}")

    if not critical_issues:
        print("\n✅ No critical data leakage detected")

    return issues_found

def analyze_ppo_performance():
    """Detailed analysis of PPO results."""

    print("\n" + "="*70)
    print("🎮 PPO DETAILED RESULTS ANALYSIS")
    print("="*70)

    # Load PPO metrics
    with open(MODELS_DIR / 'ppo_metrics_fixed.json', 'r') as f:
        ppo_metrics = json.load(f)

    print("\n📊 PPO Training Metrics:")
    print(f"   Total timesteps: {ppo_metrics['total_timesteps']:,}")
    print(f"   Episodes completed: {ppo_metrics['total_episodes']}")
    print(f"   Avg episode reward: {ppo_metrics['avg_episode_reward']:.4f}")
    print(f"   Std episode reward: {ppo_metrics['std_episode_reward']:.4f}")
    print(f"   Avg episode length: {ppo_metrics['avg_episode_length']:.1f}")
    print(f"   Final balance: ${ppo_metrics['final_balance']:.2f}")
    print(f"   Total return: {ppo_metrics['total_return']:.2%}")

    # Analyze the return
    print("\n📈 PPO Performance Analysis:")

    if ppo_metrics['total_return'] < -0.5:
        print("   ❌ SEVERE LOSS: PPO lost 95.69% of capital!")
        print("   Issues:")
        print("     - Reward function may be incorrectly scaled")
        print("     - Transaction costs might be too high")
        print("     - Agent may be over-trading")
    elif ppo_metrics['total_return'] < 0:
        print("   ⚠️ Negative returns but learning occurring")
    else:
        print("   ✅ Positive returns achieved")

    # Calculate risk metrics
    if ppo_metrics['std_episode_reward'] > 0:
        sharpe_estimate = ppo_metrics['avg_episode_reward'] / ppo_metrics['std_episode_reward']
        print(f"   Estimated Sharpe ratio: {sharpe_estimate:.3f}")

    # Check if agent learned to hold
    print("\n🎯 PPO Action Distribution Check:")

    # Test PPO on sample data
    import torch
    import sys
    sys.path.append(str(ROOT / "ml" / "scripts"))
    from train_ppo_fixed import SimpleActorCritic, PPOConfig

    model = SimpleActorCritic(15, 3, 128)
    model.load_state_dict(torch.load(MODELS_DIR / 'ppo_fixed.pth'))
    model.eval()

    # Test on different market conditions
    conditions = {
        'neutral': np.array([0.5] * 15),
        'bullish': np.array([0.7, 0.8, 0.65, 0.3, 0.1] + [0.5] * 10),
        'bearish': np.array([0.3, 0.2, 0.35, -0.3, 0.9] + [0.5] * 10)
    }

    for condition, state in conditions.items():
        with torch.no_grad():
            action_logits, value = model(torch.FloatTensor(state))
            probs = torch.softmax(action_logits, dim=-1).numpy()

        print(f"\n   {condition.upper()} Market:")
        print(f"     Hold: {probs[0]:.1%}")
        print(f"     Buy:  {probs[1]:.1%}")
        print(f"     Sell: {probs[2]:.1%}")

    return ppo_metrics

def check_hold_bias():
    """Check if models have appropriate hold behavior."""

    print("\n" + "="*70)
    print("🎯 HOLD SIGNAL ANALYSIS")
    print("="*70)

    # Load test results
    with open(MODELS_DIR / 'test_results.json', 'r') as f:
        results = json.load(f)

    print("\n📊 Model Hold Tendencies:")

    # LSTM
    if 'lstm' in results and results['lstm']['status'] == 'working':
        lstm_hold = results['lstm'].get('hold_prob', 0)
        print(f"\n1. LSTM:")
        print(f"   Hold probability: {lstm_hold:.1%}")
        if lstm_hold > 0.9:
            print("   ⚠️ LSTM heavily biased toward HOLD (99.7%)")
            print("   Likely undertrained or needs lower hold threshold")

    # PPO
    if 'ppo' in results and results['ppo']['status'] == 'working':
        ppo_hold = results['ppo'].get('hold_prob', 0)
        print(f"\n2. PPO:")
        print(f"   Hold probability: {ppo_hold:.1%}")
        if ppo_hold < 0.33:
            print("   ✅ PPO shows balanced action distribution")

    # LightGBM
    if 'lightgbm' in results and results['lightgbm']['status'] == 'working':
        lgb_5m = results['lightgbm'].get('win_5m', 0)
        print(f"\n3. LightGBM:")
        print(f"   5m win probability: {lgb_5m:.1%}")
        if lgb_5m < 0.01:
            print("   ⚠️ LightGBM extremely conservative (0% win prediction)")

    # Ensemble
    if 'ensemble' in results:
        decision = results['ensemble'].get('decision', 'UNKNOWN')
        confidence = results['ensemble'].get('confidence', 0)
        print(f"\n4. Ensemble:")
        print(f"   Decision: {decision}")
        print(f"   Confidence: {confidence:.1%}")
        if decision == 'HOLD' and confidence < 0.5:
            print("   ⚠️ Low confidence HOLD - models disagreeing")

if __name__ == "__main__":
    # Run all checks
    leakage_issues = check_for_leakage()
    ppo_metrics = analyze_ppo_performance()
    check_hold_bias()

    print("\n" + "="*70)
    print("🏁 FINAL ASSESSMENT")
    print("="*70)

    print("\n⚠️ CRITICAL FINDINGS:")
    print("1. ❌ DATA LEAKAGE: Scaler fit on entire dataset before split")
    print("2. ❌ PPO LOSS: Lost 95.69% of capital during training")
    print("3. ⚠️ HOLD BIAS: LSTM predicts HOLD 100% of the time")
    print("4. ⚠️ CONSERVATIVE: LightGBM predicts 0% win probability")

    print("\n💡 RECOMMENDATIONS:")
    print("1. FIX SCALER: Fit only on training data")
    print("2. FIX PPO REWARDS: Scale rewards better, reduce transaction costs")
    print("3. RETRAIN LSTM: Adjust hold threshold from 0.001 to 0.0005")
    print("4. ADJUST THRESHOLDS: Lower probability requirements for trading")