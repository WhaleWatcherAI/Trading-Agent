#!/usr/bin/env python3
"""
Test all fixed models together - LightGBM, LSTM, and PPO.
"""

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import lightgbm as lgb
import pickle
import warnings
warnings.filterwarnings('ignore')

# Add scripts to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "ml" / "scripts"))

from train_lstm_fixed import SimpleLSTM
from train_ppo_fixed import SimpleActorCritic, PPOConfig

DATA_DIR = ROOT / "ml" / "data"
MODELS_DIR = ROOT / "ml" / "models"


def test_all_models():
    """Test all three model types together."""

    print("\n" + "="*70)
    print("🧪 TESTING ALL FIXED MODELS TOGETHER")
    print("="*70)

    # Load test data
    spy_data = pd.read_parquet(DATA_DIR / 'spy_real_data.parquet')
    print(f"\n📊 Using SPY data: {len(spy_data)} bars")

    # Prepare a sample for testing
    sample_idx = len(spy_data) - 100
    sample = spy_data.iloc[sample_idx]

    print(f"\nTest sample from: {spy_data.index[sample_idx]}")
    print(f"  SPY Price: ${sample['close']:.2f}")
    print(f"  RSI: {sample['rsi']:.1f}")
    print(f"  Volume: {sample['volume']:,.0f}")

    results = {}

    # ========================================
    # 1. TEST LIGHTGBM MODELS
    # ========================================
    print("\n" + "-"*50)
    print("1️⃣ LIGHTGBM MODELS")
    print("-"*50)

    try:
        # Load models
        lgb_5m = lgb.Booster(model_file=str(MODELS_DIR / 'meta_label_5m.txt'))
        lgb_30m = lgb.Booster(model_file=str(MODELS_DIR / 'meta_label_30m.txt'))

        # Prepare features (15 features as trained)
        lgb_features = np.array([
            sample.get('rsi', 50),
            sample.get('macd', 0),
            sample.get('macd_signal', 0),
            sample.get('bb_upper', sample['close'] + 10),
            sample.get('bb_lower', sample['close'] - 10),
            sample.get('bb_position', 0.5),
            sample.get('sma_20', sample['close']),
            sample.get('sma_50', sample['close']),
            sample.get('volume_ratio', 1),
            sample.get('volatility', 0.01),
            sample.get('atr', 1),
            sample.get('close', 100) - sample.get('sma_20', 100),  # dist_to_poc_ticks
            sample.get('close', 100) * 1.01,  # vwap approximation
            sample.get('close', 100),  # price
            0.5  # confidence
        ]).reshape(1, -1)

        # Get predictions
        prob_5m = lgb_5m.predict(lgb_features)[0]
        prob_30m = lgb_30m.predict(lgb_features)[0]

        print("✅ LightGBM Results:")
        print(f"  5-min win probability: {prob_5m:.3%}")
        print(f"  30-min win probability: {prob_30m:.3%}")

        results['lightgbm'] = {
            'status': 'working',
            'win_5m': float(prob_5m),
            'win_30m': float(prob_30m)
        }

    except Exception as e:
        print(f"❌ LightGBM Error: {e}")
        results['lightgbm'] = {'status': 'error', 'error': str(e)}

    # ========================================
    # 2. TEST LSTM MODEL
    # ========================================
    print("\n" + "-"*50)
    print("2️⃣ LSTM MODEL (FIXED)")
    print("-"*50)

    try:
        # Load model and scaler
        lstm_model = SimpleLSTM(
            input_size=5,  # 5 features as trained
            hidden_size=64,
            num_layers=2
        )
        lstm_model.load_state_dict(torch.load(MODELS_DIR / 'lstm_fixed.pth'))
        lstm_model.eval()

        with open(MODELS_DIR / 'lstm_scaler_fixed.pkl', 'rb') as f:
            scaler = pickle.load(f)

        # Prepare sequence (last 60 time steps)
        start_idx = max(0, sample_idx - 59)
        sequence_data = spy_data.iloc[start_idx:sample_idx+1]

        # Select features
        feature_cols = ['close', 'volume', 'rsi', 'macd', 'volatility']
        sequence = sequence_data[feature_cols].values

        # Scale
        sequence_scaled = scaler.transform(sequence)

        # Convert to tensor
        sequence_tensor = torch.FloatTensor(sequence_scaled).unsqueeze(0)

        # Get predictions
        with torch.no_grad():
            signals, movement = lstm_model(sequence_tensor)

        signal_probs = signals[0].numpy()
        movement_pred = movement[0].item()

        print("✅ LSTM Results:")
        print(f"  Hold probability: {signal_probs[0]:.3%}")
        print(f"  Buy probability: {signal_probs[1]:.3%}")
        print(f"  Sell probability: {signal_probs[2]:.3%}")
        print(f"  Expected movement: {movement_pred:.3f}%")

        # Determine signal
        signal_idx = np.argmax(signal_probs)
        signal_map = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}

        results['lstm'] = {
            'status': 'working',
            'signal': signal_map[signal_idx],
            'buy_prob': float(signal_probs[1]),
            'sell_prob': float(signal_probs[2]),
            'hold_prob': float(signal_probs[0]),
            'movement': float(movement_pred)
        }

    except Exception as e:
        print(f"❌ LSTM Error: {e}")
        results['lstm'] = {'status': 'error', 'error': str(e)}

    # ========================================
    # 3. TEST PPO MODEL
    # ========================================
    print("\n" + "-"*50)
    print("3️⃣ PPO MODEL (FIXED)")
    print("-"*50)

    try:
        # Load model
        config = PPOConfig()
        ppo_model = SimpleActorCritic(
            config.state_dim,
            config.action_dim,
            config.hidden_dim
        )
        ppo_model.load_state_dict(torch.load(MODELS_DIR / 'ppo_fixed.pth'))
        ppo_model.eval()

        # Prepare state (15 features)
        ppo_state = np.array([
            sample.get('close', 100) / 1000,
            sample.get('volume', 1000) / 10000,
            sample.get('rsi', 50) / 100,
            sample.get('macd', 0) / 10,
            sample.get('volatility', 0.01),
            0.2,  # position (normalized)
            1.0,  # balance (normalized)
            0.01,  # P&L (normalized)
            sample.get('sma_20', 100) / 1000,
            sample.get('sma_50', 100) / 1000,
            sample.get('bb_upper', 100) / 1000,
            sample.get('bb_lower', 100) / 1000,
            sample.get('atr', 1) / 10,
            sample.get('volume_ratio', 1),
            0.5  # time progress
        ])

        # Get predictions
        with torch.no_grad():
            state_tensor = torch.FloatTensor(ppo_state)
            action_logits, value = ppo_model(state_tensor)
            action_probs = torch.softmax(action_logits, dim=-1).numpy()

        print("✅ PPO Results:")
        print(f"  Hold probability: {action_probs[0]:.3%}")
        print(f"  Buy probability: {action_probs[1]:.3%}")
        print(f"  Sell probability: {action_probs[2]:.3%}")
        print(f"  State value: {value.item():.3f}")

        # Determine action
        action_idx = np.argmax(action_probs)
        action_map = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}

        results['ppo'] = {
            'status': 'working',
            'action': action_map[action_idx],
            'hold_prob': float(action_probs[0]),
            'buy_prob': float(action_probs[1]),
            'sell_prob': float(action_probs[2]),
            'state_value': float(value.item())
        }

    except Exception as e:
        print(f"❌ PPO Error: {e}")
        results['ppo'] = {'status': 'error', 'error': str(e)}

    # ========================================
    # 4. ENSEMBLE DECISION
    # ========================================
    print("\n" + "-"*50)
    print("4️⃣ ENSEMBLE DECISION")
    print("-"*50)

    # Combine all model predictions
    buy_votes = 0
    sell_votes = 0
    hold_votes = 0

    # Weight each model's opinion
    if results.get('lstm', {}).get('status') == 'working':
        if results['lstm']['signal'] == 'BUY':
            buy_votes += 2  # LSTM gets 2 votes
        elif results['lstm']['signal'] == 'SELL':
            sell_votes += 2
        else:
            hold_votes += 2

    if results.get('ppo', {}).get('status') == 'working':
        if results['ppo']['action'] == 'BUY':
            buy_votes += 2  # PPO gets 2 votes
        elif results['ppo']['action'] == 'SELL':
            sell_votes += 2
        else:
            hold_votes += 2

    if results.get('lightgbm', {}).get('status') == 'working':
        if results['lightgbm']['win_5m'] > 0.6:
            buy_votes += 1  # LightGBM gets 1 vote
        elif results['lightgbm']['win_5m'] < 0.4:
            sell_votes += 1
        else:
            hold_votes += 1

    # Determine ensemble decision
    if buy_votes > sell_votes and buy_votes > hold_votes:
        ensemble_decision = 'BUY'
        confidence = buy_votes / (buy_votes + sell_votes + hold_votes)
    elif sell_votes > buy_votes and sell_votes > hold_votes:
        ensemble_decision = 'SELL'
        confidence = sell_votes / (buy_votes + sell_votes + hold_votes)
    else:
        ensemble_decision = 'HOLD'
        confidence = hold_votes / (buy_votes + sell_votes + hold_votes)

    print(f"\n🎯 ENSEMBLE DECISION: {ensemble_decision}")
    print(f"📊 Confidence: {confidence:.1%}")
    print(f"   Buy votes: {buy_votes}")
    print(f"   Sell votes: {sell_votes}")
    print(f"   Hold votes: {hold_votes}")

    # ========================================
    # SUMMARY
    # ========================================
    print("\n" + "="*70)
    print("📊 SUMMARY OF ALL MODELS")
    print("="*70)

    working_models = sum(1 for m in results.values() if m.get('status') == 'working')
    print(f"\n✅ Working Models: {working_models}/3")

    if results.get('lightgbm', {}).get('status') == 'working':
        print("\n✅ LightGBM: WORKING")
        print(f"   - 5m win prob: {results['lightgbm']['win_5m']:.1%}")
        print(f"   - 30m win prob: {results['lightgbm']['win_30m']:.1%}")
    else:
        print("\n❌ LightGBM: NOT WORKING")

    if results.get('lstm', {}).get('status') == 'working':
        print("\n✅ LSTM: WORKING")
        print(f"   - Signal: {results['lstm']['signal']}")
        print(f"   - Movement: {results['lstm']['movement']:.2f}%")
        print(f"   - Buy prob: {results['lstm']['buy_prob']:.1%}")
    else:
        print("\n❌ LSTM: NOT WORKING")

    if results.get('ppo', {}).get('status') == 'working':
        print("\n✅ PPO: WORKING")
        print(f"   - Action: {results['ppo']['action']}")
        print(f"   - Buy prob: {results['ppo']['buy_prob']:.1%}")
        print(f"   - Value: {results['ppo']['state_value']:.2f}")
    else:
        print("\n❌ PPO: NOT WORKING")

    print("\n" + "="*70)
    print(f"🎯 FINAL TRADING DECISION: {ensemble_decision} (Confidence: {confidence:.1%})")
    print("="*70)

    # Save results
    results['ensemble'] = {
        'decision': ensemble_decision,
        'confidence': confidence,
        'buy_votes': buy_votes,
        'sell_votes': sell_votes,
        'hold_votes': hold_votes
    }

    with open(MODELS_DIR / 'test_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    results = test_all_models()

    if all(m.get('status') == 'working' for m in results.values() if isinstance(m, dict)):
        print("\n🎉 ALL MODELS FIXED AND WORKING TOGETHER!")
    else:
        print("\n⚠️ Some models still have issues - check errors above")