#!/usr/bin/env python3
"""
Test script to simulate 30-second polling with partial candles on today's data.
Compares against what would happen with 60-second polling on complete candles.
Uses the saved models from disk (same as live).
"""

import json
import sys
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from collections import defaultdict
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from no_whale_regime_backtest import (
    LongTermLSTM, ShortTermLSTM, RegimeLSTM,
    extract_xgb_features, extract_longterm_sequence, extract_shortterm_sequence, extract_regime_sequence,
    calculate_cvd_1min, aggregate_bars, calculate_volume_profile,
    calculate_footprint_candles, calculate_atr, calculate_bollinger_bands,
    calculate_adx, calculate_choppiness_index, calculate_rsi,
    LONGTERM_SEQ_LEN, SHORTTERM_SEQ_LEN, REGIME_SEQ_LEN, DEVICE,
)
import xgboost as xgb

# Paths
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')
BARS_FILE = os.path.join(DATA_DIR, 'bars_1s_nq.json')

# Config
HOLD_BARS = 20  # 20 minutes hold
STOP_LOSS_POINTS = 7.0
TAKE_PROFIT_POINTS = 35.0
POINT_VALUE = 20

def load_models():
    """Load saved models from disk."""
    models = {}

    # Load XGBoost models
    models['stage1_xgb'] = xgb.XGBClassifier()
    models['stage1_xgb'].load_model(os.path.join(MODEL_DIR, 'stage1_xgb.json'))

    models['timing_xgb'] = xgb.XGBClassifier()
    models['timing_xgb'].load_model(os.path.join(MODEL_DIR, 'timing_xgb.json'))

    models['final_xgb'] = xgb.XGBClassifier()
    models['final_xgb'].load_model(os.path.join(MODEL_DIR, 'final_xgb.json'))

    # Load LSTM models (with correct input dimensions)
    HIDDEN_DIM = 64
    NUM_LAYERS = 2
    EMBEDDING_DIM = 32

    models['longterm_lstm'] = LongTermLSTM(input_dim=7, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, embedding_dim=EMBEDDING_DIM).to(DEVICE)
    models['longterm_lstm'].load_state_dict(torch.load(os.path.join(MODEL_DIR, 'longterm_lstm.pt'), map_location=DEVICE))
    models['longterm_lstm'].eval()

    models['shortterm_lstm'] = ShortTermLSTM(input_dim=9, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, embedding_dim=EMBEDDING_DIM).to(DEVICE)
    models['shortterm_lstm'].load_state_dict(torch.load(os.path.join(MODEL_DIR, 'shortterm_lstm.pt'), map_location=DEVICE))
    models['shortterm_lstm'].eval()

    models['regime_lstm'] = RegimeLSTM(input_dim=18, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, embedding_dim=EMBEDDING_DIM).to(DEVICE)
    models['regime_lstm'].load_state_dict(torch.load(os.path.join(MODEL_DIR, 'regime_lstm.pt'), map_location=DEVICE))
    models['regime_lstm'].eval()

    return models

def generate_signal_at_time(models: Dict, bars_1s: List[Dict]) -> Optional[Dict]:
    """
    Generate signal using bars up to current point in time.
    Returns signal dict or None.
    """
    if len(bars_1s) < 7200:
        return None

    # Use last 6 hours
    bars_1s_window = bars_1s[-21600:]

    # Aggregate to different timeframes
    bars_1min, cvd_1min, cvd_ema_1min = calculate_cvd_1min(bars_1s_window)
    bars_5min = aggregate_bars(bars_1s_window, 5)
    bars_15min = aggregate_bars(bars_1s_window, 15)
    bars_1h = aggregate_bars(bars_1s_window, 60)

    if len(bars_1min) < 60 or len(bars_5min) < 72:
        return None

    current_price = bars_1s_window[-1]['c']
    bar_idx = len(bars_1min) - 1

    vp = calculate_volume_profile(bars_1s_window)

    try:
        # Stage 1
        xgb_features = extract_xgb_features(
            bars_1min, bars_5min, bars_15min, bars_1h,
            cvd_1min, cvd_ema_1min, bar_idx, vp
        )
        if xgb_features is None:
            return None

        X_stage1 = np.array(list(xgb_features.values())).reshape(1, -1)
        stage1_pred = models['stage1_xgb'].predict_proba(X_stage1)[0]
        stage1_direction = 1 if stage1_pred[1] > 0.5 else 0
        stage1_confidence = max(stage1_pred)

        if stage1_confidence < 0.55:
            return {'skip': 'stage1', 'confidence': stage1_confidence, 'price': current_price}

        # Features for LSTM
        footprint_1min = calculate_footprint_candles(bars_1s_window)
        atr_1min = calculate_atr(bars_1min)
        bb_middle, bb_upper, bb_lower = calculate_bollinger_bands(bars_1min)
        adx_1min = calculate_adx(bars_1min)
        chop_1min = calculate_choppiness_index(bars_1min)
        rsi_1min = calculate_rsi(bars_1min)

        longterm_seq = extract_longterm_sequence(bars_5min, cvd_1min, bar_idx)
        shortterm_seq = extract_shortterm_sequence(bars_1s_window, footprint_1min, len(bars_1s_window) - 1)
        regime_seq = extract_regime_sequence(
            bars_1min, cvd_1min, cvd_ema_1min, atr_1min,
            bb_middle, bb_upper, bb_lower,
            adx_1min, chop_1min, rsi_1min,
            bar_idx, vp
        )

        if longterm_seq is None or shortterm_seq is None or regime_seq is None:
            return None

        with torch.no_grad():
            longterm_tensor = torch.FloatTensor(longterm_seq).unsqueeze(0).to(DEVICE)
            shortterm_tensor = torch.FloatTensor(shortterm_seq).unsqueeze(0).to(DEVICE)
            regime_tensor = torch.FloatTensor(regime_seq).unsqueeze(0).to(DEVICE)

            longterm_embed = models['longterm_lstm'](longterm_tensor).cpu().numpy().flatten()
            shortterm_embed = models['shortterm_lstm'](shortterm_tensor).cpu().numpy().flatten()
            regime_embed = models['regime_lstm'](regime_tensor).cpu().numpy().flatten()

        # Timing
        current_atr = atr_1min[-1] if len(atr_1min) > 0 else 1.0
        current_adx = adx_1min[-1] if len(adx_1min) > 0 else 0
        current_chop = chop_1min[-1] if len(chop_1min) > 0 else 50
        current_rsi = rsi_1min[-1] if len(rsi_1min) > 0 else 50
        current_bb_up = bb_upper[-1] if len(bb_upper) > 0 else current_price
        current_bb_low = bb_lower[-1] if len(bb_lower) > 0 else current_price

        bb_width = (current_bb_up - current_bb_low) / current_price if current_price > 0 else 0
        bb_position = (current_price - current_bb_low) / (current_bb_up - current_bb_low) if current_bb_up > current_bb_low else 0.5
        atr_norm = current_atr / current_price if current_price > 0 else 0

        timing_features = np.concatenate([
            regime_embed,
            np.array([atr_norm, current_adx / 100.0, current_chop / 100.0, (current_rsi - 50) / 50.0, bb_width, bb_position])
        ])

        timing_pred = models['timing_xgb'].predict_proba(timing_features.reshape(1, -1))[0]
        timing_confidence = timing_pred[1]

        if timing_confidence < 0.6:
            return {'skip': 'timing', 'stage1': stage1_confidence, 'timing': timing_confidence, 'price': current_price}

        # Final
        xgb_features_arr = np.array(list(xgb_features.values()))
        final_features = np.concatenate([
            xgb_features_arr, longterm_embed, shortterm_embed, regime_embed,
            np.array([stage1_confidence, timing_confidence])
        ])

        final_pred = models['final_xgb'].predict_proba(final_features.reshape(1, -1))[0]
        final_confidence = final_pred[1]

        if final_confidence < 0.55:
            return {'skip': 'final', 'stage1': stage1_confidence, 'timing': timing_confidence, 'final': final_confidence, 'price': current_price}

        direction = 'buy' if stage1_direction == 1 else 'sell'

        return {
            'signal': True,
            'direction': direction,
            'price': current_price,
            'stage1': stage1_confidence,
            'timing': timing_confidence,
            'final': final_confidence,
        }

    except Exception as e:
        return {'error': str(e)}

def simulate_polling(bars_1s: List[Dict], poll_interval_seconds: int, date_filter: str = None):
    """
    Simulate polling at specified interval.
    Returns list of trades with outcomes.
    """
    print(f"\n{'='*60}")
    print(f"SIMULATING {poll_interval_seconds}s POLLING")
    print(f"{'='*60}")

    models = load_models()
    print("Models loaded from disk")

    # Filter to specific date if provided
    if date_filter:
        bars_1s = [b for b in bars_1s if date_filter in b['t']]
        print(f"Filtered to {date_filter}: {len(bars_1s):,} bars")

    # Build time index
    bar_times = {}
    for i, bar in enumerate(bars_1s):
        ts = bar['t'][:19]  # Truncate to seconds
        bar_times[ts] = i

    # Get all historical bars before today for context
    all_bars = []
    with open(BARS_FILE, 'r') as f:
        data = json.load(f)
    all_bars = data['bars']

    # Find where today starts
    today_start_idx = None
    for i, bar in enumerate(all_bars):
        if date_filter in bar['t']:
            today_start_idx = i
            break

    if today_start_idx is None:
        print("Could not find today's data")
        return []

    print(f"Today starts at index {today_start_idx:,}")
    print(f"Historical context: {today_start_idx:,} bars")

    # Simulate polling
    trades = []
    signals = []
    in_position_until = None

    # Start from market open (roughly 9:30 ET = 14:30 UTC)
    start_time = datetime.fromisoformat(f"{date_filter}T14:30:00")
    end_time = datetime.fromisoformat(f"{date_filter}T21:00:00")  # 4pm ET

    current_time = start_time
    poll_count = 0

    while current_time < end_time:
        poll_count += 1
        time_str = current_time.strftime("%Y-%m-%dT%H:%M:%S")

        # Find bar index at this time
        bar_idx = bar_times.get(time_str)
        if bar_idx is None:
            # Find closest earlier bar
            for offset in range(10):
                check_time = (current_time - timedelta(seconds=offset)).strftime("%Y-%m-%dT%H:%M:%S")
                if check_time in bar_times:
                    bar_idx = bar_times[check_time]
                    break

        if bar_idx is not None:
            # Build bars up to this point
            absolute_idx = today_start_idx + bar_idx
            bars_up_to_now = all_bars[:absolute_idx + 1]

            # Check if we're in a position
            if in_position_until and current_time < in_position_until:
                pass  # Skip, in position
            else:
                # Generate signal
                signal = generate_signal_at_time(models, bars_up_to_now)

                if signal and signal.get('signal'):
                    entry_price = signal['price']
                    direction = signal['direction']

                    # Calculate exit (20 minutes later)
                    exit_time = current_time + timedelta(minutes=HOLD_BARS)
                    exit_time_str = exit_time.strftime("%Y-%m-%dT%H:%M:%S")

                    # Find exit price
                    exit_bar_idx = bar_times.get(exit_time_str)
                    if exit_bar_idx is None:
                        for offset in range(30):
                            check_time = (exit_time + timedelta(seconds=offset)).strftime("%Y-%m-%dT%H:%M:%S")
                            if check_time in bar_times:
                                exit_bar_idx = bar_times[check_time]
                                break

                    if exit_bar_idx is not None:
                        exit_price = bars_1s[exit_bar_idx]['c']

                        if direction == 'buy':
                            pnl_pts = exit_price - entry_price
                        else:
                            pnl_pts = entry_price - exit_price

                        pnl_dollars = pnl_pts * POINT_VALUE

                        trade = {
                            'entry_time': time_str,
                            'exit_time': exit_time_str,
                            'direction': direction,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'pnl_pts': pnl_pts,
                            'pnl_dollars': pnl_dollars,
                            'stage1': signal['stage1'],
                            'timing': signal['timing'],
                            'final': signal['final'],
                        }
                        trades.append(trade)
                        in_position_until = exit_time

                        print(f"\n{time_str[11:]} | {direction.upper()} @ {entry_price:.2f}")
                        print(f"  Stage1: {signal['stage1']:.0%} | Timing: {signal['timing']:.0%} | Final: {signal['final']:.0%}")
                        print(f"  Exit @ {exit_price:.2f} -> P&L: {pnl_pts:+.2f} pts (${pnl_dollars:+.2f})")

                signals.append(signal)

        current_time += timedelta(seconds=poll_interval_seconds)

    # Summary
    print(f"\n{'='*60}")
    print(f"RESULTS: {poll_interval_seconds}s POLLING")
    print(f"{'='*60}")
    print(f"Polls: {poll_count}")
    print(f"Trades: {len(trades)}")

    if trades:
        wins = sum(1 for t in trades if t['pnl_pts'] > 0)
        losses = len(trades) - wins
        total_pnl = sum(t['pnl_pts'] for t in trades)
        total_dollars = sum(t['pnl_dollars'] for t in trades)

        print(f"Wins: {wins} | Losses: {losses} | Win Rate: {wins/len(trades):.1%}")
        print(f"Total P&L: {total_pnl:+.2f} pts (${total_dollars:+.2f})")

        print(f"\nTrade Details:")
        for i, t in enumerate(trades, 1):
            print(f"  {i}. {t['entry_time'][11:]} {t['direction'].upper()} @ {t['entry_price']:.2f} -> {t['exit_price']:.2f} = {t['pnl_pts']:+.2f} pts")

    return trades

def main():
    print("Loading bar data...")
    with open(BARS_FILE, 'r') as f:
        data = json.load(f)
    all_bars = data['bars']
    print(f"Loaded {len(all_bars):,} bars")

    today = "2025-12-09"
    today_bars = [b for b in all_bars if today in b['t']]
    print(f"Today's bars: {len(today_bars):,}")

    # Test 30-second polling (like live)
    trades_30s = simulate_polling(all_bars, poll_interval_seconds=30, date_filter=today)

    # Test 60-second polling (minute bar close)
    trades_60s = simulate_polling(all_bars, poll_interval_seconds=60, date_filter=today)

    # Compare
    print(f"\n{'='*60}")
    print("COMPARISON")
    print(f"{'='*60}")

    pnl_30s = sum(t['pnl_pts'] for t in trades_30s) if trades_30s else 0
    pnl_60s = sum(t['pnl_pts'] for t in trades_60s) if trades_60s else 0

    print(f"30s polling: {len(trades_30s)} trades, {pnl_30s:+.2f} pts (${pnl_30s * POINT_VALUE:+.2f})")
    print(f"60s polling: {len(trades_60s)} trades, {pnl_60s:+.2f} pts (${pnl_60s * POINT_VALUE:+.2f})")
    print(f"Difference: {pnl_30s - pnl_60s:+.2f} pts")

if __name__ == "__main__":
    main()
