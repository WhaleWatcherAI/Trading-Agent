#!/usr/bin/env python3
"""
No-Whale Regime Live Trading with TopstepX

Uses the EXACT same XGBoost + LSTM ensemble from no_whale_regime_backtest.py.
Data flow:
1. Load local bars_1s_nq.json as base historical data
2. Fetch any missing bars from API (last saved → now)
3. Merge and save updated bars back to disk
4. Train with tiered validation (multiple holdout days)
5. Cache live bars in memory, continuously appending
"""

import json
import sys
import os
import time
import signal
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

# Load .env file (override=False so command-line env vars take precedence)
from dotenv import load_dotenv
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '.env')
load_dotenv(env_path, override=False)

import numpy as np
import requests

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import all the backtest functions - reuse exactly
from no_whale_regime_backtest import (
    # LSTM models
    LongTermLSTM, ShortTermLSTM, RegimeLSTM,
    # Feature extraction
    extract_longterm_sequence, extract_shortterm_sequence, extract_regime_sequence,
    extract_xgb_features,
    # Indicator calculations
    calculate_atr, calculate_bollinger_bands, calculate_adx, calculate_choppiness_index, calculate_rsi,
    calculate_cvd_1min, calculate_volume_profile, calculate_footprint_candles,
    aggregate_bars, find_swing_highs_lows, detect_trade_trigger,
    # Training functions
    train_lstm_models, train_stage1_xgboost, train_timing_xgboost, build_final_training_data,
    # Backtest/simulation functions
    run_single_day,
    # Constants
    LONGTERM_SEQ_LEN, SHORTTERM_SEQ_LEN, REGIME_SEQ_LEN, EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS,
    MIN_BARS_BETWEEN_TRADES, HOLD_BARS, DEVICE,
)

import torch
import xgboost as xgb

# =============================================================================
# CONFIGURATION
# =============================================================================

# Load from environment
TOPSTEPX_API_KEY = os.environ.get('TOPSTEPX_API_KEY', '')
TOPSTEPX_USERNAME = os.environ.get('TOPSTEPX_USERNAME', '')
TOPSTEPX_BASE_URL = os.environ.get('TOPSTEPX_BASE_URL', 'https://api.topstepx.com')
TOPSTEPX_ACCOUNT_ID = int(os.environ.get('TOPSTEPX_ACCOUNT_ID', '0'))
TOPSTEPX_CONTRACT_ID = os.environ.get('TOPSTEPX_CONTRACT_ID', 'CON.F.US.ENQ.Z25')

# Derive symbol from contract ID for file naming
def get_symbol_from_contract(contract_id):
    """Extract symbol from contract ID - each contract size gets its own file."""
    # Micros
    if 'MNQ' in contract_id:
        return 'mnq'
    elif 'MES' in contract_id:
        return 'mes'
    elif 'MGC' in contract_id:
        return 'mgc'
    # Minis / Full-size
    elif 'ENQ' in contract_id:
        return 'enq'
    elif 'EP' in contract_id:
        return 'ep'
    elif 'GCE' in contract_id:
        return 'gce'
    # Legacy fallback for older GC contract IDs
    elif 'GC' in contract_id:
        return 'gc'
    else:
        # Default to last part before expiry
        return contract_id.split('.')[-2].lower()

CONTRACT_SYMBOL = get_symbol_from_contract(TOPSTEPX_CONTRACT_ID)

# Point value based on contract
if 'MNQ' in TOPSTEPX_CONTRACT_ID:
    POINT_VALUE = 2
elif 'MES' in TOPSTEPX_CONTRACT_ID:
    POINT_VALUE = 5
elif 'MGC' in TOPSTEPX_CONTRACT_ID:
    POINT_VALUE = 10
elif 'ENQ' in TOPSTEPX_CONTRACT_ID:
    POINT_VALUE = 20
elif 'EP' in TOPSTEPX_CONTRACT_ID:
    POINT_VALUE = 50
elif 'GC' in TOPSTEPX_CONTRACT_ID:
    POINT_VALUE = 100
else:
    POINT_VALUE = int(os.environ.get('POINT_VALUE', '20'))

# Trading parameters
STOP_LOSS_POINTS = 7
TAKE_PROFIT_POINTS = 35
MIN_TRAINING_DAYS = 5
POLL_INTERVAL_SECONDS = 60  # Poll every 60 seconds to match backtest 1-minute bars
MAX_POSITION_BARS = 20
VALIDATION_DAYS = 3  # Number of days to hold out for tiered validation

# Paths - contract-specific data and models
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
MODELS_DIR = os.path.join(SCRIPT_DIR, '..', 'models', 'baseline_full_year_no_l2')
LOGS_DIR = os.path.join(SCRIPT_DIR, '..', 'logs')
HISTORICAL_BARS_FILE = os.path.join(DATA_DIR, f'bars_1s_{CONTRACT_SYMBOL}.json')

# Create models directory if needed
os.makedirs(MODELS_DIR, exist_ok=True)

# Model files - contract-specific
MODEL_FILES = {
    'stage1_xgb': os.path.join(MODELS_DIR, 'stage1_xgb.json'),
    'timing_xgb': os.path.join(MODELS_DIR, 'timing_xgb.json'),
    'final_xgb': os.path.join(MODELS_DIR, 'final_xgb.json'),
    'longterm_lstm': os.path.join(MODELS_DIR, 'longterm_lstm.pt'),
    'shortterm_lstm': os.path.join(MODELS_DIR, 'shortterm_lstm.pt'),
    'regime_lstm': os.path.join(MODELS_DIR, 'regime_lstm.pt'),
    'metadata': os.path.join(MODELS_DIR, 'metadata.json'),
}

# Global state
session_token = None
token_expiry = 0
cached_bars_1s = []  # All 1s bars (local + API fetched + live)
last_bar_timestamp = None  # Timestamp of last bar in cache
current_position = None
last_trade_time = None
position_bar_count = 0
logger = None

# Session statistics
session_stats = {
    'start_time': None,
    'trades': 0,
    'wins': 0,
    'losses': 0,
    'pnl_points': 0.0,
    'pnl_dollars': 0.0,
    'signals_generated': 0,
    'signals_skipped': 0,
    'last_price': 0.0,
    'loop_count': 0,
}

# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging():
    """Setup file and console logging."""
    global logger
    os.makedirs(LOGS_DIR, exist_ok=True)

    log_file = os.path.join(LOGS_DIR, f'live_trading_{CONTRACT_SYMBOL}_{datetime.utcnow().strftime("%Y%m%d")}.log')
    logger = logging.getLogger('live_trading')
    logger.setLevel(logging.DEBUG)

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s | %(levelname)-5s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(fh)

    # Console handler (INFO only)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(ch)

    return log_file

# =============================================================================
# TOPSTEPX API
# =============================================================================

def authenticate() -> str:
    """Authenticate with TopstepX and get session token."""
    global session_token, token_expiry

    if session_token and time.time() < token_expiry:
        return session_token

    response = requests.post(
        f"{TOPSTEPX_BASE_URL}/api/Auth/loginKey",
        json={"userName": TOPSTEPX_USERNAME, "apiKey": TOPSTEPX_API_KEY},
        headers={"Accept": "text/plain", "Content-Type": "application/json"},
        timeout=30
    )

    data = response.json()
    if not data.get('success') or data.get('errorCode') != 0:
        raise Exception(f"Authentication failed: {data}")

    session_token = data['token']
    token_expiry = time.time() + 23 * 3600
    return session_token

def get_auth_headers() -> Dict[str, str]:
    """Get headers with current auth token."""
    return {
        "Authorization": f"Bearer {authenticate()}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

def fetch_bars_from_api(start_time: datetime, end_time: datetime, chunk_minutes: int = 30, max_retries: int = 3) -> List[Dict]:
    """
    Fetch 1-second bars from TopstepX API in chunks.
    Returns bars in format: {'t': timestamp, 'o': open, 'h': high, 'l': low, 'c': close, 'v': volume}

    Includes retry logic with exponential backoff for rate limiting.
    """
    all_bars = []
    current_start = start_time

    chunk_num = 0
    total_minutes = (end_time - start_time).total_seconds() / 60
    total_chunks = int(total_minutes / chunk_minutes) + 1
    consecutive_failures = 0

    while current_start < end_time:
        chunk_end = min(current_start + timedelta(minutes=chunk_minutes), end_time)
        chunk_num += 1

        payload = {
            "contractId": TOPSTEPX_CONTRACT_ID,
            "live": False,
            "startTime": current_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "endTime": chunk_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "unit": 1,  # Seconds
            "unitNumber": 1,
            "limit": 20000,
            "includePartialBar": True,
        }

        # Retry logic with exponential backoff
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{TOPSTEPX_BASE_URL}/api/History/retrieveBars",
                    json=payload,
                    headers=get_auth_headers(),
                    timeout=60
                )

                data = response.json()
                if data.get('success'):
                    bars = data.get('bars') or []
                    for bar in bars:
                        all_bars.append({
                            't': bar.get('t') or bar.get('timestamp'),
                            'o': float(bar.get('o') or bar.get('open') or 0),
                            'h': float(bar.get('h') or bar.get('high') or 0),
                            'l': float(bar.get('l') or bar.get('low') or 0),
                            'c': float(bar.get('c') or bar.get('close') or 0),
                            'v': int(bar.get('v') or bar.get('volume') or 0),
                        })
                    if logger:
                        logger.debug(f"  API chunk {chunk_num}/{total_chunks}: {len(bars)} bars")
                    consecutive_failures = 0
                    break  # Success, exit retry loop
                else:
                    raise Exception(f"API returned success=false: {data.get('errorMessage', 'unknown')}")

            except Exception as e:
                consecutive_failures += 1
                if attempt < max_retries - 1:
                    # Exponential backoff: 2s, 4s, 8s...
                    backoff = 2 ** (attempt + 1)
                    if logger:
                        logger.warning(f"  API chunk {chunk_num} attempt {attempt+1} failed: {e}. Retrying in {backoff}s...")
                    time.sleep(backoff)
                else:
                    if logger:
                        logger.error(f"  API chunk {chunk_num} failed after {max_retries} attempts: {e}")

        # If too many consecutive failures, take a longer break (rate limiting)
        if consecutive_failures >= 5:
            if logger:
                logger.warning(f"  Rate limiting detected - pausing for 30 seconds...")
            time.sleep(30)
            consecutive_failures = 0

        current_start = chunk_end
        time.sleep(0.5)  # Increased base rate limiting between chunks

    return all_bars

# =============================================================================
# DATA MANAGEMENT
# =============================================================================

def load_local_bars() -> Tuple[List[Dict], str]:
    """
    Load bars from local JSON file.
    Returns (bars_list, last_timestamp)
    """
    if not os.path.exists(HISTORICAL_BARS_FILE):
        if logger:
            logger.warning(f"Local bars file not found: {HISTORICAL_BARS_FILE}")
        return [], None

    with open(HISTORICAL_BARS_FILE, 'r') as f:
        data = json.load(f)

    bars = data.get('bars', [])
    if not bars:
        return [], None

    # Get last timestamp
    last_ts = bars[-1].get('t')

    if logger:
        logger.info(f"Loaded {len(bars):,} bars from local file")
        logger.info(f"  Date range: {bars[0]['t'][:10]} to {bars[-1]['t'][:10]}")

    return bars, last_ts

def save_bars_to_disk(bars: List[Dict]):
    """Save bars back to the local JSON file."""
    if not bars:
        return

    # Deduplicate and sort
    seen = set()
    unique_bars = []
    for bar in bars:
        if bar['t'] not in seen:
            seen.add(bar['t'])
            unique_bars.append(bar)
    unique_bars.sort(key=lambda x: x['t'])

    data = {
        "symbol": "NQZ5",
        "contractId": TOPSTEPX_CONTRACT_ID,
        "unit": 1,
        "unitNumber": 1,
        "startTime": unique_bars[0]['t'],
        "endTime": unique_bars[-1]['t'],
        "barCount": len(unique_bars),
        "bars": unique_bars
    }

    with open(HISTORICAL_BARS_FILE, 'w') as f:
        json.dump(data, f)

    if logger:
        logger.info(f"Saved {len(unique_bars):,} bars to disk")

def update_bars_from_api() -> List[Dict]:
    """
    Load local bars, fetch any missing from API, merge and save.
    Returns complete bar list.
    """
    global cached_bars_1s, last_bar_timestamp

    # Load local bars
    local_bars, last_ts = load_local_bars()

    if not local_bars:
        if logger:
            logger.warning("No local bars found - fetching last 14 days from API")
        start_time = datetime.utcnow() - timedelta(days=14)
    else:
        # Parse last timestamp
        if '+' in last_ts:
            last_ts_clean = last_ts.split('+')[0]
        else:
            last_ts_clean = last_ts.replace('Z', '')
        start_time = datetime.fromisoformat(last_ts_clean) + timedelta(seconds=1)

    end_time = datetime.utcnow()

    # Check if we need to fetch anything
    if start_time >= end_time:
        if logger:
            logger.info("Local data is up to date")
        cached_bars_1s = local_bars
        last_bar_timestamp = local_bars[-1]['t'] if local_bars else None
        return local_bars

    # Fetch missing bars
    minutes_to_fetch = (end_time - start_time).total_seconds() / 60
    if logger:
        logger.info(f"Fetching {minutes_to_fetch:.0f} minutes of data from API...")

    api_bars = fetch_bars_from_api(start_time, end_time)

    if api_bars:
        if logger:
            logger.info(f"Fetched {len(api_bars):,} bars from API")

        # Merge
        all_bars = local_bars + api_bars

        # Deduplicate and sort
        seen = set()
        unique_bars = []
        for bar in all_bars:
            if bar['t'] not in seen:
                seen.add(bar['t'])
                unique_bars.append(bar)
        unique_bars.sort(key=lambda x: x['t'])

        # Save to disk
        save_bars_to_disk(unique_bars)

        cached_bars_1s = unique_bars
        last_bar_timestamp = unique_bars[-1]['t'] if unique_bars else None
        return unique_bars
    else:
        cached_bars_1s = local_bars
        last_bar_timestamp = local_bars[-1]['t'] if local_bars else None
        return local_bars

def append_live_bars(new_bars: List[Dict]):
    """Append new live bars to the cache."""
    global cached_bars_1s, last_bar_timestamp

    if not new_bars:
        return

    # Get timestamps we already have
    existing_ts = set(bar['t'] for bar in cached_bars_1s[-10000:])  # Check last 10k for efficiency

    added = 0
    for bar in new_bars:
        if bar['t'] not in existing_ts:
            cached_bars_1s.append(bar)
            existing_ts.add(bar['t'])
            added += 1

    if added > 0:
        # Keep sorted
        cached_bars_1s.sort(key=lambda x: x['t'])
        last_bar_timestamp = cached_bars_1s[-1]['t']

        if logger:
            logger.debug(f"Added {added} live bars to cache (total: {len(cached_bars_1s):,})")

def fetch_live_update() -> List[Dict]:
    """Fetch any new bars since last cached timestamp."""
    global last_bar_timestamp

    if not last_bar_timestamp:
        return []

    # Parse last timestamp
    if '+' in last_bar_timestamp:
        last_ts_clean = last_bar_timestamp.split('+')[0]
    else:
        last_ts_clean = last_bar_timestamp.replace('Z', '')

    start_time = datetime.fromisoformat(last_ts_clean) + timedelta(seconds=1)
    end_time = datetime.utcnow()

    # Only fetch if there's a gap
    gap_seconds = (end_time - start_time).total_seconds()
    if gap_seconds < 5:
        return []

    # Fetch new bars (small chunk)
    new_bars = fetch_bars_from_api(start_time, end_time, chunk_minutes=5)

    if new_bars:
        append_live_bars(new_bars)

    return new_bars

# =============================================================================
# TRAINING WITH TIERED VALIDATION
# =============================================================================

def get_days_from_bars(bars: List[Dict]) -> Dict[str, List[Dict]]:
    """Group bars by date."""
    days = defaultdict(list)
    for bar in bars:
        date_str = bar['t'][:10]
        days[date_str].append(bar)
    return dict(days)

def train_with_tiered_validation(bars_1s: List[Dict]) -> Tuple[Dict, Dict]:
    """
    Train models with tiered validation.
    Returns (models_dict, validation_results)
    """
    days = get_days_from_bars(bars_1s)
    sorted_dates = sorted(days.keys())

    if len(sorted_dates) < MIN_TRAINING_DAYS + VALIDATION_DAYS:
        raise ValueError(f"Need at least {MIN_TRAINING_DAYS + VALIDATION_DAYS} days, have {len(sorted_dates)}")

    if logger:
        logger.info(f"\n{'='*60}")
        logger.info(f"TRAINING WITH TIERED VALIDATION")
        logger.info(f"{'='*60}")
        logger.info(f"Total days available: {len(sorted_dates)}")
        logger.info(f"Date range: {sorted_dates[0]} to {sorted_dates[-1]}")

    # Split: train on all except last VALIDATION_DAYS
    train_dates = sorted_dates[:-VALIDATION_DAYS]
    val_dates = sorted_dates[-VALIDATION_DAYS:]

    if logger:
        logger.info(f"\nTraining days: {len(train_dates)} ({train_dates[0]} to {train_dates[-1]})")
        logger.info(f"Validation days: {len(val_dates)} ({val_dates[0]} to {val_dates[-1]})")

    # Count training bars
    train_bar_count = sum(len(days[d]) for d in train_dates)

    if logger:
        logger.info(f"\nTraining on {train_bar_count:,} bars...")

    # Train LSTMs - pass days dict and train_dates list (matching backtest interface)
    longterm_lstm, shortterm_lstm, regime_lstm = train_lstm_models(days, train_dates)

    # Train Stage 1 XGBoost (only needs bars and dates)
    stage1_xgb = train_stage1_xgboost(days, train_dates)

    # Train Timing XGBoost (needs bars, dates, and regime_lstm)
    timing_xgb = train_timing_xgboost(days, train_dates, regime_lstm)

    # Build final training data and train Final XGBoost
    # Signature: (all_bars_1s, train_dates, stage1_xgb, timing_xgb, longterm_lstm, shortterm_lstm, regime_lstm)
    final_X, final_y = build_final_training_data(
        days, train_dates, stage1_xgb, timing_xgb,
        longterm_lstm, shortterm_lstm, regime_lstm
    )

    final_xgb = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    final_xgb.fit(final_X, final_y)

    models = {
        'longterm_lstm': longterm_lstm,
        'shortterm_lstm': shortterm_lstm,
        'regime_lstm': regime_lstm,
        'stage1_xgb': stage1_xgb,
        'timing_xgb': timing_xgb,
        'final_xgb': final_xgb,
    }

    # Validate on each holdout day
    if logger:
        logger.info(f"\n{'='*60}")
        logger.info(f"VALIDATION RESULTS (Per Day)")
        logger.info(f"{'='*60}")

    validation_results = {}
    total_pnl = 0.0
    total_trades = 0
    total_wins = 0

    # Track equity curve for drawdown calculation
    equity_curve = [0.0]
    peak_equity = 0.0
    max_drawdown_pts = 0.0

    for val_date in val_dates:
        val_day_bars = days[val_date]

        # Run backtest on this day
        # Signature: run_single_day(bars_1s, target_date, stage1_xgb, timing_xgb, final_xgb,
        #                          longterm_lstm, shortterm_lstm, regime_lstm, starting_equity=0.0)
        # Returns: (trades_list, analysis_dict, cumulative_equity, peak_equity)
        trades_list, analysis, _, _ = run_single_day(
            val_day_bars,
            val_date,
            stage1_xgb, timing_xgb, final_xgb,
            longterm_lstm, shortterm_lstm, regime_lstm,
        )

        # Extract stats from trades list
        day_trades = len(trades_list)
        day_wins = sum(1 for t in trades_list if t.get('pnl_points', 0) > 0)
        day_losses = day_trades - day_wins
        day_pnl_pts = sum(t.get('pnl_points', 0) for t in trades_list)

        # Calculate drawdown for this day
        day_max_dd = 0.0
        for trade in trades_list:
            pnl = trade.get('pnl_points', 0)
            equity_curve.append(equity_curve[-1] + pnl)
            if equity_curve[-1] > peak_equity:
                peak_equity = equity_curve[-1]
            dd = peak_equity - equity_curve[-1]
            if dd > max_drawdown_pts:
                max_drawdown_pts = dd
            if dd > day_max_dd:
                day_max_dd = dd

        result = {
            'trades': day_trades,
            'wins': day_wins,
            'losses': day_losses,
            'pnl_points': day_pnl_pts,
            'max_dd_points': day_max_dd,
        }

        validation_results[val_date] = result
        total_pnl += day_pnl_pts
        total_trades += day_trades
        total_wins += day_wins

        win_rate = (day_wins / day_trades * 100) if day_trades > 0 else 0
        if logger:
            logger.info(f"  {val_date}: {day_trades} trades, {day_wins}W/{day_losses}L "
                       f"({win_rate:.1f}%), P&L: {day_pnl_pts:+.1f} pts (${day_pnl_pts * POINT_VALUE:+.2f}), "
                       f"MaxDD: {day_max_dd:.1f} pts")

    # Summary
    overall_win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0
    if logger:
        logger.info(f"\n  TOTAL: {total_trades} trades, {total_wins}W/{total_trades - total_wins}L "
                   f"({overall_win_rate:.1f}%), P&L: {total_pnl:+.1f} pts (${total_pnl * POINT_VALUE:+.2f})")
        logger.info(f"  Max Drawdown: {max_drawdown_pts:.1f} pts (${max_drawdown_pts * POINT_VALUE:.2f})")

    # Store overall max DD in results
    validation_results['_summary'] = {
        'max_drawdown_points': max_drawdown_pts,
    }

    return models, validation_results

def save_models(models: Dict, validation_results: Dict = None):
    """Save all models to disk."""
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Save XGBoost models
    models['stage1_xgb'].save_model(MODEL_FILES['stage1_xgb'])
    models['timing_xgb'].save_model(MODEL_FILES['timing_xgb'])
    models['final_xgb'].save_model(MODEL_FILES['final_xgb'])

    # Save LSTM models
    torch.save(models['longterm_lstm'].state_dict(), MODEL_FILES['longterm_lstm'])
    torch.save(models['shortterm_lstm'].state_dict(), MODEL_FILES['shortterm_lstm'])
    torch.save(models['regime_lstm'].state_dict(), MODEL_FILES['regime_lstm'])

    # Save metadata with validation results
    days = get_days_from_bars(cached_bars_1s)
    sorted_dates = sorted(days.keys())

    # Calculate validation totals
    val_total_trades = 0
    val_total_wins = 0
    val_total_pnl = 0.0
    val_max_dd = 0.0
    val_days_data = []

    if validation_results:
        # Get overall max drawdown from summary
        if '_summary' in validation_results:
            val_max_dd = validation_results['_summary'].get('max_drawdown_points', 0)

        for date, result in sorted(validation_results.items()):
            if date.startswith('_'):  # Skip summary keys
                continue
            val_total_trades += result.get('trades', 0)
            val_total_wins += result.get('wins', 0)
            val_total_pnl += result.get('pnl_points', 0)
            val_days_data.append({
                'date': date,
                'trades': result.get('trades', 0),
                'wins': result.get('wins', 0),
                'losses': result.get('losses', 0),
                'pnl_points': result.get('pnl_points', 0),
                'max_dd_points': result.get('max_dd_points', 0),
            })

    val_win_rate = (val_total_wins / val_total_trades * 100) if val_total_trades > 0 else 0

    # Split dates into training and validation
    train_dates = sorted_dates[:-VALIDATION_DAYS]
    val_dates = sorted_dates[-VALIDATION_DAYS:]

    metadata = {
        'trained_at': datetime.utcnow().isoformat(),
        'training_days': len(train_dates),
        'validation_days': len(val_dates),
        'train_date_list': train_dates,
        'val_date_list': val_dates,
        'date_range': f"{sorted_dates[0]} to {sorted_dates[-1]}",
        'total_bars': len(cached_bars_1s),
        'validation': {
            'total_trades': val_total_trades,
            'total_wins': val_total_wins,
            'total_losses': val_total_trades - val_total_wins,
            'win_rate': round(val_win_rate, 1),
            'pnl_points': round(val_total_pnl, 1),
            'pnl_dollars': round(val_total_pnl * POINT_VALUE, 2),
            'max_dd_points': round(val_max_dd, 1),
            'max_dd_dollars': round(val_max_dd * POINT_VALUE, 2),
            'days': val_days_data,
        }
    }

    with open(MODEL_FILES['metadata'], 'w') as f:
        json.dump(metadata, f, indent=2)

    if logger:
        logger.info(f"\nModels saved to {MODELS_DIR}")

def load_models() -> Optional[Dict]:
    """Load models from disk if they exist."""
    # Check all files exist
    for name, path in MODEL_FILES.items():
        if not os.path.exists(path):
            return None

    try:
        # Load XGBoost models
        stage1_xgb = xgb.XGBClassifier()
        stage1_xgb.load_model(MODEL_FILES['stage1_xgb'])

        timing_xgb = xgb.XGBClassifier()
        timing_xgb.load_model(MODEL_FILES['timing_xgb'])

        final_xgb = xgb.XGBClassifier()
        final_xgb.load_model(MODEL_FILES['final_xgb'])

        # Load LSTM models - use same parameters as backtest
        longterm_lstm = LongTermLSTM(
            input_dim=7, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, embedding_dim=EMBEDDING_DIM
        ).to(DEVICE)
        longterm_lstm.load_state_dict(torch.load(MODEL_FILES['longterm_lstm'], map_location=DEVICE))
        longterm_lstm.eval()

        shortterm_lstm = ShortTermLSTM(
            input_dim=9, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, embedding_dim=EMBEDDING_DIM
        ).to(DEVICE)
        shortterm_lstm.load_state_dict(torch.load(MODEL_FILES['shortterm_lstm'], map_location=DEVICE))
        shortterm_lstm.eval()

        regime_lstm = RegimeLSTM(
            input_dim=18, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, embedding_dim=EMBEDDING_DIM
        ).to(DEVICE)
        regime_lstm.load_state_dict(torch.load(MODEL_FILES['regime_lstm'], map_location=DEVICE))
        regime_lstm.eval()

        # Load metadata
        with open(MODEL_FILES['metadata'], 'r') as f:
            metadata = json.load(f)

        if logger:
            logger.info(f"Loaded models from disk")
            logger.info(f"  Trained at: {metadata.get('trained_at', 'unknown')}")

            # Show training and validation dates
            train_dates = metadata.get('train_date_list', [])
            val_date_list = metadata.get('val_date_list', [])

            if train_dates:
                logger.info(f"\n  TRAINING DAYS ({len(train_dates)}):")
                logger.info(f"    {', '.join(train_dates)}")
            else:
                logger.info(f"  Training days: {metadata.get('training_days', 'unknown')}")

            if val_date_list:
                logger.info(f"  HOLDOUT/VALIDATION DAYS ({len(val_date_list)}):")
                logger.info(f"    {', '.join(val_date_list)}")

            # Display validation results or run validation if missing
            val = metadata.get('validation', {})
            if val.get('total_trades', 0) > 0 or val.get('days'):
                # Show saved validation results
                logger.info(f"\n{'='*60}")
                logger.info(f"LAST VALIDATION RESULTS")
                logger.info(f"{'='*60}")
                for day in val.get('days', []):
                    win_rate = (day['wins'] / day['trades'] * 100) if day['trades'] > 0 else 0
                    pnl_dollars = day['pnl_points'] * POINT_VALUE
                    dd_pts = day.get('max_dd_points', 0)
                    logger.info(f"  {day['date']}: {day['trades']} trades, "
                               f"{day['wins']}W/{day['losses']}L ({win_rate:.1f}%), "
                               f"P&L: {day['pnl_points']:+.1f} pts (${pnl_dollars:+.0f}), "
                               f"DD: {dd_pts:.1f} pts")
                logger.info(f"  {'-'*56}")
                logger.info(f"  TOTAL: {val['total_trades']} trades, "
                           f"{val['total_wins']}W/{val['total_losses']}L ({val['win_rate']:.1f}%), "
                           f"P&L: {val['pnl_points']:+.1f} pts (${val['pnl_dollars']:+.0f})")
                max_dd_pts = val.get('max_dd_points', 0)
                max_dd_dollars = val.get('max_dd_dollars', max_dd_pts * POINT_VALUE)
                logger.info(f"  Max Drawdown: {max_dd_pts:.1f} pts (${max_dd_dollars:.0f})")
                logger.info(f"{'='*60}")
            else:
                # No validation results saved - just warn (old model format)
                logger.info(f"\n  (No validation results in metadata - will show after next retrain)")

        return {
            'longterm_lstm': longterm_lstm,
            'shortterm_lstm': shortterm_lstm,
            'regime_lstm': regime_lstm,
            'stage1_xgb': stage1_xgb,
            'timing_xgb': timing_xgb,
            'final_xgb': final_xgb,
        }
    except Exception as e:
        if logger:
            logger.error(f"Error loading models: {e}")
        return None

# =============================================================================
# SIGNAL GENERATION (from cached data)
# =============================================================================

def generate_signal(models: Dict) -> Optional[Dict]:
    """
    Generate trading signal using cached 1s bars and trained models.
    Always logs a snapshot of all agent confidences for every decision.
    """
    global cached_bars_1s

    if len(cached_bars_1s) < 7200:  # Need at least 2 hours of data
        return None

    # Use last 6 hours of data for feature calculation
    bars_1s = cached_bars_1s[-21600:]  # 6 hours * 60 min * 60 sec

    # Aggregate to different timeframes (same as backtest)
    bars_1min, cvd_1min, cvd_ema_1min = calculate_cvd_1min(bars_1s)
    bars_5min = aggregate_bars(bars_1s, 5)
    bars_15min = aggregate_bars(bars_1s, 15)
    bars_1h = aggregate_bars(bars_1s, 60)

    if len(bars_1min) < 60 or len(bars_5min) < 72:
        return None

    current_price = bars_1s[-1]['c']
    bar_idx = len(bars_1min) - 1

    # Calculate volume profile (same as backtest - uses 1min bars UP TO current bar with lookback=30)
    vp = calculate_volume_profile(bars_1min[:bar_idx+1], lookback=30)

    # Check for valid trade trigger FIRST (same as backtest)
    trigger = detect_trade_trigger(bars_1min, cvd_1min, cvd_ema_1min, vp, bar_idx)
    if trigger is None:
        return None  # No valid trigger - skip like backtest does

    # Track all confidences for logging
    stage1_confidence = None
    stage1_direction = None
    timing_confidence = None
    final_confidence = None
    direction_str = '-'

    # Extract features exactly as backtest does
    try:
        # Stage 1: Direction prediction
        xgb_features = extract_xgb_features(
            bars_1min, bars_5min, bars_15min, bars_1h,
            cvd_1min, cvd_ema_1min, bar_idx, vp,
            trigger_sentiment=trigger  # Pass trigger like backtest
        )
        if xgb_features is None:
            if logger:
                logger.debug(f"SNAPSHOT | Price: {current_price:.2f} | Stage1: - | Timing: - | Final: - | NO FEATURES")
            return None

        # Convert dict to array (same as backtest)
        X_stage1 = np.array(list(xgb_features.values())).reshape(1, -1)
        stage1_pred = models['stage1_xgb'].predict(X_stage1)[0]
        stage1_proba = models['stage1_xgb'].predict_proba(X_stage1)[0]
        stage1_direction = stage1_pred  # 0 or 1
        stage1_confidence = stage1_proba[stage1_pred]
        direction_str = 'BUY' if stage1_direction == 1 else 'SELL'

        # Backtest DOES filter on stage1 confidence >= 0.6 (see run_single_day line 722)
        stage1_pass = stage1_confidence >= 0.6

        # Calculate footprint candles for shortterm LSTM
        footprint_1min = calculate_footprint_candles(bars_1s)

        # Calculate all technical indicators for regime LSTM (same as backtest)
        atr_1min = calculate_atr(bars_1min)
        bb_middle, bb_upper, bb_lower = calculate_bollinger_bands(bars_1min)
        adx_1min = calculate_adx(bars_1min)
        chop_1min = calculate_choppiness_index(bars_1min)
        rsi_1min = calculate_rsi(bars_1min)

        # Extract LSTM embeddings (matching backtest signatures EXACTLY)
        longterm_seq = extract_longterm_sequence(bars_5min, cvd_1min, bar_idx, seq_len=72)
        shortterm_seq = extract_shortterm_sequence(bars_1s, footprint_1min, len(bars_1s) - 1, seq_len=120)
        regime_seq = extract_regime_sequence(
            bars_1min, cvd_1min, cvd_ema_1min,
            atr_1min, bb_middle, bb_upper, bb_lower,
            adx_1min, chop_1min, rsi_1min,
            bar_idx, vp, seq_len=REGIME_SEQ_LEN
        )

        if longterm_seq is None or shortterm_seq is None or regime_seq is None:
            if logger:
                logger.info(f"SNAPSHOT | Price: {current_price:.2f} | Stage1: {stage1_confidence:.0%} | Timing: - | Final: - | Dir: {direction_str} | NO SEQ")
            return None

        with torch.no_grad():
            longterm_tensor = torch.FloatTensor(longterm_seq).unsqueeze(0).to(DEVICE)
            shortterm_tensor = torch.FloatTensor(shortterm_seq).unsqueeze(0).to(DEVICE)
            regime_tensor = torch.FloatTensor(regime_seq).unsqueeze(0).to(DEVICE)

            longterm_embed = models['longterm_lstm'](longterm_tensor).cpu().numpy().flatten()
            shortterm_embed = models['shortterm_lstm'](shortterm_tensor).cpu().numpy().flatten()
            regime_embed = models['regime_lstm'](regime_tensor).cpu().numpy().flatten()

        # Timing decision
        current_price = bars_1min[-1]['c']
        current_atr = atr_1min[-1] if len(atr_1min) > 0 else 1.0
        current_adx = adx_1min[-1] if len(adx_1min) > 0 else 0
        current_chop = chop_1min[-1] if len(chop_1min) > 0 else 50
        current_rsi = rsi_1min[-1] if len(rsi_1min) > 0 else 50
        current_bb_mid = bb_middle[-1] if len(bb_middle) > 0 else current_price
        current_bb_up = bb_upper[-1] if len(bb_upper) > 0 else current_price
        current_bb_low = bb_lower[-1] if len(bb_lower) > 0 else current_price

        bb_width = (current_bb_up - current_bb_low) / current_price if current_price > 0 else 0
        bb_position = (current_price - current_bb_low) / (current_bb_up - current_bb_low) if current_bb_up > current_bb_low else 0.5
        atr_norm = current_atr / current_price if current_price > 0 else 0

        timing_features = np.concatenate([
            regime_embed,
            np.array([
                atr_norm,
                current_adx / 100.0,
                current_chop / 100.0,
                (current_rsi - 50) / 50.0,
                bb_width,
                bb_position,
            ])
        ])

        # Use predict() for binary decision like backtest (if timing_pred != 1: continue)
        timing_pred_class = models['timing_xgb'].predict(timing_features.reshape(1, -1))[0]
        timing_pred_proba = models['timing_xgb'].predict_proba(timing_features.reshape(1, -1))[0]
        timing_confidence = timing_pred_proba[1]
        timing_pass = timing_pred_class == 1  # Match backtest: if timing_pred != 1: continue

        # Final decision
        xgb_features_arr = np.array(list(xgb_features.values()))
        final_features = np.concatenate([
            xgb_features_arr,
            longterm_embed,
            shortterm_embed,
            regime_embed,
            np.array([stage1_confidence, timing_confidence])
        ])

        # Backtest uses probability threshold >= 0.55 (see run_single_day line 777)
        final_proba = models['final_xgb'].predict_proba(final_features.reshape(1, -1))[0]
        final_confidence = final_proba[1]
        final_pass = final_confidence >= 0.55  # Match backtest threshold

        # Log snapshot of all confidences
        s1_mark = "OK" if stage1_pass else "LOW"
        tm_mark = "OK" if timing_pass else "LOW"
        fn_mark = "OK" if final_pass else "LOW"

        all_pass = stage1_pass and timing_pass and final_pass
        status = "SIGNAL" if all_pass else "NO TRADE"

        if logger:
            logger.info(f"SNAPSHOT | Price: {current_price:.2f} | Stage1: {stage1_confidence:.0%} ({s1_mark}) | Timing: {timing_confidence:.0%} ({tm_mark}) | Final: {final_confidence:.0%} ({fn_mark}) | Dir: {direction_str} | {status}")

        # Only return signal if all thresholds pass
        if not all_pass:
            return None

        # Determine direction and build signal
        direction = 'buy' if stage1_direction == 1 else 'sell'

        # Calculate dynamic SL/TP using swing highs/lows (EXACTLY like backtest - slice up to current bar)
        swing_highs, swing_lows = find_swing_highs_lows(bars_5min[:bar_idx//5+1], lookback=10)

        if direction == 'buy':
            # For longs: SL below recent swing low, TP at 3:1 R:R
            valid_lows = [sl for sl in swing_lows if sl < current_price] if swing_lows else []
            stop_loss = min(valid_lows) - 1 if valid_lows else current_price - 10
            risk = current_price - stop_loss
            take_profit = current_price + (risk * 3)  # 3:1 R:R
        else:
            # For shorts: SL above recent swing high, TP at 3:1 R:R
            valid_highs = [sh for sh in swing_highs if sh > current_price] if swing_highs else []
            stop_loss = max(valid_highs) + 1 if valid_highs else current_price + 10
            risk = stop_loss - current_price
            take_profit = current_price - (risk * 3)  # 3:1 R:R

        return {
            'direction': direction,
            'price': current_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'stage1_confidence': stage1_confidence,
            'timing_confidence': timing_confidence,
            'final_score': final_confidence,
        }

    except Exception as e:
        if logger:
            logger.error(f"Signal generation error: {e}")
        return None

# =============================================================================
# ORDER EXECUTION - Real TopstepX API (same as qwen-topstep agent)
# =============================================================================

# Active position tracking with broker order IDs
active_position_state = {
    'has_position': False,
    'side': None,  # 'long' or 'short'
    'entry_price': None,
    'entry_time': None,
    'entry_order_id': None,
    'stop_order_id': None,
    'tp_order_id': None,
    'contracts': 0,
    'stop_loss': None,
    'take_profit': None,
}

# Position timeout (20 bars = 20 minutes at 60s polling, matches backtest HOLD_BARS=20)
POSITION_TIMEOUT_BARS = 20

def place_order(direction: str, price: float, stop_loss: float, take_profit: float) -> bool:
    """
    Place a market order with OCO bracket (SL/TP) via TopstepX API.
    Tracks order IDs from broker for cancellation on timeout.

    Order Types: 1=Limit, 2=Market, 4=Stop, 5=TrailingStop, 6=JoinBid, 7=JoinAsk
    Side: 0=Buy, 1=Sell
    TimeInForce: 0=IOC, 1=GTC, 3=FOK
    """
    global active_position_state

    try:
        # SAFETY CHECK: Query broker for open orders - if we have SL/TP brackets, we have a position
        try:
            # First check positions API
            pos_response = requests.get(
                f"{TOPSTEPX_BASE_URL}/api/Account/{TOPSTEPX_ACCOUNT_ID}/positions",
                headers=get_auth_headers(),
                timeout=10
            )
            if pos_response.ok:
                positions = pos_response.json()
                if positions:
                    for pos in positions:
                        if pos.get('contractId') == TOPSTEPX_CONTRACT_ID:
                            size = pos.get('netSize') or pos.get('size') or 0
                            if abs(size) > 0:
                                if logger:
                                    logger.warning(f"BLOCKED: Already have position ({size} contracts) - not placing new order")
                                return False

            # ALSO check for open bracket orders as backup (positions API can return 404 even with orders)
            orders_response = requests.post(
                f"{TOPSTEPX_BASE_URL}/api/Order/searchOpen",
                json={"accountId": TOPSTEPX_ACCOUNT_ID},
                headers=get_auth_headers(),
                timeout=10
            )
            if orders_response.ok:
                data = orders_response.json()
                orders = data.get('orders', [])
                # Count SL orders (type=4) for our contract - each SL means an open position
                sl_orders = [o for o in orders if o.get('contractId') == TOPSTEPX_CONTRACT_ID and o.get('type') == 4]
                if sl_orders:
                    if logger:
                        logger.warning(f"BLOCKED: Found {len(sl_orders)} open SL order(s) - position already exists")
                    return False
        except Exception as e:
            if logger:
                logger.warning(f"Position safety check failed: {e} - proceeding cautiously")

        # Determine side
        side = 0 if direction == 'buy' else 1  # 0=Buy, 1=Sell
        opposite_side = 1 if direction == 'buy' else 0

        # Calculate bracket ticks - tick size varies by contract
        if 'GC' in TOPSTEPX_CONTRACT_ID or 'MGC' in TOPSTEPX_CONTRACT_ID:
            tick_size = 0.10  # Gold tick size
        else:
            tick_size = 0.25  # NQ/ES tick size

        # TopstepX bracket ticks are signed relative to direction:
        # BUY (long): SL ticks negative (below entry), TP ticks positive (above entry)
        # SELL (short): SL ticks positive (above entry), TP ticks negative (below entry)
        sl_ticks_abs = int(abs(price - stop_loss) / tick_size)
        tp_ticks_abs = int(abs(take_profit - price) / tick_size)

        # CLAMP to broker max of 1000 ticks
        MAX_TICKS = 1000
        if sl_ticks_abs > MAX_TICKS:
            if logger:
                logger.warning(f"SL ticks {sl_ticks_abs} exceeds max {MAX_TICKS}, clamping")
            sl_ticks_abs = MAX_TICKS
            stop_loss = price - (sl_ticks_abs * tick_size) if direction == 'buy' else price + (sl_ticks_abs * tick_size)
        if tp_ticks_abs > MAX_TICKS:
            if logger:
                logger.warning(f"TP ticks {tp_ticks_abs} exceeds max {MAX_TICKS}, clamping")
            tp_ticks_abs = MAX_TICKS
            take_profit = price + (tp_ticks_abs * tick_size) if direction == 'buy' else price - (tp_ticks_abs * tick_size)

        if direction == 'buy':
            sl_ticks = -sl_ticks_abs  # Stop below entry = negative
            tp_ticks = tp_ticks_abs   # Target above entry = positive
        else:
            sl_ticks = sl_ticks_abs   # Stop above entry = positive
            tp_ticks = -tp_ticks_abs  # Target below entry = negative

        if logger:
            logger.info(f"=" * 50)
            logger.info(f"PLACING {direction.upper()} ORDER @ MARKET")
            logger.info(f"  Contract: {TOPSTEPX_CONTRACT_ID}")
            logger.info(f"  Account: {TOPSTEPX_ACCOUNT_ID}")
            logger.info(f"  Entry: ~{price:.2f}")
            logger.info(f"  SL: {stop_loss:.2f} ({sl_ticks} ticks)")
            logger.info(f"  TP: {take_profit:.2f} ({tp_ticks} ticks)")

        # Build order payload with native OCO brackets
        payload = {
            "accountId": TOPSTEPX_ACCOUNT_ID,
            "contractId": TOPSTEPX_CONTRACT_ID,
            "side": side,
            "size": 1,
            "type": 2,  # Market order
            "timeInForce": 1,  # GTC
            "stopLossBracket": {
                "ticks": sl_ticks,
                "type": 4  # Stop order
            },
            "takeProfitBracket": {
                "ticks": tp_ticks,
                "type": 1  # Limit order
            }
        }

        # Send order to TopstepX
        response = requests.post(
            f"{TOPSTEPX_BASE_URL}/api/Order/place",
            json=payload,
            headers=get_auth_headers(),
            timeout=30
        )

        data = response.json()

        if data.get('success'):
            entry_order_id = data.get('orderId') or data.get('id')

            # Update position state with broker IDs
            active_position_state.update({
                'has_position': True,
                'side': 'long' if direction == 'buy' else 'short',
                'entry_price': price,
                'entry_time': datetime.utcnow(),
                'entry_order_id': entry_order_id,
                'contracts': 1,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'bars_held': 0,
            })

            if logger:
                logger.info(f"ORDER PLACED SUCCESSFULLY")
                logger.info(f"  Entry Order ID: {entry_order_id}")
                logger.info(f"  Response: {json.dumps(data)}")
                logger.info(f"=" * 50)

            # Now fetch open orders to get bracket order IDs
            fetch_bracket_order_ids()

            return True
        else:
            error_msg = data.get('errorMessage') or data.get('message') or str(data)
            if logger:
                logger.error(f"ORDER FAILED: {error_msg}")
                logger.error(f"  Full response: {json.dumps(data)}")
            return False

    except Exception as e:
        if logger:
            logger.error(f"ORDER ERROR: {e}")
        return False

def fetch_bracket_order_ids():
    """Fetch SL/TP bracket order IDs from broker after entry."""
    global active_position_state

    try:
        response = requests.post(
            f"{TOPSTEPX_BASE_URL}/api/Order/searchOpen",
            json={"accountId": TOPSTEPX_ACCOUNT_ID},
            headers=get_auth_headers(),
            timeout=30
        )

        if not response.ok:
            return

        data = response.json()
        orders = data.get('orders', [])

        for order in orders:
            if order.get('contractId') == TOPSTEPX_CONTRACT_ID:
                order_type = order.get('type')
                order_id = order.get('id')

                # Type 4 = Stop (our SL), Type 1 = Limit (our TP)
                if order_type == 4:
                    active_position_state['stop_order_id'] = order_id
                    if logger:
                        logger.info(f"  Stop Loss Order ID: {order_id}")
                elif order_type == 1:
                    active_position_state['tp_order_id'] = order_id
                    if logger:
                        logger.info(f"  Take Profit Order ID: {order_id}")

    except Exception as e:
        if logger:
            logger.warning(f"Failed to fetch bracket order IDs: {e}")

def check_position() -> bool:
    """Check if we have an open position via TopstepX API or open SL orders."""
    global current_position, active_position_state

    try:
        # First check positions API
        response = requests.get(
            f"{TOPSTEPX_BASE_URL}/api/Account/{TOPSTEPX_ACCOUNT_ID}/positions",
            headers=get_auth_headers(),
            timeout=30
        )

        if response.ok:
            positions = response.json()
            if positions and len(positions) > 0:
                for pos in positions:
                    if pos.get('contractId') == TOPSTEPX_CONTRACT_ID:
                        size = pos.get('netSize') or pos.get('size') or 0
                        if abs(size) > 0:
                            avg_price = pos.get('averagePrice', 0)
                            if logger:
                                logger.debug(f"Open position: {size} contracts @ {avg_price}")
                            return True

        # ALSO check for open SL orders (positions API can return 404 even with orders)
        try:
            orders_response = requests.post(
                f"{TOPSTEPX_BASE_URL}/api/Order/searchOpen",
                json={"accountId": TOPSTEPX_ACCOUNT_ID},
                headers=get_auth_headers(),
                timeout=10
            )
            if orders_response.ok:
                data = orders_response.json()
                orders = data.get('orders', [])
                sl_orders = [o for o in orders if o.get('contractId') == TOPSTEPX_CONTRACT_ID and o.get('type') == 4]
                if sl_orders:
                    if logger:
                        logger.debug(f"Found {len(sl_orders)} open SL order(s) - position exists")
                    return True
        except Exception as e:
            if logger:
                logger.warning(f"SL order check failed: {e}")

        # No position found but we thought we had one
        if active_position_state['has_position']:
            if logger:
                logger.info("Position closed by broker (SL/TP hit)")
            reset_position_state()

        current_position = None
        return False

    except Exception as e:
        if logger:
            logger.warning(f"Position check error: {e}")
        return active_position_state['has_position']

def reset_position_state():
    """Reset position tracking state."""
    global active_position_state
    active_position_state = {
        'has_position': False,
        'side': None,
        'entry_price': None,
        'entry_time': None,
        'entry_order_id': None,
        'stop_order_id': None,
        'tp_order_id': None,
        'contracts': 0,
        'stop_loss': None,
        'take_profit': None,
        'bars_held': 0,
    }

def check_position_timeout() -> bool:
    """Check if position has exceeded timeout and should be closed."""
    global active_position_state

    if not active_position_state['has_position']:
        return False

    bars_held = active_position_state.get('bars_held', 0)
    active_position_state['bars_held'] = bars_held + 1

    if bars_held >= POSITION_TIMEOUT_BARS:
        if logger:
            logger.info(f"Position timeout ({bars_held} bars) - closing position")
        close_position_with_market_order()
        return True

    return False

def close_position_with_market_order():
    """Close position with market order and cancel brackets."""
    global active_position_state

    if not active_position_state['has_position']:
        return

    try:
        # First cancel any open bracket orders
        cancel_bracket_orders()

        # Determine close side (opposite of entry)
        close_side = 1 if active_position_state['side'] == 'long' else 0  # Sell to close long, Buy to close short

        # Place market order to close
        payload = {
            "accountId": TOPSTEPX_ACCOUNT_ID,
            "contractId": TOPSTEPX_CONTRACT_ID,
            "side": close_side,
            "size": active_position_state['contracts'],
            "type": 2,  # Market
            "timeInForce": 1,  # GTC
        }

        response = requests.post(
            f"{TOPSTEPX_BASE_URL}/api/Order/place",
            json=payload,
            headers=get_auth_headers(),
            timeout=30
        )

        data = response.json()

        if data.get('success'):
            if logger:
                logger.info(f"POSITION CLOSED: Market {'SELL' if close_side == 1 else 'BUY'} order placed")
                logger.info(f"  Order ID: {data.get('orderId') or data.get('id')}")
            reset_position_state()
        else:
            if logger:
                logger.error(f"Failed to close position: {data}")

    except Exception as e:
        if logger:
            logger.error(f"Close position error: {e}")

def cancel_bracket_orders():
    """Cancel SL and TP bracket orders using stored IDs."""
    global active_position_state

    orders_to_cancel = []
    if active_position_state.get('stop_order_id'):
        orders_to_cancel.append(('Stop Loss', active_position_state['stop_order_id']))
    if active_position_state.get('tp_order_id'):
        orders_to_cancel.append(('Take Profit', active_position_state['tp_order_id']))

    for order_name, order_id in orders_to_cancel:
        try:
            response = requests.post(
                f"{TOPSTEPX_BASE_URL}/api/Order/cancel",
                json={"accountId": TOPSTEPX_ACCOUNT_ID, "orderId": str(order_id)},
                headers=get_auth_headers(),
                timeout=30
            )
            if logger:
                status = "OK" if response.ok else f"FAILED ({response.status_code})"
                logger.info(f"  Cancelled {order_name} order #{order_id}: {status}")
        except Exception as e:
            if logger:
                logger.warning(f"  Failed to cancel {order_name} #{order_id}: {e}")

def cancel_all_orders():
    """Cancel all open orders for this contract."""
    try:
        # First get open orders
        response = requests.post(
            f"{TOPSTEPX_BASE_URL}/api/Order/searchOpen",
            json={"accountId": TOPSTEPX_ACCOUNT_ID},
            headers=get_auth_headers(),
            timeout=30
        )

        if not response.ok:
            return

        data = response.json()
        orders = data.get('orders', [])

        for order in orders:
            if order.get('contractId') == TOPSTEPX_CONTRACT_ID:
                order_id = order.get('id')
                cancel_response = requests.post(
                    f"{TOPSTEPX_BASE_URL}/api/Order/cancel",
                    json={"accountId": TOPSTEPX_ACCOUNT_ID, "orderId": str(order_id)},
                    headers=get_auth_headers(),
                    timeout=30
                )
                if logger:
                    logger.info(f"Cancelled order #{order_id}: {cancel_response.ok}")

    except Exception as e:
        if logger:
            logger.warning(f"Cancel orders error: {e}")

# =============================================================================
# MAIN TRADING LOOP
# =============================================================================

def trading_loop(models: Dict):
    """Main trading loop using cached data."""
    global session_stats, current_position, last_trade_time

    session_stats['loop_count'] += 1

    # Fetch any new bars from API
    fetch_live_update()

    if len(cached_bars_1s) < 7200:
        if logger:
            logger.warning(f"Insufficient cached data: {len(cached_bars_1s)} bars")
        return

    current_price = cached_bars_1s[-1]['c']
    session_stats['last_price'] = current_price

    # Check if we have a position
    if check_position():
        # Check for position timeout (close after POSITION_TIMEOUT_BARS bars)
        check_position_timeout()
        return

    # Check minimum time between trades
    if last_trade_time:
        elapsed = (datetime.utcnow() - last_trade_time).total_seconds()
        if elapsed < MIN_BARS_BETWEEN_TRADES * 60:
            return

    # Generate signal
    signal = generate_signal(models)

    if signal:
        session_stats['signals_generated'] += 1

        if logger:
            logger.info(f"SIGNAL: {signal['direction'].upper()} @ {signal['price']:.2f}")
            logger.info(f"  Stage1: {signal['stage1_confidence']:.1%} | Timing: {signal['timing_confidence']:.1%} | Final: {signal['final_score']:.1%}")

        # Place order
        if place_order(signal['direction'], signal['price'], signal['stop_loss'], signal['take_profit']):
            current_position = signal
            last_trade_time = datetime.utcnow()
            session_stats['trades'] += 1

def print_dashboard():
    """Print live dashboard."""
    os.system('clear' if os.name != 'nt' else 'cls')

    uptime_mins = 0
    if session_stats['start_time']:
        uptime_mins = (datetime.utcnow() - session_stats['start_time']).total_seconds() / 60

    print("=" * 60)
    print("     NO-WHALE REGIME LIVE TRADING")
    print("=" * 60)
    print(f"  Contract: {TOPSTEPX_CONTRACT_ID}  |  Point Value: ${POINT_VALUE}")
    print(f"  Cached bars: {len(cached_bars_1s):,}  |  Uptime: {int(uptime_mins)}m")
    print("-" * 60)
    print(f"  LAST PRICE: {session_stats['last_price']:.2f}")
    print("-" * 60)
    print(f"  Trades: {session_stats['trades']}  |  Signals: {session_stats['signals_generated']}")
    print("=" * 60)

# =============================================================================
# MAIN
# =============================================================================

def main():
    global session_stats

    print("\n" + "=" * 60)
    print("     NO-WHALE REGIME LIVE TRADING")
    print("=" * 60)

    # Setup logging
    log_file = setup_logging()
    logger.info("=" * 50)
    logger.info("Starting No-Whale Regime Live Trading")
    logger.info("=" * 50)

    # Validate config
    if not TOPSTEPX_ACCOUNT_ID:
        logger.error("TOPSTEPX_ACCOUNT_ID not set")
        return

    logger.info(f"Account ID: {TOPSTEPX_ACCOUNT_ID}")
    logger.info(f"Contract: {TOPSTEPX_CONTRACT_ID}")
    logger.info(f"Point Value: ${POINT_VALUE}")

    # Authenticate
    logger.info("\nAuthenticating with TopstepX...")
    try:
        authenticate()
        logger.info("  Authentication successful")
    except Exception as e:
        logger.error(f"  Authentication failed: {e}")
        return

    # Load and update data
    logger.info("\nLoading data...")
    bars = update_bars_from_api()

    if len(bars) < MIN_TRAINING_DAYS * 20000:  # Rough estimate
        logger.error(f"Insufficient data for training: {len(bars)} bars")
        return

    # Check if we need to train
    models = load_models()

    if models is None:
        logger.info("\nTraining new models...")
        try:
            models, val_results = train_with_tiered_validation(bars)
            save_models(models, val_results)
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return
    else:
        # Check if models are stale (> 24 hours old)
        try:
            with open(MODEL_FILES['metadata'], 'r') as f:
                metadata = json.load(f)
            trained_at = datetime.fromisoformat(metadata['trained_at'])
            age_hours = (datetime.utcnow() - trained_at).total_seconds() / 3600

            if age_hours > 24:
                logger.info(f"\nModels are {age_hours:.1f} hours old - retraining...")
                models, val_results = train_with_tiered_validation(bars)
                save_models(models, val_results)
        except:
            pass

    # Fetch any bars we missed during training
    fetch_live_update()

    # Start trading loop
    session_stats['start_time'] = datetime.utcnow()
    logger.info(f"\nStarting trading loop (every {POLL_INTERVAL_SECONDS}s)...")
    logger.info("Press Ctrl+C to stop\n")

    running = True

    def signal_handler(sig, frame):
        nonlocal running
        running = False
        logger.info("\nShutdown requested...")

    signal.signal(signal.SIGINT, signal_handler)

    while running:
        try:
            # Wait until second 1-2 of new minute (after candle close at :00)
            now = datetime.utcnow()
            if now.second < 1:
                # We're at :00, wait 1 second for candle to fully close
                time.sleep(1)
            elif now.second > 2:
                # Wait until next minute's :01
                seconds_until_next_bar = (60 - now.second) + 1
                time.sleep(seconds_until_next_bar)

            trading_loop(models)
            print_dashboard()

            # Wait until next minute's :01 (candle close + 1 second buffer)
            now = datetime.utcnow()
            seconds_until_next = (60 - now.second) + 1
            if seconds_until_next > 60:
                seconds_until_next = 1
            time.sleep(max(1, seconds_until_next))
        except Exception as e:
            logger.error(f"Loop error: {e}")
            time.sleep(5)

    # Save final data
    logger.info("\nSaving data to disk...")
    save_bars_to_disk(cached_bars_1s)

    logger.info("=" * 50)
    logger.info("SESSION ENDED")
    logger.info(f"Total trades: {session_stats['trades']}")
    logger.info("=" * 50)

if __name__ == "__main__":
    main()
