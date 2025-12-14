#!/usr/bin/env python3
"""
No-Whale Regime Live Trading - MNQ Version

STRATEGY: Analyzes NQ data for decisions, executes on MNQ contracts
- Uses full-year baseline NQ models for signal generation
- Fetches NQ 1s bars for analysis
- Executes trades on MNQ contract
- Supports 1-5 contract sizes via CONTRACT_SIZE env var
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

# Load .env file
from dotenv import load_dotenv
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '.env')
load_dotenv(env_path, override=False)

import numpy as np
import requests

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import backtest functions
from no_whale_regime_backtest import (
    LongTermLSTM, ShortTermLSTM, RegimeLSTM,
    extract_longterm_sequence, extract_shortterm_sequence, extract_regime_sequence,
    extract_xgb_features,
    calculate_atr, calculate_bollinger_bands, calculate_adx, calculate_choppiness_index, calculate_rsi,
    calculate_cvd_1min, calculate_volume_profile, calculate_footprint_candles,
    aggregate_bars, find_swing_highs_lows, detect_trade_trigger,
    LONGTERM_SEQ_LEN, SHORTTERM_SEQ_LEN, REGIME_SEQ_LEN, EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS,
    MIN_BARS_BETWEEN_TRADES, HOLD_BARS, DEVICE,
)

import torch
import xgboost as xgb

# =============================================================================
# CONFIGURATION
# =============================================================================

# TopstepX credentials
TOPSTEPX_API_KEY = os.environ.get('TOPSTEPX_API_KEY', '')
TOPSTEPX_USERNAME = os.environ.get('TOPSTEPX_USERNAME', '')
TOPSTEPX_BASE_URL = os.environ.get('TOPSTEPX_BASE_URL', 'https://api.topstepx.com')
TOPSTEPX_ACCOUNT_ID = int(os.environ.get('TOPSTEPX_ACCOUNT_ID', '0'))

# MNQ contract for execution
TOPSTEPX_MNQ_CONTRACT_ID = os.environ.get('TOPSTEPX_CONTRACT_ID', 'CON.F.US.MNQ.Z25')

# NQ contract for data/analysis (ALWAYS use NQ for decisions)
NQ_DATA_CONTRACT_ID = 'CON.F.US.ENQ.Z25'

# Contract size (1-5)
CONTRACT_SIZE = int(os.environ.get('CONTRACT_SIZE', '1'))

# Point values
POINT_VALUE_MNQ = 2  # MNQ = $2/pt
POINT_VALUE_NQ = 20  # NQ = $20/pt (used for data fetching)

# Trading parameters
POLL_INTERVAL_SECONDS = 60
POSITION_TIMEOUT_BARS = 20

# Paths - use NQ models (full-year baseline)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
MODELS_DIR = os.path.join(SCRIPT_DIR, '..', 'models', 'baseline_full_year_no_l2')
LOGS_DIR = os.path.join(SCRIPT_DIR, '..', 'logs')

# Data file - ALWAYS NQ for analysis
HISTORICAL_BARS_FILE = os.path.join(DATA_DIR, 'bars_1s_nq.json')

# Model files
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
cached_bars_1s = []
last_bar_timestamp = None
logger = None

# Active position tracking
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

# Session statistics
session_stats = {
    'start_time': None,
    'trades': 0,
    'signals_generated': 0,
    'last_price': 0.0,
}

# =============================================================================
# LOGGING
# =============================================================================

def setup_logging():
    """Setup file and console logging."""
    global logger
    os.makedirs(LOGS_DIR, exist_ok=True)

    log_file = os.path.join(LOGS_DIR, f'no_whale_regime_live_full_year_mnq_{CONTRACT_SIZE}x_{datetime.utcnow().strftime("%Y%m%d")}.log')
    logger = logging.getLogger('mnq_live_trading')
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s | %(levelname)-5s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(fh)

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

def fetch_bars_from_api(contract_id: str, start_time: datetime, end_time: datetime, chunk_minutes: int = 30) -> List[Dict]:
    """Fetch 1-second bars from TopstepX API."""
    all_bars = []
    current_start = start_time

    while current_start < end_time:
        chunk_end = min(current_start + timedelta(minutes=chunk_minutes), end_time)

        payload = {
            "contractId": contract_id,
            "live": False,
            "startTime": current_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "endTime": chunk_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "unit": 1,
            "unitNumber": 1,
            "limit": 20000,
            "includePartialBar": True,
        }

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
        except Exception as e:
            if logger:
                logger.warning(f"API fetch error: {e}")

        current_start = chunk_end
        time.sleep(0.5)

    return all_bars

# =============================================================================
# DATA MANAGEMENT
# =============================================================================

def load_local_nq_bars() -> Tuple[List[Dict], str]:
    """Load NQ bars from local file."""
    if not os.path.exists(HISTORICAL_BARS_FILE):
        return [], None

    with open(HISTORICAL_BARS_FILE, 'r') as f:
        data = json.load(f)

    bars = data.get('bars', [])
    if not bars:
        return [], None

    last_ts = bars[-1].get('t')

    if logger:
        logger.info(f"Loaded {len(bars):,} NQ bars from local file")
        logger.info(f"  Date range: {bars[0]['t'][:10]} to {bars[-1]['t'][:10]}")

    return bars, last_ts

def save_nq_bars_to_disk(bars: List[Dict]):
    """Save NQ bars to disk."""
    if not bars:
        return

    seen = set()
    unique_bars = []
    for bar in bars:
        if bar['t'] not in seen:
            seen.add(bar['t'])
            unique_bars.append(bar)
    unique_bars.sort(key=lambda x: x['t'])

    data = {
        "symbol": "NQ",
        "contractId": NQ_DATA_CONTRACT_ID,
        "unit": 1,
        "unitNumber": 1,
        "startTime": unique_bars[0]['t'],
        "endTime": unique_bars[-1]['t'],
        "barCount": len(unique_bars),
        "bars": unique_bars
    }

    with open(HISTORICAL_BARS_FILE, 'w') as f:
        json.dump(data, f)

def update_nq_bars_from_api() -> List[Dict]:
    """Load local NQ bars, fetch any missing, merge and save."""
    global cached_bars_1s, last_bar_timestamp

    local_bars, last_ts = load_local_nq_bars()

    if not local_bars:
        if logger:
            logger.warning("No local NQ bars - fetching last 14 days")
        start_time = datetime.utcnow() - timedelta(days=14)
    else:
        if '+' in last_ts:
            last_ts_clean = last_ts.split('+')[0]
        else:
            last_ts_clean = last_ts.replace('Z', '')
        start_time = datetime.fromisoformat(last_ts_clean) + timedelta(seconds=1)

    end_time = datetime.utcnow()

    if start_time >= end_time:
        cached_bars_1s = local_bars
        last_bar_timestamp = local_bars[-1]['t'] if local_bars else None
        return local_bars

    if logger:
        logger.info(f"Fetching NQ data from API...")

    # Fetch NQ data (not MNQ!)
    api_bars = fetch_bars_from_api(NQ_DATA_CONTRACT_ID, start_time, end_time)

    if api_bars:
        all_bars = local_bars + api_bars

        seen = set()
        unique_bars = []
        for bar in all_bars:
            if bar['t'] not in seen:
                seen.add(bar['t'])
                unique_bars.append(bar)
        unique_bars.sort(key=lambda x: x['t'])

        save_nq_bars_to_disk(unique_bars)

        cached_bars_1s = unique_bars
        last_bar_timestamp = unique_bars[-1]['t'] if unique_bars else None
        return unique_bars
    else:
        cached_bars_1s = local_bars
        last_bar_timestamp = local_bars[-1]['t'] if local_bars else None
        return local_bars

def fetch_live_nq_update() -> List[Dict]:
    """Fetch any new NQ bars since last cached timestamp."""
    global last_bar_timestamp, cached_bars_1s

    if not last_bar_timestamp:
        return []

    if '+' in last_bar_timestamp:
        last_ts_clean = last_bar_timestamp.split('+')[0]
    else:
        last_ts_clean = last_bar_timestamp.replace('Z', '')

    start_time = datetime.fromisoformat(last_ts_clean) + timedelta(seconds=1)
    end_time = datetime.utcnow()

    gap_seconds = (end_time - start_time).total_seconds()
    if gap_seconds < 5:
        return []

    new_bars = fetch_bars_from_api(NQ_DATA_CONTRACT_ID, start_time, end_time, chunk_minutes=5)

    if new_bars:
        existing_ts = set(bar['t'] for bar in cached_bars_1s[-10000:])
        added = 0
        for bar in new_bars:
            if bar['t'] not in existing_ts:
                cached_bars_1s.append(bar)
                existing_ts.add(bar['t'])
                added += 1

        if added > 0:
            cached_bars_1s.sort(key=lambda x: x['t'])
            last_bar_timestamp = cached_bars_1s[-1]['t']

    return new_bars

# =============================================================================
# MODEL LOADING
# =============================================================================

def load_models() -> Optional[Dict]:
    """Load pre-trained NQ models from disk."""
    for name, path in MODEL_FILES.items():
        if not os.path.exists(path):
            return None

    try:
        stage1_xgb = xgb.XGBClassifier()
        stage1_xgb.load_model(MODEL_FILES['stage1_xgb'])

        timing_xgb = xgb.XGBClassifier()
        timing_xgb.load_model(MODEL_FILES['timing_xgb'])

        final_xgb = xgb.XGBClassifier()
        final_xgb.load_model(MODEL_FILES['final_xgb'])

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

        with open(MODEL_FILES['metadata'], 'r') as f:
            metadata = json.load(f)

        if logger:
            logger.info(f"Loaded NQ models from {MODELS_DIR}")
            logger.info(f"  Trained: {metadata.get('trained_at', 'unknown')}")
            val = metadata.get('validation', {})
            if val.get('total_trades', 0) > 0:
                logger.info(f"  Validation: {val['total_trades']} trades, {val['win_rate']:.1f}% WR, {val['pnl_points']:+.1f} pts")

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
# SIGNAL GENERATION (uses NQ data)
# =============================================================================

def generate_signal(models: Dict) -> Optional[Dict]:
    """Generate signal using NQ data analysis."""
    global cached_bars_1s

    if len(cached_bars_1s) < 7200:
        return None

    bars_1s = cached_bars_1s[-21600:]

    bars_1min, cvd_1min, cvd_ema_1min = calculate_cvd_1min(bars_1s)
    bars_5min = aggregate_bars(bars_1s, 5)
    bars_15min = aggregate_bars(bars_1s, 15)
    bars_1h = aggregate_bars(bars_1s, 60)

    if len(bars_1min) < 60 or len(bars_5min) < 72:
        return None

    current_price = bars_1s[-1]['c']
    bar_idx = len(bars_1min) - 1

    vp = calculate_volume_profile(bars_1min[:bar_idx+1], lookback=30)
    trigger = detect_trade_trigger(bars_1min, cvd_1min, cvd_ema_1min, vp, bar_idx)

    if trigger is None:
        return None

    try:
        xgb_features = extract_xgb_features(
            bars_1min, bars_5min, bars_15min, bars_1h,
            cvd_1min, cvd_ema_1min, bar_idx, vp,
            trigger_sentiment=trigger
        )
        if xgb_features is None:
            return None

        X_stage1 = np.array(list(xgb_features.values())).reshape(1, -1)
        stage1_pred = models['stage1_xgb'].predict(X_stage1)[0]
        stage1_proba = models['stage1_xgb'].predict_proba(X_stage1)[0]
        stage1_confidence = stage1_proba[stage1_pred]

        if stage1_confidence < 0.6:
            return None

        footprint_1min = calculate_footprint_candles(bars_1s)
        atr_1min = calculate_atr(bars_1min)
        bb_middle, bb_upper, bb_lower = calculate_bollinger_bands(bars_1min)
        adx_1min = calculate_adx(bars_1min)
        chop_1min = calculate_choppiness_index(bars_1min)
        rsi_1min = calculate_rsi(bars_1min)

        longterm_seq = extract_longterm_sequence(bars_5min, cvd_1min, bar_idx, seq_len=72)
        shortterm_seq = extract_shortterm_sequence(bars_1s, footprint_1min, len(bars_1s) - 1, seq_len=120)
        regime_seq = extract_regime_sequence(
            bars_1min, cvd_1min, cvd_ema_1min,
            atr_1min, bb_middle, bb_upper, bb_lower,
            adx_1min, chop_1min, rsi_1min,
            bar_idx, vp, seq_len=REGIME_SEQ_LEN
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
            np.array([atr_norm, current_adx / 100.0, current_chop / 100.0,
                     (current_rsi - 50) / 50.0, bb_width, bb_position])
        ])

        timing_pred = models['timing_xgb'].predict(timing_features.reshape(1, -1))[0]
        timing_proba = models['timing_xgb'].predict_proba(timing_features.reshape(1, -1))[0]
        timing_confidence = timing_proba[1]

        if timing_pred != 1:
            return None

        final_features = np.concatenate([
            np.array(list(xgb_features.values())),
            longterm_embed, shortterm_embed, regime_embed,
            np.array([stage1_confidence, timing_confidence])
        ])

        final_proba = models['final_xgb'].predict_proba(final_features.reshape(1, -1))[0]
        final_confidence = final_proba[1]

        if final_confidence < 0.55:
            return None

        direction = 'buy' if stage1_pred == 1 else 'sell'

        swing_highs, swing_lows = find_swing_highs_lows(bars_5min[:bar_idx//5+1], lookback=10)

        if direction == 'buy':
            valid_lows = [sl for sl in swing_lows if sl < current_price] if swing_lows else []
            stop_loss = min(valid_lows) - 1 if valid_lows else current_price - 10
            risk = current_price - stop_loss
            take_profit = current_price + (risk * 3)
        else:
            valid_highs = [sh for sh in swing_highs if sh > current_price] if swing_highs else []
            stop_loss = max(valid_highs) + 1 if valid_highs else current_price + 10
            risk = stop_loss - current_price
            take_profit = current_price - (risk * 3)

        if logger:
            logger.info(f"SIGNAL: {direction.upper()} @ {current_price:.2f}")
            logger.info(f"  Stage1: {stage1_confidence:.0%} | Timing: {timing_confidence:.0%} | Final: {final_confidence:.0%}")

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
# ORDER EXECUTION (MNQ contract)
# =============================================================================

def place_mnq_order(direction: str, price: float, stop_loss: float, take_profit: float) -> bool:
    """Place MNQ order with OCO bracket."""
    global active_position_state

    try:
        # Check for existing position
        try:
            orders_response = requests.post(
                f"{TOPSTEPX_BASE_URL}/api/Order/searchOpen",
                json={"accountId": TOPSTEPX_ACCOUNT_ID},
                headers=get_auth_headers(),
                timeout=10
            )
            if orders_response.ok:
                orders = orders_response.json().get('orders', [])
                sl_orders = [o for o in orders if o.get('contractId') == TOPSTEPX_MNQ_CONTRACT_ID and o.get('type') == 4]
                if sl_orders:
                    if logger:
                        logger.warning(f"BLOCKED: Found {len(sl_orders)} open MNQ SL order(s)")
                    return False
        except:
            pass

        side = 0 if direction == 'buy' else 1
        tick_size = 0.25

        sl_ticks_abs = int(abs(price - stop_loss) / tick_size)
        tp_ticks_abs = int(abs(take_profit - price) / tick_size)

        MAX_TICKS = 1000
        if sl_ticks_abs > MAX_TICKS:
            sl_ticks_abs = MAX_TICKS
            stop_loss = price - (sl_ticks_abs * tick_size) if direction == 'buy' else price + (sl_ticks_abs * tick_size)
        if tp_ticks_abs > MAX_TICKS:
            tp_ticks_abs = MAX_TICKS
            take_profit = price + (tp_ticks_abs * tick_size) if direction == 'buy' else price - (tp_ticks_abs * tick_size)

        if direction == 'buy':
            sl_ticks = -sl_ticks_abs
            tp_ticks = tp_ticks_abs
        else:
            sl_ticks = sl_ticks_abs
            tp_ticks = -tp_ticks_abs

        if logger:
            logger.info(f"=" * 50)
            logger.info(f"PLACING MNQ {direction.upper()} ORDER ({CONTRACT_SIZE} contracts)")
            logger.info(f"  Contract: {TOPSTEPX_MNQ_CONTRACT_ID}")
            logger.info(f"  Entry: ~{price:.2f}")
            logger.info(f"  SL: {stop_loss:.2f} ({sl_ticks} ticks)")
            logger.info(f"  TP: {take_profit:.2f} ({tp_ticks} ticks)")

        payload = {
            "accountId": TOPSTEPX_ACCOUNT_ID,
            "contractId": TOPSTEPX_MNQ_CONTRACT_ID,
            "side": side,
            "size": CONTRACT_SIZE,
            "type": 2,
            "timeInForce": 1,
            "stopLossBracket": {"ticks": sl_ticks, "type": 4},
            "takeProfitBracket": {"ticks": tp_ticks, "type": 1}
        }

        response = requests.post(
            f"{TOPSTEPX_BASE_URL}/api/Order/place",
            json=payload,
            headers=get_auth_headers(),
            timeout=30
        )

        data = response.json()

        if data.get('success'):
            entry_order_id = data.get('orderId') or data.get('id')

            active_position_state.update({
                'has_position': True,
                'side': 'long' if direction == 'buy' else 'short',
                'entry_price': price,
                'entry_time': datetime.utcnow(),
                'entry_order_id': entry_order_id,
                'contracts': CONTRACT_SIZE,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'bars_held': 0,
            })

            if logger:
                logger.info(f"ORDER PLACED: Entry Order ID {entry_order_id}")
                logger.info(f"=" * 50)

            return True
        else:
            if logger:
                logger.error(f"ORDER FAILED: {data.get('errorMessage', data)}")
            return False

    except Exception as e:
        if logger:
            logger.error(f"ORDER ERROR: {e}")
        return False

def check_mnq_position() -> bool:
    """Check if we have an open MNQ position."""
    try:
        orders_response = requests.post(
            f"{TOPSTEPX_BASE_URL}/api/Order/searchOpen",
            json={"accountId": TOPSTEPX_ACCOUNT_ID},
            headers=get_auth_headers(),
            timeout=10
        )
        if orders_response.ok:
            orders = orders_response.json().get('orders', [])
            sl_orders = [o for o in orders if o.get('contractId') == TOPSTEPX_MNQ_CONTRACT_ID and o.get('type') == 4]
            return len(sl_orders) > 0
        return active_position_state['has_position']
    except:
        return active_position_state['has_position']

def check_position_timeout() -> bool:
    """Check position timeout and close if needed."""
    global active_position_state

    if not active_position_state['has_position']:
        return False

    bars_held = active_position_state.get('bars_held', 0)
    active_position_state['bars_held'] = bars_held + 1

    if bars_held >= POSITION_TIMEOUT_BARS:
        if logger:
            logger.info(f"Position timeout ({bars_held} bars) - closing")
        # TODO: Implement market close
        active_position_state['has_position'] = False
        return True

    return False

# =============================================================================
# MAIN LOOP
# =============================================================================

def trading_loop(models: Dict):
    """Main trading loop."""
    global session_stats

    fetch_live_nq_update()

    if len(cached_bars_1s) < 7200:
        return

    current_price = cached_bars_1s[-1]['c']
    session_stats['last_price'] = current_price

    if check_mnq_position():
        check_position_timeout()
        return

    signal = generate_signal(models)

    if signal:
        session_stats['signals_generated'] += 1
        if place_mnq_order(signal['direction'], signal['price'], signal['stop_loss'], signal['take_profit']):
            session_stats['trades'] += 1

def print_dashboard():
    """Print dashboard."""
    os.system('clear' if os.name != 'nt' else 'cls')

    uptime_mins = 0
    if session_stats['start_time']:
        uptime_mins = (datetime.utcnow() - session_stats['start_time']).total_seconds() / 60

    print("=" * 60)
    print(f"  NO-WHALE REGIME MNQ LIVE ({CONTRACT_SIZE}x)")
    print("=" * 60)
    print(f"  Strategy: NQ analysis → MNQ execution")
    print(f"  Contract: {TOPSTEPX_MNQ_CONTRACT_ID}")
    print(f"  Size: {CONTRACT_SIZE} contracts @ $2/pt")
    print(f"  Cached NQ bars: {len(cached_bars_1s):,}  |  Uptime: {int(uptime_mins)}m")
    print("-" * 60)
    print(f"  LAST NQ PRICE: {session_stats['last_price']:.2f}")
    print("-" * 60)
    print(f"  Trades: {session_stats['trades']}  |  Signals: {session_stats['signals_generated']}")
    print("=" * 60)

# =============================================================================
# MAIN
# =============================================================================

def main():
    global session_stats

    print("\n" + "=" * 60)
    print(f"  NO-WHALE REGIME MNQ LIVE ({CONTRACT_SIZE}x)")
    print("=" * 60)

    setup_logging()
    logger.info("=" * 50)
    logger.info(f"MNQ Live Trading ({CONTRACT_SIZE} contracts)")
    logger.info(f"Strategy: NQ analysis → MNQ execution")
    logger.info("=" * 50)
    logger.info(f"Account: {TOPSTEPX_ACCOUNT_ID}")
    logger.info(f"MNQ Contract: {TOPSTEPX_MNQ_CONTRACT_ID}")
    logger.info(f"NQ Data Contract: {NQ_DATA_CONTRACT_ID}")
    logger.info(f"Contract Size: {CONTRACT_SIZE}")

    logger.info("\nAuthenticating...")
    try:
        authenticate()
        logger.info("  Authentication successful")
    except Exception as e:
        logger.error(f"  Authentication failed: {e}")
        return

    logger.info("\nLoading NQ models...")
    models = load_models()
    if models is None:
        logger.error("Failed to load NQ models - ensure full-year baseline models exist")
        return

    logger.info("\nLoading NQ data...")
    update_nq_bars_from_api()

    if len(cached_bars_1s) < 7200:
        logger.error(f"Insufficient NQ data: {len(cached_bars_1s)} bars")
        return

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
            now = datetime.utcnow()
            if now.second < 1:
                time.sleep(1)
            elif now.second > 2:
                seconds_until_next_bar = (60 - now.second) + 1
                time.sleep(seconds_until_next_bar)

            trading_loop(models)
            print_dashboard()

            now = datetime.utcnow()
            seconds_until_next = (60 - now.second) + 1
            if seconds_until_next > 60:
                seconds_until_next = 1
            time.sleep(max(1, seconds_until_next))
        except Exception as e:
            logger.error(f"Loop error: {e}")
            time.sleep(5)

    logger.info("\nSaving NQ data...")
    save_nq_bars_to_disk(cached_bars_1s)

    logger.info("=" * 50)
    logger.info("SESSION ENDED")
    logger.info(f"Total trades: {session_stats['trades']}")
    logger.info("=" * 50)

if __name__ == "__main__":
    main()
