#!/usr/bin/env python3
"""
Multi-LSTM → Multi-XGBoost ensemble (no leakage, 1-hour data only).

Design:
- Inputs: 1-hour OHLCV + SMA(50/100/200) features (8 per timestep)
- No volume profile, no CVD, no multi-timeframe bars
- Four parallel LSTM branches (different hidden sizes) produce embeddings
- Embeddings + simple statistical features feed three XGBoost heads:
    * Next 1h up/down
    * Next 4h up/down (using 1h bars)
    * Next 24h (1d) up/down (using 1h bars)
- Time-series split (70/15/15), scaler fit only on train (no leakage)
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Paths
DATA_PATH = Path("/home/costa/Trading-Agent/qwen-topstepx-bundle/ml/data_TEST/longterm/bars_1h_enq_2025_merged.json")
OUTPUT_DIR = Path("/home/costa/Trading-Agent/qwen-topstepx-bundle/ml/models_TEST/ensemble_lstm_only")

# Sequence / feature config
LOOKBACK_HOURS = 365   # ~15 days of hourly bars
FEATURE_COUNT = 8      # OHLCV + SMA distances (50/100/200)

# LSTM branches (hidden dims)
LSTM_HIDDEN_SIZES = [64, 96, 128, 160]
LSTM_LAYERS = 2
LSTM_DROPOUT = 0.3
EMBEDDING_DIM = 128    # fused embedding size

# Training
BATCH_SIZE = 32
EPOCHS = 50
PATIENCE = 10
LR = 0.001

# Splits
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# XGBoost params (shared)
XGB_PARAMS = {
    'objective': 'binary:logistic',
    'max_depth': 4,
    'learning_rate': 0.05,
    'n_estimators': 150,
    'min_child_weight': 4,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'gamma': 0.1,
    'reg_alpha': 1.0,
    'reg_lambda': 2.0,
    'random_state': 42,
    'tree_method': 'hist',
    'early_stopping_rounds': 20,
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_bars(path: Path) -> List[Dict]:
    """Load 1-hour bars from JSON file."""
    with path.open('r') as f:
        data = json.load(f)
    return data.get('bars', [])


def calculate_sma(closes: List[float], period: int) -> List[float]:
    """Simple moving average with no leakage (uses only past/current)."""
    sma = [0.0] * len(closes)
    for i in range(len(closes)):
        if i >= period - 1:
            window = closes[i - period + 1:i + 1]
            sma[i] = sum(window) / period
        else:
            sma[i] = sum(closes[:i + 1]) / (i + 1)
    return sma


def build_sequences(
    bars: List[Dict],
    lookback: int = LOOKBACK_HOURS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build sequences and labels for 1h/4h/1d predictions (no leakage).

    Features per timestep (8):
      - o_norm, h_norm, l_norm, range_norm, body_norm
      - sma50_dist, sma100_dist, sma200_dist
    Labels:
      - y_1h: next bar close > current close
      - y_4h: close 4 bars ahead > current close
      - y_1d: close 24 bars ahead > current close
    """
    closes = [b['c'] for b in bars]
    sma50 = calculate_sma(closes, 50)
    sma100 = calculate_sma(closes, 100)
    sma200 = calculate_sma(closes, 200)

    sequences: List[List[List[float]]] = []
    y_1h: List[float] = []
    y_4h: List[float] = []
    y_1d: List[float] = []

    horizon_1h = 1
    horizon_4h = 4
    horizon_1d = 24

    start_idx = max(lookback, 200)
    last_idx = len(bars) - horizon_1d  # ensure we can look 24h ahead

    for i in range(start_idx, last_idx):
        seq = []
        for j in range(i - lookback, i):
            bar = bars[j]
            close = bar['c'] or 1.0

            o_norm = (bar['o'] - close) / close
            h_norm = (bar['h'] - close) / close
            l_norm = (bar['l'] - close) / close
            range_norm = (bar['h'] - bar['l']) / close
            body_norm = (bar['c'] - bar['o']) / close

            sma50_dist = (close - sma50[j]) / sma50[j] if sma50[j] else 0.0
            sma100_dist = (close - sma100[j]) / sma100[j] if sma100[j] else 0.0
            sma200_dist = (close - sma200[j]) / sma200[j] if sma200[j] else 0.0

            seq.append([
                o_norm, h_norm, l_norm, range_norm, body_norm,
                sma50_dist, sma100_dist, sma200_dist
            ])

        current_close = bars[i - 1]['c']
        next_1h_close = bars[i + horizon_1h - 1]['c']
        next_4h_close = bars[i + horizon_4h - 1]['c']
        next_1d_close = bars[i + horizon_1d - 1]['c']

        sequences.append(seq)
        y_1h.append(1.0 if next_1h_close > current_close else 0.0)
        y_4h.append(1.0 if next_4h_close > current_close else 0.0)
        y_1d.append(1.0 if next_1d_close > current_close else 0.0)

    X = np.array(sequences, dtype=np.float32)
    y1 = np.array(y_1h, dtype=np.float32)
    y4 = np.array(y_4h, dtype=np.float32)
    y24 = np.array(y_1d, dtype=np.float32)

    print(f"  Sequences: {X.shape}, Pos ratios -> 1h: {y1.mean():.3%}, 4h: {y4.mean():.3%}, 1d: {y24.mean():.3%}")
    return X, y1, y4, y24


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MultiLSTM(nn.Module):
    """Multiple LSTM branches fused into a single embedding."""
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        num_layers: int,
        dropout: float,
        embedding_dim: int,
    ):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.LSTM(
                input_size=input_size,
                hidden_size=h,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0
            ) for h in hidden_sizes
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(h) for h in hidden_sizes])

        fused_size = sum(hidden_sizes)
        self.fusion = nn.Sequential(
            nn.Linear(fused_size, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, 1)
        )

    def forward(self, x):
        embeddings = []
        for lstm, norm in zip(self.branches, self.norms):
            out, (h_n, _) = lstm(x)
            embeddings.append(norm(h_n[-1]))

        combined = torch.cat(embeddings, dim=1)
        fused = self.fusion(combined)
        logits = self.head(fused)
        return fused, logits


def train_lstm_feature_extractor(X_train, X_val, y_train, y_val) -> MultiLSTM:
    """Train multi-LSTM on next-hour label (feature extractor)."""
    print("\n" + "="*70)
    print("STEP 1: TRAINING MULTI-LSTM FEATURE EXTRACTOR")
    print("="*70)

    class SeqDataset(Dataset):
        def __init__(self, X, y):
            self.X = torch.FloatTensor(X)
            self.y = torch.FloatTensor(y)
        def __len__(self):
            return len(self.X)
        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    train_loader = DataLoader(SeqDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(SeqDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

    model = MultiLSTM(
        input_size=X_train.shape[2],
        hidden_sizes=LSTM_HIDDEN_SIZES,
        num_layers=LSTM_LAYERS,
        dropout=LSTM_DROPOUT,
        embedding_dim=EMBEDDING_DIM,
    ).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val = float('inf')
    patience = 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            _, logits = model(batch_X)
            loss = criterion(logits.squeeze(1), batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
                _, logits = model(batch_X)
                loss = criterion(logits.squeeze(1), batch_y)
                val_loss += loss.item()

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{EPOCHS} - Train {avg_train:.4f} | Val {avg_val:.4f}")

        if avg_val < best_val:
            best_val = avg_val
            patience = 0
        else:
            patience += 1
            if patience >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    print(f"✓ LSTM extractor ready (best val loss {best_val:.4f})")
    return model


def extract_embeddings(model: MultiLSTM, X: np.ndarray) -> np.ndarray:
    """Run sequences through feature extractor to get embeddings."""
    loader = DataLoader(torch.FloatTensor(X), batch_size=BATCH_SIZE, shuffle=False)
    model.eval()
    embs = []
    with torch.no_grad():
        for batch_X in loader:
            batch_X = batch_X.to(DEVICE)
            fused, _ = model(batch_X)
            embs.append(fused.cpu().numpy())
    return np.vstack(embs)


def train_xgb(
    X_train, X_val, X_test,
    y_train, y_val, y_test,
    tag: str
) -> Tuple[xgb.Booster, float]:
    """Train a single XGBoost head."""
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test, label=y_test)

    evals = [(dtrain, 'train'), (dval, 'val')]
    model = xgb.train(
        params=XGB_PARAMS,
        dtrain=dtrain,
        num_boost_round=XGB_PARAMS['n_estimators'],
        evals=evals,
        early_stopping_rounds=XGB_PARAMS['early_stopping_rounds'],
        verbose_eval=False
    )

    test_preds = (model.predict(dtest) > 0.5).astype(int)
    test_acc = float(np.mean(test_preds == y_test))
    print(f"  [{tag}] Test accuracy: {test_acc:.3%}")
    return model, test_acc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("MULTI-LSTM → MULTI-XGBOOST ENSEMBLE (1H ONLY, NO VP/CVD)")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Lookback hours: {LOOKBACK_HOURS}")
    print(f"Branches: {len(LSTM_HIDDEN_SIZES)} LSTMs -> fused embedding {EMBEDDING_DIM}")
    print("Heads: XGBoost for next 1h, next 4h, next 24h")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\nLoading 1h bars...")
    bars = load_bars(DATA_PATH)
    print(f"  Loaded {len(bars)} bars from {DATA_PATH}")

    # Build sequences / labels
    print("\nBuilding sequences (no leakage)...")
    X, y1, y4, y24 = build_sequences(bars, lookback=LOOKBACK_HOURS)

    # Time-based split
    n = len(X)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

    X_train, X_val, X_test = X[:train_end], X[train_end:val_end], X[val_end:]
    y1_train, y1_val, y1_test = y1[:train_end], y1[train_end:val_end], y1[val_end:]
    y4_train, y4_val, y4_test = y4[:train_end], y4[train_end:val_end], y4[val_end:]
    y24_train, y24_val, y24_test = y24[:train_end], y24[train_end:val_end], y24[val_end:]

    print(f"Splits -> Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Normalize (fit on train only)
    print("\nNormalizing features (no leakage)...")
    scaler = StandardScaler()
    X_train_flat = X_train.reshape(-1, FEATURE_COUNT)
    X_val_flat = X_val.reshape(-1, FEATURE_COUNT)
    X_test_flat = X_test.reshape(-1, FEATURE_COUNT)

    X_train_scaled = scaler.fit_transform(X_train_flat).reshape(X_train.shape)
    X_val_scaled = scaler.transform(X_val_flat).reshape(X_val.shape)
    X_test_scaled = scaler.transform(X_test_flat).reshape(X_test.shape)

    # Train LSTM extractor (use 1h label)
    lstm_model = train_lstm_feature_extractor(X_train_scaled, X_val_scaled, y1_train, y1_val)

    # Extract embeddings
    print("\n" + "=" * 70)
    print("STEP 2: EXTRACTING EMBEDDINGS")
    print("=" * 70)
    emb_train = extract_embeddings(lstm_model, X_train_scaled)
    emb_val = extract_embeddings(lstm_model, X_val_scaled)
    emb_test = extract_embeddings(lstm_model, X_test_scaled)

    print(f"  Embedding shape: {emb_train.shape}")

    # Statistical features (mean over sequence)
    feat_train = np.mean(X_train, axis=1)
    feat_val = np.mean(X_val, axis=1)
    feat_test = np.mean(X_test, axis=1)

    # Combine embeddings + stats
    def combine(emb, feat):
        return np.hstack([emb, feat])

    X1_train = combine(emb_train, feat_train)
    X1_val = combine(emb_val, feat_val)
    X1_test = combine(emb_test, feat_test)

    # Train three XGB heads
    print("\n" + "=" * 70)
    print("STEP 3: TRAINING XGBOOST HEADS")
    print("=" * 70)
    xgb_1h, acc_1h = train_xgb(X1_train, X1_val, X1_test, y1_train, y1_val, y1_test, "1h")
    xgb_4h, acc_4h = train_xgb(X1_train, X1_val, X1_test, y4_train, y4_val, y4_test, "4h")
    xgb_1d, acc_1d = train_xgb(X1_train, X1_val, X1_test, y24_train, y24_val, y24_test, "1d")

    # Save artifacts
    print("\n" + "=" * 70)
    print("STEP 4: SAVING MODELS")
    print("=" * 70)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    torch.save({
        'state_dict': lstm_model.state_dict(),
        'scaler': scaler,
        'config': {
            'input_size': FEATURE_COUNT,
            'lookback_hours': LOOKBACK_HOURS,
            'hidden_sizes': LSTM_HIDDEN_SIZES,
            'embedding_dim': EMBEDDING_DIM,
        }
    }, OUTPUT_DIR / "multi_lstm_feature_extractor.pt")

    xgb_1h.save_model(str(OUTPUT_DIR / "xgb_head_1h.json"))
    xgb_4h.save_model(str(OUTPUT_DIR / "xgb_head_4h.json"))
    xgb_1d.save_model(str(OUTPUT_DIR / "xgb_head_1d.json"))

    metadata = {
        'architecture': 'Multi-LSTM -> Multi-XGBoost (1h only)',
        'features': 'OHLCV + SMA50/100/200',
        'lookback_hours': LOOKBACK_HOURS,
        'embedding_dim': EMBEDDING_DIM,
        'hidden_sizes': LSTM_HIDDEN_SIZES,
        'xgb_params': XGB_PARAMS,
        'test_accuracy': {
            '1h': acc_1h,
            '4h': acc_4h,
            '1d': acc_1d,
        }
    }
    with (OUTPUT_DIR / "metadata.json").open('w') as f:
        json.dump(metadata, f, indent=2)

    print("✓ Saved models to", OUTPUT_DIR)
    print("\n" + "=" * 70)
    print("FINAL HOLDOUT ACCURACY")
    print("=" * 70)
    print(f"1h: {acc_1h:.3%} | 4h: {acc_4h:.3%} | 1d: {acc_1d:.3%}")
    print("=" * 70)


if __name__ == "__main__":
    main()
