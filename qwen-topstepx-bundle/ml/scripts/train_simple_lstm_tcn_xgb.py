#!/usr/bin/env python3
"""
Simplified LSTM + TCN → XGBoost Ensemble

Clean architecture with NO data leakage:
- Input: 1-hour OHLCV candles only
- Features: OHLCV + SMA(50, 100, 200) = 8 features per timestep
- Lookback: 365 hours (about 2 weeks of 24/7 trading)
- LSTM + TCN extract embeddings
- XGBoost makes next-hour up/down prediction
- Train on ~8 months of data with proper holdout

NO volume profile, NO CVD, NO multi-timeframe - just pure price action!
"""

import json
import os
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

# Configuration
LOOKBACK_HOURS = 365  # ~15 days of hourly bars
BATCH_SIZE = 32
EMBEDDING_DIM = 64
HIDDEN_DIM = 128
NUM_LAYERS = 3
DROPOUT = 0.3
TCN_CHANNELS = [64, 128, 128, 64]
KERNEL_SIZE = 3

# Training
LSTM_EPOCHS = 50
PATIENCE = 10
LEARNING_RATE = 0.001

# Data splits
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# XGBoost params
XGB_PARAMS = {
    'objective': 'binary:logistic',
    'max_depth': 4,  # Reduced to prevent overfitting
    'learning_rate': 0.05,
    'n_estimators': 100,
    'min_child_weight': 3,  # Increased for regularization
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'gamma': 0.1,
    'reg_alpha': 1.0,  # L1 regularization
    'reg_lambda': 2.0,  # L2 regularization
    'random_state': 42,
    'tree_method': 'hist',
    'early_stopping_rounds': 15
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =============================================================================
# DATA LOADING
# =============================================================================

def load_bars(filepath: str) -> List[Dict]:
    """Load 1-hour bars."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data.get('bars', [])


def calculate_sma(closes: List[float], period: int) -> List[float]:
    """Calculate Simple Moving Average - NO LEAKAGE."""
    sma = [0.0] * len(closes)

    for i in range(len(closes)):
        if i >= period - 1:
            # Use only PREVIOUS bars, not including current
            window = closes[i - period + 1:i + 1]
            sma[i] = sum(window) / period
        else:
            # Not enough data - use what we have
            if i > 0:
                sma[i] = sum(closes[:i+1]) / (i + 1)
            else:
                sma[i] = closes[i]

    return sma


def extract_features(bars: List[Dict], lookback: int = LOOKBACK_HOURS) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract features from 1-hour bars - NO LEAKAGE.

    Features per timestep (8 total):
    - o_norm: (open - close) / close
    - h_norm: (high - close) / close
    - l_norm: (low - close) / close
    - range_norm: (high - low) / close
    - body_norm: (close - open) / close
    - sma50_dist: (close - sma50) / sma50
    - sma100_dist: (close - sma100) / sma100
    - sma200_dist: (close - sma200) / sma200

    Returns:
        X: [n_samples, lookback, 8]
        y: [n_samples] (1 = next hour up, 0 = next hour down)
    """
    closes = [b['c'] for b in bars]

    # Calculate SMAs - using only historical data (no leakage)
    print("  Calculating SMAs (50, 100, 200)...")
    sma50 = calculate_sma(closes, 50)
    sma100 = calculate_sma(closes, 100)
    sma200 = calculate_sma(closes, 200)

    X_all = []
    y_all = []

    # Start from lookback + 200 (need 200 bars for SMA200)
    start_idx = max(lookback, 200)

    print(f"  Creating sequences (lookback={lookback})...")
    for i in range(start_idx, len(bars) - 1):  # -1 because we need next bar for label
        # Extract sequence of lookback hours ENDING at bar i-1 (not including i)
        # This ensures no leakage when predicting bar i
        sequence = []

        for j in range(i - lookback, i):  # NOT including bar i
            bar = bars[j]
            close = bar['c']

            if close == 0:
                close = 1  # Avoid division by zero

            # Normalize by close
            o_norm = (bar['o'] - close) / close
            h_norm = (bar['h'] - close) / close
            l_norm = (bar['l'] - close) / close
            range_norm = (bar['h'] - bar['l']) / close
            body_norm = (bar['c'] - bar['o']) / close

            # SMA distances
            sma50_dist = (close - sma50[j]) / sma50[j] if sma50[j] > 0 else 0
            sma100_dist = (close - sma100[j]) / sma100[j] if sma100[j] > 0 else 0
            sma200_dist = (close - sma200[j]) / sma200[j] if sma200[j] > 0 else 0

            features = [
                o_norm, h_norm, l_norm, range_norm, body_norm,
                sma50_dist, sma100_dist, sma200_dist
            ]

            sequence.append(features)

        X_all.append(sequence)

        # Label: is next hour (bar i) higher than current bar (bar i-1)?
        current_close = bars[i-1]['c']
        next_close = bars[i]['c']
        label = 1.0 if next_close > current_close else 0.0
        y_all.append(label)

    X = np.array(X_all, dtype=np.float32)
    y = np.array(y_all, dtype=np.float32)

    print(f"  Created {len(X)} sequences")
    print(f"  Feature shape: {X.shape}")
    print(f"  Positive ratio: {y.mean():.3%}")

    return X, y


# =============================================================================
# MODELS
# =============================================================================

class Chomp1d(nn.Module):
    """Remove padding from end of sequence."""
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size > 0:
            return x[:, :, :-self.chomp_size].contiguous()
        return x


class TemporalBlock(nn.Module):
    """TCN temporal block with causal convolutions."""
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()

        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout2(out)

        res = x if self.downsample is None else self.downsample(x)

        if out.size(2) != res.size(2):
            min_len = min(out.size(2), res.size(2))
            out = out[:, :, :min_len]
            res = res[:, :, :min_len]

        return self.relu(out + res)


class TCN(nn.Module):
    """Temporal Convolutional Network."""
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        super().__init__()

        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]

            layers.append(
                TemporalBlock(
                    in_channels, out_channels, kernel_size,
                    stride=1, dilation=dilation_size,
                    padding=(kernel_size-1) * dilation_size,
                    dropout=dropout
                )
            )

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = x.transpose(1, 2)
        out = self.network(x)
        out = out.transpose(1, 2)
        return out


class HybridLSTMTCN(nn.Module):
    """Hybrid LSTM + TCN for feature extraction."""
    def __init__(self, input_size, lstm_hidden=HIDDEN_DIM, lstm_layers=NUM_LAYERS,
                 tcn_channels=TCN_CHANNELS, kernel_size=KERNEL_SIZE,
                 embedding_dim=EMBEDDING_DIM, dropout=DROPOUT):
        super().__init__()

        # LSTM branch
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        self.lstm_norm = nn.LayerNorm(lstm_hidden)

        # TCN branch
        self.tcn = TCN(
            num_inputs=input_size,
            num_channels=tcn_channels,
            kernel_size=kernel_size,
            dropout=dropout
        )
        self.tcn_norm = nn.LayerNorm(tcn_channels[-1])

        # Fusion
        combined_size = lstm_hidden + tcn_channels[-1]
        self.fusion = nn.Sequential(
            nn.Linear(combined_size, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Prediction head (for training)
        self.head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, 1)
        )

    def forward(self, x):
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        lstm_embedding = self.lstm_norm(h_n[-1])

        # TCN
        tcn_out = self.tcn(x)
        tcn_embedding = self.tcn_norm(tcn_out[:, -1, :])

        # Fuse
        combined = torch.cat([lstm_embedding, tcn_embedding], dim=1)
        embedding = self.fusion(combined)

        # Prediction
        logits = self.head(embedding)

        return embedding, logits


def extract_embeddings(model, dataloader, device):
    """Extract embeddings from trained LSTM+TCN."""
    model.eval()
    all_embeddings = []

    with torch.no_grad():
        for batch_X in dataloader:
            batch_X = batch_X.to(device)
            embedding, _ = model(batch_X)
            all_embeddings.append(embedding.cpu().numpy())

    return np.vstack(all_embeddings)


# =============================================================================
# TRAINING
# =============================================================================

def train_feature_extractor(X_train, X_val, y_train, y_val):
    """Train LSTM+TCN as feature extractor."""
    print("\n" + "="*70)
    print("STEP 1: TRAINING LSTM+TCN FEATURE EXTRACTOR")
    print("="*70)

    class SimpleDataset(Dataset):
        def __init__(self, X, y):
            self.X = torch.FloatTensor(X)
            self.y = torch.FloatTensor(y)

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    train_dataset = SimpleDataset(X_train, y_train)
    val_dataset = SimpleDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize model
    model = HybridLSTMTCN(
        input_size=X_train.shape[2],
        lstm_hidden=HIDDEN_DIM,
        lstm_layers=NUM_LAYERS,
        tcn_channels=TCN_CHANNELS,
        kernel_size=KERNEL_SIZE,
        embedding_dim=EMBEDDING_DIM,
        dropout=DROPOUT
    ).to(DEVICE)

    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(LSTM_EPOCHS):
        model.train()
        train_loss = 0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad()
            _, logits = model(batch_X)
            loss = criterion(logits, batch_y.unsqueeze(1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(DEVICE)
                batch_y = batch_y.to(DEVICE)

                _, logits = model(batch_X)
                loss = criterion(logits, batch_y.unsqueeze(1))
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{LSTM_EPOCHS} - Train: {avg_train_loss:.4f}, Val: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    print(f"✓ Feature extractor trained (Best Val Loss: {best_val_loss:.4f})")
    return model


def train_xgboost(embeddings_train, embeddings_val, embeddings_test,
                  features_train, features_val, features_test,
                  y_train, y_val, y_test):
    """Train XGBoost using embeddings + statistical features."""
    print("\n" + "="*70)
    print("STEP 3: TRAINING XGBOOST")
    print("="*70)

    # Combine embeddings with statistical features
    X_train_combined = np.hstack([embeddings_train, features_train])
    X_val_combined = np.hstack([embeddings_val, features_val])
    X_test_combined = np.hstack([embeddings_test, features_test])

    print(f"  Combined features: {X_train_combined.shape}")
    print(f"  ({embeddings_train.shape[1]} embeddings + {features_train.shape[1]} statistical)")

    dtrain = xgb.DMatrix(X_train_combined, label=y_train)
    dval = xgb.DMatrix(X_val_combined, label=y_val)
    dtest = xgb.DMatrix(X_test_combined, label=y_test)

    evals = [(dtrain, 'train'), (dval, 'val')]

    model = xgb.train(
        XGB_PARAMS,
        dtrain,
        num_boost_round=XGB_PARAMS['n_estimators'],
        evals=evals,
        early_stopping_rounds=XGB_PARAMS['early_stopping_rounds'],
        verbose_eval=False
    )

    # Predictions
    train_preds = (model.predict(dtrain) > 0.5).astype(int)
    val_preds = (model.predict(dval) > 0.5).astype(int)
    test_preds = (model.predict(dtest) > 0.5).astype(int)

    train_acc = np.mean(train_preds == y_train)
    val_acc = np.mean(val_preds == y_val)
    test_acc = np.mean(test_preds == y_test)

    print(f"  Train Acc: {train_acc:.3%}, Val Acc: {val_acc:.3%}, Test Acc: {test_acc:.3%}")

    return model, test_acc


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*70)
    print("SIMPLE LSTM + TCN → XGBOOST")
    print("="*70)
    print("Architecture: 1-hour OHLCV + SMAs → LSTM+TCN → XGBoost")
    print(f"Features: OHLCV + SMA(50,100,200) = 8 features")
    print(f"Lookback: {LOOKBACK_HOURS} hours")
    print(f"Prediction: Next hour up/down")
    print(f"Device: {DEVICE}")
    print("="*70)

    # Paths
    data_dir = Path("/home/costa/Trading-Agent/qwen-topstepx-bundle/ml/data_TEST/longterm")
    output_dir = Path("/home/costa/Trading-Agent/qwen-topstepx-bundle/ml/models_TEST/simple")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\nLoading 1-hour bars...")
    bars = load_bars(data_dir / "bars_1h_enq_2025_merged.json")
    print(f"  Loaded {len(bars)} hourly bars")

    # Extract features
    print("\nExtracting features (NO LEAKAGE)...")
    X, y = extract_features(bars, lookback=LOOKBACK_HOURS)

    # Split data
    n_samples = len(X)
    train_end = int(n_samples * TRAIN_RATIO)
    val_end = int(n_samples * (TRAIN_RATIO + VAL_RATIO))

    X_train = X[:train_end]
    X_val = X[train_end:val_end]
    X_test = X[val_end:]

    y_train = y[:train_end]
    y_val = y[train_end:val_end]
    y_test = y[val_end:]

    print(f"\nData splits: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    # Normalize (NO LEAKAGE - fit only on train)
    print("\nNormalizing...")
    scaler = StandardScaler()
    X_train_flat = X_train.reshape(-1, X_train.shape[-1])
    X_val_flat = X_val.reshape(-1, X_val.shape[-1])
    X_test_flat = X_test.reshape(-1, X_test.shape[-1])

    X_train_scaled = scaler.fit_transform(X_train_flat).reshape(X_train.shape)
    X_val_scaled = scaler.transform(X_val_flat).reshape(X_val.shape)
    X_test_scaled = scaler.transform(X_test_flat).reshape(X_test.shape)

    # Train LSTM+TCN
    lstm_tcn_model = train_feature_extractor(X_train_scaled, X_val_scaled, y_train, y_val)

    # Extract embeddings
    print("\n" + "="*70)
    print("STEP 2: EXTRACTING EMBEDDINGS")
    print("="*70)

    class EmbDataset(Dataset):
        def __init__(self, X):
            self.X = torch.FloatTensor(X)
        def __len__(self):
            return len(self.X)
        def __getitem__(self, idx):
            return self.X[idx]

    train_loader = DataLoader(EmbDataset(X_train_scaled), batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(EmbDataset(X_val_scaled), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(EmbDataset(X_test_scaled), batch_size=BATCH_SIZE, shuffle=False)

    emb_train = extract_embeddings(lstm_tcn_model, train_loader, DEVICE)
    emb_val = extract_embeddings(lstm_tcn_model, val_loader, DEVICE)
    emb_test = extract_embeddings(lstm_tcn_model, test_loader, DEVICE)

    print(f"  Embeddings shape: {emb_train.shape}")

    # Statistical features (mean over sequence)
    feat_train = np.mean(X_train, axis=1)
    feat_val = np.mean(X_val, axis=1)
    feat_test = np.mean(X_test, axis=1)

    # Train XGBoost
    xgb_model, test_acc = train_xgboost(
        emb_train, emb_val, emb_test,
        feat_train, feat_val, feat_test,
        y_train, y_val, y_test
    )

    # Save models
    print("\n" + "="*70)
    print("SAVING MODELS")
    print("="*70)

    torch.save({
        'model_state_dict': lstm_tcn_model.state_dict(),
        'scaler': scaler,
        'config': {
            'input_size': X_train.shape[2],
            'lookback': LOOKBACK_HOURS,
            'embedding_dim': EMBEDDING_DIM
        }
    }, output_dir / 'lstm_tcn_simple.pt')

    xgb_model.save_model(str(output_dir / 'xgb_simple.json'))

    metadata = {
        'architecture': 'Simple LSTM+TCN→XGBoost',
        'features': 'OHLCV + SMA(50,100,200)',
        'lookback_hours': LOOKBACK_HOURS,
        'test_accuracy': float(test_acc),
        'trained_samples': len(X_train)
    }

    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Saved to {output_dir}/")

    print("\n" + "="*70)
    print("FINAL RESULTS (HOLDOUT TEST SET)")
    print("="*70)
    print(f"Next Hour Prediction Accuracy: {test_acc:.3%}")
    print("="*70)


if __name__ == "__main__":
    main()
