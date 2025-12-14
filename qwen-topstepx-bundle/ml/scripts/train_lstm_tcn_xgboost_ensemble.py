#!/usr/bin/env python3
"""
Hierarchical Ensemble: LSTM + TCN → XGBoost

Architecture:
1. LSTM extracts sequential embeddings (64-dim)
2. TCN extracts temporal pattern features (64-dim)
3. Combined embeddings (128-dim) + original features → XGBoost
4. Separate XGBoost models for 1d, 4h, 1h predictions

This creates a powerful ensemble where deep learning handles feature extraction
and XGBoost makes final predictions.
"""

import json
import sys
import os
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Import model components from the v2 script
import importlib.util
spec = importlib.util.spec_from_file_location(
    "train_v2",
    "/home/costa/Trading-Agent/qwen-topstepx-bundle/ml/scripts/train_htf_lstm_tcn_multitask_v2.py"
)
train_v2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_v2)

# Import components
HybridLSTMTCN = train_v2.HybridLSTMTCN
load_bars = train_v2.load_bars
load_microstructure_features = train_v2.load_microstructure_features
calculate_cvd_hourly = train_v2.calculate_cvd_hourly
calculate_daily_volume_profile = train_v2.calculate_daily_volume_profile
extract_longterm_features = train_v2.extract_longterm_features

# Configuration
DAILY_SEQ_LEN = 60
EMBEDDING_DIM = 64
HIDDEN_DIM = 128
NUM_LAYERS = 3
DROPOUT = 0.3
TCN_CHANNELS = [64, 128, 128, 64]
KERNEL_SIZE = 3

# Training configuration
BATCH_SIZE = 32
LEARNING_RATE = 0.001
LSTM_EPOCHS = 50  # Fewer epochs since we're using it for feature extraction
PATIENCE = 10

# Data splits
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# XGBoost hyperparameters
XGB_PARAMS = {
    'objective': 'binary:logistic',
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 200,
    'min_child_weight': 1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 0,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'tree_method': 'hist',
    'early_stopping_rounds': 20
}


def extract_embeddings(model, dataloader, device):
    """
    Extract LSTM and TCN embeddings from the trained model.

    Returns:
        numpy array of shape [n_samples, embedding_dim * 2]
    """
    model.eval()
    all_embeddings = []

    with torch.no_grad():
        for batch_X in dataloader:
            batch_X = batch_X.to(device)

            # Forward through LSTM
            lstm_out, (h_n, c_n) = model.lstm(batch_X)
            lstm_embedding = model.lstm_norm(h_n[-1])

            # Forward through TCN
            tcn_out = model.tcn(batch_X)
            tcn_embedding = model.tcn_norm(tcn_out[:, -1, :])

            # Combine embeddings
            combined = torch.cat([lstm_embedding, tcn_embedding], dim=1)

            all_embeddings.append(combined.cpu().numpy())

    return np.vstack(all_embeddings)


def train_lstm_tcn_feature_extractor(X_train, X_val, y_train_1d, y_val_1d):
    """
    Train LSTM+TCN as a feature extractor.

    We'll train it on 1d predictions to get good representations.
    """
    print("\n" + "="*70)
    print("STEP 1: TRAINING LSTM+TCN FEATURE EXTRACTOR")
    print("="*70)

    # Create datasets (only need X for embedding extraction)
    class SimpleDataset(Dataset):
        def __init__(self, X, y=None):
            self.X = torch.FloatTensor(X)
            self.y = torch.FloatTensor(y) if y is not None else None

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            if self.y is not None:
                return self.X[idx], self.y[idx]
            return self.X[idx]

    train_dataset = SimpleDataset(X_train, y_train_1d)
    val_dataset = SimpleDataset(X_val, y_val_1d)

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

    # Simple training on 1d task to get good representations
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

            # Only use 1d head for feature extractor training
            logits_1d, _, _ = model(batch_X)
            loss = criterion(logits_1d, batch_y.unsqueeze(1))

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

                logits_1d, _, _ = model(batch_X)
                loss = criterion(logits_1d, batch_y.unsqueeze(1))
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{LSTM_EPOCHS} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    print(f"✓ LSTM+TCN feature extractor trained (Best Val Loss: {best_val_loss:.4f})")

    return model


def train_xgboost_ensemble(embeddings_train, embeddings_val, embeddings_test,
                           features_train, features_val, features_test,
                           y_train, y_val, y_test, task_name):
    """
    Train XGBoost using LSTM+TCN embeddings + original features.

    Args:
        embeddings_*: LSTM+TCN embeddings [n_samples, 128]
        features_*: Statistical features [n_samples, n_features]
        y_*: Labels [n_samples]
        task_name: "1d", "4h", or "1h"
    """
    print(f"\n  Training XGBoost for {task_name} prediction...")

    # Combine embeddings with features
    X_train_combined = np.hstack([embeddings_train, features_train])
    X_val_combined = np.hstack([embeddings_val, features_val])
    X_test_combined = np.hstack([embeddings_test, features_test])

    print(f"    Combined features shape: {X_train_combined.shape}")
    print(f"    ({embeddings_train.shape[1]} embeddings + {features_train.shape[1]} statistical features)")

    # Train XGBoost
    dtrain = xgb.DMatrix(X_train_combined, label=y_train)
    dval = xgb.DMatrix(X_val_combined, label=y_val)
    dtest = xgb.DMatrix(X_test_combined, label=y_test)

    # Train with early stopping on validation set
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

    # Accuracies
    train_acc = np.mean(train_preds == y_train)
    val_acc = np.mean(val_preds == y_val)
    test_acc = np.mean(test_preds == y_test)

    print(f"    Train Acc: {train_acc:.3%}, Val Acc: {val_acc:.3%}, Test Acc: {test_acc:.3%}")

    return model, test_acc, test_preds


def main():
    print("="*70)
    print("HIERARCHICAL ENSEMBLE: LSTM + TCN → XGBOOST")
    print("="*70)
    print(f"Device: {DEVICE}")
    print(f"Architecture: Deep Learning (feature extraction) + XGBoost (decisions)")
    print("="*70)

    # Paths
    data_dir = "/home/costa/Trading-Agent/qwen-topstepx-bundle/ml/data_TEST/longterm"
    output_dir = "/home/costa/Trading-Agent/qwen-topstepx-bundle/ml/models_TEST/longterm"
    os.makedirs(output_dir, exist_ok=True)

    # Load bars
    print("\nLoading bar data...")
    bars_1h = load_bars(os.path.join(data_dir, 'bars_1h_enq_2025_merged.json'))
    bars_4h = load_bars(os.path.join(data_dir, 'bars_4h_enq_2025_merged.json'))
    bars_daily = load_bars(os.path.join(data_dir, 'bars_1d_enq_2025_merged.json'))

    print(f"  1-hour bars: {len(bars_1h)}")
    print(f"  4-hour bars: {len(bars_4h)}")
    print(f"  Daily bars: {len(bars_daily)}")

    # Load microstructure features
    daily_vp_1s, cvd_5min, cvd_ema_5min = load_microstructure_features(data_dir)

    # Calculate traditional features
    print("\nComputing features...")
    cvd_1h, cvd_ema_1h = calculate_cvd_hourly(bars_1h)
    daily_vp = calculate_daily_volume_profile(bars_daily, lookback=30)

    # Create samples
    print("\nCreating samples...")
    closes_daily = [b['c'] for b in bars_daily]
    closes_4h = [b['c'] for b in bars_4h]
    closes_1h = [b['c'] for b in bars_1h]

    X_all = []
    y_all_1d = []
    y_all_4h = []
    y_all_1h = []

    for i in range(DAILY_SEQ_LEN, len(bars_daily) - 1):
        seq_features = extract_longterm_features(
            bars_1h[:i*24], bars_4h[:i*6], bars_daily[:i],
            cvd_1h[:i*24], cvd_ema_1h[:i*24], daily_vp[:i],
            daily_vp_1s, cvd_5min, cvd_ema_5min,
            num_days=DAILY_SEQ_LEN
        )

        if seq_features is not None:
            X_all.append(seq_features)

            # Labels
            current_price_1d = closes_daily[i]
            future_price_1d = closes_daily[i + 1]
            y_all_1d.append(1.0 if future_price_1d > current_price_1d else 0.0)

            h4_idx = min(i * 6, len(closes_4h) - 1)
            next_h4_idx = min(h4_idx + 1, len(closes_4h) - 1)
            current_price_4h = closes_4h[h4_idx]
            future_price_4h = closes_4h[next_h4_idx]
            y_all_4h.append(1.0 if future_price_4h > current_price_4h else 0.0)

            h1_idx = min(i * 24, len(closes_1h) - 1)
            next_h1_idx = min(h1_idx + 1, len(closes_1h) - 1)
            current_price_1h = closes_1h[h1_idx]
            future_price_1h = closes_1h[next_h1_idx]
            y_all_1h.append(1.0 if future_price_1h > current_price_1h else 0.0)

    X_all = np.array(X_all)
    y_all_1d = np.array(y_all_1d)
    y_all_4h = np.array(y_all_4h)
    y_all_1h = np.array(y_all_1h)

    print(f"  Total samples: {len(X_all)}")
    print(f"  Feature shape: {X_all.shape}")

    # Time-series split
    n_samples = len(X_all)
    train_end = int(n_samples * TRAIN_RATIO)
    val_end = int(n_samples * (TRAIN_RATIO + VAL_RATIO))

    X_train = X_all[:train_end]
    X_val = X_all[train_end:val_end]
    X_test = X_all[val_end:]

    y_train_1d = y_all_1d[:train_end]
    y_val_1d = y_all_1d[train_end:val_end]
    y_test_1d = y_all_1d[val_end:]

    y_train_4h = y_all_4h[:train_end]
    y_val_4h = y_all_4h[train_end:val_end]
    y_test_4h = y_all_4h[val_end:]

    y_train_1h = y_all_1h[:train_end]
    y_val_1h = y_all_1h[train_end:val_end]
    y_test_1h = y_all_1h[val_end:]

    print(f"\nData splits: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    # Normalize features (NO LEAKAGE - fit only on train)
    print("\nNormalizing features...")
    scaler = StandardScaler()
    X_train_flat = X_train.reshape(-1, X_train.shape[-1])
    X_val_flat = X_val.reshape(-1, X_val.shape[-1])
    X_test_flat = X_test.reshape(-1, X_test.shape[-1])

    X_train_scaled = scaler.fit_transform(X_train_flat).reshape(X_train.shape)
    X_val_scaled = scaler.transform(X_val_flat).reshape(X_val.shape)
    X_test_scaled = scaler.transform(X_test_flat).reshape(X_test.shape)

    # Train LSTM+TCN feature extractor
    lstm_tcn_model = train_lstm_tcn_feature_extractor(
        X_train_scaled, X_val_scaled, y_train_1d, y_val_1d
    )

    # Extract embeddings
    print("\n" + "="*70)
    print("STEP 2: EXTRACTING EMBEDDINGS FROM LSTM+TCN")
    print("="*70)

    class EmbeddingDataset(Dataset):
        def __init__(self, X):
            self.X = torch.FloatTensor(X)
        def __len__(self):
            return len(self.X)
        def __getitem__(self, idx):
            return self.X[idx]

    train_emb_loader = DataLoader(EmbeddingDataset(X_train_scaled), batch_size=BATCH_SIZE, shuffle=False)
    val_emb_loader = DataLoader(EmbeddingDataset(X_val_scaled), batch_size=BATCH_SIZE, shuffle=False)
    test_emb_loader = DataLoader(EmbeddingDataset(X_test_scaled), batch_size=BATCH_SIZE, shuffle=False)

    embeddings_train = extract_embeddings(lstm_tcn_model, train_emb_loader, DEVICE)
    embeddings_val = extract_embeddings(lstm_tcn_model, val_emb_loader, DEVICE)
    embeddings_test = extract_embeddings(lstm_tcn_model, test_emb_loader, DEVICE)

    print(f"  Extracted embeddings shape: {embeddings_train.shape}")
    print(f"  (64 LSTM + 64 TCN = 128 total)")

    # Create statistical features (flatten the sequence, take last timestep, or aggregate)
    # For simplicity, let's take mean across the sequence
    features_train = np.mean(X_train, axis=1)  # [n_samples, 32]
    features_val = np.mean(X_val, axis=1)
    features_test = np.mean(X_test, axis=1)

    print(f"  Statistical features shape: {features_train.shape}")

    # Train XGBoost models
    print("\n" + "="*70)
    print("STEP 3: TRAINING XGBOOST ENSEMBLE")
    print("="*70)

    xgb_1d, acc_1d, preds_1d = train_xgboost_ensemble(
        embeddings_train, embeddings_val, embeddings_test,
        features_train, features_val, features_test,
        y_train_1d, y_val_1d, y_test_1d, "1d"
    )

    xgb_4h, acc_4h, preds_4h = train_xgboost_ensemble(
        embeddings_train, embeddings_val, embeddings_test,
        features_train, features_val, features_test,
        y_train_4h, y_val_4h, y_test_4h, "4h"
    )

    xgb_1h, acc_1h, preds_1h = train_xgboost_ensemble(
        embeddings_train, embeddings_val, embeddings_test,
        features_train, features_val, features_test,
        y_train_1h, y_val_1h, y_test_1h, "1h"
    )

    # Save models
    print("\n" + "="*70)
    print("SAVING ENSEMBLE MODELS")
    print("="*70)

    # Save LSTM+TCN
    torch.save({
        'model_state_dict': lstm_tcn_model.state_dict(),
        'scaler': scaler,
        'config': {
            'input_size': X_train.shape[2],
            'lstm_hidden': HIDDEN_DIM,
            'lstm_layers': NUM_LAYERS,
            'tcn_channels': TCN_CHANNELS,
            'kernel_size': KERNEL_SIZE,
            'embedding_dim': EMBEDDING_DIM,
            'dropout': DROPOUT,
        }
    }, os.path.join(output_dir, 'lstm_tcn_feature_extractor.pt'))

    # Save XGBoost models
    xgb_1d.save_model(os.path.join(output_dir, 'xgb_ensemble_1d.json'))
    xgb_4h.save_model(os.path.join(output_dir, 'xgb_ensemble_4h.json'))
    xgb_1h.save_model(os.path.join(output_dir, 'xgb_ensemble_1h.json'))

    # Save metadata
    metadata = {
        'architecture': 'LSTM+TCN → XGBoost Ensemble',
        'embedding_dim': 128,
        'statistical_features': 32,
        'total_features': 160,
        'test_performance': {
            '1d_accuracy': float(acc_1d),
            '4h_accuracy': float(acc_4h),
            '1h_accuracy': float(acc_1h)
        },
        'trained_at': datetime.utcnow().isoformat()
    }

    with open(os.path.join(output_dir, 'ensemble_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print("✓ Models saved:")
    print(f"  - lstm_tcn_feature_extractor.pt")
    print(f"  - xgb_ensemble_1d.json")
    print(f"  - xgb_ensemble_4h.json")
    print(f"  - xgb_ensemble_1h.json")
    print(f"  - ensemble_metadata.json")

    # Final results
    print("\n" + "="*70)
    print("ENSEMBLE TEST SET RESULTS (HOLDOUT)")
    print("="*70)
    print(f"1-Day Prediction:   {acc_1d:.3%}")
    print(f"4-Hour Prediction:  {acc_4h:.3%}")
    print(f"1-Hour Prediction:  {acc_1h:.3%}")
    print("="*70)

    return metadata


if __name__ == "__main__":
    main()
