#!/usr/bin/env python3
"""
LSTM-based trading model for sequential pattern learning.
Processes time series data to predict entry signals and price movements.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "ml" / "data"
MODELS_DIR = ROOT / "ml" / "models"
LSTM_MODEL_PATH = MODELS_DIR / "lstm_trading_model.pth"
SCALER_PATH = MODELS_DIR / "lstm_scaler.pkl"

# Configuration
SEQUENCE_LENGTH = 60  # 60 minutes of data for prediction
PREDICTION_HORIZON = 5  # Predict 5 minutes ahead
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.2


class TradingDataset(Dataset):
    """Custom dataset for trading time series data."""

    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


class LSTMTradingModel(nn.Module):
    """
    LSTM model for trading predictions.
    Learns temporal patterns in market data to predict:
    1. Entry signals (buy/sell/hold)
    2. Expected price movement
    3. Risk assessment
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                 output_size: int, dropout: float = 0.2):
        super(LSTMTradingModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layers with dropout for regularization
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True  # Bidirectional for better context
        )

        # Attention mechanism for focusing on important time steps
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

        # Output layers
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )

        # Separate heads for different predictions
        self.signal_head = nn.Linear(output_size, 3)  # Buy/Sell/Hold
        self.movement_head = nn.Linear(output_size, 1)  # Price movement
        self.risk_head = nn.Linear(output_size, 1)  # Risk score

    def forward(self, x):
        # LSTM processing
        lstm_out, _ = self.lstm(x)

        # Apply attention
        attention_weights = torch.softmax(
            self.attention(lstm_out).squeeze(-1),
            dim=1
        )
        attended = torch.sum(
            lstm_out * attention_weights.unsqueeze(-1),
            dim=1
        )

        # Main features
        features = self.fc_layers(attended)

        # Multiple outputs
        signals = torch.softmax(self.signal_head(features), dim=-1)
        movement = self.movement_head(features)
        risk = torch.sigmoid(self.risk_head(features))

        return signals, movement, risk


def prepare_sequences(df: pd.DataFrame, feature_cols: List[str],
                      sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare sequential data for LSTM training.
    Creates overlapping windows of historical data.
    """
    sequences = []
    targets = []

    # Sort by time
    df_sorted = df.sort_values('entry_time')

    # Get feature values
    feature_data = df_sorted[feature_cols].values

    # Create sequences
    for i in range(sequence_length, len(feature_data) - PREDICTION_HORIZON):
        seq = feature_data[i - sequence_length:i]

        # Multi-task targets
        # Signal: 0=hold, 1=buy, 2=sell (based on future price movement)
        future_prices = feature_data[i:i + PREDICTION_HORIZON]
        price_change = np.mean(future_prices[:, 0]) - feature_data[i, 0]

        if abs(price_change) < 0.001:  # Threshold for hold
            signal = 0
        elif price_change > 0:
            signal = 1
        else:
            signal = 2

        # Movement: normalized price change
        movement = price_change / (feature_data[i, 0] + 1e-8)

        # Risk: volatility in next period
        risk = np.std(future_prices[:, 0]) / (np.mean(future_prices[:, 0]) + 1e-8)

        sequences.append(seq)
        targets.append([signal, movement, risk])

    return np.array(sequences), np.array(targets)


def train_lstm_model(df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, float]:
    """Train the LSTM model with time series validation."""

    # Prepare data
    print("Preparing sequences...")
    sequences, targets = prepare_sequences(df, feature_cols, SEQUENCE_LENGTH)

    if len(sequences) < 100:
        raise ValueError(f"Insufficient data: only {len(sequences)} sequences")

    # Normalize features
    scaler = StandardScaler()
    sequences_reshaped = sequences.reshape(-1, sequences.shape[-1])
    sequences_scaled = scaler.fit_transform(sequences_reshaped)
    sequences_scaled = sequences_scaled.reshape(sequences.shape)

    # Time series split for validation
    tscv = TimeSeriesSplit(n_splits=3)
    val_losses = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(sequences_scaled)):
        print(f"\nTraining fold {fold + 1}/3...")

        # Split data
        X_train = sequences_scaled[train_idx]
        y_train = targets[train_idx]
        X_val = sequences_scaled[val_idx]
        y_val = targets[val_idx]

        # Create datasets
        train_dataset = TradingDataset(X_train, y_train)
        val_dataset = TradingDataset(X_val, y_val)

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False  # Keep temporal order
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False
        )

        # Initialize model
        model = LSTMTradingModel(
            input_size=len(feature_cols),
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            output_size=64,
            dropout=DROPOUT
        )

        # Loss functions
        criterion_signal = nn.CrossEntropyLoss()
        criterion_movement = nn.MSELoss()
        criterion_risk = nn.MSELoss()

        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=5,
            factor=0.5
        )

        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(EPOCHS):
            # Training
            model.train()
            train_losses = []

            for batch_seq, batch_target in train_loader:
                optimizer.zero_grad()

                signals, movement, risk = model(batch_seq)

                # Multi-task loss
                loss_signal = criterion_signal(
                    signals,
                    batch_target[:, 0].long()
                )
                loss_movement = criterion_movement(
                    movement.squeeze(),
                    batch_target[:, 1]
                )
                loss_risk = criterion_risk(
                    risk.squeeze(),
                    batch_target[:, 2]
                )

                total_loss = loss_signal + loss_movement + loss_risk
                total_loss.backward()

                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                train_losses.append(total_loss.item())

            # Validation
            model.eval()
            val_loss_epoch = []

            with torch.no_grad():
                for batch_seq, batch_target in val_loader:
                    signals, movement, risk = model(batch_seq)

                    loss_signal = criterion_signal(
                        signals,
                        batch_target[:, 0].long()
                    )
                    loss_movement = criterion_movement(
                        movement.squeeze(),
                        batch_target[:, 1]
                    )
                    loss_risk = criterion_risk(
                        risk.squeeze(),
                        batch_target[:, 2]
                    )

                    total_loss = loss_signal + loss_movement + loss_risk
                    val_loss_epoch.append(total_loss.item())

            avg_val_loss = np.mean(val_loss_epoch)
            scheduler.step(avg_val_loss)

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss: {np.mean(train_losses):.4f}, "
                      f"Val Loss: {avg_val_loss:.4f}")

            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), LSTM_MODEL_PATH)
            else:
                patience_counter += 1
                if patience_counter > 10:
                    print(f"Early stopping at epoch {epoch}")
                    break

        val_losses.append(best_val_loss)

    # Save scaler
    import pickle
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)

    metrics = {
        "model_type": "LSTM",
        "sequence_length": SEQUENCE_LENGTH,
        "prediction_horizon": PREDICTION_HORIZON,
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        "avg_val_loss": float(np.mean(val_losses)),
        "std_val_loss": float(np.std(val_losses)),
        "total_sequences": len(sequences),
        "features": len(feature_cols),
        "trained_at": datetime.utcnow().isoformat()
    }

    return metrics


def main():
    # Load data
    parquet_path = DATA_DIR / "meta_label.parquet"
    if not parquet_path.exists():
        print("Dataset not found. Run build_dataset.py first.")
        sys.exit(1)

    df = pd.read_parquet(parquet_path)
    if len(df) < 200:
        print(f"Insufficient data: {len(df)} rows. Need at least 200.")
        sys.exit(1)

    # Get feature columns
    feature_cols = [c for c in df.columns
                   if c not in ("symbol", "entry_time", "win_5m", "win_30m")]

    print(f"Training LSTM model with {len(df)} rows, {len(feature_cols)} features")

    # Ensure models directory exists
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Train model
    metrics = train_lstm_model(df, feature_cols)

    # Save metrics
    metrics_path = MODELS_DIR / "lstm_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print(f"\nModel saved to: {LSTM_MODEL_PATH}")
    print(f"Metrics: {json.dumps(metrics, indent=2)}")


if __name__ == "__main__":
    main()