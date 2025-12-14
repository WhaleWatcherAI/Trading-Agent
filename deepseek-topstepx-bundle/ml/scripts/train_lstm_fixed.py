#!/usr/bin/env python3
"""
Fixed LSTM training script with proper data handling.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "ml" / "data"
MODELS_DIR = ROOT / "ml" / "models"

# Configuration
SEQUENCE_LENGTH = 60
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.2


class TradingDataset(Dataset):
    """Dataset for LSTM training."""

    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


class SimpleLSTM(nn.Module):
    """Simplified LSTM for trading predictions."""

    def __init__(self, input_size, hidden_size, num_layers, output_size=3):
        super(SimpleLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=DROPOUT if num_layers > 1 else 0
        )

        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden_size // 2, output_size)
        )

        # Separate heads
        self.signal_head = nn.Linear(output_size, 3)  # Buy/Sell/Hold
        self.movement_head = nn.Linear(output_size, 1)  # Price movement

    def forward(self, x):
        # LSTM forward
        lstm_out, _ = self.lstm(x)

        # Take last output
        last_output = lstm_out[:, -1, :]

        # Get features
        features = self.fc(last_output)

        # Get predictions
        signals = torch.softmax(self.signal_head(features), dim=-1)
        movement = self.movement_head(features)

        return signals, movement


def prepare_lstm_data():
    """Prepare clean data for LSTM training."""
    print("\n📊 Preparing LSTM data...")

    # Load SPY data if available, otherwise use regular data
    spy_path = DATA_DIR / 'spy_real_data.parquet'
    regular_path = DATA_DIR / 'market_data.parquet'

    if spy_path.exists():
        df = pd.read_parquet(spy_path)
        print(f"Using real SPY data: {len(df)} rows")
    elif regular_path.exists():
        df = pd.read_parquet(regular_path)
        print(f"Using market data: {len(df)} rows")
    else:
        print("No data found!")
        return None, None, None, None

    # Select key features
    feature_cols = ['close', 'volume', 'rsi', 'macd', 'volatility']

    # Ensure columns exist
    for col in feature_cols:
        if col not in df.columns:
            if col == 'close':
                df['close'] = df.get('Close', 100)
            elif col == 'volume':
                df['volume'] = df.get('Volume', 1000)
            elif col == 'rsi':
                df['rsi'] = df.get('RSI', 50)
            elif col == 'macd':
                df['macd'] = df.get('MACD', 0)
            elif col == 'volatility':
                df['volatility'] = df.get('Volatility', 0.01)

    # Clean data - remove NaN and inf values
    print("Cleaning data...")
    for col in feature_cols:
        if col in df.columns:
            # Replace inf with NaN first
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            # Fill NaN with column mean
            df[col] = df[col].fillna(df[col].mean())
            # If still NaN (all values were NaN), fill with 0
            df[col] = df[col].fillna(0)

    # Normalize features
    scaler = StandardScaler()
    df_scaled = df[feature_cols].copy()
    df_scaled[feature_cols] = scaler.fit_transform(df_scaled[feature_cols])

    # Save scaler
    import pickle
    with open(MODELS_DIR / 'lstm_scaler_fixed.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    # Create sequences
    sequences = []
    targets = []

    for i in range(SEQUENCE_LENGTH, len(df_scaled) - 5):
        # Get sequence
        seq = df_scaled.iloc[i-SEQUENCE_LENGTH:i][feature_cols].values

        # Calculate target (next 5 periods)
        current_price = df.iloc[i]['close']
        future_price = df.iloc[min(i+5, len(df)-1)]['close']

        # Price movement
        price_change = (future_price - current_price) / (current_price + 1e-8)

        # Signal: 0=hold, 1=buy, 2=sell
        if price_change > 0.001:  # 0.1% up
            signal = 1  # Buy
        elif price_change < -0.001:  # 0.1% down
            signal = 2  # Sell
        else:
            signal = 0  # Hold

        sequences.append(seq)
        targets.append([signal, price_change * 100])  # Scale movement to percentage

    sequences = np.array(sequences)
    targets = np.array(targets)

    print(f"Created {len(sequences)} sequences")
    print(f"Sequence shape: {sequences.shape}")
    print(f"Target shape: {targets.shape}")

    # Check for NaN in prepared data
    if np.isnan(sequences).any():
        print("Warning: NaN found in sequences, cleaning...")
        sequences = np.nan_to_num(sequences, 0)

    if np.isnan(targets).any():
        print("Warning: NaN found in targets, cleaning...")
        targets = np.nan_to_num(targets, 0)

    return sequences, targets, scaler, feature_cols


def train_lstm():
    """Train the LSTM model with clean data."""
    print("\n🚀 Training Fixed LSTM Model...")

    # Prepare data
    sequences, targets, scaler, feature_cols = prepare_lstm_data()

    if sequences is None:
        print("No data available for training!")
        return None

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        sequences, targets, test_size=0.2, shuffle=False
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    # Create datasets
    train_dataset = TradingDataset(X_train, y_train)
    test_dataset = TradingDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize model
    model = SimpleLSTM(
        input_size=len(feature_cols),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS
    )

    # Loss functions
    criterion_signal = nn.CrossEntropyLoss()
    criterion_movement = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

    # Training loop
    train_losses = []
    test_losses = []
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(EPOCHS):
        # Training
        model.train()
        epoch_train_loss = []

        for batch_seq, batch_target in train_loader:
            optimizer.zero_grad()

            # Get predictions
            signals, movement = model(batch_seq)

            # Calculate losses
            signal_targets = batch_target[:, 0].long()
            movement_targets = batch_target[:, 1].unsqueeze(1)

            loss_signal = criterion_signal(signals, signal_targets)
            loss_movement = criterion_movement(movement, movement_targets)

            total_loss = loss_signal + loss_movement * 0.1  # Weight movement less

            # Check for NaN
            if torch.isnan(total_loss):
                print(f"NaN loss detected at epoch {epoch}, skipping batch")
                continue

            total_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            epoch_train_loss.append(total_loss.item())

        # Validation
        model.eval()
        epoch_test_loss = []
        correct_signals = 0
        total_signals = 0

        with torch.no_grad():
            for batch_seq, batch_target in test_loader:
                signals, movement = model(batch_seq)

                signal_targets = batch_target[:, 0].long()
                movement_targets = batch_target[:, 1].unsqueeze(1)

                loss_signal = criterion_signal(signals, signal_targets)
                loss_movement = criterion_movement(movement, movement_targets)

                total_loss = loss_signal + loss_movement * 0.1
                epoch_test_loss.append(total_loss.item())

                # Calculate accuracy
                _, predicted = torch.max(signals, 1)
                correct_signals += (predicted == signal_targets).sum().item()
                total_signals += signal_targets.size(0)

        # Calculate metrics
        avg_train_loss = np.mean(epoch_train_loss) if epoch_train_loss else float('inf')
        avg_test_loss = np.mean(epoch_test_loss) if epoch_test_loss else float('inf')
        accuracy = correct_signals / total_signals if total_signals > 0 else 0

        train_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)

        # Update scheduler
        scheduler.step(avg_test_loss)

        # Print progress
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{EPOCHS}")
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  Test Loss: {avg_test_loss:.4f}")
            print(f"  Signal Accuracy: {accuracy:.2%}")

        # Early stopping
        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), MODELS_DIR / 'lstm_fixed.pth')
        else:
            patience_counter += 1
            if patience_counter >= 10:
                print(f"Early stopping at epoch {epoch}")
                break

    # Final metrics
    print("\n" + "="*50)
    print("LSTM TRAINING COMPLETE!")
    print("="*50)
    print(f"Best Test Loss: {best_loss:.4f}")
    print(f"Final Accuracy: {accuracy:.2%}")

    # Save training history
    metrics = {
        'model_type': 'LSTM_Fixed',
        'epochs_trained': len(train_losses),
        'best_loss': float(best_loss),
        'final_accuracy': float(accuracy),
        'sequence_length': SEQUENCE_LENGTH,
        'hidden_size': HIDDEN_SIZE,
        'num_layers': NUM_LAYERS,
        'features': feature_cols,
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'trained_at': datetime.utcnow().isoformat()
    }

    with open(MODELS_DIR / 'lstm_metrics_fixed.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\nModel saved to: {MODELS_DIR / 'lstm_fixed.pth'}")
    print(f"Metrics saved to: {MODELS_DIR / 'lstm_metrics_fixed.json'}")

    return model, metrics


if __name__ == "__main__":
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model, metrics = train_lstm()

    if model is not None:
        print("\n✅ LSTM successfully trained and ready for predictions!")