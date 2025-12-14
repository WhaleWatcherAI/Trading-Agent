#!/usr/bin/env python3
"""
Improved LSTM training with 5-minute bars and proper data handling.
No leakage, proper validation, better predictions.
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "ml" / "data"
MODELS_DIR = ROOT / "ml" / "models"

# Configuration
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.001
HIDDEN_SIZE = 128
NUM_LAYERS = 3
DROPOUT = 0.3
PATIENCE = 15

class TradingDataset(Dataset):
    """Dataset for LSTM training with pre-prepared data."""

    def __init__(self, X, y):
        """X shape: (samples, timesteps, features), y shape: (samples, 2)"""
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class ImprovedLSTM(nn.Module):
    """Improved LSTM for trading with better architecture."""

    def __init__(self, input_size, hidden_size, num_layers, dropout=0.3):
        super(ImprovedLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM with dropout between layers
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False  # No bidirectional to avoid future leakage
        )

        # Attention mechanism
        self.attention = nn.Linear(hidden_size, 1)

        # Output heads
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Separate heads for signal and movement
        self.signal_head = nn.Linear(hidden_size // 4, 3)  # Buy/Hold/Sell
        self.movement_head = nn.Linear(hidden_size // 4, 1)  # Expected return

    def forward(self, x):
        # LSTM forward
        lstm_out, (hidden, cell) = self.lstm(x)

        # Apply attention
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attention_weights * lstm_out, dim=1)

        # Get features
        features = self.fc(context)

        # Get predictions
        signals = self.signal_head(features)
        movement = self.movement_head(features)

        return signals, movement

def train_model(model, train_loader, val_loader, epochs=100, patience=15):
    """Train the LSTM model with early stopping."""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Loss functions
    criterion_signal = nn.CrossEntropyLoss()
    criterion_movement = nn.MSELoss()

    # Optimizer with scheduler
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5
    )

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            # Forward pass
            signal_pred, movement_pred = model(batch_X)

            # Split targets
            signal_target = batch_y[:, 0].long()
            movement_target = batch_y[:, 1].unsqueeze(1)

            # Calculate losses
            loss_signal = criterion_signal(signal_pred, signal_target)
            loss_movement = criterion_movement(movement_pred, movement_target)

            # Combined loss (weighted)
            total_loss = loss_signal + 0.1 * loss_movement

            # Backward pass
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Track metrics
            train_loss += total_loss.item()
            _, predicted = torch.max(signal_pred, 1)
            train_correct += (predicted == signal_target).sum().item()
            train_total += signal_target.size(0)

        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)

                signal_pred, movement_pred = model(batch_X)

                signal_target = batch_y[:, 0].long()
                movement_target = batch_y[:, 1].unsqueeze(1)

                loss_signal = criterion_signal(signal_pred, signal_target)
                loss_movement = criterion_movement(movement_pred, movement_target)

                total_loss = loss_signal + 0.1 * loss_movement

                val_loss += total_loss.item()
                _, predicted = torch.max(signal_pred, 1)
                val_correct += (predicted == signal_target).sum().item()
                val_total += signal_target.size(0)

        # Calculate epoch metrics
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total

        # Update history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        # Update scheduler
        scheduler.step(avg_val_loss)

        # Print progress
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {avg_train_loss:.4f}, Acc: {train_acc:.3%}")
            print(f"  Val Loss: {avg_val_loss:.4f}, Acc: {val_acc:.3%}")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), MODELS_DIR / 'lstm_improved.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    return history, best_val_loss

def evaluate_on_test(model, test_loader):
    """Evaluate model on test set."""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    all_predictions = []
    all_targets = []
    all_movements = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            signal_pred, movement_pred = model(batch_X)

            _, predicted = torch.max(signal_pred, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(batch_y[:, 0].cpu().numpy())
            all_movements.extend(movement_pred.cpu().numpy().flatten())

    # Calculate metrics
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    all_movements = np.array(all_movements)

    accuracy = np.mean(all_predictions == all_targets)

    # Confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(all_targets, all_predictions)

    # Per-class accuracy
    signal_map = {0: 'Hold', 1: 'Buy', 2: 'Sell'}
    print("\n📊 Test Set Performance:")
    print(f"  Overall Accuracy: {accuracy:.3%}")
    print("\n  Confusion Matrix:")
    print("         Pred Hold  Buy  Sell")
    for i in range(3):
        print(f"  True {signal_map[i]:4s}: {cm[i][0]:4d} {cm[i][1]:4d} {cm[i][2]:4d}")

    # Movement prediction accuracy
    movement_mae = np.mean(np.abs(all_movements - batch_y[:, 1].cpu().numpy()[:len(all_movements)]))
    print(f"\n  Movement MAE: {movement_mae:.3f}%")

    return accuracy, cm, all_predictions

def main():
    print("\n" + "="*70)
    print("🚀 TRAINING IMPROVED LSTM WITH 5-MINUTE DATA")
    print("="*70)

    # Load prepared data
    data = np.load(DATA_DIR / 'prepared_5min_data.npz')
    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']
    X_test = data['X_test']
    y_test = data['y_test']

    print(f"\n📊 Data loaded:")
    print(f"  Train: {X_train.shape}")
    print(f"  Val: {X_val.shape}")
    print(f"  Test: {X_test.shape}")

    # Normalize features (fit only on training data!)
    n_features = X_train.shape[2]
    scaler = StandardScaler()

    # Reshape for scaling
    X_train_flat = X_train.reshape(-1, n_features)
    X_val_flat = X_val.reshape(-1, n_features)
    X_test_flat = X_test.reshape(-1, n_features)

    # Fit scaler ONLY on training data
    X_train_scaled = scaler.fit_transform(X_train_flat)
    X_val_scaled = scaler.transform(X_val_flat)  # Use fitted scaler
    X_test_scaled = scaler.transform(X_test_flat)  # Use fitted scaler

    # Reshape back
    X_train = X_train_scaled.reshape(X_train.shape)
    X_val = X_val_scaled.reshape(X_val.shape)
    X_test = X_test_scaled.reshape(X_test.shape)

    print("✅ Features normalized (scaler fit on training data only)")

    # Create datasets
    train_dataset = TradingDataset(X_train, y_train)
    val_dataset = TradingDataset(X_val, y_val)
    test_dataset = TradingDataset(X_test, y_test)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize model
    model = ImprovedLSTM(
        input_size=n_features,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    )

    print(f"\n🧠 Model architecture:")
    print(f"  Input size: {n_features}")
    print(f"  Hidden size: {HIDDEN_SIZE}")
    print(f"  Layers: {NUM_LAYERS}")
    print(f"  Dropout: {DROPOUT}")

    # Train model
    print(f"\n🎯 Training for {EPOCHS} epochs...")
    history, best_val_loss = train_model(
        model, train_loader, val_loader,
        epochs=EPOCHS, patience=PATIENCE
    )

    # Load best model
    model.load_state_dict(torch.load(MODELS_DIR / 'lstm_improved.pth'))

    # Evaluate on test set
    test_acc, confusion, predictions = evaluate_on_test(model, test_loader)

    # Save scaler
    import pickle
    with open(MODELS_DIR / 'lstm_improved_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    # Save metrics
    metrics = {
        'model_type': 'LSTM_Improved_5min',
        'data_config': {
            'timeframe': '5min',
            'feature_window': 60,
            'prediction_horizon': 20,
            'gap': 5,
            'n_features': n_features
        },
        'architecture': {
            'hidden_size': HIDDEN_SIZE,
            'num_layers': NUM_LAYERS,
            'dropout': DROPOUT
        },
        'training': {
            'epochs_trained': len(history['train_loss']),
            'best_val_loss': float(best_val_loss),
            'final_train_acc': float(history['train_acc'][-1]),
            'final_val_acc': float(history['val_acc'][-1])
        },
        'test_performance': {
            'accuracy': float(test_acc),
            'confusion_matrix': confusion.tolist()
        },
        'data_splits': {
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test)
        },
        'trained_at': datetime.utcnow().isoformat()
    }

    with open(MODELS_DIR / 'lstm_improved_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "="*70)
    print("✅ LSTM TRAINING COMPLETE!")
    print("="*70)
    print(f"  Best Val Loss: {best_val_loss:.4f}")
    print(f"  Test Accuracy: {test_acc:.3%}")
    print(f"\n📁 Files saved:")
    print(f"  - lstm_improved.pth (model weights)")
    print(f"  - lstm_improved_scaler.pkl (feature scaler)")
    print(f"  - lstm_improved_metrics.json (performance metrics)")

    return model, metrics

if __name__ == "__main__":
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model, metrics = main()