#!/usr/bin/env python3
"""
Train LSTMs once on full year of data, then save for backtest use.
This prevents memory issues by training LSTMs once instead of daily.
"""

import json
import sys
import os
from typing import List, Dict
from datetime import datetime
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn

# Import LSTM models from the backtest script
sys.path.insert(0, os.path.dirname(__file__))
from no_whale_regime_backtest import (
    LongTermLSTM, ShortTermLSTM, RegimeLSTM,
    LONGTERM_SEQ_LEN, SHORTTERM_SEQ_LEN, REGIME_SEQ_LEN,
    EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS,
    load_1s_bars, extract_longterm_sequence, extract_shortterm_sequence,
    extract_regime_sequence
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_lstms_on_full_data(
    all_bars_1s: Dict[str, List[Dict]],
    all_dates: List[str],
    epochs: int = 10,
    output_dir: str = "ml/models"
):
    """
    Train all 3 LSTMs on the entire dataset at once.
    Uses CPU to avoid GPU memory issues with large dataset.
    """
    print(f"Training LSTMs on {len(all_dates)} days of data...")
    print(f"Device: {DEVICE}")

    # Force CPU for training to avoid OOM
    device = torch.device('cpu')
    print("Using CPU for training to handle large dataset")

    # Collect all sequences from all days
    print("\nBuilding sequences from all dates...")
    longterm_seqs = []
    longterm_labels = []
    shortterm_seqs = []
    shortterm_labels = []
    regime_seqs = []
    regime_labels = []

    for date_idx, date in enumerate(all_dates):
        if date not in all_bars_1s:
            continue

        bars_1s = all_bars_1s[date]
        if len(bars_1s) < 1000:
            continue

        print(f"  Processing {date} ({date_idx+1}/{len(all_dates)})...", end='\r')

        # Build 5-min bars for long-term
        bars_5min = []
        for i in range(0, len(bars_1s), 300):
            chunk = bars_1s[i:i+300]
            if len(chunk) < 60:
                continue
            bar_5min = {
                't': chunk[0]['t'],
                'o': chunk[0]['o'],
                'h': max(b['h'] for b in chunk),
                'l': min(b['l'] for b in chunk),
                'c': chunk[-1]['c'],
                'v': sum(b.get('v', 0) for b in chunk),
                'cvd': chunk[-1].get('cvd', 0)
            }
            bars_5min.append(bar_5min)

        # Build 1-min bars for regime
        bars_1min = []
        for i in range(0, len(bars_1s), 60):
            chunk = bars_1s[i:i+60]
            if len(chunk) < 20:
                continue
            bar_1min = {
                't': chunk[0]['t'],
                'o': chunk[0]['o'],
                'h': max(b['h'] for b in chunk),
                'l': min(b['l'] for b in chunk),
                'c': chunk[-1]['c'],
                'v': sum(b.get('v', 0) for b in chunk),
                'cvd': chunk[-1].get('cvd', 0)
            }
            bars_1min.append(bar_1min)

        # Calculate volume profile for regime sequences
        vp = {'poc': 0, 'vah': 0, 'val': 0}
        if bars_1s:
            prices = [b['c'] for b in bars_1s]
            vp['poc'] = np.median(prices)
            vp['vah'] = np.percentile(prices, 70)
            vp['val'] = np.percentile(prices, 30)

        # Sample sequences (don't take every single one to avoid excessive memory)
        step = max(1, len(bars_5min) // 100)  # Sample ~100 sequences per day
        for i in range(LONGTERM_SEQ_LEN, len(bars_5min), step):
            seq = extract_longterm_sequence(bars_5min, i, seq_len=LONGTERM_SEQ_LEN)
            if seq is not None:
                # Simple label: price goes up in next 5 bars
                future_idx = min(i + 5, len(bars_5min) - 1)
                label = 1 if bars_5min[future_idx]['c'] > bars_5min[i]['c'] else 0
                longterm_seqs.append(seq)
                longterm_labels.append(label)

        step = max(1, len(bars_1s) // 100)
        for i in range(SHORTTERM_SEQ_LEN, len(bars_1s), step):
            seq = extract_shortterm_sequence(bars_1s, i, seq_len=SHORTTERM_SEQ_LEN)
            if seq is not None:
                # Simple label: price goes up in next 60 seconds
                future_idx = min(i + 60, len(bars_1s) - 1)
                label = 1 if bars_1s[future_idx]['c'] > bars_1s[i]['c'] else 0
                shortterm_seqs.append(seq)
                shortterm_labels.append(label)

        step = max(1, len(bars_1min) // 50)
        for i in range(REGIME_SEQ_LEN, len(bars_1min), step):
            seq = extract_regime_sequence(bars_1min, i, vp, seq_len=REGIME_SEQ_LEN)
            if seq is not None:
                # Label: good trading opportunity (range exists)
                future_idx = min(i + 20, len(bars_1min) - 1)
                price_range = (max(b['h'] for b in bars_1min[i:future_idx+1]) -
                              min(b['l'] for b in bars_1min[i:future_idx+1]))
                label = 1 if price_range / bars_1min[i]['c'] > 0.002 else 0
                regime_seqs.append(seq)
                regime_labels.append(label)

    print(f"\n\nCollected sequences:")
    print(f"  Long-term: {len(longterm_seqs):,}")
    print(f"  Short-term: {len(shortterm_seqs):,}")
    print(f"  Regime: {len(regime_seqs):,}")

    # Initialize models on CPU
    print("\nInitializing models...")
    longterm_lstm = LongTermLSTM(
        input_dim=7, hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS, embedding_dim=EMBEDDING_DIM
    ).to(device)

    shortterm_lstm = ShortTermLSTM(
        input_dim=9, hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS, embedding_dim=EMBEDDING_DIM
    ).to(device)

    regime_lstm = RegimeLSTM(
        input_dim=18, hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS, embedding_dim=EMBEDDING_DIM
    ).to(device)

    # Train long-term LSTM
    if len(longterm_seqs) > 100:
        print("\nTraining Long-term LSTM...")
        X_lt = torch.tensor(np.array(longterm_seqs), dtype=torch.float32).to(device)
        y_lt = torch.tensor(np.array(longterm_labels), dtype=torch.long).to(device)

        classifier = nn.Linear(EMBEDDING_DIM, 2).to(device)
        optimizer = torch.optim.Adam(
            list(longterm_lstm.parameters()) + list(classifier.parameters()),
            lr=0.001
        )
        criterion = nn.CrossEntropyLoss()

        longterm_lstm.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            embeddings = longterm_lstm(X_lt)
            logits = classifier(embeddings)
            loss = criterion(logits, y_lt)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 2 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

        del X_lt, y_lt, classifier, optimizer

    # Train short-term LSTM
    if len(shortterm_seqs) > 100:
        print("\nTraining Short-term LSTM...")
        X_st = torch.tensor(np.array(shortterm_seqs), dtype=torch.float32).to(device)
        y_st = torch.tensor(np.array(shortterm_labels), dtype=torch.long).to(device)

        classifier = nn.Linear(EMBEDDING_DIM, 2).to(device)
        optimizer = torch.optim.Adam(
            list(shortterm_lstm.parameters()) + list(classifier.parameters()),
            lr=0.001
        )
        criterion = nn.CrossEntropyLoss()

        shortterm_lstm.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            embeddings = shortterm_lstm(X_st)
            logits = classifier(embeddings)
            loss = criterion(logits, y_st)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 2 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

        del X_st, y_st, classifier, optimizer

    # Train regime LSTM
    if len(regime_seqs) > 100:
        print("\nTraining Regime LSTM...")
        X_reg = torch.tensor(np.array(regime_seqs), dtype=torch.float32).to(device)
        y_reg = torch.tensor(np.array(regime_labels), dtype=torch.long).to(device)

        classifier = nn.Linear(EMBEDDING_DIM, 2).to(device)
        optimizer = torch.optim.Adam(
            list(regime_lstm.parameters()) + list(classifier.parameters()),
            lr=0.001
        )
        criterion = nn.CrossEntropyLoss()

        regime_lstm.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            embeddings = regime_lstm(X_reg)
            logits = classifier(embeddings)
            loss = criterion(logits, y_reg)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 2 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

        del X_reg, y_reg, classifier, optimizer

    # Save models
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nSaving models to {output_dir}...")
    torch.save(longterm_lstm.state_dict(), os.path.join(output_dir, 'longterm_lstm.pt'))
    torch.save(shortterm_lstm.state_dict(), os.path.join(output_dir, 'shortterm_lstm.pt'))
    torch.save(regime_lstm.state_dict(), os.path.join(output_dir, 'regime_lstm.pt'))

    print("\nDone! Models saved:")
    print(f"  {output_dir}/longterm_lstm.pt")
    print(f"  {output_dir}/shortterm_lstm.pt")
    print(f"  {output_dir}/regime_lstm.pt")

    return longterm_lstm, shortterm_lstm, regime_lstm


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train LSTMs on full year')
    parser.add_argument('--bars', required=True, help='Path to 1s bars JSON')
    parser.add_argument('--output-dir', default='ml/models/full_year_lstms',
                       help='Directory to save trained models')
    parser.add_argument('--epochs', type=int, default=10, help='Training epochs')
    args = parser.parse_args()

    print("=" * 70)
    print("TRAINING LSTMs ON FULL YEAR")
    print("=" * 70)

    print("\nLoading bars data...")
    all_bars = load_1s_bars(args.bars)

    all_bars_1s = defaultdict(list)
    for bar in all_bars:
        ts = bar['t']
        if isinstance(ts, str):
            date = ts[:10]
        else:
            date = ts.strftime('%Y-%m-%d')
        all_bars_1s[date].append(bar)

    dates = sorted(all_bars_1s.keys())
    print(f"Found {len(dates)} dates: {dates[0]} to {dates[-1]}")

    train_lstms_on_full_data(all_bars_1s, dates, epochs=args.epochs,
                            output_dir=args.output_dir)


if __name__ == "__main__":
    main()
