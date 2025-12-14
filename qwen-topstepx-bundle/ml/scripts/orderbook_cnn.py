#!/usr/bin/env python3
"""
Order Book CNN Pattern Recognition

Detects L2 patterns from MBP-10 snapshots:
- Bid/Ask walls (large orders at specific levels)
- Absorption (volume absorption without price change)
- Exhaustion (decreasing volume at extremes)
- Imbalance shifts
- Spread dynamics

Architecture:
  Input: Order book snapshot (10 bids + 10 asks × [price, size])
  → Conv1D layers to detect spatial patterns
  → LSTM to track temporal evolution
  → Output: Pattern embeddings for ensemble
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple
from collections import deque


class OrderBookCNN(nn.Module):
    """
    CNN for detecting order book patterns.

    Input shape: (batch, sequence_len, 40)
      - 10 bids: [price_0, size_0, price_1, size_1, ...]
      - 10 asks: [price_0, size_0, price_1, size_1, ...]

    Output: (batch, embedding_dim) pattern embeddings
    """

    def __init__(self, embedding_dim=32, hidden_dim=64):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Normalize order book (prices are absolute, need relative)
        # Conv layers to detect patterns across price levels
        self.conv1 = nn.Conv1d(
            in_channels=40,  # 20 levels × 2 (price, size)
            out_channels=64,
            kernel_size=3,
            padding=1
        )
        self.conv2 = nn.Conv1d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            padding=1
        )
        self.conv3 = nn.Conv1d(
            in_channels=128,
            out_channels=64,
            kernel_size=3,
            padding=1
        )

        # LSTM to capture temporal evolution
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )

        # Output projection
        self.fc = nn.Linear(hidden_dim, embedding_dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, 40) order book sequences

        Returns:
            (batch, embedding_dim) pattern embeddings
        """
        batch_size, seq_len, features = x.shape

        # Reshape for conv: (batch × seq, features, 1)
        x = x.view(batch_size * seq_len, features, 1)

        # Conv layers
        x = self.relu(self.conv1(x))
        x = self.dropout(x)
        x = self.relu(self.conv2(x))
        x = self.dropout(x)
        x = self.relu(self.conv3(x))  # (batch × seq, 64, 1)

        # Reshape for LSTM: (batch, seq, 64)
        x = x.squeeze(-1).view(batch_size, seq_len, 64)

        # LSTM
        _, (h_n, _) = self.lstm(x)  # h_n: (2, batch, hidden_dim)
        x = h_n[-1]  # Take last layer: (batch, hidden_dim)

        # Output
        x = self.fc(x)  # (batch, embedding_dim)
        return x


class OrderBookFeatureExtractor:
    """
    Extract order book pattern features from L2 snapshots.

    Features:
    1. Bid/Ask Walls: Large orders ≥3x average size
    2. Absorption: Volume removed without price change
    3. Exhaustion: Decreasing volume at higher price levels
    4. Imbalance: Bid vs Ask volume ratio
    5. Spread dynamics: Widening/tightening
    """

    def __init__(self, window_size=30):
        """
        Args:
            window_size: Number of snapshots to track for pattern detection
        """
        self.window_size = window_size
        self.history = deque(maxlen=window_size)

    def extract_snapshot_features(self, snapshot: Dict) -> np.ndarray:
        """
        Convert single L2 snapshot to feature vector.

        Args:
            snapshot: {
                "bids": [{"price": float, "size": int}, ...],  # 10 levels
                "asks": [{"price": float, "size": int}, ...],
                "spread": float,
                "mid": float,
                "bid_depth": int,
                "ask_depth": int,
                "imbalance": float
            }

        Returns:
            (40,) array: [bid_0_price, bid_0_size, ..., ask_9_price, ask_9_size]
        """
        features = np.zeros(40, dtype=np.float32)

        bids = snapshot.get("bids", [])
        asks = snapshot.get("asks", [])
        mid = snapshot.get("mid", 0)

        if mid == 0:
            return features

        # Normalize prices relative to mid
        for i, bid in enumerate(bids[:10]):
            features[i * 2] = (bid["price"] - mid) / mid  # Relative price
            features[i * 2 + 1] = bid["size"]  # Absolute size

        for i, ask in enumerate(asks[:10]):
            features[20 + i * 2] = (ask["price"] - mid) / mid
            features[20 + i * 2 + 1] = ask["size"]

        return features

    def detect_walls(self, snapshot: Dict) -> Tuple[List[int], List[int]]:
        """
        Detect bid/ask walls (large orders ≥3x average).

        Returns:
            (bid_wall_levels, ask_wall_levels): Lists of level indices with walls
        """
        bids = snapshot.get("bids", [])
        asks = snapshot.get("asks", [])

        if len(bids) < 3 or len(asks) < 3:
            return [], []

        # Calculate average size
        bid_avg = np.mean([b["size"] for b in bids])
        ask_avg = np.mean([a["size"] for a in asks])

        # Find walls (≥3x average)
        bid_walls = [i for i, b in enumerate(bids) if b["size"] >= 3 * bid_avg]
        ask_walls = [i for i, a in enumerate(asks) if a["size"] >= 3 * ask_avg]

        return bid_walls, ask_walls

    def detect_absorption(self, current: Dict, previous: Dict) -> Dict[str, float]:
        """
        Detect absorption (volume removed without price change).

        Returns:
            {
                "bid_absorption": float,  # Volume removed from bids
                "ask_absorption": float,  # Volume removed from asks
            }
        """
        if not previous:
            return {"bid_absorption": 0, "ask_absorption": 0}

        curr_bids = {b["price"]: b["size"] for b in current.get("bids", [])}
        prev_bids = {b["price"]: b["size"] for b in previous.get("bids", [])}

        curr_asks = {a["price"]: a["size"] for a in current.get("asks", [])}
        prev_asks = {a["price"]: a["size"] for a in previous.get("asks", [])}

        # Calculate volume removed at same prices
        bid_absorption = sum(
            prev_bids.get(price, 0) - curr_bids.get(price, 0)
            for price in prev_bids
            if curr_bids.get(price, 0) < prev_bids.get(price, 0)
        )

        ask_absorption = sum(
            prev_asks.get(price, 0) - curr_asks.get(price, 0)
            for price in prev_asks
            if curr_asks.get(price, 0) < prev_asks.get(price, 0)
        )

        return {
            "bid_absorption": max(0, bid_absorption),
            "ask_absorption": max(0, ask_absorption),
        }

    def detect_exhaustion(self, snapshot: Dict) -> Dict[str, bool]:
        """
        Detect exhaustion (decreasing volume away from best price).

        Returns:
            {
                "bid_exhaustion": bool,  # Bids getting thinner
                "ask_exhaustion": bool,  # Asks getting thinner
            }
        """
        bids = snapshot.get("bids", [])
        asks = snapshot.get("asks", [])

        if len(bids) < 5 or len(asks) < 5:
            return {"bid_exhaustion": False, "ask_exhaustion": False}

        # Check if volume decreases monotonically
        bid_sizes = [b["size"] for b in bids[:5]]
        ask_sizes = [a["size"] for a in asks[:5]]

        bid_exhaustion = all(bid_sizes[i] > bid_sizes[i+1] for i in range(4))
        ask_exhaustion = all(ask_sizes[i] > ask_sizes[i+1] for i in range(4))

        return {
            "bid_exhaustion": bid_exhaustion,
            "ask_exhaustion": ask_exhaustion,
        }

    def get_advanced_features(self, snapshot: Dict) -> Dict[str, float]:
        """
        Calculate advanced order book features.

        Returns:
            {
                "has_bid_wall": float (0/1),
                "has_ask_wall": float (0/1),
                "bid_wall_distance": float (levels from best),
                "ask_wall_distance": float,
                "bid_exhaustion": float (0/1),
                "ask_exhaustion": float (0/1),
                "imbalance": float,
                "spread_bps": float (basis points),
                "depth_ratio": float (bid_depth / ask_depth),
            }
        """
        bid_walls, ask_walls = self.detect_walls(snapshot)
        exhaustion = self.detect_exhaustion(snapshot)

        mid = snapshot.get("mid", 0)
        spread = snapshot.get("spread", 0)
        bid_depth = snapshot.get("bid_depth", 1)
        ask_depth = snapshot.get("ask_depth", 1)

        return {
            "has_bid_wall": float(len(bid_walls) > 0),
            "has_ask_wall": float(len(ask_walls) > 0),
            "bid_wall_distance": float(min(bid_walls)) if bid_walls else 10.0,
            "ask_wall_distance": float(min(ask_walls)) if ask_walls else 10.0,
            "bid_exhaustion": float(exhaustion["bid_exhaustion"]),
            "ask_exhaustion": float(exhaustion["ask_exhaustion"]),
            "imbalance": snapshot.get("imbalance", 0),
            "spread_bps": (spread / mid * 10000) if mid > 0 else 0,
            "depth_ratio": bid_depth / max(ask_depth, 1),
        }

    def build_sequence(self, snapshots: List[Dict], seq_len=30) -> np.ndarray:
        """
        Build order book sequence for CNN input.

        Args:
            snapshots: List of L2 snapshots (most recent last)
            seq_len: Sequence length

        Returns:
            (seq_len, 40) array of order book features
        """
        # Pad if not enough snapshots
        if len(snapshots) < seq_len:
            padding = [snapshots[0]] * (seq_len - len(snapshots))
            snapshots = padding + snapshots
        else:
            snapshots = snapshots[-seq_len:]

        sequence = np.array([
            self.extract_snapshot_features(snap)
            for snap in snapshots
        ])

        return sequence  # (seq_len, 40)


def train_orderbook_cnn(
    snapshot_files: List[str],
    model_path: str,
    epochs: int = 10,
    device: str = "cuda"
):
    """
    Train the order book CNN on historical L2 data.

    Args:
        snapshot_files: Paths to L2 snapshot JSONL files
        model_path: Where to save trained model
        epochs: Training epochs
        device: "cuda" or "cpu"
    """
    # TODO: Implement training loop
    # For now, just create and save an initialized model

    model = OrderBookCNN(embedding_dim=32, hidden_dim=64)
    model.to(device)

    print(f"Order Book CNN created: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'embedding_dim': 32,
        'hidden_dim': 64,
    }, model_path)

    print(f"Model saved to {model_path}")

    return model


if __name__ == "__main__":
    # Test the model
    print("Testing Order Book CNN...")

    # Create model
    model = OrderBookCNN(embedding_dim=32, hidden_dim=64)

    # Test input: (batch=2, seq=30, features=40)
    test_input = torch.randn(2, 30, 40)
    output = model(test_input)

    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output (pattern embeddings): {output}")

    # Test feature extractor
    extractor = OrderBookFeatureExtractor()

    test_snapshot = {
        "bids": [{"price": 20000 - i*0.25, "size": 100 - i*5} for i in range(10)],
        "asks": [{"price": 20001 + i*0.25, "size": 95 - i*5} for i in range(10)],
        "spread": 1.0,
        "mid": 20000.5,
        "bid_depth": 550,
        "ask_depth": 500,
        "imbalance": 0.048,
    }

    features = extractor.extract_snapshot_features(test_snapshot)
    print(f"\nSnapshot features shape: {features.shape}")

    advanced = extractor.get_advanced_features(test_snapshot)
    print(f"Advanced features: {advanced}")

    walls_bid, walls_ask = extractor.detect_walls(test_snapshot)
    print(f"Bid walls at levels: {walls_bid}")
    print(f"Ask walls at levels: {walls_ask}")

    print("\n✓ Order Book CNN module ready")
