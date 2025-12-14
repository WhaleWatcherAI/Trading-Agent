#!/usr/bin/env python3
"""
L2 Execution Feature Extractor

Extracts order book features for execution optimization:
- Spread metrics
- Bid/Ask walls
- Absorption indicators
- Order book imbalance
- Volume pressure
"""

import numpy as np
from typing import Dict, List, Optional
from collections import deque


class L2ExecutionFeatures:
    """Extract L2 features for execution timing optimization."""

    def __init__(self, lookback_ms: int = 1000):
        """
        Args:
            lookback_ms: How many milliseconds of history to track
        """
        self.lookback_ms = lookback_ms
        self.snapshot_history = deque(maxlen=100)  # ~3.5 seconds at 28/sec

    def add_snapshot(self, snapshot: Dict):
        """Add L2 snapshot to history."""
        self.snapshot_history.append(snapshot)

    def extract_features(self, direction: str) -> Dict[str, float]:
        """
        Extract L2 features for execution scoring.

        Args:
            direction: "LONG" or "SHORT"

        Returns:
            Dictionary of features
        """
        if not self.snapshot_history:
            return {}

        current = self.snapshot_history[-1]
        features = {}

        # === SPREAD FEATURES ===
        features['spread'] = current.get('spread', 0.0)
        features['spread_bps'] = (features['spread'] / current.get('mid', 1.0)) * 10000

        # === IMMEDIATE LIQUIDITY ===
        bids = current.get('bids', [])
        asks = current.get('asks', [])

        if bids:
            features['best_bid'] = bids[0]['price']
            features['best_bid_size'] = bids[0]['size']
        else:
            features['best_bid'] = 0
            features['best_bid_size'] = 0

        if asks:
            features['best_ask'] = asks[0]['price']
            features['best_ask_size'] = asks[0]['size']
        else:
            features['best_ask'] = 0
            features['best_ask_size'] = 0

        # === ORDER BOOK DEPTH ===
        features['bid_depth_L3'] = sum(b['size'] for b in bids[:3])
        features['ask_depth_L3'] = sum(a['size'] for a in asks[:3])
        features['bid_depth_L10'] = sum(b['size'] for b in bids[:10])
        features['ask_depth_L10'] = sum(a['size'] for a in asks[:10])

        # === IMBALANCE ===
        features['imbalance'] = current.get('imbalance', 0.0)

        # === WALL DETECTION ===
        bid_sizes = [b['size'] for b in bids]
        ask_sizes = [a['size'] for a in asks]

        if bid_sizes:
            features['max_bid_size'] = max(bid_sizes)
            avg_bid = sum(bid_sizes) / len(bid_sizes)
            features['bid_wall_ratio'] = features['max_bid_size'] / avg_bid if avg_bid > 0 else 1.0
        else:
            features['max_bid_size'] = 0
            features['bid_wall_ratio'] = 1.0

        if ask_sizes:
            features['max_ask_size'] = max(ask_sizes)
            avg_ask = sum(ask_sizes) / len(ask_sizes)
            features['ask_wall_ratio'] = features['max_ask_size'] / avg_ask if avg_ask > 0 else 1.0
        else:
            features['max_ask_size'] = 0
            features['ask_wall_ratio'] = 1.0

        # === TIME-SERIES FEATURES (if we have history) ===
        if len(self.snapshot_history) >= 2:
            prev = self.snapshot_history[-2]

            # Price movement
            prev_mid = prev.get('mid', features.get('best_bid', 0))
            curr_mid = current.get('mid', features.get('best_bid', 0))
            features['mid_delta_1tick'] = curr_mid - prev_mid

            # Spread changes
            prev_spread = prev.get('spread', 0)
            features['spread_delta_1tick'] = features['spread'] - prev_spread

            # Best bid/ask size changes (absorption indicators)
            prev_bids = prev.get('bids', [])
            prev_asks = prev.get('asks', [])

            if prev_bids and bids:
                # Check if same price level
                if prev_bids[0]['price'] == bids[0]['price']:
                    features['bid_L1_absorption'] = bids[0]['size'] - prev_bids[0]['size']
                else:
                    features['bid_L1_absorption'] = 0.0
            else:
                features['bid_L1_absorption'] = 0.0

            if prev_asks and asks:
                if prev_asks[0]['price'] == asks[0]['price']:
                    features['ask_L1_absorption'] = asks[0]['size'] - prev_asks[0]['size']
                else:
                    features['ask_L1_absorption'] = 0.0
            else:
                features['ask_L1_absorption'] = 0.0
        else:
            features['mid_delta_1tick'] = 0.0
            features['spread_delta_1tick'] = 0.0
            features['bid_L1_absorption'] = 0.0
            features['ask_L1_absorption'] = 0.0

        # === Rolling statistics (over ~1 second) ===
        if len(self.snapshot_history) >= 10:
            recent = list(self.snapshot_history)[-28:]  # Last ~1 second

            spreads = [s.get('spread', 0) for s in recent]
            features['spread_avg_1s'] = np.mean(spreads)
            features['spread_std_1s'] = np.std(spreads)
            features['spread_min_1s'] = np.min(spreads)

            imbalances = [s.get('imbalance', 0) for s in recent]
            features['imbalance_avg_1s'] = np.mean(imbalances)
            features['imbalance_trend_1s'] = imbalances[-1] - imbalances[0] if len(imbalances) > 1 else 0
        else:
            features['spread_avg_1s'] = features['spread']
            features['spread_std_1s'] = 0.0
            features['spread_min_1s'] = features['spread']
            features['imbalance_avg_1s'] = features['imbalance']
            features['imbalance_trend_1s'] = 0.0

        # === DIRECTION-SPECIFIC FEATURES ===
        if direction == "LONG":
            # For buying: want tight spread, no ask walls, positive imbalance
            features['execution_spread'] = features['spread']  # Cost to cross spread
            features['wall_blocking'] = features['ask_wall_ratio']  # Ask walls block us
            features['wall_supporting'] = features['bid_wall_ratio']  # Bid walls help us
            features['flow_favorable'] = max(0, features['imbalance'])  # Positive = bullish
            features['absorption_against'] = max(0, -features['ask_L1_absorption'])  # Ask absorption bad

        else:  # SHORT
            # For selling: want tight spread, no bid walls, negative imbalance
            features['execution_spread'] = features['spread']
            features['wall_blocking'] = features['bid_wall_ratio']  # Bid walls block us
            features['wall_supporting'] = features['ask_wall_ratio']  # Ask walls help us
            features['flow_favorable'] = max(0, -features['imbalance'])  # Negative = bearish
            features['absorption_against'] = max(0, -features['bid_L1_absorption'])  # Bid absorption bad

        return features

    def get_feature_vector(self, direction: str) -> np.ndarray:
        """
        Get feature vector as numpy array for XGBoost.

        Returns array of features in consistent order.
        """
        features = self.extract_features(direction)

        # Ordered feature list
        feature_names = [
            'spread', 'spread_bps', 'best_bid_size', 'best_ask_size',
            'bid_depth_L3', 'ask_depth_L3', 'bid_depth_L10', 'ask_depth_L10',
            'imbalance', 'max_bid_size', 'max_ask_size',
            'bid_wall_ratio', 'ask_wall_ratio',
            'mid_delta_1tick', 'spread_delta_1tick',
            'bid_L1_absorption', 'ask_L1_absorption',
            'spread_avg_1s', 'spread_std_1s', 'spread_min_1s',
            'imbalance_avg_1s', 'imbalance_trend_1s',
            'execution_spread', 'wall_blocking', 'wall_supporting',
            'flow_favorable', 'absorption_against'
        ]

        return np.array([features.get(name, 0.0) for name in feature_names])

    @staticmethod
    def get_feature_names() -> List[str]:
        """Return ordered list of feature names."""
        return [
            'spread', 'spread_bps', 'best_bid_size', 'best_ask_size',
            'bid_depth_L3', 'ask_depth_L3', 'bid_depth_L10', 'ask_depth_L10',
            'imbalance', 'max_bid_size', 'max_ask_size',
            'bid_wall_ratio', 'ask_wall_ratio',
            'mid_delta_1tick', 'spread_delta_1tick',
            'bid_L1_absorption', 'ask_L1_absorption',
            'spread_avg_1s', 'spread_std_1s', 'spread_min_1s',
            'imbalance_avg_1s', 'imbalance_trend_1s',
            'execution_spread', 'wall_blocking', 'wall_supporting',
            'flow_favorable', 'absorption_against'
        ]


if __name__ == "__main__":
    # Test feature extraction
    print("L2 Execution Feature Extractor")
    print(f"Total features: {len(L2ExecutionFeatures.get_feature_names())}")
    print("\nFeature names:")
    for i, name in enumerate(L2ExecutionFeatures.get_feature_names(), 1):
        print(f"  {i:2d}. {name}")
