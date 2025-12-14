#!/usr/bin/env python3
"""
Advanced prediction script combining LSTM and PPO models with existing LightGBM.
Provides ensemble predictions for trading decisions.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import lightgbm as lgb

# Import model classes from training scripts
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "ml" / "scripts"))

from train_lstm_model import LSTMTradingModel, SEQUENCE_LENGTH
from train_ppo_agent import ActorCriticNetwork, TradingState, PPOConfig

DATA_DIR = ROOT / "ml" / "data"
MODELS_DIR = ROOT / "ml" / "models"

# Model paths
LIGHTGBM_5M = MODELS_DIR / "meta_label_5m.txt"
LIGHTGBM_30M = MODELS_DIR / "meta_label_30m.txt"
LSTM_MODEL = MODELS_DIR / "lstm_trading_model.pth"
LSTM_SCALER = MODELS_DIR / "lstm_scaler.pkl"
PPO_MODEL = MODELS_DIR / "ppo_trading_agent.pth"
FEATURES_JSON = MODELS_DIR / "features.json"


class EnsemblePredictor:
    """
    Ensemble predictor combining multiple model types:
    - LightGBM for win probability
    - LSTM for temporal pattern recognition
    - PPO for action recommendations
    """

    def __init__(self):
        self.models_loaded = {}
        self.load_models()

    def load_models(self):
        """Load all available models."""
        # Load LightGBM models
        if LIGHTGBM_5M.exists():
            self.lgb_5m = lgb.Booster(model_file=str(LIGHTGBM_5M))
            self.models_loaded['lgb_5m'] = True

        if LIGHTGBM_30M.exists():
            self.lgb_30m = lgb.Booster(model_file=str(LIGHTGBM_30M))
            self.models_loaded['lgb_30m'] = True

        # Load LSTM model
        if LSTM_MODEL.exists() and LSTM_SCALER.exists():
            try:
                # Load scaler
                with open(LSTM_SCALER, 'rb') as f:
                    self.lstm_scaler = pickle.load(f)

                # Load model
                feature_cols = self._load_features()
                self.lstm_model = LSTMTradingModel(
                    input_size=len(feature_cols),
                    hidden_size=128,
                    num_layers=2,
                    output_size=64,
                    dropout=0.2
                )
                self.lstm_model.load_state_dict(torch.load(LSTM_MODEL))
                self.lstm_model.eval()
                self.models_loaded['lstm'] = True
            except Exception as e:
                print(f"Warning: Could not load LSTM model: {e}")

        # Load PPO model
        if PPO_MODEL.exists():
            try:
                config = PPOConfig()
                self.ppo_network = ActorCriticNetwork(
                    config.state_dim,
                    config.action_dim,
                    config.hidden_dim
                )
                self.ppo_network.load_state_dict(torch.load(PPO_MODEL))
                self.ppo_network.eval()
                self.models_loaded['ppo'] = True
            except Exception as e:
                print(f"Warning: Could not load PPO model: {e}")

    def _load_features(self) -> List[str]:
        """Load feature columns from metadata."""
        if FEATURES_JSON.exists():
            try:
                data = json.loads(FEATURES_JSON.read_text())
                return data.get("feature_columns", [])
            except Exception:
                pass
        return []

    def predict_lightgbm(self, features: np.ndarray) -> Dict[str, float]:
        """Get LightGBM predictions."""
        predictions = {}

        if self.models_loaded.get('lgb_5m'):
            predictions['win_5m_prob'] = float(self.lgb_5m.predict(features.reshape(1, -1))[0])

        if self.models_loaded.get('lgb_30m'):
            predictions['win_30m_prob'] = float(self.lgb_30m.predict(features.reshape(1, -1))[0])

        return predictions

    def predict_lstm(self, sequence: np.ndarray) -> Dict[str, float]:
        """Get LSTM predictions for sequential data."""
        if not self.models_loaded.get('lstm'):
            return {}

        try:
            # Normalize sequence
            sequence_scaled = self.lstm_scaler.transform(
                sequence.reshape(-1, sequence.shape[-1])
            ).reshape(sequence.shape)

            # Convert to tensor
            seq_tensor = torch.FloatTensor(sequence_scaled).unsqueeze(0)

            # Get predictions
            with torch.no_grad():
                signals, movement, risk = self.lstm_model(seq_tensor)

            # Convert to probabilities
            signal_probs = signals[0].numpy()
            movement_val = movement[0].item()
            risk_val = risk[0].item()

            return {
                'lstm_buy_prob': float(signal_probs[1]),
                'lstm_sell_prob': float(signal_probs[2]),
                'lstm_hold_prob': float(signal_probs[0]),
                'lstm_expected_movement': float(movement_val),
                'lstm_risk_score': float(risk_val)
            }
        except Exception as e:
            print(f"LSTM prediction error: {e}")
            return {}

    def predict_ppo(self, state: TradingState) -> Dict[str, float]:
        """Get PPO action recommendations."""
        if not self.models_loaded.get('ppo'):
            return {}

        try:
            with torch.no_grad():
                action_logits, value = self.ppo_network(state)

            # Convert logits to probabilities
            action_probs = torch.softmax(action_logits, dim=-1).numpy()

            return {
                'ppo_hold_prob': float(action_probs[0]),
                'ppo_buy_prob': float(action_probs[1]),
                'ppo_sell_prob': float(action_probs[2]),
                'ppo_state_value': float(value.item())
            }
        except Exception as e:
            print(f"PPO prediction error: {e}")
            return {}

    def ensemble_predict(self, market_data: Dict) -> Dict[str, any]:
        """
        Generate ensemble predictions from all models.

        Args:
            market_data: Dictionary containing:
                - features: Current market features
                - sequence: Historical sequence for LSTM
                - position: Current position
                - balance: Current balance
                - unrealized_pnl: Current unrealized PnL
        """
        predictions = {
            'timestamp': datetime.utcnow().isoformat(),
            'models_used': list(self.models_loaded.keys())
        }

        # Get features
        features = np.array(market_data.get('features', []))

        # LightGBM predictions
        if features.size > 0:
            lgb_preds = self.predict_lightgbm(features)
            predictions.update(lgb_preds)

        # LSTM predictions
        sequence = market_data.get('sequence')
        if sequence is not None and len(sequence) >= SEQUENCE_LENGTH:
            lstm_preds = self.predict_lstm(sequence)
            predictions.update(lstm_preds)

        # PPO predictions
        if 'position' in market_data:
            state = TradingState(
                market_features=features,
                position=market_data.get('position', 0),
                unrealized_pnl=market_data.get('unrealized_pnl', 0),
                balance=market_data.get('balance', 100000),
                recent_actions=market_data.get('recent_actions', [0] * 10),
                timestep=market_data.get('timestep', 0)
            )
            ppo_preds = self.predict_ppo(state)
            predictions.update(ppo_preds)

        # Calculate ensemble recommendation
        predictions['ensemble_recommendation'] = self._calculate_ensemble_action(predictions)
        predictions['confidence_score'] = self._calculate_confidence(predictions)

        return predictions

    def _calculate_ensemble_action(self, predictions: Dict) -> str:
        """
        Calculate ensemble action recommendation based on all model outputs.
        Uses weighted voting based on model confidence.
        """
        buy_score = 0
        sell_score = 0
        hold_score = 0
        total_weight = 0

        # Weight LightGBM predictions
        if 'win_5m_prob' in predictions:
            weight = 1.0
            if predictions['win_5m_prob'] > 0.6:
                buy_score += predictions['win_5m_prob'] * weight
            elif predictions['win_5m_prob'] < 0.4:
                sell_score += (1 - predictions['win_5m_prob']) * weight
            else:
                hold_score += weight * 0.5
            total_weight += weight

        # Weight LSTM predictions
        if 'lstm_buy_prob' in predictions:
            weight = 1.5  # Higher weight for LSTM due to temporal analysis
            buy_score += predictions['lstm_buy_prob'] * weight
            sell_score += predictions['lstm_sell_prob'] * weight
            hold_score += predictions['lstm_hold_prob'] * weight
            total_weight += weight

        # Weight PPO predictions
        if 'ppo_buy_prob' in predictions:
            weight = 2.0  # Highest weight for PPO due to RL optimization
            buy_score += predictions['ppo_buy_prob'] * weight
            sell_score += predictions['ppo_sell_prob'] * weight
            hold_score += predictions['ppo_hold_prob'] * weight
            total_weight += weight

        if total_weight == 0:
            return 'HOLD'

        # Normalize scores
        buy_score /= total_weight
        sell_score /= total_weight
        hold_score /= total_weight

        # Determine action with confidence threshold
        if buy_score > 0.55 and buy_score > sell_score and buy_score > hold_score:
            return 'BUY'
        elif sell_score > 0.55 and sell_score > buy_score and sell_score > hold_score:
            return 'SELL'
        else:
            return 'HOLD'

    def _calculate_confidence(self, predictions: Dict) -> float:
        """
        Calculate confidence score for the ensemble prediction.
        Higher when models agree, lower when they disagree.
        """
        confidences = []

        # Check agreement between models
        buy_probs = []
        sell_probs = []

        if 'lstm_buy_prob' in predictions:
            buy_probs.append(predictions['lstm_buy_prob'])
            sell_probs.append(predictions['lstm_sell_prob'])

        if 'ppo_buy_prob' in predictions:
            buy_probs.append(predictions['ppo_buy_prob'])
            sell_probs.append(predictions['ppo_sell_prob'])

        if buy_probs:
            # Calculate standard deviation as measure of disagreement
            buy_std = np.std(buy_probs)
            sell_std = np.std(sell_probs)

            # Lower std means higher agreement
            agreement_score = 1 - (buy_std + sell_std) / 2
            confidences.append(agreement_score)

        # Add risk-adjusted confidence
        if 'lstm_risk_score' in predictions:
            risk_confidence = 1 - predictions['lstm_risk_score']
            confidences.append(risk_confidence)

        if confidences:
            return float(np.mean(confidences))
        return 0.5


def predict_from_snapshot(snapshot: Dict) -> Dict:
    """
    Make prediction from a market snapshot.

    Args:
        snapshot: Dictionary containing market data snapshot
    """
    predictor = EnsemblePredictor()

    # Prepare market data
    features = snapshot.get('features', {})
    feature_array = np.array(list(features.values()))

    market_data = {
        'features': feature_array,
        'position': snapshot.get('position', 0),
        'balance': snapshot.get('balance', 100000),
        'unrealized_pnl': snapshot.get('unrealized_pnl', 0),
        'recent_actions': snapshot.get('recent_actions', [0] * 10),
        'timestep': snapshot.get('timestep', 0)
    }

    # Add sequence data if available
    if 'sequence' in snapshot:
        market_data['sequence'] = np.array(snapshot['sequence'])

    # Generate predictions
    predictions = predictor.ensemble_predict(market_data)

    return predictions


def main():
    """Main function for command-line usage."""
    predictor = EnsemblePredictor()

    print(f"Loaded models: {list(predictor.models_loaded.keys())}")
    print("\nReading market snapshot from stdin...")

    # Read JSON from stdin
    input_data = sys.stdin.read()
    try:
        snapshot = json.loads(input_data)
    except json.JSONDecodeError:
        print("Error: Invalid JSON input")
        sys.exit(1)

    # Generate predictions
    predictions = predict_from_snapshot(snapshot)

    # Output predictions
    print(json.dumps(predictions, indent=2))

    # Print human-readable summary
    print("\n" + "="*50)
    print("TRADING SIGNAL SUMMARY")
    print("="*50)
    print(f"Ensemble Recommendation: {predictions['ensemble_recommendation']}")
    print(f"Confidence Score: {predictions['confidence_score']:.2%}")

    if 'win_5m_prob' in predictions:
        print(f"\nLightGBM Predictions:")
        print(f"  5min Win Probability: {predictions['win_5m_prob']:.2%}")
        if 'win_30m_prob' in predictions:
            print(f"  30min Win Probability: {predictions['win_30m_prob']:.2%}")

    if 'lstm_buy_prob' in predictions:
        print(f"\nLSTM Predictions:")
        print(f"  Buy: {predictions['lstm_buy_prob']:.2%}")
        print(f"  Sell: {predictions['lstm_sell_prob']:.2%}")
        print(f"  Hold: {predictions['lstm_hold_prob']:.2%}")
        print(f"  Expected Movement: {predictions['lstm_expected_movement']:.4f}")
        print(f"  Risk Score: {predictions['lstm_risk_score']:.2%}")

    if 'ppo_buy_prob' in predictions:
        print(f"\nPPO Recommendations:")
        print(f"  Buy: {predictions['ppo_buy_prob']:.2%}")
        print(f"  Sell: {predictions['ppo_sell_prob']:.2%}")
        print(f"  Hold: {predictions['ppo_hold_prob']:.2%}")
        print(f"  State Value: {predictions['ppo_state_value']:.2f}")


if __name__ == "__main__":
    main()