# Advanced ML Trading Models - LSTM & PPO Guide

## Overview

This guide explains how to use the newly implemented **LSTM** (Long Short-Term Memory) and **PPO** (Proximal Policy Optimization) models alongside your existing LightGBM system for enhanced trading predictions.

## Model Types

### 1. LSTM (Sequential Pattern Recognition)
- **Purpose**: Learns temporal patterns in market data
- **Strengths**: Captures time-dependent relationships, trend recognition
- **Outputs**:
  - Trading signals (Buy/Sell/Hold probabilities)
  - Expected price movement
  - Risk assessment score

### 2. PPO (Reinforcement Learning)
- **Purpose**: Learns optimal trading policies through market interaction
- **Strengths**: Adapts to market conditions, optimizes for total returns
- **Outputs**:
  - Action recommendations with probabilities
  - State value estimation
  - Risk-adjusted decisions

### 3. Ensemble Predictor
- Combines LSTM, PPO, and LightGBM predictions
- Weighted voting system for final recommendations
- Confidence scoring based on model agreement

## Installation

```bash
cd Trading-Agent/qwen-topstepx-bundle/ml

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Training Models

### 1. Prepare Data
First, ensure you have trading data collected:

```bash
# Build dataset from trading decisions and outcomes
python3 scripts/build_dataset.py
```

### 2. Train LSTM Model

```bash
python3 scripts/train_lstm_model.py
```

**Configuration** (in script):
- `SEQUENCE_LENGTH = 60`: Uses 60 minutes of historical data
- `PREDICTION_HORIZON = 5`: Predicts 5 minutes ahead
- `HIDDEN_SIZE = 128`: LSTM hidden layer size
- `NUM_LAYERS = 2`: Number of LSTM layers

**Features**:
- Bidirectional LSTM for better context understanding
- Attention mechanism to focus on important time steps
- Multi-task learning (signals, movement, risk)
- Time series cross-validation
- Early stopping to prevent overfitting

### 3. Train PPO Agent

```bash
python3 scripts/train_ppo_agent.py
```

**Configuration** (in script):
- `initial_balance = 100000`: Starting capital
- `max_position_size = 5`: Maximum contracts
- `learning_rate = 3e-4`: PPO learning rate
- `total_timesteps = 1000000`: Training duration

**Features**:
- Realistic trading environment simulation
- Transaction costs and slippage modeling
- Risk-adjusted rewards (Sharpe ratio)
- Generalized Advantage Estimation (GAE)
- Parallel environment training

## Using Predictions

### 1. Individual Model Predictions

For real-time predictions, use the advanced predictor:

```bash
# Prepare market snapshot JSON
echo '{
  "symbol": "NQZ5",
  "timestamp": "2024-01-01T12:00:00Z",
  "features": {
    "close": 16500,
    "volume": 1000,
    "rsi": 45,
    "dist_to_poc_ticks": 4
  },
  "position": 2,
  "balance": 102000,
  "unrealized_pnl": 500
}' | python3 scripts/predict_advanced.py
```

### 2. Ensemble Predictions

The `predict_advanced.py` script automatically combines all available models:

```python
from predict_advanced import EnsemblePredictor

predictor = EnsemblePredictor()

market_data = {
    'features': feature_array,
    'sequence': historical_sequence,  # For LSTM
    'position': current_position,
    'balance': current_balance,
    'unrealized_pnl': unrealized_pnl,
    'recent_actions': [0, 1, 0, 2, ...],  # Last 10 actions
}

predictions = predictor.ensemble_predict(market_data)
```

### 3. Output Format

```json
{
  "timestamp": "2024-01-01T12:00:00Z",
  "models_used": ["lgb_5m", "lgb_30m", "lstm", "ppo"],

  "win_5m_prob": 0.65,
  "win_30m_prob": 0.58,

  "lstm_buy_prob": 0.45,
  "lstm_sell_prob": 0.20,
  "lstm_hold_prob": 0.35,
  "lstm_expected_movement": 0.0025,
  "lstm_risk_score": 0.15,

  "ppo_buy_prob": 0.55,
  "ppo_sell_prob": 0.10,
  "ppo_hold_prob": 0.35,
  "ppo_state_value": 102.5,

  "ensemble_recommendation": "BUY",
  "confidence_score": 0.72
}
```

## Integration with Trading System

### 1. TypeScript Integration

Create a service to call the Python models:

```typescript
// lib/advancedMLService.ts
import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);

export interface MLPrediction {
  ensemble_recommendation: 'BUY' | 'SELL' | 'HOLD';
  confidence_score: number;
  lstm_buy_prob?: number;
  ppo_buy_prob?: number;
  // ... other fields
}

export async function getAdvancedPrediction(
  marketData: any
): Promise<MLPrediction> {
  const input = JSON.stringify(marketData);

  try {
    const { stdout } = await execAsync(
      `echo '${input}' | python3 ml/scripts/predict_advanced.py`,
      { cwd: process.cwd() }
    );

    const lines = stdout.split('\n');
    const jsonStart = lines.findIndex(l => l.startsWith('{'));
    const jsonEnd = lines.findIndex(l => l.startsWith('}')) + 1;
    const jsonStr = lines.slice(jsonStart, jsonEnd).join('\n');

    return JSON.parse(jsonStr);
  } catch (error) {
    console.error('ML prediction error:', error);
    return {
      ensemble_recommendation: 'HOLD',
      confidence_score: 0
    };
  }
}
```

### 2. Using in Trading Logic

```typescript
// In your trading strategy
const prediction = await getAdvancedPrediction({
  features: currentFeatures,
  sequence: historicalData,
  position: this.position,
  balance: this.balance
});

if (prediction.confidence_score > 0.7) {
  switch (prediction.ensemble_recommendation) {
    case 'BUY':
      if (prediction.lstm_risk_score < 0.2) {
        // Execute buy order
        await this.executeBuy();
      }
      break;
    case 'SELL':
      // Execute sell order
      await this.executeSell();
      break;
    case 'HOLD':
      // Do nothing
      break;
  }
}
```

## Performance Tips

### LSTM Optimization
1. **Sequence Length**: Adjust `SEQUENCE_LENGTH` based on your trading timeframe
   - Shorter (30-60): For scalping and day trading
   - Longer (120-240): For swing trading

2. **Feature Engineering**: Add more technical indicators
   - Moving averages, Bollinger Bands
   - Volume indicators (OBV, VWAP)
   - Market microstructure features

3. **Model Architecture**: Experiment with
   - GRU instead of LSTM for faster training
   - Transformer models for better long-range dependencies
   - CNN-LSTM hybrid for pattern recognition

### PPO Optimization
1. **Reward Shaping**: Customize rewards in `TradingEnvironment.step()`
   - Add drawdown penalties
   - Reward consistent profits
   - Penalize excessive trading

2. **State Representation**: Enhance `TradingState`
   - Add order book features
   - Include market regime indicators
   - Add sentiment scores

3. **Hyperparameter Tuning**:
   ```python
   # In train_ppo_agent.py
   config.learning_rate = 1e-4  # Try different values
   config.clip_epsilon = 0.1    # Reduce for more conservative updates
   config.n_steps = 4096        # Increase for more stable training
   ```

## Backtesting

Evaluate model performance:

```python
# Add to a new script: backtest_models.py
from predict_advanced import EnsemblePredictor
import pandas as pd

def backtest(data: pd.DataFrame, predictor: EnsemblePredictor):
    initial_balance = 100000
    balance = initial_balance
    position = 0
    trades = []

    for i in range(60, len(data)):  # Start after warmup
        # Prepare market data
        market_data = prepare_market_data(data, i)

        # Get prediction
        prediction = predictor.ensemble_predict(market_data)

        # Execute trades based on prediction
        if prediction['ensemble_recommendation'] == 'BUY' and position < 5:
            position += 1
            balance -= data.iloc[i]['close']
            trades.append({'action': 'buy', 'price': data.iloc[i]['close']})

        elif prediction['ensemble_recommendation'] == 'SELL' and position > -5:
            position -= 1
            balance += data.iloc[i]['close']
            trades.append({'action': 'sell', 'price': data.iloc[i]['close']})

    # Calculate metrics
    total_return = (balance - initial_balance) / initial_balance
    sharpe_ratio = calculate_sharpe(trades)
    max_drawdown = calculate_max_drawdown(balance_history)

    return {
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'num_trades': len(trades)
    }
```

## Model Monitoring

Track model performance in production:

```python
# Add logging to predict_advanced.py
import logging

logging.basicConfig(
    filename='ml/logs/predictions.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

# In ensemble_predict method:
logging.info(f"Prediction: {predictions['ensemble_recommendation']}, "
            f"Confidence: {predictions['confidence_score']:.2%}")
```

## Troubleshooting

### Common Issues

1. **Insufficient Data**
   - Need at least 1000 rows for PPO
   - Need at least 200 rows for LSTM
   - Solution: Collect more trading data or use synthetic data

2. **LSTM Overfitting**
   - Symptoms: High training accuracy, poor validation
   - Solutions: Increase dropout, reduce model size, more data

3. **PPO Not Converging**
   - Symptoms: Rewards not improving
   - Solutions: Reduce learning rate, adjust reward function

4. **Memory Issues**
   - For large datasets, use batch processing:
   ```python
   # Process in chunks
   for chunk in pd.read_parquet(file, chunksize=1000):
       process_chunk(chunk)
   ```

## Future Enhancements

1. **Advanced Models**
   - Implement Transformer-based models (Attention is All You Need)
   - Add Graph Neural Networks for market correlation
   - Implement A3C or SAC for better RL performance

2. **Feature Engineering**
   - Add sentiment analysis from news
   - Include order flow imbalance
   - Add cross-market correlations

3. **Risk Management**
   - Implement Kelly Criterion for position sizing
   - Add Value at Risk (VaR) calculations
   - Dynamic stop-loss based on volatility

4. **Real-time Processing**
   - Stream processing with Apache Kafka
   - GPU acceleration with CUDA
   - Distributed training with PyTorch Distributed

## Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [Stable Baselines3 PPO](https://stable-baselines3.readthedocs.io/)
- [Time Series Analysis](https://otexts.com/fpp3/)
- [Reinforcement Learning in Finance](https://arxiv.org/abs/1907.04373)

## Support

For questions or issues:
1. Check the logs in `ml/logs/`
2. Verify data quality in `ml/data/`
3. Review model metrics in `ml/models/`

Remember to always test thoroughly in simulation before live trading!
