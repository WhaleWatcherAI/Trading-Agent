# Task Allocation for ML Trading Models

## Data Requirements Summary

### Minimum Data Needed to Start Training

| Model | Minimum Data | Optimal Data | Time Period |
|-------|-------------|--------------|-------------|
| **LightGBM** | 50 trades | 500+ trades | 1-2 weeks |
| **LSTM** | 200 sequences (1000 hours) | 5000+ sequences | 1-3 months |
| **PPO** | 1000 episodes | 10000+ episodes | 2-6 months |

### Required Data Format

```json
// 1. Market Data (5-minute candles minimum)
{
  "timestamp": "2024-01-01T09:30:00Z",
  "open": 16500.25,
  "high": 16510.50,
  "low": 16495.00,
  "close": 16508.75,
  "volume": 2450,
  "indicators": {
    "rsi": 52.3,
    "macd": 2.5,
    "bb_upper": 16520,
    "bb_lower": 16480,
    "vwap": 16505.50,
    "poc": 16502.00,
    "volatility": 0.0023
  }
}

// 2. Trading Decisions
{
  "id": "decision_12345",
  "timestamp": "2024-01-01T09:35:00Z",
  "symbol": "NQ",
  "signal": "BUY",
  "price": 16508.75,
  "confidence": 0.75,
  "features": {...}
}

// 3. Trading Outcomes
{
  "decisionId": "decision_12345",
  "entryPrice": 16508.75,
  "exitPrice": 16518.50,
  "profitLoss": 9.75,
  "win_5m": true,
  "win_30m": true,
  "executedTime": "2024-01-01T09:35:00Z",
  "closedTime": "2024-01-01T09:40:00Z"
}
```

## Task Allocation by Model

### 1. **LSTM Tasks** - Temporal Pattern Recognition

#### Primary Responsibilities:
```python
LSTM_TASKS = {
    "entry_timing": {
        "description": "Identify optimal entry points",
        "inputs": ["60-min price history", "volume patterns", "momentum indicators"],
        "outputs": ["entry_signal", "expected_movement", "time_to_target"]
    },

    "trend_detection": {
        "description": "Identify and follow market trends",
        "inputs": ["price sequences", "moving averages", "momentum"],
        "outputs": ["trend_direction", "trend_strength", "reversal_probability"]
    },

    "volatility_prediction": {
        "description": "Forecast market volatility",
        "inputs": ["historical volatility", "volume", "price ranges"],
        "outputs": ["volatility_forecast", "risk_score", "stop_loss_suggestion"]
    },

    "pattern_recognition": {
        "description": "Detect chart patterns",
        "inputs": ["OHLC sequences", "volume", "indicators"],
        "outputs": ["pattern_type", "pattern_confidence", "breakout_probability"]
    }
}
```

#### Implementation:
```python
# LSTM handles time-sensitive decisions
def lstm_entry_decision(market_data):
    sequence = prepare_sequence(market_data, length=60)

    # LSTM predictions
    signals, movement, risk = lstm_model.predict(sequence)

    if movement > 0.003 and risk < 0.2:  # 0.3% expected profit, low risk
        return {
            'action': 'ENTER_LONG',
            'target': price + (movement * price),
            'stop': price - (risk * price * 2),
            'confidence': signals[1]  # buy probability
        }
```

### 2. **PPO Tasks** - Strategic Position Management

#### Primary Responsibilities:
```python
PPO_TASKS = {
    "position_sizing": {
        "description": "Determine optimal position size",
        "inputs": ["account_balance", "risk_tolerance", "market_conditions"],
        "outputs": ["position_size", "max_risk", "leverage"]
    },

    "portfolio_management": {
        "description": "Manage multiple positions",
        "inputs": ["current_positions", "correlation_matrix", "total_exposure"],
        "outputs": ["rebalance_actions", "hedge_recommendations", "risk_adjustments"]
    },

    "exit_strategy": {
        "description": "Optimize trade exits",
        "inputs": ["unrealized_pnl", "time_in_trade", "market_conditions"],
        "outputs": ["exit_signal", "partial_exit_size", "trailing_stop"]
    },

    "risk_management": {
        "description": "Dynamic risk adjustment",
        "inputs": ["drawdown", "volatility", "win_rate"],
        "outputs": ["position_limit", "stop_loss_adjustment", "risk_per_trade"]
    }
}
```

#### Implementation:
```python
# PPO handles strategic decisions
def ppo_position_decision(state):
    # Current state
    trading_state = TradingState(
        position=current_position,
        balance=account_balance,
        unrealized_pnl=current_pnl,
        market_features=features
    )

    # PPO recommendation
    action_probs, state_value = ppo_model.predict(trading_state)

    # Position sizing based on Kelly Criterion
    kelly_fraction = calculate_kelly(
        win_prob=action_probs[1],  # buy probability
        avg_win=historical_avg_win,
        avg_loss=historical_avg_loss
    )

    return {
        'action': 'INCREASE_POSITION' if action_probs[1] > 0.6 else 'HOLD',
        'size': min(kelly_fraction * balance / price, max_position),
        'confidence': max(action_probs)
    }
```

### 3. **LightGBM Tasks** - Statistical Edge Detection

#### Primary Responsibilities:
```python
LIGHTGBM_TASKS = {
    "win_probability": {
        "description": "Calculate trade success probability",
        "inputs": ["market_features", "technical_indicators", "market_regime"],
        "outputs": ["win_5m_prob", "win_30m_prob", "expected_return"]
    },

    "feature_importance": {
        "description": "Identify key market factors",
        "inputs": ["all_features"],
        "outputs": ["feature_rankings", "correlation_matrix", "predictive_power"]
    },

    "regime_classification": {
        "description": "Identify market regimes",
        "inputs": ["volatility", "trend", "volume"],
        "outputs": ["market_regime", "regime_confidence", "regime_duration"]
    }
}
```

### 4. **Ensemble Tasks** - Unified Decision Making

#### Primary Responsibilities:
```python
ENSEMBLE_TASKS = {
    "final_decision": {
        "description": "Combine all model outputs",
        "logic": """
        if all_models_agree and confidence > 0.75:
            execute_trade(size='full')
        elif majority_agree and confidence > 0.65:
            execute_trade(size='half')
        elif lstm_strong_signal and ppo_confirms:
            execute_trade(size='small')
        else:
            wait_for_better_setup()
        """
    },

    "conflict_resolution": {
        "description": "Handle model disagreements",
        "rules": {
            "lstm_vs_ppo": "Trust LSTM for timing, PPO for sizing",
            "high_risk": "If any model shows risk > 0.7, reduce or skip",
            "low_confidence": "If ensemble confidence < 0.5, no trade"
        }
    }
}
```

## Complete Trading Workflow

```python
# Main trading loop with task allocation
async def trading_loop():
    while market_is_open():
        # 1. LSTM: Check for entry signals
        lstm_signal = await lstm_check_entry(market_data)

        if lstm_signal['confidence'] > 0.65:
            # 2. LightGBM: Verify statistical edge
            win_prob = await lightgbm_check_probability(market_features)

            if win_prob['5m'] > 0.60:
                # 3. PPO: Determine position size and risk
                position_params = await ppo_calculate_position(
                    account_state,
                    lstm_signal
                )

                # 4. Ensemble: Final decision
                final_decision = ensemble_decide(
                    lstm_signal,
                    win_prob,
                    position_params
                )

                if final_decision['execute']:
                    # Execute trade with parameters from each model
                    trade = {
                        'entry_price': market_data['price'],  # Current
                        'direction': lstm_signal['direction'],  # LSTM
                        'size': position_params['size'],  # PPO
                        'stop_loss': lstm_signal['stop'],  # LSTM risk
                        'take_profit': calculate_target(
                            lstm_signal['expected_movement'],  # LSTM
                            win_prob['expected_return']  # LightGBM
                        ),
                        'confidence': final_decision['confidence']  # Ensemble
                    }

                    await execute_trade(trade)

        # PPO: Manage existing positions
        if has_open_positions():
            exit_signal = await ppo_check_exits(open_positions)
            if exit_signal['should_exit']:
                await close_position(exit_signal['position_id'])

        await sleep(1)  # Check every second
```

## Data Collection Script Usage

To prepare data for training:

```bash
# 1. Install required packages
pip install yfinance pandas numpy

# 2. Collect and prepare data
python3 ml/scripts/data_collector.py

# This will:
# - Download 3 months of 5-minute market data
# - Generate technical indicators
# - Create sample trading decisions
# - Generate outcomes for training
# - Format everything for ML models

# 3. Build the dataset
python3 ml/scripts/build_dataset.py

# 4. Train models
python3 ml/scripts/train_lstm_model.py    # Train LSTM
python3 ml/scripts/train_ppo_agent.py      # Train PPO
python3 ml/scripts/train_meta_label.py     # Train LightGBM

# 5. Test ensemble
python3 ml/example_usage.py
```

## Quick Start with Live Data

If you want to use your actual trading data:

```typescript
// In your TopStepX trading agent, add data logging:

// Log decisions
function logDecision(signal: string, price: number, features: any) {
    const decision = {
        id: `decision_${Date.now()}`,
        timestamp: new Date().toISOString(),
        symbol: 'NQ',
        signal: signal,
        price: price,
        features: features
    };

    fs.appendFileSync(
        'ml/trading-db/decisions.jsonl',
        JSON.stringify(decision) + '\n'
    );
}

// Log outcomes
function logOutcome(decisionId: string, entry: number, exit: number, pnl: number) {
    const outcome = {
        decisionId: decisionId,
        entryPrice: entry,
        exitPrice: exit,
        profitLoss: pnl,
        win_5m: pnl > 2,
        win_30m: pnl > 5,
        timestamp: new Date().toISOString()
    };

    fs.appendFileSync(
        'ml/trading-db/outcomes.jsonl',
        JSON.stringify(outcome) + '\n'
    );
}

// Log market snapshots
function logSnapshot(marketData: any) {
    const snapshot = {
        symbol: 'NQ',
        timestamp: new Date().toISOString(),
        features: {
            price: marketData.price,
            rsi: marketData.rsi,
            macd: marketData.macd,
            volume: marketData.volume,
            // ... other features
        }
    };

    fs.appendFileSync(
        'ml/data/snapshots.jsonl',
        JSON.stringify(snapshot) + '\n'
    );
}
```

## Performance Expectations

| Model | Training Time | Accuracy | Best For |
|-------|--------------|----------|----------|
| **LSTM** | 10-30 min | 65-75% directional | Entry timing, trend following |
| **PPO** | 1-2 hours | 55-65% profitable trades | Position sizing, risk management |
| **LightGBM** | 2-5 min | 60-70% win rate | Quick decisions, filtering |
| **Ensemble** | Combined | 70-80% confidence trades | Live trading with real money |

## Start Simple

For your first implementation:

1. **Week 1**: Collect data using `data_collector.py`
2. **Week 2**: Train LightGBM only (fastest, simplest)
3. **Week 3**: Add LSTM for better entry timing
4. **Week 4**: Add PPO for position management
5. **Week 5+**: Use ensemble for live trading

The key is to start collecting data immediately - the models improve dramatically with more data!