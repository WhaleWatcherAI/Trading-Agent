# Self-Learning Trading Agent - Enhanced Fabio System

## Overview
This document describes the enhanced self-learning capabilities added to the Fabio trading agent. The system now features true autonomous learning through market regime inference, performance-based strategy adaptation, and POC cross tracking.

## Key Enhancements vs Original System

### Original Fabio System (40-50% Implementation)
- ✅ Volume profile analysis
- ✅ Basic LLM decision making
- ✅ Strategy configuration
- ❌ POC cross tracking
- ❌ Raw stats for regime inference
- ❌ Performance feedback loops
- ❌ Active strategy management
- ❌ Historical notes learning

### Enhanced Self-Learning System (100% Implementation)
- ✅ POC cross tracking with time windows
- ✅ Raw market statistics (LLM infers regime)
- ✅ Performance metrics computation per strategy
- ✅ Strategy parameters that affect execution
- ✅ Historical notes to future self
- ✅ Session range percentile tracking
- ✅ Time in/outside value area tracking
- ✅ Dynamic strategy enable/disable based on performance

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Enhanced Engine                        │
│                  (engine_enhanced.py)                    │
└────────────┬────────────────────────┬───────────────────┘
             │                        │
    ┌────────▼──────────┐   ┌────────▼──────────┐
    │ Enhanced Features  │   │   Enhanced LLM     │
    │ (POC Tracking,     │   │ (Regime Inference, │
    │  Raw Stats)        │   │  Self-Learning)    │
    └────────┬──────────┘   └────────┬──────────┘
             │                        │
             └────────┬───────────────┘
                      │
            ┌─────────▼──────────┐
            │ Enhanced Execution  │
            │ (Active Strategy    │
            │  Management)        │
            └────────────────────┘
```

## Component Details

### 1. Enhanced Feature Engine (`features_enhanced.py`)

**Key Features:**
- POC cross tracking with 5min, 15min, 30min windows
- Session range tracking with historical percentiles
- Time spent in/above/below value area
- Raw market statistics for LLM inference

**POC Cross Tracking:**
```python
poc_cross_stats = {
    "count_last_5min": 2,
    "count_last_15min": 5,
    "count_last_30min": 12,
    "time_since_last_cross_sec": 45.3,
    "current_side": "above_poc"
}
```

**Market Stats Provided:**
```python
market_stats = {
    "session_range_ticks": 120,
    "session_range_percentile": 0.75,  # 75th percentile
    "distance_to_poc_ticks": 15,
    "time_above_value_sec": 1800,
    "time_below_value_sec": 600,
    "time_in_value_sec": 2400,
    "cvd_slope_5min": 0.8,
    "cvd_slope_15min": 0.6
}
```

### 2. Enhanced LLM Client (`llm_client_enhanced.py`)

**Regime Inference Rules:**

| Regime | Indicators |
|--------|------------|
| **Trend** | - Session range > 60th percentile<br>- Distance to POC > 20 ticks<br>- Time outside value > 50%<br>- CVD slope > 0.5<br>- POC crosses < 10/30min |
| **Range** | - Session range 30th-70th percentile<br>- Price rotates VAH-VAL<br>- POC crosses 10-25/30min<br>- CVD alternates direction |
| **Chop** | - Session range < 30th percentile<br>- Distance to POC < 5 ticks<br>- POC crosses > 25/30min<br>- CVD slope < 0.1 |

**Self-Learning Behavior:**
```python
# If strategy has poor performance
if win_rate < 0.4 or net_pnl < 0:
    disable_strategy()
    note_failure_conditions()

# If strategy performs well
if win_rate > 0.6 and net_pnl > 0:
    increase_risk_fraction(+0.001)
    note_success_conditions()
```

### 3. Enhanced Execution Engine (`execution_enhanced.py`)

**Active Strategy Management:**
- Strategy state that actually affects trading
- Dynamic enable/disable based on performance
- Risk fraction adjustments
- Per-strategy trade limits

**Key Methods:**
```python
apply_strategy_updates(updates)  # Apply LLM decisions
can_trade(strategy_name)        # Check if strategy can trade
process_trade_decision(decision) # Execute with strategy rules
get_performance_stats()          # Return metrics for learning
```

### 4. Enhanced Main Engine (`engine_enhanced.py`)

**Integration Features:**
- Loads persisted strategy state from logs
- Reconstructs historical notes
- Calculates performance history
- Manages importance zones
- Variable LLM call frequency based on proximity to zones

## Deployment Guide

### Step 1: Environment Setup

Ensure your `.env` file has all required variables:
```env
# OpenAI Configuration
OPENAI_API_KEY=your_api_key
OPENAI_MODEL=gpt-4-turbo-preview

# TopStep Configuration
TOPSTEP_USERNAME=your_username
TOPSTEP_PASSWORD=your_password
TOPSTEP_ACCOUNT_ID=your_account_id

# Trading Configuration
SYMBOL=NQZ5
MODE=live
ACCOUNT_BALANCE=150000

# LLM Decision Intervals (seconds)
LLM_DECISION_INTERVAL_DEFAULT=30
LLM_DECISION_INTERVAL_OUTER_BAND=15
LLM_DECISION_INTERVAL_INNER_BAND=5
```

### Step 2: Install Dependencies

```bash
# Install Python dependencies
pip install httpx pandas numpy

# Ensure TypeScript dependencies are installed
npm install
```

### Step 3: Test Individual Components

```bash
# Test enhanced feature engine
python -c "from features_enhanced import EnhancedFeatureEngine; print('Features OK')"

# Test enhanced LLM client
python -c "from llm_client_enhanced import EnhancedLLMClient; print('LLM OK')"

# Test enhanced execution
python -c "from execution_enhanced import EnhancedExecutionEngine; print('Execution OK')"
```

### Step 4: Run the Enhanced Agent

**Option 1: Direct Python Execution**
```bash
python engine_enhanced.py
```

**Option 2: PM2 Process Management**
```bash
# Add to ecosystem.config.js
pm2 start ecosystem.config.js --only fabio-enhanced

# Monitor logs
pm2 logs fabio-enhanced
```

**Option 3: Test Mode First**
```bash
# Set MODE=test in .env for paper trading
MODE=test python engine_enhanced.py
```

### Step 5: Monitor Self-Learning

Check the LLM log for learning behavior:
```bash
# Watch strategy updates
tail -f logs/llm_exchanges.jsonl | grep strategy_updates

# Check performance metrics
tail -f logs/llm_exchanges.jsonl | grep strategy_performance

# View market assessments
tail -f logs/llm_exchanges.jsonl | grep market_assessment
```

## Migration from Original System

### 1. Backup Current State
```bash
cp engine.py engine_original.py
cp llm_client.py llm_client_original.py
cp features.py features_original.py
cp execution.py execution_original.py
```

### 2. Gradual Migration Path

**Phase 1: Test Enhanced Features Only**
```python
# In your existing engine.py, import enhanced features
from features_enhanced import EnhancedFeatureEngine
feature_engine = EnhancedFeatureEngine()  # Use enhanced version
```

**Phase 2: Add Enhanced LLM**
```python
from llm_client_enhanced import EnhancedLLMClient
llm = EnhancedLLMClient(settings)  # Use enhanced LLM
```

**Phase 3: Full Enhanced System**
```python
# Switch to engine_enhanced.py completely
python engine_enhanced.py
```

## Performance Monitoring

### Key Metrics to Track

1. **Strategy Performance**
   - Win rate per strategy
   - Average P&L per strategy
   - Trade frequency per strategy

2. **Market Regime Accuracy**
   - Compare LLM inferred regime to actual market behavior
   - Track success rate by regime type

3. **Self-Learning Effectiveness**
   - Strategy parameter changes over time
   - Performance improvement trends
   - Adaptation speed to new conditions

### Dashboard Integration

The enhanced system works with the existing multi-symbol dashboard:
```javascript
// Dashboard will show enhanced metrics
socket.on('trade_update', (data) => {
    // Now includes strategy performance
    console.log(data.strategy_performance);
    // Shows active strategies
    console.log(data.active_strategies);
    // Displays market assessment
    console.log(data.market_assessment);
});
```

## Troubleshooting

### Common Issues

**1. LLM Not Inferring Regime Correctly**
- Check if POC data is being calculated
- Verify CVD slopes are computed
- Ensure time windows are properly tracked

**2. Strategies Not Updating**
- Check `apply_strategy_updates()` is being called
- Verify strategy_state persistence
- Look for strategy_tweaks in LLM responses

**3. Performance Not Improving**
- Review historical_notes in logs
- Check if performance metrics are accurate
- Ensure sufficient trade history (30+ trades)

**4. High LLM Costs**
- Adjust decision intervals in .env
- Increase importance zone thresholds
- Consider using gpt-3.5-turbo for testing

### Debug Commands

```bash
# Check POC crosses
grep "poc_cross_stats" logs/llm_exchanges.jsonl | tail -5

# View strategy updates
grep "strategy_tweaks" logs/llm_exchanges.jsonl | tail -5

# Monitor performance
grep "strategy_performance" logs/llm_exchanges.jsonl | tail -5

# Check market assessments
grep "market_assessment" logs/llm_exchanges.jsonl | tail -5
```

## Advanced Configuration

### Custom Strategy Addition

Add new strategies to `execution_enhanced.py`:
```python
def initialize_default_strategies(self):
    self.strategy_state["YourNewStrategy"] = {
        "enabled": False,
        "risk_fraction": 0.005,
        "max_trades_per_session": 3,
        "min_rr": 2.0,
        "stop_ticks": 20,
        "target_ticks": 40
    }
```

### Regime Inference Tuning

Modify thresholds in `llm_client_enhanced.py`:
```python
# Adjust these values based on your market
"Trend":
- Session range > 60th percentile  # Change to 70th for stricter
- Distance to POC > 20 ticks       # Increase for less sensitive
```

### Performance Feedback Tuning

Adjust learning thresholds:
```python
# In llm_client_enhanced.py system prompt
if win_rate < 0.4:  # Make stricter with 0.35
    disable_strategy()
if win_rate > 0.6:  # Make stricter with 0.65
    increase_risk()
```

## Summary

The enhanced self-learning system transforms the Fabio agent from a rule-based system to a truly adaptive, self-improving trading agent. Key improvements:

1. **POC Cross Tracking** - Critical market structure information
2. **Raw Stats for Inference** - LLM determines market regime
3. **Performance Feedback** - Strategies adapt based on results
4. **Active Management** - Strategy updates affect execution
5. **Historical Learning** - Notes to future self provide context

The system is designed to continuously improve through:
- Learning which strategies work in which conditions
- Adapting risk parameters based on performance
- Remembering lessons from past trading sessions
- Inferring market conditions rather than being told

## Next Steps

1. **Deploy in Test Mode** - Run with `MODE=test` for 1-2 days
2. **Review Learning Logs** - Check strategy adaptations
3. **Fine-tune Thresholds** - Adjust based on your market
4. **Go Live Gradually** - Start with small position sizes
5. **Monitor Continuously** - Watch for unexpected behaviors

For questions or issues, check the logs in `logs/llm_exchanges.jsonl` which contain complete records of all learning decisions and market assessments.