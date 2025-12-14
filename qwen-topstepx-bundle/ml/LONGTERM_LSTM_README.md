# Long-Term Multi-Timeframe LSTM - TEST VERSION

This system creates a long-term trend detection LSTM that uses 1-hour, 4-hour, and daily bars with a full year of lookback (365 days).

## Architecture

### Multi-Timeframe LSTM
- **Timeframes**: 1-hour, 4-hour, daily bars (aligned)
- **Lookback**: 365 days (1 year)
- **Features per timestep**: 26
  - Daily OHLCV (5)
  - Daily Volume Profile (price vs POC/VAH/VAL) (3)
  - 4-hour OHLCV (5)
  - 1-hour OHLCV (5)
  - 1-hour CVD and CVD EMA (2)
  - Daily RSI, MACD, ATR (3)
  - Bollinger Band position (1)
  - Trend features (MA20, MA50) (2)
- **Output**: 32-dimensional embedding
- **Architecture**: 3-layer LSTM with 128 hidden units

### Features
- **CVD (Cumulative Volume Delta)**: Calculated on 1-hour bars to detect institutional buying/selling pressure
- **Daily Volume Profile**: POC, VAH, VAL calculated with 30-day rolling window
- **Multi-timeframe alignment**: Daily, 4h, and 1h bars synchronized
- **Self-supervised training**: Predicts 5-day forward price direction

## Data Requirements

### 2024 ENQ Contracts
The system fetches data from all 2024 E-mini NASDAQ contracts:
- **CON.F.US.ENQ.H24** (March 2024): Jan 1 - Mar 15
- **CON.F.US.ENQ.M24** (June 2024): Mar 1 - Jun 21
- **CON.F.US.ENQ.U24** (September 2024): Jun 1 - Sep 20
- **CON.F.US.ENQ.Z24** (December 2024): Sep 1 - Dec 31

## Usage

### Step 1: Fetch Historical Data

```bash
cd /home/costa/Trading-Agent/qwen-topstepx-bundle/ml

# Fetch 1h, 4h, and daily bars for all 2024 ENQ contracts
python scripts/fetch_longterm_bars_TEST.py --output-dir data_TEST/longterm
```

This will:
1. Fetch bars for each contract and timeframe
2. Save individual contract files (e.g., `bars_1h_enq_h24.json`)
3. Merge all contracts into combined files (e.g., `bars_1h_enq_2024_merged.json`)

**Expected output files:**
- `data_TEST/longterm/bars_1h_enq_2024_merged.json` (~8,760 bars for 365 days)
- `data_TEST/longterm/bars_4h_enq_2024_merged.json` (~2,190 bars)
- `data_TEST/longterm/bars_1d_enq_2024_merged.json` (~365 bars)

### Step 2: Train the Long-Term LSTM

```bash
# Train the model
python scripts/longterm_lstm_backtest_TEST.py \
  --data-dir data_TEST/longterm \
  --output-dir models_TEST/longterm \
  --epochs 50
```

This will:
1. Load 1h, 4h, and daily bars
2. Calculate 1-hour CVD
3. Calculate daily Volume Profile
4. Extract aligned multi-timeframe features
5. Train LSTM to predict 5-day price direction
6. Save model to `models_TEST/longterm/longterm_multitf_lstm.pt`

**Training output:**
- Feature extraction for 365-day sequences
- Self-supervised learning (predict if price higher in 5 days)
- Model checkpoint with architecture config

### Step 3: Use in Ensemble (Future Integration)

The trained LSTM can be loaded and used as a trend filter:

```python
import torch
from longterm_lstm_backtest_TEST import LongTermMultiTimeframeLSTM, extract_longterm_features

# Load model
checkpoint = torch.load('models_TEST/longterm/longterm_multitf_lstm.pt')
model = LongTermMultiTimeframeLSTM(
    input_size=checkpoint['input_size'],
    hidden_size=checkpoint['hidden_size'],
    num_layers=checkpoint['num_layers'],
    output_size=checkpoint['output_size']
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Extract features from current data
features = extract_longterm_features(
    bars_1h, bars_4h, bars_daily,
    cvd_1h, cvd_ema_1h, daily_vp,
    num_days=365
)

# Get trend embedding
with torch.no_grad():
    trend_embedding = model(torch.FloatTensor(features).unsqueeze(0))

# Use embedding with XGBoost or as trend filter
# trend_embedding is [1, 32] - can be concatenated with other features
```

## Architecture Details

### LSTM Model
```
LongTermMultiTimeframeLSTM(
  (lstm): LSTM(26, 128, num_layers=3, batch_first=True, dropout=0.2)
  (layer_norm): LayerNorm(128)
  (fc): Linear(128, 32)
  (activation): Tanh()
)
```

### Feature Alignment
Each of the 365 daily timesteps includes:
- **Daily bar**: OHLCV from that day
- **4-hour bar**: Last 4h bar of that day (simplified alignment)
- **1-hour bar**: Last 1h bar of that day
- **Indicators**: Calculated on respective timeframes

### CVD Calculation (1-hour)
```
For each 1-hour bar:
  close_position = (close - low) / (high - low)
  delta = volume * (2 * close_position - 1)
  CVD[i] = CVD[i-1] + delta
  CVD_EMA = EMA(CVD, period=20)
```

### Volume Profile (Daily)
```
For each day:
  Use last 30 days of data
  Bin prices into levels
  Distribute volume across bins
  POC = price level with most volume
  VAH/VAL = 70% volume area bounds
```

## File Structure

```
data_TEST/longterm/
├── bars_1h_enq_h24.json          # March contract 1h bars
├── bars_1h_enq_m24.json          # June contract 1h bars
├── bars_1h_enq_u24.json          # September contract 1h bars
├── bars_1h_enq_z24.json          # December contract 1h bars
├── bars_1h_enq_2024_merged.json  # All 2024 1h bars merged
├── bars_4h_enq_2024_merged.json  # All 2024 4h bars merged
└── bars_1d_enq_2024_merged.json  # All 2024 daily bars merged

models_TEST/longterm/
└── longterm_multitf_lstm.pt      # Trained model checkpoint

scripts/
├── fetch_longterm_bars_TEST.py          # Data fetching script
└── longterm_lstm_backtest_TEST.py       # LSTM training script
```

## Environment Variables Required

Set these in your `.env` file:

```bash
TOPSTEPX_API_KEY=your_api_key
TOPSTEPX_USERNAME=your_username
TOPSTEPX_BASE_URL=https://api.topstepx.com
```

## Expected Data Volume

### API Calls
- 4 contracts × 3 timeframes = **12 API requests**
- Rate limited to ~1 request per second
- Total fetch time: ~2-3 minutes

### Data Size
- 1-hour bars: ~8,760 bars (365 days × 24 hours)
- 4-hour bars: ~2,190 bars (365 days × 6 4h-bars)
- Daily bars: ~365 bars
- Total JSON size: ~5-10 MB

## Training Performance

### Expected Metrics
- **Training samples**: ~200-300 (depends on data availability)
- **Epochs**: 50 (adjustable)
- **Training time**:
  - CPU: ~5-10 minutes
  - GPU: ~1-2 minutes
- **Accuracy**: Target >55% (random baseline is 50%)

### Self-Supervised Task
The model predicts whether price will be higher 5 days in the future:
- **Label 1**: Price[day+5] > Price[day]
- **Label 0**: Price[day+5] <= Price[day]

This teaches the model to recognize bullish/bearish trends.

## Integration with Existing Ensemble

To add this as a trend filter to your existing short-term ensemble:

1. **Train the long-term LSTM** using this system
2. **Load in your live/backtest script** alongside existing LSTMs
3. **Extract trend embedding** from last 365 days
4. **Add to XGBoost features** or use as pre-filter
5. **Only take trades aligned with trend**:
   - Calculate trend score from embedding
   - For LONG trades: require bullish trend
   - For SHORT trades: require bearish trend

Example filter logic:
```python
# Get trend embedding
trend_emb = longterm_model(features_365d)

# Simple trend classifier (sum of embedding values)
trend_score = trend_emb.sum().item()

# Filter trades by trend alignment
if signal_direction == 'long' and trend_score < 0:
    skip_trade()  # Bearish trend, skip long
elif signal_direction == 'short' and trend_score > 0:
    skip_trade()  # Bullish trend, skip short
```

## Advantages Over Short-Term LSTM

1. **Macro trend detection**: Sees the bigger picture (weeks/months)
2. **Reduces false signals**: Filters counter-trend scalps
3. **Position bias**: Prefer longs in uptrends, shorts in downtrends
4. **Regime awareness**: Detects bull/bear market phases
5. **Smoother signals**: Less noise from intraday fluctuations

## Next Steps

1. ✅ Fetch 2024 historical data
2. ✅ Train long-term LSTM
3. ⬜ Evaluate trend predictions
4. ⬜ Integrate with existing ensemble
5. ⬜ Backtest with trend filter
6. ⬜ Compare performance vs no filter
7. ⬜ Deploy to live trading

## Notes

- This is a **TEST version** - all files use `_TEST` suffix
- Original production models remain untouched
- Data fetching respects TopstepX rate limits
- Training is self-supervised (no manual labels needed)
- Model outputs 32-dim embedding (same as other LSTMs)
- Can be combined with existing short-term LSTMs in final XGBoost

## Troubleshooting

### "Not enough daily bars"
- Need at least 365 days of data
- Check if all 2024 contracts fetched successfully
- Verify merged file has enough bars

### "Rate limited"
- Script automatically retries with backoff
- Wait time increases with each retry
- Total fetch may take 5-10 minutes

### "Feature extraction failed"
- Check bar data alignment
- Ensure 1h, 4h, daily bars are all present
- Verify bars have required fields: t, o, h, l, c, v

### Low training accuracy
- 50-55% is expected (only slightly better than random)
- This is normal for self-supervised learning
- The embedding quality matters more than classification accuracy
- Use embedding as features, not as direct predictor
