# Dashboard Quick Reference Card

## 🚀 Quick Start

```bash
# 1. Open dashboard in browser
open http://localhost:3337

# 2. Start Fabio with dashboard integration
python3 fabio_dashboard.py --symbol NQZ5 --mode paper_trading
```

## 📍 New Features - Where to Find Them

### 1. Threshold Controls
**Location:** Right sidebar → "Detection Thresholds"

```
Detection Thresholds
├─ Absorption (contracts)          [  50  ]
├─ Exhaustion push (contracts)      [  50  ]
└─ Drop factor (0-1)                 [ 0.4 ]
```

### 2. Volume Profile Toggle
**Location:** Right sidebar → "Volume Profile"

```
Volume Profile
└─ ☑ Show volume profile
```

### 3. LLM Prompts Viewer
**Location:** Main content → Below "Activity Log"

```
🤖 LLM Prompts & Decisions    [☐ Auto-expand] [Clear]

⚡ LLM Request [14:32:15]  ▶  ← Click to expand
⚡ LLM Request [14:32:28]  ▼  ← Expanded
⚡ LLM Request [14:32:45]  ▶
```

## 🎯 How to Use

### Adjust Thresholds
1. Find "Detection Thresholds" in right sidebar
2. Type new values in input boxes
3. Changes apply immediately

### View LLM Prompts
1. Scroll to "LLM Prompts & Decisions" panel
2. Click any prompt to expand/see full JSON
3. Check "Auto-expand" to auto-expand new prompts
4. Click "Clear" to reset display

### Toggle Volume Profile
1. Find "Volume Profile" checkbox
2. Check/uncheck to show/hide volume bars on chart

## 📊 What's in the LLM Prompts?

Each prompt shows exactly what Fabio sends to OpenAI:

- **mode** - Trading mode (live_decision)
- **symbol** - Trading instrument (NQZ5)
- **timestamp** - Exact time of request
- **price** - Current price
- **ohlc** - Open, High, Low, Close data
- **profiles** - Volume profile data
- **derived_state** - CVD, trend, volatility
- **orderflow** - Imbalance, absorption data
- **open_positions** - Current positions
- **importance_zones** - High-priority price levels
- **recent_performance_summary** - Session P&L
- **historical_notes_snippet** - Fabio's notes to self

## 🔧 Troubleshooting

**Q: Don't see LLM prompts?**
A: Make sure Fabio is running and connected to the dashboard

**Q: Prompts not expanding?**
A: Click on the ▶ arrow next to the timestamp

**Q: Volume profile not showing?**
A: Check the "Show volume profile" checkbox is checked

**Q: Threshold changes not working?**
A: Values apply immediately - watch the chart for changes

## 📖 Full Documentation

- **Complete Guide:** `ENHANCED_DASHBOARD_COMPLETE.md`
- **Integration Setup:** `FABIO_DASHBOARD_INTEGRATION.md`
- **Quick Start:** `FABIO_DASHBOARD_QUICKSTART.md`

## 🎉 Summary

✅ **Three new features added:**
1. Threshold adjustment controls
2. Volume profile toggle
3. LLM prompt viewer with timestamps

✅ **Everything works automatically** when you run Fabio

✅ **Just refresh your browser** to see the new features!

---

**Need help?** Check `ENHANCED_DASHBOARD_COMPLETE.md` for detailed instructions.
