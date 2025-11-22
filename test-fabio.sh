#!/bin/bash

echo "Testing Fabio Dashboard..."
echo "Starting in paper trading mode for NQZ5"
echo "=================================="

# Run for a few seconds then quit
python3 fabio_dashboard.py --symbol NQZ5 --mode paper_trading &
PID=$!

echo "Fabio PID: $PID"
echo "Running for 15 seconds to observe output..."

sleep 15

echo "=================================="
echo "Stopping Fabio..."
kill $PID 2>/dev/null

echo "Test complete. Check if you saw any LLM prompts above."