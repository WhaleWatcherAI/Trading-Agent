#!/bin/bash
# Start all 4 trading symbol instances

echo "🚀 Starting Multi-Symbol Trading System..."
echo "========================================"

# Create logs directory if it doesn't exist
mkdir -p logs

# Kill any existing instances
echo "Stopping any running instances..."
pkill -f "live-topstepx.*winner" 2>/dev/null
sleep 2

# Start each instance in background
echo ""
echo "Starting MNQ instance on port 3333..."
npx tsx live-topstepx-nq-winner-enhanced.ts >> logs/mnq.log 2>&1 &
MNQ_PID=$!
echo "  ✓ MNQ started (PID: $MNQ_PID)"

sleep 3

echo "Starting MES instance on port 3334..."
npx tsx live-topstepx-mes-winner.ts >> logs/mes.log 2>&1 &
MES_PID=$!
echo "  ✓ MES started (PID: $MES_PID)"

sleep 3

echo "Starting MGC instance on port 3335..."
npx tsx live-topstepx-mgc-winner.ts >> logs/mgc.log 2>&1 &
MGC_PID=$!
echo "  ✓ MGC started (PID: $MGC_PID)"

sleep 3

echo "Starting M6E instance on port 3336..."
npx tsx live-topstepx-m6e-winner.ts >> logs/m6e.log 2>&1 &
M6E_PID=$!
echo "  ✓ M6E started (PID: $M6E_PID)"

sleep 3

echo ""
echo "========================================"
echo "✅ All instances started successfully!"
echo ""
echo "Dashboards:"
echo "  • MNQ: http://localhost:3333"
echo "  • MES: http://localhost:3334"
echo "  • MGC: http://localhost:3335"
echo "  • M6E: http://localhost:3336"
echo ""
echo "  • UNIFIED: http://localhost:3333/multi-symbol-dashboard.html"
echo ""
echo "Logs:"
echo "  • MNQ: logs/mnq.log"
echo "  • MES: logs/mes.log"
echo "  • MGC: logs/mgc.log"
echo "  • M6E: logs/m6e.log"
echo ""
echo "To stop all instances: ./stop-all-symbols.sh"
echo "To view logs: tail -f logs/*.log"
echo ""
