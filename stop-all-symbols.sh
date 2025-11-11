#!/bin/bash
# Stop all trading symbol instances

echo "🛑 Stopping Multi-Symbol Trading System..."
echo "========================================"

# Kill all instances
pkill -f "live-topstepx-nq-winner" && echo "  ✓ MNQ stopped"
pkill -f "live-topstepx-mes-winner" && echo "  ✓ MES stopped"
pkill -f "live-topstepx-mgc-winner" && echo "  ✓ MGC stopped"
pkill -f "live-topstepx-m6e-winner" && echo "  ✓ M6E stopped"

sleep 1

echo ""
echo "========================================"
echo "✅ All instances stopped"
echo ""
