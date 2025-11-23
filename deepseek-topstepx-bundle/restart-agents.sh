#!/bin/bash
# Restart trading agents with fixed bracket management code

echo "🛑 Stopping old agents..."
pkill -f "live-fabio-agent-playbook.ts"
sleep 3

echo "🧹 Cleaning up old processes..."
pkill -9 -f "live-fabio-agent-playbook.ts"
sleep 1

# Check if still running
if pgrep -f "live-fabio-agent-playbook.ts" > /dev/null; then
    echo "❌ ERROR: Old agents still running. Please manually kill processes:"
    ps aux | grep "live-fabio-agent-playbook.ts" | grep -v grep
    exit 1
fi

echo "✅ Old agents stopped"
echo ""
echo "🚀 Starting NQ agent..."
cd /Users/coreycosta/trading-agent

# Start NQ agent
TOPSTEPX_SYMBOL=NQZ5 \
TOPSTEPX_ACCOUNT_ID=13230351 \
LIVE_TRADING=true \
TOPSTEPX_ENABLE_NATIVE_BRACKETS=true \
nohup npx tsx live-fabio-agent-playbook.ts > logs/nq.log 2>&1 &

NQ_PID=$!
echo "✅ NQ agent started (PID: $NQ_PID)"
sleep 2

echo ""
echo "🚀 Starting Gold agent..."

# Start Gold agent
TOPSTEPX_SYMBOL=GCZ5 \
TOPSTEPX_ACCOUNT_ID=13230351 \
LIVE_TRADING=true \
TOPSTEPX_ENABLE_NATIVE_BRACKETS=true \
nohup npx tsx live-fabio-agent-playbook.ts > logs/gcz5.log 2>&1 &

GC_PID=$!
echo "✅ Gold agent started (PID: $GC_PID)"

echo ""
echo "⏳ Waiting 5 seconds for agents to initialize..."
sleep 5

echo ""
echo "📊 Checking agent status..."
if pgrep -f "live-fabio-agent-playbook.ts" > /dev/null; then
    echo "✅ Agents are running:"
    ps aux | grep "live-fabio-agent-playbook.ts" | grep -v grep | awk '{print "  PID", $2, "started at", $9}'
else
    echo "❌ ERROR: Agents failed to start!"
    exit 1
fi

echo ""
echo "📋 Recent NQ log output:"
tail -10 logs/nq.log
echo ""
echo "📋 Recent Gold log output:"
tail -10 logs/gcz5.log

echo ""
echo "✅ Restart complete!"
echo ""
echo "🔍 Monitor with these commands:"
echo "  tail -f logs/nq.log | grep -E '(🛡️|ExecutionManager|adjustActiveProtection)'"
echo "  tail -f logs/gcz5.log | grep -E '(🛡️|ExecutionManager|adjustActiveProtection)'"
