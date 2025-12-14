#!/bin/bash
# Stop MGCG6 Trading Agent
echo "Stopping MGCG6 Agent..."

pkill -f "live-fabio-agent-playbook-mgc.ts" 2>/dev/null

if [ -f /tmp/mgcg6-agent.pid ]; then
    kill -9 $(cat /tmp/mgcg6-agent.pid) 2>/dev/null
    rm -f /tmp/mgcg6-agent.pid
fi

sleep 1

# Verify stopped
if pgrep -f "live-fabio-agent-playbook-mgc.ts" > /dev/null; then
    echo "Agent still running, force killing..."
    pkill -9 -f "live-fabio-agent-playbook-mgc.ts" 2>/dev/null
fi

echo "MGCG6 Agent stopped."
