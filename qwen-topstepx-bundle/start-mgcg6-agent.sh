#!/bin/bash
# Start MGCG6 Trading Agent
cd /home/costa/Trading-Agent/qwen-topstepx-bundle

# Kill any existing instance
pkill -f "live-fabio-agent-playbook-mgc.ts" 2>/dev/null
sleep 1

# Start the agent
export NODE_PATH=/home/costa/Trading-Agent/qwen-topstepx-bundle/node_modules
export DASHBOARD_PORT=3338
export LIVE_TRADING=true
export TOPSTEPX_SYMBOL=MGCG6
export TOPSTEPX_ACCOUNT_ID=13230351
export TOPSTEPX_ENABLE_NATIVE_BRACKETS=true
export OLLAMA_HOST=http://127.0.0.1:11434
export OLLAMA_MODEL=qwen2.5:7b

nohup npx tsx live-fabio-agent-playbook-mgc.ts >> fabio-mgc.log 2>&1 &
echo $! > /tmp/mgcg6-agent.pid

echo "MGCG6 Agent started with PID: $(cat /tmp/mgcg6-agent.pid)"
echo "Dashboard: http://localhost:3338"
echo "Log: ~/Trading-Agent/qwen-topstepx-bundle/fabio-mgc.log"
sleep 2
