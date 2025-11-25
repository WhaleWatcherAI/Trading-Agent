#!/bin/bash

# Test Launch NQ without output redirection
echo "🚀 Starting NQ Fabio Trading Agent (TEST)..."
cd /Users/coreycosta/trading-agent

# Kill any existing instances
pkill -f "live-fabio-agent-playbook.cjs" 2>/dev/null
rm -f /tmp/fabio-nq.lock
sleep 1

# Start the agent WITHOUT output redirection
echo "📊 Starting NQ Agent..."
OPENAI_MODEL=deepseek-reasoner npx tsx live-fabio-agent-playbook.ts
