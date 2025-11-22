#!/bin/bash

# Fabio Dashboard Integration - Startup Script
# This script starts the Fabio LLM agent with dashboard integration

set -e

echo "================================================================================"
echo "FABIO LLM AGENT + DASHBOARD INTEGRATION"
echo "================================================================================"
echo ""

# Check if Node.js dashboard server is running
echo "Checking if dashboard server is running..."
if curl -s http://localhost:3337/api/strategy/config > /dev/null 2>&1; then
    echo "✅ Dashboard server is already running on port 3337"
else
    echo "⚠️  Dashboard server is not running!"
    echo ""
    echo "Please start the Node.js dashboard server in another terminal:"
    echo "  npx tsx live-topstepx-nq-ict.ts"
    echo ""
    echo "Or run this command:"
    echo "  npx tsx live-topstepx-nq-ict.ts &"
    echo ""
    read -p "Start it now in background? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Starting dashboard server in background..."
        npx tsx live-topstepx-nq-ict.ts > /tmp/fabio-dashboard-server.log 2>&1 &
        SERVER_PID=$!
        echo "Dashboard server starting (PID: $SERVER_PID)..."
        echo "Waiting 10 seconds for it to initialize..."
        sleep 10

        if curl -s http://localhost:3337/api/strategy/config > /dev/null 2>&1; then
            echo "✅ Dashboard server is now running!"
        else
            echo "❌ Dashboard server failed to start. Check /tmp/fabio-dashboard-server.log"
            exit 1
        fi
    else
        echo "Exiting. Please start the dashboard server first."
        exit 1
    fi
fi

echo ""
echo "================================================================================"
echo ""

# Default values
MODE="${TRADING_MODE:-paper_trading}"
SYMBOL="${TRADING_SYMBOL:-NQZ5}"
DASHBOARD_URL="http://localhost:3337"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --symbol)
            SYMBOL="$2"
            shift 2
            ;;
        --dashboard-url)
            DASHBOARD_URL="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --mode MODE              Trading mode (paper_trading or live_trading)"
            echo "                           Default: \$TRADING_MODE or paper_trading"
            echo "  --symbol SYMBOL          Trading symbol (e.g., NQZ5, ESZ5)"
            echo "                           Default: \$TRADING_SYMBOL or NQZ5"
            echo "  --dashboard-url URL      Dashboard server URL"
            echo "                           Default: http://localhost:3337"
            echo "  --help                   Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --symbol NQZ5 --mode paper_trading"
            echo "  $0 --symbol ESZ5 --mode live_trading"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "Configuration:"
echo "  Symbol: $SYMBOL"
echo "  Mode: $MODE"
echo "  Dashboard URL: $DASHBOARD_URL"
echo ""
echo "================================================================================"
echo ""

# Check Python dependencies
echo "Checking Python dependencies..."
python3 -c "import socketio, aiohttp" 2>/dev/null || {
    echo "❌ Missing Python dependencies!"
    echo ""
    echo "Install with:"
    echo "  pip3 install 'python-socketio[asyncio_client]' aiohttp"
    echo ""
    exit 1
}
echo "✅ Python dependencies installed"
echo ""

# Show dashboard URL
echo "Dashboard is available at:"
echo "  🌐 $DASHBOARD_URL"
echo ""
echo "Open this URL in your browser to see Fabio's trades in real-time!"
echo ""
echo "================================================================================"
echo ""

# Start Fabio with dashboard integration
echo "Starting Fabio LLM agent with dashboard integration..."
echo ""
python3 fabio_dashboard.py --symbol "$SYMBOL" --mode "$MODE" --dashboard-url "$DASHBOARD_URL"
