#!/bin/bash
# Restart all trading strategies sequentially to avoid WebSocket connection conflicts

echo "Restarting trading strategies sequentially..."

echo "Restarting MNQ..."
pm2 restart mnq-trading
sleep 3

echo "Restarting MES..."
pm2 restart mes-trading
sleep 3

echo "Restarting MGC..."
pm2 restart mgc-trading
sleep 3

echo "Restarting M6E..."
pm2 restart m6e-trading
sleep 3

echo "All strategies restarted!"
pm2 status
