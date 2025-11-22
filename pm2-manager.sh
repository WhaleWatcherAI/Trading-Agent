#!/bin/bash

# PM2 Manager Script for Trading Agent
# Prevents duplicate processes and manages all trading strategies

echo "🤖 Trading Agent PM2 Manager"
echo "============================"

case "$1" in
  start)
    echo "Starting all trading processes..."
    pm2 start ecosystem.config.js
    pm2 save
    echo "✅ All processes started"
    pm2 status
    ;;

  stop)
    echo "Stopping all trading processes..."
    pm2 stop all
    echo "✅ All processes stopped"
    ;;

  restart)
    echo "Restarting all trading processes..."
    pm2 restart all
    echo "✅ All processes restarted"
    pm2 status
    ;;

  status)
    pm2 status
    ;;

  logs)
    if [ -z "$2" ]; then
      pm2 logs --lines 50
    else
      pm2 logs "$2" --lines 50
    fi
    ;;

  monitor)
    pm2 monit
    ;;

  clean)
    echo "⚠️  Cleaning up all processes and logs..."
    pm2 kill
    pm2 flush
    echo "✅ Cleanup complete"
    ;;

  setup)
    echo "Installing PM2 globally..."
    npm install -g pm2
    pm2 install pm2-logrotate
    pm2 set pm2-logrotate:max_size 100M
    pm2 set pm2-logrotate:retain 7
    echo "✅ PM2 setup complete"
    ;;

  *)
    echo "Usage: $0 {start|stop|restart|status|logs|monitor|clean|setup}"
    echo ""
    echo "Commands:"
    echo "  start    - Start all trading processes"
    echo "  stop     - Stop all trading processes"
    echo "  restart  - Restart all trading processes"
    echo "  status   - Show process status"
    echo "  logs [name] - Show recent logs (optionally for specific process)"
    echo "  monitor  - Open PM2 monitoring dashboard"
    echo "  clean    - Kill all processes and flush logs"
    echo "  setup    - Install PM2 and configure log rotation"
    exit 1
    ;;
esac