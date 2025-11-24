module.exports = {
  apps: [
    {
      name: 'mnq-trading',
      script: 'npx',
      args: 'tsx live-topstepx-nq-winner-enhanced.ts',
      cwd: '/Users/coreycosta/trading-agent',
      error_file: 'logs/mnq-error.log',
      out_file: 'logs/mnq.log',
      time: true,
      autorestart: true,
      max_restarts: 3,  // Reduced to prevent infinite loops
      min_uptime: '30s', // Increased to ensure process is stable
      restart_delay: 10000, // Exponential backoff would be better
      max_memory_restart: '2G', // Restart if memory usage exceeds 2GB
      watch: false,
      env: {
        NODE_ENV: 'production'
      }
    },
    {
      name: 'mes-trading',
      script: 'npx',
      args: 'tsx live-topstepx-mes-winner.ts',
      cwd: '/Users/coreycosta/trading-agent',
      error_file: 'logs/mes-error.log',
      out_file: 'logs/mes.log',
      time: true,
      autorestart: true,
      max_restarts: 3,
      min_uptime: '30s',
      restart_delay: 10000,
      max_memory_restart: '2G',
      watch: false,
      env: {
        NODE_ENV: 'production'
      }
    },
    {
      name: 'mgc-trading',
      script: 'npx',
      args: 'tsx live-topstepx-mgc-winner.ts',
      cwd: '/Users/coreycosta/trading-agent',
      error_file: 'logs/mgc-error.log',
      out_file: 'logs/mgc.log',
      time: true,
      autorestart: true,
      max_restarts: 3,
      min_uptime: '30s',
      restart_delay: 10000,
      max_memory_restart: '2G',
      watch: false,
      env: {
        NODE_ENV: 'production'
      }
    },
    {
      name: 'm6e-trading',
      script: 'npx',
      args: 'tsx live-topstepx-m6e-winner.ts',
      cwd: '/Users/coreycosta/trading-agent',
      error_file: 'logs/m6e-error.log',
      out_file: 'logs/m6e.log',
      time: true,
      autorestart: true,
      max_restarts: 3,
      min_uptime: '30s',
      restart_delay: 10000,
      max_memory_restart: '2G',
      watch: false,
      env: {
        NODE_ENV: 'production'
      }
    },
    {
      name: 'fabio-nq',
      script: 'python3',
      args: 'fabio_dashboard.py --symbol NQZ5 --mode paper_trading',
      cwd: '/Users/coreycosta/trading-agent',
      error_file: 'logs/fabio-error.log',
      out_file: 'logs/fabio.log',
      time: true,
      autorestart: true,
      max_restarts: 3,
      min_uptime: '30s',
      restart_delay: 10000,
      max_memory_restart: '2G',
      watch: false,
      env: {
        NODE_ENV: 'production'
      }
    }
  ]
};
