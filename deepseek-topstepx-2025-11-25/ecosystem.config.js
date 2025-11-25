module.exports = {
  apps: [
    {
      name: 'fabio-nq',
      script: 'npx',
      args: 'tsx live-fabio-agent-playbook.ts',
      cwd: '/Users/coreycosta/trading-agent',
      error_file: 'logs/fabio-nq-error.log',
      out_file: 'logs/fabio-nq.log',
      time: true,
      autorestart: true,
      max_restarts: 3,
      min_uptime: '30s',
      restart_delay: 10000,
      max_memory_restart: '2G',
      watch: false,
      env: {
        NODE_ENV: 'production',
        OPENAI_MODEL: 'deepseek-reasoner'
      }
    },
    {
      name: 'fabio-gold',
      script: 'npx',
      args: 'tsx live-fabio-agent-playbook-mgc.ts',
      cwd: '/Users/coreycosta/trading-agent',
      error_file: 'logs/fabio-gold-error.log',
      out_file: 'logs/fabio-gold.log',
      time: true,
      autorestart: true,
      max_restarts: 3,
      min_uptime: '30s',
      restart_delay: 10000,
      max_memory_restart: '2G',
      watch: false,
      env: {
        NODE_ENV: 'production',
        OPENAI_MODEL: 'deepseek-reasoner'
      }
    },
    {
      name: 'fabio-gold-simple',
      script: 'npx',
      args: 'tsx live-fabio-agent-playbook-mgc-simple.ts',
      cwd: '/Users/coreycosta/trading-agent',
      error_file: 'logs/fabio-gold-simple-error.log',
      out_file: 'logs/fabio-gold-simple.log',
      time: true,
      autorestart: true,
      max_restarts: 3,
      min_uptime: '30s',
      restart_delay: 10000,
      max_memory_restart: '2G',
      watch: false,
      env: {
        NODE_ENV: 'production',
        OPENAI_MODEL: 'deepseek-reasoner'
      }
    }
  ]
};
