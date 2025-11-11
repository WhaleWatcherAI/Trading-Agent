/** @type {import('next').NextConfig} */
const nextConfig = {
  serverExternalPackages: ['@microsoft/signalr'],
  turbopack: {
    root: '/Users/coreycosta/trading-agent',
  },
};

module.exports = nextConfig;
