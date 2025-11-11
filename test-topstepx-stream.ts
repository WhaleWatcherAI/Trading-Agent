import * as signalR from '@microsoft/signalr';
import { HttpTransportType } from '@microsoft/signalr';
import dotenv from 'dotenv';
import { authenticate } from './lib/topstepx';

dotenv.config();

const MARKET_HUB_URL = process.env.TOPSTEPX_MARKET_HUB_URL!;
const USER_HUB_URL = process.env.TOPSTEPX_USER_HUB_URL!;
const CONTRACT_ID = process.env.TOPSTEPX_CONTRACT_ID!;
const ACCOUNT_ID = process.env.TOPSTEPX_ACCOUNT_ID ? Number(process.env.TOPSTEPX_ACCOUNT_ID) : undefined;

interface GatewayQuote {
  symbol: string;
  symbolName: string;
  lastPrice: number;
  bestBid: number;
  bestAsk: number;
  change: number;
  changePercent: number;
  open: number;
  high: number;
  low: number;
  lastUpdated: string;
  timestamp: string;
}

async function testStreamingWebSocket() {
  console.log('🔌 TopstepX WebSocket Streaming Test\n');
  console.log('━'.repeat(60));
  console.log('Configuration:');
  console.log(`  Market Hub: ${MARKET_HUB_URL}`);
  console.log(`  User Hub: ${USER_HUB_URL}`);
  console.log(`  Contract: ${CONTRACT_ID}`);
  console.log(`  Account ID: ${ACCOUNT_ID || 'Not set'}`);
  console.log('━'.repeat(60));
  console.log('');

  try {
    // Step 1: Get JWT token
    console.log('1️⃣  Authenticating with TopstepX API...');
    const jwtToken = await authenticate();
    console.log('✅ JWT token obtained\n');

    // Step 2: Build Market Hub connection
    console.log('2️⃣  Building Market Hub connection...');
    const marketHub = new signalR.HubConnectionBuilder()
      .withUrl(`${MARKET_HUB_URL}?access_token=${encodeURIComponent(jwtToken)}`, {
        skipNegotiation: true,
        transport: HttpTransportType.WebSockets,
        accessTokenFactory: () => jwtToken,
      })
      .withAutomaticReconnect()
      .configureLogging(signalR.LogLevel.Warning) // Less verbose
      .build();

    // Step 3: Build User Hub connection
    console.log('3️⃣  Building User Hub connection...');
    const userHub = new signalR.HubConnectionBuilder()
      .withUrl(`${USER_HUB_URL}?access_token=${encodeURIComponent(jwtToken)}`, {
        skipNegotiation: true,
        transport: HttpTransportType.WebSockets,
        accessTokenFactory: () => jwtToken,
      })
      .withAutomaticReconnect()
      .configureLogging(signalR.LogLevel.Warning)
      .build();

    // Step 4: Set up event handlers for market data
    let quoteCount = 0;
    let tradeCount = 0;
    let depthCount = 0;
    let lastQuoteTime = Date.now();

    marketHub.on('GatewayQuote', (contractId: string, quote: GatewayQuote) => {
      quoteCount++;
      lastQuoteTime = Date.now();

      const now = new Date().toLocaleTimeString();
      console.log(`\n📊 [${now}] Quote #${quoteCount} - ${quote.symbolName}`);
      console.log(`   💰 Last: ${quote.lastPrice.toFixed(2)}`);
      console.log(`   📈 Bid: ${quote.bestBid.toFixed(2)} | Ask: ${quote.bestAsk.toFixed(2)}`);
      console.log(`   📉 Spread: ${(quote.bestAsk - quote.bestBid).toFixed(2)}`);
      console.log(`   🔄 Change: ${quote.change > 0 ? '+' : ''}${quote.change.toFixed(2)} (${(quote.changePercent * 100).toFixed(2)}%)`);
      console.log(`   📊 Range: ${quote.low.toFixed(2)} - ${quote.high.toFixed(2)}`);
    });

    marketHub.on('GatewayTrade', (contractId: string, trade: any) => {
      tradeCount++;
      const now = new Date().toLocaleTimeString();
      console.log(`\n📈 [${now}] Trade #${tradeCount}`);
      console.log(`   ${JSON.stringify(trade, null, 2)}`);
    });

    marketHub.on('GatewayMarketDepth', (contractId: string, depth: any) => {
      depthCount++;
      // Don't print depth data as it's very verbose
    });

    // Step 5: Set up event handlers for user data (account, positions, orders, trades)
    userHub.on('GatewayUserAccount', (data: any) => {
      const now = new Date().toLocaleTimeString();
      console.log(`\n💼 [${now}] Account Update:`);
      console.log(`   ${JSON.stringify(data, null, 2)}`);
    });

    userHub.on('GatewayUserPosition', (data: any) => {
      const now = new Date().toLocaleTimeString();
      console.log(`\n📍 [${now}] Position Update:`);
      console.log(`   ${JSON.stringify(data, null, 2)}`);
    });

    userHub.on('GatewayUserOrder', (data: any) => {
      const now = new Date().toLocaleTimeString();
      console.log(`\n📝 [${now}] Order Update:`);
      console.log(`   ${JSON.stringify(data, null, 2)}`);
    });

    userHub.on('GatewayUserTrade', (contractId: string, trade: any) => {
      const now = new Date().toLocaleTimeString();
      console.log(`\n✅ [${now}] Trade Executed:`);
      console.log(`   ${JSON.stringify(trade, null, 2)}`);
    });

    // Reconnection handlers
    marketHub.onreconnecting((error) => {
      console.log('🔄 Market Hub reconnecting...', error?.message);
    });

    marketHub.onreconnected(() => {
      console.log('✅ Market Hub reconnected!');
    });

    userHub.onreconnecting((error) => {
      console.log('🔄 User Hub reconnecting...', error?.message);
    });

    userHub.onreconnected(() => {
      console.log('✅ User Hub reconnected!');
    });

    // Step 6: Start connections
    console.log('4️⃣  Starting Market Hub...');
    await marketHub.start();
    console.log('✅ Market Hub connected!\n');

    console.log('5️⃣  Starting User Hub...');
    await userHub.start();
    console.log('✅ User Hub connected!\n');

    // Step 7: Subscribe to market data
    console.log('6️⃣  Subscribing to market data...');
    const subscribeMarket = async () => {
      await marketHub.invoke('SubscribeContractQuotes', CONTRACT_ID);
      await marketHub.invoke('SubscribeContractTrades', CONTRACT_ID);
      await marketHub.invoke('SubscribeContractMarketDepth', CONTRACT_ID);
    };

    await subscribeMarket();
    marketHub.onreconnected(subscribeMarket);
    console.log('✅ Subscribed to Contract Quotes');
    console.log('✅ Subscribed to Contract Trades');
    console.log('✅ Subscribed to Contract Market Depth\n');

    // Step 8: Subscribe to user data (account, positions, orders)
    if (ACCOUNT_ID) {
      console.log('7️⃣  Subscribing to account data...');
      const subscribeUser = async () => {
        await userHub.invoke('SubscribeAccounts');
        await userHub.invoke('SubscribeOrders', ACCOUNT_ID);
        await userHub.invoke('SubscribePositions', ACCOUNT_ID);
        await userHub.invoke('SubscribeTrades', ACCOUNT_ID);
      };

      await subscribeUser();
      userHub.onreconnected(subscribeUser);
      console.log('✅ Subscribed to Accounts');
      console.log('✅ Subscribed to Orders');
      console.log('✅ Subscribed to Positions');
      console.log('✅ Subscribed to Trades\n');
    }

    console.log('━'.repeat(60));
    console.log('📡 LIVE STREAMING (Press Ctrl+C to stop)');
    console.log('━'.repeat(60));
    console.log('');

    // Print statistics every 10 seconds
    const statsInterval = setInterval(() => {
      const timeSinceLastQuote = Math.floor((Date.now() - lastQuoteTime) / 1000);
      console.log(`\n📊 Statistics:`);
      console.log(`   Quotes: ${quoteCount} | Trades: ${tradeCount} | Depth updates: ${depthCount}`);
      console.log(`   Last quote: ${timeSinceLastQuote}s ago`);
      console.log(`   Market Hub: ${marketHub.state} | User Hub: ${userHub.state}`);
    }, 10000);

    // Handle graceful shutdown
    process.on('SIGINT', async () => {
      console.log('\n\n⏸️  Shutting down...');
      clearInterval(statsInterval);

      await marketHub.stop();
      await userHub.stop();

      console.log('✅ Disconnected cleanly');
      console.log(`\n📊 Final Statistics:`);
      console.log(`   Total Quotes: ${quoteCount}`);
      console.log(`   Total Trades: ${tradeCount}`);
      console.log(`   Total Depth Updates: ${depthCount}`);

      process.exit(0);
    });

    // Keep the process alive
    await new Promise(() => {});

  } catch (error: any) {
    console.error('❌ Error:', error.message);
    if (error.stack) {
      console.error('Stack:', error.stack);
    }
    process.exit(1);
  }
}

testStreamingWebSocket().catch(console.error);
