import * as signalR from '@microsoft/signalr';
import { HttpTransportType } from '@microsoft/signalr';
import dotenv from 'dotenv';
import { authenticate } from './lib/topstepx';

dotenv.config();

// Production WebSocket URLs
const MARKET_HUB_URL = process.env.TOPSTEPX_MARKET_HUB_URL || 'https://rtc.topstepx.com/hubs/market';
const USER_HUB_URL = process.env.TOPSTEPX_USER_HUB_URL || 'https://rtc.topstepx.com/hubs/user';
const CONTRACT_ID = process.env.TOPSTEPX_CONTRACT_ID || 'CON.F.US.MES.Z25';

interface MarketDataSnapshot {
  contractId: string;
  bid: number;
  ask: number;
  last: number;
  volume?: number;
  timestamp: string;
}

interface BarData {
  t: string; // timestamp
  o: number; // open
  h: number; // high
  l: number; // low
  c: number; // close
  v?: number; // volume
}

async function testWebSocket() {
  console.log('🔌 Testing TopstepX WebSocket Connection...\n');

  try {
    // Step 1: Get JWT token
    console.log('1️⃣  Getting JWT token from TopstepX REST API...');
    const jwtToken = await authenticate();
    console.log('✅ JWT token obtained\n');

    // Step 2: Build Market Hub connection
    console.log('2️⃣  Building Market Hub connection...');
    console.log('    URL:', MARKET_HUB_URL);
    console.log('    Contract:', CONTRACT_ID, '\n');

    const marketHub = new signalR.HubConnectionBuilder()
      .withUrl(`${MARKET_HUB_URL}?access_token=${encodeURIComponent(jwtToken)}`, {
        skipNegotiation: true,
        transport: HttpTransportType.WebSockets,
        accessTokenFactory: () => jwtToken,
      })
      .withAutomaticReconnect()
      .configureLogging(signalR.LogLevel.Information)
      .build();

    // Step 3: Set up event handlers
    let dataReceived = 0;

    const handleQuote = (contractId: string, quote: any) => {
      dataReceived++;
      console.log(`\n📊 Gateway Quote (${contractId}):`);
      console.log(JSON.stringify(quote, null, 2));
    };
    marketHub.on('GatewayQuote', handleQuote);
    marketHub.on('gatewayquote', handleQuote);

    const handleTrade = (contractId: string, trade: any) => {
      dataReceived++;
      console.log(`\n📈 Gateway Trade (${contractId}):`);
      console.log(JSON.stringify(trade, null, 2));
    };
    marketHub.on('GatewayTrade', handleTrade);
    marketHub.on('gatewaytrade', handleTrade);

    const handleDepth = (contractId: string, depth: any) => {
      dataReceived++;
      console.log(`\n📚 Gateway Market Depth (${contractId}):`);
      console.log(JSON.stringify(depth, null, 2));
    };
    marketHub.on('GatewayMarketDepth', handleDepth);
    marketHub.on('gatewaydepth', handleDepth);

    marketHub.onreconnecting((error) => {
      console.log('🔄 Reconnecting...', error?.message);
    });

    marketHub.onreconnected((connectionId) => {
      console.log('✅ Reconnected! Connection ID:', connectionId);
    });

    marketHub.onclose((error) => {
      console.log('❌ Connection closed:', error?.message);
    });

    // Step 4: Start connection
    console.log('3️⃣  Starting Market Hub connection...');
    await marketHub.start();
    console.log('✅ Connected to Market Hub!\n');

    // Step 5: Subscribe to market data
    console.log('4️⃣  Subscribing to market data for', CONTRACT_ID);
    try {
      await marketHub.invoke('SubscribeContractQuotes', CONTRACT_ID);
      console.log('✅ Subscribed to Contract Quotes');
    } catch (error: any) {
      console.log('⚠️  SubscribeContractQuotes error:', error.message);
    }

    try {
      await marketHub.invoke('SubscribeContractTrades', CONTRACT_ID);
      console.log('✅ Subscribed to Contract Trades');
    } catch (error: any) {
      console.log('⚠️  SubscribeContractTrades error:', error.message);
    }

    try {
      await marketHub.invoke('SubscribeContractMarketDepth', CONTRACT_ID);
      console.log('✅ Subscribed to Contract Market Depth\n');
    } catch (error: any) {
      console.log('⚠️  SubscribeContractMarketDepth error:', error.message);
    }

    console.log('📡 Listening for data... (will run for 60 seconds)\n');
    console.log('Press Ctrl+C to stop\n');

    // Keep connection alive for 60 seconds to receive data
    await new Promise((resolve) => setTimeout(resolve, 60000));

    // Cleanup
    console.log(`\n6️⃣  Closing connection... (received ${dataReceived} data events)`);
    await marketHub.stop();
    console.log('✅ Connection closed cleanly\n');

  } catch (error: any) {
    console.error('❌ Error:', error.message);
    if (error.response) {
      console.error('Response data:', error.response.data);
    }
    process.exit(1);
  }
}

// Run the test
testWebSocket().catch(console.error);
