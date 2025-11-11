import * as signalR from '@microsoft/signalr';
import { HttpTransportType } from '@microsoft/signalr';
import dotenv from 'dotenv';
import { authenticate } from './lib/topstepx';

dotenv.config();

const MARKET_HUB_URL = process.env.TOPSTEPX_MARKET_HUB_URL || 'https://gateway-rtc-demo.s2f.projectx.com/hubs/market';

async function testSimpleWebSocket() {
  console.log('🔌 Testing Simple TopstepX WebSocket Connection...\n');

  try {
    console.log('1️⃣  Getting JWT token...');
    const jwtToken = await authenticate();
    console.log('✅ JWT token obtained\n');

    console.log('2️⃣  Building connection to:', MARKET_HUB_URL);
    const marketHub = new signalR.HubConnectionBuilder()
      .withUrl(`${MARKET_HUB_URL}?access_token=${encodeURIComponent(jwtToken)}`, {
        skipNegotiation: true,
        transport: HttpTransportType.WebSockets,
        accessTokenFactory: () => jwtToken,
      })
      .withAutomaticReconnect()
      .configureLogging(signalR.LogLevel.Debug) // More verbose logging
      .build();

    // Listen for ALL events
    marketHub.on('*', (...args: any[]) => {
      console.log('\n📡 Received event:');
      console.log('Arguments:', JSON.stringify(args, null, 2));
    });

    marketHub.onreconnecting((error) => {
      console.log('🔄 Reconnecting...', error?.message);
    });

    marketHub.onreconnected((connectionId) => {
      console.log('✅ Reconnected! Connection ID:', connectionId);
    });

    marketHub.onclose((error) => {
      console.log('❌ Connection closed:', error?.message || 'No error message');
      if (error) {
        console.log('Error details:', error);
      }
    });

    console.log('3️⃣  Starting connection...');
    await marketHub.start();
    console.log('✅ Connected!\n');

    console.log('4️⃣  Connection state:', marketHub.state);
    console.log('📡 Listening for events... (30 seconds, no subscriptions)\n');
    console.log('Press Ctrl+C to stop\n');

    // Wait 30 seconds to see if the connection stays alive
    await new Promise((resolve) => setTimeout(resolve, 30000));

    console.log('\n5️⃣  Connection still alive?', marketHub.state);
    await marketHub.stop();
    console.log('✅ Done\n');

  } catch (error: any) {
    console.error('❌ Error:', error.message);
    if (error.stack) {
      console.error('Stack:', error.stack);
    }
  }
}

testSimpleWebSocket().catch(console.error);
