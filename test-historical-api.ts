import { authenticate } from './lib/topstepx';
import axios from 'axios';

async function testHistoricalAPI() {
  try {
    // Authenticate
    const token = await authenticate();
    console.log('✅ Authenticated');

    const baseUrl = process.env.TOPSTEPX_BASE_URL || 'https://api.topstepx.com';
    const contractId = 'CON.F.US.ENQ.Z25'; // NQZ5

    // Try different parameter formats
    const formats = [
      {
        name: 'Format 1: Direct params',
        body: {
          contractId,
          startTime: new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString(),
          endTime: new Date().toISOString(),
          unit: 2,
          unitNumber: 5,
          limit: 100,
          live: false,
          includePartialBar: false,
        }
      },
      {
        name: 'Format 2: Nested in request',
        body: {
          request: {
            contractId,
            startTime: new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString(),
            endTime: new Date().toISOString(),
            unit: 2,
            unitNumber: 5,
            limit: 100,
            live: false,
            includePartialBar: false,
          }
        }
      }
    ];

    for (const format of formats) {
      console.log(`\n📝 Testing: ${format.name}`);
      try {
        const response = await axios.post(
          `${baseUrl}/api/History/retrieveBars`,
          format.body,
          {
            headers: {
              'Authorization': `Bearer ${token}`,
              'Content-Type': 'application/json',
              'Accept': 'application/json',
            },
            timeout: 30000,
          }
        );

        const barsCount = response.data.bars ? response.data.bars.length : 0;
        if (response.data.success) {
          console.log(`✅ SUCCESS! Got ${barsCount} bars`);
          console.log('Response structure:', JSON.stringify(response.data, null, 2).slice(0, 500));
          break;
        } else {
          console.log('❌ Failed:', response.data);
        }
      } catch (error: any) {
        console.log('❌ Error:', error.response?.data || error.message);
      }
    }
  } catch (error: any) {
    console.error('Fatal error:', error.message);
  }
}

testHistoricalAPI();
