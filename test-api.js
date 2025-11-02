// Quick test script to verify API connections
require('dotenv').config();
const axios = require('axios');

async function testAPIs() {
  console.log('🧪 Testing API Connections...\n');

  // Test OpenAI
  console.log('1. Testing OpenAI API...');
  if (process.env.OPENAI_API_KEY) {
    console.log('   ✅ OpenAI API Key found');
    console.log('   Key preview:', process.env.OPENAI_API_KEY.substring(0, 10) + '...');
  } else {
    console.log('   ❌ OpenAI API Key missing!');
  }

  // Test Tradier
  console.log('\n2. Testing Tradier API...');
  if (process.env.TRADIER_API_KEY) {
    console.log('   ✅ Tradier API Key found');
    try {
      const response = await axios.get('https://api.tradier.com/v1/markets/quotes', {
        params: { symbols: 'SPY' },
        headers: {
          'Authorization': `Bearer ${process.env.TRADIER_API_KEY}`,
          'Accept': 'application/json',
        },
      });
      console.log('   ✅ Tradier API working! SPY price:', response.data.quotes.quote.last);
    } catch (error) {
      console.log('   ❌ Tradier API error:', error.response?.status, error.response?.data || error.message);
    }
  } else {
    console.log('   ❌ Tradier API Key missing!');
  }

  // Test Unusual Whales
  console.log('\n3. Testing Unusual Whales API...');
  if (process.env.UNUSUAL_WHALES_API_KEY) {
    console.log('   ✅ Unusual Whales API Key found');
    console.log('   Key preview:', process.env.UNUSUAL_WHALES_API_KEY.substring(0, 10) + '...');

    // Test different endpoints
    const today = new Date().toISOString().split('T')[0];

    try {
      console.log('   Testing option-trades/flow-alerts endpoint...');
      const optionsResponse = await axios.get('https://api.unusualwhales.com/api/option-trades/flow-alerts', {
        headers: {
          'Authorization': `Bearer ${process.env.UNUSUAL_WHALES_API_KEY}`,
          'Accept': 'application/json',
        },
      });
      console.log('   ✅ Options flow working! Count:', optionsResponse.data.data?.length || 0);
    } catch (error) {
      console.log('   ❌ Options flow error:', error.response?.status, error.response?.statusText);
      console.log('   Error details:', error.response?.data);
    }

    try {
      console.log('   Testing news/headlines endpoint...');
      const newsResponse = await axios.get('https://api.unusualwhales.com/api/news/headlines', {
        headers: {
          'Authorization': `Bearer ${process.env.UNUSUAL_WHALES_API_KEY}`,
          'Accept': 'application/json',
        },
      });
      console.log('   ✅ News working! Count:', newsResponse.data.data?.length || 0);
    } catch (error) {
      console.log('   ❌ News error:', error.response?.status, error.response?.statusText);
    }

    try {
      console.log('   Testing darkpool/recent endpoint...');
      const darkpoolResponse = await axios.get('https://api.unusualwhales.com/api/darkpool/recent', {
        headers: {
          'Authorization': `Bearer ${process.env.UNUSUAL_WHALES_API_KEY}`,
          'Accept': 'application/json',
        },
      });
      console.log('   ✅ Darkpool working! Count:', darkpoolResponse.data.data?.length || 0);
    } catch (error) {
      console.log('   ❌ Darkpool error:', error.response?.status, error.response?.statusText);
    }

    try {
      console.log('   Testing market/market-tide endpoint...');
      const tideResponse = await axios.get('https://api.unusualwhales.com/api/market/market-tide', {
        headers: {
          'Authorization': `Bearer ${process.env.UNUSUAL_WHALES_API_KEY}`,
          'Accept': 'application/json',
        },
      });
      console.log('   ✅ Market tide working!', tideResponse.data.data);
    } catch (error) {
      console.log('   ❌ Market tide error:', error.response?.status, error.response?.statusText);
    }
  } else {
    console.log('   ❌ Unusual Whales API Key missing!');
  }

  console.log('\n✅ Test complete!');
}

testAPIs().catch(console.error);
