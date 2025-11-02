import axios from 'axios';

const COINBASE_BASE_URL = 'https://api.coinbase.com/v2';

export interface CryptoPrice {
  symbol: string;
  price: number;
  change24h: number;
  volume24h: number;
  timestamp: Date;
}

export async function getCryptoPrice(symbol: string = 'BTC'): Promise<CryptoPrice> {
  try {
    const [priceResponse, statsResponse] = await Promise.all([
      axios.get(`${COINBASE_BASE_URL}/prices/${symbol}-USD/spot`),
      axios.get(`${COINBASE_BASE_URL}/prices/${symbol}-USD/stats`),
    ]);

    const price = parseFloat(priceResponse.data.data.amount);
    const stats = statsResponse.data.data;

    return {
      symbol,
      price,
      change24h: stats.change_24h ? parseFloat(stats.change_24h) : 0,
      volume24h: stats.volume_24h ? parseFloat(stats.volume_24h) : 0,
      timestamp: new Date(),
    };
  } catch (error) {
    console.error(`Error fetching crypto price for ${symbol}:`, error);
    throw error;
  }
}

export async function getMultipleCryptoPrices(symbols: string[]): Promise<CryptoPrice[]> {
  try {
    const promises = symbols.map(symbol => getCryptoPrice(symbol));
    return await Promise.all(promises);
  } catch (error) {
    console.error('Error fetching multiple crypto prices:', error);
    throw error;
  }
}
