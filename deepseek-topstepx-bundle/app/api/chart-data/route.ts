import { NextRequest, NextResponse } from 'next/server';
import axios from 'axios';

const TRADIER_API_KEY = process.env.TRADIER_API_KEY || '';
const TRADIER_BASE_URL = process.env.TRADIER_BASE_URL || 'https://sandbox.tradier.com/v1';

const sanitizeTicker = (value: string) =>
  value
    .toUpperCase()
    .replace(/[^A-Z0-9.\-]/g, '')
    .slice(0, 21);

export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams;
  const requestedSymbol = sanitizeTicker(searchParams.get('symbol') || 'AAPL');
  if (!requestedSymbol) {
    return NextResponse.json({ error: 'Invalid symbol provided' }, { status: 400 });
  }
  const rawInterval = searchParams.get('interval');
  // Guard against "undefined" string and null/empty values
  const interval = (rawInterval && rawInterval !== 'undefined' && rawInterval !== 'null') ? rawInterval : 'daily';
  let days = parseInt(searchParams.get('days') || '365');

  // Limit days based on interval due to Tradier API constraints
  const maxDays: Record<string, number> = {
    '1min': 5,      // Conservative limit for 1min data
    '5min': 15,     // Conservative limit for 5min data
    '15min': 30,    // Conservative limit for 15min data
    '30min': 30,    // Conservative limit for 30min data
    '1hour': 60,    // Conservative limit for 1hour data
    '4hour': 90,    // Conservative limit for 4hour data
  };

  if (maxDays[interval] && days > maxDays[interval]) {
    days = maxDays[interval];
  }

  try {
    // Calculate start and end dates
    const endDate = new Date();
    const startDate = new Date();
    startDate.setDate(startDate.getDate() - days);

    // Map frontend intervals to Tradier API intervals
    // Tradier uses /markets/timesales for intraday and /markets/history for daily+
    const intervalMap: Record<string, string> = {
      '1min': '1min',
      '5min': '5min',
      '15min': '15min',
      '30min': '30min',  // Not officially supported, may not work
      '1hour': '1hour',  // Not officially supported, may not work
      '4hour': '4hour',  // Not officially supported, may not work
      'daily': 'daily',
      'weekly': 'weekly',
      'monthly': 'monthly',
    };

    const tradierInterval = intervalMap[interval] || 'daily';
    const isIntraday = ['1min', '5min', '15min', '30min', '1hour', '4hour'].includes(interval);
    let resolvedInterval = interval;

    // Use different endpoints for intraday vs daily+ data
    const endpoint = isIntraday ? '/markets/timesales' : '/markets/history';
    console.log(`Fetching ${requestedSymbol} with interval: ${tradierInterval} (endpoint: ${endpoint})`);

    let bars: any[] = [];
    let usingIntraday = isIntraday;

    if (isIntraday) {
      try {
        const response = await axios.get(`${TRADIER_BASE_URL}/markets/timesales`, {
          headers: {
            Authorization: `Bearer ${TRADIER_API_KEY}`,
            Accept: 'application/json',
          },
          params: {
            symbol: requestedSymbol,
            interval: tradierInterval,
            start: startDate.toISOString().split('T')[0],
            end: endDate.toISOString().split('T')[0],
          },
        });

        const series = response.data?.series;
        if (series?.data) {
          bars = Array.isArray(series.data) ? series.data : [series.data];
          console.log(`Received ${bars.length} intraday bars for ${requestedSymbol}`);
        } else {
          console.warn(`No timesales data payload for ${requestedSymbol}; falling back to historical data.`);
          bars = [];
          usingIntraday = false;
        }
      } catch (intradayError: any) {
        usingIntraday = false;
        const status = intradayError?.response?.status;
        const reason = intradayError?.response?.data || intradayError?.message;
        console.warn(`Intraday request failed for ${requestedSymbol} (status ${status ?? 'n/a'}). Falling back to historical data.`, reason);
      }
    }

    if (!bars.length) {
      // Fallback to history endpoint (daily/weekly/monthly)
      const historyInterval = ['weekly', 'monthly'].includes(interval) ? tradierInterval : 'daily';
      resolvedInterval = historyInterval;
      const historyResponse = await axios.get(`${TRADIER_BASE_URL}/markets/history`, {
        headers: {
          Authorization: `Bearer ${TRADIER_API_KEY}`,
          Accept: 'application/json',
        },
        params: {
          symbol: requestedSymbol,
          interval: historyInterval,
          start: startDate.toISOString().split('T')[0],
          end: endDate.toISOString().split('T')[0],
        },
      });

      const history = historyResponse.data?.history;
      if (!history || !history.day) {
        console.error(`No historical data for ${requestedSymbol}`);
        console.error('Response data:', JSON.stringify(historyResponse.data, null, 2));
        return NextResponse.json({ error: 'No historical data available' }, { status: 404 });
      }
      bars = Array.isArray(history.day) ? history.day : [history.day];
      console.log(`Received ${bars.length} historical bars for ${requestedSymbol} (resolved interval ${resolvedInterval})`);
    }

    const isTradierIntraday = usingIntraday;

    // Convert to TradingView Lightweight Charts format
    const candlestickData = bars.map((bar: any) => {
      let time;
      if (isTradierIntraday) {
        // Timesales data has "timestamp" field (Unix timestamp)
        time = bar.timestamp;
      } else if (bar.date && bar.time) {
        // Daily data with time: "2025-01-15 09:30:00"
        const timestamp = Math.floor(new Date(bar.date + ' ' + bar.time).getTime() / 1000);
        time = timestamp;
      } else {
        // Daily/weekly/monthly only has date: "2025-01-15"
        time = bar.date;
      }

      return {
        time,
        open: parseFloat(bar.open),
        high: parseFloat(bar.high),
        low: parseFloat(bar.low),
        close: parseFloat(bar.close),
      };
    });

    const volumeData = bars.map((bar: any) => {
      let time;
      if (isTradierIntraday) {
        // Timesales data has "timestamp" field (Unix timestamp)
        time = bar.timestamp;
      } else if (bar.date && bar.time) {
        // Daily data with time
        const timestamp = Math.floor(new Date(bar.date + ' ' + bar.time).getTime() / 1000);
        time = timestamp;
      } else {
        // Daily/weekly/monthly only has date
        time = bar.date;
      }

      const rawVolume = typeof bar.volume === 'number' ? bar.volume : parseInt(bar.volume ?? '0', 10);
      const volumeValue = Number.isFinite(rawVolume) ? rawVolume : 0;

      return {
        time,
        value: volumeValue,
        color: parseFloat(bar.close) >= parseFloat(bar.open) ? 'rgba(0, 150, 136, 0.5)' : 'rgba(255, 82, 82, 0.5)',
      };
    });

    return NextResponse.json({
      symbol: requestedSymbol,
      interval: resolvedInterval,
      requestedInterval: interval,
      candlesticks: candlestickData,
      volume: volumeData,
      count: candlestickData.length,
    });
  } catch (error: any) {
    console.error('Error fetching chart data:', error);
    return NextResponse.json(
      { error: 'Failed to fetch chart data', details: error.message },
      { status: 500 }
    );
  }
}
