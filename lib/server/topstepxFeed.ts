import { EventEmitter } from 'events';
import { loadSignalR } from '@/lib/server/signalr-cjs';
import { RSI, ADX } from 'technicalindicators';
import { authenticate, fetchTopstepXFuturesMetadata, fetchTopstepXFuturesBars } from '@/lib/topstepx';
import { ensureSignalRPolyfills } from '@/lib/server/signalrPolyfill';
import type { PriceBar } from '@/lib/ttmSqueeze';
import { calculateTtmSqueeze } from '@/lib/ttmSqueeze';

interface Candle {
  time: string;
  open: number;
  high: number;
  low: number;
  close: number;
  sma?: number | null;
  rsi?: number | null;
  adx?: number | null;
  bbUpper?: number | null;
  bbLower?: number | null;
  bbBasis?: number | null;
  kcUpper?: number | null;
  kcLower?: number | null;
  ttmSqueezeOn?: boolean | null;
  ttmSqueezeOff?: boolean | null;
  ttmMomentum?: number | null;
  ttmSentiment?: string | null;
}

interface FeedEvent {
  type: 'snapshot' | 'candle' | 'tick';
  price: number;
  timestamp: string;
  candle?: Candle & { sma?: number | null; rsi?: number | null; adx?: number | null };
}

const CONFIG = {
  symbol: process.env.TOPSTEPX_SECOND_SMA_SYMBOL || process.env.TOPSTEPX_SMA_SYMBOL || 'MESZ5',
  marketHubUrl: process.env.TOPSTEPX_MARKET_HUB_URL || 'https://rtc.topstepx.com/hubs/market',
  userHubUrl: process.env.TOPSTEPX_USER_HUB_URL || 'https://rtc.topstepx.com/hubs/user',
  smaPeriod: Number(process.env.TOPSTEPX_SECOND_SMA_PERIOD || '200'),
  rsiPeriod: Number(process.env.TOPSTEPX_SECOND_SMA_RSI_PERIOD || '24'),
  adxPeriod: Number(process.env.TOPSTEPX_SECOND_SMA_ADX_PERIOD || '24'),
  historySeconds: Number(process.env.TOPSTEPX_SECOND_FEED_HISTORY || '3600'),
  bbPeriod: Number(
    process.env.TOPSTEPX_FEED_BB_PERIOD ||
      process.env.TOPSTEPX_MR_LIVE_BB_PERIOD ||
      process.env.TOPSTEPX_SECOND_SMA_PERIOD ||
      '200',
  ),
  bbStdDev: Number(process.env.TOPSTEPX_FEED_BB_STDDEV || process.env.TOPSTEPX_MR_LIVE_BB_STDDEV || '3'),
  ttmLookback: Number(process.env.TOPSTEPX_FEED_TTM_LOOKBACK || '20'),
  ttmBbStdDev: Number(process.env.TOPSTEPX_FEED_TTM_BB_STDDEV || '2'),
  ttmAtrMultiplier: Number(process.env.TOPSTEPX_FEED_TTM_ATR_MULTIPLIER || '1.5'),
  ttmMomentumThreshold: Number(process.env.TOPSTEPX_FEED_TTM_MOMENTUM || '0.00001'),
};

type HubConnection = any;

class TopstepxFeed extends EventEmitter {
  async bootstrapHistory(contractId: string) {
    const end = new Date();
    const start = new Date(end.getTime() - CONFIG.historySeconds * 1000);
    const bars = await fetchTopstepXFuturesBars({
      contractId,
      startTime: start.toISOString(),
      endTime: end.toISOString(),
      unit: 1,
      unitNumber: 1,
      limit: CONFIG.historySeconds,
    });
    const ordered = [...bars].reverse();
    for (const bar of ordered) {
      this.finalizeBootstrapCandle({
        time: bar.timestamp,
        open: bar.open,
        high: bar.high,
        low: bar.low,
        close: bar.close,
      });
    }
  }

  private finalizeBootstrapCandle(bucket: Candle) {
    const enriched = this.enrichCandle(bucket);
    this.candles.push(enriched);
    if (this.candles.length > CONFIG.historySeconds) {
      this.candles = this.candles.slice(-CONFIG.historySeconds);
    }
  }

  private marketHub: HubConnection | null = null;
  private contractId: string | null = null;
  private candles: Candle[] = [];
  private currentBucket: Candle | null = null;
  private closes: number[] = [];
  private highs: number[] = [];
  private lows: number[] = [];
  private barHistory: PriceBar[] = [];
  private lastPrice = 0;
  private lastTimestamp = '';
  private starting = false;
  private maxHistory = Math.max(
    CONFIG.historySeconds,
    CONFIG.bbPeriod + 50,
    CONFIG.ttmLookback + 50,
  );

  private rsiIndicator = new RSI({ period: CONFIG.rsiPeriod, values: [] as number[] });
  private adxIndicator = new ADX({
    period: CONFIG.adxPeriod,
    high: [],
    low: [],
    close: [],
  });

  async ensureStarted() {
    if (this.marketHub || this.starting) {
      return;
    }
    this.starting = true;
    try {
      const { HubConnectionBuilder, HttpTransportType, LogLevel } = await loadSignalR();
      const metadata = await fetchTopstepXFuturesMetadata(
        process.env.TOPSTEPX_SECOND_SMA_CONTRACT_ID ||
          process.env.TOPSTEPX_CONTRACT_ID ||
          CONFIG.symbol,
      );
      if (!metadata) {
        throw new Error('Unable to resolve TopstepX contract metadata for feed');
      }
      this.contractId = metadata.id;
      const token = await authenticate();
      ensureSignalRPolyfills();
      this.marketHub = new HubConnectionBuilder()
        .withUrl(`${CONFIG.marketHubUrl}?access_token=${encodeURIComponent(token)}`, {
          skipNegotiation: true,
          transport: HttpTransportType.WebSockets,
          accessTokenFactory: () => token,
        })
        .configureLogging(LogLevel.Error)
        .withAutomaticReconnect()
        .build();

      this.marketHub.on('GatewayQuote', (_cid: string, quote: any) => {
        const price = Number(quote.lastPrice ?? quote.bestBid ?? quote.bestAsk ?? 0);
        if (!price) return;
        const timestamp = quote.timestamp || quote.lastUpdated || new Date().toISOString();
        this.processPrice(price, timestamp);
      });

      this.marketHub.on('GatewayTrade', (_cid: string, trade: any) => {
        const price = Number(trade.price ?? trade.lastPrice ?? 0);
        if (!price) return;
        const timestamp = trade.timestamp || trade.lastUpdated || new Date().toISOString();
        this.processPrice(price, timestamp);
      });

      const subscribe = async () => {
        if (!this.contractId) return;
        try {
          await this.marketHub?.invoke('SubscribeContractQuotes', this.contractId);
          await this.marketHub?.invoke('SubscribeContractTrades', this.contractId);
        } catch (err) {
          console.error('[topstepx-feed] failed to subscribe:', err);
        }
      };

      await this.marketHub.start();

      await this.bootstrapHistory(this.contractId);
      await subscribe();
      this.marketHub.onreconnected(() => {
        subscribe();
      });
    } finally {
      this.starting = false;
    }
  }

  private processPrice(price: number, timestamp: string) {
    this.lastPrice = price;
    this.lastTimestamp = timestamp;
    this.emitEvent({
      type: 'tick',
      price,
      timestamp,
    });

    const bucketTs = new Date(timestamp);
    const secondStart = new Date(bucketTs);
    secondStart.setMilliseconds(0);
    const bucketTimeIso = secondStart.toISOString();

    if (!this.currentBucket || this.currentBucket.time !== bucketTimeIso) {
      if (this.currentBucket) {
        this.finalizeBucket(this.currentBucket);
      }
      this.currentBucket = {
        time: bucketTimeIso,
        open: price,
        high: price,
        low: price,
        close: price,
      };
    } else {
      this.currentBucket.high = Math.max(this.currentBucket.high, price);
      this.currentBucket.low = Math.min(this.currentBucket.low, price);
      this.currentBucket.close = price;
    }
  }

  private finalizeBucket(bucket: Candle) {
    const enriched = this.enrichCandle(bucket);
    this.candles.push(enriched);
    if (this.candles.length > CONFIG.historySeconds) {
      this.candles = this.candles.slice(-CONFIG.historySeconds);
    }

    this.emitEvent({
      type: 'candle',
      price: bucket.close,
      timestamp: bucket.time,
      candle: enriched,
    });
  }

  getSnapshot() {
    return {
      candles: this.candles,
      price: this.lastPrice,
      timestamp: this.lastTimestamp,
    };
  }

  private emitEvent(event: FeedEvent) {
    this.emit('event', event);
  }

  private enrichCandle(bucket: Candle): Candle {
    this.registerHistory(bucket);

    const sma =
      this.closes.length >= CONFIG.smaPeriod
        ? this.closes.slice(-CONFIG.smaPeriod).reduce((sum, val) => sum + val, 0) /
          CONFIG.smaPeriod
        : null;
    const rsiValue = this.rsiIndicator.nextValue(bucket.close) ?? null;
    const adxValue = this.adxIndicator.nextValue({
      high: bucket.high,
      low: bucket.low,
      close: bucket.close,
    }) as { adx?: number } | number | undefined;
    const adx =
      typeof adxValue === 'number'
        ? adxValue
        : adxValue && typeof adxValue === 'object'
          ? adxValue.adx ?? null
          : null;
    const bollinger = this.computeBollinger();
    const ttm = this.computeTtm();

    return {
      ...bucket,
      sma,
      rsi: rsiValue,
      adx,
      bbUpper: bollinger?.upper ?? null,
      bbLower: bollinger?.lower ?? null,
      bbBasis: bollinger?.middle ?? null,
      kcUpper: ttm?.kcUpper ?? null,
      kcLower: ttm?.kcLower ?? null,
      ttmSqueezeOn: ttm ? ttm.squeezeOn : null,
      ttmSqueezeOff: ttm ? ttm.squeezeOff : null,
      ttmMomentum: ttm?.momentum ?? null,
      ttmSentiment: ttm?.sentiment ?? null,
    };
  }

  private registerHistory(bucket: Candle) {
    this.closes.push(bucket.close);
    this.highs.push(bucket.high);
    this.lows.push(bucket.low);
    this.barHistory.push({ high: bucket.high, low: bucket.low, close: bucket.close });

    if (this.closes.length > this.maxHistory) {
      this.closes = this.closes.slice(-this.maxHistory);
    }
    if (this.highs.length > this.maxHistory) {
      this.highs = this.highs.slice(-this.maxHistory);
    }
    if (this.lows.length > this.maxHistory) {
      this.lows = this.lows.slice(-this.maxHistory);
    }
    if (this.barHistory.length > this.maxHistory) {
      this.barHistory = this.barHistory.slice(-this.maxHistory);
    }
  }

  private computeBollinger() {
    if (this.closes.length < CONFIG.bbPeriod) {
      return null;
    }
    const slice = this.closes.slice(-CONFIG.bbPeriod);
    const mean = slice.reduce((sum, value) => sum + value, 0) / CONFIG.bbPeriod;
    const variance =
      slice.reduce((sum, value) => sum + Math.pow(value - mean, 2), 0) / CONFIG.bbPeriod;
    const stdDev = Math.sqrt(variance);
    return {
      upper: mean + stdDev * CONFIG.bbStdDev,
      lower: mean - stdDev * CONFIG.bbStdDev,
      middle: mean,
    };
  }

  private computeTtm() {
    if (this.barHistory.length < CONFIG.ttmLookback + 2) {
      return null;
    }
    return calculateTtmSqueeze(this.barHistory, {
      lookback: CONFIG.ttmLookback,
      bbStdDev: CONFIG.ttmBbStdDev,
      atrMultiplier: CONFIG.ttmAtrMultiplier,
      momentumThreshold: CONFIG.ttmMomentumThreshold,
    });
  }
}

let feedInstance: TopstepxFeed | null = null;
let feedPromise: Promise<TopstepxFeed> | null = null;

export async function getTopstepxFeed() {
  if (feedInstance) {
    return feedInstance;
  }
  if (!feedPromise) {
    feedPromise = (async () => {
      const feed = new TopstepxFeed();
      await feed.ensureStarted();
      feedInstance = feed;
      return feed;
    })();
  }
  return feedPromise;
}
