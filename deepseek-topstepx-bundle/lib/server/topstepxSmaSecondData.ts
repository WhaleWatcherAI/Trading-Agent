import { existsSync, promises as fs } from 'fs';
import path from 'path';

const LOG_PATH =
  process.env.TOPSTEPX_SECOND_SMA_TRADE_LOG || path.join(process.cwd(), 'logs/topstepx-sma-second-live.jsonl');
const CONTRACT_MULTIPLIER = Number(process.env.TOPSTEPX_SECOND_CONTRACT_MULTIPLIER || '50');

export interface StrategyTrade {
  tradeId: string;
  side: 'long' | 'short';
  entryPrice: number;
  entryTime: string;
  contracts: number;
  entryRSI?: number;
  entryADX?: number | null;
  exitPrice?: number;
  exitTime?: string;
  exitReason?: string;
  pnl?: number;
  status: 'open' | 'closed';
}

interface TradeEvent {
  timestamp: string;
  type: string;
  tradeId?: string;
  direction?: 'long' | 'short';
  price?: number;
  contracts?: number;
  rsi?: number;
  adx?: number | null;
  reason?: string;
  exitPrice?: number;
  realized?: number;
}

async function readLogLines(limit = 1000): Promise<TradeEvent[]> {
  if (!existsSync(LOG_PATH)) {
    return [];
  }

  try {
    const raw = await fs.readFile(LOG_PATH, 'utf-8');
    if (!raw.trim()) {
      return [];
    }
    const lines = raw
      .trim()
      .split(/\r?\n/)
      .filter(Boolean)
      .slice(-limit);

    const events: TradeEvent[] = [];
    for (const line of lines) {
      try {
        const parsed = JSON.parse(line);
        events.push(parsed);
      } catch (err) {
        // ignore malformed line but continue
        console.warn('[topstepx-sma-second] Failed to parse log line:', err);
      }
    }
    return events;
  } catch (err) {
    console.error('[topstepx-sma-second] Unable to read log file:', err);
    return [];
  }
}

export async function getStrategyTrades(limit = 1000) {
  const events = await readLogLines(limit);
  const trades = new Map<string, StrategyTrade>();

  for (const event of events) {
    if (!event.tradeId) continue;

    if (event.type === 'entry') {
      trades.set(event.tradeId, {
        tradeId: event.tradeId,
        side: event.direction === 'short' ? 'short' : 'long',
        entryPrice: event.price ?? 0,
        entryTime: event.timestamp,
        contracts: event.contracts ?? 0,
        entryRSI: event.rsi,
        entryADX: event.adx,
        status: 'open',
      });
    } else if (event.type === 'exit') {
      const existing = trades.get(event.tradeId);
      if (existing) {
        const exitPrice = event.exitPrice ?? event.price ?? event.realized ?? 0;
        const pnl =
          (exitPrice - existing.entryPrice) *
          (existing.side === 'long' ? 1 : -1) *
          CONTRACT_MULTIPLIER *
          Math.max(existing.contracts, 1);

        trades.set(event.tradeId, {
          ...existing,
          exitPrice,
          exitTime: event.timestamp,
          exitReason: event.reason,
          pnl,
          status: 'closed',
        });
      }
    }
  }

  const allTrades = Array.from(trades.values());
  const openTrade = allTrades.find(trade => trade.status === 'open') ?? null;
  const closedTrades = allTrades
    .filter(trade => trade.status === 'closed')
    .sort((a, b) => (b.exitTime || '').localeCompare(a.exitTime || ''));

  return {
    openTrade,
    closedTrades,
  };
}
