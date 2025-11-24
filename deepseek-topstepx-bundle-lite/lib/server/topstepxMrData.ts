import { existsSync, promises as fs } from 'fs';
import path from 'path';

const LOG_PATH =
  process.env.TOPSTEPX_MR_TRADE_LOG ||
  path.join(process.cwd(), 'logs/topstepx-mean-reversion-1s.jsonl');
const CONTRACT_MULTIPLIER = Number(process.env.TOPSTEPX_MR_CONTRACT_MULTIPLIER || '5');

export interface MrTrade {
  tradeId: string;
  side: 'long' | 'short';
  entryPrice: number;
  entryTime: string;
  qty: number;
  stopLoss?: number | null;
  target?: number | null;
  scaled?: boolean;
  scalePrice?: number | null;
  remainingQty?: number | null;
  scalePnL?: number | null;
  exitPrice?: number | null;
  exitTime?: string;
  exitReason?: string;
  pnl?: number | null;
  status: 'open' | 'closed';
}

interface TradeEvent {
  timestamp: string;
  type: 'entry' | 'scale' | 'exit' | string;
  tradeId?: string;
  side?: string;
  price?: number;
  marketPrice?: number;
  qty?: number;
  remainingQty?: number;
  pnl?: number;
  totalPnL?: number;
  exitPrice?: number;
  reason?: string;
  stopLoss?: number;
  target?: number;
  scaled?: boolean;
  [key: string]: any;
}

async function readLogLines(limit = 2000): Promise<TradeEvent[]> {
  if (!existsSync(LOG_PATH)) {
    return [];
  }

  try {
    const raw = await fs.readFile(LOG_PATH, 'utf-8');
    if (!raw.trim()) {
      return [];
    }
    return raw
      .trim()
      .split(/\r?\n/)
      .filter(Boolean)
      .slice(-limit)
      .map(line => {
        try {
          return JSON.parse(line);
        } catch {
          return null;
        }
      })
      .filter((event): event is TradeEvent => !!event);
  } catch (err) {
    console.error('[topstepx-mr-data] failed to read log file:', err);
    return [];
  }
}

function resolveTradeId(event: TradeEvent) {
  if (event.tradeId) return event.tradeId;
  if (event.entryOrderId) return String(event.entryOrderId);
  return `${event.timestamp}-${event.type}`;
}

export async function getMrTrades(limit = 2000) {
  const events = await readLogLines(limit);
  const trades = new Map<string, MrTrade>();

  for (const event of events) {
    const tradeId = resolveTradeId(event);
    if (!tradeId) {
      continue;
    }

    if (event.type === 'entry') {
      trades.set(tradeId, {
        tradeId,
        side: event.side === 'SHORT' ? 'short' : 'long',
        entryPrice: Number(event.price ?? event.marketPrice ?? 0),
        entryTime: event.timestamp,
        qty: Number(event.qty ?? 0),
        remainingQty: Number(event.qty ?? 0),
        stopLoss: typeof event.stopLoss === 'number' ? event.stopLoss : null,
        target: typeof event.target === 'number' ? event.target : null,
        status: 'open',
      });
      continue;
    }

    const existing = trades.get(tradeId);
    if (!existing) {
      continue;
    }

    if (event.type === 'scale') {
      existing.scaled = true;
      existing.scalePrice = typeof event.price === 'number' ? event.price : existing.scalePrice;
      existing.remainingQty =
        typeof event.remainingQty === 'number' ? event.remainingQty : existing.remainingQty;
      if (typeof event.pnl === 'number') {
        existing.scalePnL = event.pnl;
      }
      if (typeof event.newStop === 'number') {
        existing.stopLoss = event.newStop;
      }
      if (typeof event.newTarget === 'number') {
        existing.target = event.newTarget;
      }
      continue;
    }

    if (event.type === 'exit') {
      existing.exitPrice = typeof event.exitPrice === 'number' ? event.exitPrice : event.price ?? existing.exitPrice ?? null;
      existing.exitTime = event.timestamp;
      existing.exitReason = event.reason ?? existing.exitReason;
      const pnlValue =
        typeof event.totalPnL === 'number'
          ? event.totalPnL
          : typeof event.exitPnL === 'number'
            ? event.exitPnL
            : null;
      existing.pnl =
        pnlValue != null
          ? pnlValue
          : existing.exitPrice != null
            ? (existing.exitPrice - existing.entryPrice) *
              (existing.side === 'long' ? 1 : -1) *
              CONTRACT_MULTIPLIER *
              Math.max(existing.qty || 1, 1)
            : null;
      existing.status = 'closed';
      trades.set(tradeId, existing);
    }
  }

  const ordered = Array.from(trades.values());
  const openTrade =
    ordered.find(trade => trade.status === 'open') ??
    null;
  const closedTrades = ordered
    .filter(trade => trade.status === 'closed')
    .sort((a, b) => (b.exitTime || '').localeCompare(a.exitTime || ''));

  return {
    openTrade,
    closedTrades,
  };
}
