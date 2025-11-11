import { describe, expect, it } from 'vitest';

const sampleOptionTrade = {
  positionSide: 'long' as const,
  entryFillPrice: 2,
  multiplier: 100,
};

const sampleShortStockTrade = {
  positionSide: 'short' as const,
  entryFillPrice: 100,
  multiplier: 1,
};

describe('calculatePnL', () => {
  it('treats long puts (short direction) as positive when premium expands', async () => {
    process.env.MR5_DISABLE_MAIN = 'true';
    const { calculatePnL } = await import('../run-live-mean-reversion-5min');
    const pnl = calculatePnL(sampleOptionTrade as any, 3, 1);
    expect(pnl).toBe(100);
  });

  it('treats short stock exits correctly when price falls', async () => {
    process.env.MR5_DISABLE_MAIN = 'true';
    const { calculatePnL } = await import('../run-live-mean-reversion-5min');
    const pnl = calculatePnL(sampleShortStockTrade as any, 95, 50);
    expect(pnl).toBe(250);
  });

  it('returns negative PnL when a short position loses money', async () => {
    process.env.MR5_DISABLE_MAIN = 'true';
    const { calculatePnL } = await import('../run-live-mean-reversion-5min');
    const pnl = calculatePnL(sampleShortStockTrade as any, 105, 50);
    expect(pnl).toBe(-250);
  });
});
