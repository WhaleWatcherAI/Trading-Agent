/**
 * Enhanced Features for Self-Learning Trading Agent
 * Implements POC cross tracking, market statistics, and performance metrics
 * Based on SELF_LEARNING_SYSTEM_DOCS.md specification
 */

export interface POCCrossStats {
  count_last_5min: number;
  count_last_15min: number;
  count_last_30min: number; // Actually total crosses since trading day start (6 PM ET)
  time_since_last_cross_sec: number;
  current_side: 'above_poc' | 'below_poc' | 'at_poc';
}

export interface MarketStatistics {
  session_range_ticks: number;
  session_range_percentile: number; // 0-1 (e.g., 0.75 = 75th percentile)
  distance_to_poc_ticks: number;
  time_above_value_sec: number;
  time_below_value_sec: number;
  time_in_value_sec: number;
  cvd_slope_5min: number;
  cvd_slope_15min: number;
}

export interface PerformanceMetrics {
  win_rate: number; // 0-1
  avg_pnl: number;
  trade_count: number;
  profit_factor: number;
  avg_win: number;
  avg_loss: number;
}

export interface HistoricalNote {
  timestamp: string;
  note: string;
  context: string; // e.g., "POC chop", "trend failure", etc.
}

/**
 * Get trading day start time (6 PM ET previous day)
 * Futures trading day starts at 6 PM ET
 */
function getTradingDayStart(): Date {
  const now = new Date();

  // Convert to ET timezone (UTC-5 or UTC-4 depending on DST)
  const etOffset = -5 * 60; // ET is UTC-5 (adjust for DST if needed)
  const etNow = new Date(now.getTime() + etOffset * 60 * 1000);

  // Get current hour in ET
  const etHour = etNow.getUTCHours();

  // If before 5pm ET (17:00), trading day started 6pm ET yesterday
  // If after 5pm ET, trading day starts at 6pm ET today
  let tradingDayStart: Date;
  if (etHour < 17) {
    // Before 5pm ET - use 6pm yesterday
    tradingDayStart = new Date(etNow);
    tradingDayStart.setUTCDate(tradingDayStart.getUTCDate() - 1);
    tradingDayStart.setUTCHours(18, 0, 0, 0); // 6pm ET = 18:00 ET
  } else {
    // After 5pm ET - use 6pm today
    tradingDayStart = new Date(etNow);
    tradingDayStart.setUTCHours(18, 0, 0, 0);
  }

  // Convert back to UTC
  return new Date(tradingDayStart.getTime() - etOffset * 60 * 1000);
}

/**
 * POC Cross Tracker
 * Tracks how many times price crosses the POC since trading day start (6 PM ET)
 */
export class POCCrossTracker {
  private crosses: Array<{ timestamp: number; direction: 'up' | 'down' }> = [];
  private currentSide: 'above_poc' | 'below_poc' | 'at_poc' = 'at_poc';
  private lastPrice = 0;
  private lastPOC = 0;

  update(currentPrice: number, poc: number): POCCrossStats {
    const now = Date.now();

    // Determine current side
    if (currentPrice > poc + 0.25) {
      this.currentSide = 'above_poc';
    } else if (currentPrice < poc - 0.25) {
      this.currentSide = 'below_poc';
    } else {
      this.currentSide = 'at_poc';
    }

    // Detect cross
    if (this.lastPrice && this.lastPOC) {
      const wasBelowPOC = this.lastPrice < this.lastPOC;
      const isAbovePOC = currentPrice > poc;

      if (wasBelowPOC && isAbovePOC) {
        this.crosses.push({ timestamp: now, direction: 'up' });
      } else if (!wasBelowPOC && !isAbovePOC) {
        this.crosses.push({ timestamp: now, direction: 'down' });
      }
    }

    this.lastPrice = currentPrice;
    this.lastPOC = poc;

    // Clean old crosses (keep since trading day start at 6 PM ET)
    const tradingDayStart = getTradingDayStart();
    const tradingDayStartMs = tradingDayStart.getTime();
    this.crosses = this.crosses.filter(c => c.timestamp > tradingDayStartMs);

    // Count crosses for entire trading day (not just 5/15/30 min windows)
    const fiveMinutesAgo = now - 5 * 60 * 1000;
    const fifteenMinutesAgo = now - 15 * 60 * 1000;

    const count_last_5min = this.crosses.filter(c => c.timestamp > fiveMinutesAgo).length;
    const count_last_15min = this.crosses.filter(c => c.timestamp > fifteenMinutesAgo).length;
    const count_last_30min = this.crosses.length; // All crosses since trading day start

    const lastCrossTime = this.crosses.length > 0
      ? this.crosses[this.crosses.length - 1].timestamp
      : tradingDayStartMs;
    const time_since_last_cross_sec = (now - lastCrossTime) / 1000;

    return {
      count_last_5min,
      count_last_15min,
      count_last_30min,
      time_since_last_cross_sec,
      current_side: this.currentSide,
    };
  }
}

/**
 * Market Statistics Calculator
 * Calculates raw market statistics for LLM regime inference
 */
export class MarketStatsCalculator {
  private sessionHigh = 0;
  private sessionLow = Infinity;
  private sessionStart = Date.now();
  private timeInValueStart: number | null = null;
  private timeAboveValueStart: number | null = null;
  private timeBelowValueStart: number | null = null;
  private totalTimeInValue = 0;
  private totalTimeAboveValue = 0;
  private totalTimeBelowValue = 0;
  private cvdHistory: Array<{ timestamp: number; cvd: number }> = [];
  private historicalRanges: number[] = []; // Track historical session ranges for percentiles

  updateSession(high: number, low: number) {
    this.sessionHigh = Math.max(this.sessionHigh, high);
    this.sessionLow = Math.min(this.sessionLow, low);
  }

  updateTimeInValue(currentPrice: number, vah: number, val: number) {
    const now = Date.now();
    const inValue = currentPrice >= val && currentPrice <= vah;
    const aboveValue = currentPrice > vah;
    const belowValue = currentPrice < val;

    // Track time in value area
    if (inValue) {
      if (!this.timeInValueStart) {
        this.timeInValueStart = now;
      }
      if (this.timeAboveValueStart) {
        this.totalTimeAboveValue += (now - this.timeAboveValueStart) / 1000;
        this.timeAboveValueStart = null;
      }
      if (this.timeBelowValueStart) {
        this.totalTimeBelowValue += (now - this.timeBelowValueStart) / 1000;
        this.timeBelowValueStart = null;
      }
    } else if (aboveValue) {
      if (!this.timeAboveValueStart) {
        this.timeAboveValueStart = now;
      }
      if (this.timeInValueStart) {
        this.totalTimeInValue += (now - this.timeInValueStart) / 1000;
        this.timeInValueStart = null;
      }
      if (this.timeBelowValueStart) {
        this.totalTimeBelowValue += (now - this.timeBelowValueStart) / 1000;
        this.timeBelowValueStart = null;
      }
    } else if (belowValue) {
      if (!this.timeBelowValueStart) {
        this.timeBelowValueStart = now;
      }
      if (this.timeInValueStart) {
        this.totalTimeInValue += (now - this.timeInValueStart) / 1000;
        this.timeInValueStart = null;
      }
      if (this.timeAboveValueStart) {
        this.totalTimeAboveValue += (now - this.timeAboveValueStart) / 1000;
        this.timeAboveValueStart = null;
      }
    }
  }

  updateCVD(cvd: number) {
    const now = Date.now();
    this.cvdHistory.push({ timestamp: now, cvd });

    // Keep last 15 minutes of CVD data
    const fifteenMinutesAgo = now - 15 * 60 * 1000;
    this.cvdHistory = this.cvdHistory.filter(h => h.timestamp > fifteenMinutesAgo);
  }

  calculateCVDSlope(windowMinutes: number): number {
    const now = Date.now();
    const windowStart = now - windowMinutes * 60 * 1000;
    const relevantData = this.cvdHistory.filter(h => h.timestamp > windowStart);

    if (relevantData.length < 2) return 0;

    // Simple linear regression slope
    const n = relevantData.length;
    const sumX = relevantData.reduce((sum, _, i) => sum + i, 0);
    const sumY = relevantData.reduce((sum, d) => sum + d.cvd, 0);
    const sumXY = relevantData.reduce((sum, d, i) => sum + i * d.cvd, 0);
    const sumX2 = relevantData.reduce((sum, _, i) => sum + i * i, 0);

    const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
    return isNaN(slope) ? 0 : slope;
  }

  addHistoricalRange(range: number) {
    this.historicalRanges.push(range);
    // Keep last 100 sessions for percentile calculation
    if (this.historicalRanges.length > 100) {
      this.historicalRanges.shift();
    }
  }

  calculatePercentile(value: number): number {
    if (this.historicalRanges.length === 0) return 0.5;

    const sorted = [...this.historicalRanges].sort((a, b) => a - b);
    const count = sorted.filter(v => v <= value).length;
    return count / sorted.length;
  }

  calculate(currentPrice: number, poc: number, vah: number, val: number, tickSize: number = 0.25): MarketStatistics {
    const sessionRange = this.sessionHigh - this.sessionLow;
    const session_range_ticks = sessionRange / tickSize;
    const session_range_percentile = this.calculatePercentile(session_range_ticks);
    const distance_to_poc_ticks = Math.abs(currentPrice - poc) / tickSize;

    // Finalize current time tracking
    const now = Date.now();
    let time_in_value_sec = this.totalTimeInValue;
    let time_above_value_sec = this.totalTimeAboveValue;
    let time_below_value_sec = this.totalTimeBelowValue;

    if (this.timeInValueStart) {
      time_in_value_sec += (now - this.timeInValueStart) / 1000;
    }
    if (this.timeAboveValueStart) {
      time_above_value_sec += (now - this.timeAboveValueStart) / 1000;
    }
    if (this.timeBelowValueStart) {
      time_below_value_sec += (now - this.timeBelowValueStart) / 1000;
    }

    const cvd_slope_5min = this.calculateCVDSlope(5);
    const cvd_slope_15min = this.calculateCVDSlope(15);

    return {
      session_range_ticks,
      session_range_percentile,
      distance_to_poc_ticks,
      time_above_value_sec,
      time_below_value_sec,
      time_in_value_sec,
      cvd_slope_5min,
      cvd_slope_15min,
    };
  }

  reset() {
    this.sessionHigh = 0;
    this.sessionLow = Infinity;
    this.sessionStart = Date.now();
    this.timeInValueStart = null;
    this.timeAboveValueStart = null;
    this.timeBelowValueStart = null;
    this.totalTimeInValue = 0;
    this.totalTimeAboveValue = 0;
    this.totalTimeBelowValue = 0;

    // Add current range to historical before reset
    const currentRange = (this.sessionHigh - this.sessionLow) / 0.25; // ticks
    if (currentRange > 0) {
      this.addHistoricalRange(currentRange);
    }
  }
}

/**
 * Performance Tracker
 * Tracks trading performance for self-learning feedback
 */
export class PerformanceTracker {
  private trades: Array<{
    timestamp: number;
    pnl: number;
    win: boolean;
  }> = [];

  recordTrade(pnl: number) {
    this.trades.push({
      timestamp: Date.now(),
      pnl,
      win: pnl > 0,
    });

    // Keep last 100 trades
    if (this.trades.length > 100) {
      this.trades.shift();
    }
  }

  getMetrics(): PerformanceMetrics {
    if (this.trades.length === 0) {
      return {
        win_rate: 0,
        avg_pnl: 0,
        trade_count: 0,
        profit_factor: 0,
        avg_win: 0,
        avg_loss: 0,
      };
    }

    const wins = this.trades.filter(t => t.win);
    const losses = this.trades.filter(t => !t.win);

    const totalWins = wins.reduce((sum, t) => sum + t.pnl, 0);
    const totalLosses = Math.abs(losses.reduce((sum, t) => sum + t.pnl, 0));

    const avg_win = wins.length > 0 ? totalWins / wins.length : 0;
    const avg_loss = losses.length > 0 ? totalLosses / losses.length : 0;
    const profit_factor = totalLosses > 0 ? totalWins / totalLosses : 0;

    return {
      win_rate: wins.length / this.trades.length,
      avg_pnl: this.trades.reduce((sum, t) => sum + t.pnl, 0) / this.trades.length,
      trade_count: this.trades.length,
      profit_factor,
      avg_win,
      avg_loss,
    };
  }
}

/**
 * Historical Notes Manager
 * Stores notes for future self-learning
 */
export class HistoricalNotesManager {
  private notes: HistoricalNote[] = [];

  addNote(note: string, context: string) {
    this.notes.push({
      timestamp: new Date().toISOString(),
      note,
      context,
    });

    // Keep last 50 notes
    if (this.notes.length > 50) {
      this.notes.shift();
    }
  }

  getRecentNotes(count: number = 10): HistoricalNote[] {
    return this.notes.slice(-count);
  }

  getNotesForContext(context: string): HistoricalNote[] {
    return this.notes.filter(n => n.context === context);
  }
}
