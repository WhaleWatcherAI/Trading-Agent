/**
 * Self-Learning Trading Database
 * Stores decision predictions and tracks actual outcomes for learning
 */

import { writeFileSync, readFileSync, existsSync, appendFileSync } from 'fs';
import path from 'path';
import { MarketState, SetupModel } from './fabioPlaybook';

/**
 * Structure for storing a trading decision prediction
 */
export interface TradingDecision {
  id: string;
  timestamp: string;
  symbol: string;

  // Fabio's 3-layer decision framework
  marketState: MarketState;
  location: string;
  setupModel: SetupModel | null;

  // Prediction
  decision: 'BUY' | 'SELL' | 'HOLD';
  confidence: number; // 0-100

  // Entry setup
  entryPrice: number;
  stopLoss: number;
  target: number;
  riskPercent: number;

  // Source
  source: 'rule_based' | 'openai' | 'hybrid'; // Which system made the decision
  reasoning: string;

  // Market context at prediction time
  cvd: number;
  cvdTrend: 'up' | 'down' | 'neutral';
  currentPrice: number;
  buyAbsorption: number;
  sellAbsorption: number;

  // Execution status
  status: 'pending' | 'filled' | 'rejected';
  filledAt?: number;
  filledPrice?: number;
  filledTime?: string;
}

/**
 * Structure for tracking actual trade outcome
 */
export interface TradeOutcome {
  decisionId: string;
  symbol: string;
  timestamp: string;

  // Execution
  executedPrice: number;
  executedTime: string;

  // Closure
  closedPrice: number;
  closedTime: string;

  // Results
  profitLoss: number;
  profitLossPercent: number;
  riskRewardActual: number;

  // Analysis
  wasCorrect: boolean; // Prediction matched actual result
  reason: string; // Why did it win/lose

  // For learning
  feedbackNotes?: string;
}

/**
 * Complete trade record combining decision + outcome
 */
export interface CompleteTradeRecord extends TradingDecision {
  outcome?: TradeOutcome;
}

/**
 * Database manager
 */
class TradingDatabase {
  private decisionsFile: string;
  private outcomesFile: string;
  private decisions: Map<string, TradingDecision> = new Map();
  private outcomes: Map<string, TradeOutcome> = new Map();

  constructor(dbDir: string = './trading-db') {
    this.decisionsFile = path.join(dbDir, 'decisions.jsonl');
    this.outcomesFile = path.join(dbDir, 'outcomes.jsonl');

    // Ensure directory exists
    const fs = require('fs');
    if (!fs.existsSync(dbDir)) {
      fs.mkdirSync(dbDir, { recursive: true });
    }

    this.loadFromDisk();
  }

  /**
   * Generate unique decision ID
   */
  private generateId(): string {
    return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }

  /**
   * Record a new trading decision
   */
  recordDecision(decision: Omit<TradingDecision, 'id' | 'timestamp' | 'status'>): TradingDecision {
    const newDecision: TradingDecision = {
      ...decision,
      id: this.generateId(),
      timestamp: new Date().toISOString(),
      status: 'pending',
    };

    this.decisions.set(newDecision.id, newDecision);
    this.appendToFile(this.decisionsFile, newDecision);

    return newDecision;
  }

  /**
   * Mark decision as filled
   */
  markDecisionFilled(decisionId: string, filledPrice: number): TradingDecision | null {
    const decision = this.decisions.get(decisionId);
    if (!decision) return null;

    decision.status = 'filled';
    decision.filledPrice = filledPrice;
    decision.filledTime = new Date().toISOString();

    this.decisions.set(decisionId, decision);
    return decision;
  }

  /**
   * Record actual trade outcome
   */
  recordOutcome(decisionId: string, outcome: Omit<TradeOutcome, 'decisionId' | 'timestamp'>): TradeOutcome {
    const decision = this.decisions.get(decisionId);
    if (!decision) {
      throw new Error(`Decision ${decisionId} not found`);
    }

    const newOutcome: TradeOutcome = {
      ...outcome,
      decisionId,
      timestamp: new Date().toISOString(),
    };

    this.outcomes.set(decisionId, newOutcome);
    this.appendToFile(this.outcomesFile, newOutcome);

    return newOutcome;
  }

  /**
   * Get decision by ID
   */
  getDecision(id: string): TradingDecision | undefined {
    return this.decisions.get(id);
  }

  /**
   * Get outcome by decision ID
   */
  getOutcome(decisionId: string): TradeOutcome | undefined {
    return this.outcomes.get(decisionId);
  }

  /**
   * Get complete trade record (decision + outcome)
   */
  getTradeRecord(id: string): CompleteTradeRecord | undefined {
    const decision = this.decisions.get(id);
    if (!decision) return undefined;

    const outcome = this.outcomes.get(id);
    return {
      ...decision,
      outcome,
    };
  }

  /**
   * Get all decisions for a symbol
   */
  getDecisionsBySymbol(symbol: string): TradingDecision[] {
    return Array.from(this.decisions.values())
      .filter(d => d.symbol === symbol)
      .sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());
  }

  /**
   * Get all outcomes for a symbol
   */
  getOutcomesBySymbol(symbol: string): TradeOutcome[] {
    return Array.from(this.outcomes.values())
      .filter(o => {
        const decision = this.decisions.get(o.decisionId);
        return decision?.symbol === symbol;
      })
      .sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());
  }

  /**
   * Calculate statistics for learning
   */
  calculateStats(symbol?: string): {
    totalDecisions: number;
    totalFilled: number;
    totalOutcomes: number;
    winners: number;
    losers: number;
    winRate: number;
    avgWin: number;
    avgLoss: number;
    profitFactor: number;
    bySource: { [key: string]: any };
    bySetupModel: { [key: string]: any };
  } {
    const decisions = symbol
      ? this.getDecisionsBySymbol(symbol)
      : Array.from(this.decisions.values());

    const outcomes = symbol
      ? this.getOutcomesBySymbol(symbol)
      : Array.from(this.outcomes.values());

    const winners = outcomes.filter(o => o.profitLoss > 0);
    const losers = outcomes.filter(o => o.profitLoss < 0);

    const totalWinAmount = winners.reduce((sum, o) => sum + o.profitLoss, 0);
    const totalLossAmount = Math.abs(losers.reduce((sum, o) => sum + o.profitLoss, 0));

    // Group by source
    const bySource: { [key: string]: any } = {};
    decisions.forEach(d => {
      if (!bySource[d.source]) {
        bySource[d.source] = { count: 0, wins: 0, losses: 0 };
      }
      bySource[d.source].count++;

      const outcome = this.outcomes.get(d.id);
      if (outcome) {
        if (outcome.profitLoss > 0) bySource[d.source].wins++;
        if (outcome.profitLoss < 0) bySource[d.source].losses++;
      }
    });

    // Group by setup model
    const bySetupModel: { [key: string]: any } = {};
    decisions.forEach(d => {
      const model = d.setupModel || 'none';
      if (!bySetupModel[model]) {
        bySetupModel[model] = { count: 0, wins: 0, losses: 0 };
      }
      bySetupModel[model].count++;

      const outcome = this.outcomes.get(d.id);
      if (outcome) {
        if (outcome.profitLoss > 0) bySetupModel[model].wins++;
        if (outcome.profitLoss < 0) bySetupModel[model].losses++;
      }
    });

    return {
      totalDecisions: decisions.length,
      totalFilled: decisions.filter(d => d.status === 'filled').length,
      totalOutcomes: outcomes.length,
      winners: winners.length,
      losers: losers.length,
      winRate: outcomes.length > 0 ? (winners.length / outcomes.length) * 100 : 0,
      avgWin: winners.length > 0 ? totalWinAmount / winners.length : 0,
      avgLoss: losers.length > 0 ? totalLossAmount / losers.length : 0,
      profitFactor: totalLossAmount > 0 ? totalWinAmount / totalLossAmount : 0,
      bySource,
      bySetupModel,
    };
  }

  /**
   * Get recent decisions for model confidence calibration
   */
  getRecentDecisions(limit: number = 50, symbol?: string): TradingDecision[] {
    const decisions = symbol
      ? this.getDecisionsBySymbol(symbol)
      : Array.from(this.decisions.values());

    return decisions.slice(0, limit);
  }

  /**
   * Get decisions grouped by confidence for analysis
   */
  getDecisionsByConfidence(symbol?: string): { [key: string]: TradingDecision[] } {
    const decisions = symbol
      ? this.getDecisionsBySymbol(symbol)
      : Array.from(this.decisions.values());

    const grouped: { [key: string]: TradingDecision[] } = {
      'very_high': [],
      'high': [],
      'medium': [],
      'low': [],
    };

    decisions.forEach(d => {
      if (d.confidence >= 80) grouped['very_high'].push(d);
      else if (d.confidence >= 60) grouped['high'].push(d);
      else if (d.confidence >= 40) grouped['medium'].push(d);
      else grouped['low'].push(d);
    });

    return grouped;
  }

  /**
   * Private helper: append to JSONL file
   */
  private appendToFile(filePath: string, data: any) {
    const line = JSON.stringify(data) + '\n';
    appendFileSync(filePath, line);
  }

  /**
   * Load database from disk
   */
  private loadFromDisk() {
    // Load decisions
    if (existsSync(this.decisionsFile)) {
      const content = readFileSync(this.decisionsFile, 'utf-8');
      content.split('\n').forEach(line => {
        if (line.trim()) {
          try {
            const decision = JSON.parse(line) as TradingDecision;
            this.decisions.set(decision.id, decision);
          } catch (e) {
            console.error('Failed to parse decision:', e);
          }
        }
      });
    }

    // Load outcomes
    if (existsSync(this.outcomesFile)) {
      const content = readFileSync(this.outcomesFile, 'utf-8');
      content.split('\n').forEach(line => {
        if (line.trim()) {
          try {
            const outcome = JSON.parse(line) as TradeOutcome;
            this.outcomes.set(outcome.decisionId, outcome);
          } catch (e) {
            console.error('Failed to parse outcome:', e);
          }
        }
      });
    }
  }

  /**
   * Export all data for analysis
   */
  exportData() {
    return {
      decisions: Array.from(this.decisions.values()),
      outcomes: Array.from(this.outcomes.values()),
      stats: this.calculateStats(),
    };
  }
}

// Export singleton instance
export const tradingDB = new TradingDatabase();
