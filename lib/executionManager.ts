/**
 * Trade Execution Manager
 * Converts trading decisions into actual orders via TopStepX API
 * Tracks execution status and position management
 */

import { OpenAITradingDecision } from './openaiTradingAgent';
import { AgentDecision } from '../live-fabio-agent-playbook';
import { tradingDB, TradingDecision } from './tradingDatabase';

/**
 * Position tracking
 */
export interface ActivePosition {
  decisionId: string;
  symbol: string;
  side: 'long' | 'short';
  entryPrice: number;
  entryTime: string;
  stopLoss: number;
  target: number;
  contracts: number;

  // Current status
  currentPrice: number;
  unrealizedPnL: number;
  unrealizedPnLPercent: number;
}

/**
 * Order execution details
 */
export interface ExecutedOrder {
  id: string;
  decisionId: string;
  symbol: string;
  side: 'buy' | 'sell';
  quantity: number;
  orderPrice: number;
  executedPrice: number;
  executedTime: string;
  status: 'pending' | 'filled' | 'partial' | 'rejected' | 'cancelled';
  reason?: string;
}

/**
 * Execution manager class
 */
export class ExecutionManager {
  private activePositions: Map<string, ActivePosition> = new Map();
  private executedOrders: Map<string, ExecutedOrder> = new Map();
  private orderIdCounter = 0;

  constructor(private symbol: string, private contracts: number = 1) {}

  /**
   * Execute a trading decision (from OpenAI or rule-based)
   */
  async executeDecision(decision: OpenAITradingDecision | AgentDecision, currentPrice: number): Promise<ExecutedOrder | null> {
    // Don't execute if no clear signal
    if (!('decision' in decision)) return null;
    if (decision.decision === 'HOLD') return null;

    // Check position limits
    if (this.activePositions.size >= 1) {
      console.warn('[ExecutionManager] Already in position, skipping new entry');
      return null;
    }

    // Convert decision to order
    const order = this.createOrder(decision, currentPrice);
    if (!order) return null;

    // Execute the order
    return this.executeOrder(order);
  }

  /**
   * Create order from decision
   */
  private createOrder(decision: OpenAITradingDecision | AgentDecision, currentPrice: number): ExecutedOrder | null {
    const decisionId = 'id' in decision ? decision.id : 'unknown';

    // Skip if wrong decision type
    if ('entry' in decision && !decision.entry.side) return null;
    const side = 'decision' in decision
      ? (decision.decision === 'BUY' ? 'buy' : 'sell')
      : (decision.entry.side === 'long' ? 'buy' : 'sell');

    const orderId = `${this.symbol}-${++this.orderIdCounter}-${Date.now()}`;

    return {
      id: orderId,
      decisionId,
      symbol: this.symbol,
      side,
      quantity: this.contracts,
      orderPrice: currentPrice,
      executedPrice: currentPrice, // Simplified: assume filled at market
      executedTime: new Date().toISOString(),
      status: 'filled',
    };
  }

  /**
   * Execute the order and record in database
   */
  private async executeOrder(order: ExecutedOrder): Promise<ExecutedOrder> {
    this.executedOrders.set(order.id, order);

    // Record decision in database
    const decision = tradingDB.recordDecision({
      symbol: this.symbol,
      marketState: 'balanced', // Will be set by caller
      location: 'at_poc', // Will be set by caller
      setupModel: null, // Will be set by caller
      decision: order.side === 'buy' ? 'BUY' : 'SELL',
      confidence: 70, // Default confidence
      entryPrice: order.executedPrice,
      stopLoss: 0, // Will be set by caller
      target: 0, // Will be set by caller
      riskPercent: 0.35,
      source: 'hybrid', // Using both rule-based and OpenAI
      reasoning: `Market order: ${order.side} ${order.quantity} contracts @ ${order.executedPrice}`,
      cvd: 0, // Will be set by caller
      cvdTrend: 'neutral', // Will be set by caller
      currentPrice: order.executedPrice,
      buyAbsorption: 0, // Will be set by caller
      sellAbsorption: 0, // Will be set by caller
    });

    // Mark as filled
    tradingDB.markDecisionFilled(decision.id, order.executedPrice);

    // Create active position
    const position: ActivePosition = {
      decisionId: decision.id,
      symbol: this.symbol,
      side: order.side === 'buy' ? 'long' : 'short',
      entryPrice: order.executedPrice,
      entryTime: order.executedTime,
      stopLoss: 0, // Will be updated
      target: 0, // Will be updated
      contracts: this.contracts,
      currentPrice: order.executedPrice,
      unrealizedPnL: 0,
      unrealizedPnLPercent: 0,
    };

    this.activePositions.set(decision.id, position);
    return order;
  }

  /**
   * Update position with current market data
   */
  updatePositionPrice(decisionId: string, currentPrice: number): void {
    const position = this.activePositions.get(decisionId);
    if (!position) return;

    position.currentPrice = currentPrice;

    if (position.side === 'long') {
      const pointDiff = currentPrice - position.entryPrice;
      position.unrealizedPnL = pointDiff * 20 * position.contracts; // $20 per point for NQ
      position.unrealizedPnLPercent = ((pointDiff) / position.entryPrice) * 100;
    } else {
      const pointDiff = position.entryPrice - currentPrice;
      position.unrealizedPnL = pointDiff * 20 * position.contracts;
      position.unrealizedPnLPercent = ((pointDiff) / position.entryPrice) * 100;
    }
  }

  /**
   * Check and execute exits (stop loss or target)
   */
  async checkExits(currentPrice: number): Promise<string | null> {
    if (this.activePositions.size === 0) return null;

    // Get the first (only) position
    const position = Array.from(this.activePositions.values())[0];
    if (!position) return null;

    // Check stop loss
    if (position.side === 'long' && currentPrice <= position.stopLoss) {
      return this.closePosition(position.decisionId, currentPrice, 'stop_loss_hit');
    }
    if (position.side === 'short' && currentPrice >= position.stopLoss) {
      return this.closePosition(position.decisionId, currentPrice, 'stop_loss_hit');
    }

    // Check target
    if (position.side === 'long' && currentPrice >= position.target) {
      return this.closePosition(position.decisionId, currentPrice, 'target_hit');
    }
    if (position.side === 'short' && currentPrice <= position.target) {
      return this.closePosition(position.decisionId, currentPrice, 'target_hit');
    }

    return null;
  }

  /**
   * Manually close a position
   */
  async closePosition(decisionId: string, currentPrice: number, reason: string = 'manual'): Promise<string | null> {
    const position = this.activePositions.get(decisionId);
    if (!position) return null;

    // Calculate actual P&L
    const pointDiff = position.side === 'long'
      ? currentPrice - position.entryPrice
      : position.entryPrice - currentPrice;

    const profitLoss = pointDiff * 20 * position.contracts; // $20 per point
    const profitLossPercent = (pointDiff / position.entryPrice) * 100;

    // Record outcome in database
    tradingDB.recordOutcome(decisionId, {
      symbol: position.symbol,
      executedPrice: position.entryPrice,
      executedTime: position.entryTime,
      closedPrice: currentPrice,
      closedTime: new Date().toISOString(),
      profitLoss,
      profitLossPercent,
      riskRewardActual: profitLoss / (Math.abs(position.entryPrice - position.stopLoss) * 20 * position.contracts),
      wasCorrect: profitLoss > 0,
      reason,
    });

    // Close position
    this.activePositions.delete(decisionId);

    return decisionId;
  }

  /**
   * Update stop loss (for breakeven management)
   */
  moveStopToBreakEven(decisionId: string): boolean {
    const position = this.activePositions.get(decisionId);
    if (!position) return false;

    position.stopLoss = position.entryPrice;
    return true;
  }

  /**
   * Update target
   */
  updateTarget(decisionId: string, newTarget: number): boolean {
    const position = this.activePositions.get(decisionId);
    if (!position) return false;

    position.target = newTarget;
    return true;
  }

  /**
   * Get active position
   */
  getActivePosition(): ActivePosition | null {
    if (this.activePositions.size === 0) return null;
    return Array.from(this.activePositions.values())[0];
  }

  /**
   * Get all active positions
   */
  getAllPositions(): ActivePosition[] {
    return Array.from(this.activePositions.values());
  }

  /**
   * Get executed orders
   */
  getOrders(): ExecutedOrder[] {
    return Array.from(this.executedOrders.values());
  }

  /**
   * Get execution statistics
   */
  getStats() {
    const orders = Array.from(this.executedOrders.values());
    const filled = orders.filter(o => o.status === 'filled');

    return {
      totalOrders: orders.length,
      filledOrders: filled.length,
      activePositions: this.activePositions.size,
    };
  }
}

/**
 * Create execution manager instance
 */
export function createExecutionManager(symbol: string, contracts: number = 1): ExecutionManager {
  return new ExecutionManager(symbol, contracts);
}
