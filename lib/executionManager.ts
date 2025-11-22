/**
 * Trade Execution Manager
 * Converts trading decisions into actual orders via TopStepX API
 * Tracks execution status and position management
 */

import { OpenAITradingDecision } from './openaiTradingAgent';
import { AgentDecision } from '../live-fabio-agent-playbook';
import { tradingDB, TradingDecision } from './tradingDatabase';
import {
  submitTopstepXOrder,
  selectTradingAccount,
  fetchTopstepXAccounts,
  TopstepXAccount,
  OrderSide,
} from './topstepx';
import { createProjectXRest, ProjectXOrderRecord } from '../projectx-rest';

interface ExecutionManagerOptions {
  tickSize?: number;
  preferredAccountId?: number;
  enableNativeBrackets?: boolean;
}

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
  stopOrderId?: string | number;
  targetOrderId?: string | number;
  usesNativeBracket?: boolean;
}

/**
 * Order execution details
 */
// Order intent categorization for validation
export type OrderIntent = 'ENTRY' | 'EXIT' | 'MODIFY_BRACKET';

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
  intent?: OrderIntent; // Track order purpose for validation
}

interface ExecutionResult {
  order: ExecutedOrder;
  stopLoss: number;
  takeProfit: number;
  stopOrderId?: string | number;
  targetOrderId?: string | number;
  usesNativeBracket?: boolean;
}

/**
 * Execution manager class
 */
export class ExecutionManager {
  private activePositions: Map<string, ActivePosition> = new Map();
  private executedOrders: Map<string, ExecutedOrder> = new Map();
  private orderIdCounter = 0;
  private tradingAccount: TopstepXAccount | null = null;
  private restClient: ReturnType<typeof createProjectXRest> | null = null;
  private readonly tickSize: number;
  private readonly riskPoints = 25;
  private readonly rewardPoints = 50;
  private readonly preferredAccountId?: number;
  private readonly enableNativeBrackets: boolean;
  private lastBrokerSync = 0;
  private readonly brokerSyncIntervalMs = 15_000;
  private lastEntryOrderTime = 0;
  private readonly entryOrderGracePeriodMs = 90_000; // 90 seconds to prevent duplicate entries after stop loss

  // Broker position monitoring to prevent naked close orders
  private brokerHasPosition = false;
  private lastBrokerPositionCheck = 0;
  private readonly brokerPositionCheckIntervalMs = 2_000; // Check every 2 seconds

  constructor(
    private symbol: string,
    private contractId: string,
    private contracts: number = 1,
    private live: boolean = false,  // false = sim/demo, true = live trading
    options: ExecutionManagerOptions = {}
  ) {
    this.tickSize = options.tickSize && options.tickSize > 0 ? options.tickSize : 0.25;
    this.preferredAccountId = options.preferredAccountId;
    this.enableNativeBrackets = Boolean(options.enableNativeBrackets);
    try {
      this.restClient = createProjectXRest();
    } catch (error: any) {
      console.warn('[ExecutionManager] Unable to initialize ProjectX REST client (falling back to legacy order flow):', error?.message || error);
      this.restClient = null;
    }
  }

  private async ensureRestContext(): Promise<boolean> {
    if (!this.restClient) {
      try {
        this.restClient = createProjectXRest();
      } catch (error: any) {
        console.warn('[ExecutionManager] Unable to initialize REST client on-demand:', error?.message || error);
        return false;
      }
    }

    if (!this.tradingAccount) {
      const initialized = await this.initializeTradingAccount();
      if (!initialized) {
        console.warn('[ExecutionManager] No trading account available; cannot adjust protective orders.');
        return false;
      }
    }

    return true;
  }

  private async executeWithRest(
    order: ExecutedOrder,
    currentPrice: number,
    overrides?: ExecutionOverrides
  ): Promise<ExecutionResult | null> {
    const hasRestContext = await this.ensureRestContext();
    if (!hasRestContext || !this.restClient || !this.tradingAccount) {
      return null;
    }

    // CRITICAL: Validate ENTRY order against broker state
    console.log(`[ExecutionManager] 🔍 [ENTRY ORDER VALIDATION] Checking broker is FLAT...`);
    const brokerHasPosition = await this.refreshBrokerPositionStatus();

    if (brokerHasPosition) {
      console.error(`[ExecutionManager] 🚫 [ENTRY ORDER BLOCKED] Position already exists at broker! Cannot enter duplicate position.`);
      order.status = 'rejected';
      order.reason = 'ENTRY blocked: Position already exists at broker';
      return null;
    }

    console.log(`[ExecutionManager] ✅ [ENTRY ORDER APPROVED] Broker is FLAT - proceeding with entry`);

    const entryReference = overrides?.entryPrice ?? currentPrice;
    const plannedLevels = this.calculateBracketLevels(entryReference, order.side, overrides);
    const { stopTicks, targetTicks } = this.getBracketTicks(
      order.side,
      entryReference,
      plannedLevels.stopLoss,
      plannedLevels.takeProfit,
    );

    try {
      console.log(`[ExecutionManager] Submitting ${order.side.toUpperCase()} OCO bracket via REST API...`);
      order.intent = 'ENTRY';
      this.lastEntryOrderTime = Date.now();
      const response = await this.restClient.placeOrder({
        accountId: this.tradingAccount.id,
        contractId: this.contractId,
        side: order.side === 'buy' ? 0 : 1,
        size: order.quantity,
        type: 2, // Market
        timeInForce: 1, // GTC so brackets persist
        stopLossBracket: {
          ticks: stopTicks,
          type: 4, // Stop market
        },
        takeProfitBracket: {
          ticks: targetTicks,
          type: 1, // Limit
        },
      });

      if (response?.success === false) {
        console.error('[ExecutionManager] REST entry rejected:', response.errorMessage);
        order.status = 'rejected';
        order.reason = response.errorMessage || 'Order rejected';
        return null;
      }

      order.status = 'filled';
      order.executedPrice = typeof response?.averagePrice === 'number'
        ? response.averagePrice
        : currentPrice;
      order.executedTime = new Date().toISOString();
      order.id = response?.orderId ? String(response.orderId) : order.id;

      const finalEntryRef = overrides?.entryPrice ?? order.executedPrice;
      const { stopLoss, takeProfit } = this.calculateBracketLevels(
        finalEntryRef,
        order.side,
        overrides
      );

      let stopOrderId = response?.stopOrderId
        ?? response?.stopLossOrderId
        ?? response?.stopBracketOrderId;
      let targetOrderId = response?.targetOrderId
        ?? response?.takeProfitOrderId
        ?? response?.takeProfitBracketOrderId;

      console.log(`[ExecutionManager] 📋 Extracted bracket IDs from response:`, {
        stopOrderId: stopOrderId || 'MISSING',
        targetOrderId: targetOrderId || 'MISSING',
        rawResponse: JSON.stringify(response).substring(0, 200)
      });

      // ALWAYS verify to find the bracket IDs created by the broker
      console.log('[ExecutionManager] Verifying native bracket legs via searchOpenOrders...');
      const verified = await this.verifyNativeBracket(stopLoss, takeProfit, order.side);
      stopOrderId = stopOrderId ?? verified.stopOrderId;
      targetOrderId = targetOrderId ?? verified.targetOrderId;
      const hasNativeBracket = Boolean(stopOrderId && targetOrderId);

      if (!hasNativeBracket) {
        console.warn('[ExecutionManager] ⚠️ REST bracket legs not found after verification. Placing protective orders manually.');
        const fallbackBrackets = await this.submitBracketOrders(order, stopLoss, takeProfit);

        // CRITICAL VALIDATION: Ensure BOTH brackets were created successfully
        const hasBothBrackets = Boolean(fallbackBrackets.stopOrderId && fallbackBrackets.targetOrderId);

        if (!hasBothBrackets) {
          console.error('[ExecutionManager] ❌ CRITICAL: Fallback brackets FAILED! Position would be NAKED!');
          console.error(`[ExecutionManager] Stop: ${fallbackBrackets.stopOrderId ? 'OK' : 'MISSING'}, Target: ${fallbackBrackets.targetOrderId ? 'OK' : 'MISSING'}`);
          console.error('[ExecutionManager] 🚫 EMERGENCY: Closing naked position at broker immediately!');

          // CRITICAL: Entry order was already filled at broker, must close it immediately
          try {
            const closeSide = order.side === 'buy' ? 1 : 0; // Opposite side to close
            console.log(`[ExecutionManager] 🚨 Placing emergency MARKET ${closeSide === 0 ? 'BUY' : 'SELL'} to close naked position...`);

            const closeResponse = await this.restClient.placeOrder({
              accountId: this.tradingAccount.id,
              contractId: this.contractId,
              side: closeSide,
              size: order.quantity,
              type: 2, // Market order to close immediately
              timeInForce: 3, // IOC (Immediate or Cancel)
            });

            if (closeResponse?.success) {
              console.log('[ExecutionManager] ✅ Naked position successfully closed at broker');
            } else {
              console.error('[ExecutionManager] ❌ FAILED TO CLOSE NAKED POSITION!', closeResponse?.errorMessage);
              console.error('[ExecutionManager] 🚨 MANUAL INTERVENTION REQUIRED - Naked position exists at broker!');
            }
          } catch (closeError: any) {
            console.error('[ExecutionManager] ❌ Exception while closing naked position:', closeError.message);
            console.error('[ExecutionManager] 🚨 MANUAL INTERVENTION REQUIRED - Naked position exists at broker!');
          }

          // Mark order as failed and return null to prevent local tracking
          order.status = 'rejected';
          order.reason = 'Failed to create protective brackets - naked position closed at broker';
          return null;
        }

        console.log('[ExecutionManager] ✅ Fallback brackets successfully created - Stop & Target both confirmed');
        return {
          order,
          stopLoss,
          takeProfit,
          stopOrderId: fallbackBrackets.stopOrderId,
          targetOrderId: fallbackBrackets.targetOrderId,
          usesNativeBracket: false,
        };
      }

      console.log(`[ExecutionManager] ✅ REST bracket placed (Order ID: ${order.id})`);
      return {
        order,
        stopLoss,
        takeProfit,
        stopOrderId,
        targetOrderId,
        usesNativeBracket: hasNativeBracket,
      };
    } catch (error: any) {
      console.error('[ExecutionManager] REST bracket submission failed:', error?.message || error);
      return null;
    }
  }

  private async executeWithLegacy(order: ExecutedOrder, overrides?: ExecutionOverrides): Promise<ExecutionResult | null> {
    try {
      console.log(`[ExecutionManager] Submitting ${order.side.toUpperCase()} ENTRY order to TopStepX...`);
      this.lastEntryOrderTime = Date.now();

      const topstepXOrder = {
        contractId: this.contractId,
        accountId: this.tradingAccount!.id,
        side: order.side as OrderSide,
        quantity: order.quantity,
        orderType: 'market' as const,
        live: this.live,
      };

      const result = await submitTopstepXOrder(topstepXOrder);

      if (!result.success) {
        console.error('[ExecutionManager] Entry order rejected by TopStepX:', result.errorMessage);
        order.status = 'rejected';
        order.reason = result.errorMessage || 'Order rejected';
        return null;
      }

      order.status = 'filled';
      order.executedPrice = result.averagePrice || order.executedPrice;
      order.id = result.orderId || order.id;
      order.executedTime = new Date().toISOString();

      console.log(`[ExecutionManager] ✅ Entry order filled: ${order.id} @ ${order.executedPrice}`);

      const finalEntryRef = overrides?.entryPrice ?? order.executedPrice;
      const { stopLoss, takeProfit } = this.calculateBracketLevels(finalEntryRef, order.side, overrides);

      const { stopOrderId, targetOrderId } = await this.submitBracketOrders(order, stopLoss, takeProfit);

      return {
        order,
        stopLoss,
        takeProfit,
        stopOrderId,
        targetOrderId,
        usesNativeBracket: false,
      };
    } catch (error: any) {
      console.error('[ExecutionManager] Order execution failed:', error.message);
      order.status = 'rejected';
      order.reason = error.message;
      return null;
    }
  }

  /**
   * Initialize trading account (select account with balance < 40k)
   */
  async initializeTradingAccount(): Promise<boolean> {
    try {
      if (this.preferredAccountId && this.preferredAccountId > 0) {
        const accounts = await fetchTopstepXAccounts(true);
        const desired = accounts.find(acc => acc.id === this.preferredAccountId);
        if (!desired) {
          console.error(`[ExecutionManager] Preferred account ${this.preferredAccountId} not found or not accessible.`);
        } else if (!desired.canTrade) {
          console.error(`[ExecutionManager] Preferred account ${this.preferredAccountId} cannot trade (disabled).`);
        } else {
          this.tradingAccount = desired;
          console.log(`[ExecutionManager] Using preferred account: ${desired.name} (ID: ${desired.id})`);
        }
      }

      if (!this.tradingAccount) {
        this.tradingAccount = await selectTradingAccount(40000);
      }

      if (!this.tradingAccount) {
        console.error('[ExecutionManager] No eligible trading account found with balance < $40,000');
        return false;
      }

      console.log(`[ExecutionManager] Initialized with account: ${this.tradingAccount.name} (Balance: $${this.tradingAccount.balance.toFixed(2)})`);
      return true;
    } catch (error: any) {
      console.error('[ExecutionManager] Failed to initialize trading account:', error.message);
      return false;
    }
  }

  /**
   * Execute a trading decision (from OpenAI or rule-based)
   */
  async executeDecision(
    decision: OpenAITradingDecision | AgentDecision,
    currentPrice: number,
    overrides?: ExecutionOverrides
  ): Promise<ExecutedOrder | null> {
    // Don't execute if no clear signal
    if (!('decision' in decision)) return null;
    if (decision.decision === 'HOLD') return null;

    // CRITICAL: Check broker for open positions BEFORE executing
    // This prevents duplicate entries when AI response is stale/delayed
    console.log('[ExecutionManager] 🔍 Checking broker for existing positions before entry...');
    try {
      const hasRestContext = await this.ensureRestContext();
      if (hasRestContext && this.restClient && this.tradingAccount) {
        const brokerPositions = await this.restClient.getPositions(this.tradingAccount.id);
        const existingPosition = (brokerPositions || []).find(
          (pos: any) => this.extractNetQuantity(pos) !== 0 && pos.contractId === this.contractId
        );

        if (existingPosition) {
          const qty = this.extractNetQuantity(existingPosition);
          console.warn(`[ExecutionManager] 🚫 BROKER CHECK: Position already exists at broker (${qty > 0 ? 'LONG' : 'SHORT'} ${Math.abs(qty)} contracts). Rejecting new entry to prevent duplicate.`);
          return null;
        }

        // Also check for open orders that might be filling
        const openOrders = await this.restClient.searchOpenOrders({ accountId: this.tradingAccount.id });
        const pendingEntryOrders = (openOrders?.orders || []).filter(
          (order: any) => order.contractId === this.contractId && order.type === 2 && order.status === 1
        );

        if (pendingEntryOrders.length > 0) {
          console.warn(`[ExecutionManager] 🚫 BROKER CHECK: ${pendingEntryOrders.length} pending entry order(s) at broker. Rejecting new entry to prevent duplicate.`);
          return null;
        }

        console.log('[ExecutionManager] ✅ BROKER CHECK: No existing position or pending orders. Safe to proceed.');
      }
    } catch (error: any) {
      console.error('[ExecutionManager] ⚠️ Broker check failed:', error.message);
      // Continue anyway - don't block trading due to API issues
    }

    // Check position limits (local state as backup)
    if (this.activePositions.size >= 1) {
      console.warn('[ExecutionManager] Already in position (local check), skipping new entry');
      return null;
    }

    // CRITICAL: Check grace period to prevent duplicate entries after stop loss
    // If we just entered a position within the last 30 seconds, don't enter another
    const timeSinceLastEntry = Date.now() - this.lastEntryOrderTime;
    if (this.lastEntryOrderTime > 0 && timeSinceLastEntry < this.entryOrderGracePeriodMs) {
      const secondsRemaining = Math.ceil((this.entryOrderGracePeriodMs - timeSinceLastEntry) / 1000);
      console.warn(`[ExecutionManager] 🚫 DUPLICATE ENTRY PREVENTION: Last entry was ${Math.ceil(timeSinceLastEntry / 1000)}s ago. Must wait ${secondsRemaining}s before new entry (grace period: ${this.entryOrderGracePeriodMs / 1000}s)`);
      return null;
    }

    // Convert decision to order
    const order = this.createOrder(decision, currentPrice);
    if (!order) return null;

    // Execute the order
    return this.executeOrder(order, currentPrice, overrides);
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
  private async executeOrder(
    order: ExecutedOrder,
    currentPrice: number,
    overrides?: ExecutionOverrides
  ): Promise<ExecutedOrder | null> {
    // Ensure trading account is initialized
    if (!this.tradingAccount) {
      console.error('[ExecutionManager] Trading account not initialized. Call initializeTradingAccount() first.');
      order.status = 'rejected';
      order.reason = 'No trading account configured';
      return null;
    }

    let executionResult: ExecutionResult | null = null;

    if (this.restClient && this.enableNativeBrackets) {
      executionResult = await this.executeWithRest(order, currentPrice, overrides);
      if (!executionResult) {
        console.warn('[ExecutionManager] REST bracket order failed, falling back to legacy submission');
      }
    }

    if (!executionResult) {
      executionResult = await this.executeWithLegacy(order, overrides);
    }

    if (!executionResult) {
      console.error('[ExecutionManager] Unable to execute order via any channel');
      return null;
    }

    const { stopLoss, takeProfit, stopOrderId, targetOrderId, usesNativeBracket } = executionResult;

    this.executedOrders.set(order.id, order);

    // Record decision in database (basic placeholder; enriched elsewhere)
    const decision = tradingDB.recordDecision({
      symbol: this.symbol,
      marketState: 'balanced',
      location: 'at_poc',
      setupModel: null,
      decision: order.side === 'buy' ? 'BUY' : 'SELL',
      confidence: 70,
      entryPrice: order.executedPrice,
      stopLoss,
      target: takeProfit,
      riskPercent: 0.35,
      source: 'hybrid',
      reasoning: `Market order: ${order.side} ${order.quantity} contracts @ ${order.executedPrice}`,
      cvd: 0,
      cvdTrend: 'neutral',
      currentPrice: order.executedPrice,
      buyAbsorption: 0,
      sellAbsorption: 0,
    });

    tradingDB.markDecisionFilled(decision.id, order.executedPrice);

    const position: ActivePosition = {
      decisionId: decision.id,
      symbol: this.symbol,
      side: order.side === 'buy' ? 'long' : 'short',
      entryPrice: order.executedPrice,
      entryTime: order.executedTime,
      stopLoss,
      target: takeProfit,
      contracts: this.contracts,
      currentPrice: order.executedPrice,
      unrealizedPnL: 0,
      unrealizedPnLPercent: 0,
      stopOrderId,
      targetOrderId,
      usesNativeBracket,
    };

    this.activePositions.set(decision.id, position);

    if (!position.stopOrderId || !position.targetOrderId) {
      console.log('[ExecutionManager] ⚠️ Missing bracket order IDs, attempting to sync from broker...');
      await this.ensureBracketOrderIds(position);

      // CRITICAL VALIDATION: Verify brackets exist after sync attempt
      if (!position.stopOrderId || !position.targetOrderId) {
        console.error('[ExecutionManager] ❌ CRITICAL: Could not find or create bracket order IDs!');
        console.error(`[ExecutionManager] Stop ID: ${position.stopOrderId || 'MISSING'}, Target ID: ${position.targetOrderId || 'MISSING'}`);
        console.error('[ExecutionManager] 🚫 EMERGENCY: Closing naked position at broker immediately!');

        // CRITICAL: Position exists at broker without protection, must close immediately
        try {
          const closeSide = order.side === 'buy' ? 1 : 0; // Opposite side to close
          console.log(`[ExecutionManager] 🚨 Placing emergency MARKET ${closeSide === 0 ? 'BUY' : 'SELL'} to close naked position...`);

          const closeResponse = await this.restClient!.placeOrder({
            accountId: this.tradingAccount!.id,
            contractId: this.contractId,
            side: closeSide,
            size: order.quantity,
            type: 2, // Market order to close immediately
            timeInForce: 3, // IOC (Immediate or Cancel)
          });

          if (closeResponse?.success) {
            console.log('[ExecutionManager] ✅ Naked position successfully closed at broker');
          } else {
            console.error('[ExecutionManager] ❌ FAILED TO CLOSE NAKED POSITION!', closeResponse?.errorMessage);
            console.error('[ExecutionManager] 🚨 MANUAL INTERVENTION REQUIRED - Naked position exists at broker!');
          }
        } catch (closeError: any) {
          console.error('[ExecutionManager] ❌ Exception while closing naked position:', closeError.message);
          console.error('[ExecutionManager] 🚨 MANUAL INTERVENTION REQUIRED - Naked position exists at broker!');
        }

        // Remove the naked position from tracking
        this.activePositions.delete(decision.id);

        // Mark order as failed
        order.status = 'rejected';
        order.reason = 'Could not establish protective brackets - naked position closed at broker';

        return null;
      }

      console.log('[ExecutionManager] ✅ Bracket order IDs successfully synced');
    }
    return order;
  }

  /**
   * Calculate strategic stop loss and take profit levels
   * Uses ATR-based risk management with 2:1 reward-to-risk ratio
   */
  private calculateBracketLevels(
    entryPrice: number,
    side: 'buy' | 'sell',
    overrides?: ExecutionOverrides
  ): { stopLoss: number; takeProfit: number } {
    const riskPoints = this.riskPoints;
    const rewardPoints = this.rewardPoints;

    const defaultBuy = {
      stopLoss: entryPrice - riskPoints,
      takeProfit: entryPrice + rewardPoints,
    };
    const defaultSell = {
      stopLoss: entryPrice + riskPoints,
      takeProfit: entryPrice - rewardPoints,
    };

    const defaults = side === 'buy' ? defaultBuy : defaultSell;

    return {
      stopLoss: overrides?.stopLoss ?? defaults.stopLoss,
      takeProfit: overrides?.takeProfit ?? defaults.takeProfit,
    };
  }

  private getBracketTicks(
    side: 'buy' | 'sell',
    entryReference: number,
    stopPrice: number,
    targetPrice: number
  ): { stopTicks: number; targetTicks: number } {
    const rawStopTicks = Math.round((stopPrice - entryReference) / this.tickSize);
    const rawTargetTicks = Math.round((targetPrice - entryReference) / this.tickSize);

    const stopTicks = rawStopTicks !== 0
      ? rawStopTicks
      : side === 'buy'
        ? -1
        : 1;

    const targetTicks = rawTargetTicks !== 0
      ? rawTargetTicks
      : side === 'buy'
        ? 1
        : -1;

    return {
      stopTicks,
      targetTicks,
    };
  }

  private normalizePrice(price: number): number {
    if (!Number.isFinite(price)) {
      return price;
    }
    const normalized = Math.round(price / this.tickSize) * this.tickSize;

    // Fix floating point precision errors by rounding to appropriate decimal places
    // Determine decimal places needed for the tick size
    // For 0.25: need 2 places, for 0.1: need 1 place, for 1: need 0 places
    // Handle scientific notation by using toFixed with a large number first
    const tickStr = this.tickSize < 1 ? this.tickSize.toFixed(10).replace(/0+$/, '') : this.tickSize.toString();
    const decimalIndex = tickStr.indexOf('.');
    const decimalPlaces = decimalIndex >= 0 ? tickStr.length - decimalIndex - 1 : 0;

    return Number(normalized.toFixed(decimalPlaces));
  }

  private getProtectiveSide(position: { side: 'long' | 'short' }): 0 | 1 {
    return position.side === 'long' ? 1 : 0;
  }

  private extractNetQuantity(positionData: any): number {
    const fields = ['netQty', 'quantity', 'size', 'position'];
    for (const key of fields) {
      const value = Number(positionData?.[key]);
      if (Number.isFinite(value) && value !== 0) {
        return value;
      }
    }
    return 0;
  }

  private extractEntryPrice(positionData: any): number {
    const fields = ['avgPrice', 'averagePrice', 'entryPrice', 'price', 'filledPrice', 'openPrice', 'lastPrice', 'markPrice'];
    for (const key of fields) {
      const value = Number(positionData?.[key]);
      if (Number.isFinite(value) && value !== 0) {
        return value;
      }
    }
    return 0;
  }

  private extractEntryTimestamp(positionData: any): string {
    const candidates = [
      positionData?.openTimestamp,
      positionData?.creationTimestamp,
      positionData?.updateTimestamp,
      positionData?.timestamp,
    ];

    for (const ts of candidates) {
      if (typeof ts === 'string') {
        const parsed = Date.parse(ts);
        if (!Number.isNaN(parsed)) {
          return new Date(parsed).toISOString();
        }
      }
    }

    return new Date().toISOString();
  }

  private pickClosestOrder(
    orders: ProjectXOrderRecord[],
    desiredPrice: number,
    field: 'stopPrice' | 'limitPrice'
  ): ProjectXOrderRecord | undefined {
    const filtered = orders.filter(order => typeof order[field] === 'number' && Number.isFinite(order[field] as number));
    if (filtered.length === 0) {
      return undefined;
    }

    return filtered.reduce((best, current) => {
      if (!best) return current;
      const bestPrice = best[field] as number;
      const currentPrice = current[field] as number;
      return Math.abs(currentPrice - desiredPrice) < Math.abs(bestPrice - desiredPrice)
        ? current
        : best;
    });
  }

  private async ensureBracketOrderIds(position: ActivePosition): Promise<void> {
    if (!this.restClient || !this.tradingAccount) {
      return;
    }

    if (position.stopOrderId && position.targetOrderId) {
      return;
    }

    try {
      const response = await this.restClient.searchOpenOrders({ accountId: this.tradingAccount.id });
      if (!response?.success) {
        console.warn('[ExecutionManager] Failed to search open orders:', response?.errorMessage || 'Unknown error');
        return;
      }

      const protectiveSide = this.getProtectiveSide(position);
      const relevantOrders = (response.orders || []).filter(order =>
        order.contractId === this.contractId && order.side === protectiveSide
      );

      if (!position.stopOrderId) {
        const stopOrder = this.pickClosestOrder(relevantOrders, position.stopLoss, 'stopPrice');
        if (stopOrder) {
          position.stopOrderId = stopOrder.id;
          console.log(`[ExecutionManager] Synced stop order ID: ${stopOrder.id}`);
        }
      }

      if (!position.targetOrderId) {
        const targetOrder = this.pickClosestOrder(relevantOrders, position.target, 'limitPrice');
        if (targetOrder) {
          position.targetOrderId = targetOrder.id;
          console.log(`[ExecutionManager] Synced target order ID: ${targetOrder.id}`);
        }
      }
    } catch (error: any) {
      console.warn('[ExecutionManager] Unable to sync bracket order IDs:', error?.message || error);
    }
  }

  /**
   * Check broker state to confirm flat (no position and no protective orders)
   */
  private async verifyPositionFlattened(): Promise<boolean> {
    if (!this.restClient || !this.tradingAccount) {
      return false;
    }

    try {
      // Check open orders for this contract
      const openOrders = await this.restClient.searchOpenOrders({ accountId: this.tradingAccount.id });
      const hasProtective = (openOrders?.orders || []).some(o => o.contractId === this.contractId);

      // Check positions
      const positions = await this.restClient.getPositions(this.tradingAccount.id);
      const hasPosition = Array.isArray(positions) && positions.some((p: any) => p.contractId === this.contractId && this.extractNetQuantity(p) !== 0);

      return !hasProtective && !hasPosition;
    } catch (error: any) {
      console.warn('[ExecutionManager] verifyPositionFlattened error:', error?.message || error);
      return false;
    }
  }

  private async verifyNativeBracket(
    stopLoss: number,
    takeProfit: number,
    side: 'buy' | 'sell'
  ): Promise<{ stopOrderId?: string | number; targetOrderId?: string | number }> {
    if (!this.restClient || !this.tradingAccount) {
      return {};
    }

    // Retry up to 3 times with delays to give broker time to create bracket orders
    const maxRetries = 3;
    const retryDelayMs = 1500;

    for (let attempt = 1; attempt <= maxRetries; attempt++) {
      try {
        if (attempt > 1) {
          console.log(`[ExecutionManager] verifyNativeBracket: Retry attempt ${attempt}/${maxRetries} after ${retryDelayMs}ms delay...`);
          await new Promise(resolve => setTimeout(resolve, retryDelayMs));
        }

        const response = await this.restClient.searchOpenOrders({ accountId: this.tradingAccount.id });
        if (!response?.success) {
          console.warn('[ExecutionManager] verifyNativeBracket: searchOpenOrders failed:', response?.errorMessage || 'Unknown error');
          if (attempt === maxRetries) {
            return {};
          }
          continue;
        }

        const protectiveSide = side === 'buy' ? 1 : 0;
        const relevantOrders = (response.orders || []).filter(order =>
          order.contractId === this.contractId && order.side === protectiveSide
        );

        console.log(`[ExecutionManager] verifyNativeBracket: Found ${relevantOrders.length} protective orders on attempt ${attempt}`);

        const stopOrder = this.pickClosestOrder(relevantOrders, stopLoss, 'stopPrice');
        const targetOrder = this.pickClosestOrder(relevantOrders, takeProfit, 'limitPrice');

        if (stopOrder && targetOrder) {
          console.log(`[ExecutionManager] ✅ verifyNativeBracket: Found both bracket legs on attempt ${attempt} - Stop: ${stopOrder.id}, Target: ${targetOrder.id}`);
          return {
            stopOrderId: stopOrder.id,
            targetOrderId: targetOrder.id,
          };
        }

        if (attempt < maxRetries) {
          console.log(`[ExecutionManager] verifyNativeBracket: Missing ${!stopOrder ? 'stop' : ''}${!stopOrder && !targetOrder ? ' and ' : ''}${!targetOrder ? 'target' : ''} on attempt ${attempt}, will retry...`);
        }
      } catch (error: any) {
        console.warn(`[ExecutionManager] verifyNativeBracket failed on attempt ${attempt}:`, error?.message || error);
        if (attempt === maxRetries) {
          return {};
        }
      }
    }

    console.warn(`[ExecutionManager] ⚠️ verifyNativeBracket: Failed to find both bracket legs after ${maxRetries} attempts`);
    return {};
  }

  async rehydrateActivePosition(): Promise<ActivePosition | null> {
    console.log(`[ExecutionManager] 🔄 REHYDRATE: Starting position rehydration for ${this.contractId}...`);

    if (!this.restClient || !this.tradingAccount) {
      console.warn('[ExecutionManager] ❌ REHYDRATE: Cannot rehydrate without REST client and trading account.');
      return null;
    }

    if (this.activePositions.size > 0) {
      console.log(`[ExecutionManager] ✅ REHYDRATE: Position already exists in memory (${this.activePositions.size} active position(s))`);
      return this.getActivePosition();
    }

    console.log(`[ExecutionManager] 🔍 REHYDRATE: Fetching positions from broker API for account ${this.tradingAccount.id}...`);

    try {
      const brokerPositions = await this.restClient.getPositions(this.tradingAccount.id);
      console.log(`[ExecutionManager] 📊 REHYDRATE: Broker API returned ${brokerPositions?.length || 0} position(s)`);

      if (brokerPositions && brokerPositions.length > 0) {
        console.log(`[ExecutionManager] 📋 REHYDRATE: Position details:`,
          JSON.stringify(brokerPositions.map((p: any) => ({
            contractId: p.contractId,
            quantity: this.extractNetQuantity(p),
            entryPrice: this.extractEntryPrice(p)
          })), null, 2)
        );
      }

      const matching = (brokerPositions || []).find((pos: any) => pos.contractId === this.contractId);

      if (!matching) {
        console.log(`[ExecutionManager] ⚠️ REHYDRATE: No broker position found for ${this.contractId} via positions endpoint.`);
        console.log(`[ExecutionManager] 🔄 REHYDRATE FALLBACK: Attempting to reconstruct position from open bracket orders...`);

        // FALLBACK: Try to reconstruct position from open bracket orders
        try {
          const openOrdersResponse = await this.restClient.searchOpenOrders({ accountId: this.tradingAccount.id });
          console.log(`[ExecutionManager] 📋 REHYDRATE FALLBACK: Open orders API returned: success=${openOrdersResponse?.success}, orderCount=${openOrdersResponse?.orders?.length || 0}`);

          if (openOrdersResponse?.success && Array.isArray(openOrdersResponse.orders) && openOrdersResponse.orders.length > 0) {
            const contractOrders = openOrdersResponse.orders.filter(order => order.contractId === this.contractId);
            console.log(`[ExecutionManager] 🔍 REHYDRATE FALLBACK: Found ${contractOrders.length} open orders for ${this.contractId}`);

            if (contractOrders.length === 0) {
              console.log(`[ExecutionManager] ❌ REHYDRATE FALLBACK: No open orders found for ${this.contractId} - position truly doesn't exist.`);
              return null;
            }

            // Find stop loss and take profit orders
            const stopOrders = contractOrders.filter(order => typeof order.stopPrice === 'number' && Number.isFinite(order.stopPrice));
            const limitOrders = contractOrders.filter(order => typeof order.limitPrice === 'number' && Number.isFinite(order.limitPrice));

            console.log(`[ExecutionManager] 📊 REHYDRATE FALLBACK: Found ${stopOrders.length} stop order(s) and ${limitOrders.length} limit order(s)`);

            if (stopOrders.length === 0 && limitOrders.length === 0) {
              console.log(`[ExecutionManager] ❌ REHYDRATE FALLBACK: No bracket orders found - cannot reconstruct position.`);
              return null;
            }

            // Determine position side from protective order side
            // If protective orders are BUY -> position is SHORT
            // If protective orders are SELL -> position is LONG
            const protectiveOrder = stopOrders[0] || limitOrders[0];
            const protectiveSide = protectiveOrder.side; // 0=Buy, 1=Sell
            const positionSide: 'long' | 'short' = protectiveSide === 0 ? 'short' : 'long';
            const quantity = protectiveOrder.size || 1;

            console.log(`[ExecutionManager] 🎯 REHYDRATE FALLBACK: Inferred position side: ${positionSide.toUpperCase()} (protective orders are ${protectiveSide === 0 ? 'BUY' : 'SELL'})`);
            console.log(`[ExecutionManager] 📊 REHYDRATE FALLBACK: Inferred quantity: ${quantity} contract(s)`);

            // Get stop loss and take profit levels
            let stopLoss: number | undefined;
            let target: number | undefined;
            let stopOrderId: string | number | undefined;
            let targetOrderId: string | number | undefined;

            if (stopOrders.length > 0) {
              const stopOrder = stopOrders[0];
              stopLoss = Number(stopOrder.stopPrice);
              stopOrderId = stopOrder.id;
              console.log(`[ExecutionManager] 🛑 REHYDRATE FALLBACK: Stop loss found at ${stopLoss.toFixed(2)} (Order ID: ${stopOrderId})`);
            }

            if (limitOrders.length > 0) {
              const limitOrder = limitOrders[0];
              target = Number(limitOrder.limitPrice);
              targetOrderId = limitOrder.id;
              console.log(`[ExecutionManager] 🎯 REHYDRATE FALLBACK: Take profit found at ${target.toFixed(2)} (Order ID: ${targetOrderId})`);
            }

            if (!stopLoss && !target) {
              console.log(`[ExecutionManager] ❌ REHYDRATE FALLBACK: No bracket levels found - cannot reconstruct position.`);
              return null;
            }

            // Try to find actual filled entry order from recent order history
            let entryPrice: number | null = null;
            let actualEntryTime: string | null = null;

            try {
              console.log(`[ExecutionManager] 🔍 REHYDRATE FALLBACK: Searching for filled entry order in recent history...`);
              const now = new Date();
              const oneDayAgo = new Date(now.getTime() - 24 * 60 * 60 * 1000);

              const ordersResponse = await this.restClient.searchOrders({
                accountId: this.tradingAccount.id,
                startTimestamp: oneDayAgo.toISOString(),
                endTimestamp: now.toISOString(),
              });

              if (ordersResponse?.success && Array.isArray(ordersResponse.orders)) {
                const entrySide = positionSide === 'long' ? 0 : 1; // 0=BUY, 1=SELL
                const filledEntryOrders = ordersResponse.orders.filter(order =>
                  order.contractId === this.contractId &&
                  order.side === entrySide &&
                  order.status === 2 && // 2 = Filled
                  order.fillVolume > 0 &&
                  order.filledPrice
                );

                console.log(`[ExecutionManager] 📋 REHYDRATE FALLBACK: Found ${filledEntryOrders.length} filled ${positionSide === 'long' ? 'BUY' : 'SELL'} order(s) in last 24h`);

                if (filledEntryOrders.length > 0) {
                  // Take the most recent filled entry order
                  const mostRecent = filledEntryOrders.sort((a, b) => {
                    const timeA = new Date(a.updateTimestamp || a.creationTimestamp).getTime();
                    const timeB = new Date(b.updateTimestamp || b.creationTimestamp).getTime();
                    return timeB - timeA;
                  })[0];

                  entryPrice = Number(mostRecent.filledPrice);
                  actualEntryTime = mostRecent.updateTimestamp || mostRecent.creationTimestamp;
                  console.log(`[ExecutionManager] ✅ REHYDRATE FALLBACK: Found actual filled entry order (ID: ${mostRecent.id}) at ${entryPrice.toFixed(2)}`);
                }
              }
            } catch (searchError: any) {
              console.warn(`[ExecutionManager] ⚠️ REHYDRATE FALLBACK: Could not search order history:`, searchError?.message);
            }

            // If we couldn't find actual entry, estimate from brackets
            if (!entryPrice) {
              console.log(`[ExecutionManager] 💡 REHYDRATE FALLBACK: No filled entry order found, estimating from brackets...`);

              // Estimate entry price from bracket levels
              // For LONG: entry should be between stop (below) and target (above)
              // For SHORT: entry should be between target (below) and stop (above)
              if (positionSide === 'long') {
                // LONG: stop < entry < target
                if (stopLoss && target) {
                  entryPrice = (stopLoss + target) / 2;
                } else if (stopLoss) {
                  entryPrice = stopLoss + 20; // Estimate: stop + $20
                } else if (target) {
                  entryPrice = target - 20; // Estimate: target - $20
                } else {
                  console.log(`[ExecutionManager] ❌ REHYDRATE FALLBACK: Cannot estimate entry price for LONG position.`);
                  return null;
                }
              } else {
                // SHORT: target < entry < stop
                if (target && stopLoss) {
                  entryPrice = (target + stopLoss) / 2;
                } else if (stopLoss) {
                  entryPrice = stopLoss - 20; // Estimate: stop - $20
                } else if (target) {
                  entryPrice = target + 20; // Estimate: target + $20
                } else {
                  console.log(`[ExecutionManager] ❌ REHYDRATE FALLBACK: Cannot estimate entry price for SHORT position.`);
                  return null;
                }
              }

              console.log(`[ExecutionManager] 💰 REHYDRATE FALLBACK: Estimated entry price: ${entryPrice.toFixed(2)}`);
            }

            // If we're missing stop or target, calculate fallback levels
            const orderSide: 'buy' | 'sell' = positionSide === 'long' ? 'buy' : 'sell';
            const fallbackLevels = this.calculateBracketLevels(entryPrice, orderSide);

            if (!stopLoss) {
              stopLoss = fallbackLevels.stopLoss;
              console.log(`[ExecutionManager] ⚠️ REHYDRATE FALLBACK: Using calculated stop loss: ${stopLoss.toFixed(2)}`);
            }

            if (!target) {
              target = fallbackLevels.takeProfit;
              console.log(`[ExecutionManager] ⚠️ REHYDRATE FALLBACK: Using calculated take profit: ${target.toFixed(2)}`);
            }

            // Create position from reconstructed data
            const decision = tradingDB.recordDecision({
              symbol: this.symbol,
              marketState: 'balanced',
              location: 'at_poc',
              setupModel: null,
              decision: positionSide === 'long' ? 'BUY' : 'SELL',
              confidence: 50,
              entryPrice,
              stopLoss,
              target,
              riskPercent: 0.35,
              source: 'hybrid',
              reasoning: 'Rehydrated position from bracket orders after positions endpoint failed',
              cvd: 0,
              cvdTrend: 'neutral',
              currentPrice: entryPrice,
              buyAbsorption: 0,
              sellAbsorption: 0,
            });

            tradingDB.markDecisionFilled(decision.id, entryPrice);

            const activePosition: ActivePosition = {
              decisionId: decision.id,
              symbol: this.symbol,
              side: positionSide,
              entryPrice,
              entryTime: actualEntryTime || protectiveOrder.creationTimestamp || new Date().toISOString(),
              stopLoss,
              target,
              contracts: quantity,
              currentPrice: entryPrice,
              unrealizedPnL: 0,
              unrealizedPnLPercent: 0,
              stopOrderId,
              targetOrderId,
              usesNativeBracket: Boolean(stopOrderId && targetOrderId),
            };

            this.activePositions.set(decision.id, activePosition);

            // Set lastEntryOrderTime to prevent immediate grace period expiry
            this.lastEntryOrderTime = Date.now();

            console.log(`[ExecutionManager] ✅ REHYDRATE FALLBACK COMPLETE: Successfully reconstructed ${positionSide.toUpperCase()} position from bracket orders!`);
            console.log(`[ExecutionManager] 📊 REHYDRATE FALLBACK SUMMARY:
  • Contract: ${this.contractId}
  • Side: ${positionSide.toUpperCase()} (inferred from protective orders)
  • Quantity: ${quantity} contract(s)
  • Entry Price: ${entryPrice.toFixed(2)} (${actualEntryTime ? 'actual fill' : 'estimated'})
  • Entry Time: ${actualEntryTime || protectiveOrder.creationTimestamp || 'unknown'}
  • Stop Loss: ${stopLoss.toFixed(2)} ${stopOrderId ? `(Order ID: ${stopOrderId})` : '(calculated)'}
  • Target: ${target.toFixed(2)} ${targetOrderId ? `(Order ID: ${targetOrderId})` : '(calculated)'}
  • Native Bracket: ${activePosition.usesNativeBracket ? 'Yes' : 'Partial'}
  • Decision ID: ${decision.id}
  • Source: Bracket Order Reconstruction`
            );

            console.log(`[ExecutionManager] 🎉 REHYDRATE FALLBACK: Position is now active and ready for management!`);
            return activePosition;
          } else {
            console.log(`[ExecutionManager] ❌ REHYDRATE FALLBACK: No open orders found - position truly doesn't exist.`);
          }
        } catch (fallbackError: any) {
          console.error(`[ExecutionManager] ❌ REHYDRATE FALLBACK FAILED:`, fallbackError?.message || fallbackError);
        }

        console.log(`[ExecutionManager] 💡 REHYDRATE: This might be expected if:
  1. No position is currently open for this contract
  2. The position was closed before restart
  3. The REST API /api/Account/{accountId}/positions endpoint returned 404

  NOTE: If a position exists but wasn't detected, consider:
  - Using SignalR position subscriptions (userHub.invoke('SubscribePositions', accountId))
  - Checking if the position is tracked under a different contract ID
  - Verifying the REST API endpoint is correct for your broker`);
        return null;
      }

      console.log(`[ExecutionManager] ✅ REHYDRATE: Found matching position for ${this.contractId}`);
      console.log(`[ExecutionManager] 📝 REHYDRATE: Raw position data:`, JSON.stringify(matching, null, 2));

      const rawQty = this.extractNetQuantity(matching);
      console.log(`[ExecutionManager] 🔢 REHYDRATE: Extracted net quantity: ${rawQty}`);

      if (!rawQty) {
        console.log('[ExecutionManager] ⚠️ REHYDRATE: Broker position quantity is zero - nothing to rehydrate.');
        return null;
      }

      const side: 'long' | 'short' = rawQty > 0 ? 'long' : 'short';
      console.log(`[ExecutionManager] 📊 REHYDRATE: Position side determined: ${side.toUpperCase()} (${Math.abs(rawQty)} contracts)`);

      const entryPrice = this.extractEntryPrice(matching);
      console.log(`[ExecutionManager] 💰 REHYDRATE: Extracted entry price: ${entryPrice}`);

      if (!entryPrice || !Number.isFinite(entryPrice)) {
        console.warn('[ExecutionManager] ❌ REHYDRATE: Unable to determine entry price for broker position.');
        return null;
      }

      const entryTime = this.extractEntryTimestamp(matching);
      console.log(`[ExecutionManager] 🕐 REHYDRATE: Entry timestamp: ${entryTime || 'unknown'}`);

      const orderSide: 'buy' | 'sell' = side === 'long' ? 'buy' : 'sell';
      const fallbackLevels = this.calculateBracketLevels(entryPrice, orderSide);
      console.log(`[ExecutionManager] 🎯 REHYDRATE: Calculated fallback bracket levels: Stop=${fallbackLevels.stopLoss.toFixed(2)}, Target=${fallbackLevels.takeProfit.toFixed(2)}`);

      let stopLoss = fallbackLevels.stopLoss;
      let target = fallbackLevels.takeProfit;
      let stopOrderId: string | number | undefined;
      let targetOrderId: string | number | undefined;

      console.log(`[ExecutionManager] 🔍 REHYDRATE: Searching for existing bracket orders in open orders...`);

      try {
        const openOrdersResponse = await this.restClient.searchOpenOrders({ accountId: this.tradingAccount.id });
        console.log(`[ExecutionManager] 📋 REHYDRATE: Open orders API returned: success=${openOrdersResponse?.success}, orderCount=${openOrdersResponse?.orders?.length || 0}`);

        if (openOrdersResponse?.success && Array.isArray(openOrdersResponse.orders)) {
          const protectiveSide = this.getProtectiveSide({ side });
          const relevantOrders = openOrdersResponse.orders.filter(order => order.contractId === this.contractId && order.side === protectiveSide);
          console.log(`[ExecutionManager] 🔎 REHYDRATE: Found ${relevantOrders.length} relevant protective orders (${protectiveSide}) for ${this.contractId}`);

          const stopCandidates = relevantOrders.filter(order => typeof order.stopPrice === 'number' && Number.isFinite(order.stopPrice as number));
          console.log(`[ExecutionManager] 🛑 REHYDRATE: Found ${stopCandidates.length} stop loss candidate(s)`);

          const directionalStop = stopCandidates.filter(order => {
            const price = Number(order.stopPrice);
            return side === 'long' ? price < entryPrice : price > entryPrice;
          });
          console.log(`[ExecutionManager] 🛑 REHYDRATE: Filtered to ${directionalStop.length} directionally correct stop order(s)`);

          const stopOrderRecord = this.pickClosestOrder(
            directionalStop.length ? directionalStop : stopCandidates,
            fallbackLevels.stopLoss,
            'stopPrice'
          );
          if (stopOrderRecord && typeof stopOrderRecord.stopPrice === 'number') {
            stopLoss = Number(stopOrderRecord.stopPrice);
            stopOrderId = stopOrderRecord.id;
            console.log(`[ExecutionManager] ✅ REHYDRATE: Matched stop loss order (ID: ${stopOrderId}) at ${stopLoss.toFixed(2)}`);
          } else {
            console.log(`[ExecutionManager] ⚠️ REHYDRATE: No stop loss order found, will use fallback: ${stopLoss.toFixed(2)}`);
          }

          const targetCandidates = relevantOrders.filter(order => typeof order.limitPrice === 'number' && Number.isFinite(order.limitPrice as number));
          console.log(`[ExecutionManager] 🎯 REHYDRATE: Found ${targetCandidates.length} target/limit candidate(s)`);

          const directionalTarget = targetCandidates.filter(order => {
            const price = Number(order.limitPrice);
            return side === 'long' ? price > entryPrice : price < entryPrice;
          });
          console.log(`[ExecutionManager] 🎯 REHYDRATE: Filtered to ${directionalTarget.length} directionally correct target order(s)`);

          const targetOrderRecord = this.pickClosestOrder(
            directionalTarget.length ? directionalTarget : targetCandidates,
            fallbackLevels.takeProfit,
            'limitPrice'
          );
          if (targetOrderRecord && typeof targetOrderRecord.limitPrice === 'number') {
            target = Number(targetOrderRecord.limitPrice);
            targetOrderId = targetOrderRecord.id;
            console.log(`[ExecutionManager] ✅ REHYDRATE: Matched target order (ID: ${targetOrderId}) at ${target.toFixed(2)}`);
          } else {
            console.log(`[ExecutionManager] ⚠️ REHYDRATE: No target order found, will use fallback: ${target.toFixed(2)}`);
          }
        }
      } catch (error: any) {
        console.warn('[ExecutionManager] ⚠️ REHYDRATE: Unable to inspect open orders during rehydrate:', error?.message || error);
        console.log(`[ExecutionManager] 💡 REHYDRATE: Will proceed with fallback bracket levels`);
      }

      const decision = tradingDB.recordDecision({
        symbol: this.symbol,
        marketState: 'balanced',
        location: 'at_poc',
        setupModel: null,
        decision: side === 'long' ? 'BUY' : 'SELL',
        confidence: 50,
        entryPrice,
        stopLoss,
        target,
        riskPercent: 0.35,
        source: 'hybrid',
        reasoning: 'Rehydrated broker position after restart',
        cvd: 0,
        cvdTrend: 'neutral',
        currentPrice: entryPrice,
        buyAbsorption: 0,
        sellAbsorption: 0,
      });

      tradingDB.markDecisionFilled(decision.id, entryPrice);

      const activePosition: ActivePosition = {
        decisionId: decision.id,
        symbol: this.symbol,
        side,
        entryPrice,
        entryTime,
        stopLoss,
        target,
        contracts: Math.abs(rawQty),
        currentPrice: entryPrice,
        unrealizedPnL: 0,
        unrealizedPnLPercent: 0,
        stopOrderId,
        targetOrderId,
        usesNativeBracket: Boolean(stopOrderId && targetOrderId),
      };

      this.activePositions.set(decision.id, activePosition);

      // Set lastEntryOrderTime to prevent immediate grace period expiry
      this.lastEntryOrderTime = Date.now();

      console.log(
        `[ExecutionManager] ✅ REHYDRATE COMPLETE: Successfully rehydrated ${side.toUpperCase()} position!`
      );
      console.log(`[ExecutionManager] 📊 REHYDRATE SUMMARY:
  • Contract: ${this.contractId}
  • Side: ${side.toUpperCase()}
  • Quantity: ${Math.abs(rawQty)} contract(s)
  • Entry Price: ${entryPrice.toFixed(2)}
  • Stop Loss: ${stopLoss.toFixed(2)} ${stopOrderId ? `(Order ID: ${stopOrderId})` : '(fallback)'}
  • Target: ${target.toFixed(2)} ${targetOrderId ? `(Order ID: ${targetOrderId})` : '(fallback)'}
  • Native Bracket: ${activePosition.usesNativeBracket ? 'Yes' : 'No'}
  • Decision ID: ${decision.id}
  • Entry Time: ${entryTime || 'N/A'}`
      );

      if (!stopOrderId || !targetOrderId) {
        console.log(`[ExecutionManager] ⚠️ REHYDRATE: Missing ${!stopOrderId && !targetOrderId ? 'both stop and target' : !stopOrderId ? 'stop' : 'target'} order ID(s), will ensure bracket orders...`);
        await this.ensureBracketOrderIds(activePosition);
      }

      console.log(`[ExecutionManager] 🎉 REHYDRATE: Position is now active and ready for management!`);
      return activePosition;
    } catch (error: any) {
      console.error('[ExecutionManager] ❌ REHYDRATE FAILED:', error?.message || error);
      console.error('[ExecutionManager] Stack trace:', error?.stack);
      return null;
    }
  }

  /**
   * Submit bracket orders (stop loss + take profit)
   * Submits protective orders immediately after entry fill
   */
  private async submitBracketOrders(
    entryOrder: ExecutedOrder,
    stopLoss: number,
    takeProfit: number
  ): Promise<{ stopOrderId?: string | number; targetOrderId?: string | number }> {
    if (!this.tradingAccount) {
      return {};
    }

    const oppositeSide: OrderSide = entryOrder.side === 'buy' ? 'sell' : 'buy';

    try {
      // Submit stop loss order (stop market order)
      console.log(`[ExecutionManager] Submitting STOP LOSS order @ ${stopLoss.toFixed(2)}...`);

      const stopLossOrder = {
        contractId: this.contractId,
        accountId: this.tradingAccount.id,
        side: oppositeSide,
        quantity: entryOrder.quantity,
        orderType: 'market' as const, // Will become market order when stop is hit
        stopPrice: stopLoss,
        live: this.live,
      };

      const stopResult = await submitTopstepXOrder(stopLossOrder);

      let stopOrderId: string | number | undefined;
      let targetOrderId: string | number | undefined;

      if (stopResult.success) {
        console.log(`[ExecutionManager] ✅ Stop loss order placed @ ${stopLoss.toFixed(2)}`);
        stopOrderId = stopResult.orderId ?? stopResult.id;
      } else {
        console.error(`[ExecutionManager] ⚠️ Stop loss order failed: ${stopResult.errorMessage}`);
      }

      // Submit take profit order (limit order)
      console.log(`[ExecutionManager] Submitting TAKE PROFIT order @ ${takeProfit.toFixed(2)}...`);

      const takeProfitOrder = {
        contractId: this.contractId,
        accountId: this.tradingAccount.id,
        side: oppositeSide,
        quantity: entryOrder.quantity,
        orderType: 'limit' as const,
        limitPrice: takeProfit,
        live: this.live,
      };

      const profitResult = await submitTopstepXOrder(takeProfitOrder);

      if (profitResult.success) {
        console.log(`[ExecutionManager] ✅ Take profit order placed @ ${takeProfit.toFixed(2)}`);
        targetOrderId = profitResult.orderId ?? profitResult.id;
      } else {
        console.error(`[ExecutionManager] ⚠️ Take profit order failed: ${profitResult.errorMessage}`);
      }

      console.log(`[ExecutionManager] 📊 Bracket Orders Summary:`);
      console.log(`  Entry: ${entryOrder.executedPrice.toFixed(2)}`);
      console.log(`  Stop Loss: ${stopLoss.toFixed(2)} (Risk: ${Math.abs(entryOrder.executedPrice - stopLoss).toFixed(2)} points)`);
      console.log(`  Take Profit: ${takeProfit.toFixed(2)} (Reward: ${Math.abs(takeProfit - entryOrder.executedPrice).toFixed(2)} points)`);

      return { stopOrderId, targetOrderId };
    } catch (error: any) {
      console.error('[ExecutionManager] Failed to submit bracket orders:', error.message);
      return {};
    }
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

    // CRITICAL: For positions with native brackets, DO NOT simulate exits based on price!
    // The broker will auto-fill the brackets and checkBrokerPosition() will detect the closure.
    // Price-based simulation causes phantom positions where agent thinks it's flat but broker has active brackets.
    if (position.usesNativeBracket) {
      console.log('[ExecutionManager] ⚠️ Position uses native brackets - exit detection handled by broker, not price simulation');
      return null;
    }

    // For manual bracket management (legacy), simulate exits based on price
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

    const normalizeReason = reason.toLowerCase();

    // Determine if brackets will auto-cancel (only for broker-filled stops/targets)
    const bracketShouldAutoCancel = position.usesNativeBracket
      && (normalizeReason === 'target_hit' || normalizeReason === 'stop_loss_hit');

    // For manual/risk management closes, ALWAYS close the actual position first
    const isManualClose = normalizeReason === 'manual'
      || normalizeReason === 'risk_management_close'
      || normalizeReason === 'immediate_risk_management_close';

    if (isManualClose && this.restClient && this.tradingAccount) {
      // CRITICAL: Check if position actually exists at broker before closing (EXIT order validation)
      console.log(`[ExecutionManager] 🔍 [EXIT ORDER VALIDATION] Checking broker position status...`);

      const brokerHasPosition = await this.refreshBrokerPositionStatus();

      if (!brokerHasPosition) {
        console.warn(`[ExecutionManager] 🚫 [EXIT ORDER BLOCKED] Position already FLAT at broker - skipping close order to prevent naked market order`);
        // Position already closed at broker, just clean up local state
        this.activePositions.delete(decisionId);
        console.log(`[ExecutionManager] 🧹 Cleaned up local position tracking`);

        // Still cancel any orphaned brackets
        console.log(`[ExecutionManager] 🗑️ Cancelling any orphaned bracket orders...`);
        await this.forceCancelBrackets(position);

        // Start grace period to prevent immediate re-entry
        this.lastEntryOrderTime = Date.now();
        console.log(`[ExecutionManager] 🚫 Grace period started: No new entries for 90s to prevent duplicates`);

        return decisionId;
      }

      console.log(`[ExecutionManager] ✅ [EXIT ORDER APPROVED] Position exists at broker - proceeding with close`);

      // Place market order to close the position
      console.log(`[ExecutionManager] ⚠️ Placing MARKET order to close ${position.side.toUpperCase()} position...`);

      try {
        const closeSide = position.side === 'long' ? 1 : 0; // Opposite side: long → sell (1), short → buy (0)
        const closeResponse = await this.restClient.placeOrder({
          accountId: this.tradingAccount.id,
          contractId: this.contractId,
          side: closeSide,
          size: position.contracts,
          type: 2, // Market order
          timeInForce: 3, // IOC (Immediate or Cancel) - close immediately
        });

        if (closeResponse?.success === false) {
          console.error('[ExecutionManager] ❌ Failed to close position:', closeResponse.errorMessage);
          // Don't cancel brackets if we couldn't close the position
          return null;
        }

        console.log(`[ExecutionManager] ✅ Position closed at market (Order ID: ${closeResponse?.orderId})`);

        // Small delay to ensure the close order is processed before cancelling brackets
        await new Promise(resolve => setTimeout(resolve, 500));

      } catch (error: any) {
        console.error('[ExecutionManager] ❌ Error closing position:', error.message);
        // Don't cancel brackets if we couldn't close the position
        return null;
      }
    }

    // Now cancel the brackets (after position is closed or if auto-cancel)
    if (!bracketShouldAutoCancel || isManualClose) {
      console.log(`[ExecutionManager] Cancelling brackets for ${normalizeReason} close...`);
      await this.cancelProtectiveOrders(position, true); // Force cancel
    }

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

    // CRITICAL: Update lastEntryOrderTime to enforce grace period after position close
    // This prevents immediate re-entry after stop loss or manual close
    this.lastEntryOrderTime = Date.now();
    console.log(`[ExecutionManager] 🚫 Grace period started: No new entries for ${this.entryOrderGracePeriodMs / 1000}s to prevent duplicates`);

    return decisionId;
  }

  /**
   * Cancel protective orders (stop loss and take profit) with retry logic
   * Enhanced to ensure brackets are actually removed to prevent orphaned orders
   */
  private async cancelProtectiveOrders(position: ActivePosition, forceCancel: boolean = false): Promise<void> {
    if (!this.restClient || !this.tradingAccount) {
      console.warn('[ExecutionManager] ⚠️ Cannot cancel protective orders - no REST client or account');
      return;
    }

    // Skip auto-cancel only if NOT forced and using native brackets
    if (position.usesNativeBracket && !forceCancel) {
      console.log('[ExecutionManager] Skipping protective cancel – native bracket should auto-OCO on broker.');
      return;
    }

    console.log(`[ExecutionManager] 🗑️ ${forceCancel ? 'FORCE ' : ''}Cancelling protective orders to prevent orphaned brackets...`);

    const cancelIds = [
      { id: position.stopOrderId, type: 'STOP' },
      { id: position.targetOrderId, type: 'TARGET' }
    ].filter(item => item.id !== undefined && item.id !== null);

    if (cancelIds.length === 0) {
      console.log('[ExecutionManager] ℹ️ No bracket order IDs found - position may already be flat');
      return;
    }

    console.log(`[ExecutionManager] 📋 Found ${cancelIds.length} bracket(s) to cancel: ${cancelIds.map(i => `${i.type}=${i.id}`).join(', ')}`);

    let canceledCount = 0;
    let failedCount = 0;

    for (const { id, type } of cancelIds) {
      const canceled = await this.cancelOrderWithRetry(id!, type, 3);

      if (canceled) {
        canceledCount++;
        // Clear the order ID from position
        if (type === 'STOP') {
          position.stopOrderId = undefined;
        } else if (type === 'TARGET') {
          position.targetOrderId = undefined;
        }
        console.log(`[ExecutionManager] ✅ ${type} order ${id} canceled successfully`);
      } else {
        failedCount++;
        console.error(`[ExecutionManager] ❌ ${type} order ${id} cancellation FAILED after retries`);
      }
    }

    // Final verification
    if (failedCount > 0) {
      console.error(`[ExecutionManager] ⚠️ WARNING: ${failedCount}/${cancelIds.length} brackets failed to cancel! Check for orphaned orders.`);

      // Try one more verification check
      await this.verifyBracketsCanceled(position);
    } else {
      console.log(`[ExecutionManager] ✅ All ${canceledCount} protective orders canceled successfully - no orphaned brackets`);
    }
  }

  /**
   * Cancel a single order with retry logic
   */
  private async cancelOrderWithRetry(
    orderId: string | number,
    orderType: string,
    maxRetries: number = 3
  ): Promise<boolean> {
    if (!this.restClient || !this.tradingAccount) {
      return false;
    }

    for (let attempt = 1; attempt <= maxRetries; attempt++) {
      try {
        console.log(`[ExecutionManager] 🔄 Attempt ${attempt}/${maxRetries} to cancel ${orderType} order ${orderId}`);

        const result = await this.restClient.cancelOrder({
          accountId: this.tradingAccount.id,
          orderId: String(orderId),
        });

        // Check if cancellation succeeded
        if (result?.success !== false) {
          console.log(`[ExecutionManager] ✅ ${orderType} cancellation succeeded on attempt ${attempt}`);
          return true;
        }

        // Check if order was already canceled/filled (common legitimate case)
        const errorMsg = result?.errorMessage?.toLowerCase() || '';
        if (errorMsg.includes('not found') || errorMsg.includes('already') || errorMsg.includes('closed')) {
          console.log(`[ExecutionManager] ℹ️ ${orderType} order ${orderId} already canceled/filled - this is OK`);
          return true; // Not an error - order is already gone
        }

        console.warn(`[ExecutionManager] ⚠️ ${orderType} cancellation attempt ${attempt} rejected:`, result?.errorMessage || 'Unknown error');

        // Wait before retry (exponential backoff)
        if (attempt < maxRetries) {
          const delayMs = 500 * Math.pow(2, attempt - 1); // 500ms, 1s, 2s
          console.log(`[ExecutionManager] ⏳ Waiting ${delayMs}ms before retry...`);
          await new Promise(resolve => setTimeout(resolve, delayMs));
        }
      } catch (error: any) {
        const errorMsg = error?.message?.toLowerCase() || '';

        // Check if error indicates order already canceled/filled
        if (errorMsg.includes('not found') || errorMsg.includes('already') || errorMsg.includes('closed')) {
          console.log(`[ExecutionManager] ℹ️ ${orderType} order ${orderId} already canceled/filled (caught in exception) - this is OK`);
          return true;
        }

        console.error(`[ExecutionManager] ❌ ${orderType} cancellation attempt ${attempt} threw error:`, error?.message || error);

        if (attempt < maxRetries) {
          const delayMs = 500 * Math.pow(2, attempt - 1);
          await new Promise(resolve => setTimeout(resolve, delayMs));
        }
      }
    }

    console.error(`[ExecutionManager] ❌ ${orderType} order ${orderId} cancellation failed after ${maxRetries} attempts`);
    return false;
  }

  /**
   * Verify that bracket orders were actually canceled by checking open orders
   */
  private async verifyBracketsCanceled(position: ActivePosition): Promise<void> {
    if (!this.restClient || !this.tradingAccount) {
      return;
    }

    try {
      console.log('[ExecutionManager] 🔍 Verifying brackets are canceled by checking open orders...');

      const response = await this.restClient.searchOpenOrders({
        accountId: this.tradingAccount.id
      });

      if (!response?.success || !response?.orders) {
        console.warn('[ExecutionManager] ⚠️ Could not verify bracket cancellation - API call failed');
        return;
      }

      const openOrders = response.orders || [];
      const relevantOrders = openOrders.filter(order =>
        order.contractId === this.contractId &&
        (order.id === position.stopOrderId || order.id === position.targetOrderId)
      );

      if (relevantOrders.length > 0) {
        console.error('[ExecutionManager] 🚨 CRITICAL: Found orphaned bracket orders still active!');
        relevantOrders.forEach(order => {
          console.error(`[ExecutionManager] 🚨 Orphaned order: ID=${order.id}, Type=${order.type}, Price=${order.limitPrice || order.stopPrice}`);
        });

        // Force cancel them one more time
        for (const order of relevantOrders) {
          console.log(`[ExecutionManager] 🔧 Attempting emergency cancel of orphaned order ${order.id}...`);
          try {
            await this.restClient.cancelOrder({
              accountId: this.tradingAccount.id,
              orderId: String(order.id),
            });
            console.log(`[ExecutionManager] ✅ Emergency cancel succeeded for ${order.id}`);
          } catch (error: any) {
            console.error(`[ExecutionManager] ❌ Emergency cancel failed for ${order.id}:`, error?.message);
          }
        }
      } else {
        console.log('[ExecutionManager] ✅ Verification complete: No orphaned brackets found');
      }
    } catch (error: any) {
      console.warn('[ExecutionManager] ⚠️ Bracket verification check failed:', error?.message || error);
    }
  }

  async syncWithBrokerState(): Promise<void> {
    if (!this.restClient || !this.tradingAccount) {
      return;
    }

    const now = Date.now();
    if (now - this.lastBrokerSync < this.brokerSyncIntervalMs) {
      return;
    }
    this.lastBrokerSync = now;

    try {
      const [positions, openOrders] = await Promise.all([
        this.restClient.getPositions(this.tradingAccount.id),
        this.restClient.searchOpenOrders({ accountId: this.tradingAccount.id }),
      ]);

      const openOrderList = openOrders?.orders || [];
      const brokerPosition = (positions || []).find(pos => this.extractNetQuantity(pos) !== 0 && pos.contractId === this.contractId);
      const active = this.getActivePosition();

      if (!brokerPosition) {
        if (active) {
          const timeSinceEntry = now - this.lastEntryOrderTime;
          const inGracePeriod = timeSinceEntry < this.entryOrderGracePeriodMs;

          // FIRST: Check for matching protective orders (regardless of grace period)
          const protectiveSide = this.getProtectiveSide(active);
          const protectiveOrders = openOrderList.filter(order =>
            order.contractId === this.contractId && order.side === protectiveSide
          );

          console.log(`[ExecutionManager] 🔍 Broker flat but checking for protective orders... Found ${protectiveOrders.length} protective order(s)`);

          if (protectiveOrders.length > 0) {
            // Check if these orders match our expected stop/target levels
            const stopOrder = this.pickClosestOrder(protectiveOrders, active.stopLoss, 'stopPrice');
            const targetOrder = this.pickClosestOrder(protectiveOrders, active.target, 'limitPrice');

            console.log(`[ExecutionManager] 🔍 Matching check: stop=${stopOrder?.id || 'none'} (expected ${active.stopLoss}), target=${targetOrder?.id || 'none'} (expected ${active.target})`);

            // CRITICAL FIX: If we have at least 1-2 protective orders, PRESERVE the position
            // Even if they don't match perfectly, they're protecting the position
            // Only require matching if we have exactly 2 orders, otherwise be lenient
            if (stopOrder || targetOrder || protectiveOrders.length >= 1) {
              console.log(`[ExecutionManager] ✅ POSITION PRESERVED: Found ${protectiveOrders.length} protective order(s). TopStepX positions API is lagging but brackets exist and protect position.`);
              // Update the bracket IDs if found
              if (stopOrder) active.stopOrderId = stopOrder.id;
              if (targetOrder) active.targetOrderId = targetOrder.id;
              // PERSIST POSITION INDEFINITELY - protective orders exist
              return;
            }

            // If we somehow get here with protective orders but none matched (shouldn't happen)
            if (inGracePeriod) {
              console.log(`[ExecutionManager] Within grace period (${Math.round(timeSinceEntry / 1000)}s / ${this.entryOrderGracePeriodMs / 1000}s) and found ${protectiveOrders.length} protective orders. Waiting...`);
              return;
            }

            // ONLY cancel if we're absolutely sure these are orphans (very rare case)
            console.log(`[ExecutionManager] ⚠️ WARNING: Broker flat after grace period, found ${protectiveOrders.length} protective order(s) that couldn't be matched. This is RARE - manual review recommended.`);
            // DO NOT automatically cancel - let user review
            console.log(`[ExecutionManager] 🛡️ SAFETY: Keeping protective orders to prevent naked position. Order IDs: ${protectiveOrders.map(o => o.id).join(', ')}`);
            return; // Keep position and orders for safety
          } else {
            // No protective orders at all
            if (inGracePeriod) {
              console.log(`[ExecutionManager] Broker flat but within entry grace period (${Math.round(timeSinceEntry / 1000)}s / ${this.entryOrderGracePeriodMs / 1000}s). Waiting for position and brackets to propagate...`);
              return;
            }
          }

          console.log('[ExecutionManager] ❌ Broker reports FLAT, no matching brackets found. Clearing local position state.');
          this.activePositions.delete(active.decisionId);

          // CRITICAL: Update lastEntryOrderTime to enforce grace period after broker close
          // This prevents immediate re-entry when stop loss was hit
          this.lastEntryOrderTime = now;
          console.log(`[ExecutionManager] 🚫 Grace period started: No new entries for ${this.entryOrderGracePeriodMs / 1000}s to prevent duplicates`);
        }
        return;
      }

      if (!active) {
        await this.rehydrateActivePosition();
        return;
      }

      const brokerQty = Math.abs(this.extractNetQuantity(brokerPosition));
      active.contracts = brokerQty || active.contracts;
      const protectiveSide = this.getProtectiveSide(active);
      const relevantOrders = openOrderList.filter(order => order.contractId === this.contractId && order.side === protectiveSide);

      const locateOrderById = (id?: string | number) => {
        if (!id) return undefined;
        return relevantOrders.find(order => String(order.id) === String(id));
      };

      const stopRecord = locateOrderById(active.stopOrderId);
      if (!stopRecord) {
        if (active.stopOrderId) {
          console.log('[ExecutionManager] Stop order missing on broker. Clearing local reference.');
          active.stopOrderId = undefined;
        }
      } else if (typeof stopRecord.stopPrice === 'number') {
        active.stopLoss = stopRecord.stopPrice;
      }

      const targetRecord = locateOrderById(active.targetOrderId);
      if (!targetRecord) {
        if (active.targetOrderId) {
          console.log('[ExecutionManager] Target order missing on broker. Clearing local reference.');
          active.targetOrderId = undefined;
        }
      } else if (typeof targetRecord.limitPrice === 'number') {
        active.target = targetRecord.limitPrice;
      }
    } catch (error: any) {
      console.warn('[ExecutionManager] Failed to sync broker state:', error?.message || error);
    }
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
   * Dynamically adjust protective orders via TopstepX modify API
   * Enhanced with detailed logging, retry logic, and cancel-replace fallback
   */
  async adjustActiveProtection(newStop?: number | null, newTarget?: number | null): Promise<boolean> {
    // Hard guard: if broker shows flat or no protective orders remain, skip adjustments
    const brokerPositionCheck = await this.rehydrateActivePosition();
    if (!brokerPositionCheck) {
      console.warn('[ExecutionManager] 🚫 adjustActiveProtection aborted: broker reports flat / no active position.');
      return false;
    }

    const position = this.getActivePosition();
    if (!position) {
      console.warn('[ExecutionManager] 🚫 adjustActiveProtection: No active position found');
      return false;
    }

    const hasNewStop = typeof newStop === 'number' && Number.isFinite(newStop);
    const hasNewTarget = typeof newTarget === 'number' && Number.isFinite(newTarget);

    console.log(`[ExecutionManager] 🛡️ adjustActiveProtection called:`, {
      currentStop: position.stopLoss,
      currentTarget: position.target,
      requestedStop: hasNewStop ? (newStop as number).toFixed(2) : 'none',
      requestedTarget: hasNewTarget ? (newTarget as number).toFixed(2) : 'none',
      stopOrderId: position.stopOrderId || 'MISSING',
      targetOrderId: position.targetOrderId || 'MISSING',
    });

    const stopNeedsUpdate = hasNewStop && Math.abs((newStop as number) - position.stopLoss) >= this.tickSize / 2;
    const targetNeedsUpdate = hasNewTarget && Math.abs((newTarget as number) - position.target) >= this.tickSize / 2;

    if (!stopNeedsUpdate && !targetNeedsUpdate) {
      console.log('[ExecutionManager] ✅ No updates needed (prices within tolerance)');
      return false;
    }

    const hasRestContext = await this.ensureRestContext();
    if (!hasRestContext || !this.restClient || !this.tradingAccount) {
      console.error('[ExecutionManager] ❌ Cannot adjust protective orders without REST client or trading account context.');
      return false;
    }

    // Try to find bracket order IDs if missing
    console.log('[ExecutionManager] 🔍 Ensuring bracket order IDs are available...');
    await this.ensureBracketOrderIds(position);

    // Validate we have order IDs after sync attempt
    if ((stopNeedsUpdate && !position.stopOrderId) || (targetNeedsUpdate && !position.targetOrderId)) {
      console.error('[ExecutionManager] ❌ CRITICAL: Missing bracket order IDs after sync attempt!', {
        stopOrderId: position.stopOrderId || 'MISSING',
        targetOrderId: position.targetOrderId || 'MISSING',
        stopNeeded: stopNeedsUpdate,
        targetNeeded: targetNeedsUpdate,
      });

      // Check broker open orders; if none, avoid naked state by flattening
      const brokerFlat = await this.verifyPositionFlattened();
      if (brokerFlat) {
        console.warn('[ExecutionManager] 🛑 Broker reports flat and no bracket IDs present — skipping adjustments.');
        return false;
      }

      // Try cancel-and-replace fallback
      console.log('[ExecutionManager] 🔄 Attempting cancel-and-replace fallback...');
      const replaced = await this.cancelAndReplaceBrackets(position, newStop, newTarget);
      if (!replaced) {
        console.error('[ExecutionManager] ❌ Cancel-and-replace fallback failed; cannot risk naked position.');
        return false;
      }
      return true;
    }

    let updated = false;

    // Modify stop loss with retry logic
    if (stopNeedsUpdate) {
      let normalizedStop = this.normalizePrice(newStop as number);

      // Validate stop against current market price to prevent broker rejection
      const currentPrice = position.currentPrice;
      const minStopDistance = this.tickSize * 4; // Minimum 4 ticks away from market
      let skipStopUpdate = false;

      if (position.side === 'short') {
        // For SHORT: stop-sell order must be ABOVE current ask price
        const minValidStop = currentPrice + minStopDistance;
        if (normalizedStop < minValidStop) {
          // Check if adjusting would worsen our stop (move it higher/looser)
          if (minValidStop > position.stopLoss) {
            console.log(`[ExecutionManager] ⚠️ SHORT stop adjustment skipped: requested ${normalizedStop.toFixed(2)} too close to market ${currentPrice.toFixed(2)}, would need ${minValidStop.toFixed(2)} but current ${position.stopLoss.toFixed(2)} is better. Keeping existing stop.`);
            skipStopUpdate = true;
          } else {
            console.log(`[ExecutionManager] ⚠️ SHORT stop ${normalizedStop.toFixed(2)} too close to market ${currentPrice.toFixed(2)}, adjusting to valid ${minValidStop.toFixed(2)}`);
            normalizedStop = this.normalizePrice(minValidStop);
          }
        }
      } else {
        // For LONG: stop-buy order must be BELOW current bid price
        const maxValidStop = currentPrice - minStopDistance;
        if (normalizedStop > maxValidStop) {
          // Check if adjusting would worsen our stop (move it lower/looser)
          if (maxValidStop < position.stopLoss) {
            console.log(`[ExecutionManager] ⚠️ LONG stop adjustment skipped: requested ${normalizedStop.toFixed(2)} too close to market ${currentPrice.toFixed(2)}, would need ${maxValidStop.toFixed(2)} but current ${position.stopLoss.toFixed(2)} is better. Keeping existing stop.`);
            skipStopUpdate = true;
          } else {
            console.log(`[ExecutionManager] ⚠️ LONG stop ${normalizedStop.toFixed(2)} too close to market ${currentPrice.toFixed(2)}, adjusting to valid ${maxValidStop.toFixed(2)}`);
            normalizedStop = this.normalizePrice(maxValidStop);
          }
        }
      }

      if (!skipStopUpdate) {
        console.log(`[ExecutionManager] 🎯 Attempting to modify STOP: ${position.stopLoss.toFixed(2)} -> ${normalizedStop.toFixed(2)} (OrderID: ${position.stopOrderId})`);

        const stopModified = await this.modifyOrderWithRetry(
          position.stopOrderId!,
          { stopPrice: normalizedStop },
          'STOP',
          3 // 3 retries
        );

        if (stopModified) {
          position.stopLoss = normalizedStop;
          updated = true;
          console.log(`[ExecutionManager] ✅ Stop loss successfully updated to ${normalizedStop.toFixed(2)}`);
        } else {
          console.error(`[ExecutionManager] ❌ Stop modification failed after retries - attempting cancel-and-replace...`);
          // Try cancel-and-replace for this specific order
          const replaced = await this.cancelAndReplaceStop(position, normalizedStop);
          if (replaced) {
            updated = true;
            console.log(`[ExecutionManager] ✅ Stop loss replaced via cancel-and-replace at ${normalizedStop.toFixed(2)}`);
          }
        }
      }
    }

    // Modify target with retry logic
    if (targetNeedsUpdate) {
      const normalizedTarget = this.normalizePrice(newTarget as number);
      console.log(`[ExecutionManager] 🎯 Attempting to modify TARGET: ${position.target.toFixed(2)} -> ${normalizedTarget.toFixed(2)} (OrderID: ${position.targetOrderId})`);

      const targetModified = await this.modifyOrderWithRetry(
        position.targetOrderId!,
        { limitPrice: normalizedTarget },
        'TARGET',
        3 // 3 retries
      );

      if (targetModified) {
        position.target = normalizedTarget;
        updated = true;
        console.log(`[ExecutionManager] ✅ Target successfully updated to ${normalizedTarget.toFixed(2)}`);
      } else {
        console.error(`[ExecutionManager] ❌ Target modification failed after retries - attempting cancel-and-replace...`);
        // Try cancel-and-replace for this specific order
        const replaced = await this.cancelAndReplaceTarget(position, normalizedTarget);
        if (replaced) {
          updated = true;
          console.log(`[ExecutionManager] ✅ Target replaced via cancel-and-replace at ${normalizedTarget.toFixed(2)}`);
        }
      }
    }

    if (updated) {
      console.log(`[ExecutionManager] ✅ Bracket adjustment completed successfully`);
    } else {
      console.error(`[ExecutionManager] ❌ Bracket adjustment FAILED - no updates were applied`);
    }

    return updated;
  }

  /**
   * Helper: Modify order with retry logic
   */
  private async modifyOrderWithRetry(
    orderId: string | number,
    params: { stopPrice?: number; limitPrice?: number },
    orderType: 'STOP' | 'TARGET',
    maxRetries: number = 3
  ): Promise<boolean> {
    if (!this.restClient || !this.tradingAccount) {
      return false;
    }

    for (let attempt = 1; attempt <= maxRetries; attempt++) {
      try {
        console.log(`[ExecutionManager] 🔄 Attempt ${attempt}/${maxRetries} to modify ${orderType} order ${orderId}`);

        const result = await this.restClient.modifyOrder({
          accountId: this.tradingAccount.id,
          orderId,
          ...params,
        });

        if (result?.success !== false) {
          console.log(`[ExecutionManager] ✅ ${orderType} modification succeeded on attempt ${attempt}`);
          return true;
        }

        console.warn(`[ExecutionManager] ⚠️ ${orderType} modification attempt ${attempt} rejected:`, result?.errorMessage || 'Unknown error');

        // Wait before retry (exponential backoff)
        if (attempt < maxRetries) {
          const delayMs = 1000 * Math.pow(2, attempt - 1); // 1s, 2s, 4s
          console.log(`[ExecutionManager] ⏳ Waiting ${delayMs}ms before retry...`);
          await new Promise(resolve => setTimeout(resolve, delayMs));
        }
      } catch (error: any) {
        console.error(`[ExecutionManager] ❌ ${orderType} modification attempt ${attempt} threw error:`, error?.message || error);

        if (attempt < maxRetries) {
          const delayMs = 1000 * Math.pow(2, attempt - 1);
          await new Promise(resolve => setTimeout(resolve, delayMs));
        }
      }
    }

    console.error(`[ExecutionManager] ❌ ${orderType} modification failed after ${maxRetries} attempts`);
    return false;
  }

  /**
   * Fallback: Cancel and replace both brackets
   */
  private async cancelAndReplaceBrackets(
    position: ActivePosition,
    newStop?: number | null,
    newTarget?: number | null
  ): Promise<boolean> {
    console.log('[ExecutionManager] 🔄 Executing cancel-and-replace for brackets...');

    const stopReplaced = newStop ? await this.cancelAndReplaceStop(position, newStop) : true;
    const targetReplaced = newTarget ? await this.cancelAndReplaceTarget(position, newTarget) : true;

    return stopReplaced && targetReplaced;
  }

  /**
   * Cancel and replace stop loss order
   */
  private async cancelAndReplaceStop(position: ActivePosition, newStop: number): Promise<boolean> {
    if (!this.restClient || !this.tradingAccount) {
      return false;
    }

    try {
      const previousStopId = position.stopOrderId;

      // Place new stop order BEFORE canceling the old one to avoid naked exposure
      const oppositeSide = position.side === 'long' ? 1 : 0; // Opposite side for protective order
      console.log(`[ExecutionManager] 📝 Placing new stop order @ ${newStop.toFixed(2)}`);

      const result = await this.restClient.placeOrder({
        accountId: this.tradingAccount.id,
        contractId: this.contractId,
        side: oppositeSide,
        size: position.contracts,
        type: 4, // Stop market
        stopPrice: newStop,
        timeInForce: 1, // GTC
      });

      if (result?.success !== false && result?.orderId) {
        position.stopOrderId = String(result.orderId);
        position.stopLoss = newStop;
        console.log(`[ExecutionManager] ✅ New stop order placed: ${result.orderId}`);

        // Now cancel the old stop (if it exists) to avoid duplicates, but only after the new one is live
        if (previousStopId && String(previousStopId) !== String(result.orderId)) {
          try {
            console.log(`[ExecutionManager] 🗑️ Canceling previous stop order ${previousStopId} after successful replacement`);
            await this.restClient.cancelOrder({
              accountId: this.tradingAccount.id,
              orderId: String(previousStopId),
            });
          } catch (cancelError: any) {
            console.warn('[ExecutionManager] ⚠️ Failed to cancel previous stop after replacement - leaving both stops active:', cancelError?.message || cancelError);
          }
        }
        return true;
      }

      console.error('[ExecutionManager] ❌ Failed to place replacement stop:', result?.errorMessage);
      // Keep existing stop intact if placement failed
      if (previousStopId) {
        position.stopOrderId = previousStopId;
      }
      return false;
    } catch (error: any) {
      console.error('[ExecutionManager] ❌ Cancel-and-replace stop failed:', error?.message || error);
      return false;
    }
  }

  /**
   * Cancel and replace target order
   */
  private async cancelAndReplaceTarget(position: ActivePosition, newTarget: number): Promise<boolean> {
    if (!this.restClient || !this.tradingAccount) {
      return false;
    }

    try {
      // Cancel existing target if we have the ID
      if (position.targetOrderId) {
        console.log(`[ExecutionManager] 🗑️ Canceling existing target order ${position.targetOrderId}`);
        await this.restClient.cancelOrder({
          accountId: this.tradingAccount.id,
          orderId: String(position.targetOrderId),
        });
      }

      // Place new target order
      const oppositeSide = position.side === 'long' ? 1 : 0; // Opposite side for protective order
      console.log(`[ExecutionManager] 📝 Placing new target order @ ${newTarget.toFixed(2)}`);

      const result = await this.restClient.placeOrder({
        accountId: this.tradingAccount.id,
        contractId: this.contractId,
        side: oppositeSide,
        size: position.contracts,
        type: 1, // Limit
        limitPrice: newTarget,
        timeInForce: 1, // GTC
      });

      if (result?.success !== false && result?.orderId) {
        position.targetOrderId = String(result.orderId);
        position.target = newTarget;
        console.log(`[ExecutionManager] ✅ New target order placed: ${result.orderId}`);
        return true;
      }

      console.error('[ExecutionManager] ❌ Failed to place replacement target:', result?.errorMessage);
      return false;
    } catch (error: any) {
      console.error('[ExecutionManager] ❌ Cancel-and-replace target failed:', error?.message || error);
      return false;
    }
  }

  /**
   * Get active position
   */
  getActivePosition(): ActivePosition | null {
    if (this.activePositions.size === 0) return null;
    return Array.from(this.activePositions.values())[0];
  }

  /**
   * Check if we can enter a new trade (respects grace period)
   */
  canEnterNewTrade(): boolean {
    // Check if we have an active position
    if (this.activePositions.size > 0) {
      return false;
    }

    // Check grace period - prevent entries too soon after last position
    const timeSinceLastEntry = Date.now() - this.lastEntryOrderTime;
    if (this.lastEntryOrderTime > 0 && timeSinceLastEntry < this.entryOrderGracePeriodMs) {
      const secondsRemaining = Math.ceil((this.entryOrderGracePeriodMs - timeSinceLastEntry) / 1000);
      console.log(`[ExecutionManager] ⏳ Grace period active: ${secondsRemaining}s remaining before new entries allowed`);
      return false;
    }

    return true;
  }

  /**
   * Check broker position status frequently (called before any exit/modify order)
   * Updates brokerHasPosition flag
   */
  async refreshBrokerPositionStatus(): Promise<boolean> {
    const now = Date.now();

    // Check frequently (every 2 seconds)
    if (now - this.lastBrokerPositionCheck < this.brokerPositionCheckIntervalMs && this.lastBrokerPositionCheck > 0) {
      return this.brokerHasPosition;
    }

    this.lastBrokerPositionCheck = now;

    if (!this.restClient || !this.tradingAccount) {
      return this.brokerHasPosition;
    }

    try {
      const positions = await this.restClient.getPositions(this.tradingAccount.id);
      const brokerPosition = (positions || []).find(
        (pos: any) => this.extractNetQuantity(pos) !== 0 && pos.contractId === this.contractId
      );

      this.brokerHasPosition = Boolean(brokerPosition);
      console.log(`[ExecutionManager] 📊 Broker position check: ${this.brokerHasPosition ? 'POSITION EXISTS' : 'FLAT'}`);
      return this.brokerHasPosition;
    } catch (error: any) {
      console.error(`[ExecutionManager] ⚠️ Error checking broker position:`, error.message);
      return this.brokerHasPosition; // Return cached value on error
    }
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
export function createExecutionManager(
  symbol: string,
  contractId: string,
  contracts: number = 1,
  live: boolean = false,
  options: ExecutionManagerOptions = {}
): ExecutionManager {
  return new ExecutionManager(symbol, contractId, contracts, live, options);
}
