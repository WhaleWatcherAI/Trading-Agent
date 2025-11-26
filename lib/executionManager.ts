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
import { getTopstepxAccountFeed } from './server/topstepxAccountFeed';

interface ExecutionManagerOptions {
  tickSize?: number;
  preferredAccountId?: number;
  enableNativeBrackets?: boolean;
  requireNativeBrackets?: boolean;
  brokerSyncOffsetMs?: number; // Offset for broker sync timing (for multi-agent staggering)
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
  private positionVersionBySymbol: Record<string, number> = {};
  private orderIdCounter = 0;
  private tradingAccount: TopstepXAccount | null = null;
  private restClient: ReturnType<typeof createProjectXRest> | null = null;
  private readonly tickSize: number;
  private readonly riskPoints = 25;
  private readonly rewardPoints = 50;
  private readonly preferredAccountId?: number;
  private readonly enableNativeBrackets: boolean;
  private readonly requireNativeBrackets: boolean;
  private lastBrokerSync = 0;
  private readonly brokerSyncIntervalMs = 15_000;
  private lastEntryOrderTime = 0;
  private readonly entryOrderGracePeriodMs = 90_000; // 90 seconds to prevent duplicate entries after stop loss
  private protectionLockUntilMs = 0; // block early cancel/replace right after entry
  // Debounce broker-flat signals to avoid clearing/canceling on feed lag
  private flatSeenAtMs: number | null = null;
  private readonly flatConfirmMs = 2500; // require 2.5s consecutive flat before cleanup

  // Broker position monitoring to prevent naked close orders
  private brokerHasPosition = false;
  private lastBrokerPositionCheck = 0;
  private readonly brokerPositionCheckIntervalMs = 2_000; // Check every 2 seconds
  private readonly emergencyCooldownMs = 20_000; // short lockout after circuit breaker flatten
  private lastEmergencyFlatten = 0;
  private flatCooldownUntilBySymbol: Record<string, number> = {};
  private lastExternalVersionBump = 0;
  private accountFeed: any = null; // Websocket feed for real-time position updates
  private accountFeedInitialized = false;
  private lastProtectiveModifyMs = 0; // track recent stop/target modifies to avoid false “no stop” detection
  private lastStopModifyMs = 0;
  private lastTargetModifyMs = 0;

  constructor(
    private symbol: string,
    private contractId: string,
    private contracts: number = 1,
    private live: boolean = false,  // false = sim/demo, true = live trading
    options: ExecutionManagerOptions = {}
  ) {
    this.tickSize = options.tickSize && options.tickSize > 0 ? options.tickSize : 0.25;
    console.log(`[ExecutionManager] Initialized with tickSize=${this.tickSize} for ${symbol} (provided: ${options.tickSize})`);
    this.preferredAccountId = options.preferredAccountId;
    this.enableNativeBrackets = options.enableNativeBrackets !== false; // default true to force native when supported
    this.requireNativeBrackets = true; // always require native brackets to avoid non-OCO exposure

    // Apply broker sync offset for multi-agent staggering (Gold uses offset to avoid rate limiting with NQ)
    if (options.brokerSyncOffsetMs && options.brokerSyncOffsetMs > 0) {
      this.lastBrokerSync = Date.now() - this.brokerSyncIntervalMs + options.brokerSyncOffsetMs;
      console.log(`[ExecutionManager] Broker sync offset: ${options.brokerSyncOffsetMs}ms (first sync delayed)`);
    }

    try {
      this.restClient = createProjectXRest();
    } catch (error: any) {
      console.warn('[ExecutionManager] Unable to initialize ProjectX REST client (falling back to legacy order flow):', error?.message || error);
      this.restClient = null;
    }
  }

  private markProtectionLock(seconds: number = 6) {
    this.protectionLockUntilMs = Date.now() + seconds * 1000;
    console.warn(`[ExecutionManager] 🔒 Protection lock active for ${seconds}s to prevent early bracket cancels/adjusts.`);
  }

  private isProtectionLocked(): boolean {
    return Date.now() < this.protectionLockUntilMs;
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

  /**
   * Initialize websocket account feed for real-time position updates
   */
  async initializeAccountFeed(): Promise<void> {
    if (this.accountFeedInitialized || !this.tradingAccount) {
      return;
    }

    try {
      console.log(`[ExecutionManager] 🔌 Initializing websocket position feed for account ${this.tradingAccount.id}...`);
      this.accountFeed = await getTopstepxAccountFeed(this.tradingAccount.id);

      // Listen for position updates
      this.accountFeed.on('update', (snapshot: any) => {
        this.handleAccountFeedUpdate(snapshot);
      });

      this.accountFeedInitialized = true;
      console.log(`[ExecutionManager] ✅ Websocket position feed connected and listening`);
    } catch (error: any) {
      console.error('[ExecutionManager] ❌ Failed to initialize account feed:', error?.message || error);
    }
  }

  /**
   * Get open orders from WebSocket cache instead of API call
   * Returns same format as searchOpenOrders for compatibility
   */
  private async getOpenOrdersCached(): Promise<any> {
    // Fallback to API if account feed not initialized yet
    if (!this.accountFeedInitialized || !this.accountFeed || typeof this.accountFeed.getOpenOrders !== 'function') {
      console.warn('[ExecutionManager] Account feed not ready, falling back to API call');
      return await this.restClient.searchOpenOrders({ accountId: this.tradingAccount.id });
    }

    try {
      const orders = this.accountFeed.getOpenOrders();
      // Convert to same format as searchOpenOrders API response
      return {
        success: true,
        errorCode: 0,
        errorMessage: null,
        orders: orders.map((o: any) => ({
          id: o.orderId,
          accountId: this.tradingAccount.id,
          contractId: o.contractId,
          symbolId: o.symbol,
          status: o.status,
          type: o.type,
          side: o.side,
          size: o.size,
          limitPrice: o.limitPrice,
          stopPrice: o.stopPrice,
          fillVolume: o.filledSize,
        })),
      };
    } catch (error: any) {
      console.warn('[ExecutionManager] Error getting cached orders, falling back to API:', error?.message);
      return await this.restClient.searchOpenOrders({ accountId: this.tradingAccount.id });
    }
  }

  /**
   * Handle real-time position updates from websocket
   */
  private handleAccountFeedUpdate(snapshot: any): void {
    if (!snapshot || !Array.isArray(snapshot.positions)) {
      return;
    }

    // Find position for our contract
    const positionData = snapshot.positions.find((p: any) =>
      p.contractId === this.contractId || p.symbol === this.symbol
    );

    const currentPosition = this.getActivePosition();

    // Position closed externally (manual close, stop hit, target hit)
    if (currentPosition && (!positionData || positionData.netQty === 0)) {
      console.log(`[ExecutionManager] 🔔 Websocket detected position close for ${this.symbol} - syncing...`);
      this.clearIfBrokerFlat();
      console.log(`[ExecutionManager] ✅ Position cleared from tracking (external close detected)`);
    }

    // Position opened externally
    else if (!currentPosition && positionData && positionData.netQty !== 0) {
      console.log(`[ExecutionManager] 🔔 Websocket detected new position for ${this.symbol} (netQty=${positionData.netQty})`);
      this.handleWebsocketPositionUpdate(positionData);
    }

    // CRITICAL: Detect filled entry when TopStepX doesn't send position events
    // If we have brackets but no position and no entry order, the entry filled!
    else if (!currentPosition && Array.isArray(snapshot.orders) && snapshot.orders.length > 0) {
      const contractOrders = snapshot.orders.filter((o: any) =>
        o.contractId === this.contractId || o.symbol === this.symbol
      );

      // Find bracket orders
      const stopOrder = contractOrders.find((o: any) =>
        (o.type === 4 || o.type === 5) && o.stopPrice != null
      );
      const targetOrder = contractOrders.find((o: any) =>
        o.type === 1 && o.limitPrice != null
      );

      // If we have both brackets but no position, entry must have filled
      if (stopOrder && targetOrder) {
        console.log(`[ExecutionManager] 🔔 Websocket detected FILLED ENTRY (brackets exist but no position event) - reconstructing position...`);

        // Reconstruct position from brackets
        const positionSide: 'long' | 'short' = stopOrder.side === 0 ? 'short' : 'long'; // Protective order side is opposite
        const stopLoss = Number(stopOrder.stopPrice);
        const target = Number(targetOrder.limitPrice);
        const contracts = stopOrder.size || 1;

        // Estimate entry from brackets
        const entryPrice = positionSide === 'long'
          ? (stopLoss + target) / 2
          : (target + stopLoss) / 2;

        console.log(`[ExecutionManager] 📊 Reconstructed ${positionSide.toUpperCase()} position: Entry=${entryPrice.toFixed(2)}, Stop=${stopLoss.toFixed(2)}, Target=${target.toFixed(2)}`);

        // Create position tracking
        const decision = tradingDB.getLastPendingDecision();
        if (decision) {
          tradingDB.markDecisionFilled(decision.id, entryPrice);

          const activePosition: ActivePosition = {
            decisionId: decision.id,
            symbol: this.symbol,
            side: positionSide,
            entryPrice,
            entryTime: new Date().toISOString(),
            stopLoss,
            target,
            contracts,
            stopOrderId: stopOrder.orderId,
            targetOrderId: targetOrder.orderId,
            currentPrice: entryPrice,
            positionAgeSeconds: 0,
            unrealizedPnL: 0,
            usesNativeBracket: true,
            positionVersion: 1,
          };

          this.activePositions.set(this.symbol, activePosition);
          console.log(`[ExecutionManager] ✅ Position created from websocket bracket detection`);
        }
      }
    }

    // Sync stop/target from WebSocket orders if position exists
    if (currentPosition && Array.isArray(snapshot.orders) && snapshot.orders.length > 0) {
      const contractOrders = snapshot.orders.filter((o: any) =>
        o.contractId === this.contractId || o.symbol === this.symbol
      );

      // Find stop and target orders
      const stopOrder = contractOrders.find((o: any) =>
        (o.type === 4 || o.type === 5) && o.stopPrice != null // type 4=Stop, type 5=TrailingStop
      );
      const targetOrder = contractOrders.find((o: any) =>
        o.type === 1 && o.limitPrice != null // type 1=Limit
      );

      let updated = false;

      // Update stop if changed
      if (stopOrder && stopOrder.stopPrice != null) {
        const newStop = Number(stopOrder.stopPrice);
        if (Math.abs(newStop - currentPosition.stopLoss) >= this.tickSize / 4) {
          console.log(`[ExecutionManager] 🔄 WebSocket: Syncing stop from ${currentPosition.stopLoss.toFixed(2)} to ${newStop.toFixed(2)} (Order ID ${stopOrder.orderId})`);
          currentPosition.stopLoss = newStop;
          currentPosition.stopOrderId = stopOrder.orderId;
          updated = true;
        }
      }

      // Update target if changed
      if (targetOrder && targetOrder.limitPrice != null) {
        const newTarget = Number(targetOrder.limitPrice);
        if (Math.abs(newTarget - currentPosition.target) >= this.tickSize / 4) {
          console.log(`[ExecutionManager] 🔄 WebSocket: Syncing target from ${currentPosition.target.toFixed(2)} to ${newTarget.toFixed(2)} (Order ID ${targetOrder.orderId})`);
          currentPosition.target = newTarget;
          currentPosition.targetOrderId = targetOrder.orderId;
          updated = true;
        }
      }

      if (updated) {
        console.log(`[ExecutionManager] ✅ Position brackets synced from WebSocket (stop=${currentPosition.stopLoss.toFixed(2)}, target=${currentPosition.target.toFixed(2)})`);
      }
    }
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

    // CRITICAL: Validate ENTRY order against broker state and working entries
    console.log(`[ExecutionManager] 🔍 [ENTRY ORDER VALIDATION] Checking broker is FLAT...`);
    const brokerHasPosition = await this.refreshBrokerPositionStatus();

    if (brokerHasPosition) {
      console.error(`[ExecutionManager] 🚫 [ENTRY ORDER BLOCKED] Position already exists at broker! Cannot enter duplicate position.`);
      order.status = 'rejected';
      order.reason = 'ENTRY blocked: Position already exists at broker';
      return null;
    }

    // Block if any working entry orders exist for this contract (either side)
    try {
      const openOrders = await this.getOpenOrdersCached();
      const workingEntry = (openOrders?.orders || []).some((o: any) =>
        o.contractId === this.contractId &&
        this.isOrderWorking(o) &&
        !this.isProtectiveOrder(o, undefined) // any non-protective order while flat counts as entry risk
      );
      if (workingEntry) {
        console.error(`[ExecutionManager] 🚫 [ENTRY ORDER BLOCKED] Existing working entry order detected for contract ${this.contractId}.`);
        order.status = 'rejected';
        order.reason = 'ENTRY blocked: existing working entry order';
        return null;
      }
    } catch (error: any) {
      console.warn('[ExecutionManager] ⚠️ Unable to check working entry orders:', error?.message || error);
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

    // Validate bracket orientation before submit
    if (order.side === 'buy') {
      if (plannedLevels.stopLoss >= entryReference || plannedLevels.takeProfit <= entryReference) {
        console.error('[ExecutionManager] 🚫 Invalid LONG brackets (stop/target orientation). Aborting entry.');
        order.status = 'rejected';
        order.reason = 'Invalid LONG brackets';
        return null;
      }
    } else {
      if (plannedLevels.stopLoss <= entryReference || plannedLevels.takeProfit >= entryReference) {
        console.error('[ExecutionManager] 🚫 Invalid SHORT brackets (stop/target orientation). Aborting entry.');
        order.status = 'rejected';
        order.reason = 'Invalid SHORT brackets';
        return null;
      }
    }

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

      // If still missing any leg, retry open-order search a few times before giving up
      if (!stopOrderId || !targetOrderId) {
        console.warn('[ExecutionManager] ⚠️ Bracket IDs missing after initial verify; retrying open-order lookup...');
        const protectiveSide = this.getProtectiveSide({ side: order.side === 'buy' ? 'long' : 'short' });
        const retryFound = await this.findProtectiveOrdersWithRetry(protectiveSide, stopLoss, takeProfit, 4, 1000);
        stopOrderId = stopOrderId ?? retryFound.stop?.id;
        targetOrderId = targetOrderId ?? retryFound.target?.id;
        if (retryFound.stop) {
          console.log(`[ExecutionManager] 🔍 Retry found stop ID ${retryFound.stop.id} @ ${retryFound.stop.stopPrice}`);
        }
        if (retryFound.target) {
          console.log(`[ExecutionManager] 🔍 Retry found target ID ${retryFound.target.id} @ ${retryFound.target.limitPrice}`);
        }
      }

      const hasNativeBracket = Boolean(stopOrderId && targetOrderId);

      if (!hasNativeBracket) {
        if (this.requireNativeBrackets) {
          console.error('[ExecutionManager] ❌ Native bracket legs missing after verification and native required. Flattening entry to avoid non-OCO exposure.');
          try {
            const closeSide = order.side === 'buy' ? 1 : 0;
            await this.restClient.placeOrder({
              accountId: this.tradingAccount.id,
              contractId: this.contractId,
              side: closeSide,
              size: order.quantity,
              type: 2, // Market
              timeInForce: 3, // IOC
            });
          } catch (closeError: any) {
            console.error('[ExecutionManager] ❌ Failed to flatten after native bracket missing:', closeError?.message || closeError);
          }
          order.status = 'rejected';
          order.reason = 'Native bracket verification failed';
          return null;
        } else {
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
        const openOrders = await this.getOpenOrdersCached();
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
        // If order was explicitly rejected by safeguards, DO NOT fallback to legacy
        if (order.status === 'rejected') {
          console.error('[ExecutionManager] 🚫 Order rejected by safeguards - NO FALLBACK TO LEGACY');
          return null;
        }
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

    this.activePositions.set(this.symbol, position);
    this.markProtectionLock(6);

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

    // Apply overrides if provided
    let desiredStop = overrides?.stopLoss ?? defaults.stopLoss;
    let desiredTarget = overrides?.takeProfit ?? defaults.takeProfit;

    // Directional safety clamps
    if (side === 'buy') {
      // Stop must be below entry
      if (desiredStop >= entryPrice) {
        console.warn(`[ExecutionManager] ⚠️ Long stop override invalid (${desiredStop.toFixed(2)} >= entry ${entryPrice.toFixed(2)}). Resetting to default.`);
        desiredStop = defaults.stopLoss;
      }
      // Target must be above entry
      if (desiredTarget <= entryPrice) {
        console.warn(`[ExecutionManager] ⚠️ Long target override invalid (${desiredTarget.toFixed(2)} <= entry ${entryPrice.toFixed(2)}). Resetting to default.`);
        desiredTarget = defaults.takeProfit;
      }
    } else {
      // SHORT
      // Stop must be above entry
      if (desiredStop <= entryPrice) {
        console.warn(`[ExecutionManager] ⚠️ Short stop override invalid (${desiredStop.toFixed(2)} <= entry ${entryPrice.toFixed(2)}). Resetting to default.`);
        desiredStop = defaults.stopLoss;
      }
      // Target must be below entry
      if (desiredTarget >= entryPrice) {
        console.warn(`[ExecutionManager] ⚠️ Short target override invalid (${desiredTarget.toFixed(2)} >= entry ${entryPrice.toFixed(2)}). Resetting to default.`);
        desiredTarget = defaults.takeProfit;
      }
    }

    // Clamp stop distance so it cannot exceed target distance (keeps stops tighter than TP)
    const stopDist = Math.abs(desiredStop - entryPrice);
    const targetDist = Math.abs(desiredTarget - entryPrice);
    if (targetDist > 0 && stopDist > targetDist) {
      const clampedStop = entryPrice + Math.sign(desiredStop - entryPrice) * targetDist;
      console.warn(`[ExecutionManager] ⚠️ Stop distance (${stopDist.toFixed(2)}) exceeds target distance (${targetDist.toFixed(2)}). Clamping stop to ${clampedStop.toFixed(2)}.`);
      desiredStop = clampedStop;
    }

    return {
      stopLoss: desiredStop,
      takeProfit: desiredTarget,
    };
  }

  private getBracketTicks(
    side: 'buy' | 'sell',
    entryReference: number,
    stopPrice: number,
    targetPrice: number
  ): { stopTicks: number; targetTicks: number } {
    const MAX_TICKS = 1000; // broker limit for bracket distances

    // Calculate raw signed distances in ticks (rounded to nearest tick)
    const rawStopTicks = Math.round((stopPrice - entryReference) / this.tickSize);
    const rawTargetTicks = Math.round((targetPrice - entryReference) / this.tickSize);

    // TopStepX expects signed ticks:
    //  - LONG: stopTicks should be negative (below), targetTicks positive (above)
    //  - SHORT: stopTicks should be positive (above), targetTicks negative (below)
    const stopSign = side === 'buy' ? -1 : 1;
    const targetSign = side === 'buy' ? 1 : -1;

    let stopTicksMagnitude = rawStopTicks !== 0 ? Math.abs(rawStopTicks) : 1;
    let targetTicksMagnitude = rawTargetTicks !== 0 ? Math.abs(rawTargetTicks) : 1;

    if (stopTicksMagnitude > MAX_TICKS) {
      console.warn(`[ExecutionManager] ⚠️ stopTicks magnitude ${stopTicksMagnitude} exceeds ${MAX_TICKS}, clamping.`);
      stopTicksMagnitude = MAX_TICKS;
    }
    if (targetTicksMagnitude > MAX_TICKS) {
      console.warn(`[ExecutionManager] ⚠️ targetTicks magnitude ${targetTicksMagnitude} exceeds ${MAX_TICKS}, clamping.`);
      targetTicksMagnitude = MAX_TICKS;
    }

    const stopTicks = stopSign * stopTicksMagnitude;
    const targetTicks = targetSign * targetTicksMagnitude;

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

  /**
   * Retry helper: find protective orders over several attempts to allow broker to populate OCO legs.
   */
  private async findProtectiveOrdersWithRetry(
    protectiveSide: number,
    desiredStop: number,
    desiredTarget: number,
    attempts: number = 3,
    delayMs: number = 750
  ): Promise<{ stop?: ProjectXOrderRecord; target?: ProjectXOrderRecord }> {
    for (let i = 0; i < attempts; i++) {
      try {
        const response = await this.getOpenOrdersCached();
        if (response?.success && Array.isArray(response.orders)) {
          const relevantOrders = response.orders.filter(order =>
            order.contractId === this.contractId && order.side === protectiveSide
          );
          const stopOrder = this.pickClosestOrder(
            relevantOrders.filter(o => typeof o.stopPrice === 'number'),
            desiredStop,
            'stopPrice'
          );
          const targetOrder = this.pickClosestOrder(
            relevantOrders.filter(o => typeof o.limitPrice === 'number'),
            desiredTarget,
            'limitPrice'
          );

          if (stopOrder || targetOrder) {
            return { stop: stopOrder, target: targetOrder };
          }
        }
      } catch (err) {
        console.warn('[ExecutionManager] findProtectiveOrdersWithRetry error:', (err as any)?.message || err);
      }

      if (i < attempts - 1) {
        await new Promise(resolve => setTimeout(resolve, delayMs));
      }
    }
    return {};
  }

  private async ensureBracketOrderIds(position: ActivePosition): Promise<void> {
    if (!this.restClient || !this.tradingAccount) {
      return;
    }

    if (position.stopOrderId && position.targetOrderId) {
      return;
    }

    try {
      const response = await this.getOpenOrdersCached();
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
   * Hard sync protective order IDs (stop/target) from open orders for this contract.
   * Optionally recreates any missing leg to avoid running unprotected.
   */
  async syncProtectiveOrdersFromOpenOrders(
    position: ActivePosition,
    createIfMissing: boolean = false
  ): Promise<void> {
    if (!this.restClient || !this.tradingAccount) {
      return;
    }

    // For native brackets, never recreate single legs (would break OCO linkage)
    const allowRecreate = createIfMissing && !position.usesNativeBracket;

    try {
      const response = await this.getOpenOrdersCached();
      if (!response?.success || !Array.isArray(response.orders)) {
        console.warn('[ExecutionManager] syncProtectiveOrdersFromOpenOrders: failed to fetch open orders');
        return;
      }

      const protectiveSide = this.getProtectiveSide(position);
      const relevantOrders = (response.orders || []).filter(order =>
        order.contractId === this.contractId && order.side === protectiveSide
      );

      const stopCandidates = relevantOrders.filter(order => typeof order.stopPrice === 'number' && Number.isFinite(order.stopPrice as number));
      const targetCandidates = relevantOrders.filter(order => typeof order.limitPrice === 'number' && Number.isFinite(order.limitPrice as number));

      const stopOrder = this.pickClosestOrder(stopCandidates, position.stopLoss, 'stopPrice');
      const targetOrder = this.pickClosestOrder(targetCandidates, position.target, 'limitPrice');

      if (stopOrder) {
        position.stopOrderId = stopOrder.id;
        position.stopLoss = Number(stopOrder.stopPrice);
        console.log(`[ExecutionManager] 🔄 Synced stop from open orders: ${position.stopLoss.toFixed(2)} (ID ${stopOrder.id})`);
      }

      if (targetOrder) {
        position.targetOrderId = targetOrder.id;
        position.target = Number(targetOrder.limitPrice);
        console.log(`[ExecutionManager] 🔄 Synced target from open orders: ${position.target.toFixed(2)} (ID ${targetOrder.id})`);
      }

      if (allowRecreate) {
        if (!stopOrder) {
          console.warn('[ExecutionManager] ⚠️ No stop order found in open orders; recreating protective stop.');
          await this.cancelAndReplaceStop(position, position.stopLoss);
        }
        if (!targetOrder) {
          console.warn('[ExecutionManager] ⚠️ No target order found in open orders; recreating protective target.');
          await this.cancelAndReplaceTarget(position, position.target);
        }
      } else if (createIfMissing && position.usesNativeBracket && (!stopOrder || !targetOrder)) {
        console.warn('[ExecutionManager] ⚠️ Native bracket missing IDs; refusing to recreate single legs to avoid breaking OCO linkage.');
      }
    } catch (error: any) {
      console.warn('[ExecutionManager] syncProtectiveOrdersFromOpenOrders error:', error?.message || error);
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
      const openOrders = await this.getOpenOrdersCached();
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

        const response = await this.getOpenOrdersCached();
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
          const openOrdersResponse = await this.getOpenOrdersCached();
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

                  // CRITICAL: Check if exit order also filled (stop or target)
                  // If both entry AND exit filled, position is CLOSED (ghost brackets)
                  const entryFillTime = new Date(actualEntryTime).getTime();
                  const exitSide = positionSide === 'long' ? 1 : 0; // LONG exits with SELL (1), SHORT exits with BUY (0)

                  const filledExitOrders = ordersResponse.orders.filter(order =>
                    order.contractId === this.contractId &&
                    order.side === exitSide &&
                    order.status === 2 && // 2 = Filled
                    order.fillVolume > 0 &&
                    order.filledPrice &&
                    new Date(order.updateTimestamp || order.creationTimestamp).getTime() > entryFillTime
                  );

                  if (filledExitOrders.length > 0) {
                    const exitOrder = filledExitOrders[0];
                    console.log(`[ExecutionManager] 🚫 REHYDRATE FALLBACK: Found FILLED EXIT order (ID: ${exitOrder.id}) at ${Number(exitOrder.filledPrice).toFixed(2)} after entry - POSITION ALREADY CLOSED!`);
                    console.log(`[ExecutionManager] ❌ REHYDRATE FALLBACK: Brackets are orphaned from closed position. Skipping rehydration.`);
                    console.log(`[ExecutionManager] 💡 TIP: Cancel orphaned bracket orders manually to avoid confusion.`);
                    return null;
                  }
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

            this.activePositions.set(this.symbol, activePosition);

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
        const openOrdersResponse = await this.getOpenOrdersCached();
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

      this.activePositions.set(this.symbol, activePosition);

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

      const response = await this.getOpenOrdersCached();

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

    const symbolKey = this.symbol || this.contractId || 'default';
    const prevActive = this.getActivePosition();
    const prevQtySigned = prevActive
      ? (prevActive.side === 'long' ? 1 : -1) * (prevActive.contracts || 0)
      : 0;

    let bumpedThisSync = false;

    try {
      const [positions, openOrders] = await Promise.all([
        this.restClient.getPositions(this.tradingAccount.id),
        this.getOpenOrdersCached(),
      ]);

      const openOrderList = openOrders?.orders || [];
      const brokerPosition = (positions || []).find(pos => this.extractNetQuantity(pos) !== 0 && pos.contractId === this.contractId);
      const active = this.getActivePosition();

      // Invariant 1: If flat at broker, there should be no protective orders (do not cancel resting entries)
      // CRITICAL: Only cancel protective orders if we have NO locally tracked active position
      // TopStepX positions API is laggy, so we must check local state first to prevent canceling live brackets
      if (!brokerPosition && !active) {
        const orphanProtective = openOrderList.filter(order =>
          order.contractId === this.contractId &&
          this.isProtectiveOrder(order, undefined)
        );
        if (orphanProtective.length > 0) {
          console.log(`[ExecutionManager] ⚠️ Flat at broker AND no active local position - found ${orphanProtective.length} true orphan protective order(s). Canceling for safety...`);
          await this.cancelOrders(orphanProtective);
        }
      } else if (!brokerPosition && active) {
        console.log(`[ExecutionManager] ⚠️ TopStepX positions API reports flat but we have active local position - API lag detected, preserving protective orders`);
      }

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
          if (!bumpedThisSync) {
            this.bumpPositionVersion(symbolKey);
            bumpedThisSync = true;
          }

          // CRITICAL: Update lastEntryOrderTime to enforce grace period after broker close
          // This prevents immediate re-entry when stop loss was hit
          this.lastEntryOrderTime = now;
          console.log(`[ExecutionManager] 🚫 Grace period started: No new entries for ${this.entryOrderGracePeriodMs / 1000}s to prevent duplicates`);
        }
        return;
      }

      if (!active) {
        await this.rehydrateActivePosition();
        if (!bumpedThisSync) {
          this.bumpPositionVersion(symbolKey);
          bumpedThisSync = true;
        }
        return;
      }

      const brokerQty = Math.abs(this.extractNetQuantity(brokerPosition));
      active.contracts = brokerQty || active.contracts;
      const protectiveSide = this.getProtectiveSide(active);
      const relevantOrders = openOrderList.filter(order => order.contractId === this.contractId && order.side === protectiveSide);
      const hasStop = relevantOrders.some(o => this.isStopLike(o) && this.isOrderWorking(o) && this.isProtectiveOrder(o, protectiveSide));

      // When we just modified protectives, broker may cancel/replace internally; allow a short transition window
      const inProtectiveTransition = Date.now() - this.lastProtectiveModifyMs < 5000;

      // Invariant 2: position exists -> must have a protective stop sized to position
      if (!hasStop) {
        if (inProtectiveTransition) {
          console.warn('[ExecutionManager] ⏳ Protective stop missing but within post-modify transition window; holding to avoid false flatten.');
          return;
        }
        console.warn('[ExecutionManager] 🚨 Broker shows position but no working protective stop detected. Triggering safety flatten.');
        // Attempt to cancel any open orders to avoid unintended flips, then flatten
        await this.cancelOrders(openOrderList.filter(order => order.contractId === this.contractId));
        await this.flattenPosition(brokerQty, active.side === 'long' ? 'sell' : 'buy', currentPrice);
        this.lastEmergencyFlatten = Date.now();
        this.markFlatCooldown(symbolKey);
        return;
      }

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

    // Version bump if signed qty changed
    const newActive = this.getActivePosition();
    const newQtySigned = newActive
      ? (newActive.side === 'long' ? 1 : -1) * (newActive.contracts || 0)
      : 0;
    if (newQtySigned !== prevQtySigned && !bumpedThisSync) {
      this.bumpPositionVersion(symbolKey);
      bumpedThisSync = true;
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
  async adjustActiveProtection(newStop?: number | null, newTarget?: number | null, positionVersion?: number): Promise<boolean> {
    const adjustTraceId = `adj-${Date.now()}`;
    const trace = (msg: string, extra?: any) => {
      if (extra !== undefined) {
        console.warn(`[AdjustTrace ${adjustTraceId}] ${msg}`, extra);
      } else {
        console.warn(`[AdjustTrace ${adjustTraceId}] ${msg}`);
      }
    };
    trace(`start newStop=${newStop ?? 'none'} newTarget=${newTarget ?? 'none'} version=${positionVersion ?? 'n/a'}`);

    // Hard guard: if broker shows flat or no protective orders remain, skip adjustments
    const brokerPositionCheck = await this.rehydrateActivePosition();
    if (!brokerPositionCheck) {
      trace('abort: rehydrateActivePosition returned false (broker flat / no active position)');
      console.warn('[ExecutionManager] 🚫 adjustActiveProtection aborted: broker reports flat / no active position.');
      return false;
    }

    const position = this.getActivePosition();
    if (!position && !this.brokerHasPosition) {
      trace('abort: no active position locally or at broker');
      console.warn('[ExecutionManager] 🚫 adjustActiveProtection: No active position locally or at broker');
      return false;
    }

    // REMOVED: Protection lock - risk manager needs to act immediately to move stops to breakeven
    // The AI risk manager is smart enough to handle early adjustments properly
    // if (this.isProtectionLocked()) {
    //   trace('abort: protection lock active');
    //   console.warn('[ExecutionManager] 🛑 adjustActiveProtection skipped: protection lock active right after entry.');
    //   return false;
    // }

    // Belt-and-suspenders: if broker is flat but we still have a local position, do NOT adjust/cancel;
    // hold brackets until broker position is confirmed gone.
    const brokerHasPos = await this.refreshBrokerPositionStatus();
    if (!brokerHasPos && position) {
      trace('abort: broker flat signal but local position exists (holding)');
      console.warn('[ExecutionManager] ⚠️ adjustActiveProtection skipped: broker flat signal but local position exists. Holding brackets.');
      return false;
    } else if (!brokerHasPos) {
      trace('abort: broker flat at modification time');
      console.warn('[ExecutionManager] 🚫 adjustActiveProtection blocked: broker flat at modification time.');
      return false;
    }

    // Cooldown: if we recently flattened, ignore any risk adjustments
    const symbolKey = position.symbol || this.contractId || 'default';
    if (this.inFlatCooldown(symbolKey)) {
      trace('abort: flat cooldown active');
      console.warn('[ExecutionManager] ⏳ adjustActiveProtection blocked: flat cooldown active.');
      return false;
    }

    // Stale decision guard: require matching positionVersion if provided
    const currentVersion = this.positionVersionBySymbol[position.symbol] ?? 0;
    if (typeof positionVersion === 'number' && positionVersion !== currentVersion) {
      trace(`abort: stale decision (decisionV=${positionVersion}, currentV=${currentVersion})`);
      console.warn(`[ExecutionManager] 🚫 adjustActiveProtection: stale decision ignored (decisionV=${positionVersion}, currentV=${currentVersion})`);
      return false;
    }

    const hasNewStop = typeof newStop === 'number' && Number.isFinite(newStop);
    const hasNewTarget = typeof newTarget === 'number' && Number.isFinite(newTarget);

    // Always refresh protective order IDs from broker before attempting modifications
    await this.syncProtectiveOrdersFromOpenOrders(position, false);

    // If broker is flat per open orders + positions, clear and refuse any adjustment
    const brokerFlat = await this.verifyPositionFlattened();
    if (brokerFlat) {
      trace('abort: verifyPositionFlattened returned true');
      console.warn('[ExecutionManager] 🚫 Broker flat per verifyPositionFlattened; clearing local position and skipping adjustments.');
      this.activePositions.clear();
      this.positionVersionBySymbol = {};
      this.brokerHasPosition = false;
      return false;
    }

    // For native brackets: if we don't have both IDs after sync, refuse to adjust to avoid breaking OCO linkage
    if (position.usesNativeBracket) {
      if ((hasNewStop && !position.stopOrderId) || (hasNewTarget && !position.targetOrderId)) {
        trace('abort: native bracket leg missing after sync', {
          stopNeeded: hasNewStop,
          targetNeeded: hasNewTarget,
          stopOrderId: position.stopOrderId || 'MISSING',
          targetOrderId: position.targetOrderId || 'MISSING',
        });
        console.warn('[ExecutionManager] 🚫 Native bracket leg missing; refusing adjustment to avoid breaking OCO.', {
          stopNeeded: hasNewStop,
          targetNeeded: hasNewTarget,
          stopOrderId: position.stopOrderId || 'MISSING',
          targetOrderId: position.targetOrderId || 'MISSING',
        });
        return false;
      }
    }

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
    trace('stop/target need update flags', { stopNeedsUpdate, targetNeedsUpdate, tickSize: this.tickSize, currentStop: position.stopLoss, requestedStop: newStop, currentTarget: position.target, requestedTarget: newTarget });

    if (!stopNeedsUpdate && !targetNeedsUpdate) {
      console.log('[ExecutionManager] ✅ No updates needed (prices within tolerance)');
      trace('no-op: within tolerance');
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

    // If still missing, hard-sync from open orders and recreate if necessary
    if ((stopNeedsUpdate && !position.stopOrderId) || (targetNeedsUpdate && !position.targetOrderId)) {
      console.warn('[ExecutionManager] ⚠️ Missing bracket IDs after initial sync; attempting hard sync from open orders...');
      await this.syncProtectiveOrdersFromOpenOrders(position, !position.usesNativeBracket);

      // If still missing any leg, refuse to adjust and warn
      if ((stopNeedsUpdate && !position.stopOrderId) || (targetNeedsUpdate && !position.targetOrderId)) {
        console.error('[ExecutionManager] 🚫 Protective leg missing after hard sync; refusing to adjust to avoid naked state.', {
          stopNeeded: stopNeedsUpdate,
          targetNeeded: targetNeedsUpdate,
          stopOrderId: position.stopOrderId || 'MISSING',
          targetOrderId: position.targetOrderId || 'MISSING',
          usesNativeBracket: position.usesNativeBracket,
        });
        return false;
      }
    }

    // Validate we have order IDs after sync attempt
    if ((stopNeedsUpdate && !position.stopOrderId) || (targetNeedsUpdate && !position.targetOrderId)) {
      console.error('[ExecutionManager] ❌ CRITICAL: Missing bracket order IDs after sync attempt!', {
        stopOrderId: position.stopOrderId || 'MISSING',
        targetOrderId: position.targetOrderId || 'MISSING',
        stopNeeded: stopNeedsUpdate,
        targetNeeded: targetNeedsUpdate,
      });
      console.warn('[ExecutionManager] 🚫 Modify-only mode: refusing to recreate or cancel/replace missing legs. Holding existing protection.');
      return false;
    }

    let updated = false;

      // Modify stop loss with retry logic
      if (stopNeedsUpdate) {
        let normalizedStop = this.normalizePrice(newStop as number);

        // Validate stop against current market price to prevent broker rejection
        const currentPrice = position.currentPrice;
        const isGoldSymbol = this.symbol.startsWith('GC') || this.symbol.startsWith('MGC');
        const minStopDistance = this.tickSize * (isGoldSymbol ? 2 : 4); // tighter for gold, default stricter for others
        let skipStopUpdate = false;

        const rawSide = position.side;
        const side = (rawSide || '').toString().toLowerCase();
        const isShort = side.startsWith('short') || side === 'sell';
        const isLong = side.startsWith('long') || side === 'buy';

        if (!isShort && !isLong) {
          console.warn(`[ExecutionManager] 🛑 Stop trail skipped: unknown side=${rawSide}`);
          skipStopUpdate = true;
        } else if (isShort) {
          // For SHORT: stop must stay ABOVE current price
          if (normalizedStop <= currentPrice) {
            console.warn(`[ExecutionManager] 🛑 SHORT trail skipped: newStop ${normalizedStop.toFixed(2)} is <= current ${currentPrice.toFixed(2)} (wrong side)`);
            skipStopUpdate = true;
          }
          const minValidStop = currentPrice + minStopDistance;
          if (!skipStopUpdate && normalizedStop < minValidStop) {
            if (minValidStop > position.stopLoss) {
              console.log(`[ExecutionManager] ⚠️ SHORT stop adjustment skipped: requested ${normalizedStop.toFixed(2)} too close to market ${currentPrice.toFixed(2)}, would need ${minValidStop.toFixed(2)} but current ${position.stopLoss.toFixed(2)} is better. Keeping existing stop.`);
              skipStopUpdate = true;
            } else {
              console.log(`[ExecutionManager] ⚠️ SHORT stop ${normalizedStop.toFixed(2)} too close to market ${currentPrice.toFixed(2)}, adjusting to valid ${minValidStop.toFixed(2)}`);
              normalizedStop = this.normalizePrice(minValidStop);
            }
          }
        } else {
          // For LONG: stop must stay BELOW current price
          if (normalizedStop >= currentPrice) {
            console.warn(`[ExecutionManager] 🛑 LONG trail skipped: newStop ${normalizedStop.toFixed(2)} is >= current ${currentPrice.toFixed(2)} (wrong side)`);
            skipStopUpdate = true;
          }
          const maxValidStop = currentPrice - minStopDistance;
          if (!skipStopUpdate && normalizedStop > maxValidStop) {
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
          // Throttle stop modifies to avoid churn
          if (Date.now() - this.lastStopModifyMs < 8000) {
            console.warn('[ExecutionManager] ⏳ Stop modify throttled (min 8s between modifies).');
          } else if (Math.abs(normalizedStop - position.stopLoss) < this.tickSize * 3) {
            console.warn('[ExecutionManager] ⏳ Stop modify skipped: change < 3 ticks.');
          } else {
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
          this.lastStopModifyMs = Date.now();
          console.log(`[ExecutionManager] ✅ Stop loss successfully updated to ${normalizedStop.toFixed(2)}`);
          // Refresh IDs in case broker handled modify as cancel/replace
          await this.syncProtectiveOrdersFromOpenOrders(position, true);
          this.lastProtectiveModifyMs = Date.now();
        } else {
          console.error(`[ExecutionManager] ❌ Stop modification failed after retries.`);
          // For safety, do NOT cancel/replace if modify fails; keep existing stop to avoid orphaning
        }
      }
    }
    }

    // Modify target with retry logic
    if (targetNeedsUpdate) {
      const normalizedTarget = this.normalizePrice(newTarget as number);
      console.log(`[ExecutionManager] 🎯 Attempting to modify TARGET: ${position.target.toFixed(2)} -> ${normalizedTarget.toFixed(2)} (OrderID: ${position.targetOrderId})`);

      // Throttle target modifies to avoid churn
      if (Date.now() - this.lastTargetModifyMs < 8000) {
        console.warn('[ExecutionManager] ⏳ Target modify throttled (min 8s between modifies).');
      } else if (Math.abs(normalizedTarget - position.target) < this.tickSize * 3) {
        console.warn('[ExecutionManager] ⏳ Target modify skipped: change < 3 ticks.');
      } else {
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
          this.lastTargetModifyMs = Date.now();
          console.log(`[ExecutionManager] ✅ Target successfully updated to ${normalizedTarget.toFixed(2)}`);
          // Refresh IDs in case broker handled modify as cancel/replace
          await this.syncProtectiveOrdersFromOpenOrders(position, true);
          this.lastProtectiveModifyMs = Date.now();
        } else {
          console.error(`[ExecutionManager] ❌ Target modification failed after retries.`);
          // Safety: do not cancel/replace on modify failure to avoid orphaning
        }
      }
    }

    if (updated) {
      console.log(`[ExecutionManager] ✅ Bracket adjustment completed successfully`);

      // Post-adjustment safety: ensure both protective legs exist at broker
      await this.syncProtectiveOrdersFromOpenOrders(position, true);
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

        const numericOrderId = Number(orderId);
        if (!Number.isFinite(numericOrderId)) {
          console.warn(`[ExecutionManager] 🚫 ${orderType} modification aborted: non-numeric orderId ${orderId}`);
          return false;
        }

        const result = await this.restClient.modifyOrder({
          accountId: this.tradingAccount.id,
          orderId: numericOrderId,
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
   * Get active position
   */
  getActivePosition(): ActivePosition | null {
    const pos = this.activePositions.get(this.symbol) || null;
    if (!pos && this.activePositions.size > 0) {
      console.warn(`[ExecutionManager] getActivePosition() -> null for ${this.symbol}. Keys: ${Array.from(this.activePositions.keys()).join(',')}`);
    }
    return pos;
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
      // If a recent emergency flatten occurred, respect cooldown
      if (this.lastEmergencyFlatten > 0 && now - this.lastEmergencyFlatten < this.emergencyCooldownMs) {
        return false;
      }
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

      if (brokerPosition) {
        this.brokerHasPosition = true;
      } else {
        /**
         * The positions endpoint can intermittently return 404/[] even when a position exists.
         * If we are actively tracking a position in memory, avoid falsely marking flat.
         * As a secondary signal, check for any protective open orders on this contract.
         */
        if (this.activePositions.size > 0) {
          const openOrders = await this.getOpenOrdersCached();
          const hasProtective = (openOrders?.orders || []).some(o => o.contractId === this.contractId);
          // Assume position exists to avoid blocking protective adjustments when rest endpoint is flaky
          this.brokerHasPosition = true;
          const activePosition = this.getActivePosition();
          if (activePosition) {
            await this.syncProtectiveOrdersFromOpenOrders(activePosition, false);
          }
          console.warn(`[ExecutionManager] ⚠️ Positions API returned empty; inferring position exists from memory${hasProtective ? ' and protective orders' : ''}.`);
        } else {
          this.brokerHasPosition = false;
        }
      }

      console.log(`[ExecutionManager] 📊 Broker position check: ${this.brokerHasPosition ? 'POSITION EXISTS' : 'FLAT'}`);
      return this.brokerHasPosition;
    } catch (error: any) {
      console.error(`[ExecutionManager] ⚠️ Error checking broker position:`, error.message);
      return this.brokerHasPosition; // Return cached value on error
    }
  }

  /**
   * Force-clear local position if broker is flat (from websocket or REST/open orders)
   */
  async clearIfBrokerFlat(): Promise<boolean> {
    const brokerFlat = await this.verifyPositionFlattened();
    if (brokerFlat) {
      const localPos = this.getActivePosition();

      // Debounce: if we still have a local position, treat flat as a transient blip until confirmed
      if (localPos) {
        if (this.flatSeenAtMs === null) {
          this.flatSeenAtMs = Date.now();
          console.warn('[ExecutionManager] ⚠️ Broker flat blip detected; starting debounce. Holding local position/brackets.');
          return false;
        }

        const flatDuration = Date.now() - this.flatSeenAtMs;
        if (flatDuration < this.flatConfirmMs) {
          console.warn(`[ExecutionManager] ⚠️ Broker still flat but within debounce (${flatDuration}ms/${this.flatConfirmMs}ms). Holding local position/brackets.`);
          return false;
        }

        console.warn('[ExecutionManager] ✅ Broker flat confirmed after debounce — clearing local active position.');
      }

      // Confirmed flat (or no local position)
      this.flatSeenAtMs = null;
      this.activePositions.clear();
      this.positionVersionBySymbol = {};
      this.brokerHasPosition = false;
      return true;
    }
    // Not flat; reset debounce
    this.flatSeenAtMs = null;
    return false;
  }

  /**
   * Ingest websocket position updates (SignalR) to keep local state in sync
   */
  handleWebsocketPositionUpdate(positionMsg: any) {
    try {
      const qty = this.extractNetQuantity(positionMsg);
      const isSameContract = positionMsg.contractId === this.contractId;

      // Ignore updates that are not for this contract
      if (!isSameContract) {
        return;
      }

      const hasPosition = qty !== 0;
      if (!hasPosition) {
        // Do not immediately clear brackets on a transient flat message; wait for confirmation elsewhere
        console.log(`[ExecutionManager] ⚠️ WS reports FLAT for ${this.symbol}; preserving local position/brackets until confirmed.`);
        this.brokerHasPosition = false;
        return;
      }

      const side: 'long' | 'short' = qty > 0 ? 'long' : 'short';
      const entryPrice = this.extractEntryPrice(positionMsg);
      if (!entryPrice) return;

      const existing = this.getActivePosition();
      const isSame = existing && existing.entryPrice === entryPrice && existing.side === side;

      if (!existing || !isSame) {
        const fallbackLevels = this.calculateBracketLevels(entryPrice, side === 'long' ? 'buy' : 'sell');
        const activePosition: ActivePosition = {
          decisionId: existing?.decisionId || `ws-${Date.now()}`,
          symbol: this.symbol,
          side,
          entryPrice,
          entryTime: positionMsg.updateTimestamp || new Date().toISOString(),
          stopLoss: existing?.stopLoss ?? fallbackLevels.stopLoss,
          target: existing?.target ?? fallbackLevels.takeProfit,
          contracts: Math.abs(qty),
          currentPrice: entryPrice,
          unrealizedPnL: 0,
          unrealizedPnLPercent: 0,
          stopOrderId: existing?.stopOrderId,
          targetOrderId: existing?.targetOrderId,
          usesNativeBracket: existing?.usesNativeBracket,
        };
        this.activePositions.clear();
        // Store positions keyed by symbol for consistent lookups
        this.activePositions.set(this.symbol, activePosition);
        console.log(`[ExecutionManager] 🔔 Websocket position update synced: ${side.toUpperCase()} @ ${entryPrice.toFixed(2)} (${activePosition.contracts} contracts)`);
      }
      this.brokerHasPosition = true;
    } catch (err: any) {
      console.warn('[ExecutionManager] handleWebsocketPositionUpdate error:', err?.message || err);
    }
  }

  /**
   * Proactively clean up any orphaned protective orders when flat
   * Called periodically by agents to ensure no resting limit/stop orders remain
   */
  async cleanupOrphanedOrders(): Promise<void> {
    if (!this.restClient || !this.tradingAccount) {
      return;
    }

    try {
      // First verify we're actually flat at the broker
      const brokerHasPosition = await this.refreshBrokerPositionStatus();
      if (brokerHasPosition) {
        // We have a position, so protective orders are legitimate
        return;
      }

      console.log(`[ExecutionManager] 🧹 Checking for orphaned protective orders while flat...`);

      // Search for any working orders for this contract
      const response = await this.getOpenOrdersCached();

      if (!response?.orders || response.orders.length === 0) {
        return;
      }

      // Find working protective orders for this contract
      const orphanedOrders = response.orders.filter((order: any) =>
        order.contractId === this.contractId &&
        this.isOrderWorking(order) &&
        this.isProtectiveOrder(order, undefined)
      );

      if (orphanedOrders.length === 0) {
        return;
      }

      console.error(`[ExecutionManager] 🚨 Found ${orphanedOrders.length} orphaned protective order(s) while flat!`);

      // Cancel each orphaned order
      for (const order of orphanedOrders) {
        try {
          console.log(`[ExecutionManager] 🗑️ Canceling orphaned order ${order.id} (${order.stopPrice ? 'STOP' : 'LIMIT'} @ ${order.stopPrice || order.limitPrice})`);
          await this.restClient.cancelOrder({
            accountId: this.tradingAccount.id,
            orderId: String(order.id),
          });
          console.log(`[ExecutionManager] ✅ Successfully canceled orphaned order ${order.id}`);
        } catch (error: any) {
          console.error(`[ExecutionManager] ❌ Failed to cancel orphaned order ${order.id}:`, error?.message || error);
        }
      }
    } catch (error: any) {
      console.warn('[ExecutionManager] ⚠️ Orphan cleanup check failed:', error?.message || error);
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

  private isStopLike(order: any): boolean {
    return typeof order.stopPrice === 'number' ||
      typeof order.triggerPrice === 'number' ||
      typeof order.stop === 'number';
  }

  private bumpPositionVersion(symbol: string) {
    this.positionVersionBySymbol[symbol] = (this.positionVersionBySymbol[symbol] ?? 0) + 1;
  }

  private markFlatCooldown(symbol: string, durationMs: number = 10_000) {
    this.flatCooldownUntilBySymbol[symbol] = Date.now() + durationMs;
  }

  private inFlatCooldown(symbol: string): boolean {
    return (this.flatCooldownUntilBySymbol[symbol] ?? 0) > Date.now();
  }

  private isOrderWorking(order: any): boolean {
    const status = (order?.status || '').toString().toLowerCase();
    if (!status) return true;
    return !['filled', 'cancelled', 'canceled', 'rejected', 'expired', 'done'].some(term => status.includes(term));
  }

  private isProtectiveOrder(order: any, protectiveSide?: 'buy' | 'sell'): boolean {
    if (protectiveSide && order.side && order.side !== protectiveSide) return false;
    const typeRaw = order.type ?? order.orderType;
    const typeStr = String(typeRaw || '').toLowerCase();
    const typeIsStop = typeStr.includes('stop') || typeRaw === 3 || typeRaw === 4; // common enums for stop / stop-limit
    const hasStopLike = this.isStopLike(order);
    const reduceOnly =
      order.reduceOnly === true ||
      order.isReduceOnly === true ||
      order.closePosition === true;
    const coi = String(order.clientOrderId || '').toLowerCase();
    const bracketTagged =
      Boolean(order.parentOrderId) ||
      Boolean(order.bracketId) ||
      Boolean(order.ocoGroupId) ||
      coi.includes('bracket') || coi.includes('tp') || coi.includes('sl') || coi.includes('oco');
    return hasStopLike && (reduceOnly || bracketTagged || typeIsStop);
  }

  private async cancelOrders(orders: any[]) {
    for (const order of orders) {
      try {
        await this.restClient?.cancelOrder({
          accountId: this.tradingAccount?.id,
          orderId: String(order.id),
        });
        console.log(`[ExecutionManager] ✅ Cancelled order ${order.id}`);
      } catch (err: any) {
        console.warn(`[ExecutionManager] ⚠️ Failed to cancel order ${order.id}:`, err?.message || err);
      }
    }
  }

  private async flattenPosition(qty: number, side: 'buy' | 'sell', currentPrice: number) {
    if (!this.restClient || !this.tradingAccount) return;
    try {
      console.log(`[ExecutionManager] 🛑 Emergency flatten: sending market ${side.toUpperCase()} for qty=${qty}`);
      await this.restClient.placeOrder({
        accountId: this.tradingAccount.id,
        contractId: this.contractId,
        side: side === 'buy' ? 0 : 1,
        size: qty,
        type: 2, // Market
        timeInForce: 0, // IOC
      });
      const symbolKey = this.symbol || this.contractId || 'default';
      this.bumpPositionVersion(symbolKey);
      this.markFlatCooldown(symbolKey);
    } catch (err: any) {
      console.error('[ExecutionManager] ❌ Emergency flatten failed:', err?.message || err);
    }
  }

  /**
   * Expose current position version for a symbol
   */
  getPositionVersion(symbol: string): number {
    return this.positionVersionBySymbol[symbol] ?? 0;
  }

  /**
   * External version bump hook (e.g., from WS execution events)
   */
  bumpVersionExternal(): void {
    const symbolKey = this.symbol || this.contractId || 'default';
    this.bumpPositionVersion(symbolKey);
    this.lastExternalVersionBump = Date.now();
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
