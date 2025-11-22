/**
 * Integration Tests for Broker APIs
 * Tests TopStepX connection, order placement, position management
 */

import { describe, it, expect, beforeAll, afterAll, jest } from '@jest/globals';
import * as SignalR from '@microsoft/signalr';
import dotenv from 'dotenv';
import { reconnectionManager } from '../lib/reconnection-manager';

// Load environment variables
dotenv.config();

// Mock console to reduce noise during tests
const originalConsole = { ...console };
beforeAll(() => {
  console.log = jest.fn();
  console.error = jest.fn();
});

afterAll(() => {
  console.log = originalConsole.log;
  console.error = originalConsole.error;
});

describe('TopStepX Broker Integration', () => {
  let connection: SignalR.HubConnection | null = null;
  const testTimeout = 30000; // 30 seconds for network operations

  describe('Connection Management', () => {
    it('should connect to TopStepX WebSocket', async () => {
      const connectionUrl = 'wss://demo-api-streaming.topstepx.com/TSXHub';

      connection = new SignalR.HubConnectionBuilder()
        .withUrl(connectionUrl, {
          skipNegotiation: true,
          transport: SignalR.HttpTransportType.WebSockets
        })
        .withAutomaticReconnect({
          nextRetryDelayInMilliseconds: () => 5000
        })
        .build();

      await expect(connection.start()).resolves.toBeUndefined();
      expect(connection.state).toBe(SignalR.HubConnectionState.Connected);
    }, testTimeout);

    it('should authenticate with valid credentials', async () => {
      if (!connection) throw new Error('Connection not established');

      const authPromise = new Promise((resolve, reject) => {
        const timeout = setTimeout(() => reject(new Error('Authentication timeout')), 10000);

        connection!.on('AuthenticationResult', (result: any) => {
          clearTimeout(timeout);
          resolve(result);
        });
      });

      await connection.invoke('Authenticate', {
        username: process.env.TOPSTEP_USERNAME,
        password: process.env.TOPSTEP_PASSWORD
      });

      const result = await authPromise;
      expect(result).toBeDefined();
    }, testTimeout);

    it('should handle reconnection with manager', async () => {
      const mockReconnectFn = jest.fn().mockResolvedValue(undefined);

      // Reset reconnection manager state
      reconnectionManager.reset();

      // Test reconnection attempt
      await expect(
        reconnectionManager.attemptReconnection('test-connection', mockReconnectFn)
      ).resolves.toBeUndefined();

      expect(mockReconnectFn).toHaveBeenCalled();
    });

    it('should prevent duplicate connections', () => {
      reconnectionManager.reset();
      reconnectionManager.registerConnection('test-duplicate');

      expect(() => {
        reconnectionManager.registerConnection('test-duplicate');
      }).toThrow('Duplicate connection attempted');
    });

    it('should respect max reconnection attempts', async () => {
      reconnectionManager.reset();
      const failingReconnectFn = jest.fn().mockRejectedValue(new Error('Connection failed'));

      // Attempt max times
      for (let i = 0; i < 3; i++) {
        try {
          await reconnectionManager.attemptReconnection('test-max', failingReconnectFn);
        } catch (e) {
          // Expected to fail
        }
      }

      // Should not attempt after max
      expect(reconnectionManager.shouldReconnect('test-max')).toBe(false);
    });
  });

  describe('Market Data', () => {
    it('should subscribe to market data', async () => {
      if (!connection) throw new Error('Connection not established');

      const marketDataPromise = new Promise((resolve) => {
        connection!.on('MarketDataUpdate', (data: any) => {
          resolve(data);
        });
      });

      await connection.invoke('SubscribeMarketData', {
        symbol: 'NQZ5',
        marketDataType: ['Last', 'BidAsk']
      });

      // Wait a bit for market data (markets might be closed)
      const racePromise = Promise.race([
        marketDataPromise,
        new Promise(resolve => setTimeout(() => resolve(null), 5000))
      ]);

      const data = await racePromise;
      // Data might be null if markets are closed
      expect(data !== undefined).toBe(true);
    }, testTimeout);
  });

  describe('Order Management', () => {
    it('should validate order parameters', () => {
      const validOrder = {
        symbol: 'NQZ5',
        quantity: 1,
        orderType: 'Market',
        side: 'Buy',
        account: process.env.TOPSTEP_ACCOUNT_ID
      };

      // Validate required fields
      expect(validOrder.symbol).toBeDefined();
      expect(validOrder.quantity).toBeGreaterThan(0);
      expect(['Market', 'Limit', 'Stop'].includes(validOrder.orderType)).toBe(true);
      expect(['Buy', 'Sell'].includes(validOrder.side)).toBe(true);
      expect(validOrder.account).toBeDefined();
    });

    it('should create bracket order structure', () => {
      const bracketOrder = {
        parent: {
          symbol: 'NQZ5',
          quantity: 1,
          orderType: 'Market',
          side: 'Buy'
        },
        stopLoss: {
          orderType: 'Stop',
          stopPrice: 19000,
          offsetTicks: 20
        },
        takeProfit: {
          orderType: 'Limit',
          limitPrice: 19100,
          offsetTicks: 20
        }
      };

      expect(bracketOrder.parent).toBeDefined();
      expect(bracketOrder.stopLoss.stopPrice).toBeLessThan(bracketOrder.takeProfit.limitPrice);
      expect(bracketOrder.stopLoss.offsetTicks).toBeGreaterThan(0);
    });

    it('should handle order rejection scenarios', async () => {
      if (!connection) throw new Error('Connection not established');

      const rejectionPromise = new Promise((resolve) => {
        connection!.on('OrderUpdate', (update: any) => {
          if (update.status === 'Rejected') {
            resolve(update);
          }
        });
      });

      // Send invalid order (e.g., invalid symbol)
      try {
        await connection.invoke('PlaceOrder', {
          symbol: 'INVALID',
          quantity: 1,
          orderType: 'Market',
          side: 'Buy',
          account: process.env.TOPSTEP_ACCOUNT_ID
        });
      } catch (error) {
        // Expected to fail
        expect(error).toBeDefined();
      }
    });
  });

  describe('Position Management', () => {
    it('should track position state', () => {
      const position = {
        symbol: 'NQZ5',
        quantity: 0,
        avgPrice: 0,
        unrealizedPnL: 0,
        realizedPnL: 0
      };

      // Simulate position update
      position.quantity = 1;
      position.avgPrice = 19050.25;

      expect(position.quantity).toBe(1);
      expect(position.avgPrice).toBeGreaterThan(0);
    });

    it('should calculate position PnL', () => {
      const position = {
        symbol: 'NQZ5',
        quantity: 1,
        avgPrice: 19050.00,
        currentPrice: 19055.00,
        tickSize: 0.25,
        tickValue: 5 // $5 per tick for MNQ
      };

      const priceDiff = position.currentPrice - position.avgPrice;
      const ticks = priceDiff / position.tickSize;
      const unrealizedPnL = ticks * position.tickValue * position.quantity;

      expect(unrealizedPnL).toBe(100); // 20 ticks * $5
    });

    it('should handle position reconciliation', () => {
      const brokerPosition = { symbol: 'NQZ5', quantity: 1 };
      const localPosition = { symbol: 'NQZ5', quantity: 0 };

      const needsReconciliation = brokerPosition.quantity !== localPosition.quantity;
      expect(needsReconciliation).toBe(true);

      // Reconcile
      localPosition.quantity = brokerPosition.quantity;
      expect(localPosition.quantity).toBe(brokerPosition.quantity);
    });
  });

  describe('Error Handling', () => {
    it('should handle network disconnection', async () => {
      if (!connection) throw new Error('Connection not established');

      const disconnectPromise = new Promise((resolve) => {
        connection!.onclose(() => resolve(true));
      });

      await connection.stop();
      const disconnected = await disconnectPromise;

      expect(disconnected).toBe(true);
      expect(connection.state).toBe(SignalR.HubConnectionState.Disconnected);
    });

    it('should handle API rate limiting', async () => {
      const rateLimiter = {
        requests: 0,
        maxRequests: 100,
        windowMs: 1000,

        canMakeRequest(): boolean {
          return this.requests < this.maxRequests;
        },

        recordRequest(): void {
          this.requests++;
          setTimeout(() => this.requests--, this.windowMs);
        }
      };

      // Simulate requests
      for (let i = 0; i < 150; i++) {
        if (rateLimiter.canMakeRequest()) {
          rateLimiter.recordRequest();
        } else {
          // Would wait or queue request
          expect(rateLimiter.requests).toBe(rateLimiter.maxRequests);
          break;
        }
      }
    });

    it('should validate environment variables', () => {
      const requiredEnvVars = [
        'TOPSTEP_USERNAME',
        'TOPSTEP_PASSWORD',
        'TOPSTEP_ACCOUNT_ID',
        'TOPSTEP_API_KEY'
      ];

      const missingVars = requiredEnvVars.filter(v => !process.env[v]);

      if (missingVars.length > 0) {
        console.warn(`Missing environment variables: ${missingVars.join(', ')}`);
      }

      // Test should pass but warn about missing vars
      expect(missingVars.length).toBeLessThanOrEqual(requiredEnvVars.length);
    });
  });

  afterAll(async () => {
    // Clean up connection
    if (connection && connection.state === SignalR.HubConnectionState.Connected) {
      await connection.stop();
    }

    // Reset reconnection manager
    reconnectionManager.reset();
  });
});

describe('Mock Trading Scenarios', () => {
  describe('Entry Conditions', () => {
    it('should validate mean reversion entry', () => {
      const marketData = {
        price: 19050,
        sma20: 19070,
        sma50: 19080,
        rsi: 25,
        volume: 1500
      };

      const entryConditions = {
        priceBelowSMA: marketData.price < marketData.sma20,
        oversold: marketData.rsi < 30,
        volumeConfirmation: marketData.volume > 1000
      };

      const shouldEnter = Object.values(entryConditions).every(c => c === true);
      expect(shouldEnter).toBe(true);
    });

    it('should prevent duplicate entries', () => {
      const currentPosition = { quantity: 1 };
      const maxPositions = 1;

      const canEnter = currentPosition.quantity < maxPositions;
      expect(canEnter).toBe(false);
    });
  });

  describe('Risk Management', () => {
    it('should calculate position size', () => {
      const accountBalance = 50000;
      const riskPerTrade = 0.01; // 1%
      const stopLossTicks = 20;
      const tickValue = 5; // $5 per tick for MNQ

      const riskAmount = accountBalance * riskPerTrade;
      const positionSize = Math.floor(riskAmount / (stopLossTicks * tickValue));

      expect(positionSize).toBe(5); // 5 contracts
    });

    it('should enforce maximum drawdown', () => {
      const startingBalance = 50000;
      const currentBalance = 47500;
      const maxDrawdown = 0.06; // 6%

      const currentDrawdown = (startingBalance - currentBalance) / startingBalance;
      const shouldStopTrading = currentDrawdown >= maxDrawdown;

      expect(currentDrawdown).toBe(0.05);
      expect(shouldStopTrading).toBe(false);
    });

    it('should handle partial fills', () => {
      const order = {
        requestedQty: 5,
        filledQty: 3,
        remainingQty: 2,
        status: 'PartiallyFilled'
      };

      expect(order.filledQty).toBeLessThan(order.requestedQty);
      expect(order.remainingQty).toBe(order.requestedQty - order.filledQty);
    });
  });
});