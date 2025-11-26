import { EventEmitter } from 'events';
import { loadSignalR } from '@/lib/server/signalr-cjs';
import { authenticate, fetchTopstepXAccounts } from '@/lib/topstepx';
import { ensureSignalRPolyfills } from '@/lib/server/signalrPolyfill';
import { createProjectXRest } from '@/projectx-rest';

interface PositionInfo {
  key: string;
  symbol: string;
  netQty: number;
  avgPrice: number;
  contractId?: string;
  lastUpdate: string;
}

interface OrderInfo {
  orderId: string | number;
  contractId?: string;
  symbol?: string;
  side: number;
  size: number;
  type: number;
  status: number;
  limitPrice?: number;
  stopPrice?: number;
  filledSize?: number;
  lastUpdate: string;
}

interface AccountSnapshot {
  accountId: number;
  account?: Record<string, any> | null;
  positions: PositionInfo[];
  orders: OrderInfo[];
  lastUpdate?: string;
}

type HubConnection = any;

class TopstepxAccountFeed extends EventEmitter {
  private hydratePromise: Promise<void> | null = null;
  private ordersHydratePromise: Promise<void> | null = null;

  private async hydrateAccountDetails() {
    if (!this.hydratePromise) {
      this.hydratePromise = (async () => {
        try {
          const metadata = await fetchTopstepXAccounts(false);
          const account = metadata.find(acc => acc.id === this.accountId) || null;
          if (account) {
            this.snapshot.account = {
              ...this.snapshot.account,
              ...account,
            };
            this.snapshot.lastUpdate = new Date().toISOString();
            this.emit('update', this.getSnapshot());
          }
        } catch (err) {
          console.warn('[topstepx-account-feed] failed to hydrate account details', err);
        }
      })();
    }
    return this.hydratePromise;
  }

  private async hydrateOrders() {
    if (!this.ordersHydratePromise) {
      this.ordersHydratePromise = (async () => {
        try {
          console.log(`[topstepx-account-feed] 🔄 Rehydrating open orders for account ${this.accountId}...`);
          const restClient = createProjectXRest();
          const response = await restClient.searchOpenOrders({ accountId: this.accountId });

          if (response.success && response.orders) {
            console.log(`[topstepx-account-feed] 📋 Found ${response.orders.length} open order(s) from REST API`);

            // Populate orders map from API response
            for (const order of response.orders) {
              const info: OrderInfo = {
                orderId: order.id,
                contractId: order.contractId,
                symbol: order.symbolId,
                side: order.side,
                size: order.size,
                type: order.type,
                status: order.status,
                limitPrice: order.limitPrice,
                stopPrice: order.stopPrice,
                filledSize: order.fillVolume,
                lastUpdate: new Date().toISOString(),
              };
              this.snapshot.orders.set(order.id, info);
              console.log(`[topstepx-account-feed] ✅ Rehydrated order ${order.id}: ${info.symbol} ${info.side===0?'BUY':'SELL'} ${info.size} @ ${info.limitPrice || info.stopPrice} (type=${info.type}, status=${info.status})`);
            }

            this.snapshot.lastUpdate = new Date().toISOString();
            this.emit('update', this.getSnapshot());
          } else {
            console.log('[topstepx-account-feed] ℹ️ No open orders found via REST API');
          }
        } catch (err: any) {
          console.warn('[topstepx-account-feed] ⚠️ Failed to hydrate orders:', err?.message || err);
        }
      })();
    }
    return this.ordersHydratePromise;
  }

  private hub: HubConnection | null = null;
  private snapshot: {
    account: Record<string, any> | null;
    positions: Map<string, PositionInfo>;
    orders: Map<string | number, OrderInfo>;
    lastUpdate?: string;
  } = { account: null, positions: new Map(), orders: new Map() };
  private starting = false;

  constructor(private accountId: number) {
    super();
  }

  async ensureStarted() {
    if (this.hub || this.starting) {
      return;
    }
    this.starting = true;
    try {
      const { HubConnectionBuilder, HttpTransportType, LogLevel } = await loadSignalR();
      const token = await authenticate();
      ensureSignalRPolyfills();
      this.hub = new HubConnectionBuilder()
        .withUrl(`${process.env.TOPSTEPX_USER_HUB_URL || 'https://rtc.topstepx.com/hubs/user'}?access_token=${encodeURIComponent(token)}`, {
          skipNegotiation: true,
          transport: HttpTransportType.WebSockets,
          accessTokenFactory: () => token,
        })
        .withAutomaticReconnect()
        .configureLogging(LogLevel.Error)
        .build();

      this.hub.on('GatewayUserAccount', data => {
        this.snapshot.account = data;
        this.snapshot.lastUpdate = new Date().toISOString();
        this.emit('update', this.getSnapshot());
      });

      this.hub.on('GatewayUserPosition', data => {
        const key =
          String(data.contractId ?? data.symbol ?? data.instrumentId ?? data.id ?? Date.now());
        const netQty = Number(data.netQty ?? data.position ?? data.size ?? data.qty ?? 0);
        if (!netQty) {
          this.snapshot.positions.delete(key);
        } else {
          const info: PositionInfo = {
            key,
            symbol: data.symbol ?? data.contractName ?? data.name ?? key,
            netQty,
            avgPrice: Number(data.avgPrice ?? data.price ?? data.entryPrice ?? 0),
            contractId: data.contractId ?? data.instrumentId ?? undefined,
            lastUpdate: new Date().toISOString(),
          };
          this.snapshot.positions.set(key, info);
        }
        this.emit('update', this.getSnapshot());
      });

      // Debug: log ALL events received from the hub
      (this.hub as any).onclose = (error: any) => {
        console.error('[topstepx-account-feed] 🔴 Hub connection closed:', error);
      };

      this.hub.on('GatewayUserOrder', data => {
        console.log('[topstepx-account-feed] 📨 GatewayUserOrder event received:', JSON.stringify(data));

        // TopstepX sometimes wraps data in data.data property
        const orderData = data.data ?? data;

        const orderId = orderData.id ?? orderData.orderId ?? orderData.orderID;
        if (!orderId) {
          console.warn('[topstepx-account-feed] ⚠️ GatewayUserOrder missing orderId. Raw data:', data, 'Unwrapped:', orderData);
          return;
        }

        // OrderStatus enum: 0=None, 1=Open, 2=Filled, 3=Cancelled, 4=Expired, 5=Rejected, 6=Pending
        const status = Number(orderData.status ?? 0);

        // Remove filled, cancelled, expired, or rejected orders (only track Open/Pending)
        if (status === 2 || status === 3 || status === 4 || status === 5) {
          const statusName = status === 2 ? 'Filled' : status === 3 ? 'Cancelled' : status === 4 ? 'Expired' : 'Rejected';
          console.log(`[topstepx-account-feed] 🗑️ Removing order ${orderId} (status=${status}: ${statusName})`);
          this.snapshot.orders.delete(orderId);
        } else {
          const info: OrderInfo = {
            orderId,
            contractId: orderData.contractId ?? orderData.instrumentId,
            symbol: orderData.symbol ?? orderData.contractName,
            side: Number(orderData.side ?? 0),
            size: Number(orderData.size ?? orderData.quantity ?? 0),
            type: Number(orderData.type ?? orderData.orderType ?? 0),
            status,
            limitPrice: orderData.limitPrice != null ? Number(orderData.limitPrice) : undefined,
            stopPrice: orderData.stopPrice != null ? Number(orderData.stopPrice) : undefined,
            filledSize: orderData.filledSize != null ? Number(orderData.filledSize) : undefined,
            lastUpdate: new Date().toISOString(),
          };
          console.log(`[topstepx-account-feed] ✅ Tracking order ${orderId}: ${info.symbol} ${info.side===0?'BUY':'SELL'} ${info.size} @ ${info.limitPrice || info.stopPrice} (type=${info.type}, status=${status})`);
          this.snapshot.orders.set(orderId, info);
        }
        this.emit('update', this.getSnapshot());
      });

      const subscribe = async () => {
        if (!this.hub) return;
        try {
          console.log(`[topstepx-account-feed] 📡 Subscribing to account ${this.accountId} data streams...`);
          await this.hub.invoke('SubscribeAccounts');
          console.log('[topstepx-account-feed] ✅ SubscribeAccounts succeeded');
          await this.hub.invoke('SubscribeOrders', this.accountId);
          console.log(`[topstepx-account-feed] ✅ SubscribeOrders(${this.accountId}) succeeded`);
          await this.hub.invoke('SubscribePositions', this.accountId);
          console.log(`[topstepx-account-feed] ✅ SubscribePositions(${this.accountId}) succeeded`);
          await this.hub.invoke('SubscribeTrades', this.accountId);
          console.log(`[topstepx-account-feed] ✅ SubscribeTrades(${this.accountId}) succeeded`);
        } catch (err) {
          console.error('[topstepx-account-feed] ❌ subscribe failed', err);
        }
      };

      await this.hub.start();
      await subscribe();

      // Rehydrate existing open orders from REST API
      await this.hydrateOrders();

      this.hub.onreconnected(() => {
        subscribe();
      });
    } finally {
      this.starting = false;
    }
  }

  getSnapshot(): AccountSnapshot {
    return {
      accountId: this.accountId,
      account: this.snapshot.account,
      positions: Array.from(this.snapshot.positions.values()),
      orders: Array.from(this.snapshot.orders.values()),
      lastUpdate: this.snapshot.lastUpdate,
    };
  }

  getOpenOrders(): OrderInfo[] {
    return Array.from(this.snapshot.orders.values());
  }
}

const feeds = new Map<number, TopstepxAccountFeed>();

export async function getTopstepxAccountFeed(accountId: number) {
  let feed = feeds.get(accountId);
  if (!feed) {
    feed = new TopstepxAccountFeed(accountId);
    feeds.set(accountId, feed);
  }
  await feed.ensureStarted();
  await feed.hydrateAccountDetails();
  return feed;
}
