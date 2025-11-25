import { EventEmitter } from 'events';
import { loadSignalR } from '@/lib/server/signalr-cjs';
import { authenticate, fetchTopstepXAccounts } from '@/lib/topstepx';
import { ensureSignalRPolyfills } from '@/lib/server/signalrPolyfill';

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

      this.hub.on('GatewayUserOrder', data => {
        const orderId = data.id ?? data.orderId ?? data.orderID;
        if (!orderId) return;

        // Status codes: 0=PendingNew, 1=New, 2=PartialFill, 3=Filled, 4=Canceled, 5=Rejected, etc.
        const status = Number(data.status ?? 0);

        // Remove filled or canceled orders
        if (status === 3 || status === 4 || status === 5) {
          this.snapshot.orders.delete(orderId);
        } else {
          const info: OrderInfo = {
            orderId,
            contractId: data.contractId ?? data.instrumentId,
            symbol: data.symbol ?? data.contractName,
            side: Number(data.side ?? 0),
            size: Number(data.size ?? data.quantity ?? 0),
            type: Number(data.type ?? data.orderType ?? 0),
            status,
            limitPrice: data.limitPrice != null ? Number(data.limitPrice) : undefined,
            stopPrice: data.stopPrice != null ? Number(data.stopPrice) : undefined,
            filledSize: data.filledSize != null ? Number(data.filledSize) : undefined,
            lastUpdate: new Date().toISOString(),
          };
          this.snapshot.orders.set(orderId, info);
        }
        this.emit('update', this.getSnapshot());
      });

      const subscribe = async () => {
        if (!this.hub) return;
        try {
          await this.hub.invoke('SubscribeAccounts');
          await this.hub.invoke('SubscribeOrders', this.accountId);
          await this.hub.invoke('SubscribePositions', this.accountId);
          await this.hub.invoke('SubscribeTrades', this.accountId);
        } catch (err) {
          console.error('[topstepx-account-feed] subscribe failed', err);
        }
      };

      await this.hub.start();
      await subscribe();
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
