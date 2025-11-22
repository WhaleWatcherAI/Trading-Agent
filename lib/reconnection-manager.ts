/**
 * Reconnection Manager for TopStepX
 * Prevents infinite reconnection loops and duplicate connections
 */

export interface ReconnectionConfig {
  maxAttempts: number;
  baseDelayMs: number;
  maxDelayMs: number;
  resetAfterMs: number;
}

export class ReconnectionManager {
  private static instance: ReconnectionManager;
  private reconnectAttempts = new Map<string, number>();
  private lastAttemptTime = new Map<string, number>();
  private activeConnections = new Set<string>();

  private config: ReconnectionConfig = {
    maxAttempts: 3,
    baseDelayMs: 1000,
    maxDelayMs: 30000,
    resetAfterMs: 300000 // 5 minutes
  };

  private constructor() {}

  static getInstance(): ReconnectionManager {
    if (!this.instance) {
      this.instance = new ReconnectionManager();
    }
    return this.instance;
  }

  /**
   * Check if we should attempt reconnection
   */
  shouldReconnect(connectionId: string): boolean {
    const attempts = this.reconnectAttempts.get(connectionId) || 0;
    const lastAttempt = this.lastAttemptTime.get(connectionId) || 0;
    const now = Date.now();

    // Reset counter if enough time has passed
    if (now - lastAttempt > this.config.resetAfterMs) {
      this.reconnectAttempts.set(connectionId, 0);
      console.log(`[ReconnectionManager] Reset attempts for ${connectionId} after ${this.config.resetAfterMs}ms`);
    }

    if (attempts >= this.config.maxAttempts) {
      console.error(`[ReconnectionManager] Max reconnection attempts (${this.config.maxAttempts}) reached for ${connectionId}`);
      return false;
    }

    return true;
  }

  /**
   * Get delay before next reconnection attempt (exponential backoff)
   */
  getReconnectDelay(connectionId: string): number {
    const attempts = this.reconnectAttempts.get(connectionId) || 0;
    const delay = Math.min(
      this.config.baseDelayMs * Math.pow(2, attempts),
      this.config.maxDelayMs
    );

    console.log(`[ReconnectionManager] Reconnect delay for ${connectionId}: ${delay}ms (attempt ${attempts + 1})`);
    return delay;
  }

  /**
   * Register a reconnection attempt
   */
  async attemptReconnection(
    connectionId: string,
    reconnectFn: () => Promise<void>
  ): Promise<void> {
    if (!this.shouldReconnect(connectionId)) {
      console.error(`[ReconnectionManager] Stopping ${connectionId} - max attempts reached`);
      process.exit(1); // Clean shutdown
    }

    const attempts = this.reconnectAttempts.get(connectionId) || 0;
    this.reconnectAttempts.set(connectionId, attempts + 1);
    this.lastAttemptTime.set(connectionId, Date.now());

    const delay = this.getReconnectDelay(connectionId);

    console.log(`[ReconnectionManager] Waiting ${delay}ms before reconnection attempt ${attempts + 1}/${this.config.maxAttempts} for ${connectionId}`);
    await new Promise(resolve => setTimeout(resolve, delay));

    try {
      await reconnectFn();
      // Success - reset counter
      this.reconnectAttempts.set(connectionId, 0);
      console.log(`[ReconnectionManager] Successfully reconnected ${connectionId}`);
    } catch (error) {
      console.error(`[ReconnectionManager] Reconnection failed for ${connectionId}:`, error);
      // Will retry on next attempt if under limit
      throw error;
    }
  }

  /**
   * Check if connection is already active (prevent duplicates)
   */
  isConnectionActive(connectionId: string): boolean {
    return this.activeConnections.has(connectionId);
  }

  /**
   * Register active connection
   */
  registerConnection(connectionId: string): void {
    if (this.isConnectionActive(connectionId)) {
      console.warn(`[ReconnectionManager] Connection ${connectionId} already active! Preventing duplicate.`);
      throw new Error(`Duplicate connection attempted for ${connectionId}`);
    }
    this.activeConnections.add(connectionId);
    console.log(`[ReconnectionManager] Registered connection: ${connectionId}`);
  }

  /**
   * Unregister connection
   */
  unregisterConnection(connectionId: string): void {
    this.activeConnections.delete(connectionId);
    console.log(`[ReconnectionManager] Unregistered connection: ${connectionId}`);
  }

  /**
   * Reset all connections (for cleanup)
   */
  reset(): void {
    this.reconnectAttempts.clear();
    this.lastAttemptTime.clear();
    this.activeConnections.clear();
    console.log(`[ReconnectionManager] Reset all connections`);
  }

  /**
   * Get status for monitoring
   */
  getStatus(): {
    activeConnections: string[];
    reconnectAttempts: Record<string, number>;
  } {
    return {
      activeConnections: Array.from(this.activeConnections),
      reconnectAttempts: Object.fromEntries(this.reconnectAttempts)
    };
  }
}

// Export singleton instance
export const reconnectionManager = ReconnectionManager.getInstance();