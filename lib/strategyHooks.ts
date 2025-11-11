export interface StrategyHooks {
  onTradeEvent?: (strategyId: string, payload: Record<string, any>) => void;
  onPnLUpdate?: (strategyId: string, realizedPnL: number) => void;
}

export interface RunningStrategy {
  task: Promise<void>;
  shutdown: (reason?: string) => Promise<void>;
}
