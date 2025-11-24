import { EventEmitter } from 'events';
import {
  RegimeTradeLifecycle,
  RegimeStage2Summary,
  RegimeStage3Profile,
} from '@/types';

type LifecycleMode = 'scalp' | 'swing' | 'leaps';

interface LifecyclePayload {
  mode: LifecycleMode;
  symbol: string;
  trades: RegimeTradeLifecycle[];
  stage2: RegimeStage2Summary;
  stage3: RegimeStage3Profile;
  timestamp: string;
}

const emitter = new EventEmitter();
emitter.setMaxListeners(100);

const eventKey = (mode: LifecycleMode, symbol: string) =>
  `${mode}:${symbol.toUpperCase()}`;

export function emitLifecycleUpdate(payload: LifecyclePayload): void {
  const key = eventKey(payload.mode, payload.symbol);
  emitter.emit(key, payload);
  emitter.emit('broadcast', payload);
}

export function subscribeLifecycleUpdates(
  mode: LifecycleMode,
  symbol: string,
  listener: (payload: LifecyclePayload) => void,
): () => void {
  const key = eventKey(mode, symbol);
  emitter.on(key, listener);
  return () => {
    emitter.off(key, listener);
  };
}

export function subscribeLifecycleBroadcast(
  listener: (payload: LifecyclePayload) => void,
): () => void {
  emitter.on('broadcast', listener);
  return () => {
    emitter.off('broadcast', listener);
  };
}
