import { createRequire } from 'module';

let signalRPromise: Promise<any> | null = null;

function resolveRequire() {
  if (typeof createRequire === 'function') {
    return createRequire(process.cwd() + '/next.config.js');
  }
  if (typeof require === 'function') {
    return require;
  }
  if (typeof globalThis !== 'undefined' && typeof (globalThis as any).__non_webpack_require__ === 'function') {
    return (globalThis as any).__non_webpack_require__;
  }
  return null;
}

export async function loadSignalR() {
  if (!signalRPromise) {
    signalRPromise = Promise.resolve().then(() => {
      const dynamicRequire = resolveRequire();
      if (!dynamicRequire) {
        throw new Error('Unable to resolve require() for SignalR');
      }
      return dynamicRequire('@microsoft/signalr');
    });
  }
  return signalRPromise;
}
