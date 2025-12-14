type FetchJwt = () => Promise<{ token: string; expiresAtMs?: number }>;

export function createJwtManager(fetchJwt: FetchJwt, preload?: { token: string; expiresAtMs?: number }) {
  let token = preload?.token ?? process.env.PROJECTX_JWT ?? '';
  let exp = preload?.expiresAtMs ?? (Date.now() + 23 * 3600 * 1000);
  let refreshPromise: Promise<void> | null = null;
  let lastRefreshTime = 0;
  const MIN_REFRESH_INTERVAL = 5000; // Don't refresh more than once every 5 seconds

  async function refresh(reason: string) {
    // Debounce rapid refresh calls
    const now = Date.now();
    if (now - lastRefreshTime < MIN_REFRESH_INTERVAL) {
      console.log(`[auth] JWT refresh debounced (last refresh ${now - lastRefreshTime}ms ago)`);
      return;
    }

    // If already refreshing, wait for that to complete
    if (refreshPromise) {
      console.log(`[auth] JWT refresh already in progress, waiting...`);
      await refreshPromise;
      return;
    }

    // Start refresh with mutex
    refreshPromise = (async () => {
      try {
        const r = await fetchJwt();
        token = r.token;
        exp = r.expiresAtMs ?? (Date.now() + 23 * 3600 * 1000);
        lastRefreshTime = Date.now();
        console.log(`[auth] JWT refreshed (${reason}); ttl≈${Math.round((exp - Date.now()) / 3600000)}h`);
      } finally {
        refreshPromise = null;
      }
    })();

    await refreshPromise;
  }

  return {
    getJwt: async () => {
      if (Date.now() > exp - 120_000) await refresh('proactive');
      return token;
    },
    markStaleAndRefresh: async () => refresh('401'),
    setManual: (t: string, ttlSec = 23 * 3600) => {
      token = t;
      exp = Date.now() + ttlSec * 1000;
    },
  };
}
