type FetchJwt = () => Promise<{ token: string; expiresAtMs?: number }>;

export function createJwtManager(fetchJwt: FetchJwt, preload?: { token: string; expiresAtMs?: number }) {
  let token = preload?.token ?? process.env.PROJECTX_JWT ?? '';
  let exp = preload?.expiresAtMs ?? (Date.now() + 23 * 3600 * 1000);

  async function refresh(reason: string) {
    const r = await fetchJwt();
    token = r.token;
    exp = r.expiresAtMs ?? (Date.now() + 23 * 3600 * 1000);
    console.log(`[auth] JWT refreshed (${reason}); ttl≈${Math.round((exp - Date.now()) / 3600000)}h`);
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
