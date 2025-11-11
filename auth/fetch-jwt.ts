import { authenticate } from '../lib/topstepx';

export async function fetchJwt(): Promise<{ token: string; expiresAtMs?: number }> {
  if (process.env.PROJECTX_JWT) {
    return {
      token: process.env.PROJECTX_JWT,
      expiresAtMs: Date.now() + 23 * 3600 * 1000,
    };
  }

  // Fall back to TopstepX login when a standalone ProjectX token isn't provided.
  const token = await authenticate();
  return {
    token,
    expiresAtMs: Date.now() + 23 * 3600 * 1000,
  };
}
