import fs from 'node:fs/promises';
import path from 'node:path';

type Timeframe = {
  label: string;
  unit: 1 | 2 | 3 | 4;
  unitNumber: number;
};

type TopstepBar = {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;
};

const CONTRACT_IDS = [
  'CON.F.US.ENQ.H25',
  'CON.F.US.ENQ.M25',
  'CON.F.US.ENQ.U25',
  'CON.F.US.ENQ.Z25',
];

const TIMEFRAMES: Timeframe[] = [
  { label: '1d', unit: 4, unitNumber: 1 }, // Daily bars
  { label: '4h', unit: 3, unitNumber: 4 }, // 4-hour bars
  { label: '1h', unit: 3, unitNumber: 1 }, // Hourly bars
  { label: '15m', unit: 2, unitNumber: 15 }, // 15-minute bars
];

const START_TIME = '2025-01-01T00:00:00Z';
const END_TIME = '2025-12-31T23:59:59Z';
const OUTPUT_DIR = path.join(process.cwd(), 'tmp', 'topstepx-candles-2025');

async function loadEnv() {
  const envPath = path.join(process.cwd(), '.env');
  try {
    const raw = await fs.readFile(envPath, 'utf8');
    for (const line of raw.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed || trimmed.startsWith('#') || !trimmed.includes('=')) continue;
      const [key, ...rest] = trimmed.split('=');
      const value = rest.join('=');
      if (key && !(key in process.env)) {
        process.env[key] = value;
      }
    }
  } catch (err: any) {
    console.warn(`No .env file found at ${envPath}:`, err?.message || err);
  }
}

function getEnv(key: string): string {
  const value = process.env[key];
  if (!value) {
    throw new Error(`${key} must be set in environment/.env`);
  }
  return value;
}

async function authenticate(baseUrl: string, username: string, apiKey: string): Promise<string> {
  const res = await fetch(`${baseUrl}/api/Auth/loginKey`, {
    method: 'POST',
    headers: {
      Accept: 'text/plain',
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      userName: username,
      apiKey,
    }),
  });

  if (!res.ok) {
    throw new Error(`Auth HTTP ${res.status}: ${await res.text()}`);
  }

  const data = await res.json();
  if (!data.success || data.errorCode !== 0 || !data.token) {
    throw new Error(`Auth failed: ${JSON.stringify(data)}`);
  }

  return data.token as string;
}

async function fetchBars(
  baseUrl: string,
  token: string,
  params: {
    contractId: string;
    unit: 1 | 2 | 3 | 4;
    unitNumber: number;
  },
): Promise<TopstepBar[]> {
  const body = {
    contractId: params.contractId,
    startTime: START_TIME,
    endTime: END_TIME,
    unit: params.unit,
    unitNumber: params.unitNumber,
    limit: 20000,
    live: false,
    includePartialBar: false,
  };

  const res = await fetch(`${baseUrl}/api/History/retrieveBars`, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${token}`,
      'Content-Type': 'application/json',
      Accept: 'application/json',
    },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    throw new Error(`History HTTP ${res.status}: ${await res.text()}`);
  }

  const data = await res.json();
  if (!data.success || !Array.isArray(data.bars)) {
    console.warn('Unexpected history response:', JSON.stringify(data).slice(0, 500));
    return [];
  }

  return data.bars.map((bar: any) => ({
    timestamp: bar.t,
    open: Number(bar.o),
    high: Number(bar.h),
    low: Number(bar.l),
    close: Number(bar.c),
    volume: bar.v != null ? Number(bar.v) : undefined,
  }));
}

async function saveJson(filePath: string, data: any) {
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  await fs.writeFile(filePath, JSON.stringify(data, null, 2));
}

async function fetchAndStore() {
  await loadEnv();

  const username = getEnv('TOPSTEPX_USERNAME');
  const apiKey = getEnv('TOPSTEPX_API_KEY');
  const baseUrl = process.env.TOPSTEPX_BASE_URL || 'https://api.topstepx.com';

  console.log('Authenticating to TopstepX...');
  const token = await authenticate(baseUrl, username, apiKey);
  console.log('✅ Authenticated');

  console.log('Fetching TopstepX candles using credentials from .env...');
  console.log(`Date range: ${START_TIME} to ${END_TIME}`);
  console.log(`Output directory: ${OUTPUT_DIR}`);

  for (const contractId of CONTRACT_IDS) {
    for (const tf of TIMEFRAMES) {
      const label = `[${contractId}][${tf.label}]`;
      try {
        const bars = await fetchBars(baseUrl, token, {
          contractId,
          unit: tf.unit,
          unitNumber: tf.unitNumber,
        });

        const outPath = path.join(
          OUTPUT_DIR,
          `${contractId.replace(/\./g, '_')}-${tf.label}.json`,
        );

        await saveJson(outPath, {
          contractId,
          timeframe: tf.label,
          unit: tf.unit,
          unitNumber: tf.unitNumber,
          startTime: START_TIME,
          endTime: END_TIME,
          count: bars.length,
          bars,
        });

        console.log(`${label} saved ${bars.length} bars -> ${outPath}`);
      } catch (error: any) {
        console.error(`${label} failed:`, error?.message || error);
      }
    }
  }
}

fetchAndStore().catch(err => {
  console.error('Fatal error:', err?.message || err);
  process.exit(1);
});
