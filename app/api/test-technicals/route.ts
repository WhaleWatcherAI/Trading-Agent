import { NextResponse } from 'next/server';
import { getCached } from '@/lib/dataCache';

export async function GET() {
  const cached = getCached<Record<string, any>>('technicals');

  if (!cached) {
    return NextResponse.json({ error: 'No technicals in cache' });
  }

  return NextResponse.json({
    symbols: Object.keys(cached.data),
    data: cached.data,
    updatedAt: new Date(cached.updatedAt).toISOString(),
    source: cached.source,
  }, { status: 200 });
}
