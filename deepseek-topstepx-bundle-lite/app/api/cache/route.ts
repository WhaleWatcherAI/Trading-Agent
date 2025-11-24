import { NextRequest, NextResponse } from 'next/server';
import { getCacheStats } from '@/lib/dataCache';
import {
  getPollingStatus,
  startPollingService,
  stopPollingService,
} from '@/lib/pollingService';

/**
 * Cache Status & Control API
 *
 * GET  /api/cache - Get cache and polling status
 * POST /api/cache - Start/stop polling service
 */

export async function GET(request: NextRequest) {
  try {
    const cacheStats = getCacheStats();
    const pollingStatus = getPollingStatus();

    return NextResponse.json({
      cache: cacheStats,
      polling: pollingStatus,
      timestamp: new Date().toISOString(),
    });
  } catch (error: any) {
    return NextResponse.json(
      { error: 'Failed to get cache status', details: error.message },
      { status: 500 }
    );
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { action } = body;

    if (action === 'start') {
      startPollingService();
      return NextResponse.json({ message: 'Polling service started' });
    } else if (action === 'stop') {
      stopPollingService();
      return NextResponse.json({ message: 'Polling service stopped' });
    } else {
      return NextResponse.json(
        { error: 'Invalid action. Use "start" or "stop"' },
        { status: 400 }
      );
    }
  } catch (error: any) {
    return NextResponse.json(
      { error: 'Failed to control polling service', details: error.message },
      { status: 500 }
    );
  }
}
