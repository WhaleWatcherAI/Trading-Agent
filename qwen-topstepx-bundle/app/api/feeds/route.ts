import { NextRequest, NextResponse } from 'next/server';
import {
  getGreekFlow,
  getSectorFlow,
  getSectorTide,
  getSpotGEX,
  getVolatilityStats,
  getInstitutionalActivity,
  getNewsHeadlines,
} from '@/lib/unusualwhales';

export async function GET(request: NextRequest) {
  try {
    const today = new Date().toISOString().split('T')[0];

    console.log('🔄 Fetching all data feeds...');

    // Fetch all endpoints in parallel with error handling
    const [
      greekFlow,
      sectorFlow,
      sectorTide,
      spotGEX,
      volatilityStats,
      institutionalActivity,
      newsHeadlines,
    ] = await Promise.all([
      getGreekFlow(today).catch((err) => { console.error('Greek Flow failed:', err); return []; }),
      getSectorFlow(today).catch((err) => { console.error('Sector Flow failed:', err); return []; }),
      getSectorTide(today).catch((err) => { console.error('Sector Tide failed:', err); return []; }),
      getSpotGEX(today).catch((err) => { console.error('Spot GEX failed:', err); return []; }),
      getVolatilityStats(today).catch((err) => { console.error('Volatility Stats failed:', err); return []; }),
      getInstitutionalActivity(today).catch((err) => { console.error('Institutional Activity failed:', err); return []; }),
      getNewsHeadlines(today).catch((err) => { console.error('News Headlines failed:', err); return []; }),
    ]);

    console.log(`✅ Greek Flow: ${greekFlow.length} items`);
    console.log(`✅ Sector Flow (Insider): ${sectorFlow.length} items`);
    console.log(`✅ Sector Tide: ${sectorTide.length} items`);
    console.log(`✅ Spot GEX: ${spotGEX.length} items`);
    console.log(`✅ Volatility Stats: ${volatilityStats.length} items`);
    console.log(`✅ Institutional Activity: ${institutionalActivity.length} items`);
    console.log(`✅ News Headlines: ${newsHeadlines.length} items`);

    return NextResponse.json({
      greekFlow,
      sectorFlow,
      sectorTide,
      spotGEX,
      volatilityStats,
      institutionalActivity,
      newsHeadlines,
      timestamp: new Date(),
    });
  } catch (error) {
    console.error('Error fetching feeds:', error);
    return NextResponse.json(
      { error: 'Failed to fetch feeds' },
      { status: 500 }
    );
  }
}
