import { NextRequest, NextResponse } from 'next/server';
import { calculateGexForSymbol, GexMode } from '@/lib/gexCalculator';

export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams;
  const symbol = searchParams.get('symbol') || 'SPY';
  const mode = (searchParams.get('mode') || 'intraday') as GexMode;

  try {
    const result = await calculateGexForSymbol(symbol, mode);
    return NextResponse.json(result);
  } catch (error: any) {
    console.error('Error fetching Tradier GEX data:', error);

    return NextResponse.json(
      { error: 'Failed to fetch GEX data from Tradier', details: error.message },
      { status: error.response?.status || 500 }
    );
  }
}
