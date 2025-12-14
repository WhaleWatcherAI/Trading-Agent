export interface TwelveDataBar {
  time: string;
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

function getSecondSundayOfMarch(year: number): number {
  const marchFirst = new Date(Date.UTC(year, 2, 1));
  const firstSundayOffset = (7 - marchFirst.getUTCDay()) % 7;
  return 1 + firstSundayOffset + 7;
}

function getFirstSundayOfNovember(year: number): number {
  const novemberFirst = new Date(Date.UTC(year, 10, 1));
  const firstSundayOffset = (7 - novemberFirst.getUTCDay()) % 7;
  return 1 + firstSundayOffset;
}

function isUsEasternDst(year: number, month: number, day: number): boolean {
  if (month < 3 || month > 11) {
    return false;
  }
  if (month > 3 && month < 11) {
    return true;
  }
  if (month === 3) {
    const secondSunday = getSecondSundayOfMarch(year);
    return day >= secondSunday;
  }
  if (month === 11) {
    const firstSunday = getFirstSundayOfNovember(year);
    return day < firstSunday;
  }
  return false;
}

function convertEasternToUtc(datetime: string): string | null {
  const [datePart, timePart] = datetime.split(' ');
  if (!datePart || !timePart) return null;
  const [yearStr, monthStr, dayStr] = datePart.split('-');
  const [hourStr, minuteStr, secondStr] = timePart.split(':');
  if (!yearStr || !monthStr || !dayStr || !hourStr || !minuteStr || !secondStr) {
    return null;
  }
  const year = Number(yearStr);
  const month = Number(monthStr);
  const day = Number(dayStr);
  const hour = Number(hourStr);
  const minute = Number(minuteStr);
  const second = Number(secondStr);
  if (![year, month, day, hour, minute, second].every(v => Number.isFinite(v))) {
    return null;
  }

  const monthIndex = month - 1;
  const localUtcMillis = Date.UTC(year, monthIndex, day, hour, minute, second);
  const isDst = isUsEasternDst(year, month, day);
  const offsetMinutes = isDst ? -4 * 60 : -5 * 60;
  const utcMillis = localUtcMillis - offsetMinutes * 60 * 1000;
  return new Date(utcMillis).toISOString();
}

export async function fetchTwelveDataBars(
  symbol: string,
  date: string,
  intervalMinutes: number,
  apiKey: string,
): Promise<TwelveDataBar[]> {
  const interval = `${intervalMinutes}min`;
  const start = `${date} 09:30:00`;
  const end = `${date} 16:00:00`;

  const params = new URLSearchParams({
    symbol,
    interval,
    start_date: start,
    end_date: end,
    order: 'ASC',
    timezone: 'America/New_York',
    apikey: apiKey,
    outputsize: '5000',
  });

  const url = `https://api.twelvedata.com/time_series?${params.toString()}`;
  const response = await fetch(url);
  if (!response.ok) {
    const text = await response.text();
    throw new Error(`Twelve Data HTTP ${response.status}: ${text}`);
  }
  const data = await response.json();
  if (data?.status === 'error') {
    throw new Error(data?.message || 'Unknown Twelve Data error');
  }

  const values: any[] = Array.isArray(data?.values) ? data.values : [];
  return values
    .map(item => {
      const datetime = item?.datetime;
      if (!datetime) return null;
      const timestamp = convertEasternToUtc(datetime);
      if (!timestamp) return null;
      const open = Number(item?.open);
      const high = Number(item?.high);
      const low = Number(item?.low);
      const close = Number(item?.close);
      const volume = Number(item?.volume ?? 0);
      if (![open, high, low, close].every(v => Number.isFinite(v))) {
        return null;
      }
      return {
        time: timestamp,
        timestamp,
        open,
        high,
        low,
        close,
        volume: Number.isFinite(volume) ? volume : 0,
      } satisfies TwelveDataBar;
    })
    .filter((bar): bar is TwelveDataBar => bar !== null);
}
