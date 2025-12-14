import fs from 'fs';
import path from 'path';
import { NewsItem, OptionsTrade, InstitutionalTrade } from '@/types';

const DATA_DIR = path.join(process.cwd(), 'data');

// Ensure data directory exists
if (!fs.existsSync(DATA_DIR)) {
  fs.mkdirSync(DATA_DIR, { recursive: true });
}

interface DailyData {
  date: string;
  news: NewsItem[];
  optionsTrades: OptionsTrade[];
  institutionalTrades: InstitutionalTrade[];
  lastUpdated: string;
}

function getTodayString(): string {
  return new Date().toISOString().split('T')[0];
}

function getFilePath(date: string): string {
  return path.join(DATA_DIR, `${date}.json`);
}

function loadDailyData(date: string): DailyData {
  const filePath = getFilePath(date);

  if (fs.existsSync(filePath)) {
    const content = fs.readFileSync(filePath, 'utf-8');
    const data = JSON.parse(content);
    // Parse dates back to Date objects
    data.news = data.news.map((n: any) => ({ ...n, timestamp: new Date(n.timestamp) }));
    data.optionsTrades = data.optionsTrades.map((o: any) => ({
      ...o,
      timestamp: new Date(o.timestamp),
    }));
    data.institutionalTrades = data.institutionalTrades.map((i: any) => ({
      ...i,
      timestamp: new Date(i.timestamp),
    }));
    return data;
  }

  return {
    date,
    news: [],
    optionsTrades: [],
    institutionalTrades: [],
    lastUpdated: new Date().toISOString(),
  };
}

function saveDailyData(data: DailyData): void {
  const filePath = getFilePath(data.date);
  data.lastUpdated = new Date().toISOString();
  fs.writeFileSync(filePath, JSON.stringify(data, null, 2), 'utf-8');
}

// Deduplication helpers
function getNewsKey(news: NewsItem): string {
  return `${news.url}-${news.timestamp.getTime()}`;
}

function getOptionsTradeKey(trade: OptionsTrade): string {
  return `${trade.symbol}-${trade.strike}-${trade.expiration}-${trade.timestamp.getTime()}`;
}

function getInstitutionalTradeKey(trade: InstitutionalTrade): string {
  return `${trade.symbol}-${trade.shares}-${trade.price}-${trade.timestamp.getTime()}`;
}

export function addNews(newsItems: NewsItem[]): { added: number; total: number } {
  const today = getTodayString();
  const data = loadDailyData(today);

  const existingKeys = new Set(data.news.map(getNewsKey));
  const newItems = newsItems.filter(item => !existingKeys.has(getNewsKey(item)));

  data.news.push(...newItems);
  data.news.sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime()); // Most recent first

  saveDailyData(data);

  return { added: newItems.length, total: data.news.length };
}

export function addOptionsTrades(trades: OptionsTrade[]): { added: number; total: number } {
  const today = getTodayString();
  const data = loadDailyData(today);

  const existingKeys = new Set(data.optionsTrades.map(getOptionsTradeKey));
  const newTrades = trades.filter(trade => !existingKeys.has(getOptionsTradeKey(trade)));

  data.optionsTrades.push(...newTrades);
  data.optionsTrades.sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime());

  saveDailyData(data);

  return { added: newTrades.length, total: data.optionsTrades.length };
}

export function addInstitutionalTrades(
  trades: InstitutionalTrade[]
): { added: number; total: number } {
  const today = getTodayString();
  const data = loadDailyData(today);

  const existingKeys = new Set(data.institutionalTrades.map(getInstitutionalTradeKey));
  const newTrades = trades.filter(trade => !existingKeys.has(getInstitutionalTradeKey(trade)));

  data.institutionalTrades.push(...newTrades);
  data.institutionalTrades.sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime());

  saveDailyData(data);

  return { added: newTrades.length, total: data.institutionalTrades.length };
}

export function getTodayData(): DailyData {
  const today = getTodayString();
  return loadDailyData(today);
}

export function getDateData(date: string): DailyData {
  return loadDailyData(date);
}

export function getAllDates(): string[] {
  if (!fs.existsSync(DATA_DIR)) {
    return [];
  }

  const files = fs.readdirSync(DATA_DIR);
  return files
    .filter(f => f.endsWith('.json'))
    .map(f => f.replace('.json', ''))
    .sort()
    .reverse(); // Most recent first
}

export function getStats() {
  const today = getTodayString();
  const data = loadDailyData(today);

  return {
    date: today,
    newsCount: data.news.length,
    optionsTradesCount: data.optionsTrades.length,
    institutionalTradesCount: data.institutionalTrades.length,
    lastUpdated: data.lastUpdated,
  };
}
