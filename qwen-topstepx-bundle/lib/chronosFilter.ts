/**
 * Chronos-2 Pre-Filter Service
 * Uses Amazon's Chronos foundation model for time series forecasting
 * to filter trading opportunities before LLM analysis.
 */

import { spawn } from 'child_process';
import * as path from 'path';

export interface ChronosFilterResult {
  pass: boolean;
  direction: 'up' | 'down' | 'neutral';
  confidence: number;
  expected_move?: number;
  expected_move_pct?: number;
  current_price?: number;
  forecast_median?: number;
  forecast_q10?: number;
  forecast_q90?: number;
  uncertainty?: number;
  prediction_steps?: number;
  samples_up?: number;
  samples_down?: number;
  model?: string;
  symbol?: string;
  reason?: string;
  error?: string;
}

export interface ChronosFilterInput {
  prices: number[];
  symbol?: string;
  prediction_length?: number; // Default: 5
  confidence_threshold?: number; // Default: 0.55
  num_samples?: number; // Default: 20
}

// Cache the Python process for faster subsequent calls
let cachedPythonPath: string | null = null;

function getPythonPath(): string {
  if (cachedPythonPath) return cachedPythonPath;

  const mlDir = path.join(__dirname, '..', 'ml');
  const venvPython = path.join(mlDir, '.venv', 'bin', 'python3');

  // Check if venv exists, otherwise use system python
  try {
    require('fs').accessSync(venvPython);
    cachedPythonPath = venvPython;
  } catch {
    cachedPythonPath = 'python3';
  }

  return cachedPythonPath;
}

/**
 * Run Chronos filter prediction
 *
 * @param input - Price data and optional parameters
 * @param timeoutMs - Timeout in milliseconds (default 30s, first call may take longer due to model loading)
 * @returns Filter result indicating whether to proceed with LLM analysis
 */
export async function runChronosFilter(
  input: ChronosFilterInput,
  timeoutMs: number = 30000
): Promise<ChronosFilterResult> {
  return new Promise((resolve) => {
    const pythonPath = getPythonPath();
    const scriptPath = path.join(__dirname, '..', 'ml', 'scripts', 'chronos_filter.py');

    const startTime = Date.now();

    const proc = spawn(pythonPath, [scriptPath], {
      cwd: path.join(__dirname, '..', 'ml'),
      env: {
        ...process.env,
        PYTHONUNBUFFERED: '1',
      },
    });

    let stdout = '';
    let stderr = '';

    proc.stdout.on('data', (data) => {
      stdout += data.toString();
    });

    proc.stderr.on('data', (data) => {
      stderr += data.toString();
    });

    // Set timeout
    const timeout = setTimeout(() => {
      proc.kill('SIGKILL');
      resolve({
        pass: false,
        direction: 'neutral',
        confidence: 0,
        reason: 'timeout',
        error: `Chronos filter timed out after ${timeoutMs}ms`,
      });
    }, timeoutMs);

    proc.on('close', (code) => {
      clearTimeout(timeout);
      const elapsed = Date.now() - startTime;

      if (stderr && !stderr.includes('[Chronos]')) {
        console.error(`[ChronosFilter] stderr: ${stderr}`);
      }

      if (code !== 0 && !stdout.trim()) {
        resolve({
          pass: false,
          direction: 'neutral',
          confidence: 0,
          reason: 'process_error',
          error: `Process exited with code ${code}`,
        });
        return;
      }

      try {
        const result = JSON.parse(stdout.trim());
        console.log(
          `[ChronosFilter] ${result.direction.toUpperCase()} (conf: ${(result.confidence * 100).toFixed(1)}%) ` +
            `pass=${result.pass} in ${elapsed}ms`
        );
        resolve(result);
      } catch (e: any) {
        resolve({
          pass: false,
          direction: 'neutral',
          confidence: 0,
          reason: 'parse_error',
          error: `Failed to parse output: ${e.message}`,
        });
      }
    });

    proc.on('error', (err) => {
      clearTimeout(timeout);
      resolve({
        pass: false,
        direction: 'neutral',
        confidence: 0,
        reason: 'spawn_error',
        error: err.message,
      });
    });

    // Send input
    proc.stdin.write(JSON.stringify(input));
    proc.stdin.end();
  });
}

/**
 * Helper to extract prices from candle data
 */
export function extractPricesFromCandles(
  candles: Array<{ close: number } | { c: number }>
): number[] {
  return candles.map((c) => ('close' in c ? c.close : (c as any).c));
}

/**
 * Pre-filter check that returns whether to proceed with LLM analysis
 *
 * @param prices - Recent price history (recommend 30-60 data points)
 * @param symbol - Trading symbol for logging
 * @param minConfidence - Minimum confidence to pass (default 0.55)
 * @returns Object with pass decision and prediction details
 */
export async function shouldProceedWithAnalysis(
  prices: number[],
  symbol?: string,
  minConfidence: number = 0.55
): Promise<{
  proceed: boolean;
  chronosResult: ChronosFilterResult;
  suggestedDirection?: 'long' | 'short' | null;
}> {
  // Skip filter if insufficient data
  if (prices.length < 10) {
    return {
      proceed: true, // Proceed anyway, let LLM decide
      chronosResult: {
        pass: true,
        direction: 'neutral',
        confidence: 0,
        reason: 'insufficient_data_bypassed',
      },
    };
  }

  const result = await runChronosFilter({
    prices,
    symbol,
    confidence_threshold: minConfidence,
    prediction_length: 5,
    num_samples: 20,
  });

  // Map direction to suggested trade direction
  let suggestedDirection: 'long' | 'short' | null = null;
  if (result.pass) {
    if (result.direction === 'up') suggestedDirection = 'long';
    else if (result.direction === 'down') suggestedDirection = 'short';
  }

  return {
    proceed: result.pass,
    chronosResult: result,
    suggestedDirection,
  };
}
