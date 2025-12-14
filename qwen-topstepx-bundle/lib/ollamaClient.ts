/**
 * Ollama Client - Local LLM interface compatible with existing OpenAI code
 * Uses Qwen2.5:7b for fast trading decisions
 *
 * REWRITTEN: Uses fetch + AbortController for clean request handling
 * - No zombie curl processes
 * - Proper timeout/abort handling
 * - Startup reset via `ollama stop` to clear stale state
 */

import { spawn } from 'child_process';

const OLLAMA_HOST = process.env.OLLAMA_HOST || 'http://127.0.0.1:11434';
const DEFAULT_MODEL = process.env.OLLAMA_MODEL || 'qwen2.5:7b';
const TIMEOUT_MS = 30000; // 30 second default timeout
const BACKLOG_TIMEOUT_MS = 10000; // 10 second timeout when backed up

let activeRequestId = 0;

// Mutex implementation - guarantees only one request at a time
let mutexLock: Promise<void> | null = null;

async function withMutex<T>(fn: () => Promise<T>): Promise<T> {
  // Wait for any existing lock to release
  while (mutexLock) {
    await mutexLock;
  }

  let releaseLock!: () => void;
  mutexLock = new Promise<void>((resolve) => {
    releaseLock = resolve;
  });

  try {
    return await fn();
  } finally {
    releaseLock();
    mutexLock = null;
  }
}

// Track active AbortController for cancellation
let activeController: AbortController | null = null;

export interface OllamaGenerateRequest {
  model: string;
  prompt: string;
  stream?: boolean;
  temperature?: number;
  top_p?: number;
  num_predict?: number;
  stop?: string[];
}

export interface OllamaGenerateResponse {
  model: string;
  response: string;
  done: boolean;
  total_duration?: number;
  load_duration?: number;
  prompt_eval_count?: number;
  eval_count?: number;
  eval_duration?: number;
}

export interface OllamaChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

export interface OllamaChatRequest {
  model: string;
  messages: OllamaChatMessage[];
  stream?: boolean;
  temperature?: number;
  top_p?: number;
  timeoutMs?: number; // Optional timeout override
}

export interface OllamaChatResponse {
  model: string;
  message: OllamaChatMessage;
  done: boolean;
  total_duration?: number;
  eval_count?: number;
}

/**
 * Soft reset Ollama on startup - stops model and verifies server is online
 * This clears any in-progress generations from previous sessions
 * Call this once during app initialization
 */
export async function resetOllamaOnStartup(): Promise<void> {
  console.log('🔄 [Ollama] Soft reset on startup...');

  // Helper to run a command with timeout
  const runCommand = (cmd: string, args: string[], timeoutMs: number = 10000): Promise<{ code: number | null; stdout: string; stderr: string }> => {
    return new Promise((resolve) => {
      const proc = spawn(cmd, args, { stdio: 'pipe' });
      let stdout = '';
      let stderr = '';

      const timeout = setTimeout(() => {
        proc.kill();
        resolve({ code: null, stdout, stderr: stderr + ' (killed by timeout)' });
      }, timeoutMs);

      proc.stdout?.on('data', (data) => { stdout += data.toString(); });
      proc.stderr?.on('data', (data) => { stderr += data.toString(); });

      proc.on('exit', (code) => {
        clearTimeout(timeout);
        resolve({ code, stdout, stderr });
      });

      proc.on('error', (err) => {
        clearTimeout(timeout);
        resolve({ code: -1, stdout, stderr: err.message });
      });
    });
  };

  // Step 1: Stop the model (clears any in-progress generation)
  const stopResult = await runCommand('ollama', ['stop', DEFAULT_MODEL], 5000);
  if (stopResult.code === 0) {
    console.log(`  ✅ Model ${DEFAULT_MODEL} stopped`);
  } else {
    console.log(`  ℹ️ Model stop: code=${stopResult.code} (may not have been running)`);
  }

  // Step 2: Small delay to let Ollama clean up
  await new Promise(r => setTimeout(r, 500));

  // Step 3: Verify Ollama is online
  try {
    const response = await fetch(`${OLLAMA_HOST}/api/tags`, {
      method: 'GET',
      signal: AbortSignal.timeout(5000)
    });
    if (response.ok) {
      console.log('  ✅ Ollama server online');
    } else {
      console.log(`  ⚠️ Ollama server returned ${response.status}`);
    }
  } catch (e) {
    console.log(`  ⚠️ Could not reach Ollama server: ${(e as Error).message}`);
  }

  console.log('✅ [Ollama] Reset complete');
}

/**
 * Cancel the currently active Ollama request (if any)
 */
export function cancelActiveRequest(): void {
  if (activeController) {
    console.log('🛑 [Ollama] Aborting active request');
    activeController.abort();
    activeController = null;
  }
}

/**
 * Make HTTP request to Ollama using fetch + AbortController
 * Clean abort handling - timeout covers ENTIRE request including body read
 */
async function makeRequest<T>(
  endpoint: string,
  body: object,
  timeoutMs: number = TIMEOUT_MS,
  trackAsActive: boolean = false
): Promise<T> {
  return withMutex(async () => {
    const startTime = Date.now();
    const requestId = ++activeRequestId;
    const url = `${OLLAMA_HOST}${endpoint}`;

    // Cancel any previous active request if this is a tracked one
    if (trackAsActive && activeController) {
      console.log('🛑 [Ollama] Aborting previous request before starting new one');
      activeController.abort();
      activeController = null;
    }

    // Create new AbortController for this request
    const controller = new AbortController();
    if (trackAsActive) {
      activeController = controller;
    }

    // Setup timeout - stays armed until ENTIRE request completes (including body read)
    const timeout = setTimeout(() => {
      console.log(`🔴 [Ollama] #${requestId} TIMEOUT after ${timeoutMs}ms - aborting`);
      controller.abort();
    }, timeoutMs);

    console.log(`🔵 [Ollama] #${requestId} Starting fetch to ${endpoint} (timeout: ${timeoutMs}ms)`);

    try {
      const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
        signal: controller.signal,
      });

      // DON'T clear timeout here - keep it armed during body read!
      const elapsedHeaders = Date.now() - startTime;

      if (!response.ok) {
        const errorText = await response.text().catch(() => '');
        console.log(`❌ [Ollama] #${requestId} HTTP ${response.status} after ${elapsedHeaders}ms: ${errorText.slice(0, 200)}`);
        throw new Error(`Ollama HTTP ${response.status}: ${errorText.slice(0, 200)}`);
      }

      // Read and parse response - timeout still armed here!
      const text = await response.text();
      const elapsedTotal = Date.now() - startTime;

      // HARD FAIL on empty response
      if (!text || text.length === 0) {
        console.log(`❌ [Ollama] #${requestId} EMPTY RESPONSE (0 bytes) after ${elapsedTotal}ms`);
        throw new Error('Ollama returned empty response (0 bytes) - model may be stuck');
      }

      console.log(`✅ [Ollama] #${requestId} completed in ${elapsedTotal}ms (data: ${text.length} bytes)`);

      // Parse JSON
      let parsed: T;
      try {
        parsed = JSON.parse(text);
      } catch (e) {
        console.log(`❌ [Ollama] #${requestId} Failed to parse JSON: ${text.substring(0, 200)}`);
        throw new Error(`Failed to parse Ollama response: ${text.substring(0, 500)}`);
      }

      // Validate response has content (for chat responses)
      const anyParsed = parsed as any;
      if (anyParsed.message && !anyParsed.message.content) {
        console.log(`❌ [Ollama] #${requestId} Response has empty message.content`);
        throw new Error('Ollama returned empty message content');
      }

      return parsed;

    } catch (err: any) {
      const elapsed = Date.now() - startTime;

      if (err.name === 'AbortError') {
        console.log(`🔴 [Ollama] #${requestId} Aborted after ${elapsed}ms`);
        throw new Error(`Ollama timeout after ${timeoutMs}ms`);
      }

      // Log and re-throw other errors
      console.log(`❌ [Ollama] #${requestId} Error after ${elapsed}ms: ${String(err.message || err)}`);
      throw err;
    } finally {
      // ALWAYS clear timeout and controller in finally block
      clearTimeout(timeout);
      if (trackAsActive && activeController === controller) {
        activeController = null;
      }
    }
  });
}

/**
 * Generate completion using Ollama /api/generate
 */
export async function generate(
  prompt: string,
  options: Partial<OllamaGenerateRequest> = {}
): Promise<OllamaGenerateResponse> {
  const request: OllamaGenerateRequest = {
    model: options.model || DEFAULT_MODEL,
    prompt,
    stream: false,
    temperature: options.temperature ?? 0.3,
    top_p: options.top_p ?? 0.9,
    ...options,
  };

  return makeRequest<OllamaGenerateResponse>('/api/generate', request);
}

/**
 * Chat completion using Ollama /api/chat
 */
export async function chat(
  messages: OllamaChatMessage[],
  options: Partial<OllamaChatRequest> = {}
): Promise<OllamaChatResponse> {
  const timeoutMs = options.timeoutMs ?? TIMEOUT_MS;
  const request: OllamaChatRequest = {
    model: options.model || DEFAULT_MODEL,
    messages,
    stream: false,
    temperature: options.temperature ?? 0.3,
    top_p: options.top_p ?? 0.9,
    ...options,
  };

  return makeRequest<OllamaChatResponse>('/api/chat', request, timeoutMs);
}

/**
 * Chat completion that cancels any previous active request
 * Use this for trading analysis to prevent queue backup
 */
export async function chatCancellable(
  messages: OllamaChatMessage[],
  options: Partial<OllamaChatRequest> = {}
): Promise<OllamaChatResponse> {
  const timeoutMs = options.timeoutMs ?? TIMEOUT_MS;
  const request: OllamaChatRequest = {
    model: options.model || DEFAULT_MODEL,
    messages,
    stream: false,
    temperature: options.temperature ?? 0.3,
    top_p: options.top_p ?? 0.9,
    ...options,
  };

  // This will automatically cancel any previous request
  return makeRequest<OllamaChatResponse>('/api/chat', request, timeoutMs, true);
}

/**
 * Get the backlog timeout value for use in callers
 */
export function getBacklogTimeoutMs(): number {
  return BACKLOG_TIMEOUT_MS;
}

/**
 * Check if Ollama is running and model is available
 * Uses a quick warmup request
 */
export async function checkHealth(): Promise<{ ok: boolean; model: string; error?: string }> {
  try {
    const response = await chat(
      [{ role: 'user', content: 'ping' }],
      { timeoutMs: 60000 } // Give warmup longer timeout
    );
    return { ok: true, model: DEFAULT_MODEL };
  } catch (e) {
    return { ok: false, model: DEFAULT_MODEL, error: (e as Error).message };
  }
}

/**
 * OpenAI-compatible wrapper for easier migration
 * Mimics the OpenAI chat.completions.create interface
 */
export const ollamaOpenAICompat = {
  chat: {
    completions: {
      create: async (params: {
        model?: string;
        messages: Array<{ role: string; content: string }>;
        temperature?: number;
        max_tokens?: number;
        response_format?: { type: string };
      }): Promise<{
        choices: Array<{
          message: { role: string; content: string };
          finish_reason: string;
        }>;
        usage?: { prompt_tokens: number; completion_tokens: number; total_tokens: number };
      }> => {
        const messages: OllamaChatMessage[] = params.messages.map((m) => ({
          role: m.role as 'system' | 'user' | 'assistant',
          content: m.content,
        }));

        const response = await chat(messages, {
          model: params.model || DEFAULT_MODEL,
          temperature: params.temperature,
        });

        return {
          choices: [
            {
              message: {
                role: response.message.role,
                content: response.message.content,
              },
              finish_reason: 'stop',
            },
          ],
          usage: {
            prompt_tokens: response.eval_count || 0,
            completion_tokens: response.eval_count || 0,
            total_tokens: (response.eval_count || 0) * 2,
          },
        };
      },
    },
  },
};

export default {
  generate,
  chat,
  chatCancellable,
  cancelActiveRequest,
  checkHealth,
  resetOllamaOnStartup,
  openai: ollamaOpenAICompat,
};
