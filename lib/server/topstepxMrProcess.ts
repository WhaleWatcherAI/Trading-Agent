import { spawn, ChildProcessWithoutNullStreams } from 'child_process';
import { existsSync } from 'fs';
import path from 'path';

interface StrategyLogLine {
  timestamp: string;
  type: 'stdout' | 'stderr';
  line: string;
}

export interface StrategyProcessStatus {
  running: boolean;
  pid?: number;
  startedAt?: string;
  lastExitAt?: string;
  exitCode?: number | null;
  exitSignal?: NodeJS.Signals | null;
  logs: StrategyLogLine[];
  accountId?: number | null;
}

interface MrProcessState {
  child: ChildProcessWithoutNullStreams | null;
  startedAt?: string;
  lastExitAt?: string;
  lastExitCode?: number | null;
  lastExitSignal?: NodeJS.Signals | null;
  logs: StrategyLogLine[];
  currentAccountId: number | null;
}

const globalRef = globalThis as typeof globalThis & {
  __topstepxMrState?: MrProcessState;
};

const SCRIPT_PATH = path.join(process.cwd(), 'live-topstepx-mean-reversion-1s.ts');
const MAX_LOG_LINES = 200;

if (!globalRef.__topstepxMrState) {
  globalRef.__topstepxMrState = {
    child: null,
    logs: [],
    currentAccountId: null,
  };
}

const state = globalRef.__topstepxMrState!;

const getChild = () => state.child;
const setChild = (proc: ChildProcessWithoutNullStreams | null) => {
  state.child = proc;
  globalRef.__topstepxMrState = state;
};

function appendLog(type: StrategyLogLine['type'], chunk: Buffer) {
  const lines = chunk
    .toString()
    .split(/\r?\n/)
    .filter(Boolean);
  for (const line of lines) {
    state.logs.push({ timestamp: new Date().toISOString(), type, line });
  }
  if (state.logs.length > MAX_LOG_LINES) {
    state.logs = state.logs.slice(-MAX_LOG_LINES);
  }
}

export function getTopstepxMrStatus(): StrategyProcessStatus {
  return {
    running: !!state.child,
    pid: state.child?.pid,
    startedAt: state.startedAt,
    lastExitAt: state.lastExitAt,
    exitCode: state.lastExitCode,
    exitSignal: state.lastExitSignal,
    logs: state.logs,
    accountId: state.currentAccountId,
  };
}

export function startTopstepxMrProcess(options: { accountId?: number } = {}) {
  if (state.child) {
    return { success: false, message: 'Strategy already running.' };
  }
  if (!existsSync(SCRIPT_PATH)) {
    return { success: false, message: `Strategy script not found at ${SCRIPT_PATH}` };
  }

  const npmExec = process.platform === 'win32' ? 'npx.cmd' : 'npx';
  const proc = spawn(
    npmExec,
    ['tsx', SCRIPT_PATH],
    {
      cwd: process.cwd(),
      env: {
        ...process.env,
        ...(options.accountId ? { TOPSTEPX_ACCOUNT_ID: String(options.accountId) } : {}),
      },
      stdio: ['ignore', 'pipe', 'pipe'],
    },
  );

  setChild(proc);
  state.startedAt = new Date().toISOString();
  state.lastExitAt = undefined;
  state.lastExitCode = undefined;
  state.lastExitSignal = undefined;
  state.currentAccountId =
    options.accountId ?? (process.env.TOPSTEPX_ACCOUNT_ID ? Number(process.env.TOPSTEPX_ACCOUNT_ID) : null);

  proc.stdout.on('data', chunk => appendLog('stdout', chunk));
  proc.stderr.on('data', chunk => appendLog('stderr', chunk));

  proc.on('exit', (code, signal) => {
    state.lastExitAt = new Date().toISOString();
    state.lastExitCode = code;
    state.lastExitSignal = signal;
    setChild(null);
    state.currentAccountId = null;
    appendLog('stdout', Buffer.from(`process exited (code=${code ?? 'n/a'} signal=${signal ?? 'n/a'})`));
  });

  proc.on('error', err => {
    appendLog('stderr', Buffer.from(`process error: ${err.message}`));
  });

  return { success: true, message: 'Strategy process started.', pid: proc.pid };
}

export function stopTopstepxMrProcess() {
  const runningChild = getChild();
  if (!runningChild) {
    return { success: false, message: 'Strategy is not running.' };
  }
  const stopped = runningChild.kill('SIGTERM');
  if (!stopped) {
    return { success: false, message: 'Failed to send SIGTERM to strategy process.' };
  }
  return { success: true, message: 'Stop signal sent.' };
}

export function flattenTopstepxMrPosition() {
  const runningChild = getChild();
  if (!runningChild) {
    return { success: false, message: 'Strategy is not running.' };
  }
  const ok = runningChild.kill('SIGUSR2');
  if (!ok) {
    return { success: false, message: 'Failed to send flatten signal (SIGUSR2).' };
  }
  return { success: true, message: 'Flatten signal sent.' };
}
