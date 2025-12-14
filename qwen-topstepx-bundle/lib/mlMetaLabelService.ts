import { spawnSync } from 'child_process';
import path from 'path';
import { MlFeatureSnapshot } from './mlFeatureExtractor';

export interface MetaLabelScores {
  p_win_5m?: number;
  p_win_30m?: number;
  modelVersion?: string;
}

function parseNumber(value: any): number | undefined {
  const num = typeof value === 'string' ? Number(value) : value;
  return Number.isFinite(num) ? Number(num) : undefined;
}

export function predictMetaLabel(snapshot: MlFeatureSnapshot): MetaLabelScores | null {
  const pythonBin = process.env.ML_PYTHON_BIN || 'python3';
  const scriptPath = path.resolve(__dirname, '..', 'ml', 'scripts', 'predict_meta_label.py');

  const proc = spawnSync(pythonBin, [scriptPath], {
    input: JSON.stringify(snapshot),
    encoding: 'utf-8',
  });

  if (proc.error) {
    console.warn('[ML] Failed to spawn python:', proc.error.message);
    return null;
  }
  if (proc.status !== 0) {
    const err = proc.stderr?.toString()?.trim();
    if (err) {
      console.warn('[ML] predict_meta_label stderr:', err);
    }
    return null;
  }

  const out = proc.stdout?.toString()?.trim();
  if (!out) return null;

  try {
    const parsed = JSON.parse(out);
    return {
      p_win_5m: parseNumber(parsed.p_win_5m),
      p_win_30m: parseNumber(parsed.p_win_30m),
    };
  } catch (error) {
    console.warn('[ML] Failed to parse predict_meta_label output:', error);
    return null;
  }
}
