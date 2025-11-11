interface CommissionTable {
  [symbol: string]: number;
}

// TopstepX commission rates (per side)
const COMMISSION_PER_SIDE: CommissionTable = {
  ES: 1.40,      // $2.80 RT / 2 = $1.40 per side
  MES: 0.37,     // $0.74 RT / 2 = $0.37 per side
  NQ: 1.40,      // $2.80 RT / 2 = $1.40 per side
  MNQ: 0.37,     // $0.74 RT / 2 = $0.37 per side
  GC: 1.62,      // Assuming same as 6E (update if different)
  MGC: 1.62,     // Assuming same as 6E (update if different)
  '6E': 1.62,    // $3.24 RT / 2 = $1.62 per side
};

type Identifier = string | null | undefined;

function extractSymbolFromContract(identifier: string): string | null {
  const upper = identifier.toUpperCase();
  const parts = upper.split('.');

  if (parts.length >= 4 && parts[0] === 'CON' && parts[1] === 'F') {
    const symbolPart = parts[3];
    if (symbolPart) {
      return symbolPart;
    }
  }

  const monthCodeRegex = /^([A-Z0-9]+?)([FGHJKMNQUVXZ]\d{1,2})$/;
  const monthMatch = upper.match(monthCodeRegex);
  if (monthMatch) {
    return monthMatch[1];
  }

  const genericMatch = upper.match(/^([A-Z0-9]+)/);
  return genericMatch ? genericMatch[1] : null;
}

/**
 * Infer the per-side futures commission (in USD) for known contracts.
 * Falls back to the provided value when the symbol is not in the table.
 */
export function inferFuturesCommissionPerSide(
  identifier: Identifier | Identifier[],
  fallback: number = 0,
): number {
  const identifiers = Array.isArray(identifier) ? identifier : [identifier];

  for (const candidate of identifiers) {
    if (!candidate) continue;
    const symbol = extractSymbolFromContract(candidate);
    if (!symbol) continue;
    const commission = COMMISSION_PER_SIDE[symbol];
    if (commission != null) {
      return commission;
    }
  }

  return fallback;
}

export function getKnownFuturesCommissions(): CommissionTable {
  return { ...COMMISSION_PER_SIDE };
}
