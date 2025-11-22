#!/usr/bin/env tsx
/**
 * Test the normalizePrice function to ensure it handles tick sizes correctly
 */

function normalizePrice(price: number, tickSize: number): number {
  if (!Number.isFinite(price)) {
    return price;
  }
  const normalized = Math.round(price / tickSize) * tickSize;

  // Fix floating point precision errors by rounding to appropriate decimal places
  // Determine decimal places needed for the tick size
  // For 0.25: need 2 places, for 0.1: need 1 place, for 1: need 0 places
  // Handle scientific notation by using toFixed with a large number first
  const tickStr = tickSize < 1 ? tickSize.toFixed(10).replace(/0+$/, '') : tickSize.toString();
  const decimalIndex = tickStr.indexOf('.');
  const decimalPlaces = decimalIndex >= 0 ? tickStr.length - decimalIndex - 1 : 0;

  console.log(`  tickSize=${tickSize} → tickStr="${tickStr}" → decimalPlaces=${decimalPlaces}`);

  return Number(normalized.toFixed(decimalPlaces));
}

console.log('Testing normalizePrice function:\n');

const tests = [
  { price: 24217.75, tickSize: 0.25, expected: 24217.75 },
  { price: 24217.74, tickSize: 0.25, expected: 24217.75 },
  { price: 24217.76, tickSize: 0.25, expected: 24217.75 },
  { price: 24217.80, tickSize: 0.25, expected: 24217.75 },
  { price: 4071.4, tickSize: 0.1, expected: 4071.4 },
  { price: 4071.43, tickSize: 0.1, expected: 4071.4 },
  { price: 4071.47, tickSize: 0.1, expected: 4071.5 },
  { price: 1.12345, tickSize: 0.00001, expected: 1.12345 },
  { price: 1.123456, tickSize: 0.00001, expected: 1.12346 },
];

let passed = 0;
let failed = 0;

tests.forEach(test => {
  const result = normalizePrice(test.price, test.tickSize);
  const matches = Math.abs(result - test.expected) < 0.000001;

  if (matches) {
    console.log(`✅ normalizePrice(${test.price}, ${test.tickSize}) = ${result} (expected ${test.expected})`);
    passed++;
  } else {
    console.log(`❌ normalizePrice(${test.price}, ${test.tickSize}) = ${result} (expected ${test.expected})`);
    failed++;
  }
});

console.log(`\n${'='.repeat(60)}`);
console.log(`✅ Passed: ${passed}`);
console.log(`❌ Failed: ${failed}`);
console.log(`📊 Total: ${tests.length}`);
console.log(`${'='.repeat(60)}`);

process.exit(failed > 0 ? 1 : 0);
