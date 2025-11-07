// Test if option price adjustment logic is correct

console.log('═══════════════════════════════════════════════════');
console.log('  TESTING OPTION PRICE ADJUSTMENT LOGIC');
console.log('═══════════════════════════════════════════════════\n');

// Scenario 1: Entering CALL when price crosses UP through SMA
console.log('SCENARIO 1: CALL Entry');
console.log('─────────────────────────────────────────────────');
const callPremiumAtClose = 10.00;
const callDelta = 0.5;
const underlyingAtClose = 445;
const underlyingAtEntry = 444; // Entered at SMA (below close)

console.log(`Option: CALL premium at close = $${callPremiumAtClose}, delta = ${callDelta}`);
console.log(`Underlying at close: $${underlyingAtClose}`);
console.log(`Underlying at entry (SMA): $${underlyingAtEntry}`);
console.log(`\nUnderlying difference: $${underlyingAtEntry} - $${underlyingAtClose} = $${underlyingAtEntry - underlyingAtClose}`);

// Current backtest logic
const priceDiff1 = underlyingAtEntry - underlyingAtClose;
const adjustment1 = callDelta * priceDiff1;
const adjustedPremium1 = callPremiumAtClose - adjustment1;

console.log(`\nCURRENT LOGIC:`);
console.log(`  Premium adjustment: ${callDelta} × $${priceDiff1} = $${adjustment1}`);
console.log(`  Adjusted premium: $${callPremiumAtClose} - $${adjustment1} = $${adjustedPremium1}`);
console.log(`  ❓ Does this make sense?`);
console.log(`  → Entered at LOWER price ($444 vs $445)`);
console.log(`  → CALL should be CHEAPER`);
console.log(`  → But we got $${adjustedPremium1} vs $${callPremiumAtClose}`);

// Correct logic
const adjustedPremiumCorrect1 = callPremiumAtClose + adjustment1;
console.log(`\nCORRECT LOGIC:`);
console.log(`  Adjusted premium: $${callPremiumAtClose} + $${adjustment1} = $${adjustedPremiumCorrect1}`);
console.log(`  ✅ CALL is cheaper when underlying is lower!`);

// Scenario 2: Entering PUT when price crosses DOWN through SMA
console.log('\n\nSCENARIO 2: PUT Entry');
console.log('─────────────────────────────────────────────────');
const putPremiumAtClose = 10.00;
const putDelta = -0.5;
const underlyingAtClose2 = 445;
const underlyingAtEntry2 = 446; // Entered at SMA (above close)

console.log(`Option: PUT premium at close = $${putPremiumAtClose}, delta = ${putDelta}`);
console.log(`Underlying at close: $${underlyingAtClose2}`);
console.log(`Underlying at entry (SMA): $${underlyingAtEntry2}`);
console.log(`\nUnderlying difference: $${underlyingAtEntry2} - $${underlyingAtClose2} = $${underlyingAtEntry2 - underlyingAtClose2}`);

// Current backtest logic
const priceDiff2 = underlyingAtEntry2 - underlyingAtClose2;
const adjustment2 = putDelta * priceDiff2;
const adjustedPremium2 = putPremiumAtClose - adjustment2;

console.log(`\nCURRENT LOGIC:`);
console.log(`  Premium adjustment: ${putDelta} × $${priceDiff2} = $${adjustment2}`);
console.log(`  Adjusted premium: $${putPremiumAtClose} - (${adjustment2}) = $${adjustedPremium2}`);
console.log(`  ❓ Does this make sense?`);
console.log(`  → Entered at HIGHER price ($446 vs $445)`);
console.log(`  → PUT should be CHEAPER`);
console.log(`  → But we got $${adjustedPremium2} vs $${putPremiumAtClose}`);

// Correct logic
const adjustedPremiumCorrect2 = putPremiumAtClose + adjustment2;
console.log(`\nCORRECT LOGIC:`);
console.log(`  Adjusted premium: $${putPremiumAtClose} + (${adjustment2}) = $${adjustedPremiumCorrect2}`);
console.log(`  ✅ PUT is cheaper when underlying is higher!`);

console.log('\n═══════════════════════════════════════════════════');
console.log('CONCLUSION:');
console.log('─────────────────────────────────────────────────');
console.log('The formula should be:');
console.log('  adjustedPremium = premium + (delta × priceDiff)');
console.log('NOT:');
console.log('  adjustedPremium = premium - (delta × priceDiff)');
console.log('═══════════════════════════════════════════════════\n');
