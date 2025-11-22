#!/usr/bin/env tsx
import 'dotenv/config';
import { createProjectXRest } from './projectx-rest';
import { authenticate } from './lib/topstepx';

const ACCOUNT_ID = 13056702;
const CONTRACT_ID = 'CON.F.US.ENQ.Z25';

async function checkPosition() {
  await authenticate();
  const rest = createProjectXRest();

  console.log('\n📊 Checking position status...\n');

  try {
    const positions = await rest.getPositions(ACCOUNT_ID);

    if (positions && positions.length > 0) {
      console.log(`✅ Found ${positions.length} position(s):\n`);
      positions.forEach((pos: any) => {
        console.log(JSON.stringify(pos, null, 2));
      });
    } else {
      console.log('⭕ No open positions');
    }
  } catch (err: any) {
    console.error('❌ Error:', err.message);
  }

  process.exit(0);
}

checkPosition();
