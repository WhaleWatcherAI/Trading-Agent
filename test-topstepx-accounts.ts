import dotenv from 'dotenv';
import { fetchTopstepXAccounts } from './lib/topstepx';

dotenv.config();

async function testAccounts() {
  console.log('🔍 Fetching TopstepX Accounts...\n');

  try {
    console.log('📋 Your Trading Accounts:');
    const accounts = await fetchTopstepXAccounts(true); // true = only active

    console.log(`Found ${accounts.length} active account(s):\n`);

    accounts.forEach(account => {
      console.log(`  Account ID: ${account.id}`);
      console.log(`  Name: ${account.name}`);
      console.log(`  Balance: $${account.balance}`);
      console.log(`  Can Trade: ${account.canTrade ? '✅ Yes' : '❌ No'}`);
      console.log(`  Visible: ${account.isVisible ? 'Yes' : 'No'}`);
      console.log('');
    });

    if (accounts.length === 0) {
      console.log('No active accounts found. You may need to:');
      console.log('  1. Set up a trading account on TopstepX');
      console.log('  2. Check if you have the right permissions');
      console.log('  3. Verify your API key has the correct access');
    } else {
      const tradableAccount = accounts.find(a => a.canTrade);
      if (tradableAccount) {
        console.log(`\n✅ You can trade with account: ${tradableAccount.id} (${tradableAccount.name})`);
        console.log(`\nAdd this to your .env file:`);
        console.log(`TOPSTEPX_ACCOUNT_ID=${tradableAccount.id}`);
      } else {
        console.log('\n⚠️  None of your accounts have trading enabled');
      }
    }

  } catch (error: any) {
    console.error('❌ Error:', error.message);
    if (error.response) {
      console.error('Response:', JSON.stringify(error.response.data, null, 2));
    }
  }
}

testAccounts().catch(console.error);
