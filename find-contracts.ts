import { fetchTopstepXContracts } from './lib/topstepx';

async function findContracts() {
  const contracts = await fetchTopstepXContracts(false);

  console.log('\n=== GOLD CONTRACTS ===');
  contracts.filter(c => c.name?.includes('GC') || c.description?.toLowerCase().includes('gold'))
    .forEach(c => console.log(`${c.id} - ${c.name} - ${c.description}`));

  console.log('\n=== NASDAQ (NQ) CONTRACTS ===');
  contracts.filter(c => c.name?.includes('NQ') || c.name?.includes('MNQ') || c.description?.toLowerCase().includes('nasdaq'))
    .forEach(c => console.log(`${c.id} - ${c.name} - ${c.description}`));

  console.log('\n=== EURO CONTRACTS ===');
  contracts.filter(c => c.name?.includes('6E') || c.name?.includes('M6E') || c.description?.toLowerCase().includes('euro'))
    .forEach(c => console.log(`${c.id} - ${c.name} - ${c.description}`));
}

findContracts().catch(console.error);
