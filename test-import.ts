console.log('Starting import test...');

(async () => {
  try {
    console.log('About to import...');
    await import('./live-fabio-agent-playbook.ts');
    console.log('Import successful');
  } catch (error) {
    console.error('Import failed:', error);
    console.error('Stack:', error.stack);
    process.exit(1);
  }

  console.log('Test complete');
})();
