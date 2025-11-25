console.log('Starting Gold import test...');

(async () => {
  try {
    console.log('About to import Gold agent...');
    await import('./live-fabio-agent-playbook-mgc.ts');
    console.log('Import successful');
  } catch (error) {
    console.error('Import failed:', error);
    console.error('Stack:', error.stack);
    process.exit(1);
  }

  console.log('Test complete');
  setTimeout(() => process.exit(0), 5000);
})();
