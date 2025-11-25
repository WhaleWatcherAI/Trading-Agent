#!/usr/bin/env tsx
console.log('Step 1: File loaded');

import 'dotenv/config';
console.log('Step 2: dotenv loaded');

import { fabioPlaybook } from './lib/fabioPlaybook';
console.log('Step 3: fabioPlaybook imported');

import { authenticate } from './lib/topstepx';
console.log('Step 4: topstepx imported');

import express from 'express';
import { Server } from 'socket.io';
console.log('Step 5: express and socket.io imported');

async function main() {
  console.log('Step 6: main() called');
  console.log('Playbook:', fabioPlaybook.philosophy);

  // Try to authenticate
  console.log('Step 7: Calling authenticate()...');
  const token = await authenticate();
  console.log('Step 8: Authenticated, token length:', token.length);
}

console.log('Step 9: About to call main()');
main().catch((e) => {
  console.error('Error in main:', e);
  process.exit(1);
});
console.log('Step 10: main() called');
