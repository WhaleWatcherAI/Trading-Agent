/**
 * Jest Test Setup
 * Configure test environment and global mocks
 */

import dotenv from 'dotenv';

// Load test environment variables
dotenv.config({ path: '.env.test' });
dotenv.config(); // Fallback to regular .env

// Set test timeout for network operations
jest.setTimeout(30000);

// Mock timers for testing reconnection logic
global.setTimeout = jest.fn(setTimeout) as any;
global.setInterval = jest.fn(setInterval) as any;

// Suppress console output during tests unless debugging
if (process.env.DEBUG_TESTS !== 'true') {
  global.console = {
    ...console,
    log: jest.fn(),
    error: jest.fn(),
    warn: jest.fn(),
    info: jest.fn(),
    debug: jest.fn(),
  };
}

// Clean up after all tests
afterAll(() => {
  jest.clearAllMocks();
  jest.clearAllTimers();
});