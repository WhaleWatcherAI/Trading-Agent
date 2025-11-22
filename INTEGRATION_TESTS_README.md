# Integration Tests for Trading Agent

## Overview
Comprehensive integration tests for broker APIs have been created to ensure reliable connections and operations with TopStepX and other trading platforms.

## Files Created

### 1. Test Suite
**File:** `tests/broker-integration.test.ts`
- Complete integration test suite for TopStepX broker
- Tests WebSocket connections, authentication, market data, order management
- Includes reconnection logic testing with the new reconnection manager
- Mock trading scenarios with entry conditions and risk management

### 2. Test Runner
**File:** `tests/run-integration-tests.sh`
- Automated test runner script
- Checks environment variables
- Installs dependencies automatically
- Detects market hours and warns if markets are closed

### 3. Jest Configuration
**File:** `jest.config.js`
- TypeScript support via ts-jest
- Code coverage reporting
- Sequential test execution for broker connections
- 30-second timeout for network operations

### 4. Test Setup
**File:** `tests/setup.ts`
- Environment variable loading
- Global mock configuration
- Test timeout settings

## Test Coverage

### Connection Management
- ✅ WebSocket connection establishment
- ✅ Authentication with credentials
- ✅ Reconnection with exponential backoff
- ✅ Duplicate connection prevention
- ✅ Max reconnection attempts enforcement

### Market Data
- ✅ Market data subscription
- ✅ Real-time data handling (when markets open)

### Order Management
- ✅ Order parameter validation
- ✅ Bracket order structure
- ✅ Order rejection handling

### Position Management
- ✅ Position state tracking
- ✅ PnL calculations
- ✅ Position reconciliation

### Error Handling
- ✅ Network disconnection handling
- ✅ API rate limiting
- ✅ Environment variable validation

### Risk Management
- ✅ Position size calculations
- ✅ Maximum drawdown enforcement
- ✅ Partial fill handling

## Running the Tests

### Quick Start
```bash
# Run all integration tests
npm run test:integration

# Run with Jest directly
npm test

# Watch mode for development
npm run test:watch

# Generate coverage report
npm run test:coverage
```

### Manual Execution
```bash
# Make script executable (one time)
chmod +x tests/run-integration-tests.sh

# Run tests
./tests/run-integration-tests.sh

# Run with coverage
./tests/run-integration-tests.sh --coverage

# Run in watch mode
./tests/run-integration-tests.sh --watch
```

## Environment Variables Required

Add these to your `.env` file:
```env
TOPSTEP_USERNAME=your_username
TOPSTEP_PASSWORD=your_password
TOPSTEP_ACCOUNT_ID=your_account_id
TOPSTEP_API_KEY=your_api_key
```

## Important Notes

### Market Hours
- Tests have limited functionality when markets are closed
- Best to run during market hours (Sunday 6PM - Friday 5PM EST)
- The test runner will warn you if markets are closed

### Test Execution
- Tests run sequentially to avoid connection conflicts
- Each test has a 30-second timeout for network operations
- Console output is suppressed unless DEBUG_TESTS=true

### Reconnection Manager Integration
- Tests use the new `reconnection-manager.ts` module
- Validates exponential backoff behavior
- Ensures no infinite reconnection loops
- Maximum 3 reconnection attempts before clean shutdown

## Next Steps

1. **Run Tests Now:**
   ```bash
   npm run test:integration
   ```

2. **Continuous Integration:**
   - Add to CI/CD pipeline
   - Run before each deployment
   - Monitor test results

3. **Extend Coverage:**
   - Add more broker-specific tests
   - Test additional order types
   - Add performance benchmarks

## Troubleshooting

### Missing Dependencies
The test runner automatically installs required packages:
- jest, @types/jest, ts-jest
- @microsoft/signalr

### Test Failures
- Check if environment variables are set correctly
- Ensure markets are open for full testing
- Verify network connectivity to TopStepX

### Debug Mode
Enable verbose output:
```bash
DEBUG_TESTS=true npm test
```

## Summary
✅ Integration tests created and ready to use
✅ Comprehensive coverage of broker API operations
✅ Reconnection logic properly tested
✅ Risk management scenarios included
✅ Easy-to-use test runner with automatic setup

The integration tests are now ready to ensure your trading system's reliability!