#!/bin/bash

# Integration Test Runner for Trading Agent
# Runs broker API tests with proper error handling

echo "🧪 Trading Agent Integration Tests"
echo "=================================="
echo ""

# Check if required environment variables are set
check_env() {
  local missing_vars=()

  if [ -z "$TOPSTEP_USERNAME" ]; then
    missing_vars+=("TOPSTEP_USERNAME")
  fi

  if [ -z "$TOPSTEP_PASSWORD" ]; then
    missing_vars+=("TOPSTEP_PASSWORD")
  fi

  if [ -z "$TOPSTEP_ACCOUNT_ID" ]; then
    missing_vars+=("TOPSTEP_ACCOUNT_ID")
  fi

  if [ -z "$TOPSTEP_API_KEY" ]; then
    missing_vars+=("TOPSTEP_API_KEY")
  fi

  if [ ${#missing_vars[@]} -gt 0 ]; then
    echo "⚠️  WARNING: Missing environment variables:"
    printf '%s\n' "${missing_vars[@]}"
    echo ""
    echo "Tests will run but some may fail without proper credentials."
    echo "Add these to your .env file for full testing."
    echo ""
  else
    echo "✅ All environment variables configured"
    echo ""
  fi
}

# Install test dependencies if needed
install_deps() {
  echo "📦 Checking test dependencies..."

  if ! npm list jest &>/dev/null; then
    echo "Installing Jest..."
    npm install --save-dev jest @types/jest ts-jest
  fi

  if ! npm list @microsoft/signalr &>/dev/null; then
    echo "Installing SignalR..."
    npm install @microsoft/signalr
  fi

  echo "✅ Dependencies ready"
  echo ""
}

# Run the tests
run_tests() {
  echo "🏃 Running integration tests..."
  echo ""

  # Check market status
  CURRENT_HOUR=$(date +%H)
  CURRENT_DAY=$(date +%u)

  if [[ $CURRENT_DAY -ge 6 ]] || [[ $CURRENT_DAY -eq 1 && $CURRENT_HOUR -lt 18 ]]; then
    echo "⚠️  Note: Markets are currently CLOSED"
    echo "   Some tests may have limited functionality"
    echo ""
  fi

  # Run tests with coverage
  if [ "$1" == "--watch" ]; then
    npm test -- --watch
  elif [ "$1" == "--coverage" ]; then
    npm test -- --coverage
  else
    npm test
  fi
}

# Main execution
main() {
  cd /Users/coreycosta/trading-agent

  # Load environment variables
  if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
  fi

  check_env
  install_deps
  run_tests "$1"

  EXIT_CODE=$?

  if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✅ All tests passed!"
  else
    echo ""
    echo "❌ Some tests failed. Check output above for details."
  fi

  exit $EXIT_CODE
}

# Run main function
main "$@"