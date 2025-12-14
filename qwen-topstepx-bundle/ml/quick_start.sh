#!/bin/bash

# Quick Start Script for ML Trading Models
# This script sets up and trains LSTM and PPO models using Alpaca data

echo "=================================================="
echo "ML TRADING MODELS - QUICK START"
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo -e "${RED}Error: Please run this script from the ml/ directory${NC}"
    exit 1
fi

# Step 1: Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}Creating Python virtual environment...${NC}"
    python3 -m venv .venv
fi

# Step 2: Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source .venv/bin/activate

# Step 3: Install requirements
echo -e "${YELLOW}Installing required packages...${NC}"
pip install --upgrade pip
pip install -r requirements.txt

# Step 4: Check if .env file exists
if [ ! -f "../.env" ]; then
    echo -e "${RED}Error: .env file not found in parent directory${NC}"
    echo "Please ensure your Alpaca API keys are in the .env file"
    exit 1
fi

# Step 5: Collect data from Alpaca
echo -e "${GREEN}Step 1: Collecting market data from Alpaca...${NC}"
python3 scripts/alpaca_data_collector.py --symbol NQ --days 90 --trades 1000

# Check if data collection was successful
if [ $? -ne 0 ]; then
    echo -e "${RED}Data collection failed. Please check your Alpaca API credentials.${NC}"
    exit 1
fi

# Step 6: Build dataset
echo -e "${GREEN}Step 2: Building ML dataset...${NC}"
python3 scripts/build_dataset.py

# Step 7: Train models (optional based on user input)
echo ""
echo -e "${YELLOW}Data collection complete! Would you like to train the models now?${NC}"
echo "This will train:"
echo "  1. LightGBM (fast - 2-5 minutes)"
echo "  2. LSTM (medium - 10-30 minutes)"
echo "  3. PPO (slow - 1-2 hours)"
echo ""
read -p "Train all models now? (y/n): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Train LightGBM
    echo -e "${GREEN}Training LightGBM model...${NC}"
    python3 scripts/train_meta_label.py

    # Train LSTM
    echo -e "${GREEN}Training LSTM model...${NC}"
    python3 scripts/train_lstm_model.py

    # Train PPO (this takes longer)
    echo -e "${GREEN}Training PPO model (this may take a while)...${NC}"
    python3 scripts/train_ppo_agent.py

    # Test the ensemble
    echo -e "${GREEN}Testing ensemble predictions...${NC}"
    python3 example_usage.py

    echo -e "${GREEN}✅ All models trained successfully!${NC}"
else
    echo -e "${YELLOW}You can train models later with:${NC}"
    echo "  python3 scripts/train_meta_label.py  # LightGBM"
    echo "  python3 scripts/train_lstm_model.py  # LSTM"
    echo "  python3 scripts/train_ppo_agent.py   # PPO"
fi

echo ""
echo "=================================================="
echo "SETUP COMPLETE!"
echo "=================================================="
echo ""
echo "Next steps:"
echo "1. Review the models in ml/models/"
echo "2. Check training metrics in ml/models/*.json"
echo "3. Use predict_advanced.py for live predictions"
echo "4. Read TASK_ALLOCATION.md for integration guide"
echo ""
echo "To make live predictions:"
echo "  echo '{\"features\": {...}}' | python3 scripts/predict_advanced.py"
echo ""
echo "Happy trading! 🚀"