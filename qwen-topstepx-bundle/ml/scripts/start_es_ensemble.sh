#!/bin/bash
# ES (S&P 500 Full) Ensemble Launcher
export TOPSTEPX_CONTRACT_ID="CON.F.US.EP.Z25"
export POINT_VALUE=50
cd /home/costa/Trading-Agent/qwen-topstepx-bundle
python3 -u ml/scripts/no_whale_regime_live.py
