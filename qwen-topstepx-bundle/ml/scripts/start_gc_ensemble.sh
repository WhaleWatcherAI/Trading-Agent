#!/bin/bash
# GC (Gold Full) Ensemble Launcher
export TOPSTEPX_CONTRACT_ID="CON.F.US.GC.G25"
export POINT_VALUE=100
cd /home/costa/Trading-Agent/qwen-topstepx-bundle
python3 -u ml/scripts/no_whale_regime_live.py
