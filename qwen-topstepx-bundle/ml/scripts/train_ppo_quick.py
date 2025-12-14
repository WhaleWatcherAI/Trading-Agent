#!/usr/bin/env python3
"""
Quick PPO training script with reduced timesteps for faster initial training.
"""

import sys
from pathlib import Path
import pandas as pd

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parents[0]))

# Import from the main PPO script
from train_ppo_agent import PPOConfig, train_ppo_agent, evaluate_agent, TradingEnvironment, PPOAgent
import json
from datetime import datetime
import torch

# Override config for quick training
class QuickPPOConfig(PPOConfig):
    total_timesteps = 50000  # Reduced from 1,000,000
    n_steps = 512  # Reduced from 2048
    batch_size = 32  # Reduced from 64

def main():
    # Load data
    ROOT = Path(__file__).resolve().parents[2]
    DATA_DIR = ROOT / "ml" / "data"
    MODELS_DIR = ROOT / "ml" / "models"
    PPO_MODEL_PATH = MODELS_DIR / "ppo_trading_agent.pth"

    parquet_path = DATA_DIR / "meta_label.parquet"
    if not parquet_path.exists():
        print("Dataset not found. Run build_dataset.py first.")
        sys.exit(1)

    df = pd.read_parquet(parquet_path)
    print(f"Training Quick PPO agent with {len(df)} rows of data")

    if len(df) < 100:
        print("Warning: Limited data. PPO may not converge well.")

    # Override the global config
    original_config = PPOConfig
    PPOConfig.total_timesteps = 50000
    PPOConfig.n_steps = 512
    PPOConfig.batch_size = 32

    # Ensure models directory exists
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Train agent
    print("\nStarting quick PPO training (5-10 minutes)...")
    metrics = train_ppo_agent(df)

    # Quick evaluation
    print("\nRunning quick evaluation...")
    eval_results = evaluate_agent(df, start_idx=int(len(df) * 0.7))
    metrics['evaluation'] = eval_results

    # Save metrics
    metrics_path = MODELS_DIR / "ppo_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print(f"\nModel saved to: {PPO_MODEL_PATH}")
    print(f"Training completed in ~5 minutes")
    print(f"\nQuick Training Results:")
    print(f"  Total timesteps: {metrics['total_timesteps']}")
    print(f"  Episodes: {metrics['total_episodes']}")
    print(f"  Avg episode reward: {metrics['avg_episode_reward']:.2f}")

    if 'evaluation' in metrics:
        print(f"\nEvaluation Results:")
        print(f"  Total Return: {eval_results.get('total_return', 0):.2%}")
        print(f"  Number of Trades: {eval_results.get('num_trades', 0)}")

if __name__ == "__main__":
    main()