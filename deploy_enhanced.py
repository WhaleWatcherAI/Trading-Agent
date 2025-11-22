#!/usr/bin/env python3
"""
Deployment script for Enhanced Self-Learning Fabio Agent
Provides easy switching between original and enhanced versions
"""

import os
import sys
import subprocess
import time
from pathlib import Path
import shutil
from datetime import datetime


def check_requirements():
    """Check if all required files exist"""
    required_files = [
        "engine_enhanced.py",
        "features_enhanced.py",
        "llm_client_enhanced.py",
        "execution_enhanced.py",
        "config.py",
        "topstep_client.py",
        ".env"
    ]

    missing = []
    for file in required_files:
        if not Path(file).exists():
            missing.append(file)

    if missing:
        print(f"❌ Missing required files: {', '.join(missing)}")
        return False

    print("✅ All required files present")
    return True


def backup_original():
    """Backup original files before switching to enhanced"""
    backup_dir = Path("backups") / datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir.mkdir(parents=True, exist_ok=True)

    files_to_backup = [
        "engine.py",
        "features.py",
        "llm_client.py",
        "execution.py"
    ]

    for file in files_to_backup:
        if Path(file).exists():
            shutil.copy2(file, backup_dir / file)

    print(f"✅ Original files backed up to {backup_dir}")
    return backup_dir


def test_imports():
    """Test that enhanced modules can be imported"""
    try:
        import features_enhanced
        import llm_client_enhanced
        import execution_enhanced
        print("✅ Enhanced modules import successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


def check_env_vars():
    """Check if required environment variables are set"""
    from dotenv import load_dotenv
    load_dotenv()

    required = [
        "OPENAI_API_KEY",
        "TOPSTEP_USERNAME",
        "TOPSTEP_PASSWORD",
        "TOPSTEP_ACCOUNT_ID",
        "SYMBOL"
    ]

    missing = []
    for var in required:
        if not os.getenv(var):
            missing.append(var)

    if missing:
        print(f"⚠️  Missing environment variables: {', '.join(missing)}")
        print("   Please update your .env file")
        return False

    print("✅ Environment variables configured")
    return True


def run_test_mode(duration_seconds=30):
    """Run enhanced engine in test mode for a short duration"""
    print(f"\n🧪 Running enhanced engine in TEST mode for {duration_seconds} seconds...")
    print("   This will validate the system without placing real trades\n")

    # Set test mode temporarily
    os.environ["MODE"] = "test"

    # Start the enhanced engine
    proc = subprocess.Popen(
        ["python", "engine_enhanced.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    start_time = time.time()
    output_lines = []

    try:
        while time.time() - start_time < duration_seconds:
            if proc.poll() is not None:
                # Process ended early
                stdout, stderr = proc.communicate()
                if stderr:
                    print(f"❌ Engine error:\n{stderr}")
                    return False
                break

            # Check for output
            line = proc.stdout.readline()
            if line:
                print(f"   {line.rstrip()}")
                output_lines.append(line)

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")

    finally:
        if proc.poll() is None:
            proc.terminate()
            time.sleep(1)
            if proc.poll() is None:
                proc.kill()

    # Check if test was successful
    if any("Fatal error" in line for line in output_lines):
        print("\n❌ Test failed - errors detected")
        return False

    if any("Starting market data stream" in line for line in output_lines):
        print("\n✅ Test successful - engine started correctly")
        return True

    print("\n⚠️  Test inconclusive - check output above")
    return None


def deploy_enhanced(mode="test"):
    """Deploy the enhanced self-learning agent"""
    print("\n" + "="*60)
    print("🚀 ENHANCED SELF-LEARNING FABIO AGENT DEPLOYMENT")
    print("="*60)

    # Step 1: Check requirements
    if not check_requirements():
        print("\n❌ Deployment aborted - missing files")
        return False

    # Step 2: Test imports
    if not test_imports():
        print("\n❌ Deployment aborted - import errors")
        return False

    # Step 3: Check environment
    if not check_env_vars():
        print("\n⚠️  Environment not fully configured")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return False

    # Step 4: Backup original files
    backup_dir = backup_original()

    # Step 5: Run test if requested
    if mode == "test":
        print("\n" + "="*60)
        print("RUNNING TEST MODE")
        print("="*60)

        success = run_test_mode(30)
        if success:
            print("\n✅ Test passed!")
            print("\nTo run in production mode:")
            print("  python engine_enhanced.py")
            print("\nTo run with PM2:")
            print("  pm2 start engine_enhanced.py --name fabio-enhanced")
        else:
            print("\n⚠️  Test had issues - review output above")

    elif mode == "live":
        print("\n" + "="*60)
        print("STARTING LIVE MODE")
        print("="*60)
        print("\n⚠️  WARNING: This will run with real money!")
        response = input("Are you sure? Type 'yes' to continue: ")

        if response == "yes":
            print("\n🚀 Starting enhanced agent in LIVE mode...")
            subprocess.run(["python", "engine_enhanced.py"])
        else:
            print("\n❌ Live deployment cancelled")
            return False

    else:
        print(f"\n❌ Unknown mode: {mode}")
        return False

    print("\n" + "="*60)
    print("DEPLOYMENT COMPLETE")
    print("="*60)
    print(f"\n📁 Original files backed up to: {backup_dir}")
    print("\n📊 Monitor logs:")
    print("  tail -f logs/llm_exchanges.jsonl")
    print("\n🔍 Check strategy updates:")
    print("  grep strategy_updates logs/llm_exchanges.jsonl | tail -5")
    print("\n📈 View performance:")
    print("  grep strategy_performance logs/llm_exchanges.jsonl | tail -5")

    return True


def main():
    """Main entry point"""
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    else:
        print("\nDeployment Mode:")
        print("  1. Test Mode (recommended)")
        print("  2. Live Mode (real money)")
        choice = input("\nSelect mode (1 or 2): ")
        mode = "test" if choice == "1" else "live" if choice == "2" else None

        if mode is None:
            print("❌ Invalid choice")
            sys.exit(1)

    success = deploy_enhanced(mode)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()