#!/usr/bin/env python3
"""
Quick test script to verify enhanced Fabio agent configuration
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("=" * 60)
print("ENHANCED FABIO AGENT - CONFIGURATION TEST")
print("=" * 60)

# Check critical variables
config_status = {
    "SYMBOL": os.getenv("SYMBOL"),
    "MODE": os.getenv("MODE"),
    "ACCOUNT_BALANCE": os.getenv("ACCOUNT_BALANCE"),
    "TOPSTEP_USERNAME": os.getenv("TOPSTEP_USERNAME"),
    "TOPSTEP_PASSWORD": os.getenv("TOPSTEP_PASSWORD"),
    "TOPSTEPX_ACCOUNT_ID": os.getenv("TOPSTEPX_ACCOUNT_ID"),
    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
    "OPENAI_MODEL": os.getenv("OPENAI_MODEL"),
    "LLM_API_URL": os.getenv("LLM_API_URL"),
    "LLM_DECISION_INTERVAL_DEFAULT_SEC": os.getenv("LLM_DECISION_INTERVAL_DEFAULT_SEC"),
}

print("\n📋 Configuration Status:\n")
all_good = True

for key, value in config_status.items():
    if value and value != "YOUR_TOPSTEP_PASSWORD_HERE":
        status = "✅"
        display_value = value[:20] + "..." if len(str(value)) > 20 else value
        if key in ["TOPSTEP_PASSWORD", "OPENAI_API_KEY"]:
            display_value = "***HIDDEN***"
    else:
        status = "❌"
        display_value = "MISSING"
        all_good = False

    print(f"{status} {key}: {display_value}")

print("\n" + "=" * 60)

if not all_good:
    print("\n⚠️  ISSUES FOUND:")
    print("")

    if not config_status["TOPSTEP_PASSWORD"] or config_status["TOPSTEP_PASSWORD"] == "YOUR_TOPSTEP_PASSWORD_HERE":
        print("1. TOPSTEP_PASSWORD is not set!")
        print("   👉 Edit .env file and replace 'YOUR_TOPSTEP_PASSWORD_HERE'")
        print("      with your actual TopStep password")
        print("")

    missing = [k for k, v in config_status.items() if not v and k != "TOPSTEP_PASSWORD"]
    if missing:
        print(f"2. Missing configuration: {', '.join(missing)}")
        print("   👉 These were just added to your .env file")
        print("")
else:
    print("\n✅ All configuration looks good!")
    print("\n🚀 Ready to run the enhanced agent:")
    print("   python3 engine_enhanced.py")
    print("\n📊 Or test first:")
    print("   MODE=test python3 engine_enhanced.py")

print("\n" + "=" * 60)

# Test imports
print("\n🔍 Testing Enhanced Module Imports...")
try:
    import features_enhanced
    print("✅ features_enhanced.py loads")
except ImportError as e:
    print(f"❌ features_enhanced.py error: {e}")

try:
    import llm_client_enhanced
    print("✅ llm_client_enhanced.py loads")
except ImportError as e:
    print(f"❌ llm_client_enhanced.py error: {e}")

try:
    import execution_enhanced
    print("✅ execution_enhanced.py loads")
except ImportError as e:
    print(f"❌ execution_enhanced.py error: {e}")

try:
    import engine_enhanced
    print("✅ engine_enhanced.py loads")
except ImportError as e:
    print(f"❌ engine_enhanced.py error: {e}")

print("\n" + "=" * 60)