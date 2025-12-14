#!/usr/bin/env python3
"""
Live monitoring of training progress
"""
import re
import time
import subprocess
import sys

def monitor_training():
    """Monitor training progress in real-time"""
    print("\n" + "="*70)
    print("📊 LIVE TRAINING MONITOR")
    print("="*70)

    last_step = 0
    last_time = time.time()
    stuck_counter = 0

    # Monitor the most recent log
    cmd = "tail -f /tmp/ppo_training.log 2>/dev/null || echo 'No log file'"

    try:
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )

        print("\n⏱️ Monitoring training progress...")
        print("   (Press Ctrl+C to stop)\n")

        for line in iter(process.stdout.readline, ''):
            if not line:
                break

            # Extract step number from progress bar
            match = re.search(r'Training:\s+\d+%.*?(\d+)/100000', line)
            if match:
                current_step = int(match.group(1))
                current_time = time.time()

                if current_step > last_step:
                    # Calculate speed
                    time_diff = current_time - last_time
                    step_diff = current_step - last_step

                    if time_diff > 0:
                        speed = step_diff / time_diff
                        eta = (100000 - current_step) / speed if speed > 0 else 0

                        # Clear line and print update
                        sys.stdout.write('\r' + ' '*100 + '\r')
                        sys.stdout.write(
                            f"📈 Step: {current_step:,}/100,000 | "
                            f"Speed: {speed:.1f} it/s | "
                            f"Progress: {current_step/1000:.1f}% | "
                            f"ETA: {eta/60:.1f} min"
                        )
                        sys.stdout.flush()

                    last_step = current_step
                    last_time = current_time
                    stuck_counter = 0
                else:
                    # Check if stuck
                    if current_time - last_time > 30:  # No progress for 30 seconds
                        stuck_counter += 1
                        if stuck_counter % 10 == 0:
                            print(f"\n⚠️ WARNING: No progress for {int(current_time - last_time)} seconds at step {current_step}")

    except KeyboardInterrupt:
        print("\n\n✋ Monitoring stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        if process:
            process.terminate()

if __name__ == "__main__":
    monitor_training()