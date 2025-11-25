#!/usr/bin/env python3

stack = []  # Stack to track (line_no, line_content)

with open('live-fabio-agent-playbook.ts', 'r') as f:
    for line_no, line in enumerate(f, 1):
        stripped = line.strip()

        for char in line:
            if char == '{':
                stack.append((line_no, stripped[:60]))
            elif char == '}':
                if stack:
                    opening = stack.pop()
                    if line_no == 2363:  # The closing brace we added
                        print(f"Line 2363 closing brace matches opening at:")
                        print(f"  Line {opening[0]}: {opening[1]}")
                        break

print(f"\nRemaining unclosed braces: {len(stack)}")
if stack:
    print("Last few unclosed:")
    for item in stack[-5:]:
        print(f"  Line {item[0]}: {item[1]}")
