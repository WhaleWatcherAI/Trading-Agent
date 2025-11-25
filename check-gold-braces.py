#!/usr/bin/env python3

stack = []  # Stack to track (line_no, line_content)

with open('live-fabio-agent-playbook-mgc.ts', 'r') as f:
    for line_no, line in enumerate(f, 1):
        stripped = line.strip()

        for char in line:
            if char == '{':
                stack.append((line_no, stripped[:80]))
            elif char == '}':
                if stack:
                    stack.pop()
                else:
                    print(f"ERROR: Extra closing brace at line {line_no}: {stripped[:80]}")

print(f"\nRemaining unclosed braces: {len(stack)}")
if stack:
    print("Unclosed braces:")
    for item in stack:
        print(f"  Line {item[0]}: {item[1]}")
