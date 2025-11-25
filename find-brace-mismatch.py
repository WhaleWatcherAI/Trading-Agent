#!/usr/bin/env python3
import sys

depth = 0
with open('live-fabio-agent-playbook.ts', 'r') as f:
    for line_no, line in enumerate(f, 1):
        # Count braces on this line
        opens = line.count('{')
        closes = line.count('}')

        # Update depth after opens
        depth += opens

        # Check if depth goes negative after closes
        if depth - closes < 0:
            print(f"Line {line_no}: depth would go negative ({depth} - {closes} = {depth-closes})")
            print(f"  {line.rstrip()}")
            break

        depth -= closes

        # Show lines that bring depth to negative
        if depth < 0:
            print(f"Line {line_no}: depth={depth} | {line.rstrip()}")
            if line_no > 320:
                break

print(f"\nFinal depth: {depth}")
