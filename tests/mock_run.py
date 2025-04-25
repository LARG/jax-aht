#!/usr/bin/env python3
import sys
import random
import time

# Simulate some work
time.sleep(random.uniform(0.1, 0.5))

# Randomly succeed or fail (80% success rate)
if random.random() < 0.8:
    print(f"Mock success for task: {sys.argv[1]}")
    sys.exit(0)
else:
    print(f"Mock failure for task: {sys.argv[1]}", file=sys.stderr)
    sys.exit(1) 