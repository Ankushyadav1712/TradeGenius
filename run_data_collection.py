#!/usr/bin/env python3
"""
Quick script to run data collection for Phase 1.
This is a convenience script that runs the data collector.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_collector import main

if __name__ == "__main__":
    print("=" * 60)
    print("Phase 1: Data Collection")
    print("=" * 60)
    print()
    main()

