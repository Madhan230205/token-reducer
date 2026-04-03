#!/usr/bin/env python3
"""Backward-compatible entry point. Logic lives in token_reducer package."""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from token_reducer.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
