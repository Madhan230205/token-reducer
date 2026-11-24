#!/usr/bin/env python3
"""Stdio MCP server entry point: ``python scripts/token_reducer_mcp.py``."""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from token_reducer.mcp_server import main

if __name__ == "__main__":
    raise SystemExit(main())
