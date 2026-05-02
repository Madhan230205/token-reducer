#!/usr/bin/env python3
"""CLI entry point for token-reducer (``python scripts/context_pipeline.py …``).

The dominant retrieval pipeline lives in ``token_reducer.context_pipeline`` and
``token_reducer.orchestrator`` (intent → retrieve → rerank → compress → expand).
This file forwards to :mod:`token_reducer.cli` for backward compatibility.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from token_reducer.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
