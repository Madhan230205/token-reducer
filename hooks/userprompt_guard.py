#!/usr/bin/env python3
"""Backward-compatible alias — logic lives in prompt_guard_lib / pre_processor."""

from __future__ import annotations

from prompt_guard_lib import main

if __name__ == "__main__":
    raise SystemExit(main())
