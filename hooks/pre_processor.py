#!/usr/bin/env python3
"""Single pre-hook entrypoint: prompt guard + hygiene (deterministic order)."""

from __future__ import annotations

from prompt_guard_lib import main

if __name__ == "__main__":
    raise SystemExit(main())
