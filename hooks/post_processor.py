#!/usr/bin/env python3
"""Post-hook stub: passthrough JSON. Wire Stop/Session hooks here when supported."""

from __future__ import annotations

import json
import sys


def main() -> int:
    try:
        json.load(sys.stdin)
    except Exception:
        print(json.dumps({}), file=sys.stdout)
        return 0
    print(json.dumps({}), file=sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
