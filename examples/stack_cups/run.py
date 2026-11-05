#!/usr/bin/env python3
"""Module CLI entry wrapper for ``python -m examples.stack_cups.run``."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stack_cups import main

if __name__ == "__main__":
    raise SystemExit(main())

