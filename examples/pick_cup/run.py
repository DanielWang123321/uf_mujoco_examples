#!/usr/bin/env python3
"""Module CLI entry wrapper for ``python -m examples.pick_cup.run``."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pick_cup import main

if __name__ == "__main__":
    raise SystemExit(main())

