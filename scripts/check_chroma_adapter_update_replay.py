#!/usr/bin/env python3
"""Chroma wrapper for the shared OneTrainer adapter update replay gate."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_adapter_update_replay import main


if __name__ == "__main__":
    sys.argv = [sys.argv[0], "chroma", *sys.argv[1:]]
    raise SystemExit(main())
