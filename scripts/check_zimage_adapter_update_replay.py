#!/usr/bin/env python3
"""ZImage adapter update-bearing readiness gate."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_update_bearing_readiness import main


if __name__ == "__main__":
    sys.argv = [sys.argv[0], "zimage", *sys.argv[1:]]
    raise SystemExit(main())
