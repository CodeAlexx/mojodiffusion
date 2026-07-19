#!/usr/bin/env python3
"""Compatibility wrapper for the SerenityMojo offload block-profile guard."""

from __future__ import annotations

import runpy
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
TARGET = REPO / "serenitymojo/offload/check_rust_block_profile_contract.py"


if __name__ == "__main__":
    runpy.run_path(str(TARGET), run_name="__main__")
