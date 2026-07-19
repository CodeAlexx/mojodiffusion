#!/usr/bin/env bash
# run_oracle.sh — regenerate the Python argparse oracle output for the three
# representative inputs used by parser_test.mojo. The printed blocks must match
# the EXPECTED_* constants embedded in parser_test.mojo (newline-joined). Run
# from anywhere; uses paths relative to this script.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
oracle="$here/argparse_oracle.py"

echo "=== input1: -v --name alice --count 3 src1 dst1 ==="
python3 "$oracle" -- -v --name alice --count 3 src1 dst1
echo "=== input2: --level mid --inc a --inc b s d extra1 extra2 ==="
python3 "$oracle" -- --level mid --inc a --inc b s d extra1 extra2
echo "=== input3: -n bob s2 d2 ==="
python3 "$oracle" -- -n bob s2 d2
