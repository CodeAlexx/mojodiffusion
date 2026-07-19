#!/usr/bin/env python
"""Launch the Mojo side of the ltx2 trainer fwd parity gate with argv built
from pairs.txt (robust to spaces in paths — no shell quoting involved)."""
import os
import subprocess
import sys

OUT = sys.argv[1] if len(sys.argv) > 1 else "/home/alex/mojodiffusion/output/ltx2_parity_fwd"
PROBE = sys.argv[2] if len(sys.argv) > 2 else "/tmp/ltx2_fwd_parity_probe"
CKPT = "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-dev-fp8.safetensors"

argv = [PROBE, os.path.join(OUT, "fixture.safetensors"), CKPT,
        os.path.join(OUT, "mojo_out.safetensors")]
for ln in open(os.path.join(OUT, "pairs.txt")):
    ln = ln.rstrip("\n")
    if not ln:
        continue
    arm, rest = ln.split(" ", 1)
    sigma = rest.rsplit(" ", 1)[1]
    mid = rest.rsplit(" ", 1)[0]
    cut = mid.index(".safetensors ") + len(".safetensors")
    lat, te = mid[:cut], mid[cut + 1:]
    argv += [arm, lat, te, sigma]

print("launching probe with", (len(argv) - 4) // 4, "pairs")
sys.exit(subprocess.call(argv))
