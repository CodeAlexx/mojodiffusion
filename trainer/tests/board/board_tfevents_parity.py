#!/usr/bin/env python3
"""SerenityBoard <-> TensorBoard VALUE parity gate (MJ/board, 2026-07-06).

Imports a real TensorBoard run (default: the SerenityTrainer alina_zimage 766-step
segment) via scripts/board_import_tfevents.py, then asserts EVERY (step,value)
served by the live board API equals the tfevents record at f32 precision
(events store float32; the board stores the exact f64 upcast — the comparison
re-rounds the served value to f32 and requires equality).

Run with a python that has tensorboard (e.g. SerenityTrainer's venv):
  /home/alex/SerenityTrainer/venv/bin/python tests/board/board_tfevents_parity.py [tb_run_dir] [run_name]
Requires the webui up on :8188.
"""
import json, struct, subprocess, sys, os, urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUN_DIR = sys.argv[1] if len(sys.argv) > 1 else \
    "/home/alex/SerenityTrainer/workspace/alina_zimage_serenity_preset_2000/tensorboard/2026-05-24_16-14-47"
RUN = sys.argv[2] if len(sys.argv) > 2 else "serenity_zimage_alina"

subprocess.run([sys.executable, os.path.join(REPO, "scripts", "board_import_tfevents.py"),
                RUN_DIR, RUN], check=True)

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
ea = EventAccumulator(RUN_DIR, size_guidance={"scalars": 0})
ea.Reload()
bad = total = 0
for tag in ea.Tags()["scalars"]:
    ref = {int(e.step): float(e.value) for e in ea.Scalars(tag)}
    url = f"http://localhost:8188/api/board/runs/{RUN}/scalars?tag={urllib.request.quote(tag, safe='')}"
    got = {int(r[0]): float(r[2]) for r in json.load(urllib.request.urlopen(url))}
    total += len(ref)
    for s, v in ref.items():
        if s not in got or struct.unpack("f", struct.pack("f", got[s]))[0] != v:
            bad += 1
    print(f"  {tag}: ref={len(ref)} served={len(got)}")
print("BOARD TFEVENTS VALUE PARITY:", "PASS" if bad == 0 else f"FAIL ({bad}/{total})")
sys.exit(0 if bad == 0 else 1)
