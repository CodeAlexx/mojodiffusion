#!/usr/bin/env python3
# ai-toolkit-style tqdm progress for the Mojo Klein trainer.
# Pipe the trainer's stdout/stderr through this:
#   pixi run mojo run -I . serenitymojo/training/train_klein_real.mojo 2>&1 | python3 scripts/train_progress.py
# Renders a tqdm bar from `PROG step=k total=N loss=.. grad=.. lr=.. secs=..` lines,
# surfaces `EVENT ...` lines above the bar, and tees everything to output/train_klein_real.log.
import sys, re, os
from tqdm import tqdm

LOG = os.path.join("output", "train_klein_real.log")
os.makedirs("output", exist_ok=True)
log = open(LOG, "a", buffering=1)

prog_re = re.compile(r"PROG\s+step=(\d+)\s+total=(\d+)\s+loss=([-\d.eE+]+)\s+grad=([-\d.eE+]+)\s+lr=([-\d.eE+]+)\s+secs=([-\d.eE+]+)")
evt_re  = re.compile(r"EVENT\s+(\w+)\s+(.*)")

bar = None
DESC = "klein_alina_lora"
for raw in sys.stdin:
    line = raw.rstrip("\n")
    log.write(line + "\n")
    m = prog_re.search(line)
    if m:
        step, total, loss, grad, lr, secs = m.groups()
        step, total = int(step), int(total)
        if bar is None:
            bar = tqdm(total=total, desc=DESC, dynamic_ncols=True)
        bar.n = step
        bar.set_postfix_str(f"lr: {float(lr):.1e} loss: {float(loss):.3e} grad: {float(grad):.2e} {float(secs):.1f}s/it")
        bar.refresh()
        continue
    e = evt_re.search(line)
    if e:
        kind, rest = e.groups()
        msg = f"  ▸ {kind}: {rest}"
        (bar.write(msg) if bar else print(msg, flush=True))
        continue
    # non-PROG/EVENT chatter: keep loader/sample noise out of the bar, but show it
    if bar:
        # only surface clearly-interesting lines to avoid spamming the bar
        if any(k in line for k in ("PASS", "FAIL", "std", "FINITE", "saved", "sample", "Error", "error", "OOM")):
            bar.write(line)
    else:
        print(line, flush=True)
if bar:
    bar.close()
log.close()
