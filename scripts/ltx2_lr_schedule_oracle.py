#!/usr/bin/env python3
# ltx2_lr_schedule_oracle.py — transformers.optimization LR reference for the
# LTX2 LR-lever parity gate. torchref's LR path IS transformers.get_scheduler, so
# this dumps get_scheduler's per-step LR factor (base_lr=1.0) as the oracle the
# Mojo transformers_lr_for_step must reproduce.
#
# Dumps 9 rows x TOTAL factors, flat little-endian f32, row-major, to
#   serenitymojo/models/ltx2/parity/ltx2_lr_schedule_ref.bin
# Row order + grid are mirrored EXACTLY by ltx2_lr_schedule_parity.mojo.
#
#   Env (has torch + transformers):
#     /home/alex/serenityflow-v2/.venv/bin/python scripts/ltx2_lr_schedule_oracle.py
#
# factors[t] = lambda(t) for the 0-based scheduler index t (transformers
# LambdaLR sets lr=base*lambda(0) at construction; each scheduler.step()
# advances last_epoch). optimizer step k (1-based) consumes lambda(k-1).
import os
import struct
import warnings

import torch
from transformers import get_scheduler

warnings.filterwarnings("ignore")  # silence scheduler-before-optimizer order note

TOTAL = 100
BASE = 1.0
# (transformers scheduler name, num_warmup_steps). constant ignores warmup, so
# it is probed at warmup 0 only; constant_with_warmup carries the ramp. 10 ==
# int(0.1 * TOTAL) (the "float 0.1 of total" case resolved to an absolute int).
ROWS = [
    ("constant", 0),
    ("constant_with_warmup", 5),
    ("constant_with_warmup", 10),
    ("linear", 0),
    ("linear", 5),
    ("linear", 10),
    ("cosine", 0),
    ("cosine", 5),
    ("cosine", 10),
]

OUT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "serenitymojo",
        "models",
        "ltx2",
        "parity",
        "ltx2_lr_schedule_ref.bin",
    )
)


def factors(name: str, warmup: int) -> list[float]:
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([p], lr=BASE)
    sched = get_scheduler(
        name, opt, num_warmup_steps=warmup, num_training_steps=TOTAL
    )
    out = []
    for _ in range(TOTAL):
        out.append(sched.get_last_lr()[0] / BASE)  # lambda(t)
        sched.step()
    return out


def main() -> None:
    vals: list[float] = []
    for name, w in ROWS:
        vals.extend(factors(name, w))
    assert len(vals) == len(ROWS) * TOTAL, len(vals)
    with open(OUT, "wb") as f:
        for v in vals:
            f.write(struct.pack("<f", float(v)))
    print(f"wrote {len(vals)} f32 ({len(ROWS)} rows x {TOTAL}) -> {OUT}")


if __name__ == "__main__":
    main()
