#!/usr/bin/env python3
# flux_sched_oracle.py — BFL reference for the FLUX.1 sigma schedule (Gate B2).
#
# Inlines BFL get_schedule (flux/sampling.py is not importable standalone — it
# drags in imwatermark via image_embedders). The math is verbatim: linear mu
# (0.5 @ 256 tokens, 1.15 @ 4096), exponential time_shift with sigma=1.
# Dumps the (num_steps+1) descending sigmas to flux_sched_ref.bin (raw F32).
#
# Usage: python3 flux_sched_oracle.py [STEPS] [IMAGE_SEQ_LEN]

import sys, math, struct, os

OUT = "/home/alex/mojodiffusion/serenitymojo/models/flux/parity/flux_sched_ref.bin"
os.makedirs(os.path.dirname(OUT), exist_ok=True)


def get_schedule(num_steps, image_seq_len, base_shift=0.5, max_shift=1.15):
    m = (max_shift - base_shift) / (4096 - 256)
    b = base_shift - m * 256
    mu = m * image_seq_len + b
    ts = [1 - i / num_steps for i in range(num_steps + 1)]

    def shift(t):
        if t <= 0 or t >= 1:
            return t
        return math.exp(mu) / (math.exp(mu) + (1 / t - 1))

    return [shift(t) for t in ts]


def main():
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    seqlen = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
    sch = get_schedule(steps, seqlen)
    print(f"[oracle] BFL schedule steps={steps} seq_len={seqlen}:",
          [round(x, 6) for x in sch])
    with open(OUT, "wb") as f:
        for x in sch:
            f.write(struct.pack("<f", float(x)))
    print("[oracle] wrote", OUT)


if __name__ == "__main__":
    main()
