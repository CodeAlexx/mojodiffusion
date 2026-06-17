#!/usr/bin/env python
# boogu_sched_oracle.py — C8a (v1 scheduler) parity oracle. Dev tool, NOT shipped.
# Constructs the REAL Boogu FlowMatchEulerDiscreteScheduler from its config and
# dumps _timesteps (N+1 values, the ascending shifted schedule + trailing 1.0).
import os, sys
import numpy as np
SCHED_DIR = "/home/alex/Boogu-Image/models/Boogu-Image-0.1-Base/scheduler"
OUT = "/home/alex/mojodiffusion/serenitymojo/models/dit/parity/boogu_dumps"
os.makedirs(OUT, exist_ok=True)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 8


def main():
    from boogu.schedulers.scheduling_flow_match_euler_discrete_time_shifting import (
        FlowMatchEulerDiscreteScheduler,
    )
    sched = FlowMatchEulerDiscreteScheduler.from_pretrained(SCHED_DIR)
    print("[sched-oracle] config:", dict(sched.config))
    sched.set_timesteps(num_inference_steps=N, num_tokens=256)
    ts = np.asarray(sched._timesteps, dtype="<f4")   # N+1 values
    ts.tofile(os.path.join(OUT, f"boogu_sched_ts_{N}.bin"))
    print(f"[sched-oracle] N={N} _timesteps({len(ts)}): {ts.tolist()}")


if __name__ == "__main__":
    main()
