# gen_cosmos_sampler_reference.py — DEV-ONLY oracle for the Cosmos samplers.
#
# Builds the GROUND-TRUTH next-latent trajectories directly from the CANONICAL
# Cosmos sampler module (NOT a transcription):
#   /home/alex/refs/cosmos-predict2.5/cosmos_predict2/_src/predict2/models/fm_solvers_unipc.py
# loaded by file path so the package __init__ CUDA-extra check is bypassed.
#
# Two oracles, both with a FIXED model-output (velocity) per step + FIXED sigma
# schedule (shift=5, num_train=1000), so the Mojo sampler step is gated as pure
# numeric step-parity (no model in the loop):
#
#   1. UniPC  — the canonical FlowUniPCMultistepScheduler.step run for N steps
#               (bh2, solver_order=2, predict_x0=True, lower_order_final=True).
#   2. RF     — rectified-flow / FlowMatch Euler:
#               sigmas built by FlowUniPCMultistepScheduler.set_timesteps (the
#               SAME schedule the RF reference cosmos_rf.rs documents), then
#               x_next = x + (sigma_next - sigma) * v   (diffusers FlowMatch).
#
# Emits Mojo List[Float32] literals to stdout for paste into cosmos_sampler_parity.mojo.
#
# Run:
#   cd /home/alex/refs/cosmos-predict2.5 && \
#     python3 /home/alex/mojodiffusion/serenitymojo/sampling/parity/gen_cosmos_sampler_reference.py

import importlib.util
import numpy as np
import torch

SCHED_PATH = (
    "/home/alex/refs/cosmos-predict2.5/cosmos_predict2/"
    "_src/predict2/models/fm_solvers_unipc.py"
)

_spec = importlib.util.spec_from_file_location("fmsolv", SCHED_PATH)
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)
FlowUniPCMultistepScheduler = _m.FlowUniPCMultistepScheduler

torch.manual_seed(0)
np.random.seed(0)

N_INF = 6
SHIFT = 5.0
NUM_TRAIN = 1000
SOLVER_ORDER = 2
DIM = 8  # vector length per step


def mojo_list(name, arr):
    flat = np.asarray(arr).reshape(-1)
    vals = ", ".join(f"{float(x):.8f}" for x in flat)
    return f"# {name}\n    [{vals}],"


def fixed_inputs():
    """Deterministic x_init + per-step velocity (model output) vectors.

    Shaped [1, 1, DIM] so the canonical sampler's einsum "k,bkc...->bc..."
    (which needs >=3 dims) is exercised exactly as in real inference.
    """
    rng = np.random.default_rng(1234)
    x0 = rng.standard_normal((1, 1, DIM)).astype(np.float64)
    vs = [rng.standard_normal((1, 1, DIM)).astype(np.float64) for _ in range(N_INF)]
    return x0, vs


def run_unipc(x0, vs):
    sch = FlowUniPCMultistepScheduler(
        num_train_timesteps=NUM_TRAIN,
        solver_order=SOLVER_ORDER,
        prediction_type="flow_prediction",
        shift=SHIFT,
        predict_x0=True,
        solver_type="bh2",
        lower_order_final=True,
        final_sigmas_type="zero",
    )
    sch.set_timesteps(num_inference_steps=N_INF, device="cpu")
    sigmas = sch.sigmas.float().cpu().numpy().tolist()
    x = torch.from_numpy(x0.copy()).to(torch.float64)
    traj = []
    for i in range(N_INF):
        ts = sch.timesteps[i]
        v = torch.from_numpy(vs[i].copy()).to(torch.float64)
        out = sch.step(v, ts, x, return_dict=True).prev_sample
        x = out
        traj.append(out.cpu().numpy().astype(np.float64))
    return sigmas, traj


def run_rf(x0, vs):
    """Rectified-flow Euler over the SAME schedule (cosmos_rf.rs convention)."""
    sch = FlowUniPCMultistepScheduler(
        num_train_timesteps=NUM_TRAIN,
        solver_order=SOLVER_ORDER,
        shift=SHIFT,
        final_sigmas_type="zero",
    )
    sch.set_timesteps(num_inference_steps=N_INF, device="cpu")
    sigmas = sch.sigmas.float().cpu().numpy().tolist()  # len N_INF+1, last 0
    x = torch.from_numpy(x0.copy()).to(torch.float64)
    traj = []
    for i in range(N_INF):
        dt = sigmas[i + 1] - sigmas[i]
        v = torch.from_numpy(vs[i].copy()).to(torch.float64)
        x = x + dt * v
        traj.append(x.cpu().numpy().astype(np.float64))
    return sigmas, traj


def main():
    x0, vs = fixed_inputs()
    unipc_sigmas, unipc_traj = run_unipc(x0, vs)
    rf_sigmas, rf_traj = run_rf(x0, vs)

    print("# ==== paste into cosmos_sampler_parity.mojo ====")
    print(f"comptime N_INF = {N_INF}")
    print(f"comptime DIM = {DIM}")
    print(f"comptime SHIFT = {SHIFT}")
    print()
    print("# x_init")
    print(mojo_list("x_init", x0))
    print()
    print("# per-step velocity (model output)")
    for i, v in enumerate(vs):
        print(mojo_list(f"v[{i}]", v))
    print()
    print("# RF sigmas (len N_INF+1)")
    print(mojo_list("rf_sigmas", rf_sigmas))
    print("# RF per-step next-latent (oracle)")
    for i, t in enumerate(rf_traj):
        print(mojo_list(f"rf_step[{i}]", t))
    print()
    print("# UniPC sigmas (len N_INF+1)")
    print(mojo_list("unipc_sigmas", unipc_sigmas))
    print("# UniPC per-step next-latent (oracle)")
    for i, t in enumerate(unipc_traj):
        print(mojo_list(f"unipc_step[{i}]", t))


if __name__ == "__main__":
    main()
