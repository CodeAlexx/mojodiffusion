# models/lingbotvideo/parity/c_scheduler_probe.mojo — CHUNK C gate.
#
# Replays oracle_c.safetensors against the pure-Mojo FlowUniPCMultistepScheduler.
#   (1) set_timesteps(40, shift=3) -> compare timesteps + sigmas vs oracle.
#   (2) UniPC is STATEFUL: step i in order, feeding the oracle's in_i as the sample
#       and mo_i as the model_output, keeping our own multistep history. Compare our
#       per-step output to out_i (== in_{i+1}, so this replays AND cross-checks).
#
# Run (JIT):
#   cd /home/alex/mojodiffusion && \
#     pixi run mojo run -I . serenitymojo/models/lingbotvideo/parity/c_scheduler_probe.mojo

from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.parity import ParityHarness
from serenitymojo.models.lingbotvideo.scheduler import FlowUniPCMultistepScheduler

comptime PARITY_DIR = "/home/alex/mojodiffusion/serenitymojo/models/lingbotvideo/parity"
comptime STEPS = 40
comptime SHIFT = 3.0


def _load_f32(
    st: ShardedSafeTensors, name: String, ctx: DeviceContext
) raises -> List[Float32]:
    var tv = st.tensor_view(name)
    var t = Tensor.from_view(tv, ctx)
    return t.to_host(ctx)


def main() raises:
    var ctx = DeviceContext()
    print("[chunk C] FlowUniPCMultistepScheduler parity — steps=", STEPS, " shift=", SHIFT)

    var oracle = ShardedSafeTensors.open(String(PARITY_DIR) + "/oracle_c.safetensors")
    var harness = ParityHarness(0.999)

    # ── set_timesteps ────────────────────────────────────────────────────────
    var sch = FlowUniPCMultistepScheduler(num_train_timesteps=1000)
    sch.set_timesteps(STEPS, SHIFT)

    # gate 1: timesteps
    var ts_ref = _load_f32(oracle, "timesteps", ctx)
    var ts_mine = List[Float32]()
    for k in range(len(sch.timesteps)):
        ts_mine.append(Float32(sch.timesteps[k]))
    var ts_res = harness.compare_host(ts_mine, ts_ref)
    print("[gate 1] timesteps: cos=", ts_res.cos, " max_abs=", ts_res.max_abs,
          " (n=", ts_res.n, ")")

    # gate 2: sigmas
    var sig_ref = _load_f32(oracle, "sigmas", ctx)
    var sig_mine = List[Float32]()
    for k in range(len(sch.sigmas)):
        sig_mine.append(Float32(sch.sigmas[k]))
    var sig_res = harness.compare_host(sig_mine, sig_ref)
    print("[gate 2] sigmas:    cos=", sig_res.cos, " max_abs=", sig_res.max_abs,
          " (n=", sig_res.n, ")")

    # ── gate 3: per-step trajectory replay ───────────────────────────────────
    var min_cos: Float64 = 2.0
    var min_cos_step = -1
    var cos0: Float64 = 0.0
    var cos1: Float64 = 0.0
    var cos20: Float64 = 0.0
    var cos39: Float64 = 0.0
    for i in range(STEPS):
        var in_i = _load_f32(oracle, String("in_") + String(i), ctx)
        var mo_i = _load_f32(oracle, String("mo_") + String(i), ctx)
        var out_ref = _load_f32(oracle, String("out_") + String(i), ctx)
        var out_mine = sch.step(mo_i, in_i)
        var res = harness.compare_host(out_mine, out_ref)
        if res.cos < min_cos:
            min_cos = res.cos
            min_cos_step = i
        if i == 0:
            cos0 = res.cos
        elif i == 1:
            cos1 = res.cos
        elif i == 20:
            cos20 = res.cos
        elif i == 39:
            cos39 = res.cos

    print("[gate 3] per-step output cos:")
    print("         i=0  cos=", cos0)
    print("         i=1  cos=", cos1)
    print("         i=20 cos=", cos20)
    print("         i=39 cos=", cos39)
    print("         MIN cos across 40 steps =", min_cos, " (at step ", min_cos_step, ")")

    # ── verdict ──────────────────────────────────────────────────────────────
    var ts_ok = ts_res.max_abs < 1.0e-3
    var sig_ok = sig_res.max_abs < 1.0e-3
    var traj_ok = min_cos >= 0.999
    print("[verdict] timesteps max_abs<1e-3:", ts_ok,
          "  sigmas max_abs<1e-3:", sig_ok,
          "  min_step_cos>=0.999:", traj_ok)
    if ts_ok and sig_ok and traj_ok:
        print("[chunk C] GATE PASS")
    else:
        print("[chunk C] GATE FAIL")
