# Numeric step-parity gate for the MagiHuman SR distill UniPC step_ddim.
# Reference: inference-flame/src/sampling/magihuman_unipc.rs (FlowUniPcDDim).
#   schedule: sampling/unipc.build_unipc_sigma_schedule (shift=5, N_train=1000)
#   step_ddim: cur_clean = curr_state - curr_t*v
#              prev_state = prev_t*noise + (1-prev_t)*cur_clean
# We replicate the canonical formula in host F64 and compare the Mojo GPU step at
# several step indices. Gate cos >= 0.999. (No checkpoint needed — pure sampler.)

from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness
from serenitymojo.sampling.unipc import build_unipc_sigma_schedule
from serenitymojo.models.dit.magihuman_sr_dit import magihuman_unipc_step_ddim

comptime N_INF = 8
comptime SHIFT = 5.0
comptime N_TRAIN = 1000
comptime DIM = 4096


def main() raises:
    var ctx = DeviceContext()
    var sigmas = build_unipc_sigma_schedule(N_INF, SHIFT, N_TRAIN)
    print("sigma schedule (len", len(sigmas), "):")
    for i in range(len(sigmas)):
        print("  sigmas[", i, "] =", sigmas[i])

    # Deterministic pseudo-random host data for velocity / curr_state / noise.
    var vh = List[Float32]()
    var ch = List[Float32]()
    var nh = List[Float32]()
    var seed: Int = 12345
    for i in range(DIM):
        seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
        vh.append(Float32((seed % 2000) - 1000) / 500.0)
        seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
        ch.append(Float32((seed % 2000) - 1000) / 300.0)
        seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
        nh.append(Float32((seed % 2000) - 1000) / 400.0)

    var sh = List[Int]()
    sh.append(DIM)
    var v_t = Tensor.from_host(vh.copy(), sh.copy(), STDtype.F32, ctx)
    var c_t = Tensor.from_host(ch.copy(), sh.copy(), STDtype.F32, ctx)
    var n_t = Tensor.from_host(nh.copy(), sh.copy(), STDtype.F32, ctx)

    var worst_cos = 2.0
    for idx in range(N_INF):
        var out = magihuman_unipc_step_ddim(v_t, idx, c_t, n_t, sigmas, ctx)
        # Host reference (F64): prev = prev_t*noise + (1-prev_t)*(curr - curr_t*v)
        var curr_t = sigmas[idx]
        var prev_t = sigmas[idx + 1]
        var refv = List[Float32]()
        for i in range(DIM):
            var cur_clean = Float64(ch[i]) - curr_t * Float64(vh[i])
            var prev = prev_t * Float64(nh[i]) + (1.0 - prev_t) * cur_clean
            refv.append(Float32(prev))
        var harness = ParityHarness(0.999)
        var res = harness.compare(out, refv^, ctx)
        if res.cos < worst_cos:
            worst_cos = res.cos
        print("step", idx, " curr_t=", curr_t, " prev_t=", prev_t, " cos=", res.cos, " max_abs=", res.max_abs)

    print("MagiHuman SR UniPC step_ddim gate:")
    print("  worst_cos =", worst_cos)
    if worst_cos >= 0.999:
        print("  GATE: PASS (cos >= 0.999)")
    else:
        print("  GATE: FAIL (cos < 0.999)")
