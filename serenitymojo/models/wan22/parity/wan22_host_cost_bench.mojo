# models/wan22/parity/wan22_host_cost_bench.mojo
#
# MEASUREMENT (not a parity gate): price the two remaining HOST costs in the wan
# step, at the REAL dims, so the next optimization is chosen by a number.
#
# CONTEXT: after moving the LoRA forward on-device the step is 8.1 s, but nsys says
# only ~1.3 s/step is CUDA API and ~0.7 s/step is GPU kernels → ~6 s/step is still
# host CPU work OUTSIDE CUDA. nsys CPU sampling is unavailable on this box, and an
# earlier element-count estimate proved WRONG (removing ~628M List.copy() element
# copies changed nothing — Mojo's List.copy() is a bulk memcpy, not per-element).
# So measure, don't count elements.
#
# A) `_block_modvecs` — a pure-CPU nested loop, S*dim iterations x 6 appends per
#    block (wan22_stack_lora.mojo:~700). Runs 40x/step (once per block).
# B) `Tensor.to_host` of a [S,dim] bf16 tensor — the bf16→F32 widening that every
#    remaining readback pays (nsys still shows ~1.2k DtoH/step).
#
# Build (rm -f serenitymojo.mojopkg first):
#   pixi run mojo build --optimization-level 2 -I . -Xlinker -lm -Xlinker -lcuda \
#     -Xlinker -L.pixi/envs/default/lib -Xlinker -lsqlite3 \
#     serenitymojo/models/wan22/parity/wan22_host_cost_bench.mojo -o /tmp/wan_host_bench
#
# Mojo 1.0.0b1, NVIDIA.

from std.gpu.host import DeviceContext
from std.collections import List
from std.time import perf_counter_ns
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype

comptime S = 256
comptime DIM = 5120
comptime BLOCKS = 40      # per-step call count for _block_modvecs


def _randn(n: Int, seed: UInt64) -> List[Float32]:
    var out = List[Float32]()
    var s = seed
    for _ in range(n):
        s = s * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        out.append(Float32((s >> 33) & UInt64(0x7FFFFF)) / Float32(8388608.0))
    return out^


# EXACT copy of wan22_stack_lora.mojo::_block_modvecs (kept verbatim so the number
# is the real one, not an approximation of it).
def _block_modvecs_copy(
    e0_flat: List[Float32], block_mod_h: List[Float32], bi: Int, S_: Int, dim: Int,
) -> Int:
    var shift_sa = List[Float32]()
    var scale_sa = List[Float32]()
    var gate_sa = List[Float32]()
    var shift_ffn = List[Float32]()
    var scale_ffn = List[Float32]()
    var gate_ffn = List[Float32]()
    for s in range(S_):
        for d in range(dim):
            shift_sa.append(e0_flat[s * 6 * dim + 0 * dim + d] + block_mod_h[0 * dim + d])
            scale_sa.append(e0_flat[s * 6 * dim + 1 * dim + d] + block_mod_h[1 * dim + d])
            gate_sa.append(e0_flat[s * 6 * dim + 2 * dim + d] + block_mod_h[2 * dim + d])
            shift_ffn.append(e0_flat[s * 6 * dim + 3 * dim + d] + block_mod_h[3 * dim + d])
            scale_ffn.append(e0_flat[s * 6 * dim + 4 * dim + d] + block_mod_h[4 * dim + d])
            gate_ffn.append(e0_flat[s * 6 * dim + 5 * dim + d] + block_mod_h[5 * dim + d])
    return len(shift_sa) + len(gate_ffn)


def main() raises:
    var ctx = DeviceContext()
    print("==== wan host-cost bench (S=", S, " dim=", DIM, ") ====")

    # ── A) _block_modvecs, 40 calls = one step's worth ──
    var e0 = _randn(S * 6 * DIM, 7)
    var bmod = _randn(6 * DIM, 8)
    var warm = _block_modvecs_copy(e0, bmod, 0, S, DIM)
    var t0 = perf_counter_ns()
    var acc = 0
    for bi in range(BLOCKS):
        acc += _block_modvecs_copy(e0, bmod, bi, S, DIM)
    var ns_a = perf_counter_ns() - t0
    var ms_a = Float64(ns_a) / 1.0e6
    print("A) _block_modvecs x", BLOCKS, "(one step) =", ms_a, "ms   per-call=",
          ms_a / Float64(BLOCKS), "ms   [warm=", warm, " acc=", acc, "]")

    # ── B) to_host of [S,dim] bf16 (the widening every readback pays) ──
    var t = Tensor.from_host(_randn(S * DIM, 9), [S, DIM], STDtype.BF16, ctx)
    var wh = t.to_host(ctx)
    ctx.synchronize()
    var t1 = perf_counter_ns()
    var n = 0
    for _ in range(40):
        var h = t.to_host(ctx)
        n += len(h)
    var ns_b = perf_counter_ns() - t1
    var ms_b = Float64(ns_b) / 1.0e6
    print("B) to_host[S,dim] x40 =", ms_b, "ms   per-call=", ms_b / 40.0,
          "ms   [warm_len=", len(wh), " n=", n, "]")

    print("")
    print("Step budget check: measured step = 8.1 s, of which nsys attributes")
    print("~1.3 s CUDA API + ~0.7 s GPU. A) above is the per-step modvec cost.")
