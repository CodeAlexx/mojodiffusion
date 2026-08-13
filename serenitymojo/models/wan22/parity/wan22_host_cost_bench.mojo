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

from max.gpu.host import DeviceContext
from std.collections import List
from std.time import perf_counter_ns
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.models.wan22.wan22_block import _t16, _expand_rope_per_head
from serenitymojo.models.klein.lora_block import lora_adapter_to_device
from serenitymojo.training.train_step import LoraAdapter

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

    # ── C) _t16: List[Float32] -> BF16 device (every modvec/mod tensor upload) ──
    var lst = _randn(S * DIM, 11)
    var wt = _t16(lst, [S, DIM], ctx)
    ctx.synchronize()
    var t2 = perf_counter_ns()
    for _ in range(40):
        var tt = _t16(lst, [S, DIM], ctx)
        n += Int(tt.numel() > 0)
    ctx.synchronize()
    var ms_c = Float64(perf_counter_ns() - t2) / 1.0e6
    print("C) _t16[S,dim] F32list->BF16dev x40 =", ms_c, "ms  per-call=", ms_c / 40.0,
          "ms   [warm=", wt.numel(), "]")

    # ── D) _expand_rope_per_head at real dims (2 per block: cos + sin) ──
    var tbl = Tensor.from_host(_randn(S * 64, 12), [S, 64], STDtype.F32, ctx)
    var we = _expand_rope_per_head(tbl, S, 40, 64, ctx)
    ctx.synchronize()
    var t3 = perf_counter_ns()
    for _ in range(80):          # 2 per block x 40 blocks = one step
        var e = _expand_rope_per_head(tbl, S, 40, 64, ctx)
        n += Int(e.numel() > 0)
    ctx.synchronize()
    var ms_d = Float64(perf_counter_ns() - t3) / 1.0e6
    print("D) _expand_rope_per_head x80 (one step) =", ms_d, "ms  per-call=",
          ms_d / 80.0, "ms   [warm=", we.numel(), "]")

    # ── E) lora_adapter_to_device (400/step: 10 proj x 40 blocks) ──
    var zr = List[Float32]()
    for _ in range(16 * DIM):
        zr.append(0.0)
    var ad = LoraAdapter(_randn(16 * DIM, 13), _randn(DIM * 16, 14), 16, DIM, DIM,
                         Float32(1.0), zr.copy(), zr.copy(), zr.copy(), zr.copy())
    var wd2 = lora_adapter_to_device(ad, ctx)
    ctx.synchronize()
    var t4 = perf_counter_ns()
    for _ in range(400):
        var d2 = lora_adapter_to_device(ad, ctx)
        n += Int(d2.rank > 0)
    ctx.synchronize()
    var ms_e = Float64(perf_counter_ns() - t4) / 1.0e6
    print("E) lora_adapter_to_device x400 (one step) =", ms_e, "ms  per-call=",
          ms_e / 400.0, "ms   [warm rank=", wd2.rank, "]")

    print("")
    print("Step is now 5.638 s: nsys attributes ~1.34 s CUDA API + ~0.70 s GPU,")
    print("leaving ~3.6 s/step unaccounted host. Per-step totals above:")
    print("  C x (12 per block x 40) =", ms_c / 40.0 * 480.0 / 1000.0, "s   (6 fwd + 6 graph)")
    print("  D (one step)            =", ms_d / 1000.0, "s")
    print("  E (one step)            =", ms_e / 1000.0, "s")
    print("  _ = ", n)
