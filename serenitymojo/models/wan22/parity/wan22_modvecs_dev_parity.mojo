# models/wan22/parity/wan22_modvecs_dev_parity.mojo — modvec builder BIT gate.
#
# Gates `_block_modvecs_dev` (GPU broadcast add) against `_block_modvecs` (the
# pure-CPU nested loop it replaces) at the REAL wan dims. Both compute
#     e[s,j,d] = e0_flat[s,j,d] + block_mod[j,d]
# with the SAME F32 elementwise add — IEEE add is deterministic elementwise, so
# only WHERE it runs differs and the result must be BIT-EQUAL. A non-bit result
# means the shapes/broadcast/slice mapping is wrong, not that "GPU rounds
# differently": treat any mismatch as a REAL bug.
#
# WHY the replacement: the host loop is 64.66 ms/call × 40 blocks = 2.586 s/step
# = ~32% of the 8.4 s step (models/wan22/parity/wan22_host_cost_bench.mojo).
#
# This ALSO times both so the win is recorded with the gate.
#
# Build (rm -f serenitymojo.mojopkg first):
#   pixi run mojo build --optimization-level 2 -I . -Xlinker -lm -Xlinker -lcuda \
#     -Xlinker -L.pixi/envs/default/lib -Xlinker -lsqlite3 \
#     -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa \
#     serenitymojo/models/wan22/parity/wan22_modvecs_dev_parity.mojo -o /tmp/wan_mv_parity
#
# Mojo 1.0.0b1, NVIDIA.

from std.gpu.host import DeviceContext
from std.collections import List
from std.time import perf_counter_ns
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.tensor_algebra import reshape
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.models.wan22.wan22_block import WanModVecs, _t16
from serenitymojo.models.wan22.wan22_stack_lora import (
    _block_modvecs, _block_modvecs_dev, _block_modvecs_dev16,
)

comptime S = 256
comptime DIM = 5120
comptime BLOCKS = 40


def _randn(n: Int, seed: UInt64) -> List[Float32]:
    var out = List[Float32]()
    var s = seed
    for _ in range(n):
        s = s * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        out.append((Float32((s >> 33) & UInt64(0x7FFFFF)) / Float32(8388608.0)) - Float32(0.5))
    return out^


def _cmp(name: String, got: List[Float32], want: List[Float32], mut allok: Bool):
    var bad = 0
    if len(got) != len(want):
        bad = -1
    else:
        for i in range(len(got)):
            if got[i] != want[i]:
                bad += 1
    var verdict = String("PASS") if bad == 0 else String("FAIL")
    if bad != 0:
        allok = False
    print("GATE", name, verdict, "n_mismatch=", bad, "/", len(want))


def main() raises:
    var ctx = DeviceContext()
    print("==== wan22 modvecs: device builder vs host loop (BIT) ====")
    print("dims S=", S, " dim=", DIM)

    var e0 = _randn(S * 6 * DIM, 21)
    var bmod = _randn(6 * DIM, 22)
    var e0_t = Tensor.from_host(e0.copy(), [S, 6, DIM], STDtype.F32, ctx)
    var bmod_t = Tensor.from_host(bmod.copy(), [1, 6, DIM], STDtype.F32, ctx)

    var host = _block_modvecs(e0, bmod, 0, S, DIM)
    var dev = _block_modvecs_dev(e0_t, bmod_t, S, DIM, ctx)

    var allok = True
    _cmp("shift_sa", dev.shift_sa, host.shift_sa, allok)
    _cmp("scale_sa", dev.scale_sa, host.scale_sa, allok)
    _cmp("gate_sa", dev.gate_sa, host.gate_sa, allok)
    _cmp("shift_ffn", dev.shift_ffn, host.shift_ffn, allok)
    _cmp("scale_ffn", dev.scale_ffn, host.scale_ffn, allok)
    _cmp("gate_ffn", dev.gate_ffn, host.gate_ffn, allok)

    # teeth: a degenerate all-zero compare must not read as PASS
    var nz = 0
    for i in range(len(dev.shift_sa)):
        if dev.shift_sa[i] != 0.0:
            nz += 1
    print("teeth: shift_sa nonzero=", nz, "/", len(dev.shift_sa), " (must be > 0)")

    # ── device-mode (BF16 tensors, `_block_modvecs_dev16`) vs what the consumers
    # upload today (`_t16` of the host list) ──
    # This is the gate for the "hold the modvecs as device tensors" path. It is NOT
    # a CPU-vs-GPU numerics claim: both sides apply the SAME `.cast[bfloat16]()` to
    # the SAME F32 values, one in `Tensor.from_host`'s pack loop and one in the
    # cast kernel, so anything but n_mismatch=0 is a real bug (wrong slice mapping,
    # wrong order, or a cast that is not round-to-nearest-even on one side).
    # Compared as F32-widened BF16 so the check sees the stored bf16 bits exactly.
    var dev16 = _block_modvecs_dev16(e0_t, bmod_t, S, DIM, ctx)
    if not dev16.on_device():
        raise Error("dev16 builder did not return device mode")
    var names = ["shift_sa", "scale_sa", "gate_sa", "shift_ffn", "scale_ffn", "gate_ffn"]
    var nz16 = 0
    for j in range(6):
        var want = cast_tensor(_t16(host.host_vec(j), [S, DIM], ctx), STDtype.F32, ctx).to_host(ctx)
        var got = cast_tensor(dev16.dev[j][], STDtype.F32, ctx).to_host(ctx)
        _cmp(String("bf16 dev vs _t16: ") + names[j], got, want, allok)
        if j == 0:
            for i in range(len(got)):
                if got[i] != 0.0:
                    nz16 += 1
    print("teeth: bf16 shift_sa nonzero=", nz16, "/", S * DIM, " (must be > 0)")
    if nz16 == 0:
        allok = False

    # ── timing: one step's worth (40 blocks) each ──
    var t0 = perf_counter_ns()
    for _ in range(BLOCKS):
        var h = _block_modvecs(e0, bmod, 0, S, DIM)
        _ = len(h.shift_sa)
    var host_ms = Float64(perf_counter_ns() - t0) / 1.0e6
    var t1 = perf_counter_ns()
    for _ in range(BLOCKS):
        var d = _block_modvecs_dev(e0_t, bmod_t, S, DIM, ctx)
        _ = len(d.shift_sa)
    ctx.synchronize()
    var dev_ms = Float64(perf_counter_ns() - t1) / 1.0e6
    print("TIMING x", BLOCKS, "(one step): host=", host_ms, "ms  device=", dev_ms,
          "ms  speedup=", host_ms / dev_ms, "x  saved=", (host_ms - dev_ms) / 1000.0, "s/step")

    if allok and nz > 0:
        print("VERDICT: PASS — device modvecs BIT-EQUAL to the host loop")
    else:
        print("VERDICT: FAIL — device modvecs diverged")
