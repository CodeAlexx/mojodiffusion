# autograd_v2/tests/ideogram4_block_parity.mojo — Stage-1 SAME-PROCESS bit gate
# (contract C14): the engine's ideogram4_block_graph_backward must equal the
# hand-chain ideogram4_block_lora_backward BIT-for-BIT on a real-shaped synthetic
# block with NONZERO LoRA B (so d_A is non-degenerate — a degenerate compare must
# FAIL). IDEOGRAM4_SDPA_FLASH is off (math SDPA, deterministic) so bit equality is
# achievable. Mirrors autograd_v2/tests/klein_block_parity.mojo.

from std.gpu.host import DeviceContext
from std.memory import ArcPointer
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.random import randn
from serenitymojo.ops.tensor_algebra import add_scalar, mul_scalar, zeros_device
from serenitymojo.scratch_ring import ScratchRingAllocator
from serenitymojo.models.ideogram4.block import (
    I4_SLOTS_PER_BLOCK,
    Ideogram4BlockWeights,
    build_ideogram4_lora_set,
    ideogram4_block_lora_forward,
    ideogram4_block_lora_backward,
    LArc,
)
from serenitymojo.autograd_v2.ideogram4_block_graph import (
    ideogram4_block_graph_backward,
)

comptime TArc = ArcPointer[Tensor]


def _s1(a: Int) -> List[Int]:
    var s = List[Int](); s.append(a); return s^


def _s2(a: Int, b: Int) -> List[Int]:
    var s = List[Int](); s.append(a); s.append(b); return s^


def _s3(a: Int, b: Int, c: Int) -> List[Int]:
    var s = List[Int](); s.append(a); s.append(b); s.append(c); return s^


def _w(out_f: Int, in_f: Int, seed: UInt64, ctx: DeviceContext) raises -> Tensor:
    return mul_scalar(randn(_s2(out_f, in_f), seed, STDtype.BF16, ctx), Float32(0.04), ctx)


def _ones(n: Int, ctx: DeviceContext) raises -> Tensor:
    return add_scalar(zeros_device(_s1(n), STDtype.BF16, ctx), Float32(1.0), ctx)


def _arc(t: Tensor) raises -> TArc:
    return TArc(Tensor(t.buf.copy(), t.shape(), t.dtype()))


def _diff(name: String, a: Tensor, b: Tensor, ctx: DeviceContext) raises -> Int:
    var ah = a.to_host(ctx)
    var bh = b.to_host(ctx)
    var nm = 0
    for i in range(len(ah)):
        if ah[i] != bh[i]:
            nm += 1
    print("  ", name, "n_mismatch", nm, "/", len(ah))
    return nm


def main() raises:
    var ctx = DeviceContext()
    comptime S = 4
    comptime Hidden = 16
    comptime Heads = 4
    comptime Dh = 4
    comptime FF = 24
    comptime Adaln = 8

    var w = Ideogram4BlockWeights(
        _w(4 * Hidden, Adaln, UInt64(1), ctx),
        zeros_device(_s1(4 * Hidden), STDtype.BF16, ctx),
        _ones(Hidden, ctx), _ones(Hidden, ctx), _ones(Hidden, ctx), _ones(Hidden, ctx),
        _w(3 * Hidden, Hidden, UInt64(2), ctx),
        _w(Hidden, Hidden, UInt64(3), ctx),
        _ones(Dh, ctx), _ones(Dh, ctx),
        _w(FF, Hidden, UInt64(4), ctx),
        _w(Hidden, FF, UInt64(5), ctx),
        _w(FF, Hidden, UInt64(6), ctx),
    )

    var loras = build_ideogram4_lora_set[Hidden, FF, Adaln](4, Float32(4.0), ctx, 1)
    var bl = List[LArc]()
    for slot in range(I4_SLOTS_PER_BLOCK):
        bl.append(loras.ad[slot])
    # NONZERO B (the gate's non-degeneracy requirement): overwrite each B with randn.
    for slot in range(I4_SLOTS_PER_BLOCK):
        bl[slot][].b = mul_scalar(
            randn(bl[slot][].b.shape(), UInt64(200 + slot), STDtype.BF16, ctx),
            Float32(0.04), ctx,
        )

    var x = randn(_s2(S, Hidden), UInt64(100), STDtype.BF16, ctx)
    var adaln = randn(_s2(1, Adaln), UInt64(101), STDtype.BF16, ctx)
    var cosf = add_scalar(zeros_device(_s3(1, S, Dh), STDtype.BF16, ctx), Float32(1.0), ctx)
    var sinf = zeros_device(_s3(1, S, Dh), STDtype.BF16, ctx)
    var d_out = randn(_s2(S, Hidden), UInt64(300), STDtype.BF16, ctx)

    # ── HAND-CHAIN (the oracle) ────────────────────────────────────────────────
    var fwd = ideogram4_block_lora_forward[S, Hidden, Heads, Dh, FF, Adaln](
        x, adaln, cosf, sinf, w, bl, ctx
    )
    var hand = ideogram4_block_lora_backward[S, Hidden, Heads, Dh, FF, Adaln](
        d_out, fwd.acts^, cosf, sinf, w, bl, ctx
    )

    # ── ENGINE (graph) ─────────────────────────────────────────────────────────
    var scratch = ScratchRingAllocator(ctx, 64 * 1024 * 1024, 2)
    var eng = ideogram4_block_graph_backward[S, Hidden, Heads, Dh, FF, Adaln](
        _arc(d_out), _arc(x), _arc(adaln), cosf, sinf, w, bl, ctx, scratch
    )

    # ── BIT compare ────────────────────────────────────────────────────────────
    var total = 0
    total += _diff("d_x", hand.d_x, eng.d_x, ctx)
    total += _diff("d_adaln", hand.d_adaln_input, eng.d_adaln_input, ctx)
    for slot in range(I4_SLOTS_PER_BLOCK):
        total += _diff(
            String("d_a") + String(slot),
            hand.lora_grads.d_a[slot][], eng.lora_grads.d_a[slot][], ctx,
        )
        total += _diff(
            String("d_b") + String(slot),
            hand.lora_grads.d_b[slot][], eng.lora_grads.d_b[slot][], ctx,
        )
    if total == 0:
        print("IDEOGRAM4 BLOCK PARITY PASS (engine == hand-chain, bit-equal)")
    else:
        raise Error(
            String("IDEOGRAM4 BLOCK PARITY FAIL: total mismatches ") + String(total)
        )
