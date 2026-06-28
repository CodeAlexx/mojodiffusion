# models/zimage/parity/zimage_lokr_orchestration_smoke.mojo
#
# Proves the NEW zimage LoKr SET orchestration end-to-end (build → carrier_lists
# → chain → adamw → save), WITHOUT the zimage stack forward (the carrier CORE
# lokr_carrier_adapter/lokr_chain_carrier_grads is already gated by
# lokr_st_parity; the stack-composition path is the SAME proven path klein uses).
# Toy dims, self-contained.
#
# Asserts: (1) carrier list has total_blocks*7 entries, all finite; (2) chained
# master grads from a synthetic carrier d_a/d_b move the w2 zero-leg off zero
# after one AdamW step (the masters actually learn); (3) save_zimage_lokr writes
# the lycoris-key file and it reopens.
#
# Build/run:
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#   pixi run mojo build --optimization-level 2 -I . -Xlinker -lm -Xlinker -lcuda \
#     -Xlinker -L.pixi/envs/default/lib -Xlinker -lsqlite3 \
#     serenitymojo/models/zimage/parity/zimage_lokr_orchestration_smoke.mojo \
#     -o /tmp/zimage_lokr_orch && /tmp/zimage_lokr_orch
from std.collections import List
from std.gpu.host import DeviceContext
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.models.zimage.lora_block import ZIMAGE_SLOTS
from serenitymojo.models.zimage.zimage_lokr_stack import (
    ZImageLoKrSet, build_zimage_lokr_set, zimage_lokr_carrier_lists,
    zimage_lokr_carrier_total_bytes, zimage_lokr_chain_all, zimage_lokr_adamw_step,
    zimage_lokr_zero_leg_l1, save_zimage_lokr,
)

comptime SAVE_PATH = "/tmp/zimage_lokr_orch.safetensors"


def _fill(n: Int, seed: UInt64, scale: Float32) -> List[Float32]:
    var out = List[Float32]()
    var state = seed
    for _ in range(n):
        state = state * 6364136223846793005 + 1442695040888963407
        var u = Float32(Int(state >> 40)) * Float32(1.0 / 16777216.0)
        out.append((u - Float32(0.5)) * scale)
    return out^


def _all_finite(h: List[Float32]) -> Bool:
    for i in range(len(h)):
        var v = h[i]
        if v != v:
            return False
        var a = v if v >= 0.0 else -v
        if a > Float32(1.0e30):
            return False
    return True


def main() raises:
    var ctx = DeviceContext()
    var D = 64
    var F = 96
    var NR = 1
    var CR = 1
    var MAIN = 2
    var RANK = 4
    var ALPHA = Float32(8.0)
    var total_blocks = NR + CR + MAIN
    var nslots = total_blocks * ZIMAGE_SLOTS

    print("=== zimage LoKr orchestration smoke (D=", D, " F=", F,
          " blocks=", total_blocks, " slots=", nslots, ") ===")

    # both-factored (decompose_both) → small r² carriers; targets=2 (attn+ff).
    var masters = build_zimage_lokr_set(
        NR, CR, MAIN, D, F, RANK, ALPHA, -1, True, False, 2,
        UInt64(12345),
    )
    var nbytes = zimage_lokr_carrier_total_bytes(masters)
    print("  built ZImageLoKrSet; carrier bytes:", nbytes)

    var carriers = zimage_lokr_carrier_lists(masters, D, F)
    if len(carriers) != nslots:
        raise Error("carrier list count != total_blocks*7")
    var ok = True
    for i in range(len(carriers)):
        ref a = carriers[i].a
        ref b = carriers[i].b
        for j in range(len(a)):
            if a[j].cast[DType.float32]() != a[j].cast[DType.float32]():
                ok = False
        for j in range(len(b)):
            if b[j].cast[DType.float32]() != b[j].cast[DType.float32]():
                ok = False
    if not ok:
        raise Error("carrier contains NaN")
    print("  materialized", len(carriers), "carriers, all finite ✓")

    # synthesize a carrier d_a/d_b matching each carrier's (a,b) shape (active
    # slots get nonzero grad so the chain drives the masters off the zero leg).
    var d_a = List[List[Float32]]()
    var d_b = List[List[Float32]]()
    for i in range(len(carriers)):
        var r = carriers[i].rank
        var inf = carriers[i].in_f
        var outf = carriers[i].out_f
        d_a.append(_fill(r * inf, UInt64(7) * UInt64(i + 1) + 1, 0.5))
        d_b.append(_fill(outf * r, UInt64(11) * UInt64(i + 1) + 3, 0.5))

    var mg = zimage_lokr_chain_all(masters, d_a, d_b)

    var zero_before = zimage_lokr_zero_leg_l1(masters)
    zimage_lokr_adamw_step(masters, mg, 1, Float32(1.0e-3),
                           Float32(0.9), Float32(0.999), Float32(1.0e-8), Float32(0.01))
    var zero_after = zimage_lokr_zero_leg_l1(masters)
    print("  w2 zero-leg L1: before =", zero_before, " after =", zero_after)
    if zero_after <= zero_before:
        raise Error("AdamW did not move the w2 zero-leg off zero (master no-op?)")
    print("  master AdamW moved the zero-leg ✓")

    var nmods = save_zimage_lokr(masters, SAVE_PATH, ctx)
    print("  save_zimage_lokr wrote", nmods, "modules")
    if nmods <= 0:
        raise Error("save_zimage_lokr wrote nothing")
    var st = SafeTensors.open(SAVE_PATH)
    print("  reopened", SAVE_PATH, "✓")

    print("ALL GATES PASS — zimage_lokr_orchestration_smoke")
