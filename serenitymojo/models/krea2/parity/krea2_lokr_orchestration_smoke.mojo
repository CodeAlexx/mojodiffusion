# models/krea2/parity/krea2_lokr_orchestration_smoke.mojo
# Proves the krea2 LoKr SET orchestration (build → carrier_lists → chain → adamw
# → save) at toy dims. Carrier core gated by lokr_st_parity; this proves the
# krea2 per-block 8-slot set + chain/step/save.
#
# Build/run:
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#   pixi run mojo build --optimization-level 2 -I . -Xlinker -lm -Xlinker -lcuda \
#     -Xlinker -L.pixi/envs/default/lib -Xlinker -lsqlite3 \
#     serenitymojo/models/krea2/parity/krea2_lokr_orchestration_smoke.mojo \
#     -o /tmp/krea2_lokr_orch && /tmp/krea2_lokr_orch
from std.collections import List
from std.gpu.host import DeviceContext
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.models.krea2.krea2_lokr_stack import (
    Krea2LoKrSet, build_krea2_lokr_set, krea2_lokr_carrier_lists,
    krea2_lokr_carrier_total_bytes, krea2_lokr_chain_all, krea2_lokr_adamw_step,
    krea2_lokr_zero_leg_l1, save_krea2_lokr, KREA2_SLOTS,
)

comptime SAVE_PATH = "/tmp/krea2_lokr_orch.safetensors"


def _fill(n: Int, seed: UInt64, scale: Float32) -> List[Float32]:
    var out = List[Float32]()
    var state = seed
    for _ in range(n):
        state = state * 6364136223846793005 + 1442695040888963407
        var u = Float32(Int(state >> 40)) * Float32(1.0 / 16777216.0)
        out.append((u - Float32(0.5)) * scale)
    return out^


def main() raises:
    var ctx = DeviceContext()
    var D = 64; var F = 96; var qdim = 64; var kvdim = 32
    var NB = 2; var RANK = 4
    var nslots = NB * KREA2_SLOTS

    print("=== krea2 LoKr orchestration smoke (D=", D, " F=", F, " blocks=", NB, " slots=", nslots, ") ===")

    var masters = build_krea2_lokr_set(NB, D, F, qdim, kvdim, RANK, Float32(8.0), -1, True, False, 2, UInt64(12345))
    print("  built Krea2LoKrSet; carrier bytes:", krea2_lokr_carrier_total_bytes(masters))

    var carriers = krea2_lokr_carrier_lists(masters, D, F, qdim, kvdim)
    if len(carriers) != nslots:
        raise Error("carrier list count != num_blocks*8")
    print("  materialized", len(carriers), "carriers ✓")

    var d_a = List[List[Float32]]()
    var d_b = List[List[Float32]]()
    for i in range(len(carriers)):
        var r = carriers[i].rank; var inf = carriers[i].in_f; var outf = carriers[i].out_f
        d_a.append(_fill(r * inf, UInt64(7) * UInt64(i + 1) + 1, 0.5))
        d_b.append(_fill(outf * r, UInt64(11) * UInt64(i + 1) + 3, 0.5))

    var mg = krea2_lokr_chain_all(masters, d_a, d_b)
    var zero_before = krea2_lokr_zero_leg_l1(masters)
    krea2_lokr_adamw_step(masters, mg, 1, Float32(1.0e-3),
                          Float32(0.9), Float32(0.999), Float32(1.0e-8), Float32(0.01))
    var zero_after = krea2_lokr_zero_leg_l1(masters)
    print("  w2 zero-leg L1: before =", zero_before, " after =", zero_after)
    if zero_after <= zero_before:
        raise Error("AdamW did not move the w2 zero-leg off zero (master no-op?)")
    print("  master AdamW moved the zero-leg ✓")

    var nmods = save_krea2_lokr(masters, SAVE_PATH, ctx)
    print("  save_krea2_lokr wrote", nmods, "modules")
    if nmods <= 0:
        raise Error("save_krea2_lokr wrote nothing")
    var st = SafeTensors.open(SAVE_PATH)
    print("  reopened ✓")
    print("ALL GATES PASS — krea2_lokr_orchestration_smoke")
