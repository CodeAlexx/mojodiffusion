# tests/klein_oft_orchestration_smoke.mojo
# Proves the klein OFT SET orchestration build→preflight→carrier_lists→chain→
# adamw (vec moves off zero) at SMALL dims. The carrier core is gated by
# oft_carrier_parity. SAVE is a follow-on (OneTrainer-OFT triu-vec save format
# is net-new). Full-delta r_eff=in → VRAM-bound at real klein dims.
#
# Build/run:
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#   pixi run mojo build --optimization-level 2 -I . -Xlinker -lm -Xlinker -lcuda \
#     -Xlinker -L.pixi/envs/default/lib -Xlinker -lsqlite3 \
#     serenitymojo/training/tests/klein_oft_orchestration_smoke.mojo \
#     -o /tmp/klein_oft_orch && /tmp/klein_oft_orch
from std.collections import List
from serenitymojo.training.oft_stack import (
    KleinOFTSet, build_klein_oft_set, klein_oft_preflight,
    klein_oft_carrier_lists, klein_oft_carrier_total_bytes,
    klein_oft_chain_all, klein_oft_adamw_step, klein_oft_vec_l1,
)


def _fill(n: Int, seed: UInt64, scale: Float32) -> List[Float32]:
    var out = List[Float32]()
    var state = seed
    for _ in range(n):
        state = state * 6364136223846793005 + 1442695040888963407
        var u = Float32(Int(state >> 40)) * Float32(1.0 / 16777216.0)
        out.append((u - Float32(0.5)) * scale)
    return out^


def main() raises:
    var D = 32
    var F = 48
    print("=== klein OFT orchestration smoke (D=", D, " F=", F, " b=4 targets=attn) ===")

    var masters = build_klein_oft_set(1, 1, D, F, 4, 1, UInt64(12345))
    klein_oft_preflight(masters)
    print("  built KleinOFTSet; carrier bytes:", klein_oft_carrier_total_bytes(masters), "(preflight OK)")

    var carriers = klein_oft_carrier_lists(masters, D, F)
    print("  materialized carriers: dbl=", len(carriers[0]), " sgl=", len(carriers[1]))

    var dbl_d_a = List[List[Float32]]()
    var dbl_d_b = List[List[Float32]]()
    for i in range(len(carriers[0])):
        var r = carriers[0][i].rank; var inf = carriers[0][i].in_f; var outf = carriers[0][i].out_f
        dbl_d_a.append(_fill(r * inf, UInt64(7) * UInt64(i + 1) + 1, 0.5))
        dbl_d_b.append(_fill(outf * r, UInt64(11) * UInt64(i + 1) + 3, 0.5))
    var sgl_d_a = List[List[Float32]]()
    var sgl_d_b = List[List[Float32]]()
    for i in range(len(carriers[1])):
        var r = carriers[1][i].rank; var inf = carriers[1][i].in_f; var outf = carriers[1][i].out_f
        sgl_d_a.append(_fill(r * inf, UInt64(13) * UInt64(i + 1) + 1, 0.5))
        sgl_d_b.append(_fill(outf * r, UInt64(17) * UInt64(i + 1) + 3, 0.5))

    var mg = klein_oft_chain_all(masters, dbl_d_a, dbl_d_b, sgl_d_a, sgl_d_b)
    var vec_before = klein_oft_vec_l1(masters)
    klein_oft_adamw_step(masters, mg, 1, Float32(1.0e-3),
                         Float32(0.9), Float32(0.999), Float32(1.0e-8), Float32(0.01))
    var vec_after = klein_oft_vec_l1(masters)
    print("  vec L1: before =", vec_before, " after =", vec_after)
    if vec_after <= vec_before:
        raise Error("AdamW did not move the OFT vec off zero (master no-op?)")
    print("  master AdamW moved the vec ✓")
    print("ALL GATES PASS — klein_oft_orchestration_smoke (save = follow-on)")
