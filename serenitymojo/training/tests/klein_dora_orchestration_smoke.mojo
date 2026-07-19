# tests/klein_dora_orchestration_smoke.mojo
# Proves the klein DoRA SET orchestration (build → preflight → carrier_lists →
# chain → adamw → save) at SMALL dims (the full-delta r_eff=in carrier is
# VRAM-bound at real klein dims; preflight fails loud there). The carrier core
# is already gated by dora_carrier_parity; this proves the per-slot set + w_orig
# threading + chain/step/save.
#
# Build/run:
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#   pixi run mojo build --optimization-level 2 -I . -Xlinker -lm -Xlinker -lcuda \
#     -Xlinker -L.pixi/envs/default/lib -Xlinker -lsqlite3 \
#     serenitymojo/training/tests/klein_dora_orchestration_smoke.mojo \
#     -o /tmp/klein_dora_orch && /tmp/klein_dora_orch
from std.collections import List
from std.gpu.host import DeviceContext
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.training.dora_stack import (
    KleinDoRASet, build_klein_dora_set, klein_dora_preflight,
    klein_dora_carrier_lists, klein_dora_carrier_total_bytes,
    klein_dora_chain_all, klein_dora_adamw_step, klein_dora_zero_leg_l1,
    save_klein_dora,
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
    var ctx = DeviceContext()
    var D = 32
    var F = 48
    print("=== klein DoRA orchestration smoke (D=", D, " F=", F, " targets=attn) ===")

    # targets=1 (attn only) keeps the full-delta carriers small enough to fit.
    var masters = build_klein_dora_set(1, 1, D, F, 4, Float32(8.0), 1, UInt64(12345))
    klein_dora_preflight(masters)
    print("  built KleinDoRASet; carrier bytes:", klein_dora_carrier_total_bytes(masters), "(preflight OK)")

    var carriers = klein_dora_carrier_lists(masters, D, F)
    print("  materialized carriers: dbl=", len(carriers[0]), " sgl=", len(carriers[1]))

    # synthetic carrier grads (d_b = d_W_eff drives the chain; d_a discarded).
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

    var mg = klein_dora_chain_all(masters, dbl_d_a, dbl_d_b, sgl_d_a, sgl_d_b)

    var zero_before = klein_dora_zero_leg_l1(masters)
    klein_dora_adamw_step(masters, mg, 1, Float32(1.0e-3),
                          Float32(0.9), Float32(0.999), Float32(1.0e-8), Float32(0.01))
    var zero_after = klein_dora_zero_leg_l1(masters)
    print("  B zero-leg L1: before =", zero_before, " after =", zero_after)
    if zero_after <= zero_before:
        raise Error("AdamW did not move the DoRA B zero-leg off zero (master no-op?)")
    print("  master AdamW moved the zero-leg ✓")

    var nmods = save_klein_dora(masters, "/tmp/klein_dora_orch.safetensors", ctx)
    print("  save_klein_dora wrote", nmods, "modules")
    if nmods <= 0:
        raise Error("save_klein_dora wrote nothing")
    var st = SafeTensors.open("/tmp/klein_dora_orch.safetensors")
    print("  reopened ✓")
    print("ALL GATES PASS — klein_dora_orchestration_smoke")
