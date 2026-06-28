# models/zimage/parity/zimage_loha_orchestration_smoke.mojo
# Twin of zimage_lokr_orchestration_smoke for LoHa: proves the zimage LoHa SET
# orchestration (build → carrier_lists → chain → adamw → save). Carrier core
# already gated by loha_carrier_parity.
#
# Build/run:
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#   pixi run mojo build --optimization-level 2 -I . -Xlinker -lm -Xlinker -lcuda \
#     -Xlinker -L.pixi/envs/default/lib -Xlinker -lsqlite3 \
#     serenitymojo/models/zimage/parity/zimage_loha_orchestration_smoke.mojo \
#     -o /tmp/zimage_loha_orch && /tmp/zimage_loha_orch
from std.collections import List
from std.gpu.host import DeviceContext
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.models.zimage.lora_block import ZIMAGE_SLOTS
from serenitymojo.models.zimage.zimage_loha_stack import (
    ZImageLoHaSet, build_zimage_loha_set, zimage_loha_carrier_lists,
    zimage_loha_carrier_total_bytes, zimage_loha_chain_all, zimage_loha_adamw_step,
    zimage_loha_zero_leg_l1, save_zimage_loha,
)

comptime SAVE_PATH = "/tmp/zimage_loha_orch.safetensors"


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
    var D = 64
    var F = 96
    var NR = 1
    var CR = 1
    var MAIN = 2
    var RANK = 4
    var ALPHA = Float32(8.0)
    var total_blocks = NR + CR + MAIN
    var nslots = total_blocks * ZIMAGE_SLOTS

    print("=== zimage LoHa orchestration smoke (D=", D, " F=", F,
          " blocks=", total_blocks, " slots=", nslots, ") ===")

    var masters = build_zimage_loha_set(NR, CR, MAIN, D, F, RANK, ALPHA, 2, UInt64(12345))
    print("  built ZImageLoHaSet; carrier bytes:", zimage_loha_carrier_total_bytes(masters))

    var carriers = zimage_loha_carrier_lists(masters, D, F)
    if len(carriers) != nslots:
        raise Error("carrier list count != total_blocks*7")
    print("  materialized", len(carriers), "carriers ✓")

    var d_a = List[List[Float32]]()
    var d_b = List[List[Float32]]()
    for i in range(len(carriers)):
        var r = carriers[i].rank
        var inf = carriers[i].in_f
        var outf = carriers[i].out_f
        d_a.append(_fill(r * inf, UInt64(7) * UInt64(i + 1) + 1, 0.5))
        d_b.append(_fill(outf * r, UInt64(11) * UInt64(i + 1) + 3, 0.5))

    var mg = zimage_loha_chain_all(masters, d_a, d_b)
    var zero_before = zimage_loha_zero_leg_l1(masters)
    zimage_loha_adamw_step(masters, mg, 1, Float32(1.0e-3),
                           Float32(0.9), Float32(0.999), Float32(1.0e-8), Float32(0.01))
    var zero_after = zimage_loha_zero_leg_l1(masters)
    print("  w2a zero-leg L1: before =", zero_before, " after =", zero_after)
    if zero_after <= zero_before:
        raise Error("AdamW did not move the w2a zero-leg off zero (master no-op?)")
    print("  master AdamW moved the zero-leg ✓")

    var nmods = save_zimage_loha(masters, SAVE_PATH, ctx)
    print("  save_zimage_loha wrote", nmods, "modules")
    if nmods <= 0:
        raise Error("save_zimage_loha wrote nothing")
    var st = SafeTensors.open(SAVE_PATH)
    print("  reopened", SAVE_PATH, "✓")

    print("ALL GATES PASS — zimage_loha_orchestration_smoke")
