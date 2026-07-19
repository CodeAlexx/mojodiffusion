# models/ideogram4/parity/ideogram4_lycoris_orchestration_smoke.mojo
# Proves the ideogram4 LoKr + LoHa SET orchestration (build -> carrier device set
# -> host-list chain -> adamw -> zero-leg movement -> save) at TOY dims. The
# carrier CORE (lokr/loha carrier_adapter + chain + adamw) is already gated by
# lokr_st_parity / loha_carrier_parity; this proves the ideogram4 per-block
# 6-slot geometry + the host<->device bridge compile and wire.
#
# Build (compile-only gate — no GPU asserted here; run needs a device):
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#   pixi run mojo build --optimization-level 2 -I . -Xlinker -lm -Xlinker -lcuda \
#     -Xlinker -L.pixi/envs/default/lib -Xlinker -lsqlite3 \
#     serenitymojo/models/ideogram4/parity/ideogram4_lycoris_orchestration_smoke.mojo \
#     -o /tmp/ideogram4_lycoris_orch && /tmp/ideogram4_lycoris_orch
from std.collections import List
from std.gpu.host import DeviceContext
from serenitymojo.training.lokr_stack import lokr_carrier_r_eff
from serenitymojo.training.loha_stack import loha_carrier_r_eff
from serenitymojo.models.ideogram4.block import I4_SLOTS_PER_BLOCK
from serenitymojo.models.ideogram4.ideogram4_lokr_stack import (
    Ideogram4LoKrSet, build_ideogram4_lokr_set,
    ideogram4_lokr_carrier_total_bytes, ideogram4_lokr_carrier_device_set,
    ideogram4_lokr_chain_all, ideogram4_lokr_grad_norm, ideogram4_lokr_adamw_step,
    ideogram4_lokr_zero_leg_l1, save_ideogram4_lokr,
)
from serenitymojo.models.ideogram4.ideogram4_loha_stack import (
    Ideogram4LoHaSet, build_ideogram4_loha_set,
    ideogram4_loha_carrier_total_bytes, ideogram4_loha_carrier_device_set,
    ideogram4_loha_chain_all, ideogram4_loha_grad_norm, ideogram4_loha_adamw_step,
    ideogram4_loha_zero_leg_l1, save_ideogram4_loha,
)

comptime LOKR_SAVE = "/tmp/ideogram4_lokr_orch.safetensors"
comptime LOHA_SAVE = "/tmp/ideogram4_loha_orch.safetensors"


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
    # toy ideogram4 geometry (H,F,A shrunk; slot map preserved).
    var H = 48; var F = 64; var A = 32
    var NL = 2; var RANK = 4
    var nslots = NL * I4_SLOTS_PER_BLOCK

    print("=== ideogram4 LoKr orchestration smoke (H=", H, " F=", F, " A=", A, " layers=", NL, " slots=", nslots, ") ===")
    var lokr = build_ideogram4_lokr_set(
        NL, H, F, A, RANK, Float32(8.0), -1, True, False, 2, UInt64(12345)
    )
    print("  carrier device bytes:", ideogram4_lokr_carrier_total_bytes(lokr, H, F, A))
    var lokr_dev = ideogram4_lokr_carrier_device_set(lokr, H, F, A, ctx)
    if len(lokr_dev.ad) != nslots:
        raise Error("lokr device set count != layers*slots")
    print("  materialized", len(lokr_dev.ad), "device carriers")

    var kd_a = List[List[Float32]]()
    var kd_b = List[List[Float32]]()
    for i in range(len(lokr.ad)):
        if lokr.active[i]:
            var r = lokr_carrier_r_eff(lokr.ad[i])
            kd_a.append(_fill(r * lokr.ad[i].in_f, UInt64(7) * UInt64(i + 1) + 1, 0.5))
            kd_b.append(_fill(lokr.ad[i].out_f * r, UInt64(11) * UInt64(i + 1) + 3, 0.5))
        else:
            kd_a.append(List[Float32]())
            kd_b.append(List[Float32]())
    var kg = ideogram4_lokr_chain_all(lokr, kd_a, kd_b)
    var kzero_before = ideogram4_lokr_zero_leg_l1(lokr)
    ideogram4_lokr_adamw_step(lokr, kg, 1, Float32(1.0e-3), Float32(0.9), Float32(0.999), Float32(1.0e-8), Float32(0.01))
    var kzero_after = ideogram4_lokr_zero_leg_l1(lokr)
    print("  lokr grad_norm=", ideogram4_lokr_grad_norm(kg), " zero_leg before=", kzero_before, " after=", kzero_after)
    if kzero_after <= kzero_before:
        raise Error("lokr zero-leg did not move off zero after a step")
    var kn = save_ideogram4_lokr(lokr, LOKR_SAVE, ctx)
    print("  saved", kn, "lokr adapters ->", LOKR_SAVE)

    print("=== ideogram4 LoHa orchestration smoke ===")
    var loha = build_ideogram4_loha_set(NL, H, F, A, RANK, Float32(8.0), 2, UInt64(54321))
    print("  carrier device bytes:", ideogram4_loha_carrier_total_bytes(loha, H, F, A))
    var loha_dev = ideogram4_loha_carrier_device_set(loha, H, F, A, ctx)
    if len(loha_dev.ad) != nslots:
        raise Error("loha device set count != layers*slots")
    print("  materialized", len(loha_dev.ad), "device carriers")

    var hd_a = List[List[Float32]]()
    var hd_b = List[List[Float32]]()
    for i in range(len(loha.ad)):
        if loha.active[i]:
            var r = loha_carrier_r_eff(loha.ad[i])
            hd_a.append(_fill(r * loha.ad[i].in_f, UInt64(13) * UInt64(i + 1) + 5, 0.5))
            hd_b.append(_fill(loha.ad[i].out_f * r, UInt64(17) * UInt64(i + 1) + 7, 0.5))
        else:
            hd_a.append(List[Float32]())
            hd_b.append(List[Float32]())
    var hg = ideogram4_loha_chain_all(loha, hd_a, hd_b)
    var hzero_before = ideogram4_loha_zero_leg_l1(loha)
    ideogram4_loha_adamw_step(loha, hg, 1, Float32(1.0e-3), Float32(0.9), Float32(0.999), Float32(1.0e-8), Float32(0.01))
    var hzero_after = ideogram4_loha_zero_leg_l1(loha)
    print("  loha grad_norm=", ideogram4_loha_grad_norm(hg), " zero_leg before=", hzero_before, " after=", hzero_after)
    if hzero_after <= hzero_before:
        raise Error("loha zero-leg did not move off zero after a step")
    var hn = save_ideogram4_loha(loha, LOHA_SAVE, ctx)
    print("  saved", hn, "loha adapters ->", LOHA_SAVE)

    print("=== ideogram4 LyCORIS orchestration smoke PASS ===")
