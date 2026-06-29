# models/flux/parity/flux_lycoris_orchestration_smoke.mojo
# Flux/Chroma block-projection LoKr/LoHa orchestration gate.
#
# Build/run:
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#   pixi run mojo build --optimization-level 0 -I . -Xlinker -lm -Xlinker -lcuda \
#     -Xlinker -L.pixi/envs/default/lib -Xlinker -lsqlite3 \
#     serenitymojo/models/flux/parity/flux_lycoris_orchestration_smoke.mojo \
#     -o /tmp/flux_lycoris_orch && /tmp/flux_lycoris_orch

from std.collections import List
from std.gpu.host import DeviceContext
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.models.flux.flux_stack_lora import total_adapters
from serenitymojo.models.flux.flux_lycoris_stack import (
    build_flux_lokr_set, flux_lokr_carrier_set, flux_lokr_carrier_total_bytes,
    flux_lokr_chain_all, flux_lokr_adamw_step, flux_lokr_zero_leg_l1,
    save_flux_lokr,
    build_flux_loha_set, flux_loha_carrier_set, flux_loha_carrier_total_bytes,
    flux_loha_chain_all, flux_loha_adamw_step, flux_loha_zero_leg_l1,
    save_flux_loha,
)

comptime LOKR_OUT = "/tmp/flux_lokr_orch.safetensors"
comptime LOHA_OUT = "/tmp/flux_loha_orch.safetensors"


def _fill(n: Int, seed: UInt64, scale: Float32) -> List[Float32]:
    var out = List[Float32]()
    var state = seed
    for _ in range(n):
        state = state * 6364136223846793005 + 1442695040888963407
        var u = Float32(Int(state >> 40)) * Float32(1.0 / 16777216.0)
        out.append((u - Float32(0.5)) * scale)
    return out^


def _grad_lists(
    n: Int, rank_in: List[Int], out_rank: List[Int]
) -> Tuple[List[List[Float32]], List[List[Float32]]]:
    var d_a = List[List[Float32]]()
    var d_b = List[List[Float32]]()
    for i in range(n):
        d_a.append(_fill(rank_in[i], UInt64(7) * UInt64(i + 1) + 1, 0.5))
        d_b.append(_fill(out_rank[i], UInt64(11) * UInt64(i + 1) + 3, 0.5))
    return (d_a^, d_b^)


def main() raises:
    var ctx = DeviceContext()
    var D = 32
    var F = 48
    var ND = 2
    var NS = 3

    print("=== Flux LyCORIS orchestration smoke (D=", D, " F=", F, " double=", ND, " single=", NS, ") ===")

    var lokr = build_flux_lokr_set(
        ND, NS, D, F, 4, Float32(8.0), -1, 0, 0, True, False, 2, UInt64(101),
    )
    print("  LoKr carrier bytes:", flux_lokr_carrier_total_bytes(lokr, D, F))
    var lokr_carriers = flux_lokr_carrier_set(lokr, D, F)
    var nslots = total_adapters(lokr_carriers)
    if len(lokr_carriers.ad) != nslots:
        raise Error("LoKr carrier count mismatch")
    var lokr_rank_in = List[Int]()
    var lokr_out_rank = List[Int]()
    for i in range(len(lokr_carriers.ad)):
        lokr_rank_in.append(lokr_carriers.ad[i].rank * lokr_carriers.ad[i].in_f)
        lokr_out_rank.append(lokr_carriers.ad[i].out_f * lokr_carriers.ad[i].rank)
    var lokr_grads = _grad_lists(len(lokr_carriers.ad), lokr_rank_in, lokr_out_rank)
    var lkg = flux_lokr_chain_all(lokr, lokr_grads[0], lokr_grads[1])
    var lokr_before = flux_lokr_zero_leg_l1(lokr)
    flux_lokr_adamw_step(lokr, lkg, 1, Float32(1.0e-3), Float32(0.9), Float32(0.999), Float32(1.0e-8), Float32(0.01))
    var lokr_after = flux_lokr_zero_leg_l1(lokr)
    if lokr_after <= lokr_before:
        raise Error("LoKr zero-leg did not move")
    var lokr_saved = save_flux_lokr(lokr, String(LOKR_OUT), ctx)
    _ = SafeTensors.open(String(LOKR_OUT))
    print("  LoKr moved zero-leg and saved modules:", lokr_saved)

    var loha = build_flux_loha_set(ND, NS, D, F, 3, Float32(6.0), 2, UInt64(202))
    print("  LoHa carrier bytes:", flux_loha_carrier_total_bytes(loha, D, F))
    var loha_carriers = flux_loha_carrier_set(loha, D, F)
    if len(loha_carriers.ad) != nslots:
        raise Error("LoHa carrier count mismatch")
    var loha_rank_in = List[Int]()
    var loha_out_rank = List[Int]()
    for i in range(len(loha_carriers.ad)):
        loha_rank_in.append(loha_carriers.ad[i].rank * loha_carriers.ad[i].in_f)
        loha_out_rank.append(loha_carriers.ad[i].out_f * loha_carriers.ad[i].rank)
    var loha_grads = _grad_lists(len(loha_carriers.ad), loha_rank_in, loha_out_rank)
    var lhg = flux_loha_chain_all(loha, loha_grads[0], loha_grads[1])
    var loha_before = flux_loha_zero_leg_l1(loha)
    flux_loha_adamw_step(loha, lhg, 1, Float32(1.0e-3), Float32(0.9), Float32(0.999), Float32(1.0e-8), Float32(0.01))
    var loha_after = flux_loha_zero_leg_l1(loha)
    if loha_after <= loha_before:
        raise Error("LoHa zero-leg did not move")
    var loha_saved = save_flux_loha(loha, String(LOHA_OUT), ctx)
    _ = SafeTensors.open(String(LOHA_OUT))
    print("  LoHa moved zero-leg and saved modules:", loha_saved)
    print("ALL GATES PASS -- flux_lycoris_orchestration_smoke")
