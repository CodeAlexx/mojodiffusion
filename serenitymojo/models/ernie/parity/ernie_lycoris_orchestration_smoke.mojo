# models/ernie/parity/ernie_lycoris_orchestration_smoke.mojo
# ERNIE LoKr/LoHa/DoRA/OFT orchestration gate: build -> carrier -> chain -> AdamW
# -> save/reopen. Shared adapter tests cover the carrier math in detail.
#
# Build/run:
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#   pixi run mojo build --optimization-level 0 -I . -Xlinker -lm -Xlinker -lcuda \
#     -Xlinker -L.pixi/envs/default/lib -Xlinker -lsqlite3 \
#     serenitymojo/models/ernie/parity/ernie_lycoris_orchestration_smoke.mojo \
#     -o /tmp/ernie_lycoris_orch && /tmp/ernie_lycoris_orch

from std.collections import List
from std.gpu.host import DeviceContext
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.models.ernie.lora_block import ERNIE_SLOTS
from serenitymojo.models.ernie.ernie_lycoris_stack import (
    ernie_full_delta_carrier_bytes_estimate, ernie_full_delta_preflight,
    build_ernie_lokr_set, ernie_lokr_carrier_set, ernie_lokr_carrier_total_bytes,
    ernie_lokr_chain_all, ernie_lokr_adamw_step, ernie_lokr_zero_leg_l1,
    save_ernie_lokr,
    build_ernie_loha_set, ernie_loha_carrier_set, ernie_loha_carrier_total_bytes,
    ernie_loha_chain_all, ernie_loha_adamw_step, ernie_loha_zero_leg_l1,
    save_ernie_loha,
    build_ernie_dora_set, ernie_dora_carrier_set, ernie_dora_carrier_total_bytes,
    ernie_dora_preflight, ernie_dora_chain_all, ernie_dora_adamw_step,
    ernie_dora_zero_leg_l1, save_ernie_dora,
    build_ernie_oft_set, ernie_oft_carrier_set, ernie_oft_carrier_total_bytes,
    ernie_oft_preflight, ernie_oft_chain_all, ernie_oft_adamw_step,
    ernie_oft_vec_l1, save_ernie_oft,
)
from serenitymojo.training.lokr_stack import LOKR_CARRIER_MAX_DEVICE_BYTES

comptime LOKR_OUT = "/tmp/ernie_lokr_orch.safetensors"
comptime LOHA_OUT = "/tmp/ernie_loha_orch.safetensors"
comptime DORA_OUT = "/tmp/ernie_dora_orch.safetensors"
comptime OFT_OUT = "/tmp/ernie_oft_orch.safetensors"


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
    var NL = 2
    var nslots = NL * ERNIE_SLOTS
    print("=== ERNIE LyCORIS orchestration smoke (D=", D, " F=", F, " layers=", NL, " slots=", nslots, ") ===")

    var real_attn = ernie_full_delta_preflight(36, 4096, 12288, 1, LOKR_CARRIER_MAX_DEVICE_BYTES)
    if real_attn != 9667215360:
        raise Error("ERNIE attention-only full-delta byte estimate changed")
    var real_all = ernie_full_delta_carrier_bytes_estimate(36, 4096, 12288, 2)
    if real_all != 33822867456:
        raise Error("ERNIE all-target full-delta byte estimate changed")
    var all_rejected = False
    try:
        _ = ernie_full_delta_preflight(36, 4096, 12288, 2, LOKR_CARRIER_MAX_DEVICE_BYTES)
    except:
        all_rejected = True
    if not all_rejected:
        raise Error("ERNIE all-target full-delta preflight unexpectedly passed")
    print("  real ERNIE attention-only full-delta carrier bytes:", real_attn)
    print("  real ERNIE all-target full-delta carrier bytes rejected:", real_all)

    var lokr = build_ernie_lokr_set(
        NL, D, F, 4, Float32(8.0), -1, 0, 0, True, False, 3, UInt64(101),
    )
    print("  LoKr carrier bytes:", ernie_lokr_carrier_total_bytes(lokr, D, F))
    var lokr_carriers = ernie_lokr_carrier_set(lokr, D, F)
    if len(lokr_carriers.ad) != nslots:
        raise Error("LoKr carrier count mismatch")
    var lokr_rank_in = List[Int]()
    var lokr_out_rank = List[Int]()
    for i in range(len(lokr_carriers.ad)):
        lokr_rank_in.append(lokr_carriers.ad[i].rank * lokr_carriers.ad[i].in_f)
        lokr_out_rank.append(lokr_carriers.ad[i].out_f * lokr_carriers.ad[i].rank)
    var lokr_grads = _grad_lists(len(lokr_carriers.ad), lokr_rank_in, lokr_out_rank)
    var lkg = ernie_lokr_chain_all(lokr, lokr_grads[0], lokr_grads[1])
    var lokr_before = ernie_lokr_zero_leg_l1(lokr)
    ernie_lokr_adamw_step(lokr, lkg, 1, Float32(1.0e-3), Float32(0.9), Float32(0.999), Float32(1.0e-8), Float32(0.01))
    var lokr_after = ernie_lokr_zero_leg_l1(lokr)
    if lokr_after <= lokr_before:
        raise Error("LoKr zero-leg did not move")
    var lokr_saved = save_ernie_lokr(lokr, String(LOKR_OUT), ctx)
    _ = SafeTensors.open(String(LOKR_OUT))
    print("  LoKr moved zero-leg and saved modules:", lokr_saved)

    var loha = build_ernie_loha_set(NL, D, F, 3, Float32(6.0), 3, UInt64(202))
    print("  LoHa carrier bytes:", ernie_loha_carrier_total_bytes(loha, D, F))
    var loha_carriers = ernie_loha_carrier_set(loha, D, F)
    if len(loha_carriers.ad) != nslots:
        raise Error("LoHa carrier count mismatch")
    var loha_rank_in = List[Int]()
    var loha_out_rank = List[Int]()
    for i in range(len(loha_carriers.ad)):
        loha_rank_in.append(loha_carriers.ad[i].rank * loha_carriers.ad[i].in_f)
        loha_out_rank.append(loha_carriers.ad[i].out_f * loha_carriers.ad[i].rank)
    var loha_grads = _grad_lists(len(loha_carriers.ad), loha_rank_in, loha_out_rank)
    var lhg = ernie_loha_chain_all(loha, loha_grads[0], loha_grads[1])
    var loha_before = ernie_loha_zero_leg_l1(loha)
    ernie_loha_adamw_step(loha, lhg, 1, Float32(1.0e-3), Float32(0.9), Float32(0.999), Float32(1.0e-8), Float32(0.01))
    var loha_after = ernie_loha_zero_leg_l1(loha)
    if loha_after <= loha_before:
        raise Error("LoHa zero-leg did not move")
    var loha_saved = save_ernie_loha(loha, String(LOHA_OUT), ctx)
    _ = SafeTensors.open(String(LOHA_OUT))
    print("  LoHa moved zero-leg and saved modules:", loha_saved)

    var dora = build_ernie_dora_set(NL, D, F, 4, Float32(8.0), 1, UInt64(303), False)
    var dora_bytes = ernie_dora_carrier_total_bytes(dora, D, F)
    ernie_dora_preflight(dora, D, F, LOKR_CARRIER_MAX_DEVICE_BYTES)
    print("  DoRA carrier bytes:", dora_bytes)
    var dora_carriers = ernie_dora_carrier_set(dora, D, F)
    if len(dora_carriers.ad) != nslots:
        raise Error("DoRA carrier count mismatch")
    var dora_rank_in = List[Int]()
    var dora_out_rank = List[Int]()
    for i in range(len(dora_carriers.ad)):
        dora_rank_in.append(dora_carriers.ad[i].rank * dora_carriers.ad[i].in_f)
        dora_out_rank.append(dora_carriers.ad[i].out_f * dora_carriers.ad[i].rank)
    var dora_grads = _grad_lists(len(dora_carriers.ad), dora_rank_in, dora_out_rank)
    var dg = ernie_dora_chain_all(dora, dora_grads[0], dora_grads[1])
    var dora_before = ernie_dora_zero_leg_l1(dora)
    ernie_dora_adamw_step(dora, dg, 1, Float32(1.0e-3), Float32(0.9), Float32(0.999), Float32(1.0e-8), Float32(0.01))
    var dora_after = ernie_dora_zero_leg_l1(dora)
    if dora_after <= dora_before:
        raise Error("DoRA zero-leg did not move")
    var dora_saved = save_ernie_dora(dora, String(DORA_OUT), ctx)
    _ = SafeTensors.open(String(DORA_OUT))
    print("  DoRA moved zero-leg and saved modules:", dora_saved)

    var oft = build_ernie_oft_set(NL, D, F, 4, 1, UInt64(404))
    var oft_bytes = ernie_oft_carrier_total_bytes(oft, D, F)
    ernie_oft_preflight(oft, D, F, LOKR_CARRIER_MAX_DEVICE_BYTES)
    print("  OFT carrier bytes:", oft_bytes)
    var oft_carriers = ernie_oft_carrier_set(oft, D, F)
    if len(oft_carriers.ad) != nslots:
        raise Error("OFT carrier count mismatch")
    var oft_rank_in = List[Int]()
    var oft_out_rank = List[Int]()
    for i in range(len(oft_carriers.ad)):
        oft_rank_in.append(oft_carriers.ad[i].rank * oft_carriers.ad[i].in_f)
        oft_out_rank.append(oft_carriers.ad[i].out_f * oft_carriers.ad[i].rank)
    var oft_grads = _grad_lists(len(oft_carriers.ad), oft_rank_in, oft_out_rank)
    var og = ernie_oft_chain_all(oft, oft_grads[0], oft_grads[1])
    var oft_before = ernie_oft_vec_l1(oft)
    ernie_oft_adamw_step(oft, og, 1, Float32(1.0e-3), Float32(0.9), Float32(0.999), Float32(1.0e-8), Float32(0.01))
    var oft_after = ernie_oft_vec_l1(oft)
    if oft_after <= oft_before:
        raise Error("OFT vec did not move")
    var oft_saved = save_ernie_oft(oft, String(OFT_OUT), ctx)
    _ = SafeTensors.open(String(OFT_OUT))
    print("  OFT moved vec and saved modules:", oft_saved)
    print("ALL GATES PASS -- ernie_lycoris_orchestration_smoke")
