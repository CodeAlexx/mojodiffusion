# models/anima/parity/anima_lycoris_orchestration_smoke.mojo
# Anima LoKr/LoHa/DoRA/OFT orchestration gate.

from std.collections import List
from std.gpu.host import DeviceContext
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.models.anima.lora_block import ANIMA_SLOTS
from serenitymojo.models.anima.anima_lycoris_stack import (
    anima_full_delta_carrier_bytes_estimate, anima_full_delta_preflight,
    build_anima_lokr_set, anima_lokr_carrier_set, anima_lokr_carrier_total_bytes,
    anima_lokr_chain_all, anima_lokr_adamw_step, anima_lokr_zero_leg_l1,
    save_anima_lokr,
    build_anima_loha_set, anima_loha_carrier_set, anima_loha_carrier_total_bytes,
    anima_loha_chain_all, anima_loha_adamw_step, anima_loha_zero_leg_l1,
    save_anima_loha,
    build_anima_dora_set, anima_dora_carrier_set, anima_dora_carrier_total_bytes,
    anima_dora_preflight, anima_dora_chain_all, anima_dora_adamw_step,
    anima_dora_zero_leg_l1, save_anima_dora,
    build_anima_oft_set, anima_oft_carrier_set, anima_oft_carrier_total_bytes,
    anima_oft_preflight, anima_oft_chain_all, anima_oft_adamw_step,
    anima_oft_vec_l1, save_anima_oft,
)
from serenitymojo.training.lokr_stack import LOKR_CARRIER_MAX_DEVICE_BYTES

comptime LOKR_OUT = "/tmp/anima_lokr_orch.safetensors"
comptime LOHA_OUT = "/tmp/anima_loha_orch.safetensors"
comptime DORA_OUT = "/tmp/anima_dora_orch.safetensors"
comptime OFT_OUT = "/tmp/anima_oft_orch.safetensors"


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
    var JOINT = 24
    var F = 48
    var NB = 2
    var nslots = NB * ANIMA_SLOTS
    print("=== Anima LyCORIS orchestration smoke (D=", D, " JOINT=", JOINT, " F=", F, " blocks=", NB, ") ===")

    var real_est = anima_full_delta_carrier_bytes_estimate(28, 2048, 1024, 8192, 2)
    var real_pf = anima_full_delta_preflight(28, 2048, 1024, 8192, 2, LOKR_CARRIER_MAX_DEVICE_BYTES)
    if real_est != real_pf:
        raise Error("Anima real-size full-delta preflight mismatch")
    print("  real Anima all-target full-delta carrier bytes:", real_est)

    var lokr = build_anima_lokr_set(
        NB, D, JOINT, F, 4, Float32(8.0), -1, 0, 0, True, False, 2, UInt64(101),
    )
    print("  LoKr carrier bytes:", anima_lokr_carrier_total_bytes(lokr, D, JOINT, F))
    var lokr_carriers = anima_lokr_carrier_set(lokr, D, JOINT, F)
    if len(lokr_carriers.ad) != nslots:
        raise Error("LoKr carrier count mismatch")
    var lokr_rank_in = List[Int]()
    var lokr_out_rank = List[Int]()
    for i in range(len(lokr_carriers.ad)):
        lokr_rank_in.append(lokr_carriers.ad[i].rank * lokr_carriers.ad[i].in_f)
        lokr_out_rank.append(lokr_carriers.ad[i].out_f * lokr_carriers.ad[i].rank)
    var lokr_grads = _grad_lists(len(lokr_carriers.ad), lokr_rank_in, lokr_out_rank)
    var lkg = anima_lokr_chain_all(lokr, lokr_grads[0], lokr_grads[1])
    var lokr_before = anima_lokr_zero_leg_l1(lokr)
    anima_lokr_adamw_step(lokr, lkg, 1, Float32(1.0e-3), Float32(0.9), Float32(0.999), Float32(1.0e-8), Float32(0.01))
    var lokr_after = anima_lokr_zero_leg_l1(lokr)
    if lokr_after <= lokr_before:
        raise Error("LoKr zero-leg did not move")
    var lokr_saved = save_anima_lokr(lokr, String(LOKR_OUT), ctx)
    _ = SafeTensors.open(String(LOKR_OUT))
    print("  LoKr moved zero-leg and saved modules:", lokr_saved)

    var loha = build_anima_loha_set(NB, D, JOINT, F, 3, Float32(6.0), 2, UInt64(202))
    print("  LoHa carrier bytes:", anima_loha_carrier_total_bytes(loha, D, JOINT, F))
    var loha_carriers = anima_loha_carrier_set(loha, D, JOINT, F)
    if len(loha_carriers.ad) != nslots:
        raise Error("LoHa carrier count mismatch")
    var loha_rank_in = List[Int]()
    var loha_out_rank = List[Int]()
    for i in range(len(loha_carriers.ad)):
        loha_rank_in.append(loha_carriers.ad[i].rank * loha_carriers.ad[i].in_f)
        loha_out_rank.append(loha_carriers.ad[i].out_f * loha_carriers.ad[i].rank)
    var loha_grads = _grad_lists(len(loha_carriers.ad), loha_rank_in, loha_out_rank)
    var lhg = anima_loha_chain_all(loha, loha_grads[0], loha_grads[1])
    var loha_before = anima_loha_zero_leg_l1(loha)
    anima_loha_adamw_step(loha, lhg, 1, Float32(1.0e-3), Float32(0.9), Float32(0.999), Float32(1.0e-8), Float32(0.01))
    var loha_after = anima_loha_zero_leg_l1(loha)
    if loha_after <= loha_before:
        raise Error("LoHa zero-leg did not move")
    var loha_saved = save_anima_loha(loha, String(LOHA_OUT), ctx)
    _ = SafeTensors.open(String(LOHA_OUT))
    print("  LoHa moved zero-leg and saved modules:", loha_saved)

    var dora = build_anima_dora_set(NB, D, JOINT, F, 4, Float32(8.0), 2, UInt64(303), False)
    var dora_bytes = anima_dora_carrier_total_bytes(dora, D, JOINT, F)
    anima_dora_preflight(dora, D, JOINT, F, LOKR_CARRIER_MAX_DEVICE_BYTES)
    print("  DoRA carrier bytes:", dora_bytes)
    var dora_carriers = anima_dora_carrier_set(dora, D, JOINT, F)
    if len(dora_carriers.ad) != nslots:
        raise Error("DoRA carrier count mismatch")
    var dora_rank_in = List[Int]()
    var dora_out_rank = List[Int]()
    for i in range(len(dora_carriers.ad)):
        dora_rank_in.append(dora_carriers.ad[i].rank * dora_carriers.ad[i].in_f)
        dora_out_rank.append(dora_carriers.ad[i].out_f * dora_carriers.ad[i].rank)
    var dora_grads = _grad_lists(len(dora_carriers.ad), dora_rank_in, dora_out_rank)
    var dg = anima_dora_chain_all(dora, dora_grads[0], dora_grads[1])
    var dora_before = anima_dora_zero_leg_l1(dora)
    anima_dora_adamw_step(dora, dg, 1, Float32(1.0e-3), Float32(0.9), Float32(0.999), Float32(1.0e-8), Float32(0.01))
    var dora_after = anima_dora_zero_leg_l1(dora)
    if dora_after <= dora_before:
        raise Error("DoRA zero-leg did not move")
    var dora_saved = save_anima_dora(dora, String(DORA_OUT), ctx)
    _ = SafeTensors.open(String(DORA_OUT))
    print("  DoRA moved zero-leg and saved modules:", dora_saved)

    var oft = build_anima_oft_set(NB, D, JOINT, F, 4, 2, UInt64(404))
    var oft_bytes = anima_oft_carrier_total_bytes(oft, D, JOINT, F)
    anima_oft_preflight(oft, D, JOINT, F, LOKR_CARRIER_MAX_DEVICE_BYTES)
    print("  OFT carrier bytes:", oft_bytes)
    var oft_carriers = anima_oft_carrier_set(oft, D, JOINT, F)
    if len(oft_carriers.ad) != nslots:
        raise Error("OFT carrier count mismatch")
    var oft_rank_in = List[Int]()
    var oft_out_rank = List[Int]()
    for i in range(len(oft_carriers.ad)):
        oft_rank_in.append(oft_carriers.ad[i].rank * oft_carriers.ad[i].in_f)
        oft_out_rank.append(oft_carriers.ad[i].out_f * oft_carriers.ad[i].rank)
    var oft_grads = _grad_lists(len(oft_carriers.ad), oft_rank_in, oft_out_rank)
    var og = anima_oft_chain_all(oft, oft_grads[0], oft_grads[1])
    var oft_before = anima_oft_vec_l1(oft)
    anima_oft_adamw_step(oft, og, 1, Float32(1.0e-3), Float32(0.9), Float32(0.999), Float32(1.0e-8), Float32(0.01))
    var oft_after = anima_oft_vec_l1(oft)
    if oft_after <= oft_before:
        raise Error("OFT vec did not move")
    var oft_saved = save_anima_oft(oft, String(OFT_OUT), ctx)
    _ = SafeTensors.open(String(OFT_OUT))
    print("  OFT moved vec and saved modules:", oft_saved)
    print("ALL GATES PASS -- anima_lycoris_orchestration_smoke")
