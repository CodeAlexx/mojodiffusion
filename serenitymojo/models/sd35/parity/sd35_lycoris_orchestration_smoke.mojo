# models/sd35/parity/sd35_lycoris_orchestration_smoke.mojo
# SD3.5 LoKr/LoHa orchestration smoke: build -> carrier -> chain -> AdamW ->
# save/reopen. Shared carrier parity is tested elsewhere.

from std.collections import List
from std.gpu.host import DeviceContext
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.models.sd35.sd35_stack_lora import SLOTS_PER_BLOCK
from serenitymojo.models.sd35.sd35_lycoris_stack import (
    build_sd35_lokr_set, sd35_lokr_carrier_set, sd35_lokr_carrier_total_bytes,
    sd35_lokr_chain_all, sd35_lokr_adamw_step, sd35_lokr_zero_leg_l1,
    save_sd35_lokr,
    build_sd35_loha_set, sd35_loha_carrier_set, sd35_loha_carrier_total_bytes,
    sd35_loha_chain_all, sd35_loha_adamw_step, sd35_loha_zero_leg_l1,
    save_sd35_loha,
)

comptime LOKR_OUT = "/tmp/sd35_lokr_orch.safetensors"
comptime LOHA_OUT = "/tmp/sd35_loha_orch.safetensors"


def _fill(n: Int, seed: UInt64, scale: Float32) -> List[Float32]:
    var out = List[Float32]()
    var state = seed
    for _ in range(n):
        state = state * 6364136223846793005 + 1442695040888963407
        var u = Float32(Int(state >> 40)) * Float32(1.0 / 16777216.0)
        out.append((u - Float32(0.5)) * scale)
    return out^


def _grad_lists(rank_in: List[Int], out_rank: List[Int]) -> Tuple[List[List[Float32]], List[List[Float32]]]:
    var d_a = List[List[Float32]]()
    var d_b = List[List[Float32]]()
    for i in range(len(rank_in)):
        d_a.append(_fill(rank_in[i], UInt64(7) * UInt64(i + 1) + 1, 0.5))
        d_b.append(_fill(out_rank[i], UInt64(11) * UInt64(i + 1) + 3, 0.5))
    return (d_a^, d_b^)


def main() raises:
    var ctx = DeviceContext()
    var D = 32
    var F = 64
    var depth = 2
    var nslots = depth * SLOTS_PER_BLOCK
    print("=== SD3.5 LyCORIS orchestration smoke (D=", D, " F=", F, " depth=", depth, " slots=", nslots, ") ===")

    var lokr = build_sd35_lokr_set(
        depth, D, F, 4, Float32(8.0), -1, 0, 0, True, False, 3, UInt64(301),
    )
    print("  LoKr carrier bytes:", sd35_lokr_carrier_total_bytes(lokr, D, F))
    var lokr_carriers = sd35_lokr_carrier_set(lokr, D, F)
    if len(lokr_carriers.ad) != nslots:
        raise Error("LoKr carrier count mismatch")
    var lokr_rank_in = List[Int]()
    var lokr_out_rank = List[Int]()
    for i in range(len(lokr_carriers.ad)):
        lokr_rank_in.append(lokr_carriers.ad[i].rank * lokr_carriers.ad[i].in_f)
        lokr_out_rank.append(lokr_carriers.ad[i].out_f * lokr_carriers.ad[i].rank)
    var lokr_grads = _grad_lists(lokr_rank_in, lokr_out_rank)
    var lkg = sd35_lokr_chain_all(lokr, lokr_grads[0], lokr_grads[1])
    var lokr_before = sd35_lokr_zero_leg_l1(lokr)
    sd35_lokr_adamw_step(lokr, lkg, 1, Float32(1.0e-3), Float32(0.9), Float32(0.999), Float32(1.0e-8), Float32(0.01))
    var lokr_after = sd35_lokr_zero_leg_l1(lokr)
    if lokr_after <= lokr_before:
        raise Error("LoKr zero-leg did not move")
    var lokr_saved = save_sd35_lokr(lokr, String(LOKR_OUT), ctx)
    _ = SafeTensors.open(String(LOKR_OUT))
    print("  LoKr moved zero-leg and saved modules:", lokr_saved)

    var loha = build_sd35_loha_set(depth, D, F, 3, Float32(6.0), 3, UInt64(302))
    print("  LoHa carrier bytes:", sd35_loha_carrier_total_bytes(loha, D, F))
    var loha_carriers = sd35_loha_carrier_set(loha, D, F)
    if len(loha_carriers.ad) != nslots:
        raise Error("LoHa carrier count mismatch")
    var loha_rank_in = List[Int]()
    var loha_out_rank = List[Int]()
    for i in range(len(loha_carriers.ad)):
        loha_rank_in.append(loha_carriers.ad[i].rank * loha_carriers.ad[i].in_f)
        loha_out_rank.append(loha_carriers.ad[i].out_f * loha_carriers.ad[i].rank)
    var loha_grads = _grad_lists(loha_rank_in, loha_out_rank)
    var lhg = sd35_loha_chain_all(loha, loha_grads[0], loha_grads[1])
    var loha_before = sd35_loha_zero_leg_l1(loha)
    sd35_loha_adamw_step(loha, lhg, 1, Float32(1.0e-3), Float32(0.9), Float32(0.999), Float32(1.0e-8), Float32(0.01))
    var loha_after = sd35_loha_zero_leg_l1(loha)
    if loha_after <= loha_before:
        raise Error("LoHa zero-leg did not move")
    var loha_saved = save_sd35_loha(loha, String(LOHA_OUT), ctx)
    _ = SafeTensors.open(String(LOHA_OUT))
    print("  LoHa moved zero-leg and saved modules:", loha_saved)
    print("ALL GATES PASS -- sd35_lycoris_orchestration_smoke")
