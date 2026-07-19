# models/qwenimage/parity/qwen_lycoris_orchestration_smoke.mojo
# Qwen-Image LoKr/LoHa orchestration gate: build -> carrier -> chain -> AdamW
# -> save/reopen. Carrier math is covered by the shared LoKr/LoHa parity tests.
#
# Build/run:
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#   pixi run mojo build --optimization-level 0 -I . -Xlinker -lm -Xlinker -lcuda \
#     -Xlinker -L.pixi/envs/default/lib -Xlinker -lsqlite3 \
#     serenitymojo/models/qwenimage/parity/qwen_lycoris_orchestration_smoke.mojo \
#     -o /tmp/qwen_lycoris_orch && /tmp/qwen_lycoris_orch

from std.collections import List
from std.gpu.host import DeviceContext
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.models.qwenimage.qwenimage_stack_lora import DBL_SLOTS
from serenitymojo.models.qwenimage.qwenimage_lycoris_stack import (
    build_qwen_lokr_set, qwen_lokr_carrier_set, qwen_lokr_carrier_total_bytes,
    qwen_lokr_chain_all, qwen_lokr_adamw_step, qwen_lokr_zero_leg_l1,
    save_qwen_lokr,
    build_qwen_loha_set, qwen_loha_carrier_set, qwen_loha_carrier_total_bytes,
    qwen_loha_chain_all, qwen_loha_adamw_step, qwen_loha_zero_leg_l1,
    save_qwen_loha,
)

comptime LOKR_OUT = "/tmp/qwen_lokr_orch.safetensors"
comptime LOHA_OUT = "/tmp/qwen_loha_orch.safetensors"


def _fill(n: Int, seed: UInt64, scale: Float32) -> List[Float32]:
    var out = List[Float32]()
    var state = seed
    for _ in range(n):
        state = state * 6364136223846793005 + 1442695040888963407
        var u = Float32(Int(state >> 40)) * Float32(1.0 / 16777216.0)
        out.append((u - Float32(0.5)) * scale)
    return out^


def _grad_lists(n: Int, rank_in: List[Int], out_rank: List[Int]) -> Tuple[List[List[Float32]], List[List[Float32]]]:
    var d_a = List[List[Float32]]()
    var d_b = List[List[Float32]]()
    for i in range(n):
        d_a.append(_fill(rank_in[i], UInt64(7) * UInt64(i + 1) + 1, 0.5))
        d_b.append(_fill(out_rank[i], UInt64(11) * UInt64(i + 1) + 3, 0.5))
    return (d_a^, d_b^)


def _carrier_grad_shapes_lokr() raises -> Tuple[List[List[Float32]], List[List[Float32]]]:
    var D = 32
    var F = 48
    var NB = 2
    var masters = build_qwen_lokr_set(
        NB, D, F, 4, Float32(8.0), -1, 0, 0, True, False, 3, UInt64(101),
    )
    var carriers = qwen_lokr_carrier_set(masters, D, F)
    var rank_in = List[Int]()
    var out_rank = List[Int]()
    for i in range(len(carriers.dbl)):
        rank_in.append(carriers.dbl[i].rank * carriers.dbl[i].in_f)
        out_rank.append(carriers.dbl[i].out_f * carriers.dbl[i].rank)
    return _grad_lists(len(carriers.dbl), rank_in, out_rank)


def main() raises:
    var ctx = DeviceContext()
    var D = 32
    var F = 48
    var NB = 2
    var nslots = NB * DBL_SLOTS
    print("=== Qwen LyCORIS orchestration smoke (D=", D, " F=", F, " blocks=", NB, " slots=", nslots, ") ===")

    var lokr = build_qwen_lokr_set(
        NB, D, F, 4, Float32(8.0), -1, 0, 0, True, False, 3, UInt64(101),
    )
    print("  LoKr carrier bytes:", qwen_lokr_carrier_total_bytes(lokr, D, F))
    var lokr_carriers = qwen_lokr_carrier_set(lokr, D, F)
    if len(lokr_carriers.dbl) != nslots:
        raise Error("LoKr carrier count mismatch")
    var lokr_rank_in = List[Int]()
    var lokr_out_rank = List[Int]()
    for i in range(len(lokr_carriers.dbl)):
        lokr_rank_in.append(lokr_carriers.dbl[i].rank * lokr_carriers.dbl[i].in_f)
        lokr_out_rank.append(lokr_carriers.dbl[i].out_f * lokr_carriers.dbl[i].rank)
    var lokr_grads = _grad_lists(len(lokr_carriers.dbl), lokr_rank_in, lokr_out_rank)
    var lkg = qwen_lokr_chain_all(lokr, lokr_grads[0], lokr_grads[1])
    var lokr_before = qwen_lokr_zero_leg_l1(lokr)
    qwen_lokr_adamw_step(lokr, lkg, 1, Float32(1.0e-3), Float32(0.9), Float32(0.999), Float32(1.0e-8), Float32(0.01))
    var lokr_after = qwen_lokr_zero_leg_l1(lokr)
    if lokr_after <= lokr_before:
        raise Error("LoKr zero-leg did not move")
    var lokr_saved = save_qwen_lokr(lokr, String(LOKR_OUT), ctx)
    _ = SafeTensors.open(String(LOKR_OUT))
    print("  LoKr moved zero-leg and saved modules:", lokr_saved)

    var loha = build_qwen_loha_set(NB, D, F, 3, Float32(6.0), 3, UInt64(202))
    print("  LoHa carrier bytes:", qwen_loha_carrier_total_bytes(loha, D, F))
    var loha_carriers = qwen_loha_carrier_set(loha, D, F)
    if len(loha_carriers.dbl) != nslots:
        raise Error("LoHa carrier count mismatch")
    var loha_rank_in = List[Int]()
    var loha_out_rank = List[Int]()
    for i in range(len(loha_carriers.dbl)):
        loha_rank_in.append(loha_carriers.dbl[i].rank * loha_carriers.dbl[i].in_f)
        loha_out_rank.append(loha_carriers.dbl[i].out_f * loha_carriers.dbl[i].rank)
    var loha_grads = _grad_lists(len(loha_carriers.dbl), loha_rank_in, loha_out_rank)
    var lhg = qwen_loha_chain_all(loha, loha_grads[0], loha_grads[1])
    var loha_before = qwen_loha_zero_leg_l1(loha)
    qwen_loha_adamw_step(loha, lhg, 1, Float32(1.0e-3), Float32(0.9), Float32(0.999), Float32(1.0e-8), Float32(0.01))
    var loha_after = qwen_loha_zero_leg_l1(loha)
    if loha_after <= loha_before:
        raise Error("LoHa zero-leg did not move")
    var loha_saved = save_qwen_loha(loha, String(LOHA_OUT), ctx)
    _ = SafeTensors.open(String(LOHA_OUT))
    print("  LoHa moved zero-leg and saved modules:", loha_saved)
    print("ALL GATES PASS -- qwen_lycoris_orchestration_smoke")
