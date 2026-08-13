# smoke/sdxl_flat_lycoris_orchestration_smoke.mojo
#
# PER-MODEL LoKr + LoHa CARRIER ORCHESTRATION smoke for the SDXL trainer's REAL
# slot geometry. SDXL dispatches LyCORIS through the SHARED flat carrier stack
# (serenitymojo/training/flat_lycoris_stack.mojo): build_flat_{lokr,loha}_set →
# flat_{lokr,loha}_carrier_list → flat_{lokr,loha}_chain_all → adamw → save.
# No per-model orchestration gate existed; this closes that audit cell.
#
# It proves, at SDXL's REAL per-ST channel geometry (NOT toy dims):
#   (0) SLOT COUNT: the built carrier set has exactly sum(depth_i)*SDXL_SLOTS=700
#       carriers across the 11 SpatialTransformers.
#   (1) SLOT DIMS: every carrier's (in_f,out_f) equals the SDXL slot rule
#       (attn q/k/v/o, cross-attn k/v from CCTX, GEGLU proj out=2*Cff, ff.net.2
#       in=Cff). A mismatch here = a REAL geometry bug in the shared stack.
#   (2) RECONSTRUCTION r_eff: for every LoKr slot that factorizes BOTH sides
#       (decompose_both L3) the carrier expands to r_eff=rank² (kron formula);
#       every LoHa carrier expands to r_eff=rank² (hadamard). Numeric kron/hada
#       VALUES are gated by the core carrier parity (lokr_stack/loha_stack
#       carrier tests) — here we gate the r_eff the model's real dims produce.
#   (3) ZERO-LEG INIT: flat_*_zero_leg_l1 == 0 at init → initial ΔW is ZERO.
#   (4) UPDATE FLOW: after one master AdamW step on synthetic carrier grads the
#       zero-leg moves off zero → gradients chain carrier→master and update.
#   (5) SAVE: save_flat_* writes >0 modules per ST and the file reopens.
#
# DERIVED GEOMETRY (cited):
#   N_ST=11, SDXL_SLOTS=10                     serenitymojo/models/sdxl/sdxl_real_train.mojo:232, lora_block.mojo:56
#   per-ST C/Cff/depth                         serenitymojo/models/sdxl/real_weights.mojo:204-224 (Cff=4*C)
#   slot in/out rule (CCTX=2048)               serenity_trainer/trainer/train_sdxl_real.mojo:466-478,137
#   SDXL LoKr/LoHa trains ALL 10 slots/block   train_sdxl_real.mojo:_build_sdxl_flat_lokr (no _sdxl_flat_active mask)
#
# BUILD (deliverable — NO GPU run needed to compile):
#   cd trainer && rm -f /home/alex/mojodiffusion/serenitymojo.mojopkg && \
#   MEM_MAX=24G MEM_HIGH=20G pixi run bash /home/alex/mojodiffusion/scripts/mem_safe.sh \
#     mojo build -O2 -I . -I src -I /home/alex/mojodiffusion \
#       -Xlinker -lm -Xlinker -lcuda \
#       -Xlinker -L/home/alex/mojodiffusion/.pixi/envs/default/lib -Xlinker -lsqlite3 \
#       smoke/sdxl_flat_lycoris_orchestration_smoke.mojo \
#       -o target/sdxl_flat_lycoris_orchestration_smoke
# RUN (optional, host+GPU-ctx; writes /tmp/sdxl_*_stage*.safetensors):
#   ./target/sdxl_flat_lycoris_orchestration_smoke
# EXPECT: "ALL GATES PASS — sdxl_flat_lycoris_orchestration_smoke".

from std.collections import List
from max.gpu.host import DeviceContext
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.models.sdxl.real_weights import (
    sdxl_st_C, sdxl_st_Cff, sdxl_st_depth, sdxl_st_prefixes,
)
from serenitymojo.models.sdxl.sdxl_unet_stack_lora import sdxl_lora_prefixes
from serenitymojo.models.sdxl.lora_block import SDXL_SLOTS
from serenitymojo.models.sdxl.sdxl_real_train import N_ST
from serenitymojo.training.flat_lycoris_stack import (
    FlatLoKrSet, build_flat_lokr_set, flat_lokr_carrier_list,
    flat_lokr_chain_all, flat_lokr_adamw_step, flat_lokr_zero_leg_l1, save_flat_lokr,
    FlatLoHaSet, build_flat_loha_set, flat_loha_carrier_list,
    flat_loha_chain_all, flat_loha_adamw_step, flat_loha_zero_leg_l1, save_flat_loha,
)
from serenitymojo.training.lokr_stack import lokr_carrier_r_eff

comptime CCTX = 2048               # train_sdxl_real.mojo:137
comptime RANK = 4
comptime ALPHA = Float32(8.0)
comptime FACTOR = -1               # auto sqrt-ish factorization
comptime DECOMPOSE_BOTH = True     # LoKr L3 both-factored → r_eff = rank²
comptime FULL_MATRIX = False


def _fill(n: Int, seed: UInt64, scale: Float32) -> List[Float32]:
    var out = List[Float32]()
    var state = seed
    for _ in range(n):
        state = state * 6364136223846793005 + 1442695040888963407
        var u = Float32(Int(state >> 40)) * Float32(1.0 / 16777216.0)
        out.append((u - Float32(0.5)) * scale)
    return out^


# Replicated from train_sdxl_real.mojo:466-478 (_sdxl_slot_in/_sdxl_slot_out).
def _slot_in(slot: Int, C: Int, Cff: Int) -> Int:
    if slot == 5 or slot == 6:     # attn2 cross-attn k/v ← context
        return CCTX
    if slot == 9:                  # ff.net.2 in = Cff
        return Cff
    return C


def _slot_out(slot: Int, C: Int, Cff: Int) -> Int:
    if slot == 8:                  # ff.net.0 GEGLU proj out = 2*Cff
        return 2 * Cff
    return C


def _stage_in_dims(depth: Int, C: Int, Cff: Int) -> List[Int]:
    var out = List[Int]()
    for _bi in range(depth):
        for slot in range(SDXL_SLOTS):
            out.append(_slot_in(slot, C, Cff))
    return out^


def _stage_out_dims(depth: Int, C: Int, Cff: Int) -> List[Int]:
    var out = List[Int]()
    for _bi in range(depth):
        for slot in range(SDXL_SLOTS):
            out.append(_slot_out(slot, C, Cff))
    return out^


def _expected_total() -> Int:
    var t = 0
    for i in range(N_ST):
        t += sdxl_st_depth(i) * SDXL_SLOTS
    return t


def main() raises:
    var ctx = DeviceContext()
    var prefixes = sdxl_st_prefixes()
    var expected_total = _expected_total()
    print("=== SDXL flat-LyCORIS carrier orchestration smoke ===")
    print("  N_ST=", N_ST, " SDXL_SLOTS=", SDXL_SLOTS,
          " expected total carriers =", expected_total, " (sum depth*10)")

    # ─────────────────────── LoKr phase ───────────────────────
    print("-- LoKr --")
    var total = 0
    var l3 = 0
    var zsum_before = Float64(0.0)
    var zsum_after = Float64(0.0)
    var saved = 0
    for i in range(N_ST):
        var C = sdxl_st_C(i)
        var Cff = sdxl_st_Cff(i)
        var depth = sdxl_st_depth(i)
        var ins = _stage_in_dims(depth, C, Cff)
        var outs = _stage_out_dims(depth, C, Cff)
        var names = sdxl_lora_prefixes(prefixes[i], depth)
        var ms = build_flat_lokr_set(
            ins, outs, names, RANK, ALPHA, FACTOR,
            DECOMPOSE_BOTH, FULL_MATRIX, UInt64(11) * UInt64(i + 1) + 1,
        )
        var carriers = flat_lokr_carrier_list(ms)
        if len(carriers) != depth * SDXL_SLOTS:
            raise Error("SDXL LoKr ST " + String(i) + ": carrier count "
                + String(len(carriers)) + " != depth*SDXL_SLOTS "
                + String(depth * SDXL_SLOTS))
        for k in range(len(carriers)):
            var slot = k % SDXL_SLOTS
            var ein = _slot_in(slot, C, Cff)
            var eout = _slot_out(slot, C, Cff)
            if carriers[k].in_f != ein or carriers[k].out_f != eout:
                raise Error("SDXL LoKr GEOMETRY MISMATCH ST " + String(i)
                    + " slot " + String(slot) + ": carrier ("
                    + String(carriers[k].in_f) + "," + String(carriers[k].out_f)
                    + ") != expected (" + String(ein) + "," + String(eout) + ")")
            ref lo = ms.ad[k]
            if carriers[k].rank != lokr_carrier_r_eff(lo):
                raise Error("SDXL LoKr r_eff carrier/master mismatch ST "
                    + String(i) + " slot " + String(slot))
            if lo.w1_factored and lo.w2_factored:
                if lokr_carrier_r_eff(lo) != RANK * RANK:
                    raise Error("SDXL LoKr L3 r_eff " + String(lokr_carrier_r_eff(lo))
                        + " != rank² " + String(RANK * RANK) + " ST " + String(i)
                        + " slot " + String(slot))
                l3 += 1
        var zb = flat_lokr_zero_leg_l1(ms)
        if zb != Float64(0.0):
            raise Error("SDXL LoKr ST " + String(i)
                + ": zero-leg L1 at init = " + String(zb) + " (expect 0 → ΔW=0)")
        var d_a = List[List[Float32]]()
        var d_b = List[List[Float32]]()
        for k in range(len(carriers)):
            var r = carriers[k].rank
            d_a.append(_fill(r * carriers[k].in_f, UInt64(7) * UInt64(k + 1) + 1, 0.5))
            d_b.append(_fill(carriers[k].out_f * r, UInt64(13) * UInt64(k + 1) + 3, 0.5))
        var g = flat_lokr_chain_all(ms, d_a, d_b)
        flat_lokr_adamw_step(ms, g, 1, Float32(1.0e-3),
                             Float32(0.9), Float32(0.999), Float32(1.0e-8), Float32(0.01))
        var za = flat_lokr_zero_leg_l1(ms)
        if za <= zb:
            raise Error("SDXL LoKr ST " + String(i)
                + ": AdamW did not move zero-leg off zero (before=" + String(zb)
                + " after=" + String(za) + ")")
        zsum_before += zb
        zsum_after += za
        total += len(carriers)
        saved += save_flat_lokr(ms, "/tmp/sdxl_lokr_stage" + String(i) + ".safetensors", ctx)
    if total != expected_total:
        raise Error("SDXL LoKr total carriers " + String(total)
            + " != expected " + String(expected_total))
    print("  carriers:", total, " (L3 both-factored:", l3, ")")
    print("  zero-leg L1: init sum =", zsum_before, " after-step sum =", zsum_after)
    print("  save_flat_lokr modules across 11 STs:", saved)
    if saved <= 0:
        raise Error("SDXL LoKr saved 0 modules")
    _ = SafeTensors.open("/tmp/sdxl_lokr_stage0.safetensors")
    print("  reopened stage0 LoKr file ✓")

    # ─────────────────────── LoHa phase ───────────────────────
    print("-- LoHa --")
    total = 0
    zsum_before = Float64(0.0)
    zsum_after = Float64(0.0)
    saved = 0
    for i in range(N_ST):
        var C = sdxl_st_C(i)
        var Cff = sdxl_st_Cff(i)
        var depth = sdxl_st_depth(i)
        var ins = _stage_in_dims(depth, C, Cff)
        var outs = _stage_out_dims(depth, C, Cff)
        var names = sdxl_lora_prefixes(prefixes[i], depth)
        var ms = build_flat_loha_set(ins, outs, names, RANK, ALPHA,
                                     UInt64(17) * UInt64(i + 1) + 5)
        var carriers = flat_loha_carrier_list(ms)
        if len(carriers) != depth * SDXL_SLOTS:
            raise Error("SDXL LoHa ST " + String(i) + ": carrier count mismatch")
        for k in range(len(carriers)):
            var slot = k % SDXL_SLOTS
            var ein = _slot_in(slot, C, Cff)
            var eout = _slot_out(slot, C, Cff)
            if carriers[k].in_f != ein or carriers[k].out_f != eout:
                raise Error("SDXL LoHa GEOMETRY MISMATCH ST " + String(i)
                    + " slot " + String(slot) + ": carrier ("
                    + String(carriers[k].in_f) + "," + String(carriers[k].out_f)
                    + ") != expected (" + String(ein) + "," + String(eout) + ")")
            if carriers[k].rank != RANK * RANK:
                raise Error("SDXL LoHa r_eff " + String(carriers[k].rank)
                    + " != rank² " + String(RANK * RANK) + " ST " + String(i)
                    + " slot " + String(slot))
        var zb = flat_loha_zero_leg_l1(ms)
        if zb != Float64(0.0):
            raise Error("SDXL LoHa ST " + String(i)
                + ": zero-leg L1 at init = " + String(zb) + " (expect 0)")
        var d_a = List[List[Float32]]()
        var d_b = List[List[Float32]]()
        for k in range(len(carriers)):
            var r = carriers[k].rank
            d_a.append(_fill(r * carriers[k].in_f, UInt64(7) * UInt64(k + 1) + 1, 0.5))
            d_b.append(_fill(carriers[k].out_f * r, UInt64(13) * UInt64(k + 1) + 3, 0.5))
        var g = flat_loha_chain_all(ms, d_a, d_b)
        flat_loha_adamw_step(ms, g, 1, Float32(1.0e-3),
                             Float32(0.9), Float32(0.999), Float32(1.0e-8), Float32(0.01))
        var za = flat_loha_zero_leg_l1(ms)
        if za <= zb:
            raise Error("SDXL LoHa ST " + String(i) + ": AdamW zero-leg no-op")
        zsum_before += zb
        zsum_after += za
        total += len(carriers)
        saved += save_flat_loha(ms, "/tmp/sdxl_loha_stage" + String(i) + ".safetensors", ctx)
    if total != expected_total:
        raise Error("SDXL LoHa total carriers mismatch")
    print("  carriers:", total, " (all r_eff=rank²)")
    print("  zero-leg L1: init sum =", zsum_before, " after-step sum =", zsum_after)
    print("  save_flat_loha modules across 11 STs:", saved)
    if saved <= 0:
        raise Error("SDXL LoHa saved 0 modules")
    _ = SafeTensors.open("/tmp/sdxl_loha_stage0.safetensors")
    print("  reopened stage0 LoHa file ✓")

    print("ALL GATES PASS — sdxl_flat_lycoris_orchestration_smoke")
