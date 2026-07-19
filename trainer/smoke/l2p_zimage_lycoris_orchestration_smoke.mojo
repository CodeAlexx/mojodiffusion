# smoke/l2p_zimage_lycoris_orchestration_smoke.mojo
#
# PER-MODEL LoKr + LoHa CARRIER ORCHESTRATION smoke for the L2P (Z-Image L2P)
# trainer's REAL slot geometry. L2P dispatches LyCORIS through the SHARED ZIMAGE
# carrier stack (serenitymojo/models/zimage/zimage_{lokr,loha}_stack.mojo):
# build_zimage_{lokr,loha}_set → zimage_{lokr,loha}_carrier_lists →
# zimage_{lokr,loha}_chain_all → adamw → save. No per-model orchestration gate
# existed; this closes that audit cell.
#
# L2P vs Z-IMAGE geometry: L2P reuses the Z-Image carrier stack VERBATIM. The
# slot set is the SAME 7-slot flat geometry (q/k/v/o + SwiGLU w1/w3/w2); the
# per-model delta is the block partition NR=2 / CR=2 / MAIN=30 (34 blocks) at
# D=3840, F=10240, and — crucially — L2P TRAINS ONLY THE MAIN BLOCKS: it flips
# the first TRAIN_ADAPTER_START = (NR+CR)*7 = 28 adapters inactive
# (train_l2p_real.mojo:190,866-867). This smoke reproduces that: builds all 238,
# deactivates the leading 28, and gates the 210 active MAIN carriers.
#
# Proves at L2P's REAL dims (NOT toy):
#   (0) SLOT COUNT: carrier set = (NR+CR+MAIN)*7 = 34*7 = 238; active = 30*7 = 210.
#   (1) SLOT DIMS: every ACTIVE carrier's (in_f,out_f) equals zimage_lokr_slot_dims
#       (attn q/k/v/o D→D, SwiGLU w1/w3 D→F, w2 F→D). Mismatch = REAL geometry bug.
#   (2) RECONSTRUCTION r_eff: LoKr both-factored → r_eff=rank² (kron); every LoHa
#       → r_eff=rank² (hadamard). Numeric kron/hada gated by core carrier parity.
#   (3) ZERO-LEG INIT: zimage_*_zero_leg_l1 == 0 at init → initial ΔW ZERO.
#   (4) UPDATE FLOW: one master AdamW step on synthetic carrier grads moves the
#       zero-leg off zero (only the 210 active MAIN masters update).
#   (5) SAVE: save_zimage_* writes >0 modules and the file reopens.
#
# DERIVED GEOMETRY (cited):
#   D=3840, F=10240, NUM_NR=2, NUM_CR=2, MAIN_DEPTH=30   train_l2p_real.mojo:140-141,172-174
#   TRAIN_ADAPTER_START = (NR+CR)*ZIMAGE_SLOTS = 28      train_l2p_real.mojo:190
#   ZIMAGE_SLOTS=7, slot dims                            zimage_lokr_stack.mojo:3, zimage_lokr_slot_dims
#   build/deactivate/carrier call site                  train_l2p_real.mojo:863-882
#
# BUILD (deliverable — NO GPU run needed to compile):
#   cd trainer && rm -f /home/alex/mojodiffusion/serenitymojo.mojopkg && \
#   MEM_MAX=24G MEM_HIGH=20G pixi run bash /home/alex/mojodiffusion/scripts/mem_safe.sh \
#     mojo build -O2 -I . -I src -I /home/alex/mojodiffusion \
#       -Xlinker -lm -Xlinker -lcuda \
#       -Xlinker -L/home/alex/mojodiffusion/.pixi/envs/default/lib -Xlinker -lsqlite3 \
#       smoke/l2p_zimage_lycoris_orchestration_smoke.mojo \
#       -o target/l2p_zimage_lycoris_orchestration_smoke
# RUN (optional; writes /tmp/l2p_{lokr,loha}.safetensors):
#   ./target/l2p_zimage_lycoris_orchestration_smoke
# EXPECT: "ALL GATES PASS — l2p_zimage_lycoris_orchestration_smoke".

from std.collections import List
from std.gpu.host import DeviceContext
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.models.zimage.lora_block import ZIMAGE_SLOTS
from serenitymojo.models.zimage.zimage_lokr_stack import (
    build_zimage_lokr_set, zimage_lokr_carrier_lists,
    zimage_lokr_chain_all, zimage_lokr_adamw_step, zimage_lokr_zero_leg_l1,
    save_zimage_lokr, zimage_lokr_slot_dims,
)
from serenitymojo.models.zimage.zimage_loha_stack import (
    build_zimage_loha_set, zimage_loha_carrier_lists,
    zimage_loha_chain_all, zimage_loha_adamw_step, zimage_loha_zero_leg_l1,
    save_zimage_loha,
)
from serenitymojo.training.lokr_stack import lokr_carrier_r_eff

comptime D = 3840
comptime F = 10240
comptime NUM_NR = 2
comptime NUM_CR = 2
comptime MAIN_DEPTH = 30
comptime RANK = 4
comptime ALPHA = Float32(8.0)
comptime FACTOR = -1
comptime DECOMPOSE_BOTH = True
comptime FULL_MATRIX = False
comptime TARGETS = 2               # 2 = all (attn + SwiGLU) → every slot targeted


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
    var total_blocks = NUM_NR + NUM_CR + MAIN_DEPTH
    var expected_total = total_blocks * ZIMAGE_SLOTS
    var train_start = (NUM_NR + NUM_CR) * ZIMAGE_SLOTS   # 28: leading NR+CR inactive
    var expected_active = MAIN_DEPTH * ZIMAGE_SLOTS      # 210
    print("=== L2P Z-Image-LyCORIS carrier orchestration smoke ===")
    print("  blocks NR+CR+MAIN =", total_blocks, " ZIMAGE_SLOTS=", ZIMAGE_SLOTS,
          " total carriers =", expected_total, " active(main) =", expected_active,
          " (deactivate first", train_start, ")")

    # ─────────────────────── LoKr phase ───────────────────────
    print("-- LoKr --")
    var set = build_zimage_lokr_set(
        NUM_NR, NUM_CR, MAIN_DEPTH, D, F, RANK, ALPHA,
        FACTOR, DECOMPOSE_BOTH, FULL_MATRIX, TARGETS, UInt64(53) * UInt64(11) + 11,
    )
    for i in range(train_start):
        set.active[i] = False       # mirror train_l2p_real.mojo:866-867 (main-only)
    var carriers = zimage_lokr_carrier_lists(set, D, F)
    if len(carriers) != expected_total:
        raise Error("L2P LoKr carrier count " + String(len(carriers))
            + " != expected " + String(expected_total))
    var active = 0
    var l3 = 0
    for i in range(len(carriers)):
        if not set.active[i]:
            continue
        active += 1
        var slot = i % ZIMAGE_SLOTS
        var e = zimage_lokr_slot_dims(slot, D, F)
        if carriers[i].in_f != e[0] or carriers[i].out_f != e[1]:
            raise Error("L2P LoKr GEOMETRY MISMATCH idx " + String(i) + " slot "
                + String(slot) + ": carrier (" + String(carriers[i].in_f) + ","
                + String(carriers[i].out_f) + ") != (" + String(e[0]) + ","
                + String(e[1]) + ")")
        ref lo = set.ad[i]
        if carriers[i].rank != lokr_carrier_r_eff(lo):
            raise Error("L2P LoKr r_eff carrier/master mismatch idx " + String(i))
        if lo.w1_factored and lo.w2_factored:
            if lokr_carrier_r_eff(lo) != RANK * RANK:
                raise Error("L2P LoKr L3 r_eff " + String(lokr_carrier_r_eff(lo))
                    + " != rank² " + String(RANK * RANK) + " idx " + String(i))
            l3 += 1
    if active != expected_active:
        raise Error("L2P LoKr active carriers " + String(active)
            + " != expected " + String(expected_active))
    var zb = zimage_lokr_zero_leg_l1(set)
    if zb != Float64(0.0):
        raise Error("L2P LoKr zero-leg L1 at init = " + String(zb) + " (expect 0)")
    var d_a = List[List[Float32]]()
    var d_b = List[List[Float32]]()
    for i in range(len(carriers)):
        var r = carriers[i].rank
        d_a.append(_fill(r * carriers[i].in_f, UInt64(7) * UInt64(i + 1) + 1, 0.5))
        d_b.append(_fill(carriers[i].out_f * r, UInt64(13) * UInt64(i + 1) + 3, 0.5))
    zimage_lokr_adamw_step(set, zimage_lokr_chain_all(set, d_a, d_b), 1,
                           Float32(1.0e-3), Float32(0.9), Float32(0.999), Float32(1.0e-8), Float32(0.01))
    var za = zimage_lokr_zero_leg_l1(set)
    if za <= zb:
        raise Error("L2P LoKr AdamW zero-leg no-op (before=" + String(zb)
            + " after=" + String(za) + ")")
    print("  carriers:", len(carriers), " active:", active, " (L3:", l3, ")")
    print("  zero-leg L1: init =", zb, " after-step =", za)
    var nl = save_zimage_lokr(set, "/tmp/l2p_lokr.safetensors", ctx)
    print("  save_zimage_lokr modules:", nl)
    if nl <= 0:
        raise Error("L2P LoKr saved 0 modules")
    _ = SafeTensors.open("/tmp/l2p_lokr.safetensors")
    print("  reopened LoKr file ✓")

    # ─────────────────────── LoHa phase ───────────────────────
    print("-- LoHa --")
    var lset = build_zimage_loha_set(
        NUM_NR, NUM_CR, MAIN_DEPTH, D, F, RANK, ALPHA, TARGETS,
        UInt64(53) * UInt64(11) + 11,
    )
    for i in range(train_start):
        lset.active[i] = False
    var lcar = zimage_loha_carrier_lists(lset, D, F)
    if len(lcar) != expected_total:
        raise Error("L2P LoHa carrier count mismatch")
    var lactive = 0
    for i in range(len(lcar)):
        if not lset.active[i]:
            continue
        lactive += 1
        var slot = i % ZIMAGE_SLOTS
        var e = zimage_lokr_slot_dims(slot, D, F)
        if lcar[i].in_f != e[0] or lcar[i].out_f != e[1]:
            raise Error("L2P LoHa GEOMETRY MISMATCH idx " + String(i) + " slot "
                + String(slot) + ": carrier (" + String(lcar[i].in_f) + ","
                + String(lcar[i].out_f) + ") != (" + String(e[0]) + "," + String(e[1]) + ")")
        if lcar[i].rank != RANK * RANK:
            raise Error("L2P LoHa r_eff " + String(lcar[i].rank)
                + " != rank² " + String(RANK * RANK) + " idx " + String(i))
    if lactive != expected_active:
        raise Error("L2P LoHa active carriers mismatch")
    var lzb = zimage_loha_zero_leg_l1(lset)
    if lzb != Float64(0.0):
        raise Error("L2P LoHa zero-leg L1 at init = " + String(lzb) + " (expect 0)")
    var la = List[List[Float32]]()
    var lb = List[List[Float32]]()
    for i in range(len(lcar)):
        var r = lcar[i].rank
        la.append(_fill(r * lcar[i].in_f, UInt64(7) * UInt64(i + 1) + 1, 0.5))
        lb.append(_fill(lcar[i].out_f * r, UInt64(13) * UInt64(i + 1) + 3, 0.5))
    zimage_loha_adamw_step(lset, zimage_loha_chain_all(lset, la, lb), 1,
                           Float32(1.0e-3), Float32(0.9), Float32(0.999), Float32(1.0e-8), Float32(0.01))
    var lza = zimage_loha_zero_leg_l1(lset)
    if lza <= lzb:
        raise Error("L2P LoHa AdamW zero-leg no-op")
    print("  carriers:", len(lcar), " active:", lactive, " (all r_eff=rank²)")
    print("  zero-leg L1: init =", lzb, " after-step =", lza)
    var nh = save_zimage_loha(lset, "/tmp/l2p_loha.safetensors", ctx)
    print("  save_zimage_loha modules:", nh)
    if nh <= 0:
        raise Error("L2P LoHa saved 0 modules")
    _ = SafeTensors.open("/tmp/l2p_loha.safetensors")
    print("  reopened LoHa file ✓")

    print("ALL GATES PASS — l2p_zimage_lycoris_orchestration_smoke")
