# smoke/chroma_flux_lycoris_orchestration_smoke.mojo
#
# PER-MODEL LoKr + LoHa CARRIER ORCHESTRATION smoke for the Chroma trainer's REAL
# slot geometry. Chroma dispatches LyCORIS through the SHARED FLUX carrier stack
# (serenitymojo/models/flux/flux_lycoris_stack.mojo): build_flux_{lokr,loha}_set
# → flux_{lokr,loha}_carrier_set → flux_{lokr,loha}_chain_all → adamw → save.
# No per-model orchestration gate existed; this closes that audit cell.
#
# CHROMA vs FLUX geometry: Chroma IS a FLUX-family DiT and reuses the flux slot
# set VERBATIM — the ONLY delta is depth. Chroma = NUM_DOUBLE=19 / NUM_SINGLE=38
# double/single blocks at D=3072, FMLP=12288 (train_chroma_real.mojo:146-152),
# vs stock FLUX.1 = 19/38 as well; the carrier slot dims (dbl 6-slot, sgl 5-slot)
# are identical. So this smoke drives flux_lycoris_stack at CHROMA's actual
# (num_double,num_single,D,F,rank,alpha) and asserts the produced carrier dims
# equal the flux slot rule for those dims.
#
# Proves at Chroma's REAL dims (NOT toy):
#   (0) SLOT COUNT: carrier set = NUM_DOUBLE*2*6 + NUM_SINGLE*5 = 228+190 = 418.
#   (1) SLOT DIMS: every carrier's (in_f,out_f) equals flux_lycoris_{dbl,sgl}_slot_dims
#       (img+txt streams; D_MLP0 D→F, D_MLP2 F→D; sgl S_PMLP D→F, S_L2 (D+F)→D).
#       Mismatch = REAL geometry bug in the shared FLUX stack for Chroma.
#   (2) RECONSTRUCTION r_eff: LoKr both-factored → r_eff=rank² (kron); every
#       LoHa → r_eff=rank² (hadamard). Numeric kron/hada gated by core parity.
#   (3) ZERO-LEG INIT: flux_*_zero_leg_l1 == 0 at init → initial ΔW ZERO.
#   (4) UPDATE FLOW: one master AdamW step on synthetic carrier grads moves the
#       zero-leg off zero.
#   (5) SAVE: save_flux_* writes >0 modules and the file reopens.
#
# DERIVED GEOMETRY (cited):
#   D=3072, FMLP=12288, NUM_DOUBLE=19, NUM_SINGLE=38   train_chroma_real.mojo:146-152
#   RANK=16, ALPHA=16                                  train_chroma_real.mojo:168-169
#   DBL_STREAM_SLOTS=6, SGL_SLOTS=5                    serenitymojo/models/flux/lora_block.mojo:271,280
#   slot dims (public helpers)                         flux_lycoris_stack.mojo:42-59
#   build/carrier/chain call site                      train_chroma_real.mojo:622-655
#
# BUILD (deliverable — NO GPU run needed to compile):
#   cd trainer && rm -f /home/alex/mojodiffusion/serenitymojo.mojopkg && \
#   MEM_MAX=24G MEM_HIGH=20G pixi run bash /home/alex/mojodiffusion/scripts/mem_safe.sh \
#     mojo build -O2 -I . -I src -I /home/alex/mojodiffusion \
#       -Xlinker -lm -Xlinker -lcuda \
#       -Xlinker -L/home/alex/mojodiffusion/.pixi/envs/default/lib -Xlinker -lsqlite3 \
#       smoke/chroma_flux_lycoris_orchestration_smoke.mojo \
#       -o target/chroma_flux_lycoris_orchestration_smoke
# RUN (optional; writes /tmp/chroma_{lokr,loha}.safetensors):
#   ./target/chroma_flux_lycoris_orchestration_smoke
# EXPECT: "ALL GATES PASS — chroma_flux_lycoris_orchestration_smoke".

from std.collections import List
from std.gpu.host import DeviceContext
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.models.flux.lora_block import DBL_STREAM_SLOTS, SGL_SLOTS
from serenitymojo.models.flux.flux_lycoris_stack import (
    build_flux_lokr_set, flux_lokr_carrier_set,
    flux_lokr_chain_all, flux_lokr_adamw_step, flux_lokr_zero_leg_l1, save_flux_lokr,
    build_flux_loha_set, flux_loha_carrier_set,
    flux_loha_chain_all, flux_loha_adamw_step, flux_loha_zero_leg_l1, save_flux_loha,
    flux_lycoris_dbl_slot_dims, flux_lycoris_sgl_slot_dims,
)
from serenitymojo.training.lokr_stack import lokr_carrier_r_eff

comptime D = 3072
comptime F = 12288                 # FMLP
comptime NUM_DOUBLE = 19
comptime NUM_SINGLE = 38
comptime RANK = 16
comptime ALPHA = Float32(16.0)
comptime FACTOR = -1
comptime DECOMPOSE_BOTH = True
comptime FULL_MATRIX = False
comptime TARGETS = 2               # 2 = all (attn + mlp) → every slot active


def _fill(n: Int, seed: UInt64, scale: Float32) -> List[Float32]:
    var out = List[Float32]()
    var state = seed
    for _ in range(n):
        state = state * 6364136223846793005 + 1442695040888963407
        var u = Float32(Int(state >> 40)) * Float32(1.0 / 16777216.0)
        out.append((u - Float32(0.5)) * scale)
    return out^


# Expected (in,out) for carrier index i in the flux flat layout:
#   [ 0 .. num_double*2*DBL_STREAM_SLOTS )  → double (img|txt streams)
#   [ that .. end )                         → single
def _expected_dims(i: Int) raises -> Tuple[Int, Int]:
    var dbl_span = NUM_DOUBLE * 2 * DBL_STREAM_SLOTS
    if i < dbl_span:
        return flux_lycoris_dbl_slot_dims(i % DBL_STREAM_SLOTS, D, F)
    var j = i - dbl_span
    return flux_lycoris_sgl_slot_dims(j % SGL_SLOTS, D, F)


def main() raises:
    var ctx = DeviceContext()
    var expected_total = NUM_DOUBLE * 2 * DBL_STREAM_SLOTS + NUM_SINGLE * SGL_SLOTS
    print("=== Chroma FLUX-LyCORIS carrier orchestration smoke ===")
    print("  NUM_DOUBLE=", NUM_DOUBLE, " NUM_SINGLE=", NUM_SINGLE,
          " dbl_slots=", DBL_STREAM_SLOTS, " sgl_slots=", SGL_SLOTS,
          " expected total carriers =", expected_total)

    # ─────────────────────── LoKr phase ───────────────────────
    print("-- LoKr --")
    var set = build_flux_lokr_set(
        NUM_DOUBLE, NUM_SINGLE, D, F, RANK, ALPHA,
        FACTOR, FACTOR, FACTOR, DECOMPOSE_BOTH, FULL_MATRIX, TARGETS, UInt64(910701),
    )
    var cs = flux_lokr_carrier_set(set, D, F)
    if len(cs.ad) != expected_total:
        raise Error("Chroma LoKr carrier count " + String(len(cs.ad))
            + " != expected " + String(expected_total))
    var active = 0
    var l3 = 0
    for i in range(len(cs.ad)):
        if not set.active[i]:
            continue
        active += 1
        var e = _expected_dims(i)
        if cs.ad[i].in_f != e[0] or cs.ad[i].out_f != e[1]:
            raise Error("Chroma LoKr GEOMETRY MISMATCH idx " + String(i) + ": carrier ("
                + String(cs.ad[i].in_f) + "," + String(cs.ad[i].out_f) + ") != ("
                + String(e[0]) + "," + String(e[1]) + ")")
        ref lo = set.ad[i]
        if cs.ad[i].rank != lokr_carrier_r_eff(lo):
            raise Error("Chroma LoKr r_eff carrier/master mismatch idx " + String(i))
        if lo.w1_factored and lo.w2_factored:
            if lokr_carrier_r_eff(lo) != RANK * RANK:
                raise Error("Chroma LoKr L3 r_eff " + String(lokr_carrier_r_eff(lo))
                    + " != rank² " + String(RANK * RANK) + " idx " + String(i))
            l3 += 1
    var zb = flux_lokr_zero_leg_l1(set)
    if zb != Float64(0.0):
        raise Error("Chroma LoKr zero-leg L1 at init = " + String(zb) + " (expect 0)")
    var d_a = List[List[Float32]]()
    var d_b = List[List[Float32]]()
    for i in range(len(cs.ad)):
        var r = cs.ad[i].rank
        d_a.append(_fill(r * cs.ad[i].in_f, UInt64(7) * UInt64(i + 1) + 1, 0.5))
        d_b.append(_fill(cs.ad[i].out_f * r, UInt64(13) * UInt64(i + 1) + 3, 0.5))
    flux_lokr_adamw_step(set, flux_lokr_chain_all(set, d_a, d_b), 1,
                         Float32(1.0e-3), Float32(0.9), Float32(0.999), Float32(1.0e-8), Float32(0.01))
    var za = flux_lokr_zero_leg_l1(set)
    if za <= zb:
        raise Error("Chroma LoKr AdamW zero-leg no-op (before=" + String(zb)
            + " after=" + String(za) + ")")
    print("  carriers:", len(cs.ad), " active:", active, " (L3:", l3, ")")
    print("  zero-leg L1: init =", zb, " after-step =", za)
    var nl = save_flux_lokr(set, "/tmp/chroma_lokr.safetensors", ctx)
    print("  save_flux_lokr modules:", nl)
    if nl <= 0:
        raise Error("Chroma LoKr saved 0 modules")
    _ = SafeTensors.open("/tmp/chroma_lokr.safetensors")
    print("  reopened LoKr file ✓")

    # ─────────────────────── LoHa phase ───────────────────────
    print("-- LoHa --")
    var lset = build_flux_loha_set(NUM_DOUBLE, NUM_SINGLE, D, F, RANK, ALPHA,
                                   TARGETS, UInt64(920701))
    var lcs = flux_loha_carrier_set(lset, D, F)
    if len(lcs.ad) != expected_total:
        raise Error("Chroma LoHa carrier count mismatch")
    var lactive = 0
    for i in range(len(lcs.ad)):
        if not lset.active[i]:
            continue
        lactive += 1
        var e = _expected_dims(i)
        if lcs.ad[i].in_f != e[0] or lcs.ad[i].out_f != e[1]:
            raise Error("Chroma LoHa GEOMETRY MISMATCH idx " + String(i) + ": carrier ("
                + String(lcs.ad[i].in_f) + "," + String(lcs.ad[i].out_f) + ") != ("
                + String(e[0]) + "," + String(e[1]) + ")")
        if lcs.ad[i].rank != RANK * RANK:
            raise Error("Chroma LoHa r_eff " + String(lcs.ad[i].rank)
                + " != rank² " + String(RANK * RANK) + " idx " + String(i))
    var lzb = flux_loha_zero_leg_l1(lset)
    if lzb != Float64(0.0):
        raise Error("Chroma LoHa zero-leg L1 at init = " + String(lzb) + " (expect 0)")
    var la = List[List[Float32]]()
    var lb = List[List[Float32]]()
    for i in range(len(lcs.ad)):
        var r = lcs.ad[i].rank
        la.append(_fill(r * lcs.ad[i].in_f, UInt64(7) * UInt64(i + 1) + 1, 0.5))
        lb.append(_fill(lcs.ad[i].out_f * r, UInt64(13) * UInt64(i + 1) + 3, 0.5))
    flux_loha_adamw_step(lset, flux_loha_chain_all(lset, la, lb), 1,
                         Float32(1.0e-3), Float32(0.9), Float32(0.999), Float32(1.0e-8), Float32(0.01))
    var lza = flux_loha_zero_leg_l1(lset)
    if lza <= lzb:
        raise Error("Chroma LoHa AdamW zero-leg no-op")
    print("  carriers:", len(lcs.ad), " active:", lactive, " (all r_eff=rank²)")
    print("  zero-leg L1: init =", lzb, " after-step =", lza)
    var nh = save_flux_loha(lset, "/tmp/chroma_loha.safetensors", ctx)
    print("  save_flux_loha modules:", nh)
    if nh <= 0:
        raise Error("Chroma LoHa saved 0 modules")
    _ = SafeTensors.open("/tmp/chroma_loha.safetensors")
    print("  reopened LoHa file ✓")

    print("ALL GATES PASS — chroma_flux_lycoris_orchestration_smoke")
