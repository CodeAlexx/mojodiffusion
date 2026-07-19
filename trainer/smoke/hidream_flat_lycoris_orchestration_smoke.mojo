# smoke/hidream_flat_lycoris_orchestration_smoke.mojo
#
# PER-MODEL LoKr + LoHa CARRIER ORCHESTRATION smoke for the HiDream-O1 trainer's
# REAL slot geometry. HiDream dispatches LyCORIS through the SHARED flat carrier
# stack (serenitymojo/training/flat_lycoris_stack.mojo) in TWO sets — 252 block
# adapters (36 language-model layers × 7 slots q/k/v/o/gate/up/down) + 5 resident
# HEAD adapters (ai-toolkit wraps every Linear ⇒ 252+5=257). No per-model
# orchestration gate existed; this closes that audit cell.
#
# Proves at HiDream's REAL dims (NOT toy):
#   (0) SLOT COUNT: block carrier set = LAYERS*7 = 252, head set = N_HEADS = 5.
#   (1) SLOT DIMS: every carrier's (in_f,out_f) equals the HiDream slot/head
#       rule (GQA k/v = HKV*Dh, o = H*Dh→D, mlp gate/up D→F, down F→D; heads
#       x_embedder/t_embedder/final_layer). Mismatch = REAL geometry bug.
#   (2) RECONSTRUCTION r_eff: LoKr both-factored slots → r_eff=rank² (kron);
#       every LoHa carrier → r_eff=rank² (hadamard). Numeric kron/hada VALUES
#       gated by the core carrier parity; here we gate r_eff at real dims.
#   (3) ZERO-LEG INIT: flat_*_zero_leg_l1(block)+(head) == 0 → initial ΔW ZERO.
#   (4) UPDATE FLOW: one master AdamW step on synthetic carrier grads moves the
#       zero-leg off zero (carrier→master chain updates).
#   (5) SAVE: save_flat_lokr_pair writes 257 modules (block+head) and reopens.
#
# DERIVED GEOMETRY (cited, serenity_trainer/trainer/train_hidream_o1_real.mojo):
#   D=4096,H=32,HKV=8,Dh=128,F=12288,LAYERS=36,PATCH_VEC=3072   :228-235
#   block _slot_dims(0..6)                                       :266-277
#   N_HEADS=5, XEMB_MID=1024, _head_dims(0..4)                   :296-311
#   two-set build + save_flat_lokr_pair                          :1324-1355,2266
#
# BUILD (deliverable — NO GPU run needed to compile):
#   cd trainer && rm -f /home/alex/mojodiffusion/serenitymojo.mojopkg && \
#   MEM_MAX=24G MEM_HIGH=20G pixi run bash /home/alex/mojodiffusion/scripts/mem_safe.sh \
#     mojo build -O2 -I . -I src -I /home/alex/mojodiffusion \
#       -Xlinker -lm -Xlinker -lcuda \
#       -Xlinker -L/home/alex/mojodiffusion/.pixi/envs/default/lib -Xlinker -lsqlite3 \
#       smoke/hidream_flat_lycoris_orchestration_smoke.mojo \
#       -o target/hidream_flat_lycoris_orchestration_smoke
# RUN (optional; writes /tmp/hidream_{lokr,loha}_pair.safetensors):
#   ./target/hidream_flat_lycoris_orchestration_smoke
# EXPECT: "ALL GATES PASS — hidream_flat_lycoris_orchestration_smoke".

from std.collections import List
from std.gpu.host import DeviceContext
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.training.flat_lycoris_stack import (
    FlatLoKrSet, build_flat_lokr_set, flat_lokr_carrier_list,
    flat_lokr_chain_all, flat_lokr_adamw_step, flat_lokr_zero_leg_l1,
    save_flat_lokr_pair,
    FlatLoHaSet, build_flat_loha_set, flat_loha_carrier_list,
    flat_loha_chain_all, flat_loha_adamw_step, flat_loha_zero_leg_l1,
    save_flat_loha_pair,
)
from serenitymojo.training.lokr_stack import lokr_carrier_r_eff

comptime D = 4096
comptime H = 32
comptime HKV = 8
comptime Dh = 128
comptime F = 12288
comptime LAYERS = 36
comptime PATCH_VEC = 3072
comptime N_HEADS = 5
comptime XEMB_MID = 1024
comptime BLOCK_SLOTS = 7
comptime RANK = 4
comptime ALPHA = Float32(8.0)
comptime FACTOR = -1
comptime DECOMPOSE_BOTH = True
comptime FULL_MATRIX = False


def _fill(n: Int, seed: UInt64, scale: Float32) -> List[Float32]:
    var out = List[Float32]()
    var state = seed
    for _ in range(n):
        state = state * 6364136223846793005 + 1442695040888963407
        var u = Float32(Int(state >> 40)) * Float32(1.0 / 16777216.0)
        out.append((u - Float32(0.5)) * scale)
    return out^


# Replicated from train_hidream_o1_real.mojo:266-277 (_slot_dims).
def _slot_dims(slot: Int) -> Tuple[Int, Int]:
    if slot == 0:
        return (D, H * Dh)
    if slot == 1 or slot == 2:
        return (D, HKV * Dh)
    if slot == 3:
        return (H * Dh, D)
    if slot == 4 or slot == 5:
        return (D, F)
    return (F, D)


# Replicated from train_hidream_o1_real.mojo:300-311 (_head_dims).
def _head_dims(h: Int) -> Tuple[Int, Int]:
    if h == 0:
        return (PATCH_VEC, XEMB_MID)
    if h == 1:
        return (XEMB_MID, D)
    if h == 2:
        return (256, D)
    if h == 3:
        return (D, D)
    return (D, PATCH_VEC)


def _block_in_dims() -> List[Int]:
    var out = List[Int]()
    for _li in range(LAYERS):
        for sl in range(BLOCK_SLOTS):
            out.append(_slot_dims(sl)[0])
    return out^


def _block_out_dims() -> List[Int]:
    var out = List[Int]()
    for _li in range(LAYERS):
        for sl in range(BLOCK_SLOTS):
            out.append(_slot_dims(sl)[1])
    return out^


def _block_names() -> List[String]:
    var out = List[String]()
    for li in range(LAYERS):
        for sl in range(BLOCK_SLOTS):
            out.append("diffusion_model.language_model.layers." + String(li)
                + ".slot" + String(sl))
    return out^


def _head_in_dims() -> List[Int]:
    var out = List[Int]()
    for h in range(N_HEADS):
        out.append(_head_dims(h)[0])
    return out^


def _head_out_dims() -> List[Int]:
    var out = List[Int]()
    for h in range(N_HEADS):
        out.append(_head_dims(h)[1])
    return out^


def _head_names() -> List[String]:
    var out = List[String]()
    for h in range(N_HEADS):
        out.append("diffusion_model.head." + String(h))
    return out^


def main() raises:
    var ctx = DeviceContext()
    print("=== HiDream-O1 flat-LyCORIS carrier orchestration smoke ===")
    print("  block carriers = LAYERS*7 =", LAYERS * BLOCK_SLOTS, " head =", N_HEADS,
          " total =", LAYERS * BLOCK_SLOTS + N_HEADS)

    var bin = _block_in_dims()
    var bout = _block_out_dims()
    var bnames = _block_names()
    var hin = _head_in_dims()
    var hout = _head_out_dims()
    var hnames = _head_names()

    # ─────────────────────── LoKr phase ───────────────────────
    print("-- LoKr --")
    var block = build_flat_lokr_set(bin, bout, bnames, RANK, ALPHA, FACTOR,
                                    DECOMPOSE_BOTH, FULL_MATRIX, UInt64(910701))
    var head = build_flat_lokr_set(hin, hout, hnames, RANK, ALPHA, FACTOR,
                                   DECOMPOSE_BOTH, FULL_MATRIX, UInt64(910777))
    var bc = flat_lokr_carrier_list(block)
    var hc = flat_lokr_carrier_list(head)
    if len(bc) != LAYERS * BLOCK_SLOTS:
        raise Error("HiDream LoKr block carrier count " + String(len(bc)) + " != 252")
    if len(hc) != N_HEADS:
        raise Error("HiDream LoKr head carrier count " + String(len(hc)) + " != 5")
    var l3 = 0
    for k in range(len(bc)):
        var sl = k % BLOCK_SLOTS
        var e = _slot_dims(sl)
        if bc[k].in_f != e[0] or bc[k].out_f != e[1]:
            raise Error("HiDream LoKr BLOCK GEOMETRY MISMATCH layer-slot " + String(k)
                + " slot " + String(sl) + ": carrier (" + String(bc[k].in_f) + ","
                + String(bc[k].out_f) + ") != (" + String(e[0]) + "," + String(e[1]) + ")")
        ref lo = block.ad[k]
        if bc[k].rank != lokr_carrier_r_eff(lo):
            raise Error("HiDream LoKr block r_eff carrier/master mismatch idx " + String(k))
        if lo.w1_factored and lo.w2_factored:
            if lokr_carrier_r_eff(lo) != RANK * RANK:
                raise Error("HiDream LoKr block L3 r_eff != rank² at slot " + String(sl))
            l3 += 1
    for h in range(len(hc)):
        var e = _head_dims(h)
        if hc[h].in_f != e[0] or hc[h].out_f != e[1]:
            raise Error("HiDream LoKr HEAD GEOMETRY MISMATCH head " + String(h)
                + ": carrier (" + String(hc[h].in_f) + "," + String(hc[h].out_f)
                + ") != (" + String(e[0]) + "," + String(e[1]) + ")")
        ref lo = head.ad[h]
        if hc[h].rank != lokr_carrier_r_eff(lo):
            raise Error("HiDream LoKr head r_eff carrier/master mismatch head " + String(h))
    var zb = flat_lokr_zero_leg_l1(block) + flat_lokr_zero_leg_l1(head)
    if zb != Float64(0.0):
        raise Error("HiDream LoKr zero-leg L1 at init = " + String(zb) + " (expect 0)")
    # block grads
    var bd_a = List[List[Float32]]()
    var bd_b = List[List[Float32]]()
    for k in range(len(bc)):
        var r = bc[k].rank
        bd_a.append(_fill(r * bc[k].in_f, UInt64(1000) + UInt64(7) * UInt64(k + 1) + 1, 0.5))
        bd_b.append(_fill(bc[k].out_f * r, UInt64(1000) + UInt64(13) * UInt64(k + 1) + 3, 0.5))
    flat_lokr_adamw_step(block, flat_lokr_chain_all(block, bd_a, bd_b), 1,
                         Float32(1.0e-3), Float32(0.9), Float32(0.999), Float32(1.0e-8), Float32(0.01))
    # head grads
    var hd_a = List[List[Float32]]()
    var hd_b = List[List[Float32]]()
    for h in range(len(hc)):
        var r = hc[h].rank
        hd_a.append(_fill(r * hc[h].in_f, UInt64(2000) + UInt64(7) * UInt64(h + 1) + 1, 0.5))
        hd_b.append(_fill(hc[h].out_f * r, UInt64(2000) + UInt64(13) * UInt64(h + 1) + 3, 0.5))
    flat_lokr_adamw_step(head, flat_lokr_chain_all(head, hd_a, hd_b), 1,
                         Float32(1.0e-3), Float32(0.9), Float32(0.999), Float32(1.0e-8), Float32(0.01))
    var za = flat_lokr_zero_leg_l1(block) + flat_lokr_zero_leg_l1(head)
    if za <= zb:
        raise Error("HiDream LoKr AdamW zero-leg no-op (before=" + String(zb)
            + " after=" + String(za) + ")")
    print("  block+head carriers:", len(bc) + len(hc), " (block L3:", l3, ")")
    print("  zero-leg L1: init =", zb, " after-step =", za)
    var nl = save_flat_lokr_pair(block, head, "/tmp/hidream_lokr_pair.safetensors", ctx)
    print("  save_flat_lokr_pair modules:", nl)
    if nl <= 0:
        raise Error("HiDream LoKr saved 0 modules")
    _ = SafeTensors.open("/tmp/hidream_lokr_pair.safetensors")
    print("  reopened LoKr pair ✓")

    # ─────────────────────── LoHa phase ───────────────────────
    print("-- LoHa --")
    var lblock = build_flat_loha_set(bin, bout, bnames, RANK, ALPHA, UInt64(920701))
    var lhead = build_flat_loha_set(hin, hout, hnames, RANK, ALPHA, UInt64(920777))
    var lbc = flat_loha_carrier_list(lblock)
    var lhc = flat_loha_carrier_list(lhead)
    if len(lbc) != LAYERS * BLOCK_SLOTS or len(lhc) != N_HEADS:
        raise Error("HiDream LoHa carrier count mismatch")
    for k in range(len(lbc)):
        var sl = k % BLOCK_SLOTS
        var e = _slot_dims(sl)
        if lbc[k].in_f != e[0] or lbc[k].out_f != e[1]:
            raise Error("HiDream LoHa BLOCK GEOMETRY MISMATCH idx " + String(k))
        if lbc[k].rank != RANK * RANK:
            raise Error("HiDream LoHa block r_eff " + String(lbc[k].rank) + " != rank²")
    for h in range(len(lhc)):
        var e = _head_dims(h)
        if lhc[h].in_f != e[0] or lhc[h].out_f != e[1]:
            raise Error("HiDream LoHa HEAD GEOMETRY MISMATCH head " + String(h))
        if lhc[h].rank != RANK * RANK:
            raise Error("HiDream LoHa head r_eff != rank²")
    var lzb = flat_loha_zero_leg_l1(lblock) + flat_loha_zero_leg_l1(lhead)
    if lzb != Float64(0.0):
        raise Error("HiDream LoHa zero-leg L1 at init = " + String(lzb) + " (expect 0)")
    var lbd_a = List[List[Float32]]()
    var lbd_b = List[List[Float32]]()
    for k in range(len(lbc)):
        var r = lbc[k].rank
        lbd_a.append(_fill(r * lbc[k].in_f, UInt64(3000) + UInt64(7) * UInt64(k + 1) + 1, 0.5))
        lbd_b.append(_fill(lbc[k].out_f * r, UInt64(3000) + UInt64(13) * UInt64(k + 1) + 3, 0.5))
    flat_loha_adamw_step(lblock, flat_loha_chain_all(lblock, lbd_a, lbd_b), 1,
                         Float32(1.0e-3), Float32(0.9), Float32(0.999), Float32(1.0e-8), Float32(0.01))
    var lhd_a = List[List[Float32]]()
    var lhd_b = List[List[Float32]]()
    for h in range(len(lhc)):
        var r = lhc[h].rank
        lhd_a.append(_fill(r * lhc[h].in_f, UInt64(4000) + UInt64(7) * UInt64(h + 1) + 1, 0.5))
        lhd_b.append(_fill(lhc[h].out_f * r, UInt64(4000) + UInt64(13) * UInt64(h + 1) + 3, 0.5))
    flat_loha_adamw_step(lhead, flat_loha_chain_all(lhead, lhd_a, lhd_b), 1,
                         Float32(1.0e-3), Float32(0.9), Float32(0.999), Float32(1.0e-8), Float32(0.01))
    var lza = flat_loha_zero_leg_l1(lblock) + flat_loha_zero_leg_l1(lhead)
    if lza <= lzb:
        raise Error("HiDream LoHa AdamW zero-leg no-op")
    print("  block+head carriers:", len(lbc) + len(lhc), " (all r_eff=rank²)")
    print("  zero-leg L1: init =", lzb, " after-step =", lza)
    var nh = save_flat_loha_pair(lblock, lhead, "/tmp/hidream_loha_pair.safetensors", ctx)
    print("  save_flat_loha_pair modules:", nh)
    if nh <= 0:
        raise Error("HiDream LoHa saved 0 modules")
    _ = SafeTensors.open("/tmp/hidream_loha_pair.safetensors")
    print("  reopened LoHa pair ✓")

    print("ALL GATES PASS — hidream_flat_lycoris_orchestration_smoke")
