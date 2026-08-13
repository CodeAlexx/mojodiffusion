# serenitymojo/models/krea2/parity/krea2_omini_c3_lora_gate.mojo
#
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  C3 COND-ROW LoRA PARITY GATE — krea2 OminiControl EDIT vertical         ║
# ║  The adapter acts on the CONDITION ROWS ONLY; text + image rows run the  ║
# ║  frozen base. Gates the MOJO forward against the CUDA-generated fixtures ║
# ║  written by scripts/krea2_omini_torch_oracle.py. CPU torch is NEVER an   ║
# ║  oracle here: every reference number comes out of that fixture, which    ║
# ║  the oracle produced on the RTX 5080 with the real Krea-2-Raw bf16       ║
# ║  weights and the real krea2 activation cache.                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# WHAT C3 IMPLEMENTS (models/krea2/krea2_block.mojo)
#   `krea2_single_stream_block_lora` gained ONE new optional parameter,
#   `cond_len`, next to C2's `vec_cond` / `cond_off`. With all three supplied the
#   block routes the LoRA delta at all 8 Linears to the condition rows:
#       y = x @ Wᵀ                                over the FULL sequence
#       y[:, c_off : c_off+c_len] += scale * ((x_c @ Aᵀ) @ Bᵀ)
#   The FROZEN BASE is never restricted (it always runs all L rows); only the
#   low-rank delta shrinks from L rows to c_len rows. Helpers: `_lora_delta_rows`
#   (row-sliced delta) and `_add_delta_rows` (scatter-add), both carrying the
#   c_off < 0 sentinel that reduces them to the pre-C3 kernels exactly.
#   This mirrors OminiControl trainer.py:57 `adapter_names=[None,None,"default"]`
#   and the oracle's `lin_cond_lora` (krea2_omini_torch_oracle.py).
#
# WHAT IS GATED (in the order the C3 brief requires; verdict is the AND)
#   Every kref_* below means the *_f32mod (SCHEDULE B) variant gated on
#   env_cos_*_f32mod; the schedule-A key is printed as info. See the
#   ROUNDING SCHEDULE box further down.
#   1. kref_xm          — per-segment modulation is unchanged by LoRA routing
#                         (it is computed BEFORE any projection). Re-gated here
#                         so a C3 regression in the shared path is caught.
#   2. kref_attn_raw    — the LoRA-**ON** attention seam. C2 could not gate this
#                         (a LoRA-off forward cannot match it) and had to rebuild
#                         the reference with a gate-local helper; C3 gates it with
#                         the REAL block forward, which is the point of the chunk.
#   3. kref_gated       — sdpa * sigmoid(gate): carries the wq/wk/wv/gate cond-row
#                         deltas. This is the first key that fails if the routing
#                         window is wrong.
#   4. kref_attn        — wo(gated), i.e. + the wo cond-row delta.
#   5. kref_out         — the whole block output (+ mlp_gate/up/down deltas and
#                         both per-segment residual gates).
#   6. kref_out_img     — the IMG row slice: the ONLY rows the loss reads
#                         (OminiControl discards the cond outputs).
#   7. ROUTING ISOLATION — a direct, bit-level proof that the delta lands only on
#                         the cond rows (section 7; see the note there for why
#                         only the FIRST-seam projection can carry this proof),
#                         plus a CONTRAST run proving the adapters are not inert.
#   8. CONDLEN=0 BIT-EQUALITY — with the SAME 8 real adapters attached, the
#                         pre-C2 call form and the post-C3 call form of a no-cond
#                         build must be BIT-IDENTICAL (bitdiff 0).
#
# THRESHOLDS — TAKEN FROM THE FIXTURE, NEVER INVENTED
#   Every key is gated on ITS OWN `env_cos_<key>_f32mod`, the MEASURED
#   bf16-vs-f32 cosine of the oracle's own two runs on the same CUDA device with
#   the same real weights — the ceiling for any bf16 implementation of that seam
#   UNDER THE ROUNDING SCHEDULE THE MOJO KERNELS ACTUALLY COMPUTE. `_env()`
#   hard-fails if an envelope is missing, so a gate can never silently fall back
#   to a borrowed threshold. Nothing is asserted tighter than the shipped
#   envelope and nothing is ever loosened.
#   NOTE the envelopes are NOT uniform across seams: at block 0
#   env_cos_attn_raw = 0.984 while env_cos_out = 0.99995. The attention seam is
#   genuinely the bf16-noisiest point of this block, which is exactly why
#   borrowing one key's envelope for another is unsound in both directions.
#
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  WHICH ROUNDING SCHEDULE THIS GATE TARGETS — the fix that made these      ║
# ║  comparisons apples-to-apples. READ BEFORE CHANGING A THRESHOLD.          ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#   The oracle ships each gated seam under TWO bf16 rounding schedules. Same
#   seed, same real krea2 weights, same real cached inputs, same layout, mask,
#   RoPE, LoRA routing, F32 matmul accumulation, same CUDA device. The ONLY
#   difference is WHERE bf16 rounding happens inside modulate/residual_gate:
#     SCHEDULE A  kref_<key>            (1+scale)*h+shift evaluated in bf16 ->
#                 env_cos_<key>         THREE roundings per modulate, TWO per
#                                       residual gate. Plain torch semantics;
#                                       the schedule the mmdit fidelity check
#                                       validates the hand forward against.
#     SCHEDULE B  kref_<key>_f32mod     the same algebra in F32, rounded ONCE at
#                 env_cos_<key>_f32mod  store.
#   serenitymojo/ops/elementwise.mojo `_modulate_kernel_bf16` and
#   `_resgate_kernel_bf16` upcast x/scale/shift to F32, compute in F32, and
#   .cast[bfloat16]() exactly ONCE, at the store. THE MOJO FORWARD IS SCHEDULE B.
#   PRIMARY gate = Mojo vs kref_<key>_f32mod at env_cos_<key>_f32mod.
#
#   WHY (a criterion bug that was actually shipped, not a hypothetical):
#   env_cos_<key> measures bf16-storage-vs-f32-storage UNDER SCHEDULE A's OWN
#   ROUNDING, so it does NOT bound a correct implementation that uses schedule
#   B. The fixture proves it at block 27:
#       env_cos_out_img          1-cos = 1.043e-05   (the old threshold)
#       env_cos_out_img_modround 1-cos = 1.097e-05   (A-vs-B distance ALONE)
#   Merely choosing the F32-math modulate puts a provably-correct implementation
#   outside that key's envelope. kref_out_img failed at block 27 by ~1e-7 for
#   exactly that reason (and kref_out_img alone in this gate; the other C3 keys
#   have looser envelopes and were passing anyway).
#
#   THIS IS NOT A TOLERANCE WIDENING. At block 27 the schedule-B envelopes are
#   TIGHTER than the schedule-A ones they replace, e.g. out_img
#   1-env: 1.043e-05 (A) -> 7.75e-06 (B); out 1.198e-05 -> 9.60e-06; xm
#   6.74e-06 -> 4.29e-06. Schedule B rounds less, so it sits closer to F32 and
#   its ceiling is lower. The gate got STRICTER and simultaneously correct.
#
#   HONEST CAVEAT, measured on this GPU: "schedule B is tighter" is TRUE FOR ALL
#   NINE KEYS AT BLOCK 27 but NOT universally. At BLOCK 0 four of the nine keys
#   have a slightly LOOSER schedule-B envelope (1-env, A -> B): attn 9.791e-04 ->
#   9.826e-04, attn_raw 1.5865e-02 -> 1.5892e-02, attn_raw_nolora 1.1297e-02 ->
#   1.1321e-02, gated 2.2441e-03 -> 2.2460e-03; the other five are tighter (xm
#   1.1146e-05 -> 8.941e-06, out 4.613e-05 -> 4.399e-05, out_img 1.0765e-04 ->
#   1.0562e-04, out_nolora 3.266e-05 -> 3.052e-05, out_nolora_img 7.170e-05 ->
#   6.962e-05). Those four are the attention-dominated seams, where block 0's
#   bf16 noise is ~1e-2 and the modulate rounding point is irrelevant; the
#   loosening is 0.2-0.3% RELATIVE on an envelope the Mojo forward clears by
#   three orders of magnitude (cos 0.99996 vs envelope 0.9887). It is a measured
#   consequence of matching the schedule, not a threshold anyone chose.
#
#   The schedule-A comparison is still run and printed on an `info(schedule A)`
#   line, with its own envelope and its would-be verdict, so nothing is hidden.
#   It never sets the verdict.
#
# MASK / PAD-ROW SEMANTICS
#   The oracle masks pad rows both as key columns and as query rows; the Mojo
#   production path is the cuDNN flash tail-padmask, which masks only the key
#   columns and leaves the pad QUERY rows as garbage the trainer drops. The two
#   agree exactly on rows [0, real_len), so the GATE is the real prefix. Full
#   tensors (pad rows included) are printed alongside and NOT gated.
#
# WHAT IS NOT ASSERTED (deliberately)
#   The IMG rows are NOT bit-equal to a LoRA-off forward. Editing the condition
#   rows' K/V propagates to every row through bidirectional attention (intake
#   §1.4) — the oracle measures max|delta| = 8.0 on the img rows at block 0 and
#   FAILS if it is zero. That coupling is the method working, not a leak.
#
# BACKWARD IS STILL UNIFORM-MOD, FULL-SEQUENCE (C4's job). Nothing here enables
# cond args in a training path.
#
# RUN (never chained after a mojo build with &&):
#   cd /home/alex/mojodiffusion
#   rm -f serenitymojo.mojopkg
#   pixi run mojo run -I . \
#       -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa \
#       serenitymojo/models/krea2/parity/krea2_omini_c3_lora_gate.mojo

from max.gpu.host import DeviceContext
from std.collections import List, Optional
from std.math import sqrt
from std.memory import ArcPointer
from serenitymojo.tensor import Tensor
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.tensor_algebra import slice, concat
from serenitymojo.models.klein.lora_block import LoraAdapterDevice
from serenitymojo.models.krea2.krea2_block import (
    Krea2BlockWeights, Krea2BlockLora, krea2_single_stream_block_lora,
)
from serenitymojo.models.dit.krea2_dit import build_krea2_rope, _tile_rope_table
from serenitymojo.training.krea2_omini_layout import (
    Krea2OminiLayout, krea2_omini_mod_split,
    krea2_omini_pos_combined,
)

comptime TArc = ArcPointer[Tensor]

comptime CKPT = "/home/alex/.serenity/models/checkpoints/krea2-raw.safetensors"
comptime FIX_DIR = "/home/alex/mojodiffusion/serenitymojo/models/krea2/parity/"

# ── shapes: MUST equal the fixture's meta_* scalars (verified at runtime) ─────
comptime HEADS = 48
comptime KVHEADS = 12
comptime HEADDIM = 128
comptime FEATURES = HEADS * HEADDIM        # 6144
comptime HALF = HEADDIM // 2               # 64
comptime MLPDIM = 16384
comptime LTMAX = 384
comptime GRID = 32                         # 64x64 latent, patch 2 -> 32x32
comptime S_IMG = GRID * GRID               # 1024
comptime S_COND = GRID * GRID              # 1024
comptime LT = 190                          # the cache sample's real caption len
comptime LFULL = LTMAX + S_IMG + S_COND    # 2432
comptime COND_OFF = LT + S_IMG             # 1214
comptime PAD_OFF = COND_OFF + S_COND       # 2238 == real_len
comptime L_NC = LTMAX + S_IMG              # 1408  (CONDLEN=0 regression length)
comptime RANK = 16                         # krea2.json preset (oracle meta_rank)
comptime EPS = Float32(1e-5)
comptime THETA = Float32(1000.0)


# ── tiny shape helpers ───────────────────────────────────────────────────────
def _s1(a: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a)
    return s^


# ── fixture / checkpoint readers ─────────────────────────────────────────────
def _fv(st: ShardedSafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    """Load a tensor PRESERVING its on-disk dtype (bf16 stays bf16)."""
    return Tensor.from_view(st.tensor_view(name), ctx)


def _fv_bf16(
    st: ShardedSafeTensors, name: String, ctx: DeviceContext
) raises -> TArc:
    return TArc(Tensor.from_view_as_bf16(st.tensor_view(name), ctx))


def _fv_f32(
    st: ShardedSafeTensors, name: String, ctx: DeviceContext
) raises -> TArc:
    return TArc(Tensor.from_view_as_f32(st.tensor_view(name), ctx))


def _host(
    st: ShardedSafeTensors, name: String, ctx: DeviceContext
) raises -> List[Float32]:
    """Fixture tensor -> host F32 (bf16 fixtures upcast losslessly)."""
    var t = Tensor.from_view(st.tensor_view(name), ctx)
    return t.to_host(ctx)


def _meta_i(st: ShardedSafeTensors, name: String) raises -> Int:
    var b = st.tensor_bytes(name)
    if len(b) != 4:
        raise Error(String("meta ") + name + ": expected 4 bytes")
    var v = (
        Int(b[0]) | (Int(b[1]) << 8) | (Int(b[2]) << 16) | (Int(b[3]) << 24)
    )
    if v >= 2147483648:
        v = v - 4294967296
    return v


def _expect_i(st: ShardedSafeTensors, name: String, want: Int) raises:
    var got = _meta_i(st, name)
    if got != want:
        raise Error(
            String("FIXTURE/GATE SHAPE MISMATCH: ") + name + " = " + String(got)
            + " but this gate is built for " + String(want)
        )


# ── per-key envelope reader — THE ONLY source of a threshold in this file ────
def _env(fx: ShardedSafeTensors, key: String, ctx: DeviceContext) raises -> Float64:
    var v = _host(fx, "env_cos_" + key, ctx)
    if len(v) != 1:
        raise Error("fixture ships no env_cos_" + key)
    return Float64(v[0])


def _modr(fx: ShardedSafeTensors, key: String, ctx: DeviceContext) raises -> Float64:
    """env_cos_<key>_modround — INFORMATIONAL ONLY (never a threshold). See the
    ROUNDING-POINT FREEDOM section of scripts/krea2_omini_torch_oracle.py."""
    var v = _host(fx, "env_cos_" + key + "_modround", ctx)
    if len(v) != 1:
        raise Error("fixture ships no env_cos_" + key + "_modround")
    return Float64(v[0])


# ── comparison (F64 host math, plus relL2 and an exact-mismatch count) ───────
@fieldwise_init
struct Cmp(Copyable, Movable):
    var cos: Float64
    var rel: Float64      # ||a-b|| / ||b||
    var max_abs: Float64
    var n_diff: Int       # exact (bit-level, after F32 upcast) mismatches
    var n: Int


def _cmp2(
    a: List[Float32], b: List[Float32], off_a: Int, off_b: Int, n: Int
) raises -> Cmp:
    if off_a + n > len(a) or off_b + n > len(b):
        raise Error("cmp: range out of bounds")
    var dot: Float64 = 0.0
    var na: Float64 = 0.0
    var nb: Float64 = 0.0
    var nd: Float64 = 0.0
    var mx: Float64 = 0.0
    var ndiff = 0
    for i in range(n):
        var av = a[off_a + i]
        var bv = b[off_b + i]
        var x = Float64(av)
        var y = Float64(bv)
        dot += x * y
        na += x * x
        nb += y * y
        var d = x - y
        nd += d * d
        if av != bv:
            ndiff += 1
        if d < 0.0:
            d = -d
        if d > mx:
            mx = d
    var denom = sqrt(na) * sqrt(nb)
    var cs: Float64
    if denom == 0.0:
        cs = 1.0 if (na == 0.0 and nb == 0.0) else 0.0
    else:
        cs = dot / denom
    var rl: Float64
    if nb == 0.0:
        rl = sqrt(nd)
    else:
        rl = sqrt(nd) / sqrt(nb)
    return Cmp(cs, rl, mx, ndiff, n)


def _cmp(a: List[Float32], b: List[Float32], off: Int, n: Int) raises -> Cmp:
    return _cmp2(a, b, off, off, n)


def _report(
    name: String, c: Cmp, thresh: Float64, note: String, mut allok: Bool,
    modround: Float64 = -1.0,
):
    var ok = c.cos >= thresh
    if not ok:
        allok = False
    print(
        "  ", "PASS" if ok else "FAIL",
        " ", name,
        "  cos=", c.cos,
        "  relL2=", c.rel,
        "  max_abs=", c.max_abs,
        "  n=", c.n,
        "  bitdiff=", c.n_diff,
        "  | envelope=", thresh, " ", note,
    )
    # SCALE CONTEXT (informational, never a threshold). env_cos_<key>_modround is
    # the measured distance between the two rounding schedules. With the PRIMARY
    # gate now schedule-matched (Mojo schedule B vs kref_*_f32mod at
    # env_cos_*_f32mod) this is no longer an excuse for a miss — it is printed
    # only so the size of the effect that caused the old cross-schedule failures
    # stays visible next to the residual that is actually being measured.
    if modround >= 0.0:
        print(
            "         [scale] 1-cos=", 1.0 - c.cos,
            "  1-envelope=", 1.0 - thresh,
            "  1-modround(schedule A vs B, informational)=", 1.0 - modround,
        )


# ── SCHEDULE-A cross-check: INFO ONLY, never sets the verdict ────────────────
# Mojo (schedule B math) compared against the SCHEDULE-A reference under the
# SCHEDULE-A envelope. This is a CROSS-SCHEDULE comparison — the two sides round
# bf16 in different (both correct) places — so it is not a valid pass/fail
# criterion for this implementation. Printed in full, with its own envelope and
# the verdict it WOULD have produced, so nothing is hidden by the fix.
def _report_sched_a(
    name: String, c: Cmp, env_a: Float64, modround: Float64
):
    print(
        "   info(schedule A) ", name,
        "  cos=", c.cos,
        "  relL2=", c.rel,
        "  max_abs=", c.max_abs,
        "  bitdiff=", c.n_diff, "/", c.n,
        "  | env_cos(A)=", env_a,
        "  would-be verdict=", "PASS" if c.cos >= env_a else "FAIL",
    )
    print(
        "         1-cos=", 1.0 - c.cos, "  1-env(A)=", 1.0 - env_a,
        "  1-modround(A vs B)=", 1.0 - modround,
        "  <- DIFFERENT ROUNDING SCHEDULE (bf16-chain, 3 roundings per",
        " modulate) than the Mojo kernels compute; NOT GATED.",
    )


def _report_info(name: String, c: Cmp, note: String):
    print(
        "   info ", name,
        "  cos=", c.cos,
        "  relL2=", c.rel,
        "  max_abs=", c.max_abs,
        "  n=", c.n,
        "  bitdiff=", c.n_diff,
        "  ", note,
    )


# ── adapter sets ─────────────────────────────────────────────────────────────
def _no_lora() -> Krea2BlockLora:
    return Krea2BlockLora(
        Optional[LoraAdapterDevice](None), Optional[LoraAdapterDevice](None),
        Optional[LoraAdapterDevice](None), Optional[LoraAdapterDevice](None),
        Optional[LoraAdapterDevice](None), Optional[LoraAdapterDevice](None),
        Optional[LoraAdapterDevice](None), Optional[LoraAdapterDevice](None),
    )


def _slot(
    fx: ShardedSafeTensors, name: String, in_f: Int, out_f: Int,
    lscale: Float32, ctx: DeviceContext,
) raises -> Optional[LoraAdapterDevice]:
    """One adapter straight from the fixture: kin_lo_<slot>_A [rank,in] and
    kin_lo_<slot>_B [out,rank], BF16 exactly as the oracle stored them, with the
    oracle's own meta_lora_scale (alpha/rank)."""
    var a = _fv_bf16(fx, "kin_lo_" + name + "_A", ctx)
    var b = _fv_bf16(fx, "kin_lo_" + name + "_B", ctx)
    if a[].shape()[0] != RANK or a[].shape()[1] != in_f:
        raise Error("adapter A shape mismatch for slot " + name)
    if b[].shape()[0] != out_f or b[].shape()[1] != RANK:
        raise Error("adapter B shape mismatch for slot " + name)
    return Optional[LoraAdapterDevice](
        LoraAdapterDevice(a^, b^, RANK, in_f, out_f, lscale)
    )


def _real_lora(
    fx: ShardedSafeTensors, lscale: Float32, ctx: DeviceContext
) raises -> Krea2BlockLora:
    return Krea2BlockLora(
        _slot(fx, "wq", FEATURES, HEADS * HEADDIM, lscale, ctx),
        _slot(fx, "wk", FEATURES, KVHEADS * HEADDIM, lscale, ctx),
        _slot(fx, "wv", FEATURES, KVHEADS * HEADDIM, lscale, ctx),
        _slot(fx, "gate", FEATURES, FEATURES, lscale, ctx),
        _slot(fx, "wo", FEATURES, FEATURES, lscale, ctx),
        _slot(fx, "mlp_gate", FEATURES, MLPDIM, lscale, ctx),
        _slot(fx, "mlp_up", FEATURES, MLPDIM, lscale, ctx),
        _slot(fx, "mlp_down", MLPDIM, FEATURES, lscale, ctx),
    )


# ── one block's frozen weights: 8 matmuls from the CHECKPOINT (bf16, exactly
#    the oracle's load_block dtype), small params from the FIXTURE copies ─────
def _block_weights(
    ck: ShardedSafeTensors, fx: ShardedSafeTensors, bi: Int, ctx: DeviceContext
) raises -> Krea2BlockWeights:
    var p = "blocks." + String(bi) + "."
    return Krea2BlockWeights(
        _fv_bf16(ck, p + "attn.wq.weight", ctx),
        _fv_bf16(ck, p + "attn.wk.weight", ctx),
        _fv_bf16(ck, p + "attn.wv.weight", ctx),
        _fv_bf16(ck, p + "attn.gate.weight", ctx),
        _fv_bf16(ck, p + "attn.wo.weight", ctx),
        _fv_bf16(ck, p + "mlp.gate.weight", ctx),
        _fv_bf16(ck, p + "mlp.up.weight", ctx),
        _fv_bf16(ck, p + "mlp.down.weight", ctx),
        _fv_f32(fx, "kin_qnorm", ctx),
        _fv_f32(fx, "kin_knorm", ctx),
        _fv_f32(fx, "kin_prenorm", ctx),
        _fv_f32(fx, "kin_postnorm", ctx),
        _fv_bf16(fx, "kin_mod_lin", ctx),
    )


# ══════════════════════════════════════════════════════════════════════════════
# GATES 1-6 — the block forward with COND-ROW LoRA ROUTING ON
# ══════════════════════════════════════════════════════════════════════════════
def _gate_lora_forward(
    ck: ShardedSafeTensors, fx: ShardedSafeTensors, bi: Int,
    ctx: DeviceContext, mut allok: Bool,
) raises:
    var lscale = Float32(_host(fx, "meta_lora_scale", ctx)[0])
    var w = _block_weights(ck, fx, bi, ctx)
    var lora = _real_lora(fx, lscale, ctx)

    var x = TArc(_fv(fx, "kin_x", ctx))                      # [1,LFULL,F] bf16
    var vt = _fv(fx, "kin_blk_vec_t", ctx)                   # [1,6F] bf16
    var vc = TArc(_fv(fx, "kin_blk_vec_cond", ctx))          # [1,6F] bf16

    # RoPE tables over the EDIT layout (the C2-gated path, recomputed here).
    var lay_p = Krea2OminiLayout(LTMAX, S_IMG, S_COND, LT)
    var pcomb = krea2_omini_pos_combined(lay_p, GRID, GRID, 0, 0, Float32(1.0))
    var pos_flat = Tensor.from_host(pcomb.copy(), _s1(LFULL * 3), STDtype.F32, ctx)
    var axes0 = List[Int]()
    axes0.append(32)
    axes0.append(48)
    axes0.append(48)
    var rope0 = build_krea2_rope(pos_flat, axes0, THETA, ctx, STDtype.F32)
    var rcos = rope0[0].clone(ctx)
    var rsin = rope0[1].clone(ctx)
    var cos_q = _tile_rope_table(rcos, LFULL, HEADS, HALF, ctx)
    var sin_q = _tile_rope_table(rsin, LFULL, HEADS, HALF, ctx)
    var cos_k = _tile_rope_table(rcos, LFULL, KVHEADS, HALF, ctx)
    var sin_k = _tile_rope_table(rsin, LFULL, KVHEADS, HALF, ctx)

    var lay = Krea2OminiLayout(LTMAX, S_IMG, S_COND, LT)
    var split = krea2_omini_mod_split(lay)                   # == COND_OFF
    print("")
    print("---- block", bi, ": EDIT forward, COND-ROW LoRA ON, real_len=",
          lay.real_len(), " cond rows [", lay.cond_off(), ",", lay.pad_off(),
          ")  rank=", RANK, " scale=", lscale, " ----")

    var fwd = krea2_single_stream_block_lora[LFULL, HEADS, KVHEADS, HEADDIM](
        x.copy(), vt, w, lora, rcos, rsin, cos_q, sin_q, cos_k, sin_k, EPS, ctx,
        Optional[Int](lay.real_len()), Optional[TArc](vc.copy()),
        Optional[Int](split), Optional[Int](lay.cond_len()),
    )

    var pre_n = PAD_OFF * FEATURES          # elements in the real prefix
    var all_n = LFULL * FEATURES
    var cond_n = S_COND * FEATURES
    var img_n = S_IMG * FEATURES

    # ── 1. per-segment modulation (LoRA-independent; re-gated for regression) ─
    print("")
    print("---- 1. kref_xm — per-segment modulation (unchanged by C3) ----")
    var xm_h = fwd.saved.xm[].to_host(ctx)
    var xm_b = _host(fx, "kref_xm_f32mod", ctx)          # SCHEDULE B == Mojo
    _report("kref_xm_f32mod [full LFULL]", _cmp(xm_h, xm_b, 0, all_n),
            _env(fx, "xm_f32mod", ctx),
            "(PRIMARY: schedule-B ref + env_cos_xm_f32mod)", allok,
            _modr(fx, "xm", ctx))
    var xm_a = _host(fx, "kref_xm", ctx)                 # SCHEDULE A, info only
    _report_sched_a("kref_xm [full LFULL]", _cmp(xm_h, xm_a, 0, all_n),
                    _env(fx, "xm", ctx), _modr(fx, "xm", ctx))

    # ── 2. LoRA-ON attention seam ────────────────────────────────────────────
    print("")
    print("---- 2. kref_attn_raw — LoRA-ON attention (C2 could NOT gate this) ----")
    var at_h = fwd.saved.attn_flat[].to_host(ctx)
    var at_b = _host(fx, "kref_attn_raw_f32mod", ctx)
    _report("kref_attn_raw_f32mod [prefix 0:real_len]",
            _cmp(at_h, at_b, 0, pre_n), _env(fx, "attn_raw_f32mod", ctx),
            "(PRIMARY: schedule-B ref + env_cos_attn_raw_f32mod)", allok,
            _modr(fx, "attn_raw", ctx))
    _report_info("kref_attn_raw_f32mod [COND rows]",
                 _cmp(at_h, at_b, COND_OFF * FEATURES, cond_n),
                 "<- the adapter acts DIRECTLY on these")
    _report_info("kref_attn_raw_f32mod [IMG rows]",
                 _cmp(at_h, at_b, LT * FEATURES, img_n),
                 "<- reached only through bidirectional attention")
    var at_a = _host(fx, "kref_attn_raw", ctx)
    _report_sched_a("kref_attn_raw [prefix 0:real_len]",
                    _cmp(at_h, at_a, 0, pre_n), _env(fx, "attn_raw", ctx),
                    _modr(fx, "attn_raw", ctx))
    var at_off = _host(fx, "kref_attn_raw_nolora", ctx)
    _report_info("  (fixture) kref_attn_raw vs kref_attn_raw_nolora [prefix]",
                 _cmp2(at_a, at_off, 0, 0, pre_n),
                 "<- the size of the effect this gate is measuring")

    # ── 3. kref_gated — carries wq/wk/wv/gate cond-row deltas ────────────────
    print("")
    print("---- 3. kref_gated — sdpa * sigmoid(gate), the wo LoRA input ----")
    var g_h = fwd.saved.gated[].to_host(ctx)
    var g_b = _host(fx, "kref_gated_f32mod", ctx)
    _report("kref_gated_f32mod [prefix 0:real_len]", _cmp(g_h, g_b, 0, pre_n),
            _env(fx, "gated_f32mod", ctx),
            "(PRIMARY: schedule-B ref + env_cos_gated_f32mod)", allok,
            _modr(fx, "gated", ctx))
    _report_info("kref_gated_f32mod [full LFULL]", _cmp(g_h, g_b, 0, all_n),
                 "<- NOT gated (pad rows)")
    _report_info("kref_gated_f32mod [COND rows]",
                 _cmp(g_h, g_b, COND_OFF * FEATURES, cond_n), "")
    _report_info("kref_gated_f32mod [IMG rows]",
                 _cmp(g_h, g_b, LT * FEATURES, img_n), "")
    var g_a = _host(fx, "kref_gated", ctx)
    _report_sched_a("kref_gated [prefix 0:real_len]", _cmp(g_h, g_a, 0, pre_n),
                    _env(fx, "gated", ctx), _modr(fx, "gated", ctx))

    # ── 4. kref_attn — + the wo cond-row delta ───────────────────────────────
    print("")
    print("---- 4. kref_attn — wo(gated), with the wo adapter on cond rows ----")
    var a_h = fwd.saved.a[].to_host(ctx)
    var a_b = _host(fx, "kref_attn_f32mod", ctx)
    _report("kref_attn_f32mod [prefix 0:real_len]", _cmp(a_h, a_b, 0, pre_n),
            _env(fx, "attn_f32mod", ctx),
            "(PRIMARY: schedule-B ref + env_cos_attn_f32mod)", allok,
            _modr(fx, "attn", ctx))
    _report_info("kref_attn_f32mod [COND rows]",
                 _cmp(a_h, a_b, COND_OFF * FEATURES, cond_n), "")
    var a_a = _host(fx, "kref_attn", ctx)
    _report_sched_a("kref_attn [prefix 0:real_len]", _cmp(a_h, a_a, 0, pre_n),
                    _env(fx, "attn", ctx), _modr(fx, "attn", ctx))

    # ── 5. kref_out — the whole block ────────────────────────────────────────
    print("")
    print("---- 5. kref_out — BLOCK OUTPUT, all 8 adapters routed to cond rows ----")
    var out_h = fwd.out[].to_host(ctx)
    var out_b = _host(fx, "kref_out_f32mod", ctx)
    _report("kref_out_f32mod [prefix 0:real_len]", _cmp(out_h, out_b, 0, pre_n),
            _env(fx, "out_f32mod", ctx),
            "(PRIMARY: schedule-B ref + env_cos_out_f32mod)", allok,
            _modr(fx, "out", ctx))
    _report_info("kref_out_f32mod [full LFULL]", _cmp(out_h, out_b, 0, all_n),
                 "<- NOT gated (pad rows)")
    _report_info("kref_out_f32mod [COND rows]",
                 _cmp(out_h, out_b, COND_OFF * FEATURES, cond_n), "")
    _report_info("kref_out_f32mod [TXT_real rows]",
                 _cmp(out_h, out_b, 0, LT * FEATURES), "")
    var out_a = _host(fx, "kref_out", ctx)
    _report_sched_a("kref_out [prefix 0:real_len]", _cmp(out_h, out_a, 0, pre_n),
                    _env(fx, "out", ctx), _modr(fx, "out", ctx))

    # ── 6. kref_out_img — the rows the loss reads ────────────────────────────
    # THIS is the key the mis-specified criterion failed on at block 27: the
    # schedule-A envelope (1-cos 1.043e-05) is SMALLER than the pure A-vs-B
    # rounding-schedule distance (1.097e-05), so no correct schedule-B
    # implementation could pass it. Primary is now schedule-matched.
    print("")
    print("---- 6. kref_out_img — IMG rows = what the loss reads ----")
    var img_b = _host(fx, "kref_out_img_f32mod", ctx)
    _report("kref_out_img_f32mod [IMG rows]",
            _cmp2(out_h, img_b, LT * FEATURES, 0, img_n),
            _env(fx, "out_img_f32mod", ctx),
            "(PRIMARY: schedule-B ref + env_cos_out_img_f32mod)", allok,
            _modr(fx, "out_img", ctx))
    var img_a = _host(fx, "kref_out_img", ctx)
    _report_sched_a("kref_out_img [IMG rows]",
                    _cmp2(out_h, img_a, LT * FEATURES, 0, img_n),
                    _env(fx, "out_img", ctx), _modr(fx, "out_img", ctx))

    # ══════════════════════════════════════════════════════════════════════
    # 7. ROUTING ISOLATION — bit-level proof the delta lands on cond rows only.
    #
    # Re-run the SAME block with the adapters REMOVED (everything else
    # identical) and compare `saved.gate_pre`, the attention gate projection.
    # WHY THAT SEAM AND NO OTHER: gate_pre = base(xm) [+ delta], and xm is
    # computed before any projection, so gate_pre is a PURE PER-ROW map of the
    # block input — no cross-row mixing has happened yet. Every later seam
    # (kref_attn, kref_out, the MLP seams) sits downstream of attention, where
    # the cond rows' K/V legitimately reach every other row, so a bit-equality
    # assertion there would be WRONG (see "WHAT IS NOT ASSERTED" in the header).
    # Expected: rows [0,cond_off) and [pad_off,L) BIT-EQUAL, cond rows CHANGED.
    #
    # A third run with the adapters applied over the FULL sequence (cond_len
    # omitted) is the contrast: that one MUST change the img rows of gate_pre.
    # Without it, "bit-equal img rows" could just mean the adapters were silently
    # dropped.
    # ══════════════════════════════════════════════════════════════════════
    print("")
    print("---- 7. ROUTING ISOLATION at the gate projection (pre-attention) ----")
    var fwd_off = krea2_single_stream_block_lora[LFULL, HEADS, KVHEADS, HEADDIM](
        x.copy(), vt, w, _no_lora(), rcos, rsin, cos_q, sin_q, cos_k, sin_k,
        EPS, ctx, Optional[Int](lay.real_len()), Optional[TArc](vc.copy()),
        Optional[Int](split), Optional[Int](lay.cond_len()),
    )
    var lora2 = _real_lora(fx, lscale, ctx)
    var fwd_uni = krea2_single_stream_block_lora[LFULL, HEADS, KVHEADS, HEADDIM](
        x.copy(), vt, w, lora2, rcos, rsin, cos_q, sin_q, cos_k, sin_k,
        EPS, ctx, Optional[Int](lay.real_len()), Optional[TArc](vc.copy()),
        Optional[Int](split),                       # NO cond_len -> full-seq LoRA
    )
    var gp_on = fwd.saved.gate_pre[].to_host(ctx)
    var gp_off = fwd_off.saved.gate_pre[].to_host(ctx)
    var gp_uni = fwd_uni.saved.gate_pre[].to_host(ctx)

    var c_head = _cmp(gp_on, gp_off, 0, COND_OFF * FEATURES)
    var c_tail = _cmp(gp_on, gp_off, PAD_OFF * FEATURES,
                      (LFULL - PAD_OFF) * FEATURES)
    var c_cond = _cmp(gp_on, gp_off, COND_OFF * FEATURES, cond_n)
    var iso_ok = (c_head.n_diff == 0) and (c_tail.n_diff == 0) \
        and (c_cond.n_diff > 0)
    if not iso_ok:
        allok = False
    print(
        "  ", "PASS" if iso_ok else "FAIL",
        " routed gate_pre vs LoRA-off:  txt+img rows bitdiff=", c_head.n_diff,
        "/", c_head.n, " (must be 0)   pad rows bitdiff=", c_tail.n_diff,
        "/", c_tail.n, " (must be 0)   COND rows bitdiff=", c_cond.n_diff,
        "/", c_cond.n, " (must be > 0)  max_abs=", c_cond.max_abs,
    )
    var c_uni_img = _cmp(gp_uni, gp_off, 0, COND_OFF * FEATURES)
    var contrast_ok = c_uni_img.n_diff > 0
    if not contrast_ok:
        allok = False
    print(
        "  ", "PASS" if contrast_ok else "FAIL",
        " CONTRAST — full-sequence LoRA (cond_len omitted) vs LoRA-off:",
        " txt+img rows bitdiff=", c_uni_img.n_diff, "/", c_uni_img.n,
        " (must be > 0, else the adapters were silently inert)",
        " max_abs=", c_uni_img.max_abs,
    )
    # Routed vs full-sequence LoRA ON THE COND ROWS. Algebraically these are the
    # same numbers — routing changes WHERE the delta is applied, not its value.
    # NOT ASSERTED BIT-EQUAL, and the reason is honest and specific: with
    # KREA2_BATCH_LORA_GROUPS the four qkvg adapters share ONE down-projection
    # GEMM, and routing changes that GEMM's M from L (2432) to c_len (1024).
    # cuBLAS is free to pick a different kernel/tiling for a different M, and it
    # does; the accumulation order over K then differs. MEASURED below: a few
    # hundred of 6.3M elements differ, by at most one bf16 ULP. Asserting
    # bit-equality here would be asserting GEMM shape-invariance, which is not a
    # property cuBLAS provides — so this line reports and does not gate.
    var c_same = _cmp(gp_on, gp_uni, COND_OFF * FEATURES, cond_n)
    print(
        "   info  routed vs full-sequence LoRA on the COND rows: bitdiff=",
        c_same.n_diff, "/", c_same.n, " cos=", c_same.cos,
        " relL2=", c_same.rel, " max_abs=", c_same.max_abs,
        " <- NOT gated: the shared LoRA down-projection GEMM's M changes"
        " 2432->1024, and cuBLAS accumulation is not shape-invariant",
    )

    # The block-level img-row coupling the oracle asserts nonzero (max|delta|
    # 8.0 at block 0). Reported, never asserted bit-equal — see the header.
    var out_off_h = fwd_off.out[].to_host(ctx)
    var c_couple = _cmp(out_h, out_off_h, LT * FEATURES, img_n)
    print("   info  block-out IMG-row coupling (cond LoRA on vs off): max_abs=",
          c_couple.max_abs, " bitdiff=", c_couple.n_diff, "/", c_couple.n,
          " <- MUST be nonzero (bidirectional attention); NOT a leak")
    if c_couple.max_abs == 0.0:
        allok = False
        print("   FAIL  the condition LoRA never reached the image rows")


# ══════════════════════════════════════════════════════════════════════════════
# GATE 8 — CONDLEN=0 BIT-EQUALITY, now WITH the 8 real adapters attached.
# Run A: the PRE-C2 call form (exactly what krea2_stack emits today).
# Run B: the POST-C3 call form of a CONDLEN=0 build — vec_cond IS supplied,
#        cond_off = krea2_omini_mod_split(lay0) which is -1 for s_cond == 0, and
#        cond_len = 0. Both guards must fall through to the pre-existing path.
# Compared on the block output AND on gate_pre (the LoRA seam itself), so a
# regression in _lora_delta_rows / _add_delta_rows cannot hide behind a
# downstream reduction.
# ══════════════════════════════════════════════════════════════════════════════
def _gate_condlen0(
    ck: ShardedSafeTensors, fx: ShardedSafeTensors, bi: Int,
    ctx: DeviceContext, mut allok: Bool,
) raises:
    print("")
    print("---- 8. CONDLEN=0 BIT-EQUALITY with the adapters ATTACHED ----")
    var lscale = Float32(_host(fx, "meta_lora_scale", ctx)[0])
    var lay0 = Krea2OminiLayout(LTMAX, S_IMG, 0, LT)
    lay0.check_flash_prefix()
    var split0 = krea2_omini_mod_split(lay0)
    print("   layout s_cond=0: lfull=", lay0.lfull(), " real_len=", lay0.real_len(),
          " mod_split=", split0, " cond_len=", lay0.cond_len(),
          " (-1 / 0 => block runs the UNCHANGED path)")
    if split0 != -1:
        allok = False
        print("   FAIL  krea2_omini_mod_split(s_cond=0) must be -1")

    var p0 = krea2_omini_pos_combined(lay0, GRID, GRID, 0, 0, Float32(1.0))
    var pos_flat = Tensor.from_host(p0.copy(), _s1(L_NC * 3), STDtype.F32, ctx)
    var axes = List[Int]()
    axes.append(32)
    axes.append(48)
    axes.append(48)
    var rope = build_krea2_rope(pos_flat, axes, THETA, ctx, STDtype.F32)
    var rcos = rope[0].clone(ctx)
    var rsin = rope[1].clone(ctx)
    var cos_q = _tile_rope_table(rcos, L_NC, HEADS, HALF, ctx)
    var sin_q = _tile_rope_table(rsin, L_NC, HEADS, HALF, ctx)
    var cos_k = _tile_rope_table(rcos, L_NC, KVHEADS, HALF, ctx)
    var sin_k = _tile_rope_table(rsin, L_NC, KVHEADS, HALF, ctx)

    # input with the COND segment removed: [TXT_real | IMG | TXT_pad]
    var xf = _fv(fx, "kin_x", ctx)
    var xh = slice(xf, 1, 0, COND_OFF, ctx)
    var xt = slice(xf, 1, PAD_OFF, LTMAX - LT, ctx)
    var x_nc = TArc(concat(1, ctx, xh, xt))                  # [1, L_NC, F] bf16

    var w = _block_weights(ck, fx, bi, ctx)
    var vt = _fv(fx, "kin_blk_vec_t", ctx)
    var vc = TArc(_fv(fx, "kin_blk_vec_cond", ctx))

    var fa = krea2_single_stream_block_lora[L_NC, HEADS, KVHEADS, HEADDIM](
        x_nc.copy(), vt, w, _real_lora(fx, lscale, ctx), rcos, rsin,
        cos_q, sin_q, cos_k, sin_k, EPS, ctx, Optional[Int](lay0.real_len()),
    )
    var a_out = fa.out[].to_host(ctx)
    var a_gp = fa.saved.gate_pre[].to_host(ctx)

    var fb = krea2_single_stream_block_lora[L_NC, HEADS, KVHEADS, HEADDIM](
        x_nc.copy(), vt, w, _real_lora(fx, lscale, ctx), rcos, rsin,
        cos_q, sin_q, cos_k, sin_k, EPS, ctx, Optional[Int](lay0.real_len()),
        Optional[TArc](vc.copy()), Optional[Int](split0),
        Optional[Int](lay0.cond_len()),
    )
    var b_out = fb.out[].to_host(ctx)
    var b_gp = fb.saved.gate_pre[].to_host(ctx)

    var n = L_NC * FEATURES
    var c_out = _cmp(a_out, b_out, 0, n)
    var c_gp = _cmp(a_gp, b_gp, 0, n)
    var bit_ok = (c_out.n_diff == 0) and (c_gp.n_diff == 0)
    if not bit_ok:
        allok = False
    print(
        "  ", "PASS" if bit_ok else "FAIL",
        " condlen=0 BIT-EQUAL  block out: bitdiff=", c_out.n_diff, "/", c_out.n,
        " max_abs=", c_out.max_abs,
        " | gate_pre LoRA seam: bitdiff=", c_gp.n_diff, "/", c_gp.n,
        " max_abs=", c_gp.max_abs,
    )
    var nz = 0
    for i in range(n):
        if a_out[i] != 0.0:
            nz += 1
    print("   (non-degeneracy: ", nz, "/", n,
          " nonzero elements in the condlen=0 block output)")


# ══════════════════════════════════════════════════════════════════════════════
def _run_block(bi: Int, ck: ShardedSafeTensors, ctx: DeviceContext) raises -> Bool:
    var path = FIX_DIR + "krea2_omini_block" + ("0" if bi < 10 else "") \
        + String(bi) + "_oracle.safetensors"
    print("")
    print("==============================================================")
    print("FIXTURE:", path)
    var fx = ShardedSafeTensors.open(path)

    _expect_i(fx, "meta_block_index", bi)
    _expect_i(fx, "meta_ltmax", LTMAX)
    _expect_i(fx, "meta_s_img", S_IMG)
    _expect_i(fx, "meta_s_cond", S_COND)
    _expect_i(fx, "meta_lt", LT)
    _expect_i(fx, "meta_lfull", LFULL)
    _expect_i(fx, "meta_cond_off", COND_OFF)
    _expect_i(fx, "meta_pad_off", PAD_OFF)
    _expect_i(fx, "meta_real_len", PAD_OFF)
    _expect_i(fx, "meta_features", FEATURES)
    _expect_i(fx, "meta_heads", HEADS)
    _expect_i(fx, "meta_kvheads", KVHEADS)
    _expect_i(fx, "meta_headdim", HEADDIM)
    _expect_i(fx, "meta_mlpdim", MLPDIM)
    _expect_i(fx, "meta_grid", GRID)
    _expect_i(fx, "meta_rank", RANK)
    _expect_i(fx, "meta_cond_delta_h", 0)
    _expect_i(fx, "meta_cond_delta_w", 0)

    print("THRESHOLDS (from the fixture, not invented).")
    print("   PRIMARY = schedule B (*_f32mod) — modulate/residual_gate in F32,")
    print("   rounded ONCE at store, which is exactly what")
    print("   serenitymojo/ops/elementwise.mojo _modulate_kernel_bf16 and")
    print("   _resgate_kernel_bf16 compute. Apples-to-apples.")
    print("   key            env(B)=GATED         env(A)=info-only")
    print("   xm            ", _env(fx, "xm_f32mod", ctx), "  ",
          _env(fx, "xm", ctx))
    print("   attn_raw      ", _env(fx, "attn_raw_f32mod", ctx), "  ",
          _env(fx, "attn_raw", ctx))
    print("   gated         ", _env(fx, "gated_f32mod", ctx), "  ",
          _env(fx, "gated", ctx))
    print("   attn          ", _env(fx, "attn_f32mod", ctx), "  ",
          _env(fx, "attn", ctx))
    print("   out           ", _env(fx, "out_f32mod", ctx), "  ",
          _env(fx, "out", ctx))
    print("   out_img       ", _env(fx, "out_img_f32mod", ctx), "  ",
          _env(fx, "out_img", ctx))
    print("   These are the oracle's OWN bf16-vs-f32 cosines per seam under")
    print("   THAT schedule on this block, both measured against the SAME F32-")
    print("   storage run — the ceiling for any bf16 implementation of that")
    print("   schedule. Nothing gated tighter, nothing loosened, nothing")
    print("   borrowed. The env(A) column is printed for reference only.")

    var allok = True
    _gate_lora_forward(ck, fx, bi, ctx, allok)
    _gate_condlen0(ck, fx, bi, ctx, allok)

    print("")
    if allok:
        print("BLOCK", bi, "VERDICT: PASS")
    else:
        print("BLOCK", bi, "VERDICT: FAIL")
    return allok


def main() raises:
    var ctx = DeviceContext()
    print("==== krea2 OminiControl EDIT — C3 COND-ROW LoRA parity gate ====")
    print("layout [TXT_real(", LT, ") | IMG(", S_IMG, ") | COND(", S_COND,
          ") | TXT_pad(", LTMAX - LT, ")]  LFULL=", LFULL,
          " real_len=", PAD_OFF, " cond_off=", COND_OFF)
    print("LoRA routed to rows [", COND_OFF, ",", PAD_OFF,
          ") at all 8 Linears; frozen base runs the FULL sequence.")
    print("checkpoint:", CKPT)
    var ck = ShardedSafeTensors.open(String(CKPT))

    var ok0 = _run_block(0, ck, ctx)
    var ok27 = _run_block(27, ck, ctx)

    print("")
    print("==============================================================")
    if ok0 and ok27:
        print("C3 COND-ROW LoRA GATE: PASS on BOTH block00 and block27")
    else:
        print("C3 COND-ROW LoRA GATE: FAIL — block00 ",
              "PASS" if ok0 else "FAIL", ", block27 ",
              "PASS" if ok27 else "FAIL")
