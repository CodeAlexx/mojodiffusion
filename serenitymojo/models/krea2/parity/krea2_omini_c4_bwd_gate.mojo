# serenitymojo/models/krea2/parity/krea2_omini_c4_bwd_gate.mojo
#
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  C4 BACKWARD PARITY GATE — krea2 OminiControl EDIT vertical              ║
# ║  Per-segment d_mod (two temb chains) + cond-row dA/dB + full-sequence dX ║
# ║  against the CUDA fixtures written by scripts/krea2_omini_torch_oracle.py║
# ║  CPU torch is NEVER an oracle here: every reference number comes out of  ║
# ║  that fixture, produced on the RTX 5080 with the real Krea-2-Raw bf16    ║
# ║  weights and the real krea2 activation cache.                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# WHAT C4 IMPLEMENTS (models/krea2/krea2_block.mojo)
#   `krea2_single_stream_block_lora_backward` and `..._backward_dev` gained the
#   SAME three optional switches the C2/C3 forward has — vec_cond, cond_off,
#   cond_len — and with them:
#     * PER-SEGMENT d_x: every modulate / residual-gate backward runs once per
#       span, against the chunk that actually modulated that row
#       (`_modulate_backward_seg2` / `_gate_residual_backward_seg2`).
#     * PER-SEGMENT d_mod: the chunk grads are accumulated SEPARATELY into
#       d_vec_t (rows [0, cond_off): TXT_real + IMG) and d_vec_cond (rows
#       [cond_off, real_len): COND) — the trainer's two temb chains. Packed
#       F32 [6*features] in `_mod6` chunk order by `_pack_d_vec`. Since
#       mods = vec + mod.lin, this is the grad w.r.t. BOTH.
#     * COND-ROW dA/dB: `_linear_bwd_dx` / `_linear_bwd_dx_dev` gained c_off /
#       c_len, the exact mirror of C3's forward routing — the adapter sees only
#       the cond rows, its d_x contribution is scatter-added back onto only
#       those rows, and the FROZEN BASE dX still runs the FULL sequence.
#   With the switches absent every one of those lines is the pre-C4 call with
#   the pre-C4 arguments (section 5 proves it bit-for-bit).
#
# ══════════════════════════════════════════════════════════════════════════════
# THE PAD TAIL IS EXCLUDED FROM d_mod — THE DECISION AND ITS EVIDENCE
# ══════════════════════════════════════════════════════════════════════════════
#   Under per-segment modulation the TXT_pad tail rides the t=0 span, so the
#   naive reduction "rows [cond_off, L)" would put pad-row gradient into
#   d_vec_cond. C4 stops the reduction at real_len. The evidence is in the
#   fixture, measured by the oracle on the same CUDA device with the same real
#   weights (its "[pad-row evidence]" block; shipped as env_cos_*_padcontrib and
#   env_bit_*_padzero_equal, re-printed by section 0 below):
#     * pad rows CANNOT reach the real rows. With pad KEY columns masked the
#       softmax weight on them is exp(-inf) = 0 EXACTLY, so their dK/dV are
#       exactly zero. Measured: d_x[0:real_len] and ALL 16 LoRA dA/dB are
#       BIT-EQUAL whether or not the pad tail carries an upstream gradient.
#     * they DO reach d_vec_cond, and hard: cos(with pad, without pad) = 0.679
#       at block 0 and 0.994 at block 27. d_vec_t is BIT-EQUAL (its span has no
#       pad rows). So the pad tail's ONLY effect is corrupting the t=0 chunks.
#     * in the Mojo production path those rows are not merely "different": the
#       cuDNN flash TAIL-padmask kernel masks only KEY columns, so pad QUERY
#       rows carry masked-out garbage BY DESIGN.
#     * and the trainer's loss reads the IMG rows only, so the real pad-row
#       d_out is zero regardless.
#   Consequences for this gate: the oracle's SCHEDULE-B backward references are
#   all produced with kin_d_out_pz (kin_d_out with rows [real_len:] zeroed) and
#   THIS GATE FEEDS THE SAME TENSOR. Section 4 then re-runs the production
#   backward with the ORIGINAL pad-nonzero kin_d_out and asks which of the two
#   shipped references the result is closer to — the pad-EXCLUDED
#   kref_d_blk_vec_cond_f32mod or the pad-INCLUDED
#   kref_d_blk_vec_cond_padincl_f32mod. That is a threshold-free discrimination
#   (see section 4's own note for why a bit-equality test is impossible here).
#
# WHAT IS GATED, IN THE BRIEF'S ORDER (verdict is the AND of the PRIMARY lines)
#   1. kref_d_x            — full sequence (reported full + prefix + per segment)
#   2. the 8 kref_<slot>_dA / _dB pairs — cond-row routed
#   3. kref_d_blk_vec_t and kref_d_blk_vec_cond — the per-segment d_mod pair
#   4. PAD-ROW EXCLUSION   — which reference the production backward matches
#   6. DEVICE-GRAD TWIN    — `..._backward_dev` bit-identical to `..._backward`
#   5. CONDLEN=0 BACKWARD BIT-EQUALITY — the live-trainer regression guard
#
# ══════════════════════════════════════════════════════════════════════════════
# THREE MOJO PATHS — WHICH ONE CARRIES A VERDICT, AND WHY
# ══════════════════════════════════════════════════════════════════════════════
#   Every path runs the SAME C4 code with the SAME switches on the SAME real
#   weights and the SAME fixture inputs. They differ in storage dtype and in
#   which SDPA kernel runs.
#
#   PATH 1 — bf16, L = real_len, `sdpa_nomask`   ......... PRIMARY (env rule)
#     The unpadded real prefix. This is the bf16 path whose ATTENTION MATH is
#     the one the oracle computes (F32 scores / softmax / AV with bf16 storage —
#     krea2_block.mojo calls `real_len == L -> sdpa_nomask` the parity gate
#     path), and whose modulate rounding is SCHEDULE B. Both sides of the
#     comparison therefore differ ONLY in implementation, which is exactly what
#     env_cos_<key>_f32mod bounds.
#     Reference kref_<key>_f32mod, threshold env_cos_<key>_f32mod.
#     EXCEPT: the fixture itself marks keys whose envelope is below 0.999 as
#     "NOT bf16-gateable, use *_f32env" — at the EARLY blocks the t=0 condition
#     modulation roughly doubles the block's output dynamic range and the
#     resulting cancellation in the backward falls below bf16's 8-bit mantissa
#     (the oracle's isolation probe: cos(d_x) 0.99849 with uniform mods(t) vs
#     0.49377 per-segment at block 0; both 0.99993 at block 27). MEASURED,
#     env_cos_d_x_f32mod = 0.4437 at block 0 and 0.99994 at block 27. A "gate"
#     at 0.4437 asserts nothing, so this file applies the FIXTURE'S OWN RULE: a
#     bf16 key is PRIMARY when its shipped envelope is >= 0.999, INFO otherwise,
#     and every INFO key carries its verdict on path 3. No threshold is invented
#     and none is loosened.
#
#   PATH 2 — bf16, L = LFULL, real_len < L => the cuDNN flash TAIL-PADMASK
#     kernel. THIS IS THE PRODUCTION TRAINER PATH, and it is reported in full
#     (every key, its residual, and the verdict it WOULD have carried) but it
#     does NOT set the verdict. Two measured reasons, both printed:
#       (a) its attention backward is a DIFFERENT ACCUMULATION ORDER, not a
#           different rounding point — the oracle's own header says the cuDNN
#           flash kernel "is the same math with a different accumulation order;
#           it is a value-tolerance path, not a bit path". env_cos_<key>_f32mod
#           is measured with the MATH attention on both sides, so it does not
#           bound it. Gating a flash run on it would be the same class of
#           mis-specified criterion C3 had to fix.
#       (b) its dQ is NONDETERMINISTIC run to run (cuDNN atomics —
#           krea2_block.mojo documents this). Section 2c measures it directly by
#           running the identical call twice.
#     Section 2b isolates (a): the same Mojo code, same dtype, same inputs, only
#     the SDPA kernel changed (path 2 vs path 1 on the shared prefix rows).
#
#   PATH 3 — F32, L = real_len, `sdpa_nomask`   ................... PRIMARY
#     Reference kref_<key>_f32env_pz, the oracle's F32-storage backward on the
#     same device with the same real weights and the same pad-zeroed gradient.
#     THRESHOLD F32_BAR = 0.999, the stack's standard parity bar for an
#     independent implementation (the bar krea2_block_parity.mojo uses and the
#     one the oracle header names for the forward). Deliberately NOT an oracle
#     envelope, and the reason is stated plainly: no oracle-internal measurement
#     bounds cross-library F32 differences (cuBLAS vs Mojo GEMM tiling,
#     reduction order in rmsnorm/softmax). The one thing the oracle CAN measure
#     — how much the unpadded formulation alone moves the answer — is shipped as
#     env_cos_<key>_f32unpad and printed on every path-3 line; it is 1.0
#     (1-cos <= 2.4e-07) for all 19 keys on both blocks, i.e. the padding
#     formulation contributes nothing and a path-3 residual is the
#     implementation's own.
#
# WHAT IS NOT ASSERTED (deliberately)
#   * No bit-equality is claimed for any IMG-row quantity. Editing the condition
#     rows' K/V propagates to every row through bidirectional attention; that is
#     the method working (C3 gate header, oracle "[coupling]").
#   * The pad ROWS of d_x are reported, never gated: with the pad-zeroed
#     gradient they are zero on both sides by construction, which is a property
#     of the input, not evidence about the implementation.
#   * Nothing on the flash path is asserted bit-equal — see PATH 2 (b).
#   * The batch-2 backward entries are NOT covered: the b2 FORWARD has no
#     vec_cond/cond_off/cond_len at all, so b2 + EDIT is not a reachable
#     configuration. See the SCOPE note in krea2_block.mojo's forward.
#
# RUN (never chained after a mojo build with &&):
#   cd /home/alex/mojodiffusion
#   rm -f serenitymojo.mojopkg
#   pixi run mojo run -I . \
#       -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa \
#       serenitymojo/models/krea2/parity/krea2_omini_c4_bwd_gate.mojo

from std.gpu.host import DeviceContext
from std.collections import List, Optional
from std.math import sqrt
from std.memory import ArcPointer
from serenitymojo.tensor import Tensor
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.tensor_algebra import slice, concat
from serenitymojo.models.klein.lora_block import LoraAdapterDevice
from serenitymojo.models.krea2.krea2_block import (
    Krea2BlockWeights, Krea2BlockLora, Krea2LoraGrad, Krea2BlockGrads,
    Krea2LoraGradT,
    krea2_single_stream_block_lora,
    krea2_single_stream_block_lora_backward,
    krea2_single_stream_block_lora_backward_dev,
)
from serenitymojo.models.dit.krea2_dit import build_krea2_rope, _tile_rope_table
from serenitymojo.training.krea2_omini_layout import (
    Krea2OminiLayout, krea2_omini_mod_split, krea2_omini_pos_combined,
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
comptime GRID = 32
comptime S_IMG = GRID * GRID               # 1024
comptime S_COND = GRID * GRID              # 1024
comptime LT = 190                          # the cache sample's real caption len
comptime LFULL = LTMAX + S_IMG + S_COND    # 2432  (path 2: bf16 flash-padmask)
comptime COND_OFF = LT + S_IMG             # 1214
comptime PAD_OFF = COND_OFF + S_COND       # 2238 == real_len
comptime L_RL = PAD_OFF                    # 2238  (paths 1/3: unpadded prefix)
comptime L_NC = LTMAX + S_IMG              # 1408  (CONDLEN=0 regression length)
comptime RANK = 16
comptime EPS = Float32(1e-5)
comptime THETA = Float32(1000.0)

# The FIXTURE'S OWN rule for whether a bf16 key may carry a verdict: the oracle
# flags every key whose measured bf16-vs-F32 envelope is below this as
# "NOT bf16-gateable, use *_f32env". Not a threshold on any comparison — a
# threshold on whether the ENVELOPE is meaningful.
comptime GATEABLE_MIN = Float64(0.999)
# The stack's standard parity bar for an independent implementation, used ONLY
# on the F32 path. See PATH 3 for why this is a bar and not an envelope.
comptime F32_BAR = Float64(0.999)

# report modes
comptime MODE_BF16 = 0      # PRIMARY when env >= GATEABLE_MIN, else INFO
comptime MODE_F32 = 1       # PRIMARY at F32_BAR
comptime MODE_FLASH = 2     # REPORT ONLY (would-be verdict printed)


def _s1(a: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a)
    return s^


# ── fixture / checkpoint readers ─────────────────────────────────────────────
def _fv(st: ShardedSafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    return Tensor.from_view(st.tensor_view(name), ctx)


def _fv_bf16(st: ShardedSafeTensors, name: String, ctx: DeviceContext) raises -> TArc:
    return TArc(Tensor.from_view_as_bf16(st.tensor_view(name), ctx))


def _fv_f32(st: ShardedSafeTensors, name: String, ctx: DeviceContext) raises -> TArc:
    return TArc(Tensor.from_view_as_f32(st.tensor_view(name), ctx))


def _host(st: ShardedSafeTensors, name: String, ctx: DeviceContext) raises -> List[Float32]:
    var t = Tensor.from_view(st.tensor_view(name), ctx)
    return t.to_host(ctx)


def _meta_i(st: ShardedSafeTensors, name: String) raises -> Int:
    var b = st.tensor_bytes(name)
    if len(b) != 4:
        raise Error(String("meta ") + name + ": expected 4 bytes")
    var v = Int(b[0]) | (Int(b[1]) << 8) | (Int(b[2]) << 16) | (Int(b[3]) << 24)
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


# ── the ONLY source of a threshold in this file ──────────────────────────────
def _env(fx: ShardedSafeTensors, key: String, ctx: DeviceContext) raises -> Float64:
    var v = _host(fx, "env_cos_" + key, ctx)
    if len(v) != 1:
        raise Error("fixture ships no env_cos_" + key)
    return Float64(v[0])


def _scalar(fx: ShardedSafeTensors, key: String, ctx: DeviceContext) raises -> Float64:
    var v = _host(fx, key, ctx)
    if len(v) != 1:
        raise Error("fixture ships no " + key)
    return Float64(v[0])


# ── comparison (F64 host math + relL2 + exact-mismatch count) ────────────────
@fieldwise_init
struct Cmp(Copyable, Movable):
    var cos: Float64
    var rel: Float64
    var max_abs: Float64
    var n_diff: Int
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


def _report_info(name: String, c: Cmp, note: String):
    print(
        "   info ", name,
        "  cos=", c.cos, "  relL2=", c.rel, "  max_abs=", c.max_abs,
        "  n=", c.n, "  bitdiff=", c.n_diff, "  ", note,
    )


# ONE reporter for all three paths. `env` is the fixture's own number for the
# key on that path; `mode` decides whether the line sets the verdict.
def _report(
    mode: Int, name: String, c: Cmp, env: Float64, note: String, mut allok: Bool,
):
    if mode == MODE_FLASH:
        print(
            "   report(PRODUCTION FLASH, not gated) ", name,
            "  cos=", c.cos, "  relL2=", c.rel, "  max_abs=", c.max_abs,
            "  | env_cos(B)=", env,
            "  would-be verdict=", "PASS" if c.cos >= env else "FAIL",
        )
        print("         1-cos=", 1.0 - c.cos, "  1-env=", 1.0 - env,
              "  <- different SDPA ACCUMULATION ORDER + nondeterministic dQ;",
              " env_cos is measured with math attention on BOTH sides, so it",
              " does not bound this path. See PATH 2 in the header.")
        return
    if mode == MODE_BF16 and env < GATEABLE_MIN:
        print(
            "   info(bf16 NOT GATEABLE) ", name,
            "  cos=", c.cos, "  relL2=", c.rel, "  max_abs=", c.max_abs,
            "  | env_cos(B)=", env,
            "  <- fixture envelope < 0.999: the oracle marks this key",
            " 'use *_f32env'; its VERDICT is on the F32 path below.",
        )
        return
    var thresh = env if mode == MODE_BF16 else F32_BAR
    var ok = c.cos >= thresh
    if not ok:
        allok = False
    print(
        "  ", "PASS" if ok else "FAIL", " ", name,
        "  cos=", c.cos, "  relL2=", c.rel, "  max_abs=", c.max_abs,
        "  n=", c.n, "  | threshold=", thresh, " ", note,
    )
    print("         [scale] 1-cos=", 1.0 - c.cos, "  1-threshold=", 1.0 - thresh)


# ── adapter sets ─────────────────────────────────────────────────────────────
def _slot(
    fx: ShardedSafeTensors, name: String, in_f: Int, out_f: Int,
    lscale: Float32, ctx: DeviceContext, f32: Bool,
) raises -> Optional[LoraAdapterDevice]:
    """One adapter straight from the fixture, in the run's storage dtype. The
    F32 path upcasts the SAME bf16 bytes the oracle's F32 run upcast."""
    var a = _fv_f32(fx, "kin_lo_" + name + "_A", ctx) if f32 \
        else _fv_bf16(fx, "kin_lo_" + name + "_A", ctx)
    var b = _fv_f32(fx, "kin_lo_" + name + "_B", ctx) if f32 \
        else _fv_bf16(fx, "kin_lo_" + name + "_B", ctx)
    if a[].shape()[0] != RANK or a[].shape()[1] != in_f:
        raise Error("adapter A shape mismatch for slot " + name)
    if b[].shape()[0] != out_f or b[].shape()[1] != RANK:
        raise Error("adapter B shape mismatch for slot " + name)
    return Optional[LoraAdapterDevice](
        LoraAdapterDevice(a^, b^, RANK, in_f, out_f, lscale)
    )


def _real_lora(
    fx: ShardedSafeTensors, lscale: Float32, ctx: DeviceContext, f32: Bool = False
) raises -> Krea2BlockLora:
    return Krea2BlockLora(
        _slot(fx, "wq", FEATURES, HEADS * HEADDIM, lscale, ctx, f32),
        _slot(fx, "wk", FEATURES, KVHEADS * HEADDIM, lscale, ctx, f32),
        _slot(fx, "wv", FEATURES, KVHEADS * HEADDIM, lscale, ctx, f32),
        _slot(fx, "gate", FEATURES, FEATURES, lscale, ctx, f32),
        _slot(fx, "wo", FEATURES, FEATURES, lscale, ctx, f32),
        _slot(fx, "mlp_gate", FEATURES, MLPDIM, lscale, ctx, f32),
        _slot(fx, "mlp_up", FEATURES, MLPDIM, lscale, ctx, f32),
        _slot(fx, "mlp_down", MLPDIM, FEATURES, lscale, ctx, f32),
    )


def _block_weights(
    ck: ShardedSafeTensors, fx: ShardedSafeTensors, bi: Int, ctx: DeviceContext,
    f32: Bool = False,
) raises -> Krea2BlockWeights:
    """The 8 frozen matmuls from the CHECKPOINT, the small params from the
    fixture copies. f32=True upcasts exactly what the oracle's F32 run does
    (W[s].float() on the same bf16 bytes). mod.lin follows the run dtype: the
    oracle's `mods()` is `vec + mod_lin.to(vec.dtype)`, so bf16 for a bf16 run
    and F32 for an F32 run."""
    var p = "blocks." + String(bi) + "."
    if f32:
        return Krea2BlockWeights(
            _fv_f32(ck, p + "attn.wq.weight", ctx),
            _fv_f32(ck, p + "attn.wk.weight", ctx),
            _fv_f32(ck, p + "attn.wv.weight", ctx),
            _fv_f32(ck, p + "attn.gate.weight", ctx),
            _fv_f32(ck, p + "attn.wo.weight", ctx),
            _fv_f32(ck, p + "mlp.gate.weight", ctx),
            _fv_f32(ck, p + "mlp.up.weight", ctx),
            _fv_f32(ck, p + "mlp.down.weight", ctx),
            _fv_f32(fx, "kin_qnorm", ctx),
            _fv_f32(fx, "kin_knorm", ctx),
            _fv_f32(fx, "kin_prenorm", ctx),
            _fv_f32(fx, "kin_postnorm", ctx),
            _fv_f32(fx, "kin_mod_lin", ctx),
        )
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
# THE THREE RUNS. All three call the SAME C4 code with the SAME switches:
#   vec_cond = kin_blk_vec_cond, cond_off = krea2_omini_mod_split(lay),
#   cond_len = lay.cond_len(); upstream gradient = kin_d_out_pz unless stated.
# ══════════════════════════════════════════════════════════════════════════════
def _bwd_prefix(
    ck: ShardedSafeTensors, fx: ShardedSafeTensors, bi: Int,
    ctx: DeviceContext, f32: Bool,
) raises -> Krea2BlockGrads:
    """PATH 1 (f32=False) / PATH 3 (f32=True): the UNPADDED real prefix,
    L = real_len so `real_len == L -> sdpa_nomask` — the math SDPA the oracle
    computes and krea2_block.mojo's documented parity path."""
    var lscale = Float32(_host(fx, "meta_lora_scale", ctx)[0])
    var w = _block_weights(ck, fx, bi, ctx, f32)
    var lora = _real_lora(fx, lscale, ctx, f32)

    var x_all = _fv_f32(fx, "kin_x", ctx) if f32 else _fv_bf16(fx, "kin_x", ctx)
    var x = TArc(slice(x_all[], 1, 0, L_RL, ctx))
    var vt = _fv_f32(fx, "kin_blk_vec_t", ctx) if f32 \
        else _fv_bf16(fx, "kin_blk_vec_t", ctx)
    var vc = _fv_f32(fx, "kin_blk_vec_cond", ctx) if f32 \
        else _fv_bf16(fx, "kin_blk_vec_cond", ctx)
    var do_all = _fv_f32(fx, "kin_d_out_pz", ctx) if f32 \
        else _fv_bf16(fx, "kin_d_out_pz", ctx)
    var d_out = slice(do_all[], 1, 0, L_RL, ctx)

    var lay = Krea2OminiLayout(LTMAX, S_IMG, S_COND, LT)
    var split = krea2_omini_mod_split(lay)                 # == COND_OFF
    var pcomb = krea2_omini_pos_combined(lay, GRID, GRID, 0, 0, Float32(1.0))
    # COMBINED order is [TXT_real | IMG | COND | TXT_pad]; keep the first
    # real_len rows — exactly the rows the padded run's mask keeps.
    var pos_pre = List[Float32]()
    for i in range(L_RL * 3):
        pos_pre.append(pcomb[i])
    var pos_flat = Tensor.from_host(pos_pre^, _s1(L_RL * 3), STDtype.F32, ctx)
    var axes = List[Int]()
    axes.append(32)
    axes.append(48)
    axes.append(48)
    var rope = build_krea2_rope(pos_flat, axes, THETA, ctx, STDtype.F32)
    var rcos = rope[0].clone(ctx)
    var rsin = rope[1].clone(ctx)
    var cos_q = _tile_rope_table(rcos, L_RL, HEADS, HALF, ctx)
    var sin_q = _tile_rope_table(rsin, L_RL, HEADS, HALF, ctx)
    var cos_k = _tile_rope_table(rcos, L_RL, KVHEADS, HALF, ctx)
    var sin_k = _tile_rope_table(rsin, L_RL, KVHEADS, HALF, ctx)

    var fwd = krea2_single_stream_block_lora[L_RL, HEADS, KVHEADS, HEADDIM](
        x.copy(), vt[], w, lora, rcos, rsin, cos_q, sin_q, cos_k, sin_k, EPS, ctx,
        Optional[Int](None),                    # real_len == L -> sdpa_nomask
        Optional[TArc](vc.copy()), Optional[Int](split),
        Optional[Int](lay.cond_len()),
    )
    var lora_b = _real_lora(fx, lscale, ctx, f32)
    return krea2_single_stream_block_lora_backward[
        L_RL, HEADS, KVHEADS, HEADDIM
    ](
        d_out, vt[], w, lora_b, fwd.saved, cos_q, sin_q, cos_k, sin_k, EPS, ctx,
        Optional[Int](None), -1,
        Optional[TArc](vc.copy()), Optional[Int](split),
        Optional[Int](lay.cond_len()),
    )


def _bwd_flash(
    ck: ShardedSafeTensors, fx: ShardedSafeTensors, bi: Int,
    ctx: DeviceContext, pad_zero_dout: Bool,
) raises -> Krea2BlockGrads:
    """PATH 2: the PRODUCTION shape — bf16, L = LFULL, real_len < L so the cuDNN
    flash tail-padmask kernel runs. pad_zero_dout=False feeds the ORIGINAL
    pad-nonzero kin_d_out (section 4's pad-row probe)."""
    var lscale = Float32(_host(fx, "meta_lora_scale", ctx)[0])
    var w = _block_weights(ck, fx, bi, ctx)
    var lora = _real_lora(fx, lscale, ctx)

    var x = TArc(_fv(fx, "kin_x", ctx))
    var vt = _fv(fx, "kin_blk_vec_t", ctx)
    var vc = TArc(_fv(fx, "kin_blk_vec_cond", ctx))
    var d_out = _fv(fx, "kin_d_out_pz", ctx) if pad_zero_dout \
        else _fv(fx, "kin_d_out", ctx)

    var lay = Krea2OminiLayout(LTMAX, S_IMG, S_COND, LT)
    var split = krea2_omini_mod_split(lay)
    var pcomb = krea2_omini_pos_combined(lay, GRID, GRID, 0, 0, Float32(1.0))
    var pos_flat = Tensor.from_host(pcomb.copy(), _s1(LFULL * 3), STDtype.F32, ctx)
    var axes = List[Int]()
    axes.append(32)
    axes.append(48)
    axes.append(48)
    var rope = build_krea2_rope(pos_flat, axes, THETA, ctx, STDtype.F32)
    var rcos = rope[0].clone(ctx)
    var rsin = rope[1].clone(ctx)
    var cos_q = _tile_rope_table(rcos, LFULL, HEADS, HALF, ctx)
    var sin_q = _tile_rope_table(rsin, LFULL, HEADS, HALF, ctx)
    var cos_k = _tile_rope_table(rcos, LFULL, KVHEADS, HALF, ctx)
    var sin_k = _tile_rope_table(rsin, LFULL, KVHEADS, HALF, ctx)

    var fwd = krea2_single_stream_block_lora[LFULL, HEADS, KVHEADS, HEADDIM](
        x.copy(), vt, w, lora, rcos, rsin, cos_q, sin_q, cos_k, sin_k, EPS, ctx,
        Optional[Int](lay.real_len()), Optional[TArc](vc.copy()),
        Optional[Int](split), Optional[Int](lay.cond_len()),
    )
    var lora_b = _real_lora(fx, lscale, ctx)
    return krea2_single_stream_block_lora_backward[
        LFULL, HEADS, KVHEADS, HEADDIM
    ](
        d_out, vt, w, lora_b, fwd.saved, cos_q, sin_q, cos_k, sin_k, EPS, ctx,
        Optional[Int](lay.real_len()), -1,
        Optional[TArc](vc.copy()), Optional[Int](split),
        Optional[Int](lay.cond_len()),
    )


# ── one LoRA dA/dB pair, on any path ─────────────────────────────────────────
def _gate_pair(
    fx: ShardedSafeTensors, ctx: DeviceContext, slot: String, g: Krea2LoraGrad,
    in_f: Int, out_f: Int, mode: Int, mut allok: Bool,
) raises:
    if not g.d_a or not g.d_b:
        allok = False
        print("   FAIL ", slot, ": the routed backward returned no dA/dB")
        return
    var da = g.d_a.value().copy()
    var db = g.d_b.value().copy()
    if len(da) != RANK * in_f or len(db) != out_f * RANK:
        allok = False
        print("   FAIL ", slot, ": dA/dB element count ", len(da), "/", len(db),
              " != ", RANK * in_f, "/", out_f * RANK)
        return
    var sfx = "_f32env_pz" if mode == MODE_F32 else "_f32mod"
    var ra = _host(fx, "kref_" + slot + "_dA" + sfx, ctx)
    var rb = _host(fx, "kref_" + slot + "_dB" + sfx, ctx)
    var ea = _env(fx, slot + "_dA_f32mod", ctx)
    var eb = _env(fx, slot + "_dB_f32mod", ctx)
    var na = "(vs kref_" + slot + "_dA" + sfx + ")"
    var nb = "(vs kref_" + slot + "_dB" + sfx + ")"
    if mode == MODE_F32:
        na = na + " 1-env_f32unpad=" + String(
            1.0 - _env(fx, slot + "_dA_f32unpad", ctx))
        nb = nb + " 1-env_f32unpad=" + String(
            1.0 - _env(fx, slot + "_dB_f32unpad", ctx))
    _report(mode, slot + "_dA", _cmp(da, ra, 0, len(ra)), ea, na, allok)
    _report(mode, slot + "_dB", _cmp(db, rb, 0, len(rb)), eb, nb, allok)


def _gate_all_pairs(
    fx: ShardedSafeTensors, ctx: DeviceContext, gr: Krea2BlockGrads,
    mode: Int, mut allok: Bool,
) raises:
    _gate_pair(fx, ctx, "wq", gr.wq, FEATURES, HEADS * HEADDIM, mode, allok)
    _gate_pair(fx, ctx, "wk", gr.wk, FEATURES, KVHEADS * HEADDIM, mode, allok)
    _gate_pair(fx, ctx, "wv", gr.wv, FEATURES, KVHEADS * HEADDIM, mode, allok)
    _gate_pair(fx, ctx, "gate", gr.gate_w, FEATURES, FEATURES, mode, allok)
    _gate_pair(fx, ctx, "wo", gr.wo, FEATURES, FEATURES, mode, allok)
    _gate_pair(fx, ctx, "mlp_gate", gr.mlp_gate_w, FEATURES, MLPDIM, mode, allok)
    _gate_pair(fx, ctx, "mlp_up", gr.mlp_up_w, FEATURES, MLPDIM, mode, allok)
    _gate_pair(fx, ctx, "mlp_down", gr.mlp_down_w, MLPDIM, FEATURES, mode, allok)


# ── the three gated sections, shared by every path ───────────────────────────
def _gate_one_path(
    fx: ShardedSafeTensors, ctx: DeviceContext, gr: Krea2BlockGrads,
    mode: Int, rows: Int, mut allok: Bool,
) raises:
    var sfx = "_f32env_pz" if mode == MODE_F32 else "_f32mod"

    print("")
    print("---- 1. kref_d_x — the input grad ----")
    var dx = gr.d_x[].to_host(ctx)
    var rx = _host(fx, "kref_d_x" + sfx, ctx)
    var n = PAD_OFF * FEATURES
    var envx = _env(fx, "d_x_prefix_f32mod", ctx)
    var nx = "(vs kref_d_x" + sfx + ", rows 0:real_len)"
    if mode == MODE_F32:
        nx = nx + " 1-env_f32unpad=" + String(1.0 - _env(fx, "d_x_f32unpad", ctx))
    _report(mode, "kref_d_x [rows 0:real_len]", _cmp(dx, rx, 0, n), envx, nx, allok)
    _report_info("kref_d_x [TXT_real rows]", _cmp(dx, rx, 0, LT * FEATURES), "")
    _report_info("kref_d_x [IMG rows]",
                 _cmp(dx, rx, LT * FEATURES, S_IMG * FEATURES), "")
    _report_info("kref_d_x [COND rows]",
                 _cmp(dx, rx, COND_OFF * FEATURES, S_COND * FEATURES), "")
    if rows > PAD_OFF:
        _report_info("kref_d_x [PAD rows]",
                     _cmp(dx, rx, PAD_OFF * FEATURES, (rows - PAD_OFF) * FEATURES),
                     "<- both sides zero with the pad-zeroed d_out; never gated")

    print("")
    print("---- 2. the 8 kref_<slot>_dA/_dB pairs — COND-ROW routed ----")
    _gate_all_pairs(fx, ctx, gr, mode, allok)

    print("")
    print("---- 3. kref_d_blk_vec_t / _cond — PER-SEGMENT d_mod ----")
    if not gr.d_vec_t or not gr.d_vec_cond:
        allok = False
        print("   FAIL  the per-segment backward returned no d_vec_t/d_vec_cond")
        return
    var dvt = gr.d_vec_t.value()[].to_host(ctx)
    var dvc = gr.d_vec_cond.value()[].to_host(ctx)
    var rt = _host(fx, "kref_d_blk_vec_t" + sfx, ctx)
    var rc = _host(fx, "kref_d_blk_vec_cond" + sfx, ctx)
    var nt = "(vs kref_d_blk_vec_t" + sfx + "; rows 0:cond_off -> mods(t))"
    var nc = "(vs kref_d_blk_vec_cond" + sfx
    nc = nc + "; rows cond_off:real_len -> mods(0))"
    if mode == MODE_F32:
        nt = nt + " 1-env_f32unpad=" + String(
            1.0 - _env(fx, "d_blk_vec_t_f32unpad", ctx))
        nc = nc + " 1-env_f32unpad=" + String(
            1.0 - _env(fx, "d_blk_vec_cond_f32unpad", ctx))
    _report(mode, "kref_d_blk_vec_t", _cmp(dvt, rt, 0, len(rt)),
            _env(fx, "d_blk_vec_t_f32mod", ctx), nt, allok)
    _report(mode, "kref_d_blk_vec_cond", _cmp(dvc, rc, 0, len(rc)),
            _env(fx, "d_blk_vec_cond_f32mod", ctx), nc, allok)
    _report_info("d_vec_t vs d_vec_cond (Mojo, the two segments)",
                 _cmp2(dvt, dvc, 0, 0, len(rt)),
                 "<- MUST be far apart: two different temb chains")


# ── path-2 vs path-1 isolation: same code, same dtype, only the SDPA changed ──
def _iso_pair(
    tag: String, a: Krea2LoraGrad, b: Krea2LoraGrad, na: Int, nb: Int
) raises:
    if not a.d_a or not b.d_a or not a.d_b or not b.d_b:
        return
    var a0 = a.d_a.value().copy()
    var b0 = b.d_a.value().copy()
    var a1 = a.d_b.value().copy()
    var b1 = b.d_b.value().copy()
    _report_info(tag + "_dA", _cmp2(a0, b0, 0, 0, na), "")
    _report_info(tag + "_dB", _cmp2(a1, b1, 0, 0, nb), "")


def _isolate_flash(
    ctx: DeviceContext, a: Krea2BlockGrads, b: Krea2BlockGrads, tag: String,
) raises:
    var ax = a.d_x[].to_host(ctx)
    var bx = b.d_x[].to_host(ctx)
    _report_info(tag + " d_x [rows 0:real_len]",
                 _cmp2(ax, bx, 0, 0, PAD_OFF * FEATURES), "")
    _iso_pair(tag + " wq", a.wq, b.wq, RANK * FEATURES, FEATURES * RANK)
    _iso_pair(tag + " wo", a.wo, b.wo, RANK * FEATURES, FEATURES * RANK)
    if a.d_vec_t:
        if b.d_vec_t:
            _report_info(
                tag + " d_vec_t",
                _cmp2(a.d_vec_t.value()[].to_host(ctx),
                      b.d_vec_t.value()[].to_host(ctx), 0, 0, 6 * FEATURES), "")
    if a.d_vec_cond:
        if b.d_vec_cond:
            _report_info(
                tag + " d_vec_cond",
                _cmp2(a.d_vec_cond.value()[].to_host(ctx),
                      b.d_vec_cond.value()[].to_host(ctx), 0, 0, 6 * FEATURES), "")


def _bit_pair(
    a: Krea2LoraGrad, b: Krea2LoraGrad,
    mut npair: Int, mut nbad: Int, mut worst: Float64,
) raises:
    """Bit-compare one adapter's dA and dB across two backward calls."""
    if not a.d_a or not a.d_b or not b.d_a or not b.d_b:
        nbad += 2
        npair += 2
        return
    var a0 = a.d_a.value().copy()
    var b0 = b.d_a.value().copy()
    var a1 = a.d_b.value().copy()
    var b1 = b.d_b.value().copy()
    var c0 = _cmp2(a0, b0, 0, 0, len(a0))
    var c1 = _cmp2(a1, b1, 0, 0, len(a1))
    npair += 2
    if c0.n_diff != 0:
        nbad += 1
    if c1.n_diff != 0:
        nbad += 1
    if c0.max_abs > worst:
        worst = c0.max_abs
    if c1.max_abs > worst:
        worst = c1.max_abs


# ══════════════════════════════════════════════════════════════════════════════
# 6. DEVICE-GRAD TWIN — `krea2_single_stream_block_lora_backward_dev` took the
# SAME three C4 switches and the SAME per-segment / cond-row code; the file
# contract is that the two entry points differ ONLY in grad packaging (host
# List[Float32] vs device TArc), so on ONE saved tape they must be BIT-IDENTICAL.
# Without this section the _dev path would ship on the strength of a code
# reading. Run on the deterministic math SDPA so a bit test is meaningful.
# ══════════════════════════════════════════════════════════════════════════════
def _dev_bits(
    tag: String, h: Krea2LoraGrad, d: Krea2LoraGradT, ctx: DeviceContext,
    mut npair: Int, mut nbad: Int,
) raises:
    if not h.d_a or not h.d_b or not d.d_a or not d.d_b:
        npair += 2
        nbad += 2
        print("   (missing grad on ", tag, ")")
        return
    var ha = h.d_a.value().copy()
    var hb = h.d_b.value().copy()
    var da = d.d_a.value()[].to_host(ctx)
    var db = d.d_b.value()[].to_host(ctx)
    var c0 = _cmp2(ha, da, 0, 0, len(ha))
    var c1 = _cmp2(hb, db, 0, 0, len(hb))
    npair += 2
    if c0.n_diff != 0:
        nbad += 1
    if c1.n_diff != 0:
        nbad += 1


def _gate_dev_twin(
    ck: ShardedSafeTensors, fx: ShardedSafeTensors, bi: Int,
    ctx: DeviceContext, mut allok: Bool,
) raises:
    print("")
    print("---- 6. DEVICE-GRAD TWIN — _backward_dev vs _backward on ONE tape ----")
    var lscale = Float32(_host(fx, "meta_lora_scale", ctx)[0])
    var w = _block_weights(ck, fx, bi, ctx)
    var x_all = _fv_bf16(fx, "kin_x", ctx)
    var x = TArc(slice(x_all[], 1, 0, L_RL, ctx))
    var vt = _fv_bf16(fx, "kin_blk_vec_t", ctx)
    var vc = _fv_bf16(fx, "kin_blk_vec_cond", ctx)
    var do_all = _fv_bf16(fx, "kin_d_out_pz", ctx)
    var d_out = slice(do_all[], 1, 0, L_RL, ctx)

    var lay = Krea2OminiLayout(LTMAX, S_IMG, S_COND, LT)
    var split = krea2_omini_mod_split(lay)
    var pcomb = krea2_omini_pos_combined(lay, GRID, GRID, 0, 0, Float32(1.0))
    var pos_pre = List[Float32]()
    for i in range(L_RL * 3):
        pos_pre.append(pcomb[i])
    var pos_flat = Tensor.from_host(pos_pre^, _s1(L_RL * 3), STDtype.F32, ctx)
    var axes = List[Int]()
    axes.append(32)
    axes.append(48)
    axes.append(48)
    var rope = build_krea2_rope(pos_flat, axes, THETA, ctx, STDtype.F32)
    var rcos = rope[0].clone(ctx)
    var rsin = rope[1].clone(ctx)
    var cos_q = _tile_rope_table(rcos, L_RL, HEADS, HALF, ctx)
    var sin_q = _tile_rope_table(rsin, L_RL, HEADS, HALF, ctx)
    var cos_k = _tile_rope_table(rcos, L_RL, KVHEADS, HALF, ctx)
    var sin_k = _tile_rope_table(rsin, L_RL, KVHEADS, HALF, ctx)

    var fwd = krea2_single_stream_block_lora[L_RL, HEADS, KVHEADS, HEADDIM](
        x.copy(), vt[], w, _real_lora(fx, lscale, ctx), rcos, rsin,
        cos_q, sin_q, cos_k, sin_k, EPS, ctx, Optional[Int](None),
        Optional[TArc](vc.copy()), Optional[Int](split),
        Optional[Int](lay.cond_len()),
    )
    var gh = krea2_single_stream_block_lora_backward[
        L_RL, HEADS, KVHEADS, HEADDIM
    ](
        d_out, vt[], w, _real_lora(fx, lscale, ctx), fwd.saved,
        cos_q, sin_q, cos_k, sin_k, EPS, ctx, Optional[Int](None), -1,
        Optional[TArc](vc.copy()), Optional[Int](split),
        Optional[Int](lay.cond_len()),
    )
    var gd = krea2_single_stream_block_lora_backward_dev[
        L_RL, HEADS, KVHEADS, HEADDIM
    ](
        d_out, vt[], w, _real_lora(fx, lscale, ctx), fwd.saved,
        cos_q, sin_q, cos_k, sin_k, EPS, ctx, Optional[Int](None),
        Optional[TArc](vc.copy()), Optional[Int](split),
        Optional[Int](lay.cond_len()),
    )
    var hx = gh.d_x[].to_host(ctx)
    var dx = gd.d_x[].to_host(ctx)
    var cx = _cmp(hx, dx, 0, len(hx))
    var npair = 0
    var nbad = 0
    _dev_bits("wq", gh.wq, gd.wq, ctx, npair, nbad)
    _dev_bits("wk", gh.wk, gd.wk, ctx, npair, nbad)
    _dev_bits("wv", gh.wv, gd.wv, ctx, npair, nbad)
    _dev_bits("gate", gh.gate_w, gd.gate_w, ctx, npair, nbad)
    _dev_bits("wo", gh.wo, gd.wo, ctx, npair, nbad)
    _dev_bits("mlp_gate", gh.mlp_gate_w, gd.mlp_gate_w, ctx, npair, nbad)
    _dev_bits("mlp_up", gh.mlp_up_w, gd.mlp_up_w, ctx, npair, nbad)
    _dev_bits("mlp_down", gh.mlp_down_w, gd.mlp_down_w, ctx, npair, nbad)
    var cvt = 0
    var cvc = 0
    var have = False
    if gh.d_vec_t:
        if gd.d_vec_t:
            if gh.d_vec_cond:
                if gd.d_vec_cond:
                    have = True
                    cvt = _cmp2(gh.d_vec_t.value()[].to_host(ctx),
                                gd.d_vec_t.value()[].to_host(ctx),
                                0, 0, 6 * FEATURES).n_diff
                    cvc = _cmp2(gh.d_vec_cond.value()[].to_host(ctx),
                                gd.d_vec_cond.value()[].to_host(ctx),
                                0, 0, 6 * FEATURES).n_diff
    var ok = (cx.n_diff == 0) and (nbad == 0) and (npair == 16) and have \
        and (cvt == 0) and (cvc == 0)
    if not ok:
        allok = False
    print(
        "  ", "PASS" if ok else "FAIL",
        " _backward_dev == _backward, BIT-for-BIT:  d_x bitdiff=", cx.n_diff,
        "/", cx.n, " max_abs=", cx.max_abs,
        " | dA/dB ", npair - nbad, "/", npair, " tensors bit-equal",
        " | d_vec_t bitdiff=", cvt, "  d_vec_cond bitdiff=", cvc,
        " (both device paths present=", have, ")",
    )


# ══════════════════════════════════════════════════════════════════════════════
# 5. CONDLEN=0 BACKWARD BIT-EQUALITY — the live-trainer regression guard.
# ONE forward feeds BOTH backwards, so the comparison isolates the BACKWARD:
#   run A: the pre-C4 call form — no vec_cond, no cond_off, no cond_len.
#   run B: the post-C4 call form of a CONDLEN=0 build — vec_cond IS supplied,
#          cond_off = krea2_omini_mod_split(lay0) which is -1 for s_cond == 0,
#          cond_len = 0. Both guards must fall through to the pre-C4 kernels.
# d_x AND all 8 dA/dB must be BIT-IDENTICAL and both runs must report NO d_vec
# pair (the uniform path emits no modulation grad, exactly as before C4).
#
# THE TEST RUNS ON THE MATH SDPA (real_len=None), and that is not a convenience:
# on the cuDNN flash path a bit test is IMPOSSIBLE for reasons that predate C4 —
# krea2_block.mojo documents "FLASH dQ is NONDETERMINISTIC run-to-run (cuDNN
# atomics on the dQ accumulation)". The flash pair is therefore ALSO run, next
# to a CONTROL that calls the IDENTICAL form twice, so the reader can see the
# A-vs-B residual sitting inside the nondeterminism the control measures.
# ══════════════════════════════════════════════════════════════════════════════
def _gate_condlen0(
    ck: ShardedSafeTensors, fx: ShardedSafeTensors, bi: Int,
    ctx: DeviceContext, mut allok: Bool,
) raises:
    print("")
    print("---- 5. CONDLEN=0 BACKWARD BIT-EQUALITY (adapters ATTACHED) ----")
    var lscale = Float32(_host(fx, "meta_lora_scale", ctx)[0])
    var lay0 = Krea2OminiLayout(LTMAX, S_IMG, 0, LT)
    lay0.check_flash_prefix()
    var split0 = krea2_omini_mod_split(lay0)
    print("   layout s_cond=0: lfull=", lay0.lfull(), " real_len=", lay0.real_len(),
          " mod_split=", split0, " cond_len=", lay0.cond_len(),
          " (-1 / 0 => both C4 guards fall through)")
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

    var xf = _fv(fx, "kin_x", ctx)
    var xh = slice(xf, 1, 0, COND_OFF, ctx)
    var xt = slice(xf, 1, PAD_OFF, LTMAX - LT, ctx)
    var x_nc = TArc(concat(1, ctx, xh, xt))                # [1, L_NC, F] bf16
    var df = _fv(fx, "kin_d_out_pz", ctx)
    var dh = slice(df, 1, 0, COND_OFF, ctx)
    var dt2 = slice(df, 1, PAD_OFF, LTMAX - LT, ctx)
    var d_nc = concat(1, ctx, dh, dt2)

    var w = _block_weights(ck, fx, bi, ctx)
    var vt = _fv(fx, "kin_blk_vec_t", ctx)
    var vc = TArc(_fv(fx, "kin_blk_vec_cond", ctx))

    # ── 5a. DETERMINISTIC math SDPA — the GATED bit-equality ────────────────
    var fwd = krea2_single_stream_block_lora[L_NC, HEADS, KVHEADS, HEADDIM](
        x_nc.copy(), vt, w, _real_lora(fx, lscale, ctx), rcos, rsin,
        cos_q, sin_q, cos_k, sin_k, EPS, ctx,
    )
    var ga = krea2_single_stream_block_lora_backward[
        L_NC, HEADS, KVHEADS, HEADDIM
    ](
        d_nc, vt, w, _real_lora(fx, lscale, ctx), fwd.saved,
        cos_q, sin_q, cos_k, sin_k, EPS, ctx,
    )
    var gb = krea2_single_stream_block_lora_backward[
        L_NC, HEADS, KVHEADS, HEADDIM
    ](
        d_nc, vt, w, _real_lora(fx, lscale, ctx), fwd.saved,
        cos_q, sin_q, cos_k, sin_k, EPS, ctx, Optional[Int](None), -1,
        Optional[TArc](vc.copy()), Optional[Int](split0),
        Optional[Int](lay0.cond_len()),
    )
    var a_dx = ga.d_x[].to_host(ctx)
    var b_dx = gb.d_x[].to_host(ctx)
    var c_dx = _cmp(a_dx, b_dx, 0, len(a_dx))
    var ok = c_dx.n_diff == 0
    var nz = 0
    for i in range(len(a_dx)):
        if a_dx[i] != 0.0:
            nz += 1
    var npair = 0
    var nbad = 0
    var worst = Float64(0.0)
    _bit_pair(ga.wq, gb.wq, npair, nbad, worst)
    _bit_pair(ga.wk, gb.wk, npair, nbad, worst)
    _bit_pair(ga.wv, gb.wv, npair, nbad, worst)
    _bit_pair(ga.gate_w, gb.gate_w, npair, nbad, worst)
    _bit_pair(ga.wo, gb.wo, npair, nbad, worst)
    _bit_pair(ga.mlp_gate_w, gb.mlp_gate_w, npair, nbad, worst)
    _bit_pair(ga.mlp_up_w, gb.mlp_up_w, npair, nbad, worst)
    _bit_pair(ga.mlp_down_w, gb.mlp_down_w, npair, nbad, worst)
    if nbad != 0 or npair != 16:
        ok = False
    var no_dvec = (not ga.d_vec_t) and (not ga.d_vec_cond) \
        and (not gb.d_vec_t) and (not gb.d_vec_cond)
    if not no_dvec:
        ok = False
    if not ok:
        allok = False
    print(
        "  ", "PASS" if ok else "FAIL",
        " 5a DETERMINISTIC (sdpa_nomask) condlen=0 BACKWARD BIT-EQUAL:",
        " d_x bitdiff=", c_dx.n_diff, "/", c_dx.n, " max_abs=", c_dx.max_abs,
        " | dA/dB ", npair - nbad, "/", npair, " tensors bit-equal, worst max_abs=",
        worst, " | d_vec pair absent on both runs=", no_dvec,
    )
    print("   (non-degeneracy: ", nz, "/", len(a_dx),
          " nonzero elements in the condlen=0 d_x)")

    # ── 5b. the FLASH pair + its own nondeterminism CONTROL ─────────────────
    var fwd_f = krea2_single_stream_block_lora[L_NC, HEADS, KVHEADS, HEADDIM](
        x_nc.copy(), vt, w, _real_lora(fx, lscale, ctx), rcos, rsin,
        cos_q, sin_q, cos_k, sin_k, EPS, ctx, Optional[Int](lay0.real_len()),
    )
    var fa = krea2_single_stream_block_lora_backward[
        L_NC, HEADS, KVHEADS, HEADDIM
    ](
        d_nc, vt, w, _real_lora(fx, lscale, ctx), fwd_f.saved,
        cos_q, sin_q, cos_k, sin_k, EPS, ctx, Optional[Int](lay0.real_len()),
    )
    var fa2 = krea2_single_stream_block_lora_backward[
        L_NC, HEADS, KVHEADS, HEADDIM
    ](
        d_nc, vt, w, _real_lora(fx, lscale, ctx), fwd_f.saved,
        cos_q, sin_q, cos_k, sin_k, EPS, ctx, Optional[Int](lay0.real_len()),
    )
    var fb = krea2_single_stream_block_lora_backward[
        L_NC, HEADS, KVHEADS, HEADDIM
    ](
        d_nc, vt, w, _real_lora(fx, lscale, ctx), fwd_f.saved,
        cos_q, sin_q, cos_k, sin_k, EPS, ctx, Optional[Int](lay0.real_len()), -1,
        Optional[TArc](vc.copy()), Optional[Int](split0),
        Optional[Int](lay0.cond_len()),
    )
    var f1 = fa.d_x[].to_host(ctx)
    var f2 = fa2.d_x[].to_host(ctx)
    var f3 = fb.d_x[].to_host(ctx)
    var ctl = _cmp(f1, f2, 0, len(f1))
    var ab = _cmp(f1, f3, 0, len(f1))
    print("   info 5b FLASH path, d_x:  CONTROL (identical call twice) bitdiff=",
          ctl.n_diff, "/", ctl.n, " max_abs=", ctl.max_abs,
          "  |  pre-C4 vs C4-form bitdiff=", ab.n_diff, "/", ab.n,
          " max_abs=", ab.max_abs)
    print("        <- NOT gated: cuDNN flash dQ is nondeterministic run-to-run;")
    print("           the CONTROL is the floor any flash comparison sits on.")


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

    print("")
    print("THRESHOLDS (from the fixture, not invented).  env(B)=env_cos_*_f32mod")
    print("  key                 env(B) = the bf16 ceiling     1-env(f32unpad)")
    print("  d_x                ", _env(fx, "d_x_prefix_f32mod", ctx), "   ",
          1.0 - _env(fx, "d_x_f32unpad", ctx))
    print("  d_blk_vec_t        ", _env(fx, "d_blk_vec_t_f32mod", ctx), "   ",
          1.0 - _env(fx, "d_blk_vec_t_f32unpad", ctx))
    print("  d_blk_vec_cond     ", _env(fx, "d_blk_vec_cond_f32mod", ctx), "   ",
          1.0 - _env(fx, "d_blk_vec_cond_f32unpad", ctx))
    print("  wq_dA / wq_dB      ", _env(fx, "wq_dA_f32mod", ctx), " ",
          _env(fx, "wq_dB_f32mod", ctx))
    print("  wk_dA / wk_dB      ", _env(fx, "wk_dA_f32mod", ctx), " ",
          _env(fx, "wk_dB_f32mod", ctx))
    print("  wv_dA / wv_dB      ", _env(fx, "wv_dA_f32mod", ctx), " ",
          _env(fx, "wv_dB_f32mod", ctx))
    print("  gate_dA / gate_dB  ", _env(fx, "gate_dA_f32mod", ctx), " ",
          _env(fx, "gate_dB_f32mod", ctx))
    print("  wo_dA / wo_dB      ", _env(fx, "wo_dA_f32mod", ctx), " ",
          _env(fx, "wo_dB_f32mod", ctx))
    print("  mlp_gate_dA / _dB  ", _env(fx, "mlp_gate_dA_f32mod", ctx), " ",
          _env(fx, "mlp_gate_dB_f32mod", ctx))
    print("  mlp_up_dA / _dB    ", _env(fx, "mlp_up_dA_f32mod", ctx), " ",
          _env(fx, "mlp_up_dB_f32mod", ctx))
    print("  mlp_down_dA / _dB  ", _env(fx, "mlp_down_dA_f32mod", ctx), " ",
          _env(fx, "mlp_down_dB_f32mod", ctx))
    print("  Keys below", GATEABLE_MIN, "are INFO on the bf16 path (the oracle's")
    print("  own 'use *_f32env' rule) and carry their verdict on the F32 path.")
    print("")
    print("PAD-ROW EVIDENCE shipped by the oracle for this block:")
    print("   cos(d_blk_vec_cond with pad-tail grad, without) =",
          _scalar(fx, "env_cos_d_blk_vec_cond_padcontrib", ctx))
    print("   cos(d_blk_vec_t   with pad-tail grad, without) =",
          _scalar(fx, "env_cos_d_blk_vec_t_padcontrib", ctx),
          "  bit_equal=", _scalar(fx, "env_bit_d_blk_vec_t_padzero_equal", ctx))
    print("   d_x[prefix] bit_equal with vs without          =",
          _scalar(fx, "env_bit_d_x_prefix_padzero_equal", ctx))
    print("   wq_dA bit_equal with vs without                =",
          _scalar(fx, "env_bit_wq_dA_padzero_equal", ctx))
    print("   => pad rows CANNOT reach the real rows, but they DO corrupt the")
    print("      t=0 chunk grads. C4 excludes them; section 4 proves it in Mojo.")

    var allok = True

    # ── PATH 1 — bf16, math SDPA, unpadded prefix: PRIMARY (env rule) ───────
    print("")
    print("==== PATH 1 — bf16, L=", L_RL, " = real_len, sdpa_nomask ====")
    print("     schedule B AND the oracle's own attention math on both sides.")
    var g1 = _bwd_prefix(ck, fx, bi, ctx, False)
    _gate_one_path(fx, ctx, g1, MODE_BF16, L_RL, allok)

    # ── PATH 3 — F32, math SDPA, unpadded prefix: PRIMARY at F32_BAR ────────
    print("")
    print("==== PATH 3 — F32, L=", L_RL, " = real_len, sdpa_nomask ====")
    print("     krea2_block.mojo's documented F32 parity gate guard.")
    print("     Reference kref_*_f32env_pz, threshold", F32_BAR, ".")
    var g3 = _bwd_prefix(ck, fx, bi, ctx, True)
    _gate_one_path(fx, ctx, g3, MODE_F32, L_RL, allok)

    # ── PATH 2 — the PRODUCTION flash shape: REPORTED, not gated ────────────
    print("")
    print("==== PATH 2 — bf16, L=", LFULL, " real_len=", PAD_OFF,
          " (cuDNN flash tail-padmask = the PRODUCTION SDPA) ====")
    print("     Reported in full with its would-be verdict; see PATH 2 in the")
    print("     header for why env_cos_* does not bound this kernel.")
    var g2 = _bwd_flash(ck, fx, bi, ctx, True)
    var ignore = True
    _gate_one_path(fx, ctx, g2, MODE_FLASH, LFULL, ignore)

    print("")
    print("---- 2b. FLASH-vs-MATH ISOLATION (same C4 code, same bf16 storage,")
    print("     same inputs; ONLY the SDPA kernel differs — path 2 vs path 1) ----")
    _isolate_flash(ctx, g2, g1, "flash-vs-math")

    # ── 4. PAD-ROW EXCLUSION ────────────────────────────────────────────────
    print("")
    print("---- 4. PAD-ROW EXCLUSION — which reference does the PRODUCTION")
    print("     backward match? Re-run with the ORIGINAL pad-NONZERO kin_d_out.")
    print("     A bit-equality test is impossible here (flash dQ is")
    print("     nondeterministic), so this is a THRESHOLD-FREE discrimination:")
    print("     the fixture ships BOTH the pad-EXCLUDED reference and its")
    print("     pad-INCLUDED twin, and a correct C4 must be closer to the first.")
    var g4 = _bwd_flash(ck, fx, bi, ctx, False)
    if not g4.d_vec_cond or not g2.d_vec_cond:
        allok = False
        print("   FAIL  pad-row probe returned no d_vec pair")
    else:
        var dvc4 = g4.d_vec_cond.value()[].to_host(ctx)
        var ex = _host(fx, "kref_d_blk_vec_cond_f32mod", ctx)
        var inc = _host(fx, "kref_d_blk_vec_cond_padincl_f32mod", ctx)
        var c_ex = _cmp(dvc4, ex, 0, len(ex))
        var c_in = _cmp(dvc4, inc, 0, len(inc))
        var ok4 = c_ex.cos > c_in.cos
        if not ok4:
            allok = False
        print(
            "  ", "PASS" if ok4 else "FAIL",
            " d_vec_cond (pad-NONZERO d_out) is closer to the pad-EXCLUDED ref:",
            "  cos(excluded)=", c_ex.cos, "  cos(INCLUDED)=", c_in.cos,
            "  (excluded MUST win)",
        )
        print("         1-cos(excluded)=", 1.0 - c_ex.cos,
              "  1-cos(included)=", 1.0 - c_in.cos,
              "  <- the fixture's own reference separation for this block is",
              " 1-cos=", 1.0 - _scalar(fx, "env_cos_d_blk_vec_cond_padcontrib", ctx))
        var dvc2 = g2.d_vec_cond.value()[].to_host(ctx)
        _report_info("d_vec_cond: pad-NONZERO vs pad-ZEROED d_out (same Mojo path)",
                     _cmp2(dvc4, dvc2, 0, 0, len(dvc2)),
                     "<- bitdiff is the flash nondeterminism floor, not a pad"
                     " contribution; compare it to the separation above")

    _gate_dev_twin(ck, fx, bi, ctx, allok)
    _gate_condlen0(ck, fx, bi, ctx, allok)

    print("")
    if allok:
        print("BLOCK", bi, "VERDICT: PASS")
    else:
        print("BLOCK", bi, "VERDICT: FAIL")
    return allok


def main() raises:
    var ctx = DeviceContext()
    print("==== krea2 OminiControl EDIT — C4 BACKWARD parity gate ====")
    print("layout [TXT_real(", LT, ") | IMG(", S_IMG, ") | COND(", S_COND,
          ") | TXT_pad(", LTMAX - LT, ")]  LFULL=", LFULL,
          " real_len=", PAD_OFF, " cond_off=", COND_OFF)
    print("d_mod(t)    <- rows [0,", COND_OFF, ")   (TXT_real + IMG)")
    print("d_mod(t=0)  <- rows [", COND_OFF, ",", PAD_OFF,
          ")   (COND; the TXT_pad tail is EXCLUDED — see the header)")
    print("dA/dB       <- the COND rows only; dX stays FULL sequence.")
    print("checkpoint:", CKPT)
    var ck = ShardedSafeTensors.open(String(CKPT))

    var ok0 = _run_block(0, ck, ctx)
    var ok27 = _run_block(27, ck, ctx)

    print("")
    print("==============================================================")
    if ok0 and ok27:
        print("C4 BACKWARD GATE: PASS on BOTH block00 and block27")
    else:
        print("C4 BACKWARD GATE: FAIL — block00 ",
              "PASS" if ok0 else "FAIL", ", block27 ",
              "PASS" if ok27 else "FAIL")
