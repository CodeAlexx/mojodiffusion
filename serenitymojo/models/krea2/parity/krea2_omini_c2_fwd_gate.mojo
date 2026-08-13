# serenitymojo/models/krea2/parity/krea2_omini_c2_fwd_gate.mojo
#
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  C2 FORWARD PARITY GATE — krea2 OminiControl EDIT row layout             ║
# ║  [ TXT_real(lt) | IMG | COND | TXT_pad ]  with per-segment modulation.   ║
# ║  Gates the MOJO forward against the CUDA-generated fixtures written by   ║
# ║  scripts/krea2_omini_torch_oracle.py. CPU torch is NEVER an oracle here: ║
# ║  every reference number comes out of that fixture, which the oracle      ║
# ║  produced on the RTX 5080 with the real Krea-2-Raw bf16 weights.         ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# WHAT IS GATED (in the order the C2 brief requires; a FAIL stops nothing but is
# reported per key, and the verdict is the AND over all keys):
#   1. kin_pos_src / kin_pos / kin_cos / kin_sin
#        Position + RoPE-table math over the new layout. Pure integer/table math
#        with no weights => BIT-EXACT expected. Reported as an exact-mismatch
#        COUNT as well as cos/relL2/max_abs.
#        Mojo side: training/krea2_omini_layout.krea2_omini_pos_src /
#        krea2_omini_pos_combined -> models/dit/krea2_dit.build_krea2_rope.
#   2. kref_xm_f32mod   (schedule B; kref_xm shown as info)
#        Per-segment modulation: (1+prescale)*prenorm(x)+preshift with the
#        mods(t) chunks on rows [0,cond_off) and the mods(t=0) chunks on rows
#        [cond_off, LFULL). ONE elementwise op away from the fixture inputs, so
#        a mismatch here is unambiguous. Mojo side: krea2_block._modulate_seg2.
#   3. kref_attn_raw_nolora_f32mod  (schedule B; schedule A shown as info)
#        Masked attention (real-length contiguous prefix, pad tail masked) with
#        RoPE + GQA over the new layout, ADAPTER DISABLED.
#        HISTORY — do not lose this, it cost a debugging cycle: the ORIGINAL C2
#        fixture shipped only `kref_attn_raw`, and that key is the LoRA-**ON**
#        attention output (the oracle dumps seams["attn_raw"] from the
#        adapters-enabled forward). A LoRA-off forward cannot match it and no
#        threshold change can fix that, so C2 had to rebuild the reference's own
#        conditions with a gate-local cond-row LoRA helper. The oracle now ships
#        `kref_attn_raw_nolora` — the LoRA-OFF twin — with its own
#        `env_cos_attn_raw_nolora`, so this gate now compares the plain LoRA-off
#        Mojo attention against a reference it CAN match. The LoRA-ON
#        `kref_attn_raw` is gated by the C3 gate against the real cond-row-routed
#        block forward (parity/krea2_omini_c3_lora_gate.mojo).
#   4. kref_out_nolora_f32mod / kref_out_nolora_img_f32mod (schedule B)
#        Whole block forward, LoRA DISABLED (LoRA routing is C3, backward is C4;
#        neither is touched here).
#   5. CONDLEN=0 REGRESSION (mandatory): the existing krea2 training path must be
#        BIT-EQUAL when there is no condition segment. Proven at real dims with
#        the real weights, through the real trainer-facing API
#        (krea2_omini_mod_split(lay) == -1 for s_cond == 0).
#
# THRESHOLDS — TAKEN FROM THE FIXTURE, NEVER INVENTED
#   The fixture ships env_cos_<key> = the MEASURED bf16-vs-f32 cosine of the
#   ORACLE'S OWN forward on the same CUDA device with the same real weights.
#   That is the ceiling for any bf16 implementation OF THAT ROUNDING SCHEDULE.
#   As of the C3 oracle regeneration EVERY key gated here ships its OWN
#   envelope — nothing is borrowed.
#
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  WHICH ROUNDING SCHEDULE THIS GATE TARGETS — the fix that made these      ║
# ║  comparisons apples-to-apples. READ BEFORE CHANGING A THRESHOLD.          ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#   The oracle ships each gated seam under TWO bf16 rounding schedules. Same
#   seed, same real krea2 weights, same real cached inputs, same layout, mask,
#   RoPE, LoRA routing, F32 matmul accumulation, same CUDA device. The ONLY
#   difference is WHERE bf16 rounding happens inside modulate/residual_gate:
#     SCHEDULE A  kref_<key>         (1+scale)*h+shift done in bf16 -> THREE
#                 env_cos_<key>      roundings per modulate, TWO per res-gate.
#                                    Plain torch semantics; this is the schedule
#                                    the mmdit fidelity check validates.
#     SCHEDULE B  kref_<key>_f32mod  the same algebra in F32, rounded ONCE at
#                 env_cos_<key>_f32mod store.
#   serenitymojo/ops/elementwise.mojo `_modulate_kernel_bf16` and
#   `_resgate_kernel_bf16` upcast x/scale/shift to F32, compute in F32, and
#   .cast[bfloat16]() exactly ONCE at the store. THE MOJO FORWARD IS SCHEDULE B.
#   So the PRIMARY gate here is Mojo vs kref_<key>_f32mod at env_cos_<key>_f32mod.
#
#   WHY (this is a criterion bug that was actually shipped, not a hypothetical):
#   env_cos_<key> measures bf16-storage-vs-f32-storage UNDER SCHEDULE A's OWN
#   ROUNDING. It therefore does NOT bound a correct implementation that uses
#   schedule B. The fixture proves it at block 27:
#       env_cos_out_img          1-cos = 1.043e-05   (the old threshold)
#       env_cos_out_img_modround 1-cos = 1.097e-05   (A-vs-B distance ALONE)
#   i.e. merely choosing the F32-math modulate puts a provably-correct
#   implementation outside that key's envelope. Four keys failed at block 27 by
#   1e-7..9e-7 for exactly that reason (kref_out_img, kref_attn_raw_nolora
#   [flash], kref_out_nolora, kref_out_nolora_img).
#
#   THIS IS NOT A TOLERANCE WIDENING. At block 27 the schedule-B envelopes are
#   TIGHTER than the schedule-A ones they replace, e.g. out_nolora_img
#   1-env: 1.049e-05 (A) -> 7.93e-06 (B); xm 6.74e-06 (A) -> 4.29e-06 (B).
#   Schedule B rounds less, so it sits closer to F32 and its ceiling is lower.
#   The gate got STRICTER and simultaneously correct.
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
#   It never sets the verdict. Nothing is asserted TIGHTER than the key's own
#   shipped envelope for its own schedule, and nothing is ever loosened.
#
#   dtype: the forward runs the PRODUCTION bf16 path (env_cos_out >= 0.99995 at
#   both blocks says bf16 is comfortably gateable for the FORWARD). The F32
#   block path is NOT needed here; it is the documented fallback for the
#   early-block BACKWARD (env_cos_d_x = 0.494 at block 0), which is C4's problem.
#
# MASK / PAD-ROW SEMANTICS (why the primary comparison is the real prefix)
#   The oracle masks pad rows BOTH as key columns and as query rows (it zeroes
#   the pad query rows after the softmax). The Mojo production path is the cuDNN
#   flash tail-padmask, which masks only the key columns and leaves the pad
#   QUERY rows as masked-out garbage — the trainer drops them (pad d_out is
#   zero, and the loss reads only the IMG rows). The two are equivalent exactly
#   on rows [0, real_len). The oracle itself compares on `[:, :REAL_LEN]`
#   (krea2_omini_torch_oracle.py:1103-1105). So the GATE is the real prefix; the
#   full-tensor number (pad rows included) is printed alongside for honesty and
#   is NOT gated.
#
# WEIGHTS / PARAMS — where each number comes from
#   * the 8 base matmuls: read from the SAME checkpoint the oracle read
#     (/home/alex/.serenity/models/checkpoints/krea2-raw.safetensors), bf16,
#     matching the oracle's `.to(dtype=bfloat16)` (oracle load_block()).
#   * qnorm/knorm/prenorm/postnorm: the fixture's F32 copies == the oracle's
#     `.float()` params.
#   * mod.lin: the fixture's F32 copy rounded to BF16 == the oracle's
#     `P["mod_lin"].to(vec.dtype)` with vec bf16 (oracle omini_block_forward
#     mods()). This is also what the production loader does (_stream_wb).
#   * activations: the fixture's kin_x / kin_blk_vec_t / kin_blk_vec_cond, which
#     came from the REAL krea2 training cache through the REAL krea2 preamble.
#
# RUN (never chained after a mojo build with &&):
#   cd /home/alex/mojodiffusion
#   rm -f serenitymojo.mojopkg
#   pixi run mojo run -I . \
#       -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa \
#       serenitymojo/models/krea2/parity/krea2_omini_c2_fwd_gate.mojo

from max.gpu.host import DeviceContext
from std.collections import List, Optional
from std.math import sqrt
from std.memory import ArcPointer
from serenitymojo.tensor import Tensor
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.tensor_algebra import slice, concat
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.attention import sdpa_qwen_keymask
from serenitymojo.models.klein.lora_block import LoraAdapterDevice
from serenitymojo.models.krea2.krea2_block import (
    Krea2BlockWeights, Krea2BlockLora, krea2_single_stream_block_lora,
)
from serenitymojo.models.dit.krea2_dit import (
    build_krea2_rope, _tile_rope_table,
)
from serenitymojo.models.krea2.krea2_cache_reader import krea2_build_pos
from serenitymojo.training.krea2_omini_layout import (
    Krea2OminiLayout, krea2_omini_mod_split,
    krea2_omini_pos_src, krea2_omini_pos_combined,
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
comptime REAL_NC = LT + S_IMG              # 1214
comptime EPS = Float32(1e-5)
comptime THETA = Float32(1000.0)
comptime LH = GRID * 2                     # 64 (latent rows)
comptime LW = GRID * 2                     # 64


# ── tiny shape helpers ───────────────────────────────────────────────────────
def _s1(a: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a)
    return s^


def _s2(a: Int, b: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a)
    s.append(b)
    return s^


def _s3(a: Int, b: Int, c: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a)
    s.append(b)
    s.append(c)
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
    """Read an I32 [1] meta scalar (little-endian) straight from the mmap."""
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


# ── comparison (F64 host math, like ParityHarness, plus relL2) ───────────────
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
    """Compare a[off_a : off_a+n] against b[off_b : off_b+n] in F64."""
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


# ── LoRA-off adapter set (C2 is LoRA-DISABLED; routing is C3) ────────────────
def _no_lora() -> Krea2BlockLora:
    return Krea2BlockLora(
        Optional[LoraAdapterDevice](None), Optional[LoraAdapterDevice](None),
        Optional[LoraAdapterDevice](None), Optional[LoraAdapterDevice](None),
        Optional[LoraAdapterDevice](None), Optional[LoraAdapterDevice](None),
        Optional[LoraAdapterDevice](None), Optional[LoraAdapterDevice](None),
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
        # BF16 == the oracle's P["mod_lin"].to(vec.dtype) (vec is bf16), and ==
        # the production loader (_stream_wb). _mod6's `add` requires the same
        # dtype as the [1,6F] bf16 modulation vector anyway.
        _fv_bf16(fx, "kin_mod_lin", ctx),
    )


# ── per-key envelope reader ──────────────────────────────────────────────────
# THE ONLY source of a threshold in this file. Fails loud if the fixture does
# not ship the key's envelope — a missing envelope must never silently become a
# borrowed one again.
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


# ══════════════════════════════════════════════════════════════════════════════
# GATE 1 — positions + RoPE tables over the new layout
# ══════════════════════════════════════════════════════════════════════════════
def _gate_positions(
    fx: ShardedSafeTensors, ctx: DeviceContext, mut allok: Bool
) raises -> Tuple[Tensor, Tensor]:
    print("")
    print("---- 1. positions + RoPE tables (layout/table math; BIT-EXACT expected) ----")
    var lay = Krea2OminiLayout(LTMAX, S_IMG, S_COND, LT)
    lay.check_flash_prefix()

    var psrc = krea2_omini_pos_src(lay, GRID, GRID, 0, 0, Float32(1.0))
    var pcomb = krea2_omini_pos_combined(lay, GRID, GRID, 0, 0, Float32(1.0))

    var ref_src = _host(fx, "kin_pos_src", ctx)
    var ref_comb = _host(fx, "kin_pos", ctx)
    _report("kin_pos_src", _cmp(psrc, ref_src, 0, LFULL * 3), 1.0,
            "(exact; bitdiff MUST be 0)", allok)
    _report("kin_pos    ", _cmp(pcomb, ref_comb, 0, LFULL * 3), 1.0,
            "(exact; bitdiff MUST be 0)", allok)

    var pos_flat = Tensor.from_host(pcomb.copy(), _s1(LFULL * 3), STDtype.F32, ctx)
    var axes = List[Int]()
    axes.append(32)
    axes.append(48)
    axes.append(48)
    var rope = build_krea2_rope(pos_flat, axes, THETA, ctx, STDtype.F32)
    var rcos = rope[0].clone(ctx)
    var rsin = rope[1].clone(ctx)

    var cos_h = rcos.to_host(ctx)
    var sin_h = rsin.to_host(ctx)
    var ref_cos = _host(fx, "kin_cos", ctx)
    var ref_sin = _host(fx, "kin_sin", ctx)
    _report("kin_cos    ", _cmp(cos_h, ref_cos, 0, LFULL * HALF), 0.999999999,
            "(F64-reduced table math; bitdiff should be 0)", allok)
    _report("kin_sin    ", _cmp(sin_h, ref_sin, 0, LFULL * HALF), 0.999999999,
            "(F64-reduced table math; bitdiff should be 0)", allok)
    return (rcos^, rsin^)


# ══════════════════════════════════════════════════════════════════════════════
# GATES 2-4 — the block forward (LoRA OFF) over the EDIT layout
# ══════════════════════════════════════════════════════════════════════════════
def _gate_forward(
    ck: ShardedSafeTensors, fx: ShardedSafeTensors, bi: Int,
    rcos: Tensor, rsin: Tensor,
    ctx: DeviceContext, mut allok: Bool,
) raises:
    var w = _block_weights(ck, fx, bi, ctx)
    var lora = _no_lora()

    var x = TArc(_fv(fx, "kin_x", ctx))                      # [1,LFULL,F] bf16
    var vt = _fv(fx, "kin_blk_vec_t", ctx)                   # [1,6F] bf16
    var vc = TArc(_fv(fx, "kin_blk_vec_cond", ctx))          # [1,6F] bf16

    var cos_q = _tile_rope_table(rcos, LFULL, HEADS, HALF, ctx)
    var sin_q = _tile_rope_table(rsin, LFULL, HEADS, HALF, ctx)
    var cos_k = _tile_rope_table(rcos, LFULL, KVHEADS, HALF, ctx)
    var sin_k = _tile_rope_table(rsin, LFULL, KVHEADS, HALF, ctx)

    var lay = Krea2OminiLayout(LTMAX, S_IMG, S_COND, LT)
    var split = krea2_omini_mod_split(lay)                   # == COND_OFF
    print("")
    print("---- block", bi, ": EDIT forward, LoRA OFF, real_len=", lay.real_len(),
          " mod split=", split, " ----")

    var fwd = krea2_single_stream_block_lora[LFULL, HEADS, KVHEADS, HEADDIM](
        x.copy(), vt, w, lora, rcos, rsin, cos_q, sin_q, cos_k, sin_k, EPS, ctx,
        Optional[Int](lay.real_len()), Optional[TArc](vc.copy()),
        Optional[Int](split),
    )

    var pre_n = PAD_OFF * FEATURES          # elements in the real prefix
    var all_n = LFULL * FEATURES

    # ── 2. per-segment modulation ────────────────────────────────────────────
    print("")
    print("---- 2. kref_xm — PER-SEGMENT modulation (the new structural code) ----")
    var xm_h = fwd.saved.xm[].to_host(ctx)
    var xm_b = _host(fx, "kref_xm_f32mod", ctx)      # SCHEDULE B == Mojo's math
    _report("kref_xm_f32mod [full LFULL]", _cmp(xm_h, xm_b, 0, all_n),
            _env(fx, "xm_f32mod", ctx),
            "(PRIMARY: schedule-B ref, gated on env_cos_xm_f32mod)", allok,
            _modr(fx, "xm", ctx))
    _report_info("kref_xm_f32mod [prefix]  ", _cmp(xm_h, xm_b, 0, pre_n), "")
    _report_info("kref_xm_f32mod [cond rows]",
                 _cmp(xm_h, xm_b, COND_OFF * FEATURES, S_COND * FEATURES),
                 "<- rows that must use mods(t=0)")
    _report_info("kref_xm_f32mod [txt+img] ",
                 _cmp(xm_h, xm_b, 0, COND_OFF * FEATURES),
                 "<- rows that must use mods(t)")
    var xm_a = _host(fx, "kref_xm", ctx)             # SCHEDULE A, info only
    _report_sched_a("kref_xm [full LFULL]", _cmp(xm_h, xm_a, 0, all_n),
                    _env(fx, "xm", ctx), _modr(fx, "xm", ctx))

    # ══════════════════════════════════════════════════════════════════════
    # 3. MASKED ATTENTION over the new layout, ADAPTER DISABLED.
    #
    # The reference is kref_attn_raw_NOLORA (new in the C3 oracle) — the LoRA-OFF
    # attention seam, gated on its own env_cos_attn_raw_nolora. The LoRA-**ON**
    # kref_attn_raw is a DIFFERENT tensor and is NOT gated here; C3 gates it with
    # the real cond-row-routed block forward. Both numbers are printed so the
    # LoRA-attributable gap stays visible.
    #
    # TWO Mojo attention paths are measured:
    #   * the PRODUCTION cuDNN flash-padmask kernel (what training runs),
    #   * the deterministic F32 math masked path (sdpa_qwen_keymask) — the same
    #     math as the oracle's attn_f32, i.e. krea2_block.mojo's documented
    #     "F32 parity gate guard".
    # Both are GATED on the same shipped envelope. Flash has a different
    # accumulation order and is documented in krea2_block.mojo as a
    # value-tolerance path, so if the two ever disagree in verdict that
    # disagreement is the finding and is reported, not smoothed over.
    # ══════════════════════════════════════════════════════════════════════
    print("")
    print("---- 3. masked SDPA (RoPE + GQA) over the new layout, LoRA OFF ----")
    var at_b = _host(fx, "kref_attn_raw_nolora_f32mod", ctx)   # SCHEDULE B
    var env_arb = _env(fx, "attn_raw_nolora_f32mod", ctx)

    var q32 = cast_tensor(fwd.saved.q_rope[], STDtype.F32, ctx)
    var k32 = cast_tensor(fwd.saved.k_full[], STDtype.F32, ctx)
    var v32 = cast_tensor(fwd.saved.v_full[], STDtype.F32, ctx)
    var att32 = sdpa_qwen_keymask[1, LFULL, HEADS, HEADDIM, LFULL](
        q32, k32, v32, lay.real_len(),
        Float32(1.0) / sqrt(Float32(HEADDIM)), ctx,
    )
    var m32_h = att32.to_host(ctx)
    var at_h = fwd.saved.attn_flat[].to_host(ctx)
    _report("kref_attn_raw_nolora_f32mod [F32 math masked path, prefix]",
            _cmp(m32_h, at_b, 0, pre_n), env_arb,
            "(PRIMARY: schedule-B ref + env_cos_attn_raw_nolora_f32mod)", allok,
            _modr(fx, "attn_raw_nolora", ctx))
    _report("kref_attn_raw_nolora_f32mod [production flash-padmask, prefix]",
            _cmp(at_h, at_b, 0, pre_n), env_arb,
            "(PRIMARY: same schedule-B envelope)", allok,
            _modr(fx, "attn_raw_nolora", ctx))
    _report_info("  [TXT_real rows, flash]", _cmp(at_h, at_b, 0, LT * FEATURES), "")
    _report_info("  [IMG rows, flash]",
                 _cmp(at_h, at_b, LT * FEATURES, S_IMG * FEATURES), "")
    _report_info("  [COND rows, flash]",
                 _cmp(at_h, at_b, COND_OFF * FEATURES, S_COND * FEATURES), "")
    _report_info("  [pad tail rows — oracle zeroes them]",
                 _cmp(at_h, at_b, PAD_OFF * FEATURES,
                      (LFULL - PAD_OFF) * FEATURES),
                 "<- NOT gated (flash leaves pad query rows as garbage)")

    # SCHEDULE A cross-check (info only) — this is the comparison that produced
    # the reported block-27 near-miss on the flash path.
    var at_a = _host(fx, "kref_attn_raw_nolora", ctx)
    var env_ar = _env(fx, "attn_raw_nolora", ctx)
    _report_sched_a("kref_attn_raw_nolora [F32 math masked path, prefix]",
                    _cmp(m32_h, at_a, 0, pre_n), env_ar,
                    _modr(fx, "attn_raw_nolora", ctx))
    _report_sched_a("kref_attn_raw_nolora [production flash-padmask, prefix]",
                    _cmp(at_h, at_a, 0, pre_n), env_ar,
                    _modr(fx, "attn_raw_nolora", ctx))

    var at_on = _host(fx, "kref_attn_raw", ctx)
    print("   the LoRA-ON/LoRA-OFF gap in the FIXTURE itself (why C2 could not")
    print("   gate kref_attn_raw with a LoRA-off forward):")
    _report_info("  kref_attn_raw(ON) vs kref_attn_raw_nolora(OFF) [prefix]",
                 _cmp2(at_on, at_a, 0, 0, pre_n), "<- both are oracle tensors")
    _report_info("  LoRA-off Mojo flash vs kref_attn_raw (LoRA-ON) [prefix]",
                 _cmp(at_h, at_on, 0, pre_n), "<- the C2 mismatch, reproduced")

    # ── 4. whole block forward, LoRA disabled ────────────────────────────────
    print("")
    print("---- 4. kref_out_nolora — whole block forward, adapter DISABLED ----")
    var out_h = fwd.out[].to_host(ctx)
    var out_b = _host(fx, "kref_out_nolora_f32mod", ctx)       # SCHEDULE B
    _report("kref_out_nolora_f32mod [prefix 0:real_len]",
            _cmp(out_h, out_b, 0, pre_n),
            _env(fx, "out_nolora_f32mod", ctx),
            "(PRIMARY: schedule-B ref + env_cos_out_nolora_f32mod)", allok,
            _modr(fx, "out_nolora", ctx))
    _report_info("kref_out_nolora_f32mod [full LFULL]",
                 _cmp(out_h, out_b, 0, all_n),
                 "<- NOT gated (pad rows, see header)")
    _report_info("kref_out_nolora_f32mod [cond rows]",
                 _cmp(out_h, out_b, COND_OFF * FEATURES, S_COND * FEATURES), "")
    var out_a = _host(fx, "kref_out_nolora", ctx)              # SCHEDULE A, info
    _report_sched_a("kref_out_nolora [prefix 0:real_len]",
                    _cmp(out_h, out_a, 0, pre_n), _env(fx, "out_nolora", ctx),
                    _modr(fx, "out_nolora", ctx))

    var img_b = _host(fx, "kref_out_nolora_img_f32mod", ctx)
    _report("kref_out_nolora_img_f32mod [IMG rows = what the loss reads]",
            _cmp2(out_h, img_b, LT * FEATURES, 0, S_IMG * FEATURES),
            _env(fx, "out_nolora_img_f32mod", ctx),
            "(PRIMARY: schedule-B ref + env_cos_out_nolora_img_f32mod)", allok,
            _modr(fx, "out_nolora_img", ctx))
    var img_a = _host(fx, "kref_out_nolora_img", ctx)
    _report_sched_a("kref_out_nolora_img [IMG rows]",
                    _cmp2(out_h, img_a, LT * FEATURES, 0, S_IMG * FEATURES),
                    _env(fx, "out_nolora_img", ctx),
                    _modr(fx, "out_nolora_img", ctx))


# ══════════════════════════════════════════════════════════════════════════════
# GATE 5 — CONDLEN=0 REGRESSION (mandatory): with no condition segment the
# existing krea2 training path must be BIT-EQUAL. Real weights, real activations,
# real dims, through the real trainer-facing API.
#
# Run A: the PRE-C2 call form — the block invoked exactly as krea2_stack does
#        today (no vec_cond, no cond_off).
# Run B: the POST-C2 call form a CONDLEN=0 trainer build would emit — vec_cond
#        IS supplied, and cond_off = krea2_omini_mod_split(lay) which returns -1
#        because s_cond == 0, so the block's guard falls through.
# Also re-checks the POSITION path: the s_cond=0 combined table must equal the
# trainer's _reorder_pos_for_combined (train_krea2.mojo:886-891) applied to
# krea2_build_pos's output — computed here with the SAME ops the trainer uses.
# ══════════════════════════════════════════════════════════════════════════════
def _gate_condlen0(
    ck: ShardedSafeTensors, fx: ShardedSafeTensors, bi: Int,
    ctx: DeviceContext, mut allok: Bool,
) raises:
    print("")
    print("---- 5. CONDLEN=0 REGRESSION (bit-equality of the existing path) ----")
    var lay0 = Krea2OminiLayout(LTMAX, S_IMG, 0, LT)
    lay0.check_flash_prefix()
    var split0 = krea2_omini_mod_split(lay0)
    print("   layout s_cond=0: lfull=", lay0.lfull(), " real_len=", lay0.real_len(),
          " mod_split=", split0, " (-1 => block runs the UNCHANGED path)")
    if split0 != -1:
        allok = False
        print("   FAIL  krea2_omini_mod_split(s_cond=0) must be -1")

    # ── position path: layout builder vs the trainer's reorder, same ops ──────
    var p0 = krea2_omini_pos_combined(lay0, GRID, GRID, 0, 0, Float32(1.0))
    var pos_src = krea2_build_pos[LH, LW](LTMAX, ctx)         # [1,LTMAX+S_IMG,3]
    var pr = slice(pos_src, 1, 0, LT, ctx)
    var pi = slice(pos_src, 1, LTMAX, L_NC - LTMAX, ctx)
    var pp = slice(pos_src, 1, LT, LTMAX - LT, ctx)
    var ph = concat(1, ctx, pr, pi)
    var pos_tr = concat(1, ctx, ph, pp)                      # _reorder_pos_for_combined
    var pos_tr_h = pos_tr.to_host(ctx)
    var c_pos = _cmp(p0, pos_tr_h, 0, L_NC * 3)
    _report("condlen=0 pos == trainer _reorder_pos_for_combined", c_pos, 1.0,
            "(exact; bitdiff MUST be 0)", allok)

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

    # ── input with the COND segment removed: [TXT_real | IMG | TXT_pad] ──────
    var xf = _fv(fx, "kin_x", ctx)
    var xh = slice(xf, 1, 0, COND_OFF, ctx)
    var xt = slice(xf, 1, PAD_OFF, LTMAX - LT, ctx)
    var x_nc = TArc(concat(1, ctx, xh, xt))                  # [1, L_NC, F] bf16

    var w = _block_weights(ck, fx, bi, ctx)
    var vt = _fv(fx, "kin_blk_vec_t", ctx)
    var vc = TArc(_fv(fx, "kin_blk_vec_cond", ctx))

    # Run A — the PRE-C2 call form (what krea2_stack.mojo:407-410 emits today).
    var fa = krea2_single_stream_block_lora[L_NC, HEADS, KVHEADS, HEADDIM](
        x_nc.copy(), vt, w, _no_lora(), rcos, rsin,
        cos_q, sin_q, cos_k, sin_k, EPS, ctx, Optional[Int](lay0.real_len()),
    )
    var a_h = fa.out[].to_host(ctx)
    var a_xm = fa.saved.xm[].to_host(ctx)

    # Run B — the POST-C2 call form of a CONDLEN=0 build (split == -1).
    var fb = krea2_single_stream_block_lora[L_NC, HEADS, KVHEADS, HEADDIM](
        x_nc.copy(), vt, w, _no_lora(), rcos, rsin,
        cos_q, sin_q, cos_k, sin_k, EPS, ctx, Optional[Int](lay0.real_len()),
        Optional[TArc](vc.copy()), Optional[Int](split0),
    )
    var b_h = fb.out[].to_host(ctx)
    var b_xm = fb.saved.xm[].to_host(ctx)

    var n = L_NC * FEATURES
    var c_out = _cmp(a_h, b_h, 0, n)
    var c_xm = _cmp(a_xm, b_xm, 0, n)
    var bit_ok = (c_out.n_diff == 0) and (c_xm.n_diff == 0)
    if not bit_ok:
        allok = False
    print(
        "  ", "PASS" if bit_ok else "FAIL",
        " condlen=0 BIT-EQUAL  block out: bitdiff=", c_out.n_diff, "/", c_out.n,
        " max_abs=", c_out.max_abs,
        " | xm seam: bitdiff=", c_xm.n_diff, "/", c_xm.n,
        " max_abs=", c_xm.max_abs,
    )
    var nz = 0
    for i in range(n):
        if a_h[i] != 0.0:
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

    # fail loud if the fixture ever stops matching this gate's comptime shape
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
    _expect_i(fx, "meta_cond_delta_h", 0)
    _expect_i(fx, "meta_cond_delta_w", 0)

    # THRESHOLDS: each key's OWN measured bf16-vs-f32 envelope FOR THE ROUNDING
    # SCHEDULE THE MOJO KERNELS COMPUTE (schedule B, *_f32mod), straight from the
    # fixture. _env() hard-fails if an envelope is missing.
    print("THRESHOLDS (from the fixture, not invented).")
    print("   PRIMARY = schedule B (*_f32mod) — modulate/residual_gate in F32,")
    print("   rounded ONCE at store, which is exactly what")
    print("   serenitymojo/ops/elementwise.mojo _modulate_kernel_bf16 and")
    print("   _resgate_kernel_bf16 compute. Apples-to-apples.")
    print("   key                        env(B)=GATED        env(A)=info-only")
    print("   xm                        ", _env(fx, "xm_f32mod", ctx), "  ",
          _env(fx, "xm", ctx))
    print("   attn_raw_nolora           ",
          _env(fx, "attn_raw_nolora_f32mod", ctx), "  ",
          _env(fx, "attn_raw_nolora", ctx))
    print("   out_nolora                ", _env(fx, "out_nolora_f32mod", ctx),
          "  ", _env(fx, "out_nolora", ctx))
    print("   out_nolora_img            ",
          _env(fx, "out_nolora_img_f32mod", ctx), "  ",
          _env(fx, "out_nolora_img", ctx))
    print("   Each is the oracle's OWN bf16-vs-f32 cosine for THAT seam under")
    print("   THAT schedule on this block, both measured against the SAME F32-")
    print("   storage run (with F32 storage the two schedules are the identical")
    print("   sequence of F32 ops). Nothing gated tighter than its own envelope,")
    print("   nothing loosened. Mojo runs the PRODUCTION bf16 path.")

    var allok = True
    var tables = _gate_positions(fx, ctx, allok)
    _gate_forward(ck, fx, bi, tables[0], tables[1], ctx, allok)
    _gate_condlen0(ck, fx, bi, ctx, allok)

    print("")
    if allok:
        print("BLOCK", bi, "VERDICT: PASS")
    else:
        print("BLOCK", bi, "VERDICT: FAIL")
    return allok


def main() raises:
    var ctx = DeviceContext()
    print("==== krea2 OminiControl EDIT — C2 FORWARD parity gate ====")
    print("layout [TXT_real(", LT, ") | IMG(", S_IMG, ") | COND(", S_COND,
          ") | TXT_pad(", LTMAX - LT, ")]  LFULL=", LFULL,
          " real_len=", PAD_OFF, " cond_off=", COND_OFF)
    print("checkpoint:", CKPT)
    var ck = ShardedSafeTensors.open(String(CKPT))

    var ok0 = _run_block(0, ck, ctx)
    var ok27 = _run_block(27, ck, ctx)

    print("")
    print("==============================================================")
    if ok0 and ok27:
        print("C2 FORWARD GATE: PASS on BOTH block00 and block27")
    else:
        print("C2 FORWARD GATE: FAIL — block00 ",
              "PASS" if ok0 else "FAIL", ", block27 ",
              "PASS" if ok27 else "FAIL")
