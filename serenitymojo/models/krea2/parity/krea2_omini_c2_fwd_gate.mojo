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
#   2. kref_xm
#        Per-segment modulation: (1+prescale)*prenorm(x)+preshift with the
#        mods(t) chunks on rows [0,cond_off) and the mods(t=0) chunks on rows
#        [cond_off, LFULL). ONE elementwise op away from the fixture inputs, so
#        a mismatch here is unambiguous. Mojo side: krea2_block._modulate_seg2.
#   3. kref_attn_raw
#        Masked attention (real-length contiguous prefix, pad tail masked) with
#        RoPE + GQA over the new layout.
#        FINDING (proved bit-exactly, see section 3): the SHIPPED kref_attn_raw
#        is the LoRA-**ON** attention output — the oracle dumps
#        seams["attn_raw"] from the adapters-enabled forward
#        (krea2_omini_torch_oracle.py:791-793 -> :880); its only LoRA-off
#        references are kref_out_nolora / kref_out_nolora_img (:891-892). So a
#        LoRA-off C2 forward cannot match it, and the C2 brief's ordering of
#        this key as a LoRA-off target does not hold for the fixture as
#        generated. This gate therefore (a) reports the LoRA-off numbers as
#        INFORMATIONAL and (b) GATES the same Mojo masked-attention chain
#        rebuilt under the reference's own conditions. Fixing the fixture would
#        mean adding a `kref_attn_raw_nolora` key to the oracle.
#   4. kref_out_nolora / kref_out_nolora_img
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
#   That is the ceiling for any bf16 implementation. This gate uses
#   `env_cos_out` (0.99995387 at block 0, 0.99998802 at block 27) as the
#   threshold for every forward-activation key. The fixture ships NO per-key
#   envelope for xm / attn_raw, so env_cos_out — the block-level envelope — is
#   used for those too and is labelled as such in the output. Nothing is ever
#   asserted TIGHTER than the shipped envelope, and nothing is loosened.
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

from std.gpu.host import DeviceContext
from std.collections import List, Optional
from std.math import sqrt
from std.memory import ArcPointer
from serenitymojo.tensor import Tensor
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.tensor_algebra import slice, concat, add, mul_scalar
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.attention import sdpa_qwen_keymask
from serenitymojo.ops.linear import linear
from serenitymojo.ops.rope import rope_interleaved
from serenitymojo.ops.gqa_backward import repeat_kv_f32
from serenitymojo.ops.tensor_algebra import reshape_owned
from serenitymojo.models.klein.lora_block import LoraAdapterDevice
from serenitymojo.models.krea2.krea2_block import (
    Krea2BlockWeights, Krea2BlockLora, krea2_single_stream_block_lora,
)
from serenitymojo.models.dit.krea2_dit import (
    build_krea2_rope, _tile_rope_table, krea2_rmsnorm,
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


def _report(name: String, c: Cmp, thresh: Float64, note: String, mut allok: Bool):
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


# ── GATE-LOCAL DIAGNOSTIC ONLY — NOT a C2 seam, NOT wired into krea2_block ───
# The frozen base projection over the whole sequence + the fixture's low-rank
# delta added to the CONDITION ROWS only (the oracle's lin_cond_lora,
# krea2_omini_torch_oracle.py:460-473). Needed here purely to reconstruct the
# conditions under which the shipped kref_attn_raw was produced, so that the
# masked-attention/RoPE/GQA math C2 owns can be gated against it. Implementing
# this routing inside the block is C3 and is deliberately not done.
def _proj_cond_lora(
    x: Tensor, wgt: Tensor, fx: ShardedSafeTensors, slot: String,
    lscale: Float32, ctx: DeviceContext,
) raises -> Tensor:
    var nb = Optional[Tensor](None)
    var y = linear(x, wgt, nb^, ctx)
    var a = _fv(fx, "kin_lo_" + slot + "_A", ctx)     # [rank, in]
    var b = _fv(fx, "kin_lo_" + slot + "_B", ctx)     # [out, rank]
    var xc = slice(x, 1, COND_OFF, S_COND, ctx)
    var nb2 = Optional[Tensor](None)
    var t = linear(xc, a, nb2^, ctx)                  # [1, S_COND, rank]
    var nb3 = Optional[Tensor](None)
    var d = linear(t, b, nb3^, ctx)                   # [1, S_COND, out]
    var yc = slice(y, 1, COND_OFF, S_COND, ctx)
    var yc2 = add(yc, mul_scalar(d, lscale, ctx), ctx)
    var head = slice(y, 1, 0, COND_OFF, ctx)
    var tail = slice(y, 1, COND_OFF + S_COND, LFULL - COND_OFF - S_COND, ctx)
    var h2 = concat(1, ctx, head, yc2)
    return concat(1, ctx, h2, tail)


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
    rcos: Tensor, rsin: Tensor, env: Float64,
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
    var xm_r = _host(fx, "kref_xm", ctx)
    _report("kref_xm [full LFULL]", _cmp(xm_h, xm_r, 0, all_n), env,
            "(GATED on a BORROWED envelope — no env_cos_xm shipped)", allok)
    _report_info("kref_xm [prefix]  ", _cmp(xm_h, xm_r, 0, pre_n), "")
    _report_info("kref_xm [cond rows]",
                 _cmp(xm_h, xm_r, COND_OFF * FEATURES, S_COND * FEATURES),
                 "<- rows that must use mods(t=0)")
    _report_info("kref_xm [txt+img] ",
                 _cmp(xm_h, xm_r, 0, COND_OFF * FEATURES),
                 "<- rows that must use mods(t)")

    # ══════════════════════════════════════════════════════════════════════
    # 3. MASKED ATTENTION over the new layout.
    #
    # FINDING (measured, see the two blocks below — this is NOT an excuse, it is
    # a bit-exact demonstration): kref_attn_raw in the shipped fixture is the
    # LoRA-**ON** attention output. The oracle calls omini_block_forward WITH
    # the adapters (krea2_omini_torch_oracle.py:791-793) and dumps
    # seams["attn_raw"] from THAT run (:880); the ONLY LoRA-off references it
    # writes are kref_out_nolora / kref_out_nolora_img (:891-892, from the
    # separate `out_off` run at :833-835). kref_xm is unaffected because it is
    # computed BEFORE any projection, which is why gate 2 passes.
    # C2 is LoRA-OFF by definition, so a LoRA-off Mojo attention CANNOT match
    # kref_attn_raw, and gating it that way would be gating the LoRA delta, not
    # the layout. It is therefore reported here as INFORMATIONAL and NOT counted
    # in the verdict; it becomes a real gate in C3 when LoRA routing lands.
    #
    # What IS gated in C2 for this seam:
    #   (3a) the LoRA-OFF Mojo attention (production cuDNN flash-padmask AND the
    #        F32 math masked path) vs kref_attn_raw — INFORMATIONAL only, with
    #        the LoRA-attributable gap quantified.
    #   (3b) THE ACTUAL GATE: the same Mojo masked-attention chain
    #        (q/k/v -> QKNorm -> RoPE -> GQA -> masked SDPA over
    #        [TXT_real|IMG|COND|TXT_pad]) fed the fixture's own kref_xm, with the
    #        fixture's cond-row LoRA delta added to q/k/v purely as a DIAGNOSTIC
    #        (10 lines here in the gate; NOTHING is wired into krea2_block —
    #        cond-row LoRA routing remains C3). This reconstructs kref_attn_raw's
    #        exact production conditions, so it gates the layout / RoPE / mask /
    #        GQA math that C2 owns against a reference that is bit-reproducible.
    #   (4)  kref_out_nolora — the LoRA-off whole-block output, which contains
    #        the attention path end to end. That is the fixture's own C2 target.
    # ══════════════════════════════════════════════════════════════════════
    print("")
    print("---- 3. masked SDPA (RoPE + GQA) over the new layout ----")
    var at_r = _host(fx, "kref_attn_raw", ctx)

    var q32 = cast_tensor(fwd.saved.q_rope[], STDtype.F32, ctx)
    var k32 = cast_tensor(fwd.saved.k_full[], STDtype.F32, ctx)
    var v32 = cast_tensor(fwd.saved.v_full[], STDtype.F32, ctx)
    var att32 = sdpa_qwen_keymask[1, LFULL, HEADS, HEADDIM, LFULL](
        q32, k32, v32, lay.real_len(),
        Float32(1.0) / sqrt(Float32(HEADDIM)), ctx,
    )
    var m32_h = att32.to_host(ctx)
    var at_h = fwd.saved.attn_flat[].to_host(ctx)
    print("  3a. LoRA-OFF Mojo attention vs kref_attn_raw (LoRA-ON reference) —")
    print("      NOT GATED; the residual IS the cond-row LoRA delta.")
    _report_info("  LoRA-off flash-padmask [prefix]", _cmp(at_h, at_r, 0, pre_n), "")
    _report_info("  LoRA-off F32 math      [prefix]", _cmp(m32_h, at_r, 0, pre_n), "")
    _report_info("  LoRA-off flash [TXT rows]",
                 _cmp(at_h, at_r, 0, LT * FEATURES),
                 "<- LoRA reaches these only through attention")
    _report_info("  LoRA-off flash [IMG rows]",
                 _cmp(at_h, at_r, LT * FEATURES, S_IMG * FEATURES), "")
    _report_info("  LoRA-off flash [COND rows]",
                 _cmp(at_h, at_r, COND_OFF * FEATURES, S_COND * FEATURES),
                 "<- LoRA acts DIRECTLY here: worst, as the method predicts")

    # ── 3b. THE GATE: Mojo masked attention over the new layout, reconstructed
    #        under kref_attn_raw's own conditions (kref_xm input + cond-row LoRA
    #        on q/k/v as a gate-local diagnostic; no block wiring). ───────────
    var lscale = Float32(_host(fx, "meta_lora_scale", ctx)[0])
    var xmr = _fv(fx, "kref_xm", ctx)

    var pq = _proj_cond_lora(xmr, w.wq[], fx, "wq", lscale, ctx)
    var pk = _proj_cond_lora(xmr, w.wk[], fx, "wk", lscale, ctx)
    var pv = _proj_cond_lora(xmr, w.wv[], fx, "wv", lscale, ctx)
    var pqr = reshape_owned(pq^, [1, LFULL, HEADS, HEADDIM])
    var pkr = reshape_owned(pk^, [1, LFULL, KVHEADS, HEADDIM])
    var pvr = reshape_owned(pv^, [1, LFULL, KVHEADS, HEADDIM])
    var pqn = krea2_rmsnorm(pqr, w.qnorm_scale[], EPS, ctx)
    var pkn = krea2_rmsnorm(pkr, w.knorm_scale[], EPS, ctx)
    var pqo = rope_interleaved(pqn, cos_q, sin_q, ctx)
    var pko = rope_interleaved(pkn, cos_k, sin_k, ctx)
    var pkf = repeat_kv_f32(pko, LFULL, KVHEADS, HEADS // KVHEADS, HEADDIM, ctx)
    var pvf = repeat_kv_f32(pvr, LFULL, KVHEADS, HEADS // KVHEADS, HEADDIM, ctx)
    var pa = sdpa_qwen_keymask[1, LFULL, HEADS, HEADDIM, LFULL](
        cast_tensor(pqo, STDtype.F32, ctx),
        cast_tensor(pkf, STDtype.F32, ctx),
        cast_tensor(pvf, STDtype.F32, ctx),
        lay.real_len(), Float32(1.0) / sqrt(Float32(HEADDIM)), ctx,
    )
    var pa_h = pa.to_host(ctx)
    print("")
    print("  3b. GATED: Mojo masked attention over the EDIT layout, rebuilt from")
    print("      kref_xm under kref_attn_raw's own (LoRA-on) conditions.")
    _report("kref_attn_raw [Mojo masked attn over the new layout, prefix]",
            _cmp(pa_h, at_r, 0, pre_n), env,
            "(GATED on a BORROWED envelope — no env_cos_attn_raw shipped)",
            allok)
    _report_info("  [TXT_real rows]", _cmp(pa_h, at_r, 0, LT * FEATURES), "")
    _report_info("  [IMG rows]",
                 _cmp(pa_h, at_r, LT * FEATURES, S_IMG * FEATURES), "")
    _report_info("  [COND rows]",
                 _cmp(pa_h, at_r, COND_OFF * FEATURES, S_COND * FEATURES), "")
    _report_info("  [pad tail rows — oracle zeroes them]",
                 _cmp(pa_h, at_r, PAD_OFF * FEATURES,
                      (LFULL - PAD_OFF) * FEATURES),
                 "<- NOT gated")

    # ── 4. whole block forward, LoRA disabled ────────────────────────────────
    print("")
    print("---- 4. kref_out_nolora — whole block forward, adapter DISABLED ----")
    var out_h = fwd.out[].to_host(ctx)
    var out_r = _host(fx, "kref_out_nolora", ctx)
    _report("kref_out_nolora [prefix 0:real_len]", _cmp(out_h, out_r, 0, pre_n),
            env, "(GATED)", allok)
    _report_info("kref_out_nolora [full LFULL]", _cmp(out_h, out_r, 0, all_n),
                 "<- NOT gated (pad rows, see header)")
    _report_info("kref_out_nolora [cond rows]",
                 _cmp(out_h, out_r, COND_OFF * FEATURES, S_COND * FEATURES), "")

    var img_r = _host(fx, "kref_out_nolora_img", ctx)
    _report("kref_out_nolora_img [IMG rows = what the loss reads]",
            _cmp2(out_h, img_r, LT * FEATURES, 0, S_IMG * FEATURES), env,
            "(GATED)", allok)


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

    # THE THRESHOLD: the fixture's own measured bf16-vs-f32 forward envelope.
    var envl = _host(fx, "env_cos_out", ctx)
    var env = Float64(envl[0])
    print("THRESHOLD (from the fixture, not invented): env_cos_out =", env)
    print("   = the oracle's OWN bf16-vs-f32 cosine on this block, i.e. the")
    print("     ceiling for any bf16 implementation. Nothing is gated tighter,")
    print("     nothing is loosened. Mojo runs the PRODUCTION bf16 path.")
    print("   NOTE: the fixture ships env_cos ONLY for `out` (and for backward")
    print("     keys). kref_xm and kref_attn_raw have NO shipped envelope; they")
    print("     are gated on env_cos_out as a BORROWED threshold and labelled")
    print("     as such at every line. Adding env_cos_xm / env_cos_attn_raw to")
    print("     the oracle would remove the borrowing.")

    var allok = True
    var tables = _gate_positions(fx, ctx, allok)
    _gate_forward(ck, fx, bi, tables[0], tables[1], env, ctx, allok)
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
