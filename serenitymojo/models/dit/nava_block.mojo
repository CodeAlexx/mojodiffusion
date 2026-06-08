# models/dit/nava_block.mojo — pure-Mojo NAVA WanDoubleStreamAttentionBlock.
#
# Implements nava_double_block(...) which mirrors WanDoubleStreamAttentionBlock.forward
# with no_split_norm_ffn=True, masking_modality=False (joint attention), b=1.
#
# Key simplifications verified by the orchestrator:
#   - Oracle monkeypatches flash_attention→torch SDPA which IGNORES k_lens →
#     NO attention masking anywhere; full SDPA over all keys.
#   - For b=1 the joint-attn gather/scatter reorder is the identity (all tokens
#     valid) → SKIPPED entirely.
#
# Weight layout (NAVA_fp8.safetensors, prefix backbone.double_blocks.<blk>.):
#   FP8 linears: <name>.weight [out,in] F8_E4M3, <name>.weight_scale [out] BF16, <name>.bias [out] BF16
#   BF16 plain: modulation.modulation [1,6,3072], modulation_audio.modulation [1,6,3072]
#              norm3.weight [3072], norm3.bias [3072]
#              self_attn.{norm_q,norm_k,norm_q_audio,norm_k_audio}.weight [3072]
#              cross_attn.{norm_q,norm_k,norm_q_audio,norm_k_audio}.weight [3072]
#
# NavaRope struct: pre-built RoPE tables (constant across ALL 30 blocks + ALL denoise
# steps). Build once via build_nava_rope_tables(ctx) then pass to each block call.
# rope_interleaved consumes its cos/sin args → each block call clones from the struct
# (GPU memcpy, negligible vs. H2D rebuild + host position loop per call).
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.ops.fp8 import fp8_e4m3_dequant_perrow_to_bf16
from serenitymojo.ops.linear import linear
from serenitymojo.ops.norm import rms_norm, layer_norm, layer_norm_no_affine
from serenitymojo.ops.attention import sdpa_nomask, sdpa_nomask_tiled, sdpa_cross_nomask
from serenitymojo.ops.activations import gelu
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.rope_tables import build_multiaxis_rope_tables
from serenitymojo.ops.rope import rope_interleaved
from serenitymojo.ops.tensor_algebra import (
    add, mul, add_scalar, slice, concat, reshape,
)


# ── Module-level comptime constants ──────────────────────────────────────────
# Defined here (top-level) so both the NavaRope builder and the block forwards
# can reference them without forward-declaration issues.
comptime _VID = 320
comptime _AUD = 34
comptime _CTX = 512
comptime _SEQ = 354   # _VID + _AUD
comptime _DIM = 3072
comptime _NH  = 24
comptime _HD  = 128

# ── Hi-res (832×480) comptime constants ──────────────────────────────────────
# Latent grid: F=5, Hlat=30, Wlat=52 → patch [1,2,2] → Hp=15, Wp=26
# VID = 5*15*26 = 1950, SEQ = 1950+34 = 1984
comptime _VID_HR = 1950
comptime _SEQ_HR = 1984   # _VID_HR + _AUD


# ── NavaRope: pre-built RoPE tables (constant across all blocks + all steps) ──
#
# Uses ArcPointer[Tensor] fields so the struct is Copyable (ArcPointer has copy
# semantics via ref-counting). This allows NavaRope to be passed by value to block
# functions in a loop without consuming it from the caller — each copy just bumps
# the ArcPointer refcount; the underlying GPU tensors are shared. Inside each block,
# arc[].clone(ctx) issues a cheap GPU memcpy to get the Tensor that rope_interleaved
# will consume. See weight-dict pattern (Dict[String, ArcPointer[Tensor]]) for prior art.
#
struct NavaRope(Movable):
    """Pre-built RoPE cos/sin tables for video (3D) and audio (1D partial).

    Stores the Tuple[Tensor, Tensor] pairs returned by build_multiaxis_rope_tables.
    Tuple subscript in Mojo returns a borrow, so rope.vid[0] / rope.vid[1] can be
    passed to rope_interleaved (which borrows its cos/sin args in def context)
    without consuming the Tuple — correct for reuse across 30 blocks × N steps.

    vid: Tuple[Tensor, Tensor] = (vid_cos [7680,64] BF16, vid_sin [7680,64] BF16)
    aud: Tuple[Tensor, Tensor] = (aud_cos [816,22]  BF16, aud_sin [816,22]  BF16)
    """
    var vid: Tuple[Tensor, Tensor]
    var aud: Tuple[Tensor, Tensor]

    def __init__(out self, var v: Tuple[Tensor, Tensor], var a: Tuple[Tensor, Tensor]):
        self.vid = v^
        self.aud = a^


def build_nava_rope_tables(ctx: DeviceContext) raises -> NavaRope:
    """Build RoPE cos/sin tables for NAVA once and return as a NavaRope struct.

    This is the EXACT computation previously inlined inside every nava_double_block
    and nava_single_block call. Call once at model load; pass the resulting NavaRope
    to every block forward. Each block does arc[].clone(ctx) at each rope_interleaved
    call site — GPU memcpy, negligible vs. host position loop + H2D per block call.

    Grid is fixed: vid = (f=5, h=8, w=8) = 320 tokens, aud = 34 tokens, 24 heads.
    Axes: vid [44,42,42] (sum=128=head_dim), aud [44] (partial, first 44 of 128).
    θ = 10000.0 for both.
    """
    # ── Video: 3D tables over (f=5,h=8,w=8)=320 tokens, 24 heads ────────────
    var vid_rows = _VID * _NH   # 7680
    var vid_pos = List[Float32]()
    for rr in range(vid_rows):
        var tok = rr // _NH
        var f   = tok // 64        # 64 = H_G * W_G = 8*8
        var rem = tok % 64
        var hh  = rem // 8
        var ww  = rem % 8
        vid_pos.append(Float32(f))
        vid_pos.append(Float32(hh))
        vid_pos.append(Float32(ww))
    var vid_positions = Tensor.from_host(vid_pos^, [vid_rows * 3], STDtype.F32, ctx)
    var vid_axes = [44, 42, 42]
    var vid_cs = build_multiaxis_rope_tables(vid_positions, vid_axes, Float32(10000.0), ctx, STDtype.BF16)

    # ── Audio: partial 1D tables, rotate first 44 dims, 34 tokens × 24 heads ─
    var aud_rows = _AUD * _NH   # 816
    var aud_pos = List[Float32]()
    for rr in range(aud_rows):
        aud_pos.append(Float32(rr // _NH) * Float32(0.24))
    var aud_positions = Tensor.from_host(aud_pos^, [aud_rows], STDtype.F32, ctx)
    var aud_axes = [44]
    var aud_cs = build_multiaxis_rope_tables(aud_positions, aud_axes, Float32(10000.0), ctx, STDtype.BF16)

    # Store the Tuples directly — avoids extracting Tensors from Tuples (Mojo
    # Tuple subscript returns a borrow, can't be directly moved/transferred).
    return NavaRope(vid_cs^, aud_cs^)


# ── Weight store type alias ───────────────────────────────────────────────────
# Weights are stored in a Dict[String, ArcPointer[Tensor]] — access via w["key"][].
# (Tensor is Movable-not-Copyable; ArcPointer gives shared immutable access.)
# For linear bias, always wrap as Optional(w["b"][]  .clone(ctx)).


def _fp8_linear_to_bf16(
    st: ShardedSafeTensors,
    key: String,
    ctx: DeviceContext,
) raises -> Tensor:
    """Dequantise one FP8 linear weight from the safetensors file to BF16."""
    var wkey = key + ".weight"
    var skey = key + ".weight_scale"
    var wi = st.tensor_info(wkey)
    var w = Tensor.from_view_raw(
        from_parts(wi.dtype, wi.shape.copy(), st.tensor_bytes(wkey)), ctx
    )
    var si = st.tensor_info(skey)
    var scale = Tensor.from_view_as_f32(
        from_parts(si.dtype, si.shape.copy(), st.tensor_bytes(skey)), ctx
    )
    return fp8_e4m3_dequant_perrow_to_bf16(w, scale, ctx)


def load_nava_double_block(
    st: ShardedSafeTensors,
    block_prefix: String,
    ctx: DeviceContext,
) raises -> Dict[String, ArcPointer[Tensor]]:
    """Load and dequantise all weights for one NAVA double block.

    block_prefix: e.g. "backbone.double_blocks.0."
    Returns a Dict[String, ArcPointer[Tensor]] keyed by short name.
    Access weights as:  w["self_attn.q.weight"][]
    For linear bias:    Optional(w["self_attn.q.bias"][].clone(ctx))
    """
    var w = Dict[String, ArcPointer[Tensor]]()

    # FP8 linears (self-attn)
    var sa_linears = ["self_attn.q", "self_attn.k", "self_attn.v", "self_attn.o",
                      "self_attn.q_audio", "self_attn.k_audio", "self_attn.v_audio", "self_attn.o_audio"]
    for name in sa_linears:
        var sname = String(name)
        var full = block_prefix + sname
        w[sname + ".weight"] = ArcPointer(_fp8_linear_to_bf16(st, full, ctx))
        var bias = Tensor.from_view(st.tensor_view(full + ".bias"), ctx)
        w[sname + ".bias"] = ArcPointer(bias^)

    # FP8 linears (cross-attn)
    var ca_linears = ["cross_attn.q", "cross_attn.k", "cross_attn.v", "cross_attn.o",
                      "cross_attn.q_audio", "cross_attn.k_audio", "cross_attn.v_audio", "cross_attn.o_audio"]
    for name in ca_linears:
        var sname = String(name)
        var full = block_prefix + sname
        w[sname + ".weight"] = ArcPointer(_fp8_linear_to_bf16(st, full, ctx))
        var bias = Tensor.from_view(st.tensor_view(full + ".bias"), ctx)
        w[sname + ".bias"] = ArcPointer(bias^)

    # FP8 linears (ffn)
    var ffn_linears = ["ffn.0", "ffn.2"]
    for name in ffn_linears:
        var sname = String(name)
        var full = block_prefix + sname
        w[sname + ".weight"] = ArcPointer(_fp8_linear_to_bf16(st, full, ctx))
        var bias = Tensor.from_view(st.tensor_view(full + ".bias"), ctx)
        w[sname + ".bias"] = ArcPointer(bias^)

    # BF16 RMSNorm scales
    var rms_keys = ["self_attn.norm_q.weight", "self_attn.norm_k.weight",
                    "self_attn.norm_q_audio.weight", "self_attn.norm_k_audio.weight",
                    "cross_attn.norm_q.weight", "cross_attn.norm_k.weight",
                    "cross_attn.norm_q_audio.weight", "cross_attn.norm_k_audio.weight"]
    for name in rms_keys:
        var sname = String(name)
        var t = Tensor.from_view(st.tensor_view(block_prefix + sname), ctx)
        w[sname] = ArcPointer(t^)

    # BF16 norm3 weights
    var n3w = Tensor.from_view(st.tensor_view(block_prefix + "norm3.weight"), ctx)
    w["norm3.weight"] = ArcPointer(n3w^)
    var n3b = Tensor.from_view(st.tensor_view(block_prefix + "norm3.bias"), ctx)
    w["norm3.bias"] = ArcPointer(n3b^)

    # BF16 modulation tensors [1,6,3072]
    var mod_v = Tensor.from_view(st.tensor_view(block_prefix + "modulation.modulation"), ctx)
    w["modulation.modulation"] = ArcPointer(mod_v^)
    var mod_a = Tensor.from_view(st.tensor_view(block_prefix + "modulation_audio.modulation"), ctx)
    w["modulation_audio.modulation"] = ArcPointer(mod_a^)

    return w^


# ── Linear helper (mirrors wan22_dit._lin pattern) ──────────────────────────
def _nava_lin(
    x: Tensor,
    w: Dict[String, ArcPointer[Tensor]],
    stem: String,
    ctx: DeviceContext,
) raises -> Tensor:
    """linear(x, w[stem+.weight], Optional(w[stem+.bias].clone(ctx)), ctx)."""
    var wk = stem + ".weight"
    var bk = stem + ".bias"
    return linear(x, w[wk][], Optional(w[bk][].clone(ctx)), ctx)


# ── Modulation helper ────────────────────────────────────────────────────────
def _mod_chunk(
    e: Tensor,      # [1, L, 6, 3072]
    mod: Tensor,    # [1, 6, 3072]
    i: Int,
    Ltok: Int,
    dim: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    """Extract e[:,:,i,:] + mod[:,i,:] -> [1,Ltok,dim]."""
    var es4 = slice(e,   2, i, 1, ctx)         # [1,L,1,dim]
    var ms  = slice(mod, 1, i, 1, ctx)         # [1,1,dim]
    var ms4 = reshape(ms, [1, 1, 1, dim], ctx) # [1,1,1,dim]
    var t4  = add(es4, ms4, ctx)               # [1,L,1,dim]
    return reshape(t4, [1, Ltok, dim], ctx)    # [1,L,dim]


# ── Main block forward ────────────────────────────────────────────────────────
def nava_double_block(
    x: Tensor,          # [1, 354, 3072]  BF16
    e_vid: Tensor,      # [1, 320, 6, 3072] BF16
    e_audio: Tensor,    # [1, 34, 6, 3072]  BF16
    context: Tensor,    # [1, 512, 3072]   BF16
    w: Dict[String, ArcPointer[Tensor]],
    rope: NavaRope,
    ctx: DeviceContext,
    masking_modality: Bool = False,
) raises -> Tensor:
    """NAVA WanDoubleStreamAttentionBlock forward (b=1).

    When masking_modality=False (default): joint self-attention over all 354 tokens.
    When masking_modality=True: non-joint self-attention — video tokens attend to video
    only (320), audio tokens attend to audio only (34), as separate SDPAs.

    rope: NavaRope with pre-built tables (ArcPointer fields; safe to pass by value
    in a loop — each copy bumps ArcPointer refcount, underlying GPU buffers shared).
    Returns [1, 354, 3072] BF16.
    """
    var EPS   = Float32(1e-6)
    var SCALE = Float32(0.08838834764831843)  # 1/sqrt(128)

    # ── 1) Split x into video and audio streams ──────────────────────────────
    var x_vid   = slice(x, 1, 0,    _VID, ctx)    # [1,320,3072]
    var x_audio = slice(x, 1, _VID, _AUD, ctx)    # [1,34,3072]

    # ── 2) ModulationAdd + chunk(6) ──────────────────────────────────────────
    # modulation.modulation is [1,6,3072]; e_vid is [1,320,6,3072].
    # ev[i] = e_vid[:,:,i,:] + modulation[:,i,:]  -> [1,320,3072]
    # Pass w["..."][] directly to avoid copy; _mod_chunk borrows via def.
    # ev[0]=shift_msa, ev[1]=scale_msa, ev[2]=gate_msa, ev[3]=shift_mlp, ev[4]=scale_mlp, ev[5]=gate_mlp
    var ev0 = _mod_chunk(e_vid,   w["modulation.modulation"][],       0, _VID, _DIM, ctx)
    var ev1 = _mod_chunk(e_vid,   w["modulation.modulation"][],       1, _VID, _DIM, ctx)
    var ev2 = _mod_chunk(e_vid,   w["modulation.modulation"][],       2, _VID, _DIM, ctx)
    var ev3 = _mod_chunk(e_vid,   w["modulation.modulation"][],       3, _VID, _DIM, ctx)
    var ev4 = _mod_chunk(e_vid,   w["modulation.modulation"][],       4, _VID, _DIM, ctx)
    var ev5 = _mod_chunk(e_vid,   w["modulation.modulation"][],       5, _VID, _DIM, ctx)

    var ea0 = _mod_chunk(e_audio, w["modulation_audio.modulation"][], 0, _AUD, _DIM, ctx)
    var ea1 = _mod_chunk(e_audio, w["modulation_audio.modulation"][], 1, _AUD, _DIM, ctx)
    var ea2 = _mod_chunk(e_audio, w["modulation_audio.modulation"][], 2, _AUD, _DIM, ctx)
    var ea3 = _mod_chunk(e_audio, w["modulation_audio.modulation"][], 3, _AUD, _DIM, ctx)
    var ea4 = _mod_chunk(e_audio, w["modulation_audio.modulation"][], 4, _AUD, _DIM, ctx)
    var ea5 = _mod_chunk(e_audio, w["modulation_audio.modulation"][], 5, _AUD, _DIM, ctx)

    # ── 3) adaLN norm1 (no affine): xvn = (1+scale_msa)*norm(x) + shift_msa ──
    var xvn_norm = layer_norm_no_affine(x_vid,   EPS, ctx)
    var xan_norm = layer_norm_no_affine(x_audio, EPS, ctx)
    var xvn = add(mul(xvn_norm, add_scalar(ev1, 1.0, ctx), ctx), ev0, ctx)
    var xan = add(mul(xan_norm, add_scalar(ea1, 1.0, ctx), ctx), ea0, ctx)

    # ── 4) Joint self-attention ───────────────────────────────────────────────
    # qk-RMSNorm on full 3072 BEFORE reshape to heads.

    # Video Q,K,V
    var qv_lin = _nava_lin(xvn, w, "self_attn.q", ctx)
    var kv_lin = _nava_lin(xvn, w, "self_attn.k", ctx)
    var vv_lin = _nava_lin(xvn, w, "self_attn.v", ctx)
    var qv_norm = rms_norm(qv_lin, w["self_attn.norm_q.weight"][], EPS, ctx)
    var kv_norm = rms_norm(kv_lin, w["self_attn.norm_k.weight"][], EPS, ctx)
    var qv4 = reshape(qv_norm, [1, _VID, _NH, _HD], ctx)
    var kv4 = reshape(kv_norm, [1, _VID, _NH, _HD], ctx)
    var vv4 = reshape(vv_lin,  [1, _VID, _NH, _HD], ctx)

    # Audio Q,K,V
    var qa_lin = _nava_lin(xan, w, "self_attn.q_audio", ctx)
    var ka_lin = _nava_lin(xan, w, "self_attn.k_audio", ctx)
    var va_lin = _nava_lin(xan, w, "self_attn.v_audio", ctx)
    var qa_norm = rms_norm(qa_lin, w["self_attn.norm_q_audio.weight"][], EPS, ctx)
    var ka_norm = rms_norm(ka_lin, w["self_attn.norm_k_audio.weight"][], EPS, ctx)
    var qa4 = reshape(qa_norm, [1, _AUD, _NH, _HD], ctx)
    var ka4 = reshape(ka_norm, [1, _AUD, _NH, _HD], ctx)
    var va4 = reshape(va_lin,  [1, _AUD, _NH, _HD], ctx)

    # ── RoPE video: borrow Tuple elements (rope.vid[0] = cos, [1] = sin) ────────
    var qv_roped = rope_interleaved(qv4, rope.vid[0], rope.vid[1], ctx)
    var kv_roped = rope_interleaved(kv4, rope.vid[0], rope.vid[1], ctx)

    # ── RoPE audio: borrow Tuple elements (rope.aud[0] = cos, [1] = sin) ─────
    var qa_rot  = slice(qa4, 3, 0,  44, ctx)
    var qa_pass = slice(qa4, 3, 44, 84, ctx)
    var ka_rot  = slice(ka4, 3, 0,  44, ctx)
    var ka_pass = slice(ka4, 3, 44, 84, ctx)
    var qa_rotd = rope_interleaved(qa_rot, rope.aud[0], rope.aud[1], ctx)
    var ka_rotd = rope_interleaved(ka_rot, rope.aud[0], rope.aud[1], ctx)
    var qa_roped = concat(3, ctx, qa_rotd, qa_pass)
    var ka_roped = concat(3, ctx, ka_rotd, ka_pass)

    # ── Self-attention: joint or non-joint path ──────────────────────────────
    var yv: Tensor
    var ya: Tensor
    if masking_modality:
        # Non-joint: video attends to video only, audio to audio only.
        var att_vid = sdpa_nomask[1, _VID, _NH, _HD](qv_roped, kv_roped, vv4, SCALE, ctx)
        var att_aud = sdpa_nomask[1, _AUD, _NH, _HD](qa_roped, ka_roped, va4, SCALE, ctx)
        yv = _nava_lin(reshape(att_vid, [1, _VID, _DIM], ctx),
                       w, "self_attn.o", ctx)
        ya = _nava_lin(reshape(att_aud, [1, _AUD, _DIM], ctx),
                       w, "self_attn.o_audio", ctx)
    else:
        # Joint: all 354 tokens attend together.
        var q_joint = concat(1, ctx, qv_roped, qa_roped)
        var k_joint = concat(1, ctx, kv_roped, ka_roped)
        var v_joint = concat(1, ctx, vv4,      va4)
        var att = sdpa_nomask_tiled[1, _SEQ, _NH, _HD](q_joint, k_joint, v_joint, SCALE, ctx)
        var att_vid = slice(att, 1, 0,    _VID, ctx)
        var att_aud = slice(att, 1, _VID, _AUD, ctx)
        yv = _nava_lin(reshape(att_vid, [1, _VID, _DIM], ctx),
                       w, "self_attn.o", ctx)
        ya = _nava_lin(reshape(att_aud, [1, _AUD, _DIM], ctx),
                       w, "self_attn.o_audio", ctx)

    # ── gate_msa residual ─────────────────────────────────────────────────────
    x_vid   = add(x_vid,   mul(yv, ev2, ctx), ctx)
    x_audio = add(x_audio, mul(ya, ea2, ctx), ctx)

    # ── 5) Cross-attention to text (norm3 affine LayerNorm, SHARED) ──────────
    var xvn3 = layer_norm(x_vid,   w["norm3.weight"][], w["norm3.bias"][], EPS, ctx)
    var xan3 = layer_norm(x_audio, w["norm3.weight"][], w["norm3.bias"][], EPS, ctx)

    # Video cross-attn: q from xvn3, k/v from context
    var qv3   = _nava_lin(xvn3,    w, "cross_attn.q", ctx)
    var kc    = _nava_lin(context, w, "cross_attn.k", ctx)
    var vc    = _nava_lin(context, w, "cross_attn.v", ctx)
    var qv3_n = rms_norm(qv3, w["cross_attn.norm_q.weight"][], EPS, ctx)
    var kc_n  = rms_norm(kc,  w["cross_attn.norm_k.weight"][], EPS, ctx)
    var qv3_4 = reshape(qv3_n, [1, _VID, _NH, _HD], ctx)
    var kc_4  = reshape(kc_n,  [1, _CTX, _NH, _HD], ctx)
    var vc_4  = reshape(vc,    [1, _CTX, _NH, _HD], ctx)
    var cvid  = sdpa_cross_nomask[1, _VID, _CTX, _NH, _HD](qv3_4, kc_4, vc_4, SCALE, ctx)
    var xva   = _nava_lin(reshape(cvid, [1, _VID, _DIM], ctx),
                          w, "cross_attn.o", ctx)

    # Audio cross-attn: q from xan3, k/v from context (audio-specific weights)
    var qa3   = _nava_lin(xan3,    w, "cross_attn.q_audio", ctx)
    var kca   = _nava_lin(context, w, "cross_attn.k_audio", ctx)
    var vca   = _nava_lin(context, w, "cross_attn.v_audio", ctx)
    var qa3_n = rms_norm(qa3, w["cross_attn.norm_q_audio.weight"][], EPS, ctx)
    var kca_n = rms_norm(kca, w["cross_attn.norm_k_audio.weight"][], EPS, ctx)
    var qa3_4 = reshape(qa3_n, [1, _AUD, _NH, _HD], ctx)
    var kca_4 = reshape(kca_n, [1, _CTX, _NH, _HD], ctx)
    var vca_4 = reshape(vca,   [1, _CTX, _NH, _HD], ctx)
    var caud  = sdpa_cross_nomask[1, _AUD, _CTX, _NH, _HD](qa3_4, kca_4, vca_4, SCALE, ctx)
    var xaa   = _nava_lin(reshape(caud, [1, _AUD, _DIM], ctx),
                          w, "cross_attn.o_audio", ctx)

    # Residual (NO gate in cross-attn)
    x_vid   = add(x_vid,   xva, ctx)
    x_audio = add(x_audio, xaa, ctx)

    # ── 6) FFN with mlp modulation ───────────────────────────────────────────
    # ffn(z) = linear( gelu( linear(z, ffn0.w, ffn0.b) ), ffn2.w, ffn2.b )

    # Video FFN: no-affine norm2, scale/shift (ev[4,3]), gate ev[5]
    var xvn2     = layer_norm_no_affine(x_vid, EPS, ctx)
    var xvn2_mod = add(mul(xvn2, add_scalar(ev4, 1.0, ctx), ctx), ev3, ctx)
    var h_vid    = _nava_lin(xvn2_mod, w, "ffn.0", ctx)
    var hg_vid   = gelu(h_vid, ctx)
    var yv_ffn   = _nava_lin(hg_vid, w, "ffn.2", ctx)
    x_vid        = add(x_vid, mul(yv_ffn, ev5, ctx), ctx)

    # Audio FFN: no-affine norm2, scale/shift (ea[4,3]), gate ea[5]
    var xan2     = layer_norm_no_affine(x_audio, EPS, ctx)
    var xan2_mod = add(mul(xan2, add_scalar(ea4, 1.0, ctx), ctx), ea3, ctx)
    var h_aud    = _nava_lin(xan2_mod, w, "ffn.0", ctx)
    var hg_aud   = gelu(h_aud, ctx)
    var ya_ffn   = _nava_lin(hg_aud, w, "ffn.2", ctx)
    x_audio      = add(x_audio, mul(ya_ffn, ea5, ctx), ctx)

    # ── 7) Concat -> [1, 354, 3072] ─────────────────────────────────────────
    return concat(1, ctx, x_vid, x_audio)


# ══════════════════════════════════════════════════════════════════════════════
# NAVA WanAttentionBlock (single-stream) — chunk 6
# ══════════════════════════════════════════════════════════════════════════════
#
# Mirrors WanAttentionBlock.forward with split_av_qk_norm_modulation=False,
# masking_modality=False (joint attention), b=1.
#
# Weight layout (NAVA_fp8.safetensors, prefix backbone.single_blocks.<blk>.):
#   FP8 linears: self_attn.{q,k,v,o}, cross_attn.{q,k,v,o}, ffn.0, ffn.2
#   BF16 plain : modulation.modulation [1,6,3072], norm3.{weight,bias} [3072]
#                self_attn.{norm_q,norm_k}.weight, cross_attn.{norm_q,norm_k}.weight
#   NO norm1/norm2 weights, NO _audio variants, NO modulation_audio.
#
def load_nava_single_block(
    st: ShardedSafeTensors,
    block_prefix: String,
    ctx: DeviceContext,
) raises -> Dict[String, ArcPointer[Tensor]]:
    """Load and dequantise all weights for one NAVA single block.

    block_prefix: e.g. "backbone.single_blocks.0."
    Returns Dict[String, ArcPointer[Tensor]] keyed by short name.
    """
    var w = Dict[String, ArcPointer[Tensor]]()

    # FP8 linears (self-attn)
    var sa_linears = ["self_attn.q", "self_attn.k", "self_attn.v", "self_attn.o"]
    for name in sa_linears:
        var sname = String(name)
        var full = block_prefix + sname
        w[sname + ".weight"] = ArcPointer(_fp8_linear_to_bf16(st, full, ctx))
        var bias = Tensor.from_view(st.tensor_view(full + ".bias"), ctx)
        w[sname + ".bias"] = ArcPointer(bias^)

    # FP8 linears (cross-attn)
    var ca_linears = ["cross_attn.q", "cross_attn.k", "cross_attn.v", "cross_attn.o"]
    for name in ca_linears:
        var sname = String(name)
        var full = block_prefix + sname
        w[sname + ".weight"] = ArcPointer(_fp8_linear_to_bf16(st, full, ctx))
        var bias = Tensor.from_view(st.tensor_view(full + ".bias"), ctx)
        w[sname + ".bias"] = ArcPointer(bias^)

    # FP8 linears (ffn)
    var ffn_linears = ["ffn.0", "ffn.2"]
    for name in ffn_linears:
        var sname = String(name)
        var full = block_prefix + sname
        w[sname + ".weight"] = ArcPointer(_fp8_linear_to_bf16(st, full, ctx))
        var bias = Tensor.from_view(st.tensor_view(full + ".bias"), ctx)
        w[sname + ".bias"] = ArcPointer(bias^)

    # BF16 RMSNorm scales (self-attn and cross-attn, NO _audio variants)
    var rms_keys = ["self_attn.norm_q.weight", "self_attn.norm_k.weight",
                    "cross_attn.norm_q.weight", "cross_attn.norm_k.weight"]
    for name in rms_keys:
        var sname = String(name)
        var t = Tensor.from_view(st.tensor_view(block_prefix + sname), ctx)
        w[sname] = ArcPointer(t^)

    # BF16 norm3 weights (affine LayerNorm for cross-attn pre-norm)
    var n3w = Tensor.from_view(st.tensor_view(block_prefix + "norm3.weight"), ctx)
    w["norm3.weight"] = ArcPointer(n3w^)
    var n3b = Tensor.from_view(st.tensor_view(block_prefix + "norm3.bias"), ctx)
    w["norm3.bias"] = ArcPointer(n3b^)

    # BF16 modulation tensor [1,6,3072] — single modulation over all 354 tokens
    var mod_v = Tensor.from_view(st.tensor_view(block_prefix + "modulation.modulation"), ctx)
    w["modulation.modulation"] = ArcPointer(mod_v^)

    return w^


def nava_single_block(
    x: Tensor,          # [1, 354, 3072]  BF16
    e_vid: Tensor,      # [1, 320, 6, 3072] BF16
    e_audio: Tensor,    # [1, 34, 6, 3072]  BF16
    context: Tensor,    # [1, 512, 3072]   BF16
    w: Dict[String, ArcPointer[Tensor]],
    rope: NavaRope,
    ctx: DeviceContext,
    masking_modality: Bool = False,
) raises -> Tensor:
    """NAVA WanAttentionBlock forward (b=1, no split modulation).

    When masking_modality=False (default): joint self-attention over all 354 tokens.
    When masking_modality=True: non-joint self-attention — video and audio attend
    separately, then concatenated before the single output projection.

    rope: NavaRope with pre-built tables (ArcPointer fields; safe to pass by value
    in a loop — each copy bumps ArcPointer refcount, underlying GPU buffers shared).
    Returns [1, 354, 3072] BF16.
    """
    var EPS   = Float32(1e-6)
    var SCALE = Float32(0.08838834764831843)  # 1/sqrt(128)

    # ── 1) Concat e_vid and e_audio along token dim ──────────────────────────
    # e_cat = [1, 354, 6, 3072]  (320 vid ++ 34 audio on dim=1)
    var e_cat = concat(1, ctx, e_vid, e_audio)

    # ── 2) ModulationAdd + chunk(6) over the combined 354 tokens ─────────────
    # modulation.modulation [1,6,3072] broadcasts over the 354 token dim.
    # _mod_chunk extracts e_cat[:,:,i,:] + mod[:,i,:] -> [1,354,3072].
    # Indices: 0=shift_msa, 1=scale_msa, 2=gate_msa, 3=shift_mlp, 4=scale_mlp, 5=gate_mlp
    var e0 = _mod_chunk(e_cat, w["modulation.modulation"][], 0, _SEQ, _DIM, ctx)
    var e1 = _mod_chunk(e_cat, w["modulation.modulation"][], 1, _SEQ, _DIM, ctx)
    var e2 = _mod_chunk(e_cat, w["modulation.modulation"][], 2, _SEQ, _DIM, ctx)
    var e3 = _mod_chunk(e_cat, w["modulation.modulation"][], 3, _SEQ, _DIM, ctx)
    var e4 = _mod_chunk(e_cat, w["modulation.modulation"][], 4, _SEQ, _DIM, ctx)
    var e5 = _mod_chunk(e_cat, w["modulation.modulation"][], 5, _SEQ, _DIM, ctx)

    # ── 3) adaLN norm1 (no affine): xn = (1+scale_msa)*norm(x) + shift_msa ──
    var xn_norm = layer_norm_no_affine(x, EPS, ctx)
    var xn = add(mul(xn_norm, add_scalar(e1, 1.0, ctx), ctx), e0, ctx)

    # ── 4) Joint self-attention (single projection over all 354 tokens) ───────
    # qk-RMSNorm on full 3072 BEFORE reshape to heads.
    var q_lin  = _nava_lin(xn, w, "self_attn.q", ctx)
    var k_lin  = _nava_lin(xn, w, "self_attn.k", ctx)
    var v_lin  = _nava_lin(xn, w, "self_attn.v", ctx)
    var q_norm = rms_norm(q_lin, w["self_attn.norm_q.weight"][], EPS, ctx)
    var k_norm = rms_norm(k_lin, w["self_attn.norm_k.weight"][], EPS, ctx)
    var q4     = reshape(q_norm, [1, _SEQ, _NH, _HD], ctx)
    var k4     = reshape(k_norm, [1, _SEQ, _NH, _HD], ctx)
    var v4     = reshape(v_lin,  [1, _SEQ, _NH, _HD], ctx)

    # ── RoPE: video tokens 0..319, audio tokens 320..353 ─────────────────────
    # Slice q4/k4 into vid and audio portions.
    var qv4 = slice(q4, 1, 0,    _VID, ctx)   # [1,320,24,128]
    var qa4 = slice(q4, 1, _VID, _AUD, ctx)   # [1,34,24,128]
    var kv4 = slice(k4, 1, 0,    _VID, ctx)
    var ka4 = slice(k4, 1, _VID, _AUD, ctx)

    # Video RoPE: borrow Tuple elements (rope.vid[0] = cos, [1] = sin)
    var qv_roped = rope_interleaved(qv4, rope.vid[0], rope.vid[1], ctx)
    var kv_roped = rope_interleaved(kv4, rope.vid[0], rope.vid[1], ctx)

    # Audio RoPE: borrow Tuple elements (rope.aud[0] = cos, [1] = sin)
    var qa_rot   = slice(qa4, 3, 0,  44, ctx)
    var qa_pass  = slice(qa4, 3, 44, 84, ctx)
    var ka_rot   = slice(ka4, 3, 0,  44, ctx)
    var ka_pass  = slice(ka4, 3, 44, 84, ctx)
    var qa_rotd  = rope_interleaved(qa_rot, rope.aud[0], rope.aud[1], ctx)
    var ka_rotd  = rope_interleaved(ka_rot, rope.aud[0], rope.aud[1], ctx)
    var qa_roped = concat(3, ctx, qa_rotd, qa_pass)
    var ka_roped = concat(3, ctx, ka_rotd, ka_pass)

    # ── Self-attention: joint or non-joint path ──────────────────────────────
    var x1: Tensor
    if masking_modality:
        # Non-joint: video and audio attend separately; concat, then one output proj.
        var vv_s = slice(v4, 1, 0,    _VID, ctx)  # [1,320,24,128]
        var va_s = slice(v4, 1, _VID, _AUD, ctx)  # [1,34,24,128]
        var att_vid = sdpa_nomask[1, _VID, _NH, _HD](qv_roped, kv_roped, vv_s, SCALE, ctx)
        var att_aud = sdpa_nomask[1, _AUD, _NH, _HD](qa_roped, ka_roped, va_s, SCALE, ctx)
        var att = concat(1, ctx, att_vid, att_aud)  # [1,354,24,128]
        var y_sa = _nava_lin(reshape(att, [1, _SEQ, _DIM], ctx), w, "self_attn.o", ctx)
        x1 = add(x, mul(y_sa, e2, ctx), ctx)
    else:
        # Joint: all 354 tokens attend together.
        var q_roped = concat(1, ctx, qv_roped, qa_roped)
        var k_roped = concat(1, ctx, kv_roped, ka_roped)
        var att = sdpa_nomask_tiled[1, _SEQ, _NH, _HD](q_roped, k_roped, v4, SCALE, ctx)
        var y_sa = _nava_lin(reshape(att, [1, _SEQ, _DIM], ctx), w, "self_attn.o", ctx)
        x1 = add(x, mul(y_sa, e2, ctx), ctx)

    # ── 5) Cross-attention (norm3 affine, q over 354 tokens, k/v over 512) ───
    # norm3 is affine LayerNorm (norm3.weight / norm3.bias present).
    var xn3 = layer_norm(x1, w["norm3.weight"][], w["norm3.bias"][], EPS, ctx)
    var qc_lin = _nava_lin(xn3,    w, "cross_attn.q", ctx)
    var kc_lin = _nava_lin(context, w, "cross_attn.k", ctx)
    var vc_lin = _nava_lin(context, w, "cross_attn.v", ctx)
    var qc_n   = rms_norm(qc_lin, w["cross_attn.norm_q.weight"][], EPS, ctx)
    var kc_n   = rms_norm(kc_lin, w["cross_attn.norm_k.weight"][], EPS, ctx)
    var qc4    = reshape(qc_n, [1, _SEQ, _NH, _HD], ctx)
    var kc4    = reshape(kc_n, [1, _CTX, _NH, _HD], ctx)
    var vc4    = reshape(vc_lin, [1, _CTX, _NH, _HD], ctx)
    var cx     = sdpa_cross_nomask[1, _SEQ, _CTX, _NH, _HD](qc4, kc4, vc4, SCALE, ctx)
    var y_ca   = _nava_lin(reshape(cx, [1, _SEQ, _DIM], ctx), w, "cross_attn.o", ctx)
    # Residual without gate (matches WanT2VCrossAttention forward: x = x + cross_attn(...))
    var x2 = add(x1, y_ca, ctx)

    # ── 6) FFN with mlp modulation ────────────────────────────────────────────
    var xn2     = layer_norm_no_affine(x2, EPS, ctx)
    var xn2_mod = add(mul(xn2, add_scalar(e4, 1.0, ctx), ctx), e3, ctx)
    var h_ffn   = _nava_lin(xn2_mod, w, "ffn.0", ctx)
    var hg_ffn  = gelu(h_ffn, ctx)
    var y_ffn   = _nava_lin(hg_ffn, w, "ffn.2", ctx)
    return add(x2, mul(y_ffn, e5, ctx), ctx)^


# ══════════════════════════════════════════════════════════════════════════════
# Hi-res (832×480) variants: VID=1950, SEQ=1984
# Latent grid F=5, Hlat=30, Wlat=52 → patch [1,2,2] → Hp=15, Wp=26
# rope grid: tok//(15*26=390), (tok%390)//26, %26
# ══════════════════════════════════════════════════════════════════════════════

def build_nava_rope_tables_hires(ctx: DeviceContext) raises -> NavaRope:
    """Build RoPE cos/sin tables for NAVA hi-res (832×480) once.

    Grid: vid = (F=5, Hp=15, Wp=26) = 1950 tokens, aud = 34 tokens, 24 heads.
    Position decode: tok//(Hp*Wp), (tok%(Hp*Wp))//Wp, %Wp.
    Axes: vid [44,42,42] (sum=128=head_dim), aud [44] partial.
    θ = 10000.0 for both.
    """
    comptime _Hp = 15
    comptime _Wp = 26
    comptime _HpWp = 390   # 15 * 26

    # ── Video: 3D tables over (f=5,h=15,w=26)=1950 tokens, 24 heads ──────────
    var vid_rows = _VID_HR * _NH   # 1950 * 24 = 46800
    var vid_pos = List[Float32]()
    for rr in range(vid_rows):
        var tok = rr // _NH
        var f   = tok // _HpWp
        var rem = tok % _HpWp
        var hh  = rem // _Wp
        var ww  = rem % _Wp
        vid_pos.append(Float32(f))
        vid_pos.append(Float32(hh))
        vid_pos.append(Float32(ww))
    var vid_positions = Tensor.from_host(vid_pos^, [vid_rows * 3], STDtype.F32, ctx)
    var vid_axes = [44, 42, 42]
    var vid_cs = build_multiaxis_rope_tables(vid_positions, vid_axes, Float32(10000.0), ctx, STDtype.BF16)

    # ── Audio: same as standard — partial 1D tables, 34 tokens × 24 heads ─────
    var aud_rows = _AUD * _NH   # 816
    var aud_pos = List[Float32]()
    for rr in range(aud_rows):
        aud_pos.append(Float32(rr // _NH) * Float32(0.24))
    var aud_positions = Tensor.from_host(aud_pos^, [aud_rows], STDtype.F32, ctx)
    var aud_axes = [44]
    var aud_cs = build_multiaxis_rope_tables(aud_positions, aud_axes, Float32(10000.0), ctx, STDtype.BF16)

    return NavaRope(vid_cs^, aud_cs^)


def nava_double_block_hires(
    x: Tensor,          # [1, 1984, 3072]  BF16
    e_vid: Tensor,      # [1, 1950, 6, 3072] BF16
    e_audio: Tensor,    # [1, 34, 6, 3072]   BF16
    context: Tensor,    # [1, 512, 3072]    BF16
    w: Dict[String, ArcPointer[Tensor]],
    rope: NavaRope,
    ctx: DeviceContext,
    masking_modality: Bool = False,
) raises -> Tensor:
    """NAVA WanDoubleStreamAttentionBlock forward — hi-res (VID=1950, SEQ=1984)."""
    var EPS   = Float32(1e-6)
    var SCALE = Float32(0.08838834764831843)  # 1/sqrt(128)

    # ── 1) Split x into video and audio streams ──────────────────────────────
    var x_vid   = slice(x, 1, 0,         _VID_HR, ctx)   # [1,1950,3072]
    var x_audio = slice(x, 1, _VID_HR,   _AUD,    ctx)   # [1,34,3072]

    # ── 2) ModulationAdd + chunk(6) ──────────────────────────────────────────
    var ev0 = _mod_chunk(e_vid,   w["modulation.modulation"][],       0, _VID_HR, _DIM, ctx)
    var ev1 = _mod_chunk(e_vid,   w["modulation.modulation"][],       1, _VID_HR, _DIM, ctx)
    var ev2 = _mod_chunk(e_vid,   w["modulation.modulation"][],       2, _VID_HR, _DIM, ctx)
    var ev3 = _mod_chunk(e_vid,   w["modulation.modulation"][],       3, _VID_HR, _DIM, ctx)
    var ev4 = _mod_chunk(e_vid,   w["modulation.modulation"][],       4, _VID_HR, _DIM, ctx)
    var ev5 = _mod_chunk(e_vid,   w["modulation.modulation"][],       5, _VID_HR, _DIM, ctx)

    var ea0 = _mod_chunk(e_audio, w["modulation_audio.modulation"][], 0, _AUD, _DIM, ctx)
    var ea1 = _mod_chunk(e_audio, w["modulation_audio.modulation"][], 1, _AUD, _DIM, ctx)
    var ea2 = _mod_chunk(e_audio, w["modulation_audio.modulation"][], 2, _AUD, _DIM, ctx)
    var ea3 = _mod_chunk(e_audio, w["modulation_audio.modulation"][], 3, _AUD, _DIM, ctx)
    var ea4 = _mod_chunk(e_audio, w["modulation_audio.modulation"][], 4, _AUD, _DIM, ctx)
    var ea5 = _mod_chunk(e_audio, w["modulation_audio.modulation"][], 5, _AUD, _DIM, ctx)

    # ── 3) adaLN norm1 ───────────────────────────────────────────────────────
    var xvn_norm = layer_norm_no_affine(x_vid,   EPS, ctx)
    var xan_norm = layer_norm_no_affine(x_audio, EPS, ctx)
    var xvn = add(mul(xvn_norm, add_scalar(ev1, 1.0, ctx), ctx), ev0, ctx)
    var xan = add(mul(xan_norm, add_scalar(ea1, 1.0, ctx), ctx), ea0, ctx)

    # ── 4) Joint self-attention ───────────────────────────────────────────────
    # Video Q,K,V
    var qv_lin = _nava_lin(xvn, w, "self_attn.q", ctx)
    var kv_lin = _nava_lin(xvn, w, "self_attn.k", ctx)
    var vv_lin = _nava_lin(xvn, w, "self_attn.v", ctx)
    var qv_norm = rms_norm(qv_lin, w["self_attn.norm_q.weight"][], EPS, ctx)
    var kv_norm = rms_norm(kv_lin, w["self_attn.norm_k.weight"][], EPS, ctx)
    var qv4 = reshape(qv_norm, [1, _VID_HR, _NH, _HD], ctx)
    var kv4 = reshape(kv_norm, [1, _VID_HR, _NH, _HD], ctx)
    var vv4 = reshape(vv_lin,  [1, _VID_HR, _NH, _HD], ctx)

    # Audio Q,K,V
    var qa_lin = _nava_lin(xan, w, "self_attn.q_audio", ctx)
    var ka_lin = _nava_lin(xan, w, "self_attn.k_audio", ctx)
    var va_lin = _nava_lin(xan, w, "self_attn.v_audio", ctx)
    var qa_norm = rms_norm(qa_lin, w["self_attn.norm_q_audio.weight"][], EPS, ctx)
    var ka_norm = rms_norm(ka_lin, w["self_attn.norm_k_audio.weight"][], EPS, ctx)
    var qa4 = reshape(qa_norm, [1, _AUD, _NH, _HD], ctx)
    var ka4 = reshape(ka_norm, [1, _AUD, _NH, _HD], ctx)
    var va4 = reshape(va_lin,  [1, _AUD, _NH, _HD], ctx)

    # RoPE video
    var qv_roped = rope_interleaved(qv4, rope.vid[0], rope.vid[1], ctx)
    var kv_roped = rope_interleaved(kv4, rope.vid[0], rope.vid[1], ctx)

    # RoPE audio (partial, first 44 dims)
    var qa_rot  = slice(qa4, 3, 0,  44, ctx)
    var qa_pass = slice(qa4, 3, 44, 84, ctx)
    var ka_rot  = slice(ka4, 3, 0,  44, ctx)
    var ka_pass = slice(ka4, 3, 44, 84, ctx)
    var qa_rotd = rope_interleaved(qa_rot, rope.aud[0], rope.aud[1], ctx)
    var ka_rotd = rope_interleaved(ka_rot, rope.aud[0], rope.aud[1], ctx)
    var qa_roped = concat(3, ctx, qa_rotd, qa_pass)
    var ka_roped = concat(3, ctx, ka_rotd, ka_pass)

    # Self-attention
    var yv: Tensor
    var ya: Tensor
    if masking_modality:
        var att_vid = sdpa_nomask[1, _VID_HR, _NH, _HD](qv_roped, kv_roped, vv4, SCALE, ctx)
        var att_aud = sdpa_nomask[1, _AUD,    _NH, _HD](qa_roped, ka_roped, va4, SCALE, ctx)
        yv = _nava_lin(reshape(att_vid, [1, _VID_HR, _DIM], ctx),
                       w, "self_attn.o", ctx)
        ya = _nava_lin(reshape(att_aud, [1, _AUD,    _DIM], ctx),
                       w, "self_attn.o_audio", ctx)
    else:
        var q_joint = concat(1, ctx, qv_roped, qa_roped)
        var k_joint = concat(1, ctx, kv_roped, ka_roped)
        var v_joint = concat(1, ctx, vv4,      va4)
        var att = sdpa_nomask_tiled[1, _SEQ_HR, _NH, _HD](q_joint, k_joint, v_joint, SCALE, ctx)
        var att_vid = slice(att, 1, 0,       _VID_HR, ctx)
        var att_aud = slice(att, 1, _VID_HR, _AUD,    ctx)
        yv = _nava_lin(reshape(att_vid, [1, _VID_HR, _DIM], ctx),
                       w, "self_attn.o", ctx)
        ya = _nava_lin(reshape(att_aud, [1, _AUD,    _DIM], ctx),
                       w, "self_attn.o_audio", ctx)

    # gate_msa residual
    x_vid   = add(x_vid,   mul(yv, ev2, ctx), ctx)
    x_audio = add(x_audio, mul(ya, ea2, ctx), ctx)

    # ── 5) Cross-attention to text ───────────────────────────────────────────
    var xvn3 = layer_norm(x_vid,   w["norm3.weight"][], w["norm3.bias"][], EPS, ctx)
    var xan3 = layer_norm(x_audio, w["norm3.weight"][], w["norm3.bias"][], EPS, ctx)

    var qv3   = _nava_lin(xvn3,    w, "cross_attn.q", ctx)
    var kc    = _nava_lin(context, w, "cross_attn.k", ctx)
    var vc    = _nava_lin(context, w, "cross_attn.v", ctx)
    var qv3_n = rms_norm(qv3, w["cross_attn.norm_q.weight"][], EPS, ctx)
    var kc_n  = rms_norm(kc,  w["cross_attn.norm_k.weight"][], EPS, ctx)
    var qv3_4 = reshape(qv3_n, [1, _VID_HR, _NH, _HD], ctx)
    var kc_4  = reshape(kc_n,  [1, _CTX, _NH, _HD], ctx)
    var vc_4  = reshape(vc,    [1, _CTX, _NH, _HD], ctx)
    var cvid  = sdpa_cross_nomask[1, _VID_HR, _CTX, _NH, _HD](qv3_4, kc_4, vc_4, SCALE, ctx)
    var xva   = _nava_lin(reshape(cvid, [1, _VID_HR, _DIM], ctx),
                          w, "cross_attn.o", ctx)

    var qa3   = _nava_lin(xan3,    w, "cross_attn.q_audio", ctx)
    var kca   = _nava_lin(context, w, "cross_attn.k_audio", ctx)
    var vca   = _nava_lin(context, w, "cross_attn.v_audio", ctx)
    var qa3_n = rms_norm(qa3, w["cross_attn.norm_q_audio.weight"][], EPS, ctx)
    var kca_n = rms_norm(kca, w["cross_attn.norm_k_audio.weight"][], EPS, ctx)
    var qa3_4 = reshape(qa3_n, [1, _AUD, _NH, _HD], ctx)
    var kca_4 = reshape(kca_n, [1, _CTX, _NH, _HD], ctx)
    var vca_4 = reshape(vca,   [1, _CTX, _NH, _HD], ctx)
    var caud  = sdpa_cross_nomask[1, _AUD, _CTX, _NH, _HD](qa3_4, kca_4, vca_4, SCALE, ctx)
    var xaa   = _nava_lin(reshape(caud, [1, _AUD, _DIM], ctx),
                          w, "cross_attn.o_audio", ctx)

    x_vid   = add(x_vid,   xva, ctx)
    x_audio = add(x_audio, xaa, ctx)

    # ── 6) FFN ───────────────────────────────────────────────────────────────
    var xvn2     = layer_norm_no_affine(x_vid, EPS, ctx)
    var xvn2_mod = add(mul(xvn2, add_scalar(ev4, 1.0, ctx), ctx), ev3, ctx)
    var h_vid    = _nava_lin(xvn2_mod, w, "ffn.0", ctx)
    var hg_vid   = gelu(h_vid, ctx)
    var yv_ffn   = _nava_lin(hg_vid, w, "ffn.2", ctx)
    x_vid        = add(x_vid, mul(yv_ffn, ev5, ctx), ctx)

    var xan2     = layer_norm_no_affine(x_audio, EPS, ctx)
    var xan2_mod = add(mul(xan2, add_scalar(ea4, 1.0, ctx), ctx), ea3, ctx)
    var h_aud    = _nava_lin(xan2_mod, w, "ffn.0", ctx)
    var hg_aud   = gelu(h_aud, ctx)
    var ya_ffn   = _nava_lin(hg_aud, w, "ffn.2", ctx)
    x_audio      = add(x_audio, mul(ya_ffn, ea5, ctx), ctx)

    # ── 7) Concat -> [1, 1984, 3072] ────────────────────────────────────────
    return concat(1, ctx, x_vid, x_audio)


def nava_single_block_hires(
    x: Tensor,          # [1, 1984, 3072]  BF16
    e_vid: Tensor,      # [1, 1950, 6, 3072] BF16
    e_audio: Tensor,    # [1, 34, 6, 3072]   BF16
    context: Tensor,    # [1, 512, 3072]    BF16
    w: Dict[String, ArcPointer[Tensor]],
    rope: NavaRope,
    ctx: DeviceContext,
    masking_modality: Bool = False,
) raises -> Tensor:
    """NAVA WanAttentionBlock forward — hi-res (VID=1950, SEQ=1984)."""
    var EPS   = Float32(1e-6)
    var SCALE = Float32(0.08838834764831843)  # 1/sqrt(128)

    # ── 1) Concat e_vid and e_audio along token dim ──────────────────────────
    var e_cat = concat(1, ctx, e_vid, e_audio)   # [1, 1984, 6, 3072]

    # ── 2) ModulationAdd + chunk(6) over the combined 1984 tokens ─────────────
    var e0 = _mod_chunk(e_cat, w["modulation.modulation"][], 0, _SEQ_HR, _DIM, ctx)
    var e1 = _mod_chunk(e_cat, w["modulation.modulation"][], 1, _SEQ_HR, _DIM, ctx)
    var e2 = _mod_chunk(e_cat, w["modulation.modulation"][], 2, _SEQ_HR, _DIM, ctx)
    var e3 = _mod_chunk(e_cat, w["modulation.modulation"][], 3, _SEQ_HR, _DIM, ctx)
    var e4 = _mod_chunk(e_cat, w["modulation.modulation"][], 4, _SEQ_HR, _DIM, ctx)
    var e5 = _mod_chunk(e_cat, w["modulation.modulation"][], 5, _SEQ_HR, _DIM, ctx)

    # ── 3) adaLN norm1 ───────────────────────────────────────────────────────
    var xn_norm = layer_norm_no_affine(x, EPS, ctx)
    var xn = add(mul(xn_norm, add_scalar(e1, 1.0, ctx), ctx), e0, ctx)

    # ── 4) Joint self-attention ───────────────────────────────────────────────
    var q_lin  = _nava_lin(xn, w, "self_attn.q", ctx)
    var k_lin  = _nava_lin(xn, w, "self_attn.k", ctx)
    var v_lin  = _nava_lin(xn, w, "self_attn.v", ctx)
    var q_norm = rms_norm(q_lin, w["self_attn.norm_q.weight"][], EPS, ctx)
    var k_norm = rms_norm(k_lin, w["self_attn.norm_k.weight"][], EPS, ctx)
    var q4     = reshape(q_norm, [1, _SEQ_HR, _NH, _HD], ctx)
    var k4     = reshape(k_norm, [1, _SEQ_HR, _NH, _HD], ctx)
    var v4     = reshape(v_lin,  [1, _SEQ_HR, _NH, _HD], ctx)

    # RoPE: video tokens 0.._VID_HR-1, audio tokens _VID_HR.._SEQ_HR-1
    var qv4_s = slice(q4, 1, 0,       _VID_HR, ctx)
    var qa4_s = slice(q4, 1, _VID_HR, _AUD,    ctx)
    var kv4_s = slice(k4, 1, 0,       _VID_HR, ctx)
    var ka4_s = slice(k4, 1, _VID_HR, _AUD,    ctx)

    var qv_roped = rope_interleaved(qv4_s, rope.vid[0], rope.vid[1], ctx)
    var kv_roped = rope_interleaved(kv4_s, rope.vid[0], rope.vid[1], ctx)

    var qa_rot   = slice(qa4_s, 3, 0,  44, ctx)
    var qa_pass  = slice(qa4_s, 3, 44, 84, ctx)
    var ka_rot   = slice(ka4_s, 3, 0,  44, ctx)
    var ka_pass  = slice(ka4_s, 3, 44, 84, ctx)
    var qa_rotd  = rope_interleaved(qa_rot, rope.aud[0], rope.aud[1], ctx)
    var ka_rotd  = rope_interleaved(ka_rot, rope.aud[0], rope.aud[1], ctx)
    var qa_roped = concat(3, ctx, qa_rotd, qa_pass)
    var ka_roped = concat(3, ctx, ka_rotd, ka_pass)

    # Self-attention
    var x1: Tensor
    if masking_modality:
        var vv_s = slice(v4, 1, 0,       _VID_HR, ctx)
        var va_s = slice(v4, 1, _VID_HR, _AUD,    ctx)
        var att_vid = sdpa_nomask[1, _VID_HR, _NH, _HD](qv_roped, kv_roped, vv_s, SCALE, ctx)
        var att_aud = sdpa_nomask[1, _AUD,    _NH, _HD](qa_roped, ka_roped, va_s, SCALE, ctx)
        var att = concat(1, ctx, att_vid, att_aud)   # [1,1984,24,128]
        var y_sa = _nava_lin(reshape(att, [1, _SEQ_HR, _DIM], ctx), w, "self_attn.o", ctx)
        x1 = add(x, mul(y_sa, e2, ctx), ctx)
    else:
        var q_roped = concat(1, ctx, qv_roped, qa_roped)
        var k_roped = concat(1, ctx, kv_roped, ka_roped)
        var att = sdpa_nomask_tiled[1, _SEQ_HR, _NH, _HD](q_roped, k_roped, v4, SCALE, ctx)
        var y_sa = _nava_lin(reshape(att, [1, _SEQ_HR, _DIM], ctx), w, "self_attn.o", ctx)
        x1 = add(x, mul(y_sa, e2, ctx), ctx)

    # ── 5) Cross-attention ───────────────────────────────────────────────────
    var xn3 = layer_norm(x1, w["norm3.weight"][], w["norm3.bias"][], EPS, ctx)
    var qc_lin = _nava_lin(xn3,     w, "cross_attn.q", ctx)
    var kc_lin = _nava_lin(context, w, "cross_attn.k", ctx)
    var vc_lin = _nava_lin(context, w, "cross_attn.v", ctx)
    var qc_n   = rms_norm(qc_lin, w["cross_attn.norm_q.weight"][], EPS, ctx)
    var kc_n   = rms_norm(kc_lin, w["cross_attn.norm_k.weight"][], EPS, ctx)
    var qc4    = reshape(qc_n, [1, _SEQ_HR, _NH, _HD], ctx)
    var kc4    = reshape(kc_n, [1, _CTX,    _NH, _HD], ctx)
    var vc4    = reshape(vc_lin, [1, _CTX,  _NH, _HD], ctx)
    var cx     = sdpa_cross_nomask[1, _SEQ_HR, _CTX, _NH, _HD](qc4, kc4, vc4, SCALE, ctx)
    var y_ca   = _nava_lin(reshape(cx, [1, _SEQ_HR, _DIM], ctx), w, "cross_attn.o", ctx)
    var x2 = add(x1, y_ca, ctx)

    # ── 6) FFN ───────────────────────────────────────────────────────────────
    var xn2     = layer_norm_no_affine(x2, EPS, ctx)
    var xn2_mod = add(mul(xn2, add_scalar(e4, 1.0, ctx), ctx), e3, ctx)
    var h_ffn   = _nava_lin(xn2_mod, w, "ffn.0", ctx)
    var hg_ffn  = gelu(h_ffn, ctx)
    var y_ffn   = _nava_lin(hg_ffn, w, "ffn.2", ctx)
    return add(x2, mul(y_ffn, e5, ctx), ctx)^
