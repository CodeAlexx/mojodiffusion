# models/dit/ltx2_connector.mojo — LTX-2 Embeddings1DConnector (pure Mojo+MAX).
#
# Port of inference-flame/src/models/ltx2_model.rs:
#   VideoEmbeddingsConnector::forward  (line 2487) -> LTX2Connector.forward
#   ConnectorBlock::forward            (line 2443) -> _connector_block_forward
#   compute_rope_frequencies (1D)      (line 373)  -> build_connector_rope_1d
#   apply_rotary_emb                   (line 492)  -> apply_ltx2_rope (reused)
#   LTX2Attention::forward (gated)     (line 739)  -> _connector_attention
#   FeedForward::forward               (line 610)  -> reuse linear+gelu
#   rms_norm                           (line 292)  -> reuse ops/norm rms_norm
#
# This is P2.5 (LTX2_PORT_PLAN_2026-05-28.md): the cached embeds are
# PRE-connector (`text_hidden [1,1024,4096]`, the cached video_context). Before
# any joint-AV block we must run the in-Mojo connector on them. The connector
# REPLACES caption_projection in the ComfyUI LTX-2.3 checkpoint — there is NO
# separate caption_projection / audio_caption_projection in this checkpoint
# (verified: 0 such keys); the connector IS the projection. Its 4096-dim output
# is the video context; the audio connector (2048-dim) produces the audio
# context (audio context is PROJECTED, not a sidecar — plan P6).
#
# ── Connector block (NO AdaLN; differs from the DiT block) ───────────────────
#   x  = x + attn(  rms_norm_no_affine(x) )          (gated self-attn + 1D RoPE)
#   x  = x + ff(    rms_norm_no_affine(x) )           (GELU-approx FFN)
# Final: rms_norm_no_affine(x).   (ltx2_model.rs:2443-2452, 2599-2601)
#
# ── 1D split-RoPE (differs from the 3D video RoPE) ───────────────────────────
#   num_pos_dims = 1 ; max_pos = connector_positional_embedding_max_pos = 4096
#   position = i (token index, midpoint of (i,i)) ; freq_count = inner_dim/2
#   rope_freqs = freq_count == half_dim -> NO front padding.
#   grid = i/max_pos ; scaled = 2*grid-1 ; angle[i] = scaled*freq[i].
#   freq[i] = theta^(i/max(freq_count-1,1)) * pi/2   (ltx2_model.rs:405-408)
#   half-vector reshaped half_dim -> [num_heads, head_dim/2] (contiguous slice).
#
# ── Attention (gated, ltx2_model.rs:739) ─────────────────────────────────────
#   q,k,v = linear(x) ; q=rms_norm(q,q_norm) ; k=rms_norm(k,k_norm)
#   q,k = split-RoPE ; attn = sdpa(q,k,v) ; if to_gate_logits:
#       gates = 2*sigmoid(linear(x, gate_w, gate_b)) ; attn *= gates per head
#   out = linear(attn, to_out.0)
#
# Mojo 1.0.0b1, NVIDIA GPU. Numeric gate: P2.5 smoke vs the Python connector
# reference (scripts/ltx2_dit_forward_parity_ref.py), cos >= 0.999.

from std.gpu.host import DeviceContext
from std.math import sqrt, cos as fcos, sin as fsin, pow as fpow, pi
from std.memory import ArcPointer

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.linear import linear
from serenitymojo.ops.norm import rms_norm
from serenitymojo.ops.activations import gelu, sigmoid
from serenitymojo.ops.attention import sdpa_nomask
from serenitymojo.ops.tensor_algebra import (
    reshape,
    add,
    mul,
    mul_scalar,
)
from serenitymojo.ops.rope import rope_halfsplit
from serenitymojo.ops.cast import cast_tensor


# ── Config ────────────────────────────────────────────────────────────────────
@fieldwise_init
struct LTX2ConnectorConfig(Copyable, Movable, ImplicitlyCopyable):
    """One connector's hyperparameters (video or audio).

    video: inner_dim=4096, heads=32, head_dim=128, ffn=16384, 8 blocks.
    audio: inner_dim=2048, heads=32, head_dim=64,  ffn=8192,  8 blocks.
    """

    var num_blocks: Int
    var num_heads: Int
    var head_dim: Int
    var inner_dim: Int       # num_heads * head_dim
    var rope_theta: Float64  # 10000.0
    var rope_max_pos: Float64  # connector_positional_embedding_max_pos = 4096
    var eps: Float32         # 1e-6

    @staticmethod
    def video() -> LTX2ConnectorConfig:
        return LTX2ConnectorConfig(
            8, 32, 128, 4096, Float64(10000.0), Float64(4096.0), Float32(1e-6)
        )

    @staticmethod
    def audio() -> LTX2ConnectorConfig:
        return LTX2ConnectorConfig(
            8, 32, 64, 2048, Float64(10000.0), Float64(4096.0), Float32(1e-6)
        )


# ── Weights ─────────────────────────────────────────────────────────────────
struct LTX2ConnectorWeights(Movable):
    """All N connector-block weights for one connector, looked up by canonical
    key `b{i}.{leaf}` (e.g. `b0.attn1.to_q.weight`). Uploaded BF16."""

    var weights: List[ArcPointer[Tensor]]
    var name_to_idx: Dict[String, Int]
    var config: LTX2ConnectorConfig

    def __init__(
        out self,
        var weights: List[ArcPointer[Tensor]],
        var name_to_idx: Dict[String, Int],
        config: LTX2ConnectorConfig,
    ):
        self.weights = weights^
        self.name_to_idx = name_to_idx^
        self.config = config

    @staticmethod
    def load(
        checkpoint_path: String,
        connector_prefix: String,  # e.g. "video_embeddings_connector"
        config: LTX2ConnectorConfig,
        ctx: DeviceContext,
    ) raises -> LTX2ConnectorWeights:
        """Load all transformer_1d_blocks for one connector. Tries the diffusers
        prefix (`{connector}.`) first, then the ComfyUI prefix
        (`model.diffusion_model.{connector}.`). Attn QK-norm key is `q_norm`/
        `k_norm` (ComfyUI) or `norm_q`/`norm_k` (diffusers)."""
        var st = ShardedSafeTensors.open(checkpoint_path)

        var pfx_diff = connector_prefix + "."
        var pfx_comfy = String("model.diffusion_model.") + connector_prefix + "."
        var base = pfx_diff
        var probe = String("transformer_1d_blocks.0.attn1.to_q.weight")
        if not _st_has(st, pfx_diff + probe):
            if _st_has(st, pfx_comfy + probe):
                base = pfx_comfy
            else:
                raise Error(
                    String("LTX2 connector '")
                    + connector_prefix
                    + "' not found under either prefix"
                )

        # QK-norm naming.
        var qn = String("q_norm.weight")
        var kn = String("k_norm.weight")
        if not _st_has(
            st, base + "transformer_1d_blocks.0.attn1." + qn
        ):
            qn = String("norm_q.weight")
            kn = String("norm_k.weight")

        var weights = List[ArcPointer[Tensor]]()
        var name_to_idx = Dict[String, Int]()

        for i in range(config.num_blocks):
            var bp = base + "transformer_1d_blocks." + String(i) + "."
            var canon = String("b") + String(i) + "."

            # canonical-leaf -> source-leaf
            var leaves = List[Tuple[String, String]]()
            leaves.append(("attn1.to_q.weight", "attn1.to_q.weight"))
            leaves.append(("attn1.to_q.bias", "attn1.to_q.bias"))
            leaves.append(("attn1.to_k.weight", "attn1.to_k.weight"))
            leaves.append(("attn1.to_k.bias", "attn1.to_k.bias"))
            leaves.append(("attn1.to_v.weight", "attn1.to_v.weight"))
            leaves.append(("attn1.to_v.bias", "attn1.to_v.bias"))
            leaves.append(("attn1.q_norm.weight", "attn1." + qn))
            leaves.append(("attn1.k_norm.weight", "attn1." + kn))
            leaves.append(("attn1.to_out.0.weight", "attn1.to_out.0.weight"))
            leaves.append(("attn1.to_out.0.bias", "attn1.to_out.0.bias"))
            leaves.append(("ff.net.0.proj.weight", "ff.net.0.proj.weight"))
            leaves.append(("ff.net.0.proj.bias", "ff.net.0.proj.bias"))
            leaves.append(("ff.net.2.weight", "ff.net.2.weight"))
            leaves.append(("ff.net.2.bias", "ff.net.2.bias"))

            for ref lv in leaves:
                var src = bp + lv[1]
                var tv = st.tensor_view(src)
                name_to_idx[canon + lv[0]] = len(weights)
                weights.append(ArcPointer(Tensor.from_view_as_bf16(tv, ctx)))

            # Optional per-head gate (present in ComfyUI 22B for both connectors).
            if _st_has(st, bp + "attn1.to_gate_logits.weight"):
                var gw = st.tensor_view(bp + "attn1.to_gate_logits.weight")
                name_to_idx[canon + "attn1.to_gate_logits.weight"] = len(weights)
                weights.append(ArcPointer(Tensor.from_view_as_bf16(gw, ctx)))
                var gb = st.tensor_view(bp + "attn1.to_gate_logits.bias")
                name_to_idx[canon + "attn1.to_gate_logits.bias"] = len(weights)
                weights.append(ArcPointer(Tensor.from_view_as_bf16(gb, ctx)))

        return LTX2ConnectorWeights(weights^, name_to_idx^, config)

    def _has(self, name: String) -> Bool:
        return name in self.name_to_idx

    def _w(self, name: String) raises -> ref [self.weights] Tensor:
        if name not in self.name_to_idx:
            raise Error(String("LTX2 connector: missing weight ") + name)
        return self.weights[self.name_to_idx[name]][]

    def to_f32(self, ctx: DeviceContext) raises -> LTX2ConnectorWeights:
        """Return a copy with every connector weight cast to F32.

        This is for Python-oracle parity gates that intentionally run the whole
        connector/DiT stack in F32. The production entrypoints keep BF16 storage
        and rely on F32 accumulation inside kernels.
        """
        var weights = List[ArcPointer[Tensor]]()
        for ref w in self.weights:
            weights.append(ArcPointer(cast_tensor(w[], STDtype.F32, ctx)))
        return LTX2ConnectorWeights(weights^, self.name_to_idx.copy(), self.config)

    def _clone(self, x: Tensor, ctx: DeviceContext) raises -> Tensor:
        return x.clone(ctx)

    def _linear_b(
        self, x: Tensor, w_key: String, b_key: String, ctx: DeviceContext
    ) raises -> Tensor:
        ref w = self._w(w_key)
        ref b = self._w(b_key)
        return linear(x, w, Optional[Tensor](self._clone(b, ctx)), ctx)


# ── helpers ───────────────────────────────────────────────────────────────────
def _st_has(st: ShardedSafeTensors, name: String) -> Bool:
    for ref nm in st.names():
        if nm == name:
            return True
    return False


def _shape1(a: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a)
    return s^


def _shape3(a: Int, b: Int, c: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a)
    s.append(b)
    s.append(c)
    return s^


def _shape4(a: Int, b: Int, c: Int, d: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a)
    s.append(b)
    s.append(c)
    s.append(d)
    return s^


# RMSNorm with NO affine (ones weight) over the last dim — connector norm1/norm2
# and the final norm have no weight in the checkpoint (ltx2_model.rs uses
# rms_norm(x, None, eps) for those). Preserves the input dtype; kernels
# accumulate in F32 and store back to the hidden dtype.
def _rms_norm_no_affine(
    x: Tensor, eps: Float32, ctx: DeviceContext
) raises -> Tensor:
    var d = x.shape()[len(x.shape()) - 1]
    var ones = List[Float32]()
    for _ in range(d):
        ones.append(Float32(1.0))
    var w = Tensor.from_host(ones, _shape1(d), x.dtype(), ctx)
    return rms_norm(x, w, eps, ctx)


def _to_bshd(x: Tensor, n: Int, h: Int, dh: Int, ctx: DeviceContext) raises -> Tensor:
    return reshape(x, _shape4(1, n, h, dh), ctx)


def _from_bshd(x: Tensor, n: Int, h: Int, dh: Int, ctx: DeviceContext) raises -> Tensor:
    return reshape(x, _shape3(1, n, h * dh), ctx)


# ── 1D connector RoPE table (port of compute_rope_frequencies, num_pos_dims=1) ─
# Builds per (token, head) cos/sin tables shaped [N*H, head_dim/2] in BSHD-flat
# (token, head) row order so rope_halfsplit on a [1,N,H,Dh] tensor lines up.
def build_connector_rope_1d(
    seq_len: Int,
    num_heads: Int,
    head_dim: Int,
    theta: Float64,
    max_pos: Float64,
    dtype: STDtype,
    ctx: DeviceContext,
) raises -> Tuple[Tensor, Tensor]:
    var inner_dim = num_heads * head_dim
    var half_dim = inner_dim // 2
    var half_head = head_dim // 2
    # 1D: num_rope_elems = 2 ; freq_count = inner_dim/2 ; rope_freqs = freq_count
    # == half_dim -> no front padding.
    var freq_count = inner_dim // 2
    if freq_count != half_dim:
        raise Error("connector rope: freq_count != half_dim (1D invariant)")
    if num_heads * half_head != half_dim:
        raise Error("connector rope: inner_dim/2 != heads * head_dim/2")

    var denom = Float64(freq_count - 1)
    if denom < 1.0:
        denom = 1.0
    var freq = List[Float32]()
    for i in range(freq_count):
        var t = Float64(i) / denom
        freq.append(Float32(fpow(theta, t) * pi / 2.0))

    # Per-token half-vector [seq, half_dim], then reshape to (token, head).
    var cos_rows = List[Float32]()
    var sin_rows = List[Float32]()
    for tok in range(seq_len):
        # midpoint of (i,i) == i ; grid = i/max_pos ; scaled = 2*grid - 1
        var grid = Float64(tok) / max_pos
        var scaled = 2.0 * grid - 1.0
        for h in range(num_heads):
            var base = h * half_head
            for j in range(half_head):
                var a = scaled * Float64(freq[base + j])
                cos_rows.append(Float32(fcos(a)))
                sin_rows.append(Float32(fsin(a)))

    var sh = List[Int]()
    sh.append(seq_len * num_heads)
    sh.append(half_head)
    var cos_t = Tensor.from_host(cos_rows, sh.copy(), dtype, ctx)
    var sin_t = Tensor.from_host(sin_rows, sh^, dtype, ctx)
    return (cos_t^, sin_t^)


# ── per-head gate (ltx2_model.rs:858-872, connector variant) ─────────────────
# gate input is the (un-modulated) block-attention input x: gates =
# 2*sigmoid(linear(x, gate_w, gate_b)) -> [1,N,H]; multiply attn [1,N,H,Dh].
def _apply_head_gate(
    attn_flat: Tensor,   # [1, N, H*Dh]
    gate_src: Tensor,    # [1, N, H*Dh]
    gate_w: Tensor,      # [H, H*Dh]
    var gate_b: Tensor,  # [H]
    seq_len: Int,
    num_heads: Int,
    head_dim: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    var gate_logits = linear(gate_src, gate_w, Optional[Tensor](gate_b^), ctx)  # [1,N,H]
    var gates = mul_scalar(sigmoid(gate_logits, ctx), Float32(2.0), ctx)        # [1,N,H]
    var gates4 = reshape(gates, _shape4(1, seq_len, num_heads, 1), ctx)
    var attn4 = reshape(attn_flat, _shape4(1, seq_len, num_heads, head_dim), ctx)
    var gated = mul(attn4, gates4, ctx)  # broadcast [1,N,H,1] over [1,N,H,Dh]
    return reshape(gated, _shape3(1, seq_len, num_heads * head_dim), ctx)


# ── connector gated self-attention ───────────────────────────────────────────
def _connector_attention[N: Int, H: Int, Dh: Int](
    weights: LTX2ConnectorWeights,
    x: Tensor,           # [1, N, inner] (already rms_norm'd by the block)
    canon_attn: String,  # e.g. "b0.attn1."
    eps: Float32,
    rope_cos: Tensor,
    rope_sin: Tensor,
    ctx: DeviceContext,
) raises -> Tensor:
    var q = weights._linear_b(x, canon_attn + "to_q.weight", canon_attn + "to_q.bias", ctx)
    var k = weights._linear_b(x, canon_attn + "to_k.weight", canon_attn + "to_k.bias", ctx)
    var v = weights._linear_b(x, canon_attn + "to_v.weight", canon_attn + "to_v.bias", ctx)

    q = rms_norm(q, weights._w(canon_attn + "q_norm.weight"), eps, ctx)
    k = rms_norm(k, weights._w(canon_attn + "k_norm.weight"), eps, ctx)

    var q4 = _to_bshd(q, N, H, Dh, ctx)
    var k4 = _to_bshd(k, N, H, Dh, ctx)
    var v4 = _to_bshd(v, N, H, Dh, ctx)

    q4 = rope_halfsplit(q4, rope_cos, rope_sin, ctx)
    k4 = rope_halfsplit(k4, rope_cos, rope_sin, ctx)

    var scale = Float32(1.0) / Float32(sqrt(Float64(Dh)))
    var attn = sdpa_nomask[1, N, H, Dh](q4, k4, v4, scale, ctx)
    var attn_flat = _from_bshd(attn, N, H, Dh, ctx)  # [1, N, inner]

    if weights._has(canon_attn + "to_gate_logits.weight"):
        attn_flat = _apply_head_gate(
            attn_flat,
            x,
            weights._w(canon_attn + "to_gate_logits.weight"),
            weights._clone(weights._w(canon_attn + "to_gate_logits.bias"), ctx),
            N, H, Dh, ctx,
        )

    return weights._linear_b(
        attn_flat, canon_attn + "to_out.0.weight", canon_attn + "to_out.0.bias", ctx
    )


# ── connector block (no AdaLN) ───────────────────────────────────────────────
# Hidden/weight storage follows the incoming dtype (BF16 in the active LTX2
# runtime). Linear, norm, attention, and RoPE kernels still use F32 internal
# accumulation before storing back to that dtype.
def _connector_block_forward[N: Int, H: Int, Dh: Int](
    weights: LTX2ConnectorWeights,
    hidden: Tensor,   # [1, N, inner] BF16 in the active runtime
    block_idx: Int,
    eps: Float32,
    rope_cos: Tensor,
    rope_sin: Tensor,
    ctx: DeviceContext,
) raises -> Tensor:
    var canon = String("b") + String(block_idx) + "."

    # 1. rms_norm (no affine) -> gated self-attn -> residual
    var norm1 = _rms_norm_no_affine(hidden, eps, ctx)
    var attn_out = _connector_attention[N, H, Dh](
        weights, norm1, canon + "attn1.", eps, rope_cos, rope_sin, ctx
    )
    var hs = add(hidden, attn_out, ctx)

    # 2. rms_norm (no affine) -> FFN (GELU-approx) -> residual
    var norm2 = _rms_norm_no_affine(hs, eps, ctx)
    var ff = weights._linear_b(
        norm2, canon + "ff.net.0.proj.weight", canon + "ff.net.0.proj.bias", ctx
    )
    ff = gelu(ff, ctx)  # GELU-approximate (tanh) — ltx2_model.rs:279-281
    ff = weights._linear_b(
        ff, canon + "ff.net.2.weight", canon + "ff.net.2.bias", ctx
    )
    return add(hs, ff, ctx)


# ── connector forward (the public entry) ─────────────────────────────────────
# Runs the connector on the PRE-connector context `x` [1, N, inner] with
# mask=None (whole sequence treated as real, no learnable-register replacement).
# Returns the post-connector context [1, N, inner] (final unweighted RMSNorm).
def ltx2_connector_forward[N: Int, H: Int, Dh: Int](
    weights: LTX2ConnectorWeights,
    x: Tensor,   # [1, N, inner] BF16
    ctx: DeviceContext,
) raises -> Tensor:
    var cfg = weights.config
    if H != cfg.num_heads or Dh != cfg.head_dim:
        raise Error("ltx2_connector_forward: H/Dh != config")

    # RoPE kernels require x/cos/sin dtype parity and accumulate internally in F32.
    var rope = build_connector_rope_1d(
        N, cfg.num_heads, cfg.head_dim, cfg.rope_theta, cfg.rope_max_pos,
        x.dtype(), ctx,
    )
    ref rope_cos = rope[0]
    ref rope_sin = rope[1]

    var h = weights._clone(x, ctx)
    for i in range(cfg.num_blocks):
        h = _connector_block_forward[N, H, Dh](
            weights, h, i, cfg.eps, rope_cos, rope_sin, ctx
        )
    # Final unweighted RMSNorm (ltx2_model.rs:2599-2601).
    return _rms_norm_no_affine(h, cfg.eps, ctx)
