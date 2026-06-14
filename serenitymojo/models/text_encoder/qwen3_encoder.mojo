# models/text_encoder/qwen3_encoder.mojo — Qwen3 text encoder forward (GPU).
#
# Pure-Mojo, inference-only port of inference-flame/src/models/qwen3_encoder.rs
# (the Qwen3 causal-LM text encoder shared by Klein and Z-Image). Reuses the
# Phase-A foundation ops VERBATIM — rms_norm / rope_halfsplit / sdpa / linear /
# swiglu — and adds only ENCODER-LOCAL glue that is not a foundation op:
#   * embedding gather  (token-id -> embedding row)
#   * residual add      (x + y; foundation has only residual_gate(x,gate,y))
#   * GQA kv-head repeat (BSHD [1,N,H_kv,Dh] -> [1,N,H,Dh])
#   * host-side RoPE cos/sin tables  (per (position, head) row order)
#   * host-side additive causal mask
#   * reshape (metadata-only; row-major) — Tensor carries no reshape method
#
# Architecture per layer (qwen3_encoder.rs:354 layer_forward):
#   h -> rms_norm(input_layernorm) -> q/k/v Linear
#     -> q_norm/k_norm (per-head RMSNorm over Dh)
#     -> rope_halfsplit on q,k -> repeat_kv(k,v) -> sdpa(causal) -> o_proj
#     -> residual
#     -> rms_norm(post_attention_layernorm) -> swiglu(gate,up)*down -> residual
# Final: rms_norm(model.norm) applied by the CALLER for last_hidden_state
# (encode() returns the configured extract layer PRE-final-norm, matching the
# Rust encode() which returns hidden states before model.norm — the Rust parity
# test applies model.norm separately, see qwen3_encoder.rs:694-704).
#
# Config (Z-Image text_encoder/config.json, Qwen3ForCausalLM):
#   hidden=2560, layers=36, heads=32, kv_heads=8 (GQA n_rep=4), head_dim=128,
#   rms_norm_eps=1e-6, rope_theta=1e6, hidden_act=silu, intermediate=9728.
# RoPE = HALF-SPLIT (HF rotate_half), NOT interleaved (PHASE_AB_PLAN line 11,
# rope.mojo header "HALFSPLIT (Z-Image)", qwen3_encoder.rs:272).
#
# Mojo 1.0.0b1, NVIDIA GPU. BF16 storage, F32 accumulation in foundation ops.

from std.math import cos as fcos, sin as fsin, exp as fexp, log as flog, sqrt
from std.memory import ArcPointer
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu import global_idx
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.norm import rms_norm
from serenitymojo.ops.rope import rope_halfsplit
from serenitymojo.ops.attention import sdpa
from serenitymojo.ops.tensor_algebra import concat, slice
from serenitymojo.ops.linear import linear
from serenitymojo.ops.activations import swiglu


comptime _DYN1 = Layout.row_major(-1)
comptime _BLOCK = 256


# ── Config ──────────────────────────────────────────────────────────────────
@fieldwise_init
struct Qwen3Config(Copyable, Movable, ImplicitlyCopyable):
    """Qwen3 text-encoder hyperparameters (Z-Image / Klein)."""

    var hidden_size: Int
    var num_layers: Int
    var num_heads: Int
    var num_kv_heads: Int
    var head_dim: Int
    var rms_norm_eps: Float32
    var rope_theta: Float64

    @staticmethod
    def zimage() -> Qwen3Config:
        """Z-Image single-layer extraction config (Qwen3, hidden=2560)."""
        return Qwen3Config(2560, 36, 32, 8, 128, Float32(1e-6), Float64(1e6))

    @staticmethod
    def klein_4b() -> Qwen3Config:
        """Klein 4B text encoder config (Qwen3-4B, hidden=2560)."""
        return Qwen3Config(2560, 36, 32, 8, 128, Float32(1e-6), Float64(1e6))

    @staticmethod
    def klein_9b() -> Qwen3Config:
        """Klein 9B text encoder config (Qwen3-8B, hidden=4096)."""
        return Qwen3Config(4096, 36, 32, 8, 128, Float32(1e-6), Float64(1e6))

    @staticmethod
    def qwen3_06b() -> Qwen3Config:
        """Qwen3-0.6B text encoder config (Anima text path).
        hidden=1024, layers=28, heads=16, kv_heads=8 (GQA n_rep=2),
        head_dim=128, eps=1e-6, theta=1e6. Verified from
        qwen_3_06b_base.safetensors (q_proj[2048,1024]=16x128;
        k_proj[1024,1024]=8x128)."""
        return Qwen3Config(1024, 28, 16, 8, 128, Float32(1e-6), Float64(1e6))


def klein_extract_layers() -> List[Int]:
    """Klein conditioning layers, 0-indexed: [8, 17, 26]."""
    var out = List[Int]()
    out.append(8)
    out.append(17)
    out.append(26)
    return out^


# ── encoder-local glue kernels (NOT foundation ops) ─────────────────────────
#
# Embedding gather: out[t, j] = table[ ids[t], j ]. One thread per output
# element. `ids` is an I32 device buffer (token ids); `table` is [vocab, hidden]
# in the compute dtype. BF16-path only matters at the store boundary; we copy
# the raw element bytes (no arithmetic), so a single uint16/uint32-width copy is
# dtype-exact. We implement per-dtype to keep the element width correct.
def _embed_kernel_bf16(
    table: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    ids: LayoutTensor[DType.int32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    seq: Int,
    hidden: Int,
):
    var idx = Int(global_idx.x)
    var total = seq * hidden
    if idx < total:
        var t = idx // hidden
        var j = idx % hidden
        var tok = Int(rebind[Scalar[DType.int32]](ids[t]))
        o[idx] = rebind[o.element_type](table[tok * hidden + j])


def _embed_kernel_f32(
    table: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    ids: LayoutTensor[DType.int32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    seq: Int,
    hidden: Int,
):
    var idx = Int(global_idx.x)
    var total = seq * hidden
    if idx < total:
        var t = idx // hidden
        var j = idx % hidden
        var tok = Int(rebind[Scalar[DType.int32]](ids[t]))
        o[idx] = rebind[o.element_type](table[tok * hidden + j])


def _embed_kernel_f16(
    table: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    ids: LayoutTensor[DType.int32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    seq: Int,
    hidden: Int,
):
    var idx = Int(global_idx.x)
    var total = seq * hidden
    if idx < total:
        var t = idx // hidden
        var j = idx % hidden
        var tok = Int(rebind[Scalar[DType.int32]](ids[t]))
        o[idx] = rebind[o.element_type](table[tok * hidden + j])


# Residual add: o = a + b, elementwise, F32 math. (Foundation only ships
# residual_gate(x, gate[D], y) = x + gate*y, which needs a per-channel gate
# vector; a plain residual is cleaner as a tiny local kernel.)
def _add_kernel_bf16(
    a: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    b: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        var av = rebind[Scalar[DType.bfloat16]](a[i]).cast[DType.float32]()
        var bv = rebind[Scalar[DType.bfloat16]](b[i]).cast[DType.float32]()
        o[i] = rebind[o.element_type]((av + bv).cast[DType.bfloat16]())


def _add_kernel_f32(
    a: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    b: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        var av = rebind[Scalar[DType.float32]](a[i])
        var bv = rebind[Scalar[DType.float32]](b[i])
        o[i] = rebind[o.element_type](av + bv)


def _add_kernel_f16(
    a: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    b: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        var av = rebind[Scalar[DType.float16]](a[i]).cast[DType.float32]()
        var bv = rebind[Scalar[DType.float16]](b[i]).cast[DType.float32]()
        o[i] = rebind[o.element_type]((av + bv).cast[DType.float16]())


# GQA repeat in BSHD: src [1, N, H_kv, Dh] -> dst [1, N, H, Dh], where output
# head h reads kv-head (h // n_rep). One thread per output element. Pure copy
# (element bytes), so dtype-exact per width.
def _repeat_kv_kernel_bf16(
    src: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    dst: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    seq: Int,
    h: Int,
    h_kv: Int,
    dh: Int,
    n_rep: Int,
):
    var idx = Int(global_idx.x)
    var total = seq * h * dh
    if idx < total:
        var dh_i = idx % dh
        var rest = idx // dh
        var head = rest % h
        var t = rest // h
        var kvh = head // n_rep
        var src_idx = (t * h_kv + kvh) * dh + dh_i
        dst[idx] = rebind[dst.element_type](src[src_idx])


def _repeat_kv_kernel_f32(
    src: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    dst: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    seq: Int,
    h: Int,
    h_kv: Int,
    dh: Int,
    n_rep: Int,
):
    var idx = Int(global_idx.x)
    var total = seq * h * dh
    if idx < total:
        var dh_i = idx % dh
        var rest = idx // dh
        var head = rest % h
        var t = rest // h
        var kvh = head // n_rep
        var src_idx = (t * h_kv + kvh) * dh + dh_i
        dst[idx] = rebind[dst.element_type](src[src_idx])


def _repeat_kv_kernel_f16(
    src: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    dst: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    seq: Int,
    h: Int,
    h_kv: Int,
    dh: Int,
    n_rep: Int,
):
    var idx = Int(global_idx.x)
    var total = seq * h * dh
    if idx < total:
        var dh_i = idx % dh
        var rest = idx // dh
        var head = rest % h
        var t = rest // h
        var kvh = head // n_rep
        var src_idx = (t * h_kv + kvh) * dh + dh_i
        dst[idx] = rebind[dst.element_type](src[src_idx])


# ── glue helpers (host-side dispatch of the above kernels) ──────────────────
def _add(a: Tensor, b: Tensor, ctx: DeviceContext) raises -> Tensor:
    """o = a + b, elementwise; same shape/dtype. F32 math."""
    if a.numel() != b.numel():
        raise Error("add: numel mismatch")
    if a.dtype() != b.dtype():
        raise Error("add: dtype mismatch")
    var dt = a.dtype().to_mojo_dtype()
    var n = a.numel()
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](a.nbytes())
    var rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var grid = (n + _BLOCK - 1) // _BLOCK
    if dt == DType.float32:
        var A = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            a.buf.unsafe_ptr().bitcast[Float32](), rl
        )
        var B = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            b.buf.unsafe_ptr().bitcast[Float32](), rl
        )
        var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), rl
        )
        ctx.enqueue_function[_add_kernel_f32, _add_kernel_f32](
            A, B, O, n, grid_dim=grid, block_dim=_BLOCK
        )
    elif dt == DType.bfloat16:
        var A = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            a.buf.unsafe_ptr().bitcast[BFloat16](), rl
        )
        var B = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            b.buf.unsafe_ptr().bitcast[BFloat16](), rl
        )
        var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), rl
        )
        ctx.enqueue_function[_add_kernel_bf16, _add_kernel_bf16](
            A, B, O, n, grid_dim=grid, block_dim=_BLOCK
        )
    else:
        var A = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            a.buf.unsafe_ptr().bitcast[Float16](), rl
        )
        var B = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            b.buf.unsafe_ptr().bitcast[Float16](), rl
        )
        var O = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), rl
        )
        ctx.enqueue_function[_add_kernel_f16, _add_kernel_f16](
            A, B, O, n, grid_dim=grid, block_dim=_BLOCK
        )
    ctx.synchronize()
    return Tensor(out_buf^, a.shape(), a.dtype())


def _repeat_kv(
    var x: Tensor, h: Int, h_kv: Int, ctx: DeviceContext
) raises -> Tensor:
    """BSHD GQA repeat: [1, N, H_kv, Dh] -> [1, N, H, Dh]."""
    var xs = x.shape()
    if len(xs) != 4:
        raise Error("repeat_kv: x must be rank-4 [1,N,H_kv,Dh]")
    var seq = xs[1]
    var dh = xs[3]
    if xs[2] != h_kv:
        raise Error("repeat_kv: x head dim != h_kv")
    var n_rep = h // h_kv
    if n_rep == 1:
        return x^  # move-return (no-op for MHA)
    var dt = x.dtype().to_mojo_dtype()
    var out_n = seq * h * dh
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](
        out_n * x.dtype().byte_size()
    )
    var src_n = seq * h_kv * dh
    var src_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](src_n))
    var dst_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](out_n))
    var grid = (out_n + _BLOCK - 1) // _BLOCK
    if dt == DType.float32:
        var S = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float32](), src_rl
        )
        var D = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), dst_rl
        )
        ctx.enqueue_function[_repeat_kv_kernel_f32, _repeat_kv_kernel_f32](
            S, D, seq, h, h_kv, dh, n_rep, grid_dim=grid, block_dim=_BLOCK
        )
    elif dt == DType.bfloat16:
        var S = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[BFloat16](), src_rl
        )
        var D = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), dst_rl
        )
        ctx.enqueue_function[_repeat_kv_kernel_bf16, _repeat_kv_kernel_bf16](
            S, D, seq, h, h_kv, dh, n_rep, grid_dim=grid, block_dim=_BLOCK
        )
    else:
        var S = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float16](), src_rl
        )
        var D = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), dst_rl
        )
        ctx.enqueue_function[_repeat_kv_kernel_f16, _repeat_kv_kernel_f16](
            S, D, seq, h, h_kv, dh, n_rep, grid_dim=grid, block_dim=_BLOCK
        )
    ctx.synchronize()
    var out_shape = List[Int]()
    out_shape.append(1)
    out_shape.append(seq)
    out_shape.append(h)
    out_shape.append(dh)
    return Tensor(out_buf^, out_shape^, x.dtype())


# Reshape (row-major contiguous). Tensor uniquely owns its buffer and exposes
# no buffer-take API, and moving a single field out of `x` is rejected
# ("destroyed out of the middle of a value"). So we copy the bytes into a fresh
# device buffer with the new shape — a small contiguous D2D copy (~6/layer),
# negligible vs the matmuls. Caller guarantees numel match.
def _reshape(x: Tensor, var shape: List[Int], ctx: DeviceContext) raises -> Tensor:
    var want = 1
    for i in range(len(shape)):
        want *= shape[i]
    if want != x.numel():
        raise Error("reshape: numel mismatch")
    var nbytes = x.nbytes()
    var dev = ctx.enqueue_create_buffer[DType.uint8](nbytes)
    ctx.enqueue_copy(dst_buf=dev, src_buf=x.buf)
    ctx.synchronize()
    return Tensor(dev^, shape^, x.dtype())


# ── host-side table builders ─────────────────────────────────────────────────
# RoPE cos/sin built in row order (position, head): row index = t*H + head uses
# position t's angles (RoPE angle depends only on position, repeated across all
# heads). Half-split: angle[t, i] = t * theta^(-2i/Dh), i in [0, Dh/2).
# Returns flat F32 host list of length seq*H*(Dh/2), to be uploaded as the
# compute dtype matching x (so rope_halfsplit sees x/cos/sin same dtype).
def _build_rope_tables(
    seq: Int, heads: Int, head_dim: Int, theta: Float64
) raises -> List[List[Float32]]:
    var half = head_dim // 2
    var cos_vals = List[Float32]()
    var sin_vals = List[Float32]()
    var log_theta = flog(Float32(theta))
    for t in range(seq):
        # angles for this position, one per pair index i
        for _h in range(heads):
            for i in range(half):
                var exponent = (
                    -log_theta * Float32(2 * i) / Float32(head_dim)
                )
                var inv_freq = fexp(exponent)
                var angle = Float32(t) * inv_freq
                cos_vals.append(fcos(angle))
                sin_vals.append(fsin(angle))
    var out = List[List[Float32]]()
    out.append(cos_vals^)
    out.append(sin_vals^)
    return out^


# Additive causal mask [1, H, N, N]: 0.0 where j <= i (attend), else a large
# negative (HF uses min finite of dtype; -1e4 is a safe BF16-representable
# sentinel that drives softmax to ~0). real_len handles right-padding (columns
# >= real_len are masked for all rows).
def _build_causal_mask(
    seq: Int, heads: Int, real_len: Int
) raises -> List[Float32]:
    var neg = Float32(-1.0e4)
    var data = List[Float32]()
    for _hh in range(heads):
        for i in range(seq):
            for j in range(seq):
                if j <= i and j < real_len:
                    data.append(Float32(0.0))
                else:
                    data.append(neg)
    return data^


# ── Qwen3Encoder ──────────────────────────────────────────────────────────────
struct Qwen3Encoder:
    """Qwen3 text encoder. Owns all weights (ArcPointer because Tensor is
    Movable-not-Copyable, so a bare List[Tensor]/Dict[String,Tensor] won't
    compile — same discipline as the sharded loader). Forward runs on GPU."""

    var weights: List[ArcPointer[Tensor]]
    var name_to_idx: Dict[String, Int]
    var config: Qwen3Config

    def __init__(
        out self,
        var weights: List[ArcPointer[Tensor]],
        var name_to_idx: Dict[String, Int],
        config: Qwen3Config,
    ):
        self.weights = weights^
        self.name_to_idx = name_to_idx^
        self.config = config

    @staticmethod
    def load(
        dir: String, config: Qwen3Config, ctx: DeviceContext, max_layer: Int = -1
    ) raises -> Qwen3Encoder:
        """Load tensors from a sharded text_encoder dir into GPU Tensors
        (ShardedSafeTensors + Tensor.from_view, H2D copy). When `max_layer >= 0`
        (P2), SKIP weights an extract-at-max_layer encode never executes —
        `lm_head.weight` and transformer layers with index > max_layer — saving
        ~1 GB. The forward MUST stop at max_layer (encode() passes extract_layer
        as stop_at), or it would `_w()`-miss the skipped layer."""
        var sharded = ShardedSafeTensors.open(dir)
        var weights = List[ArcPointer[Tensor]]()
        var name_to_idx = Dict[String, Int]()
        # Deterministic order is irrelevant; we look up by name.
        for ref nm in sharded.names():
            if max_layer >= 0:
                if nm == "lm_head.weight":
                    continue
                var lp = String("model.layers.")
                if nm.find(lp) == 0:
                    var nb = nm.as_bytes()
                    var ci = len(lp)
                    var li = 0
                    while ci < len(nb) and Int(nb[ci]) >= 48 and Int(nb[ci]) <= 57:
                        li = li * 10 + (Int(nb[ci]) - 48)
                        ci += 1
                    if li > max_layer:
                        continue
            var tv = sharded.tensor_view(nm)
            var t = Tensor.from_view(tv, ctx)
            var idx = len(weights)
            weights.append(ArcPointer(t^))
            name_to_idx[nm] = idx
        return Qwen3Encoder(weights^, name_to_idx^, config)

    def lm_logits_last(self, token_ids: List[Int], pos: Int, ctx: DeviceContext) raises -> Tensor:
        """Autoregressive head: full forward over token_ids, then
        lm_head(model.norm(hidden[pos])) -> [1,1,vocab] logits. Requires the
        checkpoint to ship `lm_head.weight` (Qwen3-8B/instruct does; the Ideogram
        text_encoder does NOT). Used by the pure-Mojo magic-prompt generator."""
        var states = self.encode_layer_states(token_ids, ctx)
        var row = slice(states[len(states) - 1][], 1, pos, 1, ctx)  # [1,1,hidden]
        var normed = rms_norm(row, self._w(String("model.norm.weight")), self.config.rms_norm_eps, ctx)
        return linear(normed, self._w(String("lm_head.weight")), None, ctx)  # [1,1,vocab]

    def _w(self, name: String) raises -> ref [self.weights] Tensor:
        """Borrow a weight Tensor by name (no copy)."""
        if name not in self.name_to_idx:
            raise Error(String("missing weight: ") + name)
        var idx = self.name_to_idx[name]
        return self.weights[idx][]

    def _has(self, name: String) -> Bool:
        return name in self.name_to_idx

    # ── embedding ────────────────────────────────────────────────────────────
    def _embed(
        self, ids: List[Int], ctx: DeviceContext
    ) raises -> Tensor:
        """Gather embedding rows -> [1, seq, hidden]."""
        ref table = self._w(String("model.embed_tokens.weight"))
        var ts = table.shape()
        var hidden = ts[len(ts) - 1]
        var seq = len(ids)
        var dt = table.dtype().to_mojo_dtype()

        # Upload token ids as an I32 device buffer.
        var id_host = ctx.enqueue_create_host_buffer[DType.uint8](seq * 4)
        var ip = id_host.unsafe_ptr().bitcast[Int32]()
        for i in range(seq):
            ip[i] = Int32(ids[i])
        var id_dev = ctx.enqueue_create_buffer[DType.uint8](seq * 4)
        ctx.enqueue_copy(dst_buf=id_dev, src_buf=id_host)
        ctx.synchronize()

        var out_buf = ctx.enqueue_create_buffer[DType.uint8](
            seq * hidden * table.dtype().byte_size()
        )
        var tab_n = table.numel()
        var tab_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](tab_n))
        var id_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](seq))
        var out_rl = RuntimeLayout[_DYN1].row_major(
            IndexList[1](seq * hidden)
        )
        var total = seq * hidden
        var grid = (total + _BLOCK - 1) // _BLOCK
        var IDS = LayoutTensor[DType.int32, _DYN1, MutAnyOrigin](
            id_dev.unsafe_ptr().bitcast[Int32](), id_rl
        )
        if dt == DType.float32:
            var T = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
                table.buf.unsafe_ptr().bitcast[Float32](), tab_rl
            )
            var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
                out_buf.unsafe_ptr().bitcast[Float32](), out_rl
            )
            ctx.enqueue_function[_embed_kernel_f32, _embed_kernel_f32](
                T, IDS, O, seq, hidden, grid_dim=grid, block_dim=_BLOCK
            )
        elif dt == DType.bfloat16:
            var T = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
                table.buf.unsafe_ptr().bitcast[BFloat16](), tab_rl
            )
            var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
                out_buf.unsafe_ptr().bitcast[BFloat16](), out_rl
            )
            ctx.enqueue_function[_embed_kernel_bf16, _embed_kernel_bf16](
                T, IDS, O, seq, hidden, grid_dim=grid, block_dim=_BLOCK
            )
        else:
            var T = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
                table.buf.unsafe_ptr().bitcast[Float16](), tab_rl
            )
            var O = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
                out_buf.unsafe_ptr().bitcast[Float16](), out_rl
            )
            ctx.enqueue_function[_embed_kernel_f16, _embed_kernel_f16](
                T, IDS, O, seq, hidden, grid_dim=grid, block_dim=_BLOCK
            )
        ctx.synchronize()
        var sh = List[Int]()
        sh.append(1)
        sh.append(seq)
        sh.append(hidden)
        return Tensor(out_buf^, sh^, table.dtype())

    # ── one transformer layer ──────────────────────────────────────────────
    def _layer(
        self,
        layer_idx: Int,
        hidden: Tensor,
        cos_q: Tensor,
        sin_q: Tensor,
        cos_k: Tensor,
        sin_k: Tensor,
        mask: Tensor,
        ctx: DeviceContext,
    ) raises -> Tensor:
        var cfg = self.config
        var h = cfg.num_heads
        var h_kv = cfg.num_kv_heads
        var dh = cfg.head_dim
        var eps = cfg.rms_norm_eps
        var scale = Float32(1.0) / sqrt(Float32(dh))
        var p = String("model.layers.") + String(layer_idx)

        var hs = hidden.shape()
        var seq = hs[1]

        # --- self-attention ---
        ref in_ln = self._w(p + ".input_layernorm.weight")
        var normed = rms_norm(hidden, in_ln, eps, ctx)

        ref qw = self._w(p + ".self_attn.q_proj.weight")
        ref kw = self._w(p + ".self_attn.k_proj.weight")
        ref vw = self._w(p + ".self_attn.v_proj.weight")
        var q = linear(normed, qw, None, ctx)  # [1, seq, h*dh]
        var k = linear(normed, kw, None, ctx)  # [1, seq, h_kv*dh]
        var v = linear(normed, vw, None, ctx)  # [1, seq, h_kv*dh]

        # reshape to BSHD [1, seq, H, Dh] / [1, seq, H_kv, Dh]
        var q_sh = List[Int]()
        q_sh.append(1)
        q_sh.append(seq)
        q_sh.append(h)
        q_sh.append(dh)
        q = _reshape(q, q_sh^, ctx)
        var k_sh = List[Int]()
        k_sh.append(1)
        k_sh.append(seq)
        k_sh.append(h_kv)
        k_sh.append(dh)
        k = _reshape(k, k_sh^, ctx)
        var v_sh = List[Int]()
        v_sh.append(1)
        v_sh.append(seq)
        v_sh.append(h_kv)
        v_sh.append(dh)
        v = _reshape(v, v_sh^, ctx)

        # per-head QK-norm (RMSNorm over Dh). rms_norm flattens leading dims
        # (1*seq*H rows) and normalizes the last dim Dh. Weight is [Dh].
        ref qn = self._w(p + ".self_attn.q_norm.weight")
        ref kn = self._w(p + ".self_attn.k_norm.weight")
        q = rms_norm(q, qn, eps, ctx)
        k = rms_norm(k, kn, eps, ctx)

        # RoPE half-split on q,k. rope_halfsplit flattens leading dims to rows
        # (q: seq*H, k: seq*H_kv) and rotates the last dim Dh using cos/sin
        # whose flat numel == rows*(Dh/2). cos_q/sin_q ordered (position, head)
        # for H heads; cos_k/sin_k for H_kv heads.
        q = rope_halfsplit(q, cos_q, sin_q, ctx)
        k = rope_halfsplit(k, cos_k, sin_k, ctx)

        # GQA: repeat kv heads to H, then SDPA in BSHD [1, seq, H, Dh].
        var k_rep = _repeat_kv(k^, h, h_kv, ctx)
        var v_rep = _repeat_kv(v^, h, h_kv, ctx)

        # sdpa is comptime-parameterized on B/S/H/Dh; dispatch via a shaped
        # wrapper that materializes the comptime case (see _sdpa_dispatch).
        var attn = _sdpa_dispatch(q, k_rep, v_rep, mask, scale, seq, h, dh, ctx)

        # attn [1, seq, H, Dh] -> [1, seq, H*Dh]
        var attn_sh = List[Int]()
        attn_sh.append(1)
        attn_sh.append(seq)
        attn_sh.append(h * dh)
        attn = _reshape(attn, attn_sh^, ctx)

        ref ow = self._w(p + ".self_attn.o_proj.weight")
        var attn_out = linear(attn, ow, None, ctx)  # [1, seq, hidden]

        var hidden2 = _add(hidden, attn_out, ctx)

        # --- MLP (SwiGLU) ---
        ref post_ln = self._w(p + ".post_attention_layernorm.weight")
        var normed2 = rms_norm(hidden2, post_ln, eps, ctx)
        ref gw = self._w(p + ".mlp.gate_proj.weight")
        ref uw = self._w(p + ".mlp.up_proj.weight")
        ref dw = self._w(p + ".mlp.down_proj.weight")
        var gate = linear(normed2, gw, None, ctx)
        var up = linear(normed2, uw, None, ctx)
        var act = swiglu(gate, up, ctx)  # silu(gate) * up
        var mlp_out = linear(act, dw, None, ctx)

        return _add(hidden2, mlp_out, ctx)

    # ── full forward ──────────────────────────────────────────────────────────
    def encode_layer_states(
        self, token_ids: List[Int], ctx: DeviceContext, stop_at: Int = -1
    ) raises -> List[ArcPointer[Tensor]]:
        """Run all layers, returning hidden states AFTER each layer (index i =
        output of layer i), PRE-final-norm. Index 0..num_layers-1. The caller
        applies model.norm to whichever layer it extracts (matches Rust
        encode() which returns pre-final-norm hidden states)."""
        var cfg = self.config
        var seq = len(token_ids)
        var h = cfg.num_heads
        var h_kv = cfg.num_kv_heads
        var dh = cfg.head_dim

        # right-pad detection (pad id 151643, per qwen3_encoder.rs:476).
        var pad_id = 151643
        var real_len = seq
        for i in range(seq):
            if token_ids[i] == pad_id:
                real_len = i
                break

        # RoPE tables (host trig in F32), uploaded as compute dtype.
        var dtype = self._w(String("model.embed_tokens.weight")).dtype()
        var q_tables = _build_rope_tables(seq, h, dh, cfg.rope_theta)
        var k_tables = _build_rope_tables(seq, h_kv, dh, cfg.rope_theta)
        var half = dh // 2
        var cq_sh = List[Int]()
        cq_sh.append(seq * h * half)
        var ck_sh = List[Int]()
        ck_sh.append(seq * h_kv * half)
        var cos_q = Tensor.from_host(q_tables[0], cq_sh.copy(), dtype, ctx)
        var sin_q = Tensor.from_host(q_tables[1], cq_sh.copy(), dtype, ctx)
        var cos_k = Tensor.from_host(k_tables[0], ck_sh.copy(), dtype, ctx)
        var sin_k = Tensor.from_host(k_tables[1], ck_sh.copy(), dtype, ctx)

        # Additive causal mask [1, H, seq, seq].
        var mask_data = _build_causal_mask(seq, h, real_len)
        var mask_sh = List[Int]()
        mask_sh.append(1)
        mask_sh.append(h)
        mask_sh.append(seq)
        mask_sh.append(seq)
        var mask = Tensor.from_host(mask_data, mask_sh^, dtype, ctx)

        var hidden = self._embed(token_ids, ctx)
        var states = List[ArcPointer[Tensor]]()
        # P2: stop at the extract layer — running later layers can't change an
        # earlier state, and they'd `_w()`-miss the layers load(max_layer) skipped.
        var n_run = cfg.num_layers if stop_at < 0 else (stop_at + 1)
        for i in range(n_run):
            hidden = self._layer(
                i, hidden, cos_q, sin_q, cos_k, sin_k, mask, ctx
            )
            # store a copy of the post-layer hidden state
            states.append(ArcPointer(_clone(hidden, ctx)))
        return states^

    def encode(
        self, token_ids: List[Int], extract_layer: Int, ctx: DeviceContext
    ) raises -> Tensor:
        """Return the hidden state after `extract_layer` (PRE-final-norm),
        shape [1, seq, hidden]. For Z-Image last_hidden_state, call with
        extract_layer = num_layers-1 then apply final_norm() separately."""
        var states = self.encode_layer_states(token_ids, ctx, extract_layer)
        if extract_layer < 0 or extract_layer >= len(states):
            raise Error("encode: extract_layer out of range")
        return _clone(states[extract_layer][], ctx)

    def encode_klein(self, token_ids: List[Int], ctx: DeviceContext) raises -> Tensor:
        """Return Klein stacked conditioning from layers [8,17,26].

        Output shape is `[1, seq, 3 * hidden_size]`: `[1,512,7680]` for
        Klein 4B and `[1,512,12288]` for Klein 9B. This is the
        inference-flame convention; Modular's comments may describe the same
        hidden-state positions as `[9,18,27]` in HF's embedding-output-indexed
        convention.
        """
        var states = self.encode_layer_states(token_ids, ctx)
        if len(states) <= 26:
            raise Error("encode_klein: encoder has fewer than 27 layers")
        var h8 = _clone(states[8][], ctx)
        var h17 = _clone(states[17][], ctx)
        var h26 = _clone(states[26][], ctx)
        return concat(2, ctx, h8, h17, h26)

    def final_norm(self, x: Tensor, ctx: DeviceContext) raises -> Tensor:
        """Apply model.norm (RMSNorm) — the final norm producing
        last_hidden_state. Separate from encode() to mirror the Rust path."""
        ref nw = self._w(String("model.norm.weight"))
        return rms_norm(x, nw, self.config.rms_norm_eps, ctx)

    def debug_pre_attn(
        self, token_ids: List[Int], ctx: DeviceContext
    ) raises -> List[ArcPointer[Tensor]]:
        """Run the layer-0 pre-attention path (embed, input_layernorm, q/k
        proj, per-head qk-norm, rope) and return the intermediates:
            [0]=embed [1, seq, hidden]
            [1]=l0_input_norm [1, seq, hidden]
            [2]=l0_q_rope [1, seq, H, Dh]
            [3]=l0_k_rope [1, seq, H_kv, Dh]
        This exercises EVERY foundation op the encoder uses EXCEPT sdpa, so it
        validates the embedding / rms_norm / linear / qk-norm / rope wiring even
        while the SDK flash_attention has the Dh=128 instantiation wall (see
        report). Reuses the exact same code paths as _layer up to rope."""
        var cfg = self.config
        var seq = len(token_ids)
        var h = cfg.num_heads
        var h_kv = cfg.num_kv_heads
        var dh = cfg.head_dim
        var eps = cfg.rms_norm_eps
        var dtype = self._w(String("model.embed_tokens.weight")).dtype()

        var q_tables = _build_rope_tables(seq, h, dh, cfg.rope_theta)
        var k_tables = _build_rope_tables(seq, h_kv, dh, cfg.rope_theta)
        var half = dh // 2
        var cq_sh = List[Int]()
        cq_sh.append(seq * h * half)
        var ck_sh = List[Int]()
        ck_sh.append(seq * h_kv * half)
        var cos_q = Tensor.from_host(q_tables[0], cq_sh.copy(), dtype, ctx)
        var sin_q = Tensor.from_host(q_tables[1], cq_sh.copy(), dtype, ctx)
        var cos_k = Tensor.from_host(k_tables[0], ck_sh.copy(), dtype, ctx)
        var sin_k = Tensor.from_host(k_tables[1], ck_sh.copy(), dtype, ctx)

        var embed = self._embed(token_ids, ctx)
        ref in_ln = self._w(String("model.layers.0.input_layernorm.weight"))
        var normed = rms_norm(embed, in_ln, eps, ctx)

        ref qw = self._w(String("model.layers.0.self_attn.q_proj.weight"))
        ref kw = self._w(String("model.layers.0.self_attn.k_proj.weight"))
        var q = linear(normed, qw, None, ctx)
        var k = linear(normed, kw, None, ctx)
        var q_sh = List[Int]()
        q_sh.append(1)
        q_sh.append(seq)
        q_sh.append(h)
        q_sh.append(dh)
        q = _reshape(q, q_sh^, ctx)
        var k_sh = List[Int]()
        k_sh.append(1)
        k_sh.append(seq)
        k_sh.append(h_kv)
        k_sh.append(dh)
        k = _reshape(k, k_sh^, ctx)
        ref qn = self._w(String("model.layers.0.self_attn.q_norm.weight"))
        ref kn = self._w(String("model.layers.0.self_attn.k_norm.weight"))
        q = rms_norm(q, qn, eps, ctx)
        k = rms_norm(k, kn, eps, ctx)
        q = rope_halfsplit(q, cos_q, sin_q, ctx)
        k = rope_halfsplit(k, cos_k, sin_k, ctx)

        var out = List[ArcPointer[Tensor]]()
        out.append(ArcPointer(_clone(embed, ctx)))
        out.append(ArcPointer(_clone(normed, ctx)))
        out.append(ArcPointer(q^))
        out.append(ArcPointer(k^))
        return out^


# ── SDPA dispatch (B/S/H/Dh are comptime in the foundation sdpa) ────────────
# The foundation sdpa requires B/S/H/Dh as compile-time params. seq varies at
# runtime, so we materialize the common Z-Image parities here. Add cases as
# needed; an unsupported (seq,h,dh) raises with a clear message.
def _sdpa_dispatch(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Tensor,
    scale: Float32,
    seq: Int,
    h: Int,
    dh: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    # Z-Image: H=32, Dh=128. Enumerate supported sequence lengths at comptime.
    if h == 32 and dh == 128:
        if seq == 8:
            return sdpa[1, 8, 32, 128](q, k, v, mask, scale, ctx)
        if seq == 16:
            return sdpa[1, 16, 32, 128](q, k, v, mask, scale, ctx)
        if seq == 32:
            return sdpa[1, 32, 32, 128](q, k, v, mask, scale, ctx)
        if seq == 64:
            return sdpa[1, 64, 32, 128](q, k, v, mask, scale, ctx)
        if seq == 128:
            return sdpa[1, 128, 32, 128](q, k, v, mask, scale, ctx)
        if seq == 256:
            return sdpa[1, 256, 32, 128](q, k, v, mask, scale, ctx)
        if seq == 512:
            return sdpa[1, 512, 32, 128](q, k, v, mask, scale, ctx)
        if seq == 1024:
            return sdpa[1, 1024, 32, 128](q, k, v, mask, scale, ctx)
        if seq == 2048:
            return sdpa[1, 2048, 32, 128](q, k, v, mask, scale, ctx)
    # Qwen3-0.6B (Anima text path): H=16, Dh=128.
    if h == 16 and dh == 128:
        if seq == 8:
            return sdpa[1, 8, 16, 128](q, k, v, mask, scale, ctx)
        if seq == 16:
            return sdpa[1, 16, 16, 128](q, k, v, mask, scale, ctx)
        if seq == 32:
            return sdpa[1, 32, 16, 128](q, k, v, mask, scale, ctx)
        if seq == 64:
            return sdpa[1, 64, 16, 128](q, k, v, mask, scale, ctx)
        if seq == 128:
            return sdpa[1, 128, 16, 128](q, k, v, mask, scale, ctx)
        if seq == 256:
            return sdpa[1, 256, 16, 128](q, k, v, mask, scale, ctx)
        if seq == 512:
            return sdpa[1, 512, 16, 128](q, k, v, mask, scale, ctx)
    raise Error(
        String("sdpa_dispatch: unsupported (seq,h,dh)=(")
        + String(seq) + "," + String(h) + "," + String(dh)
        + "). Add a comptime case."
    )


# Clone a Tensor's device buffer (deep copy) so it can be stored independently.
def _clone(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    var nbytes = x.nbytes()
    var dev = ctx.enqueue_create_buffer[DType.uint8](nbytes)
    ctx.enqueue_copy(dst_buf=dev, src_buf=x.buf)
    ctx.synchronize()
    return Tensor(dev^, x.shape(), x.dtype())
