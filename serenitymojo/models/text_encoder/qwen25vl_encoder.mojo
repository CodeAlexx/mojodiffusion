# models/text_encoder/qwen25vl_encoder.mojo — Qwen2.5-VL text encoder (GPU).
#
# Pure-Mojo, inference-only port of the TEXT-ONLY forward path of
# inference-flame/src/models/qwen25vl_encoder.rs (the Qwen2.5-VL language model
# used as Qwen-Image's text encoder). MIRRORS qwen3_encoder.mojo verbatim for
# structure and reuses the SAME Phase-A foundation ops — rms_norm / linear /
# rope_halfsplit / sdpa / swiglu — plus the SAME encoder-local glue (embedding
# gather, residual add, GQA kv-head repeat, RoPE/causal-mask host tables,
# reshape). Only THREE things differ from Qwen3:
#
#   1. Q/K/V projections HAVE BIASES (config attention_bias=True); o_proj is
#      bias-free. (Qwen3 had no projection biases.)  -> linear(x, w, Some(b))
#   2. NO per-head q_norm / k_norm. (Qwen3 applied RMSNorm over Dh to q,k
#      BEFORE rope; Qwen2.5-VL does NOT — the q_norm/k_norm weights do not
#      exist in the checkpoint.)
#   3. Config: 28 layers, hidden 3584, 28 q-heads, 4 kv-heads (GQA n_rep=7),
#      head_dim 128, intermediate 18944, rms_eps 1e-6, rope_theta 1e6.
#
# Architecture per layer (qwen25vl_encoder.rs:335 layer_forward):
#   h -> rms_norm(input_layernorm) -> q/k/v Linear(+bias)
#     -> rope_halfsplit on q,k -> repeat_kv(k,v) -> sdpa(causal) -> o_proj
#     -> residual
#     -> rms_norm(post_attention_layernorm) -> swiglu(gate,up)*down -> residual
# Final: rms_norm(model.norm) producing last_hidden_state (applied by the
# CALLER via final_norm(), mirroring the Rust encode() which applies model.norm
# itself — encode_layer_states() here returns PRE-final-norm states; the parity
# driver applies final_norm() to the last layer state to match the Rust
# last_hidden_state).
#
# RoPE = HALF-SPLIT (HF rotate_half), theta=1e6. For the TEXT-ONLY path the
# Qwen2.5-VL mRoPE collapses to standard 1D positions (all three mrope sections
# share the same per-token position id when there are no vision tokens), so the
# 1D half-split table is exact — same as qwen3_encoder.mojo. Dh=128 -> the
# foundation sdpa routes to its math-mode path (works on sm_86; SDPA_DH128_REPRO
# .md is stale, a math fallback was since added to ops/attention.mojo).
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
from serenitymojo.ops.linear import linear
from serenitymojo.ops.activations import swiglu


comptime _DYN1 = Layout.row_major(-1)
comptime _BLOCK = 256


# ── Config ──────────────────────────────────────────────────────────────────
@fieldwise_init
struct Qwen25VLConfig(Copyable, Movable, ImplicitlyCopyable):
    """Qwen2.5-VL text-encoder hyperparameters (Qwen-Image)."""

    var hidden_size: Int
    var num_layers: Int
    var num_heads: Int
    var num_kv_heads: Int
    var head_dim: Int
    var rms_norm_eps: Float32
    var rope_theta: Float64

    @staticmethod
    def qwen_image() -> Qwen25VLConfig:
        """Qwen-Image text encoder (Qwen2.5-VL-7B text path, hidden=3584).

        From Qwen-Image-2512 text_encoder/config.json:
          hidden_size=3584, num_hidden_layers=28, num_attention_heads=28,
          num_key_value_heads=4 (GQA n_rep=7), head_dim=128 (3584/28),
          rms_norm_eps=1e-6, rope_theta=1e6, intermediate_size=18944.
        """
        return Qwen25VLConfig(3584, 28, 28, 4, 128, Float32(1e-6), Float64(1e6))


# ── encoder-local glue kernels (NOT foundation ops) ─────────────────────────
# Identical to qwen3_encoder.mojo — pure copies / F32 elementwise, dtype-exact.
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


# Reshape (row-major contiguous), small D2D copy. See qwen3_encoder.mojo rationale.
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
# Identical to qwen3: RoPE cos/sin in row order (position, head); half-split.
def _build_rope_tables(
    seq: Int, heads: Int, head_dim: Int, theta: Float64
) raises -> List[List[Float32]]:
    var half = head_dim // 2
    var cos_vals = List[Float32]()
    var sin_vals = List[Float32]()
    var log_theta = flog(Float32(theta))
    for t in range(seq):
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


# Additive causal mask [1, H, N, N]: 0.0 where j <= i (attend) and j < real_len,
# else -1e4 (BF16-safe sentinel). Matches the foundation sdpa additive-mask
# convention (mask is added to QKᵀ scores).
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


# ── Qwen25VLEncoder ──────────────────────────────────────────────────────────
struct Qwen25VLEncoder:
    """Qwen2.5-VL text encoder (text-only path). Owns all weights (ArcPointer
    because Tensor is Movable-not-Copyable). Forward runs on GPU. Loads only the
    `model.*` text-path tensors; the checkpoint also contains `visual.*` and
    `lm_head.weight` which are skipped (we look up by name)."""

    var weights: List[ArcPointer[Tensor]]
    var name_to_idx: Dict[String, Int]
    var config: Qwen25VLConfig

    def __init__(
        out self,
        var weights: List[ArcPointer[Tensor]],
        var name_to_idx: Dict[String, Int],
        config: Qwen25VLConfig,
    ):
        self.weights = weights^
        self.name_to_idx = name_to_idx^
        self.config = config

    @staticmethod
    def load(
        dir: String, config: Qwen25VLConfig, ctx: DeviceContext
    ) raises -> Qwen25VLEncoder:
        """Load text-path tensors from a sharded text_encoder dir into GPU
        Tensors via ShardedSafeTensors + Tensor.from_view (H2D copy). Only
        names beginning with `model.` are loaded (skip visual.* + lm_head)."""
        var sharded = ShardedSafeTensors.open(dir)
        var weights = List[ArcPointer[Tensor]]()
        var name_to_idx = Dict[String, Int]()
        for ref nm in sharded.names():
            if not nm.startswith("model."):
                continue
            var tv = sharded.tensor_view(nm)
            var t = Tensor.from_view(tv, ctx)
            var idx = len(weights)
            weights.append(ArcPointer(t^))
            name_to_idx[nm] = idx
        return Qwen25VLEncoder(weights^, name_to_idx^, config)

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

        # Q/K/V projections — ALL THREE have biases in Qwen2.5-VL.
        ref qw = self._w(p + ".self_attn.q_proj.weight")
        ref qb = self._w(p + ".self_attn.q_proj.bias")
        ref kw = self._w(p + ".self_attn.k_proj.weight")
        ref kb = self._w(p + ".self_attn.k_proj.bias")
        ref vw = self._w(p + ".self_attn.v_proj.weight")
        ref vb = self._w(p + ".self_attn.v_proj.bias")
        # Bias is borrowed (ref) from self.weights and Tensor is not Copyable,
        # so clone each tiny [out] bias into an owned Tensor and transfer it.
        var q = linear(normed, qw, Optional[Tensor](_clone(qb, ctx)), ctx)
        var k = linear(normed, kw, Optional[Tensor](_clone(kb, ctx)), ctx)
        var v = linear(normed, vw, Optional[Tensor](_clone(vb, ctx)), ctx)

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

        # NO per-head QK-norm in Qwen2.5-VL (unlike Qwen3).

        # RoPE half-split on q,k. cos_q/sin_q ordered (position, head) for H
        # heads; cos_k/sin_k for H_kv heads.
        q = rope_halfsplit(q, cos_q, sin_q, ctx)
        k = rope_halfsplit(k, cos_k, sin_k, ctx)

        # GQA: repeat kv heads to H, then SDPA in BSHD [1, seq, H, Dh].
        var k_rep = _repeat_kv(k^, h, h_kv, ctx)
        var v_rep = _repeat_kv(v^, h, h_kv, ctx)

        var attn = _sdpa_dispatch(q, k_rep, v_rep, mask, scale, seq, h, dh, ctx)

        # attn [1, seq, H, Dh] -> [1, seq, H*Dh]
        var attn_sh = List[Int]()
        attn_sh.append(1)
        attn_sh.append(seq)
        attn_sh.append(h * dh)
        attn = _reshape(attn, attn_sh^, ctx)

        ref ow = self._w(p + ".self_attn.o_proj.weight")
        var attn_out = linear(attn, ow, None, ctx)  # o_proj is bias-free

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
        self, token_ids: List[Int], ctx: DeviceContext
    ) raises -> List[ArcPointer[Tensor]]:
        """Run all layers, returning hidden states AFTER each layer (index i =
        output of layer i), PRE-final-norm. The caller applies model.norm to
        whichever layer it extracts (matches Rust encode())."""
        var cfg = self.config
        var seq = len(token_ids)
        var h = cfg.num_heads
        var h_kv = cfg.num_kv_heads
        var dh = cfg.head_dim

        # right-pad detection (pad id 151643, per qwen25vl_encoder.rs:455).
        var pad_id = 151643
        var real_len = seq
        for i in range(seq):
            if token_ids[i] == pad_id:
                real_len = i
                break

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

        var mask_data = _build_causal_mask(seq, h, real_len)
        var mask_sh = List[Int]()
        mask_sh.append(1)
        mask_sh.append(h)
        mask_sh.append(seq)
        mask_sh.append(seq)
        var mask = Tensor.from_host(mask_data, mask_sh^, dtype, ctx)

        var hidden = self._embed(token_ids, ctx)
        var states = List[ArcPointer[Tensor]]()
        for i in range(cfg.num_layers):
            hidden = self._layer(
                i, hidden, cos_q, sin_q, cos_k, sin_k, mask, ctx
            )
            states.append(ArcPointer(_clone(hidden, ctx)))
        return states^

    def encode(
        self, token_ids: List[Int], extract_layer: Int, ctx: DeviceContext
    ) raises -> Tensor:
        """Return the hidden state after `extract_layer` (PRE-final-norm),
        shape [1, seq, hidden]. For last_hidden_state, call with
        extract_layer = num_layers-1 then apply final_norm() separately."""
        var states = self.encode_layer_states(token_ids, ctx)
        if extract_layer < 0 or extract_layer >= len(states):
            raise Error("encode: extract_layer out of range")
        return _clone(states[extract_layer][], ctx)

    def final_norm(self, x: Tensor, ctx: DeviceContext) raises -> Tensor:
        """Apply model.norm (RMSNorm) — the final norm producing
        last_hidden_state."""
        ref nw = self._w(String("model.norm.weight"))
        return rms_norm(x, nw, self.config.rms_norm_eps, ctx)

    def debug_pre_attn(
        self, token_ids: List[Int], ctx: DeviceContext
    ) raises -> List[ArcPointer[Tensor]]:
        """Run the layer-0 pre-attention path (embed, input_layernorm, q/k
        proj+bias, rope) and return intermediates:
            [0]=embed [1, seq, hidden]
            [1]=l0_input_norm [1, seq, hidden]
            [2]=l0_q_rope [1, seq, H, Dh]
            [3]=l0_k_rope [1, seq, H_kv, Dh]
        Exercises every foundation op EXCEPT sdpa, validating embedding /
        rms_norm / linear+bias / rope_halfsplit wiring. NO qk-norm (Qwen2.5-VL).
        """
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
        ref qb = self._w(String("model.layers.0.self_attn.q_proj.bias"))
        ref kw = self._w(String("model.layers.0.self_attn.k_proj.weight"))
        ref kb = self._w(String("model.layers.0.self_attn.k_proj.bias"))
        var q = linear(normed, qw, Optional[Tensor](_clone(qb, ctx)), ctx)
        var k = linear(normed, kw, Optional[Tensor](_clone(kb, ctx)), ctx)
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
        q = rope_halfsplit(q, cos_q, sin_q, ctx)
        k = rope_halfsplit(k, cos_k, sin_k, ctx)

        var out = List[ArcPointer[Tensor]]()
        out.append(ArcPointer(_clone(embed, ctx)))
        out.append(ArcPointer(_clone(normed, ctx)))
        out.append(ArcPointer(q^))
        out.append(ArcPointer(k^))
        return out^


# ── SDPA dispatch (B/S/H/Dh are comptime in the foundation sdpa) ────────────
# Qwen2.5-VL: H=28, Dh=128 (math-mode path). Enumerate supported seq lengths.
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
    if h == 28 and dh == 128:
        if seq == 8:
            return sdpa[1, 8, 28, 128](q, k, v, mask, scale, ctx)
        if seq == 16:
            return sdpa[1, 16, 28, 128](q, k, v, mask, scale, ctx)
        if seq == 32:
            return sdpa[1, 32, 28, 128](q, k, v, mask, scale, ctx)
        if seq == 64:
            return sdpa[1, 64, 28, 128](q, k, v, mask, scale, ctx)
        if seq == 128:
            return sdpa[1, 128, 28, 128](q, k, v, mask, scale, ctx)
        if seq == 256:
            return sdpa[1, 256, 28, 128](q, k, v, mask, scale, ctx)
        if seq == 512:
            return sdpa[1, 512, 28, 128](q, k, v, mask, scale, ctx)
        if seq == 546:
            return sdpa[1, 546, 28, 128](q, k, v, mask, scale, ctx)
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
