# KleinTextEncoder.mojo — FLUX.2/Klein Qwen3 text encoder (forward) + chat
# tokenizer seam. (Klein == Serenity FLUX_2 with Qwen3ForCausalLM text encoder;
# the dev variant uses Mistral, num_attention_heads==48 — see Flux2Model.is_dev.)
#
# ════════════════════════════════════════════════════════════════════════════
# PORT SPEC: Serenity modules/model/Flux2Model.py::encode_text (Klein branch,
# lines 194-225).
#   1. qwen3_format_input(prompt) = [{"role":"user","content":prompt}]
#      (Flux2Model.py:29-32).
#   2. apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
#      enable_thinking=False) -> templated string (Flux2Model.py:197-203).
#      VERIFIED Qwen3 template (4B/8B): the non-thinking + generation-prompt
#      form appends an EMPTY think block, so the templated string is
#      "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n".
#   3. tokenizer(text, max_length=text_encoder_sequence_length,
#      padding='max_length', truncation=True) -> input_ids + attention_mask
#      (Flux2Model.py:205-211).
#   4. text_encoder(tokens, attention_mask, output_hidden_states=True,
#      use_cache=False) (Flux2Model.py:216-223), then:
#        torch.cat([hidden_states[k] for k in QWEN3_HIDDEN_STATES_LAYERS], dim=2)
#      with QWEN3_HIDDEN_STATES_LAYERS = [9, 18, 27]  (Flux2Model.py:27,224-225).
#      -> conditioning [1, max_length, 3 * hidden_size].
#   NOTE: HF output_hidden_states gives
#     hidden_states = [embed_out, layer0_out, ..., layer35_out]  (num_layers+1).
#   So hidden_states[9]/[18]/[27] == the outputs of layers 8/17/26 (0-indexed
#   post-layer states), PRE-final-norm. Our encode_layer_states[i] is exactly the
#   post-layer-i state, so we extract indices [8,17,26]. NO final_norm() and NO
#   attention-mask narrowing — unlike Z-Image, Flux2.predict feeds the FULL
#   max_length-padded conditioning to the transformer (BaseFlux2Setup.predict
#   passes text_encoder_output straight through; no mask filter).
#
# Qwen3 config (Klein text_encoder/config.json, Qwen3ForCausalLM):
#   4B:  hidden=2560, layers=36, heads=32, kv_heads=8 (GQA n_rep=4), head_dim=128
#   9B:  hidden=4096, layers=36, heads=32, kv_heads=8,                head_dim=128
#   rms_norm_eps=1e-6, rope_theta=1e6, hidden_act=silu. Both have >=27 layers, so
#   the [9,18,27] extraction is valid for either. RoPE = HALF-SPLIT (HF
#   rotate_half). BF16 storage, F32 accumulation in foundation ops.
#
# ════════════════════════════════════════════════════════════════════════════
# BORROW POLICY: model-level forward COPIED into the port (namespace
# serenity_trainer) — NOT imported from serenitymojo for model logic.
#   BORROWED FROM: serenitymojo/models/text_encoder/qwen3_encoder.mojo
#     (Qwen3Config, Qwen3Encoder, encode_klein, all encoder-local kernels/glue:
#      _embed*, _add*, _repeat_kv*, _reshape, _build_rope_tables,
#      _build_causal_mask, _sdpa_dispatch, _clone) — adapted namespace to
#      serenity_trainer.model.KleinTextEncoder so the port owns + can modify it
#      (e.g. activation-checkpointing per
#      BaseFlux2Setup.enable_checkpointing_for_qwen3_encoder_layers).
#   TOKENIZER SEAM: the raw byte-level BPE (Qwen3Tokenizer) is FOUNDATION-tier
#     infra (like io/), imported from serenitymojo. The Klein-specific chat
#     template (qwen3_format_input + apply_chat_template) lives HERE in
#     KleinChatTokenizer, which the port owns.

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
# Foundation-tier BPE (byte-merge algorithm), imported like io/.
from serenitymojo.tokenizer.tokenizer import Qwen3Tokenizer


comptime _DYN1 = Layout.row_major(-1)
comptime _BLOCK = 256

# Klein text path constants (Flux2Model.py).
comptime KLEIN_PAD_ID = 151643          # Qwen pad token; right-pad, causal-masked.
# QWEN3_HIDDEN_STATES_LAYERS = [9,18,27] (HF embedding-output-indexed). As
# post-layer states (our encode_layer_states indexing) these are [8,17,26].
comptime KLEIN_EXTRACT_0 = 8
comptime KLEIN_EXTRACT_1 = 17
comptime KLEIN_EXTRACT_2 = 26


# ── Config ──────────────────────────────────────────────────────────────────
@fieldwise_init
struct Qwen3Config(Copyable, Movable, ImplicitlyCopyable):
    """Qwen3 text-encoder hyperparameters (Klein 4B / 9B)."""

    var hidden_size: Int
    var num_layers: Int
    var num_heads: Int
    var num_kv_heads: Int
    var head_dim: Int
    var rms_norm_eps: Float32
    var rope_theta: Float64

    @staticmethod
    def klein_4b() -> Qwen3Config:
        """Klein 4B text encoder (Qwen3-4B, hidden=2560)."""
        return Qwen3Config(2560, 36, 32, 8, 128, Float32(1e-6), Float64(1e6))

    @staticmethod
    def klein_9b() -> Qwen3Config:
        """Klein 9B text encoder (Qwen3-8B, hidden=4096)."""
        return Qwen3Config(4096, 36, 32, 8, 128, Float32(1e-6), Float64(1e6))


# ── encoder-local glue kernels (NOT foundation ops) ─────────────────────────
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
    seq: Int, h: Int, h_kv: Int, dh: Int, n_rep: Int,
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
    seq: Int, h: Int, h_kv: Int, dh: Int, n_rep: Int,
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
    seq: Int, h: Int, h_kv: Int, dh: Int, n_rep: Int,
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


# ── glue helpers (host-side dispatch) ───────────────────────────────────────
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


def _repeat_kv(var x: Tensor, h: Int, h_kv: Int, ctx: DeviceContext) raises -> Tensor:
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
        return x^
    var dt = x.dtype().to_mojo_dtype()
    var out_n = seq * h * dh
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](out_n * x.dtype().byte_size())
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


def _reshape(x: Tensor, var shape: List[Int], ctx: DeviceContext) raises -> Tensor:
    """Row-major contiguous reshape (metadata only; bytes copied)."""
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
def _build_rope_tables(
    seq: Int, heads: Int, head_dim: Int, theta: Float64
) raises -> List[List[Float32]]:
    """RoPE cos/sin in (position, head) row order; half-split angles."""
    var half = head_dim // 2
    var cos_vals = List[Float32]()
    var sin_vals = List[Float32]()
    var log_theta = flog(Float32(theta))
    for t in range(seq):
        for _h in range(heads):
            for i in range(half):
                var exponent = -log_theta * Float32(2 * i) / Float32(head_dim)
                var inv_freq = fexp(exponent)
                var angle = Float32(t) * inv_freq
                cos_vals.append(fcos(angle))
                sin_vals.append(fsin(angle))
    var out = List[List[Float32]]()
    out.append(cos_vals^)
    out.append(sin_vals^)
    return out^


def _build_causal_mask(seq: Int, heads: Int, real_len: Int) raises -> List[Float32]:
    """Additive causal mask [1,H,N,N]: 0 where j<=i and j<real_len else -1e4."""
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
    """Qwen3 text encoder (Klein). Owns all weights (ArcPointer because Tensor is
    Movable-not-Copyable). Forward runs on GPU."""

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
    def load(dir: String, config: Qwen3Config, ctx: DeviceContext) raises -> Qwen3Encoder:
        """Load all tensors from a sharded text_encoder dir into GPU Tensors."""
        var sharded = ShardedSafeTensors.open(dir)
        var weights = List[ArcPointer[Tensor]]()
        var name_to_idx = Dict[String, Int]()
        for ref nm in sharded.names():
            var tv = sharded.tensor_view(nm)
            var t = Tensor.from_view(tv, ctx)
            var idx = len(weights)
            weights.append(ArcPointer(t^))
            name_to_idx[nm] = idx
        return Qwen3Encoder(weights^, name_to_idx^, config)

    def _w(self, name: String) raises -> ref [self.weights] Tensor:
        if name not in self.name_to_idx:
            raise Error(String("missing weight: ") + name)
        var idx = self.name_to_idx[name]
        return self.weights[idx][]

    def _has(self, name: String) -> Bool:
        return name in self.name_to_idx

    # ── embedding ────────────────────────────────────────────────────────────
    def _embed(self, ids: List[Int], ctx: DeviceContext) raises -> Tensor:
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
        var out_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](seq * hidden))
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
        var q = linear(normed, qw, None, ctx)
        var k = linear(normed, kw, None, ctx)
        var v = linear(normed, vw, None, ctx)

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

        # per-head QK-norm (RMSNorm over Dh).
        ref qn = self._w(p + ".self_attn.q_norm.weight")
        ref kn = self._w(p + ".self_attn.k_norm.weight")
        q = rms_norm(q, qn, eps, ctx)
        k = rms_norm(k, kn, eps, ctx)

        # RoPE half-split on q,k.
        q = rope_halfsplit(q, cos_q, sin_q, ctx)
        k = rope_halfsplit(k, cos_k, sin_k, ctx)

        # GQA: repeat kv heads to H, then SDPA in BSHD.
        var k_rep = _repeat_kv(k^, h, h_kv, ctx)
        var v_rep = _repeat_kv(v^, h, h_kv, ctx)

        var attn = _sdpa_dispatch(q, k_rep, v_rep, mask, scale, seq, h, dh, ctx)

        var attn_sh = List[Int]()
        attn_sh.append(1)
        attn_sh.append(seq)
        attn_sh.append(h * dh)
        attn = _reshape(attn, attn_sh^, ctx)

        ref ow = self._w(p + ".self_attn.o_proj.weight")
        var attn_out = linear(attn, ow, None, ctx)

        var hidden2 = _add(hidden, attn_out, ctx)

        # --- MLP (SwiGLU) ---
        ref post_ln = self._w(p + ".post_attention_layernorm.weight")
        var normed2 = rms_norm(hidden2, post_ln, eps, ctx)
        ref gw = self._w(p + ".mlp.gate_proj.weight")
        ref uw = self._w(p + ".mlp.up_proj.weight")
        ref dw = self._w(p + ".mlp.down_proj.weight")
        var gate = linear(normed2, gw, None, ctx)
        var up = linear(normed2, uw, None, ctx)
        var act = swiglu(gate, up, ctx)
        var mlp_out = linear(act, dw, None, ctx)

        return _add(hidden2, mlp_out, ctx)

    # ── full forward ──────────────────────────────────────────────────────────
    def encode_layer_states(
        self, token_ids: List[Int], ctx: DeviceContext
    ) raises -> List[ArcPointer[Tensor]]:
        """Hidden states AFTER each layer (index i = output of layer i),
        PRE-final-norm. Index 0..num_layers-1."""
        var cfg = self.config
        var seq = len(token_ids)
        var h = cfg.num_heads
        var h_kv = cfg.num_kv_heads
        var dh = cfg.head_dim

        var real_len = seq
        for i in range(seq):
            if token_ids[i] == KLEIN_PAD_ID:
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
            hidden = self._layer(i, hidden, cos_q, sin_q, cos_k, sin_k, mask, ctx)
            states.append(ArcPointer(_clone(hidden, ctx)))
        return states^

    def encode(
        self, token_ids: List[Int], extract_layer: Int, ctx: DeviceContext
    ) raises -> Tensor:
        """Hidden state after `extract_layer` (PRE-final-norm) [1, seq, hidden]."""
        var states = self.encode_layer_states(token_ids, ctx)
        if extract_layer < 0 or extract_layer >= len(states):
            raise Error("encode: extract_layer out of range")
        return _clone(states[extract_layer][], ctx)

    def encode_klein(self, token_ids: List[Int], ctx: DeviceContext) raises -> Tensor:
        """Klein conditioning: concat(hidden_states[9,18,27], dim=2) =
        concat(post-layer states [8,17,26], dim=2) -> [1, seq, 3*hidden_size].
        Reproduces Flux2Model.encode_text Klein branch (Flux2Model.py:224-225)."""
        var states = self.encode_layer_states(token_ids, ctx)
        if len(states) <= KLEIN_EXTRACT_2:
            raise Error("encode_klein: encoder has fewer than 27 layers")
        var h8 = _clone(states[KLEIN_EXTRACT_0][], ctx)
        var h17 = _clone(states[KLEIN_EXTRACT_1][], ctx)
        var h26 = _clone(states[KLEIN_EXTRACT_2][], ctx)
        return concat(2, ctx, h8, h17, h26)

    def final_norm(self, x: Tensor, ctx: DeviceContext) raises -> Tensor:
        """Apply model.norm (RMSNorm). NOT used for the Klein [9,18,27] extraction
        (those are pre-final-norm hidden states)."""
        ref nw = self._w(String("model.norm.weight"))
        return rms_norm(x, nw, self.config.rms_norm_eps, ctx)


# ── SDPA dispatch (B/S/H/Dh comptime in the foundation sdpa) ────────────────
def _sdpa_dispatch(
    q: Tensor, k: Tensor, v: Tensor, mask: Tensor, scale: Float32,
    seq: Int, h: Int, dh: Int, ctx: DeviceContext,
) raises -> Tensor:
    # Klein 4B/9B: H=32, Dh=128. Enumerate supported sequence lengths at comptime.
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
    raise Error(
        String("sdpa_dispatch: unsupported (seq,h,dh)=(")
        + String(seq) + "," + String(h) + "," + String(dh)
        + "). Add a comptime case."
    )


def _clone(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    var nbytes = x.nbytes()
    var dev = ctx.enqueue_create_buffer[DType.uint8](nbytes)
    ctx.enqueue_copy(dst_buf=dev, src_buf=x.buf)
    ctx.synchronize()
    return Tensor(dev^, x.shape(), x.dtype())


# ════════════════════════════════════════════════════════════════════════════
# KleinChatTokenizer — PORT-OWNED chat-template seam over the foundation BPE.
# Mirrors Flux2Model.encode_text Klein steps 1-3: qwen3_format_input ->
# apply_chat_template(add_generation_prompt=True, enable_thinking=False) ->
# tokenize. The Qwen3 non-thinking template wraps the prompt as:
#   <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n
# enable_thinking=False means NO <think>...</think> block is injected; the
# assistant turn is left open for the encoder to read conditioning.
# ════════════════════════════════════════════════════════════════════════════
struct KleinChatTokenizer(Movable):
    var bpe: Qwen3Tokenizer

    def __init__(out self, tokenizer_json_path: String) raises:
        self.bpe = Qwen3Tokenizer(tokenizer_json_path)

    def apply_chat_template(self, prompt: String) -> String:
        """Klein Qwen3 chat template, add_generation_prompt=True,
        enable_thinking=False (Flux2Model.py:197-203).

        VERIFIED against transformers AutoTokenizer for Qwen3-8B / Qwen3-4B
        (apply_chat_template([{user}], add_generation_prompt=True,
        enable_thinking=False)). The non-thinking template appends an EMPTY
        think block "<think>\\n\\n</think>\\n\\n" after the open assistant turn:
          <|im_start|>user\\n{prompt}<|im_end|>\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n
        This suffix is part of the canonical Qwen3 template and changes the
        tokenized length/positions, so it MUST be included to match Serenity.
        """
        return (
            String("<|im_start|>user\n") + prompt
            + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )

    def encode(self, prompt: String) raises -> List[Int]:
        """Templated prompt -> token ids (Flux2Model.encode_text steps 1-3)."""
        return self.bpe.encode(self.apply_chat_template(prompt))

    def encode_raw(self, text: String) raises -> List[Int]:
        """Tokenize an already-templated string (no template applied)."""
        return self.bpe.encode(text)


# ── high-level port seam: text_encode(prompt) -> Klein conditioning ─────────
# Mirrors Flux2Model.encode_text (Klein branch) for ONE prompt at a fixed
# padded seq length. Returns the FULL [1, pad_to_seq, 3*HIDDEN] conditioning —
# Flux2.predict feeds the whole max_length-padded tensor to the transformer (NO
# mask-narrowing; that is a Z-Image-only behavior). pad_to_seq corresponds to
# config.text_encoder_sequence_length and must be a supported SDPA length.
def text_encode(
    tok: KleinChatTokenizer,
    enc: Qwen3Encoder,
    prompt: String,
    pad_to_seq: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    """prompt -> Klein conditioning [1, pad_to_seq, 3*hidden_size].

    Right-pad with PAD_ID to pad_to_seq (== text_encoder_sequence_length),
    encode, then concat hidden_states[9,18,27] (post-layer states [8,17,26])
    on dim 2. No mask-narrowing — the full padded tensor is the conditioning."""
    var ids_full = tok.encode(prompt)
    var real_len = len(ids_full)
    if real_len > pad_to_seq:
        raise Error(
            String("text_encode: prompt tokens=") + String(real_len)
            + " exceed pad_to_seq=" + String(pad_to_seq)
        )
    var ids = List[Int](capacity=pad_to_seq)
    for i in range(real_len):
        ids.append(ids_full[i])
    for _ in range(pad_to_seq - real_len):
        ids.append(KLEIN_PAD_ID)
    return enc.encode_klein(ids, ctx)
