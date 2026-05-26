# models/text_encoder/clip_encoder.mojo — CLIP-L + CLIP-G text encoders (GPU).
#
# Pure-Mojo, inference-only port of
#   inference-flame/src/models/clip_encoder.rs (read FULL, 614 L)
# and the SDXL embedding assembly in
#   inference-flame/src/bin/sdxl_encode.rs (read FULL, 224 L).
#
# Two configs share one struct (CLIP transformer is identical in shape):
#   CLIP-L: vocab 49408, hidden 768,  layers 12, heads 12, head_dim 64,
#           intermediate 3072,  quick_gelu, eos/pad id 49407, max_pos 77.
#   CLIP-G: vocab 49408, hidden 1280, layers 32, heads 20, head_dim 64,
#           intermediate 5120,  standard gelu, eos/pad id 49407, max_pos 77.
#           (OpenCLIP ViT-bigG-14; ships a `text_projection.weight` outside
#            `text_model.*` — caller applies it; see SDXL `y` assembly note.)
#
# Per-layer (clip_encoder.rs:201 layer_forward):
#   residual = h
#   h = layer_norm1(h) -> q/k/v Linear(+bias) -> reshape BSHD [1,77,H,64]
#     -> sdpa(causal mask, scale 1/sqrt(64)) -> reshape [1,77,H*64]
#     -> out_proj(+bias) -> residual + h
#   residual = h
#   h = layer_norm2(h) -> fc1(+bias) -> (quick_gelu|gelu) -> fc2(+bias)
#     -> residual + h
# Final: layer_norm(final_layer_norm) over the full last hidden state, then the
# pooled vector is the EOS-position row of that post-LN state.
#
# OUTPUT (encode_sdxl): (last_hidden_state [1,77,hidden], pooled [1,hidden]).
# The SDXL `context` is cat([clip_l_hidden, clip_g_hidden], dim=2) -> [1,77,2048].
# The SDXL `y` is cat([clip_l_pool, clip_g_text_embeds, zeros[768]], dim=1)
#   -> [1,2816]; clip_g_text_embeds = clip_g_pool @ text_projection^T.
# (Assembly lives in the pipeline, not here — this file only encodes.)
#
# Foundation ops reused: linear, layer_norm, sdpa, gelu, tensor_algebra.{add,
# slice, mul}. CLIP-LOCAL glue (not a foundation op): token+position embedding
# gather (host-built ids -> gather_rows-style kernel), causal mask (host F32),
# quick_gelu (x * sigmoid(1.702x); foundation ships NO sigmoid op — flagged).
#
# Mojo 1.0.0b1, NVIDIA GPU. BF16 storage, F32 accumulation.

from std.math import exp as fexp
from std.memory import ArcPointer
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu import global_idx
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.linear import linear
from serenitymojo.ops.norm import layer_norm
from serenitymojo.ops.attention import sdpa
from serenitymojo.ops.activations import gelu
from serenitymojo.ops.tensor_algebra import add, reshape


comptime _DYN1 = Layout.row_major(-1)
comptime _BLOCK = 256
comptime MAX_LEN = 77
comptime EOS_ID = 49407
comptime HEAD_DIM = 64


# ── Config ──────────────────────────────────────────────────────────────────
@fieldwise_init
struct ClipConfig(Copyable, Movable, ImplicitlyCopyable):
    """CLIP text-encoder hyperparameters."""

    var vocab_size: Int
    var hidden_size: Int
    var num_layers: Int
    var intermediate_size: Int
    var num_heads: Int
    var head_dim: Int
    var layer_norm_eps: Float32
    var use_quick_gelu: Bool  # CLIP-L=True, CLIP-G=False

    @staticmethod
    def clip_l() -> ClipConfig:
        """CLIP-L: 768-dim, 12 layers, 12 heads, quick_gelu."""
        return ClipConfig(49408, 768, 12, 3072, 12, 64, Float32(1e-5), True)

    @staticmethod
    def clip_g() -> ClipConfig:
        """CLIP-G: 1280-dim, 32 layers, 20 heads, standard gelu."""
        return ClipConfig(49408, 1280, 32, 5120, 20, 64, Float32(1e-5), False)


# ── CLIP-local glue kernels ─────────────────────────────────────────────────
#
# quick_gelu: o = x * sigmoid(1.702 * x). The foundation ships silu/gelu/swiglu
# but NOT a standalone sigmoid, so this is CLIP-local. F32 math, store-dtype out.
def _quick_gelu_kernel_bf16(
    x: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        var v = rebind[Scalar[DType.bfloat16]](x[i]).cast[DType.float32]()
        var s = Float32(1.0) / (Float32(1.0) + fexp(-Float32(1.702) * v))
        o[i] = rebind[o.element_type]((v * s).cast[DType.bfloat16]())


def _quick_gelu_kernel_f32(
    x: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        var v = rebind[Scalar[DType.float32]](x[i])
        var s = Float32(1.0) / (Float32(1.0) + fexp(-Float32(1.702) * v))
        o[i] = rebind[o.element_type](v * s)


def _quick_gelu_kernel_f16(
    x: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        var v = rebind[Scalar[DType.float16]](x[i]).cast[DType.float32]()
        var s = Float32(1.0) / (Float32(1.0) + fexp(-Float32(1.702) * v))
        o[i] = rebind[o.element_type]((v * s).cast[DType.float16]())


def _quick_gelu(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    var dt = x.dtype().to_mojo_dtype()
    var n = x.numel()
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
    var rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var grid = (n + _BLOCK - 1) // _BLOCK
    if dt == DType.float32:
        var X = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float32](), rl
        )
        var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), rl
        )
        ctx.enqueue_function[_quick_gelu_kernel_f32, _quick_gelu_kernel_f32](
            X, O, n, grid_dim=grid, block_dim=_BLOCK
        )
    elif dt == DType.bfloat16:
        var X = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[BFloat16](), rl
        )
        var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), rl
        )
        ctx.enqueue_function[_quick_gelu_kernel_bf16, _quick_gelu_kernel_bf16](
            X, O, n, grid_dim=grid, block_dim=_BLOCK
        )
    else:
        var X = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float16](), rl
        )
        var O = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), rl
        )
        ctx.enqueue_function[_quick_gelu_kernel_f16, _quick_gelu_kernel_f16](
            X, O, n, grid_dim=grid, block_dim=_BLOCK
        )
    ctx.synchronize()
    return Tensor(out_buf^, x.shape(), x.dtype())


# Token+position embedding gather. One thread per output element:
#   o[t, j] = token_table[ids[t], j] + pos_table[t, j].
# ids is an I32 device buffer; both tables are [V|max_pos, hidden] same dtype.
def _embed_kernel_bf16(
    tok: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    pos: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
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
        var tk = Int(rebind[Scalar[DType.int32]](ids[t]))
        var tv = rebind[Scalar[DType.bfloat16]](tok[tk * hidden + j]).cast[
            DType.float32
        ]()
        var pv = rebind[Scalar[DType.bfloat16]](pos[t * hidden + j]).cast[
            DType.float32
        ]()
        o[idx] = rebind[o.element_type]((tv + pv).cast[DType.bfloat16]())


def _embed_kernel_f32(
    tok: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    pos: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
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
        var tk = Int(rebind[Scalar[DType.int32]](ids[t]))
        var tv = rebind[Scalar[DType.float32]](tok[tk * hidden + j])
        var pv = rebind[Scalar[DType.float32]](pos[t * hidden + j])
        o[idx] = rebind[o.element_type](tv + pv)


def _embed_kernel_f16(
    tok: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
    pos: LayoutTensor[DType.float16, _DYN1, MutAnyOrigin],
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
        var tk = Int(rebind[Scalar[DType.int32]](ids[t]))
        var tv = rebind[Scalar[DType.float16]](tok[tk * hidden + j]).cast[
            DType.float32
        ]()
        var pv = rebind[Scalar[DType.float16]](pos[t * hidden + j]).cast[
            DType.float32
        ]()
        o[idx] = rebind[o.element_type]((tv + pv).cast[DType.float16]())


# Additive causal + key-padding mask [1, H, 77, 77] (clip_encoder.rs build_pad_mask):
# attend (0.0) iff j <= i (causal) AND j <= valid_key_end (no pad keys), else a
# large negative driving softmax to ~0. Host F32 -> uploaded as compute dtype.
def _build_pad_mask(
    seq: Int, heads: Int, valid_key_end: Int
) raises -> List[Float32]:
    var neg = Float32(-1.0e4)
    var data = List[Float32]()
    for _h in range(heads):
        for i in range(seq):
            for j in range(seq):
                if j <= i and j <= valid_key_end:
                    data.append(Float32(0.0))
                else:
                    data.append(neg)
    return data^


# ── ClipEncoder ──────────────────────────────────────────────────────────────
struct ClipEncoder:
    """CLIP text encoder. Owns all `text_model.*` weights (ArcPointer because
    Tensor is Movable-not-Copyable). Forward runs on GPU. `text_projection`
    (CLIP-G only) lives outside `text_model.*` and is loaded/applied by the
    pipeline, not this struct."""

    var weights: List[ArcPointer[Tensor]]
    var name_to_idx: Dict[String, Int]
    var config: ClipConfig

    def __init__(
        out self,
        var weights: List[ArcPointer[Tensor]],
        var name_to_idx: Dict[String, Int],
        config: ClipConfig,
    ):
        self.weights = weights^
        self.name_to_idx = name_to_idx^
        self.config = config

    @staticmethod
    def load(
        dir_or_file: String, config: ClipConfig, ctx: DeviceContext
    ) raises -> ClipEncoder:
        """Load all `text_model.*` tensors from a CLIP safetensors file/dir into
        GPU Tensors via ShardedSafeTensors + Tensor.from_view. Non-`text_model.*`
        keys (e.g. CLIP-G's `text_projection.weight`, `logit_scale`) are skipped
        here — the pipeline loads `text_projection` separately when needed."""
        var sharded = ShardedSafeTensors.open(dir_or_file)
        var weights = List[ArcPointer[Tensor]]()
        var name_to_idx = Dict[String, Int]()
        for ref nm in sharded.names():
            if not nm.startswith("text_model."):
                continue
            var tv = sharded.tensor_view(nm)
            var t = Tensor.from_view(tv, ctx)
            var idx = len(weights)
            weights.append(ArcPointer(t^))
            name_to_idx[nm] = idx
        return ClipEncoder(weights^, name_to_idx^, config)

    def _w(self, name: String) raises -> ref [self.weights] Tensor:
        if name not in self.name_to_idx:
            raise Error(String("missing CLIP weight: ") + name)
        var idx = self.name_to_idx[name]
        return self.weights[idx][]

    # ── token + position embeddings ─────────────────────────────────────────
    def _embed(self, ids: List[Int], ctx: DeviceContext) raises -> Tensor:
        """Gather token rows + add position rows -> [1, seq, hidden]."""
        ref tok_table = self._w(
            String("text_model.embeddings.token_embedding.weight")
        )
        ref pos_table = self._w(
            String("text_model.embeddings.position_embedding.weight")
        )
        var hidden = self.config.hidden_size
        var seq = len(ids)
        var dt = tok_table.dtype().to_mojo_dtype()

        # Upload token ids as an I32 device buffer.
        var id_host = ctx.enqueue_create_host_buffer[DType.uint8](seq * 4)
        var ip = id_host.unsafe_ptr().bitcast[Int32]()
        for i in range(seq):
            ip[i] = Int32(ids[i])
        var id_dev = ctx.enqueue_create_buffer[DType.uint8](seq * 4)
        ctx.enqueue_copy(dst_buf=id_dev, src_buf=id_host)
        ctx.synchronize()

        var out_buf = ctx.enqueue_create_buffer[DType.uint8](
            seq * hidden * tok_table.dtype().byte_size()
        )
        var tok_rl = RuntimeLayout[_DYN1].row_major(
            IndexList[1](tok_table.numel())
        )
        var pos_rl = RuntimeLayout[_DYN1].row_major(
            IndexList[1](pos_table.numel())
        )
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
            var TK = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
                tok_table.buf.unsafe_ptr().bitcast[Float32](), tok_rl
            )
            var PO = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
                pos_table.buf.unsafe_ptr().bitcast[Float32](), pos_rl
            )
            var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
                out_buf.unsafe_ptr().bitcast[Float32](), out_rl
            )
            ctx.enqueue_function[_embed_kernel_f32, _embed_kernel_f32](
                TK, PO, IDS, O, seq, hidden, grid_dim=grid, block_dim=_BLOCK
            )
        elif dt == DType.bfloat16:
            var TK = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
                tok_table.buf.unsafe_ptr().bitcast[BFloat16](), tok_rl
            )
            var PO = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
                pos_table.buf.unsafe_ptr().bitcast[BFloat16](), pos_rl
            )
            var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
                out_buf.unsafe_ptr().bitcast[BFloat16](), out_rl
            )
            ctx.enqueue_function[_embed_kernel_bf16, _embed_kernel_bf16](
                TK, PO, IDS, O, seq, hidden, grid_dim=grid, block_dim=_BLOCK
            )
        else:
            var TK = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
                tok_table.buf.unsafe_ptr().bitcast[Float16](), tok_rl
            )
            var PO = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
                pos_table.buf.unsafe_ptr().bitcast[Float16](), pos_rl
            )
            var O = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
                out_buf.unsafe_ptr().bitcast[Float16](), out_rl
            )
            ctx.enqueue_function[_embed_kernel_f16, _embed_kernel_f16](
                TK, PO, IDS, O, seq, hidden, grid_dim=grid, block_dim=_BLOCK
            )
        ctx.synchronize()
        var sh = List[Int]()
        sh.append(1)
        sh.append(seq)
        sh.append(hidden)
        return Tensor(out_buf^, sh^, tok_table.dtype())

    # ── one transformer layer ──────────────────────────────────────────────
    # S is a comptime param (77) so the comptime-shaped sdpa can be called.
    def _layer[
        S: Int
    ](
        self, layer_idx: Int, hidden: Tensor, mask: Tensor, ctx: DeviceContext
    ) raises -> Tensor:
        var cfg = self.config
        var h = cfg.num_heads
        var d = cfg.head_dim
        var hid = cfg.hidden_size
        var eps = cfg.layer_norm_eps
        var scale = Float32(1.0) / Float32(8.0)  # 1/sqrt(64)
        var p = String("text_model.encoder.layers.") + String(layer_idx)

        # --- self-attention ---
        ref ln1w = self._w(p + ".layer_norm1.weight")
        ref ln1b = self._w(p + ".layer_norm1.bias")
        var normed = layer_norm(hidden, ln1w, ln1b, eps, ctx)

        ref qw = self._w(p + ".self_attn.q_proj.weight")
        ref qb = self._w(p + ".self_attn.q_proj.bias")
        ref kw = self._w(p + ".self_attn.k_proj.weight")
        ref kb = self._w(p + ".self_attn.k_proj.bias")
        ref vw = self._w(p + ".self_attn.v_proj.weight")
        ref vb = self._w(p + ".self_attn.v_proj.bias")
        var q = linear(normed, qw, Optional[Tensor](_clone(qb, ctx)), ctx)
        var k = linear(normed, kw, Optional[Tensor](_clone(kb, ctx)), ctx)
        var v = linear(normed, vw, Optional[Tensor](_clone(vb, ctx)), ctx)

        # reshape [1,S,hid] -> BSHD [1,S,H,Dh]
        var bshd = List[Int]()
        bshd.append(1)
        bshd.append(S)
        bshd.append(h)
        bshd.append(d)
        q = reshape(q, bshd.copy(), ctx)
        k = reshape(k, bshd.copy(), ctx)
        v = reshape(v, bshd.copy(), ctx)

        # CLIP self-attention is square (q-seq == kv-seq == S); heads vary by
        # config (12 vs 20), so _sdpa_clip dispatches on (S, H) at comptime.
        var attn = _sdpa_clip[S](q, k, v, mask, scale, h, ctx)

        # [1,S,H,Dh] -> [1,S,hid]
        var flat = List[Int]()
        flat.append(1)
        flat.append(S)
        flat.append(hid)
        attn = reshape(attn, flat^, ctx)

        ref ow = self._w(p + ".self_attn.out_proj.weight")
        ref ob = self._w(p + ".self_attn.out_proj.bias")
        var attn_out = linear(attn, ow, Optional[Tensor](_clone(ob, ctx)), ctx)
        var h1 = add(hidden, attn_out, ctx)

        # --- MLP ---
        ref ln2w = self._w(p + ".layer_norm2.weight")
        ref ln2b = self._w(p + ".layer_norm2.bias")
        var normed2 = layer_norm(h1, ln2w, ln2b, eps, ctx)

        ref f1w = self._w(p + ".mlp.fc1.weight")
        ref f1b = self._w(p + ".mlp.fc1.bias")
        ref f2w = self._w(p + ".mlp.fc2.weight")
        ref f2b = self._w(p + ".mlp.fc2.bias")
        var mh = linear(normed2, f1w, Optional[Tensor](_clone(f1b, ctx)), ctx)
        if cfg.use_quick_gelu:
            mh = _quick_gelu(mh, ctx)
        else:
            mh = gelu(mh, ctx)
        var mlp_out = linear(mh, f2w, Optional[Tensor](_clone(f2b, ctx)), ctx)
        return add(h1, mlp_out, ctx)

    # ── full forward (SDXL): (last_hidden_state, pooled) ─────────────────────
    # SDXL token ids are right-padded to 77 with EOS_ID; the pooled vector is
    # the post-final-LN hidden state at the FIRST EOS position (argmax over
    # id==EOS), matching HF CLIPTextTransformer.pooler_output. Comptime S=77.
    def encode_sdxl[
        S: Int = MAX_LEN
    ](
        self, var token_ids: List[Int], ctx: DeviceContext
    ) raises -> Tuple[Tensor, Tensor]:
        var cfg = self.config

        # pad / truncate to S, EOS-pad.
        if len(token_ids) > S:
            var trimmed = List[Int]()
            for i in range(S):
                trimmed.append(token_ids[i])
            trimmed[S - 1] = EOS_ID
            token_ids = trimmed^
        else:
            while len(token_ids) < S:
                token_ids.append(EOS_ID)

        # first EOS position (argmax over id==eos returns the FIRST 1).
        var real_eos = S - 1
        for i in range(S):
            if token_ids[i] == EOS_ID:
                real_eos = i
                break

        var hidden = self._embed(token_ids, ctx)
        var dtype = hidden.dtype()

        # combined causal + key-padding mask [1, H, S, S].
        var mask_data = _build_pad_mask(S, cfg.num_heads, real_eos)
        var msh = List[Int]()
        msh.append(1)
        msh.append(cfg.num_heads)
        msh.append(S)
        msh.append(S)
        var mask = Tensor.from_host(mask_data, msh^, dtype, ctx)

        for i in range(cfg.num_layers):
            hidden = self._layer[S](i, hidden, mask, ctx)

        # final layer norm over the full last hidden state.
        ref fw = self._w(String("text_model.final_layer_norm.weight"))
        ref fb = self._w(String("text_model.final_layer_norm.bias"))
        var last_hidden = layer_norm(hidden, fw, fb, cfg.layer_norm_eps, ctx)

        # pooled = post-LN hidden at the first EOS position, [1, hidden].
        var pooled_slice = slice_seq(last_hidden, real_eos, ctx)  # [1,1,hidden]
        var psh = List[Int]()
        psh.append(1)
        psh.append(cfg.hidden_size)
        var pooled = reshape(pooled_slice, psh^, ctx)
        return (last_hidden^, pooled^)


# ── helpers ──────────────────────────────────────────────────────────────────
# Deep-copy a Tensor's device buffer (Tensor is Movable-not-Copyable, so weight
# `ref`s can't be passed where an owned Tensor is needed — clone instead).
def _clone(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    var dev = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
    ctx.enqueue_copy(dst_buf=dev, src_buf=x.buf)
    ctx.synchronize()
    return Tensor(dev^, x.shape(), x.dtype())


# Slice token `idx` out of [1, S, hidden] -> [1, 1, hidden] via the foundation
# `slice` (narrow dim=1, length 1).
from serenitymojo.ops.tensor_algebra import slice as ta_slice


def slice_seq(x: Tensor, idx: Int, ctx: DeviceContext) raises -> Tensor:
    return ta_slice(x, 1, idx, 1, ctx)


# CLIP self-attention is SQUARE (q-seq == kv-seq == S), so the foundation
# comptime sdpa applies directly. Heads differ by config (12 for CLIP-L, 20 for
# CLIP-G) so we dispatch on (S, H) at comptime. Dh is always 64 (flash path).
def _sdpa_clip[
    S: Int
](
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Tensor,
    scale: Float32,
    h: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    if h == 12:
        return sdpa[1, S, 12, 64](q, k, v, mask, scale, ctx)
    if h == 20:
        return sdpa[1, S, 20, 64](q, k, v, mask, scale, ctx)
    raise Error(
        String("_sdpa_clip: unsupported head count ") + String(h)
        + " (expected 12 CLIP-L or 20 CLIP-G)"
    )
