# models/dit/zimage_dit.mojo — Z-Image NextDiT transformer forward (GPU).
#
# Pure-Mojo, inference-only port of the Z-Image diffusion transformer. The
# REFERENCE is the diffusers implementation (the parity oracle), read line by
# line, NOT the flame-core Rust (which differs in a few conventions — t-embed
# (1-t) inversion, [cap,image] concat order, final negate). Diffusers source:
#   diffusers/models/transformers/transformer_z_image.py (ZImageTransformer2DModel)
#
# Architecture (basic / non-omni mode):
#   adaln_input = t_embedder(t * t_scale)                        # [1, 256]
#   patchify image (C,F=1,H,W) p=2 -> [img_tokens, 64] ; pad to mult 32
#   x = all_x_embedder(x_patches) ; substitute x_pad_token at padded rows
#   noise_refiner (2 blocks, modulated) on x  (image self-attn)
#   cap = cap_embedder(cap_feats)=RMSNorm+Linear ; pad to mult 32 ; cap_pad_token
#   context_refiner (2 blocks, UNMODULATED) on cap (text self-attn)
#   unified = concat([x, cap], dim=1)                            # basic order
#   main layers (30 blocks, modulated) on unified
#   final_layer: LayerNorm(no-affine) * (1 + Linear(SiLU(adaln_input))) -> Linear
#   take image tokens (first img_tokens of unified), unpatchify -> velocity
#
# Block (modulated):  mod = adaLN_modulation.0(adaln_input)  -> chunk4
#     scale_msa,gate_msa,scale_mlp,gate_mlp ; gate=tanh(gate); scale=1+scale
#   attn_out = attention( attention_norm1(x) * scale_msa )
#   x = x + gate_msa * attention_norm2(attn_out)
#   x = x + gate_mlp * ffn_norm2( feed_forward( ffn_norm1(x) * scale_mlp ) )
# Block (unmodulated, context_refiner):
#   x = x + attention_norm2( attention(attention_norm1(x)) )
#   x = x + ffn_norm2( feed_forward(ffn_norm1(x)) )
#
# Attention (ZSingleStreamAttnProcessor): q/k/v = to_q/to_k/to_v(x) ;
#   unflatten -> [.., H, Dh] ; norm_q/norm_k = RMSNorm(Dh, eps=1e-5) ;
#   RoPE interleaved (complex view) on q,k ; SDPA (no mask, full) ; flatten ;
#   to_out.0.
#
# RoPE: per-axis (axes_dims=[32,48,48], theta=256) interleaved complex form.
#   freqs_cis[token] = concat over axes of cis(pos[axis] * inv_freq[axis][i]).
#   inv_freq[axis][i] = theta^(-2i / axis_dim), i in [0, axis_dim/2).
#   Applied as INTERLEAVED RoPE (foundation rope_interleaved): pair (x[2i],x[2i+1]).
#
# Config: dim=3840, n_heads=30, head_dim=128, n_layers=30, n_refiner=2,
#   cap_feat_dim=2560, norm_eps=1e-5, rope_theta=256, t_scale=1000,
#   axes_dims=[32,48,48], axes_lens=[1536,512,512]. qk_norm eps=1e-5,
#   final norm_final eps=1e-6.
#
# Compile-time params HL, WL, CAPLEN make the sequence lengths comptime so the
# foundation sdpa[B,S,H,Dh] (comptime-shaped) can be dispatched.
#
# Mojo 1.0.0b1, NVIDIA GPU. BF16 storage, F32 accumulation in foundation ops.

from std.math import cos as fcos, sin as fsin, exp as fexp, log as flog, sqrt, tanh as ftanh
from std.memory import ArcPointer
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu import global_idx
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.norm import rms_norm, layer_norm
from serenitymojo.ops.rope import rope_interleaved
from serenitymojo.ops.attention import sdpa
from serenitymojo.ops.linear import linear
from serenitymojo.ops.activations import silu, swiglu
from serenitymojo.ops.tensor_algebra import add, mul, concat, slice, reshape, add_scalar, permute


comptime _DYN1 = Layout.row_major(-1)
comptime _BLOCK = 256


# ── Config ──────────────────────────────────────────────────────────────────
@fieldwise_init
struct NextDiTConfig(Copyable, Movable, ImplicitlyCopyable):
    """Z-Image NextDiT hyperparameters (basic / non-omni)."""

    var dim: Int
    var n_heads: Int
    var head_dim: Int
    var n_layers: Int
    var n_refiner: Int
    var cap_feat_dim: Int
    var norm_eps: Float32
    var rope_theta: Float64
    var t_scale: Float32
    var patch_size: Int
    var in_channels: Int
    var adaln_embed_dim: Int  # min(dim, 256) = 256
    # axes_dims are fixed [32,48,48]; stored individually for kernels.
    var axis0: Int
    var axis1: Int
    var axis2: Int

    @staticmethod
    def zimage() -> NextDiTConfig:
        return NextDiTConfig(
            3840, 30, 128, 30, 2, 2560,
            # t_scale=1000: diffusers transformer does t_embedder(t * t_scale)
            # (transformer_z_image.py:916, t_scale=1000.0). Confirmed correct.
            Float32(1e-5), Float64(256.0), Float32(1000.0),
            2, 16, 256, 32, 48, 48,
        )


# ── local glue kernels (NOT foundation ops) ─────────────────────────────────
#
# pad-token substitution: rows [real_len, total_len) of a [total_len, dim]
# feature get overwritten with the pad_token [dim]. One thread per output elem.
def _padtok_kernel_bf16(
    feat: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    pad: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    total_len: Int, dim: Int, real_len: Int,
):
    var idx = Int(global_idx.x)
    var total = total_len * dim
    if idx < total:
        var r = idx // dim
        var c = idx % dim
        if r >= real_len:
            o[idx] = rebind[o.element_type](pad[c])
        else:
            o[idx] = rebind[o.element_type](feat[idx])


def _padtok_kernel_f32(
    feat: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    pad: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    total_len: Int, dim: Int, real_len: Int,
):
    var idx = Int(global_idx.x)
    var total = total_len * dim
    if idx < total:
        var r = idx // dim
        var c = idx % dim
        if r >= real_len:
            o[idx] = rebind[o.element_type](pad[c])
        else:
            o[idx] = rebind[o.element_type](feat[idx])


# tanh, elementwise (gates). F32 math.
def _tanh_kernel_bf16(
    x: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        var v = rebind[Scalar[DType.bfloat16]](x[i]).cast[DType.float32]()
        o[i] = rebind[o.element_type](ftanh(v).cast[DType.bfloat16]())


def _tanh_kernel_f32(
    x: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        var v = rebind[Scalar[DType.float32]](x[i])
        o[i] = rebind[o.element_type](ftanh(v))


# RoPE cos/sin pair (Tensor is not Copyable, so a Tuple can't be element-moved
# cleanly — a named struct lets the caller transfer each field out with `^`).
struct _RopePair(Movable):
    var cos: Tensor
    var sin: Tensor

    def __init__(out self, var cos: Tensor, var sin: Tensor):
        self.cos = cos^
        self.sin = sin^


# ── NextDiT ──────────────────────────────────────────────────────────────────
struct NextDiT[HL: Int, WL: Int, CAPLEN: Int]:
    """Z-Image NextDiT. Comptime-parameterized on latent H/W and caption length
    so the unified sequence length is a compile-time constant for the foundation
    `sdpa` dispatch. Weights stored as ArcPointer[Tensor] (Tensor is Movable but
    not Copyable, so List/Dict require the Arc indirection — same discipline as
    the text encoder)."""

    var weights: List[ArcPointer[Tensor]]
    var name_to_idx: Dict[String, Int]
    var config: NextDiTConfig

    def __init__(
        out self,
        var weights: List[ArcPointer[Tensor]],
        var name_to_idx: Dict[String, Int],
        config: NextDiTConfig,
    ):
        self.weights = weights^
        self.name_to_idx = name_to_idx^
        self.config = config

    @staticmethod
    def load(dir: String, ctx: DeviceContext) raises -> NextDiT[Self.HL, Self.WL, Self.CAPLEN]:
        """Load all 521 transformer tensors from a sharded dir into GPU Tensors
        via ShardedSafeTensors + Tensor.from_view (H2D copy)."""
        var sharded = ShardedSafeTensors.open(dir)
        var weights = List[ArcPointer[Tensor]]()
        var name_to_idx = Dict[String, Int]()
        for ref nm in sharded.names():
            var tv = sharded.tensor_view(nm)
            var t = Tensor.from_view(tv, ctx)
            var idx = len(weights)
            weights.append(ArcPointer(t^))
            name_to_idx[nm] = idx
        return NextDiT[Self.HL, Self.WL, Self.CAPLEN](weights^, name_to_idx^, NextDiTConfig.zimage())

    def _w(self, name: String) raises -> ref [self.weights] Tensor:
        if name not in self.name_to_idx:
            raise Error(String("missing weight: ") + name)
        var idx = self.name_to_idx[name]
        return self.weights[idx][]

    def _has(self, name: String) -> Bool:
        return name in self.name_to_idx

    def _dtype(self) raises -> STDtype:
        return self._w(String("x_pad_token")).dtype()

    # ── glue: tanh ─────────────────────────────────────────────────────────
    def _tanh(self, x: Tensor, ctx: DeviceContext) raises -> Tensor:
        var dt = x.dtype().to_mojo_dtype()
        var n = x.numel()
        var out_buf = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
        var rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
        var grid = (n + _BLOCK - 1) // _BLOCK
        if dt == DType.bfloat16:
            var X = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
                x.buf.unsafe_ptr().bitcast[BFloat16](), rl
            )
            var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
                out_buf.unsafe_ptr().bitcast[BFloat16](), rl
            )
            ctx.enqueue_function[_tanh_kernel_bf16, _tanh_kernel_bf16](
                X, O, n, grid_dim=grid, block_dim=_BLOCK
            )
        else:
            var X = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
                x.buf.unsafe_ptr().bitcast[Float32](), rl
            )
            var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
                out_buf.unsafe_ptr().bitcast[Float32](), rl
            )
            ctx.enqueue_function[_tanh_kernel_f32, _tanh_kernel_f32](
                X, O, n, grid_dim=grid, block_dim=_BLOCK
            )
        ctx.synchronize()
        return Tensor(out_buf^, x.shape(), x.dtype())

    # ── glue: substitute pad token in rows [real_len, total_len) ───────────
    def _apply_pad_token(
        self, var feat: Tensor, pad_key: String, real_len: Int, ctx: DeviceContext
    ) raises -> Tensor:
        # feat: [total_len, dim]. pad: [1, dim] or [dim].
        var fs = feat.shape()
        var total_len = fs[0]
        var dim = fs[1]
        if real_len >= total_len:
            return feat^
        ref pad = self._w(pad_key)
        var dt = feat.dtype().to_mojo_dtype()
        var out_buf = ctx.enqueue_create_buffer[DType.uint8](feat.nbytes())
        var rl = RuntimeLayout[_DYN1].row_major(IndexList[1](total_len * dim))
        var prl = RuntimeLayout[_DYN1].row_major(IndexList[1](dim))
        var grid = (total_len * dim + _BLOCK - 1) // _BLOCK
        if dt == DType.bfloat16:
            var F = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
                feat.buf.unsafe_ptr().bitcast[BFloat16](), rl
            )
            var P = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
                pad.buf.unsafe_ptr().bitcast[BFloat16](), prl
            )
            var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
                out_buf.unsafe_ptr().bitcast[BFloat16](), rl
            )
            ctx.enqueue_function[_padtok_kernel_bf16, _padtok_kernel_bf16](
                F, P, O, total_len, dim, real_len, grid_dim=grid, block_dim=_BLOCK
            )
        else:
            var F = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
                feat.buf.unsafe_ptr().bitcast[Float32](), rl
            )
            var P = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
                pad.buf.unsafe_ptr().bitcast[Float32](), prl
            )
            var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
                out_buf.unsafe_ptr().bitcast[Float32](), rl
            )
            ctx.enqueue_function[_padtok_kernel_f32, _padtok_kernel_f32](
                F, P, O, total_len, dim, real_len, grid_dim=grid, block_dim=_BLOCK
            )
        ctx.synchronize()
        return Tensor(out_buf^, feat.shape(), feat.dtype())

    # ── timestep embedder: timestep_embedding(t*t_scale,256) -> mlp ─────────
    # Sinusoidal: half=128; freq_i = exp(-ln(10000)*i/half); COS first then SIN.
    def _t_embedder(self, t_val: Float32, ctx: DeviceContext) raises -> Tensor:
        var dim = self.config.adaln_embed_dim  # 256
        var half = dim // 2
        var max_period = Float32(10000.0)
        var scaled = t_val * self.config.t_scale
        var emb = List[Float32]()
        var log_mp = flog(max_period)
        for i in range(half):
            var freq = fexp(-log_mp * Float32(i) / Float32(half))
            emb.append(fcos(scaled * freq))
        for i in range(half):
            var freq = fexp(-log_mp * Float32(i) / Float32(half))
            emb.append(fsin(scaled * freq))
        var dtype = self._w(String("t_embedder.mlp.0.weight")).dtype()
        var sh = List[Int]()
        sh.append(1)
        sh.append(dim)
        var t_freq = Tensor.from_host(emb, sh^, dtype, ctx)
        ref w0 = self._w(String("t_embedder.mlp.0.weight"))
        ref b0 = self._w(String("t_embedder.mlp.0.bias"))
        var h = linear(t_freq, w0, Optional[Tensor](self._clone(b0, ctx)), ctx)
        var ha = silu(h, ctx)
        ref w2 = self._w(String("t_embedder.mlp.2.weight"))
        ref b2 = self._w(String("t_embedder.mlp.2.bias"))
        return linear(ha, w2, Optional[Tensor](self._clone(b2, ctx)), ctx)

    # ── caption embedder: RMSNorm(cap_feat_dim) + Linear ────────────────────
    def _cap_embedder(self, cap_feats: Tensor, ctx: DeviceContext) raises -> Tensor:
        ref nw = self._w(String("cap_embedder.0.weight"))
        var normed = rms_norm(cap_feats, nw, self.config.norm_eps, ctx)
        ref lw = self._w(String("cap_embedder.1.weight"))
        if self._has(String("cap_embedder.1.bias")):
            ref lb = self._w(String("cap_embedder.1.bias"))
            return linear(normed, lw, Optional[Tensor](self._clone(lb, ctx)), ctx)
        return linear(normed, lw, None, ctx)

    # ── attention (single-stream) ──────────────────────────────────────────
    # x: [1, S, dim]. cos/sin: interleaved RoPE tables [S*H, Dh/2] (per token,
    # repeated across heads). Returns [1, S, dim]. S is comptime via the caller.
    def _attention[S: Int](
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        prefix: String,
        ctx: DeviceContext,
    ) raises -> Tensor:
        var h = self.config.n_heads
        var dh = self.config.head_dim
        var eps = self.config.norm_eps
        var scale = Float32(1.0) / sqrt(Float32(dh))

        ref qw = self._w(prefix + ".attention.to_q.weight")
        ref kw = self._w(prefix + ".attention.to_k.weight")
        ref vw = self._w(prefix + ".attention.to_v.weight")
        var q = linear(x, qw, None, ctx)  # [1, S, h*dh]
        var k = linear(x, kw, None, ctx)
        var v = linear(x, vw, None, ctx)

        # reshape to BSHD [1, S, H, Dh]
        var sh = List[Int]()
        sh.append(1)
        sh.append(S)
        sh.append(h)
        sh.append(dh)
        q = reshape(q, sh.copy(), ctx)
        k = reshape(k, sh.copy(), ctx)
        v = reshape(v, sh.copy(), ctx)

        # per-head RMSNorm over Dh (norm_q / norm_k), eps=1e-5
        ref qn = self._w(prefix + ".attention.norm_q.weight")
        ref kn = self._w(prefix + ".attention.norm_k.weight")
        q = rms_norm(q, qn, eps, ctx)
        k = rms_norm(k, kn, eps, ctx)

        # RoPE interleaved on q,k (flattens leading dims to S*H rows)
        q = rope_interleaved(q, cos, sin, ctx)
        k = rope_interleaved(k, cos, sin, ctx)

        # SDPA, no mask (full attention; single batch all-attend). Build a
        # zero additive mask [1, H, S, S].
        var mask = self._zero_mask[S](ctx)
        var attn = sdpa[1, S, 30, 128](q, k, v, mask, scale, ctx)  # [1,S,H,Dh]

        var asz = List[Int]()
        asz.append(1)
        asz.append(S)
        asz.append(h * dh)
        attn = reshape(attn, asz^, ctx)

        ref ow = self._w(prefix + ".attention.to_out.0.weight")
        return linear(attn, ow, None, ctx)

    def _zero_mask[S: Int](self, ctx: DeviceContext) raises -> Tensor:
        var dtype = self._dtype()
        var n = self.config.n_heads * S * S
        var dev = ctx.enqueue_create_buffer[DType.uint8](n * dtype.byte_size())
        ctx.enqueue_memset[DType.uint8](dev, 0)
        ctx.synchronize()
        var sh = List[Int]()
        sh.append(1)
        sh.append(self.config.n_heads)
        sh.append(S)
        sh.append(S)
        return Tensor(dev^, sh^, dtype)

    # ── feed forward: w2(silu(w1(x)) * w3(x)) ──────────────────────────────
    def _feed_forward(self, x: Tensor, prefix: String, ctx: DeviceContext) raises -> Tensor:
        ref w1 = self._w(prefix + ".feed_forward.w1.weight")
        ref w3 = self._w(prefix + ".feed_forward.w3.weight")
        var g = linear(x, w1, None, ctx)
        var u = linear(x, w3, None, ctx)
        var act = swiglu(g, u, ctx)  # silu(g) * u
        ref w2 = self._w(prefix + ".feed_forward.w2.weight")
        return linear(act, w2, None, ctx)

    # ── transformer block ──────────────────────────────────────────────────
    # x: [1, S, dim]. adaln: [1, 256] or None (context_refiner -> None).
    def _block[S: Int](
        self,
        var x: Tensor,
        cos: Tensor,
        sin: Tensor,
        adaln: Optional[Tensor],
        prefix: String,
        ctx: DeviceContext,
    ) raises -> Tensor:
        var eps = self.config.norm_eps
        var dim = self.config.dim
        ref n1 = self._w(prefix + ".attention_norm1.weight")
        ref n2 = self._w(prefix + ".attention_norm2.weight")
        ref fn1 = self._w(prefix + ".ffn_norm1.weight")
        ref fn2 = self._w(prefix + ".ffn_norm2.weight")

        if adaln:
            # mod = adaLN_modulation.0(adaln_input) -> [1, 4*dim] -> chunk 4
            ref mw = self._w(prefix + ".adaLN_modulation.0.weight")
            ref mb = self._w(prefix + ".adaLN_modulation.0.bias")
            var mod = linear(adaln.value(), mw, Optional[Tensor](self._clone(mb, ctx)), ctx)  # [1, 4*dim]
            var scale_msa = slice(mod, 1, 0 * dim, dim, ctx)        # [1, dim]
            var gate_msa = slice(mod, 1, 1 * dim, dim, ctx)
            var scale_mlp = slice(mod, 1, 2 * dim, dim, ctx)
            var gate_mlp = slice(mod, 1, 3 * dim, dim, ctx)
            gate_msa = self._tanh(gate_msa, ctx)
            gate_mlp = self._tanh(gate_mlp, ctx)
            # scale = 1 + scale ; reshape scale/gate to [1,1,dim] for broadcast
            var b1d = List[Int]()
            b1d.append(1)
            b1d.append(1)
            b1d.append(dim)
            scale_msa = self._add_scalar_reshape(scale_msa, 1.0, b1d.copy(), ctx)
            scale_mlp = self._add_scalar_reshape(scale_mlp, 1.0, b1d.copy(), ctx)
            gate_msa = reshape(gate_msa, b1d.copy(), ctx)
            gate_mlp = reshape(gate_mlp, b1d.copy(), ctx)

            # attn_out = attention( norm1(x) * scale_msa )
            var xn1 = rms_norm(x, n1, eps, ctx)            # [1,S,dim]
            var xn1s = mul(xn1, scale_msa, ctx)            # broadcast [1,1,dim]
            var attn = self._attention[S](xn1s, cos, sin, prefix, ctx)
            var attn_n2 = rms_norm(attn, n2, eps, ctx)
            var gated_attn = mul(gate_msa, attn_n2, ctx)   # broadcast
            x = add(x, gated_attn, ctx)

            # FFN
            var xfn1 = rms_norm(x, fn1, eps, ctx)
            var xfn1s = mul(xfn1, scale_mlp, ctx)
            var ff = self._feed_forward(xfn1s, prefix, ctx)
            var ff_n2 = rms_norm(ff, fn2, eps, ctx)
            var gated_ff = mul(gate_mlp, ff_n2, ctx)
            return add(x, gated_ff, ctx)
        else:
            # Unmodulated (context_refiner).
            var xn1 = rms_norm(x, n1, eps, ctx)
            var attn = self._attention[S](xn1, cos, sin, prefix, ctx)
            var attn_n2 = rms_norm(attn, n2, eps, ctx)
            x = add(x, attn_n2, ctx)
            var xfn1 = rms_norm(x, fn1, eps, ctx)
            var ff = self._feed_forward(xfn1, prefix, ctx)
            var ff_n2 = rms_norm(ff, fn2, eps, ctx)
            return add(x, ff_n2, ctx)

    # helper: (x + s) reshaped to new shape (for 1+scale then broadcast layout)
    def _add_scalar_reshape(
        self, x: Tensor, s: Float32, var new_shape: List[Int], ctx: DeviceContext
    ) raises -> Tensor:
        var y = add_scalar(x, s, ctx)
        return reshape(y, new_shape^, ctx)

    # ── final layer ─────────────────────────────────────────────────────────
    # x: [1, S, dim]. scale = 1 + Linear(SiLU(adaln)) [1,dim] -> [1,1,dim].
    # LayerNorm(no affine, eps=1e-6) * scale -> Linear.
    def _final_layer(self, x: Tensor, adaln: Tensor, ctx: DeviceContext) raises -> Tensor:
        var dim = self.config.dim
        # adaLN_modulation = Sequential(SiLU, Linear) -> key .1
        var c_silu = silu(adaln, ctx)
        ref mw = self._w(String("all_final_layer.2-1.adaLN_modulation.1.weight"))
        ref mb = self._w(String("all_final_layer.2-1.adaLN_modulation.1.bias"))
        var scale = linear(c_silu, mw, Optional[Tensor](self._clone(mb, ctx)), ctx)  # [1, dim]
        var b1d = List[Int]()
        b1d.append(1)
        b1d.append(1)
        b1d.append(dim)
        scale = self._add_scalar_reshape(scale, 1.0, b1d^, ctx)  # [1,1,dim]
        # LayerNorm with no affine, eps 1e-6. Foundation layer_norm needs
        # weight+bias; build a ones-weight / zeros-bias of [dim].
        var ones = List[Float32]()
        var zeros = List[Float32]()
        for _ in range(dim):
            ones.append(Float32(1.0))
            zeros.append(Float32(0.0))
        var dtype = x.dtype()
        var wsh = List[Int]()
        wsh.append(dim)
        var w_ones = Tensor.from_host(ones, wsh.copy(), dtype, ctx)
        var b_zero = Tensor.from_host(zeros, wsh^, dtype, ctx)
        var xn = layer_norm(x, w_ones, b_zero, Float32(1e-6), ctx)
        var xs = mul(xn, scale, ctx)
        ref lw = self._w(String("all_final_layer.2-1.linear.weight"))
        ref lb = self._w(String("all_final_layer.2-1.linear.bias"))
        return linear(xs, lw, Optional[Tensor](self._clone(lb, ctx)), ctx)

    # ── RoPE table build (host trig) ────────────────────────────────────────
    # positions: List of [pos0,pos1,pos2] per token (length S). Builds
    # interleaved cos/sin [S*H, Dh/2] (angle repeated across H heads).
    # Per axis a (dim da, half ha): inv_freq_i = theta^(-2i/da), i in [0,ha).
    # freqs_cis[token] = concat_a( pos_a * inv_freq_a[i] ). Total angles = Dh/2.
    def _build_rope(
        self, positions: List[List[Int]], ctx: DeviceContext
    ) raises -> _RopePair:
        var h = self.config.n_heads
        var dh = self.config.head_dim
        var half = dh // 2  # 64
        var s = len(positions)
        var theta = Float32(self.config.rope_theta)
        var log_theta = flog(theta)
        var axes = List[Int]()
        axes.append(self.config.axis0)
        axes.append(self.config.axis1)
        axes.append(self.config.axis2)

        var cos_vals = List[Float32]()
        var sin_vals = List[Float32]()
        # row order: token t, head head -> same angle vector per head.
        for t in range(s):
            # build the half-length angle vector for this token (over 3 axes)
            var angles = List[Float32]()
            for a in range(3):
                var da = axes[a]
                var ha = da // 2
                var pos = Float32(positions[t][a])
                for i in range(ha):
                    var inv_freq = fexp(-log_theta * Float32(2 * i) / Float32(da))
                    angles.append(pos * inv_freq)
            # angles has length half (16+24+24=64)
            for _head in range(h):
                for i in range(half):
                    cos_vals.append(fcos(angles[i]))
                    sin_vals.append(fsin(angles[i]))

        var dtype = self._dtype()
        var rows = s * h
        var fsh = List[Int]()
        fsh.append(rows)
        fsh.append(half)
        var cos_t = Tensor.from_host(cos_vals, fsh.copy(), dtype, ctx)
        var sin_t = Tensor.from_host(sin_vals, fsh^, dtype, ctx)
        return _RopePair(cos_t^, sin_t^)

    # ── Z-Image patchify/unpatchify (channel-MINOR within-patch flatten) ────
    # NOTE: the foundation ops/layout.patchify uses the FLUX convention
    # (c, ph, pw) — channel-MAJOR. Z-Image diffusers uses (pF, pH, pW, C) —
    # channel-MINOR (see _patchify_image / unpatchify). So we build patchify
    # from foundation reshape+permute (general ops, NOT a new kernel) to match
    # the x_embedder's expected 64-dim ordering exactly.
    #   patchify: latent [1, C, H, W] (F=1) ->
    #     view  [C, Ht, pH, Wt, pW]            (pF=1 dropped)
    #     permute (Ht, Wt, pH, pW, C)          # diffusers (1,3,5,2,4,6,0) w/ F=pF=1
    #     reshape [Ht*Wt, pH*pW*C]
    def _patchify_zimage(self, x: Tensor, ctx: DeviceContext) raises -> Tensor:
        var p = self.config.patch_size
        var s = x.shape()  # [1, C, H, W]
        var c = s[1]
        var hh = s[2]
        var ww = s[3]
        var ht = hh // p
        var wt = ww // p
        # squeeze batch -> [C, H, W] -> view [C, Ht, p, Wt, p]
        var v = List[Int]()
        v.append(c); v.append(ht); v.append(p); v.append(wt); v.append(p)
        var xv = reshape(x, v^, ctx)  # row-major reinterpret of [1,C,H,W]
        # permute (C,Ht,pH,Wt,pW) -> (Ht,Wt,pH,pW,C) = axes [1,3,2,4,0]
        var perm = List[Int]()
        perm.append(1); perm.append(3); perm.append(2); perm.append(4); perm.append(0)
        var xp = permute(xv, perm^, ctx)  # [Ht,Wt,pH,pW,C]
        # reshape -> [1, Ht*Wt, p*p*C]
        var r = List[Int]()
        r.append(1); r.append(ht * wt); r.append(p * p * c)
        return reshape(xp, r^, ctx)

    #   unpatchify: seq [1, Ht*Wt, pH*pW*C] -> image [1, C, H, W]
    #     view [Ht, Wt, pH, pW, C]
    #     permute (C, Ht, pH, Wt, pW) = diffusers (6,0,3,1,4,2,5) with F=pF=1
    #       -> from (Ht,Wt,pH,pW,C) axes: (4,0,2,1,3)
    #     reshape [1, C, H, W]
    def _unpatchify_zimage(self, seq: Tensor, ctx: DeviceContext) raises -> Tensor:
        var p = self.config.patch_size
        var c = self.config.in_channels
        var ht = Self.HL // p
        var wt = Self.WL // p
        var v = List[Int]()
        v.append(ht); v.append(wt); v.append(p); v.append(p); v.append(c)
        var sv = reshape(seq, v^, ctx)  # [Ht,Wt,pH,pW,C]
        # permute to (C, Ht, pH, Wt, pW) = axes [4,0,2,1,3]
        var perm = List[Int]()
        perm.append(4); perm.append(0); perm.append(2); perm.append(1); perm.append(3)
        var sp = permute(sv, perm^, ctx)  # [C, Ht, pH, Wt, pW]
        var r = List[Int]()
        r.append(1); r.append(c); r.append(Self.HL); r.append(Self.WL)
        return reshape(sp, r^, ctx)

    # ── full forward ─────────────────────────────────────────────────────────
    # x: latent [1, 16, Self.HL, Self.WL]. timestep: scalar Float32 in [0,1].
    # cap_feats: [Self.CAPLEN, cap_feat_dim] (caption embeddings, pre-cap_embedder).
    # Returns predicted velocity latent [1, 16, Self.HL, Self.WL].
    def forward(
        self, x: Tensor, timestep: Float32, cap_feats: Tensor, ctx: DeviceContext
    ) raises -> Tensor:
        return self._forward_impl(x, timestep, cap_feats, ctx, -1)

    # capture_stage: -1 = full; otherwise dump nothing extra (parity uses the
    # dedicated debug method below). Kept simple: one code path.
    def _forward_impl(
        self, x: Tensor, timestep: Float32, cap_feats: Tensor, ctx: DeviceContext,
        capture_stage: Int,
    ) raises -> Tensor:
        var cfg = self.config
        var dim = cfg.dim
        var p = cfg.patch_size

        # image token grid
        comptime img_tokens = (Self.HL // 2) * (Self.WL // 2)
        comptime img_pad = (-img_tokens) % 32
        comptime img_padded = img_tokens + img_pad
        comptime cap_pad = (-Self.CAPLEN) % 32
        comptime cap_padded = Self.CAPLEN + cap_pad
        comptime unified_len = img_padded + cap_padded

        # adaln_input = t_embedder(t * t_scale)
        var adaln = self._t_embedder(timestep, ctx)  # [1, 256]

        # patchify image -> [1, img_tokens, 64]
        var xp = self._patchify_zimage(x, ctx)  # [1, img_tokens, 64] (channel-minor)
        # embed
        ref xw = self._w(String("all_x_embedder.2-1.weight"))
        ref xb = self._w(String("all_x_embedder.2-1.bias"))
        var xe = linear(xp, xw, Optional[Tensor](self._clone(xb, ctx)), ctx)  # [1,img_tokens,dim]
        # pad image tokens to img_padded (repeat last token), then substitute
        # x_pad_token at padded rows. Reshape to [img_tokens, dim] first.
        var xe2 = self._pad_seq(xe, img_tokens, img_padded, ctx)  # [1, img_padded, dim]
        # substitute pad token: flatten to [img_padded, dim]
        var xe_flat = self._squeeze0(xe2, ctx)  # [img_padded, dim]
        xe_flat = self._apply_pad_token(xe_flat^, String("x_pad_token"), img_tokens, ctx)
        var x_seq = self._unsqueeze0(xe_flat, ctx)  # [1, img_padded, dim]

        # cap embed -> [Self.CAPLEN, dim] -> pad to cap_padded -> cap_pad_token
        var cap_e = self._cap_embedder(cap_feats, ctx)  # [Self.CAPLEN, dim]
        var cap_padded_t = self._pad_rows(cap_e, Self.CAPLEN, cap_padded, ctx)  # [cap_padded, dim]
        cap_padded_t = self._apply_pad_token(cap_padded_t^, String("cap_pad_token"), Self.CAPLEN, ctx)
        var cap_seq = self._unsqueeze0(cap_padded_t, ctx)  # [1, cap_padded, dim]

        # ── RoPE positions ──
        # cap: axis0 = 1..cap_padded ; axis1=axis2=0
        var cap_pos = List[List[Int]]()
        for i in range(cap_padded):
            var pl = List[Int]()
            pl.append(i + 1)
            pl.append(0)
            pl.append(0)
            cap_pos.append(pl^)
        # x: real tokens (cap_padded+1, h, w) for h in [0,Hl/2), w in [0,Wl/2);
        # padded tokens -> (0,0,0)
        var x_pos = List[List[Int]]()
        var ht = Self.HL // p
        var wt = Self.WL // p
        var x0 = cap_padded + 1
        for ih in range(ht):
            for iw in range(wt):
                var pl = List[Int]()
                pl.append(x0)
                pl.append(ih)
                pl.append(iw)
                x_pos.append(pl^)
        for _ in range(img_pad):
            var pl = List[Int]()
            pl.append(0)
            pl.append(0)
            pl.append(0)
            x_pos.append(pl^)

        var x_rope = self._build_rope(x_pos, ctx)
        var cap_rope = self._build_rope(cap_pos, ctx)

        # ── noise refiner (modulated) on x_seq ──
        for i in range(cfg.n_refiner):
            var pre = String("noise_refiner.") + String(i)
            x_seq = self._block[img_padded](x_seq^, x_rope.cos, x_rope.sin, Optional[Tensor](self._clone(adaln, ctx)), pre, ctx)

        # ── context refiner (unmodulated) on cap_seq ──
        for i in range(cfg.n_refiner):
            var pre = String("context_refiner.") + String(i)
            cap_seq = self._block[cap_padded](cap_seq^, cap_rope.cos, cap_rope.sin, None, pre, ctx)

        # ── unified = concat([x, cap], dim=1) ──
        var unified = concat(1, ctx, x_seq, cap_seq)  # [1, unified_len, dim]
        # unified RoPE = concat([x_rope.cos, cap_rope.cos]) along rows. The cos/sin tables
        # are [S*H, half] row-major (token-major, head-minor). concat over the
        # token axis means interleaving by head blocks, so rebuild from positions.
        var uni_pos = List[List[Int]]()
        for i in range(len(x_pos)):
            uni_pos.append(x_pos[i].copy())
        for i in range(len(cap_pos)):
            uni_pos.append(cap_pos[i].copy())
        var uni_rope = self._build_rope(uni_pos, ctx)

        # ── main layers ──
        for i in range(cfg.n_layers):
            var pre = String("layers.") + String(i)
            unified = self._block[unified_len](unified^, uni_rope.cos, uni_rope.sin, Optional[Tensor](self._clone(adaln, ctx)), pre, ctx)

        # ── final layer on the FULL unified sequence (diffusers order) ──
        var uni_final = self._final_layer(unified, adaln, ctx)  # [1, unified_len, 64]

        # ── extract image tokens (first img_tokens of unified [x, cap]) ──
        var x_final = slice(uni_final, 1, 0, img_tokens, ctx)  # [1, img_tokens, 64]

        # ── unpatchify -> [1, 16, Self.HL, Self.WL] ──
        return self._unpatchify_zimage(x_final, ctx)

    # ── small shape glue ─────────────────────────────────────────────────────
    def _clone(self, x: Tensor, ctx: DeviceContext) raises -> Tensor:
        var dev = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
        ctx.enqueue_copy(dst_buf=dev, src_buf=x.buf)
        ctx.synchronize()
        return Tensor(dev^, x.shape(), x.dtype())

    def _squeeze0(self, x: Tensor, ctx: DeviceContext) raises -> Tensor:
        # [1, a, b] -> [a, b]
        var s = x.shape()
        var ns = List[Int]()
        for i in range(1, len(s)):
            ns.append(s[i])
        return reshape(x, ns^, ctx)

    def _unsqueeze0(self, x: Tensor, ctx: DeviceContext) raises -> Tensor:
        # [a, b] -> [1, a, b]
        var s = x.shape()
        var ns = List[Int]()
        ns.append(1)
        for i in range(len(s)):
            ns.append(s[i])
        return reshape(x, ns^, ctx)

    # pad a [1, n, dim] seq to [1, total, dim] by repeating the LAST token row.
    def _pad_seq(self, x: Tensor, n: Int, total: Int, ctx: DeviceContext) raises -> Tensor:
        if total == n:
            return self._clone(x, ctx)
        var flat = self._squeeze0(x, ctx)        # [n, dim]
        var padded = self._pad_rows(flat, n, total, ctx)  # [total, dim]
        return self._unsqueeze0(padded, ctx)

    # pad a [n, dim] matrix to [total, dim] by repeating the LAST row.
    def _pad_rows(self, x: Tensor, n: Int, total: Int, ctx: DeviceContext) raises -> Tensor:
        if total == n:
            return self._clone(x, ctx)
        var s = x.shape()
        var dim = s[len(s) - 1]
        # last row slice [1, dim]
        var last = slice(x, 0, n - 1, 1, ctx)  # [1, dim]
        # build the pad block by concatenating `last` (total-n) times
        var pad_count = total - n
        # concat needs variadic; build iteratively via repeated concat.
        var padblock = self._clone(last, ctx)
        for _ in range(pad_count - 1):
            padblock = concat(0, ctx, padblock, last)
        return concat(0, ctx, x, padblock)

    # ── debug forward: return a named intermediate as a Tensor ──────────────
    # stage codes:
    #   0 t_emb  1 x_after_embedder  2 cap_after_embedder
    #   3 x_after_noise_refiner_1  4 cap_after_context_refiner_1
    #   5 unified_initial  6 unified_after_layer_0  7 unified_after_main
    #   8 after_final_layer  9 out
    # Return one of the noise_refiner.0 modulation chunks (1+scale or tanh(gate)).
    # which: 0=scale_msa 1=gate_msa 2=scale_mlp 3=gate_mlp. Shape [1, dim].
    def debug_nr0_mod(self, timestep: Float32, which: Int, ctx: DeviceContext) raises -> Tensor:
        var dim = self.config.dim
        var adaln = self._t_embedder(timestep, ctx)
        ref mw = self._w(String("noise_refiner.0.adaLN_modulation.0.weight"))
        ref mb = self._w(String("noise_refiner.0.adaLN_modulation.0.bias"))
        var mod = linear(adaln, mw, Optional[Tensor](self._clone(mb, ctx)), ctx)
        var chunk = slice(mod, 1, which * dim, dim, ctx)  # [1, dim]
        if which == 0 or which == 2:
                return add_scalar(chunk, 1.0, ctx)
        return self._tanh(chunk, ctx)

    # Sub-step probe of noise_refiner.0's modulated attention branch.
    # which: 0=norm1  1=norm1_scaled  2=attn_out
    def debug_nr0_attn[S: Int](
        self, x_seq: Tensor, timestep: Float32, x_cos: Tensor, x_sin: Tensor,
        which: Int, ctx: DeviceContext
    ) raises -> Tensor:
        var eps = self.config.norm_eps
        var dim = self.config.dim
        var adaln = self._t_embedder(timestep, ctx)
        ref mw = self._w(String("noise_refiner.0.adaLN_modulation.0.weight"))
        ref mb = self._w(String("noise_refiner.0.adaLN_modulation.0.bias"))
        var mod = linear(adaln, mw, Optional[Tensor](self._clone(mb, ctx)), ctx)
        var scale_msa = slice(mod, 1, 0, dim, ctx)
        var b1d = List[Int]()
        b1d.append(1); b1d.append(1); b1d.append(dim)
        scale_msa = self._add_scalar_reshape(scale_msa, 1.0, b1d^, ctx)
        ref n1 = self._w(String("noise_refiner.0.attention_norm1.weight"))
        var xn1 = rms_norm(x_seq, n1, eps, ctx)
        if which == 0:
            return self._clone(xn1, ctx)
        var xn1s = mul(xn1, scale_msa, ctx)
        if which == 1:
            return self._clone(xn1s, ctx)
        return self._attention[S](xn1s, x_cos, x_sin, String("noise_refiner.0"), ctx)

    def debug_stage(
        self, x: Tensor, timestep: Float32, cap_feats: Tensor, stage: Int, ctx: DeviceContext
    ) raises -> Tensor:
        var cfg = self.config
        var dim = cfg.dim
        var p = cfg.patch_size
        comptime img_tokens = (Self.HL // 2) * (Self.WL // 2)
        comptime img_pad = (-img_tokens) % 32
        comptime img_padded = img_tokens + img_pad
        comptime cap_pad = (-Self.CAPLEN) % 32
        comptime cap_padded = Self.CAPLEN + cap_pad
        comptime unified_len = img_padded + cap_padded

        var adaln = self._t_embedder(timestep, ctx)
        if stage == 0:
            return self._clone(adaln, ctx)

        var xp = self._patchify_zimage(x, ctx)
        ref xw = self._w(String("all_x_embedder.2-1.weight"))
        ref xb = self._w(String("all_x_embedder.2-1.bias"))
        var xe = linear(xp, xw, Optional[Tensor](self._clone(xb, ctx)), ctx)
        if stage == 1:
            return self._squeeze0(xe, ctx)  # [img_tokens, dim] -> matches oracle [img_padded,dim]? no, oracle is padded

        var xe2 = self._pad_seq(xe, img_tokens, img_padded, ctx)
        var xe_flat = self._squeeze0(xe2, ctx)
        xe_flat = self._apply_pad_token(xe_flat^, String("x_pad_token"), img_tokens, ctx)
        var x_seq = self._unsqueeze0(xe_flat, ctx)

        var cap_e = self._cap_embedder(cap_feats, ctx)
        if stage == 2:
            return self._clone(cap_e, ctx)  # [Self.CAPLEN, dim]
        var cap_padded_t = self._pad_rows(cap_e, Self.CAPLEN, cap_padded, ctx)
        cap_padded_t = self._apply_pad_token(cap_padded_t^, String("cap_pad_token"), Self.CAPLEN, ctx)
        var cap_seq = self._unsqueeze0(cap_padded_t, ctx)

        var cap_pos = List[List[Int]]()
        for i in range(cap_padded):
            var pl = List[Int]()
            pl.append(i + 1); pl.append(0); pl.append(0)
            cap_pos.append(pl^)
        var x_pos = List[List[Int]]()
        var ht = Self.HL // p
        var wt = Self.WL // p
        var x0 = cap_padded + 1
        for ih in range(ht):
            for iw in range(wt):
                var pl = List[Int]()
                pl.append(x0); pl.append(ih); pl.append(iw)
                x_pos.append(pl^)
        for _ in range(img_pad):
            var pl = List[Int]()
            pl.append(0); pl.append(0); pl.append(0)
            x_pos.append(pl^)

        var x_rope = self._build_rope(x_pos, ctx)
        var cap_rope = self._build_rope(cap_pos, ctx)

        if stage == 17:
            return self._clone(x_seq, ctx)  # x after embed+pad+padtoken [1,img_padded,dim]
        if stage == 14:
            return self.debug_nr0_attn[img_padded](x_seq, timestep, x_rope.cos, x_rope.sin, 0, ctx)
        if stage == 15:
            return self.debug_nr0_attn[img_padded](x_seq, timestep, x_rope.cos, x_rope.sin, 1, ctx)
        if stage == 16:
            return self.debug_nr0_attn[img_padded](x_seq, timestep, x_rope.cos, x_rope.sin, 2, ctx)

        for i in range(cfg.n_refiner):
            var pre = String("noise_refiner.") + String(i)
            x_seq = self._block[img_padded](x_seq^, x_rope.cos, x_rope.sin, Optional[Tensor](self._clone(adaln, ctx)), pre, ctx)
            if stage == 11 and i == 0:
                return self._clone(x_seq, ctx)
            if stage == 12 and i == 0:
                # only the real image tokens (first img_tokens rows)
                return slice(x_seq, 1, 0, img_tokens, ctx)
        if stage == 3:
            return self._clone(x_seq, ctx)

        for i in range(cfg.n_refiner):
            var pre = String("context_refiner.") + String(i)
            cap_seq = self._block[cap_padded](cap_seq^, cap_rope.cos, cap_rope.sin, None, pre, ctx)
        if stage == 4:
            return self._clone(cap_seq, ctx)

        var unified = concat(1, ctx, x_seq, cap_seq)
        if stage == 5:
            return self._clone(unified, ctx)
        var uni_pos = List[List[Int]]()
        for i in range(len(x_pos)):
            uni_pos.append(x_pos[i].copy())
        for i in range(len(cap_pos)):
            uni_pos.append(cap_pos[i].copy())
        var uni_rope = self._build_rope(uni_pos, ctx)

        for i in range(cfg.n_layers):
            var pre = String("layers.") + String(i)
            unified = self._block[unified_len](unified^, uni_rope.cos, uni_rope.sin, Optional[Tensor](self._clone(adaln, ctx)), pre, ctx)
            if stage == 6 and i == 0:
                return self._clone(unified, ctx)
        if stage == 7:
            return self._clone(unified, ctx)

        if stage == 13:
            # uni RoPE cos table, head 0 only -> [unified_len, half]
            # table is [unified_len*H, half] token-major head-minor; head 0 rows
            # are at t*H. Build a strided gather via slice per token is costly;
            # instead reshape to [unified_len, H, half] and slice head 0.
            var uh = List[Int]()
            uh.append(unified_len)
            uh.append(cfg.n_heads)
            uh.append(cfg.head_dim // 2)
            var cos_r = reshape(uni_rope.cos, uh^, ctx)  # [S, H, half]
            return slice(cos_r, 1, 0, 1, ctx)  # [S, 1, half]

        var uni_final = self._final_layer(unified, adaln, ctx)  # [1, unified_len, 64]
        if stage == 8:
            return self._clone(uni_final, ctx)
        var x_final = slice(uni_final, 1, 0, img_tokens, ctx)
        return self._unpatchify_zimage(x_final, ctx)

    # ── per-main-layer capture: run full pipeline up to and INCLUDING main
    # layer `target` (0..n_layers-1), return the unified [1, unified_len, dim]
    # sequence at that point. Mirrors the oracle's unified_after_layer_<target>.
    def debug_main_layer(
        self, x: Tensor, timestep: Float32, cap_feats: Tensor, target: Int, ctx: DeviceContext
    ) raises -> Tensor:
        var cfg = self.config
        var dim = cfg.dim
        var p = cfg.patch_size
        comptime img_tokens = (Self.HL // 2) * (Self.WL // 2)
        comptime img_pad = (-img_tokens) % 32
        comptime img_padded = img_tokens + img_pad
        comptime cap_pad = (-Self.CAPLEN) % 32
        comptime cap_padded = Self.CAPLEN + cap_pad
        comptime unified_len = img_padded + cap_padded

        var adaln = self._t_embedder(timestep, ctx)

        var xp = self._patchify_zimage(x, ctx)
        ref xw = self._w(String("all_x_embedder.2-1.weight"))
        ref xb = self._w(String("all_x_embedder.2-1.bias"))
        var xe = linear(xp, xw, Optional[Tensor](self._clone(xb, ctx)), ctx)
        var xe2 = self._pad_seq(xe, img_tokens, img_padded, ctx)
        var xe_flat = self._squeeze0(xe2, ctx)
        xe_flat = self._apply_pad_token(xe_flat^, String("x_pad_token"), img_tokens, ctx)
        var x_seq = self._unsqueeze0(xe_flat, ctx)

        var cap_e = self._cap_embedder(cap_feats, ctx)
        var cap_padded_t = self._pad_rows(cap_e, Self.CAPLEN, cap_padded, ctx)
        cap_padded_t = self._apply_pad_token(cap_padded_t^, String("cap_pad_token"), Self.CAPLEN, ctx)
        var cap_seq = self._unsqueeze0(cap_padded_t, ctx)

        var cap_pos = List[List[Int]]()
        for i in range(cap_padded):
            var pl = List[Int]()
            pl.append(i + 1); pl.append(0); pl.append(0)
            cap_pos.append(pl^)
        var x_pos = List[List[Int]]()
        var ht = Self.HL // p
        var wt = Self.WL // p
        var x0 = cap_padded + 1
        for ih in range(ht):
            for iw in range(wt):
                var pl = List[Int]()
                pl.append(x0); pl.append(ih); pl.append(iw)
                x_pos.append(pl^)
        for _ in range(img_pad):
            var pl = List[Int]()
            pl.append(0); pl.append(0); pl.append(0)
            x_pos.append(pl^)

        var x_rope = self._build_rope(x_pos, ctx)
        var cap_rope = self._build_rope(cap_pos, ctx)

        for i in range(cfg.n_refiner):
            var pre = String("noise_refiner.") + String(i)
            x_seq = self._block[img_padded](x_seq^, x_rope.cos, x_rope.sin, Optional[Tensor](self._clone(adaln, ctx)), pre, ctx)
        for i in range(cfg.n_refiner):
            var pre = String("context_refiner.") + String(i)
            cap_seq = self._block[cap_padded](cap_seq^, cap_rope.cos, cap_rope.sin, None, pre, ctx)

        var unified = concat(1, ctx, x_seq, cap_seq)
        var uni_pos = List[List[Int]]()
        for i in range(len(x_pos)):
            uni_pos.append(x_pos[i].copy())
        for i in range(len(cap_pos)):
            uni_pos.append(cap_pos[i].copy())
        var uni_rope = self._build_rope(uni_pos, ctx)

        for i in range(cfg.n_layers):
            var pre = String("layers.") + String(i)
            unified = self._block[unified_len](unified^, uni_rope.cos, uni_rope.sin, Optional[Tensor](self._clone(adaln, ctx)), pre, ctx)
            if i == target:
                return self._clone(unified, ctx)
        return self._clone(unified, ctx)

    # ── isolation: apply ONLY the final Linear to an external [1,S,dim] input.
    # Used to test whether my linear op itself is the gap vs upstream error.
    def debug_final_linear_only(self, scaled: Tensor, ctx: DeviceContext) raises -> Tensor:
        ref lw = self._w(String("all_final_layer.2-1.linear.weight"))
        ref lb = self._w(String("all_final_layer.2-1.linear.bias"))
        return linear(scaled, lw, Optional[Tensor](self._clone(lb, ctx)), ctx)

    # ── final-layer sub-step capture. Runs full pipeline to unified_after_main,
    # then returns one of:  which: 0=fl_scale [1,1,dim]  1=fl_norm [1,S,dim]
    #   2=fl_scaled [1,S,dim]. Mirrors the oracle fl_scale/fl_norm/fl_scaled.
    def debug_final_sub(
        self, x: Tensor, timestep: Float32, cap_feats: Tensor, which: Int, ctx: DeviceContext
    ) raises -> Tensor:
        var cfg = self.config
        var dim = cfg.dim
        # full pipeline up to and including the last main layer
        var unified = self.debug_main_layer(x, timestep, cap_feats, cfg.n_layers - 1, ctx)
        var adaln = self._t_embedder(timestep, ctx)
        # scale = 1 + Linear(SiLU(adaln))
        var c_silu = silu(adaln, ctx)
        ref mw = self._w(String("all_final_layer.2-1.adaLN_modulation.1.weight"))
        ref mb = self._w(String("all_final_layer.2-1.adaLN_modulation.1.bias"))
        var scale = linear(c_silu, mw, Optional[Tensor](self._clone(mb, ctx)), ctx)  # [1, dim]
        var b1d = List[Int]()
        b1d.append(1); b1d.append(1); b1d.append(dim)
        scale = self._add_scalar_reshape(scale, 1.0, b1d^, ctx)  # [1,1,dim]
        if which == 0:
            return self._clone(scale, ctx)
        var ones = List[Float32]()
        var zeros = List[Float32]()
        for _ in range(dim):
            ones.append(Float32(1.0))
            zeros.append(Float32(0.0))
        var dtype = unified.dtype()
        var wsh = List[Int]()
        wsh.append(dim)
        var w_ones = Tensor.from_host(ones, wsh.copy(), dtype, ctx)
        var b_zero = Tensor.from_host(zeros, wsh^, dtype, ctx)
        var xn = layer_norm(unified, w_ones, b_zero, Float32(1e-6), ctx)
        if which == 1:
            return self._clone(xn, ctx)
        return mul(xn, scale, ctx)
