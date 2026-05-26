# nucleus_dit.mojo — Nucleus-Image 17B sparse MoE DiT (pure Mojo, inference).
#
# Reference (read line-by-line):
#   /home/alex/EriDiffusion/inference-flame/src/models/nucleus_dit.rs
#   /home/alex/EriDiffusion/inference-flame/src/bin/nucleus_infer.rs
#   diffusers transformer_nucleusmoe_image.py (the oracle behind the Rust).
#
# Architecture (NucleusConfig::nucleus_image_default):
#   32 layers: layers 0..2 DENSE (plain SwiGLU FFN), layers 3..31 MoE
#   (64 experts, expert-choice routing; capacity_factor 4.0 for layers 3-4,
#   2.0 for 5-31; shared SwiGLU expert added to routed output).
#   D=2048, heads=16, kv_heads=4 (GQA n_rep=4), head_dim=128,
#   joint_attention_dim=4096 (Qwen3-VL text), mlp_ratio=4.0,
#   axes_dims_rope=[16,56,56] (sum=128), rope_theta=10000, route_scale=2.5,
#   patch_size=2, in_channels=64, out_channels=16.
#
#   Block (NucleusMoEImageTransformerBlock.forward):
#     temb_silu = silu(temb); mod = Linear(temb_silu).chunk(4)
#       -> scale1, gate1(clamp ±2), scale2, gate2(clamp ±2)
#     context = encoder_proj(enc)              # per-block, [B,S_txt,joint]->[B,S_txt,D]
#     img1 = LN(x)*(1+scale1)                  # LayerNorm no-affine, eps 1e-6
#     attn = JointAttention(img1, context)     # img Q ; img+txt K/V
#     x = x + tanh(gate1)*attn
#     img2 = LN(x)*(1+scale2)
#     mlp = Dense(img2) | MoE(img2, LN(x), temb)
#     x = x + tanh(gate2)*mlp
#
#   temb = RMSNorm(linear2(silu(linear1(timestep_embedding(t)))))
#   txt_norm: top-level RMSNorm on encoder_hidden_states before per-block proj.
#   norm_out: AdaLNContinuous — emb=Linear(silu(temb)).chunk(2)=[scale,shift];
#             out = LN(x)*(1+scale)+shift ; proj_out -> [B,S,patch²*out_ch=64].
#
# ops/moe DIVERGENCE: shared ops/moe is token-choice; Nucleus is expert-choice.
# We use models/dit/nucleus_moe.mojo (NEW) for routing+grouped FFN and reuse
# ops/moe.gated_scatter_add inside it. We do NOT modify ops/moe.
#
# Memory: 17B won't fit resident on 24GB; the bulk is the 29×64 MoE expert
# weights. A streaming runtime (mirroring Klein9BOffloaded + NucleusInferDit)
# is sketched at the end (NOT compiled in this pass — see STUB note).

from std.gpu.host import DeviceContext
from std.math import (
    sqrt,
    cos as fcos,
    sin as fsin,
    exp as fexp,
    log as flog,
    tanh as ftanh,
)
from std.memory import ArcPointer

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.linear import linear
from serenitymojo.ops.norm import rms_norm, layer_norm
from serenitymojo.ops.activations import silu, swiglu
from serenitymojo.ops.rope import rope_interleaved
from serenitymojo.ops.attention import sdpa
from serenitymojo.ops.softmax import softmax_lastdim
from serenitymojo.ops.tensor_algebra import (
    reshape,
    permute,
    transpose,
    concat,
    add,
    mul,
    add_scalar,
    mul_scalar,
)
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.models.dit.nucleus_moe import nucleus_moe_expert_forward


# Nucleus head geometry is fixed; sdpa needs these comptime.
comptime _N_HEADS = 16
comptime _HEAD_DIM = 128


# ── Config ────────────────────────────────────────────────────────────────────
@fieldwise_init
struct NucleusConfig(Copyable, Movable, ImplicitlyCopyable):
    var num_layers: Int
    var num_heads: Int
    var num_kv_heads: Int
    var head_dim: Int
    var hidden_size: Int
    var joint_attention_dim: Int
    var num_experts: Int
    var moe_intermediate_dim: Int
    var route_scale: Float32
    var patch_size: Int
    var in_channels: Int
    var out_channels: Int
    var rope_theta: Float64
    var eps: Float32
    # axes_dims_rope packed as three ints (sum must == head_dim).
    var axis0: Int
    var axis1: Int
    var axis2: Int

    @staticmethod
    def nucleus_image() -> NucleusConfig:
        return NucleusConfig(
            32, 16, 4, 128, 2048, 4096, 64, 1344, Float32(2.5),
            2, 64, 16, Float64(10000.0), Float32(1e-6), 16, 56, 56,
        )

    def dense_inner_dim(self) -> Int:
        # int(D * mlp_ratio(4.0) * 2/3) // 128 * 128. For D=2048: 5376.
        var raw = Int(Float64(self.hidden_size) * 4.0 * 2.0 / 3.0)
        return (raw // 128) * 128

    def capacity_factor_for(self, layer_idx: Int) -> Float32:
        if layer_idx < 3:
            return Float32(0.0)
        if layer_idx == 3 or layer_idx == 4:
            return Float32(4.0)
        return Float32(2.0)


# ── 3D RoPE table build (host-side; mirrors build_nucleus_3d_rope) ────────────
@fieldwise_init
struct NucleusRope(Movable):
    """Interleaved-pair cos/sin tables, each [rows, head_dim/2]."""

    var img_cos: Tensor
    var img_sin: Tensor
    var txt_cos: Tensor
    var txt_sin: Tensor


def build_nucleus_3d_rope(
    f: Int,
    h: Int,
    w: Int,
    s_txt: Int,
    cfg: NucleusConfig,
    dtype: STDtype,
    ctx: DeviceContext,
) raises -> NucleusRope:
    """Build img + txt RoPE cos/sin tables. scale_rope=True (signed h/w pos).

    Returns cos/sin shaped [S, head_dim/2] for rope_interleaved (rows-flattened).
    Mirrors the Rust build_nucleus_3d_rope axis-by-axis fill exactly.
    """
    var head_dim = cfg.head_dim
    if head_dim % 2 != 0:
        raise Error("build_nucleus_3d_rope: head_dim must be even")
    var half = head_dim // 2
    var theta = cfg.rope_theta

    var axes = List[Int]()
    axes.append(cfg.axis0)
    axes.append(cfg.axis1)
    axes.append(cfg.axis2)

    # Per-axis position lists (scale_rope=True).
    var pos_t = List[Float32]()
    for i in range(f):
        pos_t.append(Float32(i))
    var pos_h = _signed_positions(h)
    var pos_w = _signed_positions(w)

    var max_vid_index = (h // 2)
    if (w // 2) > max_vid_index:
        max_vid_index = (w // 2)
    var pos_txt = List[Float32]()
    for i in range(s_txt):
        pos_txt.append(Float32(max_vid_index + i))

    var img_seq = f * h * w
    var img_cos = List[Float32]()
    var img_sin = List[Float32]()
    img_cos.resize(img_seq * half, Float32(0.0))
    img_sin.resize(img_seq * half, Float32(0.0))
    var txt_cos = List[Float32]()
    var txt_sin = List[Float32]()
    txt_cos.resize(s_txt * half, Float32(0.0))
    txt_sin.resize(s_txt * half, Float32(0.0))

    var axis_offset = 0
    for axis_idx in range(3):
        var axis_dim = axes[axis_idx]
        var half_axis = axis_dim // 2
        var inv_freqs = List[Float64]()
        for i in range(half_axis):
            inv_freqs.append(
                1.0 / (theta ** (Float64(i) / Float64(half_axis)))
            )

        for fi in range(f):
            for hi in range(h):
                for wi in range(w):
                    var pos: Float32
                    if axis_idx == 0:
                        pos = pos_t[fi]
                    elif axis_idx == 1:
                        pos = pos_h[hi]
                    else:
                        pos = pos_w[wi]
                    var token_idx = (fi * h + hi) * w + wi
                    for k in range(half_axis):
                        var ang = Float64(pos) * inv_freqs[k]
                        var base = token_idx * half + axis_offset + k
                        img_cos[base] = Float32(fcos(ang))
                        img_sin[base] = Float32(fsin(ang))

        for i in range(s_txt):
            var pos = pos_txt[i]
            for k in range(half_axis):
                var ang = Float64(pos) * inv_freqs[k]
                var base = i * half + axis_offset + k
                txt_cos[base] = Float32(fcos(ang))
                txt_sin[base] = Float32(fsin(ang))

        axis_offset += half_axis

    var img_sh = List[Int]()
    img_sh.append(img_seq)
    img_sh.append(half)
    var txt_sh = List[Int]()
    txt_sh.append(s_txt)
    txt_sh.append(half)

    var ic = Tensor.from_host(img_cos, img_sh.copy(), dtype, ctx)
    var iss = Tensor.from_host(img_sin, img_sh^, dtype, ctx)
    var tc = Tensor.from_host(txt_cos, txt_sh.copy(), dtype, ctx)
    var ts = Tensor.from_host(txt_sin, txt_sh^, dtype, ctx)
    return NucleusRope(ic^, iss^, tc^, ts^)


def _signed_positions(n: Int) -> List[Float32]:
    """scale_rope position list: neg block then [0, n//2). neg_count = n - n//2."""
    var neg_count = n - (n // 2)
    var v = List[Float32]()
    for i in range(neg_count):
        v.append(Float32(-neg_count + i))
    for i in range(n // 2):
        v.append(Float32(i))
    return v^


# ── DiT (compile-time img/txt sequence lengths for the comptime sdpa) ─────────
struct NucleusDiT[S_IMG: Int, S_TXT: Int]:
    """Nucleus-Image MoE DiT. S_IMG = patchified image-token count
    (F*H_grid*W_grid), S_TXT = text-token count. Both compile-time so the
    joint-attention sequence length (S_IMG + S_TXT) is a constant for sdpa.

    All-resident loader (load all tensors of the transformer dir). For the full
    17B model use the streaming variant (STUB at end of file)."""

    var weights: List[ArcPointer[Tensor]]
    var name_to_idx: Dict[String, Int]
    var config: NucleusConfig

    def __init__(
        out self,
        var weights: List[ArcPointer[Tensor]],
        var name_to_idx: Dict[String, Int],
        cfg: NucleusConfig,
    ):
        self.weights = weights^
        self.name_to_idx = name_to_idx^
        self.config = cfg

    @staticmethod
    def load(
        dir: String, ctx: DeviceContext
    ) raises -> NucleusDiT[Self.S_IMG, Self.S_TXT]:
        """Load every transformer tensor by its diffusers name. The on-disk
        keys are `transformer_blocks.{i}.<...>` plus top-level (`img_in`,
        `time_text_embed.*`, `txt_norm`, `norm_out.*`, `proj_out`)."""
        var sharded = ShardedSafeTensors.open(dir)
        var weights = List[ArcPointer[Tensor]]()
        var name_to_idx = Dict[String, Int]()
        for ref nm in sharded.names():
            var tv = sharded.tensor_view(nm)
            var t = Tensor.from_view(tv, ctx)
            var idx = len(weights)
            weights.append(ArcPointer(t^))
            name_to_idx[nm] = idx
        return NucleusDiT[Self.S_IMG, Self.S_TXT](
            weights^, name_to_idx^, NucleusConfig.nucleus_image()
        )

    def _w(self, name: String) raises -> ref [self.weights] Tensor:
        if name not in self.name_to_idx:
            raise Error(String("NucleusDiT: missing weight: ") + name)
        var idx = self.name_to_idx[name]
        return self.weights[idx][]

    def _has(self, name: String) -> Bool:
        return name in self.name_to_idx

    def _dtype(self) raises -> STDtype:
        # Use img_in.weight dtype as the model compute dtype (BF16 expected).
        return self._w("img_in.weight").dtype()

    # ── per-head RMSNorm over head_dim (norm_q / norm_k / norm_added_k) ────────
    # x is [rows, head_dim]; weight is [head_dim].
    def _qk_norm(
        self, x: Tensor, weight: Tensor, ctx: DeviceContext
    ) raises -> Tensor:
        return rms_norm(x, weight, self.config.eps, ctx)

    # ── timestep embedding -> temb = RMSNorm(l2(silu(l1(emb)))) ────────────────
    def _time_text_embed(
        self, timestep: Float32, ctx: DeviceContext
    ) raises -> Tensor:
        var dim = self.config.hidden_size
        var half = dim // 2
        var dtype = self._dtype()
        # Sinusoidal embedding (cos then sin; matches diffusers flip_sin_to_cos).
        # diffusers scales timestep by 1000 then uses max_period=10000.
        var t = Float64(timestep) * 1000.0
        var emb = List[Float32]()
        emb.resize(dim, Float32(0.0))
        var neg_ln_mp = -flog(10000.0)
        for i in range(half):
            var freq = fexp(neg_ln_mp * Float64(i) / Float64(half))
            var ang = t * freq
            emb[i] = Float32(fcos(ang))
            emb[half + i] = Float32(fsin(ang))
        var esh = List[Int]()
        esh.append(1)
        esh.append(dim)
        var emb_t = Tensor.from_host(emb, esh^, dtype, ctx)  # [1, D]

        var l1 = linear(
            emb_t,
            self._w("time_text_embed.timestep_embedder.linear_1.weight"),
            Optional[Tensor](self._bias_opt(
                "time_text_embed.timestep_embedder.linear_1.bias", ctx)),
            ctx,
        )
        var l1a = silu(l1, ctx)
        var l2 = linear(
            l1a,
            self._w("time_text_embed.timestep_embedder.linear_2.weight"),
            Optional[Tensor](self._bias_opt(
                "time_text_embed.timestep_embedder.linear_2.bias", ctx)),
            ctx,
        )
        return rms_norm(l2, self._w("time_text_embed.norm.weight"),
                        self.config.eps, ctx)

    def _bias_opt(self, name: String, ctx: DeviceContext) raises -> Tensor:
        # Copy a bias weight into a fresh owned tensor (so Optional can take it).
        ref b = self._w(name)
        var dev = ctx.enqueue_create_buffer[DType.uint8](b.nbytes())
        ctx.enqueue_copy(dst_buf=dev, src_buf=b.buf)
        ctx.synchronize()
        return Tensor(dev^, b.shape(), b.dtype())

    # ── joint attention: img Q ; img+txt K/V ──────────────────────────────────
    def _attention(
        self,
        prefix: String,
        img1: Tensor,       # [1, S_IMG, D] modulated image hidden
        context: Tensor,    # [1, S_TXT, D] encoder_proj output
        rope: NucleusRope,
        mask: Tensor,       # [1, H, S, S] zero additive mask (S = S_IMG+S_TXT)
        ctx: DeviceContext,
    ) raises -> Tensor:
        var d = self.config.hidden_size
        var h = self.config.num_heads
        var kvh = self.config.num_kv_heads
        var hd = self.config.head_dim
        var n_rep = h // kvh
        comptime S = Self.S_IMG + Self.S_TXT

        # img Q/K/V.
        var img_q = linear(img1, self._w(prefix + "attn.to_q.weight"), None, ctx)
        var img_k = linear(img1, self._w(prefix + "attn.to_k.weight"), None, ctx)
        var img_v = linear(img1, self._w(prefix + "attn.to_v.weight"), None, ctx)
        # Reshape to [S_IMG, H, hd] / [S_IMG, kvh, hd] (B=1).
        img_q = reshape(img_q, _sh3(Self.S_IMG * h, hd, 1), ctx)  # tmp flatten
        img_q = reshape(img_q, _sh2(Self.S_IMG * h, hd), ctx)
        var img_q_n = self._qk_norm(img_q, self._w(prefix + "attn.norm_q.weight"), ctx)
        img_k = reshape(img_k, _sh2(Self.S_IMG * kvh, hd), ctx)
        var img_k_n = self._qk_norm(img_k, self._w(prefix + "attn.norm_k.weight"), ctx)

        # RoPE on q,k (interleaved). cos/sin are [S_IMG, hd/2]; q has H heads, k
        # has kvh heads, both share per-token cos/sin. rope_interleaved flattens
        # leading dims to rows and expects cos rows == x rows. We tile cos/sin
        # per head by reshaping q to [S_IMG*H, hd] and replicating the table.
        var img_q_r = self._rope_per_head(img_q_n, rope.img_cos, rope.img_sin,
                                          Self.S_IMG, h, hd, ctx)
        var img_k_r = self._rope_per_head(img_k_n, rope.img_cos, rope.img_sin,
                                          Self.S_IMG, kvh, hd, ctx)

        # txt K/V from context.
        var txt_k = linear(context, self._w(prefix + "attn.add_k_proj.weight"), None, ctx)
        var txt_v = linear(context, self._w(prefix + "attn.add_v_proj.weight"), None, ctx)
        txt_k = reshape(txt_k, _sh2(Self.S_TXT * kvh, hd), ctx)
        var txt_k_n = self._qk_norm(txt_k, self._w(prefix + "attn.norm_added_k.weight"), ctx)
        var txt_k_r = self._rope_per_head(txt_k_n, rope.txt_cos, rope.txt_sin,
                                          Self.S_TXT, kvh, hd, ctx)

        # Assemble [1, S, H, hd] tensors for sdpa. q already H heads.
        # K/V: concat img+txt along seq then GQA-expand kvh->H.
        var q_bshd = reshape(img_q_r, _sh4(1, Self.S_IMG, h, hd), ctx)
        # For the joint K/V we need [1, S, H, hd]; build from kvh then repeat.
        var k_img = reshape(img_k_r, _sh4(1, Self.S_IMG, kvh, hd), ctx)
        var v_img = reshape(img_v, _sh4(1, Self.S_IMG, kvh, hd), ctx)
        var k_txt = reshape(txt_k_r, _sh4(1, Self.S_TXT, kvh, hd), ctx)
        var v_txt = reshape(txt_v, _sh4(1, Self.S_TXT, kvh, hd), ctx)
        var k_joint = concat(1, ctx, k_img, k_txt)  # [1, S, kvh, hd]
        var v_joint = concat(1, ctx, v_img, v_txt)  # [1, S, kvh, hd]
        var k_full = self._repeat_kv(k_joint, S, kvh, hd, n_rep, ctx)  # [1,S,H,hd]
        var v_full = self._repeat_kv(v_joint, S, kvh, hd, n_rep, ctx)

        # Pad q to length S with zeros (image only attends; full mask is zeros).
        # diffusers concatenates K/V over joint seq; Q stays S_IMG long. We run
        # sdpa with S_q = S by zero-padding q's seq then slice the image rows.
        # Simpler: run sdpa with q seq = S by appending zero txt-query rows, then
        # take the first S_IMG outputs. Build padded q.
        var q_full = self._pad_q_to_joint(q_bshd, S, h, hd, ctx)  # [1, S, H, hd]

        var scale = Float32(1.0) / Float32(sqrt(Float64(hd)))
        var attn = sdpa[1, S, _N_HEADS, _HEAD_DIM](
            q_full, k_full, v_full, mask, scale, ctx
        )  # [1, S, H, hd]
        # Take image rows [0, S_IMG), reshape [1, S_IMG, D].
        var attn_img = self._take_seq_prefix(attn, S, Self.S_IMG, h * hd, ctx)
        var attn_2d = reshape(attn_img, _sh2(Self.S_IMG, d), ctx)
        var out = linear(attn_2d, self._w(prefix + "attn.to_out.0.weight"), None, ctx)
        return reshape(out, _sh3(1, Self.S_IMG, d), ctx)

    # rope per-head: x is [seq*heads, hd]; cos/sin are [seq, hd/2]. Tile cos/sin
    # to [seq*heads, hd/2] so each head reuses the per-token table, then apply.
    def _rope_per_head(
        self,
        x: Tensor,          # [seq*heads, hd]
        cos: Tensor,        # [seq, hd/2]
        sin: Tensor,        # [seq, hd/2]
        seq: Int,
        heads: Int,
        hd: Int,
        ctx: DeviceContext,
    ) raises -> Tensor:
        # x rows are ordered (token-major then head?) — reshape from [seq,heads,hd]
        # depends on caller layout. linear output is [seq, heads*hd] row-major, so
        # reshaping to [seq*heads, hd] gives rows ordered (token, head). We need
        # cos row = token of that row. Tile cos/sin by repeating each token row
        # `heads` times.
        var half = hd // 2
        var cos_host = cos.to_host(ctx)  # seq*half
        var sin_host = sin.to_host(ctx)
        var ct = List[Float32]()
        var st = List[Float32]()
        ct.resize(seq * heads * half, Float32(0.0))
        st.resize(seq * heads * half, Float32(0.0))
        for tkn in range(seq):
            for hh in range(heads):
                var dst = (tkn * heads + hh) * half
                var src = tkn * half
                for k in range(half):
                    ct[dst + k] = cos_host[src + k]
                    st[dst + k] = sin_host[src + k]
        var dtype = x.dtype()
        var tsh = List[Int]()
        tsh.append(seq * heads)
        tsh.append(half)
        var cos_t = Tensor.from_host(ct, tsh.copy(), dtype, ctx)
        var sin_t = Tensor.from_host(st, tsh^, dtype, ctx)
        return rope_interleaved(x, cos_t, sin_t, ctx)

    # GQA expand [1, S, kvh, hd] -> [1, S, kvh*n_rep, hd] (repeat each kv head).
    def _repeat_kv(
        self,
        x: Tensor,          # [1, S, kvh, hd]
        s: Int,
        kvh: Int,
        hd: Int,
        n_rep: Int,
        ctx: DeviceContext,
    ) raises -> Tensor:
        if n_rep == 1:
            var dev = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
            ctx.enqueue_copy(dst_buf=dev, src_buf=x.buf)
            ctx.synchronize()
            return Tensor(dev^, x.shape(), x.dtype())
        # Host tile: out[s, kh*n_rep + r, :] = x[s, kh, :].
        var host = x.to_host(ctx)  # S*kvh*hd
        var out = List[Float32]()
        var h_full = kvh * n_rep
        out.resize(s * h_full * hd, Float32(0.0))
        for si in range(s):
            for kh in range(kvh):
                var src = (si * kvh + kh) * hd
                for r in range(n_rep):
                    var dst = (si * h_full + kh * n_rep + r) * hd
                    for di in range(hd):
                        out[dst + di] = host[src + di]
        var osh = List[Int]()
        osh.append(1)
        osh.append(s)
        osh.append(h_full)
        osh.append(hd)
        return Tensor.from_host(out, osh^, x.dtype(), ctx)

    # Zero-pad q seq from S_IMG to S (append (S-S_IMG) zero token rows over heads).
    def _pad_q_to_joint(
        self, q: Tensor, s: Int, h: Int, hd: Int, ctx: DeviceContext
    ) raises -> Tensor:
        var host = q.to_host(ctx)  # 1*S_IMG*h*hd
        var out = List[Float32]()
        out.resize(s * h * hd, Float32(0.0))
        var n_img = Self.S_IMG * h * hd
        for i in range(n_img):
            out[i] = host[i]
        var osh = List[Int]()
        osh.append(1)
        osh.append(s)
        osh.append(h)
        osh.append(hd)
        return Tensor.from_host(out, osh^, q.dtype(), ctx)

    # Take the first `prefix_len` seq rows of [1, S, H, hd] -> [1, prefix, H, hd].
    def _take_seq_prefix(
        self, x: Tensor, s: Int, prefix_len: Int, hhd: Int, ctx: DeviceContext
    ) raises -> Tensor:
        var host = x.to_host(ctx)  # S * hhd
        var out = List[Float32]()
        out.resize(prefix_len * hhd, Float32(0.0))
        for i in range(prefix_len * hhd):
            out[i] = host[i]
        var osh = List[Int]()
        osh.append(1)
        osh.append(prefix_len)
        # keep head split implicit; caller reshapes to [1, prefix, D].
        osh.append(hhd)
        return Tensor.from_host(out, osh^, x.dtype(), ctx)

    # ── dense SwiGLU FFN ([up, gate] chunk order) ─────────────────────────────
    def _dense_ffn(
        self,
        x: Tensor,          # [1, S_IMG, D]
        gate_up_name: String,
        down_name: String,
        ctx: DeviceContext,
    ) raises -> Tensor:
        var gu = linear(x, self._w(gate_up_name), None, ctx)  # [1,S,2*inner]
        var inner = self._w(gate_up_name).shape()[0] // 2
        # chunk(2): up = [:inner], gate = [inner:]; swiglu(gate, up)=silu(gate)*up.
        var up = self._chunk_last(gu, 0, inner, ctx)
        var gate = self._chunk_last(gu, inner, inner, ctx)
        var act = swiglu(gate, up, ctx)
        return linear(act, self._w(down_name), None, ctx)

    # Slice last dim: [..., start : start+count].
    def _chunk_last(
        self, x: Tensor, start: Int, count: Int, ctx: DeviceContext
    ) raises -> Tensor:
        var sh = x.shape()
        var last = sh[len(sh) - 1]
        var rows = 1
        for i in range(len(sh) - 1):
            rows *= sh[i]
        var host = x.to_host(ctx)
        var out = List[Float32]()
        out.resize(rows * count, Float32(0.0))
        for r in range(rows):
            var src = r * last + start
            var dst = r * count
            for c in range(count):
                out[dst + c] = host[src + c]
        var osh = List[Int]()
        for i in range(len(sh) - 1):
            osh.append(sh[i])
        osh.append(count)
        return Tensor.from_host(out, osh^, x.dtype(), ctx)

    # ── MoE FFN (expert-choice; uses nucleus_moe + shared expert) ─────────────
    def _moe_ffn(
        self,
        modulated: Tensor,    # [1, S_IMG, D] img2
        unmodulated: Tensor,  # [1, S_IMG, D] LN(x) before scale2
        temb: Tensor,         # [1, D]
        prefix: String,
        capacity_factor: Float32,
        ctx: DeviceContext,
    ) raises -> Tensor:
        var d = self.config.hidden_size
        var e = self.config.num_experts
        var s = Self.S_IMG

        # capacity = ceil(cap_factor * S / E), min 1.
        var cap_raw = (capacity_factor * Float32(s)) / Float32(e)
        var capacity = Int(cap_raw)
        if Float32(capacity) < cap_raw:
            capacity += 1
        if capacity < 1:
            capacity = 1

        # Router input = cat([temb_tiled, unmodulated], -1) -> [1, S, 2D].
        var router_in = self._router_input(temb, unmodulated, s, d, ctx)  # [S, 2D]
        var logits = linear(router_in, self._w(prefix + "img_mlp.gate.weight"), None, ctx)  # [S, E]
        # scores = softmax(logits.float(), -1) (BF16 round to match recipe).
        var logits_f32 = cast_tensor(logits, STDtype.F32, ctx)
        var scores = softmax_lastdim(logits_f32, ctx)  # [S, E] F32
        # Round through BF16 then back to F32 (recipe precision parity).
        var scores_bf = cast_tensor(scores, STDtype.BF16, ctx)
        var scores_f = cast_tensor(scores_bf, STDtype.F32, ctx)
        # affinity = scores.transpose(0,1) -> [E, S] then [1, E, S].
        var aff_es = transpose(scores_f, 0, 1, ctx)  # [E, S]
        var affinity = reshape(aff_es, _sh3(1, e, s), ctx)  # [1, E, S] F32

        # Routed experts (expert-choice). modulated flattened [S, D].
        var mod_flat = reshape(modulated, _sh2(s, d), ctx)
        var routed_flat = nucleus_moe_expert_forward(
            mod_flat,
            affinity,
            self._w(prefix + "img_mlp.experts.gate_up_proj"),
            self._w(prefix + "img_mlp.experts.down_proj"),
            capacity,
            self.config.route_scale,
            ctx,
        )  # [S, D] F32
        # Cast routed back to model dtype.
        var routed = cast_tensor(routed_flat, self._dtype(), ctx)
        routed = reshape(routed, _sh3(1, s, d), ctx)

        # Shared expert: dense SwiGLU on modulated ([up, gate] layout).
        var shared = self._dense_ffn(
            modulated,
            prefix + "img_mlp.shared_expert.net.0.proj.weight",
            prefix + "img_mlp.shared_expert.net.2.weight",
            ctx,
        )
        return add(shared, routed, ctx)

    # Router input: tile temb [1,D] over S, concat with unmodulated [1,S,D] -> [S, 2D].
    def _router_input(
        self, temb: Tensor, unmod: Tensor, s: Int, d: Int, ctx: DeviceContext
    ) raises -> Tensor:
        var temb_host = temb.to_host(ctx)  # D
        var unmod_host = unmod.to_host(ctx)  # S*D
        var out = List[Float32]()
        out.resize(s * 2 * d, Float32(0.0))
        for si in range(s):
            var dst = si * 2 * d
            for j in range(d):
                out[dst + j] = temb_host[j]
            var usrc = si * d
            for j in range(d):
                out[dst + d + j] = unmod_host[usrc + j]
        var osh = List[Int]()
        osh.append(s)
        osh.append(2 * d)
        return Tensor.from_host(out, osh^, temb.dtype(), ctx)

    # ── one transformer block ─────────────────────────────────────────────────
    def _block_forward(
        self,
        layer_idx: Int,
        x_in: Tensor,        # [1, S_IMG, D]
        enc_normed: Tensor,  # [1, S_TXT, joint] post txt_norm
        temb: Tensor,        # [1, D]
        rope: NucleusRope,
        mask: Tensor,
        ctx: DeviceContext,
    ) raises -> Tensor:
        var prefix = String("transformer_blocks.") + String(layer_idx) + "."
        var d = self.config.hidden_size

        # Modulation: Linear(silu(temb)).chunk(4).
        var temb_silu = silu(temb, ctx)  # [1, D]
        var mod_out = linear(temb_silu, self._w(prefix + "img_mod.1.weight"),
                             Optional[Tensor](self._bias_opt(prefix + "img_mod.1.bias", ctx)),
                             ctx)  # [1, 4D]
        var scale1 = self._chunk_last(mod_out, 0, d, ctx)        # [1, D]
        var gate1 = self._clamp(self._chunk_last(mod_out, d, d, ctx), Float32(-2.0), Float32(2.0), ctx)
        var scale2 = self._chunk_last(mod_out, 2 * d, d, ctx)
        var gate2 = self._clamp(self._chunk_last(mod_out, 3 * d, d, ctx), Float32(-2.0), Float32(2.0), ctx)

        # encoder_proj: [1,S_txt,joint] -> [1,S_txt,D].
        var context = linear(enc_normed, self._w(prefix + "encoder_proj.weight"),
                             Optional[Tensor](self._bias_opt(prefix + "encoder_proj.bias", ctx)),
                             ctx)

        # pre_attn LN (no affine) then *(1+scale1).
        var img_normed = self._layer_norm_noaffine(x_in, ctx)
        var img_mod1 = self._mul_1p_scale(img_normed, scale1, ctx)

        var attn_out = self._attention(prefix, img_mod1, context, rope, mask, ctx)

        # x = x + tanh(gate1) * attn_out.
        var x = self._gate_residual(x_in, gate1, attn_out, ctx)

        # pre_mlp LN + *(1+scale2).
        var img_normed2 = self._layer_norm_noaffine(x, ctx)
        var img_mod2 = self._mul_1p_scale(img_normed2, scale2, ctx)

        var mlp_out: Tensor
        if layer_idx < 3:
            mlp_out = self._dense_ffn(
                img_mod2,
                prefix + "img_mlp.net.0.proj.weight",
                prefix + "img_mlp.net.2.weight",
                ctx,
            )
        else:
            mlp_out = self._moe_ffn(
                img_mod2, img_normed2, temb, prefix,
                self.config.capacity_factor_for(layer_idx), ctx,
            )

        return self._gate_residual(x, gate2, mlp_out, ctx)

    # x = x + tanh(gate) * y, gate [1,D] broadcast over seq.
    def _gate_residual(
        self, x: Tensor, gate: Tensor, y: Tensor, ctx: DeviceContext
    ) raises -> Tensor:
        var d = self.config.hidden_size
        var sh = x.shape()
        var s = sh[1]
        var gate_host = gate.to_host(ctx)  # D
        var gt = List[Float32]()
        gt.resize(d, Float32(0.0))
        for j in range(d):
            gt[j] = Float32(ftanh(Float64(gate_host[j])))
        # broadcast gate over [1,S,D]: build [1,S,D] gate tensor.
        var gfull = List[Float32]()
        gfull.resize(s * d, Float32(0.0))
        for si in range(s):
            for j in range(d):
                gfull[si * d + j] = gt[j]
        var gsh = List[Int]()
        gsh.append(1)
        gsh.append(s)
        gsh.append(d)
        var gtensor = Tensor.from_host(gfull, gsh^, x.dtype(), ctx)
        var gy = mul(gtensor, y, ctx)
        return add(x, gy, ctx)

    # LayerNorm with no affine (gamma=1, beta=0), eps from config.
    def _layer_norm_noaffine(self, x: Tensor, ctx: DeviceContext) raises -> Tensor:
        var d = self.config.hidden_size
        var dtype = x.dtype()
        var ones = List[Float32]()
        var zeros = List[Float32]()
        ones.resize(d, Float32(1.0))
        zeros.resize(d, Float32(0.0))
        var wsh = List[Int]()
        wsh.append(d)
        var w = Tensor.from_host(ones, wsh.copy(), dtype, ctx)
        var b = Tensor.from_host(zeros, wsh^, dtype, ctx)
        return layer_norm(x, w, b, self.config.eps, ctx)

    # out = x * (1 + scale), scale [1,D] broadcast over seq.
    def _mul_1p_scale(
        self, x: Tensor, scale: Tensor, ctx: DeviceContext
    ) raises -> Tensor:
        var d = self.config.hidden_size
        var sh = x.shape()
        var s = sh[1]
        var scale_host = scale.to_host(ctx)  # D
        var sfull = List[Float32]()
        sfull.resize(s * d, Float32(0.0))
        for si in range(s):
            for j in range(d):
                sfull[si * d + j] = Float32(1.0) + scale_host[j]
        var ssh = List[Int]()
        ssh.append(1)
        ssh.append(s)
        ssh.append(d)
        var st = Tensor.from_host(sfull, ssh^, x.dtype(), ctx)
        return mul(x, st, ctx)

    def _clamp(
        self, x: Tensor, lo: Float32, hi: Float32, ctx: DeviceContext
    ) raises -> Tensor:
        var host = x.to_host(ctx)
        var out = List[Float32]()
        out.resize(len(host), Float32(0.0))
        for i in range(len(host)):
            var v = host[i]
            if v < lo:
                v = lo
            elif v > hi:
                v = hi
            out[i] = v
        return Tensor.from_host(out, x.shape(), x.dtype(), ctx)

    # ── full forward (one denoise step) ───────────────────────────────────────
    def forward(
        self,
        hidden_states: Tensor,          # [1, S_IMG, in_channels=64]
        encoder_hidden_states: Tensor,  # [1, S_TXT, joint=4096]
        timestep: Float32,
        f: Int,
        h_grid: Int,
        w_grid: Int,
        ctx: DeviceContext,
    ) raises -> Tensor:
        var cfg = self.config
        var d = cfg.hidden_size

        # img_in: [1,S,64] -> [1,S,D].
        var x = linear(hidden_states, self._w("img_in.weight"),
                       Optional[Tensor](self._bias_opt("img_in.bias", ctx)), ctx)

        # txt_norm RMSNorm on encoder_hidden_states (over joint dim).
        var enc_normed = rms_norm(encoder_hidden_states, self._w("txt_norm.weight"),
                                  cfg.eps, ctx)

        var temb = self._time_text_embed(timestep, ctx)  # [1, D]

        var rope = build_nucleus_3d_rope(f, h_grid, w_grid, Self.S_TXT, cfg,
                                         self._dtype(), ctx)
        comptime S = Self.S_IMG + Self.S_TXT
        var mask = self._zero_mask[S](ctx)

        for layer_idx in range(cfg.num_layers):
            x = self._block_forward(layer_idx, x, enc_normed, temb, rope, mask, ctx)

        # norm_out: AdaLNContinuous. emb = Linear(silu(temb)).chunk(2)=[scale,shift];
        # out = LN(x)*(1+scale)+shift.
        var temb_silu = silu(temb, ctx)
        var mod_emb = linear(temb_silu, self._w("norm_out.linear.weight"),
                            Optional[Tensor](self._bias_opt("norm_out.linear.bias", ctx)),
                            ctx)  # [1, 2D]
        var scale = self._chunk_last(mod_emb, 0, d, ctx)
        var shift = self._chunk_last(mod_emb, d, d, ctx)
        var x_normed = self._layer_norm_noaffine(x, ctx)
        var x_scaled = self._mul_1p_scale(x_normed, scale, ctx)
        var x_mod = self._add_shift(x_scaled, shift, ctx)

        # proj_out: [1,S,D] -> [1,S, patch²*out_ch = 64].
        return linear(x_mod, self._w("proj_out.weight"), None, ctx)

    # out = x + shift, shift [1,D] broadcast over seq.
    def _add_shift(self, x: Tensor, shift: Tensor, ctx: DeviceContext) raises -> Tensor:
        var d = self.config.hidden_size
        var sh = x.shape()
        var s = sh[1]
        var shift_host = shift.to_host(ctx)
        var sf = List[Float32]()
        sf.resize(s * d, Float32(0.0))
        for si in range(s):
            for j in range(d):
                sf[si * d + j] = shift_host[j]
        var ssh = List[Int]()
        ssh.append(1)
        ssh.append(s)
        ssh.append(d)
        var st = Tensor.from_host(sf, ssh^, x.dtype(), ctx)
        return add(x, st, ctx)

    def _zero_mask[S: Int](self, ctx: DeviceContext) raises -> Tensor:
        var dtype = self._dtype()
        var n = self.config.num_heads * S * S
        var dev = ctx.enqueue_create_buffer[DType.uint8](n * dtype.byte_size())
        ctx.enqueue_memset[DType.uint8](dev, 0)
        ctx.synchronize()
        var sh = List[Int]()
        sh.append(1)
        sh.append(self.config.num_heads)
        sh.append(S)
        sh.append(S)
        return Tensor(dev^, sh^, dtype)


# ── shape helpers ─────────────────────────────────────────────────────────────
def _sh2(a: Int, b: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a)
    s.append(b)
    return s^


def _sh3(a: Int, b: Int, c: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a)
    s.append(b)
    s.append(c)
    return s^


def _sh4(a: Int, b: Int, c: Int, d: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a)
    s.append(b)
    s.append(c)
    s.append(d)
    return s^
