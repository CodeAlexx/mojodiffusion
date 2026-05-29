# models/dit/chroma_dit.mojo - Chroma real-weight DiT helpers.
#
# This is the first runtime slice beyond the header contract: it loads the
# real Chroma distilled_guidance_layer and builds the per-step cache used by
# the Rust Chroma DiT. It also exposes bounded real-weight block/proj slices.

from std.gpu.host import DeviceContext
from std.math import cos as fcos, exp as fexp, log as flog, sin as fsin, sqrt
from std.memory import ArcPointer

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.models.dit.chroma_contract import (
    CHROMA_DIT_APPROX_HIDDEN,
    CHROMA_DIT_APPROX_IN,
    CHROMA_DIT_APPROX_LAYERS,
    CHROMA_DIT_HEAD_DIM,
    CHROMA_DIT_HEADS,
    CHROMA_DIT_DOUBLE_BLOCKS,
    CHROMA_DIT_HIDDEN,
    CHROMA_DIT_MOD_INDEX,
    CHROMA_DIT_SINGLE_BLOCKS,
    CHROMA_IMAGE_TOKENS,
    CHROMA_T5_SEQ_LEN,
    chroma_default_checkpoint_path,
)
from serenitymojo.models.dit.flux1_dit import build_flux1_rope_tables
from serenitymojo.ops.activations import gelu, silu
from serenitymojo.ops.attention import sdpa_nomask
from serenitymojo.ops.elementwise import modulate, residual_gate
from serenitymojo.ops.linear import linear
from serenitymojo.ops.norm import layer_norm, rms_norm
from serenitymojo.ops.rope import rope_interleaved
from serenitymojo.ops.tensor_algebra import add, concat, reshape, slice
from serenitymojo.tensor import Tensor


@fieldwise_init
struct ChromaConfig(Copyable, Movable, ImplicitlyCopyable):
    var inner_dim: Int
    var num_heads: Int
    var head_dim: Int
    var approximator_in_channels: Int
    var approximator_hidden_dim: Int
    var approximator_num_layers: Int
    var mod_index_length: Int
    var eps: Float32

    @staticmethod
    def chroma1_hd() -> ChromaConfig:
        return ChromaConfig(
            CHROMA_DIT_HIDDEN,
            CHROMA_DIT_HEADS,
            CHROMA_DIT_HEAD_DIM,
            CHROMA_DIT_APPROX_IN,
            CHROMA_DIT_APPROX_HIDDEN,
            CHROMA_DIT_APPROX_LAYERS,
            CHROMA_DIT_MOD_INDEX,
            Float32(1.0e-6),
        )


struct ChromaStepCache(Movable):
    var pooled_temb: Tensor
    var rope_cos: Tensor
    var rope_sin: Tensor

    def __init__(out self, var pooled_temb: Tensor, var rope_cos: Tensor, var rope_sin: Tensor):
        self.pooled_temb = pooled_temb^
        self.rope_cos = rope_cos^
        self.rope_sin = rope_sin^


struct ChromaDitCache(Movable):
    var weights: List[ArcPointer[Tensor]]
    var name_to_idx: Dict[String, Int]
    var config: ChromaConfig

    def __init__(
        out self,
        var weights: List[ArcPointer[Tensor]],
        var name_to_idx: Dict[String, Int],
        config: ChromaConfig,
    ):
        self.weights = weights^
        self.name_to_idx = name_to_idx^
        self.config = config

    @staticmethod
    def load(path: String, ctx: DeviceContext) raises -> ChromaDitCache:
        """Load only the real distilled_guidance_layer tensors."""
        var sharded = ShardedSafeTensors.open(path)
        var weights = List[ArcPointer[Tensor]]()
        var name_to_idx = Dict[String, Int]()
        for ref nm in sharded.names():
            if not nm.startswith("distilled_guidance_layer."):
                continue
            var tv = sharded.tensor_view(nm)
            var t = Tensor.from_view(tv, ctx)
            var idx = len(weights)
            weights.append(ArcPointer(t^))
            name_to_idx[nm] = idx
        return ChromaDitCache(weights^, name_to_idx^, ChromaConfig.chroma1_hd())

    @staticmethod
    def load_default(ctx: DeviceContext) raises -> ChromaDitCache:
        return ChromaDitCache.load(chroma_default_checkpoint_path(), ctx)

    @staticmethod
    def load_block0_smoke(path: String, ctx: DeviceContext) raises -> ChromaDitCache:
        """Load the step cache, input projections, and first double block."""
        var sharded = ShardedSafeTensors.open(path)
        var weights = List[ArcPointer[Tensor]]()
        var name_to_idx = Dict[String, Int]()
        for ref nm in sharded.names():
            if not _is_block0_smoke_weight(nm):
                continue
            var tv = sharded.tensor_view(nm)
            var t = Tensor.from_view(tv, ctx)
            var idx = len(weights)
            weights.append(ArcPointer(t^))
            name_to_idx[nm] = idx
        return ChromaDitCache(weights^, name_to_idx^, ChromaConfig.chroma1_hd())

    @staticmethod
    def load_default_block0_smoke(ctx: DeviceContext) raises -> ChromaDitCache:
        return ChromaDitCache.load_block0_smoke(chroma_default_checkpoint_path(), ctx)

    @staticmethod
    def load_stage_smoke(path: String, ctx: DeviceContext) raises -> ChromaDitCache:
        """Load step cache, input projections, double blocks 0-1, single blocks 0-1, and proj_out."""
        var sharded = ShardedSafeTensors.open(path)
        var weights = List[ArcPointer[Tensor]]()
        var name_to_idx = Dict[String, Int]()
        for ref nm in sharded.names():
            if not _is_stage_smoke_weight(nm):
                continue
            var tv = sharded.tensor_view(nm)
            var t = Tensor.from_view(tv, ctx)
            var idx = len(weights)
            weights.append(ArcPointer(t^))
            name_to_idx[nm] = idx
        return ChromaDitCache(weights^, name_to_idx^, ChromaConfig.chroma1_hd())

    @staticmethod
    def load_default_stage_smoke(ctx: DeviceContext) raises -> ChromaDitCache:
        return ChromaDitCache.load_stage_smoke(chroma_default_checkpoint_path(), ctx)

    def _w(self, name: String) raises -> ref [self.weights] Tensor:
        if name not in self.name_to_idx:
            raise Error(String("missing Chroma weight: ") + name)
        var idx = self.name_to_idx[name]
        return self.weights[idx][]

    def _clone(self, x: Tensor, ctx: DeviceContext) raises -> Tensor:
        var dev = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
        ctx.enqueue_copy(dst_buf=dev, src_buf=x.buf)
        ctx.synchronize()
        return Tensor(dev^, x.shape(), x.dtype())

    def _linear_b(self, x: Tensor, w_key: String, b_key: String, ctx: DeviceContext) raises -> Tensor:
        ref w = self._w(w_key)
        ref b = self._w(b_key)
        return linear(x, w, Optional[Tensor](self._clone(b, ctx)), ctx)

    def _ones(self, d: Int, dtype: STDtype, ctx: DeviceContext) raises -> Tensor:
        var vals = List[Float32]()
        for _ in range(d):
            vals.append(1.0)
        var sh = List[Int]()
        sh.append(d)
        return Tensor.from_host(vals, sh^, dtype, ctx)

    def _zeros_vec(self, d: Int, dtype: STDtype, ctx: DeviceContext) raises -> Tensor:
        var vals = List[Float32]()
        for _ in range(d):
            vals.append(0.0)
        var sh = List[Int]()
        sh.append(d)
        return Tensor.from_host(vals, sh^, dtype, ctx)

    def _layer_norm_no_affine(self, x: Tensor, ctx: DeviceContext) raises -> Tensor:
        var shape = x.shape()
        var d = shape[len(shape) - 1]
        var ones = self._ones(d, x.dtype(), ctx)
        var zeros = self._zeros_vec(d, x.dtype(), ctx)
        return layer_norm(x, ones, zeros, self.config.eps, ctx)

    def _modulate_pre(
        self, x: Tensor, shift: Tensor, scale: Tensor, ctx: DeviceContext
    ) raises -> Tensor:
        var normed = self._layer_norm_no_affine(x, ctx)
        return modulate(normed, scale, shift, ctx)

    def _pooled_row(
        self, pooled_temb: Tensor, row: Int, ctx: DeviceContext
    ) raises -> Tensor:
        var part = slice(pooled_temb, 1, row, 1, ctx)
        var sh = List[Int]()
        sh.append(self.config.inner_dim)
        return reshape(part, sh^, ctx)

    def _to_bshd[
        N: Int
    ](self, x: Tensor, ctx: DeviceContext) raises -> Tensor:
        var sh = List[Int]()
        sh.append(1)
        sh.append(N)
        sh.append(self.config.num_heads)
        sh.append(self.config.head_dim)
        return reshape(x, sh^, ctx)

    def _from_bshd[
        N: Int
    ](self, x: Tensor, ctx: DeviceContext) raises -> Tensor:
        var sh = List[Int]()
        sh.append(1)
        sh.append(N)
        sh.append(self.config.inner_dim)
        return reshape(x, sh^, ctx)

    def project_image_tokens(self, img_tokens: Tensor, ctx: DeviceContext) raises -> Tensor:
        return self._linear_b(
            img_tokens,
            String("x_embedder.weight"),
            String("x_embedder.bias"),
            ctx,
        )

    def project_text_tokens(self, txt_tokens: Tensor, ctx: DeviceContext) raises -> Tensor:
        return self._linear_b(
            txt_tokens,
            String("context_embedder.weight"),
            String("context_embedder.bias"),
            ctx,
        )

    def _approximator_input(self, timestep: Float32, ctx: DeviceContext) raises -> Tensor:
        """Build `[1, 344, 64]` BF16 input for Chroma's approximator."""
        var cfg = self.config
        var num_channels = cfg.approximator_in_channels // 4
        var time = _sinusoid_values(timestep * Float32(1000.0), num_channels)
        var guidance = _sinusoid_values(Float32(0.0), num_channels)
        var mod_proj = _mod_proj_values(cfg.mod_index_length, 2 * num_channels)

        var vals = List[Float32]()
        for row in range(cfg.mod_index_length):
            for i in range(len(time)):
                vals.append(time[i])
            for i in range(len(guidance)):
                vals.append(guidance[i])
            for i in range(2 * num_channels):
                vals.append(mod_proj[row * (2 * num_channels) + i])

        var sh = List[Int]()
        sh.append(1)
        sh.append(cfg.mod_index_length)
        sh.append(cfg.approximator_in_channels)
        return Tensor.from_host(vals, sh^, STDtype.BF16, ctx)

    def approximator_forward(self, x: Tensor, ctx: DeviceContext) raises -> Tensor:
        """Run distilled_guidance_layer, returning `[1, 344, 3072]` BF16."""
        var h = self._linear_b(
            x,
            String("distilled_guidance_layer.in_proj.weight"),
            String("distilled_guidance_layer.in_proj.bias"),
            ctx,
        )
        for i in range(self.config.approximator_num_layers):
            var prefix = String("distilled_guidance_layer.layers.") + String(i)
            var norm_key = String("distilled_guidance_layer.norms.") + String(i) + String(".weight")
            ref norm_w = self._w(norm_key)
            var n = rms_norm(h, norm_w, self.config.eps, ctx)
            var h1 = self._linear_b(
                n,
                prefix + String(".linear_1.weight"),
                prefix + String(".linear_1.bias"),
                ctx,
            )
            var a = silu(h1, ctx)
            var h2 = self._linear_b(
                a,
                prefix + String(".linear_2.weight"),
                prefix + String(".linear_2.bias"),
                ctx,
            )
            h = add(h, h2, ctx)
        return self._linear_b(
            h,
            String("distilled_guidance_layer.out_proj.weight"),
            String("distilled_guidance_layer.out_proj.bias"),
            ctx,
        )

    def precompute_step_cache[
        N_IMG: Int, N_TXT: Int, S: Int
    ](
        self,
        timestep: Float32,
        img_h2: Int,
        img_w2: Int,
        ctx: DeviceContext,
    ) raises -> ChromaStepCache:
        comptime assert S == N_IMG + N_TXT, "Chroma cache S must be N_IMG + N_TXT"
        var approx_in = self._approximator_input(timestep, ctx)
        var pooled = self.approximator_forward(approx_in, ctx)
        var rope = build_flux1_rope_tables[N_IMG, N_TXT, CHROMA_DIT_HEADS, CHROMA_DIT_HEAD_DIM](
            img_h2, img_w2, ctx, STDtype.BF16
        )
        # Tuple element indexing cannot move Tensor values directly in this
        # Mojo release, so materialize owned copies for the cache struct.
        var rope_cos = self._clone(rope[0], ctx)
        var rope_sin = self._clone(rope[1], ctx)
        return ChromaStepCache(pooled^, rope_cos^, rope_sin^)

    def double_block0_smoke_forward[
        N_IMG: Int, N_TXT: Int, S: Int
    ](
        self,
        img: Tensor,
        txt: Tensor,
        cache: ChromaStepCache,
        ctx: DeviceContext,
    ) raises -> Tensor:
        """Run Chroma transformer_blocks.0 and return cat([txt, img])."""
        return self.double_block_smoke_forward[N_IMG, N_TXT, S](0, img, txt, cache, ctx)

    def double_block_smoke_forward[
        N_IMG: Int, N_TXT: Int, S: Int
    ](
        self,
        block_idx: Int,
        img: Tensor,
        txt: Tensor,
        cache: ChromaStepCache,
        ctx: DeviceContext,
    ) raises -> Tensor:
        """Run one Chroma transformer_blocks.{block_idx} and return cat([txt, img])."""
        comptime assert S == N_IMG + N_TXT, "S must equal N_IMG + N_TXT"
        var p = String("transformer_blocks.") + String(block_idx)
        var img_mod_start = 3 * CHROMA_DIT_SINGLE_BLOCKS
        var txt_mod_start = img_mod_start + 6 * CHROMA_DIT_DOUBLE_BLOCKS
        var block_mod = 6 * block_idx

        var img_shift1 = self._pooled_row(cache.pooled_temb, img_mod_start + block_mod + 0, ctx)
        var img_scale1 = self._pooled_row(cache.pooled_temb, img_mod_start + block_mod + 1, ctx)
        var img_gate1 = self._pooled_row(cache.pooled_temb, img_mod_start + block_mod + 2, ctx)
        var img_shift2 = self._pooled_row(cache.pooled_temb, img_mod_start + block_mod + 3, ctx)
        var img_scale2 = self._pooled_row(cache.pooled_temb, img_mod_start + block_mod + 4, ctx)
        var img_gate2 = self._pooled_row(cache.pooled_temb, img_mod_start + block_mod + 5, ctx)

        var txt_shift1 = self._pooled_row(cache.pooled_temb, txt_mod_start + block_mod + 0, ctx)
        var txt_scale1 = self._pooled_row(cache.pooled_temb, txt_mod_start + block_mod + 1, ctx)
        var txt_gate1 = self._pooled_row(cache.pooled_temb, txt_mod_start + block_mod + 2, ctx)
        var txt_shift2 = self._pooled_row(cache.pooled_temb, txt_mod_start + block_mod + 3, ctx)
        var txt_scale2 = self._pooled_row(cache.pooled_temb, txt_mod_start + block_mod + 4, ctx)
        var txt_gate2 = self._pooled_row(cache.pooled_temb, txt_mod_start + block_mod + 5, ctx)

        var img_norm = self._modulate_pre(img, img_shift1, img_scale1, ctx)
        var txt_norm = self._modulate_pre(txt, txt_shift1, txt_scale1, ctx)

        var img_q = self._to_bshd[N_IMG](self._linear_b(img_norm, p + ".attn.to_q.weight", p + ".attn.to_q.bias", ctx), ctx)
        var img_k = self._to_bshd[N_IMG](self._linear_b(img_norm, p + ".attn.to_k.weight", p + ".attn.to_k.bias", ctx), ctx)
        var img_v = self._to_bshd[N_IMG](self._linear_b(img_norm, p + ".attn.to_v.weight", p + ".attn.to_v.bias", ctx), ctx)
        var txt_q = self._to_bshd[N_TXT](self._linear_b(txt_norm, p + ".attn.add_q_proj.weight", p + ".attn.add_q_proj.bias", ctx), ctx)
        var txt_k = self._to_bshd[N_TXT](self._linear_b(txt_norm, p + ".attn.add_k_proj.weight", p + ".attn.add_k_proj.bias", ctx), ctx)
        var txt_v = self._to_bshd[N_TXT](self._linear_b(txt_norm, p + ".attn.add_v_proj.weight", p + ".attn.add_v_proj.bias", ctx), ctx)

        img_q = rms_norm(img_q, self._w(p + ".attn.norm_q.weight"), self.config.eps, ctx)
        img_k = rms_norm(img_k, self._w(p + ".attn.norm_k.weight"), self.config.eps, ctx)
        txt_q = rms_norm(txt_q, self._w(p + ".attn.norm_added_q.weight"), self.config.eps, ctx)
        txt_k = rms_norm(txt_k, self._w(p + ".attn.norm_added_k.weight"), self.config.eps, ctx)

        var q = concat(1, ctx, txt_q, img_q)
        var k = concat(1, ctx, txt_k, img_k)
        var v = concat(1, ctx, txt_v, img_v)
        q = rope_interleaved(q, cache.rope_cos, cache.rope_sin, ctx)
        k = rope_interleaved(k, cache.rope_cos, cache.rope_sin, ctx)
        var att = sdpa_nomask[1, S, 24, 128](
            q, k, v, Float32(1.0) / sqrt(Float32(128)), ctx
        )

        var txt_att = self._from_bshd[N_TXT](slice(att, 1, 0, N_TXT, ctx), ctx)
        var img_att = self._from_bshd[N_IMG](slice(att, 1, N_TXT, N_IMG, ctx), ctx)
        var img_o = self._linear_b(img_att, p + ".attn.to_out.0.weight", p + ".attn.to_out.0.bias", ctx)
        var txt_o = self._linear_b(txt_att, p + ".attn.to_add_out.weight", p + ".attn.to_add_out.bias", ctx)

        var img_r = residual_gate(img, img_gate1, img_o, ctx)
        var txt_r = residual_gate(txt, txt_gate1, txt_o, ctx)

        var img_ff_in = self._modulate_pre(img_r, img_shift2, img_scale2, ctx)
        var img_ff = self._linear_b(img_ff_in, p + ".ff.net.0.proj.weight", p + ".ff.net.0.proj.bias", ctx)
        img_ff = gelu(img_ff, ctx)
        img_ff = self._linear_b(img_ff, p + ".ff.net.2.weight", p + ".ff.net.2.bias", ctx)
        var img_final = residual_gate(img_r, img_gate2, img_ff, ctx)

        var txt_ff_in = self._modulate_pre(txt_r, txt_shift2, txt_scale2, ctx)
        var txt_ff = self._linear_b(txt_ff_in, p + ".ff_context.net.0.proj.weight", p + ".ff_context.net.0.proj.bias", ctx)
        txt_ff = gelu(txt_ff, ctx)
        txt_ff = self._linear_b(txt_ff, p + ".ff_context.net.2.weight", p + ".ff_context.net.2.bias", ctx)
        var txt_final = residual_gate(txt_r, txt_gate2, txt_ff, ctx)

        return concat(1, ctx, txt_final, img_final)

    def two_double_blocks_smoke_forward[
        N_IMG: Int, N_TXT: Int, S: Int
    ](
        self,
        img: Tensor,
        txt: Tensor,
        cache: ChromaStepCache,
        ctx: DeviceContext,
    ) raises -> Tensor:
        """Run transformer_blocks.0 then .1 and return cat([txt, img])."""
        comptime assert S == N_IMG + N_TXT, "S must equal N_IMG + N_TXT"
        var merged0 = self.double_block_smoke_forward[N_IMG, N_TXT, S](
            0, img, txt, cache, ctx
        )
        var txt1 = slice(merged0, 1, 0, N_TXT, ctx)
        var img1 = slice(merged0, 1, N_TXT, N_IMG, ctx)
        return self.double_block_smoke_forward[N_IMG, N_TXT, S](
            1, img1, txt1, cache, ctx
        )

    def single_block0_smoke_forward[
        S: Int
    ](
        self,
        x: Tensor,
        cache: ChromaStepCache,
        ctx: DeviceContext,
    ) raises -> Tensor:
        """Run Chroma single_transformer_blocks.0 over cat([txt, img])."""
        return self.single_block_smoke_forward[S](0, x, cache, ctx)

    def single_block_smoke_forward[
        S: Int
    ](
        self,
        block_idx: Int,
        x: Tensor,
        cache: ChromaStepCache,
        ctx: DeviceContext,
    ) raises -> Tensor:
        """Run Chroma single_transformer_blocks.{block_idx} over cat([txt, img])."""
        var p = String("single_transformer_blocks.") + String(block_idx)
        var mod_start = 3 * block_idx
        var shift = self._pooled_row(cache.pooled_temb, mod_start + 0, ctx)
        var scale = self._pooled_row(cache.pooled_temb, mod_start + 1, ctx)
        var gate = self._pooled_row(cache.pooled_temb, mod_start + 2, ctx)

        var x_norm = self._modulate_pre(x, shift, scale, ctx)
        var q = self._to_bshd[S](self._linear_b(x_norm, p + ".attn.to_q.weight", p + ".attn.to_q.bias", ctx), ctx)
        var k = self._to_bshd[S](self._linear_b(x_norm, p + ".attn.to_k.weight", p + ".attn.to_k.bias", ctx), ctx)
        var v = self._to_bshd[S](self._linear_b(x_norm, p + ".attn.to_v.weight", p + ".attn.to_v.bias", ctx), ctx)

        q = rms_norm(q, self._w(p + ".attn.norm_q.weight"), self.config.eps, ctx)
        k = rms_norm(k, self._w(p + ".attn.norm_k.weight"), self.config.eps, ctx)
        q = rope_interleaved(q, cache.rope_cos, cache.rope_sin, ctx)
        k = rope_interleaved(k, cache.rope_cos, cache.rope_sin, ctx)
        var att = sdpa_nomask[1, S, 24, 128](
            q, k, v, Float32(1.0) / sqrt(Float32(128)), ctx
        )
        var att_flat = self._from_bshd[S](att, ctx)

        var mlp = self._linear_b(x_norm, p + ".proj_mlp.weight", p + ".proj_mlp.bias", ctx)
        mlp = gelu(mlp, ctx)
        var out_in = concat(2, ctx, att_flat, mlp)
        var out = self._linear_b(out_in, p + ".proj_out.weight", p + ".proj_out.bias", ctx)
        return residual_gate(x, gate, out, ctx)

    def two_single_blocks_smoke_forward[
        S: Int
    ](
        self,
        x: Tensor,
        cache: ChromaStepCache,
        ctx: DeviceContext,
    ) raises -> Tensor:
        """Run single_transformer_blocks.0 then .1 over cat([txt, img])."""
        var h = self.single_block_smoke_forward[S](0, x, cache, ctx)
        return self.single_block_smoke_forward[S](1, h, cache, ctx)

    def final_image_projection_smoke[
        N_IMG: Int, N_TXT: Int, S: Int
    ](
        self,
        x: Tensor,
        cache: ChromaStepCache,
        ctx: DeviceContext,
    ) raises -> Tensor:
        """Apply Chroma norm_out modulation and proj_out to the image slice."""
        comptime assert S == N_IMG + N_TXT, "S must equal N_IMG + N_TXT"
        var img = slice(x, 1, N_TXT, N_IMG, ctx)
        var shift = self._pooled_row(cache.pooled_temb, self.config.mod_index_length - 2, ctx)
        var scale = self._pooled_row(cache.pooled_temb, self.config.mod_index_length - 1, ctx)
        var normed = self._layer_norm_no_affine(img, ctx)
        var modulated = modulate(normed, scale, shift, ctx)
        return self._linear_b(
            modulated,
            String("proj_out.weight"),
            String("proj_out.bias"),
            ctx,
        )


def _sinusoid_values(t: Float32, dim: Int) raises -> List[Float32]:
    if dim % 2 != 0:
        raise Error("_sinusoid_values: dim must be even")
    var half = dim // 2
    var vals = List[Float32]()
    var neg_ln_mp = -flog(Float32(10000.0))
    for i in range(half):
        var freq = fexp(neg_ln_mp * (Float32(i) / Float32(half)))
        vals.append(fcos(t * freq))
    for i in range(half):
        var freq = fexp(neg_ln_mp * (Float32(i) / Float32(half)))
        vals.append(fsin(t * freq))
    return vals^


def _mod_proj_values(out_dim: Int, dim: Int) raises -> List[Float32]:
    if dim % 2 != 0:
        raise Error("_mod_proj_values: dim must be even")
    var half = dim // 2
    var vals = List[Float32]()
    var neg_ln_mp = -flog(Float32(10000.0))
    for row in range(out_dim):
        var t = Float32(row) * Float32(1000.0)
        for i in range(half):
            var freq = fexp(neg_ln_mp * (Float32(i) / Float32(half)))
            vals.append(fcos(t * freq))
        for i in range(half):
            var freq = fexp(neg_ln_mp * (Float32(i) / Float32(half)))
            vals.append(fsin(t * freq))
    return vals^


def _is_block0_smoke_weight(name: String) -> Bool:
    return (
        name.startswith("distilled_guidance_layer.")
        or name.startswith("transformer_blocks.0.")
        or name == "x_embedder.weight"
        or name == "x_embedder.bias"
        or name == "context_embedder.weight"
        or name == "context_embedder.bias"
    )


def _is_stage_smoke_weight(name: String) -> Bool:
    return (
        _is_block0_smoke_weight(name)
        or name.startswith("transformer_blocks.1.")
        or name.startswith("single_transformer_blocks.0.")
        or name.startswith("single_transformer_blocks.1.")
        or name == "proj_out.weight"
        or name == "proj_out.bias"
    )


def chroma_step_cache_stats(cache: ChromaStepCache, ctx: DeviceContext) raises -> Tuple[Float32, Float32, Float32]:
    var h = cache.pooled_temb.to_host(ctx)
    var n = len(h)
    var s = 0.0
    var s2 = 0.0
    var amax = 0.0
    for i in range(n):
        var v = Float64(h[i])
        s += v
        s2 += v * v
        var av = v if v >= 0.0 else -v
        if av > amax:
            amax = av
    var mean = s / Float64(n)
    var var_ = s2 / Float64(n) - mean * mean
    if var_ < 0.0:
        var_ = 0.0
    return (Float32(mean), Float32(sqrt(var_)), Float32(amax))


def chroma_full_image_tokens() -> Int:
    return CHROMA_IMAGE_TOKENS


def chroma_full_text_tokens() -> Int:
    return CHROMA_T5_SEQ_LEN
