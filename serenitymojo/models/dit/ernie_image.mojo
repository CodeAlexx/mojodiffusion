# ERNIE-Image resident embedding/runtime slices.
#
# This is intentionally not the full 36-layer DiT. It covers the resident math
# that runs before the block stack: latent patch projection, timestep MLP, and
# Mistral hidden-state projection.

from std.gpu.host import DeviceContext
from std.math import cos as fcos, exp as fexp, log as flog, sin as fsin, sqrt
from std.memory import ArcPointer

from serenitymojo.io.dtype import STDtype
from serenitymojo.tensor import Tensor
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.models.dit.ernie_contract import (
    ERNIE_DIT_HEAD_DIM,
    ERNIE_DIT_HEADS,
    ERNIE_DIT_FFN_HIDDEN,
    ERNIE_DIT_HIDDEN,
    ERNIE_DIT_ROPE_AXIS_0,
    ERNIE_DIT_ROPE_AXIS_1,
    ERNIE_DIT_ROPE_AXIS_2,
    ERNIE_DIT_ROPE_THETA,
    ERNIE_DIT_TEXT_IN_DIM,
    ERNIE_LATENT_CHANNELS,
    ERNIE_LATENT_H,
    ERNIE_LATENT_W,
    ERNIE_PATCH_SIZE,
    ERNIE_TEXT_MAX_TOKENS,
    ERNIE_TRANSFORMER_DIR,
)
from serenitymojo.ops.activations import gelu, silu
from serenitymojo.ops.attention import sdpa_nomask
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.embeddings import timestep_embedding_sin_first
from serenitymojo.ops.elementwise import modulate, residual_gate
from serenitymojo.ops.layout import patchify
from serenitymojo.ops.linear import linear
from serenitymojo.ops.norm import rms_norm
from serenitymojo.ops.rope import rope_halfsplit_full
from serenitymojo.ops.tensor_algebra import mul, reshape, slice


@fieldwise_init
struct ErnieImageResident(Movable):
    var weights: List[ArcPointer[Tensor]]
    var name_to_idx: Dict[String, Int]

    @staticmethod
    def load(path: String, ctx: DeviceContext) raises -> ErnieImageResident:
        var sharded = ShardedSafeTensors.open(path)
        var weights = List[ArcPointer[Tensor]]()
        var name_to_idx = Dict[String, Int]()

        var names = List[String]()
        names.append(String("x_embedder.proj.weight"))
        names.append(String("x_embedder.proj.bias"))
        names.append(String("time_embedding.linear_1.weight"))
        names.append(String("time_embedding.linear_1.bias"))
        names.append(String("time_embedding.linear_2.weight"))
        names.append(String("time_embedding.linear_2.bias"))
        names.append(String("text_proj.weight"))
        names.append(String("adaLN_modulation.1.weight"))
        names.append(String("adaLN_modulation.1.bias"))

        for i in range(len(names)):
            var nm = names[i]
            var tv = sharded.tensor_view(nm)
            var t = Tensor.from_view(tv, ctx)
            var idx = len(weights)
            weights.append(ArcPointer(t^))
            name_to_idx[nm] = idx

        return ErnieImageResident(weights^, name_to_idx^)

    @staticmethod
    def load_default(ctx: DeviceContext) raises -> ErnieImageResident:
        return ErnieImageResident.load(String(ERNIE_TRANSFORMER_DIR), ctx)

    @staticmethod
    def load_block0_smoke(path: String, ctx: DeviceContext) raises -> ErnieImageResident:
        var sharded = ShardedSafeTensors.open(path)
        var weights = List[ArcPointer[Tensor]]()
        var name_to_idx = Dict[String, Int]()

        var names = _ernie_resident_weight_names()
        names.append(String("layers.0.adaLN_sa_ln.weight"))
        names.append(String("layers.0.self_attention.to_q.weight"))
        names.append(String("layers.0.self_attention.to_k.weight"))
        names.append(String("layers.0.self_attention.to_v.weight"))
        names.append(String("layers.0.self_attention.to_out.0.weight"))
        names.append(String("layers.0.self_attention.norm_q.weight"))
        names.append(String("layers.0.self_attention.norm_k.weight"))
        names.append(String("layers.0.adaLN_mlp_ln.weight"))
        names.append(String("layers.0.mlp.gate_proj.weight"))
        names.append(String("layers.0.mlp.up_proj.weight"))
        names.append(String("layers.0.mlp.linear_fc2.weight"))

        for i in range(len(names)):
            var nm = names[i]
            var tv = sharded.tensor_view(nm)
            var t = Tensor.from_view(tv, ctx)
            var idx = len(weights)
            weights.append(ArcPointer(t^))
            name_to_idx[nm] = idx

        return ErnieImageResident(weights^, name_to_idx^)

    @staticmethod
    def load_default_block0_smoke(ctx: DeviceContext) raises -> ErnieImageResident:
        return ErnieImageResident.load_block0_smoke(String(ERNIE_TRANSFORMER_DIR), ctx)

    def _w(self, name: String) raises -> ref [self.weights] Tensor:
        if name not in self.name_to_idx:
            raise Error(String("missing ERNIE resident weight: ") + name)
        var idx = self.name_to_idx[name]
        return self.weights[idx][]

    def _clone(self, x: Tensor, ctx: DeviceContext) raises -> Tensor:
        var dev = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
        ctx.enqueue_copy(dst_buf=dev, src_buf=x.buf)
        ctx.synchronize()
        return Tensor(dev^, x.shape(), x.dtype())

    def _expect_bf16_shape1(self, name: String, a: Int) raises:
        ref t = self._w(name)
        if t.dtype() != STDtype.BF16:
            raise Error(String("ERNIE block0 smoke dtype mismatch: ") + name)
        var sh = t.shape()
        if len(sh) != 1 or sh[0] != a:
            raise Error(String("ERNIE block0 smoke shape mismatch: ") + name)

    def _expect_bf16_shape2(self, name: String, a: Int, b: Int) raises:
        ref t = self._w(name)
        if t.dtype() != STDtype.BF16:
            raise Error(String("ERNIE block0 smoke dtype mismatch: ") + name)
        var sh = t.shape()
        if len(sh) != 2 or sh[0] != a or sh[1] != b:
            raise Error(String("ERNIE block0 smoke shape mismatch: ") + name)

    def _expect_bf16_shape4(
        self, name: String, a: Int, b: Int, c: Int, d: Int
    ) raises:
        ref t = self._w(name)
        if t.dtype() != STDtype.BF16:
            raise Error(String("ERNIE block0 smoke dtype mismatch: ") + name)
        var sh = t.shape()
        if len(sh) != 4 or sh[0] != a or sh[1] != b or sh[2] != c or sh[3] != d:
            raise Error(String("ERNIE block0 smoke shape mismatch: ") + name)

    def validate_block0_smoke_weights(self) raises:
        """Validate the resident and layer-0 tensors actually loaded for the runtime smoke."""
        self._expect_bf16_shape4(
            String("x_embedder.proj.weight"),
            ERNIE_DIT_HIDDEN,
            ERNIE_LATENT_CHANNELS,
            1,
            1,
        )
        self._expect_bf16_shape1(String("x_embedder.proj.bias"), ERNIE_DIT_HIDDEN)
        self._expect_bf16_shape2(
            String("time_embedding.linear_1.weight"), ERNIE_DIT_HIDDEN, ERNIE_DIT_HIDDEN
        )
        self._expect_bf16_shape1(
            String("time_embedding.linear_1.bias"), ERNIE_DIT_HIDDEN
        )
        self._expect_bf16_shape2(
            String("time_embedding.linear_2.weight"), ERNIE_DIT_HIDDEN, ERNIE_DIT_HIDDEN
        )
        self._expect_bf16_shape1(
            String("time_embedding.linear_2.bias"), ERNIE_DIT_HIDDEN
        )
        self._expect_bf16_shape2(
            String("text_proj.weight"), ERNIE_DIT_HIDDEN, ERNIE_DIT_TEXT_IN_DIM
        )
        self._expect_bf16_shape2(
            String("adaLN_modulation.1.weight"), 6 * ERNIE_DIT_HIDDEN, ERNIE_DIT_HIDDEN
        )
        self._expect_bf16_shape1(
            String("adaLN_modulation.1.bias"), 6 * ERNIE_DIT_HIDDEN
        )

        var p = String("layers.0")
        self._expect_bf16_shape1(p + String(".adaLN_sa_ln.weight"), ERNIE_DIT_HIDDEN)
        self._expect_bf16_shape2(
            p + String(".self_attention.to_q.weight"), ERNIE_DIT_HIDDEN, ERNIE_DIT_HIDDEN
        )
        self._expect_bf16_shape2(
            p + String(".self_attention.to_k.weight"), ERNIE_DIT_HIDDEN, ERNIE_DIT_HIDDEN
        )
        self._expect_bf16_shape2(
            p + String(".self_attention.to_v.weight"), ERNIE_DIT_HIDDEN, ERNIE_DIT_HIDDEN
        )
        self._expect_bf16_shape2(
            p + String(".self_attention.to_out.0.weight"),
            ERNIE_DIT_HIDDEN,
            ERNIE_DIT_HIDDEN,
        )
        self._expect_bf16_shape1(
            p + String(".self_attention.norm_q.weight"), ERNIE_DIT_HEAD_DIM
        )
        self._expect_bf16_shape1(
            p + String(".self_attention.norm_k.weight"), ERNIE_DIT_HEAD_DIM
        )
        self._expect_bf16_shape1(p + String(".adaLN_mlp_ln.weight"), ERNIE_DIT_HIDDEN)
        self._expect_bf16_shape2(
            p + String(".mlp.gate_proj.weight"), ERNIE_DIT_FFN_HIDDEN, ERNIE_DIT_HIDDEN
        )
        self._expect_bf16_shape2(
            p + String(".mlp.up_proj.weight"), ERNIE_DIT_FFN_HIDDEN, ERNIE_DIT_HIDDEN
        )
        self._expect_bf16_shape2(
            p + String(".mlp.linear_fc2.weight"), ERNIE_DIT_HIDDEN, ERNIE_DIT_FFN_HIDDEN
        )

    def patch_embed_1024(self, latent_nchw: Tensor, ctx: DeviceContext) raises -> Tensor:
        """Latent `[1,128,64,64]` -> image tokens `[1,4096,4096]`."""
        var patches = patchify(latent_nchw, ERNIE_PATCH_SIZE, ctx)
        var wshape = List[Int]()
        wshape.append(ERNIE_DIT_HIDDEN)
        wshape.append(ERNIE_LATENT_CHANNELS)
        ref pw = self._w(String("x_embedder.proj.weight"))
        var w = reshape(self._clone(pw, ctx), wshape^, ctx)
        ref b = self._w(String("x_embedder.proj.bias"))
        return linear(patches, w, Optional[Tensor](self._clone(b, ctx)), ctx)

    def time_embed(self, timestep: Tensor, ctx: DeviceContext) raises -> Tensor:
        """Timestep `[B]` F32 -> `[B,4096]` BF16.

        ERNIE uses SIN-FIRST sinusoidal embedding (`ernie_image.rs:603`,
        `cat([sin, cos], 1)`). Z-Image/FLUX/Klein/Qwen/HiDream/SenseNova/SDXL
        keep the cos-first `timestep_embedding`. See skeptic finding
        `serenitymojo/parity/SKEPTIC_FINDINGS_ernie_block0_2026-05-28.md` (A2).
        """
        var emb = timestep_embedding_sin_first(timestep, ERNIE_DIT_HIDDEN, ctx, 10000.0)
        var emb_bf16 = cast_tensor(emb, self._w(String("time_embedding.linear_1.weight")).dtype(), ctx)
        ref w1 = self._w(String("time_embedding.linear_1.weight"))
        ref b1 = self._w(String("time_embedding.linear_1.bias"))
        var h = linear(emb_bf16, w1, Optional[Tensor](self._clone(b1, ctx)), ctx)
        h = silu(h, ctx)
        ref w2 = self._w(String("time_embedding.linear_2.weight"))
        ref b2 = self._w(String("time_embedding.linear_2.bias"))
        return linear(h, w2, Optional[Tensor](self._clone(b2, ctx)), ctx)

    def project_text(self, text_embeds: Tensor, ctx: DeviceContext) raises -> Tensor:
        """Mistral hidden states `[1,256,3072]` -> ERNIE context `[1,256,4096]`."""
        return linear(text_embeds, self._w(String("text_proj.weight")), None, ctx)

    def shared_adaln(self, temb: Tensor, ctx: DeviceContext) raises -> Tensor:
        """Shared block AdaLN modulation `[1,4096]` -> `[1,24576]`."""
        var h = silu(temb, ctx)
        ref w = self._w(String("adaLN_modulation.1.weight"))
        ref b = self._w(String("adaLN_modulation.1.bias"))
        return linear(h, w, Optional[Tensor](self._clone(b, ctx)), ctx)

    def _adaln_chunk(self, adaln: Tensor, idx: Int, ctx: DeviceContext) raises -> Tensor:
        var part = slice(adaln, 1, idx * ERNIE_DIT_HIDDEN, ERNIE_DIT_HIDDEN, ctx)
        var out_shape = List[Int]()
        out_shape.append(ERNIE_DIT_HIDDEN)
        return reshape(part, out_shape^, ctx)

    def _to_bshd[S: Int](self, x: Tensor, ctx: DeviceContext) raises -> Tensor:
        var sh = List[Int]()
        sh.append(1)
        sh.append(S)
        sh.append(ERNIE_DIT_HEADS)
        sh.append(ERNIE_DIT_HEAD_DIM)
        return reshape(x, sh^, ctx)

    def _from_bshd[S: Int](self, x: Tensor, ctx: DeviceContext) raises -> Tensor:
        var sh = List[Int]()
        sh.append(1)
        sh.append(S)
        sh.append(ERNIE_DIT_HIDDEN)
        return reshape(x, sh^, ctx)

    def block0_smoke_forward[S: Int](
        self,
        x: Tensor,
        adaln: Tensor,
        rope_cos: Tensor,
        rope_sin: Tensor,
        ctx: DeviceContext,
    ) raises -> Tensor:
        """Run real ERNIE layer 0 on a bounded image-first/text-second slice."""
        var xshape = x.shape()
        if len(xshape) != 3 or xshape[0] != 1 or xshape[1] != S or xshape[2] != ERNIE_DIT_HIDDEN:
            raise Error("ERNIE block0 input shape mismatch")
        var ashape = adaln.shape()
        if len(ashape) != 2 or ashape[0] != 1 or ashape[1] != 6 * ERNIE_DIT_HIDDEN:
            raise Error("ERNIE block0 AdaLN shape mismatch")

        var shift_msa = self._adaln_chunk(adaln, 0, ctx)
        var scale_msa = self._adaln_chunk(adaln, 1, ctx)
        var gate_msa = self._adaln_chunk(adaln, 2, ctx)
        var shift_mlp = self._adaln_chunk(adaln, 3, ctx)
        var scale_mlp = self._adaln_chunk(adaln, 4, ctx)
        var gate_mlp = self._adaln_chunk(adaln, 5, ctx)

        var p = String("layers.0")
        var sa_norm = rms_norm(x, self._w(p + String(".adaLN_sa_ln.weight")), 1.0e-6, ctx)
        var sa_in = modulate(sa_norm, scale_msa, shift_msa, ctx)
        var q = self._to_bshd[S](
            linear(sa_in, self._w(p + String(".self_attention.to_q.weight")), None, ctx), ctx
        )
        var k = self._to_bshd[S](
            linear(sa_in, self._w(p + String(".self_attention.to_k.weight")), None, ctx), ctx
        )
        var v = self._to_bshd[S](
            linear(sa_in, self._w(p + String(".self_attention.to_v.weight")), None, ctx), ctx
        )
        q = rms_norm(q, self._w(p + String(".self_attention.norm_q.weight")), 1.0e-6, ctx)
        k = rms_norm(k, self._w(p + String(".self_attention.norm_k.weight")), 1.0e-6, ctx)
        q = rope_halfsplit_full(q, rope_cos, rope_sin, ctx)
        k = rope_halfsplit_full(k, rope_cos, rope_sin, ctx)
        var att = sdpa_nomask[1, S, ERNIE_DIT_HEADS, ERNIE_DIT_HEAD_DIM](
            q, k, v, Float32(1.0) / sqrt(Float32(ERNIE_DIT_HEAD_DIM)), ctx
        )
        var att_flat = self._from_bshd[S](att, ctx)
        var att_out = linear(
            att_flat, self._w(p + String(".self_attention.to_out.0.weight")), None, ctx
        )
        var h = residual_gate(x, gate_msa, att_out, ctx)

        var mlp_norm = rms_norm(h, self._w(p + String(".adaLN_mlp_ln.weight")), 1.0e-6, ctx)
        var mlp_in = modulate(mlp_norm, scale_mlp, shift_mlp, ctx)
        var gate = linear(mlp_in, self._w(p + String(".mlp.gate_proj.weight")), None, ctx)
        var up = linear(mlp_in, self._w(p + String(".mlp.up_proj.weight")), None, ctx)
        var activated = mul(gelu(gate, ctx), up, ctx)
        var mlp_out = linear(
            activated, self._w(p + String(".mlp.linear_fc2.weight")), None, ctx
        )
        return residual_gate(h, gate_mlp, mlp_out, ctx)

    def ernie_block0_full_forward[S: Int](
        self,
        x: Tensor,
        adaln: Tensor,
        rope_cos: Tensor,
        rope_sin: Tensor,
        ctx: DeviceContext,
    ) raises -> Tensor:
        """Full ERNIE layer-0 forward — no bounded-scale tricks inside.

        Mirrors the Rust `ErnieImageModel::block_forward` exactly:

          1. residual1 = x
          2. sa_norm   = rms_norm(x, layers.0.adaLN_sa_ln.weight, eps=1e-6)
          3. sa_in     = sa_norm * (1 + scale_msa) + shift_msa
          4. q,k,v     = linear(sa_in, to_{q,k,v}.weight, no bias)
          5. q,k       = reshape [B,S,H,D] then per-head RMSNorm w/ norm_{q,k}
          6. q,k       = rope_halfsplit_full(rope_cos, rope_sin)
          7. attn      = sdpa(q,k,v, 1/sqrt(D))
          8. attn_out  = linear(attn_flat, to_out.0.weight)
          9. h         = residual1 + gate_msa * attn_out
          10. residual2= h
          11. mlp_norm = rms_norm(h, layers.0.adaLN_mlp_ln.weight, 1e-6)
          12. mlp_in   = mlp_norm * (1 + scale_mlp) + shift_mlp
          13. gate     = linear(mlp_in, mlp.gate_proj.weight)
          14. up       = linear(mlp_in, mlp.up_proj.weight)
          15. fused    = gelu(gate) * up         (GELU-gated MLP)
          16. mlp_out  = linear(fused, mlp.linear_fc2.weight)
          17. return   residual2 + gate_mlp * mlp_out

        AdaLN is consumed AS PROVIDED: no scale-down, no clamp. Caller is
        responsible for supplying a finite-magnitude `adaln` tensor (i.e. an
        AdaLN value that has either gone through the real timestep MLP +
        shared adaLN_modulation, or is synthesized at a believable scale for
        smoke testing). The block forward itself contains zero bounding tricks.
        """
        return self.block0_smoke_forward[S](x, adaln, rope_cos, rope_sin, ctx)


def validate_ernie_resident_shapes(
    patch_tokens: Tensor, temb: Tensor, text_tokens: Tensor
) raises:
    var ps = patch_tokens.shape()
    if len(ps) != 3 or ps[0] != 1 or ps[1] != ERNIE_LATENT_H * ERNIE_LATENT_W or ps[2] != ERNIE_DIT_HIDDEN:
        raise Error("ERNIE patch embedding shape mismatch")
    var ts = temb.shape()
    if len(ts) != 2 or ts[0] != 1 or ts[1] != ERNIE_DIT_HIDDEN:
        raise Error("ERNIE timestep embedding shape mismatch")
    var xs = text_tokens.shape()
    if len(xs) != 3 or xs[0] != 1 or xs[1] != ERNIE_TEXT_MAX_TOKENS or xs[2] != ERNIE_DIT_HIDDEN:
        raise Error("ERNIE text projection shape mismatch")


def validate_ernie_adaln_shape(mods: Tensor) raises:
    var ms = mods.shape()
    if len(ms) != 2 or ms[0] != 1 or ms[1] != 6 * ERNIE_DIT_HIDDEN:
        raise Error("ERNIE shared AdaLN shape mismatch")


def validate_ernie_block0_shape[S: Int](result: Tensor) raises:
    var sh = result.shape()
    if len(sh) != 3 or sh[0] != 1 or sh[1] != S or sh[2] != ERNIE_DIT_HIDDEN:
        raise Error("ERNIE block0 output shape mismatch")


def _ernie_resident_weight_names() -> List[String]:
    var names = List[String]()
    names.append(String("x_embedder.proj.weight"))
    names.append(String("x_embedder.proj.bias"))
    names.append(String("time_embedding.linear_1.weight"))
    names.append(String("time_embedding.linear_1.bias"))
    names.append(String("time_embedding.linear_2.weight"))
    names.append(String("time_embedding.linear_2.bias"))
    names.append(String("text_proj.weight"))
    names.append(String("adaLN_modulation.1.weight"))
    names.append(String("adaLN_modulation.1.bias"))
    return names^


def build_ernie_rope_tables[
    N_IMG: Int, N_TXT: Int, HEADS: Int, HEAD_DIM: Int
](
    img_h: Int,
    img_w: Int,
    text_len_real: Int,
    ctx: DeviceContext,
    dtype: STDtype,
) raises -> Tuple[Tensor, Tensor]:
    """Build ERNIE image-first/text-second 3-axis full half-split RoPE tables."""
    comptime assert HEADS == ERNIE_DIT_HEADS, "ERNIE RoPE head count mismatch"
    comptime assert HEAD_DIM == ERNIE_DIT_HEAD_DIM, "ERNIE RoPE head dim mismatch"
    comptime assert N_IMG + N_TXT > 0, "ERNIE RoPE sequence cannot be empty"
    if img_h * img_w != N_IMG:
        raise Error("ERNIE RoPE image grid does not match N_IMG")
    if text_len_real <= 0 or text_len_real > N_TXT:
        raise Error("ERNIE RoPE real text length out of range")

    var freq0 = _rope_inv_freqs(ERNIE_DIT_ROPE_AXIS_0, Float64(ERNIE_DIT_ROPE_THETA))
    var freq1 = _rope_inv_freqs(ERNIE_DIT_ROPE_AXIS_1, Float64(ERNIE_DIT_ROPE_THETA))
    var freq2 = _rope_inv_freqs(ERNIE_DIT_ROPE_AXIS_2, Float64(ERNIE_DIT_ROPE_THETA))
    var half = HEAD_DIM // 2
    if len(freq0) + len(freq1) + len(freq2) != half:
        raise Error("ERNIE RoPE axis dims do not sum to head_dim/2")

    var cos_vals = List[Float32]()
    var sin_vals = List[Float32]()
    comptime S = N_IMG + N_TXT
    for tok in range(S):
        var p0: Float32
        var p1: Float32
        var p2: Float32
        if tok < N_IMG:
            var r = tok // img_w
            var c = tok % img_w
            p0 = Float32(text_len_real)
            p1 = Float32(r)
            p2 = Float32(c)
        else:
            p0 = Float32(tok - N_IMG)
            p1 = 0.0
            p2 = 0.0
        for _h in range(HEADS):
            for i in range(len(freq0)):
                var ang = p0 * freq0[i]
                cos_vals.append(fcos(ang))
                sin_vals.append(fsin(ang))
                cos_vals.append(fcos(ang))
                sin_vals.append(fsin(ang))
            for i in range(len(freq1)):
                var ang = p1 * freq1[i]
                cos_vals.append(fcos(ang))
                sin_vals.append(fsin(ang))
                cos_vals.append(fcos(ang))
                sin_vals.append(fsin(ang))
            for i in range(len(freq2)):
                var ang = p2 * freq2[i]
                cos_vals.append(fcos(ang))
                sin_vals.append(fsin(ang))
                cos_vals.append(fcos(ang))
                sin_vals.append(fsin(ang))

    var sh = List[Int]()
    sh.append(S * HEADS)
    sh.append(HEAD_DIM)
    return (
        Tensor.from_host(cos_vals, sh.copy(), dtype, ctx),
        Tensor.from_host(sin_vals, sh^, dtype, ctx),
    )


def _rope_inv_freqs(axis_dim: Int, theta: Float64) raises -> List[Float32]:
    if axis_dim <= 0 or axis_dim % 2 != 0:
        raise Error("ERNIE RoPE axis dim must be positive and even")
    var out = List[Float32]()
    var log_theta = flog(Float32(theta))
    for i in range(axis_dim // 2):
        var exponent = -log_theta * Float32(2 * i) / Float32(axis_dim)
        out.append(fexp(exponent))
    return out^
