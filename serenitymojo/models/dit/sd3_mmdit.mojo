# SD3.5 MMDiT — full joint-block forward (Medium + Large).
#
# Extends the resident/pre-block gate with a streaming joint-block forward that
# port the Rust sd3_mmdit.rs architecture line-by-line:
#   - Per-block adaLN (SiLU(c) -> Linear -> chunk)
#   - Joint attention: concat ctx+x Q/K/V, SDPA, split back
#   - QK norm: RMSNorm (per Rust qk_norm_4d comment: sd3.5 uses RMSNorm for ln_q/ln_k)
#   - SD3.5 Medium dual attention for first 13 x_blocks (9*hidden adaLN)
#   - Last context_block is pre_only (2 mods, no proj/MLP)
#   - GELU(tanh) MLP: fc1 -> GELU -> fc2 (ops/activations.gelu)
#   - Final layer: LayerNorm(no_affine) -> adaLN -> Linear -> unpatchify
#
# Block weights are streamed via BlockLoader (one block at a time) so the
# full Medium checkpoint (~5GB) does not need to be resident simultaneously
# with the score/attention buffers.
#
# Weight prefix: "model.diffusion_model." is stripped at load time.
# Key layout (stripped):
#   joint_blocks.{i}.x_block.adaLN_modulation.1.weight/bias
#   joint_blocks.{i}.x_block.attn.qkv.weight/bias
#   joint_blocks.{i}.x_block.attn.proj.weight/bias
#   joint_blocks.{i}.x_block.attn.ln_q.weight  (no bias, RMSNorm)
#   joint_blocks.{i}.x_block.attn.ln_k.weight
#   joint_blocks.{i}.x_block.mlp.fc1.weight/bias
#   joint_blocks.{i}.x_block.mlp.fc2.weight/bias
#   (dual attn blocks also have attn2.qkv/proj/ln_q/ln_k)
#   joint_blocks.{i}.context_block.adaLN_modulation.1.weight/bias
#   joint_blocks.{i}.context_block.attn.qkv.weight/bias
#   joint_blocks.{i}.context_block.attn.proj.weight/bias  (not pre_only)
#   joint_blocks.{i}.context_block.attn.ln_q.weight
#   joint_blocks.{i}.context_block.attn.ln_k.weight
#   joint_blocks.{i}.context_block.mlp.fc1.weight/bias    (not pre_only)
#   joint_blocks.{i}.context_block.mlp.fc2.weight/bias    (not pre_only)

from std.gpu.host import DeviceContext
from std.gpu import global_idx
from std.math import sqrt
from std.memory import ArcPointer
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.models.dit.sd3_contract import (
    SD3_LARGE_CONTEXT_DIM,
    SD3_LARGE_DEPTH,
    SD3_LARGE_HEAD_DIM,
    SD3_LARGE_HIDDEN,
    SD3_LARGE_IMAGE_TOKENS,
    SD3_LARGE_LATENT_CHANNELS,
    SD3_LARGE_LATENT_H,
    SD3_LARGE_LATENT_W,
    SD3_LARGE_NUM_HEADS,
    SD3_LARGE_PATCH_GRID_H,
    SD3_LARGE_PATCH_GRID_W,
    SD3_LARGE_PATCH_SIZE,
    SD3_LARGE_PATCH_VECTOR_DIM,
    SD3_LARGE_POS_EMBED_GRID,
    SD3_LARGE_POOLED_DIM,
    SD3_LARGE_TEXT_TOKENS,
    SD3_LARGE_TIMESTEP_DIM,
    SD3_MEDIUM_CONTEXT_DIM,
    SD3_MEDIUM_DEPTH,
    SD3_MEDIUM_DUAL_ATTENTION_BLOCKS,
    SD3_MEDIUM_HEAD_DIM,
    SD3_MEDIUM_HIDDEN,
    SD3_MEDIUM_IMAGE_TOKENS,
    SD3_MEDIUM_LATENT_CHANNELS,
    SD3_MEDIUM_LATENT_H,
    SD3_MEDIUM_LATENT_W,
    SD3_MEDIUM_NUM_HEADS,
    SD3_MEDIUM_PATCH_GRID_H,
    SD3_MEDIUM_PATCH_GRID_W,
    SD3_MEDIUM_PATCH_SIZE,
    SD3_MEDIUM_PATCH_VECTOR_DIM,
    SD3_MEDIUM_POS_EMBED_GRID,
    SD3_MEDIUM_POOLED_DIM,
    SD3_MEDIUM_TEXT_TOKENS,
    SD3_MEDIUM_TIMESTEP_DIM,
    sd3_large_model_timestep,
    sd3_medium_model_timestep,
    validate_sd3_large_checkpoint_header,
    validate_sd3_medium_checkpoint_header,
)
from serenitymojo.offload.block_loader import BlockLoader, unload_block
from serenitymojo.ops.activations import gelu, silu
from serenitymojo.ops.attention import sdpa_nomask
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.embeddings import t_embedder
from serenitymojo.ops.layout import patchify, unpatchify
from serenitymojo.ops.linear import linear
from serenitymojo.ops.norm import layer_norm, rms_norm
from serenitymojo.ops.tensor_algebra import add, add_scalar, concat, mul, mul_scalar, reshape, slice, permute, sub
from serenitymojo.runtime.model_manifest import (
    ModelManifest,
    sd3_5_large_default_manifest,
    sd3_5_medium_default_manifest,
)
from serenitymojo.tensor import Tensor


def _shape1(a: Int) -> List[Int]:
    var sh = List[Int]()
    sh.append(a)
    return sh^


def _shape2(a: Int, b: Int) -> List[Int]:
    var sh = List[Int]()
    sh.append(a)
    sh.append(b)
    return sh^


def _shape3(a: Int, b: Int, c: Int) -> List[Int]:
    var sh = List[Int]()
    sh.append(a)
    sh.append(b)
    sh.append(c)
    return sh^


def _shape4(a: Int, b: Int, c: Int, d: Int) -> List[Int]:
    var sh = List[Int]()
    sh.append(a)
    sh.append(b)
    sh.append(c)
    sh.append(d)
    return sh^


def _sd3_key(name: String) -> String:
    return String("model.diffusion_model.") + name


def _load_weight_bf16(
    ref st: SafeTensors, name: String, ctx: DeviceContext
) raises -> Tensor:
    var full = _sd3_key(name)
    var info = st.tensor_info(full)
    var bytes = st.tensor_bytes(full)
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    return Tensor.from_view_as_bf16(tv, ctx)


def _clone(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    return x.clone(ctx)


def _copy_selected_pos_bytes(
    ref st: SafeTensors,
    name: String,
    max_grid: Int,
    crop_h: Int,
    crop_w: Int,
    hidden: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    var full = _sd3_key(name)
    var info = st.tensor_info(full)
    if len(info.shape) != 3 or info.shape[0] != 1 or info.shape[2] != hidden:
        raise Error("SD3 pos_embed shape mismatch")
    if info.shape[1] != max_grid * max_grid:
        raise Error("SD3 pos_embed grid mismatch")
    if crop_h <= 0 or crop_w <= 0 or crop_h > max_grid or crop_w > max_grid:
        raise Error("SD3 pos_embed crop out of range")

    var bytes = st.tensor_bytes(full)
    var n_tokens = crop_h * crop_w
    var n = n_tokens * hidden
    var out_nbytes = n * STDtype.BF16.byte_size()
    var host_out = ctx.enqueue_create_host_buffer[DType.uint8](out_nbytes)
    var outp = host_out.unsafe_ptr()
    var top = (max_grid - crop_h) // 2
    var left = (max_grid - crop_w) // 2

    var in_bsz = info.dtype.byte_size()
    var selected_in_nbytes = n * in_bsz
    var host_in = ctx.enqueue_create_host_buffer[DType.uint8](selected_in_nbytes)
    var inp = host_in.unsafe_ptr()

    for row in range(crop_h):
        for col in range(crop_w):
            var src_token = (top + row) * max_grid + (left + col)
            var dst_token = row * crop_w + col
            var src_byte = src_token * hidden * in_bsz
            var dst_byte = dst_token * hidden * in_bsz
            for j in range(hidden * in_bsz):
                inp[dst_byte + j] = bytes[src_byte + j]

    if info.dtype == STDtype.BF16:
        for i in range(out_nbytes):
            outp[i] = inp[i]
    else:
        var bp = host_out.unsafe_ptr().bitcast[BFloat16]()
        if info.dtype == STDtype.F16:
            var hp = host_in.unsafe_ptr().bitcast[Float16]()
            for i in range(n):
                bp[i] = hp[i].cast[DType.float32]().cast[DType.bfloat16]()
        elif info.dtype == STDtype.F32:
            var fp = host_in.unsafe_ptr().bitcast[Float32]()
            for i in range(n):
                bp[i] = fp[i].cast[DType.bfloat16]()
        else:
            raise Error("SD3 pos_embed source dtype must be BF16/F16/F32")

    var dev = ctx.enqueue_create_buffer[DType.uint8](out_nbytes)
    ctx.enqueue_copy(dst_buf=dev, src_buf=host_out)
    ctx.synchronize()
    return Tensor(dev^, _shape3(1, n_tokens, hidden), STDtype.BF16)


# ── Resident pre-block gate (patch embed, pos embed, embedders, final layer) ──

@fieldwise_init
struct SD3MMDiTPreBlockGate(Movable):
    var variant: String
    var hidden: Int
    var context_dim: Int
    var pooled_dim: Int
    var timestep_dim: Int
    var latent_channels: Int
    var latent_h: Int
    var latent_w: Int
    var patch_size: Int
    var patch_grid_h: Int
    var patch_grid_w: Int
    var patch_vector_dim: Int
    var timestep_scale_kind: Int  # 0=Large, 1=Medium. Both are sigma*1000 today.
    var pos_embed: Tensor
    var x_w: Tensor
    var x_b: Tensor
    var t_w0: Tensor
    var t_b0: Tensor
    var t_w2: Tensor
    var t_b2: Tensor
    var y_w0: Tensor
    var y_b0: Tensor
    var y_w2: Tensor
    var y_b2: Tensor
    var ctx_w: Tensor
    var ctx_b: Tensor
    var final_mod_w: Tensor
    var final_mod_b: Tensor
    var final_linear_w: Tensor
    var final_linear_b: Tensor

    @staticmethod
    def _load(
        manifest: ModelManifest,
        variant: String,
        hidden: Int,
        context_dim: Int,
        pooled_dim: Int,
        timestep_dim: Int,
        latent_channels: Int,
        latent_h: Int,
        latent_w: Int,
        patch_size: Int,
        patch_grid_h: Int,
        patch_grid_w: Int,
        patch_vector_dim: Int,
        pos_grid: Int,
        timestep_scale_kind: Int,
        ctx: DeviceContext,
    ) raises -> SD3MMDiTPreBlockGate:
        var st = SafeTensors.open(manifest.denoiser_path)
        var x_w_oihw = _load_weight_bf16(
            st, String("x_embedder.proj.weight"), ctx
        )
        var x_w = reshape(x_w_oihw, _shape2(hidden, patch_vector_dim), ctx)
        return SD3MMDiTPreBlockGate(
            variant,
            hidden,
            context_dim,
            pooled_dim,
            timestep_dim,
            latent_channels,
            latent_h,
            latent_w,
            patch_size,
            patch_grid_h,
            patch_grid_w,
            patch_vector_dim,
            timestep_scale_kind,
            _copy_selected_pos_bytes(
                st,
                String("pos_embed"),
                pos_grid,
                patch_grid_h,
                patch_grid_w,
                hidden,
                ctx,
            ),
            x_w^,
            _load_weight_bf16(st, String("x_embedder.proj.bias"), ctx),
            _load_weight_bf16(st, String("t_embedder.mlp.0.weight"), ctx),
            _load_weight_bf16(st, String("t_embedder.mlp.0.bias"), ctx),
            _load_weight_bf16(st, String("t_embedder.mlp.2.weight"), ctx),
            _load_weight_bf16(st, String("t_embedder.mlp.2.bias"), ctx),
            _load_weight_bf16(st, String("y_embedder.mlp.0.weight"), ctx),
            _load_weight_bf16(st, String("y_embedder.mlp.0.bias"), ctx),
            _load_weight_bf16(st, String("y_embedder.mlp.2.weight"), ctx),
            _load_weight_bf16(st, String("y_embedder.mlp.2.bias"), ctx),
            _load_weight_bf16(st, String("context_embedder.weight"), ctx),
            _load_weight_bf16(st, String("context_embedder.bias"), ctx),
            _load_weight_bf16(
                st, String("final_layer.adaLN_modulation.1.weight"), ctx
            ),
            _load_weight_bf16(
                st, String("final_layer.adaLN_modulation.1.bias"), ctx
            ),
            _load_weight_bf16(st, String("final_layer.linear.weight"), ctx),
            _load_weight_bf16(st, String("final_layer.linear.bias"), ctx),
        )

    @staticmethod
    def load_large_default(ctx: DeviceContext) raises -> SD3MMDiTPreBlockGate:
        var manifest = sd3_5_large_default_manifest()
        validate_sd3_large_checkpoint_header(manifest)
        return SD3MMDiTPreBlockGate._load(
            manifest^,
            String("sd3_5_large"),
            SD3_LARGE_HIDDEN,
            SD3_LARGE_CONTEXT_DIM,
            SD3_LARGE_POOLED_DIM,
            SD3_LARGE_TIMESTEP_DIM,
            SD3_LARGE_LATENT_CHANNELS,
            SD3_LARGE_LATENT_H,
            SD3_LARGE_LATENT_W,
            SD3_LARGE_PATCH_SIZE,
            SD3_LARGE_PATCH_GRID_H,
            SD3_LARGE_PATCH_GRID_W,
            SD3_LARGE_PATCH_VECTOR_DIM,
            SD3_LARGE_POS_EMBED_GRID,
            0,
            ctx,
        )

    @staticmethod
    def load_medium_default(ctx: DeviceContext) raises -> SD3MMDiTPreBlockGate:
        var manifest = sd3_5_medium_default_manifest()
        validate_sd3_medium_checkpoint_header(manifest)
        return SD3MMDiTPreBlockGate._load(
            manifest^,
            String("sd3_5_medium"),
            SD3_MEDIUM_HIDDEN,
            SD3_MEDIUM_CONTEXT_DIM,
            SD3_MEDIUM_POOLED_DIM,
            SD3_MEDIUM_TIMESTEP_DIM,
            SD3_MEDIUM_LATENT_CHANNELS,
            SD3_MEDIUM_LATENT_H,
            SD3_MEDIUM_LATENT_W,
            SD3_MEDIUM_PATCH_SIZE,
            SD3_MEDIUM_PATCH_GRID_H,
            SD3_MEDIUM_PATCH_GRID_W,
            SD3_MEDIUM_PATCH_VECTOR_DIM,
            SD3_MEDIUM_POS_EMBED_GRID,
            1,
            ctx,
        )

    def latent_patch_embed[H: Int, W: Int](
        self, latents_nchw: Tensor, ctx: DeviceContext
    ) raises -> Tensor:
        comptime assert H % 2 == 0 and W % 2 == 0, "SD3 latent H/W must divide by patch size 2"
        var sh = latents_nchw.shape()
        if (
            len(sh) != 4
            or sh[0] != 1
            or sh[1] != self.latent_channels
            or sh[2] != H
            or sh[3] != W
        ):
            raise Error("SD3 latent_patch_embed expects [1,16,H,W] NCHW")
        if latents_nchw.dtype() != STDtype.BF16:
            raise Error("SD3 latent_patch_embed expects BF16 latents")
        if H != self.latent_h or W != self.latent_w:
            raise Error("SD3 pre-block gate is loaded for the manifest latent grid")
        var patches = patchify(latents_nchw, self.patch_size, ctx)
        var embedded = linear(
            patches,
            self.x_w,
            Optional[Tensor](_clone(self.x_b, ctx)),
            ctx,
        )
        return add(embedded, self.pos_embed, ctx)

    def timestep_embed(self, sigma: Float32, ctx: DeviceContext) raises -> Tensor:
        var scaled: Float32
        if self.timestep_scale_kind == 0:
            scaled = sd3_large_model_timestep(sigma)
        else:
            scaled = sd3_medium_model_timestep(sigma)
        var values = List[Float32]()
        values.append(scaled)
        var t = Tensor.from_host(values^, _shape1(1), STDtype.F32, ctx)
        return t_embedder(
            t,
            self.timestep_dim,
            self.t_w0,
            Optional[Tensor](_clone(self.t_b0, ctx)),
            self.t_w2,
            Optional[Tensor](_clone(self.t_b2, ctx)),
            ctx,
            Float32(10000.0),
        )

    def pooled_embed(self, pooled: Tensor, ctx: DeviceContext) raises -> Tensor:
        var sh = pooled.shape()
        if len(sh) != 2 or sh[0] != 1 or sh[1] != self.pooled_dim:
            raise Error("SD3 pooled_embed expects [1,2048]")
        if pooled.dtype() != STDtype.BF16:
            raise Error("SD3 pooled_embed expects BF16 pooled projections")
        var h = linear(
            pooled,
            self.y_w0,
            Optional[Tensor](_clone(self.y_b0, ctx)),
            ctx,
        )
        h = silu(h, ctx)
        return linear(
            h,
            self.y_w2,
            Optional[Tensor](_clone(self.y_b2, ctx)),
            ctx,
        )

    def conditioning(
        self, sigma: Float32, pooled: Tensor, ctx: DeviceContext
    ) raises -> Tensor:
        var t = self.timestep_embed(sigma, ctx)
        var y = self.pooled_embed(pooled, ctx)
        return add(t, y, ctx)

    def context_embed[CTX: Int](
        self, encoder_hidden_states: Tensor, ctx: DeviceContext
    ) raises -> Tensor:
        var sh = encoder_hidden_states.shape()
        if (
            len(sh) != 3
            or sh[0] != 1
            or sh[1] != CTX
            or sh[2] != self.context_dim
        ):
            raise Error("SD3 context_embed expects [1,CTX,4096]")
        if encoder_hidden_states.dtype() != STDtype.BF16:
            raise Error("SD3 context_embed expects BF16 encoder states")
        return linear(
            encoder_hidden_states,
            self.ctx_w,
            Optional[Tensor](_clone(self.ctx_b, ctx)),
            ctx,
        )

    def _ones(self, d: Int, dtype: STDtype, ctx: DeviceContext) raises -> Tensor:
        var vals = List[Float32](capacity=d)
        for _ in range(d):
            vals.append(1.0)
        return Tensor.from_host(vals^, _shape1(d), dtype, ctx)

    def _zeros_vec(
        self, d: Int, dtype: STDtype, ctx: DeviceContext
    ) raises -> Tensor:
        var vals = List[Float32](capacity=d)
        for _ in range(d):
            vals.append(0.0)
        return Tensor.from_host(vals^, _shape1(d), dtype, ctx)

    def _layer_norm_no_affine(self, x: Tensor, ctx: DeviceContext) raises -> Tensor:
        var sh = x.shape()
        var d = sh[len(sh) - 1]
        var ones = self._ones(d, x.dtype(), ctx)
        var zeros = self._zeros_vec(d, x.dtype(), ctx)
        return layer_norm(x, ones, zeros, Float32(1e-6), ctx)

    def _modulate(
        self, x: Tensor, shift: Tensor, scale: Tensor, ctx: DeviceContext
    ) raises -> Tensor:
        var factor = add_scalar(scale, 1.0, ctx)
        var scaled = mul(x, factor, ctx)
        return add(scaled, shift, ctx)

    def final_layer_tokens(self, x_tokens: Tensor, c: Tensor, ctx: DeviceContext) raises -> Tensor:
        var xsh = x_tokens.shape()
        if len(xsh) != 3 or xsh[0] != 1 or xsh[2] != self.hidden:
            raise Error("SD3 final_layer_tokens expects [1,N,hidden]")
        var csh = c.shape()
        if len(csh) != 2 or csh[0] != 1 or csh[1] != self.hidden:
            raise Error("SD3 final_layer_tokens expects conditioning [1,hidden]")
        var x_norm = self._layer_norm_no_affine(x_tokens, ctx)
        var c_silu = silu(c, ctx)
        var mods = linear(
            c_silu,
            self.final_mod_w,
            Optional[Tensor](_clone(self.final_mod_b, ctx)),
            ctx,
        )
        var shift = slice(mods, 1, 0, self.hidden, ctx)
        var scale = slice(mods, 1, self.hidden, self.hidden, ctx)
        var x_mod = self._modulate(x_norm, shift, scale, ctx)
        return linear(
            x_mod,
            self.final_linear_w,
            Optional[Tensor](_clone(self.final_linear_b, ctx)),
            ctx,
        )

    def final_unpatchify[H: Int, W: Int](
        self, patch_tokens: Tensor, ctx: DeviceContext
    ) raises -> Tensor:
        """SD3-specific unpatchify using spatial-outer (ph,pw,c) convention.
        Matches Rust 'nhwpqc->nchpwq' einsum and HuggingFace SD3 implementation.
        f = (ph * patch + pw) * C + c  (spatial-outer, channels-inner)."""
        return _sd3_unpatchify(
            patch_tokens,
            self.latent_channels,
            H,
            W,
            self.patch_size,
            ctx,
        )


# ── SD3 spatial-outer unpatchify ──────────────────────────────────────────────
# SD3 uses 'nhwpqc->nchpwq' convention (spatial-outer, channels-inner):
#   f = (ph * patch + pw) * C + c
# This differs from the generic patchify/unpatchify in ops/layout.mojo which
# uses channels-outer (matching Conv2d input). For the final_layer output,
# the model weights encode spatial-outer vectors.

comptime _DYN1_SD3 = Layout.row_major(-1)
comptime _BLOCK_SD3 = 256


def _sd3_unpatchify_kernel_bf16(
    seq: LayoutTensor[DType.bfloat16, _DYN1_SD3, MutAnyOrigin],
    o: LayoutTensor[DType.bfloat16, _DYN1_SD3, MutAnyOrigin],
    B: Int, C: Int, H: Int, W: Int, p: Int, GH: Int, GW: Int,
):
    """f = (ph * p + pw) * C + c  (SD3 spatial-outer, channels-inner)."""
    var idx = Int(global_idx.x)
    var total = B * C * H * W
    if idx < total:
        var iw = idx % W
        var rem = idx // W
        var ih = rem % H
        rem = rem // H
        var c = rem % C
        var b = rem // C
        var gh = ih // p
        var ph = ih % p
        var gw = iw // p
        var pw = iw % p
        var L = GH * GW
        var F = C * p * p
        var n = gh * GW + gw
        var f = (ph * p + pw) * C + c   # spatial-outer
        var seq_off = (b * L + n) * F + f
        o[idx] = rebind[o.element_type](rebind[Scalar[DType.bfloat16]](seq[seq_off]))


def _sd3_unpatchify(
    seq: Tensor, channels: Int, height: Int, width: Int, patch: Int, ctx: DeviceContext
) raises -> Tensor:
    """SD3 unpatchify: [B, N, C*p*p] -> [B, C, H, W] using spatial-outer (ph,pw,c) order."""
    var sshape = seq.shape()
    if len(sshape) != 3:
        raise Error("_sd3_unpatchify: seq must be rank-3 [B, N, C*p*p]")
    var B = sshape[0]
    var L = sshape[1]
    var Fdim = sshape[2]
    var C = channels
    var H = height
    var W = width
    var p = patch
    if p <= 0 or H % p != 0 or W % p != 0:
        raise Error("_sd3_unpatchify: patch must divide H and W")
    var GH = H // p
    var GW = W // p
    if L != GH * GW:
        raise Error("_sd3_unpatchify: L != (H/p)*(W/p)")
    if Fdim != C * p * p:
        raise Error("_sd3_unpatchify: last dim != C*p*p")
    var total = B * C * H * W
    var dt = seq.dtype().to_mojo_dtype()
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](seq.nbytes())
    var rl = RuntimeLayout[_DYN1_SD3].row_major(IndexList[1](total))
    var grid = (total + _BLOCK_SD3 - 1) // _BLOCK_SD3
    if dt == DType.bfloat16:
        var S = LayoutTensor[DType.bfloat16, _DYN1_SD3, MutAnyOrigin](
            seq.buf.unsafe_ptr().bitcast[BFloat16](), rl
        )
        var O = LayoutTensor[DType.bfloat16, _DYN1_SD3, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), rl
        )
        ctx.enqueue_function[_sd3_unpatchify_kernel_bf16, _sd3_unpatchify_kernel_bf16](
            S, O, B, C, H, W, p, GH, GW, grid_dim=grid, block_dim=_BLOCK_SD3
        )
    else:
        raise Error("_sd3_unpatchify: only BF16 supported for SD3 output")
    ctx.synchronize()
    var out_sh = List[Int]()
    out_sh.append(B)
    out_sh.append(C)
    out_sh.append(H)
    out_sh.append(W)
    return Tensor(out_buf^, out_sh^, seq.dtype())


# ── Block weight helpers ───────────────────────────────────────────────────────

comptime _Block = Dict[String, ArcPointer[Tensor]]


def _ones_bf16(n: Int, ctx: DeviceContext) raises -> Tensor:
    var vals = List[Float32](capacity=n)
    for _ in range(n):
        vals.append(1.0)
    var sh = List[Int]()
    sh.append(n)
    return cast_tensor(Tensor.from_host(vals^, sh^, STDtype.F32, ctx), STDtype.BF16, ctx)


def _zeros_bf16(n: Int, ctx: DeviceContext) raises -> Tensor:
    var vals = List[Float32](capacity=n)
    for _ in range(n):
        vals.append(0.0)
    var sh = List[Int]()
    sh.append(n)
    return cast_tensor(Tensor.from_host(vals^, sh^, STDtype.F32, ctx), STDtype.BF16, ctx)


def _ones_dtype(n: Int, dtype: STDtype, ctx: DeviceContext) raises -> Tensor:
    var vals = List[Float32](capacity=n)
    for _ in range(n):
        vals.append(1.0)
    var sh = List[Int]()
    sh.append(n)
    return cast_tensor(Tensor.from_host(vals^, sh^, STDtype.F32, ctx), dtype, ctx)

def _zeros_dtype(n: Int, dtype: STDtype, ctx: DeviceContext) raises -> Tensor:
    var vals = List[Float32](capacity=n)
    for _ in range(n):
        vals.append(0.0)
    var sh = List[Int]()
    sh.append(n)
    return cast_tensor(Tensor.from_host(vals^, sh^, STDtype.F32, ctx), dtype, ctx)

def _layer_norm_no_affine(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    """LayerNorm without affine (elementwise_affine=False). Uses ones/zeros matching x dtype."""
    var sh = x.shape()
    var d = sh[len(sh) - 1]
    var ones = _ones_dtype(d, x.dtype(), ctx)
    var zeros = _zeros_dtype(d, x.dtype(), ctx)
    return layer_norm(x, ones, zeros, Float32(1e-6), ctx)


def _modulate(x: Tensor, shift: Tensor, scale: Tensor, ctx: DeviceContext) raises -> Tensor:
    """adaLN modulate: x * (1 + scale) + shift. Broadcast over [B,N,D]."""
    var factor = add_scalar(scale, 1.0, ctx)
    var scaled = mul(x, factor, ctx)
    return add(scaled, shift, ctx)


# ── QK RMSNorm (per head) ─────────────────────────────────────────────────────
# q/k: [B, S_joint, H, Dh] stored as [B*S_joint, H*Dh] logically.
# We flatten the last two dims -> [rows, Dh], apply rms_norm, reshape back.
# rms_norm weight: [Dh] (ln_q.weight or ln_k.weight, no bias in SD3).

def _qk_rms_norm(x: Tensor, weight: Tensor, ctx: DeviceContext) raises -> Tensor:
    """RMSNorm over the last dim (head_dim). x is [B, S, H, Dh] in BSHD layout."""
    var sh = x.shape()
    var b = sh[0]
    var s = sh[1]
    var h = sh[2]
    var d = sh[3]
    # flatten [B, S, H, Dh] -> [B*S*H, Dh] for rms_norm
    var flat_sh = List[Int]()
    flat_sh.append(b * s * h)
    flat_sh.append(d)
    var flat = reshape(x, flat_sh^, ctx)
    # Cast weight to match x dtype (weight should already be BF16)
    var normed = rms_norm(flat, weight, Float32(1e-6), ctx)
    # Reshape back to [B, S, H, Dh]
    var out_sh = List[Int]()
    out_sh.append(b)
    out_sh.append(s)
    out_sh.append(h)
    out_sh.append(d)
    return reshape(normed, out_sh^, ctx)




def _w_bf16(blk: _Block, key: String, ctx: DeviceContext) raises -> Tensor:
    """Get a block weight cast to BF16 (F16 block weights -> BF16 for BF16 compute consistency)."""
    if blk[key][].dtype() == STDtype.BF16:
        return _clone(blk[key][], ctx)
    return cast_tensor(blk[key][], STDtype.BF16, ctx)


def _qkv_project(
    x: Tensor,
    blk: _Block,
    qkv_w_key: String,
    qkv_b_key: String,
    ln_q_key: String,
    ln_k_key: String,
    num_heads: Int,
    head_dim: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    """Project x to QKV with QK-RMSNorm. Returns [B, N, 3, H, Dh] (5D).
    Caller slices dim=2 to get q, k, v each [B, N, 1, H, Dh] then reshapes to [B, N, H, Dh]."""
    var xsh = x.shape()
    var b = xsh[0]
    var n = xsh[1]
    var qkv_w = _w_bf16(blk, qkv_w_key, ctx)
    var qkv_b = _w_bf16(blk, qkv_b_key, ctx)
    var qkv_flat = linear(x, qkv_w, Optional[Tensor](qkv_b^), ctx)
    # Reshape [B, N, 3*H*Dh] -> [B, N, 3, H, Dh]
    var qkv_sh5 = List[Int]()
    qkv_sh5.append(b)
    qkv_sh5.append(n)
    qkv_sh5.append(3)
    qkv_sh5.append(num_heads)
    qkv_sh5.append(head_dim)
    var qkv5 = reshape(qkv_flat, qkv_sh5^, ctx)
    # QK RMSNorm: slice out q and k, normalize, put back
    var head_sh = List[Int]()
    head_sh.append(b)
    head_sh.append(n)
    head_sh.append(num_heads)
    head_sh.append(head_dim)
    var q_slice = slice(qkv5, 2, 0, 1, ctx)
    var k_slice = slice(qkv5, 2, 1, 1, ctx)
    var v_slice = slice(qkv5, 2, 2, 1, ctx)
    var q = reshape(q_slice, head_sh.copy(), ctx)
    var k = reshape(k_slice, head_sh.copy(), ctx)
    var v = reshape(v_slice, head_sh^, ctx)
    var ln_q_w = _w_bf16(blk, ln_q_key, ctx)
    var ln_k_w = _w_bf16(blk, ln_k_key, ctx)
    q = _qk_rms_norm(q, ln_q_w, ctx)
    k = _qk_rms_norm(k, ln_k_w, ctx)
    # Re-concat as [B, N, 3*H, Dh] then reshape to [B, N, 3, H, Dh]
    # concat on dim=2 (the H dim) gives [B, N, 3*H, Dh]
    var qkv_cat = concat(2, ctx, q, k, v)  # [B, N, 3*H, Dh]
    var out_sh = List[Int]()
    out_sh.append(b)
    out_sh.append(n)
    out_sh.append(3)
    out_sh.append(num_heads)
    out_sh.append(head_dim)
    return reshape(qkv_cat, out_sh^, ctx)  # [B, N, 3, H, Dh]


# ── GELU MLP (fc1 -> GELU -> fc2) ────────────────────────────────────────────

def _gelu_mlp(
    x: Tensor,
    blk: _Block,
    prefix: String,
    ctx: DeviceContext,
) raises -> Tensor:
    var fc1_w = prefix + String(".mlp.fc1.weight")
    var fc1_b = prefix + String(".mlp.fc1.bias")
    var fc2_w = prefix + String(".mlp.fc2.weight")
    var fc2_b = prefix + String(".mlp.fc2.bias")
    var fc1_wt = _w_bf16(blk, fc1_w, ctx)
    var fc1_bt = _w_bf16(blk, fc1_b, ctx)
    var h = linear(x, fc1_wt, Optional[Tensor](fc1_bt^), ctx)
    h = gelu(h, ctx)
    var fc2_wt = _w_bf16(blk, fc2_w, ctx)
    var fc2_bt = _w_bf16(blk, fc2_b, ctx)
    return linear(h, fc2_wt, Optional[Tensor](fc2_bt^), ctx)


# ── Full joint block forward ──────────────────────────────────────────────────
# S_JOINT = N_CTX + N_IMG (comptime for sdpa_nomask).
# Updates context and x in-place (inout). For the last (pre_only) block,
# context is left unchanged.

def _sd3_joint_block[
    B: Int,
    S_JOINT: Int,
    N_CTX: Int,
    N_IMG: Int,
    H: Int,
    Dh: Int,
](
    mut context: Tensor,  # [B, N_CTX, hidden] BF16 — updated in place (except last block)
    mut x: Tensor,        # [B, N_IMG, hidden] BF16 — updated in place
    c: Tensor,              # [B, hidden] BF16 (timestep + pooled conditioning)
    blk: _Block,
    block_idx: Int,
    is_last: Bool,        # True for the final pre_only context block
    num_dual_blocks: Int, # first num_dual_blocks x_blocks have attn2
    hidden: Int,
    ctx: DeviceContext,
) raises:
    # Block keys in the safetensors file include the "model.diffusion_model." prefix.
    var pfx = String("model.diffusion_model.")
    var x_pfx = pfx + String("joint_blocks.") + String(block_idx) + String(".x_block")
    var ctx_pfx = pfx + String("joint_blocks.") + String(block_idx) + String(".context_block")

    # Block weights are stored as F16, but we keep computation in BF16 (matching Rust
    # which casts to BF16 at load time). Build a BF16-cast view of the block.
    # Use cast_tensor per-key at point of use to avoid building a second dict.
    # context, x, c are already BF16 (from preblock gate or previous block cast-back).
    # All F16 weights are cast to BF16 inline via _cast_w() below.

    # ── Context adaLN modulation ─────────────────────────────────────────────
    var c_silu = silu(c, ctx)
    var ctx_ada_w = ctx_pfx + String(".adaLN_modulation.1.weight")
    var ctx_ada_b = ctx_pfx + String(".adaLN_modulation.1.bias")
    var ctx_ada_wt = _w_bf16(blk, ctx_ada_w, ctx)
    var ctx_ada_bt = _w_bf16(blk, ctx_ada_b, ctx)
    var ctx_mods_raw = linear(c_silu, ctx_ada_wt, Optional[Tensor](ctx_ada_bt^), ctx)
    # Context norm (no affine)
    var ctx_norm = _layer_norm_no_affine(context, ctx)

    var ctx_q: Tensor
    var ctx_k: Tensor
    var ctx_v: Tensor
    var ctx_gate_msa: Optional[Tensor]
    var ctx_shift_mlp: Optional[Tensor]
    var ctx_scale_mlp: Optional[Tensor]
    var ctx_gate_mlp: Optional[Tensor]

    var ctx_qkv5: Tensor  # [B, N_CTX, 3, H, Dh]
    if is_last:
        # pre_only: 2 mods (shift_msa, scale_msa)
        var ctx_shift = slice(ctx_mods_raw, 1, 0, hidden, ctx)
        var ctx_scale = slice(ctx_mods_raw, 1, hidden, hidden, ctx)
        var ctx_mod = _modulate(ctx_norm, ctx_shift, ctx_scale, ctx)
        ctx_qkv5 = _qkv_project(ctx_mod, blk,
            ctx_pfx + String(".attn.qkv.weight"),
            ctx_pfx + String(".attn.qkv.bias"),
            ctx_pfx + String(".attn.ln_q.weight"),
            ctx_pfx + String(".attn.ln_k.weight"),
            H, Dh, ctx)
        ctx_gate_msa = None
        ctx_shift_mlp = None
        ctx_scale_mlp = None
        ctx_gate_mlp = None
    else:
        # 6 mods: shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
        var ctx_shift_msa = slice(ctx_mods_raw, 1, 0, hidden, ctx)
        var ctx_scale_msa = slice(ctx_mods_raw, 1, hidden, hidden, ctx)
        var ctx_gate = slice(ctx_mods_raw, 1, 2 * hidden, hidden, ctx)
        var ctx_smlp = slice(ctx_mods_raw, 1, 3 * hidden, hidden, ctx)
        var ctx_scmlp = slice(ctx_mods_raw, 1, 4 * hidden, hidden, ctx)
        var ctx_gmlp = slice(ctx_mods_raw, 1, 5 * hidden, hidden, ctx)
        var ctx_mod = _modulate(ctx_norm, ctx_shift_msa, ctx_scale_msa, ctx)
        ctx_qkv5 = _qkv_project(ctx_mod, blk,
            ctx_pfx + String(".attn.qkv.weight"),
            ctx_pfx + String(".attn.qkv.bias"),
            ctx_pfx + String(".attn.ln_q.weight"),
            ctx_pfx + String(".attn.ln_k.weight"),
            H, Dh, ctx)
        ctx_gate_msa = ctx_gate^
        ctx_shift_mlp = ctx_smlp^
        ctx_scale_mlp = ctx_scmlp^
        ctx_gate_mlp = ctx_gmlp^
    # Slice ctx QKV: [B, N_CTX, 3, H, Dh] -> q,k,v each [B, N_CTX, 1, H, Dh] -> reshape [B, N_CTX, H, Dh]
    var ctx_head_sh = List[Int]()
    ctx_head_sh.append(B)
    ctx_head_sh.append(N_CTX)
    ctx_head_sh.append(H)
    ctx_head_sh.append(Dh)
    ctx_q = reshape(slice(ctx_qkv5, 2, 0, 1, ctx), ctx_head_sh.copy(), ctx)
    ctx_k = reshape(slice(ctx_qkv5, 2, 1, 1, ctx), ctx_head_sh.copy(), ctx)
    ctx_v = reshape(slice(ctx_qkv5, 2, 2, 1, ctx), ctx_head_sh^, ctx)

    # ── X adaLN modulation ───────────────────────────────────────────────────
    # Re-use c_silu (already computed above; c is not mutated)
    var c_silu2 = silu(c, ctx)
    var x_ada_w = x_pfx + String(".adaLN_modulation.1.weight")
    var x_ada_b = x_pfx + String(".adaLN_modulation.1.bias")
    var x_ada_wt = _w_bf16(blk, x_ada_w, ctx)
    var x_ada_bt = _w_bf16(blk, x_ada_b, ctx)
    var x_mods_raw = linear(c_silu2, x_ada_wt, Optional[Tensor](x_ada_bt^), ctx)
    var x_norm = _layer_norm_no_affine(x, ctx)
    # Detect dual attention by adaLN output width
    var x_mods_sh = x_mods_raw.shape()
    var ada_width = x_mods_sh[len(x_mods_sh) - 1]
    var block_has_dual = (ada_width // hidden) == 9

    var x_shift_msa = slice(x_mods_raw, 1, 0, hidden, ctx)
    var x_scale_msa = slice(x_mods_raw, 1, hidden, hidden, ctx)
    var x_gate_msa  = slice(x_mods_raw, 1, 2 * hidden, hidden, ctx)
    var x_shift_mlp = slice(x_mods_raw, 1, 3 * hidden, hidden, ctx)
    var x_scale_mlp = slice(x_mods_raw, 1, 4 * hidden, hidden, ctx)
    var x_gate_mlp  = slice(x_mods_raw, 1, 5 * hidden, hidden, ctx)

    var x_mod = _modulate(x_norm, x_shift_msa, x_scale_msa, ctx)
    var x_qkv5 = _qkv_project(x_mod, blk,
        x_pfx + String(".attn.qkv.weight"),
        x_pfx + String(".attn.qkv.bias"),
        x_pfx + String(".attn.ln_q.weight"),
        x_pfx + String(".attn.ln_k.weight"),
        H, Dh, ctx)
    var x_head_sh = List[Int]()
    x_head_sh.append(B)
    x_head_sh.append(N_IMG)
    x_head_sh.append(H)
    x_head_sh.append(Dh)
    var x_q = reshape(slice(x_qkv5, 2, 0, 1, ctx), x_head_sh.copy(), ctx)
    var x_k = reshape(slice(x_qkv5, 2, 1, 1, ctx), x_head_sh.copy(), ctx)
    var x_v = reshape(slice(x_qkv5, 2, 2, 1, ctx), x_head_sh^, ctx)

    # Optional second modulated input for dual attention
    var x_mod2: Optional[Tensor] = None
    var x_gate_msa2: Optional[Tensor] = None
    if block_has_dual:
        var x_shift_msa2 = slice(x_mods_raw, 1, 6 * hidden, hidden, ctx)
        var x_scale_msa2 = slice(x_mods_raw, 1, 7 * hidden, hidden, ctx)
        var x_gate2 = slice(x_mods_raw, 1, 8 * hidden, hidden, ctx)
        x_mod2 = _modulate(x_norm, x_shift_msa2, x_scale_msa2, ctx)
        x_gate_msa2 = x_gate2^

    # ── Joint attention: concat ctx + x along sequence dim ──────────────────
    # q/k/v: concat along S dim (dim 1 of BSHD layout)
    var joint_q = concat(1, ctx, ctx_q, x_q)
    var joint_k = concat(1, ctx, ctx_k, x_k)
    var joint_v = concat(1, ctx, ctx_v, x_v)

    # SDPA (math-mode; S_JOINT, H=24, Dh=64)
    comptime SCALE = Float32(1.0) / Float32(8.0)  # 1/sqrt(64)
    var attn_out = sdpa_nomask[B, S_JOINT, H, Dh](
        joint_q, joint_k, joint_v, SCALE, ctx
    )

    # Split back: context = attn_out[:, :N_CTX, :, :], x = attn_out[:, N_CTX:, :, :]
    var ctx_attn_bshd = slice(attn_out, 1, 0, N_CTX, ctx)       # [B, N_CTX, H, Dh]
    var x_attn_bshd   = slice(attn_out, 1, N_CTX, N_IMG, ctx)   # [B, N_IMG, H, Dh]

    # Reshape BSHD [B, N, H, Dh] -> [B, N, H*Dh]
    var ctx_attn_sh = List[Int]()
    ctx_attn_sh.append(B)
    ctx_attn_sh.append(N_CTX)
    ctx_attn_sh.append(H * Dh)
    var x_attn_sh = List[Int]()
    x_attn_sh.append(B)
    x_attn_sh.append(N_IMG)
    x_attn_sh.append(H * Dh)
    var ctx_attn = reshape(ctx_attn_bshd, ctx_attn_sh^, ctx)
    var x_attn   = reshape(x_attn_bshd, x_attn_sh^, ctx)

    # ── Context post-attention ───────────────────────────────────────────────
    # For pre_only (is_last), no context post-attention; context remains unchanged.
    if not is_last:
        var ctx_proj_w = ctx_pfx + String(".attn.proj.weight")
        var ctx_proj_b = ctx_pfx + String(".attn.proj.bias")
        var ctx_proj_wt = _w_bf16(blk, ctx_proj_w, ctx)
        var ctx_proj_bt = _w_bf16(blk, ctx_proj_b, ctx)
        var ctx_proj = linear(ctx_attn, ctx_proj_wt, Optional[Tensor](ctx_proj_bt^), ctx)
        # gate_msa is [B, hidden]; unsqueeze to [B, 1, hidden] for broadcast
        var gate_sh = List[Int]()
        gate_sh.append(B)
        gate_sh.append(1)
        gate_sh.append(hidden)
        var ctx_gate_3d = reshape(ctx_gate_msa.value(), gate_sh^, ctx)
        var ctx_gated = mul(ctx_gate_3d, ctx_proj, ctx)
        var ctx_res = add(context, ctx_gated, ctx)
        # MLP
        var ctx_norm2 = _layer_norm_no_affine(ctx_res, ctx)
        var ctx_mlp_in = _modulate(ctx_norm2, ctx_shift_mlp.value(), ctx_scale_mlp.value(), ctx)
        var ctx_mlp = _gelu_mlp(ctx_mlp_in, blk, ctx_pfx, ctx)
        var gate_mlp_sh = List[Int]()
        gate_mlp_sh.append(B)
        gate_mlp_sh.append(1)
        gate_mlp_sh.append(hidden)
        var ctx_gmlp_3d = reshape(ctx_gate_mlp.value(), gate_mlp_sh^, ctx)
        var ctx_mlp_gated = mul(ctx_gmlp_3d, ctx_mlp, ctx)
        # Update context inout
        context = add(ctx_res, ctx_mlp_gated, ctx)
    # If is_last (pre_only): context not updated

    # ── X post-attention ─────────────────────────────────────────────────────
    var x_proj_w = x_pfx + String(".attn.proj.weight")
    var x_proj_b = x_pfx + String(".attn.proj.bias")
    var x_proj_wt = _w_bf16(blk, x_proj_w, ctx)
    var x_proj_bt = _w_bf16(blk, x_proj_b, ctx)
    var x_proj = linear(x_attn, x_proj_wt, Optional[Tensor](x_proj_bt^), ctx)
    var xg_sh = List[Int]()
    xg_sh.append(B)
    xg_sh.append(1)
    xg_sh.append(hidden)
    var x_gate_3d = reshape(x_gate_msa, xg_sh^, ctx)
    var x_gated = mul(x_gate_3d, x_proj, ctx)
    var x_out = add(x, x_gated, ctx)

    # ── X dual attention (attn2, SD3.5 Medium blocks 0-12 only) ─────────────
    if block_has_dual:
        var x2_qkv5 = _qkv_project(x_mod2.value(), blk,
            x_pfx + String(".attn2.qkv.weight"),
            x_pfx + String(".attn2.qkv.bias"),
            x_pfx + String(".attn2.ln_q.weight"),
            x_pfx + String(".attn2.ln_k.weight"),
            H, Dh, ctx)
        var x2_head_sh = List[Int]()
        x2_head_sh.append(B)
        x2_head_sh.append(N_IMG)
        x2_head_sh.append(H)
        x2_head_sh.append(Dh)
        var x2_q = reshape(slice(x2_qkv5, 2, 0, 1, ctx), x2_head_sh.copy(), ctx)
        var x2_k = reshape(slice(x2_qkv5, 2, 1, 1, ctx), x2_head_sh.copy(), ctx)
        var x2_v = reshape(slice(x2_qkv5, 2, 2, 1, ctx), x2_head_sh^, ctx)
        # Self-attention on x only (no context concatenation)
        var attn2_out_bshd = sdpa_nomask[B, N_IMG, H, Dh](
            x2_q, x2_k, x2_v, SCALE, ctx
        )
        var attn2_sh = List[Int]()
        attn2_sh.append(B)
        attn2_sh.append(N_IMG)
        attn2_sh.append(H * Dh)
        var attn2_flat = reshape(attn2_out_bshd, attn2_sh^, ctx)
        var a2_proj_w = x_pfx + String(".attn2.proj.weight")
        var a2_proj_b = x_pfx + String(".attn2.proj.bias")
        var a2_proj_wt = _w_bf16(blk, a2_proj_w, ctx)
        var a2_proj_bt = _w_bf16(blk, a2_proj_b, ctx)
        var attn2_proj = linear(attn2_flat, a2_proj_wt, Optional[Tensor](a2_proj_bt^), ctx)
        var xg2_sh = List[Int]()
        xg2_sh.append(B)
        xg2_sh.append(1)
        xg2_sh.append(hidden)
        var x_gate2_3d = reshape(x_gate_msa2.value(), xg2_sh^, ctx)
        var attn2_gated = mul(x_gate2_3d, attn2_proj, ctx)
        x_out = add(x_out, attn2_gated, ctx)

    # ── X MLP ────────────────────────────────────────────────────────────────
    var x_norm2 = _layer_norm_no_affine(x_out, ctx)
    var x_mlp_in = _modulate(x_norm2, x_shift_mlp, x_scale_mlp, ctx)
    var x_mlp = _gelu_mlp(x_mlp_in, blk, x_pfx, ctx)
    var xgmlp_sh = List[Int]()
    xgmlp_sh.append(B)
    xgmlp_sh.append(1)
    xgmlp_sh.append(hidden)
    var x_gmlp_3d = reshape(x_gate_mlp, xgmlp_sh^, ctx)
    var x_mlp_gated = mul(x_gmlp_3d, x_mlp, ctx)
    x_out = add(x_out, x_mlp_gated, ctx)
    # Update x inout
    x = x_out^
