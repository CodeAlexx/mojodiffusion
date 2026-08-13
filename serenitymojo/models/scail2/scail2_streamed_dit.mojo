# SCAIL-2 14B forward over the persistent per-block E4M3 stream.
#
# The caller supplies the official prepared channel tensors (video/ref/pose 20,
# masks 28), raw UMT5 rows, and 257x1280 visual features. Shared embeddings stay
# BF16-resident; exactly one 32-tensor transformer block is dequantized at once.

from std.collections import Dict, List, Optional
from max.gpu.host import DeviceContext
from std.memory import ArcPointer
from std.time import perf_counter

from serenitymojo.io.dtype import STDtype
from serenitymojo.models.dit.wan22_dit import (
    _chunk2,
    _pad_context,
    _unsqueeze2,
    wan22_mod_pre,
)
from serenitymojo.models.scail2.scail2_block import scail2_block_forward
from serenitymojo.models.scail2.scail2_config import Scail2Config
from serenitymojo.models.scail2.scail2_fp8_stream import Scail2A14BFP8Stream
from serenitymojo.models.scail2.scail2_rope import (
    Scail2SequencePlan,
    build_scail2_rope_tables,
)
from serenitymojo.ops.activations import gelu, gelu_exact, silu
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.embeddings import timestep_embedding
from serenitymojo.ops.linear import linear
from serenitymojo.ops.norm import layer_norm
from serenitymojo.ops.patchify3d import patchify3d, unpatchify3d
from serenitymojo.ops.tensor_algebra import add, concat, full_device, reshape, slice
from serenitymojo.tensor import Tensor


comptime TArc = ArcPointer[Tensor]


def scail2_image_projection_activation(
    x: Tensor, ctx: DeviceContext
) raises -> Tensor:
    """Creator MLPProj activation: bare torch nn.GELU(), exact erf form."""
    return gelu_exact(x, ctx)


struct Scail2PredictionPair(Movable):
    var cond: Tensor
    var uncond: Tensor

    def __init__(out self, var cond: Tensor, var uncond: Tensor):
        self.cond = cond^
        self.uncond = uncond^


struct Scail2StreamedDiT(Movable):
    var weights: Dict[String, TArc]
    var stream: Scail2A14BFP8Stream
    var config: Scail2Config

    @staticmethod
    def open(cache_dir: String, ctx: DeviceContext) raises -> Scail2StreamedDiT:
        var stream = Scail2A14BFP8Stream.open(cache_dir)
        var shared = stream.load_shared(ctx)
        return Scail2StreamedDiT(shared^, stream^)

    def __init__(
        out self, var weights: Dict[String, TArc], var stream: Scail2A14BFP8Stream
    ):
        self.weights = weights^
        self.stream = stream^
        self.config = Scail2Config.scail2_14b()

    def _w(self, name: String) raises -> ref [self.weights[String("")]] Tensor:
        if name not in self.weights:
            raise Error(String("SCAIL-2 shared cache missing tensor: ") + name)
        return self.weights[name][]

    def _patch(
        self,
        x: Tensor,
        channels: Int,
        weight_name: String,
        bias_name: String,
        ctx: DeviceContext,
    ) raises -> Tensor:
        var cfg = self.config
        # PyTorch autocast presents Conv3d with parameter dtype. Make that cast
        # explicit before the shared patchify+linear lowering.
        var x_weight_dtype = cast_tensor(x, self._w(weight_name).dtype(), ctx)
        var patches = patchify3d(
            x_weight_dtype, cfg.patch_f, cfg.patch_h, cfg.patch_w, ctx
        )
        var weight_shape = List[Int]()
        weight_shape.append(cfg.dim)
        weight_shape.append(channels * cfg.patch_f * cfg.patch_h * cfg.patch_w)
        var weight = reshape(self._w(weight_name), weight_shape^, ctx)
        return linear(
            patches, weight, Optional(self._w(bias_name).clone(ctx)), ctx
        )

    def _embed_base_sequence[
        FT: Int, GH: Int, GW: Int, S: Int
    ](
        self,
        video20: Tensor,
        primary_ref20: Tensor,
        pose20: Tensor,
        primary_video_masks28: Tensor,
        driving_masks28: Tensor,
        ctx: DeviceContext,
    ) raises -> Tensor:
        var cfg = self.config
        var expected = (1 + FT) * GH * GW + FT * (GH // 2) * (GW // 2)
        if S != expected:
            raise Error("SCAIL-2 base sequence specialization mismatch")
        var ref_video = concat(1, ctx, primary_ref20, video20)
        var main = self._patch(
            ref_video, cfg.in_dim, "patch_embedding.weight",
            "patch_embedding.bias", ctx,
        )
        var main_mask = self._patch(
            primary_video_masks28, cfg.mask_dim, "patch_embedding_mask.weight",
            "patch_embedding_mask.bias", ctx,
        )
        main = add(main, main_mask, ctx)
        var pose = self._patch(
            pose20, cfg.in_dim, "patch_embedding_pose.weight",
            "patch_embedding_pose.bias", ctx,
        )
        var driving = self._patch(
            driving_masks28, cfg.mask_dim, "patch_embedding_mask.weight",
            "patch_embedding_mask.bias", ctx,
        )
        pose = add(pose, driving, ctx)
        var sequence2d = concat(0, ctx, main, pose)
        var shape = List[Int]()
        shape.append(1)
        shape.append(S)
        shape.append(cfg.dim)
        return reshape(sequence2d, shape^, ctx)

    def _embed_sequence_with_additional[
        FT: Int, GH: Int, GW: Int, AR: Int, S: Int
    ](
        self,
        video20: Tensor,
        primary_ref20: Tensor,
        pose20: Tensor,
        primary_video_masks28: Tensor,
        driving_masks28: Tensor,
        additional_refs20: Tensor,
        additional_ref_masks28: Tensor,
        ctx: DeviceContext,
    ) raises -> Tensor:
        if AR <= 0:
            raise Error("SCAIL-2 additional-reference count must be positive")
        var cfg = self.config
        var expected = (
            AR * GH * GW + (1 + FT) * GH * GW
            + FT * (GH // 2) * (GW // 2)
        )
        if S != expected:
            raise Error("SCAIL-2 additional-reference sequence mismatch")
        var additional = self._patch(
            additional_refs20, cfg.in_dim, "patch_embedding.weight",
            "patch_embedding.bias", ctx,
        )
        additional = add(
            additional,
            self._patch(
                additional_ref_masks28, cfg.mask_dim,
                "patch_embedding_mask.weight", "patch_embedding_mask.bias", ctx,
            ),
            ctx,
        )
        var base = self._embed_base_sequence[
            FT, GH, GW,
            (1 + FT) * GH * GW + FT * (GH // 2) * (GW // 2),
        ](
            video20, primary_ref20, pose20, primary_video_masks28,
            driving_masks28, ctx,
        )
        var base2_shape = List[Int]()
        base2_shape.append(
            (1 + FT) * GH * GW + FT * (GH // 2) * (GW // 2)
        )
        base2_shape.append(cfg.dim)
        var base2 = reshape(base, base2_shape^, ctx)
        var sequence2d = concat(0, ctx, additional, base2)
        var shape = List[Int]()
        shape.append(1)
        shape.append(S)
        shape.append(cfg.dim)
        return reshape(sequence2d, shape^, ctx)

    def _time(
        self, timestep: Float32, dtype: STDtype, ctx: DeviceContext
    ) raises -> Tuple[Tensor, Tensor]:
        var cfg = self.config
        var t_shape = List[Int]()
        t_shape.append(1)
        var t = full_device(t_shape^, timestep, STDtype.F32, ctx)
        var sinusoid = timestep_embedding(
            t, cfg.freq_dim, ctx, cfg.rope_theta, dtype
        )
        var e = linear(
            sinusoid, self._w("time_embedding.0.weight"),
            Optional(self._w("time_embedding.0.bias").clone(ctx)), ctx,
        )
        e = silu(e, ctx)
        e = linear(
            e, self._w("time_embedding.2.weight"),
            Optional(self._w("time_embedding.2.bias").clone(ctx)), ctx,
        )
        var e0 = linear(
            silu(e, ctx), self._w("time_projection.1.weight"),
            Optional(self._w("time_projection.1.bias").clone(ctx)), ctx,
        )
        var e0_shape = List[Int]()
        e0_shape.append(1)
        e0_shape.append(1)
        e0_shape.append(6)
        e0_shape.append(cfg.dim)
        e0 = reshape(cast_tensor(e0, STDtype.F32, ctx), e0_shape^, ctx)
        var head_shape = List[Int]()
        head_shape.append(1)
        head_shape.append(1)
        head_shape.append(cfg.dim)
        var e_head = reshape(cast_tensor(e, STDtype.F32, ctx), head_shape^, ctx)
        return (e0^, e_head^)

    def _text[TXT: Int, CTXL: Int](
        self, context: Tensor, dtype: STDtype, ctx: DeviceContext
    ) raises -> Tensor:
        var cfg = self.config
        var padded = _pad_context(context, CTXL, TXT, cfg.text_dim, dtype, ctx)
        var text = linear(
            padded, self._w("text_embedding.0.weight"),
            Optional(self._w("text_embedding.0.bias").clone(ctx)), ctx,
        )
        text = gelu(text, ctx)
        return linear(
            text, self._w("text_embedding.2.weight"),
            Optional(self._w("text_embedding.2.bias").clone(ctx)), ctx,
        )

    def _image[IMG: Int](
        self, clip_features: Tensor, ctx: DeviceContext
    ) raises -> Tensor:
        var cfg = self.config
        if IMG != cfg.image_len:
            raise Error("SCAIL-2 visual encoder must return 257 tokens")
        var image_input = cast_tensor(
            clip_features, self._w("img_emb.proj.0.weight").dtype(), ctx
        )
        var x = layer_norm(
            image_input, self._w("img_emb.proj.0.weight"),
            self._w("img_emb.proj.0.bias"), cfg.image_norm_eps, ctx,
        )
        x = linear(
            x, self._w("img_emb.proj.1.weight"),
            Optional(self._w("img_emb.proj.1.bias").clone(ctx)), ctx,
        )
        x = scail2_image_projection_activation(x, ctx)
        x = linear(
            x, self._w("img_emb.proj.3.weight"),
            Optional(self._w("img_emb.proj.3.bias").clone(ctx)), ctx,
        )
        return layer_norm(
            x, self._w("img_emb.proj.4.weight"),
            self._w("img_emb.proj.4.bias"), cfg.image_norm_eps, ctx,
        )

    def _head_video[
        FT: Int, GH: Int, GW: Int, AR: Int, S: Int
    ](
        self, img: Tensor, e_head: Tensor, dtype: STDtype, ctx: DeviceContext
    ) raises -> Tensor:
        var cfg = self.config
        var video_offset = (AR + 1) * GH * GW
        var video_rows = FT * GH * GW
        var video = slice(img, 1, video_offset, video_rows, ctx)
        var head_mod = cast_tensor(self._w("head.modulation"), STDtype.F32, ctx)
        var head_e = add(head_mod, _unsqueeze2(e_head, 1, cfg.dim, ctx), ctx)
        var shift = _chunk2(head_e, 0, 1, cfg.dim, ctx)
        var scale = _chunk2(head_e, 1, 1, cfg.dim, ctx)
        var head_in = cast_tensor(
            wan22_mod_pre(video, scale, shift, cfg.dim, cfg.eps, ctx), dtype, ctx
        )
        var projected = linear(
            head_in, self._w("head.head.weight"),
            Optional(self._w("head.head.bias").clone(ctx)), ctx,
        )
        var out_shape = List[Int]()
        out_shape.append(video_rows)
        out_shape.append(cfg.out_dim * 4)
        var projected2 = reshape(projected, out_shape^, ctx)
        return unpatchify3d(
            projected2, cfg.out_dim, FT, GH * 2, GW * 2, 1, 2, 2, ctx
        )

    def _run_pair[
        FT: Int, GH: Int, GW: Int, AR: Int, S: Int,
        TXT: Int, CTXL: Int, IMG: Int, H: Int, DH: Int,
    ](
        mut self,
        var sequence: Tensor,
        timestep: Float32,
        cond_context: Tensor,
        uncond_context: Tensor,
        clip_features: Tensor,
        cond_valid: Int,
        uncond_valid: Int,
        replace_flag: Bool,
        ctx: DeviceContext,
    ) raises -> Scail2PredictionPair:
        var cfg = self.config
        var plan = Scail2SequencePlan(
            video_t=FT, grid_h=GH, grid_w=GW,
            additional_ref_count=AR, replace_flag=replace_flag,
        )
        plan.validate()
        if plan.sequence_length() != S:
            raise Error("SCAIL-2 sequence/RoPE length mismatch")
        var dtype = sequence.dtype()
        var cond = sequence^
        var uncond = cond.clone(ctx)
        var time = self._time(timestep, dtype, ctx)
        var text_cond = self._text[TXT, CTXL](cond_context, dtype, ctx)
        var text_uncond = self._text[TXT, CTXL](uncond_context, dtype, ctx)
        var image = self._image[IMG](clip_features, ctx)
        var rope = build_scail2_rope_tables(plan, ctx, dtype)
        var load_seconds = Float64(0.0)
        var compute_seconds = Float64(0.0)

        for bi in range(cfg.num_layers):
            ctx.synchronize()
            var load_start = perf_counter()
            var block = self.stream.load_block_fp8(bi, ctx)
            load_seconds += perf_counter() - load_start
            var compute_start = perf_counter()
            cond = scail2_block_forward[S, TXT, IMG, H, DH](
                cond, time[0], text_cond, image, rope[0], rope[1], block,
                cfg, ctx, cond_valid,
            )
            uncond = scail2_block_forward[S, TXT, IMG, H, DH](
                uncond, time[0], text_uncond, image, rope[0], rope[1], block,
                cfg, ctx, uncond_valid,
            )
            ctx.synchronize()
            compute_seconds += perf_counter() - compute_start

        print(
            "  [scail2-profile] block_load_s=", load_seconds,
            " block_compute_s=", compute_seconds,
        )

        return Scail2PredictionPair(
            self._head_video[FT, GH, GW, AR, S](
                cond, time[1], dtype, ctx
            ),
            self._head_video[FT, GH, GW, AR, S](
                uncond, time[1], dtype, ctx
            ),
        )

    def forward_cfg_pair[
        FT: Int, GH: Int, GW: Int, S: Int,
        TXT: Int, CTXL: Int, IMG: Int, H: Int, DH: Int,
    ](
        mut self,
        video20: Tensor,
        primary_ref20: Tensor,
        pose20: Tensor,
        primary_video_masks28: Tensor,
        driving_masks28: Tensor,
        timestep: Float32,
        cond_context: Tensor,
        uncond_context: Tensor,
        clip_features: Tensor,
        cond_valid: Int,
        uncond_valid: Int,
        replace_flag: Bool,
        ctx: DeviceContext,
    ) raises -> Scail2PredictionPair:
        """CFG pair without optional additional references."""
        var sequence = self._embed_base_sequence[FT, GH, GW, S](
            video20, primary_ref20, pose20, primary_video_masks28,
            driving_masks28, ctx,
        )
        return self._run_pair[
            FT, GH, GW, 0, S, TXT, CTXL, IMG, H, DH,
        ](
            sequence^, timestep, cond_context, uncond_context, clip_features,
            cond_valid, uncond_valid, replace_flag, ctx,
        )

    def forward_cfg_pair_with_additional[
        FT: Int, GH: Int, GW: Int, AR: Int, S: Int,
        TXT: Int, CTXL: Int, IMG: Int, H: Int, DH: Int,
    ](
        mut self,
        video20: Tensor,
        primary_ref20: Tensor,
        pose20: Tensor,
        primary_video_masks28: Tensor,
        driving_masks28: Tensor,
        additional_refs20: Tensor,
        additional_ref_masks28: Tensor,
        timestep: Float32,
        cond_context: Tensor,
        uncond_context: Tensor,
        clip_features: Tensor,
        cond_valid: Int,
        uncond_valid: Int,
        replace_flag: Bool,
        ctx: DeviceContext,
    ) raises -> Scail2PredictionPair:
        var sequence = self._embed_sequence_with_additional[
            FT, GH, GW, AR, S,
        ](
            video20, primary_ref20, pose20, primary_video_masks28,
            driving_masks28, additional_refs20, additional_ref_masks28, ctx,
        )
        return self._run_pair[
            FT, GH, GW, AR, S, TXT, CTXL, IMG, H, DH,
        ](
            sequence^, timestep, cond_context, uncond_context, clip_features,
            cond_valid, uncond_valid, replace_flag, ctx,
        )
