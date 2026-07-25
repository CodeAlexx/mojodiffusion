# Inference-only Wan2.2 A14B forward over the persistent per-block FP8 stream.
#
# This is a thin residency adapter around the existing parity-gated Wan math.
# It does not implement a new transformer block.  Shared weights stay BF16 on
# device; one cached expert block is E4M3-streamed/dequantized at a time.  The
# CFG-paired forward loads each block once and applies it to conditional and
# unconditional activations before releasing it.

from std.collections import Dict, List, Optional
from std.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.io.dtype import STDtype
from serenitymojo.models.dit.wan22_dit import (
    Wan22Config,
    _chunk2,
    _pad_context,
    _unsqueeze2,
    wan22_block_forward,
    wan22_build_rope,
    wan22_mod_pre,
)
from serenitymojo.models.wan22.wan22_fp8_stream import Wan22A14BFP8Stream
from serenitymojo.lora import LoraSet
from serenitymojo.ops.activations import gelu, silu
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.embeddings import timestep_embedding
from serenitymojo.ops.linear import linear
from serenitymojo.ops.patchify3d import patchify3d, unpatchify3d
from serenitymojo.ops.tensor_algebra import add, full_device, reshape
from serenitymojo.tensor import Tensor


comptime TArc = ArcPointer[Tensor]


struct Wan22A14BPair(Movable):
    var cond: Tensor
    var uncond: Tensor

    def __init__(out self, var cond: Tensor, var uncond: Tensor):
        self.cond = cond^
        self.uncond = uncond^


struct Wan22A14BStreamedDiT(Movable):
    var weights: Dict[String, TArc]
    var stream: Wan22A14BFP8Stream
    var config: Wan22Config
    # Optional trained-LoRA overlay, applied per block right after the E4M3
    # dequant (see `_apply_block_lora`). None = the plain base model.
    var lora: Optional[LoraSet]
    var lora_mult: Float32

    @staticmethod
    def open(cache_dir: String, ctx: DeviceContext) raises -> Wan22A14BStreamedDiT:
        var stream = Wan22A14BFP8Stream.open(cache_dir)
        var shared = stream.load_shared(ctx)
        return Wan22A14BStreamedDiT(shared^, stream^)

    def __init__(
        out self, var weights: Dict[String, TArc], var stream: Wan22A14BFP8Stream,
    ):
        self.weights = weights^
        self.stream = stream^
        self.config = Wan22Config.t2v_a14b()
        self.lora = Optional[LoraSet](None)
        self.lora_mult = Float32(1.0)

    def set_lora(mut self, var lora: LoraSet, multiplier: Float32) raises:
        """Attach a trained LoRA (the trainer's `diffusion_model.…lora_A/lora_B`
        artifact, loaded with `LoraSet.load`) to every block forward.

        Fail-loud on a LoRA that matches nothing: a silently-inert overlay renders
        the BASE model and looks like "the LoRA did not train", which is the single
        most expensive way for this to go wrong."""
        var hits = 0
        for ref m in lora.mappings:
            if m.base_key.startswith(String("blocks.")):
                hits += 1
        if hits == 0:
            raise Error(
                String("Wan22A14BStreamedDiT.set_lora: none of the ")
                + String(len(lora.mappings))
                + " LoRA modules target a `blocks.N.…` key — wrong model or wrong"
                " key format (expected the wan22 trainer's diffusion_model.blocks.*)"
            )
        print("[lora] attached", hits, "wan22 block modules (multiplier",
              multiplier, ", format", lora.format_name(), ")")
        self.lora = Optional[LoraSet](lora^)
        self.lora_mult = multiplier

    def _apply_block_lora(
        self, mut block: Dict[String, TArc], bi: Int, ctx: DeviceContext,
    ) raises -> Int:
        """Fold this block's LoRA deltas into the freshly dequantized BF16 weights.

        The seam matters: `load_block_bf16` hands back block-LOCAL keys
        (`self_attn.q.weight`) while `LoraSet` mappings carry FULL keys
        (`blocks.7.self_attn.q.weight`), so we strip `blocks.{bi}.` here. This is the
        krea2 `_blk_w8` pattern (models/dit/krea2_dit.mojo:1822) — overlay AFTER the
        E4M3 dequant, never on the fp8 bytes.

        Merge form (materialize [out,in] and add) rather than krea2's low-rank side
        cache: at rank 16 the delta is built once per block per step and the block is
        already resident in BF16, so this keeps the change to one place. If sampling
        turns out to be too slow, `build_krea2_lora_side_cache` is the faster shape."""
        if not self.lora:
            return 0
        ref ls = self.lora.value()
        var prefix = String("blocks.") + String(bi) + String(".")
        var applied = 0
        for ref m in ls.mappings:
            if not m.base_key.startswith(prefix):
                continue
            var local = String(m.base_key.removeprefix(prefix))
            if local not in block:
                continue
            ref w = block[local][]
            var scale = ls._module_scale(m, self.lora_mult, ctx)
            var delta = ls._compute_delta(m, scale, w.dtype(), ctx)
            block[local] = TArc(add(w, delta, ctx))
            applied += 1
        return applied

    def _w(self, name: String) raises -> ref [self.weights] Tensor:
        if name not in self.weights:
            raise Error(String("Wan A14B shared cache missing tensor: ") + name)
        return self.weights[name][]

    def _patch[
        FG: Int, HG: Int, WG: Int, S: Int
    ](self, x_lat: Tensor, ctx: DeviceContext) raises -> Tensor:
        var cfg = self.config
        var patched = patchify3d(x_lat, 1, 2, 2, ctx)
        var pe_shape = List[Int]()
        pe_shape.append(cfg.dim)
        pe_shape.append(cfg.in_dim * 4)
        var pe_w = reshape(self._w("patch_embedding.weight"), pe_shape^, ctx)
        var img2d = linear(
            patched, pe_w,
            Optional(self._w("patch_embedding.bias").clone(ctx)), ctx,
        )
        var shape = List[Int]()
        shape.append(1)
        shape.append(S)
        shape.append(cfg.dim)
        return reshape(img2d, shape^, ctx)

    def _time[
        S: Int
    ](
        self, timestep: Float32, dtype: STDtype, ctx: DeviceContext,
    ) raises -> Tuple[Tensor, Tensor]:
        var cfg = self.config
        var t_shape = List[Int]()
        # One batch item has one timestep. The creator expands this row only to
        # support packed samples with different timesteps; block modulation is
        # elementwise, so retaining [1,1,6,D] and broadcasting is exact and
        # avoids two ~4.1 GB F32 tensors at the 848x480x81 product geometry.
        t_shape.append(1)
        var t_vec = full_device(t_shape^, timestep, STDtype.F32, ctx)
        var sinusoid = timestep_embedding(
            t_vec, cfg.freq_dim, ctx, cfg.rope_theta, dtype
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
        var e_silu = silu(e, ctx)
        var e0_flat = linear(
            e_silu, self._w("time_projection.1.weight"),
            Optional(self._w("time_projection.1.bias").clone(ctx)), ctx,
        )
        var e0_shape = List[Int]()
        e0_shape.append(1)
        e0_shape.append(1)
        e0_shape.append(6)
        e0_shape.append(cfg.dim)
        var e0 = reshape(cast_tensor(e0_flat, STDtype.F32, ctx), e0_shape^, ctx)
        var head_shape = List[Int]()
        head_shape.append(1)
        head_shape.append(1)
        head_shape.append(cfg.dim)
        var e_head = reshape(cast_tensor(e, STDtype.F32, ctx), head_shape^, ctx)
        return (e0^, e_head^)

    def _text[
        TXT: Int, CTXL: Int
    ](
        self, context: Tensor, dtype: STDtype, ctx: DeviceContext,
    ) raises -> Tensor:
        var cfg = self.config
        var padded = _pad_context(
            context, CTXL, TXT, cfg.text_dim, dtype, ctx
        )
        var text = linear(
            padded, self._w("text_embedding.0.weight"),
            Optional(self._w("text_embedding.0.bias").clone(ctx)), ctx,
        )
        text = gelu(text, ctx)
        return linear(
            text, self._w("text_embedding.2.weight"),
            Optional(self._w("text_embedding.2.bias").clone(ctx)), ctx,
        )

    def _head[
        FG: Int, HG: Int, WG: Int, S: Int
    ](
        self, img: Tensor, e_head: Tensor, dtype: STDtype, ctx: DeviceContext,
    ) raises -> Tensor:
        var cfg = self.config
        var head_mod = cast_tensor(
            self._w("head.modulation"), STDtype.F32, ctx
        )
        var time_rows = S
        if e_head.numel() == cfg.dim:
            time_rows = 1
        elif e_head.numel() != S * cfg.dim:
            raise Error("Wan A14B head timestep embedding shape mismatch")
        var e_head_u = _unsqueeze2(e_head, time_rows, cfg.dim, ctx)
        var head_e = add(head_mod, e_head_u, ctx)
        var shift = _chunk2(head_e, 0, time_rows, cfg.dim, ctx)
        var scale = _chunk2(head_e, 1, time_rows, cfg.dim, ctx)
        var head_in = cast_tensor(
            wan22_mod_pre(img, scale, shift, cfg.dim, cfg.eps, ctx), dtype, ctx
        )
        var projected = linear(
            head_in, self._w("head.head.weight"),
            Optional(self._w("head.head.bias").clone(ctx)), ctx,
        )
        var shape = List[Int]()
        shape.append(S)
        shape.append(cfg.out_dim * 4)
        var projected_2d = reshape(projected, shape^, ctx)
        return unpatchify3d(
            projected_2d, cfg.out_dim, FG, HG * 2, WG * 2, 1, 2, 2, ctx
        )

    def forward_cfg_pair[
        FG: Int, HG: Int, WG: Int, S: Int, TXT: Int, CTXL: Int, H: Int, DH: Int
    ](
        mut self,
        x_lat: Tensor,
        timestep: Float32,
        cond_context: Tensor,
        uncond_context: Tensor,
        cond_valid: Int,
        uncond_valid: Int,
        ctx: DeviceContext,
    ) raises -> Wan22A14BPair:
        """Two Bernini T2V predictions with one weight load per expert block."""
        var cfg = self.config
        var dtype = x_lat.dtype()
        var img_cond = self._patch[FG, HG, WG, S](x_lat, ctx)
        var img_uncond = img_cond.clone(ctx)
        var time = self._time[S](timestep, dtype, ctx)
        var text_cond = self._text[TXT, CTXL](cond_context, dtype, ctx)
        var text_uncond = self._text[TXT, CTXL](uncond_context, dtype, ctx)
        var rope = wan22_build_rope(
            FG, HG, WG, DH, cfg.rope_theta, dtype, ctx
        )

        var lora_applied = 0
        for bi in range(cfg.num_layers):
            var block = self.stream.load_block_bf16(bi, ctx)
            # LoRA overlay on the dequantized block, BEFORE either forward — both
            # the cond and uncond pass must see the same adapted weights.
            lora_applied += self._apply_block_lora(block, bi, ctx)
            img_cond = wan22_block_forward[S, TXT, H, DH](
                img_cond, time[0], text_cond, rope[0], rope[1], block, cfg, ctx,
                cond_valid,
            )
            img_uncond = wan22_block_forward[S, TXT, H, DH](
                img_uncond, time[0], text_uncond, rope[0], rope[1], block, cfg, ctx,
                uncond_valid,
            )

        if self.lora and lora_applied == 0:
            raise Error(
                "Wan22A14BStreamedDiT: a LoRA is attached but ZERO modules matched"
                " any streamed block weight — the render would silently be the base"
                " model. Check the key namespace of the LoRA file."
            )

        var cond = self._head[FG, HG, WG, S](
            img_cond, time[1], dtype, ctx
        )
        var uncond = self._head[FG, HG, WG, S](
            img_uncond, time[1], dtype, ctx
        )
        return Wan22A14BPair(cond^, uncond^)
