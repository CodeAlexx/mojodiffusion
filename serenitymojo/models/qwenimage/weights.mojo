# models/qwenimage/weights.mojo — real safetensors -> training weight structs.
#
# Loads Qwen-Image MMDiT weights from the diffusers sharded transformer dir into
# the DoubleBlockWeights / QwenStackBase the verified block + stack fwd/bwd
# consume. Mirrors the inference loader (models/dit/qwenimage_dit.mojo::load,
# ShardedSafeTensors + Tensor.from_view) and the Klein training loader
# (models/klein/weights.mojo).
#
# Per-block key layout (per transformer_blocks.{bi}.), from the inference
# _block_forward (qwenimage_dit.mojo:490-632) + EDv2 qwenimage.rs:419-430:
#   attn.to_q / to_k / to_v / to_out.0          (.weight [D,D], .bias [D])   img
#   attn.add_q_proj / add_k_proj / add_v_proj / to_add_out                   txt
#   attn.norm_q / norm_k / norm_added_q / norm_added_k (.weight [Dh])
#   img_mlp.net.0.proj / net.2 (.weight, .bias)   txt_mlp.net.0.proj / net.2
#   img_mod.1 / txt_mod.1 (.weight [6D,D], .bias [6D])   (frozen modulation MLP)
# Top-level: img_in / txt_in / txt_norm / time_text_embed.* / norm_out.linear /
#   proj_out.
#
# Frozen modulation: this loader returns the per-block img_mod.1/txt_mod.1 +
# norm_out weights so the trainer can compute per-block ModVecs from a temb. The
# ModVecs themselves are computed per-step (modulation depends on the timestep).
#
# Mojo 1.0.0b1, NVIDIA GPU. Qwen-Image base is BF16 in the checkpoint; loaded
# host lists are cast up to F32 (training master precision).

from std.collections import List, Optional
from std.gpu.host import DeviceContext
from std.memory import ArcPointer
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.linear import linear
from serenitymojo.ops.activations import silu
from serenitymojo.ops.embeddings import timestep_embedding
from serenitymojo.ops.tensor_algebra import reshape, slice
from serenitymojo.models.qwenimage.qwenimage_block import (
    StreamWeights, DoubleBlockWeights, ModVecs,
)
from serenitymojo.models.qwenimage.qwenimage_stack import QwenStackBase


# Read one named tensor from the sharded transformer as a host List[Float32]
# (casts up from the stored dtype — Qwen-Image base is BF16).
def _load_host_f32(
    st: ShardedSafeTensors, name: String, ctx: DeviceContext
) raises -> List[Float32]:
    var tv = st.tensor_view(name)
    var t = Tensor.from_view(tv, ctx)
    var t32 = cast_tensor(t, STDtype.F32, ctx)
    return t32.to_host(ctx)


# ── per-stream weights from a block (img or txt prefix selects the projections) ─
def _stream_weights_from_block(
    st: ShardedSafeTensors, block_prefix: String, is_img: Bool,
    D: Int, F: Int, Dh: Int, ctx: DeviceContext,
) raises -> StreamWeights:
    var qp: String
    var kp: String
    var vp: String
    var op: String
    var nqp: String
    var nkp: String
    var mlp: String
    if is_img:
        qp = ".attn.to_q"
        kp = ".attn.to_k"
        vp = ".attn.to_v"
        op = ".attn.to_out.0"
        nqp = ".attn.norm_q"
        nkp = ".attn.norm_k"
        mlp = ".img_mlp"
    else:
        qp = ".attn.add_q_proj"
        kp = ".attn.add_k_proj"
        vp = ".attn.add_v_proj"
        op = ".attn.to_add_out"
        nqp = ".attn.norm_added_q"
        nkp = ".attn.norm_added_k"
        mlp = ".txt_mlp"

    return StreamWeights(
        _load_host_f32(st, block_prefix + qp + ".weight", ctx),
        _load_host_f32(st, block_prefix + kp + ".weight", ctx),
        _load_host_f32(st, block_prefix + vp + ".weight", ctx),
        _load_host_f32(st, block_prefix + qp + ".bias", ctx),
        _load_host_f32(st, block_prefix + kp + ".bias", ctx),
        _load_host_f32(st, block_prefix + vp + ".bias", ctx),
        _load_host_f32(st, block_prefix + op + ".weight", ctx),
        _load_host_f32(st, block_prefix + op + ".bias", ctx),
        _load_host_f32(st, block_prefix + mlp + ".net.0.proj.weight", ctx),
        _load_host_f32(st, block_prefix + mlp + ".net.0.proj.bias", ctx),
        _load_host_f32(st, block_prefix + mlp + ".net.2.weight", ctx),
        _load_host_f32(st, block_prefix + mlp + ".net.2.bias", ctx),
        _load_host_f32(st, block_prefix + nqp + ".weight", ctx),
        _load_host_f32(st, block_prefix + nkp + ".weight", ctx),
        D, F, Dh, ctx,
    )


def double_weights_from_block(
    st: ShardedSafeTensors, bi: Int, D: Int, F: Int, Dh: Int, ctx: DeviceContext
) raises -> DoubleBlockWeights:
    var p = String("transformer_blocks.") + String(bi)
    var img = _stream_weights_from_block(st, p, True, D, F, Dh, ctx)
    var txt = _stream_weights_from_block(st, p, False, D, F, Dh, ctx)
    return DoubleBlockWeights(img^, txt^)


def load_qwen_double_weights(
    st: ShardedSafeTensors, num_double: Int, D: Int, F: Int, Dh: Int, ctx: DeviceContext
) raises -> List[DoubleBlockWeights]:
    var out = List[DoubleBlockWeights]()
    for bi in range(num_double):
        out.append(double_weights_from_block(st, bi, D, F, Dh, ctx))
    return out^


def load_qwen_stack_base(
    st: ShardedSafeTensors, D: Int, in_ch: Int, txt_ch: Int, out_ch: Int, ctx: DeviceContext
) raises -> QwenStackBase:
    # txt_in input is RMSNorm(txt_norm)(encoder_hidden) [N_TXT,txt_ch]; the txt_norm
    # scale is applied OUTSIDE the stack (caller pre-normalizes), so here txt_in is
    # just the projection. img_in/txt_in/proj_out are biased linears.
    return QwenStackBase(
        _load_host_f32(st, "img_in.weight", ctx), _load_host_f32(st, "img_in.bias", ctx),
        _load_host_f32(st, "txt_in.weight", ctx), _load_host_f32(st, "txt_in.bias", ctx),
        _load_host_f32(st, "proj_out.weight", ctx), _load_host_f32(st, "proj_out.bias", ctx),
        D, in_ch, txt_ch, out_ch, ctx,
    )


# ── per-block ModVecs from temb via the frozen mod MLP (img_mod.1 / txt_mod.1) ─
# temb_h: [1,D] silu-able timestep embedding (the trainer precomputes temb from
# the timestep_embedder MLP). mod_w [6D,D], mod_b [6D]: the frozen modulation
# linear for this block+stream. Returns the 6 chunks shift1,scale1,gate1,shift2,
# scale2,gate2 each [D] (diffusers chunk order, qwenimage.rs:1683).
def modvecs_from_temb(
    temb_h: List[Float32], mod_w: List[Float32], mod_b: List[Float32],
    D: Int, ctx: DeviceContext,
) raises -> ModVecs:
    var temb = Tensor.from_host(temb_h.copy(), [1, D], STDtype.F32, ctx)
    var act = silu(temb, ctx)
    var bt = Tensor.from_host(mod_b.copy(), [6 * D], STDtype.F32, ctx)
    var mods = linear(
        act, Tensor.from_host(mod_w.copy(), [6 * D, D], STDtype.F32, ctx),
        Optional[Tensor](bt^), ctx,
    ).to_host(ctx)   # [1,6D]
    return ModVecs(
        _chunk_d(mods, 0, D), _chunk_d(mods, 1, D), _chunk_d(mods, 2, D),
        _chunk_d(mods, 3, D), _chunk_d(mods, 4, D), _chunk_d(mods, 5, D),
    )


# extract chunk `off` of width D from a [6D]/[2D] flat list.
def _chunk_d(mods: List[Float32], off: Int, D: Int) -> List[Float32]:
    var o = List[Float32]()
    for i in range(D):
        o.append(mods[off * D + i])
    return o^


# Compute all per-block ModVecs for the stack (img + txt) from one temb.
struct QwenPerBlockMods(Movable):
    var img_mods: List[ModVecs]
    var txt_mods: List[ModVecs]
    var final_scale: List[Float32]
    var final_shift: List[Float32]

    def __init__(
        out self, var img_mods: List[ModVecs], var txt_mods: List[ModVecs],
        var final_scale: List[Float32], var final_shift: List[Float32],
    ):
        self.img_mods = img_mods^
        self.txt_mods = txt_mods^
        self.final_scale = final_scale^
        self.final_shift = final_shift^


def build_qwen_per_block_mods(
    st: ShardedSafeTensors, temb_h: List[Float32], num_double: Int, D: Int, ctx: DeviceContext
) raises -> QwenPerBlockMods:
    var img_mods = List[ModVecs]()
    var txt_mods = List[ModVecs]()
    for bi in range(num_double):
        var p = String("transformer_blocks.") + String(bi)
        var imw = _load_host_f32(st, p + ".img_mod.1.weight", ctx)
        var imb = _load_host_f32(st, p + ".img_mod.1.bias", ctx)
        var tmw = _load_host_f32(st, p + ".txt_mod.1.weight", ctx)
        var tmb = _load_host_f32(st, p + ".txt_mod.1.bias", ctx)
        img_mods.append(modvecs_from_temb(temb_h, imw, imb, D, ctx))
        txt_mods.append(modvecs_from_temb(temb_h, tmw, tmb, D, ctx))

    # final layer: norm_out.linear -> [2D]: chunk 0 scale, chunk 1 shift
    var temb = Tensor.from_host(temb_h.copy(), [1, D], STDtype.F32, ctx)
    var act = silu(temb, ctx)
    var fb = _load_host_f32(st, "norm_out.linear.bias", ctx)
    var fw = _load_host_f32(st, "norm_out.linear.weight", ctx)
    var fmods = linear(
        act, Tensor.from_host(fw^, [2 * D, D], STDtype.F32, ctx),
        Optional[Tensor](Tensor.from_host(fb^, [2 * D], STDtype.F32, ctx)), ctx,
    ).to_host(ctx)   # [1,2D]
    var fscale = List[Float32]()
    var fshift = List[Float32]()
    for i in range(D):
        fscale.append(fmods[i])
        fshift.append(fmods[D + i])
    return QwenPerBlockMods(img_mods^, txt_mods^, fscale^, fshift^)
