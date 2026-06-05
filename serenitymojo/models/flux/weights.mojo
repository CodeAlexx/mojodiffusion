# models/flux/weights.mojo — Blocker B: real flux1-dev.safetensors -> the Flux
# block weight structs (DoubleBlockWeights / SingleBlockWeights) that
# models/flux/block.mojo consumes. Mirrors models/klein/weights.mojo (417 L).
#
# === Reference: BFL flux1-dev key layout (verified against the on-disk header) ===
# Per DOUBLE block bi (img stream; txt mirror with txt_ prefix) — 10 tensors/stream:
#   double_blocks.{bi}.{img,txt}_attn.qkv.weight            [3D, D]   -> StreamWeights.wqkv
#   double_blocks.{bi}.{img,txt}_attn.qkv.bias              [3D]      -> .bqkv
#   double_blocks.{bi}.{img,txt}_attn.proj.weight           [D, D]    -> .wproj
#   double_blocks.{bi}.{img,txt}_attn.proj.bias             [D]       -> .bproj
#   double_blocks.{bi}.{img,txt}_mlp.0.weight               [Fmlp, D] -> .wmlp0
#   double_blocks.{bi}.{img,txt}_mlp.0.bias                 [Fmlp]    -> .bmlp0
#   double_blocks.{bi}.{img,txt}_mlp.2.weight               [D, Fmlp] -> .wmlp2
#   double_blocks.{bi}.{img,txt}_mlp.2.bias                 [D]       -> .bmlp2
#   double_blocks.{bi}.{img,txt}_attn.norm.query_norm.scale [Dh]      -> .q_norm
#   double_blocks.{bi}.{img,txt}_attn.norm.key_norm.scale   [Dh]      -> .k_norm
#   (modulation lives in double_blocks.{bi}.{img,txt}_mod.lin.* — a STACK-level
#    embedder that produces ModVecs; NOT a block weight, handled at stack phase.)
#
# Per SINGLE block bi — 6 tensors:
#   single_blocks.{bi}.linear1.weight   [3D+Fmlp, D]  -> SingleBlockWeights.w1
#   single_blocks.{bi}.linear1.bias     [3D+Fmlp]     -> .b1
#   single_blocks.{bi}.linear2.weight   [D, D+Fmlp]   -> .w2
#   single_blocks.{bi}.linear2.bias     [D]           -> .b2
#   single_blocks.{bi}.norm.query_norm.scale [Dh]     -> .q_norm
#   single_blocks.{bi}.norm.key_norm.scale   [Dh]     -> .k_norm
#
# flux1-dev: D = inner_dim = 3072, Dh = head_dim = 128, n_heads = 24,
#   Fmlp = mlp_hidden = 12288 (= D*4). 19 double blocks, 38 single blocks.
#
# DTYPE: checkpoint is BF16. This loader converts to host F32 lists because the
# legacy constructors take List[Float32]; the block structs immediately upload
# BF16 device tensors in __init__, so inference weights stay BF16 at runtime.
#
# Mojo 0.26.x+: `def` not `fn`; move-only Tensor; reuses io.safetensors.SafeTensors
# + io.tensor_view.from_parts + ops.cast.cast_tensor, exactly like Klein.

from std.collections import List, Optional
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.models.flux.block import (
    StreamWeights, DoubleBlockWeights, SingleBlockWeights,
)
from serenitymojo.models.flux.flux_stack import (
    FluxStackBase, EmbedMlp, ModLin, DoubleModLin,
)


# Read one named tensor from the safetensors as a host List[Float32] (casts up
# from the stored dtype — flux1-dev base is BF16). Mirrors klein/weights.mojo.
def _load_host_f32(st: SafeTensors, name: String, ctx: DeviceContext) raises -> List[Float32]:
    var info = st.tensor_info(name)
    var bytes = st.tensor_bytes(name)
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    var t = Tensor.from_view(tv, ctx)
    var t32 = cast_tensor(t, STDtype.F32, ctx)
    return t32.to_host(ctx)


# Dim 0 / dim 1 of a named tensor's stored shape (for deriving D/F/Dh).
def _dim0(st: SafeTensors, name: String) raises -> Int:
    var info = st.tensor_info(name)
    return Int(info.shape[0])


def _dim1(st: SafeTensors, name: String) raises -> Int:
    var info = st.tensor_info(name)
    return Int(info.shape[1])


# ── DOUBLE block: per-stream loader ───────────────────────────────────────────
# Dims derived from the stored shapes (no hardcoding):
#   D    = dim1(proj.weight)   (proj is [D, D])
#   Fmlp = dim0(mlp.0.weight)  (mlp.0 is [Fmlp, D])
#   Dh   = dim0(norm.query_norm.scale)
def _load_stream(
    st: SafeTensors, dp: String, stream: String, ctx: DeviceContext
) raises -> StreamWeights:
    var ap = dp + String(".") + stream + String("_attn")
    var mp = dp + String(".") + stream + String("_mlp")
    var D = _dim1(st, ap + String(".proj.weight"))            # proj [D, D] -> D
    var Fmlp = _dim0(st, mp + String(".0.weight"))            # mlp.0 [Fmlp, D] -> Fmlp
    var Dh = _dim0(st, ap + String(".norm.query_norm.scale")) # q_norm [Dh]
    return StreamWeights(
        _load_host_f32(st, ap + String(".qkv.weight"), ctx),            # wqkv [3D, D]
        _load_host_f32(st, ap + String(".qkv.bias"), ctx),             # bqkv [3D]
        _load_host_f32(st, ap + String(".proj.weight"), ctx),          # wproj [D, D]
        _load_host_f32(st, ap + String(".proj.bias"), ctx),            # bproj [D]
        _load_host_f32(st, mp + String(".0.weight"), ctx),             # wmlp0 [Fmlp, D]
        _load_host_f32(st, mp + String(".0.bias"), ctx),               # bmlp0 [Fmlp]
        _load_host_f32(st, mp + String(".2.weight"), ctx),             # wmlp2 [D, Fmlp]
        _load_host_f32(st, mp + String(".2.bias"), ctx),               # bmlp2 [D]
        _load_host_f32(st, ap + String(".norm.query_norm.scale"), ctx),# q_norm [Dh]
        _load_host_f32(st, ap + String(".norm.key_norm.scale"), ctx),  # k_norm [Dh]
        D, Fmlp, Dh, ctx,
    )


# Load double block `block_idx`'s real weights into DoubleBlockWeights.
def load_double_block_weights(
    st: SafeTensors, block_idx: Int, ctx: DeviceContext
) raises -> DoubleBlockWeights:
    var dp = String("double_blocks.") + String(block_idx)
    return DoubleBlockWeights(
        _load_stream(st, dp, String("img"), ctx),
        _load_stream(st, dp, String("txt"), ctx),
    )


# ── SINGLE block loader ───────────────────────────────────────────────────────
# Dims from the stored shapes:
#   D    = dim0(linear2.weight)   (linear2 is [D, D+Fmlp])
#   Fmlp = dim1(linear2.weight) - D
#   Dh   = dim0(norm.query_norm.scale)
def load_single_block_weights(
    st: SafeTensors, block_idx: Int, ctx: DeviceContext
) raises -> SingleBlockWeights:
    var sp = String("single_blocks.") + String(block_idx)
    var D = _dim0(st, sp + String(".linear2.weight"))         # w2 [D, D+Fmlp] -> D
    var Fmlp = _dim1(st, sp + String(".linear2.weight")) - D  # D+Fmlp - D
    var Dh = _dim0(st, sp + String(".norm.query_norm.scale"))
    return SingleBlockWeights(
        _load_host_f32(st, sp + String(".linear1.weight"), ctx),         # w1 [3D+Fmlp, D]
        _load_host_f32(st, sp + String(".linear1.bias"), ctx),           # b1 [3D+Fmlp]
        _load_host_f32(st, sp + String(".linear2.weight"), ctx),         # w2 [D, D+Fmlp]
        _load_host_f32(st, sp + String(".linear2.bias"), ctx),           # b2 [D]
        _load_host_f32(st, sp + String(".norm.query_norm.scale"), ctx),  # q_norm [Dh]
        _load_host_f32(st, sp + String(".norm.key_norm.scale"), ctx),    # k_norm [Dh]
        D, Fmlp, Dh, ctx,
    )


# ── STACK-LEVEL base loader (Blocker B, stack half) ───────────────────────────
# Loads every NON-streamed transformer weight into a FluxStackBase: the input
# projections (img_in/txt_in), the three embed MLPs (time/guidance/vector_in),
# the PER-BLOCK modulation linears (double_blocks.{i}.{img,txt}_mod.lin and
# single_blocks.{i}.modulation.lin), and the final layer
# (final_layer.adaLN_modulation.1 + final_layer.linear). These are the weights
# FluxStackBase holds resident (the block attn/mlp weights are streamed via the
# offload loader instead). Real flux1-dev keys verified against the on-disk
# header (img_in.weight [3072,64], txt_in.weight [3072,4096],
# {time,guidance,vector}_in.{in,out}_layer, img_mod.lin.weight [18432,3072] = [6D,D],
# modulation.lin.weight [9216,3072] = [3D,D], final_layer.adaLN_modulation.1.weight
# [6144,3072] = [2D,D], final_layer.linear.weight [64,3072] = [out_ch,D]).
#
# DIMS are derived from the stored shapes (no hardcoding except out_ch is read).
#   D      = dim0(img_in.weight)        (img_in [D, in_ch])
#   in_ch  = dim1(img_in.weight)
#   txt_ch = dim1(txt_in.weight)        (txt_in [D, txt_ch])
#   T_DIM  = dim1(time_in.in_layer.weight)   (in_layer [D, T_DIM])
#   VEC_DIM= dim1(vector_in.in_layer.weight)
#   out_ch = dim0(final_layer.linear.weight)
def _load_embed_mlp(
    st: SafeTensors, prefix: String, ctx: DeviceContext
) raises -> EmbedMlp:
    """time_in / guidance_in / vector_in: in_layer -> silu -> out_layer."""
    var in_w = prefix + String(".in_layer.weight")
    var D = _dim0(st, in_w)
    var in_dim = _dim1(st, in_w)
    return EmbedMlp(
        _load_host_f32(st, in_w, ctx),
        _load_host_f32(st, prefix + String(".in_layer.bias"), ctx),
        _load_host_f32(st, prefix + String(".out_layer.weight"), ctx),
        _load_host_f32(st, prefix + String(".out_layer.bias"), ctx),
        in_dim, D, ctx,
    )


def _load_mod_lin(
    st: SafeTensors, prefix: String, chunk: Int, D: Int, ctx: DeviceContext
) raises -> ModLin:
    """A modulation linear (silu(vec) -> [chunk, D] -> [chunk])."""
    return ModLin(
        _load_host_f32(st, prefix + String(".weight"), ctx),
        _load_host_f32(st, prefix + String(".bias"), ctx),
        chunk, D, ctx,
    )


def load_flux_stack_base(
    st: SafeTensors, num_double: Int, num_single: Int,
    has_guidance: Bool, ctx: DeviceContext,
) raises -> FluxStackBase:
    """Load all FluxStackBase weights from a real flux1-dev safetensors."""
    var D = _dim0(st, String("img_in.weight"))
    var in_ch = _dim1(st, String("img_in.weight"))
    var txt_ch = _dim1(st, String("txt_in.weight"))
    var out_ch = _dim0(st, String("final_layer.linear.weight"))

    var time_in = _load_embed_mlp(st, String("time_in"), ctx)
    # guidance_in may be absent (Schnell); load it when has_guidance, else a
    # zero placeholder shaped like time_in (never used when has_guidance=False).
    var guid_in: EmbedMlp
    if has_guidance:
        guid_in = _load_embed_mlp(st, String("guidance_in"), ctx)
    else:
        var t_dim = _dim1(st, String("time_in.in_layer.weight"))
        guid_in = EmbedMlp(
            _zeros(D * t_dim), _zeros(D), _zeros(D * D), _zeros(D), t_dim, D, ctx
        )
    var vec_in = _load_embed_mlp(st, String("vector_in"), ctx)

    var dbl_mod = List[DoubleModLin]()
    for bi in range(num_double):
        var dp = String("double_blocks.") + String(bi)
        var im = _load_mod_lin(st, dp + String(".img_mod.lin"), 6 * D, D, ctx)
        var tm = _load_mod_lin(st, dp + String(".txt_mod.lin"), 6 * D, D, ctx)
        dbl_mod.append(DoubleModLin(im^, tm^))

    var sgl_mod = List[ModLin]()
    for bi in range(num_single):
        var sp = String("single_blocks.") + String(bi)
        sgl_mod.append(_load_mod_lin(st, sp + String(".modulation.lin"), 3 * D, D, ctx))

    return FluxStackBase(
        _load_host_f32(st, String("img_in.weight"), ctx),
        _load_host_f32(st, String("img_in.bias"), ctx),
        _load_host_f32(st, String("txt_in.weight"), ctx),
        _load_host_f32(st, String("txt_in.bias"), ctx),
        time_in^, has_guidance, guid_in^, vec_in^,
        dbl_mod^, sgl_mod^,
        _load_host_f32(st, String("final_layer.adaLN_modulation.1.weight"), ctx),
        _load_host_f32(st, String("final_layer.adaLN_modulation.1.bias"), ctx),
        _load_host_f32(st, String("final_layer.linear.weight"), ctx),
        _load_host_f32(st, String("final_layer.linear.bias"), ctx),
        D, in_ch, txt_ch, out_ch, ctx,
    )


def _zeros(n: Int) -> List[Float32]:
    var o = List[Float32]()
    for _ in range(n):
        o.append(Float32(0.0))
    return o^
