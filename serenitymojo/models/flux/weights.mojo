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
# DTYPE: checkpoint is BF16. Production loaders keep checkpoint tensors in their
# stored dtype with Tensor.from_view.
#
# Mojo 0.26.x+: `def` not `fn`; move-only Tensor; reuses io.safetensors.SafeTensors
# + io.tensor_view.from_parts, exactly like Klein.

from std.collections import List, Optional
from std.memory import ArcPointer
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.models.flux.block import (
    StreamWeights, DoubleBlockWeights, SingleBlockWeights,
)
from serenitymojo.models.flux.flux_stack import (
    FluxStackBase, EmbedMlp, ModLin, DoubleModLin,
)


comptime TArc = ArcPointer[Tensor]


def _load_tensor(st: SafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    var info = st.tensor_info(name)
    var bytes = st.tensor_bytes(name)
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    return Tensor.from_view(tv, ctx)


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
    return StreamWeights(
        TArc(_load_tensor(st, ap + String(".qkv.weight"), ctx)),            # wqkv [3D, D]
        TArc(_load_tensor(st, ap + String(".qkv.bias"), ctx)),              # bqkv [3D]
        TArc(_load_tensor(st, ap + String(".proj.weight"), ctx)),           # wproj [D, D]
        TArc(_load_tensor(st, ap + String(".proj.bias"), ctx)),             # bproj [D]
        TArc(_load_tensor(st, mp + String(".0.weight"), ctx)),              # wmlp0 [Fmlp, D]
        TArc(_load_tensor(st, mp + String(".0.bias"), ctx)),                # bmlp0 [Fmlp]
        TArc(_load_tensor(st, mp + String(".2.weight"), ctx)),              # wmlp2 [D, Fmlp]
        TArc(_load_tensor(st, mp + String(".2.bias"), ctx)),                # bmlp2 [D]
        TArc(_load_tensor(st, ap + String(".norm.query_norm.scale"), ctx)), # q_norm [Dh]
        TArc(_load_tensor(st, ap + String(".norm.key_norm.scale"), ctx)),   # k_norm [Dh]
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
    return SingleBlockWeights(
        TArc(_load_tensor(st, sp + String(".linear1.weight"), ctx)),        # w1 [3D+Fmlp, D]
        TArc(_load_tensor(st, sp + String(".linear1.bias"), ctx)),          # b1 [3D+Fmlp]
        TArc(_load_tensor(st, sp + String(".linear2.weight"), ctx)),        # w2 [D, D+Fmlp]
        TArc(_load_tensor(st, sp + String(".linear2.bias"), ctx)),          # b2 [D]
        TArc(_load_tensor(st, sp + String(".norm.query_norm.scale"), ctx)), # q_norm [Dh]
        TArc(_load_tensor(st, sp + String(".norm.key_norm.scale"), ctx)),   # k_norm [Dh]
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
    return EmbedMlp(
        TArc(_load_tensor(st, in_w, ctx)),
        TArc(_load_tensor(st, prefix + String(".in_layer.bias"), ctx)),
        TArc(_load_tensor(st, prefix + String(".out_layer.weight"), ctx)),
        TArc(_load_tensor(st, prefix + String(".out_layer.bias"), ctx)),
    )


def _load_mod_lin(
    st: SafeTensors, prefix: String, ctx: DeviceContext
) raises -> ModLin:
    """A modulation linear (silu(vec) -> [chunk, D] -> [chunk])."""
    return ModLin(
        TArc(_load_tensor(st, prefix + String(".weight"), ctx)),
        TArc(_load_tensor(st, prefix + String(".bias"), ctx)),
    )


def load_flux_stack_base(
    st: SafeTensors, num_double: Int, num_single: Int,
    has_guidance: Bool, ctx: DeviceContext,
) raises -> FluxStackBase:
    """Load all FluxStackBase weights from a real flux1-dev safetensors."""
    var D = _dim0(st, String("img_in.weight"))

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
        var im = _load_mod_lin(st, dp + String(".img_mod.lin"), ctx)
        var tm = _load_mod_lin(st, dp + String(".txt_mod.lin"), ctx)
        dbl_mod.append(DoubleModLin(im^, tm^))

    var sgl_mod = List[ModLin]()
    for bi in range(num_single):
        var sp = String("single_blocks.") + String(bi)
        sgl_mod.append(_load_mod_lin(st, sp + String(".modulation.lin"), ctx))

    return FluxStackBase(
        TArc(_load_tensor(st, String("img_in.weight"), ctx)),
        TArc(_load_tensor(st, String("img_in.bias"), ctx)),
        TArc(_load_tensor(st, String("txt_in.weight"), ctx)),
        TArc(_load_tensor(st, String("txt_in.bias"), ctx)),
        time_in^, has_guidance, guid_in^, vec_in^,
        dbl_mod^, sgl_mod^,
        TArc(_load_tensor(st, String("final_layer.adaLN_modulation.1.weight"), ctx)),
        TArc(_load_tensor(st, String("final_layer.adaLN_modulation.1.bias"), ctx)),
        TArc(_load_tensor(st, String("final_layer.linear.weight"), ctx)),
        TArc(_load_tensor(st, String("final_layer.linear.bias"), ctx)),
    )


def _zeros(n: Int) -> List[Float32]:
    var o = List[Float32]()
    for _ in range(n):
        o.append(Float32(0.0))
    return o^
