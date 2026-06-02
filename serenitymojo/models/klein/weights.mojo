# models/klein/weights.mojo — G1: real safetensors -> training weight structs.
#
# Loads Klein double-block weights from a real .safetensors into the
# DoubleBlockWeights the verified double_block_forward/backward consume. The
# inference path (models/dit/klein_dit.mojo) already reads these exact keys; this
# is the cross-pollination into the TRAINING weight structs (host List[Float32]).
#
# Key layout (per double block bi), from klein_dit.mojo:112-123 — 12 tensors:
#   double_blocks.{bi}.{img,txt}_attn.qkv.weight            -> StreamWeights.wqkv
#   double_blocks.{bi}.{img,txt}_attn.proj.weight           -> .wproj
#   double_blocks.{bi}.{img,txt}_attn.norm.query_norm.scale -> .q_norm
#   double_blocks.{bi}.{img,txt}_attn.norm.key_norm.scale   -> .k_norm
#   double_blocks.{bi}.{img,txt}_mlp.0.weight               -> .wgu  (fused gate+up)
#   double_blocks.{bi}.{img,txt}_mlp.2.weight               -> .wd
#
# Key layout (per single block bi), from klein_dit.mojo `_single_block` — 4 tensors:
#   single_blocks.{bi}.linear1.weight            -> SingleBlockWeights.w1 [3D+2F, D]
#   single_blocks.{bi}.linear2.weight            -> .w2 [D, D+F]
#   single_blocks.{bi}.norm.query_norm.scale     -> .q_norm [Dh]
#   single_blocks.{bi}.norm.key_norm.scale       -> .k_norm [Dh]
# These feed the verified single_block_forward/backward (models/klein/single_block.mojo).

from std.collections import List, Optional
from std.gpu.host import DeviceContext
from std.memory import ArcPointer
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.linear import linear
from serenitymojo.ops.activations import silu
from serenitymojo.ops.embeddings import t_embedder
from serenitymojo.ops.tensor_algebra import reshape_owned, slice
from serenitymojo.models.klein.double_block import (
    StreamWeights, DoubleBlockWeights, ModVecs, ModVecsDevice,
)
from serenitymojo.models.klein.single_block import (
    SingleBlockWeights, SingleModVecs, SingleModVecsDevice,
)
from serenitymojo.models.klein.klein_stack import KleinStackBase


struct KleinStepModWeights(Movable):
    """Frozen timestep/modulation weights reused across every training step."""

    var t_in: Tensor
    var t_out: Tensor
    var img_mod: Tensor
    var txt_mod: Tensor
    var single_mod: Tensor
    var final_mod: Tensor   # final_layer.adaLN_modulation.1.weight [2D, D]

    def __init__(
        out self,
        var t_in: Tensor,
        var t_out: Tensor,
        var img_mod: Tensor,
        var txt_mod: Tensor,
        var single_mod: Tensor,
        var final_mod: Tensor,
    ):
        self.t_in = t_in^
        self.t_out = t_out^
        self.img_mod = img_mod^
        self.txt_mod = txt_mod^
        self.single_mod = single_mod^
        self.final_mod = final_mod^


# Read one named tensor from the safetensors as a host List[Float32] (casts up
# from the stored dtype — Klein base is BF16).
def _load_host_f32(st: SafeTensors, name: String, ctx: DeviceContext) raises -> List[Float32]:
    var info = st.tensor_info(name)
    var bytes = st.tensor_bytes(name)
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    var t = Tensor.from_view(tv, ctx)
    var t32 = cast_tensor(t, STDtype.F32, ctx)
    return t32.to_host(ctx)


# Dim 0 of a named tensor's stored shape (for deriving D/F/Dh from the weights).
def _dim0(st: SafeTensors, name: String) raises -> Int:
    var info = st.tensor_info(name)
    return Int(info.shape[0])


# A2: StreamWeights now uploads its 6 matrices to the device ONCE in __init__
# (TArc). The loader still reads host F32 lists, then derives D/F/Dh from the
# stored shapes and hands them to the constructor for the one-time upload:
#   wqkv [3D, D]  -> D  = dim1(wqkv) = dim0(wqkv)//3 ; here from wproj [D,D] dim0.
#   wgu  [2F, D]  -> F  = dim0(wgu)//2.
#   q_norm [Dh]   -> Dh = dim0(q_norm).
def _load_stream(
    st: SafeTensors, dp: String, stream: String, ctx: DeviceContext
) raises -> StreamWeights:
    var ap = dp + String(".") + stream + String("_attn")
    var mp = dp + String(".") + stream + String("_mlp")
    var D = _dim0(st, ap + String(".proj.weight"))        # wproj [D, D] -> D
    var F = _dim0(st, mp + String(".0.weight")) // 2       # wgu  [2F, D] -> F
    var Dh = _dim0(st, ap + String(".norm.query_norm.scale"))  # q_norm [Dh]
    return StreamWeights(
        _load_host_f32(st, ap + String(".qkv.weight"), ctx),               # wqkv
        _load_host_f32(st, ap + String(".proj.weight"), ctx),              # wproj
        _load_host_f32(st, mp + String(".0.weight"), ctx),                 # wgu
        _load_host_f32(st, mp + String(".2.weight"), ctx),                 # wd
        _load_host_f32(st, ap + String(".norm.query_norm.scale"), ctx),    # q_norm
        _load_host_f32(st, ap + String(".norm.key_norm.scale"), ctx),      # k_norm
        D, F, Dh, ctx,
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


# Load single block `block_idx`'s real weights into SingleBlockWeights.
# Keys: single_blocks.{bi}.linear1.weight, .linear2.weight,
#       .norm.query_norm.scale, .norm.key_norm.scale (BF16 -> F32 -> host).
# A2: SingleBlockWeights uploads its 4 matrices to the device ONCE in __init__.
# Dims from stored shapes: w2 [D, D+F] -> D = dim0(w2); F = dim1(w2) - D where
# dim1(w2) = dim0(w1)? no: derive D from w2 dim0, F from (w1 dim0 - 3D)/2... use
# the explicit shapes: w1 [3D+2F, D] (dim1=D), w2 [D, D+F] (dim0=D). So
#   D  = dim0(w2)
#   F  = (dim0(w1) - 3*D) // 2
#   Dh = dim0(q_norm)
def load_single_block_weights(
    st: SafeTensors, block_idx: Int, ctx: DeviceContext, keep_w2: Bool = True
) raises -> SingleBlockWeights:
    var sp = String("single_blocks.") + String(block_idx)
    var D = _dim0(st, sp + String(".linear2.weight"))         # w2 [D, D+F] -> D
    var F = (_dim0(st, sp + String(".linear1.weight")) - 3 * D) // 2  # w1 [3D+2F, D]
    var Dh = _dim0(st, sp + String(".norm.query_norm.scale"))
    return SingleBlockWeights(
        _load_host_f32(st, sp + String(".linear1.weight"), ctx),           # w1
        _load_host_f32(st, sp + String(".linear2.weight"), ctx),           # w2
        _load_host_f32(st, sp + String(".norm.query_norm.scale"), ctx),    # q_norm
        _load_host_f32(st, sp + String(".norm.key_norm.scale"), ctx),      # k_norm
        D, F, Dh, ctx, keep_w2,
    )


# ── shared base weights for the full stack (input proj + final layer) ─────────
# Loads img_in.weight, txt_in.weight, final_layer.linear.weight, and the two
# final-layer adaLN chunks (shift=chunk0, scale=chunk1) of
# final_layer.adaLN_modulation.1.weight applied to vec_silu. `d` is the inner dim.
# vec_silu [1,D] is the (single-timestep) modulation feature (see build_klein_modvecs).
# Dim 1 of a named tensor's stored shape.
def _dim1(st: SafeTensors, name: String) raises -> Int:
    var info = st.tensor_info(name)
    return Int(info.shape[1])


# A2: KleinStackBase uploads img_in/txt_in/final_lin/final_shift/final_scale to
# the device ONCE in __init__. The big input/output projections were re-uploaded
# every step before. Dims from stored shapes: img_in [D, in_ch], txt_in [D,
# txt_ch], final_lin [out_ch, D]. The final adaLN shift/scale are still computed
# on host here (small [D], frozen once with the seed sigma) then uploaded once.
def load_klein_stack_base(
    st: SafeTensors, vec_silu: List[Float32], d: Int, ctx: DeviceContext
) raises -> KleinStackBase:
    var in_ch = _dim1(st, String("img_in.weight"))     # img_in [D, in_ch]
    var txt_ch = _dim1(st, String("txt_in.weight"))    # txt_in [D, txt_ch]
    var out_ch = _dim0(st, String("final_layer.linear.weight"))  # final_lin [out_ch, D]
    var img_in = _load_host_f32(st, String("img_in.weight"), ctx)
    var txt_in = _load_host_f32(st, String("txt_in.weight"), ctx)
    var final_lin = _load_host_f32(st, String("final_layer.linear.weight"), ctx)
    # final adaLN: linear(vec_silu, final_mod_w) -> [1, 2D]; chunk 0=shift, 1=scale.
    var final_mod_w = _load_host_f32(st, String("final_layer.adaLN_modulation.1.weight"), ctx)
    var final_mod = _linear_row(vec_silu, final_mod_w, d, 2 * d, ctx)   # [2D]
    var final_shift = _chunk(final_mod, 0, d)
    var final_scale = _chunk(final_mod, 1, d)
    return KleinStackBase(
        img_in^, txt_in^, final_lin^, final_shift^, final_scale^,
        d, in_ch, txt_ch, out_ch, ctx,
    )


# linear of a single [in_dim] row by a [out_dim, in_dim] weight -> [out_dim].
def _linear_row(
    x: List[Float32], w: List[Float32], in_dim: Int, out_dim: Int, ctx: DeviceContext
) raises -> List[Float32]:
    var no_bias = Optional[Tensor](None)
    return linear(
        Tensor.from_host(x, [1, in_dim], STDtype.F32, ctx),
        Tensor.from_host(w, [out_dim, in_dim], STDtype.F32, ctx),
        no_bias^, ctx,
    ).to_host(ctx)


# extract chunk `idx` of width `d` from a flat [k*d] list.
def _chunk(src: List[Float32], idx: Int, d: Int) -> List[Float32]:
    var o = List[Float32]()
    var base = idx * d
    for i in range(d):
        o.append(src[base + i])
    return o^


# Build the SHARED modulation feature vec_silu = silu(t_embedder(timestep, ...)).
# timestep: [1] (single sample). Returns vec_silu [1, D] as a host list.
def build_klein_vec_silu(
    st: SafeTensors, timestep: Tensor, timestep_dim: Int, d: Int, ctx: DeviceContext
) raises -> List[Float32]:
    var t_in = _load_tensor(st, String("time_in.in_layer.weight"), ctx)
    var t_out = _load_tensor(st, String("time_in.out_layer.weight"), ctx)
    var vec = t_embedder(
        timestep, timestep_dim, t_in, Optional[Tensor](None), t_out, Optional[Tensor](None), ctx
    )
    var vec_silu = silu(vec, ctx)
    return vec_silu.to_host(ctx)


# Build the shared double/single modulation vectors from vec_silu and the real
# modulation linears: mod = linear(vec_silu, mod_w); chunk into ModVecs/SingleModVecs.
def build_klein_double_modvecs(
    st: SafeTensors, vec_silu: List[Float32], stream: String, d: Int, ctx: DeviceContext
) raises -> ModVecs:
    var mod_w = _load_host_f32(
        st, String("double_stream_modulation_") + stream + String(".lin.weight"), ctx
    )
    var mod = _linear_row(vec_silu, mod_w, d, 6 * d, ctx)   # [6D]
    return ModVecs(
        _chunk(mod, 0, d), _chunk(mod, 1, d), _chunk(mod, 2, d),
        _chunk(mod, 3, d), _chunk(mod, 4, d), _chunk(mod, 5, d),
    )


def build_klein_single_modvecs(
    st: SafeTensors, vec_silu: List[Float32], d: Int, ctx: DeviceContext
) raises -> SingleModVecs:
    var mod_w = _load_host_f32(st, String("single_stream_modulation.lin.weight"), ctx)
    var mod = _linear_row(vec_silu, mod_w, d, 3 * d, ctx)   # [3D]
    return SingleModVecs(_chunk(mod, 0, d), _chunk(mod, 1, d), _chunk(mod, 2, d))


def load_klein_step_mod_weights(
    st: SafeTensors, d: Int, ctx: DeviceContext
) raises -> KleinStepModWeights:
    """Load frozen per-step modulation weights once for the timed loop.

    Timestep MLP weights stay in their checkpoint dtype (BF16). The downstream
    modulation weights are promoted to F32 once so the math matches the legacy
    host `_linear_row` path that used F32 lists.
    """
    var img_h = _load_host_f32(
        st, String("double_stream_modulation_img.lin.weight"), ctx
    )
    var txt_h = _load_host_f32(
        st, String("double_stream_modulation_txt.lin.weight"), ctx
    )
    var single_h = _load_host_f32(
        st, String("single_stream_modulation.lin.weight"), ctx
    )
    var final_h = _load_host_f32(
        st, String("final_layer.adaLN_modulation.1.weight"), ctx
    )
    return KleinStepModWeights(
        _load_tensor(st, String("time_in.in_layer.weight"), ctx),
        _load_tensor(st, String("time_in.out_layer.weight"), ctx),
        Tensor.from_host(img_h^, [6 * d, d], STDtype.F32, ctx),
        Tensor.from_host(txt_h^, [6 * d, d], STDtype.F32, ctx),
        Tensor.from_host(single_h^, [3 * d, d], STDtype.F32, ctx),
        Tensor.from_host(final_h^, [2 * d, d], STDtype.F32, ctx),
    )


def build_klein_vec_silu_device(
    weights: KleinStepModWeights,
    timestep: Tensor,
    timestep_dim: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    var vec = t_embedder(
        timestep,
        timestep_dim,
        weights.t_in,
        Optional[Tensor](None),
        weights.t_out,
        Optional[Tensor](None),
        ctx,
    )
    var vec_silu = silu(vec, ctx)
    if vec_silu.dtype() == STDtype.F32:
        return vec_silu^
    return cast_tensor(vec_silu, STDtype.F32, ctx)


def _modvec_from_device(mod: Tensor, d: Int, ctx: DeviceContext) raises -> ModVecs:
    var host = mod.to_host(ctx)
    return ModVecs(
        _chunk(host, 0, d), _chunk(host, 1, d), _chunk(host, 2, d),
        _chunk(host, 3, d), _chunk(host, 4, d), _chunk(host, 5, d),
    )


def _single_modvec_from_device(
    mod: Tensor, d: Int, ctx: DeviceContext
) raises -> SingleModVecs:
    var host = mod.to_host(ctx)
    return SingleModVecs(_chunk(host, 0, d), _chunk(host, 1, d), _chunk(host, 2, d))


def _chunk_tensor_1d(
    x: Tensor, start_chunk: Int, d: Int, ctx: DeviceContext
) raises -> Tensor:
    var chunk2d = slice(x, 1, start_chunk * d, d, ctx)
    return reshape_owned(chunk2d^, [d])


def _modvec_device_from_tensor(
    mod: Tensor, d: Int, ctx: DeviceContext
) raises -> ModVecsDevice:
    var shift1 = _chunk_tensor_1d(mod, 0, d, ctx)
    var scale1 = _chunk_tensor_1d(mod, 1, d, ctx)
    var gate1 = _chunk_tensor_1d(mod, 2, d, ctx)
    var shift2 = _chunk_tensor_1d(mod, 3, d, ctx)
    var scale2 = _chunk_tensor_1d(mod, 4, d, ctx)
    var gate2 = _chunk_tensor_1d(mod, 5, d, ctx)
    return ModVecsDevice(
        ArcPointer[Tensor](shift1^),
        ArcPointer[Tensor](scale1^),
        ArcPointer[Tensor](gate1^),
        ArcPointer[Tensor](shift2^),
        ArcPointer[Tensor](scale2^),
        ArcPointer[Tensor](gate2^),
    )


def _single_modvec_device_from_tensor(
    mod: Tensor, d: Int, ctx: DeviceContext
) raises -> SingleModVecsDevice:
    var shift = _chunk_tensor_1d(mod, 0, d, ctx)
    var scale = _chunk_tensor_1d(mod, 1, d, ctx)
    var gate = _chunk_tensor_1d(mod, 2, d, ctx)
    return SingleModVecsDevice(
        ArcPointer[Tensor](shift^),
        ArcPointer[Tensor](scale^),
        ArcPointer[Tensor](gate^),
    )


def build_klein_step_mods_cached(
    weights: KleinStepModWeights,
    sigma: Float32,
    timestep_dim: Int,
    d: Int,
    ctx: DeviceContext,
) raises -> Tuple[ModVecs, ModVecs, SingleModVecs]:
    var tvals = List[Float32]()
    tvals.append(sigma * Float32(1000.0))
    var tsh = List[Int]()
    tsh.append(1)
    var ts = Tensor.from_host(tvals, tsh^, STDtype.F32, ctx)
    var vec_silu = build_klein_vec_silu_device(weights, ts, timestep_dim, ctx)

    var no_bias_img = Optional[Tensor](None)
    var img_mod = linear(vec_silu, weights.img_mod, no_bias_img^, ctx)
    var no_bias_txt = Optional[Tensor](None)
    var txt_mod = linear(vec_silu, weights.txt_mod, no_bias_txt^, ctx)
    var no_bias_single = Optional[Tensor](None)
    var single_mod = linear(vec_silu, weights.single_mod, no_bias_single^, ctx)
    return (
        _modvec_from_device(img_mod, d, ctx),
        _modvec_from_device(txt_mod, d, ctx),
        _single_modvec_from_device(single_mod, d, ctx),
    )


def build_klein_step_mods_device_cached(
    weights: KleinStepModWeights,
    sigma: Float32,
    timestep_dim: Int,
    d: Int,
    ctx: DeviceContext,
) raises -> Tuple[ModVecsDevice, ModVecsDevice, SingleModVecsDevice, ArcPointer[Tensor], ArcPointer[Tensor]]:
    var tvals = List[Float32]()
    tvals.append(sigma * Float32(1000.0))
    var tsh = List[Int]()
    tsh.append(1)
    var ts = Tensor.from_host(tvals, tsh^, STDtype.F32, ctx)
    var vec_silu = build_klein_vec_silu_device(weights, ts, timestep_dim, ctx)

    var no_bias_img = Optional[Tensor](None)
    var img_mod = linear(vec_silu, weights.img_mod, no_bias_img^, ctx)
    var no_bias_txt = Optional[Tensor](None)
    var txt_mod = linear(vec_silu, weights.txt_mod, no_bias_txt^, ctx)
    var no_bias_single = Optional[Tensor](None)
    var single_mod = linear(vec_silu, weights.single_mod, no_bias_single^, ctx)
    # per-step final-layer adaLN mod (FIX: was static sigma=0.5). final_mod [1,2D];
    # chunk0=shift, chunk1=scale (matches the static load_klein_stack_base path).
    var no_bias_final = Optional[Tensor](None)
    var final_mod = linear(vec_silu, weights.final_mod, no_bias_final^, ctx)
    var final_shift = _chunk_tensor_1d(final_mod, 0, d, ctx)   # [d] cols 0:d
    var final_scale = _chunk_tensor_1d(final_mod, 1, d, ctx)   # [d] cols d:2d
    return (
        _modvec_device_from_tensor(img_mod, d, ctx),
        _modvec_device_from_tensor(txt_mod, d, ctx),
        _single_modvec_device_from_tensor(single_mod, d, ctx),
        ArcPointer[Tensor](final_shift^),
        ArcPointer[Tensor](final_scale^),
    )


# load a named tensor as a device Tensor (BF16 stored) without casting to host.
def _load_tensor(st: SafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    var info = st.tensor_info(name)
    var bytes = st.tensor_bytes(name)
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    return Tensor.from_view(tv, ctx)
