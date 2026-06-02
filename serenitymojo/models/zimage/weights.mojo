# models/zimage/weights.mojo — real safetensors -> Z-Image (NextDiT) training
# block weights.
#
# Loads ONE Z-Image main-layer block's weights from the real sharded safetensors
# into the ZImageBlockWeights struct that block.mojo's forward/backward consume.
# The inference path (models/dit/zimage_dit.mojo `_attention`/`_feed_forward`/
# `_block`) reads these exact keys; this is the cross-pollination into the
# TRAINING weight structs. Mirrors models/ernie/weights.mojo.
#
# Key layout (per main layer `bi`), from zimage_dit.mojo + zimage_nextdit.rs:
#   layers.{bi}.attention.to_q.weight        -> wq       [dim, dim]
#   layers.{bi}.attention.to_k.weight        -> wk       [dim, dim]
#   layers.{bi}.attention.to_v.weight        -> wv       [dim, dim]
#   layers.{bi}.attention.to_out.0.weight    -> wo       [dim, dim]
#   layers.{bi}.attention.norm_q.weight      -> q_norm   [head_dim]  (per-head RMSNorm)
#   layers.{bi}.attention.norm_k.weight      -> k_norm   [head_dim]
#   layers.{bi}.attention_norm1.weight       -> n1       [dim]       (pre-attn RMSNorm)
#   layers.{bi}.attention_norm2.weight       -> n2       [dim]       (post-attn RMSNorm)
#   layers.{bi}.ffn_norm1.weight             -> fn1      [dim]       (pre-ffn RMSNorm)
#   layers.{bi}.ffn_norm2.weight             -> fn2      [dim]       (post-ffn RMSNorm)
#   layers.{bi}.feed_forward.w1.weight       -> w1       [ffn, dim]  (SwiGLU gate)
#   layers.{bi}.feed_forward.w3.weight       -> w3       [ffn, dim]  (SwiGLU up)
#   layers.{bi}.feed_forward.w2.weight       -> w2       [dim, ffn]  (SwiGLU down)
#   layers.{bi}.adaLN_modulation.0.weight    -> adaln_w  [4*dim, t_dim]
#   layers.{bi}.adaLN_modulation.0.bias      -> adaln_b  [4*dim]
#
# NOTE on weight orientation: safetensors store nn.Linear weight as [out, in].
# The Mojo `linear(x, W, bias)` computes x @ Wᵀ, so the stored [out, in] layout is
# consumed DIRECTLY (no pre-transpose) — same contract as ernie/weights.mojo.
#
# The adaLN modulation is PER-LAYER (each Z-Image block has its own
# adaLN_modulation.0 Linear that maps the shared timestep embedding [t_dim] ->
# 4*dim, then chunks into scale_msa | gate_msa | scale_mlp | gate_mlp). The 4
# RAW modulation vectors (pre tanh / pre 1+) are what the block consumes; the
# block applies tanh (gates) and +1 (scales) internally so it owns those grads.

from std.collections import List, Optional
from std.gpu.host import DeviceContext
from std.memory import ArcPointer
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.ops.cast import cast_tensor


comptime TArc = ArcPointer[Tensor]


# ── per-block weights (device-resident, uploaded ONCE; frozen base) ───────────
struct ZImageBlockWeights(Copyable, Movable):
    var n1: TArc         # [dim]              attention_norm1 (pre-attn RMSNorm)
    var wq: TArc         # [dim, dim]         attention.to_q
    var wk: TArc         # [dim, dim]         attention.to_k
    var wv: TArc         # [dim, dim]         attention.to_v
    var wo: TArc         # [dim, dim]         attention.to_out.0
    var q_norm: TArc     # [head_dim]         per-head RMSNorm scale (Q)
    var k_norm: TArc     # [head_dim]         per-head RMSNorm scale (K)
    var n2: TArc         # [dim]              attention_norm2 (post-attn RMSNorm)
    var fn1: TArc        # [dim]              ffn_norm1 (pre-ffn RMSNorm)
    var w1: TArc         # [ffn, dim]         feed_forward.w1 (SwiGLU gate)
    var w3: TArc         # [ffn, dim]         feed_forward.w3 (SwiGLU up)
    var w2: TArc         # [dim, ffn]         feed_forward.w2 (SwiGLU down)
    var fn2: TArc        # [dim]              ffn_norm2 (post-ffn RMSNorm)

    def __init__(
        out self,
        var n1: TArc,
        var wq: TArc, var wk: TArc, var wv: TArc, var wo: TArc,
        var q_norm: TArc, var k_norm: TArc,
        var n2: TArc, var fn1: TArc,
        var w1: TArc, var w3: TArc, var w2: TArc, var fn2: TArc,
    ):
        self.n1 = n1^
        self.wq = wq^
        self.wk = wk^
        self.wv = wv^
        self.wo = wo^
        self.q_norm = q_norm^
        self.k_norm = k_norm^
        self.n2 = n2^
        self.fn1 = fn1^
        self.w1 = w1^
        self.w3 = w3^
        self.w2 = w2^
        self.fn2 = fn2^


# Read one named tensor from the sharded safetensors as a device F32 Tensor at
# its real stored shape (casts up from BF16). Mirrors ernie _load_f32_device.
def _load_f32_device(
    st: ShardedSafeTensors, name: String, ctx: DeviceContext
) raises -> Tensor:
    var info = st.tensor_info(name)
    var bytes = st.tensor_bytes(name)
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    var t = Tensor.from_view(tv, ctx)
    return cast_tensor(t, STDtype.F32, ctx)


# Load ONE block's real weights given an explicit stream prefix
# (noise_refiner.{i} / context_refiner.{i} / layers.{i}). The diffusers
# transformer dir (/home/alex/.serenity/models/zimage_base/transformer) stores
# UNFUSED to_q/to_k/to_v/to_out.0 + per-head norm_q/norm_k under each stream,
# matching the training stack's separate-projection ZImageBlockWeights. Used by
# the real trainer to populate nr/cr/main block lists. Mirrors the layers-only
# loader below.
def load_zimage_block_weights_prefixed(
    st: ShardedSafeTensors, prefix: String, ctx: DeviceContext
) raises -> ZImageBlockWeights:
    var ap = prefix + String(".attention")
    var fp = prefix + String(".feed_forward")
    return ZImageBlockWeights(
        TArc(_load_f32_device(st, prefix + String(".attention_norm1.weight"), ctx)),
        TArc(_load_f32_device(st, ap + String(".to_q.weight"), ctx)),
        TArc(_load_f32_device(st, ap + String(".to_k.weight"), ctx)),
        TArc(_load_f32_device(st, ap + String(".to_v.weight"), ctx)),
        TArc(_load_f32_device(st, ap + String(".to_out.0.weight"), ctx)),
        TArc(_load_f32_device(st, ap + String(".norm_q.weight"), ctx)),
        TArc(_load_f32_device(st, ap + String(".norm_k.weight"), ctx)),
        TArc(_load_f32_device(st, prefix + String(".attention_norm2.weight"), ctx)),
        TArc(_load_f32_device(st, prefix + String(".ffn_norm1.weight"), ctx)),
        TArc(_load_f32_device(st, fp + String(".w1.weight"), ctx)),
        TArc(_load_f32_device(st, fp + String(".w3.weight"), ctx)),
        TArc(_load_f32_device(st, fp + String(".w2.weight"), ctx)),
        TArc(_load_f32_device(st, prefix + String(".ffn_norm2.weight"), ctx)),
    )


# Load main-layer block `block_idx`'s real weights (device-resident).
def load_zimage_block_weights(
    st: ShardedSafeTensors, block_idx: Int, ctx: DeviceContext
) raises -> ZImageBlockWeights:
    var p = String("layers.") + String(block_idx)
    var ap = p + String(".attention")
    var fp = p + String(".feed_forward")
    return ZImageBlockWeights(
        TArc(_load_f32_device(st, p + String(".attention_norm1.weight"), ctx)),
        TArc(_load_f32_device(st, ap + String(".to_q.weight"), ctx)),
        TArc(_load_f32_device(st, ap + String(".to_k.weight"), ctx)),
        TArc(_load_f32_device(st, ap + String(".to_v.weight"), ctx)),
        TArc(_load_f32_device(st, ap + String(".to_out.0.weight"), ctx)),
        TArc(_load_f32_device(st, ap + String(".norm_q.weight"), ctx)),
        TArc(_load_f32_device(st, ap + String(".norm_k.weight"), ctx)),
        TArc(_load_f32_device(st, p + String(".attention_norm2.weight"), ctx)),
        TArc(_load_f32_device(st, p + String(".ffn_norm1.weight"), ctx)),
        TArc(_load_f32_device(st, fp + String(".w1.weight"), ctx)),
        TArc(_load_f32_device(st, fp + String(".w3.weight"), ctx)),
        TArc(_load_f32_device(st, fp + String(".w2.weight"), ctx)),
        TArc(_load_f32_device(st, p + String(".ffn_norm2.weight"), ctx)),
    )


# Load ALL `num_layers` main-layer blocks into a List[ZImageBlockWeights].
def load_zimage_all_blocks(
    st: ShardedSafeTensors, num_layers: Int, ctx: DeviceContext
) raises -> List[ZImageBlockWeights]:
    var blocks = List[ZImageBlockWeights]()
    for i in range(num_layers):
        blocks.append(load_zimage_block_weights(st, i, ctx))
    return blocks^


# Assert one named tensor loads at the expected shape (real header read + shape
# compare; skeptic-checkable receipt). Mirrors ernie _expect_shape.
def _expect_shape(
    st: ShardedSafeTensors, name: String, var want: List[Int]
) raises:
    var info = st.tensor_info(name)
    var got = List[Int]()
    for i in range(len(info.shape)):
        got.append(Int(info.shape[i]))
    if len(got) != len(want):
        raise Error(String("ZImage ") + name + String(" rank mismatch"))
    for i in range(len(want)):
        if got[i] != want[i]:
            raise Error(
                String("ZImage ") + name + String(" shape mismatch at dim ")
                + String(i) + String(": got ") + String(got[i])
                + String(" want ") + String(want[i])
            )
    print("  OK", name, "[", end="")
    for i in range(len(got)):
        print(got[i], end="," if i + 1 < len(got) else "")
    print("]")


# Verify a representative sample of main-layer block tensors load at the expected
# shapes (skeptic-checkable receipt: real header reads with shape asserts, RC=0 on
# success). Checks block 0 + the deepest main layer.
def verify_zimage_block_shapes(
    st: ShardedSafeTensors, num_layers: Int,
    dim: Int, head_dim: Int, ffn: Int, t_dim: Int,
) raises:
    var probe = List[Int]()
    probe.append(0)
    probe.append(num_layers - 1)
    for pi in range(len(probe)):
        var i = probe[pi]
        var p = String("layers.") + String(i)
        var ap = p + String(".attention")
        var fp = p + String(".feed_forward")
        _expect_shape(st, p + String(".attention_norm1.weight"), [dim])
        _expect_shape(st, p + String(".attention_norm2.weight"), [dim])
        _expect_shape(st, p + String(".ffn_norm1.weight"), [dim])
        _expect_shape(st, p + String(".ffn_norm2.weight"), [dim])
        _expect_shape(st, ap + String(".to_q.weight"), [dim, dim])
        _expect_shape(st, ap + String(".to_out.0.weight"), [dim, dim])
        _expect_shape(st, ap + String(".norm_q.weight"), [head_dim])
        _expect_shape(st, fp + String(".w1.weight"), [ffn, dim])
        _expect_shape(st, fp + String(".w2.weight"), [dim, ffn])
        _expect_shape(st, p + String(".adaLN_modulation.0.weight"), [4 * dim, t_dim])
        _expect_shape(st, p + String(".adaLN_modulation.0.bias"), [4 * dim])
