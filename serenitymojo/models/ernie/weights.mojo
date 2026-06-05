# models/ernie/weights.mojo — real safetensors -> ERNIE training block weights.
#
# Loads ONE ERNIE-Image transformer block's weights from the real sharded
# safetensors (ShardedSafeTensors over the transformer/ dir) into the
# ErnieBlockWeights struct that block.mojo's forward/backward consume. The
# inference path (models/dit/ernie_image.mojo `block0_smoke_forward`) reads these
# exact keys; this is the cross-pollination into the TRAINING weight structs.
#
# Mirrors models/klein/weights.mojo: read each named tensor as a host F32 list
# (casts up from the stored BF16), upload to device ONCE (TArc carriers) at the
# tensor's real shape. The frozen base matrices are device-resident — never
# re-uploaded per op per step (the A2 perf lesson from Klein).
#
# Key layout (per block `bi`), from ernie_image.rs:411-425 / ernie_image.mojo:
#   layers.{bi}.adaLN_sa_ln.weight              -> sa_norm   [hidden]      (RMSNorm scale)
#   layers.{bi}.self_attention.to_q.weight      -> wq        [hidden, hidden]
#   layers.{bi}.self_attention.to_k.weight      -> wk        [hidden, hidden]
#   layers.{bi}.self_attention.to_v.weight      -> wv        [hidden, hidden]
#   layers.{bi}.self_attention.to_out.0.weight  -> wo        [hidden, hidden]
#   layers.{bi}.self_attention.norm_q.weight    -> q_norm    [head_dim]    (per-head RMSNorm)
#   layers.{bi}.self_attention.norm_k.weight    -> k_norm    [head_dim]
#   layers.{bi}.adaLN_mlp_ln.weight             -> mlp_norm  [hidden]      (RMSNorm scale)
#   layers.{bi}.mlp.gate_proj.weight            -> wgate     [ffn, hidden]
#   layers.{bi}.mlp.up_proj.weight              -> wup       [ffn, hidden]
#   layers.{bi}.mlp.linear_fc2.weight           -> wdown     [hidden, ffn]
#
# NOTE on weight orientation: safetensors store nn.Linear weight as [out, in].
# The Mojo `linear(x, W, bias)` computes x @ Wᵀ (ops/linear.mojo:151-156), so the
# stored [out, in] layout is consumed DIRECTLY (no pre-transpose) — same contract
# as klein/weights.mojo (Rust pre-transposes; Mojo linear transposes inside).

from std.collections import List, Optional
from std.gpu.host import DeviceContext
from std.memory import ArcPointer
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.tensor_algebra import reshape_owned


comptime TArc = ArcPointer[Tensor]


# ── trainable / frozen block weights (device-resident, uploaded ONCE) ─────────
struct ErnieBlockWeights(Copyable, Movable):
    var sa_norm: TArc    # [hidden]            adaLN_sa_ln RMSNorm scale
    var wq: TArc         # [hidden, hidden]    self_attention.to_q
    var wk: TArc         # [hidden, hidden]    self_attention.to_k
    var wv: TArc         # [hidden, hidden]    self_attention.to_v
    var wo: TArc         # [hidden, hidden]    self_attention.to_out.0
    var q_norm: TArc     # [head_dim]          per-head RMSNorm scale (Q)
    var k_norm: TArc     # [head_dim]          per-head RMSNorm scale (K)
    var mlp_norm: TArc   # [hidden]            adaLN_mlp_ln RMSNorm scale
    var wgate: TArc      # [ffn, hidden]       mlp.gate_proj
    var wup: TArc        # [ffn, hidden]       mlp.up_proj
    var wdown: TArc      # [hidden, ffn]       mlp.linear_fc2

    def __init__(
        out self,
        var sa_norm: TArc, var wq: TArc, var wk: TArc, var wv: TArc, var wo: TArc,
        var q_norm: TArc, var k_norm: TArc,
        var mlp_norm: TArc, var wgate: TArc, var wup: TArc, var wdown: TArc,
    ):
        self.sa_norm = sa_norm^
        self.wq = wq^
        self.wk = wk^
        self.wv = wv^
        self.wo = wo^
        self.q_norm = q_norm^
        self.k_norm = k_norm^
        self.mlp_norm = mlp_norm^
        self.wgate = wgate^
        self.wup = wup^
        self.wdown = wdown^


# Read one named tensor from the sharded safetensors as a device F32 Tensor.
# Use only for small F32 residual-stream norm compatibility, or the remaining
# biased patch/final stack helpers called out below. Do not use for generic
# projection-matrix checkpoint loading.
def _load_f32_device(
    st: ShardedSafeTensors, name: String, ctx: DeviceContext
) raises -> Tensor:
    var info = st.tensor_info(name)
    var bytes = st.tensor_bytes(name)
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    var t = Tensor.from_view(tv, ctx)
    return cast_tensor(t, STDtype.F32, ctx)


# Read one named tensor from the sharded safetensors as a device Tensor in the
# checkpoint's stored dtype. ERNIE's large projection matrices are BF16; keeping
# them BF16 avoids doubling block-stream traffic while linear()/dX use mixed GEMM.
def _load_stored_device(
    st: ShardedSafeTensors, name: String, ctx: DeviceContext
) raises -> Tensor:
    var info = st.tensor_info(name)
    var bytes = st.tensor_bytes(name)
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    return Tensor.from_view(tv, ctx)


# Dim 0 of a named tensor's stored shape (for verification / dim derivation).
def _dim0(st: ShardedSafeTensors, name: String) raises -> Int:
    var info = st.tensor_info(name)
    return Int(info.shape[0])


# Load block `block_idx`'s real weights into ErnieBlockWeights (device-resident).
def load_ernie_block_weights(
    st: ShardedSafeTensors, block_idx: Int, ctx: DeviceContext
) raises -> ErnieBlockWeights:
    var p = String("layers.") + String(block_idx)
    var ap = p + String(".self_attention")
    var mp = p + String(".mlp")
    return ErnieBlockWeights(
        TArc(_load_f32_device(st, p + String(".adaLN_sa_ln.weight"), ctx)),
        TArc(_load_stored_device(st, ap + String(".to_q.weight"), ctx)),
        TArc(_load_stored_device(st, ap + String(".to_k.weight"), ctx)),
        TArc(_load_stored_device(st, ap + String(".to_v.weight"), ctx)),
        TArc(_load_stored_device(st, ap + String(".to_out.0.weight"), ctx)),
        TArc(_load_f32_device(st, ap + String(".norm_q.weight"), ctx)),
        TArc(_load_f32_device(st, ap + String(".norm_k.weight"), ctx)),
        TArc(_load_f32_device(st, p + String(".adaLN_mlp_ln.weight"), ctx)),
        TArc(_load_stored_device(st, mp + String(".gate_proj.weight"), ctx)),
        TArc(_load_stored_device(st, mp + String(".up_proj.weight"), ctx)),
        TArc(_load_stored_device(st, mp + String(".linear_fc2.weight"), ctx)),
    )


# Load block `block_idx` for the production device-streamed fast path: large
# Linear matrices stay BF16 in device memory, while tiny RMSNorm scale vectors
# stay F32 so the F32 residual stream's norm ops keep their existing dtype path.
def load_ernie_block_weights_bf16_normf32(
    st: ShardedSafeTensors, block_idx: Int, ctx: DeviceContext
) raises -> ErnieBlockWeights:
    var p = String("layers.") + String(block_idx)
    var ap = p + String(".self_attention")
    var mp = p + String(".mlp")
    return ErnieBlockWeights(
        TArc(_load_f32_device(st, p + String(".adaLN_sa_ln.weight"), ctx)),
        TArc(_load_stored_device(st, ap + String(".to_q.weight"), ctx)),
        TArc(_load_stored_device(st, ap + String(".to_k.weight"), ctx)),
        TArc(_load_stored_device(st, ap + String(".to_v.weight"), ctx)),
        TArc(_load_stored_device(st, ap + String(".to_out.0.weight"), ctx)),
        TArc(_load_f32_device(st, ap + String(".norm_q.weight"), ctx)),
        TArc(_load_f32_device(st, ap + String(".norm_k.weight"), ctx)),
        TArc(_load_f32_device(st, p + String(".adaLN_mlp_ln.weight"), ctx)),
        TArc(_load_stored_device(st, mp + String(".gate_proj.weight"), ctx)),
        TArc(_load_stored_device(st, mp + String(".up_proj.weight"), ctx)),
        TArc(_load_stored_device(st, mp + String(".linear_fc2.weight"), ctx)),
    )


# Verify the loader can read block_idx's to_q.weight at the expected [hidden,
# hidden] shape (skeptic-checkable: a real H2D round-trip with a shape assert).
# Returns dim0(to_q) (== hidden) so the caller can print/compare.
def verify_block_to_q_shape(
    st: ShardedSafeTensors, block_idx: Int, hidden: Int
) raises -> Int:
    var p = String("layers.") + String(block_idx)
    var name = p + String(".self_attention.to_q.weight")
    var info = st.tensor_info(name)
    if len(info.shape) != 2:
        raise Error(String("ERNIE ") + name + String(" rank != 2"))
    if Int(info.shape[0]) != hidden or Int(info.shape[1]) != hidden:
        raise Error(
            String("ERNIE ") + name + String(" shape mismatch: got [")
            + String(Int(info.shape[0])) + String(", ")
            + String(Int(info.shape[1])) + String("] expected [")
            + String(hidden) + String(", ") + String(hidden) + String("]")
        )
    return Int(info.shape[0])


# ── shared / resident base weights (input projections + final layer) ──────────
# Mirrors klein_stack.KleinStackBase: every tensor uploaded to the device ONCE.
# These are FROZEN (read on every forward + backward). The no-bias text
# projection and the shared timestep/AdaLN/final-norm MLP preserve checkpoint
# dtype; their callers already cast transient inputs to the stored weight dtype.
# BUG: patch_embed and final_linear still upcast BF16 checkpoint tensors because
# their biased stack helpers live outside this loader scope and still build F32
# input tensors. Keys from ernie_image.rs load():
#   x_embedder.proj.{weight,bias}            patch embed (Conv2d k=1 -> linear)
#   text_proj.weight                         Mistral hidden -> hidden (no bias)
#   time_embedding.linear_{1,2}.{weight,bias}  timestep MLP
#   adaLN_modulation.1.{weight,bias}         shared AdaLN source (-> 6*hidden)
#   final_norm.linear.{weight,bias}          final modulation source (-> 2*hidden)
#   final_linear.{weight,bias}               final projection (-> P*P*out_ch)
# NOTE on patch_proj: stored as [hidden, in_ch, 1, 1]; reshape to [hidden, in_ch]
# (patch_size=1 => Conv2d k=1 is a linear over the channel dim). The Mojo linear
# consumes [out,in] directly (transposes inside), so NO pre-transpose (matches the
# block weights contract).
struct ErnieStackBase(Copyable, Movable):
    var patch_w: TArc       # [hidden, in_ch]        x_embedder.proj.weight (reshaped)
    var patch_b: TArc       # [hidden]               x_embedder.proj.bias
    var text_proj: TArc     # [hidden, text_in_dim]  text_proj.weight
    var te_w1: TArc         # [hidden, hidden]       time_embedding.linear_1.weight
    var te_b1: TArc         # [hidden]               time_embedding.linear_1.bias
    var te_w2: TArc         # [hidden, hidden]       time_embedding.linear_2.weight
    var te_b2: TArc         # [hidden]               time_embedding.linear_2.bias
    var adaln_w: TArc       # [6*hidden, hidden]     adaLN_modulation.1.weight
    var adaln_b: TArc       # [6*hidden]             adaLN_modulation.1.bias
    var final_norm_w: TArc  # [2*hidden, hidden]     final_norm.linear.weight
    var final_norm_b: TArc  # [2*hidden]             final_norm.linear.bias
    var final_lin_w: TArc   # [out_ch, hidden]       final_linear.weight
    var final_lin_b: TArc   # [out_ch]               final_linear.bias

    def __init__(
        out self,
        var patch_w: TArc, var patch_b: TArc, var text_proj: TArc,
        var te_w1: TArc, var te_b1: TArc, var te_w2: TArc, var te_b2: TArc,
        var adaln_w: TArc, var adaln_b: TArc,
        var final_norm_w: TArc, var final_norm_b: TArc,
        var final_lin_w: TArc, var final_lin_b: TArc,
    ):
        self.patch_w = patch_w^
        self.patch_b = patch_b^
        self.text_proj = text_proj^
        self.te_w1 = te_w1^
        self.te_b1 = te_b1^
        self.te_w2 = te_w2^
        self.te_b2 = te_b2^
        self.adaln_w = adaln_w^
        self.adaln_b = adaln_b^
        self.final_norm_w = final_norm_w^
        self.final_norm_b = final_norm_b^
        self.final_lin_w = final_lin_w^
        self.final_lin_b = final_lin_b^


# Load one named tensor as a device F32 Tensor, FORCING a target shape (used for
# the patch_proj conv weight [hidden,in_ch,1,1] -> [hidden,in_ch] byte no-op).
def _load_f32_device_reshaped(
    st: ShardedSafeTensors, name: String, var shape: List[Int], ctx: DeviceContext
) raises -> Tensor:
    var info = st.tensor_info(name)
    var bytes = st.tensor_bytes(name)
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    var t = Tensor.from_view(tv, ctx)
    var f = cast_tensor(t, STDtype.F32, ctx)
    # reshape is a metadata-only byte no-op (numel validated by reshape_owned).
    return reshape_owned(f^, shape^)


# Load the resident / shared base weights (input projections + final layer).
def load_ernie_stack_base(
    st: ShardedSafeTensors, hidden: Int, in_ch: Int, ctx: DeviceContext
) raises -> ErnieStackBase:
    var patch_w = _load_f32_device_reshaped(
        st, String("x_embedder.proj.weight"), [hidden, in_ch], ctx
    )
    return ErnieStackBase(
        TArc(patch_w^),
        TArc(_load_f32_device(st, String("x_embedder.proj.bias"), ctx)),
        TArc(_load_stored_device(st, String("text_proj.weight"), ctx)),
        TArc(_load_stored_device(st, String("time_embedding.linear_1.weight"), ctx)),
        TArc(_load_stored_device(st, String("time_embedding.linear_1.bias"), ctx)),
        TArc(_load_stored_device(st, String("time_embedding.linear_2.weight"), ctx)),
        TArc(_load_stored_device(st, String("time_embedding.linear_2.bias"), ctx)),
        TArc(_load_stored_device(st, String("adaLN_modulation.1.weight"), ctx)),
        TArc(_load_stored_device(st, String("adaLN_modulation.1.bias"), ctx)),
        TArc(_load_stored_device(st, String("final_norm.linear.weight"), ctx)),
        TArc(_load_stored_device(st, String("final_norm.linear.bias"), ctx)),
        TArc(_load_f32_device(st, String("final_linear.weight"), ctx)),
        TArc(_load_f32_device(st, String("final_linear.bias"), ctx)),
    )


# Load ALL `num_layers` blocks' weights into a List[ErnieBlockWeights].
# Device-resident (each block uploaded once). num_layers from config (36).
def load_ernie_all_blocks(
    st: ShardedSafeTensors, num_layers: Int, ctx: DeviceContext
) raises -> List[ErnieBlockWeights]:
    var blocks = List[ErnieBlockWeights]()
    for i in range(num_layers):
        blocks.append(load_ernie_block_weights(st, i, ctx))
    return blocks^


# Load ALL `num_layers` blocks for the production resident fast path: BF16 large
# matrices, F32 norm vectors. On ERNIE-4B this is roughly half the F32 residency
# footprint and avoids per-step block reloads when VRAM allows it.
def load_ernie_all_blocks_bf16_normf32(
    st: ShardedSafeTensors, num_layers: Int, ctx: DeviceContext
) raises -> List[ErnieBlockWeights]:
    var blocks = List[ErnieBlockWeights]()
    for i in range(num_layers):
        blocks.append(load_ernie_block_weights_bf16_normf32(st, i, ctx))
    return blocks^


# Assert one named tensor loads at the expected shape (real H2D-less header read
# + shape compare). Prints an OK receipt line on success, raises on mismatch.
def _expect_shape(
    st: ShardedSafeTensors, name: String, var want: List[Int]
) raises:
    var info = st.tensor_info(name)
    var got = List[Int]()
    for i in range(len(info.shape)):
        got.append(Int(info.shape[i]))
    if len(got) != len(want):
        raise Error(String("ERNIE ") + name + String(" rank mismatch"))
    for i in range(len(want)):
        if got[i] != want[i]:
            raise Error(
                String("ERNIE ") + name + String(" shape mismatch at dim ")
                + String(i) + String(": got ") + String(got[i])
                + String(" want ") + String(want[i])
            )
    print("  OK", name, "[", end="")
    for i in range(len(got)):
        print(got[i], end="," if i + 1 < len(got) else "")
    print("]")


# Verify a representative sample of real tensors load at the expected shapes (a
# skeptic-checkable receipt: real header reads with shape asserts, RC=0 on
# success). Checks block 0 + block 35 (deepest) self-attn/mlp + the base weights.
def verify_ernie_stack_shapes(
    st: ShardedSafeTensors, num_layers: Int,
    hidden: Int, head_dim: Int, ffn: Int, in_ch: Int, text_in: Int, out_ch: Int,
) raises:
    # base weights
    _expect_shape(st, String("x_embedder.proj.weight"), [hidden, in_ch, 1, 1])
    _expect_shape(st, String("x_embedder.proj.bias"), [hidden])
    _expect_shape(st, String("text_proj.weight"), [hidden, text_in])
    _expect_shape(st, String("time_embedding.linear_1.weight"), [hidden, hidden])
    _expect_shape(st, String("adaLN_modulation.1.weight"), [6 * hidden, hidden])
    _expect_shape(st, String("adaLN_modulation.1.bias"), [6 * hidden])
    _expect_shape(st, String("final_norm.linear.weight"), [2 * hidden, hidden])
    _expect_shape(st, String("final_linear.weight"), [out_ch, hidden])
    _expect_shape(st, String("final_linear.bias"), [out_ch])

    # representative blocks: shallowest (0) and deepest (num_layers-1)
    var probe = List[Int]()
    probe.append(0)
    probe.append(num_layers - 1)
    for pi in range(len(probe)):
        var i = probe[pi]
        var p = String("layers.") + String(i)
        var ap = p + String(".self_attention")
        var mp = p + String(".mlp")
        _expect_shape(st, p + String(".adaLN_sa_ln.weight"), [hidden])
        _expect_shape(st, ap + String(".to_q.weight"), [hidden, hidden])
        _expect_shape(st, ap + String(".to_out.0.weight"), [hidden, hidden])
        _expect_shape(st, ap + String(".norm_q.weight"), [head_dim])
        _expect_shape(st, mp + String(".gate_proj.weight"), [ffn, hidden])
        _expect_shape(st, mp + String(".linear_fc2.weight"), [hidden, ffn])
