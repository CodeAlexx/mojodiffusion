# models/sd35/weights.mojo — SD3.5-Large transformer safetensors -> block weight structs.
#
# SD3.5 stores weights under "model.diffusion_model." prefix. Each joint block:
#   joint_blocks.{i}.{x_block,context_block}.adaLN_modulation.1.{weight,bias}
#   joint_blocks.{i}.{x_block,context_block}.attn.{qkv.weight,qkv.bias,proj.weight,proj.bias}
#   joint_blocks.{i}.{x_block,context_block}.attn.{ln_q.weight,ln_k.weight} (no bias)
#   joint_blocks.{i}.{x_block,context_block}.mlp.{fc1.weight,fc1.bias,fc2.weight,fc2.bias}
#
# qkv is PRE-FUSED in the checkpoint ([3D,D] weight, [3D] bias) — no row-stacking needed.
# adaLN_modulation.1 produces 6D (shift_msa,scale_msa,gate_msa,shift_mlp,scale_mlp,gate_mlp).
#
# Stack-level resident base (SD35StackBase uses host List[Float32] — no Tensor allocation):
#   x_embedder.proj.{weight [D,16,2,2] -> flattened [D*64], bias [D]}
#   context_embedder.{weight [D,4096], bias [D]}
#   t_embedder.mlp.{0.weight [D,256], 0.bias [D], 2.weight [D,D], 2.bias [D]}
#   y_embedder.mlp.{0.weight [D,2048], 0.bias [D], 2.weight [D,D], 2.bias [D]}
#   final_layer.{adaLN_modulation.1.weight [2D,D], adaLN_modulation.1.bias [2D],
#                linear.weight [64,D], linear.bias [64]}
#
# DIMS (confirmed from sd3.5_large.safetensors header):
#   D=2432, H=38, Dh=64, MLP=9728, depth=38 joint blocks, no dual attention.
#
# Mojo 0.26.x+: def not fn; Tensor move-only; host List[Float32] carriers.

from std.collections import List
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.models.sd35.sd35_block import StreamWeights, JointBlockWeights


comptime _PREFIX = "model.diffusion_model."


def _load_host_f32(st: SafeTensors, name: String, ctx: DeviceContext) raises -> List[Float32]:
    """Load a named tensor from safetensors as host F32 (casts from BF16/F16)."""
    var full = String(_PREFIX) + name
    var info = st.tensor_info(full)
    var bytes = st.tensor_bytes(full)
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    var t = Tensor.from_view(tv, ctx)
    return cast_tensor(t, STDtype.F32, ctx).to_host(ctx)


def _load_stream_weights(
    st: SafeTensors, bp: String, ctx: DeviceContext
) raises -> StreamWeights:
    """Load one stream (x_block or context_block) from a joint block."""
    # qkv is already fused: [3D,D] weight, [3D] bias
    var wqkv = _load_host_f32(st, bp + String("attn.qkv.weight"), ctx)
    var bqkv = _load_host_f32(st, bp + String("attn.qkv.bias"), ctx)
    var wproj = _load_host_f32(st, bp + String("attn.proj.weight"), ctx)
    var bproj = _load_host_f32(st, bp + String("attn.proj.bias"), ctx)
    var wfc1 = _load_host_f32(st, bp + String("mlp.fc1.weight"), ctx)
    var bfc1 = _load_host_f32(st, bp + String("mlp.fc1.bias"), ctx)
    var wfc2 = _load_host_f32(st, bp + String("mlp.fc2.weight"), ctx)
    var bfc2 = _load_host_f32(st, bp + String("mlp.fc2.bias"), ctx)
    var q_norm = _load_host_f32(st, bp + String("attn.ln_q.weight"), ctx)
    var k_norm = _load_host_f32(st, bp + String("attn.ln_k.weight"), ctx)
    return StreamWeights(
        wqkv^, bqkv^, wproj^, bproj^,
        wfc1^, bfc1^, wfc2^, bfc2^,
        q_norm^, k_norm^,
    )


def load_joint_block_weights(
    st: SafeTensors, block_idx: Int, ctx: DeviceContext
) raises -> JointBlockWeights:
    """Load both streams of joint_blocks.{block_idx}."""
    var pfx = String("joint_blocks.") + String(block_idx) + String(".")
    var ctx_bp = pfx + String("context_block.")
    var x_bp = pfx + String("x_block.")
    var ctxw = _load_stream_weights(st, ctx_bp, ctx)
    var xw = _load_stream_weights(st, x_bp, ctx)
    return JointBlockWeights(ctxw^, xw^)


# ── adaLN modulation weights per stream (both streams of a block) ─────────────
# adaLN_modulation.1: weight [6D,D] + bias [6D].
# silu(c) @ W.T + b gives 6D -> chunk into 6 [D] vectors.
# For training we load these from the streamed block (not resident).
def _load_ada_weights(
    st: SafeTensors, bp: String, ctx: DeviceContext
) raises -> (List[Float32], List[Float32]):
    """Returns (w [6D,D], b [6D]) as host F32 for one stream's adaLN_modulation.1."""
    var w = _load_host_f32(st, bp + String("adaLN_modulation.1.weight"), ctx)
    var b = _load_host_f32(st, bp + String("adaLN_modulation.1.bias"), ctx)
    return w^, b^


# ── Stack-level resident base (frozen during LoRA training) ──────────────────
from serenitymojo.models.sd35.sd35_stack_lora import SD35StackBase


def load_sd35_stack_base(
    st: SafeTensors, ctx: DeviceContext
) raises -> SD35StackBase:
    """Load the frozen stack-level base tensors (embedders, final layer) as host F32."""
    # x_embedder.proj.weight is [D, 16, 2, 2] (Conv2d stored as OIHW).
    # For training patchify we flatten: [D, 16*2*2] = [D, 64].
    # The Conv2d unfolds to a linear map; elements are stored contiguously
    # so the flat host list is row-major [D, 64].
    var xe_w = _load_host_f32(st, String("x_embedder.proj.weight"), ctx)
    var xe_b = _load_host_f32(st, String("x_embedder.proj.bias"), ctx)

    # context_embedder: [D, 4096]
    var ce_w = _load_host_f32(st, String("context_embedder.weight"), ctx)
    var ce_b = _load_host_f32(st, String("context_embedder.bias"), ctx)

    # t_embedder: two-layer MLP [D,256] -> SiLU -> [D,D]
    var t_w0 = _load_host_f32(st, String("t_embedder.mlp.0.weight"), ctx)
    var t_b0 = _load_host_f32(st, String("t_embedder.mlp.0.bias"), ctx)
    var t_w2 = _load_host_f32(st, String("t_embedder.mlp.2.weight"), ctx)
    var t_b2 = _load_host_f32(st, String("t_embedder.mlp.2.bias"), ctx)

    # y_embedder (pooled CLIP): [D,2048] -> SiLU -> [D,D]
    var y_w0 = _load_host_f32(st, String("y_embedder.mlp.0.weight"), ctx)
    var y_b0 = _load_host_f32(st, String("y_embedder.mlp.0.bias"), ctx)
    var y_w2 = _load_host_f32(st, String("y_embedder.mlp.2.weight"), ctx)
    var y_b2 = _load_host_f32(st, String("y_embedder.mlp.2.bias"), ctx)

    # final_layer: adaLN_modulation.1 [2D,D] + linear [64,D]
    var fl_ada_w = _load_host_f32(st, String("final_layer.adaLN_modulation.1.weight"), ctx)
    var fl_ada_b = _load_host_f32(st, String("final_layer.adaLN_modulation.1.bias"), ctx)
    var fl_lin_w = _load_host_f32(st, String("final_layer.linear.weight"), ctx)
    var fl_lin_b = _load_host_f32(st, String("final_layer.linear.bias"), ctx)

    return SD35StackBase(
        xe_w^, xe_b^,
        ce_w^, ce_b^,
        t_w0^, t_b0^, t_w2^, t_b2^,
        y_w0^, y_b0^, y_w2^, y_b2^,
        fl_ada_w^, fl_ada_b^, fl_lin_w^, fl_lin_b^,
    )
