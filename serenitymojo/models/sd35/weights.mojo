# models/sd35/weights.mojo — SD3.5-Large stack-base safetensors loader.
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
# Stack-level resident base:
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
# The stack-base loader preserves checkpoint dtype with Tensor.from_view. SD3.5
# block/offload streaming still has a separate host-F32 blocker in
# sd35_stack_lora.mojo.

from std.gpu.host import DeviceContext
from std.memory import ArcPointer
from serenitymojo.tensor import Tensor
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.models.sd35.sd35_stack_lora import SD35StackBase


comptime _PREFIX = "model.diffusion_model."
comptime TArc = ArcPointer[Tensor]


def _load_tensor(st: SafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    """Load a named SD3.5 tensor with checkpoint dtype preserved."""
    var full = String(_PREFIX) + name
    var info = st.tensor_info(full)
    var bytes = st.tensor_bytes(full)
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    return Tensor.from_view(tv, ctx)


def load_sd35_stack_base(
    st: SafeTensors, ctx: DeviceContext
) raises -> SD35StackBase:
    """Load frozen stack-level base tensors preserving checkpoint dtype."""
    # x_embedder.proj.weight is [D, 16, 2, 2] (Conv2d stored as OIHW).
    # For training patchify we flatten: [D, 16*2*2] = [D, 64].
    # The Conv2d unfolds to a linear map; elements are stored contiguously
    # so the flat host list is row-major [D, 64].
    var xe_w = _load_tensor(st, String("x_embedder.proj.weight"), ctx)
    var xe_b = _load_tensor(st, String("x_embedder.proj.bias"), ctx)

    # context_embedder: [D, 4096]
    var ce_w = _load_tensor(st, String("context_embedder.weight"), ctx)
    var ce_b = _load_tensor(st, String("context_embedder.bias"), ctx)

    # t_embedder: two-layer MLP [D,256] -> SiLU -> [D,D]
    var t_w0 = _load_tensor(st, String("t_embedder.mlp.0.weight"), ctx)
    var t_b0 = _load_tensor(st, String("t_embedder.mlp.0.bias"), ctx)
    var t_w2 = _load_tensor(st, String("t_embedder.mlp.2.weight"), ctx)
    var t_b2 = _load_tensor(st, String("t_embedder.mlp.2.bias"), ctx)

    # y_embedder (pooled CLIP): [D,2048] -> SiLU -> [D,D]
    var y_w0 = _load_tensor(st, String("y_embedder.mlp.0.weight"), ctx)
    var y_b0 = _load_tensor(st, String("y_embedder.mlp.0.bias"), ctx)
    var y_w2 = _load_tensor(st, String("y_embedder.mlp.2.weight"), ctx)
    var y_b2 = _load_tensor(st, String("y_embedder.mlp.2.bias"), ctx)

    # final_layer: adaLN_modulation.1 [2D,D] + linear [64,D]
    var fl_ada_w = _load_tensor(st, String("final_layer.adaLN_modulation.1.weight"), ctx)
    var fl_ada_b = _load_tensor(st, String("final_layer.adaLN_modulation.1.bias"), ctx)
    var fl_lin_w = _load_tensor(st, String("final_layer.linear.weight"), ctx)
    var fl_lin_b = _load_tensor(st, String("final_layer.linear.bias"), ctx)

    return SD35StackBase(
        TArc(xe_w^), TArc(xe_b^),
        TArc(ce_w^), TArc(ce_b^),
        TArc(t_w0^), TArc(t_b0^), TArc(t_w2^), TArc(t_b2^),
        TArc(y_w0^), TArc(y_b0^), TArc(y_w2^), TArc(y_b2^),
        TArc(fl_ada_w^), TArc(fl_ada_b^), TArc(fl_lin_w^), TArc(fl_lin_b^),
    )
