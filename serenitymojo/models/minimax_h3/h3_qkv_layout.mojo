# serenitymojo/models/minimax_h3/h3_qkv_layout.mojo
#
# Shared MiniMax-H3 released-checkpoint QKV row-layout conversion.
#
# The checkpoint stores fused QKV rows per head:
#   [h0.q, h0.k, h0.v, h1.q, h1.k, h1.v, ...]
# while both the training block and product block consume contiguous parts:
#   [q_all, k_all, v_all].
#
# This is a pure row permutation. Keeping it in one small module prevents the
# training streamer and inference loader from silently choosing different base
# functions again.
from max.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.ops.tensor_algebra import gather_rows


def h3_qkv_deinterleave_row_ids(heads: Int, head_dim: Int) raises -> List[Int]:
    if heads <= 0 or head_dim <= 0:
        raise Error("H3 QKV layout requires positive heads and head_dim")
    var ids = List[Int](capacity=3 * heads * head_dim)
    for part in range(3):
        for head in range(heads):
            for dim in range(head_dim):
                ids.append(head * 3 * head_dim + part * head_dim + dim)
    return ids^


def h3_qkv_deinterleave_rows(
    raw: Tensor, heads: Int, head_dim: Int, ctx: DeviceContext
) raises -> Tensor:
    """Per-head-interleaved rows -> contiguous [q_all;k_all;v_all]."""
    var shape = raw.shape()
    if len(shape) != 2 or shape[0] != 3 * heads * head_dim:
        raise Error(
            "H3 QKV layout: expected rank-2 tensor with 3*heads*head_dim rows"
        )
    var ids = h3_qkv_deinterleave_row_ids(heads, head_dim)
    return gather_rows(raw, ids, ctx)
