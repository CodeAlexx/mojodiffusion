# autograd_v2/ltx2_video_block_graph.mojo — LTX-2.3 VIDEO per-block LoRA backward
# driven by the graph engine, COARSE (Option A; AUTOGRAD_V2_MOJO_DESIGN.md P7
# rollout recipe, the OPK_KREA2_SINGLE_BLOCK precedent).
#
# Recompute-style checkpoint, exactly like the trainer's stack loop
# (ltx2_video_stack_lora_backward, models/ltx2/ltx2_video_stack.mojo:397-419):
# the stack hands each block its saved INPUT `hidden`; this module records ONE
# composite node (OPK_LTX2V_VIDEO_BLOCK) whose leaf is that input, then
# execute_ltx2v_block's apply arm RECOMPUTES the block forward
# (ltx2_video_block_train_forward) from the saved input and calls the WHOLE
# ltx2_video_block_backward oracle — so every internal >=3-way fold (gated
# residual, the 3 rms/AdaLN modulate folds, attn/FFN) stays INSIDE the oracle
# (C15 trivially satisfied at graph level; the ONLY tracked edge is `hidden`).
#
# Bit-equality vs the hand-chain (C14, ltx2 is MATH-MODE recompute-softmax
# attention — ltx2_av_backward.mojo:377 — so BIT gates, not a variance class):
# the apply arm calls the SAME train-forward + block-backward the stack loop
# calls on the same operands (weights carry the attached LoRA the caller set via
# _attach_block_lora), so the grads are bit-identical by construction. The gate
# is autograd_v2/tests/ltx2_block_parity.mojo (same-process, NONZERO LoRA B).
#
# The n_slots LoRA grad pairs are HOST List[Float32] (LoraPairGrad) and cannot
# flow through the engine's TArc-only edge/Dict machinery, so execute returns the
# WHOLE LTX2VideoBlockGrads (d_hidden sunk through the engine leaf; the host LoRA
# list captured out-of-band) — the krea2 pattern (node.mojo:82).
#
# Conductor (C10): NOT here — the stack-graph loop
# (ltx2_video_stack_lora_backward_graph, models/ltx2/ltx2_video_stack_graph.mojo)
# keeps the reverse block loop + src.get_block + _attach_block_lora + the offload
# seam around each per-block graph call. NO StepSlab / NO capture in this coarse
# phase (L5 follow-on, gated on the sync/alloc feasibility audit).
#
# Mojo 1.0.0b1, NVIDIA.

from std.gpu.host import DeviceContext, DeviceBuffer
from std.collections import Optional
from serenitymojo.tensor import Tensor
from serenitymojo.autograd_v2.node import TArc
from serenitymojo.autograd_v2.graph import Graph
from serenitymojo.autograd_v2.ops_record import record_ltx2v_video_block
from serenitymojo.autograd_v2.engine import (
    execute_ltx2v_block, execute_ltx2v_block_slab, execute_ltx2v_block_slab_dev,
)
from serenitymojo.autograd_v2.step_slab import StepSlab
from serenitymojo.models.dit.ltx2_dit import LTX2AVBlockWeights
from serenitymojo.models.ltx2.ltx2_video_backward import (
    LTX2VideoBlockGrads, LTX2VideoBlockGradsDev,
)
from serenitymojo.models.ltx2.ltx2_av_backward import Ltx2LoraGradStore


def ltx2_video_block_graph_backward[S_V: Int, N_TXT: Int](
    d_video: TArc,
    w: LTX2AVBlockWeights,
    block_input: TArc,
    enc: Tensor, v_temb: Tensor, v_prompt_ts: Tensor,
    v_cos: Tensor, v_sin: Tensor,
    eps: Float32, ctx: DeviceContext,
) raises -> LTX2VideoBlockGrads:
    """Graph-engine replacement for the trainer-loop per-block pair (recompute
    ltx2_video_block_train_forward + ltx2_video_block_backward): record the block
    from the saved INPUT (single tracked leaf = hidden), execute the backward
    seeded with d_video, return the SAME LTX2VideoBlockGrads the hand-chain block
    backward returns (d_hidden + n_slots host-list LoRA pairs). `w` MUST already
    carry the block's attached LoRA (caller's _attach_block_lora, the stack loop's
    exact order)."""
    var g = Graph()

    # block input `hidden`: the ONE tracked leaf (its accumulated grad = the
    # returned d_hidden). Zero-copy re-box so the id stamp never mutates the shared
    # saved arc (the klein_block_graph.mojo:86 idiom).
    var x_t = Tensor(block_input[].buf.copy(), block_input[].shape(), block_input[].dtype())
    x_t.set_id(g.fresh_tensor_id())
    var x_id = x_t.id
    _ = g.leaf(x_id)
    var x = TArc(x_t^)

    var out = record_ltx2v_video_block(g, x, x_id, ctx)
    var root = g.node_of_tensor[out[].id]
    var grads = execute_ltx2v_block[S_V, N_TXT](
        g, root, d_video.copy(),
        w, enc, v_temb, v_prompt_ts, v_cos, v_sin, eps, ctx,
    )
    return grads^


def ltx2_video_block_graph_backward_slab[S_V: Int, N_TXT: Int](
    d_video: TArc,
    w: LTX2AVBlockWeights,
    block_input: TArc,
    enc: Tensor, v_temb: Tensor, v_prompt_ts: Tensor,
    v_cos: Tensor, v_sin: Tensor,
    eps: Float32, ctx: DeviceContext,
    mut slab: StepSlab,
) raises -> LTX2VideoBlockGrads:
    """StepSlab variant of `ltx2_video_block_graph_backward` (this file :45) —
    identical recording; execute routes through execute_ltx2v_block_slab so the
    block fwd+bwd recompute allocates from the StepSlab (apply_ltx2v_slab
    mark/rewind per block, C8). Bit-identical d_hidden + LoRA grads."""
    var g = Graph()
    var x_t = Tensor(block_input[].buf.copy(), block_input[].shape(), block_input[].dtype())
    x_t.set_id(g.fresh_tensor_id())
    var x_id = x_t.id
    _ = g.leaf(x_id)
    var x = TArc(x_t^)

    var out = record_ltx2v_video_block(g, x, x_id, ctx)
    var root = g.node_of_tensor[out[].id]
    var grads = execute_ltx2v_block_slab[S_V, N_TXT](
        g, root, d_video.copy(),
        w, enc, v_temb, v_prompt_ts, v_cos, v_sin, eps, ctx, slab,
    )
    return grads^


def ltx2_video_block_graph_backward_slab_dev[S_V: Int, N_TXT: Int](
    d_video: TArc,
    w: LTX2AVBlockWeights,
    block_input: TArc,
    enc: Tensor, v_temb: Tensor, v_prompt_ts: Tensor,
    v_cos: Tensor, v_sin: Tensor,
    eps: Float32, ctx: DeviceContext,
    mut slab: StepSlab,
    store: Ltx2LoraGradStore = Ltx2LoraGradStore(),
    block_idx: Int = -1,
    d_hidden_dst: Optional[DeviceBuffer[DType.uint8]] = None,
) raises -> LTX2VideoBlockGradsDev:
    """Device-grad + StepSlab variant of `ltx2_video_block_graph_backward` —
    identical recording; execute routes through execute_ltx2v_block_slab_dev so
    the block fwd+bwd recompute allocates from the StepSlab AND the LoRA d_A/d_B
    stay device-resident (rung 1/2). d_hidden bit-identical; the LoRA grads come
    back as LTX2VideoBlockGradsDev for the stack loop's boundary readback.

    d_hidden_dst (rung 3 ping-pong): when set, d_hidden lands in the caller's
    FIXED buffer instead of a fresh .clone (capture-safe). None = .clone path."""
    var g = Graph()
    var x_t = Tensor(block_input[].buf.copy(), block_input[].shape(), block_input[].dtype())
    x_t.set_id(g.fresh_tensor_id())
    var x_id = x_t.id
    _ = g.leaf(x_id)
    var x = TArc(x_t^)

    var out = record_ltx2v_video_block(g, x, x_id, ctx)
    var root = g.node_of_tensor[out[].id]
    var grads = execute_ltx2v_block_slab_dev[S_V, N_TXT](
        g, root, d_video.copy(),
        w, enc, v_temb, v_prompt_ts, v_cos, v_sin, eps, ctx, slab, store, block_idx,
        d_hidden_dst,
    )
    return grads^
