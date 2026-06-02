# checkpoint.mojo -- gradient checkpointing + activation offload (Phase T0).
#
# FULL_PORT_TRAINING_PLAN.md Phase T0, Tier-5 kill-risk #2: CheckpointOffload-
# Boundary. This is the un-done full-fine-tune blocker -- 24 GB cannot hold the
# full-DiT activation stack, so a checkpointed block must NOT save its internal
# activations. It saves ONLY its input (optionally offloaded to host) and
# RECOMPUTES the forward during backward before backpropagating through it.
#
# This is the Mojo port of flame-core's `checkpoint_offload_boundary` +
# `backward_checkpoint_offload_boundary` (src/autograd.rs ~3208). The Rust
# contract is:
#   forward : output = recompute_fn(input); offload input -> host; record a tape
#             entry holding (offloaded_input, recompute_fn, output_id).
#   backward: restore input host->device; requires_grad(input); recompute the
#             forward on a fresh sub-tape; backprop with grad_output; extract the
#             input gradient.
#
# ===========================================================================
# MOJO 1.0.0b1 LIMITATION (the critical finding the prompt asks me to surface):
# ---------------------------------------------------------------------------
# flame-core takes `recompute_fn: impl Fn(&Tensor) -> Result<Tensor>` -- a
# CLOSURE stored in the TapeEntry. Mojo 1.0.0b1 cannot store a heterogeneous
# captured closure in a struct field (no boxed `dyn Fn`, no existential trait
# objects, no first-class closure type that captures + erases its environment).
# So the GENERAL closure-based form CANNOT be ported as-is.
#
# The supported substitute (this file) is a CONCRETE checkpointed block: the
# recompute path is a fixed op sequence (linear -> silu) whose forward is
# re-run by calling the known forward ops directly, then backprop'd through the
# known *_backward kernels. The autograd.mojo tape already uses this exact
# pattern -- Op-tag dispatch instead of boxed backward fns (autograd.mojo:39-42).
# Generalizing to arbitrary blocks needs EITHER (a) one Op-tag per checkpointed
# block kind (a finite, model-driven enum -- the pragmatic path), OR (b) Mojo
# gaining storable closures. Both are recorded in the handoff.
# ===========================================================================
#
# Mojo 1.0.0b1: def not fn; Tensor move-only; host offload via raw device<->host
# byte copy (NOT tensor.to_host, which widens to F32) -- we copy storage bytes
# verbatim so restore is bit-exact, mirroring the from_view H2D byte-copy path.

from collections import List
from collections.optional import Optional
from std.gpu.host import DeviceContext, DeviceBuffer
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.linear import linear
from serenitymojo.ops.activations import silu
from serenitymojo.ops.activation_backward import silu_backward
from serenitymojo.ops.linalg_backward import linear_backward, LinearGrads


# ---------------------------------------------------------------------------
# Host-offload save / restore: copy a Tensor's device bytes -> host RAM and
# back. Mirrors flame-core training_offload.rs (offload_to_host / to_device) and
# the loaders' enqueue_copy d2d pattern, but staged through a host buffer. We
# copy RAW STORAGE BYTES (dtype-agnostic, no F32 widen) so restore is bit-exact.
# ---------------------------------------------------------------------------

struct HostOffload(Copyable, Movable):
    """An activation parked in host RAM. Owns a host byte buffer (List[UInt8],
    self-managing memory) + the metadata needed to rebuild the device Tensor on
    restore. This is the `saved_input` flame-core stows in the TapeEntry, except
    the bytes live on the host (the 24 GB-reclaim mechanism for full-FT)."""
    var host: List[UInt8]
    var shape: List[Int]
    var dtype: STDtype

    def __init__(
        out self,
        var host: List[UInt8],
        var shape: List[Int],
        dtype: STDtype,
    ):
        self.host = host^
        self.shape = shape^
        self.dtype = dtype


def offload_to_host(t: Tensor, ctx: DeviceContext) raises -> HostOffload:
    """Device -> host: copy t's raw storage bytes into a host List[UInt8].

    This is the forward-pass 'offload the saved input' step. Pure byte copy --
    the host buffer holds exactly the same bytes as the device buffer (bf16 stays
    bf16, f32 stays f32). The caller drops the device Tensor after this returns;
    that drop is what reclaims the VRAM (the whole point on 24 GB)."""
    var n = t.nbytes()
    # DeviceContext can only D2H into one of its own host buffers; stage there,
    # then copy bytes into an owned List (mirrors tensor.from_view's H2D staging).
    var staging = ctx.enqueue_create_host_buffer[DType.uint8](n)
    ctx.enqueue_copy(dst_buf=staging, src_buf=t.buf)
    ctx.synchronize()
    var sp = staging.unsafe_ptr()
    var host = List[UInt8]()
    for i in range(n):
        host.append(sp[i])
    return HostOffload(host^, t.shape(), t.dtype())


def restore_to_device(off: HostOffload, ctx: DeviceContext) raises -> Tensor:
    """Host -> device: rebuild the Tensor from the parked host bytes.

    The backward-pass 'restore offloaded input' step. Stages the host bytes into
    a pinned host buffer, H2D-copies to a fresh device buffer, yielding a Tensor
    byte-identical to the one that was offloaded."""
    var n = len(off.host)
    var staging = ctx.enqueue_create_host_buffer[DType.uint8](n)
    var sp = staging.unsafe_ptr()
    for i in range(n):
        sp[i] = off.host[i]
    var dev = ctx.enqueue_create_buffer[DType.uint8](n)
    ctx.enqueue_copy(dst_buf=dev, src_buf=staging)
    ctx.synchronize()
    return Tensor(dev^, off.shape.copy(), off.dtype, 0)


# ---------------------------------------------------------------------------
# The concrete checkpointed block: y = silu(x @ W^T).  (linear -> silu)
# This is the 2-op block the prompt specifies. The forward saves NOTHING beyond
# the input (offloaded); backward recomputes the whole forward from the restored
# input and backprops through it.
# ---------------------------------------------------------------------------

struct BlockGrads(Movable):
    """Grads produced by the checkpointed block backward: dx (input grad) and
    dW (weight grad). Multi-return Movable struct (Mojo has no tuple of move-only
    Tensors)."""
    var dx: Tensor
    var dW: Tensor

    def __init__(out self, var dx: Tensor, var dW: Tensor):
        self.dx = dx^
        self.dW = dW^


def block_forward(x: Tensor, w: Tensor, ctx: DeviceContext) raises -> Tensor:
    """Forward of the concrete block: linear (x @ W^T, no bias) then silu.

    x: [M,K], w: [N,K] (PyTorch nn.Linear weight, y = x @ W^T = [M,N]).
    Returns silu(x @ W^T). Used by BOTH the non-checkpointed reference path and
    the recompute path -- so the two run identical math, which is the gate's
    whole point."""
    var pre = linear(x, w, Optional[Tensor](), ctx)   # [M,N] = x @ W^T (no bias)
    var y = silu(pre, ctx)                             # [M,N]
    return y^


def block_backward_saveall(
    grad_out: Tensor,
    x: Tensor,
    w: Tensor,
    M: Int, K: Int, N: Int,
    ctx: DeviceContext,
) raises -> BlockGrads:
    """NON-checkpointed backward (the 'save all' oracle).

    Recomputes the linear pre-activation here for clarity (a real save-all path
    would have STORED it; recomputing changes nothing numerically). The
    difference that matters for the gate is that the checkpoint path additionally
    round-trips x through host RAM.
      pre   = x @ W^T                        [M,N]
      d_pre = silu_backward(grad_out, pre)   [M,N]
      dx,dW = linear_backward(d_pre, x, w)   ([M,K],[N,K])"""
    var pre = linear(x, w, Optional[Tensor](), ctx)
    var d_pre = silu_backward(grad_out, pre, ctx)
    var lg = linear_backward(d_pre, x, w, M, K, N, ctx)
    # No-bias block: lg.d_b is an unused [N] grad. Take d_x/d_w by clone and let
    # the whole `lg` (incl. d_b) destroy as one value -- Mojo 1.0.0b1 rejects
    # moving fields out of the middle of a struct it must still destroy.
    var dx = lg.d_x.clone(ctx)
    var dW = lg.d_w.clone(ctx)
    return BlockGrads(dx^, dW^)


def checkpoint_recompute(
    saved_input: HostOffload,
    w: Tensor,
    grad_out: Tensor,
    M: Int, K: Int, N: Int,
    ctx: DeviceContext,
) raises -> BlockGrads:
    """CHECKPOINTED backward (the deliverable).

    Mirrors flame-core backward_checkpoint_offload_boundary:
      1. restore the offloaded input host -> device,
      2. RECOMPUTE the block forward's internal activation from the restored
         input (pre = x @ W^T -- the activation a non-checkpointed forward would
         have KEPT; not saving it is the memory win),
      3. backprop through the recomputed forward (silu then linear),
      4. produce dx (input grad) and dW.

    saved_input : the block input, parked on host by offload_to_host (forward).
    w           : the block weight (stays resident; weights aren't checkpointed).
    grad_out    : dL/dy, upstream grad of the block output.
    Returns dx, dW -- must equal block_backward_saveall to cos >= 0.9999."""
    var x = restore_to_device(saved_input, ctx)          # (1)
    var pre = linear(x, w, Optional[Tensor](), ctx)      # (2) recompute
    var d_pre = silu_backward(grad_out, pre, ctx)        # (3) silu bwd
    var lg = linear_backward(d_pre, x, w, M, K, N, ctx)  # (3) linear bwd
    # clone d_x/d_w out; let the whole lg (incl. unused no-bias d_b) destroy.
    var dx = lg.d_x.clone(ctx)                            # (4) dx, dW
    var dW = lg.d_w.clone(ctx)
    return BlockGrads(dx^, dW^)
