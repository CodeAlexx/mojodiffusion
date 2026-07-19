# 1:1 port of Serenity modules/util/loss/masked_loss.py
# Source of truth: /home/alex/Serenity/modules/util/loss/masked_loss.py
#
# masked_losses / masked_losses_with_prior: clamp the mask into
# [unmasked_weight, 1], multiply it into the per-element losses, and (optionally)
# normalize by the mask's mean over the spatial dims (1,2,3) keepdim.
#
# Serenity operates these on the per-element loss tensor (already in F32 — the
# predicted/target were cast bf16->f32 upstream). The loss tensor is a host
# statistic, not stored model state, so host-side F32 compute is consistent with
# the bf16-storage policy. The clamp/mean ops have no foundation kernel, so this
# is done host-side on the F32 view returned by Tensor.to_host. The result is
# rematerialized as a fresh Tensor in `losses`' storage dtype.
#
# BROADCAST: `mask` is loaded with channels=1 -> shape [N, 1, H, W] while
# `losses` (= F.mse_loss(reduction='none')) is [N, C, H, W]. PyTorch broadcasts
# the singleton channel across C (masked_loss.py:13 `losses *= clamped_mask`).
# We replicate this by indexing the mask with broadcast-aware strides: stride 0
# on any dim where mask.shape[d]==1 and losses.shape[d]>1. The normalize divisor
# `clamped_mask.mean(dim=(1,2,3), keepdim=True)` is the mean over the *mask*'s own
# trailing dims (mask_inner elements), then broadcast across the loss channels.

from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype


# helper: per-item normalizer = mean over dims (1..n-1), keepdim, broadcast back.
# For shape [N, d1, d2, ...] this divides each item i by mean over the trailing
# dims. Mirrors `clamped_mask.mean(dim=(1, 2, 3), keepdim=True)`. Computed over
# whatever tensor's shape is passed (the mask shape for clamped_mask).
fn _inner_count(shape: List[Int]) -> Int:
    var c = 1
    for i in range(1, len(shape)):
        c = c * shape[i]
    return c


# Broadcast-aware row-major strides for `mask_shape` against `loss_shape`.
# Returns one stride per dim: the mask's contiguous stride where the dims match,
# or 0 where mask_shape[d]==1 and loss_shape[d]>1 (singleton broadcast). Mirrors
# PyTorch numpy-style broadcasting (both tensors share rank here: [N,*,H,W]).
fn _broadcast_strides(mask_shape: List[Int], loss_shape: List[Int]) -> List[Int]:
    var rank = len(loss_shape)
    # contiguous row-major strides of the mask
    var mask_stride = List[Int]()
    for _ in range(rank):
        mask_stride.append(0)
    var acc = 1
    for d in range(rank - 1, -1, -1):
        mask_stride[d] = acc
        acc = acc * mask_shape[d]
    # zero out strides on broadcast (singleton) dims
    var out = List[Int]()
    for d in range(rank):
        if mask_shape[d] == 1 and loss_shape[d] > 1:
            out.append(0)
        else:
            out.append(mask_stride[d])
    return out^


# Map a row-major flat index `i` over `loss_shape` to the corresponding flat
# index into the mask, using broadcast-aware strides.
fn _mask_index(
    i: Int, loss_shape: List[Int], strides: List[Int]
) -> Int:
    var rank = len(loss_shape)
    var rem = i
    var midx = 0
    # decompose i into per-dim coords (row-major), accumulate mask offset
    for d in range(rank - 1, -1, -1):
        var coord = rem % loss_shape[d]
        rem = rem // loss_shape[d]
        midx = midx + coord * strides[d]
    return midx


# torch.clamp(mask, unmasked_weight, 1)  (masked_loss.py:11 / :29)
fn _clamp(v: Float32, lo: Float32, hi: Float32) -> Float32:
    var x = v
    if x < lo:
        x = lo
    if x > hi:
        x = hi
    return x


def masked_losses(
    losses: Tensor,
    mask: Tensor,
    unmasked_weight: Float32,
    normalize_masked_area_loss: Bool,
    ctx: DeviceContext,
) raises -> Tensor:
    # masked_loss.py:5-18
    var shape = losses.shape()
    var n = losses.numel()
    var batch = shape[0]
    var mask_shape = mask.shape()
    var mask_inner = _inner_count(mask_shape)
    var strides = _broadcast_strides(mask_shape, shape)

    var loss_h = losses.to_host(ctx)
    var mask_h = mask.to_host(ctx)

    # clamped_mask = torch.clamp(mask, unmasked_weight, 1)   (:11)
    # clamped lives in the MASK's element space (mask.numel() entries).
    var clamped = List[Float32]()
    for i in range(mask.numel()):
        clamped.append(_clamp(mask_h[i], unmasked_weight, 1.0))

    # losses *= clamped_mask  with channel broadcast               (:13)
    var out = List[Float32]()
    for i in range(n):
        out.append(loss_h[i] * clamped[_mask_index(i, shape, strides)])

    # if normalize_masked_area_loss:
    #     losses = losses / clamped_mask.mean(dim=(1,2,3), keepdim=True)   (:15-16)
    # Divisor is the mean over the MASK's trailing dims (mask_inner per item),
    # broadcast across the loss channels.
    if normalize_masked_area_loss:
        for b in range(batch):
            var s = Float32(0.0)
            for j in range(mask_inner):
                s = s + clamped[b * mask_inner + j]
            var m = s / Float32(mask_inner)
            for i in range(b * (n // batch), (b + 1) * (n // batch)):
                out[i] = out[i] / m

    return Tensor.from_host(out^, shape^, losses.dtype(), ctx)


def masked_losses_with_prior(
    losses: Tensor,
    prior_losses: Optional[Tensor],
    mask: Tensor,
    unmasked_weight: Float32,
    normalize_masked_area_loss: Bool,
    masked_prior_preservation_weight: Float32,
    ctx: DeviceContext,
) raises -> Tensor:
    # masked_loss.py:21-45
    var shape = losses.shape()
    var n = losses.numel()
    var batch = shape[0]
    var mask_shape = mask.shape()
    var mask_inner = _inner_count(mask_shape)
    var strides = _broadcast_strides(mask_shape, shape)
    var loss_inner = n // batch

    var loss_h = losses.to_host(ctx)
    var mask_h = mask.to_host(ctx)

    # clamped_mask = torch.clamp(mask, unmasked_weight, 1)   (:29)
    var clamped = List[Float32]()
    for i in range(mask.numel()):
        clamped.append(_clamp(mask_h[i], unmasked_weight, 1.0))

    # losses *= clamped_mask  with channel broadcast               (:31)
    var out = List[Float32]()
    for i in range(n):
        out.append(loss_h[i] * clamped[_mask_index(i, shape, strides)])

    # if normalize_masked_area_loss: losses /= clamped.mean(1,2,3)  (:33-34)
    if normalize_masked_area_loss:
        for b in range(batch):
            var s = Float32(0.0)
            for j in range(mask_inner):
                s = s + clamped[b * mask_inner + j]
            var m = s / Float32(mask_inner)
            for i in range(b * loss_inner, (b + 1) * loss_inner):
                out[i] = out[i] / m

    # if masked_prior_preservation_weight == 0 or prior_losses is None:
    #     return losses                                       (:36-37)
    if masked_prior_preservation_weight == 0.0 or not prior_losses:
        return Tensor.from_host(out^, shape^, losses.dtype(), ctx)

    var prior_h = prior_losses.value().to_host(ctx)

    # clamped_mask = (1 - clamped_mask)                       (:39)
    # inv lives in the mask's element space (matches `clamped`).
    var inv = List[Float32]()
    for i in range(mask.numel()):
        inv.append(1.0 - clamped[i])

    # prior_losses *= clamped_mask * masked_prior_preservation_weight   (:40)
    # with channel broadcast of the (1-mask) factor.
    var prior_out = List[Float32]()
    for i in range(n):
        var inv_v = inv[_mask_index(i, shape, strides)]
        prior_out.append(prior_h[i] * inv_v * masked_prior_preservation_weight)

    # if normalize_masked_area_loss: prior /= clamped.mean(1,2,3)  (:42-43)
    # divisor is mean over the inverted MASK's trailing dims.
    if normalize_masked_area_loss:
        for b in range(batch):
            var s = Float32(0.0)
            for j in range(mask_inner):
                s = s + inv[b * mask_inner + j]
            var m = s / Float32(mask_inner)
            for i in range(b * loss_inner, (b + 1) * loss_inner):
                prior_out[i] = prior_out[i] / m

    # return losses + prior_losses                            (:45)
    var res = List[Float32]()
    for i in range(n):
        res.append(out[i] + prior_out[i])

    return Tensor.from_host(res, shape, losses.dtype(), ctx)
