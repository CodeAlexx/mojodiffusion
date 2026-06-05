# masked_loss.mojo -- Musubi-compatible LTX2 AV masked loss contracts.

from std.collections import List
from std.math import abs


comptime LTX2_LOSS_MSE = 0
comptime LTX2_LOSS_MAE = 1
comptime LTX2_LOSS_HUBER = 2


def ltx2_loss_kind_from_string(name: String) raises -> Int:
    if name == "mse":
        return LTX2_LOSS_MSE
    if name == "mae" or name == "l1":
        return LTX2_LOSS_MAE
    if name == "huber" or name == "smooth_l1":
        return LTX2_LOSS_HUBER
    raise Error(String("unknown LTX2 loss kind: ") + name)


def _require_len(name: String, got: Int, expected: Int) raises:
    if got != expected:
        raise Error(
            name + String(" length mismatch: got ") + String(got)
            + String(", expected ") + String(expected)
        )


def _per_elem_loss(pred: Float32, target: Float32, kind: Int, huber_delta: Float32) raises -> Float32:
    var d = pred - target
    var ad = abs(d)
    if kind == LTX2_LOSS_MAE:
        return ad
    if kind == LTX2_LOSS_HUBER:
        if huber_delta <= Float32(0.0):
            return ad
        if ad < huber_delta:
            return Float32(0.5) * d * d / huber_delta
        return ad - Float32(0.5) * huber_delta
    return d * d


def masked_loss_unmasked(
    pred: List[Float32],
    target: List[Float32],
    kind: Int = LTX2_LOSS_MSE,
    huber_delta: Float32 = Float32(1.0),
) raises -> Float32:
    _require_len("target", len(target), len(pred))
    if len(pred) == 0:
        raise Error("masked loss requires at least one element")
    var total = Float32(0.0)
    for i in range(len(pred)):
        total += _per_elem_loss(pred[i], target[i], kind, huber_delta)
    return total / Float32(len(pred))


def _reduce_masked_sum(
    masked_sum: Float32,
    total_elems: Int,
    mask_positions: Int,
    true_count: Int,
    fallback_unmasked: Float32,
) raises -> Float32:
    if mask_positions <= 0:
        raise Error("mask must have at least one element")
    if true_count == 0:
        return fallback_unmasked
    var per_elem_mean = masked_sum / Float32(total_elems)
    var denom = Float32(true_count) / Float32(mask_positions)
    return per_elem_mean / denom


def masked_loss_video_bf_mask(
    pred: List[Float32],
    target: List[Float32],
    mask_bf: List[Bool],
    batch: Int,
    channels: Int,
    frames: Int,
    height: Int,
    width: Int,
    kind: Int = LTX2_LOSS_MSE,
    huber_delta: Float32 = Float32(1.0),
) raises -> Float32:
    var total_elems = batch * channels * frames * height * width
    _require_len("pred", len(pred), total_elems)
    _require_len("target", len(target), total_elems)
    _require_len("video [B,F] mask", len(mask_bf), batch * frames)
    var fallback = masked_loss_unmasked(pred, target, kind, huber_delta)
    var masked_sum = Float32(0.0)
    var true_count = 0
    for b in range(batch):
        for f in range(frames):
            var m = mask_bf[b * frames + f]
            if m:
                true_count += 1
            for c in range(channels):
                for h in range(height):
                    for w in range(width):
                        if m:
                            var idx = ((((b * channels + c) * frames + f) * height + h) * width + w)
                            masked_sum += _per_elem_loss(pred[idx], target[idx], kind, huber_delta)
    return _reduce_masked_sum(masked_sum, total_elems, batch * frames, true_count, fallback)


def masked_loss_video_5d_mask(
    pred: List[Float32],
    target: List[Float32],
    mask: List[Bool],
    batch: Int,
    channels: Int,
    frames: Int,
    height: Int,
    width: Int,
    kind: Int = LTX2_LOSS_MSE,
    huber_delta: Float32 = Float32(1.0),
) raises -> Float32:
    var total_elems = batch * channels * frames * height * width
    _require_len("pred", len(pred), total_elems)
    _require_len("target", len(target), total_elems)
    if len(mask) == batch * frames:
        return masked_loss_video_bf_mask(
            pred, target, mask, batch, channels, frames, height, width, kind, huber_delta
        )
    _require_len("video 5D mask", len(mask), total_elems)
    var fallback = masked_loss_unmasked(pred, target, kind, huber_delta)
    var masked_sum = Float32(0.0)
    var true_count = 0
    for i in range(total_elems):
        if mask[i]:
            true_count += 1
            masked_sum += _per_elem_loss(pred[i], target[i], kind, huber_delta)
    return _reduce_masked_sum(masked_sum, total_elems, total_elems, true_count, fallback)


def masked_loss_audio_bt_mask(
    pred: List[Float32],
    target: List[Float32],
    mask_bt: List[Bool],
    batch: Int,
    channels: Int,
    time: Int,
    mel: Int,
    kind: Int = LTX2_LOSS_MSE,
    huber_delta: Float32 = Float32(1.0),
) raises -> Float32:
    var total_elems = batch * channels * time * mel
    _require_len("pred", len(pred), total_elems)
    _require_len("target", len(target), total_elems)
    _require_len("audio [B,T] mask", len(mask_bt), batch * time)
    var fallback = masked_loss_unmasked(pred, target, kind, huber_delta)
    var masked_sum = Float32(0.0)
    var true_count = 0
    for b in range(batch):
        for t in range(time):
            var m = mask_bt[b * time + t]
            if m:
                true_count += 1
            for c in range(channels):
                for m_bin in range(mel):
                    if m:
                        var idx = (((b * channels + c) * time + t) * mel + m_bin)
                        masked_sum += _per_elem_loss(pred[idx], target[idx], kind, huber_delta)
    return _reduce_masked_sum(masked_sum, total_elems, batch * time, true_count, fallback)


def masked_loss_audio_4d_mask(
    pred: List[Float32],
    target: List[Float32],
    mask: List[Bool],
    batch: Int,
    channels: Int,
    time: Int,
    mel: Int,
    kind: Int = LTX2_LOSS_MSE,
    huber_delta: Float32 = Float32(1.0),
) raises -> Float32:
    var total_elems = batch * channels * time * mel
    _require_len("pred", len(pred), total_elems)
    _require_len("target", len(target), total_elems)
    if len(mask) == batch * time:
        return masked_loss_audio_bt_mask(pred, target, mask, batch, channels, time, mel, kind, huber_delta)
    _require_len("audio 4D mask", len(mask), total_elems)
    var fallback = masked_loss_unmasked(pred, target, kind, huber_delta)
    var masked_sum = Float32(0.0)
    var true_count = 0
    for i in range(total_elems):
        if mask[i]:
            true_count += 1
            masked_sum += _per_elem_loss(pred[i], target[i], kind, huber_delta)
    return _reduce_masked_sum(masked_sum, total_elems, total_elems, true_count, fallback)
