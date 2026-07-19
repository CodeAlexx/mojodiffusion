# Bernini-R Adaptive Projected Guidance.
#
# Exact inference port of the pinned ByteDance creator implementation in
# /home/alex/Bernini/bernini/models/wan_diffusion.py.  The unusual reduction
# axes are intentional: PyTorch dims [-1,-2,-4] on [B,C,T,H,W] reduce C/H/W
# independently for each [B,T] group.  Do not replace this with a whole-sample
# norm.  Momentum state persists across denoise steps.

from std.collections import List
from std.math import sqrt


struct BerniniAPGMomentum(Movable):
    var momentum: Float32
    var running_average: List[Float32]
    var initialized: Bool

    def __init__(out self, momentum: Float32):
        self.momentum = momentum
        self.running_average = List[Float32]()
        self.initialized = False


def _check_5d_numel(
    values: List[Float32], batch: Int, channels: Int, frames: Int,
    height: Int, width: Int, label: String,
) raises:
    if batch <= 0 or channels <= 0 or frames <= 0 or height <= 0 or width <= 0:
        raise Error("bernini_apg: all dimensions must be positive")
    var expected = batch * channels * frames * height * width
    if len(values) != expected:
        raise Error(
            String("bernini_apg: ") + label + String(" numel mismatch: got ")
            + String(len(values)) + String(", expected ") + String(expected)
        )


def _index_5d(
    b: Int, c: Int, t: Int, h: Int, w: Int,
    channels: Int, frames: Int, height: Int, width: Int,
) -> Int:
    return ((((b * channels + c) * frames + t) * height + h) * width + w)


def bernini_apg_normalize_diff(
    diff_input: List[Float32], base_pred: List[Float32],
    mut state: BerniniAPGMomentum,
    eta: Float32, norm_threshold: Float32,
    batch: Int, channels: Int, frames: Int, height: Int, width: Int,
) raises -> List[Float32]:
    """Creator `_normalize_diff`, including persistent momentum and C/H/W norms."""
    _check_5d_numel(diff_input, batch, channels, frames, height, width, "diff")
    _check_5d_numel(base_pred, batch, channels, frames, height, width, "base")

    var diff = diff_input.copy()
    if state.initialized:
        if len(state.running_average) != len(diff):
            raise Error("bernini_apg: momentum state numel mismatch")
        for i in range(len(diff)):
            diff[i] = diff[i] + state.momentum * state.running_average[i]
    state.running_average = diff.copy()
    state.initialized = True

    # norm_threshold clamps each [b,t] slice over [c,h,w], matching
    # diff.norm(p=2, dim=[-1,-2,-4], keepdim=True).
    if norm_threshold > 0.0:
        for b in range(batch):
            for t in range(frames):
                var sum_sq: Float64 = 0.0
                for c in range(channels):
                    for h in range(height):
                        for w in range(width):
                            var i = _index_5d(b, c, t, h, w, channels, frames, height, width)
                            var v = Float64(diff[i])
                            sum_sq += v * v
                var norm = sqrt(sum_sq)
                var scale = Float32(1.0)
                if norm > Float64(norm_threshold):
                    scale = Float32(Float64(norm_threshold) / norm)
                if scale < 1.0:
                    for c in range(channels):
                        for h in range(height):
                            for w in range(width):
                                var i = _index_5d(b, c, t, h, w, channels, frames, height, width)
                                diff[i] *= scale

    var out = diff.copy()
    for b in range(batch):
        for t in range(frames):
            var base_sum_sq: Float64 = 0.0
            for c in range(channels):
                for h in range(height):
                    for w in range(width):
                        var i = _index_5d(b, c, t, h, w, channels, frames, height, width)
                        var v = Float64(base_pred[i])
                        base_sum_sq += v * v
            # torch.nn.functional.normalize(..., eps=1e-12), evaluated in F64.
            var base_norm = sqrt(base_sum_sq)
            if base_norm < 1.0e-12:
                base_norm = 1.0e-12
            var dot: Float64 = 0.0
            for c in range(channels):
                for h in range(height):
                    for w in range(width):
                        var i = _index_5d(b, c, t, h, w, channels, frames, height, width)
                        dot += Float64(diff[i]) * Float64(base_pred[i]) / base_norm
            # orthogonal + eta*parallel == diff + (eta-1)*parallel.
            var projection_scale = Float64(eta - 1.0) * dot / base_norm
            for c in range(channels):
                for h in range(height):
                    for w in range(width):
                        var i = _index_5d(b, c, t, h, w, channels, frames, height, width)
                        out[i] = Float32(
                            Float64(diff[i]) + projection_scale * Float64(base_pred[i])
                        )
    return out^


def bernini_apg_guidance(
    pred_cond: List[Float32], pred_uncond: List[Float32], guidance_scale: Float32,
    mut state: BerniniAPGMomentum, eta: Float32, norm_threshold: Float32,
    batch: Int, channels: Int, frames: Int, height: Int, width: Int,
) raises -> List[Float32]:
    """Creator `normalized_guidance` for Bernini-R T2V/V2V APG."""
    _check_5d_numel(pred_cond, batch, channels, frames, height, width, "cond")
    _check_5d_numel(pred_uncond, batch, channels, frames, height, width, "uncond")
    var diff = List[Float32]()
    for i in range(len(pred_cond)):
        diff.append(pred_cond[i] - pred_uncond[i])
    var normalized = bernini_apg_normalize_diff(
        diff^, pred_cond, state, eta, norm_threshold,
        batch, channels, frames, height, width,
    )
    var out = List[Float32]()
    for i in range(len(pred_uncond)):
        out.append(pred_uncond[i] + guidance_scale * normalized[i])
    return out^
