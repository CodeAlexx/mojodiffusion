# serenitymojo/models/minimax_h3/h3_train_sigma.mojo
#
# MiniMax-H3 training sigma policy, noising, and joint velocity loss.
# Oracle: musubi-tuner akane/minimax-h3 @ 04324c28 —
#   minimax_h3/training.py: _shift_unchecked (:61), prepare_joint_noisy_inputs
#   (:103-179), _modality_loss/_joint_loss (:272-352);
#   minimax_h3_train_network.py: _base_sigma (:1239, uniform => base sigma IS
#   the U[0,1) draw), _apply_timestep_focus (:109), parser defaults (:2506:
#   timestep_sampling=uniform, discrete_flow_shift pinned 1.0).
#
# Recipe (musubi defaults, AV video item):
#   base_sigma u ~ U[0,1)          (shared unshifted coordinate, ONE per item)
#   sigma_m    = shift(u, s_m)     shift(σ,s) = s·σ / (1 + (s−1)·σ)
#                s_video = 12.0, s_audio = 3.0 (images: both 1.0)
#   x_t^m      = (1 − σ_m)·x0 + σ_m·noise_m    (audio noise = SEPARATE draw)
#   target^m   = x0 − noise_m                  (data-pointing velocity)
#   timestep_m = 1 − σ_m
#   loss       = (w_v·Σ_v + w_a·Σ_a) / (w_v·n_v + w_a·n_a)   ("token" balance)
#                Σ_m = masked sum of (pred−target)², n_m = masked element count
#                defaults w_v = w_a = 1.0, no sample weighting
#
# DTYPE CONTRACT (matches torch bf16 semantics exactly): musubi expands sigma
# with .to(dtype=latents.dtype) — σ is bf16-ROUNDED before the noising math,
# and every elementwise op computes in f32 then rounds to bf16 (torch CUDA
# bf16 kernels upcast internally). Our mul_scalar/add/sub bf16 paths do the
# same, so x_t and target are BIT-IDENTICAL to torch given identical inputs.
from max.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.ops.tensor_algebra import add, sub, mul, mul_scalar

comptime H3_VIDEO_FLOW_SHIFT = Float32(12.0)
comptime H3_AUDIO_FLOW_SHIFT = Float32(3.0)
comptime H3_OBSERVED_VIDEO_SIGMA = Float32(0.001)
comptime H3_OBSERVED_AUDIO_SIGMA = Float32(0.0)


def h3_shift_sigma(sigma: Float32, shift: Float32) raises -> Float32:
    """training.py:61 _shift_unchecked — H3's exponential flow shift."""
    if shift <= 0:
        raise Error("h3 flow shift must be positive")
    if sigma < 0 or sigma > 1:
        raise Error("h3 sigma must be in [0, 1]")
    return shift * sigma / (1.0 + (shift - 1.0) * sigma)


def h3_apply_timestep_focus(
    base: Float32, low: Float32, high: Float32, probability: Float32
) -> Float32:
    """minimax_h3_train_network.py:109 — uniform/background mixture from one draw."""
    if probability <= 0.0:
        return base
    if probability >= 1.0:
        return low + (high - low) * base
    if base < probability:
        return low + (high - low) * (base / probability)
    return (base - probability) / (1.0 - probability)


def _bf16_round(x: Float32) -> Float32:
    return Float32(BFloat16(x))


def h3_noisy_input(
    x0: Tensor, noise: Tensor, sigma: Float32, ctx: DeviceContext
) raises -> Tensor:
    """x_t = (1 − σ)·x0 + σ·noise with musubi's exact bf16 rounding: σ is
    bf16-rounded first (torch .to(dtype=bf16) expansion), (1 − σ_bf16) is a
    bf16 tensor-op result, and each multiply/add rounds to bf16."""
    var s = _bf16_round(sigma)
    var one_minus = _bf16_round(Float32(1.0) - s)
    var a = mul_scalar(x0, one_minus, ctx)
    var b = mul_scalar(noise, s, ctx)
    return add(a, b, ctx)


def h3_velocity_target(x0: Tensor, noise: Tensor, ctx: DeviceContext) raises -> Tensor:
    """target = x0 − noise (bf16, one rounding — same as torch)."""
    return sub(x0, noise, ctx)


struct H3ModalityLoss(Copyable, Movable):
    var total: Float64      # masked sum of squared error (f64 accumulation)
    var elements: Int       # masked element count

    def __init__(out self, total: Float64, elements: Int):
        self.total = total
        self.elements = elements


def h3_modality_loss(
    pred: Tensor, target: Tensor, mask: List[Bool], ctx: DeviceContext
) raises -> H3ModalityLoss:
    """Masked sum of (pred−target)² + count. `mask` is per TRAILING position:
    empty = no mask; length == numel/lead means it repeats over leading dims
    (the torch broadcast of [T] over [B,2,32,T] or [F,H,W] over [B,24,F,H,W]).
    The DIFF is taken in bf16 (torch: bf16 sub, then .float().square())."""
    var diff = sub(pred, target, ctx)
    var h = diff.to_host(ctx)
    var n = len(h)
    if len(mask) == 0:
        var s = Float64(0)
        for i in range(n):
            var v = Float64(h[i])
            s += v * v
        return H3ModalityLoss(s, n)
    if n % len(mask) != 0:
        raise Error("h3 loss: mask length does not divide element count")
    var m = len(mask)
    var s = Float64(0)
    var count = 0
    for i in range(n):
        if mask[i % m]:
            var v = Float64(h[i])
            s += v * v
            count += 1
    return H3ModalityLoss(s, count)


def h3_joint_token_loss(
    video: H3ModalityLoss, audio: H3ModalityLoss,
    video_weight: Float64, audio_weight: Float64,
) raises -> Float64:
    """_joint_loss balance="token" (training.py:344-347): weighted totals over
    weighted element counts. Weights zero out per musubi when a modality has
    no active elements."""
    var wv = video_weight if video.elements > 0 else Float64(0)
    var wa = audio_weight if audio.elements > 0 else Float64(0)
    if wv + wa == 0:
        return Float64(0)
    var denom = wv * Float64(video.elements) + wa * Float64(audio.elements)
    return (wv * video.total + wa * audio.total) / denom


def h3_loss_grad(
    pred: Tensor, target: Tensor, mask: Optional[Tensor],
    weight: Float64, denom: Float64, ctx: DeviceContext,
) raises -> Tensor:
    """d loss / d pred = 2·(pred−target)·mask·(w / denom), bf16 out.

    Matches torch's autograd chain through `.float().square()`: diff is the
    bf16 sub result, the f32 scale 2·w/denom applies in f32 and rounds to
    bf16 once at the cast boundary; the 0/1 mask multiply is exact.
    `denom` is the joint token denominator (w_v·n_v + w_a·n_a). `mask` (if
    present) must broadcast against pred's shape (bf16 zeros/ones)."""
    var diff = sub(pred, target, ctx)
    var s = Float32(2.0 * weight / denom)
    var g = mul_scalar(diff, s, ctx)
    if mask:
        return mul(g, mask.value(), ctx)
    return g^
