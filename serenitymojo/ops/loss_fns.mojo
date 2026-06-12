# ops/loss_fns.mojo — host-math training loss functions + min-SNR-γ weight
# (Tier-1 parity campaign phase T1.A, TIER1_PARITY_CAMPAIGN_2026-06-11.md).
#
# Pure host F32 list math (no device): each loss fn takes (pred, target) as
# List[Float32] and returns (loss, d_pred = dloss/dpred). The trainer chains
# its own pred-side sign/weight (zimage: pred = -raw_out → d_out = -d_pred)
# and pads the seq tail with zeros, exactly like the existing MSE block.
#
# Accumulation discipline: per-element math in F32 (what torch's F32 kernels
# see), reductions accumulated in F64 then cast to F32 — same pattern as the
# existing trainer loss blocks (train_zimage_real.mojo `sse += Float64(...)`)
# and well inside the 1e-6 rel gate vs torch's F32 mean reduction.
#
# ── Formulas (gated vs torch oracle, ops/tests/loss_fns_parity.mojo) ──────────
# MSE (torch.nn.functional.mse_loss, reduction='mean') — the existing trainers'
# math, kept for the oracle baseline:
#     loss   = mean((pred - target)^2)
#     d_pred = 2*(pred - target)/N
#
# Huber (torch.nn.functional.huber_loss(delta=δ), reduction='mean'):
#     d      = pred - target
#     loss_i = 0.5*d^2            if |d| <= δ
#            = δ*(|d| - 0.5*δ)    otherwise
#     d_pred = clamp(d, -δ, δ)/N
# NOTE divergence from SimpleTuner's OWN "huber": SimpleTuner implements the
# kohya pseudo-Huber  2c·(sqrt(d²+c²) − c)  (simpletuner/helpers/models/
# common.py:4470-4472) — phase T1.A spec mandates torch semantics; the
# pseudo-Huber is a separate lever if ever wanted. Also note SimpleTuner's
# FLOW_MATCHING branch always uses plain L2 (common.py:4562-4564); huber/
# smooth_l1 there only apply to epsilon/v-pred models (common.py:4600-4602).
#
# Smooth-L1 (torch.nn.functional.smooth_l1_loss(beta=β), reduction='mean'):
#     loss_i = 0.5*d^2/β          if |d| < β
#            = |d| - 0.5*β        otherwise
#     d_pred = (d/β)/N            if |d| < β
#            = sign(d)/N          otherwise            (β > 0 required)
#
# Masked weighted losses (phase T1.E) — per-element weights INTO the loss and
# its grad BEFORE the mean reduction:
#     loss   = mean(w_i * loss_i)            [/ mean(w) if normalize]
#     d_pred = w_i * dloss_i/dpred_i / N     [/ mean(w) if normalize]
# SimpleTuner: per-element loss (reduction='none', simpletuner/helpers/models/
# common.py:4564 flow L2 / :4629-4635 huber), `loss = loss * mask_image`
# (common.py:4694), then per-sample mean + batch mean (common.py:4704-4707) —
# i.e. weights == mask, no normalize. The unmasked_weight / normalize levers
# (the serenity-trainer UI keys) are OneTrainer modules/util/loss/
# masked_loss.py:11-18: weights = clamp(mask, unmasked_weight, 1)
# (mask_weights below), normalize divides by clamped_mask.mean(dim=(1,2,3),
# keepdim=True) — for one flattened sample that's the global mean(w). The
# mask is CONSTANT wrt pred, so the normalize divisor just scales the grad.
# Gated by ops/tests/masked_loss_parity.mojo vs a torch-autograd oracle.
#
# min-SNR-γ for FLOW-MATCH sigma ∈ (0,1] (Hang et al. 2023 adapted):
#     SNR(σ) = ((1-σ)/σ)^2
#     w      = min(SNR, γ)/SNR
# SimpleTuner match: snr = (alpha/sigma_noise)^2 with alpha=sqrt(ᾱ),
# sigma_noise=sqrt(1-ᾱ) (simpletuner/helpers/training/min_snr_gamma.py:40);
# flow-match interpolant x_t = σ·noise + (1-σ)·x0 ⇒ alpha=1-σ, sigma_noise=σ
# ⇒ SNR = ((1-σ)/σ)^2. Weight = min(snr, γ·1)/snr_divisor with
# snr_divisor = snr for non-v-pred models (simpletuner/helpers/models/
# common.py:4660-4672, the torch.stack(...).min(dim=1)[0]/snr_divisor line;
# the v-pred branch's snr+1 divisor at :4662-4663 is NOT taken here — phase
# spec mandates the ε-style min(SNR,γ)/SNR form). Same SNR(σ) definition as
# training/loss_weight.mojo:31 (_snr_from_sigma, EDv2 loss_weight.rs:39);
# σ and SNR floored at 1e-8 like there so σ→1 (SNR→0) yields w→1, not NaN.
#
# Mojo 1.0.0b1. Host-only — no GPU imports.


comptime _SIGMA_FLOOR = Float32(1.0e-8)
comptime _SNR_FLOOR = Float32(1.0e-8)


struct LossGrad(Movable):
    """A loss value + its gradient wrt pred (mean reduction)."""

    var loss: Float32
    var d_pred: List[Float32]

    def __init__(out self, loss: Float32, var d_pred: List[Float32]):
        self.loss = loss
        self.d_pred = d_pred^


def _check_inputs(pred: List[Float32], target: List[Float32], name: String) raises:
    if len(pred) != len(target):
        raise Error(name + ": pred/target len mismatch")
    if len(pred) == 0:
        raise Error(name + ": empty input")


# ── MSE (the existing trainers' math; oracle baseline) ───────────────────────
def mse_loss_grad(pred: List[Float32], target: List[Float32]) raises -> LossGrad:
    """loss = mean((pred-target)^2); d_pred = 2*(pred-target)/N.

    Matches torch.nn.functional.mse_loss(reduction='mean') value + autograd,
    and the existing train_zimage_real.mojo loss block element math."""
    _check_inputs(pred, target, "mse_loss_grad")
    var n = len(pred)
    var inv_n2 = Float32(2.0) / Float32(n)
    var sse = 0.0
    var d_pred = List[Float32]()
    for i in range(n):
        var d = pred[i] - target[i]
        sse += Float64(d) * Float64(d)
        d_pred.append(inv_n2 * d)
    return LossGrad(Float32(sse / Float64(n)), d_pred^)


# ── Huber (torch huber_loss(delta=δ), mean reduction) ────────────────────────
def huber_loss_grad(
    pred: List[Float32], target: List[Float32], delta: Float32
) raises -> LossGrad:
    """loss_i = 0.5*d^2 if |d|<=δ else δ*(|d|-0.5*δ); d_pred = clamp(d,-δ,δ)/N.

    Matches torch.nn.functional.huber_loss(delta=δ, reduction='mean') value +
    autograd (same grad form as ops/loss_swiglu_backward.mojo huber_backward)."""
    _check_inputs(pred, target, "huber_loss_grad")
    if delta <= Float32(0.0):
        raise Error("huber_loss_grad: delta must be > 0")
    var n = len(pred)
    var inv_n = Float32(1.0) / Float32(n)
    var acc = 0.0
    var d_pred = List[Float32]()
    for i in range(n):
        var d = pred[i] - target[i]
        var a = abs(d)
        if a <= delta:
            acc += Float64(Float32(0.5) * d * d)
        else:
            acc += Float64(delta * (a - Float32(0.5) * delta))
        var c = d
        if c > delta:
            c = delta
        elif c < -delta:
            c = -delta
        d_pred.append(c * inv_n)
    return LossGrad(Float32(acc / Float64(n)), d_pred^)


# ── Smooth-L1 (torch smooth_l1_loss(beta=β), mean reduction) ─────────────────
def smooth_l1_loss_grad(
    pred: List[Float32], target: List[Float32], beta: Float32
) raises -> LossGrad:
    """loss_i = 0.5*d^2/β if |d|<β else |d|-0.5*β; grad (d/β)/N | sign(d)/N.

    Matches torch.nn.functional.smooth_l1_loss(beta=β, reduction='mean') value
    + autograd (strict |d| < β for the quadratic branch, as torch)."""
    _check_inputs(pred, target, "smooth_l1_loss_grad")
    if beta <= Float32(0.0):
        raise Error("smooth_l1_loss_grad: beta must be > 0")
    var n = len(pred)
    var inv_n = Float32(1.0) / Float32(n)
    var acc = 0.0
    var d_pred = List[Float32]()
    for i in range(n):
        var d = pred[i] - target[i]
        var a = abs(d)
        if a < beta:
            acc += Float64(Float32(0.5) * d * d / beta)
            d_pred.append((d / beta) * inv_n)
        else:
            acc += Float64(a - Float32(0.5) * beta)
            var s = Float32(0.0)
            if d > Float32(0.0):
                s = Float32(1.0)
            elif d < Float32(0.0):
                s = Float32(-1.0)
            d_pred.append(s * inv_n)
    return LossGrad(Float32(acc / Float64(n)), d_pred^)


# ── masked weighted losses (T1.E; see header for ST/OneTrainer citations) ────
def mask_weights(
    mask: List[Float32], unmasked_weight: Float32
) raises -> List[Float32]:
    """Per-element loss weights from a mask: w_i = clamp(mask_i, uw, 1).

    OneTrainer masked_loss.py:11 `torch.clamp(mask, unmasked_weight, 1)`
    (1:1-ported precedent: serenity-trainer util/loss/masked_loss.mojo). For
    binary masks this equals the lerp form mask + (1-mask)*uw. uw=0 with a
    [0,1] mask degenerates to weights == mask, SimpleTuner common.py:4694."""
    if len(mask) == 0:
        raise Error("mask_weights: empty mask")
    var out = List[Float32]()
    for i in range(len(mask)):
        var w = mask[i]
        if w < unmasked_weight:
            w = unmasked_weight
        if w > Float32(1.0):
            w = Float32(1.0)
        out.append(w)
    return out^


def _check_weights(
    pred: List[Float32], weights: List[Float32], name: String
) raises:
    if len(weights) != len(pred):
        raise Error(name + ": weights len mismatch")


def _normalize_in_place(
    mut loss64: Float64, mut d_pred: List[Float32], wsum: Float64, n: Int,
    name: String,
) raises:
    """Divide loss + grad by mean(weights) — OneTrainer masked_loss.py:15-16
    (the mask is constant wrt pred, so the grad scales by the same factor)."""
    var wm = wsum / Float64(n)
    if wm <= 0.0:
        raise Error(name + ": normalize_masked_area_loss with mask mean <= 0")
    loss64 = loss64 / wm
    var s = Float32(1.0 / wm)
    for i in range(len(d_pred)):
        d_pred[i] = s * d_pred[i]


def masked_mse_loss_grad(
    pred: List[Float32], target: List[Float32], weights: List[Float32],
    normalize: Bool,
) raises -> LossGrad:
    """loss = mean(w*(pred-target)^2) [/mean(w)]; d_pred = 2*w*d/N [/mean(w)].

    SimpleTuner common.py:4564+4694+4704-4707 (weights==mask, normalize=False);
    normalize per OneTrainer masked_loss.py:15-16. Existing mse_loss_grad is
    untouched — this is the flag-gated masked path."""
    _check_inputs(pred, target, "masked_mse_loss_grad")
    _check_weights(pred, weights, "masked_mse_loss_grad")
    var n = len(pred)
    var inv_n = Float32(1.0) / Float32(n)
    var acc = 0.0
    var wsum = 0.0
    var d_pred = List[Float32]()
    for i in range(n):
        var d = pred[i] - target[i]
        var w = weights[i]
        acc += Float64(w) * Float64(d) * Float64(d)
        wsum += Float64(w)
        d_pred.append(Float32(2.0) * w * d * inv_n)
    var loss64 = acc / Float64(n)
    if normalize:
        _normalize_in_place(loss64, d_pred, wsum, n, "masked_mse_loss_grad")
    return LossGrad(Float32(loss64), d_pred^)


def masked_huber_loss_grad(
    pred: List[Float32], target: List[Float32], weights: List[Float32],
    delta: Float32, normalize: Bool,
) raises -> LossGrad:
    """Torch-semantics huber elements (huber_loss_grad math) with per-element
    weights before the mean: loss = mean(w*l_i) [/mean(w)];
    d_pred = w*clamp(d,-δ,δ)/N [/mean(w)]."""
    _check_inputs(pred, target, "masked_huber_loss_grad")
    _check_weights(pred, weights, "masked_huber_loss_grad")
    if delta <= Float32(0.0):
        raise Error("masked_huber_loss_grad: delta must be > 0")
    var n = len(pred)
    var inv_n = Float32(1.0) / Float32(n)
    var acc = 0.0
    var wsum = 0.0
    var d_pred = List[Float32]()
    for i in range(n):
        var d = pred[i] - target[i]
        var a = abs(d)
        var w = weights[i]
        var li: Float32
        if a <= delta:
            li = Float32(0.5) * d * d
        else:
            li = delta * (a - Float32(0.5) * delta)
        acc += Float64(w) * Float64(li)
        wsum += Float64(w)
        var c = d
        if c > delta:
            c = delta
        elif c < -delta:
            c = -delta
        d_pred.append(w * c * inv_n)
    var loss64 = acc / Float64(n)
    if normalize:
        _normalize_in_place(loss64, d_pred, wsum, n, "masked_huber_loss_grad")
    return LossGrad(Float32(loss64), d_pred^)


def masked_smooth_l1_loss_grad(
    pred: List[Float32], target: List[Float32], weights: List[Float32],
    beta: Float32, normalize: Bool,
) raises -> LossGrad:
    """Torch-semantics smooth-L1 elements (smooth_l1_loss_grad math) with
    per-element weights before the mean reduction."""
    _check_inputs(pred, target, "masked_smooth_l1_loss_grad")
    _check_weights(pred, weights, "masked_smooth_l1_loss_grad")
    if beta <= Float32(0.0):
        raise Error("masked_smooth_l1_loss_grad: beta must be > 0")
    var n = len(pred)
    var inv_n = Float32(1.0) / Float32(n)
    var acc = 0.0
    var wsum = 0.0
    var d_pred = List[Float32]()
    for i in range(n):
        var d = pred[i] - target[i]
        var a = abs(d)
        var w = weights[i]
        wsum += Float64(w)
        if a < beta:
            acc += Float64(w) * Float64(Float32(0.5) * d * d / beta)
            d_pred.append(w * (d / beta) * inv_n)
        else:
            acc += Float64(w) * Float64(a - Float32(0.5) * beta)
            var s = Float32(0.0)
            if d > Float32(0.0):
                s = Float32(1.0)
            elif d < Float32(0.0):
                s = Float32(-1.0)
            d_pred.append(w * s * inv_n)
    var loss64 = acc / Float64(n)
    if normalize:
        _normalize_in_place(
            loss64, d_pred, wsum, n, "masked_smooth_l1_loss_grad"
        )
    return LossGrad(Float32(loss64), d_pred^)


# ── min-SNR-γ weight for flow-match sigma (see header for the ST citation) ───
def min_snr_gamma_weight(sigma: Float32, gamma: Float32) -> Float32:
    """w = min(SNR, γ)/SNR with SNR = ((1-σ)/σ)^2, σ ∈ (0,1].

    SimpleTuner: min_snr_gamma.py:40 snr=(alpha/sigma)^2 + common.py:4660-4672
    weights = min(snr, γ)/snr_divisor, snr_divisor = snr (non-v-pred branch).
    σ floored at 1e-8 (training/loss_weight.mojo precedent). For γ > 0 and
    SNR < 1e-8 (σ→1) min(SNR,γ) == SNR so the exact limit is 1.0 — returned
    directly to avoid 0/0."""
    var s = sigma
    if s < _SIGMA_FLOOR:
        s = _SIGMA_FLOOR
    var r = (Float32(1.0) - s) / s
    var snr = r * r
    var cap = snr
    if gamma < cap:
        cap = gamma
    if snr < _SNR_FLOOR:
        # SNR ≈ 0 (σ ≈ 1): cap == snr (γ >= 0 > snr only if γ negative — γ>0
        # here), min(SNR,γ)/SNR == 1 in the limit. Return 1 exactly.
        return Float32(1.0)
    return cap / snr
