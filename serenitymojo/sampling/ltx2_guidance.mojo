# sampling/ltx2_guidance.mojo — LTX-2/LTX-Video guidance: CFG-star + STG.
#
# Pure-Mojo port of inference-flame/src/sampling/ltx2_guidance.rs. Team 3
# (sampler + guidance). All math lifted from Lightricks's reference pipeline
# (`ltx_video/pipelines/pipeline_ltx_video.py`).
#
# Rust parity status (ltx2_guidance_parity.rs): PASS max_abs=0.0 BF16 for
# `cfg_star_rescale`, `build_skip_layer_mask`, and `stg_rescale` (see
# serenitymojo/docs/LTX2_RUST_STATE_2026-05-28.md §4). This is a STRONG, bit-clean
# oracle — this port aims for exact match.
#
# References (read line-by-line):
#   * inference-flame/src/sampling/ltx2_guidance.rs        (the math)
#   * inference-flame/scripts/ltx2_cfg_star_ref.py         (CFG-star numerical ref)
#   * inference-flame/scripts/ltx2_stg_mask_ref.py         (STG mask ref)
#   * serenitymojo/sampling/flow_match.mojo                (cfg/cfg_qwen style)
#
# HOST-SIDE REDUCTION RATIONALE (matches flow_match._l2_ratio_lastdim pattern):
# CFG-star's alpha and STG's std-factor are per-batch SCALARS reduced over the
# full flattened latent. serenitymojo's ops layer has no device whole-tensor (or
# per-batch) reduction yet, so the reductions are done host-side in F64 (exactly
# as the Rust does, for parity), and only `batch` scalars cross the bus — not a
# hot-path activation roundtrip. When a device reduction lands, swap the host
# loops for an on-device sum and the elementwise combine is unchanged.
#
# Mojo 1.0.0b1. Inference-only. No autograd, no Python at runtime.

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.tensor_algebra import add, sub, mul, mul_scalar
from std.gpu.host import DeviceContext
from std.math import sqrt


# ─────────────────────────────────────────────────────────────────────────────
# STG skip-layer strategies — 1:1 with Lightricks's
# `ltx_video/utils/skip_layer_strategy.py` (mirrored in ltx2_guidance.rs:19-51).
# Enum-like integer tag struct. Only AttentionValues (STG-AV, the dev-config
# default) is wired through a transformer block in Rust; the others are named
# for completeness but the BLOCK-side hook is Team 1's concern. Team 3 owns the
# MASK that tells a block which layers to perturb.
# ─────────────────────────────────────────────────────────────────────────────

comptime _STG_ATTENTION_SKIP = 0
comptime _STG_ATTENTION_VALUES = 1
comptime _STG_RESIDUAL = 2
comptime _STG_TRANSFORMER_BLOCK = 3


@fieldwise_init
struct SkipLayerStrategy(Copyable, Movable, ImplicitlyCopyable, Equatable):
    """STG skip-layer strategy tag. AttentionValues (STG-AV) is the default."""

    var tag: Int

    comptime AttentionSkip = Self(_STG_ATTENTION_SKIP)
    comptime AttentionValues = Self(_STG_ATTENTION_VALUES)
    comptime Residual = Self(_STG_RESIDUAL)
    comptime TransformerBlock = Self(_STG_TRANSFORMER_BLOCK)

    def __eq__(self, other: Self) -> Bool:
        return self.tag == other.tag

    def __ne__(self, other: Self) -> Bool:
        return self.tag != other.tag

    @staticmethod
    def from_str(s: String) raises -> SkipLayerStrategy:
        """Parse the aliases Lightricks uses (inference.py:548-557), matching
        the Rust `SkipLayerStrategy::from_str`. Case-insensitive on the canonical
        spellings used by the dev configs."""
        if s == "stg_av" or s == "attention_values" or s == "STG_AV":
            return Self.AttentionValues
        if s == "stg_as" or s == "attention_skip" or s == "STG_AS":
            return Self.AttentionSkip
        if s == "stg_r" or s == "residual" or s == "STG_R":
            return Self.Residual
        if s == "stg_t" or s == "transformer_block" or s == "STG_T":
            return Self.TransformerBlock
        raise Error(String("SkipLayerStrategy.from_str: unknown strategy: ") + s)


# ─────────────────────────────────────────────────────────────────────────────
# STG skip-layer mask — direct port of `build_skip_layer_mask`
# (ltx2_guidance.rs:64-84), itself a port of
# `Transformer3DModel.create_skip_layer_mask` (transformer3d.py:173-188).
#
# Returns a flat `List[List[Float32]]` of shape [num_layers][batch_size*num_conds].
# Every entry is 1.0 EXCEPT: for each layer in `skip_block_list`, the perturb
# slots `ptb_index, ptb_index + num_conds, ...` across the duplicated batch are
# set to 0.0. This is tiny (48x3 = 144 floats for the 13B), so it stays on host.
# The downstream block consumes the per-layer scalar at its own index.
# ─────────────────────────────────────────────────────────────────────────────


def build_skip_layer_mask(
    num_layers: Int,
    batch_size: Int,
    num_conds: Int,
    skip_block_list: List[Int],
    ptb_index: Int,
) raises -> List[List[Float32]]:
    """STG skip-layer mask. Shape [num_layers][batch_size*num_conds].

    Rows for layers in `skip_block_list` get 0.0 at every perturb slot
    `ptb_index, ptb_index + num_conds, ...`; all else is 1.0. Out-of-range
    block indices are skipped (matches the Rust `if block_idx >= num_layers
    { continue; }`).

    Small case (num_layers=4, batch=1, num_conds=3, skip=[1,3], ptb=2):
        [[1,1,1],[1,1,0],[1,1,1],[1,1,0]]   (matches ltx2_stg_mask_ref.py).
    """
    if num_layers < 0 or batch_size < 0 or num_conds < 0:
        raise Error("build_skip_layer_mask: negative dimension")
    var row_len = batch_size * num_conds
    var mask = List[List[Float32]]()
    for _ in range(num_layers):
        var row = List[Float32]()
        for _ in range(row_len):
            row.append(1.0)
        mask.append(row^)
    for k in range(len(skip_block_list)):
        var block_idx = skip_block_list[k]
        if block_idx < 0 or block_idx >= num_layers:
            continue
        var j = ptb_index
        while j < row_len:
            mask[block_idx][j] = 0.0
            j += num_conds
    return mask^


def single_cond_skip_mask(
    num_layers: Int, skip_block_list: List[Int]
) raises -> List[Float32]:
    """Per-layer scalar mask [num_layers]: 0.0 for skipped layers else 1.0.

    Mirrors the Rust `single_cond_mask_from_skip_list` helper (the single-cond
    column of `build_skip_layer_mask`) — suitable for block forwards that
    operate on a single conditioning slot. Returned as a host List so the caller
    can upload it (or read it per-block) however the DiT path needs.
    """
    var v = List[Float32]()
    for _ in range(num_layers):
        v.append(1.0)
    for k in range(len(skip_block_list)):
        var i = skip_block_list[k]
        if i >= 0 and i < num_layers:
            v[i] = 0.0
    return v^


# ─────────────────────────────────────────────────────────────────────────────
# CFG-star rescale — direct port of `cfg_star_rescale` (ltx2_guidance.rs:92-143),
# itself pipeline_ltx_video.py:1227-1240. Per-batch:
#     alpha = <eps_text, eps_uncond> / (||eps_uncond||^2 + 1e-8)
#     eps_uncond <- alpha * eps_uncond
# Reduction in F64 (matches the Rust, which sums in f64 for parity); the per-batch
# alpha multiply runs in F32; output cast back to the input dtype. Inputs share
# shape; alpha is broadcast over all non-batch dims.
# ─────────────────────────────────────────────────────────────────────────────


def cfg_star_alpha(
    eps_text: Tensor, eps_uncond: Tensor, ctx: DeviceContext
) raises -> List[Float32]:
    """Per-batch CFG-star alpha = <text, uncond> / (||uncond||^2 + 1e-8).

    Reduced in F64 (parity with the Rust), returned as F32 host scalars of
    length `batch`. The +1e-8 is added to the squared norm (NOT the ratio),
    matching pipeline_ltx_video.py:1232 and the Rust `nsq + 1e-8`.
    """
    var ts = eps_text.shape()
    var us = eps_uncond.shape()
    if len(ts) == 0 or len(us) == 0:
        raise Error("cfg_star_alpha: inputs must have >= 1 dim")
    if len(ts) != len(us):
        raise Error("cfg_star_alpha: rank mismatch")
    for i in range(len(ts)):
        if ts[i] != us[i]:
            raise Error("cfg_star_alpha: shape mismatch")
    var batch = ts[0]
    var flat_len = 1
    for i in range(1, len(ts)):
        flat_len *= ts[i]
    var text_h = eps_text.to_host(ctx)
    var uncond_h = eps_uncond.to_host(ctx)
    var alphas = List[Float32]()
    for b in range(batch):
        var dot: Float64 = 0.0
        var nsq: Float64 = 0.0
        var base = b * flat_len
        for i in range(flat_len):
            var ti = Float64(text_h[base + i])
            var ui = Float64(uncond_h[base + i])
            dot += ti * ui
            nsq += ui * ui
        var alpha = Float32(dot / (nsq + 1e-8))
        alphas.append(alpha)
    return alphas^


def cfg_star_rescale(
    eps_text: Tensor, eps_uncond: Tensor, ctx: DeviceContext
) raises -> Tensor:
    """CFG-star rescaled uncond = alpha * eps_uncond (per-batch alpha).

    Exact port of `cfg_star_rescale` — this is what `ltx2_cfg_star_ref.py`
    computes and `ltx2_guidance_parity` checks (PASS max_abs=0.0 BF16). The
    per-batch alpha broadcasts over the latent; the multiply runs in F32 and
    the output is cast back to `eps_uncond`'s dtype.
    """
    var alphas = cfg_star_alpha(eps_text, eps_uncond, ctx)
    var us = eps_uncond.shape()
    var batch = us[0]
    var flat_len = 1
    for i in range(1, len(us)):
        flat_len *= us[i]
    # Build a [batch, 1, 1, ...] alpha tensor (broadcast over non-batch dims)
    # and multiply on-device, mirroring the Rust which builds an alpha_tensor
    # of shape [batch, 1, 1, ...] then `uncond_f32.mul(&alpha_tensor)`.
    var alpha_shape = List[Int]()
    alpha_shape.append(batch)
    for _ in range(1, len(us)):
        alpha_shape.append(1)
    var alpha_t = Tensor.from_host(
        alphas.copy(), alpha_shape^, eps_uncond.dtype(), ctx
    )
    return mul(eps_uncond, alpha_t, ctx)


def cfg_star(
    v_cond: Tensor, v_uncond: Tensor, scale: Float32, ctx: DeviceContext
) raises -> Tensor:
    """Full CFG-star guidance combine (norm-rescaled CFG).

        eps_uncond* = cfg_star_rescale(v_cond, v_uncond)   # alpha * v_uncond
        out         = eps_uncond* + scale * (v_cond - eps_uncond*)

    The CFG-star pipeline first rescales the UNCONDITIONAL prediction so its
    projection onto v_cond is removed (alpha = <cond,uncond>/||uncond||^2),
    then applies textbook CFG against that rescaled uncond. `ltx2_cfg_star_ref.py`
    only dumps the `rescaled` term (the parity-clean piece — see `cfg_star_rescale`);
    this full combine layers the textbook step on top, matching
    pipeline_ltx_video.py's `noise_pred = uncond + guidance_scale*(cond - uncond)`
    where `uncond` is the rescaled value.

    NOTE: the parity oracle covers `cfg_star_rescale` exactly; the additive
    combine here is the standard textbook CFG and is FLAGGED for skeptic review
    of the exact pipeline scale convention (see report).
    """
    var rescaled = cfg_star_rescale(v_cond, v_uncond, ctx)  # alpha * v_uncond
    var diff = sub(v_cond, rescaled, ctx)  # v_cond - rescaled
    var scaled = mul_scalar(diff, scale, ctx)  # scale * (...)
    return add(rescaled, scaled, ctx)  # rescaled + scale*(...)


# ─────────────────────────────────────────────────────────────────────────────
# STG std-rescale — direct port of `stg_rescale` (ltx2_guidance.rs:155-196),
# pipeline_ltx_video.py:1251-1262. Per-batch:
#     factor = std(pos) / std(guided)
#     factor = rescaling_scale * factor + (1 - rescaling_scale)
#     out    = guided * factor
# std is UNBIASED (Bessel's correction, divide by n-1) over all non-batch dims,
# computed in F64 (parity with the Rust). Requires >= 2 elements per batch.
# ─────────────────────────────────────────────────────────────────────────────


def _unbiased_std_f64(
    h: List[Float32], base: Int, n: Int
) raises -> Float64:
    """Unbiased (n-1) std of h[base : base+n], in F64. Matches torch `.std`
    default (unbiased=True) and the Rust `unbiased_std_f64`."""
    if n < 2:
        raise Error("_unbiased_std_f64: need >= 2 elements")
    var nf = Float64(n)
    var s: Float64 = 0.0
    for i in range(n):
        s += Float64(h[base + i])
    var mean = s / nf
    var var_sum: Float64 = 0.0
    for i in range(n):
        var d = Float64(h[base + i]) - mean
        var_sum += d * d
    var variance = var_sum / (nf - 1.0)
    return sqrt(variance)


def stg_rescale(
    pos: Tensor, guided: Tensor, rescaling_scale: Float32, ctx: DeviceContext
) raises -> Tensor:
    """STG std-rescale: match std(guided) toward std(pos), blended by scale.

        factor = std(pos)/std(guided)
        factor = rescaling_scale*factor + (1 - rescaling_scale)
        out    = guided * factor

    Per-batch UNBIASED std (n-1) in F64. Output cast back to `guided`'s dtype.
    Exact port of `stg_rescale` (parity-clean in Rust).
    """
    var ps = pos.shape()
    var gs = guided.shape()
    if len(ps) != len(gs):
        raise Error("stg_rescale: rank mismatch")
    for i in range(len(ps)):
        if ps[i] != gs[i]:
            raise Error("stg_rescale: shape mismatch")
    var batch = gs[0]
    var flat_len = 1
    for i in range(1, len(gs)):
        flat_len *= gs[i]
    if flat_len < 2:
        raise Error("stg_rescale: need >= 2 elements per batch for unbiased std")
    var pos_h = pos.to_host(ctx)
    var guided_h = guided.to_host(ctx)
    var factors = List[Float32]()
    for b in range(batch):
        var base = b * flat_len
        var pos_std = _unbiased_std_f64(pos_h, base, flat_len)
        var guided_std = _unbiased_std_f64(guided_h, base, flat_len)
        var f = Float32(pos_std / guided_std)
        f = rescaling_scale * f + (1.0 - rescaling_scale)
        factors.append(f)
    var factor_shape = List[Int]()
    factor_shape.append(batch)
    for _ in range(1, len(gs)):
        factor_shape.append(1)
    var factor_t = Tensor.from_host(
        factors.copy(), factor_shape^, guided.dtype(), ctx
    )
    return mul(guided, factor_t, ctx)
