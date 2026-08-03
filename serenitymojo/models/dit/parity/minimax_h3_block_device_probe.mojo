# serenitymojo/models/dit/parity/minimax_h3_block_device_probe.mojo
#
# DEVICE (GPU) build+run probe for `minimax_h3_block_forward`
# (models/dit/minimax_h3_dit.mojo). NOT a parity gate: random weights, no
# oracle comparison, no claim of numeric correctness — see
# `minimax_h3_block_device_gate.mojo` (this directory) for the actual
# host-float32 oracle contract this device path is gated against (currently
# PASSING at cos=0.9999968778871595 on the tiny fixture).
#
# What this proves: the block forward COMPILES and RUNS end to end on the
# RELEASED model's real geometry (hidden 5376, heads 56, head_dim 128,
# ffn 14336, rotary_dim 96 of 128) with a small packed sequence spanning all
# three modalities and two distinct timesteps (exercising the adaLN row
# gather this model's header calls out as the easy-to-get-wrong part), the
# partial+head-broadcast rope, and the cuDNN Dh=128 attention path — and
# that the output is finite.
#
# WEIGHT CONTRACT: `minimax_h3_block_forward` expects qkv_proj/fc1 already
# pre-transformed by `minimax_h3_loader_device.mojo::minimax_h3_load_block_
# device` (de-interleaved qkv, value-first fc1), AND now enforces this via
# `minimax_h3_require_transformed_weights` — two marker keys the loader
# stamps and this function's first statement checks for
# (models/dit/minimax_h3_dit.mojo). This probe fills every real tensor with
# i.i.d. N(0,1) noise (row-permutation-INVARIANT, so it cannot exercise the
# row-order contract numerically — that needs the parity gate) and stamps
# the same two marker keys directly rather than routing through the real
# loader, since it never opens a checkpoint. See
# `minimax_h3_transform_guard_probe.mojo` (this directory) for the guard's
# own positive/negative proof against a real on-disk fixture via
# `ShardedSafeTensors`.
#
# Run (package-relative imports need -I .; sdpa_flash_infer_fwd needs the
# cuDNN shim explicitly linked — bare `mojo run -I .` fails with
# "JIT session error: Symbols not found: [ flame_cudnn_sdpa_bf16 ]"):
#   cd /home/alex/mojodiffusion && pixi run mojo run -I . -Xlinker -lm \
#     -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa \
#     serenitymojo/models/dit/parity/minimax_h3_block_device_probe.mojo

from std.collections import Dict, List
from std.memory import ArcPointer
from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.random import randn
from serenitymojo.models.minimax_h3.dit_frontend import (
    minimax_h3_rope_inv_freq,
    minimax_h3_rope_table,
)
from serenitymojo.models.dit.minimax_h3_dit import (
    MiniMaxH3DiTConfig,
    MINIMAX_H3_QKV_DEINTERLEAVED_MARKER,
    MINIMAX_H3_FC1_SWAPPED_MARKER,
    minimax_h3_released_config,
    minimax_h3_block_tensor_names,
    minimax_h3_expected_shape,
    minimax_h3_check_block_weights,
    minimax_h3_adaln_rows,
    minimax_h3_block_forward,
)

comptime S = 12


def _rand_block_weights(
    config: MiniMaxH3DiTConfig, layer: Int, ctx: DeviceContext
) raises -> Dict[String, ArcPointer[Tensor]]:
    """Every tensor `minimax_h3_check_block_weights` requires for `layer`,
    filled with deterministic N(0,1) noise at the RELEASED shapes, PLUS the
    two transform markers `minimax_h3_load_block_device` would stamp — this
    probe never opens a checkpoint, so it stamps them directly rather than
    routing through the real loader, matching what a real load produces."""
    var weights = Dict[String, ArcPointer[Tensor]]()
    var names = minimax_h3_block_tensor_names(layer)
    var seed: UInt64 = 1000
    for i in range(len(names)):
        var n = names[i]
        var shape = minimax_h3_expected_shape(n, config)
        var t = randn(shape^, seed, STDtype.BF16, ctx)
        weights[n] = ArcPointer(t^)
        seed += 1
    weights[MINIMAX_H3_QKV_DEINTERLEAVED_MARKER] = ArcPointer(
        Tensor.from_host([Float32(0.0)], [1], STDtype.BF16, ctx)
    )
    weights[MINIMAX_H3_FC1_SWAPPED_MARKER] = ArcPointer(
        Tensor.from_host([Float32(0.0)], [1], STDtype.BF16, ctx)
    )
    return weights^


def main() raises:
    var ctx = DeviceContext()
    var config = minimax_h3_released_config()
    config.validate()

    var layer = 3
    var weights = _rand_block_weights(config, layer, ctx)
    minimax_h3_check_block_weights(weights, layer, config)
    print("preflight OK: layer", layer, "block tensors present, shaped, BF16")

    # ── packed sequence: 6 video rows (2 timesteps) + 3 text + 3 audio ──
    var token_tags = List[Int]()
    var timestep_indices = List[Int]()
    for r in range(S):
        if r < 6:
            token_tags.append(0)                       # video
            timestep_indices.append(0 if r < 3 else 1)
        elif r < 9:
            token_tags.append(1)                        # text (single ts)
            timestep_indices.append(0)
        else:
            token_tags.append(2)                        # audio (single ts)
            timestep_indices.append(1)
    var adaln_indices = minimax_h3_adaln_rows(timestep_indices, token_tags)
    print("adaln_indices built, len", len(adaln_indices))

    # 2 distinct timesteps -> mod rows = 2*3 = 6
    var mod_shape: List[Int] = [6, 6 * config.hidden_size]
    var mod = randn(mod_shape^, 4242, STDtype.F32, ctx)

    # ── real rope table (unit 5, gated separately) over a trivial packed grid ──
    var inv_freq = minimax_h3_rope_inv_freq(config.rope_inv_freq_len)
    var position_ids = List[Float64]()
    for r in range(S):
        position_ids.append(Float64(r))
        position_ids.append(Float64(0))
        position_ids.append(Float64(0))
    var rope = minimax_h3_rope_table(position_ids, S, inv_freq)
    print("rope table built, rotary_dim", rope.rotary_dim)
    var cos_shape: List[Int] = [S, rope.rotary_dim]
    var sin_shape: List[Int] = [S, rope.rotary_dim]
    var cos_t = Tensor.from_host(rope.cos, cos_shape^, STDtype.F32, ctx)
    var sin_t = Tensor.from_host(rope.sin, sin_shape^, STDtype.F32, ctx)

    var x_shape: List[Int] = [1, S, config.hidden_size]
    var x = randn(x_shape^, 7, STDtype.BF16, ctx)

    var out = minimax_h3_block_forward[S](
        x, weights, layer, config, mod, adaln_indices, cos_t, sin_t,
        rope.rotary_dim, ctx,
    )

    var out_shape = out.shape()
    if len(out_shape) != 3 or out_shape[0] != 1 or out_shape[1] != S or out_shape[2] != config.hidden_size:
        raise Error("minimax_h3_block_forward probe: unexpected output shape")

    var host = out.to_host(ctx)
    var n_bad = 0
    var max_abs = Float32(0.0)
    for i in range(len(host)):
        var v = host[i]
        if v != v:                  # NaN (IEEE 754: NaN != NaN)
            n_bad += 1
        var av = v if v >= Float32(0.0) else -v
        if av > max_abs:
            max_abs = av
    print("out shape:", out_shape[0], out_shape[1], out_shape[2])
    print("elements:", len(host), "nan_count:", n_bad, "max_abs:", max_abs)
    if n_bad != 0:
        raise Error("minimax_h3_block_forward probe: NaN in output")

    # ── differential check: collapsing every row to ONE adaln index must
    # change the output. This is the exact silent-corruption class the block
    # math warns about (indexing by timestep/batch alone "still runs and
    # still denoises — into mush") — a passing shape/NaN check alone would
    # not catch a gather that quietly no-ops to row 0 for every token. ──
    var flat_indices = List[Int]()
    for _ in range(S):
        flat_indices.append(0)
    var out_flat = minimax_h3_block_forward[S](
        x, weights, layer, config, mod, flat_indices, cos_t, sin_t,
        rope.rotary_dim, ctx,
    )
    var host_flat = out_flat.to_host(ctx)
    var max_diff = Float32(0.0)
    for i in range(len(host)):
        var d = host[i] - host_flat[i]
        var ad = d if d >= Float32(0.0) else -d
        if ad > max_diff:
            max_diff = ad
    print("per-row vs flat-index max abs diff:", max_diff)
    if max_diff == Float32(0.0):
        raise Error(
            "minimax_h3_block_forward probe: per-row adaLN gather had NO"
            " effect — the row addressing is silently collapsing"
        )
    print("PASS")
