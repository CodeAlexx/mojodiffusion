# ERNIE-Image real-weight FULL block0 smoke.
#
# Drives the unbounded layer-0 forward (`ernie_block0_full_forward`) with real
# loaded weights. Unlike `ernie_block0_smoke.mojo`, the AdaLN modulation passed
# into the block is NOT scaled by `0.00001`. The block forward itself contains
# zero bounding tricks — it matches the Rust `ErnieImageModel::block_forward`
# line for line (residual + adaLN_ln rms_norm + modulate + Q/K/V + QK-RMSNorm
# + 3-axis halfsplit RoPE + SDPA + out_proj + gated residual + adaLN_mlp_ln
# rms_norm + modulate + GELU-gated MLP + gated residual).
#
# Real AdaLN chain (post 2026-05-28 sin/cos bugfix)
# -------------------------------------------------
# Earlier revisions of this smoke synthesised the AdaLN modulation as
# `randn([1, 24576]) * 0.1` because the real timestep MLP -> shared
# adaLN_modulation chain overflowed BF16 (absmax ~215_000). The skeptic
# (`SKEPTIC_FINDINGS_ernie_block0_2026-05-28.md`, A2/A5) traced this to a
# sin/cos channel-order mismatch in `ops/embeddings.timestep_embedding`
# (cos-first, correct for Z-Image NextDiT) vs ERNIE training convention
# (sin-first, `ernie_image.rs:603`). ERNIE now calls the sibling
# `timestep_embedding_sin_first` and the synthetic workaround is removed.
# This smoke drives the REAL chain end-to-end:
#     model.time_embed(timestep)  ->  model.shared_adaln(temb)  ->  block0
# Timestep set to 500.0 (non-extreme, mid-schedule equivalent of sigma~0.5).
#
# Bounded comptime sizes keep the compile fast:
#   N_IMG = 16x16 = 256 image tokens   (production: 64x64 = 4096)
#   N_TXT = 64 text tokens             (production: 256)
#   S     = 320 total                  (production: 4352)
#   hidden / heads / head_dim are FULL ERNIE production values (4096 / 32 / 128)
# so attention/FFN matmuls hit the real 4096x4096 / 4096x12288 weight matrices.

from std.gpu.host import DeviceContext
from std.math import sqrt

from serenitymojo.io.dtype import STDtype
from serenitymojo.models.dit.ernie_contract import (
    ERNIE_DIT_HEAD_DIM,
    ERNIE_DIT_HEADS,
    ERNIE_DIT_HIDDEN,
    ERNIE_DIT_TEXT_IN_DIM,
    ERNIE_LATENT_CHANNELS,
    ERNIE_LATENT_H,
    ERNIE_LATENT_W,
    ERNIE_TEXT_MAX_TOKENS,
    validate_ernie_metadata_contract,
)
from serenitymojo.models.dit.ernie_image import (
    ErnieImageResident,
    build_ernie_rope_tables,
    validate_ernie_adaln_shape,
    validate_ernie_block0_shape,
    validate_ernie_resident_shapes,
)
from serenitymojo.ops.random import randn
from serenitymojo.ops.tensor_algebra import concat, mul_scalar, slice
from serenitymojo.runtime.model_manifest import ernie_image_default_manifest
from serenitymojo.tensor import Tensor


# Deterministic seeds — distinct from the bounded smoke so they cannot share
# accidental cache state.
comptime SEED_LATENT = UInt64(20260601)
comptime SEED_TEXT   = UInt64(20260602)

# Input scale for the synthetic latent + text features (BF16-safe).
comptime INPUT_SCALE = Float32(0.25)

# Mid-schedule timestep (sigma ~0.5 -> t = 500 on ERNIE's 0..1000 timestep
# range). Picked non-extreme so the sinusoidal angles stay moderate; this
# avoids stressing BF16 rounding in the timestep MLP. Skeptic predicts the
# resulting `adaln_raw` absmax will be well under 100 once the sin/cos
# channel order is correct (see `SKEPTIC_FINDINGS_ernie_block0_2026-05-28.md`).
comptime TIMESTEP_VALUE = Float32(500.0)

# Bounded sequence sizes for fast compile.
comptime IMG_H = 16
comptime IMG_W = 16
comptime N_IMG = IMG_H * IMG_W            # 256
comptime N_TXT = 64
comptime S     = N_IMG + N_TXT            # 320


def _require_shape2(label: String, got: List[Int], a: Int, b: Int) raises:
    if len(got) != 2 or got[0] != a or got[1] != b:
        raise Error(String("shape mismatch for ") + label)


def _require_shape3(label: String, got: List[Int], a: Int, b: Int, c: Int) raises:
    if len(got) != 3 or got[0] != a or got[1] != b or got[2] != c:
        raise Error(String("shape mismatch for ") + label)


def _stats(name: String, t: Tensor, ctx: DeviceContext, max_abs_allowed: Float64) raises:
    var h = t.to_host(ctx)
    if len(h) == 0:
        raise Error(String("empty tensor stats: ") + name)
    var s = 0.0
    var s2 = 0.0
    var amax = 0.0
    for i in range(len(h)):
        var v = Float64(h[i])
        if v != v:
            raise Error(String("NaN in ") + name)
        # +/-inf check — BF16 max finite is ~65504.
        if v > Float64(1.0e30) or v < Float64(-1.0e30):
            raise Error(String("Inf in ") + name)
        if v > max_abs_allowed or v < -max_abs_allowed:
            raise Error(String("unstable value in ") + name)
        s += v
        s2 += v * v
        var av = v if v >= 0.0 else -v
        if av > amax:
            amax = av
    if amax == 0.0:
        raise Error(String("all-zero tensor stats: ") + name)
    var mean = s / Float64(len(h))
    var var_ = s2 / Float64(len(h)) - mean * mean
    if var_ < 0.0:
        var_ = 0.0
    print(
        "  ",
        name,
        "stats mean/std/absmax:",
        Float32(mean),
        Float32(sqrt(var_)),
        Float32(amax),
    )


def main() raises:
    # Manifest + checkpoint contract — keeps this smoke gated on the same
    # invariants the rest of the ERNIE port enforces.
    var manifest = ernie_image_default_manifest()
    _ = validate_ernie_metadata_contract(manifest)

    var ctx = DeviceContext()
    print("=== ERNIE-Image block0 FULL real-weight smoke (no bounded scaling) ===")
    print("  bounded comptime slice image/text/total:", N_IMG, N_TXT, S)
    print("  hidden / heads / head_dim:", ERNIE_DIT_HIDDEN, ERNIE_DIT_HEADS, ERNIE_DIT_HEAD_DIM)

    print("  [load] resident + layer0 weights")
    var model = ErnieImageResident.load_default_block0_smoke(ctx)
    model.validate_block0_smoke_weights()
    print("  [load] done")

    # --- Build the unbounded sequence input -------------------------------
    # Use the real resident path for patch_embed + text_proj so the sequence
    # has realistic Q/K/V activations at production scale. We slice the result
    # down to the bounded smoke size.
    var latent_shape = List[Int]()
    latent_shape.append(1)
    latent_shape.append(ERNIE_LATENT_CHANNELS)
    latent_shape.append(ERNIE_LATENT_H)
    latent_shape.append(ERNIE_LATENT_W)
    var latent = mul_scalar(
        randn(latent_shape^, SEED_LATENT, STDtype.BF16, ctx), INPUT_SCALE, ctx
    )

    var text_shape = List[Int]()
    text_shape.append(1)
    text_shape.append(ERNIE_TEXT_MAX_TOKENS)
    text_shape.append(ERNIE_DIT_TEXT_IN_DIM)
    var text = mul_scalar(
        randn(text_shape^, SEED_TEXT, STDtype.BF16, ctx), INPUT_SCALE, ctx
    )

    # Non-extreme timestep (sigma~0.5 -> t=500). With the sin-first sinusoid
    # this drives the trained `time_embedding.linear_{1,2}` and the shared
    # `adaLN_modulation.1` into BF16-safe territory (absmax expected < 100).
    var t_vals = List[Float32]()
    t_vals.append(TIMESTEP_VALUE)
    var t_shape = List[Int]()
    t_shape.append(1)
    var timestep = Tensor.from_host(t_vals, t_shape^, STDtype.F32, ctx)

    print("  [resident] patch/text/time/adaLN")
    var patch_tokens = model.patch_embed_1024(latent, ctx)
    var temb = model.time_embed(timestep, ctx)
    var text_tokens = model.project_text(text, ctx)
    validate_ernie_resident_shapes(patch_tokens, temb, text_tokens)
    _stats(String("patch_tokens"), patch_tokens, ctx, 64.0)
    _stats(String("text_tokens"), text_tokens, ctx, 64.0)
    # temb / adaln are the bounds skeptic predicts post-fix: temb low tens,
    # adaln well under 100. Bounds kept loose pending first GPU run.
    _stats(String("temb_raw"), temb, ctx, 200.0)

    # --- Real shared AdaLN chain (replaces synthetic randn workaround) ----
    # `shared_adaln` runs `silu -> linear(adaLN_modulation.1)` on `temb`
    # to produce the [1, 6*hidden] modulation tensor consumed by block0.
    var adaln_raw = model.shared_adaln(temb, ctx)
    validate_ernie_adaln_shape(adaln_raw)
    # ASSERT (compile-verify only): if adaln_raw absmax exceeds 500 the
    # sin/cos fix did not take effect — _stats raises immediately. Pre-fix
    # this would have been ~215040 (>> BF16 max 65504). Post-fix Rust
    # parity expects absmax < 100; bound left at 500 as a safe upper.
    _stats(String("adaln_raw"), adaln_raw, ctx, 500.0)

    # --- Bounded slice (sequence length only — hidden stays full 4096) ---
    var img = slice(patch_tokens, 1, 0, N_IMG, ctx)
    var txt = slice(text_tokens, 1, 0, N_TXT, ctx)
    var seq = concat(1, ctx, img, txt)
    _require_shape3(String("seq"), seq.shape(), 1, S, ERNIE_DIT_HIDDEN)
    _stats(String("seq"), seq, ctx, 64.0)

    # --- Build real ERNIE RoPE tables (no tricks) -------------------------
    print("  [rope] build")
    var rope = build_ernie_rope_tables[N_IMG, N_TXT, ERNIE_DIT_HEADS, ERNIE_DIT_HEAD_DIM](
        IMG_H, IMG_W, N_TXT, ctx, STDtype.BF16
    )
    _require_shape2(String("rope_cos"), rope[0].shape(), S * ERNIE_DIT_HEADS, ERNIE_DIT_HEAD_DIM)
    _require_shape2(String("rope_sin"), rope[1].shape(), S * ERNIE_DIT_HEADS, ERNIE_DIT_HEAD_DIM)

    # --- Full block-0 forward (no bounding inside) ------------------------
    print("  [block0] full forward")
    var out = model.ernie_block0_full_forward[S](
        seq, adaln_raw, rope[0], rope[1], ctx
    )
    validate_ernie_block0_shape[S](out)
    # BF16 max-finite is ~65504. Production block output stays well under
    # that with sane AdaLN; bound to 60000 so we catch overflow regressions.
    _stats(String("block0_out"), out, ctx, 60000.0)
    print("ERNIE-Image block0 FULL real-weight smoke PASS")
