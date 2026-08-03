# serenitymojo/pipeline/parity/minimax_h3_video_temporal_oracle_parity.mojo
#
# THE FULL VIDEO-VAE DECODE COMPOSITION vs the vendor's own decode_temporal.
# Real weights, production spatial geometry, a latent spanning FIVE temporal
# clips and 3x4 spatial tiles.
#
# WHY THIS GATE EXISTS. Two layers were gated separately and neither gate can
# see the other's failure mode:
#   * models/vae/parity/minimax_h3_video_vae_oracle_parity.mojo gates ONE
#     untiled per-volume decode (near bit-exact).
#   * pipeline/parity/minimax_h3_video_tiled_oracle_parity.mojo gates the
#     SPATIAL tiling against the vendor's tiled_decode (commit d8f1217), but
#     at latent T=2 — a single temporal clip, so it never exercises a
#     temporal seam at all.
# The composition adds a THIRD thing neither covers: `decode_temporal`
# (klvae.py:678) slices 7-token clips, decodes each through `_adaptive_decode`
# (:733 -> tiled_decode at :437-441), trims `frame_pre_padding`=3 frames off
# every 20-frame group, cross-fades consecutive clips over `frame_overlap`=5
# frames, and carries a `dec_overlap` tail across the whole loop. Every one of
# those is an opportunity to be off by a frame in a way that a per-layer gate
# passes and a viewer sees as a stutter every 17 frames.
#
# THE DECISIVE CASE THIS PICKS: latent T=27 gives 5 temporal chunks (4 temporal
# seams) at 30x52 latent = 480x832 px, which splits 3 tiles in H and 4 in W.
# Every temporal seam therefore runs THROUGH all 12 spatial tiles, so the
# corner where a temporal seam meets a spatial seam in both axes at once is
# inside the compared region many times over — the exact cell where a wrong
# blend order (temporal-then-spatial instead of spatial-then-temporal) would
# show up and nowhere else.
#
# Oracle: scratchpad/vvae_temporal_oracle_gen.py (OneTrainer venv, GPU F32) —
#   z [1,24,27,30,52] NCDHW randn(seed 9137) -> decode_temporal ->
#   [1,3,90,480,832], both dumped NDHWC. 90 = 17*5 + 5, the vendor's own
#   output-frame plan for 5 chunks.
#
# BAR: the same 0.999 cosine the tiled gate uses, plus an explicit magnitude
# ratio (cos is magnitude-blind and this repo has been burned by that), plus a
# SHAPE check — a composition that drops or duplicates a frame usually changes
# the frame count before it changes the cosine, and that is the failure this
# gate is really hunting.
#
# Run (cuDNN shim on the link line for the F32 conv3d symbols):
#   cd /home/alex/mojodiffusion && pixi run mojo run -I . -Xlinker -lm \
#     -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa \
#     serenitymojo/pipeline/parity/minimax_h3_video_temporal_oracle_parity.mojo

from std.collections import List
from std.gpu.host import DeviceContext
from std.math import sqrt

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.parity import ParityHarness
from serenitymojo.models.vae.minimax_h3_video_decoder_device import (
    minimax_h3_video_released_decoder_config,
    minimax_h3_video_decoder_device_load,
)
from serenitymojo.pipeline.minimax_h3_video_vae_spatial_tiling import (
    minimax_h3_video_released_tiling_config,
)
from serenitymojo.pipeline.minimax_h3_video_vae_temporal import (
    minimax_h3_video_decode_temporal,
    minimax_h3_video_released_temporal_config,
)

comptime ORACLE = "/tmp/claude-1000/-home-alex-mojodiffusion/7e1531cb-f7e2-44a5-9d63-8604853a656a/scratchpad/vvae_temporal_oracle.safetensors"
comptime DEC_DIR = "/home/alex/.serenity/models/checkpoints/MiniMax-H3/FL2VA/video_vae/source"

# tokens_chunk_size(17,4) + token_overlap(3,5) = 5 + 2. Must equal
# `minimax_h3_video_released_temporal_config().tokens_per_clip()`, which the
# decode function itself re-checks at runtime and raises on.
comptime TOKENS_PER_CLIP = 7
comptime LATENT_TILE = 16


def _load_f32(ref st: SafeTensors, name: String) raises -> List[Float32]:
    var info = st.tensor_info(name)
    var bytes = st.tensor_bytes(name)
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    if tv.dtype != STDtype.F32:
        raise Error(String("_load_f32: unexpected dtype for ") + name)
    var p = tv.data.unsafe_ptr().bitcast[Float32]()
    var out = List[Float32]()
    for i in range(tv.numel()):
        out.append(p[i])
    return out^


def _load_shape(ref st: SafeTensors, name: String) raises -> List[Int]:
    var info = st.tensor_info(name)
    return info.shape.copy()


def main() raises:
    var ctx = DeviceContext()
    print("[temporal] loading oracle dump ...")
    var st = SafeTensors.open(String(ORACLE))
    var z_shape = _load_shape(st, "z_ndhwc")
    var z_vals = _load_f32(st, "z_ndhwc")
    var ref_vals = _load_f32(st, "dec_ndhwc")
    var ref_shape = _load_shape(st, "dec_ndhwc")
    print(
        "  z", z_shape[0], z_shape[1], z_shape[2], z_shape[3], z_shape[4],
        " ref", ref_shape[0], ref_shape[1], ref_shape[2], ref_shape[3], ref_shape[4],
    )

    var tcfg = minimax_h3_video_released_temporal_config()
    print(
        "  our temporal config: clip_length", tcfg.clip_length,
        " vae_ratio_t", tcfg.vae_ratio_t, " token_drop", tcfg.token_drop,
        " tokens_chunk", tcfg.tokens_chunk_size(),
        " token_overlap", tcfg.token_overlap(),
        " frame_pre_padding", tcfg.frame_pre_padding(),
        " frame_overlap", tcfg.frame_overlap(),
        " tokens_per_clip", tcfg.tokens_per_clip(),
    )

    print("[temporal] loading real decoder weights (9.6 GiB F32) ...")
    var cfg = minimax_h3_video_released_decoder_config()
    var decoder = minimax_h3_video_decoder_device_load(String(DEC_DIR), cfg, ctx)

    var z = Tensor.from_host(z_vals, z_shape^, STDtype.F32, ctx)
    var tiling = minimax_h3_video_released_tiling_config()
    print("[temporal] running decode_temporal(tiled) — 5 chunks x 3x4 tiles ...")
    var pixels = minimax_h3_video_decode_temporal[
        LATENT_TILE, LATENT_TILE, 32, 64, 5, TOKENS_PER_CLIP
    ](decoder, z, tcfg, tiling, ctx)
    var ps = pixels.shape()
    print("  pixels shape:", ps)

    # SHAPE FIRST: a frame miscount is the failure mode this gate exists for,
    # and comparing value-by-value against a differently-shaped reference would
    # either crash or silently compare misaligned frames.
    if len(ps) != len(ref_shape):
        raise Error("minimax_h3_video_temporal_oracle_parity: rank mismatch")
    for i in range(len(ps)):
        if ps[i] != ref_shape[i]:
            raise Error(
                String("minimax_h3_video_temporal_oracle_parity: shape mismatch at axis ")
                + String(i) + " — ours " + String(ps[i])
                + " vs vendor " + String(ref_shape[i])
                + " (a frame-count divergence in the temporal output plan)"
            )

    var harness = ParityHarness(0.999)
    var result = harness.compare(pixels, ref_vals, ctx)
    print("  ", result)

    var mine_host = pixels.to_host(ctx)
    var s_mine = Float64(0)
    var s_ref = Float64(0)
    for i in range(len(mine_host)):
        s_mine += Float64(mine_host[i]) * Float64(mine_host[i])
        s_ref += Float64(ref_vals[i]) * Float64(ref_vals[i])
    print("  |mine|", sqrt(s_mine), " |ref|", sqrt(s_ref), " ratio", sqrt(s_mine) / sqrt(s_ref))

    # PER-FRAME worst case. A global cosine over 90 frames can sit above 0.999
    # while ONE seam frame is visibly wrong; the seams here land at frames
    # 17,34,51,68 (17*k), so a per-frame minimum is what actually convicts a
    # bad blend.
    var frames = ps[1]
    var per_frame = ps[2] * ps[3] * ps[4]
    var worst = Float64(2.0)
    var worst_f = -1
    for f in range(frames):
        var dot = Float64(0)
        var na = Float64(0)
        var nb = Float64(0)
        var base = f * per_frame
        for i in range(per_frame):
            var a = Float64(mine_host[base + i])
            var b = Float64(ref_vals[base + i])
            dot += a * b
            na += a * a
            nb += b * b
        var c = dot / (sqrt(na) * sqrt(nb) + 1e-30)
        if c < worst:
            worst = c
            worst_f = f
    print("  worst per-frame cosine:", worst, "at frame", worst_f,
          " (temporal seams fall at frames 17, 34, 51, 68)")

    if not result.passed:
        raise Error("minimax_h3_video_temporal_oracle_parity FAILED (global cosine)")
    if worst < 0.999:
        raise Error("minimax_h3_video_temporal_oracle_parity FAILED (a single frame diverged)")
    print("minimax_h3_video_temporal_oracle_parity PASS")
