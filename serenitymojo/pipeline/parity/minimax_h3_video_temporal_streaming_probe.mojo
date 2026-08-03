# serenitymojo/pipeline/parity/minimax_h3_video_temporal_streaming_probe.mojo
#
# STREAMING temporal decode vs the FOLDED one — must be BIT-IDENTICAL.
#
# WHY THIS BAR, AND WHY NO ORACLE. The streaming path is not an approximation
# of the folded path and is not a different algorithm: every emitted part is
# produced by the SAME sequence of slice / minimax_h3_video_tiled_decode /
# minimax_h3_video_blend calls, in the SAME order, on the SAME inputs. Only the
# accumulation differs — the folded path collects parts and concatenates, the
# streaming path hands each part out. So concatenating the streamed parts must
# reproduce the folded tensor EXACTLY, and anything short of max_abs == 0 means
# the loop diverged (a mis-sliced segment, a dropped or double-applied blend, a
# stale carried overlap), not that floating point drifted.
#
# That is a STRONGER and CHEAPER check than re-running the vendor oracle: the
# folded path is already gated against the vendor's own decode_temporal by
# minimax_h3_video_temporal_oracle_parity.mojo, so bit-equality here transfers
# that gate to the streaming path for free. The two gates compose:
#   vendor decode_temporal  ==(cos>=0.999, per-frame)==  folded Mojo
#   folded Mojo             ==(max_abs == 0)==           streamed Mojo
#
# WHAT IT EXERCISES: latent T=27 at 30x52 => 5 temporal chunks (so 4 carried
# overlaps, i.e. the dec_overlap hand-off happens four times and the LAST one
# is emitted by the trailing part rather than by a chunk) and 3x4 spatial tiles
# per clip. A streaming loop that mishandles the final carried overlap — the
# single most likely defect, since the folded path appends it after the loop
# (:332-333) while the streaming path emits it as an extra part — produces a
# tensor that is the right shape and wrong at the tail, which max_abs catches
# and a shape check alone would not.
#
# It also asserts the host-side frame plan directly: output_frames must equal
# both tensors' frame count, and `finish()` must accept the counters.
#
# Run (cuDNN shim on the link line for the F32 conv3d symbols):
#   cd /home/alex/mojodiffusion && pixi run mojo run -I . -Xlinker -lm \
#     -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa \
#     serenitymojo/pipeline/parity/minimax_h3_video_temporal_streaming_probe.mojo

from std.collections import List
from std.gpu.host import DeviceContext
from std.math import sqrt

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.random import randn
from serenitymojo.models.vae.minimax_h3_video_decoder_device import (
    minimax_h3_video_released_decoder_config,
    minimax_h3_video_decoder_device_load,
)
from serenitymojo.pipeline.minimax_h3_video_vae_spatial_tiling import (
    minimax_h3_video_released_tiling_config,
)
from serenitymojo.pipeline.minimax_h3_video_vae_temporal import (
    minimax_h3_video_decode_temporal,
    minimax_h3_video_decode_temporal_streamed,
    minimax_h3_video_released_temporal_config,
    minimax_h3_temporal_output_frame_plan,
    MiniMaxH3TemporalDecodeStream,
)

comptime DEC_DIR = "/home/alex/.serenity/models/checkpoints/MiniMax-H3/FL2VA/video_vae/source"
comptime TOKENS_PER_CLIP = 7
comptime LATENT_TILE = 16
comptime LAT_T = 27


def main() raises:
    var ctx = DeviceContext()
    var tcfg = minimax_h3_video_released_temporal_config()
    var tiling = minimax_h3_video_released_tiling_config()

    # ── Host-side plan check first: no weights, no decode, fails in a second
    # if the arithmetic is wrong rather than after a 5-chunk GPU decode.
    var pseudo = LAT_T + tcfg.token_drop
    var tcs = tcfg.tokens_chunk_size()
    var pad_tokens = 0
    if pseudo % tcs != 0:
        pad_tokens = tcs - (pseudo % tcs)
        pseudo += pad_tokens
    var num_chunks = pseudo // tcs - (1 if tcfg.token_drop > 0 else 0)
    var plan = minimax_h3_temporal_output_frame_plan(
        tcfg, LAT_T + pad_tokens, num_chunks, pad_tokens
    )
    print(
        "[stream] plan: num_chunks", num_chunks, " total_frames", plan.total_frames,
        " pad_frames", plan.pad_frames, " output_frames", plan.output_frames,
    )
    # latent_T=27 => n=5 => 17*5 + 5 = 90 frames.
    if plan.output_frames != 90:
        raise Error("streaming probe: expected 90 output frames for latent T=27")

    print("[stream] loading real decoder weights (9.6 GiB F32) ...")
    var cfg = minimax_h3_video_released_decoder_config()
    var decoder = minimax_h3_video_decoder_device_load(String(DEC_DIR), cfg, ctx)

    var z = randn([1, LAT_T, 30, 52, 24], 9137, STDtype.F32, ctx)

    print("[stream] folded decode ...")
    var folded = minimax_h3_video_decode_temporal[
        LATENT_TILE, LATENT_TILE, 32, 64, 5, TOKENS_PER_CLIP
    ](decoder, z, tcfg, tiling, ctx)
    print("  folded:", folded.shape())

    print("[stream] streamed decode ...")
    var streamed = minimax_h3_video_decode_temporal_streamed[
        LATENT_TILE, LATENT_TILE, 32, 64, 5, TOKENS_PER_CLIP
    ](decoder, z, tcfg, tiling, ctx)
    print("  streamed:", streamed.shape())

    var fs = folded.shape()
    var ss = streamed.shape()
    if len(fs) != len(ss):
        raise Error("streaming probe: rank mismatch")
    for i in range(len(fs)):
        if fs[i] != ss[i]:
            raise Error(
                String("streaming probe: shape mismatch at axis ") + String(i)
                + " folded " + String(fs[i]) + " vs streamed " + String(ss[i])
            )
    if fs[1] != plan.output_frames:
        raise Error("streaming probe: decoded frame count disagrees with the host plan")

    var a = folded.to_host(ctx)
    var b = streamed.to_host(ctx)
    if len(a) != len(b):
        raise Error("streaming probe: element-count mismatch")
    var max_abs = Float64(0)
    var worst_i = -1
    for i in range(len(a)):
        var d = Float64(a[i]) - Float64(b[i])
        if d < 0:
            d = -d
        if d > max_abs:
            max_abs = d
            worst_i = i
    var per_frame = fs[2] * fs[3] * fs[4]
    print("  max_abs diff:", max_abs, " at element", worst_i,
          " (frame", worst_i // per_frame, ")")

    # ── Also drive the stream by hand to prove the part sequence itself, not
    # just the concatenated result: part count, per-part frame counts, and the
    # running offset a PNG-writing caller depends on.
    var st = MiniMaxH3TemporalDecodeStream(z, tcfg, tiling, ctx)
    var parts = 0
    var running = 0
    while st.has_next():
        var p = st.next_part[
            LATENT_TILE, LATENT_TILE, 32, 64, 5, TOKENS_PER_CLIP
        ](decoder, ctx)
        if p:
            var pf = p.value().shape()[1]
            if st.frames_emitted() != running + pf:
                raise Error("streaming probe: frames_emitted() offset drifted")
            running += pf
            parts += 1
            print("    part", parts, ":", pf, "frames, running offset now", running)
    st.finish()
    print("  parts:", parts, " frames:", running, " output_frames:", st.output_frames())
    if running != plan.output_frames:
        raise Error("streaming probe: hand-driven stream emitted the wrong frame count")

    if max_abs != Float64(0):
        raise Error(
            String("minimax_h3_video_temporal_streaming_probe FAILED — streamed and")
            + " folded decodes differ (max_abs " + String(max_abs) + ", expected exactly 0)"
        )
    print("minimax_h3_video_temporal_streaming_probe PASS (bit-identical)")
