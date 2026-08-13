# Does a REAL LTX-2.5 transformer block load through the ff_bias=False change?
#
# LTX-2.5 sets `ff_bias=False`: measured against the shipped checkpoints, it
# drops exactly `transformer_blocks.N.ff.net.0.proj.bias` and
# `transformer_blocks.N.ff.net.2.bias` (48 blocks x 2 = 96 tensors) versus 2.3,
# and changes nothing else — 4348 tensors in common with ZERO shape differences,
# plus one new `keyframes_abs_pos_embedding`. `ltx2_dit.mojo` was changed to skip
# those two keys when absent and run the video FF bias-free.
#
# This probe exercises that on the real 42 GB bf16 checkpoint. Loading a 2.3
# block through the same path would NOT test it (2.3 still has the biases), and
# a compile proves nothing about weight lookup.
#
# Build (cuDNN shim required):
#   pixi run mojo build -O0 -j 1 -I . -I vendor/mojo-libs \
#     -Xlinker -lm -Xlinker -lcuda \
#     -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa \
#     -Xlinker -Lserenitymojo/ops/cshim/lib/cudnn_stubs -Xlinker -lcudnn \
#     -Xlinker -rpath -Xlinker '$ORIGIN/../../serenitymojo/ops/cshim/lib' \
#     -Xlinker -rpath -Xlinker '$ORIGIN/../../.pixi/envs/default/lib' \
#     serenitymojo/models/dit/parity/ltx25_block_load_probe.mojo \
#     -o output/checks/ltx25_block_load

from max.gpu.host import DeviceContext

from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.models.dit.ltx2_dit import LTX2AVBlockWeights, LTX2Config


comptime CKPT = String(
    "/home/alex/.serenity/models/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors"
)
comptime PFX = String("model.diffusion_model.transformer_blocks.")


def main() raises:
    var ctx = DeviceContext()
    print("== LTX-2.5 real block load (ff_bias=False path) ==")

    # First prove the premise directly on the file: the two video-FF biases are
    # genuinely absent while the weights and the audio-FF biases are present.
    var st = ShardedSafeTensors.open(CKPT)
    for b in [0, 5, 47]:
        var p = PFX + String(b) + "."
        print(
            "  block", b,
            " ff.net.0.proj.weight=", st.has_tensor(p + "ff.net.0.proj.weight"),
            " ff.net.0.proj.bias=", st.has_tensor(p + "ff.net.0.proj.bias"),
            " ff.net.2.bias=", st.has_tensor(p + "ff.net.2.bias"),
            " audio_ff.net.2.bias=", st.has_tensor(p + "audio_ff.net.2.bias"),
        )
    print(
        "  keyframes_abs_pos_embedding present:",
        st.has_tensor(String("model.diffusion_model.keyframes_abs_pos_embedding")),
    )

    # Then actually load blocks through the production path.
    var cfg = LTX2Config.ltx2()
    for b in [0, 5, 47]:
        var w = LTX2AVBlockWeights.load(CKPT, b, cfg, ctx)
        print("  loaded block", b, "OK")
        _ = w^
    print("LTX-2.5 BLOCKS LOAD")
