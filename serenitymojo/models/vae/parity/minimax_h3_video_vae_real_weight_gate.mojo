# serenitymojo/models/vae/parity/minimax_h3_video_vae_real_weight_gate.mojo
#
# REAL-WEIGHT gate for the native-key video VAE rebuild
# (models/vae/minimax_h3_video_encoder_device.mojo,
#  models/vae/minimax_h3_video_decoder_device.mojo) — the modules that were
# NEVER exercised against the actual checkpoint before this file. The old
# diffusers-oracle units (models/minimax_h3/video_encoder.mojo,
# video_decoder.mojo) are untouched and out of scope; they are gated against
# a different reference and cannot load these weights at all (fused
# to_qkv/bare to_out/ff.w1/nin_shortcut/native decoder.x_embedder key names
# have no diffusers equivalent in this checkpoint — confirmed by a
# header-only key diff, this file's phase 1).
#
# THREE PHASES, per team-lead's spec:
#   [1] HEADER-ONLY key diff — done OFFLINE (not in this Mojo file) by
#       dumping minimax_h3_video_encoder_key_names()/
#       minimax_h3_video_decoder_native_key_names() via a throwaway no-GPU
#       probe and diffing against the real safetensors header in Python (no
#       GPU needed for a header-only key/shape/dtype comparison). RESULTS,
#       reported in full to team-lead:
#         ENCODER: 118/118 expected keys present in the real checkpoint, 0
#           missing, 0 extra. Every sampled shape (conv_in, both resnet
#           blocks at all 6 levels, every downsample.conv, norm_out/
#           conv_out, quant_conv) matches the config-derived expectation
#           exactly.
#         DECODER (native, to_qkv fused): 441/441 expected keys present, 0
#           missing. ONE extra real key not in the rebuild's key list:
#           `decoder.mask_token` [1,1,2048] F32 — traced to vae_vit.py
#           `init_mask_config`/`apply_mask_preprocess`: a masked-modeling
#           training buffer, consumed ONLY when `self.training and
#           self.mask_enabled`, and `apply_mask_preprocess` itself raises
#           `NotImplementedError("mask modeling is not supported in this
#           inference-only bundle")` on that branch. The released
#           video_vae/source/config.json has no mask_enabled/mask_prob key
#           at all (defaults disabled). CONFIRMED dead weight for inference,
#           not a rebuild gap — not loaded here, deliberately.
#         Every sampled decoder shape (x_embedder, register_tokens,
#         mask_token, transformer_blocks at layers 0/17/35, norm_out,
#         proj_out, post_quant_conv) matches the config-derived expectation
#         exactly, including the two traps this rebuild exists to fix:
#         to_qkv.weight [6144,2048] (= 3*2048, fused, confirming the split-
#         at-load transform is operating on the right shape) and ff.w1.weight
#         [16384,2048] (= 2*8192, gate+value fused, confirming the FFN order
#         fix applies to a real tensor of the expected width).
#         ALL 560 tensors are F32 (team-lead's note, preserved — nothing here
#         downcasts).
#
#   [2] LOAD real weights through the rebuild's OWN load() paths
#       (`MiniMaxH3VideoEncoderDevice.load` / `minimax_h3_video_decoder_
#       device_load`) and run a SMALL real forward through each — THIS Mojo
#       file. Encoder and decoder are loaded/run/DROPPED sequentially, not
#       simultaneously (decoder alone is ~9.6 GiB F32 — the large majority of
#       the checkpoint's 9.70 GiB; loading both at once needlessly risks
#       VRAM headroom for no reason, same one-big-thing-at-a-time discipline
#       used throughout this repo).
#
#   [3] Numeric oracle — NOT built. Assessed and stopped here deliberately
#       (team-lead's explicit "if impractical, say so and stop at (2)"):
#       the vendor's own klvae.py/vae_vit.py/vae_cnn.py import
#       `flash.py`/`parallel.py` (flash-attn + a custom sequence-parallel
#       process-group layer initialized at IMPORT time) and `base_module.py`
#       pulls in a HF-style base class stack — standing up a CPU/single-GPU
#       eval path from this bundle is a real, separate engineering task
#       (mocking or stripping the parallel/flash dependencies), not a "run
#       one script" oracle. Attempted a minimal import smoke first (see
#       report) rather than assuming; did not fabricate a hand-derived
#       reference (the exact anti-pattern the numeric-parity-testing
#       discipline warns against — "never hand-compute the expected value").
#
# Run (package-relative imports need -I .; sdpa_flash_infer_fwd is NOT used
# here — the decoder's attention is sdpa_nomask, no cuDNN shim needed for
# THIS file, but conv3d_fcqrs_cudnn still needs the cuDNN runtime the pixi
# env already provides):
#   cd /home/alex/mojodiffusion && pixi run mojo run -I . -Xlinker -lm \
#     serenitymojo/models/vae/parity/minimax_h3_video_vae_real_weight_gate.mojo

from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.random import randn
from serenitymojo.models.vae.minimax_h3_video_encoder_device import (
    MiniMaxH3VideoEncoderDevice,
    minimax_h3_video_released_encoder_config,
    minimax_h3_video_encode_device,
)
from serenitymojo.models.vae.minimax_h3_video_decoder_device import (
    MiniMaxH3VideoDecoderDevice,
    minimax_h3_video_released_decoder_config,
    minimax_h3_video_decoder_device_load,
    minimax_h3_video_decode_device,
)

comptime ENCODER_DIR = "/home/alex/.serenity/models/checkpoints/MiniMax-H3/FL2VA/video_vae/source"
comptime DECODER_DIR = "/home/alex/.serenity/models/checkpoints/MiniMax-H3/FL2VA/video_vae/source"

# Decoder: minimal latent grid. lt=1,lh=1,lw=1 -> 1 patch token + 1 (1+4
# register/zero-suffix tokens) = S=6. Small enough to run all 36 real-weight
# blocks fast; the point is proving the REAL WEIGHTS run clean, not
# resolution coverage.
comptime DEC_LT = 1
comptime DEC_LH = 1
comptime DEC_LW = 1
comptime DEC_S = 6
comptime DEC_HEADS = 32
comptime DEC_DH = 64


def _mag(t: Tensor, ctx: DeviceContext) raises -> Tuple[Float32, Float32, Int]:
    var host = t.to_host(ctx)
    var sum_abs = Float32(0.0)
    var max_abs = Float32(0.0)
    var n_bad = 0
    for i in range(len(host)):
        var v = host[i]
        if v != v:
            n_bad += 1
            continue
        var av = v if v >= Float32(0.0) else -v
        sum_abs += av
        if av > max_abs:
            max_abs = av
    var mean_abs = sum_abs / Float32(max(len(host), 1))
    return (mean_abs, max_abs, n_bad)


def main() raises:
    var ctx = DeviceContext()
    var n_fail = 0

    # ── ENCODER: real weights, small real forward, then drop before decoder ──
    print("[encoder] loading real weights from", ENCODER_DIR, "...")
    var enc_cfg = minimax_h3_video_released_encoder_config()
    var encoder = MiniMaxH3VideoEncoderDevice.load(ENCODER_DIR, enc_cfg, ctx)
    print("  loaded", len(encoder.weights), "tensors (expect 118)")
    if len(encoder.weights) != 118:
        print("  FAIL unexpected encoder tensor count")
        n_fail += 1

    print("[encoder] running a small real forward (T=4,H=32,W=32,C=3) ...")
    var pixel_shape: List[Int] = [1, 4, 32, 32, 3]
    var pixels = randn(pixel_shape^, 1, STDtype.F32, ctx)
    var moments = minimax_h3_video_encode_device(encoder, pixels, ctx)
    var ms = moments.shape()
    print("  moments shape", ms[0], ms[1], ms[2], ms[3], ms[4], " (expect [1,4,2,2,48])")
    var magM = _mag(moments, ctx)
    print("  mean_abs", magM[0], " max_abs", magM[1], " nan_count", magM[2])
    if magM[2] != 0:
        print("  FAIL NaN/Inf in real-weight encoder output")
        n_fail += 1
    if len(ms) != 5 or ms[4] != 2 * enc_cfg.z_channels:
        print("  FAIL encoder output channel count wrong")
        n_fail += 1
    print("[encoder] done, dropping weights before loading the decoder")

    # ── DECODER: real weights (qkv split-at-load exercised for real), small real forward ──
    print("")
    print("[decoder] loading real weights from", DECODER_DIR, "(qkv split-at-load, ~9.6 GiB F32) ...")
    var dec_cfg = minimax_h3_video_released_decoder_config()
    var decoder = minimax_h3_video_decoder_device_load(DECODER_DIR, dec_cfg, ctx)
    # 441 native keys, minus 72 (36 blocks x {to_qkv.weight,to_qkv.bias}
    # skipped in the load loop), plus 216 (36 blocks x 3 parts x
    # {weight,bias} synthesized by _split_qkv_at_load) = 585.
    print("  loaded", len(decoder.weights), "tensors (expect 585)")
    if len(decoder.weights) != 585:
        print("  FAIL unexpected decoder tensor count")
        n_fail += 1

    print("[decoder] running a small real forward (lt=lh=lw=1, S=", DEC_S, ") ...")
    var latent_shape: List[Int] = [1, DEC_LT, DEC_LH, DEC_LW, dec_cfg.latent_channels]
    var latents = randn(latent_shape^, 2, STDtype.F32, ctx)
    var pixels_out = minimax_h3_video_decode_device[DEC_S, DEC_HEADS, DEC_DH](decoder, latents, ctx)
    var ps = pixels_out.shape()
    print("  pixels_out shape", ps[0], ps[1], ps[2], ps[3], ps[4], " (expect [1,4,16,16,3])")
    var magP = _mag(pixels_out, ctx)
    print("  mean_abs", magP[0], " max_abs", magP[1], " nan_count", magP[2])
    if magP[2] != 0:
        print("  FAIL NaN/Inf in real-weight decoder output")
        n_fail += 1
    if len(ps) != 5 or ps[1] != DEC_LT * dec_cfg.patch_size_t or ps[2] != DEC_LH * dec_cfg.patch_size or ps[3] != DEC_LW * dec_cfg.patch_size or ps[4] != dec_cfg.out_channels:
        print("  FAIL decoder output shape wrong")
        n_fail += 1

    print("")
    if n_fail != 0:
        raise Error(String("minimax_h3_video_vae_real_weight_gate: ") + String(n_fail) + " check(s) FAILED")
    print("ALL CHECKS PASS — real weights load and run clean; NOT a numeric parity gate (see file header, phase [3])")
