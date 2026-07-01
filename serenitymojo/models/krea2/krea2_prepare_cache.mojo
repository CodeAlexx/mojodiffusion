# krea2_prepare_cache.mojo — stage B of the Krea-2 LoRA-training cache prepare.
# (Stage A = serenitymojo/training/krea2_stage_images.py: image decode + bucketing
# + raw captions.)
#
# The staged images (images.safetensors) + raw captions (prompt.<i>.txt) go through
# the GATED Mojo encoders (QwenImageVaeEncoder, Qwen3Tokenizer + encode_krea2_stack)
# into the indexed safetensors training cache the KreaCacheReader streams:
#
#   clean.<i>     [1, 16, LH, LW]      BF16  normalized VAE latent (ai-toolkit boundary)
#   context.<i>   [1, LT, 12, 2560]    BF16  Qwen3-VL-4B 12-layer stack (== krea2_forward `context`)
#   text_len.<i>  [1]                  F32   LT (caption natural token count - DROP_IDX)
#   (optional)    context_uncond [1,LTu,12,2560] BF16 + text_len_uncond [1] F32
#
# WHY THESE EXACT TENSORS (read from the consumers):
#   * The DiT forward (models/dit/krea2_dit.mojo krea2_forward:1304) consumes `img`
#     [1,imglen,64] (PATCHIFIED latent) and `context` [1,LT,12,2560] and `pos`
#     [1,LFULL,3]. ai-toolkit keeps `latents` UNPACKED (B,C,h,w) all the way through
#     training — patchify is internal to predict_velocity/model.forward (pipeline.py
#     :85,102-117). So the cache stores the UNPACKED normalized latent `clean`
#     [1,16,LH,LW] (== ai-toolkit batch.latents) and the reader patchifies on demand
#     (after flow-noising), and builds `pos` deterministically from LH/LW + LT. This
#     lets the trainer add noise in latent space (noisy=(1-t)*clean+t*noise; target
#     = noise-clean, krea2.py:403) BEFORE the patchify, exactly like ai-toolkit.
#   * `context` is the encode_krea2_stack output [1,LT,12,2560] — un-flattened, which
#     is exactly what krea2_forward wants (predict_velocity reshapes its flattened
#     (B,Lt,n*d) back to (B,Lt,n,d); we never flatten, so no reshape needed).
#
# LATENT NORMALIZATION (the ai-toolkit data semantics, krea2.py encode_images:430-443):
#     z = vae.encode(img).latent_dist.SAMPLE()           # ai-toolkit SAMPLES the dist
#     latents_mean = cfg.latents_mean.view(1,16,1,1,1)
#     latents_std  = 1.0 / cfg.latents_std.view(1,16,1,1,1)
#     latents = (z - latents_mean) * latents_std         # == (z - mean) / std
#   We store this NORMALIZED latent. The Mojo QwenImageVaeEncoder.encode_mean returns
#   the deterministic posterior MEAN ([1,16,LH,LW], gated by qwenimage_encoder_parity
#   cos+max_abs), and we apply the SAME per-channel (z-mean)/std using the decoder's
#   _vae_mean()/_vae_std() (single source of truth — the decoder's denorm z*std+mean
#   is the exact inverse). DIVERGENCE (surfaced, not hidden): ai-toolkit uses
#   .latent_dist.SAMPLE() (a random Gaussian draw per epoch); we use the deterministic
#   MEAN (.mode()) for a REPRODUCIBLE cache. This is the standard cache choice (OT /
#   the ideogram4 cache both cache the deterministic mean); it removes the per-epoch
#   VAE-sampling noise, which is a deliberate, conventional simplification.
#
# CAPTION (== ai-toolkit text_encoder.py encode_krea_prompt + the Mojo inference path):
#   ids = tokenize(KREA2_TPL_PREFIX + <raw caption> + KREA2_TPL_SUFFIX)  (krea2_paths)
#   context = encode_krea2_stack(enc, ids)   # 12-layer stack, DROP_IDX=34 prefix-drop
#   LT = len(stack) = len(ids) - 34. Mirrors krea2_encode_cli exactly. NO ideogram4-
#   style JSON/chat-template digest — krea2's template IS the prefix/suffix.
#
# VRAM: the Qwen3-VL-4B TE load is heavy (~9.6 GB streamed; see krea2_qwen3vl_4b.mojo)
#   and the VAE encoder is small. The encoders are function-local and the process is
#   one-shot. If a 24 GB card is busy elsewhere, run this alone (the orchestrator runs
#   it; the encode is GPU-heavy so it is NOT backgrounded by an agent).
#
# Run (after stage A; GPU free). NOTE: -lm is required (the VAE encoder uses libm
# trig); build then run (mojo run also accepts -Xlinker -lm):
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#   pixi run mojo build -I . -Xlinker -lm \
#     -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa \
#     serenitymojo/models/krea2/krea2_prepare_cache.mojo -o /tmp/krea2_prepare && \
#   LD_LIBRARY_PATH=/home/alex/mojodiffusion/.pixi/envs/default/lib \
#     /tmp/krea2_prepare <stage_dir> <out_cache.safetensors> <n> <SIZE>
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.gpu.host import DeviceContext
from std.memory import ArcPointer
from std.sys import argv

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.tensor_algebra import sub, div
from serenitymojo.registry.checkpoints import path_exists
from serenitymojo.tokenizer.tokenizer import Qwen3Tokenizer
from serenitymojo.models.vae.qwenimage_encoder import QwenImageVaeEncoder
from serenitymojo.models.vae.qwenimage_decoder import _vae_mean, _vae_std
from serenitymojo.models.text_encoder.krea2_qwen3vl_4b import (
    load_krea2_qwen3vl_4b,
    encode_krea2_stack,
)
from serenitymojo.pipeline.krea2_paths import (
    KREA2_TE_DIR,
    KREA2_TOK_JSON,
    KREA2_VAE_DIR,
    KREA2_TPL_PREFIX,
    KREA2_TPL_SUFFIX,
)

comptime TArc = ArcPointer[Tensor]


def _read_text(path: String) raises -> String:
    var f = open(path, "r")
    var s = f.read()
    f.close()
    return s^


def _mean_ch(ctx: DeviceContext) raises -> Tensor:
    """latents_mean as [1,16,1,1] F32 (broadcasts over the [1,16,LH,LW] channel axis:
    NumPy right-aligned broadcast, axes 2,3 are 1 -> stride 0, axis 1 == 16). Value =
    the decoder's _vae_mean() (single source of truth; the decoder denorm z*std+mean
    is the exact inverse of the (z-mean)/std applied here)."""
    return Tensor.from_host(_vae_mean(), [1, 16, 1, 1], STDtype.F32, ctx)


def _std_ch(ctx: DeviceContext) raises -> Tensor:
    """latents_std as [1,16,1,1] F32 (same broadcast). Value = the decoder's
    _vae_std()."""
    return Tensor.from_host(_vae_std(), [1, 16, 1, 1], STDtype.F32, ctx)


def _normalize_latent(
    mean_lat: Tensor, mean_ch: Tensor, std_ch: Tensor, ctx: DeviceContext
) raises -> Tensor:
    """(z - latents_mean) / latents_std, per channel. mean_lat [1,16,LH,LW] F32;
    mean_ch/std_ch [1,16,1,1] F32. == ai-toolkit (z - mean) * (1/std)."""
    var centered = sub(mean_lat, mean_ch, ctx)   # [1,16,LH,LW]
    return div(centered, std_ch, ctx)            # [1,16,LH,LW]


def _encode_context(
    enc_dir: String, tok: Qwen3Tokenizer, caption: String, ctx: DeviceContext
) raises -> Tensor:
    """Tokenize PREFIX + caption + SUFFIX, encode the 12-layer krea2 stack -> context
    [1,LT,12,2560] BF16 (LT = stack.shape()[1] = len(ids) - DROP_IDX). Mirrors
    krea2_encode_cli._encode_one / ai-toolkit encode_krea_prompt. The encoder is
    loaded function-local (dropped on return) so its VRAM does not accumulate across
    the two encode calls."""
    var enc = load_krea2_qwen3vl_4b(enc_dir, ctx)
    var ids = tok.encode(KREA2_TPL_PREFIX + caption + KREA2_TPL_SUFFIX)
    return encode_krea2_stack(enc, ids, ctx)   # [1, LT, 12, 2560] bf16


def main() raises:
    var args = argv()
    if len(args) < 5:
        raise Error(
            "usage: krea2_prepare_cache <stage_dir> <out.safetensors> <n> <SIZE>"
        )
    var stage_dir = String(args[1])
    var out_path = String(args[2])
    var n = Int(String(args[3]))
    var size = Int(String(args[4]))
    if size % 16 != 0:
        raise Error("SIZE must be divisible by 16 (vae_scale_factor*patch)")

    var ctx = DeviceContext()
    print("[krea2-prepare] SIZE=", size, " n=", n, " -> ", out_path)

    # ── VAE encoder (square SIZE bucket). The QwenImageVaeEncoder is comptime-shaped
    # on IH/IW; we support the two production buckets 512/1024 here (add more comptime
    # arms if a new bucket is needed). encode_mean -> [1,16,LH,LW] (LH=LW=SIZE/8). ──
    print("[krea2-prepare] loading Qwen-Image VAE encoder")
    var mean_ch = _mean_ch(ctx)   # [1,16,1,1] F32
    var std_ch = _std_ch(ctx)     # [1,16,1,1] F32

    var imgs = ShardedSafeTensors.open(stage_dir + "/images.safetensors")
    var tok = Qwen3Tokenizer(String(KREA2_TOK_JSON))

    var names = List[String]()
    var tensors = List[TArc]()

    for i in range(n):
        # ── image -> deterministic MEAN latent [1,16,LH,LW] -> normalized BF16 ──
        # The VAE encoder is loaded per bucket (here a single square SIZE). Loading
        # it once outside the loop would be ideal, but the comptime IH/IW pin forces a
        # compile-time arm; we branch on SIZE and load the matching encoder once.
        var img_f32 = Tensor.from_view(
            imgs.tensor_view(String("image.") + String(i)), ctx
        )   # [1,3,SIZE,SIZE] F32
        # The Qwen-Image VAE weights are BF16 + encode_mean does NOT cast x →
        # feed a BF16 image (matches qwenimage_encoder_parity's BF16 convention).
        var img = cast_tensor(img_f32, STDtype.BF16, ctx)
        var lat_mean: Tensor
        # The encoder needs the Wan-key VAE (encoder.conv1/downsamples) — the
        # qwenimage_encoder_parity-gated checkpoint — NOT KREA2_VAE_DIR (the
        # diffusers Qwen-Image VAE: encoder.conv_in/down_blocks, which the
        # Wan-key encoder can't parse). Same weights, re-keyed (gate proves it).
        comptime KREA2_VAE_ENC_FILE = String(
            "/home/alex/.serenity/models/anima/split_files/vae/qwen_image_vae.safetensors"
        )
        if size == 512:
            var venc = QwenImageVaeEncoder[512, 512].load(KREA2_VAE_ENC_FILE, ctx)
            lat_mean = venc.encode_mean(img, ctx)         # [1,16,64,64]
        elif size == 1024:
            var venc = QwenImageVaeEncoder[1024, 1024].load(KREA2_VAE_ENC_FILE, ctx)
            lat_mean = venc.encode_mean(img, ctx)         # [1,16,128,128]
        else:
            raise Error(
                String("[krea2-prepare] unsupported SIZE ") + String(size)
                + " (add a comptime QwenImageVaeEncoder arm for it)"
            )
        # encode_mean returns BF16 (BF16 VAE weights). Normalize with F32 math, then
        # store the cache boundary as BF16, matching the product trainer path.
        var lat_f32 = cast_tensor(lat_mean, STDtype.F32, ctx)
        var clean_f32 = _normalize_latent(lat_f32, mean_ch, std_ch, ctx)
        var clean = cast_tensor(clean_f32, STDtype.BF16, ctx)

        # ── caption -> 12-layer Qwen3-VL stack [1,LT,12,2560] BF16 ──
        var prompt = _read_text(stage_dir + "/prompt." + String(i) + ".txt")
        var context = _encode_context(String(KREA2_TE_DIR), tok, prompt, ctx)
        var lt = context.shape()[1]   # LT == len(ids) - DROP_IDX

        names.append(String("clean.") + String(i))
        tensors.append(TArc(clean^))
        names.append(String("context.") + String(i))
        tensors.append(TArc(context^))
        # text_len.<i>: the caption's natural LT (== len(ids) - DROP_IDX). The reader
        # threads it so the trainer pins the comptime LT/LFULL/LPAD per sample and the
        # pos grid uses the right txt-zero count.
        var tl_host = List[Float32]()
        tl_host.append(Float32(lt))
        var tl = Tensor.from_host(tl_host^, [1], STDtype.F32, ctx)
        names.append(String("text_len.") + String(i))
        tensors.append(TArc(tl^))
        print("[krea2-prepare] sample", i, " LT=", lt, " latent=[1,16,",
              size // 8, ",", size // 8, "]")

    # ── caption dropout: optional uncond context from stage A's uncond.txt ("" ) ──
    var uncond_path = stage_dir + "/uncond.txt"
    if path_exists(uncond_path):
        var uprompt = _read_text(uncond_path)
        var u_context = _encode_context(String(KREA2_TE_DIR), tok, uprompt, ctx)
        var u_lt = u_context.shape()[1]
        names.append(String("context_uncond"))
        tensors.append(TArc(u_context^))
        var utl_host = List[Float32]()
        utl_host.append(Float32(u_lt))
        var utl = Tensor.from_host(utl_host^, [1], STDtype.F32, ctx)
        names.append(String("text_len_uncond"))
        tensors.append(TArc(utl^))
        print("[krea2-prepare] context_uncond LT=", u_lt)

    save_safetensors(names, tensors, out_path, ctx)
    print("[krea2-prepare] WROTE", out_path, " samples=", n)
