# krea2_prepare_cache.mojo — stage B of the Krea-2 LoRA-training cache prepare.
# (Stage A = the shared pure-Mojo image dataset stage.)
#
# The stage writes one `sample_NNNNN.safetensors` image tensor plus one raw
# `sample_NNNNN.txt` caption per sample. Those records go through
# the GATED Mojo encoders (QwenImageVaeEncoder, Qwen3Tokenizer + encode_krea2_stack)
# into the indexed safetensors training cache the KreaCacheReader streams:
#
#   clean.<i>     [1, 16, LH, LW]      BF16  normalized VAE latent (torchref boundary)
#   context.<i>   [1, LT, 12, 2560]    BF16  Qwen3-VL-4B 12-layer stack (== krea2_forward `context`)
#   text_len.<i>  [1]                  F32   LT (caption natural token count - DROP_IDX)
#   (optional)    context_uncond [1,LTu,12,2560] BF16 + text_len_uncond [1] F32
#   (edit)        ref.<i>        [1, 16, LH, LW]      BF16  reference/CONDITION latent
#   (omini edit)  cond_pos_delta.<i> [2] F32 + cond_pos_scale.<i> [1] F32
#                 (OminiControl condition RoPE offset/scale; EDIT = [0,0] / 1.0)
#
# WHY THESE EXACT TENSORS (read from the consumers):
#   * The DiT forward (models/dit/krea2_dit.mojo krea2_forward:1304) consumes `img`
#     [1,imglen,64] (PATCHIFIED latent) and `context` [1,LT,12,2560] and `pos`
#     [1,LFULL,3]. torchref keeps `latents` UNPACKED (B,C,h,w) all the way through
#     training — patchify is internal to predict_velocity/model.forward (pipeline.py
#     :85,102-117). So the cache stores the UNPACKED normalized latent `clean`
#     [1,16,LH,LW] (== torchref batch.latents) and the reader patchifies on demand
#     (after flow-noising), and builds `pos` deterministically from LH/LW + LT. This
#     lets the trainer add noise in latent space (noisy=(1-t)*clean+t*noise; target
#     = noise-clean, krea2.py:403) BEFORE the patchify, exactly like torchref.
#   * `context` is the encode_krea2_stack output [1,LT,12,2560] — un-flattened, which
#     is exactly what krea2_forward wants (predict_velocity reshapes its flattened
#     (B,Lt,n*d) back to (B,Lt,n,d); we never flatten, so no reshape needed).
#
# LATENT NORMALIZATION (the torchref data semantics, krea2.py encode_images:430-443):
#     z = vae.encode(img).latent_dist.SAMPLE()           # torchref SAMPLES the dist
#     latents_mean = cfg.latents_mean.view(1,16,1,1,1)
#     latents_std  = 1.0 / cfg.latents_std.view(1,16,1,1,1)
#     latents = (z - latents_mean) * latents_std         # == (z - mean) / std
#   We store this NORMALIZED latent. The Mojo QwenImageVaeEncoder.encode_mean returns
#   the deterministic posterior MEAN ([1,16,LH,LW], gated by qwenimage_encoder_parity
#   cos+max_abs), and we apply the SAME per-channel (z-mean)/std using the decoder's
#   _vae_mean()/_vae_std() (single source of truth — the decoder's denorm z*std+mean
#   is the exact inverse). DIVERGENCE (surfaced, not hidden): torchref uses
#   .latent_dist.SAMPLE() (a random Gaussian draw per epoch); we use the deterministic
#   MEAN (.mode()) for a REPRODUCIBLE cache. This matches the deterministic cache
#   convention used by the other Serenity trainers and removes the per-epoch
#   VAE-sampling noise, which is a deliberate, conventional simplification.
#
# CAPTION (== torchref text_encoder.py encode_krea_prompt + the Mojo inference path):
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
#   pixi run build-krea-cache
#   output/bin/serenity_krea2_prepare_cache \
#     <stage_dir> <out_cache.safetensors> <n> <SIZE>
#
# Mojo 1.0.0b1, NVIDIA GPU.

from max.gpu.host import DeviceContext
from std.memory import ArcPointer
from std.collections import List
from std.sys import argv

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.tensor_algebra import sub, div, zeros_device
from serenitymojo.registry.checkpoints import path_exists
from serenitymojo.tokenizer.tokenizer import Qwen3Tokenizer
from serenitymojo.models.vae.qwenimage_encoder import QwenImageVaeEncoder
from serenitymojo.models.vae.qwenimage_decoder import _vae_mean, _vae_std
from serenitymojo.models.text_encoder.krea2_qwen3vl_4b import (
    load_krea2_qwen3vl_4b,
    encode_krea2_stack,
)
from serenitymojo.models.text_encoder.qwen3_encoder import Qwen3Encoder
from serenitymojo.pipeline.krea2_paths import (
    KREA2_TE_DIR,
    KREA2_TOK_JSON,
    KREA2_VAE_DIR,
    KREA2_TPL_PREFIX,
    KREA2_TPL_SUFFIX,
)
from serenitymojo.models.krea2.krea2_buckets import (
    KREA2_LADDER_LEN,
    KREA2_LADDER_X100,
    krea2_lat_h,
    krea2_lat_w,
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
    mean_ch/std_ch [1,16,1,1] F32. == torchref (z - mean) * (1/std)."""
    var centered = sub(mean_lat, mean_ch, ctx)   # [1,16,LH,LW]
    return div(centered, std_ch, ctx)            # [1,16,LH,LW]


def _encode_context(
    enc: Qwen3Encoder, tok: Qwen3Tokenizer, caption: String, ctx: DeviceContext
) raises -> Tensor:
    """Tokenize PREFIX + caption + SUFFIX, encode the 12-layer krea2 stack -> context
    [1,LT,12,2560] BF16 (LT = stack.shape()[1] = len(ids) - DROP_IDX). Mirrors
    krea2_encode_cli._encode_one / torchref encode_krea_prompt.

    TAKES AN ALREADY-LOADED ENCODER. It used to take `enc_dir` and call
    load_krea2_qwen3vl_4b ITSELF, i.e. it re-loaded the 9.6 GB Qwen3-VL-4B from
    disk ON EVERY CALL. That was harmless where the pattern came from
    (krea2_encode_cli encodes exactly two prompts), but pass 1 below calls this
    ONCE PER SAMPLE — so a 3000-sample prepare did 3000 full model loads.
    MEASURED 2026-07-30 on the MagicBrush prepare: 59 captions/min (~1.0 s each,
    flat regardless of caption length), main thread pinned at 100% CPU streaming
    weights through the pinned host staging buffer while the GPU idled at 20%.
    That is host-side work on the critical path — the thing the no-CPU-compute
    rule exists to prevent — and it made the caption pass ~50 min for 3000
    samples where the actual encode work is seconds. Invisible at the 96-sample
    dev scale (1.6 min); a wall at production scale.

    The VRAM concern the old docstring cited is handled by SCOPE instead: pass 1
    owns the encoder and drops it before pass 2 loads any VAE, so the TE and the
    VAE are still never co-resident."""
    var ids = tok.encode(KREA2_TPL_PREFIX + caption + KREA2_TPL_SUFFIX)
    return encode_krea2_stack(enc, ids, ctx)   # [1, LT, 12, 2560] bf16


# The encoder needs the Wan-key VAE (encoder.conv1/downsamples) — the
# qwenimage_encoder_parity-gated checkpoint — NOT KREA2_VAE_DIR (the diffusers
# Qwen-Image VAE: encoder.conv_in/down_blocks, which the Wan-key encoder can't
# parse). Same weights, re-keyed (gate proves it). BF16 weights; encode_mean does
# NOT cast x, so we feed a BF16 image (qwenimage_encoder_parity's convention).
comptime KREA2_VAE_ENC_FILE = String("models/qwen-image/vae_encoder.safetensors")


def _stage_name(i: Int) -> String:
    var s = String(i)
    var pad = String("")
    for _ in range(5 - s.byte_length() if s.byte_length() < 5 else 0):
        pad += String("0")
    return String("sample_") + pad + s


def _stage_tensor_path(stage_dir: String, i: Int) -> String:
    return stage_dir + String("/") + _stage_name(i) + String(".safetensors")


def _stage_caption_path(stage_dir: String, i: Int) -> String:
    return stage_dir + String("/") + _stage_name(i) + String(".txt")


def _staged_image_hw_key(stage_dir: String, i: Int, key: String) raises -> Tuple[Int, Int]:
    """(IH, IW) of tensor `key` in the staged record ([1,3,IH,IW])."""
    var imgs = ShardedSafeTensors.open(_stage_tensor_path(stage_dir, i))
    var info = imgs.tensor_info(key)
    if len(info.shape) != 4 or info.shape[0] != 1 or info.shape[1] != 3:
        raise Error(
            String("[krea2-prepare] staged ") + key + String(".") + String(i)
            + " must be [1,3,IH,IW]"
        )
    return (info.shape[2], info.shape[3])


def _staged_image_hw(stage_dir: String, i: Int) raises -> Tuple[Int, Int]:
    """(IH, IW) from the generic Mojo dataset stage ([1,3,IH,IW])."""
    return _staged_image_hw_key(stage_dir, i, String("image"))


def _stage_has_key(stage_dir: String, i: Int, key: String) raises -> Bool:
    """True iff the staged record `i` carries tensor `key` (header-only probe)."""
    var imgs = ShardedSafeTensors.open(_stage_tensor_path(stage_dir, i))
    return key in imgs.name_to_shard


def _encode_one_latent_key[IH: Int, IW: Int](
    venc: QwenImageVaeEncoder[IH, IW],
    stage_dir: String, i: Int, key: String,
    mean_ch: Tensor, std_ch: Tensor, ctx: DeviceContext,
) raises -> Tensor:
    """<key>.<i> [1,3,IH,IW] -> deterministic MEAN latent [1,16,IH/8,IW/8] ->
    torchref-normalized BF16 latent. venc is the bucket-resident encoder. The
    key is a parameter so the OminiControl EDIT stage can push BOTH the target
    ("image") and the condition ("cond_image") of the SAME staged record through
    the IDENTICAL encoder + normalization — the condition latent must live on the
    same normalized scale as clean.<i> or the cond tokens enter the DiT off-scale."""
    var imgs = ShardedSafeTensors.open(_stage_tensor_path(stage_dir, i))
    var img_f32 = Tensor.from_view(imgs.tensor_view(key), ctx)
    var img = cast_tensor(img_f32, STDtype.BF16, ctx)
    var lat_mean = venc.encode_mean(img, ctx)              # [1,16,IH/8,IW/8] BF16
    var lat_f32 = cast_tensor(lat_mean, STDtype.F32, ctx)
    var clean_f32 = _normalize_latent(lat_f32, mean_ch, std_ch, ctx)
    return cast_tensor(clean_f32, STDtype.BF16, ctx)


def _encode_one_latent[IH: Int, IW: Int](
    venc: QwenImageVaeEncoder[IH, IW],
    stage_dir: String, i: Int,
    mean_ch: Tensor, std_ch: Tensor, ctx: DeviceContext,
) raises -> Tensor:
    """image.<i> [1,3,IH,IW] -> normalized BF16 clean latent (see the _key form)."""
    return _encode_one_latent_key[IH, IW](
        venc, stage_dir, i, String("image"), mean_ch, std_ch, ctx
    )


def _encode_black_latent[IH: Int, IW: Int](
    venc: QwenImageVaeEncoder[IH, IW],
    mean_ch: Tensor, std_ch: Tensor, ctx: DeviceContext,
) raises -> Tensor:
    """Encode RGB black in the exact staged-image domain (all channels -1).

    Omini condition dropout substitutes the VAE encoding of an empty/black
    condition; a numeric zero latent is not equivalent because the VAE and its
    latent normalization are non-zero-affine.
    """
    var zh = List[Float32]()
    zh.append(Float32(-1.0))
    var neg_one = Tensor.from_host(zh^, [1, 1, 1, 1], STDtype.BF16, ctx)
    var zero = zeros_device([1, 3, IH, IW], STDtype.BF16, ctx)
    var black = sub(zero, neg_one, ctx)
    # zero - (-1) is +1; staged black is -1, so negate via 0 - (+1).
    var black_img = sub(zero, black, ctx)
    var lat_mean = venc.encode_mean(black_img, ctx)
    var lat_f32 = cast_tensor(lat_mean, STDtype.F32, ctx)
    var clean_f32 = _normalize_latent(lat_f32, mean_ch, std_ch, ctx)
    return cast_tensor(clean_f32, STDtype.BF16, ctx)


def _copy_stage_tensor(
    stage_dir: String, i: Int, key: String, n_expect: Int, ctx: DeviceContext
) raises -> Tensor:
    """Copy a small rank-1 metadata tensor out of the staged record verbatim
    (OminiControl cond_pos_delta [2] / cond_pos_scale [1]). Fail loud on shape —
    a mis-shaped position record would silently move the condition grid."""
    var rec = ShardedSafeTensors.open(_stage_tensor_path(stage_dir, i))
    if key not in rec.name_to_shard:
        raise Error(
            String("[krea2-prepare] omini-edit: staged record ") + String(i)
            + " has cond_image but no " + key
            + " — re-stage with scripts/krea2_omini_stage_edit.py"
        )
    var t = cast_tensor(
        Tensor.from_view(rec.tensor_view(key), ctx), STDtype.F32, ctx
    )
    var sh = t.shape()
    var n = 1
    for d in range(len(sh)):
        n *= sh[d]
    if n != n_expect:
        raise Error(
            String("[krea2-prepare] omini-edit: ") + key + String(".") + String(i)
            + " has " + String(n) + " elements, expected " + String(n_expect)
        )
    return t^


def _run[E_UNITS: Int](
    stage_dir: String, out_path: String, n: Int, ref_dir: String,
    ctx: DeviceContext,
) raises:
    """Bucket-by-bucket prepare at edge units E_UNITS (512px→8, 1024px→16). Each
    staged sample is dispatched to the COMPTIME krea2 ladder arm matching its
    (IH,IW) — ONE QwenImageVaeEncoder[IH,IW] resident per bucket (mirrors
    zimage_prepare) — and its VAE latent stored at that bucket's canvas as
    clean.<i> [1,16,LH,LW]. Captions/context/text_len (aspect-independent) are
    encoded in a first pass. Fail loud if a staged shape is outside the ladder.

    IMAGE-EDIT (task #7): when `ref_dir` is non-empty, a PARALLEL staged
    <ref_dir>/sample_NNNNN.safetensors holds the REFERENCE image at the SAME index.
    Each
    reference is VAE-encoded with the SAME per-bucket encoder instance and stored
    in the additive ref.<i> slot -> an edit cache. The reference MUST land in the
    same (IH,IW) bucket as its target (same encoder arm); fail loud otherwise. The
    base clean/context/text_len keys are UNCHANGED — a reader that ignores ref.<i>
    (or an old cache without it) loads byte-identically.

    OMINICONTROL EDIT (C5): when `ref_dir` is EMPTY but the staged records carry a
    `cond_image` tensor (scripts/krea2_omini_stage_edit.py writes target+condition
    into ONE record), the CONDITION image is VAE-encoded with the same per-bucket
    encoder into the SAME additive `ref.<i>` slot, and the per-sample position
    metadata is copied through as `cond_pos_delta.<i>` [2] F32 + `cond_pos_scale.<i>`
    [1] F32. KEY-SCHEMA DECISION (recorded here because the intake proposed a new
    `cond_clean.<i>`): the condition latent REUSES `ref.<i>` because that slot
    already exists in this file and in krea2_cache_reader (has_ref/load_ref/
    ref_hw_at) with the IDENTICAL contract — [1,16,LH,LW] BF16, same encoder, same
    (z-mean)/std normalization, same bucket as the target. Adding a second key for
    the same bytes would give the trainer two ways to spell one thing. The
    OminiControl-specific part of the schema is the POSITION metadata, which is new
    and named per the intake. A cache is an Omini EDIT cache iff BOTH ref.<i> and
    cond_pos_delta.<i>/cond_pos_scale.<i> are present."""
    var edit_mode = len(ref_dir) > 0
    # Omini single-record edit staging: probe record 0 for `cond_image`.
    var omini_mode = False
    if not edit_mode and n > 0:
        omini_mode = _stage_has_key(stage_dir, 0, String("cond_image"))
    if omini_mode:
        print(
            "[krea2-prepare] OMINICONTROL EDIT MODE: cond_image in the staged",
            "records -> ref.<i> + cond_pos_delta.<i>/cond_pos_scale.<i>",
        )
    var mean_ch = _mean_ch(ctx)   # [1,16,1,1] F32
    var std_ch = _std_ch(ctx)     # [1,16,1,1] F32
    var tok = Qwen3Tokenizer(String(KREA2_TOK_JSON))

    var names = List[String]()
    var tensors = List[TArc]()

    # ── scan staged image shapes (header-only) ──────────────────────────────────
    var ihs = List[Int]()
    var iws = List[Int]()
    for i in range(n):
        var hw = _staged_image_hw(stage_dir, i)
        ihs.append(hw[0])
        iws.append(hw[1])
        if edit_mode:
            # The reference image.<i> MUST match the target's bucket (same encoder
            # arm below); fail loud so a mis-staged reference never silently skips.
            var rhw = _staged_image_hw(ref_dir, i)
            if rhw[0] != hw[0] or rhw[1] != hw[1]:
                raise Error(
                    String("[krea2-prepare] edit: reference image.") + String(i)
                    + " is " + String(rhw[0]) + "x" + String(rhw[1])
                    + " but target is " + String(hw[0]) + "x" + String(hw[1])
                    + " — a reference must be the SAME bucket canvas as its target"
                )
        if omini_mode:
            # EDIT: the condition overlaps the target spatially (delta [0,0],
            # scale 1.0), so S_COND == S_IMG — the cond canvas MUST equal the
            # target canvas. Fail loud; a mismatched grid silently breaks the
            # krea2_omini_cond_pos == img-grid identity the layout asserts.
            var chw = _staged_image_hw_key(stage_dir, i, String("cond_image"))
            if chw[0] != hw[0] or chw[1] != hw[1]:
                raise Error(
                    String("[krea2-prepare] omini-edit: cond_image.") + String(i)
                    + " is " + String(chw[0]) + "x" + String(chw[1])
                    + " but target is " + String(hw[0]) + "x" + String(hw[1])
                    + " — the EDIT condition must share the target's canvas"
                )

    # ── pass 1: captions -> context.<i> + text_len.<i> (aspect-independent) ─────
    # Done BEFORE any VAE is resident. This lexical scope owns ONE Qwen3-VL-4B
    # encoder for the whole caption pass and drops it before pass 2 loads a VAE.
    if n > 0:
        var enc = load_krea2_qwen3vl_4b(String(KREA2_TE_DIR), ctx)
        for i in range(n):
            var prompt = _read_text(_stage_caption_path(stage_dir, i))
            var context = _encode_context(enc, tok, prompt, ctx)
            var lt = context.shape()[1]   # LT == len(ids) - DROP_IDX
            names.append(String("context.") + String(i))
            tensors.append(TArc(context^))
            var tl_host = List[Float32]()
            tl_host.append(Float32(lt))
            var tl = Tensor.from_host(tl_host^, [1], STDtype.F32, ctx)
            names.append(String("text_len.") + String(i))
            tensors.append(TArc(tl^))
            print("[krea2-prepare] caption", i, " LT=", lt)

    # ── pass 2: bucket-by-bucket VAE encode -> clean.<i> [1,16,LH,LW] ───────────
    # comptime for over the krea2 ladder; ONE QwenImageVaeEncoder[IH,IW] resident
    # per bucket that has >=1 staged sample. clean.<i> may be written out of index
    # order — the reader keys by name, so order is irrelevant.
    var processed = List[Bool]()
    for _ in range(n):
        processed.append(False)

    comptime for bi in range(KREA2_LADDER_LEN):
        comptime X100_BI = KREA2_LADDER_X100[bi]
        comptime LH_BI = krea2_lat_h(X100_BI, E_UNITS)
        comptime LW_BI = krea2_lat_w(X100_BI, E_UNITS)
        comptime IH_BI = 8 * LH_BI
        comptime IW_BI = 8 * LW_BI
        var n_match = 0
        for i in range(n):
            if ihs[i] == IH_BI and iws[i] == IW_BI:
                n_match += 1
        if n_match > 0:
            print(
                "[krea2-prepare] bucket aspect_x100=", X100_BI,
                " canvas", IH_BI, "x", IW_BI, " latent", LH_BI, "x", LW_BI,
                " (", n_match, "samples ) loading QwenImageVaeEncoder",
            )
            var venc = QwenImageVaeEncoder[IH_BI, IW_BI].load(KREA2_VAE_ENC_FILE, ctx)
            if omini_mode:
                var black_lat = _encode_black_latent[IH_BI, IW_BI](
                    venc, mean_ch, std_ch, ctx
                )
                names.append(
                    String("ref_uncond.") + String(LH_BI) + String("x") + String(LW_BI)
                )
                tensors.append(TArc(black_lat^))
                print("[krea2-prepare] condition-drop black latent -> ref_uncond.",
                      LH_BI, "x", LW_BI)
            for i in range(n):
                if ihs[i] == IH_BI and iws[i] == IW_BI:
                    var clean = _encode_one_latent[IH_BI, IW_BI](
                        venc, stage_dir, i, mean_ch, std_ch, ctx
                    )
                    names.append(String("clean.") + String(i))
                    tensors.append(TArc(clean^))
                    processed[i] = True
                    print("[krea2-prepare] sample", i, " latent=[1,16,",
                          LH_BI, ",", LW_BI, "]")
                    if edit_mode:
                        # VAE-encode the REFERENCE image.<i> with the SAME resident
                        # encoder -> ref.<i> [1,16,LH,LW] BF16 (same normalized-
                        # latent contract as clean.<i>). Additive slot: the base
                        # keys above are untouched.
                        var ref_lat = _encode_one_latent[IH_BI, IW_BI](
                            venc, ref_dir, i, mean_ch, std_ch, ctx
                        )
                        names.append(String("ref.") + String(i))
                        tensors.append(TArc(ref_lat^))
                        print("[krea2-prepare] sample", i,
                              " ref latent=[1,16,", LH_BI, ",", LW_BI, "]")
                    if omini_mode:
                        # OminiControl EDIT: the CONDITION image rides in the SAME
                        # staged record. Same encoder instance, same normalization
                        # -> ref.<i>. The condition is CLEAN: the trainer never
                        # noises it (it enters at t=0), so nothing else is applied.
                        var cond_lat = _encode_one_latent_key[IH_BI, IW_BI](
                            venc, stage_dir, i, String("cond_image"),
                            mean_ch, std_ch, ctx
                        )
                        names.append(String("ref.") + String(i))
                        tensors.append(TArc(cond_lat^))
                        var cpd = _copy_stage_tensor(
                            stage_dir, i, String("cond_pos_delta"), 2, ctx
                        )
                        names.append(String("cond_pos_delta.") + String(i))
                        tensors.append(TArc(cpd^))
                        var cps = _copy_stage_tensor(
                            stage_dir, i, String("cond_pos_scale"), 1, ctx
                        )
                        names.append(String("cond_pos_scale.") + String(i))
                        tensors.append(TArc(cps^))
                        print("[krea2-prepare] sample", i,
                              " COND latent=[1,16,", LH_BI, ",", LW_BI, "]",
                              " -> ref.", i, " + cond_pos_*")

    for i in range(n):
        if not processed[i]:
            raise Error(
                String("[krea2-prepare] staged sample ") + String(i) + String(" (")
                + String(ihs[i]) + String("x") + String(iws[i])
                + String(") is outside the krea2 comptime ladder at edge units ")
                + String(E_UNITS) + String(" — re-stage with --buckets matching the")
                + String(" ladder (krea2_bucket_ladder_gate prints the pixel canvases).")
            )

    # ── caption dropout: optional uncond context from stage A's uncond.txt ("" ) ──
    var uncond_path = stage_dir + "/uncond.txt"
    if path_exists(uncond_path):
        var uprompt = _read_text(uncond_path)
        # MJ-1042: context_uncond MUST be a REAL empty-prompt encode (template only,
        # LT small ~5). The shared stage writes uncond.txt="" when dropout is enabled.
        # If it instead holds caption text, _encode_context would fold that text into
        # a caption-shaped tensor that silently masquerades as the uncond — the
        # "byte-identical FAKE cache_uncond" symptom. Fail loud rather than stage it.
        var u_stripped = String(uprompt.strip())
        if len(u_stripped) != 0:
            raise Error(
                String("[krea2-prepare] uncond.txt must be EMPTY (caption-dropout")
                + String(" uncond is a template-only empty-prompt encode); got ")
                + String(len(u_stripped))
                + String(" non-whitespace chars: '") + u_stripped
                + String("'. Re-stage from a config with caption dropout enabled.")
            )
        # Keep this load after pass 2 so names/tensors retain their historical
        # append order. It is the second and final TE load for this prepare.
        var u_enc = load_krea2_qwen3vl_4b(String(KREA2_TE_DIR), ctx)
        var u_context = _encode_context(u_enc, tok, uprompt, ctx)
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


def main() raises:
    var args = argv()
    if len(args) < 5:
        raise Error(
            "usage: krea2_prepare_cache <stage_dir> <out.safetensors> <n> <SIZE>"
            " [ref_stage_dir]  (SIZE = area budget 512 or 1024; per-sample bucket"
            " shapes come from the staged image tensors, dispatched over the krea2"
            " aspect ladder. IMAGE-EDIT: pass a 5th arg <ref_stage_dir> holding the"
            " REFERENCE images as image.<i> at the SAME index+bucket as the target;"
            " each reference is VAE-encoded and stored in the additive ref.<i> slot"
            " -> an edit cache.)"
        )
    var stage_dir = String(args[1])
    var out_path = String(args[2])
    var n = Int(String(args[3]))
    var size = Int(String(args[4]))
    # IMAGE-EDIT (task #7): optional parallel reference-image stage dir. Matched by
    # the SAME base index image.<i> across the two dirs (krea2's indexed analog of
    # Klein's sample_<i> filename match).
    var ref_dir = String("")
    if len(args) >= 6:
        ref_dir = String(args[5])

    var ctx = DeviceContext()
    print("[krea2-prepare] area-budget SIZE=", size, " n=", n, " -> ", out_path)
    if len(ref_dir) > 0:
        print("[krea2-prepare] EDIT MODE ref stage:", ref_dir, "(-> ref.<i> slot)")

    # SIZE selects the ladder EDGE units (E/align, align 64): 512px→e=8, 1024px→e=16.
    # The comptime QwenImageVaeEncoder[IH,IW] arms are instantiated per ladder bucket
    # inside _run[E_UNITS]; each staged image is dispatched by its (IH,IW).
    if size == 512:
        _run[8](stage_dir, out_path, n, ref_dir, ctx)
    elif size == 1024:
        _run[16](stage_dir, out_path, n, ref_dir, ctx)
    else:
        raise Error(
            String("[krea2-prepare] unsupported area budget SIZE ") + String(size)
            + " (only 512 and 1024 have comptime VAE ladder arms; add an edge-units"
            " branch in main() for a new area budget)"
        )
