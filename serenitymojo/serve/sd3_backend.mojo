# serenitymojo.serve.sd3_backend — the real SD3.5 Large 1024x1024 GenBackend.
#
# Wraps the VERIFIED SD3.5 Large inference stages behind the pull-based GenBackend
# seam (backend.mojo). Unlike sd3_sample_cli.mojo — which loads a PRE-CACHED
# triple-encoder sidecar (context_cond/context_uncond/pooled_cond/pooled_uncond) —
# THIS backend encodes the REAL params.prompt + params.negative at runtime through
# the verified CLIP-L + CLIP-G + T5-XXL modules, assembling SD3's context+pooled
# EXACTLY as inference-flame's sd3_infer.rs (the reference that produced the
# verified sidecars the Mojo SD3 MMDiT was anchored against):
#
#   # CLIP-L (CLIPTextModelWithProjection, but the SD3 CLIP-L safetensors ships NO
#   #         text_projection → raw EOS pool):
#   clip_l_hidden  = clip_l.encode_sd3(ids).hidden_states[-2]   [1,77,768]
#   clip_l_pooled  = pooler_output (post-final-LN @ EOS)          [1,768]
#   # CLIP-G (text_projection.weight [1280,1280] present → projected pool):
#   clip_g_hidden  = clip_g.encode_sd3(ids).hidden_states[-2]   [1,77,1280]
#   clip_g_pooled  = text_projection(pooler_output)              [1,1280]
#   # T5-XXL v1.1 (256 tokens, narrowed):
#   t5_hidden      = t5.encode(ids)                              [1,256,4096]
#
#   context = cat([ pad(clip_l_hidden, 0..4096),
#                   pad(clip_g_hidden, 0..4096),
#                   t5_hidden ], dim=1)                          [1,410,4096]
#   pooled  = cat([ clip_l_pooled, clip_g_pooled ], dim=1)       [1,2048]
#   (same for the negative prompt -> context_uncond / pooled_uncond.)
#
# THREE numeric details replicated verbatim from sd3_infer.rs (measured load-bearing):
#   1. CLIP uses hidden_states[-2] — the encoder output after layer (num_layers-2),
#      NOT the post-final-LN last_hidden_state. (encode_sdxl returns the WRONG tensor
#      for SD3; this backend reimplements the SD3 CLIP forward over the verified
#      ClipEncoder building blocks: _embed -> _layer[77]×L capturing penultimate.)
#   2. CLIP pooled = final_layer_norm(last layer) @ EOS, then text_projection IF the
#      file ships it (CLIP-G yes, CLIP-L no). Without CLIP-G's projection the pooled
#      has the wrong scale and SD3 Large blows up in block 0 (uniform output).
#   3. CLIP attention uses a PURE CAUSAL mask (no key-padding) — the reference
#      build_causal_mask masks only j<=i, so padded positions (after EOS) attend to
#      all earlier real tokens. We get pure-causal from _build_pad_mask(S,heads,S-1)
#      (valid_key_end=S-1 makes the j<=valid_key_end clause always true).
#   4. T5 ids: the Mojo T5Tokenizer.encode appends ONE EOS=1; sd3_infer.rs ALSO does
#      ids.push(1) after HF tokenization → a SECOND trailing EOS=1 before pad-to-256
#      (pad id 0). We replicate the second push to match the reference token stream.
#
# The denoise (28-step shifted rectified-flow CFG Euler) and VAE decode reuse
# sd3_sample_cli's exact block math (_sd3_large_forward visits 38 joint blocks;
# sd3_cfg/sd3_euler_step; load_sd3_embedded_ldm_decoder).
#
# Residency model (single-GPU, 24 GB):
#   * All 38 SD3.5 Large MMDiT blocks live in a complete pinned-host FP8 store.
#     Denoise stages from RAM only. The small pre/post-block gate
#     (SD3MMDiTPreBlockGate: patch/pos/context/cond/final weights) is loaded on
#     the GPU per job; the host store survives VAE decode and repeat jobs.
#   * The CLIP-L (~250 MB) + CLIP-G (~1.4 GB) + T5-XXL (~9.5 GB BF16) encoders are
#     loaded -> used -> freed PER JOB inside the ENCODE step (Movable-not-Copyable
#     Tensors drop at scope exit; the encode is staged so they never co-reside with
#     the resident gate at peak beyond the encoder footprint).
#   * The VAE decoder (~330 MB) is loaded PER JOB inside the DECODE step and freed.
#     SD3 drops only transient GPU staging before decode and keeps the host
#     store warm. The pre/post gate is released before VAE decode so the strict server
#     product gate can fit on a 24 GB card even when another small GPU context is
#     present; a later job reloads only the gate.
#
# step() state machine: ENCODE (per-job, blocking — announced phase="encoding")
#   → LOAD (gate + complete host store, once, announced phase="loading") → DENOISE×steps
#   (two RAM-staged forwards + Euler update per tick) → DECODE (announced
#   phase="decoding") → done. cancel() makes the next step() return cancelled and
#   frees all per-job tensors.
#
# Size support: the finite shared seven-shape 1MP image ladder. Each latent grid,
# learned-position crop, image-token count, joint-attention sequence, and VAE
# decoder is comptime-dispatched. steps/cfg/seed are honored at runtime.
#
# SimpleTuner/LyCORIS full-factor LoKr is applied through the exact
# SwarmUI/Comfy efficient Kronecker bypass. img2img remains fail-loud.

from std.collections import Optional
from std.ffi import external_call
from std.gpu.host import DeviceContext
from std.memory import alloc, ArcPointer
from std.time import perf_counter_ns

from image.buffer import Image
from image.png import encode_png_with_text

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.ffi import BytePtr
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.image.png import _quantize, ValueRange
from serenitymojo.offload.vmm_cuda import cu_mempool_trim_current, cu_mem_get_info
from serenitymojo.offload.turbo_planned_loader import TurboPlannedLoader
from serenitymojo.offload.plan import (
    OffloadConfig,
    build_sd35_large_block_plan,
)
from serenitymojo.registry.checkpoints import path_exists

from serenitymojo.tokenizer.clip_tokenizer import ClipTokenizer
from serenitymojo.tokenizer.t5_tokenizer import T5Tokenizer
from serenitymojo.models.text_encoder.clip_encoder import (
    ClipEncoder, ClipConfig, _build_pad_mask, slice_seq,
)
from serenitymojo.models.text_encoder.t5_encoder import T5Encoder, T5Config
from serenitymojo.models.dit.sd3_contract import (
    SD3_LARGE_DEPTH, SD3_LARGE_HEAD_DIM, SD3_LARGE_HIDDEN,
    SD3_LARGE_LATENT_CHANNELS, SD3_LARGE_NUM_HEADS, SD3_LARGE_TEXT_TOKENS,
    sd3_large_schedule_shift,
)
from serenitymojo.models.dit.sd3_mmdit import (
    SD3MMDiTPreBlockGate, _sd3_joint_block,
)
from serenitymojo.models.sd35.sd3_lokr_overlay import (
    Sd3LokrOverlay, load_sd3_large_lokr,
)
from serenitymojo.pipeline.sd3_tiled_decode import sd3_tiled_decode_5x5_lowmem
from serenitymojo.models.vae.ldm_decoder import load_sd3_embedded_ldm_decoder
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.linear import linear
from serenitymojo.ops.norm import layer_norm as _ops_layer_norm
from serenitymojo.ops.random import randn
from serenitymojo.ops.tensor_algebra import reshape, concat
from serenitymojo.sampling.sd3_flow_match import (
    SD3FlowMatchScheduler, sd3_cfg, sd3_euler_step,
)
from serenitymojo.sampling.dpmpp_2m import (
    MultistepHistory, denoised_from_velocity, dpmpp_2m_step,
    lambda_from_sigma_f64,
)
from serenitymojo.sampling.variation_noise import variation_noise_chw
from serenitymojo.sampling.sampler_registry import (
    sampler_admission_for_backend, scheduler_admission_for_backend,
)
from serenitymojo.serve.backend import (
    GenBackend, JobParams, StepResult, reject_unsupported_common_runtime_params,
    reject_unsupported_reference_image_params, reject_unsupported_mask_image_params,
    reject_unsupported_inpaint_conditioning_params,
    reject_unsupported_qwen_edit_conditioning_params,
    reject_unsupported_conditioning_mask_params, reject_unsupported_lanpaint_params,
    warn_unsupported_advanced_sampling_params,
)
from serenitymojo.serve.product_manifest import (
    json_bool, json_escape, peak_vram_mib, write_text_file,
)
from serenitymojo.training.aspect_buckets import (
    DEFAULT_ASPECT_LADDER_LEN, DEFAULT_ASPECT_LADDER_X100,
    aspect_lat_h_units, aspect_lat_w_units,
)


comptime GENPARAMS_TEXT_KEY = "serenity.genparams.v1"
comptime SD3_PRODUCT_EDGE_UNITS = 16


def _sd3_shape_supported(width: Int, height: Int) -> Bool:
    comptime for bi in range(DEFAULT_ASPECT_LADDER_LEN):
        comptime X100_BI = DEFAULT_ASPECT_LADDER_X100[bi]
        comptime LH_BI = aspect_lat_h_units(X100_BI, SD3_PRODUCT_EDGE_UNITS)
        comptime LW_BI = aspect_lat_w_units(X100_BI, SD3_PRODUCT_EDGE_UNITS)
        if width == LW_BI * 8 and height == LH_BI * 8:
            return True
    return False

comptime LC = SD3_LARGE_LATENT_CHANNELS     # 16

comptime CLIP_LEN = 77
comptime T5_LEN = 256

# SD3 Large joint sequence: image + text tokens (text = 77 + 77 + 256 = 410).
comptime N_CTX = SD3_LARGE_TEXT_TOKENS      # 410
comptime DEPTH = SD3_LARGE_DEPTH            # 38
comptime H_HEADS = SD3_LARGE_NUM_HEADS      # 38
comptime H_DIM = SD3_LARGE_HEAD_DIM         # 64
comptime HIDDEN = SD3_LARGE_HIDDEN          # 2432
comptime DUAL_BLOCKS = 0                    # No dual attention in Large
comptime CONTEXT_DIM = 4096                 # joint_attention_dim (T5 hidden width)

# ── verified model + tokenizer paths (match sd3_infer.rs constants exactly) ──
comptime MODEL_PATH = "models/sd3.5/transformer.safetensors"
comptime CLIP_L_PATH = "models/text-encoders/clip_l.safetensors"
comptime CLIP_G_PATH = "models/text-encoders/clip_g.safetensors"
comptime T5_PATH = "models/text-encoders/t5xxl_fp16.safetensors"
comptime CLIP_L_TOK = "models/text-encoders/clip_l.tokenizer.json"
comptime CLIP_G_TOK = "models/text-encoders/clip_g.tokenizer.json"
comptime T5_TOK = "models/text-encoders/t5xxl_fp16.tokenizer.json"
comptime CLIP_G_TEXT_PROJ = "text_projection.weight"
comptime CONDITIONING_PRODUCER = "output/bin/serenity_sd3_conditioning"

comptime CLIP_PAD_ID = 49407   # CLIP eos == pad
comptime CLIP_EOS_ID = 49407
comptime T5_EOS_ID = 1
comptime T5_PAD_ID = 0

# MJ-1053: SD3.5 Large gate-recipe defaults (sd3_large_pipeline_1024_multistep.mojo:
# NUM_STEPS=28 line 72, seed 42 line 75). Applied ONLY for the degenerate
# steps<=0 / seed<0 inputs — see the note at the apply site in start(). The frozen
# wire contract carries no "unset" sentinel, so an omitted field cannot be told
# apart from a user who deliberately chose the JobParams global default
# (steps=20 / seed=0); a proper fix needs a sentinel added to the wire crate +
# JobParams, which is out of scope here (do NOT touch backend.mojo/the wire).
comptime SD3_DEFAULT_STEPS = 28
comptime SD3_DEFAULT_SEED = 42


comptime S3PHASE_IDLE = 0
comptime S3PHASE_ENCODE = 1
comptime S3PHASE_LOAD = 2
comptime S3PHASE_DENOISE = 3
comptime S3PHASE_DECODE = 4

# Whole-image VAE decode is preferred when it fits: tiled decode is MEASURED to
# degrade output (MJ-1054). SD3 releases GPU staging and its resident gate
# before decode, so free VRAM is usually ample. After that release + mempool trim
# we query free VRAM and decode whole when it clears this bar, else fall back to
# the (degrading) 5x5-lowmem tiled path. 14 GiB is a conservative estimate to be
# tightened by measurement.
comptime WHOLE_DECODE_MIN_FREE_BYTES = 14 * 1024 * 1024 * 1024  # 14 GiB


def _shell(cmd: String) -> Int:
    var n = cmd.byte_length()
    var buf = alloc[UInt8](n + 1)
    var src = cmd.as_bytes()
    for i in range(n):
        buf[i] = src[i]
    buf[n] = 0
    var status = Int(external_call["system", Int32](BytePtr(unsafe_from_address=Int(buf))))
    buf.free()
    return status


def _print_vram(tag: String):
    _ = _shell(
        String("echo -n '[sd3][vram] ") + tag
        + ": ' && nvidia-smi --query-gpu=memory.used --format=csv,noheader"
    )


def _to_bf16(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    """F16/F32/BF16 -> BF16 (F16 goes through F32 to avoid a direct F16->BF16 path)."""
    if x.dtype() == STDtype.BF16:
        return cast_tensor(x, STDtype.BF16, ctx)
    if x.dtype() == STDtype.F16:
        var x_f32 = cast_tensor(x, STDtype.F32, ctx)
        return cast_tensor(x_f32, STDtype.BF16, ctx)
    return cast_tensor(x, STDtype.BF16, ctx)


def _fit_clip_ids(var ids: List[Int]) -> List[Int]:
    """Pad/truncate CLIP ids to 77, keeping a real EOS at the tail (HF CLIP: pad==eos).
    ClipTokenizer.encode already wrapped with BOS(49406)+EOS(49407). Matches
    sd3_infer.rs tokenize_clip (truncate to 77, pad 49407)."""
    if len(ids) > CLIP_LEN:
        var trimmed = List[Int]()
        for i in range(CLIP_LEN):
            trimmed.append(ids[i])
        trimmed[CLIP_LEN - 1] = CLIP_EOS_ID
        return trimmed^
    while len(ids) < CLIP_LEN:
        ids.append(CLIP_PAD_ID)
    return ids^


def _fit_t5_ids(var ids: List[Int]) -> List[Int]:
    """T5Tokenizer.encode already appended ONE EOS=1; sd3_infer.rs ALSO does
    ids.push(1) after HF tokenization -> append a SECOND EOS=1, then truncate to 256
    and pad with 0 (matches tokenize_t5 verbatim)."""
    ids.append(T5_EOS_ID)
    if len(ids) > T5_LEN:
        var trimmed = List[Int]()
        for i in range(T5_LEN):
            trimmed.append(ids[i])
        return trimmed^
    while len(ids) < T5_LEN:
        ids.append(T5_PAD_ID)
    return ids^


def _save_rgb_png_with_text(
    rgb: Tensor, path: String, params_json: String, ctx: DeviceContext
) raises:
    """[1,3,H,W] SIGNED float tensor → 8-bit RGB PNG with the job params in a
    serenity.genparams.v1 tEXt chunk. Quantization math == save_png's
    (_quantize, ValueRange.SIGNED); only the writer differs (tEXt support).
    Identical to sdxl_backend / qwenimage_backend._save_rgb_png_with_text."""
    var shape = rgb.shape()
    if len(shape) != 4 or shape[0] != 1 or shape[1] != 3:
        raise Error("sd3_backend: expected [1,3,H,W] rgb tensor")
    var height = shape[2]
    var width = shape[3]
    var host = rgb.to_host(ctx)
    var plane = height * width
    if len(host) != 3 * plane:
        raise Error("sd3_backend: rgb to_host size mismatch")
    var img = Image.new(width, height, 3)
    for y in range(height):
        var row = y * width
        for x in range(width):
            var off = row + x
            img.set(x, y, 0, _quantize(host[0 * plane + off], ValueRange.SIGNED))
            img.set(x, y, 1, _quantize(host[1 * plane + off], ValueRange.SIGNED))
            img.set(x, y, 2, _quantize(host[2 * plane + off], ValueRange.SIGNED))
    var kws = List[String]()
    var vals = List[String]()
    kws.append(String(GENPARAMS_TEXT_KEY))
    vals.append(params_json.copy())
    encode_png_with_text(img, path, kws, vals)


# ── SD3 CLIP encode: (penultimate_hidden [1,77,hidden], pooled [1,proj_dim]) ──
# Reimplements inference-flame ClipEncoder::encode_sd3 over the verified Mojo
# ClipEncoder building blocks (_embed, _layer[77], _w, slice_seq, _build_pad_mask):
#   - hidden_states[-2] = output after layer (num_layers - 2), captured pre-final-LN.
#   - pooled = final_layer_norm(last-layer hidden) @ EOS, then text_projection if
#     `text_proj` is provided (CLIP-G) else raw (CLIP-L).
#   - PURE CAUSAL mask (no key-padding): _build_pad_mask(S, heads, S-1).
# `text_proj` is an Optional CLIP-G text_projection.weight [hidden,hidden]; pass
# None for CLIP-L. Returns BF16 tensors (the encoders store BF16; layer math is
# F32-accum, BF16-store — matches the verified encode_sdxl path).
def _clip_encode_sd3(
    enc: ClipEncoder,
    var token_ids: List[Int],
    text_proj: Optional[Tensor],
    ctx: DeviceContext,
) raises -> Tuple[Tensor, Tensor]:
    var cfg = enc.config
    var hid = cfg.hidden_size
    var num_layers = cfg.num_layers
    var heads = cfg.num_heads

    # pad / truncate to 77, EOS-pad (already done by _fit_clip_ids, but be safe).
    token_ids = _fit_clip_ids(token_ids^)

    # first EOS position (argmax over id==eos returns the FIRST 1).
    var real_eos = CLIP_LEN - 1
    for i in range(CLIP_LEN):
        if token_ids[i] == CLIP_EOS_ID:
            real_eos = i
            break

    var hidden = enc._embed(token_ids, ctx)         # [1,77,hidden]
    var dtype = hidden.dtype()

    # PURE causal mask [1, H, 77, 77]: valid_key_end = S-1 => the j<=valid_key_end
    # clause is always true, leaving only the causal j<=i term (matches the
    # reference build_causal_mask).
    var mask_data = _build_pad_mask(CLIP_LEN, heads, CLIP_LEN - 1)
    var msh = List[Int]()
    msh.append(1)
    msh.append(heads)
    msh.append(CLIP_LEN)
    msh.append(CLIP_LEN)
    var mask = Tensor.from_host(mask_data, msh^, dtype, ctx)

    # transformer layers, capturing hidden_states[-2] (output after layer L-2).
    # `penultimate` must be definitely-initialized before the loop (the capture is
    # conditional), so seed it with a clone of the embeddings (cheap, no extra
    # embed kernel); it is overwritten at i == penultimate_idx below.
    var penultimate_idx = num_layers - 2
    var penultimate = hidden.clone(ctx)  # placeholder, overwritten below
    for i in range(num_layers):
        hidden = enc._layer[CLIP_LEN](i, hidden, mask, ctx)
        if i == penultimate_idx:
            penultimate = hidden.clone(ctx)

    # pooled = final_layer_norm(last-layer hidden) @ EOS, optionally projected.
    # _w returns a borrow of a Movable-not-Copyable Tensor, so bind with `ref`
    # (matching the encoder's own encode_sdxl) — `var` would attempt a copy.
    ref fw = enc._w(String("text_model.final_layer_norm.weight"))
    ref fb = enc._w(String("text_model.final_layer_norm.bias"))
    # The encoder exposes no public LN, so apply the verified ops.norm.layer_norm
    # directly (same affine LN the encoder uses for its own final_layer_norm).
    var last_hidden = _ops_layer_norm(hidden, fw, fb, cfg.layer_norm_eps, ctx)
    var pooled_slice = slice_seq(last_hidden, real_eos, ctx)   # [1,1,hidden]
    var psh = List[Int]()
    psh.append(1)
    psh.append(hid)
    var pooled = reshape(pooled_slice, psh^, ctx)              # [1,hidden]

    if text_proj:
        # text_embeds = pooled @ text_projection^T  (HF: text_projection is a
        # no-bias Linear [out,in]; ops.linear does y = x @ W^T).
        pooled = linear(pooled, text_proj.value(), Optional[Tensor](None), ctx)

    return (penultimate^, _to_bf16(pooled, ctx))


# ── zero-pad a [1,77,hidden] CLIP hidden state's last dim up to 4096 ──
def _zero_pad_last_dim(x: Tensor, target: Int, ctx: DeviceContext) raises -> Tensor:
    var sh = x.shape()
    if len(sh) != 3:
        raise Error("sd3_backend: _zero_pad_last_dim expects [1,S,C]")
    var cur = sh[2]
    if cur == target:
        return _to_bf16(x, ctx)
    if cur > target:
        raise Error("sd3_backend: _zero_pad_last_dim source wider than target")
    var pad_c = target - cur
    var zeros_host = List[Float32]()
    var nz = sh[0] * sh[1] * pad_c
    for _ in range(nz):
        zeros_host.append(Float32(0.0))
    var zsh = List[Int]()
    zsh.append(sh[0])
    zsh.append(sh[1])
    zsh.append(pad_c)
    var zeros = Tensor.from_host(zeros_host, zsh^, STDtype.BF16, ctx)
    return concat(2, ctx, _to_bf16(x, ctx), zeros)


# ── SD3 conditioning bundle (per job) ──────────────────────────────────────────
struct Sd3Caps(Movable):
    var context: Tensor         # [1,410,4096] BF16 (cond)
    var context_uncond: Tensor  # [1,410,4096] BF16 (uncond)
    var pooled: Tensor          # [1,2048]     BF16 (cond)
    var pooled_uncond: Tensor   # [1,2048]     BF16 (uncond)

    def __init__(
        out self, var context: Tensor, var context_uncond: Tensor,
        var pooled: Tensor, var pooled_uncond: Tensor,
    ):
        self.context = context^
        self.context_uncond = context_uncond^
        self.pooled = pooled^
        self.pooled_uncond = pooled_uncond^


# ── assemble SD3 context [1,410,4096] + pooled [1,2048] for ONE prompt ──
def _assemble_one(
    text: String,
    clip_l: ClipEncoder,
    clip_g: ClipEncoder,
    text_proj_g: Tensor,
    t5: T5Encoder[T5_LEN],
    clip_l_tok: ClipTokenizer,
    clip_g_tok: ClipTokenizer,
    t5_tok: T5Tokenizer,
    ctx: DeviceContext,
) raises -> Tuple[Tensor, Tensor]:
    # CLIP-L: penultimate hidden [1,77,768], raw pooled [1,768] (no text_projection).
    var l_ids = _fit_clip_ids(clip_l_tok.encode(text))
    var l_out = _clip_encode_sd3(clip_l, l_ids^, Optional[Tensor](None), ctx)
    var l_hidden = l_out[0].clone(ctx)
    var l_pool = l_out[1].clone(ctx)

    # CLIP-G: penultimate hidden [1,77,1280], projected pooled [1,1280].
    var g_ids = _fit_clip_ids(clip_g_tok.encode(text))
    var g_out = _clip_encode_sd3(
        clip_g, g_ids^, Optional[Tensor](text_proj_g.clone(ctx)), ctx
    )
    var g_hidden = g_out[0].clone(ctx)
    var g_pool = g_out[1].clone(ctx)

    # T5-XXL: [1,256,4096] (encode appends one EOS; _fit_t5_ids adds the second + pad).
    var t5_ids = _fit_t5_ids(t5_tok.encode(text))
    var t5_hidden = _to_bf16(t5.encode(t5_ids^, ctx), ctx)     # [1,256,4096]

    # context = cat([pad(l_hidden,4096), pad(g_hidden,4096), t5_hidden], dim=1).
    var l_pad = _zero_pad_last_dim(l_hidden, CONTEXT_DIM, ctx)  # [1,77,4096]
    var g_pad = _zero_pad_last_dim(g_hidden, CONTEXT_DIM, ctx)  # [1,77,4096]
    var context = concat(1, ctx, l_pad, g_pad, t5_hidden)      # [1,410,4096]
    context = _to_bf16(context, ctx)

    # pooled = cat([l_pool, g_pool], dim=1) -> [1,2048].
    var pooled = concat(1, ctx, _to_bf16(l_pool, ctx), _to_bf16(g_pool, ctx))
    pooled = _to_bf16(pooled, ctx)

    return (context^, pooled^)


# ── SD3 Large MMDiT forward (one pass) — verbatim from sd3_sample_cli ──
def _sd3_large_forward[LH_: Int, LW_: Int, N_IMG_: Int, S_JOINT_: Int](
    latent: Tensor,          # [1, 16, LH, LW] BF16
    sigma: Float32,
    context: Tensor,         # [1, N_CTX, 4096] BF16
    pooled: Tensor,          # [1, 2048] BF16
    gate: SD3MMDiTPreBlockGate,
    mut loader: TurboPlannedLoader,
    ctx: DeviceContext,
    lora: Optional[ArcPointer[Sd3LokrOverlay]] = None,
) raises -> Tensor:
    var x_tokens = gate.latent_patch_embed[LH_, LW_](latent, ctx)
    var c = gate.conditioning(sigma, pooled, ctx)
    var ctx_tokens = gate.context_embed[N_CTX](context, ctx)

    loader.set_config(OffloadConfig.synchronous_single())
    loader.prefetch_with_ctx(0, ctx)
    for i in range(DEPTH):
        var is_last = (i == DEPTH - 1)
        var handle = loader.await_block(i, ctx)
        loader.prefetch_next_with_ctx(i, ctx)
        _sd3_joint_block[1, S_JOINT_, N_CTX, N_IMG_, H_HEADS, H_DIM](
            ctx_tokens, x_tokens, c, handle.block, i, is_last,
            DUAL_BLOCKS, HIDDEN, ctx,
            lora,
        )
        loader.mark_active_block_done(ctx)

    var patch_out = gate.final_layer_tokens(x_tokens, c, ctx)
    return gate.final_unpatchify[LH_, LW_](patch_out, ctx)


struct Sd3Backend(GenBackend, Movable):
    var ctx: DeviceContext

    # ── resident across jobs (pre/post gate + complete host store) ────────────
    var loaded: Bool
    var gate: List[ArcPointer[SD3MMDiTPreBlockGate]]  # 0/1 (resident gate)
    var loader: List[ArcPointer[TurboPlannedLoader]]  # 0/1 (complete host store)
    var loader_checkpoint: String
    var lora: List[ArcPointer[Sd3LokrOverlay]]         # 0/1 runtime LoKr
    var lora_target_count: Int

    # ── per-job state (cleared on done/failed/cancelled) ──
    var active: Bool
    var cancel_flag: Bool
    var phase: Int
    var announced: Bool
    var cur: Int
    var params: JobParams
    var cfg: Float32
    var executed_sampler: String
    var dpmpp_history: MultistepHistory
    var dpmpp_update_steps: Int
    var dpmpp_second_order_steps: Int
    var caps: List[ArcPointer[Sd3Caps]]                 # 0/1
    var sched: List[ArcPointer[SD3FlowMatchScheduler]]  # 0/1
    var latent: List[ArcPointer[Tensor]]                # 0/1 ([1,16,LH,LW] BF16)
    var job_t0_ns: UInt
    var load_seconds: Float64
    var text_encode_seconds: Float64
    var prepare_seconds: Float64
    var denoise_seconds: Float64
    var vae_decode_seconds: Float64
    var total_vram_bytes: Int
    var min_free_bytes: Int

    def __init__(out self) raises:
        self.ctx = DeviceContext()
        self.loaded = False
        self.gate = List[ArcPointer[SD3MMDiTPreBlockGate]]()
        self.loader = List[ArcPointer[TurboPlannedLoader]]()
        self.loader_checkpoint = String("")
        self.lora = List[ArcPointer[Sd3LokrOverlay]]()
        self.lora_target_count = 0
        self.active = False
        self.cancel_flag = False
        self.phase = S3PHASE_IDLE
        self.announced = False
        self.cur = 0
        self.params = JobParams()
        self.cfg = Float32(4.5)
        self.executed_sampler = String("sd3_flowmatch_euler")
        self.dpmpp_history = MultistepHistory(1)
        self.dpmpp_update_steps = 0
        self.dpmpp_second_order_steps = 0
        self.caps = List[ArcPointer[Sd3Caps]]()
        self.sched = List[ArcPointer[SD3FlowMatchScheduler]]()
        self.latent = List[ArcPointer[Tensor]]()
        self.job_t0_ns = UInt(0)
        self.load_seconds = 0.0
        self.text_encode_seconds = 0.0
        self.prepare_seconds = 0.0
        self.denoise_seconds = 0.0
        self.vae_decode_seconds = 0.0
        self.total_vram_bytes = 0
        self.min_free_bytes = 0

    def backend_name(self) -> String:
        return String("sd3")

    def model_name(self) -> String:
        return (
            self.params.model.copy()
            if self.params.model.byte_length() > 0
            else String("SD3.5 Large")
        )

    def resident_model(self) -> String:
        return self.params.model.copy() if self.loaded else String("")

    def _checkpoint_path(self) -> String:
        if self.params.checkpoint_path.byte_length() > 0:
            return self.params.checkpoint_path.copy()
        return String(MODEL_PATH)

    # ── job admission ─────────────────────────────────────────────────────────
    def start(mut self, params: JobParams) raises:
        if self.active:
            raise Error("Sd3Backend.start: a job is already running")
        reject_unsupported_common_runtime_params(params, String("sd3"))
        reject_unsupported_reference_image_params(params, String("sd3"))
        reject_unsupported_inpaint_conditioning_params(params, String("sd3"))
        reject_unsupported_qwen_edit_conditioning_params(params, String("sd3"))
        reject_unsupported_conditioning_mask_params(params, String("sd3"))
        reject_unsupported_mask_image_params(params, String("sd3"))
        reject_unsupported_lanpaint_params(params, String("sd3"))
        var sampler_admission = sampler_admission_for_backend(String("sd3"), params.sampler)
        if not sampler_admission.supported:
            raise Error(
                String("sd3: unsupported sampler '") + params.sampler
                + String("'; ") + sampler_admission.reason
            )
        var scheduler_admission = scheduler_admission_for_backend(String("sd3"), params.scheduler)
        if not scheduler_admission.supported:
            raise Error(
                String("sd3: unsupported scheduler '") + params.scheduler
                + String("'; ") + scheduler_admission.reason
            )
        # Finite comptime dispatch over the shared seven-shape 1MP product ladder.
        if not _sd3_shape_supported(params.width, params.height):
            raise Error(
                String("sd3: unsupported size ") + String(params.width)
                + "x" + String(params.height)
                + " — supported sizes are the compiled seven-shape 1MP aspect ladder"
            )
        if len(params.loras) > 1:
            raise Error(
                "sd3: this runtime currently supports one creator-compatible"
                " LyCORIS LoKr at a time"
            )
        if params.init_image.byte_length() > 0:
            raise Error(
                "sd3: img2img is not supported for SD3.5 Large yet;"
                " submit without an init image"
            )
        if params.checkpoint_path.byte_length() > 0 and not path_exists(params.checkpoint_path):
            raise Error(
                String("sd3: selected checkpoint not found: ")
                + params.checkpoint_path
            )
        # Warn-loud (never silently drop) on any advanced-sampling knob set but
        # unsupported by this fixed flow-match Euler path.
        warn_unsupported_advanced_sampling_params(params, String("sd3"), List[String]())
        var next_checkpoint = (
            params.checkpoint_path.copy()
            if params.checkpoint_path.byte_length() > 0
            else String(MODEL_PATH)
        )
        # A cancelled/failed job can leave GPU gate/LoKr state resident. Drop
        # that per-job state, but retain a matching complete host store.
        if self.loaded:
            self.gate = List[ArcPointer[SD3MMDiTPreBlockGate]]()
            self.lora = List[ArcPointer[Sd3LokrOverlay]]()
            self.lora_target_count = 0
            self.loaded = False
        if len(self.loader) == 1 and self.loader_checkpoint != next_checkpoint:
            print("[sd3] selected checkpoint changed; dropping old host store")
            self.loader = List[ArcPointer[TurboPlannedLoader]]()
            self.loader_checkpoint = String("")
        self.params = params.copy()
        self.executed_sampler = sampler_admission.executed
        self.dpmpp_history = MultistepHistory(1)
        self.dpmpp_update_steps = 0
        self.dpmpp_second_order_steps = 0
        # MJ-1053: backfill the SD3 gate-recipe defaults ONLY for degenerate
        # inputs. KNOWN LIMITATION (accepted): the wire protocol has no "unset"
        # sentinel, so a request that omits steps/seed arrives indistinguishable
        # from the JobParams global default (steps=20 / seed=0); we therefore
        # cannot promote an omitted 20 to the SD3-tuned 28 without also clobbering
        # a user who really asked for 20. We only fill steps<=0 -> 28 and
        # seed<0 -> 42 (values no real request sets). See SD3_DEFAULT_* above.
        if self.params.steps <= 0:
            self.params.steps = SD3_DEFAULT_STEPS
        if self.params.seed < 0:
            self.params.seed = SD3_DEFAULT_SEED
        self.cfg = Float32(params.cfg)
        self.active = True
        self.cancel_flag = False
        self.cur = 0
        self.announced = False
        self.phase = S3PHASE_ENCODE
        self.job_t0_ns = perf_counter_ns()
        self.load_seconds = 0.0
        self.text_encode_seconds = 0.0
        self.prepare_seconds = 0.0
        self.denoise_seconds = 0.0
        self.vae_decode_seconds = 0.0
        var mem = cu_mem_get_info()
        self.total_vram_bytes = mem.total_bytes
        self.min_free_bytes = mem.free_bytes
        self._record_vram()

    def cancel(mut self):
        self.cancel_flag = True

    def between_jobs_trim(mut self) raises:
        """Reclaim the per-job transient peak (CLIP-L+G+T5 encoders ~11 GB, the VAE
        decoder ~330 MB, per-forward staged block + activations) back to the OS via
        cuMemPoolTrimTo. The pre/post gate is released before VAE decode on the
        product path, so this mostly clears allocator residue; denoiser weights
        are staged from the complete FP8 host store, never the checkpoint."""
        var before = cu_mem_get_info()
        self.ctx.synchronize()
        cu_mempool_trim_current(0)
        self.ctx.synchronize()
        var after = cu_mem_get_info()
        print("[sd3] between-jobs trim: used",
              before.used_bytes() // (1024 * 1024), "->",
              after.used_bytes() // (1024 * 1024), "MiB (reclaimed",
              (before.used_bytes() - after.used_bytes()) // (1024 * 1024), "MiB)")

    def _record_vram(mut self) raises:
        var mem = cu_mem_get_info()
        if self.total_vram_bytes == 0:
            self.total_vram_bytes = mem.total_bytes
        if self.min_free_bytes == 0 or mem.free_bytes < self.min_free_bytes:
            self.min_free_bytes = mem.free_bytes

    def _release_resident_for_decode(mut self) raises:
        """Drop SD3 GPU state before VAE while preserving the host store."""
        if self.loaded:
            print("[sd3] releasing SD3 gate + GPU staging; preserving host store")
        self.ctx.synchronize()
        if len(self.loader) == 1:
            self.loader[0][].discard_fp8h_device_staging()
        self.gate = List[ArcPointer[SD3MMDiTPreBlockGate]]()
        self.lora = List[ArcPointer[Sd3LokrOverlay]]()
        self.loaded = False
        self.ctx.synchronize()
        cu_mempool_trim_current(0)
        self.ctx.synchronize()
        _print_vram("after resident release before VAE")

    def _write_result_manifest(mut self, png_path: String) raises -> String:
        self._record_vram()
        var manifest_path = png_path + String(".sd3_daemon_result.json")
        var denoise_per_step = Float64(0.0)
        if self.params.steps > 0:
            denoise_per_step = self.denoise_seconds / Float64(self.params.steps)
        var total_wall_seconds = Float64(perf_counter_ns() - self.job_t0_ns) / 1.0e9
        var peak_mib = Float64(0.0)
        if self.total_vram_bytes > 0 and self.min_free_bytes > 0:
            peak_mib = peak_vram_mib(self.total_vram_bytes, self.min_free_bytes)

        var content = String("{\n")
        content += String('  "schema":"serenity.sd3.daemon_result.v1",\n')
        content += String('  "backend":"sd3_daemon",\n')
        content += String('  "model":"') + json_escape(self.model_name()) + String('",\n')
        content += String('  "checkpoint_path":"') + json_escape(self._checkpoint_path()) + String('",\n')
        content += String('  "readiness_label":"experimental",\n')
        content += String('  "accepted_sampler_parity":false,\n')
        content += String('  "accepted_speed_parity":false,\n')
        content += String('  "run_identity":{\n')
        content += String('    "job_id":"') + json_escape(self.params.job_id) + String('",\n')
        content += String('    "prompt":"') + json_escape(self.params.prompt) + String('",\n')
        content += String('    "negative":"') + json_escape(self.params.negative) + String('",\n')
        content += String('    "seed":') + String(self.params.seed) + String(",\n")
        content += String('    "resolution":{"width":') + String(self.params.width) + String(',"height":') + String(self.params.height) + String("},\n")
        content += String('    "steps":') + String(self.params.steps) + String(",\n")
        content += String('    "guidance":') + String(self.params.cfg) + String(",\n")
        content += String('    "sampler_registry_backend":"sd3",\n')
        content += String('    "requested_sampler":"') + json_escape(self.params.sampler) + String('",\n')
        content += String('    "requested_scheduler":"') + json_escape(self.params.scheduler) + String('",\n')
        content += String('    "executed_sampler":"') + json_escape(
            self.executed_sampler
        ) + String('",\n')
        content += String('    "executed_scheduler":"sd3_simple_flowmatch",\n')
        content += String('    "schedule_source":"sd3_large_shifted_flowmatch",\n')
        content += String('    "host_resident_blocks":') + String(DEPTH) + String(",\n")
        content += String('    "variation_seed":') + String(self.params.variation_seed) + String(",\n")
        content += String('    "variation_strength":') + String(self.params.variation_strength) + String(",\n")
        content += String('    "variation_applied":') + json_bool(self.params.variation_strength > 0.0) + String(",\n")
        content += String('    "released_resident_mmdit_before_vae":true,\n')
        content += String('    "vae_decode_tile_grid":"5x5_lowmem",\n')
        content += String('    "image_index":') + String(self.params.image_index) + String(",\n")
        content += String('    "image_count":') + String(self.params.image_count) + String(",\n")
        content += String('    "lora_count":') + String(len(self.params.loras)) + String(",\n")
        if len(self.params.loras) == 1:
            content += String('    "loaded_lora":"') + json_escape(
                self.params.loras[0].name
            ) + String('",\n')
            content += String('    "loaded_lora_weight":') + String(
                self.params.loras[0].weight
            ) + String(",\n")
        else:
            content += String('    "loaded_lora":"",\n')
            content += String('    "loaded_lora_weight":0,\n')
        content += String('    "lora_target_count":') + String(
            self.lora_target_count
        ) + String(",\n")
        content += String('    "sampler_trace":{"algorithm":"') + json_escape(
            self.executed_sampler
        ) + String('","history_capacity":1,"history_final_len":')
        content += String(self.dpmpp_history.len()) + String(
            ',"dpmpp_update_steps":'
        ) + String(self.dpmpp_update_steps) + String(
            ',"dpmpp_second_order_steps":'
        ) + String(self.dpmpp_second_order_steps) + String("},\n")
        content += String('    "dtype":"bf16_mmdit_bf16_latent"\n')
        content += String("  },\n")
        content += String('  "mojo":{\n')
        content += String('    "load_seconds":') + String(self.load_seconds) + String(",\n")
        content += String('    "text_encode_seconds":') + String(self.text_encode_seconds) + String(",\n")
        content += String('    "prepare_seconds":') + String(self.prepare_seconds) + String(",\n")
        content += String('    "denoise_seconds":') + String(self.denoise_seconds) + String(",\n")
        content += String('    "denoise_seconds_per_step":') + String(denoise_per_step) + String(",\n")
        content += String('    "vae_decode_seconds":') + String(self.vae_decode_seconds) + String(",\n")
        content += String('    "total_wall_seconds":') + String(total_wall_seconds) + String(",\n")
        content += String('    "peak_vram_mib":') + String(peak_mib) + String(",\n")
        content += String('    "artifact_paths":["') + json_escape(png_path) + String('","') + json_escape(manifest_path) + String('"]\n')
        content += String("  },\n")
        content += String('  "output_png":"') + json_escape(png_path) + String('",\n')
        content += String('  "note":"Rust-server Mojo worker product-path result; SD3 blocks stage from a complete FP8 host-resident store with no denoise checkpoint I/O. Speed parity remains unaccepted until paired baseline evidence exists."\n')
        content += String("}\n")
        write_text_file(manifest_path, content)
        return manifest_path

    # ── per-job prep ───────────────────────────────────────────────────────────
    def _encode_in_process(mut self) raises:
        """Runtime CLIP-L+G+T5 encode of params.prompt AND params.negative into the
        SD3 context [1,410,4096] / pooled [1,2048] (encoders loaded then freed)."""
        _print_vram("before CLIP-L+G + T5 load")
        var clip_l = ClipEncoder.load(String(CLIP_L_PATH), ClipConfig.clip_l(), self.ctx)
        var clip_g = ClipEncoder.load(String(CLIP_G_PATH), ClipConfig.clip_g(), self.ctx)
        # CLIP-G text_projection.weight lives OUTSIDE text_model.* so ClipEncoder.load
        # skips it; load it directly from the CLIP-G safetensors. [1280,1280].
        var g_st = ShardedSafeTensors.open(String(CLIP_G_PATH))
        var text_proj_g = Tensor.from_view(g_st.tensor_view(String(CLIP_G_TEXT_PROJ)), self.ctx)
        var t5 = T5Encoder[T5_LEN].load(String(T5_PATH), T5Config.t5_xxl(), self.ctx)
        var clip_l_tok = ClipTokenizer(String(CLIP_L_TOK))
        var clip_g_tok = ClipTokenizer(String(CLIP_G_TOK))
        var t5_tok = T5Tokenizer(String(T5_TOK))

        var pos = _assemble_one(
            self.params.prompt, clip_l, clip_g, text_proj_g, t5,
            clip_l_tok, clip_g_tok, t5_tok, self.ctx,
        )
        var neg = _assemble_one(
            self.params.negative, clip_l, clip_g, text_proj_g, t5,
            clip_l_tok, clip_g_tok, t5_tok, self.ctx,
        )
        # Tensor is Movable-not-Copyable and a tuple subscript yields a BORROW, so
        # materialize each owned conditioning tensor via the proven .clone(ctx) idiom
        # before the encoders drop at scope exit.
        var caps = Sd3Caps(
            pos[0].clone(self.ctx), neg[0].clone(self.ctx),
            pos[1].clone(self.ctx), neg[1].clone(self.ctx),
        )
        # clip_l/clip_g/t5/text_proj_g drop only when this function returns.
        # The caller trims the CUDA pool after that scope exit; trimming here
        # would retain all encoder allocations and leaves no first-block headroom.
        _print_vram("conditioning assembled (encoder scope still live)")
        self.caps = List[ArcPointer[Sd3Caps]]()
        self.caps.append(ArcPointer(caps^))

    def _encode(mut self) raises:
        """Encode in a short-lived Mojo process, then load the four BF16 tensors.

        Process exit is the measured reclamation boundary on this runtime. It
        keeps the proven `_assemble_one` math while ensuring CLIP/T5 allocations
        are gone before the SD3.5 MMDiT worker loads its first block.
        """
        var stem = self.params.out_dir + "/." + self.params.job_id
        var prompt_path = stem + ".sd3_prompt.txt"
        var negative_path = stem + ".sd3_negative.txt"
        var cache_path = stem + ".sd3_conditioning.safetensors"
        write_text_file(prompt_path, self.params.prompt)
        write_text_file(negative_path, self.params.negative)
        var cmd = String(CONDITIONING_PRODUCER) + " '" + prompt_path + "' '"
        cmd += negative_path + "' '" + cache_path + "'"
        var rc = _shell(cmd)
        if rc != 0:
            raise Error(
                String("sd3: conditioning producer failed with status ") + String(rc)
            )
        var st = ShardedSafeTensors.open(cache_path)
        var caps = Sd3Caps(
            Tensor.from_view_as_bf16(st.tensor_view(String("context")), self.ctx),
            Tensor.from_view_as_bf16(st.tensor_view(String("context_uncond")), self.ctx),
            Tensor.from_view_as_bf16(st.tensor_view(String("pooled")), self.ctx),
            Tensor.from_view_as_bf16(st.tensor_view(String("pooled_uncond")), self.ctx),
        )
        self.caps = List[ArcPointer[Sd3Caps]]()
        self.caps.append(ArcPointer(caps^))
        _print_vram("after process-isolated conditioning load")

    def _load_model_shape[LH_: Int, LW_: Int](mut self) raises:
        """Load the SD3.5 Large gate + complete FP8 host block store."""
        if self.loaded:
            if self.gate[0][].latent_h == LH_ and self.gate[0][].latent_w == LW_:
                return
            print("[sd3] admitted shape changed; reloading rectangular pre/post gate")
            self.gate = List[ArcPointer[SD3MMDiTPreBlockGate]]()
            self.loaded = False
        _print_vram("before SD3 gate + host-store load")
        print("[sd3] loading SD3.5 Large pre/post gate", LH_, "x", LW_,
              "(resident) + FP8 host store for", DEPTH, "blocks")
        self.gate = List[ArcPointer[SD3MMDiTPreBlockGate]]()
        var checkpoint_path = self._checkpoint_path()
        self.gate.append(ArcPointer(
            SD3MMDiTPreBlockGate.load_large_shape_from_path[LH_, LW_](
                checkpoint_path.copy(), self.ctx
            )
        ))
        if len(self.loader) == 0:
            var plan = build_sd35_large_block_plan()
            var loader = TurboPlannedLoader.open(
                checkpoint_path.copy(),
                plan^,
                OffloadConfig.synchronous_single(),
                self.ctx,
                fill_block_store=False,
            )
            var host_blocks = loader.pin_residents_fp8_host(1 << 60, self.ctx)
            loader.require_all_blocks_memory_resident()
            loader.release_checkpoint_pages()
            loader.discard_unused_raw_streaming_slots(self.ctx)
            loader.set_fp8h_overlap(True)
            print("[sd3] host-resident FP8 denoiser blocks:", host_blocks, "/", DEPTH)
            self.loader.append(ArcPointer(loader^))
            self.loader_checkpoint = checkpoint_path.copy()
        else:
            self.loader[0][].require_all_blocks_memory_resident()
            print("[sd3] reusing complete host-resident denoiser store:", DEPTH, "/", DEPTH)
        self.lora = List[ArcPointer[Sd3LokrOverlay]]()
        self.lora_target_count = 0
        if len(self.params.loras) == 1:
            print(
                "[sd3] loading creator-compatible LoKr:",
                self.params.loras[0].name,
                "weight",
                self.params.loras[0].weight,
            )
            var overlay = load_sd3_large_lokr(
                self.params.loras[0].name,
                DEPTH,
                HIDDEN,
                Float32(self.params.loras[0].weight),
                self.ctx,
            )
            self.lora_target_count = overlay.count()
            print("[sd3] loaded", self.lora_target_count, "LoKr attention targets")
            self.lora.append(ArcPointer(overlay^))
        self.loaded = True
        _print_vram("after SD3 gate + host-store load (resident)")

    def _load_model(mut self) raises:
        comptime for bi in range(DEFAULT_ASPECT_LADDER_LEN):
            comptime X100_BI = DEFAULT_ASPECT_LADDER_X100[bi]
            comptime LH_BI = aspect_lat_h_units(X100_BI, SD3_PRODUCT_EDGE_UNITS)
            comptime LW_BI = aspect_lat_w_units(X100_BI, SD3_PRODUCT_EDGE_UNITS)
            if self.params.width == LW_BI * 8 and self.params.height == LH_BI * 8:
                self._load_model_shape[LH_BI, LW_BI]()
                return
        raise Error("sd3: admitted load shape was not compiled")

    def _prepare_job_shape[LH_: Int, LW_: Int](mut self) raises:
        """Flow-match scheduler (honors steps + large shift) + seeded BF16 latent."""
        self.sched = List[ArcPointer[SD3FlowMatchScheduler]]()
        self.sched.append(
            ArcPointer(SD3FlowMatchScheduler(self.params.steps, sd3_large_schedule_shift()))
        )
        var nsh = [1, LC, LH_, LW_]
        var noise = randn(nsh.copy(), UInt64(self.params.seed), STDtype.BF16, self.ctx)
        if self.params.variation_strength > 0.0:
            var vnoise = randn(
                nsh.copy(),
                UInt64(self.params.variation_seed + self.params.image_index),
                STDtype.BF16,
                self.ctx,
            )
            var base_h = noise.to_host(self.ctx)
            var var_h = vnoise.to_host(self.ctx)
            var blended = variation_noise_chw(
                base_h, var_h, LC, LH_, LW_, self.params.variation_strength
            )
            noise = Tensor.from_host(blended, nsh.copy(), STDtype.BF16, self.ctx)
        self.latent = List[ArcPointer[Tensor]]()
        self.latent.append(ArcPointer(noise^))
        print(
            "[sd3] job", self.params.job_id, ":", self.params.steps,
            "steps, cfg", self.cfg, "seed", self.params.seed,
            "size", self.params.width, "x", self.params.height,
        )

    def _prepare_job(mut self) raises:
        comptime for bi in range(DEFAULT_ASPECT_LADDER_LEN):
            comptime X100_BI = DEFAULT_ASPECT_LADDER_X100[bi]
            comptime LH_BI = aspect_lat_h_units(X100_BI, SD3_PRODUCT_EDGE_UNITS)
            comptime LW_BI = aspect_lat_w_units(X100_BI, SD3_PRODUCT_EDGE_UNITS)
            if self.params.width == LW_BI * 8 and self.params.height == LH_BI * 8:
                self._prepare_job_shape[LH_BI, LW_BI]()
                return
        raise Error("sd3: admitted prepare shape was not compiled")

    # ── one denoise step (CFG dual streamed forward + flow-match Euler) ──────────
    def _denoise_one_shape[LH_: Int, LW_: Int, N_IMG_: Int, S_JOINT_: Int](
        mut self
    ) raises:
        var i = self.cur
        var sigma = self.sched[0][].timestep(i)
        var dt = self.sched[0][].dt(i)
        var v_cond = _sd3_large_forward[LH_, LW_, N_IMG_, S_JOINT_](
            self.latent[0][], sigma, self.caps[0][].context, self.caps[0][].pooled,
            self.gate[0][], self.loader[0][], self.ctx,
            (
                Optional[ArcPointer[Sd3LokrOverlay]](self.lora[0])
                if len(self.lora) == 1
                else Optional[ArcPointer[Sd3LokrOverlay]]()
            ),
        )
        var v_uncond = _sd3_large_forward[LH_, LW_, N_IMG_, S_JOINT_](
            self.latent[0][], sigma, self.caps[0][].context_uncond,
            self.caps[0][].pooled_uncond, self.gate[0][], self.loader[0][], self.ctx,
            (
                Optional[ArcPointer[Sd3LokrOverlay]](self.lora[0])
                if len(self.lora) == 1
                else Optional[ArcPointer[Sd3LokrOverlay]]()
            ),
        )
        var velocity = sd3_cfg(v_cond, v_uncond, self.cfg, self.ctx)
        var x_new: Tensor
        if self.executed_sampler == "dpmpp_2m":
            var sigma_next = sigma + dt
            var latent_f32 = cast_tensor(
                self.latent[0][], STDtype.F32, self.ctx
            )
            var velocity_f32 = cast_tensor(velocity, STDtype.F32, self.ctx)
            var denoised = denoised_from_velocity(
                latent_f32, velocity_f32, sigma, self.ctx
            )
            if not self.dpmpp_history.is_empty():
                self.dpmpp_second_order_steps += 1
            var stepped = dpmpp_2m_step(
                latent_f32,
                denoised,
                sigma,
                sigma_next,
                self.dpmpp_history,
                self.ctx,
            )
            self.dpmpp_history.push(
                denoised^, lambda_from_sigma_f64(Float64(sigma))
            )
            self.dpmpp_update_steps += 1
            x_new = cast_tensor(stepped, STDtype.BF16, self.ctx)
        else:
            x_new = sd3_euler_step(
                self.latent[0][], velocity, dt, self.ctx
            )
        self.latent = List[ArcPointer[Tensor]]()
        self.latent.append(ArcPointer(x_new^))

    def _denoise_one(mut self) raises:
        comptime for bi in range(DEFAULT_ASPECT_LADDER_LEN):
            comptime X100_BI = DEFAULT_ASPECT_LADDER_X100[bi]
            comptime LH_BI = aspect_lat_h_units(X100_BI, SD3_PRODUCT_EDGE_UNITS)
            comptime LW_BI = aspect_lat_w_units(X100_BI, SD3_PRODUCT_EDGE_UNITS)
            comptime N_IMG_BI = (LH_BI // 2) * (LW_BI // 2)
            comptime S_JOINT_BI = N_CTX + N_IMG_BI
            if self.params.width == LW_BI * 8 and self.params.height == LH_BI * 8:
                self._denoise_one_shape[LH_BI, LW_BI, N_IMG_BI, S_JOINT_BI]()
                return
        raise Error("sd3: admitted denoise shape was not compiled")

    # ── final decode + PNG(tEXt) ──────────────────────────────────────────────
    def _decode_and_save_shape[LH_: Int, LW_: Int](mut self) raises -> String:
        var png_path = self.params.out_dir + "/" + self.params.job_id + ".png"
        var latent = self.latent[0][].clone(self.ctx)
        # Per-job conditioning is dead weight at decode; free before the decoder.
        self.caps = List[ArcPointer[Sd3Caps]]()
        self.sched = List[ArcPointer[SD3FlowMatchScheduler]]()
        self.latent = List[ArcPointer[Tensor]]()
        # Release GPU gate/staging before VAE while retaining the pinned-host
        # store. A later job reloads only the much smaller gate.
        self._release_resident_for_decode()
        # Prefer whole-image decode when VRAM allows; tiled degrades output (MJ-1054).
        var mem = cu_mem_get_info()
        var free_gib = Float64(mem.free_bytes) / 1073741824.0
        if mem.free_bytes > WHOLE_DECODE_MIN_FREE_BYTES:
            print("[sd3] WHOLE-image decode (free=", free_gib,
                  "GiB) — tiled measured to degrade output (MJ-1054)")
            var dec = load_sd3_embedded_ldm_decoder[LH_, LW_](
                self._checkpoint_path(), self.ctx
            )
            var wimg = dec.decode(latent, self.ctx)
            _save_rgb_png_with_text(wimg, png_path, self.params.params_json, self.ctx)
            return png_path
        print("[sd3] tiled VAE decode FALLBACK (5x5 lowmem overlap+blend) — free=",
              free_gib, "GiB below whole-image threshold; tiled degrades output (MJ-1054)")
        var img = sd3_tiled_decode_5x5_lowmem[LH_, LW_](
            latent, self._checkpoint_path(), self.ctx
        )
        _save_rgb_png_with_text(img, png_path, self.params.params_json, self.ctx)
        return png_path

    def _decode_and_save(mut self) raises -> String:
        comptime for bi in range(DEFAULT_ASPECT_LADDER_LEN):
            comptime X100_BI = DEFAULT_ASPECT_LADDER_X100[bi]
            comptime LH_BI = aspect_lat_h_units(X100_BI, SD3_PRODUCT_EDGE_UNITS)
            comptime LW_BI = aspect_lat_w_units(X100_BI, SD3_PRODUCT_EDGE_UNITS)
            if self.params.width == LW_BI * 8 and self.params.height == LH_BI * 8:
                return self._decode_and_save_shape[LH_BI, LW_BI]()
        raise Error("sd3: admitted decode shape was not compiled")

    def _clear_job(mut self):
        self.active = False
        self.phase = S3PHASE_IDLE
        self.cur = 0
        self.cancel_flag = False
        self.announced = False
        self.caps = List[ArcPointer[Sd3Caps]]()
        self.sched = List[ArcPointer[SD3FlowMatchScheduler]]()
        self.latent = List[ArcPointer[Tensor]]()

    # ── the pull-based tick ───────────────────────────────────────────────────
    def step(mut self) raises -> StepResult:
        var r = StepResult()
        r.total = self.params.steps
        if not self.active:
            r.failed = True
            r.error = String("no active job")
            return r^
        if self.cancel_flag:
            r.step = self.cur
            self._clear_job()
            r.cancelled = True
            return r^
        try:
            if self.phase == S3PHASE_ENCODE:
                if not self.announced:
                    # announce BEFORE the long blocking encode tick (per-job
                    # CLIP-L + CLIP-G + T5-XXL load + dual-prompt forward).
                    self.announced = True
                    r.step = 0
                    r.phase = String("encoding")
                    return r^
                var encode_t0 = perf_counter_ns()
                self._encode()
                # `_encode` owns ~11 GiB of temporary CLIP/T5 tensors. They are
                # destructed at the return boundary above, but the CUDA async
                # pool retains those freed pages until explicitly trimmed.
                self.ctx.synchronize()
                cu_mempool_trim_current(0)
                self.ctx.synchronize()
                _print_vram("after encoder scope-exit trim")
                self.text_encode_seconds = Float64(perf_counter_ns() - encode_t0) / 1.0e9
                self._record_vram()
                self.announced = False
                self.phase = S3PHASE_LOAD
                r.step = 0
                return r^
            if self.phase == S3PHASE_LOAD:
                var load_t0 = perf_counter_ns()
                if not self.loaded:
                    if not self.announced:
                        self.announced = True
                        r.step = 0
                        r.phase = String("loading")
                        return r^
                    self._load_model()
                self.load_seconds += Float64(perf_counter_ns() - load_t0) / 1.0e9
                self.announced = False
                var prep_t0 = perf_counter_ns()
                self._prepare_job()
                self.prepare_seconds += Float64(perf_counter_ns() - prep_t0) / 1.0e9
                self._record_vram()
                self.phase = S3PHASE_DENOISE
                r.step = 0
                r.phase = String("sampling")
                return r^
            if self.phase == S3PHASE_DENOISE:
                var denoise_t0 = perf_counter_ns()
                self._denoise_one()
                self.denoise_seconds += Float64(perf_counter_ns() - denoise_t0) / 1.0e9
                self._record_vram()
                self.cur += 1
                r.step = self.cur
                r.phase = String("sampling")
                if self.cur >= self.params.steps:
                    self.phase = S3PHASE_DECODE
                return r^
            if not self.announced:
                # announce BEFORE the long blocking VAE-decode tick.
                self.announced = True
                r.step = self.params.steps
                r.phase = String("decoding")
                return r^
            var decode_t0 = perf_counter_ns()
            var path = self._decode_and_save()
            self.vae_decode_seconds = Float64(perf_counter_ns() - decode_t0) / 1.0e9
            self._record_vram()
            var manifest = self._write_result_manifest(path)
            print("[sd3][manifest] saved:", manifest)
            r.step = self.params.steps
            self._clear_job()
            r.done = True
            r.output_path = path
            return r^
        except e:
            self._clear_job()
            r.failed = True
            r.error = String(e)
            return r^
