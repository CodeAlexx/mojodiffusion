# serenitymojo.serve.sdxl_backend — the real SDXL 1024x1024 GenBackend.
#
# Wraps the VERIFIED SDXL inference stages (serenitymojo/pipeline/sdxl_sample_cli.mojo
# + models/text_encoder/clip_encoder.mojo + sampling/sdxl_euler.mojo + models/vae/
# ldm_decoder.mojo) behind the pull-based GenBackend seam (backend.mojo). Unlike the
# sample CLI — which loads a PRE-CACHED CLIP-embedding sidecar — THIS backend encodes
# the REAL params.prompt + params.negative at runtime through the verified CLIP-L +
# CLIP-G modules, exactly mirroring inference-flame's sdxl_encode.rs assembly:
#
#   context        = cat([clip_l_hidden [1,77,768], clip_g_hidden [1,77,1280]], dim=2)
#                    -> [1,77,2048]                         (cross-attention context)
#   context_uncond = same for the negative prompt          -> [1,77,2048]
#   y              = cat([clip_g_text_embeds [1,1280],
#                         sin_embed_256([h,w,0,0,h,w]) [1,1536]], dim=1)
#                    -> [1,2816] (SerenityTrainer/diffusers ADM contract)
#   y_uncond       = same for the negative prompt          -> [1,2816]
#
# The denoise (30-step epsilon-prediction Euler +
# CFG) and VAE decode reuse sdxl_sample_cli's exact math (SDXLUNet[LH,LW].forward,
# SDXLEulerScheduler, sdxl_cfg/sdxl_euler_step/sdxl_input_scale, load_sdxl_ldm_decoder).
#
# Residency model (single-GPU):
#   * The SDXL UNet (~5 GB BF16) is loaded ONCE (first job) and STAYS RESIDENT across
#     jobs (the residency win — like Qwen-Image's offloader handle, but here the whole
#     UNet fits, so the weights themselves stay resident).
#   * The CLIP-L (~250 MB) + CLIP-G (~1.4 GB) encoders are loaded → used → freed PER
#     JOB inside the ENCODE step (Movable-not-Copyable Tensors drop at scope exit).
#   * The VAE decoder (~330 MB F32) is loaded PER JOB inside the DECODE step and freed.
#
# step() state machine: ENCODE (per-job, blocking — announced phase="encoding")
#   → LOAD (UNet, once, announced phase="loading") → DENOISE×steps (one CFG dual-
#   forward + Euler update per tick) → DECODE (announced phase="decoding") → done.
#   cancel() makes the next step() return cancelled and frees all per-job tensors.
#
# Size support: the finite seven-shape 1024-area product ladder. Every arm is
# comptime-specialized at its exact latent H/W; steps/cfg/seed stay runtime.
#
# SDXL kohya LoRAs are merged sequentially at UNet load, including creator-style
# convolution adapters. Img2img remains fail-loud until its conditioning path is
# implemented.

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

from serenitymojo.tokenizer.clip_tokenizer import ClipTokenizer
from serenitymojo.models.text_encoder.clip_encoder import ClipEncoder, ClipConfig
from serenitymojo.models.dit.sdxl_unet import SDXLUNet
from serenitymojo.models.sdxl.conditioning import sdxl_adm_y
from serenitymojo.models.vae.ldm_decoder import load_sdxl_ldm_decoder
from serenitymojo.models.vae.sdxl_tiled_decode import sdxl_tiled_decode
from serenitymojo.registry.checkpoints import default_manifest_by_id
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.linear import linear
from serenitymojo.ops.random import randn
from serenitymojo.ops.tensor_algebra import mul_scalar, concat
from serenitymojo.sampling.sdxl_euler import (
    SDXLEulerScheduler,
    sdxl_cfg,
    sdxl_denoised_from_eps,
    sdxl_dpmpp_2m_step,
    sdxl_euler_step,
    sdxl_initial_noise_sigma,
    sdxl_input_scale,
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
from serenitymojo.serve.sdxl_decode_subprocess import decode_tiled_subprocess


comptime GENPARAMS_TEXT_KEY = "serenity.genparams.v1"

# ── explicit compiled product shapes ─────────────────────────────────────────
comptime LH_SQUARE = 128
comptime LW_SQUARE = 128
comptime LH_1152X896 = 112
comptime LW_1152X896 = 144
comptime LH_896X1152 = 144
comptime LW_896X1152 = 112
comptime LH_LANDSCAPE = 96
comptime LW_LANDSCAPE = 168
comptime LH_PORTRAIT = 168
comptime LW_PORTRAIT = 96
comptime LH_1280X832 = 104
comptime LW_1280X832 = 160
comptime LH_832X1280 = 160
comptime LW_832X1280 = 104
comptime CLIP_LEN = 77

# ── verified model + tokenizer paths (match sdxl_sample_cli's manifest + the
#    inference-flame sdxl_encode.rs CLIP defaults) ──
comptime CLIP_L_PATH = "models/text-encoders/clip_l.safetensors"
comptime CLIP_G_PATH = "models/text-encoders/clip_g.safetensors"
comptime CLIP_L_TOK = "models/text-encoders/clip_l.tokenizer.json"
comptime CLIP_G_TOK = "models/text-encoders/clip_g.tokenizer.json"
comptime CLIP_G_TEXT_PROJ = "text_projection.weight"

comptime CLIP_PAD_ID = 49407   # CLIP eos == pad
comptime CLIP_EOS_ID = 49407


comptime SPHASE_IDLE = 0
comptime SPHASE_ENCODE = 1
comptime SPHASE_LOAD = 2
comptime SPHASE_DENOISE = 3
comptime SPHASE_DECODE = 4

# Whole-image VAE decode is preferred when it fits: tiled decode is MEASURED to
# degrade output (MJ-1054). After the ~6 GB UNet free + mempool trim below we
# query free VRAM and decode whole only when it clears this bar, else fall back
# to the (degrading) 3x3 tiled path. The whole 1024^2 SDXL decode is lighter than
# the other models (LATENT_CH=4); 10 GiB is a conservative estimate to be
# tightened by measurement.
comptime WHOLE_DECODE_MIN_FREE_BYTES = 22 * 1024 * 1024 * 1024  # 22 GiB
# The 1024^2 SDXL whole-image VAE decode OOM'd (CUDA_ERROR_OUT_OF_MEMORY) at 15.78
# GiB free through :7811 (2026-07-12): `_decode_and_save` frees the ~6 GB UNet +
# cu_mempool_trim_current(0) BEFORE decode, but MAX's pool retains it (trim reclaimed
# 0 MiB), so free stays ~15.78 GiB and the whole-image decode needs MORE than that.
# The old 10 GiB bar let it pick whole-image and OOM. Raise the bar above the card's
# realistic free so it takes the tiled 3x3 fallback (which succeeds even at ~2 GiB
# free, cf. klein/sd3) unless the card is nearly empty. tiled degrades slightly
# (MJ-1054) but RENDERS — strictly better than OOM.


def _sdxl_sampler_name(name: String) -> String:
    var normalized = String(name.lower())
    if normalized == String(""):
        return String("euler")
    if normalized == String("dpm++ 2m") or normalized == String("dpmpp 2m"):
        return String("dpmpp_2m")
    return normalized^


def _sdxl_scheduler_name(name: String) -> String:
    var normalized = String(name.lower())
    return String("normal") if normalized == String("") else normalized^


def _sdxl_executed_sampler(name: String) -> String:
    var normalized = _sdxl_sampler_name(name)
    if normalized == String("dpmpp_2m"):
        return String("sdxl_dpmpp_2m")
    if normalized == String("ddim"):
        # This is a genuine Comfy alias: sampler_object("ddim") dispatches Euler.
        return String("sdxl_euler_ddim_alias")
    return String("sdxl_euler")


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
        String("echo -n '[sdxl][vram] ") + tag
        + ": ' && nvidia-smi --query-gpu=memory.used --format=csv,noheader"
    )


# ── CLIP conditioning bundle (per job) ──────────────────────────────────────
struct SdxlCaps(Movable):
    var context: Tensor         # [1,77,2048] BF16 (cond)
    var context_uncond: Tensor  # [1,77,2048] BF16 (uncond)
    var y: Tensor               # [1,2816]    BF16 (cond)
    var y_uncond: Tensor        # [1,2816]    BF16 (uncond)

    def __init__(
        out self, var context: Tensor, var context_uncond: Tensor,
        var y: Tensor, var y_uncond: Tensor,
    ):
        self.context = context^
        self.context_uncond = context_uncond^
        self.y = y^
        self.y_uncond = y_uncond^


# ── helpers ─────────────────────────────────────────────────────────────────
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
    encode() already wrapped with BOS(49406)+EOS(49407)."""
    if len(ids) > CLIP_LEN:
        var trimmed = List[Int]()
        for i in range(CLIP_LEN):
            trimmed.append(ids[i])
        trimmed[CLIP_LEN - 1] = CLIP_EOS_ID
        return trimmed^
    while len(ids) < CLIP_LEN:
        ids.append(CLIP_PAD_ID)
    return ids^


def _save_rgb_png_with_text(
    rgb: Tensor, path: String, params_json: String, ctx: DeviceContext
) raises:
    """[1,3,H,W] SIGNED float tensor → 8-bit RGB PNG with the job params in a
    serenity.genparams.v1 tEXt chunk. Quantization math == save_png's
    (_quantize, ValueRange.SIGNED); only the writer differs (tEXt support).
    Identical to qwenimage_backend._save_rgb_png_with_text."""
    var shape = rgb.shape()
    if len(shape) != 4 or shape[0] != 1 or shape[1] != 3:
        raise Error("sdxl_backend: expected [1,3,H,W] rgb tensor")
    var height = shape[2]
    var width = shape[3]
    var host = rgb.to_host(ctx)
    var plane = height * width
    if len(host) != 3 * plane:
        raise Error("sdxl_backend: rgb to_host size mismatch")
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


# ── CLIP-L+G runtime encode → SDXL context/y assembly (mirrors sdxl_encode.rs) ──
# Encodes ONE prompt string through both CLIP encoders (already-loaded, passed in)
# and returns (context [1,77,2048] BF16, y [1,2816] BF16). `text_proj` is CLIP-G's
# text_projection.weight [1280,1280]; clip_g_text_embeds = clip_g_pool @ text_projᵀ
# (HF convention: text_projection is [out,in], applied as a no-bias Linear — which
# is exactly what ops.linear does: y = x @ Wᵀ).
def _encode_one(
    text: String,
    clip_l: ClipEncoder,
    clip_g: ClipEncoder,
    text_proj: Tensor,
    clip_l_tok: ClipTokenizer,
    clip_g_tok: ClipTokenizer,
    height: Int,
    width: Int,
    ctx: DeviceContext,
) raises -> Tuple[Tensor, Tensor]:
    var l_ids = _fit_clip_ids(clip_l_tok.encode(text))
    var g_ids = _fit_clip_ids(clip_g_tok.encode(text))

    # CLIP-L: (last_hidden [1,77,768], pooled [1,768])
    var l_out = clip_l.encode_sdxl[CLIP_LEN](l_ids^, ctx, True)
    var l_hidden = _to_bf16(l_out[0], ctx)

    # CLIP-G: (last_hidden [1,77,1280], pooled_raw [1,1280])
    var g_out = clip_g.encode_sdxl[CLIP_LEN](g_ids^, ctx, True)
    var g_hidden = _to_bf16(g_out[0], ctx)
    # clip_g_text_embeds = clip_g_pool_raw @ text_projectionᵀ -> [1,1280]
    var g_pool = linear(g_out[1], text_proj, Optional[Tensor](None), ctx)
    g_pool = _to_bf16(g_pool, ctx)

    # context = cat([l_hidden, g_hidden], dim=2) -> [1,77,2048]
    var context = concat(2, ctx, l_hidden, g_hidden)
    context = _to_bf16(context, ctx)

    # SerenityTrainer/diffusers ADM: projected CLIP-G pool plus six 256-d
    # original/crop/target size embeddings. CLIP-L pooled is not part of y.
    var y = sdxl_adm_y(g_pool, height, width, ctx)
    y = _to_bf16(y, ctx)

    return (context^, y^)


struct SdxlBackend(GenBackend, Movable):
    var ctx: DeviceContext

    # ── resident across jobs (UNet weights, loaded once on first job) ──
    var loaded: Bool
    var loaded_checkpoint: String
    var loaded_lora_signature: String
    var model_width: Int
    var model_height: Int
    var model_square: List[ArcPointer[SDXLUNet[LH_SQUARE, LW_SQUARE]]]
    var model_1152x896: List[ArcPointer[SDXLUNet[LH_1152X896, LW_1152X896]]]
    var model_896x1152: List[ArcPointer[SDXLUNet[LH_896X1152, LW_896X1152]]]
    var model_landscape: List[ArcPointer[SDXLUNet[LH_LANDSCAPE, LW_LANDSCAPE]]]
    var model_portrait: List[ArcPointer[SDXLUNet[LH_PORTRAIT, LW_PORTRAIT]]]
    var model_1280x832: List[ArcPointer[SDXLUNet[LH_1280X832, LW_1280X832]]]
    var model_832x1280: List[ArcPointer[SDXLUNet[LH_832X1280, LW_832X1280]]]

    # One exact conditioning entry survives repeat jobs. SDXL ADM conditioning
    # depends on prompt text and target geometry, so both dimensions are part of
    # the key. The four BF16 tensors are well under 1 MiB combined.
    var cap_cache_prompt: String
    var cap_cache_negative: String
    var cap_cache_width: Int
    var cap_cache_height: Int
    var cap_cache: List[ArcPointer[SdxlCaps]]

    # ── per-job state (cleared on done/failed/cancelled) ──
    var active: Bool
    var cancel_flag: Bool
    var phase: Int
    var announced: Bool
    var cur: Int
    var params: JobParams
    var cfg: Float32
    var caps: List[ArcPointer[SdxlCaps]]            # 0/1
    var sched: List[ArcPointer[SDXLEulerScheduler]] # 0/1
    var latent: List[ArcPointer[Tensor]]            # 0/1 ([1,4,LH,LW] F32)
    var previous_denoised: List[ArcPointer[Tensor]] # 0/1, DPM++ 2M history
    var previous_sigma: Float32
    var job_t0_ns: UInt
    var load_seconds: Float64
    var text_encode_seconds: Float64
    var prepare_seconds: Float64
    var denoise_seconds: Float64
    var vae_decode_seconds: Float64
    var text_conditioning_cache_hit: Bool
    var total_vram_bytes: Int
    var min_free_bytes: Int

    def __init__(out self) raises:
        self.ctx = DeviceContext()
        self.loaded = False
        self.loaded_checkpoint = String("")
        self.loaded_lora_signature = String("")
        self.model_width = 0
        self.model_height = 0
        self.model_square = List[ArcPointer[SDXLUNet[LH_SQUARE, LW_SQUARE]]]()
        self.model_1152x896 = List[ArcPointer[SDXLUNet[LH_1152X896, LW_1152X896]]]()
        self.model_896x1152 = List[ArcPointer[SDXLUNet[LH_896X1152, LW_896X1152]]]()
        self.model_landscape = List[ArcPointer[SDXLUNet[LH_LANDSCAPE, LW_LANDSCAPE]]]()
        self.model_portrait = List[ArcPointer[SDXLUNet[LH_PORTRAIT, LW_PORTRAIT]]]()
        self.model_1280x832 = List[ArcPointer[SDXLUNet[LH_1280X832, LW_1280X832]]]()
        self.model_832x1280 = List[ArcPointer[SDXLUNet[LH_832X1280, LW_832X1280]]]()
        self.cap_cache_prompt = String("")
        self.cap_cache_negative = String("")
        self.cap_cache_width = 0
        self.cap_cache_height = 0
        self.cap_cache = List[ArcPointer[SdxlCaps]]()
        self.active = False
        self.cancel_flag = False
        self.phase = SPHASE_IDLE
        self.announced = False
        self.cur = 0
        self.params = JobParams()
        self.cfg = Float32(7.5)
        self.caps = List[ArcPointer[SdxlCaps]]()
        self.sched = List[ArcPointer[SDXLEulerScheduler]]()
        self.latent = List[ArcPointer[Tensor]]()
        self.previous_denoised = List[ArcPointer[Tensor]]()
        self.previous_sigma = 0.0
        self.job_t0_ns = UInt(0)
        self.load_seconds = 0.0
        self.text_encode_seconds = 0.0
        self.prepare_seconds = 0.0
        self.denoise_seconds = 0.0
        self.vae_decode_seconds = 0.0
        self.text_conditioning_cache_hit = False
        self.total_vram_bytes = 0
        self.min_free_bytes = 0

    def backend_name(self) -> String:
        return String("sdxl")

    def model_name(self) -> String:
        return String("SDXL")

    def resident_model(self) -> String:
        """Best-effort match to a /v1/models scan entry for the resident UNet
        (the flat .serenity/models/checkpoints/sdxl_unet_bf16.safetensors
        checkpoint)."""
        return self.params.model.copy() if self.loaded else String("")

    # ── job admission ─────────────────────────────────────────────────────────
    def start(mut self, params: JobParams) raises:
        if self.active:
            raise Error("SdxlBackend.start: a job is already running")
        reject_unsupported_common_runtime_params(params, String("sdxl"))
        reject_unsupported_reference_image_params(params, String("sdxl"))
        reject_unsupported_inpaint_conditioning_params(params, String("sdxl"))
        reject_unsupported_qwen_edit_conditioning_params(params, String("sdxl"))
        reject_unsupported_conditioning_mask_params(params, String("sdxl"))
        reject_unsupported_mask_image_params(params, String("sdxl"))
        reject_unsupported_lanpaint_params(params, String("sdxl"))
        var sampler_admission = sampler_admission_for_backend(String("sdxl"), params.sampler)
        if not sampler_admission.supported:
            raise Error(
                String("sdxl: unsupported sampler '") + params.sampler
                + String("'; ") + sampler_admission.reason
            )
        var scheduler_admission = scheduler_admission_for_backend(String("sdxl"), params.scheduler)
        if not scheduler_admission.supported:
            raise Error(
                String("sdxl: unsupported scheduler '") + params.scheduler
                + String("'; ") + scheduler_admission.reason
            )
        if not (
            (params.width == 1024 and params.height == 1024)
            or (params.width == 1152 and params.height == 896)
            or (params.width == 896 and params.height == 1152)
            or (params.width == 1344 and params.height == 768)
            or (params.width == 768 and params.height == 1344)
            or (params.width == 1280 and params.height == 832)
            or (params.width == 832 and params.height == 1280)
        ):
            raise Error(
                String("sdxl: unsupported size ") + String(params.width)
                + "x" + String(params.height)
                + " — supported compiled shapes are 1024x1024, 1152x896,"
                + " 896x1152, 1344x768, 768x1344, 1280x832, and 832x1280"
            )
        if params.init_image.byte_length() > 0:
            raise Error(
                "sdxl: img2img is not supported for SDXL yet;"
                " submit without an init image"
            )
        # Warn-loud (never silently drop) on any advanced-sampling knob set but
        # unsupported by this fixed Euler path.
        warn_unsupported_advanced_sampling_params(params, String("sdxl"), List[String]())
        self.params = params.copy()
        self.cfg = Float32(params.cfg)
        self.active = True
        self.cancel_flag = False
        self.cur = 0
        self.announced = False
        self.phase = SPHASE_ENCODE
        self.job_t0_ns = perf_counter_ns()
        self.load_seconds = 0.0
        self.text_encode_seconds = 0.0
        self.prepare_seconds = 0.0
        self.denoise_seconds = 0.0
        self.vae_decode_seconds = 0.0
        self.text_conditioning_cache_hit = False
        var mem = cu_mem_get_info()
        self.total_vram_bytes = mem.total_bytes
        self.min_free_bytes = mem.free_bytes
        self._record_vram()

    def cancel(mut self):
        self.cancel_flag = True

    def between_jobs_trim(mut self) raises:
        """Reclaim the per-job transient peak (CLIP-L+G encoders ~1.7 GB, the VAE
        decoder ~330 MB, 1024² forward + decode activations) back to the OS via
        cuMemPoolTrimTo. The resident UNet weights have live suballocations and are
        NOT reclaimed."""
        var before = cu_mem_get_info()
        self.ctx.synchronize()
        cu_mempool_trim_current(0)
        self.ctx.synchronize()
        var after = cu_mem_get_info()
        print("[sdxl] between-jobs trim: used",
              before.used_bytes() // (1024 * 1024), "->",
              after.used_bytes() // (1024 * 1024), "MiB (reclaimed",
              (before.used_bytes() - after.used_bytes()) // (1024 * 1024), "MiB)")

    def _record_vram(mut self) raises:
        var mem = cu_mem_get_info()
        if self.total_vram_bytes == 0:
            self.total_vram_bytes = mem.total_bytes
        if self.min_free_bytes == 0 or mem.free_bytes < self.min_free_bytes:
            self.min_free_bytes = mem.free_bytes

    def _write_result_manifest(mut self, png_path: String) raises -> String:
        self._record_vram()
        var manifest_path = png_path + String(".sdxl_daemon_result.json")
        var denoise_per_step = Float64(0.0)
        if self.params.steps > 0:
            denoise_per_step = self.denoise_seconds / Float64(self.params.steps)
        var total_wall_seconds = Float64(perf_counter_ns() - self.job_t0_ns) / 1.0e9
        var peak_mib = Float64(0.0)
        if self.total_vram_bytes > 0 and self.min_free_bytes > 0:
            peak_mib = peak_vram_mib(self.total_vram_bytes, self.min_free_bytes)

        var content = String("{\n")
        content += String('  "schema":"serenity.sdxl.daemon_result.v1",\n')
        content += String('  "backend":"sdxl_daemon",\n')
        content += String('  "model":"sdxl",\n')
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
        content += String('    "sampler_registry_backend":"sdxl",\n')
        content += String('    "requested_sampler":"') + json_escape(self.params.sampler) + String('",\n')
        content += String('    "requested_scheduler":"') + json_escape(self.params.scheduler) + String('",\n')
        content += String('    "executed_sampler":"') + json_escape(
            _sdxl_executed_sampler(self.params.sampler)
        ) + String('",\n')
        content += String('    "executed_scheduler":"') + json_escape(
            _sdxl_scheduler_name(self.params.scheduler)
        ) + String('",\n')
        content += String('    "variation_seed":') + String(self.params.variation_seed) + String(",\n")
        content += String('    "variation_strength":') + String(self.params.variation_strength) + String(",\n")
        content += String('    "variation_applied":') + json_bool(self.params.variation_strength > 0.0) + String(",\n")
        content += String('    "image_index":') + String(self.params.image_index) + String(",\n")
        content += String('    "image_count":') + String(self.params.image_count) + String(",\n")
        content += String('    "lora_count":') + String(len(self.params.loras)) + String(",\n")
        content += String('    "lora_signature":"') + json_escape(
            self._requested_lora_signature()
        ) + String('",\n')
        content += String('    "text_conditioning_cache_hit":') + json_bool(
            self.text_conditioning_cache_hit
        ) + String(",\n")
        content += String('    "dtype":"bf16_unet_f32_latent"\n')
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
        content += String('  "note":"Rust-server Mojo worker product-path result; timing and VRAM are measured in the backend process. Speed parity remains unaccepted until paired baseline evidence exists."\n')
        content += String("}\n")
        write_text_file(manifest_path, content)
        return manifest_path

    # ── per-job prep ───────────────────────────────────────────────────────────
    def _encode(mut self) raises:
        """Runtime CLIP-L+G encode of params.prompt AND params.negative into the
        SDXL context/y conditioning (encoders + text_projection loaded then freed)."""
        if (
            len(self.cap_cache) == 1
            and self.cap_cache_prompt == self.params.prompt
            and self.cap_cache_negative == self.params.negative
            and self.cap_cache_width == self.params.width
            and self.cap_cache_height == self.params.height
        ):
            ref cached_caps = self.cap_cache[0][]
            var caps = SdxlCaps(
                cached_caps.context.clone(self.ctx),
                cached_caps.context_uncond.clone(self.ctx),
                cached_caps.y.clone(self.ctx),
                cached_caps.y_uncond.clone(self.ctx),
            )
            self.caps = List[ArcPointer[SdxlCaps]]()
            self.caps.append(ArcPointer(caps^))
            self.ctx.synchronize()
            self.text_conditioning_cache_hit = True
            print("[sdxl] conditioning cache HIT (prompts and geometry unchanged)")
            return
        _print_vram("before CLIP-L+G load")
        var clip_l = ClipEncoder.load(String(CLIP_L_PATH), ClipConfig.clip_l(), self.ctx)
        var clip_g = ClipEncoder.load(String(CLIP_G_PATH), ClipConfig.clip_g(), self.ctx)
        # text_projection.weight lives OUTSIDE text_model.* so ClipEncoder.load skips
        # it; load it directly from the CLIP-G safetensors. [1280,1280] F16.
        var g_st = ShardedSafeTensors.open(String(CLIP_G_PATH))
        var text_proj = Tensor.from_view(g_st.tensor_view(String(CLIP_G_TEXT_PROJ)), self.ctx)
        var clip_l_tok = ClipTokenizer(String(CLIP_L_TOK))
        var clip_g_tok = ClipTokenizer(String(CLIP_G_TOK))

        # positive (cond) + negative (uncond)
        var pos = _encode_one(
            self.params.prompt, clip_l, clip_g, text_proj,
            clip_l_tok, clip_g_tok, self.params.height, self.params.width, self.ctx,
        )
        var neg = _encode_one(
            self.params.negative, clip_l, clip_g, text_proj,
            clip_l_tok, clip_g_tok, self.params.height, self.params.width, self.ctx,
        )
        # Tensor is Movable-not-Copyable AND a tuple subscript (pos[0]) yields a
        # BORROW — so neither `pos[0]^` (can't transfer out of a borrow) nor
        # `pos[0].copy()` (no such method) compiles. Materialize each owned
        # conditioning tensor via the proven `.clone(ctx)` idiom (cf.
        # sdxl_unet_stack_lora.mojo `gf[0].clone(ctx)`); the four tensors are
        # tiny (context ~315 KB BF16, y ~5.5 KB) so the transient copy is free.
        var caps = SdxlCaps(
            pos[0].clone(self.ctx), neg[0].clone(self.ctx),
            pos[1].clone(self.ctx), neg[1].clone(self.ctx),
        )
        # clip_l/clip_g/text_proj drop here (Movable-not-Copyable -> freed at scope exit).
        _print_vram("after CLIP encode (encoders freed)")
        self.caps = List[ArcPointer[SdxlCaps]]()
        self.caps.append(ArcPointer(caps^))
        ref encoded_caps = self.caps[0][]
        var cached = SdxlCaps(
            encoded_caps.context.clone(self.ctx),
            encoded_caps.context_uncond.clone(self.ctx),
            encoded_caps.y.clone(self.ctx),
            encoded_caps.y_uncond.clone(self.ctx),
        )
        self.cap_cache = List[ArcPointer[SdxlCaps]]()
        self.cap_cache.append(ArcPointer(cached^))
        self.cap_cache_prompt = self.params.prompt.copy()
        self.cap_cache_negative = self.params.negative.copy()
        self.cap_cache_width = self.params.width
        self.cap_cache_height = self.params.height

    def _load_model(mut self) raises:
        """Load the selected compatible SDXL checkpoint (once; stays resident)."""
        var manifest = default_manifest_by_id(String("sdxl"))
        var checkpoint = (
            self.params.checkpoint_path.copy()
            if self.params.checkpoint_path != String("")
            else manifest.denoiser_path.copy()
        )
        var lora_signature = self._requested_lora_signature()
        if (
            self.loaded and self.model_width == self.params.width
            and self.model_height == self.params.height
            and self.loaded_checkpoint == checkpoint
            and self.loaded_lora_signature == lora_signature
        ):
            return
        self.model_square = List[ArcPointer[SDXLUNet[LH_SQUARE, LW_SQUARE]]]()
        self.model_1152x896 = List[ArcPointer[SDXLUNet[LH_1152X896, LW_1152X896]]]()
        self.model_896x1152 = List[ArcPointer[SDXLUNet[LH_896X1152, LW_896X1152]]]()
        self.model_landscape = List[ArcPointer[SDXLUNet[LH_LANDSCAPE, LW_LANDSCAPE]]]()
        self.model_portrait = List[ArcPointer[SDXLUNet[LH_PORTRAIT, LW_PORTRAIT]]]()
        self.model_1280x832 = List[ArcPointer[SDXLUNet[LH_1280X832, LW_1280X832]]]()
        self.model_832x1280 = List[ArcPointer[SDXLUNet[LH_832X1280, LW_832X1280]]]()
        self.loaded = False
        self.loaded_checkpoint = String("")
        self.loaded_lora_signature = String("")
        self.model_width = 0
        self.model_height = 0
        self.ctx.synchronize()
        cu_mempool_trim_current(0)
        self.ctx.synchronize()
        _print_vram("before SDXL UNet load")
        var lora_paths = List[String]()
        var lora_weights = List[Float32]()
        for i in range(len(self.params.loras)):
            lora_paths.append(self.params.loras[i].name)
            lora_weights.append(Float32(self.params.loras[i].weight))
        if self.params.width == 1024:
            print("[sdxl] loading SDXLUNet[128,128] from", checkpoint)
            self.model_square.append(ArcPointer(
                SDXLUNet[LH_SQUARE, LW_SQUARE].load_with_loras(
                    checkpoint, lora_paths, lora_weights, self.ctx
                )
            ))
        elif self.params.width == 1152:
            print("[sdxl] loading SDXLUNet[112,144] from", checkpoint)
            self.model_1152x896.append(ArcPointer(
                SDXLUNet[LH_1152X896, LW_1152X896].load_with_loras(
                    checkpoint, lora_paths, lora_weights, self.ctx
                )
            ))
        elif self.params.width == 896:
            print("[sdxl] loading SDXLUNet[144,112] from", checkpoint)
            self.model_896x1152.append(ArcPointer(
                SDXLUNet[LH_896X1152, LW_896X1152].load_with_loras(
                    checkpoint, lora_paths, lora_weights, self.ctx
                )
            ))
        elif self.params.width == 1344:
            print("[sdxl] loading SDXLUNet[96,168] from", checkpoint)
            self.model_landscape.append(ArcPointer(
                SDXLUNet[LH_LANDSCAPE, LW_LANDSCAPE].load_with_loras(
                    checkpoint, lora_paths, lora_weights, self.ctx
                )
            ))
        elif self.params.width == 768:
            print("[sdxl] loading SDXLUNet[168,96] from", checkpoint)
            self.model_portrait.append(ArcPointer(
                SDXLUNet[LH_PORTRAIT, LW_PORTRAIT].load_with_loras(
                    checkpoint, lora_paths, lora_weights, self.ctx
                )
            ))
        elif self.params.width == 1280:
            print("[sdxl] loading SDXLUNet[104,160] from", checkpoint)
            self.model_1280x832.append(ArcPointer(
                SDXLUNet[LH_1280X832, LW_1280X832].load_with_loras(
                    checkpoint, lora_paths, lora_weights, self.ctx
                )
            ))
        else:
            print("[sdxl] loading SDXLUNet[160,104] from", checkpoint)
            self.model_832x1280.append(ArcPointer(
                SDXLUNet[LH_832X1280, LW_832X1280].load_with_loras(
                    checkpoint, lora_paths, lora_weights, self.ctx
                )
            ))
        self.loaded = True
        self.loaded_checkpoint = checkpoint^
        self.loaded_lora_signature = lora_signature^
        self.model_width = self.params.width
        self.model_height = self.params.height
        _print_vram("after SDXL UNet load (resident)")

    def _requested_lora_signature(self) -> String:
        var out = String("")
        for i in range(len(self.params.loras)):
            if i > 0:
                out += String("|")
            out += self.params.loras[i].name
            out += String("@")
            out += String(self.params.loras[i].weight)
        return out

    def _prepare_job(mut self) raises:
        """SwarmUI scheduler + seeded scaled initial latent (honors seed)."""
        self.sched = List[ArcPointer[SDXLEulerScheduler]]()
        var scheduler_name = _sdxl_scheduler_name(self.params.scheduler)
        var sched = SDXLEulerScheduler(self.params.steps, scheduler_name)
        var sigmas = sched.sigmas()
        var init_sigma = sdxl_initial_noise_sigma(sigmas[0])
        var lh = self.params.height // 8
        var lw = self.params.width // 8
        var nsh = [1, 4, lh, lw]
        var noise = randn(nsh.copy(), UInt64(self.params.seed), STDtype.F32, self.ctx)
        if self.params.variation_strength > 0.0:
            var vnoise = randn(
                nsh.copy(),
                UInt64(self.params.variation_seed + self.params.image_index),
                STDtype.F32,
                self.ctx,
            )
            var base_h = noise.to_host(self.ctx)
            var var_h = vnoise.to_host(self.ctx)
            var blended = variation_noise_chw(
                base_h, var_h, 4, lh, lw, self.params.variation_strength
            )
            noise = Tensor.from_host(blended, nsh.copy(), STDtype.F32, self.ctx)
        var x = mul_scalar(noise, init_sigma, self.ctx)
        self.sched.append(ArcPointer(sched^))
        self.latent = List[ArcPointer[Tensor]]()
        self.latent.append(ArcPointer(x^))
        self.previous_denoised = List[ArcPointer[Tensor]]()
        self.previous_sigma = 0.0
        print(
            "[sdxl] job", self.params.job_id, ":", self.params.steps,
            "steps, cfg", self.cfg, "seed", self.params.seed,
            "size", self.params.width, "x", self.params.height,
            "sampler", _sdxl_sampler_name(self.params.sampler),
            "scheduler", scheduler_name,
        )

    # ── one denoise step (CFG dual forward + selected SwarmUI sampler) ─────────
    def _denoise_one(mut self) raises:
        var i = self.cur
        var sigmas = self.sched[0][].sigmas()
        var sigma = sigmas[i]
        var sigma_next = sigmas[i + 1]
        var t_i = self.sched[0][].timestep(i)

        var c_in = sdxl_input_scale(sigma)
        var x_in_f32 = mul_scalar(self.latent[0][], c_in, self.ctx)
        var x_in = cast_tensor(x_in_f32, STDtype.BF16, self.ctx)

        var eps_cond: Tensor
        var eps_uncond: Tensor
        if self.params.width == 1024:
            eps_cond = cast_tensor(self.model_square[0][].forward(
                x_in, t_i, self.caps[0][].context, self.caps[0][].y, self.ctx
            ), STDtype.F32, self.ctx)
            eps_uncond = cast_tensor(self.model_square[0][].forward(
                x_in, t_i, self.caps[0][].context_uncond,
                self.caps[0][].y_uncond, self.ctx
            ), STDtype.F32, self.ctx)
        elif self.params.width == 1152:
            eps_cond = cast_tensor(self.model_1152x896[0][].forward(
                x_in, t_i, self.caps[0][].context, self.caps[0][].y, self.ctx
            ), STDtype.F32, self.ctx)
            eps_uncond = cast_tensor(self.model_1152x896[0][].forward(
                x_in, t_i, self.caps[0][].context_uncond,
                self.caps[0][].y_uncond, self.ctx
            ), STDtype.F32, self.ctx)
        elif self.params.width == 896:
            eps_cond = cast_tensor(self.model_896x1152[0][].forward(
                x_in, t_i, self.caps[0][].context, self.caps[0][].y, self.ctx
            ), STDtype.F32, self.ctx)
            eps_uncond = cast_tensor(self.model_896x1152[0][].forward(
                x_in, t_i, self.caps[0][].context_uncond,
                self.caps[0][].y_uncond, self.ctx
            ), STDtype.F32, self.ctx)
        elif self.params.width == 1344:
            eps_cond = cast_tensor(self.model_landscape[0][].forward(
                x_in, t_i, self.caps[0][].context, self.caps[0][].y, self.ctx
            ), STDtype.F32, self.ctx)
            eps_uncond = cast_tensor(self.model_landscape[0][].forward(
                x_in, t_i, self.caps[0][].context_uncond,
                self.caps[0][].y_uncond, self.ctx
            ), STDtype.F32, self.ctx)
        elif self.params.width == 768:
            eps_cond = cast_tensor(self.model_portrait[0][].forward(
                x_in, t_i, self.caps[0][].context, self.caps[0][].y, self.ctx
            ), STDtype.F32, self.ctx)
            eps_uncond = cast_tensor(self.model_portrait[0][].forward(
                x_in, t_i, self.caps[0][].context_uncond,
                self.caps[0][].y_uncond, self.ctx
            ), STDtype.F32, self.ctx)
        elif self.params.width == 1280:
            eps_cond = cast_tensor(self.model_1280x832[0][].forward(
                x_in, t_i, self.caps[0][].context, self.caps[0][].y, self.ctx
            ), STDtype.F32, self.ctx)
            eps_uncond = cast_tensor(self.model_1280x832[0][].forward(
                x_in, t_i, self.caps[0][].context_uncond,
                self.caps[0][].y_uncond, self.ctx
            ), STDtype.F32, self.ctx)
        else:
            eps_cond = cast_tensor(self.model_832x1280[0][].forward(
                x_in, t_i, self.caps[0][].context, self.caps[0][].y, self.ctx
            ), STDtype.F32, self.ctx)
            eps_uncond = cast_tensor(self.model_832x1280[0][].forward(
                x_in, t_i, self.caps[0][].context_uncond,
                self.caps[0][].y_uncond, self.ctx
            ), STDtype.F32, self.ctx)
        var eps = sdxl_cfg(eps_cond, eps_uncond, self.cfg, self.ctx)
        var sampler_name = _sdxl_sampler_name(self.params.sampler)
        var x_new: Tensor
        if sampler_name == String("dpmpp_2m"):
            var denoised = sdxl_denoised_from_eps(
                self.latent[0][], eps, sigma, self.ctx
            )
            var have_previous = len(self.previous_denoised) > 0
            var previous = (
                self.previous_denoised[0][].clone(self.ctx)
                if have_previous
                else denoised.clone(self.ctx)
            )
            x_new = sdxl_dpmpp_2m_step(
                self.latent[0][],
                denoised,
                previous,
                have_previous,
                sigma,
                sigma_next,
                self.previous_sigma,
                self.ctx,
            )
            self.previous_denoised = List[ArcPointer[Tensor]]()
            self.previous_denoised.append(ArcPointer(denoised^))
            self.previous_sigma = sigma
        else:
            # Comfy's `ddim` sampler object is intentionally the Euler sampler.
            x_new = sdxl_euler_step(
                self.latent[0][], eps, sigma, sigma_next, self.ctx
            )
        self.latent = List[ArcPointer[Tensor]]()
        self.latent.append(ArcPointer(x_new^))

    # ── final decode + PNG(tEXt) ──────────────────────────────────────────────
    def _decode_shape[LH_: Int, LW_: Int](
        mut self, latent: Tensor, png_path: String
    ) raises:
        var manifest = default_manifest_by_id(String("sdxl"))
        var mem = cu_mem_get_info()
        var free_gib = Float64(mem.free_bytes) / 1073741824.0
        if mem.free_bytes > WHOLE_DECODE_MIN_FREE_BYTES:
            print("[sdxl] WHOLE-image decode (free=", free_gib,
                  "GiB) — tiled measured to degrade output (MJ-1054)")
            var dec = load_sdxl_ldm_decoder[LH_, LW_](manifest.vae_path, self.ctx)
            var wimg = dec.decode(latent, self.ctx)
            _save_rgb_png_with_text(wimg, png_path, self.params.params_json, self.ctx)
            return
        print("[sdxl] tiled VAE decode FALLBACK (3x3 overlap) — free=", free_gib,
              "GiB below whole-image threshold; tiled degrades output (MJ-1054)")
        var img = sdxl_tiled_decode[LH_, LW_](latent, manifest.vae_path, self.ctx)
        _save_rgb_png_with_text(img, png_path, self.params.params_json, self.ctx)

    def _decode_child_shape[LH_: Int, LW_: Int](
        mut self, latent: Tensor, png_path: String
    ) raises -> Bool:
        """Decode in a separate GPU process without dropping the resident UNet.

        The child executes the same 3x3 tiled decoder used by the existing 24 GB
        fallback. If CUDA cannot admit both processes, return False so the proven
        release-and-decode path below still completes the job.
        """
        var manifest = default_manifest_by_id(String("sdxl"))
        try:
            var img = decode_tiled_subprocess(
                latent, manifest.vae_path, LH_, LW_, self.ctx
            )
            _save_rgb_png_with_text(
                img, png_path, self.params.params_json, self.ctx
            )
            return True
        except e:
            print("[sdxl] resident child decode unavailable; using release fallback:", e)
            return False

    def _try_resident_decode(
        mut self, latent: Tensor, png_path: String
    ) raises -> Bool:
        if self.params.width == 1024:
            return self._decode_child_shape[LH_SQUARE, LW_SQUARE](latent, png_path)
        elif self.params.width == 1152:
            return self._decode_child_shape[LH_1152X896, LW_1152X896](latent, png_path)
        elif self.params.width == 896:
            return self._decode_child_shape[LH_896X1152, LW_896X1152](latent, png_path)
        elif self.params.width == 1344:
            return self._decode_child_shape[LH_LANDSCAPE, LW_LANDSCAPE](latent, png_path)
        elif self.params.width == 768:
            return self._decode_child_shape[LH_PORTRAIT, LW_PORTRAIT](latent, png_path)
        elif self.params.width == 1280:
            return self._decode_child_shape[LH_1280X832, LW_1280X832](latent, png_path)
        return self._decode_child_shape[LH_832X1280, LW_832X1280](latent, png_path)

    def _decode_and_save(mut self) raises -> String:
        var png_path = self.params.out_dir + "/" + self.params.job_id + ".png"
        var latent = self.latent[0][].clone(self.ctx)
        # Per-job conditioning is dead weight at decode; free before the decoder.
        self.caps = List[ArcPointer[SdxlCaps]]()
        self.sched = List[ArcPointer[SDXLEulerScheduler]]()
        self.latent = List[ArcPointer[Tensor]]()
        self.previous_denoised = List[ArcPointer[Tensor]]()
        self.previous_sigma = 0.0
        # First try the process-separated decoder. It uses the same tiled decode
        # math as the existing 24 GB path while preserving the resident UNet for
        # the next image. Trim only dead transient allocations in the parent.
        self.ctx.synchronize()
        cu_mempool_trim_current(0)
        self.ctx.synchronize()
        if self._try_resident_decode(latent, png_path):
            return png_path
        # Admission can fail when device-global headroom is insufficient. Preserve
        # the old completion path: release UNet, trim, and decode in this process.
        self.model_square = List[ArcPointer[SDXLUNet[LH_SQUARE, LW_SQUARE]]]()
        self.model_1152x896 = List[ArcPointer[SDXLUNet[LH_1152X896, LW_1152X896]]]()
        self.model_896x1152 = List[ArcPointer[SDXLUNet[LH_896X1152, LW_896X1152]]]()
        self.model_landscape = List[ArcPointer[SDXLUNet[LH_LANDSCAPE, LW_LANDSCAPE]]]()
        self.model_portrait = List[ArcPointer[SDXLUNet[LH_PORTRAIT, LW_PORTRAIT]]]()
        self.model_1280x832 = List[ArcPointer[SDXLUNet[LH_1280X832, LW_1280X832]]]()
        self.model_832x1280 = List[ArcPointer[SDXLUNet[LH_832X1280, LW_832X1280]]]()
        self.loaded = False
        self.loaded_checkpoint = String("")
        self.loaded_lora_signature = String("")
        self.model_width = 0
        self.model_height = 0
        self.ctx.synchronize()
        cu_mempool_trim_current(0)
        self.ctx.synchronize()
        if self.params.width == 1024:
            self._decode_shape[LH_SQUARE, LW_SQUARE](latent, png_path)
        elif self.params.width == 1152:
            self._decode_shape[LH_1152X896, LW_1152X896](latent, png_path)
        elif self.params.width == 896:
            self._decode_shape[LH_896X1152, LW_896X1152](latent, png_path)
        elif self.params.width == 1344:
            self._decode_shape[LH_LANDSCAPE, LW_LANDSCAPE](latent, png_path)
        elif self.params.width == 768:
            self._decode_shape[LH_PORTRAIT, LW_PORTRAIT](latent, png_path)
        elif self.params.width == 1280:
            self._decode_shape[LH_1280X832, LW_1280X832](latent, png_path)
        else:
            self._decode_shape[LH_832X1280, LW_832X1280](latent, png_path)
        return png_path

    def _clear_job(mut self):
        self.active = False
        self.phase = SPHASE_IDLE
        self.cur = 0
        self.cancel_flag = False
        self.announced = False
        self.caps = List[ArcPointer[SdxlCaps]]()
        self.sched = List[ArcPointer[SDXLEulerScheduler]]()
        self.latent = List[ArcPointer[Tensor]]()
        self.previous_denoised = List[ArcPointer[Tensor]]()
        self.previous_sigma = 0.0

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
            if self.phase == SPHASE_ENCODE:
                if not self.announced:
                    # announce BEFORE the long blocking encode tick (per-job
                    # CLIP-L + CLIP-G load + dual-prompt forward).
                    self.announced = True
                    r.step = 0
                    r.phase = String("encoding")
                    return r^
                var encode_t0 = perf_counter_ns()
                self._encode()
                self.text_encode_seconds = Float64(perf_counter_ns() - encode_t0) / 1.0e9
                self._record_vram()
                self.announced = False
                self.phase = SPHASE_LOAD
                r.step = 0
                return r^
            if self.phase == SPHASE_LOAD:
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
                self.phase = SPHASE_DENOISE
                r.step = 0
                r.phase = String("sampling")
                return r^
            if self.phase == SPHASE_DENOISE:
                var denoise_t0 = perf_counter_ns()
                self._denoise_one()
                self.denoise_seconds += Float64(perf_counter_ns() - denoise_t0) / 1.0e9
                self._record_vram()
                self.cur += 1
                r.step = self.cur
                r.phase = String("sampling")
                if self.cur >= self.params.steps:
                    self.phase = SPHASE_DECODE
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
            print("[sdxl][manifest] saved:", manifest)
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
