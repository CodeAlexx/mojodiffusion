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
#   y              = cat([clip_l_pool [1,768],
#                         clip_g_text_embeds [1,1280],      (= clip_g_pool @ text_projᵀ)
#                         zeros [1,768]], dim=1) -> [1,2816] (ADM pooled+time-id vector)
#   y_uncond       = same for the negative prompt          -> [1,2816]
#
# The last 768 of `y` is the SDXL size/crop time-id placeholder, which the reference
# Python (scripts/sdxl_encode.py) and inference-flame's sdxl_encode.rs both fill with
# ZEROS — replicated here verbatim. The denoise (30-step epsilon-prediction Euler +
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
# Size support: 1024x1024 ONLY (the SDXLUNet kernel is compiled at fixed LH=LW=128).
# steps/cfg/seed ARE honored at runtime here (the denoise loop reads them from
# JobParams; the Euler sigma table is built per job for params.steps).
#
# LoRA / img2img: NOT supported yet — rejected at admission so they never silently
# no-op (matches sdxl_sample_cli's "accepted-and-ignored" caveat, made fail-loud here).

from std.collections import Optional
from std.ffi import external_call
from std.gpu.host import DeviceContext
from std.memory import alloc, ArcPointer

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
from serenitymojo.models.vae.ldm_decoder import load_sdxl_ldm_decoder
from serenitymojo.registry.checkpoints import default_manifest_by_id
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.linear import linear
from serenitymojo.ops.random import randn
from serenitymojo.ops.tensor_algebra import mul_scalar, concat
from serenitymojo.sampling.sdxl_euler import (
    SDXLEulerScheduler,
    sdxl_cfg,
    sdxl_euler_step,
    sdxl_initial_noise_sigma,
    sdxl_input_scale,
)
from serenitymojo.sampling.variation_noise import swarm_variation_noise_chw
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


comptime GENPARAMS_TEXT_KEY = "serenity.genparams.v1"

# ── shape constants (1024x1024, matching sdxl_sample_cli + SDXLUNet kernel) ──
comptime WIDTH = 1024
comptime HEIGHT = 1024
comptime LH = HEIGHT // 8      # 128
comptime LW = WIDTH // 8       # 128
comptime CLIP_LEN = 77

# ── verified model + tokenizer paths (match sdxl_sample_cli's manifest + the
#    inference-flame sdxl_encode.rs CLIP defaults) ──
comptime CLIP_L_PATH = "/home/alex/.serenity/models/text_encoders/clip_l.safetensors"
comptime CLIP_G_PATH = "/home/alex/.serenity/models/text_encoders/clip_g.safetensors"
comptime CLIP_L_TOK = "/home/alex/.serenity/models/text_encoders/clip_l.tokenizer.json"
comptime CLIP_G_TOK = "/home/alex/.serenity/models/text_encoders/clip_g.tokenizer.json"
comptime CLIP_G_TEXT_PROJ = "text_projection.weight"

comptime CLIP_PAD_ID = 49407   # CLIP eos == pad
comptime CLIP_EOS_ID = 49407


comptime SPHASE_IDLE = 0
comptime SPHASE_ENCODE = 1
comptime SPHASE_LOAD = 2
comptime SPHASE_DENOISE = 3
comptime SPHASE_DECODE = 4


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
    ctx: DeviceContext,
) raises -> Tuple[Tensor, Tensor]:
    var l_ids = _fit_clip_ids(clip_l_tok.encode(text))
    var g_ids = _fit_clip_ids(clip_g_tok.encode(text))

    # CLIP-L: (last_hidden [1,77,768], pooled [1,768])
    var l_out = clip_l.encode_sdxl[CLIP_LEN](l_ids^, ctx)
    var l_hidden = _to_bf16(l_out[0], ctx)
    var l_pool = _to_bf16(l_out[1], ctx)

    # CLIP-G: (last_hidden [1,77,1280], pooled_raw [1,1280])
    var g_out = clip_g.encode_sdxl[CLIP_LEN](g_ids^, ctx)
    var g_hidden = _to_bf16(g_out[0], ctx)
    # clip_g_text_embeds = clip_g_pool_raw @ text_projectionᵀ -> [1,1280]
    var g_pool = linear(g_out[1], text_proj, Optional[Tensor](None), ctx)
    g_pool = _to_bf16(g_pool, ctx)

    # context = cat([l_hidden, g_hidden], dim=2) -> [1,77,2048]
    var context = concat(2, ctx, l_hidden, g_hidden)
    context = _to_bf16(context, ctx)

    # y = cat([l_pool [1,768], g_pool [1,1280], zeros [1,768]], dim=1) -> [1,2816]
    var zeros_host = List[Float32]()
    for _ in range(768):
        zeros_host.append(Float32(0.0))
    var zeros_pad = Tensor.from_host(zeros_host, [1, 768], STDtype.BF16, ctx)
    var y = concat(1, ctx, l_pool, g_pool, zeros_pad)
    y = _to_bf16(y, ctx)

    return (context^, y^)


struct SdxlBackend(GenBackend, Movable):
    var ctx: DeviceContext

    # ── resident across jobs (UNet weights, loaded once on first job) ──
    var loaded: Bool
    var model: List[ArcPointer[SDXLUNet[LH, LW]]]  # 0/1 (resident UNet)

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

    def __init__(out self) raises:
        self.ctx = DeviceContext()
        self.loaded = False
        self.model = List[ArcPointer[SDXLUNet[LH, LW]]]()
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

    def backend_name(self) -> String:
        return String("sdxl")

    def model_name(self) -> String:
        return String("SDXL")

    def resident_model(self) -> String:
        """Best-effort match to a /v1/models scan entry for the resident UNet
        (the flat sdxl_unet_bf16.safetensors checkpoint). NOTE: the UNet lives
        under /home/alex/EriDiffusion/Models/checkpoints/, not the daemon's
        scanned checkpoints dir, so the scan may not list it — the dispatch is
        wired by the orchestrator regardless."""
        return String("sdxl_unet_bf16.safetensors") if self.loaded else String("")

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
        # 1024x1024 only: the SDXLUNet kernel is compiled at fixed LH=LW=128.
        if not (params.width == 1024 and params.height == 1024):
            raise Error(
                String("sdxl: unsupported size ") + String(params.width)
                + "x" + String(params.height)
                + " — only 1024x1024 is served (the SDXLUNet kernel is compiled at"
                + " fixed LH=LW=128; resolution changes need a recompile)"
            )
        if len(params.loras) > 0:
            raise Error(
                "sdxl: LoRA is not supported for SDXL in this backend yet"
                " (no LoRA overlay path wired); submit without a LoRA"
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

    # ── per-job prep ───────────────────────────────────────────────────────────
    def _encode(mut self) raises:
        """Runtime CLIP-L+G encode of params.prompt AND params.negative into the
        SDXL context/y conditioning (encoders + text_projection loaded then freed)."""
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
            clip_l_tok, clip_g_tok, self.ctx,
        )
        var neg = _encode_one(
            self.params.negative, clip_l, clip_g, text_proj,
            clip_l_tok, clip_g_tok, self.ctx,
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

    def _load_model(mut self) raises:
        """Load the SDXL UNet (once; stays resident). Path from the registered
        manifest (the verified sdxl_sample_cli source of truth)."""
        if self.loaded:
            return
        _print_vram("before SDXL UNet load")
        var manifest = default_manifest_by_id(String("sdxl"))
        print("[sdxl] loading SDXLUNet[", LH, ",", LW, "] from", manifest.denoiser_path)
        self.model = List[ArcPointer[SDXLUNet[LH, LW]]]()
        self.model.append(ArcPointer(SDXLUNet[LH, LW].load(manifest.denoiser_path, self.ctx)))
        self.loaded = True
        _print_vram("after SDXL UNet load (resident)")

    def _prepare_job(mut self) raises:
        """Euler scheduler (honors steps) + seeded scaled initial latent (honors seed)."""
        self.sched = List[ArcPointer[SDXLEulerScheduler]]()
        var sched = SDXLEulerScheduler(self.params.steps)
        var sigmas = sched.sigmas()
        var init_sigma = sdxl_initial_noise_sigma(sigmas[0])
        var nsh = [1, 4, LH, LW]
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
            var blended = swarm_variation_noise_chw(
                base_h, var_h, 4, LH, LW, self.params.variation_strength
            )
            noise = Tensor.from_host(blended, nsh.copy(), STDtype.F32, self.ctx)
        var x = mul_scalar(noise, init_sigma, self.ctx)
        self.sched.append(ArcPointer(sched^))
        self.latent = List[ArcPointer[Tensor]]()
        self.latent.append(ArcPointer(x^))
        print(
            "[sdxl] job", self.params.job_id, ":", self.params.steps,
            "steps, cfg", self.cfg, "seed", self.params.seed,
            "size", self.params.width, "x", self.params.height,
        )

    # ── one denoise step (CFG dual forward + Euler) ────────────────────────────
    # Verbatim from sdxl_sample_cli._denoise's per-step body.
    def _denoise_one(mut self) raises:
        var i = self.cur
        var sigmas = self.sched[0][].sigmas()
        var sigma = sigmas[i]
        var sigma_next = sigmas[i + 1]
        var t_i = self.sched[0][].timestep(i)

        var c_in = sdxl_input_scale(sigma)
        var x_in_f32 = mul_scalar(self.latent[0][], c_in, self.ctx)
        var x_in = cast_tensor(x_in_f32, STDtype.BF16, self.ctx)

        var eps_cond = cast_tensor(
            self.model[0][].forward(
                x_in, t_i, self.caps[0][].context, self.caps[0][].y, self.ctx
            ),
            STDtype.F32, self.ctx,
        )
        var eps_uncond = cast_tensor(
            self.model[0][].forward(
                x_in, t_i, self.caps[0][].context_uncond, self.caps[0][].y_uncond,
                self.ctx,
            ),
            STDtype.F32, self.ctx,
        )
        var eps = sdxl_cfg(eps_cond, eps_uncond, self.cfg, self.ctx)
        var x_new = sdxl_euler_step(
            self.latent[0][], eps, sigma, sigma_next, self.ctx
        )
        self.latent = List[ArcPointer[Tensor]]()
        self.latent.append(ArcPointer(x_new^))

    # ── final decode + PNG(tEXt) ──────────────────────────────────────────────
    def _decode_and_save(mut self) raises -> String:
        var png_path = self.params.out_dir + "/" + self.params.job_id + ".png"
        var latent = self.latent[0][].clone(self.ctx)
        # Per-job conditioning is dead weight at decode; free before the decoder.
        self.caps = List[ArcPointer[SdxlCaps]]()
        self.sched = List[ArcPointer[SDXLEulerScheduler]]()
        self.latent = List[ArcPointer[Tensor]]()
        # The 1024^2 SDXL VAE decode activations OOM on a 24 GB card if the ~6 GB
        # resident UNet stays put (MEASURED: CUDA_ERROR_OUT_OF_MEMORY at decode with
        # ~15 GB already used). Free the UNet + trim the mempool before decoding; the
        # next job reloads it in SPHASE_LOAD (self.loaded=False).
        self.model = List[ArcPointer[SDXLUNet[LH, LW]]]()
        self.loaded = False
        self.ctx.synchronize()
        cu_mempool_trim_current(0)
        self.ctx.synchronize()
        print("[sdxl] loading VAE decoder + decode")
        var manifest = default_manifest_by_id(String("sdxl"))
        var vae = load_sdxl_ldm_decoder[LH, LW](manifest.vae_path, self.ctx)
        var img = vae.decode(latent, self.ctx)
        _save_rgb_png_with_text(img, png_path, self.params.params_json, self.ctx)
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
                self._encode()
                self.announced = False
                self.phase = SPHASE_LOAD
                r.step = 0
                return r^
            if self.phase == SPHASE_LOAD:
                if not self.loaded:
                    if not self.announced:
                        self.announced = True
                        r.step = 0
                        r.phase = String("loading")
                        return r^
                    self._load_model()
                    self.announced = False
                self._prepare_job()
                self.phase = SPHASE_DENOISE
                r.step = 0
                return r^
            if self.phase == SPHASE_DENOISE:
                self._denoise_one()
                self.cur += 1
                r.step = self.cur
                if self.cur >= self.params.steps:
                    self.phase = SPHASE_DECODE
                return r^
            if not self.announced:
                # announce BEFORE the long blocking VAE-decode tick.
                self.announced = True
                r.step = self.params.steps
                r.phase = String("decoding")
                return r^
            var path = self._decode_and_save()
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
