# serenitymojo.serve.flux_backend — the real FLUX.1-dev GenBackend.
#
# Wraps the VERIFIED serenitymojo/pipeline/flux_sample_cli.mojo stages behind the
# pull-based GenBackend seam (backend.mojo). EVERY numeric convention is reused
# from flux_sample_cli (its helpers — encode_text/_pack_latent/_unpack_latent and
# the shape constants — are imported, NOT re-derived):
#
#   tokenizer (CLIP-L BPE + T5-XXL Unigram, both bit-exact vs HF) →
#   CLIP-L pooled [1,768] + T5-XXL hidden [1,512,4096] →
#   FLUX.1-dev offloaded DiT, GUIDANCE-DISTILLED single forward per step
#   (guidance_vec is a MODEL INPUT scalar, NOT a CFG multiplier — no negative
#   prompt / no dual forward) + flow-match Euler update →
#   FLUX VAE TILED decode (3x3 overlap+feather) → PNG SIGNED (genparams tEXt).
#
# FLUX is guidance-distilled, so — exactly like flux_sample_cli — the negative
# prompt is read and acknowledged but discarded (there is no CFG path in the
# DiT). params.cfg is the guidance scalar fed to the model.
#
# Residency model (24 GB GPU, FLUX.1-dev ~23 GB on disk — TIGHT):
#   * The FLUX.1-dev DiT is BLOCK-STREAMED from disk (Flux1Offloaded: the shared
#     non-block weights + a BlockLoader that mmaps and streams one block at a
#     time). The OFFLOADER HANDLE + the RoPE tables are loaded ONCE (first job)
#     and STAY RESIDENT across jobs — the residency win for an offloaded model
#     (subsequent base jobs skip the loader rebuild + rope build). Exactly like
#     qwenimage_backend's offloader handle.
#   * The CLIP-L (~250 MB) + T5-XXL (~9.5 GB F16) encoders are loaded → used →
#     freed PER JOB inside the ENCODE step (encode_text does the load+free).
#   * The FLUX VAE (~330 MB) is loaded PER JOB inside the TILED DECODE step.
#   * Before the 1024² VAE decode, the resident DiT offloader handle + rope are
#     FREED and the mempool TRIMMED (MEASURED on SDXL: a 1024² VAE decode OOMs a
#     24 GB card when the denoiser stays resident; flux_sample_cli's staged-
#     loading note records the same FLUX behaviour). self.loaded is reset so the
#     NEXT job reloads the DiT in the LOAD phase — same pattern as sdxl_backend.
#
# step() state machine: ENCODE (per-job, blocking — announced phase="encoding")
#   → LOAD (DiT offloader + rope, announced phase="loading") → DENOISE×steps
#   (one guidance-distilled forward + Euler update per tick) → DECODE (announced
#   phase="decoding") → done. cancel() makes the next step() return cancelled and
#   frees all per-job tensors.
#
# Size support: 1024x1024 ONLY (the FLUX DiT rope/pack/tile shapes — N_IMG,
# LATENT_H/W, TILE_H/W — are comptime-fixed in flux_sample_cli). steps/guidance
# (=cfg)/seed ARE honored at runtime (the denoise loop reads them from JobParams;
# the flow-match sigma table is built per job for params.steps).
#
# LoRA: HONORED via Flux1Offloaded.load_with_lora (Kohya/sd-scripts BFL FLUX
# LoRA, additive overlay W += scale·up@down at multiplier 1.0 — the saved
# checkpoint is never fused, per the LoRA-never-fused rule). Because a LoRA
# changes the resident DiT, a LoRA job (or a LoRA change) reloads the DiT; only
# base (no-LoRA) jobs keep the resident handle. Diffusers-format FLUX LoRAs are
# not supported by the model loader yet (it fails loud inside load_with_lora).

from std.collections import Optional
from std.ffi import external_call
from std.gpu.host import DeviceContext
from std.memory import alloc, ArcPointer

from image.buffer import Image
from image.png import encode_png_with_text

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.ffi import BytePtr
from serenitymojo.image.png import _quantize, ValueRange
from serenitymojo.offload.vmm_cuda import cu_mempool_trim_current, cu_mem_get_info

from serenitymojo.models.dit.flux1_dit import (
    Flux1Config, Flux1Offloaded, build_flux1_rope_tables,
)
from serenitymojo.sampling.flux1_dev import build_flux1_sigma_schedule
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.random import randn
from serenitymojo.ops.tensor_algebra import add, mul_scalar
from serenitymojo.sampling.sampler_registry import (
    sampler_admission_for_backend, scheduler_admission_for_backend,
)
from serenitymojo.sampling.variation_noise import swarm_variation_noise_chw
from serenitymojo.pipeline.flux_tiled_decode import flux_tiled_decode
from serenitymojo.pipeline.flux_sample_cli import (
    FluxCaps, encode_text, _pack_latent, _unpack_latent,
    DIT_PATH, VAE_PATH,
    HEIGHT, WIDTH, AE_IN_CHANNELS, LATENT_H, LATENT_W, IMG_H2, IMG_W2,
    N_IMG, N_TXT, S,
)
from serenitymojo.serve.backend import (
    GenBackend, JobParams, StepResult, LoraSpec,
    reject_unsupported_common_runtime_params,
    reject_unsupported_reference_image_params, reject_unsupported_mask_image_params,
    reject_unsupported_inpaint_conditioning_params,
    reject_unsupported_qwen_edit_conditioning_params,
    reject_unsupported_conditioning_mask_params, reject_unsupported_lanpaint_params,
    warn_unsupported_advanced_sampling_params,
)

comptime GENPARAMS_TEXT_KEY = "serenity.genparams.v1"


comptime FPHASE_IDLE = 0
comptime FPHASE_ENCODE = 1
comptime FPHASE_LOAD = 2
comptime FPHASE_DENOISE = 3
comptime FPHASE_DECODE = 4


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
        String("echo -n '[flux][vram] ") + tag
        + ": ' && nvidia-smi --query-gpu=memory.used --format=csv,noheader"
    )


def _save_rgb_png_with_text(
    rgb: Tensor, path: String, params_json: String, ctx: DeviceContext
) raises:
    """[1,3,H,W] SIGNED float tensor → 8-bit RGB PNG with the job params in a
    serenity.genparams.v1 tEXt chunk. Quantization math == save_png's
    (_quantize, ValueRange.SIGNED); only the writer differs (tEXt support).
    Identical to qwenimage_backend/sdxl_backend._save_rgb_png_with_text."""
    var shape = rgb.shape()
    if len(shape) != 4 or shape[0] != 1 or shape[1] != 3:
        raise Error("flux_backend: expected [1,3,H,W] rgb tensor")
    var height = shape[2]
    var width = shape[3]
    var host = rgb.to_host(ctx)
    var plane = height * width
    if len(host) != 3 * plane:
        raise Error("flux_backend: rgb to_host size mismatch")
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


# Single LoRA only (one additive overlay through load_with_lora). Returns the
# selected LoRA path ("" = base). Mirrors the CLI's single argv[2] LoRA slot.
def _select_lora_path(loras: List[LoraSpec]) -> String:
    if len(loras) == 0:
        return String("")
    return loras[0].name.copy()


struct FluxBackend(GenBackend, Movable):
    var ctx: DeviceContext

    # ── resident across BASE jobs (offloader handle + rope, first base job) ──
    # ArcPointer wrappers: Flux1Offloaded / Tensor are Movable-not-Copyable, and
    # List[T] requires T: Copyable — Arc is Copyable (refcount), so List[Arc[..]]
    # holds the 0/1. `loaded_lora` tracks which LoRA the resident DiT carries
    # ("" = base) so a LoRA change forces a reload.
    var loaded: Bool
    var loaded_lora: String
    var model: List[ArcPointer[Flux1Offloaded]]  # 0/1 (resident offloader)
    var rope_cos: List[ArcPointer[Tensor]]       # 0/1 (resident rope cos)
    var rope_sin: List[ArcPointer[Tensor]]       # 0/1 (resident rope sin)

    # ── per-job state (cleared on done/failed/cancelled) ──
    var active: Bool
    var cancel_flag: Bool
    var phase: Int
    var announced: Bool
    var cur: Int
    var params: JobParams
    var guidance: Float32
    var caps: List[ArcPointer[FluxCaps]]   # 0/1
    var sched: List[Float32]               # flow-match sigma table (steps+1)
    var latent: List[ArcPointer[Tensor]]   # 0/1 (packed [1,N_IMG,64] BF16-castable)

    def __init__(out self) raises:
        self.ctx = DeviceContext()
        self.loaded = False
        self.loaded_lora = String("")
        self.model = List[ArcPointer[Flux1Offloaded]]()
        self.rope_cos = List[ArcPointer[Tensor]]()
        self.rope_sin = List[ArcPointer[Tensor]]()
        self.active = False
        self.cancel_flag = False
        self.phase = FPHASE_IDLE
        self.announced = False
        self.cur = 0
        self.params = JobParams()
        self.guidance = Float32(3.5)
        self.caps = List[ArcPointer[FluxCaps]]()
        self.sched = List[Float32]()
        self.latent = List[ArcPointer[Tensor]]()

    def backend_name(self) -> String:
        return String("flux")

    def model_name(self) -> String:
        return String("FLUX.1-dev")

    def resident_model(self) -> String:
        """Matches the /v1/models scan entry for the resident checkpoint
        (the flux1-dev.safetensors checkpoint)."""
        return String("flux1-dev.safetensors") if self.loaded else String("")

    # ── job admission ─────────────────────────────────────────────────────────
    def start(mut self, params: JobParams) raises:
        if self.active:
            raise Error("FluxBackend.start: a job is already running")
        reject_unsupported_common_runtime_params(params, String("flux"))
        reject_unsupported_reference_image_params(params, String("flux"))
        reject_unsupported_inpaint_conditioning_params(params, String("flux"))
        reject_unsupported_qwen_edit_conditioning_params(params, String("flux"))
        reject_unsupported_conditioning_mask_params(params, String("flux"))
        reject_unsupported_mask_image_params(params, String("flux"))
        reject_unsupported_lanpaint_params(params, String("flux"))
        var sampler_admission = sampler_admission_for_backend(String("flux"), params.sampler)
        if not sampler_admission.supported:
            raise Error(
                String("flux: unsupported sampler '") + params.sampler
                + String("'; ") + sampler_admission.reason
            )
        var scheduler_admission = scheduler_admission_for_backend(String("flux"), params.scheduler)
        if not scheduler_admission.supported:
            raise Error(
                String("flux: unsupported scheduler '") + params.scheduler
                + String("'; ") + scheduler_admission.reason
            )
        # 1024x1024 only: the FLUX DiT rope/pack/tile shapes are comptime-fixed.
        if not (params.width == WIDTH and params.height == HEIGHT):
            raise Error(
                String("flux: unsupported size ") + String(params.width)
                + "x" + String(params.height)
                + " — only " + String(WIDTH) + "x" + String(HEIGHT)
                + " is served (the FLUX DiT rope/pack/tile shapes are comptime-"
                + "fixed; resolution changes need a recompile)"
            )
        if len(params.loras) > 1:
            raise Error(
                "flux: only a single LoRA overlay is supported per job"
                " (one additive Kohya-BFL overlay); submit at most one LoRA"
            )
        if params.init_image.byte_length() > 0:
            raise Error(
                "flux: img2img is not supported for FLUX.1-dev yet;"
                " submit without an init image"
            )
        # Warn-loud (never silently drop) on any advanced-sampling knob set but
        # unsupported by this fixed flow-match Euler path.
        warn_unsupported_advanced_sampling_params(params, String("flux"), List[String]())
        self.params = params.copy()
        # FLUX is guidance-distilled: params.cfg is the guidance scalar fed to the
        # DiT (NOT a CFG multiplier). Negative prompt is discarded (no CFG path).
        self.guidance = Float32(params.cfg)
        self.active = True
        self.cancel_flag = False
        self.cur = 0
        self.announced = False
        self.phase = FPHASE_ENCODE

    def cancel(mut self):
        self.cancel_flag = True

    def between_jobs_trim(mut self) raises:
        """F3: reclaim the per-job transient peak (CLIP-L + T5-XXL encoders
        ~10 GB, the VAE decoder, 1024² forward + decode activations) back to the
        OS via cuMemPoolTrimTo. The resident DiT offloader buffers (when a base
        job left them resident) have live suballocations and are NOT reclaimed."""
        var before = cu_mem_get_info()
        self.ctx.synchronize()
        cu_mempool_trim_current(0)
        self.ctx.synchronize()
        var after = cu_mem_get_info()
        print("[flux] between-jobs trim: used",
              before.used_bytes() // (1024 * 1024), "->",
              after.used_bytes() // (1024 * 1024), "MiB (reclaimed",
              (before.used_bytes() - after.used_bytes()) // (1024 * 1024), "MiB)")

    # ── per-job prep ───────────────────────────────────────────────────────────
    def _encode(mut self) raises:
        """Real CLIP-L pooled + T5-XXL hidden encode of params.prompt (encoders
        loaded + freed inside encode_text). FLUX is guidance-distilled — only the
        positive prompt is encoded; the negative is discarded (no CFG path)."""
        _print_vram("before CLIP-L + T5-XXL load")
        var caps = encode_text(self.params.prompt, self.ctx)
        _print_vram("after text encode (encoders freed)")
        self.caps = List[ArcPointer[FluxCaps]]()
        self.caps.append(ArcPointer(caps^))

    def _free_dit(mut self):
        """Drop the resident DiT offloader handle + rope tables and reset the
        residency flags (so the next LOAD reloads)."""
        self.model = List[ArcPointer[Flux1Offloaded]]()
        self.rope_cos = List[ArcPointer[Tensor]]()
        self.rope_sin = List[ArcPointer[Tensor]]()
        self.loaded = False
        self.loaded_lora = String("")

    def _load_model(mut self) raises:
        """Load the FLUX.1-dev DiT offloader handle + rope tables. Resident across
        BASE jobs; a LoRA job (or a LoRA change vs the resident handle) reloads."""
        var want_lora = _select_lora_path(self.params.loras)
        # If a different LoRA (or base-vs-LoRA mismatch) is resident, drop it.
        if self.loaded and self.loaded_lora != want_lora:
            print("[flux] LoRA selection changed ('", self.loaded_lora,
                  "' -> '", want_lora, "') — reloading DiT")
            self._free_dit()
        if self.loaded:
            return
        _print_vram("before FLUX DiT offloader load")
        self.model = List[ArcPointer[Flux1Offloaded]]()
        if want_lora != String(""):
            print("[flux] loading FLUX.1-dev DiT (offloaded) + LoRA overlay:", want_lora)
            self.model.append(ArcPointer(Flux1Offloaded.load_with_lora(
                DIT_PATH, Flux1Config.dev(), want_lora, Float32(1.0), self.ctx
            )))
        else:
            print("[flux] loading FLUX.1-dev DiT (offloaded) from", DIT_PATH)
            self.model.append(ArcPointer(Flux1Offloaded.load(
                DIT_PATH, Flux1Config.dev(), self.ctx
            )))
        # RoPE tables (resident with the offloader; rebuilt only on reload).
        var rope = build_flux1_rope_tables[N_IMG, N_TXT, 24, 128](
            IMG_H2, IMG_W2, self.ctx, STDtype.BF16
        )
        self.rope_cos = List[ArcPointer[Tensor]]()
        self.rope_sin = List[ArcPointer[Tensor]]()
        self.rope_cos.append(ArcPointer(rope[0].clone(self.ctx)))
        self.rope_sin.append(ArcPointer(rope[1].clone(self.ctx)))
        self.loaded = True
        self.loaded_lora = want_lora
        _print_vram("after FLUX DiT offloader load (resident)")

    def _prepare_job(mut self) raises:
        """Flow-match sigma table (honors steps) + seeded initial packed latent
        (honors seed). Mirrors flux_sample_cli.denoise's noise+pack."""
        self.sched = build_flux1_sigma_schedule(self.params.steps, N_IMG)
        var nsh = [1, AE_IN_CHANNELS, LATENT_H, LATENT_W]
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
                base_h, var_h, AE_IN_CHANNELS, LATENT_H, LATENT_W,
                self.params.variation_strength,
            )
            noise = Tensor.from_host(blended, nsh.copy(), STDtype.F32, self.ctx)
        var packed = _pack_latent(noise, self.ctx)
        self.latent = List[ArcPointer[Tensor]]()
        self.latent.append(ArcPointer(packed^))
        print(
            "[flux] job", self.params.job_id, ":", self.params.steps,
            "steps, guidance", self.guidance, "seed", self.params.seed,
            "size", self.params.width, "x", self.params.height,
            "(FLUX Dev guidance-distilled; negative discarded)",
        )

    # ── one denoise step (guidance-distilled single forward + Euler) ───────────
    # Verbatim from flux_sample_cli.denoise's per-step body.
    def _denoise_one(mut self) raises:
        var i = self.cur
        var t_curr = self.sched[i]
        var t_prev = self.sched[i + 1]

        # t_vec / guidance_vec pre-scaled by 1000 (BFL time_factor convention;
        # the foundation t_embedder does NOT apply the 1000x internally).
        var tvals = List[Float32]()
        tvals.append(t_curr * 1000.0)
        var t_vec = Tensor.from_host(tvals, [1], STDtype.F32, self.ctx)

        var gvals = List[Float32]()
        gvals.append(self.guidance * 1000.0)
        var g_vec = Tensor.from_host(gvals, [1], STDtype.F32, self.ctx)

        var img_bf = cast_tensor(self.latent[0][], STDtype.BF16, self.ctx)
        var pred = cast_tensor(
            self.model[0][].forward[N_IMG, N_TXT, S](
                img_bf, self.caps[0][].txt, t_vec, Optional[Tensor](g_vec^),
                self.caps[0][].vector, self.rope_cos[0][], self.rope_sin[0][],
                self.ctx,
            ),
            STDtype.F32,
            self.ctx,
        )
        # Euler step: img = img + (t_prev - t_curr) * pred
        var dt = t_prev - t_curr
        var x_new = add(self.latent[0][], mul_scalar(pred, dt, self.ctx), self.ctx)
        self.latent = List[ArcPointer[Tensor]]()
        self.latent.append(ArcPointer(x_new^))

    # ── final decode + PNG(tEXt) ──────────────────────────────────────────────
    def _decode_and_save(mut self) raises -> String:
        var png_path = self.params.out_dir + "/" + self.params.job_id + ".png"
        # Unpack the packed latent [1,N_IMG,64] -> NCHW [1,16,LATENT_H,LATENT_W]
        # (F32) BEFORE freeing the DiT, while the latent is still on device.
        var latent_f32 = cast_tensor(self.latent[0][], STDtype.F32, self.ctx)
        var latent = _unpack_latent(latent_f32, self.ctx)
        # Per-job conditioning is dead weight at decode; free before the decoder.
        self.caps = List[ArcPointer[FluxCaps]]()
        self.sched = List[Float32]()
        self.latent = List[ArcPointer[Tensor]]()
        # MEASURED (SDXL) + flux_sample_cli staged-loading note: the 1024² FLUX
        # VAE decode activations OOM a 24 GB card if the offloaded DiT stays put
        # (the offloader's pool is at a high-water mark). Free the DiT + rope +
        # trim the mempool before decoding; the next job reloads it in LOAD
        # (self.loaded=False).
        self._free_dit()
        self.ctx.synchronize()
        cu_mempool_trim_current(0)
        self.ctx.synchronize()
        print("[flux] tiled VAE decode (3x3 overlap+blend) + save")
        var img = flux_tiled_decode[LATENT_H, LATENT_W](latent, String(VAE_PATH), self.ctx)
        _save_rgb_png_with_text(img, png_path, self.params.params_json, self.ctx)
        return png_path

    def _clear_job(mut self):
        self.active = False
        self.phase = FPHASE_IDLE
        self.cur = 0
        self.cancel_flag = False
        self.announced = False
        self.caps = List[ArcPointer[FluxCaps]]()
        self.sched = List[Float32]()
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
            if self.phase == FPHASE_ENCODE:
                if not self.announced:
                    # announce BEFORE the long blocking encode tick (per-job
                    # CLIP-L + T5-XXL load + forward).
                    self.announced = True
                    r.step = 0
                    r.phase = String("encoding")
                    return r^
                self._encode()
                self.announced = False
                self.phase = FPHASE_LOAD
                r.step = 0
                return r^
            if self.phase == FPHASE_LOAD:
                if not self.loaded:
                    if not self.announced:
                        self.announced = True
                        r.step = 0
                        r.phase = String("loading")
                        return r^
                    self._load_model()
                    self.announced = False
                self._prepare_job()
                self.phase = FPHASE_DENOISE
                r.step = 0
                return r^
            if self.phase == FPHASE_DENOISE:
                self._denoise_one()
                self.cur += 1
                r.step = self.cur
                if self.cur >= self.params.steps:
                    self.phase = FPHASE_DECODE
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
