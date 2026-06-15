# serenitymojo.serve.qwenimage_backend — the real Qwen-Image GenBackend.
#
# Wraps the VERIFIED serenitymojo/pipeline/qwenimage_sample_cli.mojo stages
# behind the pull-based GenBackend seam (backend.mojo). EVERY numeric
# convention is reused from qwenimage_sample_cli (its helpers are imported, NOT
# re-derived):
#   tokenizer → Qwen2.5-VL layer-27 encode (template-drop, N_TXT_KEPT=512) →
#   CFG dual-forward denoise (flow-match Euler, dynamic shift) → Qwen-Image VAE
#   decode → PNG SIGNED (with serenity.genparams.v1 tEXt).
#
# Residency model (24 GB GPU):
#   * The Qwen-Image MMDiT is BLOCK-STREAMED from disk (QwenImageDitOffloaded:
#     60 layers, ~40 GB on disk, streamed one block at a time). The OFFLOADER
#     HANDLE (mmap'd shards + offload buffers + RoPE/config) is loaded ONCE
#     (first job) and STAYS RESIDENT across jobs — this is the residency win
#     available for an offloaded model: subsequent jobs skip the loader rebuild
#     + the on-device offload-buffer reservation.  (Unlike Z-Image, the full
#     weight set is NOT all-resident — it cannot fit — so the win is the loader
#     state, not the weights.)
#   * The Qwen2.5-VL text encoder (~16 GB across 4 shards) is loaded → used →
#     freed PER JOB inside the ENCODE step, exactly like Z-Image's encoder.
#
# step() state machine: ENCODE (per-job, blocking — announced phase="encoding")
#   → LOAD (DiT offloader, once, announced phase="loading") → DENOISE×steps
#   (one CFG dual-forward + Euler update per tick) → DECODE (announced
#   phase="decoding") → done. cancel() makes the next step() return cancelled
#   and frees all per-job tensors.
#
# Size support: 1024x1024 ONLY (the Qwen-Image DiT attention shape N_IMG /
# N_TXT_KEPT / S_POS / S_NEG is comptime-fixed — see qwenimage_sample_cli
# header). steps/cfg/seed ARE honored at runtime here (the denoise loop reads
# them from JobParams; the scheduler sigma table is built per job).
#
# LoRA: NOT supported for Qwen-Image yet (no LoRA path in the model) — a LoRA
# request is rejected at admission so it never silently no-ops.

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
from serenitymojo.models.text_encoder.qwen25vl_encoder import (
    Qwen25VLEncoder, Qwen25VLConfig,
)
from serenitymojo.tokenizer.tokenizer import Qwen3Tokenizer
from serenitymojo.models.dit.qwenimage_dit import QwenImageDitOffloaded
from serenitymojo.models.vae.qwenimage_decoder import QwenImageVaeDecoder
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.random import randn
from serenitymojo.ops.layout import patchify, unpatchify
from serenitymojo.sampling.flow_match import Scheduler, cfg_qwen
from serenitymojo.sampling.sampler_registry import (
    sampler_admission_for_backend, scheduler_admission_for_backend,
)
from serenitymojo.sampling.variation_noise import swarm_variation_noise_chw
from serenitymojo.pipeline.qwenimage_sample_cli import (
    QwenCaps, encode_captions_from_strings,
    QWENIMAGE_DIR, DIT_DIR, VAE_DIR,
    LH, LW, PATCH, N_IMG, N_TXT_KEPT, S_POS, S_NEG, FRAME, FH, FW,
)
from serenitymojo.serve.qwenimage_encode_subprocess import encode_captions_subprocess
from serenitymojo.serve.backend import (
    GenBackend, JobParams, StepResult, reject_unsupported_common_runtime_params,
    reject_unsupported_reference_image_params, reject_unsupported_mask_image_params,
    reject_unsupported_inpaint_conditioning_params,
    reject_unsupported_qwen_edit_conditioning_params,
    reject_unsupported_conditioning_mask_params, reject_unsupported_lanpaint_params,
)

comptime GENPARAMS_TEXT_KEY = "serenity.genparams.v1"


comptime QPHASE_IDLE = 0
comptime QPHASE_ENCODE = 1
comptime QPHASE_LOAD = 2
comptime QPHASE_DENOISE = 3
comptime QPHASE_DECODE = 4


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
        String("echo -n '[qwenimage][vram] ") + tag
        + ": ' && nvidia-smi --query-gpu=memory.used --format=csv,noheader"
    )


def _save_rgb_png_with_text(
    rgb: Tensor, path: String, params_json: String, ctx: DeviceContext
) raises:
    """[1,3,H,W] SIGNED float tensor → 8-bit RGB PNG with the job params in a
    serenity.genparams.v1 tEXt chunk. Quantization math == save_png's
    (_quantize, ValueRange.SIGNED); only the writer differs (tEXt support)."""
    var shape = rgb.shape()
    if len(shape) != 4 or shape[0] != 1 or shape[1] != 3:
        raise Error("qwenimage_backend: expected [1,3,H,W] rgb tensor")
    var height = shape[2]
    var width = shape[3]
    var host = rgb.to_host(ctx)
    var plane = height * width
    if len(host) != 3 * plane:
        raise Error("qwenimage_backend: rgb to_host size mismatch")
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


struct QwenImageBackend(GenBackend, Movable):
    var ctx: DeviceContext

    # ── resident across jobs (offloader handle, loaded once, first job) ──
    # ArcPointer wrappers: QwenImageDitOffloaded / QwenCaps / Scheduler / Tensor
    # are Movable-but-not-Copyable, and List[T] requires T: Copyable (Mojo
    # 1.0.0b1) — Arc is Copyable (refcount), so List[Arc[..]] holds the 0/1.
    var loaded: Bool
    var model: List[ArcPointer[QwenImageDitOffloaded]]  # 0/1 (resident loader)

    # ── per-job state (cleared on done/failed/cancelled) ──
    var active: Bool
    var cancel_flag: Bool
    var phase: Int
    var announced: Bool
    var cur: Int
    var params: JobParams
    var cfg: Float32
    var caps: List[ArcPointer[QwenCaps]]      # 0/1
    var sched: List[ArcPointer[Scheduler]]    # 0/1
    var latent: List[ArcPointer[Tensor]]      # 0/1 (packed)

    def __init__(out self) raises:
        self.ctx = DeviceContext()
        self.loaded = False
        self.model = List[ArcPointer[QwenImageDitOffloaded]]()
        self.active = False
        self.cancel_flag = False
        self.phase = QPHASE_IDLE
        self.announced = False
        self.cur = 0
        self.params = JobParams()
        self.cfg = Float32(4.0)
        self.caps = List[ArcPointer[QwenCaps]]()
        self.sched = List[ArcPointer[Scheduler]]()
        self.latent = List[ArcPointer[Tensor]]()

    def backend_name(self) -> String:
        return String("qwenimage")

    def model_name(self) -> String:
        return String("Qwen-Image")

    def resident_model(self) -> String:
        """Matches the /v1/models scan entry name for the resident checkpoint
        (the qwen-image-2512/ directory entry)."""
        return String("qwen-image-2512") if self.loaded else String("")

    # ── job admission ─────────────────────────────────────────────────────────
    def start(mut self, params: JobParams) raises:
        if self.active:
            raise Error("QwenImageBackend.start: a job is already running")
        reject_unsupported_common_runtime_params(params, String("qwenimage"))
        reject_unsupported_reference_image_params(params, String("qwenimage"))
        reject_unsupported_inpaint_conditioning_params(params, String("qwenimage"))
        reject_unsupported_qwen_edit_conditioning_params(params, String("qwenimage"))
        reject_unsupported_conditioning_mask_params(params, String("qwenimage"))
        reject_unsupported_mask_image_params(params, String("qwenimage"))
        reject_unsupported_lanpaint_params(params, String("qwenimage"))
        var sampler_admission = sampler_admission_for_backend(String("qwenimage"), params.sampler)
        if not sampler_admission.supported:
            raise Error(
                String("qwenimage: unsupported sampler '") + params.sampler
                + String("'; ") + sampler_admission.reason
            )
        var scheduler_admission = scheduler_admission_for_backend(String("qwenimage"), params.scheduler)
        if not scheduler_admission.supported:
            raise Error(
                String("qwenimage: unsupported scheduler '") + params.scheduler
                + String("'; ") + scheduler_admission.reason
            )
        # 1024x1024 only: the DiT attention shape (N_IMG/N_TXT_KEPT/S_POS/S_NEG)
        # is comptime-fixed. Reject other sizes up front (no false advertising).
        if not (params.width == 1024 and params.height == 1024):
            raise Error(
                String("qwenimage: unsupported size ") + String(params.width)
                + "x" + String(params.height)
                + " — only 1024x1024 is served (the Qwen-Image DiT attention"
                + " shape is comptime-fixed; resolution changes need a recompile)"
            )
        if len(params.loras) > 0:
            raise Error(
                "qwenimage: LoRA is not supported for Qwen-Image yet"
                " (no LoRA path in the model); submit without a LoRA"
            )
        if params.init_image.byte_length() > 0:
            raise Error(
                "qwenimage: img2img is not supported for Qwen-Image yet;"
                " submit without an init image"
            )
        self.params = params.copy()
        self.cfg = Float32(params.cfg)
        self.active = True
        self.cancel_flag = False
        self.cur = 0
        self.announced = False
        self.phase = QPHASE_ENCODE

    def cancel(mut self):
        self.cancel_flag = True

    def between_jobs_trim(mut self) raises:
        """F3: reclaim the per-job transient peak (Qwen2.5-VL encoder ~16 GB,
        1024² forward + decode activations) back to the OS via cuMemPoolTrimTo.
        The resident DiT offloader buffers have live suballocations and are NOT
        reclaimed."""
        var before = cu_mem_get_info()
        self.ctx.synchronize()
        cu_mempool_trim_current(0)
        self.ctx.synchronize()
        var after = cu_mem_get_info()
        print("[qwenimage] between-jobs trim: used",
              before.used_bytes() // (1024 * 1024), "->",
              after.used_bytes() // (1024 * 1024), "MiB (reclaimed",
              (before.used_bytes() - after.used_bytes()) // (1024 * 1024), "MiB)")

    # ── per-job prep ───────────────────────────────────────────────────────────
    def _encode(mut self) raises:
        """Qwen2.5-VL encode. Run the ~16 GB encoder in a fork+execv CHILD process
        so its VRAM is reclaimed by process death (in-process it gets stuck in this
        worker's CUDA pool, fragmented around the resident DiT offloader buffers —
        cu_mempool_trim reclaims ~0, MEASURED, leaving ~2 GB headroom so the block
        prefetch can't overlap). The produced caps are byte-identical to the
        in-process path (raw-byte cap_cache round-trip). Falls back to in-process
        encode on any host that does not route `encode-child` or on subprocess
        failure (see serve/qwenimage_encode_subprocess.mojo)."""
        var caps = encode_captions_subprocess(
            self.params.prompt, self.params.negative, self.ctx
        )
        _print_vram("after text encode (encoder child reaped)")
        self.caps = List[ArcPointer[QwenCaps]]()
        self.caps.append(ArcPointer(caps^))

    def _load_model(mut self) raises:
        """Load the Qwen-Image DiT offloader handle (once; stays resident)."""
        if self.loaded:
            return
        _print_vram("before DiT offloader load")
        print("[qwenimage] loading Qwen-Image MMDiT offloader from", DIT_DIR)
        self.model = List[ArcPointer[QwenImageDitOffloaded]]()
        self.model.append(ArcPointer(QwenImageDitOffloaded.load(DIT_DIR, self.ctx)))
        self.loaded = True
        _print_vram("after DiT offloader load (resident)")

    def _prepare_job(mut self) raises:
        """Scheduler (honors steps) + seeded initial packed latent (honors seed)."""
        self.sched = List[ArcPointer[Scheduler]]()
        self.sched.append(ArcPointer(Scheduler.qwen(self.params.steps, Float32(N_IMG))))
        var nchw_shape = [1, 16, LH, LW]
        var noise = randn(nchw_shape.copy(), UInt64(self.params.seed), STDtype.BF16, self.ctx)
        if self.params.variation_strength > 0.0:
            var vnoise = randn(
                nchw_shape.copy(),
                UInt64(self.params.variation_seed + self.params.image_index),
                STDtype.BF16,
                self.ctx,
            )
            var base_h = noise.to_host(self.ctx)
            var var_h = vnoise.to_host(self.ctx)
            var blended = swarm_variation_noise_chw(
                base_h, var_h, 16, LH, LW, self.params.variation_strength
            )
            noise = Tensor.from_host(blended, nchw_shape^, STDtype.BF16, self.ctx)
        var packed = patchify(noise, PATCH, self.ctx)
        self.latent = List[ArcPointer[Tensor]]()
        self.latent.append(ArcPointer(packed^))
        print(
            "[qwenimage] job", self.params.job_id, ":", self.params.steps,
            "steps, cfg", self.cfg, "seed", self.params.seed,
            "size", self.params.width, "x", self.params.height,
        )

    # ── one denoise step (CFG dual forward + Euler) ────────────────────────────
    def _denoise_one(mut self) raises:
        var i = self.cur
        var sigmas = self.sched[0][].sigmas()
        var preds = self.model[0][].forward_cfg_mixed_text[
            N_IMG, N_TXT_KEPT, S_POS, N_TXT_KEPT, S_NEG
        ](
            self.latent[0][], self.caps[0][].pos, self.caps[0][].neg, sigmas[i],
            self.caps[0][].real_pos, self.caps[0][].real_neg,
            FRAME, FH, FW, self.ctx,
        )
        var pred = cfg_qwen(preds.pos, preds.neg, self.cfg, self.ctx)
        var x_new = self.sched[0][].step(self.latent[0][], pred, i, self.ctx)
        self.latent = List[ArcPointer[Tensor]]()
        self.latent.append(ArcPointer(x_new^))

    # ── final decode + PNG(tEXt) ──────────────────────────────────────────────
    def _decode_and_save(mut self) raises -> String:
        var png_path = self.params.out_dir + "/" + self.params.job_id + ".png"
        var latent = unpatchify(self.latent[0][], 16, LH, LW, PATCH, self.ctx)
        latent = cast_tensor(latent, STDtype.BF16, self.ctx)
        # Per-job conditioning is dead weight at decode; free before the decoder.
        self.caps = List[ArcPointer[QwenCaps]]()
        self.sched = List[ArcPointer[Scheduler]]()
        self.latent = List[ArcPointer[Tensor]]()
        print("[qwenimage] loading VAE decoder + decode")
        var vae = QwenImageVaeDecoder[LH, LW].load(VAE_DIR, self.ctx)
        var img = vae.decode(latent, self.ctx)
        _save_rgb_png_with_text(img, png_path, self.params.params_json, self.ctx)
        return png_path

    def _clear_job(mut self):
        self.active = False
        self.phase = QPHASE_IDLE
        self.cur = 0
        self.cancel_flag = False
        self.announced = False
        self.caps = List[ArcPointer[QwenCaps]]()
        self.sched = List[ArcPointer[Scheduler]]()
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
            if self.phase == QPHASE_ENCODE:
                if not self.announced:
                    # announce BEFORE the long blocking encode tick (per-job
                    # Qwen2.5-VL load+forward).
                    self.announced = True
                    r.step = 0
                    r.phase = String("encoding")
                    return r^
                self._encode()
                self.announced = False
                self.phase = QPHASE_LOAD
                r.step = 0
                return r^
            if self.phase == QPHASE_LOAD:
                if not self.loaded:
                    if not self.announced:
                        self.announced = True
                        r.step = 0
                        r.phase = String("loading")
                        return r^
                    self._load_model()
                    self.announced = False
                self._prepare_job()
                self.phase = QPHASE_DENOISE
                r.step = 0
                return r^
            if self.phase == QPHASE_DENOISE:
                self._denoise_one()
                self.cur += 1
                r.step = self.cur
                if self.cur >= self.params.steps:
                    self.phase = QPHASE_DECODE
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
