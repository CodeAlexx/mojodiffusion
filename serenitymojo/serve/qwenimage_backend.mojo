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
#   * The Qwen-Image MMDiT uses the installed Serenity raw-E4M3 safetensors.
#     All 60 blocks are copied into pinned host memory before sampling step 0,
#     then staged/dequantized from RAM. The checkpoint mmap is never consulted
#     during denoise. The worker fails closed if the complete host store cannot
#     be built.
#   * The Qwen2.5-VL text encoder (~16 GB across 4 shards) is loaded → used →
#     freed PER JOB inside the ENCODE step, exactly like Z-Image's encoder.
#
# step() state machine: ENCODE (per-job, blocking — announced phase="encoding")
#   → LOAD (DiT offloader, once, announced phase="loading") → DENOISE×steps
#   (one CFG dual-forward + Euler update per tick) → DECODE (announced
#   phase="decoding") → done. cancel() makes the next step() return cancelled
#   and frees all per-job tensors.
#
# Size support: the shared seven-arm 1024px-area image aspect ladder. Each
# Qwen-Image DiT N_IMG / S_POS / S_NEG and VAE decode shape remains comptime-
# fixed inside its selected arm. steps/cfg/seed are honored at runtime.
#
# LoRA: one sparse canonical Qwen PEFT/Serenity adapter is loaded through the
# existing parity-gated 60-block Qwen LoRA device math. Present projections are
# applied exactly; missing projections remain absent and incompatible shapes
# fail loud.

from std.ffi import external_call
from max.gpu.host import DeviceContext
from std.memory import alloc, ArcPointer, UnsafePointer
from std.time import sleep

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
from serenitymojo.models.dit.qwenimage_dit import (
    QwenImageDitOffloaded,
    QwenImageCfgPreds,
)
from serenitymojo.models.qwenimage.qwenimage_stack_lora import (
    QwenLoraDeviceSet,
    load_qwenimage_lora_device_set,
)
from serenitymojo.models.vae.qwenimage_tiled_decode import qwenimage_tiled_decode
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.random import randn
from serenitymojo.ops.layout import patchify, unpatchify
from serenitymojo.sampling.flow_match import Scheduler, cfg_qwen_device
from serenitymojo.sampling.dpmpp_2m import (
    MultistepHistory,
    denoised_from_velocity,
    dpmpp_2m_step,
    lambda_from_sigma_f64,
)
from serenitymojo.sampling.sampler_registry import (
    sampler_admission_for_backend, scheduler_admission_for_backend,
)
from serenitymojo.sampling.variation_noise import variation_noise_chw
from serenitymojo.pipeline.qwenimage_sample_cli import (
    QwenCaps, encode_captions_from_strings,
    VAE_DIR,
    PATCH, N_TXT_KEPT,
)
from serenitymojo.training.aspect_buckets import (
    DEFAULT_ASPECT_LADDER_LEN, DEFAULT_ASPECT_LADDER_X100,
    aspect_lat_h_units, aspect_lat_w_units,
)
from serenitymojo.io.cap_cache import load_tensor_bin
from serenitymojo.serve.proc_ipc import (
    build_argv, cstr, sys_execv, sys__exit, sys_waitpid, proc_kill_wait,
    SELF_EXE, SIGKILL, WNOHANG,
)
from serenitymojo.serve.qwenimage_encode_subprocess import _read_meta
from net.syscalls import sys_fork, errno_str
from serenitymojo.serve.backend import (
    GenBackend, JobParams, StepResult, reject_unsupported_common_runtime_params,
    reject_unsupported_reference_image_params, reject_unsupported_mask_image_params,
    reject_unsupported_inpaint_conditioning_params,
    reject_unsupported_qwen_edit_conditioning_params,
    reject_unsupported_conditioning_mask_params, reject_unsupported_lanpaint_params,
    advanced_sampling_params_set,
)
from serenitymojo.serve.product_manifest import (
    json_escape,
    write_text_file,
)

comptime GENPARAMS_TEXT_KEY = "serenity.genparams.v1"

# MJ-1058: Qwen-Image gate-recipe CFG (qwenimage_pipeline_1024_multistep.mojo:63,
# CFG=4.0). Applied ONLY for the degenerate cfg<=0 input — see the note at the
# apply site in start(). The frozen wire contract carries no "unset" sentinel, so
# an omitted cfg arrives indistinguishable from a user who chose the JobParams
# global default (4.5); we cannot promote an omitted 4.5 to 4.0 without clobbering
# a deliberate 4.5, so only cfg<=0 gets the gate default (do NOT touch JobParams).
comptime QWENIMAGE_DEFAULT_CFG = Float32(4.0)
comptime QWENIMAGE_EDGE_UNITS = 16
comptime QWENIMAGE_FP8_CHECKPOINT = (
    "/home/alex/.serenity/models/checkpoints/qwen_image_fp8_e4m3fn.safetensors"
)


comptime QPHASE_IDLE = 0
comptime QPHASE_ENCODE = 1
comptime QPHASE_LOAD = 2
comptime QPHASE_DENOISE = 3
comptime QPHASE_DECODE = 4


def _qwenimage_shape_supported(width: Int, height: Int) -> Bool:
    comptime for bi in range(DEFAULT_ASPECT_LADDER_LEN):
        comptime X100_BI = DEFAULT_ASPECT_LADDER_X100[bi]
        comptime LH_BI = aspect_lat_h_units(X100_BI, QWENIMAGE_EDGE_UNITS)
        comptime LW_BI = aspect_lat_w_units(X100_BI, QWENIMAGE_EDGE_UNITS)
        if width == LW_BI * 8 and height == LH_BI * 8:
            return True
    return False


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


def _reject_qwen_unsupported_runtime_params(params: JobParams) raises:
    if params.cfg_override >= 0.0:
        raise Error("qwenimage: cfg_override is not supported yet")
    if (
        params.cfg_override_start_percent != 0.0
        or params.cfg_override_end_percent != 1.0
    ):
        raise Error("qwenimage: cfg_override percent window is not supported yet")
    if params.sigma_shift != 3.0:
        raise Error("qwenimage: sigma_shift override is not supported yet")
    if params.creativity != 0.5:
        raise Error("qwenimage: creativity/partial denoise is not supported yet")
    if (
        params.sample_caps_pos.byte_length() > 0
        or params.sample_caps_neg.byte_length() > 0
    ):
        raise Error("qwenimage: pre-encoded sample caps are not supported by the live Qwen text encoder path")
    var advanced = advanced_sampling_params_set(params)
    if len(advanced) > 0:
        raise Error(
            String("qwenimage: advanced sampling parameter '")
            + advanced[0]
            + String("' is not supported yet")
        )


# ── low-VRAM encode-child fork (16 GB fit) ────────────────────────────────────
# serve/qwenimage_encode_subprocess.encode_captions_subprocess preflights at the
# RESIDENT encoder-child footprint (17400 MiB) — a bar a 16 GB card can NEVER
# clear (total VRAM < 17.4 GB; measured "free VRAM 14770 MiB < required 17400
# MiB"), so the qwenimage worker was permanently blocked on this box. The CHILD
# now routes itself: encode_captions_from_strings picks the LAYER-STREAMED
# Qwen2.5-VL encoder (qwenimage_qwen25vl_streamed, peak ~2.7 GB, parity-gated
# DiT-consumed cos 0.999876 vs the bf16 torch oracle) whenever device-global
# free is below the resident need, and keeps the resident whole-load on 24 GB
# boxes. So the PARENT-side fork gate here only has to cover the STREAMED
# child's peak + CUDA context. Same fork/reap/read-back protocol and strictness
# as the serve module (raise, never in-parent encode), with the corrected gate.
comptime _ENCODE_CHILD_TIMEOUT_S = 300.0
comptime _ENCODE_POLL_S = 0.05
comptime _ENCODE_CHILD_MIN_FREE_BYTES_LOWVRAM = Int(3600) * 1024 * 1024


def _getpid() -> Int:
    return Int(external_call["getpid", Int32]())


def _encode_captions_child_lowvram(
    prompt: String, negative: String, ctx: DeviceContext
) raises -> QwenCaps:
    """fork+execv `serenity_worker_qwenimage encode-child` with the STREAMED-fit
    preflight, blocking-reap it (VRAM released by process death), read back the
    BF16 caps. The child body (serve/qwenimage_encode_subprocess.encode_child_run
    → encode_captions_from_strings) auto-selects streamed vs resident encode by
    free VRAM. Strict like the serve module: any failure raises."""
    var prefix = String("/tmp/serenity_qwenimage_caps_") + String(_getpid())
    var pos_path = prefix + String(".pos.bin")
    var neg_path = prefix + String(".neg.bin")
    var meta_path = prefix + String(".meta")

    var free_bytes = cu_mem_get_info().free_bytes
    if free_bytes < _ENCODE_CHILD_MIN_FREE_BYTES_LOWVRAM:
        raise Error(
            String("qwenimage encoder child preflight failed: free VRAM ")
            + String(free_bytes // (1024 * 1024))
            + String(" MiB < required ")
            + String(_ENCODE_CHILD_MIN_FREE_BYTES_LOWVRAM // (1024 * 1024))
            + String(" MiB (streamed-encoder child)")
        )

    # argv + execv path built BEFORE fork (no allocation between fork and execv).
    var args = List[String]()
    args.append(SELF_EXE)                  # argv[0]
    args.append(String("encode-child"))
    args.append(prefix)
    args.append(prompt)
    args.append(negative)
    var argv = build_argv(args)
    var path = cstr(SELF_EXE)

    print("[qwenimage] cache MISS → fork encoder child (parent pid", _getpid(),
          ", streamed-fit preflight)")
    var pid = sys_fork()
    if pid == 0:
        # CHILD: async-signal-safe only, then execv into a fresh image.
        _ = sys_execv(path, argv)
        sys__exit(127)                     # execv failed
    if pid < 0:
        raise Error(String("qwenimage encoder child fork failed: ") + errno_str())

    # PARENT: bounded WNOHANG reap (hang backstop). Blocking-reap once it exits
    # so the OS has released the child's VRAM before we load the caps.
    var st = alloc[Int32](1)
    var stp = rebind[UnsafePointer[Int32, MutExternalOrigin]](st)
    var waited = 0.0
    var reaped = Int32(0)
    while waited < _ENCODE_CHILD_TIMEOUT_S:
        reaped = sys_waitpid(pid, stp, WNOHANG)
        if reaped == pid:
            break
        if reaped < 0:
            break
        sleep(_ENCODE_POLL_S)
        waited += _ENCODE_POLL_S
    var status = Int(st[0])
    st.free()

    if reaped != pid:
        proc_kill_wait(pid, SIGKILL)
        raise Error("qwenimage encoder child timed out or waitpid failed")

    var exited_ok = (status & 0x7F) == 0 and ((status >> 8) & 0xFF) == 0
    if not exited_ok:
        raise Error(
            String("qwenimage encoder child abnormal exit status ")
            + String(status)
        )

    try:
        var meta = _read_meta(meta_path)
        var pos = load_tensor_bin(pos_path, ctx)
        var neg = load_tensor_bin(neg_path, ctx)
        print("[qwenimage] encoder child reaped → caps loaded (encoder VRAM reclaimed)")
        return QwenCaps(pos^, neg^, meta[0], meta[1])
    except e:
        raise Error(String("qwenimage encoder caps read-back failed: ") + String(e))


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
    # Keyed (prompt \x1f negative) conditioning cache, capacity 4 drop-oldest
    # — this backend previously printed "cache MISS -> fork encoder child"
    # with NO cache behind it; every repeat prompt re-paid the ~16 GB child
    # encoder fork (3-25 s).
    var cap_cache_keys: List[String]
    var cap_cache: List[ArcPointer[QwenCaps]]
    var sched: List[ArcPointer[Scheduler]]    # 0/1
    var latent: List[ArcPointer[Tensor]]      # 0/1 (packed)
    var lora: List[ArcPointer[QwenLoraDeviceSet]]  # 0/1, per job
    var lora_target_count: Int
    var executed_sampler: String
    var executed_scheduler: String
    var dpmpp_history: MultistepHistory
    var dpmpp_history_final_len: Int
    var dpmpp_update_steps: Int
    var dpmpp_second_order_steps: Int

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
        self.cap_cache_keys = List[String]()
        self.cap_cache = List[ArcPointer[QwenCaps]]()
        self.sched = List[ArcPointer[Scheduler]]()
        self.latent = List[ArcPointer[Tensor]]()
        self.lora = List[ArcPointer[QwenLoraDeviceSet]]()
        self.lora_target_count = 0
        self.executed_sampler = String("qwenimage_flowmatch_euler")
        self.executed_scheduler = String("qwenimage_simple_flowmatch")
        self.dpmpp_history = MultistepHistory(1)
        self.dpmpp_history_final_len = 0
        self.dpmpp_update_steps = 0
        self.dpmpp_second_order_steps = 0

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
        _reject_qwen_unsupported_runtime_params(params)
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
        # Runtime request dispatches to a concrete COMPTIME 1024px-area aspect
        # arm. Each arm uses the same Qwen model/conditioning math; only latent,
        # RoPE, attention, and tiled-decode geometry are monomorphized.
        if not _qwenimage_shape_supported(params.width, params.height):
            raise Error(
                String("qwenimage: unsupported size ") + String(params.width)
                + "x" + String(params.height)
                + " — choose a compiled Qwen 1024px-area aspect bucket"
            )
        if len(params.loras) > 1:
            raise Error(
                "qwenimage: this runtime currently supports one compatible"
                " Qwen PEFT/Serenity adapter at a time"
            )
        if params.init_image.byte_length() > 0:
            raise Error(
                "qwenimage: img2img is not supported for Qwen-Image yet;"
                " submit without an init image"
            )
        self.params = params.copy()
        # MJ-1058: backfill the gate-recipe CFG only for the degenerate cfg<=0
        # input (see QWENIMAGE_DEFAULT_CFG note). Mirrors sensenova_backend's
        # cfg-default idiom.
        self.cfg = Float32(params.cfg) if params.cfg > 0.0 else QWENIMAGE_DEFAULT_CFG
        self.executed_sampler = sampler_admission.executed.copy()
        self.executed_scheduler = scheduler_admission.executed.copy()
        self.dpmpp_history = MultistepHistory(1)
        self.dpmpp_history_final_len = 0
        self.dpmpp_update_steps = 0
        self.dpmpp_second_order_steps = 0
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
        failure (see serve/qwenimage_encode_subprocess.mojo). 16 GB fit: the
        fork preflight is the STREAMED-encoder child bar (~3.6 GB) — the child
        itself picks streamed vs resident encode by free VRAM (see
        _encode_captions_child_lowvram above)."""
        var want_key = self.params.prompt + String("\x1f") + self.params.negative
        for slot in range(len(self.cap_cache_keys)):
            if self.cap_cache_keys[slot] != want_key:
                continue
            print("[qwenimage] conditioning cache HIT (slot", slot, "of",
                  len(self.cap_cache_keys), ") — skipping encoder child")
            ref cached = self.cap_cache[slot][]
            self.caps = List[ArcPointer[QwenCaps]]()
            self.caps.append(ArcPointer(QwenCaps(
                cached.pos.clone(self.ctx), cached.neg.clone(self.ctx),
                cached.real_pos, cached.real_neg,
            )))
            return
        var caps = _encode_captions_child_lowvram(
            self.params.prompt, self.params.negative, self.ctx
        )
        _print_vram("after text encode (encoder child reaped)")
        if len(self.cap_cache_keys) >= 4:
            _ = self.cap_cache_keys.pop(0)
            _ = self.cap_cache.pop(0)
        self.cap_cache_keys.append(want_key.copy())
        self.cap_cache.append(ArcPointer(QwenCaps(
            caps.pos.clone(self.ctx), caps.neg.clone(self.ctx),
            caps.real_pos, caps.real_neg,
        )))
        self.caps = List[ArcPointer[QwenCaps]]()
        self.caps.append(ArcPointer(caps^))

    def _load_model(mut self) raises:
        """Load the no-disk raw-FP8 host store and the current job's LoRA."""
        if not self.loaded:
            _print_vram("before DiT offloader load")
            print(
                "[qwenimage] loading Qwen-Image raw-FP8 host store from",
                QWENIMAGE_FP8_CHECKPOINT,
            )
            self.model = List[ArcPointer[QwenImageDitOffloaded]]()
            self.model.append(ArcPointer(
                QwenImageDitOffloaded.load_fp8_host_resident(
                    String(QWENIMAGE_FP8_CHECKPOINT), self.ctx
                )
            ))
            print("[qwenimage] 60/60 denoiser blocks resident in host RAM")
            self.loaded = True
            _print_vram("after DiT offloader load (resident)")

        self.lora = List[ArcPointer[QwenLoraDeviceSet]]()
        self.lora_target_count = 0
        if len(self.params.loras) == 1:
            print(
                "[qwenimage] loading sparse canonical adapter:",
                self.params.loras[0].name,
                "weight",
                self.params.loras[0].weight,
            )
            var loaded_lora = load_qwenimage_lora_device_set(
                self.params.loras[0].name,
                Float32(self.params.loras[0].weight),
                self.ctx,
            )
            self.lora_target_count = loaded_lora.target_count
            self.lora.append(ArcPointer(loaded_lora^))
            print(
                "[qwenimage] loaded",
                self.lora_target_count,
                "compatible projection factors",
            )

    def _prepare_job_shape[LH_: Int, LW_: Int, N_IMG_: Int](mut self) raises:
        """Scheduler (honors steps) + seeded initial packed latent (honors seed)."""
        self.sched = List[ArcPointer[Scheduler]]()
        self.sched.append(ArcPointer(Scheduler.qwen(self.params.steps, Float32(N_IMG_))))
        var nchw_shape: List[Int] = [1, 16, LH_, LW_]
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
            var blended = variation_noise_chw(
                base_h, var_h, 16, LH_, LW_, self.params.variation_strength
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

    def _prepare_job(mut self) raises:
        comptime for bi in range(DEFAULT_ASPECT_LADDER_LEN):
            comptime X100_BI = DEFAULT_ASPECT_LADDER_X100[bi]
            comptime LH_BI = aspect_lat_h_units(X100_BI, QWENIMAGE_EDGE_UNITS)
            comptime LW_BI = aspect_lat_w_units(X100_BI, QWENIMAGE_EDGE_UNITS)
            comptime N_IMG_BI = (LH_BI // PATCH) * (LW_BI // PATCH)
            if self.params.width == LW_BI * 8 and self.params.height == LH_BI * 8:
                self._prepare_job_shape[LH_BI, LW_BI, N_IMG_BI]()
                return
        raise Error("qwenimage: admitted prepare shape was not compiled")

    # ── one denoise step (CFG dual forward + Euler) ────────────────────────────
    def _denoise_one_shape[LH_: Int, LW_: Int, N_IMG_: Int](mut self) raises:
        var i = self.cur
        var sigmas = self.sched[0][].sigmas()
        comptime S_BI = N_IMG_ + N_TXT_KEPT
        var preds: QwenImageCfgPreds
        if len(self.lora) == 1:
            preds = self.model[0][].forward_cfg_mixed_text[
                N_IMG_, N_TXT_KEPT, S_BI, N_TXT_KEPT, S_BI
            ](
                self.latent[0][], self.caps[0][].pos, self.caps[0][].neg, sigmas[i],
                self.caps[0][].real_pos, self.caps[0][].real_neg,
                1, LH_ // PATCH, LW_ // PATCH, self.ctx,
                Optional[QwenLoraDeviceSet](self.lora[0][].copy()),
            )
        else:
            preds = self.model[0][].forward_cfg_mixed_text[
                N_IMG_, N_TXT_KEPT, S_BI, N_TXT_KEPT, S_BI
            ](
                self.latent[0][], self.caps[0][].pos, self.caps[0][].neg, sigmas[i],
                self.caps[0][].real_pos, self.caps[0][].real_neg,
                1, LH_ // PATCH, LW_ // PATCH, self.ctx,
            )
        var pred = cfg_qwen_device(preds.pos, preds.neg, self.cfg, self.ctx)
        var x_new: Tensor
        if self.executed_sampler == "dpmpp_2m":
            var sigma_next = sigmas[i + 1]
            var latent_f32 = cast_tensor(
                self.latent[0][], STDtype.F32, self.ctx
            )
            var pred_f32 = cast_tensor(pred, STDtype.F32, self.ctx)
            var denoised = denoised_from_velocity(
                latent_f32, pred_f32, sigmas[i], self.ctx
            )
            if not self.dpmpp_history.is_empty():
                self.dpmpp_second_order_steps += 1
            var stepped = dpmpp_2m_step(
                latent_f32,
                denoised,
                sigmas[i],
                sigma_next,
                self.dpmpp_history,
                self.ctx,
            )
            self.dpmpp_history.push(
                denoised^,
                lambda_from_sigma_f64(Float64(sigmas[i])),
            )
            self.dpmpp_update_steps += 1
            x_new = cast_tensor(stepped, STDtype.BF16, self.ctx)
        else:
            x_new = self.sched[0][].step(
                self.latent[0][], pred, i, self.ctx
            )
        self.latent = List[ArcPointer[Tensor]]()
        self.latent.append(ArcPointer(x_new^))

    def _denoise_one(mut self) raises:
        comptime for bi in range(DEFAULT_ASPECT_LADDER_LEN):
            comptime X100_BI = DEFAULT_ASPECT_LADDER_X100[bi]
            comptime LH_BI = aspect_lat_h_units(X100_BI, QWENIMAGE_EDGE_UNITS)
            comptime LW_BI = aspect_lat_w_units(X100_BI, QWENIMAGE_EDGE_UNITS)
            comptime N_IMG_BI = (LH_BI // PATCH) * (LW_BI // PATCH)
            if self.params.width == LW_BI * 8 and self.params.height == LH_BI * 8:
                self._denoise_one_shape[LH_BI, LW_BI, N_IMG_BI]()
                return
        raise Error("qwenimage: admitted denoise shape was not compiled")

    # ── final decode + PNG(tEXt) ──────────────────────────────────────────────
    def _decode_and_save_shape[LH_: Int, LW_: Int](mut self) raises -> String:
        var png_path = self.params.out_dir + "/" + self.params.job_id + ".png"
        var latent = unpatchify(self.latent[0][], 16, LH_, LW_, PATCH, self.ctx)
        latent = cast_tensor(latent, STDtype.BF16, self.ctx)
        # Per-job conditioning is dead weight at decode; free before the decoder.
        self.caps = List[ArcPointer[QwenCaps]]()
        self.sched = List[ArcPointer[Scheduler]]()
        self.latent = List[ArcPointer[Tensor]]()
        self.lora = List[ArcPointer[QwenLoraDeviceSet]]()
        self.dpmpp_history_final_len = self.dpmpp_history.len()
        self.dpmpp_history = MultistepHistory(1)
        print("[qwenimage] tiled VAE decode (3x3 overlap) + save")
        var img = qwenimage_tiled_decode[LH_, LW_](latent, VAE_DIR, self.ctx)
        _save_rgb_png_with_text(img, png_path, self.params.params_json, self.ctx)
        return png_path

    def _decode_and_save(mut self) raises -> String:
        comptime for bi in range(DEFAULT_ASPECT_LADDER_LEN):
            comptime X100_BI = DEFAULT_ASPECT_LADDER_X100[bi]
            comptime LH_BI = aspect_lat_h_units(X100_BI, QWENIMAGE_EDGE_UNITS)
            comptime LW_BI = aspect_lat_w_units(X100_BI, QWENIMAGE_EDGE_UNITS)
            if self.params.width == LW_BI * 8 and self.params.height == LH_BI * 8:
                return self._decode_and_save_shape[LH_BI, LW_BI]()
        raise Error("qwenimage: admitted decode shape was not compiled")

    def _write_result_manifest(self, png_path: String) raises -> String:
        var manifest_path = png_path + String(".qwenimage_daemon_result.json")
        var content = String("{\n")
        content += String('  "schema":"serenity.qwenimage.daemon_result.v1",\n')
        content += String('  "backend":"qwenimage_daemon",\n')
        content += String('  "model":"qwen-image-2512",\n')
        content += String('  "accepted_sampler_parity":false,\n')
        content += String('  "run_identity":{\n')
        content += String('    "job_id":"') + json_escape(self.params.job_id) + String('",\n')
        content += String('    "prompt":"') + json_escape(self.params.prompt) + String('",\n')
        content += String('    "negative":"') + json_escape(self.params.negative) + String('",\n')
        content += String('    "seed":') + String(self.params.seed) + String(",\n")
        content += String('    "resolution":{"width":') + String(self.params.width)
        content += String(',"height":') + String(self.params.height) + String("},\n")
        content += String('    "steps":') + String(self.params.steps) + String(",\n")
        content += String('    "cfg":') + String(self.cfg) + String(",\n")
        content += String('    "requested_sampler":"') + json_escape(self.params.sampler) + String('",\n')
        content += String('    "requested_scheduler":"') + json_escape(self.params.scheduler) + String('",\n')
        content += String('    "executed_sampler":"') + json_escape(self.executed_sampler) + String('",\n')
        content += String('    "executed_scheduler":"') + json_escape(self.executed_scheduler) + String('",\n')
        content += String('    "lora_count":') + String(len(self.params.loras)) + String(",\n")
        content += String('    "loaded_lora":"') + json_escape(
            self.params.loras[0].name if len(self.params.loras) == 1 else String("")
        ) + String('",\n')
        content += String('    "loaded_lora_weight":') + String(
            self.params.loras[0].weight if len(self.params.loras) == 1 else Float64(0.0)
        ) + String(",\n")
        content += String('    "lora_target_count":') + String(self.lora_target_count) + String(",\n")
        content += String('    "sampler_trace":{"history_capacity":1,"history_final_len":')
        content += String(self.dpmpp_history_final_len) + String(',"dpmpp_update_steps":')
        content += String(self.dpmpp_update_steps) + String(',"dpmpp_second_order_steps":')
        content += String(self.dpmpp_second_order_steps) + String("},\n")
        content += String('    "dtype":"bf16_mmdit_bf16_latent"\n')
        content += String("  },\n")
        content += String('  "output_png":"') + json_escape(png_path) + String('"\n')
        content += String("}\n")
        write_text_file(manifest_path, content)
        return manifest_path

    def _clear_job(mut self):
        self.active = False
        self.phase = QPHASE_IDLE
        self.cur = 0
        self.cancel_flag = False
        self.announced = False
        self.caps = List[ArcPointer[QwenCaps]]()
        self.sched = List[ArcPointer[Scheduler]]()
        self.latent = List[ArcPointer[Tensor]]()
        self.lora = List[ArcPointer[QwenLoraDeviceSet]]()
        self.lora_target_count = 0
        self.dpmpp_history = MultistepHistory(1)
        self.dpmpp_history_final_len = 0
        self.dpmpp_update_steps = 0
        self.dpmpp_second_order_steps = 0

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
                r.phase = String("sampling")
                return r^
            if self.phase == QPHASE_DENOISE:
                self._denoise_one()
                self.cur += 1
                r.step = self.cur
                r.phase = String("sampling")
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
            var manifest = self._write_result_manifest(path)
            print("[qwenimage][manifest] saved:", manifest)
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
