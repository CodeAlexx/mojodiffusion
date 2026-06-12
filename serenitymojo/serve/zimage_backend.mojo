# serenitymojo.serve.zimage_backend — the real Z-Image GenBackend.
#
# Wraps the VERIFIED serenitymojo/pipeline/zimage_generate.mojo stages behind
# the pull-based GenBackend seam (backend.mojo). EVERY numeric convention is
# reused from zimage_generate (its helpers are imported, NOT re-derived):
#   tokenizer → Qwen3-4B layer-34 encode → CFG dual-forward denoise
#   (rectified-flow Euler, shift=6.0 sigmas) → Z-Image VAE decode → PNG SIGNED.
#
# Residency model (24 GB GPU):
#   * DiT weights (aux + nr/cr/main blocks + final linear) and BOTH VAE decoder
#     specializations (64²- and 128²-latent) are loaded ONCE (first job) and
#     stay resident across jobs.
#   * The Qwen3-4B text encoder (~7.5 GB bf16) is loaded → used → freed PER JOB
#     inside the ENCODE step (encode_captions_fixed frees it on return), because
#     DiT-resident (~12 GB) + encoder + dual-forward activations does fit, but
#     keeping the encoder resident too leaves no headroom for denoise
#     activations at S=4352. Per-job encoder reload is served from page cache.
#
# step() state machine: LOAD(chunked, once) → ENCODE → DENOISE×steps →
# DECODE → done. One denoise step (CFG dual forward + Euler update) per step()
# call; the cold LOAD is split into bounded chunks (F6) and the long ENCODE /
# DECODE ticks are announced via StepResult.phase; cancel() makes the next
# step() return cancelled and frees all per-job tensors.
#
# Size support: 512x512 ONLY this phase (HL=WL=64). 1024x1024 (HL=WL=128) is
# wired but DISABLED at start() pending Phase-4 VRAM work: the whole-frame
# 128-grid decode peaks over 24 GB beside the resident DiT, and the device
# pool retains ~8 GB between jobs (skeptic F2/F3; SERENITYUI_TODO.md).
#
# LoRA: forward overlay only — never fused. At most ONE overlay per job. Two
# file formats load: the trainer resume format (load_zimage_lora_main_only_
# resume) and the comfy/kohya export (musubi-tuner networks.lora_zimage,
# lora_unet_layers_* keys with fused qkv → exact split, F5). Bare names
# resolve against the scanner's loras dir (F4).

from std.ffi import external_call
from std.gpu.host import DeviceContext
from std.memory import alloc, ArcPointer
from std.time import perf_counter_ns

from image.buffer import Image
from image.png import encode_png_with_text
from image.transform import resize_bilinear

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.ffi import BytePtr
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.image.png import _quantize, ValueRange
from serenitymojo.models.vae.zimage_decoder import ZImageDecoder
from serenitymojo.models.vae.zimage_encoder import (
    ZImageVaeEncoder, ZIMG_SCALING, ZIMG_SHIFT,
)
from serenitymojo.models.zimage.weights import (
    ZImageBlockWeights, load_zimage_block_weights_prefixed_mixed,
)
from serenitymojo.models.zimage.zimage_stack_lora import (
    ZImageLoraSet, ZImageLoraDeviceSet, load_zimage_lora_main_only_resume,
    load_zimage_lora_main_only_comfy, zimage_lora_file_is_comfy,
    build_zimage_zero_lora_device_set, zimage_lora_set_to_device,
    merge_zimage_lora_sets_for_inference,
)
from serenitymojo.models.zimage.real_weights import (
    ZImageRealAux, load_zimage_real_aux, build_rope, build_positions,
)
from serenitymojo.offload.vmm_cuda import cu_mempool_trim_current, cu_mem_get_info
from serenitymojo.ops.tensor_algebra import mul_scalar, add
from serenitymojo.pipeline.zimage_generate import (
    encode_captions_fixed, gaussian_noise, _cast, _cap_seq_with_pad,
    _unified_positions, _cfg_pred_overlay, _build_sigmas, _path_exists,
    _json_escape, _write_text_file, _peak_vram_mib,
    TRANSFORMER, VAE_DIR,
    H, Dh, D, F, PATCH, CAPLEN_MAX, ROPE_THETA, AXIS0, AXIS1, AXIS2,
    NUM_NR, NUM_CR, MAIN_DEPTH, RANK, ALPHA,
)
from serenitymojo.sampling.sampler_registry import (
    sampler_admission_for_backend, scheduler_admission_for_backend,
)
from serenitymojo.sampling.dpmpp_2m import (
    MultistepHistory, denoised_from_velocity, dpmpp_2m_step,
    lambda_from_sigma_f64,
)
from serenitymojo.sampling.unipc import UniPcMultistepScheduler
from serenitymojo.sampling.variation_noise import swarm_variation_noise_chw
from serenitymojo.serve.backend import (
    GenBackend, JobParams, StepResult, reject_unsupported_common_runtime_params,
)
from serenitymojo.serve.image_io import decode_image_any, image_to_signed_nchw
from serenitymojo.serve.model_scan import LORAS_DIR

comptime TArc = ArcPointer[Tensor]

comptime PHASE_IDLE = 0
comptime PHASE_LOAD = 1
comptime PHASE_ENCODE = 2
comptime PHASE_DENOISE = 3
comptime PHASE_DECODE = 4

comptime GENPARAMS_TEXT_KEY = "serenity.genparams.v1"


def _json_bool(v: Bool) -> String:
    if v:
        return String("true")
    return String("false")


def _float32_list_json(vals: List[Float32]) -> String:
    var out = String("[")
    for i in range(len(vals)):
        if i > 0:
            out += String(",")
        out += String(vals[i])
    out += String("]")
    return out^


def _lora_stack_json(params: JobParams, paths: List[String]) raises -> String:
    var out = String("[")
    for i in range(len(params.loras)):
        if i > 0:
            out += String(",")
        var resolved = String("")
        if i < len(paths):
            resolved = paths[i].copy()
        out += String("{")
        out += String('"name":"') + _json_escape(params.loras[i].name) + String('",')
        out += String('"weight":') + String(params.loras[i].weight) + String(",")
        out += String('"resolved_path":"') + _json_escape(resolved) + String('"')
        out += String("}")
    out += String("]")
    return out^


def _resolve_zimage_lora_path(name: String) raises -> String:
    # Bare names from /v1/models resolve against the scanner's LoRA directory;
    # absolute/relative paths are still accepted for developer and imported use.
    if _path_exists(name):
        return name.copy()
    if _path_exists(name + ".safetensors"):
        return name + ".safetensors"
    if _path_exists(String(LORAS_DIR) + "/" + name):
        return String(LORAS_DIR) + "/" + name
    if _path_exists(String(LORAS_DIR) + "/" + name + ".safetensors"):
        return String(LORAS_DIR) + "/" + name + ".safetensors"
    raise Error(
        String("zimage: LoRA file not found: ") + name
        + " (tried as a path and under " + LORAS_DIR + ")"
    )

# F6: the cold resident-DiT load must not stall the event loop tens of seconds
# in ONE step() — load this many DiT blocks per step() call instead (measured
# ~1-2 s/tick at 2 blocks, the same stall class as one denoise step, which is
# the accepted per-tick bound this phase). The remaining single-tick stalls are
# ENCODE (per-job Qwen3-4B load+forward, tens of seconds — announced to clients
# with phase="encoding") and DECODE (announced with phase="decoding").
comptime LOAD_BLOCKS_PER_TICK = 2


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
        String("echo -n '[zimage][vram] ") + tag
        + ": ' && nvidia-smi --query-gpu=memory.used --format=csv,noheader"
    )


def _blocks_bytes(blocks: List[ZImageBlockWeights]) -> Int:
    var total = 0
    for i in range(len(blocks)):
        total += blocks[i].n1[].nbytes()
        total += blocks[i].wq[].nbytes() + blocks[i].wk[].nbytes()
        total += blocks[i].wv[].nbytes() + blocks[i].wo[].nbytes()
        total += blocks[i].q_norm[].nbytes() + blocks[i].k_norm[].nbytes()
        total += blocks[i].n2[].nbytes() + blocks[i].fn1[].nbytes()
        total += blocks[i].w1[].nbytes() + blocks[i].w3[].nbytes()
        total += blocks[i].w2[].nbytes() + blocks[i].fn2[].nbytes()
    return total


def _save_rgb_png_with_text(
    rgb: Tensor, path: String, params_json: String, ctx: DeviceContext
) raises:
    """[1,3,H,W] SIGNED float tensor → 8-bit RGB PNG with the job params in a
    serenity.genparams.v1 tEXt chunk. Quantization math == save_png's
    (_quantize, ValueRange.SIGNED); only the writer differs (tEXt support)."""
    var shape = rgb.shape()
    if len(shape) != 4 or shape[0] != 1 or shape[1] != 3:
        raise Error("zimage_backend: expected [1,3,H,W] rgb tensor")
    var height = shape[2]
    var width = shape[3]
    var host = rgb.to_host(ctx)
    var plane = height * width
    if len(host) != 3 * plane:
        raise Error("zimage_backend: rgb to_host size mismatch")
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


struct ZImageBackend(GenBackend, Movable):
    var ctx: DeviceContext

    # ── resident across jobs (loaded once, first job) ──
    var loaded: Bool
    var aux: List[ArcPointer[ZImageRealAux]]            # 0/1
    var nr_blocks: List[ZImageBlockWeights]
    var cr_blocks: List[ZImageBlockWeights]
    var main_blocks: List[ZImageBlockWeights]
    var final_lin: List[TArc]                           # [w, b]
    var x_pad_h: List[Float32]
    var cap_pad_h: List[Float32]
    var vae64: List[ArcPointer[ZImageDecoder[64, 64]]]  # 0/1
    var vae128: List[ArcPointer[ZImageDecoder[128, 128]]]
    # chunked-load state (F6): the shard set stays open across LOAD ticks and
    # load_cursor tracks how many DiT blocks (nr|cr|main flat) are loaded.
    var st_open: List[ArcPointer[ShardedSafeTensors]]   # 0/1
    var load_cursor: Int

    # ── per-job state (cleared on done/failed/cancelled) ──
    var active: Bool
    var cancel_flag: Bool
    var phase: Int
    var announced: Bool   # one "encoding"/"decoding" announce tick sent (F6)
    var cur: Int
    var params: JobParams
    var hl: Int
    var wl: Int
    var cfg: Float32
    var lora_path: String       # resolved file ("" = base / zero overlay)
    var lora_mult: Float32
    var lora_paths: List[String]
    var lora_mults: List[Float32]
    var sigmas: List[Float32]
    var executed_sampler: String
    var executed_scheduler: String
    var denoise_start_step: Int
    var dpmpp_history: MultistepHistory
    var dpmpp_update_steps: Int
    var dpmpp_second_order_steps: Int
    var unipc: List[ArcPointer[UniPcMultistepScheduler]]
    var unipc_update_steps: Int
    var unipc_corrector_steps: Int
    var unipc_second_order_steps: Int
    var cap_seq_cond: List[Float32]
    var cap_seq_uncond: List[Float32]
    var rope: List[TArc]        # 12: cond x/cap/uni cos+sin, then uncond
    var lora_dev: List[ZImageLoraDeviceSet]  # 0/1
    var latent: List[TArc]      # 0/1
    var job_t0_ns: UInt
    var load_seconds: Float64
    var text_encode_seconds: Float64
    var denoise_seconds: Float64
    var vae_decode_seconds: Float64
    var total_vram_bytes: Int
    var min_free_bytes: Int

    def __init__(out self) raises:
        self.ctx = DeviceContext()
        self.loaded = False
        self.aux = List[ArcPointer[ZImageRealAux]]()
        self.nr_blocks = List[ZImageBlockWeights]()
        self.cr_blocks = List[ZImageBlockWeights]()
        self.main_blocks = List[ZImageBlockWeights]()
        self.final_lin = List[TArc]()
        self.x_pad_h = List[Float32]()
        self.cap_pad_h = List[Float32]()
        self.vae64 = List[ArcPointer[ZImageDecoder[64, 64]]]()
        self.vae128 = List[ArcPointer[ZImageDecoder[128, 128]]]()
        self.st_open = List[ArcPointer[ShardedSafeTensors]]()
        self.load_cursor = 0
        self.active = False
        self.cancel_flag = False
        self.phase = PHASE_IDLE
        self.announced = False
        self.cur = 0
        self.params = JobParams()
        self.hl = 0
        self.wl = 0
        self.cfg = Float32(0.0)
        self.lora_path = String("")
        self.lora_mult = Float32(1.0)
        self.lora_paths = List[String]()
        self.lora_mults = List[Float32]()
        self.sigmas = List[Float32]()
        self.executed_sampler = String("flowmatch_euler")
        self.executed_scheduler = String("simple_flowmatch")
        self.denoise_start_step = 0
        self.dpmpp_history = MultistepHistory(1)
        self.dpmpp_update_steps = 0
        self.dpmpp_second_order_steps = 0
        self.unipc = List[ArcPointer[UniPcMultistepScheduler]]()
        self.unipc_update_steps = 0
        self.unipc_corrector_steps = 0
        self.unipc_second_order_steps = 0
        self.cap_seq_cond = List[Float32]()
        self.cap_seq_uncond = List[Float32]()
        self.rope = List[TArc]()
        self.lora_dev = List[ZImageLoraDeviceSet]()
        self.latent = List[TArc]()
        self.job_t0_ns = 0
        self.load_seconds = 0.0
        self.text_encode_seconds = 0.0
        self.denoise_seconds = 0.0
        self.vae_decode_seconds = 0.0
        self.total_vram_bytes = 0
        self.min_free_bytes = 0

    def backend_name(self) -> String:
        return String("zimage")

    def model_name(self) -> String:
        return String("Z-Image")

    def resident_model(self) -> String:
        """Matches the /v1/models scan entry name for the resident checkpoint
        (the zimage_base/ directory entry)."""
        return String("zimage_base") if self.loaded else String("")

    # ── job admission ─────────────────────────────────────────────────────────
    def start(mut self, params: JobParams) raises:
        if self.active:
            raise Error("ZImageBackend.start: a job is already running")
        reject_unsupported_common_runtime_params(params, String("zimage"))
        var sampler_admission = sampler_admission_for_backend(String("zimage"), params.sampler)
        if not sampler_admission.supported:
            raise Error(
                String("zimage: unsupported sampler '") + params.sampler
                + String("'; ") + sampler_admission.reason
            )
        var scheduler_admission = scheduler_admission_for_backend(String("zimage"), params.scheduler)
        if not scheduler_admission.supported:
            raise Error(
                String("zimage: unsupported scheduler '") + params.scheduler
                + String("'; ") + scheduler_admission.reason
            )
        # F2/F3: 1024x1024 is DISABLED this phase — the whole-frame 128-grid
        # decode peaks over 24 GB beside the resident DiT (CUDA OOM) and the
        # device pool retains ~8 GB between jobs; both are real VRAM work
        # deferred to Phase 4 (see serenityUI/SERENITYUI_TODO.md, "Phase 4
        # VRAM work"). Reject up front instead of failing mid-job.
        if not (params.width == 512 and params.height == 512):
            raise Error(
                String("zimage: unsupported size ") + String(params.width)
                + "x" + String(params.height)
                + " — only 512x512 is served. A 3x3 overlap-blend tiled 1024"
                + " decode (models/vae/zimage_tiled_decode.mojo, FLUX precedent)"
                + " is implemented to fix the whole-frame decode OOM, but 1024"
                + " stays gated pending GPU verification + the unresolved F3"
                + " device-pool retention (the Mojo-runtime caching allocator pins the"
                + " high-water mark; cuMemPoolTrimTo reclaims 0 — see Phase-4"
                + " findings in SERENITYUI_TODO.md). Use 512x512."
            )
        var hl = params.height // 8
        var wl = params.width // 8
        # LoRA: resolve + validate the full stack up front so a bad request
        # fails before expensive model work. Multiple adapters are merged into
        # one rank-concat inference overlay in _prepare_job.
        var lora_path = String("")
        var lora_mult = Float32(1.0)
        var lora_paths = List[String]()
        var lora_mults = List[Float32]()
        for li in range(len(params.loras)):
            var resolved = _resolve_zimage_lora_path(params.loras[li].name)
            lora_paths.append(resolved)
            lora_mults.append(Float32(params.loras[li].weight))
        if len(lora_paths) > 0:
            lora_path = lora_paths[0].copy()
            lora_mult = lora_mults[0]
        if sampler_admission.executed == "uni_pc_bh2" and params.init_image.byte_length() > 0:
            raise Error(
                "zimage: uni_pc_bh2 img2img is not supported yet; the "
                "sliced-sigma UniPC state needs separate artifact evidence"
            )
        # P7 img2img: validate the init image up front (fail at admission,
        # not mid-job). Decodability is checked at ENCODE time (decode raises
        # a clear error that fails the job through the step() except path).
        if params.init_image.byte_length() > 0:
            if not _path_exists(params.init_image):
                raise Error(
                    String("zimage: init image not found: ") + params.init_image
                )
        self.params = params.copy()
        self.hl = hl
        self.wl = wl
        self.cfg = Float32(params.cfg)
        self.lora_path = lora_path^
        self.lora_mult = lora_mult
        self.lora_paths = lora_paths^
        self.lora_mults = lora_mults^
        self.executed_sampler = sampler_admission.executed.copy()
        self.executed_scheduler = scheduler_admission.executed.copy()
        self.denoise_start_step = 0
        self.dpmpp_history = MultistepHistory(1)
        self.dpmpp_update_steps = 0
        self.dpmpp_second_order_steps = 0
        self.unipc = List[ArcPointer[UniPcMultistepScheduler]]()
        self.unipc_update_steps = 0
        self.unipc_corrector_steps = 0
        self.unipc_second_order_steps = 0
        self.active = True
        self.cancel_flag = False
        self.cur = 0
        self.announced = False
        self.phase = PHASE_LOAD
        self.job_t0_ns = perf_counter_ns()
        self.load_seconds = 0.0
        self.text_encode_seconds = 0.0
        self.denoise_seconds = 0.0
        self.vae_decode_seconds = 0.0
        var mem = cu_mem_get_info()
        self.total_vram_bytes = mem.total_bytes
        self.min_free_bytes = mem.free_bytes
        self._record_vram()

    def cancel(mut self):
        self.cancel_flag = True

    def between_jobs_trim(mut self) raises:
        """F3: reclaim the per-job transient peak (Qwen3-4B encoder ~7.5 GB,
        decode activations) back to the OS via cuMemPoolTrimTo. The resident
        DiT + VAE decoders have live suballocations and are NOT reclaimed."""
        var before = cu_mem_get_info()
        self.ctx.synchronize()
        cu_mempool_trim_current(0)
        self.ctx.synchronize()
        var after = cu_mem_get_info()
        print("[zimage] between-jobs trim: used",
              before.used_bytes() // (1024 * 1024), "->",
              after.used_bytes() // (1024 * 1024), "MiB (reclaimed",
              (before.used_bytes() - after.used_bytes()) // (1024 * 1024), "MiB)")

    # ── resident weights (DiT + both VAE grids), loaded in bounded chunks ────
    def _load_chunk(mut self) raises -> Bool:
        """F6: one bounded slice of the resident-DiT load per step() call.
        Returns True once the DiT is fully resident. Tick 1 opens the shard
        set + loads the aux (embedders/final/pads); every later tick loads
        LOAD_BLOCKS_PER_TICK DiT blocks (nr -> cr -> main, flat cursor)."""
        if self.loaded:
            return True
        if len(self.st_open) == 0:
            _print_vram("before resident load")
            print("[zimage] loading resident DiT weights from", TRANSFORMER,
                  "(chunked:", LOAD_BLOCKS_PER_TICK, "blocks/tick)")
            self.st_open.append(
                ArcPointer(ShardedSafeTensors.open(String(TRANSFORMER)))
            )
            var aux = load_zimage_real_aux(
                self.st_open[0][], NUM_NR, MAIN_DEPTH, self.ctx
            )
            self.final_lin.append(TArc(aux.final_lin_w[].clone(self.ctx)))
            self.final_lin.append(TArc(aux.final_lin_b[].clone(self.ctx)))
            self.x_pad_h = aux.x_pad_token[].to_host(self.ctx)
            self.cap_pad_h = aux.cap_pad_token[].to_host(self.ctx)
            self.aux.append(ArcPointer(aux^))
            self.load_cursor = 0
            return False
        var total = NUM_NR + NUM_CR + MAIN_DEPTH
        var done = 0
        while self.load_cursor < total and done < LOAD_BLOCKS_PER_TICK:
            var i = self.load_cursor
            if i < NUM_NR:
                self.nr_blocks.append(
                    load_zimage_block_weights_prefixed_mixed(
                        self.st_open[0][], String("noise_refiner.") + String(i),
                        self.ctx,
                    )
                )
            elif i < NUM_NR + NUM_CR:
                self.cr_blocks.append(
                    load_zimage_block_weights_prefixed_mixed(
                        self.st_open[0][],
                        String("context_refiner.") + String(i - NUM_NR),
                        self.ctx,
                    )
                )
            else:
                self.main_blocks.append(
                    load_zimage_block_weights_prefixed_mixed(
                        self.st_open[0][],
                        String("layers.") + String(i - NUM_NR - NUM_CR),
                        self.ctx,
                    )
                )
            self.load_cursor += 1
            done += 1
        if self.load_cursor < total:
            return False
        self.st_open = List[ArcPointer[ShardedSafeTensors]]()  # close the shard mmaps
        # VAE decoders are loaded lazily per grid on first decode (and then
        # stay resident): the 128-grid decode is VRAM-critical and must
        # allocate AFTER the DiT release, not before the first job.
        self.loaded = True
        var dit_bytes = (
            _blocks_bytes(self.nr_blocks) + _blocks_bytes(self.cr_blocks)
            + _blocks_bytes(self.main_blocks)
            + self.final_lin[0][].nbytes() + self.final_lin[1][].nbytes()
        )
        print("[zimage] resident DiT block bytes:", dit_bytes,
              "(", dit_bytes // (1024 * 1024), "MiB )")
        _print_vram("after resident load")
        return True

    def _reset_partial_load(mut self):
        """Drop a half-done chunked load (cancel/error mid-LOAD), so the next
        job restarts the load cleanly instead of appending duplicate blocks."""
        if self.loaded:
            return
        if len(self.st_open) == 0 and len(self.aux) == 0:
            return  # load never started
        print("[zimage] dropping partial resident load (", self.load_cursor,
              "blocks )")
        self.st_open = List[ArcPointer[ShardedSafeTensors]]()
        self.load_cursor = 0
        self.aux = List[ArcPointer[ZImageRealAux]]()
        self.nr_blocks = List[ZImageBlockWeights]()
        self.cr_blocks = List[ZImageBlockWeights]()
        self.main_blocks = List[ZImageBlockWeights]()
        self.final_lin = List[TArc]()
        self.x_pad_h = List[Float32]()
        self.cap_pad_h = List[Float32]()

    # ── per-job prep: text encode + rope + noise + sigmas + LoRA overlay ─────
    def _prepare_job(mut self) raises:
        # Qwen3-4B is loaded, used, and freed inside encode_captions_fixed.
        var caps = encode_captions_fixed(
            self.params.prompt, self.params.negative, self.ctx
        )
        _print_vram("after text encode (encoder freed)")
        self.cap_seq_cond = _cap_seq_with_pad(
            self.aux[0][], caps.cond, caps.real_cond, self.cap_pad_h, self.ctx
        )
        self.cap_seq_uncond = _cap_seq_with_pad(
            self.aux[0][], caps.uncond, caps.real_uncond, self.cap_pad_h, self.ctx
        )

        var ht = self.hl // PATCH
        var wt = self.wl // PATCH
        var n_img_real = ht * wt
        var img_pad = (32 - (n_img_real % 32)) % 32
        var n_img = n_img_real + img_pad

        # rope tables: cond then uncond, each x/cap/uni cos+sin (== _denoise).
        self.rope = List[TArc]()
        var pos_cond = build_positions(n_img, ht, wt, CAPLEN_MAX, caps.real_cond)
        var x_pos_cond = pos_cond[0].copy()
        var cap_pos_cond = pos_cond[1].copy()
        var uni_pos_cond = _unified_positions(x_pos_cond, cap_pos_cond)
        var xr_cond = build_rope(x_pos_cond, H, Dh, ROPE_THETA, AXIS0, AXIS1, AXIS2, self.ctx)
        self.rope.append(xr_cond[0].copy()); self.rope.append(xr_cond[1].copy())
        var cr_cond = build_rope(cap_pos_cond, H, Dh, ROPE_THETA, AXIS0, AXIS1, AXIS2, self.ctx)
        self.rope.append(cr_cond[0].copy()); self.rope.append(cr_cond[1].copy())
        var ur_cond = build_rope(uni_pos_cond, H, Dh, ROPE_THETA, AXIS0, AXIS1, AXIS2, self.ctx)
        self.rope.append(ur_cond[0].copy()); self.rope.append(ur_cond[1].copy())

        var pos_uncond = build_positions(n_img, ht, wt, CAPLEN_MAX, caps.real_uncond)
        var x_pos_uncond = pos_uncond[0].copy()
        var cap_pos_uncond = pos_uncond[1].copy()
        var uni_pos_uncond = _unified_positions(x_pos_uncond, cap_pos_uncond)
        var xr_uncond = build_rope(x_pos_uncond, H, Dh, ROPE_THETA, AXIS0, AXIS1, AXIS2, self.ctx)
        self.rope.append(xr_uncond[0].copy()); self.rope.append(xr_uncond[1].copy())
        var cr_uncond = build_rope(cap_pos_uncond, H, Dh, ROPE_THETA, AXIS0, AXIS1, AXIS2, self.ctx)
        self.rope.append(cr_uncond[0].copy()); self.rope.append(cr_uncond[1].copy())
        var ur_uncond = build_rope(uni_pos_uncond, H, Dh, ROPE_THETA, AXIS0, AXIS1, AXIS2, self.ctx)
        self.rope.append(ur_uncond[0].copy()); self.rope.append(ur_uncond[1].copy())

        # LoRA forward overlay (NEVER fused) — or the zero overlay in base mode.
        # Multi-LoRA stacks are merged as rank-concat adapters with scales
        # folded into B, which is mathematically equal to sum_i delta_i.
        self.lora_dev = List[ZImageLoraDeviceSet]()
        if len(self.lora_paths) > 0:
            var lora_sets = List[ZImageLoraSet]()
            for li in range(len(self.lora_paths)):
                var path = self.lora_paths[li]
                var mult = self.lora_mults[li]
                if zimage_lora_file_is_comfy(path):
                    # F5: comfy/kohya export (musubi-tuner networks.lora_zimage:
                    # lora_unet_layers_* keys, fused qkv) — key-rename + exact
                    # qkv split; rank/alpha come from the file.
                    print("[zimage][lora] loading comfy/kohya overlay", path)
                    var lora = load_zimage_lora_main_only_comfy(
                        NUM_NR, NUM_CR, MAIN_DEPTH, D, F, mult,
                        path, self.ctx,
                    )
                    lora_sets.append(lora^)
                else:
                    var lora_alpha = ALPHA * mult
                    print("[zimage][lora] loading overlay", path)
                    var lora = load_zimage_lora_main_only_resume(
                        NUM_NR, NUM_CR, MAIN_DEPTH, RANK, lora_alpha, D, F,
                        path, self.ctx,
                    )
                    lora_sets.append(lora^)
            if len(lora_sets) == 1:
                self.lora_dev.append(zimage_lora_set_to_device(lora_sets[0], self.ctx))
            else:
                print("[zimage][lora] merging", len(lora_sets), "LoRA overlays")
                var merged = merge_zimage_lora_sets_for_inference(lora_sets)
                self.lora_dev.append(zimage_lora_set_to_device(merged, self.ctx))
        else:
            self.lora_dev.append(
                build_zimage_zero_lora_device_set(NUM_NR, NUM_CR, MAIN_DEPTH, self.ctx)
            )

        # Seeded initial latent (verbatim base noise math), kept BF16 at the
        # tensor boundary. Variation uses the SwarmKSampler CHW slerp contract.
        self.denoise_start_step = 0
        self.dpmpp_history = MultistepHistory(1)
        self.dpmpp_update_steps = 0
        self.dpmpp_second_order_steps = 0
        self.unipc = List[ArcPointer[UniPcMultistepScheduler]]()
        self.unipc_update_steps = 0
        self.unipc_corrector_steps = 0
        self.unipc_second_order_steps = 0
        var noise = gaussian_noise(16 * self.hl * self.wl, UInt64(self.params.seed))
        if self.params.variation_strength > 0.0:
            var vnoise = gaussian_noise(
                16 * self.hl * self.wl,
                UInt64(self.params.variation_seed + self.params.image_index),
            )
            noise = swarm_variation_noise_chw(
                noise, vnoise, 16, self.hl, self.wl,
                self.params.variation_strength,
            )
        self.sigmas = _build_sigmas(self.params.steps)
        if self.executed_sampler == "uni_pc_bh2":
            var unipc_sigmas = List[Float64]()
            for si in range(len(self.sigmas)):
                unipc_sigmas.append(Float64(self.sigmas[si]))
            self.unipc.append(ArcPointer(
                UniPcMultistepScheduler.from_sigmas(unipc_sigmas^, 2)
            ))
        var lat_host = noise.copy()  # txt2img default: x = noise (sigma0 = 1.0)
        if self.params.init_image.byte_length() > 0:
            # ── P7 img2img (rectified-flow forward-noising convention) ──
            # EXACT FORMULA (reference citations):
            #   z      = (mu - VAE_SHIFT) * VAE_SCALE
            #            = (encode_mean(init) - 0.1159) * 0.3611
            #            (training/train_zimage_real.mojo:540)
            #   sigma0 = sigmas[i0], i0 = smallest index with
            #            sigmas[i0] <= creativity — "start the loop at the
            #            step whose sigma ~= creativity" on the FlowMatchEuler
            #            shift=6.0 schedule (pipeline/zimage_generate.mojo
            #            _build_sigmas; sigmas[0]=1.0 .. sigmas[steps]=0.0)
            #   x0     = sigma0 * noise + (1 - sigma0) * z
            #            (training/train_zimage_real.mojo:541 —
            #             noisy = noise*sig + lat*(1-sig))
            #   denoise starts at step i0 (self.cur = i0); creativity 1.0
            #   degenerates to txt2img (i0=0, x0 = noise), creativity 0.0 to
            #   a VAE round-trip (i0=steps, no denoise).
            var z = self._encode_init_latent()
            if len(z) != len(lat_host):
                raise Error("zimage img2img: init latent size mismatch")
            var i0 = 0
            var creat = Float32(self.params.creativity)
            while i0 < self.params.steps and self.sigmas[i0] > creat:
                i0 += 1
            var sig0 = self.sigmas[i0]
            for i in range(len(lat_host)):
                lat_host[i] = sig0 * noise[i] + (Float32(1.0) - sig0) * z[i]
            self.cur = i0
            self.denoise_start_step = i0
            print(
                "[zimage][img2img] creativity", self.params.creativity,
                "-> start step", i0, "/", self.params.steps,
                "( sigma0 =", sig0, ")",
            )
        var nshape = [1, 16, self.hl, self.wl]
        self.latent = List[TArc]()
        self.latent.append(
            TArc(Tensor.from_host(lat_host, nshape^, STDtype.BF16, self.ctx))
        )
        print(
            "[zimage] job", self.params.job_id, ":", self.params.steps,
            "steps, cfg", self.cfg, "seed", self.params.seed,
            "size", self.params.width, "x", self.params.height,
        )

    def _encode_init_latent(mut self) raises -> List[Float32]:
        """P7: decode the init image (MOJO-libs png/jpeg/webp), resize to the
        job WxH (bilinear), [-1,1] RGB NCHW, encode via the Z-Image VAE
        ENCODER (encode_mean — the verified trainer path), then rescale into
        diffusion latent space: z = (mu - VAE_SHIFT) * VAE_SCALE
        (training/train_zimage_real.mojo:540). The encoder is loaded per job
        and freed on return (VRAM headroom beside the resident DiT)."""
        var img = decode_image_any(self.params.init_image)
        var resized = resize_bilinear(img, self.params.width, self.params.height)
        var host = image_to_signed_nchw(resized)
        var ishape = [1, 3, self.params.height, self.params.width]
        var image_t = Tensor.from_host(host, ishape^, STDtype.BF16, self.ctx)
        print("[zimage][img2img] init image", self.params.init_image,
              "(", img.width, "x", img.height, ") -> VAE encode_mean")
        # 512x512 only this phase (enforced in start()): the 64-latent grid.
        var enc = ZImageVaeEncoder[64, 64].load(String(VAE_DIR), self.ctx)
        var mu = enc.encode_mean(image_t, self.ctx)
        var mu_h = mu.to_host(self.ctx)
        var z = List[Float32](capacity=len(mu_h))
        for i in range(len(mu_h)):
            z.append((mu_h[i] - ZIMG_SHIFT) * ZIMG_SCALING)
        _print_vram("after init-image encode (encoder freed on return)")
        return z^

    # ── one denoise step (CFG dual forward + Euler), comptime grid ───────────
    def _denoise_one[HL: Int, WL: Int](mut self) raises:
        comptime HT = HL // PATCH
        comptime WT = WL // PATCH
        comptime N_IMG_REAL = HT * WT
        comptime IMG_PAD = (32 - (N_IMG_REAL % 32)) % 32
        comptime N_IMG = N_IMG_REAL + IMG_PAD
        comptime N_TXT = CAPLEN_MAX
        comptime S = N_IMG + N_TXT
        var i = self.cur
        var t = Float32(1.0) - self.sigmas[i]  # DiT timestep convention (= 1 - sigma)
        var x = self.latent[0].copy()
        # pred = -(vc + cfg*(vc - vu)) — _cfg_pred_overlay applies the negation.
        var pred = _cfg_pred_overlay[HL, WL, N_IMG, N_TXT, S](
            x[], t, self.cfg, self.cap_seq_cond, self.cap_seq_uncond,
            self.nr_blocks, self.cr_blocks, self.main_blocks, self.lora_dev[0],
            self.aux[0][], self.final_lin[0][], self.final_lin[1][], self.x_pad_h,
            self.rope[0][], self.rope[1][], self.rope[2][], self.rope[3][],
            self.rope[4][], self.rope[5][],
            self.rope[6][], self.rope[7][], self.rope[8][], self.rope[9][],
            self.rope[10][], self.rope[11][],
            False, self.ctx,
        )
        # Euler: x += (sigma_next - sigma) * pred (latent F32 compute, BF16 carrier).
        # DPM++ 2M uses the same CFG velocity tensor converted to denoised x0.
        var sigma = self.sigmas[i]
        var sigma_next = self.sigmas[i + 1]
        var x_compute = _cast(x[], STDtype.F32, self.ctx)
        var x_new: Tensor
        if self.executed_sampler == "dpmpp_2m":
            if sigma <= Float32(0.0) or sigma_next == sigma:
                x_new = _cast(x_compute, STDtype.BF16, self.ctx)
            else:
                if not self.dpmpp_history.is_empty():
                    self.dpmpp_second_order_steps += 1
                var denoised = denoised_from_velocity(x_compute, pred, sigma, self.ctx)
                var x_next = dpmpp_2m_step(
                    x_compute, denoised, sigma, sigma_next, self.dpmpp_history,
                    self.ctx,
                )
                self.dpmpp_history.push(
                    denoised^, lambda_from_sigma_f64(Float64(sigma))
                )
                self.dpmpp_update_steps += 1
                x_new = _cast(x_next, STDtype.BF16, self.ctx)
        elif self.executed_sampler == "uni_pc_bh2":
            if sigma <= Float32(0.0) or sigma_next == sigma:
                x_new = _cast(x_compute, STDtype.BF16, self.ctx)
            else:
                if len(self.unipc) == 0:
                    raise Error("zimage: uni_pc_bh2 scheduler was not initialized")
                if self.unipc[0][].step_index() > 0:
                    self.unipc_corrector_steps += 1
                var x_next = self.unipc[0][].step(pred, x_compute, self.ctx)
                self.unipc_update_steps += 1
                if self.unipc[0][].this_order() >= 2:
                    self.unipc_second_order_steps += 1
                x_new = _cast(x_next, STDtype.BF16, self.ctx)
        else:
            var dt = sigma_next - sigma
            x_new = _cast(
                add(x_compute, mul_scalar(pred, dt, self.ctx), self.ctx),
                STDtype.BF16, self.ctx,
            )
        self.latent = List[TArc]()
        self.latent.append(TArc(x_new^))

    def _record_vram(mut self) raises:
        var mem = cu_mem_get_info()
        if self.total_vram_bytes == 0:
            self.total_vram_bytes = mem.total_bytes
        if self.min_free_bytes == 0 or mem.free_bytes < self.min_free_bytes:
            self.min_free_bytes = mem.free_bytes

    def _write_result_manifest(mut self, png_path: String) raises -> String:
        self._record_vram()
        var manifest_path = png_path + String(".zimage_daemon_result.json")
        var denoise_per_step = Float64(0.0)
        if self.params.steps > 0:
            denoise_per_step = self.denoise_seconds / Float64(self.params.steps)
        var total_wall_seconds = Float64(perf_counter_ns() - self.job_t0_ns) / 1.0e9
        var peak_vram_mib = Float64(0.0)
        if self.total_vram_bytes > 0 and self.min_free_bytes > 0:
            peak_vram_mib = _peak_vram_mib(self.total_vram_bytes, self.min_free_bytes)
        var steps_executed = self.params.steps - self.denoise_start_step
        if steps_executed < 0:
            steps_executed = 0
        var content = String("{\n")
        content += String('  "schema":"serenity.zimage.daemon_result.v1",\n')
        content += String('  "backend":"zimage_daemon",\n')
        content += String('  "model":"zimage",\n')
        content += String('  "readiness_label":"experimental",\n')
        content += String('  "accepted_sampler_parity":false,\n')
        content += String('  "accepted_speed_parity":false,\n')
        content += String('  "run_identity":{\n')
        content += String('    "job_id":"') + _json_escape(self.params.job_id) + String('",\n')
        content += String('    "prompt":"') + _json_escape(self.params.prompt) + String('",\n')
        content += String('    "negative":"') + _json_escape(self.params.negative) + String('",\n')
        content += String('    "seed":') + String(self.params.seed) + String(",\n")
        content += String('    "resolution":{"width":') + String(self.params.width) + String(',"height":') + String(self.params.height) + String("},\n")
        content += String('    "steps":') + String(self.params.steps) + String(",\n")
        content += String('    "guidance":') + String(self.params.cfg) + String(",\n")
        content += String('    "sampler_registry_backend":"zimage",\n')
        content += String('    "requested_sampler":"') + _json_escape(self.params.sampler) + String('",\n')
        content += String('    "requested_scheduler":"') + _json_escape(self.params.scheduler) + String('",\n')
        content += String('    "executed_sampler":"') + _json_escape(self.executed_sampler) + String('",\n')
        content += String('    "executed_scheduler":"') + _json_escape(self.executed_scheduler) + String('",\n')
        content += String('    "denoise_start_step":') + String(self.denoise_start_step) + String(",\n")
        content += String('    "steps_executed":') + String(steps_executed) + String(",\n")
        content += String('    "sigma_trace":') + _float32_list_json(self.sigmas) + String(",\n")
        content += String('    "sampler_trace":{')
        if self.executed_sampler == "dpmpp_2m":
            content += String('"algorithm":"dpmpp_2m",')
            content += String('"history_capacity":1,')
            content += String('"history_final_len":') + String(self.dpmpp_history.len()) + String(",")
            content += String('"dpmpp_update_steps":') + String(self.dpmpp_update_steps) + String(",")
            content += String('"dpmpp_second_order_steps":') + String(self.dpmpp_second_order_steps)
        elif self.executed_sampler == "uni_pc_bh2":
            var unipc_step_index = 0
            var unipc_lower_order_nums = 0
            if len(self.unipc) > 0:
                unipc_step_index = self.unipc[0][].step_index()
                unipc_lower_order_nums = self.unipc[0][].lower_order_nums()
            content += String('"algorithm":"uni_pc_bh2",')
            content += String('"solver_type":"bh2",')
            content += String('"solver_order":2,')
            content += String('"schedule_source":"zimage_build_sigmas",')
            content += String('"unipc_update_steps":') + String(self.unipc_update_steps) + String(",")
            content += String('"unipc_corrector_steps":') + String(self.unipc_corrector_steps) + String(",")
            content += String('"unipc_second_order_steps":') + String(self.unipc_second_order_steps) + String(",")
            content += String('"unipc_step_index":') + String(unipc_step_index) + String(",")
            content += String('"unipc_lower_order_nums":') + String(unipc_lower_order_nums)
        else:
            content += String('"algorithm":"flowmatch_euler",')
            content += String('"update_steps":') + String(steps_executed)
        content += String("},\n")
        content += String('    "variation_seed":') + String(self.params.variation_seed) + String(",\n")
        content += String('    "variation_strength":') + String(self.params.variation_strength) + String(",\n")
        content += String('    "variation_applied":') + _json_bool(self.params.variation_strength > 0.0) + String(",\n")
        content += String('    "image_index":') + String(self.params.image_index) + String(",\n")
        content += String('    "image_count":') + String(self.params.image_count) + String(",\n")
        content += String('    "lora_count":') + String(len(self.params.loras)) + String(",\n")
        content += String('    "lora_merge_strategy":"rank_concat_scaled_b",\n')
        content += String('    "lora_stack":') + _lora_stack_json(self.params, self.lora_paths) + String(",\n")
        content += String('    "dtype":"bf16"\n')
        content += String("  },\n")
        content += String('  "mojo":{\n')
        content += String('    "load_seconds":') + String(self.load_seconds) + String(",\n")
        content += String('    "text_encode_seconds":') + String(self.text_encode_seconds) + String(",\n")
        content += String('    "denoise_seconds":') + String(self.denoise_seconds) + String(",\n")
        content += String('    "denoise_seconds_per_step":') + String(denoise_per_step) + String(",\n")
        content += String('    "vae_decode_seconds":') + String(self.vae_decode_seconds) + String(",\n")
        content += String('    "total_wall_seconds":') + String(total_wall_seconds) + String(",\n")
        content += String('    "peak_vram_mib":') + String(peak_vram_mib) + String(",\n")
        content += String('    "artifact_paths":["') + _json_escape(png_path) + String('","') + _json_escape(manifest_path) + String('"]\n')
        content += String("  },\n")
        content += String('  "output_png":"') + _json_escape(png_path) + String('",\n')
        content += String('  "note":"Daemon product-path result; timings are daemon phase timings and peak_vram_mib is sampled from the active CUDA context. Speed parity is not accepted without paired baseline evidence."\n')
        content += String("}\n")
        _write_text_file(manifest_path, content)
        return manifest_path

    # ── final decode + PNG(tEXt) ──────────────────────────────────────────────
    def _decode_and_save(mut self) raises -> String:
        var png_path = self.params.out_dir + "/" + self.params.job_id + ".png"
        var lat = self.latent[0].copy()
        # Per-job conditioning tensors are dead weight at decode time; free
        # them BEFORE the decode allocations (the 1024² decode is VRAM-tight).
        self.rope = List[TArc]()
        self.cap_seq_cond = List[Float32]()
        self.cap_seq_uncond = List[Float32]()
        self.lora_dev = List[ZImageLoraDeviceSet]()
        self.latent = List[TArc]()
        if self.hl == 64:
            if len(self.vae64) == 0:
                print("[zimage] loading VAE decoder (64-latent grid; stays resident)")
                self.vae64.append(ArcPointer(ZImageDecoder[64, 64].load(VAE_DIR, self.ctx)))
            var rgb = self.vae64[0][].decode(
                _cast(lat[], STDtype.BF16, self.ctx), self.ctx
            )
            _save_rgb_png_with_text(rgb, png_path, self.params.params_json, self.ctx)
        else:
            if len(self.vae128) == 0:
                print("[zimage] loading VAE decoder (128-latent grid; stays resident)")
                self.vae128.append(ArcPointer(ZImageDecoder[128, 128].load(VAE_DIR, self.ctx)))
            var rgb = self.vae128[0][].decode(
                _cast(lat[], STDtype.BF16, self.ctx), self.ctx
            )
            _save_rgb_png_with_text(rgb, png_path, self.params.params_json, self.ctx)
        return png_path

    def _clear_job(mut self):
        """Free per-job tensors (resident weights stay; a PARTIAL chunked
        load is dropped so the next job restarts it cleanly)."""
        self._reset_partial_load()
        self.active = False
        self.phase = PHASE_IDLE
        self.cur = 0
        self.cancel_flag = False
        self.announced = False
        self.sigmas = List[Float32]()
        self.cap_seq_cond = List[Float32]()
        self.cap_seq_uncond = List[Float32]()
        self.rope = List[TArc]()
        self.lora_dev = List[ZImageLoraDeviceSet]()
        self.lora_path = String("")
        self.lora_mult = Float32(1.0)
        self.lora_paths = List[String]()
        self.lora_mults = List[Float32]()
        self.latent = List[TArc]()
        self.executed_sampler = String("flowmatch_euler")
        self.executed_scheduler = String("simple_flowmatch")
        self.denoise_start_step = 0
        self.dpmpp_history = MultistepHistory(1)
        self.dpmpp_update_steps = 0
        self.dpmpp_second_order_steps = 0
        self.unipc = List[ArcPointer[UniPcMultistepScheduler]]()
        self.unipc_update_steps = 0
        self.unipc_corrector_steps = 0
        self.unipc_second_order_steps = 0
        self.job_t0_ns = 0
        self.load_seconds = 0.0
        self.text_encode_seconds = 0.0
        self.denoise_seconds = 0.0
        self.vae_decode_seconds = 0.0
        self.total_vram_bytes = 0
        self.min_free_bytes = 0

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
            if self.phase == PHASE_LOAD:
                # F6: bounded chunks — one shard-open+aux tick, then
                # LOAD_BLOCKS_PER_TICK DiT blocks per tick, each a WS event
                # with phase="loading" so clients can show the cold load.
                var load_t0 = perf_counter_ns()
                if self._load_chunk():
                    self.phase = PHASE_ENCODE
                self.load_seconds += Float64(perf_counter_ns() - load_t0) / 1.0e9
                self._record_vram()
                r.step = 0
                r.phase = String("loading")
                return r^
            if self.phase == PHASE_ENCODE:
                if not self.announced:
                    # announce BEFORE the long blocking encode tick (per-job
                    # Qwen3-4B load+forward) so clients see what's happening.
                    self.announced = True
                    r.step = 0
                    r.phase = String("encoding")
                    return r^
                var encode_t0 = perf_counter_ns()
                self._prepare_job()
                self.text_encode_seconds = Float64(perf_counter_ns() - encode_t0) / 1.0e9
                self._record_vram()
                self.announced = False
                self.phase = PHASE_DENOISE
                r.step = 0
                return r^
            if self.phase == PHASE_DENOISE:
                if self.cur >= self.params.steps:
                    # img2img with creativity 0.0: i0 == steps — no denoise
                    # steps remain (pure VAE round-trip); go straight to decode.
                    self.phase = PHASE_DECODE
                    r.step = self.cur
                    return r^
                var denoise_t0 = perf_counter_ns()
                if self.hl == 64:
                    self._denoise_one[64, 64]()
                else:
                    self._denoise_one[128, 128]()
                self.denoise_seconds += Float64(perf_counter_ns() - denoise_t0) / 1.0e9
                self._record_vram()
                self.cur += 1
                r.step = self.cur
                if self.cur >= self.params.steps:
                    self.phase = PHASE_DECODE
                return r^
            if not self.announced:
                # announce BEFORE the long blocking VAE-decode tick.
                self.announced = True
                r.step = self.params.steps
                r.phase = String("decoding")
                return r^
            # PHASE_DECODE: VAE decode + PNG with serenity.genparams.v1 tEXt.
            # MEASURED (2026-06-10, 24 GB GPU): resident DiT (13.2 GiB) +
            # whole-frame 1024² decode peaks at 23.6 GiB → CUDA OOM. The
            # verified standalone pipeline always freed the DiT before its
            # 1024² decode; mirror that for the 128-grid only: release the
            # resident DiT, decode, and reload on the next job. 512² decode
            # fits with the DiT resident and keeps the full residency win.
            if self.hl == 128 and self.loaded:
                print("[zimage] 1024x1024 decode: releasing resident DiT to fit decode (next job reloads)")
                self.nr_blocks = List[ZImageBlockWeights]()
                self.cr_blocks = List[ZImageBlockWeights]()
                self.main_blocks = List[ZImageBlockWeights]()
                self.aux = List[ArcPointer[ZImageRealAux]]()
                self.final_lin = List[TArc]()
                self.x_pad_h = List[Float32]()
                self.cap_pad_h = List[Float32]()
                self.loaded = False
            var decode_t0 = perf_counter_ns()
            var path = self._decode_and_save()
            self.vae_decode_seconds = Float64(perf_counter_ns() - decode_t0) / 1.0e9
            self._record_vram()
            var manifest = self._write_result_manifest(path)
            print("[zimage][manifest] saved:", manifest)
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
