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
#   * The FLUX.1-dev DiT uses a complete pinned-host FP8 store. All 57 blocks
#     are copied before step 0, required resident, and staged from RAM during
#     denoise. Checkpoint pages are released after store construction.
#   * The CLIP-L (~250 MB) + T5-XXL (~9.5 GB F16) encoders are loaded → used →
#     freed PER JOB inside the ENCODE step (encode_text does the load+free).
#   * The FLUX VAE (~330 MB) is loaded PER JOB inside the TILED DECODE step.
#   * Before unpack + 1024² VAE decode, the resident DiT offloader handle + rope
#     are FREED and the mempool TRIMMED (MEASURED on SDXL/Flux gates: VAE decode
#     OOMs a 24 GB card when the denoiser/offloader high-water is still present).
#     self.loaded is reset so the NEXT job reloads the DiT in the LOAD phase.
#
# step() state machine: ENCODE (per-job, blocking — announced phase="encoding")
#   → LOAD (DiT offloader + rope, announced phase="loading") → DENOISE×steps
#   (one guidance-distilled forward + Euler update per tick) → DECODE (announced
#   phase="decoding") → done. cancel() makes the next step() return cancelled and
#   frees all per-job tensors.
#
# Size support: the finite seven-shape ~1 MP product ladder. Each dispatch arm
# specializes latent pack/unpack, image-token count, joint attention length,
# rectangular RoPE, and whole/tiled VAE decode at compile time. steps/guidance
# (=cfg)/seed remain runtime job inputs.
#
# LoRA: HONORED via Flux1Offloaded.load_with_lora (Kohya/sd-scripts BFL FLUX
# LoRA, additive overlay W += scale·up@down at the requested multiplier — the saved
# checkpoint is never fused, per the LoRA-never-fused rule). Because a LoRA
# changes the resident DiT, a LoRA job (or a LoRA change) reloads the DiT; only
# base (no-LoRA) jobs keep the resident handle. The loader accepts both
# Kohya/sd-scripts BFL keys and common Diffusers/PEFT Transformer2DModel keys.

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
from serenitymojo.image.png import _quantize, ValueRange
from serenitymojo.offload.vmm_cuda import cu_mempool_trim_current, cu_mem_get_info
from serenitymojo.offload.turbo_planned_loader import TurboPlannedLoader
from serenitymojo.offload.plan import OffloadConfig, build_flux1_dev_block_plan

from serenitymojo.models.dit.flux1_dit import (
    Flux1Config, Flux1Offloaded, build_flux1_rope_tables,
)
from serenitymojo.sampling.swarmui_schedules import build_swarm_flux_schedule
from serenitymojo.sampling.dpmpp_2m import (
    MultistepHistory,
    denoised_from_velocity,
    dpmpp_2m_step,
    lambda_from_sigma_f64,
)
from serenitymojo.ops.activations import silu
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.linear import linear
from serenitymojo.ops.random import randn
from serenitymojo.ops.tensor_algebra import add, concat, mul_scalar, slice
from serenitymojo.sampling.sampler_registry import (
    sampler_admission_for_backend, scheduler_admission_for_backend,
)
from serenitymojo.sampling.variation_noise import variation_noise_chw
from serenitymojo.training.aspect_buckets import (
    DEFAULT_ASPECT_LADDER_LEN, DEFAULT_ASPECT_LADDER_X100,
    aspect_lat_h_units, aspect_lat_w_units,
)
from serenitymojo.pipeline.flux_tiled_decode import flux_tiled_decode
from serenitymojo.models.vae.ldm_decoder import load_flux1_ldm_decoder
from serenitymojo.pipeline.flux_sample_cli import (
    FluxCaps, encode_text, _pack_latent_shape, _unpack_latent_shape,
    DIT_PATH, VAE_PATH,
    AE_IN_CHANNELS, N_TXT,
)
from serenitymojo.serve.flux_encode_subprocess import encode_text_subprocess
from serenitymojo.serve.proc_ipc import (
    prefault_self_executable, prefault_mapped_shared_libraries,
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
from serenitymojo.serve.product_manifest import (
    json_bool, json_escape, peak_vram_mib, write_text_file,
)

comptime GENPARAMS_TEXT_KEY = "serenity.genparams.v1"


comptime FPHASE_IDLE = 0
comptime FPHASE_ENCODE = 1
comptime FPHASE_LOAD = 2
comptime FPHASE_DENOISE = 3
comptime FPHASE_DECODE = 4
comptime FLUX_PRODUCT_EDGE_UNITS = 16

# Whole-image VAE decode is preferred when it fits: tiled decode is MEASURED to
# degrade output; the ungated 5x5-lowmem variant is the WORST (84.8% px differ
# same-latent). After the DiT free + mempool trim we query free VRAM and decode
# whole when it clears this bar, else fall back to the GATED 3x3 flux_tiled_decode
# (parity-gated, unlike 5x5). A live 1024² job with 16.64 GiB free still OOMed
# inside whole decode on 2026-07-28; require 20 GiB so a high-water parent uses
# the safe tiled route instead of failing after a completed denoise. See MJ-1054.
comptime WHOLE_DECODE_MIN_FREE_BYTES = 20 * 1024 * 1024 * 1024


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


def _flux_shape_supported(width: Int, height: Int) -> Bool:
    comptime for bi in range(DEFAULT_ASPECT_LADDER_LEN):
        comptime X100_BI = DEFAULT_ASPECT_LADDER_X100[bi]
        comptime LH_BI = aspect_lat_h_units(X100_BI, FLUX_PRODUCT_EDGE_UNITS)
        comptime LW_BI = aspect_lat_w_units(X100_BI, FLUX_PRODUCT_EDGE_UNITS)
        if width == LW_BI * 8 and height == LH_BI * 8:
            return True
    return False


def _unpack_flux_packed_latent_shape[LH_: Int, LW_: Int](
    packed: Tensor, ctx: DeviceContext
) raises -> Tensor:
    var latent_f32 = cast_tensor(packed, STDtype.F32, ctx)
    return _unpack_latent_shape[LH_, LW_](latent_f32, ctx)


# Single LoRA only (one additive overlay through load_with_lora). Returns the
# selected LoRA path ("" = base). Mirrors the CLI's single argv[2] LoRA slot.
def _select_lora_path(loras: List[LoraSpec]) -> String:
    if len(loras) == 0:
        return String("")
    return loras[0].name.copy()


def _select_lora_multiplier(loras: List[LoraSpec]) -> Float32:
    if len(loras) == 0:
        return Float32(1.0)
    return Float32(loras[0].weight)


def _select_lora_signature(loras: List[LoraSpec]) -> String:
    if len(loras) == 0:
        return String("")
    return loras[0].name + String("@") + String(loras[0].weight)


# ── Turbo-streamed FLUX.1-dev DiT forward ─────────────────────────────────────
# BYTE-IDENTICAL math to Flux1Offloaded.forward (flux1_dit.mojo): same input
# projections, same _embed_vec, same per-block img_mod/txt_mod/single_mod
# linears, same _double_block/_single_block calls (via the same transient
# _block_model / _block_model_lora weight tables, so the LoRA overlay path is
# unchanged), same final layer. ONLY the weight staging changes: the offloader's
# internal naive synchronous BlockLoader is bypassed in favor of the complete
# FP8 host-resident store: block i+1 is DMA-staged from RAM on the explicit copy
# stream WHILE block
# i computes on the default stream (await_block fences via DeviceEvent;
# mark_active_slot_compute_done lets the copy stream reuse a slot only after
# the block's kernels are queued).
def _flux_forward_turbo[
    FN_IMG: Int, FN_TXT: Int, FS: Int
](
    model: Flux1Offloaded,
    mut tloader: TurboPlannedLoader,
    img_tokens: Tensor,
    txt_tokens: Tensor,
    timestep: Tensor,
    guidance: Optional[Tensor],
    vector: Tensor,
    cos: Tensor,
    sin: Tensor,
    ctx: DeviceContext,
) raises -> Tensor:
    comptime assert FS == FN_IMG + FN_TXT, "S must equal N_IMG + N_TXT"
    var cfg = model.shared.config

    # Stage block 0 first: its DMA overlaps the input projections + vec embed.
    tloader.set_config(OffloadConfig.synchronous_single())
    tloader.prefetch_with_ctx(0, ctx)

    # input projections (with bias).
    var img = linear(
        img_tokens,
        model.shared._w(String("img_in.weight")),
        Optional[Tensor](model.shared._w(String("img_in.bias")).clone(ctx)),
        ctx,
    )
    var txt = linear(
        txt_tokens,
        model.shared._w(String("txt_in.weight")),
        Optional[Tensor](model.shared._w(String("txt_in.bias")).clone(ctx)),
        ctx,
    )

    # vec = time + guidance + vector.
    var vec = model.shared._embed_vec(timestep, guidance, vector, ctx)

    # double blocks — await this block's slot, immediately stage the NEXT block
    # into the idle slot, then queue this block's compute.
    for bi in range(cfg.num_double):
        var prefix = String("double_blocks.") + String(bi)
        var handle = tloader.await_block(bi, ctx)
        tloader.prefetch_next_with_ctx(bi, ctx)
        var bm = model._block_model(handle.block) if model.lora.count() == 0 else model._block_model_lora(handle.block, ctx)
        var vec_silu = silu(vec, ctx)
        var img_mod = linear(
            vec_silu,
            bm._w(prefix + ".img_mod.lin.weight"),
            Optional[Tensor](bm._w(prefix + ".img_mod.lin.bias").clone(ctx)),
            ctx,
        )
        var txt_mod = linear(
            vec_silu,
            bm._w(prefix + ".txt_mod.lin.weight"),
            Optional[Tensor](bm._w(prefix + ".txt_mod.lin.bias").clone(ctx)),
            ctx,
        )
        var merged = bm._double_block[FN_IMG, FN_TXT, FS](
            prefix, bm, img, txt, img_mod, txt_mod, cos, sin, ctx
        )
        txt = slice(merged, 1, 0, FN_TXT, ctx)
        img = slice(merged, 1, FN_TXT, FN_IMG, ctx)
        tloader.mark_active_block_done(ctx)

    # merge cat([txt, img]) for the single-stream blocks.
    var x = concat(1, ctx, txt, img)
    for bi in range(cfg.num_single):
        var prefix = String("single_blocks.") + String(bi)
        var plan_idx = cfg.num_double + bi
        var handle = tloader.await_block(plan_idx, ctx)
        tloader.prefetch_next_with_ctx(plan_idx, ctx)
        var bm = model._block_model(handle.block) if model.lora.count() == 0 else model._block_model_lora(handle.block, ctx)
        var vec_silu = silu(vec, ctx)
        var single_mod = linear(
            vec_silu,
            bm._w(prefix + ".modulation.lin.weight"),
            Optional[Tensor](bm._w(prefix + ".modulation.lin.bias").clone(ctx)),
            ctx,
        )
        x = bm._single_block[FS](prefix, bm, x, single_mod, cos, sin, ctx)
        tloader.mark_active_block_done(ctx)

    # extract image tokens (txt first in the merged seq) and run final layer.
    var img_out = slice(x, 1, FN_TXT, FN_IMG, ctx)
    var vec_silu_final = silu(vec, ctx)
    return model.shared._final_layer(img_out, vec_silu_final, ctx)


struct FluxBackend(GenBackend, Movable):
    var ctx: DeviceContext

    # ── resident across BASE jobs (offloader handle + rope, first base job) ──
    # ArcPointer wrappers: Flux1Offloaded / Tensor are Movable-not-Copyable, and
    # List[T] requires T: Copyable — Arc is Copyable (refcount), so List[Arc[..]]
    # holds the 0/1. `loaded_lora` tracks the path AND requested multiplier
    # ("" = base) so changing either forces a reload.
    var loaded: Bool
    var loaded_lora: String
    var loaded_width: Int
    var loaded_height: Int
    var model: List[ArcPointer[Flux1Offloaded]]  # 0/1 (resident offloader)
    var tloader: List[ArcPointer[TurboPlannedLoader]]  # 0/1 complete host store
    var rope_cos: List[ArcPointer[Tensor]]       # 0/1 (resident rope cos)
    var rope_sin: List[ArcPointer[Tensor]]       # 0/1 (resident rope sin)
    var warmed_shapes: List[String]              # CUDA kernels activated per WxH

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
    var runtime_steps: Int                 # len(sched)-1; DDIM may exceed request
    var latent: List[ArcPointer[Tensor]]   # 0/1 (packed [1,N_IMG,64] BF16-castable)
    var executed_sampler: String
    var executed_scheduler: String
    var dpmpp_history: MultistepHistory
    var dpmpp_history_final_len: Int
    var dpmpp_update_steps: Int
    var dpmpp_second_order_steps: Int
    var vae_decode_grid: String            # executed decode path (result manifest)
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
        self.loaded_lora = String("")
        self.loaded_width = 0
        self.loaded_height = 0
        self.model = List[ArcPointer[Flux1Offloaded]]()
        self.tloader = List[ArcPointer[TurboPlannedLoader]]()
        self.rope_cos = List[ArcPointer[Tensor]]()
        self.rope_sin = List[ArcPointer[Tensor]]()
        self.warmed_shapes = List[String]()
        self.active = False
        self.cancel_flag = False
        self.phase = FPHASE_IDLE
        self.announced = False
        self.cur = 0
        self.params = JobParams()
        self.guidance = Float32(3.5)
        self.caps = List[ArcPointer[FluxCaps]]()
        self.sched = List[Float32]()
        self.runtime_steps = 0
        self.latent = List[ArcPointer[Tensor]]()
        self.executed_sampler = String("flux_flowmatch_euler")
        self.executed_scheduler = String("simple")
        self.dpmpp_history = MultistepHistory(1)
        self.dpmpp_history_final_len = 0
        self.dpmpp_update_steps = 0
        self.dpmpp_second_order_steps = 0
        self.vae_decode_grid = String("")
        self.job_t0_ns = UInt(0)
        self.load_seconds = 0.0
        self.text_encode_seconds = 0.0
        self.prepare_seconds = 0.0
        self.denoise_seconds = 0.0
        self.vae_decode_seconds = 0.0
        self.total_vram_bytes = 0
        self.min_free_bytes = 0

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
        if not _flux_shape_supported(params.width, params.height):
            raise Error(
                String("flux: unsupported size ") + String(params.width)
                + "x" + String(params.height)
                + " — supported product sizes are 1024x1024, 1152x896,"
                + " 896x1152, 1344x768, 768x1344, 1280x832, and 832x1280"
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
        self.executed_sampler = sampler_admission.executed.copy()
        self.executed_scheduler = scheduler_admission.normalized.copy()
        self.dpmpp_history = MultistepHistory(1)
        self.dpmpp_history_final_len = 0
        self.dpmpp_update_steps = 0
        self.dpmpp_second_order_steps = 0
        self.runtime_steps = 0
        # FLUX is guidance-distilled: params.cfg is the guidance scalar fed to the
        # DiT (NOT a CFG multiplier). Negative prompt is discarded (no CFG path).
        # cfg<=0 means unset/invalid -> FLUX recipe default 3.5, mirroring the
        # gated CLI guard (flux_sample_cli.mojo DEFAULT_GUIDANCE). MJ-1057.
        self.guidance = Float32(params.cfg) if params.cfg > 0.0 else Float32(3.5)
        self.active = True
        self.cancel_flag = False
        self.cur = 0
        self.announced = False
        self.phase = FPHASE_ENCODE
        self.job_t0_ns = perf_counter_ns()
        self.load_seconds = 0.0
        self.text_encode_seconds = 0.0
        self.prepare_seconds = 0.0
        self.denoise_seconds = 0.0
        self.vae_decode_seconds = 0.0
        self.vae_decode_grid = String("")
        var mem = cu_mem_get_info()
        self.total_vram_bytes = mem.total_bytes
        self.min_free_bytes = mem.free_bytes
        self._record_vram()

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

    def _record_vram(mut self) raises:
        var mem = cu_mem_get_info()
        if self.total_vram_bytes == 0:
            self.total_vram_bytes = mem.total_bytes
        if self.min_free_bytes == 0 or mem.free_bytes < self.min_free_bytes:
            self.min_free_bytes = mem.free_bytes

    def _write_result_manifest(mut self, png_path: String) raises -> String:
        self._record_vram()
        var manifest_path = png_path + String(".flux_daemon_result.json")
        var denoise_per_step = Float64(0.0)
        if self.runtime_steps > 0:
            denoise_per_step = self.denoise_seconds / Float64(self.runtime_steps)
        var total_wall_seconds = Float64(perf_counter_ns() - self.job_t0_ns) / 1.0e9
        var peak_mib = Float64(0.0)
        if self.total_vram_bytes > 0 and self.min_free_bytes > 0:
            peak_mib = peak_vram_mib(self.total_vram_bytes, self.min_free_bytes)

        var content = String("{\n")
        content += String('  "schema":"serenity.flux.daemon_result.v1",\n')
        content += String('  "backend":"flux_daemon",\n')
        content += String('  "model":"flux1-dev",\n')
        content += String('  "readiness_label":"experimental",\n')
        content += String('  "accepted_sampler_parity":false,\n')
        content += String('  "accepted_speed_parity":false,\n')
        content += String('  "run_identity":{\n')
        content += String('    "job_id":"') + json_escape(self.params.job_id) + String('",\n')
        content += String('    "prompt":"') + json_escape(self.params.prompt) + String('",\n')
        content += String('    "negative":"') + json_escape(self.params.negative) + String('",\n')
        content += String('    "negative_prompt_used":false,\n')
        content += String('    "seed":') + String(self.params.seed) + String(",\n")
        content += String('    "resolution":{"width":') + String(self.params.width) + String(',"height":') + String(self.params.height) + String("},\n")
        content += String('    "steps":') + String(self.params.steps) + String(",\n")
        content += String('    "executed_steps":') + String(self.runtime_steps) + String(",\n")
        content += String('    "guidance":') + String(self.guidance) + String(",\n")
        content += String('    "sampler_registry_backend":"flux",\n')
        content += String('    "requested_sampler":"') + json_escape(self.params.sampler) + String('",\n')
        content += String('    "requested_scheduler":"') + json_escape(self.params.scheduler) + String('",\n')
        content += String('    "executed_sampler":"') + json_escape(self.executed_sampler) + String('",\n')
        content += String('    "executed_scheduler":"flux_swarmui_') + json_escape(self.executed_scheduler) + String('",\n')
        content += String('    "schedule_source":"swarmui_comfy_model_sampling_flux_1_15",\n')
        content += String('    "variation_seed":') + String(self.params.variation_seed) + String(",\n")
        content += String('    "variation_strength":') + String(self.params.variation_strength) + String(",\n")
        content += String('    "variation_applied":') + json_bool(self.params.variation_strength > 0.0) + String(",\n")
        content += String('    "released_resident_dit_before_unpack":true,\n')
        content += String('    "image_index":') + String(self.params.image_index) + String(",\n")
        content += String('    "image_count":') + String(self.params.image_count) + String(",\n")
        content += String('    "lora_count":') + String(len(self.params.loras)) + String(",\n")
        # Decode releases the resident DiT before the manifest is written, so
        # `self.loaded_lora` is intentionally cleared by then. Record the job's
        # selected overlay from immutable request state instead.
        content += String('    "loaded_lora":"') + json_escape(
            _select_lora_path(self.params.loras)
        ) + String('",\n')
        content += String('    "lora_weight":') + String(
            _select_lora_multiplier(self.params.loras)
        ) + String(",\n")
        content += String('    "sampler_trace":{"history_capacity":1,"history_final_len":')
        content += String(self.dpmpp_history_final_len) + String(',"dpmpp_update_steps":')
        content += String(self.dpmpp_update_steps) + String(',"dpmpp_second_order_steps":')
        content += String(self.dpmpp_second_order_steps) + String("},\n")
        content += String('    "vae_decode_tile_grid":"') + json_escape(self.vae_decode_grid) + String('",\n')
        content += String('    "dtype":"fp8_e4m3_host_to_bf16_dit_f32_latent"\n')
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
        content += String('  "note":"Rust-server Mojo worker product-path result; FLUX.1-dev uses guidance-distilled single-forward denoise and process-local offload. Speed parity remains unaccepted until paired baseline evidence exists."\n')
        content += String("}\n")
        write_text_file(manifest_path, content)
        return manifest_path

    # ── per-job prep ───────────────────────────────────────────────────────────
    def _encode(mut self) raises:
        """Real CLIP-L pooled + T5-XXL hidden encode of params.prompt, in a
        fork+execv CHILD PROCESS (flux_encode_subprocess — the chroma-proven
        split): MEASURED job-0077, the in-process encode grew this worker's
        pool 1062→13830 MiB with nothing reclaimable ("encoders freed" changed
        nothing) and the offloaded DiT then OOM'd mid-denoise at 14698 MiB on
        the 16 GB card. The child runs the identical encode_text and exits —
        process death reclaims its VRAM unconditionally; falls back to the
        in-process encode_text on any subprocess failure. FLUX is
        guidance-distilled — only the positive prompt is encoded; the negative
        is discarded (no CFG path)."""
        _print_vram("before CLIP-L + T5-XXL load")
        var caps = encode_text_subprocess(self.params.prompt, self.ctx)
        _print_vram("after text encode (encoders freed)")
        self.caps = List[ArcPointer[FluxCaps]]()
        self.caps.append(ArcPointer(caps^))

    def _free_dit(mut self):
        """Drop GPU-resident shared weights and RoPE before VAE decode.

        The complete pinned-host block store survives across jobs. Its transient
        GPU staging slabs are discarded separately after synchronization."""
        if self.loaded:
            print("[flux] releasing FLUX shared weights + rope; preserving host denoiser store")
        self.model = List[ArcPointer[Flux1Offloaded]]()
        self.rope_cos = List[ArcPointer[Tensor]]()
        self.rope_sin = List[ArcPointer[Tensor]]()
        self.loaded = False
        self.loaded_lora = String("")
        self.loaded_width = 0
        self.loaded_height = 0

    def _load_model_shape[
        N_IMG_: Int, IMG_H2_: Int, IMG_W2_: Int
    ](mut self) raises:
        """Load the FLUX.1-dev DiT offloader handle + rope tables. Resident across
        BASE jobs; a LoRA job (or a LoRA change vs the resident handle) reloads."""
        var want_lora = _select_lora_path(self.params.loras)
        var want_lora_signature = _select_lora_signature(self.params.loras)
        var want_lora_multiplier = _select_lora_multiplier(self.params.loras)
        # Cancellation/failure can leave the offloader resident. Its RoPE is
        # shape-specific, so a LoRA path/weight OR shape change must rebuild the handle.
        if self.loaded and (
            self.loaded_lora != want_lora_signature
            or self.loaded_width != self.params.width
            or self.loaded_height != self.params.height
        ):
            print("[flux] resident LoRA/shape changed — reloading DiT")
            self._free_dit()
        if self.loaded:
            return
        _print_vram("before FLUX DiT offloader load")
        self.model = List[ArcPointer[Flux1Offloaded]]()
        if want_lora != String(""):
            print(
                "[flux] loading FLUX.1-dev DiT (offloaded) + LoRA overlay:",
                want_lora,
                "weight",
                want_lora_multiplier,
            )
            self.model.append(ArcPointer(Flux1Offloaded.load_with_lora(
                DIT_PATH, Flux1Config.dev(), want_lora, want_lora_multiplier, self.ctx
            )))
        else:
            print("[flux] loading FLUX.1-dev DiT (offloaded) from", DIT_PATH)
            self.model.append(ArcPointer(Flux1Offloaded.load(
                DIT_PATH, Flux1Config.dev(), self.ctx
            )))
        # Complete FP8 host-resident store for the denoise loop. Per-block
        # prefetch is pure async DMA from RAM on the copy stream, overlapped with
        # compute. The offloader handle stays for shared weights, LoRA overlay,
        # and the _block_model weight tables; its mmap pages are dropped below.
        if len(self.tloader) == 0:
            var plan = build_flux1_dev_block_plan()
            var tloader = TurboPlannedLoader.open(
                String(DIT_PATH), plan^, OffloadConfig.synchronous_single(), self.ctx,
                fill_block_store=False,
            )
            var host_blocks = tloader.pin_residents_fp8_host(1 << 60, self.ctx)
            tloader.require_all_blocks_memory_resident()
            tloader.release_checkpoint_pages()
            tloader.discard_unused_raw_streaming_slots(self.ctx)
            tloader.set_fp8h_overlap(True)
            print("[flux] host-resident denoiser blocks:", host_blocks, "/57")
            self.tloader.append(ArcPointer(tloader^))
        else:
            self.tloader[0][].require_all_blocks_memory_resident()
            print("[flux] reusing complete host-resident denoiser store: 57/57")
        self.model[0][].loader.sharded.release_to_os()
        # RoPE tables (resident with the offloader; rebuilt only on reload).
        var rope = build_flux1_rope_tables[N_IMG_, N_TXT, 24, 128](
            IMG_H2_, IMG_W2_, self.ctx, STDtype.BF16
        )
        self.rope_cos = List[ArcPointer[Tensor]]()
        self.rope_sin = List[ArcPointer[Tensor]]()
        self.rope_cos.append(ArcPointer(rope[0].clone(self.ctx)))
        self.rope_sin.append(ArcPointer(rope[1].clone(self.ctx)))
        self.loaded = True
        self.loaded_lora = want_lora_signature
        self.loaded_width = self.params.width
        self.loaded_height = self.params.height
        _print_vram("after FLUX DiT offloader load (resident)")

    def _load_model(mut self) raises:
        comptime for bi in range(DEFAULT_ASPECT_LADDER_LEN):
            comptime X100_BI = DEFAULT_ASPECT_LADDER_X100[bi]
            comptime LH_BI = aspect_lat_h_units(X100_BI, FLUX_PRODUCT_EDGE_UNITS)
            comptime LW_BI = aspect_lat_w_units(X100_BI, FLUX_PRODUCT_EDGE_UNITS)
            comptime IMG_H2_BI = LH_BI // 2
            comptime IMG_W2_BI = LW_BI // 2
            comptime N_IMG_BI = IMG_H2_BI * IMG_W2_BI
            if self.params.width == LW_BI * 8 and self.params.height == LH_BI * 8:
                self._load_model_shape[N_IMG_BI, IMG_H2_BI, IMG_W2_BI]()
                return
        raise Error("flux: admitted load shape was not compiled")

    def _prepare_job_shape[
        LATENT_H_: Int, LATENT_W_: Int, N_IMG_: Int
    ](mut self) raises:
        """Flow-match sigma table (honors steps) + seeded initial packed latent
        (honors seed). Mirrors flux_sample_cli.denoise's noise+pack."""
        self.sched = build_swarm_flux_schedule(
            self.executed_scheduler, self.params.steps
        )
        # Comfy's ddim_uniform intentionally uses range(1, timesteps,
        # timesteps//requested_steps), which can produce one extra interval for
        # non-divisor step counts. Execute the complete creator schedule instead
        # of silently dropping its final interval.
        self.runtime_steps = len(self.sched) - 1
        var nsh = [1, AE_IN_CHANNELS, LATENT_H_, LATENT_W_]
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
                base_h, var_h, AE_IN_CHANNELS, LATENT_H_, LATENT_W_,
                self.params.variation_strength,
            )
            noise = Tensor.from_host(blended, nsh.copy(), STDtype.F32, self.ctx)
        var packed = _pack_latent_shape[LATENT_H_, LATENT_W_](noise, self.ctx)
        self.latent = List[ArcPointer[Tensor]]()
        self.latent.append(ArcPointer(packed^))
        print(
            "[flux] job", self.params.job_id, ":", self.params.steps,
            "requested steps,", self.runtime_steps, "executed steps, guidance",
            self.guidance, "seed", self.params.seed,
            "size", self.params.width, "x", self.params.height,
            "(FLUX Dev guidance-distilled; negative discarded)",
        )

    def _prepare_job(mut self) raises:
        comptime for bi in range(DEFAULT_ASPECT_LADDER_LEN):
            comptime X100_BI = DEFAULT_ASPECT_LADDER_X100[bi]
            comptime LH_BI = aspect_lat_h_units(X100_BI, FLUX_PRODUCT_EDGE_UNITS)
            comptime LW_BI = aspect_lat_w_units(X100_BI, FLUX_PRODUCT_EDGE_UNITS)
            comptime N_IMG_BI = (LH_BI // 2) * (LW_BI // 2)
            if self.params.width == LW_BI * 8 and self.params.height == LH_BI * 8:
                self._prepare_job_shape[LH_BI, LW_BI, N_IMG_BI]()
                return
        raise Error("flux: admitted prepare shape was not compiled")

    # ── one denoise step (guidance-distilled single forward + Euler) ───────────
    # Verbatim from flux_sample_cli.denoise's per-step body.
    def _denoise_one_shape[N_IMG_: Int](mut self) raises:
        comptime S_ = N_TXT + N_IMG_
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
            _flux_forward_turbo[N_IMG_, N_TXT, S_](
                self.model[0][], self.tloader[0][],
                img_bf, self.caps[0][].txt, t_vec, Optional[Tensor](g_vec^),
                self.caps[0][].vector, self.rope_cos[0][], self.rope_sin[0][],
                self.ctx,
            ),
            STDtype.F32,
            self.ctx,
        )
        var x_new: Tensor
        if self.executed_sampler == "dpmpp_2m":
            var denoised = denoised_from_velocity(
                self.latent[0][], pred, t_curr, self.ctx
            )
            if not self.dpmpp_history.is_empty():
                self.dpmpp_second_order_steps += 1
            x_new = dpmpp_2m_step(
                self.latent[0][],
                denoised,
                t_curr,
                t_prev,
                self.dpmpp_history,
                self.ctx,
            )
            self.dpmpp_history.push(
                denoised^,
                lambda_from_sigma_f64(Float64(t_curr)),
            )
            self.dpmpp_update_steps += 1
        else:
            # Euler step: img = img + (t_prev - t_curr) * pred
            var dt = t_prev - t_curr
            x_new = add(self.latent[0][], mul_scalar(pred, dt, self.ctx), self.ctx)
        self.latent = List[ArcPointer[Tensor]]()
        self.latent.append(ArcPointer(x_new^))

    def _denoise_one(mut self) raises:
        comptime for bi in range(DEFAULT_ASPECT_LADDER_LEN):
            comptime X100_BI = DEFAULT_ASPECT_LADDER_X100[bi]
            comptime LH_BI = aspect_lat_h_units(X100_BI, FLUX_PRODUCT_EDGE_UNITS)
            comptime LW_BI = aspect_lat_w_units(X100_BI, FLUX_PRODUCT_EDGE_UNITS)
            comptime N_IMG_BI = (LH_BI // 2) * (LW_BI // 2)
            if self.params.width == LW_BI * 8 and self.params.height == LH_BI * 8:
                self._denoise_one_shape[N_IMG_BI]()
                return
        raise Error("flux: admitted denoise shape was not compiled")

    def _warm_denoiser_for_shape(mut self) raises:
        """Activate lazy CUDA/cuDNN kernels once per compiled image shape.

        The warm forward is non-mutating: it forces the Euler branch, restores
        the original latent and sampler selection, and never advances `cur`.
        """
        var key = String(self.params.width) + String("x") + String(self.params.height)
        for warmed in self.warmed_shapes:
            if warmed == key:
                print("[flux] denoiser kernel warm-up HIT for", key)
                return
        var saved_sampler = self.executed_sampler.copy()
        var saved_latent = self.latent[0][].clone(self.ctx)
        self.executed_sampler = String("euler")
        self._denoise_one()
        self.ctx.synchronize()
        self.latent = List[ArcPointer[Tensor]]()
        self.latent.append(ArcPointer(saved_latent^))
        self.executed_sampler = saved_sampler^
        self.warmed_shapes.append(key^)
        print("[flux] pre-step-0 denoiser kernel warm-up complete")

    # ── final decode + PNG(tEXt) ──────────────────────────────────────────────
    def _decode_and_save_shape[LATENT_H_: Int, LATENT_W_: Int](
        mut self
    ) raises -> String:
        var png_path = self.params.out_dir + "/" + self.params.job_id + ".png"
        # Keep only a tiny packed-latent clone, then release the offloaded DiT
        # before unpack + VAE decode to lower allocator high-water on 24 GB cards.
        var packed = self.latent[0][].clone(self.ctx)
        self.caps = List[ArcPointer[FluxCaps]]()
        self.sched = List[Float32]()
        self.latent = List[ArcPointer[Tensor]]()
        self.dpmpp_history_final_len = self.dpmpp_history.len()
        self.dpmpp_history = MultistepHistory(1)
        self.ctx.synchronize()
        if len(self.tloader) == 1:
            self.tloader[0][].discard_fp8h_device_staging()
        self._free_dit()
        cu_mempool_trim_current(0)
        self.ctx.synchronize()
        _print_vram("after resident release before VAE")
        var latent = _unpack_flux_packed_latent_shape[LATENT_H_, LATENT_W_](
            packed, self.ctx
        )
        # Prefer whole-image decode when VRAM allows; tiled degrades output (MJ-1054).
        var mem = cu_mem_get_info()
        var free_gib = Float64(mem.free_bytes) / 1073741824.0
        if mem.free_bytes > WHOLE_DECODE_MIN_FREE_BYTES:
            print("[flux] WHOLE-image decode (free=", free_gib,
                  "GiB) — tiled measured to degrade output (MJ-1054)")
            var dec = load_flux1_ldm_decoder[LATENT_H_, LATENT_W_](
                String(VAE_PATH), self.ctx
            )
            var wimg = dec.decode(latent, self.ctx)
            self.vae_decode_grid = String("whole_image")
            _save_rgb_png_with_text(wimg, png_path, self.params.params_json, self.ctx)
            return png_path
        # Fallback: GATED 3x3 flux_tiled_decode (parity-gated; NOT the worst-measured
        # 5x5-lowmem variant this backend previously used). See MJ-1054.
        print("[flux] tiled VAE decode FALLBACK (3x3 gated) — free=", free_gib,
              "GiB below whole-image threshold; tiled degrades output (MJ-1054)")
        var img = flux_tiled_decode[LATENT_H_, LATENT_W_](
            latent, String(VAE_PATH), self.ctx
        )
        self.vae_decode_grid = String("3x3_tiled_fallback")
        _save_rgb_png_with_text(img, png_path, self.params.params_json, self.ctx)
        return png_path

    def _decode_and_save(mut self) raises -> String:
        comptime for bi in range(DEFAULT_ASPECT_LADDER_LEN):
            comptime X100_BI = DEFAULT_ASPECT_LADDER_X100[bi]
            comptime LH_BI = aspect_lat_h_units(X100_BI, FLUX_PRODUCT_EDGE_UNITS)
            comptime LW_BI = aspect_lat_w_units(X100_BI, FLUX_PRODUCT_EDGE_UNITS)
            if self.params.width == LW_BI * 8 and self.params.height == LH_BI * 8:
                return self._decode_and_save_shape[LH_BI, LW_BI]()
        raise Error("flux: admitted decode shape was not compiled")

    def _clear_job(mut self):
        self.active = False
        self.phase = FPHASE_IDLE
        self.cur = 0
        self.cancel_flag = False
        self.announced = False
        self.caps = List[ArcPointer[FluxCaps]]()
        self.sched = List[Float32]()
        self.runtime_steps = 0
        self.latent = List[ArcPointer[Tensor]]()
        self.dpmpp_history = MultistepHistory(1)
        self.dpmpp_history_final_len = 0
        self.dpmpp_update_steps = 0
        self.dpmpp_second_order_steps = 0

    # ── the pull-based tick ───────────────────────────────────────────────────
    def step(mut self) raises -> StepResult:
        var r = StepResult()
        r.total = (
            self.runtime_steps if self.runtime_steps > 0 else self.params.steps
        )
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
                var encode_t0 = perf_counter_ns()
                self._encode()
                self.text_encode_seconds = Float64(perf_counter_ns() - encode_t0) / 1.0e9
                self._record_vram()
                self.announced = False
                self.phase = FPHASE_LOAD
                r.step = 0
                return r^
            if self.phase == FPHASE_LOAD:
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
                self._warm_denoiser_for_shape()
                self._record_vram()
                print("[flux] prefaulted worker image bytes=", prefault_self_executable())
                print(
                    "[flux] prefaulted mapped shared-library bytes=",
                    prefault_mapped_shared_libraries(),
                )
                self.prepare_seconds += Float64(perf_counter_ns() - prep_t0) / 1.0e9
                self.phase = FPHASE_DENOISE
                r.step = 0
                r.phase = String("sampling")
                return r^
            if self.phase == FPHASE_DENOISE:
                var denoise_t0 = perf_counter_ns()
                self._denoise_one()
                self.denoise_seconds += Float64(perf_counter_ns() - denoise_t0) / 1.0e9
                self._record_vram()
                self.cur += 1
                r.step = self.cur
                r.phase = String("sampling")
                if self.cur >= self.runtime_steps:
                    self.phase = FPHASE_DECODE
                return r^
            if not self.announced:
                # announce BEFORE the long blocking VAE-decode tick.
                self.announced = True
                r.step = self.runtime_steps
                r.phase = String("decoding")
                return r^
            var decode_t0 = perf_counter_ns()
            var path = self._decode_and_save()
            self.vae_decode_seconds = Float64(perf_counter_ns() - decode_t0) / 1.0e9
            self._record_vram()
            var manifest = self._write_result_manifest(path)
            print("[flux][manifest] saved:", manifest)
            r.step = self.runtime_steps
            self._clear_job()
            r.done = True
            r.output_path = path
            return r^
        except e:
            self._clear_job()
            r.failed = True
            r.error = String(e)
            return r^
