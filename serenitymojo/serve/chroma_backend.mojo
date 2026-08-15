# serenitymojo.serve.chroma_backend — the real Chroma1-HD GenBackend.
#
# Wraps the PROVEN serenitymojo/pipeline/chroma_pipeline_1024_multistep.mojo
# stages behind the pull-based GenBackend seam (backend.mojo). EVERY numeric
# convention is reused from that pipeline (its ChromaShared/block/pack helpers
# are imported and shape-specialized, not re-derived):
#
#   T5-XXL Unigram tokenizer (bit-exact vs HF, same tokenizer flux_backend
#   uses) → T5-XXL hidden [1,512,4096] cond + uncond (LIVE per-job encode —
#   Chroma is T5-ONLY, no CLIP) →
#   Chroma DiT (19 double + 38 single blocks staged from a complete pinned-host
#   FP8 store; no checkpoint access during denoise;
#   guidance via the distilled_guidance_layer approximator, NOT guidance_in) →
#   real CFG: pred = uncond + cfg * (cond - uncond) + flow-match Euler update →
#   FLUX VAE WHOLE-image decode (the pipeline's proven decode; ae.safetensors)
#   → PNG SIGNED (genparams tEXt).
#
# Unlike FLUX.1-dev (guidance-distilled, negative discarded), Chroma runs REAL
# CFG: the negative prompt IS encoded and drives the uncond forward; params.cfg
# is the CFG multiplier (publisher profile default 3.0).
#
# Residency model (16 GB GPU, Chroma1-HD ~17.8 GB BF16 on disk):
#   * The Chroma DiT is host-resident: ChromaShared (approximator +
#     x/context embedders + proj_out, small) + a complete FP8 host store that stages
#     ONE block at a time with ASYNC DOUBLE-BUFFERING (pinned host store +
#     two device slots + explicit copy stream: block i+1 DMA-stages while
#     block i computes — replaces the naive synchronous BlockLoader that
#     idled the GPU on every transfer, measured 4.78 s/step). The forward math
#     is byte-identical to chroma_pipeline_1024_multistep (_chroma_forward_turbo
#     below calls the same imported block functions in the same order). The
#     shared weights + loader handle + RoPE tables are loaded in the LOAD phase
#     and freed before VAE decode (every job reloads — the 16 GB card cannot
#     keep them through decode).
#   * The T5-XXL (~9.5 GB F16) encoder runs PER JOB in a fork+execv CHILD
#     PROCESS (chroma_encode_subprocess, the zimage-proven split): the child
#     encodes cond AND uncond, writes the caps to /tmp, and exits — process
#     death reclaims its VRAM unconditionally (in-process free + trim measured
#     to reclaim ~0 on this card: jobs 0075/0076). The parent stays ~1 GB.
#   * The FLUX VAE decode runs PER JOB after the DiT weights are freed +
#     mempool trimmed, in this order: (1) WHOLE-image decode in a fork+execv
#     CHILD PROCESS (chroma_decode_subprocess) — measured job-0078: ~11.7 GB
#     decode peak on the parent's ~3 GB floor OOMs the 16 GB card in-process;
#     job-0079: even the fresh-context child can lose the ~13.3 GB coin-flip.
#     (2) fallback: IN-PROCESS 3x3 TILED decode (flux_tiled_decode — authored
#     for exactly this post-offloaded-DiT high-water state; klein ships tiled
#     VAE decode in production on this card). The executed path is reported in
#     the result manifest (vae_decode_tile_grid).
#
# step() state machine: ENCODE (per-job, blocking — announced phase="encoding")
#   → LOAD (ChromaShared + complete host store + rope, announced phase="loading")
#   → DENOISE×steps (one CFG step per tick: cond forward + uncond forward +
#   CFG combine + Euler update) → DECODE (announced phase="decoding") → done.
#   cancel() makes the next step() return cancelled and frees per-job tensors.
#
# The finite seven-shape ~1 MP core is compiled: each arm specializes latent
# geometry, image-token count, joint-attention length, rectangular RoPE, and
# VAE decode. Product admission follows this exact finite ladder. Rectangular
# denoise remains slower than square on the 16GB product GPU.
#
# LoRA: one runtime additive Chroma overlay is supported.  The product path
# accepts torchref BFL-fused and Diffusers/PEFT projection keys and never
# modifies the checkpoint. img2img: NOT supported — fail loud.

from std.collections import Optional
from std.ffi import external_call
from max.gpu.host import DeviceContext
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
from serenitymojo.offload.plan import OffloadConfig, build_chroma1_hd_block_plan

from serenitymojo.models.dit.flux1_dit import build_flux1_rope_tables
from serenitymojo.models.chroma.chroma_lora_overlay import (
    ChromaLoraOverlay, load_chroma_lora,
)
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.elementwise import modulate
from serenitymojo.serve.chroma_encode_subprocess import encode_captions_subprocess
from serenitymojo.serve.chroma_decode_subprocess import (
    decode_whole_subprocess, decode_tiled_subprocess,
)
from serenitymojo.pipeline.flux_tiled_decode import flux_tiled_decode
from serenitymojo.ops.random import randn
from serenitymojo.ops.tensor_algebra import add, concat, mul_scalar, slice
from serenitymojo.sampling.swarmui_schedules import (
    build_swarm_flux_schedule,
)
from serenitymojo.sampling.dpmpp_2m import (
    MultistepHistory,
    denoised_from_velocity,
    dpmpp_2m_step,
    lambda_from_sigma_f64,
)
from serenitymojo.sampling.sampler_registry import (
    normalize_sampler_name, normalize_scheduler_name,
)
from serenitymojo.sampling.variation_noise import variation_noise_chw
from serenitymojo.training.aspect_buckets import (
    DEFAULT_ASPECT_LADDER_LEN, DEFAULT_ASPECT_LADDER_X100,
    aspect_lat_h_units, aspect_lat_w_units,
)

# The PROVEN Chroma pipeline math — imported, never re-derived. Its helpers run
# the approximator + 19 double + 38 single blocks plus
# norm_out/proj_out; the shape helpers are the same 2x2 patch pack.
from serenitymojo.pipeline.chroma_pipeline_1024_multistep import (
    ChromaShared, _pack_latent_shape, _unpack_latent_shape, _clone,
    _build_approximator_input, _approximator_forward,
    _double_block_shape, _single_block_shape, _linear_b, _pooled_row,
    _layer_norm_no_affine,
    CHROMA_CKPT, VAE_PATH,
    LC, N_TXT, HEADS, HEAD_DIM,
    N_DBL, N_SGL, MOD_SGL_OFF, MOD_DBL_IMG_OFF, MOD_DBL_TXT_OFF, MOD_IDX,
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
from serenitymojo.io.env import env_int

comptime CHROMA_CAP_CACHE_SLOTS = 4
comptime GENPARAMS_TEXT_KEY = "serenity.genparams.v1"

# Chroma1-HD publisher inference profile.
comptime DEFAULT_STEPS = 40
comptime DEFAULT_GUIDANCE = Float32(3.0)


comptime CPHASE_IDLE = 0
comptime CPHASE_ENCODE = 1
comptime CPHASE_LOAD = 2
comptime CPHASE_DENOISE = 3
comptime CPHASE_DECODE = 4
comptime CHROMA_PRODUCT_EDGE_UNITS = 16
# Chroma's whole-image decoder requires 12.5 GiB device-global free.  Reserve
# another 2.5 GiB for CFG denoise allocator growth and use the remainder for
# persistent device FP8 blocks. Runtime-sweepable for measured hardware gates.
comptime CHROMA_DENOISE_VRAM_RESERVE_MIB = 15360


def _chroma_shape_supported(width: Int, height: Int) -> Bool:
    # Release gate: rectangular specializations compile and pass geometry
    # parity, but a measured 1152x896 product run took ~111 s/step versus the
    # established ~4.5 s/step square path. Do not advertise unusable shapes.
    return width == 1024 and height == 1024


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
        String("echo -n '[chroma][vram] ") + tag
        + ": ' && nvidia-smi --query-gpu=memory.used --format=csv,noheader"
    )


def _save_rgb_png_with_text(
    rgb: Tensor, path: String, params_json: String, ctx: DeviceContext
) raises:
    """[1,3,H,W] SIGNED float tensor → 8-bit RGB PNG with the job params in a
    serenity.genparams.v1 tEXt chunk. Quantization math == save_png's
    (_quantize, ValueRange.SIGNED); only the writer differs (tEXt support).
    Identical to flux_backend/qwenimage_backend._save_rgb_png_with_text."""
    var shape = rgb.shape()
    if len(shape) != 4 or shape[0] != 1 or shape[1] != 3:
        raise Error("chroma_backend: expected [1,3,H,W] rgb tensor")
    var height = shape[2]
    var width = shape[3]
    var host = rgb.to_host(ctx)
    var plane = height * width
    if len(host) != 3 * plane:
        raise Error("chroma_backend: rgb to_host size mismatch")
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


# ── Turbo-streamed Chroma DiT forward ─────────────────────────────────────────
# BYTE-IDENTICAL math to chroma_pipeline_1024_multistep._chroma_forward: the
# same imported _approximator_forward / _double_block / _single_block calls, in
# the same order, with the same mod-slice offsets and the same norm_out/proj_out
# tail. ONLY the weight staging changes: block i+1 is DMA-staged from the
# complete FP8 host-resident store on the explicit
# copy stream WHILE block i computes on the default stream (await_block fences
# via DeviceEvent; mark_active_block_done lets the copy stream reuse a
# slot only after the block's kernels are queued).
def _chroma_forward_turbo[N_IMG_: Int](
    shared: ChromaShared,
    mut loader: TurboPlannedLoader,
    img_packed: Tensor,
    txt: Tensor,
    timestep: Float32,
    rope_cos: Tensor,
    rope_sin: Tensor,
    real_txt_len: Int,
    lora: Optional[ArcPointer[ChromaLoraOverlay]],
    ctx: DeviceContext,
) raises -> Tensor:
    # Stage block 0 first: its DMA overlaps the approximator + embedder compute.
    loader.set_config(OffloadConfig.synchronous_single())
    loader.prefetch_with_ctx(0, ctx)

    # 1. Approximator: build per-step modulation table [1, 344, 3072]
    var approx_in = _build_approximator_input(timestep, ctx)
    var pooled_temb = _approximator_forward(shared, approx_in, ctx)

    # 2. Input projections
    var img = _linear_b(
        img_packed,
        shared._w("x_embedder.weight"),
        shared._w("x_embedder.bias"),
        ctx,
    )
    var x_txt = _linear_b(
        txt,
        shared._w("context_embedder.weight"),
        shared._w("context_embedder.bias"),
        ctx,
    )

    # 3. Double blocks (19) — await this block's slot, immediately stage the
    #    NEXT block into the idle slot, then queue this block's compute.
    for i in range(N_DBL):
        var handle = loader.await_block(i, ctx)
        loader.prefetch_next_with_ctx(i, ctx)

        # Slice mod params: img [114+6i : 114+6i+6], txt [228+6i : 228+6i+6]
        var img_mod_s = slice(pooled_temb, 1, MOD_DBL_IMG_OFF + 6 * i, 6, ctx)
        var txt_mod_s = slice(pooled_temb, 1, MOD_DBL_TXT_OFF + 6 * i, 6, ctx)

        var res = _double_block_shape[N_IMG_](
            i, img, x_txt, img_mod_s, txt_mod_s, rope_cos, rope_sin,
            real_txt_len, handle.block, ctx, lora,
        )
        img = _clone(res[0], ctx)
        x_txt = _clone(res[1], ctx)
        loader.mark_active_block_done(ctx)

    # 4. Concat [txt | img] -> [1, S, HIDDEN] for single blocks
    var x = concat(1, ctx, x_txt, img)

    # 5. Single blocks (38) — same await → prefetch-next → compute rotation.
    for i in range(N_SGL):
        var plan_idx = N_DBL + i
        var handle = loader.await_block(plan_idx, ctx)
        loader.prefetch_next_with_ctx(plan_idx, ctx)

        # Slice mod params: [3*i : 3*i+3]
        var sgl_mod_s = slice(pooled_temb, 1, MOD_SGL_OFF + 3 * i, 3, ctx)

        x = _single_block_shape[N_IMG_](
            i, x, sgl_mod_s, rope_cos, rope_sin, real_txt_len,
            handle.block, ctx, lora,
        )
        loader.mark_active_block_done(ctx)

    # 6. Extract img portion, apply norm_out + proj_out (identical to pipeline)
    var img_out = slice(x, 1, N_TXT, N_IMG_, ctx)
    var norm_shift = _pooled_row(pooled_temb, MOD_IDX - 2, ctx)
    var norm_scale = _pooled_row(pooled_temb, MOD_IDX - 1, ctx)
    var normed = _layer_norm_no_affine(img_out, ctx)
    var modulated = modulate(normed, norm_scale, norm_shift, ctx)
    return _linear_b(
        modulated,
        shared._w("proj_out.weight"),
        shared._w("proj_out.bias"),
        ctx,
    )


struct ChromaBackend(GenBackend, Movable):
    var ctx: DeviceContext

    # ── DiT handles (GPU state freed before VAE; host store kept warm) ─────
    # ArcPointer wrappers: ChromaShared / TurboPlannedLoader / Tensor are
    # Movable-not-Copyable and List[T] requires T: Copyable — Arc is Copyable
    # (refcount), so List[Arc[..]] holds the 0/1.
    var loaded: Bool
    var shared: List[ArcPointer[ChromaShared]]   # 0/1 (approximator+embedders)
    var loader: List[ArcPointer[TurboPlannedLoader]]  # 0/1 complete host store
    var rope_cos: List[ArcPointer[Tensor]]       # 0/1
    var rope_sin: List[ArcPointer[Tensor]]       # 0/1
    var lora: List[ArcPointer[ChromaLoraOverlay]]  # 0/1 additive runtime overlay
    var lora_factor_count: Int
    # Exact prompt-pair T5 conditioning is only ~8 MiB.  Preserve it across jobs
    # so seed/step/CFG variations do not reload T5-XXL.
    # Keyed (prompt \x1f negative) multi-slot cache, capacity
    # CHROMA_CAP_CACHE_SLOTS drop-oldest — the old single slot missed 100%
    # on prompt-alternating A/B and grid workloads. Parallel lists share the
    # slot index.
    var cap_cache_keys: List[String]
    var cap_cache_cond: List[ArcPointer[Tensor]]
    var cap_cache_uncond: List[ArcPointer[Tensor]]
    var cap_cache_cond_real_len: List[Int]
    var cap_cache_uncond_real_len: List[Int]

    # ── per-job state (cleared on done/failed/cancelled) ──
    var active: Bool
    var cancel_flag: Bool
    var phase: Int
    var announced: Bool
    var cur: Int
    var params: JobParams
    var guidance: Float32
    var executed_sampler: String
    var executed_scheduler: String
    var dpmpp_history: MultistepHistory
    var dpmpp_history_final_len: Int
    var dpmpp_update_steps: Int
    var dpmpp_second_order_steps: Int
    var t5_cond: List[ArcPointer[Tensor]]        # 0/1 [1,512,4096] BF16
    var t5_uncond: List[ArcPointer[Tensor]]      # 0/1 [1,512,4096] BF16
    var cond_real_len: Int                       # unpadded T5 token count (MJ-1048 mask)
    var uncond_real_len: Int
    var sched: List[Float32]                     # flow-match sigma table (steps+1)
    var runtime_steps: Int                       # len(sched)-1; DDIM may exceed request
    var latent: List[ArcPointer[Tensor]]         # 0/1 (packed [1,N_IMG,64] BF16)
    var vae_decode_grid: String                  # executed decode path (manifest)
    var job_t0_ns: Int
    var load_seconds: Float64
    var text_encode_seconds: Float64
    var text_conditioning_cache_hit: Bool
    var device_resident_blocks: Int
    var device_resident_budget_bytes: Int
    var prepare_seconds: Float64
    var denoise_seconds: Float64
    var vae_decode_seconds: Float64
    var total_vram_bytes: Int
    var min_free_bytes: Int

    def __init__(out self) raises:
        self.ctx = DeviceContext()
        self.loaded = False
        self.shared = List[ArcPointer[ChromaShared]]()
        self.loader = List[ArcPointer[TurboPlannedLoader]]()
        self.rope_cos = List[ArcPointer[Tensor]]()
        self.rope_sin = List[ArcPointer[Tensor]]()
        self.lora = List[ArcPointer[ChromaLoraOverlay]]()
        self.lora_factor_count = 0
        self.cap_cache_keys = List[String]()
        self.cap_cache_cond = List[ArcPointer[Tensor]]()
        self.cap_cache_uncond = List[ArcPointer[Tensor]]()
        self.cap_cache_cond_real_len = List[Int]()
        self.cap_cache_uncond_real_len = List[Int]()
        self.active = False
        self.cancel_flag = False
        self.phase = CPHASE_IDLE
        self.announced = False
        self.cur = 0
        self.params = JobParams()
        self.guidance = DEFAULT_GUIDANCE
        self.executed_sampler = String("chroma_cfg_flowmatch_euler")
        self.executed_scheduler = String("")
        self.dpmpp_history = MultistepHistory(1)
        self.dpmpp_history_final_len = 0
        self.dpmpp_update_steps = 0
        self.dpmpp_second_order_steps = 0
        self.t5_cond = List[ArcPointer[Tensor]]()
        self.t5_uncond = List[ArcPointer[Tensor]]()
        self.cond_real_len = N_TXT
        self.uncond_real_len = N_TXT
        self.sched = List[Float32]()
        self.runtime_steps = 0
        self.latent = List[ArcPointer[Tensor]]()
        self.vae_decode_grid = String("")
        self.job_t0_ns = Int(0)
        self.load_seconds = 0.0
        self.text_encode_seconds = 0.0
        self.text_conditioning_cache_hit = False
        self.device_resident_blocks = 0
        self.device_resident_budget_bytes = 0
        self.prepare_seconds = 0.0
        self.denoise_seconds = 0.0
        self.vae_decode_seconds = 0.0
        self.total_vram_bytes = 0
        self.min_free_bytes = 0

    def backend_name(self) -> String:
        return String("chroma")

    def model_name(self) -> String:
        return String("Chroma1-HD")

    def resident_model(self) -> String:
        """Matches the /v1/models scan entry for the resident checkpoint."""
        return String("chroma1_hd_bf16.safetensors") if self.loaded else String("")

    # ── job admission ─────────────────────────────────────────────────────────
    def start(mut self, params: JobParams) raises:
        if self.active:
            raise Error("ChromaBackend.start: a job is already running")
        # Base shared weights are retained across jobs. A LoRA request needs a
        # freshly overlaid shared state; evict the base before admission so the
        # LOAD phase cannot accidentally reuse it.
        if self.loaded and len(params.loras) > 0:
            self._free_dit()
        reject_unsupported_common_runtime_params(params, String("chroma"))
        reject_unsupported_reference_image_params(params, String("chroma"))
        reject_unsupported_inpaint_conditioning_params(params, String("chroma"))
        reject_unsupported_qwen_edit_conditioning_params(params, String("chroma"))
        reject_unsupported_conditioning_mask_params(params, String("chroma"))
        reject_unsupported_mask_image_params(params, String("chroma"))
        reject_unsupported_lanpaint_params(params, String("chroma"))
        # Chroma's SwarmUI creator default is Euler + beta.  Keep the previous
        # Serenity simple flow schedule available as an explicit user choice.
        var norm_sampler = normalize_sampler_name(params.sampler)
        if norm_sampler == String(""):
            norm_sampler = String("euler")
        if not (
            norm_sampler == String("euler")
            or norm_sampler == String("flowmatch_euler")
            or norm_sampler == String("dpmpp_2m")
        ):
            raise Error(
                String("chroma: unsupported sampler '") + params.sampler
                + String("'; supported: euler/flowmatch_euler, dpmpp_2m")
            )
        var norm_scheduler = normalize_scheduler_name(params.scheduler)
        if norm_scheduler == String(""):
            norm_scheduler = String("beta")
        if not (
            norm_scheduler == String("normal")
            or norm_scheduler == String("karras")
            or norm_scheduler == String("exponential")
            or norm_scheduler == String("simple")
            or norm_scheduler == String("ddim_uniform")
            or norm_scheduler == String("sgm_uniform")
            or norm_scheduler == String("beta")
            or norm_scheduler == String("linear_quadratic")
            or norm_scheduler == String("kl_optimal")
        ):
            raise Error(
                String("chroma: unsupported scheduler '") + params.scheduler
                + String("'; supported: normal, karras, exponential, simple,"
                         " ddim_uniform, sgm_uniform, beta, linear_quadratic,"
                         " kl_optimal")
            )
        if not _chroma_shape_supported(params.width, params.height):
            raise Error(
                String("chroma: unsupported size ") + String(params.width)
                + "x" + String(params.height)
                + " — only 1024x1024 is product-admitted; rectangular Chroma"
                + " core arms remain gated after measured ~24x denoise slowdown"
            )
        if len(params.loras) > 1:
            raise Error(
                "chroma: this product path currently accepts one additive LoRA;"
                " remove extra LoRAs"
            )
        if len(params.loras) == 1 and params.loras[0].name == String(""):
            raise Error("chroma: selected LoRA path is empty")
        if params.init_image.byte_length() > 0:
            raise Error(
                "chroma: img2img is not supported for Chroma1-HD yet;"
                " submit without an init image"
            )
        # Warn-loud (never silently drop) on any advanced-sampling knob set but
        # unsupported by this fixed CFG flow-match Euler path.
        warn_unsupported_advanced_sampling_params(params, String("chroma"), List[String]())
        self.params = params.copy()
        self.executed_sampler = (
            String("dpmpp_2m")
            if norm_sampler == String("dpmpp_2m")
            else String("chroma_cfg_flowmatch_euler")
        )
        self.executed_scheduler = norm_scheduler
        self.dpmpp_history = MultistepHistory(1)
        self.dpmpp_history_final_len = 0
        self.dpmpp_update_steps = 0
        self.dpmpp_second_order_steps = 0
        self.runtime_steps = 0
        # Honor steps at runtime; <=0 means unset/invalid -> the verified
        # Publisher default is 40 inference steps.
        if self.params.steps <= 0:
            self.params.steps = DEFAULT_STEPS
        # Chroma runs REAL CFG: params.cfg is the CFG multiplier
        # (pred = uncond + cfg*(cond-uncond)). cfg<=0 means unset/invalid ->
        # the publisher default 3.0.
        self.guidance = Float32(self.params.cfg) if self.params.cfg > 0.0 else DEFAULT_GUIDANCE
        self.active = True
        self.cancel_flag = False
        self.cur = 0
        self.announced = False
        self.phase = CPHASE_ENCODE
        self.job_t0_ns = perf_counter_ns()
        self.load_seconds = 0.0
        self.text_encode_seconds = 0.0
        self.text_conditioning_cache_hit = False
        self.device_resident_blocks = 0
        self.device_resident_budget_bytes = 0
        self.prepare_seconds = 0.0
        self.denoise_seconds = 0.0
        self.vae_decode_seconds = 0.0
        self.lora_factor_count = 0
        var mem = cu_mem_get_info()
        self.total_vram_bytes = mem.total_bytes
        self.min_free_bytes = mem.free_bytes
        self._record_vram()

    def cancel(mut self):
        self.cancel_flag = True

    def between_jobs_trim(mut self) raises:
        """Reclaim the per-job transient peak (T5-XXL encoder ~9.5 GB, staged
        DiT blocks, the VAE decoder, 1024² forward + decode activations) back to
        the OS via cuMemPoolTrimTo. The host denoiser store survives; GPU state
        is released before decode."""
        var before = cu_mem_get_info()
        self.ctx.synchronize()
        cu_mempool_trim_current(0)
        self.ctx.synchronize()
        var after = cu_mem_get_info()
        print("[chroma] between-jobs trim: used",
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
        var manifest_path = png_path + String(".chroma_daemon_result.json")
        var denoise_per_step = Float64(0.0)
        if self.runtime_steps > 0:
            denoise_per_step = self.denoise_seconds / Float64(self.runtime_steps)
        var total_wall_seconds = Float64(perf_counter_ns() - self.job_t0_ns) / 1.0e9
        var peak_mib = Float64(0.0)
        if self.total_vram_bytes > 0 and self.min_free_bytes > 0:
            peak_mib = peak_vram_mib(self.total_vram_bytes, self.min_free_bytes)

        var content = String("{\n")
        content += String('  "schema":"serenity.chroma.daemon_result.v1",\n')
        content += String('  "backend":"chroma_daemon",\n')
        content += String('  "model":"chroma1-hd",\n')
        content += String('  "readiness_label":"experimental",\n')
        content += String('  "accepted_sampler_parity":false,\n')
        content += String('  "accepted_speed_parity":false,\n')
        content += String('  "run_identity":{\n')
        content += String('    "job_id":"') + json_escape(self.params.job_id) + String('",\n')
        content += String('    "prompt":"') + json_escape(self.params.prompt) + String('",\n')
        content += String('    "negative":"') + json_escape(self.params.negative) + String('",\n')
        content += String('    "negative_prompt_used":true,\n')
        content += String('    "seed":') + String(self.params.seed) + String(",\n")
        content += String('    "resolution":{"width":') + String(self.params.width) + String(',"height":') + String(self.params.height) + String("},\n")
        content += String('    "steps":') + String(self.params.steps) + String(",\n")
        content += String('    "executed_steps":') + String(self.runtime_steps) + String(",\n")
        content += String('    "cfg":') + String(self.guidance) + String(",\n")
        content += String('    "sampler_registry_backend":"chroma",\n')
        content += String('    "requested_sampler":"') + json_escape(self.params.sampler) + String('",\n')
        content += String('    "requested_scheduler":"') + json_escape(self.params.scheduler) + String('",\n')
        content += String('    "executed_sampler":"') + json_escape(self.executed_sampler) + String('",\n')
        content += String('    "executed_scheduler":"chroma_swarmui_') + json_escape(self.executed_scheduler) + String('",\n')
        content += String('    "schedule_source":"swarmui_comfy_model_sampling_flux_1_15",\n')
        content += String('    "variation_seed":') + String(self.params.variation_seed) + String(",\n")
        content += String('    "variation_strength":') + String(self.params.variation_strength) + String(",\n")
        content += String('    "variation_applied":') + json_bool(self.params.variation_strength > 0.0) + String(",\n")
        content += String('    "released_streamed_dit_before_unpack":true,\n')
        content += String('    "image_index":') + String(self.params.image_index) + String(",\n")
        content += String('    "image_count":') + String(self.params.image_count) + String(",\n")
        content += String('    "lora_count":') + String(len(self.params.loras)) + String(",\n")
        content += String('    "loaded_lora":"') + json_escape(
            self.params.loras[0].name if len(self.params.loras) == 1 else String("")
        ) + String('",\n')
        content += String('    "lora_weight":') + String(
            self.params.loras[0].weight if len(self.params.loras) == 1 else Float64(1.0)
        ) + String(",\n")
        content += String('    "lora_factor_count":') + String(self.lora_factor_count) + String(",\n")
        content += String('    "sampler_trace":{"history_capacity":1,"history_final_len":')
        content += String(self.dpmpp_history_final_len) + String(',"dpmpp_update_steps":')
        content += String(self.dpmpp_update_steps) + String(',"dpmpp_second_order_steps":')
        content += String(self.dpmpp_second_order_steps) + String("},\n")
        content += String('    "cond_real_len":') + String(self.cond_real_len) + String(",\n")
        content += String('    "uncond_real_len":') + String(self.uncond_real_len) + String(",\n")
        content += String('    "text_conditioning_cache_hit":') + json_bool(
            self.text_conditioning_cache_hit
        ) + String(",\n")
        content += String('    "vae_decode_tile_grid":"') + json_escape(self.vae_decode_grid) + String('",\n')
        content += String('    "dtype":"bf16_dit_bf16_latent"\n')
        content += String("  },\n")
        content += String('  "mojo":{\n')
        content += String('    "load_seconds":') + String(self.load_seconds) + String(",\n")
        content += String('    "text_encode_seconds":') + String(self.text_encode_seconds) + String(",\n")
        content += String('    "device_resident_blocks":') + String(
            self.device_resident_blocks
        ) + String(",\n")
        content += String('    "device_resident_budget_mib":') + String(
            self.device_resident_budget_bytes // (1024 * 1024)
        ) + String(",\n")
        content += String('    "prepare_seconds":') + String(self.prepare_seconds) + String(",\n")
        content += String('    "denoise_seconds":') + String(self.denoise_seconds) + String(",\n")
        content += String('    "denoise_seconds_per_step":') + String(denoise_per_step) + String(",\n")
        content += String('    "vae_decode_seconds":') + String(self.vae_decode_seconds) + String(",\n")
        content += String('    "total_wall_seconds":') + String(total_wall_seconds) + String(",\n")
        content += String('    "peak_vram_mib":') + String(peak_mib) + String(",\n")
        content += String('    "artifact_paths":["') + json_escape(png_path) + String('","') + json_escape(manifest_path) + String('"]\n')
        content += String("  },\n")
        content += String('  "output_png":"') + json_escape(png_path) + String('",\n')
        content += String('  "note":"Rust-server Mojo worker product-path result; Chroma1-HD uses live per-job T5-XXL encode, a complete host-resident FP8 DiT store with RAM-to-GPU staging, distilled-guidance approximator, and real CFG. Speed parity remains unaccepted until paired baseline evidence exists."\n')
        content += String("}\n")
        write_text_file(manifest_path, content)
        return manifest_path

    # ── per-job prep ───────────────────────────────────────────────────────────
    def _encode(mut self) raises:
        """LIVE T5-XXL hidden encode of params.prompt AND params.negative
        (Chroma is T5-ONLY: no CLIP) in a fork+execv CHILD PROCESS
        (chroma_encode_subprocess, the zimage-proven split): the child loads the
        ~9.5 GB encoder in a FRESH CUDA context, writes the [1,512,4096] BF16
        caps + unpadded token counts to /tmp, and EXITS — process death reclaims
        its VRAM unconditionally. MEASURED WHY (jobs 0075/0076): loading T5 in
        THIS process grew the pool by ~12.8 GB and neither dropping the encoder
        nor host-bouncing the caps let cuMemPoolTrimTo return it (13841 MiB
        used after trim, was 1073 pre-load) → VAE decode OOM'd the 16 GB card.
        Falls back to the in-process encode on any subprocess failure. The
        encode math is byte-identical either way; the real lengths feed the DiT
        T5 pad-row attention mask (MJ-1048)."""
        var want_key = self.params.prompt + String("\x1f") + self.params.negative
        for slot in range(len(self.cap_cache_keys)):
            if self.cap_cache_keys[slot] != want_key:
                continue
            print("[chroma] conditioning cache HIT (slot", slot, "of",
                  len(self.cap_cache_keys), ") — skipping T5")
            self.text_conditioning_cache_hit = True
            self.t5_cond = List[ArcPointer[Tensor]]()
            self.t5_cond.append(ArcPointer(
                self.cap_cache_cond[slot][].clone(self.ctx)
            ))
            self.t5_uncond = List[ArcPointer[Tensor]]()
            self.t5_uncond.append(ArcPointer(
                self.cap_cache_uncond[slot][].clone(self.ctx)
            ))
            self.cond_real_len = self.cap_cache_cond_real_len[slot]
            self.uncond_real_len = self.cap_cache_uncond_real_len[slot]
            return

        _print_vram("before T5-XXL encode (subprocess)")
        var caps = encode_captions_subprocess(
            self.params.prompt, self.params.negative, self.ctx
        )
        self.cond_real_len = caps.real_cond
        self.uncond_real_len = caps.real_uncond
        # _clone (bit-identical device copy, ~4 MiB each): Mojo forbids moving
        # Tensor fields out of the caps struct, and borrowing them beyond this
        # scope isn't possible — the copy is trivial next to the encode itself.
        self.t5_cond = List[ArcPointer[Tensor]]()
        self.t5_cond.append(ArcPointer(_clone(caps.cond, self.ctx)))
        self.t5_uncond = List[ArcPointer[Tensor]]()
        self.t5_uncond.append(ArcPointer(_clone(caps.uncond, self.ctx)))
        if len(self.cap_cache_keys) >= CHROMA_CAP_CACHE_SLOTS:
            _ = self.cap_cache_keys.pop(0)
            _ = self.cap_cache_cond.pop(0)
            _ = self.cap_cache_uncond.pop(0)
            _ = self.cap_cache_cond_real_len.pop(0)
            _ = self.cap_cache_uncond_real_len.pop(0)
        self.cap_cache_keys.append(
            self.params.prompt + String("\x1f") + self.params.negative
        )
        self.cap_cache_cond.append(ArcPointer(self.t5_cond[0][].clone(self.ctx)))
        self.cap_cache_uncond.append(ArcPointer(self.t5_uncond[0][].clone(self.ctx)))
        self.cap_cache_cond_real_len.append(self.cond_real_len)
        self.cap_cache_uncond_real_len.append(self.uncond_real_len)

    def _free_dit(mut self):
        """Drop GPU shared weights and RoPE while preserving the host store."""
        if self.loaded:
            print("[chroma] releasing shared weights + rope; preserving host denoiser store")
        self.shared = List[ArcPointer[ChromaShared]]()
        self.rope_cos = List[ArcPointer[Tensor]]()
        self.rope_sin = List[ArcPointer[Tensor]]()
        self.lora = List[ArcPointer[ChromaLoraOverlay]]()
        self.loaded = False

    def _load_model_shape[LH_: Int, LW_: Int, N_IMG_: Int](mut self) raises:
        """Load the Chroma shared weights (approximator + embedders + proj_out),
        build the complete FP8 host store, and build the rope tables. Every DiT
        block is host-resident before the first denoise step."""
        if self.loaded:
            return
        _print_vram("before Chroma shared-weight load")
        self.lora = List[ArcPointer[ChromaLoraOverlay]]()
        if len(self.params.loras) == 1:
            print(
                "[chroma] loading additive LoRA:",
                self.params.loras[0].name,
                "weight",
                self.params.loras[0].weight,
            )
            var overlay = load_chroma_lora(
                self.params.loras[0].name,
                N_DBL,
                N_SGL,
                Float32(self.params.loras[0].weight),
                self.ctx,
            )
            self.lora_factor_count = overlay.count()
            self.lora.append(ArcPointer(overlay^))
        else:
            self.lora_factor_count = 0
            self.lora.append(ArcPointer(ChromaLoraOverlay.empty()))
        print("[chroma] loading Chroma1-HD shared weights from", CHROMA_CKPT)
        self.shared = List[ArcPointer[ChromaShared]]()
        self.shared.append(ArcPointer(ChromaShared.load(String(CHROMA_CKPT), self.ctx)))
        # Complete FP8 host-resident store: every block is copied once before
        # step 0, then staged from RAM. The raw ~17.8 GiB pinned store is avoided.
        if len(self.loader) == 0:
            var plan = build_chroma1_hd_block_plan()
            var loader = TurboPlannedLoader.open(
                String(CHROMA_CKPT), plan^, OffloadConfig.synchronous_single(), self.ctx,
                fill_block_store=False,
            )
            var host_blocks = loader.pin_residents_fp8_host(1 << 60, self.ctx)
            loader.require_all_blocks_memory_resident()
            loader.release_checkpoint_pages()
            loader.discard_unused_raw_streaming_slots(self.ctx)
            loader.set_fp8h_overlap(True)
            print("[chroma] host-resident denoiser blocks:", host_blocks, "/57")
            self.loader.append(ArcPointer(loader^))
        else:
            self.loader[0][].require_all_blocks_memory_resident()
            print("[chroma] reusing complete host-resident denoiser store: 57/57")
        # RoPE tables — the pipeline's Stage 4 (same for all steps).
        var rope = build_flux1_rope_tables[N_IMG_, N_TXT, HEADS, HEAD_DIM](
            LH_ // 2, LW_ // 2, self.ctx, STDtype.BF16
        )
        self.rope_cos = List[ArcPointer[Tensor]]()
        self.rope_sin = List[ArcPointer[Tensor]]()
        self.rope_cos.append(ArcPointer(_clone(rope[0], self.ctx)))
        self.rope_sin.append(ArcPointer(_clone(rope[1], self.ctx)))
        self.loaded = True
        _print_vram("after Chroma shared-weight load (blocks stream per step)")

    def _load_model(mut self) raises:
        comptime for bi in range(DEFAULT_ASPECT_LADDER_LEN):
            comptime X100_BI = DEFAULT_ASPECT_LADDER_X100[bi]
            comptime LH_BI = aspect_lat_h_units(X100_BI, CHROMA_PRODUCT_EDGE_UNITS)
            comptime LW_BI = aspect_lat_w_units(X100_BI, CHROMA_PRODUCT_EDGE_UNITS)
            comptime N_IMG_BI = (LH_BI // 2) * (LW_BI // 2)
            if self.params.width == LW_BI * 8 and self.params.height == LH_BI * 8:
                self._load_model_shape[LH_BI, LW_BI, N_IMG_BI]()
                return
        raise Error("chroma: admitted load shape was not compiled")

    def _ensure_denoise_residency(mut self) raises:
        if len(self.loader) != 1:
            raise Error("chroma: denoise residency requested before loader init")
        var reserve_mib = env_int(
            String("SERENITY_CHROMA_DENOISE_RESERVE_MIB"),
            CHROMA_DENOISE_VRAM_RESERVE_MIB,
        )
        var free_bytes = cu_mem_get_info().free_bytes
        var reserve_bytes = reserve_mib * 1024 * 1024
        var budget = 0
        if free_bytes > reserve_bytes:
            budget = free_bytes - reserve_bytes
        var promoted = self.loader[0][].promote_fp8_host_to_device(
            budget, self.ctx, True, reserve_bytes
        )
        self.device_resident_blocks = promoted
        self.device_resident_budget_bytes = budget
        print(
            "[chroma] adaptive device residency:", promoted, "/57 blocks;",
            budget // (1024 * 1024), "MiB budget;", reserve_mib,
            "MiB denoise/decode reserve",
        )
        _print_vram("after adaptive Chroma device residency")

    def _prepare_job_shape[LH_: Int, LW_: Int, N_IMG_: Int](mut self) raises:
        """Selected creator-compatible sigma table + seeded initial packed latent."""
        self.sched = build_swarm_flux_schedule(
            self.executed_scheduler, self.params.steps
        )
        # Comfy's ddim_uniform can contain one more denoise interval than the
        # requested count when 10,000 is not evenly divisible. Preserve and
        # execute that exact creator schedule.
        self.runtime_steps = len(self.sched) - 1
        var nsh: List[Int] = [1, LC, LH_, LW_]
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
                base_h, var_h, LC, LH_, LW_,
                self.params.variation_strength,
            )
            noise = Tensor.from_host(blended, nsh.copy(), STDtype.BF16, self.ctx)
        var packed = _pack_latent_shape[LH_, LW_](noise, self.ctx)
        self.latent = List[ArcPointer[Tensor]]()
        self.latent.append(ArcPointer(packed^))
        print(
            "[chroma] job", self.params.job_id, ":", self.params.steps,
            "requested steps,", self.runtime_steps, "executed steps, cfg",
            self.guidance, "seed", self.params.seed,
            "size", self.params.width, "x", self.params.height,
            "(Chroma real CFG; negative prompt IS used)",
        )

    def _prepare_job(mut self) raises:
        comptime for bi in range(DEFAULT_ASPECT_LADDER_LEN):
            comptime X100_BI = DEFAULT_ASPECT_LADDER_X100[bi]
            comptime LH_BI = aspect_lat_h_units(X100_BI, CHROMA_PRODUCT_EDGE_UNITS)
            comptime LW_BI = aspect_lat_w_units(X100_BI, CHROMA_PRODUCT_EDGE_UNITS)
            comptime N_IMG_BI = (LH_BI // 2) * (LW_BI // 2)
            if self.params.width == LW_BI * 8 and self.params.height == LH_BI * 8:
                self._prepare_job_shape[LH_BI, LW_BI, N_IMG_BI]()
                return
        raise Error("chroma: admitted prepare shape was not compiled")

    # ── one denoise step (cond + uncond forward, CFG combine, Euler) ──────────
    # Verbatim from chroma_pipeline_1024_multistep's Stage 7 per-step body.
    def _denoise_one_shape[N_IMG_: Int](mut self) raises:
        var i = self.cur
        var t_curr = self.sched[i]
        var t_next = self.sched[i + 1]
        var dt = t_next - t_curr  # negative (descending schedule)

        # Conditioned forward (host-resident; math identical to _chroma_forward)
        var pred_cond = _chroma_forward_turbo[N_IMG_](
            self.shared[0][], self.loader[0][], self.latent[0][],
            self.t5_cond[0][], t_curr, self.rope_cos[0][], self.rope_sin[0][],
            self.cond_real_len,
            Optional[ArcPointer[ChromaLoraOverlay]](self.lora[0]),
            self.ctx,
        )
        # Unconditioned forward
        var pred_uncond = _chroma_forward_turbo[N_IMG_](
            self.shared[0][], self.loader[0][], self.latent[0][],
            self.t5_uncond[0][], t_curr, self.rope_cos[0][], self.rope_sin[0][],
            self.uncond_real_len,
            Optional[ArcPointer[ChromaLoraOverlay]](self.lora[0]),
            self.ctx,
        )

        # CFG: pred = uncond + guidance * (cond - uncond)
        var neg_uncond = mul_scalar(pred_uncond, Float32(-1.0), self.ctx)
        var diff = add(pred_cond, neg_uncond, self.ctx)
        var scaled_diff = mul_scalar(diff, self.guidance, self.ctx)
        var pred = add(pred_uncond, scaled_diff, self.ctx)

        var x_new: Tensor
        if self.executed_sampler == "dpmpp_2m":
            var latent_f32 = cast_tensor(self.latent[0][], STDtype.F32, self.ctx)
            var pred_f32 = cast_tensor(pred, STDtype.F32, self.ctx)
            var denoised = denoised_from_velocity(
                latent_f32, pred_f32, t_curr, self.ctx
            )
            if not self.dpmpp_history.is_empty():
                self.dpmpp_second_order_steps += 1
            var stepped = dpmpp_2m_step(
                latent_f32,
                denoised,
                t_curr,
                t_next,
                self.dpmpp_history,
                self.ctx,
            )
            self.dpmpp_history.push(
                denoised^,
                lambda_from_sigma_f64(Float64(t_curr)),
            )
            self.dpmpp_update_steps += 1
            x_new = cast_tensor(stepped, STDtype.BF16, self.ctx)
        else:
            # Euler step: x = x + dt * pred
            var step_delta = mul_scalar(pred, dt, self.ctx)
            x_new = add(self.latent[0][], step_delta, self.ctx)
        self.latent = List[ArcPointer[Tensor]]()
        self.latent.append(ArcPointer(x_new^))

    def _denoise_one(mut self) raises:
        comptime for bi in range(DEFAULT_ASPECT_LADDER_LEN):
            comptime X100_BI = DEFAULT_ASPECT_LADDER_X100[bi]
            comptime LH_BI = aspect_lat_h_units(X100_BI, CHROMA_PRODUCT_EDGE_UNITS)
            comptime LW_BI = aspect_lat_w_units(X100_BI, CHROMA_PRODUCT_EDGE_UNITS)
            comptime N_IMG_BI = (LH_BI // 2) * (LW_BI // 2)
            if self.params.width == LW_BI * 8 and self.params.height == LH_BI * 8:
                self._denoise_one_shape[N_IMG_BI]()
                return
        raise Error("chroma: admitted denoise shape was not compiled")

    # ── final decode + PNG(tEXt) ──────────────────────────────────────────────
    def _decode_and_save_shape[LH_: Int, LW_: Int](mut self) raises -> String:
        var png_path = self.params.out_dir + "/" + self.params.job_id + ".png"
        # Keep only a tiny packed-latent clone. The complete pinned-host FP8
        # store stays warm across jobs. Adaptive device promotions are backed
        # by explicit CUDA VMM, so this phase boundary can unmap/release their
        # physical bytes before the quality-first whole-image decoder and map
        # them again from pinned host memory for the next denoise.
        var packed = self.latent[0][].clone(self.ctx)
        self.t5_cond = List[ArcPointer[Tensor]]()
        self.t5_uncond = List[ArcPointer[Tensor]]()
        self.sched = List[Float32]()
        self.latent = List[ArcPointer[Tensor]]()
        self.dpmpp_history_final_len = self.dpmpp_history.len()
        self.dpmpp_history = MultistepHistory(1)
        self.ctx.synchronize()
        if len(self.loader) == 1:
            self.loader[0][].discard_fp8h_device_staging()
            self.loader[0][].discard_fp8_host_promotions()
            print("[chroma] released phase-local VMM denoiser promotions before VAE")
        cu_mempool_trim_current(0)
        self.ctx.synchronize()
        _print_vram("after transient Chroma staging release before VAE")
        # Pipeline Stage 8 (unpack BF16 → F32 cast → FLUX VAE decode), tried in
        # this order:
        #   1) WHOLE-image decode in a CHILD PROCESS (chroma_decode_subprocess,
        #      byte-identical Stage-8 math). MEASURED job-0078: whole decode
        #      allocates ~11.7 GB on the parent's ~3 GB floor → parent OOM;
        #      job-0079: the child itself lost a coin-flip margin (status 256,
        #      fresh-context overhead + decode peak vs ~13.28 GB free). The
        #      attempt stays first — it wins whenever the parent floor is lower.
        #   2) CHILD-PROCESS 3x3 TILED decode (same gated flux_tiled_decode math
        #      as the former fallback). Process exit guarantees its VAE memory
        #      is reclaimed while the resident denoiser survives for job N+1.
        #   3) IN-PROCESS tiled decode only as the final resilience fallback.
        var rgb: Tensor
        try:
            rgb = decode_whole_subprocess(packed, LH_, LW_, self.ctx)
            self.vae_decode_grid = String("whole_child_process")
            _print_vram("after decode child reaped (rgb loaded)")
        except e:
            print("[chroma] decode child unavailable (", e,
                  ") → tiled VAE child")
            try:
                rgb = decode_tiled_subprocess(packed, LH_, LW_, self.ctx)
                self.vae_decode_grid = String("3x3_tiled_child_process")
                _print_vram("after tiled decode child reaped (rgb loaded)")
            except tiled_e:
                print("[chroma] tiled decode child unavailable (", tiled_e,
                      ") → emergency in-process flux_tiled_decode")
                var latent = _unpack_latent_shape[LH_, LW_](packed, self.ctx)
                var latent_f32 = cast_tensor(latent, STDtype.F32, self.ctx)
                rgb = flux_tiled_decode[LH_, LW_](
                    latent_f32, String(VAE_PATH), self.ctx
                )
                self.vae_decode_grid = String("3x3_tiled_inprocess_emergency")
                _print_vram("after emergency in-process tiled VAE decode")
        _save_rgb_png_with_text(rgb, png_path, self.params.params_json, self.ctx)
        return png_path

    def _decode_and_save(mut self) raises -> String:
        comptime for bi in range(DEFAULT_ASPECT_LADDER_LEN):
            comptime X100_BI = DEFAULT_ASPECT_LADDER_X100[bi]
            comptime LH_BI = aspect_lat_h_units(X100_BI, CHROMA_PRODUCT_EDGE_UNITS)
            comptime LW_BI = aspect_lat_w_units(X100_BI, CHROMA_PRODUCT_EDGE_UNITS)
            if self.params.width == LW_BI * 8 and self.params.height == LH_BI * 8:
                return self._decode_and_save_shape[LH_BI, LW_BI]()
        raise Error("chroma: admitted decode shape was not compiled")

    def _clear_job(mut self):
        # A LoRA overlay is request-specific; discard it after the job. The base
        # model remains warm for the normal repeated-generation path.
        if len(self.params.loras) > 0:
            self._free_dit()
        self.active = False
        self.phase = CPHASE_IDLE
        self.cur = 0
        self.cancel_flag = False
        self.announced = False
        self.t5_cond = List[ArcPointer[Tensor]]()
        self.t5_uncond = List[ArcPointer[Tensor]]()
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
            if self.phase == CPHASE_ENCODE:
                if not self.announced:
                    # announce BEFORE the long blocking encode tick (per-job
                    # T5-XXL load + two forwards).
                    self.announced = True
                    r.step = 0
                    r.phase = String("encoding")
                    return r^
                var encode_t0 = perf_counter_ns()
                self._encode()
                self.text_encode_seconds = Float64(perf_counter_ns() - encode_t0) / 1.0e9
                self._record_vram()
                # The encode ran in a CHILD process (its VRAM died with it);
                # this parent only gained the two ~4 MiB caps. Trim is a cheap
                # no-op safety net (also covers the in-process fallback path).
                self.ctx.synchronize()
                cu_mempool_trim_current(0)
                self.ctx.synchronize()
                _print_vram("after text encode (child reaped; parent pool trimmed)")
                self.announced = False
                self.phase = CPHASE_LOAD
                r.step = 0
                return r^
            if self.phase == CPHASE_LOAD:
                var load_t0 = perf_counter_ns()
                if not self.loaded:
                    if not self.announced:
                        self.announced = True
                        r.step = 0
                        r.phase = String("loading")
                        return r^
                    self._load_model()
                self._ensure_denoise_residency()
                self.load_seconds += Float64(perf_counter_ns() - load_t0) / 1.0e9
                self.announced = False
                var prep_t0 = perf_counter_ns()
                self._prepare_job()
                self.prepare_seconds += Float64(perf_counter_ns() - prep_t0) / 1.0e9
                self._record_vram()
                self.phase = CPHASE_DENOISE
                r.step = 0
                r.phase = String("sampling")
                return r^
            if self.phase == CPHASE_DENOISE:
                var denoise_t0 = perf_counter_ns()
                self._denoise_one()
                self.denoise_seconds += Float64(perf_counter_ns() - denoise_t0) / 1.0e9
                self._record_vram()
                self.cur += 1
                r.step = self.cur
                r.phase = String("sampling")
                if self.cur >= self.runtime_steps:
                    self.phase = CPHASE_DECODE
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
            print("[chroma][manifest] saved:", manifest)
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
