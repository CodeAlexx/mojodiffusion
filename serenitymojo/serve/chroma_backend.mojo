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
#   Chroma DiT (19 double + 38 single blocks, BLOCK-STREAMED via BlockLoader;
#   guidance via the distilled_guidance_layer approximator, NOT guidance_in) →
#   real CFG: pred = uncond + cfg * (cond - uncond) + flow-match Euler update →
#   FLUX VAE WHOLE-image decode (the pipeline's proven decode; ae.safetensors)
#   → PNG SIGNED (genparams tEXt).
#
# Unlike FLUX.1-dev (guidance-distilled, negative discarded), Chroma runs REAL
# CFG: the negative prompt IS encoded and drives the uncond forward; params.cfg
# is the CFG multiplier (default 4.0 — the verified pipeline's GUIDANCE).
#
# Residency model (16 GB GPU, Chroma1-HD ~17.8 GB BF16 on disk — must stream):
#   * The Chroma DiT is NEVER fully resident: ChromaShared (approximator +
#     x/context embedders + proj_out, small) + a TurboBlockLoader that streams
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
#   → LOAD (ChromaShared + BlockLoader + rope, announced phase="loading")
#   → DENOISE×steps (one CFG step per tick: cond forward + uncond forward +
#   CFG combine + Euler update) → DECODE (announced phase="decoding") → done.
#   cancel() makes the next step() return cancelled and frees per-job tensors.
#
# The finite seven-shape ~1 MP core is compiled: each arm specializes latent
# geometry, image-token count, joint-attention length, rectangular RoPE, and
# VAE decode. Product admission follows this exact finite ladder. Rectangular
# denoise remains slower than square on the 16GB product GPU.
#
# LoRA: NOT supported (Chroma has no LoRA hook in the Mojo stack yet) — fail
# loud. img2img: NOT supported — fail loud.

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
from serenitymojo.offload.turbo_loader import TurboBlockLoader

from serenitymojo.models.dit.flux1_dit import build_flux1_rope_tables
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.elementwise import modulate
from serenitymojo.serve.chroma_encode_subprocess import encode_captions_subprocess
from serenitymojo.serve.chroma_decode_subprocess import decode_whole_subprocess
from serenitymojo.pipeline.flux_tiled_decode import flux_tiled_decode
from serenitymojo.ops.random import randn
from serenitymojo.ops.tensor_algebra import add, concat, mul_scalar, slice
from serenitymojo.sampling.flux1_dev import build_flux1_sigma_schedule
from serenitymojo.sampling.sampler_registry import (
    normalize_sampler_name, normalize_scheduler_name,
)
from serenitymojo.sampling.variation_noise import swarm_variation_noise_chw
from serenitymojo.training.aspect_buckets import (
    DEFAULT_ASPECT_LADDER_LEN, DEFAULT_ASPECT_LADDER_X100,
    aspect_lat_h_units, aspect_lat_w_units,
)

# The PROVEN Chroma pipeline math — imported, never re-derived. Its helpers run
# the approximator + 19 double + 38 single blocks (BlockLoader-streamed) plus
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

comptime GENPARAMS_TEXT_KEY = "serenity.genparams.v1"

# Verified pipeline defaults (chroma_pipeline_1024_multistep NUM_STEPS/GUIDANCE).
comptime DEFAULT_STEPS = 30
comptime DEFAULT_GUIDANCE = Float32(4.0)


comptime CPHASE_IDLE = 0
comptime CPHASE_ENCODE = 1
comptime CPHASE_LOAD = 2
comptime CPHASE_DENOISE = 3
comptime CPHASE_DECODE = 4
comptime CHROMA_PRODUCT_EDGE_UNITS = 16


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
# tail. ONLY the weight staging changes: the naive synchronous BlockLoader
# (madvise prefetch + blocking H2D per block — GPU idle during every transfer;
# measured 4.78 s/step) is replaced by TurboBlockLoader's async double-buffered
# rotation: block i+1 is DMA-staged from the pinned host store on the explicit
# copy stream WHILE block i computes on the default stream (await_block fences
# via DeviceEvent; mark_active_slot_compute_done lets the copy stream reuse a
# slot only after the block's kernels are queued).
def _chroma_forward_turbo[N_IMG_: Int](
    shared: ChromaShared,
    mut loader: TurboBlockLoader,
    img_packed: Tensor,
    txt: Tensor,
    timestep: Float32,
    rope_cos: Tensor,
    rope_sin: Tensor,
    real_txt_len: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    # Stage block 0 first: its DMA overlaps the approximator + embedder compute.
    loader.prefetch(String("transformer_blocks.0"), ctx)

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
        var prefix = String("transformer_blocks.") + String(i)
        var block = loader.await_block(prefix, ctx)
        if i + 1 < N_DBL:
            loader.prefetch(String("transformer_blocks.") + String(i + 1), ctx)
        else:
            loader.prefetch(String("single_transformer_blocks.0"), ctx)

        # Slice mod params: img [114+6i : 114+6i+6], txt [228+6i : 228+6i+6]
        var img_mod_s = slice(pooled_temb, 1, MOD_DBL_IMG_OFF + 6 * i, 6, ctx)
        var txt_mod_s = slice(pooled_temb, 1, MOD_DBL_TXT_OFF + 6 * i, 6, ctx)

        var res = _double_block_shape[N_IMG_](
            i, img, x_txt, img_mod_s, txt_mod_s, rope_cos, rope_sin,
            real_txt_len, block, ctx,
        )
        img = _clone(res[0], ctx)
        x_txt = _clone(res[1], ctx)
        loader.mark_active_slot_compute_done(ctx)

    # 4. Concat [txt | img] -> [1, S, HIDDEN] for single blocks
    var x = concat(1, ctx, x_txt, img)

    # 5. Single blocks (38) — same await → prefetch-next → compute rotation.
    for i in range(N_SGL):
        var prefix = String("single_transformer_blocks.") + String(i)
        var block = loader.await_block(prefix, ctx)
        if i + 1 < N_SGL:
            loader.prefetch(String("single_transformer_blocks.") + String(i + 1), ctx)

        # Slice mod params: [3*i : 3*i+3]
        var sgl_mod_s = slice(pooled_temb, 1, MOD_SGL_OFF + 3 * i, 3, ctx)

        x = _single_block_shape[N_IMG_](
            i, x, sgl_mod_s, rope_cos, rope_sin, real_txt_len, block, ctx,
        )
        loader.mark_active_slot_compute_done(ctx)

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

    # ── DiT handles (loaded in the LOAD phase, freed before VAE decode) ──
    # ArcPointer wrappers: ChromaShared / BlockLoader / Tensor are
    # Movable-not-Copyable and List[T] requires T: Copyable — Arc is Copyable
    # (refcount), so List[Arc[..]] holds the 0/1.
    var loaded: Bool
    var shared: List[ArcPointer[ChromaShared]]   # 0/1 (approximator+embedders)
    var loader: List[ArcPointer[TurboBlockLoader]]  # 0/1 (async block-stream handle)
    var rope_cos: List[ArcPointer[Tensor]]       # 0/1
    var rope_sin: List[ArcPointer[Tensor]]       # 0/1

    # ── per-job state (cleared on done/failed/cancelled) ──
    var active: Bool
    var cancel_flag: Bool
    var phase: Int
    var announced: Bool
    var cur: Int
    var params: JobParams
    var guidance: Float32
    var t5_cond: List[ArcPointer[Tensor]]        # 0/1 [1,512,4096] BF16
    var t5_uncond: List[ArcPointer[Tensor]]      # 0/1 [1,512,4096] BF16
    var cond_real_len: Int                       # unpadded T5 token count (MJ-1048 mask)
    var uncond_real_len: Int
    var sched: List[Float32]                     # flow-match sigma table (steps+1)
    var latent: List[ArcPointer[Tensor]]         # 0/1 (packed [1,N_IMG,64] BF16)
    var vae_decode_grid: String                  # executed decode path (manifest)
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
        self.shared = List[ArcPointer[ChromaShared]]()
        self.loader = List[ArcPointer[TurboBlockLoader]]()
        self.rope_cos = List[ArcPointer[Tensor]]()
        self.rope_sin = List[ArcPointer[Tensor]]()
        self.active = False
        self.cancel_flag = False
        self.phase = CPHASE_IDLE
        self.announced = False
        self.cur = 0
        self.params = JobParams()
        self.guidance = DEFAULT_GUIDANCE
        self.t5_cond = List[ArcPointer[Tensor]]()
        self.t5_uncond = List[ArcPointer[Tensor]]()
        self.cond_real_len = N_TXT
        self.uncond_real_len = N_TXT
        self.sched = List[Float32]()
        self.latent = List[ArcPointer[Tensor]]()
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
        reject_unsupported_common_runtime_params(params, String("chroma"))
        reject_unsupported_reference_image_params(params, String("chroma"))
        reject_unsupported_inpaint_conditioning_params(params, String("chroma"))
        reject_unsupported_qwen_edit_conditioning_params(params, String("chroma"))
        reject_unsupported_conditioning_mask_params(params, String("chroma"))
        reject_unsupported_mask_image_params(params, String("chroma"))
        reject_unsupported_lanpaint_params(params, String("chroma"))
        # Local sampler/scheduler admission: the sampler registry has no chroma
        # arm yet (its unknown-backend fallback ACCEPTS everything, which is not
        # an admission gate) — so gate here on the one executed pair.
        var norm_sampler = normalize_sampler_name(params.sampler)
        if norm_sampler == String(""):
            norm_sampler = String("euler")
        if not (norm_sampler == String("euler") or norm_sampler == String("flowmatch_euler")):
            raise Error(
                String("chroma: unsupported sampler '") + params.sampler
                + String("'; only euler/flowmatch_euler is served (the verified"
                         " Chroma CFG flow-match Euler path)")
            )
        var norm_scheduler = normalize_scheduler_name(params.scheduler)
        if norm_scheduler == String(""):
            norm_scheduler = String("simple")
        if norm_scheduler != String("simple"):
            raise Error(
                String("chroma: unsupported scheduler '") + params.scheduler
                + String("'; only the simple flow-match schedule is served")
            )
        if not _chroma_shape_supported(params.width, params.height):
            raise Error(
                String("chroma: unsupported size ") + String(params.width)
                + "x" + String(params.height)
                + " — only 1024x1024 is product-admitted; rectangular Chroma"
                + " core arms remain gated after measured ~24x denoise slowdown"
            )
        if len(params.loras) > 0:
            raise Error(
                "chroma: LoRA is not supported for Chroma1-HD yet;"
                " submit without LoRAs"
            )
        if params.init_image.byte_length() > 0:
            raise Error(
                "chroma: img2img is not supported for Chroma1-HD yet;"
                " submit without an init image"
            )
        # Warn-loud (never silently drop) on any advanced-sampling knob set but
        # unsupported by this fixed CFG flow-match Euler path.
        warn_unsupported_advanced_sampling_params(params, String("chroma"), List[String]())
        self.params = params.copy()
        # Honor steps at runtime; <=0 means unset/invalid -> the verified
        # pipeline default 30 (chroma_pipeline_1024_multistep NUM_STEPS).
        if self.params.steps <= 0:
            self.params.steps = DEFAULT_STEPS
        # Chroma runs REAL CFG: params.cfg is the CFG multiplier
        # (pred = uncond + cfg*(cond-uncond)). cfg<=0 means unset/invalid ->
        # the verified pipeline default 4.0 (GUIDANCE).
        self.guidance = Float32(self.params.cfg) if self.params.cfg > 0.0 else DEFAULT_GUIDANCE
        self.active = True
        self.cancel_flag = False
        self.cur = 0
        self.announced = False
        self.phase = CPHASE_ENCODE
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
        """Reclaim the per-job transient peak (T5-XXL encoder ~9.5 GB, streamed
        DiT blocks, the VAE decoder, 1024² forward + decode activations) back to
        the OS via cuMemPoolTrimTo. Nothing stays resident between chroma jobs
        (the DiT handles are freed before decode), so this reclaims it all."""
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
        if self.params.steps > 0:
            denoise_per_step = self.denoise_seconds / Float64(self.params.steps)
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
        content += String('    "cfg":') + String(self.guidance) + String(",\n")
        content += String('    "sampler_registry_backend":"chroma",\n')
        content += String('    "requested_sampler":"') + json_escape(self.params.sampler) + String('",\n')
        content += String('    "requested_scheduler":"') + json_escape(self.params.scheduler) + String('",\n')
        content += String('    "executed_sampler":"chroma_cfg_flowmatch_euler",\n')
        content += String('    "executed_scheduler":"flux_simple_flowmatch",\n')
        content += String('    "schedule_source":"flux1_dev_flowmatch",\n')
        content += String('    "variation_seed":') + String(self.params.variation_seed) + String(",\n")
        content += String('    "variation_strength":') + String(self.params.variation_strength) + String(",\n")
        content += String('    "variation_applied":') + json_bool(self.params.variation_strength > 0.0) + String(",\n")
        content += String('    "released_streamed_dit_before_unpack":true,\n')
        content += String('    "image_index":') + String(self.params.image_index) + String(",\n")
        content += String('    "image_count":') + String(self.params.image_count) + String(",\n")
        content += String('    "lora_count":') + String(len(self.params.loras)) + String(",\n")
        content += String('    "cond_real_len":') + String(self.cond_real_len) + String(",\n")
        content += String('    "uncond_real_len":') + String(self.uncond_real_len) + String(",\n")
        content += String('    "vae_decode_tile_grid":"') + json_escape(self.vae_decode_grid) + String('",\n')
        content += String('    "dtype":"bf16_dit_bf16_latent"\n')
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
        content += String('  "note":"Rust-server Mojo worker product-path result; Chroma1-HD uses live per-job T5-XXL encode, block-streamed DiT with distilled-guidance approximator, and real CFG. Speed parity remains unaccepted until paired baseline evidence exists."\n')
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

    def _free_dit(mut self):
        """Drop the ChromaShared weights + BlockLoader handle + rope tables and
        reset the residency flag (so the next LOAD reloads)."""
        if self.loaded:
            print("[chroma] releasing Chroma shared weights + turbo loader + rope before VAE decode")
        self.shared = List[ArcPointer[ChromaShared]]()
        self.loader = List[ArcPointer[TurboBlockLoader]]()
        self.rope_cos = List[ArcPointer[Tensor]]()
        self.rope_sin = List[ArcPointer[Tensor]]()
        self.loaded = False

    def _load_model_shape[LH_: Int, LW_: Int, N_IMG_: Int](mut self) raises:
        """Load the Chroma shared weights (approximator + embedders + proj_out),
        open the BlockLoader stream handle, and build the rope tables — exactly
        the pipeline's Stage 2/3/4. The DiT blocks themselves are NEVER fully
        resident: _chroma_forward streams them one at a time (16 GB card)."""
        if self.loaded:
            return
        _print_vram("before Chroma shared-weight load")
        print("[chroma] loading Chroma1-HD shared weights from", CHROMA_CKPT)
        self.shared = List[ArcPointer[ChromaShared]]()
        self.shared.append(ArcPointer(ChromaShared.load(String(CHROMA_CKPT), self.ctx)))
        # TurboBlockLoader: async double-buffered block streaming (pinned host
        # store populated once here; per-block prefetch is then pure async DMA
        # on the copy stream, overlapped with compute — replaces the naive
        # synchronous BlockLoader that idled the GPU on every transfer).
        self.loader = List[ArcPointer[TurboBlockLoader]]()
        # fill_block_store=False: bounded 2-slot pinned staging (the ~17.8 GiB
        # persistent host store is the qwen-freeze class; see qwenimage_dit.mojo).
        self.loader.append(ArcPointer(TurboBlockLoader.open_with_copy_mode(
            String(CHROMA_CKPT), self.ctx, False, fill_block_store=False)))
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

    def _prepare_job_shape[LH_: Int, LW_: Int, N_IMG_: Int](mut self) raises:
        """Flow-match sigma table (honors steps) + seeded initial packed latent
        (honors seed). Mirrors the pipeline's Stage 5/6 (BF16 noise → pack)."""
        self.sched = build_flux1_sigma_schedule(self.params.steps, N_IMG_)
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
            var blended = swarm_variation_noise_chw(
                base_h, var_h, LC, LH_, LW_,
                self.params.variation_strength,
            )
            noise = Tensor.from_host(blended, nsh.copy(), STDtype.BF16, self.ctx)
        var packed = _pack_latent_shape[LH_, LW_](noise, self.ctx)
        self.latent = List[ArcPointer[Tensor]]()
        self.latent.append(ArcPointer(packed^))
        print(
            "[chroma] job", self.params.job_id, ":", self.params.steps,
            "steps, cfg", self.guidance, "seed", self.params.seed,
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

        # Conditioned forward (turbo-streamed; math identical to _chroma_forward)
        var pred_cond = _chroma_forward_turbo[N_IMG_](
            self.shared[0][], self.loader[0][], self.latent[0][],
            self.t5_cond[0][], t_curr, self.rope_cos[0][], self.rope_sin[0][],
            self.cond_real_len, self.ctx,
        )
        # Unconditioned forward
        var pred_uncond = _chroma_forward_turbo[N_IMG_](
            self.shared[0][], self.loader[0][], self.latent[0][],
            self.t5_uncond[0][], t_curr, self.rope_cos[0][], self.rope_sin[0][],
            self.uncond_real_len, self.ctx,
        )

        # CFG: pred = uncond + guidance * (cond - uncond)
        var neg_uncond = mul_scalar(pred_uncond, Float32(-1.0), self.ctx)
        var diff = add(pred_cond, neg_uncond, self.ctx)
        var scaled_diff = mul_scalar(diff, self.guidance, self.ctx)
        var pred = add(pred_uncond, scaled_diff, self.ctx)

        # Euler step: x = x + dt * pred
        var step_delta = mul_scalar(pred, dt, self.ctx)
        var x_new = add(self.latent[0][], step_delta, self.ctx)
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
        # Keep only a tiny packed-latent clone, then release the shared weights
        # + block loader before unpack + VAE decode (the pipeline's Stage 8 drop;
        # mandatory on a 16 GB card).
        var packed = self.latent[0][].clone(self.ctx)
        self.t5_cond = List[ArcPointer[Tensor]]()
        self.t5_uncond = List[ArcPointer[Tensor]]()
        self.sched = List[Float32]()
        self.latent = List[ArcPointer[Tensor]]()
        self._free_dit()
        self.ctx.synchronize()
        cu_mempool_trim_current(0)
        self.ctx.synchronize()
        _print_vram("after DiT release before VAE")
        # Pipeline Stage 8 (unpack BF16 → F32 cast → FLUX VAE decode), tried in
        # this order:
        #   1) WHOLE-image decode in a CHILD PROCESS (chroma_decode_subprocess,
        #      byte-identical Stage-8 math). MEASURED job-0078: whole decode
        #      allocates ~11.7 GB on the parent's ~3 GB floor → parent OOM;
        #      job-0079: the child itself lost a coin-flip margin (status 256,
        #      fresh-context overhead + decode peak vs ~13.28 GB free). The
        #      attempt stays first — it wins whenever the parent floor is lower.
        #   2) IN-PROCESS 3x3 TILED decode (flux_tiled_decode — authored for
        #      exactly this post-offloaded-DiT high-water state, same flux
        #      ae.safetensors VAE). Klein ships tiled VAE decode in production
        #      on this 16 GB card, so tiled here is consistent with shipped
        #      quality (MJ-1054 was a different arm). Printed LOUD.
        var rgb: Tensor
        try:
            rgb = decode_whole_subprocess(packed, LH_, LW_, self.ctx)
            self.vae_decode_grid = String("whole_child_process")
            _print_vram("after decode child reaped (rgb loaded)")
        except e:
            print("[chroma] decode child unavailable (", e,
                  ") → tiled VAE decode (flux_tiled_decode)")
            var latent = _unpack_latent_shape[LH_, LW_](packed, self.ctx)
            var latent_f32 = cast_tensor(latent, STDtype.F32, self.ctx)
            rgb = flux_tiled_decode[LH_, LW_](latent_f32, String(VAE_PATH), self.ctx)
            self.vae_decode_grid = String("3x3_tiled_fallback")
            _print_vram("after tiled VAE decode (3x3 flux_tiled_decode)")
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
        self._free_dit()
        self.active = False
        self.phase = CPHASE_IDLE
        self.cur = 0
        self.cancel_flag = False
        self.announced = False
        self.t5_cond = List[ArcPointer[Tensor]]()
        self.t5_uncond = List[ArcPointer[Tensor]]()
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
                self.load_seconds += Float64(perf_counter_ns() - load_t0) / 1.0e9
                self.announced = False
                var prep_t0 = perf_counter_ns()
                self._prepare_job()
                self.prepare_seconds += Float64(perf_counter_ns() - prep_t0) / 1.0e9
                self._record_vram()
                self.phase = CPHASE_DENOISE
                r.step = 0
                return r^
            if self.phase == CPHASE_DENOISE:
                var denoise_t0 = perf_counter_ns()
                self._denoise_one()
                self.denoise_seconds += Float64(perf_counter_ns() - denoise_t0) / 1.0e9
                self._record_vram()
                self.cur += 1
                r.step = self.cur
                if self.cur >= self.params.steps:
                    self.phase = CPHASE_DECODE
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
            print("[chroma][manifest] saved:", manifest)
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
