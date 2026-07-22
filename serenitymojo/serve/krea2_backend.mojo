# serenitymojo.serve.krea2_backend — GenBackend for Krea-2 (krea2) txt2img.
#
# Wraps the VERIFIED krea2 inference sampler stack (models/krea2/krea2_infer.mojo,
# hoisted from the trainer's proven inline sampler) behind the daemon/worker
# GenBackend seam, exactly like ZImageBackend. One resident worker per GPU, spawned
# lazily by serenity-server as `serenity_worker_krea2 <fd>`.
#
# V1 VRAM design (single 24 GB card): the Qwen3-VL-4B TE (~22 GB measured peak)
# and the DiT do NOT coexist, so a cache-miss job runs the proven sequential path:
#   1) encode POS/NEG contexts in the worker's reusable MAX pool and write .bins;
#      the TE tensors drop before the DiT is loaded.
#   2) build the fp8-resident base (~12 GB) + conditioning weights + final layer.
#   3) fixed-LTMAX length-bucket denoise (real_len flash-padmask) + LoRA overlay + CFG.
#   4) VAE decode → <out_dir>/<job_id>.png.
# resident_model() is "" because the server still owns model-family routing. The
# backend keeps keyed last-request FlowEdit/LanPaint contexts and normalized
# source latents, plus the matching loaded int8 DiT while authored inputs remain
# unchanged. A prompt/source cache miss drops the DiT before loading the TE/VAE
# because those stages do not fit beside it on the product GPU. This gives
# Regenerate the unchanged-node reuse users expect without weakening the safe
# cross-model eviction policy.
#
# step() runs the whole job in ONE blocking tick (the driver has no per-step
# watchdog — only a 15 s Ready handshake, sent at worker-loop start before any load).
# cancel() is honored between the major phases.

from std.gpu.host import DeviceContext
from std.collections import Optional
from std.time import perf_counter_ns

from serenitymojo.serve.backend import (
    GenBackend, JobParams, StepResult, reject_unsupported_common_runtime_params,
    has_lanpaint_runtime_params, has_lanpaint_sampler_runtime_params,
)
from serenitymojo.serve.proc_ipc import write_msg
from serenitymojo.serve.model_scan import LORAS_DIR
from serenitymojo.io.env import env_int
from serenitymojo.io.ffi import sys_open, sys_close, O_RDONLY
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.offload.vmm_cuda import cu_mempool_trim_current

from serenitymojo.serve.krea2_encode_subprocess import (
    krea2_encode_contexts_subprocess,
)
from serenitymojo.models.krea2.krea2_infer import (
    Krea2ResidentCond, load_krea2_resident_cond,
    Krea2InlineCond, inline_cond_from_bin,
    load_krea2_stack_lora, empty_krea2_stack_lora,
    krea2_sample_latent, krea2_sample_lanpaint_latent,
    krea2_decode_latent_to_png,
)
from serenitymojo.models.krea2.krea2_stack import (
    Krea2StackLora, Krea2StreamFinal,
    Krea2ResidentFp8, build_krea2_resident_fp8,
)
from serenitymojo.models.krea2.krea2_buckets import (
    KREA2_LADDER_LEN, KREA2_LADDER_X100, krea2_lat_h, krea2_lat_w,
)
from serenitymojo.pipeline.krea2_paths import (
    KREA2_RAW, KREA2_TURBO, KREA2_VAE_DIR, KREA2_RAW_KEY_PREFIX,
)

# ── krea2 FlowEdit worker mode (Phase C1 part 2): reuse the VERIFIED pipeline
#    helpers (pipeline/krea2_flowedit.mojo, the 512² 18s/edit vertical) behind
#    the GenBackend seam. edit_src_image != "" routes step() to _step_flowedit.
from serenitymojo.pipeline.krea2_flowedit import (
    HEIGHT as FE_HEIGHT, WIDTH as FE_WIDTH, LH as FE_LH, LW as FE_LW,
    NTOK as FE_NTOK, NBLOCKS as FE_NBLOCKS, LT_SHARED as FE_LT_SHARED,
    _build_pos_shape, _load_context_padded_shape, _velocity_shape,
    _accum_saliency_shape, _mask_from_saliency_shape,
    _blend_outside_mask_shape, _save_mask_png_shape,
)
from serenitymojo.models.dit.krea2_dit import (
    Krea2ResidentInt8, Krea2HostInt8Inf, Krea2SharedResident,
    build_krea2_shared_resident, build_krea2_host_int8_inf,
)
from serenitymojo.models.krea2.krea2_stack import build_krea2_resident_int8
from serenitymojo.models.krea2.krea2_int8_cache import (
    krea2_int8_cache_path, krea2_int8_cache_valid,
    load_krea2_int8_cache_resident, load_krea2_int8_cache_host,
    load_krea2_int8_cache_shared, save_krea2_int8_cache,
)
from serenitymojo.models.krea2.krea2_prepare_cache import (
    _mean_ch, _std_ch, _normalize_latent, KREA2_VAE_ENC_FILE,
)
from serenitymojo.models.vae.qwenimage_encoder import QwenImageVaeEncoder
from serenitymojo.models.vae.qwenimage_decoder import QwenImageVaeDecoder
from serenitymojo.sampling.krea2_sampler import krea2_timesteps, krea2_packed_seq_len
from serenitymojo.serve.image_io import (
    decode_image_any, image_to_signed_nchw,
    image_area_resize_to_signed_nchw,
    load_lanpaint_latent_preserve_mask, load_lanpaint_pixel_blend_mask,
    apply_lanpaint_mask_blend_signed_chw,
)
from image.transform import resize_bilinear, resize_nearest
from serenitymojo.ops.tensor_algebra import mul_scalar, add, sub
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.torch_bf16 import torch_f32_to_bf16_rne
from serenitymojo.ops.random import randn
from serenitymojo.io.dtype import STDtype
from serenitymojo.image.png import save_png, ValueRange
from serenitymojo.tensor import Tensor

# ── compiled config: 1024px-area, seven-arm trainer-parity aspect ladder. ─────
# The trainer/cache producer already owns and gates this exact SimpleTuner ladder.
# Reuse it here so inference, training, and staged cache geometry cannot drift.
comptime KREA2_EDGE_UNITS = 16
# Square uses the proven 768 text bucket. Non-square ladder arms carry 4032 or
# 4160 image tokens, so their text bucket is padded by the minimum extra tail
# needed to keep the joint flash-attention buffer 128-aligned. Real text length
# is unchanged and the existing padmask excludes every added tail token.
comptime KREA2_LTMAX_BASE = 768
comptime KREA2_NBLOCKS = 28


def _emit_flowedit_progress(fd: Int32, step: Int, total: Int):
    if fd < 0:
        return
    try:
        write_msg(
            fd,
            String("{\"ev\":\"progress\",\"step\":")
            + String(step)
            + String(",\"total\":")
            + String(total)
            + String(",\"phase\":\"sampling\",\"preview\":\"\"}"),
        )
    except e:
        print("[krea2-edit] progress IPC skipped:", String(e))


def _krea2_t2i_shape_supported(width: Int, height: Int) -> Bool:
    comptime for bi in range(KREA2_LADDER_LEN):
        comptime X100_BI = KREA2_LADDER_X100[bi]
        comptime LH_BI = krea2_lat_h(X100_BI, KREA2_EDGE_UNITS)
        comptime LW_BI = krea2_lat_w(X100_BI, KREA2_EDGE_UNITS)
        if width == LW_BI * 8 and height == LH_BI * 8:
            return True
    return False


struct Krea2LanPaintSource(Movable):
    var latent: Tensor
    var signed_chw: List[Float32]

    def __init__(
        out self, var latent: Tensor, var signed_chw: List[Float32]
    ):
        self.latent = latent^
        self.signed_chw = signed_chw^


def _encode_krea2_lanpaint_source[HEIGHT_: Int, WIDTH_: Int](
    path: String, ctx: DeviceContext
) raises -> Krea2LanPaintSource:
    """Encode one source image, then release the VAE encoder on return."""
    var img = decode_image_any(path)
    # Official Krea2 LanPaint graph: nearest-exact feeds VAEEncode, while a
    # separate area resize supplies image1 to the final MaskBlend node.
    var resized = resize_nearest(img, WIDTH_, HEIGHT_)
    var vae_signed_chw = image_to_signed_nchw(resized)
    var signed_chw = image_area_resize_to_signed_nchw(img, WIDTH_, HEIGHT_)
    var image_t = Tensor.from_host(
        vae_signed_chw, [1, 3, HEIGHT_, WIDTH_], STDtype.F32, ctx
    )
    var img_bf16 = cast_tensor(image_t, STDtype.BF16, ctx)
    var enc = QwenImageVaeEncoder[HEIGHT_, WIDTH_].load(
        KREA2_VAE_ENC_FILE, ctx
    )
    var lat_mean = enc.encode_mean(img_bf16, ctx)
    var lat_f32 = cast_tensor(lat_mean, STDtype.F32, ctx)
    var mean_ch = _mean_ch(ctx)
    var std_ch = _std_ch(ctx)
    var latent = _normalize_latent(lat_f32, mean_ch, std_ch, ctx)
    ctx.synchronize()
    return Krea2LanPaintSource(latent^, signed_chw^)


def _path_exists(path: String) -> Bool:
    if path == String(""):
        return False
    var fd = sys_open(path, O_RDONLY, 0)
    if fd < 0:
        return False
    _ = sys_close(fd)
    return True


def _resolve_krea2_lora_path(name: String) raises -> String:
    # Bare names resolve against the scanner's LoRA dir; absolute/relative paths
    # (developer / imported use) are accepted as-is.
    if _path_exists(name):
        return name.copy()
    if _path_exists(name + ".safetensors"):
        return name + ".safetensors"
    if _path_exists(String(LORAS_DIR) + "/" + name):
        return String(LORAS_DIR) + "/" + name
    if _path_exists(String(LORAS_DIR) + "/" + name + ".safetensors"):
        return String(LORAS_DIR) + "/" + name + ".safetensors"
    raise Error(
        String("krea2: LoRA file not found: ") + name
        + " (tried as a path and under " + LORAS_DIR + ")"
    )


struct Krea2Backend(GenBackend, Movable):
    var ctx: DeviceContext
    var active: Bool
    var cancel_flag: Bool
    var params: JobParams
    var lora_path: String        # resolved file ("" = base / zero overlay)
    var lora_mult: Float32
    var progress_fd: Int32
    # Last-request FlowEdit caches. Context bins are keyed by all four authored
    # strings; the source latent is keyed by the server's unique upload path and
    # compiled geometry. These mirror Comfy's unchanged-node reuse without
    # retaining the 22 GiB text encoder or VAE on device.
    var edit_ctx_cache_valid: Bool
    var edit_ctx_cache_src_prompt: String
    var edit_ctx_cache_src_negative: String
    var edit_ctx_cache_tgt_prompt: String
    var edit_ctx_cache_tgt_negative: String
    var edit_ctx_cache_sp_bin: String
    var edit_ctx_cache_sn_bin: String
    var edit_ctx_cache_tp_bin: String
    var edit_ctx_cache_tn_bin: String
    var edit_latent_cache_valid: Bool
    var edit_latent_cache_source: String
    var edit_latent_cache_width: Int
    var edit_latent_cache_height: Int
    var edit_latent_cache: List[Float32]
    var edit_base_cache_valid: Bool
    var edit_base_cache_turbo: Bool
    var edit_base_cache_resident_blocks: Int
    var edit_base_resident_i8: Optional[Krea2ResidentInt8]
    var edit_base_host_i8: Optional[Krea2HostInt8Inf]
    var edit_base_shared: Optional[Krea2SharedResident]
    # Last-request LanPaint caches. These are separate from FlowEdit because
    # LanPaint uses a different source resize/latent contract and conditioning
    # carrier. Cache keys are authored inputs plus the selected base profile;
    # changed inputs release the resident DiT before TE/VAE work.
    var lanpaint_ctx_cache_valid: Bool
    var lanpaint_ctx_cache_prompt: String
    var lanpaint_ctx_cache_negative: String
    var lanpaint_ctx_cache_pos_bin: String
    var lanpaint_ctx_cache_neg_bin: String
    var lanpaint_source_cache_valid: Bool
    var lanpaint_source_cache_path: String
    var lanpaint_source_cache_latent: List[Float32]
    var lanpaint_source_cache_signed_chw: List[Float32]
    var lanpaint_base_cache_valid: Bool
    var lanpaint_base_cache_turbo: Bool
    var lanpaint_base_cache_resident_blocks: Int
    var lanpaint_base_cond: Optional[Krea2ResidentCond]
    var lanpaint_base_fin: Optional[Krea2StreamFinal]
    var lanpaint_base_resident_i8: Optional[Krea2ResidentInt8]
    var lanpaint_base_host_i8: Optional[Krea2HostInt8Inf]

    def __init__(out self) raises:
        self.ctx = DeviceContext()
        self.active = False
        self.cancel_flag = False
        self.params = JobParams()
        self.lora_path = String("")
        self.lora_mult = Float32(1.0)
        self.progress_fd = Int32(-1)
        self.edit_ctx_cache_valid = False
        self.edit_ctx_cache_src_prompt = String("")
        self.edit_ctx_cache_src_negative = String("")
        self.edit_ctx_cache_tgt_prompt = String("")
        self.edit_ctx_cache_tgt_negative = String("")
        self.edit_ctx_cache_sp_bin = String("")
        self.edit_ctx_cache_sn_bin = String("")
        self.edit_ctx_cache_tp_bin = String("")
        self.edit_ctx_cache_tn_bin = String("")
        self.edit_latent_cache_valid = False
        self.edit_latent_cache_source = String("")
        self.edit_latent_cache_width = 0
        self.edit_latent_cache_height = 0
        self.edit_latent_cache = List[Float32]()
        self.edit_base_cache_valid = False
        self.edit_base_cache_turbo = False
        self.edit_base_cache_resident_blocks = 0
        self.edit_base_resident_i8 = Optional[Krea2ResidentInt8](None)
        self.edit_base_host_i8 = Optional[Krea2HostInt8Inf](None)
        self.edit_base_shared = Optional[Krea2SharedResident](None)
        self.lanpaint_ctx_cache_valid = False
        self.lanpaint_ctx_cache_prompt = String("")
        self.lanpaint_ctx_cache_negative = String("")
        self.lanpaint_ctx_cache_pos_bin = String("")
        self.lanpaint_ctx_cache_neg_bin = String("")
        self.lanpaint_source_cache_valid = False
        self.lanpaint_source_cache_path = String("")
        self.lanpaint_source_cache_latent = List[Float32]()
        self.lanpaint_source_cache_signed_chw = List[Float32]()
        self.lanpaint_base_cache_valid = False
        self.lanpaint_base_cache_turbo = False
        self.lanpaint_base_cache_resident_blocks = 0
        self.lanpaint_base_cond = Optional[Krea2ResidentCond](None)
        self.lanpaint_base_fin = Optional[Krea2StreamFinal](None)
        self.lanpaint_base_resident_i8 = Optional[Krea2ResidentInt8](None)
        self.lanpaint_base_host_i8 = Optional[Krea2HostInt8Inf](None)

    def set_progress_fd(mut self, fd: Int32):
        self.progress_fd = fd

    def backend_name(self) -> String:
        return String("krea2")

    def model_name(self) -> String:
        return String("Krea-2")

    def resident_model(self) -> String:
        # The server owns cross-family routing. FlowEdit/LanPaint may retain a
        # keyed int8 base internally, but it is not a general-purpose resident
        # checkpoint.
        return String("")

    def start(mut self, params: JobParams) raises:
        if self.active:
            raise Error("Krea2Backend.start: a job is already running")
        reject_unsupported_common_runtime_params(params, String("krea2"))
        var lanpaint = has_lanpaint_sampler_runtime_params(params)
        if has_lanpaint_runtime_params(params) and not lanpaint:
            raise Error(
                "krea2: LanPaint_MaskBlend requires a LanPaint sampler request; "
                "a blend-only request is not a mask-aware inpaint"
            )
        if lanpaint:
            if params.edit_src_image != String(""):
                raise Error("krea2: LanPaint and FlowEdit cannot run in the same request")
            if params.init_image == String("") or params.mask_image == String(""):
                raise Error("krea2 LanPaint requires both init_image and mask_image")
            if params.width != 1024 or params.height != 1024:
                raise Error(
                    String("krea2 LanPaint: unsupported size ")
                    + String(params.width) + "x" + String(params.height)
                    + "; choose the compiled 1024x1024 profile"
                )
            if params.lanpaint_mask_channel == String(""):
                raise Error("krea2 LanPaint requires an explicit mask channel")
            if params.lanpaint_num_steps < 0:
                raise Error("krea2 LanPaint requires lanpaint_num_steps")
            if params.lanpaint_lambda <= 0.0:
                raise Error("krea2 LanPaint requires lanpaint_lambda > 0")
            if params.lanpaint_step_size <= 0.0:
                raise Error("krea2 LanPaint requires lanpaint_step_size > 0")
            if params.lanpaint_beta <= 0.0:
                raise Error("krea2 LanPaint requires lanpaint_beta > 0")
            if params.lanpaint_friction <= 0.0:
                raise Error("krea2 LanPaint requires lanpaint_friction > 0")
            var prompt_mode = params.lanpaint_prompt_mode.lower()
            if (
                prompt_mode != String("image first")
                and prompt_mode != String("prompt first")
            ):
                raise Error(
                    "krea2 LanPaint prompt mode must be Image First or Prompt First"
                )
            if params.lanpaint_inpainting_mode.lower().find("video") >= 0:
                raise Error("krea2 LanPaint worker admits image inpainting only")
            if params.sampler != String("") and params.sampler.lower() != String("euler"):
                raise Error("krea2 LanPaint currently requires the Euler sampler")
            if params.scheduler != String("") and params.scheduler.lower() != String("simple"):
                raise Error("krea2 LanPaint currently requires the Simple scheduler")
            if params.creativity < 0.999999:
                raise Error("krea2 LanPaint currently requires denoise=1.0")
            if params.lanpaint_add_noise.lower() == String("disable"):
                raise Error("krea2 LanPaint add_noise=disable is not supported")
            if params.lanpaint_start_at_step > 0:
                raise Error("krea2 LanPaint start_at_step must be 0/full schedule")
            if (
                params.lanpaint_end_at_step >= 0
                and params.lanpaint_end_at_step < params.steps
            ):
                raise Error("krea2 LanPaint early end_at_step is not supported")
            if params.lanpaint_return_with_leftover_noise.lower() == String("enable"):
                raise Error("krea2 LanPaint leftover-noise output is not supported")
            if params.lanpaint_early_stop < 0:
                raise Error("krea2 LanPaint requires lanpaint_early_stop")
            if params.lanpaint_inner_threshold > 0.0:
                raise Error("krea2 LanPaint semantic inner-threshold stopping is not supported")
            if (
                params.outpaint_left >= 0 or params.outpaint_top >= 0
                or params.outpaint_right >= 0 or params.outpaint_bottom >= 0
                or params.outpaint_feathering >= 0
                or params.threshold_mask_value >= 0.0
                or params.threshold_mask_operator != String("")
            ):
                raise Error("krea2 LanPaint outpaint preprocessing is not supported in this profile")
        elif params.init_image != String("") or params.mask_image != String(""):
            raise Error(
                "krea2: init_image/mask_image requires the LanPaint sampler; "
                "the text-to-image path will not ignore image inputs"
            )
        # T2I dispatches over the existing trainer/cache-producer 1024px-area
        # aspect ladder. FlowEdit has separate compiled 512x512 and 1024x1024 arms.
        if params.edit_src_image == String("") and not lanpaint:
            if not _krea2_t2i_shape_supported(params.width, params.height):
                raise Error(
                    String("krea2: unsupported size ") + String(params.width) + "x"
                    + String(params.height)
                    + " — choose a compiled Krea-2 1024px-area aspect bucket."
                )
        if params.steps < 1:
            raise Error("krea2: steps must be >= 1")
        # Resolve at most one LoRA overlay up front (fail before expensive work).
        var lora_path = String("")
        var lora_mult = Float32(1.0)
        if len(params.loras) > 0:
            lora_path = _resolve_krea2_lora_path(params.loras[0].name)
            lora_mult = Float32(params.loras[0].weight)
        self.params = params.copy()
        self.lora_path = lora_path^
        self.lora_mult = lora_mult
        self.active = True
        self.cancel_flag = False

    def cancel(mut self):
        self.cancel_flag = True

    def _cancelled(mut self, mut r: StepResult) -> Bool:
        if self.cancel_flag:
            self.active = False
            r.step = 0
            r.cancelled = True
            return True
        return False

    def _drop_edit_base_cache(mut self) raises:
        """Release the FlowEdit DiT before a TE/VAE cache miss or base change."""
        if not self.edit_base_cache_valid:
            return
        self.edit_base_resident_i8 = Optional[Krea2ResidentInt8](None)
        self.edit_base_host_i8 = Optional[Krea2HostInt8Inf](None)
        self.edit_base_shared = Optional[Krea2SharedResident](None)
        self.edit_base_cache_valid = False
        self.ctx.synchronize()
        cu_mempool_trim_current(0)
        self.ctx.synchronize()
        print("[krea2-edit] released resident DiT for changed conditioning/source/base")

    def _drop_lanpaint_base_cache(mut self) raises:
        """Release the LanPaint DiT before changed-prompt TE/source-VAE work."""
        if not self.lanpaint_base_cache_valid:
            return
        self.lanpaint_base_cond = Optional[Krea2ResidentCond](None)
        self.lanpaint_base_fin = Optional[Krea2StreamFinal](None)
        self.lanpaint_base_resident_i8 = Optional[Krea2ResidentInt8](None)
        self.lanpaint_base_host_i8 = Optional[Krea2HostInt8Inf](None)
        self.lanpaint_base_cache_valid = False
        self.ctx.synchronize()
        cu_mempool_trim_current(0)
        self.ctx.synchronize()
        print("[krea2-lanpaint] released resident DiT for changed conditioning/source/base")

    def _step_t2i_shape[LH_: Int, LW_: Int, LTMAX_: Int, LFULL_: Int](
        mut self, pos_bin: String, neg_bin: String,
    ) raises -> StepResult:
        """Run the existing inference path at serenity trainer-parity bucket arm."""
        var r = StepResult()
        r.total = self.params.steps

        # 2) load the two contexts (small H2D; LT derived from tensor shape).
        var cond = inline_cond_from_bin[LH_, LW_, LTMAX_](pos_bin, self.ctx)
        var uncond = inline_cond_from_bin[LH_, LW_, LTMAX_](neg_bin, self.ctx)

        # 3) build the fp8-resident base + conditioning weights + final layer.
        var turbo = self.params.model.lower().find("turbo") >= 0
        var checkpoint = String(KREA2_TURBO) if turbo else String(KREA2_RAW)
        print("[krea2] model=", "Krea-2-Turbo" if turbo else "Krea-2-Raw",
              " checkpoint=", checkpoint)
        var st = ShardedSafeTensors.open(checkpoint)
        var cond_w = load_krea2_resident_cond(st, String(KREA2_RAW_KEY_PREFIX), self.ctx)
        var fin = Krea2StreamFinal.load(st, String(KREA2_RAW_KEY_PREFIX), self.ctx)
        # 16GB fit: PARTIAL fp8 residency. The full 28-block fp8 base (~12GB) +
        # 1024-area activations OOMs on a 16GB card, so keep only the FIRST K
        # blocks resident and stream the remainder through the same block math.
        var res_k = env_int(String("KREA2_FP8_RESIDENT_BLOCKS"), 14)
        if res_k < 0:
            res_k = 0
        if res_k > KREA2_NBLOCKS:
            res_k = KREA2_NBLOCKS
        print("[krea2] building fp8-resident base (", res_k, "of", KREA2_NBLOCKS,
              "blocks resident, rest streamed per step) ...")
        var resident = Optional[Krea2ResidentFp8](
            build_krea2_resident_fp8(
                st, String(KREA2_RAW_KEY_PREFIX), KREA2_NBLOCKS, res_k, self.ctx,
            )
        )
        if self._cancelled(r):
            return r^

        # 4) LoRA overlay (ADDED, never fused) or base-only.
        var lora: Krea2StackLora
        if self.lora_path != String(""):
            lora = load_krea2_stack_lora(self.lora_path, self.lora_mult, self.ctx)
        else:
            lora = empty_krea2_stack_lora()

        # 5) fixed-LTMAX length-bucket denoise + creator CFG semantics -> latent.
        # Krea's public guidance value is already cond + cfg*(cond-uncond), and
        # cfg=0 disables the unconditional branch. Do not translate it into the
        # unrelated textbook/Comfy convention at this model boundary.
        var krea_guidance = Float32(self.params.cfg)
        print("[krea2] creator CFG=", self.params.cfg,
              " fixed_mu_1_15=", turbo)
        var latent = krea2_sample_latent[LH_, LW_, LTMAX_, LFULL_](
            st, String(KREA2_RAW_KEY_PREFIX), cond_w, fin, lora, cond, uncond,
            self.params.steps, krea_guidance, UInt64(self.params.seed),
            resident, self.ctx,
            use_fixed_mu_1_15=turbo,
            progress_fd=self.progress_fd,
        )
        if self._cancelled(r):
            return r^

        # 6) VAE decode -> <out_dir>/<job_id>.png.
        var png_path = self.params.out_dir + "/" + self.params.job_id + ".png"
        krea2_decode_latent_to_png[LH_, LW_](
            latent, String(KREA2_VAE_DIR), png_path, self.ctx,
        )

        self.active = False
        r.step = self.params.steps
        r.done = True
        r.output_path = png_path
        return r^

    def _step_lanpaint_shape[
        HEIGHT_: Int, WIDTH_: Int, LH_: Int, LW_: Int,
        LTMAX_: Int, LFULL_: Int,
    ](
        mut self, pos_bin: String, neg_bin: String,
    ) raises -> StepResult:
        """Run the complete Krea2 LanPaint image-inpaint request in Mojo."""
        var r = StepResult()
        r.total = self.params.steps
        var job_t0 = perf_counter_ns()

        # Contexts were produced by the isolated Mojo text-encoder child before
        # this compiled geometry arm was selected.
        var cond = inline_cond_from_bin[LH_, LW_, LTMAX_](pos_bin, self.ctx)
        var uncond = inline_cond_from_bin[LH_, LW_, LTMAX_](neg_bin, self.ctx)

        # VAE encode and mask preparation happen before the DiT becomes
        # resident. An unchanged upload reuses its normalized latent and final
        # blend pixels; a changed source releases the retained DiT first so the
        # encoder cannot collide with it on the product GPU.
        var source_cache_hit = (
            self.lanpaint_source_cache_valid
            and self.lanpaint_source_cache_path == self.params.init_image
            and len(self.lanpaint_source_cache_latent) == 16 * LH_ * LW_
            and len(self.lanpaint_source_cache_signed_chw) == 3 * HEIGHT_ * WIDTH_
        )
        if not source_cache_hit:
            self._drop_lanpaint_base_cache()
        var source: Krea2LanPaintSource
        if source_cache_hit:
            var cached_latent = self.lanpaint_source_cache_latent.copy()
            var cached_signed = self.lanpaint_source_cache_signed_chw.copy()
            var source_latent = Tensor.from_host(
                cached_latent^, [1, 16, LH_, LW_], STDtype.F32, self.ctx
            )
            source = Krea2LanPaintSource(source_latent^, cached_signed^)
            print("[krea2-lanpaint] source latent cache HIT (upload unchanged)")
        else:
            source = _encode_krea2_lanpaint_source[HEIGHT_, WIDTH_](
                self.params.init_image, self.ctx
            )
            self.lanpaint_source_cache_latent = source.latent.to_host(self.ctx)
            self.lanpaint_source_cache_signed_chw = source.signed_chw.copy()
            self.lanpaint_source_cache_path = self.params.init_image.copy()
            self.lanpaint_source_cache_valid = True
        var preserve_host = load_lanpaint_latent_preserve_mask(
            self.params.mask_image, self.params.lanpaint_mask_channel,
            LW_, LH_,
        )
        var preserve_mask = Tensor.from_host(
            preserve_host.values, [1, 1, LH_, LW_], STDtype.F32, self.ctx
        )
        self.ctx.synchronize()
        cu_mempool_trim_current(0)
        print(
            "[krea2-lanpaint] PHASE source-encode+mask =",
            Float64(perf_counter_ns() - job_t0) / 1.0e9, "s",
        )
        if self._cancelled(r):
            return r^

        var turbo = self.params.model.lower().find("turbo") >= 0
        var checkpoint = String(KREA2_TURBO) if turbo else String(KREA2_RAW)
        print(
            "[krea2-lanpaint] model=",
            "Krea-2-Turbo" if turbo else "Krea-2-Raw",
            " checkpoint=", checkpoint,
        )
        var st = ShardedSafeTensors.open(checkpoint)
        var res_k = env_int(String("KREA2_FP8_RESIDENT_BLOCKS"), 14)
        if res_k < 0:
            res_k = 0
        if res_k > KREA2_NBLOCKS:
            res_k = KREA2_NBLOCKS
        var resident = Optional[Krea2ResidentFp8](None)
        var resident_i8 = Optional[Krea2ResidentInt8](None)
        var host_i8 = Optional[Krea2HostInt8Inf](None)
        var i8_cache = krea2_int8_cache_path(checkpoint)
        var i8_res_blocks = env_int(
            String("KREA2_LANPAINT_I8_RESIDENT_BLOCKS"), 20
        )
        if i8_res_blocks < 0:
            i8_res_blocks = 0
        if i8_res_blocks > KREA2_NBLOCKS:
            i8_res_blocks = KREA2_NBLOCKS
        var i8_cache_valid = krea2_int8_cache_valid(
            i8_cache, checkpoint, KREA2_NBLOCKS
        )
        var base_cache_hit = (
            i8_cache_valid
            and self.lanpaint_base_cache_valid
            and self.lanpaint_base_cache_turbo == turbo
            and self.lanpaint_base_cache_resident_blocks == i8_res_blocks
            and Bool(self.lanpaint_base_cond)
            and Bool(self.lanpaint_base_fin)
        )
        var cond_w: Krea2ResidentCond
        var fin: Krea2StreamFinal
        if base_cache_hit:
            cond_w = self.lanpaint_base_cond.value().copy()
            fin = self.lanpaint_base_fin.value().copy()
            resident_i8 = self.lanpaint_base_resident_i8.copy()
            host_i8 = self.lanpaint_base_host_i8.copy()
            print("[krea2-lanpaint] resident DiT cache HIT (checkpoint/profile unchanged)")
        elif i8_cache_valid:
            self._drop_lanpaint_base_cache()
            cond_w = load_krea2_resident_cond(
                st, String(KREA2_RAW_KEY_PREFIX), self.ctx
            )
            fin = Krea2StreamFinal.load(
                st, String(KREA2_RAW_KEY_PREFIX), self.ctx
            )
            print(
                "[krea2-lanpaint] int8 sidecar resident/host split=",
                i8_res_blocks, "/", KREA2_NBLOCKS - i8_res_blocks,
            )
            if i8_res_blocks > 0:
                resident_i8 = Optional[Krea2ResidentInt8](
                    load_krea2_int8_cache_resident(
                        i8_cache, i8_res_blocks, self.ctx
                    )
                )
            if i8_res_blocks < KREA2_NBLOCKS:
                host_i8 = Optional[Krea2HostInt8Inf](
                    load_krea2_int8_cache_host(
                        i8_cache, KREA2_NBLOCKS, i8_res_blocks, self.ctx
                    )
                )
            self.lanpaint_base_cond = Optional[Krea2ResidentCond](cond_w.copy())
            self.lanpaint_base_fin = Optional[Krea2StreamFinal](fin.copy())
            self.lanpaint_base_resident_i8 = resident_i8.copy()
            self.lanpaint_base_host_i8 = host_i8.copy()
            self.lanpaint_base_cache_turbo = turbo
            self.lanpaint_base_cache_resident_blocks = i8_res_blocks
            self.lanpaint_base_cache_valid = True
        else:
            self._drop_lanpaint_base_cache()
            cond_w = load_krea2_resident_cond(
                st, String(KREA2_RAW_KEY_PREFIX), self.ctx
            )
            fin = Krea2StreamFinal.load(
                st, String(KREA2_RAW_KEY_PREFIX), self.ctx
            )
            print(
                "[krea2-lanpaint] WARN no fresh int8 sidecar; using bounded ",
                res_k, "-block FP8 resident fallback",
            )
            resident = Optional[Krea2ResidentFp8](
                build_krea2_resident_fp8(
                    st, String(KREA2_RAW_KEY_PREFIX), KREA2_NBLOCKS,
                    res_k, self.ctx,
                )
            )
        var lora: Krea2StackLora
        if self.lora_path != String(""):
            lora = load_krea2_stack_lora(
                self.lora_path, self.lora_mult, self.ctx
            )
        else:
            lora = empty_krea2_stack_lora()
        if self._cancelled(r):
            return r^

        var noise_seed = self.params.seed
        if self.params.lanpaint_noise_seed >= 0:
            noise_seed = self.params.lanpaint_noise_seed
        var latent = krea2_sample_lanpaint_latent[
            LH_, LW_, LTMAX_, LFULL_
        ](
            st, String(KREA2_RAW_KEY_PREFIX), cond_w, fin, lora,
            cond, uncond, source.latent, preserve_mask,
            self.params.steps, Float32(self.params.cfg), UInt64(noise_seed),
            self.params.lanpaint_num_steps,
            Float32(self.params.lanpaint_lambda),
            Float32(self.params.lanpaint_step_size),
            Float32(self.params.lanpaint_beta),
            Float32(self.params.lanpaint_friction),
            self.params.lanpaint_prompt_mode,
            self.params.lanpaint_early_stop,
            resident, resident_i8, host_i8, self.ctx,
            use_fixed_mu_1_15=turbo,
            progress_fd=self.progress_fd,
        )
        if self._cancelled(r):
            return r^

        var decode_t0 = perf_counter_ns()
        var dec = QwenImageVaeDecoder[LH_, LW_].load(
            String(KREA2_VAE_DIR), self.ctx
        )
        var latent_bf16 = torch_f32_to_bf16_rne(latent, self.ctx)
        var painted = dec.decode(latent_bf16, self.ctx)
        var output_image: Tensor
        if self.params.lanpaint_mask_blend_overlap >= 0:
            var painted_host = painted.to_host(self.ctx)
            var pixel_mask = load_lanpaint_pixel_blend_mask(
                self.params.mask_image, self.params.lanpaint_mask_channel,
                WIDTH_, HEIGHT_, self.params.lanpaint_mask_blend_overlap,
            )
            var blended = apply_lanpaint_mask_blend_signed_chw(
                source.signed_chw, painted_host, pixel_mask
            )
            output_image = Tensor.from_host(
                blended^, [1, 3, HEIGHT_, WIDTH_], STDtype.F32, self.ctx
            )
        else:
            output_image = painted^
        var png_path = self.params.out_dir + "/" + self.params.job_id + ".png"
        save_png(output_image, png_path, self.ctx, ValueRange.SIGNED)
        print(
            "[krea2-lanpaint] PHASE decode+mask-blend =",
            Float64(perf_counter_ns() - decode_t0) / 1.0e9,
            "s; total =", Float64(perf_counter_ns() - job_t0) / 1.0e9, "s",
        )

        self.active = False
        r.step = self.params.steps
        r.done = True
        r.output_path = png_path
        return r^

    def step(mut self) raises -> StepResult:
        var r = StepResult()
        r.total = self.params.steps
        if not self.active:
            r.failed = True
            r.error = String("no active job")
            return r^
        if self._cancelled(r):
            return r^
        try:
            if self.params.edit_src_image != String(""):
                self._drop_lanpaint_base_cache()
                return self._step_flowedit()
            var lanpaint = has_lanpaint_sampler_runtime_params(self.params)
            # FlowEdit and LanPaint use distinct resident carriers. Never let a
            # preceding FlowEdit base collide with either path's text encoder.
            self._drop_edit_base_cache()
            var jid = self.params.job_id
            var pos_bin = String("/tmp/serenity_krea2_ctx_") + jid + String(".pos.bin")
            var neg_bin = String("/tmp/serenity_krea2_ctx_") + jid + String(".neg.bin")

            # 1) ENCODE (fork+execv child TE -> context bins, VRAM reclaimed on
            # exit). LanPaint Regenerate reuses unchanged bins and can retain
            # its DiT; a prompt miss releases that DiT before the child starts.
            var lanpaint_ctx_hit = (
                lanpaint
                and self.lanpaint_ctx_cache_valid
                and self.lanpaint_ctx_cache_prompt == self.params.prompt
                and self.lanpaint_ctx_cache_negative == self.params.negative
                and _path_exists(self.lanpaint_ctx_cache_pos_bin)
                and _path_exists(self.lanpaint_ctx_cache_neg_bin)
            )
            if lanpaint_ctx_hit:
                pos_bin = self.lanpaint_ctx_cache_pos_bin.copy()
                neg_bin = self.lanpaint_ctx_cache_neg_bin.copy()
                print("[krea2-lanpaint] conditioning cache HIT (prompts unchanged)")
            else:
                self._drop_lanpaint_base_cache()
                krea2_encode_contexts_subprocess(
                    self.params.prompt, self.params.negative,
                    pos_bin, neg_bin, self.ctx,
                )
                if lanpaint:
                    self.lanpaint_ctx_cache_valid = True
                    self.lanpaint_ctx_cache_prompt = self.params.prompt.copy()
                    self.lanpaint_ctx_cache_negative = self.params.negative.copy()
                    self.lanpaint_ctx_cache_pos_bin = pos_bin.copy()
                    self.lanpaint_ctx_cache_neg_bin = neg_bin.copy()
            if self._cancelled(r):
                return r^

            if lanpaint:
                return self._step_lanpaint_shape[
                    1024, 1024, 128, 128, 768, 4864
                ](pos_bin, neg_bin)

            # Plain text-to-image does not currently retain an int8 base.
            self._drop_lanpaint_base_cache()

            # Runtime request -> existing COMPTIME trainer/cache bucket arm.
            comptime for bi in range(KREA2_LADDER_LEN):
                comptime X100_BI = KREA2_LADDER_X100[bi]
                comptime LH_BI = krea2_lat_h(X100_BI, KREA2_EDGE_UNITS)
                comptime LW_BI = krea2_lat_w(X100_BI, KREA2_EDGE_UNITS)
                comptime IMGLEN_BI = (LH_BI // 2) * (LW_BI // 2)
                comptime LUNALIGNED_BI = KREA2_LTMAX_BASE + IMGLEN_BI
                comptime LTMAX_BI = (
                    KREA2_LTMAX_BASE + (128 - (LUNALIGNED_BI % 128)) % 128
                )
                comptime LFULL_BI = (
                    LTMAX_BI + IMGLEN_BI
                )
                if (
                    self.params.width == LW_BI * 8
                    and self.params.height == LH_BI * 8
                ):
                    return self._step_t2i_shape[LH_BI, LW_BI, LTMAX_BI, LFULL_BI](
                        pos_bin, neg_bin,
                    )
            raise Error("krea2: admitted aspect bucket dispatch was not compiled")
        except e:
            self.active = False
            r.failed = True
            r.error = String(e)
            return r^

    def _step_flowedit_shape[FE_HEIGHT_: Int, FE_WIDTH_: Int](
        mut self
    ) raises -> StepResult:
        """FlowEdit worker mode (Phase C1): training-free instruction edit at the
        pipeline's compiled square geometry. Ports pipeline/krea2_flowedit.main:
        encode 4 contexts in-process -> stage+VAE-encode the source PNG ->
        int8 W8A8 base (sidecar cache) -> FlowEdit ODE (4 forwards/step,
        optional auto-mask) -> decode + save. Fail-loud on unsupported geometry
        and over-length prompts (LT_SHARED), never silent."""
        comptime FE_LH_ = FE_HEIGHT_ // 8
        comptime FE_LW_ = FE_WIDTH_ // 8
        comptime FE_NTOK_ = (FE_LH_ // 2) * (FE_LW_ // 2)
        var r = StepResult()
        r.total = self.params.steps
        var jid = self.params.job_id
        var job_t0 = perf_counter_ns()
        if len(self.params.loras) > 0:
            raise Error("krea2 FlowEdit: LoRA is not admitted in the edit mode (training-free path)")

        # ── 1) FOUR contexts via the in-process TE (two encode passes). ──
        var sp_bin = String("/tmp/serenity_krea2_edit_") + jid + String(".srcpos.bin")
        var sn_bin = String("/tmp/serenity_krea2_edit_") + jid + String(".srcneg.bin")
        var tp_bin = String("/tmp/serenity_krea2_edit_") + jid + String(".tgtpos.bin")
        var tn_bin = String("/tmp/serenity_krea2_edit_") + jid + String(".tgtneg.bin")
        var ctx_cache_hit = (
            self.edit_ctx_cache_valid
            and self.edit_ctx_cache_src_prompt == self.params.edit_src_prompt
            and self.edit_ctx_cache_src_negative == self.params.edit_src_negative
            and self.edit_ctx_cache_tgt_prompt == self.params.prompt
            and self.edit_ctx_cache_tgt_negative == self.params.negative
            and _path_exists(self.edit_ctx_cache_sp_bin)
            and _path_exists(self.edit_ctx_cache_sn_bin)
            and _path_exists(self.edit_ctx_cache_tp_bin)
            and _path_exists(self.edit_ctx_cache_tn_bin)
        )
        # A text-encoder child cannot coexist with the resident DiT on the
        # product GPU. Drop it before a prompt miss; the new DiT becomes the
        # resident cache again after the encode phases complete.
        if not ctx_cache_hit:
            self._drop_edit_base_cache()
        if ctx_cache_hit:
            sp_bin = self.edit_ctx_cache_sp_bin.copy()
            sn_bin = self.edit_ctx_cache_sn_bin.copy()
            tp_bin = self.edit_ctx_cache_tp_bin.copy()
            tn_bin = self.edit_ctx_cache_tn_bin.copy()
            print("[krea2-edit] conditioning cache HIT (four prompts unchanged)")
        else:
            krea2_encode_contexts_subprocess(
                self.params.edit_src_prompt, self.params.edit_src_negative,
                sp_bin, sn_bin, self.ctx,
            )
            if self._cancelled(r):
                return r^
            krea2_encode_contexts_subprocess(
                self.params.prompt, self.params.negative, tp_bin, tn_bin, self.ctx,
            )
            self.edit_ctx_cache_valid = True
            self.edit_ctx_cache_src_prompt = self.params.edit_src_prompt.copy()
            self.edit_ctx_cache_src_negative = self.params.edit_src_negative.copy()
            self.edit_ctx_cache_tgt_prompt = self.params.prompt.copy()
            self.edit_ctx_cache_tgt_negative = self.params.negative.copy()
            self.edit_ctx_cache_sp_bin = sp_bin.copy()
            self.edit_ctx_cache_sn_bin = sn_bin.copy()
            self.edit_ctx_cache_tp_bin = tp_bin.copy()
            self.edit_ctx_cache_tn_bin = tn_bin.copy()
        print("[krea2-edit] PHASE text-encode =",
              Float64(perf_counter_ns() - job_t0) / 1.0e9, "s")
        if self._cancelled(r):
            return r^
        # _load_context_padded pads to LT_SHARED and FAIL-LOUDS on over-length
        # prompts ("shorten the prompt") — the C-phase node contract.
        var src_pos_pair = _load_context_padded_shape[FE_LT_SHARED](sp_bin, String("SRC_POS"), self.ctx)
        var src_neg_pair = _load_context_padded_shape[FE_LT_SHARED](sn_bin, String("SRC_NEG"), self.ctx)
        var tgt_pos_pair = _load_context_padded_shape[FE_LT_SHARED](tp_bin, String("TGT_POS"), self.ctx)
        var tgt_neg_pair = _load_context_padded_shape[FE_LT_SHARED](tn_bin, String("TGT_NEG"), self.ctx)
        var ctx_src_pos = src_pos_pair[0].clone(self.ctx)
        var lt_src_pos = src_pos_pair[1]
        var ctx_src_neg = src_neg_pair[0].clone(self.ctx)
        var lt_src_neg = src_neg_pair[1]
        var ctx_tgt_pos = tgt_pos_pair[0].clone(self.ctx)
        var lt_tgt_pos = tgt_pos_pair[1]
        var ctx_tgt_neg = tgt_neg_pair[0].clone(self.ctx)
        var lt_tgt_neg = tgt_neg_pair[1]
        var pos_grid = _build_pos_shape[FE_LH_, FE_LW_, FE_LT_SHARED](self.ctx)

        # ── 2) source PNG -> square signed NCHW -> normalized Z0_src. ──
        var source_cache_hit = (
            self.edit_latent_cache_valid
            and self.edit_latent_cache_source == self.params.edit_src_image
            and self.edit_latent_cache_width == FE_WIDTH_
            and self.edit_latent_cache_height == FE_HEIGHT_
        )
        # Same constraint as the TE: a changed source needs the VAE encoder, so
        # release any retained DiT before loading that encoder.
        if not source_cache_hit:
            self._drop_edit_base_cache()
        var z0_src: Tensor
        if source_cache_hit:
            var cached_source = self.edit_latent_cache.copy()
            z0_src = Tensor.from_host(
                cached_source^, [1, 16, FE_LH_, FE_LW_], STDtype.F32, self.ctx
            )
            print("[krea2-edit] source latent cache HIT (upload and geometry unchanged)")
        else:
            var img = decode_image_any(self.params.edit_src_image)
            var resized = resize_bilinear(img, FE_WIDTH_, FE_HEIGHT_)
            var host = image_to_signed_nchw(resized)
            var image_t = Tensor.from_host(host, [1, 3, FE_HEIGHT_, FE_WIDTH_], STDtype.F32, self.ctx)
            var img_bf16 = cast_tensor(image_t, STDtype.BF16, self.ctx)
            var enc = QwenImageVaeEncoder[FE_HEIGHT_, FE_WIDTH_].load(KREA2_VAE_ENC_FILE, self.ctx)
            var lat_mean = enc.encode_mean(img_bf16, self.ctx)
            var lat_f32 = cast_tensor(lat_mean, STDtype.F32, self.ctx)
            var mean_ch = _mean_ch(self.ctx)
            var std_ch = _std_ch(self.ctx)
            z0_src = _normalize_latent(lat_f32, mean_ch, std_ch, self.ctx)
            self.edit_latent_cache = z0_src.to_host(self.ctx)
            self.edit_latent_cache_source = self.params.edit_src_image.copy()
            self.edit_latent_cache_width = FE_WIDTH_
            self.edit_latent_cache_height = FE_HEIGHT_
            self.edit_latent_cache_valid = True
        self.ctx.synchronize()
        var vae_encode_done = perf_counter_ns()
        print("[krea2-edit] PHASE vae-encode =",
              Float64(vae_encode_done - job_t0) / 1.0e9, "s cumulative")
        cu_mempool_trim_current(0)
        if self._cancelled(r):
            return r^

        # ── 3) int8 W8A8 base, sidecar-cache-first (the fast startup path). ──
        var turbo = self.params.model.lower().find("turbo") >= 0
        var checkpoint = String(KREA2_TURBO) if turbo else String(KREA2_RAW)
        print("[krea2-edit] model=", "Krea-2-Turbo" if turbo else "Krea-2-Raw",
              " checkpoint=", checkpoint)
        var st = ShardedSafeTensors.open(checkpoint)
        var i8_res_blocks = env_int(String("KREA2_EDIT_I8_RESIDENT_BLOCKS"), 20)
        if i8_res_blocks < 0:
            i8_res_blocks = 0
        if i8_res_blocks > FE_NBLOCKS:
            i8_res_blocks = FE_NBLOCKS
        var resident_i8 = Optional[Krea2ResidentInt8](None)
        var host_i8 = Optional[Krea2HostInt8Inf](None)
        var shared = Optional[Krea2SharedResident](None)
        var base_cache_hit = (
            self.edit_base_cache_valid
            and self.edit_base_cache_turbo == turbo
            and self.edit_base_cache_resident_blocks == i8_res_blocks
        )
        if base_cache_hit:
            resident_i8 = self.edit_base_resident_i8.copy()
            host_i8 = self.edit_base_host_i8.copy()
            shared = self.edit_base_shared.copy()
            print("[krea2-edit] resident DiT cache HIT (checkpoint/profile unchanged)")
        else:
            self._drop_edit_base_cache()
            var i8_cache = krea2_int8_cache_path(checkpoint)
            if krea2_int8_cache_valid(i8_cache, checkpoint, FE_NBLOCKS):
                print("[krea2-edit] int8 sidecar found:", i8_cache)
                if i8_res_blocks > 0:
                    resident_i8 = Optional[Krea2ResidentInt8](
                        load_krea2_int8_cache_resident(i8_cache, i8_res_blocks, self.ctx)
                    )
                if i8_res_blocks < FE_NBLOCKS:
                    host_i8 = Optional[Krea2HostInt8Inf](
                        load_krea2_int8_cache_host(i8_cache, FE_NBLOCKS, i8_res_blocks, self.ctx)
                    )
                shared = Optional[Krea2SharedResident](
                    load_krea2_int8_cache_shared(i8_cache, self.ctx)
                )
            else:
                print("[krea2-edit] building int8 W8A8 base: resident", i8_res_blocks,
                      "/", FE_NBLOCKS, "blocks ...")
                if i8_res_blocks > 0:
                    resident_i8 = Optional[Krea2ResidentInt8](
                        build_krea2_resident_int8(
                            st, KREA2_RAW_KEY_PREFIX, FE_NBLOCKS, i8_res_blocks, self.ctx
                        )
                    )
                if i8_res_blocks < FE_NBLOCKS:
                    host_i8 = Optional[Krea2HostInt8Inf](
                        build_krea2_host_int8_inf(
                            st, KREA2_RAW_KEY_PREFIX, FE_NBLOCKS, i8_res_blocks, self.ctx
                        )
                    )
                shared = Optional[Krea2SharedResident](
                    build_krea2_shared_resident(st, KREA2_RAW_KEY_PREFIX, self.ctx)
                )
                try:
                    save_krea2_int8_cache(
                        resident_i8, host_i8, shared.value(), checkpoint,
                        i8_cache, FE_NBLOCKS, self.ctx,
                    )
                except e:
                    print("[krea2-edit] WARN could not write int8 sidecar:", String(e))
            self.edit_base_resident_i8 = resident_i8.copy()
            self.edit_base_host_i8 = host_i8.copy()
            self.edit_base_shared = shared.copy()
            self.edit_base_cache_turbo = turbo
            self.edit_base_cache_resident_blocks = i8_res_blocks
            self.edit_base_cache_valid = True
        self.ctx.synchronize()
        var base_load_done = perf_counter_ns()
        print("[krea2-edit] PHASE base-load =",
              Float64(base_load_done - vae_encode_done) / 1.0e9, "s")
        if self._cancelled(r):
            return r^

        # ── 4) FlowEdit ODE (Euler on the velocity DIFFERENCE). ──
        var steps = self.params.steps
        var n_max = self.params.edit_nmax
        if n_max < 0:
            n_max = 24
        if n_max > steps:
            n_max = steps
        var n_min = self.params.edit_nmin
        if n_min < 0:
            n_min = 0
        var src_cfg = Float32(self.params.edit_src_cfg)
        if self.params.edit_src_cfg < 0.0:
            src_cfg = Float32(1.5)
        var tgt_cfg = Float32(self.params.cfg)
        var seed = UInt64(self.params.seed)
        var auto_mask = self.params.edit_auto_mask
        var mask_q = Float32(self.params.edit_mask_q)
        var mask_dilate = self.params.edit_mask_dilate
        var mask_warmup = self.params.edit_mask_warmup
        var seq = krea2_packed_seq_len(FE_HEIGHT_, FE_WIDTH_)
        var ts = krea2_timesteps(
            seq, steps, Float32(1.15), use_mu_override=turbo
        )
        var skip_before = steps - n_max
        var stop_at = steps - n_min
        print("[krea2-edit] FlowEdit", FE_HEIGHT_, "x", FE_WIDTH_, " steps=", steps,
              " window=[", skip_before, ",", stop_at, ") src_cfg=", src_cfg,
              " tgt_cfg=", tgt_cfg, " seed=", seed, " auto_mask=", auto_mask)
        var denoise_t0 = perf_counter_ns()

        var z_edit = z0_src.clone(self.ctx)
        var z0_host = List[Float32]()
        if auto_mask:
            z0_host = self.edit_latent_cache.copy()
        var sal = List[Float32]()
        for _ in range(FE_NTOK_):
            sal.append(Float32(0.0))
        var active_count = 0
        for si in range(steps):
            if si < skip_before or si >= stop_at:
                _emit_flowedit_progress(self.progress_fd, si + 1, steps)
                continue
            if self._cancelled(r):
                return r^
            var t_cur = ts[si]
            var t_prev = ts[si + 1]
            var t_t = Tensor.from_host([t_cur], [1], STDtype.F32, self.ctx)
            var noise = randn([1, 16, FE_LH_, FE_LW_], seed + UInt64(si), STDtype.F32, self.ctx)
            var zt_src = add(
                mul_scalar(z0_src, Float32(1.0) - t_cur, self.ctx),
                mul_scalar(noise, t_cur, self.ctx),
                self.ctx,
            )
            var zt_tgt = add(z_edit, sub(zt_src, z0_src, self.ctx), self.ctx)
            var v_src = _velocity_shape[FE_LH_, FE_LW_, FE_LT_SHARED](
                st, zt_src, ctx_src_pos, lt_src_pos, ctx_src_neg, lt_src_neg,
                pos_grid, t_t, src_cfg, resident_i8, host_i8, shared, self.ctx,
            )
            var v_tgt = _velocity_shape[FE_LH_, FE_LW_, FE_LT_SHARED](
                st, zt_tgt, ctx_tgt_pos, lt_tgt_pos, ctx_tgt_neg, lt_tgt_neg,
                pos_grid, t_t, tgt_cfg, resident_i8, host_i8, shared, self.ctx,
            )
            var dv = sub(v_tgt, v_src, self.ctx)
            z_edit = add(z_edit, mul_scalar(dv, t_prev - t_cur, self.ctx), self.ctx)
            if auto_mask:
                active_count += 1
                var dv_host = dv.to_host(self.ctx)
                _accum_saliency_shape[FE_LH_, FE_LW_](dv_host, sal)
                if active_count > mask_warmup:
                    var mask = _mask_from_saliency_shape[FE_LH_, FE_LW_](
                        sal, mask_q, mask_dilate
                    )
                    var z_host = z_edit.to_host(self.ctx)
                    _blend_outside_mask_shape[FE_LH_, FE_LW_](
                        z_host, z0_host, mask
                    )
                    z_edit = Tensor.from_host(
                        z_host^, [1, 16, FE_LH_, FE_LW_], STDtype.F32, self.ctx
                    )
            r.step = si + 1
            _emit_flowedit_progress(self.progress_fd, si + 1, steps)
            print("[krea2-edit] step", si, "/", steps, " t=", t_cur)

        self.ctx.synchronize()
        var denoise_done = perf_counter_ns()
        print("[krea2-edit] PHASE denoise =",
              Float64(denoise_done - denoise_t0) / 1.0e9, "s; active_steps=",
              active_count if auto_mask else n_max)

        # ── 5) decode + save (+ mask debug artifact). ──
        var png_path = self.params.out_dir + "/" + jid + ".png"
        if auto_mask and active_count > 0:
            var final_mask = _mask_from_saliency_shape[FE_LH_, FE_LW_](
                sal, mask_q, mask_dilate
            )
            var mask_path = _save_mask_png_shape[
                FE_HEIGHT_, FE_WIDTH_, FE_LH_, FE_LW_
            ](final_mask, png_path, self.ctx)
            print("[krea2-edit] mask artifact:", mask_path)
        var dec = QwenImageVaeDecoder[FE_LH_, FE_LW_].load(String(KREA2_VAE_DIR), self.ctx)
        var latent_bf16 = torch_f32_to_bf16_rne(z_edit, self.ctx)
        var image = dec.decode(latent_bf16, self.ctx)
        save_png(image, png_path, self.ctx, ValueRange.SIGNED)
        print("[krea2-edit] wrote", png_path)
        print("[krea2-edit] PHASE decode-save =",
              Float64(perf_counter_ns() - denoise_done) / 1.0e9, "s; total =",
              Float64(perf_counter_ns() - job_t0) / 1.0e9, "s")

        self.active = False
        r.step = self.params.steps
        r.done = True
        r.output_path = png_path
        return r^

    def _step_flowedit(mut self) raises -> StepResult:
        """Dispatch the user-selected Krea FlowEdit square to a compiled arm."""
        if self.params.width == 512 and self.params.height == 512:
            return self._step_flowedit_shape[512, 512]()
        if self.params.width == 1024 and self.params.height == 1024:
            return self._step_flowedit_shape[1024, 1024]()
        raise Error(
            String("krea2 FlowEdit: unsupported size ")
            + String(self.params.width) + "x" + String(self.params.height)
            + "; choose the compiled 512x512 or 1024x1024 profile"
        )

    def between_jobs_trim(mut self) raises:
        # Release transient allocations while preserving any Arc-owned FlowEdit
        # or LanPaint base tensors retained by the keyed unchanged-request cache.
        self.ctx.synchronize()
        cu_mempool_trim_current(0)
        self.ctx.synchronize()
