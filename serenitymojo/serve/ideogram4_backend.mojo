# serenitymojo.serve.ideogram4_backend -- bounded native Ideogram-4 GenBackend.
#
# This is a real Mojo backend path, not a subprocess wrapper. It reuses the
# native Ideogram-4 components:
#   Qwen3-VL 13-tap text encode -> fp8 cond/uncond DiT Euler denoise on either
#   the Ideogram logit-normal schedule or the Comfy simple AuraFlow schedule ->
#   latent denorm -> Ideogram VAE decode.
#
# Current product limits are explicit and fail-loud:
#   * txt2img uses the seven compiled 1024-area core shapes. FlowEdit remains
#     1024-only and has its own request gate.
#   * no negative prompt, init image, variation, or non-Ideogram schedulers
#   * txt2img accepts one additive Ideogram PEFT/Serenity LoRA on the conditional
#     creator trunk; FlowEdit remains fail-loud for LoRA
#   * fixed 1024 token text window so the DiT sequence is compile-time static
#
# Residency note: the Qwen3-VL text encoder and the two fp8 transformers do not
# fit together on the 24 GB target class. Each job encodes text first, then loads
# cond/uncond transformers for denoise, then frees them before VAE decode. The
# DiT weights are resident across denoise steps, not across jobs.

from std.gpu.host import DeviceContext
from std.memory import alloc, ArcPointer
from std.time import perf_counter_ns

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.ffi import (
    BytePtr, sys_open, sys_close, sys_pwrite, O_WRONLY, O_CREAT, O_TRUNC, O_RDONLY,
)
from serenitymojo.io.sharded import ShardedSafeTensors
from image.buffer import Image
from image.png import encode_png_with_text
from serenitymojo.image.png import _quantize, ValueRange
from serenitymojo.models.dit.ideogram4_mrope import build_ideogram4_mrope
from serenitymojo.models.dit.ideogram4_resident import (
    Ideogram4Weights, Ideogram4Masks, ideogram4_forward_r,
    ideogram4_forward_r_masked, ideogram4_build_masks,
)
from serenitymojo.models.text_encoder.ideogram_qwen3vl_streamed import (
    encode_ideogram_taps_streamed,
)

# ── ideogram4 FlowEdit worker mode (Phase C3): reuse the VERIFIED pipeline
#    (pipeline/ideogram4_flowedit.mojo — streamed TE + single trunk, 1024²)
#    behind the GenBackend seam. edit_src_image != "" routes step() there.
from serenitymojo.pipeline.ideogram4_flowedit import (
    _encode_prompt_pair as i4fe_encode_prompt_pair,
    _flowedit_denoise as i4fe_denoise,
    _decode_and_report as i4fe_decode,
    LATENT_NORM as I4FE_LATENT_NORM,
    VAE as I4FE_VAE,
    HEIGHT as I4FE_HEIGHT, WIDTH as I4FE_WIDTH,
    LH as I4FE_LH, LW as I4FE_LW, NIMG as I4FE_NIMG,
)
from serenitymojo.models.vae.ldm_encoder import (
    load_ideogram4_vae_encoder, encode_ideogram4_latents,
)
from serenitymojo.serve.image_io import decode_image_any, image_to_signed_nchw
from image.transform import resize_bilinear
from serenitymojo.serve.product_manifest import write_text_file
from serenitymojo.models.vae.ideogram4_tiled_decode import ideogram4_tiled_decode
from serenitymojo.models.vae.ldm_decoder import load_ideogram4_vae_decoder
from serenitymojo.offload.vmm_cuda import cu_mempool_trim_current, cu_mem_get_info
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.random import randn
from serenitymojo.ops.tensor_algebra import (
    add, concat, mul, mul_scalar, permute, reshape, slice,
)
from serenitymojo.sampling.ideogram4_schedule import (
    ideogram4_logitnormal, ideogram4_schedule_mean, make_step_intervals,
)
from serenitymojo.sampling.sampler_registry import (
    sampler_admission_for_backend, scheduler_admission_for_backend,
)
from serenitymojo.serve.backend import (
    GenBackend, JobParams, StepResult, reject_unsupported_common_runtime_params,
    reject_unsupported_reference_image_params, reject_unsupported_mask_image_params,
    reject_unsupported_inpaint_conditioning_params,
    reject_unsupported_qwen_edit_conditioning_params,
    reject_unsupported_conditioning_mask_params, reject_unsupported_lanpaint_params,
)
from serenitymojo.tokenizer.tokenizer import Qwen3Tokenizer


comptime COND = "models/ideogram4/transformer/diffusion_pytorch_model.safetensors"
comptime UNCOND = "models/ideogram4/unconditional_transformer/diffusion_pytorch_model.safetensors"
comptime TE = "models/ideogram4/text_encoder/model.safetensors"
comptime TOK_JSON = "models/ideogram4/tokenizer/tokenizer.json"
comptime VAE = "models/ideogram4/vae/diffusion_pytorch_model.safetensors"
comptime LATENT_NORM = "serenitymojo/models/dit/parity/ideogram4_fx_latentnorm.safetensors"

comptime IMG_OFFSET = 65536
comptime PAD_ID = 151643
comptime TEXT_TOKENS = 1024
# Default bucket = 1024x1024 latent grid (GH=GW=64). The per-resolution work is
# now factored onto comptime [GH, GW] (see IdeoBucket / the _*_b[GH,GW] helpers);
# these module constants are the 1024x1024 specialization the original path used
# and stay bit-identical to that path.
comptime GH = 64
comptime GW = 64
comptime NIMG = GH * GW
comptime TOTAL = TEXT_TOKENS + NIMG
comptime LLM_DIM = 53248
comptime HIDDEN = 4608
comptime HEADS = 18
comptime HEAD_DIM = 256
comptime LAYERS = 34
comptime LATENT_DIM = 128
comptime VAE_H = 2 * GH
comptime VAE_W = 2 * GW

# Whole-image VAE decode is preferred when it fits: tiled decode is MEASURED to
# degrade output (same-latent A/B: 3x3 = 64.3% px differ / +2.2 brightness vs
# whole on a 1024 portrait; MJ-1054). After the DiT free + mempool trim below we
# query free VRAM and decode whole only when it clears this bar, else fall back
# to the (degrading) tiled path. Whole 1024 ideogram4 decode fits in a clean
# process; 14 GiB is a conservative estimate to be tightened by measurement.
comptime WHOLE_DECODE_MIN_FREE_BYTES = 14 * 1024 * 1024 * 1024  # 14 GiB

# ── Resolution buckets (comptime [GH, GW] dispatch) ──────────────────────────
# patch == 16 image px per latent-grid cell (verified vs the 1024 path: the latent
# grid is GH x GW, each cell unpacks 2x2 (reshape [1,GH,GW,2,2,32] -> permute ->
# [1,32,2*GH,2*GW]) and the VAE upsamples x8, so pixels = 16*GH by 16*GW).
# Keep this finite and explicit: each arm becomes a compile-time transformer
# sequence and VAE decoder specialization. The order matches the shared product
# ladder used by the other image backends.
comptime BUCKET_SQUARE = 0          # 1024(W) x 1024(H) -> GH=64, GW=64
comptime BUCKET_1152X896 = 1        # 1152(W) x 896(H)  -> GH=56, GW=72
comptime BUCKET_896X1152 = 2        # 896(W)  x 1152(H) -> GH=72, GW=56
comptime BUCKET_1344X768 = 3        # 1344(W) x 768(H)  -> GH=48, GW=84
comptime BUCKET_768X1344 = 4        # 768(W)  x 1344(H) -> GH=84, GW=48
comptime BUCKET_1280X832 = 5        # 1280(W) x 832(H)  -> GH=52, GW=80
comptime BUCKET_832X1280 = 6        # 832(W)  x 1280(H) -> GH=80, GW=52
comptime BUCKET_COUNT = 7

comptime GH_1152X896 = 56
comptime GW_1152X896 = 72
comptime GH_896X1152 = 72
comptime GW_896X1152 = 56
comptime GH_1344X768 = 48
comptime GW_1344X768 = 84
comptime GH_768X1344 = 84
comptime GW_768X1344 = 48
comptime GH_1280X832 = 52
comptime GW_1280X832 = 80
comptime GH_832X1280 = 80
comptime GW_832X1280 = 52

comptime IPHASE_IDLE = 0
comptime IPHASE_ENCODE = 1
comptime IPHASE_LOAD = 2
comptime IPHASE_PREPARE = 3
comptime IPHASE_DENOISE = 4
comptime IPHASE_DECODE = 5

comptime TArc = ArcPointer[Tensor]
comptime GENPARAMS_TEXT_KEY = "serenity.genparams.v1"


def _byte_string(c: UInt8) raises -> String:
    var b = List[UInt8]()
    b.append(c)
    return String(from_utf8=b)


def _json_escape(s: String) raises -> String:
    var out = String("")
    var bs = s.as_bytes()
    for i in range(s.byte_length()):
        var ch = bs[i]
        if ch == 0x22:
            out += String("\\\"")
        elif ch == 0x5C:
            out += String("\\\\")
        elif ch == 0x0A:
            out += String("\\n")
        elif ch == 0x0D:
            out += String("\\r")
        elif ch == 0x09:
            out += String("\\t")
        else:
            out += _byte_string(ch)
    return out^


def _json_bool(v: Bool) -> String:
    return String("true") if v else String("false")


def _float32_list_json(vals: List[Float32]) -> String:
    var out = String("[")
    for i in range(len(vals)):
        if i > 0:
            out += String(",")
        out += String(vals[i])
    out += String("]")
    return out^


def _ideogram4_flow_sigma(t: Float64, sigma_shift: Float64) -> Float32:
    return Float32(
        (sigma_shift * t) / (1.0 + (sigma_shift - 1.0) * t)
    )


def _build_ideogram4_simple_sigmas(steps: Int, sigma_shift: Float64) raises -> List[Float32]:
    if steps <= 0:
        raise Error("ideogram4: steps must be positive")
    if sigma_shift <= 0.0:
        raise Error("ideogram4: sigma_shift must be positive")
    var out = List[Float32]()
    var stride = 1000.0 / Float64(steps)
    for i in range(steps):
        var timestep_index = 1000 - Int(Float64(i) * stride)
        if timestep_index < 1:
            timestep_index = 1
        var t = Float64(timestep_index) / 1000.0
        out.append(_ideogram4_flow_sigma(t, sigma_shift))
    out.append(Float32(0.0))
    return out^


def _ideogram4_flow_percent_to_sigma(percent: Float64, sigma_shift: Float64) -> Float32:
    if percent <= 0.0:
        return Float32(1.0)
    if percent >= 1.0:
        return Float32(0.0)
    return _ideogram4_flow_sigma(1.0 - percent, sigma_shift)


def _extend_intermediate_sigmas(
    sigmas: List[Float32], steps: Int, start_at: Float32, end_at: Float32
) raises -> List[Float32]:
    # Faithful port of ComfyUI ExtendIntermediateSigmas (linear spacing), as used
    # by the KJ Ideogram-4 unconditional recipe: ExtendIntermediateSigmas(steps=2,
    # start_at_sigma=1, end_at_sigma=0.98, 'linear'). For each adjacent sigma pair
    # whose CURRENT sigma is in [end_at, start_at], insert linspace(0,1,steps+1)[1:-1]
    # interpolation points. This refines the first high-noise steps where the
    # AuraFlow trajectory commits to image-vs-safety-block.
    var out = List[Float32]()
    var n = len(sigmas)
    if n == 0:
        return out^
    # computed_spacing = linspace(0, 1, steps+1)[1:-1] with the linear interpolator
    # (identity): values k/steps for k in 1..steps-1.
    var spacing = List[Float32]()
    for k in range(1, steps):
        spacing.append(Float32(Float64(k) / Float64(steps)))
    for i in range(n - 1):
        var cur = sigmas[i]
        var nxt = sigmas[i + 1]
        out.append(cur)
        if cur >= end_at and cur <= start_at:
            for k in range(len(spacing)):
                out.append(cur + spacing[k] * (nxt - cur))
    out.append(sigmas[n - 1])
    return out^


def _write_text_file(path: String, content: String) raises:
    var fd = sys_open(path, O_CREAT | O_WRONLY | O_TRUNC, Int32(0o644))
    if fd < 0:
        raise Error(String("ideogram4_backend: cannot create ") + path)
    var n = content.byte_length()
    var buf = alloc[UInt8](n if n > 0 else 1)
    var src = content.as_bytes()
    for i in range(n):
        buf[i] = src[i]
    var wrote = sys_pwrite(fd, BytePtr(unsafe_from_address=Int(buf)), n, 0)
    buf.free()
    _ = sys_close(fd)
    if wrote != n:
        raise Error(String("ideogram4_backend: short write to ") + path)


def _save_rgb_png_with_text(
    rgb: Tensor, path: String, params_json: String, ctx: DeviceContext
) raises:
    """[1,3,H,W] SIGNED float tensor -> PNG with serenity.genparams.v1 tEXt."""
    var shape = rgb.shape()
    if len(shape) != 4 or shape[0] != 1 or shape[1] != 3:
        raise Error("ideogram4_backend: expected [1,3,H,W] rgb tensor")
    var height = shape[2]
    var width = shape[3]
    var host = rgb.to_host(ctx)
    var plane = height * width
    if len(host) != 3 * plane:
        raise Error("ideogram4_backend: rgb to_host size mismatch")
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


def _peak_vram_mib(total_vram: Int, min_free: Int) -> Float64:
    return Float64(total_vram - min_free) / 1048576.0


def _zero_host(n: Int) -> List[Float32]:
    var out = List[Float32](capacity=n)
    for _ in range(n):
        out.append(0.0)
    return out^


def _render_chat_prompt(prompt: String) -> String:
    return (
        String("<|im_start|>user\n") + prompt
        + String("<|im_end|>\n<|im_start|>assistant\n")
    )


def _build_fixed_inputs[GH_: Int, GW_: Int](ctx: DeviceContext) raises -> List[TArc]:
    """Fixed 1024 text-token window + GH_*GW_ image tokens for the bucket.

    GH_=GW_=64 reproduces the original fixed 1024x1024 layout bit-for-bit.
    """
    comptime NIMG_ = GH_ * GW_
    comptime TOTAL_ = TEXT_TOKENS + NIMG_
    var pos = List[Float32]()
    var ind = List[Float32]()
    var npos = List[Float32]()
    var nind = List[Float32]()
    for l in range(TEXT_TOKENS):
        pos.append(Float32(l))
        pos.append(Float32(l))
        pos.append(Float32(l))
        ind.append(3.0)  # LLM_TOKEN_INDICATOR
    for h in range(GH_):
        for w in range(GW_):
            var t0 = Float32(IMG_OFFSET)
            var hh = Float32(IMG_OFFSET + h)
            var ww = Float32(IMG_OFFSET + w)
            pos.append(t0)
            pos.append(hh)
            pos.append(ww)
            npos.append(t0)
            npos.append(hh)
            npos.append(ww)
            ind.append(2.0)   # OUTPUT_IMAGE_INDICATOR
            nind.append(2.0)
    var out = List[TArc]()
    out.append(TArc(Tensor.from_host(pos^, [1, TOTAL_, 3], STDtype.F32, ctx)))
    out.append(TArc(Tensor.from_host(ind^, [1, TOTAL_], STDtype.F32, ctx)))
    out.append(TArc(Tensor.from_host(npos^, [1, NIMG_, 3], STDtype.F32, ctx)))
    out.append(TArc(Tensor.from_host(nind^, [1, NIMG_], STDtype.F32, ctx)))
    return out^


struct Ideogram4RopeSet(Movable):
    var cond: Tuple[Tensor, Tensor]
    var uncond: Tuple[Tensor, Tensor]

    def __init__(
        out self, var cond: Tuple[Tensor, Tensor], var uncond: Tuple[Tensor, Tensor]
    ):
        self.cond = cond^
        self.uncond = uncond^


def _tiny_bf16(ctx: DeviceContext) raises -> Tensor:
    """[1,1,1] BF16 zero — placeholder for unbuilt per-bucket static slots."""
    var z = List[Float32]()
    z.append(0.0)
    return Tensor.from_host(z^, [1, 1, 1], STDtype.BF16, ctx)


def _empty_masks(ctx: DeviceContext) raises -> Ideogram4Masks:
    var ids = List[Int]()
    return Ideogram4Masks(_tiny_bf16(ctx), _tiny_bf16(ctx), ids^)


def _empty_ropeset(ctx: DeviceContext) raises -> Ideogram4RopeSet:
    return Ideogram4RopeSet(
        (_tiny_bf16(ctx), _tiny_bf16(ctx)), (_tiny_bf16(ctx), _tiny_bf16(ctx))
    )


struct Ideogram4Backend(GenBackend, Movable):
    var ctx: DeviceContext

    # Transformer weights are resident only during a job's denoise phase.
    var loaded: Bool
    var load_stage: Int
    var cond: List[ArcPointer[Ideogram4Weights]]
    var uncond: List[ArcPointer[Ideogram4Weights]]

    # Static per-bucket sequence helpers, safe to retain across jobs. Each of the
    # three lists holds one entry per resolution bucket (index == bucket id), built
    # lazily by _ensure_static_b on first use of that bucket and reused after.
    var static_ready: List[Bool]
    var cond_masks: List[ArcPointer[Ideogram4Masks]]
    var uncond_masks: List[ArcPointer[Ideogram4Masks]]
    var ropes: List[ArcPointer[Ideogram4RopeSet]]

    # Per-job state.
    var active: Bool
    var cancel_flag: Bool
    var bucket: Int   # one of the seven finite BUCKET_* specializations above
    var phase: Int
    var announced: Bool
    var cur: Int
    var params: JobParams
    var cfg: Float32
    var prompt_tokens: Int
    var text_features: List[TArc]
    var llm: List[TArc]
    var neg_llm: List[TArc]
    var text_zpad: List[TArc]
    var latent: List[TArc]
    var intervals: List[Float32]
    var sigma_trace: List[Float32]
    var executed_sampler: String
    var executed_scheduler: String
    var lora_target_count: Int
    var job_t0_ns: UInt
    var load_seconds: Float64
    var text_encode_seconds: Float64
    var prepare_seconds: Float64
    var denoise_seconds: Float64
    var vae_decode_seconds: Float64
    var total_vram_bytes: Int
    var min_free_bytes: Int

    def __init__(out self) raises:
        var ctx = DeviceContext()
        self.ctx = ctx
        self.loaded = False
        self.load_stage = 0
        self.cond = List[ArcPointer[Ideogram4Weights]]()
        self.uncond = List[ArcPointer[Ideogram4Weights]]()
        # Per-bucket static caches, built lazily and retained across jobs.
        self.static_ready = List[Bool](capacity=BUCKET_COUNT)
        var cms = List[ArcPointer[Ideogram4Masks]]()
        var ums = List[ArcPointer[Ideogram4Masks]]()
        var rps = List[ArcPointer[Ideogram4RopeSet]]()
        for _ in range(BUCKET_COUNT):
            self.static_ready.append(False)
            cms.append(ArcPointer(_empty_masks(ctx)))
            ums.append(ArcPointer(_empty_masks(ctx)))
            rps.append(ArcPointer(_empty_ropeset(ctx)))
        self.cond_masks = cms^
        self.uncond_masks = ums^
        self.ropes = rps^
        self.active = False
        self.cancel_flag = False
        self.bucket = BUCKET_SQUARE
        self.phase = IPHASE_IDLE
        self.announced = False
        self.cur = 0
        self.params = JobParams()
        self.cfg = Float32(7.0)
        self.prompt_tokens = 0
        self.text_features = List[TArc]()
        self.llm = List[TArc]()
        self.neg_llm = List[TArc]()
        self.text_zpad = List[TArc]()
        self.latent = List[TArc]()
        self.intervals = List[Float32]()
        self.sigma_trace = List[Float32]()
        self.executed_sampler = String("ideogram4_logitnormal_euler")
        self.executed_scheduler = String("ideogram4_logitnormal")
        self.lora_target_count = 0
        self.job_t0_ns = 0
        self.load_seconds = 0.0
        self.text_encode_seconds = 0.0
        self.prepare_seconds = 0.0
        self.denoise_seconds = 0.0
        self.vae_decode_seconds = 0.0
        self.total_vram_bytes = 0
        self.min_free_bytes = 0

    def backend_name(self) -> String:
        return String("ideogram4")

    def model_name(self) -> String:
        return String("Ideogram-4")

    def resident_model(self) -> String:
        return String("ideogram-4-fp8") if self.loaded else String("")

    def start(mut self, params: JobParams) raises:
        if self.active:
            raise Error("Ideogram4Backend.start: a job is already running")
        reject_unsupported_common_runtime_params(params, String("ideogram4"))
        reject_unsupported_reference_image_params(params, String("ideogram4"))
        reject_unsupported_inpaint_conditioning_params(params, String("ideogram4"))
        reject_unsupported_qwen_edit_conditioning_params(params, String("ideogram4"))
        reject_unsupported_conditioning_mask_params(params, String("ideogram4"))
        reject_unsupported_mask_image_params(params, String("ideogram4"))
        reject_unsupported_lanpaint_params(params, String("ideogram4"))
        # FlowEdit jobs run the pipeline's own ODE schedule — the t2i
        # sampler/scheduler admission does not apply (graph edits carry none);
        # executed_sched stays the logit-normal default for the manifest.
        var executed_sched = String("ideogram4_logitnormal")
        var executed_samp = String("ideogram4_logitnormal_euler")
        if params.edit_src_image == String(""):
            var sampler_admission = sampler_admission_for_backend(String("ideogram4"), params.sampler)
            if not sampler_admission.supported:
                raise Error(
                    String("ideogram4: unsupported sampler '") + params.sampler
                    + String("'; ") + sampler_admission.reason
                )
            var scheduler_admission = scheduler_admission_for_backend(String("ideogram4"), params.scheduler)
            if not scheduler_admission.supported:
                raise Error(
                    String("ideogram4: unsupported scheduler '") + params.scheduler
                    + String("'; ") + scheduler_admission.reason
                )
            executed_sched = scheduler_admission.executed.copy()
            executed_samp = sampler_admission.executed.copy()
        var bucket: Int
        if params.width == 1024 and params.height == 1024:
            bucket = BUCKET_SQUARE
        elif params.width == 1152 and params.height == 896:
            bucket = BUCKET_1152X896
        elif params.width == 896 and params.height == 1152:
            bucket = BUCKET_896X1152
        elif params.width == 1344 and params.height == 768:
            bucket = BUCKET_1344X768
        elif params.width == 768 and params.height == 1344:
            bucket = BUCKET_768X1344
        elif params.width == 1280 and params.height == 832:
            bucket = BUCKET_1280X832
        elif params.width == 832 and params.height == 1280:
            bucket = BUCKET_832X1280
        else:
            raise Error(
                String("ideogram4: unsupported size ") + String(params.width)
                + "x" + String(params.height)
                + " -- supported compiled sizes: 1024x1024, 1152x896, 896x1152,"
                + " 1344x768, 768x1344, 1280x832, 832x1280"
            )
        if params.edit_src_image != String("") and bucket != BUCKET_SQUARE:
            raise Error("ideogram4 FlowEdit: only 1024x1024 is supported")
        if params.negative.byte_length() > 0:
            raise Error("ideogram4: negative prompt is not supported in this bounded slice")
        if len(params.loras) > 1:
            raise Error(
                "ideogram4: this runtime currently supports one compatible"
                " Ideogram PEFT/Serenity adapter at a time"
            )
        if params.edit_src_image != String("") and len(params.loras) > 0:
            raise Error("ideogram4 FlowEdit: LoRA is not supported yet")
        if params.init_image.byte_length() > 0:
            raise Error("ideogram4: img2img/init image is not supported in this bounded slice")
        if params.creativity != 0.5:
            raise Error("ideogram4: creativity/denoise control is not supported in this bounded txt2img slice")
        if params.variation_strength > 0.0:
            raise Error("ideogram4: variation noise is not supported in this bounded slice")
        if params.cfg <= 0.0:
            raise Error("ideogram4: cfg must be positive")
        if params.cfg_override >= 0.0:
            if executed_sched != "ideogram4_simple_flowmatch":
                raise Error("ideogram4: cfg_override is supported only with the simple AuraFlow scheduler")
            if params.cfg_override == 0.0:
                raise Error("ideogram4: cfg_override must be positive when set")
            if params.cfg_override_start_percent > params.cfg_override_end_percent:
                raise Error("ideogram4: cfg_override_start_percent must be <= cfg_override_end_percent")

        # A previous job may have failed mid-denoise; never try to encode text
        # while the cond/uncond transformers are still resident.
        self._free_transformers()
        self.params = params.copy()
        self.bucket = bucket
        self.cfg = Float32(params.cfg)
        self.executed_sampler = executed_samp.copy()
        self.executed_scheduler = executed_sched.copy()
        self.lora_target_count = 0
        self.active = True
        self.cancel_flag = False
        self.phase = IPHASE_ENCODE
        self.announced = False
        self.cur = 0
        self.prompt_tokens = 0
        self.text_features = List[TArc]()
        self.llm = List[TArc]()
        self.neg_llm = List[TArc]()
        self.text_zpad = List[TArc]()
        self.latent = List[TArc]()
        self.intervals = List[Float32]()
        self.sigma_trace = List[Float32]()
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
        var before = cu_mem_get_info()
        self.ctx.synchronize()
        cu_mempool_trim_current(0)
        self.ctx.synchronize()
        var after = cu_mem_get_info()
        print("[ideogram4] between-jobs trim: used",
              before.used_bytes() // (1024 * 1024), "->",
              after.used_bytes() // (1024 * 1024), "MiB (reclaimed",
              (before.used_bytes() - after.used_bytes()) // (1024 * 1024), "MiB)")

    def _free_transformers(mut self):
        self.cond = List[ArcPointer[Ideogram4Weights]]()
        self.uncond = List[ArcPointer[Ideogram4Weights]]()
        self.loaded = False
        self.load_stage = 0

    def _ensure_static_b[GH_: Int, GW_: Int](mut self, idx: Int) raises:
        if self.static_ready[idx]:
            return
        print("[ideogram4] building fixed masks and MRoPE for grid",
              GH_, "x", GW_, "(", 16 * GW_, "x", 16 * GH_, "px )")
        var inp = _build_fixed_inputs[GH_, GW_](self.ctx)
        # mrope_section is the per-axis HEAD_DIM band split (24+20+20 over the
        # head_dim halving), independent of resolution — unchanged for all buckets.
        var sec = [24, 20, 20]
        var cs = build_ideogram4_mrope(
            inp[0][], HEAD_DIM, sec, Float32(5000000.0), self.ctx, STDtype.BF16
        )
        var ncs = build_ideogram4_mrope(
            inp[2][], HEAD_DIM, sec, Float32(5000000.0), self.ctx, STDtype.BF16
        )
        self.ropes[idx] = ArcPointer(Ideogram4RopeSet(cs^, ncs^))
        self.cond_masks[idx] = ArcPointer(ideogram4_build_masks(inp[1][], self.ctx))
        self.uncond_masks[idx] = ArcPointer(ideogram4_build_masks(inp[3][], self.ctx))
        self.static_ready[idx] = True

    def _ensure_static(mut self) raises:
        if self.bucket == BUCKET_1152X896:
            self._ensure_static_b[GH_1152X896, GW_1152X896](BUCKET_1152X896)
        elif self.bucket == BUCKET_896X1152:
            self._ensure_static_b[GH_896X1152, GW_896X1152](BUCKET_896X1152)
        elif self.bucket == BUCKET_1344X768:
            self._ensure_static_b[GH_1344X768, GW_1344X768](BUCKET_1344X768)
        elif self.bucket == BUCKET_768X1344:
            self._ensure_static_b[GH_768X1344, GW_768X1344](BUCKET_768X1344)
        elif self.bucket == BUCKET_1280X832:
            self._ensure_static_b[GH_1280X832, GW_1280X832](BUCKET_1280X832)
        elif self.bucket == BUCKET_832X1280:
            self._ensure_static_b[GH_832X1280, GW_832X1280](BUCKET_832X1280)
        else:
            self._ensure_static_b[GH, GW](BUCKET_SQUARE)

    def _encode(mut self) raises:
        var tok = Qwen3Tokenizer(String(TOK_JSON))
        var ids = tok.encode(_render_chat_prompt(self.params.prompt))
        self.prompt_tokens = len(ids)
        if self.prompt_tokens > TEXT_TOKENS:
            raise Error(
                String("ideogram4: prompt tokenized to ") + String(self.prompt_tokens)
                + " tokens; maximum supported by this bounded fixed-shape path is "
                + String(TEXT_TOKENS)
            )
        for _ in range(TEXT_TOKENS - self.prompt_tokens):
            ids.append(PAD_ID)
        print("[ideogram4] encoding prompt tokens:", self.prompt_tokens, "/", TEXT_TOKENS)
        # Streamed TE (16GB-fit): layer-by-layer from disk, peak ~1.9GB vs the
        # ~15.1GB dequanted-resident load_ideogram_qwen3vl, which OOMs the
        # seq-1024 forward on a 16GB card (measured 2026-07-14, task #25 gate 2).
        # Parity-gated vs the resident encoder (ideogram4_te_streamed_parity).
        var ids_list = List[List[Int]]()
        ids_list.append(ids^)
        var feats_list = encode_ideogram_taps_streamed(
            String(TE), ids_list, self.ctx
        )
        # Zero the pad feature rows [prompt_tokens, TEXT_TOKENS) — the
        # ideogram4_prepare._encode_padded convention the verified singletrunk
        # CLI uses. Unzeroed PAD-token features (e.g. 1012 of 1024 rows for a
        # short prompt) flood the DiT llm conditioning and produce incoherent
        # output (e2e-observed 2026-07-15).
        if self.prompt_tokens < TEXT_TOKENS:
            var mask_host = List[Float32]()
            for j in range(TEXT_TOKENS):
                if j < self.prompt_tokens:
                    mask_host.append(Float32(1.0))
                else:
                    mask_host.append(Float32(0.0))
            var mask_f32 = Tensor.from_host(
                mask_host^, [1, TEXT_TOKENS, 1], STDtype.F32, self.ctx
            )
            var mask = cast_tensor(mask_f32, STDtype.BF16, self.ctx)
            var zeroed = cast_tensor(
                mul(feats_list[0][], mask, self.ctx), STDtype.BF16, self.ctx
            )
            self.text_features = List[TArc]()
            self.text_features.append(TArc(zeroed^))
        else:
            self.text_features = feats_list^

    def _single_trunk(self) raises -> Bool:
        """True when the card cannot hold both 8.7GB fp8 trunks (17.4GB > any
        16GB device): run the uncond pass through the ONE resident cond
        transformer with zeroed llm features — the accepted 16GB-fit precedent
        from Ideogram4SampleResident (trainer inline sampling) and
        pipeline/ideogram4_t2i_singletrunk.mojo. neg_llm is already image-only
        zeros in this backend, so sharing the cond weights IS the single-trunk
        semantics. Dual-trunk (the KJ recipe's true uncond weights) is kept on
        >=20GB cards."""
        return self.total_vram_bytes < 20 * 1024 * 1024 * 1024

    def _load_one(mut self) raises -> Bool:
        if self.loaded:
            return True
        if self.load_stage == 0:
            print("[ideogram4] loading conditional fp8 transformer")
            self.cond = List[ArcPointer[Ideogram4Weights]]()
            var cond = Ideogram4Weights.load(
                ShardedSafeTensors.open(String(COND)), self.ctx
            )
            if len(self.params.loras) == 1:
                print(
                    "[ideogram4] loading additive conditional LoRA:",
                    self.params.loras[0].name,
                    "weight",
                    self.params.loras[0].weight,
                )
                self.lora_target_count = cond.load_lora(
                    self.params.loras[0].name,
                    self.ctx,
                    Float32(self.params.loras[0].weight),
                )
                print(
                    "[ideogram4] loaded",
                    self.lora_target_count,
                    "compatible projection factors",
                )
            self.cond.append(ArcPointer(cond^))
            self.load_stage = 1
            return False
        if self.load_stage == 1:
            if self._single_trunk():
                print("[ideogram4] single-trunk uncond (<20GB card): sharing cond weights")
                self.uncond = List[ArcPointer[Ideogram4Weights]]()
                self.uncond.append(self.cond[0])
            else:
                print("[ideogram4] loading unconditional fp8 transformer")
                self.uncond = List[ArcPointer[Ideogram4Weights]]()
                self.uncond.append(ArcPointer(
                    Ideogram4Weights.load(ShardedSafeTensors.open(String(UNCOND)), self.ctx)
                ))
            self.loaded = True
            self.load_stage = 2
            return True
        return True

    def _prepare_job_b[GH_: Int, GW_: Int](mut self) raises:
        comptime NIMG_ = GH_ * GW_
        if len(self.text_features) == 0:
            raise Error("ideogram4: missing text features")
        self._ensure_static()

        var zllm = _zero_host(NIMG_ * LLM_DIM)
        var img_zeros = Tensor.from_host(zllm^, [1, NIMG_, LLM_DIM], STDtype.BF16, self.ctx)
        var llm_full = concat(1, self.ctx, self.text_features[0][], img_zeros)
        self.llm = List[TArc]()
        self.llm.append(TArc(llm_full^))
        self.text_features = List[TArc]()

        var nllm = _zero_host(NIMG_ * LLM_DIM)
        self.neg_llm = List[TArc]()
        self.neg_llm.append(TArc(
            Tensor.from_host(nllm^, [1, NIMG_, LLM_DIM], STDtype.BF16, self.ctx)
        ))

        var zpad = _zero_host(TEXT_TOKENS * LATENT_DIM)
        self.text_zpad = List[TArc]()
        self.text_zpad.append(TArc(
            Tensor.from_host(zpad^, [1, TEXT_TOKENS, LATENT_DIM], STDtype.F32, self.ctx)
        ))

        self.latent = List[TArc]()
        self.latent.append(TArc(
            randn([1, NIMG_, LATENT_DIM], UInt64(self.params.seed), STDtype.F32, self.ctx)
        ))

        self.sigma_trace = List[Float32]()
        if self.executed_scheduler == "ideogram4_simple_flowmatch":
            self.intervals = List[Float32]()
            self.sigma_trace = _build_ideogram4_simple_sigmas(
                self.params.steps, self.params.sigma_shift
            )
            # KJ Ideogram-4 unconditional recipe inserts ExtendIntermediateSigmas
            # between the BasicScheduler and the sampler. Replicate it on the KJ
            # path (cfg_override set) so the early high-noise trajectory matches
            # ComfyUI; bump the loop's step count to the extended length.
            if self.params.cfg_override >= 0.0:
                self.sigma_trace = _extend_intermediate_sigmas(
                    self.sigma_trace, 2, Float32(1.0), Float32(0.98)
                )
                self.params.steps = len(self.sigma_trace) - 1
        else:
            self.intervals = make_step_intervals(self.params.steps)
            # logit-normal schedule mean is resolution-aware (num_px term); the 16px
            # patch makes pixels = 16*GW_ by 16*GH_. The square bucket gives exactly
            # ideogram4_schedule_mean(1024,1024,0.0) as before.
            var mean = ideogram4_schedule_mean(16 * GH_, 16 * GW_, 0.0)
            # preset logit-normal std: V4_DEFAULT_20 / V4_TURBO_12 use 1.75;
            # only the 48-step V4_QUALITY preset uses 1.5. Using 1.5 on a <=20-step
            # run is the wrong preset and softens output (inference-flame scheduler.rs).
            var lstd = Float64(1.75) if self.params.steps <= 20 else Float64(1.5)
            for i in range(len(self.intervals)):
                self.sigma_trace.append(
                    ideogram4_logitnormal(Float64(self.intervals[i]), mean, lstd)
                )
        print(
            "[ideogram4] job", self.params.job_id, ":", self.params.steps,
            "steps, cfg", self.cfg, "scheduler", self.executed_scheduler,
            "shift", self.params.sigma_shift, "seed", self.params.seed,
            "size", self.params.width, "x", self.params.height,
        )

    def _prepare_job(mut self) raises:
        if self.bucket == BUCKET_1152X896:
            self._prepare_job_b[GH_1152X896, GW_1152X896]()
        elif self.bucket == BUCKET_896X1152:
            self._prepare_job_b[GH_896X1152, GW_896X1152]()
        elif self.bucket == BUCKET_1344X768:
            self._prepare_job_b[GH_1344X768, GW_1344X768]()
        elif self.bucket == BUCKET_768X1344:
            self._prepare_job_b[GH_768X1344, GW_768X1344]()
        elif self.bucket == BUCKET_1280X832:
            self._prepare_job_b[GH_1280X832, GW_1280X832]()
        elif self.bucket == BUCKET_832X1280:
            self._prepare_job_b[GH_832X1280, GW_832X1280]()
        else:
            self._prepare_job_b[GH, GW]()

    def _cfg_for_sigma(self, sigma: Float32) -> Float32:
        if self.params.cfg_override < 0.0:
            return self.cfg
        var sigma_hi = _ideogram4_flow_percent_to_sigma(
            self.params.cfg_override_start_percent, self.params.sigma_shift
        )
        var sigma_lo = _ideogram4_flow_percent_to_sigma(
            self.params.cfg_override_end_percent, self.params.sigma_shift
        )
        if sigma >= sigma_lo and sigma <= sigma_hi:
            return Float32(self.params.cfg_override)
        return self.cfg

    def _denoise_one_b[GH_: Int, GW_: Int](mut self, idx: Int) raises:
        comptime NIMG_ = GH_ * GW_
        comptime TOTAL_ = TEXT_TOKENS + NIMG_
        var t_val: Float32
        var s_val: Float32
        # model_t is what the DiT receives. Ideogram4 model-time is INVERTED
        # (tau: 0 = noise -> 1 = image; see pipeline/ideogram4_flowedit.mojo and
        # the verified t2i CLIs, whose final step feeds t ~= 1). The Comfy
        # simple branch's sigma_trace descends 1 -> 0 in NOISE convention, so
        # the model input must be tau = 1 - sigma; feeding sigma directly told
        # the model "nearly clean" at pure noise and produced structureless
        # texture (e2e-observed 2026-07-15). The Euler coefficient stays
        # (s_val - t_val): in sigma space it equals d-tau exactly.
        var model_t: Float32
        if self.executed_scheduler == "ideogram4_simple_flowmatch":
            # trace descends in NOISE sigma; z currently sits at trace[cur].
            # Evaluate the model at the CURRENT point (the CLI convention:
            # its first step feeds tau ~= 0 = pure noise): tau = 1 - sigma_cur.
            s_val = self.sigma_trace[self.cur]
            t_val = self.sigma_trace[self.cur + 1]
            model_t = Float32(1.0) - s_val
        else:
            # logit-normal trace is already in model-time; t_val IS the current
            # tau (matches the CLI: t = lognorm(si[step+1]) = where z is now).
            var step_idx = self.params.steps - 1 - self.cur
            t_val = self.sigma_trace[step_idx + 1]
            s_val = self.sigma_trace[step_idx]
            model_t = t_val
        var sigma_for_cfg = s_val
        var step_cfg = self._cfg_for_sigma(sigma_for_cfg)
        # V4 asymmetric polish schedule (MJ-1051): unless the user set an explicit
        # cfg_override window, drop guidance to 3.0 on the final low-noise steps,
        # matching the Rust presets (scheduler.rs build_gw: 48->(3.0,3), 20->(3.0,2),
        # 12->(3.0,1)) and pipeline/ideogram4_generate.mojo:116. steps_left counts
        # remaining Euler steps INCLUDING this one (cur is 0-based ascending in
        # both scheduler branches).
        if self.params.cfg_override < 0.0:
            var polish = 2
            if self.params.steps >= 48:
                polish = 3
            elif self.params.steps <= 12:
                polish = 1
            var steps_left = self.params.steps - self.cur
            if steps_left <= polish:
                step_cfg = Float32(3.0)
        var t = Tensor.from_host([model_t], [1], STDtype.F32, self.ctx)
        var pos_z = cast_tensor(
            concat(1, self.ctx, self.text_zpad[0][], self.latent[0][]),
            STDtype.BF16,
            self.ctx,
        )
        var cout = ideogram4_forward_r_masked[TOTAL_, TEXT_TOKENS](
            self.cond[0][], pos_z, self.llm[0][], t, self.cond_masks[idx][],
            self.ropes[idx][].cond[0], self.ropes[idx][].cond[1],
            self.prompt_tokens,
            LAYERS, HEADS, HEAD_DIM, HIDDEN, self.ctx,
        )
        var pos_v = slice(cout, 1, TEXT_TOKENS, NIMG_, self.ctx)
        var t2 = Tensor.from_host([model_t], [1], STDtype.F32, self.ctx)
        var z_bf = cast_tensor(self.latent[0][], STDtype.BF16, self.ctx)
        var nout = ideogram4_forward_r[NIMG_](
            self.uncond[0][], z_bf, self.neg_llm[0][], t2, self.uncond_masks[idx][],
            self.ropes[idx][].uncond[0], self.ropes[idx][].uncond[1],
            LAYERS, HEADS, HEAD_DIM, HIDDEN, self.ctx,
        )
        var v = add(
            mul_scalar(pos_v, step_cfg, self.ctx),
            mul_scalar(nout, Float32(1.0) - step_cfg, self.ctx),
            self.ctx,
        )
        var z_new = add(
            self.latent[0][],
            mul_scalar(v, s_val - t_val, self.ctx),
            self.ctx,
        )
        self.latent = List[TArc]()
        self.latent.append(TArc(z_new^))
        print("  [ideogram4] step", self.cur, "t", t_val, "s", s_val, "cfg", step_cfg)

    def _denoise_one(mut self) raises:
        if self.bucket == BUCKET_1152X896:
            self._denoise_one_b[GH_1152X896, GW_1152X896](BUCKET_1152X896)
        elif self.bucket == BUCKET_896X1152:
            self._denoise_one_b[GH_896X1152, GW_896X1152](BUCKET_896X1152)
        elif self.bucket == BUCKET_1344X768:
            self._denoise_one_b[GH_1344X768, GW_1344X768](BUCKET_1344X768)
        elif self.bucket == BUCKET_768X1344:
            self._denoise_one_b[GH_768X1344, GW_768X1344](BUCKET_768X1344)
        elif self.bucket == BUCKET_1280X832:
            self._denoise_one_b[GH_1280X832, GW_1280X832](BUCKET_1280X832)
        elif self.bucket == BUCKET_832X1280:
            self._denoise_one_b[GH_832X1280, GW_832X1280](BUCKET_832X1280)
        else:
            self._denoise_one_b[GH, GW](BUCKET_SQUARE)

    def _record_vram(mut self) raises:
        var mem = cu_mem_get_info()
        if self.total_vram_bytes == 0:
            self.total_vram_bytes = mem.total_bytes
        if self.min_free_bytes == 0 or mem.free_bytes < self.min_free_bytes:
            self.min_free_bytes = mem.free_bytes

    def _write_result_manifest(mut self, png_path: String) raises -> String:
        self._record_vram()
        var manifest_path = png_path + String(".ideogram4_daemon_result.json")
        var denoise_per_step = Float64(0.0)
        if self.params.steps > 0:
            denoise_per_step = self.denoise_seconds / Float64(self.params.steps)
        var total_wall_seconds = Float64(perf_counter_ns() - self.job_t0_ns) / 1.0e9
        var peak_vram_mib = Float64(0.0)
        if self.total_vram_bytes > 0 and self.min_free_bytes > 0:
            peak_vram_mib = _peak_vram_mib(self.total_vram_bytes, self.min_free_bytes)
        var sampler_algorithm = String("ideogram4_logitnormal_euler")
        var schedule_source = String("ideogram4_logitnormal")
        var schedule_extra = String('"std":1.5')
        if self.executed_scheduler == "ideogram4_simple_flowmatch":
            sampler_algorithm = String("ideogram4_simple_flowmatch_euler")
            schedule_source = String("ideogram4_comfy_simple_aura_flow")
            schedule_extra = String('"sigma_shift":') + String(self.params.sigma_shift)

        var content = String("{\n")
        content += String('  "schema":"serenity.ideogram4.daemon_result.v1",\n')
        content += String('  "backend":"ideogram4_daemon",\n')
        content += String('  "model":"ideogram-4-fp8",\n')
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
        content += String('    "cfg_override":') + String(self.params.cfg_override) + String(",\n")
        content += String('    "cfg_override_start_percent":') + String(self.params.cfg_override_start_percent) + String(",\n")
        content += String('    "cfg_override_end_percent":') + String(self.params.cfg_override_end_percent) + String(",\n")
        content += String('    "sigma_shift":') + String(self.params.sigma_shift) + String(",\n")
        content += String('    "sampler_registry_backend":"ideogram4",\n')
        content += String('    "requested_sampler":"') + _json_escape(self.params.sampler) + String('",\n')
        content += String('    "requested_scheduler":"') + _json_escape(self.params.scheduler) + String('",\n')
        content += String('    "executed_sampler":"') + _json_escape(self.executed_sampler) + String('",\n')
        content += String('    "executed_scheduler":"') + _json_escape(self.executed_scheduler) + String('",\n')
        content += String('    "sigma_trace":') + _float32_list_json(self.sigma_trace) + String(",\n")
        content += String('    "sampler_trace":{"algorithm":"') + _json_escape(sampler_algorithm) + String('","schedule_source":"') + _json_escape(schedule_source) + String('",') + schedule_extra + String(',"cfg_override":') + String(self.params.cfg_override) + String(',"fixed_text_window_tokens":') + String(TEXT_TOKENS) + String("},\n")
        content += String('    "prompt_tokens":') + String(self.prompt_tokens) + String(",\n")
        content += String('    "text_window_tokens":') + String(TEXT_TOKENS) + String(",\n")
        content += String('    "text_padding_policy":"pad-token features are zeroed and padded text is excluded from conditional attention",\n')
        content += String('    "image_index":') + String(self.params.image_index) + String(",\n")
        content += String('    "image_count":') + String(self.params.image_count) + String(",\n")
        content += String('    "variation_applied":false,\n')
        content += String('    "lora_count":') + String(len(self.params.loras)) + String(",\n")
        if len(self.params.loras) == 1:
            content += String('    "loaded_lora":"') + _json_escape(
                self.params.loras[0].name
            ) + String('",\n')
            content += String('    "loaded_lora_weight":') + String(
                self.params.loras[0].weight
            ) + String(",\n")
            content += String('    "lora_target_count":') + String(
                self.lora_target_count
            ) + String(",\n")
        else:
            content += String('    "loaded_lora":"",\n')
            content += String('    "loaded_lora_weight":0,\n')
            content += String('    "lora_target_count":0,\n')
        content += String('    "dtype":"fp8_transformer_bf16_activations_f32_latent"\n')
        content += String("  },\n")
        content += String('  "mojo":{\n')
        content += String('    "load_seconds":') + String(self.load_seconds) + String(",\n")
        content += String('    "text_encode_seconds":') + String(self.text_encode_seconds) + String(",\n")
        content += String('    "prepare_seconds":') + String(self.prepare_seconds) + String(",\n")
        content += String('    "denoise_seconds":') + String(self.denoise_seconds) + String(",\n")
        content += String('    "denoise_seconds_per_step":') + String(denoise_per_step) + String(",\n")
        content += String('    "vae_decode_seconds":') + String(self.vae_decode_seconds) + String(",\n")
        content += String('    "total_wall_seconds":') + String(total_wall_seconds) + String(",\n")
        content += String('    "peak_vram_mib":') + String(peak_vram_mib) + String(",\n")
        content += String('    "transformer_resident_across_jobs":false,\n')
        content += String('    "transformer_resident_across_denoise_steps":true,\n')
        content += String('    "artifact_paths":["') + _json_escape(png_path) + String('","') + _json_escape(manifest_path) + String('"]\n')
        content += String("  },\n")
        content += String('  "output_png":"') + _json_escape(png_path) + String('",\n')
        content += String('  "note":"Bounded daemon product-path result. The text encoder forces transformer unload/reload around each job on 24GB GPUs; sampler and speed parity remain unaccepted until paired runtime evidence exists."\n')
        content += String("}\n")
        _write_text_file(manifest_path, content)
        return manifest_path

    def _decode_and_save_b[GH_: Int, GW_: Int](mut self) raises -> String:
        comptime VAE_H_ = 2 * GH_
        comptime VAE_W_ = 2 * GW_
        var png_path = self.params.out_dir + "/" + self.params.job_id + ".png"
        self.llm = List[TArc]()
        self.neg_llm = List[TArc]()
        self.text_zpad = List[TArc]()
        # Decode needs VAE headroom; the DiT is no longer needed after denoise.
        self._free_transformers()
        self.ctx.synchronize()
        try:
            cu_mempool_trim_current(0)
        except:
            pass
        self.ctx.synchronize()

        var ln = ShardedSafeTensors.open(String(LATENT_NORM))
        var scale = reshape(Tensor.from_view(ln.tensor_view("latent_scale"), self.ctx), [1, 1, LATENT_DIM], self.ctx)
        var shift = reshape(Tensor.from_view(ln.tensor_view("latent_shift"), self.ctx), [1, 1, LATENT_DIM], self.ctx)
        var zd = add(mul(self.latent[0][], scale, self.ctx), shift, self.ctx)
        var z6 = reshape(zd, [1, GH_, GW_, 2, 2, 32], self.ctx)
        var zp = permute(z6, [0, 5, 1, 3, 2, 4], self.ctx)
        var latent = reshape(zp, [1, 32, VAE_H_, VAE_W_], self.ctx)
        var latent_bf = cast_tensor(latent, STDtype.BF16, self.ctx)
        self.latent = List[TArc]()
        # Release whole-frame unpatch/denorm temporaries before VAE load. The
        # tiled decoder only needs the BF16 latent crop source.
        zd = _tiny_bf16(self.ctx)
        z6 = _tiny_bf16(self.ctx)
        zp = _tiny_bf16(self.ctx)
        latent = _tiny_bf16(self.ctx)
        self.ctx.synchronize()
        try:
            cu_mempool_trim_current(0)
        except:
            pass
        self.ctx.synchronize()
        # Prefer whole-image decode when VRAM allows; tiled degrades output (MJ-1054).
        var mem = cu_mem_get_info()
        var free_gib = Float64(mem.free_bytes) / 1073741824.0
        if mem.free_bytes > WHOLE_DECODE_MIN_FREE_BYTES:
            print("[ideogram4] WHOLE-image decode (free=", free_gib,
                  "GiB) — tiled measured to degrade output (MJ-1054)")
            var dec = load_ideogram4_vae_decoder[VAE_H_, VAE_W_](String(VAE), self.ctx)
            var wimg = dec.decode(latent_bf, self.ctx)
            _save_rgb_png_with_text(wimg, png_path, self.params.params_json, self.ctx)
            return png_path
        print("[ideogram4] tiled VAE decode FALLBACK — free=", free_gib,
              "GiB below whole-image threshold; tiled degrades output (MJ-1054)")
        var img = ideogram4_tiled_decode[VAE_H_, VAE_W_](latent_bf, String(VAE), self.ctx)
        _save_rgb_png_with_text(img, png_path, self.params.params_json, self.ctx)
        return png_path

    def _decode_and_save(mut self) raises -> String:
        if self.bucket == BUCKET_1152X896:
            return self._decode_and_save_b[GH_1152X896, GW_1152X896]()
        elif self.bucket == BUCKET_896X1152:
            return self._decode_and_save_b[GH_896X1152, GW_896X1152]()
        elif self.bucket == BUCKET_1344X768:
            return self._decode_and_save_b[GH_1344X768, GW_1344X768]()
        elif self.bucket == BUCKET_768X1344:
            return self._decode_and_save_b[GH_768X1344, GW_768X1344]()
        elif self.bucket == BUCKET_1280X832:
            return self._decode_and_save_b[GH_1280X832, GW_1280X832]()
        elif self.bucket == BUCKET_832X1280:
            return self._decode_and_save_b[GH_832X1280, GW_832X1280]()
        else:
            return self._decode_and_save_b[GH, GW]()

    def _clear_job(mut self):
        self._free_transformers()
        self.active = False
        self.bucket = BUCKET_SQUARE
        self.phase = IPHASE_IDLE
        self.announced = False
        self.cancel_flag = False
        self.cur = 0
        self.prompt_tokens = 0
        self.text_features = List[TArc]()
        self.llm = List[TArc]()
        self.neg_llm = List[TArc]()
        self.text_zpad = List[TArc]()
        self.latent = List[TArc]()
        self.intervals = List[Float32]()
        self.sigma_trace = List[Float32]()
        self.executed_sampler = String("ideogram4_logitnormal_euler")
        self.executed_scheduler = String("ideogram4_logitnormal")
        self.job_t0_ns = 0
        self.load_seconds = 0.0
        self.text_encode_seconds = 0.0
        self.prepare_seconds = 0.0
        self.denoise_seconds = 0.0
        self.vae_decode_seconds = 0.0
        self.total_vram_bytes = 0
        self.min_free_bytes = 0

    def _step_flowedit(mut self) raises -> StepResult:
        """FlowEdit worker mode (Phase C3): training-free instruction edit at the
        pipeline's verified 1024² geometry (streamed TE + single cond trunk).
        One long blocking tick — ports pipeline/ideogram4_flowedit.main via its
        importable helpers; both prompts MUST be structured JSON captions
        (admission enforces this pre-queue)."""
        var r = StepResult()
        r.total = self.params.steps
        var jid = self.params.job_id
        if self.params.width != I4FE_WIDTH or self.params.height != I4FE_HEIGHT:
            raise Error(
                String("ideogram4 FlowEdit: only ") + String(I4FE_WIDTH) + "x"
                + String(I4FE_HEIGHT) + " is compiled (requested "
                + String(self.params.width) + "x" + String(self.params.height) + ")"
            )

        # ── 1) prompts -> temp caption files -> streamed-TE pair encode. ──
        var src_cap = String("/tmp/serenity_i4edit_") + jid + String(".src.json")
        var tgt_cap = String("/tmp/serenity_i4edit_") + jid + String(".tgt.json")
        write_text_file(src_cap, self.params.edit_src_prompt)
        write_text_file(tgt_cap, self.params.prompt)
        var texts = i4fe_encode_prompt_pair(src_cap, tgt_cap, self.ctx)
        cu_mempool_trim_current(0)
        if self.cancel_flag:
            self._clear_job()
            r.cancelled = True
            return r^

        # ── 2) source PNG -> 1024² staged -> normalized latent tokens Z0_src
        #    (ports _encode_source_tokens from an in-memory tensor). ──
        var img = decode_image_any(self.params.edit_src_image)
        var resized = resize_bilinear(img, I4FE_WIDTH, I4FE_HEIGHT)
        var host = image_to_signed_nchw(resized)
        var image_t = Tensor.from_host(
            host, [1, 3, I4FE_HEIGHT, I4FE_WIDTH], STDtype.F32, self.ctx
        )
        var img_bf = cast_tensor(image_t, STDtype.BF16, self.ctx)
        var ln = ShardedSafeTensors.open(String(I4FE_LATENT_NORM))
        var l_shift = Tensor.from_view(ln.tensor_view("latent_shift"), self.ctx)
        var l_scale = Tensor.from_view(ln.tensor_view("latent_scale"), self.ctx)
        var venc = load_ideogram4_vae_encoder[I4FE_LH, I4FE_LW](String(I4FE_VAE), self.ctx)
        var z_grid = encode_ideogram4_latents[I4FE_LH, I4FE_LW](
            venc, img_bf, l_shift, l_scale, self.ctx
        )
        var z_hwc = permute(z_grid, [0, 2, 3, 1], self.ctx)
        var z0_src = reshape(z_hwc, [1, I4FE_NIMG, 128], self.ctx)
        self.ctx.synchronize()
        cu_mempool_trim_current(0)
        if self.cancel_flag:
            self._clear_job()
            r.cancelled = True
            return r^

        # ── 3) FlowEdit ODE (loads the single cond trunk inside) + decode. ──
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
        var png_path = self.params.out_dir + "/" + jid + ".png"
        var z = i4fe_denoise(
            z0_src, texts[0][], texts[1][],
            steps, n_max, n_min, src_cfg, Float32(self.params.cfg),
            UInt64(self.params.seed),
            self.params.edit_auto_mask, Float32(self.params.edit_mask_q),
            self.params.edit_mask_dilate, self.params.edit_mask_warmup,
            png_path, self.ctx,
        )
        # _decode_and_report SAVES the PNG, then computes a diagnostic
        # source-vs-output MAD/PSNR by re-opening src as a STAGED safetensors —
        # our source is a plain PNG, so that report step raises after the save.
        # Tolerate exactly that: if the PNG landed, the job succeeded.
        try:
            i4fe_decode(z, self.params.edit_src_image, png_path, self.ctx)
        except e:
            var probe = sys_open(png_path, O_RDONLY, 0)
            if probe < 0:
                raise Error(String(e))
            _ = sys_close(probe)
            print("[ideogram4-edit] WARN: skipped src-vs-out report (PNG source):", String(e))

        self.active = False
        self._clear_job()
        r.step = self.params.steps
        r.total = self.params.steps
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
        if self.cancel_flag:
            r.step = self.cur
            self._clear_job()
            r.cancelled = True
            return r^
        try:
            if self.params.edit_src_image != String(""):
                return self._step_flowedit()
            if self.phase == IPHASE_ENCODE:
                if not self.announced:
                    self.announced = True
                    r.step = 0
                    r.phase = String("encoding")
                    return r^
                var encode_t0 = perf_counter_ns()
                self._encode()
                self.text_encode_seconds = Float64(perf_counter_ns() - encode_t0) / 1.0e9
                # CRITICAL (24GB fit): _encode frees the ~8GB Qwen encoder by scope, but the
                # CUDA mempool CACHES it. Trim NOW so the 17.4GB cond+uncond fp8 transformers
                # have room — without this, cached-encoder(8) + transformers(17.4) = ~25.6GB
                # OOMs on the 2nd transformer (measured: stall at ~22GB/0% util on a 24GB GPU).
                cu_mempool_trim_current(0)
                self._record_vram()
                self.announced = False
                self.phase = IPHASE_LOAD
                r.step = 0
                return r^
            if self.phase == IPHASE_LOAD:
                var load_t0 = perf_counter_ns()
                if self._load_one():
                    self.phase = IPHASE_PREPARE
                self.load_seconds += Float64(perf_counter_ns() - load_t0) / 1.0e9
                self._record_vram()
                r.step = 0
                r.phase = String("loading")
                return r^
            if self.phase == IPHASE_PREPARE:
                if not self.announced:
                    self.announced = True
                    r.step = 0
                    r.phase = String("preparing")
                    return r^
                var prep_t0 = perf_counter_ns()
                self._prepare_job()
                self.prepare_seconds = Float64(perf_counter_ns() - prep_t0) / 1.0e9
                self._record_vram()
                self.announced = False
                self.phase = IPHASE_DENOISE
                r.step = 0
                return r^
            if self.phase == IPHASE_DENOISE:
                var denoise_t0 = perf_counter_ns()
                self._denoise_one()
                self.denoise_seconds += Float64(perf_counter_ns() - denoise_t0) / 1.0e9
                self._record_vram()
                self.cur += 1
                r.step = self.cur
                if self.cur >= self.params.steps:
                    self.phase = IPHASE_DECODE
                return r^
            if not self.announced:
                self.announced = True
                r.step = self.params.steps
                r.phase = String("decoding")
                return r^
            var decode_t0 = perf_counter_ns()
            var path = self._decode_and_save()
            self.vae_decode_seconds = Float64(perf_counter_ns() - decode_t0) / 1.0e9
            self._record_vram()
            var manifest = self._write_result_manifest(path)
            print("[ideogram4][manifest] saved:", manifest)
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
