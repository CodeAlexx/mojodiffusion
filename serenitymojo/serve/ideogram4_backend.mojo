# serenitymojo.serve.ideogram4_backend -- bounded native Ideogram-4 GenBackend.
#
# This is a real Mojo backend path, not a subprocess wrapper. It reuses the
# native Ideogram-4 components:
#   Qwen3-VL 13-tap text encode -> fp8 cond/uncond DiT Euler denoise on either
#   the Ideogram logit-normal schedule or the Comfy simple AuraFlow schedule ->
#   latent denorm -> Ideogram VAE decode.
#
# Current product limits are intentionally narrow and fail-loud:
#   * txt2img only, 1024x1024 only
#   * no negative prompt, LoRA, init image, variation, or non-Ideogram schedulers
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
    BytePtr, sys_open, sys_close, sys_pwrite, O_WRONLY, O_CREAT, O_TRUNC,
)
from serenitymojo.io.sharded import ShardedSafeTensors
from image.buffer import Image
from image.png import encode_png_with_text
from serenitymojo.image.png import _quantize, ValueRange
from serenitymojo.models.dit.ideogram4_mrope import build_ideogram4_mrope
from serenitymojo.models.dit.ideogram4_resident import (
    Ideogram4Weights, Ideogram4Masks, ideogram4_forward_r, ideogram4_build_masks,
)
from serenitymojo.models.text_encoder.ideogram_qwen3vl import (
    load_ideogram_qwen3vl, encode_ideogram_taps,
)
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
    reject_unsupported_qwen_edit_conditioning_params, reject_unsupported_lanpaint_params,
)
from serenitymojo.tokenizer.tokenizer import Qwen3Tokenizer


comptime COND = "/home/alex/.serenity/models/ideogram-4-fp8/transformer/diffusion_pytorch_model.safetensors"
comptime UNCOND = "/home/alex/.serenity/models/ideogram-4-fp8/unconditional_transformer/diffusion_pytorch_model.safetensors"
comptime TE = "/home/alex/.serenity/models/ideogram-4-fp8/text_encoder/model.safetensors"
comptime TOK_JSON = "/home/alex/.serenity/models/ideogram-4-fp8/tokenizer/tokenizer.json"
comptime VAE = "/home/alex/.serenity/models/ideogram-4-fp8/vae/diffusion_pytorch_model.safetensors"
comptime LATENT_NORM = "/home/alex/mojodiffusion/serenitymojo/models/dit/parity/ideogram4_fx_latentnorm.safetensors"

comptime IMG_OFFSET = 65536
comptime PAD_ID = 151643
comptime TEXT_TOKENS = 1024
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


def _build_fixed_inputs(ctx: DeviceContext) raises -> List[TArc]:
    """Fixed 1024 text-token window + 4096 image tokens for 1024x1024."""
    var pos = List[Float32]()
    var ind = List[Float32]()
    var npos = List[Float32]()
    var nind = List[Float32]()
    for l in range(TEXT_TOKENS):
        pos.append(Float32(l))
        pos.append(Float32(l))
        pos.append(Float32(l))
        ind.append(3.0)  # LLM_TOKEN_INDICATOR
    for h in range(GH):
        for w in range(GW):
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
    out.append(TArc(Tensor.from_host(pos^, [1, TOTAL, 3], STDtype.F32, ctx)))
    out.append(TArc(Tensor.from_host(ind^, [1, TOTAL], STDtype.F32, ctx)))
    out.append(TArc(Tensor.from_host(npos^, [1, NIMG, 3], STDtype.F32, ctx)))
    out.append(TArc(Tensor.from_host(nind^, [1, NIMG], STDtype.F32, ctx)))
    return out^


struct Ideogram4RopeSet(Movable):
    var cond: Tuple[Tensor, Tensor]
    var uncond: Tuple[Tensor, Tensor]

    def __init__(
        out self, var cond: Tuple[Tensor, Tensor], var uncond: Tuple[Tensor, Tensor]
    ):
        self.cond = cond^
        self.uncond = uncond^


struct Ideogram4Backend(GenBackend, Movable):
    var ctx: DeviceContext

    # Transformer weights are resident only during a job's denoise phase.
    var loaded: Bool
    var load_stage: Int
    var cond: List[ArcPointer[Ideogram4Weights]]
    var uncond: List[ArcPointer[Ideogram4Weights]]

    # Static 1024x1024 sequence helpers, safe to retain across jobs.
    var static_ready: Bool
    var cond_masks: List[ArcPointer[Ideogram4Masks]]
    var uncond_masks: List[ArcPointer[Ideogram4Masks]]
    var ropes: List[ArcPointer[Ideogram4RopeSet]]

    # Per-job state.
    var active: Bool
    var cancel_flag: Bool
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
        self.load_stage = 0
        self.cond = List[ArcPointer[Ideogram4Weights]]()
        self.uncond = List[ArcPointer[Ideogram4Weights]]()
        self.static_ready = False
        self.cond_masks = List[ArcPointer[Ideogram4Masks]]()
        self.uncond_masks = List[ArcPointer[Ideogram4Masks]]()
        self.ropes = List[ArcPointer[Ideogram4RopeSet]]()
        self.active = False
        self.cancel_flag = False
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
        reject_unsupported_mask_image_params(params, String("ideogram4"))
        reject_unsupported_lanpaint_params(params, String("ideogram4"))
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
        if not (params.width == 1024 and params.height == 1024):
            raise Error(
                String("ideogram4: unsupported size ") + String(params.width)
                + "x" + String(params.height)
                + " -- only 1024x1024 is served by the current fixed-shape path"
            )
        if params.negative.byte_length() > 0:
            raise Error("ideogram4: negative prompt is not supported in this bounded slice")
        if len(params.loras) > 0:
            raise Error("ideogram4: LoRA is not supported in this bounded slice")
        if params.init_image.byte_length() > 0:
            raise Error("ideogram4: img2img/init image is not supported in this bounded slice")
        if params.creativity != 0.5:
            raise Error("ideogram4: creativity/denoise control is not supported in this bounded txt2img slice")
        if params.variation_strength > 0.0:
            raise Error("ideogram4: variation noise is not supported in this bounded slice")
        if params.cfg <= 0.0:
            raise Error("ideogram4: cfg must be positive")
        if params.cfg_override >= 0.0:
            if scheduler_admission.executed != "ideogram4_simple_flowmatch":
                raise Error("ideogram4: cfg_override is supported only with the simple AuraFlow scheduler")
            if params.cfg_override == 0.0:
                raise Error("ideogram4: cfg_override must be positive when set")
            if params.cfg_override_start_percent > params.cfg_override_end_percent:
                raise Error("ideogram4: cfg_override_start_percent must be <= cfg_override_end_percent")

        # A previous job may have failed mid-denoise; never try to encode text
        # while the cond/uncond transformers are still resident.
        self._free_transformers()
        self.params = params.copy()
        self.cfg = Float32(params.cfg)
        self.executed_sampler = sampler_admission.executed.copy()
        self.executed_scheduler = scheduler_admission.executed.copy()
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

    def _ensure_static(mut self) raises:
        if self.static_ready:
            return
        print("[ideogram4] building fixed 1024x1024 masks and MRoPE")
        var inp = _build_fixed_inputs(self.ctx)
        var sec = [24, 20, 20]
        var cs = build_ideogram4_mrope(
            inp[0][], HEAD_DIM, sec, Float32(5000000.0), self.ctx, STDtype.BF16
        )
        var ncs = build_ideogram4_mrope(
            inp[2][], HEAD_DIM, sec, Float32(5000000.0), self.ctx, STDtype.BF16
        )
        self.ropes = List[ArcPointer[Ideogram4RopeSet]]()
        self.ropes.append(ArcPointer(Ideogram4RopeSet(cs^, ncs^)))
        self.cond_masks = List[ArcPointer[Ideogram4Masks]]()
        self.uncond_masks = List[ArcPointer[Ideogram4Masks]]()
        self.cond_masks.append(ArcPointer(ideogram4_build_masks(inp[1][], self.ctx)))
        self.uncond_masks.append(ArcPointer(ideogram4_build_masks(inp[3][], self.ctx)))
        self.static_ready = True

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
        var enc = load_ideogram_qwen3vl(String(TE), self.ctx)
        var features = encode_ideogram_taps(enc, ids, self.ctx)
        self.text_features = List[TArc]()
        self.text_features.append(TArc(features^))

    def _load_one(mut self) raises -> Bool:
        if self.loaded:
            return True
        if self.load_stage == 0:
            print("[ideogram4] loading conditional fp8 transformer")
            self.cond = List[ArcPointer[Ideogram4Weights]]()
            self.cond.append(ArcPointer(
                Ideogram4Weights.load(ShardedSafeTensors.open(String(COND)), self.ctx)
            ))
            self.load_stage = 1
            return False
        if self.load_stage == 1:
            print("[ideogram4] loading unconditional fp8 transformer")
            self.uncond = List[ArcPointer[Ideogram4Weights]]()
            self.uncond.append(ArcPointer(
                Ideogram4Weights.load(ShardedSafeTensors.open(String(UNCOND)), self.ctx)
            ))
            self.loaded = True
            self.load_stage = 2
            return True
        return True

    def _prepare_job(mut self) raises:
        if len(self.text_features) == 0:
            raise Error("ideogram4: missing text features")
        self._ensure_static()

        var zllm = _zero_host(NIMG * LLM_DIM)
        var img_zeros = Tensor.from_host(zllm^, [1, NIMG, LLM_DIM], STDtype.BF16, self.ctx)
        var llm_full = concat(1, self.ctx, self.text_features[0][], img_zeros)
        self.llm = List[TArc]()
        self.llm.append(TArc(llm_full^))
        self.text_features = List[TArc]()

        var nllm = _zero_host(NIMG * LLM_DIM)
        self.neg_llm = List[TArc]()
        self.neg_llm.append(TArc(
            Tensor.from_host(nllm^, [1, NIMG, LLM_DIM], STDtype.BF16, self.ctx)
        ))

        var zpad = _zero_host(TEXT_TOKENS * LATENT_DIM)
        self.text_zpad = List[TArc]()
        self.text_zpad.append(TArc(
            Tensor.from_host(zpad^, [1, TEXT_TOKENS, LATENT_DIM], STDtype.F32, self.ctx)
        ))

        self.latent = List[TArc]()
        self.latent.append(TArc(
            randn([1, NIMG, LATENT_DIM], UInt64(self.params.seed), STDtype.F32, self.ctx)
        ))

        self.sigma_trace = List[Float32]()
        if self.executed_scheduler == "ideogram4_simple_flowmatch":
            self.intervals = List[Float32]()
            self.sigma_trace = _build_ideogram4_simple_sigmas(
                self.params.steps, self.params.sigma_shift
            )
        else:
            self.intervals = make_step_intervals(self.params.steps)
            var mean = ideogram4_schedule_mean(1024, 1024, 0.0)
            for i in range(len(self.intervals)):
                self.sigma_trace.append(
                    ideogram4_logitnormal(Float64(self.intervals[i]), mean, 1.5)
                )
        print(
            "[ideogram4] job", self.params.job_id, ":", self.params.steps,
            "steps, cfg", self.cfg, "scheduler", self.executed_scheduler,
            "shift", self.params.sigma_shift, "seed", self.params.seed,
            "size", self.params.width, "x", self.params.height,
        )

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

    def _denoise_one(mut self) raises:
        var t_val: Float32
        var s_val: Float32
        if self.executed_scheduler == "ideogram4_simple_flowmatch":
            s_val = self.sigma_trace[self.cur]
            t_val = self.sigma_trace[self.cur + 1]
        else:
            # Preserve the existing logit-normal runtime convention; the Comfy
            # simple path above uses the descending sigma order directly.
            var step_idx = self.params.steps - 1 - self.cur
            t_val = self.sigma_trace[step_idx + 1]
            s_val = self.sigma_trace[step_idx]
        var sigma_for_cfg = s_val
        var step_cfg = self._cfg_for_sigma(sigma_for_cfg)
        var t = Tensor.from_host([t_val], [1], STDtype.F32, self.ctx)
        var pos_z = cast_tensor(
            concat(1, self.ctx, self.text_zpad[0][], self.latent[0][]),
            STDtype.BF16,
            self.ctx,
        )
        var cout = ideogram4_forward_r[TOTAL](
            self.cond[0][], pos_z, self.llm[0][], t, self.cond_masks[0][],
            self.ropes[0][].cond[0], self.ropes[0][].cond[1],
            LAYERS, HEADS, HEAD_DIM, HIDDEN, self.ctx,
        )
        var pos_v = slice(cout, 1, TEXT_TOKENS, NIMG, self.ctx)
        var t2 = Tensor.from_host([t_val], [1], STDtype.F32, self.ctx)
        var z_bf = cast_tensor(self.latent[0][], STDtype.BF16, self.ctx)
        var nout = ideogram4_forward_r[NIMG](
            self.uncond[0][], z_bf, self.neg_llm[0][], t2, self.uncond_masks[0][],
            self.ropes[0][].uncond[0], self.ropes[0][].uncond[1],
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
        content += String('    "text_padding_policy":"Qwen pad-token features are included to keep the DiT sequence shape static; sampler parity is not accepted",\n')
        content += String('    "image_index":') + String(self.params.image_index) + String(",\n")
        content += String('    "image_count":') + String(self.params.image_count) + String(",\n")
        content += String('    "variation_applied":false,\n')
        content += String('    "lora_count":0,\n')
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

    def _decode_and_save(mut self) raises -> String:
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
        var z6 = reshape(zd, [1, GH, GW, 2, 2, 32], self.ctx)
        var zp = permute(z6, [0, 5, 1, 3, 2, 4], self.ctx)
        var latent = reshape(zp, [1, 32, VAE_H, VAE_W], self.ctx)
        self.latent = List[TArc]()
        print("[ideogram4] loading VAE decoder + decode")
        var dec = load_ideogram4_vae_decoder[VAE_H, VAE_W](String(VAE), self.ctx)
        var img = dec.decode(cast_tensor(latent, STDtype.BF16, self.ctx), self.ctx)
        _save_rgb_png_with_text(img, png_path, self.params.params_json, self.ctx)
        return png_path

    def _clear_job(mut self):
        self._free_transformers()
        self.active = False
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
            if self.phase == IPHASE_ENCODE:
                if not self.announced:
                    self.announced = True
                    r.step = 0
                    r.phase = String("encoding")
                    return r^
                var encode_t0 = perf_counter_ns()
                self._encode()
                self.text_encode_seconds = Float64(perf_counter_ns() - encode_t0) / 1.0e9
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
