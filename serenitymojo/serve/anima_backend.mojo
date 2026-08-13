# serenitymojo.serve.anima_backend — the real Anima 1024x1024 GenBackend.
#
# Wraps the VERIFIED Anima inference stages (pipeline/anima_serenity_cli.mojo
# math + models/anima/anima_text_context.mojo + models/text_encoder/
# qwen3_encoder.mojo + models/anima/anima_stack_lora.mojo + models/vae/
# qwenimage_decoder.mojo) behind the pull-based GenBackend seam (backend.mojo).
#
# RUNTIME TEXT ENCODE — REAL (not a cached sidecar). Unlike anima_serenity_cli /
# anima_sample_cli — which read a PRE-ENCODED context safetensors sidecar because
# their header pre-dates the Mojo tokenizers — THIS backend tokenizes the live
# params.prompt + params.negative at runtime and runs the FULL SerenityTrainer text
# path in pure Mojo:
#
#   qwen_ids  = Qwen3Tokenizer.encode(prompt)  -> truncate/pad@512 (pad 151643)
#   t5_ids    = T5Tokenizer.encode(prompt)     -> (EOS=1 appended) trunc/pad@512 (pad 0)
#   qwen_hidden = Qwen3-0.6B(qwen_ids).last_hidden_state  [1,512,1024]   (Mojo)
#   qwen_hidden *= mask                                    (AnimaModel.py:218)  (Mojo)
#   context     = net.llm_adapter(t5_ids, qwen_hidden, mask) [1,512,1024] F32   (Mojo)
#
# This is anima_text_context_from_tokens() verbatim, called once for the prompt
# (cond) and once for the negative/empty prompt (uncond). The TOKENIZERS are
# verified id-for-id against the HF references the reference trainer recipe uses:
#   * Qwen3-0.6B-Base == Qwen2TokenizerFast (BPE, vocab 151643, no BOS, pad/eos
#     151643); the on-disk qwen-image-2512 tokenizer.json is the SAME BPE and
#     produces identical ids for plain text (measured: "a photo of a cat" ->
#     [64,6548,315,264,8251] both ways). NO chat template is applied (reference trainer feeds the
#     raw prompt). encode() adds no BOS/EOS, matching Qwen2TokenizerFast.
#   * T5TokenizerFast google/t5-v1_1-xxl == the Unigram SentencePiece the Mojo
#     T5Tokenizer already passed parity on (t5-base shares the spiece); EOS=1
#     appended by encode(), pad 0, unk 2 (measured: "a photo of a cat" ->
#     [3,9,1202,13,3,9,1712,1]; empty "" -> [1]).
#
# Residency model (single-GPU, 24 GB lesson):
#   * The Anima DiT (base projections + t_embedder + 28 resident blocks ~3.7 GiB
#     BF16 + RoPE tables + base-only LoRA overlay) is loaded ONCE (first job) and
#     STAYS RESIDENT across jobs (the residency win — like SDXL's UNet).
#   * The Qwen3-0.6B encoder (~1.2 GB) + net.llm_adapter weights are loaded ->
#     used -> freed PER JOB inside the ENCODE step.
#   * The Qwen-Image VAE (3D-conv, multi-GiB upsample peak at 1024²) is loaded
#     PER JOB inside the DECODE step. Because the resident DiT blocks would OOM a
#     24 GB card during the VAE upsample (the same failure SDXL measured), the
#     resident DiT is FREED + the mempool trimmed BEFORE decode; the next job
#     reloads it in APHASE_LOAD (self.loaded=False). The DiT host-latent survives
#     the free (host List[Float32]), exactly like anima_serenity_cli.
#
# step() state machine: ENCODE (per-job, blocking — announced phase="encoding")
#   -> LOAD (DiT, once/post-decode, announced phase="loading") -> DENOISE×steps
#   (one CFG dual-forward + direct-velocity Euler update per tick) -> DECODE
#   (announced phase="decoding") -> done. cancel() makes the next step() return
#   cancelled and frees all per-job state.
#
# Size support: the explicit seven-shape 1024px-area product ladder. Its DiT
# attention dispatch reduces to three compiled S_IMG values (4032/4096/4160),
# with per-shape RoPE tables and Wan21 tiled VAE decode. steps/cfg/seed are
# honored at runtime; sigmas reuse the parity-gated Anima FlowMatch shift=3 path.
#
# LoRA: the base-only run keeps the zero-B identity overlay. A requested
# SerenityTrainer/PEFT Anima adapter is loaded through the model's canonical
# prefix table, its file alpha/rank scale is preserved, and the user's
# multiplier composes on top. img2img remains fail-loud.

from std.collections import List, Optional
from std.ffi import external_call
from max.gpu.host import DeviceContext
from std.math import sqrt, log as flog, cos as fcos, sin as fsin, exp as fexp
from std.memory import alloc, ArcPointer
from std.time import perf_counter_ns

from image.buffer import Image
from image.png import encode_png_with_text

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.ffi import BytePtr
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.offload.vmm_cuda import cu_mempool_trim_current, cu_mem_get_info

from serenitymojo.tokenizer.tokenizer import Qwen3Tokenizer
from serenitymojo.tokenizer.tokenizer_cache import qwen3_tokenizer_open
from serenitymojo.tokenizer.t5_tokenizer import T5Tokenizer

from serenitymojo.ops.linear import linear
from serenitymojo.ops.activations import silu
from serenitymojo.ops.norm import rms_norm

from serenitymojo.models.text_encoder.qwen3_encoder import Qwen3Config, Qwen3Encoder
from serenitymojo.models.anima.anima_text_context import (
    ANIMA_T5_PAD_ID, AnimaAdapterWeights, tokenize_anima_text,
    encode_anima_context_host,
)
from serenitymojo.models.anima.weights import (
    AnimaStackBase, load_anima_stack_base, verify_anima_stack_shapes,
    AnimaBlockWeights, load_anima_block_weights_bf16_normf32,
)
from serenitymojo.models.anima.anima_stack_lora import (
    AnimaLoraSet, build_anima_lora_set,
    AnimaLoraDeviceSet, anima_lora_set_to_device,
    anima_stack_lora_forward_device_resident_nosave,
    load_anima_lora_resume,
)
from serenitymojo.training.train_step import LoraAdapter
from serenitymojo.models.dit.anima_contract import (
    ANIMA_HIDDEN, ANIMA_NUM_HEADS, ANIMA_HEAD_DIM, ANIMA_DEPTH,
    ANIMA_LATENT_CHANNELS, ANIMA_PATCH_SIZE, ANIMA_ADAPTER_DIM,
    ANIMA_DIT_PATH, ANIMA_VAE_PATH,
)
from serenitymojo.models.vae.qwenimage_tiled_decode import wan21_image_tiled_decode
from serenitymojo.sampling.anima_sampling import AnimaLinearFlowScheduler
from serenitymojo.image.png import _quantize, ValueRange

from serenitymojo.sampling.sampler_registry import (
    sampler_admission_for_backend, scheduler_admission_for_backend,
)
from serenitymojo.serve.backend import (
    GenBackend, JobParams, StepResult, reject_unsupported_common_runtime_params,
    reject_unsupported_reference_image_params, reject_unsupported_mask_image_params,
    reject_unsupported_inpaint_conditioning_params,
    reject_unsupported_qwen_edit_conditioning_params,
    reject_unsupported_conditioning_mask_params, reject_unsupported_lanpaint_params,
    warn_unsupported_advanced_sampling_params,
)
from serenitymojo.serve.product_manifest import (
    json_bool, json_escape, peak_vram_mib, write_text_file,
)


comptime TArc = ArcPointer[Tensor]
comptime GENPARAMS_TEXT_KEY = "serenity.genparams.v1"

# ── Anima dims (from anima_contract / anima_serenity_cli) ────────────────────
comptime B = 1
comptime H = ANIMA_NUM_HEADS        # 16
comptime Dh = ANIMA_HEAD_DIM        # 128
comptime D = ANIMA_HIDDEN           # 2048
comptime F = 8192                   # GELU MLP hidden
comptime JOINT = ANIMA_ADAPTER_DIM  # 1024 cross-attn context dim
comptime C = ANIMA_LATENT_CHANNELS  # 16
comptime PS = ANIMA_PATCH_SIZE      # 2
comptime IN_PATCH = (C + 1) * PS * PS   # 68
comptime OUT_PATCH = C * PS * PS        # 64
comptime EPS = Float32(1e-06)

comptime S_IMG_LOW = 4032                   # 4:3/3:4 + 16:9/9:16
comptime S_IMG_SQUARE = 4096                # 1024x1024
comptime S_IMG_HIGH = 4160                  # 3:2/2:3
comptime S_TXT = 512                        # trained context length

# ── LoRA recipe (rank=16, alpha=16, matches anima_serenity_cli) ──────────────
comptime RANK = 16
comptime ALPHA = Float32(16.0)

# ── runtime tokenizer assets (VERIFIED id-parity with the reference trainer HF references) ──
# Qwen3-0.6B-Base BPE == the on-disk qwen-image-2512 tokenizer.json (identical
# ids for plain text; pad/eos 151643, no BOS). T5 == t5xxl Unigram SentencePiece
# (t5-v1_1-xxl-compatible; EOS=1 appended by encode, pad 0).
comptime QWEN3_TOK_JSON = "models/qwen-image/tokenizer/tokenizer.json"
comptime T5_TOK_JSON = "models/text-encoders/t5xxl_fp16.tokenizer.json"
comptime QWEN3_DIR = "models/anima/split_files/text_encoders/qwen_3_06b_base.safetensors"

# ── FlowMatch shift=3 schedule + direct-velocity Euler ──────────────────────
comptime DEFAULT_CFG = Float32(4.5)
comptime ANIMA_SCHEDULE_SHIFT = Float32(3.0)


comptime APHASE_IDLE = 0
comptime APHASE_ENCODE = 1
comptime APHASE_LOAD = 2
comptime APHASE_DENOISE = 3
comptime APHASE_DECODE = 4


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
        String("echo -n '[anima][vram] ") + tag
        + ": ' && nvidia-smi --query-gpu=memory.used --format=csv,noheader"
    )


def _save_rgb_png_with_text(
    rgb: Tensor, path: String, params_json: String, ctx: DeviceContext
) raises:
    """[1,3,H,W] SIGNED float tensor -> PNG with serenity.genparams.v1 tEXt."""
    var shape = rgb.shape()
    if len(shape) != 4 or shape[0] != 1 or shape[1] != 3:
        raise Error("anima_backend: expected [1,3,H,W] rgb tensor")
    var height = shape[2]
    var width = shape[3]
    var host = rgb.to_host(ctx)
    var plane = height * width
    if len(host) != 3 * plane:
        raise Error("anima_backend: rgb to_host size mismatch")
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


# ── runtime tokenization → the three reference trainer id arrays (qwen_ids, qwen_mask, t5_ids) ─
# Mirrors AnimaModel.encode_text (AnimaModel.py:190-208): HF tokenizes at
# max_length=512, padding='max_length', truncation=True. Here:
#   qwen: Qwen3Tokenizer.encode (no specials) -> truncate to 512 -> right-pad
#         151643; mask = 1 for real tokens, 0 for pad.
#   t5  : T5Tokenizer.encode (appends EOS=1) -> truncate to 512 -> right-pad 0.
# ── deterministic host gaussian noise (Box-Muller; verbatim from
#    anima_serenity_cli so the seed convention matches the proven path) ────────
def _host_noise(n: Int, seed: UInt64) -> List[Float32]:
    var out = List[Float32]()
    var state = seed
    var i = 0
    while i < n:
        state = state * 6364136223846793005 + 1442695040888963407
        var u1f = Float64(Int((state >> 11) & 0xFFFFFFFFFFFFF)) * (1.0 / 4503599627370496.0)
        state = state * 6364136223846793005 + 1442695040888963407
        var u2f = Float64(Int((state >> 11) & 0xFFFFFFFFFFFFF)) * (1.0 / 4503599627370496.0)
        if u1f < 1.0e-12:
            u1f = 1.0e-12
        var r = sqrt(-2.0 * flog(Float64(u1f)))
        var theta = 6.283185307179586 * u2f
        out.append(Float32(r * fcos(Float64(theta))))
        if i + 1 < n:
            out.append(Float32(r * fsin(Float64(theta))))
        i += 2
    return out^


# ── t_embedder forward (RAW sigma; verbatim from anima_serenity_cli) ─────────
struct _TEmb(Movable):
    var t_cond: List[Float32]
    var base_adaln: List[Float32]

    def __init__(out self, var t_cond: List[Float32], var base_adaln: List[Float32]):
        self.t_cond = t_cond^
        self.base_adaln = base_adaln^


def _sinusoidal_host(val: Float32, dim: Int) -> List[Float32]:
    var half = dim // 2
    var neg_ln = -flog(Float32(10000.0))
    var out = List[Float32]()
    out.reserve(dim)
    for _ in range(dim):
        out.append(Float32(0.0))
    for i in range(half):
        var freq = fexp(neg_ln * (Float32(i) / Float32(half)))
        var angle = val * freq
        out[i] = fcos(angle)
        out[half + i] = fsin(angle)
    return out^


def _prepare_timestep(
    sigma: Float32, base: AnimaStackBase, ctx: DeviceContext
) raises -> _TEmb:
    var emb_l = _sinusoidal_host(sigma, D)
    var emb = Tensor.from_host(emb_l, [B, D], STDtype.F32, ctx)
    var h = linear(emb, base.te_lin1[], Optional[Tensor](None), ctx)
    var hidden = silu(h, ctx)
    var base_adaln = linear(hidden, base.te_lin2[], Optional[Tensor](None), ctx)
    var t_cond = rms_norm(emb, base.t_norm[], EPS, ctx)
    return _TEmb(t_cond.to_host(ctx), base_adaln.to_host(ctx))


# ── 3D-RoPE tables (verbatim from anima_serenity_cli._rope_tables) ───────────
struct _Rope(Movable):
    var cos: Tensor
    var sin: Tensor

    def __init__(out self, var cos: Tensor, var sin: Tensor):
        self.cos = cos^
        self.sin = sin^


def _rope_tables(nh: Int, nw: Int, ctx: DeviceContext) raises -> _Rope:
    var half = Dh // 2
    var full_d = Dh
    var t_frames = 1
    var s_img = nh * nw

    var dim_h = full_d // 6 * 2
    var dim_w = dim_h
    var dim_t = full_d - 2 * dim_h
    var bins_t = dim_t // 2
    var bins_h = dim_h // 2
    var bins_w = dim_w // 2

    var base_theta: Float64 = 10000.0
    var h_exp = Float64(dim_h) / (Float64(dim_h) - 2.0)
    var w_exp = Float64(dim_w) / (Float64(dim_w) - 2.0)
    var h_ntk = fexp(flog(Float64(4.0)) * h_exp)
    var w_ntk = fexp(flog(Float64(4.0)) * w_exp)
    var t_ntk = Float64(1.0)
    var theta_h = Float64(base_theta * h_ntk)
    var theta_w = Float64(base_theta * w_ntk)
    var theta_t = Float64(base_theta * t_ntk)

    var freqs_t = List[Float32]()
    for i in range(bins_t):
        var ev = Float64(2 * i) / Float64(dim_t)
        freqs_t.append(Float32(fexp(-flog(theta_t) * ev)))
    var freqs_h = List[Float32]()
    for i in range(bins_h):
        var ev = Float64(2 * i) / Float64(dim_h)
        freqs_h.append(Float32(fexp(-flog(theta_h) * ev)))
    var freqs_w = List[Float32]()
    for i in range(bins_w):
        var ev = Float64(2 * i) / Float64(dim_w)
        freqs_w.append(Float32(fexp(-flog(theta_w) * ev)))

    var cosl = List[Float32]()
    var sinl = List[Float32]()
    for _b in range(B):
        for tf in range(t_frames):
            for ih in range(nh):
                for iw in range(nw):
                    for _hd in range(H):
                        for fi in range(bins_t):
                            var ang = Float32(tf) * freqs_t[fi]
                            cosl.append(fcos(ang))
                            sinl.append(fsin(ang))
                        for fi in range(bins_h):
                            var ang = Float32(ih) * freqs_h[fi]
                            cosl.append(fcos(ang))
                            sinl.append(fsin(ang))
                        for fi in range(bins_w):
                            var ang = Float32(iw) * freqs_w[fi]
                            cosl.append(fcos(ang))
                            sinl.append(fsin(ang))
    var cos_t = Tensor.from_host(cosl, [B * s_img * H, half], STDtype.F32, ctx)
    var sin_t = Tensor.from_host(sinl, [B * s_img * H, half], STDtype.F32, ctx)
    return _Rope(cos_t^, sin_t^)


# ── patchify / unpatchify (verbatim from anima_serenity_cli) ─────────────────
def _patchify_in(
    x: List[Float32], Bd: Int, Hd: Int, Wd: Int, Cd: Int
) -> List[Float32]:
    var pH = PS
    var pW = PS
    var nH = Hd // pH
    var nW = Wd // pW
    var Cp = Cd + 1
    var N = nH * nW
    var pd = Cp * pH * pW
    var out = List[Float32]()
    out.reserve(Bd * N * pd)
    for _ in range(Bd * N * pd):
        out.append(Float32(0.0))
    for b in range(Bd):
        for ih in range(nH):
            for iw in range(nW):
                var pn = ih * nW + iw
                for c in range(Cp):
                    for ph in range(pH):
                        for pw in range(pW):
                            var od = (b * N + pn) * pd + (c * pH * pW + ph * pW + pw)
                            if c < Cd:
                                var hh = ih * pH + ph
                                var ww = iw * pW + pw
                                var src = ((b * Hd + hh) * Wd + ww) * Cd + c
                                out[od] = x[src]
    return out^


def _unpatchify_out(
    p: List[Float32], Bd: Int, Hd: Int, Wd: Int, Cd: Int
) -> List[Float32]:
    var pH = PS
    var pW = PS
    var nH = Hd // pH
    var nW = Wd // pW
    var pd = Cd * pH * pW
    var N = nH * nW
    var out = List[Float32]()
    out.reserve(Bd * Hd * Wd * Cd)
    for _ in range(Bd * Hd * Wd * Cd):
        out.append(Float32(0.0))
    for b in range(Bd):
        for ih in range(nH):
            for iw in range(nW):
                var pn = ih * nW + iw
                for ph in range(pH):
                    for pw in range(pW):
                        for c in range(Cd):
                            var od = (b * N + pn) * pd + (ph * pW * Cd + pw * Cd + c)
                            var hh = ih * pH + ph
                            var ww = iw * pW + pw
                            var dst = ((b * Hd + hh) * Wd + ww) * Cd + c
                            out[dst] = p[od]
    return out^


# ── zero-B overlay: forward reduces to frozen base DiT (verbatim) ────────────
def _zero_b_set(set: AnimaLoraSet) -> AnimaLoraSet:
    var ad = List[LoraAdapter]()
    for i in range(len(set.ad)):
        var src = set.ad[i].copy()
        for j in range(len(src.b)):
            src.b[j] = BFloat16(0.0)
        ad.append(src^)
    return AnimaLoraSet(ad^, set.num_blocks, set.rank)


def _multiply_lora_set(
    set: AnimaLoraSet, multiplier: Float32
) -> AnimaLoraSet:
    """Compose the UI multiplier with the adapter's file alpha/rank scale."""
    var ad = List[LoraAdapter]()
    for i in range(len(set.ad)):
        var src = set.ad[i].copy()
        src.scale *= multiplier
        ad.append(src^)
    return AnimaLoraSet(ad^, set.num_blocks, set.rank)


def _count_nonfinite(v: List[Float32]) -> Int:
    var bad = 0
    for i in range(len(v)):
        var x = v[i]
        if (x != x) or (x - x != Float32(0.0)):
            bad += 1
    return bad


# ── channels-last [B,H,W,C] F32 -> NCHW BF16 Tensor (verbatim) ───────────────
def _bhwc_to_nchw_tensor(
    bhwc: List[Float32], lh: Int, lw: Int, ctx: DeviceContext
) raises -> Tensor:
    var nchw = List[Float32]()
    nchw.resize(B * C * lh * lw, Float32(0.0))
    for b in range(B):
        for hd in range(lh):
            for wd in range(lw):
                for c in range(C):
                    var src = ((b * lh + hd) * lw + wd) * C + c
                    var dst = ((b * C + c) * lh + hd) * lw + wd
                    nchw[dst] = bhwc[src]
    var sh = List[Int]()
    sh.append(1); sh.append(C); sh.append(lh); sh.append(lw)
    return Tensor.from_host(nchw, sh^, STDtype.BF16, ctx)


# ── resident DiT bundle (base + 28 blocks + RoPE + base-only LoRA device set) ─
struct AnimaDiTResident(Movable):
    var base: AnimaStackBase
    var blocks: List[AnimaBlockWeights]
    var lora_dev: AnimaLoraDeviceSet
    var rope: _Rope

    def __init__(
        out self, var base: AnimaStackBase, var blocks: List[AnimaBlockWeights],
        var lora_dev: AnimaLoraDeviceSet, var rope: _Rope,
    ):
        self.base = base^
        self.blocks = blocks^
        self.lora_dev = lora_dev^
        self.rope = rope^


struct AnimaBackend(GenBackend, Movable):
    var ctx: DeviceContext

    # ── resident across jobs (DiT bundle, loaded once / after a decode-free) ──
    var loaded: Bool
    var model: List[ArcPointer[AnimaDiTResident]]   # 0/1 (resident DiT)
    var model_width: Int
    var model_height: Int

    # ── per-job state (cleared on done/failed/cancelled) ──
    var active: Bool
    var cancel_flag: Bool
    var phase: Int
    var announced: Bool
    var cur: Int
    var params: JobParams
    var cfg: Float32
    var steps: Int
    var sched: List[ArcPointer[AnimaLinearFlowScheduler]]
    # context host lists (cond + uncond), [B*S_TXT*JOINT] F32, produced at ENCODE.
    var has_ctx: Bool
    var ctx_cond: List[Float32]
    # Keyed (prompt \x1f negative) conditioning cache, capacity 4 drop-oldest
    # — each repeat prompt previously re-loaded Qwen3-0.6B + the adapter from
    # disk and re-ran both forwards.
    var cap_cache_keys: List[String]
    var cap_cache_cond: List[List[Float32]]
    var cap_cache_uncond: List[List[Float32]]
    var ctx_uncond: List[Float32]
    # working latent (channels-last [B,H,W,C] F32), produced at LOAD, updated each
    # DENOISE tick.
    var has_latent: Bool
    var latent: List[Float32]
    var job_t0_ns: Int
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
        self.model = List[ArcPointer[AnimaDiTResident]]()
        self.model_width = 0
        self.model_height = 0
        self.active = False
        self.cancel_flag = False
        self.phase = APHASE_IDLE
        self.announced = False
        self.cur = 0
        self.params = JobParams()
        self.cfg = DEFAULT_CFG
        self.steps = 30
        self.sched = List[ArcPointer[AnimaLinearFlowScheduler]]()
        self.has_ctx = False
        self.ctx_cond = List[Float32]()
        self.ctx_uncond = List[Float32]()
        self.cap_cache_keys = List[String]()
        self.cap_cache_cond = List[List[Float32]]()
        self.cap_cache_uncond = List[List[Float32]]()
        self.has_latent = False
        self.latent = List[Float32]()
        self.job_t0_ns = Int(0)
        self.load_seconds = 0.0
        self.text_encode_seconds = 0.0
        self.prepare_seconds = 0.0
        self.denoise_seconds = 0.0
        self.vae_decode_seconds = 0.0
        self.total_vram_bytes = 0
        self.min_free_bytes = 0

    def backend_name(self) -> String:
        return String("anima")

    def model_name(self) -> String:
        return String("Anima")

    def resident_model(self) -> String:
        """Matches the /v1/models scan entry for the resident Anima base DiT
        (the anima-base-v1.0.safetensors checkpoint)."""
        return String("anima-base-v1.0.safetensors") if self.loaded else String("")

    # ── job admission ─────────────────────────────────────────────────────────
    def start(mut self, params: JobParams) raises:
        if self.active:
            raise Error("AnimaBackend.start: a job is already running")
        reject_unsupported_common_runtime_params(params, String("anima"))
        reject_unsupported_reference_image_params(params, String("anima"))
        reject_unsupported_inpaint_conditioning_params(params, String("anima"))
        reject_unsupported_qwen_edit_conditioning_params(params, String("anima"))
        reject_unsupported_conditioning_mask_params(params, String("anima"))
        reject_unsupported_mask_image_params(params, String("anima"))
        reject_unsupported_lanpaint_params(params, String("anima"))
        var sampler_admission = sampler_admission_for_backend(String("anima"), params.sampler)
        if not sampler_admission.supported:
            raise Error(
                String("anima: unsupported sampler '") + params.sampler
                + String("'; ") + sampler_admission.reason
            )
        var scheduler_admission = scheduler_admission_for_backend(String("anima"), params.scheduler)
        if not scheduler_admission.supported:
            raise Error(
                String("anima: unsupported scheduler '") + params.scheduler
                + String("'; ") + scheduler_admission.reason
            )
        if not (
            (params.width == 1024 and params.height == 1024)
            or (params.width == 1152 and params.height == 896)
            or (params.width == 896 and params.height == 1152)
            or (params.width == 1344 and params.height == 768)
            or (params.width == 768 and params.height == 1344)
            or (params.width == 1280 and params.height == 832)
            or (params.width == 832 and params.height == 1280)
        ):
            raise Error(
                String("anima: unsupported size ") + String(params.width)
                + "x" + String(params.height)
                + " — supported compiled shapes are 1024x1024, 1152x896,"
                + " 896x1152, 1344x768, 768x1344, 1280x832, and 832x1280"
            )
        if params.sigma_shift != 3.0:
            raise Error("anima: sigma_shift override is not supported; use 3.0")
        if len(params.loras) > 1:
            raise Error(
                "anima: this runtime currently supports one canonical"
                " SerenityTrainer/PEFT adapter at a time"
            )
        if params.init_image.byte_length() > 0:
            raise Error(
                "anima: img2img is not supported for Anima yet;"
                " submit without an init image"
            )
        # Warn-loud (never silently drop) on any advanced-sampling knob set but
        # unsupported by this fixed-schedule flow-match path.
        warn_unsupported_advanced_sampling_params(params, String("anima"), List[String]())
        self.params = params.copy()
        self.cfg = Float32(params.cfg) if params.cfg > 0.0 else DEFAULT_CFG
        self.steps = params.steps if params.steps > 0 else 30
        self.active = True
        self.cancel_flag = False
        self.cur = 0
        self.announced = False
        self.has_ctx = False
        self.has_latent = False
        self.phase = APHASE_ENCODE
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
        """Reclaim the per-job transient peak (Qwen3-0.6B encoder ~1.2 GB +
        adapter, the per-forward SDPA scratch, the Qwen-Image VAE decode peak)
        back to the OS via cuMemPoolTrimTo. If a resident DiT survives (it does
        not here — it is freed before decode), its suballocations are NOT
        reclaimed."""
        var before = cu_mem_get_info()
        self.ctx.synchronize()
        cu_mempool_trim_current(0)
        self.ctx.synchronize()
        var after = cu_mem_get_info()
        print("[anima] between-jobs trim: used",
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
        var manifest_path = png_path + String(".anima_daemon_result.json")
        var denoise_per_step = Float64(0.0)
        if self.steps > 0:
            denoise_per_step = self.denoise_seconds / Float64(self.steps)
        var total_wall_seconds = Float64(perf_counter_ns() - self.job_t0_ns) / 1.0e9
        var peak_mib = Float64(0.0)
        if self.total_vram_bytes > 0 and self.min_free_bytes > 0:
            peak_mib = peak_vram_mib(self.total_vram_bytes, self.min_free_bytes)

        var content = String("{\n")
        content += String('  "schema":"serenity.anima.daemon_result.v1",\n')
        content += String('  "backend":"anima_daemon",\n')
        content += String('  "model":"anima",\n')
        content += String('  "readiness_label":"experimental",\n')
        content += String('  "accepted_sampler_parity":false,\n')
        content += String('  "accepted_speed_parity":false,\n')
        content += String('  "run_identity":{\n')
        content += String('    "job_id":"') + json_escape(self.params.job_id) + String('",\n')
        content += String('    "prompt":"') + json_escape(self.params.prompt) + String('",\n')
        content += String('    "negative":"') + json_escape(self.params.negative) + String('",\n')
        content += String('    "seed":') + String(self.params.seed) + String(",\n")
        content += String('    "resolution":{"width":') + String(self.params.width) + String(',"height":') + String(self.params.height) + String("},\n")
        content += String('    "steps":') + String(self.steps) + String(",\n")
        content += String('    "guidance":') + String(self.params.cfg) + String(",\n")
        content += String('    "sampler_registry_backend":"anima",\n')
        content += String('    "requested_sampler":"') + json_escape(self.params.sampler) + String('",\n')
        content += String('    "requested_scheduler":"') + json_escape(self.params.scheduler) + String('",\n')
        content += String('    "executed_sampler":"anima_euler",\n')
        content += String('    "executed_scheduler":"normal",\n')
        content += String('    "schedule_source":"anima_flowmatch_shift3",\n')
        content += String('    "sigma_shift":3.0,\n')
        content += String('    "variation_seed":') + String(self.params.variation_seed) + String(",\n")
        content += String('    "variation_strength":') + String(self.params.variation_strength) + String(",\n")
        content += String('    "variation_applied":') + json_bool(self.params.variation_strength > 0.0) + String(",\n")
        content += String('    "image_index":') + String(self.params.image_index) + String(",\n")
        content += String('    "image_count":') + String(self.params.image_count) + String(",\n")
        content += String('    "lora_count":') + String(len(self.params.loras)) + String(",\n")
        if len(self.params.loras) == 1:
            content += String('    "loaded_lora":"') + json_escape(
                self.params.loras[0].name
            ) + String('",\n')
            content += String('    "loaded_lora_weight":') + String(
                self.params.loras[0].weight
            ) + String(",\n")
            content += String('    "lora_target_count":') + String(
                ANIMA_DEPTH * 10
            ) + String(",\n")
        else:
            content += String('    "loaded_lora":"",\n')
            content += String('    "loaded_lora_weight":0,\n')
            content += String('    "lora_target_count":0,\n')
        content += String('    "dtype":"bf16_dit_f32_host_latent"\n')
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
        content += String('  "note":"Rust-server Mojo worker product-path result; Anima uses runtime Qwen3/T5 conditioning and host latent denoise before Qwen-Image VAE decode. Speed parity remains unaccepted until paired baseline evidence exists."\n')
        content += String("}\n")
        write_text_file(manifest_path, content)
        return manifest_path

    # ── per-job prep ───────────────────────────────────────────────────────────
    def _encode(mut self) raises:
        """REAL runtime text encode: tokenize prompt+negative, run Qwen3-0.6B +
        net.llm_adapter (loaded then freed) -> context_cond / context_uncond host
        F32 lists. The negative is the literal params.negative (empty string ->
        empty-prompt CFG, matching the reference trainer/diffusers unconditional)."""
        var want_key = self.params.prompt + String("\x1f") + self.params.negative
        for slot in range(len(self.cap_cache_keys)):
            if self.cap_cache_keys[slot] != want_key:
                continue
            print("[anima] conditioning cache HIT (slot", slot, "of",
                  len(self.cap_cache_keys), ") — skipping Qwen3+adapter")
            self.ctx_cond = self.cap_cache_cond[slot].copy()
            self.ctx_uncond = self.cap_cache_uncond[slot].copy()
            self.has_ctx = True
            return
        _print_vram("before text encode (Qwen3 + adapter load)")
        var qtok = qwen3_tokenizer_open(
            String(QWEN3_TOK_JSON), String(QWEN3_TOK_JSON) + ".mjtok"
        )
        var t5tok = T5Tokenizer.load(String(T5_TOK_JSON))
        var pos_tokens = tokenize_anima_text(self.params.prompt, qtok, t5tok)
        var neg_tokens = tokenize_anima_text(self.params.negative, qtok, t5tok)
        print("[anima] tokenized prompt: qwen nonpad=",
              _nonpad(pos_tokens.qwen_mask), " t5 len(incl EOS,trunc)=",
              _t5_len(pos_tokens.t5_ids))

        # Qwen3-0.6B encoder + adapter weights load here; both drop at scope exit.
        var enc = Qwen3Encoder.load(String(QWEN3_DIR), Qwen3Config.qwen3_06b(), self.ctx)
        var wts = AnimaAdapterWeights.load_checkpoint(_checkpoint_path(), self.ctx)
        self.ctx_cond = encode_anima_context_host(pos_tokens^, enc, wts, self.ctx)
        self.ctx_uncond = encode_anima_context_host(neg_tokens^, enc, wts, self.ctx)
        self.has_ctx = True
        if len(self.cap_cache_keys) >= 4:
            _ = self.cap_cache_keys.pop(0)
            _ = self.cap_cache_cond.pop(0)
            _ = self.cap_cache_uncond.pop(0)
        self.cap_cache_keys.append(want_key.copy())
        self.cap_cache_cond.append(self.ctx_cond.copy())
        self.cap_cache_uncond.append(self.ctx_uncond.copy())
        # enc / wts drop here (Movable-not-Copyable -> freed at scope exit).
        _print_vram("after text encode (encoder + adapter freed)")

    def _load_model(mut self) raises:
        """Load the Anima DiT resident bundle (base projections + t_embedder + 28
        BF16 blocks + base-only LoRA device overlay + RoPE). Loaded once on the
        first job and after each decode-free."""
        if (
            self.loaded and self.model_width == self.params.width
            and self.model_height == self.params.height
        ):
            return
        self.model = List[ArcPointer[AnimaDiTResident]]()
        self.loaded = False
        self.model_width = 0
        self.model_height = 0
        self.ctx.synchronize()
        cu_mempool_trim_current(0)
        self.ctx.synchronize()
        _print_vram("before Anima DiT load")
        print("[anima] loading Anima base DiT from", ANIMA_DIT_PATH)
        var st = SafeTensors.open(String(ANIMA_DIT_PATH))
        verify_anima_stack_shapes(st, ANIMA_DEPTH)
        var base = load_anima_stack_base(st, self.ctx)

        # Base-only uses an exact zero-B identity overlay. Requested files use
        # the same canonical prefix table as training/resume, then compose the
        # UI multiplier with the file alpha/rank scale.
        var lora: AnimaLoraSet
        if len(self.params.loras) == 1:
            print(
                "[anima] loading canonical adapter:",
                self.params.loras[0].name,
                "weight",
                self.params.loras[0].weight,
            )
            lora = _multiply_lora_set(
                load_anima_lora_resume(
                    ANIMA_DEPTH,
                    RANK,
                    ALPHA,
                    self.params.loras[0].name,
                    self.ctx,
                ),
                Float32(self.params.loras[0].weight),
            )
            print("[anima] loaded", ANIMA_DEPTH * 10, "projection factors")
        else:
            lora = _zero_b_set(
                build_anima_lora_set(ANIMA_DEPTH, D, JOINT, F, RANK, ALPHA)
            )
        var lora_dev = anima_lora_set_to_device(lora, STDtype.BF16, self.ctx)

        var blocks = List[AnimaBlockWeights]()
        for bi in range(ANIMA_DEPTH):
            blocks.append(load_anima_block_weights_bf16_normf32(st, bi, self.ctx))
        var rope = _rope_tables(
            self.params.height // (8 * PS), self.params.width // (8 * PS), self.ctx
        )

        var resident = AnimaDiTResident(base^, blocks^, lora_dev^, rope^)
        self.model = List[ArcPointer[AnimaDiTResident]]()
        self.model.append(ArcPointer(resident^))
        self.loaded = True
        self.model_width = self.params.width
        self.model_height = self.params.height
        _print_vram("after Anima DiT load (resident)")

    def _prepare_job(mut self) raises:
        """Seeded initial channels-last latent (honors seed)."""
        var lh = self.params.height // 8
        var lw = self.params.width // 8
        var n_lat = B * lh * lw * C
        self.latent = _host_noise(n_lat, UInt64(self.params.seed))
        self.sched = List[ArcPointer[AnimaLinearFlowScheduler]]()
        self.sched.append(ArcPointer(
            AnimaLinearFlowScheduler(self.steps, ANIMA_SCHEDULE_SHIFT)
        ))
        self.has_latent = True
        print(
            "[anima] job", self.params.job_id, ":", self.steps,
            "steps, cfg", self.cfg, "seed", self.params.seed,
            "size", self.params.width, "x", self.params.height,
        )

    def _denoise_shape[S_IMG_: Int](
        mut self, lh: Int, lw: Int, sigma: Float32, dt: Float32
    ) raises:
        var patches = _patchify_in(self.latent, B, lh, lw, C)
        var temb = _prepare_timestep(sigma, self.model[0][].base, self.ctx)

        var out_c = anima_stack_lora_forward_device_resident_nosave[H, Dh, S_IMG_, S_TXT](
            patches.copy(), temb.t_cond.copy(), temb.base_adaln.copy(), self.ctx_cond.copy(),
            self.model[0][].base, self.model[0][].blocks, self.model[0][].lora_dev,
            self.model[0][].rope.cos, self.model[0][].rope.sin,
            B, D, JOINT, F, IN_PATCH, OUT_PATCH, EPS, self.ctx,
        )
        var v_c = _unpatchify_out(out_c, B, lh, lw, C)

        var out_u = anima_stack_lora_forward_device_resident_nosave[H, Dh, S_IMG_, S_TXT](
            patches.copy(), temb.t_cond.copy(), temb.base_adaln.copy(), self.ctx_uncond.copy(),
            self.model[0][].base, self.model[0][].blocks, self.model[0][].lora_dev,
            self.model[0][].rope.cos, self.model[0][].rope.sin,
            B, D, JOINT, F, IN_PATCH, OUT_PATCH, EPS, self.ctx,
        )
        var v_u = _unpatchify_out(out_u, B, lh, lw, C)

        for j in range(len(self.latent)):
            var pred = v_u[j] + self.cfg * (v_c[j] - v_u[j])
            self.latent[j] = self.latent[j] + dt * pred
        self.ctx.synchronize()

    # ── one denoise step (CFG dual forward + shifted FlowMatch Euler) ─────────
    def _denoise_one(mut self) raises:
        var sigma = self.sched[0][].sigma(self.cur)
        var dt = self.sched[0][].dt(self.cur)
        var lh = self.params.height // 8
        var lw = self.params.width // 8
        if self.params.width == 1024:
            self._denoise_shape[S_IMG_SQUARE](lh, lw, sigma, dt)
        elif self.params.width == 1280 or self.params.width == 832:
            self._denoise_shape[S_IMG_HIGH](lh, lw, sigma, dt)
        else:
            self._denoise_shape[S_IMG_LOW](lh, lw, sigma, dt)

    # ── final tiled Wan21 decode + PNG ───────────────────────────────────────
    def _decode_shape[LH_: Int, LW_: Int](
        mut self, png_path: String
    ) raises:
        var lat = _bhwc_to_nchw_tensor(self.latent, LH_, LW_, self.ctx)
        var rgb = wan21_image_tiled_decode[LH_, LW_](
            lat, String(ANIMA_VAE_PATH), self.ctx
        )
        _save_rgb_png_with_text(rgb, png_path, self.params.params_json, self.ctx)

    def _decode_and_save(mut self) raises -> String:
        var png_path = self.params.out_dir + "/" + self.params.job_id + ".png"
        var nbad = _count_nonfinite(self.latent)
        if nbad > 0:
            raise Error("anima: final latent contains non-finite values")

        # The Qwen-Image VAE 3D-conv upsample at 1024² OOMs a 24 GB card if the
        # ~3.7 GiB resident DiT stays put (the SDXL failure mode). Free the
        # resident DiT + trim the mempool before decoding; the next job reloads it
        # in APHASE_LOAD (self.loaded=False). The host latent survives the free.
        self.model = List[ArcPointer[AnimaDiTResident]]()
        self.loaded = False
        self.model_width = 0
        self.model_height = 0
        self.ctx.synchronize()
        cu_mempool_trim_current(0)
        self.ctx.synchronize()

        print("[anima] tiled VAE decode (Wan21 keys) + save")
        if self.params.width == 1024:
            self._decode_shape[128, 128](png_path)
        elif self.params.width == 1152:
            self._decode_shape[112, 144](png_path)
        elif self.params.width == 896:
            self._decode_shape[144, 112](png_path)
        elif self.params.width == 1344:
            self._decode_shape[96, 168](png_path)
        elif self.params.width == 768:
            self._decode_shape[168, 96](png_path)
        elif self.params.width == 1280:
            self._decode_shape[104, 160](png_path)
        else:
            self._decode_shape[160, 104](png_path)
        return png_path

    def _clear_job(mut self):
        self.active = False
        self.phase = APHASE_IDLE
        self.cur = 0
        self.cancel_flag = False
        self.announced = False
        self.has_ctx = False
        self.has_latent = False
        self.ctx_cond = List[Float32]()
        self.ctx_uncond = List[Float32]()
        self.latent = List[Float32]()
        self.sched = List[ArcPointer[AnimaLinearFlowScheduler]]()

    # ── the pull-based tick ───────────────────────────────────────────────────
    def step(mut self) raises -> StepResult:
        var r = StepResult()
        r.total = self.steps
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
            if self.phase == APHASE_ENCODE:
                if not self.announced:
                    # announce BEFORE the long blocking encode tick (per-job
                    # Qwen3-0.6B load + adapter forward, cond + uncond).
                    self.announced = True
                    r.step = 0
                    r.phase = String("encoding")
                    return r^
                var encode_t0 = perf_counter_ns()
                self._encode()
                self.text_encode_seconds = Float64(perf_counter_ns() - encode_t0) / 1.0e9
                self._record_vram()
                self.announced = False
                self.phase = APHASE_LOAD
                r.step = 0
                return r^
            if self.phase == APHASE_LOAD:
                var load_t0 = perf_counter_ns()
                var needs_load = (
                    not self.loaded or self.model_width != self.params.width
                    or self.model_height != self.params.height
                )
                if needs_load:
                    if not self.announced:
                        self.announced = True
                        r.step = 0
                        r.phase = String("loading")
                        return r^
                    self._load_model()
                    self.announced = False
                self.load_seconds += Float64(perf_counter_ns() - load_t0) / 1.0e9
                var prep_t0 = perf_counter_ns()
                self._prepare_job()
                self.prepare_seconds += Float64(perf_counter_ns() - prep_t0) / 1.0e9
                self._record_vram()
                self.phase = APHASE_DENOISE
                r.step = 0
                r.phase = String("sampling")
                return r^
            if self.phase == APHASE_DENOISE:
                var denoise_t0 = perf_counter_ns()
                self._denoise_one()
                self.denoise_seconds += Float64(perf_counter_ns() - denoise_t0) / 1.0e9
                self._record_vram()
                self.cur += 1
                r.step = self.cur
                r.phase = String("sampling")
                if self.cur >= self.steps:
                    self.phase = APHASE_DECODE
                return r^
            if not self.announced:
                # announce BEFORE the long blocking VAE-decode tick.
                self.announced = True
                r.step = self.steps
                r.phase = String("decoding")
                return r^
            var decode_t0 = perf_counter_ns()
            var path = self._decode_and_save()
            self.vae_decode_seconds = Float64(perf_counter_ns() - decode_t0) / 1.0e9
            self._record_vram()
            var manifest = self._write_result_manifest(path)
            print("[anima][manifest] saved:", manifest)
            r.step = self.steps
            self._clear_job()
            r.done = True
            r.output_path = path
            return r^
        except e:
            self._clear_job()
            r.failed = True
            r.error = String(e)
            return r^


# ── small helpers ─────────────────────────────────────────────────────────────
def _checkpoint_path() raises -> String:
    return String(ANIMA_DIT_PATH)


def _nonpad(mask: List[Int]) -> Int:
    var n = 0
    for i in range(len(mask)):
        if mask[i] != 0:
            n += 1
    return n


def _t5_len(ids: List[Int]) -> Int:
    # count up to the first pad (0); the encode appends EOS=1 before padding.
    var n = 0
    for i in range(len(ids)):
        if ids[i] == ANIMA_T5_PAD_ID:
            break
        n += 1
    return n
