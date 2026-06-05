# asymflux2_klein9b_gen.mojo — AsymFLUX.2 Klein 9B end-to-end T2I generation.
#
# Pure Mojo+MAX, inference-only, GPU-only. Mirrors the runnable Rust pipeline
# inference-flame/src/bin/asymflux2_klein9b_infer.rs:
#   text encode (Qwen3-8B) → adapter load (58 LoRA fusions + 3 raw replacements
#   + 2 AsymFlow buffers, ADDED in-memory never baked to disk) → pixel-space
#   sample loop (patchify → calibrate → k-scale → Klein forward → asymflow
#   velocity → orthogonal CFG → Oklab clamp → Euler step) → Oklab decode → PNG.
#
# The Klein DiT (models/dit/klein_dit.mojo) and the asymflow velocity algebra
# (models/asymflux2/asymflow.mojo) are REUSED verbatim; this file owns only the
# asymflux2-specific wiring: adapter apply + the asymflow sample loop + the
# Oklab pixel codec + the AsymFLUX.2 sqrt-shift sigma schedule.

from std.gpu.host import DeviceContext
from std.math import sqrt as fsqrt, cbrt as fcbrt, pow as fpow, log as flog
from std.memory import ArcPointer

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.tokenizer.tokenizer import Qwen3Tokenizer
from serenitymojo.models.text_encoder.qwen3_encoder import Qwen3Encoder, Qwen3Config
from serenitymojo.models.dit.klein_dit import (
    Klein9BDiT,
    KleinConfig,
    build_klein_rope_tables,
)
from serenitymojo.models.asymflux2.asymflow import (
    AsymFlowCalibration,
    compute_calibration,
    asymflow_velocity,
    pixel_to_packed,
    scale_for_embedder,
    velocity_to_pixel,
)
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.random import randn
from serenitymojo.ops.linear import linear
from serenitymojo.ops.tensor_algebra import (
    add,
    sub,
    mul,
    mul_scalar,
    reshape,
    permute,
    transpose,
    slice,
    concat,
)
from serenitymojo.image.png import save_png, ValueRange


# ─────────────────────────────────────────────────────────────────────────────
# Config — mirrors asymflux2_klein9b_infer.rs defaults
# ─────────────────────────────────────────────────────────────────────────────
comptime QWEN8_DIR = (
    "/home/alex/.cache/huggingface/hub/models--Qwen--Qwen3-8B/"
    "snapshots/b968826d9c46dd6066d109eabc6255188de91218"
)
comptime TOK_JSON = QWEN8_DIR + "/tokenizer.json"
comptime BASE_PATH = (
    "/home/alex/EriDiffusion/Models/checkpoints/flux-2-klein-base-9b.safetensors"
)
comptime ADAPTER_PATH = (
    "/home/alex/EriDiffusion/Models/checkpoints/asymflux2-klein-9b.safetensors"
)
comptime OUT = "/home/alex/mojodiffusion/output/asymflux2_512.png"

comptime PAD_ID = 151643
comptime SEQ = 512  # Klein text token count
comptime PATCH = 16
comptime H = 512
comptime W = 512
comptime HP = H // PATCH  # 32
comptime WP = W // PATCH  # 32
comptime N_IMG = HP * WP  # 1024 image tokens
comptime N_TXT = SEQ  # 512 text tokens
comptime S = N_IMG + N_TXT  # 1536
comptime PACKED_D = 3 * PATCH * PATCH  # 768

comptime NUM_DOUBLE = 8
comptime NUM_SINGLE = 24
comptime STEPS = 28
comptime CFG_SCALE = Float32(4.0)
comptime CFG_ORTHO = Float32(1.0)
comptime SIGMA_MIN = Float32(1.0e-4)
comptime SEED = UInt64(42)

comptime PROMPT = "a photograph of a red apple on a wooden table"
comptime NEGATIVE = (
    "Low quality, worst quality, blurry, deformed, bad anatomy, unclear text"
)

# Oklab constants (LakonLab OklabColorEncoder, Apache 2.0; port of
# inference-flame/src/vae/oklab.rs).
comptime AFFINE_MEAN0 = Float32(0.56)
comptime AFFINE_MEAN1 = Float32(0.0)
comptime AFFINE_MEAN2 = Float32(0.01)
comptime AFFINE_STD = Float32(0.16)


# ─────────────────────────────────────────────────────────────────────────────
# Text encoding (mirrors klein9b pipeline)
# ─────────────────────────────────────────────────────────────────────────────
@fieldwise_init
struct KleinCaps(Movable):
    var pos: Tensor
    var neg: Tensor


def _klein_template(prompt: String) -> String:
    return (
        String("<|im_start|>user\n")
        + prompt
        + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def _tokenize_512(tok: Qwen3Tokenizer, prompt: String) raises -> List[Int]:
    var ids_full = tok.encode(_klein_template(prompt))
    if len(ids_full) > SEQ:
        raise Error("Klein prompt too long for 512 tokens")
    var ids = List[Int](capacity=SEQ)
    for i in range(len(ids_full)):
        ids.append(ids_full[i])
    for _ in range(SEQ - len(ids_full)):
        ids.append(PAD_ID)
    print("  tokens:", len(ids_full), "->", SEQ)
    return ids^


def encode_captions(ctx: DeviceContext) raises -> KleinCaps:
    print("[text] Qwen3-8B Klein conditioning")
    var tok = Qwen3Tokenizer(TOK_JSON)
    var pos_ids = _tokenize_512(tok, PROMPT)
    var neg_ids = _tokenize_512(tok, NEGATIVE)
    var enc = Qwen3Encoder.load(QWEN8_DIR, Qwen3Config.klein_9b(), ctx)
    var pos = enc.encode_klein(pos_ids, ctx)
    var neg = enc.encode_klein(neg_ids, ctx)
    return KleinCaps(pos^, neg^)


# ─────────────────────────────────────────────────────────────────────────────
# Adapter apply — ADD in-memory, NEVER baked to disk.
#
# Mirrors asymflux2_klein9b_infer.rs::load_adapter:
#   - 3 raw replacements (x_embedder/proj_out/norm_out), with the adaLN
#     [scale,shift]→[shift,scale] row-block swap for the final_layer.
#   - 58 LoRA fusions  W += B @ A  (scale 1.0, alpha=rank).
#   - extract proj_buffer (768,128) F32 + scalar scale_buffer.
# All mutations happen on the GPU-resident base weights of a loaded Klein9BDiT;
# nothing is serialized back. This is the runtime additive-overlay rule.
# ─────────────────────────────────────────────────────────────────────────────
struct AsymFlux2Klein(Movable):
    var transformer: Klein9BDiT
    var proj_buffer: Tensor  # (768, 128) F32
    var scale_buffer: Float32

    def __init__(
        out self, var transformer: Klein9BDiT, var proj_buffer: Tensor,
        scale_buffer: Float32,
    ):
        self.transformer = transformer^
        self.proj_buffer = proj_buffer^
        self.scale_buffer = scale_buffer


def _adapter_tensor(
    st: SafeTensors, name: String, ctx: DeviceContext
) raises -> Tensor:
    var info = st.tensor_info(name)
    var bytes = st.tensor_bytes(name)
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    return Tensor.from_view(tv, ctx)


def _replace_weight(
    mut model: Klein9BDiT, name: String, var new_w: Tensor
) raises:
    """In-memory replace of a Klein base weight (raw adapter replacement)."""
    if name not in model.name_to_idx:
        raise Error(String("replace: missing base key ") + name)
    var idx = model.name_to_idx[name]
    model.weights[idx] = ArcPointer(new_w^)


def _fuse_lora(
    mut model: Klein9BDiT, base_key: String, lora_a: Tensor, lora_b: Tensor,
    ctx: DeviceContext,
) raises:
    """W += B @ A, in-memory additive overlay (never baked to disk).

    lora_a: [rank, in]  lora_b: [out, rank].  delta = B @ A = linear(B, Aᵀ)
    since linear(x, Wm) = x @ Wmᵀ ⇒ linear(B, Aᵀ) = B @ A. All BF16."""
    if base_key not in model.name_to_idx:
        raise Error(String("fuse: missing base key ") + base_key)
    var idx = model.name_to_idx[base_key]
    ref w = model.weights[idx][]  # base BF16 [out, in]
    # delta = B @ A.  A is [rank, in]; transpose → [in, rank] so linear(B, Aᵀ)
    # = B[out,rank] @ (Aᵀ)ᵀ[rank,in] = B @ A → [out, in].
    var a_bf16 = cast_tensor(lora_a, STDtype.BF16, ctx)
    var b_bf16 = cast_tensor(lora_b, STDtype.BF16, ctx)
    var a_t = transpose(a_bf16, 0, 1, ctx)  # [in, rank]
    var delta = linear(b_bf16, a_t, None, ctx)  # [out, in] BF16
    var fused = add(w, delta, ctx)  # BF16 + BF16
    model.weights[idx] = ArcPointer(fused^)


def load_adapter(ctx: DeviceContext) raises -> AsymFlux2Klein:
    print("[adapter] loading BASE Klein 9B (fully resident)")
    var model = Klein9BDiT.load_full(BASE_PATH, ctx)
    print("[adapter] loading ADAPTER", ADAPTER_PATH)
    var st = SafeTensors.open(ADAPTER_PATH)

    # 1. AsymFlow buffers.
    var proj_buffer = cast_tensor(
        _adapter_tensor(st, "proj_buffer", ctx), STDtype.F32, ctx
    )
    var scale_host = _adapter_tensor(st, "scale_buffer", ctx).to_host(ctx)
    var scale_buffer = scale_host[0]
    print("[adapter] proj_buffer (768,128) F32, scale_buffer =", scale_buffer)

    # 2. Raw replacements (diffusers key → BFL base key).
    #    x_embedder.weight    → img_in.weight
    #    proj_out.weight      → final_layer.linear.weight
    #    norm_out.linear.weight → final_layer.adaLN_modulation.1.weight (swap)
    var x_emb = cast_tensor(
        _adapter_tensor(st, "x_embedder.weight", ctx), STDtype.BF16, ctx
    )
    _replace_weight(model, String("img_in.weight"), x_emb^)
    var proj_out = cast_tensor(
        _adapter_tensor(st, "proj_out.weight", ctx), STDtype.BF16, ctx
    )
    _replace_weight(model, String("final_layer.linear.weight"), proj_out^)

    # norm_out is diffusers AdaLayerNormContinuous which emits [scale, shift]
    # along rows; BFL final_layer.adaLN decodes [shift, scale]. Swap the two
    # row halves before insertion (asymflux2_klein9b_infer.rs:297-343).
    var norm_out = cast_tensor(
        _adapter_tensor(st, "norm_out.linear.weight", ctx), STDtype.BF16, ctx
    )
    var no_shape = norm_out.shape()
    var hh = no_shape[0]
    var half = hh // 2
    var scale_rows = slice(norm_out, 0, 0, half, ctx)
    var shift_rows = slice(norm_out, 0, half, half, ctx)
    var swapped = concat(0, ctx, shift_rows, scale_rows)
    _replace_weight(
        model, String("final_layer.adaLN_modulation.1.weight"), swapped^
    )
    print("[adapter] 3 raw replacements applied (adaLN [scale,shift]→[shift,scale])")

    # 3. LoRA fusions (58 pairs).
    var fused = 0
    # timestep embedder
    _fuse_lora(
        model, String("time_in.in_layer.weight"),
        _adapter_tensor(st, "time_guidance_embed.timestep_embedder.linear_1.lora_A.weight", ctx),
        _adapter_tensor(st, "time_guidance_embed.timestep_embedder.linear_1.lora_B.weight", ctx),
        ctx,
    )
    fused += 1
    _fuse_lora(
        model, String("time_in.out_layer.weight"),
        _adapter_tensor(st, "time_guidance_embed.timestep_embedder.linear_2.lora_A.weight", ctx),
        _adapter_tensor(st, "time_guidance_embed.timestep_embedder.linear_2.lora_B.weight", ctx),
        ctx,
    )
    fused += 1
    # double-stream: ff → img_mlp, ff_context → txt_mlp
    for i in range(NUM_DOUBLE):
        var di = String(i)
        var dp = String("double_blocks.") + di
        var tp = String("transformer_blocks.") + di
        _fuse_lora(
            model, dp + ".img_mlp.0.weight",
            _adapter_tensor(st, tp + ".ff.linear_in.lora_A.weight", ctx),
            _adapter_tensor(st, tp + ".ff.linear_in.lora_B.weight", ctx), ctx,
        )
        _fuse_lora(
            model, dp + ".img_mlp.2.weight",
            _adapter_tensor(st, tp + ".ff.linear_out.lora_A.weight", ctx),
            _adapter_tensor(st, tp + ".ff.linear_out.lora_B.weight", ctx), ctx,
        )
        _fuse_lora(
            model, dp + ".txt_mlp.0.weight",
            _adapter_tensor(st, tp + ".ff_context.linear_in.lora_A.weight", ctx),
            _adapter_tensor(st, tp + ".ff_context.linear_in.lora_B.weight", ctx), ctx,
        )
        _fuse_lora(
            model, dp + ".txt_mlp.2.weight",
            _adapter_tensor(st, tp + ".ff_context.linear_out.lora_A.weight", ctx),
            _adapter_tensor(st, tp + ".ff_context.linear_out.lora_B.weight", ctx), ctx,
        )
        fused += 4
    # single-stream: attn.to_out → linear2
    for i in range(NUM_SINGLE):
        var si = String(i)
        var sp = String("single_blocks.") + si
        var stp = String("single_transformer_blocks.") + si
        _fuse_lora(
            model, sp + ".linear2.weight",
            _adapter_tensor(st, stp + ".attn.to_out.lora_A.weight", ctx),
            _adapter_tensor(st, stp + ".attn.to_out.lora_B.weight", ctx), ctx,
        )
        fused += 1
    print("[adapter] fused", fused, "LoRA pairs (W += B@A, in-memory, not baked)")

    return AsymFlux2Klein(model^, proj_buffer^, scale_buffer)


# ─────────────────────────────────────────────────────────────────────────────
# AsymFLUX.2 sqrt dynamic-shift sigma schedule (asymflux2.rs:321-378)
# ─────────────────────────────────────────────────────────────────────────────
def klein_dynamic_shift() -> Float32:
    var seq_len = Float32(H * W)
    var base_shift = Float32(17.0)
    var max_shift = Float32(34.0)
    var base_seq = Float32(1024.0 * 1024.0)
    var max_seq = Float32(2048.0 * 2048.0)
    var m = (max_shift - base_shift) / (fsqrt(max_seq) - fsqrt(base_seq))
    return (fsqrt(seq_len) - fsqrt(base_seq)) * m + base_shift


def compute_sigma_schedule(num_steps: Int, shift: Float32) -> List[Float32]:
    var sigmas = List[Float32]()
    for i in range(num_steps):
        var raw = 1.0 - Float32(i) / Float32(num_steps)
        var s = shift * raw / (1.0 + (shift - 1.0) * raw)
        sigmas.append(s)
    sigmas.append(Float32(0.0))
    return sigmas^


# ─────────────────────────────────────────────────────────────────────────────
# Oklab decode (port of vae/oklab.rs::decode_planar), host-side F32.
# Input planar (3, H*W) normalized Oklab → output planar (3, H*W) sRGB [-1,1].
# ─────────────────────────────────────────────────────────────────────────────
# Lazily-inverted matrices (computed once, see oklab.rs invert_3x3).
def _invert3x3(m: List[Float32]) -> List[Float32]:
    var a = m[0]; var b = m[1]; var c = m[2]
    var d = m[3]; var e = m[4]; var f = m[5]
    var g = m[6]; var h = m[7]; var i = m[8]
    var det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    var inv = 1.0 / det
    var o = List[Float32]()
    o.append((e * i - f * h) * inv)
    o.append((c * h - b * i) * inv)
    o.append((b * f - c * e) * inv)
    o.append((f * g - d * i) * inv)
    o.append((a * i - c * g) * inv)
    o.append((c * d - a * f) * inv)
    o.append((d * h - e * g) * inv)
    o.append((b * g - a * h) * inv)
    o.append((a * e - b * d) * inv)
    return o^


def _lms_to_oklab() -> List[Float32]:
    var m = List[Float32]()
    m.append(0.2104542553); m.append(0.7936177850); m.append(-0.0040720468)
    m.append(1.9779984951); m.append(-2.4285922050); m.append(0.4505937099)
    m.append(0.0259040371); m.append(0.7827717662); m.append(-0.8086757660)
    return m^


def _lrgb_to_lms() -> List[Float32]:
    var m = List[Float32]()
    m.append(0.4122214708); m.append(0.5363325363); m.append(0.0514459929)
    m.append(0.2119034982); m.append(0.6806995451); m.append(0.1073969566)
    m.append(0.0883024619); m.append(0.2817188376); m.append(0.6299787005)
    return m^


def _clamp_nan(x: Float32, lo: Float32, hi: Float32) -> Float32:
    if x != x:
        return x
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def _clamp_lo_nan(x: Float32, lo: Float32) -> Float32:
    if x != x:
        return x
    if x < lo:
        return lo
    return x


def _lrgb_to_srgb_one(l_in: Float32) -> Float32:
    var l = _clamp_lo_nan(l_in, 0.0)
    if l <= 0.0031308:
        return l * 12.92
    return 1.055 * fpow(l, Float32(1.0) / Float32(2.4)) - 0.055


def _srgb_to_lrgb_one(s: Float32) -> Float32:
    if s <= 0.04045:
        return s / 12.92
    return fpow((s + 0.055) / 1.055, Float32(2.4))


def oklab_decode_planar(oklab: List[Float32]) raises -> List[Float32]:
    var n3 = len(oklab)
    var n = n3 // 3
    var m_o2l = _invert3x3(_lms_to_oklab())
    var m_l2r = _invert3x3(_lrgb_to_lms())
    var out = List[Float32](capacity=n3)
    for _ in range(n3):
        out.append(Float32(0.0))
    for p in range(n):
        var l = oklab[p] * AFFINE_STD + AFFINE_MEAN0
        var a = oklab[n + p] * AFFINE_STD + AFFINE_MEAN1
        var bb = oklab[2 * n + p] * AFFINE_STD + AFFINE_MEAN2
        # Oklab → LMS^(1/3)
        var c0 = m_o2l[0] * l + m_o2l[1] * a + m_o2l[2] * bb
        var c1 = m_o2l[3] * l + m_o2l[4] * a + m_o2l[5] * bb
        var c2 = m_o2l[6] * l + m_o2l[7] * a + m_o2l[8] * bb
        # cube
        var lms0 = c0 * c0 * c0
        var lms1 = c1 * c1 * c1
        var lms2 = c2 * c2 * c2
        # LMS → linear RGB
        var lr = m_l2r[0] * lms0 + m_l2r[1] * lms1 + m_l2r[2] * lms2
        var lg = m_l2r[3] * lms0 + m_l2r[4] * lms1 + m_l2r[5] * lms2
        var lb = m_l2r[6] * lms0 + m_l2r[7] * lms1 + m_l2r[8] * lms2
        lr = _clamp_nan(lr, 0.0, 1.0)
        lg = _clamp_nan(lg, 0.0, 1.0)
        lb = _clamp_nan(lb, 0.0, 1.0)
        # linear → sRGB → [-1, 1]
        out[p] = _lrgb_to_srgb_one(lr) * 2.0 - 1.0
        out[n + p] = _lrgb_to_srgb_one(lg) * 2.0 - 1.0
        out[2 * n + p] = _lrgb_to_srgb_one(lb) * 2.0 - 1.0
    return out^


def oklab_encode_planar(pix: List[Float32]) raises -> List[Float32]:
    var n3 = len(pix)
    var n = n3 // 3
    var m_l2lms = _lrgb_to_lms()
    var m_lms2ok = _lms_to_oklab()
    var out = List[Float32](capacity=n3)
    for _ in range(n3):
        out.append(Float32(0.0))
    for p in range(n):
        var rs = pix[p] * 0.5 + 0.5
        var gs = pix[n + p] * 0.5 + 0.5
        var bs = pix[2 * n + p] * 0.5 + 0.5
        var lr = _srgb_to_lrgb_one(rs)
        var lg = _srgb_to_lrgb_one(gs)
        var lb = _srgb_to_lrgb_one(bs)
        var l0 = _clamp_lo_nan(m_l2lms[0] * lr + m_l2lms[1] * lg + m_l2lms[2] * lb, 0.0)
        var l1 = _clamp_lo_nan(m_l2lms[3] * lr + m_l2lms[4] * lg + m_l2lms[5] * lb, 0.0)
        var l2 = _clamp_lo_nan(m_l2lms[6] * lr + m_l2lms[7] * lg + m_l2lms[8] * lb, 0.0)
        var c0 = fcbrt(l0)
        var c1 = fcbrt(l1)
        var c2 = fcbrt(l2)
        var ok0 = m_lms2ok[0] * c0 + m_lms2ok[1] * c1 + m_lms2ok[2] * c2
        var ok1 = m_lms2ok[3] * c0 + m_lms2ok[4] * c1 + m_lms2ok[5] * c2
        var ok2 = m_lms2ok[6] * c0 + m_lms2ok[7] * c1 + m_lms2ok[8] * c2
        out[p] = (ok0 - AFFINE_MEAN0) / AFFINE_STD
        out[n + p] = (ok1 - AFFINE_MEAN1) / AFFINE_STD
        out[2 * n + p] = (ok2 - AFFINE_MEAN2) / AFFINE_STD
    return out^


# ─────────────────────────────────────────────────────────────────────────────
# clamp_denoised_oklab (asymflux2.rs:441-508), host round-trip. x_t,model_output
# are (1,3,H,W) F32. Returns adjusted model_output (1,3,H,W) F32.
# ─────────────────────────────────────────────────────────────────────────────
def clamp_denoised_oklab(
    x_t: Tensor, model_output: Tensor, sigma: Float32, ctx: DeviceContext
) raises -> Tensor:
    var scaled = mul_scalar(model_output, sigma, ctx)
    var denoised = sub(x_t, scaled, ctx)  # (1,3,H,W) F32
    var denoised_host = denoised.to_host(ctx)  # planar 3*H*W (CHW)
    var srgb = oklab_decode_planar(denoised_host)
    # clamp NaN-preserving to [-1,1]
    for i in range(len(srgb)):
        srgb[i] = _clamp_nan(srgb[i], -1.0, 1.0)
    var clipped = oklab_encode_planar(srgb)
    var sh = List[Int]()
    sh.append(1); sh.append(3); sh.append(H); sh.append(W)
    var denoised_clipped = Tensor.from_host(clipped, sh^, STDtype.F32, ctx)
    var diff = sub(x_t, denoised_clipped, ctx)
    var denom = sigma
    if sigma == sigma and sigma < SIGMA_MIN:
        denom = SIGMA_MIN
    return mul_scalar(diff, 1.0 / denom, ctx)


# ─────────────────────────────────────────────────────────────────────────────
# Orthogonal guidance bias (asymflux2.rs:536-617). B=1 → host scalar coef.
# pos,neg,parallel_dir are (1,n,768) F32. Returns bias (same shape) F32.
# ─────────────────────────────────────────────────────────────────────────────
def guidance_bias(
    pos: Tensor, neg: Tensor, gscale: Float32, ortho: Float32,
    parallel_dir: Tensor, ctx: DeviceContext,
) raises -> Tensor:
    var diff = sub(pos, neg, ctx)
    var bias = mul_scalar(diff, gscale - 1.0, ctx)
    if ortho == 0.0:
        return bias^
    # coef = mean(bias*par)/max(mean(par*par),1e-6) * ortho  (host reduce, B=1)
    var bp = mul(bias, parallel_dir, ctx).to_host(ctx)
    var pp = mul(parallel_dir, parallel_dir, ctx).to_host(ctx)
    var sum_bp = Float32(0.0)
    var sum_pp = Float32(0.0)
    var n = len(bp)
    for i in range(n):
        sum_bp += bp[i]
        sum_pp += pp[i]
    var mean_bp = sum_bp / Float32(n)
    var mean_pp = sum_pp / Float32(n)
    var denom = mean_pp
    if mean_pp == mean_pp and mean_pp < 1.0e-6:
        denom = 1.0e-6
    var coef = mean_bp / denom * ortho
    var scaled_par = mul_scalar(parallel_dir, coef, ctx)
    return sub(bias, scaled_par, ctx)


# ─────────────────────────────────────────────────────────────────────────────
# Per-step wrapped forward (asymflux2_klein9b_infer.rs::wrapped_forward).
# x_t_packed is (1, N_IMG, 768) F32. Returns velocity packed (1, N_IMG, 768) F32.
# ─────────────────────────────────────────────────────────────────────────────
def wrapped_forward(
    model: AsymFlux2Klein,
    x_t_packed: Tensor,  # (1, N_IMG, 768) F32
    text_emb: Tensor,
    cos: Tensor,
    sin: Tensor,
    timestep: Float32,
    ctx: DeviceContext,
) raises -> Tensor:
    var cal = compute_calibration(timestep, model.scale_buffer, Float32(1.0))
    # hidden = x_t_packed * k, cast BF16
    var hidden = scale_for_embedder(x_t_packed, cal, ctx)
    var hidden_bf16 = cast_tensor(hidden, STDtype.BF16, ctx)
    # cal_timestep fed to t_embedder (NOT *1000: the asymflux2 ref passes
    # cal_timestep directly; klein.rs time_factor scaling is absorbed because
    # AsymFLUX.2 num_timesteps=1 and the LakonLab pipeline pre-divides).
    var tvals = List[Float32]()
    tvals.append(cal.cal_timestep * 1000.0)
    var tsh = List[Int]()
    tsh.append(1)
    var t_vec = Tensor.from_host(tvals, tsh^, STDtype.F32, ctx)
    var u_a = model.transformer.forward_full[N_IMG, N_TXT, S](
        hidden_bf16, text_emb, t_vec, cos, sin, ctx
    )
    var u_a_f32 = cast_tensor(u_a, STDtype.F32, ctx)
    return asymflow_velocity(
        u_a_f32, x_t_packed, cal, model.proj_buffer, SIGMA_MIN, ctx
    )


# ─────────────────────────────────────────────────────────────────────────────
# Initial pixel noise (1, 3, H, W) BF16 at the tensor boundary.
# ─────────────────────────────────────────────────────────────────────────────
def make_pixel_noise(ctx: DeviceContext) raises -> Tensor:
    var sh = List[Int]()
    sh.append(1); sh.append(3); sh.append(H); sh.append(W)
    return randn(sh^, SEED, STDtype.BF16, ctx)


def sample(
    model: AsymFlux2Klein, caps: KleinCaps, cos: Tensor, sin: Tensor,
    ctx: DeviceContext,
) raises -> Tensor:
    var x_t = make_pixel_noise(ctx)  # (1,3,H,W) BF16 storage
    var shift = klein_dynamic_shift()
    var sigmas = compute_sigma_schedule(STEPS, shift)
    print("[sample] shift", shift, "sigma0", sigmas[0], "->", sigmas[STEPS - 1])
    for step in range(STEPS):
        var sigma_cur = sigmas[step]
        var sigma_next = sigmas[step + 1]
        # patchify+pack pixel → (1, N_IMG, 768) F32
        var x_t_compute = cast_tensor(x_t, STDtype.F32, ctx)
        var x_t_packed = pixel_to_packed(x_t_compute, PATCH, ctx)
        # cond
        var u_cond = wrapped_forward(
            model, x_t_packed, caps.pos, cos, sin, sigma_cur, ctx
        )
        var u_final: Tensor
        if CFG_SCALE != 1.0:
            var u_uncond = wrapped_forward(
                model, x_t_packed, caps.neg, cos, sin, sigma_cur, ctx
            )
            # velocity → pixel for both, build pixel-space denoised parallel dir
            var u_cond_px = velocity_to_pixel(u_cond, H, W, PATCH, ctx)
            var u_uncond_px = velocity_to_pixel(u_uncond, H, W, PATCH, ctx)
            var scaled = mul_scalar(u_cond_px, sigma_cur, ctx)
            var denoised = sub(x_t_compute, scaled, ctx)
            var bias = guidance_bias(
                u_cond_px, u_uncond_px, CFG_SCALE, CFG_ORTHO, denoised, ctx
            )
            u_final = add(u_cond_px, bias, ctx)
        else:
            u_final = velocity_to_pixel(u_cond, H, W, PATCH, ctx)
        # Oklab gamut clamp
        var u_for_step = clamp_denoised_oklab(x_t_compute, u_final, sigma_cur, ctx)
        # Euler: x = x + u * (sigma_next - sigma_cur)
        var dt = sigma_next - sigma_cur
        x_t = cast_tensor(add(x_t_compute, mul_scalar(u_for_step, dt, ctx), ctx), STDtype.BF16, ctx)
        if step == 0 or step == STEPS - 1 or step % 5 == 0:
            print("  step", step + 1, "/", STEPS, "sigma", sigma_cur, "->", sigma_next)
    return x_t^


def main() raises:
    var ctx = DeviceContext()
    print("=== AsymFLUX.2 Klein 9B — pure Mojo (pixel + Oklab + AsymFlow) ===")
    print("  ", W, "x", H, ",", STEPS, "steps, g", CFG_SCALE, "ortho", CFG_ORTHO)
    print("  prompt:", PROMPT)
    var caps = encode_captions(ctx)
    var model = load_adapter(ctx)
    # RoPE tables: N_IMG square grid (32x32), N_TXT=512, 32 heads, head_dim 128.
    var rope = build_klein_rope_tables[N_IMG, N_TXT, 32, 128](ctx, STDtype.BF16)
    var x_final = sample(model, caps, rope[0], rope[1], ctx)
    # Oklab decode (1,3,H,W) → sRGB [-1,1] → PNG
    print("[decode] Oklab → sRGB")
    var oklab_host = x_final.to_host(ctx)
    var srgb = oklab_decode_planar(oklab_host)
    var sh = List[Int]()
    sh.append(1); sh.append(3); sh.append(H); sh.append(W)
    var img = Tensor.from_host(srgb, sh^, STDtype.F32, ctx)
    save_png(img, OUT, ctx, ValueRange.SIGNED)
    print("[done] saved-png", OUT)
