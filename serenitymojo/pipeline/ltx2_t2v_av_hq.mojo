# LTX-2.3 T2V + AUDIO HQ capstone — HIGH-QUALITY recipe upgrade of the P7 MVP.
#
# This is a copy of pipeline/ltx2_t2v_av_mvp.mojo upgraded to the PROVEN HQ
# recipe from the desktop app (archive_pre_lightricks_20260411/{pipeline,config}.py
# + HANDOFF_HQ_SAMPLER_DEBUG.md). The HANDOFF proved on real hardware:
#   "Distilled FP8 + res2s + (LoRA@0.25)  ->  GOOD output (std~1.2, range [-5,5])"
# i.e. the DISTILLED model (the fp8 one we already stream) + the second-order
# Runge-Kutta res_2s sampler + the distilled LoRA at PARTIAL strength is the
# proven-sharp path. The dev model -> garbage (HANDOFF bugs 2/3), so we stay on
# distilled.
#
# UPGRADES vs the MVP (each a quality lever the soft MVP was missing):
#   1. RESOLUTION 768x512 (was 256x256): NH=16, NW=24 -> S_V=NF*16*24.
#   2. FULL 1024-token text/audio context (was a 128-token tail slice): N_TXT=1024,
#      using the entire feature_extract_and_project dump.
#   3. res_2s SECOND-ORDER sampler (was Euler): 15-step LTX2Scheduler with the
#      token-count-dependent sigma shift (ltx-core LTX2Scheduler.execute), TWO
#      model evals per step (current sigma + geometric-mean midpoint sub_sigma).
#      The model is a velocity predictor; res_2s wants a DENOISER, so per eval we
#      convert  denoised = x - v*sigma  (rectified-flow x0 estimate).
#   4. distilled LoRA applied at multiplier=0.25 via the P6 LoraSet RUNTIME ADD
#      (HARD RULE: never fused). 0.25 is the config's hq_distilled_lora_strength_s1
#      ("0.25-0.3 recommended; 1.0 over-sharpens").
#
# CFG NOTE: in the reference, DISTILLED mode runs `simple_denoising_func` — a
# single un-guided forward (guidance_scale=1, no negative context). True CFG-star
# is the DEV path (`multi_modal_guider_denoising_func`, needs a negative-prompt
# Gemma encode we do not have a dump for). We therefore run the proven distilled
# simple-denoise per eval. The cfg_star combine in ltx2_guidance.mojo is wired and
# available, but with only the positive context the honest, proven recipe is the
# un-guided distilled forward + res_2s + LoRA@0.25 — the sampler/res upgrades are
# the dominant sharpness levers the HANDOFF measured.
#
# Run (GPU; FP8 streaming keeps DiT bounded):
#   pixi run mojo run -I . serenitymojo/pipeline/ltx2_t2v_av_hq.mojo

from std.gpu.host import DeviceContext
from std.math import sqrt, cos as fcos, sin as fsin, pow as fpow, log as flog, exp as fexp, pi
from std.memory import alloc, ArcPointer
from sys import argv

from serenitymojo.io.ffi import (
    sys_open, sys_pwrite, sys_close, sys_system,
    O_WRONLY, O_CREAT, O_TRUNC, BytePtr,
)
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.linear import linear
from serenitymojo.ops.norm import layer_norm
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.activations import silu
from serenitymojo.ops.random import randn
from serenitymojo.ops.tensor_algebra import (
    reshape, add, sub, mul, mul_scalar, add_scalar, slice, permute,
)
from serenitymojo.models.dit.ltx2_dit import (
    LTX2Config, LTX2AVBlockWeights, ltx2_block_forward_av,
)
from serenitymojo.models.dit.ltx2_connector import (
    LTX2ConnectorConfig, LTX2ConnectorWeights, ltx2_connector_forward,
)
from serenitymojo.offload.ltx2_block_stream import LTX2BlockStream
from serenitymojo.sampling.ltx2_sampling import (
    LTX2Scheduler, res2s_coefficients, res2s_substep, res2s_combine, Res2sCoeffs,
    ltx2_stage2_distilled_sigmas,
)
from serenitymojo.models.vae.ltx2_vae_decoder import (
    LTX2VaeDecoderWeights, decode as decode_video,
)
from serenitymojo.models.vae.ltx2_audio_vae import (
    LTX2AudioVaeDecoderWeights, decode as decode_audio,
)
from serenitymojo.models.vocoder.ltx2_vocoder import LTX2VocoderWithBWE
from serenitymojo.models.upsampler.ltx2_upsampler import (
    LatentUpsampler, upsample_video,
)
from serenitymojo.image.png import save_png, ValueRange
from serenitymojo.lora import LoraSet, FMT_LTX2_DISTILLED


# ── paths ─────────────────────────────────────────────────────────────────────
comptime CKPT_FP8 = "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-distilled-fp8.safetensors"
comptime CKPT_BF16 = "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-distilled.safetensors"
comptime AUDIO_CTX_DUMP = "/home/alex/EriDiffusion/inference-flame/output/audio_context_dump/ltx2_audio_context.safetensors"
comptime OUT_DIR = "/home/alex/mojodiffusion/output/ltx2_hq"
comptime OUT_DIR2 = "/home/alex/mojodiffusion/output/ltx2_hq2"
comptime SPATIAL_UPSCALER = "/home/alex/.serenity/models/ltx2_upscalers/ltx-2-spatial-upscaler-x2-1.0.safetensors"
comptime LORA_PATH = "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-distilled-lora-384.safetensors"
comptime LORA_MULT = Float32(0.25)   # hq_distilled_lora_strength_s1 (config.py:50)

# ── HQ shape (768x512) ───────────────────────────────────────────────────────
# latent_h = 512//32 = 16 ; latent_w = 768//32 = 24
# NUM_FRAMES=16 -> latent_f = (16-1)//8 + 1 = 2 ; S_V = 2*16*24 = 768
# audio_frames = round(16/25 * 25) = 16 ; S_A = 16
comptime NUM_FRAMES = 16
comptime NF = 2          # latent frames
comptime NH = 16         # latent height (512/32)
comptime NW = 24         # latent width  (768/32)
comptime S_V = NF * NH * NW   # 768
comptime S_A = 16             # audio tokens
comptime N_TXT = 1024         # FULL text-context length (whole dump, no slice)
comptime S_VPAD = 1024        # max(S_V=768, N_TXT=1024, S_A=16)
comptime S_APAD = 1024
comptime NUM_LAYERS = 48

# ── STAGE-2 shape (spatial 2x upsample of stage-1 latent) ──
# Stage-1 latent [1,128,NF,NH,NW] -> upsampler x2 spatial -> [1,128,NF,2NH,2NW].
# NH2=32, NW2=48 => S_V2 = NF*32*48 = 3072. Padding = max(S_V2, N_TXT, S_A).
comptime NH2 = NH * 2        # 32
comptime NW2 = NW * 2        # 48
comptime S_V2 = NF * NH2 * NW2   # 3072
comptime S_VPAD2 = 3072      # max(S_V2=3072, N_TXT=1024, S_A=16)
# Stage-2 output pixel resolution: 1536x1024 (768x512 * 2).
comptime OUT_W2 = 1536
comptime OUT_H2 = 1024

comptime VD = 4096      # video inner_dim
comptime AD = 2048      # audio inner_dim
comptime V_HEADS = 32
comptime V_HDIM = 128
comptime A_HEADS = 32
comptime A_HDIM = 64
comptime CA_DIM = 2048  # audio_cross_attention_dim
comptime EPS = Float32(1e-6)

# config constants (LTX2Config::default)
comptime ROPE_THETA = Float64(10000.0)
comptime POS_EMBED_MAX_POS = Float64(20.0)
comptime BASE_HW = Float64(2048.0)
comptime CAUSAL_OFFSET = Float64(1.0)
comptime VAE_SF0 = Float64(8.0)
comptime VAE_SF1 = Float64(32.0)
comptime VAE_SF2 = Float64(32.0)
comptime AUDIO_SCALE_FACTOR = Float64(4.0)
comptime FRAME_RATE = Float64(25.0)
comptime TS_MULT = Float32(1000.0)
comptime SEED = UInt64(42)

comptime AUDIO_C = 8     # audio latent channels
comptime AUDIO_MEL = 16  # audio latent mel bins (C*F=128)

# res_2s HQ sampler: 15-step LTX2Scheduler (token-shifted sigmas).
comptime HQ_STEPS = 15
# LTX2Scheduler.execute defaults (ltx-core schedulers.py:25-29)
comptime SCHED_MAX_SHIFT = Float64(2.05)
comptime SCHED_BASE_SHIFT = Float64(0.95)
comptime SCHED_TERMINAL = Float64(0.1)
comptime SCHED_BASE_ANCHOR = Float64(1024.0)
comptime SCHED_MAX_ANCHOR = Float64(4096.0)


# ── shape helpers ──────────────────────────────────────────────────────────────
def _sh2(a: Int, b: Int) -> List[Int]:
    var s = List[Int](); s.append(a); s.append(b); return s^

def _sh3(a: Int, b: Int, c: Int) -> List[Int]:
    var s = List[Int](); s.append(a); s.append(b); s.append(c); return s^

def _sh4(a: Int, b: Int, c: Int, d: Int) -> List[Int]:
    var s = List[Int](); s.append(a); s.append(b); s.append(c); s.append(d)
    return s^

def _sh5(a: Int, b: Int, c: Int, d: Int, e: Int) -> List[Int]:
    var s = List[Int](); s.append(a); s.append(b); s.append(c); s.append(d)
    s.append(e); return s^

def _sh1d(a: Int) -> List[Int]:
    var s = List[Int](); s.append(a); return s^


def _is_boundary(i: Int) -> Bool:
    return i == 0 or i == 1 or i == 2 or i == 3 or i == 47


def _st_has(st: ShardedSafeTensors, name: String) -> Bool:
    for ref nm in st.names():
        if nm == name:
            return True
    return False


def _load_global_f32(
    st: ShardedSafeTensors, name: String, ctx: DeviceContext
) raises -> Tensor:
    var key = String("model.diffusion_model.") + name
    if not _st_has(st, key):
        key = name
    var tv = st.tensor_view(key)
    return cast_tensor(Tensor.from_view_as_bf16(tv, ctx), STDtype.F32, ctx)


def _load_global_weights_dict(
    st: ShardedSafeTensors, ctx: DeviceContext
) raises -> Dict[String, ArcPointer[Tensor]]:
    var gw = Dict[String, ArcPointer[Tensor]]()
    var patch_keys = List[String]()
    patch_keys.append(String("patchify_proj.weight"))
    patch_keys.append(String("audio_patchify_proj.weight"))
    patch_keys.append(String("proj_out.weight"))
    patch_keys.append(String("audio_proj_out.weight"))
    var adaln_families = List[String]()
    adaln_families.append(String("adaln_single"))
    adaln_families.append(String("audio_adaln_single"))
    adaln_families.append(String("prompt_adaln_single"))
    adaln_families.append(String("audio_prompt_adaln_single"))
    adaln_families.append(String("av_ca_video_scale_shift_adaln_single"))
    adaln_families.append(String("av_ca_audio_scale_shift_adaln_single"))
    adaln_families.append(String("av_ca_a2v_gate_adaln_single"))
    adaln_families.append(String("av_ca_v2a_gate_adaln_single"))
    for ref fam in patch_keys:
        gw[fam] = ArcPointer[Tensor](_load_global_f32(st, fam, ctx))
    for ref fam in adaln_families:
        var k1 = fam + ".emb.timestep_embedder.linear_1.weight"
        var k2 = fam + ".emb.timestep_embedder.linear_2.weight"
        var k3 = fam + ".linear.weight"
        gw[k1] = ArcPointer[Tensor](_load_global_f32(st, k1, ctx))
        gw[k2] = ArcPointer[Tensor](_load_global_f32(st, k2, ctx))
        gw[k3] = ArcPointer[Tensor](_load_global_f32(st, k3, ctx))
    return gw^


def _clone(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    var dev = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
    ctx.enqueue_copy(dst_buf=dev, src_buf=x.buf)
    ctx.synchronize()
    return Tensor(dev^, x.shape(), x.dtype())


def _linear_b(x: Tensor, w: Tensor, b: Tensor, ctx: DeviceContext) raises -> Tensor:
    return linear(x, w, Optional[Tensor](_clone(b, ctx)), ctx)


def _stats(name: String, t: Tensor, ctx: DeviceContext) raises -> Bool:
    var h = t.to_host(ctx)
    var s = 0.0
    var s2 = 0.0
    var amax = 0.0
    var finite = True
    for i in range(len(h)):
        var v = Float64(h[i])
        if not (v == v) or v > 1.0e38 or v < -1.0e38:
            finite = False
        s += v
        s2 += v * v
        var av = v if v >= 0.0 else -v
        if av > amax:
            amax = av
    var mean = s / Float64(len(h))
    var var_ = s2 / Float64(len(h)) - mean * mean
    if var_ < 0.0:
        var_ = 0.0
    print("   [stat]", name, "mean=", Float32(mean), "std=", Float32(sqrt(var_)),
          "absmax=", Float32(amax), "finite=", finite)
    return finite


# ── HQ LTX2Scheduler sigma schedule (token-count shift) ──
# Port of ltx-core LTX2Scheduler.execute (schedulers.py:21-57) for a velocity
# model. tokens = prod(latent.shape[2:]) = NF*NH*NW = S_V (video). Returns
# HQ_STEPS+1 sigmas descending from 1.0 to 0.0 (trailing 0.0 terminator).
def _ltx2_scheduler_sigmas(steps: Int, tokens: Int) -> List[Float32]:
    var x1 = SCHED_BASE_ANCHOR
    var x2 = SCHED_MAX_ANCHOR
    var mm = (SCHED_MAX_SHIFT - SCHED_BASE_SHIFT) / (x2 - x1)
    var b = SCHED_BASE_SHIFT - mm * x1
    var sigma_shift = Float64(tokens) * mm + b
    var es = fexp(sigma_shift)

    # linspace(1.0, 0.0, steps+1)
    var lin = List[Float64]()
    for i in range(steps + 1):
        lin.append(1.0 - Float64(i) / Float64(steps))

    # power=1: shifted = es / (es + (1/sigma - 1))   (sigma!=0); 0 stays 0.
    var shifted = List[Float64]()
    for i in range(steps + 1):
        var s = lin[i]
        if s != 0.0:
            shifted.append(es / (es + (1.0 / s - 1.0)))
        else:
            shifted.append(0.0)

    # stretch so terminal (last non-zero) matches SCHED_TERMINAL.
    # one_minus_z = 1 - nonzero ; scale = one_minus_z[-1] / (1 - terminal)
    # stretched = 1 - one_minus_z/scale
    var last_nz = -1
    for i in range(steps + 1):
        if shifted[i] != 0.0:
            last_nz = i
    var scale_factor = (1.0 - shifted[last_nz]) / (1.0 - SCHED_TERMINAL)
    var out = List[Float32]()
    for i in range(steps + 1):
        var s = shifted[i]
        if s != 0.0:
            var omz = 1.0 - s
            out.append(Float32(1.0 - omz / scale_factor))
        else:
            out.append(Float32(0.0))
    return out^


# ── sinusoidal timestep embedding (diffusers get_timestep_embedding) ──
def _timestep_embedding(ts: List[Float32], dim: Int, ctx: DeviceContext) raises -> Tensor:
    var n = len(ts)
    var half = dim // 2
    var out = List[Float32]()
    out.resize(n * dim, Float32(0.0))
    for r in range(n):
        var t = Float64(ts[r])
        for i in range(half):
            var freq = fpow(Float64(2.718281828459045), -Float64(i) * flog(10000.0) / Float64(half))
            var arg = t * freq
            out[r * dim + i] = Float32(fcos(arg))
            out[r * dim + half + i] = Float32(fsin(arg))
    return Tensor.from_host(out, _sh2(n, dim), STDtype.F32, ctx)


def _adaln_single(
    st: ShardedSafeTensors,
    gw: Dict[String, ArcPointer[Tensor]],
    base: String,
    ts_vals: List[Float32],
    ctx: DeviceContext,
) raises -> Tuple[Tensor, Tensor]:
    var emb = _timestep_embedding(ts_vals, 256, ctx)
    var w1 = _clone(gw[base + ".emb.timestep_embedder.linear_1.weight"][], ctx)
    var b1 = _load_global_f32(st, base + ".emb.timestep_embedder.linear_1.bias", ctx)
    var h = _linear_b(emb, w1, b1, ctx)
    h = silu(h, ctx)
    var w2 = _clone(gw[base + ".emb.timestep_embedder.linear_2.weight"][], ctx)
    var b2 = _load_global_f32(st, base + ".emb.timestep_embedder.linear_2.bias", ctx)
    var embedded = _linear_b(h, w2, b2, ctx)
    var h2 = silu(embedded, ctx)
    var lw = _clone(gw[base + ".linear.weight"][], ctx)
    var lb = _load_global_f32(st, base + ".linear.bias", ctx)
    var mod = _linear_b(h2, lw, lb, ctx)
    return (mod^, embedded^)


def _compute_rope(
    coords: List[Float64],
    num_pos_dims: Int,
    P: Int,
    dim: Int,
    max_pos: List[Float64],
    theta: Float64,
    num_heads: Int,
    ctx: DeviceContext,
) raises -> Tuple[Tensor, Tensor]:
    var num_rope_elems = num_pos_dims * 2
    var freq_count = dim // num_rope_elems
    var half_dim = dim // 2
    var rope_freqs = freq_count * num_pos_dims
    var pad = half_dim - rope_freqs
    if pad < 0:
        raise Error("rope: rope_freqs > half_dim")

    var denom = Float64(freq_count - 1)
    if denom < 1.0:
        denom = 1.0
    var freq = List[Float64]()
    for i in range(freq_count):
        freq.append(fpow(theta, Float64(i) / denom) * pi / 2.0)

    var cos_tok = List[Float32]()
    var sin_tok = List[Float32]()
    cos_tok.resize(P * half_dim, Float32(0.0))
    sin_tok.resize(P * half_dim, Float32(0.0))
    var half_head = (dim // 2) // num_heads

    for p in range(P):
        var scaled = List[Float64]()
        for d in range(num_pos_dims):
            var s0 = coords[(d * P + p) * 2 + 0]
            var e0 = coords[(d * P + p) * 2 + 1]
            var mid = (s0 + e0) * 0.5
            var g = mid / max_pos[d]
            scaled.append(2.0 * g - 1.0)
        var base = p * half_dim
        for q in range(pad):
            cos_tok[base + q] = Float32(1.0)
            sin_tok[base + q] = Float32(0.0)
        var off = base + pad
        for i in range(freq_count):
            for d in range(num_pos_dims):
                var a = scaled[d] * freq[i]
                cos_tok[off] = Float32(fcos(a))
                sin_tok[off] = Float32(fsin(a))
                off += 1

    var cos_rows = List[Float32]()
    var sin_rows = List[Float32]()
    for p in range(P):
        for h in range(num_heads):
            var src = p * half_dim + h * half_head
            for j in range(half_head):
                cos_rows.append(cos_tok[src + j])
                sin_rows.append(sin_tok[src + j])
    var sh = _sh2(P * num_heads, half_head)
    var cos_t = Tensor.from_host(cos_rows, sh.copy(), STDtype.F32, ctx)
    var sin_t = Tensor.from_host(sin_rows, sh^, STDtype.F32, ctx)
    return (cos_t^, sin_t^)


def _build_video_coords_dims(nf: Int, nh: Int, nw: Int) -> List[Float64]:
    """Video 3D RoPE coord boxes for an arbitrary latent grid (nf,nh,nw).

    The H/W coordinate scales (VAE_SF1/2 = 32) are PIXEL-space step sizes per
    latent cell and DO NOT change with resolution — stage-2 has more, finer
    cells but the same per-cell pixel extent, exactly matching the reference
    VideoLatentShape positions builder. So we just iterate the new grid.
    """
    var s_v = nf * nh * nw
    var out = List[Float64]()
    out.resize(3 * s_v * 2, Float64(0.0))
    var vae_t = VAE_SF0
    for f in range(nf):
        for h in range(nh):
            for w in range(nw):
                var tok = f * nh * nw + h * nw + w
                var fs = Float64(f) * VAE_SF0
                var fe = Float64(f + 1) * VAE_SF0
                var fsc = fs + CAUSAL_OFFSET - vae_t
                var fec = fe + CAUSAL_OFFSET - vae_t
                if fsc < 0.0: fsc = 0.0
                if fec < 0.0: fec = 0.0
                fsc = fsc / FRAME_RATE
                fec = fec / FRAME_RATE
                out[(0 * s_v + tok) * 2 + 0] = fsc
                out[(0 * s_v + tok) * 2 + 1] = fec
                out[(1 * s_v + tok) * 2 + 0] = Float64(h) * VAE_SF1
                out[(1 * s_v + tok) * 2 + 1] = Float64(h + 1) * VAE_SF1
                out[(2 * s_v + tok) * 2 + 0] = Float64(w) * VAE_SF2
                out[(2 * s_v + tok) * 2 + 1] = Float64(w + 1) * VAE_SF2
    return out^


def _build_video_coords() -> List[Float64]:
    return _build_video_coords_dims(NF, NH, NW)


def _build_audio_coords() -> List[Float64]:
    var out = List[Float64]()
    out.resize(S_A * 2, Float64(0.0))
    var mel_to_sec = 16000.0 / 160.0
    var scale = AUDIO_SCALE_FACTOR
    for t in range(S_A):
        var ms = Float64(t) * scale
        var me = Float64(t + 1) * scale
        var msc = ms + CAUSAL_OFFSET - scale
        var mec = me + CAUSAL_OFFSET - scale
        if msc < 0.0: msc = 0.0
        if mec < 0.0: mec = 0.0
        out[t * 2 + 0] = msc / mel_to_sec
        out[t * 2 + 1] = mec / mel_to_sec
    return out^


def _video_temporal_coords_dims(vc: List[Float64], s_v: Int) -> List[Float64]:
    var out = List[Float64]()
    out.resize(s_v * 2, Float64(0.0))
    for p in range(s_v):
        out[p * 2 + 0] = vc[(0 * s_v + p) * 2 + 0]
        out[p * 2 + 1] = vc[(0 * s_v + p) * 2 + 1]
    return out^


def _video_temporal_coords(vc: List[Float64]) -> List[Float64]:
    return _video_temporal_coords_dims(vc, S_V)


def _mp1() -> List[Float64]:
    var m = List[Float64](); m.append(POS_EMBED_MAX_POS); return m^

def _mp3() -> List[Float64]:
    var m = List[Float64](); m.append(POS_EMBED_MAX_POS); m.append(BASE_HW)
    m.append(BASE_HW); return m^


struct _Globals(Movable):
    var v_pin_w: Tensor
    var v_pin_b: Tensor
    var a_pin_w: Tensor
    var a_pin_b: Tensor
    var v_pout_w: Tensor
    var v_pout_b: Tensor
    var a_pout_w: Tensor
    var a_pout_b: Tensor
    var v_sst: Tensor
    var a_sst: Tensor

    def __init__(out self, var v_pin_w: Tensor, var v_pin_b: Tensor,
                 var a_pin_w: Tensor, var a_pin_b: Tensor,
                 var v_pout_w: Tensor, var v_pout_b: Tensor,
                 var a_pout_w: Tensor, var a_pout_b: Tensor,
                 var v_sst: Tensor, var a_sst: Tensor):
        self.v_pin_w = v_pin_w^; self.v_pin_b = v_pin_b^
        self.a_pin_w = a_pin_w^; self.a_pin_b = a_pin_b^
        self.v_pout_w = v_pout_w^; self.v_pout_b = v_pout_b^
        self.a_pout_w = a_pout_w^; self.a_pout_b = a_pout_b^
        self.v_sst = v_sst^; self.a_sst = a_sst^


def _output_stage(
    hs: Tensor, sst: Tensor, embedded: Tensor,
    proj_w: Tensor, proj_b: Tensor, dim: Int, ctx: DeviceContext,
) raises -> Tensor:
    var ones = List[Float32](); var zeros = List[Float32]()
    for _ in range(dim):
        ones.append(Float32(1.0)); zeros.append(Float32(0.0))
    var w_ln = Tensor.from_host(ones, _sh1d(dim), STDtype.F32, ctx)
    var b_ln = Tensor.from_host(zeros, _sh1d(dim), STDtype.F32, ctx)
    var normed = layer_norm(hs, w_ln, b_ln, EPS, ctx)
    var shift_row = reshape(slice(sst, 0, 0, 1, ctx), _sh3(1, 1, dim), ctx)
    var scale_row = reshape(slice(sst, 0, 1, 1, ctx), _sh3(1, 1, dim), ctx)
    var v_shift = add(shift_row, embedded, ctx)
    var v_scale = add(scale_row, embedded, ctx)
    var one_plus = add_scalar(v_scale, Float32(1.0), ctx)
    var out = add(mul(normed, one_plus, ctx), v_shift, ctx)
    return _linear_b(out, proj_w, proj_b, ctx)


struct _Mod(Movable):
    var v_temb: Tensor
    var a_temb: Tensor
    var v_embedded: Tensor
    var a_embedded: Tensor
    var v_ca_ss: Tensor
    var a_ca_ss: Tensor
    var v_ca_gate: Tensor
    var a_ca_gate: Tensor
    var v_prompt_ts: Tensor
    var a_prompt_ts: Tensor

    def __init__(out self, var v_temb: Tensor, var a_temb: Tensor,
                 var v_embedded: Tensor, var a_embedded: Tensor,
                 var v_ca_ss: Tensor, var a_ca_ss: Tensor,
                 var v_ca_gate: Tensor, var a_ca_gate: Tensor,
                 var v_prompt_ts: Tensor, var a_prompt_ts: Tensor):
        self.v_temb = v_temb^; self.a_temb = a_temb^
        self.v_embedded = v_embedded^; self.a_embedded = a_embedded^
        self.v_ca_ss = v_ca_ss^; self.a_ca_ss = a_ca_ss^
        self.v_ca_gate = v_ca_gate^; self.a_ca_gate = a_ca_gate^
        self.v_prompt_ts = v_prompt_ts^; self.a_prompt_ts = a_prompt_ts^


def _build_mod_dims(
    st: ShardedSafeTensors,
    gw: Dict[String, ArcPointer[Tensor]],
    sigma: Float32,
    s_v: Int,
    ctx: DeviceContext,
) raises -> _Mod:
    var ts_v = List[Float32]()
    for _ in range(s_v): ts_v.append(sigma * TS_MULT)
    var vt = _adaln_single(st, gw, String("adaln_single"), ts_v, ctx)
    var v_temb = reshape(vt[0], _sh3(1, s_v, 9 * VD), ctx)
    var v_embedded = reshape(vt[1], _sh3(1, s_v, VD), ctx)

    var ts_a = List[Float32]()
    for _ in range(S_A): ts_a.append(sigma * TS_MULT)
    var at = _adaln_single(st, gw, String("audio_adaln_single"), ts_a, ctx)
    var a_temb = reshape(at[0], _sh3(1, S_A, 9 * AD), ctx)
    var a_embedded = reshape(at[1], _sh3(1, S_A, AD), ctx)

    var g1 = List[Float32](); g1.append(sigma * TS_MULT)
    var vcs = _adaln_single(st, gw, String("av_ca_video_scale_shift_adaln_single"), g1, ctx)
    var v_ca_ss = reshape(vcs[0], _sh3(1, 1, 4 * VD), ctx)
    var acs = _adaln_single(st, gw, String("av_ca_audio_scale_shift_adaln_single"), g1, ctx)
    var a_ca_ss = reshape(acs[0], _sh3(1, 1, 4 * AD), ctx)
    var vcg = _adaln_single(st, gw, String("av_ca_a2v_gate_adaln_single"), g1, ctx)
    var v_ca_gate = reshape(vcg[0], _sh3(1, 1, VD), ctx)
    var acg = _adaln_single(st, gw, String("av_ca_v2a_gate_adaln_single"), g1, ctx)
    var a_ca_gate = reshape(acg[0], _sh3(1, 1, AD), ctx)

    var ts_p = List[Float32]()
    for _ in range(N_TXT): ts_p.append(sigma * TS_MULT)
    var vpt = _adaln_single(st, gw, String("prompt_adaln_single"), ts_p, ctx)
    var v_prompt_ts = reshape(vpt[0], _sh3(1, N_TXT, 2 * VD), ctx)
    var apt = _adaln_single(st, gw, String("audio_prompt_adaln_single"), ts_p, ctx)
    var a_prompt_ts = reshape(apt[0], _sh3(1, N_TXT, 2 * AD), ctx)

    return _Mod(v_temb^, a_temb^, v_embedded^, a_embedded^, v_ca_ss^, a_ca_ss^,
                v_ca_gate^, a_ca_gate^, v_prompt_ts^, a_prompt_ts^)


def _build_mod(
    st: ShardedSafeTensors,
    gw: Dict[String, ArcPointer[Tensor]],
    sigma: Float32,
    ctx: DeviceContext,
) raises -> _Mod:
    return _build_mod_dims(st, gw, sigma, S_V, ctx)


# ── ONE model forward at a given sigma. Returns velocity (video, audio) in the
#    LATENT layout: video [1,128,NF,NH,NW], audio [1,8,S_A,16]. This is the same
#    forward the MVP did per step, factored out so res_2s can call it twice/step
#    (current sigma + midpoint sub_sigma). All inputs (contexts/rope/globals) are
#    sigma-independent and passed in; only the modulation rebuilds per sigma.    ──
def _model_forward_p[
    S_V_CT: Int, S_VPAD_CT: Int
](
    ck: ShardedSafeTensors,
    gw: Dict[String, ArcPointer[Tensor]],
    lora: LoraSet,
    apply_lora: Bool,
    cfg: LTX2Config,
    g: _Globals,
    stream: LTX2BlockStream,
    video_x: Tensor,
    audio_x: Tensor,
    enc: Tensor, aenc: Tensor,
    v_cos: Tensor, v_sin: Tensor, a_cos: Tensor, a_sin: Tensor,
    ca_v_cos: Tensor, ca_v_sin: Tensor, ca_a_cos: Tensor, ca_a_sin: Tensor,
    sigma: Float32,
    nf: Int, nh: Int, nw: Int,
    ctx: DeviceContext,
) raises -> Tuple[Tensor, Tensor]:
    """ONE DiT forward at `sigma`. Comptime params are the (token-count, pad)
    needed by the block kernel; runtime nf/nh/nw drive the (un)patchify reshape
    and the per-sigma modulation. Stage 1 uses [S_V,S_VPAD]; stage 2 (post 2x
    spatial upsample) uses [S_V2,S_VPAD2] with nh/nw doubled."""
    # patchify
    var v_flat = permute(reshape(video_x, _sh3(1, 128, S_V_CT), ctx),
                         _video_perm(), ctx)
    var a_flat = reshape(permute(audio_x, _audio_perm(), ctx),
                         _sh3(1, S_A, 128), ctx)
    var hs = _linear_b(v_flat, g.v_pin_w, g.v_pin_b, ctx)
    var ahs = _linear_b(a_flat, g.a_pin_w, g.a_pin_b, ctx)

    var mod = _build_mod_dims(ck, gw, sigma, S_V_CT, ctx)

    for i in range(NUM_LAYERS):
        var w: LTX2AVBlockWeights
        if _is_boundary(i):
            w = LTX2AVBlockWeights.load(String(CKPT_FP8), i, cfg, ctx).to_f32(ctx)
        else:
            var blk = stream.load_block_bf16(i, ctx)
            w = LTX2AVBlockWeights.from_fp8_block(blk^, cfg, ctx).to_f32(ctx)
        if apply_lora:
            _ = lora.apply_to_av_block(i, w, LORA_MULT, ctx)
        var outs = ltx2_block_forward_av[S_V_CT, S_A, N_TXT, S_VPAD_CT, S_APAD](
            w, hs, ahs, enc, aenc,
            mod.v_temb, mod.a_temb, mod.v_ca_ss, mod.a_ca_ss,
            mod.v_ca_gate, mod.a_ca_gate,
            mod.v_prompt_ts, mod.a_prompt_ts,
            v_cos, v_sin, a_cos, a_sin,
            ca_v_cos, ca_v_sin, ca_a_cos, ca_a_sin, EPS, ctx,
        )
        hs = _clone(outs[0], ctx)
        ahs = _clone(outs[1], ctx)

    var v_vel_flat = _output_stage(hs, g.v_sst, mod.v_embedded, g.v_pout_w,
                                   g.v_pout_b, VD, ctx)
    var a_vel_flat = _output_stage(ahs, g.a_sst, mod.a_embedded, g.a_pout_w,
                                   g.a_pout_b, AD, ctx)
    var v_vel = reshape(permute(v_vel_flat, _unvideo_perm(), ctx),
                        _sh5(1, 128, nf, nh, nw), ctx)
    var a_vel = permute(reshape(a_vel_flat, _sh4(1, S_A, AUDIO_C, AUDIO_MEL), ctx),
                        _unaudio_perm(), ctx)
    return (v_vel^, a_vel^)


def _model_forward(
    ck: ShardedSafeTensors,
    gw: Dict[String, ArcPointer[Tensor]],
    lora: LoraSet,
    apply_lora: Bool,
    cfg: LTX2Config,
    g: _Globals,
    stream: LTX2BlockStream,
    video_x: Tensor,
    audio_x: Tensor,
    enc: Tensor, aenc: Tensor,
    v_cos: Tensor, v_sin: Tensor, a_cos: Tensor, a_sin: Tensor,
    ca_v_cos: Tensor, ca_v_sin: Tensor, ca_a_cos: Tensor, ca_a_sin: Tensor,
    sigma: Float32,
    ctx: DeviceContext,
) raises -> Tuple[Tensor, Tensor]:
    return _model_forward_p[S_V, S_VPAD](
        ck, gw, lora, apply_lora, cfg, g, stream, video_x, audio_x, enc, aenc,
        v_cos, v_sin, a_cos, a_sin, ca_v_cos, ca_v_sin, ca_a_cos, ca_a_sin,
        sigma, NF, NH, NW, ctx,
    )


# denoised x0 estimate from a velocity prediction (rectified flow): x - v*sigma.
def _denoise_from_vel(x: Tensor, vel: Tensor, sigma: Float32, ctx: DeviceContext) raises -> Tensor:
    return sub(x, mul_scalar(vel, sigma, ctx), ctx)


# NCDHW [1,128,F,H,W] -> NDHWC [1,F,H,W,128]  (perm 0,2,3,4,1).
def _ncdhw_to_ndhwc(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    var p = List[Int]()
    p.append(0); p.append(2); p.append(3); p.append(4); p.append(1)
    return permute(x, p^, ctx)


# NDHWC [1,F,H,W,128] -> NCDHW [1,128,F,H,W]  (perm 0,4,1,2,3).
def _ndhwc_to_ncdhw(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    var p = List[Int]()
    p.append(0); p.append(4); p.append(1); p.append(2); p.append(3)
    return permute(x, p^, ctx)


# Load a VAE per-channel-stat [128] from the bf16 checkpoint as F32 device tensor.
def _load_vae_stat(name: String, ctx: DeviceContext) raises -> Tensor:
    var st = ShardedSafeTensors.open(String(CKPT_BF16))
    var tv = st.tensor_view(String("vae.per_channel_statistics.") + name)
    return cast_tensor(Tensor.from_view_as_bf16(tv, ctx), STDtype.F32, ctx)


# Stage-2 forward-noise the upscaled latent at noise_scale (GaussianNoiser, mask=1):
#   x = noise * noise_scale + upscaled * (1 - noise_scale)
# (ltx-core components/noisers.py GaussianNoiser.__call__, denoise_mask all-ones.)
def _noise_init(
    upscaled: Tensor, noise_scale: Float32, seed: UInt64, ctx: DeviceContext
) raises -> Tensor:
    var noise = randn(upscaled.shape().copy(), seed, STDtype.F32, ctx)
    var a = mul_scalar(noise, noise_scale, ctx)
    var b = mul_scalar(upscaled, Float32(1.0) - noise_scale, ctx)
    return add(a, b, ctx)


def run(apply_lora: Bool, out_dir: String, max_steps: Int) raises:
    var ctx = DeviceContext()
    var cfg = LTX2Config.ltx2()
    print("=== LTX-2.3 T2V+AUDIO HQ (res_2s) — 768x512 / 16f distilled ===")
    print("  res:768x512  NF/NH/NW:", NF, NH, NW, " S_V:", S_V, " S_A:", S_A,
          " N_TXT:", N_TXT, " blocks:", NUM_LAYERS)
    print("  sampler: res_2s (2nd-order RK)  steps:", HQ_STEPS,
          "  LoRA:", "ON @0.25" if apply_lora else "OFF", " out_dir:", out_dir)

    var ck = ShardedSafeTensors.open(String(CKPT_FP8))

    var lora = LoraSet.load(String(LORA_PATH))
    if apply_lora:
        if lora.format != FMT_LTX2_DISTILLED:
            raise Error("HQ: LoRA not FMT_LTX2_DISTILLED")
        print("  [lora] loaded", lora.num_mappings(),
              "mappings (", lora.format_name(), ") @ mult", LORA_MULT)

    print("  [load] globals")
    var gw = _load_global_weights_dict(ck, ctx)
    if apply_lora:
        var n_global = lora.apply_to_globals(gw, LORA_MULT, ctx)
        print("  [lora] global deltas applied (one-time additive @0.25):", n_global)
    var g = _Globals(
        _clone(gw[String("patchify_proj.weight")][], ctx),
        _load_global_f32(ck, "patchify_proj.bias", ctx),
        _clone(gw[String("audio_patchify_proj.weight")][], ctx),
        _load_global_f32(ck, "audio_patchify_proj.bias", ctx),
        _clone(gw[String("proj_out.weight")][], ctx),
        _load_global_f32(ck, "proj_out.bias", ctx),
        _clone(gw[String("audio_proj_out.weight")][], ctx),
        _load_global_f32(ck, "audio_proj_out.bias", ctx),
        _load_global_f32(ck, "scale_shift_table", ctx),
        _load_global_f32(ck, "audio_scale_shift_table", ctx),
    )

    print("  [connector] loading + running video/audio (F32)")
    var v_conn = LTX2ConnectorWeights.load(
        String(CKPT_FP8), String("video_embeddings_connector"),
        LTX2ConnectorConfig.video(), ctx,
    )
    var a_conn = LTX2ConnectorWeights.load(
        String(CKPT_FP8), String("audio_embeddings_connector"),
        LTX2ConnectorConfig.audio(), ctx,
    )

    # FULL 1024-token contexts (no tail slice): use the entire dump.
    var dump = ShardedSafeTensors.open(String(AUDIO_CTX_DUMP))
    var video_pre = cast_tensor(
        Tensor.from_view_as_bf16(dump.tensor_view("video_context"), ctx),
        STDtype.F32, ctx,
    )  # [1,1024,4096]
    var audio_pre = cast_tensor(
        Tensor.from_view_as_bf16(dump.tensor_view("audio_context"), ctx),
        STDtype.F32, ctx,
    )  # [1,1024,2048]
    _ = _stats(String("video_pre(FULL 1024)"), video_pre, ctx)
    _ = _stats(String("audio_pre(FULL 1024 REAL)"), audio_pre, ctx)

    var enc = ltx2_connector_forward[N_TXT, V_HEADS, V_HDIM](v_conn, video_pre, ctx)
    var aenc = ltx2_connector_forward[N_TXT, A_HEADS, A_HDIM](a_conn, audio_pre, ctx)
    _ = _stats(String("enc"), enc, ctx)
    _ = _stats(String("aenc"), aenc, ctx)

    print("  [rope] building 3D video + 1D audio + temporal cross tables")
    var vc = _build_video_coords()
    var ac = _build_audio_coords()
    var vtc = _video_temporal_coords(vc)
    var vrope = _compute_rope(vc, 3, S_V, VD, _mp3(), ROPE_THETA, V_HEADS, ctx)
    var arope = _compute_rope(ac, 1, S_A, AD, _mp1(), ROPE_THETA, A_HEADS, ctx)
    var cavrope = _compute_rope(vtc, 1, S_V, CA_DIM, _mp1(), ROPE_THETA, V_HEADS, ctx)
    var caarope = _compute_rope(ac, 1, S_A, CA_DIM, _mp1(), ROPE_THETA, A_HEADS, ctx)
    var v_cos = _clone(vrope[0], ctx); var v_sin = _clone(vrope[1], ctx)
    var a_cos = _clone(arope[0], ctx); var a_sin = _clone(arope[1], ctx)
    var ca_v_cos = _clone(cavrope[0], ctx); var ca_v_sin = _clone(cavrope[1], ctx)
    var ca_a_cos = _clone(caarope[0], ctx); var ca_a_sin = _clone(caarope[1], ctx)

    var stream = LTX2BlockStream.open(String(CKPT_FP8))
    if stream.block_count() != NUM_LAYERS:
        raise Error("stream block_count != 48")

    print("  [noise] init video + audio latents")
    var video_x = randn(_sh5(1, 128, NF, NH, NW), SEED, STDtype.F32, ctx)
    var audio_x = randn(_sh4(1, AUDIO_C, S_A, AUDIO_MEL), SEED + 1, STDtype.F32, ctx)

    # ── HQ res_2s sampler: 15-step token-shifted LTX2Scheduler ──
    var sig_list = _ltx2_scheduler_sigmas(HQ_STEPS, S_V)
    var sched = LTX2Scheduler(sig_list.copy())
    var sigmas = sched.sigmas()
    print("  [sampler] LTX2Scheduler token-shifted sigmas (", len(sigmas), "):", end="")
    for i in range(len(sigmas)):
        print(" ", sigmas[i], end="")
    print("")

    var n_steps = sched.num_steps
    if max_steps > 0 and max_steps < n_steps:
        n_steps = max_steps

    for step in range(n_steps):
        var sigma = sigmas[step]
        var sigma_next = sigmas[step + 1]
        print("  --- step", step + 1, "/", sched.num_steps, " sigma=", sigma,
              " -> ", sigma_next, "---")

        if sigma_next == 0.0:
            # Final step: res_2s returns the denoised estimate (single eval).
            var f1 = _model_forward(
                ck, gw, lora, apply_lora, cfg, g, stream,
                video_x, audio_x, enc, aenc,
                v_cos, v_sin, a_cos, a_sin,
                ca_v_cos, ca_v_sin, ca_a_cos, ca_a_sin, sigma, ctx,
            )
            video_x = _denoise_from_vel(video_x, f1[0], sigma, ctx)
            audio_x = _denoise_from_vel(audio_x, f1[1], sigma, ctx)
            _ = _stats(String("video_x(final denoise)"), video_x, ctx)
            _ = _stats(String("audio_x(final denoise)"), audio_x, ctx)
            continue

        var c = res2s_coefficients(sigma, sigma_next)  # h, a21, b1, b2, sub_sigma

        # STAGE 1 — model @ current sigma. denoised_1 = x - v*sigma.
        var s1 = _model_forward(
            ck, gw, lora, apply_lora, cfg, g, stream,
            video_x, audio_x, enc, aenc,
            v_cos, v_sin, a_cos, a_sin,
            ca_v_cos, ca_v_sin, ca_a_cos, ca_a_sin, sigma, ctx,
        )
        var v_den1 = _denoise_from_vel(video_x, s1[0], sigma, ctx)
        var a_den1 = _denoise_from_vel(audio_x, s1[1], sigma, ctx)

        # midpoint sample: x_mid = x + h*a21*(denoised_1 - x)
        var v_mid = res2s_substep(video_x, v_den1, c.h, c.a21, ctx)
        var a_mid = res2s_substep(audio_x, a_den1, c.h, c.a21, ctx)

        # STAGE 2 — model @ geometric-mean midpoint sigma.
        var s2 = _model_forward(
            ck, gw, lora, apply_lora, cfg, g, stream,
            v_mid, a_mid, enc, aenc,
            v_cos, v_sin, a_cos, a_sin,
            ca_v_cos, ca_v_sin, ca_a_cos, ca_a_sin, c.sub_sigma, ctx,
        )
        # denoised_2 = x_mid - v*sub_sigma, but residual is vs the ANCHOR x:
        #   eps_2 = denoised_2 - x  (res2s_combine subtracts x internally)
        var v_den2 = _denoise_from_vel(v_mid, s2[0], c.sub_sigma, ctx)
        var a_den2 = _denoise_from_vel(a_mid, s2[1], c.sub_sigma, ctx)

        # COMBINE: x_next = x + h*(b1*(den1-x) + b2*(den2-x))
        video_x = res2s_combine(video_x, v_den1, v_den2, c.h, c.b1, c.b2, ctx)
        audio_x = res2s_combine(audio_x, a_den1, a_den2, c.h, c.b1, c.b2, ctx)
        _ = _stats(String("video_x"), video_x, ctx)
        _ = _stats(String("audio_x"), audio_x, ctx)

    print("  [denoise] done -> decoding")

    # ── DECODE VIDEO ──
    print("  [decode] video VAE (latent -> frames)")
    var vae = LTX2VaeDecoderWeights.load(String(CKPT_BF16), ctx)
    var video_x_bf16 = cast_tensor(video_x, STDtype.BF16, ctx)
    var frames = decode_video[1, 128, NF, NH, NW](vae, video_x_bf16, ctx)
    var fsh = frames.shape()
    var n_frames_out = fsh[2]
    print("  frames NCDHW: [", fsh[0], ",", fsh[1], ",", fsh[2], ",", fsh[3],
          ",", fsh[4], "]")
    _ = _stats(String("frames"), frames, ctx)

    var png_paths = List[String]()
    for fr in range(n_frames_out):
        var fslice = slice(frames, 2, fr, 1, ctx)
        var fs = fslice.shape()
        var chw = reshape(fslice, _sh4(fs[0], fs[1], fs[3], fs[4]), ctx)
        var p = out_dir + "/hq_frame" + _pad2(fr) + ".png"
        save_png(chw, p, ctx, ValueRange.SIGNED)
        png_paths.append(p)
    print("  saved", n_frames_out, "frame PNGs ->", out_dir)

    if max_steps != 0:
        print("  [decode] audio SKIPPED (fast gen-gate run); frames saved ->",
              out_dir)
        print("=== HQ (fast gen-gate) DONE ===")
        return

    var wav_out = out_dir + "/hq_audio.wav"
    var mp4_out = out_dir + "/ltx2_t2v_av_hq.mp4"
    print("  [decode] audio VAE (latent -> mel) -> vocoder (mel -> 48kHz wav)")
    var avae = LTX2AudioVaeDecoderWeights.load(String(CKPT_BF16), ctx)
    var audio_x_bf16 = cast_tensor(audio_x, STDtype.BF16, ctx)
    var mel_raw = decode_audio(avae, audio_x_bf16, ctx)
    var mel = cast_tensor(mel_raw, STDtype.F32, ctx)
    var msh = mel.shape()
    print("  mel NCHW: [", msh[0], ",", msh[1], ",", msh[2], ",", msh[3], "] dtype-cast F32")
    _ = _stats(String("mel"), mel, ctx)

    var voc = LTX2VocoderWithBWE.from_file(String(CKPT_BF16), ctx)
    var wav = voc.forward(mel, ctx)
    var wsh = wav.shape()
    var L = wsh[2]
    print("  wav [", wsh[0], ",", wsh[1], ",", wsh[2], "] @", voc.output_sample_rate(), "Hz")

    var wh = wav.to_host(ctx)
    var rms = 0.0
    var amax = 0.0
    var bad = False
    for i in range(len(wh)):
        var v = Float64(wh[i])
        if not (v == v): bad = True
        rms += v * v
        var av = v if v >= 0.0 else -v
        if av > amax: amax = av
    rms = sqrt(rms / Float64(len(wh)))
    print("  AUDIO rms=", Float32(rms), " absmax=", Float32(amax), " finite=",
          not bad, " nonsilent=", rms > 1.0e-4)

    var inter = List[Float32]()
    inter.resize(L * 2, Float32(0.0))
    for ch in range(2):
        for s in range(L):
            inter[s * 2 + ch] = wh[ch * L + s]
    _write_wav(wav_out, inter, voc.output_sample_rate())
    print("  wrote wav:", wav_out)

    print("  [mux] ffmpeg frames + wav -> mp4")
    _mux_mp4(out_dir, n_frames_out, wav_out, mp4_out)
    print("=== HQ DONE ===")
    print("  mp4:", mp4_out)
    print("  wav:", wav_out)
    print("  frames:", out_dir, "/hq_frame00.png ..")


# ════════════════════════════════════════════════════════════════════════════
# FULL STAGED HQ generate() — faithful to pipeline.py:1139-1337.
#   Stage 1 : LTX2Scheduler 15-step token-shifted sigmas + res_2s denoise.
#   UPSCALE : un_normalize -> LatentUpsampler(spatial x2) -> normalize.
#   Stage 2 : forward-noise upscaled @ s2_sigmas[0]=0.909375, then 3-step res_2s
#             refine (STAGE_2_DISTILLED_SIGMA_VALUES) at the 2x latent grid.
#   decode video (2x res) + audio VAE -> vocoder, mux to mp4.
#
# NAG: pipeline.py applies NAGPatch (null-Gemma-encoding baseline) around BOTH
# stages. serenitymojo HAS the NAG combine + ltx2_block_forward_av_nag wired and
# gated, but the NULL-context input is the Gemma encoding of the empty string ""
# (nag.py / pipeline.py:466) — and NO such null-encoding dump exists on disk
# (only the positive video/audio context dump). NAG is therefore reported as
# blocked-on-input here; the dominant sharpness levers (the previously-MISSING
# spatial upscale + stage-2 refine) ARE wired and run. Audio is single-stage
# (refined only by the video-coupled stage-2; the reference carries s1 audio
# into stage 2 unchanged when no audio refine).
# ════════════════════════════════════════════════════════════════════════════
def run_staged(apply_lora: Bool, out_dir: String, max_steps: Int) raises:
    var ctx = DeviceContext()
    var cfg = LTX2Config.ltx2()
    print("=== LTX-2.3 T2V+AUDIO HQ *FULL STAGED* (res_2s + 2x upscale + refine) ===")
    print("  Stage1 res:768x512  NF/NH/NW:", NF, NH, NW, " S_V:", S_V)
    print("  Stage2 res:", OUT_W2, "x", OUT_H2, " NF/NH2/NW2:", NF, NH2, NW2,
          " S_V2:", S_V2)
    print("  sampler: res_2s  Stage1 steps:", HQ_STEPS,
          "  Stage2 steps: 3 (STAGE_2_DISTILLED_SIGMA_VALUES)")
    print("  LoRA:", "ON @0.25" if apply_lora else "OFF", " out_dir:", out_dir)

    var ck = ShardedSafeTensors.open(String(CKPT_FP8))

    var lora = LoraSet.load(String(LORA_PATH))
    if apply_lora:
        if lora.format != FMT_LTX2_DISTILLED:
            raise Error("HQ: LoRA not FMT_LTX2_DISTILLED")
        print("  [lora] loaded", lora.num_mappings(),
              "mappings (", lora.format_name(), ") @ mult", LORA_MULT)

    print("  [load] globals")
    var gw = _load_global_weights_dict(ck, ctx)
    if apply_lora:
        var n_global = lora.apply_to_globals(gw, LORA_MULT, ctx)
        print("  [lora] global deltas applied (additive @0.25):", n_global)
    var g = _Globals(
        _clone(gw[String("patchify_proj.weight")][], ctx),
        _load_global_f32(ck, "patchify_proj.bias", ctx),
        _clone(gw[String("audio_patchify_proj.weight")][], ctx),
        _load_global_f32(ck, "audio_patchify_proj.bias", ctx),
        _clone(gw[String("proj_out.weight")][], ctx),
        _load_global_f32(ck, "proj_out.bias", ctx),
        _clone(gw[String("audio_proj_out.weight")][], ctx),
        _load_global_f32(ck, "audio_proj_out.bias", ctx),
        _load_global_f32(ck, "scale_shift_table", ctx),
        _load_global_f32(ck, "audio_scale_shift_table", ctx),
    )

    print("  [connector] loading + running video/audio (F32)")
    var v_conn = LTX2ConnectorWeights.load(
        String(CKPT_FP8), String("video_embeddings_connector"),
        LTX2ConnectorConfig.video(), ctx,
    )
    var a_conn = LTX2ConnectorWeights.load(
        String(CKPT_FP8), String("audio_embeddings_connector"),
        LTX2ConnectorConfig.audio(), ctx,
    )
    var dump = ShardedSafeTensors.open(String(AUDIO_CTX_DUMP))
    var video_pre = cast_tensor(
        Tensor.from_view_as_bf16(dump.tensor_view("video_context"), ctx),
        STDtype.F32, ctx,
    )
    var audio_pre = cast_tensor(
        Tensor.from_view_as_bf16(dump.tensor_view("audio_context"), ctx),
        STDtype.F32, ctx,
    )
    var enc = ltx2_connector_forward[N_TXT, V_HEADS, V_HDIM](v_conn, video_pre, ctx)
    var aenc = ltx2_connector_forward[N_TXT, A_HEADS, A_HDIM](a_conn, audio_pre, ctx)

    var stream = LTX2BlockStream.open(String(CKPT_FP8))
    if stream.block_count() != NUM_LAYERS:
        raise Error("stream block_count != 48")

    # ── Audio RoPE (sigma-independent, same both stages) ──
    var ac = _build_audio_coords()
    var arope = _compute_rope(ac, 1, S_A, AD, _mp1(), ROPE_THETA, A_HEADS, ctx)
    var caarope = _compute_rope(ac, 1, S_A, CA_DIM, _mp1(), ROPE_THETA, A_HEADS, ctx)
    var a_cos = _clone(arope[0], ctx); var a_sin = _clone(arope[1], ctx)
    var ca_a_cos = _clone(caarope[0], ctx); var ca_a_sin = _clone(caarope[1], ctx)

    # ── STAGE-1 RoPE (NF,NH,NW) ──
    print("  [rope] stage-1 video tables (S_V=", S_V, ")")
    var vc1 = _build_video_coords_dims(NF, NH, NW)
    var vtc1 = _video_temporal_coords_dims(vc1, S_V)
    var vrope1 = _compute_rope(vc1, 3, S_V, VD, _mp3(), ROPE_THETA, V_HEADS, ctx)
    var cavrope1 = _compute_rope(vtc1, 1, S_V, CA_DIM, _mp1(), ROPE_THETA, V_HEADS, ctx)
    var v_cos1 = _clone(vrope1[0], ctx); var v_sin1 = _clone(vrope1[1], ctx)
    var ca_v_cos1 = _clone(cavrope1[0], ctx); var ca_v_sin1 = _clone(cavrope1[1], ctx)

    print("  [noise] init video + audio latents")
    var video_x = randn(_sh5(1, 128, NF, NH, NW), SEED, STDtype.F32, ctx)
    var audio_x = randn(_sh4(1, AUDIO_C, S_A, AUDIO_MEL), SEED + 1, STDtype.F32, ctx)

    var sig1 = _ltx2_scheduler_sigmas(HQ_STEPS, S_V)
    var sched1 = LTX2Scheduler(sig1.copy())
    var sigmas1 = sched1.sigmas()
    var n1 = sched1.num_steps
    if max_steps > 0 and max_steps < n1:
        n1 = max_steps
    print("  [Stage1] res_2s denoise,", n1, "steps")

    for step in range(n1):
        var sigma = sigmas1[step]
        var sigma_next = sigmas1[step + 1]
        print("  S1 step", step + 1, "/", sched1.num_steps, " sigma=", sigma, "->", sigma_next)
        if sigma_next == 0.0:
            var f1 = _model_forward_p[S_V, S_VPAD](
                ck, gw, lora, apply_lora, cfg, g, stream, video_x, audio_x, enc, aenc,
                v_cos1, v_sin1, a_cos, a_sin, ca_v_cos1, ca_v_sin1, ca_a_cos, ca_a_sin,
                sigma, NF, NH, NW, ctx,
            )
            video_x = _denoise_from_vel(video_x, f1[0], sigma, ctx)
            audio_x = _denoise_from_vel(audio_x, f1[1], sigma, ctx)
            continue
        var c = res2s_coefficients(sigma, sigma_next)
        var s1a = _model_forward_p[S_V, S_VPAD](
            ck, gw, lora, apply_lora, cfg, g, stream, video_x, audio_x, enc, aenc,
            v_cos1, v_sin1, a_cos, a_sin, ca_v_cos1, ca_v_sin1, ca_a_cos, ca_a_sin,
            sigma, NF, NH, NW, ctx,
        )
        var v_den1 = _denoise_from_vel(video_x, s1a[0], sigma, ctx)
        var a_den1 = _denoise_from_vel(audio_x, s1a[1], sigma, ctx)
        var v_mid = res2s_substep(video_x, v_den1, c.h, c.a21, ctx)
        var a_mid = res2s_substep(audio_x, a_den1, c.h, c.a21, ctx)
        var s2a = _model_forward_p[S_V, S_VPAD](
            ck, gw, lora, apply_lora, cfg, g, stream, v_mid, a_mid, enc, aenc,
            v_cos1, v_sin1, a_cos, a_sin, ca_v_cos1, ca_v_sin1, ca_a_cos, ca_a_sin,
            c.sub_sigma, NF, NH, NW, ctx,
        )
        var v_den2 = _denoise_from_vel(v_mid, s2a[0], c.sub_sigma, ctx)
        var a_den2 = _denoise_from_vel(a_mid, s2a[1], c.sub_sigma, ctx)
        video_x = res2s_combine(video_x, v_den1, v_den2, c.h, c.b1, c.b2, ctx)
        audio_x = res2s_combine(audio_x, a_den1, a_den2, c.h, c.b1, c.b2, ctx)
    _ = _stats(String("video_x(stage1)"), video_x, ctx)
    print("  [Stage1] done")

    # ══════════════ SPATIAL UPSAMPLE (x2) ══════════════
    print("  [upscale] loading spatial-x2 LatentUpsampler + VAE per-channel stats")
    var up_st = ShardedSafeTensors.open(String(SPATIAL_UPSCALER))
    var upsampler = LatentUpsampler(up_st, False, ctx)  # is_temporal=False
    var std_t = _load_vae_stat(String("std-of-means"), ctx)
    var mean_t = _load_vae_stat(String("mean-of-means"), ctx)
    # NCDHW [1,128,NF,NH,NW] -> NDHWC, upsample (un_normalize/normalize wrapped),
    # back to NCDHW [1,128,NF,2NH,2NW].
    var v_ndhwc = _ncdhw_to_ndhwc(video_x, ctx)
    var up_ndhwc = upsample_video(v_ndhwc, std_t, mean_t, upsampler, ctx)
    var upscaled = _ndhwc_to_ncdhw(up_ndhwc, ctx)
    var ush = upscaled.shape()
    print("  [upscale] upscaled latent NCDHW: [", ush[0], ",", ush[1], ",",
          ush[2], ",", ush[3], ",", ush[4], "]  (expect [1,128,", NF, ",", NH2,
          ",", NW2, "])")
    _ = _stats(String("upscaled"), upscaled, ctx)

    # ══════════════ STAGE-2 RoPE (doubled spatial grid) ══════════════
    print("  [rope] stage-2 video tables (S_V2=", S_V2, ")")
    var vc2 = _build_video_coords_dims(NF, NH2, NW2)
    var vtc2 = _video_temporal_coords_dims(vc2, S_V2)
    var vrope2 = _compute_rope(vc2, 3, S_V2, VD, _mp3(), ROPE_THETA, V_HEADS, ctx)
    var cavrope2 = _compute_rope(vtc2, 1, S_V2, CA_DIM, _mp1(), ROPE_THETA, V_HEADS, ctx)
    var v_cos2 = _clone(vrope2[0], ctx); var v_sin2 = _clone(vrope2[1], ctx)
    var ca_v_cos2 = _clone(cavrope2[0], ctx); var ca_v_sin2 = _clone(cavrope2[1], ctx)

    # ══════════════ STAGE-2 refine (3-step res_2s) ══════════════
    var s2sig = ltx2_stage2_distilled_sigmas()  # [0.909375, 0.725, 0.421875, 0.0]
    var s2_noise_scale = s2sig[0]
    print("  [Stage2] forward-noise upscaled @", s2_noise_scale, " then 3-step res_2s refine (joint A/V)")
    # Forward-noise BOTH video (upscaled) and audio (stage-1 result) at noise_scale.
    # GaussianNoiser, denoise_mask=1: x = noise*scale + init*(1-scale). (Reference
    # ti2vid_two_stages_hq.py passes both initial_video_latent + initial_audio_latent.)
    var vx2 = _noise_init(upscaled, s2_noise_scale, SEED + 100, ctx)
    var ax2 = _noise_init(audio_x, s2_noise_scale, SEED + 101, ctx)
    var sched2 = LTX2Scheduler(s2sig.copy())
    var sigmas2 = sched2.sigmas()
    var n2 = sched2.num_steps

    for step in range(n2):
        var sigma = sigmas2[step]
        var sigma_next = sigmas2[step + 1]
        print("  S2 step", step + 1, "/", n2, " sigma=", sigma, "->", sigma_next)
        if sigma_next == 0.0:
            var f1 = _model_forward_p[S_V2, S_VPAD2](
                ck, gw, lora, apply_lora, cfg, g, stream, vx2, ax2, enc, aenc,
                v_cos2, v_sin2, a_cos, a_sin, ca_v_cos2, ca_v_sin2, ca_a_cos, ca_a_sin,
                sigma, NF, NH2, NW2, ctx,
            )
            vx2 = _denoise_from_vel(vx2, f1[0], sigma, ctx)
            ax2 = _denoise_from_vel(ax2, f1[1], sigma, ctx)
            continue
        var c = res2s_coefficients(sigma, sigma_next)
        var s1a = _model_forward_p[S_V2, S_VPAD2](
            ck, gw, lora, apply_lora, cfg, g, stream, vx2, ax2, enc, aenc,
            v_cos2, v_sin2, a_cos, a_sin, ca_v_cos2, ca_v_sin2, ca_a_cos, ca_a_sin,
            sigma, NF, NH2, NW2, ctx,
        )
        var v_den1 = _denoise_from_vel(vx2, s1a[0], sigma, ctx)
        var a_den1 = _denoise_from_vel(ax2, s1a[1], sigma, ctx)
        var v_mid = res2s_substep(vx2, v_den1, c.h, c.a21, ctx)
        var a_mid = res2s_substep(ax2, a_den1, c.h, c.a21, ctx)
        var s2a = _model_forward_p[S_V2, S_VPAD2](
            ck, gw, lora, apply_lora, cfg, g, stream, v_mid, a_mid, enc, aenc,
            v_cos2, v_sin2, a_cos, a_sin, ca_v_cos2, ca_v_sin2, ca_a_cos, ca_a_sin,
            c.sub_sigma, NF, NH2, NW2, ctx,
        )
        var v_den2 = _denoise_from_vel(v_mid, s2a[0], c.sub_sigma, ctx)
        var a_den2 = _denoise_from_vel(a_mid, s2a[1], c.sub_sigma, ctx)
        vx2 = res2s_combine(vx2, v_den1, v_den2, c.h, c.b1, c.b2, ctx)
        ax2 = res2s_combine(ax2, a_den1, a_den2, c.h, c.b1, c.b2, ctx)
    _ = _stats(String("vx2(stage2)"), vx2, ctx)
    print("  [Stage2] done -> decoding at 2x resolution")

    # ══════════════ DECODE VIDEO (2x res) ══════════════
    var vae = LTX2VaeDecoderWeights.load(String(CKPT_BF16), ctx)
    var vx2_bf16 = cast_tensor(vx2, STDtype.BF16, ctx)
    var frames = decode_video[1, 128, NF, NH2, NW2](vae, vx2_bf16, ctx)
    var fsh = frames.shape()
    var n_frames_out = fsh[2]
    print("  frames NCDHW: [", fsh[0], ",", fsh[1], ",", fsh[2], ",", fsh[3],
          ",", fsh[4], "]")
    _ = _stats(String("frames"), frames, ctx)

    for fr in range(n_frames_out):
        var fslice = slice(frames, 2, fr, 1, ctx)
        var fs = fslice.shape()
        var chw = reshape(fslice, _sh4(fs[0], fs[1], fs[3], fs[4]), ctx)
        save_png(chw, out_dir + "/hq_frame" + _pad2(fr) + ".png", ctx, ValueRange.SIGNED)
    print("  saved", n_frames_out, "frame PNGs ->", out_dir)

    if max_steps != 0:
        print("  [decode] audio SKIPPED (fast gate); frames saved ->", out_dir)
        print("=== HQ STAGED (fast gate) DONE ===")
        return

    # ══════════════ AUDIO + MUX ══════════════
    var wav_out = out_dir + "/hq_audio.wav"
    var mp4_out = out_dir + "/ltx2_t2v_av_hq2.mp4"
    print("  [decode] audio VAE -> vocoder")
    var avae = LTX2AudioVaeDecoderWeights.load(String(CKPT_BF16), ctx)
    var audio_x_bf16 = cast_tensor(ax2, STDtype.BF16, ctx)
    var mel = cast_tensor(decode_audio(avae, audio_x_bf16, ctx), STDtype.F32, ctx)
    var voc = LTX2VocoderWithBWE.from_file(String(CKPT_BF16), ctx)
    var wav = voc.forward(mel, ctx)
    var wsh = wav.shape()
    var L = wsh[2]
    var wh = wav.to_host(ctx)
    var inter = List[Float32]()
    inter.resize(L * 2, Float32(0.0))
    for ch in range(2):
        for s in range(L):
            inter[s * 2 + ch] = wh[ch * L + s]
    _write_wav(wav_out, inter, voc.output_sample_rate())
    print("  wrote wav:", wav_out)
    _mux_mp4(out_dir, n_frames_out, wav_out, mp4_out)
    print("=== HQ STAGED DONE ===")
    print("  mp4:", mp4_out)
    print("  frames:", out_dir, "/hq_frame00.png ..  (", OUT_W2, "x", OUT_H2, ")")


def _mkdir(path: String) raises:
    var cmd = String("mkdir -p ") + path + " >/dev/null 2>&1"
    _ = sys_system(cmd)


# ── argv-driven entry ──
#   hq [base|lora] [out_dir] [max_steps]            -> single-pass (legacy)
#   hq staged [base|lora] [out_dir] [max_steps]     -> FULL staged (upscale+refine)
def main() raises:
    var a = argv()
    var staged = len(a) >= 2 and String(a[1]) == "staged"
    var argbase = 2 if staged else 1
    var apply_lora = False
    var max_steps = 0
    if len(a) >= argbase + 1 and String(a[argbase]) == "lora":
        apply_lora = True
    var out_dir = String(OUT_DIR2) if staged else String(OUT_DIR)
    if len(a) >= argbase + 2:
        out_dir = String(a[argbase + 1])
    if len(a) >= argbase + 3:
        max_steps = atol(String(a[argbase + 2]))
    _mkdir(out_dir)
    if staged:
        run_staged(apply_lora, out_dir, max_steps)
    else:
        run(apply_lora, out_dir, max_steps)


# ── patchify/unpatchify permutations ──
def _video_perm() -> List[Int]:
    var p = List[Int](); p.append(0); p.append(2); p.append(1); return p^

def _unvideo_perm() -> List[Int]:
    var p = List[Int](); p.append(0); p.append(2); p.append(1); return p^

def _audio_perm() -> List[Int]:
    var p = List[Int](); p.append(0); p.append(2); p.append(1); p.append(3)
    return p^

def _unaudio_perm() -> List[Int]:
    var p = List[Int](); p.append(0); p.append(2); p.append(1); p.append(3)
    return p^


def _pad2(n: Int) -> String:
    if n < 10:
        return String("0") + String(n)
    return String(n)


# ── WAV writer (16-bit PCM LE stereo) ──
def _write_wav(path: String, samples: List[Float32], sr: Int) raises:
    var n = len(samples)
    var channels = 2
    var bits = 16
    var byte_rate = sr * channels * (bits // 8)
    var block_align = channels * (bits // 8)
    var data_bytes = n * (bits // 8)
    var total = 44 + data_bytes
    var buf = alloc[UInt8](total)

    var riff = String("RIFF")
    for i in range(4): buf[i] = UInt8(ord(riff[byte=i]))
    var v0 = 36 + data_bytes
    buf[4] = UInt8(v0 & 0xFF); buf[5] = UInt8((v0 >> 8) & 0xFF)
    buf[6] = UInt8((v0 >> 16) & 0xFF); buf[7] = UInt8((v0 >> 24) & 0xFF)
    var wave = String("WAVE")
    for i in range(4): buf[8 + i] = UInt8(ord(wave[byte=i]))
    var fmt = String("fmt ")
    for i in range(4): buf[12 + i] = UInt8(ord(fmt[byte=i]))
    buf[16] = 16; buf[17] = 0; buf[18] = 0; buf[19] = 0
    buf[20] = 1; buf[21] = 0
    buf[22] = UInt8(channels & 0xFF); buf[23] = UInt8((channels >> 8) & 0xFF)
    buf[24] = UInt8(sr & 0xFF); buf[25] = UInt8((sr >> 8) & 0xFF)
    buf[26] = UInt8((sr >> 16) & 0xFF); buf[27] = UInt8((sr >> 24) & 0xFF)
    buf[28] = UInt8(byte_rate & 0xFF); buf[29] = UInt8((byte_rate >> 8) & 0xFF)
    buf[30] = UInt8((byte_rate >> 16) & 0xFF); buf[31] = UInt8((byte_rate >> 24) & 0xFF)
    buf[32] = UInt8(block_align & 0xFF); buf[33] = UInt8((block_align >> 8) & 0xFF)
    buf[34] = UInt8(bits & 0xFF); buf[35] = UInt8((bits >> 8) & 0xFF)
    var data = String("data")
    for i in range(4): buf[36 + i] = UInt8(ord(data[byte=i]))
    buf[40] = UInt8(data_bytes & 0xFF); buf[41] = UInt8((data_bytes >> 8) & 0xFF)
    buf[42] = UInt8((data_bytes >> 16) & 0xFF); buf[43] = UInt8((data_bytes >> 24) & 0xFF)

    for i in range(n):
        var v = samples[i]
        if v < Float32(-1.0): v = Float32(-1.0)
        elif v > Float32(1.0): v = Float32(1.0)
        var s16 = Int(v * Float32(32767.0))
        if s16 < -32768: s16 = -32768
        elif s16 > 32767: s16 = 32767
        var u = s16 if s16 >= 0 else (s16 + 65536)
        var off = 44 + i * 2
        buf[off] = UInt8(u & 0xFF)
        buf[off + 1] = UInt8((u >> 8) & 0xFF)

    var fd = sys_open(path, O_WRONLY | O_CREAT | O_TRUNC, Int32(0o644))
    if fd < 0:
        buf.free()
        raise Error(String("write_wav: cannot open ") + path)
    var bp = BytePtr(unsafe_from_address=Int(buf))
    var done = 0
    while done < total:
        var got = sys_pwrite(fd, bp + done, total - done, done)
        if got <= 0:
            break
        done += got
    _ = sys_close(fd)
    buf.free()
    if done != total:
        raise Error("write_wav: short write")


# ── ffmpeg mux via system() ──
def _mux_mp4(out_dir: String, n_frames: Int, wav: String, mp4: String) raises:
    var cmd = String("ffmpeg -y -framerate 25 -i ")
    cmd += out_dir + "/hq_frame%02d.png -i " + wav
    cmd += " -c:v libx264 -pix_fmt yuv420p -c:a aac -shortest " + mp4
    cmd += " >/dev/null 2>&1"
    var rc = sys_system(cmd)
    if rc != 0:
        print("  [mux] WARNING: ffmpeg returned", rc, "(frames+wav still saved)")
