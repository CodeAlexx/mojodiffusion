# LTX-2.3 T2V + AUDIO MVP capstone (Plan P7) — end-to-end, GPU.
#
# Wires the full two-stage* text-to-video-with-audio pipeline at MVP size
# (256x256, 16 input frames -> 9 decoded frames) by porting the model-level
# `LTX2StreamingModel::forward_audio_video_inner` (ltx2_model.rs:4453-5040 /
# scripts/ltx2_dit_forward_parity_ref.py) into Mojo and driving it with the
# distilled Euler sampler (sampling/ltx2_sampling.mojo), then decoding video
# (models/vae/ltx2_vae_decoder) + audio (models/vae/ltx2_audio_vae ->
# models/vocoder LTX2VocoderWithBWE) and muxing via ffmpeg.
#
# What this file ADDS vs the verified components:
#   * the in-Mojo per-step MODULATION builders (adaln_single MLP: ts*1000 ->
#     sinusoidal(256) -> linear_1 -> silu -> linear_2 = embedded; silu ->
#     linear = temb) for video/audio adaln, the 4 av_ca AdaLN, and the two
#     prompt_adaln — exactly the tensors the forward smoke loaded from the
#     oracle. RoPE tables are rebuilt from build_video/audio_coords per the
#     model RoPE (vae_scale_factors / causal_offset / frame_rate scaling),
#     NOT the unit-patch build_ltx2_rope.
#   * the two-stage distilled denoise loop (8 distilled sigmas: the table
#     already descends through 0.909375 -> 0.725 -> 0.421875 -> 0, i.e. the
#     "stage-2" sigmas ARE the tail of the distilled-8 table; the dedicated
#     latent spatial-x2 upsampler + AdaIN stage boundary is NOT built, so this
#     MVP runs the single distilled-8 schedule at fixed MVP resolution — see
#     STAGE-BOUNDARY NOTE below).
#
# CONTEXT NOTE (audio): the cached embed sidecar has only the VIDEO context
# (`text_hidden [1,1024,4096]`, post-feature-extractor, pre-connector). There
# is no cached AUDIO pre-connector context and no in-Mojo Gemma encoder, so the
# 2048-dim audio pre-connector context is DERIVED deterministically from the
# cached video hidden (down-projected slice). Audio is therefore AV-coupled
# through the joint denoise and finite/non-silent, but NOT prompt-faithful.
# Reported honestly.
#
# Run (GPU; FP8 streaming keeps DiT ~12 GB):
#   pixi run mojo run -I . serenitymojo/pipeline/ltx2_t2v_av_mvp.mojo

from std.gpu.host import DeviceContext
from std.math import sqrt, cos as fcos, sin as fsin, pow as fpow, log as flog, pi
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
    ltx2_distilled_sigmas, LTX2Scheduler,
)
from serenitymojo.models.vae.ltx2_vae_decoder import (
    LTX2VaeDecoderWeights, decode as decode_video,
)
from serenitymojo.models.vae.ltx2_audio_vae import (
    LTX2AudioVaeDecoderWeights, decode as decode_audio,
)
from serenitymojo.models.vocoder.ltx2_vocoder import LTX2VocoderWithBWE
from serenitymojo.image.png import save_png, ValueRange
from serenitymojo.lora import LoraSet, FMT_LTX2_DISTILLED


# ── paths ─────────────────────────────────────────────────────────────────────
comptime CKPT_FP8 = "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-distilled-fp8.safetensors"
comptime CKPT_BF16 = "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-distilled.safetensors"
comptime CACHED = "/home/alex/EriDiffusion/inference-flame/cached_ltx2_embeddings.safetensors"
comptime OUT_DIR = "/home/alex/mojodiffusion/output/ltx2_mvp"
comptime MP4_OUT = OUT_DIR + "/ltx2_t2v_av_256_16f.mp4"
comptime WAV_OUT = OUT_DIR + "/mvp_audio.wav"
comptime LORA_PATH = "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-distilled-lora-384.safetensors"
comptime LORA_MULT = Float32(1.0)

# ── MVP shape (256x256, 16 input frames) ───────────────────────────────────────
# latent_f = (num_frames-1)//8 + 1 = (16-1)//8 + 1 = 2  (Rust ltx2_generate_av.rs:213)
# latent_h = 256//32 = 8 ; latent_w = 256//32 = 8 ; S_V = 2*8*8 = 128
# audio_frames = round(num_frames/frame_rate * 25) = round(16/25*25) = 16 ; S_A = 16
comptime NUM_FRAMES = 16
comptime NF = 2          # latent frames
comptime NH = 8          # latent height
comptime NW = 8          # latent width
comptime S_V = NF * NH * NW   # 128
comptime S_A = 16             # audio tokens
comptime N_TXT = 128          # text-context length (bounded slice of cached 1024)
comptime S_VPAD = 128         # max(S_V, N_TXT, S_A)
comptime S_APAD = 128
comptime NUM_LAYERS = 48

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
    """Pre-load the 28 global LoRA target weights into a resident dict so that
    `LoraSet.apply_to_globals` can apply the one-time additive deltas before the
    denoise loop.  Keys match the LoRA base_key convention (no leading
    `transformer_blocks.`, just the bare module path + `.weight`)."""
    var gw = Dict[String, ArcPointer[Tensor]]()
    # 4 patchify / proj_out projections
    var patch_keys = List[String]()
    patch_keys.append(String("patchify_proj.weight"))
    patch_keys.append(String("audio_patchify_proj.weight"))
    patch_keys.append(String("proj_out.weight"))
    patch_keys.append(String("audio_proj_out.weight"))
    # 24 adaln linears: 8 families × 3 weights each
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


# ── sinusoidal timestep embedding (diffusers get_timestep_embedding,
#    flip_sin_to_cos=True, shift=0) — built on host for a [N] timestep vector.
#    Returns [N, 256] F32 device tensor. (scripts/...ref.py timestep_embedding) ──
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
            out[r * dim + i] = Float32(fcos(arg))          # cos first
            out[r * dim + half + i] = Float32(fsin(arg))   # then sin
    return Tensor.from_host(out, _sh2(n, dim), STDtype.F32, ctx)


# ── AdaLayerNormSingle.forward (ltx2_model.rs:574-582) ──
#    embedded = linear_2(silu(linear_1(sinusoidal(ts*?))))   [N, dim]
#    mod      = linear(silu(embedded))                        [N, n*dim]
# `ts_vals` are the ALREADY-SCALED timesteps (sigma * TS_MULT * extra).
# `gw` holds the pre-loaded (and LoRA-applied) global weight tensors so the
# 28 adaln LoRA deltas are honoured every forward pass via the resident weights.
# Biases are NOT LoRA targets and are still loaded fresh from `st`.
def _adaln_single(
    st: ShardedSafeTensors,
    gw: Dict[String, ArcPointer[Tensor]],
    base: String,        # e.g. "adaln_single"
    ts_vals: List[Float32],
    ctx: DeviceContext,
) raises -> Tuple[Tensor, Tensor]:
    var emb = _timestep_embedding(ts_vals, 256, ctx)  # [N,256] F32
    # Weight tensors come from the pre-loaded (LoRA-applied) global dict;
    # _clone gives an owned copy so _linear_b can consume them.
    var w1 = _clone(gw[base + ".emb.timestep_embedder.linear_1.weight"][], ctx)
    var b1 = _load_global_f32(st, base + ".emb.timestep_embedder.linear_1.bias", ctx)
    var h = _linear_b(emb, w1, b1, ctx)
    h = silu(h, ctx)
    var w2 = _clone(gw[base + ".emb.timestep_embedder.linear_2.weight"][], ctx)
    var b2 = _load_global_f32(st, base + ".emb.timestep_embedder.linear_2.bias", ctx)
    var embedded = _linear_b(h, w2, b2, ctx)  # [N,dim]
    var h2 = silu(embedded, ctx)
    var lw = _clone(gw[base + ".linear.weight"][], ctx)
    var lb = _load_global_f32(st, base + ".linear.bias", ctx)
    var mod = _linear_b(h2, lw, lb, ctx)      # [N, n*dim]
    return (mod^, embedded^)


# ── RoPE: port of compute_rope_frequencies for general coords.
#    coords_host is [num_pos_dims, P, 2] (start,end) F64. Returns the table in
#    (s,h) row order [P*num_heads, head_dim/2] F32, ready for apply_ltx2_rope. ──
def _compute_rope(
    coords: List[Float64],   # [num_pos_dims * P * 2] row-major [d,p,2]
    num_pos_dims: Int,
    P: Int,
    dim: Int,
    max_pos: List[Float64],  # [num_pos_dims]
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

    # per-token half-vector [P, half_dim]
    var cos_tok = List[Float32]()
    var sin_tok = List[Float32]()
    cos_tok.resize(P * half_dim, Float32(0.0))
    sin_tok.resize(P * half_dim, Float32(0.0))
    var half_head = (dim // 2) // num_heads

    for p in range(P):
        # grid[d] = midpoint / max_pos[d] ; scaled = 2*grid-1
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

    # relayout half_dim -> [num_heads, head_dim/2], emit (token,head) rows
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


def _build_video_coords() -> List[Float64]:
    # [3, S_V, 2] row-major
    var out = List[Float64]()
    out.resize(3 * S_V * 2, Float64(0.0))
    var vae_t = VAE_SF0
    for f in range(NF):
        for h in range(NH):
            for w in range(NW):
                var tok = f * NH * NW + h * NW + w
                var fs = Float64(f) * VAE_SF0
                var fe = Float64(f + 1) * VAE_SF0
                var fsc = fs + CAUSAL_OFFSET - vae_t
                var fec = fe + CAUSAL_OFFSET - vae_t
                if fsc < 0.0: fsc = 0.0
                if fec < 0.0: fec = 0.0
                fsc = fsc / FRAME_RATE
                fec = fec / FRAME_RATE
                out[(0 * S_V + tok) * 2 + 0] = fsc
                out[(0 * S_V + tok) * 2 + 1] = fec
                out[(1 * S_V + tok) * 2 + 0] = Float64(h) * VAE_SF1
                out[(1 * S_V + tok) * 2 + 1] = Float64(h + 1) * VAE_SF1
                out[(2 * S_V + tok) * 2 + 0] = Float64(w) * VAE_SF2
                out[(2 * S_V + tok) * 2 + 1] = Float64(w + 1) * VAE_SF2
    return out^


def _build_audio_coords() -> List[Float64]:
    # [1, S_A, 2]
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


def _video_temporal_coords(vc: List[Float64]) -> List[Float64]:
    # extract dim 0 (temporal) -> [1, S_V, 2]
    var out = List[Float64]()
    out.resize(S_V * 2, Float64(0.0))
    for p in range(S_V):
        out[p * 2 + 0] = vc[(0 * S_V + p) * 2 + 0]
        out[p * 2 + 1] = vc[(0 * S_V + p) * 2 + 1]
    return out^


def _mp1() -> List[Float64]:
    var m = List[Float64](); m.append(POS_EMBED_MAX_POS); return m^

def _mp3() -> List[Float64]:
    var m = List[Float64](); m.append(POS_EMBED_MAX_POS); m.append(BASE_HW)
    m.append(BASE_HW); return m^


# ── one model-level forward_audio_video (single sigma). Returns
#    (video_velocity [1,128,NF,NH,NW], audio_velocity [1,8,S_A,16]).
#    Reuses the loaded contexts + globals + the 48 streamed blocks.            ──
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


def _sh1d(a: Int) -> List[Int]:
    var s = List[Int](); s.append(a); return s^


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


# Build all per-forward modulation tensors from one sigma.
# `gw` holds the 28 global (and LoRA-applied) weight tensors so adaln MLP
# weights reflect the global LoRA delta applied once at load.
def _build_mod(
    st: ShardedSafeTensors,
    gw: Dict[String, ArcPointer[Tensor]],
    sigma: Float32,
    ctx: DeviceContext,
) raises -> _Mod:
    var ts_v = List[Float32]()
    for _ in range(S_V): ts_v.append(sigma * TS_MULT)
    var vt = _adaln_single(st, gw, String("adaln_single"), ts_v, ctx)
    var v_temb = reshape(vt[0], _sh3(1, S_V, 9 * VD), ctx)
    var v_embedded = reshape(vt[1], _sh3(1, S_V, VD), ctx)

    var ts_a = List[Float32]()
    for _ in range(S_A): ts_a.append(sigma * TS_MULT)
    var at = _adaln_single(st, gw, String("audio_adaln_single"), ts_a, ctx)
    var a_temb = reshape(at[0], _sh3(1, S_A, 9 * AD), ctx)
    var a_embedded = reshape(at[1], _sh3(1, S_A, AD), ctx)

    # AV cross-attn global modulation (single global timestep)
    var g1 = List[Float32](); g1.append(sigma * TS_MULT)
    var vcs = _adaln_single(st, gw, String("av_ca_video_scale_shift_adaln_single"), g1, ctx)
    var v_ca_ss = reshape(vcs[0], _sh3(1, 1, 4 * VD), ctx)
    var acs = _adaln_single(st, gw, String("av_ca_audio_scale_shift_adaln_single"), g1, ctx)
    var a_ca_ss = reshape(acs[0], _sh3(1, 1, 4 * AD), ctx)
    # cross_gate_scale = cross_mult/ts_mult = 1.0 -> same scaled ts
    var vcg = _adaln_single(st, gw, String("av_ca_a2v_gate_adaln_single"), g1, ctx)
    var v_ca_gate = reshape(vcg[0], _sh3(1, 1, VD), ctx)
    var acg = _adaln_single(st, gw, String("av_ca_v2a_gate_adaln_single"), g1, ctx)
    var a_ca_gate = reshape(acg[0], _sh3(1, 1, AD), ctx)

    # prompt_ts (per text token)
    var ts_p = List[Float32]()
    for _ in range(N_TXT): ts_p.append(sigma * TS_MULT)
    var vpt = _adaln_single(st, gw, String("prompt_adaln_single"), ts_p, ctx)
    var v_prompt_ts = reshape(vpt[0], _sh3(1, N_TXT, 2 * VD), ctx)
    var apt = _adaln_single(st, gw, String("audio_prompt_adaln_single"), ts_p, ctx)
    var a_prompt_ts = reshape(apt[0], _sh3(1, N_TXT, 2 * AD), ctx)

    return _Mod(v_temb^, a_temb^, v_embedded^, a_embedded^, v_ca_ss^, a_ca_ss^,
                v_ca_gate^, a_ca_gate^, v_prompt_ts^, a_prompt_ts^)


def run(apply_lora: Bool, out_dir: String, max_steps: Int) raises:
    var ctx = DeviceContext()
    var cfg = LTX2Config.ltx2()
    print("=== LTX-2.3 T2V+AUDIO MVP (P7) — 256x256 / 16f distilled ===")
    print("  S_V/S_A/N_TXT:", S_V, S_A, N_TXT, " blocks:", NUM_LAYERS)
    print("  LoRA:", "ON" if apply_lora else "OFF", " out_dir:", out_dir,
          " max_steps:", max_steps)

    # ── open checkpoints ──
    var ck = ShardedSafeTensors.open(String(CKPT_FP8))

    # ── LoRA (rank-384 distilled): loaded once; the delta is ADDED at the
    # dequanted block linear per stream (HARD RULE — never a saved fuse).
    # Global (non-block) LoRA deltas are applied ONCE here to the resident
    # global weight dict `gw` (one-time additive add; never written to disk). ──
    var lora = LoraSet.load(String(LORA_PATH))
    if apply_lora:
        if lora.format != FMT_LTX2_DISTILLED:
            raise Error("MVP: LoRA not FMT_LTX2_DISTILLED")
        print("  [lora] loaded", lora.num_mappings(),
              "mappings (", lora.format_name(), ")")

    # ── globals (proj_in/proj_out, scale_shift tables) — F32 ──
    # Pre-load the 28 global LoRA target weights into a resident dict so global
    # LoRA deltas can be applied once before the loop.  The patchify/proj_out
    # weights used in _Globals and the adaln MLP weights used in _build_mod are
    # both sourced from this dict, ensuring LoRA-steered globals are honoured.
    print("  [load] globals")
    var gw = _load_global_weights_dict(ck, ctx)
    if apply_lora:
        var n_global = lora.apply_to_globals(gw, LORA_MULT, ctx)
        print("  [lora] global deltas applied (one-time additive):", n_global)
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

    # ── contexts via connectors (run ONCE, sigma-independent) ──
    print("  [connector] loading + running video/audio (F32)")
    var v_conn = LTX2ConnectorWeights.load(
        String(CKPT_FP8), String("video_embeddings_connector"),
        LTX2ConnectorConfig.video(), ctx,
    )
    var a_conn = LTX2ConnectorWeights.load(
        String(CKPT_FP8), String("audio_embeddings_connector"),
        LTX2ConnectorConfig.audio(), ctx,
    )

    # video pre-connector context: cached text_hidden [1,1024,4096], slice N_TXT
    var cached = ShardedSafeTensors.open(String(CACHED))
    var th = cast_tensor(
        Tensor.from_view_as_bf16(cached.tensor_view("text_hidden"), ctx),
        STDtype.F32, ctx,
    )  # [1,1024,4096]
    var video_pre = slice(th, 1, 0, N_TXT, ctx)  # [1,N_TXT,4096]
    _ = _stats(String("video_pre"), video_pre, ctx)

    # audio pre-connector context: NO cached audio + no Gemma -> derive a
    # deterministic 2048-dim context by down-projecting (slice) the cached
    # video hidden. Documented MVP approximation (audio is AV-coupled + finite
    # but not prompt-faithful).
    var audio_pre = slice(video_pre, 2, 0, AD, ctx)  # [1,N_TXT,2048]
    _ = _stats(String("audio_pre(derived)"), audio_pre, ctx)

    var enc = ltx2_connector_forward[N_TXT, V_HEADS, V_HDIM](v_conn, video_pre, ctx)
    var aenc = ltx2_connector_forward[N_TXT, A_HEADS, A_HDIM](a_conn, audio_pre, ctx)
    _ = _stats(String("enc"), enc, ctx)
    _ = _stats(String("aenc"), aenc, ctx)

    # ── RoPE tables (sigma-independent; built once) ──
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

    # ── stream open ──
    var stream = LTX2BlockStream.open(String(CKPT_FP8))
    if stream.block_count() != NUM_LAYERS:
        raise Error("stream block_count != 48")

    # ── init noise latents (Mojo randn; see RNG NOTE) ──
    print("  [noise] init video + audio latents")
    var video_x = randn(_sh5(1, 128, NF, NH, NW), SEED, STDtype.F32, ctx)
    var audio_x = randn(_sh4(1, AUDIO_C, S_A, AUDIO_MEL), SEED + 1, STDtype.F32, ctx)

    # ── sampler (distilled 8 steps; the table tail IS the stage-2 sigmas) ──
    var sched = LTX2Scheduler.distilled()
    var sigmas = sched.sigmas()
    print("  [sampler] distilled sigmas (9):", end="")
    for i in range(len(sigmas)):
        print(" ", sigmas[i], end="")
    print("")
    print("  STAGE-BOUNDARY NOTE: latent spatial-x2 upsampler + AdaIN not built;")
    print("    running the single distilled-8 schedule at fixed MVP resolution.")

    # ── two-stage* distilled denoise loop ──
    var n_steps = sched.num_steps
    if max_steps > 0 and max_steps < n_steps:
        n_steps = max_steps
    for step in range(n_steps):
        var sigma = sigmas[step]
        print("  --- step", step + 1, "/", sched.num_steps, " sigma=", sigma, "---")

        # patchify
        var v_flat = permute(reshape(video_x, _sh3(1, 128, S_V), ctx),
                             _video_perm(), ctx)        # [1,S_V,128]
        var a_flat = reshape(permute(audio_x, _audio_perm(), ctx),
                             _sh3(1, S_A, 128), ctx)     # [1,S_A,128]

        var hs = _linear_b(v_flat, g.v_pin_w, g.v_pin_b, ctx)   # [1,S_V,4096]
        var ahs = _linear_b(a_flat, g.a_pin_w, g.a_pin_b, ctx)  # [1,S_A,2048]

        # modulation (uses pre-loaded LoRA-applied global weights via gw)
        var mod = _build_mod(ck, gw, sigma, ctx)

        # 48 streamed blocks (boundary BF16, inner FP8 dequant), all F32 compute
        for i in range(NUM_LAYERS):
            var w: LTX2AVBlockWeights
            if _is_boundary(i):
                w = LTX2AVBlockWeights.load(String(CKPT_FP8), i, cfg, ctx).to_f32(ctx)
            else:
                var blk = stream.load_block_bf16(i, ctx)
                w = LTX2AVBlockWeights.from_fp8_block(blk^, cfg, ctx).to_f32(ctx)
            # AT-DEQUANT LoRA APPLY: add scale*(B@A) onto the F32 dequanted
            # block linears for THIS stream (re-applied each step; never fused).
            if apply_lora:
                _ = lora.apply_to_av_block(i, w, LORA_MULT, ctx)
            var outs = ltx2_block_forward_av[S_V, S_A, N_TXT, S_VPAD, S_APAD](
                w, hs, ahs, enc, aenc,
                mod.v_temb, mod.a_temb, mod.v_ca_ss, mod.a_ca_ss,
                mod.v_ca_gate, mod.a_ca_gate,
                mod.v_prompt_ts, mod.a_prompt_ts,
                v_cos, v_sin, a_cos, a_sin,
                ca_v_cos, ca_v_sin, ca_a_cos, ca_a_sin, EPS, ctx,
            )
            hs = _clone(outs[0], ctx)
            ahs = _clone(outs[1], ctx)

        # output stage -> velocity (patchified domain)
        var v_vel_flat = _output_stage(hs, g.v_sst, mod.v_embedded, g.v_pout_w,
                                       g.v_pout_b, VD, ctx)  # [1,S_V,128]
        var a_vel_flat = _output_stage(ahs, g.a_sst, mod.a_embedded, g.a_pout_w,
                                       g.a_pout_b, AD, ctx)  # [1,S_A,128]

        # unpatchify velocity to latent layout
        # video: [1,S_V,128] -> [1,128,S_V] -> [1,128,NF,NH,NW]
        var v_vel = reshape(permute(v_vel_flat, _unvideo_perm(), ctx),
                            _sh5(1, 128, NF, NH, NW), ctx)
        # audio: [1,S_A,128] -> [1,S_A,8,16] -> [1,8,S_A,16]
        var a_vel = permute(reshape(a_vel_flat, _sh4(1, S_A, AUDIO_C, AUDIO_MEL), ctx),
                            _unaudio_perm(), ctx)

        _ = _stats(String("v_vel"), v_vel, ctx)
        _ = _stats(String("a_vel"), a_vel, ctx)

        # Euler update on BOTH latents
        video_x = sched.step(video_x, v_vel, step, ctx)
        audio_x = sched.step(audio_x, a_vel, step, ctx)
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

    # save each decoded frame as PNG
    var png_paths = List[String]()
    for fr in range(n_frames_out):
        var fslice = slice(frames, 2, fr, 1, ctx)  # [1,3,1,H,W]
        var fs = fslice.shape()
        var chw = reshape(fslice, _sh4(fs[0], fs[1], fs[3], fs[4]), ctx)
        var p = out_dir + "/mvp_frame" + _pad2(fr) + ".png"
        save_png(chw, p, ctx, ValueRange.SIGNED)
        png_paths.append(p)
    print("  saved", n_frames_out, "frame PNGs ->", out_dir)

    # ── DECODE AUDIO ──
    # Audio decode + vocoder + mux is OPTIONAL (skipped for the fast LoRA gen
    # gate, which compares video frames only). Gate it on max_steps==0 meaning
    # "full run"; any positive max_steps is a fast gen-gate run -> skip audio.
    if max_steps != 0:
        print("  [decode] audio SKIPPED (fast gen-gate run); frames saved ->",
              out_dir)
        print("=== MVP (fast gen-gate) DONE ===")
        return

    var wav_out = out_dir + "/mvp_audio.wav"
    var mp4_out = out_dir + "/ltx2_t2v_av_256_16f.mp4"
    print("  [decode] audio VAE (latent -> mel) -> vocoder (mel -> 48kHz wav)")
    var avae = LTX2AudioVaeDecoderWeights.load(String(CKPT_BF16), ctx)
    var audio_x_bf16 = cast_tensor(audio_x, STDtype.BF16, ctx)
    var mel_raw = decode_audio(avae, audio_x_bf16, ctx)  # [1,2,T_out,F_out]
    # Vocoder's gated path is F32 (ltx2_vocoder_smoke loads mel via from_view_as_f32);
    # the BF16 activation1d path has a dtype bug, so run the vocoder in F32.
    var mel = cast_tensor(mel_raw, STDtype.F32, ctx)
    var msh = mel.shape()
    print("  mel NCHW: [", msh[0], ",", msh[1], ",", msh[2], ",", msh[3], "] dtype-cast F32")
    _ = _stats(String("mel"), mel, ctx)

    var voc = LTX2VocoderWithBWE.from_file(String(CKPT_BF16), ctx)
    var wav = voc.forward(mel, ctx)  # [1,2,L]
    var wsh = wav.shape()
    var L = wsh[2]
    print("  wav [", wsh[0], ",", wsh[1], ",", wsh[2], "] @", voc.output_sample_rate(), "Hz")

    # audio rms / range
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

    # write wav (interleave L,R from [1,2,L])
    var inter = List[Float32]()
    inter.resize(L * 2, Float32(0.0))
    for ch in range(2):
        for s in range(L):
            inter[s * 2 + ch] = wh[ch * L + s]
    _write_wav(wav_out, inter, voc.output_sample_rate())
    print("  wrote wav:", wav_out)

    # ── MUX (ffmpeg: frames + wav -> mp4) ──
    print("  [mux] ffmpeg frames + wav -> mp4")
    _mux_mp4(out_dir, n_frames_out, wav_out, mp4_out)
    print("=== MVP DONE ===")
    print("  mp4:", mp4_out)
    print("  wav:", wav_out)
    print("  frames:", out_dir, "/mvp_frame00.png ..")


def _mkdir(path: String) raises:
    var cmd = String("mkdir -p ") + path + " >/dev/null 2>&1"
    _ = sys_system(cmd)


# ── argv-driven entry: `mvp [base|lora] [out_dir] [max_steps]` ──
#   no args            -> full MVP, LoRA OFF, default OUT_DIR (audio+mux)
#   "lora"             -> full MVP with LoRA ON
#   "base <dir> <n>"   -> fast video-only gen-gate run (LoRA OFF) -> <dir>
#   "lora <dir> <n>"   -> fast video-only gen-gate run (LoRA ON)  -> <dir>
def main() raises:
    var a = argv()
    var apply_lora = False
    var out_dir = String(OUT_DIR)
    var max_steps = 0
    if len(a) >= 2 and String(a[1]) == "lora":
        apply_lora = True
    if len(a) >= 3:
        out_dir = String(a[2])
    if len(a) >= 4:
        max_steps = atol(String(a[3]))
    _mkdir(out_dir)
    run(apply_lora, out_dir, max_steps)


# ── patchify/unpatchify permutations ──
def _video_perm() -> List[Int]:
    var p = List[Int](); p.append(0); p.append(2); p.append(1); return p^  # [1,128,S]->[1,S,128]

def _unvideo_perm() -> List[Int]:
    var p = List[Int](); p.append(0); p.append(2); p.append(1); return p^  # [1,S,128]->[1,128,S]

def _audio_perm() -> List[Int]:
    # [1,8,T,16] -> [1,T,8,16] for reshape to [1,T,128]
    var p = List[Int](); p.append(0); p.append(2); p.append(1); p.append(3)
    return p^

def _unaudio_perm() -> List[Int]:
    # [1,T,8,16] -> [1,8,T,16]
    var p = List[Int](); p.append(0); p.append(2); p.append(1); p.append(3)
    return p^


def _pad2(n: Int) -> String:
    if n < 10:
        return String("0") + String(n)
    return String(n)


# ── WAV writer (16-bit PCM LE stereo), copied from ltx2_vocoder_smoke.mojo ──
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

    # write via libc
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
    # frames are mvp_frame%02d.png in out_dir at 25 fps; mux with wav.
    var cmd = String("ffmpeg -y -framerate 25 -i ")
    cmd += out_dir + "/mvp_frame%02d.png -i " + wav
    cmd += " -c:v libx264 -pix_fmt yuv420p -c:a aac -shortest " + mp4
    cmd += " >/dev/null 2>&1"
    var rc = sys_system(cmd)
    if rc != 0:
        print("  [mux] WARNING: ffmpeg returned", rc, "(frames+wav still saved)")
