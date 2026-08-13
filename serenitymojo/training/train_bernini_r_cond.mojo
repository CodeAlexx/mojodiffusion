# serenitymojo/training/train_bernini_r_cond.mojo
#
# BERNINI-R conditioning trainer — the reference-guided renderer, ALL 12 tasks.
# Pure Mojo + MAX, GPU, block-swap offload. Sequence-concat of N CLEAN
# conditioning latents (each with its own source-id RoPE) AHEAD of the noised
# TARGET, full self-attention over the packed sequence, velocity-MSE loss on the
# TARGET region ONLY.
#
# ── This file is the ASSEMBLY + LOOP only. It reimplements NO block/stack math ──
# It REUSES verbatim:
#   * the certified Wan2.2-A14B LoRA engine (wan22_stack_lora) — the block runs
#     over WHATEVER sequence + cos/sin it is given, so a packed
#     [cond_1 | ... | cond_N | target] sequence needs no new kernel;
#   * the Tier-2a source-id RoPE (bernini_src_id_rope.build_bernini_src_id_rope);
#   * the proven patchify (ops/patchify3d) and safetensors reader;
#   * the Bernini timestep samplers (schedule.bernini_sample_sigma — logit_normal
#     + mode density, per-shift window) and EMA (schedule.ema_update math).
#
# ── The 12 tasks (task -> recipe in training/bernini_tasks.mojo) ────────────────
#   task selected by env BERNINI_TASK (default "t2v"). The recipe fixes:
#     shift (3/4/5), weighting (logit_normal image / mode video), n_cond segments.
#   n_cond=0  (t2i/t2v)  -> Tier-1 T2V path (packed == target, src_id=0).
#   n_cond>=1            -> N CLEAN conditioning segments src_id 1..N packed AHEAD
#                           of the target (src_id 0), Bernini pack_vae_latents order.
#   BERNINI_NO_COND=1 forces n_cond=0 regardless of task (regression switch).
#
#   Each packed sample shares ONE timestep (data.py:365) applied to all tokens —
#   matches the certified stack's single-temb AdaLN bit-for-bit.
#
# ── BUILD (binary; JIT can't resolve the driver calls) ─────────────────────────
#   cd /home/alex/mojodiffusion; rm -f serenitymojo.mojopkg
#   pixi run mojo build --optimization-level 2 --target-accelerator sm_120 \
#     -I . -I /home/alex/MOJO-libs -Xlinker -lm -Xlinker -lcuda \
#     serenitymojo/training/train_bernini_r_cond.mojo -o output/bin/train_bernini_r_cond
# ── RUN ────────────────────────────────────────────────────────────────────────
#   BERNINI_TASK=i2v output/bin/train_bernini_r_cond [config.json]
#
# Mojo 1.0.0b1, NVIDIA GPU (sm_120 / 16GB).

from max.gpu.host import DeviceContext
from std.collections import List
from std.math import sqrt, exp
from std.time import perf_counter_ns
from std.sys import argv
from std.ffi import external_call
from std.memory import alloc

from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.ffi import sys_open, sys_close, O_RDONLY
from serenitymojo.tensor import Tensor
from serenitymojo.ops.patchify3d import patchify3d
from serenitymojo.io.train_config_reader import read_model_config
from serenitymojo.training.train_config import TrainConfig

# REUSE: the certified A14B LoRA engine (NO block/stack math re-implemented here).
from serenitymojo.models.wan22.weights import load_wan22_stack_base, detect_wan22_prefix
from serenitymojo.models.wan22.wan22_stack_lora import (
    Wan22StackBase, Wan22LoraSet, Wan22LoraGradSet, Wan22StackForward,
    build_wan22_lora_set, wan22_total_adapters,
    wan22_stack_lora_forward_offload, wan22_stack_lora_backward_offload,
    wan22_lora_adamw_step, save_wan22_lora, save_wan22_lora_state,
    # WAN_DEVNATIVE (Phase-1b device-native path)
    Wan22LoraDeviceGradSet, wan22_stack_lora_backward_offload_devnative,
    wan22_lora_adamw_state_init,
)
from serenitymojo.training.lora_adamw_plain_fused import (
    LoraAdamWPlainDeviceState, fused_lora_adamw_plain_step_resident_device_grads,
)
from serenitymojo.offload.wan22_plan import build_wan22_block_plan
from serenitymojo.offload.plan import OffloadConfig
from serenitymojo.offload.turbo_planned_loader import TurboPlannedLoader

# REUSE: the Tier-2a source-id RoPE (the packed-sequence rope builder).
from serenitymojo.models.wan22.bernini_src_id_rope import (
    BerniniRopeSegment, build_bernini_src_id_rope,
)

# REUSE: the Bernini timestep samplers (logit_normal + mode, per-shift window).
from serenitymojo.training.schedule import (
    BerniniWindow, bernini_task_window, bernini_sample_sigma,
)

# REUSE: the LoRA adapter carrier (to build the EMA-shadow set for saving).
from serenitymojo.training.train_step import LoraAdapter

# The per-task recipe table (shift / weighting / n_cond / prompt) + smoke geometry.
from serenitymojo.training.bernini_tasks import (
    BerniniTaskRecipe, bernini_recipe_for, BerniniCondSeg, bernini_smoke_cond_segments,
    bernini_task_name_for_id,
)


# ── A14B architecture (comptime; the wired engine's dims — same as Tier-1) ─────
comptime H = 40
comptime Dh = 128
comptime DIM = H * Dh            # 5120
comptime FFN = 13824
comptime IN_CH = 64              # VAE latent 16ch * patch(1,2,2)
comptime OUT_CH = 64             # velocity of the 16-ch latent
comptime TEXT_DIM = 4096
comptime FREQ_DIM = 256
comptime NUM_BLOCKS = 40
comptime EPS = Float32(1.0e-6)
comptime ROPE_THETA = Float32(10000.0)

# ── Target geometry (fixed): latent [16,1,32,32] -> patchify(1,2,2) -> (1,16,16). ─
comptime TGT_FG = 1
comptime TGT_HG = 16
comptime TGT_WG = 16
comptime S_TGT = TGT_FG * TGT_HG * TGT_WG          # 256 target tokens
comptime TXT = 512                                 # umt5 cross-attn kv length
comptime TGT_SRC_ID = Float32(0.0)                 # target == stock wan rope

# Bernini noise window (bernini_renderer_high.yaml).
comptime NOISE_TMIN = Float64(0.875)
comptime NOISE_TMAX = Float64(1.0)
comptime LOGIT_MEAN = Float64(0.5)
comptime LOGIT_STD = Float64(1.0)
comptime MODE_SCALE = Float64(1.29)

# ── optimizer / recipe defaults (config overrides where present) ───────────────
comptime DEFAULT_LR = Float32(2.0e-4)
comptime DEFAULT_MAX_GRAD_NORM = Float32(1.0)
comptime DEFAULT_EMA_DECAY = Float32(0.9999)       # bernini_renderer_high.yaml ema_decay
comptime DEFAULT_CKPT_LOW =
    "/home/alex/.serenity/models/checkpoints/Bernini-R-Diffusers/serenity_fp8_e4m3_de8c462/low"
comptime DEFAULT_CONFIG =
    "serenitymojo/configs/bernini_r_t2v_cond_smoke.json"


# ══════════════════════════════════════════════════════════════════════════════
# small host helpers (self-contained; the RNG mirrors train_wan22_real exactly)
# ══════════════════════════════════════════════════════════════════════════════
def _file_exists(path: String) -> Bool:
    var fd = sys_open(path, O_RDONLY, Int32(0))
    if fd < 0:
        return False
    _ = sys_close(fd)
    return True


comptime _EnvPtr = UnsafePointer[UInt8, MutExternalOrigin]


def _env_is_set(name: String) -> Bool:
    var n = name.byte_length()
    var buf = alloc[UInt8](n + 1)
    var src = name.as_bytes()
    for i in range(n):
        buf[i] = src[i]
    buf[n] = 0
    var cname = _EnvPtr(unsafe_from_address=Int(buf))
    var ret = external_call["getenv", _EnvPtr](cname)
    buf.free()
    if Int(ret) == 0:
        return False
    return ret[0] == UInt8(49) and ret[1] == UInt8(0)


# Read an env var STRING value (empty String if unset). Small fixed buffer read.
def _env_str(name: String) -> String:
    var n = name.byte_length()
    var buf = alloc[UInt8](n + 1)
    var src = name.as_bytes()
    for i in range(n):
        buf[i] = src[i]
    buf[n] = 0
    var cname = _EnvPtr(unsafe_from_address=Int(buf))
    var ret = external_call["getenv", _EnvPtr](cname)
    buf.free()
    if Int(ret) == 0:
        return String("")
    var out = String("")
    var i = 0
    while ret[i] != 0:
        out += chr(Int(ret[i]))
        i += 1
    return out


def _base_file_for(ckpt: String) -> String:
    var shared = ckpt + String("/shared.safetensors")
    if _file_exists(shared):
        return shared
    return ckpt


def _splitmix64(state: UInt64) -> UInt64:
    var z = state + UInt64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
    return z ^ (z >> 31)


def _u01(seed: UInt64) -> Float32:
    var w = _splitmix64(seed)
    return Float32(Float64(w >> 40) * (1.0 / 16777216.0))


def _fcos(x: Float32) -> Float32:
    from std.math import cos as _c
    return Float32(_c(Float64(x)))


def _flog(x: Float32) -> Float32:
    from std.math import log as _l
    return Float32(_l(Float64(x)))


def _randn(seed: UInt64) -> Float32:
    var u1 = _u01(seed * UInt64(2654435761) + 1)
    var u2 = _u01(seed * UInt64(1442695040888963407) + 2)
    if u1 < Float32(1.0e-7):
        u1 = Float32(1.0e-7)
    var r = sqrt(Float32(-2.0) * _flog(u1))
    var ang = Float32(6.2831853071795864769) * u2
    return r * _fcos(ang)


def _noise(n: Int, seed: UInt64) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        out.append(_randn(seed + UInt64(i) * 2 + 1))
    return out^


def _synth(n: Int, seed: UInt64) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        out.append(_randn(seed * UInt64(1099511628211) + UInt64(i) * 3 + 5))
    return out^


# global grad clip over all LoRA d_A/d_B (musubi max_grad_norm).
def _clip(mut grads: Wan22LoraGradSet, max_norm: Float32) -> Float32:
    var ss = Float32(0.0)
    for i in range(len(grads.d_a)):
        for j in range(len(grads.d_a[i])):
            ss += grads.d_a[i][j] * grads.d_a[i][j]
        for j in range(len(grads.d_b[i])):
            ss += grads.d_b[i][j] * grads.d_b[i][j]
    var gn = sqrt(ss)
    if gn > max_norm and gn > Float32(0.0):
        var scl = max_norm / gn
        for i in range(len(grads.d_a)):
            for j in range(len(grads.d_a[i])):
                grads.d_a[i][j] = grads.d_a[i][j] * scl
            for j in range(len(grads.d_b[i])):
                grads.d_b[i][j] = grads.d_b[i][j] * scl
    return gn


def _grad_abs_sum(grads: Wan22LoraGradSet) -> Float32:
    var s = Float32(0.0)
    for i in range(len(grads.d_a)):
        for j in range(len(grads.d_a[i])):
            s += abs(grads.d_a[i][j])
        for j in range(len(grads.d_b[i])):
            s += abs(grads.d_b[i][j])
    return s


# ── Load a CLEAN latent [16, ff, fh, fw] and patchify(1,2,2) -> [S, IN_CH] flat. ─
# ff = temporal frames (1 for images, >1 for video conditioning). Returns
# (flat [S*IN_CH]) and writes std/mean of the source latent.
def _patchify_latent(
    vals: List[Float32], ff: Int, fh: Int, fw: Int, ctx: DeviceContext,
    mut out_std: Float32, mut out_mean: Float32,
) raises -> List[Float32]:
    var n = 16 * ff * fh * fw
    if len(vals) != n:
        raise Error("latent numel mismatch in _patchify_latent")
    var s = Float64(0.0)
    var ss = Float64(0.0)
    for i in range(n):
        s += Float64(vals[i])
        ss += Float64(vals[i]) * Float64(vals[i])
    var mean = s / Float64(n)
    var varp = ss / Float64(n) - mean * mean
    if varp < 0.0:
        varp = 0.0
    out_mean = Float32(mean)
    out_std = Float32(sqrt(varp))
    var shp = List[Int]()
    shp.append(16)
    shp.append(ff)
    shp.append(fh)
    shp.append(fw)
    var t = Tensor.from_host(vals.copy(), shp^, STDtype.F32, ctx)
    var patched = patchify3d(t, 1, 2, 2, ctx)   # [S, IN_CH]
    return patched.to_host(ctx)


# ── Read the cached target latent [16,F,H,W] f32 from sample_{idx}. ────────────
# Generalized: the target may be single-frame (image, F=1) OR multi-frame (video/
# clip, F>1). Returns the patchified token grid [S_tgt, IN_CH] flat and writes the
# post-patchify grid (out_f=F, out_h=H/2, out_w=W/2). S_tgt = out_f*out_h*out_w.
def _load_cache_target(
    cache_dir: String, idx: Int, ctx: DeviceContext,
    mut out_std: Float32, mut out_mean: Float32,
    mut out_f: Int, mut out_h: Int, mut out_w: Int,
) raises -> List[Float32]:
    var path = cache_dir + String("/sample_") + String(idx) + String(".safetensors")
    var st = SafeTensors.open(path)
    var info = st.tensor_info(String("latent"))
    if info.dtype != STDtype.F32:
        raise Error(String("cache latent dtype != F32 in ") + path)
    if len(info.shape) != 4 or info.shape[0] != 16:
        raise Error(String("cache latent must be [16,F,H,W] in ") + path)
    var ff = info.shape[1]
    var hl = info.shape[2]
    var wl = info.shape[3]
    if hl % 2 != 0 or wl % 2 != 0:
        raise Error(String("cache latent spatial dims must be even in ") + path)
    var nlat = 16 * ff * hl * wl
    var lb = st.tensor_bytes(String("latent"))
    var fp = lb.unsafe_ptr().bitcast[Float32]()
    var vals = List[Float32]()
    for i in range(nlat):
        vals.append(fp[i])
    out_f = ff
    out_h = hl // 2
    out_w = wl // 2
    return _patchify_latent(vals^, ff, hl, wl, ctx, out_std, out_mean)


# Task id (0..11) stored in sample_{idx}; -1 if the key is absent (legacy cache).
def _load_cache_task_id(cache_dir: String, idx: Int) raises -> Int:
    var st = SafeTensors.open(_cache_sample_path(cache_dir, idx))
    if not _st_has(st, String("task_id")):
        return -1
    var lb = st.tensor_bytes(String("task_id"))
    var ip = lb.unsafe_ptr().bitcast[Int32]()
    return Int(ip[0])


# Number of consecutive sample_{i}.safetensors files (starting at 0) in cache_dir.
def _count_cache_samples(cache_dir: String) -> Int:
    var n = 0
    while _file_exists(_cache_sample_path(cache_dir, n)):
        n += 1
    return n


# ── Real-cache conditioning + text readers (Bernini-R sample schema) ───────────
# sample_{idx}.safetensors keys (scripts/bernini_r_build_cache.py):
#   latent [16,F,H,W] f32 (target)  |  cond_count [1] i32  |
#   cond_{k} [16,f,h,w] f32 (CLEAN)  |  cond_{k}_src_id [1] i32  |
#   text [512,4096] bf16  |  text_len [1] i32  |  task_id [1] i32
def _cache_sample_path(cache_dir: String, idx: Int) -> String:
    return cache_dir + String("/sample_") + String(idx) + String(".safetensors")


def _st_has(st: SafeTensors, name: String) -> Bool:
    var nms = st.names()
    for i in range(len(nms)):
        if nms[i] == name:
            return True
    return False


# Number of conditioning segments stored in the sample (0 if key absent).
def _load_cache_cond_count(cache_dir: String, idx: Int) raises -> Int:
    var st = SafeTensors.open(_cache_sample_path(cache_dir, idx))
    if not _st_has(st, String("cond_count")):
        return 0
    var lb = st.tensor_bytes(String("cond_count"))
    var ip = lb.unsafe_ptr().bitcast[Int32]()
    return Int(ip[0])


# One CLEAN conditioning latent: [16,f,hl,wl] -> patchify(1,2,2) -> [S,IN_CH]
# tokens, returning the post-patchify token grid (f, hl/2, wl/2) + its source id.
struct CacheCond(Movable):
    var tokens: List[Float32]
    var f: Int
    var h: Int
    var w: Int
    var src_id: Float32

    def __init__(out self, var tokens: List[Float32], f: Int, h: Int, w: Int,
                src_id: Float32):
        self.tokens = tokens^
        self.f = f
        self.h = h
        self.w = w
        self.src_id = src_id


# Header-only conditioning grid (f, hl/2, wl/2) + source id — no token load.
struct CacheGrid(Copyable, Movable):
    var f: Int
    var h: Int
    var w: Int
    var src_id: Float32

    def __init__(out self, f: Int, h: Int, w: Int, src_id: Float32):
        self.f = f
        self.h = h
        self.w = w
        self.src_id = src_id


def _load_cache_cond_grid(cache_dir: String, idx: Int, k: Int) raises -> CacheGrid:
    var st = SafeTensors.open(_cache_sample_path(cache_dir, idx))
    var key = String("cond_") + String(k)
    var info = st.tensor_info(key)
    if len(info.shape) != 4 or info.shape[0] != 16:
        raise Error(String("cache cond must be [16,f,h,w]: ") + key)
    var ff = info.shape[1]
    var hl = info.shape[2]
    var wl = info.shape[3]
    if hl % 2 != 0 or wl % 2 != 0:
        raise Error(String("cache cond spatial dims must be even: ") + key)
    var sid = Float32(k + 1)
    var sid_key = key + String("_src_id")
    if _st_has(st, sid_key):
        var sb = st.tensor_bytes(sid_key)
        var sp = sb.unsafe_ptr().bitcast[Int32]()
        sid = Float32(Int(sp[0]))
    return CacheGrid(ff, hl // 2, wl // 2, sid)


def _load_cache_cond(
    cache_dir: String, idx: Int, k: Int, ctx: DeviceContext,
) raises -> CacheCond:
    var st = SafeTensors.open(_cache_sample_path(cache_dir, idx))
    var key = String("cond_") + String(k)
    var info = st.tensor_info(key)
    if info.dtype != STDtype.F32:
        raise Error(String("cache cond dtype != F32: ") + key)
    if len(info.shape) != 4 or info.shape[0] != 16:
        raise Error(String("cache cond must be [16,f,h,w]: ") + key)
    var ff = info.shape[1]
    var hl = info.shape[2]
    var wl = info.shape[3]
    if hl % 2 != 0 or wl % 2 != 0:
        raise Error(String("cache cond spatial dims must be even: ") + key)
    var nlat = 16 * ff * hl * wl
    var lb = st.tensor_bytes(key)
    var fp = lb.unsafe_ptr().bitcast[Float32]()
    var vals = List[Float32]()
    for i in range(nlat):
        vals.append(fp[i])
    var cs = Float32(0.0)
    var cm = Float32(0.0)
    var toks = _patchify_latent(vals^, ff, hl, wl, ctx, cs, cm)
    # source id (target=0, conditioning k -> k+1 per pack_vae_latents ordering).
    var sid = Float32(k + 1)
    var sid_key = key + String("_src_id")
    if _st_has(st, sid_key):
        var sb = st.tensor_bytes(sid_key)
        var sp = sb.unsafe_ptr().bitcast[Int32]()
        sid = Float32(Int(sp[0]))
    print("[data] cond seg", k, "(cache CLEAN, src_id=", sid, "): std=", cs,
          " mean=", cm, " grid(f,h,w)=(", ff, ",", hl // 2, ",", wl // 2, ")")
    return CacheCond(toks^, ff, hl // 2, wl // 2, sid)


# umt5 text [512,4096] bf16 -> flat f32 [TXT*TEXT_DIM].
def _load_cache_text(
    cache_dir: String, idx: Int, ctx: DeviceContext,
) raises -> List[Float32]:
    var st = SafeTensors.open(_cache_sample_path(cache_dir, idx))
    var info = st.tensor_info(String("text"))
    if len(info.shape) != 2 or info.shape[0] != TXT or info.shape[1] != TEXT_DIM:
        raise Error("cache text must be [512,4096]")
    var n = TXT * TEXT_DIM
    var lb = st.tensor_bytes(String("text"))
    var out = List[Float32]()
    if info.dtype == STDtype.BF16:
        var bp = lb.unsafe_ptr().bitcast[BFloat16]()
        for i in range(n):
            out.append(Float32(bp[i]))
    elif info.dtype == STDtype.F32:
        var fp = lb.unsafe_ptr().bitcast[Float32]()
        for i in range(n):
            out.append(fp[i])
    else:
        raise Error("cache text dtype must be BF16 or F32")
    return out^


# ── Save an EMA-shadow LoRA sibling (host F32 ema_a/ema_b -> PEFT safetensors). ─
def _save_ema_sibling(
    lora: Wan22LoraSet, ema_a: List[List[Float32]], ema_b: List[List[Float32]],
    path: String, ctx: DeviceContext,
) raises -> Int:
    var ad = List[LoraAdapter]()
    for i in range(len(lora.ad)):
        var na = len(ema_a[i])
        var nb = len(ema_b[i])
        var za = List[Float32]()
        var va = List[Float32]()
        for _ in range(na):
            za.append(Float32(0.0))
            va.append(Float32(0.0))
        var zb = List[Float32]()
        var vb = List[Float32]()
        for _ in range(nb):
            zb.append(Float32(0.0))
            vb.append(Float32(0.0))
        ad.append(LoraAdapter(
            ema_a[i].copy(), ema_b[i].copy(),
            lora.ad[i].rank, lora.ad[i].in_f, lora.ad[i].out_f, lora.ad[i].scale,
            za^, va^, zb^, vb^,
        ))
    var ema_set = Wan22LoraSet(ad^, lora.num_blocks, lora.rank)
    return save_wan22_lora(ema_set, path, ctx)




# Read a float env var (empty/unset -> default). Used for the C13 default-off
# condition-dropout probabilities (text/img/video).
def _env_float(name: String, default: Float32) -> Float32:
    var s = _env_str(name)
    if s.byte_length() == 0:
        return default
    var n = s.byte_length()
    var buf = alloc[UInt8](n + 1)
    var src = s.as_bytes()
    for i in range(n):
        buf[i] = src[i]
    buf[n] = 0
    var cptr = _EnvPtr(unsafe_from_address=Int(buf))
    var v = external_call["atof", Float64](cptr)
    buf.free()
    return Float32(v)


# ══════════════════════════════════════════════════════════════════════════════
# Per-sample precomputed training bundle (host). One per cache sample (round-robin
# scheduled) or a single synthetic bundle. The TARGET is ALWAYS the trailing
# n_tgt_tokens of the packed sequence; conditioning (N segments, source-id 1..N)
# is packed AHEAD. task_id drives the per-sample recipe (shift/weighting/window).
# ══════════════════════════════════════════════════════════════════════════════
@fieldwise_init
struct BerniniSample(Copyable, Movable):
    var task_id: Int
    var task_name: String
    var is_mode: Bool
    var shift: Float64
    var win: BerniniWindow
    var pin_sigma: Float32
    var no_cond: Bool
    var seq_len: Int
    var n_tgt_tokens: Int
    var tgt_std: Float32
    var x0: List[Float32]            # n_tgt_tokens * IN_CH (clean, patchified)
    var cond_tokens: List[Float32]   # n_cond_tokens * IN_CH (clean, patchified)
    var txt_tokens: List[Float32]    # TXT * TEXT_DIM
    var cos_h: List[Float32]         # seq_len * Dh/2
    var sin_h: List[Float32]
    var seg_f: List[Int]             # per-cond-seg post-patchify grid + src id
    var seg_h: List[Int]
    var seg_w: List[Int]
    var seg_src: List[Float32]
    var seg_kind: List[String]


@fieldwise_init
struct StepOut(Copyable, Movable):
    var loss: Float32
    var gn: Float32
    var absum: Float32
    var nonfinite: Int
    var sigma: Float32


# Build the packed source-id rope (cond segs src 1..N then target src 0) -> host.
def _build_sample_rope(
    seg_f: List[Int], seg_h: List[Int], seg_w: List[Int], seg_src: List[Float32],
    tgt_f: Int, tgt_h: Int, tgt_w: Int, seq_len: Int, ctx: DeviceContext,
    mut cos_h: List[Float32], mut sin_h: List[Float32],
) raises:
    var rope_segments = List[BerniniRopeSegment]()
    for i in range(len(seg_f)):
        rope_segments.append(
            BerniniRopeSegment(seg_f[i], seg_h[i], seg_w[i], seg_src[i])
        )
    rope_segments.append(BerniniRopeSegment(tgt_f, tgt_h, tgt_w, TGT_SRC_ID))
    var rope = build_bernini_src_id_rope(rope_segments, Dh, ROPE_THETA, STDtype.F32, ctx)
    cos_h = rope[0].to_host(ctx)
    sin_h = rope[1].to_host(ctx)
    var expect_rope = seq_len * (Dh // 2)
    if len(cos_h) != expect_rope or len(sin_h) != expect_rope:
        raise Error("packed rope length != seq_len*Dh/2 — segment/geometry mismatch")


# Build one bundle from a REAL cache sample. task_id (if stored) drives the recipe;
# otherwise env_task is used (legacy caches). force_no_cond ignores conditioning.
def _build_cache_sample(
    cache_dir: String, idx: Int, env_task: String, force_no_cond: Bool,
    seed: UInt64, ctx: DeviceContext,
) raises -> BerniniSample:
    var tid = _load_cache_task_id(cache_dir, idx)
    var task_name = env_task
    if tid >= 0:
        var nm = bernini_task_name_for_id(tid)
        if nm.byte_length() > 0:
            task_name = nm
    var recipe = bernini_recipe_for(task_name)
    var win = bernini_task_window(recipe.shift, NOISE_TMIN, NOISE_TMAX)
    var pin = bernini_sample_sigma(
        recipe.is_mode, recipe.shift, seed + UInt64(idx) * UInt64(7919),
        LOGIT_MEAN, LOGIT_STD, MODE_SCALE, win,
    )

    var cond_count = _load_cache_cond_count(cache_dir, idx)
    var no_cond = force_no_cond or cond_count == 0

    # ── conditioning segments (skipped when no_cond) ─────────────────────────────
    var seg_f = List[Int]()
    var seg_h = List[Int]()
    var seg_w = List[Int]()
    var seg_src = List[Float32]()
    var seg_kind = List[String]()
    var cond_tokens = List[Float32]()
    var n_cond_tokens = 0
    if not no_cond:
        for k in range(cond_count):
            var cc = _load_cache_cond(cache_dir, idx, k, ctx)
            var want = cc.f * cc.h * cc.w * IN_CH
            if len(cc.tokens) != want:
                raise Error("cache conditioning seg patch len mismatch")
            for j in range(len(cc.tokens)):
                cond_tokens.append(cc.tokens[j])
            seg_f.append(cc.f)
            seg_h.append(cc.h)
            seg_w.append(cc.w)
            seg_src.append(cc.src_id)
            seg_kind.append(String("video") if cc.f > 1 else String("image"))
            n_cond_tokens += cc.f * cc.h * cc.w

    # ── target latent (single- OR multi-frame) ───────────────────────────────────
    var tgt_std = Float32(0.0)
    var tgt_mean = Float32(0.0)
    var tgt_f = 0
    var tgt_h = 0
    var tgt_w = 0
    var x0 = _load_cache_target(cache_dir, idx, ctx, tgt_std, tgt_mean,
                               tgt_f, tgt_h, tgt_w)
    var n_tgt_tokens = tgt_f * tgt_h * tgt_w
    if len(x0) != n_tgt_tokens * IN_CH:
        raise Error("target patch len != n_tgt_tokens*IN_CH")
    var seq_len = n_cond_tokens + n_tgt_tokens

    # ── text + rope ──────────────────────────────────────────────────────────────
    var txt = _load_cache_text(cache_dir, idx, ctx)
    var cos_h = List[Float32]()
    var sin_h = List[Float32]()
    _build_sample_rope(seg_f, seg_h, seg_w, seg_src, tgt_f, tgt_h, tgt_w,
                       seq_len, ctx, cos_h, sin_h)

    return BerniniSample(
        tid, task_name, recipe.is_mode, recipe.shift, win^, pin, no_cond,
        seq_len, n_tgt_tokens, tgt_std, x0^, cond_tokens^, txt^, cos_h^, sin_h^,
        seg_f^, seg_h^, seg_w^, seg_src^, seg_kind^,
    )


# Build one synthetic bundle (no cache): env_task recipe + smoke cond geometry.
def _build_synth_sample(
    env_task: String, force_no_cond: Bool, seed: UInt64, ctx: DeviceContext,
) raises -> BerniniSample:
    var recipe = bernini_recipe_for(env_task)
    var win = bernini_task_window(recipe.shift, NOISE_TMIN, NOISE_TMAX)
    var pin = bernini_sample_sigma(
        recipe.is_mode, recipe.shift, seed, LOGIT_MEAN, LOGIT_STD, MODE_SCALE, win,
    )
    var no_cond = force_no_cond or recipe.n_cond == 0

    var seg_f = List[Int]()
    var seg_h = List[Int]()
    var seg_w = List[Int]()
    var seg_src = List[Float32]()
    var seg_kind = List[String]()
    var cond_tokens = List[Float32]()
    var n_cond_tokens = 0
    if not no_cond:
        var segs = bernini_smoke_cond_segments(env_task)
        for i in range(len(segs)):
            var seg = segs[i].copy()
            var cs = Float32(0.0)
            var cm = Float32(0.0)
            var cl = _synth(16 * seg.f * (seg.h * 2) * (seg.w * 2),
                            seed * 7 + UInt64(101 + i * 13))
            var toks = _patchify_latent(cl^, seg.f, seg.h * 2, seg.w * 2, ctx, cs, cm)
            for j in range(len(toks)):
                cond_tokens.append(toks[j])
            seg_f.append(seg.f)
            seg_h.append(seg.h)
            seg_w.append(seg.w)
            seg_src.append(Float32(i + 1))
            seg_kind.append(String("video") if seg.f > 1 else String("image"))
            n_cond_tokens += seg.f * seg.h * seg.w

    var tgt_std = Float32(0.0)
    var tgt_mean = Float32(0.0)
    var tl = _synth(16 * 1 * 32 * 32, seed * 13 + 1)
    var x0 = _patchify_latent(tl^, 1, 32, 32, ctx, tgt_std, tgt_mean)
    var n_tgt_tokens = 1 * 16 * 16       # single-frame 32x32 target
    var seq_len = n_cond_tokens + n_tgt_tokens

    var txt = _synth(TXT * TEXT_DIM, seed * 777 + 9)
    var cos_h = List[Float32]()
    var sin_h = List[Float32]()
    _build_sample_rope(seg_f, seg_h, seg_w, seg_src, 1, 16, 16,
                       seq_len, ctx, cos_h, sin_h)

    return BerniniSample(
        -1, env_task, recipe.is_mode, recipe.shift, win^, pin, no_cond,
        seq_len, n_tgt_tokens, tgt_std, x0^, cond_tokens^, txt^, cos_h^, sin_h^,
        seg_f^, seg_h^, seg_w^, seg_src^, seg_kind^,
    )


# ══════════════════════════════════════════════════════════════════════════════
# ONE training step, comptime-monomorphized on the packed sequence length SEQ.
# Assembles [cond_1..N (clean, per-seg dropout) | target (noised)], forwards the
# certified stack, velocity-MSE on the TARGET region ONLY, LoRA backward + AdamW +
# EMA. Condition dropout (per Bernini config): a dropped modality's tokens are
# replaced with the NULL/empty (zero) embedding; text dropout zeros the umt5 kv.
# ══════════════════════════════════════════════════════════════════════════════
def _run_one_step[SEQ: Int](
    smp: BerniniSample,
    base: Wan22StackBase, mut loader: TurboPlannedLoader, mut lora: Wan22LoraSet,
    mut ema_a: List[List[Float32]], mut ema_b: List[List[Float32]],
    step: Int, sample_idx: Int, seed: UInt64, stochastic: Bool,
    drop_text: Bool, drop_img: Bool, drop_video: Bool,
    lr: Float32, max_grad_norm: Float32,
    beta1: Float32, beta2: Float32, opt_eps: Float32, weight_decay: Float32,
    ema_decay: Float32,
    mut adamw_state: Optional[LoraAdamWPlainDeviceState], wan_devnative: Bool,
    ctx: DeviceContext,
) raises -> StepOut:
    # timestep sigma: overfit pin (per-sample fixed) or Bernini per-step re-draw.
    var t = smp.pin_sigma
    if stochastic:
        t = bernini_sample_sigma(
            smp.is_mode, smp.shift,
            seed + UInt64(sample_idx) * UInt64(1000003)
                 + UInt64(step) * UInt64(2654435761) + 1,
            LOGIT_MEAN, LOGIT_STD, MODE_SCALE, smp.win,
        )

    var n_tgt = smp.n_tgt_tokens
    var ntgt_ch = n_tgt * IN_CH

    # ── flow-match noise on the TARGET only (conditioning stays clean) ───────────
    var noise_seed = seed * UInt64(2000003) + UInt64(sample_idx) * 100003 + 3
    if stochastic:
        noise_seed = seed * UInt64(2000003) + UInt64(sample_idx) * 100003 \
                     + UInt64(step) * 104729 + 3
    var noise = _noise(ntgt_ch, noise_seed)
    var tgt_in = List[Float32]()
    var target = List[Float32]()
    for i in range(ntgt_ch):
        tgt_in.append((Float32(1.0) - t) * smp.x0[i] + t * noise[i])
        target.append(noise[i] - smp.x0[i])

    # ── ASSEMBLE packed [cond_1..N (clean; dropped seg -> null/zero) | target] ────
    var model_in = List[Float32]()
    var off = 0
    for si in range(len(smp.seg_f)):
        var seg_tok_ch = smp.seg_f[si] * smp.seg_h[si] * smp.seg_w[si] * IN_CH
        var is_video = smp.seg_f[si] > 1
        var drop = (is_video and drop_video) or ((not is_video) and drop_img)
        if drop:
            for _ in range(seg_tok_ch):
                model_in.append(Float32(0.0))
        else:
            for j in range(seg_tok_ch):
                model_in.append(smp.cond_tokens[off + j])
        off += seg_tok_ch
    for i in range(len(tgt_in)):
        model_in.append(tgt_in[i])
    if len(model_in) != SEQ * IN_CH:
        raise Error("packed model_in len != SEQ*IN_CH")

    # ── text: dropped -> null/empty (zero) embedding ─────────────────────────────
    var txt_in = List[Float32]()
    if drop_text:
        for _ in range(len(smp.txt_tokens)):
            txt_in.append(Float32(0.0))
    else:
        txt_in = smp.txt_tokens.copy()

    var t_model = t * Float32(1000.0) + Float32(1.0)

    var fwd = wan22_stack_lora_forward_offload[H, Dh, SEQ, TXT](
        model_in.copy(), txt_in.copy(), t_model,
        base, loader, lora, smp.cos_h.copy(), smp.sin_h.copy(),
        DIM, FFN, IN_CH, TEXT_DIM, OUT_CH, FREQ_DIM, EPS, ctx,
    )

    # ── velocity-MSE on the TARGET region ONLY (trailing n_tgt tokens) ───────────
    var nout = len(fwd.out)
    var tgt_off = (SEQ - n_tgt) * OUT_CH
    var ntgt = n_tgt * OUT_CH
    var inv_n = Float32(2.0) / Float32(ntgt)
    var d_out = List[Float32]()
    for _ in range(tgt_off):
        d_out.append(Float32(0.0))
    var loss = Float32(0.0)
    for i in range(ntgt):
        var pred = fwd.out[tgt_off + i]
        var diff = pred - target[i]
        loss += diff * diff
        d_out.append(inv_n * diff)
    loss = loss / Float32(ntgt)
    if len(d_out) != nout:
        raise Error("d_out len != stack out len")

    var absum = Float32(0.0)
    var gn = Float32(0.0)
    var nonfinite_grads = 0

    if wan_devnative:
        # ── DEVICE-NATIVE path: recompute + device grads -> resident device-grad
        #    AdamW (no host grad round-trip). grad-norm/clip done on device inside
        #    the optimizer; params sync back to the host LoRA mirror each step so
        #    the (host-LoRA) forward stays current. ──
        var dgrads = wan22_stack_lora_backward_offload_devnative[H, Dh, SEQ, TXT](
            d_out, model_in.copy(), txt_in.copy(),
            base, loader, lora, smp.cos_h.copy(), smp.sin_h.copy(), fwd,
            DIM, FFN, IN_CH, TEXT_DIM, OUT_CH, FREQ_DIM, EPS, ctx,
        )
        gn = fused_lora_adamw_plain_step_resident_device_grads(
            adamw_state.value(), lora.ad, dgrads.grad_indices,
            dgrads.d_a, dgrads.d_b, step + 1, lr,
            beta1, beta2, opt_eps, weight_decay, ctx,
            Float32(1.0), True, max_grad_norm,
        )
    else:
        var grads = wan22_stack_lora_backward_offload[H, Dh, SEQ, TXT](
            d_out, model_in.copy(), txt_in.copy(),
            base, loader, lora, smp.cos_h.copy(), smp.sin_h.copy(), fwd,
            DIM, FFN, IN_CH, TEXT_DIM, OUT_CH, FREQ_DIM, EPS, ctx,
        )
        absum = _grad_abs_sum(grads)
        gn = _clip(grads, max_grad_norm)
        nonfinite_grads = grads.nonfinite_lora_grads
        wan22_lora_adamw_step(lora, grads, step + 1, lr, ctx,
                              beta1, beta2, opt_eps, weight_decay)

    # ── EMA shadow: ema = decay*ema + (1-decay)*live (host F32) ──────────────────
    var one_m = Float32(1.0) - ema_decay
    for i in range(len(lora.ad)):
        for j in range(len(lora.ad[i].a)):
            ema_a[i][j] = ema_decay * ema_a[i][j] + one_m * Float32(lora.ad[i].a[j])
        for j in range(len(lora.ad[i].b)):
            ema_b[i][j] = ema_decay * ema_b[i][j] + one_m * Float32(lora.ad[i].b[j])

    return StepOut(loss, gn, absum, nonfinite_grads, t)


# ══════════════════════════════════════════════════════════════════════════════
# Multi-sample driver: round-robin over samples[0..M-1], each step using THAT
# sample's task/recipe/cond-layout and dispatching on ITS packed seq_len.
# ══════════════════════════════════════════════════════════════════════════════
def _train_run(
    var samples: List[BerniniSample],
    base: Wan22StackBase, mut loader: TurboPlannedLoader, mut lora: Wan22LoraSet,
    steps: Int, seed: UInt64, stochastic: Bool,
    prob_text: Float32, prob_img: Float32, prob_video: Float32,
    lr: Float32, max_grad_norm: Float32,
    beta1: Float32, beta2: Float32, opt_eps: Float32, weight_decay: Float32,
    ema_decay: Float32, out_path: String, ctx: DeviceContext,
) raises:
    var M = len(samples)
    var first_loss = Float32(0.0)
    var last_loss = Float32(0.0)
    var min_loss = Float32(0.0)
    var have_first = False
    var train_start = perf_counter_ns()

    # EMA shadow init (host F32, mirrors each adapter's live BF16 a/b).
    var ema_a = List[List[Float32]]()
    var ema_b = List[List[Float32]]()
    for i in range(len(lora.ad)):
        var a = List[Float32]()
        for j in range(len(lora.ad[i].a)):
            a.append(Float32(lora.ad[i].a[j]))
        ema_a.append(a^)
        var b = List[Float32]()
        for j in range(len(lora.ad[i].b)):
            b.append(Float32(lora.ad[i].b[j]))
        ema_b.append(b^)

    # WAN_DEVNATIVE (Phase-1b): device-native recompute backward + resident
    # device-grad AdamW. Runtime env flag — BOTH paths are always compiled and the
    # host hand-chain stays the DEFAULT (env unset -> off). The resident optimizer
    # state (device P/M/V) is initialized once here and reused every step.
    var wan_devnative = _env_is_set("WAN_DEVNATIVE")
    var adamw_state = Optional[LoraAdamWPlainDeviceState](None)
    if wan_devnative:
        print("[bernini-r] WAN_DEVNATIVE=1 — device-native backward + resident device-grad AdamW")
        adamw_state = Optional[LoraAdamWPlainDeviceState](
            wan22_lora_adamw_state_init(lora, ctx)
        )

    print("")
    print("step  smp  task     seq  sigma      target_MSE      grad_norm    grad_absum   sec")
    for step in range(steps):
        var s = step % M
        var t0 = perf_counter_ns()

        # ── seeded, deterministic per-(step,sample) condition dropout ────────────
        var dbase = seed + UInt64(step) * UInt64(0x9E3779B1) \
                         + UInt64(s) * UInt64(0x85EBCA77)
        var drop_text = prob_text > Float32(0.0) \
            and _u01(dbase + UInt64(101)) < prob_text
        var drop_img = prob_img > Float32(0.0) \
            and _u01(dbase + UInt64(202)) < prob_img
        var drop_video = prob_video > Float32(0.0) \
            and _u01(dbase + UInt64(303)) < prob_video

        var seq = samples[s].seq_len
        var r: StepOut
        if seq == 256:
            r = _run_one_step[256](samples[s], base, loader, lora, ema_a, ema_b,
                step, s, seed, stochastic, drop_text, drop_img, drop_video,
                lr, max_grad_norm, beta1, beta2, opt_eps, weight_decay, ema_decay,
                adamw_state, wan_devnative, ctx)
        elif seq == 320:
            r = _run_one_step[320](samples[s], base, loader, lora, ema_a, ema_b,
                step, s, seed, stochastic, drop_text, drop_img, drop_video,
                lr, max_grad_norm, beta1, beta2, opt_eps, weight_decay, ema_decay,
                adamw_state, wan_devnative, ctx)
        elif seq == 384:
            r = _run_one_step[384](samples[s], base, loader, lora, ema_a, ema_b,
                step, s, seed, stochastic, drop_text, drop_img, drop_video,
                lr, max_grad_norm, beta1, beta2, opt_eps, weight_decay, ema_decay,
                adamw_state, wan_devnative, ctx)
        elif seq == 448:
            r = _run_one_step[448](samples[s], base, loader, lora, ema_a, ema_b,
                step, s, seed, stochastic, drop_text, drop_img, drop_video,
                lr, max_grad_norm, beta1, beta2, opt_eps, weight_decay, ema_decay,
                adamw_state, wan_devnative, ctx)
        elif seq == 512:
            r = _run_one_step[512](samples[s], base, loader, lora, ema_a, ema_b,
                step, s, seed, stochastic, drop_text, drop_img, drop_video,
                lr, max_grad_norm, beta1, beta2, opt_eps, weight_decay, ema_decay,
                adamw_state, wan_devnative, ctx)
        else:
            raise Error(String("unsupported packed seq_len ") + String(seq) +
                        String(" (sample ") + String(s) +
                        String(", task ") + samples[s].task_name +
                        String("); add a monomorphization"))

        if not have_first:
            first_loss = r.loss
            min_loss = r.loss
            have_first = True
        if r.loss < min_loss:
            min_loss = r.loss
        last_loss = r.loss

        var secs = Float64(perf_counter_ns() - t0) / 1.0e9
        var dstr = String("")
        if drop_text or drop_img or drop_video:
            dstr = String("  drop[")
            if drop_text: dstr += String("txt ")
            if drop_img: dstr += String("img ")
            if drop_video: dstr += String("vid ")
            dstr += String("]")
        print(step, "  ", s, "  ", samples[s].task_name, "  ", seq, "  ",
              r.sigma, "  ", r.loss, "  ", r.gn, "  ", r.absum, "  ", secs, dstr)
        if r.nonfinite != 0:
            print("  !! nonfinite lora grads =", r.nonfinite)

    var npairs = save_wan22_lora(lora, out_path, ctx)
    var meta = List[Float32]()
    meta.append(Float32(steps))
    meta.append(Float32(Int(seed)))
    var state_path = out_path + String(".state")
    var nstate = save_wan22_lora_state(lora, state_path, ctx, meta^)
    var ema_path = out_path + String(".ema.safetensors")
    var nema = _save_ema_sibling(lora, ema_a, ema_b, ema_path, ctx)
    print("")
    print("[save] wrote", npairs, "PEFT pairs ->", out_path)
    print("[save] wrote", nstate, "state tensors ->", state_path)
    print("[save] wrote", nema, "EMA (decay=", ema_decay, ") PEFT pairs ->", ema_path)
    print("trained steps=", steps, " samples(M)=", M, " first target_MSE=", first_loss,
          " last=", last_loss, " min=", min_loss,
          " total sec=", Float64(perf_counter_ns() - train_start) / 1.0e9)
    print("RESULT: bernini-r multi-sample CONDITIONED renderer trainer ran",
          "(round-robin over", M, "sample(s); per-sample task/recipe/cond-layout;",
          "src-id rope; velocity-MSE on target region only).")


def main() raises:
    var ctx = DeviceContext()

    var args = argv()
    var config_path = String(DEFAULT_CONFIG)
    if len(args) > 1:
        config_path = String(args[1])
    if not _file_exists(config_path):
        raise Error(String("config not found: ") + config_path)
    var cfg = read_model_config(config_path)

    # ── task selection (legacy / synthetic / cache-without-task_id fallback) ─────
    var task = _env_str(String("BERNINI_TASK"))
    if task.byte_length() == 0:
        task = String("t2v")

    var stochastic = _env_is_set(String("BERNINI_STOCHASTIC"))
    var force_no_cond = _env_is_set(String("BERNINI_NO_COND"))

    # ── condition dropout probabilities (Bernini config 0.1; C13 default-off) ────
    var prob_text = _env_float(String("BERNINI_TEXT_DROPOUT"), Float32(0.0))
    var prob_img = _env_float(String("BERNINI_IMG_DROPOUT"), Float32(0.0))
    var prob_video = _env_float(String("BERNINI_VIDEO_DROPOUT"), Float32(0.0))

    # ── data source ──────────────────────────────────────────────────────────────
    var cache_dir = cfg.dataset_cache_dir
    var use_cache = cache_dir.byte_length() > 0 and _file_exists(
        _cache_sample_path(cache_dir, 0)
    )

    var rank = cfg.lora_rank if cfg.lora_rank > 0 else 16
    var alpha = cfg.lora_alpha if cfg.lora_alpha > Float32(0.0) else Float32(16.0)
    var lr = cfg.lr if cfg.lr > Float32(0.0) else DEFAULT_LR
    var max_grad_norm = cfg.max_grad_norm if cfg.max_grad_norm > Float32(0.0) \
        else DEFAULT_MAX_GRAD_NORM
    var steps = cfg.max_steps if cfg.max_steps > 0 else 8
    var seed = cfg.seed
    var beta1 = cfg.beta1 if cfg.beta1 > Float32(0.0) else Float32(0.9)
    var beta2 = cfg.beta2 if cfg.beta2 > Float32(0.0) else Float32(0.999)
    var opt_eps = cfg.eps if cfg.eps > Float32(0.0) else Float32(1.0e-8)
    var weight_decay = cfg.weight_decay if cfg.weight_decay >= Float32(0.0) \
        else Float32(0.01)
    var ema_decay = DEFAULT_EMA_DECAY

    var ckpt = cfg.checkpoint if cfg.checkpoint.byte_length() > 0 \
        else String(DEFAULT_CKPT_LOW)
    var out_path = cfg.output_model_destination
    if out_path.byte_length() == 0:
        out_path = String(
            "/home/alex/mojodiffusion/output/bernini_r_lora/bernini_r_cond_lora.safetensors"
        )

    # ── build the per-sample bundles (round-robin schedule) ──────────────────────
    var samples = List[BerniniSample]()
    var M = 1
    if use_cache:
        M = _count_cache_samples(cache_dir)
        if M < 1:
            M = 1
        for i in range(M):
            samples.append(
                _build_cache_sample(cache_dir, i, task, force_no_cond, seed, ctx)
            )
    else:
        samples.append(_build_synth_sample(task, force_no_cond, seed, ctx))

    print("==== Bernini-R multi-sample CONDITIONED LoRA trainer (12-task renderer) ====")
    print("data:", ("REAL cache " + cache_dir) if use_cache else String("SYNTHETIC (no cache)"))
    print("samples(M)=", len(samples), " round-robin over steps")
    print("condition dropout: text=", prob_text, " img=", prob_img, " video=", prob_video,
          " (C13 default-off at 0.0)")
    if stochastic:
        print("timestep/noise: STOCHASTIC Bernini per-step (sigma re-drawn each step)")
    else:
        print("timestep/noise: OVERFIT PIN (per-sample fixed sigma + fixed noise)")
    print("arch: dim=", DIM, " blocks=", NUM_BLOCKS, " heads=", H, " head_dim=", Dh,
          " ffn=", FFN, " in_ch=", IN_CH, " out_ch=", OUT_CH, " TXT=", TXT)
    print("lora: rank=", rank, " alpha=", alpha, " (10/block: 8 attn + ffn.0 + ffn.2)")
    print("optim: lr=", lr, " max_grad_norm=", max_grad_norm, " ema_decay=", ema_decay)
    print("steps:", steps, " seed:", seed)
    for i in range(len(samples)):
        var sm = samples[i].copy()
        print("  sample", i, ": task=", sm.task_name, " (id=", sm.task_id, ")",
              " shift=", sm.shift,
              " weighting=", ("mode(video)" if sm.is_mode else "logit_normal(image)"),
              " n_cond=", len(sm.seg_f), " no_cond=", sm.no_cond,
              " tgt_tokens=", sm.n_tgt_tokens, " seq_len=", sm.seq_len,
              " pin_sigma=", sm.pin_sigma)
        for k in range(len(sm.seg_f)):
            print("      cond seg", k, " src_id=", sm.seg_src[k], " kind=", sm.seg_kind[k],
                  " grid(f,h,w)=(", sm.seg_f[k], ",", sm.seg_h[k], ",", sm.seg_w[k], ")",
                  " tokens=", sm.seg_f[k] * sm.seg_h[k] * sm.seg_w[k])
    print("ckpt (low):", ckpt)

    # ── checkpoint-absent guard (type-check path, exit 0) ────────────────────────
    if not _file_exists(_base_file_for(ckpt)):
        print("")
        print("[bernini-r] fp8 base not present, skipping real step:")
        print("        ", ckpt)
        print("        (recipe/geometry/packing + seq_len monomorphizations",
              "type-checked; re-run once the fp8 cache lands.)")
        print("RESULT: bernini-r cond trainer wired OK (weights-absent path, exit 0)")
        return

    # ══════════════════════════════════════════════════════════════════════════
    # REAL WEIGHTED RUN — resident base + block-swap loader (single low expert).
    # ══════════════════════════════════════════════════════════════════════════
    print("")
    print("[bernini-r] fp8 base present — building resident base + block-swap loader")
    var off_cfg = OffloadConfig.synchronous_single()
    var base_st = SafeTensors.open(_base_file_for(ckpt))
    var ckpt_prefix = detect_wan22_prefix(base_st)
    print("[bernini-r] detected checkpoint key prefix: '", ckpt_prefix, "'", sep="")
    var base = load_wan22_stack_base(base_st, ctx, ckpt_prefix)
    var plan = build_wan22_block_plan(NUM_BLOCKS, ckpt_prefix)
    var loader = TurboPlannedLoader.open(ckpt, plan^, off_cfg, ctx, False)

    var lora = build_wan22_lora_set(NUM_BLOCKS, DIM, FFN, rank, alpha)
    print("[lora] adapters:", wan22_total_adapters(lora))

    _train_run(samples^, base, loader, lora, steps, seed, stochastic,
               prob_text, prob_img, prob_video, lr, max_grad_norm,
               beta1, beta2, opt_eps, weight_decay, ema_decay, out_path, ctx)
