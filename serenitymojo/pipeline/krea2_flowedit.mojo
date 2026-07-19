# pipeline/krea2_flowedit.mojo — FlowEdit: training-free instruction editing (task #20).
#
# FlowEdit (ICCV'25, arXiv 2412.08629, github.com/fallenshock/FlowEdit, MIT) edits a
# REAL image via a direct source→target ODE — NO inversion, NO training, sampler-only.
# Proven on FLUX (krea2's closest cousin). PURE BASE MODEL: no LoRA, no img_in_ref.
#
# Algorithm (paper Alg. 1, n_avg=1):
#   Z0_src = VAE-encoded source (normalized latent space), Z_edit = Z0_src (F32).
#   For step i with t = ts[i], t_prev = ts[i+1] (krea2 schedule, t: 1 → 0), ONLY for
#   i in [T - n_max, T - n_min)  (FLUX config T=28, n_max=24, n_min=0 → SKIP the
#   first 4 highest-t steps; Z_edit stays = Z0_src during skipped steps):
#     1. N ~ randn (fresh per step, seed+i deterministic)
#     2. Zt_src = (1-t)*Z0_src + t*N                (noised source at level t)
#     3. Zt_tgt = Z_edit + (Zt_src - Z0_src)        (target candidate)
#     4. V_src  = v(Zt_src, t, c_src)  CFG 1.5      (krea2 real CFG, src pos/neg ctx)
#     5. V_tgt  = v(Zt_tgt, t, c_tgt)  CFG 5.5
#     6. Z_edit += (t_prev - t) * (V_tgt - V_src)   (Euler on the velocity DIFFERENCE;
#                                                    t_prev - t < 0, same sign
#                                                    convention as krea2_euler_step)
#   Decode Z_edit → PNG.
#
# VARIABLE-LT CONTEXTS: FlowEdit needs FOUR contexts (src pos/neg, tgt pos/neg) with
# DIFFERENT natural token lengths. We reuse the TRAINER's length-bucket contract
# (krea2_cache_reader.sample_padded): zero-row-pad each raw [1,lt,12,2560] stack to a
# shared comptime LT_SHARED, and pass real_text_len=lt to krea2_forward, which
# reorders to [TXT_real | IMG | TXT_pad] and flash-padmasks the tail — the exact
# padding semantics training used (txtfusion unmasked over zero pad rows, pad text
# masked out of the main blocks).
#
# TWO-PROCESS SPLIT (same as krea2_pipeline): run krea2_encode_cli TWICE first —
# once for the source description, once for the target (edited) description.
#
# AUTO-MASK (task #21 phase A, UniEdit-Flow / Follow-Your-Shape idea): the
# per-token velocity difference ||V_tgt - V_src||_2 — computed FREE every active
# step — localizes the edit. With --auto-mask we accumulate that saliency across
# active steps (running sum; quantile thresholding is scale-invariant so no
# normalization is needed), threshold at quantile --mask-q on the 32x32 token
# grid, dilate --mask-dilate times (3x3 neighborhood) to avoid seams, and after
# each Euler update HARD-COPY the source latent Z0_src back into every token
# OUTSIDE the mask. First --mask-warmup active steps only accumulate saliency
# (mask needs signal before it's trustworthy). The final mask is saved as
# <out>_mask.png (32x32 upscaled x16 nearest -> 512) for inspection.
#
# Run (GPU):
#   <bin> <src_512.safetensors> <src_ctx_pos> <src_ctx_neg> <tgt_ctx_pos>
#         <tgt_ctx_neg> <out.png> [--steps 28] [--nmax 24] [--nmin 0]
#         [--src-cfg 1.5] [--tgt-cfg 5.5] [--seed N] [--i8-resident-blocks N]
#         [--auto-mask] [--mask-q 0.7] [--mask-dilate 1] [--mask-warmup 4]
#
# Mojo 1.0.0b1, NVIDIA GPU. Inference-only.

from std.gpu.host import DeviceContext
from std.collections import Optional
from std.math import sqrt
from std.sys import argv
from std.time import perf_counter_ns

from serenitymojo.lora import LoraSet
from serenitymojo.models.krea2.krea2_stack import build_krea2_resident_int8
from serenitymojo.models.dit.krea2_dit import (
    Krea2ResidentFp8, Krea2ResidentInt8, Krea2HostInt8Inf,
    Krea2LoraSideCache, Krea2SharedResident, build_krea2_shared_resident,
    build_krea2_host_int8_inf, krea2_forward,
)
from serenitymojo.offload.vmm_cuda import cu_mempool_trim_current
from serenitymojo.models.krea2.krea2_int8_cache import (
    krea2_int8_cache_path, krea2_int8_cache_valid,
    load_krea2_int8_cache_resident, load_krea2_int8_cache_host,
    load_krea2_int8_cache_shared, save_krea2_int8_cache,
)
from serenitymojo.models.krea2.krea2_prepare_cache import (
    _mean_ch,
    _std_ch,
    _normalize_latent,
    KREA2_VAE_ENC_FILE,
)
from serenitymojo.models.vae.qwenimage_encoder import QwenImageVaeEncoder
from serenitymojo.models.vae.qwenimage_decoder import QwenImageVaeDecoder

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.cap_cache import load_tensor_bin
from serenitymojo.ops.tensor_algebra import (
    reshape, permute, mul_scalar, add, sub, concat, zeros_device,
)
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.torch_bf16 import torch_f32_to_bf16_rne
from serenitymojo.ops.random import randn
from serenitymojo.sampling.krea2_sampler import (
    krea2_timesteps,
    krea2_packed_seq_len,
    krea2_cfg,
)
from serenitymojo.image.png import save_png, ValueRange
from serenitymojo.pipeline.krea2_paths import (
    KREA2_RAW,
    KREA2_VAE_DIR,
    KREA2_RAW_KEY_PREFIX,
)


# ── RUN CONFIG (512x512, same as the edit-training resolution) ─────────────────
comptime HEIGHT = 512
comptime WIDTH = 512
comptime LH = HEIGHT // 8      # 64
comptime LW = WIDTH // 8       # 64
comptime IMGLEN = (LH // 2) * (LW // 2)   # 1024
comptime FEATURES = 6144
comptime STEPS_DEFAULT = 28               # FlowEdit FLUX config T=28
comptime NMAX_DEFAULT = 24                # skip the first T-n_max=4 highest-t steps
comptime NMIN_DEFAULT = 0
comptime SRC_CFG_DEFAULT = Float32(1.5)
comptime TGT_CFG_DEFAULT = Float32(5.5)
comptime SEED_DEFAULT = UInt64(88888)
comptime NBLOCKS = 28

# Shared PADDED text length: every context (src/tgt × pos/neg) is zero-row-padded
# to this bucket; krea2_forward gets the real length per context. ONE comptime
# instantiation covers all four contexts. Fail-loud if a prompt exceeds it.
comptime LT_SHARED = 32
comptime LFULL = LT_SHARED + IMGLEN                 # 1056
comptime LPAD = ((LFULL + 255) // 256) * 256        # 1280

comptime KREA2_TXT_LAYERS = 12
comptime KREA2_TXT_DIM = 2560


def _load_context_padded(
    path: String, name: String, ctx: DeviceContext
) raises -> Tuple[Tensor, Int]:
    """Load a krea2_encode_cli context bin [1, lt, 12, 2560] bf16 and zero-row-pad
    it to [1, LT_SHARED, 12, 2560] — EXACTLY the trainer's sample_padded contract
    (krea2_cache_reader.mojo:437-443). Returns (padded stack, real lt)."""
    var stack = load_tensor_bin(path, ctx)
    var sh = stack.shape()
    if len(sh) != 4 or sh[0] != 1 or sh[2] != KREA2_TXT_LAYERS or sh[3] != KREA2_TXT_DIM:
        raise Error(
            String("[flowedit] ") + name + " cached context has unexpected shape "
            + "(expected [1, LT, 12, 2560]); regenerate with krea2_encode_cli: " + path
        )
    var lt = sh[1]
    print("[flowedit] ", name, " context LT =", lt, " (pad to ", LT_SHARED, ")")
    if lt > LT_SHARED:
        raise Error(
            String("[flowedit] ") + name + " LT=" + String(lt) + " > LT_SHARED="
            + String(LT_SHARED) + " — raise LT_SHARED (and rebuild)."
        )
    if lt == LT_SHARED:
        return (stack^, lt)
    var pad = zeros_device(
        [1, LT_SHARED - lt, KREA2_TXT_LAYERS, KREA2_TXT_DIM], STDtype.BF16, ctx
    )
    var padded = concat(1, ctx, stack, pad)          # [1, LT_SHARED, 12, 2560]
    return (padded^, lt)


def _build_pos(ctx: DeviceContext) raises -> Tensor:
    """pos [1, LT_SHARED+IMGLEN, 3] f32 = cat(txt zeros, img (0,gh_i,gw_i) grid).
    Shared by all four contexts (text pos rows are all-zero; krea2_forward reorders
    internally per real_text_len)."""
    var gh = LH // 2
    var gw = LW // 2
    var host = List[Float32]()
    for _ in range(LT_SHARED * 3):
        host.append(Float32(0.0))
    for hi in range(gh):
        for wi in range(gw):
            host.append(Float32(0.0))
            host.append(Float32(hi))
            host.append(Float32(wi))
    return Tensor.from_host(host^, [1, LFULL, 3], STDtype.F32, ctx)


def _patchify_latent(latent_nchw: Tensor, ctx: DeviceContext) raises -> Tensor:
    """[1,16,LH,LW] -> [1, IMGLEN, 64] ((c ph pw) patchify, ph=pw=2)."""
    var gh = LH // 2
    var gw = LW // 2
    var x6 = reshape(latent_nchw, [1, 16, gh, 2, gw, 2], ctx)
    var xp = permute(x6, [0, 2, 4, 1, 3, 5], ctx)   # [1, gh, gw, 16, 2, 2]
    return reshape(xp, [1, gh * gw, 64], ctx)        # [1, IMGLEN, 64]


def _unpatch(tokens: Tensor, ctx: DeviceContext) raises -> Tensor:
    var gh = LH // 2
    var gw = LW // 2
    var x6 = reshape(tokens, [1, gh, gw, 16, 2, 2], ctx)
    var xp = permute(x6, [0, 3, 1, 4, 2, 5], ctx)   # [1, 16, gh, 2, gw, 2]
    return reshape(xp, [1, 16, gh * 2, gw * 2], ctx)  # [1, 16, LH, LW]


# ── AUTO-MASK helpers (host-side; 1024 tokens — trivially cheap vs 4 DiT
#    forwards/step). Latent [1,16,LH,LW] host layout: c*LH*LW + y*LW + x. Token
#    (gy,gx) on the (LH/2)x(LW/2) grid owns the 16-channel 2x2 latent block at
#    (2gy..2gy+1, 2gx..2gx+1) — exactly the DiT's patchify token. ───────────────

comptime TOK_GH = LH // 2               # 32
comptime TOK_GW = LW // 2               # 32
comptime NTOK = TOK_GH * TOK_GW         # 1024 == IMGLEN
comptime LATPLANE = LH * LW             # 4096


def _accum_saliency(dv_host: List[Float32], mut sal: List[Float32]):
    """sal[j] += ||dv token j||_2 over the 16x2x2 latent block (64 values —
    identical content to the [1,IMGLEN,64] token row, just latent layout)."""
    for gy in range(TOK_GH):
        for gx in range(TOK_GW):
            var acc = Float32(0.0)
            for c in range(16):
                var base = c * LATPLANE
                for ph in range(2):
                    var row = base + (2 * gy + ph) * LW + 2 * gx
                    for pw in range(2):
                        var v = dv_host[row + pw]
                        acc += v * v
            sal[gy * TOK_GW + gx] += sqrt(acc)


def _mask_from_saliency(
    sal: List[Float32], mask_q: Float32, dilate: Int
) -> List[Bool]:
    """Threshold accumulated saliency at quantile mask_q (tokens >= the q-th
    sorted value are IN the edit mask), then dilate `dilate` times with a 3x3
    neighborhood on the token grid."""
    # quantile threshold (insertion sort; NTOK=1024 host floats).
    var sorted_sal = sal.copy()
    for i in range(1, len(sorted_sal)):
        var key = sorted_sal[i]
        var j = i - 1
        while j >= 0 and sorted_sal[j] > key:
            sorted_sal[j + 1] = sorted_sal[j]
            j -= 1
        sorted_sal[j + 1] = key
    var qi = Int(mask_q * Float32(NTOK))
    if qi < 0:
        qi = 0
    if qi > NTOK - 1:
        qi = NTOK - 1
    var thr = sorted_sal[qi]
    var mask = List[Bool]()
    for j in range(NTOK):
        mask.append(sal[j] >= thr)
    # dilation: token is masked if ANY 3x3 neighbor was masked.
    for _ in range(dilate):
        var grown = mask.copy()
        for gy in range(TOK_GH):
            for gx in range(TOK_GW):
                if grown[gy * TOK_GW + gx]:
                    continue
                var hit = False
                for dy in range(-1, 2):
                    var ny = gy + dy
                    if ny < 0 or ny >= TOK_GH:
                        continue
                    for dx in range(-1, 2):
                        var nx = gx + dx
                        if nx < 0 or nx >= TOK_GW:
                            continue
                        if mask[ny * TOK_GW + nx]:
                            hit = True
                if hit:
                    grown[gy * TOK_GW + gx] = True
        mask = grown^
    return mask^


def _blend_outside_mask(
    mut z_host: List[Float32], z0_host: List[Float32], mask: List[Bool]
):
    """HARD-COPY source latent into every token OUTSIDE the mask (pixel-exact
    background in latent space; only VAE-decode receptive-field bleed remains)."""
    for gy in range(TOK_GH):
        for gx in range(TOK_GW):
            if mask[gy * TOK_GW + gx]:
                continue
            for c in range(16):
                var base = c * LATPLANE
                for ph in range(2):
                    var row = base + (2 * gy + ph) * LW + 2 * gx
                    for pw in range(2):
                        z_host[row + pw] = z0_host[row + pw]


def _save_mask_png(
    mask: List[Bool], out_png: String, ctx: DeviceContext
) raises -> String:
    """Save the token mask as <out>_mask.png: 32x32 upscaled x16 nearest to
    512x512, white = edit region."""
    var mask_path = out_png
    if mask_path.endswith(".png"):
        mask_path = String(mask_path.removesuffix(".png"))
    mask_path += "_mask.png"
    var host = List[Float32]()
    var scale_h = HEIGHT // TOK_GH   # 16
    var scale_w = WIDTH // TOK_GW    # 16
    for _ in range(3):               # R, G, B planes identical
        for y in range(HEIGHT):
            var gy = y // scale_h
            for x in range(WIDTH):
                var gx = x // scale_w
                if mask[gy * TOK_GW + gx]:
                    host.append(Float32(1.0))
                else:
                    host.append(Float32(0.0))
    var img = Tensor.from_host(host^, [1, 3, HEIGHT, WIDTH], STDtype.F32, ctx)
    save_png(img, mask_path, ctx, ValueRange.UNIT)
    return mask_path


def _encode_source_latent(src_path: String, ctx: DeviceContext) raises -> Tensor:
    """Staged source [1,3,512,512] f32 -> NORMALIZED VAE latent [1,16,LH,LW] F32
    (encode_mean -> (z-mean)/std — the same normalized space the denoise carrier
    and the decoder's internal unnormalize use). This is Z0_src."""
    var imgs = ShardedSafeTensors.open(src_path)
    var img_f32 = Tensor.from_view(imgs.tensor_view(String("image")), ctx)  # [1,3,512,512]
    var img = cast_tensor(img_f32, STDtype.BF16, ctx)
    ctx.synchronize(); var _t0 = Int(perf_counter_ns())
    var enc = QwenImageVaeEncoder[HEIGHT, WIDTH].load(KREA2_VAE_ENC_FILE, ctx)
    ctx.synchronize(); var _t1 = Int(perf_counter_ns())
    print("[flowedit] PHASE vae-enc-load =", Float64(_t1 - _t0) / 1e9, "s")
    var lat_mean = enc.encode_mean(img, ctx)                # [1,16,LH,LW] BF16
    ctx.synchronize()
    print("[flowedit] PHASE vae-encode =",
          Float64(Int(perf_counter_ns()) - _t1) / 1e9, "s")
    var lat_f32 = cast_tensor(lat_mean, STDtype.F32, ctx)
    var mean_ch = _mean_ch(ctx)
    var std_ch = _std_ch(ctx)
    var clean_f32 = _normalize_latent(lat_f32, mean_ch, std_ch, ctx)  # [1,16,LH,LW] F32
    print("[flowedit] Z0_src encoded [1,16,", LH, ",", LW, "] F32 (normalized)")
    return clean_f32^


def _velocity(
    st: ShardedSafeTensors,
    latent_f32: Tensor,        # [1,16,LH,LW] F32
    ctx_pos: Tensor, lt_pos: Int,
    ctx_neg: Tensor, lt_neg: Int,
    pos_grid: Tensor,          # [1, LFULL, 3] F32 (shared)
    t_t: Tensor,               # [1] F32
    cfg_scale: Float32,
    resident_i8: Optional[Krea2ResidentInt8],
    host_i8: Optional[Krea2HostInt8Inf],
    shared: Optional[Krea2SharedResident],
    ctx: DeviceContext,
) raises -> Tensor:
    """One CFG velocity in latent space: [1,16,LH,LW] F32. PURE BASE (no LoRA,
    no img_in_ref — FlowEdit is training-free)."""
    var tok_f32 = _patchify_latent(latent_f32, ctx)             # [1,IMGLEN,64] f32
    var tok = torch_f32_to_bf16_rne(tok_f32, ctx)               # bf16 feed
    var v_cond = krea2_forward[LFULL, LPAD, LT_SHARED, NBLOCKS](
        st, tok, ctx_pos, t_t, pos_grid, ctx, KREA2_RAW_KEY_PREFIX,
        Optional[LoraSet](None), Float32(1.0),
        Optional[Krea2ResidentFp8](None),
        Optional[Tensor](None), Optional[Tensor](None),
        resident_i8, host_i8,
        Optional[Krea2LoraSideCache](None), shared,
        Optional[Int](lt_pos),
    )
    var v_uncond = krea2_forward[LFULL, LPAD, LT_SHARED, NBLOCKS](
        st, tok, ctx_neg, t_t, pos_grid, ctx, KREA2_RAW_KEY_PREFIX,
        Optional[LoraSet](None), Float32(1.0),
        Optional[Krea2ResidentFp8](None),
        Optional[Tensor](None), Optional[Tensor](None),
        resident_i8, host_i8,
        Optional[Krea2LoraSideCache](None), shared,
        Optional[Int](lt_neg),
    )
    var v_bf16 = krea2_cfg(v_cond, v_uncond, cfg_scale, ctx)    # [1,IMGLEN,64] bf16
    var v_f32 = cast_tensor(v_bf16, STDtype.F32, ctx)
    return _unpatch(v_f32, ctx)                                 # [1,16,LH,LW] F32


def main() raises:
    var _t_launch = Int(perf_counter_ns())
    var ctx = DeviceContext()
    var args = argv()

    if len(args) < 7:
        raise Error(
            "usage: krea2_flowedit <src_512.safetensors> <src_ctx_pos> <src_ctx_neg>"
            " <tgt_ctx_pos> <tgt_ctx_neg> <out.png> [--steps 28] [--nmax 24]"
            " [--nmin 0] [--src-cfg 1.5] [--tgt-cfg 5.5] [--seed N]"
            " [--i8-resident-blocks N] [--auto-mask] [--mask-q 0.7]"
            " [--mask-dilate 1] [--mask-warmup 4]"
        )
    var src_path = String(args[1])
    var src_pos_path = String(args[2])
    var src_neg_path = String(args[3])
    var tgt_pos_path = String(args[4])
    var tgt_neg_path = String(args[5])
    var out_png = String(args[6])

    var steps = STEPS_DEFAULT
    var n_max = NMAX_DEFAULT
    var n_min = NMIN_DEFAULT
    var src_cfg = SRC_CFG_DEFAULT
    var tgt_cfg = TGT_CFG_DEFAULT
    var seed = SEED_DEFAULT
    var i8_res_blocks = 20
    var auto_mask = False
    var mask_q = Float32(0.7)
    var mask_dilate = 1
    var mask_warmup = 4
    for i in range(len(args)):
        if args[i] == String("--steps") and i + 1 < len(args):
            steps = Int(String(args[i + 1]))
        if args[i] == String("--nmax") and i + 1 < len(args):
            n_max = Int(String(args[i + 1]))
        if args[i] == String("--nmin") and i + 1 < len(args):
            n_min = Int(String(args[i + 1]))
        if args[i] == String("--src-cfg") and i + 1 < len(args):
            src_cfg = Float32(Float64(String(args[i + 1])))
        if args[i] == String("--tgt-cfg") and i + 1 < len(args):
            tgt_cfg = Float32(Float64(String(args[i + 1])))
        if args[i] == String("--seed") and i + 1 < len(args):
            seed = UInt64(Int(String(args[i + 1])))
        if args[i] == String("--i8-resident-blocks") and i + 1 < len(args):
            i8_res_blocks = Int(String(args[i + 1]))
        if args[i] == String("--auto-mask"):
            auto_mask = True
        if args[i] == String("--mask-q") and i + 1 < len(args):
            mask_q = Float32(Float64(String(args[i + 1])))
        if args[i] == String("--mask-dilate") and i + 1 < len(args):
            mask_dilate = Int(String(args[i + 1]))
        if args[i] == String("--mask-warmup") and i + 1 < len(args):
            mask_warmup = Int(String(args[i + 1]))
    if n_max > steps:
        n_max = steps
    if n_min < 0:
        n_min = 0

    print("[flowedit] FlowEdit", HEIGHT, "x", WIDTH, " steps=", steps,
          " n_max=", n_max, " n_min=", n_min, " src_cfg=", src_cfg,
          " tgt_cfg=", tgt_cfg, " seed=", seed)
    if auto_mask:
        print("[flowedit] AUTO-MASK on: q=", mask_q, " dilate=", mask_dilate,
              " warmup=", mask_warmup, " active steps")

    var _t_ph = Int(perf_counter_ns())

    # ── 1) source -> Z0_src (encode + free the encoder BEFORE the resident base).
    var z0_src = _encode_source_latent(src_path, ctx)           # [1,16,LH,LW] F32
    ctx.synchronize()
    print("[flowedit] PHASE source-encode =",
          Float64(Int(perf_counter_ns()) - _t_ph) / 1e9, "s")
    _t_ph = Int(perf_counter_ns())

    # ── 2) FOUR contexts, zero-row-padded to LT_SHARED (trainer contract) + pos.
    var src_pos_pair = _load_context_padded(src_pos_path, String("SRC_POS"), ctx)
    var src_neg_pair = _load_context_padded(src_neg_path, String("SRC_NEG"), ctx)
    var tgt_pos_pair = _load_context_padded(tgt_pos_path, String("TGT_POS"), ctx)
    var tgt_neg_pair = _load_context_padded(tgt_neg_path, String("TGT_NEG"), ctx)
    var ctx_src_pos = src_pos_pair[0].clone(ctx)
    var lt_src_pos = src_pos_pair[1]
    var ctx_src_neg = src_neg_pair[0].clone(ctx)
    var lt_src_neg = src_neg_pair[1]
    var ctx_tgt_pos = tgt_pos_pair[0].clone(ctx)
    var lt_tgt_pos = tgt_pos_pair[1]
    var ctx_tgt_neg = tgt_neg_pair[0].clone(ctx)
    var lt_tgt_neg = tgt_neg_pair[1]
    var pos_grid = _build_pos(ctx)

    ctx.synchronize()
    print("[flowedit] PHASE ctx-load =",
          Float64(Int(perf_counter_ns()) - _t_ph) / 1e9, "s")
    _t_ph = Int(perf_counter_ns())

    # ── 3) schedule (t: 1 -> 0), FlowEdit step window. ─────────────────────────
    var seq = krea2_packed_seq_len(HEIGHT, WIDTH)
    var ts = krea2_timesteps(seq, steps)
    var skip_before = steps - n_max      # steps [0:skip_before) are SKIPPED
    var stop_at = steps - n_min          # steps [stop_at:steps) are SKIPPED
    print("[flowedit] schedule seq =", seq, " steps =", len(ts) - 1,
          " active window = [", skip_before, ",", stop_at, ")",
          " ts[0]=", ts[0], " ts[-1]=", ts[len(ts) - 1])

    # ── 4) DiT base: int8 W8A8 hybrid + disk sidecar (the fast startup path). ──
    var st = ShardedSafeTensors.open(String(KREA2_RAW))
    if i8_res_blocks < 0:
        i8_res_blocks = 0
    if i8_res_blocks > NBLOCKS:
        i8_res_blocks = NBLOCKS
    ctx.synchronize()
    cu_mempool_trim_current(0)
    var resident_i8 = Optional[Krea2ResidentInt8](None)
    var host_i8 = Optional[Krea2HostInt8Inf](None)
    var shared = Optional[Krea2SharedResident](None)
    var i8_cache_path = krea2_int8_cache_path(String(KREA2_RAW))
    var i8_from_cache = False
    if krea2_int8_cache_valid(i8_cache_path, String(KREA2_RAW), NBLOCKS):
        print("[int8cache] fresh sidecar found:", i8_cache_path)
        if i8_res_blocks > 0:
            resident_i8 = Optional[Krea2ResidentInt8](
                load_krea2_int8_cache_resident(i8_cache_path, i8_res_blocks, ctx)
            )
        if i8_res_blocks < NBLOCKS:
            host_i8 = Optional[Krea2HostInt8Inf](
                load_krea2_int8_cache_host(i8_cache_path, NBLOCKS, i8_res_blocks, ctx)
            )
        shared = Optional[Krea2SharedResident](
            load_krea2_int8_cache_shared(i8_cache_path, ctx)
        )
        i8_from_cache = True
        print("[int8cache] loaded", NBLOCKS,
              "blocks + shared from sidecar (skipped quantize).")
    if not i8_from_cache:
        print("[flowedit] building int8 W8A8 base: resident", i8_res_blocks,
              "/", NBLOCKS, "blocks + pinned-host remainder (load-once) ...")
        if i8_res_blocks > 0:
            resident_i8 = Optional[Krea2ResidentInt8](
                build_krea2_resident_int8(
                    st, KREA2_RAW_KEY_PREFIX, NBLOCKS, i8_res_blocks, ctx
                )
            )
        if i8_res_blocks < NBLOCKS:
            host_i8 = Optional[Krea2HostInt8Inf](
                build_krea2_host_int8_inf(
                    st, KREA2_RAW_KEY_PREFIX, NBLOCKS, i8_res_blocks, ctx
                )
            )
        shared = Optional[Krea2SharedResident](
            build_krea2_shared_resident(st, KREA2_RAW_KEY_PREFIX, ctx)
        )
        try:
            save_krea2_int8_cache(
                resident_i8, host_i8, shared.value(), String(KREA2_RAW),
                i8_cache_path, NBLOCKS, ctx,
            )
            print("[int8cache] wrote sidecar for future fast launches:",
                  i8_cache_path)
        except e:
            print("[int8cache] WARN could not write sidecar (continuing):",
                  String(e))
    print("[flowedit] int8 base DONE (resident", i8_res_blocks, "+ host",
          NBLOCKS - i8_res_blocks, ").")
    ctx.synchronize()
    print("[flowedit] PHASE int8-base =",
          Float64(Int(perf_counter_ns()) - _t_ph) / 1e9, "s")
    print("[flowedit] STARTUP (launch -> denoise) =",
          Float64(Int(perf_counter_ns()) - _t_launch) / 1e9, "s")

    # ── 5) FlowEdit ODE: Euler on the velocity DIFFERENCE (4 forwards/step). ───
    var z_edit = z0_src.clone(ctx)                              # accumulator F32
    var z0_host = z0_src.to_host(ctx)                           # for latent blend
    var sal = List[Float32]()                                   # saliency sum
    for _ in range(NTOK):
        sal.append(Float32(0.0))
    var active_count = 0
    for si in range(steps):
        if si < skip_before or si >= stop_at:
            print("[flowedit] step", si, " t=", ts[si], " SKIPPED (window)")
            continue
        var t_cur = ts[si]
        var t_prev = ts[si + 1]
        var t_t = Tensor.from_host([t_cur], [1], STDtype.F32, ctx)

        ctx.synchronize(); var _f0 = Int(perf_counter_ns())
        # 1-2) fresh noise; Zt_src = (1-t)*Z0_src + t*N
        var noise = randn([1, 16, LH, LW], seed + UInt64(si), STDtype.F32, ctx)
        var zt_src = add(
            mul_scalar(z0_src, Float32(1.0) - t_cur, ctx),
            mul_scalar(noise, t_cur, ctx),
            ctx,
        )
        # 3) Zt_tgt = Z_edit + (Zt_src - Z0_src)
        var zt_tgt = add(z_edit, sub(zt_src, z0_src, ctx), ctx)
        # 4) V_src on the SOURCE prompt (CFG src_cfg)
        var v_src = _velocity(
            st, zt_src, ctx_src_pos, lt_src_pos, ctx_src_neg, lt_src_neg,
            pos_grid, t_t, src_cfg, resident_i8, host_i8, shared, ctx,
        )
        # 5) V_tgt on the TARGET prompt (CFG tgt_cfg)
        var v_tgt = _velocity(
            st, zt_tgt, ctx_tgt_pos, lt_tgt_pos, ctx_tgt_neg, lt_tgt_neg,
            pos_grid, t_t, tgt_cfg, resident_i8, host_i8, shared, ctx,
        )
        # 6) Z_edit += (t_prev - t) * (V_tgt - V_src)
        var dv = sub(v_tgt, v_src, ctx)
        z_edit = add(z_edit, mul_scalar(dv, t_prev - t_cur, ctx), ctx)
        # 7) AUTO-MASK: accumulate |V_tgt - V_src| saliency (free — dv is already
        #    computed); past warmup, hard-blend Z0_src back outside the mask.
        if auto_mask:
            active_count += 1
            var dv_host = dv.to_host(ctx)
            _accum_saliency(dv_host, sal)
            if active_count > mask_warmup:
                var mask = _mask_from_saliency(sal, mask_q, mask_dilate)
                var masked_n = 0
                for j in range(NTOK):
                    if mask[j]:
                        masked_n += 1
                var z_host = z_edit.to_host(ctx)
                _blend_outside_mask(z_host, z0_host, mask)
                z_edit = Tensor.from_host(
                    z_host^, [1, 16, LH, LW], STDtype.F32, ctx
                )
                print("[flowedit]   auto-mask blend: edit region =", masked_n,
                      "/", NTOK, "tokens")
        ctx.synchronize(); var _f1 = Int(perf_counter_ns())
        print("[flowedit] step", si, " t=", t_cur,
              " STEP=", Float64(_f1 - _f0) / 1e9, "s (4 forwards)")

    # ── 5b) AUTO-MASK debug artifact: the FINAL mask (same as the last blend's
    #        mask — saliency is complete at this point). ────────────────────────
    if auto_mask and active_count > 0:
        var final_mask = _mask_from_saliency(sal, mask_q, mask_dilate)
        var mask_path = _save_mask_png(final_mask, out_png, ctx)
        print("[flowedit] wrote mask debug artifact:", mask_path)

    # ── 6) VAE decode -> PNG. ──────────────────────────────────────────────────
    print("[flowedit] decoding Z_edit -> image ...")
    var dec = QwenImageVaeDecoder[LH, LW].load(String(KREA2_VAE_DIR), ctx)
    var latent_bf16 = torch_f32_to_bf16_rne(z_edit, ctx)
    var image = dec.decode(latent_bf16, ctx)                    # [1,3,H,W] [-1,1]
    save_png(image, out_png, ctx, ValueRange.SIGNED)
    print("[flowedit] wrote", out_png)
