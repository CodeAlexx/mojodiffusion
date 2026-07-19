# pipeline/lingbot_flowedit.mojo — FlowEdit VIDEO editing on LingBot DENSE-1.3B (task #24).
#
# Training-free video editing: port of the proven image FlowEdit
# (pipeline/krea2_flowedit.mojo — ICCV'25, arXiv 2412.08629) to the LingBot-Video
# dense T2V arm. Direct source→target ODE — NO inversion, NO training. PURE BASE
# dense model, fully resident.
#
# DYNAEDIT (task #29, arXiv 2603.17989 — the FlowEdit authors' follow-up):
# --dynaedit adds exactly TWO sampler-only mechanisms enabling ACTION/dynamics
# edits at n_max=N (pure-noise start; --dynaedit defaults n_max=steps unless
# --nmax is given explicitly):
#   ANC — Annealed Noise Correlation (Eq. 7): the per-step fresh noise is
#     replaced by a persistent Markovian stream per SGA sample j
#       w̃_j ← sqrt(a)·w̃_j + sqrt(1-a)·w_fresh,   w̃ init = 0
#       a(sigma) = clip((1-sigma)/0.75, 0, 1)   — a on the ACTUAL shifted sigma
#     (a=0 at sigma=1 → fresh noise; a=1 for sigma<0.25 → frozen stream; both
#     coefficients use the SAME a per Eq. 7 — Alg-2's sqrt(a_i) on term 1 is a
#     typo; variance stays 1).
#   SGA — Similarity Guided Aggregation (Eqs. 4-6, collapsed): the first
#     --sga-steps (3) active steps evaluate n_SGA=--sga-n (5) velocity diffs
#     Vd[j] = V_tgt_j - V_src_j from 5 independent w̃_j streams, weight them by
#     softmax(cossim(Z0_src, Z_edit - sigma·Vd[j]) / tau) (--tau, 0.01) and
#     Euler-step on the weighted mean; the WINNING stream (argmax) is carried
#     into the n_SGA=1 phase as w̃[0].
#   Auto-mask interaction (OUR addition, not in the paper): at n_max=N the
#     early V-diffs have no connection to the source, so with --dynaedit
#     saliency only accumulates once sigma < --mask-sigma-start (0.7); warmup
#     then counts ACCUMULATING steps; blend hard-copy unchanged.
# With --dynaedit OFF the loop is byte-identical to the pre-DynaEdit code
# (same randn calls, same seeds, same float op order).
#
# 3D spatiotemporal AUTO-MASK (--auto-mask, default OFF): per-voxel L2 of
# (V_tgt - V_src) over the 16 latent channels, accumulated across active steps
# into a [GT,LH,LW] saliency volume; quantile-thresholded (--mask-q) and 3D-
# dilated (--mask-dilate, 3x3x3 over (t,y,x) — temporal neighbors included so a
# moving subject's shifted tokens stay covered); past --mask-warmup active steps
# Z0_src is HARD-COPIED back into every outside-mask voxel after each Euler
# update (pixel-exact background up to VAE decode receptive-field bleed).
# Debug: <out>_mask_t{0..GT-1}.png (one per latent frame, LH x LW nearest x8).
#
# Algorithm (paper Alg. 1, n_avg=1), on the SHIFTED FlowUniPC sigma grid but with
# PLAIN EULER on the velocity difference (the UniPC multistep `step` is bypassed —
# exactly like krea2_flowedit uses krea2_timesteps + its own Euler update):
#   Z0_src = encode_video(clip)          (normalized Wan-VAE latent, F32)
#   Z_edit = Z0_src
#   sigmas = FlowUniPC set_timesteps(steps, SHIFT=3.0)  (t: 0.999 → 0, + final 0)
#   for step i (ONLY i in [steps-n_max, steps-n_min)):
#     t = sigmas[i], t_prev = sigmas[i+1]
#     N       ~ randn(seed+i)                     (fresh per step, deterministic)
#     Zt_src  = (1-t)*Z0_src + t*N
#     Zt_tgt  = Z_edit + (Zt_src - Z0_src)
#     V_src   = CFG(dense(Zt_src, c_src), dense(Zt_src, c_neg), src_cfg)
#     V_tgt   = CFG(dense(Zt_tgt, c_tgt), dense(Zt_tgt, c_neg), tgt_cfg)
#     Z_edit += (t_prev - t) * (V_tgt - V_src)    (t_prev - t < 0)
#   de-normalize (z*std+mean) → decode_video → pixels [0,1] safetensors.
#
# COMPTIME S: the dense backbone needs a comptime joint seq len S = N_VIDEO + L.
# There is NO pad mask (sdpa_nomask), so THREE instantiations are compiled:
#   S_SRC = N_VIDEO + L_SRC   (source prompt forwards)
#   S_TGT = N_VIDEO + L_TGT   (target prompt forwards)
#   S_NEG = N_VIDEO + L_NEG   (shared negative, 2 forwards/step)
# The L values were MEASURED offline with the tokenizer (HF tokenizers on the
# same tokenizer.json + LINGBOT template, validated against the known
# L_COND=607/L_UNCOND=234 of lingbot_t2v_mojo) and hardcoded below; main()
# fail-louds with "recompile with L_*=" if a prompt file changes. GATE A
# (tgt prompt == src prompt) is handled WITHOUT a recompile: when the runtime
# target length equals L_SRC, the target forwards runtime-dispatch to the S_SRC
# instantiation.
#
# Geometry: -D parameterized (see the comptime block). Default = the 13f slice
# (13f @ 576x320, GT=4, N_VIDEO=2880, tiled SDPA). FULL-clip build (task #27):
#   -D VFE_FRAMES=121 -D VFE_HEIGHT=480 -D VFE_WIDTH=832 -D VFE_L_SRC=591
#   -D VFE_L_TGT=618  → GT=31, LH=60, LW=104, N_VIDEO=48360, S≈48.6-49.0K —
#   flash SDPA territory, the cudnn cshim link is REQUIRED. Source clip = the
#   dense T2V robot-dance clip (parity/flowedit_121f_src_clip.safetensors,
#   staged from dense_t2v_mojo_pixels.safetensors with key "pixel").
#
# Run (GPU):
#   cd /home/alex/mojodiffusion && pixi run mojo run -I . \
#     -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa \
#     -Xlinker -lm \
#     serenitymojo/pipeline/lingbot_flowedit.mojo \
#     <src_clip.safetensors> <src_prompt.txt> <tgt_prompt.txt> <out_pixels.safetensors> \
#     [--steps 30] [--nmax 26] [--nmin 0] [--src-cfg 1.5] [--tgt-cfg 5.0] [--seed N] \
#     [--auto-mask] [--mask-q 0.7] [--mask-dilate 1] [--mask-warmup 4] [--tiled-encode] \
#     [--dynaedit] [--tau 0.01] [--sga-n 5] [--sga-steps 3] [--mask-sigma-start 0.7]
#
# Mojo 1.0.0b1, NVIDIA GPU. Inference-only.

from std.math import exp, sqrt
from std.pathlib import Path
from std.gpu.host import DeviceContext
from std.memory import ArcPointer
from std.sys import argv
from std.sys.defines import get_defined_int
from std.time import perf_counter_ns

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.image.png import save_png, ValueRange
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.random import randn
from serenitymojo.ops.torch_bf16 import torch_f32_to_bf16_rne, torch_bf16_rne_value
from serenitymojo.tokenizer.tokenizer import Qwen3Tokenizer
from serenitymojo.models.lingbotvideo.lingbot_tokenize import tokenize_lingbot_prompt
from serenitymojo.models.text_encoder.lingbot_qwen3vl import (
    load_lingbot_qwen3vl, encode_lingbot_text,
)
from serenitymojo.models.lingbotvideo.backbone import LingBotAttnConfig
from serenitymojo.models.lingbotvideo.dense import (
    LingBotDenseModel, lingbot_dense_backbone,
)
from serenitymojo.models.lingbotvideo.scheduler import FlowUniPCMultistepScheduler
from serenitymojo.models.lingbotvideo.vae_encoder import LingBotWanVaeEncoder
from serenitymojo.models.lingbotvideo.vae_decoder import LingBotWanVaeDecoder


# ── clip geometry: -D parameterized (KREA2_LTMAX-style), ONE file both sizes ─
#   default (no -D):  13 frames @ 576x320  → GT=4,  N_VIDEO=2880   (tiled SDPA)
#   FULL clip build:  -D VFE_FRAMES=121 -D VFE_HEIGHT=480 -D VFE_WIDTH=832
#                     → GT=31, LH=60, LW=104, N_VIDEO=48360, S≈49K (flash SDPA
#                     — link the cudnn cshim). GT/LH/LW/GH/GW/N_VIDEO derive.
comptime FRAMES = get_defined_int["VFE_FRAMES", 13]()
comptime HEIGHT = get_defined_int["VFE_HEIGHT", 576]()
comptime WIDTH = get_defined_int["VFE_WIDTH", 320]()
comptime GT = 1 + (FRAMES - 1) // 4  # latent frames (temporal 4x, causal)
comptime LH = HEIGHT // 8            # latent H
comptime LW = WIDTH // 8             # latent W
comptime GH = LH // 2                # patchified grid H
comptime GW = LW // 2                # patchified grid W
comptime N_VIDEO = GT * GH * GW      # video tokens
comptime OUT_CH = 16
comptime DEPTH = 24
comptime THETA = Float32(256.0)

# ── text lengths (MEASURED with the tokenizer; fail-loud check in main).
#    -D overridable per prompt pair. Defaults = the 13f tweed-dance gates:
#      L_SRC 607 = parity/t2v_prompt.txt
#      L_TGT 631 = parity/flowedit_style_tgt_prompt.txt (webtoon); red-jacket 610
#    121f robot-dance gates: -D VFE_L_SRC=591 (flowedit_121f_src_prompt.txt)
#      -D VFE_L_TGT=618 (flowedit_121f_tgt_prompt.txt, red-jacket robot). ─────
comptime L_SRC = get_defined_int["VFE_L_SRC", 607]()
comptime L_TGT = get_defined_int["VFE_L_TGT", 631]()
comptime L_NEG = get_defined_int["VFE_L_NEG", 234]()  # parity/i2v_neg_prompt.txt
comptime S_SRC = N_VIDEO + L_SRC     # 13f: 3487   121f: 48951
comptime S_TGT = N_VIDEO + L_TGT     # 13f: 3511   121f: 48978
comptime S_NEG = N_VIDEO + L_NEG     # 13f: 3114   121f: 48594

# ── FlowEdit defaults ─────────────────────────────────────────────────────────
comptime STEPS_DEFAULT = 30
comptime NMAX_DEFAULT = 26           # skip the first steps-nmax = 4 highest-t steps
comptime NMIN_DEFAULT = 0
comptime SRC_CFG_DEFAULT = Float32(1.5)
comptime TGT_CFG_DEFAULT = Float32(5.0)
comptime SEED_DEFAULT = UInt64(20260714)
comptime SHIFT = 3.0

# ── DynaEdit defaults (arXiv 2603.17989; sampler-only, all OFF by default) ───
comptime TAU_DEFAULT = Float32(0.01)
comptime SGA_N_DEFAULT = 5
comptime SGA_STEPS_DEFAULT = 3
comptime MASK_SIGMA_START_DEFAULT = Float32(0.7)
comptime DE_STREAM_STRIDE = UInt64(1000003)  # per-stream randn seed offset

# ── 3D auto-mask defaults (mirrors krea2_flowedit's 2D flags) ────────────────
comptime MASK_Q_DEFAULT = Float32(0.7)
comptime MASK_DILATE_DEFAULT = 1
comptime MASK_WARMUP_DEFAULT = 4
comptime LATPLANE = LH * LW          # voxels / latent frame (13f: 2880, 121f: 6240)
comptime NVOX = GT * LATPLANE        # [GT,LH,LW] mask grid (13f: 11520, 121f: 193440)
                                     # (channel stride in [1,16,GT,LH,LW] == NVOX)

comptime MODEL_DIR = "/mnt/disk1/models/lingbot-video-dense/transformer"
comptime VAE_FILE = "/mnt/disk1/models/lingbot-video-moe/vae/diffusion_pytorch_model.safetensors"
comptime TE_DIR = "/mnt/disk1/models/lingbot-video-moe/text_encoder"
comptime TOK_JSON = "/mnt/disk1/models/lingbot-video-moe/text_encoder/tokenizer.json"
comptime NEG_TXT = "/home/alex/mojodiffusion/serenitymojo/models/lingbotvideo/parity/i2v_neg_prompt.txt"


# ── 3D AUTO-MASK helpers (host-side; 11520 voxels — trivially cheap vs the 4
#    dense forwards/step). Latent [1,16,GT,LH,LW] host layout: flat index
#    c*NVOX + t*LATPLANE + y*LW + x, so voxel (t,y,x) owns the 16-channel
#    column {c*NVOX + vox : c in 0..16} at vox = t*LATPLANE + y*LW + x. ────────


def _accum_saliency_3d(
    v_tgt: List[Float32], v_src: List[Float32], mut sal: List[Float32]
):
    """sal[vox] += ||V_tgt - V_src||_2 over the 16-channel column of latent
    voxel vox — a per-voxel edit-saliency running sum across active steps."""
    for vox in range(NVOX):
        var acc = Float32(0.0)
        for c in range(OUT_CH):
            var d = v_tgt[c * NVOX + vox] - v_src[c * NVOX + vox]
            acc += d * d
        sal[vox] += sqrt(acc)


def _accum_saliency_diff_3d(vd: List[Float32], mut sal: List[Float32]):
    """DynaEdit variant of _accum_saliency_3d taking the already-aggregated
    velocity diff Vd_bar (same math: sal[vox] += ||Vd||_2 over 16 channels)."""
    for vox in range(NVOX):
        var acc = Float32(0.0)
        for c in range(OUT_CH):
            var d = vd[c * NVOX + vox]
            acc += d * d
        sal[vox] += sqrt(acc)


def _kth_smallest(sal: List[Float32], k: Int) -> Float32:
    """k-th order statistic (0-based) via iterative Hoare quickselect with a
    median-of-3 pivot on a scratch copy — O(n) average. Replaces the original
    insertion sort (fine at NVOX=11520, O(n^2)≈37e9 ops at the 121f NVOX=193440).
    Returns exactly sorted(sal)[k]."""
    var a = sal.copy()
    var lo = 0
    var hi = len(a) - 1
    while lo < hi:
        # median-of-3 pivot: order a[lo], a[mid], a[hi], pivot = a[mid].
        var mid = lo + (hi - lo) // 2
        if a[mid] < a[lo]:
            var t0 = a[mid]; a[mid] = a[lo]; a[lo] = t0
        if a[hi] < a[lo]:
            var t1 = a[hi]; a[hi] = a[lo]; a[lo] = t1
        if a[hi] < a[mid]:
            var t2 = a[hi]; a[hi] = a[mid]; a[mid] = t2
        var pivot = a[mid]
        # Hoare partition.
        var i = lo
        var j = hi
        while i <= j:
            while a[i] < pivot:
                i += 1
            while a[j] > pivot:
                j -= 1
            if i <= j:
                var t = a[i]; a[i] = a[j]; a[j] = t
                i += 1
                j -= 1
        if k <= j:
            hi = j
        elif k >= i:
            lo = i
        else:
            return a[k]
    return a[k]


def _mask_from_saliency_3d(
    sal: List[Float32], mask_q: Float32, dilate: Int
) -> List[Bool]:
    """Threshold accumulated saliency at quantile mask_q (voxels >= the q-th
    sorted value are IN the edit mask), then dilate `dilate` times with a FULL
    3x3x3 neighborhood on the (GT,LH,LW) grid — temporal neighbors too, so a
    subject that moves between latent frames keeps its shifted voxels masked."""
    # quantile threshold (quickselect; identical result to sorting then [qi]).
    var qi = Int(mask_q * Float32(NVOX))
    if qi < 0:
        qi = 0
    if qi > NVOX - 1:
        qi = NVOX - 1
    var thr = _kth_smallest(sal, qi)
    var mask = List[Bool]()
    for vox in range(NVOX):
        mask.append(sal[vox] >= thr)
    # 3D dilation: voxel is masked if ANY 3x3x3 (t,y,x) neighbor was masked.
    for _ in range(dilate):
        var grown = mask.copy()
        for t in range(GT):
            for y in range(LH):
                for x in range(LW):
                    if grown[t * LATPLANE + y * LW + x]:
                        continue
                    var hit = False
                    for dt in range(-1, 2):
                        var nt = t + dt
                        if nt < 0 or nt >= GT:
                            continue
                        for dy in range(-1, 2):
                            var ny = y + dy
                            if ny < 0 or ny >= LH:
                                continue
                            for dx in range(-1, 2):
                                var nx = x + dx
                                if nx < 0 or nx >= LW:
                                    continue
                                if mask[nt * LATPLANE + ny * LW + nx]:
                                    hit = True
                    if hit:
                        grown[t * LATPLANE + y * LW + x] = True
        mask = grown^
    return mask^


def _blend_outside_mask_3d(
    mut z_host: List[Float32], z0_host: List[Float32], mask: List[Bool]
):
    """HARD-COPY the source latent into every voxel OUTSIDE the mask (pixel-
    exact background in latent space; only VAE-decode receptive-field bleed
    remains — spatial AND temporal for the causal Wan video VAE)."""
    for vox in range(NVOX):
        if mask[vox]:
            continue
        for c in range(OUT_CH):
            z_host[c * NVOX + vox] = z0_host[c * NVOX + vox]


def _save_mask_pngs(
    mask: List[Bool], out_path: String, ctx: DeviceContext
) raises:
    """Save one debug PNG per latent frame: <out>_mask_t{t}.png, LHxLW upscaled
    nearest x8 to HEIGHTxWIDTH (576x320), white = edit region."""
    var base = out_path
    if base.endswith(".safetensors"):
        base = String(base.removesuffix(".safetensors"))
    for t in range(GT):
        var host = List[Float32]()
        host.resize(3 * HEIGHT * WIDTH, Float32(0.0))
        var k = 0
        for _ in range(3):           # R, G, B planes identical
            for y in range(HEIGHT):
                var ly = y // 8
                for x in range(WIDTH):
                    var lx = x // 8
                    if mask[t * LATPLANE + ly * LW + lx]:
                        host[k] = Float32(1.0)
                    k += 1
        var img = Tensor.from_host(host^, [1, 3, HEIGHT, WIDTH], STDtype.F32, ctx)
        var path = base + "_mask_t" + String(t) + ".png"
        save_png(img, path, ctx, ValueRange.UNIT)
        print("[vflowedit] wrote mask debug artifact:", path)


def _vae_mean() -> List[Float32]:
    var m: List[Float32] = [
        -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
        0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
    ]
    return m^


def _vae_std() -> List[Float32]:
    var s: List[Float32] = [
        2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
        3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160,
    ]
    return s^


def _forward_vel[
    S: Int
](
    lat_bf: Tensor,            # [1,16,GT,LH,LW] F32 (bf16-rounded values)
    tb: Tensor,                # [1] F32
    embeds: Tensor,            # [1,L,2560] F32 (bf16 values)
    text_len: Int,
    dm: LingBotDenseModel,
    cfg: LingBotAttnConfig,
    ctx: DeviceContext,
) raises -> List[Float32]:
    """One resident dense backbone forward → velocity host F32 [1,16,GT,LH,LW]."""
    var no_taps = List[Int]()
    var out = lingbot_dense_backbone[S](
        lat_bf, tb, embeds, dm,
        GT, GH, GW, text_len, 1, 2, 2, OUT_CH, THETA,
        no_taps, cfg, ctx,
    )
    return out.velocity[].to_host(ctx)


def _cfg_combine(
    cond: List[Float32], uncond: List[Float32], gs: Float32
) -> List[Float32]:
    """noise = uncond + gs*(cond - uncond) — the pipeline's F32 CFG combine."""
    var out = List[Float32]()
    out.resize(len(cond), Float32(0.0))
    for j in range(len(cond)):
        out[j] = uncond[j] + gs * (cond[j] - uncond[j])
    return out^


def _round_latent_bf16(latent_f32: Tensor, ctx: DeviceContext) raises -> Tensor:
    """Torch-RNE bf16 rounding kept as F32 — the pipeline's backbone-input contract."""
    var bf = torch_f32_to_bf16_rne(latent_f32, ctx)
    return cast_tensor(bf, STDtype.F32, ctx)


def _cossim_z0_pred(
    z0: List[Float32], z_edit: List[Float32], vd: List[Float32], sigma: Float32
) -> Float32:
    """SGA score (Eq. 5 collapsed): cossim(flatten(Z0_src),
    flatten(Z_edit - sigma*Vd)) — F64 accumulation over all 16*NVOX elements."""
    var dot = Float64(0.0)
    var n0 = Float64(0.0)
    var n1 = Float64(0.0)
    for j in range(len(z0)):
        var a = Float64(z0[j])
        var b = Float64(z_edit[j]) - Float64(sigma) * Float64(vd[j])
        dot += a * b
        n0 += a * a
        n1 += b * b
    return Float32(dot / (sqrt(n0) * sqrt(n1) + 1e-20))


def _sga_softmax(scores: List[Float32], tau: Float32) -> List[Float32]:
    """softmax(scores / tau), max-shifted for stability."""
    var m = scores[0]
    for j in range(len(scores)):
        if scores[j] > m:
            m = scores[j]
    var out = List[Float32]()
    var ssum = Float32(0.0)
    for j in range(len(scores)):
        var e = exp((scores[j] - m) / tau)
        out.append(e)
        ssum += e
    for j in range(len(out)):
        out[j] = out[j] / ssum
    return out^


def _velocity_diff_eval(
    w: List[Float32],          # ANC noise stream w̃_j (host F32, len 16*NVOX)
    z0: List[Float32],
    z_edit: List[Float32],
    t_cur: Float32,
    tb: Tensor,
    emb_src: Tensor,
    emb_tgt: Tensor,
    emb_neg: Tensor,
    l_tgt_rt: Int,
    tgt_uses_src_s: Bool,
    src_cfg: Float32,
    tgt_cfg: Float32,
    dm: LingBotDenseModel,
    cfg: LingBotAttnConfig,
    ctx: DeviceContext,
) raises -> List[Float32]:
    """One DynaEdit velocity-difference evaluation (4 dense forwards):
    Zt_src = (1-t)*Z0 + t*w̃;  Zt_tgt = Z_edit + (Zt_src - Z0);
    returns Vd = CFG_tgt(Zt_tgt) - CFG_src(Zt_src) as a host F32 list."""
    var n = len(z0)
    var lat_shape: List[Int] = [1, OUT_CH, GT, LH, LW]
    var zt_src = List[Float32]()
    zt_src.resize(n, Float32(0.0))
    var zt_tgt = List[Float32]()
    zt_tgt.resize(n, Float32(0.0))
    for j in range(n):
        var zs = (Float32(1.0) - t_cur) * z0[j] + t_cur * w[j]
        zt_src[j] = zs
        zt_tgt[j] = z_edit[j] + zs - z0[j]
    var zt_src_t = Tensor.from_host(zt_src, lat_shape.copy(), STDtype.F32, ctx)
    var zt_tgt_t = Tensor.from_host(zt_tgt, lat_shape.copy(), STDtype.F32, ctx)
    var src_bf = _round_latent_bf16(zt_src_t, ctx)
    var tgt_bf = _round_latent_bf16(zt_tgt_t, ctx)
    var v_src_c = _forward_vel[S_SRC](src_bf, tb, emb_src, L_SRC, dm, cfg, ctx)
    var v_src_u = _forward_vel[S_NEG](src_bf, tb, emb_neg, L_NEG, dm, cfg, ctx)
    var v_src = _cfg_combine(v_src_c, v_src_u, src_cfg)
    var v_tgt_c: List[Float32]
    if tgt_uses_src_s:
        v_tgt_c = _forward_vel[S_SRC](tgt_bf, tb, emb_tgt, l_tgt_rt, dm, cfg, ctx)
    else:
        v_tgt_c = _forward_vel[S_TGT](tgt_bf, tb, emb_tgt, l_tgt_rt, dm, cfg, ctx)
    var v_tgt_u = _forward_vel[S_NEG](tgt_bf, tb, emb_neg, L_NEG, dm, cfg, ctx)
    var v_tgt = _cfg_combine(v_tgt_c, v_tgt_u, tgt_cfg)
    var vd = List[Float32]()
    vd.resize(n, Float32(0.0))
    for j in range(n):
        vd[j] = v_tgt[j] - v_src[j]
    return vd^


def main() raises:
    var _t_launch = Int(perf_counter_ns())
    var ctx = DeviceContext()
    var cfg = LingBotAttnConfig.default()
    var args = argv()

    if len(args) < 5:
        raise Error(
            "usage: lingbot_flowedit <src_clip.safetensors> <src_prompt.txt>"
            " <tgt_prompt.txt> <out_pixels.safetensors> [--steps 30] [--nmax 26]"
            " [--nmin 0] [--src-cfg 1.5] [--tgt-cfg 5.0] [--seed N]"
            " [--auto-mask] [--mask-q 0.7] [--mask-dilate 1] [--mask-warmup 4]"
            " [--tiled-encode] [--dynaedit] [--tau 0.01] [--sga-n 5]"
            " [--sga-steps 3] [--mask-sigma-start 0.7]"
        )
    var clip_path = String(args[1])
    var src_prompt_path = String(args[2])
    var tgt_prompt_path = String(args[3])
    var out_path = String(args[4])

    var steps = STEPS_DEFAULT
    var n_max = NMAX_DEFAULT
    var n_min = NMIN_DEFAULT
    var src_cfg = SRC_CFG_DEFAULT
    var tgt_cfg = TGT_CFG_DEFAULT
    var seed = SEED_DEFAULT
    var auto_mask = False
    var mask_q = MASK_Q_DEFAULT
    var mask_dilate = MASK_DILATE_DEFAULT
    var mask_warmup = MASK_WARMUP_DEFAULT
    var tiled_encode = False
    var dynaedit = False
    var tau = TAU_DEFAULT
    var sga_n = SGA_N_DEFAULT
    var sga_steps = SGA_STEPS_DEFAULT
    var mask_sigma_start = MASK_SIGMA_START_DEFAULT
    var nmax_explicit = False
    for i in range(len(args)):
        if args[i] == String("--tiled-encode"):
            tiled_encode = True
        if args[i] == String("--steps") and i + 1 < len(args):
            steps = Int(String(args[i + 1]))
        if args[i] == String("--nmax") and i + 1 < len(args):
            n_max = Int(String(args[i + 1]))
            nmax_explicit = True
        if args[i] == String("--dynaedit"):
            dynaedit = True
        if args[i] == String("--tau") and i + 1 < len(args):
            tau = Float32(Float64(String(args[i + 1])))
        if args[i] == String("--sga-n") and i + 1 < len(args):
            sga_n = Int(String(args[i + 1]))
        if args[i] == String("--sga-steps") and i + 1 < len(args):
            sga_steps = Int(String(args[i + 1]))
        if args[i] == String("--mask-sigma-start") and i + 1 < len(args):
            mask_sigma_start = Float32(Float64(String(args[i + 1])))
        if args[i] == String("--nmin") and i + 1 < len(args):
            n_min = Int(String(args[i + 1]))
        if args[i] == String("--src-cfg") and i + 1 < len(args):
            src_cfg = Float32(Float64(String(args[i + 1])))
        if args[i] == String("--tgt-cfg") and i + 1 < len(args):
            tgt_cfg = Float32(Float64(String(args[i + 1])))
        if args[i] == String("--seed") and i + 1 < len(args):
            seed = UInt64(Int(String(args[i + 1])))
        if args[i] == String("--auto-mask"):
            auto_mask = True
        if args[i] == String("--mask-q") and i + 1 < len(args):
            mask_q = Float32(Float64(String(args[i + 1])))
        if args[i] == String("--mask-dilate") and i + 1 < len(args):
            mask_dilate = Int(String(args[i + 1]))
        if args[i] == String("--mask-warmup") and i + 1 < len(args):
            mask_warmup = Int(String(args[i + 1]))
    if dynaedit and not nmax_explicit:
        n_max = steps                    # DynaEdit edits from PURE NOISE (n_max=N)
    if sga_n < 1:
        sga_n = 1
    if sga_steps < 0:
        sga_steps = 0
    if n_max > steps:
        n_max = steps
    if n_min < 0:
        n_min = 0

    print("[vflowedit] FlowEdit VIDEO", FRAMES, "f @", HEIGHT, "x", WIDTH,
          " GT=", GT, " N_VIDEO=", N_VIDEO, " steps=", steps,
          " n_max=", n_max, " n_min=", n_min,
          " src_cfg=", src_cfg, " tgt_cfg=", tgt_cfg, " seed=", seed)
    if dynaedit:
        print("[vflowedit] DYNAEDIT on (ANC + SGA): sga_n=", sga_n,
              " sga_steps=", sga_steps, " tau=", tau,
              " n_max=", n_max, ("(--nmax explicit)" if nmax_explicit
                                 else "(forced to steps)"))
    if auto_mask:
        print("[vflowedit] AUTO-MASK 3D on: q=", mask_q, " dilate=", mask_dilate,
              " (3x3x3) warmup=", mask_warmup, " active steps  grid=[",
              GT, ",", LH, ",", LW, "]")
        if dynaedit:
            print("[vflowedit] AUTO-MASK saliency gated: accumulate only when"
                  " sigma <", mask_sigma_start, "(--mask-sigma-start)")

    # ── 1) tokenize + Qwen3-VL text encode (3 contexts), then FREE the TE. ─────
    var _t_ph = Int(perf_counter_ns())
    var tok = Qwen3Tokenizer(String(TOK_JSON))
    var sp = tokenize_lingbot_prompt(tok, Path(src_prompt_path).read_text())
    var tp = tokenize_lingbot_prompt(tok, Path(tgt_prompt_path).read_text())
    var np_ = tokenize_lingbot_prompt(tok, Path(String(NEG_TXT)).read_text())
    var s_ids = sp[0].copy()
    var s_crop = sp[1]
    var t_ids = tp[0].copy()
    var t_crop = tp[1]
    var n_ids = np_[0].copy()
    var n_crop = np_[1]
    var l_src_rt = len(s_ids) - s_crop
    var l_tgt_rt = len(t_ids) - t_crop
    var l_neg_rt = len(n_ids) - n_crop
    print("[vflowedit] runtime L: src=", l_src_rt, " tgt=", l_tgt_rt,
          " neg=", l_neg_rt, " (comptime ", L_SRC, "/", L_TGT, "/", L_NEG, ")")
    if l_src_rt != L_SRC or l_neg_rt != L_NEG:
        raise Error(String("recompile with L_SRC=") + String(l_src_rt)
                    + " L_NEG=" + String(l_neg_rt))
    # GATE A (tgt prompt == src prompt) dispatches to the S_SRC instantiation.
    var tgt_uses_src_s = (l_tgt_rt == L_SRC)
    if (not tgt_uses_src_s) and l_tgt_rt != L_TGT:
        raise Error(String("recompile with L_TGT=") + String(l_tgt_rt))

    var te = load_lingbot_qwen3vl(String(TE_DIR), ctx)
    var emb_src = encode_lingbot_text(te, s_ids, s_crop, ctx)   # [1,L_SRC,2560]
    var emb_tgt = encode_lingbot_text(te, t_ids, t_crop, ctx)   # [1,l_tgt,2560]
    var emb_neg = encode_lingbot_text(te, n_ids, n_crop, ctx)   # [1,L_NEG,2560]
    _ = te^
    ctx.synchronize()
    print("[vflowedit] PHASE text-encode =",
          Float64(Int(perf_counter_ns()) - _t_ph) / 1e9, "s")
    _t_ph = Int(perf_counter_ns())

    # ── 2) staged clip [1,3,F,H,W] [0,1] → Z0_src (normalized latent, F32). ───
    var clip_st = ShardedSafeTensors.open(clip_path)
    var clip = Tensor.from_view_as_f32(clip_st.tensor_view(String("pixel")), ctx)
    var csh = clip.shape()
    if len(csh) != 5 or csh[2] != FRAMES or csh[3] != HEIGHT or csh[4] != WIDTH:
        raise Error(String("staged clip must be [1,3,") + String(FRAMES) + ","
                    + String(HEIGHT) + "," + String(WIDTH) + "] [0,1] f32")
    var enc = LingBotWanVaeEncoder[HEIGHT, WIDTH].load(String(VAE_FILE), ctx)
    var z0_t: Tensor
    if tiled_encode:
        print("[vflowedit] VAE encode: TILED (256px tiles, stride 192)")
        z0_t = enc.encode_video_tiled(clip, ctx)    # [1,16,GT,LH,LW] NORMALIZED
    else:
        z0_t = enc.encode_video(clip, ctx)          # [1,16,GT,LH,LW] NORMALIZED
    _ = enc^
    var z0 = z0_t.to_host(ctx)                      # host F32 master copy
    var n = len(z0)
    ctx.synchronize()
    print("[vflowedit] PHASE vae-encode =",
          Float64(Int(perf_counter_ns()) - _t_ph) / 1e9, "s  (Z0_src n=", n, ")")
    _t_ph = Int(perf_counter_ns())

    # ── 3) resident dense model (2.6GB bf16, loaded ONCE). ────────────────────
    var model = ShardedSafeTensors.open(String(MODEL_DIR))
    var dm = LingBotDenseModel.load(model, DEPTH, ctx)
    ctx.synchronize()
    print("[vflowedit] PHASE dense-load =",
          Float64(Int(perf_counter_ns()) - _t_ph) / 1e9, "s")
    print("[vflowedit] STARTUP (launch -> denoise) =",
          Float64(Int(perf_counter_ns()) - _t_launch) / 1e9, "s")

    # ── 4) shifted sigma grid (FlowUniPC set_timesteps; PLAIN EULER below). ───
    var sch = FlowUniPCMultistepScheduler(num_train_timesteps=1000)
    sch.set_timesteps(steps, SHIFT)
    var skip_before = steps - n_max     # steps [0:skip_before) SKIPPED
    var stop_at = steps - n_min         # steps [stop_at:steps) SKIPPED
    print("[vflowedit] sigma grid: sigmas[0]=", sch.sigmas[0],
          " sigmas[-2]=", sch.sigmas[steps - 1],
          " active window = [", skip_before, ",", stop_at, ")")

    # ── 5) FlowEdit ODE (4 dense forwards / active step). ─────────────────────
    var lat_shape: List[Int] = [1, OUT_CH, GT, LH, LW]
    var z_edit = z0.copy()              # host F32 accumulator
    var sal = List[Float32]()           # 3D saliency running sum [GT,LH,LW]
    sal.resize(NVOX, Float32(0.0))
    var active_count = 0
    # DynaEdit state: persistent ANC noise streams w̃_j (init 0) + active index.
    var w_streams = List[List[Float32]]()
    if dynaedit:
        for _ in range(sga_n):
            var w = List[Float32]()
            w.resize(n, Float32(0.0))
            w_streams.append(w^)
    var de_active_idx = 0
    var _t_loop = Int(perf_counter_ns())
    for si in range(steps):
        if si < skip_before or si >= stop_at:
            print("[vflowedit] step", si, " t=", sch.sigmas[si], " SKIPPED (window)")
            continue
        var _f0 = Int(perf_counter_ns())
        var t_cur = Float32(sch.sigmas[si])
        var t_prev = Float32(sch.sigmas[si + 1])

        # timestep_batch = (bf16(t_int/1000))*1000 (the pipeline's contract).
        var t_int = sch.timesteps[si]
        var sig_bf = torch_bf16_rne_value(Float32(t_int) / Float32(1000.0))
        var tb_host: List[Float32] = [Float32(sig_bf) * Float32(1000.0)]
        var tb = Tensor.from_host(tb_host, [1], STDtype.F32, ctx)

        # ── DynaEdit step (ANC + SGA); the non-dynaedit path below is UNTOUCHED.
        if dynaedit:
            # ANC (Eq. 7) on the ACTUAL shifted sigma: a = clip((1-t)/0.75,0,1).
            var a_anc = (Float32(1.0) - t_cur) / Float32(0.75)
            if a_anc < Float32(0.0):
                a_anc = Float32(0.0)
            if a_anc > Float32(1.0):
                a_anc = Float32(1.0)
            var sa = sqrt(a_anc)
            var sb = sqrt(Float32(1.0) - a_anc)
            var n_j = sga_n if de_active_idx < sga_steps else 1
            # advance streams + evaluate Vd[j] (4 dense forwards each).
            var vds = List[List[Float32]]()
            for j in range(n_j):
                var fresh = randn(
                    lat_shape.copy(),
                    seed + UInt64(si) + UInt64(j) * DE_STREAM_STRIDE,
                    STDtype.F32, ctx,
                )
                var fh = fresh.to_host(ctx)
                for k in range(n):
                    w_streams[j][k] = sa * w_streams[j][k] + sb * fh[k]
                vds.append(
                    _velocity_diff_eval(
                        w_streams[j], z0, z_edit, t_cur, tb,
                        emb_src, emb_tgt, emb_neg, l_tgt_rt, tgt_uses_src_s,
                        src_cfg, tgt_cfg, dm, cfg, ctx,
                    )
                )
            # SGA (Eqs. 4-6): softmax(cossim/tau)-weighted mean of the Vd[j].
            var vd_bar: List[Float32]
            if n_j > 1:
                var scores = List[Float32]()
                for j in range(n_j):
                    scores.append(_cossim_z0_pred(z0, z_edit, vds[j], t_cur))
                var wts = _sga_softmax(scores, tau)
                var jmax = 0
                for j in range(n_j):
                    if scores[j] > scores[jmax]:
                        jmax = j
                vd_bar = List[Float32]()
                vd_bar.resize(n, Float32(0.0))
                for j in range(n_j):
                    var wj = wts[j]
                    for k in range(n):
                        vd_bar[k] += wj * vds[j][k]
                print("[vflowedit]   SGA n=", n_j, " argmax=", jmax, " scores:",
                      end="")
                for j in range(n_j):
                    print(" ", scores[j], "(w=", wts[j], ")", end="")
                print()
                # carry the WINNING stream into the n_SGA=1 phase.
                if de_active_idx == sga_steps - 1:
                    w_streams[0] = w_streams[jmax].copy()
                    print("[vflowedit]   SGA handoff: w̃[0] <- stream", jmax)
            else:
                vd_bar = vds[0].copy()
            # Euler on the aggregated diff (unchanged update rule).
            var de_dt = t_prev - t_cur
            var de_dv_l1 = Float64(0.0)
            for k in range(n):
                var dv = vd_bar[k]
                z_edit[k] += de_dt * dv
                de_dv_l1 += Float64(dv if dv >= 0 else -dv)
            # auto-mask (OUR addition): saliency only once sigma < start;
            # warmup counts ACCUMULATING steps; blend hard-copy as today.
            if auto_mask and t_cur < mask_sigma_start:
                active_count += 1
                _accum_saliency_diff_3d(vd_bar, sal)
                if active_count > mask_warmup:
                    var mask = _mask_from_saliency_3d(sal, mask_q, mask_dilate)
                    var masked_n = 0
                    for vox in range(NVOX):
                        if mask[vox]:
                            masked_n += 1
                    _blend_outside_mask_3d(z_edit, z0, mask)
                    print("[vflowedit]   auto-mask blend: edit region =",
                          masked_n, "/", NVOX, "voxels")
            de_active_idx += 1
            var _de1 = Int(perf_counter_ns())
            print("[vflowedit] step", si, " t=", t_cur, " dt=", de_dt,
                  " a_anc=", a_anc, " |dV|_mean=", de_dv_l1 / Float64(n),
                  " STEP=", Float64(_de1 - _f0) / 1e9, "s (", 4 * n_j,
                  "forwards, DYNAEDIT)")
            continue

        # 1-2) fresh per-step noise; Zt_src = (1-t)*Z0 + t*N  (host F32).
        var noise = randn(lat_shape.copy(), seed + UInt64(si), STDtype.F32, ctx)
        var nh = noise.to_host(ctx)
        var _p0 = Int(perf_counter_ns())
        var zt_src = List[Float32]()
        zt_src.resize(n, Float32(0.0))
        var zt_tgt = List[Float32]()
        zt_tgt.resize(n, Float32(0.0))
        for j in range(n):
            var zs = (Float32(1.0) - t_cur) * z0[j] + t_cur * nh[j]
            zt_src[j] = zs
            # 3) Zt_tgt = Z_edit + (Zt_src - Z0)
            zt_tgt[j] = z_edit[j] + zs - z0[j]

        var zt_src_t = Tensor.from_host(zt_src, lat_shape.copy(), STDtype.F32, ctx)
        var zt_tgt_t = Tensor.from_host(zt_tgt, lat_shape.copy(), STDtype.F32, ctx)
        var src_bf = _round_latent_bf16(zt_src_t, ctx)
        var tgt_bf = _round_latent_bf16(zt_tgt_t, ctx)
        var _p1 = Int(perf_counter_ns())

        # 4) V_src = CFG(src prompt vs neg) at Zt_src.
        var v_src_c = _forward_vel[S_SRC](src_bf, tb, emb_src, L_SRC, dm, cfg, ctx)
        var _p2 = Int(perf_counter_ns())
        var v_src_u = _forward_vel[S_NEG](src_bf, tb, emb_neg, L_NEG, dm, cfg, ctx)
        var _p3 = Int(perf_counter_ns())
        var v_src = _cfg_combine(v_src_c, v_src_u, src_cfg)
        var _p4 = Int(perf_counter_ns())
        print("[vflowedit]   prep=", Float64(_p1 - _p0) / 1e9,
              " noise=", Float64(_p0 - _f0) / 1e9,
              " fwd_src_c=", Float64(_p2 - _p1) / 1e9,
              " fwd_src_u=", Float64(_p3 - _p2) / 1e9,
              " cfg=", Float64(_p4 - _p3) / 1e9)

        # 5) V_tgt = CFG(tgt prompt vs neg) at Zt_tgt. GATE A (tgt==src prompt)
        #    runtime-dispatches to the S_SRC instantiation.
        var v_tgt_c: List[Float32]
        if tgt_uses_src_s:
            v_tgt_c = _forward_vel[S_SRC](tgt_bf, tb, emb_tgt, l_tgt_rt, dm, cfg, ctx)
        else:
            v_tgt_c = _forward_vel[S_TGT](tgt_bf, tb, emb_tgt, l_tgt_rt, dm, cfg, ctx)
        var v_tgt_u = _forward_vel[S_NEG](tgt_bf, tb, emb_neg, L_NEG, dm, cfg, ctx)
        var v_tgt = _cfg_combine(v_tgt_c, v_tgt_u, tgt_cfg)

        # 6) Z_edit += (t_prev - t) * (V_tgt - V_src)   (plain Euler on the diff).
        var dt = t_prev - t_cur
        var dv_l1 = Float64(0.0)
        for j in range(n):
            var dv = v_tgt[j] - v_src[j]
            z_edit[j] += dt * dv
            dv_l1 += Float64(dv if dv >= 0 else -dv)
        # 7) AUTO-MASK: accumulate ||V_tgt - V_src|| voxel saliency (free — the
        #    diff is already on host); past warmup, hard-blend Z0_src back into
        #    every outside-mask voxel of the freshly updated Z_edit.
        if auto_mask:
            active_count += 1
            _accum_saliency_3d(v_tgt, v_src, sal)
            if active_count > mask_warmup:
                var _m0 = Int(perf_counter_ns())
                var mask = _mask_from_saliency_3d(sal, mask_q, mask_dilate)
                var _m1 = Int(perf_counter_ns())
                var masked_n = 0
                for vox in range(NVOX):
                    if mask[vox]:
                        masked_n += 1
                _blend_outside_mask_3d(z_edit, z0, mask)
                print("[vflowedit]   auto-mask blend: edit region =", masked_n,
                      "/", NVOX, "voxels  mask-build=",
                      Float64(_m1 - _m0) / 1e9, "s (quickselect+dilate)")
        var _f1 = Int(perf_counter_ns())
        print("[vflowedit] step", si, " t=", t_cur, " dt=", dt,
              " |dV|_mean=", dv_l1 / Float64(n),
              " STEP=", Float64(_f1 - _f0) / 1e9, "s (4 forwards)")
    print("[vflowedit] PHASE flowedit-loop =",
          Float64(Int(perf_counter_ns()) - _t_loop) / 1e9, "s")
    _ = dm^

    # ── 5b) AUTO-MASK debug artifacts: the FINAL per-latent-frame masks (same
    #        as the last blend's mask — saliency is complete at this point). ──
    if auto_mask and active_count > 0:
        var final_mask = _mask_from_saliency_3d(sal, mask_q, mask_dilate)
        for t in range(GT):
            var cnt = 0
            for p in range(LATPLANE):
                if final_mask[t * LATPLANE + p]:
                    cnt += 1
            print("[vflowedit] final mask latent frame", t, ": ", cnt, "/",
                  LATPLANE, "voxels")
        _save_mask_pngs(final_mask, out_path, ctx)

    # ── 6) de-normalize (z*std+mean) → temporal VAE decode → pixels [0,1]. ────
    _t_ph = Int(perf_counter_ns())
    var chstride = GT * LH * LW
    var vmean = _vae_mean()
    var vstd = _vae_std()
    var zhost = List[Float32]()
    zhost.resize(n, Float32(0.0))
    for j in range(n):
        var ch = (j // chstride) % OUT_CH
        zhost[j] = z_edit[j] * vstd[ch] + vmean[ch]
    var z = Tensor.from_host(zhost, lat_shape.copy(), STDtype.F32, ctx)

    var dec = LingBotWanVaeDecoder[LH, LW].load(String(VAE_FILE), ctx)
    var pixels_raw = dec.decode_video(z, ctx)       # [1,3,F,H,W] in [-1,1]
    var ps = pixels_raw.shape()
    var raw = pixels_raw.to_host(ctx)
    var px01 = List[Float32]()
    px01.resize(len(raw), Float32(0.0))
    for i in range(len(raw)):
        px01[i] = (raw[i] + Float32(1.0)) * Float32(0.5)

    # per-frame MAD vs the staged source clip (free correctness telemetry).
    var src_px = clip.to_host(ctx)
    var fplane = ps[3] * ps[4]
    print("[vflowedit] per-frame MAD (edited vs source):")
    for f in range(ps[2]):
        var acc = Float64(0.0)
        for c in range(3):
            var off = (c * ps[2] + f) * fplane
            for k in range(fplane):
                var d = px01[off + k] - src_px[off + k]
                acc += Float64(d if d >= 0 else -d)
        print("   frame", f, " MAD =", acc / Float64(3 * fplane))

    var pix = Tensor.from_host(px01, ps.copy(), STDtype.F32, ctx)
    var names: List[String] = [String("pixels")]
    var tens = List[ArcPointer[Tensor]]()
    tens.append(ArcPointer(pix^))
    save_safetensors(names, tens, out_path, ctx)
    ctx.synchronize()
    print("[vflowedit] PHASE decode+save =",
          Float64(Int(perf_counter_ns()) - _t_ph) / 1e9, "s")
    print("[vflowedit] E2E =",
          Float64(Int(perf_counter_ns()) - _t_launch) / 1e9, "s")
    print("[vflowedit] SAVED", out_path)
