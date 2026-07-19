# serenitymojo/models/klein/parity/klein_batch2_parity.mojo
#
# TRUE BATCH-2 PARITY GATE for the Klein FULL DiT stack + LoRA.
#
# Proves the batch-2 driver pair (klein_stack_lora_forward_b2 /
# klein_stack_lora_backward_b2) implements SerenityTrainer batch semantics:
#     loss_B2  == mean(loss_B1(s0), loss_B1(s1))            (within 1e-3 rel)
#     grad_B2  == mean(grad_B1(s0), grad_B1(s1))            (cosine >= 0.999,
#                                                            rel-L2 <= 1e-3)
# per adapter (144 = 8*12 double + 24*2 single).
#
# The two paired cache samples get INDEPENDENT sigmas (=> independent img/txt/
# single modvecs + final adaLN scale/shift), exactly as a real batch-2 step.
# The joint 2N-mean MSE loss means each sample's output grad is HALF the single-
# sample d_loss, so the summed per-sample backward == the batch-mean gradient.
# The reference is TWO single-sample runs of the resident (no-loader) B1 driver
# with compute_input_grads=compute_aux_grads=False (the batch-2 driver never
# computes input/aux grads either — apples-to-apples LoRA-grad comparison).
#
# The trainer's fp8-resident/streaming batch-2 path
# (klein_stack_lora_{forward,backward}_offload_turbo_moddev_rope_scratch_b2)
# runs the SAME per-sample block calls + the SAME grad-summation orchestration;
# it differs only in weight source (loader vs dbw/sbw) and the scratch ring —
# both already B1-parity-proven — so this gate's evidence carries to it.
#
# Two real cache samples from the eri2_klein9b_512 cache supply the img/txt
# tokens; a fixed rope table is shared by B1 and B2 (the identity is rope-
# agnostic — B1 and B2 receive byte-identical inputs).
#
# BUILD (no GPU; the lead runs it):
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#   pixi run mojo build --optimization-level 2 -I . -I /home/alex/MOJO-libs \
#     -Xlinker -lm -Xlinker -lcuda \
#     -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa \
#     -Xlinker -rpath -Xlinker /home/alex/mojodiffusion/serenitymojo/ops/cshim/lib \
#     serenitymojo/models/klein/parity/klein_batch2_parity.mojo \
#     -o /tmp/klein_batch2_parity

from std.collections import List, Optional
from std.gpu.host import DeviceContext
from std.math import sqrt

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.ops.tensor_algebra import reshape

from serenitymojo.models.klein.double_block import DoubleBlockWeights
from serenitymojo.models.klein.single_block import SingleBlockWeights
from serenitymojo.models.klein.klein_stack import KleinStackBase
from serenitymojo.models.klein.klein_stack_lora import (
    KleinLoraSet, build_klein_lora_set, klein_lora_set_to_device,
    KleinLoraDeviceSet,
    klein_stack_lora_forward_device_inputs_resident_moddev_rope,
    klein_stack_lora_backward_resident_moddev_rope,
    klein_stack_lora_forward_b2, klein_stack_lora_backward_b2,
    KleinStackForwardB2,
    DBL_SLOTS, SGL_SLOTS, TArc,
)
from serenitymojo.models.klein.weights import (
    load_double_block_weights, load_single_block_weights,
    load_klein_stack_base, build_klein_vec_silu,
    load_klein_step_mod_weights, build_klein_step_mods_device_cached,
)
from serenitymojo.training.klein_dataset import KleinCache


comptime KLEIN9B_PATH = "/home/alex/.serenity/models/checkpoints/flux-2-klein-base-9b.safetensors"
comptime CACHE_DIR = "/home/alex/flame-diffusion-archive/klein-trainer/cache/eri2_klein9b_512"

comptime H = 32
comptime Dh = 128
comptime N_IMG = 1024
comptime N_TXT = 512
comptime S = N_IMG + N_TXT
comptime NUM_DOUBLE = 4   # gate depth: full 8+24 F32-resident OOMs 24GB; parity is depth-independent
comptime NUM_SINGLE = 8
comptime TIMESTEP_DIM = 256
comptime RANK = 8
comptime ALPHA = Float32(8.0)

comptime COS_THRESH = Float64(0.999)
comptime REL_L2_THRESH = Float64(1.0e-3)
comptime LOSS_REL_THRESH = Float64(1.0e-3)


def _fill(n: Int, seed: UInt64, scale: Float32) -> List[Float32]:
    var out = List[Float32]()
    var state = seed
    for _ in range(n):
        state = state * 6364136223846793005 + 1442695040888963407
        var u = Float32(Int(state >> 40)) * Float32(1.0 / 16777216.0)
        out.append((u - Float32(0.5)) * scale)
    return out^


# klein LoraAdapter stores A/B in BF16; grads stay F32. This perturbs B off
# zero so the A-gradient path is exercised.
def _fill_bf16(n: Int, seed: UInt64, scale: Float32) -> List[BFloat16]:
    var out = List[BFloat16]()
    var state = seed
    for _ in range(n):
        state = state * 6364136223846793005 + 1442695040888963407
        var u = Float32(Int(state >> 40)) * Float32(1.0 / 16777216.0)
        out.append(((u - Float32(0.5)) * scale).cast[DType.bfloat16]())
    return out^


# MSE loss/grad matching the trainer's default path (_klein_loss_grad):
#   loss = mean_i (pred-target)^2 ;  d_loss_i = (2/nout)(pred-target).
def _mse_loss(pred: List[Float32], target: List[Float32]) raises -> Float64:
    if len(pred) != len(target):
        raise Error("mse loss: length mismatch")
    var sse = Float64(0.0)
    for i in range(len(pred)):
        var d = Float64(pred[i] - target[i])
        sse += d * d
    return sse / Float64(len(pred))


def _mse_dloss(pred: List[Float32], target: List[Float32]) raises -> List[Float32]:
    var inv_n = Float32(2.0) / Float32(len(pred))
    var out = List[Float32]()
    for i in range(len(pred)):
        out.append(inv_n * (pred[i] - target[i]))
    return out^


def _scaled(a: List[Float32], s: Float32) -> List[Float32]:
    var out = List[Float32]()
    for i in range(len(a)):
        out.append(a[i] * s)
    return out^


# mean of two per-sample grad vectors: 0.5*(g0 + g1).
def _mean2(g0: List[Float32], g1: List[Float32]) raises -> List[Float32]:
    if len(g0) != len(g1):
        raise Error("mean2: length mismatch")
    var out = List[Float32]()
    for i in range(len(g0)):
        out.append(Float32(0.5) * (g0[i] + g1[i]))
    return out^


def _cosine(a: List[Float32], refv: List[Float32]) raises -> Float64:
    if len(a) != len(refv):
        raise Error("cosine: length mismatch")
    var dot = Float64(0.0); var na = Float64(0.0); var nr = Float64(0.0)
    for i in range(len(a)):
        var av = Float64(a[i]); var rv = Float64(refv[i])
        dot += av * rv; na += av * av; nr += rv * rv
    # Both near-zero (should not happen with nonzero B) -> treat as a match.
    if na == 0.0 and nr == 0.0:
        return 1.0
    if na == 0.0 or nr == 0.0:
        return 0.0
    return dot / (sqrt(na) * sqrt(nr))


def _rel_l2(a: List[Float32], refv: List[Float32]) raises -> Float64:
    if len(a) != len(refv):
        raise Error("rel_l2: length mismatch")
    var num = Float64(0.0); var den = Float64(0.0)
    for i in range(len(a)):
        var d = Float64(a[i] - refv[i])
        num += d * d
        den += Float64(refv[i]) * Float64(refv[i])
    if den == 0.0:
        return 0.0 if num == 0.0 else 1.0e30
    return sqrt(num) / sqrt(den)


def main() raises:
    var ctx = DeviceContext()
    var D = 4096
    var F = 12288
    var IN_CH = 128
    var TXT_CH = 12288
    var OUT_CH = 128
    var eps = Float32(1.0e-6)
    var sigma0 = Float32(0.3)
    var sigma1 = Float32(0.7)

    print("=== Klein-9B TRUE batch-2 PARITY gate (loss + grad == mean of B1) ===")
    print("  ckpt:", KLEIN9B_PATH)
    print("  cache:", CACHE_DIR)
    print("  D=", D, " H=", H, " Dh=", Dh, " F=", F,
          " N_IMG=", N_IMG, " N_TXT=", N_TXT, " S=", S,
          " double=", NUM_DOUBLE, " single=", NUM_SINGLE, " rank=", RANK)

    var st = SafeTensors.open(KLEIN9B_PATH)

    # ── frozen base weights + per-sample modulation ──────────────────────────
    var ts = Tensor.from_host([Float32(0.5)], [1], STDtype.F32, ctx)
    var vec_silu = build_klein_vec_silu(st, ts, TIMESTEP_DIM, D, ctx)
    var base = load_klein_stack_base(st, vec_silu, D, ctx)  # img_in/txt_in/final_lin

    var mod_weights = load_klein_step_mod_weights(st, D, ctx)
    var mods0 = build_klein_step_mods_device_cached(mod_weights, sigma0, TIMESTEP_DIM, D, ctx)
    var img_mod0 = mods0[0].copy(); var txt_mod0 = mods0[1].copy(); var single_mod0 = mods0[2].copy()
    var final_shift0 = mods0[3].copy(); var final_scale0 = mods0[4].copy()
    var mods1 = build_klein_step_mods_device_cached(mod_weights, sigma1, TIMESTEP_DIM, D, ctx)
    var img_mod1 = mods1[0].copy(); var txt_mod1 = mods1[1].copy(); var single_mod1 = mods1[2].copy()
    var final_shift1 = mods1[3].copy(); var final_scale1 = mods1[4].copy()

    # per-sample base (B1 reference reads base.final_scale/shift): same img_in/
    # txt_in/final_lin, per-sample final adaLN mod.
    var base0 = KleinStackBase(
        base.img_in.copy(), base.txt_in.copy(), base.final_lin.copy(),
        final_shift0.copy(), final_scale0.copy(),
    )
    var base1 = KleinStackBase(
        base.img_in.copy(), base.txt_in.copy(), base.final_lin.copy(),
        final_shift1.copy(), final_scale1.copy(),
    )

    var dbw = List[DoubleBlockWeights]()
    for bi in range(NUM_DOUBLE):
        dbw.append(load_double_block_weights(st, bi, ctx))
    var sbw = List[SingleBlockWeights]()
    for bi in range(NUM_SINGLE):
        sbw.append(load_single_block_weights(st, bi, ctx))
    print("  loaded", len(dbw), "double +", len(sbw), "single block weights")

    # ── LoRA set with NON-ZERO B (kaiming A is nonzero; B inits to 0 which would
    # zero the A-gradients — perturb B so both d_A and d_B are non-trivial). ──
    var lora = build_klein_lora_set(NUM_DOUBLE, NUM_SINGLE, D, F, RANK, ALPHA)
    var bseed = UInt64(1000)
    for ref ad in lora.dbl:
        ad.b = _fill_bf16(len(ad.b), bseed, Float32(0.03))
        bseed += 1
    for ref ad in lora.sgl:
        ad.b = _fill_bf16(len(ad.b), bseed, Float32(0.03))
        bseed += 1
    var lora_dev = klein_lora_set_to_device(lora, ctx)
    var nd = NUM_DOUBLE * DBL_SLOTS
    var ns = NUM_SINGLE * SGL_SLOTS
    print("  LoRA adapters:", nd, "double-slot +", ns, "single-slot (B perturbed nonzero)")

    # ── two real cache samples -> img/txt token tensors (shared by B1 and B2) ─
    var cache = KleinCache(String(CACHE_DIR))
    if cache.count() < 2:
        raise Error("klein batch-2 parity: need >= 2 cache samples")
    var samp0 = cache.load(0, ctx)
    var samp1 = cache.load(1, ctx)
    var img_tok0 = TArc(reshape(samp0.latent, [N_IMG, IN_CH], ctx))
    var txt_tok0 = TArc(reshape(samp0.text_embedding, [N_TXT, TXT_CH], ctx))
    var img_tok1 = TArc(reshape(samp1.latent, [N_IMG, IN_CH], ctx))
    var txt_tok1 = TArc(reshape(samp1.text_embedding, [N_TXT, TXT_CH], ctx))

    # fixed shared rope table (parity is rope-agnostic; B1==B2 inputs).
    var cos_t = Tensor.from_host(_fill(S * H * (Dh // 2), 500, 1.0), [S * H, Dh // 2], STDtype.F32, ctx)
    var sin_t = Tensor.from_host(_fill(S * H * (Dh // 2), 600, 1.0), [S * H, Dh // 2], STDtype.F32, ctx)

    # fixed per-sample targets (any non-degenerate vector; shared by B1 and B2).
    var target0 = _fill(N_IMG * OUT_CH, 700, 0.5)
    var target1 = _fill(N_IMG * OUT_CH, 800, 0.5)
    var empty = List[Float32]()

    # ── B1 reference: two single-sample runs ─────────────────────────────────
    print("  running B1 reference (sample 0, sigma=", sigma0, ") ...")
    var fwd0 = klein_stack_lora_forward_device_inputs_resident_moddev_rope[H, Dh, N_IMG, N_TXT, S](
        img_tok0, txt_tok0, base0, dbw, sbw, lora_dev,
        img_mod0, txt_mod0, single_mod0, cos_t, sin_t,
        D, F, IN_CH, TXT_CH, OUT_CH, eps, ctx,
    )
    var loss0 = _mse_loss(fwd0.out, target0)
    var g0 = klein_stack_lora_backward_resident_moddev_rope[H, Dh, N_IMG, N_TXT, S](
        _mse_dloss(fwd0.out, target0), empty.copy(), empty.copy(), base0, dbw, sbw, lora_dev,
        img_mod0, txt_mod0, single_mod0, cos_t, sin_t, fwd0,
        D, F, IN_CH, TXT_CH, OUT_CH, eps, ctx, False, False,
    )

    print("  running B1 reference (sample 1, sigma=", sigma1, ") ...")
    var fwd1 = klein_stack_lora_forward_device_inputs_resident_moddev_rope[H, Dh, N_IMG, N_TXT, S](
        img_tok1, txt_tok1, base1, dbw, sbw, lora_dev,
        img_mod1, txt_mod1, single_mod1, cos_t, sin_t,
        D, F, IN_CH, TXT_CH, OUT_CH, eps, ctx,
    )
    var loss1 = _mse_loss(fwd1.out, target1)
    var g1 = klein_stack_lora_backward_resident_moddev_rope[H, Dh, N_IMG, N_TXT, S](
        _mse_dloss(fwd1.out, target1), empty.copy(), empty.copy(), base1, dbw, sbw, lora_dev,
        img_mod1, txt_mod1, single_mod1, cos_t, sin_t, fwd1,
        D, F, IN_CH, TXT_CH, OUT_CH, eps, ctx, False, False,
    )
    var loss_ref = (loss0 + loss1) * 0.5
    print("  B1 loss0=", Float32(loss0), " loss1=", Float32(loss1), " mean=", Float32(loss_ref))

    # ── B2: both samples in one forward/backward ─────────────────────────────
    print("  running B2 (both samples, joint 2N-mean loss) ...")
    var fwd_b2 = klein_stack_lora_forward_b2[H, Dh, N_IMG, N_TXT, S](
        img_tok0, txt_tok0, img_tok1, txt_tok1, base, dbw, sbw, lora_dev,
        img_mod0, txt_mod0, single_mod0, img_mod1, txt_mod1, single_mod1,
        final_shift0, final_scale0, final_shift1, final_scale1, cos_t, sin_t,
        D, F, IN_CH, TXT_CH, OUT_CH, eps, ctx,
    )
    var loss_b2 = (_mse_loss(fwd_b2.s0.out, target0) + _mse_loss(fwd_b2.s1.out, target1)) * 0.5
    # joint 2N-mean d_loss per sample == HALF the single-sample d_loss.
    var d0 = _scaled(_mse_dloss(fwd_b2.s0.out, target0), Float32(0.5))
    var d1 = _scaled(_mse_dloss(fwd_b2.s1.out, target1), Float32(0.5))
    var g_b2 = klein_stack_lora_backward_b2[H, Dh, N_IMG, N_TXT, S](
        d0, d1, base, dbw, sbw, lora_dev,
        img_mod0, txt_mod0, single_mod0, img_mod1, txt_mod1, single_mod1,
        final_scale0, final_scale1, cos_t, sin_t, fwd_b2,
        D, F, IN_CH, TXT_CH, OUT_CH, eps, ctx,
    )

    # ── assertions ───────────────────────────────────────────────────────────
    var ok = True

    var denom = loss_ref if loss_ref > 1.0e-12 else 1.0e-12
    var ld = loss_b2 - loss_ref
    var loss_rel = (ld if ld >= 0.0 else -ld) / denom
    print("  loss_B2=", Float32(loss_b2), " mean(B1)=", Float32(loss_ref),
          " rel=", Float32(loss_rel))
    if loss_rel > LOSS_REL_THRESH:
        print("  FAIL: loss_B2 != mean(loss_B1) (rel", Float32(loss_rel), ">", Float32(LOSS_REL_THRESH), ")")
        ok = False

    var worst_cos = Float64(2.0)
    var worst_rel = Float64(0.0)
    # double adapters
    for i in range(nd):
        var ca = _cosine(g_b2.dbl_d_a[i], _mean2(g0.dbl_d_a[i], g1.dbl_d_a[i]))
        var ra = _rel_l2(g_b2.dbl_d_a[i], _mean2(g0.dbl_d_a[i], g1.dbl_d_a[i]))
        var cb = _cosine(g_b2.dbl_d_b[i], _mean2(g0.dbl_d_b[i], g1.dbl_d_b[i]))
        var rb = _rel_l2(g_b2.dbl_d_b[i], _mean2(g0.dbl_d_b[i], g1.dbl_d_b[i]))
        if ca < worst_cos: worst_cos = ca
        if cb < worst_cos: worst_cos = cb
        if ra > worst_rel: worst_rel = ra
        if rb > worst_rel: worst_rel = rb
        if ca < COS_THRESH or cb < COS_THRESH or ra > REL_L2_THRESH or rb > REL_L2_THRESH:
            print("  FAIL double slot", i, " cosA=", Float32(ca), " cosB=", Float32(cb),
                  " relA=", Float32(ra), " relB=", Float32(rb))
            ok = False
    # single adapters
    for i in range(ns):
        var ca = _cosine(g_b2.sgl_d_a[i], _mean2(g0.sgl_d_a[i], g1.sgl_d_a[i]))
        var ra = _rel_l2(g_b2.sgl_d_a[i], _mean2(g0.sgl_d_a[i], g1.sgl_d_a[i]))
        var cb = _cosine(g_b2.sgl_d_b[i], _mean2(g0.sgl_d_b[i], g1.sgl_d_b[i]))
        var rb = _rel_l2(g_b2.sgl_d_b[i], _mean2(g0.sgl_d_b[i], g1.sgl_d_b[i]))
        if ca < worst_cos: worst_cos = ca
        if cb < worst_cos: worst_cos = cb
        if ra > worst_rel: worst_rel = ra
        if rb > worst_rel: worst_rel = rb
        if ca < COS_THRESH or cb < COS_THRESH or ra > REL_L2_THRESH or rb > REL_L2_THRESH:
            print("  FAIL single slot", i, " cosA=", Float32(ca), " cosB=", Float32(cb),
                  " relA=", Float32(ra), " relB=", Float32(rb))
            ok = False

    print("  grad worst cosine=", Float32(worst_cos), " worst rel-L2=", Float32(worst_rel),
          " (thresholds cos>=", Float32(COS_THRESH), " rel<=", Float32(REL_L2_THRESH), ")")

    if not ok:
        raise Error("Klein batch-2 parity FAILED (loss or grad mismatch vs mean of B1)")

    print("")
    print("PASS: Klein TRUE batch-2 loss == mean(loss_B1s) and every adapter grad ==",
          "mean(grad_B1s) (cosine >=", Float32(COS_THRESH), ", rel-L2 <=", Float32(REL_L2_THRESH), ")")
