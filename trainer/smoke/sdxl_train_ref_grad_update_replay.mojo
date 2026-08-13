# Replay the real Serenity SDXL train-step dump through Mojo forward -> eps-pred
# loss -> per-ST LoRA backward -> AdamW, then print the RAW LoRA grads (d_A / d_B)
# and confirm LoRA-B grows under the optimizer step.
#
# Mirrors smoke/{ernie,anima}_train_ref_grad_update_replay.mojo in STRUCTURE,
# adapted to SDXL's conv-UNet eps-predictor + the real-dims resident
# forward/backward the production driver calls (train_sdxl_real.mojo:524/:548/:555).
#
# ── HONEST GATE STATUS (NOT a grad-norm-vs-dump parity gate) ──
# Two facts (see sdxl_train_ref_forward_replay.mojo header) make a numeric grad
# gate against this dump impossible with the current production spine:
#   (1) sdxl_real_forward/backward[L] are SQUARE-ONLY; the oracle latent / target
#       are RECTANGULAR [1,4,168,96]. We run a square crop (L).
#   (2) the serenitymojo SDXL LoRA covers only the 11 SpatialTransformer blocks
#       (700 adapters), while the dump's grad_norm_no_clip=0.0106246 sums grads
#       over a FULL-UNet LoRA (794 modules incl. convs/embeds). Different
#       trainable set -> the norms are not comparable.
# So this is the real-weight WIRING gate for the train step. It checks:
#   (a) d_A == 0 invariant — B is zero-init, so the raw LoRA A-gradient
#       d_A = scale * B^T d_y x^T == 0 EXACTLY; any material d_A is a real bug.
#   (b) raw d_B is finite (nonfinite==0) and nonzero.
#   (c) after AdamW(lr=1e-4, the dump's optimizer lr), LoRA-B |.|_1 grows 0 -> >0,
#       proving the backward + optimizer actually drive the trained params.
# The eps-pred loss uses target ε = trace.latent_noise (square crop), consistent
# with noisy = trace.scaled_noisy_latent_image. The MAIN LOOP owns the final bar.
#
# ── KNOWN BACKWARD BLOCKER (found by THIS smoke; report, do not paper over) ──
# Forward runs clean on real BF16 weights (see sdxl_train_ref_forward_replay).
# The BACKWARD does not: on the real BF16 checkpoint it raises, in order,
#   conv2d_backward: x/grad_y dtype mismatch   (fixed here by a BF16 grad seed go)
#   linear_backward: grad_y/x/weight dtype mismatch  (STILL open)
# All 1680 UNet weights are uniformly BF16, so this is NOT mixed weights: a
# backward op (group_norm_backward / silu_backward / a LoRA-path linear) emits an
# F32 grad mid-stream, which the STRICT linear_backward (linalg_backward.mojo:401
# requires grad_y==x==weight dtype) rejects. The FORWARD survived only because
# linear()/group_norm() WIDEN internally; the backward ops do not. This path was
# never exercised: the 44/44 lora_stack_parity ran in pure F32 synthetic weights,
# and the "not production-tested" driver builds an F32 grad seed that would itself
# crash at conv2d_backward. NEEDS op-level grad-dtype unification before any SDXL
# real-weight grad gate is possible. Until then the holding numeric oracle gate is
# the loss replay (sdxl_train_ref_loss_replay, PASS @ 1.4e-6).

from std.collections import List, Optional
from max.gpu.host import DeviceContext
from std.math import sqrt

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.tensor import Tensor
from serenitymojo.models.vae.decoder2d import nchw_to_nhwc

from serenity_trainer.model.sdxl.weights import (
    build_sdxl_real_weights, sdxl_st_C, sdxl_st_Cff, sdxl_st_depth,
)
from serenity_trainer.model.sdxl.sdxl_unet_stack_lora import (
    SdxlLoraSet, build_sdxl_lora_set, sdxl_real_forward, sdxl_real_backward,
    SdxlRealGrads, N_ST, SDXL_SLOTS, LoraGrads, _lora_adamw,
)


# ── arch (sdxl.json) ──────────────────────────────────────────────────────────
comptime CCTX = 2048
comptime ADM = 2816
comptime RANK = 16
comptime ALPHA = Float32(16.0)

# ── recipe (dump steps[0].optimizer_before) ───────────────────────────────────
comptime LR = Float32(1.0e-4)
comptime BETA1 = Float32(0.9)
comptime BETA2 = Float32(0.999)
comptime ADAM_EPS = Float32(1.0e-8)
comptime WEIGHT_DECAY = Float32(0.01)
comptime CLIP = Float32(1.0)
comptime REF_GRAD_NORM_NO_CLIP = Float64(0.010624590329825878)   # full-UNet LoRA; NOT comparable

# ── square crop spatial (oracle is rectangular 168x96) ────────────────────────
comptime L = 16
comptime LAT_C = 4
comptime LAT_H = 168
comptime LAT_W = 96

comptime PARITY = "trainer/parity/sdxl_train_ref_step000.safetensors"
comptime CKPT = "/home/alex/.serenity/models/checkpoints/sdxl_unet_bf16.safetensors"

from std.math import log as flog, cos as fcos, sin as fsin, exp as fexp


def _dump_f32(st: ShardedSafeTensors, key: String, ctx: DeviceContext) raises -> List[Float32]:
    var t = Tensor.from_view(st.tensor_view(key), ctx)
    if t.dtype() == STDtype.BF16:
        var bf = t.to_host_bf16(ctx)
        var out = List[Float32]()
        for i in range(len(bf)):
            out.append(bf[i].cast[DType.float32]())
        return out^
    if t.dtype() == STDtype.F32:
        return t.to_host(ctx)
    return cast_tensor(t, STDtype.F32, ctx).to_host(ctx)


def _dump_i32(st: ShardedSafeTensors, key: String) raises -> List[Int]:
    var tv = st.tensor_view(key)
    var n = 1
    for i in range(len(tv.shape)):
        n *= tv.shape[i]
    var out = List[Int]()
    for i in range(n):
        var v = (
            Int(tv.data[i * 4 + 0]) | (Int(tv.data[i * 4 + 1]) << 8)
            | (Int(tv.data[i * 4 + 2]) << 16) | (Int(tv.data[i * 4 + 3]) << 24)
        )
        out.append(v)
    return out^


# sin_embed_256 — EXACT copy of train_sdxl_real.mojo:230-241.
def _sin_embed_256(value: Float32) -> List[Float32]:
    comptime DIM = 256
    comptime half = DIM // 2
    var data = List[Float32]()
    for _ in range(DIM):
        data.append(0.0)
    for j in range(half):
        var freq = Float32(fexp(-flog(10000.0) * Float64(j) / Float64(half)))
        var angle = value * freq
        data[j] = Float32(fcos(Float64(angle)))
        data[half + j] = Float32(fsin(Float64(angle)))
    return data^


def _sh4(a: Int, b: Int, c: Int, d: Int) -> List[Int]:
    var s = List[Int](); s.append(a); s.append(b); s.append(c); s.append(d); return s^


def _crop_nchw(src: List[Float32]) -> List[Float32]:
    var o = List[Float32]()
    for c in range(LAT_C):
        for hh in range(L):
            for ww in range(L):
                o.append(src[(c * LAT_H + hh) * LAT_W + ww])
    return o^


def _absum_b(set: SdxlLoraSet) -> Float32:
    var s = Float32(0.0)
    for i in range(len(set.ad)):
        ref a = set.ad[i]
        for j in range(len(a.b)):
            var x = a.b[j].cast[DType.float32]()
            s += x if x >= Float32(0.0) else -x
    return s


# AdamW over every adapter of every ST set (mirror train_sdxl_real.mojo _adamw_all).
def _adamw_all(
    mut sets: List[SdxlLoraSet], g: SdxlRealGrads, t: Int, lr: Float32,
    ctx: DeviceContext,
) raises:
    for s in range(N_ST):
        var n = sets[s].num_blocks * SDXL_SLOTS
        for i in range(n):
            if len(g.d_a[s][i]) == 0 and len(g.d_b[s][i]) == 0:
                continue
            var lg = LoraGrads(g.d_a[s][i].copy(), g.d_b[s][i].copy())
            _lora_adamw(sets[s].ad[i], lg, t, lr, ctx, BETA1, BETA2, ADAM_EPS, WEIGHT_DECAY)


def main() raises:
    var ctx = DeviceContext()

    print("=== SDXL train-ref grad/update replay (real-weight WIRING gate) ===")
    print("[parity]", PARITY)
    print("[ckpt]  ", CKPT)
    print("[note]  RECTANGULAR real-dims fwd+bwd at", LAT_H, "x", LAT_W)
    print("[note]  NOT a grad-norm gate: Mojo trains ST-only LoRA (700 adapters);")
    print("[note]  dump grad_norm sums a FULL-UNet LoRA (794 modules). See header.")

    var st = ShardedSafeTensors.open(String(PARITY))

    # ── oracle conditioning (same as forward replay) ──
    var pooled = _dump_f32(st, String("trace.added_cond_text_embeds"), ctx)
    var time_ids = _dump_f32(st, String("trace.added_cond_time_ids"), ctx)
    var tstep = _dump_i32(st, String("trace.unet_timestep"))
    var y_h = List[Float32]()
    for i in range(len(pooled)):
        y_h.append(pooled[i])
    for k in range(6):
        var se = _sin_embed_256(time_ids[k])
        for j in range(len(se)):
            y_h.append(se[j])
    var ys = List[Int](); ys.append(1); ys.append(ADM)
    var y = cast_tensor(Tensor.from_host(y_h^, ys^, STDtype.F32, ctx), STDtype.BF16, ctx)
    var context = Tensor.from_view(st.tensor_view(String("trace.encoder_hidden_states")), ctx)
    var t_h = List[Float32](); t_h.append(Float32(tstep[0]))
    var t_s = List[Int](); t_s.append(1)
    var t = Tensor.from_host(t_h^, t_s^, STDtype.F32, ctx)

    # ── noisy input + eps target (FULL rectangular, BF16 latent -> F32 act stream) ──
    var noisy = _dump_f32(st, String("trace.scaled_noisy_latent_image"), ctx)   # NCHW flat
    var eps_target = _dump_f32(st, String("trace.latent_noise"), ctx)            # NCHW flat
    var noisy_nchw = cast_tensor(
        Tensor.from_host(noisy^, _sh4(1, LAT_C, LAT_H, LAT_W), STDtype.F32, ctx), STDtype.BF16, ctx)
    var noisy_nhwc = nchw_to_nhwc(noisy_nchw, ctx)
    print("[input] noisy/eps rectangular NHWC [1,", LAT_H, ",", LAT_W, ",4]; timestep =", tstep[0])

    # ── LoRA carrier (B=0 init) ──
    var lora = List[SdxlLoraSet]()
    var n_adapters = 0
    for i in range(N_ST):
        var ls = build_sdxl_lora_set(sdxl_st_depth(i), sdxl_st_C(i), CCTX, sdxl_st_Cff(i), RANK, ALPHA)
        n_adapters += ls.num_blocks * SDXL_SLOTS
        lora.append(ls^)
    var b0 = Float32(0.0)
    for s in range(N_ST):
        b0 += _absum_b(lora[s])
    print("[lora] sets:", N_ST, " adapters:", n_adapters, " LoRA-B |.|_1 init =", b0, "(expect 0.0)")

    # ── real base weights ──
    print("[load] assembling real UNet weights ...")
    var stw = SafeTensors.open(String(CKPT))
    var w = build_sdxl_real_weights(stw, ctx)
    print("[load] weights ready")

    # ── forward (REAL rectangular dims) ──
    var fwd = sdxl_real_forward[LAT_H, LAT_W](noisy_nhwc, t, y.clone(ctx), context.clone(ctx), w, lora, ctx)
    var pred = fwd.out.to_host(ctx)   # NHWC [LAT_H*LAT_W*4]

    # ── eps-pred MSE loss + d_loss (NHWC). target ε in NCHW order -> NHWC index. ──
    var N_LAT = LAT_C * LAT_H * LAT_W
    var inv_n = Float32(2.0) / Float32(N_LAT)
    var sse = Float64(0.0)
    var d_loss = List[Float32]()
    for hh in range(LAT_H):
        for ww in range(LAT_W):
            for c in range(LAT_C):
                var nhwc_i = (hh * LAT_W + ww) * LAT_C + c
                var nchw_i = (c * LAT_H + hh) * LAT_W + ww
                var diff = pred[nhwc_i] - eps_target[nchw_i]
                sse += Float64(diff) * Float64(diff)
                d_loss.append(inv_n * diff)
    var loss = Float32(sse / Float64(N_LAT))
    print("[loss] eps-pred MSE (rectangular) =", loss)
    # F32 grad stream: the real-dims backward runs an F32-activation/F32-grad/
    # frozen-BF16-weight (mixed_base) contract — seed an F32 output grad.
    var go = Tensor.from_host(d_loss^, _sh4(1, LAT_H, LAT_W, LAT_C), STDtype.F32, ctx)

    # ── backward -> per-ST LoRA grads ──
    var grads = sdxl_real_backward[LAT_H, LAT_W](go, fwd.acts, w, lora, ctx)

    # ── raw grad stats (d_A=0 invariant; d_B finite/nonzero) ──
    var da_max = Float64(0.0); var da_l2 = Float64(0.0)
    var db_max = Float64(0.0); var db_l2 = Float64(0.0)
    for s in range(N_ST):
        for sl in range(len(grads.d_a[s])):
            for j in range(len(grads.d_a[s][sl])):
                var v = Float64(grads.d_a[s][sl][j])
                var av = v if v >= 0.0 else -v
                if av > da_max: da_max = av
                da_l2 += v * v
            for j in range(len(grads.d_b[s][sl])):
                var v = Float64(grads.d_b[s][sl][j])
                var av = v if v >= 0.0 else -v
                if av > db_max: db_max = av
                db_l2 += v * v
    var total_norm = (da_l2 + db_l2) ** 0.5
    print("[raw grad] d_A: max =", da_max, " l2 =", da_l2 ** 0.5, "  (B=0 invariant -> must be ~0)")
    print("[raw grad] d_B: max =", db_max, " l2 =", db_l2 ** 0.5)
    print("[grad norm] total(d_A,d_B) l2 =", total_norm,
          " (dump full-UNet grad_norm =", REF_GRAD_NORM_NO_CLIP, " NOT comparable)")
    print("[nonfinite] raw lora grads =", grads.nonfinite)

    # ── AdamW(lr=1e-4) over every adapter -> B must grow ──
    _adamw_all(lora, grads, 1, LR, ctx)
    var b1 = Float32(0.0)
    for s in range(N_ST):
        b1 += _absum_b(lora[s])
    print("[adamw] lr =", LR, " LoRA-B |.|_1 after =", b1, " (grew from", b0, ")")

    if grads.nonfinite != 0:
        raise Error("SDXL grad/update replay produced nonfinite LoRA grads")
    if da_max > 1.0e-6:
        raise Error("SDXL grad/update replay: d_A nonzero (B=0 invariant violated)")
    if not (b1 > b0):
        raise Error("SDXL grad/update replay: LoRA-B did not grow under AdamW")
    print("SDXL TRAIN REF GRAD/UPDATE REPLAY (WIRING) DONE — d_A~0, B grew, finite")
