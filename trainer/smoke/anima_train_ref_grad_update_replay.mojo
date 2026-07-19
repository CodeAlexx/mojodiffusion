# Replay the real Serenity/SerenityTrainer Anima train-step dump through Mojo
# forward -> loss -> LoRA backward -> AdamW(lr=0), then print the RAW LoRA grads
# (d_A / d_B) before the optimizer step.
#
# Mirrors smoke/ernie_train_ref_grad_update_replay.mojo, adapted to Anima's
# single-step dump (anima_train_ref_step000_adapters.safetensors, 4 phases
# {adapter_pre, adapter_before, adapter_after, adapter_post} x 560 tensors =
# 280 modules x {lora_down, lora_up}; 28 blocks x 10 slots).
#
# CRITICAL — lr=0 at step 0 (warmup; meta steps[0].lr_before = 0.0, and
# adapter_after lora_up l2 == adapter_before lora_up l2 == 0):
#   the AdamW step does NOTHING (adapter_after == adapter_before), so the adapter
#   delta CANNOT gate the backward. The backward is gated by:
#     (a) the d_A=0 invariant — B is zero-init in adapter_before, so the raw LoRA
#         A-gradient d_A = scale * B^T d_y x^T == 0 EXACTLY; any material d_A is a
#         real bug in the A-grad arm.
#     (b) raw d_B l2 vs meta grad_norm_no_clip = 0.0014594601234421134. With d_A=0,
#         the global grad-norm over all trainable params == d_B l2.
#   We LOAD adapter_before A (lora_down) into the carrier (B stays 0) and print
#   BOTH raw d_A and d_B stats (max + l2) BEFORE adamw — exactly like the Ernie
#   grad smoke. We do NOT rely on adapter_after for grad parity.
#
# Loss: patch-space MSE. We patchify output.target [B,16,1,64,64] the SAME way
# the stack emits fwd.out, so mean MSE over patch space == mean MSE over
# output.{predicted,target} (bijective pack). d_loss = (2/N)(pred-tgt).
#
# EXPECTED-DIVERGENCE CAVEAT (report, do not chase): the serenitymojo Anima
# TRAINING forward uses a SIMPLIFIED single-axis RoPE + NO cross-attn text mask
# (train_anima_real.mojo _rope_tables), whereas the SerenityTrainer reference DiT used
# the real 3D RoPE + padding mask. So the Mojo loss and raw d_B WILL differ from
# the reference trainer loss_for_backward / grad_norm_no_clip; the d_A=0 invariant and nonfinite=0
# hold regardless, and the LoRA backward MATH is proven 64/64 vs torch.autograd
# (serenitymojo/models/anima/parity/lora_stack_parity.mojo). The MAIN LOOP owns
# the bar; this smoke prints numbers for re-verification.

from std.collections import List, Optional
from std.gpu.host import DeviceContext
from std.math import sqrt, log as flog, cos as fcos, sin as fsin, exp as fexp

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.tensor import Tensor

from serenitymojo.ops.linear import linear
from serenitymojo.ops.activations import silu
from serenitymojo.ops.norm import rms_norm

from serenity_trainer.model.anima.weights import (
    AnimaStackBase, load_anima_stack_base, verify_anima_stack_shapes,
)
from serenity_trainer.model.anima.anima_block import (
    ANIMA_SLOTS,
    SLOT_SA_Q, SLOT_SA_K, SLOT_SA_V, SLOT_SA_O,
    SLOT_CA_Q, SLOT_CA_K, SLOT_CA_V, SLOT_CA_O, SLOT_MLP1, SLOT_MLP2,
)
from serenity_trainer.model.anima.anima_stack_lora import (
    AnimaLoraSet, AnimaLoraGrads, build_anima_lora_set,
    anima_stack_lora_forward_streamed, anima_stack_lora_backward_streamed,
    anima_lora_adamw_step, LoraAdapter,
)


# ── arch ─────────────────────────────────────────────────────────────────────
comptime B = 1
comptime H = 16
comptime Dh = 128
comptime D = H * Dh             # 2048
comptime F = 8192
comptime JOINT = 1024
comptime C = 16
comptime PS = 2
comptime IN_PATCH = (C + 1) * PS * PS   # 68
comptime OUT_PATCH = C * PS * PS        # 64
comptime NUM_LAYERS = 28
comptime EPS = Float32(1e-06)

comptime LAT_HW = 64
comptime S_IMG = (LAT_HW // PS) * (LAT_HW // PS)   # 1024
comptime S_TXT = 512
comptime BATCH = 2

# ── recipe (meta runtime_config + steps[0]) ──────────────────────────────────
comptime RANK = 16
comptime ALPHA = Float32(1.0)         # lora_alpha=1.0 -> scale 1/16
comptime LR = Float32(0.0)            # warmup step 0: lr_before = 0.0
comptime BETA1 = Float32(0.9)
comptime BETA2 = Float32(0.999)
comptime ADAM_EPS = Float32(1.0e-8)
comptime WEIGHT_DECAY = Float32(0.01)

comptime REF_GRAD_NORM_NO_CLIP = Float64(0.0014594601234421134)
comptime REF_LOSS = Float64(0.0667838305234909)

comptime PARITY = "trainer/parity/anima_train_ref_step000.safetensors"
comptime ADAPTERS = "trainer/parity/anima_train_ref_step000_adapters.safetensors"
comptime CKPT = "/home/alex/.serenity/models/anima/split_files/diffusion_models/anima-base-v1.0.safetensors"


def _zeros(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for _ in range(n):
        out.append(Float32(0.0))
    return out^


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


def _slice_sample(flat: List[Float32], b: Int, per: Int) -> List[Float32]:
    var out = List[Float32]()
    var base = b * per
    for i in range(per):
        out.append(flat[base + i])
    return out^


# [C,H,W] channels-first (one sample, T=1) -> [H,W,C] channels-last
# (train_anima_real.mojo:589-600).
def _chw_to_hwc(src: List[Float32]) -> List[Float32]:
    var out = List[Float32]()
    var hw = LAT_HW * LAT_HW
    for _ in range(hw * C):
        out.append(Float32(0.0))
    for c in range(C):
        for h in range(LAT_HW):
            for w in range(LAT_HW):
                out[(h * LAT_HW + w) * C + c] = src[c * hw + h * LAT_HW + w]
    return out^


# _patchify_in (train_anima_real.mojo:338): [H,W,C] -> [N,68], channel SLOWEST,
# mask channel = 0.
def _patchify_in(x: List[Float32]) -> List[Float32]:
    var nH = LAT_HW // PS
    var nW = LAT_HW // PS
    var Cp = C + 1
    var N = nH * nW
    var pd = Cp * PS * PS
    var out = List[Float32]()
    for _ in range(N * pd):
        out.append(Float32(0.0))
    for ih in range(nH):
        for iw in range(nW):
            var pn = ih * nW + iw
            for c in range(Cp):
                for ph in range(PS):
                    for pw in range(PS):
                        var od = pn * pd + (c * PS * PS + ph * PS + pw)
                        if c < C:
                            var hh = ih * PS + ph
                            var ww = iw * PS + pw
                            out[od] = x[(hh * LAT_HW + ww) * C + c]
    return out^


# _patchify_out (train_anima_real.mojo:373): [H,W,C] -> [N,64], channel FASTEST.
def _patchify_out(x: List[Float32]) -> List[Float32]:
    var nH = LAT_HW // PS
    var nW = LAT_HW // PS
    var N = nH * nW
    var pd = C * PS * PS
    var out = List[Float32]()
    for _ in range(N * pd):
        out.append(Float32(0.0))
    for ih in range(nH):
        for iw in range(nW):
            var pn = ih * nW + iw
            for ph in range(PS):
                for pw in range(PS):
                    for c in range(C):
                        var od = pn * pd + (ph * PS * C + pw * C + c)
                        var hh = ih * PS + ph
                        var ww = iw * PS + pw
                        out[od] = x[(hh * LAT_HW + ww) * C + c]
    return out^


def _sinusoidal_host(sigma: Float32, dim: Int) -> List[Float32]:
    var half = dim // 2
    var neg_ln = -flog(Float32(10000.0))
    var out = List[Float32]()
    for _ in range(dim):
        out.append(Float32(0.0))
    for i in range(half):
        var freq = fexp(neg_ln * (Float32(i) / Float32(half)))
        var angle = sigma * freq
        out[i] = fcos(angle)
        out[half + i] = fsin(angle)
    return out^


struct _TEmb(Movable):
    var t_cond: List[Float32]
    var base_adaln: List[Float32]

    def __init__(out self, var t_cond: List[Float32], var base_adaln: List[Float32]):
        self.t_cond = t_cond^
        self.base_adaln = base_adaln^


def _prepare_timestep(sigma: Float32, base: AnimaStackBase, ctx: DeviceContext) raises -> _TEmb:
    var emb_l = _sinusoidal_host(sigma, D)
    var emb = Tensor.from_host(emb_l, [B, D], STDtype.F32, ctx)
    var h = linear(emb, base.te_lin1[], Optional[Tensor](None), ctx)
    var hidden = silu(h, ctx)
    var base_adaln = linear(hidden, base.te_lin2[], Optional[Tensor](None), ctx)
    var t_cond = rms_norm(emb, base.t_norm[], EPS, ctx)
    return _TEmb(t_cond.to_host(ctx), base_adaln.to_host(ctx))


struct _Rope(Movable):
    var cos: Tensor
    var sin: Tensor

    def __init__(out self, var cos: Tensor, var sin: Tensor):
        self.cos = cos^
        self.sin = sin^


def _rope_tables(ctx: DeviceContext) raises -> _Rope:
    var half = Dh // 2
    var cosl = List[Float32]()
    var sinl = List[Float32]()
    for _b in range(B):
        for s in range(S_IMG):
            for _h in range(H):
                for i in range(half):
                    var ang = Float32(s) / (Float32(10000.0) ** (Float32(2 * i) / Float32(Dh)))
                    cosl.append(fcos(ang))
                    sinl.append(fsin(ang))
    var cos = Tensor.from_host(cosl, [B * S_IMG * H, half], STDtype.F32, ctx)
    var sin = Tensor.from_host(sinl, [B * S_IMG * H, half], STDtype.F32, ctx)
    return _Rope(cos^, sin^)


# slot -> SerenityTrainer diffusers module name (serenitymojo anima_stack_lora.mojo
# _anima_serenity_module:958). The adapter dump key is
# adapter_before.transformer.transformer_blocks.{bi}.{module}.lora_{down,up}.weight
def _module_name(slot: Int) -> String:
    if slot == SLOT_SA_Q:
        return String("attn1.to_q")
    elif slot == SLOT_SA_K:
        return String("attn1.to_k")
    elif slot == SLOT_SA_V:
        return String("attn1.to_v")
    elif slot == SLOT_SA_O:
        return String("attn1.to_out.0")
    elif slot == SLOT_CA_Q:
        return String("attn2.to_q")
    elif slot == SLOT_CA_K:
        return String("attn2.to_k")
    elif slot == SLOT_CA_V:
        return String("attn2.to_v")
    elif slot == SLOT_CA_O:
        return String("attn2.to_out.0")
    elif slot == SLOT_MLP1:
        return String("ff.net.0.proj")
    return String("ff.net.2")


def _read_adapter(
    ad: ShardedSafeTensors, phase: String, bi: Int, slot: Int, kind: String, ctx: DeviceContext
) raises -> List[Float32]:
    var key = (phase + String(".transformer.transformer_blocks.") + String(bi)
               + String(".") + _module_name(slot) + String(".") + kind + String(".weight"))
    return Tensor.from_view(ad.tensor_view(key), ctx).to_host(ctx)


# Fresh adapter (A=a_vals, B=0, optimizer state 0) preserving rank/in/out/scale
# (mirrors ernie _set_adapter / chroma _set_adapter).
def _set_adapter(set: AnimaLoraSet, idx: Int, var a_vals: List[Float32]) -> LoraAdapter:
    ref a = set.ad[idx]
    var inf = a.in_f
    var outf = a.out_f
    var sc = a.scale
    return LoraAdapter(
        a_vals^, _zeros(outf * RANK), RANK, inf, outf, sc,
        _zeros(RANK * inf), _zeros(RANK * inf), _zeros(outf * RANK), _zeros(outf * RANK),
    )


def main() raises:
    var ctx = DeviceContext()

    print("=== Anima train-ref grad/update replay ===")
    print("[parity]  ", PARITY)
    print("[adapters]", ADAPTERS)
    print("[ckpt]    ", CKPT)
    print("[shape] S_IMG =", S_IMG, "S_TXT =", S_TXT, "BATCH =", BATCH)

    var st = ShardedSafeTensors.open(String(PARITY))
    var img_all = _dump_f32(st, String("trace.transformer_hidden_states"), ctx)  # [B,16,1,64,64]
    var ctx_all = _dump_f32(st, String("trace.encoder_hidden_states"), ctx)      # [B,512,1024]
    var tgt_all = _dump_f32(st, String("output.target"), ctx)                    # [B,16,1,64,64]
    var ts_all = _dump_f32(st, String("trace.transformer_timestep"), ctx)        # [B] sigma
    print("[dump] timesteps =", ts_all[0], ts_all[1])

    var adp = ShardedSafeTensors.open(String(ADAPTERS))

    var ckpt_st = SafeTensors.open(String(CKPT))
    verify_anima_stack_shapes(ckpt_st, NUM_LAYERS)
    var base = load_anima_stack_base(ckpt_st, ctx)
    print("[load] base resident; blocks stream per-block")

    var rope = _rope_tables(ctx)

    # carrier (A=randn, B=0), then OVERRIDE A from adapter_before (B stays 0).
    var lora = build_anima_lora_set(NUM_LAYERS, D, JOINT, F, RANK, ALPHA)
    for bi in range(NUM_LAYERS):
        for s in range(ANIMA_SLOTS):
            var idx = bi * ANIMA_SLOTS + s
            var a_vals = _read_adapter(adp, String("adapter_before"), bi, s, String("lora_down"), ctx)
            lora.ad[idx] = _set_adapter(lora, idx, a_vals^)
    print("[lora] adapters:", NUM_LAYERS * ANIMA_SLOTS, " A<-adapter_before, B=0")

    var img_per = C * S_IMG * PS * PS    # C*LAT_HW*LAT_HW
    var ctx_per = S_TXT * JOINT
    var n_total = BATCH * S_IMG * OUT_PATCH
    var inv_n = Float32(2.0) / Float32(n_total)

    var nonfinite_total = 0
    var loss_sum = Float64(0.0)

    var acc_a = List[List[Float32]]()
    var acc_b = List[List[Float32]]()
    var have_acc = False

    for b in range(BATCH):
        var patches = _patchify_in(_chw_to_hwc(_slice_sample(img_all, b, img_per)))
        var context = _slice_sample(ctx_all, b, ctx_per)
        var target_patches = _patchify_out(_chw_to_hwc(_slice_sample(tgt_all, b, img_per)))
        var sigma = ts_all[b]
        var temb = _prepare_timestep(sigma, base, ctx)

        var fwd = anima_stack_lora_forward_streamed[H, Dh, S_IMG, S_TXT](
            patches.copy(), temb.t_cond.copy(), temb.base_adaln.copy(), context.copy(),
            base, ckpt_st, lora, rope.cos, rope.sin,
            B, D, JOINT, F, IN_PATCH, OUT_PATCH, EPS, ctx,
        )

        var d_out = List[Float32]()
        var nout = len(fwd.out)
        for i in range(nout):
            var diff = fwd.out[i] - target_patches[i]
            loss_sum += Float64(diff) * Float64(diff)
            d_out.append(inv_n * diff)

        var grads_b = anima_stack_lora_backward_streamed[H, Dh, S_IMG, S_TXT](
            d_out, patches.copy(), temb.t_cond.copy(), temb.base_adaln.copy(), context.copy(),
            base, ckpt_st, lora, rope.cos, rope.sin, fwd,
            B, D, JOINT, F, IN_PATCH, OUT_PATCH, EPS, ctx,
        )
        nonfinite_total += grads_b.nonfinite_lora_grads
        print("[sample", b, "] forward+backward done; nonfinite_lora_grads =",
              grads_b.nonfinite_lora_grads)

        if not have_acc:
            for i in range(len(grads_b.d_a)):
                acc_a.append(grads_b.d_a[i].copy())
                acc_b.append(grads_b.d_b[i].copy())
            have_acc = True
        else:
            for i in range(len(grads_b.d_a)):
                for j in range(len(grads_b.d_a[i])):
                    acc_a[i][j] = acc_a[i][j] + grads_b.d_a[i][j]
                for j in range(len(grads_b.d_b[i])):
                    acc_b[i][j] = acc_b[i][j] + grads_b.d_b[i][j]

    var loss = loss_sum / Float64(n_total)
    print("[loss] mojo =", loss, " (ref output.loss_for_backward =", REF_LOSS,
          "; rope/mask caveat)")

    # raw grad stats BEFORE adamw (the real gate; B=0 invariant ⇒ d_A ~ 0).
    var da_max = Float64(0.0)
    var da_l2 = Float64(0.0)
    var db_max = Float64(0.0)
    var db_l2 = Float64(0.0)
    for i in range(len(acc_a)):
        for j in range(len(acc_a[i])):
            var v = Float64(acc_a[i][j])
            var av = v if v >= 0.0 else -v
            if av > da_max: da_max = av
            da_l2 += v * v
        for j in range(len(acc_b[i])):
            var v = Float64(acc_b[i][j])
            var av = v if v >= 0.0 else -v
            if av > db_max: db_max = av
            db_l2 += v * v
    var da_l2r = da_l2 ** 0.5
    var db_l2r = db_l2 ** 0.5
    var total_norm = (da_l2 + db_l2) ** 0.5
    print("[raw grad] d_A: max =", da_max, " l2 =", da_l2r, "  (B=0 invariant ⇒ must be ~0)")
    print("[raw grad] d_B: max =", db_max, " l2 =", db_l2r)
    print("[grad norm] total(d_A,d_B) l2 =", total_norm,
          "  ref grad_norm_no_clip =", REF_GRAD_NORM_NO_CLIP,
          "  ratio =", total_norm / REF_GRAD_NORM_NO_CLIP)
    print("[nonfinite] raw lora grads =", nonfinite_total)

    # AdamW with lr=0 (warmup step) : adapter_after == adapter_before.
    var grads = AnimaLoraGrads(
        acc_a^, acc_b^, List[Float32](), List[Float32](), List[Float32](),
        List[Float32](), List[Float32](), List[Float32](), List[Float32](), nonfinite_total,
    )
    anima_lora_adamw_step(lora, grads, 1, LR, ctx, BETA1, BETA2, ADAM_EPS, WEIGHT_DECAY)
    print("[adamw] applied lr =", LR, " (no-op: adapter_after == adapter_before)")

    if nonfinite_total != 0:
        raise Error("Anima grad/update replay produced nonfinite LoRA grads")
    if da_max > 1.0e-6:
        print("WARNING: d_A max", da_max, "exceeds 1e-6 — B=0 invariant violated (investigate A-grad arm)")
    print("ANIMA TRAIN REF GRAD/UPDATE REPLAY DONE (main loop owns the bar)")
