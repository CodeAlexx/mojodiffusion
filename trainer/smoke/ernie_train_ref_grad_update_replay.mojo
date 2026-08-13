# Replay the real Serenity ERNIE-Image train-step dump through Mojo
# forward -> loss -> LoRA backward -> AdamW(lr=0), then print the RAW LoRA grads
# (d_A / d_B) before the optimizer step.
#
# Mirrors smoke/chroma_train_ref_grad_update_replay.mojo, adapted to ERNIE's
# single-step dump (ernie_train_ref_step000_adapters.safetensors, 4 phases
# {adapter_pre, adapter_before, adapter_after, adapter_post} x 504 tensors =
# 252 modules x {lora_down, lora_up}).
#
# CRITICAL — lr=0 (warmup step 0; meta steps[0].optimizer_before.lr = 0.0):
#   the AdamW step does NOTHING (adapter_after == adapter_before), so the adapter
#   delta CANNOT gate the backward. The backward is gated by:
#     (a) the d_A=0 invariant — B is zero-init in adapter_before, so the raw LoRA
#         A-gradient d_A = scale * B^T d_y x^T == 0 EXACTLY; any material d_A is a
#         real bug in the A-grad arm.
#     (b) raw d_B l2 vs meta grad_norm_no_clip = 0.000828770047519356. With d_A=0,
#         the global grad-norm over all trainable params == d_B l2.
#   We therefore LOAD adapter_before A (lora_down) into the carrier (B stays 0)
#   and print BOTH raw d_A and d_B stats (max + l2) BEFORE adamw — exactly like
#   the Chroma grad smoke. We do NOT rely on adapter_after for grad parity.
#
# Loss: packed-space MSE. trace.flow [B,128,40,28] is the packed flow target;
# fwd.out is the Mojo packed prediction [N_IMG,128]. Mean MSE over packed space
# == mean MSE over output.{predicted,target} [B,32,80,56] (bijective pack), so it
# reproduces output.loss_for_backward = 0.643847644329071. d_loss = (2/N)(pred-tgt).
#
# Storage note: Mojo LoRA tensors are BF16 and block compute is BF16, so the raw
# grads carry ~bf16 rounding vs Serenity's F32 path. The MAIN LOOP owns the bar;
# this smoke prints numbers for re-verification.

from std.collections import List, Optional
from max.gpu.host import DeviceContext
from std.math import sqrt

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.tensor import Tensor

from serenitymojo.ops.linear import linear
from serenitymojo.ops.activations import silu
from serenitymojo.ops.embeddings import timestep_embedding_sin_first
from serenitymojo.models.dit.ernie_image import build_ernie_rope_tables

from serenitymojo.training.train_step import LoraAdapter

from serenity_trainer.model.ernie.weights import (
    ErnieStackBase, load_ernie_stack_base, load_ernie_all_blocks_bf16_normf32,
)
from serenity_trainer.model.ernie.ernie_block import ErnieModVecs, ERNIE_SLOTS
from serenity_trainer.model.ernie.ernie_stack_lora import (
    ErnieLoraSet, ErnieLoraGrads, build_ernie_lora_set, ernie_lora_set_to_device,
    ernie_stack_lora_forward_resident_device,
    ernie_stack_lora_backward_resident_device,
    ernie_lora_adamw_step,
)


# ── arch ─────────────────────────────────────────────────────────────────────
comptime H = 32
comptime Dh = 128
comptime D = H * Dh            # 4096
comptime F = 12288
comptime IN_CH = 128
comptime TEXT_IN = 3072
comptime OUT_CH = 128
comptime NUM_LAYERS = 36
comptime EPS = Float32(1e-06)

comptime IMG_H = 40
comptime IMG_W = 28
comptime N_IMG = IMG_H * IMG_W   # 1120
comptime N_TXT = 201
comptime S = N_IMG + N_TXT       # 1321
comptime BATCH = 2

# ── recipe (meta steps[0]) ───────────────────────────────────────────────────
comptime RANK = 16
comptime ALPHA = Float32(1.0)         # lora_alpha=1.0 -> scale 1/16
comptime LR = Float32(0.0)            # warmup step 0: optimizer lr = 0.0
comptime BETA1 = Float32(0.9)
comptime BETA2 = Float32(0.999)
comptime ADAM_EPS = Float32(1.0e-8)
comptime WEIGHT_DECAY = Float32(0.01)
comptime CLIP_GRAD_NORM = Float32(1.0)

comptime REF_GRAD_NORM_NO_CLIP = Float64(0.000828770047519356)
comptime REF_LOSS = Float64(0.643847644329071)

comptime PARITY = "trainer/parity/ernie_train_ref_step000.safetensors"
comptime ADAPTERS = "trainer/parity/ernie_train_ref_step000_adapters.safetensors"
comptime CKPT = "/home/alex/models/ERNIE-Image/transformer"


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


# Read an I32 dump tensor directly from raw bytes (cast_tensor has no I32 path).
def _dump_i32(st: ShardedSafeTensors, key: String) raises -> List[Int]:
    var tv = st.tensor_view(key)
    var n = 1
    for i in range(len(tv.shape)):
        n *= tv.shape[i]
    var out = List[Int]()
    for i in range(n):
        var v = (
            Int(tv.data[i * 4 + 0])
            | (Int(tv.data[i * 4 + 1]) << 8)
            | (Int(tv.data[i * 4 + 2]) << 16)
            | (Int(tv.data[i * 4 + 3]) << 24)
        )
        out.append(v)
    return out^


# [128,40,28] flat (NCHW; one sample) -> [N_IMG,128] token-major (NHWC); see
# ernie_train_ref_forward_replay.mojo / train_ernie_real.mojo:186-192.
def _pack_nchw_to_tokens(src: List[Float32]) -> List[Float32]:
    var out = List[Float32]()
    var hw = IMG_H * IMG_W
    for t in range(hw):
        for ch in range(IN_CH):
            out.append(src[ch * hw + t])
    return out^


def _chunk(src: List[Float32], idx: Int, width: Int) -> List[Float32]:
    var o = List[Float32]()
    var off = idx * width
    for i in range(width):
        o.append(src[off + i])
    return o^


# Shared-AdaLN source (train_ernie_real.mojo:319-348). See forward smoke.
def _shared_adaln_source(
    base: ErnieStackBase, sigma_idx: Int, ctx: DeviceContext
) raises -> Tuple[ErnieModVecs, List[Float32], List[Float32]]:
    var ts = List[Float32]()
    ts.append(Float32(sigma_idx))
    var ts_t = Tensor.from_host(ts, [1], STDtype.F32, ctx)
    var emb_in = timestep_embedding_sin_first(ts_t, D, ctx, 10000.0, base.te_w1[].dtype())
    var h1 = linear(emb_in, base.te_w1[], Optional[Tensor](base.te_b1[].clone(ctx)), ctx)
    h1 = silu(h1, ctx)
    var c = linear(h1, base.te_w2[], Optional[Tensor](base.te_b2[].clone(ctx)), ctx)
    var sc = silu(c, ctx)
    var adaln = linear(sc, base.adaln_w[], Optional[Tensor](base.adaln_b[].clone(ctx)), ctx)
    var adaln_h = adaln.to_host(ctx)
    var fmod = linear(c, base.final_norm_w[], Optional[Tensor](base.final_norm_b[].clone(ctx)), ctx)
    var fmod_h = fmod.to_host(ctx)
    var mv = ErnieModVecs(
        _chunk(adaln_h, 0, D), _chunk(adaln_h, 1, D), _chunk(adaln_h, 2, D),
        _chunk(adaln_h, 3, D), _chunk(adaln_h, 4, D), _chunk(adaln_h, 5, D),
    )
    return (mv^, _chunk(fmod_h, 0, D), _chunk(fmod_h, 1, D))


# ── trained-slot inventory (layer_filter self_attention,mlp = ALL 7 slots) ────
# returns parallel lists: carrier flat index, dump module name (block-qualified).
def _module_name(slot: Int) -> String:
    if slot == 0:
        return String("self_attention.to_q")
    elif slot == 1:
        return String("self_attention.to_k")
    elif slot == 2:
        return String("self_attention.to_v")
    elif slot == 3:
        return String("self_attention.to_out.0")
    elif slot == 4:
        return String("mlp.gate_proj")
    elif slot == 5:
        return String("mlp.up_proj")
    return String("mlp.linear_fc2")


def _trained_targets() -> Tuple[List[Int], List[String]]:
    var idxs = List[Int]()
    var keys = List[String]()
    for bi in range(NUM_LAYERS):
        var lp = String("transformer.layers.") + String(bi) + String(".")
        for s in range(ERNIE_SLOTS):
            idxs.append(bi * ERNIE_SLOTS + s)
            keys.append(lp + _module_name(s))
    return (idxs^, keys^)


def _adapter_key(phase: String, module: String, kind: String) -> String:
    return phase + String(".") + module + String(".") + kind + String(".weight")


def _read_adapter(
    ad: ShardedSafeTensors, phase: String, module: String, kind: String, ctx: DeviceContext
) raises -> List[Float32]:
    return Tensor.from_view(ad.tensor_view(_adapter_key(phase, module, kind)), ctx).to_host(ctx)


# Construct a fresh adapter (A=a_vals, B=b_vals, optimizer state 0) at a slot,
# preserving its rank/in/out/scale (mirrors chroma _set_adapter).
def _set_adapter(set: ErnieLoraSet, idx: Int, var a_vals: List[Float32], var b_vals: List[Float32]) -> LoraAdapter:
    ref a = set.ad[idx]
    var inf = a.in_f
    var outf = a.out_f
    var sc = a.scale
    return LoraAdapter(
        a_vals^, b_vals^, RANK, inf, outf, sc,
        _zeros(RANK * inf), _zeros(RANK * inf), _zeros(outf * RANK), _zeros(outf * RANK),
    )


def main() raises:
    var ctx = DeviceContext()

    print("=== Ernie train-ref grad/update replay ===")
    print("[parity]  ", PARITY)
    print("[adapters]", ADAPTERS)
    print("[ckpt]    ", CKPT)
    print("[shape] N_IMG =", N_IMG, "N_TXT =", N_TXT, "S =", S, "BATCH =", BATCH)

    var st = ShardedSafeTensors.open(String(PARITY))
    var img_all = _dump_f32(st, String("trace.transformer_hidden_states"), ctx)
    var txt_all = _dump_f32(st, String("trace.encoder_hidden_states"), ctx)
    var flow_all = _dump_f32(st, String("trace.flow"), ctx)          # packed target [B,128,40,28]
    var ts_all = _dump_i32(st, String("trace.transformer_timestep"))
    print("[dump] timesteps =", ts_all[0], ts_all[1])

    var adp = ShardedSafeTensors.open(String(ADAPTERS))

    var ckpt_st = ShardedSafeTensors.open(String(CKPT))
    var base = load_ernie_stack_base(ckpt_st, D, IN_CH, ctx)
    var blocks = load_ernie_all_blocks_bf16_normf32(ckpt_st, NUM_LAYERS, ctx)
    print("[load] base + blocks resident:", len(blocks))

    var rope = build_ernie_rope_tables[N_IMG, N_TXT, H, Dh](IMG_H, IMG_W, N_TXT, ctx, STDtype.F32)

    # ── carrier: build (A=randn, B=0), then OVERRIDE A/B from adapter_before ──
    var lora = build_ernie_lora_set(NUM_LAYERS, D, F, RANK, ALPHA)
    var tt = _trained_targets()
    var t_idx = tt[0].copy()
    var t_key = tt[1].copy()
    var n_trained = len(t_idx)
    print("[lora] adapters:", NUM_LAYERS * ERNIE_SLOTS, " trained slots:", n_trained)
    for j in range(n_trained):
        var a_vals = _read_adapter(adp, String("adapter_before"), t_key[j], String("lora_down"), ctx)
        var b_vals = _read_adapter(adp, String("adapter_before"), t_key[j], String("lora_up"), ctx)
        lora.ad[t_idx[j]] = _set_adapter(lora, t_idx[j], a_vals^, b_vals^)
    var lora_dev = ernie_lora_set_to_device(lora, STDtype.BF16, ctx)

    var img_per = IN_CH * N_IMG
    var txt_per = N_TXT * TEXT_IN
    var flow_per = IN_CH * N_IMG

    var n_total = BATCH * N_IMG * OUT_CH
    var inv_n = Float32(2.0) / Float32(n_total)

    # ── per-sample forward -> d_loss -> backward, accumulate raw grads ──
    var da_max = Float64(0.0)
    var da_l2 = Float64(0.0)
    var db_max = Float64(0.0)
    var db_l2 = Float64(0.0)
    var nonfinite_total = 0
    var loss_sum = Float64(0.0)

    # accumulate d_a/d_b across the two samples element-wise
    var acc_a = List[List[Float32]]()
    var acc_b = List[List[Float32]]()
    var have_acc = False

    for b in range(BATCH):
        var img_tokens = _pack_nchw_to_tokens(_slice_sample(img_all, b, img_per))
        var txt_tokens = _slice_sample(txt_all, b, txt_per)
        var target_packed = _pack_nchw_to_tokens(_slice_sample(flow_all, b, flow_per))
        var sigma_idx = ts_all[b]

        var src = _shared_adaln_source(base, sigma_idx, ctx)
        var mv = src[0].copy()
        var f_scale = src[1].copy()
        var f_shift = src[2].copy()

        var fwd = ernie_stack_lora_forward_resident_device[H, Dh, N_IMG, N_TXT, S](
            img_tokens.copy(), txt_tokens.copy(), base, blocks, lora_dev, mv,
            f_scale.copy(), f_shift.copy(), rope[0], rope[1],
            D, F, IN_CH, TEXT_IN, OUT_CH, EPS, ctx,
        )

        var d_loss = List[Float32]()
        var nout = len(fwd.out)
        for i in range(nout):
            var diff = fwd.out[i] - target_packed[i]
            loss_sum += Float64(diff) * Float64(diff)
            d_loss.append(inv_n * diff)

        var grads_b = ernie_stack_lora_backward_resident_device[H, Dh, N_IMG, N_TXT, S](
            d_loss, img_tokens.copy(), txt_tokens.copy(), base, blocks, lora_dev, mv,
            f_scale.copy(), f_shift.copy(), rope[0], rope[1], fwd,
            D, F, IN_CH, TEXT_IN, OUT_CH, EPS, ctx,
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
    print("[loss] mojo =", loss, " (ref output.loss_for_backward =", REF_LOSS, ")")

    # ── raw grad stats BEFORE adamw (the real gate; B=0 invariant ⇒ d_A ~ 0) ──
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

    # ── AdamW with lr=0 (warmup step) : adapter_after == adapter_before ──
    var grads = ErnieLoraGrads(
        acc_a^, acc_b^, List[Float32](), List[Float32](),
        _zeros(6 * D), _zeros(D), _zeros(D), List[Float32](), nonfinite_total,
    )
    ernie_lora_adamw_step(lora, grads, 1, LR, ctx, BETA1, BETA2, ADAM_EPS, WEIGHT_DECAY)
    print("[adamw] applied lr =", LR, " (no-op: adapter_after == adapter_before)")

    if nonfinite_total != 0:
        raise Error("Ernie grad/update replay produced nonfinite LoRA grads")
    print("ERNIE TRAIN REF GRAD/UPDATE REPLAY DONE (main loop owns the bar)")
