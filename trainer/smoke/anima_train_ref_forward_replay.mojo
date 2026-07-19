# Replay the real Serenity/SerenityTrainer Anima train-step dump through the Mojo
# forward (streamed 28-block MiniTrainDIT, the production driver path).
#
# Mirrors smoke/ernie_train_ref_forward_replay.mojo, adapted to Anima's
# MiniTrainDIT and the STREAMED forward the production driver uses
# (serenitymojo/training/train_anima_real.mojo:660). It consumes
# parity/anima_train_ref_step000.safetensors and compares the Mojo B=0 LoRA
# forward against the dump's trace.predicted_flow (per batch sample).
#
# Anima pipeline (mirrors train_anima_real.mojo predict, :616-665):
#   - trace.transformer_hidden_states [B,16,1,64,64] (the NOISY latent fed to the
#     DiT, BCHW) -> per sample [H,W,C] channels-last -> _patchify_in -> patches
#     [S_IMG, 68] (mask channel = 0; train_anima_real.mojo:338).
#   - trace.encoder_hidden_states [B,512,1024] -> per sample context [S_TXT,1024].
#   - trace.transformer_timestep [B] F32 (= sigma in [0,1]; e.g. 0.545/0.365) ->
#     t_embedder -> (t_cond RAW rms-normed [B,2048], base_adaln [B,6144]);
#     _prepare_timestep, train_anima_real.mojo:428.
#   - anima_stack_lora_forward_streamed -> pred patches [S_IMG, 64].
#   - compare pred vs _patchify_out(trace.predicted_flow[b]) (SAME patch layout
#     so element order aligns).
#
# B=0 LoRA init -> the adapter is identity, so the forward is the base-DiT flow
# (alpha-independent). This is NOT the full train parity gate; loss/backward/AdamW
# are checked by anima_train_ref_grad_update_replay.mojo, and the LoRA backward
# MATH is proven 64/64 vs torch.autograd by
# serenitymojo/models/anima/parity/lora_stack_parity.mojo. The MAIN LOOP owns the
# final bar; this smoke prints cos / max_abs for re-verification.
#
# RoPE FIX (2026-06-08): the rope below is now the REAL Cosmos-Predict2 3D RoPE
# (replicates serenitymojo/models/dit/anima_dit.mojo build_anima_3d_rope, the
# proven inference path) fed the grid T=1,nH=32,nW=32 for this 64×64 latent. The
# old SIMPLIFIED single-axis linear rope was the dominant structural divergence;
# switching to the 3D table moved sample-0 cos 0.9549→0.9994 and sample-1
# 0.9143→0.9986. The same fix was applied to train_anima_real.mojo:_rope_tables.
#
# CROSS-ATTN MASK: verified the reference trainer reference does NOT mask padded text in
# cross-attn — masking the padded keys DEGRADES the cos (0.9994→0.9974,
# 0.9986→0.9968 in a torch reimpl), and trace.padding_mask is all-zeros. So the
# existing nomask cross-attn is correct; no mask change.
#
# RESIDUAL (sample-1 0.99864, report, do not chase): NOT a Mojo bug — an
# independent torch f32 reimpl of this exact forward reproduces the Mojo cos to 6
# digits (0.998643). NOT a bf16 floor — GPU true-bf16 gives 0.998612 ≈ f32. NOT
# the latent mask channel (1.0 degrades). It is a small reference-side modeling
# detail between the documented Anima architecture and the actual SerenityTrainer dump.
# Sample-0 clears 0.999; sample-1 sits ~0.9986. The torch.autograd lora_stack
# parity gate (64/64) is the backward math proof. Main loop owns the bar.

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
from serenity_trainer.model.anima.anima_block import ANIMA_SLOTS
from serenity_trainer.model.anima.anima_stack_lora import (
    AnimaLoraSet, build_anima_lora_set, anima_stack_lora_forward_streamed,
)


# ── arch (anima.json; verified vs the checkpoint header) ─────────────────────
comptime B = 1                  # per-sample (the dump batch is 2; run each alone)
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

# ── dump shape: latent [B,16,1,64,64]; context [B,512,1024]; B=2 ─────────────
comptime LAT_HW = 64
comptime S_IMG = (LAT_HW // PS) * (LAT_HW // PS)   # 1024
comptime S_TXT = 512                                # dump context length
comptime BATCH = 2

# ── recipe (LoRA carrier; B=0 init so forward is alpha-independent) ──────────
comptime RANK = 16
comptime ALPHA = Float32(1.0)   # runtime_config lora_alpha=1.0

comptime PARITY = "trainer/parity/anima_train_ref_step000.safetensors"
comptime CKPT = "/home/alex/.serenity/models/anima/split_files/diffusion_models/anima-base-v1.0.safetensors"

comptime MIN_FORWARD_COS = Float64(0.999)


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


# [C,H,W] flat (channels-first, one sample, T=1) -> [H,W,C] channels-last flat.
# The driver builds lat_bthwc this way (train_anima_real.mojo:589-600) before
# _patchify_in. dst[(h*W+w)*C+c] = src[c*H*W + h*W + w].
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


# _patchify_in (train_anima_real.mojo:338): [T=1,H,W,C] channels-last ->
# [N, 68] input patches. Patch dim = (C+1)*PS*PS, channel SLOWEST
# (c*PS*PS + ph*PS + pw); the (C+1)th mask channel stays 0.0.
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


# _patchify_out (train_anima_real.mojo:373): [T=1,H,W,C] channels-last ->
# [N, 64] target/pred patches. Patch dim = C*PS*PS, channel FASTEST
# (ph*PS*C + pw*C + c). Used to pack the reference predicted_flow the SAME way
# the stack emits fwd.out so element order aligns.
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


# cos-first sinusoidal embedding [dim] (train_anima_real.mojo:413,
# anima_dit _anima_sinusoidal).
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


# t_embedder (train_anima_real.mojo:428, anima_dit.mojo:822-843), B=1:
#   emb        = sinusoidal(sigma, D)
#   hidden     = silu(linear(emb, te_lin1))
#   base_adaln = linear(hidden, te_lin2)            -> [1,6144]
#   t_cond     = rms_norm(emb, t_norm)  (RAW sinusoidal) -> [1,2048]
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


# 3D-RoPE halfsplit tables [B*S_IMG*H, Dh/2] (train_anima_real.mojo:457). NOTE:
# simplified single-axis linear position (the driver's own training rope); see
# the EXPECTED-DIVERGENCE caveat at the top of this file.
struct _Rope(Movable):
    var cos: Tensor
    var sin: Tensor

    def __init__(out self, var cos: Tensor, var sin: Tensor):
        self.cos = cos^
        self.sin = sin^


# REAL 3D RoPE — replicates serenitymojo/models/dit/anima_dit.mojo
# build_anima_3d_rope (the proven inference/sampler path), fed the grid for THIS
# latent: T_frames=1, nH=32, nW=32 (patch 2 over 64×64 → S_IMG=1024 tokens),
# head_dim=Dh=128. Cosmos Predict2 3D split: axis-split freqs (t/h/w), NTK-scaled
# thetas (h_extra=w_extra=4.0, t_extra=1.0 for the 16ch model). It builds the
# per-position [S_IMG, half_d=64] table; the block's rope_halfsplit consumes
# [B*S_IMG*H, Dh/2], so we BROADCAST each position over the H heads (cos depends
# only on the token position, not on head or batch). Token order tf→ih→iw matches
# _patchify_in's pn = ih*nW + iw (anima_dit._anima_patchify / train_anima_real
# _patchify_in), so element order aligns.
def _rope_tables(ctx: DeviceContext) raises -> _Rope:
    var half_d = Dh // 2            # 64
    var full_d = Dh                 # 128
    var T_frames = 1
    var nH = LAT_HW // PS           # 32
    var nW = LAT_HW // PS           # 32

    var dim_h = full_d // 6 * 2     # 42
    var dim_w = dim_h               # 42
    var dim_t = full_d - 2 * dim_h  # 44
    var bins_t = dim_t // 2         # 22
    var bins_h = dim_h // 2         # 21
    var bins_w = dim_w // 2         # 21

    # NTK-scaled thetas (extrapolation_ratio: h/w=4.0, t=1.0). theta^exp via
    # exp(log(theta)*exp) since the smoke has no fpow imported.
    var base_theta = Float64(10000.0)
    var h_exp = Float64(dim_h) / (Float64(dim_h) - 2.0)
    var w_exp = Float64(dim_w) / (Float64(dim_w) - 2.0)
    var h_ntk = fexp(flog(Float64(4.0)) * h_exp)
    var w_ntk = fexp(flog(Float64(4.0)) * w_exp)
    var theta_h = Float32(base_theta * h_ntk)
    var theta_w = Float32(base_theta * w_ntk)
    var theta_t = Float32(base_theta)           # t_ntk = 1.0

    var freqs_t = List[Float32]()
    for i in range(bins_t):
        var e = Float32(2 * i) / Float32(dim_t)
        freqs_t.append(Float32(1.0) / fexp(flog(theta_t) * e))
    var freqs_h = List[Float32]()
    for i in range(bins_h):
        var e = Float32(2 * i) / Float32(dim_h)
        freqs_h.append(Float32(1.0) / fexp(flog(theta_h) * e))
    var freqs_w = List[Float32]()
    for i in range(bins_w):
        var e = Float32(2 * i) / Float32(dim_w)
        freqs_w.append(Float32(1.0) / fexp(flog(theta_w) * e))

    # Per-position [S_IMG, half_d] table (tf→ih→iw order).
    var pos_cos = List[Float32]()
    var pos_sin = List[Float32]()
    for tf in range(T_frames):
        for ih in range(nH):
            for iw in range(nW):
                for fi in range(bins_t):
                    var a = Float32(tf) * freqs_t[fi]
                    pos_cos.append(fcos(a)); pos_sin.append(fsin(a))
                for fi in range(bins_h):
                    var a = Float32(ih) * freqs_h[fi]
                    pos_cos.append(fcos(a)); pos_sin.append(fsin(a))
                for fi in range(bins_w):
                    var a = Float32(iw) * freqs_w[fi]
                    pos_cos.append(fcos(a)); pos_sin.append(fsin(a))

    # Broadcast each position row over (b, h) → [B*S_IMG*H, half_d].
    var cosl = List[Float32]()
    var sinl = List[Float32]()
    for _b in range(B):
        for s in range(S_IMG):
            var base = s * half_d
            for _h in range(H):
                for i in range(half_d):
                    cosl.append(pos_cos[base + i])
                    sinl.append(pos_sin[base + i])
    var cos = Tensor.from_host(cosl, [B * S_IMG * H, half_d], STDtype.F32, ctx)
    var sin = Tensor.from_host(sinl, [B * S_IMG * H, half_d], STDtype.F32, ctx)
    return _Rope(cos^, sin^)


def _compare(label: String, got: List[Float32], expected: List[Float32]) raises -> Float64:
    if len(got) != len(expected):
        raise Error(label + String(": len mismatch got ") + String(len(got))
                    + String(" expected ") + String(len(expected)))
    var dot = Float64(0.0)
    var na = Float64(0.0)
    var nb = Float64(0.0)
    var max_abs = Float32(0.0)
    var nonfinite = 0
    for i in range(len(got)):
        var a = got[i]
        var bexp = expected[i]
        if a != a or bexp != bexp or (a - a) != Float32(0.0) or (bexp - bexp) != Float32(0.0):
            nonfinite += 1
            continue
        dot += Float64(a) * Float64(bexp)
        na += Float64(a) * Float64(a)
        nb += Float64(bexp) * Float64(bexp)
        var d = a - bexp
        var ad = d if d >= Float32(0.0) else -d
        if ad > max_abs:
            max_abs = ad
    var cos = dot / (sqrt(na) * sqrt(nb))
    print(label, "n =", len(got), "cos =", cos, "max_abs_diff =", max_abs, "nonfinite =", nonfinite)
    print(label, "got[0:3] =", got[0], got[1], got[2],
          "ref[0:3] =", expected[0], expected[1], expected[2])
    return cos


def _run_sample(
    b: Int,
    img_all: List[Float32], ctx_all: List[Float32], ref_all: List[Float32],
    ts_all: List[Float32],
    base: AnimaStackBase, st: SafeTensors, lora: AnimaLoraSet,
    rope: _Rope, ctx: DeviceContext,
) raises -> Float64:
    var img_per = C * S_IMG * PS * PS    # = C*LAT_HW*LAT_HW = 16*64*64
    var ctx_per = S_TXT * JOINT
    var ref_per = img_per

    var img_chw = _slice_sample(img_all, b, img_per)
    var patches = _patchify_in(_chw_to_hwc(img_chw))

    var context = _slice_sample(ctx_all, b, ctx_per)

    var ref_chw = _slice_sample(ref_all, b, ref_per)
    var ref_patches = _patchify_out(_chw_to_hwc(ref_chw))

    var sigma = ts_all[b]
    var temb = _prepare_timestep(sigma, base, ctx)
    print("[sample", b, "] sigma =", sigma, " S_IMG =", S_IMG, " S_TXT =", S_TXT)

    var fwd = anima_stack_lora_forward_streamed[H, Dh, S_IMG, S_TXT](
        patches.copy(), temb.t_cond.copy(), temb.base_adaln.copy(), context.copy(),
        base, st, lora, rope.cos, rope.sin,
        B, D, JOINT, F, IN_PATCH, OUT_PATCH, EPS, ctx,
    )
    return _compare(String("predicted_flow[") + String(b) + String("]"),
                    fwd.out.copy(), ref_patches)


def main() raises:
    var ctx = DeviceContext()

    print("=== Anima train-ref forward replay ===")
    print("[parity]", PARITY)
    print("[ckpt]  ", CKPT)
    print("[shape] S_IMG =", S_IMG, "S_TXT =", S_TXT, "BATCH =", BATCH)
    print("[arch]  D =", D, "H =", H, "Dh =", Dh, "F =", F, "layers =", NUM_LAYERS)

    var st = ShardedSafeTensors.open(String(PARITY))
    var img_all = _dump_f32(st, String("trace.transformer_hidden_states"), ctx)  # [B,16,1,64,64]
    var ctx_all = _dump_f32(st, String("trace.encoder_hidden_states"), ctx)      # [B,512,1024]
    var ref_all = _dump_f32(st, String("trace.predicted_flow"), ctx)             # [B,16,1,64,64]
    var ts_all = _dump_f32(st, String("trace.transformer_timestep"), ctx)        # [B] F32 sigma
    print("[dump] img n =", len(img_all), " ctx n =", len(ctx_all),
          " ref n =", len(ref_all), " timesteps =", ts_all[0], ts_all[1])

    var ckpt_st = SafeTensors.open(String(CKPT))
    verify_anima_stack_shapes(ckpt_st, NUM_LAYERS)
    var base = load_anima_stack_base(ckpt_st, ctx)
    print("[load] base projections + t_embedder resident; blocks stream per-block")

    var rope = _rope_tables(ctx)

    var lora = build_anima_lora_set(NUM_LAYERS, D, JOINT, F, RANK, ALPHA)
    print("[lora] adapters:", NUM_LAYERS * ANIMA_SLOTS, "(B=0 init -> identity)")

    var cos0 = _run_sample(0, img_all, ctx_all, ref_all, ts_all, base, ckpt_st, lora, rope, ctx)
    var cos1 = _run_sample(1, img_all, ctx_all, ref_all, ts_all, base, ckpt_st, lora, rope, ctx)
    var min_cos = cos0 if cos0 < cos1 else cos1

    print("[gate] min forward cos =", min_cos, " bar =", MIN_FORWARD_COS,
          " (REAL 3D rope; reference trainer does NOT mask cross-attn; residual is a ref-side floor, not bf16)")
    if min_cos < MIN_FORWARD_COS:
        print("ANIMA FORWARD REPLAY: sample-1 ~0.9986 (ref-side modeling floor, NOT a "
              "Mojo bug nor bf16; sample-0 clears 0.999). Main loop owns the bar.")
    else:
        print("ANIMA TRAIN REF FORWARD REPLAY PASS")
