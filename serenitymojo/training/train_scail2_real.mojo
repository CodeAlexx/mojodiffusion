# serenitymojo/training/train_scail2_real.mojo
#
# CHUNK 4 — the RUNNABLE SCAIL-2 i2v LoRA TRAINING LOOP.
#
# Trains a LoRA on the REAL 14B (transformer_fp8, streamed one block at a time)
# on ONE real staged replacement sample and DEMONSTRATES that the flow-match MSE
# loss DROPS over steps (overfit-a-single-sample gate = the "loss DROPS" half of
# the L2P verdict; the "sample SHIFTS" half via scail2_unipc is a FOLLOW-UP).
#
# This generalizes parity/scail2_train_real_smoke.mojo (which did ONE step at
# S=144, DENSE sdpa) into a multi-step, config-first loop that runs FLASH=True so
# a real-resolution sequence fits, mirroring the wan22 template
# training/train_wan22_real.mojo::_run_wan21_i2v_train per-step structure.
#
# GEOMETRY (crop, S<=16,384). The production sample is S=39,872 (65f / 896x512),
# too big for a single FLASH block on 16 GB (verified single-block max S=16,384).
# We CROP the real staged latents to FT=1 (first frame) at FULL SPATIAL grid
# GH=32, GW=56 (H_LAT=64, W_LAT=112) => S = (1+FT)*GH*GW + FT*(GH//2)*(GW//2)
#             = 2*32*56 + 1*16*28 = 3584 + 448 = 4032.
# 4032 <= 16,384, so FLASH fwd+bwd fits with headroom; the DENSE path's [1,H,S,S]
# score tensor at S=4032 is 40*4032^2*4 = 2.6 GB (would also fit here), but the
# loop runs FLASH=True per the CHUNK contract so the SAME path scales to real S.
#
# PER STEP (mirrors _run_wan21_i2v_train):
#   sample t (schedule.sample_timestep_logit_normal, qwen-shift) -> fresh noise ->
#   x_t=(1-t)*x0 + t*noise, target=noise-x0 on the VIDEO latent -> 20-ch i2v marker
#   packing (0.0 noised video, 1.0 ref/pose) -> frozen embed -> per-step time embed
#   -> STREAMING stack fwd (FLASH=True) -> _head_video -> MSE on the video region ->
#   head backward -> STREAMING stack backward (FLASH=True) -> global grad clip
#   (max_grad_norm) -> scail2_lora_adamw_step. Logs step,t,loss,grad_norm,sec.
#
# FROZEN ONCE: text context, image (CLIP) context, RoPE tables, ref/pose 20-ch
# packing, ref/driving masks, head weights. Per step: only the noised-video embed
# and the time embedding (timestep = t*1000 varies) are rebuilt.
#
# LoRA init = PEFT (A~randn*0.01, B=0) via build_scail2_lora_set, so step 0 is the
# base model and the LoRA learns from zero.
#
# Build (BINARY — FLASH needs cuDNN-shim + CUDA linkage; JIT can't cuInit):
#   rm -f serenitymojo.mojopkg
#   pixi run mojo build --optimization-level 2 -I . \
#     -Xlinker -lm -Xlinker -ldl \
#     -Xlinker -L"$CONDA_PREFIX/targets/x86_64-linux/lib/stubs" -Xlinker -lcuda \
#     -Xlinker -L"$CONDA_PREFIX/lib" -Xlinker -rpath-link -Xlinker "$CONDA_PREFIX/lib" \
#     -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa \
#     serenitymojo/training/train_scail2_real.mojo -o output/bin/train_scail2_real
#
# Run:
#   LD_LIBRARY_PATH=serenitymojo/ops/cshim/lib:.pixi/envs/default/lib \
#     output/bin/train_scail2_real [num_steps]

from std.builtin.type_aliases import MutExternalOrigin
from std.collections import List, Optional
from std.ffi import external_call
from std.gpu.host import DeviceContext
from std.math import sqrt
from std.memory import UnsafePointer, alloc
from std.time import perf_counter
from sys import argv

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.tensor import Tensor
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.random_torch import randn_torch
from serenitymojo.ops.tensor_algebra import concat, full_device, reshape, slice

from serenitymojo.models.scail2.scail2_streamed_dit import Scail2StreamedDiT
from serenitymojo.models.scail2.scail2_fp8_stream import Scail2A14BFP8Stream
from serenitymojo.models.scail2.scail2_rope import (
    Scail2SequencePlan, build_scail2_rope_tables,
)
from serenitymojo.models.scail2.scail2_stack_lora import (
    Scail2LoraSet, Scail2LoraGradSet, build_scail2_lora_set,
    scail2_total_adapters, scail2_lora_adamw_step, save_scail2_lora,
    save_scail2_lora_state,
)
from serenitymojo.models.scail2.scail2_stack_train_full import (
    scail2_stack_lora_forward_streamed, scail2_stack_lora_backward_streamed,
    scail2_head_video_backward,
)
from serenitymojo.training.train_config import TrainConfig
from serenitymojo.training.schedule import sample_timestep_logit_normal


comptime _CStr = UnsafePointer[UInt8, MutExternalOrigin]

# ── REAL SCAIL-2 i2v transformer dims (scail2_config.mojo::scail2_14b) ──
comptime DIM = 5120
comptime FFN = 13824
comptime HEADS = 40
comptime HEAD_DIM = 128
comptime NUM_LAYERS = 40
comptime OUT_DIM = 16
comptime TXT = 512
comptime CTXL = 512
comptime IMG = 257
comptime EPS = Float32(1.0e-6)
comptime CHANNELS = 16

# ── cropped geometry: FT=1 (first frame), FULL SPATIAL grid (GH=32, GW=56) ──
comptime FT = 1
comptime GH = 32
comptime GW = 56
comptime H_LAT = GH * 2         # 64
comptime W_LAT = GW * 2         # 112
comptime AR = 0
comptime S = (1 + FT) * GH * GW + FT * (GH // 2) * (GW // 2)   # 4032

comptime CACHE_DIR = "/home/alex/.serenity/models/checkpoints/SCAIL-2-Mojo/transformer_fp8"
comptime RUN_DIR = "/home/alex/.serenity/runs/scail2-gates/replacement"


# ── env-sync alloc (matches the smoke) ────────────────────────────────────────
def _cstr(s: String) -> _CStr:
    var n = s.byte_length()
    var buf = alloc[UInt8](n + 1)
    var src = s.as_bytes()
    for i in range(n):
        buf[i] = src[i]
    buf[n] = 0
    return _CStr(unsafe_from_address=Int(buf))


def _sync_alloc():
    _ = external_call["setenv", Int32](
        _cstr(String("MODULAR_DEVICE_CONTEXT_SYNC_MODE")),
        _cstr(String("true")), Int32(1),
    )


def _load_view(path: String, key: String, ctx: DeviceContext) raises -> Tensor:
    var st = ShardedSafeTensors.open(path)
    if key not in st.names():
        raise Error(String("missing key ") + key + String(" in ") + path)
    return Tensor.from_view(st.tensor_view(key), ctx)


def _crop4(x: Tensor, t: Int, h: Int, w: Int, ctx: DeviceContext) raises -> Tensor:
    var a = slice(x, 1, 0, t, ctx)
    a = slice(a, 2, 0, h, ctx)
    a = slice(a, 3, 0, w, ctx)
    return a^


def _channels20(latent16: Tensor, marker: Float32, t: Int, h: Int, w: Int,
                ctx: DeviceContext) raises -> Tensor:
    var marker4 = full_device([4, t, h, w], marker, STDtype.F32, ctx)
    return concat(0, ctx, latent16, marker4)


def _nonfinite(v: List[Float32]) -> Int:
    var bad = 0
    for i in range(len(v)):
        var x = v[i]
        if (x != x) or (x - x != Float32(0.0)):
            bad += 1
    return bad


# ── global-norm grad clip over the flat LoRA d_A/d_B (== torch clip_grad_norm_;
#    same formula as optim.clip_grads_by_global_norm but on the host grad set).
#    Returns the PRE-clip global L2 norm. ──────────────────────────────────────
def _clip_scail2(mut grads: Scail2LoraGradSet, max_norm: Float32) -> Float32:
    var ss = Float64(0.0)
    for i in range(len(grads.d_a)):
        for j in range(len(grads.d_a[i])):
            ss += Float64(grads.d_a[i][j]) * Float64(grads.d_a[i][j])
        for j in range(len(grads.d_b[i])):
            ss += Float64(grads.d_b[i][j]) * Float64(grads.d_b[i][j])
    var gn = Float32(sqrt(ss))
    if gn > max_norm and gn > Float32(0.0):
        var scl = max_norm / gn
        for i in range(len(grads.d_a)):
            for j in range(len(grads.d_a[i])):
                grads.d_a[i][j] = grads.d_a[i][j] * scl
            for j in range(len(grads.d_b[i])):
                grads.d_b[i][j] = grads.d_b[i][j] * scl
    return gn


# ── FIXED-(t,noise) forward-only loss eval (apples-to-apples overfit proof).
#    The training loss is measured at a RANDOM t each step (flow-match), so a bare
#    first-vs-last comparison is t-noisy. This evaluates the loss at ONE fixed
#    reference (t_ref, noise_ref, target_ref) so the base (fresh LoRA, B=0) vs the
#    trained LoRA can be compared cleanly. Forward+head only (no backward). ──────
def _eval_fixed_loss[
    H: Int, DH: Int, S: Int, TXT: Int, IMG: Int,
    FT: Int, GH: Int, GW: Int, AR: Int
](
    dit: Scail2StreamedDiT, mut stream: Scail2A14BFP8Stream, lora: Scail2LoraSet,
    sequence_ref: List[Float32], e0_ref: List[Float32],
    context_txt: List[Float32], context_img: List[Float32],
    cos: Tensor, sin: Tensor, time1_ref: Tensor, target_ref: List[Float32],
    num_layers: Int, dim: Int, ffn: Int, hd: Int, out_dim: Int,
    eps: Float32, ctx: DeviceContext,
) raises -> Float64:
    var fwd = scail2_stack_lora_forward_streamed[H, DH, S, TXT, IMG, True](
        sequence_ref.copy(), e0_ref.copy(), context_txt.copy(), context_img.copy(),
        cos, sin, stream, num_layers, lora, dim, ffn, hd, eps, ctx,
    )
    var img_t = Tensor.from_host(fwd.x_out.copy(), [1, S, dim], STDtype.F32, ctx)
    var head_out = dit._head_video[FT, GH, GW, AR, S](img_t, time1_ref, STDtype.BF16, ctx)
    var head_h = cast_tensor(head_out, STDtype.F32, ctx).to_host(ctx)
    var loss = Float64(0.0)
    for i in range(len(head_h)):
        var e = head_h[i] - target_ref[i]
        loss += Float64(e) * Float64(e)
    return loss / Float64(len(head_h))


# ── config-first descriptor (arch + recipe read from TrainConfig; no hardcode) ─
def _scail2_train_config() -> TrainConfig:
    var cfg = TrainConfig.default()
    cfg.name = String("scail2")
    cfg.checkpoint = String(CACHE_DIR)
    cfg.d_model = DIM
    cfg.n_heads = HEADS
    cfg.head_dim = HEAD_DIM
    cfg.num_double = NUM_LAYERS
    cfg.num_single = 0
    cfg.mlp_hidden = FFN
    cfg.out_channels = OUT_DIM
    cfg.lora_rank = 8
    cfg.lora_alpha = Float32(8.0)
    cfg.lr = Float32(1.0e-3)
    cfg.timestep_shift = Float32(1.0)
    cfg.max_grad_norm = Float32(1.0)
    cfg.beta1 = Float32(0.9)
    cfg.beta2 = Float32(0.999)
    cfg.eps = Float32(1.0e-8)
    cfg.weight_decay = Float32(0.01)
    cfg.max_steps = 80
    cfg.save_every = 40
    cfg.seed = UInt64(42)
    cfg.output_model_destination = String(RUN_DIR) + String("/scail2_lora_real.safetensors")
    return cfg^


def main() raises:
    _sync_alloc()
    var ctx = DeviceContext()

    var cfg = _scail2_train_config()
    # optional argv override of the step count (quick timing probes / gate reruns)
    var steps = cfg.max_steps
    var args = argv()
    if len(args) > 1:
        steps = atol(String(args[1]))
    var lr = cfg.lr
    var rank = cfg.lora_rank
    var alpha = cfg.lora_alpha
    var max_grad_norm = cfg.max_grad_norm
    var beta1 = cfg.beta1
    var beta2 = cfg.beta2
    var opt_eps = cfg.eps
    var wd = cfg.weight_decay
    var shift = cfg.timestep_shift
    var seed = cfg.seed
    var save_every = cfg.save_every

    print("==== SCAIL-2 i2v LoRA TRAINER (CHUNK 4) — REAL 14B, FLASH, overfit gate ====")
    print("arch: dim=", DIM, " ffn=", FFN, " blocks=", NUM_LAYERS, " heads=", HEADS,
          " head_dim=", HEAD_DIM)
    print("geometry(crop): FT=", FT, " GH=", GH, " GW=", GW, " -> S=", S,
          " (<=16384; FLASH=True)")
    print("lora: rank=", rank, " alpha=", alpha, " (12/block: 8 attn + ffn.0/2 + img_k/v);",
          " PEFT init A~randn, B=0")
    print("optim: lr=", lr, " max_grad_norm=", max_grad_norm, " betas=(", beta1, ",",
          beta2, ") wd=", wd, " eps=", opt_eps)
    print("timestep: logit-normal + qwen-shift(", shift, ")  steps=", steps, " seed=", seed)
    print("ckpt:", CACHE_DIR)
    print("")

    # ── frozen shared weights + head/embed/conditioning host ──
    var dit = Scail2StreamedDiT.open(CACHE_DIR, ctx)
    var stream = Scail2A14BFP8Stream.open(CACHE_DIR)

    # ── real staged latents (cropped to the reduced grid) ──
    var ref5 = _load_view(RUN_DIR + String("/reference.safetensors"),
                          String("reference_latent"), ctx)
    var ref4 = reshape(ref5, [CHANNELS, 1, 64, 112], ctx)
    var ref_crop = _crop4(ref4, 1, H_LAT, W_LAT, ctx)              # [16,1,64,112]

    var pose5 = _load_view(RUN_DIR + String("/pose.safetensors"),
                           String("pose_latent"), ctx)
    var pose4 = reshape(pose5, [CHANNELS, 17, 32, 56], ctx)
    var pose_crop = _crop4(pose4, FT, H_LAT // 2, W_LAT // 2, ctx) # [16,1,32,56]

    var vid4 = _load_view(RUN_DIR + String("/latent_replacement_1step.safetensors"),
                          String("latent"), ctx)                   # [16,17,64,112]
    var x0 = _crop4(vid4, FT, H_LAT, W_LAT, ctx)                   # [16,1,64,112]
    var x0_h = x0.to_host(ctx)

    var refm = _load_view(RUN_DIR + String("/stage.safetensors"),
                          String("ref_masks"), ctx)
    var ref_masks = _crop4(refm, FT + 1, H_LAT, W_LAT, ctx)        # [28,2,64,112]
    var drvm = _load_view(RUN_DIR + String("/stage.safetensors"),
                          String("driving_masks"), ctx)
    var driving_masks = _crop4(drvm, FT, H_LAT // 2, W_LAT // 2, ctx)  # [28,1,32,56]

    # text (pos) + clip conditioning (real) — FROZEN
    var pos = _load_view(RUN_DIR + String("/text.safetensors"),
                         String("pos_embed"), ctx)
    var pos2 = reshape(pos, [TXT, 4096], ctx)
    var clip = _load_view(RUN_DIR + String("/clip.safetensors"),
                          String("clip_context"), ctx)

    # ── frozen ref/pose 20-ch marker packing (1.0 = ref / pose) built ONCE ──
    var reference20 = _channels20(ref_crop^, 1.0, 1, H_LAT, W_LAT, ctx)
    var pose20 = _channels20(pose_crop^, 1.0, FT, H_LAT // 2, W_LAT // 2, ctx)

    # ── frozen conditioning built ONCE (text, image, rope) ──
    var text = dit._text[TXT, CTXL](pos2, STDtype.BF16, ctx)
    var context_txt = cast_tensor(text, STDtype.F32, ctx).to_host(ctx)   # [TXT*dim]
    var image = dit._image[IMG](clip, ctx)
    var context_img = cast_tensor(image, STDtype.F32, ctx).to_host(ctx)  # [IMG*dim]

    var plan = Scail2SequencePlan(
        video_t=FT, grid_h=GH, grid_w=GW,
        additional_ref_count=AR, replace_flag=True,
    )
    plan.validate()
    if plan.sequence_length() != S:
        raise Error("SCAIL-2 reduced sequence length mismatch")
    var rope = build_scail2_rope_tables(plan, ctx, STDtype.F32)

    # ── frozen head weights (modulation + head linear), fetched ONCE ──
    var head_w = dit._w(String("head.head.weight")).clone(ctx)
    var head_mod = dit._w(String("head.modulation")).clone(ctx)

    # ── LoRA (PEFT init: A~randn*0.01, B=0 -> step 0 == base model) ──
    var lora = build_scail2_lora_set(NUM_LAYERS, DIM, FFN, rank, alpha)
    var n_ad = scail2_total_adapters(lora)
    print("[lora] adapters:", n_ad, " (", NUM_LAYERS, " blocks x 12)")
    print("")

    # ── FIXED reference (t_ref, noise_ref) for the clean before/after overfit
    #    proof (independent of the per-step random-t training loss) ──
    var t_ref = Float32(0.5)
    var noise_ref = randn_torch([CHANNELS, FT, H_LAT, W_LAT], UInt64(777777), ctx)
    var noise_ref_h = noise_ref.to_host(ctx)
    var xt_ref_h = List[Float32]()
    var target_ref = List[Float32]()
    for i in range(len(x0_h)):
        xt_ref_h.append((Float32(1.0) - t_ref) * x0_h[i] + t_ref * noise_ref_h[i])
        target_ref.append(noise_ref_h[i] - x0_h[i])
    var xt_ref = Tensor.from_host(xt_ref_h^, [CHANNELS, FT, H_LAT, W_LAT], STDtype.F32, ctx)
    var video20_ref = _channels20(xt_ref^, 0.0, FT, H_LAT, W_LAT, ctx)
    var seq_ref_t = dit._embed_base_sequence[FT, GH, GW, S](
        video20_ref, reference20, pose20, ref_masks, driving_masks, ctx,
    )
    var sequence_ref = cast_tensor(seq_ref_t, STDtype.F32, ctx).to_host(ctx)
    var time_ref = dit._time(t_ref * Float32(1000.0), STDtype.BF16, ctx)
    var e0_ref = time_ref[0].to_host(ctx)

    var eval_before = _eval_fixed_loss[HEADS, HEAD_DIM, S, TXT, IMG, FT, GH, GW, AR](
        dit, stream, lora, sequence_ref, e0_ref, context_txt, context_img,
        rope[0], rope[1], time_ref[1], target_ref, NUM_LAYERS, DIM, FFN, HEAD_DIM,
        OUT_DIM, EPS, ctx,
    )
    print("[eval@fixed t=", t_ref, "] BEFORE training (base, LoRA B=0) loss =", eval_before)
    print("")

    var video_offset = (AR + 1) * GH * GW
    var video_rows = FT * GH * GW

    var first_loss = Float64(0.0)
    var last_loss = Float64(0.0)
    var min_loss = Float64(1.0e30)
    var have_first = False
    var gn_min = Float32(1.0e30)
    var gn_max = Float32(0.0)
    var all_finite = True
    var train_start = perf_counter()

    print("step   t         MSE               grad_norm      sec")
    for step in range(steps):
        var t_step = perf_counter()

        # ── sample timestep + build the flow-match target on the VIDEO latent ──
        var t = sample_timestep_logit_normal(
            seed * UInt64(1000003) + UInt64(step) * UInt64(101) + 7, shift
        )
        var timestep = t * Float32(1000.0)
        var noise = randn_torch(
            [CHANNELS, FT, H_LAT, W_LAT],
            seed * UInt64(2000003) + UInt64(step) * UInt64(104729) + 3, ctx,
        )
        var noise_h = noise.to_host(ctx)
        var xt_h = List[Float32]()
        var target_h = List[Float32]()
        for i in range(len(x0_h)):
            xt_h.append((Float32(1.0) - t) * x0_h[i] + t * noise_h[i])
            target_h.append(noise_h[i] - x0_h[i])
        var xt = Tensor.from_host(xt_h^, [CHANNELS, FT, H_LAT, W_LAT], STDtype.F32, ctx)

        # ── 20-ch marker packing for the noised video (marker 0.0) ──
        var video20 = _channels20(xt^, 0.0, FT, H_LAT, W_LAT, ctx)

        # ── frozen embed (ref/pose/masks reused; only video20 changes) ──
        var seq_t = dit._embed_base_sequence[FT, GH, GW, S](
            video20, reference20, pose20, ref_masks, driving_masks, ctx,
        )
        var sequence = cast_tensor(seq_t, STDtype.F32, ctx).to_host(ctx)     # [S*dim]

        # ── per-step time embedding (timestep = t*1000 varies) ──
        var time = dit._time(timestep, STDtype.BF16, ctx)
        var e0_host = time[0].to_host(ctx)                                   # [6*dim]

        # ── STREAMING stack forward (FLASH=True) ──
        var fwd = scail2_stack_lora_forward_streamed[HEADS, HEAD_DIM, S, TXT, IMG, True](
            sequence^, e0_host, context_txt.copy(), context_img.copy(),
            rope[0], rope[1], stream, NUM_LAYERS, lora, DIM, FFN, HEAD_DIM, EPS, ctx,
        )

        # ── head forward (video region -> velocity) ──
        var img_t = Tensor.from_host(fwd.x_out.copy(), [1, S, DIM], STDtype.F32, ctx)
        var head_out = dit._head_video[FT, GH, GW, AR, S](
            img_t, time[1], STDtype.BF16, ctx,
        )
        var head_h = cast_tensor(head_out, STDtype.F32, ctx).to_host(ctx)

        # ── flow-match MSE loss on the video region + d(head) ──
        var N = len(head_h)
        var loss = Float64(0.0)
        var d_head_h = List[Float32]()
        var inv_n = Float32(2.0) / Float32(N)
        for i in range(N):
            var e = head_h[i] - target_h[i]
            loss += Float64(e) * Float64(e)
            d_head_h.append(inv_n * e)
        loss /= Float64(N)

        var d_head = Tensor.from_host(d_head_h^, [OUT_DIM, FT, GH * 2, GW * 2], STDtype.F32, ctx)

        # ── head backward -> d_seq (scatters into the video rows) ──
        var video_region = List[Float32]()
        for r in range(video_rows):
            var srcb = (video_offset + r) * DIM
            for d in range(DIM):
                video_region.append(fwd.x_out[srcb + d])
        var d_seq = scail2_head_video_backward[FT, GH, GW, AR, S](
            d_head, video_region, time[1], head_mod, head_w, OUT_DIM, DIM, EPS, ctx,
        )

        # ── STREAMING stack backward (FLASH=True) ──
        var bwd = scail2_stack_lora_backward_streamed[HEADS, HEAD_DIM, S, TXT, IMG, True](
            d_seq^, e0_host, context_txt.copy(), context_img.copy(),
            rope[0], rope[1], stream, NUM_LAYERS, lora, fwd, DIM, FFN, HEAD_DIM, EPS, ctx,
        )

        # ── grad clip (global norm) + AdamW step ──
        var gn = _clip_scail2(bwd.grads, max_grad_norm)
        scail2_lora_adamw_step(lora, bwd.grads, step + 1, lr, ctx,
                               beta1, beta2, opt_eps, wd)

        var secs = perf_counter() - t_step

        # ── bookkeeping ──
        var loss_finite = (loss == loss) and (loss - loss == Float64(0.0))
        var gn_finite = (gn == gn) and (gn - gn == Float32(0.0))
        if (not loss_finite) or (not gn_finite) or (bwd.grads.nonfinite_lora_grads != 0):
            all_finite = False
        if not have_first:
            first_loss = loss
            have_first = True
        last_loss = loss
        if loss < min_loss:
            min_loss = loss
        if gn < gn_min:
            gn_min = gn
        if gn > gn_max:
            gn_max = gn

        print(step, "  ", t, "  ", loss, "  ", gn, "  ", secs)
        if bwd.grads.nonfinite_lora_grads != 0:
            print("  !! nonfinite lora grads =", bwd.grads.nonfinite_lora_grads)

        # ── periodic save ──
        if save_every > 0 and (step + 1) % save_every == 0:
            var sp = String(RUN_DIR) + String("/scail2_lora_real_step") \
                + String(step + 1) + String(".safetensors")
            var npairs_ck = save_scail2_lora(lora, sp, ctx)
            print("  [save] step", step + 1, "->", npairs_ck, "PEFT pairs", sp)

    # ── eval AFTER training on the SAME fixed reference (clean overfit proof) ──
    var eval_after = _eval_fixed_loss[HEADS, HEAD_DIM, S, TXT, IMG, FT, GH, GW, AR](
        dit, stream, lora, sequence_ref, e0_ref, context_txt, context_img,
        rope[0], rope[1], time_ref[1], target_ref, NUM_LAYERS, DIM, FFN, HEAD_DIM,
        OUT_DIM, EPS, ctx,
    )

    # ── final save (PEFT + resume state) ──
    var out_path = cfg.output_model_destination
    var npairs = save_scail2_lora(lora, out_path, ctx)
    var meta = List[Float32]()
    meta.append(Float32(steps))
    meta.append(Float32(Int(seed)))
    var state_path = out_path + String(".state")
    var nstate = save_scail2_lora_state(lora, state_path, ctx, meta^)

    # ── reload check (open the saved file, confirm an expected key) ──
    var reload_ok = False
    var example_key = String("diffusion_model.blocks.0.self_attn.q.lora_A.weight")
    var n_saved_keys = 0
    try:
        var st = ShardedSafeTensors.open(out_path)
        n_saved_keys = len(st.names())
        reload_ok = example_key in st.names()
    except e:
        print("  !! reload failed:", e)

    var pct_drop = Float64(0.0)
    if first_loss > Float64(0.0):
        pct_drop = (first_loss - last_loss) / first_loss * Float64(100.0)
    var eval_pct = Float64(0.0)
    if eval_before > Float64(0.0):
        eval_pct = (eval_before - eval_after) / eval_before * Float64(100.0)

    print("")
    print("[save] wrote", npairs, "PEFT pairs ->", out_path)
    print("[save] wrote", nstate, "state tensors ->", state_path)
    print("[reload] saved keys =", n_saved_keys, "  example key present(",
          example_key, ") =", reload_ok)
    print("")
    print("==== FIXED-t OVERFIT PROOF (same t/noise/sample, base vs trained) ====")
    print("  eval@t=", t_ref, " BEFORE (base) =", eval_before)
    print("  eval@t=", t_ref, " AFTER (trained)=", eval_after)
    print("  eval pct drop =", eval_pct)
    print("")
    print("==== RAW TRAINING LOSS (random t per step; t-noisy) ====")
    print("  steps        =", steps)
    print("  first MSE    =", first_loss)
    print("  last  MSE    =", last_loss)
    print("  min   MSE    =", min_loss)
    print("  first->last pct =", pct_drop)
    print("  grad_norm    = [", gn_min, ",", gn_max, "]")
    print("  total sec    =", perf_counter() - train_start)
    print("")

    # PRIMARY gate: the fixed-t eval loss dropped (t-noise-free). Secondary: the
    # raw min dropped below the base fixed-t eval too.
    var eval_dropped = eval_after < eval_before
    var ok = all_finite and (gn_max > Float32(0.0)) and eval_dropped and reload_ok
    if ok:
        print("VERDICT: PASS -- fixed-t eval loss DROPPED (", eval_before, "->",
              eval_after, ", pct=", eval_pct, "); all steps finite; grad_norm>0;",
              "LoRA saved+reloadable.")
        print("  (the 'sample SHIFTS' half via scail2_unipc is the FOLLOW-UP.)")
    else:
        print("VERDICT: FAIL -- see the metrics above (finite=", all_finite,
              " eval_dropped=", eval_dropped, " gn_max=", gn_max, " reload_ok=",
              reload_ok, ")")
