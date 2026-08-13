# Real-weights Klein (FLUX.2-klein-base-9B) full forward smoke (Phase-2b-equivalent).
#
# GOAL: prove the ported Klein forward RUNS on the real 9B weights with FINITE
# output, the LoRA-B=0 (identity) overlay path, on FIXED seeded inputs — exactly the
# way zimage_real_forward.mojo verifies Z-Image. This is the precursor to the
# forward-parity gate (cos >= 0.999 vs diffusers Flux2Transformer2DModel) that the
# reference-generator + compare slices add on top of these same fixed inputs.
#
# ── CHECKPOINT FORMAT (the load-dir decision — IMPORTANT for the other slices) ──
# The BORROWED Klein forward (model/KleinModel.klein_inference_forward via
# Flux2LoRASpec.predict -> klein_training_forward) and the weight-assembly loaders
# (model/klein/weights.mojo: load_klein_stack_base / load_double_block_weights /
# load_single_block_weights / load_klein_step_mod_weights) consume the ORIGINAL
# (fused-qkv) BFL key names:
#     img_in.weight, txt_in.weight, time_in.{in,out}_layer.weight,
#     double_blocks.<i>.{img,txt}_attn.qkv.weight / .proj.weight /
#       .norm.{query,key}_norm.scale, double_blocks.<i>.{img,txt}_mlp.{0,2}.weight,
#     double_stream_modulation_{img,txt}.lin.weight, single_stream_modulation.lin.weight,
#     single_blocks.<i>.{linear1,linear2}.weight / .norm.{query,key}_norm.scale,
#     final_layer.linear.weight, final_layer.adaLN_modulation.1.weight.
#   (verified present, 201 keys, NO guidance_in.* -> guidance_embeds=False.)
#
# Flux2ModelLoader.Flux2Weights.load(dir) reads the DIFFUSERS key names
# (transformer_blocks.<i>.attn.to_q/to_k/to_v..., x_embedder.weight,
# time_guidance_embed.timestep_embedder.linear_1...) from the HF snapshot
# transformer/ dir — a DIFFERENT key layout that the borrowed forward's loaders do
# NOT read (the diffusers->original qkv FUSION is the model-unit loader's job and is
# NOT wired into make_flux2_lora_spec's piece assembly yet). So Flux2Weights.load
# canNOT drive make_flux2_lora_spec today; the ORIGINAL single-file checkpoint is
# the matching format. This mirrors serenitymojo's own real-weight gate
# (serenitymojo/models/klein/parity/klein_stack_real_smoke.mojo:35,83 opens
# flux-2-klein-base-9b.safetensors via SafeTensors.open) — the exact pattern the
# port's weights.mojo loaders were ported from.
#
#   ORIGINAL-format transformer (what we load):
#     /home/alex/.serenity/models/checkpoints/flux-2-klein-base-9b.safetensors
#   VAE (bn.running_mean / bn.running_var [128] F32, for scale_latents):
#     /home/alex/.serenity/models/vaes/flux2-vae.safetensors
#   DIFFUSERS HF snapshot transformer dir (what Flux2Weights.load would read; NOT
#   used here, recorded for the parity/reference-generator slice which runs diffusers):
#     ~/.cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-base-9B/
#       snapshots/32773329fbe7e81a90ef971740e8ba4b0364ecf3/transformer
#
# ── SPEC INPUT ASSEMBLY (how the checkpoint splits into make_flux2_lora_spec args) ─
#   base        = load_klein_stack_base(st, vec_silu, KDIM)            (img_in/txt_in/
#                   final_layer.linear + final adaLN shift/scale)
#   dbw[i]      = load_double_block_weights(st, i)   for i in 0..7
#   sbw[i]      = load_single_block_weights(st, i)   for i in 0..23
#   step_mod_w  = load_klein_step_mod_weights(st, KDIM)               (time_in MLP +
#                   double/single/final modulation lins; guidance_in absent -> None)
#   lora        = klein_lora_set_to_device(build_klein9b_lora_set(rank, alpha))
#                   -> 144 adapters (8*12 + 24*2), B=0 (kaiming A, zero B) => identity.
#   cos_t,sin_t = build_klein_rope_tables_port[N_IMG, NTXT, KH=32, KDh=128]
#                   (N_IMG = HL*WL must be a perfect square).
#   bn_inv_scale= KleinVAE._load_bn_inv_scale(vae) = 1/sqrt(running_var + 1e-4)  [128]
#   bn_mean     = KleinVAE._load_bn_mean(vae)      = running_mean                [128]
#   txt_tokens  = FIXED seeded [NTXT, KTXT_CH=12288] BF16 (synthetic stand-in for the
#                   Qwen3 concat hidden states; B=0 forward is finite for any input).
#   latent      = FIXED seeded [1, KLEIN_LATENT_CH=32, HL*2, WL*2] BF16 — the RAW VAE
#                   MEAN latent PRE-patchify/scale; predict() patchifies it to
#                   [1,128,HL,WL] (_patchify_packed) then batch-norm scales it
#                   (_bn_apply[True]) before noising. (KIN_CH=128 = 4*32.)
#   base_seed   = fixed; guidance_scale=1.0, guidance_embeds=False (klein-base-9B).
#
# vec_silu (for load_klein_stack_base's frozen final adaLN) is built from a fixed
# representative timestep; predict() rebuilds the PER-STEP modvecs from step_mod_w at
# the integer timestep it samples, and the borrowed forward uses base.final_shift/
# scale, so the exact vec_silu timestep only fixes the (overlay-neutral) final adaLN.
#
# RESOLUTION: HL=WL=4 (N_IMG=16 packed tokens) keeps the 9B forward tractable. The
# packed token grid is HL*WL (the diffusers image_seq_len at this latent size);
# raising to a real 256^2 image is a constants-only change for the parity slice.
#
# Run (MAIN LOOP, not this agent):
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#     pixi run mojo build -I . -I trainer/src -Xlinker -lm \
#     trainer/smoke/klein_real_forward.mojo -o /tmp/klein_rf && /tmp/klein_rf

from std.collections import List
from max.gpu.host import DeviceContext
from std.time import perf_counter_ns
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.random import randn
from serenitymojo.ops.loss_swiglu_backward import mse_backward
from serenitymojo.autograd import Tape

from serenity_trainer.model.klein.double_block import DoubleBlockWeights
from serenity_trainer.model.klein.single_block import SingleBlockWeights
from serenity_trainer.model.klein.weights import (
    load_double_block_weights, load_single_block_weights,
    load_klein_stack_base, build_klein_vec_silu, load_klein_step_mod_weights,
)
from serenity_trainer.model.klein.klein_stack_lora import klein_lora_set_to_device
from serenity_trainer.model.KleinModel import (
    build_klein_rope_tables_port, build_klein9b_lora_set,
    KDIM, KH, KDh, KTXT_CH, KNUM_DOUBLE, KNUM_SINGLE, KTIMESTEP_DIM,
)
from serenity_trainer.model.KleinVAE import (
    _load_bn_inv_scale, _load_bn_mean, KLEIN_LATENT_CH,
)
from serenity_trainer.modelSetup.Flux2LoRASetup import make_flux2_lora_spec
from serenity_trainer.util.config.TrainConfig import TrainConfig


# ── load dirs ────────────────────────────────────────────────────────────────
comptime KLEIN9B_PATH = "/home/alex/.serenity/models/checkpoints/flux-2-klein-base-9b.safetensors"
comptime KLEIN_VAE_PATH = "/home/alex/.serenity/models/vaes/flux2-vae.safetensors"

# ── PATCHIFIED latent dims: N_IMG = HL*WL packed tokens (must be a perfect square) ─
comptime HL = 4
comptime WL = 4
comptime NTXT = 8          # synthetic Qwen3 concat seq len
comptime LORA_RANK = 16    # klein9b.json lora_rank
comptime LORA_ALPHA = Float32(16.0)


def main() raises:
    var ctx = DeviceContext()
    var t_all0 = perf_counter_ns()

    print("=== Klein-9B (FLUX.2-klein-base-9B) REAL-WEIGHT forward smoke ===")
    print("  transformer:", KLEIN9B_PATH)
    print("  vae        :", KLEIN_VAE_PATH)
    print("  HL=", HL, " WL=", WL, " N_IMG=", HL * WL, " NTXT=", NTXT,
          " KDIM=", KDIM, " KTXT_CH=", KTXT_CH,
          " num_double=", KNUM_DOUBLE, " num_single=", KNUM_SINGLE)
    var t_setup0 = perf_counter_ns()

    # ── open the ORIGINAL-format single-file transformer checkpoint ──
    var st = SafeTensors.open(String(KLEIN9B_PATH))

    # ── frozen final-layer adaLN feature (vec_silu) at a fixed timestep ──
    #   predict() rebuilds the PER-STEP modvecs; this only fixes base.final_shift/scale.
    var ts = Tensor.from_host([Float32(0.5)], [1], STDtype.F32, ctx)
    var vec_silu = build_klein_vec_silu(st, ts, KTIMESTEP_DIM, KDIM, ctx)
    print("  vec_silu len =", len(vec_silu), " (expect", KDIM, ")")

    # ── base (input proj + final layer) + all block weights ──
    var base = load_klein_stack_base(st, vec_silu, KDIM, ctx)
    var dbw = List[DoubleBlockWeights]()
    for bi in range(KNUM_DOUBLE):
        dbw.append(load_double_block_weights(st, bi, ctx))
    var sbw = List[SingleBlockWeights]()
    for bi in range(KNUM_SINGLE):
        sbw.append(load_single_block_weights(st, bi, ctx))
    print("  loaded base +", len(dbw), "double +", len(sbw), "single block weights")

    # ── frozen timestep/modulation weights (guidance_in absent -> None) ──
    var step_mod_w = load_klein_step_mod_weights(st, KDIM, ctx)
    print("  step_mod_w loaded; has_guidance =", step_mod_w.has_guidance(),
          " (expect False for klein-base-9B)")

    # ── 144-adapter LoRA set, B=0 (identity overlay) -> forward == base ──
    var lora_set = build_klein9b_lora_set(LORA_RANK, LORA_ALPHA)
    var lora = klein_lora_set_to_device(lora_set, ctx)
    print("  LoRA adapters: double =", len(lora.dbl), " single =", len(lora.sgl),
          " (expect", KNUM_DOUBLE * 12, "+", KNUM_SINGLE * 2, "= 144, B=0)")

    # ── RoPE tables for the joint sequence [S*H, Dh/2] ──
    var rope_tup = build_klein_rope_tables_port[HL * WL, NTXT, KH, KDh](ctx, STDtype.BF16)
    # Tuple elements are move-only Tensor values; clone keeps BF16 storage while
    # producing owned tensors for Flux2LoRASpec.
    var cos_t = rope_tup[0].clone(ctx)
    var sin_t = rope_tup[1].clone(ctx)

    # ── VAE batch-norm stats for scale_latents ──
    var vae = ShardedSafeTensors.open(String(KLEIN_VAE_PATH))
    var bn_inv_scale = _load_bn_inv_scale(vae, ctx)   # 1/sqrt(running_var+1e-4) [128]
    var bn_mean = _load_bn_mean(vae, ctx)             # running_mean              [128]
    print("  bn stats loaded (inv_scale, mean) [128] F32")

    # ── FIXED seeded inputs (B=0 forward is finite for any input) ──
    #   latent = raw VAE mean [1,32,HL*2,WL*2] BF16 (pre-patchify/scale).
    var latent = randn([1, KLEIN_LATENT_CH, HL * 2, WL * 2], UInt64(101), STDtype.BF16, ctx)
    #   txt_tokens = synthetic Qwen3 concat hidden states [NTXT, 12288] BF16.
    var txt_tokens = randn([NTXT, KTXT_CH], UInt64(202), STDtype.BF16, ctx)

    # ── assemble the spec via make_flux2_lora_spec ──
    var spec = make_flux2_lora_spec[HL, WL, NTXT](
        base^, dbw^, sbw^, step_mod_w^, lora^,
        cos_t^, sin_t^, bn_inv_scale^, bn_mean^,
        txt_tokens^, latent^, UInt64(303),
        Float32(1.0),    # guidance_scale
        False,           # guidance_embeds = False (klein-base-9B)
    )
    var t_setup1 = perf_counter_ns()

    # ── run Flux2LoRASpec.predict (patchify -> scale -> noise -> modvecs ->
    #    klein_training_forward -> unpack -> unpatchify) ──
    #    (predict -> _forward_lora -> klein_training_forward, Flux2LoRASetup.mojo:321;
    #     the block math is identical to klein_inference_forward — only the tape differs.)
    var cfg = TrainConfig.adamw_lora_defaults()
    var tape = Tape()
    print("  running Flux2LoRASpec.predict ...")
    var t_pred0 = perf_counter_ns()
    var out = spec.predict(tape, cfg, 0, ctx)
    var t_pred1 = perf_counter_ns()

    # ── velocity stats: n / mean / var / nonfinite ──
    var v = out.predicted.to_host(ctx)
    var n = len(v)
    var s = Float32(0.0)
    var s2 = Float32(0.0)
    var nf = 0
    for i in range(n):
        var x = v[i]
        if x != x:
            nf += 1
        else:
            s += x
            s2 += x * x
    var mean = s / Float32(n)
    print("velocity: n =", n, " mean =", mean,
          " var =", s2 / Float32(n) - mean * mean, " nonfinite =", nf)
    print("KLEIN REAL FORWARD OK" if nf == 0 else "KLEIN REAL FORWARD HAS NONFINITE")

    # ── backward smoke: d_flow = d mean-MSE(predicted, target) / d predicted.
    # This exercises Flux2LoRASpec.backward_lora and the hand-chained Klein LoRA
    # backward path without applying an optimizer update.
    print("  running Flux2LoRASpec.backward_lora ...")
    var d_flow = mse_backward(out.predicted, out.target, ctx)
    var t_bwd0 = perf_counter_ns()
    var grads = spec.backward_lora(d_flow, ctx)
    var t_bwd1 = perf_counter_ns()

    var g_total = 0
    var g_nonfinite = 0
    var g_abs = Float32(0.0)
    for gi in range(len(grads.dbl_d_a)):
        for j in range(len(grads.dbl_d_a[gi])):
            var x = grads.dbl_d_a[gi][j]
            g_total += 1
            if x != x:
                g_nonfinite += 1
            else:
                g_abs += x if x >= Float32(0.0) else -x
    for gi in range(len(grads.dbl_d_b)):
        for j in range(len(grads.dbl_d_b[gi])):
            var x = grads.dbl_d_b[gi][j]
            g_total += 1
            if x != x:
                g_nonfinite += 1
            else:
                g_abs += x if x >= Float32(0.0) else -x
    for gi in range(len(grads.sgl_d_a)):
        for j in range(len(grads.sgl_d_a[gi])):
            var x = grads.sgl_d_a[gi][j]
            g_total += 1
            if x != x:
                g_nonfinite += 1
            else:
                g_abs += x if x >= Float32(0.0) else -x
    for gi in range(len(grads.sgl_d_b)):
        for j in range(len(grads.sgl_d_b[gi])):
            var x = grads.sgl_d_b[gi][j]
            g_total += 1
            if x != x:
                g_nonfinite += 1
            else:
                g_abs += x if x >= Float32(0.0) else -x

    print("grads: groups =", len(grads.dbl_d_a) + len(grads.dbl_d_b) + len(grads.sgl_d_a) + len(grads.sgl_d_b),
          " elems =", g_total, " abs_sum =", g_abs, " nonfinite =", g_nonfinite)
    print("KLEIN REAL BACKWARD OK" if g_total > 0 and g_nonfinite == 0 else "KLEIN REAL BACKWARD HAS NONFINITE")
    var t_all1 = perf_counter_ns()
    var setup_s = Float32(Float64(t_setup1 - t_setup0) / 1.0e9)
    var predict_s = Float32(Float64(t_pred1 - t_pred0) / 1.0e9)
    var backward_s = Float32(Float64(t_bwd1 - t_bwd0) / 1.0e9)
    var step_no_optim_s = predict_s + backward_s
    var total_s = Float32(Float64(t_all1 - t_all0) / 1.0e9)
    print("speed: setup_s =", setup_s,
          " predict_s =", predict_s,
          " backward_s =", backward_s,
          " step_no_optim_s =", step_no_optim_s,
          " total_s =", total_s)
