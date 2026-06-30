# models/krea2/train_krea2.mojo — Krea-2-Raw LoRA TRAINER (Phase 4a).
#
# The product train loop for krea2 LoRA: drives the giger cache through the
# flow-match objective and the STREAMING single-stream LoRA stack, stepping the
# LoRA AdamW. Mirrors the zimage/ideogram4 real-trainer template
# (cache → flow-noise → conditioning → stack fwd → MSE loss → stack bwd → AdamW →
# log loss + grad_norm), reusing the shared training/ pipeline.
#
# ── THE STREAMING REQUIREMENT (resolves the measured 24GB OOM) ────────────────
# The Phase-2 stack krea2_stack_lora_forward/backward hold ALL 28 blocks' bf16
# weights resident (Krea2StackWeights, ≈24GB) → OOM at real depth. The inference
# krea2_forward (krea2_dit.mojo:1304) STREAMS each block's weights H2D inside the
# loop and frees them at iteration end. This trainer uses the STREAMING stack
# variants (krea2_stack_lora_{forward,backward}_streamed): peak = one block's
# frozen weights (~868MB) + activations + the small resident LoRA set. The
# frozen CONDITIONING prefix (embedders + 4-layer text-fusion + final-layer) is
# also streamed/loaded once per step from the checkpoint.
#
# ── LT-PAD CHOICE (length-bucket pad + cuDNN flash-padmask) ───────────────────
# The MAX device pool keys each distinct-LT size-class and does NOT reuse larger
# blocks for smaller requests → ≥3 distinct LTs OOM regardless of order (measured:
# a 1-LT run does 12 steps; 3 distinct LTs OOM at step 2). FIX: pad EVERY sample
# to a common bucket text length LTMAX so all steps allocate the SAME LFULL =
# LTMAX + imglen size → one size-class, no fragmentation. The no-mask block would
# let the pad tokens corrupt the real ones, so the block runs the cuDNN
# FLASH-PADMASK SDPA arm: cuDNN masks the [real_len:LFULL] pad tail internally
# (NO materialized [1,H,L,L] mask — the old additive-mask path needed 4.5GB
# resident + 4.5GB bwd scores and OOM'd ~step 12; flash needs neither). cuDNN's
# padmask validates a contiguous PREFIX [0:real_len] and masks the TAIL, so the
# sequence is REORDERED to [TXT_real(0:lt) | IMG(lt:lt+imglen) | TXT_pad(tail)]
# with real_len = lt + imglen (krea2 text positions are all-zero so moving image
# before the pad changes no token's rotation — see _build_conditioning). ONE
# comptime arm (LFULL) for ALL samples. The giger cache (4 samples, 1024px,
# imglen=4096) has LT ∈ {458,558,627,647} (max 647) → LTMAX=768 (clean mult of
# 256) buckets all four into LFULL=4864. Masked-pad isolation is gated by
# parity/krea2_mask_pad_gate.mojo (real-token grads pad-length invariant; FLASH so
# value-tolerance cos>=0.999, not bit-exact — flash dQ is nondeterministic).
#
# ── autograd_v2 SEAM (Phase 4b) ───────────────────────────────────────────────
# KREA2_V2_GRAPH (comptime, DEFAULT FALSE) selects the backward path. False =
# the hand-chain krea2_stack_lora_backward_streamed (this file). Phase 4b adds
# the autograd_v2 engine arm under the True branch (the all-trainers-v2 mandate);
# the default-off path is byte-identical to the hand-chain.
#
# Run (ORCHESTRATOR runs the GPU smoke — long + heavy; not backgrounded):
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#   pixi run mojo build -I . -Xlinker -lm \
#     -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa \
#     serenitymojo/models/krea2/train_krea2.mojo -o /tmp/krea2_train && \
#   LD_LIBRARY_PATH=/home/alex/mojodiffusion/.pixi/envs/default/lib:serenitymojo/ops/cshim/lib:$HOME/.local/lib/python3.12/site-packages/nvidia/cudnn/lib \
#     /tmp/krea2_train <cache.safetensors> <steps>
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.gpu.host import DeviceContext
from std.collections import List, Optional
from std.math import sqrt
from std.memory import ArcPointer
from std.sys import argv
from std.time import perf_counter_ns


# ── per-phase timing helper (KREA2_PHASE_TIMING): sync, then ms since `t0` ────
def _phase_ms(name: String, t0: Int, ctx: DeviceContext) raises -> Int:
    """ctx.synchronize() to drain the phase's async work, print ms since t0, return
    the new t0. No-op cost when KREA2_PHASE_TIMING is False (the call sites are
    comptime-guarded so this never runs in production)."""
    ctx.synchronize()
    var now = Int(perf_counter_ns())
    print("  PHASE", name, "=", Float64(now - t0) / 1e6, "ms")
    return now

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.tensor_algebra import reshape, concat, slice, permute, zeros_device

# ── shared training pipeline (REUSE) ──────────────────────────────────────────
from serenitymojo.training.train_config import TrainConfig
from serenitymojo.training.train_step import LoraAdapter
from serenitymojo.training.schedule import (
    sample_timestep_logit_normal, flow_match_noise_target,
)
from serenitymojo.training.levers import levers_loss_grad
from serenitymojo.training.lora_adamw_plain_fused import fused_lora_adamw_plain_step
from serenitymojo.training.levers import (
    levers_optimizer_active, levers_optimizer_step_host,
    levers_optimizer_validate, LeversOptimizerState,
)
from serenitymojo.training.lora_save import NamedLora, save_lora_peft, save_lora_train_state, load_lora_train_state, load_lora_for_resume
from serenitymojo.io.ffi import sys_mkdirs, sys_remove
from serenitymojo.training.sample_prompt_config import (
    SamplePromptConfig, read_sample_prompt_config,
)
from serenitymojo.io.cap_cache import load_tensor_bin
from serenitymojo.ops.torch_bf16 import torch_f32_to_bf16_rne
from serenitymojo.models.vae.qwenimage_decoder import QwenImageVaeDecoder
from serenitymojo.sampling.krea2_sampler import (
    krea2_packed_seq_len, krea2_timesteps, krea2_cfg, krea2_euler_step,
)
from serenitymojo.image.png import save_png, ValueRange

# ── krea2 config + cache reader + LoRA set ────────────────────────────────────
from serenitymojo.models.krea2.config import krea2_raw
from serenitymojo.io.train_config_reader import read_model_config
from serenitymojo.models.krea2.krea2_cache_reader import (
    KreaTrainCache, KreaTrainSample, krea2_patchify, krea2_build_pos,
    KREA2_LATENT_CHANNELS, KREA2_TXT_LAYERS, KREA2_TXT_DIM,
)
from serenitymojo.models.klein.lora_adapter import make_lora_adapter
from serenitymojo.models.klein.lora_block import (
    LoraAdapterDevice, lora_adapter_to_device,
)
# LyCORIS carrier dispatch: LoKr/LoHa use the SAME streamed stack through
# materialized plain-LoRA carriers. DoRA/OFT use the direct streamed W_eff
# block/stack path so Krea2 never materializes their dense full-delta carriers.
from serenitymojo.training.train_config import (
    TRAIN_ADAPTER_ALGO_LORA, TRAIN_ADAPTER_ALGO_FULL,
    TRAIN_ADAPTER_ALGO_LOCON,
    TRAIN_ADAPTER_ALGO_LOHA, TRAIN_ADAPTER_ALGO_DORA,
    TRAIN_ADAPTER_ALGO_LOKR, TRAIN_ADAPTER_ALGO_OFT,
    TRAIN_ADAPTER_ALGO_BOFT,
)
from serenitymojo.training.adapter_algo_policy import adapter_algo_name
from serenitymojo.training.lokr_stack import LOKR_CARRIER_MAX_DEVICE_BYTES
from serenitymojo.models.krea2.krea2_lokr_stack import (
    Krea2LoKrSet, empty_krea2_lokr_set, build_krea2_lokr_set,
    krea2_lokr_carrier_lists, krea2_lokr_carrier_total_bytes,
    krea2_lokr_chain_all, krea2_lokr_adamw_step, krea2_lokr_grad_norm,
    krea2_lokr_clip_grads, krea2_lokr_zero_leg_l1, save_krea2_lokr,
    _krea2_slot_targeted,
)
from serenitymojo.models.krea2.krea2_loha_stack import (
    Krea2LoHaSet, empty_krea2_loha_set, build_krea2_loha_set,
    krea2_loha_carrier_lists, krea2_loha_carrier_total_bytes,
    krea2_loha_chain_all, krea2_loha_adamw_step, krea2_loha_grad_norm,
    krea2_loha_clip_grads, krea2_loha_zero_leg_l1, save_krea2_loha,
)
from serenitymojo.models.krea2.krea2_direct_lycoris_stack import (
    KREA2_DIRECT_24_GIB,
    krea2_direct_dense_carrier_bytes,
    krea2_direct_dora_preflight,
    krea2_direct_oft_preflight,
    empty_krea2_direct_dora_set,
    empty_krea2_direct_oft_set,
    Krea2StackDirectDoRA,
    Krea2StackDirectOFT,
    krea2_direct_dora_append_block_weights,
    krea2_direct_dora_blocks_to_device,
    build_krea2_direct_oft_set,
    krea2_direct_oft_blocks_to_device,
    krea2_direct_dora_zero_grads,
    krea2_direct_dora_scatter_slot_grad,
    krea2_direct_dora_grad_norm,
    krea2_direct_dora_clip_grads,
    krea2_direct_dora_adamw_step,
    krea2_direct_dora_zero_leg_l1,
    krea2_direct_dora_trainable_bytes,
    krea2_direct_oft_zero_grads,
    krea2_direct_oft_scatter_slot_grad,
    krea2_direct_oft_grad_norm,
    krea2_direct_oft_clip_grads,
    krea2_direct_oft_adamw_step,
    krea2_direct_oft_vec_l1,
    krea2_direct_oft_trainable_bytes,
    save_krea2_direct_dora,
    save_krea2_direct_oft,
)
from serenitymojo.training.dora_adapter import DoRAGrads
from serenitymojo.training.oft_onetrainer import OFTOTGrads
from serenitymojo.training.flat_direct_lycoris_stack import (
    FlatDirectDoRASet, FlatDirectDoRAGrads, FlatDirectOFTSet, FlatDirectOFTGrads,
)

# ── the streaming LoRA stack + carriers ───────────────────────────────────────
from serenitymojo.models.krea2.krea2_block import Krea2BlockLora, Krea2BlockWeights
from serenitymojo.models.krea2.krea2_stack import (
    Krea2StackLora, Krea2StackForward, Krea2StackLoraGrads,
    Krea2StackDirectDoRAGradsT, Krea2StackDirectOFTGradsT,
    Krea2StreamFinal, KREA2_SLOTS_PER_BLOCK,
    krea2_stack_lora_forward_streamed, krea2_stack_lora_backward_streamed,
    krea2_stack_lora_backward_streamed_dev,
    krea2_stack_dora_forward_streamed, krea2_stack_dora_backward_streamed_dev,
    krea2_stack_oft_forward_streamed, krea2_stack_oft_backward_streamed_dev,
    _load_krea2_block_streamed, _load_krea2_block_resident,
    krea2_stack_lora_backward_graph, krea2_stack_lora_backward_graph_slab,
    Krea2ResidentFp8, build_krea2_resident_fp8,
)
from serenitymojo.scratch_ring import ScratchRingAllocator
from serenitymojo.autograd_v2.step_slab import StepSlab

# ── frozen conditioning prefix (REUSE the inference krea2_forward pieces) ──────
from serenitymojo.models.dit.krea2_dit import (
    krea2_first, krea2_temb, krea2_tmlp, krea2_tproj, krea2_txtmlp,
    krea2_text_fusion, build_krea2_rope,
    _wb, _scale, _txtf_bundle, Krea2TextFusionWeights,
)

comptime TArc = ArcPointer[Tensor]

# ── krea2 arch invariants (config.mojo / krea2.json, header-confirmed) ─────────
comptime FEATURES = 6144
comptime HEADS = 48
comptime KVHEADS = 12
comptime HEADDIM = 128
comptime MLPDIM = 16384
comptime OUT_CH = 64                 # channels*patch^2 = 16*4
comptime TXTHEADS = 20
comptime TXTHD = 128                 # txtdim/txtheads = 2560/20
comptime NLAYERS_TXT = 12
comptime TDIM = 256
comptime NBLOCKS = 28
comptime EPS = Float32(1.0e-5)
comptime THETA = Float32(1.0e3)

# ── 512px ARM (KREA2_RES_512, default False) ─────────────────────────────────
# 1024px (default): clean [1,16,128,128] → IMGLEN=4096 → LFULL=4864 (L=4864).
# 512px (True):     clean [1,16,64,64]  → IMGLEN=1024 → LFULL=1792 (L=1792).
# Just flips LH/LW=64 (everything else derives). KREA2_RES_512=True + SYNTH=False
# = REAL 512px training: reads a 64×64-latent 512px cache via sample_padded (real
# conditioning + pos). The 512px cache is built by re-staging the source images at
# 512 (krea2_stage_images.py <dataset> <stage_512> 512) then prepare_cache <stage_512>
# <cache_512> n 512. ai-toolkit trains krea2 at 512 → this is the matched-resolution
# real-data run. Default False = the 1024px production arm, byte-untouched.
comptime KREA2_RES_512 = True

# ── DIAGNOSTIC sub-flag: synthesize the sample (KREA2_RES_512_SYNTH, default False).
# True = random latents (the wall-clock TIMING diagnostic — step time depends on the
# shapes L=LFULL, not the values; no cache needed). False = read the REAL cache.
# Pairs with KREA2_RES_512: True+SYNTH=False = real 512px training; True+SYNTH=True =
# the 512px timing diagnostic (the prior synthetic arm). Default False = real data.
comptime KREA2_RES_512_SYNTH = False

# ── DIAGNOSTIC: per-phase wall timing (KREA2_PHASE_TIMING, default False) ─────
# ctx.synchronize() + perf_counter around each per-step phase to expose the split
# of the L-INDEPENDENT fixed cost (the syncs perturb absolute time but EXPOSE the
# per-phase ms — that's the goal). Prints "PHASE <name> = <ms>" per step. Pairs
# with KREA2_RES_512=True (fixed cost is ~46% there, stands out). Default False =
# production path NOT perturbed.
comptime KREA2_PHASE_TIMING = False

# ── PERF: GPU grad-norm + folded clip (KREA2_GPU_CLIP, default False) ─────────
# The MEASURED fixed cost (~4s/step, L-independent, ~49% at 512px / ~24% at 1024px)
# is the HOST `_clip_lists` (54M bounds-checked scalar `g[i][j]*=s` ops, runs every
# step since grad_norm > max_norm) + the `_grads_to_lists` deep-copy. THIS path:
# (1) computes the global L2 norm on GPU (on_device_global_norm — one kernel + a
# 4-byte D2H, vs the host per-grad to_host + sum loop), (2) FOLDS the clip scale
# into the GPU AdamW kernel (clip_scale param — a free per-element mul, NO standalone
# 54M-element host pass). Value-class (NOT bit): the GPU tree-reduction norm differs
# ~1e-4 from the host F64 sum → a slightly different clip scale s. Default False =
# the host-clip path (byte-identical C13).
comptime KREA2_GPU_CLIP = False

comptime LH = 64 if KREA2_RES_512 else 128
comptime LW = 64 if KREA2_RES_512 else 128
comptime IMGLEN = (LH // 2) * (LW // 2)   # 1024 (512px) | 4096 (1024px)

# ── LENGTH-BUCKET PAD (the multi-sample fit) ──────────────────────────────────
# ALL samples pad to a COMMON text length LTMAX → one LFULL = LTMAX + IMGLEN size
# class, so the MAX device pool allocates ONE block size and every step reuses it
# (the measured ≥3-distinct-LT OOM fix). LTMAX must be >= the dataset's max LT;
# the giger cache's max LT is 647, so 768 (a clean multiple of 256, the reference's
# pad granularity) buckets all 4 samples. The no-mask block would let the pad
# tokens corrupt the real ones → the cuDNN flash-padmask block path (real_len =
# lt + IMGLEN, pad masked as the tail) is REQUIRED here.
# ── RESOLUTION + CAPTION-LENGTH are BUILD-TIME (the SDPA kernel is comptime-shaped) ──
# KREA2_RES_512 (above) picks 512px(64×64 latent) vs 1024px(128×128); LTMAX is the
# caption bucket length (must be >= the dataset's max caption token count — the loop
# fails loud at L<the LT>LTMAX check> otherwise, and the cache reader fails loud on a
# resolution/latent-shape mismatch). The CURRENT values are the eri2/ai-toolkit 512px
# run (KREA2_RES_512=True, LTMAX=384 ≥ eri2's max LT 282). For 1024px/giger: set
# KREA2_RES_512=False + LTMAX=768 and rebuild. (Runtime config-dispatch on cfg.resolution
# is the documented follow-up — would parameterize main on [LH,LW,LTMAX].)
comptime LTMAX = 384
comptime LFULL = LTMAX + IMGLEN           # 4864 — the single comptime arm for ALL samples

# ── INLINE-SAMPLE resolution arm (independent of train res) ───────────────────
# The inline sampler generates its OWN fresh latent (randn[1,16,LH_S,LW_S]) and is
# NOT tied to the training-cache latent shape, so we can sample at 1024 while
# training at 512. LH_S/LW_S = 128 → 1024px; LFULL_S = LTMAX + (128//2)^2 = 4480.
comptime LH_S = 128
comptime LW_S = 128
comptime IMGLEN_S = (LH_S // 2) * (LW_S // 2)   # 4096 (1024px sample)
comptime LFULL_S = LTMAX + IMGLEN_S             # 4480

# Checkpoint retention (honors ai-toolkit max_step_saves_to_keep). Prune the periodic
# save KREA2_KEEP_CHECKPOINTS back, keeping every KREA2_CKPT_MILESTONE-th + the final.
# 0 = keep all (the pre-2026-06-28 behavior).
comptime KREA2_KEEP_CHECKPOINTS = 8
comptime KREA2_CKPT_MILESTONE = 500

# ── autograd_v2 backward dispatch seam (Phase 4b adds the engine arm) ─────────
# DEFAULT FALSE = hand-chain krea2_stack_lora_backward_streamed (this file). The
# all-trainers-v2 mandate ([[feedback_all_trainers_autograd_v2]]) flips this in
# Phase 4b once the per-block engine bit-gate lands; default-off stays byte-exact.
comptime KREA2_V2_GRAPH = False
# ── StepSlab segmented (engine+slab+FLASH) backward path. Requires KREA2_V2_GRAPH.
# The 2-segment activation-checkpointed slab arm (alloc-free; per-segment slab peak
# ~6.65GB fits the 12GB fp8 base on 24GB — the whole-block slab was 12.2GB). DEFAULT
# FALSE (C13). When True, ALSO set krea2_block.mojo:KREA2_SLAB_FLASH=True so the attn
# runs cuDNN flash (O(L)) — the math attn (O(L²) scores) is 13.4GB and does NOT fit;
# flash grads are value-tolerance (NOT bit). The block bit gate keeps KREA2_SLAB_FLASH
# False (math, bit-exact).
comptime KREA2_V2_SLAB = False

# ── DEVICE-grad LoRA carrier (HAND-CHAIN path only; independent of V2_GRAPH). ──
# False (DEFAULT, byte-identical) = the streamed hand-chain materializes each
# adapter's dA/dB host-side inside the block backward (224 to_host syncs/step).
# True = krea2_stack_lora_backward_streamed_dev keeps all dA/dB on DEVICE through
# the whole stack backward and the trainer does ONE batched D2H per step (224
# syncs → 1) — SAME GEMM math, so the loss is bit-identical (the gate). The device
# grads free at step end after the batched D2H (leak guard). Only the `else`
# (hand-chain) dispatch reads this; the V2_GRAPH/SLAB arms are unaffected.
comptime KREA2_DEVICE_LORA_GRAD = False


# ══════════════════════════════════════════════════════════════════════════════
# RESIDENT CONDITIONING WEIGHTS — load ONCE, bf16 (frozen, small vs the 12GB
# blocks → NO fp8). The conditioning FORWARD still runs per step (per-sample
# context), but it reads from THIS resident set instead of re-loading `st` every
# step (the remaining per-step disk read after the fp8-resident blocks). Always-on
# (numerically identical: same bf16 weights, just loaded once). Holds the embedder
# weights (first/tmlp/tproj), the 4 text-fusion bundles + projector, and the txtmlp
# weights — every tensor `_build_conditioning` previously pulled from `st`.
# ══════════════════════════════════════════════════════════════════════════════
struct Krea2ResidentCond(Copyable, Movable):
    var first_w: TArc          # first.weight
    var first_b: TArc          # first.bias
    var tmlp0_w: TArc          # tmlp.0.weight
    var tmlp0_b: TArc          # tmlp.0.bias
    var tmlp2_w: TArc          # tmlp.2.weight
    var tmlp2_b: TArc          # tmlp.2.bias
    var tproj1_w: TArc         # tproj.1.weight
    var tproj1_b: TArc         # tproj.1.bias
    var lw0: Krea2TextFusionWeights   # txtfusion.layerwise_blocks.0
    var lw1: Krea2TextFusionWeights   # txtfusion.layerwise_blocks.1
    var rf0: Krea2TextFusionWeights   # txtfusion.refiner_blocks.0
    var rf1: Krea2TextFusionWeights   # txtfusion.refiner_blocks.1
    var projector_w: TArc      # txtfusion.projector.weight
    var txtmlp0_scale: TArc    # txtmlp.0.scale (F32)
    var txtmlp1_w: TArc        # txtmlp.1.weight
    var txtmlp1_b: TArc        # txtmlp.1.bias
    var txtmlp3_w: TArc        # txtmlp.3.weight
    var txtmlp3_b: TArc        # txtmlp.3.bias

    def __init__(
        out self,
        var first_w: TArc, var first_b: TArc,
        var tmlp0_w: TArc, var tmlp0_b: TArc, var tmlp2_w: TArc, var tmlp2_b: TArc,
        var tproj1_w: TArc, var tproj1_b: TArc,
        var lw0: Krea2TextFusionWeights, var lw1: Krea2TextFusionWeights,
        var rf0: Krea2TextFusionWeights, var rf1: Krea2TextFusionWeights,
        var projector_w: TArc,
        var txtmlp0_scale: TArc, var txtmlp1_w: TArc, var txtmlp1_b: TArc,
        var txtmlp3_w: TArc, var txtmlp3_b: TArc,
    ):
        self.first_w = first_w^
        self.first_b = first_b^
        self.tmlp0_w = tmlp0_w^
        self.tmlp0_b = tmlp0_b^
        self.tmlp2_w = tmlp2_w^
        self.tmlp2_b = tmlp2_b^
        self.tproj1_w = tproj1_w^
        self.tproj1_b = tproj1_b^
        self.lw0 = lw0^
        self.lw1 = lw1^
        self.rf0 = rf0^
        self.rf1 = rf1^
        self.projector_w = projector_w^
        self.txtmlp0_scale = txtmlp0_scale^
        self.txtmlp1_w = txtmlp1_w^
        self.txtmlp1_b = txtmlp1_b^
        self.txtmlp3_w = txtmlp3_w^
        self.txtmlp3_b = txtmlp3_b^


# Load the conditioning weights ONCE (bf16, via the SAME _wb/_scale/_txtf_bundle
# loaders the per-step path used → byte-identical values). Called once in main.
def load_krea2_resident_cond(
    st: ShardedSafeTensors, key_prefix: String, ctx: DeviceContext
) raises -> Krea2ResidentCond:
    return Krea2ResidentCond(
        TArc(_wb(st, key_prefix + "first.weight", ctx)),
        TArc(_wb(st, key_prefix + "first.bias", ctx)),
        TArc(_wb(st, key_prefix + "tmlp.0.weight", ctx)),
        TArc(_wb(st, key_prefix + "tmlp.0.bias", ctx)),
        TArc(_wb(st, key_prefix + "tmlp.2.weight", ctx)),
        TArc(_wb(st, key_prefix + "tmlp.2.bias", ctx)),
        TArc(_wb(st, key_prefix + "tproj.1.weight", ctx)),
        TArc(_wb(st, key_prefix + "tproj.1.bias", ctx)),
        _txtf_bundle(st, key_prefix + "txtfusion.layerwise_blocks.0", ctx),
        _txtf_bundle(st, key_prefix + "txtfusion.layerwise_blocks.1", ctx),
        _txtf_bundle(st, key_prefix + "txtfusion.refiner_blocks.0", ctx),
        _txtf_bundle(st, key_prefix + "txtfusion.refiner_blocks.1", ctx),
        TArc(_wb(st, key_prefix + "txtfusion.projector.weight", ctx)),
        TArc(_scale(st, key_prefix + "txtmlp.0.scale", ctx)),
        TArc(_wb(st, key_prefix + "txtmlp.1.weight", ctx)),
        TArc(_wb(st, key_prefix + "txtmlp.1.bias", ctx)),
        TArc(_wb(st, key_prefix + "txtmlp.3.weight", ctx)),
        TArc(_wb(st, key_prefix + "txtmlp.3.bias", ctx)),
    )


# ══════════════════════════════════════════════════════════════════════════════
# CONDITIONING — the FROZEN krea2_forward prefix (steps 1-11) up to the block
# stack: produce `combined [1,LFULL,F]`, `blk_vec`, `tmlp_out`, and the per-token
# rope (cos,sin). `img` is the PATCHIFIED NOISED latent (caller noises in latent
# space first). All weights stream from `st` (embedders + 4 text-fusion bundles
# are small, loaded once). NO grad here (frozen). The length-bucket REORDER makes
# combined = [TXT_real | IMG | TXT_pad] (valid prefix + pad tail for the cuDNN
# flash-padmask). Mirrors krea2_forward:1358-1454 + the reorder.
# ══════════════════════════════════════════════════════════════════════════════
struct _Cond(Movable):
    var combined: TArc        # [1, LFULL, F]
    var blk_vec: Tensor       # [1, 6*F]
    var tmlp_out: Tensor      # [1, 1, F]
    var cos: Tensor           # [LFULL, HEADDIM/2]
    var sin: Tensor           # [LFULL, HEADDIM/2]

    def __init__(
        out self, var combined: TArc, var blk_vec: Tensor, var tmlp_out: Tensor,
        var cos: Tensor, var sin: Tensor,
    ):
        self.combined = combined^
        self.blk_vec = blk_vec^
        self.tmlp_out = tmlp_out^
        self.cos = cos^
        self.sin = sin^


def _build_conditioning[LT: Int, LFULL: Int](
    cond_w: Krea2ResidentCond,   # RESIDENT conditioning weights (loaded once; no
        # per-step `st` read). Numerically identical to the old per-step `_wb`/
        # `_scale`/`_txtf_bundle` loads — same bf16 weights, just loaded once.
    img: Tensor,            # [1, IMGLEN, 64] F32  PATCHIFIED noised latent
    context: Tensor,        # [1, LT, 12, 2560] BF16   (LT == LTMAX bucket length)
    pos: Tensor,            # [1, LFULL, 3] F32 (txt zeros [LTMAX] + img grid)
    t: Tensor,              # [1] F32 timestep (in [0,1])
    real_text_len: Int,     # the natural caption length lt (<= LT==LTMAX). The
        # length-bucket reorder makes the valid tokens a CONTIGUOUS PREFIX
        # [TXT_real(0:lt) | IMG(lt:lt+IMGLEN)] with TXT_pad at the tail, so the
        # cuDNN flash-padmask (tail-only) masks the pad. real_len = lt + IMGLEN.
    ctx: DeviceContext,
) raises -> _Cond:
    # 1) img = first(img) → [1, IMGLEN, F]. img is F32 → cast bf16 to match the bf16
    # `first` head (= reference v.to(bf16) on the head; img feed is bf16 in inference).
    var img_bf = cast_tensor(img, STDtype.BF16, ctx)
    var img_e = krea2_first(
        img_bf, cond_w.first_w[], cond_w.first_b[], ctx,
    )

    # 2) t = tmlp(temb(t)) → [1,1,F].
    var te = krea2_temb(t, TDIM, ctx, STDtype.BF16)            # [1, 256]
    var t_vec = krea2_tmlp(
        te,
        cond_w.tmlp0_w[], cond_w.tmlp0_b[],
        cond_w.tmlp2_w[], cond_w.tmlp2_b[],
        ctx,
    )
    var t3 = reshape(t_vec, [1, 1, FEATURES], ctx)            # [1,1,F] = tmlp_out

    # 3) blk_vec = tproj(t3) → [1, 6*F].
    var blk_vec = krea2_tproj(
        t3, cond_w.tproj1_w[], cond_w.tproj1_b[], ctx,
    )
    var blk_vec2 = reshape(blk_vec, [1, 6 * FEATURES], ctx)   # [1, 6*F]

    # 4-5) context = txtfusion(context) (b==1 → txtmask all-ones → refiner no-op).
    var ctx_fused = krea2_text_fusion[LT, NLAYERS_TXT, TXTHEADS, TXTHD](
        context, cond_w.lw0, cond_w.lw1,
        cond_w.projector_w[],
        cond_w.rf0, cond_w.rf1, Optional[Tensor](None), ctx,
    )                                                          # [1, LT, txtdim]

    # 6) context = txtmlp(context) → [1, LT, F].
    var ctx_proj = krea2_txtmlp(
        ctx_fused,
        cond_w.txtmlp0_scale[],
        cond_w.txtmlp1_w[], cond_w.txtmlp1_b[],
        cond_w.txtmlp3_w[], cond_w.txtmlp3_b[],
        ctx,
    )

    # 7-8) LENGTH-BUCKET REORDER → [TXT_real(0:lt) | IMG(lt:lt+IMGLEN) | TXT_pad(tail)].
    # ctx_proj is [1,LTMAX,F] (real text [0:lt], pad text [lt:LTMAX]); img_e is
    # [1,IMGLEN,F]. The cuDNN flash-padmask masks only the TAIL, so the valid tokens
    # (real text + image) must be a contiguous PREFIX; the pad text moves to the
    # tail. combined = cat(real_text, img, pad_text) → [1, LFULL, F].
    var combined: Tensor
    if real_text_len < LT:
        var real_text = slice(ctx_proj, 1, 0, real_text_len, ctx)          # [1,lt,F]
        var pad_text = slice(ctx_proj, 1, real_text_len, LT - real_text_len, ctx)  # [1,LTMAX-lt,F]
        var head = concat(1, ctx, real_text, img_e)                        # [1,lt+IMGLEN,F]
        combined = concat(1, ctx, head, pad_text)                          # [1,LFULL,F]
    else:
        # lt == LTMAX: no pad → the original [TXT | IMG] order (no-mask block path).
        combined = concat(1, ctx, ctx_proj, img_e)                         # [1,LFULL,F]

    # 9) flash-padmask: valid prefix = lt + IMGLEN; [real_len:LFULL] is masked pad.

    # 10) rope: pos [1,LFULL,3] is [txt_zeros(LTMAX) | img grid]. Reorder to match
    # combined: [txt_real_zeros(lt) | img grid | txt_pad_zeros(LTMAX-lt)]. Text
    # positions are ALL-ZERO (krea2_build_pos) so this reorder changes NO token's
    # rotation — it only aligns the per-token table to the reordered sequence.
    var pos_re: Tensor
    if real_text_len < LT:
        var pos_real = slice(pos, 1, 0, real_text_len, ctx)                # [1,lt,3]
        var pos_img = slice(pos, 1, LT, LFULL - LT, ctx)                   # [1,IMGLEN,3]
        var pos_pad = slice(pos, 1, real_text_len, LT - real_text_len, ctx)  # [1,LTMAX-lt,3]
        var pos_head = concat(1, ctx, pos_real, pos_img)                   # [1,lt+IMGLEN,3]
        pos_re = concat(1, ctx, pos_head, pos_pad)                         # [1,LFULL,3]
    else:
        pos_re = pos.clone(ctx)
    var pos_flat = reshape(pos_re, [LFULL * 3], ctx)
    var axes = List[Int]()
    axes.append(32); axes.append(48); axes.append(48)
    var rope = build_krea2_rope(pos_flat, axes, THETA, ctx, STDtype.F32)
    var rcos = rope[0].clone(ctx)
    var rsin = rope[1].clone(ctx)

    return _Cond(TArc(combined^), blk_vec2^, t3^, rcos^, rsin^)


# ══════════════════════════════════════════════════════════════════════════════
# ONE TRAINING SAMPLE — comptime-monomorphized on (LT, LFULL). Noises the clean
# latent in LATENT space, patchifies, builds conditioning, runs the streaming
# stack fwd, computes the flow-match MSE (target = noise - clean, on the IMAGE
# tokens), runs the streaming stack bwd. Returns grads + loss + grad_norm.
# ══════════════════════════════════════════════════════════════════════════════
struct _StepOut(Movable):
    var grads: Krea2StackLoraGrads
    var loss: Float32
    var grad_norm: Float32

    def __init__(out self, var grads: Krea2StackLoraGrads, loss: Float32, grad_norm: Float32):
        self.grads = grads^
        self.loss = loss
        self.grad_norm = grad_norm


struct _StepOutDoRAHost(Movable):
    var grads: FlatDirectDoRAGrads
    var loss: Float32

    def __init__(out self, var grads: FlatDirectDoRAGrads, loss: Float32):
        self.grads = grads^
        self.loss = loss


struct _StepOutDoRA(Movable):
    var grads: Krea2StackDirectDoRAGradsT
    var loss: Float32

    def __init__(out self, var grads: Krea2StackDirectDoRAGradsT, loss: Float32):
        self.grads = grads^
        self.loss = loss

    def to_host(
        deinit self, masters: FlatDirectDoRASet, targets: Int, ctx: DeviceContext,
    ) raises -> _StepOutDoRAHost:
        return _StepOutDoRAHost(
            _direct_dora_grads_to_host(self.grads^, masters, targets, ctx),
            self.loss,
        )


struct _StepOutOFTHost(Movable):
    var grads: FlatDirectOFTGrads
    var loss: Float32

    def __init__(out self, var grads: FlatDirectOFTGrads, loss: Float32):
        self.grads = grads^
        self.loss = loss


struct _StepOutOFT(Movable):
    var grads: Krea2StackDirectOFTGradsT
    var loss: Float32

    def __init__(out self, var grads: Krea2StackDirectOFTGradsT, loss: Float32):
        self.grads = grads^
        self.loss = loss

    def to_host(
        deinit self, masters: FlatDirectOFTSet, targets: Int, ctx: DeviceContext,
    ) raises -> _StepOutOFTHost:
        return _StepOutOFTHost(
            _direct_oft_grads_to_host(self.grads^, masters, targets, ctx),
            self.loss,
        )


def _train_one_sample(
    st: ShardedSafeTensors, key_prefix: String,
    clean: Tensor,          # [1, 16, LH, LW] F32 normalized latent
    context: Tensor,        # [1, LTMAX, 12, 2560] BF16  (PADDED to the bucket)
    pos: Tensor,            # [1, LFULL, 3] F32          (padded grid)
    lt: Int,                # natural caption length (for the additive pad mask)
    lora: Krea2StackLora, fin: Krea2StreamFinal,
    cond_w: Krea2ResidentCond,   # RESIDENT conditioning weights (loaded once; the
        # conditioning forward reads these instead of `st` every step).
    sigma: Float32,         # flow-match t (= blend coeff = model timestep), in [0,1]
    noise_seed: UInt64,
    cfg: TrainConfig,
    ctx: DeviceContext,
    resident: Optional[Krea2ResidentFp8] = Optional[Krea2ResidentFp8](None),
        # T2.B fp8-quantized-resident base (cfg.quantized_resident=="fp8_e4m3").
        # None (default) = the per-step disk stream from `st` (C13 byte-identical).
) raises -> _StepOut:
    var _pt = 0
    comptime if KREA2_PHASE_TIMING:
        ctx.synchronize(); _pt = Int(perf_counter_ns())

    # ── flow-match noise in LATENT space (before patchify; krea2.py order) ──────
    # x_t = (1-sigma)*clean + sigma*noise ; target = noise - clean  (krea2.py:403).
    var noise = _gaussian_like(clean, noise_seed, ctx)        # [1,16,LH,LW] F32
    var fm = flow_match_noise_target(clean, sigma, noise, ctx)
    var noised_lat = fm.x_t.clone(ctx)                        # [1,16,LH,LW]
    # patchify the NOISED latent → img [1, IMGLEN, 64] (== krea2_forward.img).
    var img = krea2_patchify[LH, LW](noised_lat, ctx)
    # target on the IMAGE tokens, patchified the same way → [1, IMGLEN, 64].
    var target_img = krea2_patchify[LH, LW](fm.target, ctx)
    comptime if KREA2_PHASE_TIMING:
        _pt = _phase_ms("noise+patchify", _pt, ctx)

    # ── conditioning (frozen prefix) → combined / blk_vec / tmlp_out / rope ─────
    # Monomorphized on the BUCKET text length LTMAX (context is padded to LTMAX). The
    # LENGTH-BUCKET REORDER (in _build_conditioning) makes combined =
    # [TXT_real(0:lt) | IMG(lt:lt+IMGLEN) | TXT_pad(tail)], so the valid tokens are a
    # contiguous PREFIX of length lt+IMGLEN and the cuDNN flash-padmask masks the
    # [lt+IMGLEN : LFULL] tail. Image tokens occupy [lt : lt+IMGLEN].
    var t1 = _t_scalar(sigma, ctx)                            # [1] F32 timestep
    var cond = _build_conditioning[LTMAX, LFULL](
        cond_w, img, context, pos, t1, lt, ctx,
    )
    comptime if KREA2_PHASE_TIMING:
        _pt = _phase_ms("conditioning_fwd", _pt, ctx)

    # ── length-bucket flash-padmask: the valid contiguous prefix length. lt == LTMAX
    # → real_len == LFULL → the block's no-mask (full-attn) path (no extra masking).
    var real_len = Optional[Int](lt + IMGLEN)

    # ── streaming stack forward (txtlen = lt, imglen = IMGLEN: image at [lt:lt+IMGLEN]) ─
    var fwd = krea2_stack_lora_forward_streamed[LFULL, HEADS, KVHEADS, HEADDIM](
        cond.combined, cond.blk_vec, cond.tmlp_out,
        st, key_prefix, NBLOCKS, lora, fin,
        cond.cos, cond.sin, EPS, lt, IMGLEN, ctx, real_len, resident,
    )
    comptime if KREA2_PHASE_TIMING:
        _pt = _phase_ms("stack_forward", _pt, ctx)

    # ── flow-match MSE loss (levers; default MSE) on the image-token velocity ───
    var pred_h = fwd.velocity[].to_host(ctx)                  # [IMGLEN*64]
    var tgt_h = target_img.to_host(ctx)
    var lg = levers_loss_grad(pred_h, tgt_h, sigma, cfg)
    var loss = lg.loss
    var d_velocity = Tensor.from_host(
        lg.d_pred, [1, IMGLEN, OUT_CH], STDtype.F32, ctx,
    )
    comptime if KREA2_PHASE_TIMING:
        _pt = _phase_ms("loss+to_host", _pt, ctx)

    # ── streaming stack backward (hand-chain default; v2 arm Phase 4b) ──────────
    var grads: Krea2StackLoraGrads
    comptime if KREA2_V2_GRAPH:
        comptime if KREA2_V2_SLAB:
            # engine+slab+FLASH: the 2-segment activation-checkpointed slab backward
            # (alloc-free; per-segment slab ~6.65GB fits the 12GB fp8 base on 24GB).
            # ONE StepSlab allocated once + reset per block; attn = cuDNN flash
            # (KREA2_SLAB_FLASH=True at build). 8GB slab (the worst segment 6.65GB +
            # margin). NO capture (capture OFF — the speed is engine+slab+flash).
            var slab = StepSlab(ctx, 8 * 1024 * 1024 * 1024)
            grads = krea2_stack_lora_backward_graph_slab[LFULL, HEADS, KVHEADS, HEADDIM](
                d_velocity, cond.blk_vec, cond.tmlp_out,
                st, key_prefix, NBLOCKS, lora, fin, fwd,
                cond.cos, cond.sin, EPS, ctx, slab, real_len, resident,
            )
        else:
            # Phase 4b: autograd_v2 engine arm (drop-in, SAME conductor loop + slots).
            # The coarse per-block graph calls the WHOLE block backward oracle, so this
            # is bit-identical to the streamed hand-chain ([[feedback_all_trainers_autograd_v2]]).
            # The scratch ring is a no-op safety bracket for the coarse arm (the oracle
            # manages its own transient device allocations); a small ring suffices.
            var scratch_v2 = ScratchRingAllocator(ctx, 64 * 1024 * 1024, 2)
            grads = krea2_stack_lora_backward_graph[LFULL, HEADS, KVHEADS, HEADDIM](
                d_velocity, cond.blk_vec, cond.tmlp_out,
                st, key_prefix, NBLOCKS, lora, fin, fwd,
                cond.cos, cond.sin, EPS, ctx, scratch_v2, real_len, resident,
            )
    else:
        comptime if KREA2_DEVICE_LORA_GRAD:
            # HAND-CHAIN, device-grad carrier (option B): the per-block backward keeps
            # its 8 LoRA dA/dB on DEVICE (no per-adapter to_host), then the stack
            # backward batch-copies that ONE block's 8 grads to host under the single
            # per-block fence and frees them before the next block → 28 syncs/step (vs
            # 224), ≤8 device grads (~7.7MB) ever resident (vs 215MB holding all 224,
            # which tipped the 24GB OOM). SAME GEMM math → loss bit-identical. Returns
            # the host Krea2StackLoraGrads directly (no separate trainer-side D2H).
            grads = krea2_stack_lora_backward_streamed_dev[LFULL, HEADS, KVHEADS, HEADDIM](
                d_velocity, cond.blk_vec, cond.tmlp_out,
                st, key_prefix, NBLOCKS, lora, fin, fwd,
                cond.cos, cond.sin, EPS, ctx, real_len, resident,
            )
        else:
            grads = krea2_stack_lora_backward_streamed[LFULL, HEADS, KVHEADS, HEADDIM](
                d_velocity, cond.blk_vec, cond.tmlp_out,
                st, key_prefix, NBLOCKS, lora, fin, fwd,
                cond.cos, cond.sin, EPS, ctx, real_len, resident,
            )

    comptime if KREA2_PHASE_TIMING:
        _pt = _phase_ms("stack_backward", _pt, ctx)

    var gn = _grad_norm(grads)
    comptime if KREA2_PHASE_TIMING:
        _pt = _phase_ms("grad_norm", _pt, ctx)
    return _StepOut(grads^, loss, gn)


def _train_one_sample_dora(
    st: ShardedSafeTensors, key_prefix: String,
    clean: Tensor,
    context: Tensor,
    pos: Tensor,
    lt: Int,
    dora: Krea2StackDirectDoRA, fin: Krea2StreamFinal,
    cond_w: Krea2ResidentCond,
    sigma: Float32,
    noise_seed: UInt64,
    cfg: TrainConfig,
    ctx: DeviceContext,
    resident: Optional[Krea2ResidentFp8] = Optional[Krea2ResidentFp8](None),
) raises -> _StepOutDoRA:
    var _pt = 0
    comptime if KREA2_PHASE_TIMING:
        ctx.synchronize(); _pt = Int(perf_counter_ns())

    var noise = _gaussian_like(clean, noise_seed, ctx)
    var fm = flow_match_noise_target(clean, sigma, noise, ctx)
    var noised_lat = fm.x_t.clone(ctx)
    var img = krea2_patchify[LH, LW](noised_lat, ctx)
    var target_img = krea2_patchify[LH, LW](fm.target, ctx)
    comptime if KREA2_PHASE_TIMING:
        _pt = _phase_ms("noise+patchify", _pt, ctx)

    var t1 = _t_scalar(sigma, ctx)
    var cond = _build_conditioning[LTMAX, LFULL](
        cond_w, img, context, pos, t1, lt, ctx,
    )
    comptime if KREA2_PHASE_TIMING:
        _pt = _phase_ms("conditioning_fwd", _pt, ctx)

    var real_len = Optional[Int](lt + IMGLEN)

    var fwd = krea2_stack_dora_forward_streamed[LFULL, HEADS, KVHEADS, HEADDIM](
        cond.combined, cond.blk_vec, cond.tmlp_out,
        st, key_prefix, NBLOCKS, dora, fin,
        cond.cos, cond.sin, EPS, lt, IMGLEN, ctx, real_len, resident,
    )
    comptime if KREA2_PHASE_TIMING:
        _pt = _phase_ms("stack_forward", _pt, ctx)

    var pred_h = fwd.velocity[].to_host(ctx)
    var tgt_h = target_img.to_host(ctx)
    var lg = levers_loss_grad(pred_h, tgt_h, sigma, cfg)
    var d_velocity = Tensor.from_host(
        lg.d_pred, [1, IMGLEN, OUT_CH], STDtype.F32, ctx,
    )
    comptime if KREA2_PHASE_TIMING:
        _pt = _phase_ms("loss+to_host", _pt, ctx)

    var grads = krea2_stack_dora_backward_streamed_dev[LFULL, HEADS, KVHEADS, HEADDIM](
        d_velocity, cond.blk_vec, cond.tmlp_out,
        st, key_prefix, NBLOCKS, dora, fin, fwd,
        cond.cos, cond.sin, EPS, ctx, real_len, resident,
    )
    comptime if KREA2_PHASE_TIMING:
        _pt = _phase_ms("stack_backward", _pt, ctx)

    return _StepOutDoRA(grads^, lg.loss)


def _train_one_sample_oft(
    st: ShardedSafeTensors, key_prefix: String,
    clean: Tensor,
    context: Tensor,
    pos: Tensor,
    lt: Int,
    oft: Krea2StackDirectOFT, fin: Krea2StreamFinal,
    cond_w: Krea2ResidentCond,
    sigma: Float32,
    noise_seed: UInt64,
    cfg: TrainConfig,
    ctx: DeviceContext,
    resident: Optional[Krea2ResidentFp8] = Optional[Krea2ResidentFp8](None),
) raises -> _StepOutOFT:
    var _pt = 0
    comptime if KREA2_PHASE_TIMING:
        ctx.synchronize(); _pt = Int(perf_counter_ns())

    var noise = _gaussian_like(clean, noise_seed, ctx)
    var fm = flow_match_noise_target(clean, sigma, noise, ctx)
    var noised_lat = fm.x_t.clone(ctx)
    var img = krea2_patchify[LH, LW](noised_lat, ctx)
    var target_img = krea2_patchify[LH, LW](fm.target, ctx)
    comptime if KREA2_PHASE_TIMING:
        _pt = _phase_ms("noise+patchify", _pt, ctx)

    var t1 = _t_scalar(sigma, ctx)
    var cond = _build_conditioning[LTMAX, LFULL](
        cond_w, img, context, pos, t1, lt, ctx,
    )
    comptime if KREA2_PHASE_TIMING:
        _pt = _phase_ms("conditioning_fwd", _pt, ctx)

    var real_len = Optional[Int](lt + IMGLEN)

    var fwd = krea2_stack_oft_forward_streamed[LFULL, HEADS, KVHEADS, HEADDIM](
        cond.combined, cond.blk_vec, cond.tmlp_out,
        st, key_prefix, NBLOCKS, oft, fin,
        cond.cos, cond.sin, EPS, lt, IMGLEN, ctx, real_len, resident,
    )
    comptime if KREA2_PHASE_TIMING:
        _pt = _phase_ms("stack_forward", _pt, ctx)

    var pred_h = fwd.velocity[].to_host(ctx)
    var tgt_h = target_img.to_host(ctx)
    var lg = levers_loss_grad(pred_h, tgt_h, sigma, cfg)
    var d_velocity = Tensor.from_host(
        lg.d_pred, [1, IMGLEN, OUT_CH], STDtype.F32, ctx,
    )
    comptime if KREA2_PHASE_TIMING:
        _pt = _phase_ms("loss+to_host", _pt, ctx)

    var grads = krea2_stack_oft_backward_streamed_dev[LFULL, HEADS, KVHEADS, HEADDIM](
        d_velocity, cond.blk_vec, cond.tmlp_out,
        st, key_prefix, NBLOCKS, oft, fin, fwd,
        cond.cos, cond.sin, EPS, ctx, real_len, resident,
    )
    comptime if KREA2_PHASE_TIMING:
        _pt = _phase_ms("stack_backward", _pt, ctx)

    return _StepOutOFT(grads^, lg.loss)


# ══════════════════════════════════════════════════════════════════════════════
# helpers
# ══════════════════════════════════════════════════════════════════════════════
def _t_scalar(v: Float32, ctx: DeviceContext) raises -> Tensor:
    var h = List[Float32]()
    h.append(v)
    return Tensor.from_host(h^, [1], STDtype.F32, ctx)


# Standard-normal noise tensor via the shared device generator (ops/random.randn,
# the repo Box-Muller convention — [[project_noise_boxmuller_bug]] was the bad
# Box-Muller; randn is the canonical fixed path used by inference + trainers).
from serenitymojo.ops.random import randn


def _gaussian_like(like: Tensor, seed: UInt64, ctx: DeviceContext) raises -> Tensor:
    return randn(like.shape(), seed, STDtype.F32, ctx)


# DIAGNOSTIC (KREA2_RES_512): a SYNTHETIC sample at the comptime LH/LW dims — the
# step TIME depends on the shapes (L=LFULL), not the values, so random latents are
# fine for the wall-clock test (lead-approved). Shapes mirror sample_padded's output
# (context already padded to LTMAX). text_len = LTMAX so real_len = LTMAX+IMGLEN =
# LFULL (no pad tail) — the same flash path the 1024px arm runs, just shorter L.
from serenitymojo.ops.cast import cast_tensor as _cast_t


def _synthetic_sample(idx: Int, ctx: DeviceContext) raises -> KreaTrainSample:
    var seed = UInt64(4242) + UInt64(idx)
    var clean = randn(
        [1, KREA2_LATENT_CHANNELS, LH, LW], seed, STDtype.F32, ctx
    )
    var img = randn([1, IMGLEN, 64], seed + 1, STDtype.F32, ctx)
    var context = _cast_t(
        randn([1, LTMAX, KREA2_TXT_LAYERS, KREA2_TXT_DIM], seed + 2, STDtype.F32, ctx),
        STDtype.BF16, ctx,
    )
    var pos = krea2_build_pos[LH, LW](LTMAX, ctx)        # [1, LFULL, 3]
    return KreaTrainSample(
        TArc(clean^), TArc(img^), TArc(context^), TArc(pos^), LTMAX, idx,
    )


def _grad_norm(grads: Krea2StackLoraGrads) -> Float32:
    var ss = Float64(0.0)
    for i in range(len(grads.grads)):
        var g = grads.grads[i].copy()
        if g.d_a:
            var a = g.d_a.value().copy()
            for j in range(len(a)):
                ss += Float64(a[j]) * Float64(a[j])
        if g.d_b:
            var b = g.d_b.value().copy()
            for j in range(len(b)):
                ss += Float64(b[j]) * Float64(b[j])
    return Float32(sqrt(ss))


# global-norm clip applied to the EXTRACTED plain grad lists (avoids mutating the
# Optional-wrapped Krea2LoraGrad in place; the fused AdamW reads these lists).
def _clip_lists(mut gl: _GradLists, gn: Float32, max_norm: Float32):
    if gn <= max_norm or gn == Float32(0.0):
        return
    var s = max_norm / gn
    for i in range(len(gl.d_a)):
        for j in range(len(gl.d_a[i])):
            gl.d_a[i][j] = gl.d_a[i][j] * s
        for j in range(len(gl.d_b[i])):
            gl.d_b[i][j] = gl.d_b[i][j] * s


# ── LoRA set: host List[LoraAdapter] (authoritative + AdamW moments) ──────────
# 8 adapters per block, order matches Krea2BlockLora: wq wk wv gate wo
# mlp_gate mlp_up mlp_down. in/out from the krea2 dims.
def _build_host_lora(rank: Int, alpha: Float32) -> List[LoraAdapter]:
    var ad = List[LoraAdapter]()
    var seed = UInt64(7000)
    for _ in range(NBLOCKS):
        ad.append(make_lora_adapter(rank, alpha, FEATURES, HEADS * HEADDIM, seed)); seed += 1     # wq
        ad.append(make_lora_adapter(rank, alpha, FEATURES, KVHEADS * HEADDIM, seed)); seed += 1   # wk
        ad.append(make_lora_adapter(rank, alpha, FEATURES, KVHEADS * HEADDIM, seed)); seed += 1   # wv
        ad.append(make_lora_adapter(rank, alpha, FEATURES, FEATURES, seed)); seed += 1            # gate
        ad.append(make_lora_adapter(rank, alpha, FEATURES, FEATURES, seed)); seed += 1            # wo
        ad.append(make_lora_adapter(rank, alpha, FEATURES, MLPDIM, seed)); seed += 1              # mlp_gate
        ad.append(make_lora_adapter(rank, alpha, FEATURES, MLPDIM, seed)); seed += 1              # mlp_up
        ad.append(make_lora_adapter(rank, alpha, MLPDIM, FEATURES, seed)); seed += 1              # mlp_down
    return ad^


# ── LoRA SAVE (MJ-0805): write the trained adapters as a RE-LOADABLE PEFT file ─
# The PEFT module prefix MUST match the keys ai-toolkit/inference krea2 LoRAs LOAD
# by — VERIFIED against a real ai-toolkit krea2 save (output/my_first_lora_v1/*.
# safetensors): `diffusion_model.blocks.<bi>.attn.{wq,wk,wv,gate,wo}` and
# `.mlp.{gate,up,down}`, with save_lora_peft appending `.lora_A.weight`[rank,in] /
# `.lora_B.weight`[out,rank] (BF16). The slot order matches _build_host_lora
# (0=wq 1=wk 2=wv 3=gate 4=wo 5=mlp_gate 6=mlp_up 7=mlp_down). NOTE: ai-toolkit
# ALSO trains txtfusion layerwise/refiner blocks; the Mojo trainer is main-block
# only (frozen-skip text-fusion, config.mojo), so this saves the 28×8=224 main-block
# adapters — a main-block LoRA that re-loads (txtfusion keys simply absent).
def _krea2_lora_prefix(bi: Int, slot: Int) raises -> String:
    var b = String("diffusion_model.blocks.") + String(bi)
    if slot == 0:
        return b + ".attn.wq"
    elif slot == 1:
        return b + ".attn.wk"
    elif slot == 2:
        return b + ".attn.wv"
    elif slot == 3:
        return b + ".attn.gate"
    elif slot == 4:
        return b + ".attn.wo"
    elif slot == 5:
        return b + ".mlp.gate"
    elif slot == 6:
        return b + ".mlp.up"
    elif slot == 7:
        return b + ".mlp.down"
    raise Error(String("_krea2_lora_prefix: bad slot ") + String(slot))


def _krea2_train_targets(raw_targets: Int) raises -> Int:
    if raw_targets == 1:
        return 1
    if raw_targets == 2 or raw_targets == 3:
        return 2
    raise Error("train_krea2: lokr_targets must be attn or all for Krea2")


def _krea2_block_weight_host(
    w: Krea2BlockWeights, slot: Int, ctx: DeviceContext,
) raises -> List[Float32]:
    if slot == 0:
        return w.wq[].to_host(ctx)
    if slot == 1:
        return w.wk[].to_host(ctx)
    if slot == 2:
        return w.wv[].to_host(ctx)
    if slot == 3:
        return w.gate_w[].to_host(ctx)
    if slot == 4:
        return w.wo[].to_host(ctx)
    if slot == 5:
        return w.mlp_gate_w[].to_host(ctx)
    if slot == 6:
        return w.mlp_up_w[].to_host(ctx)
    if slot == 7:
        return w.mlp_down_w[].to_host(ctx)
    raise Error(String("_krea2_block_weight_host: bad slot ") + String(slot))


def _krea2_direct_dora_block_weights_host(
    st: ShardedSafeTensors, key_prefix: String, bi: Int, targets: Int,
    ctx: DeviceContext,
    resident: Optional[Krea2ResidentFp8] = Optional[Krea2ResidentFp8](None),
) raises -> List[List[Float32]]:
    var wbi: Krea2BlockWeights
    if resident:
        wbi = _load_krea2_block_resident(resident.value(), bi, ctx)
    else:
        wbi = _load_krea2_block_streamed(st, bi, key_prefix, ctx)
    var out = List[List[Float32]]()
    for slot in range(KREA2_SLOTS_PER_BLOCK):
        if _krea2_slot_targeted(slot, targets):
            out.append(_krea2_block_weight_host(wbi, slot, ctx))
        else:
            out.append(List[Float32]())
    ctx.synchronize()
    return out^


def _build_krea2_direct_dora_set_streamed(
    st: ShardedSafeTensors, key_prefix: String, rank: Int, alpha: Float32,
    targets: Int, ctx: DeviceContext,
    resident: Optional[Krea2ResidentFp8] = Optional[Krea2ResidentFp8](None),
) raises -> FlatDirectDoRASet:
    var set = empty_krea2_direct_dora_set()
    var seed = UInt64(7000)
    for bi in range(NBLOCKS):
        var weights = _krea2_direct_dora_block_weights_host(
            st, key_prefix, bi, targets, ctx, resident,
        )
        krea2_direct_dora_append_block_weights(
            set, bi, weights^, FEATURES, MLPDIM, HEADS * HEADDIM,
            KVHEADS * HEADDIM, rank, alpha, targets,
            seed + UInt64(bi * KREA2_SLOTS_PER_BLOCK), False,
        )
        ctx.synchronize()
    return set^


# Build the LoRA output path from cfg (workspace_dir/<name>_krea2_lora_<step>.safetensors).
# save_filename_prefix overrides <name> when set (mirrors the other trainers' naming).
def _lora_save_path(cfg: TrainConfig, step: Int) raises -> String:
    var stem = cfg.save_filename_prefix if cfg.save_filename_prefix != String("") else (cfg.name + String("_krea2_lora"))
    return cfg.workspace_dir + String("/") + stem + String("_") + String(step) + String(".safetensors")


def save_krea2_lora(
    host_lora: List[LoraAdapter], path: String, ctx: DeviceContext
) raises -> Int:
    """Write the 28×8=224 trained krea2 LoRA adapters as a re-loadable PEFT
    safetensors (diffusion_model.blocks.<bi>.<module>.lora_A/B.weight). Returns the
    number of (A,B) pairs written. host_lora is the authoritative host copy
    (flat, bi*8 + slot order). The caller must ensure the output dir exists (the
    trainer mkdir -p's cfg.workspace_dir before calling)."""
    var named = List[NamedLora]()
    for bi in range(NBLOCKS):
        for s in range(KREA2_SLOTS_PER_BLOCK):
            named.append(NamedLora(
                _krea2_lora_prefix(bi, s),
                host_lora[bi * KREA2_SLOTS_PER_BLOCK + s].copy(),
            ))
    return save_lora_peft(named, path, ctx)


def save_krea2_lora_state(
    host_lora: List[LoraAdapter], path: String, ctx: DeviceContext
) raises -> Int:
    """Write the trainer-only RESUME state (A/B + AdamW moments) for the 224
    plain-LoRA adapters, in the SAME block×slot order as save_krea2_lora. Separate
    from the PEFT file (inference-only) — this carries adam_m/adam_v so a resumed
    run does not zero the optimizer moments. Loaded by load_lora_train_state."""
    var named = List[NamedLora]()
    for bi in range(NBLOCKS):
        for s in range(KREA2_SLOTS_PER_BLOCK):
            named.append(NamedLora(
                _krea2_lora_prefix(bi, s),
                host_lora[bi * KREA2_SLOTS_PER_BLOCK + s].copy(),
            ))
    return save_lora_train_state(named, path, ctx)


def _krea2_lora_resume(
    host_lora: List[LoraAdapter], scale: Float32, path: String, ctx: DeviceContext
) raises -> List[LoraAdapter]:
    """Reload the 224 plain-LoRA adapters from a checkpoint for resume. Prefers the
    `.state` file (FULL resume: A/B + AdamW moments); falls back to a plain PEFT
    file (WARM start: A/B only, moments zeroed). Returns a fresh host_lora in
    block×slot order. The caller offsets the step counter to the saved step."""
    var prefixes = List[String]()
    for bi in range(NBLOCKS):
        for s in range(KREA2_SLOTS_PER_BLOCK):
            prefixes.append(_krea2_lora_prefix(bi, s))
    var loaded: List[NamedLora]
    try:
        loaded = load_lora_train_state(prefixes, scale, path, ctx)
        print("[krea2-resume] FULL resume (A/B + AdamW moments) from", path)
    except:
        loaded = load_lora_for_resume(prefixes, scale, path, ctx)
        print("[krea2-resume] WARM start (A/B only, moments zeroed) from", path)
    var out = List[LoraAdapter]()
    for ref nl in loaded:
        out.append(nl.adapter.copy())
    if len(out) != len(host_lora):
        raise Error(String("_krea2_lora_resume: adapter count ") + String(len(out))
            + " != expected " + String(len(host_lora)))
    return out^


# convert the host LoRA set → the device Krea2StackLora the streaming stack consumes.
def _host_to_device_lora(
    host: List[LoraAdapter], ctx: DeviceContext
) raises -> Krea2StackLora:
    var blocks = List[Krea2BlockLora]()
    for bi in range(NBLOCKS):
        var base = bi * KREA2_SLOTS_PER_BLOCK
        blocks.append(Krea2BlockLora(
            Optional[LoraAdapterDevice](lora_adapter_to_device(host[base + 0], ctx)),
            Optional[LoraAdapterDevice](lora_adapter_to_device(host[base + 1], ctx)),
            Optional[LoraAdapterDevice](lora_adapter_to_device(host[base + 2], ctx)),
            Optional[LoraAdapterDevice](lora_adapter_to_device(host[base + 3], ctx)),
            Optional[LoraAdapterDevice](lora_adapter_to_device(host[base + 4], ctx)),
            Optional[LoraAdapterDevice](lora_adapter_to_device(host[base + 5], ctx)),
            Optional[LoraAdapterDevice](lora_adapter_to_device(host[base + 6], ctx)),
            Optional[LoraAdapterDevice](lora_adapter_to_device(host[base + 7], ctx)),
        ))
    return Krea2StackLora(blocks^)


# scatter the flat Krea2StackLoraGrads → parallel d_a/d_b lists for the fused AdamW
# (indexed by the SAME absolute adapter index as the host LoRA set).
def _grads_to_lists(
    grads: Krea2StackLoraGrads, n_adapters: Int
) raises -> _GradLists:
    var d_a = List[List[Float32]]()
    var d_b = List[List[Float32]]()
    for i in range(n_adapters):
        var g = grads.grads[i].copy()
        if not g.d_a or not g.d_b:
            raise Error(String("_grads_to_lists: missing grad at adapter ") + String(i))
        d_a.append(g.d_a.value().copy())
        d_b.append(g.d_b.value().copy())
    return _GradLists(d_a^, d_b^)


struct _GradLists(Movable):
    var d_a: List[List[Float32]]
    var d_b: List[List[Float32]]

    def __init__(out self, var d_a: List[List[Float32]], var d_b: List[List[Float32]]):
        self.d_a = d_a^
        self.d_b = d_b^


# (Option B moved the batched D2H INTO krea2_stack_lora_backward_streamed_dev —
# per-block, under the single per-block fence — so the trainer no longer needs a
# separate stack-wide D2H helper. See krea2_stack._block_grads_d2h_enqueue/decode.)


# ══════════════════════════════════════════════════════════════════════════════
# DRIVER — one step. With length-bucket padding ALL samples are the SAME LFULL =
# LTMAX + IMGLEN size class → ONE comptime arm (no per-LT monomorphization). The
# real caption length `lt` is passed at runtime for the additive pad mask. Returns
# the _StepOut.
# ══════════════════════════════════════════════════════════════════════════════
def _step_dispatch(
    st: ShardedSafeTensors, key_prefix: String,
    clean: Tensor, context: Tensor, pos: Tensor, lt: Int,
    lora: Krea2StackLora, fin: Krea2StreamFinal, cond_w: Krea2ResidentCond,
    sigma: Float32, noise_seed: UInt64, cfg: TrainConfig, ctx: DeviceContext,
    resident: Optional[Krea2ResidentFp8] = Optional[Krea2ResidentFp8](None),
) raises -> _StepOut:
    if lt > LTMAX:
        raise Error(
            String("train_krea2: LT=") + String(lt) + " > LTMAX=" + String(LTMAX)
            + " (raise LTMAX above the dataset's max caption length)"
        )
    return _train_one_sample(
        st, key_prefix, clean, context, pos, lt, lora, fin, cond_w,
        sigma, noise_seed, cfg, ctx, resident)


def _step_dispatch_dora(
    st: ShardedSafeTensors, key_prefix: String,
    clean: Tensor, context: Tensor, pos: Tensor, lt: Int,
    dora: Krea2StackDirectDoRA, fin: Krea2StreamFinal, cond_w: Krea2ResidentCond,
    sigma: Float32, noise_seed: UInt64, cfg: TrainConfig, ctx: DeviceContext,
    resident: Optional[Krea2ResidentFp8] = Optional[Krea2ResidentFp8](None),
) raises -> _StepOutDoRA:
    if lt > LTMAX:
        raise Error(
            String("train_krea2: LT=") + String(lt) + " > LTMAX=" + String(LTMAX)
            + " (raise LTMAX above the dataset's max caption length)"
        )
    return _train_one_sample_dora(
        st, key_prefix, clean, context, pos, lt, dora, fin, cond_w,
        sigma, noise_seed, cfg, ctx, resident)


def _step_dispatch_oft(
    st: ShardedSafeTensors, key_prefix: String,
    clean: Tensor, context: Tensor, pos: Tensor, lt: Int,
    oft: Krea2StackDirectOFT, fin: Krea2StreamFinal, cond_w: Krea2ResidentCond,
    sigma: Float32, noise_seed: UInt64, cfg: TrainConfig, ctx: DeviceContext,
    resident: Optional[Krea2ResidentFp8] = Optional[Krea2ResidentFp8](None),
) raises -> _StepOutOFT:
    if lt > LTMAX:
        raise Error(
            String("train_krea2: LT=") + String(lt) + " > LTMAX=" + String(LTMAX)
            + " (raise LTMAX above the dataset's max caption length)"
        )
    return _train_one_sample_oft(
        st, key_prefix, clean, context, pos, lt, oft, fin, cond_w,
        sigma, noise_seed, cfg, ctx, resident)


# ══════════════════════════════════════════════════════════════════════════════
# INLINE SAMPLER — resident Krea2 base + live LoRA, no save/reload round-trip.
# ══════════════════════════════════════════════════════════════════════════════
struct _Krea2InlineCond(Copyable, Movable):
    var context: TArc      # [1, LTMAX, 12, 2560] BF16
    var pos: TArc          # [1, LTMAX+imglen, 3] F32
    var text_len: Int      # natural LT before padding

    def __init__(out self, var context: TArc, var pos: TArc, text_len: Int):
        self.context = context^
        self.pos = pos^
        self.text_len = text_len


def _krea2_pad_context_to_ltmax[LTMAX: Int](
    context: Tensor, lt: Int, ctx: DeviceContext,
) raises -> Tensor:
    var sh = context.shape()
    if (
        len(sh) != 4 or sh[0] != 1
        or sh[2] != KREA2_TXT_LAYERS or sh[3] != KREA2_TXT_DIM
    ):
        raise Error("krea2 inline sampler: expected context [1, LT, 12, 2560]")
    if lt > LTMAX:
        raise Error(
            String("krea2 inline sampler: LT=") + String(lt)
            + " > LTMAX=" + String(LTMAX)
            + " (raise the compiled Krea2 LTMAX bucket)"
        )
    var ctx_bf = cast_tensor(context, STDtype.BF16, ctx)
    if lt == LTMAX:
        return ctx_bf^
    var pad = zeros_device(
        [1, LTMAX - lt, KREA2_TXT_LAYERS, KREA2_TXT_DIM], STDtype.BF16, ctx
    )
    return concat(1, ctx, ctx_bf, pad)


def _krea2_inline_cond_from_context[LH: Int, LW: Int, LTMAX: Int](
    var context: Tensor, ctx: DeviceContext,
) raises -> _Krea2InlineCond:
    var sh = context.shape()
    if (
        len(sh) != 4 or sh[0] != 1
        or sh[2] != KREA2_TXT_LAYERS or sh[3] != KREA2_TXT_DIM
    ):
        raise Error("krea2 inline sampler: expected context [1, LT, 12, 2560]")
    var lt = sh[1]
    var padded = _krea2_pad_context_to_ltmax[LTMAX](context^, lt, ctx)
    var pos = krea2_build_pos[LH, LW](LTMAX, ctx)
    return _Krea2InlineCond(TArc(padded^), TArc(pos^), lt)


def _krea2_inline_cond_from_bin[LH: Int, LW: Int, LTMAX: Int](
    path: String, ctx: DeviceContext,
) raises -> _Krea2InlineCond:
    if path == String(""):
        raise Error("krea2 inline sampler: empty context cap path")
    var context = load_tensor_bin(path, ctx)
    return _krea2_inline_cond_from_context[LH, LW, LTMAX](context^, ctx)


def _krea2_inline_cond_from_cache[LH: Int, LW: Int, LTMAX: Int](
    read cache: KreaTrainCache, sample_index: Int, ctx: DeviceContext,
) raises -> _Krea2InlineCond:
    # Read ONLY the cached caption context (res-independent); build pos at the
    # SAMPLE geometry [LH,LW]. Must NOT go through cache.sample_padded[LH,LW] —
    # that reads+validates the clean latent at [1,16,LH,LW], which mismatches when
    # the inline sample res (LH_S=128/1024px) differs from the train cache res
    # (64/512px). The context tensor is [1,LT,12,2560] regardless of image res.
    var context = cast_tensor(
        Tensor.from_view(cache.src.tensor_view(cache.context_keys[sample_index]), ctx),
        STDtype.BF16, ctx,
    )
    return _krea2_inline_cond_from_context[LH, LW, LTMAX](context^, ctx)


def _krea2_inline_uncond_from_cache[LH: Int, LW: Int, LTMAX: Int](
    read cache: KreaTrainCache, ctx: DeviceContext,
) raises -> _Krea2InlineCond:
    var u = cache.uncond[LH, LW](ctx)
    var padded = _krea2_pad_context_to_ltmax[LTMAX](u.context[], u.text_len, ctx)
    var pos = krea2_build_pos[LH, LW](LTMAX, ctx)
    return _Krea2InlineCond(TArc(padded^), TArc(pos^), u.text_len)


def _krea2_unpatch[LH: Int, LW: Int](tokens: Tensor, ctx: DeviceContext) raises -> Tensor:
    """Inverse patchify: [1,imglen,64] -> [1,16,LH,LW]."""
    comptime gh = LH // 2
    comptime gw = LW // 2
    var x6 = reshape(tokens, [1, gh, gw, 16, 2, 2], ctx)
    var xp = permute(x6, [0, 3, 1, 4, 2, 5], ctx)
    return reshape(xp, [1, 16, gh * 2, gw * 2], ctx)


def _krea2_sample_resident_latent[LH: Int, LW: Int, LTMAX: Int, LFULL_SAMPLE: Int](
    st: ShardedSafeTensors,
    key_prefix: String,
    cond_w: Krea2ResidentCond,
    fin: Krea2StreamFinal,
    lora: Krea2StackLora,
    cond: _Krea2InlineCond,
    uncond: _Krea2InlineCond,
    sample_steps: Int,
    cfg_scale: Float32,
    seed: UInt64,
    ctx: DeviceContext,
    resident: Optional[Krea2ResidentFp8] = Optional[Krea2ResidentFp8](None),
) raises -> Tensor:
    comptime imglen = (LH // 2) * (LW // 2)
    comptime assert LFULL_SAMPLE == LTMAX + imglen, "Krea2 sample LFULL mismatch"
    if sample_steps < 1:
        raise Error("krea2 inline sampler: sample steps must be >= 1")

    var latent = randn([1, 16, LH, LW], seed, STDtype.F32, ctx)
    var seq = krea2_packed_seq_len(LH * 8, LW * 8)
    var ts = krea2_timesteps(seq, sample_steps)
    print("[krea2-sample-inline] steps=", sample_steps, " cfg=", cfg_scale,
          " seed=", seed, " LT(cond)=", cond.text_len, " LT(uncond)=", uncond.text_len)

    for si in range(sample_steps):
        var t_cur = ts[si]
        var t_prev = ts[si + 1]
        var img_tokens = krea2_patchify[LH, LW](latent, ctx)
        var t_t = _t_scalar(t_cur, ctx)

        var c = _build_conditioning[LTMAX, LFULL_SAMPLE](
            cond_w, img_tokens, cond.context[], cond.pos[], t_t, cond.text_len, ctx,
        )
        var real_len_c = Optional[Int](cond.text_len + imglen)
        var pred_c = krea2_stack_lora_forward_streamed[
            LFULL_SAMPLE, HEADS, KVHEADS, HEADDIM
        ](
            c.combined, c.blk_vec, c.tmlp_out,
            st, key_prefix, NBLOCKS, lora, fin,
            c.cos, c.sin, EPS, cond.text_len, imglen, ctx, real_len_c, resident,
        )

        var v_f32: Tensor
        if cfg_scale == Float32(1.0):
            v_f32 = cast_tensor(pred_c.velocity[], STDtype.F32, ctx)
        else:
            var u = _build_conditioning[LTMAX, LFULL_SAMPLE](
                cond_w, img_tokens, uncond.context[], uncond.pos[], t_t,
                uncond.text_len, ctx,
            )
            var real_len_u = Optional[Int](uncond.text_len + imglen)
            var pred_u = krea2_stack_lora_forward_streamed[
                LFULL_SAMPLE, HEADS, KVHEADS, HEADDIM
            ](
                u.combined, u.blk_vec, u.tmlp_out,
                st, key_prefix, NBLOCKS, lora, fin,
                u.cos, u.sin, EPS, uncond.text_len, imglen, ctx,
                real_len_u, resident,
            )
            var vc_f32 = cast_tensor(pred_c.velocity[], STDtype.F32, ctx)
            var vu_f32 = cast_tensor(pred_u.velocity[], STDtype.F32, ctx)
            v_f32 = krea2_cfg(vc_f32, vu_f32, cfg_scale, ctx)

        var v_latent = _krea2_unpatch[LH, LW](v_f32, ctx)
        latent = krea2_euler_step(latent, v_latent, t_cur, t_prev, ctx)
        ctx.synchronize()
        if si == 0 or si + 1 == sample_steps or (si + 1) % 5 == 0:
            print("[krea2-sample-inline] step", si + 1, "/", sample_steps,
                  " t=", t_cur)
    return latent^


def _krea2_decode_latent_to_png[LH: Int, LW: Int](
    latent: Tensor, vae_dir: String, out_png: String, ctx: DeviceContext,
) raises:
    print("[krea2-sample-inline] decoding ->", out_png)
    var dec = QwenImageVaeDecoder[LH, LW].load(vae_dir, ctx)
    var latent_bf16 = torch_f32_to_bf16_rne(latent, ctx)
    var img = dec.decode(latent_bf16, ctx)
    save_png(img, out_png, ctx, ValueRange.SIGNED)


def _krea2_run_inline_samples[LH: Int, LW: Int, LTMAX: Int, LFULL_SAMPLE: Int](
    read cache: KreaTrainCache,
    st: ShardedSafeTensors,
    key_prefix: String,
    cond_w: Krea2ResidentCond,
    fin: Krea2StreamFinal,
    host_lora: List[LoraAdapter],
    train_cfg: TrainConfig,
    sample_cfg: SamplePromptConfig,
    completed_step: Int,
    ctx: DeviceContext,
    resident: Optional[Krea2ResidentFp8] = Optional[Krea2ResidentFp8](None),
) raises:
    var samples_dir = train_cfg.workspace_dir + String("/samples")
    _ = sys_mkdirs(samples_dir)
    var lora_dev = _host_to_device_lora(host_lora, ctx)
    for pi in range(len(sample_cfg.prompts)):
        var prompt = sample_cfg.prompts[pi].copy()
        if not prompt.enabled:
            continue
        var cond: _Krea2InlineCond
        if prompt.caps_pos.byte_length() > 0:
            cond = _krea2_inline_cond_from_bin[LH, LW, LTMAX](prompt.caps_pos, ctx)
        else:
            cond = _krea2_inline_cond_from_cache[LH, LW, LTMAX](
                cache, pi % cache.len(), ctx,
            )

        var guidance = prompt.cfg
        var uncond = cond.copy()
        if guidance != Float32(1.0):
            if prompt.caps_neg.byte_length() > 0:
                uncond = _krea2_inline_cond_from_bin[LH, LW, LTMAX](
                    prompt.caps_neg, ctx,
                )
            elif cache.context_uncond_key.byte_length() > 0:
                uncond = _krea2_inline_uncond_from_cache[LH, LW, LTMAX](cache, ctx)
            else:
                raise Error(
                    "krea2 inline sampler: cfg != 1.0 requires caps.negative "
                    + "or a training cache prepared with --uncond"
                )

        var latent = _krea2_sample_resident_latent[
            LH, LW, LTMAX, LFULL_SAMPLE
        ](
            st, key_prefix, cond_w, fin, lora_dev, cond, uncond,
            prompt.steps, guidance, prompt.seed + UInt64(completed_step * 1000 + pi),
            ctx, resident,
        )
        var out_png = (
            samples_dir + String("/step_") + String(completed_step)
            + String("_") + String(pi) + String(".png")
        )
        _krea2_decode_latent_to_png[LH, LW](latent, train_cfg.vae, out_png, ctx)
        print("[krea2-sample-inline] wrote", out_png)


def _direct_dora_grads_to_host(
    var grads: Krea2StackDirectDoRAGradsT,
    masters: FlatDirectDoRASet,
    targets: Int,
    ctx: DeviceContext,
) raises -> FlatDirectDoRAGrads:
    var out = krea2_direct_dora_zero_grads(masters)
    var compact = 0
    for bi in range(NBLOCKS):
        for slot in range(KREA2_SLOTS_PER_BLOCK):
            if not _krea2_slot_targeted(slot, targets):
                continue
            var idx = bi * KREA2_SLOTS_PER_BLOCK + slot
            if idx >= len(grads.grads):
                raise Error("_direct_dora_grads_to_host: direct grad list too short")
            var g = grads.grads[idx].copy()
            if not g.d_a:
                raise Error(String("_direct_dora_grads_to_host: missing d_a at flat slot ") + String(idx))
            if not g.d_b:
                raise Error(String("_direct_dora_grads_to_host: missing d_b at flat slot ") + String(idx))
            if not g.d_m:
                raise Error(String("_direct_dora_grads_to_host: missing d_m at flat slot ") + String(idx))
            var dg = DoRAGrads(
                g.d_a.value()[].to_host(ctx),
                g.d_b.value()[].to_host(ctx),
                g.d_m.value()[].to_host(ctx),
                List[Float32](),
            )
            krea2_direct_dora_scatter_slot_grad(out, compact, dg)
            compact += 1
    if compact != len(masters.ad):
        raise Error("_direct_dora_grads_to_host: compact grad count mismatch")
    return out^


def _direct_oft_grads_to_host(
    var grads: Krea2StackDirectOFTGradsT,
    masters: FlatDirectOFTSet,
    targets: Int,
    ctx: DeviceContext,
) raises -> FlatDirectOFTGrads:
    var out = krea2_direct_oft_zero_grads(masters)
    var compact = 0
    for bi in range(NBLOCKS):
        for slot in range(KREA2_SLOTS_PER_BLOCK):
            if not _krea2_slot_targeted(slot, targets):
                continue
            var idx = bi * KREA2_SLOTS_PER_BLOCK + slot
            if idx >= len(grads.grads):
                raise Error("_direct_oft_grads_to_host: direct grad list too short")
            var g = grads.grads[idx].copy()
            if not g.d_vec:
                raise Error(String("_direct_oft_grads_to_host: missing d_vec at flat slot ") + String(idx))
            var og = OFTOTGrads(g.d_vec.value()[].to_host(ctx), List[Float32]())
            krea2_direct_oft_scatter_slot_grad(out, compact, og)
            compact += 1
    if compact != len(masters.ad):
        raise Error("_direct_oft_grads_to_host: compact grad count mismatch")
    return out^


def main() raises:
    var args = argv()
    if len(args) < 3:
        raise Error("usage: train_krea2 <cache.safetensors> <steps> [<config.json>]")
    var cache_path = String(args[1])
    var steps = Int(String(args[2]))
    # Optional resume: args[4] = checkpoint path (.state for FULL resume incl AdamW
    # moments, or a plain PEFT file for WARM start), args[5] = step to resume AT.
    var resume_path = String("")
    var start_step = 0
    if len(args) >= 5:
        resume_path = String(args[4])
    if len(args) >= 6:
        start_step = Int(String(args[5]))

    var ctx = DeviceContext()
    # Optional 3rd arg = a config path (e.g. configs/krea2_boxjana.json); default =
    # the giger krea2.json via krea2_raw(). Lets boxjana use its own config (steps/lr/
    # optimizer/workspace) without touching the giger config.
    var cfg: TrainConfig
    if len(args) >= 4:
        var cfg_path = String(args[3])
        print("[krea2] config:", cfg_path)
        cfg = read_model_config(cfg_path)
    else:
        cfg = krea2_raw()
    var key_prefix = String("")          # real raw.safetensors stores bare torch keys

    print("==== krea2 LoRA TRAINER (Phase 4a, streaming) ====")
    print("cache=", cache_path, " steps=", steps)
    print("rank=", cfg.lora_rank, " alpha=", cfg.lora_alpha, " lr=", cfg.lr,
          " shift=", cfg.timestep_shift, " nblocks=", NBLOCKS,
          " LTMAX=", LTMAX, " LFULL=", LFULL, " (length-bucket pad+mask)",
          " V2_GRAPH=", KREA2_V2_GRAPH)

    # ── open the cache + checkpoint; load the small frozen final-layer once ─────
    var cache = KreaTrainCache.open(cache_path)
    var n = cache.len()
    print("cache samples=", n)
    var st = ShardedSafeTensors.open(cfg.checkpoint)
    var fin = Krea2StreamFinal.load(st, key_prefix, ctx)

    # ── RESIDENT conditioning weights (load ONCE; frozen, small, bf16 — no fp8) ──
    # The conditioning forward (embedders + 4 text-fusion bundles + txtmlp) reads
    # from this resident set instead of re-loading `st` EVERY step (the remaining
    # per-step disk read after the fp8-resident blocks). Always-on, numerically
    # identical (same bf16 weights, just loaded once).
    var cond_w = load_krea2_resident_cond(st, key_prefix, ctx)
    print("resident conditioning weights loaded once (embedders + txtfusion + txtmlp).")

    # ── T2.B fp8-quantized-resident base (gate on cfg.quantized_resident) ───────
    # "fp8_e4m3" = quantize the 28 frozen blocks' 8 matmul weights ONCE to E4M3 +
    # per-row F32 scale, hold resident (~12GB), dequant per block in the step → NO
    # per-step disk re-read. "" / "OFF" (default, C13) = the per-step bf16 disk
    # stream below stays UNTOUCHED (byte-identical to the pre-fp8 path).
    var resident = Optional[Krea2ResidentFp8](None)
    if cfg.quantized_resident == String("fp8_e4m3"):
        print("fp8_e4m3 resident base: quantizing", NBLOCKS, "blocks ONCE at load ...")
        resident = Optional[Krea2ResidentFp8](
            build_krea2_resident_fp8(st, key_prefix, NBLOCKS, ctx)
        )
        print("fp8_e4m3 resident base: DONE (no per-step disk re-read in the step).")
    else:
        print("quantized_resident=", cfg.quantized_resident,
              " (bf16 per-step disk stream — the C13 default path).")

    # ── host LoRA set (authoritative + AdamW moments) ───────────────────────────
    # LoKr/LoHa: host_lora is the (a,b) CARRIER of the LyCORIS masters, then
    # re-materialized after every master AdamW step.
    var lokr_active = cfg.adapter_algo == TRAIN_ADAPTER_ALGO_LOKR
    var loha_active = cfg.adapter_algo == TRAIN_ADAPTER_ALGO_LOHA
    var dora_active = cfg.adapter_algo == TRAIN_ADAPTER_ALGO_DORA
    var oft_active = cfg.adapter_algo == TRAIN_ADAPTER_ALGO_OFT
    var k2_targets = _krea2_train_targets(cfg.lokr_targets)
    var carrier_active = lokr_active or loha_active
    if cfg.adapter_algo == TRAIN_ADAPTER_ALGO_LOCON:
        print("[krea2-locon] network_algorithm=locon: using the linear LoRA-compatible down/up path")
    elif dora_active:
        var dense_bytes = krea2_direct_dense_carrier_bytes(
            NBLOCKS, FEATURES, MLPDIM, HEADS * HEADDIM, KVHEADS * HEADDIM,
            k2_targets,
        )
        var direct_bytes = krea2_direct_dora_preflight(
            NBLOCKS, FEATURES, MLPDIM, HEADS * HEADDIM, KVHEADS * HEADDIM,
            cfg.lora_rank, k2_targets, KREA2_DIRECT_24_GIB, False,
        )
        print(
            "[krea2-dora-direct] dense_carrier_bytes:", dense_bytes,
            " direct_trainable_bytes:", direct_bytes,
            " budget:", KREA2_DIRECT_24_GIB,
        )
    elif oft_active:
        var dense_bytes = krea2_direct_dense_carrier_bytes(
            NBLOCKS, FEATURES, MLPDIM, HEADS * HEADDIM, KVHEADS * HEADDIM,
            k2_targets,
        )
        var direct_bytes = krea2_direct_oft_preflight(
            NBLOCKS, FEATURES, MLPDIM, HEADS * HEADDIM, KVHEADS * HEADDIM,
            4, k2_targets, KREA2_DIRECT_24_GIB,
        )
        print(
            "[krea2-oft-direct] dense_carrier_bytes:", dense_bytes,
            " direct_trainable_bytes:", direct_bytes,
            " budget:", KREA2_DIRECT_24_GIB,
        )
    elif cfg.adapter_algo == TRAIN_ADAPTER_ALGO_BOFT:
        raise Error("krea2 trainer: BOFT is intentionally excluded; use lora, locon, loha, lokr, dora, or oft where wired")
    elif cfg.adapter_algo == TRAIN_ADAPTER_ALGO_FULL:
        raise Error("krea2 trainer: full finetune is not wired in train_krea2; supported here: lora, locon, loha, lokr")
    elif cfg.adapter_algo != TRAIN_ADAPTER_ALGO_LORA and not carrier_active:
        raise Error(
            String("krea2 trainer: network_algorithm=")
            + adapter_algo_name(cfg.adapter_algo)
            + String(" is not wired; supported here: lora, locon, loha, lokr")
        )
    if carrier_active and KREA2_DEVICE_LORA_GRAD:
        raise Error("krea2 LyCORIS carriers require KREA2_DEVICE_LORA_GRAD=False (the carrier chain needs HOST dA/dB)")
    if (carrier_active or dora_active or oft_active) and levers_optimizer_active(cfg):
        raise Error("krea2 LyCORIS direct/carrier masters use host AdamW; levers optimizers are not wired")
    var host_lora = List[LoraAdapter]()
    if not dora_active and not oft_active:
        host_lora = _build_host_lora(cfg.lora_rank, cfg.lora_alpha)
    # RESUME (plain LoRA only): overwrite the B=0-init host_lora with the saved
    # A/B (+ AdamW moments if the .state file is present) and continue at start_step.
    if resume_path != String("") and not dora_active and not oft_active and not carrier_active:
        var resume_scale = cfg.lora_alpha / Float32(cfg.lora_rank)
        host_lora = _krea2_lora_resume(host_lora, resume_scale, resume_path, ctx)
        print("[krea2-resume] reloaded", len(host_lora), "adapters; resuming at step", start_step, "/", steps)
    var n_adapters = NBLOCKS * KREA2_SLOTS_PER_BLOCK
    var lokr_masters = empty_krea2_lokr_set()
    var loha_masters = empty_krea2_loha_set()
    var dora_masters = empty_krea2_direct_dora_set()
    var oft_masters = empty_krea2_direct_oft_set()
    if lokr_active:
        lokr_masters = build_krea2_lokr_set(
            NBLOCKS, FEATURES, MLPDIM, HEADS * HEADDIM, KVHEADS * HEADDIM,
            cfg.lora_rank, cfg.lora_alpha, cfg.lokr_factor,
            cfg.lokr_decompose_both, cfg.lokr_full_matrix, k2_targets,
            UInt64(53) * 7 + 11,
        )
        var carrier_bytes = krea2_lokr_carrier_total_bytes(
            lokr_masters, FEATURES, MLPDIM, HEADS * HEADDIM, KVHEADS * HEADDIM
        )
        print("[krea2-lokr] carrier device bytes:", carrier_bytes, " budget:", LOKR_CARRIER_MAX_DEVICE_BYTES)
        if carrier_bytes > LOKR_CARRIER_MAX_DEVICE_BYTES:
            raise Error(
                String("krea2 LoKr: carrier set needs ") + String(carrier_bytes)
                + " bytes (> budget). Use a small lokr_factor or restrict lokr_targets."
            )
        host_lora = krea2_lokr_carrier_lists(lokr_masters, FEATURES, MLPDIM, HEADS * HEADDIM, KVHEADS * HEADDIM)
        print("[krea2-lokr] carrier set materialized:", len(host_lora), "adapters")
    elif loha_active:
        loha_masters = build_krea2_loha_set(
            NBLOCKS, FEATURES, MLPDIM, HEADS * HEADDIM, KVHEADS * HEADDIM,
            cfg.lora_rank, cfg.lora_alpha, k2_targets,
            UInt64(53) * 11 + 17,
        )
        var loha_bytes = krea2_loha_carrier_total_bytes(
            loha_masters, FEATURES, MLPDIM, HEADS * HEADDIM, KVHEADS * HEADDIM
        )
        print("[krea2-loha] carrier device bytes:", loha_bytes, " budget:", LOKR_CARRIER_MAX_DEVICE_BYTES)
        if loha_bytes > LOKR_CARRIER_MAX_DEVICE_BYTES:
            raise Error(
                String("krea2 LoHa: carrier set needs ") + String(loha_bytes)
                + " bytes (> budget). Reduce lora_rank or restrict lokr_targets."
            )
        host_lora = krea2_loha_carrier_lists(loha_masters, FEATURES, MLPDIM, HEADS * HEADDIM, KVHEADS * HEADDIM)
        print("[krea2-loha] carrier set materialized:", len(host_lora), "adapters")
    elif dora_active:
        print("[krea2-dora-direct] initializing DoRA magnitudes from streamed runtime weights ...")
        dora_masters = _build_krea2_direct_dora_set_streamed(
            st, key_prefix, cfg.lora_rank, cfg.lora_alpha, k2_targets, ctx,
            resident,
        )
        print("[krea2-dora-direct] trainable bytes:", krea2_direct_dora_trainable_bytes(dora_masters),
              " slots:", len(dora_masters.ad))
    elif oft_active:
        oft_masters = build_krea2_direct_oft_set(
            NBLOCKS, FEATURES, MLPDIM, HEADS * HEADDIM, KVHEADS * HEADDIM,
            4, k2_targets,
        )
        print("[krea2-oft-direct] trainable bytes:", krea2_direct_oft_trainable_bytes(oft_masters),
              " slots:", len(oft_masters.ad))
    if not dora_active and not oft_active:
        print("host LoRA adapters=", len(host_lora), " (8 per block)")

    # ── inline sample-during-training ──────────────────────────────────────────
    # Uses the live in-memory LoRA carrier (plain LoRA/LoCon/LoKr/LoHa) and the
    # resident/streamed base already opened above. Direct DoRA/OFT have their own
    # W_eff carriers and are not represented by Krea2StackLora, so fail loud if a
    # config asks for inline sampling there.
    var sample_cfg = SamplePromptConfig()
    var sample_every = cfg.sample_every
    var sample_enabled = sample_every > 0
    if sample_enabled:
        if dora_active or oft_active:
            raise Error(
                "krea2 inline sampler currently supports LoRA/LoCon/LoKr/LoHa "
                + "carriers; set sample_every=0 for direct DoRA/OFT runs"
            )
        if cfg.validation_prompts_file == String(""):
            raise Error("krea2 inline sampler requires validation_prompts_file")
        sample_cfg = read_sample_prompt_config(cfg.validation_prompts_file)
        print(
            "[krea2-sample-inline] enabled every", sample_every,
            "steps; prompts=", len(sample_cfg.prompts),
            " file=", cfg.validation_prompts_file,
        )

    # ── optimizer: AdamW (default, fused) OR a levers optimizer (automagic3 etc.).
    # levers_optimizer_active is False for ADAMW (C13: routes around levers). For
    # AUTOMAGIC3 (boxjana), the levers path runs automagic3_step + its REQUIRED
    # stochastic-rounding bf16 writeback (automagic3_writeback_bf16_sr). State is
    # lazily inited on the first levers step (no alloc for the AdamW default).
    var opt_state = LeversOptimizerState()
    if levers_optimizer_active(cfg):
        levers_optimizer_validate(cfg, String("krea2"))
        print("[krea2] optimizer = LEVERS (optimizer tag", cfg.optimizer, ") — automagic3/etc.")
    else:
        print("[krea2] optimizer = fused AdamW (default)")

    # ── LT bucketing ────────────────────────────────────────────────────────────
    # Process samples LARGEST-LT first so the device memory pool allocates the
    # max-size blocks on step 0 and the smaller steps REUSE them — avoids the
    # measured per-step LT-change fragmentation OOM (going from a smaller LT to a
    # larger one needs new bigger blocks while the smaller ones are still pooled).
    # Precompute LTs (cheap scalar reads), selection-sort the index order desc (n small).
    var lts = List[Int]()
    for i in range(n):
        lts.append(cache.text_len_at(i, ctx))
    var order = List[Int]()
    for i in range(n):
        order.append(i)
    for a in range(n):
        var mx = a
        for b in range(a + 1, n):
            if lts[order[b]] > lts[order[mx]]:
                mx = b
        if mx != a:
            var t = order[a]
            order[a] = order[mx]
            order[mx] = t
    print("LT-bucketed order (largest first): step0 = sample", order[0], "LT", lts[order[0]])

    var seed_base = cfg.seed
    print("")
    print("step  sample  LT   sigma     loss        grad_norm")

    for step in range(start_step, steps):
        var idx = order[step % n]   # LT-bucketed order kept (harmless; padding makes all
        # samples one LFULL size class — the real pool fragmentation fix).
        var sample: KreaTrainSample
        comptime if KREA2_RES_512_SYNTH:
            sample = _synthetic_sample(idx, ctx)   # diagnostic timing arm (random latents)
        else:
            # REAL data: read the cache (LH/LW comptime → 1024px reads the 128×128 cache,
            # 512px=True reads a 64×64-latent 512px cache). sample_padded gives real
            # conditioning + pos padded to LTMAX.
            sample = cache.sample_padded[LH, LW, LTMAX](idx, ctx)
        var lt = sample.text_len    # natural caption length (for the additive pad mask)

        # flow-match t (= blend coeff = model timestep) per step (seed + step stream).
        var sigma = sample_timestep_logit_normal(
            seed_base + UInt64(step), cfg.timestep_shift,
        )
        var noise_seed = seed_base * UInt64(7919) + UInt64(step)

        var _ot = 0
        comptime if KREA2_PHASE_TIMING:
            ctx.synchronize(); _ot = Int(perf_counter_ns())

        if dora_active:
            var dev_dora = krea2_direct_dora_blocks_to_device(
                dora_masters, NBLOCKS, k2_targets, ctx,
            )
            comptime if KREA2_PHASE_TIMING:
                _ot = _phase_ms("host_to_device_dora", _ot, ctx)

            var so_dora = _step_dispatch_dora(
                st, key_prefix,
                sample.clean[], sample.context[], sample.pos[], lt,
                dev_dora, fin, cond_w, sigma, noise_seed, cfg, ctx,
                resident,
            )
            var so_dora_h = so_dora^.to_host(dora_masters, k2_targets, ctx)
            var dnorm = krea2_direct_dora_grad_norm(so_dora_h.grads)
            if dnorm > Float64(cfg.max_grad_norm):
                krea2_direct_dora_clip_grads(so_dora_h.grads, cfg.max_grad_norm / Float32(dnorm))
            krea2_direct_dora_adamw_step(
                dora_masters, so_dora_h.grads, step + 1, cfg.lr,
                cfg.beta1, cfg.beta2, cfg.eps, cfg.weight_decay,
            )
            print("  [krea2-dora-direct] step=", step + 1,
                  " master_grad_norm=", Float32(dnorm),
                  " zero_leg_l1=", krea2_direct_dora_zero_leg_l1(dora_masters))
            comptime if KREA2_PHASE_TIMING:
                _ot = _phase_ms("direct_dora_optimizer", _ot, ctx)

            print(step, "  ", idx, "  ", lt, "  ", sigma, "  ", so_dora_h.loss, "  ", Float32(dnorm))
            ctx.synchronize()

            if cfg.save_every > 0 and (step + 1) % cfg.save_every == 0:
                _ = sys_mkdirs(cfg.workspace_dir)
                var sp = _lora_save_path(cfg, step + 1)
                var nmods = save_krea2_direct_dora(dora_masters, sp, ctx)
                print("  [save] wrote", nmods, "DoRA modules ->", sp)
                comptime if KREA2_KEEP_CHECKPOINTS > 0:
                    var old_step = (step + 1) - KREA2_KEEP_CHECKPOINTS * cfg.save_every
                    if old_step > 0 and old_step % KREA2_CKPT_MILESTONE != 0:
                        var op = _lora_save_path(cfg, old_step)
                        if sys_remove(op) == 0:
                            print("  [prune] removed old checkpoint ->", op)
            continue

        if oft_active:
            var dev_oft = krea2_direct_oft_blocks_to_device(
                oft_masters, NBLOCKS, k2_targets, ctx,
            )
            comptime if KREA2_PHASE_TIMING:
                _ot = _phase_ms("host_to_device_oft", _ot, ctx)

            var so_oft = _step_dispatch_oft(
                st, key_prefix,
                sample.clean[], sample.context[], sample.pos[], lt,
                dev_oft, fin, cond_w, sigma, noise_seed, cfg, ctx,
                resident,
            )
            var so_oft_h = so_oft^.to_host(oft_masters, k2_targets, ctx)
            var onorm = krea2_direct_oft_grad_norm(so_oft_h.grads)
            if onorm > Float64(cfg.max_grad_norm):
                krea2_direct_oft_clip_grads(so_oft_h.grads, cfg.max_grad_norm / Float32(onorm))
            krea2_direct_oft_adamw_step(
                oft_masters, so_oft_h.grads, step + 1, cfg.lr,
                cfg.beta1, cfg.beta2, cfg.eps, cfg.weight_decay,
            )
            print("  [krea2-oft-direct] step=", step + 1,
                  " master_grad_norm=", Float32(onorm),
                  " vec_l1=", krea2_direct_oft_vec_l1(oft_masters))
            comptime if KREA2_PHASE_TIMING:
                _ot = _phase_ms("direct_oft_optimizer", _ot, ctx)

            print(step, "  ", idx, "  ", lt, "  ", sigma, "  ", so_oft_h.loss, "  ", Float32(onorm))
            ctx.synchronize()

            if cfg.save_every > 0 and (step + 1) % cfg.save_every == 0:
                _ = sys_mkdirs(cfg.workspace_dir)
                var sp = _lora_save_path(cfg, step + 1)
                var nmods = save_krea2_direct_oft(oft_masters, sp, ctx)
                print("  [save] wrote", nmods, "OFT modules ->", sp)
                comptime if KREA2_KEEP_CHECKPOINTS > 0:
                    var old_step = (step + 1) - KREA2_KEEP_CHECKPOINTS * cfg.save_every
                    if old_step > 0 and old_step % KREA2_CKPT_MILESTONE != 0:
                        var op = _lora_save_path(cfg, old_step)
                        if sys_remove(op) == 0:
                            print("  [prune] removed old checkpoint ->", op)
            continue

        # device LoRA set for THIS step (small; rebuilt from the host authoritative).
        var dev_lora = _host_to_device_lora(host_lora, ctx)
        comptime if KREA2_PHASE_TIMING:
            _ot = _phase_ms("host_to_device_lora", _ot, ctx)

        var so = _step_dispatch(
            st, key_prefix,
            sample.clean[], sample.context[], sample.pos[], lt,
            dev_lora, fin, cond_w, sigma, noise_seed, cfg, ctx,
            resident,
        )

        # extract flat grad lists, then global-norm clip (max_grad_norm).
        var gn = so.grad_norm
        var gl = _grads_to_lists(so.grads, n_adapters)
        var clip_scale = Float32(1.0)
        comptime if KREA2_GPU_CLIP:
            # FOLD the clip into the AdamW kernel: compute the scale here, skip the
            # 54M-element host _clip_lists, pass the scale to the optimizer (free GPU
            # mul). (gn is the host grad_norm; a GPU on_device_global_norm follow-on
            # would also kill the 62ms host norm loop — separate, smaller.)
            if gn > cfg.max_grad_norm and gn > Float32(0.0):
                clip_scale = cfg.max_grad_norm / gn
        else:
            _clip_lists(gl, gn, cfg.max_grad_norm)
        comptime if KREA2_PHASE_TIMING:
            _ot = _phase_ms("grads_to_lists+clip", _ot, ctx)

        # ── OPTIMIZER SEAM (C13): default AdamW (fused) OR the levers path
        # (automagic3 etc.). levers_optimizer_step_host reads the already-host-clipped
        # gl.d_a/d_b (clip ran above via _clip_lists in the default non-GPU-clip path)
        # and does the automagic3 step + its stochastic-rounding bf16 writeback. The
        # fused AdamW keeps the clip_scale fold. ─────────────────────────────────
        if lokr_active:
            # chain carrier grads → LoKr master grads, host AdamW on masters,
            # re-materialize carriers into host_lora for the next step.
            var mg = krea2_lokr_chain_all(lokr_masters, gl.d_a, gl.d_b)
            var mnorm = krea2_lokr_grad_norm(mg)
            if mnorm > Float64(cfg.max_grad_norm):
                krea2_lokr_clip_grads(mg, cfg.max_grad_norm / Float32(mnorm))
            krea2_lokr_adamw_step(
                lokr_masters, mg, step + 1, cfg.lr,
                cfg.beta1, cfg.beta2, cfg.eps, cfg.weight_decay,
            )
            host_lora = krea2_lokr_carrier_lists(lokr_masters, FEATURES, MLPDIM, HEADS * HEADDIM, KVHEADS * HEADDIM)
            print("  [krea2-lokr] step=", step + 1, " master_grad_norm=", Float32(mnorm),
                  " zero_leg_l1=", krea2_lokr_zero_leg_l1(lokr_masters))
        elif loha_active:
            # chain carrier grads → LoHa master grads, host AdamW on masters,
            # re-materialize carriers into host_lora for the next step.
            var mg = krea2_loha_chain_all(loha_masters, gl.d_a, gl.d_b)
            var mnorm = krea2_loha_grad_norm(mg)
            if mnorm > Float64(cfg.max_grad_norm):
                krea2_loha_clip_grads(mg, cfg.max_grad_norm / Float32(mnorm))
            krea2_loha_adamw_step(
                loha_masters, mg, step + 1, cfg.lr,
                cfg.beta1, cfg.beta2, cfg.eps, cfg.weight_decay,
            )
            host_lora = krea2_loha_carrier_lists(loha_masters, FEATURES, MLPDIM, HEADS * HEADDIM, KVHEADS * HEADDIM)
            print("  [krea2-loha] step=", step + 1, " master_grad_norm=", Float32(mnorm),
                  " zero_leg_l1=", krea2_loha_zero_leg_l1(loha_masters))
        elif levers_optimizer_active(cfg):
            levers_optimizer_step_host(
                cfg, host_lora, gl.d_a, gl.d_b, step + 1, cfg.lr, 0, n_adapters,
                opt_state,
            )
        else:
            fused_lora_adamw_plain_step(
                host_lora, gl.d_a, gl.d_b, 0, n_adapters, step + 1,
                cfg.lr, cfg.beta1, cfg.beta2, cfg.eps, cfg.weight_decay, ctx,
                clip_scale,
            )
        comptime if KREA2_PHASE_TIMING:
            _ot = _phase_ms("optimizer", _ot, ctx)

        print(step, "  ", idx, "  ", lt, "  ", sigma, "  ", so.loss, "  ", gn)
        ctx.synchronize()   # per-STEP async free discipline: reclaim this step's tensors
        # (esp. the ~4.5GB pad mask + the padded sample + fwd/bwd acts) before the next
        # step — else the deferred async frees creep up in the tight LTMAX headroom and
        # OOM (~step 11). The per-block sync handles within-the-stack; this handles steps.

        # ── inline sampler: sample the just-updated live LoRA, no save/reload.
        if sample_enabled and (step + 1) % sample_every == 0:
            _krea2_run_inline_samples[LH_S, LW_S, LTMAX, LFULL_S](
                cache, st, key_prefix, cond_w, fin, host_lora, cfg, sample_cfg,
                step + 1, ctx, resident,
            )
            ctx.synchronize()

        # ── periodic LoRA save (MJ-0805) every cfg.save_every steps ─────────────
        if cfg.save_every > 0 and (step + 1) % cfg.save_every == 0:
            _ = sys_mkdirs(cfg.workspace_dir)   # save_safetensors won't create dirs
            var sp = _lora_save_path(cfg, step + 1)
            var npairs: Int
            if lokr_active:
                npairs = save_krea2_lokr(lokr_masters, sp, ctx)
            elif loha_active:
                npairs = save_krea2_loha(loha_masters, sp, ctx)
            else:
                npairs = save_krea2_lora(host_lora, sp, ctx)
                # FULL-resume sidecar: A/B + AdamW moments (load with args[4]=<sp>.state).
                _ = save_krea2_lora_state(host_lora, sp + String(".state"), ctx)
            print("  [save] wrote", npairs, "LoRA pairs ->", sp)
            # Honor ai-toolkit's max_step_saves_to_keep: prune the checkpoint
            # KREA2_KEEP_CHECKPOINTS saves back, keeping every KREA2_CKPT_MILESTONE-th
            # as a milestone (and the final save is written separately). 0 = keep all.
            comptime if KREA2_KEEP_CHECKPOINTS > 0:
                var old_step = (step + 1) - KREA2_KEEP_CHECKPOINTS * cfg.save_every
                if old_step > 0 and old_step % KREA2_CKPT_MILESTONE != 0:
                    var op = _lora_save_path(cfg, old_step)
                    if sys_remove(op) == 0:
                        print("  [prune] removed old checkpoint ->", op)

    # ── FINAL LoRA save (MJ-0805) — the trained LoRA, re-loadable PEFT ──────────
    _ = sys_mkdirs(cfg.workspace_dir)
    var final_path = _lora_save_path(cfg, steps)
    var n_pairs: Int
    if lokr_active:
        n_pairs = save_krea2_lokr(lokr_masters, final_path, ctx)
    elif loha_active:
        n_pairs = save_krea2_loha(loha_masters, final_path, ctx)
    elif dora_active:
        n_pairs = save_krea2_direct_dora(dora_masters, final_path, ctx)
    elif oft_active:
        n_pairs = save_krea2_direct_oft(oft_masters, final_path, ctx)
    else:
        n_pairs = save_krea2_lora(host_lora, final_path, ctx)
    print("")
    if dora_active:
        print("[save] FINAL DoRA:", n_pairs, "modules ->", final_path)
    elif oft_active:
        print("[save] FINAL OFT:", n_pairs, "modules ->", final_path)
    else:
        print("[save] FINAL LoRA:", n_pairs, "pairs (", n_pairs * 2, "tensors) ->", final_path)

    print("")
    print("VERDICT: ran", steps, "steps. Lead checks loss DROPPING + grad_norm",
          "nonzero + (fits, no OOM = streaming works) + LoRA SAVED (re-loadable PEFT).")
