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
from serenitymojo.ops.tensor_algebra import reshape, concat, slice

# ── shared training pipeline (REUSE) ──────────────────────────────────────────
from serenitymojo.training.train_config import TrainConfig
from serenitymojo.training.train_step import LoraAdapter
from serenitymojo.training.schedule import (
    sample_timestep_logit_normal, flow_match_noise_target,
)
from serenitymojo.training.levers import levers_loss_grad
from serenitymojo.training.lora_adamw_plain_fused import fused_lora_adamw_plain_step

# ── krea2 config + cache reader + LoRA set ────────────────────────────────────
from serenitymojo.models.krea2.config import krea2_raw
from serenitymojo.models.krea2.krea2_cache_reader import (
    KreaTrainCache, KreaTrainSample, krea2_patchify, krea2_build_pos,
    KREA2_LATENT_CHANNELS, KREA2_TXT_LAYERS, KREA2_TXT_DIM,
)
from serenitymojo.models.klein.lora_adapter import make_lora_adapter
from serenitymojo.models.klein.lora_block import (
    LoraAdapterDevice, lora_adapter_to_device,
)

# ── the streaming LoRA stack + carriers ───────────────────────────────────────
from serenitymojo.models.krea2.krea2_block import Krea2BlockLora
from serenitymojo.models.krea2.krea2_stack import (
    Krea2StackLora, Krea2StackForward, Krea2StackLoraGrads,
    Krea2StreamFinal, KREA2_SLOTS_PER_BLOCK,
    krea2_stack_lora_forward_streamed, krea2_stack_lora_backward_streamed,
    krea2_stack_lora_backward_streamed_dev,
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

# ── DIAGNOSTIC: 512px timing arm (KREA2_RES_512, default False) ──────────────
# 1024px (default): clean [1,16,128,128] → IMGLEN=4096 → LFULL=4864 (L=4864).
# 512px (diagnostic): clean [1,16,64,64] → IMGLEN=1024 → LFULL=1792 (L=1792).
# Purpose: confirm the 1024px ~8s/step is the expected cost of the 2.7× longer
# sequence (ai-toolkit trains krea2 at 512px). The step path is IDENTICAL; only
# the shapes change. In 512 mode the trainer SYNTHESIZES the sample (random 64×64
# latents) instead of reading the 128×128 giger cache — step TIME depends on the
# shapes (L=1792), not the values (lead-approved synthetic for the wall-clock test).
# Default False = the 1024px production arm, byte-untouched.
comptime KREA2_RES_512 = False

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
comptime LTMAX = 768
comptime LFULL = LTMAX + IMGLEN           # 4864 — the single comptime arm for ALL samples

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


def main() raises:
    var args = argv()
    if len(args) < 3:
        raise Error("usage: train_krea2 <cache.safetensors> <steps>")
    var cache_path = String(args[1])
    var steps = Int(String(args[2]))

    var ctx = DeviceContext()
    var cfg = krea2_raw()
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
    var host_lora = _build_host_lora(cfg.lora_rank, cfg.lora_alpha)
    var n_adapters = NBLOCKS * KREA2_SLOTS_PER_BLOCK
    print("host LoRA adapters=", len(host_lora), " (8 per block)")

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

    for step in range(steps):
        var idx = order[step % n]   # LT-bucketed order kept (harmless; padding makes all
        # samples one LFULL size class — the real pool fragmentation fix).
        var sample: KreaTrainSample
        comptime if KREA2_RES_512:
            sample = _synthetic_sample(idx, ctx)   # diagnostic 512px timing arm
        else:
            sample = cache.sample_padded[LH, LW, LTMAX](idx, ctx)  # context+pos padded to LTMAX
        var lt = sample.text_len    # natural caption length (for the additive pad mask)

        # flow-match t (= blend coeff = model timestep) per step (seed + step stream).
        var sigma = sample_timestep_logit_normal(
            seed_base + UInt64(step), cfg.timestep_shift,
        )
        var noise_seed = seed_base * UInt64(7919) + UInt64(step)

        var _ot = 0
        comptime if KREA2_PHASE_TIMING:
            ctx.synchronize(); _ot = Int(perf_counter_ns())

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

        # ── LoRA AdamW (default ADAMW; C13 flags-off). Fused plain step over the
        # full host set; mutates a/b + moments in place. ────────────────────────
        fused_lora_adamw_plain_step(
            host_lora, gl.d_a, gl.d_b, 0, n_adapters, step + 1,
            cfg.lr, cfg.beta1, cfg.beta2, cfg.eps, cfg.weight_decay, ctx,
            clip_scale,
        )
        comptime if KREA2_PHASE_TIMING:
            _ot = _phase_ms("optimizer_adamw", _ot, ctx)

        print(step, "  ", idx, "  ", lt, "  ", sigma, "  ", so.loss, "  ", gn)
        ctx.synchronize()   # per-STEP async free discipline: reclaim this step's tensors
        # (esp. the ~4.5GB pad mask + the padded sample + fwd/bwd acts) before the next
        # step — else the deferred async frees creep up in the tight LTMAX headroom and
        # OOM (~step 11). The per-block sync handles within-the-stack; this handles steps.

    print("")
    print("VERDICT: ran", steps, "steps. Lead checks loss DROPPING + grad_norm",
          "nonzero + (fits, no OOM = streaming works).")
