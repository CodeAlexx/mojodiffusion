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
# ── LT-PAD CHOICE (documented) ────────────────────────────────────────────────
# The TRAINING block uses sdpa_nomask (full attention, NO mask) — it does NOT
# support the inference pad-to-LPAD additive mask. Padding all samples to a
# common LPAD would let the zero pad-rows corrupt the real tokens' attention
# (divergent from inference, which masks them). So each sample runs at its EXACT
# LFULL = LT + imglen (NO pad), comptime-monomorphized per distinct LT and
# dispatched by a top-level `match`. The giger cache (4 samples, 1024px,
# imglen=4096) has LT ∈ {458,627,647,558} → 4 monomorphizations.
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
    KreaTrainCache, krea2_patchify, krea2_build_pos,
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
)

# ── frozen conditioning prefix (REUSE the inference krea2_forward pieces) ──────
from serenitymojo.models.dit.krea2_dit import (
    krea2_first, krea2_temb, krea2_tmlp, krea2_tproj, krea2_txtmlp,
    krea2_text_fusion, build_krea2_rope,
    _wb, _scale, _txtf_bundle,
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

# the giger 1024px cache: clean [1,16,128,128] → imglen=(128/2)*(128/2)=4096.
comptime LH = 128
comptime LW = 128
comptime IMGLEN = (LH // 2) * (LW // 2)   # 4096

# ── autograd_v2 backward dispatch seam (Phase 4b adds the engine arm) ─────────
# DEFAULT FALSE = hand-chain krea2_stack_lora_backward_streamed (this file). The
# all-trainers-v2 mandate ([[feedback_all_trainers_autograd_v2]]) flips this in
# Phase 4b once the per-block engine bit-gate lands; default-off stays byte-exact.
comptime KREA2_V2_GRAPH = False


# ══════════════════════════════════════════════════════════════════════════════
# CONDITIONING — the FROZEN krea2_forward prefix (steps 1-11) up to the block
# stack: produce `combined [1,LFULL,F]`, `blk_vec`, `tmlp_out`, and the per-token
# rope (cos,sin). `img` is the PATCHIFIED NOISED latent (caller noises in latent
# space first). All weights stream from `st` (embedders + 4 text-fusion bundles
# are small, loaded once). NO grad here (frozen); no pad (L == LFULL, no-mask
# block path). Mirrors krea2_forward:1358-1454 exactly minus the pad/mask.
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
    st: ShardedSafeTensors, key_prefix: String,
    img: Tensor,            # [1, IMGLEN, 64] F32  PATCHIFIED noised latent
    context: Tensor,        # [1, LT, 12, 2560] BF16
    pos: Tensor,            # [1, LFULL, 3] F32 (txt zeros + img grid)
    t: Tensor,              # [1] F32 timestep (in [0,1])
    ctx: DeviceContext,
) raises -> _Cond:
    # 1) img = first(img) → [1, IMGLEN, F]. img is F32 → cast bf16 to match the bf16
    # `first` head (= reference v.to(bf16) on the head; img feed is bf16 in inference).
    var img_bf = cast_tensor(img, STDtype.BF16, ctx)
    var img_e = krea2_first(
        img_bf, _wb(st, key_prefix + "first.weight", ctx),
        _wb(st, key_prefix + "first.bias", ctx), ctx,
    )

    # 2) t = tmlp(temb(t)) → [1,1,F].
    var te = krea2_temb(t, TDIM, ctx, STDtype.BF16)            # [1, 256]
    var t_vec = krea2_tmlp(
        te,
        _wb(st, key_prefix + "tmlp.0.weight", ctx),
        _wb(st, key_prefix + "tmlp.0.bias", ctx),
        _wb(st, key_prefix + "tmlp.2.weight", ctx),
        _wb(st, key_prefix + "tmlp.2.bias", ctx),
        ctx,
    )
    var t3 = reshape(t_vec, [1, 1, FEATURES], ctx)            # [1,1,F] = tmlp_out

    # 3) blk_vec = tproj(t3) → [1, 6*F].
    var blk_vec = krea2_tproj(
        t3, _wb(st, key_prefix + "tproj.1.weight", ctx),
        _wb(st, key_prefix + "tproj.1.bias", ctx), ctx,
    )
    var blk_vec2 = reshape(blk_vec, [1, 6 * FEATURES], ctx)   # [1, 6*F]

    # 4-5) context = txtfusion(context) (b==1 → txtmask all-ones → refiner no-op).
    var lw0 = _txtf_bundle(st, key_prefix + "txtfusion.layerwise_blocks.0", ctx)
    var lw1 = _txtf_bundle(st, key_prefix + "txtfusion.layerwise_blocks.1", ctx)
    var rf0 = _txtf_bundle(st, key_prefix + "txtfusion.refiner_blocks.0", ctx)
    var rf1 = _txtf_bundle(st, key_prefix + "txtfusion.refiner_blocks.1", ctx)
    var ctx_fused = krea2_text_fusion[LT, NLAYERS_TXT, TXTHEADS, TXTHD](
        context, lw0, lw1,
        _wb(st, key_prefix + "txtfusion.projector.weight", ctx),
        rf0, rf1, Optional[Tensor](None), ctx,
    )                                                          # [1, LT, txtdim]

    # 6) context = txtmlp(context) → [1, LT, F].
    var ctx_proj = krea2_txtmlp(
        ctx_fused,
        _scale(st, key_prefix + "txtmlp.0.scale", ctx),
        _wb(st, key_prefix + "txtmlp.1.weight", ctx),
        _wb(st, key_prefix + "txtmlp.1.bias", ctx),
        _wb(st, key_prefix + "txtmlp.3.weight", ctx),
        _wb(st, key_prefix + "txtmlp.3.bias", ctx),
        ctx,
    )

    # 7-8) combined = cat(context, img, dim=1) → [1, LFULL, F]. context THEN img.
    var combined = concat(1, ctx, ctx_proj, img_e)             # [1, LFULL, F]

    # 9) NO pad (L == LFULL = LT + IMGLEN; the no-mask training block attends all).

    # 10) rope table from pos [1,LFULL,3] → (cos,sin) each [LFULL, HEADDIM/2].
    var pos_flat = reshape(pos, [LFULL * 3], ctx)
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


def _train_one_sample[LT: Int, LFULL: Int](
    st: ShardedSafeTensors, key_prefix: String,
    clean: Tensor,          # [1, 16, LH, LW] F32 normalized latent
    context: Tensor,        # [1, LT, 12, 2560] BF16
    pos: Tensor,            # [1, LFULL, 3] F32
    lora: Krea2StackLora, fin: Krea2StreamFinal,
    sigma: Float32,         # flow-match t (= blend coeff = model timestep), in [0,1]
    noise_seed: UInt64,
    cfg: TrainConfig,
    ctx: DeviceContext,
) raises -> _StepOut:
    # ── flow-match noise in LATENT space (before patchify; krea2.py order) ──────
    # x_t = (1-sigma)*clean + sigma*noise ; target = noise - clean  (krea2.py:403).
    var noise = _gaussian_like(clean, noise_seed, ctx)        # [1,16,LH,LW] F32
    var fm = flow_match_noise_target(clean, sigma, noise, ctx)
    var noised_lat = fm.x_t.clone(ctx)                        # [1,16,LH,LW]
    # patchify the NOISED latent → img [1, IMGLEN, 64] (== krea2_forward.img).
    var img = krea2_patchify[LH, LW](noised_lat, ctx)
    # target on the IMAGE tokens, patchified the same way → [1, IMGLEN, 64].
    var target_img = krea2_patchify[LH, LW](fm.target, ctx)

    # ── conditioning (frozen prefix) → combined / blk_vec / tmlp_out / rope ─────
    var t1 = _t_scalar(sigma, ctx)                            # [1] F32 timestep
    var cond = _build_conditioning[LT, LFULL](
        st, key_prefix, img, context, pos, t1, ctx,
    )

    # ── streaming stack forward (txtlen = LT, imglen = IMGLEN) ──────────────────
    var fwd = krea2_stack_lora_forward_streamed[LFULL, HEADS, KVHEADS, HEADDIM](
        cond.combined, cond.blk_vec, cond.tmlp_out,
        st, key_prefix, NBLOCKS, lora, fin,
        cond.cos, cond.sin, EPS, LT, IMGLEN, ctx,
    )

    # ── flow-match MSE loss (levers; default MSE) on the image-token velocity ───
    var pred_h = fwd.velocity[].to_host(ctx)                  # [IMGLEN*64]
    var tgt_h = target_img.to_host(ctx)
    var lg = levers_loss_grad(pred_h, tgt_h, sigma, cfg)
    var loss = lg.loss
    var d_velocity = Tensor.from_host(
        lg.d_pred, [1, IMGLEN, OUT_CH], STDtype.F32, ctx,
    )

    # ── streaming stack backward (hand-chain default; v2 arm Phase 4b) ──────────
    var grads: Krea2StackLoraGrads
    comptime if KREA2_V2_GRAPH:
        # Phase 4b: autograd_v2 engine arm (drop-in, same conductor loop + slots).
        raise Error("KREA2_V2_GRAPH path is Phase 4b — not wired yet")
    else:
        grads = krea2_stack_lora_backward_streamed[LFULL, HEADS, KVHEADS, HEADDIM](
            d_velocity, cond.blk_vec, cond.tmlp_out,
            st, key_prefix, NBLOCKS, lora, fin, fwd,
            cond.cos, cond.sin, EPS, ctx,
        )

    var gn = _grad_norm(grads)
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


# ══════════════════════════════════════════════════════════════════════════════
# DRIVER — one step dispatched on the sample's LT (the comptime monomorphizations).
# The giger cache has exactly these 4 (LT, LFULL) pairs. Returns the _StepOut.
# ══════════════════════════════════════════════════════════════════════════════
def _step_dispatch(
    st: ShardedSafeTensors, key_prefix: String,
    clean: Tensor, context: Tensor, pos: Tensor, lt: Int,
    lora: Krea2StackLora, fin: Krea2StreamFinal,
    sigma: Float32, noise_seed: UInt64, cfg: TrainConfig, ctx: DeviceContext,
) raises -> _StepOut:
    # comptime LT/LFULL arms — one monomorphization per distinct caption length in
    # the giger cache. Add an arm here for any new LT (fail-loud otherwise).
    if lt == 458:
        return _train_one_sample[458, 458 + IMGLEN](
            st, key_prefix, clean, context, pos, lora, fin, sigma, noise_seed, cfg, ctx)
    elif lt == 558:
        return _train_one_sample[558, 558 + IMGLEN](
            st, key_prefix, clean, context, pos, lora, fin, sigma, noise_seed, cfg, ctx)
    elif lt == 627:
        return _train_one_sample[627, 627 + IMGLEN](
            st, key_prefix, clean, context, pos, lora, fin, sigma, noise_seed, cfg, ctx)
    elif lt == 647:
        return _train_one_sample[647, 647 + IMGLEN](
            st, key_prefix, clean, context, pos, lora, fin, sigma, noise_seed, cfg, ctx)
    else:
        raise Error(
            String("train_krea2: no comptime LT arm for LT=") + String(lt)
            + " (giger cache has {458,558,627,647}; add an arm for a new bucket)"
        )


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
          " V2_GRAPH=", KREA2_V2_GRAPH)

    # ── open the cache + checkpoint; load the small frozen final-layer once ─────
    var cache = KreaTrainCache.open(cache_path)
    var n = cache.len()
    print("cache samples=", n)
    var st = ShardedSafeTensors.open(cfg.checkpoint)
    var fin = Krea2StreamFinal.load(st, key_prefix, ctx)

    # ── host LoRA set (authoritative + AdamW moments) ───────────────────────────
    var host_lora = _build_host_lora(cfg.lora_rank, cfg.lora_alpha)
    var n_adapters = NBLOCKS * KREA2_SLOTS_PER_BLOCK
    print("host LoRA adapters=", len(host_lora), " (8 per block)")

    var seed_base = cfg.seed
    print("")
    print("step  sample  LT   sigma     loss        grad_norm")

    for step in range(steps):
        var idx = step % n
        var sample = cache.sample[LH, LW](idx, ctx)
        var lt = sample.text_len

        # flow-match t (= blend coeff = model timestep) per step (seed + step stream).
        var sigma = sample_timestep_logit_normal(
            seed_base + UInt64(step), cfg.timestep_shift,
        )
        var noise_seed = seed_base * UInt64(7919) + UInt64(step)

        # device LoRA set for THIS step (small; rebuilt from the host authoritative).
        var dev_lora = _host_to_device_lora(host_lora, ctx)

        var so = _step_dispatch(
            st, key_prefix,
            sample.clean[], sample.context[], sample.pos[], lt,
            dev_lora, fin, sigma, noise_seed, cfg, ctx,
        )

        # extract flat grad lists, then global-norm clip (max_grad_norm).
        var gn = so.grad_norm
        var gl = _grads_to_lists(so.grads, n_adapters)
        _clip_lists(gl, gn, cfg.max_grad_norm)

        # ── LoRA AdamW (default ADAMW; C13 flags-off). Fused plain step over the
        # full host set; mutates a/b + moments in place. ────────────────────────
        fused_lora_adamw_plain_step(
            host_lora, gl.d_a, gl.d_b, 0, n_adapters, step + 1,
            cfg.lr, cfg.beta1, cfg.beta2, cfg.eps, cfg.weight_decay, ctx,
        )

        print(step, "  ", idx, "  ", lt, "  ", sigma, "  ", so.loss, "  ", gn)

    print("")
    print("VERDICT: ran", steps, "steps. Lead checks loss DROPPING + grad_norm",
          "nonzero + (fits, no OOM = streaming works).")
