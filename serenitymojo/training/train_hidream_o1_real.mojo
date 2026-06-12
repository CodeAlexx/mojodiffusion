# training/train_hidream_o1_real.mojo — HiDream-O1 LoRA trainer (campaign P2,
# HIDREAM_O1_TRAINING_CAMPAIGN.md; DiffSynth-Studio = the recipe reference).
#
# RECIPE (DiffSynth, read in full — loss.py FlowMatchSFTLoss + flow_match.py
# set_timesteps_hidream_o1_image/set_training_weight + model_fn line 387-424
# + examples/hidream_o1_image/model_training):
#   sigmas  = linspace(1,0,1001)[:1000] then shift-3 remap 3s/(1+2s)
#   t_id    ~ UNIFORM over the 1000 training timesteps
#   noise   = randn * 7.5  (noise_scale; scaled BEFORE add_noise)
#   noisy   = (1-sigma)*clean + sigma*noise        [pixel patch space]
#   model_t = 1 - sigma                            [the DiT t-embed input]
#   x_pred  = DiT(...)                             [O1 is X-PREDICTION]
#   v_pred  = (noisy - x_pred) / sigma
#   target  = noise - clean
#   loss    = MSE(v_pred, target) * w(t_id);  dL/dx_pred = -w*(2/N)*diff/sigma
#   w(i)    : y=exp(-2*((t_i-500)/1000)^2); y-=min(y); w = y*(1000/sum(y))
#   LoRA    : rank 32 default, alpha=rank (scale 1.0), lr 1e-4, slots
#             q/k/v/o/gate/up/down on all 36 layers.
#
# DATA: stage-A dir (scripts/ideogram4_stage_images.py): images.safetensors
# (image.<i> [1,3,512,512] F32 [-1,1]) + caption.<i>.txt (RAW captions — the
# HiDream t2i chat template is applied HERE via the verified inference
# builders, pipeline/hidream_o1_cfg.mojo). No VAE, no text-feature cache.
#
# MEMORY (24 GB): 36 layers resident bf16; RECOMPUTE-CHECKPOINT backward —
# forward keeps only each block's INPUT (37 x [1,512,4096] bf16 ~ 155 MB);
# the backward loop re-runs the P1 block forward to rebuild the tape then
# calls the gated hidream_o1_block_lora_backward (torch parity cos>=1-4e-13).
#
# P2 SCOPE NOTES (speed pass comes later, correctness first):
#   * optimizer = host AdamW over the 504 adapter tensors (fused resident
#     AdamW is the klein/zimage speed lever, not a correctness need);
#   * SDPA = math path (flash for hidream is a later lever);
#   * mask + mrope tables rebuilt per step (host; cacheable later).
#
# GATE (P2, ideogram4 discipline): loss sane class + DECREASING + every
# LoRA-B nonzero after N steps.
#
# Run: train_hidream_o1_real <stage_dir> <steps> [lr] [rank] [out_dir]
#                            [ema_decay] [config.json] [grad_dump.safetensors]
#
# ── T2.B QUANTIZED-RESIDENT BASE (config "quantized_resident", default OFF) ──
# "fp8_e4m3": the 7 big linears per layer are quantized ONCE at load to E4M3
# bytes + per-output-row F32 scale (ops/fp8_quant.mojo encode; decode is the
# parity-gated Ideogram-4 dequant, ops/fp8.mojo) and kept resident at
# 1 byte/param (~6.9 GB saved vs bf16-resident). Each block dequants to bf16
# right before compute (forward AND the recompute backward use the same
# decode) and frees the bf16 copy when the block's weights struct drops.
# Norms / small (<1MB) tensors stay bf16. LoRA adapters, optimizer state and
# activations are bf16/F32 exactly as before. NEW NUMERICS CLASS: gated by a
# 10-step fp8-vs-bf16 loss-trajectory cosine >= 0.999 + step-1 adapter-grad
# cosine >= 0.999 (argv[8] grad dump). Flag OFF reproduces the 3-step anchor
# 0.05885428/0.33308488/0.5214583 EXACTLY (C13).
#
# ── T1 RUNTIME CONFIG + PRECEDENCE (Tier-1 lever wiring, levers.mojo) ────────
# Optional trailing argv [config.json] parses via io/train_config_reader.mojo
# read_model_config (no model-dims keys required — missing keys keep
# TrainConfig defaults). Precedence rules:
#   * argv WINS for steps / lr / rank / out_dir when explicitly given
#     (non-"-"); a "-" placeholder defers to the config (max_steps /
#     learning_rate / lora_rank) when one is present, else to the compiled
#     defaults (1e-4 / 32). CAVEAT: with a config present, a "-" argv takes
#     the TrainConfig value even when the JSON omits the key (reader cannot
#     distinguish absent-from-file from default; TrainConfig lora_rank
#     default is 16, NOT this trainer's 32 — set lora_rank in the config).
#   * config WINS for EMA when it enables it ("ema":"CPU"/"EMA"): the full
#     SimpleTuner schedule (ema_decay cap, ema_min_decay floor,
#     ema_update_after_step, ema_update_step_interval) replaces the argv
#     ema_decay. A config WITHOUT ema enabled leaves the argv ema_decay
#     lever working (every-step, floor 0 — the pre-config behavior).
#   * config-only levers: caption_dropout_prob (comptime
#     HIDREAM_CAPTION_DROPOUT stays the no-config fallback), T1.A loss
#     levers (loss_fn / huber_delta / smooth_l1_beta / min_snr_gamma_flow),
#     T1.C optimizer levers (optimizer.optimizer ADAFACTOR /
#     SCHEDULE_FREE_ADAMW + beta1/beta2/eps/weight_decay — the default
#     fused-AdamW path reads cfg hypers too; TrainConfig defaults equal the
#     old literals 0.9/0.999/1e-8/0.01, so the no-config path is unchanged).
#   * the DiffSynth gauss-shift recipe weight wt(t_id) ALWAYS applies to
#     loss AND grad — it is the MODEL RECIPE, not a lever; the levers loss
#     path multiplies it on top of levers_loss_grad.
#   * no LR scheduler in this trainer: step_lr == the constant resolved lr
#     (adafactor consumes it; schedule-free reads cfg.lr raw — the resolved
#     lr is written back into cfg.lr so argv precedence holds there too).
# Template: serenitymojo/configs/hidream_o1.json (all levers default-off).

from std.gpu.host import DeviceContext
from std.math import sqrt, exp, log as flog, cos as fcos, sin as fsin, pi
from std.memory import ArcPointer
from std.collections import Optional
from std.time import perf_counter_ns
from std.sys import argv
from std.os import makedirs

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.linalg_backward import linear_backward_dx
from serenitymojo.ops.norm import rms_norm
from serenitymojo.ops.norm_backward import rms_norm_backward_dx
from serenitymojo.ops.layout import patchify
from serenitymojo.ops.tensor_algebra import concat, slice as ta_slice
from serenitymojo.tokenizer.tokenizer import Qwen3Tokenizer
from serenitymojo.models.dit.hidream_o1 import (
    HiDreamO1Config,
    HiDreamO1Offloaded,
    _build_mrope_tables,
    _replicate_heads,
    _build_prefix_causal_mask_padded,
    _scatter_row,
)
from serenitymojo.offload.block_loader import Block
from serenitymojo.models.zimage.lora_block import ZImageLoraAdapterDevice
from serenitymojo.io.train_config_reader import read_model_config
from serenitymojo.training.train_config import TrainConfig
from serenitymojo.training.progress_display import print_trainer_progress
from serenitymojo.training.levers import (
    caption_dropout_pick, levers_loss_active, levers_loss_grad,
    LeversOptimizerState, levers_optimizer_active, levers_optimizer_step,
    levers_optimizer_validate, levers_optimizer_eval_for_save,
    levers_optimizer_train_after_save,
)
from serenitymojo.training.train_step import LoraAdapter
from serenitymojo.training.lora_adamw_plain_fused import (
    LoraAdamWPlainDeviceState,
    lora_adamw_plain_device_state_init,
    fused_lora_adamw_plain_step_resident,
)
from serenitymojo.training.lora_ema import (
    LoraEmaState, lora_ema_track, ema_update,
    ema_shadow_a_bf16, ema_shadow_b_bf16, ema_path_for_lora,
)
from serenitymojo.models.dit.hidream_o1_train_block import (
    HiDreamO1BlockWeights,
    HiDreamO1BlockLora,
    hidream_o1_block_lora_forward,
    hidream_o1_block_lora_backward,
)
# T2.B fp8-quantized-resident base (default-off): encode = ops/fp8_quant.mojo
# (new, RNE per-row absmax), decode = the parity-gated Ideogram-4 dequant.
from serenitymojo.ops.fp8 import fp8_e4m3_dequant_perrow_to_bf16
from serenitymojo.ops.fp8_quant import fp8_e4m3_rowscale, fp8_e4m3_encode_perrow
from serenitymojo.pipeline.hidream_o1_cfg import build_t2i_input, T2ISample

comptime TArc = ArcPointer[Tensor]

comptime S_TEXT = 256
comptime HP = 16
comptime WP = 16
comptime IMG_L = HP * WP          # 256
comptime S = S_TEXT + IMG_L       # 512
comptime D = 4096
comptime H = 32
comptime HKV = 8
comptime Dh = 128
comptime F = 12288
comptime LAYERS = 36
comptime PATCH = 32
comptime PATCH_VEC = 3072         # 32*32*3
comptime NOISE_SCALE = Float32(7.5)
comptime SHIFT = Float64(3.0)
comptime EPS = Float32(1.0e-6)
comptime SEED = UInt64(42)
# T1.D caption dropout probability — the NO-CONFIG fallback default. When a
# [config.json] argv is present, cfg.caption_dropout_prob replaces this (see
# the runtime-config precedence block in the header).
comptime HIDREAM_CAPTION_DROPOUT = Float32(0.0)
comptime MODEL_DIR = "/home/alex/HiDream-O1-Image-Dev-weights"
comptime TOK_PATH = "/home/alex/HiDream-O1-Image-Dev-weights/tokenizer.json"


def _slot_dims(slot: Int) raises -> List[Int]:
    """[in_f, out_f] for slot q,k,v,o,gate,up,down."""
    if slot == 0:
        return [D, H * Dh]
    if slot == 1 or slot == 2:
        return [D, HKV * Dh]
    if slot == 3:
        return [H * Dh, D]
    if slot == 4 or slot == 5:
        return [D, F]
    return [F, D]


def _slot_name(slot: Int) raises -> String:
    var names: List[String] = [
        String("self_attn.q_proj"), String("self_attn.k_proj"),
        String("self_attn.v_proj"), String("self_attn.o_proj"),
        String("mlp.gate_proj"), String("mlp.up_proj"), String("mlp.down_proj"),
    ]
    return names[slot].copy()


def _lcg_pattern(n: Int, seed: UInt64, amp: Float32) -> List[Float32]:
    var out = List[Float32]()
    var state = seed
    for _ in range(n):
        state = state * 6364136223846793005 + 1442695040888963407
        var u = Float32(Int(state >> 40)) * Float32(1.0 / 16777216.0)
        out.append((u * Float32(2.0) - Float32(1.0)) * amp)
    return out^


def _gauss(n: Int, seed: UInt64) -> List[Float32]:
    """Standard normals, Box-Muller with the repo-blessed (>>12)/2^52 uniforms
    (noise_stats_smoke discipline)."""
    var out = List[Float32]()
    var state = seed
    var i = 0
    while i < n:
        state = state * 6364136223846793005 + 1442695040888963407
        var u1 = Float64(Int((state >> 12) & 0xFFFFFFFFFFFFF)) / Float64(1 << 52)
        state = state * 6364136223846793005 + 1442695040888963407
        var u2 = Float64(Int((state >> 12) & 0xFFFFFFFFFFFFF)) / Float64(1 << 52)
        if u1 < 1.0e-300:
            u1 = 1.0e-300
        var r = sqrt(Float64(-2.0) * flog(u1))
        out.append(Float32(r * fcos(Float64(2.0) * pi * u2)))
        i += 1
        if i < n:
            out.append(Float32(r * fsin(Float64(2.0) * pi * u2)))
            i += 1
    return out^


struct SigWeights(Movable):
    var sigmas: List[Float32]
    var weights: List[Float32]

    def __init__(out self, var sigmas: List[Float32], var weights: List[Float32]):
        self.sigmas = sigmas^
        self.weights = weights^


def _sigmas_and_weights() raises -> SigWeights:
    """The 1000 training sigmas (shift-3) + gauss-shift weights — DiffSynth
    set_timesteps(training=True) + set_training_weight, ported verbatim."""
    var sig = List[Float32]()
    var sig64 = List[Float64]()
    for i in range(1000):
        var s = 1.0 - Float64(i) / 1000.0
        var sh = SHIFT * s / (1.0 + (SHIFT - 1.0) * s)
        sig64.append(sh)
        sig.append(Float32(sh))
    var ys = List[Float64]()
    var ymin = 1.0e300
    for i in range(1000):
        var t = sig64[i] * 1000.0
        var y = exp(-2.0 * ((t - 500.0) / 1000.0) * ((t - 500.0) / 1000.0))
        ys.append(y)
        if y < ymin:
            ymin = y
    var ysum = 0.0
    for i in range(1000):
        ys[i] = ys[i] - ymin
        ysum += ys[i]
    var w = List[Float32]()
    for i in range(1000):
        w.append(Float32(ys[i] * (1000.0 / ysum)))
    return SigWeights(sig^, w^)


def _read_text(path: String) raises -> String:
    var f = open(path, "r")
    var s = f.read()
    f.close()
    return s^


def _bw(block: Block, p: String, suffix: String) raises -> TArc:
    var full = p + suffix
    if full not in block:
        raise Error("train_hidream_o1: missing block weight " + full)
    return block[full].copy()


def _block_weights(block: Block, li: Int) raises -> HiDreamO1BlockWeights:
    var p = String("model.language_model.layers.") + String(li)
    return HiDreamO1BlockWeights(
        _bw(block, p, ".input_layernorm.weight"),
        _bw(block, p, ".self_attn.q_proj.weight"),
        _bw(block, p, ".self_attn.k_proj.weight"),
        _bw(block, p, ".self_attn.v_proj.weight"),
        _bw(block, p, ".self_attn.q_norm.weight"),
        _bw(block, p, ".self_attn.k_norm.weight"),
        _bw(block, p, ".self_attn.o_proj.weight"),
        _bw(block, p, ".post_attention_layernorm.weight"),
        _bw(block, p, ".mlp.gate_proj.weight"),
        _bw(block, p, ".mlp.up_proj.weight"),
        _bw(block, p, ".mlp.down_proj.weight"),
    )


# ── T2.B fp8-quantized-resident helpers (default-off; flag path only) ────────
# Quantize-ONCE-at-load: the 7 big linears of a layer become E4M3 bytes (stored
# under the original tensor name) + per-output-row F32 scales (name +
# "_scale") — the exact layout the parity-gated Ideogram-4 inference dequant
# consumes. Norms / small (<1MB) tensors keep bf16 (standard practice; matches
# how ideogram4 fp8 checkpoints leave norms/embeddings unquantized — only
# Linear weights carry weight_scale siblings).
comptime _FP8_MIN_BYTES = 1 << 20  # tensors under 1MB (bf16) stay bf16


def _quantize_block_fp8(block: Block, li: Int, ctx: DeviceContext) raises -> Block:
    """Re-key one resident layer: big 2-D linears → fp8 bytes + '_scale'; the
    rest copied through as bf16. The input block's bf16 linears are freed when
    the caller drops it (Arc refcount)."""
    var p = String("model.language_model.layers.") + String(li)
    var lin: List[String] = [
        String(".self_attn.q_proj.weight"), String(".self_attn.k_proj.weight"),
        String(".self_attn.v_proj.weight"), String(".self_attn.o_proj.weight"),
        String(".mlp.gate_proj.weight"), String(".mlp.up_proj.weight"),
        String(".mlp.down_proj.weight"),
    ]
    var keep: List[String] = [
        String(".input_layernorm.weight"), String(".self_attn.q_norm.weight"),
        String(".self_attn.k_norm.weight"),
        String(".post_attention_layernorm.weight"),
    ]
    var out = Block()
    for ref sfx in keep:
        var full = p + sfx
        if full not in block:
            raise Error("train_hidream_o1: missing block weight " + full)
        out[full] = block[full].copy()
    for ref sfx in lin:
        var full = p + sfx
        if full not in block:
            raise Error("train_hidream_o1: missing block weight " + full)
        var t = block[full].copy()
        var sh = t[].shape()
        if len(sh) != 2 or t[].nbytes() < _FP8_MIN_BYTES or t[].dtype() != STDtype.BF16:
            # <1MB / non-2D / non-bf16: keep as-is (the standard skip rule).
            out[full] = t.copy()
            continue
        var scale_t = fp8_e4m3_rowscale(t[], ctx)
        var bytes_t = fp8_e4m3_encode_perrow(t[], scale_t, ctx)
        out[full] = ArcPointer(bytes_t^)
        out[full + "_scale"] = ArcPointer(scale_t^)
    return out^


def _bw_q(block: Block, p: String, suffix: String, ctx: DeviceContext) raises -> TArc:
    """fp8-aware weight fetch: dequantize (per-row, the gated Ideogram-4
    decode) when a '_scale' sibling exists; otherwise pass the bf16 Arc
    through. The dequantized bf16 tensor is OWNED by the returned TArc — it is
    freed when the per-block weights struct drops at the end of the block."""
    var full = p + suffix
    if full not in block:
        raise Error("train_hidream_o1: missing block weight " + full)
    if (full + "_scale") in block:
        return TArc(fp8_e4m3_dequant_perrow_to_bf16(
            block[full][], block[full + "_scale"][], ctx,
        ))
    return block[full].copy()


def _block_weights_q(
    block: Block, li: Int, fp8_on: Bool, ctx: DeviceContext,
) raises -> HiDreamO1BlockWeights:
    """Dispatch: flag-off → the EXACT existing `_block_weights` path (C13);
    flag-on → per-block dequant of the fp8-resident linears. Forward and the
    recompute backward both go through here, so backward uses the SAME
    dequantized weights as forward."""
    if not fp8_on:
        return _block_weights(block, li)
    var p = String("model.language_model.layers.") + String(li)
    return HiDreamO1BlockWeights(
        _bw(block, p, ".input_layernorm.weight"),
        _bw_q(block, p, ".self_attn.q_proj.weight", ctx),
        _bw_q(block, p, ".self_attn.k_proj.weight", ctx),
        _bw_q(block, p, ".self_attn.v_proj.weight", ctx),
        _bw(block, p, ".self_attn.q_norm.weight"),
        _bw(block, p, ".self_attn.k_norm.weight"),
        _bw_q(block, p, ".self_attn.o_proj.weight", ctx),
        _bw(block, p, ".post_attention_layernorm.weight"),
        _bw_q(block, p, ".mlp.gate_proj.weight", ctx),
        _bw_q(block, p, ".mlp.up_proj.weight", ctx),
        _bw_q(block, p, ".mlp.down_proj.weight", ctx),
    )


def _adamw_host(
    mut p: List[Float32], g: List[Float32],
    mut m: List[Float32], mut v: List[Float32], m_off: Int,
    step: Int, lr: Float32,
) :
    """Plain AdamW (beta 0.9/0.999, eps 1e-8, wd 0.01) on a host mirror."""
    var b1 = Float64(0.9)
    var b2 = Float64(0.999)
    var bc1 = 1.0 - b1 ** Float64(step)
    var bc2 = 1.0 - b2 ** Float64(step)
    for i in range(len(p)):
        var gi = Float64(g[i])
        var mi = b1 * Float64(m[m_off + i]) + (1.0 - b1) * gi
        var vi = b2 * Float64(v[m_off + i]) + (1.0 - b2) * gi * gi
        m[m_off + i] = Float32(mi)
        v[m_off + i] = Float32(vi)
        var upd = (mi / bc1) / (sqrt(vi / bc2) + 1.0e-8)
        p[i] = Float32(Float64(p[i]) - Float64(lr) * (upd + 0.01 * Float64(p[i])))


def main() raises:
    var args = argv()
    if len(args) < 3:
        raise Error(
            "usage: train_hidream_o1_real <stage_dir> <steps> [lr] [rank]"
            " [out_dir] [ema_decay] [config.json] [grad_dump.safetensors]"
        )
    var stage_dir = String(args[1])
    # Optional argv[8]: dump the step-1 adapter grads (g_a.<k>/g_b.<k>, F32)
    # to a safetensors for offline gate comparison (T2.B grad-cosine gate).
    # "-" / absent = off (no behavior change).
    var grad_dump = String("")
    if len(args) > 8 and String(args[8]) != "-":
        grad_dump = String(args[8])

    # ── T1 runtime config (optional trailing argv; precedence in header) ─────
    var train_cfg = TrainConfig.default()
    var has_config = False
    if len(args) > 7 and String(args[7]) != "-":
        train_cfg = read_model_config(String(args[7]))
        has_config = True
        print("[config] ", String(args[7]))

    # argv wins for steps/lr/rank/out_dir when non-"-"; "-" defers to config.
    var steps: Int
    if String(args[2]) == "-":
        if not has_config or train_cfg.max_steps <= 0:
            raise Error(
                "train_hidream_o1: steps '-' needs a config with max_steps > 0"
            )
        steps = train_cfg.max_steps
    else:
        steps = atol(String(args[2]))
    var lr = Float32(1.0e-4)
    if has_config:
        lr = train_cfg.lr
    if len(args) > 3 and String(args[3]) != "-":
        lr = Float32(atof(String(args[3])))
    var rank = 32
    if has_config:
        rank = train_cfg.lora_rank
    if len(args) > 4 and String(args[4]) != "-":
        rank = atol(String(args[4]))
    var out_dir = String("/home/alex/mojodiffusion/output/hidream_o1_lora")
    if len(args) > 5 and String(args[5]) != "-":
        out_dir = String(args[5])
    # T1.B EMA lever, argv form: decay cap, 0 = off (config EMA wins below).
    var ema_decay = Float32(0.0)
    if len(args) > 6 and String(args[6]) != "-":
        ema_decay = Float32(atof(String(args[6])))
    # Write the EFFECTIVE recipe back into cfg so levers that read cfg
    # directly see argv-resolved values (schedule-free consumes cfg.lr RAW —
    # levers.mojo T1.C LR SEMANTICS).
    train_cfg.lr = lr
    train_cfg.lora_rank = rank
    train_cfg.max_steps = steps
    # T1.D: config replaces the comptime fallback when present.
    var caption_dropout_p = HIDREAM_CAPTION_DROPOUT
    if has_config:
        caption_dropout_p = train_cfg.caption_dropout_prob
    # T1.C: fail-loud lever validation (unsupported optimizer tags already
    # failed at config load; this re-asserts for configs built in code).
    levers_optimizer_validate(train_cfg, String("HiDream-O1 trainer"))
    if levers_optimizer_active(train_cfg):
        print("[optimizer] levers dispatch active, tag ", train_cfg.optimizer)
    # T2.B quantized-resident flag (config-only lever, default-off). The
    # reader already fail-louds on unknown tags; re-assert for configs built
    # in code (same discipline as levers_optimizer_validate).
    var quant_tag = String("")
    if has_config:
        quant_tag = train_cfg.quantized_resident.copy()
    if quant_tag == String("OFF"):
        quant_tag = String("")
    if quant_tag != String("") and quant_tag != String("fp8_e4m3"):
        raise Error(
            "train_hidream_o1: unsupported quantized_resident '" + quant_tag
            + "' (supported: OFF, fp8_e4m3)"
        )
    var fp8_resident = quant_tag == String("fp8_e4m3")
    if fp8_resident:
        print(
            "[quant] fp8_e4m3-resident base linears (per-row scale, quantize"
            " once at load, dequant per block; norms stay bf16)"
        )
    makedirs(out_dir, exist_ok=True)

    var ctx = DeviceContext()
    var cfg = HiDreamO1Config.dev_8b()

    print("[load] HiDream-O1 (offloaded: resident shared + streamed layers)")
    var dit = HiDreamO1Offloaded[S].load(String(MODEL_DIR), cfg, ctx)
    var tok = Qwen3Tokenizer(String(TOK_PATH))

    # ── LoRA (resident-set, the zimage v2 pattern): host LoraAdapter pack ->
    # ONE persistent device P/M/V upload; the model's adapter tensors are
    # zero-copy sub-buffer VIEWS into state.dev_p, so the in-place fused
    # AdamW update IS the next step's weights (no per-step re-upload; the
    # 98 s/step host-AdamW P2 placeholder measured 2026-06-11 dies here).
    var host_ads = List[LoraAdapter]()
    for li in range(LAYERS):
        for sl in range(7):
            var dims = _slot_dims(sl)
            var in_f = dims[0]
            var out_f = dims[1]
            var amp = Float32(1.0) / sqrt(Float32(in_f))
            var a_h = _lcg_pattern(rank * in_f, UInt64(li * 7 + sl + 1), amp)
            var b_h = List[Float32]()
            for _ in range(out_f * rank):
                b_h.append(0.0)
            var z1 = List[Float32]()
            for _ in range(rank * in_f):
                z1.append(0.0)
            var z2 = List[Float32]()
            for _ in range(rank * in_f):
                z2.append(0.0)
            var z3 = List[Float32]()
            for _ in range(out_f * rank):
                z3.append(0.0)
            var z4 = List[Float32]()
            for _ in range(out_f * rank):
                z4.append(0.0)
            host_ads.append(LoraAdapter(
                a_h^, b_h^, rank, in_f, out_f, Float32(1.0),
                z1^, z2^, z3^, z4^,
            ))
    var opt_state = lora_adamw_plain_device_state_init(host_ads, 0, len(host_ads), ctx)
    var loras = List[HiDreamO1BlockLora]()
    for li in range(LAYERS):
        var ads = List[Optional[ZImageLoraAdapterDevice]]()
        for sl in range(7):
            var i = li * 7 + sl
            var dims = _slot_dims(sl)
            var n_a = rank * dims[0]
            var n_b = dims[1] * rank
            var a_off = opt_state.elem_offset(i, False)
            var b_off = opt_state.elem_offset(i, True)
            ads.append(Optional[ZImageLoraAdapterDevice](ZImageLoraAdapterDevice(
                TArc(Tensor(
                    opt_state.dev_p.create_sub_buffer[DType.uint8](a_off * 2, n_a * 2),
                    [rank, dims[0]], STDtype.BF16,
                )),
                TArc(Tensor(
                    opt_state.dev_p.create_sub_buffer[DType.uint8](b_off * 2, n_b * 2),
                    [dims[1], rank], STDtype.BF16,
                )),
                rank, dims[0], dims[1], Float32(1.0),
            )))
        loras.append(HiDreamO1BlockLora(
            ads[0].copy(), ads[1].copy(), ads[2].copy(), ads[3].copy(),
            ads[4].copy(), ads[5].copy(), ads[6].copy(),
        ))
    print("[lora] adapters:", LAYERS * 7, " resident params:", opt_state.total)

    # T1.B EMA (default-off, training/lora_ema.mojo SimpleTuner semantics):
    # F32 shadows over the host_ads mirrors. Config wins when it enables EMA
    # (full schedule: cap/floor/after-step/interval); else the argv ema_decay
    # keeps the pre-config every-step form.
    var ema_cfg_on = has_config and train_cfg.ema_enabled
    var ema_on = ema_cfg_on or ema_decay > Float32(0.0)
    var ema: LoraEmaState
    if ema_cfg_on:
        ema = LoraEmaState(
            train_cfg.ema_decay, train_cfg.ema_min_decay,
            train_cfg.ema_update_after_step,
            train_cfg.ema_update_step_interval,
        )
    else:
        ema = LoraEmaState(ema_decay, Float32(0.0), 0, 1)
    if ema_on:
        _ = lora_ema_track(ema, host_ads, 0, len(host_ads))
        print(
            "[ema] tracking", len(host_ads), "adapters decay=", ema.decay,
            " min_decay=", ema.min_decay,
            " update_after_step=", ema.update_after_step,
            " interval=", ema.update_interval,
        )

    # T1.C levers optimizer state (lazy; allocates nothing on the default
    # fused-AdamW path).
    var lev_opt = LeversOptimizerState()

    # ── BF16-RESIDENT blocks (measured fix, 2026-06-11): await_block re-reads
    # + re-converts the F32 shards on EVERY visit (planned_loader.mojo:110 ->
    # load_block_as_bf16) — 72 visits x ~845 MB = ~60 GB/step = the ~100 s.
    # Converted ONCE, the whole kept set is ~15.2 GB bf16 -> RESIDENT fits
    # 24 GB (the earlier OOM was the F32 30.4 GB resident load).
    # T2.B: with fp8_resident, the bf16 block is quantized IMMEDIATELY after
    # its one load (E4M3 bytes + per-row scales resident, 1 byte/param for the
    # 7 big linears) and the bf16 copy drops with the handle — peak transient
    # = one layer's bf16 alongside its fp8 form.
    if fp8_resident:
        print("[load] converting 36 layers to fp8_e4m3-resident (once)")
    else:
        print("[load] converting 36 layers to bf16-resident (once)")
    var resident_blocks = List[Block]()
    for li in range(LAYERS):
        var h0 = dit.loader.await_block(li, ctx)
        if fp8_resident:
            resident_blocks.append(_quantize_block_fp8(h0.block, li, ctx))
        else:
            resident_blocks.append(h0.block.copy())

    var imgs = ShardedSafeTensors.open(stage_dir + "/images.safetensors")
    var n_samples = 0
    while True:
        try:
            _ = imgs.tensor_view(String("image.") + String(n_samples))
            n_samples += 1
        except:
            break
    if n_samples == 0:
        raise Error("train_hidream_o1: no image.<i> in stage dir")
    print("[data] samples:", n_samples)

    var sw = _sigmas_and_weights()
    var train_start = perf_counter_ns()

    var norm_w = dit.shared[String("model.language_model.norm.weight")].copy()
    var final_w = dit.shared[String("model.final_layer2.linear.weight")].copy()

    var smooth = Float32(0.0)
    var smooth_init = False
    for step in range(1, steps + 1):
        var t0 = perf_counter_ns()
        var idx = (step - 1) % n_samples

        # ── data: clean patches + ids/positions/mask ─────────────────────────
        var img = Tensor.from_view(imgs.tensor_view(String("image.") + String(idx)), ctx)
        var img_bf = cast_tensor(img, STDtype.BF16, ctx)
        var clean = patchify(img_bf, PATCH, ctx)               # [1,IMG_L,3072] bf16
        var clean_h = clean.to_host(ctx)

        var caption = _read_text(stage_dir + "/caption." + String(idx) + ".txt")
        # T1.D caption dropout (default-off; config caption_dropout_prob or
        # the comptime fallback): when the shared levers pick fires, train
        # this step on the EMPTY caption ("" through the same template path
        # = the uncond render).
        if caption_dropout_pick(UInt64(step), SEED, caption_dropout_p):
            caption = String("")
        # Fit the caption into the S_TEXT bucket: halve the string until the
        # templated ids fit (build_t2i_input raises on overflow — fail-loud
        # contract; long giger captions need ~1-2 halvings).
        var samp: T2ISample
        while True:
            try:
                samp = build_t2i_input(tok, cfg, caption, HP, WP, S_TEXT)
                break
            except:
                var keep = caption.byte_length() // 2
                if keep < 8:
                    raise Error("train_hidream_o1: caption cannot fit bucket")
                # byte-truncate via codepoint walk (avoid splitting UTF-8)
                var cut = String("")
                var taken = 0
                for ch in caption.codepoint_slices():
                    var l = ch.byte_length()
                    if taken + l > keep:
                        break
                    cut += String(ch)
                    taken += l
                caption = cut^

        # sigma / noise / noisy / model_t (host F32 math)
        var st = SEED * UInt64(6364136223846793005) + UInt64(step)
        st = st * 6364136223846793005 + 1442695040888963407
        var t_id = Int((st >> 33) % UInt64(1000))
        var sigma = sw.sigmas[t_id]
        var wt = sw.weights[t_id]
        var noise = _gauss(IMG_L * PATCH_VEC, SEED * UInt64(7919) + UInt64(step))
        var noisy_h = List[Float32]()
        var target_h = List[Float32]()
        for i in range(IMG_L * PATCH_VEC):
            var nz = noise[i] * NOISE_SCALE
            noisy_h.append((Float32(1.0) - sigma) * clean_h[i] + sigma * nz)
            target_h.append(nz - clean_h[i])
        var noisy = Tensor.from_host(noisy_h.copy(), [1, IMG_L, PATCH_VEC], STDtype.BF16, ctx)

        # ── embed (frozen): text + t-embed scatter + patch embed + concat ────
        var text_emb = dit._embed(samp.text_ids, ctx)
        var model_t = Float32(1.0) - sigma
        var t_emb = dit._t_embed(model_t, ctx)
        var tms_idx = -1
        for i in range(len(samp.text_ids)):
            if samp.text_ids[i] == cfg.tms_token_id:
                tms_idx = i
        var text_emb_t = _scatter_row(text_emb, t_emb, tms_idx, len(samp.text_ids), D, ctx)
        var patch_emb = dit._patch_embed(noisy, ctx)
        var hidden_t = concat(1, ctx, text_emb_t, patch_emb)

        # mrope tables + mask (per step; host)
        var tables = _build_mrope_tables(
            samp.t_pos, samp.h_pos, samp.w_pos, Dh, cfg.rope_theta,
            cfg.mrope_h, cfg.mrope_w,
        )
        comptime half = Dh // 2
        var cq_sh: List[Int] = [S * H * half]
        var ck_sh: List[Int] = [S * HKV * half]
        var cos_q = Tensor.from_host(_replicate_heads(tables[0], S, half, H), cq_sh.copy(), STDtype.F32, ctx)
        var sin_q = Tensor.from_host(_replicate_heads(tables[1], S, half, H), cq_sh^, STDtype.F32, ctx)
        var cos_k = Tensor.from_host(_replicate_heads(tables[0], S, half, HKV), ck_sh.copy(), STDtype.F32, ctx)
        var sin_k = Tensor.from_host(_replicate_heads(tables[1], S, half, HKV), ck_sh^, STDtype.F32, ctx)

        var mask_h = _build_prefix_causal_mask_padded(S, H, samp.ar_len, samp.key_valid)
        var m4_sh: List[Int] = [1, H, S, S]
        var mask4 = Tensor.from_host(mask_h.copy(), m4_sh^, STDtype.BF16, ctx)
        var mhs_sh: List[Int] = [H * S, S]
        var mask_f32 = Tensor.from_host(mask_h^, mhs_sh^, STDtype.F32, ctx)

        # ── forward stack (recompute-checkpoint: keep block INPUTS only) ─────
        var x = TArc(hidden_t^)
        var block_in = List[TArc]()
        for li in range(LAYERS):
            block_in.append(x.copy())
            # T2.B: flag-off this IS _block_weights (C13); flag-on dequants
            # the fp8-resident linears to bf16 for this block only.
            var bwts = _block_weights_q(resident_blocks[li], li, fp8_resident, ctx)
            var f = hidream_o1_block_lora_forward[S, H, HKV, Dh](
                x, bwts, loras[li], cos_q, sin_q, cos_k, sin_k, mask4,
                D, F, EPS, ctx,
            )
            x = f.out.copy()
            # f.saved drops here — recompute rebuilds it in the bwd loop.

        # final norm + final linear
        var final_in = x.copy()
        var final_normed = rms_norm(final_in[], norm_w[], EPS, ctx)
        var out_full = dit._final_layer(final_normed, ctx)     # [1,S,3072]
        var out_h = out_full.to_host(ctx)

        # ── loss + d_out (host F32; x-prediction -> velocity chain) ──────────
        comptime NOUT = IMG_L * PATCH_VEC
        var base = S_TEXT * PATCH_VEC
        var d_full = List[Float32]()
        for _ in range(base):
            d_full.append(0.0)
        var inv_sigma = Float32(1.0) / sigma
        var loss: Float32
        if levers_loss_active(train_cfg):
            # T1.A loss levers (training/levers.mojo): huber / smooth_l1 /
            # min-SNR-flow over (v_pred, target). The DiffSynth gauss-shift
            # recipe weight wt ALWAYS multiplies loss AND grad on top — it is
            # the MODEL RECIPE, not a lever. The x-prediction chain rule
            # dv_pred/dx_pred = -1/sigma is unchanged from the legacy block.
            var v_pred_l = List[Float32]()
            for i in range(NOUT):
                v_pred_l.append((noisy_h[i] - out_h[base + i]) * inv_sigma)
            var lg = levers_loss_grad(v_pred_l, target_h, sigma, train_cfg)
            loss = lg.loss * wt
            var dch = -wt * inv_sigma
            for i in range(NOUT):
                d_full.append(dch * lg.d_pred[i])
        else:
            # Literal legacy block (default path, byte-identical; the levers
            # default IS this formula but the literal code stays — C13).
            var sse = 0.0
            var dcoef = -wt * Float32(2.0) / Float32(NOUT) * inv_sigma
            for i in range(NOUT):
                var x_pred = out_h[base + i]
                var v_pred = (noisy_h[i] - x_pred) * inv_sigma
                var diff = v_pred - target_h[i]
                sse += Float64(diff) * Float64(diff)
                d_full.append(dcoef * diff)
            loss = Float32(sse / Float64(NOUT)) * wt
        var d_out_t = Tensor.from_host(d_full^, [1, S, PATCH_VEC], STDtype.BF16, ctx)

        # final backward (frozen): linear dx + rms dx
        var d_final_normed = linear_backward_dx(d_out_t, final_w[], S, D, PATCH_VEC, ctx)
        var d_x = TArc(rms_norm_backward_dx(d_final_normed, final_in[], norm_w[], EPS, ctx))

        # ── backward stack: recompute tape per block, P1 backward ────────────
        # Grads collected per adapter index; ONE fused resident AdamW step
        # after the loop (the zimage v2 optimizer — params update in place).
        var g_a = List[List[Float32]]()
        var g_b = List[List[Float32]]()
        for _ in range(LAYERS * 7):
            g_a.append(List[Float32]())
            g_b.append(List[Float32]())
        var bi = LAYERS - 1
        while bi >= 0:
            # T2.B: same dispatch as forward — backward recompute uses the
            # SAME dequantized weights forward used (deterministic decode).
            var bwts_b = _block_weights_q(resident_blocks[bi], bi, fp8_resident, ctx)
            var rf = hidream_o1_block_lora_forward[S, H, HKV, Dh](
                block_in[bi], bwts_b, loras[bi], cos_q, sin_q, cos_k, sin_k,
                mask4, D, F, EPS, ctx,
            )
            var bg = hidream_o1_block_lora_backward[S, H, HKV, Dh](
                d_x[], bwts_b, loras[bi], rf.saved,
                cos_q, sin_q, cos_k, sin_k, mask_f32, D, F, EPS, ctx,
            )
            d_x = bg.d_hidden.copy()
            for sl in range(7):
                if not bg.d_a[sl]:
                    raise Error("train_hidream_o1: missing adapter grad")
                var k = bi * 7 + sl
                g_a[k] = bg.d_a[sl].value()[].to_host(ctx)
                g_b[k] = bg.d_b[sl].value()[].to_host(ctx)
            bi -= 1
        # T2.B gate instrumentation (argv[8], default-off): dump the step-1
        # adapter grads BEFORE the optimizer touches anything, for the
        # fp8-vs-bf16 grad-cosine comparison.
        if step == 1 and grad_dump != String(""):
            var gnames = List[String]()
            var gtensors = List[TArc]()
            for k in range(LAYERS * 7):
                gnames.append(String("g_a.") + String(k))
                gtensors.append(TArc(Tensor.from_host(
                    g_a[k].copy(), [len(g_a[k])], STDtype.F32, ctx)))
                gnames.append(String("g_b.") + String(k))
                gtensors.append(TArc(Tensor.from_host(
                    g_b[k].copy(), [len(g_b[k])], STDtype.F32, ctx)))
            save_safetensors(gnames, gtensors, grad_dump, ctx)
            print("[grad-dump] ", grad_dump)
        if levers_optimizer_active(train_cfg):
            # T1.C optimizer lever (default-off): host adafactor /
            # schedule-free step on the host_ads mirrors + resident dev_p
            # sync so the device LoRA views (sub-buffers of opt_state.dev_p)
            # see the new weights next step (levers.mojo T1.C header). No LR
            # scheduler here: step_lr == the constant resolved lr.
            levers_optimizer_step(
                train_cfg, host_ads, g_a, g_b, step, lr,
                lev_opt, opt_state, ctx,
            )
        else:
            # Default fused resident AdamW. cfg hypers == the old literals
            # 0.9/0.999/1e-8/0.01 when no config (TrainConfig defaults), so
            # the no-config path is numerically unchanged; a config's
            # optimizer.{beta1,beta2,eps,weight_decay} overrides.
            fused_lora_adamw_plain_step_resident(
                opt_state, host_ads, g_a, g_b, step, lr,
                train_cfg.beta1, train_cfg.beta2, train_cfg.eps,
                train_cfg.weight_decay, ctx,
            )
        # T1.B EMA: host_ads mirrors are FRESH here — both optimizer paths
        # write updated P back into them (resident: lora_adamw_plain_fused
        # .mojo:483-502 readback; levers: host step IS the mirror).
        if ema_on:
            ema_update(ema, host_ads, step)

        var t1 = perf_counter_ns()
        if smooth_init:
            smooth = smooth * Float32(0.99) + loss * Float32(0.01)
        else:
            smooth = loss
            smooth_init = True
        var b_absum = Float32(0.0)
        for k in range(LAYERS * 7):
            for i in range(len(host_ads[k].b)):
                var av = Float32(host_ads[k].b[i])
                b_absum += av if av >= 0.0 else -av
        print(
            "[HiDreamO1-lora] step ", step, "/", steps,
            " | sigma ", sigma, " | loss ", loss, " | smooth ", smooth,
            " | B|.|1 ", b_absum,
            " | ", Float32(Float64(t1 - t0) / 1.0e9), "s/step",
        )
        # Shared UI progress line (the serenity-trainer TrainerRuntimeBridge
        # progress parser shape, same as the config-driven runners). Purely
        # additive stdout — the detail line above and all loss anchors are
        # untouched. grad_norm: this trainer does not compute a global norm.
        print_trainer_progress(
            String("HiDreamO1"), step, steps, n_samples, loss,
            0.0, Float64(t1 - t0) / 1.0e9, 0.0,
            Float64(t1 - train_start) / 1.0e9,
        )

    # ── save (DiffSynth-loadable key shape) ──────────────────────────────────
    # T1.C schedule-free save bracket (levers.mojo SAVE CONTRACT): every
    # weight save sits between eval_for_save / train_after_save. No-op for
    # ADAMW / ADAFACTOR.
    levers_optimizer_eval_for_save(train_cfg, lev_opt)
    var names = List[String]()
    var tensors = List[TArc]()
    for li in range(LAYERS):
        for sl in range(7):
            var k = li * 7 + sl
            var dims = _slot_dims(sl)
            var prefix = String("diffusion_model.model.language_model.layers.")
                + String(li) + "." + _slot_name(sl)
            names.append(prefix + ".lora_A.weight")
            tensors.append(TArc(Tensor.from_host_bf16(
                host_ads[k].a.copy(), [rank, dims[0]], ctx)))
            names.append(prefix + ".lora_B.weight")
            tensors.append(TArc(Tensor.from_host_bf16(
                host_ads[k].b.copy(), [dims[1], rank], ctx)))
    var out_path = out_dir + "/hidream_o1_lora_last.safetensors"
    save_safetensors(names, tensors, out_path, ctx)
    print("[save] ", out_path)
    if ema_on:
        # T1.B: EMA sibling — same DiffSynth key shape over the bf16-rounded
        # shadows (lora_ema.mojo copy_to cast, SimpleTuner ema.py:454).
        var ema_tensors = List[TArc]()
        for li in range(LAYERS):
            for sl in range(7):
                var k = li * 7 + sl
                var dims = _slot_dims(sl)
                ema_tensors.append(TArc(Tensor.from_host_bf16(
                    ema_shadow_a_bf16(ema, k), [rank, dims[0]], ctx)))
                ema_tensors.append(TArc(Tensor.from_host_bf16(
                    ema_shadow_b_bf16(ema, k), [dims[1], rank], ctx)))
        var ema_out = ema_path_for_lora(out_path)
        save_safetensors(names, ema_tensors, ema_out, ctx)
        print("[save] ", ema_out)
    levers_optimizer_train_after_save(train_cfg, lev_opt)
    print("DONE: hidream-o1 reached step ", steps)
