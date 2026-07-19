# Replay the real Serenity Chroma train-step dump through the Mojo forward path.
#
# Mirrors smoke/klein_train_ref_forward_replay.mojo, but for Chroma1-HD. It
# consumes parity/chroma_train_ref_step000.safetensors (from
# scripts/chroma_dump_train_ref.py) and compares the Mojo B=0 LoRA forward
# against Serenity's dumped trace.packed_predicted_flow (per batch sample).
#
# Chroma deltas vs Klein (see model/chroma/chroma_stack_lora.mojo header):
#   - Modulation comes from the FROZEN distilled_guidance_layer APPROXIMATOR
#     (serenitymojo.models.dit.chroma_dit.ChromaDitCache.approximator_forward),
#     producing a per-step pooled_temb table [mod_index=344, D=3072]. Driven by
#     trace.transformer_timestep (one value per batch sample).
#   - Block weights are streamed (block-swap offload) from the real Chroma1-HD
#     checkpoint via TurboPlannedLoader; the loader does the separate->fused
#     row-stack so the per-block compute is the proven Flux block.
#   - The dump is B=2 (two samples w/ distinct timesteps 0.907 / 0.709); the
#     stack forward is single-sample, so we run it once per sample and compare
#     each [N_IMG, OUT_CH] velocity against trace.packed_predicted_flow[b].
#
# This is NOT the full train parity gate: loss/backward/AdamW adapter deltas are
# checked by chroma_train_ref_grad_update_replay.mojo. The main loop owns the
# final parity bar; this smoke prints cos / max_abs for re-verification.

from std.collections import List, Optional
from std.gpu.host import DeviceContext
from std.math import sqrt
from std.time import perf_counter_ns

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.tensor import Tensor

from serenitymojo.models.dit.flux1_dit import build_flux1_rope_tables
from serenitymojo.models.dit.chroma_dit import ChromaDitCache
from serenitymojo.offload.plan import build_chroma1_hd_block_plan, OffloadConfig
from serenitymojo.offload.turbo_planned_loader import TurboPlannedLoader
from serenitymojo.models.flux.flux_stack_lora import (
    FluxLoraSet, build_flux_lora_set, total_adapters,
)
from serenitymojo.models.flux.lora_block import DBL_STREAM_SLOTS, SGL_SLOTS

from serenity_trainer.model.chroma.weights import load_chroma_stack_base
from serenity_trainer.model.chroma.chroma_stack_lora import (
    chroma_stack_lora_forward_offload,
)


# ── arch (chroma1-hd; verified vs the checkpoint) ────────────────────────────
comptime H = 24
comptime Dh = 128
comptime D = H * Dh            # 3072
comptime FMLP = 12288
comptime IN_CH = 64
comptime TXT_CH = 4096
comptime OUT_CH = 64
comptime NUM_DOUBLE = 19
comptime NUM_SINGLE = 38
comptime MOD_INDEX = 3 * NUM_SINGLE + 2 * 6 * NUM_DOUBLE + 2   # 344
comptime EPS = Float32(1e-06)

# ── dump shape: N_IMG=1024 (32x32 packed), N_TXT=224, B=2 ────────────────────
comptime HT = 32
comptime WT = 32
comptime N_IMG = HT * WT       # 1024
comptime N_TXT = 224           # text tokens in this dump
comptime S = N_TXT + N_IMG     # 1248 (matches trace.attention_mask [2,1248])
comptime BATCH = 2

# ── recipe (LoRA carrier; B=0 init so forward is alpha-independent) ──────────
comptime RANK = 16
comptime ALPHA = Float32(1.0)  # dump uses lora_alpha=1.0 (forward unaffected at B=0)

comptime PARITY = "trainer/parity/chroma_train_ref_step000.safetensors"
comptime CKPT = "/home/alex/.serenity/models/checkpoints/chroma1_hd_bf16.safetensors"

comptime MIN_FORWARD_COS = Float64(0.999)


def _sec(ns0: UInt, ns1: UInt) -> Float64:
    return Float64(ns1 - ns0) / Float64(1000000000.0)


def _dump_f32(st: ShardedSafeTensors, key: String, ctx: DeviceContext) raises -> List[Float32]:
    """Load a dump tensor as a flat host F32 list, staged through stored dtype."""
    var t = Tensor.from_view(st.tensor_view(key), ctx)
    if t.dtype() == STDtype.BF16:
        var bf = t.to_host_bf16(ctx)
        var out = List[Float32]()
        for i in range(len(bf)):
            out.append(bf[i].cast[DType.float32]())
        return out^
    return t.to_host(ctx)


def _slice_sample(flat: List[Float32], b: Int, per: Int) -> List[Float32]:
    var out = List[Float32]()
    var base = b * per
    for i in range(per):
        out.append(flat[base + i])
    return out^


def _compare(
    label: String, got: List[Float32], expected: List[Float32], min_cos: Float64,
) raises:
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
    if cos < min_cos:
        raise Error(label + String(": cosine below gate"))


# Build the frozen per-step modulation table [1, MOD_INDEX, D] as a host F32 list.
def _pooled_modulation(approx: ChromaDitCache, t_model: Float32, ctx: DeviceContext) raises -> List[Float32]:
    var approx_in = approx._approximator_input(t_model, ctx)
    var pooled_t = approx.approximator_forward(approx_in, ctx)   # [1, MOD_INDEX, D] BF16
    var bf = pooled_t.to_host_bf16(ctx)
    var out = List[Float32]()
    for i in range(len(bf)):
        out.append(bf[i].cast[DType.float32]())
    return out^


def main() raises:
    var ctx = DeviceContext()
    var all0 = perf_counter_ns()

    print("=== Chroma train-ref forward replay ===")
    print("[parity]", PARITY)
    print("[ckpt]  ", CKPT)
    print("[shape] N_IMG =", N_IMG, "N_TXT =", N_TXT, "S =", S, "BATCH =", BATCH)
    print("[arch]  D =", D, "H =", H, "Dh =", Dh, "Fmlp =", FMLP, "mod_index =", MOD_INDEX)

    var st = ShardedSafeTensors.open(String(PARITY))
    var img_all = _dump_f32(st, String("trace.packed_latent_input"), ctx)     # [B,1024,64]
    var txt_all = _dump_f32(st, String("trace.encoder_hidden_states"), ctx)   # [B,224,4096]
    var ref_all = _dump_f32(st, String("trace.packed_predicted_flow"), ctx)   # [B,1024,64]
    var ts_all = _dump_f32(st, String("trace.transformer_timestep"), ctx)     # [B]
    print("[dump] img n =", len(img_all), " txt n =", len(txt_all),
          " ref n =", len(ref_all), " timesteps =", ts_all[0], ts_all[1])

    var load0 = perf_counter_ns()
    # ── stack-level base (frozen; x_embedder/context_embedder/proj_out) ──
    var base_st = SafeTensors.open(String(CKPT))
    var base = load_chroma_stack_base(base_st, NUM_DOUBLE, NUM_SINGLE, ctx)
    # ── frozen approximator (distilled_guidance_layer) ──
    var approx = ChromaDitCache.load(String(CKPT), ctx)
    # ── block-swap offload loader ──
    var plan = build_chroma1_hd_block_plan()
    var cfg = OffloadConfig.synchronous_single()
    var loader = TurboPlannedLoader.open(String(CKPT), plan^, cfg, ctx)
    var load1 = perf_counter_ns()
    print("[load] base + approximator + offload loader (", loader.block_count(), "blocks)")

    # ── 3-axis RoPE tables (txt at pos 0, img 32x32 grid; built once, BF16) ──
    var rope = build_flux1_rope_tables[N_IMG, N_TXT, H, Dh](HT, WT, ctx, STDtype.BF16)
    var cos = rope[0].to_host(ctx)
    var sin = rope[1].to_host(ctx)
    print("[load] chroma 3-axis rope tables built")

    # ── LoRA carrier (B=0 init -> identity at step 0; forward alpha-independent) ──
    var lora = build_flux_lora_set(NUM_DOUBLE, NUM_SINGLE, D, FMLP, RANK, ALPHA)
    print("[lora] adapters:", total_adapters(lora))

    var img_per = N_IMG * IN_CH      # 1024*64
    var txt_per = N_TXT * TXT_CH     # 224*4096
    var ref_per = N_IMG * OUT_CH

    var fwd_total = Float64(0.0)
    for b in range(BATCH):
        var img_tokens = _slice_sample(img_all, b, img_per)
        var txt_tokens = _slice_sample(txt_all, b, txt_per)
        var ref_b = _slice_sample(ref_all, b, ref_per)
        var t_model = ts_all[b]
        var pooled = _pooled_modulation(approx, t_model, ctx)
        print("[sample", b, "] t_model =", t_model, " pooled n =", len(pooled))

        var f0 = perf_counter_ns()
        var fwd = chroma_stack_lora_forward_offload[H, Dh, N_IMG, N_TXT, S](
            img_tokens^, txt_tokens^, pooled^, MOD_INDEX,
            base, loader, lora, cos.copy(), sin.copy(),
            D, FMLP, IN_CH, TXT_CH, OUT_CH, EPS, ctx,
        )
        var f1 = perf_counter_ns()
        fwd_total += _sec(f0, f1)
        _compare(String("packed_flow[") + String(b) + String("]"),
                 fwd.out.copy(), ref_b, MIN_FORWARD_COS)

    var all1 = perf_counter_ns()
    print("time_s: load =", _sec(load0, load1), " forward =", fwd_total,
          " total =", _sec(all0, all1))
    print("CHROMA TRAIN REF FORWARD REPLAY PASS")
