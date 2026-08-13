# training/train_wan22_real.mojo — Wan2.2-T2V 14B LoRA training loop (block-swap offload).
#
# MIRRORS train_chroma_real.mojo structure (timing, flow-match recipe, progress
# display, smoke gate). Ports the wan22.rs EDv2 training recipe.
#
# ARCHITECTURE (14B; wan2.2_t2v_low_noise_14b_fp16.safetensors, confirmed):
#   dim=5120, H=40, Dh=128, ffn=13824, num_blocks=40
#   in_ch=64 (patch latent: 16ch * 2*2 patchify -> 64 elem per patch)
#   out_ch=64 (head output per patch token, same as in_ch)
#   text_dim=4096 (T5-XXL hidden dim), freq_dim=256 (sinusoidal embed)
#   S=N_IMG  (image tokens; for 512px T2V: grid depends on resolution)
#
# NOTE: This trainer targets the T2V inference checkpoint reused for image
# LoRA training (single-frame; the VAE+patchify pipeline collapses the
# temporal dimension to F=1 so S = H_patch * W_patch).
#
# FLOW-MATCH RECIPE (wan22.rs + EDv2 training config):
#   timestep_shift = 1.0 (no logit-normal shift bias; plain uniform)
#   sigma_idx = floor(uniform_sigma * 1000) clamped to [0, 999]
#   sig = (sigma_idx + 1) / 1000; t_model = sigma_idx / 1000
#   noisy = noise * sig + latent * (1 - sig)
#   target = noise - latent                 (velocity target)
#   loss = MSE(pred, target)
#
# LORA TARGETS: 8 per block (sa_{q,k,v,o} + ca_{q,k,v,o}), rank=32, alpha=32.
#   320 adapters total. All in=out=dim=5120.
#
# MEMORY SAFETY (enforced):
#   - DO NOT LOAD the full 14B checkpoint into VRAM.
#   - Block weights are streamed ONE AT A TIME by TurboPlannedLoader.
#   - Direct DoRA/OFT gates sample device memory and fail above 24 GiB.
#   - Full-depth 40-block runtime remains a bounded smoke until step time, peak
#     VRAM, trainable movement, and save artifacts are measured end to end.
#
# Build (link with the repo cshim/rpath recipe used by other trainers):
#   cd <repository-root> && \
#     pixi run mojo build -I . -Xlinker -lm -Xlinker -lcuda \
#       trainer/src/serenity_trainer/trainer/train_wan22_real.mojo \
#       -o output/verification/bin/train_wan22_real

from std.sys import argv
from std.collections import List, Optional
from max.gpu.host import DeviceContext
from std.math import sqrt, log as flog, cos as fcos, sin as fsin
from std.time import perf_counter_ns
from std.os import listdir

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.ops.cast import cast_tensor

from serenitymojo.models.wan22.weights import load_wan22_stack_base
from serenitymojo.models.wan22.wan22_stack_lora import (
    Wan22LoraSet, Wan22LoraGradSet, Wan22StackForward,
    build_wan22_lora_set, wan22_lora_adamw_step, save_wan22_lora,
    wan22_lora_adamw_state_init, wan22_lora_adamw_step_resident,
    save_wan22_lora_state, load_wan22_lora_state, load_wan22_lora_resume,
    wan22_total_adapters, WAN_SLOTS,
    wan22_stack_lora_forward_offload, wan22_stack_lora_backward_offload,
    build_wan22_direct_dora_set_from_offload,
    wan22_stack_direct_dora_forward_offload,
    wan22_stack_direct_dora_backward_offload,
    wan22_stack_direct_oft_forward_offload,
    wan22_stack_direct_oft_backward_offload,
)
from serenitymojo.offload.plan import OffloadConfig
from serenitymojo.offload.wan22_plan import build_wan22_14b_block_plan
from serenitymojo.offload.turbo_planned_loader import TurboPlannedLoader
from serenitymojo.training.schedule import sample_timestep_logit_normal
from serenitymojo.training.progress_display import print_trainer_progress
from serenitymojo.training.lora_adamw_plain_fused import (
    LoraAdamWPlainDeviceState,
    lora_adamw_plain_device_state_sync_params,
    lora_adamw_plain_device_state_sync_moments,
)
from serenitymojo.training.lora_save import lora_train_state_has_moments
from serenitymojo.training.trainer_core import (
    trainer_resolve_resume_path, trainer_warn_warm_resume,
)
from serenitymojo.io.train_config_reader import read_model_config
from serenitymojo.training.train_config import (
    TrainConfig,
    TRAIN_ADAPTER_ALGO_LORA,
    TRAIN_ADAPTER_ALGO_FULL,
    TRAIN_ADAPTER_ALGO_LOHA,
    TRAIN_ADAPTER_ALGO_LOKR,
    TRAIN_ADAPTER_ALGO_DORA,
    TRAIN_ADAPTER_ALGO_OFT,
    TRAIN_ADAPTER_ALGO_BOFT,
    TRAIN_ADAPTER_ALGO_LOCON,
)
from serenitymojo.training.adapter_algo_policy import adapter_algo_name
from serenitymojo.training.lokr_stack import LOKR_CARRIER_MAX_DEVICE_BYTES
from serenitymojo.training.flat_lycoris_stack import (
    FlatLoKrSet, empty_flat_lokr_set, build_flat_lokr_set,
    flat_lokr_carrier_list, flat_lokr_carrier_total_bytes,
    flat_lokr_chain_all, flat_lokr_grad_norm, flat_lokr_clip_grads,
    flat_lokr_adamw_step, flat_lokr_zero_leg_l1, save_flat_lokr,
    FlatLoHaSet, empty_flat_loha_set, build_flat_loha_set,
    flat_loha_carrier_list, flat_loha_carrier_total_bytes,
    flat_loha_chain_all, flat_loha_grad_norm, flat_loha_clip_grads,
    flat_loha_adamw_step, flat_loha_zero_leg_l1, save_flat_loha,
)
from serenitymojo.models.wan22.wan22_direct_lycoris_stack import (
    empty_wan22_direct_dora_set,
    empty_wan22_direct_oft_set,
    build_wan22_direct_oft_set,
    wan22_direct_dense_carrier_bytes,
    wan22_direct_dora_preflight,
    wan22_direct_oft_preflight,
    wan22_direct_dora_grad_norm,
    wan22_direct_dora_clip_grads,
    wan22_direct_dora_adamw_step,
    wan22_direct_dora_zero_leg_l1,
    wan22_direct_dora_trainable_bytes,
    wan22_direct_oft_grad_norm,
    wan22_direct_oft_clip_grads,
    wan22_direct_oft_adamw_step,
    wan22_direct_oft_vec_l1,
    wan22_direct_oft_trainable_bytes,
    save_wan22_direct_dora,
    save_wan22_direct_oft,
)


# ── arch (wan2.2_t2v_low_noise_14b; dims confirmed from safetensors header) ───
comptime H = 40
comptime Dh = 128
comptime DIM = H * Dh          # 5120
comptime FFN = 13824
comptime NUM_BLOCKS = 40
comptime FREQ_DIM = 256        # sinusoidal time embedding
comptime TEXT_DIM = 4096       # T5-XXL context channels
comptime OUT_CH = 64           # head output per patch token

# ── single-frame 512px patchify (T2V, F=1, 2x2 spatial patch, 16 latent ch) ──
# Patch embed: [16, 1, H_lat, W_lat] -> [S, in_ch=64] where S=H_patch*W_patch.
# 512px -> latent 64x64 -> 2x2 patchify -> 32x32 = 1024 tokens.
comptime LAT_C = 16
comptime LAT_H = 64
comptime LAT_W = 64
comptime PH = 2
comptime PW = 2
comptime N_IMG = (LAT_H // PH) * (LAT_W // PW)   # 1024
comptime IN_CH = LAT_C * PH * PW                  # 64 (patch_dim)
comptime S = N_IMG
comptime TXT = 512             # padded T5 sequence length

# ── recipe ────────────────────────────────────────────────────────────────────
comptime RANK = 32
comptime ALPHA = Float32(32.0)
comptime LR = Float32(1.0e-4)
comptime WAN22_DIRECT_OFT_BLOCK_SIZE = 4
comptime WAN22_DIRECT_VRAM_BUDGET_BYTES = 24 * 1024 * 1024 * 1024
# NOTE: wan22.rs uses a plain uniform timestep (shift=1.0 -> logit-normal
# collapses to near-uniform). The low-noise checkpoint is already fine-tuned
# for a shifted noise schedule; we train with uniform sampling to match that.
comptime TIMESTEP_SHIFT = Float32(1.0)
# VAE shift/scale: Wan2.2 uses a latent normalization. Actual values depend on
# the dataset preprocessing pipeline (the inference code normalises to ~ N(0,1)
# with mean≈0 and std≈1). Conservative defaults (identity): shift=0, scale=1.
# TODO: verify exact Wan2.2 VAE normalisation from wan/vae/config.json or
#       wan22.rs vae_config once the cache is built with the correct values.
comptime VAE_SHIFT = Float32(0.0)
comptime VAE_SCALE = Float32(1.0)
comptime NUM_TRAIN_TIMESTEPS = 1000
comptime CLIP_GRAD_NORM = Float32(1.0)
comptime EPS = Float32(1.0e-6)
comptime SEED_BASE = UInt64(42)

comptime FIXED_SIGMA_SMOKE = True
comptime FIXED_SIGMA_IDX = 500

comptime CKPT = "/home/alex/.serenity/models/checkpoints/wan2.2_t2v_low_noise_14b_fp16.safetensors"
# TODO: set to a real Wan2.2 prepared cache directory when available.
comptime CACHE_DIR = "/home/alex/datasets/wan22_cache"
comptime LORA_DIR = "/home/alex/mojodiffusion/output/wan22_lora"


# ── deterministic host gaussian noise (Box-Muller PCG; matches chroma trainer) ─
def _host_noise(n: Int, seed: UInt64) -> List[Float32]:
    var out = List[Float32]()
    var state = seed
    var i = 0
    while i < n:
        state = state * 6364136223846793005 + 1442695040888963407
        var u1f = Float64(Int((state >> 12) & 0xFFFFFFFFFFFFF)) * (1.0 / 4503599627370496.0)
        state = state * 6364136223846793005 + 1442695040888963407
        var u2f = Float64(Int((state >> 12) & 0xFFFFFFFFFFFFF)) * (1.0 / 4503599627370496.0)
        if u1f < 1.0e-12:
            u1f = 1.0e-12
        var r = sqrt(-2.0 * flog(Float64(u1f)))
        var theta = 6.283185307179586 * u2f
        out.append(Float32(r * fcos(Float64(theta))))
        if i + 1 < n:
            out.append(Float32(r * fsin(Float64(theta))))
        i += 2
    return out^


def _absum(v: List[Float32]) -> Float32:
    var s = Float32(0.0)
    for i in range(len(v)):
        var x = v[i]
        s += x if x >= 0.0 else -x
    return s


def _absum(v: List[BFloat16]) -> Float32:
    var s = Float32(0.0)
    for i in range(len(v)):
        var x = v[i].cast[DType.float32]()
        s += x if x >= 0.0 else -x
    return s


def _record_vram_sample(
    tag: String, ctx: DeviceContext, peak_used: Int, budget_bytes: Int,
) raises -> Int:
    ctx.synchronize()
    var mi = ctx.get_memory_info()
    var used = Int(mi[1] - mi[0])
    print("[Wan22-vram]", tag, " used_bytes:", used, " free_bytes:", mi[0], " total_bytes:", mi[1])
    if used > budget_bytes:
        raise Error(
            String("Wan22 sampled VRAM exceeded 24 GiB budget at ")
            + tag
            + String(": used ")
            + String(used)
            + String(" bytes > budget ")
            + String(budget_bytes)
        )
    return used if used > peak_used else peak_used


def _global_norm(grads: Wan22LoraGradSet) -> Float64:
    var ss = 0.0
    for i in range(len(grads.d_a)):
        for j in range(len(grads.d_a[i])):
            ss += Float64(grads.d_a[i][j]) * Float64(grads.d_a[i][j])
        for j in range(len(grads.d_b[i])):
            ss += Float64(grads.d_b[i][j]) * Float64(grads.d_b[i][j])
    return sqrt(ss)


def _clip(mut grads: Wan22LoraGradSet, max_norm: Float32) -> Float64:
    var gn = _global_norm(grads)
    if gn <= Float64(max_norm) or gn == 0.0:
        return gn
    var sc = Float32(Float64(max_norm) / gn)
    for i in range(len(grads.d_a)):
        for j in range(len(grads.d_a[i])):
            grads.d_a[i][j] = grads.d_a[i][j] * sc
        for j in range(len(grads.d_b[i])):
            grads.d_b[i][j] = grads.d_b[i][j] * sc
    return gn


def _list_cache(dir: String) raises -> List[String]:
    var raw = listdir(dir)
    var fs = List[String]()
    for i in range(len(raw)):
        if raw[i].endswith(".safetensors"):
            fs.append(dir + String("/") + raw[i])
    if len(fs) == 0:
        raise Error(String("wan22 cache: no .safetensors in ") + dir)
    for i in range(1, len(fs)):
        var j = i
        while j > 0 and fs[j - 1] > fs[j]:
            var tmp = fs[j - 1]
            fs[j - 1] = fs[j]
            fs[j] = tmp
            j -= 1
    return fs^


def _load_host(st: SafeTensors, name: String, ctx: DeviceContext) raises -> List[Float32]:
    var info = st.tensor_info(name)
    var bytes = st.tensor_bytes(name)
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    var t = Tensor.from_view(tv, ctx)
    return cast_tensor(t, STDtype.F32, ctx).to_host(ctx)


def validate_wan22_adapter_config(cfg: TrainConfig) raises:
    if cfg.adapter_algo == TRAIN_ADAPTER_ALGO_LOCON:
        print("[Wan22-locon] network_algorithm=locon: using the linear LoRA-compatible down/up path")
    elif cfg.adapter_algo == TRAIN_ADAPTER_ALGO_LOKR:
        print("[Wan22-lokr] network_algorithm=lokr: using flat attention carrier dispatch through the LoRA stack")
    elif cfg.adapter_algo == TRAIN_ADAPTER_ALGO_LOHA:
        print("[Wan22-loha] network_algorithm=loha: using flat attention carrier dispatch through the LoRA stack")
    elif cfg.adapter_algo == TRAIN_ADAPTER_ALGO_DORA:
        var dense = wan22_direct_dense_carrier_bytes(NUM_BLOCKS, DIM)
        var direct = wan22_direct_dora_preflight(
            NUM_BLOCKS, DIM, cfg.lora_rank, WAN22_DIRECT_VRAM_BUDGET_BYTES, False,
        )
        print(
            "[Wan22-dora] direct trainable bytes:", direct,
            " dense_full_delta_bytes:", dense,
            " budget:", WAN22_DIRECT_VRAM_BUDGET_BYTES,
        )
    elif cfg.adapter_algo == TRAIN_ADAPTER_ALGO_OFT:
        var dense = wan22_direct_dense_carrier_bytes(NUM_BLOCKS, DIM)
        var direct = wan22_direct_oft_preflight(
            NUM_BLOCKS, DIM, WAN22_DIRECT_OFT_BLOCK_SIZE, WAN22_DIRECT_VRAM_BUDGET_BYTES,
        )
        print(
            "[Wan22-oft] direct trainable bytes:", direct,
            " dense_full_delta_bytes:", dense,
            " budget:", WAN22_DIRECT_VRAM_BUDGET_BYTES,
        )
    elif cfg.adapter_algo == TRAIN_ADAPTER_ALGO_BOFT:
        raise Error("Wan22 trainer: BOFT is intentionally excluded; use lora, locon, loha, lokr, dora, or oft where wired")
    elif cfg.adapter_algo == TRAIN_ADAPTER_ALGO_FULL:
        raise Error("Wan22 trainer: full finetune is not wired; supported here: lora, locon, loha, lokr, dora, oft")
    elif cfg.adapter_algo != TRAIN_ADAPTER_ALGO_LORA:
        raise Error(
            String("Wan22 trainer: adapter algorithm ")
            + adapter_algo_name(cfg.adapter_algo)
            + String(" is not wired; supported here: lora, locon, loha, lokr, dora, oft")
        )


def _wan22_flat_dims(num_blocks: Int, dim: Int) -> List[Int]:
    var out = List[Int]()
    for _bi in range(num_blocks):
        for _slot in range(WAN_SLOTS):
            out.append(dim)
    return out^


def _wan22_flat_prefixes(num_blocks: Int) -> List[String]:
    var out = List[String]()
    for bi in range(num_blocks):
        var b = String("blocks.") + String(bi) + "."
        out.append(b + String("self_attn.q"))
        out.append(b + String("self_attn.k"))
        out.append(b + String("self_attn.v"))
        out.append(b + String("self_attn.o"))
        out.append(b + String("cross_attn.q"))
        out.append(b + String("cross_attn.k"))
        out.append(b + String("cross_attn.v"))
        out.append(b + String("cross_attn.o"))
    return out^


def _build_wan22_lokr(cfg: TrainConfig) raises -> FlatLoKrSet:
    var dims = _wan22_flat_dims(NUM_BLOCKS, DIM)
    var names = _wan22_flat_prefixes(NUM_BLOCKS)
    return build_flat_lokr_set(
        dims, dims, names, cfg.lora_rank, cfg.lora_alpha, cfg.lokr_factor,
        cfg.lokr_decompose_both, cfg.lokr_full_matrix,
        cfg.seed * UInt64(61) + UInt64(2201),
    )


def _build_wan22_loha(cfg: TrainConfig) raises -> FlatLoHaSet:
    var dims = _wan22_flat_dims(NUM_BLOCKS, DIM)
    var names = _wan22_flat_prefixes(NUM_BLOCKS)
    return build_flat_loha_set(
        dims, dims, names, cfg.lora_rank, cfg.lora_alpha,
        cfg.seed * UInt64(61) + UInt64(2201),
    )


# Patchify a [16, LAT_H, LAT_W] latent into [N_IMG, IN_CH] tokens.
# Channel-major 2x2 spatial patch (mirrors chroma's _pack_latents structure).
def _pack_latents(lat: List[Float32]) -> List[Float32]:
    var out = List[Float32]()
    for ih in range(LAT_H // PH):
        for iw in range(LAT_W // PW):
            for c in range(LAT_C):
                for ph in range(PH):
                    for pw in range(PW):
                        var hh = ih * PH + ph
                        var ww = iw * PW + pw
                        var idx = c * LAT_H * LAT_W + hh * LAT_W + ww
                        out.append(lat[idx])
    return out^


# Build a trivial [S, Dh/2] RoPE cosine/sine table filled with ones/zeros
# (placeholder for the real 3-axis interleaved RoPE from wan22_dit.mojo).
# TODO: replace with wan22_build_rope once the RoPE builder is callable from
# the training path without a full DiT instance.
def _rope_placeholder(S: Int, half: Int) -> List[Float32]:
    var out = List[Float32]()
    for _ in range(S * half):
        out.append(Float32(1.0))
    return out^


# ── FULL-state resume resolution + loud warm-resume guard (MJ-1088 / MJ-1077) ──
# Probe the `<ckpt>.state` sidecar so a user who passes the PEFT weights still gets
# a FULL (moment-preserving) resume; only warm-restart (loud warning) when there is
# genuinely no moment state. The probe prefix matches _wan22_lora_prefixes[0].
def _wan22_resolve_resume_path(path: String) raises -> String:
    # Thin wrapper → shared trainer_core (binds wan22's first-adapter probe prefix;
    # keeps krea2's default `.state` sidecar naming). Probe order + return logic are
    # the trainer_core skeleton, byte-identical to the pre-extraction path.
    var probe = String("blocks.0.self_attn.q")
    return trainer_resolve_resume_path(path, probe)


def _wan22_warn_warm_resume(path: String):
    # Thin wrapper → shared trainer_core (binds the wan22 log tag; keeps krea2's
    # default `.safetensors.state` FULL-resume sidecar hint). Every banner line is
    # byte-identical to the pre-extraction wan22 output.
    trainer_warn_warm_resume(String("wan22"), path)


def main() raises:
    var ctx = DeviceContext()
    var sampled_peak_vram = _record_vram_sample(
        String("start"), ctx, 0, WAN22_DIRECT_VRAM_BUDGET_BYTES,
    )
    var a = argv()
    var train_cfg = TrainConfig.default()
    train_cfg.lora_rank = RANK
    train_cfg.lora_alpha = ALPHA
    train_cfg.lr = LR
    train_cfg.max_grad_norm = CLIP_GRAD_NORM
    train_cfg.eps = Float32(1.0e-8)
    var has_config = False
    var arg_base = 1
    if len(a) >= 2 and String(a[1]).endswith(String(".json")):
        train_cfg = read_model_config(String(a[1]))
        validate_wan22_adapter_config(train_cfg)
        has_config = True
        arg_base = 2
    var run_steps = 5
    if has_config and train_cfg.max_steps > 0:
        run_steps = train_cfg.max_steps
    if len(a) > arg_base:
        var v = 0
        var bs = String(a[arg_base]).as_bytes()
        for i in range(String(a[arg_base]).byte_length()):
            v = v * 10 + Int(bs[i] - 0x30)
        run_steps = v
    # Optional argv[arg_base+1]: resume checkpoint (plain-LoRA arm). Pass either the
    # `<ckpt>.safetensors.state` sidecar (FULL moment resume) or the plain PEFT
    # `.safetensors` (WARM start — the .state sibling is auto-probed first).
    var resume_path = String("")
    if len(a) > arg_base + 1:
        resume_path = String(a[arg_base + 1])
    var ckpt_path = String(CKPT)
    var cache_dir = String(CACHE_DIR)
    if has_config:
        if train_cfg.checkpoint != String(""):
            ckpt_path = train_cfg.checkpoint.copy()
        if train_cfg.dataset_cache_dir != String(""):
            cache_dir = train_cfg.dataset_cache_dir.copy()
        elif train_cfg.cache_dir != String(""):
            cache_dir = train_cfg.cache_dir.copy()
    var lokr_active = train_cfg.adapter_algo == TRAIN_ADAPTER_ALGO_LOKR
    var loha_active = train_cfg.adapter_algo == TRAIN_ADAPTER_ALGO_LOHA
    var dora_active = train_cfg.adapter_algo == TRAIN_ADAPTER_ALGO_DORA
    var oft_active = train_cfg.adapter_algo == TRAIN_ADAPTER_ALGO_OFT
    var carrier_active = lokr_active or loha_active
    var direct_active = dora_active or oft_active

    print("=== Wan2.2-T2V 14B (low-noise) REAL LoRA training loop (block-swap offload) ===")
    print("  arch: dim=", DIM, " H=", H, " Dh=", Dh, " ffn=", FFN, " num_blocks=", NUM_BLOCKS)
    print("  tokens: S=N_IMG=", N_IMG, " TXT=", TXT, " in_ch=", IN_CH, " out_ch=", OUT_CH)
    print("  recipe: rank=", train_cfg.lora_rank, " alpha=", train_cfg.lora_alpha, " lr=", train_cfg.lr,
          " shift=", TIMESTEP_SHIFT, " vae_shift=", VAE_SHIFT, " vae_scale=", VAE_SCALE)
    print("  fixed_sigma_smoke=", FIXED_SIGMA_SMOKE)
    print("  ckpt:", ckpt_path)
    print("  cache:", cache_dir)

    # ── resident frozen base (embeddings + head) ──────────────────────────────
    print("[load] Wan22StackBase (patch_embedding, text_embedding, time_embedding,")
    print("       time_projection, head)")
    var base_st = SafeTensors.open(ckpt_path.copy())
    var base = load_wan22_stack_base(base_st, ctx)
    print("[load] base resident")
    sampled_peak_vram = _record_vram_sample(
        String("after_base_resident"), ctx, sampled_peak_vram,
        WAN22_DIRECT_VRAM_BUDGET_BYTES,
    )

    # ── block-swap offload loader ─────────────────────────────────────────────
    var plan = build_wan22_14b_block_plan()
    var cfg = OffloadConfig.synchronous_single()
    var loader = TurboPlannedLoader.open(ckpt_path.copy(), plan^, cfg, ctx)
    print("[load] offload loader opened (", loader.block_count(), " blocks)")
    sampled_peak_vram = _record_vram_sample(
        String("after_offload_loader_open"), ctx, sampled_peak_vram,
        WAN22_DIRECT_VRAM_BUDGET_BYTES,
    )

    # ── RoPE tables ───────────────────────────────────────────────────────────
    # NOTE: for the compile smoke we use a placeholder rope table. In a full
    # run these should come from wan22_build_rope (models/dit/wan22_dit.mojo).
    # The rope tables enter the block forward as Tensor [S*H, Dh/2].
    var cos = _rope_placeholder(S * H, Dh // 2)
    var sin = _rope_placeholder(S * H, Dh // 2)
    print("[rope] placeholder tables (S*H=", S * H, " x Dh/2=", Dh // 2, ")")
    print("  TODO: replace with wan22_build_rope for real training.")

    # ── LoRA / LyCORIS carrier set (identity at init) ─────────────────────────
    var lora = build_wan22_lora_set(NUM_BLOCKS, DIM, train_cfg.lora_rank, train_cfg.lora_alpha)
    var lokr_masters = empty_flat_lokr_set()
    var loha_masters = empty_flat_loha_set()
    var dora_masters = empty_wan22_direct_dora_set()
    var oft_masters = empty_wan22_direct_oft_set()
    if lokr_active:
        lokr_masters = _build_wan22_lokr(train_cfg)
        var carrier_bytes = flat_lokr_carrier_total_bytes(lokr_masters)
        print("[Wan22-lokr] carrier device bytes:", carrier_bytes, " budget:", LOKR_CARRIER_MAX_DEVICE_BYTES)
        if carrier_bytes > LOKR_CARRIER_MAX_DEVICE_BYTES:
            raise Error(
                String("Wan22 LoKr carrier set needs ")
                + String(carrier_bytes) + String(" bytes on device (> budget ")
                + String(LOKR_CARRIER_MAX_DEVICE_BYTES) + String(")")
            )
        var carriers = flat_lokr_carrier_list(lokr_masters)
        lora = Wan22LoraSet(carriers^, NUM_BLOCKS, train_cfg.lora_rank)
    elif loha_active:
        loha_masters = _build_wan22_loha(train_cfg)
        var carrier_bytes = flat_loha_carrier_total_bytes(loha_masters)
        print("[Wan22-loha] carrier device bytes:", carrier_bytes, " budget:", LOKR_CARRIER_MAX_DEVICE_BYTES)
        if carrier_bytes > LOKR_CARRIER_MAX_DEVICE_BYTES:
            raise Error(
                String("Wan22 LoHa carrier set needs ")
                + String(carrier_bytes) + String(" bytes on device (> budget ")
                + String(LOKR_CARRIER_MAX_DEVICE_BYTES) + String(")")
            )
        var carriers = flat_loha_carrier_list(loha_masters)
        lora = Wan22LoraSet(carriers^, NUM_BLOCKS, train_cfg.lora_rank)
    elif dora_active:
        var dense = wan22_direct_dense_carrier_bytes(NUM_BLOCKS, DIM)
        var direct_est = wan22_direct_dora_preflight(
            NUM_BLOCKS, DIM, train_cfg.lora_rank, WAN22_DIRECT_VRAM_BUDGET_BYTES, False,
        )
        print("[Wan22-dora] streaming W_orig init; direct_est_bytes=", direct_est,
              " dense_full_delta_bytes=", dense)
        dora_masters = build_wan22_direct_dora_set_from_offload(
            loader, NUM_BLOCKS, DIM, train_cfg.lora_rank, train_cfg.lora_alpha,
            train_cfg.seed * UInt64(61) + UInt64(2201), False, ctx,
        )
        print("[Wan22-dora] trainable bytes:", wan22_direct_dora_trainable_bytes(dora_masters),
              " budget:", WAN22_DIRECT_VRAM_BUDGET_BYTES)
        sampled_peak_vram = _record_vram_sample(
            String("after_direct_dora_init"), ctx, sampled_peak_vram,
            WAN22_DIRECT_VRAM_BUDGET_BYTES,
        )
    elif oft_active:
        var dense = wan22_direct_dense_carrier_bytes(NUM_BLOCKS, DIM)
        var direct_est = wan22_direct_oft_preflight(
            NUM_BLOCKS, DIM, WAN22_DIRECT_OFT_BLOCK_SIZE, WAN22_DIRECT_VRAM_BUDGET_BYTES,
        )
        oft_masters = build_wan22_direct_oft_set(NUM_BLOCKS, DIM, WAN22_DIRECT_OFT_BLOCK_SIZE)
        print("[Wan22-oft] trainable bytes:", wan22_direct_oft_trainable_bytes(oft_masters),
              " direct_est_bytes=", direct_est, " dense_full_delta_bytes=", dense,
              " budget:", WAN22_DIRECT_VRAM_BUDGET_BYTES)
        sampled_peak_vram = _record_vram_sample(
            String("after_direct_oft_init"), ctx, sampled_peak_vram,
            WAN22_DIRECT_VRAM_BUDGET_BYTES,
        )
    var n_adapters = wan22_total_adapters(lora)
    print("[lora] adapters:", n_adapters, " (8 per block x", NUM_BLOCKS, " blocks)")
    print("  targets: self_attn.{q,k,v,o} + cross_attn.{q,k,v,o} per block")

    var b_absum_init = Float32(0.0)
    if lokr_active:
        b_absum_init = Float32(flat_lokr_zero_leg_l1(lokr_masters))
        print("[Wan22-lokr] zero-leg L1 at init =", b_absum_init, " (expect 0.0)")
    elif loha_active:
        b_absum_init = Float32(flat_loha_zero_leg_l1(loha_masters))
        print("[Wan22-loha] zero-leg L1 at init =", b_absum_init, " (expect 0.0)")
    elif dora_active:
        b_absum_init = Float32(wan22_direct_dora_zero_leg_l1(dora_masters))
        print("[Wan22-dora] zero-leg L1 at init =", b_absum_init, " (expect 0.0)")
    elif oft_active:
        b_absum_init = Float32(wan22_direct_oft_vec_l1(oft_masters))
        print("[Wan22-oft] vec L1 at init =", b_absum_init, " (expect 0.0)")
    else:
        for i in range(n_adapters):
            b_absum_init += _absum(lora.ad[i].b)
        print("[lora] LoRA-B |.|_1 at init =", b_absum_init, " (expect 0.0)")

    # ── cache ─────────────────────────────────────────────────────────────────
    # NOTE: In FIXED_SIGMA_SMOKE=True mode we use a synthetic sample so the
    # cache directory is NOT required to exist. A real run needs a prepared cache
    # in CACHE_DIR with .safetensors files containing "latent" [16*64*64] and
    # "t5_embed" [1, seq, 4096] tensors (same format as the chroma cache).
    var files = List[String]()
    comptime if not FIXED_SIGMA_SMOKE:
        files = _list_cache(cache_dir.copy())
        print("[cache] samples:", len(files))
    else:
        print("[cache] FIXED_SIGMA_SMOKE=True: using synthetic sample, no cache needed.")

    var first_loss = Float32(0.0)
    var last_loss = Float32(0.0)
    var train_start = perf_counter_ns()

    # MJ-1085: resident fused AdamW (persistent device P/M/V, one-time pinned
    # staging). The per-step fused arm allocates fresh pinned staging every step
    # and can hit the MJ-1070 unmapped-buffer segfault under pinned pressure.
    var adamw_dev_state = Optional[LoraAdamWPlainDeviceState](None)
    var adamw_state_ready = False

    # ── optional resume (plain-LoRA arm only): reload A/B (+ AdamW moments if a
    # FULL `.state` exists) BEFORE the loop so the lazy resident-AdamW init
    # (wan22_lora_adamw_state_init) seeds dev_m/dev_v from the restored moments —
    # full-moment fidelity (MJ-1088).
    if resume_path != String("") and not carrier_active and not direct_active:
        var resolved = _wan22_resolve_resume_path(resume_path)
        var is_full = lora_train_state_has_moments(
            resolved, String("blocks.0.self_attn.q")
        )
        if is_full:
            lora = load_wan22_lora_state(
                NUM_BLOCKS, DIM, train_cfg.lora_rank, train_cfg.lora_alpha, resolved, ctx
            )
            print("[wan22-resume] FULL resume (A/B + AdamW moments) from", resolved)
        else:
            _wan22_warn_warm_resume(resolved)
            lora = load_wan22_lora_resume(
                NUM_BLOCKS, DIM, train_cfg.lora_rank, train_cfg.lora_alpha, resolved, ctx
            )
        print("[wan22-resume] reloaded", wan22_total_adapters(lora), "adapters")

    for k in range(1, run_steps + 1):
        var t0 = perf_counter_ns()
        var step_seed = UInt64(1) if FIXED_SIGMA_SMOKE else UInt64(k)

        # ── load or synthesize cache sample ──
        var lat_raw: List[Float32]
        var txt_tokens: List[Float32]
        comptime if FIXED_SIGMA_SMOKE:
            # Synthetic: random normal latent + zero text tokens (noise-only smoke).
            lat_raw = _host_noise(LAT_C * LAT_H * LAT_W, SEED_BASE * UInt64(31) + step_seed)
            txt_tokens = List[Float32]()
            for _ in range(TXT * TEXT_DIM):
                txt_tokens.append(Float32(0.0))
        else:
            var slot = (k - 1) % len(files)
            var st = SafeTensors.open(files[slot])
            lat_raw = _load_host(st, String("latent"), ctx)
            var t5_info = st.tensor_info(String("t5_embed"))
            var t5_seq = Int(t5_info.shape[1])
            var t5_flat = _load_host(st, String("t5_embed"), ctx)
            txt_tokens = List[Float32]()
            for r in range(TXT):
                if r < t5_seq:
                    for c in range(TEXT_DIM):
                        txt_tokens.append(t5_flat[r * TEXT_DIM + c])
                else:
                    for _ in range(TEXT_DIM):
                        txt_tokens.append(Float32(0.0))

        # ── VAE shift/scale then patchify ──
        for i in range(len(lat_raw)):
            lat_raw[i] = (lat_raw[i] - VAE_SHIFT) * VAE_SCALE
        var latent_packed = _pack_latents(lat_raw)

        # ── timestep ──
        var sigma_idx: Int
        if FIXED_SIGMA_SMOKE:
            sigma_idx = FIXED_SIGMA_IDX
        else:
            var sigma = sample_timestep_logit_normal(SEED_BASE + step_seed, TIMESTEP_SHIFT)
            sigma_idx = Int(sigma * Float32(NUM_TRAIN_TIMESTEPS))
            if sigma_idx > NUM_TRAIN_TIMESTEPS - 1:
                sigma_idx = NUM_TRAIN_TIMESTEPS - 1
        var sig = Float32(sigma_idx + 1) / Float32(NUM_TRAIN_TIMESTEPS)
        var t_model = Float32(sigma_idx) / Float32(NUM_TRAIN_TIMESTEPS)

        # ── flow-match ──
        var noise = _host_noise(S * IN_CH, SEED_BASE * UInt64(7919) + step_seed)
        var noisy = List[Float32]()
        var target = List[Float32]()
        for i in range(len(latent_packed)):
            noisy.append(noise[i] * sig + latent_packed[i] * (Float32(1.0) - sig))
            target.append(noise[i] - latent_packed[i])

        # ── forward ──
        var fwd: Wan22StackForward
        if dora_active:
            fwd = wan22_stack_direct_dora_forward_offload[H, Dh, S, TXT](
                noisy.copy(), txt_tokens.copy(), t_model,
                base, loader, dora_masters,
                cos.copy(), sin.copy(),
                DIM, FFN, IN_CH, TEXT_DIM, OUT_CH, FREQ_DIM, EPS, ctx,
            )
        elif oft_active:
            fwd = wan22_stack_direct_oft_forward_offload[H, Dh, S, TXT](
                noisy.copy(), txt_tokens.copy(), t_model,
                base, loader, oft_masters,
                cos.copy(), sin.copy(),
                DIM, FFN, IN_CH, TEXT_DIM, OUT_CH, FREQ_DIM, EPS, ctx,
            )
        else:
            fwd = wan22_stack_lora_forward_offload[H, Dh, S, TXT](
                noisy.copy(), txt_tokens.copy(), t_model,
                base, loader, lora,
                cos.copy(), sin.copy(),
                DIM, FFN, IN_CH, TEXT_DIM, OUT_CH, FREQ_DIM, EPS, ctx,
            )
        sampled_peak_vram = _record_vram_sample(
            String("step_") + String(k) + String("_after_forward"),
            ctx, sampled_peak_vram, WAN22_DIRECT_VRAM_BUDGET_BYTES,
        )

        # ── loss = MSE(pred, target) ──
        var nout = len(fwd.out)
        var d_loss = List[Float32]()
        var sse = 0.0
        var inv_n = Float32(2.0) / Float32(nout)
        for i in range(nout):
            var diff = fwd.out[i] - target[i]
            sse += Float64(diff) * Float64(diff)
            d_loss.append(inv_n * diff)
        var loss = Float32(sse / Float64(nout))
        if k == 1:
            first_loss = loss
        last_loss = loss

        # ── grad norm + clip + optimizer ──
        var gn_before: Float64
        var progress_label = String("Wan22-lora")
        var nonfinite_grads = 0
        if dora_active:
            progress_label = String("Wan22-dora")
            var dg = wan22_stack_direct_dora_backward_offload[H, Dh, S, TXT](
                d_loss, noisy.copy(), txt_tokens.copy(),
                base, loader, dora_masters,
                cos.copy(), sin.copy(), fwd,
                DIM, FFN, IN_CH, TEXT_DIM, OUT_CH, FREQ_DIM, EPS, ctx,
            )
            var mnorm = wan22_direct_dora_grad_norm(dg.grads)
            gn_before = mnorm
            if mnorm > Float64(train_cfg.max_grad_norm):
                wan22_direct_dora_clip_grads(dg.grads, train_cfg.max_grad_norm / Float32(mnorm))
            wan22_direct_dora_adamw_step(
                dora_masters, dg.grads, k, train_cfg.lr,
                train_cfg.beta1, train_cfg.beta2, train_cfg.eps,
                train_cfg.weight_decay,
            )
            nonfinite_grads = dg.nonfinite_grads
            print("[Wan22-dora] step=", k, " grad_norm=", Float32(mnorm),
                  " zero_leg_l1=", wan22_direct_dora_zero_leg_l1(dora_masters))
        elif oft_active:
            progress_label = String("Wan22-oft")
            var og = wan22_stack_direct_oft_backward_offload[H, Dh, S, TXT](
                d_loss, noisy.copy(), txt_tokens.copy(),
                base, loader, oft_masters,
                cos.copy(), sin.copy(), fwd,
                DIM, FFN, IN_CH, TEXT_DIM, OUT_CH, FREQ_DIM, EPS, ctx,
            )
            var mnorm = wan22_direct_oft_grad_norm(og.grads)
            gn_before = mnorm
            if mnorm > Float64(train_cfg.max_grad_norm):
                wan22_direct_oft_clip_grads(og.grads, train_cfg.max_grad_norm / Float32(mnorm))
            wan22_direct_oft_adamw_step(
                oft_masters, og.grads, k, train_cfg.lr,
                train_cfg.beta1, train_cfg.beta2, train_cfg.eps,
                train_cfg.weight_decay,
            )
            nonfinite_grads = og.nonfinite_grads
            print("[Wan22-oft] step=", k, " grad_norm=", Float32(mnorm),
                  " vec_l1=", wan22_direct_oft_vec_l1(oft_masters))
        else:
            var grads = wan22_stack_lora_backward_offload[H, Dh, S, TXT](
                d_loss, noisy.copy(), txt_tokens.copy(),
                base, loader, lora,
                cos.copy(), sin.copy(), fwd,
                DIM, FFN, IN_CH, TEXT_DIM, OUT_CH, FREQ_DIM, EPS, ctx,
            )
            nonfinite_grads = grads.nonfinite_lora_grads
            if lokr_active:
                progress_label = String("Wan22-lokr")
                var mg = flat_lokr_chain_all(lokr_masters, grads.d_a, grads.d_b)
                var mnorm = flat_lokr_grad_norm(mg)
                gn_before = mnorm
                if mnorm > Float64(train_cfg.max_grad_norm):
                    flat_lokr_clip_grads(mg, train_cfg.max_grad_norm / Float32(mnorm))
                flat_lokr_adamw_step(
                    lokr_masters, mg, k, train_cfg.lr,
                    train_cfg.beta1, train_cfg.beta2, train_cfg.eps,
                    train_cfg.weight_decay,
                )
                var carriers = flat_lokr_carrier_list(lokr_masters)
                lora = Wan22LoraSet(carriers^, NUM_BLOCKS, train_cfg.lora_rank)
                print("[Wan22-lokr] step=", k, " master_grad_norm=", Float32(mnorm),
                      " zero_leg_l1=", flat_lokr_zero_leg_l1(lokr_masters))
            elif loha_active:
                progress_label = String("Wan22-loha")
                var mg = flat_loha_chain_all(loha_masters, grads.d_a, grads.d_b)
                var mnorm = flat_loha_grad_norm(mg)
                gn_before = mnorm
                if mnorm > Float64(train_cfg.max_grad_norm):
                    flat_loha_clip_grads(mg, train_cfg.max_grad_norm / Float32(mnorm))
                flat_loha_adamw_step(
                    loha_masters, mg, k, train_cfg.lr,
                    train_cfg.beta1, train_cfg.beta2, train_cfg.eps,
                    train_cfg.weight_decay,
                )
                var carriers = flat_loha_carrier_list(loha_masters)
                lora = Wan22LoraSet(carriers^, NUM_BLOCKS, train_cfg.lora_rank)
                print("[Wan22-loha] step=", k, " master_grad_norm=", Float32(mnorm),
                      " zero_leg_l1=", flat_loha_zero_leg_l1(loha_masters))
            else:
                gn_before = _clip(grads, train_cfg.max_grad_norm)
                if not adamw_state_ready:
                    adamw_dev_state = Optional[LoraAdamWPlainDeviceState](
                        wan22_lora_adamw_state_init(lora, ctx)
                    )
                    adamw_state_ready = True
                    print("[wan22-adamw] resident fused state initialized (",
                          wan22_total_adapters(lora), "adapters )")
                wan22_lora_adamw_step_resident(
                    adamw_dev_state.value(), lora, grads, k, train_cfg.lr, ctx,
                    train_cfg.beta1, train_cfg.beta2, train_cfg.eps,
                    train_cfg.weight_decay,
                )

        sampled_peak_vram = _record_vram_sample(
            String("step_") + String(k) + String("_after_backward_optimizer"),
            ctx, sampled_peak_vram, WAN22_DIRECT_VRAM_BUDGET_BYTES,
        )

        var t1 = perf_counter_ns()
        var secs = Float64(t1 - t0) / 1.0e9
        print_trainer_progress(
            progress_label, k, run_steps, 1,
            loss, Float64(gn_before), secs, 0.0,
            Float64(t1 - train_start) / 1.0e9,
        )
        if nonfinite_grads != 0:
            print("[", progress_label, "] warning nonfinite_grads=", nonfinite_grads)

    print("")
    print("first_loss=", first_loss, " last_loss=", last_loss)
    var b_absum_final = Float32(0.0)
    if lokr_active:
        b_absum_final = Float32(flat_lokr_zero_leg_l1(lokr_masters))
    elif loha_active:
        b_absum_final = Float32(flat_loha_zero_leg_l1(loha_masters))
    elif dora_active:
        b_absum_final = Float32(wan22_direct_dora_zero_leg_l1(dora_masters))
    elif oft_active:
        b_absum_final = Float32(wan22_direct_oft_vec_l1(oft_masters))
    else:
        for i in range(n_adapters):
            b_absum_final += _absum(lora.ad[i].b)
    var trains = (b_absum_init == 0.0) and (b_absum_final > 0.0)
    if trains and (last_loss == last_loss):
        var train_label = String("LyCORIS trainable") if (carrier_active or direct_active) else String("LoRA-B")
        print("RESULT: REAL run OK — ", train_label, " grew 0 ->", b_absum_final,
              "; loss", first_loss, "->", last_loss,
              (" (DECREASED)" if last_loss < first_loss else " (see trajectory)"))
        var save_path = String(LORA_DIR) + String("/wan22_lora_smoke.safetensors")
        if lokr_active:
            _ = save_flat_lokr(lokr_masters, save_path, ctx)
        elif loha_active:
            _ = save_flat_loha(loha_masters, save_path, ctx)
        elif dora_active:
            _ = save_wan22_direct_dora(dora_masters, save_path, ctx)
        elif oft_active:
            _ = save_wan22_direct_oft(oft_masters, save_path, ctx)
        else:
            _ = save_wan22_lora(lora, save_path, ctx)
            # MJ-1088: also write the moment-carrying `.state` sidecar so a resume
            # does NOT zero AdamW momentum. Pull resident device m/v back to host
            # first (the resident step syncs params each step, not moments).
            if adamw_state_ready:
                lora_adamw_plain_device_state_sync_params(
                    adamw_dev_state.value(), lora.ad, ctx
                )
                lora_adamw_plain_device_state_sync_moments(
                    adamw_dev_state.value(), lora.ad, ctx
                )
            var meta = List[Float32]()
            meta.append(Float32(run_steps))
            meta.append(Float32(Int(train_cfg.seed)))
            var state_path = save_path + String(".state")
            _ = save_wan22_lora_state(lora, state_path, ctx, meta^)
            print("[wan22-lora] save_state (A/B + AdamW moments) path=", state_path)
        sampled_peak_vram = _record_vram_sample(
            String("after_save"), ctx, sampled_peak_vram,
            WAN22_DIRECT_VRAM_BUDGET_BYTES,
        )
    else:
        print("RESULT: FAIL trains=", trains)
    print("[Wan22-vram] sampled_peak_used_bytes:", sampled_peak_vram,
          " budget_bytes:", WAN22_DIRECT_VRAM_BUDGET_BYTES)
