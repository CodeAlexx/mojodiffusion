# serenitymojo/training/train_mageflow_real.mojo
#
# REAL Mage-Flow-Base LoRA TRAINING LOOP. Pure Mojo + MAX, GPU, block-swap
# offload. Wires the parity-gated engine (models/mageflow/mageflow_stack_lora
# .mojo over the BANKED qwenimage double_block_lora_forward/backward, gated by
# models/mageflow/parity/mageflow_block_lora_parity.mojo PASS) into a
# flow-match training loop. This file is the LOOP only — no block/stack math.
# Structure mirrors training/train_wan22_real.mojo::_run_wan21_train.
#
# ── FLOW-MATCH RECIPE (house recipe; the mage_flow reference repo
#    /home/alex/Mage/mage_flow is INFERENCE-ONLY — pipeline.py/inference.py/
#    app.py, no training script — so the training convention is the serenitymojo
#    flow-match spine, consistent with the mage inference contract) ────────────
#   * sigma ~ training/schedule.mojo::sample_timestep_logit_normal(seed, shift):
#       t = sigmoid(N(0,1)); sigma = shift*t/(1+(shift-1)*t), clamped
#       [1/1000, 1]  (schedule.mojo:253-279). shift = 6.0 — the Base
#       transformer config.json `static_shift: 6.0`, the SAME map the mage
#       inference scheduler applies to its sigmas (pipeline.py:43
#       `shift*s/(1+(shift-1)*s)`).
#   * noising/target = training/schedule.mojo::flow_match_noise_target
#       (schedule.mojo:544-560):  x_t = (1-sigma)*x0 + sigma*noise,
#       target = velocity = noise - x0.
#   * timestep fed to the model = the RAW sigma (the mage sinusoid folds
#       scale=1000), exactly like inference (pipeline.py:189 t_vec=sigma;
#       mageflow_dit.mojo::_mage_time_sinusoid).
#   * loss = training/levers.mojo::levers_loss_grad (MSE default, mean
#       reduction; config-selectable huber/smooth_l1/mae + min-SNR lever).
#   * optimizer = AdamW (betas 0.9/0.999, eps 1e-8, wd 0.01) over the 144 LoRA
#       adapters via the shared _lora_adamw; global grad-clip max_norm 1.0.
#
# ── GEOMETRY (comptime smoke: one 256x256 image) ─────────────────────────────
#   latent [1,128,16,16] -> N_IMG = 256 tokens x 128 ch (patch_size 1: token
#   (h,w) = latent[:, h, w]). Text: Qwen3-VL context [N_TXT=256, 2560]
#   (cache samples pad/truncate to N_TXT; LoRA block treats all rows as real).
#
# ── DATA ─────────────────────────────────────────────────────────────────────
#   MAGEFLOW_DATA_CACHE=<dir> reads the klein_dataset cache layout
#   (training/klein_dataset.mojo write_sample): latent [1,128,H/16,W/16] +
#   text_embedding [1,L,2560] + text_mask [1,L]. The CACHE BUILDER itself is
#   chunk 5 — this is the reader, kept ready. Without the env (or an empty
#   dir) the loop runs on deterministic SYNTHETIC x0/text so it is runnable
#   the moment weights land.
#
# ── WEIGHTS-ABSENT GUARD ─────────────────────────────────────────────────────
#   The Base transformer safetensors may still be DOWNLOADING; if absent we
#   print a banner and exit 0 (config/geometry/rope/flow-match still ran).
#
# ── BUILD / RUN ──────────────────────────────────────────────────────────────
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg
#   pixi run mojo run -I . serenitymojo/training/train_mageflow_real.mojo \
#       [serenitymojo/configs/mageflow_base_smoke.json]
#
# Mojo 1.0.0b1, NVIDIA GPU (16GB refit: block-swap offload, <=2 blocks in
# flight — see mageflow_stack_lora.mojo MEMORY DECISION).

from std.gpu.host import DeviceContext
from std.collections import List
from std.math import sqrt
from std.time import perf_counter_ns
from std.sys import argv
from std.ffi import external_call
from std.memory import alloc
from std.builtin.type_aliases import MutExternalOrigin

from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.ffi import sys_open, sys_close, O_RDONLY
from serenitymojo.io.train_config_reader import read_model_config
from serenitymojo.tensor import Tensor
from serenitymojo.ops.cast import cast_tensor

from serenitymojo.training.train_config import TrainConfig
from serenitymojo.training.schedule import (
    sample_timestep_logit_normal, flow_match_noise_target,
)
from serenitymojo.training.levers import levers_loss_grad
from serenitymojo.training.klein_dataset import (
    LATENT_KEY, TEXT_KEY, MASK_KEY, list_sorted_safetensors,
)

# PROVEN inference-geometry RoPE — image msrope rows, TEXT ROWS IDENTITY
# (cos=1/sin=0; cross-checked vs the real MageFlowEmbedRope by the chunk-1
# parity gate). Reused EXACTLY, not reimplemented.
from serenitymojo.models.dit.mageflow_dit import build_mageflow_rope_tables

from serenitymojo.offload.plan import OffloadConfig
from serenitymojo.offload.turbo_planned_loader import TurboPlannedLoader

from serenitymojo.models.mageflow.config import MageFlowTrainSpec
from serenitymojo.models.mageflow.weights import (
    MageFlowBase, load_mageflow_base, build_mageflow_block_plan,
    compute_mageflow_silu_temb,
)
from serenitymojo.models.mageflow.mageflow_stack_lora import (
    MageFlowLoraSet, MageFlowLoraGradSet,
    build_mageflow_lora_set, mageflow_total_adapters,
    mageflow_stack_lora_forward_offload, mageflow_stack_lora_backward_offload,
    mageflow_lora_adamw_step, save_mageflow_lora, save_mageflow_lora_state,
)


# ── Mage-Flow-Base architecture (comptime; confirmed vs transformer config) ───
comptime H = 24
comptime Dh = 128
comptime DIM = H * Dh            # 3072
comptime FFN = 12288
comptime IN_CH = 128
comptime OUT_CH = 128
comptime TXT_CH = 2560
comptime DEPTH = 12
comptime EPS = Float32(1.0e-6)
comptime ROPE_THETA = Float64(10000.0)

# ── smoke geometry: latent [1,128,16,16] (256px image) ────────────────────────
comptime FRAME = 1
comptime H_TOK = 16
comptime W_TOK = 16
comptime N_IMG = FRAME * H_TOK * W_TOK    # 256 image tokens
comptime N_TXT = 256                      # Qwen3-VL context rows (pad/trunc)
comptime S = N_IMG + N_TXT                # 512 joint tokens

# ── recipe defaults (config overrides where present) ──────────────────────────
comptime DEFAULT_LR = Float32(1.0e-4)
comptime DEFAULT_MAX_GRAD_NORM = Float32(1.0)
comptime DEFAULT_FLOW_SHIFT = Float32(6.0)   # Base config.json static_shift
comptime DEFAULT_CKPT =
    "/home/alex/.serenity/models/checkpoints/Mage-Flow-Base/transformer/diffusion_pytorch_model.safetensors"
comptime DEFAULT_CONFIG = "serenitymojo/configs/mageflow_base_smoke.json"


# ── file existence (pure syscall; no builtin open) ────────────────────────────
def _file_exists(path: String) -> Bool:
    var fd = sys_open(path, O_RDONLY, Int32(0))
    if fd < 0:
        return False
    _ = sys_close(fd)
    return True


# ── libc getenv -> String ("" if unset); wan22 trainer helper ─────────────────
comptime _EnvPtr = UnsafePointer[UInt8, MutExternalOrigin]


def _env_str(name: String) -> String:
    var n = name.byte_length()
    var buf = alloc[UInt8](n + 1)
    var src = name.as_bytes()
    for i in range(n):
        buf[i] = src[i]
    buf[n] = 0
    var ret = external_call["getenv", _EnvPtr](_EnvPtr(unsafe_from_address=Int(buf)))
    buf.free()
    var out = String("")
    if Int(ret) == 0:
        return out
    var i = 0
    while ret[i] != UInt8(0):
        out += chr(Int(ret[i]))
        i += 1
    return out^


# ── deterministic host RNG (splitmix64 + Box-Muller; wan22 trainer helpers) ───
def _splitmix64(state: UInt64) -> UInt64:
    var z = state + UInt64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
    return z ^ (z >> 31)


def _u01(seed: UInt64) -> Float32:
    var w = _splitmix64(seed)
    return Float32(Float64(w >> 40) * (1.0 / 16777216.0))


def _fcos(x: Float32) -> Float32:
    from std.math import cos as _c
    return Float32(_c(Float64(x)))


def _flog(x: Float32) -> Float32:
    from std.math import log as _l
    return Float32(_l(Float64(x)))


def _randn(seed: UInt64) -> Float32:
    var u1 = _u01(seed * UInt64(2654435761) + 1)
    var u2 = _u01(seed * UInt64(1442695040888963407) + 2)
    if u1 < Float32(1.0e-7):
        u1 = Float32(1.0e-7)
    var r = sqrt(Float32(-2.0) * _flog(u1))
    var ang = Float32(6.2831853071795864769) * u2
    return r * _fcos(ang)


def _noise(n: Int, seed: UInt64) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        out.append(_randn(seed + UInt64(i) * 2 + 1))
    return out^


def _synth(n: Int, seed: UInt64) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        out.append(_randn(seed * UInt64(1099511628211) + UInt64(i) * 3 + 5))
    return out^


# ── global grad clip over all LoRA d_A/d_B (max_grad_norm) ────────────────────
def _clip(mut grads: MageFlowLoraGradSet, max_norm: Float32) -> Float32:
    var ss = Float32(0.0)
    for i in range(len(grads.d_a)):
        for j in range(len(grads.d_a[i])):
            ss += grads.d_a[i][j] * grads.d_a[i][j]
        for j in range(len(grads.d_b[i])):
            ss += grads.d_b[i][j] * grads.d_b[i][j]
    var gn = sqrt(ss)
    if gn > max_norm and gn > Float32(0.0):
        var scl = max_norm / gn
        for i in range(len(grads.d_a)):
            for j in range(len(grads.d_a[i])):
                grads.d_a[i][j] = grads.d_a[i][j] * scl
            for j in range(len(grads.d_b[i])):
                grads.d_b[i][j] = grads.d_b[i][j] * scl
    return gn


# ══════════════════════════════════════════════════════════════════════════════
# REAL DATA CACHE READER (klein_dataset layout; cache BUILDER = chunk 5).
#   latent          [1,128,H/16,W/16]  (any float dtype; smoke requires 16x16)
#   text_embedding  [1,L,2560]         (pad/trunc to N_TXT rows)
#   text_mask       [1,L]              (optional; rows beyond L zero-padded)
# Latent -> tokens: patch_size 1, token t=(h*W+w) gets feature c from
# latent[0,c,h,w] (channel-slowest storage -> tokens[t*128+c] = lat[c*HW+t]).
# ══════════════════════════════════════════════════════════════════════════════
def _st_host_f32_any(st: SafeTensors, name: String, ctx: DeviceContext) raises -> List[Float32]:
    from serenitymojo.io.tensor_view import from_parts
    var info = st.tensor_info(name)
    var bytes = st.tensor_bytes(name)
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    var t = Tensor.from_view(tv, ctx)
    if t.dtype() == STDtype.F32:
        return t.to_host(ctx)
    return cast_tensor(t, STDtype.F32, ctx).to_host(ctx)


def _mstat(vals: List[Float32]) -> Tuple[Float32, Float32]:
    # print-only per-sample mean/std (cross-check vs the cache builder's stats)
    var n = len(vals)
    var s = Float64(0.0)
    var s2 = Float64(0.0)
    for i in range(n):
        var v = Float64(vals[i])
        s += v
        s2 += v * v
    var m = s / Float64(n)
    var vv = s2 / Float64(n) - m * m
    if vv < 0.0:
        vv = 0.0
    from std.math import sqrt as _sq
    return (Float32(m), Float32(_sq(vv)))


def _load_mageflow_cache_sample(
    path: String, ctx: DeviceContext, mut txt_len: Int,
) raises -> List[List[Float32]]:
    var st = SafeTensors.open(path)

    # ── latent [1,128,16,16] -> x0 tokens [N_IMG, 128] ─────────────────────────
    var lat_info = st.tensor_info(String(LATENT_KEY))
    if len(lat_info.shape) != 4 or lat_info.shape[1] != IN_CH:
        raise Error(String("cache latent must be [1,128,H,W] in ") + path)
    var lh = lat_info.shape[2]
    var lw = lat_info.shape[3]
    if lh != H_TOK or lw != W_TOK:
        raise Error(
            String("cache latent grid ") + String(lh) + "x" + String(lw)
            + " != comptime smoke grid 16x16 in " + path
        )
    var lat = _st_host_f32_any(st, String(LATENT_KEY), ctx)   # [128*256] ch-slowest
    var x0 = List[Float32]()
    var hw = H_TOK * W_TOK
    for t in range(hw):
        for c in range(IN_CH):
            x0.append(lat[c * hw + t])

    # ── text_embedding [1,L,2560] -> [N_TXT, 2560] (pad/trunc) ────────────────
    var txt_info = st.tensor_info(String(TEXT_KEY))
    if len(txt_info.shape) != 3 or txt_info.shape[2] != TXT_CH:
        raise Error(String("cache text_embedding must be [1,L,2560] in ") + path)
    var L = txt_info.shape[1]
    var emb = _st_host_f32_any(st, String(TEXT_KEY), ctx)
    var txt = List[Float32]()
    var rows = L if L < N_TXT else N_TXT
    for r in range(rows):
        for c in range(TXT_CH):
            txt.append(emb[r * TXT_CH + c])
    for _ in range((N_TXT - rows) * TXT_CH):
        txt.append(0.0)
    txt_len = rows

    var out = List[List[Float32]]()
    out.append(x0^)
    out.append(txt^)
    return out^


def main() raises:
    var ctx = DeviceContext()
    var spec = MageFlowTrainSpec.mageflow_base()

    var cfg_path = String(DEFAULT_CONFIG)
    var args = argv()
    if len(args) > 1:
        cfg_path = String(args[1])
    var cfg = read_model_config(cfg_path)

    var ckpt = cfg.checkpoint if cfg.checkpoint.byte_length() > 0 \
        else String(DEFAULT_CKPT)
    var rank = cfg.lora_rank if cfg.lora_rank > 0 else spec.default_rank
    var alpha = cfg.lora_alpha if cfg.lora_alpha > Float32(0.0) else spec.default_alpha
    var lr = cfg.lr if cfg.lr > Float32(0.0) else spec.default_lr
    var max_grad_norm = cfg.max_grad_norm if cfg.max_grad_norm > Float32(0.0) \
        else DEFAULT_MAX_GRAD_NORM
    var steps = cfg.max_steps if cfg.max_steps > 0 else 1
    var seed = cfg.seed
    var beta1 = cfg.beta1 if cfg.beta1 > Float32(0.0) else Float32(0.9)
    var beta2 = cfg.beta2 if cfg.beta2 > Float32(0.0) else Float32(0.999)
    var opt_eps = cfg.eps if cfg.eps > Float32(0.0) else Float32(1.0e-8)
    var weight_decay = cfg.weight_decay if cfg.weight_decay >= Float32(0.0) \
        else Float32(0.01)
    var flow_shift = cfg.timestep_shift if cfg.timestep_shift > Float32(0.0) \
        else DEFAULT_FLOW_SHIFT

    var out_path = cfg.output_model_destination
    if out_path.byte_length() == 0:
        out_path = String("/home/alex/mojodiffusion/output/mageflow_lora/") \
            + String("mageflow_base_lora.safetensors")

    print("==== Mage-Flow-Base LoRA trainer — flow-match (shift", flow_shift, ") ====")
    print("arch: D=", DIM, " blocks=", DEPTH, " heads=", H, " head_dim=", Dh,
          " ffn=", FFN, " in/out=", IN_CH, "/", OUT_CH, " ctx=", TXT_CH)
    print("geometry: latent[1,128,", H_TOK, ",", W_TOK, "] -> N_IMG=", N_IMG,
          " N_TXT=", N_TXT, " S=", S)
    print("lora: rank=", rank, " alpha=", alpha,
          " (12 targets/block x", DEPTH, "blocks = 144 adapters)")
    print("optim: lr=", lr, " max_grad_norm=", max_grad_norm,
          " betas=(", beta1, ",", beta2, ") eps=", opt_eps, " wd=", weight_decay)
    print("timestep: logit-normal + shift", flow_shift,
          " (schedule.mojo sample_timestep_logit_normal); model gets RAW sigma")
    print("steps:", steps, " seed:", seed)
    print("ckpt:", ckpt)

    # ── weights-absent guard (Base transformer DOWNLOADING) ───────────────────
    if not _file_exists(ckpt):
        print("")
        print("[mageflow] weights not present yet, skipping real step:")
        print("          ", ckpt)
        print("           (Base transformer downloading — config/geometry/rope/",
              "flow-match wired + type-checked; re-run once safetensors lands.)")
        print("RESULT: mageflow trainer wired OK (weights-absent path, exit 0)")
        return

    # ══════════════════════════════════════════════════════════════════════════
    # REAL STEP: resident non-block base + block-swap loader + 144-adapter LoRA.
    # ══════════════════════════════════════════════════════════════════════════
    print("")
    print("[mageflow] checkpoint present — loading resident base + block loader")
    var base_st = SafeTensors.open(ckpt)
    var base = load_mageflow_base(base_st, spec, ctx)   # header shapes CONFIRMED here
    var plan = build_mageflow_block_plan(DEPTH)
    var loader = TurboPlannedLoader.open(
        ckpt, plan^, OffloadConfig.synchronous_single(), ctx, False
    )

    var lora = build_mageflow_lora_set(DEPTH, DIM, FFN, rank, alpha)
    print("[lora] adapters:", mageflow_total_adapters(lora))

    # MageFlow msrope: image rows roped, text rows identity — built ONCE
    # (constant geometry), reused across all blocks fwd+bwd.
    var rope = build_mageflow_rope_tables(
        FRAME, H_TOK, W_TOK, N_TXT, H, ROPE_THETA, STDtype.F32, ctx
    )

    # P3: config-first data cache (cfg.dataset_cache_dir ← JSON cache_dir /
    # dataset_cache_dir); env MAGEFLOW_DATA_CACHE OVERRIDES when set. An
    # explicit dir with no .safetensors FAILS LOUD (no silent synthetic).
    var cache_dir = _env_str(String("MAGEFLOW_DATA_CACHE"))
    if cache_dir.byte_length() == 0:
        cache_dir = cfg.dataset_cache_dir
    var cache_files = List[String]()
    if cache_dir.byte_length() > 0:
        cache_files = list_sorted_safetensors(cache_dir)
        if len(cache_files) == 0:
            raise Error(
                String("[data] no .safetensors at cache dir: ") + cache_dir
                + String(" (env MAGEFLOW_DATA_CACHE > config cache_dir; fail-loud)")
            )
    var use_cache = len(cache_files) > 0
    if use_cache:
        print("[data] REAL cache:", cache_dir, " samples=", len(cache_files),
              " (round-robin; klein_dataset layout)")
    else:
        print("[data] SYNTHETIC x0/text (set config cache_dir or env",
              "MAGEFLOW_DATA_CACHE=<dir>; cache builder = chunk 5)")

    var first_loss = Float32(0.0)
    var last_loss = Float32(0.0)
    var have_first = False
    var train_start = perf_counter_ns()

    print("")
    print("step  sigma     loss           grad_norm     sec")
    for step in range(steps):
        var t0 = perf_counter_ns()

        # sigma stream: seed + step (levers.mojo:104 stream convention).
        var sigma = sample_timestep_logit_normal(seed + UInt64(step), flow_shift)

        var x0: List[Float32]
        var txt_raw: List[Float32]
        if use_cache:
            var idx = step % len(cache_files)
            var c_txtlen = 0
            var pair = _load_mageflow_cache_sample(cache_files[idx], ctx, c_txtlen)
            var lstat = _mstat(pair[0])
            print("  [cache] sample", idx, " (", cache_files[idx], ") txt_len=",
                  c_txtlen, " latent mean=", lstat[0], " std=", lstat[1])
            txt_raw = pair.pop()
            x0 = pair.pop()
        else:
            x0 = _synth(N_IMG * IN_CH, seed + UInt64(step) * 101 + 1)
            txt_raw = _synth(N_TXT * TXT_CH, seed + UInt64(step) * 777 + 9)

        # flow-match: x_t=(1-sigma)x0+sigma*noise, target=noise-x0
        # (schedule.mojo::flow_match_noise_target — noise stream seed*7919+step).
        var noise_h = _noise(N_IMG * IN_CH, seed * UInt64(7919) + UInt64(step))
        var fm = flow_match_noise_target(
            Tensor.from_host(x0.copy(), [N_IMG, IN_CH], STDtype.F32, ctx),
            sigma,
            Tensor.from_host(noise_h^, [N_IMG, IN_CH], STDtype.F32, ctx),
            ctx,
        )
        var img_tokens = fm.x_t.to_host(ctx)
        var target = fm.target.to_host(ctx)

        # temb: mage bf16-rounded sinusoid pipeline; RAW sigma in.
        var silu_temb = compute_mageflow_silu_temb(base, sigma, spec, ctx)

        var fwd = mageflow_stack_lora_forward_offload[H, Dh, N_IMG, N_TXT, S](
            img_tokens, txt_raw, silu_temb, base, loader, lora,
            rope[0], rope[1], DIM, FFN, IN_CH, TXT_CH, OUT_CH, EPS, ctx,
        )

        # loss + d_pred through the shared levers entry (MSE default).
        var lg = levers_loss_grad(fwd.out, target, sigma, cfg)
        if not have_first:
            first_loss = lg.loss
            have_first = True
        last_loss = lg.loss

        var grads = mageflow_stack_lora_backward_offload[H, Dh, N_IMG, N_TXT, S](
            lg.d_pred, silu_temb, base, loader, lora,
            rope[0], rope[1], fwd, DIM, FFN, OUT_CH, EPS, ctx,
        )
        var gn = _clip(grads, max_grad_norm)
        mageflow_lora_adamw_step(lora, grads, step + 1, lr, ctx,
                                 beta1, beta2, opt_eps, weight_decay)
        var secs = Float64(perf_counter_ns() - t0) / 1.0e9
        print(step, "  ", sigma, "  ", lg.loss, "  ", gn, "  ", secs)
        if grads.nonfinite_lora_grads != 0:
            print("  !! nonfinite lora grads =", grads.nonfinite_lora_grads)

    var npairs = save_mageflow_lora(lora, out_path, ctx)
    var meta = List[Float32]()
    meta.append(Float32(steps))
    meta.append(Float32(Int(seed)))
    var state_path = out_path + String(".state")
    var nstate = save_mageflow_lora_state(lora, state_path, ctx, meta^)
    print("")
    print("[save] wrote", npairs, "PEFT pairs (diffusion_model.* keys) ->", out_path)
    print("[save] wrote", nstate, "state tensors ->", state_path)
    print("trained steps=", steps, " first loss=", first_loss,
          " last loss=", last_loss,
          " total sec=", Float64(perf_counter_ns() - train_start) / 1.0e9)
    print("RESULT: mageflow Base LoRA trainer ran (block math == chunk-1",
          "parity-gated engine; offload streaming, 16GB refit).")
