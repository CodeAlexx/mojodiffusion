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
# ── GEOMETRY (comptime production: one 512x512 image) ────────────────────────
#   latent [1,128,32,32] -> N_IMG = 1024 tokens x 128 ch (patch_size 1: token
#   (h,w) = latent[:, h, w]). Text: Qwen3-VL context [N_TXT=256, 2560]
#   (cache samples pad/truncate to N_TXT; LoRA block treats all rows as real).
#
# ── SAVE CADENCE ─────────────────────────────────────────────────────────────
#   cfg.save_every > 0 -> step-numbered artifacts every save_every optimizer
#   steps (<prefix>_step{N}.safetensors + .state — the ltx2/wan22 house
#   pattern) + the final save at out_path.
#
# ── IN-TRAIN SAMPLING (cfg.sample_every > 0) ─────────────────────────────────
#   3 baked prompts (trigger vrtlEri2) at 1024x1024 (SH=64, N_IMG=4096 — the
#   PROVEN mageflow pipeline monomorphization), sampled at step 0 (pre-train
#   baseline) and every sample_every optimizer steps. Text conds are encoded
#   ONCE at startup (Qwen3-VL loaded + freed BEFORE the trainer loads base +
#   block loader — the pipeline's offload staging), held host-side. Each sample
#   point runs a 20-step flow-match Euler denoise (build_sigma_schedule, shift
#   6.0, RAW sigma to the model, bf16 latent at step boundaries — the pipeline
#   trajectory discipline) through mageflow_stack_lora_sample_forward: the SAME
#   gated LoRA block math as training with the CURRENT adapters applied live
#   (the klein_sampler validation pattern), streaming blocks through the
#   trainer's own TurboPlannedLoader (no second loader, no saved activations).
#   CFG 5.0 (Mage-Flow-Base default): cond + uncond (empty-prompt) forwards per
#   step, pred = uncond + cfg*(cond-uncond). MageVAE decode (weights loaded /
#   freed per call) -> samples/step{N}_p{i}.png.
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

from max.gpu.host import DeviceContext
from std.collections import List
from std.math import sqrt
from std.time import perf_counter_ns
from std.sys import argv
from std.ffi import external_call
from std.memory import alloc

from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.ffi import sys_open, sys_close, O_RDONLY
from serenitymojo.io.train_config_reader import read_model_config
from serenitymojo.tensor import Tensor
from serenitymojo.ops.cast import cast_tensor

from serenitymojo.training.train_config import TrainConfig
from serenitymojo.training.schedule import (
    sample_timestep_logit_normal, flow_match_noise_target,
    sample_timestep_logit_normal_win,
)
from serenitymojo.training.levers import levers_loss_grad
from serenitymojo.training.klein_dataset import (
    LATENT_KEY, TEXT_KEY, MASK_KEY, list_sorted_safetensors,
)
from serenitymojo.training.trainer_core import (
    trainer_resolve_resume_path, trainer_warn_warm_resume,
)
from serenitymojo.training.lora_save import (
    lora_train_state_has_moments, load_lora_train_state_meta,
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
    load_mageflow_lora_state, load_mageflow_lora_resume, mageflow_lora_prefixes,
    mageflow_stack_lora_sample_forward,
    # DEVICE arms (resident weights + device activations + fused AdamW):
    mageflow_stack_lora_forward_offload_device,
    mageflow_stack_lora_backward_offload_device,
    mageflow_stack_lora_sample_forward_device,
    mageflow_lora_adamw_state_init, mageflow_lora_adamw_step_resident,
    mageflow_lora_adamw_sync_moments,
)

# in-train sampling reuses the PROVEN pipeline pieces: tokenizer+template,
# seeded Gaussian init, sigma schedule, MageVAE decode->PNG.
from serenitymojo.io.ffi import sys_system
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.sampling.flow_match import build_sigma_schedule
from serenitymojo.pipeline.mageflow_pipeline import (
    mageflow_tokenize, gaussian_noise, mageflow_decode_latent,
)
from serenitymojo.image.png import save_png, ValueRange
from serenitymojo.models.text_encoder.mageflow_qwen3vl import (
    encode_mageflow_text, load_krea2_qwen3vl_4b, MAGEFLOW_T2I_DROP_IDX,
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

# ── training geometry: latent [1,128,32,32] (512px image) ─────────────────────
comptime FRAME = 1
comptime H_TOK = 32
comptime W_TOK = 32
comptime N_IMG = FRAME * H_TOK * W_TOK    # 1024 image tokens
comptime N_TXT = 256                      # Qwen3-VL context rows (pad/trunc)
comptime S = N_IMG + N_TXT                # 1280 joint tokens

# ── in-train sampling geometry: 1024x1024 (proven pipeline monomorphization) ──
comptime MF_BASE_DIR = "/home/alex/.serenity/models/checkpoints/Mage-Flow-Base"
comptime MF_TE_DIR = MF_BASE_DIR + "/text_encoder"
comptime MF_TOK_JSON = MF_TE_DIR + "/tokenizer.json"
comptime MF_VAE_PATH = MF_BASE_DIR + "/vae/diffusion_pytorch_model.safetensors"
comptime SAMPLE_SH = 64                       # 1024/16 latent side
comptime SAMPLE_NIMG = SAMPLE_SH * SAMPLE_SH  # 4096 image tokens
comptime SAMPLE_STEPS = 20                    # Mage-Flow-BASE needs ~20 (not turbo-4)
comptime SAMPLE_CFG = Float32(5.0)            # base default; 1.0 => single-forward

# 3 fixed sample prompts (trigger vrtlEri2) + the empty uncond prompt. keep
# token counts (L_full - drop 34) measured with the Base tokenizer via the
# mageflow_count_tokens probe pattern; VERIFIED at runtime against the real
# tokenizer output in _encode_sample_prompts (fail loud on drift).
comptime SP1 = (
    "vrtlEri2, a beautiful woman, standing in a sunlit olive garden, white"
    " summer dress, golden hour portrait, photorealistic"
)
comptime SP2 = (
    "vrtlEri2, a beautiful woman, in a black leather jacket on a neon-lit city"
    " street at night, cinematic"
)
comptime SP3 = (
    "close-up portrait of vrtlEri2, a beautiful woman, smiling, soft studio"
    " light, grey backdrop, 85mm"
)
comptime SP1_KEEP = 34
comptime SP2_KEEP = 31
comptime SP3_KEEP = 32
comptime UNCOND_KEEP = 5

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
            + " != comptime training grid " + String(H_TOK) + "x"
            + String(W_TOK) + " in " + path
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


# ══════════════════════════════════════════════════════════════════════════════
# IN-TRAIN SAMPLING (1024², live LoRA, trainer loader reused)
# ══════════════════════════════════════════════════════════════════════════════
def _bf16r(x: Float32) -> Float32:
    # host bf16 RNE round-trip (the pipeline latent-boundary discipline; the
    # same cast idiom weights.mojo::mage_time_sinusoid_host uses).
    return x.cast[DType.bfloat16]().cast[DType.float32]()


def _derive_step_from_name(path: String) -> Int:
    # `<prefix>_step{N}.safetensors[.state]` -> N (the ltx2/wan22 house
    # artifact naming). Uses the LAST `_step{digits}` run in the path; 0 when
    # the name carries none (e.g. the final out_path artifact).
    var b = path.as_bytes()
    var best = -1
    var i = 0
    while i + 5 < len(b):
        if (b[i] == UInt8(95)          # '_'
                and b[i + 1] == UInt8(115)   # 's'
                and b[i + 2] == UInt8(116)   # 't'
                and b[i + 3] == UInt8(101)   # 'e'
                and b[i + 4] == UInt8(112)): # 'p'
            var j = i + 5
            var n = 0
            var got = False
            while j < len(b) and b[j] >= UInt8(48) and b[j] <= UInt8(57):
                n = n * 10 + Int(b[j] - UInt8(48))
                j += 1
                got = True
            if got:
                best = n
        i += 1
    return best if best >= 0 else 0


def _dirname(path: String) -> String:
    var b = path.as_bytes()
    var last = -1
    for i in range(len(b)):
        if b[i] == UInt8(47):   # '/'
            last = i
    if last <= 0:
        return String(".")
    var out = List[UInt8]()
    for i in range(last):
        out.append(b[i])
    return String(unsafe_from_utf8=out)


# Encode the 3 sample prompts + the empty uncond prompt ONCE. The 8.9 GB
# Qwen3-VL encoder is loaded HERE and freed on return — the caller invokes this
# BEFORE loading the resident base + block loader (the pipeline's staging), so
# the encoder and the streamed DiT never co-reside. Returns 4 host-F32 conds
# [keep*2560] in order [SP1, SP2, SP3, uncond]; verifies each keep count
# against the baked comptime (fail loud on tokenizer drift).
def _encode_sample_prompts(ctx: DeviceContext) raises -> List[List[Float32]]:
    var prompts = List[String]()
    prompts.append(String(SP1))
    prompts.append(String(SP2))
    prompts.append(String(SP3))
    prompts.append(String(""))
    var expected = List[Int]()
    expected.append(SP1_KEEP)
    expected.append(SP2_KEEP)
    expected.append(SP3_KEEP)
    expected.append(UNCOND_KEEP)

    print("[sample] encoding 3 prompts + uncond (Qwen3-VL loaded ONCE, freed",
          "before the trainer's loader) ...")
    var enc = load_krea2_qwen3vl_4b(String(MF_TE_DIR), ctx)
    var out = List[List[Float32]]()
    for i in range(len(prompts)):
        var ids = mageflow_tokenize(String(MF_TOK_JSON), prompts[i])
        var keep = len(ids) - MAGEFLOW_T2I_DROP_IDX
        if keep != expected[i]:
            raise Error(
                String("_encode_sample_prompts: prompt ") + String(i)
                + " keep=" + String(keep) + " != baked comptime "
                + String(expected[i]) + " — recount tokens and rebake."
            )
        var txt = encode_mageflow_text(enc, ids, MAGEFLOW_T2I_DROP_IDX, ctx)
        out.append(cast_tensor(txt, STDtype.F32, ctx).to_host(ctx))
        print("[sample]   cond", i, "keep=", keep, "(",
              prompts[i].byte_length(), "chars )")
    print("[sample] conds cached host-side (encoder freed on return)")
    return out^


# Denoise ONE prompt at 1024² with the CURRENT LoRA applied live (the
# klein_sampler validation pattern: same gated block math as training), CFG
# cond/uncond, then MageVAE decode -> PNG. Streams blocks via the TRAINER's
# loader (blocks stream anyway; no second loader).
def _sample_one[
    NT: Int
](
    tag: String,
    txt_cond: List[Float32], txt_uncond: List[Float32],
    base: MageFlowBase, mut loader: TurboPlannedLoader,
    lora: MageFlowLoraSet, spec: MageFlowTrainSpec,
    flow_shift: Float32, seed: UInt64, out_png: String,
    ctx: DeviceContext, host_path: Bool,
) raises:
    comptime SC = SAMPLE_NIMG + NT
    comptime SU = SAMPLE_NIMG + UNCOND_KEEP
    var t0 = perf_counter_ns()

    var rope_c = build_mageflow_rope_tables(
        FRAME, SAMPLE_SH, SAMPLE_SH, NT, H, ROPE_THETA, STDtype.F32, ctx
    )
    var rope_u = build_mageflow_rope_tables(
        FRAME, SAMPLE_SH, SAMPLE_SH, UNCOND_KEEP, H, ROPE_THETA, STDtype.F32, ctx
    )
    var sigmas = build_sigma_schedule(SAMPLE_STEPS, flow_shift)
    var x = gaussian_noise(SAMPLE_NIMG * IN_CH, seed)   # F32 host

    for i in range(SAMPLE_STEPS):
        # bf16 latent at the step boundary (pipeline trajectory discipline).
        var x_bf = List[Float32]()
        for j in range(len(x)):
            x_bf.append(_bf16r(x[j]))
        var silu_temb = compute_mageflow_silu_temb(base, sigmas[i], spec, ctx)

        var vel: List[Float32]
        if host_path:
            vel = mageflow_stack_lora_sample_forward[H, Dh, SAMPLE_NIMG, NT, SC](
                x_bf, txt_cond, silu_temb, base, loader, lora,
                rope_c[0], rope_c[1], DIM, FFN, IN_CH, TXT_CH, OUT_CH, EPS, ctx,
            )
        else:
            vel = mageflow_stack_lora_sample_forward_device[
                H, Dh, SAMPLE_NIMG, NT, SC
            ](
                x_bf, txt_cond, silu_temb, base, loader, lora,
                rope_c[0], rope_c[1], DIM, FFN, IN_CH, TXT_CH, OUT_CH, EPS, ctx,
            )
        if SAMPLE_CFG != Float32(1.0):
            var vel_u: List[Float32]
            if host_path:
                vel_u = mageflow_stack_lora_sample_forward[
                    H, Dh, SAMPLE_NIMG, UNCOND_KEEP, SU
                ](
                    x_bf, txt_uncond, silu_temb, base, loader, lora,
                    rope_u[0], rope_u[1], DIM, FFN, IN_CH, TXT_CH, OUT_CH, EPS, ctx,
                )
            else:
                vel_u = mageflow_stack_lora_sample_forward_device[
                    H, Dh, SAMPLE_NIMG, UNCOND_KEEP, SU
                ](
                    x_bf, txt_uncond, silu_temb, base, loader, lora,
                    rope_u[0], rope_u[1], DIM, FFN, IN_CH, TXT_CH, OUT_CH, EPS, ctx,
                )
            for j in range(len(vel)):
                vel[j] = vel_u[j] + SAMPLE_CFG * (vel[j] - vel_u[j])

        var dt = sigmas[i + 1] - sigmas[i]
        for j in range(len(x)):
            x[j] = _bf16r(x_bf[j] + _bf16r(vel[j] * dt))
        print("[sample]   ", tag, " step", i + 1, "/", SAMPLE_STEPS,
              " sigma", sigmas[i], "->", sigmas[i + 1])

    # MageVAE decode (weights loaded/freed inside) -> PNG.
    var lat = Tensor.from_host(x^, [1, SAMPLE_NIMG, IN_CH], STDtype.F32, ctx)
    var rgb = mageflow_decode_latent[SAMPLE_NIMG, SAMPLE_SH, SAMPLE_SH](
        String(MF_VAE_PATH), lat, ctx
    )
    save_png(rgb, out_png, ctx, ValueRange.SIGNED)
    print("[sample]   ", tag, " saved:", out_png, " (",
          Float64(perf_counter_ns() - t0) / 1.0e9, "sec )")


# One sample point: all 3 prompts at the given step label. Seeds are FIXED per
# prompt (cfg seed + 100 + i) so progress across sample points is comparable.
def _run_sample_point(
    step_label: Int,
    conds: List[List[Float32]],
    base: MageFlowBase, mut loader: TurboPlannedLoader,
    lora: MageFlowLoraSet, spec: MageFlowTrainSpec,
    flow_shift: Float32, seed: UInt64, samples_dir: String,
    ctx: DeviceContext, host_path: Bool,
) raises:
    print("")
    print("[sample] === step", step_label, ": 3 prompts @ 1024², ",
          SAMPLE_STEPS, "-step, CFG", SAMPLE_CFG,
          "(cond+uncond per step), live LoRA ===")
    var pre = samples_dir + "/step" + String(step_label) + "_p"
    _sample_one[SP1_KEEP](
        String("p0(olive garden)"), conds[0], conds[3], base, loader, lora,
        spec, flow_shift, seed + UInt64(100), pre + "0.png", ctx, host_path,
    )
    _sample_one[SP2_KEEP](
        String("p1(neon street)"), conds[1], conds[3], base, loader, lora,
        spec, flow_shift, seed + UInt64(101), pre + "1.png", ctx, host_path,
    )
    _sample_one[SP3_KEEP](
        String("p2(85mm portrait)"), conds[2], conds[3], base, loader, lora,
        spec, flow_shift, seed + UInt64(102), pre + "2.png", ctx, host_path,
    )


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
    # TRAINING sigma-draw shift (decoupled 2026-07-22): train_timestep_shift
    # if set (>0), else the legacy coupling to timestep_shift. Inference/sample
    # schedules always use flow_shift.
    var train_shift = cfg.train_timestep_shift if cfg.train_timestep_shift > Float32(0.0) else flow_shift
    var warmup = cfg.lr_warmup_steps if cfg.lr_warmup_steps > 0 else 0

    # ── RESUME knobs: config keys resume_state / start_step / warm_resume
    #   (the webui override route); optional argv pair args[2]=<path>
    #   [args[3]=<step>] overrides (the per-backend slot-table route).
    #   start_step = GLOBAL optimizer steps already completed — the loop runs
    #   [start_step, steps); -1 = derive from the artifact `_step{N}` name. ──
    var resume_path = cfg.resume_state
    var start_step = cfg.start_step
    var warm_resume = cfg.warm_resume
    if len(args) > 2:
        resume_path = String(args[2])
    if len(args) > 3:
        start_step = Int(String(args[3]))
    if resume_path.byte_length() == 0:
        if start_step > 0:
            print("[resume] start_step", start_step,
                  "ignored (no resume_state) -> fresh run from 0")
        start_step = 0
    else:
        if start_step < 0:
            start_step = _derive_step_from_name(resume_path)
            print("[resume] start_step derived from artifact name:", start_step)
        if start_step >= steps:
            raise Error(
                String("[resume] start_step ") + String(start_step)
                + " >= max_steps " + String(steps) + " — nothing to train"
            )

    var out_path = cfg.output_model_destination
    if out_path.byte_length() == 0:
        out_path = String("/home/alex/mojodiffusion/output/mageflow_lora/") \
            + String("mageflow_base_lora.safetensors")
    var save_prefix = out_path
    if save_prefix.endswith(".safetensors"):
        save_prefix = String(save_prefix.removesuffix(".safetensors"))
    var out_dir = _dirname(out_path)
    var samples_dir = out_dir + "/samples"

    print("==== Mage-Flow-Base LoRA trainer — flow-match (shift", flow_shift, ") ====")
    print("arch: D=", DIM, " blocks=", DEPTH, " heads=", H, " head_dim=", Dh,
          " ffn=", FFN, " in/out=", IN_CH, "/", OUT_CH, " ctx=", TXT_CH)
    print("geometry: latent[1,128,", H_TOK, ",", W_TOK, "] -> N_IMG=", N_IMG,
          " N_TXT=", N_TXT, " S=", S)
    print("lora: rank=", rank, " alpha=", alpha,
          " (12 targets/block x", DEPTH, "blocks = 144 adapters)")
    print("optim: lr=", lr, " max_grad_norm=", max_grad_norm,
          " betas=(", beta1, ",", beta2, ") eps=", opt_eps, " wd=", weight_decay,
          " warmup=", warmup, "(constant after)")
    print("timestep: logit-normal + shift", flow_shift,
          " (schedule.mojo sample_timestep_logit_normal); model gets RAW sigma")
    print("steps:", steps, " seed:", seed,
          " save_every:", cfg.save_every, " sample_every:", cfg.sample_every)
    if resume_path.byte_length() > 0:
        print("resume:", resume_path, " start_step:", start_step,
              " (WARM — moments zeroed)" if warm_resume
              else " (FULL — A/B + AdamW moments)")
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

    # ── in-train sampling conds: encode BEFORE the base/loader load (staged) ──
    var sample_conds = List[List[Float32]]()
    if cfg.sample_every > 0:
        sample_conds = _encode_sample_prompts(ctx)
        _ = sys_system(String("mkdir -p '") + samples_dir + String("'"))
    _ = sys_system(String("mkdir -p '") + out_dir + String("'"))

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

    # ── path select: DEVICE arm default (resident-12 blocks + device
    # activations), HOST AdamW — this chain reproduces the host-arm anchor
    # loss stream BIT-EXACT (gated 2026-07-22, smoke12 losses+grad norms).
    # MAGEFLOW_HOST_PATH=1 = the original host parity fallback (streamed
    # blocks, host block API) — the anchor reference itself.
    # MAGEFLOW_FUSED_ADAMW=1 = fused device AdamW (opt 0.065->0.010 s/step);
    # deterministic but ±1-ulp FMA class vs the host optimizer
    # (lora_adamw_plain_fused.mojo contract) — losses drift ulp-level from
    # step 3 (measured max |dloss| ~2e-4 over 12 steps), so it is OPT-IN.
    var host_path = _env_str(String("MAGEFLOW_HOST_PATH")) == String("1")
    var use_fused = (not host_path) \
        and _env_str(String("MAGEFLOW_FUSED_ADAMW")) == String("1")
    if host_path:
        print("[path] HOST parity fallback (MAGEFLOW_HOST_PATH=1): streamed",
              "blocks + host block API + host AdamW")
    else:
        # all 12 blocks device-resident ONCE (byte-identical bytes; await_block
        # becomes a pure sub-buffer-view build — no per-step streaming).
        var pinned = loader.pin_residents(10 * 1024 * 1024 * 1024, ctx)
        if pinned != DEPTH:
            raise Error(
                String("[resident] pinned ") + String(pinned) + " of "
                + String(DEPTH) + " blocks — 16GB budget regression, refusing"
                + " partial residency (unset via MAGEFLOW_HOST_PATH=1)"
            )
        print("[path] DEVICE arm:", pinned, "blocks device-resident (~8.2GB),",
              "device activations,",
              "FUSED device AdamW (MAGEFLOW_FUSED_ADAMW=1, ulp-class)"
              if use_fused else "host AdamW (anchor-exact default)")

    # ── LoRA set: fresh init OR resume (BEFORE the device AdamW state init so
    #   the device M/V buffers seed from the LOADED host moments — the
    #   lora_adamw_plain_device_state_init pack path uploads ma/va/mb/vb). ───
    var lora: MageFlowLoraSet
    if resume_path.byte_length() > 0:
        if not _file_exists(resume_path):
            raise Error(
                String("[mageflow-resume] resume_state not found: ")
                + resume_path
            )
        var probe = mageflow_lora_prefixes(DEPTH)[0]
        # PEFT path given but `.state` sidecar present -> resolve to the
        # sidecar (the MJ-1077 trainer_resolve_resume_path house rule).
        var resolved = trainer_resolve_resume_path(resume_path, probe)
        if warm_resume:
            trainer_warn_warm_resume(String("mageflow"), resolved)
            lora = load_mageflow_lora_resume(DEPTH, rank, alpha, resolved, ctx)
        elif not lora_train_state_has_moments(resolved, probe):
            raise Error(
                String("[mageflow-resume] PEFT-as-resume: '") + resolved
                + "' carries no AdamW moments (bare PEFT, no `.state`"
                + " sidecar found). Pass the `.state` sidecar or use"
                + " warm_resume (A/B only, moments zeroed)."
            )
        else:
            print("[mageflow-resume] FULL resume (A/B + AdamW moments) from",
                  resolved)
            lora = load_mageflow_lora_state(DEPTH, rank, alpha, resolved, ctx)
            # warn-only step/seed guard (mageflow .state meta = [opt_step, seed];
            # sigma/noise/order are seed+step-derived, so these two knobs ARE
            # the whole resume scope).
            var rmeta = load_lora_train_state_meta(resolved, ctx)
            if len(rmeta) >= 2:
                if Int(rmeta[0]) != start_step:
                    print("  [mageflow-resume][WARN] .state saved at step=",
                          Int(rmeta[0]), " but resuming AT start_step=",
                          start_step, "— streams are step-derived; exact",
                          "continuation needs the saved step")
                if Int(rmeta[1]) != Int(Float32(Int(seed))):
                    print("  [mageflow-resume][WARN] seed changed: .state=",
                          Int(rmeta[1]), " live=", Int(seed),
                          "— the seed+step streams will DIVERGE")
    else:
        lora = build_mageflow_lora_set(DEPTH, DIM, FFN, rank, alpha)
    print("[lora] adapters:", mageflow_total_adapters(lora))
    # fused device AdamW state (persistent device P/M/V, ~300MB). Initialized
    # unconditionally (branch-free typing); only the fused arm steps it.
    # On resume this packs the LOADED host moments (not zeros) — device state
    # is seeded, not just host.
    var opt_state = mageflow_lora_adamw_state_init(lora, ctx)

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

    # ── step-0 sample: pre-training baseline (LoRA B=0 => base-model output).
    #   FRESH runs only — a resumed segment samples at cadence points inside
    #   the loop (step % sample_every), never unconditionally at start. ──────
    if cfg.sample_every > 0 and start_step == 0:
        _run_sample_point(0, sample_conds, base, loader, lora, spec,
                          flow_shift, seed, samples_dir, ctx, host_path)

    print("")
    print("step  sigma     loss           grad_norm     sec")
    # GLOBAL step numbering: [start_step, steps). opt_step = step+1 stays the
    # global optimizer step, so on resume the AdamW bias correction t, LR
    # warmup, save/sample cadence, artifact numbering, data round-robin
    # (step % n_cache) and the seed+step sigma/noise streams all CONTINUE the
    # uninterrupted run's sequence exactly.
    for step in range(start_step, steps):
        var t0 = perf_counter_ns()

        # sigma stream: seed + step (levers.mojo:104 stream convention).
        var sigma = sample_timestep_logit_normal_win(
            seed + UInt64(step), train_shift,
            cfg.min_noising_strength, cfg.max_noising_strength)

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
        var t_data = Float64(perf_counter_ns() - t0) / 1.0e9

        # fwd + loss + bwd — HOST parity arm or the DEVICE arm (same gated
        # block math; device tensors between blocks). Phase-timed.
        var step_loss: Float32
        var grads: MageFlowLoraGradSet
        var t_fwd: Float64
        var t_bwd: Float64
        if host_path:
            var tf0 = perf_counter_ns()
            var fwd = mageflow_stack_lora_forward_offload[H, Dh, N_IMG, N_TXT, S](
                img_tokens, txt_raw, silu_temb, base, loader, lora,
                rope[0], rope[1], DIM, FFN, IN_CH, TXT_CH, OUT_CH, EPS, ctx,
            )
            # loss + d_pred through the shared levers entry (MSE default).
            var lg = levers_loss_grad(fwd.out, target, sigma, cfg)
            step_loss = lg.loss
            t_fwd = Float64(perf_counter_ns() - tf0) / 1.0e9
            var tb0 = perf_counter_ns()
            grads = mageflow_stack_lora_backward_offload[H, Dh, N_IMG, N_TXT, S](
                lg.d_pred, silu_temb, base, loader, lora,
                rope[0], rope[1], fwd, DIM, FFN, OUT_CH, EPS, ctx,
            )
            t_bwd = Float64(perf_counter_ns() - tb0) / 1.0e9
        else:
            var tf0 = perf_counter_ns()
            var fwd = mageflow_stack_lora_forward_offload_device[
                H, Dh, N_IMG, N_TXT, S
            ](
                img_tokens, txt_raw, silu_temb, base, loader, lora,
                rope[0], rope[1], DIM, FFN, IN_CH, TXT_CH, OUT_CH, EPS, ctx,
            )
            var lg = levers_loss_grad(fwd.out, target, sigma, cfg)
            step_loss = lg.loss
            t_fwd = Float64(perf_counter_ns() - tf0) / 1.0e9
            var tb0 = perf_counter_ns()
            grads = mageflow_stack_lora_backward_offload_device[
                H, Dh, N_IMG, N_TXT, S
            ](
                lg.d_pred, silu_temb, base, loader, lora,
                rope[0], rope[1], fwd, DIM, FFN, OUT_CH, EPS, ctx,
            )
            t_bwd = Float64(perf_counter_ns() - tb0) / 1.0e9
        if not have_first:
            first_loss = step_loss
            have_first = True
        last_loss = step_loss

        var tc0 = perf_counter_ns()
        var gn = _clip(grads, max_grad_norm)
        var t_clip = Float64(perf_counter_ns() - tc0) / 1.0e9
        # constant LR + linear warmup over the first `warmup` optimizer steps.
        var opt_step = step + 1
        var step_lr = lr
        if warmup > 0 and opt_step < warmup:
            step_lr = lr * Float32(opt_step) / Float32(warmup)
        var to0 = perf_counter_ns()
        if use_fused:
            mageflow_lora_adamw_step_resident(
                opt_state, lora, grads, opt_step, step_lr, ctx,
                beta1, beta2, opt_eps, weight_decay,
            )
        else:
            mageflow_lora_adamw_step(lora, grads, opt_step, step_lr, ctx,
                                     beta1, beta2, opt_eps, weight_decay)
        var t_opt = Float64(perf_counter_ns() - to0) / 1.0e9
        var secs = Float64(perf_counter_ns() - t0) / 1.0e9
        print(step, "  ", sigma, "  ", step_loss, "  ", gn, "  ", secs,
              "  lr=", step_lr)
        print("  [phase] data=", t_data, " fwd=", t_fwd, " bwd=", t_bwd,
              " clip=", t_clip, " opt=", t_opt)
        if grads.nonfinite_lora_grads != 0:
            print("  !! nonfinite lora grads =", grads.nonfinite_lora_grads)

        # ── save cadence: step-numbered artifacts (ltx2/wan22 house pattern) ──
        if cfg.save_every > 0 and opt_step % cfg.save_every == 0:
            if use_fused:
                # params sync every step; moments only at save cadence.
                mageflow_lora_adamw_sync_moments(opt_state, lora, ctx)
            var sp = save_prefix + "_step" + String(opt_step)
            var np_ = save_mageflow_lora(lora, sp + ".safetensors", ctx)
            var smeta = List[Float32]()
            smeta.append(Float32(opt_step))
            smeta.append(Float32(Int(seed)))
            var ns_ = save_mageflow_lora_state(
                lora, sp + ".safetensors.state", ctx, smeta^
            )
            print("  [save] step", opt_step, ":", np_, "PEFT pairs ->",
                  sp + ".safetensors", " (+", ns_, "state tensors)")

        # ── sample cadence: 3 prompts @1024² with the CURRENT adapters ────────
        if cfg.sample_every > 0 and opt_step % cfg.sample_every == 0:
            _run_sample_point(opt_step, sample_conds, base, loader, lora, spec,
                              flow_shift, seed, samples_dir, ctx, host_path)

    if use_fused:
        mageflow_lora_adamw_sync_moments(opt_state, lora, ctx)
    var npairs = save_mageflow_lora(lora, out_path, ctx)
    var meta = List[Float32]()
    meta.append(Float32(steps))
    meta.append(Float32(Int(seed)))
    var state_path = out_path + String(".state")
    var nstate = save_mageflow_lora_state(lora, state_path, ctx, meta^)
    print("")
    print("[save] wrote", npairs, "PEFT pairs (diffusion_model.* keys) ->", out_path)
    print("[save] wrote", nstate, "state tensors ->", state_path)
    print("trained steps=", steps - start_step, " first loss=", first_loss,
          " last loss=", last_loss,
          " total sec=", Float64(perf_counter_ns() - train_start) / 1.0e9)
    print("RESULT: mageflow Base LoRA trainer ran (block math == chunk-1",
          "parity-gated engine; offload streaming, 16GB refit).")
