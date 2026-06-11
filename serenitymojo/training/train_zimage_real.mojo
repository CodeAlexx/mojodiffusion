# train_zimage_real.mojo — Z-Image (NextDiT) LoRA REAL training loop.
#
# Z-Image LoRA stack (models/zimage/zimage_stack_lora.mojo). Real base weights,
# local Mojo-prepared cache; no synthetic tensors and no Rust/Python cache
# dependency. Mirrors train_klein_real.mojo's loop structure (timing, grad clip,
# shared progress display, PEFT save, and optimizer-state sidecar).
#
# Per step (translated from train_zimage.rs main loop):
#   1. load cached {latent [1,16,72,56], text_embedding [1,512,2560], text_mask}
#   2. latent <- (latent - VAE_SHIFT) * VAE_SCALE         (train_zimage.rs:1051)
#   3. x_seq  = x_embedder(patchify(latent))              (post-embedder tokens)
#      cap_seq= cap_embedder(text_embedding)
#   4. sigma  = logit_normal(shift=1.0) ; sigma_idx = floor(sigma*1000) clamp
#      t_value= (1000 - sigma_idx)/1000                   (train_zimage.rs:1125)
#   5. adaln  = t_embedder(t_value); per-block RAW modvecs + f_scale
#   6. flow-match in LATENT space:
#        noisy_latent = sigma*noise + (1-sigma)*latent
#        target       = patchify(noise - latent)            (v-prediction)
#   7. x_embedder(noisy_latent) -> zimage_stack_lora_forward -> velocity [N_IMG, OUT_CH]
#   8. loss = MSE(-raw_velocity, target_img); d_raw = -(2/N)(-raw_velocity - target_img)
#      (the stack outputs ONLY the N_IMG image rows, so the flow-match target is
#       taken on the IMAGE-token sub-sequence — see _img_target.)
#   9. zimage_stack_lora_backward -> LoRA grads; grad_norm = L2; clip(1.0)
#  10. zimage_lora_adamw_step_main_only; print shared progress display
#
# Recipe scalars (train_zimage.rs released-preset defaults):
#   lr=3e-4, rank=16, alpha=1.0, timestep_shift=1.0, clip_grad_norm=1.0,
#   VAE_SHIFT=0.1159, VAE_SCALE=0.3611, NUM_TRAIN_TIMESTEPS=1000.
#
# HARD DTYPE RULE (2026-06-02): Z-Image training is BF16/BP16 for base model
# weights. OneTrainer does not train a full-F32 Z-Image model, and neither
# should this trainer. A full-F32 base/model load will OOM on 24 GB cards.
#
# The current stack still carries activations, scalar reductions, LoRA masters,
# and a few small norm compatibility tensors as F32. That is not a full-F32
# model. Large block projection and MLP weights must stay in checkpoint dtype
# via load_zimage_block_weights_prefixed_mixed until a full mixed/offload stack
# lands.
#
# MEMORY (measured budget): full-depth resident all-F32 base = 24.6 GB > 24 GB.
# Full-depth LoRA training must preserve BF16/BP16 base projections and avoid
# materializing frozen base d_W. If this path OOMs, add block offload; do not
# fall back to all-F32 or reduced-depth and call it a training baseline.
#
# SELF-CONTAINED TRAINER RULE (2026-06-02): Z-Image prepare/train/sample runtime
# must be Mojo-owned. OneTrainer is the read-only source of truth for formulas and
# baselines; Rust/Python may be used only for offline parity evidence, not as the
# training cache producer or runtime dependency.
#
# Run (real 512-bucket LoRA training):
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#     pixi run mojo run -I . serenitymojo/training/train_zimage_real.mojo [steps] [start_step] [state.safetensors]

from std.sys import argv
from std.collections import List, Optional
from std.gpu.host import DeviceContext
from std.math import sqrt, log as flog, cos as fcos, sin as fsin
from std.memory import alloc
from std.time import perf_counter_ns

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.ffi import (
    sys_system, sys_open, sys_close, sys_pwrite, BytePtr,
    O_WRONLY, O_CREAT, O_TRUNC,
)

from serenitymojo.models.zimage.weights import (
    ZImageBlockWeights, load_zimage_block_weights_prefixed_mixed,
)
from serenitymojo.models.zimage.block import ZImageModVecs
from serenitymojo.models.zimage.lora_block import ZIMAGE_SLOTS
from serenitymojo.models.zimage.zimage_stack import ZImageStackForward
from serenitymojo.models.zimage.zimage_stack_lora import (
    ZImageLoraSet, ZImageLoraGrads, ZImageLoraDeviceSet, build_zimage_lora_set,
    zimage_lora_set_to_device,
    zimage_stack_lora_forward_main_device, zimage_stack_lora_backward_main_device,
    zimage_stack_lora_forward_main_device_v2, zimage_stack_lora_backward_main_device_v2,
    zimage_stack_lora_forward_main_device_b2, zimage_stack_lora_backward_main_device_b2,
    zimage_lora_adamw_step_main_only, save_zimage_lora_main_only,
    save_zimage_lora_main_only_state, load_zimage_lora_main_only_state,
    zimage_lora_set_to_device_resident,
)
from serenitymojo.training.lora_adamw_plain_fused import (
    LoraAdamWPlainDeviceState, lora_adamw_plain_device_state_init,
    fused_lora_adamw_plain_step_resident,
    lora_adamw_plain_device_state_sync_moments,
)
from serenitymojo.models.zimage.lora_block import (
    ZImageModVecsDevice, zimage_modvecs_pack2_to_device,
    ZImageModVecsAllDevice, zimage_modvecs_all_to_device,
)
from serenitymojo.models.zimage.real_weights import (
    ZImageRealAux, load_zimage_real_aux, build_adaln, build_block_modvecs,
    build_f_scale, build_cap_seq, build_x_seq, build_rope, build_positions,
)
from serenitymojo.io.train_config_reader import read_model_config
from serenitymojo.training.schedule import sample_timestep_logit_normal
from serenitymojo.training.klein_dataset import KleinCache
from serenitymojo.training.progress_display import print_trainer_progress
from serenitymojo.training.sample_prompt_config import (
    SampleCadence, read_sample_cadence_config,
    validate_step_sample_cadence, should_sample_completed_step,
    next_sample_completed_step, sample_time_unit_name,
    SAMPLE_UNIT_STEP, SAMPLE_UNIT_NEVER,
)
from serenitymojo.training.onetrainer_train_loop_policy import (
    OT_GRAD_POLICY_ON_ONLY,
    ot_cache_dir_from_train_config,
    ot_lr_for_optimizer_step,
    ot_output_lora_path_from_train_config,
    ot_sample_cadence_from_train_config,
    ot_sampling_enabled,
    ot_should_save_before_sample,
    ot_should_save_checkpoint,
    ot_step_lora_path,
    validate_ot_gradient_checkpointing_policy,
    validate_ot_train_math_policy,
)
from serenitymojo.training.train_config import (
    TrainConfig, GRADIENT_CHECKPOINTING_ON,
)
from serenitymojo.training.onetrainer_cache_preflight import (
    create_onetrainer_cache_preflight_plan,
    validate_onetrainer_cache_preflight_plan,
)


# ── arch (Z-Image, from transformer config; H/Dh/D fixed comptime) ───────────
comptime H = 30
comptime Dh = 128
comptime D = H * Dh          # 3840
comptime F = 10240           # SwiGLU per-gate hidden
comptime CAP_DIM = 2560      # Qwen3 hidden
comptime ADALN_DIM = 256
comptime T_SCALE = Float32(1000.0)
comptime ROPE_THETA = Float32(256.0)
comptime AXIS0 = 32
comptime AXIS1 = 48
comptime AXIS2 = 48
comptime EPS = Float32(1e-5)
comptime FINAL_EPS = Float32(1e-6)
comptime OUT_CH = 64         # patchified output channels (16ch * 2 * 2)
comptime PATCH = 2

# ── resolution: OneTrainer Alina "512" bucket is 576x448 image -> latent
# [16,72,56] -> patch2 -> 36x28=1008 real image tokens. Diffusers pads image
# tokens to a multiple of 32, so the transformer sees 1024 image rows and loss
# is applied only to the first 1008 rows.
# v2 ENGINE SWAP (maintainer mandate 2026-06-11, HANDOFF_2026-06-11_OVERNIGHT
# _OT_PARITY.md): route the B=1 step through the gated batch engine —
# device-resident mod-vecs (ONE packed upload/step) + frozen-skip batch
# backward. False = the previous per-block-upload path, byte-identical to the
# 06-10 anchors (gate-don't-delete, flame Stage-6a pattern).
comptime ZIMAGE_V2_ENGINE = True

comptime LAT_C = 16
comptime LAT_H = 72
comptime LAT_W = 56
comptime HT = LAT_H // PATCH  # 36
comptime WT = LAT_W // PATCH  # 28
comptime N_IMG_REAL = HT * WT # 1008
comptime IMG_PAD = (32 - (N_IMG_REAL % 32)) % 32
comptime N_IMG = N_IMG_REAL + IMG_PAD # 1024
comptime CAP_LEN = 224        # first-sample OneTrainer cap seq after mask prune + pad32
comptime N_TXT = CAP_LEN
comptime S = N_IMG + N_TXT    # 1248

# ── full Z-Image depth. Reduced-depth runs are smoke-only and not a baseline. ─
comptime NUM_NR = 2
comptime NUM_CR = 2
comptime MAIN_DEPTH = 30
comptime OVERFIT_PROBE = False

# ── recipe (train_zimage.rs released-preset) ─────────────────────────────────
comptime RANK = 16
comptime ALPHA = Float32(1.0)
comptime LR = Float32(3.0e-4)
comptime TIMESTEP_SHIFT = Float32(1.0)
comptime VAE_SHIFT = Float32(0.1159)
comptime VAE_SCALE = Float32(0.3611)
comptime NUM_TRAIN_TIMESTEPS = 1000
comptime CLIP_GRAD_NORM = Float32(1.0)
comptime SEED_BASE = UInt64(42)

comptime TRANSFORMER_DIR = "/home/alex/.serenity/models/zimage_base/transformer"
comptime CACHE_DIR = "/home/alex/mojodiffusion/output/alina_zimage_cache"
comptime LORA_DIR = "/home/alex/mojodiffusion/output/alina_zimage"
comptime DEFAULT_CONFIG = "/home/alex/mojodiffusion/serenitymojo/configs/zimage.json"
comptime DEFAULT_RUN_STEPS = 5
comptime TRAIN_ADAPTER_START = (NUM_NR + NUM_CR) * ZIMAGE_SLOTS
comptime TRAIN_ADAPTER_COUNT = MAIN_DEPTH * ZIMAGE_SLOTS
comptime ZIMAGE_GENERATE_SOURCE = "serenitymojo/pipeline/zimage_generate.mojo"
comptime ZIMAGE_GENERATE_BINARY = "/tmp/zimage_generate_prod"


def _is_nonnegative_int(s: String) -> Bool:
    if s.byte_length() == 0:
        return False
    var bs = s.as_bytes()
    for i in range(s.byte_length()):
        if bs[i] < 0x30 or bs[i] > 0x39:
            return False
    return True


def _parse_nonnegative_int(s: String) raises -> Int:
    if not _is_nonnegative_int(s):
        raise Error(String("expected non-negative integer, got ") + s)
    var out = 0
    var bs = s.as_bytes()
    for i in range(s.byte_length()):
        out = out * 10 + Int(bs[i] - 0x30)
    return out


def _close_f32(a: Float32, b: Float32, tol: Float32 = Float32(1.0e-7)) -> Bool:
    var d = a - b
    if d < Float32(0.0):
        d = -d
    return d <= tol


def zimage_patchified_out_channels(cfg: TrainConfig) -> Int:
    return cfg.out_channels * PATCH * PATCH


def validate_zimage_train_config(cfg: TrainConfig) raises:
    if cfg.checkpoint == String(""):
        raise Error("Z-Image trainer config must set checkpoint transformer dir")
    if cfg.n_heads != H:
        raise Error(String("Z-Image config n_heads ") + String(cfg.n_heads) + String(" != H ") + String(H))
    if cfg.head_dim != Dh:
        raise Error(String("Z-Image config head_dim ") + String(cfg.head_dim) + String(" != Dh ") + String(Dh))
    if cfg.d_model != D:
        raise Error(String("Z-Image config d_model ") + String(cfg.d_model) + String(" != D ") + String(D))
    if cfg.in_channels != LAT_C:
        raise Error(String("Z-Image config in_channels ") + String(cfg.in_channels) + String(" != LAT_C ") + String(LAT_C))
    if cfg.joint_attention_dim != CAP_DIM:
        raise Error(String("Z-Image config joint_attention_dim ") + String(cfg.joint_attention_dim) + String(" != CAP_DIM ") + String(CAP_DIM))
    if zimage_patchified_out_channels(cfg) != OUT_CH:
        raise Error(
            String("Z-Image config out_channels ") + String(cfg.out_channels)
            + String(" with patch_size=2 gives ")
            + String(zimage_patchified_out_channels(cfg))
            + String(" patchified channels, expected ") + String(OUT_CH)
        )
    if cfg.num_double != 0 or cfg.num_single != MAIN_DEPTH:
        raise Error(
            String("Z-Image trainer requires 0 double blocks and ")
            + String(MAIN_DEPTH)
            + String(" main layers; got double=")
            + String(cfg.num_double)
            + String(" single=")
            + String(cfg.num_single)
        )
    if cfg.mlp_hidden != F:
        raise Error(String("Z-Image config mlp_hidden ") + String(cfg.mlp_hidden) + String(" != F ") + String(F))
    if not _close_f32(Float32(cfg.rope_theta), ROPE_THETA):
        raise Error(String("Z-Image config rope_theta ") + String(cfg.rope_theta) + String(" != ") + String(ROPE_THETA))
    if cfg.lora_rank != RANK:
        raise Error(
            String("Z-Image trainer is compiled for lora_rank=")
            + String(RANK)
            + String("; parsed ")
            + String(cfg.lora_rank)
        )
    if not _close_f32(cfg.lora_alpha, ALPHA):
        raise Error("Z-Image trainer lora_alpha does not match compiled constant")
    if not _close_f32(cfg.lr, LR, Float32(1.0e-9)):
        raise Error("Z-Image trainer learning_rate does not match compiled constant")
    if not _close_f32(cfg.timestep_shift, TIMESTEP_SHIFT):
        raise Error("Z-Image trainer timestep_shift does not match compiled constant")
    if not _close_f32(cfg.max_grad_norm, CLIP_GRAD_NORM):
        raise Error("Z-Image trainer max_grad_norm does not match compiled constant")
    validate_ot_train_math_policy(cfg, String("Z-Image trainer"))
    validate_ot_gradient_checkpointing_policy(
        cfg, String("Z-Image trainer"), OT_GRAD_POLICY_ON_ONLY
    )


def zimage_cache_dir_from_train_config(cfg: TrainConfig) -> String:
    return ot_cache_dir_from_train_config(cfg, String(CACHE_DIR))


def zimage_transformer_dir_from_train_config(cfg: TrainConfig) -> String:
    if cfg.checkpoint != String(""):
        return cfg.checkpoint.copy()
    return String(TRANSFORMER_DIR)


def zimage_output_lora_path_from_train_config(cfg: TrainConfig, completed_step: Int) -> String:
    return ot_output_lora_path_from_train_config(
        cfg, String(LORA_DIR), String("zimage_lora"), completed_step
    )


def zimage_sample_cadence_from_train_config(
    cfg_path: String, cfg: TrainConfig,
) raises -> SampleCadence:
    return ot_sample_cadence_from_train_config(cfg_path, cfg)


def zimage_sampling_enabled(cadence: SampleCadence) -> Bool:
    return ot_sampling_enabled(cadence)


def zimage_should_save_checkpoint(cfg: TrainConfig, completed_step: Int) -> Bool:
    return ot_should_save_checkpoint(cfg, completed_step)


def zimage_should_save_before_sample(
    cadence: SampleCadence, completed_step: Int, saved_this_step: Bool,
) raises -> Bool:
    return ot_should_save_before_sample(cadence, completed_step, saved_this_step)


def _step_lora_path(base_path: String, step: Int) -> String:
    return ot_step_lora_path(base_path, step)


def zimage_sample_request_dir() -> String:
    return String(LORA_DIR) + String("/sample_requests")


def zimage_sample_request_path(completed_step: Int) -> String:
    return (
        zimage_sample_request_dir()
        + String("/step")
        + String(completed_step)
        + String("_request.json")
    )


def zimage_sample_output_path(completed_step: Int) -> String:
    return (
        zimage_sample_request_dir()
        + String("/step")
        + String(completed_step)
        + String("_sample.png")
    )


def zimage_sample_result_path(completed_step: Int) -> String:
    return (
        zimage_sample_request_dir()
        + String("/step")
        + String(completed_step)
        + String("_sample_result.json")
    )


def _write_text_file(path: String, content: String) raises:
    var fd = sys_open(path, O_CREAT | O_WRONLY | O_TRUNC, Int32(0o644))
    if fd < 0:
        raise Error(String("train_zimage_real: cannot create ") + path)
    var n = content.byte_length()
    var buf = alloc[UInt8](n if n > 0 else 1)
    var src = content.as_bytes()
    for i in range(n):
        buf[i] = src[i]
    var wrote = sys_pwrite(fd, BytePtr(unsafe_from_address=Int(buf)), n, 0)
    buf.free()
    _ = sys_close(fd)
    if wrote != n:
        raise Error(String("train_zimage_real: short write to ") + path)


def _write_zimage_sample_request(
    completed_step: Int,
    lora_path: String,
    state_path: String,
    sample_file: String,
) raises -> String:
    """Queue validation sampling for a later standalone process.

    Z-Image sampling loads Qwen3, the full transformer, and the VAE. Running it
    inside this train process would co-reside those allocations with the trainer
    and is not a safe 24GB product path.
    """
    var out_png = zimage_sample_output_path(completed_step)
    var result_manifest = zimage_sample_result_path(completed_step)
    var request_path = zimage_sample_request_path(completed_step)
    var build_command = (
        String("pixi run mojo build -I . -Xlinker -lm ")
        + String(ZIMAGE_GENERATE_SOURCE)
        + String(" -o ")
        + String(ZIMAGE_GENERATE_BINARY)
    )
    var run_command = (
        String(ZIMAGE_GENERATE_BINARY)
        + String(" --request ")
        + request_path
    )
    var content = String("{\n")
    content += String('  "schema":"serenity.zimage.sample_request.v1",\n')
    content += String('  "model":"zimage",\n')
    content += String('  "sampler_mode":"split_process_after_train_memory_release",\n')
    content += String('  "completed_step":') + String(completed_step) + String(",\n")
    content += String('  "lora_path":"') + lora_path + String('",\n')
    content += String('  "state_path":"') + state_path + String('",\n')
    content += String('  "sample_file":"') + sample_file + String('",\n')
    content += String('  "output_png":"') + out_png + String('",\n')
    content += String('  "result_manifest":"') + result_manifest + String('",\n')
    content += String('  "sampler_source":"') + String(ZIMAGE_GENERATE_SOURCE) + String('",\n')
    content += String('  "build_command":"') + build_command + String('",\n')
    content += String('  "run_command":"') + run_command + String('",\n')
    content += String('  "accepted_parity":false,\n')
    content += String('  "note":"request only; run standalone sampler after trainer exits or memory is released"\n')
    content += String("}\n")
    _ = sys_system(String("mkdir -p ") + zimage_sample_request_dir())
    _write_text_file(request_path, content)
    return request_path^


# ── deterministic host gaussian noise (Box-Muller PCG; per-step seed) ─────────
def _host_noise(n: Int, seed: UInt64) -> List[Float32]:
    var out = List[Float32]()
    var state = seed
    var i = 0
    while i < n:
        state = state * 6364136223846793005 + 1442695040888963407
        var u1f = Float64(Int(state >> 11)) * (1.0 / 9007199254740992.0)
        state = state * 6364136223846793005 + 1442695040888963407
        var u2f = Float64(Int(state >> 11)) * (1.0 / 9007199254740992.0)
        if u1f < 1.0e-12:
            u1f = 1.0e-12
        var r = sqrt(-2.0 * flog(Float64(u1f)))
        var theta = 6.283185307179586 * u2f
        out.append(Float32(r * fcos(Float64(theta))))
        if i + 1 < n:
            out.append(Float32(r * fsin(Float64(theta))))
        i += 2
    return out^


def _l2(h: List[Float32]) -> Float64:
    var s = 0.0
    for i in range(len(h)):
        var v = Float64(h[i])
        s += v * v
    return sqrt(s)


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


@fieldwise_init
struct FlatStats(Copyable, Movable):
    var mean: Float64
    var std: Float64
    var max_abs: Float32


def _flat_stats(v: List[Float32]) -> FlatStats:
    if len(v) == 0:
        return FlatStats(0.0, 0.0, Float32(0.0))
    var sum = 0.0
    var max_abs = Float32(0.0)
    for i in range(len(v)):
        var x = v[i]
        sum += Float64(x)
        var ax = x if x >= 0.0 else -x
        if ax > max_abs:
            max_abs = ax
    var mean = sum / Float64(len(v))
    var ss = 0.0
    for i in range(len(v)):
        var d = Float64(v[i]) - mean
        ss += d * d
    return FlatStats(mean, sqrt(ss / Float64(len(v))), max_abs)


def _global_norm(grads: ZImageLoraGrads, start: Int, end: Int) -> Float64:
    var ss = 0.0
    for i in range(start, end):
        for j in range(len(grads.d_a[i])):
            ss += Float64(grads.d_a[i][j]) * Float64(grads.d_a[i][j])
        for j in range(len(grads.d_b[i])):
            ss += Float64(grads.d_b[i][j]) * Float64(grads.d_b[i][j])
    return sqrt(ss)


def _clip(mut grads: ZImageLoraGrads, max_norm: Float32, start: Int, end: Int) -> Float64:
    var gn = _global_norm(grads, start, end)
    if gn <= Float64(max_norm) or gn == 0.0:
        return gn
    var s = Float32(Float64(max_norm) / gn)
    for i in range(start, end):
        for j in range(len(grads.d_a[i])):
            grads.d_a[i][j] = grads.d_a[i][j] * s
        for j in range(len(grads.d_b[i])):
            grads.d_b[i][j] = grads.d_b[i][j] * s
    return gn


@fieldwise_init
struct StepResult(Copyable, Movable):
    var loss: Float32
    var grad: Float32
    var secs: Float32
    var lora_b_sum: Float32
    var lora_b_nonzero: Int
    var nonfinite: Int


def _valid_cap_from_mask(mask: Tensor, ctx: DeviceContext) raises -> Int:
    if mask.dtype() == STDtype.BF16:
        var mask_bf = mask.to_host_bf16(ctx)
        var valid_cap_bf = 0
        for i in range(len(mask_bf)):
            if mask_bf[i].cast[DType.float32]() > 0.5:
                valid_cap_bf += 1
        return valid_cap_bf
    if mask.dtype() == STDtype.F16:
        var mask_f16 = mask.to_host_f16(ctx)
        var valid_cap_f16 = 0
        for i in range(len(mask_f16)):
            if mask_f16[i].cast[DType.float32]() > 0.5:
                valid_cap_f16 += 1
        return valid_cap_f16

    # F32 masks are already F32 at the cache boundary.
    var mask_h = mask.to_host(ctx)
    var valid_cap = 0
    for i in range(len(mask_h)):
        if mask_h[i] > 0.5:
            valid_cap += 1
    return valid_cap


@fieldwise_init
struct ZImageLatentStepInputs(Movable):
    var noisy_latent: Tensor
    var target_patch: List[Float32]


def _build_latent_step_inputs_bf16[
    LAT_H_B: Int, LAT_W_B: Int
](
    latent: Tensor, noise_lat: List[Float32], sig: Float32, ctx: DeviceContext
) raises -> ZImageLatentStepInputs:
    var lat_bf = latent.to_host_bf16(ctx)
    var noisy = List[Float32]()
    for i in range(len(lat_bf)):
        var lat = (lat_bf[i].cast[DType.float32]() - VAE_SHIFT) * VAE_SCALE
        noisy.append(noise_lat[i] * sig + lat * (Float32(1.0) - sig))
    var noisy_latent = Tensor.from_host(
        noisy^, [1, LAT_C, LAT_H_B, LAT_W_B], latent.dtype(), ctx,
    )
    var target = _patchify_target_bf16[LAT_H_B, LAT_W_B](noise_lat, lat_bf)
    return ZImageLatentStepInputs(noisy_latent^, target^)


def _build_latent_step_inputs_f16[
    LAT_H_B: Int, LAT_W_B: Int
](
    latent: Tensor, noise_lat: List[Float32], sig: Float32, ctx: DeviceContext
) raises -> ZImageLatentStepInputs:
    var lat_f16 = latent.to_host_f16(ctx)
    var noisy = List[Float32]()
    for i in range(len(lat_f16)):
        var lat = (lat_f16[i].cast[DType.float32]() - VAE_SHIFT) * VAE_SCALE
        noisy.append(noise_lat[i] * sig + lat * (Float32(1.0) - sig))
    var noisy_latent = Tensor.from_host(
        noisy^, [1, LAT_C, LAT_H_B, LAT_W_B], latent.dtype(), ctx,
    )
    var target = _patchify_target_f16[LAT_H_B, LAT_W_B](noise_lat, lat_f16)
    return ZImageLatentStepInputs(noisy_latent^, target^)


def _build_latent_step_inputs_f32[
    LAT_H_B: Int, LAT_W_B: Int
](
    latent: Tensor, noise_lat: List[Float32], sig: Float32, ctx: DeviceContext
) raises -> ZImageLatentStepInputs:
    var lat_f32 = latent.to_host(ctx)
    var noisy = List[Float32]()
    for i in range(len(lat_f32)):
        lat_f32[i] = (lat_f32[i] - VAE_SHIFT) * VAE_SCALE
        noisy.append(noise_lat[i] * sig + lat_f32[i] * (Float32(1.0) - sig))
    var noisy_latent = Tensor.from_host(
        noisy^, [1, LAT_C, LAT_H_B, LAT_W_B], latent.dtype(), ctx,
    )
    var target = _patchify_target_f32[LAT_H_B, LAT_W_B](noise_lat, lat_f32)
    return ZImageLatentStepInputs(noisy_latent^, target^)


def _build_latent_step_inputs[
    LAT_H_B: Int, LAT_W_B: Int
](
    latent: Tensor, noise_lat: List[Float32], sig: Float32, ctx: DeviceContext
) raises -> ZImageLatentStepInputs:
    if latent.dtype() == STDtype.BF16:
        return _build_latent_step_inputs_bf16[LAT_H_B, LAT_W_B](latent, noise_lat, sig, ctx)
    if latent.dtype() == STDtype.F16:
        return _build_latent_step_inputs_f16[LAT_H_B, LAT_W_B](latent, noise_lat, sig, ctx)
    # F32 cache tensors are already F32 at the storage boundary.
    return _build_latent_step_inputs_f32[LAT_H_B, LAT_W_B](latent, noise_lat, sig, ctx)


def _cap_tensor_from_cache_bf16[
    CAP_LEN_B: Int
](text_embedding: Tensor, valid_cap: Int, ctx: DeviceContext) raises -> Tensor:
    var cap_bf = text_embedding.to_host_bf16(ctx)
    var cap_vals = List[Float32]()
    for r in range(CAP_LEN_B):
        var src_r = r if r < valid_cap else valid_cap - 1
        for c in range(CAP_DIM):
            cap_vals.append(cap_bf[src_r * CAP_DIM + c].cast[DType.float32]())
    return Tensor.from_host(
        cap_vals^, [CAP_LEN_B, CAP_DIM], text_embedding.dtype(), ctx,
    )


def _cap_tensor_from_cache_f16[
    CAP_LEN_B: Int
](text_embedding: Tensor, valid_cap: Int, ctx: DeviceContext) raises -> Tensor:
    var cap_f16 = text_embedding.to_host_f16(ctx)
    var cap_vals = List[Float32]()
    for r in range(CAP_LEN_B):
        var src_r = r if r < valid_cap else valid_cap - 1
        for c in range(CAP_DIM):
            cap_vals.append(cap_f16[src_r * CAP_DIM + c].cast[DType.float32]())
    return Tensor.from_host(
        cap_vals^, [CAP_LEN_B, CAP_DIM], text_embedding.dtype(), ctx,
    )


def _cap_tensor_from_cache_f32[
    CAP_LEN_B: Int
](text_embedding: Tensor, valid_cap: Int, ctx: DeviceContext) raises -> Tensor:
    var cap_f32 = text_embedding.to_host(ctx)
    var cap_vals = List[Float32]()
    for r in range(CAP_LEN_B):
        var src_r = r if r < valid_cap else valid_cap - 1
        for c in range(CAP_DIM):
            cap_vals.append(cap_f32[src_r * CAP_DIM + c])
    return Tensor.from_host(
        cap_vals^, [CAP_LEN_B, CAP_DIM], text_embedding.dtype(), ctx,
    )


def _cap_tensor_from_cache[
    CAP_LEN_B: Int
](text_embedding: Tensor, valid_cap: Int, ctx: DeviceContext) raises -> Tensor:
    if text_embedding.dtype() == STDtype.BF16:
        return _cap_tensor_from_cache_bf16[CAP_LEN_B](text_embedding, valid_cap, ctx)
    if text_embedding.dtype() == STDtype.F16:
        return _cap_tensor_from_cache_f16[CAP_LEN_B](text_embedding, valid_cap, ctx)
    # F32 cache tensors are already F32 at the storage boundary.
    return _cap_tensor_from_cache_f32[CAP_LEN_B](text_embedding, valid_cap, ctx)


def _cache_valid_cap(cache: KleinCache, slot: Int, ctx: DeviceContext) raises -> Int:
    var s = cache.load(slot, ctx)
    return _valid_cap_from_mask(s.text_mask, ctx)


def _train_one_step_bucket[
    LAT_H_B: Int, LAT_W_B: Int, CAP_LEN_B: Int
](
    k: Int,
    run_steps: Int,
    slot: Int,
    step_seed: UInt64,
    cache: KleinCache,
    aux: ZImageRealAux,
    nr_blocks: List[ZImageBlockWeights],
    cr_blocks: List[ZImageBlockWeights],
    main_blocks: List[ZImageBlockWeights],
    mut lora: ZImageLoraSet,
    mut opt_state: LoraAdamWPlainDeviceState,
    resident_dev: ZImageLoraDeviceSet,
    n_adapters: Int,
    final_lin_w: Tensor,
    final_lin_b: Tensor,
    x_pad_h: List[Float32],
    cap_pad_h: List[Float32],
    train_cfg: TrainConfig,
    train_start_ns: UInt,
    ctx: DeviceContext,
) raises -> StepResult:
    comptime HT_B = LAT_H_B // PATCH
    comptime WT_B = LAT_W_B // PATCH
    comptime N_IMG_REAL_B = HT_B * WT_B
    comptime IMG_PAD_B = (32 - (N_IMG_REAL_B % 32)) % 32
    comptime N_IMG_B = N_IMG_REAL_B + IMG_PAD_B
    comptime N_TXT_B = CAP_LEN_B
    comptime S_B = N_IMG_B + N_TXT_B

    var t0 = perf_counter_ns()

    var s = cache.load(slot, ctx)
    var lsh = s.latent.shape()
    if lsh[1] != LAT_C or lsh[2] != LAT_H_B or lsh[3] != LAT_W_B:
        raise Error("train_zimage_real: dispatched sample to wrong latent bucket")

    var valid_cap = _valid_cap_from_mask(s.text_mask, ctx)
    if valid_cap <= 0 or valid_cap > CAP_LEN_B:
        raise Error("train_zimage_real: dispatched sample to wrong text bucket")

    var sigma = sample_timestep_logit_normal(SEED_BASE + step_seed, TIMESTEP_SHIFT)
    var sigma_idx = Int(sigma * Float32(NUM_TRAIN_TIMESTEPS))
    if sigma_idx > NUM_TRAIN_TIMESTEPS - 1:
        sigma_idx = NUM_TRAIN_TIMESTEPS - 1
    var sig = Float32(sigma_idx + 1) / Float32(NUM_TRAIN_TIMESTEPS)
    var t_value = Float32(NUM_TRAIN_TIMESTEPS - sigma_idx) / Float32(NUM_TRAIN_TIMESTEPS)

    var noise_lat = _host_noise(LAT_C * LAT_H_B * LAT_W_B, SEED_BASE * UInt64(7919) + step_seed)
    var latent_inputs = _build_latent_step_inputs[LAT_H_B, LAT_W_B](
        s.latent, noise_lat, sig, ctx,
    )

    var x_t = build_x_seq(aux, latent_inputs.noisy_latent, LAT_C, LAT_H_B, LAT_W_B, PATCH, ctx)
    for _pad in range(IMG_PAD_B):
        for c in range(D):
            x_t.append(x_pad_h[c])

    var cap2 = _cap_tensor_from_cache[CAP_LEN_B](s.text_embedding, valid_cap, ctx)
    var cap_seq = build_cap_seq(aux, cap2, EPS, ctx)
    for r in range(valid_cap, CAP_LEN_B):
        for c in range(D):
            cap_seq[r * D + c] = cap_pad_h[c]

    var pos_step = build_positions(N_IMG_B, HT_B, WT_B, CAP_LEN_B, valid_cap)
    var x_pos = pos_step[0].copy()
    var cap_pos = pos_step[1].copy()
    var uni_pos = List[List[Int]]()
    for i in range(len(x_pos)):
        uni_pos.append(x_pos[i].copy())
    for i in range(len(cap_pos)):
        uni_pos.append(cap_pos[i].copy())
    var xr = build_rope(x_pos, H, Dh, ROPE_THETA, AXIS0, AXIS1, AXIS2, ctx)
    var x_cos = xr[0].copy(); var x_sin = xr[1].copy()
    var cr = build_rope(cap_pos, H, Dh, ROPE_THETA, AXIS0, AXIS1, AXIS2, ctx)
    var cap_cos = cr[0].copy(); var cap_sin = cr[1].copy()
    var ur = build_rope(uni_pos, H, Dh, ROPE_THETA, AXIS0, AXIS1, AXIS2, ctx)
    var uni_cos = ur[0].copy(); var uni_sin = ur[1].copy()

    var adaln = build_adaln(aux, t_value, ADALN_DIM, T_SCALE, ctx)
    var nr_mod = List[ZImageModVecs]()
    for i in range(NUM_NR):
        nr_mod.append(build_block_modvecs(aux.nr_mod_w[i][], aux.nr_mod_b[i][], adaln, D, ctx))
    var main_mod = List[ZImageModVecs]()
    for i in range(MAIN_DEPTH):
        main_mod.append(build_block_modvecs(aux.main_mod_w[i][], aux.main_mod_b[i][], adaln, D, ctx))
    var f_scale = build_f_scale(aux, adaln, D, ctx)
    # v2 engine: all main-block mod-vecs land on device in ONE packed upload
    # (the old path re-uploaded each vec per block per pass, each with a sync).
    var mvall = Optional[ZImageModVecsAllDevice](None)
    comptime if ZIMAGE_V2_ENGINE:
        mvall = Optional[ZImageModVecsAllDevice](
            zimage_modvecs_all_to_device(main_mod, D, ctx)
        )
    var t_prep = perf_counter_ns()

    # v2 engine: resident device LoRA set (views into the persistent optimizer
    # param buffer) — no per-step upload. Old path rebuilds the set each step.
    var lora_dev: ZImageLoraDeviceSet
    comptime if ZIMAGE_V2_ENGINE:
        lora_dev = resident_dev.copy()
    else:
        lora_dev = zimage_lora_set_to_device(lora, ctx)
    var t_lora = perf_counter_ns()

    var fwd: ZImageStackForward
    comptime if ZIMAGE_V2_ENGINE:
        fwd = zimage_stack_lora_forward_main_device_v2[H, Dh, N_IMG_B, N_TXT_B, S_B](
            x_t.copy(), cap_seq.copy(),
            nr_blocks, nr_mod, cr_blocks, main_blocks,
            mvall.value().per_block, lora_dev,
            f_scale.copy(), final_lin_w, final_lin_b,
            x_cos[], x_sin[], cap_cos[], cap_sin[], uni_cos[], uni_sin[],
            D, F, OUT_CH, EPS, FINAL_EPS, ctx,
        )
    else:
        fwd = zimage_stack_lora_forward_main_device[H, Dh, N_IMG_B, N_TXT_B, S_B](
            x_t.copy(), cap_seq.copy(),
            nr_blocks, nr_mod, cr_blocks, main_blocks, main_mod, lora_dev,
            f_scale.copy(), final_lin_w, final_lin_b,
            x_cos[], x_sin[], cap_cos[], cap_sin[], uni_cos[], uni_sin[],
            D, F, OUT_CH, EPS, FINAL_EPS, ctx,
        )
    var t_fwd = perf_counter_ns()

    var tgt_patch = latent_inputs.target_patch.copy()
    var real_nout = len(tgt_patch)
    var seq_nout = len(fwd.out)
    var d_loss = List[Float32]()
    var pred_vals = List[Float32]()
    var sse = 0.0
    var inv_n = Float32(2.0) / Float32(real_nout)
    for i in range(real_nout):
        var pred = -fwd.out[i]
        pred_vals.append(pred)
        var diff = pred - tgt_patch[i]
        sse += Float64(diff) * Float64(diff)
        d_loss.append(-inv_n * diff)
    for _i in range(real_nout, seq_nout):
        d_loss.append(Float32(0.0))
    var loss = Float32(sse / Float64(real_nout))
    var t_loss = perf_counter_ns()

    if k == 1:
        var ps = _flat_stats(pred_vals)
        var ts = _flat_stats(tgt_patch)
        print("[DEBUG step=1] bucket=", LAT_H_B, "x", LAT_W_B, " cap=", CAP_LEN_B,
              " sigma_idx=", sigma_idx, " sig=", sig,
              " pred mean=", Float32(ps.mean), " std=", Float32(ps.std),
              " max_abs=", ps.max_abs, " target mean=", Float32(ts.mean),
              " std=", Float32(ts.std), " max_abs=", ts.max_abs)

    var grads: ZImageLoraGrads
    comptime if ZIMAGE_V2_ENGINE:
        grads = zimage_stack_lora_backward_main_device_v2[H, Dh, N_IMG_B, N_TXT_B, S_B](
            d_loss, main_blocks, mvall.value().per_block, lora_dev,
            f_scale.copy(), final_lin_w,
            uni_cos[], uni_sin[], fwd,
            D, F, OUT_CH, EPS, FINAL_EPS, ctx,
        )
    else:
        grads = zimage_stack_lora_backward_main_device[H, Dh, N_IMG_B, N_TXT_B, S_B](
            d_loss, main_blocks, main_mod, lora_dev,
            f_scale.copy(), final_lin_w,
            uni_cos[], uni_sin[], fwd,
            D, F, OUT_CH, EPS, FINAL_EPS, ctx,
        )
    var t_bwd = perf_counter_ns()

    var gn_before = _clip(grads, CLIP_GRAD_NORM, TRAIN_ADAPTER_START, n_adapters)
    var step_lr = ot_lr_for_optimizer_step(train_cfg, k)
    comptime if ZIMAGE_V2_ENGINE:
        # Resident AdamW: G up, in-place kernel on persistent P/M/V, P back to
        # the host mirror (b_absum/save contracts unchanged). Same kernel,
        # same values as zimage_lora_adamw_step_main_only — bit-identical
        # expected; gated on anchors + b1match-vs-b2dup cross-path identity.
        fused_lora_adamw_plain_step_resident(
            opt_state, lora.ad, grads.d_a, grads.d_b, k, step_lr,
            train_cfg.beta1, train_cfg.beta2, train_cfg.eps,
            train_cfg.weight_decay, ctx,
        )
    else:
        zimage_lora_adamw_step_main_only(
            lora, grads, k, step_lr, ctx,
            train_cfg.beta1, train_cfg.beta2, train_cfg.eps, train_cfg.weight_decay,
        )
    var t_opt = perf_counter_ns()

    var t1 = perf_counter_ns()
    var secs = Float64(t1 - t0) / 1.0e9
    var b_absum = Float32(0.0)
    var b_nonzero = 0
    for i in range(TRAIN_ADAPTER_START, n_adapters):
        var bs2 = _absum(lora.ad[i].b)
        b_absum += bs2
        if bs2 > 0.0:
            b_nonzero += 1
    print_trainer_progress(
        String("ZImage-lora"), k, run_steps, 1,
        loss, Float64(gn_before), secs, 0.0,
        Float64(t1 - train_start_ns) / 1.0e9,
    )
    if grads.nonfinite_lora_grads != 0:
        print("[ZImage-lora] warning nonfinite_lora_grads=", grads.nonfinite_lora_grads)
    print("[TIMING step=", k,
          "] prep=", Float32(Float64(t_prep - t0) / 1.0e9),
          " lora_upload=", Float32(Float64(t_lora - t_prep) / 1.0e9),
          " fwd=", Float32(Float64(t_fwd - t_lora) / 1.0e9),
          " loss=", Float32(Float64(t_loss - t_fwd) / 1.0e9),
          " bwd=", Float32(Float64(t_bwd - t_loss) / 1.0e9),
          " opt=", Float32(Float64(t_opt - t_bwd) / 1.0e9))
    return StepResult(loss, Float32(gn_before), Float32(secs), b_absum, b_nonzero, grads.nonfinite_lora_grads)


def main() raises:
    var a = argv()
    var cfg_path = String(DEFAULT_CONFIG)
    var arg_base = 1
    if len(a) >= 2:
        var first = String(a[1])
        if first.endswith(String(".json")):
            cfg_path = first.copy()
            arg_base = 2

    var train_cfg = read_model_config(cfg_path)
    validate_zimage_train_config(train_cfg)
    var cache_preflight = create_onetrainer_cache_preflight_plan(train_cfg)
    validate_onetrainer_cache_preflight_plan(cache_preflight)

    def _is_gate_mode(v: String) -> Bool:
        return (
            v == String("b2dup") or v == String("b1match")
            or v == String("b1match2")
        )

    var run_steps = DEFAULT_RUN_STEPS
    if len(a) > arg_base and not _is_gate_mode(String(a[arg_base])):
        run_steps = _parse_nonnegative_int(String(a[arg_base]))
    elif train_cfg.only_cache:
        run_steps = 0
    var start_step = 0
    if len(a) > arg_base + 1 and not _is_gate_mode(String(a[arg_base + 1])):
        start_step = _parse_nonnegative_int(String(a[arg_base + 1]))
    if start_step > run_steps:
        raise Error(String("start_step ") + String(start_step) + String(" > run_steps ") + String(run_steps))
    var resume_state = String("")
    if len(a) > arg_base + 2 and not _is_gate_mode(String(a[arg_base + 2])):
        resume_state = String(a[arg_base + 2])
    # batch-2 trajectory gate modes (see _train_one_step_bucket_b2 header):
    #   b2dup: B2 path with duplicated sample/seed -> must equal b1match run.
    var b2_dup = False
    var b1_match = False
    var b1_match2 = False
    for ai in range(1, len(a)):
        if String(a[ai]) == String("b2dup"):
            b2_dup = True
        elif String(a[ai]) == String("b1match"):
            b1_match = True
        elif String(a[ai]) == String("b1match2"):
            b1_match2 = True

    var transformer_dir = zimage_transformer_dir_from_train_config(train_cfg)
    var cache_dir = zimage_cache_dir_from_train_config(train_cfg)
    var sample_cadence = zimage_sample_cadence_from_train_config(cfg_path, train_cfg)
    var sample_enabled = zimage_sampling_enabled(sample_cadence)

    print("=== Z-Image REAL LoRA training loop ===")
    print("  config:", cfg_path)
    print("  arch: D=", D, " H=", H, " Dh=", Dh, " F=", F, " out_ch=", OUT_CH)
    print("  depth: NR=", NUM_NR, " CR=", NUM_CR, " MAIN=", MAIN_DEPTH, " (full model MAIN=30)")
    print("  buckets: 72x56 cap224/cap256, 88x48 cap224/cap256")
    print("  recipe: rank=", train_cfg.lora_rank, " alpha=", train_cfg.lora_alpha,
          " lr=", train_cfg.lr, " shift=", train_cfg.timestep_shift,
          " vae_shift=", VAE_SHIFT, " vae_scale=", VAE_SCALE)
    print("  run_steps=", run_steps, " config_max_steps=", train_cfg.max_steps)
    print(
        "  cadence: save_every=", train_cfg.save_every,
        " sample_after=", sample_cadence.sample_after,
        " unit=", sample_time_unit_name(sample_cadence.sample_after_unit),
        " skip_first=", sample_cadence.sample_skip_first,
        " sample_file=", sample_cadence.sample_definition_file_name,
    )
    print("  weights:", transformer_dir)
    print("  cache:", cache_dir)
    if train_cfg.only_cache:
        print("[ZImage] only_cache requested; no train steps will run in this trainer")
        return

    var ctx = DeviceContext()

    # ── cache first: fail before loading the 24 GB-class model if prepare has
    # not produced the local Mojo cache yet.
    var cache = KleinCache(cache_dir)
    print("[cache] samples:", cache.count())
    var k0 = cache.peek_key(0, ctx)
    print("[cache] first latent: C=", k0.c, " H=", k0.h, " W=", k0.w, " text_seq=", k0.seq)

    # ── load real base weights (frozen) ──────────────────────────────────────
    print("[load] opening sharded transformer dir")
    var st = ShardedSafeTensors.open(transformer_dir)
    print("[load] aux (embedders / per-block adaLN / final layer)")
    var aux = load_zimage_real_aux(st, NUM_NR, MAIN_DEPTH, ctx)
    print("[load] blocks: NR + CR + MAIN")
    var nr_blocks = List[ZImageBlockWeights]()
    for i in range(NUM_NR):
        nr_blocks.append(load_zimage_block_weights_prefixed_mixed(st, String("noise_refiner.") + String(i), ctx))
    var cr_blocks = List[ZImageBlockWeights]()
    for i in range(NUM_CR):
        cr_blocks.append(load_zimage_block_weights_prefixed_mixed(st, String("context_refiner.") + String(i), ctx))
    var main_blocks = List[ZImageBlockWeights]()
    for i in range(MAIN_DEPTH):
        main_blocks.append(load_zimage_block_weights_prefixed_mixed(st, String("layers.") + String(i), ctx))
    print("[load] resident blocks:", len(nr_blocks), "nr +", len(cr_blocks), "cr +", len(main_blocks), "main")
    var final_lin_w = aux.final_lin_w[].clone(ctx)
    var final_lin_b = aux.final_lin_b[].clone(ctx)

    var x_pad_h = aux.x_pad_token[].to_host(ctx)
    var cap_pad_h = aux.cap_pad_token[].to_host(ctx)
    print("[load] learned x/cap pad tokens loaded")

    # ── LoRA set (B=0 init -> identity at step 0) ────────────────────────────
    var lora = build_zimage_lora_set(NUM_NR, NUM_CR, MAIN_DEPTH, D, F, RANK, ALPHA)
    if resume_state != String("") and resume_state != String("-"):
        print("[ZImage-lora] loading resume state:", resume_state)
        lora = load_zimage_lora_main_only_state(
            NUM_NR, NUM_CR, MAIN_DEPTH, RANK, ALPHA, D, F, resume_state, ctx,
        )
    var n_adapters = (NUM_NR + NUM_CR + MAIN_DEPTH) * ZIMAGE_SLOTS
    print("[lora] adapters:", TRAIN_ADAPTER_COUNT, "trainable main-layer adapters;",
          n_adapters, "allocated total (refiners frozen/excluded)")

    # v2 engine (resident-set): persistent device P/M/V + a device LoRA set
    # whose MAIN adapters view the optimizer's live param buffer. Built ONCE
    # (after any resume load) — the per-step set upload + P/M/V round trips
    # disappear. Used only when ZIMAGE_V2_ENGINE; the off path ignores both.
    var opt_state = lora_adamw_plain_device_state_init(
        lora.ad, TRAIN_ADAPTER_START, n_adapters, ctx,
    )
    var resident_dev = zimage_lora_set_to_device_resident(lora, opt_state, ctx)

    var first_loss = Float32(0.0)
    var last_loss = Float32(0.0)
    var b_absum_init = Float32(0.0)
    for i in range(TRAIN_ADAPTER_START, n_adapters):
        b_absum_init += _absum(lora.ad[i].b)
    print("[lora] LoRA-B |.|_1 at init =", b_absum_init, " (expect 0.0)")

    if sample_enabled and should_sample_completed_step(sample_cadence, 0):
        print("[cadence] step 0 sample due; Z-Image uses split-process sampler requests")
    var next_sample = next_sample_completed_step(sample_cadence, 0, train_cfg.max_steps)
    print("[cadence] next sample completed_step=", next_sample)

    var train_start = perf_counter_ns()
    for k in range(start_step + 1, run_steps + 1):
        var slot = 0 if OVERFIT_PROBE else (k - 1) % cache.count()
        var step_seed = UInt64(1) if OVERFIT_PROBE else UInt64(k)
        if b1_match:
            slot = ((k - 1) * 2) % cache.count()
            step_seed = UInt64(2 * k)
        elif b1_match2:
            slot = ((k - 1) * 2 + 1) % cache.count()
            step_seed = UInt64(2 * k + 1)
        var key = cache.peek_key(slot, ctx)
        if key.c != LAT_C:
            raise Error("train_zimage_real: unsupported latent channel count")
        var valid_cap = _cache_valid_cap(cache, slot, ctx)
        var loss: Float32
        if train_cfg.batch_size == 2:
            # batch-2: two consecutive cache slots per step; both must share
            # the latent bucket; caption bucket = max of the two.
            var slot0 = ((k - 1) * 2) % cache.count()
            var slot1 = ((k - 1) * 2 + 1) % cache.count()
            var seed_a = UInt64(2 * k)
            var seed_b = UInt64(2 * k + 1)
            if b2_dup:
                slot1 = slot0
                seed_b = seed_a
            var key0 = cache.peek_key(slot0, ctx)
            var key1 = cache.peek_key(slot1, ctx)
            if key0.c != LAT_C or key1.c != LAT_C:
                raise Error("train_zimage_real b2: unsupported latent channels")
            if key0.h != key1.h or key0.w != key1.w:
                raise Error("train_zimage_real b2: paired samples in different buckets")
            var vc0 = _cache_valid_cap(cache, slot0, ctx)
            var vc1 = _cache_valid_cap(cache, slot1, ctx)
            var vc = vc0 if vc0 > vc1 else vc1
            if key0.h == 64 and key0.w == 64:
                if vc <= 224:
                    var rb2a = _train_one_step_bucket_b2[64, 64, 224](
                        k, run_steps, slot0, slot1, seed_a, seed_b, cache, aux,
                        nr_blocks, cr_blocks, main_blocks, lora, opt_state, resident_dev, n_adapters,
                        final_lin_w, final_lin_b, x_pad_h, cap_pad_h,
                        train_cfg, train_start, ctx,
                    )
                    loss = rb2a.loss
                elif vc <= 256:
                    var rb2b = _train_one_step_bucket_b2[64, 64, 256](
                        k, run_steps, slot0, slot1, seed_a, seed_b, cache, aux,
                        nr_blocks, cr_blocks, main_blocks, lora, opt_state, resident_dev, n_adapters,
                        final_lin_w, final_lin_b, x_pad_h, cap_pad_h,
                        train_cfg, train_start, ctx,
                    )
                    loss = rb2b.loss
                else:
                    raise Error("train_zimage_real b2: caption too long for 256 bucket")
            else:
                raise Error("train_zimage_real b2: only the 64x64 bucket is wired")
        elif key.h == 72 and key.w == 56:
            if valid_cap <= 224:
                var r_72_224 = _train_one_step_bucket[72, 56, 224](
                    k, run_steps, slot, step_seed, cache, aux, nr_blocks, cr_blocks, main_blocks,
                    lora, opt_state, resident_dev, n_adapters, final_lin_w, final_lin_b, x_pad_h, cap_pad_h,
                    train_cfg, train_start, ctx,
                )
                loss = r_72_224.loss
            elif valid_cap <= 256:
                var r_72_256 = _train_one_step_bucket[72, 56, 256](
                    k, run_steps, slot, step_seed, cache, aux, nr_blocks, cr_blocks, main_blocks,
                    lora, opt_state, resident_dev, n_adapters, final_lin_w, final_lin_b, x_pad_h, cap_pad_h,
                    train_cfg, train_start, ctx,
                )
                loss = r_72_256.loss
            else:
                raise Error("train_zimage_real: caption too long for 256-token production bucket")
        elif key.h == 88 and key.w == 48:
            if valid_cap <= 224:
                var r_88_224 = _train_one_step_bucket[88, 48, 224](
                    k, run_steps, slot, step_seed, cache, aux, nr_blocks, cr_blocks, main_blocks,
                    lora, opt_state, resident_dev, n_adapters, final_lin_w, final_lin_b, x_pad_h, cap_pad_h,
                    train_cfg, train_start, ctx,
                )
                loss = r_88_224.loss
            elif valid_cap <= 256:
                var r_88_256 = _train_one_step_bucket[88, 48, 256](
                    k, run_steps, slot, step_seed, cache, aux, nr_blocks, cr_blocks, main_blocks,
                    lora, opt_state, resident_dev, n_adapters, final_lin_w, final_lin_b, x_pad_h, cap_pad_h,
                    train_cfg, train_start, ctx,
                )
                loss = r_88_256.loss
            else:
                raise Error("train_zimage_real: caption too long for 256-token production bucket")
        elif key.h == 64 and key.w == 64:
            # square 512px bucket (OneTrainer alina_zimage_512 cache: 64x64
            # latents) — same generic step at different comptime shape params.
            if valid_cap <= 224:
                var r_64_224 = _train_one_step_bucket[64, 64, 224](
                    k, run_steps, slot, step_seed, cache, aux, nr_blocks, cr_blocks, main_blocks,
                    lora, opt_state, resident_dev, n_adapters, final_lin_w, final_lin_b, x_pad_h, cap_pad_h,
                    train_cfg, train_start, ctx,
                )
                loss = r_64_224.loss
            elif valid_cap <= 256:
                var r_64_256 = _train_one_step_bucket[64, 64, 256](
                    k, run_steps, slot, step_seed, cache, aux, nr_blocks, cr_blocks, main_blocks,
                    lora, opt_state, resident_dev, n_adapters, final_lin_w, final_lin_b, x_pad_h, cap_pad_h,
                    train_cfg, train_start, ctx,
                )
                loss = r_64_256.loss
            else:
                raise Error("train_zimage_real: caption too long for 256-token production bucket")
        else:
            raise Error("train_zimage_real: unsupported Z-Image production bucket")
        if k == start_step + 1:
            first_loss = loss
        last_loss = loss

        var saved_this_step = False
        if zimage_should_save_checkpoint(train_cfg, k):
            var save_path = _step_lora_path(
                zimage_output_lora_path_from_train_config(train_cfg, run_steps), k
            )
            _ = save_zimage_lora_main_only(lora, save_path, ctx)
            var state_path = save_path + String(".state.safetensors")
            comptime if ZIMAGE_V2_ENGINE:
                lora_adamw_plain_device_state_sync_moments(opt_state, lora.ad, ctx)
            _ = save_zimage_lora_main_only_state(lora, state_path, ctx)
            saved_this_step = True
            print("[ZImage-lora] save_state step=", k, " path=", state_path)
        if sample_enabled and should_sample_completed_step(sample_cadence, k):
            if zimage_should_save_before_sample(sample_cadence, k, saved_this_step):
                var sample_path = _step_lora_path(
                    zimage_output_lora_path_from_train_config(train_cfg, run_steps), k
                )
                _ = save_zimage_lora_main_only(lora, sample_path, ctx)
                var sample_state = sample_path + String(".state.safetensors")
                comptime if ZIMAGE_V2_ENGINE:
                    lora_adamw_plain_device_state_sync_moments(opt_state, lora.ad, ctx)
                _ = save_zimage_lora_main_only_state(lora, sample_state, ctx)
                print("[ZImage-lora] save_before_sample step=", k, " path=", sample_state)
                var request_path = _write_zimage_sample_request(
                    k, sample_path, sample_state, sample_cadence.sample_definition_file_name
                )
                print(
                    "[cadence] sample request queued completed_step=", k,
                    " request=", request_path,
                )
            else:
                var existing_lora = _step_lora_path(
                    zimage_output_lora_path_from_train_config(train_cfg, run_steps), k
                )
                var existing_state = existing_lora + String(".state.safetensors")
                var request_path2 = _write_zimage_sample_request(
                    k, existing_lora, existing_state, sample_cadence.sample_definition_file_name
                )
                print(
                    "[cadence] sample request queued completed_step=", k,
                    " request=", request_path2,
                )
            print(
                "[cadence] Z-Image sampler is split-process; run request after trainer memory is released",
            )

    print("")
    print("first_loss=", first_loss, " last_loss=", last_loss)
    var b_absum_final = Float32(0.0)
    for i in range(TRAIN_ADAPTER_START, n_adapters):
        b_absum_final += _absum(lora.ad[i].b)
    var trains = b_absum_final > 0.0
    if trains and (last_loss == last_loss):
        print("RESULT: REAL Z-IMAGE LORA TRAIN OK — LoRA-B grew 0 ->", b_absum_final,
              "; loss", first_loss, "->", last_loss,
              (" (DECREASED)" if last_loss < first_loss else " (see trajectory)"))
        _ = sys_system(String("mkdir -p ") + String(LORA_DIR))
        var lora_out = zimage_output_lora_path_from_train_config(train_cfg, run_steps)
        _ = save_zimage_lora_main_only(lora, lora_out, ctx)
        var state_out = lora_out + String(".state.safetensors")
        comptime if ZIMAGE_V2_ENGINE:
            lora_adamw_plain_device_state_sync_moments(opt_state, lora.ad, ctx)
        _ = save_zimage_lora_main_only_state(lora, state_out, ctx)
        print("[ZImage-lora] save_state step=", run_steps, " path=", state_out)
    else:
        print("RESULT: FAIL trains=", trains)


# ── helper: patchify the v-target (noise - latent) into OUT_CH channel-minor ──
# Ordering matches build_x_seq's patchify exactly: view [C,Ht,p,Wt,p] ->
# permute (Ht,Wt,p,p,C) -> reshape [Ht*Wt, p*p*C]. v-target = noise - latent.
def _patchify_target_bf16[
    LAT_H_B: Int, LAT_W_B: Int
](noise_lat: List[Float32], lat_flat: List[BFloat16]) -> List[Float32]:
    var C = LAT_C
    var Hh = LAT_H_B
    var Ww = LAT_W_B
    var p = PATCH
    var ht = Hh // p
    var wt = Ww // p
    # token target in [C,H,W]: t[c,h,w] = noise - latent
    # output ordering: token (ih,iw) -> [ph, pw, c] channel-minor (p*p*C=64).
    var out = List[Float32]()
    for ih in range(ht):
        for iw in range(wt):
            for ph in range(p):
                for pw in range(p):
                    for c in range(C):
                        var hh = ih * p + ph
                        var ww = iw * p + pw
                        var idx = c * Hh * Ww + hh * Ww + ww
                        var lat = (lat_flat[idx].cast[DType.float32]() - VAE_SHIFT) * VAE_SCALE
                        out.append(noise_lat[idx] - lat)
    return out^


def _patchify_target_f16[
    LAT_H_B: Int, LAT_W_B: Int
](noise_lat: List[Float32], lat_flat: List[Float16]) -> List[Float32]:
    var C = LAT_C
    var Hh = LAT_H_B
    var Ww = LAT_W_B
    var p = PATCH
    var ht = Hh // p
    var wt = Ww // p
    var out = List[Float32]()
    for ih in range(ht):
        for iw in range(wt):
            for ph in range(p):
                for pw in range(p):
                    for c in range(C):
                        var hh = ih * p + ph
                        var ww = iw * p + pw
                        var idx = c * Hh * Ww + hh * Ww + ww
                        var lat = (lat_flat[idx].cast[DType.float32]() - VAE_SHIFT) * VAE_SCALE
                        out.append(noise_lat[idx] - lat)
    return out^


def _patchify_target_f32[
    LAT_H_B: Int, LAT_W_B: Int
](noise_lat: List[Float32], lat_flat: List[Float32]) -> List[Float32]:
    var C = LAT_C
    var Hh = LAT_H_B
    var Ww = LAT_W_B
    var p = PATCH
    var ht = Hh // p
    var wt = Ww // p
    var out = List[Float32]()
    for ih in range(ht):
        for iw in range(wt):
            for ph in range(p):
                for pw in range(p):
                    for c in range(C):
                        var hh = ih * p + ph
                        var ww = iw * p + pw
                        var idx = c * Hh * Ww + hh * Ww + ww
                        out.append(noise_lat[idx] - lat_flat[idx])
    return out^


# ── BATCH-2 step (OT-parity batch lever, 2026-06-11) ─────────────────────────
# Two samples stacked along rows [2S, D]; per-sample sigma/noise/adaLN exactly
# like OneTrainer's per-batch-element draws. Loss = mean MSE over both samples
# (grads naturally average through the 1/(2n) factor). GATE:
# training/zimage_batch2_parity.mojo (B2 vs 2x B1, identical draws).
def _train_one_step_bucket_b2[
    LAT_H_B: Int, LAT_W_B: Int, CAP_LEN_B: Int
](
    k: Int,
    run_steps: Int,
    slot0: Int,
    slot1: Int,
    seed0: UInt64,
    seed1: UInt64,
    cache: KleinCache,
    aux: ZImageRealAux,
    nr_blocks: List[ZImageBlockWeights],
    cr_blocks: List[ZImageBlockWeights],
    main_blocks: List[ZImageBlockWeights],
    mut lora: ZImageLoraSet,
    mut opt_state: LoraAdamWPlainDeviceState,
    resident_dev: ZImageLoraDeviceSet,
    n_adapters: Int,
    final_lin_w: Tensor,
    final_lin_b: Tensor,
    x_pad_h: List[Float32],
    cap_pad_h: List[Float32],
    train_cfg: TrainConfig,
    train_start_ns: UInt,
    ctx: DeviceContext,
) raises -> StepResult:
    comptime HT_B = LAT_H_B // PATCH
    comptime WT_B = LAT_W_B // PATCH
    comptime N_IMG_REAL_B = HT_B * WT_B
    comptime IMG_PAD_B = (32 - (N_IMG_REAL_B % 32)) % 32
    comptime N_IMG_B = N_IMG_REAL_B + IMG_PAD_B
    comptime N_TXT_B = CAP_LEN_B
    comptime S_B = N_IMG_B + N_TXT_B

    var t0 = perf_counter_ns()

    # ── per-sample data prep ──────────────────────────────────────────────────
    var s0 = cache.load(slot0, ctx)
    var s1 = cache.load(slot1, ctx)
    var lsh0 = s0.latent.shape()
    var lsh1 = s1.latent.shape()
    if (
        lsh0[1] != LAT_C or lsh0[2] != LAT_H_B or lsh0[3] != LAT_W_B
        or lsh1[1] != LAT_C or lsh1[2] != LAT_H_B or lsh1[3] != LAT_W_B
    ):
        raise Error("train_zimage_real b2: sample in wrong latent bucket")
    var valid_cap0 = _valid_cap_from_mask(s0.text_mask, ctx)
    var valid_cap1 = _valid_cap_from_mask(s1.text_mask, ctx)
    if (
        valid_cap0 <= 0 or valid_cap0 > CAP_LEN_B
        or valid_cap1 <= 0 or valid_cap1 > CAP_LEN_B
    ):
        raise Error("train_zimage_real b2: sample in wrong text bucket")

    var sigma0 = sample_timestep_logit_normal(SEED_BASE + seed0, TIMESTEP_SHIFT)
    var sigma1 = sample_timestep_logit_normal(SEED_BASE + seed1, TIMESTEP_SHIFT)
    var sigma_idx0 = Int(sigma0 * Float32(NUM_TRAIN_TIMESTEPS))
    var sigma_idx1 = Int(sigma1 * Float32(NUM_TRAIN_TIMESTEPS))
    if sigma_idx0 > NUM_TRAIN_TIMESTEPS - 1:
        sigma_idx0 = NUM_TRAIN_TIMESTEPS - 1
    if sigma_idx1 > NUM_TRAIN_TIMESTEPS - 1:
        sigma_idx1 = NUM_TRAIN_TIMESTEPS - 1
    var sig0 = Float32(sigma_idx0 + 1) / Float32(NUM_TRAIN_TIMESTEPS)
    var sig1 = Float32(sigma_idx1 + 1) / Float32(NUM_TRAIN_TIMESTEPS)
    var t_value0 = Float32(NUM_TRAIN_TIMESTEPS - sigma_idx0) / Float32(NUM_TRAIN_TIMESTEPS)
    var t_value1 = Float32(NUM_TRAIN_TIMESTEPS - sigma_idx1) / Float32(NUM_TRAIN_TIMESTEPS)

    var noise0 = _host_noise(LAT_C * LAT_H_B * LAT_W_B, SEED_BASE * UInt64(7919) + seed0)
    var noise1 = _host_noise(LAT_C * LAT_H_B * LAT_W_B, SEED_BASE * UInt64(7919) + seed1)
    var li0 = _build_latent_step_inputs[LAT_H_B, LAT_W_B](s0.latent, noise0, sig0, ctx)
    var li1 = _build_latent_step_inputs[LAT_H_B, LAT_W_B](s1.latent, noise1, sig1, ctx)

    var x_t0 = build_x_seq(aux, li0.noisy_latent, LAT_C, LAT_H_B, LAT_W_B, PATCH, ctx)
    var x_t1 = build_x_seq(aux, li1.noisy_latent, LAT_C, LAT_H_B, LAT_W_B, PATCH, ctx)
    for _pad in range(IMG_PAD_B):
        for c in range(D):
            x_t0.append(x_pad_h[c])
            x_t1.append(x_pad_h[c])

    var cap2_0 = _cap_tensor_from_cache[CAP_LEN_B](s0.text_embedding, valid_cap0, ctx)
    var cap2_1 = _cap_tensor_from_cache[CAP_LEN_B](s1.text_embedding, valid_cap1, ctx)
    var cap_seq0 = build_cap_seq(aux, cap2_0, EPS, ctx)
    var cap_seq1 = build_cap_seq(aux, cap2_1, EPS, ctx)
    for r in range(valid_cap0, CAP_LEN_B):
        for c in range(D):
            cap_seq0[r * D + c] = cap_pad_h[c]
    for r in range(valid_cap1, CAP_LEN_B):
        for c in range(D):
            cap_seq1[r * D + c] = cap_pad_h[c]

    # positions/rope: x table shared; cap + uni tables per sample; the batched
    # uni table = rope over the CONCATENATED per-sample position lists.
    var pos0 = build_positions(N_IMG_B, HT_B, WT_B, CAP_LEN_B, valid_cap0)
    var pos1 = build_positions(N_IMG_B, HT_B, WT_B, CAP_LEN_B, valid_cap1)
    var x_pos = pos0[0].copy()
    var cap_pos0 = pos0[1].copy()
    var cap_pos1 = pos1[1].copy()
    var uni2_pos = List[List[Int]]()
    for i in range(len(x_pos)):
        uni2_pos.append(x_pos[i].copy())
    for i in range(len(cap_pos0)):
        uni2_pos.append(cap_pos0[i].copy())
    for i in range(len(x_pos)):
        uni2_pos.append(x_pos[i].copy())
    for i in range(len(cap_pos1)):
        uni2_pos.append(cap_pos1[i].copy())
    var xr = build_rope(x_pos, H, Dh, ROPE_THETA, AXIS0, AXIS1, AXIS2, ctx)
    var x_cos = xr[0].copy(); var x_sin = xr[1].copy()
    var cr0 = build_rope(cap_pos0, H, Dh, ROPE_THETA, AXIS0, AXIS1, AXIS2, ctx)
    var cr1 = build_rope(cap_pos1, H, Dh, ROPE_THETA, AXIS0, AXIS1, AXIS2, ctx)
    var ur2 = build_rope(uni2_pos, H, Dh, ROPE_THETA, AXIS0, AXIS1, AXIS2, ctx)
    var uni_cos2 = ur2[0].copy(); var uni_sin2 = ur2[1].copy()

    # per-sample adaLN
    var adaln0 = build_adaln(aux, t_value0, ADALN_DIM, T_SCALE, ctx)
    var adaln1 = build_adaln(aux, t_value1, ADALN_DIM, T_SCALE, ctx)
    var nr_mod0 = List[ZImageModVecs]()
    var nr_mod1 = List[ZImageModVecs]()
    for i in range(NUM_NR):
        nr_mod0.append(build_block_modvecs(aux.nr_mod_w[i][], aux.nr_mod_b[i][], adaln0, D, ctx))
        nr_mod1.append(build_block_modvecs(aux.nr_mod_w[i][], aux.nr_mod_b[i][], adaln1, D, ctx))
    var main_mod_b2 = List[ZImageModVecsDevice]()
    for i in range(MAIN_DEPTH):
        var m0 = build_block_modvecs(aux.main_mod_w[i][], aux.main_mod_b[i][], adaln0, D, ctx)
        var m1 = build_block_modvecs(aux.main_mod_w[i][], aux.main_mod_b[i][], adaln1, D, ctx)
        main_mod_b2.append(zimage_modvecs_pack2_to_device(m0, m1, D, ctx))
    var f_scale0 = build_f_scale(aux, adaln0, D, ctx)
    var f_scale2 = f_scale0.copy()
    var f_scale1 = build_f_scale(aux, adaln1, D, ctx)
    for i in range(D):
        f_scale2.append(f_scale1[i])
    var t_prep = perf_counter_ns()

    var lora_dev: ZImageLoraDeviceSet
    comptime if ZIMAGE_V2_ENGINE:
        lora_dev = resident_dev.copy()
    else:
        lora_dev = zimage_lora_set_to_device(lora, ctx)
    var t_lora = perf_counter_ns()

    var fwd = zimage_stack_lora_forward_main_device_b2[H, Dh, N_IMG_B, N_TXT_B, S_B](
        x_t0.copy(), cap_seq0.copy(), x_t1.copy(), cap_seq1.copy(),
        nr_blocks, nr_mod0, nr_mod1, cr_blocks, main_blocks, main_mod_b2, lora_dev,
        f_scale2.copy(), final_lin_w, final_lin_b,
        x_cos[], x_sin[], cr0[0][], cr0[1][], cr1[0][], cr1[1][],
        uni_cos2[], uni_sin2[],
        D, F, OUT_CH, EPS, FINAL_EPS, ctx,
    )
    var t_fwd = perf_counter_ns()

    # ── batch loss: mean MSE over BOTH samples' real outputs ─────────────────
    var tgt0 = li0.target_patch.copy()
    var tgt1 = li1.target_patch.copy()
    var real_nout = len(tgt0)
    var inv_n = Float32(1.0) / Float32(real_nout)   # = 2/(2*real_nout)
    var sse = 0.0
    var d_loss0 = List[Float32]()
    var d_loss1 = List[Float32]()
    for i in range(real_nout):
        var pred = -fwd.out0[i]
        var diff = pred - tgt0[i]
        sse += Float64(diff) * Float64(diff)
        d_loss0.append(-inv_n * diff)
    for i in range(real_nout):
        var pred = -fwd.out1[i]
        var diff = pred - tgt1[i]
        sse += Float64(diff) * Float64(diff)
        d_loss1.append(-inv_n * diff)
    var seq_nout = len(fwd.out0)
    for _i in range(real_nout, seq_nout):
        d_loss0.append(Float32(0.0))
        d_loss1.append(Float32(0.0))
    var loss = Float32(sse / Float64(2 * real_nout))
    var t_loss = perf_counter_ns()

    var grads = zimage_stack_lora_backward_main_device_b2[H, Dh, N_IMG_B, N_TXT_B, S_B](
        d_loss0, d_loss1, main_blocks, main_mod_b2, lora_dev,
        f_scale2.copy(), final_lin_w,
        uni_cos2[], uni_sin2[], fwd,
        D, F, OUT_CH, EPS, FINAL_EPS, ctx,
    )
    var t_bwd = perf_counter_ns()

    var gn_before = _clip(grads, CLIP_GRAD_NORM, TRAIN_ADAPTER_START, n_adapters)
    var step_lr = ot_lr_for_optimizer_step(train_cfg, k)
    comptime if ZIMAGE_V2_ENGINE:
        fused_lora_adamw_plain_step_resident(
            opt_state, lora.ad, grads.d_a, grads.d_b, k, step_lr,
            train_cfg.beta1, train_cfg.beta2, train_cfg.eps,
            train_cfg.weight_decay, ctx,
        )
    else:
        zimage_lora_adamw_step_main_only(
            lora, grads, k, step_lr, ctx,
            train_cfg.beta1, train_cfg.beta2, train_cfg.eps, train_cfg.weight_decay,
        )
    var t_opt = perf_counter_ns()

    var t1 = perf_counter_ns()
    var secs = Float64(t1 - t0) / 1.0e9
    var b_absum = Float32(0.0)
    var b_nonzero = 0
    for i in range(TRAIN_ADAPTER_START, n_adapters):
        var bs2 = _absum(lora.ad[i].b)
        b_absum += bs2
        if bs2 > 0.0:
            b_nonzero += 1
    print_trainer_progress(
        String("ZImage-lora-b2"), k, run_steps, 1,
        loss, Float64(gn_before), secs, 0.0,
        Float64(t1 - train_start_ns) / 1.0e9,
    )
    if grads.nonfinite_lora_grads != 0:
        print("[ZImage-lora] warning nonfinite_lora_grads=", grads.nonfinite_lora_grads)
    print("[TIMING-B2 step=", k,
          "] prep=", Float32(Float64(t_prep - t0) / 1.0e9),
          " lora_upload=", Float32(Float64(t_lora - t_prep) / 1.0e9),
          " fwd=", Float32(Float64(t_fwd - t_lora) / 1.0e9),
          " loss=", Float32(Float64(t_loss - t_fwd) / 1.0e9),
          " bwd=", Float32(Float64(t_bwd - t_loss) / 1.0e9),
          " opt=", Float32(Float64(t_opt - t_bwd) / 1.0e9))
    return StepResult(loss, Float32(gn_before), Float32(secs), b_absum, b_nonzero, grads.nonfinite_lora_grads)
