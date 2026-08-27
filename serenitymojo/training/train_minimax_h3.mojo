# train_minimax_h3.mojo — MiniMax-H3 FL2VA LoRA trainer, IMAGE-mode maiden arm.
#
# Numeric oracle = torchref (pinned upstream checkout @ 04324c28). Production
# recipe controls are cross-checked against the local Musubi H3 fork; Musubi is
# a reference only and is never part of this trainer's runtime:
#   * caches: torchref-native mmh3 pair (h3_train_cache.mojo reader, gated
#     bit-exact); text task t2va (id 0); image items ([24,1,H,W], no audio,
#     no keyframe rows packed for plain image training).
#   * LoRA: configurable dim / alpha (alpha must equal dim) on 4 block Linears
#     (qkv/out/fc1/fc2 — the full target set minus frozen adaln/norms);
#     down = kaiming_uniform(a=sqrt(5)) = U(+-1/sqrt(in)), up = zeros;
#     F32 masters + bnb-compatible blockwise AdamW8bit moments;
#     bf16 compute copies refreshed each step; batch=1.
#   * sigma (image branch): resolution-aware logit-normal —
#     mu = lin(x1=256,y1=0.5 -> x2=6400,y2=1.15)((H/2)*(W/2)), shift=e^mu,
#     t = sigmoid(z), sigma = t*shift/(1+(shift-1)t); both flow shifts 1.0.
#   * x_t / target / loss / d_pred: h3_train_sigma.mojo (gated bit-exact).
#
# DOCUMENTED DEVIATION: sigma is quantized to a 1000-node uniform grid so
# the AdaLN tables can be built ONCE with the gated modcache pass (exact
# table math at each node; 1000 levels = the standard t-grid convention).
# The upstream trainer draws continuous sigma but re-projects 24.3GB of
# adaln weights per step, which our streaming budget does not want.
#
# Per step: read cache pair -> draw+quantize sigma -> noise (h3_noisy_input,
# bit-exact) -> patchify -> packed layout + rope (gated inference frontend,
# frozen) -> frontend embed -> 50-block streamed LoRA fwd (mmap store) ->
# final-layer twin -> token loss + d_pred -> final bwd -> streamed recompute
# bwd -> device AdamW8bit on masters -> refresh bf16 copies. Save canonical
# PEFT LoRA (diffusion_model.*.lora_A/B.weight, BF16) every --save_every plus
# exact resume state (F32 masters, U8 moments/F32 absmax, RNG, recipe, step).
# Validation generation is deliberately process-separated by the run wrapper:
# the trainer exits at sample boundaries, then the pure-Mojo runtime loads the
# saved PEFT adapter after all training VRAM has been released.
from std.collections import Dict
from std.math import sqrt, exp, log, cos
from std.ffi import external_call
from std.memory import alloc, UnsafePointer
from std.memory import ArcPointer
from std.os import listdir
from std.sys import argv
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.tensor import Tensor
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.random import randn
from serenitymojo.ops.tensor_algebra import (
    full_device, concat, add, mul_scalar, mul_scalar_bf16out, slice,
)
from serenitymojo.models.klein.lora_block import LoraAdapterDevice
from serenitymojo.models.dit.minimax_h3_dit import (
    MiniMaxH3DiTConfig, minimax_h3_released_config,
)
from serenitymojo.models.dit.minimax_h3_frontend import (
    minimax_h3_video_patchify, minimax_h3_timestep_embedding,
    minimax_h3_condition_embed, minimax_h3_token_refiner_dynamic,
    _minimax_h3_video_patch_embed_bf16,
    _minimax_h3_audio_patch_embed_bf16,
)
from serenitymojo.models.dit.minimax_h3_rope import build_minimax_h3_rope_tables
from serenitymojo.models.minimax_h3.packing import (
    minimax_h3_build_packed_sequence, minimax_h3_build_row_timesteps,
    MINIMAX_H3_ANCHOR_FIRST, MINIMAX_H3_ANCHOR_LAST,
)
from serenitymojo.models.minimax_h3.dit_frontend import minimax_h3_adaln_indices
from serenitymojo.models.minimax_h3.h3_train_cache import (
    H3CacheItemPaths, h3_discover_cache_items,
    h3_read_latent_cache, h3_read_text_cache,
)
from serenitymojo.models.minimax_h3.h3_train_av import (
    h3_audio_latents_to_rows, h3_audio_rows_to_latents, h3_audio_mask_tensor,
)
from serenitymojo.models.minimax_h3.h3_train_sigma import (
    h3_noisy_input, h3_velocity_target,
    h3_modality_loss, h3_joint_token_loss, h3_loss_grad,
    h3_shift_sigma,
    H3ModalityLoss,
)
from serenitymojo.models.minimax_h3.h3_train_block_store_int8 import (
    H3TrainBlockStoreInt8,
)
from serenitymojo.models.minimax_h3.h3_train_modgrid import H3TrainModGrid
from serenitymojo.models.minimax_h3.h3_block_train import (
    H3BlockLoraDevice, H3BlockLoraGrads,
)
from serenitymojo.models.minimax_h3.h3_stack_train import (
    h3_stack_train_forward_streamed_int8,
    h3_stack_train_backward_streamed_int8,
    h3_stack_train_forward_streamed_int8_seedoff,
    h3_stack_train_backward_streamed_int8_seedoff,
    H3SeedStore,
    H3StackTrainForward,
    H3StackLoraOnlyGrads,
)
from serenitymojo.models.minimax_h3.h3_final_train import (
    H3FinalTrainWeights, h3_final_train_forward, h3_final_train_backward,
)
from serenitymojo.models.minimax_h3.h3_token_refiner_train import (
    H3TokenRefinerTrainForward, H3TokenRefinerTrainWeights,
    h3_token_refiner_swap_fc1_rows,
    h3_token_refiner_train_backward, h3_token_refiner_train_forward,
    h3_token_refiner_train_weights,
)
from serenitymojo.training.adamw8bit import (
    Adam8bitDeviceState,
    adamw8bit_device_state,
    adamw8bit_device_state_from_tensors,
    adamw8bit_device_step,
)
from serenitymojo.training.on_device_global_norm import on_device_grad_stats
from serenitymojo.training.progress_display import print_trainer_progress
from serenitymojo.io.train_config_reader import read_model_config
from serenitymojo.training.train_config import (
    TRAIN_OPTIMIZER_ADAMW_8BIT, TRAIN_DTYPE_BFLOAT_16,
)
from serenitymojo.training.lora_save import (
    F32NamedLora, save_lora_peft_host_f32,
)
from serenitymojo.models.minimax_h3.h3_lora_format import (
    h3_lora_peft_prefix, h3_lora_token_refiner_peft_prefix,
    h3_lora_legacy_slot_key,
)
from serenitymojo.io.ffi import BytePtr
from serenitymojo.pipeline.minimax_h3_t2va import _minimax_h3_load_frontend_weights

comptime TArc = ArcPointer[Tensor]
comptime H3_HEADS = 56
comptime H3_HEAD_DIM = 128
comptime H3_D = 5376
comptime H3_F = 14336
comptime H3_INNER = H3_HEADS * H3_HEAD_DIM
comptime H3_EPS = Float32(1.0e-5)
comptime SIGMA_NODES = 1000
comptime N_BLOCKS = 50
comptime N_REFINER_BLOCKS = 2

comptime SLOT_NAMES_LEN = 4
comptime H3_TARGET_MLP_FC1_FC2 = 1
comptime H3_TARGET_FULL_208 = 2
comptime H3_MAIN_STATE_COUNT = N_BLOCKS * SLOT_NAMES_LEN
comptime H3_TOTAL_STATE_COUNT = (
    H3_MAIN_STATE_COUNT + N_REFINER_BLOCKS * SLOT_NAMES_LEN
)


def _slot_active(slot: Int, target_preset: Int) -> Bool:
    if target_preset == H3_TARGET_MLP_FC1_FC2:
        return slot == 2 or slot == 3
    if target_preset == H3_TARGET_FULL_208:
        return slot >= 0 and slot < SLOT_NAMES_LEN
    return False


def _refiner_slot_active(slot: Int, target_preset: Int) -> Bool:
    return (
        target_preset == H3_TARGET_FULL_208
        and slot >= 0 and slot < SLOT_NAMES_LEN
    )


def _grad_pair(g: H3BlockLoraGrads, s: Int) raises -> Tuple[TArc, TArc]:
    if s == 0:
        return (g.qkv.value().d_a.copy(), g.qkv.value().d_b.copy())
    if s == 1:
        return (g.out.value().d_a.copy(), g.out.value().d_b.copy())
    if s == 2:
        return (g.fc1.value().d_a.copy(), g.fc1.value().d_b.copy())
    if s == 3:
        return (g.fc2.value().d_a.copy(), g.fc2.value().d_b.copy())
    raise Error("bad slot")


def _slot_out_in(slot: Int) raises -> Tuple[Int, Int]:
    if slot == 0:
        return (3 * H3_INNER, H3_D)  # attn.qkv_proj
    if slot == 1:
        return (H3_D, H3_INNER)      # attn.out_proj
    if slot == 2:
        return (2 * H3_F, H3_D)      # mlp.fc1
    if slot == 3:
        return (H3_D, H3_F)          # mlp.fc2
    raise Error("bad slot")


# ── host RNG (policy RNG — identity not parity-bound) ────────────────────────
struct _Rng(Movable):
    var state: UInt64

    def __init__(out self, seed: UInt64):
        self.state = seed if seed != 0 else UInt64(0x9E3779B97F4A7C15)

    def next_u64(mut self) -> UInt64:
        # xorshift64*
        var x = self.state
        x ^= x >> 12
        x ^= x << 25
        x ^= x >> 27
        self.state = x
        return x * UInt64(0x2545F4914F6CDD1D)

    def uniform(mut self) -> Float64:
        return Float64(self.next_u64() >> 11) * (1.0 / 9007199254740992.0)

    def normal(mut self) -> Float64:
        # Box-Muller
        var u1 = self.uniform()
        var u2 = self.uniform()
        if u1 < 1e-300:
            u1 = 1e-300
        return sqrt(-2.0 * log(u1)) * _cos_2pi(u2)


def _cos_2pi(u: Float64) -> Float64:
    return cos(6.283185307179586 * u)


def _image_sigma(mut rng: _Rng, lat_h: Int, lat_w: Int) -> Float64:
    """Upstream image branch: krea2_shift resolution-aware logit-normal.
    tokens = (H/2)*(W/2); mu = lin((256,0.5)->(6400,1.15))(tokens);
    shift = e^mu; t = sigmoid(z); sigma = t*shift/(1+(shift-1)t)."""
    var tokens = Float64((lat_h // 2) * (lat_w // 2))
    var m = (1.15 - 0.5) / (6400.0 - 256.0)
    var mu = m * (tokens - 256.0) + 0.5
    var shift = exp(mu)
    var z = rng.normal()  # sigmoid_scale = 1.0 (upstream default)
    var t = 1.0 / (1.0 + exp(-z))
    return t * shift / (1.0 + (shift - 1.0) * t)


def _normal_ppf(p_in: Float64) -> Float64:
    """Acklam inverse-normal CDF used to transform a stratified uniform draw.
    The approximation error is far below the trainer's 1/999 sigma grid."""
    var p = p_in
    if p < 1.0e-7:
        p = 1.0e-7
    if p > 1.0 - 1.0e-7:
        p = 1.0 - 1.0e-7
    var a1 = -3.969683028665376e1
    var a2 = 2.209460984245205e2
    var a3 = -2.759285104469687e2
    var a4 = 1.383577518672690e2
    var a5 = -3.066479806614716e1
    var a6 = 2.506628277459239
    var b1 = -5.447609879822406e1
    var b2 = 1.615858368580409e2
    var b3 = -1.556989798598866e2
    var b4 = 6.680131188771972e1
    var b5 = -1.328068155288572e1
    var c1 = -7.784894002430293e-3
    var c2 = -3.223964580411365e-1
    var c3 = -2.400758277161838
    var c4 = -2.549732539343734
    var c5 = 4.374664141464968
    var c6 = 2.938163982698783
    var d1 = 7.784695709041462e-3
    var d2 = 3.224671290700398e-1
    var d3 = 2.445134137142996
    var d4 = 3.754408661907416
    var plow = 0.02425
    var phigh = 1.0 - plow
    if p < plow:
        var q = sqrt(-2.0 * log(p))
        return (
            (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
            / ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
        )
    if p > phigh:
        var q = sqrt(-2.0 * log(1.0 - p))
        return -(
            (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
            / ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
        )
    var q = p - 0.5
    var r = q * q
    return (
        (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q
        / (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
    )


def _timestep_bucket(seed: Int, step: Int, buckets: Int) -> Int:
    """One deterministic Fisher-Yates permutation per bucket cycle.
    Resume therefore needs only the global step and not a second RNG state."""
    if buckets <= 1:
        return 0
    var cycle = (step - 1) // buckets
    var position = (step - 1) % buckets
    var order = List[Int]()
    for i in range(buckets):
        order.append(i)
    var brng = _Rng(
        UInt64(seed) ^ (UInt64(cycle + 1) * UInt64(0xD1342543DE82EF95))
    )
    for i in range(buckets - 1, 0, -1):
        var j = Int(brng.next_u64() % UInt64(i + 1))
        var tmp = order[i]
        order[i] = order[j]
        order[j] = tmp
    return order[position]


def _shuffle_data_order(mut order: List[Int], seed: Int, epoch: Int):
    for i in range(len(order)):
        order[i] = i
    var erng = _Rng(
        UInt64(seed) ^ (UInt64(epoch + 1) * UInt64(0x9E3779B97F4A7C15))
    )
    for i in range(len(order) - 1, 0, -1):
        var j = Int(erng.next_u64() % UInt64(i + 1))
        var tmp = order[i]
        order[i] = order[j]
        order[j] = tmp


def _bucketed_image_sigma(
    mut rng: _Rng, seed: Int, step: Int, buckets: Int,
    lat_h: Int, lat_w: Int,
) -> Float64:
    if buckets <= 1:
        return _image_sigma(rng, lat_h, lat_w)
    var bucket = _timestep_bucket(seed, step, buckets)
    var u = (Float64(bucket) + rng.uniform()) / Float64(buckets)
    var z = _normal_ppf(u)
    var tokens = Float64((lat_h // 2) * (lat_w // 2))
    var m = (1.15 - 0.5) / (6400.0 - 256.0)
    var mu = m * (tokens - 256.0) + 0.5
    var shift = exp(mu)
    var t = 1.0 / (1.0 + exp(-z))
    return t * shift / (1.0 + (shift - 1.0) * t)


def _density_scale(mut rng: _Rng, jitter: Float32) -> Float64:
    if jitter <= Float32(0.0):
        return Float64(1.0)
    var span = log(1.0 + Float64(jitter))
    return exp((rng.uniform() * 2.0 - 1.0) * span)


def _encode_rng_state(state: UInt64) -> List[Float32]:
    return [
        Float32(UInt16(state & UInt64(0xFFFF))),
        Float32(UInt16((state >> 16) & UInt64(0xFFFF))),
        Float32(UInt16((state >> 32) & UInt64(0xFFFF))),
        Float32(UInt16((state >> 48) & UInt64(0xFFFF))),
    ]


def _decode_rng_state(meta: List[Float32], offset: Int) raises -> UInt64:
    if len(meta) < offset + 4:
        raise Error("H3 trainer state is missing the exact RNG words")
    var out = UInt64(0)
    for i in range(4):
        var word = UInt64(Int(meta[offset + i]))
        if word > UInt64(0xFFFF):
            raise Error("H3 trainer state contains an invalid RNG word")
        out |= word << UInt64(16 * i)
    return out


def _arg(name: String, default: String) raises -> String:
    var a = argv()
    var flag = String("--") + name
    for i in range(len(a)):
        if String(a[i]) == flag and i + 1 < len(a):
            return String(a[i + 1])
    return default


def _arg_int(name: String, default: Int) raises -> Int:
    var s = _arg(name, String(""))
    if s == String(""):
        return default
    return Int(_atof(s))


def _arg_f32(name: String, default: Float32) raises -> Float32:
    var s = _arg(name, String(""))
    if s == String(""):
        return default
    return Float32(_atof(s))


def _atof(s: String) -> Float64:
    var n = s.byte_length()
    var buf = alloc[UInt8](n + 1)
    var src = s.as_bytes()
    for i in range(n):
        buf[i] = src[i]
    buf[n] = 0
    var v = external_call["atof", Float64](BytePtr(unsafe_from_address=Int(buf)))
    buf.free()
    return v


# ── LoRA state: F32 masters + BF16 compute copies ─────────────────────────────────
struct _AdapterState(Movable):
    var a_m: TArc  # [rank, in] F32 master
    var b_m: TArc  # [out, rank] F32 master

    def __init__(
        out self, var a_m: TArc, var b_m: TArc,
    ):
        self.a_m = a_m^
        self.b_m = b_m^


struct _WarmAdapterLoad(Movable):
    var state: _AdapterState
    var imported: Bool

    def __init__(out self, var state: _AdapterState, imported: Bool):
        self.state = state^
        self.imported = imported

    def into_state(deinit self) -> _AdapterState:
        return self.state^


def _init_adapter_state(
    rank: Int, out_f: Int, in_f: Int, mut rng: _Rng, ctx: DeviceContext
) raises -> _AdapterState:
    # down/A: kaiming_uniform(a=sqrt(5)) == U(+-1/sqrt(in)); up/B: zeros
    var bound = 1.0 / sqrt(Float64(in_f))
    var avals = List[Float32]()
    for _ in range(rank * in_f):
        avals.append(Float32((rng.uniform() * 2.0 - 1.0) * bound))
    var ash: List[Int] = [rank, in_f]
    var a_m = Tensor.from_host(avals, ash^, STDtype.F32, ctx)
    var bsh: List[Int] = [out_f, rank]
    var b_m = full_device(bsh.copy(), Float32(0.0), STDtype.F32, ctx)
    return _AdapterState(TArc(a_m^), TArc(b_m^))


def _zero_adapter_state(
    ctx: DeviceContext
) raises -> _AdapterState:
    # Inactive slots need a placeholder only to preserve the fixed block/slot
    # index map; no forward, optimizer, save, or resume path touches it.
    var sh: List[Int] = [1, 1]
    var a_m = full_device(sh.copy(), Float32(0.0), STDtype.F32, ctx)
    var b_m = full_device(sh^, Float32(0.0), STDtype.F32, ctx)
    return _AdapterState(TArc(a_m^), TArc(b_m^))


def _compute_loras(
    states: List[_AdapterState], rank: Int, scale: Float32,
    target_preset: Int, ctx: DeviceContext,
) raises -> List[H3BlockLoraDevice]:
    """bf16 compute copies of the masters, packed per block."""
    var loras = List[H3BlockLoraDevice]()
    for b in range(N_BLOCKS):
        var slots = List[Optional[LoraAdapterDevice]]()
        for s in range(SLOT_NAMES_LEN):
            if not _slot_active(s, target_preset):
                slots.append(Optional[LoraAdapterDevice](None))
                continue
            var st = b * SLOT_NAMES_LEN + s
            var dims = _slot_out_in(s)
            var a16 = cast_tensor(states[st].a_m[], STDtype.BF16, ctx)
            var b16 = cast_tensor(states[st].b_m[], STDtype.BF16, ctx)
            slots.append(Optional[LoraAdapterDevice](LoraAdapterDevice(
                TArc(a16^), TArc(b16^), rank, dims[1], dims[0], scale,
            )))
        loras.append(H3BlockLoraDevice(
            slots[0].copy(), slots[1].copy(), slots[2].copy(), slots[3].copy(),
        ))
    return loras^


def _compute_refiner_loras(
    states: List[_AdapterState], rank: Int, scale: Float32,
    target_preset: Int, ctx: DeviceContext,
) raises -> List[H3BlockLoraDevice]:
    """Runtime-layout token-refiner LoRAs; FC1 B is [value|gate]."""
    var loras = List[H3BlockLoraDevice]()
    for b in range(N_REFINER_BLOCKS):
        var slots = List[Optional[LoraAdapterDevice]]()
        for s in range(SLOT_NAMES_LEN):
            if not _refiner_slot_active(s, target_preset):
                slots.append(Optional[LoraAdapterDevice](None))
                continue
            var st = H3_MAIN_STATE_COUNT + b * SLOT_NAMES_LEN + s
            var dims = _slot_out_in(s)
            var a16 = cast_tensor(states[st].a_m[], STDtype.BF16, ctx)
            var b16 = cast_tensor(states[st].b_m[], STDtype.BF16, ctx)
            if s == 2:
                b16 = h3_token_refiner_swap_fc1_rows(b16, H3_F, ctx)
            slots.append(Optional[LoraAdapterDevice](LoraAdapterDevice(
                TArc(a16^), TArc(b16^), rank, dims[1], dims[0], scale,
            )))
        loras.append(H3BlockLoraDevice(
            slots[0].copy(), slots[1].copy(), slots[2].copy(), slots[3].copy(),
        ))
    return loras^


def _load_st(st: SafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    var info = st.tensor_info(name)
    return Tensor.from_view(
        from_parts(info.dtype, info.shape.copy(), st.tensor_bytes(name)), ctx
    )


def _warm_load_or_init_adapter(
    st: SafeTensors, prefix: String, rank: Int, out_f: Int, in_f: Int,
    mut rng: _Rng, ctx: DeviceContext,
) raises -> _WarmAdapterLoad:
    """Load one canonical PEFT pair as F32 masters, or initialize it.

    Partial warm-resume is deliberately pair-atomic: one-sided A/B artifacts
    fail loudly instead of silently mixing a trained half with a fresh half.
    """
    var a_name = prefix + String(".lora_A.weight")
    var b_name = prefix + String(".lora_B.weight")
    var has_a = st.has_tensor(a_name)
    var has_b = st.has_tensor(b_name)
    if has_a != has_b:
        raise Error(
            String("H3 warm resume has an incomplete PEFT pair for ") + prefix
        )
    if not has_a:
        var fresh = _init_adapter_state(rank, out_f, in_f, rng, ctx)
        return _WarmAdapterLoad(fresh^, False)

    var a_info = st.tensor_info(a_name)
    var b_info = st.tensor_info(b_name)
    if (
        len(a_info.shape) != 2 or a_info.shape[0] != rank
        or a_info.shape[1] != in_f
    ):
        raise Error(
            String("H3 warm resume A shape mismatch for ") + a_name
            + String(": expected [") + String(rank) + String(",")
            + String(in_f) + String("]")
        )
    if (
        len(b_info.shape) != 2 or b_info.shape[0] != out_f
        or b_info.shape[1] != rank
    ):
        raise Error(
            String("H3 warm resume B shape mismatch for ") + b_name
            + String(": expected [") + String(out_f) + String(",")
            + String(rank) + String("]")
        )
    if (
        (a_info.dtype != STDtype.F32 and a_info.dtype != STDtype.BF16
         and a_info.dtype != STDtype.F16)
        or (b_info.dtype != STDtype.F32 and b_info.dtype != STDtype.BF16
            and b_info.dtype != STDtype.F16)
    ):
        raise Error(
            String("H3 warm resume requires F32/BF16/F16 PEFT tensors for ")
            + prefix
        )
    var a_loaded = _load_st(st, a_name, ctx)
    var b_loaded = _load_st(st, b_name, ctx)
    var a_m = cast_tensor(a_loaded, STDtype.F32, ctx)
    var b_m = cast_tensor(b_loaded, STDtype.F32, ctx)
    var loaded = _AdapterState(TArc(a_m^), TArc(b_m^))
    return _WarmAdapterLoad(loaded^, True)


def _load_st_u8(
    st: SafeTensors, name: String, ctx: DeviceContext
) raises -> Tensor:
    """Raw-byte U8 upload; Tensor.from_view only accepts compute dtypes."""
    var info = st.tensor_info(name)
    if info.dtype != STDtype.U8:
        raise Error(String("H3 trainer state tensor is not U8: ") + name)
    var bytes = st.tensor_bytes(name)
    var host = ctx.enqueue_create_host_buffer[DType.uint8](len(bytes))
    var hp = host.unsafe_ptr()
    for i in range(len(bytes)):
        hp[i] = bytes[i]
    var dev = ctx.enqueue_create_buffer[DType.uint8](len(bytes))
    ctx.enqueue_copy(dst_buf=dev, src_buf=host)
    ctx.synchronize()
    return Tensor(dev^, info.shape.copy(), STDtype.U8)


def _param_state_names(target_preset: Int) raises -> List[String]:
    var names = List[String]()
    for b in range(N_BLOCKS):
        for s in range(SLOT_NAMES_LEN):
            if not _slot_active(s, target_preset):
                continue
            var base = String("b") + String(b) + "_" + h3_lora_legacy_slot_key(s)
            names.append(base + "_a_p")
            names.append(base + "_b_p")
    for b in range(N_REFINER_BLOCKS):
        for s in range(SLOT_NAMES_LEN):
            if not _refiner_slot_active(s, target_preset):
                continue
            var base = (
                String("tr") + String(b) + "_" + h3_lora_legacy_slot_key(s)
            )
            names.append(base + "_a_p")
            names.append(base + "_b_p")
    return names^


def _optimizer_params(
    states: List[_AdapterState], target_preset: Int
) -> List[TArc]:
    var params = List[TArc]()
    for b in range(N_BLOCKS):
        for s in range(SLOT_NAMES_LEN):
            if not _slot_active(s, target_preset):
                continue
            var i = b * SLOT_NAMES_LEN + s
            params.append(states[i].a_m.copy())
            params.append(states[i].b_m.copy())
    for b in range(N_REFINER_BLOCKS):
        for s in range(SLOT_NAMES_LEN):
            if not _refiner_slot_active(s, target_preset):
                continue
            var i = H3_MAIN_STATE_COUNT + b * SLOT_NAMES_LEN + s
            params.append(states[i].a_m.copy())
            params.append(states[i].b_m.copy())
    return params^


def _base_loras() -> List[H3BlockLoraDevice]:
    var out = List[H3BlockLoraDevice]()
    for _ in range(N_BLOCKS):
        out.append(H3BlockLoraDevice(
            Optional[LoraAdapterDevice](None),
            Optional[LoraAdapterDevice](None),
            Optional[LoraAdapterDevice](None),
            Optional[LoraAdapterDevice](None),
        ))
    return out^


def _base_refiner_loras() -> List[H3BlockLoraDevice]:
    var out = List[H3BlockLoraDevice]()
    for _ in range(N_REFINER_BLOCKS):
        out.append(H3BlockLoraDevice(
            Optional[LoraAdapterDevice](None),
            Optional[LoraAdapterDevice](None),
            Optional[LoraAdapterDevice](None),
            Optional[LoraAdapterDevice](None),
        ))
    return out^


comptime _CStr = UnsafePointer[UInt8, MutExternalOrigin]


def _cstr(s: String) -> _CStr:
    var n = s.byte_length()
    var buf = alloc[UInt8](n + 1)
    var src = s.as_bytes()
    for i in range(n):
        buf[i] = src[i]
    buf[n] = 0
    return buf


def main() raises:
    # sync allocator: the mod-grid build churns ~25GB of freed transients;
    # the async pool would cache them (max_cache 22GB) and OOM later allocs
    # (the MJ-1142 cumulative-peak class). Must precede the DeviceContext.
    _ = external_call["setenv", Int32](
        _cstr(String("MODULAR_DEVICE_CONTEXT_SYNC_MODE")),
        _cstr(String("true")),
        Int32(0),
    )
    var raw_args = argv()
    var config_path = _arg(String("config"), String(""))
    if (
        config_path == String("") and len(raw_args) > 1
        and not String(raw_args[1]).startswith("--")
    ):
        config_path = String(raw_args[1])
    if config_path == String(""):
        raise Error(
            "MiniMax-H3 trainer is config-driven: pass <config.json> or "
            "--config <config.json>"
        )
    var train_cfg = read_model_config(config_path)
    train_cfg.validate_serenity_trainer_policy_config()
    train_cfg.validate_offload_checkpoint_config()
    if train_cfg.name != String("minimax_h3"):
        raise Error("MiniMax-H3 trainer config requires model_type=minimax_h3")
    if train_cfg.optimizer != TRAIN_OPTIMIZER_ADAMW_8BIT:
        raise Error("MiniMax-H3 trainer requires optimizer=ADAMW_8BIT")
    if train_cfg.train_dtype != TRAIN_DTYPE_BFLOAT_16:
        raise Error("MiniMax-H3 trainer requires train_dtype=BFLOAT_16")
    if train_cfg.grad_accum_steps != 1:
        raise Error("MiniMax-H3 trainer currently requires grad_accum_steps=1")
    if train_cfg.quantized_resident != String("int_w8a8"):
        raise Error("MiniMax-H3 trainer requires quantized_resident=int_w8a8")
    if not train_cfg.layer_filter_regex:
        raise Error(
            "MiniMax-H3 trainer requires layer_filter_regex=true"
        )
    var target_preset = H3_TARGET_MLP_FC1_FC2
    if train_cfg.layer_filter_preset == String("h3_full_aitoolkit_208"):
        target_preset = H3_TARGET_FULL_208
    elif train_cfg.layer_filter_preset != String("h3_mlp_fc1_fc2"):
        raise Error(
            "MiniMax-H3 layer_filter_preset must be h3_mlp_fc1_fc2 or "
            "h3_full_aitoolkit_208"
        )
    if train_cfg.checkpoint == String(""):
        raise Error("MiniMax-H3 trainer config requires checkpoint")
    var cache_dir = train_cfg.dataset_cache_dir
    if cache_dir == String(""):
        cache_dir = train_cfg.cache_dir
    var ckpt = train_cfg.checkpoint
    var out_dir = train_cfg.workspace_dir
    var name = train_cfg.save_filename_prefix
    if name == String(""):
        raise Error("MiniMax-H3 trainer config requires save_filename_prefix")
    var max_steps = train_cfg.max_steps
    var lr = train_cfg.lr
    var optimizer_beta1 = train_cfg.beta1
    var optimizer_beta2 = train_cfg.beta2
    var optimizer_eps = train_cfg.eps
    var optimizer_weight_decay = train_cfg.weight_decay
    var rank = train_cfg.lora_rank
    var alpha = train_cfg.lora_alpha
    var seed = Int(train_cfg.seed)
    var save_every = train_cfg.save_every
    var sample_every = train_cfg.sample_every
    var timestep_buckets = train_cfg.h3_num_timestep_buckets
    var lr_warmup_steps = train_cfg.lr_warmup_steps
    var max_grad_norm = train_cfg.max_grad_norm
    var spatial_density_jitter = train_cfg.h3_spatial_density_jitter
    var base_preservation_weight = train_cfg.h3_base_preservation_loss_weight
    var base_preservation_probability = train_cfg.h3_base_preservation_probability
    var resident_blocks = train_cfg.resident_blocks
    if resident_blocks < 0:
        resident_blocks = 42
    if resident_blocks > N_BLOCKS:
        raise Error("MiniMax-H3 config resident_blocks exceeds 50")
    var baseline_only = _arg_int(String("baseline_only"), 0) != 0
    var cadence_segment = _arg_int(String("cadence_segment"), 0) != 0
    var warm_init_only = _arg_int(String("warm_init_only"), 0) != 0
    if cache_dir == String(""):
        raise Error("MiniMax-H3 config dataset_cache_dir is required")
    if out_dir == String(""):
        raise Error("MiniMax-H3 config workspace_dir is required")
    if max_steps <= 0:
        raise Error("MiniMax-H3 config max_steps must be positive")
    if rank <= 0:
        raise Error("MiniMax-H3 config lora_rank must be positive")
    if save_every <= 0 or sample_every <= 0:
        raise Error("MiniMax-H3 config save_every and sample_every must be positive")
    if timestep_buckets <= 0:
        raise Error("--num_timestep_buckets must be positive")
    if lr_warmup_steps < 0:
        raise Error("--lr_warmup_steps cannot be negative")
    if max_grad_norm <= Float32(0.0):
        raise Error("--max_grad_norm must be positive")
    if spatial_density_jitter < Float32(0.0):
        raise Error("--h3_spatial_density_jitter cannot be negative")
    if base_preservation_weight < Float32(0.0):
        raise Error("--h3_base_preservation_loss_weight cannot be negative")
    if (
        base_preservation_probability <= Float32(0.0)
        or base_preservation_probability > Float32(1.0)
    ):
        raise Error("--h3_base_preservation_probability must be within (0,1]")
    # Canonical PEFT files carry A/B only. Match the rest of the stack's
    # alpha=rank contract instead of silently changing inference strength.
    if alpha != Float32(rank):
        raise Error(
            "H3 canonical PEFT export requires lora_alpha == lora_rank because the "
            "external artifact does not carry per-module alpha"
        )
    if train_cfg.resume_state != String("") and not train_cfg.warm_resume:
        raise Error(
            "H3 config resume_state currently accepts canonical PEFT warm resume "
            "only; set warm_resume=true. Exact continuation from this run's "
            "own state sidecar remains automatic."
        )
    if train_cfg.warm_resume and train_cfg.resume_state == String(""):
        raise Error("H3 config warm_resume=true requires resume_state")
    if warm_init_only and train_cfg.resume_state == String(""):
        raise Error("--warm_init_only requires config resume_state")
    var scale = alpha / Float32(rank)
    # Keep config parsing and fail-fast validation before CUDA ownership.
    var ctx = DeviceContext()

    print("[h3-train] cache_dir:", cache_dir)
    print("[h3-train] config:", config_path)
    print("[h3-train] recipe: dim", rank, "alpha", alpha, "lr", lr,
          "steps", max_steps, "seed", seed)
    print("[h3-train] optimizer AdamW8bit; warmup", lr_warmup_steps,
          "grad_clip", max_grad_norm, "timestep_buckets", timestep_buckets,
          "betas", optimizer_beta1, optimizer_beta2,
          "eps", optimizer_eps, "weight_decay", optimizer_weight_decay)
    if target_preset == H3_TARGET_FULL_208:
        print("[h3-train] targets: 50 main + 2 token-refiner blocks x 4 (208 adapters)")
    else:
        print("[h3-train] targets: mlp.fc1/fc2 only (100 adapters)")
    print("[h3-train] guidance/preservation/density:",
          train_cfg.guidance_scale,
          base_preservation_weight, base_preservation_probability,
          spatial_density_jitter)
    if baseline_only:
        print(
            "[h3-train] BASELINE ONLY: fresh zero-output LoRA; "
            "no backward, optimizer, resume, or checkpoint writes"
        )

    var items = h3_discover_cache_items(cache_dir)
    if len(items) == 0:
        raise Error("no cache items found")
    print("[h3-train] items:", len(items))

    var config = minimax_h3_released_config()
    var sharded = ShardedSafeTensors.open(ckpt)

    # frontend weights (frozen; fp32-trap keys handled by the gated loader path)
    var frontend_w = _minimax_h3_load_frontend_weights(sharded, config, ctx)
    print("[h3-train] frontend weights loaded")

    # final-layer training twin weights (frozen, bf16 like the model)
    var final_w = H3FinalTrainWeights(
        TArc(cast_tensor(Tensor.from_view(sharded.tensor_view(String("final_layer.norm.weight")), ctx), STDtype.BF16, ctx)),
        TArc(cast_tensor(Tensor.from_view(sharded.tensor_view(String("final_layer.video_out.weight")), ctx), STDtype.BF16, ctx)),
        TArc(cast_tensor(Tensor.from_view(sharded.tensor_view(String("final_layer.video_out.bias")), ctx), STDtype.BF16, ctx)),
        TArc(cast_tensor(Tensor.from_view(sharded.tensor_view(String("final_layer.audio_out.weight")), ctx), STDtype.BF16, ctx)),
        TArc(cast_tensor(Tensor.from_view(sharded.tensor_view(String("final_layer.audio_out.bias")), ctx), STDtype.BF16, ctx)),
    )

    # AdaLN grid: exact tables at 1000 sigma nodes, HOST sidecar (frees
    # ~9.7GB VRAM for the fp8-resident base); per-step fetch is ~10MB H2D.
    var grid_path = ckpt + "/../h3_train_modgrid_" + String(SIGMA_NODES) + ".safetensors"
    var modgrid = H3TrainModGrid.build_or_load(
        grid_path, sharded, frontend_w, config, SIGMA_NODES, ctx,
    )
    ctx.synchronize()

    # Base-refined embeds remain cached for the legacy preset and the frozen
    # preservation teacher. Full-208 also caches condition_proj output, then
    # runs the trainable token refiner in each student step.
    print("[h3-train] precomputing per-item text embeds...")
    var refiner_weights = Optional[H3TokenRefinerTrainWeights](None)
    if target_preset == H3_TARGET_FULL_208:
        refiner_weights = Optional[H3TokenRefinerTrainWeights](
            h3_token_refiner_train_weights(
                frontend_w, config.token_refiner_num_layers
            )
        )
    var text_condition_host = List[List[BFloat16]]()
    var text_embeds_host = List[List[BFloat16]]()
    var text_embeds_rows = List[Int]()
    for i in range(len(items)):
        var tei = h3_read_text_cache(items[i].te_path, ctx)
        var th = tei.hidden[].clone(ctx)
        if th.dtype() != STDtype.BF16:
            th = cast_tensor(th, STDtype.BF16, ctx)
        var t0e = minimax_h3_condition_embed(th, frontend_w, ctx)
        text_condition_host.append(t0e.to_host_bf16(ctx))
        var te_ref = minimax_h3_token_refiner_dynamic[H3_HEADS, H3_HEAD_DIM](
            t0e, frontend_w, config, ctx
        )
        text_embeds_host.append(te_ref.to_host_bf16(ctx))
        text_embeds_rows.append(te_ref.shape()[0])
        ctx.synchronize()
    # Optional guidance-consistent objective for the CFG-distilled base:
    # teacher forward on EMPTY conditioning per step,
    # c_hat = (g + (s-1)*g_empty)/s, loss vs c_hat with d_g scaled by 1/s.
    # The earlier no-guidance likeness failure is not evidence against the
    # plain objective: that run trained against raw per-head-interleaved QKV
    # while sampling used deinterleaved QKV, and inference requantized away
    # much of the adapter. Re-test the default one-pass objective after those
    # two root fixes before requiring this slower two-forward arm. Guidance
    # requires empty-cond TE cache pairs.
    var guidance_scale = train_cfg.guidance_scale
    var empty_condition_host = List[BFloat16]()
    var empty_embeds_host = List[BFloat16]()
    var empty_embeds_rows = 0
    var empty_tags = List[Int]()
    if guidance_scale > Float32(0.0):
        if guidance_scale <= Float32(1.0):
            raise Error("--guidance_scale must be > 1 (or 0 to disable)")
        var te0 = h3_read_text_cache(items[0].te_path, ctx)
        if not te0.has_empty:
            raise Error(
                "--guidance_scale needs empty-cond TE caches (re-cache with"
                " the guidance-empty flag)"
            )
        var eh = te0.empty_hidden[].clone(ctx)
        if eh.dtype() != STDtype.BF16:
            eh = cast_tensor(eh, STDtype.BF16, ctx)
        var e0 = minimax_h3_condition_embed(eh, frontend_w, ctx)
        empty_condition_host = e0.to_host_bf16(ctx)
        var e_ref = minimax_h3_token_refiner_dynamic[H3_HEADS, H3_HEAD_DIM](
            e0, frontend_w, config, ctx
        )
        empty_embeds_host = e_ref.to_host_bf16(ctx)
        empty_embeds_rows = e_ref.shape()[0]
        empty_tags = te0.empty_tags.copy()
        ctx.synchronize()
        print("[h3-train] guidance objective ON: scale", guidance_scale,
              "empty tokens", empty_embeds_rows)

    # Slim the frontend dict. Full-208 retains only the token-refiner Arc
    # carriers above; the legacy preset releases them exactly as before.
    var step_w = Dict[String, ArcPointer[Tensor]]()
    step_w[String("video_patch_proj.weight")] = frontend_w[String("video_patch_proj.weight")].copy()
    step_w[String("video_patch_proj.bias")] = frontend_w[String("video_patch_proj.bias")].copy()
    step_w[String("audio_patch_proj.weight")] = frontend_w[String("audio_patch_proj.weight")].copy()
    step_w[String("audio_patch_proj.bias")] = frontend_w[String("audio_patch_proj.bias")].copy()
    var empty_w = Dict[String, ArcPointer[Tensor]]()
    frontend_w = empty_w^
    ctx.synchronize()
    if target_preset == H3_TARGET_FULL_208:
        print("[h3-train] condition/base text cached; trainable token-refiner retained")
    else:
        print("[h3-train] text embeds cached; frontend weights slimmed")

    # Direct W8A8 resident frozen base (Klein trainer pattern): compact INT8
    # weights stay on device and execute directly; no per-visit BF16 expansion.
    print("[h3-train] building direct-int8 resident base (one streamed pass)...")
    # tail blocks stream bf16 from the retained mmap store
    var tq0 = perf_counter_ns()
    var store = H3TrainBlockStoreInt8.open(ckpt, N_BLOCKS, resident_blocks, ctx)
    print("[h3-train] direct-int8 base resident in",
          Float64(perf_counter_ns() - tq0) / 1.0e9, "s")

    # LoRA state: exact own-state resume, partial PEFT warm-resume, or fresh.
    var rng = _Rng(UInt64(seed))
    var states = List[_AdapterState]()
    var start_step = 0
    var state_path = out_dir + "/" + name + "_state.safetensors"
    var have_state = False
    try:
        var st_probe = SafeTensors.open(state_path)
        _ = st_probe.names()
        have_state = True
    except:
        have_state = False
    var warm_imported = 0
    var warm_initialized = 0
    if have_state and not baseline_only:
        var st = SafeTensors.open(state_path)
        var names = _param_state_names(target_preset)
        var ni = 0
        for b in range(N_BLOCKS):
            for s in range(SLOT_NAMES_LEN):
                var dims = _slot_out_in(s)
                if _slot_active(s, target_preset):
                    var t0 = _load_st(st, names[ni], ctx)
                    var t1 = _load_st(st, names[ni + 1], ctx)
                    states.append(_AdapterState(TArc(t0^), TArc(t1^)))
                    ni += 2
                else:
                    states.append(_zero_adapter_state(ctx))
        for b in range(N_REFINER_BLOCKS):
            for s in range(SLOT_NAMES_LEN):
                var dims = _slot_out_in(s)
                if _refiner_slot_active(s, target_preset):
                    var t0 = _load_st(st, names[ni], ctx)
                    var t1 = _load_st(st, names[ni + 1], ctx)
                    states.append(_AdapterState(TArc(t0^), TArc(t1^)))
                    ni += 2
                else:
                    states.append(_zero_adapter_state(ctx))
        var meta = _load_st(st, String("train_meta"), ctx).to_host(ctx)
        start_step = Int(meta[0])
        if len(meta) < 15:
            raise Error(
                "H3 trainer state predates the AdamW8bit exact-resume format; "
                "start a fresh output name"
            )
        if Int(meta[5]) != rank:
            raise Error("H3 trainer state rank does not match --dim")
        if Int(meta[6]) != timestep_buckets:
            raise Error("H3 trainer state timestep buckets changed on resume")
        if Int(meta[7]) != lr_warmup_steps:
            raise Error("H3 trainer state LR warmup changed on resume")
        if (
            meta[8] != guidance_scale
            or meta[9] != spatial_density_jitter
            or meta[10] != base_preservation_weight
            or meta[11] != max_grad_norm
            or meta[12] != lr
            or Int(meta[13]) != target_preset
            or meta[14] != base_preservation_probability
        ):
            raise Error("H3 trainer recipe changed on resume")
        rng.state = _decode_rng_state(meta, 1)
        print("[h3-train] RESUMED from step", start_step)
    elif train_cfg.resume_state != String("") and not baseline_only:
        var warm_st = SafeTensors.open(train_cfg.resume_state)
        for b in range(N_BLOCKS):
            for s in range(SLOT_NAMES_LEN):
                var dims = _slot_out_in(s)
                if _slot_active(s, target_preset):
                    var loaded = _warm_load_or_init_adapter(
                        warm_st, h3_lora_peft_prefix(b, s), rank,
                        dims[0], dims[1], rng, ctx,
                    )
                    if loaded.imported:
                        warm_imported += 1
                    else:
                        warm_initialized += 1
                    states.append(loaded^.into_state())
                else:
                    states.append(_zero_adapter_state(ctx))
        for b in range(N_REFINER_BLOCKS):
            for s in range(SLOT_NAMES_LEN):
                var dims = _slot_out_in(s)
                if _refiner_slot_active(s, target_preset):
                    var loaded = _warm_load_or_init_adapter(
                        warm_st, h3_lora_token_refiner_peft_prefix(b, s),
                        rank, dims[0], dims[1], rng, ctx,
                    )
                    if loaded.imported:
                        warm_imported += 1
                    else:
                        warm_initialized += 1
                    states.append(loaded^.into_state())
                else:
                    states.append(_zero_adapter_state(ctx))
        if warm_imported == 0:
            raise Error(
                "H3 warm resume imported zero canonical PEFT adapters; refusing"
            )
        start_step = train_cfg.start_step if train_cfg.start_step >= 0 else 0
        print("")
        print("  ============================================================")
        print("  [h3-resume] WARM PARTIAL INIT — optimizer moments ZEROED")
        print("  source:", train_cfg.resume_state)
        print("  imported trained adapters:", warm_imported)
        print("  initialized missing adapters:", warm_initialized)
        print("  New topology cannot reuse the old optimizer-state layout.")
        print("  This is a new optimizer trajectory starting at step", start_step)
        print("  ============================================================")
        print("")
    else:
        for _ in range(N_BLOCKS):
            for s in range(SLOT_NAMES_LEN):
                var dims = _slot_out_in(s)
                if _slot_active(s, target_preset):
                    states.append(_init_adapter_state(
                        rank, dims[0], dims[1], rng, ctx
                    ))
                else:
                    states.append(_zero_adapter_state(ctx))
        for _ in range(N_REFINER_BLOCKS):
            for s in range(SLOT_NAMES_LEN):
                var dims = _slot_out_in(s)
                if _refiner_slot_active(s, target_preset):
                    states.append(_init_adapter_state(
                        rank, dims[0], dims[1], rng, ctx
                    ))
                else:
                    states.append(_zero_adapter_state(ctx))
        print("[h3-train] fresh LoRA init (kaiming-uniform down, zero up)")

    if start_step >= max_steps:
        raise Error(
            String("H3 trainer start_step ") + String(start_step)
            + String(" >= max_steps ") + String(max_steps)
        )

    var optimizer_params = _optimizer_params(states, target_preset)
    var optimizer_maybe = Optional[Adam8bitDeviceState](None)
    if have_state and not baseline_only:
        var st = SafeTensors.open(state_path)
        var m_codes = _load_st_u8(st, String("adam8_m_codes"), ctx)
        var v_codes = _load_st_u8(st, String("adam8_v_codes"), ctx)
        var m_absmax = _load_st(st, String("adam8_m_absmax"), ctx)
        var v_absmax = _load_st(st, String("adam8_v_absmax"), ctx)
        optimizer_maybe = Optional[Adam8bitDeviceState](
            adamw8bit_device_state_from_tensors(
                optimizer_params,
                TArc(m_codes^), TArc(v_codes^),
                TArc(m_absmax^), TArc(v_absmax^), ctx,
            )
        )
    else:
        optimizer_maybe = Optional[Adam8bitDeviceState](
            adamw8bit_device_state(optimizer_params, ctx)
        )
    var optimizer = optimizer_maybe.value().copy()

    if warm_init_only:
        _save_all(
            states, optimizer, rank, start_step, rng.state,
            timestep_buckets, lr_warmup_steps, guidance_scale,
            spatial_density_jitter, base_preservation_weight,
            base_preservation_probability,
            max_grad_norm, lr, target_preset, out_dir, name, ctx,
        )
        print(
            "[h3-train] WARM INIT ONLY complete at step", start_step,
            "imported", warm_imported, "initialized", warm_initialized,
        )
        return

    # Data order is independently seeded per epoch so a mid-epoch resume
    # reconstructs the identical permutation without replaying prior steps.
    var order = List[Int]()
    for i in range(len(items)):
        order.append(i)
    var active_epoch = -1

    var loras = _compute_loras(states, rank, scale, target_preset, ctx)
    var refiner_loras = _compute_refiner_loras(
        states, rank, scale, target_preset, ctx
    )
    var base_loras = _base_loras()
    # Recompute-seed offload (h3_seed_offload): pinned-host slab instead of
    # 50 device-resident [S,D] seeds — measured 2.6GB at the AV bring-up
    # geometry. Grow-only: recreated when the packed length changes.
    var seed_offload = train_cfg.h3_seed_offload
    var seeds = H3SeedStore.create(N_BLOCKS, 4, 4, ctx)
    var run_t0 = perf_counter_ns()

    # The cadence supervisor invokes one config-defined segment at a time so
    # the worker exits and releases CUDA before validation sampling. No recipe
    # scalar is overridden: the next boundary comes only from sample_every.
    var loop_stop_step = max_steps
    if cadence_segment and start_step < max_steps:
        loop_stop_step = ((start_step // sample_every) + 1) * sample_every
        if loop_stop_step > max_steps:
            loop_stop_step = max_steps

    print("[h3-train] entering loop at step", start_step + 1, "/", max_steps,
          "segment_stop", loop_stop_step)
    for step in range(start_step + 1, loop_stop_step + 1):
        var t_step0 = perf_counter_ns()
        # ── item selection (reshuffle each epoch) ────────────────────────────
        var epoch_pos = (step - 1) % len(items)
        var epoch_index = (step - 1) // len(items)
        if epoch_index != active_epoch:
            _shuffle_data_order(order, seed, epoch_index)
            active_epoch = epoch_index
        var it = items[order[epoch_pos]].copy()
        # Musubi's sparse base-preservation branch: draw once per item and
        # inverse-probability scale active losses so the expected preservation
        # gradient retains the configured weight.
        var preservation_active = (
            base_preservation_weight > Float32(0.0)
            and rng.uniform() < Float64(base_preservation_probability)
        )

        var lat = h3_read_latent_cache(it.latent_path, ctx)
        var is_av = lat.has_audio or lat.lat_f != 1
        if not lat.has_video:
            raise Error("H3 trainer requires video latents: " + it.item_key)
        if is_av and guidance_scale > Float32(0.0):
            raise Error(
                "H3 AV arm: guidance distillation is image-only today; set "
                "guidance_scale 0 for AV items"
            )
        var te = h3_read_text_cache(it.te_path, ctx)
        # upstream trains in dit_dtype bf16: cast cached latents down exactly
        # like its .to(dtype) load (caches may be stored F32)
        var x0 = lat.video[].clone(ctx)
        if x0.dtype() != STDtype.BF16:
            x0 = cast_tensor(x0, STDtype.BF16, ctx)

        # ── sigma, quantized to the grid. Image keeps the resolution-aware
        # bucketed draw; AV uses the released shared-u recipe (u ~ U[0,1),
        # per-modality shift 12/3 — h3_train_sigma, musubi parity). ───────────
        var u_av = Float64(0.0)
        var sigma_raw: Float64
        if is_av:
            u_av = rng.uniform()
            sigma_raw = Float64(h3_shift_sigma(Float32(u_av), Float32(12.0)))
        else:
            sigma_raw = _bucketed_image_sigma(
                rng, seed, step, timestep_buckets, lat.lat_h, lat.lat_w
            )
        var node = Int(sigma_raw * Float64(SIGMA_NODES - 1) + 0.5)
        if node < 0:
            node = 0
        if node > SIGMA_NODES - 1:
            node = SIGMA_NODES - 1
        var sigma = Float32(node) / Float32(SIGMA_NODES - 1)
        var t_v = Float32(1.0) - sigma
        var audio_t = 0
        if lat.has_audio:
            audio_t = lat.audio_t
        var sigma_a = Float32(0.0)
        var t_a = Float32(1.0)
        var node_a = 0
        if is_av and lat.has_audio:
            var sa_raw = Float64(h3_shift_sigma(Float32(u_av), Float32(3.0)))
            node_a = Int(sa_raw * Float64(SIGMA_NODES - 1) + 0.5)
            if node_a < 0:
                node_a = 0
            if node_a > SIGMA_NODES - 1:
                node_a = SIGMA_NODES - 1
            sigma_a = Float32(node_a) / Float32(SIGMA_NODES - 1)
            t_a = Float32(1.0) - sigma_a

        # ── noise + target (gated bit-exact vs torch) ────────────────────────
        var vsh: List[Int] = [24, lat.lat_f, lat.lat_h, lat.lat_w]
        var noise = randn(vsh^, UInt64(seed * 1000003 + step), STDtype.BF16, ctx)
        var x_t = h3_noisy_input(x0, noise, sigma, ctx)
        var target = h3_velocity_target(x0, noise, ctx)
        # audio: separate noise draw (distinct deterministic stream), the
        # gated dual-sigma path.
        var xt_audio = Optional[Tensor](None)
        var target_audio = Optional[Tensor](None)
        if is_av and lat.has_audio:
            var a0 = lat.audio[].clone(ctx)
            if a0.dtype() != STDtype.BF16:
                a0 = cast_tensor(a0, STDtype.BF16, ctx)
            var ash: List[Int] = [2, 32, audio_t]
            var noise_a = randn(
                ash^, UInt64(seed * 1000003 + step + 500009), STDtype.BF16, ctx
            )
            xt_audio = Optional[Tensor](h3_noisy_input(a0, noise_a, sigma_a, ctx))
            target_audio = Optional[Tensor](h3_velocity_target(a0, noise_a, ctx))
            _ = a0^
        # the frontend's patch projection is an fp32-trap path (checkpoint F32
        # weights): feed F32 rows (exact bf16 upcast), it rne-casts back down.
        var x_rows = cast_tensor(
            minimax_h3_video_patchify(x_t, ctx), STDtype.F32, ctx
        )
        var target_rows = minimax_h3_video_patchify(target, ctx)

        # ── packed layout + rope + frontend (frozen, gated inference code) ───
        var density_scale = _density_scale(rng, spatial_density_jitter)
        var anchors = List[Int]()
        if is_av and lat.has_keyframe_rows and te.task_id == 2:
            anchors.append(MINIMAX_H3_ANCHOR_FIRST)
            anchors.append(MINIMAX_H3_ANCHOR_LAST)
        var layout = minimax_h3_build_packed_sequence(
            te.tags, lat.lat_f, lat.lat_h, lat.lat_w, audio_t, 2, 2, anchors,
            density_scale,
        )
        var S = layout.sequence_length
        var row_ts = minimax_h3_build_row_timesteps(
            layout, t_v, t_a, Float32(0.999), Float32(1.0),
        )
        var positions_f32 = List[Float32](capacity=len(layout.position_ids))
        for i in range(len(layout.position_ids)):
            positions_f32.append(Float32(layout.position_ids[i]))
        var pos_sh: List[Int] = [S * 3]
        var pos_t = Tensor.from_host(positions_f32, pos_sh^, STDtype.F32, ctx)
        var rope = build_minimax_h3_rope_tables(pos_t, ctx, config.rope_inv_freq_len)
        var rotary_dim = rope[0].shape()[1]

        # image t2va layout is [text | video] contiguous (no cond/audio rows):
        # compose the frozen frontend directly and concat the two streams.
        var video_embeds = _minimax_h3_video_patch_embed_bf16(x_rows, step_w, ctx)
        var item_i = order[epoch_pos]
        var tesh: List[Int] = [text_embeds_rows[item_i], H3_D]
        var student_refiner_fwd = Optional[H3TokenRefinerTrainForward](None)
        var text_embeds: Tensor
        if target_preset == H3_TARGET_FULL_208:
            var text_condition = Tensor.from_host_bf16(
                text_condition_host[item_i], tesh.copy(), ctx
            )
            var trf = h3_token_refiner_train_forward[
                H3_HEADS, H3_HEAD_DIM
            ](
                text_condition, refiner_weights.value(), refiner_loras,
                H3_D, H3_F, config.norm_eps, config.qk_norm_eps,
                config.final_norm_eps, ctx,
            )
            text_embeds = trf.out[].clone(ctx)
            student_refiner_fwd = Optional[H3TokenRefinerTrainForward](trf^)
        else:
            text_embeds = Tensor.from_host_bf16(
                text_embeds_host[item_i], tesh.copy(), ctx
            )
        var hidden0: Tensor
        if is_av:
            var media = video_embeds.clone(ctx)
            if lat.has_audio:
                var a_rows = cast_tensor(
                    h3_audio_latents_to_rows(xt_audio.value(), ctx),
                    STDtype.F32, ctx,
                )
                var a_emb = _minimax_h3_audio_patch_embed_bf16(a_rows, step_w, ctx)
                media = concat(0, ctx, a_emb, media)
            if len(anchors) > 0:
                var kf_rows = cast_tensor(
                    lat.keyframe_rows[].clone(ctx), STDtype.F32, ctx
                )
                var kf_emb = _minimax_h3_video_patch_embed_bf16(kf_rows, step_w, ctx)
                media = concat(0, ctx, kf_emb, media)
            hidden0 = concat(0, ctx, text_embeds, media)
        else:
            hidden0 = concat(0, ctx, text_embeds, video_embeds)
        if hidden0.shape()[0] != S:
            raise Error("packed length mismatch (text+media != S)")

        # ── per-step mod tables: fetch ONLY this node's rows (host grid) ─────
        var ts_nodes = List[Int]()
        var node_idx = List[Int]()
        if not is_av:
            ts_nodes.append(node)
            for _ in range(S):
                node_idx.append(0)  # fetched tables carry just this node's rows
        else:
            # one grid node per DISTINCT row timestep, in row_ts order; the
            # fetched tables below concatenate one [3,6D] chunk per node.
            var node_c = Int(Float64(0.001) * Float64(SIGMA_NODES - 1) + 0.5)
            for k in range(len(row_ts.values)):
                var v = row_ts.values[k]
                if v == t_v:
                    ts_nodes.append(node)
                elif v == t_a:
                    ts_nodes.append(node_a)
                elif v == Float32(0.999):
                    ts_nodes.append(node_c)
                else:
                    raise Error("H3 AV arm: unrecognized row timestep value")
            node_idx = row_ts.indices.copy()
        var adaln_idx = minimax_h3_adaln_indices(node_idx, layout.token_tags)
        var final_mod_rows = modgrid.final_row(ts_nodes[0], ctx)
        for k in range(1, len(ts_nodes)):
            final_mod_rows = concat(
                0, ctx, final_mod_rows, modgrid.final_row(ts_nodes[k], ctx)
            )

        # ── 50-block streamed LoRA forward ───────────────────────────────────
        var mods = List[TArc]()
        for b in range(N_BLOCKS):
            var tbl = modgrid.block_rows(b, ts_nodes[0], ctx)
            for k in range(1, len(ts_nodes)):
                tbl = concat(0, ctx, tbl, modgrid.block_rows(b, ts_nodes[k], ctx))
            mods.append(TArc(tbl^))
        ctx.synchronize()
        var tp0 = perf_counter_ns()

        # ── guidance teacher: EMPTY-cond forward, keep only its video rows ──
        # Runs BEFORE the student pass so its retained activations free first
        # (peak stays one graph deep). Same node's mod tables / final rows.
        var use_guidance = guidance_scale > Float32(0.0)
        var g_empty = Optional[Tensor](None)
        if use_guidance:
            var anchors_e = List[Int]()
            var layout_e = minimax_h3_build_packed_sequence(
                empty_tags, 1, lat.lat_h, lat.lat_w, 0, 2, 2, anchors_e,
                density_scale,
            )
            var S_e = layout_e.sequence_length
            var pos_e_f32 = List[Float32](capacity=len(layout_e.position_ids))
            for i in range(len(layout_e.position_ids)):
                pos_e_f32.append(Float32(layout_e.position_ids[i]))
            var pos_e_sh: List[Int] = [S_e * 3]
            var pos_e = Tensor.from_host(pos_e_f32, pos_e_sh^, STDtype.F32, ctx)
            var rope_e = build_minimax_h3_rope_tables(
                pos_e, ctx, config.rope_inv_freq_len
            )
            var tesh_e: List[Int] = [empty_embeds_rows, H3_D]
            var e_text: Tensor
            if target_preset == H3_TARGET_FULL_208:
                var e_condition = Tensor.from_host_bf16(
                    empty_condition_host, tesh_e.copy(), ctx
                )
                var e_trf = h3_token_refiner_train_forward[
                    H3_HEADS, H3_HEAD_DIM
                ](
                    e_condition, refiner_weights.value(), refiner_loras,
                    H3_D, H3_F, config.norm_eps, config.qk_norm_eps,
                    config.final_norm_eps, ctx,
                )
                e_text = e_trf.out[].clone(ctx)
                _ = e_trf^
            else:
                e_text = Tensor.from_host_bf16(
                    empty_embeds_host, tesh_e.copy(), ctx
                )
            var hidden0_e = concat(0, ctx, e_text, video_embeds)
            if hidden0_e.shape()[0] != S_e:
                raise Error("guidance: packed length mismatch (empty+video != S_e)")
            var node_idx_e = List[Int]()
            for _ in range(S_e):
                node_idx_e.append(0)
            var adaln_idx_e = minimax_h3_adaln_indices(
                node_idx_e, layout_e.token_tags
            )
            var fwd_e = h3_stack_train_forward_streamed_int8[H3_HEADS, H3_HEAD_DIM](
                hidden0_e, store, loras, mods, adaln_idx_e,
                rope_e[0], rope_e[1], H3_D, H3_F, rope_e[0].shape()[1], H3_EPS, ctx,
            )
            var empty_idx_e = List[Int]()
            var ffwd_e = h3_final_train_forward(
                fwd_e.out[], final_w, final_mod_rows, node_idx_e,
                layout_e.video_indices, empty_idx_e, H3_EPS, ctx,
            )
            g_empty = Optional[Tensor](ffwd_e.video[].clone(ctx))
            ctx.synchronize()
            _ = ffwd_e^
            _ = fwd_e^

        # Frozen-base preservation teacher: same prompt/noise/geometry with
        # every adapter absent. It is evaluated before the graph-carrying
        # student pass so only one 50-block activation chain is live at once.
        var base_prediction = Optional[Tensor](None)
        if preservation_active:
            var base_text = Tensor.from_host_bf16(
                text_embeds_host[item_i], tesh.copy(), ctx
            )
            var text_rows_n = text_embeds.shape()[0]
            var media_part = slice(hidden0, 0, text_rows_n, S - text_rows_n, ctx)
            var hidden0_base = concat(0, ctx, base_text, media_part)
            var fwd_base = h3_stack_train_forward_streamed_int8[
                H3_HEADS, H3_HEAD_DIM
            ](
                hidden0_base, store, base_loras, mods, adaln_idx,
                rope[0], rope[1], H3_D, H3_F, rotary_dim, H3_EPS, ctx,
            )
            var empty_base_idx = List[Int]()
            var ffwd_base = h3_final_train_forward(
                fwd_base.out[], final_w, final_mod_rows, node_idx,
                layout.video_indices, empty_base_idx, H3_EPS, ctx,
            )
            base_prediction = Optional[Tensor](ffwd_base.video[].clone(ctx))
            ctx.synchronize()
            _ = ffwd_base^
            _ = fwd_base^

        var fwd: H3StackTrainForward
        if seed_offload:
            if seeds.s_len != S or seeds.d_model != H3_D:
                seeds = H3SeedStore.create(N_BLOCKS, S, H3_D, ctx)
            var fo = h3_stack_train_forward_streamed_int8_seedoff[
                H3_HEADS, H3_HEAD_DIM
            ](
                hidden0, store, seeds, loras, mods, adaln_idx,
                rope[0], rope[1], H3_D, H3_F, rotary_dim, H3_EPS, ctx,
            )
            fwd = H3StackTrainForward(fo.out.copy(), List[TArc]())
            _ = fo^
        else:
            fwd = h3_stack_train_forward_streamed_int8[H3_HEADS, H3_HEAD_DIM](
                hidden0, store, loras, mods, adaln_idx,
                rope[0], rope[1], H3_D, H3_F, rotary_dim, H3_EPS, ctx,
            )

        # ── final layer twin ─────────────────────────────────────────────────
        ctx.synchronize()
        var tp1 = perf_counter_ns()
        var empty_idx = List[Int]()
        var audio_out_idx = List[Int]()
        if is_av and lat.has_audio:
            audio_out_idx = layout.audio_indices.copy()
        var ffwd = h3_final_train_forward(
            fwd.out[], final_w, final_mod_rows, node_idx,
            layout.video_indices, audio_out_idx, H3_EPS, ctx,
        )

        # ── loss + d_pred (gated bit-exact vs torch) ─────────────────────────
        # With guidance: pred = c_hat = g/s + g_empty*(s-1)/s (F32 combine,
        # bf16 result like upstream's model-dtype arithmetic); chain rule
        # scales d_g by 1/s after the plain loss grad.
        # upstream-default "sigma" schedule: effective scale tapers to 1 at
        # low noise so identity-detail steps train at full gradient strength
        # (flat scale divides d_g by s exactly where detail is learned).
        var s_eff = Float32(1.0) + (guidance_scale - Float32(1.0)) * sigma
        var pred_rows: Tensor
        if use_guidance:
            var g32 = cast_tensor(ffwd.video[], STDtype.F32, ctx)
            var e32 = cast_tensor(g_empty.value(), STDtype.F32, ctx)
            var chat = add(
                mul_scalar(g32, Float32(1.0) / s_eff, ctx),
                mul_scalar(e32, (s_eff - Float32(1.0)) / s_eff, ctx),
                ctx,
            )
            pred_rows = cast_tensor(chat, STDtype.BF16, ctx)
        else:
            pred_rows = ffwd.video[].clone(ctx)
        var empty_mask = List[Bool]()
        var n_cond_v = layout.num_condition_video_rows
        var pred_target_rows: Tensor
        if n_cond_v == 0:
            pred_target_rows = pred_rows.clone(ctx)
        else:
            pred_target_rows = slice(
                pred_rows, 0, n_cond_v, pred_rows.shape()[0] - n_cond_v, ctx
            )
        var ml = h3_modality_loss(pred_target_rows, target_rows, empty_mask, ctx)
        var ml_a = H3ModalityLoss(Float64(0.0), 0)
        var pred_audio_lat = Optional[Tensor](None)
        if is_av and lat.has_audio:
            pred_audio_lat = Optional[Tensor](
                h3_audio_rows_to_latents(ffwd.audio[], audio_t, ctx)
            )
            ml_a = h3_modality_loss(
                pred_audio_lat.value(), target_audio.value(),
                lat.audio_loss_mask, ctx,
            )
        var loss = h3_joint_token_loss(ml, ml_a, Float64(1.0), Float64(1.0))
        var preservation_loss = Float64(0.0)
        if base_prediction:
            var pl = h3_modality_loss(
                ffwd.video[], base_prediction.value(), empty_mask, ctx
            )
            preservation_loss = pl.total / Float64(pl.elements)
            loss += (
                Float64(base_preservation_weight)
                / Float64(base_preservation_probability)
            ) * preservation_loss

        # Dataset-level untrained-loss baseline. Fresh LoRA B matrices are
        # exactly zero, so this is the corrected frozen base function. Stop at
        # the measured loss: backward/optimizer work cannot change the result
        # and would double the time needed for the 200-sample characterization.
        if baseline_only:
            ctx.synchronize()
            var tp_baseline = perf_counter_ns()
            print("[phase] fwd", Float64(tp1 - tp0) / 1.0e9,
                  "final+loss", Float64(tp_baseline - tp1) / 1.0e9,
                  "bwd", 0.0, "opt", 0.0,
                  "prep", Float64(tp0 - t_step0) / 1.0e9)
            var dt_baseline = Float64(tp_baseline - t_step0) / 1.0e9
            print("[h3-train] step", step, "loss", loss, "sigma", sigma,
                  "S", S, "item", it.item_key, "dt", dt_baseline, "s")
            continue

        var none_mask = Optional[Tensor](None)
        var denom_all = Float64(ml.elements) + Float64(ml_a.elements)
        var d_video = h3_loss_grad(
            pred_target_rows, target_rows, none_mask^, 1.0, denom_all, ctx,
        )
        if n_cond_v > 0:
            var zsh: List[Int] = [n_cond_v, d_video.shape()[1]]
            var zeros_c = full_device(zsh^, Float32(0.0), STDtype.BF16, ctx)
            d_video = concat(0, ctx, zeros_c, d_video)
        if use_guidance:
            d_video = mul_scalar_bf16out(
                cast_tensor(d_video, STDtype.F32, ctx),
                Float32(1.0) / s_eff, ctx,
            )
        if base_prediction:
            var d_preserve = h3_loss_grad(
                ffwd.video[], base_prediction.value(), none_mask^,
                Float64(base_preservation_weight)
                / Float64(base_preservation_probability),
                Float64(ml.elements), ctx,
            )
            d_video = add(d_video, d_preserve, ctx)

        # ── backward chain ───────────────────────────────────────────────────
        var d_audio: Tensor
        if is_av and lat.has_audio:
            var amask = h3_audio_mask_tensor(lat.audio_loss_mask, 32, ctx)
            var amask_opt = Optional[Tensor](amask^)
            var d_a_lat = h3_loss_grad(
                pred_audio_lat.value(), target_audio.value(), amask_opt^,
                1.0, denom_all, ctx,
            )
            d_audio = h3_audio_latents_to_rows(d_a_lat, ctx)
        else:
            var d_audio_sh: List[Int] = [1, 1]
            d_audio = full_device(d_audio_sh^, Float32(0.0), STDtype.BF16, ctx)
        var d_hidden = h3_final_train_backward(
            d_video, d_audio, ffwd.saved, final_w, node_idx,
            layout.video_indices, audio_out_idx, S, H3_EPS, ctx,
        )
        ctx.synchronize()
        var tp2 = perf_counter_ns()
        var grads: H3StackLoraOnlyGrads
        if seed_offload:
            grads = h3_stack_train_backward_streamed_int8_seedoff[
                H3_HEADS, H3_HEAD_DIM
            ](
                d_hidden, store, seeds, loras, mods, adaln_idx,
                rope[0], rope[1], H3_D, H3_F, rotary_dim, H3_EPS, ctx,
            )
        else:
            grads = h3_stack_train_backward_streamed_int8[H3_HEADS, H3_HEAD_DIM](
                d_hidden, fwd, store, loras, mods, adaln_idx,
                rope[0], rope[1], H3_D, H3_F, rotary_dim, H3_EPS, ctx,
            )
        var refiner_grad_groups = List[H3BlockLoraGrads]()
        if target_preset == H3_TARGET_FULL_208:
            if not student_refiner_fwd:
                raise Error("H3 full-208 step lost token-refiner forward state")
            var d_text = slice(
                grads.d_x[], 0, 0, text_embeds_rows[item_i], ctx
            )
            var trg = h3_token_refiner_train_backward[
                H3_HEADS, H3_HEAD_DIM
            ](
                d_text, student_refiner_fwd.value(),
                refiner_weights.value(), refiner_loras,
                H3_D, H3_F, config.norm_eps, config.qk_norm_eps,
                config.final_norm_eps, ctx,
            )
            for b in range(N_REFINER_BLOCKS):
                refiner_grad_groups.append(trg.lora[b].copy())

        ctx.synchronize()
        var tp3 = perf_counter_ns()
        # ── bnb-parity device AdamW8bit on the F32 masters ──────────────────
        var gts = List[TArc]()
        for b in range(N_BLOCKS):
            for s in range(SLOT_NAMES_LEN):
                if not _slot_active(s, target_preset):
                    continue
                var g = _grad_pair(grads.lora[b], s)
                gts.append(g[0])
                gts.append(g[1])
        for b in range(N_REFINER_BLOCKS):
            for s in range(SLOT_NAMES_LEN):
                if not _refiner_slot_active(s, target_preset):
                    continue
                var g = _grad_pair(refiner_grad_groups[b], s)
                gts.append(g[0])
                gts.append(g[1])
        var grad_stats = on_device_grad_stats(gts, ctx)
        if grad_stats.nonfinite_count != 0:
            raise Error(
                "H3 trainer found non-finite gradients: "
                + String(grad_stats.nonfinite_count)
            )
        var clip_scale = Float32(1.0)
        if grad_stats.grad_norm > max_grad_norm:
            clip_scale = max_grad_norm / grad_stats.grad_norm
        var step_lr = lr
        if lr_warmup_steps > 0 and step <= lr_warmup_steps:
            step_lr = lr * Float32(step) / Float32(lr_warmup_steps)
        adamw8bit_device_step(
            optimizer_params, gts, optimizer, step, step_lr,
            optimizer_beta1, optimizer_beta2, optimizer_eps,
            optimizer_weight_decay, ctx,
            clip_scale,
        )
        loras = _compute_loras(states, rank, scale, target_preset, ctx)
        refiner_loras = _compute_refiner_loras(
            states, rank, scale, target_preset, ctx
        )
        ctx.synchronize()

        var tp4 = perf_counter_ns()
        print("[phase] fwd", Float64(tp1 - tp0) / 1.0e9,
              "final+loss", Float64(tp2 - tp1) / 1.0e9,
              "bwd", Float64(tp3 - tp2) / 1.0e9,
              "opt", Float64(tp4 - tp3) / 1.0e9,
              "prep", Float64(tp0 - t_step0) / 1.0e9)
        var dt = Float64(perf_counter_ns() - t_step0) / 1.0e9
        print("[h3-train] step", step, "loss", loss, "sigma", sigma,
              "preserve", preservation_loss,
              "preserve_active", preservation_active,
              "density", density_scale,
              "lr", step_lr, "S", S, "item", it.item_key, "dt", dt, "s")
        print_trainer_progress(
            String("H3-lora"), step, max_steps, len(items), Float32(loss),
            Float64(grad_stats.grad_norm), dt, 0.0,
            Float64(perf_counter_ns() - run_t0) / 1.0e9,
        )

        # ── save / sample snapshots ──────────────────────────────────────────
        if not baseline_only and (
            step % save_every == 0 or step == loop_stop_step or
            step % sample_every == 0
        ):
            _save_all(
                states, optimizer, rank, step, rng.state,
                timestep_buckets, lr_warmup_steps, guidance_scale,
                spatial_density_jitter, base_preservation_weight,
                base_preservation_probability,
                max_grad_norm, lr, target_preset, out_dir, name, ctx,
            )

    print("[h3-train] DONE at step", loop_stop_step, "/", max_steps)


def _save_all(
    states: List[_AdapterState], optimizer: Adam8bitDeviceState, rank: Int,
    step: Int, rng_state: UInt64,
    timestep_buckets: Int, lr_warmup_steps: Int, guidance_scale: Float32,
    spatial_density_jitter: Float32, base_preservation_weight: Float32,
    base_preservation_probability: Float32,
    max_grad_norm: Float32, lr: Float32, target_preset: Int,
    out_dir: String, name: String,
    ctx: DeviceContext,
) raises:
    # 1) External LoRA: same canonical BF16 PEFT contract as LTX and the rest
    # of the stack. Download F32 masters and pack on the host so save cadence
    # never allocates a second ~150 MiB adapter set in scarce H3 VRAM.
    var adapters = List[F32NamedLora]()
    for b in range(N_BLOCKS):
        for s in range(SLOT_NAMES_LEN):
            if not _slot_active(s, target_preset):
                continue
            var st_i = b * SLOT_NAMES_LEN + s
            var dims = _slot_out_in(s)
            var a_host = states[st_i].a_m[].to_host(ctx)
            var b_host = states[st_i].b_m[].to_host(ctx)
            adapters.append(F32NamedLora(
                h3_lora_peft_prefix(b, s),
                a_host^,
                b_host^,
                rank,
                dims[1],
                dims[0],
            ))
    for b in range(N_REFINER_BLOCKS):
        for s in range(SLOT_NAMES_LEN):
            if not _refiner_slot_active(s, target_preset):
                continue
            var st_i = H3_MAIN_STATE_COUNT + b * SLOT_NAMES_LEN + s
            var dims = _slot_out_in(s)
            var a_host = states[st_i].a_m[].to_host(ctx)
            var b_host = states[st_i].b_m[].to_host(ctx)
            adapters.append(F32NamedLora(
                h3_lora_token_refiner_peft_prefix(b, s),
                a_host^,
                b_host^,
                rank,
                dims[1],
                dims[0],
            ))
    var lora_path = (
        out_dir + "/" + name + "_step" + String(step) + ".safetensors"
    )
    var n_pairs = save_lora_peft_host_f32(adapters, lora_path)
    print("[h3-train] saved PEFT LoRA:", lora_path, "pairs", n_pairs)

    # 2) resume state (masters + quantized bnb moments + exact RNG state)
    var snames = _param_state_names(target_preset)
    var stensors = List[TArc]()
    for b in range(N_BLOCKS):
        for s in range(SLOT_NAMES_LEN):
            if not _slot_active(s, target_preset):
                continue
            var i = b * SLOT_NAMES_LEN + s
            stensors.append(states[i].a_m)
            stensors.append(states[i].b_m)
    for b in range(N_REFINER_BLOCKS):
        for s in range(SLOT_NAMES_LEN):
            if not _refiner_slot_active(s, target_preset):
                continue
            var i = H3_MAIN_STATE_COUNT + b * SLOT_NAMES_LEN + s
            stensors.append(states[i].a_m)
            stensors.append(states[i].b_m)
    snames.append(String("adam8_m_codes"))
    stensors.append(optimizer.m_codes)
    snames.append(String("adam8_v_codes"))
    stensors.append(optimizer.v_codes)
    snames.append(String("adam8_m_absmax"))
    stensors.append(optimizer.m_absmax)
    snames.append(String("adam8_v_absmax"))
    stensors.append(optimizer.v_absmax)
    snames.append(String("train_meta"))
    var rng_words = _encode_rng_state(rng_state)
    var meta: List[Float32] = [
        Float32(step), rng_words[0], rng_words[1], rng_words[2], rng_words[3],
        Float32(rank), Float32(timestep_buckets), Float32(lr_warmup_steps),
        guidance_scale, spatial_density_jitter, base_preservation_weight,
        max_grad_norm, lr, Float32(target_preset), base_preservation_probability,
    ]
    var msh: List[Int] = [len(meta)]
    stensors.append(TArc(Tensor.from_host(meta, msh^, STDtype.F32, ctx)))
    var state_path = out_dir + "/" + name + "_state.safetensors"
    save_safetensors(snames, stensors, state_path, ctx)
    print("[h3-train] saved state:", state_path)
