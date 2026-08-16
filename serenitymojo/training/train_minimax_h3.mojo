# train_minimax_h3.mojo — MiniMax-H3 FL2VA LoRA trainer, IMAGE-mode maiden arm.
#
# Oracle = torchref (pinned upstream checkout @ 04324c28). Recipe = the
# upstream OFFICIAL image-concept recipe (docs/minimax_h3.md + examples/
# minimax_h3/image_fl2va.toml + parser set_defaults):
#   * caches: torchref-native mmh3 pair (h3_train_cache.mojo reader, gated
#     bit-exact); text task t2va (id 0); image items ([24,1,H,W], no audio,
#     no keyframe rows packed for plain image training).
#   * LoRA: dim 16 / alpha 16 (scale 1.0) on the 4 block Linears
#     (qkv/out/fc1/fc2 — the full target set minus frozen adaln/norms);
#     down = kaiming_uniform(a=sqrt(5)) = U(+-1/sqrt(in)), up = zeros;
#     F32 masters + AdamW(lr 1e-4, betas 0.9/0.999, eps 1e-8, wd 1e-2);
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
# bwd -> fused AdamW on masters -> refresh bf16 copies. Save LoRA
# (lora_unet_* torchref key format, F32) every --save_every + full resume
# state (masters + moments + step); auto-resume when the state file exists.
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
    full_device, concat, add, mul_scalar, mul_scalar_bf16out,
)
from serenitymojo.models.klein.lora_block import LoraAdapterDevice
from serenitymojo.models.dit.minimax_h3_dit import (
    MiniMaxH3DiTConfig, minimax_h3_released_config,
)
from serenitymojo.models.dit.minimax_h3_frontend import (
    minimax_h3_video_patchify, minimax_h3_timestep_embedding,
    minimax_h3_condition_embed, minimax_h3_token_refiner_dynamic,
    _minimax_h3_video_patch_embed_bf16,
)
from serenitymojo.models.dit.minimax_h3_rope import build_minimax_h3_rope_tables
from serenitymojo.models.minimax_h3.packing import (
    minimax_h3_build_packed_sequence, minimax_h3_build_row_timesteps,
)
from serenitymojo.models.minimax_h3.dit_frontend import minimax_h3_adaln_indices
from serenitymojo.models.minimax_h3.h3_train_cache import (
    H3CacheItemPaths, h3_discover_cache_items,
    h3_read_latent_cache, h3_read_text_cache,
)
from serenitymojo.models.minimax_h3.h3_train_sigma import (
    h3_noisy_input, h3_velocity_target,
    h3_modality_loss, h3_joint_token_loss, h3_loss_grad,
)
from serenitymojo.models.minimax_h3.h3_train_block_store_fp8 import (
    H3TrainBlockStoreFp8,
)
from serenitymojo.models.minimax_h3.h3_train_modgrid import H3TrainModGrid
from serenitymojo.models.minimax_h3.h3_block_train import (
    H3BlockLoraDevice, H3BlockLoraGrads,
)
from serenitymojo.models.minimax_h3.h3_stack_train import (
    h3_stack_train_forward_streamed_fp8, h3_stack_train_backward_streamed_fp8,
)
from serenitymojo.models.minimax_h3.h3_final_train import (
    H3FinalTrainWeights, h3_final_train_forward, h3_final_train_backward,
)
from serenitymojo.training.fused_adamw_multitensor import fused_adamw_step
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

comptime SLOT_NAMES_LEN = 4


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


def _slot_key(slot: Int) raises -> String:
    if slot == 0:
        return String("attn_qkv_proj")
    if slot == 1:
        return String("attn_out_proj")
    if slot == 2:
        return String("mlp_fc1")
    if slot == 3:
        return String("mlp_fc2")
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


# ── LoRA state: masters + moments (F32 device) + compute copies (bf16) ───────
struct _AdapterState(Movable):
    var a_m: TArc  # [rank, in] F32 master
    var b_m: TArc  # [out, rank] F32 master
    var m_a: TArc
    var v_a: TArc
    var m_b: TArc
    var v_b: TArc

    def __init__(
        out self, var a_m: TArc, var b_m: TArc,
        var m_a: TArc, var v_a: TArc, var m_b: TArc, var v_b: TArc,
    ):
        self.a_m = a_m^
        self.b_m = b_m^
        self.m_a = m_a^
        self.v_a = v_a^
        self.m_b = m_b^
        self.v_b = v_b^


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
    var zsh_a: List[Int] = [rank, in_f]
    var zsh_b: List[Int] = [out_f, rank]
    return _AdapterState(
        TArc(a_m^), TArc(b_m^),
        TArc(full_device(zsh_a.copy(), Float32(0.0), STDtype.F32, ctx)),
        TArc(full_device(zsh_a.copy(), Float32(0.0), STDtype.F32, ctx)),
        TArc(full_device(zsh_b.copy(), Float32(0.0), STDtype.F32, ctx)),
        TArc(full_device(zsh_b.copy(), Float32(0.0), STDtype.F32, ctx)),
    )


def _compute_loras(
    states: List[_AdapterState], rank: Int, scale: Float32, ctx: DeviceContext
) raises -> List[H3BlockLoraDevice]:
    """bf16 compute copies of the masters, packed per block."""
    var loras = List[H3BlockLoraDevice]()
    for b in range(N_BLOCKS):
        var slots = List[Optional[LoraAdapterDevice]]()
        for s in range(SLOT_NAMES_LEN):
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


def _load_st(st: SafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    var info = st.tensor_info(name)
    return Tensor.from_view(
        from_parts(info.dtype, info.shape.copy(), st.tensor_bytes(name)), ctx
    )


def _state_names() raises -> List[String]:
    var names = List[String]()
    for b in range(N_BLOCKS):
        for s in range(SLOT_NAMES_LEN):
            var base = String("b") + String(b) + "_" + _slot_key(s)
            names.append(base + "_a_p")
            names.append(base + "_b_p")
            names.append(base + "_a_m")
            names.append(base + "_a_v")
            names.append(base + "_b_m")
            names.append(base + "_b_v")
    return names^


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
    var ctx = DeviceContext()
    var cache_dir = _arg(String("cache_dir"), String(""))
    var ckpt = _arg(
        String("ckpt"),
        String("/home/alex/.serenity/models/checkpoints/MiniMax-H3/FL2VA/transformer"),
    )
    var out_dir = _arg(String("out_dir"), String("/output/h3_lora"))
    var name = _arg(String("name"), String("h3_lora"))
    var max_steps = _arg_int(String("steps"), 3000)
    var lr = _arg_f32(String("lr"), Float32(1.0e-4))
    var rank = _arg_int(String("dim"), 16)
    var alpha = _arg_f32(String("alpha"), Float32(16.0))
    var seed = _arg_int(String("seed"), 42)
    var save_every = _arg_int(String("save_every"), 250)
    var sample_every = _arg_int(String("sample_every"), 1000)
    if cache_dir == String(""):
        raise Error("--cache_dir is required")
    var scale = alpha / Float32(rank)

    print("[h3-train] cache_dir:", cache_dir)
    print("[h3-train] recipe: dim", rank, "alpha", alpha, "lr", lr,
          "steps", max_steps, "seed", seed)

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

    # Refined text embeds depend only on the cached captions — precompute
    # once per item (condition_proj + token refiner), park on HOST, then
    # drop those weights from VRAM (~0.9GB) and skip their per-step compute.
    print("[h3-train] precomputing per-item text embeds...")
    var text_embeds_host = List[List[BFloat16]]()
    var text_embeds_rows = List[Int]()
    for i in range(len(items)):
        var tei = h3_read_text_cache(items[i].te_path, ctx)
        var th = tei.hidden[].clone(ctx)
        if th.dtype() != STDtype.BF16:
            th = cast_tensor(th, STDtype.BF16, ctx)
        var t0e = minimax_h3_condition_embed(th, frontend_w, ctx)
        var te_ref = minimax_h3_token_refiner_dynamic[H3_HEADS, H3_HEAD_DIM](
            t0e, frontend_w, config, ctx
        )
        text_embeds_host.append(te_ref.to_host_bf16(ctx))
        text_embeds_rows.append(te_ref.shape()[0])
        ctx.synchronize()
    # Guidance-consistent objective (CFG-distilled base): teacher forward on
    # EMPTY conditioning per step, c_hat = (g + (s-1)*g_empty)/s, loss vs
    # c_hat with d_g scaled by 1/s. The released H3 is guidance-distilled;
    # the plain velocity loss cannot bind identity onto it (maiden 2000-step
    # run: loss fell, zero likeness). Requires empty-cond TE cache pairs.
    var guidance_scale = _arg_f32(String("guidance_scale"), Float32(0.0))
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
        var e_ref = minimax_h3_token_refiner_dynamic[H3_HEADS, H3_HEAD_DIM](
            e0, frontend_w, config, ctx
        )
        empty_embeds_host = e_ref.to_host_bf16(ctx)
        empty_embeds_rows = e_ref.shape()[0]
        empty_tags = te0.empty_tags.copy()
        ctx.synchronize()
        print("[h3-train] guidance objective ON: scale", guidance_scale,
              "empty tokens", empty_embeds_rows)

    # slim weight dict: per-step needs ONLY the video patch projection
    var step_w = Dict[String, ArcPointer[Tensor]]()
    step_w[String("video_patch_proj.weight")] = frontend_w[String("video_patch_proj.weight")].copy()
    step_w[String("video_patch_proj.bias")] = frontend_w[String("video_patch_proj.bias")].copy()
    var empty_w = Dict[String, ArcPointer[Tensor]]()
    frontend_w = empty_w^
    ctx.synchronize()
    print("[h3-train] text embeds cached; frontend weights slimmed")

    # FP8-RESIDENT frozen base (krea2 pattern): ~19.25GB on-device, zero
    # per-step host->device weight traffic.
    print("[h3-train] building fp8-resident base (one streamed pass)...")
    # tail blocks stream bf16 from the retained mmap store
    var tq0 = perf_counter_ns()
    var resident_blocks = _arg_int(String("resident_blocks"), 42)
    var store = H3TrainBlockStoreFp8.open(ckpt, N_BLOCKS, resident_blocks, ctx)
    print("[h3-train] fp8 base resident in",
          Float64(perf_counter_ns() - tq0) / 1.0e9, "s")

    # LoRA state: init or resume
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
    if have_state:
        var st = SafeTensors.open(state_path)
        var names = _state_names()
        var ni = 0
        for b in range(N_BLOCKS):
            for s in range(SLOT_NAMES_LEN):
                var t0 = _load_st(st, names[ni], ctx)
                var t1 = _load_st(st, names[ni + 1], ctx)
                var t2 = _load_st(st, names[ni + 2], ctx)
                var t3 = _load_st(st, names[ni + 3], ctx)
                var t4 = _load_st(st, names[ni + 4], ctx)
                var t5 = _load_st(st, names[ni + 5], ctx)
                states.append(_AdapterState(
                    TArc(t0^), TArc(t1^), TArc(t2^), TArc(t3^), TArc(t4^), TArc(t5^),
                ))
                ni += 6
        var meta = _load_st(st, String("train_meta"), ctx).to_host(ctx)
        start_step = Int(meta[0])
        # advance the RNG to where the interrupted run left it
        for _ in range(Int(meta[1])):
            _ = rng.next_u64()
        print("[h3-train] RESUMED from step", start_step)
    else:
        for _ in range(N_BLOCKS):
            for s in range(SLOT_NAMES_LEN):
                var dims = _slot_out_in(s)
                states.append(_init_adapter_state(rank, dims[0], dims[1], rng, ctx))
        print("[h3-train] fresh LoRA init (kaiming-uniform down, zero up)")

    var rng_draws = 0  # draws since init (persisted for exact resume)

    # data order: LCG shuffle per epoch
    var order = List[Int]()
    for i in range(len(items)):
        order.append(i)

    var loras = _compute_loras(states, rank, scale, ctx)

    print("[h3-train] entering loop at step", start_step + 1, "/", max_steps)
    for step in range(start_step + 1, max_steps + 1):
        var t_step0 = perf_counter_ns()
        # ── item selection (reshuffle each epoch) ────────────────────────────
        var epoch_pos = (step - 1) % len(items)
        if epoch_pos == 0:
            for i in range(len(items) - 1, 0, -1):
                var j = Int(rng.next_u64() % UInt64(i + 1))
                rng_draws += 1
                var tmp = order[i]
                order[i] = order[j]
                order[j] = tmp
        var it = items[order[epoch_pos]].copy()

        var lat = h3_read_latent_cache(it.latent_path, ctx)
        if not lat.has_video or lat.lat_f != 1:
            raise Error("maiden arm expects image items ([24,1,H,W]): " + it.item_key)
        var te = h3_read_text_cache(it.te_path, ctx)
        # upstream trains in dit_dtype bf16: cast cached latents down exactly
        # like its .to(dtype) load (caches may be stored F32)
        var x0 = lat.video[].clone(ctx)
        if x0.dtype() != STDtype.BF16:
            x0 = cast_tensor(x0, STDtype.BF16, ctx)

        # ── sigma (image branch), quantized to the grid ──────────────────────
        var sigma_raw = _image_sigma(rng, lat.lat_h, lat.lat_w)
        rng_draws += 2
        var node = Int(sigma_raw * Float64(SIGMA_NODES - 1) + 0.5)
        if node < 0:
            node = 0
        if node > SIGMA_NODES - 1:
            node = SIGMA_NODES - 1
        var sigma = Float32(node) / Float32(SIGMA_NODES - 1)
        var t_v = Float32(1.0) - sigma

        # ── noise + target (gated bit-exact vs torch) ────────────────────────
        var vsh: List[Int] = [24, 1, lat.lat_h, lat.lat_w]
        var noise = randn(vsh^, UInt64(seed * 1000003 + step), STDtype.BF16, ctx)
        var x_t = h3_noisy_input(x0, noise, sigma, ctx)
        var target = h3_velocity_target(x0, noise, ctx)
        # the frontend's patch projection is an fp32-trap path (checkpoint F32
        # weights): feed F32 rows (exact bf16 upcast), it rne-casts back down.
        var x_rows = cast_tensor(
            minimax_h3_video_patchify(x_t, ctx), STDtype.F32, ctx
        )
        var target_rows = minimax_h3_video_patchify(target, ctx)

        # ── packed layout + rope + frontend (frozen, gated inference code) ───
        var anchors = List[Int]()
        var layout = minimax_h3_build_packed_sequence(
            te.tags, 1, lat.lat_h, lat.lat_w, 0, 2, 2, anchors,
        )
        var S = layout.sequence_length
        var row_ts = minimax_h3_build_row_timesteps(
            layout, t_v, Float32(1.0), Float32(0.999), Float32(1.0),
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
        var text_embeds = Tensor.from_host_bf16(
            text_embeds_host[item_i], tesh^, ctx
        )
        var hidden0 = concat(0, ctx, text_embeds, video_embeds)
        if hidden0.shape()[0] != S:
            raise Error("packed length mismatch (text+video != S)")

        # ── per-step mod tables: fetch ONLY this node's rows (host grid) ─────
        var node_idx = List[Int]()
        for _ in range(S):
            node_idx.append(0)  # fetched tables carry just this node's rows
        var adaln_idx = minimax_h3_adaln_indices(node_idx, layout.token_tags)
        var final_mod_rows = modgrid.final_row(node, ctx)

        # ── 50-block streamed LoRA forward ───────────────────────────────────
        var mods = List[TArc]()
        for b in range(N_BLOCKS):
            mods.append(TArc(modgrid.block_rows(b, node, ctx)))
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
            var e_text = Tensor.from_host_bf16(empty_embeds_host, tesh_e^, ctx)
            var hidden0_e = concat(0, ctx, e_text, video_embeds)
            if hidden0_e.shape()[0] != S_e:
                raise Error("guidance: packed length mismatch (empty+video != S_e)")
            var node_idx_e = List[Int]()
            for _ in range(S_e):
                node_idx_e.append(0)
            var adaln_idx_e = minimax_h3_adaln_indices(
                node_idx_e, layout_e.token_tags
            )
            var fwd_e = h3_stack_train_forward_streamed_fp8[H3_HEADS, H3_HEAD_DIM](
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

        var fwd = h3_stack_train_forward_streamed_fp8[H3_HEADS, H3_HEAD_DIM](
            hidden0, store, loras, mods, adaln_idx,
            rope[0], rope[1], H3_D, H3_F, rotary_dim, H3_EPS, ctx,
        )

        # ── final layer twin ─────────────────────────────────────────────────
        ctx.synchronize()
        var tp1 = perf_counter_ns()
        var empty_idx = List[Int]()
        var ffwd = h3_final_train_forward(
            fwd.out[], final_w, final_mod_rows, node_idx,
            layout.video_indices, empty_idx, H3_EPS, ctx,
        )

        # ── loss + d_pred (gated bit-exact vs torch) ─────────────────────────
        # With guidance: pred = c_hat = g/s + g_empty*(s-1)/s (F32 combine,
        # bf16 result like upstream's model-dtype arithmetic); chain rule
        # scales d_g by 1/s after the plain loss grad.
        var pred_rows: Tensor
        if use_guidance:
            var s = guidance_scale
            var g32 = cast_tensor(ffwd.video[], STDtype.F32, ctx)
            var e32 = cast_tensor(g_empty.value(), STDtype.F32, ctx)
            var chat = add(
                mul_scalar(g32, Float32(1.0) / s, ctx),
                mul_scalar(e32, (s - Float32(1.0)) / s, ctx),
                ctx,
            )
            pred_rows = cast_tensor(chat, STDtype.BF16, ctx)
        else:
            pred_rows = ffwd.video[].clone(ctx)
        var empty_mask = List[Bool]()
        var ml = h3_modality_loss(pred_rows, target_rows, empty_mask, ctx)
        var loss = ml.total / Float64(ml.elements)
        var none_mask = Optional[Tensor](None)
        var d_video = h3_loss_grad(
            pred_rows, target_rows, none_mask^, 1.0, Float64(ml.elements), ctx,
        )
        if use_guidance:
            d_video = mul_scalar_bf16out(
                cast_tensor(d_video, STDtype.F32, ctx),
                Float32(1.0) / guidance_scale, ctx,
            )

        # ── backward chain ───────────────────────────────────────────────────
        var d_audio_sh: List[Int] = [1, 1]
        var d_audio = full_device(d_audio_sh^, Float32(0.0), STDtype.BF16, ctx)
        var d_hidden = h3_final_train_backward(
            d_video, d_audio, ffwd.saved, final_w, node_idx,
            layout.video_indices, empty_idx, S, H3_EPS, ctx,
        )
        ctx.synchronize()
        var tp2 = perf_counter_ns()
        var grads = h3_stack_train_backward_streamed_fp8[H3_HEADS, H3_HEAD_DIM](
            d_hidden, fwd, store, loras, mods, adaln_idx,
            rope[0], rope[1], H3_D, H3_F, rotary_dim, H3_EPS, ctx,
        )

        ctx.synchronize()
        var tp3 = perf_counter_ns()
        # ── fused AdamW on the F32 masters ───────────────────────────────────
        var params = List[TArc]()
        var gts = List[TArc]()
        var ms = List[TArc]()
        var vs = List[TArc]()
        for b in range(N_BLOCKS):
            for s in range(SLOT_NAMES_LEN):
                var st_i = b * SLOT_NAMES_LEN + s
                var g = _grad_pair(grads.lora[b], s)
                params.append(states[st_i].a_m.copy())
                gts.append(g[0])
                ms.append(states[st_i].m_a.copy())
                vs.append(states[st_i].v_a.copy())
                params.append(states[st_i].b_m.copy())
                gts.append(g[1])
                ms.append(states[st_i].m_b.copy())
                vs.append(states[st_i].v_b.copy())
        fused_adamw_step(
            params, gts, ms, vs, step, lr,
            Float32(0.9), Float32(0.999), Float32(1.0e-8), Float32(0.01), ctx,
        )
        loras = _compute_loras(states, rank, scale, ctx)
        ctx.synchronize()

        var tp4 = perf_counter_ns()
        print("[phase] fwd", Float64(tp1 - tp0) / 1.0e9,
              "final+loss", Float64(tp2 - tp1) / 1.0e9,
              "bwd", Float64(tp3 - tp2) / 1.0e9,
              "opt", Float64(tp4 - tp3) / 1.0e9,
              "prep", Float64(tp0 - t_step0) / 1.0e9)
        var dt = Float64(perf_counter_ns() - t_step0) / 1.0e9
        print("[h3-train] step", step, "loss", loss, "sigma", sigma,
              "S", S, "item", it.item_key, "dt", dt, "s")

        # ── save / sample snapshots ──────────────────────────────────────────
        if step % save_every == 0 or step == max_steps or step % sample_every == 0:
            _save_all(states, rank, alpha, step, rng_draws, out_dir, name, ctx)

    print("[h3-train] DONE at step", max_steps)


def _save_all(
    states: List[_AdapterState], rank: Int, alpha: Float32,
    step: Int, rng_draws: Int, out_dir: String, name: String,
    ctx: DeviceContext,
) raises:
    # 1) LoRA in torchref key format (F32, alpha scalars)
    var names = List[String]()
    var tensors = List[TArc]()
    for b in range(N_BLOCKS):
        for s in range(SLOT_NAMES_LEN):
            var st_i = b * SLOT_NAMES_LEN + s
            var key = String("lora_unet_blocks_") + String(b) + "_" + _slot_key(s)
            names.append(key + ".lora_down.weight")
            tensors.append(states[st_i].a_m)
            names.append(key + ".lora_up.weight")
            tensors.append(states[st_i].b_m)
            names.append(key + ".alpha")
            var avals: List[Float32] = [alpha]
            # 0-DIM scalar (shape []) — the kohya/upstream convention; a [1]
            # tensor breaks strict consumers doing float(alpha).
            var ash = List[Int]()
            tensors.append(TArc(Tensor.from_host(avals, ash^, STDtype.F32, ctx)))
    var lora_path = (
        out_dir + "/" + name + "_step" + String(step) + ".safetensors"
    )
    save_safetensors(names, tensors, lora_path, ctx)
    print("[h3-train] saved LoRA:", lora_path)

    # 2) resume state (masters + moments + step + rng position)
    var snames = _state_names()
    var stensors = List[TArc]()
    for i in range(len(states)):
        stensors.append(states[i].a_m)
        stensors.append(states[i].b_m)
        stensors.append(states[i].m_a)
        stensors.append(states[i].v_a)
        stensors.append(states[i].m_b)
        stensors.append(states[i].v_b)
    snames.append(String("train_meta"))
    var meta: List[Float32] = [Float32(step), Float32(rng_draws)]
    var msh: List[Int] = [2]
    stensors.append(TArc(Tensor.from_host(meta, msh^, STDtype.F32, ctx)))
    var state_path = out_dir + "/" + name + "_state.safetensors"
    save_safetensors(snames, stensors, state_path, ctx)
    print("[h3-train] saved state:", state_path)
