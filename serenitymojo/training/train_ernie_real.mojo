# serenitymojo/training/train_ernie_real.mojo
#
# ERNIE-Image REAL LoRA training loop (pure Mojo). TRANSLATION of the working
# Rust trainer:
#   EriDiffusion-v2/crates/eridiffusion-cli/src/bin/train_ernie.rs
#
# Pipeline per step (mirrors train_ernie.rs:853-1130 + BaseErnieSetup.py predict):
#   1. Load a cached sample produced by prepare_ernie.rs:
#        latent          [1,128,32,32] F32  (post-VAE post-patchify, 128-ch)
#        text_embedding  [1,512,3072]  F32  (Mistral-3B layer hidden states, PAD-padded)
#        text_real_len   [1]                (real token count, pre-pad)
#   2. latent -> img_tokens [N_IMG,128]  (NCHW->NHWC pack; N_IMG = 32*32 = 1024)
#      text   -> txt_tokens [N_TXT,3072] (fixed comptime trim to N_TXT rows of the cache)
#   3. Sample sigma per LOGIT_NORMAL(bias=0,scale=1); shift=1 -> identity (train_ernie.rs:904).
#        sigma_idx = floor(logit_normal * 1000) in [0,999];  sigma = (idx+1)/1000.
#        noisy  = noise*sigma + latent*(1-sigma)             (flow-match; rectified flow)
#        target = noise - latent                             (train_ernie.rs:949)
#   4. timestep fed to the DiT = sigma_idx (INTEGER-valued; train_ernie.rs:956 — NOT /1000).
#   5. Build the shared-AdaLN source ONCE from the resident base weights + the timestep
#        (the deferred E2/E5 link): c = time_embed(sigma_idx);
#        mv = chunk6(silu(c)@adaLN_modulation.1 + bias);
#        [f_scale,f_shift] = chunk2(c@final_norm.linear + bias).
#   6. ernie_stack_lora_forward_resident_device -> pred [N_IMG,128]  (BF16 block-resident path).
#   7. loss = mean MSE(pred, target) in F32;  d_loss = (2/N)*(pred - target).
#   8. ernie_stack_lora_backward_resident_device -> LoRA d_A/d_B for all 7*36 adapters.
#   9. host global-L2-norm clip (max_norm = 1.0; train_ernie.rs:1072) -> ernie_lora_adamw_step.
#  10. save_ernie_lora at the end (PEFT/ai-toolkit keys, inference-loadable).
#
# Per-step line (stdout):  PROG step=<k> total=<MAX> loss=<f> grad=<f> lr=<f> secs=<wall>
#
# WHY BF16-RESIDENT: 36 F32 blocks ~= 31 GB > 24 GB 3090, but ERNIE's released
# block matrices are BF16 on disk. Keeping large block matrices BF16 and only
# norm vectors F32 fits the local 3090 target and removes per-step block reloads.
#
# MISTRAL TEXT ENCODER IS DEFERRED for this first real run: the cache ALREADY holds
# the Mistral text_embedding [1,512,3072], so the train loop reads it directly. The
# Mistral3B port (models/text_encoder/mistral3b_encoder.mojo) is only needed to
# (re)generate caches / sample prompts — off the training hot path.
#
# Run (SEPARATE command, after build):
#   cd /home/alex/mojodiffusion
#   rm -f serenitymojo.mojopkg
#   pixi run mojo build -I . -Xlinker -lm \
#       serenitymojo/training/train_ernie_real.mojo -o /tmp/train_ernie_real
#   /tmp/train_ernie_real

from std.sys import argv
from std.gpu.host import DeviceContext
from std.collections import List, Optional
from std.math import sqrt, log as flog, exp as fexp, isfinite, sin as fsin, cos as fcos
from std.memory import ArcPointer
from std.time import perf_counter

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.linear import linear
from serenitymojo.ops.activations import silu
from serenitymojo.ops.embeddings import timestep_embedding_sin_first

from serenitymojo.models.dit.ernie_contract import ERNIE_TRANSFORMER_DIR
from serenitymojo.models.dit.ernie_image import build_ernie_rope_tables
from serenitymojo.models.ernie.weights import (
    ErnieStackBase, load_ernie_stack_base, load_ernie_all_blocks_bf16_normf32,
)
from serenitymojo.models.ernie.block import ErnieModVecs
from serenitymojo.models.ernie.lora_block import ERNIE_SLOTS
from serenitymojo.models.ernie.ernie_stack_lora import (
    ErnieLoraSet, ErnieLoraGrads, build_ernie_lora_set,
    ernie_lora_set_to_device,
    ernie_stack_lora_forward_resident_device, ernie_stack_lora_backward_resident_device,
    ernie_lora_adamw_step, save_ernie_lora,
)

comptime TArc = ArcPointer[Tensor]

# ── REAL ERNIE dims (TRAINING_PLAN_ernie.md / ernie_contract.mojo) ────────────
comptime H = 32
comptime Dh = 128
comptime D = H * Dh            # 4096
comptime F = 12288             # REAL FFN hidden
comptime IN_CH = 128           # REAL latent channels
comptime TEXT_IN = 3072        # REAL Mistral hidden (text_in_dim)
comptime OUT_CH = 128          # REAL out channels
comptime NUM_LAYERS = 36       # REAL depth
comptime EPS = Float32(1e-06)

# ── cache geometry: latent [1,128,32,32] -> 32x32 = 1024 image tokens ─────────
comptime IMG_H = 32
comptime IMG_W = 32
comptime N_IMG = IMG_H * IMG_W     # 1024
# Fixed comptime text trim. The cache pads text to 512 with PAD-embedding rows
# (the model was trained to see real-tokens + the trim). The observed real_len
# in this cache is <= ~256, so a 256-row trim keeps every real token plus a
# little pad — and bounds the per-block SDPA scores [H, S, S] on the 3090.
comptime N_TXT = 256
comptime S = N_IMG + N_TXT         # 1280

# ── run knobs (mirror train_ernie.rs Args defaults) ───────────────────────────
comptime CACHE_DIR = "/home/alex/EriDiffusion/EriDiffusion-v2/cache/boxjana_ernie_512_FIXED"
comptime MAX_STEPS = 3             # short smoke; shared 3090
comptime RANK = 16                 # train_ernie.rs --rank default
comptime ALPHA = Float32(16.0)     # plain-LoRA: alpha == rank (scale 1.0)
comptime LR = Float32(3.0e-4)      # train_ernie.rs --lr default
comptime CLIP = Float32(1.0)       # train_ernie.rs CLIP_GRAD_NORM
comptime NUM_TRAIN_TIMESTEPS = 1000
comptime SEED = UInt64(42)         # train_ernie.rs SEED
comptime SAVE_PATH = "/home/alex/mojodiffusion/serenitymojo/output/ernie_lora_real.safetensors"


# ─────────────────────────────────────────────────────────────────────────────
# cache reading: a single safetensors tensor -> host List[Float32]
# ─────────────────────────────────────────────────────────────────────────────
def _read_cache_f32(st: SafeTensors, name: String, ctx: DeviceContext) raises -> List[Float32]:
    var info = st.tensor_info(name)
    var bytes = st.tensor_bytes(name)
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    var t = Tensor.from_view(tv, ctx)
    var f = cast_tensor(t, STDtype.F32, ctx)
    return f.to_host(ctx)


def _cache_dims(st: SafeTensors, name: String) raises -> List[Int]:
    var info = st.tensor_info(name)
    var out = List[Int]()
    for i in range(len(info.shape)):
        out.append(Int(info.shape[i]))
    return out^


# ── list cache files (sorted, like the Rust trainer cache_files.sort()) ───────
from std.os import listdir
def _list_cache(dir: String) raises -> List[String]:
    var raw = listdir(dir)
    var out = List[String]()
    for i in range(len(raw)):
        if raw[i].endswith(".safetensors"):
            out.append(dir + String("/") + raw[i])
    # insertion sort (deterministic order, mirrors cache_files.sort())
    for i in range(1, len(out)):
        var key = out[i]
        var j = i - 1
        while j >= 0 and out[j] > key:
            out[j + 1] = out[j]
            j -= 1
        out[j + 1] = key
    return out^


# ─────────────────────────────────────────────────────────────────────────────
# latent [1,128,H,W] (NCHW) -> img_tokens [N_IMG, IN_CH] (NHWC, row=token, col=channel)
# Mirrors the Rust patch_embed reshape: [B,C,H,W] -> reshape [B,C,H*W] -> permute
# [B,H*W,C]. With B=1: token t (= r*W + c), channel ch -> latent[ch*H*W + t].
# ─────────────────────────────────────────────────────────────────────────────
def _latent_to_img_tokens(latent: List[Float32]) -> List[Float32]:
    var out = List[Float32]()
    var hw = IMG_H * IMG_W
    for t in range(hw):
        for ch in range(IN_CH):
            out.append(latent[ch * hw + t])
    return out^


# ── trim/pad cached text [1, T, 3072] -> [N_TXT, 3072] rows ───────────────────
def _text_to_txt_tokens(text: List[Float32], t_cache: Int) -> List[Float32]:
    var out = List[Float32]()
    for r in range(N_TXT):
        if r < t_cache:
            for c in range(TEXT_IN):
                out.append(text[r * TEXT_IN + c])
        else:
            for _c in range(TEXT_IN):
                out.append(Float32(0.0))   # beyond cache rows (unreachable; cache T=512)
    return out^


# ─────────────────────────────────────────────────────────────────────────────
# host helpers
# ─────────────────────────────────────────────────────────────────────────────
def _zeros(n: Int) -> List[Float32]:
    var o = List[Float32]()
    for _ in range(n):
        o.append(0.0)
    return o^


def _l2(h: List[Float32]) -> Float64:
    var s = Float64(0.0)
    for i in range(len(h)):
        var x = Float64(h[i])
        s += x * x
    return s


def _abs_sum(h: List[Float32]) -> Float64:
    var s = Float64(0.0)
    for i in range(len(h)):
        var x = h[i]
        s += Float64(x) if x >= 0.0 else Float64(-x)
    return s


def _scale_inplace(mut h: List[Float32], s: Float32):
    for i in range(len(h)):
        h[i] = h[i] * s


def _count_nonfinite(v: List[Float32]) -> Int:
    var bad = 0
    for i in range(len(v)):
        if not isfinite(v[i]):
            bad += 1
    return bad


# ── deterministic gaussian noise (Box-Muller on a PCG stream), F32 ────────────
def _host_noise(n: Int, seed: UInt64) -> List[Float32]:
    var out = List[Float32]()
    var state = seed
    var have = False
    var spare = Float32(0.0)
    for _ in range(n):
        if have:
            out.append(spare)
            have = False
            continue
        # two uniforms in (0,1]
        state = state * 6364136223846793005 + 1442695040888963407
        var u1 = (Float32(Int(state >> 40)) + 1.0) * Float32(1.0 / 16777217.0)
        state = state * 6364136223846793005 + 1442695040888963407
        var u2 = Float32(Int(state >> 40)) * Float32(1.0 / 16777216.0)
        var r = sqrt(Float32(-2.0) * flog(u1))
        var theta = Float32(6.283185307179586) * u2
        out.append(r * fcos(theta))
        spare = r * fsin(theta)
        have = True
    return out^


# ── logit-normal timestep sample (train_ernie.rs sample_timestep_logit_normal) ─
# z ~ Normal(0,1); ln = sigmoid(z); t = ln * 1000; shift=1 -> identity.
# Returns sigma_idx in [0, 999] plus the advanced RNG state.
def _sample_sigma_idx(state0: UInt64) -> Tuple[Int, UInt64]:
    # one Box-Muller normal sample
    var state = state0
    state = state * 6364136223846793005 + 1442695040888963407
    var u1 = (Float32(Int(state >> 40)) + 1.0) * Float32(1.0 / 16777217.0)
    state = state * 6364136223846793005 + 1442695040888963407
    var u2 = Float32(Int(state >> 40)) * Float32(1.0 / 16777216.0)
    var z = sqrt(Float32(-2.0) * flog(u1)) * fcos(Float32(6.283185307179586) * u2)
    var ln = Float32(1.0) / (Float32(1.0) + fexp(-z))
    var t = ln * Float32(NUM_TRAIN_TIMESTEPS)   # shift=1 -> identity
    var idx = Int(t)                            # floor
    if idx < 0:
        idx = 0
    if idx > NUM_TRAIN_TIMESTEPS - 1:
        idx = NUM_TRAIN_TIMESTEPS - 1
    return (idx, state)


# ─────────────────────────────────────────────────────────────────────────────
# shared-AdaLN SOURCE (the deferred E2/E5 link). Built from the RESIDENT base
# weights + sigma_idx. Mirrors ernie_image.mojo time_embed/shared_adaln + the
# final-norm chunk, and ernie_image.rs:519-552. All F32 (ErnieStackBase is F32).
#   c   = linear2(silu(linear1(timestep_embedding_sin_first(idx))))
#   mv  = chunk6(silu(c) @ adaln_w + adaln_b)
#   fs  = chunk2(c @ final_norm_w + final_norm_b)  -> [f_scale, f_shift]
# Returns (mv, f_scale, f_shift). NOTE the timestep fed = sigma_idx (integer),
# matching train_ernie.rs:956 (NOT sigma*1000).
# ─────────────────────────────────────────────────────────────────────────────
def _chunk(src: List[Float32], idx: Int, width: Int) -> List[Float32]:
    var o = List[Float32]()
    var off = idx * width
    for i in range(width):
        o.append(src[off + i])
    return o^


def _shared_adaln_source(
    base: ErnieStackBase, sigma_idx: Int, ctx: DeviceContext
) raises -> Tuple[ErnieModVecs, List[Float32], List[Float32]]:
    var ts = List[Float32]()
    ts.append(Float32(sigma_idx))
    var ts_t = Tensor.from_host(ts, [1], STDtype.F32, ctx)
    # time embed: sin-first sinusoid -> linear1 -> silu -> linear2  (all F32)
    var emb = timestep_embedding_sin_first(ts_t, D, ctx, 10000.0)   # [1,D]
    var h1 = linear(emb, base.te_w1[], Optional[Tensor](base.te_b1[].clone(ctx)), ctx)
    h1 = silu(h1, ctx)
    var c = linear(h1, base.te_w2[], Optional[Tensor](base.te_b2[].clone(ctx)), ctx)  # [1,D]

    # shared adaLN: silu(c) @ adaln_w + adaln_b -> [1, 6D]
    var sc = silu(c, ctx)
    var adaln = linear(sc, base.adaln_w[], Optional[Tensor](base.adaln_b[].clone(ctx)), ctx)
    var adaln_h = adaln.to_host(ctx)   # [6D]

    # final-norm source: c @ final_norm_w + final_norm_b -> [1, 2D]
    var fmod = linear(c, base.final_norm_w[], Optional[Tensor](base.final_norm_b[].clone(ctx)), ctx)
    var fmod_h = fmod.to_host(ctx)     # [2D]

    # chunk6 the 6D modulation in order: shift_msa,scale_msa,gate_msa,shift_mlp,scale_mlp,gate_mlp
    var mv = ErnieModVecs(
        _chunk(adaln_h, 0, D), _chunk(adaln_h, 1, D), _chunk(adaln_h, 2, D),
        _chunk(adaln_h, 3, D), _chunk(adaln_h, 4, D), _chunk(adaln_h, 5, D),
    )
    var f_scale = _chunk(fmod_h, 0, D)
    var f_shift = _chunk(fmod_h, 1, D)
    return (mv^, f_scale^, f_shift^)


# ── global L2 norm over the flat LoRA grads (clip basis) ──────────────────────
def _grad_global_norm(grads: ErnieLoraGrads) -> Float64:
    var ss = Float64(0.0)
    for i in range(len(grads.d_a)):
        ss += _l2(grads.d_a[i])
        ss += _l2(grads.d_b[i])
    return sqrt(ss)


def _clip_grads(mut grads: ErnieLoraGrads, max_norm: Float32) -> Float64:
    var gn = _grad_global_norm(grads)
    if gn <= Float64(max_norm) or gn == 0.0:
        return gn
    var s = Float32(Float64(max_norm) / gn)
    for i in range(len(grads.d_a)):
        _scale_inplace(grads.d_a[i], s)
        _scale_inplace(grads.d_b[i], s)
    return gn


# ── LoRA-B absolute-sum (across all adapters): the "is it learning?" probe ────
def _lora_b_abs_sum(set: ErnieLoraSet) -> Float64:
    var s = Float64(0.0)
    for i in range(len(set.ad)):
        s += _abs_sum(set.ad[i].b)
    return s


def _lora_b_nonzero_slots(set: ErnieLoraSet) -> Int:
    var n = 0
    for i in range(len(set.ad)):
        if _abs_sum(set.ad[i].b) > 0.0:
            n += 1
    return n


def _parse_int(s: String) -> Int:
    var out = 0
    var bytes = s.as_bytes()
    for i in range(s.byte_length()):
        out = out * 10 + Int(bytes[i] - 0x30)
    return out


def main() raises:
    var args = argv()
    var run_steps = MAX_STEPS
    if len(args) >= 2:
        run_steps = _parse_int(String(args[1]))
    if run_steps < 1:
        raise Error("run_steps must be >= 1")

    var save_path = String(SAVE_PATH)
    if len(args) >= 3:
        save_path = String(args[2])

    var cache_dir = String(CACHE_DIR)
    if len(args) >= 4:
        cache_dir = String(args[3])

    var ctx = DeviceContext()
    print("==== ERNIE REAL LoRA training loop (pure Mojo) ====")
    print("  D=", D, " H=", H, " Dh=", Dh, " F=", F, " NUM_LAYERS=", NUM_LAYERS)
    print("  N_IMG=", N_IMG, " N_TXT=", N_TXT, " S=", S, " IN_CH=", IN_CH,
          " TEXT_IN=", TEXT_IN, " OUT_CH=", OUT_CH)
    print("  cache=", cache_dir)
    print("  max_steps=", run_steps, " rank=", RANK, " alpha=", ALPHA, " lr=", LR)
    print("  save_path=", save_path)
    print("  checkpoint=", ERNIE_TRANSFORMER_DIR)

    var mem0 = ctx.get_memory_info()
    print("  free VRAM at start (bytes):", mem0[0], " total:", mem0[1])

    # ── open the real sharded transformer checkpoint (streamed block source) ──
    var st = ShardedSafeTensors.open(String(ERNIE_TRANSFORMER_DIR))
    print("  opened transformer: num_shards =", st.num_shards())
    var base = load_ernie_stack_base(st, D, IN_CH, ctx)
    print("  base weights resident (patch/text/time/adaLN/final).")
    var blocks = load_ernie_all_blocks_bf16_normf32(st, NUM_LAYERS, ctx)
    var mem_blocks = ctx.get_memory_info()
    print("  block weights resident BF16/norm-F32:", len(blocks),
          " free VRAM after block load (bytes):", mem_blocks[0])

    # ── build the LoRA set (B=0 init -> adapter identity at step 0) ──
    var lora = build_ernie_lora_set(NUM_LAYERS, D, F, RANK, ALPHA)
    var n_adapters = NUM_LAYERS * ERNIE_SLOTS
    print("  LoRA adapters:", n_adapters, " (7 slots x", NUM_LAYERS, "layers)")
    print("  LoRA-B |.|_1 at init =", _lora_b_abs_sum(lora), " (expect 0.0)")

    # ── RoPE tables for the real seq (image-first/text-second 3-axis half-split) ──
    # text_len_real for axis-0 offset: use N_TXT (the comptime trim) — every real
    # token is within [0,N_TXT). build_ernie_rope_tables requires real in (0,N_TXT].
    var rope = build_ernie_rope_tables[N_IMG, N_TXT, H, Dh](
        IMG_H, IMG_W, N_TXT, ctx, STDtype.F32
    )
    print("  RoPE tables built: cos/sin [S*H, Dh] = [", S * H, ",", Dh, "]")

    var cache_files = _list_cache(cache_dir)
    if len(cache_files) == 0:
        raise Error("no cache files found")
    print("  found", len(cache_files), "cached samples")

    var rng_state = SEED
    var t_start = perf_counter()

    for step in range(run_steps):
        var cache_idx = step % len(cache_files)
        var cs = SafeTensors.open(cache_files[cache_idx])

        var latent = _read_cache_f32(cs, String("latent"), ctx)            # [128*32*32]
        var text = _read_cache_f32(cs, String("text_embedding"), ctx)      # [512*3072]
        var tdims = _cache_dims(cs, String("text_embedding"))
        var t_cache = tdims[1]

        var img_tokens = _latent_to_img_tokens(latent)                     # [N_IMG, IN_CH]
        var txt_tokens = _text_to_txt_tokens(text, t_cache)                # [N_TXT, TEXT_IN]

        # ── sigma + flow-match noise/target (F32; train_ernie.rs:904-949) ──
        var sigma_draw = _sample_sigma_idx(rng_state)
        var sigma_idx = sigma_draw[0]
        rng_state = sigma_draw[1]
        var sigma = Float32(sigma_idx + 1) / Float32(NUM_TRAIN_TIMESTEPS)
        var n_lat = N_IMG * IN_CH
        var noise = _host_noise(n_lat, SEED ^ (UInt64(step) + 1))
        # noisy = noise*sigma + latent*(1-sigma)  (image-token space; same packing)
        var target = List[Float32]()
        for i in range(n_lat):
            target.append(noise[i] - img_tokens[i])      # target = noise - latent
        # the DiT input is the NOISY latent tokens (not the clean ones)
        var noisy_tokens = List[Float32]()
        for i in range(n_lat):
            noisy_tokens.append(noise[i] * sigma + img_tokens[i] * (Float32(1.0) - sigma))

        # ── shared-AdaLN source from the resident base + sigma_idx ──
        var src = _shared_adaln_source(base, sigma_idx, ctx)
        var mv = src[0].copy()
        var f_scale = src[1].copy()
        var f_shift = src[2].copy()

        # ── device-resident LoRA forward (BF16 base blocks loaded once) ──
        var lora_dev = ernie_lora_set_to_device(lora, STDtype.F32, ctx)
        var fwd = ernie_stack_lora_forward_resident_device[H, Dh, N_IMG, N_TXT, S](
            noisy_tokens.copy(), txt_tokens.copy(), base, blocks, lora_dev, mv,
            f_scale.copy(), f_shift.copy(), rope[0], rope[1],
            D, F, IN_CH, TEXT_IN, OUT_CH, EPS, ctx,
        )
        var pred = fwd.out.copy()                          # [N_IMG, OUT_CH]

        # ── MSE loss + upstream grad d_loss = (2/N)*(pred - target) ──
        var n_out = N_IMG * OUT_CH
        var sse = Float64(0.0)
        var d_out = List[Float32]()
        var inv_n = Float32(2.0) / Float32(n_out)
        for i in range(n_out):
            var diff = pred[i] - target[i]
            sse += Float64(diff) * Float64(diff)
            d_out.append(diff * inv_n)
        var loss = Float32(sse / Float64(n_out))

        # ── resident LoRA backward -> all 7*36 adapter grads ──
        var grads = ernie_stack_lora_backward_resident_device[H, Dh, N_IMG, N_TXT, S](
            d_out, noisy_tokens.copy(), txt_tokens.copy(), base, blocks, lora_dev, mv,
            f_scale.copy(), f_shift.copy(), rope[0], rope[1], fwd,
            D, F, IN_CH, TEXT_IN, OUT_CH, EPS, ctx,
        )

        # ── global-norm clip (max_norm = 1.0) then AdamW on every adapter ──
        var gn = _clip_grads(grads, CLIP)
        ernie_lora_adamw_step(lora, grads, step + 1, LR, ctx)

        var secs = perf_counter() - t_start
        print("PROG step=", step, " total=", run_steps, " loss=", loss,
              " grad=", Float32(gn), " lr=", LR, " secs=", secs,
              " sigma_idx=", sigma_idx, " nonfinite=", grads.nonfinite_lora_grads,
              " LoRA-B|.|1=", _lora_b_abs_sum(lora))

    # ── final LoRA-B growth report + save ──
    var b_sum = _lora_b_abs_sum(lora)
    var b_nz = _lora_b_nonzero_slots(lora)
    print("")
    print("==== RESULT ====")
    print("  LoRA-B |.|_1 after training =", b_sum, " (init 0.0)")
    print("  LoRA-B nonzero slots =", b_nz, "/", n_adapters,
          " ratio =", Float32(b_nz) / Float32(n_adapters))
    var npairs = save_ernie_lora(lora, save_path, ctx)
    print("  saved", npairs, "adapter pairs to", save_path)
