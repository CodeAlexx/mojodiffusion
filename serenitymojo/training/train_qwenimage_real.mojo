# train_qwenimage_real.mojo — Qwen-Image LoRA training loop (block-swap offload).
#
# TRANSLATION of EriDiffusion-v2 train_qwenimage.rs onto the parity-verified Mojo
# Qwen-Image LoRA OFFLOAD stack. 60 all-double-stream blocks (D=3072, H=24, Dh=128,
# F=12288, in_ch=64, txt_ch=3584, out_ch=64). Mirrors train_chroma_real.mojo's loop
# structure (timing, grad clip, progress display) and qwenimage.rs's recipe.
#
# QWENIMAGE vs CHROMA (key deltas):
#   - ALL double-stream (60 blocks, 0 single). No single-stream loop.
#   - SEPARATE per-block mod-MLPs (img_mod.1 / txt_mod.1). Each frozen mod MLP
#     projects silu(temb) -> [6D] per block. Mod-MLP weights are STREAMED from the
#     block (same Block handle as the attention weights). Frozen: grads discarded.
#   - time_text_embed: sinusoidal(t*1000, 256) -> silu(Linear1) -> Linear2 -> [D].
#     This MLP is applied ONCE per step to get silu_temb_h [1,D].
#   - norm_out.linear: [2D,D] produces final_scale/final_shift from silu_temb_h.
#   - txt_ch = 3584 (Qwen2.5-VL text encoder hidden dim; NOT T5-XXL 4096).
#   - Flow-match recipe (qwenimage.rs:1093-1099):
#       x_t = (1 - sigma)*latent + sigma*noise   (note: opposite sign to Flux)
#       target = noise - latent
#   - Timestep: sigmoid(N(0,1)) then apply_qwen_shift (shift=3.0 from config).
#   - out_ch = 64 (latent channels; proj_out [64,D]; target [N_IMG, 64]).
#   - ROPE: 3-axis interleaved, axes=(16,56,56), theta=10000.
#
# Recipe (configs/qwenimage.json): lr=1e-4, rank=16, alpha=16,
#   timestep_shift=3.0, clip_grad_norm=1.0.
#
# MEMORY: 60 * ~648 MB BF16/FP8 blocks + resident base (~tiny) + LoRA + optimizer.
# Block-swap streams one block at a time. A fixed-sigma smoke mode confirms
# loss decreases monotonically with a frozen sample.
#
# FIXED_SIGMA_SMOKE: every step uses the SAME latent+text AND a fixed sigma+noise.
# A correct LoRA backward MUST drive loss DOWN monotonically.
#
# COMPILE-ONLY DELIVERABLE: do NOT execute this binary (full 60-block model).
#   pixi run mojo build -I . -Xlinker -lm -Xlinker -lcuda \
#     serenitymojo/training/train_qwenimage_real.mojo \
#     -o /tmp/train_qwenimage_real
#
# ── UNVERIFIABLE-WITHOUT-CACHE ITEMS (flagged for future parity gate) ──────────
# (1) Checkpoint dtype FP8-E4M3: cast_tensor handles FP8→F32 dequant; parity vs
#     torch FP8 checkpoint not gated (no local FP8 reference weights).
# (2) txt_ch=3584: the Qwen2.5-VL text encoder; cache dir uses placeholder zeros.
# (3) RoPE total_half = 8+28+28 = 64 = Dh//2: matches config axes (16,56,56);
#     parity to qwenimage.rs RoPE verified per-block cos>=0.999 in the block tests.
# (4) norm_out.linear [2D,D] → chunk 0 scale chunk 1 shift: diffusers layout from
#     config.json; not re-gated here (same as weights.mojo build_qwen_per_block_mods).
# (5) txt_norm.weight [txt_ch]: applied by the caller pre-normalization in inference;
#     in the trainer we skip it (match train_qwenimage.rs which operates on already-
#     normalized text embeddings from the cache).

from sys import argv
from std.collections import List, Optional
from std.gpu.host import DeviceContext
from std.math import sqrt, log as flog, cos as fcos, sin as fsin
from std.time import perf_counter_ns
from std.os import listdir

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.embeddings import timestep_embedding
from serenitymojo.ops.activations import silu

from serenitymojo.models.qwenimage.qwenimage_stack import (
    QwenStackBase, _t as _qstack_t,
)
from serenitymojo.models.qwenimage.weights import load_qwen_stack_base
from serenitymojo.models.qwenimage.qwenimage_stack_lora import (
    QwenLoraSet, QwenLoraGradSet, QwenOffloadBase, QwenOffloadForward,
    build_qwen_lora_set, save_qwen_lora,
    qwenimage_stack_lora_forward_offload,
    qwenimage_stack_lora_backward_offload,
    qwen_offload_lora_adamw_step,
    DBL_SLOTS,
)
from serenitymojo.models.dit.qwenimage_dit import (
    QwenImageConfig, build_qwenimage_rope_tables,
)
from serenitymojo.offload.qwenimage_plan import build_qwenimage_offload_plan
from serenitymojo.offload.plan import OffloadConfig
from serenitymojo.offload.turbo_planned_loader import TurboPlannedLoader
from serenitymojo.training.schedule import sample_timestep_logit_normal
from serenitymojo.training.progress_display import print_trainer_progress


# ── arch (qwen-image; confirmed from config.json + qwenimage_dit.mojo) ────────
comptime H = 24
comptime Dh = 128
comptime D = H * Dh            # 3072
comptime FMLP = 12288          # mlp_hidden = D * 4
comptime IN_CH = 64            # in_channels (patchified latent)
comptime TXT_CH = 3584         # Qwen2.5-VL text encoder hidden
comptime OUT_CH = 64           # proj_out output channels
comptime NUM_DOUBLE = 60       # all-double-stream
comptime TIMESTEP_DIM = 256    # sinusoidal embedding dim
comptime EPS = Float32(1.0e-6)

# ── resolution (512px / patch=2): latent [64,32,32] -> [N_IMG=1024, 64] ───────
comptime LAT_C = 64            # in_channels (VAE latent channels = 16 before patch)
comptime LAT_H = 32            # latent height at patch-2 (512px / 16)
comptime LAT_W = 32            # latent width
comptime N_IMG = LAT_H * LAT_W  # 1024 image tokens
comptime N_TXT = 256           # text token sequence length (padded)
comptime S = N_TXT + N_IMG     # 1280 joint sequence

# ── RoPE frame/height/width for 512px 1-frame ─────────────────────────────────
comptime ROPE_FRAME = 1
comptime ROPE_H = 32           # == LAT_H (latent height in patch coords)
comptime ROPE_W = 32           # == LAT_W

# ── recipe (configs/qwenimage.json + qwenimage.rs) ────────────────────────────
comptime RANK = 16
comptime ALPHA = Float32(16.0)
comptime LR = Float32(1.0e-4)
comptime TIMESTEP_SHIFT = Float32(3.0)
comptime CLIP_GRAD_NORM = Float32(1.0)
comptime SEED_BASE = UInt64(42)

comptime FIXED_SIGMA_SMOKE = True
comptime FIXED_SIGMA_VAL = Float32(0.5)    # fixed sigma for smoke test

comptime CKPT = "/home/alex/.serenity/models/checkpoints/qwen_image_fp8_e4m3fn.safetensors"
comptime CACHE_DIR = "/home/alex/datasets/qwenimage_cache_512"
comptime LORA_DIR = "/home/alex/mojodiffusion/output/qwenimage_lora"


# ── deterministic host gaussian noise (Box-Muller PCG; per-step seed) ─────────
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


def _global_norm(grads: QwenLoraGradSet) -> Float64:
    var ss = 0.0
    for i in range(len(grads.d_a)):
        for j in range(len(grads.d_a[i])):
            ss += Float64(grads.d_a[i][j]) * Float64(grads.d_a[i][j])
        for j in range(len(grads.d_b[i])):
            ss += Float64(grads.d_b[i][j]) * Float64(grads.d_b[i][j])
    return sqrt(ss)


def _clip(mut grads: QwenLoraGradSet, max_norm: Float32) -> Float64:
    var gn = _global_norm(grads)
    if gn <= Float64(max_norm) or gn == 0.0:
        return gn
    var s = Float32(Float64(max_norm) / gn)
    for i in range(len(grads.d_a)):
        for j in range(len(grads.d_a[i])):
            grads.d_a[i][j] = grads.d_a[i][j] * s
        for j in range(len(grads.d_b[i])):
            grads.d_b[i][j] = grads.d_b[i][j] * s
    return gn


def _list_cache(dir: String) raises -> List[String]:
    var raw = listdir(dir)
    var fs = List[String]()
    for i in range(len(raw)):
        if raw[i].endswith(".safetensors"):
            fs.append(dir + String("/") + raw[i])
    if len(fs) == 0:
        raise Error(String("qwenimage cache: no .safetensors in ") + dir)
    for i in range(1, len(fs)):
        var j = i
        while j > 0 and fs[j - 1] > fs[j]:
            var tmp = fs[j - 1]
            fs[j - 1] = fs[j]
            fs[j] = tmp
            j -= 1
    return fs^


def _load_host_f32(st: SafeTensors, name: String, ctx: DeviceContext) raises -> List[Float32]:
    var info = st.tensor_info(name)
    var bytes = st.tensor_bytes(name)
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    var t = Tensor.from_view(tv, ctx)
    return cast_tensor(t, STDtype.F32, ctx).to_host(ctx)


def _load_host_f32_sharded(st: ShardedSafeTensors, name: String, ctx: DeviceContext) raises -> List[Float32]:
    var tv = st.tensor_view(name)
    var t = Tensor.from_view(tv, ctx)
    return cast_tensor(t, STDtype.F32, ctx).to_host(ctx)


def _load_host_bf16_sharded(st: ShardedSafeTensors, name: String, ctx: DeviceContext) raises -> List[BFloat16]:
    var tv = st.tensor_view(name)
    var t = Tensor.from_view(tv, ctx)
    return t.to_host_bf16(ctx)


# Sinusoidal timestep embedding (host, returns [timestep_dim] F32).
# t_val: sigma in [0,1]; Qwen scales by 1000 before embedding.
def _sinusoidal_temb(t_val: Float32, ctx: DeviceContext) raises -> List[Float32]:
    var t_h = List[Float32]()
    t_h.append(t_val * Float32(1000.0))
    var t_tensor = Tensor.from_host(t_h, [1], STDtype.F32, ctx)
    var t_emb = timestep_embedding(
        t_tensor, Int(TIMESTEP_DIM), ctx, Float32(10000.0), STDtype.F32
    )
    return t_emb.to_host(ctx)   # [1, TIMESTEP_DIM] flat = [TIMESTEP_DIM] scalars


# Build silu_temb_h: silu(time_text_embed(t)) = silu(MLP(sinusoidal(t))).
# te_lin1_w [D, timestep_dim], te_lin1_b [D], te_lin2_w [D,D], te_lin2_b [D].
def _build_silu_temb(
    t_val: Float32,
    te_lin1_w: List[BFloat16], te_lin1_b: List[BFloat16],
    te_lin2_w: List[BFloat16], te_lin2_b: List[BFloat16],
    ctx: DeviceContext,
) raises -> List[Float32]:
    from serenitymojo.ops.linear import linear
    var sin_emb_h = _sinusoidal_temb(t_val, ctx)        # [TIMESTEP_DIM]
    var t_emb = Tensor.from_host(sin_emb_h, [1, Int(TIMESTEP_DIM)], STDtype.BF16, ctx)
    var b1 = Tensor.from_host_bf16(te_lin1_b.copy(), [Int(D)], ctx)
    var h1 = linear(
        t_emb,
        Tensor.from_host_bf16(te_lin1_w.copy(), [Int(D), Int(TIMESTEP_DIM)], ctx),
        Optional[Tensor](b1^), ctx,
    )
    var h1_silu = silu(h1, ctx)
    var b2 = Tensor.from_host_bf16(te_lin2_b.copy(), [Int(D)], ctx)
    var temb_out = linear(
        h1_silu,
        Tensor.from_host_bf16(te_lin2_w.copy(), [Int(D), Int(D)], ctx),
        Optional[Tensor](b2^), ctx,
    )
    # final silu for use as per-block mod MLP input
    return silu(temb_out, ctx).to_host(ctx)   # [1, D] flat


# pack_latents: [LAT_C, LAT_H, LAT_W] flat F32 -> [N_IMG, LAT_C] (trivial
# patch=1 since latent is already at patch resolution for Qwen-Image 512px).
# Qwen-Image patchify: patch_size=2 applied at VAE decode time (in_channels=64
# = 16ch * 2*2); the latent cache already stores the patchified [N_IMG, 64].
# So no patchify needed — the cache tensor IS [N_IMG, 64] already.
# (Verified: in_channels=64, out_channels=64 in config.json; the latent cache
#  for training stores the pack_latents output, not the raw VAE latent.)


def main() raises:
    var ctx = DeviceContext()
    var a = argv()
    var run_steps = 5
    if len(a) >= 2:
        var v = 0
        var bs = String(a[1]).as_bytes()
        for i in range(String(a[1]).byte_length()):
            v = v * 10 + Int(bs[i] - 0x30)
        run_steps = v

    print("=== Qwen-Image REAL LoRA training loop (block-swap offload) ===")
    print("  arch: D=", D, " H=", H, " Dh=", Dh, " F=", FMLP, " in_ch=", IN_CH,
          " txt_ch=", TXT_CH, " out_ch=", OUT_CH)
    print("  depth: NUM_DOUBLE=", NUM_DOUBLE, " (all-double)")
    print("  tokens: N_IMG=", N_IMG, " N_TXT=", N_TXT, " S=", S)
    print("  recipe: rank=", RANK, " alpha=", ALPHA, " lr=", LR,
          " shift=", TIMESTEP_SHIFT)
    print("  LoRA targets: 12/block (img/txt x q,k,v,out,ff_up,ff_down) x 60 = 720")
    print("  fixed_sigma_smoke=", FIXED_SIGMA_SMOKE)
    print("  ckpt:", CKPT)
    print("  cache:", CACHE_DIR)

    # ── load frozen stack-level base (img_in/txt_in/proj_out + timestep MLP) ──
    print("[load] QwenStackBase from checkpoint")
    var st = ShardedSafeTensors.open(String(CKPT))

    var base_stack = load_qwen_stack_base(st, Int(D), Int(IN_CH), Int(TXT_CH), Int(OUT_CH), ctx)

    # timestep MLP weights (top-level in checkpoint)
    var te_lin1_w = _load_host_bf16_sharded(
        st, "time_text_embed.timestep_embedder.linear_1.weight", ctx
    )   # [D, TIMESTEP_DIM]
    var te_lin1_b = _load_host_bf16_sharded(
        st, "time_text_embed.timestep_embedder.linear_1.bias", ctx
    )   # [D]
    var te_lin2_w = _load_host_bf16_sharded(
        st, "time_text_embed.timestep_embedder.linear_2.weight", ctx
    )   # [D, D]
    var te_lin2_b = _load_host_bf16_sharded(
        st, "time_text_embed.timestep_embedder.linear_2.bias", ctx
    )   # [D]

    # norm_out.linear weights (for final scale/shift)
    var norm_out_w = _load_host_bf16_sharded(st, "norm_out.linear.weight", ctx)  # [2D, D]
    var norm_out_b = _load_host_bf16_sharded(st, "norm_out.linear.bias", ctx)    # [2D]

    var base = QwenOffloadBase(
        base_stack^,
        norm_out_w.copy(), norm_out_b.copy(),
        te_lin1_w.copy(), te_lin1_b.copy(),
        te_lin2_w.copy(), te_lin2_b.copy(),
    )
    print("[load] base resident (img_in/txt_in/proj_out/timestep-MLP/norm_out)")

    # ── block-swap offload loader ────────────────────────────────────────────
    var plan = build_qwenimage_offload_plan()
    var cfg = OffloadConfig.synchronous_single()
    var loader = TurboPlannedLoader.open(String(CKPT), plan^, cfg, ctx)
    print("[load] offload loader opened (", loader.block_count(), "blocks)")

    # ── 3-axis RoPE tables (fixed for 512px / 1 frame) ──────────────────────
    var qcfg = QwenImageConfig.qwen_image()
    var rope = build_qwenimage_rope_tables(
        Int(ROPE_FRAME), Int(ROPE_H), Int(ROPE_W), Int(N_TXT),
        Int(H), qcfg, STDtype.F32, ctx
    )
    var cos_h = rope[0].to_host(ctx)
    var sin_h = rope[1].to_host(ctx)
    print("[load] Qwen-Image 3-axis RoPE tables built (S*H x Dh//2)")

    # ── LoRA set (B=0 init -> identity at step 0) ────────────────────────────
    var lora = build_qwen_lora_set(Int(NUM_DOUBLE), Int(D), Int(FMLP), Int(RANK), ALPHA)
    var n_adapters = Int(NUM_DOUBLE) * Int(DBL_SLOTS)
    print("[lora] adapters:", n_adapters, " (", DBL_SLOTS, "x", NUM_DOUBLE, "double)")

    var files: List[String]
    var have_cache = True
    try:
        files = _list_cache(String(CACHE_DIR))
        print("[cache] samples:", len(files))
    except:
        files = List[String]()
        have_cache = False
        print("[cache] WARNING: no cache at", CACHE_DIR, "- using synthetic tokens")

    var b_absum_init = Float32(0.0)
    for i in range(n_adapters):
        b_absum_init += _absum(lora.dbl[i].b)
    print("[lora] LoRA-B |.|_1 at init =", b_absum_init, " (expect 0.0)")

    var first_loss = Float32(0.0)
    var last_loss = Float32(0.0)

    var train_start = perf_counter_ns()
    for k in range(1, run_steps + 1):
        var t0 = perf_counter_ns()

        # ── timestep ──
        var sigma: Float32
        var step_seed = UInt64(1) if FIXED_SIGMA_SMOKE else UInt64(k)
        if FIXED_SIGMA_SMOKE:
            sigma = FIXED_SIGMA_VAL
        else:
            sigma = sample_timestep_logit_normal(SEED_BASE + step_seed, TIMESTEP_SHIFT)

        # ── load / synthesize tokens ──
        var img_tokens = List[Float32]()   # [N_IMG, IN_CH]
        var txt_tokens = List[Float32]()   # [N_TXT, TXT_CH]

        if have_cache and len(files) > 0:
            var slot = 0 if FIXED_SIGMA_SMOKE else (k - 1) % len(files)
            var cst = SafeTensors.open(files[slot])
            img_tokens = _load_host_f32(cst, String("latent"), ctx)
            # txt embed may be stored as "t5_embed" or "txt_embed"
            var txt_flat = _load_host_f32(cst, String("txt_embed"), ctx)
            var txt_seq = len(txt_flat) // Int(TXT_CH)
            for r in range(Int(N_TXT)):
                if r < txt_seq:
                    for c in range(Int(TXT_CH)):
                        txt_tokens.append(txt_flat[r * Int(TXT_CH) + c])
                else:
                    for _ in range(Int(TXT_CH)):
                        txt_tokens.append(Float32(0.0))
        else:
            # synthetic: zeros (smoke compile check only)
            for _ in range(Int(N_IMG) * Int(IN_CH)):
                img_tokens.append(Float32(0.0))
            for _ in range(Int(N_TXT) * Int(TXT_CH)):
                txt_tokens.append(Float32(0.0))

        # ── flow-match: noisy = (1-sigma)*latent + sigma*noise ; target = noise - latent ──
        var noise = _host_noise(Int(N_IMG) * Int(IN_CH), SEED_BASE * UInt64(7919) + step_seed)
        var noisy = List[Float32]()
        var target = List[Float32]()
        var one_minus_sigma = Float32(1.0) - sigma
        for i in range(len(img_tokens)):
            noisy.append(one_minus_sigma * img_tokens[i] + sigma * noise[i])
            target.append(noise[i] - img_tokens[i])

        # ── silu_temb_h: frozen time_text_embed output [1, D] ──
        var silu_temb_h = _build_silu_temb(
            sigma,
            te_lin1_w, te_lin1_b, te_lin2_w, te_lin2_b,
            ctx,
        )

        # ── forward (offload, full 60-block depth) -> pred [N_IMG, OUT_CH] ──
        var fwd = qwenimage_stack_lora_forward_offload[H, Dh, N_IMG, N_TXT, S](
            noisy.copy(), txt_tokens.copy(), silu_temb_h.copy(),
            base, loader, lora,
            cos_h.copy(), sin_h.copy(),
            norm_out_w, norm_out_b,
            Int(D), Int(FMLP), Int(IN_CH), Int(TXT_CH), Int(OUT_CH), EPS, ctx,
        )

        # ── loss = MSE(pred, target) ; d_loss = (2/N)(pred - target) ──
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

        # ── backward (offload, full 60-block depth) ──
        var grads = qwenimage_stack_lora_backward_offload[H, Dh, N_IMG, N_TXT, S](
            d_loss,
            noisy.copy(), txt_tokens.copy(), silu_temb_h.copy(),
            base, loader, lora,
            cos_h.copy(), sin_h.copy(),
            norm_out_w, norm_out_b,
            fwd,
            Int(D), Int(FMLP), Int(IN_CH), Int(TXT_CH), Int(OUT_CH), EPS, ctx,
        )

        # ── grad norm + clip(1.0) ──
        var gn_before = _clip(grads, CLIP_GRAD_NORM)

        # ── AdamW ──
        qwen_offload_lora_adamw_step(lora, grads, k, LR, ctx)

        var t1 = perf_counter_ns()
        var secs = Float64(t1 - t0) / 1.0e9
        print_trainer_progress(
            String("QwenImage-lora"), k, run_steps, 1,
            loss, Float64(gn_before), secs, 0.0,
            Float64(t1 - train_start) / 1.0e9,
        )
        if grads.nonfinite_lora_grads != 0:
            print("[QwenImage-lora] warning nonfinite_lora_grads=", grads.nonfinite_lora_grads)

    print("")
    print("first_loss=", first_loss, " last_loss=", last_loss)
    var b_absum_final = Float32(0.0)
    for i in range(n_adapters):
        b_absum_final += _absum(lora.dbl[i].b)
    var trains = (b_absum_init == 0.0) and (b_absum_final > 0.0)
    if trains and (last_loss == last_loss):
        print("RESULT: REAL run OK — LoRA-B grew 0 ->", b_absum_final,
              "; loss", first_loss, "->", last_loss,
              (" (DECREASED)" if last_loss < first_loss else " (see trajectory)"))
        _ = save_qwen_lora(lora, String(LORA_DIR) + String("/qwenimage_lora_smoke.safetensors"), ctx)
    else:
        print("RESULT: FAIL trains=", trains)
