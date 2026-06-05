# ernie_pipeline_1024_multistep.mojo — ERNIE-Image 1024x1024 full pipeline.
#
# Pipeline (mirrors ernie_image_infer.rs exactly):
#   1. Load Mistral-3B text embeddings from sidecar (produced by Rust encode-only run)
#   2. Project text [1,256,3072] -> [1,256,4096]
#   3. Build RoPE tables per call (cond/uncond have different text lengths)
#   4. For each step, run sequential CFG:
#      a. Run 36-block ERNIE DiT (BlockLoader streams blocks one at a time)
#      b. CFG: uncond + scale * (cond - uncond)
#      c. Euler step: latent += velocity * dt
#   5. VAE decode via KleinVaeDecoder
#   6. Save PNG
#
# Text conditioning: Option A (Rust sidecar).
#   /tmp/ernie_embeddings/model.safetensors has keys:
#     context_cond   [1, 256, 3072] F32
#     context_uncond [1, 256, 3072] F32

from std.gpu.host import DeviceContext
from std.math import sqrt
from std.memory import ArcPointer

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.offload.block_loader import BlockLoader, unload_block
from serenitymojo.models.dit.ernie_contract import (
    ERNIE_DIT_HEAD_DIM,
    ERNIE_DIT_HEADS,
    ERNIE_DIT_FFN_HIDDEN,
    ERNIE_DIT_HIDDEN,
    ERNIE_DIT_LAYERS,
    ERNIE_DIT_TEXT_IN_DIM,
    ERNIE_LATENT_CHANNELS,
    ERNIE_LATENT_H,
    ERNIE_LATENT_W,
    ERNIE_PATCH_SIZE,
    ERNIE_IMAGE_TOKENS,
    ERNIE_TEXT_MAX_TOKENS,
    ERNIE_TRANSFORMER_DIR,
    ERNIE_VAE_FILE,
    ERNIE_DIT_ROPE_AXIS_0,
    ERNIE_DIT_ROPE_AXIS_1,
    ERNIE_DIT_ROPE_AXIS_2,
    ERNIE_DIT_ROPE_THETA,
)
from serenitymojo.models.dit.ernie_image import (
    ErnieImageResident,
    build_ernie_rope_tables,
)
from serenitymojo.models.vae.klein_decoder import KleinVaeDecoder
from serenitymojo.ops.activations import gelu, silu
from serenitymojo.ops.attention import sdpa_nomask
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.elementwise import modulate, residual_gate
from serenitymojo.ops.linear import linear
from serenitymojo.ops.norm import rms_norm, layer_norm
from serenitymojo.ops.random import randn
from serenitymojo.ops.rope import rope_halfsplit_full
from serenitymojo.ops.tensor_algebra import (
    add, sub, mul, mul_scalar, reshape, permute, slice, concat
)
from serenitymojo.sampling.ernie_sampling import (
    ErnieFlowMatchScheduler,
    ernie_cfg,
    ernie_euler_step,
    ernie_model_timestep_from_sigma,
)
from serenitymojo.image.png import save_png, ValueRange


# ── Configuration ──────────────────────────────────────────────────────────────
comptime EMBEDDINGS_DIR = "/tmp/ernie_embeddings"
comptime OUT_PATH = "/home/alex/mojodiffusion/output/ernie_1024_30step.png"

comptime N_IMG = ERNIE_IMAGE_TOKENS        # 4096
comptime N_TXT = ERNIE_TEXT_MAX_TOKENS     # 256

# Cond text length (from Rust tokenizer: "a photograph of a cat" = 6 tokens)
comptime N_TXT_COND = 6
comptime N_TXT_UNCOND = 1
comptime S_COND = N_IMG + N_TXT_COND     # 4102
comptime S_UNCOND = N_IMG + N_TXT_UNCOND # 4097

comptime NUM_STEPS = 30
comptime CFG_SCALE = Float32(4.0)
comptime SHIFT = Float32(3.0)
comptime SEED = UInt64(42)

comptime LH = ERNIE_LATENT_H  # 64
comptime LW = ERNIE_LATENT_W  # 64


# ── Stats helper ───────────────────────────────────────────────────────────────
def _stats(name: String, t: Tensor, ctx: DeviceContext) raises:
    var h = t.to_host(ctx)
    var n = len(h)
    var s = 0.0
    var s2 = 0.0
    var amax = 0.0
    for i in range(n):
        var v = Float64(h[i])
        s += v
        s2 += v * v
        var av = v if v >= 0.0 else -v
        if av > amax:
            amax = av
    var mean = s / Float64(n)
    var var_ = s2 / Float64(n) - mean * mean
    if var_ < 0.0:
        var_ = 0.0
    print(
        "  [stat]", name,
        "mean=", Float32(mean),
        "std=", Float32(sqrt(var_)),
        "absmax=", Float32(amax),
    )


# ── Clone a device tensor (fresh copy) ────────────────────────────────────────
def _clone(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    var dev = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
    ctx.enqueue_copy(dst_buf=dev, src_buf=x.buf)
    ctx.synchronize()
    return Tensor(dev^, x.shape(), x.dtype())


# ── adaln_chunk: [1, 6*H] -> one of 6 [H] chunks ─────────────────────────────
def _adaln_chunk(adaln: Tensor, idx: Int, ctx: DeviceContext) raises -> Tensor:
    var part = slice(adaln, 1, idx * ERNIE_DIT_HIDDEN, ERNIE_DIT_HIDDEN, ctx)
    var sh = List[Int]()
    sh.append(ERNIE_DIT_HIDDEN)
    return reshape(part, sh^, ctx)


# ── reshape helpers ────────────────────────────────────────────────────────────
def _to_bshd[S_: Int](x: Tensor, ctx: DeviceContext) raises -> Tensor:
    var sh = List[Int]()
    sh.append(1)
    sh.append(S_)
    sh.append(ERNIE_DIT_HEADS)
    sh.append(ERNIE_DIT_HEAD_DIM)
    return reshape(x, sh^, ctx)


def _from_bshd[S_: Int](x: Tensor, ctx: DeviceContext) raises -> Tensor:
    var sh = List[Int]()
    sh.append(1)
    sh.append(S_)
    sh.append(ERNIE_DIT_HIDDEN)
    return reshape(x, sh^, ctx)


# ── ones / zeros BF16 tensors for no-affine layer_norm ────────────────────────
def _ones_bf16(n: Int, ctx: DeviceContext) raises -> Tensor:
    var vals = List[Float32]()
    for _ in range(n):
        vals.append(1.0)
    var sh = List[Int]()
    sh.append(n)
    return Tensor.from_host(vals, sh^, STDtype.BF16, ctx)


def _zeros_bf16(n: Int, ctx: DeviceContext) raises -> Tensor:
    var vals = List[Float32]()
    for _ in range(n):
        vals.append(0.0)
    var sh = List[Int]()
    sh.append(n)
    return Tensor.from_host(vals, sh^, STDtype.BF16, ctx)


# ── Single ERNIE block forward (BlockLoader dict) ─────────────────────────────
# Mirrors ernie_image.rs::block_forward_from_map exactly.
def _ernie_block_forward[S_: Int](
    x: Tensor,
    block: Dict[String, ArcPointer[Tensor]],
    block_prefix: String,
    shift_msa: Tensor,
    scale_msa: Tensor,
    gate_msa: Tensor,
    shift_mlp: Tensor,
    scale_mlp: Tensor,
    gate_mlp: Tensor,
    rope_cos: Tensor,
    rope_sin: Tensor,
    ctx: DeviceContext,
) raises -> Tensor:
    var p = block_prefix

    # ── Self-attention (AdaLN-Zero) ─────────────────────────────────────────
    var sa_norm = rms_norm(x, block[p + ".adaLN_sa_ln.weight"][], 1.0e-6, ctx)
    var sa_in = modulate(sa_norm, scale_msa, shift_msa, ctx)

    var q = _to_bshd[S_](
        linear(sa_in, block[p + ".self_attention.to_q.weight"][], None, ctx), ctx
    )
    var k = _to_bshd[S_](
        linear(sa_in, block[p + ".self_attention.to_k.weight"][], None, ctx), ctx
    )
    var v = _to_bshd[S_](
        linear(sa_in, block[p + ".self_attention.to_v.weight"][], None, ctx), ctx
    )

    # QK RMSNorm per head
    q = rms_norm(q, block[p + ".self_attention.norm_q.weight"][], 1.0e-6, ctx)
    k = rms_norm(k, block[p + ".self_attention.norm_k.weight"][], 1.0e-6, ctx)

    # RoPE
    q = rope_halfsplit_full(q, rope_cos, rope_sin, ctx)
    k = rope_halfsplit_full(k, rope_cos, rope_sin, ctx)

    # SDPA (math-mode; flash FAILS on sm_86 for Dh=128)
    var att = sdpa_nomask[1, S_, ERNIE_DIT_HEADS, ERNIE_DIT_HEAD_DIM](
        q, k, v, Float32(1.0) / sqrt(Float32(ERNIE_DIT_HEAD_DIM)), ctx
    )
    var att_flat = _from_bshd[S_](att, ctx)
    var att_out = linear(
        att_flat, block[p + ".self_attention.to_out.0.weight"][], None, ctx
    )
    var h = residual_gate(x, gate_msa, att_out, ctx)

    # ── GELU-gated FFN (no bias) ─────────────────────────────────────────────
    var mlp_norm = rms_norm(h, block[p + ".adaLN_mlp_ln.weight"][], 1.0e-6, ctx)
    var mlp_in = modulate(mlp_norm, scale_mlp, shift_mlp, ctx)
    var gate = linear(mlp_in, block[p + ".mlp.gate_proj.weight"][], None, ctx)
    var up_ = linear(mlp_in, block[p + ".mlp.up_proj.weight"][], None, ctx)
    var activated = mul(gelu(gate, ctx), up_, ctx)
    var mlp_out = linear(
        activated, block[p + ".mlp.linear_fc2.weight"][], None, ctx
    )
    return residual_gate(h, gate_mlp, mlp_out, ctx)


# ── Full ERNIE forward (36 blocks streamed via BlockLoader) ───────────────────
# S_ = N_IMG_ + N_TXT_ (comptime). Mirrors ErnieImageSwapped::forward.
def _ernie_forward[S_: Int, N_IMG_: Int, N_TXT_: Int](
    latent_nchw: Tensor,         # [1, 128, LH, LW] BF16
    text_cond: Tensor,           # [1, N_TXT_, 4096] BF16 (projected)
    text_len_real: Int,          # real token count (RoPE axis-0 offset)
    timestep: Tensor,            # [1] F32
    resident: ErnieImageResident,
    loader: BlockLoader,
    ctx: DeviceContext,
) raises -> Tensor:
    # 1. Patch embed -> image tokens [1, N_IMG_, 4096]
    var img_tokens = resident.patch_embed_1024(latent_nchw, ctx)

    # 2. Concatenate image + text tokens -> [1, S_, 4096]
    var x = concat(1, ctx, img_tokens, text_cond)

    # 3. Timestep embedding + shared AdaLN modulation
    var temb = resident.time_embed(timestep, ctx)
    var adaln = resident.shared_adaln(temb, ctx)

    # Extract 6 AdaLN chunks [4096] each
    var shift_msa = _adaln_chunk(adaln, 0, ctx)
    var scale_msa = _adaln_chunk(adaln, 1, ctx)
    var gate_msa  = _adaln_chunk(adaln, 2, ctx)
    var shift_mlp = _adaln_chunk(adaln, 3, ctx)
    var scale_mlp = _adaln_chunk(adaln, 4, ctx)
    var gate_mlp  = _adaln_chunk(adaln, 5, ctx)

    # 4. Build RoPE tables for this call's sequence length S_
    var rope = build_ernie_rope_tables[N_IMG_, N_TXT_, ERNIE_DIT_HEADS, ERNIE_DIT_HEAD_DIM](
        LH, LW, text_len_real, ctx, STDtype.BF16
    )
    var rope_cos = _clone(rope[0], ctx)
    var rope_sin = _clone(rope[1], ctx)

    # 5. 36-block streaming loop
    for i in range(ERNIE_DIT_LAYERS):
        var prefix = String("layers.") + String(i)
        loader.prefetch_block(prefix)
        var blk = loader.load_block(prefix, ctx)
        x = _ernie_block_forward[S_](
            x, blk, prefix,
            shift_msa, scale_msa, gate_msa,
            shift_mlp, scale_mlp, gate_mlp,
            rope_cos, rope_sin,
            ctx,
        )
        unload_block(blk^)

    # 6. Final AdaLN: final_norm.linear -> scale + shift; then no-affine LN
    # Rust: final_mod = temb @ final_norm.linear.weightᵀ + bias -> [1, 8192]
    #       f_scale = final_mod[:, :4096]; f_shift = final_mod[:, 4096:]
    #       x_norm = layer_norm(x, no_affine)
    #       x_out  = x_norm * (1 + f_scale) + f_shift
    var fn_w = _clone(resident._w(String("final_norm.linear.weight")), ctx)
    var fn_b = _clone(resident._w(String("final_norm.linear.bias")), ctx)
    var final_mod = linear(temb, fn_w, Optional[Tensor](fn_b^), ctx)

    # Split [1, 8192] -> two [4096] tensors
    var f_scale_raw = slice(final_mod, 1, 0, ERNIE_DIT_HIDDEN, ctx)
    var f_shift_raw = slice(final_mod, 1, ERNIE_DIT_HIDDEN, ERNIE_DIT_HIDDEN, ctx)
    var fsc_sh = List[Int]()
    fsc_sh.append(ERNIE_DIT_HIDDEN)
    var f_scale_1d = reshape(f_scale_raw, fsc_sh.copy(), ctx)
    var f_shift_1d = reshape(f_shift_raw, fsc_sh^, ctx)

    # LayerNorm (no affine: ones/zeros)
    var ones = _ones_bf16(ERNIE_DIT_HIDDEN, ctx)
    var zeros = _zeros_bf16(ERNIE_DIT_HIDDEN, ctx)
    var x_norm = layer_norm(x, ones, zeros, 1.0e-6, ctx)
    # Modulate with learned final scale/shift
    var x_modulated = modulate(x_norm, f_scale_1d, f_shift_1d, ctx)

    # 7. Final linear: [1, S_, 4096] -> [1, S_, 128]
    var fl_w = _clone(resident._w(String("final_linear.weight")), ctx)
    var fl_b = _clone(resident._w(String("final_linear.bias")), ctx)
    var patches = linear(x_modulated, fl_w, Optional[Tensor](fl_b^), ctx)

    # 8. Extract image tokens [1, N_IMG_, 128]
    var img_out = slice(patches, 1, 0, N_IMG_, ctx)

    # 9. Unpatchify (patch_size=1): [1, N_IMG_, 128] -> [1, 128, LH, LW]
    var up_sh = List[Int]()
    up_sh.append(1)
    up_sh.append(LH)
    up_sh.append(LW)
    up_sh.append(ERNIE_LATENT_CHANNELS)
    var nhwc = reshape(img_out, up_sh^, ctx)
    var perm = List[Int]()
    perm.append(0)
    perm.append(3)
    perm.append(1)
    perm.append(2)
    return permute(nhwc, perm^, ctx)


# ── Load text embeddings from sidecar ─────────────────────────────────────────
def load_text_conditioning(
    resident: ErnieImageResident, ctx: DeviceContext
) raises -> Tuple[Tensor, Tensor]:
    """Load cond/uncond from sidecar, cast F32->BF16, project 3072->4096."""
    print("[text] loading sidecar embeddings from", EMBEDDINGS_DIR)
    var st = ShardedSafeTensors.open(String(EMBEDDINGS_DIR))

    var cond_f32 = Tensor.from_view(st.tensor_view(String("context_cond")), ctx)
    var uncond_f32 = Tensor.from_view(st.tensor_view(String("context_uncond")), ctx)

    var cond_bf16 = cast_tensor(cond_f32, STDtype.BF16, ctx)
    var uncond_bf16 = cast_tensor(uncond_f32, STDtype.BF16, ctx)

    # Project [1, 256, 3072] -> [1, 256, 4096]
    var cond_proj = resident.project_text(cond_bf16, ctx)
    var uncond_proj = resident.project_text(uncond_bf16, ctx)

    var cs = cond_proj.shape()
    var us = uncond_proj.shape()
    print("  cond:", cs[0], cs[1], cs[2], "| uncond:", us[0], us[1], us[2])
    return (cond_proj^, uncond_proj^)


# ── Load final_norm + final_linear weights into resident struct ────────────────
def load_final_weights(
    mut resident: ErnieImageResident, loader: BlockLoader, ctx: DeviceContext
) raises:
    var extra_names = List[String]()
    extra_names.append(String("final_norm.linear.weight"))
    extra_names.append(String("final_norm.linear.bias"))
    extra_names.append(String("final_linear.weight"))
    extra_names.append(String("final_linear.bias"))

    for i in range(len(extra_names)):
        var nm = extra_names[i]
        var tv = loader.sharded.tensor_view(nm)
        var t = Tensor.from_view(tv, ctx)
        var idx = len(resident.weights)
        resident.weights.append(ArcPointer(t^))
        resident.name_to_idx[nm] = idx
    print("  final weights loaded:", len(extra_names), "tensors")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() raises:
    var ctx = DeviceContext()
    print("=== ERNIE-Image 1024x1024 Mojo pipeline ===")
    print("  steps:", NUM_STEPS, " cfg:", CFG_SCALE, " shift:", SHIFT)

    # Stage 1: Load resident weights (patch_embed, timestep_mlp, text_proj, adaln)
    print("\n[1/4] Loading ERNIE resident weights...")
    var resident = ErnieImageResident.load_default(ctx)
    print("  resident:", len(resident.weights), "tensors")

    # Stage 2: Open BlockLoader + load final weights into resident
    print("\n[2/4] Opening block loader and loading final weights...")
    var loader = BlockLoader.open(String(ERNIE_TRANSFORMER_DIR))
    load_final_weights(resident, loader, ctx)

    # Stage 3: Text conditioning from sidecar
    print("\n[3/4] Loading text conditioning...")
    var text_conds = load_text_conditioning(resident, ctx)
    var cond_full = _clone(text_conds[0], ctx)    # [1, 256, 4096] BF16
    var uncond_full = _clone(text_conds[1], ctx)  # [1, 256, 4096] BF16

    # Trim to real token lengths (matches Rust sequential CFG behavior)
    var cond_trim = slice(cond_full, 1, 0, N_TXT_COND, ctx)     # [1, 6, 4096]
    var uncond_trim = slice(uncond_full, 1, 0, N_TXT_UNCOND, ctx) # [1, 1, 4096]

    # Stage 4: Denoise loop
    print("\n[4/4] Denoising (", NUM_STEPS, " steps, CFG", CFG_SCALE, ")...")
    var sched = ErnieFlowMatchScheduler(NUM_STEPS, SHIFT)

    var noise_sh = List[Int]()
    noise_sh.append(1)
    noise_sh.append(ERNIE_LATENT_CHANNELS)
    noise_sh.append(LH)
    noise_sh.append(LW)
    var latent = randn(noise_sh^, SEED, STDtype.BF16, ctx)
    _stats("initial_noise", latent, ctx)

    for step in range(NUM_STEPS):
        var sigma = sched.sigma(step)
        var sigma_next = sched.sigma(step + 1)
        var dt = sigma_next - sigma   # negative
        var t_val = ernie_model_timestep_from_sigma(sigma)

        print("  step", step + 1, "/", NUM_STEPS, "sigma=", sigma, "t=", t_val)

        var t_list = List[Float32]()
        t_list.append(t_val)
        var t_sh = List[Int]()
        t_sh.append(1)
        var timestep = Tensor.from_host(t_list, t_sh^, STDtype.F32, ctx)

        # Sequential CFG: cond forward then uncond forward
        var v_cond = _ernie_forward[S_COND, N_IMG, N_TXT_COND](
            latent, cond_trim, N_TXT_COND, timestep, resident, loader, ctx
        )
        var v_uncond = _ernie_forward[S_UNCOND, N_IMG, N_TXT_UNCOND](
            latent, uncond_trim, N_TXT_UNCOND, timestep, resident, loader, ctx
        )

        # CFG/Euler tensor ops use F32 arithmetic internally and store BF16.
        var pred = ernie_cfg(v_cond, v_uncond, CFG_SCALE, ctx)

        # Euler step
        latent = ernie_euler_step(latent, pred, dt, ctx)

    _stats("final_latent", latent, ctx)

    # Stage 5: VAE decode
    # KleinVaeDecoder accepts BF16 packed latents; inverse-BN/unpack use F32
    # arithmetic internally and return the latent storage dtype.
    print("\n[5/5] VAE decode...")
    var vae = KleinVaeDecoder[LH, LW].load(String(ERNIE_VAE_FILE), ctx)
    var img = vae.decode(latent, ctx)
    var sh = img.shape()
    print("  decoded:", sh[0], sh[1], sh[2], sh[3])
    _stats("decoded_image", img, ctx)

    save_png(img, String(OUT_PATH), ctx, ValueRange.SIGNED)
    print("\n[done] saved", OUT_PATH)
