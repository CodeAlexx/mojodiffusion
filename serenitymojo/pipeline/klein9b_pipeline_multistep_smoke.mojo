# klein9b_pipeline_multistep_smoke.mojo - Klein 9B real multi-step denoise loop.
#
# Extends the proven one-step smokes (klein9b_pipeline_{64,1024}_smoke.mojo)
# into the real multi-step Euler denoise loop, parity-faithful to the Rust
# reference:
#   * inference-flame/src/sampling/klein_sampling.rs::euler_denoise
#   * inference-flame/src/bin/klein9b_infer.rs (denoise-loop order, CFG, dtype)
#
# Loop reproduced (verified against the Rust source):
#   - sigma schedule  = build_flux2_sigma_schedule(num_steps, N_IMG)  (= Rust get_schedule)
#   - per step i: t_curr=sigmas[i], t_next=sigmas[i+1], dt = t_next - t_curr (< 0)
#   - timestep fed to DiT = raw sigma t_curr as a [1] F32 tensor (NO *1000; the
#     Rust ref and the Mojo t_embedder both consume the raw sigma directly).
#   - CFG = pred_neg + CFG*(pred_pos - pred_neg)  (flux2_cfg). NO post-CFG sign
#     flip (Klein, unlike Z-Image, does not negate).
#   - x = x + dt * pred   (direct-velocity Euler, flux2_euler_step form).
#   - latent x stays F32 across the whole loop; cast to BF16 only to feed the DiT.
#
# This file is NON-DESTRUCTIVE: the one-step smokes are untouched. Set
# NUM_STEPS / the GRID comptimes below to bisect (4 small first, then 1024).
#
# To switch to the native 1024 grid: set LH=LW=64, N_IMG=4096, swap
# Klein9BDiT.load_full -> Klein9BOffloaded.load (see commented block in denoise).

from std.gpu.host import DeviceContext
from std.math import sqrt

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.models.dit.klein_dit import Klein9BDiT, Klein9BOffloaded, build_klein_rope_tables
from serenitymojo.models.vae.klein_decoder import KleinVaeDecoder
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.random import randn
from serenitymojo.ops.tensor_algebra import add, mul_scalar, reshape, permute
from serenitymojo.sampling.flux2_klein import build_flux2_sigma_schedule, flux2_cfg
from serenitymojo.image.png import save_png, ValueRange
from serenitymojo.io.cap_cache import load_tensor_bin

# NOTE: this denoise process deliberately does NOT import Qwen3Encoder /
# Qwen3Tokenizer. The caption embeddings are produced by a SEPARATE process
# (klein9b_encode_smoke.mojo) and read from disk here, so the ~16 GB encoder and
# the Klein 9B DiT never co-reside on the 24 GB card.


comptime KLEIN9B_PATH = "/home/alex/.serenity/models/checkpoints/flux-2-klein-base-9b.safetensors"
comptime VAE_PATH = "/home/alex/.serenity/models/vaes/flux2-vae.safetensors"
comptime OUT = "/home/alex/mojodiffusion/output/klein9b_multistep_64.png"
# Caption embeddings produced by klein9b_encode_smoke.mojo (separate process).
comptime CAPS_POS = "/home/alex/mojodiffusion/output/klein9b_caps_pos.bin"
comptime CAPS_NEG = "/home/alex/mojodiffusion/output/klein9b_caps_neg.bin"

# --- Grid (small/64-token by default; flip to 64/64/4096 for native 1024) ---
comptime N_IMG = 16
comptime N_TXT = 512
comptime S = N_IMG + N_TXT
comptime LH = 4
comptime LW = 4

# --- Sampler knobs ---
comptime NUM_STEPS = 4
comptime CFG = Float32(4.0)
comptime SEED = UInt64(42)


@fieldwise_init
struct KleinCaps(Movable):
    var pos: Tensor
    var neg: Tensor


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
        "  [stat]", name, "mean=", Float32(mean), "std=", Float32(sqrt(var_)),
        "absmax=", Float32(amax), "n=", n,
    )


def load_cached_caps(ctx: DeviceContext) raises -> KleinCaps:
    # Read the two caption embeddings written by klein9b_encode_smoke.mojo.
    # No Qwen3Encoder is loaded here — that process already exited and freed its
    # ~16 GB. Bytes round-trip raw (BF16 in, BF16 out), so these are bit-
    # identical to the in-process embeddings.
    print("[text] loading cached Klein conditioning (no encoder in this process)")
    var pos = load_tensor_bin(CAPS_POS, ctx)
    var neg = load_tensor_bin(CAPS_NEG, ctx)
    var ps = pos.shape()
    print(
        "  pos shape:", ps[0], ps[1], ps[2],
        "dtype.tag", pos.dtype().tag, "nbytes", pos.nbytes(),
    )
    return KleinCaps(pos^, neg^)


def initial_tokens(ctx: DeviceContext) raises -> Tensor:
    # Match the reference layout: draw NCHW [1,128,LH,LW], then pack to
    # token NHWC [1,N_IMG,128]. The draw itself stays GPU-resident.
    var nchw_shape = List[Int]()
    nchw_shape.append(1)
    nchw_shape.append(128)
    nchw_shape.append(LH)
    nchw_shape.append(LW)
    var noise_nchw = randn(nchw_shape^, SEED, STDtype.F32, ctx)
    var p = List[Int]()
    p.append(0)
    p.append(2)
    p.append(3)
    p.append(1)
    var nhwc = permute(noise_nchw, p^, ctx)
    var sh = List[Int]()
    sh.append(1)
    sh.append(N_IMG)
    sh.append(128)
    return reshape(nhwc, sh^, ctx)


def tokens_to_packed_nchw(tokens: Tensor, ctx: DeviceContext) raises -> Tensor:
    var nhwc_shape = List[Int]()
    nhwc_shape.append(1)
    nhwc_shape.append(LH)
    nhwc_shape.append(LW)
    nhwc_shape.append(128)
    var nhwc = reshape(tokens, nhwc_shape^, ctx)
    var p = List[Int]()
    p.append(0)
    p.append(3)
    p.append(1)
    p.append(2)
    return permute(nhwc, p^, ctx)


def denoise(caps: KleinCaps, ctx: DeviceContext) raises -> Tensor:
    # Small/64 grid uses the all-on-GPU full DiT. For the native 1024 grid
    # (LH=LW=64, N_IMG=4096) swap this for:
    #   var model = Klein9BOffloaded.load(KLEIN9B_PATH, ctx)
    # The rest of the loop body is grid-agnostic.
    print("[denoise] loading Klein 9B full DiT")
    var model = Klein9BDiT.load_full(KLEIN9B_PATH, ctx)
    var rope = build_klein_rope_tables[N_IMG, N_TXT, 32, 128](ctx, STDtype.BF16)
    # build_flux2_sigma_schedule returns NUM_STEPS+1 sigmas (1.0 .. 0.0).
    var sigmas = build_flux2_sigma_schedule(NUM_STEPS, N_IMG)
    print("[denoise]", NUM_STEPS, "steps, CFG", CFG, "seed", SEED)
    # x is the latent; kept F32 across the whole loop (cast to BF16 per step
    # only to feed the DiT). x is moved/reassigned each step — never reference
    # a previous binding after `x = add(...)`.
    var x = initial_tokens(ctx)
    _stats("init_tokens", x, ctx)
    for i in range(NUM_STEPS):
        var t_curr = sigmas[i]
        var t_next = sigmas[i + 1]
        var dt = t_next - t_curr  # sigma[i+1] - sigma[i], normally < 0
        print("  step", i + 1, "/", NUM_STEPS, "sigma", t_curr, "->", t_next)
        # Timestep fed to DiT: raw sigma as a [1] F32 tensor (no *1000).
        var tvals = List[Float32]()
        tvals.append(t_curr)
        var tsh = List[Int]()
        tsh.append(1)
        var timestep = Tensor.from_host(tvals, tsh^, STDtype.F32, ctx)
        # BF16 view of the F32 latent for this step's two DiT passes.
        var xb = cast_tensor(x, STDtype.BF16, ctx)
        var pred_pos = cast_tensor(
            model.forward_full[N_IMG, N_TXT, S](
                xb, caps.pos, timestep, rope[0], rope[1], ctx
            ),
            STDtype.F32,
            ctx,
        )
        var pred_neg = cast_tensor(
            model.forward_full[N_IMG, N_TXT, S](
                xb, caps.neg, timestep, rope[0], rope[1], ctx
            ),
            STDtype.F32,
            ctx,
        )
        # CFG: neg + CFG*(pos - neg). NO post-CFG sign flip for Klein.
        var pred = flux2_cfg(pred_pos, pred_neg, CFG, ctx)
        # Direct-velocity Euler: x = x + dt * pred (F32 latent in/out).
        x = add(x, mul_scalar(pred, dt, ctx), ctx)
        _stats("step " + String(i + 1) + " latent", x, ctx)
    return x^


def main() raises:
    var ctx = DeviceContext()
    print("=== Klein 9B multistep smoke -", NUM_STEPS, "steps, grid", LH, "x", LW, "===")
    var caps = load_cached_caps(ctx)
    var tokens = denoise(caps, ctx)
    print("[vae] decode")
    var packed = tokens_to_packed_nchw(tokens, ctx)
    var vae = KleinVaeDecoder[LH, LW].load(VAE_PATH, ctx)
    var img = vae.decode(packed, ctx)
    var sh = img.shape()
    print("  image shape:", sh[0], sh[1], sh[2], sh[3])
    _stats("image", img, ctx)
    save_png(img, OUT, ctx, ValueRange.SIGNED)
    print("[done] saved", OUT)
