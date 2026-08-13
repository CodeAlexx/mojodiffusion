# lens_sampler_gates_smoke.mojo — end-to-end Lens sampler gating vs the Serenity
# trajectory + image oracle (parity/lens/traj_ref.safetensors). Reference policy:
# Serenity ONLY (lens_traj_oracle.py). No Rust.
#
# The oracle runs Serenity's LensSampler denoise math from a FIXED initial packed
# latent (latent0) with FIXED 4-layer GPT-OSS text features (pos+neg), 8 steps,
# cfg 4.0, then decodes with the real Flux2 VAE. It dumps latent0 (packed0),
# latent_final, image, and the features so the Mojo sampler can run on byte-identical
# inputs (NO re-randn, NO re-encode).
#
# This smoke drives the EXACT sampler path sample_lens uses — _predict_flow ->
# cfg_norm_rescale_pair -> lens_euler_step with build_lens_shifted_sigmas — but from
# the injected packed0 + injected features rather than sample_lens's internal
# randn/encode. It TIMES the denoise loop (wall-clock + per-step ms), then decodes
# the final latent with the real LensVAE.
#
#   GATE A (trajectory): cosine(mojo latent_final, oracle latent_final) vs the
#                        DTYPE-AWARE bar = bf16-traj-ceiling(0.998746) - 0.001 margin
#                        = 0.997746, over all 1*1024*128 elements. (cos>=0.999 vs a
#                        bf16 oracle is unachievable — the dtype bad-reference trap.)
#   GATE B (image):      PSNR(mojo decoded image, oracle image) >= 28 dB.
#   OVERALL OK = GATE A and GATE B.
#
# DTYPE: persistent Euler latent F32 (sample_lens contract, LensSampler.py:84); the
# transformer input + text features are cast to BF16 per step (the trained boundary).
# The oracle is f32/bf16 mixed — the 0.999 / 28 dB bars account for the bf16<->f32 gap.

from max.gpu.host import DeviceContext
from std.math import log10, isfinite
from std.time import perf_counter

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.cast import cast_tensor

from serenity_trainer.modelLoader.LensModelLoader import (
    LensWeights, LENS_TRANSFORMER_DIR, LENS_VAE_DIR,
)
from serenity_trainer.model.LensDiT import build_lens_lora_set, LArc
from serenity_trainer.model.LensVAE import LensVAE
from serenity_trainer.modelSetup.BaseLensSetup import unpack_latents
from serenity_trainer.modelSampler.LensSampler import _predict_flow, cfg_norm_rescale_pair
from serenity_trainer.sampling.lens_flowmatch import (
    lens_compute_empirical_mu, build_lens_raw_sigmas, lens_exponential_shift,
    lens_euler_step,
)


comptime TRAJ_REF = "trainer/parity/lens/traj_ref.safetensors"

comptime S_IMG   = 1024      # h_lat * w_lat = 32 * 32  (H=W=512)
comptime S_TXT   = 201       # traj_ref_meta.json S_txt
comptime LAT_HW  = 32        # H//16 = 512//16; unpack target + VAE template spatial
comptime STEPS   = 8
comptime CFG     = Float32(4.0)
comptime LORA_RANK = 8

# GATE A is dtype-aware: the oracle latent_final comes from a bf16 transformer +
# bf16 persistent latent over 8 steps x 2 forwards, so torch's OWN F32 denoise vs
# that bf16 oracle tops out at cos = 0.998746 (the bf16 trajectory ceiling). Asking
# cos>=0.999 against a bf16 oracle is the dtype bad-reference trap (unachievable even
# in F32); the decoded-image PSNR (GATE B) is the stronger end-to-end signal.
comptime TRAJ_CEILING = Float32(0.998746)   # measured F32-vs-bf16-oracle ceiling
comptime TRAJ_MARGIN  = Float32(0.001)
comptime COS_BAR  = TRAJ_CEILING - TRAJ_MARGIN   # 0.997746
comptime PSNR_BAR = Float64(28.0)


def _view_f32(st: ShardedSafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    return Tensor.from_view(st.tensor_view(name), ctx)


def _cosine(a: List[Float32], b: List[Float32]) raises -> Float32:
    if len(a) != len(b):
        raise Error(String("cosine: length mismatch ") + String(len(a))
                    + String(" vs ") + String(len(b)))
    var dot = Float64(0.0)
    var na = Float64(0.0)
    var nb = Float64(0.0)
    for i in range(len(a)):
        dot += Float64(a[i]) * Float64(b[i])
        na += Float64(a[i]) * Float64(a[i])
        nb += Float64(b[i]) * Float64(b[i])
    if na <= 0.0 or nb <= 0.0:
        raise Error("cosine: zero-norm vector")
    return Float32(dot / (na ** 0.5 * nb ** 0.5))


def main() raises:
    var ctx = DeviceContext()
    print("=== Lens sampler end-to-end gates smoke (traj + image oracle) ===")
    print("  reference: Serenity lens_traj_oracle.py (no Rust)")
    print("  S_IMG =", S_IMG, " S_TXT =", S_TXT, " steps =", STEPS, " cfg =", CFG)

    # ── frozen transformer weights (real checkpoint) ──────────────────────────
    print("[weights] loading Lens transformer:", String(LENS_TRANSFORMER_DIR))
    var weights = LensWeights.load(String(LENS_TRANSFORMER_DIR), ctx)
    print("  loaded", weights.count(), "tensors")

    # ── oracle tensors (byte-identical inputs; latent0 + 4-layer pos/neg feats) ─
    var st = ShardedSafeTensors.open(String(TRAJ_REF))
    # Persistent Euler carrier = F32 (sample_lens / LensSampler.py:84, the SAME path).
    # Empirically the cleanest match to the (bf16) oracle: F32 carrier with NO extra
    # state rounding gives cos 0.99820 / image PSNR 42.85 dB. Adding bf16 rounding of
    # the stored state degrades it (cos 0.99404), and a pure bf16 carrier is worse
    # still (cos 0.99177) — mojo's bf16 add accumulates noisier than diffusers'
    # f32-compute-then-round. The transformer input is bf16 each step (via
    # _predict_flow), matching the oracle's bf16 forward boundary.
    var latent = _view_f32(st, String("packed0"), ctx)              # [1,1024,128] F32
    var lsh = latent.shape()
    print("  packed0 (latent0) shape = [", lsh[0], ",", lsh[1], ",", lsh[2], "]")

    # positive / negative 4-layer GPT-OSS features → BF16 forward boundary.
    var pf0 = cast_tensor(_view_f32(st, String("pf_0"), ctx), STDtype.BF16, ctx)  # [1,201,2880]
    var pf1 = cast_tensor(_view_f32(st, String("pf_1"), ctx), STDtype.BF16, ctx)
    var pf2 = cast_tensor(_view_f32(st, String("pf_2"), ctx), STDtype.BF16, ctx)
    var pf3 = cast_tensor(_view_f32(st, String("pf_3"), ctx), STDtype.BF16, ctx)
    var nf0 = cast_tensor(_view_f32(st, String("nf_0"), ctx), STDtype.BF16, ctx)
    var nf1 = cast_tensor(_view_f32(st, String("nf_1"), ctx), STDtype.BF16, ctx)
    var nf2 = cast_tensor(_view_f32(st, String("nf_2"), ctx), STDtype.BF16, ctx)
    var nf3 = cast_tensor(_view_f32(st, String("nf_3"), ctx), STDtype.BF16, ctx)

    # all-valid mask [1, S_TXT] (the oracle uses ones; features pre-padded).
    var mvals = List[Float32]()
    for _ in range(S_TXT):
        mvals.append(Float32(1.0))
    var msh = List[Int]()
    msh.append(1); msh.append(S_TXT)
    var mask = Tensor.from_host(mvals^, msh^, STDtype.F32, ctx)

    # ── schedule (identical to build_lens_shifted_sigmas; empirical mu over S_IMG) ─
    var mu = lens_compute_empirical_mu(S_IMG, STEPS)
    var raw = build_lens_raw_sigmas(STEPS)
    var sigmas = List[Float32]()
    for i in range(len(raw)):
        sigmas.append(lens_exponential_shift(raw[i], mu))
    print("  mu =", mu, " sigma[0] =", sigmas[0], " sigma[N-1] =", sigmas[STEPS - 1])

    # ── B=0 LoRA overlay (identity) ───────────────────────────────────────────
    var loras = build_lens_lora_set(LORA_RANK, Float32(LORA_RANK), ctx)

    # ── denoise loop (sample_lens path; timed) ────────────────────────────────
    print("")
    print("── denoise loop (8 steps, cfg-norm-rescale) ─────────────────────")
    var t_loop0 = perf_counter()
    for i in range(STEPS):
        var sigma_curr = sigmas[i]
        var sigma_next = Float32(0.0)
        if i + 1 < STEPS:
            sigma_next = sigmas[i + 1]

        var t_step0 = perf_counter()
        var flow_cond = _predict_flow[S_IMG, S_TXT](
            latent, sigma_curr, pf0, pf1, pf2, pf3, mask, weights, loras, ctx
        )
        var flow_uncond = _predict_flow[S_IMG, S_TXT](
            latent, sigma_curr, nf0, nf1, nf2, nf3, mask, weights, loras, ctx
        )
        var noise_pred = cfg_norm_rescale_pair(flow_cond, flow_uncond, CFG, ctx)
        latent = lens_euler_step(latent, noise_pred, sigma_curr, sigma_next, ctx)  # F32 accum
        var t_step1 = perf_counter()
        var step_ms = (t_step1 - t_step0) * 1000.0
        print("  step", i, " sigma", sigma_curr, "->", sigma_next, " :", step_ms, "ms")
    var t_loop1 = perf_counter()
    var loop_ms = (t_loop1 - t_loop0) * 1000.0
    var ms_per_step = loop_ms / Float64(STEPS)
    print("  denoise wall-clock      =", loop_ms, "ms  (", ms_per_step, "ms/step )")

    # ── GATE A: trajectory cosine vs oracle latent_final ──────────────────────
    print("")
    print("── GATE A: trajectory cosine (latent_final) ─────────────────────")
    var mojo_lat = latent.to_host(ctx)
    for i in range(len(mojo_lat)):
        if not isfinite(mojo_lat[i]):
            raise Error(String("latent_final non-finite at i=") + String(i))
    var ref_lat = _view_f32(st, String("latent_final"), ctx).to_host(ctx)
    var cos = _cosine(mojo_lat, ref_lat)
    print("  cosine(mojo, oracle)    =", cos)
    print("  bf16 traj ceiling       =", TRAJ_CEILING, " (F32-vs-bf16-oracle)")
    print("  dtype-aware bar         =", COS_BAR, " (ceiling - margin", TRAJ_MARGIN, ")")
    var gate_a = cos >= COS_BAR
    print("  GATE A:", "OK" if gate_a else "FAIL")

    # ── GATE B: decode + PSNR vs oracle image ─────────────────────────────────
    print("")
    print("── GATE B: decoded-image PSNR ───────────────────────────────────")
    var vae = LensVAE[LAT_HW, LAT_HW].load(String(LENS_VAE_DIR), ctx)
    var unpacked = unpack_latents(latent, LAT_HW, LAT_HW, ctx)   # [1,128,32,32]
    var image = vae.decode(unpacked, ctx)                        # [1,3,512,512]
    var ish = image.shape()
    print("  decoded image shape     = [", ish[0], ",", ish[1], ",", ish[2], ",", ish[3], "]")

    var got = image.to_host(ctx)
    var ref_img = _view_f32(st, String("image"), ctx).to_host(ctx)
    if len(got) != len(ref_img):
        raise Error(String("decode length mismatch ") + String(len(got))
                    + String(" vs ") + String(len(ref_img)))

    var mse = Float64(0.0)
    var mad = Float64(0.0)
    var rmin = Float64(ref_img[0])
    var rmax = Float64(ref_img[0])
    for i in range(len(ref_img)):
        var d = Float64(got[i]) - Float64(ref_img[i])
        mse += d * d
        var ad = d
        if ad < 0: ad = -ad
        mad += ad
        var rv = Float64(ref_img[i])
        if rv < rmin: rmin = rv
        if rv > rmax: rmax = rv
    var n = Float64(len(ref_img))
    mse /= n
    mad /= n
    var data_range = rmax - rmin
    var psnr = Float64(999.0)
    if mse > 0.0:
        psnr = 10.0 * log10((data_range * data_range) / mse)
    print("  ref image range         = [", rmin, ",", rmax, "]  (data_range =", data_range, ")")
    print("  mean-abs-diff           =", mad)
    print("  MSE                     =", mse)
    print("  PSNR (dB)               =", psnr, " (bar", PSNR_BAR, ")")
    var gate_b = psnr >= PSNR_BAR
    print("  GATE B:", "OK" if gate_b else "FAIL")

    # ── speed/VRAM summary ────────────────────────────────────────────────────
    print("")
    print("── SPEED ─────────────────────────────────────────────────────────")
    print("  denoise total           =", loop_ms, "ms")
    print("  ms/step                 =", ms_per_step)
    print("  (peak VRAM not queryable in Mojo here; measure with nvidia-smi)")

    print("")
    print("──────────────────────────────────────────────────────────")
    print("  GATE A (trajectory) =", "OK" if gate_a else "FAIL", " cos =", cos)
    print("  GATE B (image)      =", "OK" if gate_b else "FAIL", " PSNR =", psnr)
    var overall = gate_a and gate_b
    print("  OVERALL:", "OK" if overall else "FAIL")
    if not overall:
        raise Error("lens_sampler_gates_smoke GATE FAIL")
