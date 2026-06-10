# sampling/ltx2_res2s_ref.mojo — FAITHFUL res_2s audio-video denoising loop.
#
# 1:1 port of `res2s_audio_video_denoising_loop` from
# /home/alex/LTX-2/packages/ltx-pipelines/src/ltx_pipelines/utils/samplers.py:199-433
# for the two-modality (video+audio both present) HQ case, plus its helpers:
#   * sigma tail rewrite  [..., 0]  ->  [..., 0.0011, 0.0]   (samplers.py:268-270)
#   * hs = -log(sigmas[1:]/sigmas[:-1]) per-step h            (samplers.py:272)
#   * get_res2s_coefficients / phi  (res2s.py)  — REUSED from
#     serenitymojo/sampling/ltx2_sampling.mojo `res2s_coefficients`, verified
#     line-by-line identical to the reference formula (a21 = c2*phi1(-h*c2),
#     b2 = phi2(-h)/c2, b1 = phi1(-h)-b2, Taylor 1/j! near 0).
#   * substep SDE injection (separate generator, eta locked 0.5)
#   * bongmath anchor refinement (only when h < 0.5 AND sigma > 0.03)
#   * stage-2 eval at sub_sigma = sqrt(sigma*sigma_next) WITH the injected noise
#   * final combine  x_anchor + h*(b1*eps1 + b2*eps2)
#   * step-level SDE injection (eta 0.5)
#   * final denoise pass when the original sigmas end in 0
#   * Res2sDiffusionStep.step / get_sde_coeff — REUSED from ltx2_sampling.mojo
#     `res2s_sde_step` / `res2s_sde_coeffs` (verified faithful, eta=0.5 form).
#   * post_process_latent masked blend (helpers.py:257-259) — identity for the
#     T2V all-ones denoise mask, implemented + guarded for when image
#     conditioning lands.
#
# *** LATENT LAYOUT CONTRACT ***
# The loop carries latents in the PATCHIFIED token layout the reference loop
# itself operates in (DiffusionStage patchifies before the loop and
# unpatchifies after — blocks.py:344-388):
#   video [1, S_V, 128]   ("b c f h w -> b (f h w) c", patch_size 1)
#   audio [1, S_A, 128]   ("b c t f  -> b t (c f)")
# All sampler arithmetic is element-wise so the layout only matters for the
# production noise normalization (channel-wise over dims (-2,-1) of the
# PATCHIFIED tensor) and for the fixture/dump tensor shapes — carrying the
# patchified layout makes both byte-identical to the reference.
#
# *** PRECISION DEVIATION (documented) ***
# The reference anchors x_anchor/eps/x_mid/x_next in torch.float64 on GPU.
# F64 GPU tensors are not available in this stack; tensor math runs in F32
# device tensors instead, while ALL h/coefficient SCALAR math stays Float64 on
# host (matching the reference's .double()/python-float coefficient path).
# States are stored BF16 between steps exactly like the reference
# (`.to(model_dtype)`), so the F32-vs-F64 deviation only affects the intra-step
# combine arithmetic — covered by the per-step parity gate (cos >= 0.995/step).
#
# *** STRUCTURE DEVIATION (documented) ***
# The reference passes `denoiser`/`transformer` callables into the loop. Mojo's
# closure support makes a heavyweight model-state closure impractical, so the
# loop is generic over a comptime `denoise` function parameter
# (`fn (Tensor, Tensor, Float32) raises capturing [_] -> Tuple[Tensor, Tensor]`):
# the pipeline instantiates it with a nested @parameter def that captures the
# model state and returns the (post-guidance) DENOISED x0 estimates for
# (video, audio) at the given sigma — i.e. exactly what
# `denoiser(...)`+`post_process_latent` produce per eval in the reference.
#
# *** NOISE SOURCE ***
# `NoiseSource` abstracts the reference's seeded torch.Generators:
#   * fixture mode — loads a safetensors of pre-recorded noises (written by
#     scripts/ltx2_hq_ref_run.py in consumption order) keyed
#     `{prefix}_sub{step:02d}_{video|audio}` (substep generator draws) and
#     `{prefix}_stp{step:02d}_{video|audio}` (step generator draws); prefix is
#     "s1"/"s2" per stage. Used for parity runs (removes RNG from the gate).
#   * production mode — Mojo randn + the reference `_get_new_noise`
#     normalization (global mean/std normalize, then channel-wise normalize
#     over dims (-2,-1)); rng-contract: mojo-native-not-pytorch-parity.

from std.gpu.host import DeviceContext
from std.math import sqrt, log

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.random import randn
from serenitymojo.ops.tensor_algebra import add, sub, mul, mul_scalar
from serenitymojo.sampling.ltx2_sampling import (
    res2s_coefficients,
    res2s_substep,
    res2s_combine,
    res2s_sde_step,
    res2s_bong_refine,
    res2s_bong_active,
)
from std.memory import ArcPointer


# ── post_process_latent (helpers.py:257-259) ─────────────────────────────────
def res2s_post_process_latent(
    denoised: Tensor, denoise_mask: Tensor, clean: Tensor, ctx: DeviceContext
) raises -> Tensor:
    """denoised*mask + clean*(1-mask). Identity for the T2V all-ones mask.

    clean*(1-mask) is expanded to clean - clean*mask (algebraically identical)
    to stay on the existing elementwise op surface."""
    var dm = mul(denoised, denoise_mask, ctx)
    var cm = sub(clean, mul(clean, denoise_mask, ctx), ctx)
    return add(dm, cm, ctx)


def _post_process_opt(
    d: Tensor,
    m: Optional[Tensor],
    c: Optional[Tensor],
    ctx: DeviceContext,
) raises -> Tensor:
    """post_process_latent guarded on an optional (mask, clean) pair.

    T2V (no conditioning) passes None/None -> identity (samplers.py applies
    post_process_latent with an all-ones mask there, same value)."""
    if m.__bool__() and c.__bool__():
        return res2s_post_process_latent(d, m.value(), c.value(), ctx)
    return d.clone(ctx)


# ── sigma tail rewrite (samplers.py:266-270) ─────────────────────────────────
comptime RES2S_REF_TERMINAL_SIGMA = Float32(0.0011)


struct Res2sRefSchedule(Movable):
    """The rewritten sigma table + loop bounds.

    n_full_steps = len(original) - 1 (computed BEFORE the rewrite, exactly like
    the reference). When the original table ends in 0 the tail is rewritten to
    [..., 0.0011, 0.0] and `final_denoise` is True: after the loop one extra
    model eval at sigmas[n_full_steps] (= 0.0011) replaces the latents with the
    denoised estimates."""

    var sigmas: List[Float32]
    var n_full_steps: Int
    var final_denoise: Bool

    def __init__(out self, var sigmas: List[Float32], n_full_steps: Int, final_denoise: Bool):
        self.sigmas = sigmas^
        self.n_full_steps = n_full_steps
        self.final_denoise = final_denoise


def res2s_ref_rewrite_sigmas(sigmas: List[Float32]) raises -> Res2sRefSchedule:
    if len(sigmas) < 2:
        raise Error("res2s_ref_rewrite_sigmas: need >= 2 sigmas")
    var n_full_steps = len(sigmas) - 1
    var out = List[Float32]()
    var final_denoise = sigmas[len(sigmas) - 1] == Float32(0.0)
    if final_denoise:
        for i in range(len(sigmas) - 1):
            out.append(sigmas[i])
        out.append(RES2S_REF_TERMINAL_SIGMA)
        out.append(Float32(0.0))
    else:
        out = sigmas.copy()
    return Res2sRefSchedule(out^, n_full_steps, final_denoise)


# ── NoiseSource ───────────────────────────────────────────────────────────────
struct NoiseSource(Movable):
    var is_fixture: Bool
    var fixture_path: String
    var seed: UInt64
    var counter: UInt64

    def __init__(out self, is_fixture: Bool, var fixture_path: String, seed: UInt64):
        self.is_fixture = is_fixture
        self.fixture_path = fixture_path^
        self.seed = seed
        self.counter = UInt64(0)

    @staticmethod
    def fixture(path: String) raises -> NoiseSource:
        return NoiseSource(True, path.copy(), UInt64(0))

    @staticmethod
    def production(seed: UInt64) -> NoiseSource:
        var empty = String("")
        return NoiseSource(False, empty^, seed)

    def has_key(self, key: String) raises -> Bool:
        if not self.is_fixture:
            return False
        var st = ShardedSafeTensors.open(self.fixture_path)
        for ref nm in st.names():
            if nm == key:
                return True
        return False

    def load_key_f32(self, key: String, ctx: DeviceContext) raises -> Tensor:
        """Load an arbitrary fixture tensor (e.g. init latents) as F32."""
        var st = ShardedSafeTensors.open(self.fixture_path)
        return Tensor.from_view_as_f32(st.tensor_view(key), ctx)

    def draw(
        mut self, key: String, var shape: List[Int], ctx: DeviceContext
    ) raises -> Tensor:
        """One SDE noise draw (F32).

        Fixture mode: loads `key` (recorded post-`_get_new_noise`
        normalization, so the value is EXACTLY what the reference stepper
        consumed). Production mode: Mojo randn + the `_get_new_noise`
        normalization (samplers.py:160-166): global (x-mean)/std, then
        channel-wise normalize over dims (-2,-1) — on the patchified
        [1, S, C] layout this loop carries, exactly like the reference."""
        if self.is_fixture:
            var t = self.load_key_f32(key, ctx)
            var tsh = t.shape()
            var n_want = 1
            for i in range(len(shape)):
                n_want *= shape[i]
            var n_got = 1
            for i in range(len(tsh)):
                n_got *= tsh[i]
            if n_got != n_want:
                raise Error(
                    String("NoiseSource: fixture noise '") + key
                    + "' element count mismatch"
                )
            return t^
        # Production: deterministic Mojo randn (seed+counter), then normalize.
        self.counter += 1
        var raw = randn(shape^, self.seed + self.counter, STDtype.F32, ctx)
        return _get_new_noise_normalize(raw, ctx)


def _get_new_noise_normalize(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    """Port of `_get_new_noise` normalization (samplers.py:160-166).

    noise = (noise - noise.mean()) / noise.std()   # global, unbiased std
    then `_channelwise_normalize`: subtract mean / divide std over dims
    (-2,-1) keepdim per batch row. With batch=1 patchified [1,S,C] both
    reductions span the same S*C elements; both are applied anyway. Host F64
    accumulation (tensors here are <= a few hundred KB)."""
    var sh = x.shape()
    if len(sh) != 3 or sh[0] != 1:
        raise Error("_get_new_noise_normalize: expects patchified [1,S,C]")
    var h = x.to_host(ctx)
    var n = len(h)

    def _norm(mut v: List[Float32]):
        var s = 0.0
        var s2 = 0.0
        for i in range(len(v)):
            var f = Float64(v[i])
            s += f
            s2 += f * f
        var mean = s / Float64(len(v))
        var var_ = (s2 - Float64(len(v)) * mean * mean) / Float64(len(v) - 1)
        if var_ < 0.0:
            var_ = 0.0
        var inv = 1.0 / sqrt(var_)
        for i in range(len(v)):
            v[i] = Float32((Float64(v[i]) - mean) * inv)

    _norm(h)   # global normalize
    _norm(h)   # channelwise over (-2,-1) == same span at batch=1
    _ = n
    return Tensor.from_host(h, x.shape(), STDtype.F32, ctx)


# ── the loop ──────────────────────────────────────────────────────────────────
def res2s_ref_loop[
    denoise: def (Tensor, Tensor, Float32) raises capturing [_] -> Tuple[
        Tensor, Tensor
    ]
](
    sigmas_in: List[Float32],
    var video_x: Tensor,            # [1, S_V, 128] BF16 (patchified state)
    var audio_x: Tensor,            # [1, S_A, 128] BF16
    mut ns: NoiseSource,
    dump_prefix: String,            # "s1" / "s2" — fixture/dump key prefix
    mut dump_names: List[String],
    mut dump_tensors: List[ArcPointer[Tensor]],
    ctx: DeviceContext,
    bongmath: Bool = True,
    bong_max_iter: Int = 100,
    video_denoise_mask: Optional[Tensor] = None,   # patchified mask (None = all-ones T2V)
    video_clean_latent: Optional[Tensor] = None,
    audio_denoise_mask: Optional[Tensor] = None,
    audio_clean_latent: Optional[Tensor] = None,
) raises -> Tuple[Tensor, Tensor]:
    """res2s_audio_video_denoising_loop (samplers.py:199-433), two modalities.

    `denoise(video_x_bf16, audio_x_bf16, sigma)` must return the post-guidance
    DENOISED x0 estimates (BF16) for both modalities — the exact tensors the
    reference's `denoiser(...)` + X0Model produce per eval.

    Appends `{dump_prefix}_s{step:02d}_video/_audio` (state AFTER the
    step-level SDE injection, BF16-rounded then stored F32) to the dump lists.
    Returns the final (video, audio) BF16 states."""
    var sched = res2s_ref_rewrite_sigmas(sigmas_in)
    ref sigmas = sched.sigmas

    for step in range(sched.n_full_steps):
        var sigma = sigmas[step]
        var sigma_next = sigmas[step + 1]

        # F32 anchors (reference: .clone().double() — see precision deviation).
        var x_anchor_v = cast_tensor(video_x, STDtype.F32, ctx)
        var x_anchor_a = cast_tensor(audio_x, STDtype.F32, ctx)

        # ── STAGE 1: evaluate at current point (samplers.py:286-297) ──
        var r1 = denoise(video_x, audio_x, sigma)
        var den_v1 = cast_tensor(
            _post_process_opt(r1[0], video_denoise_mask, video_clean_latent, ctx),
            STDtype.F32, ctx,
        )
        var den_a1 = cast_tensor(
            _post_process_opt(r1[1], audio_denoise_mask, audio_clean_latent, ctx),
            STDtype.F32, ctx,
        )

        # h / RK coefficients / sub_sigma — Float64 host scalar math.
        # res2s_coefficients == get_res2s_coefficients (phi cache irrelevant:
        # pure function of (j, neg_h)). h = log(sigma/sigma_next) ==
        # -log(sigma_next/sigma) (samplers.py:272).
        var c = res2s_coefficients(sigma, sigma_next)

        # ── substep x via a21 (samplers.py:306-320) ──
        var x_mid_v = res2s_substep(x_anchor_v, den_v1, c.h, c.a21, ctx)
        var x_mid_a = res2s_substep(x_anchor_a, den_a1, c.h, c.a21, ctx)

        # ── SDE noise injection at substep (samplers.py:322-340):
        # stepper.step(sample=x_anchor, denoised=x_mid,
        #              sigmas=[sigma, sub_sigma], eta=0.5), video then audio. ──
        var sub_key_v = dump_prefix + "_sub" + _pad2(step) + "_video"
        var sub_key_a = dump_prefix + "_sub" + _pad2(step) + "_audio"
        x_mid_v = res2s_sde_step(
            x_anchor_v, x_mid_v, Float64(sigma), Float64(c.sub_sigma),
            ns.draw(sub_key_v, video_x.shape(), ctx), ctx,
        )
        x_mid_a = res2s_sde_step(
            x_anchor_a, x_mid_a, Float64(sigma), Float64(c.sub_sigma),
            ns.draw(sub_key_a, audio_x.shape(), ctx), ctx,
        )

        # ── bong iteration (samplers.py:342-352): refine x_anchor from the
        # NOISED x_mid; eps_1 tracks the refined anchor. Gate: h<0.5, σ>0.03. ──
        if res2s_bong_active(c.h, sigma, bongmath):
            x_anchor_v = res2s_bong_refine(
                x_anchor_v, x_mid_v, den_v1, c.h, c.a21, bong_max_iter, ctx
            )
            x_anchor_a = res2s_bong_refine(
                x_anchor_a, x_mid_a, den_a1, c.h, c.a21, bong_max_iter, ctx
            )

        # ── STAGE 2: evaluate at substep point WITH noise (samplers.py:354-379)
        # x_mid cast to model dtype (bf16) exactly like the reference. ──
        var x_mid_v_b = cast_tensor(x_mid_v, STDtype.BF16, ctx)
        var x_mid_a_b = cast_tensor(x_mid_a, STDtype.BF16, ctx)
        var r2 = denoise(x_mid_v_b, x_mid_a_b, c.sub_sigma)
        var den_v2 = cast_tensor(
            _post_process_opt(r2[0], video_denoise_mask, video_clean_latent, ctx),
            STDtype.F32, ctx,
        )
        var den_a2 = cast_tensor(
            _post_process_opt(r2[1], audio_denoise_mask, audio_clean_latent, ctx),
            STDtype.F32, ctx,
        )

        # ── FINAL COMBINATION (samplers.py:381-394):
        # eps_i = den_i - x_anchor(refined); x_next = x_anchor + h(b1 e1 + b2 e2)
        var x_next_v = res2s_combine(
            x_anchor_v, den_v1, den_v2, c.h, c.b1, c.b2, ctx
        )
        var x_next_a = res2s_combine(
            x_anchor_a, den_a1, den_a2, c.h, c.b1, c.b2, ctx
        )

        # ── SDE injection at STEP level (samplers.py:396-413):
        # stepper.step(sample=x_anchor, denoised=x_next,
        #              sigmas=full schedule @ step_idx, eta=0.5). ──
        var stp_key_v = dump_prefix + "_stp" + _pad2(step) + "_video"
        var stp_key_a = dump_prefix + "_stp" + _pad2(step) + "_audio"
        x_next_v = res2s_sde_step(
            x_anchor_v, x_next_v, Float64(sigma), Float64(sigma_next),
            ns.draw(stp_key_v, video_x.shape(), ctx), ctx,
        )
        x_next_a = res2s_sde_step(
            x_anchor_a, x_next_a, Float64(sigma), Float64(sigma_next),
            ns.draw(stp_key_a, audio_x.shape(), ctx), ctx,
        )

        # State update: .to(model_dtype) (samplers.py:415-419).
        video_x = cast_tensor(x_next_v, STDtype.BF16, ctx)
        audio_x = cast_tensor(x_next_a, STDtype.BF16, ctx)

        # Dump the post-injection (BF16-rounded) state, stored F32.
        dump_names.append(dump_prefix + "_s" + _pad2(step) + "_video")
        dump_tensors.append(
            ArcPointer[Tensor](cast_tensor(video_x, STDtype.F32, ctx))
        )
        dump_names.append(dump_prefix + "_s" + _pad2(step) + "_audio")
        dump_tensors.append(
            ArcPointer[Tensor](cast_tensor(audio_x, STDtype.F32, ctx))
        )
        print(
            "  [res2s_ref]", dump_prefix, "step", step + 1, "/",
            sched.n_full_steps, " sigma=", sigma, "->", sigma_next,
            " h=", c.h, " sub_sigma=", c.sub_sigma,
        )

    # ── Final denoise pass (samplers.py:421-433): one more eval at the
    # rewritten terminal sigma (sigmas[n_full_steps] = 0.0011); the state
    # becomes the denoised estimate. ──
    if sched.final_denoise:
        var sigma_f = sigmas[sched.n_full_steps]
        var rf = denoise(video_x, audio_x, sigma_f)
        video_x = cast_tensor(
            _post_process_opt(rf[0], video_denoise_mask, video_clean_latent, ctx),
            STDtype.BF16, ctx,
        )
        audio_x = cast_tensor(
            _post_process_opt(rf[1], audio_denoise_mask, audio_clean_latent, ctx),
            STDtype.BF16, ctx,
        )
        print("  [res2s_ref]", dump_prefix, "final denoise @ sigma=", sigma_f)

    return (video_x^, audio_x^)


def _pad2(n: Int) -> String:
    if n < 10:
        return String("0") + String(n)
    return String(n)
