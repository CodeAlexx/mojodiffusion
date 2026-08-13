# ModelSetupFlowMatchingMixin.mojo — FAITHFUL 1:1 port of Serenity
# modules/modelSetup/mixin/ModelSetupFlowMatchingMixin.py.
#
# Source (read in full, 40 lines):
#   class ModelSetupFlowMatchingMixin  (Py L7)
#     __init__                          (Py L9-12)  -> __sigma = None, __one_minus_sigma = None
#     _add_noise_discrete               (Py L14-39)
#
# This file ports `_add_noise_discrete` EXACTLY. Every formula cites its Py line.
#
# Key correction over the prior approximation: sigma is NOT a guessed scalar like
# (t+1)/1000 and NOT passed in. It is INDEXED from a table built once:
#   sigma_table = arange(1, N+1) / N      (Py L23-24)  => sigma[t] = (t+1)/N
#   one_minus_table = 1.0 - sigma_table   (Py L25)
# where N = num_timesteps = timesteps.shape[-1] (Py L22). The active sigma is the
# table lookup sigma_table[timestep] (Py L29). There is NO velocity target here —
# `_add_noise_discrete` only produces the noised latent and returns sigma; the
# training target (e.g. noise - latent) is formed by each Base*Setup, not here.
#
# ── BATCHED per-sample timestep (Py L18, L29-34) — FIXED ──────────────────────
# `timestep` is a Tensor of shape (batch_size,) — one sampled index PER SAMPLE
# (confirmed: ModelSetupNoiseMixin._get_timestep_discrete returns (batch_size,);
# BaseFluxSetup.py:249 passes scaled_latent_image.shape[0]). Therefore
#   sigmas          = self.__sigma[timestep]            (Py L29)  -> shape (batch,)
#   one_minus_sigmas = self.__one_minus_sigma[timestep] (Py L30)  -> shape (batch,)
# and the Py L32-34 loop
#   while sigmas.dim() < scaled_latent_image.dim(): sigmas = sigmas.unsqueeze(-1)
# reshapes the per-sample sigma vector to (batch, 1, 1, ..., 1) so each sample i
# is scaled by its OWN sigma[i] when broadcast over its [C,H,W] latent dims.
# We mirror this EXACTLY: `timestep` is a length-`batch` List[Int] of indices, and
# we build F32 sigma / one_minus_sigma tensors of shape (batch, 1, 1, ...) of the
# same ndim as the latent, then use broadcasting mul/add. A single shared scalar
# (the prior port) is only correct at batch==1; this is the per-sample branch.
#
# ── Dtype handling (FAITHFUL to Py L27, L36-39) — FIXED ───────────────────────
#   orig_dtype = scaled_latent_image.dtype                          (Py L27)
#   scaled_noisy = latent_noise.to(sigmas.dtype) * sigmas
#                + scaled_latent_image.to(sigmas.dtype) * one_minus_sigmas  (Py L36-37)
#   return scaled_noisy.to(orig_dtype), sigmas                      (Py L39)
# torch's `all_timesteps / num_timesteps` is int32 / python-int -> F32, so
# sigmas.dtype is F32. reference trainer therefore casts BOTH latents to F32 FIRST (L36-37),
# does mul/mul/add ENTIRELY in F32, and rounds to BF16 exactly ONCE at L39.
# CRITICAL: the BF16 elementwise paths in tensor_algebra round each product back
# to BF16 per-op (two extra rounding steps reference trainer does not have). So we must:
#   1. cast_tensor(latent, F32)            -> latent_noise.to(F32)   (Py L36)
#   2. cast_tensor(scaled_latent, F32)     -> scaled_latent.to(F32)  (Py L37)
#   3. mul / mul / add  on F32 tensors     -> F32 intermediates, ONE F32 accum
#   4. cast_tensor(result, orig_dtype)     -> .to(orig_dtype)        (Py L39)
# Sigma tensors are F32 (Py L24 promotion), so all multiplies stay in F32 and the
# BF16 store-dtype-per-op rounding never fires inside the math.

from max.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.tensor_algebra import add, mul
from serenitymojo.ops.cast import cast_tensor


# ── sigma table (Py L22-25) ───────────────────────────────────────────────────
# Build the per-timestep sigma table exactly as Serenity's lazy init:
#   num_timesteps = timesteps.shape[-1]                              (Py L22)
#   all_timesteps = arange(start=1, end=N+1, step=1, dtype=int32)    (Py L23)
#   sigma         = all_timesteps / num_timesteps                    (Py L24)  -> F32
#   one_minus     = 1.0 - sigma                                      (Py L25)
# Returned as host F32 lists so a sampled `timestep` can be indexed (Py L29-30).
struct FlowMatchSigmaTable(Copyable, Movable):
    var sigma: List[Float32]            # sigma[t] = (t+1)/N      (Py L24)
    var one_minus_sigma: List[Float32]  # 1.0 - sigma[t]          (Py L25)
    var num_timesteps: Int              # N                       (Py L22)

    def __init__(out self, num_timesteps: Int):
        # arange(1, N+1)/N  => for index i in [0, N): value = (i+1)/N   (Py L23-24)
        self.num_timesteps = num_timesteps
        self.sigma = List[Float32]()
        self.one_minus_sigma = List[Float32]()
        var n_f = Float32(num_timesteps)
        for i in range(num_timesteps):
            # all_timesteps element = i+1 (arange starts at 1)          (Py L23)
            var s = Float32(i + 1) / n_f                                # Py L24
            self.sigma.append(s)
            self.one_minus_sigma.append(Float32(1.0) - s)              # Py L25


# ── _add_noise_discrete (Py L14-39) ───────────────────────────────────────────
# Faithful port. Inputs:
#   scaled_latent_image : clean (scaled) latent, BF16 storage         (Py L16)
#   latent_noise        : Gaussian noise, same shape                  (Py L17)
#   timestep            : per-sample integer indices, len == batch    (Py L18, used L29)
#                         (the (batch_size,) timestep tensor)
#   table               : prebuilt FlowMatchSigmaTable (lazy-init eqv of Py L21-25)
# Returns (scaled_noisy_latent_image, sigmas):
#   sigmas         = sigma_table[timestep]                            (Py L29)
#   one_minus      = one_minus_sigma_table[timestep]                  (Py L30)
#   sigmas/one_minus reshaped to (batch, 1, 1, ...) of latent ndim    (Py L32-34)
#   scaled_noisy   = noise.to(F32)*sigmas + latent.to(F32)*one_minus  (Py L36-37, F32)
#   return scaled_noisy.to(orig_dtype), sigmas                        (Py L39)
struct AddNoiseDiscreteResult(Movable):
    var scaled_noisy_latent_image: Tensor   # Py L36-37 result, cast to orig dtype (Py L39)
    var sigmas: Tensor                       # F32 per-sample sigmas, shape (batch,1,1,...) (Py L29/L32-34, returned L39)

    def __init__(out self, var scaled_noisy_latent_image: Tensor, var sigmas: Tensor):
        self.scaled_noisy_latent_image = scaled_noisy_latent_image^
        self.sigmas = sigmas^


def _add_noise_discrete(
    scaled_latent_image: Tensor,
    latent_noise: Tensor,
    timestep: List[Int],
    table: FlowMatchSigmaTable,
    ctx: DeviceContext,
) raises -> AddNoiseDiscreteResult:
    # orig_dtype = scaled_latent_image.dtype                           (Py L27)
    var orig_dtype = scaled_latent_image.dtype()

    # ── Per-sample sigma lookup (Py L29-30) ──────────────────────────────────
    #   sigmas          = self.__sigma[timestep]                       (Py L29)
    #   one_minus_sigmas = self.__one_minus_sigma[timestep]            (Py L30)
    # `timestep` is the (batch_size,) index tensor; advanced indexing yields a
    # (batch_size,) vector. batch = len(timestep) == scaled_latent_image.shape[0].
    var latent_shape = scaled_latent_image.shape()
    var batch = len(timestep)
    if batch != latent_shape[0]:
        raise Error(
            "_add_noise_discrete: len(timestep)="
            + String(batch)
            + " != latent batch dim="
            + String(latent_shape[0])
            + " (Py L29 advanced index is per-sample over shape[0])"
        )

    var sigma_vals = List[Float32]()
    var one_minus_vals = List[Float32]()
    for b in range(batch):
        var t = timestep[b]
        sigma_vals.append(table.sigma[t])                              # Py L29
        one_minus_vals.append(table.one_minus_sigma[t])                # Py L30

    # ── unsqueeze(-1) to latent ndim (Py L32-34) ─────────────────────────────
    #   while sigmas.dim() < scaled_latent_image.dim(): sigmas.unsqueeze(-1)
    # gives sigmas shape (batch, 1, 1, ..., 1) with ndim == latent ndim, so each
    # sample's sigma broadcasts over its own [C,H,W]. Build that F32 tensor shape.
    var bcast_shape = List[Int]()
    bcast_shape.append(batch)
    for _i in range(len(latent_shape) - 1):
        bcast_shape.append(1)

    # sigmas.dtype is F32 (table promoted to F32 at Py L24). Materialize the
    # per-sample sigma / one_minus_sigma as F32 device tensors of (batch,1,1,...).
    var sigmas = Tensor.from_host(sigma_vals, bcast_shape.copy(), STDtype.F32, ctx)        # Py L29 + L32-34
    var one_minus_sigmas = Tensor.from_host(one_minus_vals, bcast_shape.copy(), STDtype.F32, ctx)  # Py L30 + L32-34

    # ── F32 mix, single BF16 round (Py L36-39) ───────────────────────────────
    #   scaled_noisy = latent_noise.to(F32) * sigmas
    #                + scaled_latent_image.to(F32) * one_minus_sigmas  (Py L36-37)
    # Cast BOTH latents to F32 FIRST (sigmas.dtype == F32), then mul/mul/add stay
    # in F32 (broadcasting the (batch,1,1,...) sigma over (batch,C,H,W)). Exactly
    # one rounding to BF16 happens at the final .to(orig_dtype) (Py L39).
    var noise_f32 = cast_tensor(latent_noise, STDtype.F32, ctx)                 # latent_noise.to(F32)        Py L36
    var latent_f32 = cast_tensor(scaled_latent_image, STDtype.F32, ctx)        # scaled_latent_image.to(F32) Py L37

    var noise_term = mul(noise_f32, sigmas, ctx)                                # noise * sigmas              Py L36 (F32)
    var latent_term = mul(latent_f32, one_minus_sigmas, ctx)                    # latent * one_minus_sigmas   Py L37 (F32)
    var scaled_noisy_f32 = add(noise_term, latent_term, ctx)                    # the +                       Py L36-37 (F32)

    # return scaled_noisy_latent_image.to(dtype=orig_dtype), sigmas    (Py L39)
    var scaled_noisy = cast_tensor(scaled_noisy_f32, orig_dtype, ctx)          # single BF16 round           Py L39

    return AddNoiseDiscreteResult(scaled_noisy^, sigmas^)
