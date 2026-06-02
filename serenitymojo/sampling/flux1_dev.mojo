# sampling/flux1_dev.mojo - FLUX.1-dev schedule and packed-latent contracts.
#
# Host-side scalar helpers ported from inference-flame
# `src/sampling/flux1_sampling.rs`. Tensor pack/unpack remains in the pipeline
# because it is a GPU layout operation; this module owns the dimensions and
# schedule scalars that those GPU operations must obey.

from std.math import exp


def _ceil_div(a: Int, b: Int) raises -> Int:
    if a <= 0:
        raise Error("FLUX.1 ceil_div: numerator must be > 0")
    if b <= 0:
        raise Error("FLUX.1 ceil_div: denominator must be > 0")
    return (a + b - 1) // b


def flux1_mu(image_seq_len: Int) raises -> Float64:
    """BFL FLUX.1 linear `mu`: 0.5 @ 256 tokens, 1.15 @ 4096 tokens."""
    if image_seq_len <= 0:
        raise Error("flux1_mu: image_seq_len must be > 0")
    var x1: Float64 = 256.0
    var y1: Float64 = 0.5
    var x2: Float64 = 4096.0
    var y2: Float64 = 1.15
    var m = (y2 - y1) / (x2 - x1)
    var b = y1 - m * x1
    return m * Float64(image_seq_len) + b


def flux1_time_shift(mu: Float64, t: Float64) -> Float64:
    """BFL exponential time shift with sigma parameter fixed at 1."""
    if t <= 0.0 or t >= 1.0:
        return t
    var em = exp(mu)
    return em / (em + (1.0 / t - 1.0))


def build_flux1_sigma_schedule(
    num_steps: Int, image_seq_len: Int
) raises -> List[Float32]:
    """Build `num_steps + 1` descending FLUX.1 timesteps/sigmas.

    The endpoints stay exact: `out[0] == 1.0`, `out[-1] == 0.0`.
    Interior values are `linspace(1,0,N+1)` passed through `time_shift`.
    """
    if num_steps <= 0:
        raise Error("build_flux1_sigma_schedule: num_steps must be > 0")
    var mu = flux1_mu(image_seq_len)
    var out = List[Float32]()
    var denom = Float64(num_steps)
    for i in range(num_steps + 1):
        var t = 1.0 - Float64(i) / denom
        out.append(Float32(flux1_time_shift(mu, t)))
    return out^


def flux1_euler_dt(current_t: Float32, next_t: Float32) -> Float32:
    """FLUX.1 Euler delta for `img = img + (next_t - current_t) * pred`."""
    return next_t - current_t


def flux1_packed_spatial_dim(image_dim: Int) raises -> Int:
    """Packed grid dimension: `ceil(image_dim / 16)`."""
    return _ceil_div(image_dim, 16)


def flux1_latent_spatial_dim(image_dim: Int) raises -> Int:
    """VAE latent spatial dimension before 2x2 patch packing."""
    return 2 * flux1_packed_spatial_dim(image_dim)


struct Flux1PackedLatentPlan(Copyable, Movable):
    var width: Int
    var height: Int
    var text_tokens: Int
    var latent_channels: Int
    var patch_size: Int
    var latent_h: Int
    var latent_w: Int
    var packed_h: Int
    var packed_w: Int
    var image_tokens: Int
    var packed_channels: Int
    var total_sequence: Int

    def __init__(out self, width: Int, height: Int, text_tokens: Int) raises:
        if text_tokens <= 0:
            raise Error("Flux1PackedLatentPlan: text_tokens must be > 0")
        self.width = width
        self.height = height
        self.text_tokens = text_tokens
        self.latent_channels = 16
        self.patch_size = 2
        self.packed_h = flux1_packed_spatial_dim(height)
        self.packed_w = flux1_packed_spatial_dim(width)
        self.latent_h = self.packed_h * self.patch_size
        self.latent_w = self.packed_w * self.patch_size
        self.image_tokens = self.packed_h * self.packed_w
        self.packed_channels = self.latent_channels * self.patch_size * self.patch_size
        self.total_sequence = self.text_tokens + self.image_tokens

    def validate_dev_1024_contract(self) raises:
        if self.width != 1024 or self.height != 1024:
            raise Error("FLUX.1-dev contract currently targets 1024x1024")
        if self.text_tokens != 512:
            raise Error("FLUX.1-dev contract expects 512 T5 tokens")
        if self.latent_h != 128 or self.latent_w != 128:
            raise Error("FLUX.1-dev latent grid must be 128x128 before packing")
        if self.packed_h != 64 or self.packed_w != 64:
            raise Error("FLUX.1-dev packed grid must be 64x64")
        if self.image_tokens != 4096:
            raise Error("FLUX.1-dev image token count must be 4096")
        if self.packed_channels != 64:
            raise Error("FLUX.1-dev packed channel count must be 64")
        if self.total_sequence != 4608:
            raise Error("FLUX.1-dev total sequence must be 4608")


def build_flux1_packed_latent_plan(
    width: Int, height: Int, text_tokens: Int
) raises -> Flux1PackedLatentPlan:
    return Flux1PackedLatentPlan(width, height, text_tokens)


struct Flux1DevScheduler(Movable):
    var _sigmas: List[Float32]
    var num_steps: Int
    var image_seq_len: Int
    var mu: Float64

    def __init__(out self, num_steps: Int, image_seq_len: Int) raises:
        self._sigmas = build_flux1_sigma_schedule(num_steps, image_seq_len)
        self.num_steps = num_steps
        self.image_seq_len = image_seq_len
        self.mu = flux1_mu(image_seq_len)

    def sigmas(self) -> List[Float32]:
        return self._sigmas.copy()

    def timestep(self, i: Int) raises -> Float32:
        if i < 0 or i >= self.num_steps:
            raise Error("Flux1DevScheduler.timestep: step out of range")
        return self._sigmas[i]

    def dt(self, i: Int) raises -> Float32:
        if i < 0 or i >= self.num_steps:
            raise Error("Flux1DevScheduler.dt: step out of range")
        return flux1_euler_dt(self._sigmas[i], self._sigmas[i + 1])
