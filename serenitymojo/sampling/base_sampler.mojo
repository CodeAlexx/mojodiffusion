# sampling/base_sampler.mojo — SHARED sampler parts (the OneTrainer
# BaseModelSampler analogue).
#
# OneTrainer splits sampling into a BaseModelSampler (shared: device staging,
# resolution quantization, output saving) + one <Model>Sampler per architecture
# (independent: the model-specific denoise). We mirror that: this module holds
# the model-AGNOSTIC parts; serenitymojo/sampling/<model>_sampler.mojo holds the
# per-model denoise and the staged load/free.
#
# STAGING (OneTrainer Flux2Sampler.__sample_base): exactly ONE big model is on
# the GPU at a time — text_encoder_to(train) → encode → _to(temp)+gc; then
# transformer_to(train) → denoise → _to(temp)+gc; then vae_to(train) → decode →
# _to(temp)+gc. Mojo has no CPU<->GPU weight shuffle helper, so a model's
# DeviceBuffers are FREED (RAII drop on scope exit) and reloaded from disk when
# next needed — the EDv2 klein_lora_infer discipline (drop(model) before loading
# the VAE). The per-model sampler enforces this by scoping each model load in its
# own function so it frees on return.
#
# Mojo 1.0.0b1: `def` not `fn`.

from std.collections import List
from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.ops.tensor_algebra import permute, reshape, add, mul_scalar
from serenitymojo.image.png import save_png, ValueRange


# OneTrainer BaseModelSampler.quantize_resolution: round to a multiple of `q`.
def quantize_resolution(resolution: Int, quantization: Int) -> Int:
    return ((resolution + quantization // 2) // quantization) * quantization


# Latent grid edge for a pixel resolution: Klein packs res/16 tokens per side
# (vae_scale_factor 8 * patch 2). 512px -> 32, 1024px -> 64.
def latent_grid_edge(resolution: Int) -> Int:
    return resolution // 16


# Initial BF16 latent tokens: NCHW [1,128,LH,LW] randn -> NHWC -> [1, N_IMG, 128].
# Model-agnostic for the Klein/Flux2 latent family. Byte-identical to the
# validation_sampler / multistep-smoke initial_tokens routine.
def initial_tokens[N_IMG: Int, LH: Int, LW: Int](
    seed: UInt64, ctx: DeviceContext
) raises -> Tensor:
    from serenitymojo.ops.random import randn
    from serenitymojo.io.dtype import STDtype
    var nchw_shape = List[Int]()
    nchw_shape.append(1); nchw_shape.append(128); nchw_shape.append(LH); nchw_shape.append(LW)
    var noise_nchw = randn(nchw_shape^, seed, STDtype.BF16, ctx)
    var p = List[Int]()
    p.append(0); p.append(2); p.append(3); p.append(1)
    var nhwc = permute(noise_nchw, p^, ctx)
    var sh = List[Int]()
    sh.append(1); sh.append(N_IMG); sh.append(128)
    return reshape(nhwc, sh^, ctx)


# token NHWC [1,N_IMG,128] -> packed NCHW [1,128,LH,LW] for the VAE decoder.
def tokens_to_packed_nchw[LH: Int, LW: Int](
    tokens: Tensor, ctx: DeviceContext
) raises -> Tensor:
    var nhwc_shape = List[Int]()
    nhwc_shape.append(1); nhwc_shape.append(LH); nhwc_shape.append(LW); nhwc_shape.append(128)
    var nhwc = reshape(tokens, nhwc_shape^, ctx)
    var p = List[Int]()
    p.append(0); p.append(3); p.append(1); p.append(2)
    return permute(nhwc, p^, ctx)


# Direct-velocity Euler step: x = x + dt*pred. Tensor op arithmetic may use F32
# internally, but the latent carrier preserves its storage dtype.
def euler_step(x: Tensor, pred: Tensor, dt: Float32, ctx: DeviceContext) raises -> Tensor:
    return add(x, mul_scalar(pred, dt, ctx), ctx)


# OneTrainer BaseModelSampler.save_sampler_output (image branch): write a PNG.
def save_image(img: Tensor, dest_png: String, ctx: DeviceContext) raises:
    save_png(img, dest_png, ctx, ValueRange.SIGNED)
