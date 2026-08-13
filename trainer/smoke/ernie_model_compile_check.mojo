# ernie_model_compile_check.mojo - compile gate for ErnieModel build-only surface.
#
# Compile only after the build lock is clear:
#   cd trainer && \
#     /home/alex/mojodiffusion/.pixi/envs/default/bin/mojo build \
#       -I /home/alex/mojodiffusion -I src \
#       smoke/ernie_model_compile_check.mojo \
#       -o /tmp/ernie_model_compile_check
#
# This is not a parity gate. It uses tiny synthetic tensors only to instantiate
# Ernie patch/unpatch and VAE scale/unscale helpers.

from max.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenity_trainer.model.ErnieModel import (
    ErnieModel,
    ErnieSchedulerShiftConfig,
    ERNIE_PROMPT_MAX_LENGTH,
    ERNIE_HIDDEN_STATES_LAYER,
    ernie_text_encode_contract,
    ernie_text_encoder_dropout_supported,
    calculate_timestep_shift,
    patchify_latents,
    unpatchify_latents,
    scale_latents,
    unscale_latents,
)


def main() raises:
    var model = ErnieModel()
    model.has_tokenizer = True
    model.has_noise_scheduler = True
    model.has_text_encoder = True
    model.has_vae = True
    model.has_transformer = True
    model.has_transformer_lora = True
    model.to(String("compile-device"))
    model.eval()

    var adapters = model.adapters()
    var pipe = model.create_pipeline()
    print("ernie adapters =", len(adapters), " pipeline has vae =", pipe.has_vae)
    print(
        "ernie prompt max =", ERNIE_PROMPT_MAX_LENGTH,
        " hidden layer =", ERNIE_HIDDEN_STATES_LAYER,
    )

    var lengths = List[Int]()
    lengths.append(31)
    lengths.append(47)
    var contract = ernie_text_encode_contract(lengths^, 4096)
    print(
        "ernie text batch =", contract.batch_size,
        " out seq =", contract.output_seq_length,
        " hidden =", contract.hidden_size,
    )
    _ = ernie_text_encoder_dropout_supported(Float32(0.0))

    var shift_cfg = ErnieSchedulerShiftConfig(
        256, 4096, Float32(0.5), Float32(1.15)
    )
    print("ernie shift =", calculate_timestep_shift(4, 4, shift_cfg))

    var ctx = DeviceContext()
    var vals = List[Float32]()
    for i in range(4 * 4 * 4):
        vals.append(Float32(i % 17) * 0.01)
    var latent_shape = List[Int]()
    latent_shape.append(1)
    latent_shape.append(4)
    latent_shape.append(4)
    latent_shape.append(4)
    var latent = Tensor.from_host(vals^, latent_shape^, STDtype.BF16, ctx)

    var patched = patchify_latents(latent, ctx)
    var unpatched = unpatchify_latents(patched, ctx)

    var mean = List[Float32]()
    var variance = List[Float32]()
    for i in range(4):
        mean.append(Float32(i) * 0.001)
        variance.append(Float32(1.0) + Float32(i) * 0.01)
    var scaled = scale_latents(unpatched, mean^, variance^, Float32(1e-5), ctx)

    var mean2 = List[Float32]()
    var variance2 = List[Float32]()
    for i in range(4):
        mean2.append(Float32(i) * 0.001)
        variance2.append(Float32(1.0) + Float32(i) * 0.01)
    var unscaled = unscale_latents(scaled, mean2^, variance2^, Float32(1e-5), ctx)
    print("ernie latent dtype =", unscaled.dtype().name(), " rank =", len(unscaled.shape()))
    print("ERNIE MODEL COMPILE-CHECK OK")
