# anima_model_compile_check.mojo - compile gate for AnimaModel build-only surface.
#
# Compile only after the build lock is clear:
#   cd trainer && \
#     pixi run mojo build \
#       -I /home/alex/mojodiffusion -I src \
#       smoke/anima_model_compile_check.mojo \
#       -o /tmp/anima_model_compile_check
#
# This is not a parity gate. It uses tiny synthetic tensors only to instantiate
# Anima latent scale/unscale helpers and model-core metadata.

from max.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenity_trainer.model.AnimaModel import (
    AnimaImageShape,
    AnimaModel,
    AnimaSchedulerShiftConfig,
    ANIMA_PROMPT_MAX_LENGTH,
    ANIMA_TEXT_CONDITIONER_HIDDEN_SIZE,
    ANIMA_TEXT_CONDITIONER_SEQ_LENGTH,
    anima_component_names,
    anima_image_to_latent_shape,
    anima_latent_to_image_shape,
    anima_runtime_unsupported_items,
    anima_text_encode_contract,
    anima_text_encoder_dropout_supported,
    anima_transformer_block_key_renames,
    calculate_timestep_shift,
    diffusers_checkpoint_to_original,
    scale_latents,
    unscale_latents,
)


def main() raises:
    var model = AnimaModel()
    model.has_tokenizer = True
    model.has_t5_tokenizer = True
    model.has_noise_scheduler = True
    model.has_text_encoder = True
    model.has_text_conditioner = True
    model.has_vae = True
    model.has_transformer = True
    model.has_text_encoder_lora = True
    model.has_transformer_lora = True
    model.to(String("compile-device"))
    model.eval()

    var adapters = model.adapters()
    var pipe = model.create_pipeline()
    print("anima adapters =", len(adapters), " pipeline has vae =", pipe.has_vae)
    print(
        "anima prompt max =", ANIMA_PROMPT_MAX_LENGTH,
        " text seq =", ANIMA_TEXT_CONDITIONER_SEQ_LENGTH,
        " text hidden =", ANIMA_TEXT_CONDITIONER_HIDDEN_SIZE,
    )

    var components = anima_component_names()
    var top_keys = diffusers_checkpoint_to_original()
    var block_keys = anima_transformer_block_key_renames()
    var unsupported = anima_runtime_unsupported_items()
    print(
        "anima components =", len(components),
        " top rename rows =", len(top_keys),
        " block rename rows =", len(block_keys),
        " unsupported =", len(unsupported),
    )

    var contract = anima_text_encode_contract(2)
    print(
        "anima text batch =", contract.batch_size,
        " out seq =", contract.output_seq_length,
        " hidden =", contract.output_hidden_size,
    )
    _ = anima_text_encoder_dropout_supported(Float32(0.0))

    var image_shape = AnimaImageShape(1, 3, 32, 32)
    var latent_meta = anima_image_to_latent_shape(image_shape, 16, 8, 1)
    var image_roundtrip = anima_latent_to_image_shape(latent_meta, 3, 8)
    print(
        "anima latent meta c =", latent_meta.channels,
        " h =", latent_meta.height,
        " image h =", image_roundtrip.height,
    )

    var shift_cfg = AnimaSchedulerShiftConfig(
        256, 4096, Float32(0.5), Float32(1.15)
    )
    print("anima shift =", calculate_timestep_shift(4, 4, shift_cfg))

    var ctx = DeviceContext()
    var vals = List[Float32]()
    for i in range(16 * 1 * 2 * 2):
        vals.append(Float32(i % 17) * 0.01)
    var latent_shape = List[Int]()
    latent_shape.append(1)
    latent_shape.append(16)
    latent_shape.append(1)
    latent_shape.append(2)
    latent_shape.append(2)
    var latent = Tensor.from_host(vals^, latent_shape^, STDtype.BF16, ctx)

    var mean = List[Float32]()
    var std_values = List[Float32]()
    for i in range(16):
        mean.append(Float32(i) * 0.001)
        std_values.append(Float32(1.0) + Float32(i) * 0.01)
    var scaled = scale_latents(latent, mean^, std_values^, ctx)

    var mean2 = List[Float32]()
    var std_values2 = List[Float32]()
    for i in range(16):
        mean2.append(Float32(i) * 0.001)
        std_values2.append(Float32(1.0) + Float32(i) * 0.01)
    var unscaled = unscale_latents(scaled, mean2^, std_values2^, ctx)
    print(
        "anima latent dtype =", unscaled.dtype().name(),
        " rank =", len(unscaled.shape()),
    )
    print("ANIMA MODEL COMPILE-CHECK OK")
