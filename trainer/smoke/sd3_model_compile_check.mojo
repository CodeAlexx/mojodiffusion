# sd3_model_compile_check.mojo - compile gate for SD3/SD3.5 model surface.
#
# Compile only after the build lock is clear:
#   cd trainer && \
#     timeout 180 prlimit --as=24000000000 \
#       /home/alex/mojodiffusion/.pixi/envs/default/bin/mojo build \
#       -I /home/alex/mojodiffusion -I src \
#       smoke/sd3_model_compile_check.mojo \
#       -o /tmp/sd3_model_compile_check
#
# This is not a parity gate. It uses tiny synthetic tensors only to instantiate
# SD3 model metadata, shape helpers, and dtype-preserving VAE scale/shift.

from max.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenity_trainer.model.StableDiffusion3Model import (
    StableDiffusion3ImageShape,
    StableDiffusion3Model,
    SD3_PROMPT_MAX_LENGTH,
    SD3_POOLED_PROMPT_HIDDEN_SIZE,
    stable_diffusion_3_cfg_latent_model_input_shape,
    stable_diffusion_3_component_names,
    stable_diffusion_3_image_to_latent_shape,
    stable_diffusion_3_model_types,
    stable_diffusion_3_pipeline_component_names,
    stable_diffusion_3_prompt_embedding_shape,
    stable_diffusion_3_pooled_prompt_embedding_shape,
    stable_diffusion_3_runtime_unsupported_items,
    stable_diffusion_3_scheduler_timestep_contract,
    stable_diffusion_3_text_encode_contract,
    stable_diffusion_3_text_encoder_dropout_supported,
    stable_diffusion_3_text_encoder_output_shape,
    stable_diffusion_3_transformer_latents_from_vae_input,
    stable_diffusion_3_transformer_uses_latent_input_scaling,
    stable_diffusion_3_vae_decode_input,
)


def _expect_int(name: String, got: Int, expected: Int) raises:
    if got != expected:
        raise Error(
            name + String(": got ") + String(got)
            + String(", expected ") + String(expected)
        )


def _expect_bool(name: String, got: Bool, expected: Bool) raises:
    if got != expected:
        raise Error(
            name + String(": got ") + String(got)
            + String(", expected ") + String(expected)
        )


def _expect_string(name: String, got: String, expected: String) raises:
    if got != expected:
        raise Error(name + String(": got ") + got + String(", expected ") + expected)


def main() raises:
    var model = StableDiffusion3Model(String("STABLE_DIFFUSION_35"))
    model.has_tokenizer_1 = True
    model.has_tokenizer_2 = True
    model.has_tokenizer_3 = True
    model.has_noise_scheduler = True
    model.has_text_encoder_1 = True
    model.has_text_encoder_2 = True
    model.has_text_encoder_3 = True
    model.has_vae = True
    model.has_transformer = True
    model.has_text_encoder_1_lora = True
    model.has_text_encoder_2_lora = True
    model.has_text_encoder_3_lora = True
    model.has_transformer_lora = True
    model.has_embedding = True
    model.additional_embedding_count = 2
    model.to(String("compile-device"))
    model.eval()

    var adapters = model.adapters()
    var pipe = model.create_pipeline()
    _expect_int("adapter count", len(adapters), 4)
    _expect_string("adapter te1", adapters[0], String("text_encoder_1"))
    _expect_string("adapter te2", adapters[1], String("text_encoder_2"))
    _expect_string("adapter te3", adapters[2], String("text_encoder_3"))
    _expect_string("adapter transformer", adapters[3], String("transformer"))
    _expect_bool("pipe transformer", pipe.has_transformer, True)
    _expect_bool("pipe scheduler", pipe.has_scheduler, True)
    _expect_bool("pipe vae", pipe.has_vae, True)
    _expect_bool("pipe te1", pipe.has_text_encoder_1, True)
    _expect_bool("pipe tok1", pipe.has_tokenizer_1, True)
    _expect_bool("pipe te2", pipe.has_text_encoder_2, True)
    _expect_bool("pipe tok2", pipe.has_tokenizer_2, True)
    _expect_bool("pipe te3", pipe.has_text_encoder_3, True)
    _expect_bool("pipe tok3", pipe.has_tokenizer_3, True)
    _expect_string("vae device", model.vae_device, String("compile-device"))
    _expect_string("te1 device", model.text_encoder_1_device, String("compile-device"))
    _expect_string("te2 device", model.text_encoder_2_device, String("compile-device"))
    _expect_string("te3 device", model.text_encoder_3_device, String("compile-device"))
    _expect_string("transformer device", model.transformer_device, String("compile-device"))
    _expect_bool("eval called", model.eval_called, True)
    _expect_bool("transformer eval", model.transformer_eval_called, True)
    print(
        "sd3 adapters =", len(adapters),
        " pipeline has transformer =", pipe.has_transformer,
    )
    _expect_int("prompt max", SD3_PROMPT_MAX_LENGTH, 77)
    _expect_int("pooled hidden", SD3_POOLED_PROMPT_HIDDEN_SIZE, 2048)
    _expect_bool("sd3.5 predicate", model.is_stable_diffusion_3_5(), True)
    _expect_int("embedding count", model.all_embeddings_count(), 3)
    print(
        "sd3 prompt max =", SD3_PROMPT_MAX_LENGTH,
        " pooled hidden =", SD3_POOLED_PROMPT_HIDDEN_SIZE,
        " is sd3.5 =", model.is_stable_diffusion_3_5(),
    )
    print("sd3 embeddings =", model.all_embeddings_count())

    var model_types = stable_diffusion_3_model_types()
    var components = stable_diffusion_3_component_names()
    var pipe_components = stable_diffusion_3_pipeline_component_names()
    var unsupported = stable_diffusion_3_runtime_unsupported_items()
    _expect_int("model type count", len(model_types), 2)
    _expect_string("model type 0", model_types[0], String("STABLE_DIFFUSION_3"))
    _expect_string("model type 1", model_types[1], String("STABLE_DIFFUSION_35"))
    _expect_int("component count", len(components), 9)
    _expect_string("component tokenizer 1", components[0], String("tokenizer_1"))
    _expect_string("component scheduler", components[3], String("noise_scheduler"))
    _expect_string("component transformer", components[8], String("transformer"))
    _expect_int("pipeline component count", len(pipe_components), 9)
    _expect_string("pipeline transformer", pipe_components[0], String("transformer"))
    _expect_string("pipeline tokenizer 3", pipe_components[8], String("tokenizer_3"))
    _expect_int("unsupported count", len(unsupported), 8)
    _expect_string("unsupported transformer", unsupported[4], String("SD3Transformer2DModel MMDiT forward"))
    print(
        "sd3 model types =", len(model_types),
        " components =", len(components),
        " pipeline components =", len(pipe_components),
        " unsupported =", len(unsupported),
    )

    var text_contract = stable_diffusion_3_text_encode_contract(2, 4096)
    var te1_shape = stable_diffusion_3_text_encoder_output_shape(1, 2, 4096)
    var te3_shape = stable_diffusion_3_text_encoder_output_shape(3, 2, 4096)
    var prompt_shape = stable_diffusion_3_prompt_embedding_shape(2, 4096)
    var pooled_shape = stable_diffusion_3_pooled_prompt_embedding_shape(2)
    _expect_int("text contract batch", text_contract.batch_size, 2)
    _expect_int("text contract tokenizer length", text_contract.tokenizer_1_max_length, 77)
    _expect_int("text contract te1 hidden", text_contract.text_encoder_1_hidden_size, 768)
    _expect_int("text contract te2 hidden", text_contract.text_encoder_2_hidden_size, 1280)
    _expect_int("text contract te3 hidden", text_contract.text_encoder_3_hidden_size, 4096)
    _expect_int("text contract prompt seq", text_contract.prompt_embedding_seq_length, 154)
    _expect_bool("text dropout support", text_contract.dropout_supported, True)
    _expect_int("te1 rank", len(te1_shape), 3)
    _expect_int("te1 hidden", te1_shape[2], 768)
    _expect_int("te3 hidden", te3_shape[2], 4096)
    _expect_int("prompt seq", prompt_shape[1], 154)
    _expect_int("prompt hidden", prompt_shape[2], 4096)
    _expect_int("pooled hidden", pooled_shape[1], 2048)
    print(
        "sd3 text batch =", text_contract.batch_size,
        " te1 hidden =", te1_shape[2],
        " te3 hidden =", te3_shape[2],
        " prompt seq =", prompt_shape[1],
        " pooled hidden =", pooled_shape[1],
    )
    _ = stable_diffusion_3_text_encoder_dropout_supported(Float32(0.25))

    var image_shape = StableDiffusion3ImageShape(1, 3, 1024, 1024)
    var latent_shape = stable_diffusion_3_image_to_latent_shape(image_shape, 16, 8)
    var cfg_shape = stable_diffusion_3_cfg_latent_model_input_shape(latent_shape)
    var timestep_contract = stable_diffusion_3_scheduler_timestep_contract(4, 1)
    _expect_int("latent batch", latent_shape.batch, 1)
    _expect_int("latent channels", latent_shape.channels, 16)
    _expect_int("latent height", latent_shape.height, 128)
    _expect_int("latent width", latent_shape.width, 128)
    _expect_int("cfg rank", len(cfg_shape), 4)
    _expect_int("cfg batch", cfg_shape[0], 2)
    _expect_int("cfg channels", cfg_shape[1], 16)
    _expect_int("timestep count", timestep_contract.timesteps_count, 4)
    _expect_int("timestep expanded length", timestep_contract.expanded_timestep_length, 2)
    _expect_bool("transformer latent scaling", stable_diffusion_3_transformer_uses_latent_input_scaling(), False)
    print(
        "sd3 latent c =", latent_shape.channels,
        " h =", latent_shape.height,
        " cfg batch =", cfg_shape[0],
        " timesteps =", timestep_contract.timesteps_count,
        " transformer scales latents =",
        stable_diffusion_3_transformer_uses_latent_input_scaling(),
    )

    var ctx = DeviceContext()
    var vals = List[Float32]()
    for i in range(16 * 2 * 2):
        vals.append(Float32(i % 17) * 0.01)
    var latent_dims = List[Int]()
    latent_dims.append(1)
    latent_dims.append(16)
    latent_dims.append(2)
    latent_dims.append(2)
    var latent = Tensor.from_host(vals^, latent_dims^, STDtype.BF16, ctx)

    var decode_input = stable_diffusion_3_vae_decode_input(
        latent, Float32(1.5305), Float32(0.0609), ctx
    )
    var transformer_latents = stable_diffusion_3_transformer_latents_from_vae_input(
        decode_input, Float32(1.5305), Float32(0.0609), ctx
    )
    print(
        "sd3 latent dtype =", transformer_latents.dtype().name(),
        " rank =", len(transformer_latents.shape()),
    )
    _expect_string("transformer latent dtype", transformer_latents.dtype().name(), String("BF16"))
    _expect_int("transformer latent rank", len(transformer_latents.shape()), 4)
    print("SD3 MODEL COMPILE-CHECK OK")
