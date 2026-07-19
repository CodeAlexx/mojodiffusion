# flux_model_compile_check.mojo - compile gate for FLUX.1 Dev model surface.
#
# Build:
#   cd trainer && \
#     pixi run mojo build -I /home/alex/mojodiffusion -I src \
#       smoke/flux_model_compile_check.mojo \
#       -o /tmp/flux_model_compile_check
#
# This is not a parity, sampling, or training gate. It instantiates only
# build-only metadata and shape helpers from FluxModel.mojo.

from serenity_trainer.model.FluxModel import (
    FluxImageShape,
    FluxModel,
    FluxSchedulerShiftConfig,
    FLUX_NUM_LATENT_CHANNELS,
    FLUX_PROMPT_MAX_LENGTH,
    calculate_timestep_shift,
    flux_adapter_field_names,
    flux_classifier_free_guidance_duplicates_latents,
    flux_component_names,
    flux_dev_1_has_conditioning_image_input,
    flux_dev_1_has_mask_input,
    flux_dev_1_model_types,
    flux_dev_1_sample_defaults,
    flux_flow_target_shape,
    flux_image_to_latent_shape,
    flux_latent_image_ids_shape,
    flux_latent_to_image_shape,
    flux_loader_subfolders,
    flux_lora_save_prefixes,
    flux_pack_latents_shape,
    flux_pipeline_component_names,
    flux_pooled_prompt_embedding_shape,
    flux_prompt_embedding_shape,
    flux_runtime_unsupported_items,
    flux_sample_image_shape,
    flux_scheduler_mu_from_shift,
    flux_scheduler_timestep_contract,
    flux_text_encode_contract,
    flux_text_encoder_1_token_shape,
    flux_text_encoder_2_token_shape,
    flux_text_encoder_dropout_supported,
    flux_text_ids_shape,
    flux_transformer_hidden_states_shape,
    flux_transformer_timestep_input,
    flux_unpack_latents_shape,
)


def main() raises:
    var model = FluxModel()
    model.has_tokenizer_1 = True
    model.has_tokenizer_2 = True
    model.has_noise_scheduler = True
    model.has_text_encoder_1 = True
    model.has_text_encoder_2 = True
    model.has_vae = True
    model.has_transformer = True
    model.has_text_encoder_1_lora = True
    model.has_text_encoder_2_lora = True
    model.has_transformer_lora = True
    model.has_embedding = True
    model.additional_embedding_count = 2
    model.to(String("compile-device"))
    model.eval()

    var adapters = model.adapters()
    var pipe = model.create_pipeline()
    print(
        "flux adapters =", len(adapters),
        " pipeline has transformer =", pipe.has_transformer,
        " is dev =", model.is_flux_dev_1(),
    )
    print(
        "flux prompt max =", FLUX_PROMPT_MAX_LENGTH,
        " latent channels =", FLUX_NUM_LATENT_CHANNELS,
        " embeddings =", model.all_embeddings_count(),
    )

    var model_types = flux_dev_1_model_types()
    var components = flux_component_names()
    var pipe_components = flux_pipeline_component_names()
    var subfolders = flux_loader_subfolders()
    var adapter_fields = flux_adapter_field_names()
    var lora_prefixes = flux_lora_save_prefixes()
    var unsupported = flux_runtime_unsupported_items()
    print(
        "flux model types =", len(model_types),
        " components =", len(components),
        " pipeline components =", len(pipe_components),
        " subfolders =", len(subfolders),
    )
    print(
        "flux adapter fields =", len(adapter_fields),
        " prefixes =", len(lora_prefixes),
        " unsupported =", len(unsupported),
    )

    var text_contract = flux_text_encode_contract(2, 128, 512)
    var te1_tokens = flux_text_encoder_1_token_shape(2)
    var te2_tokens = flux_text_encoder_2_token_shape(2, 128, 512)
    var prompt_shape = flux_prompt_embedding_shape(2, 128, 512)
    var pooled_shape = flux_pooled_prompt_embedding_shape(2)
    var text_ids = flux_text_ids_shape(prompt_shape[1])
    _ = flux_text_encoder_dropout_supported(Float32(0.25))
    print(
        "flux text batch =", text_contract.batch_size,
        " te1 tokens =", te1_tokens[1],
        " te2 tokens =", te2_tokens[1],
        " prompt hidden =", prompt_shape[2],
        " pooled hidden =", pooled_shape[1],
        " text ids rank2 cols =", text_ids[1],
    )

    var defaults = flux_dev_1_sample_defaults()
    var image_shape = flux_sample_image_shape(1, 3, 1025, 1025)
    var latent_shape = flux_image_to_latent_shape(image_shape)
    var packed_shape = flux_pack_latents_shape(latent_shape)
    var unpacked_shape = flux_unpack_latents_shape(
        packed_shape, latent_shape.height, latent_shape.width
    )
    var decoded_shape = flux_latent_to_image_shape(unpacked_shape, 3)
    var image_ids = flux_latent_image_ids_shape(
        latent_shape.height, latent_shape.width
    )
    var transformer_input = flux_transformer_hidden_states_shape(latent_shape)
    var flow_shape = flux_flow_target_shape(latent_shape)
    print(
        "flux sample default =", defaults.width, "x", defaults.height,
        " quantized =", image_shape.width, "x", image_shape.height,
        " latent =", latent_shape.channels, latent_shape.height, latent_shape.width,
    )
    print(
        "flux packed tokens =", packed_shape.tokens,
        " packed channels =", packed_shape.channels,
        " decoded =", decoded_shape.width, "x", decoded_shape.height,
        " image ids =", image_ids[0],
        " transformer tokens =", transformer_input.tokens,
        " flow rank =", len(flow_shape),
    )

    var shift_config = FluxSchedulerShiftConfig(
        256, 4096, Float32(0.5), Float32(1.15)
    )
    var shift = calculate_timestep_shift(
        latent_shape.height, latent_shape.width, shift_config
    )
    var mu = flux_scheduler_mu_from_shift(shift)
    var model_t = flux_transformer_timestep_input(Float32(500.0))
    var scheduler_contract = flux_scheduler_timestep_contract(
        defaults.diffusion_steps, 1
    )
    print(
        "flux shift =", shift,
        " mu =", mu,
        " model_t =", model_t,
        " timesteps =", scheduler_contract.timesteps_count,
        " cfg duplicates =",
        flux_classifier_free_guidance_duplicates_latents(),
    )
    print(
        "flux dev mask =",
        flux_dev_1_has_mask_input(),
        " conditioning =",
        flux_dev_1_has_conditioning_image_input(),
    )
    print("FLUX MODEL COMPILE-CHECK OK")
