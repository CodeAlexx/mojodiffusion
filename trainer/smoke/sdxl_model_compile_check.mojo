# sdxl_model_compile_check.mojo - build/surface assertion gate for SDXL model surface.
#
# Compile:
#   cd trainer && \
#     timeout 180 prlimit --as=24000000000 pixi run mojo build \
#       -I /home/alex/mojodiffusion -I src \
#       smoke/sdxl_model_compile_check.mojo \
#       -o /tmp/sdxl_model_compile_check
#
# This is not a parity gate. It instantiates SDXL model metadata and shape
# helpers without requiring GPU architecture detection. The BF16 tensor
# round-trip check lives in `smoke/sdxl_model_tensor_contract_check.mojo`.
from serenity_trainer.model.StableDiffusionXLModel import (
    SDXL_COMBINED_TEXT_HIDDEN_SIZE,
    SDXL_POOLED_TEXT_ENCODER_2_HIDDEN_SIZE,
    SDXL_PROMPT_MAX_LENGTH,
    StableDiffusionXLImageShape,
    StableDiffusionXLModel,
    combine_text_encoder_output_shape,
    stable_diffusion_xl_add_time_ids,
    stable_diffusion_xl_cfg_add_time_ids_shape,
    stable_diffusion_xl_cfg_latent_model_input_shape,
    stable_diffusion_xl_cfg_prompt_embedding_shape,
    stable_diffusion_xl_component_names,
    stable_diffusion_xl_image_to_latent_shape,
    stable_diffusion_xl_inpaint_mask_shape,
    stable_diffusion_xl_inpaint_unet_model_input_shape,
    stable_diffusion_xl_loader_subfolders,
    stable_diffusion_xl_lora_conversion_prefixes,
    stable_diffusion_xl_model_types,
    stable_diffusion_xl_pipeline_component_names,
    stable_diffusion_xl_pooled_prompt_embedding_shape,
    stable_diffusion_xl_prompt_embedding_shape,
    stable_diffusion_xl_runtime_unsupported_items,
    stable_diffusion_xl_scheduler_timestep_contract,
    stable_diffusion_xl_text_encode_contract,
    stable_diffusion_xl_text_encoder_dropout_supported,
    stable_diffusion_xl_text_encoder_output_shape,
    stable_diffusion_xl_unet_uses_scheduler_scale_model_input,
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
    var model = StableDiffusionXLModel(
        String("STABLE_DIFFUSION_XL_10_BASE_INPAINTING")
    )
    model.has_tokenizer_1 = True
    model.has_tokenizer_2 = True
    model.has_noise_scheduler = True
    model.has_text_encoder_1 = True
    model.has_text_encoder_2 = True
    model.has_vae = True
    model.has_unet = True
    model.has_text_encoder_1_lora = True
    model.has_text_encoder_2_lora = True
    model.has_unet_lora = True
    model.has_embedding = True
    model.additional_embedding_count = 2
    model.to(String("compile-device"))
    model.eval()
    model.force_v_prediction()
    model.force_epsilon_prediction()
    model.rescale_noise_scheduler_to_zero_terminal_snr()

    var adapters = model.adapters()
    var pipe = model.create_pipeline()
    _expect_bool("model is SDXL", model.is_stable_diffusion_xl(), True)
    _expect_bool("model is SDXL inpainting", model.is_stable_diffusion_xl_inpainting(), True)
    _expect_int("adapter count", len(adapters), 3)
    _expect_string("adapter text encoder 1", adapters[0], String("text_encoder_1"))
    _expect_string("adapter text encoder 2", adapters[1], String("text_encoder_2"))
    _expect_string("adapter unet", adapters[2], String("unet"))
    _expect_bool("pipe vae", pipe.has_vae, True)
    _expect_bool("pipe te1", pipe.has_text_encoder_1, True)
    _expect_bool("pipe te2", pipe.has_text_encoder_2, True)
    _expect_bool("pipe tok1", pipe.has_tokenizer_1, True)
    _expect_bool("pipe tok2", pipe.has_tokenizer_2, True)
    _expect_bool("pipe unet", pipe.has_unet, True)
    _expect_bool("pipe scheduler", pipe.has_scheduler, True)
    _expect_bool("pipe inpainting", pipe.is_inpainting_pipeline, True)
    _expect_string("vae device", model.vae_device, String("compile-device"))
    _expect_string("te1 device", model.text_encoder_1_device, String("compile-device"))
    _expect_string("te2 device", model.text_encoder_2_device, String("compile-device"))
    _expect_string("unet device", model.unet_device, String("compile-device"))
    _expect_string("te1 lora device", model.text_encoder_1_lora_device, String("compile-device"))
    _expect_string("te2 lora device", model.text_encoder_2_lora_device, String("compile-device"))
    _expect_string("unet lora device", model.unet_lora_device, String("compile-device"))
    _expect_bool("eval called", model.eval_called, True)
    _expect_bool("vae eval", model.vae_eval_called, True)
    _expect_bool("te1 eval", model.text_encoder_1_eval_called, True)
    _expect_bool("te2 eval", model.text_encoder_2_eval_called, True)
    _expect_bool("unet eval", model.unet_eval_called, True)
    print(
        "sdxl adapters =", len(adapters),
        " pipeline has inpaint =", pipe.is_inpainting_pipeline,
    )
    _expect_int("prompt max", SDXL_PROMPT_MAX_LENGTH, 77)
    _expect_int("combined hidden", SDXL_COMBINED_TEXT_HIDDEN_SIZE, 2048)
    _expect_int("pooled hidden", SDXL_POOLED_TEXT_ENCODER_2_HIDDEN_SIZE, 1280)
    _expect_int("embedding count", model.all_embeddings_count(), 3)
    _expect_int("te1 embedding count", model.all_text_encoder_1_embeddings_count(), 3)
    _expect_int("te2 embedding count", model.all_text_encoder_2_embeddings_count(), 3)
    _expect_string("scheduler prediction", model.scheduler_prediction_type, String("epsilon"))
    _expect_string("sd config parameterization", model.sd_config_parameterization, String("epsilon"))
    _expect_string("model spec prediction", model.model_spec_prediction_type, String("epsilon"))
    _expect_bool("zero terminal snr", model.zero_terminal_snr_rescaled, True)
    print(
        "sdxl prompt max =", SDXL_PROMPT_MAX_LENGTH,
        " combined hidden =", SDXL_COMBINED_TEXT_HIDDEN_SIZE,
        " pooled hidden =", SDXL_POOLED_TEXT_ENCODER_2_HIDDEN_SIZE,
    )
    print(
        "sdxl embeddings =", model.all_embeddings_count(),
        " prediction =", model.scheduler_prediction_type,
        " zero snr =", model.zero_terminal_snr_rescaled,
    )

    var model_types = stable_diffusion_xl_model_types()
    var components = stable_diffusion_xl_component_names()
    var pipe_components = stable_diffusion_xl_pipeline_component_names()
    var subfolders = stable_diffusion_xl_loader_subfolders()
    var lora_prefixes = stable_diffusion_xl_lora_conversion_prefixes()
    var unsupported = stable_diffusion_xl_runtime_unsupported_items()
    _expect_int("model type count", len(model_types), 2)
    _expect_string("model type base", model_types[0], String("STABLE_DIFFUSION_XL_10_BASE"))
    _expect_string("model type inpaint", model_types[1], String("STABLE_DIFFUSION_XL_10_BASE_INPAINTING"))
    _expect_int("component count", len(components), 7)
    _expect_string("component tokenizer 1", components[0], String("tokenizer_1"))
    _expect_string("component tokenizer 2", components[1], String("tokenizer_2"))
    _expect_string("component scheduler", components[2], String("noise_scheduler"))
    _expect_string("component text encoder 1", components[3], String("text_encoder_1"))
    _expect_string("component text encoder 2", components[4], String("text_encoder_2"))
    _expect_string("component vae", components[5], String("vae"))
    _expect_string("component unet", components[6], String("unet"))
    _expect_int("pipeline component count", len(pipe_components), 7)
    _expect_string("pipeline vae", pipe_components[0], String("vae"))
    _expect_string("pipeline text encoder 1", pipe_components[1], String("text_encoder"))
    _expect_string("pipeline text encoder 2", pipe_components[2], String("text_encoder_2"))
    _expect_string("pipeline tokenizer 1", pipe_components[3], String("tokenizer"))
    _expect_string("pipeline tokenizer 2", pipe_components[4], String("tokenizer_2"))
    _expect_string("pipeline unet", pipe_components[5], String("unet"))
    _expect_string("pipeline scheduler", pipe_components[6], String("scheduler"))
    _expect_int("loader subfolder count", len(subfolders), 7)
    _expect_string("subfolder tokenizer 1", subfolders[0], String("tokenizer"))
    _expect_string("subfolder tokenizer 2", subfolders[1], String("tokenizer_2"))
    _expect_string("subfolder scheduler", subfolders[2], String("scheduler"))
    _expect_string("subfolder text encoder 1", subfolders[3], String("text_encoder"))
    _expect_string("subfolder text encoder 2", subfolders[4], String("text_encoder_2"))
    _expect_string("subfolder vae", subfolders[5], String("vae"))
    _expect_string("subfolder unet", subfolders[6], String("unet"))
    _expect_int("lora conversion prefix count", len(lora_prefixes), 4)
    _expect_string("lora bundle role", lora_prefixes[0], String("bundle_emb"))
    _expect_string("lora unet role", lora_prefixes[1], String("unet"))
    _expect_string("lora clip_l role", lora_prefixes[2], String("clip_l"))
    _expect_string("lora clip_g role", lora_prefixes[3], String("clip_g"))
    _expect_int("unsupported runtime item count", len(unsupported), 8)
    _expect_string("unsupported unet", unsupported[3], String("UNet2DConditionModel forward"))
    print(
        "sdxl model types =", len(model_types),
        " components =", len(components),
        " pipeline components =", len(pipe_components),
        " subfolders =", len(subfolders),
        " lora prefixes =", len(lora_prefixes),
        " unsupported =", len(unsupported),
    )

    var text_contract = stable_diffusion_xl_text_encode_contract(2)
    var te1_shape = stable_diffusion_xl_text_encoder_output_shape(1, 2)
    var te2_shape = stable_diffusion_xl_text_encoder_output_shape(2, 2)
    var combined_shape = combine_text_encoder_output_shape(2)
    var prompt_shape = stable_diffusion_xl_prompt_embedding_shape(2)
    var pooled_shape = stable_diffusion_xl_pooled_prompt_embedding_shape(2)
    var cfg_prompt_shape = stable_diffusion_xl_cfg_prompt_embedding_shape(2)
    _expect_int("text contract batch", text_contract.batch_size, 2)
    _expect_int("tokenizer 1 max length", text_contract.tokenizer_1_max_length, 77)
    _expect_int("tokenizer 2 max length", text_contract.tokenizer_2_max_length, 77)
    _expect_int("te1 default layer", text_contract.text_encoder_1_default_layer, -2)
    _expect_int("te2 default layer", text_contract.text_encoder_2_default_layer, -2)
    _expect_int("te1 hidden", text_contract.text_encoder_1_hidden_size, 768)
    _expect_int("te2 hidden", text_contract.text_encoder_2_hidden_size, 1280)
    _expect_int("pooled text encoder 2 hidden", text_contract.pooled_text_encoder_2_hidden_size, 1280)
    _expect_int("prompt seq", text_contract.prompt_embedding_seq_length, 77)
    _expect_int("prompt hidden", text_contract.prompt_embedding_hidden_size, 2048)
    _expect_bool("output embeddings supported", text_contract.output_embeddings_supported, True)
    _expect_bool("dropout supported", text_contract.dropout_supported, True)
    _expect_bool("attention mask unsupported", text_contract.attention_mask_supported, False)
    _expect_bool("layer norm not added", text_contract.layer_norm_added_by_encode_clip, False)
    _expect_int("te1 rank", len(te1_shape), 3)
    _expect_int("te1 shape batch", te1_shape[0], 2)
    _expect_int("te1 shape seq", te1_shape[1], 77)
    _expect_int("te1 shape hidden", te1_shape[2], 768)
    _expect_int("te2 shape hidden", te2_shape[2], 1280)
    _expect_int("combined rank", len(combined_shape), 3)
    _expect_int("combined hidden", combined_shape[2], 2048)
    _expect_int("prompt hidden", prompt_shape[2], 2048)
    _expect_int("pooled hidden", pooled_shape[1], 1280)
    _expect_int("cfg prompt batch", cfg_prompt_shape[0], 4)
    _expect_int("cfg prompt hidden", cfg_prompt_shape[2], 2048)
    print(
        "sdxl text batch =", text_contract.batch_size,
        " te1 hidden =", te1_shape[2],
        " te2 hidden =", te2_shape[2],
        " combined hidden =", combined_shape[2],
        " prompt seq =", prompt_shape[1],
        " cfg batch =", cfg_prompt_shape[0],
        " pooled hidden =", pooled_shape[1],
    )
    _expect_bool("dropout support function", stable_diffusion_xl_text_encoder_dropout_supported(Float32(0.25)), True)

    var add_time_ids = stable_diffusion_xl_add_time_ids(1024, 768)
    var cfg_time_shape = stable_diffusion_xl_cfg_add_time_ids_shape(1)
    _expect_int("add time original height", add_time_ids.original_height, 1024)
    _expect_int("add time original width", add_time_ids.original_width, 768)
    _expect_int("add time crop top", add_time_ids.crops_coords_top, 0)
    _expect_int("add time crop left", add_time_ids.crops_coords_left, 0)
    _expect_int("add time target height", add_time_ids.target_height, 1024)
    _expect_int("add time target width", add_time_ids.target_width, 768)
    _expect_int("cfg add time rank", len(cfg_time_shape), 2)
    _expect_int("cfg add time rows", cfg_time_shape[0], 2)
    _expect_int("cfg add time cols", cfg_time_shape[1], 6)
    print(
        "sdxl add_time target h =", add_time_ids.target_height,
        " target w =", add_time_ids.target_width,
        " cfg time rows =", cfg_time_shape[0],
    )

    var image_shape = StableDiffusionXLImageShape(1, 3, 1024, 768)
    var latent_shape = stable_diffusion_xl_image_to_latent_shape(
        image_shape, 4, 8
    )
    var cfg_shape = stable_diffusion_xl_cfg_latent_model_input_shape(latent_shape)
    var mask_shape = stable_diffusion_xl_inpaint_mask_shape(latent_shape)
    var inpaint_shape = stable_diffusion_xl_inpaint_unet_model_input_shape(
        latent_shape
    )
    var timestep_contract = stable_diffusion_xl_scheduler_timestep_contract(
        4, 1, True
    )
    _expect_int("latent batch", latent_shape.batch, 1)
    _expect_int("latent channels", latent_shape.channels, 4)
    _expect_int("latent height", latent_shape.height, 128)
    _expect_int("latent width", latent_shape.width, 96)
    _expect_int("cfg latent rank", len(cfg_shape), 4)
    _expect_int("cfg latent batch", cfg_shape[0], 2)
    _expect_int("cfg latent channels", cfg_shape[1], 4)
    _expect_int("mask rank", len(mask_shape), 4)
    _expect_int("mask channels", mask_shape[1], 1)
    _expect_int("inpaint unet rank", len(inpaint_shape), 4)
    _expect_int("inpaint unet batch", inpaint_shape[0], 2)
    _expect_int("inpaint unet channels", inpaint_shape[1], 9)
    _expect_int("timestep diffusion steps", timestep_contract.diffusion_steps, 4)
    _expect_bool("timestep force last", timestep_contract.force_last_timestep, True)
    _expect_int("timestep count min", timestep_contract.timesteps_count_min, 4)
    _expect_int("timestep count max", timestep_contract.timesteps_count_max, 5)
    _expect_int("timestep unet batch", timestep_contract.unet_batch_size, 2)
    _expect_int("timestep add time rows", timestep_contract.add_time_ids_rows, 2)
    _expect_bool("scheduler scales latents", stable_diffusion_xl_unet_uses_scheduler_scale_model_input(), True)
    print(
        "sdxl latent c =", latent_shape.channels,
        " h =", latent_shape.height,
        " cfg batch =", cfg_shape[0],
        " mask c =", mask_shape[1],
        " inpaint c =", inpaint_shape[1],
        " timesteps max =", timestep_contract.timesteps_count_max,
        " scheduler scales latents =",
        stable_diffusion_xl_unet_uses_scheduler_scale_model_input(),
    )

    print("SDXL MODEL COMPILE-CHECK OK")
