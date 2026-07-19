# SDXL data-loader/cache/output contract gate.
#
# Source of truth:
#   /home/alex/Serenity/modules/dataLoader/StableDiffusionXLBaseDataLoader.py

from serenity_trainer.dataLoader.StableDiffusionXLBaseDataLoader import (
    StableDiffusionXLBaseDataLoader,
)


def _expect_int(name: String, got: Int, expected: Int) raises:
    if got != expected:
        raise Error(name + String(": got ") + String(got) + String(", expected ") + String(expected))


def _expect_bool(name: String, got: Bool, expected: Bool) raises:
    if got != expected:
        raise Error(name + String(": got ") + String(got) + String(", expected ") + String(expected))


def _expect_string(name: String, got: String, expected: String) raises:
    if got != expected:
        raise Error(name + String(": got ") + got + String(", expected ") + expected)


def main() raises:
    var loader = StableDiffusionXLBaseDataLoader()
    var prep = loader._preparation_modules(True, True, True, True, True, True, True, False, True)
    var cache = loader._cache_modules(True, True, True, False, True)
    var output = loader._output_modules(True, True, True, False, True)
    var debug = loader._debug_modules(True, True, True)
    var dataset = loader._create_dataset_options()

    _expect_int("prep module count", len(prep.module_names), 12)
    _expect_string("prep first module", prep.module_names[0], String("RescaleImageChannels:image->image"))
    _expect_string("prep mask", prep.module_names[3], String("ScaleImage:mask->latent_mask:factor=0.125"))
    _expect_string("prep conditioning sample", prep.module_names[6], String("SampleVAEDistribution:latent_conditioning_image_distribution->latent_conditioning_image"))
    _expect_string("prep tokenizer 1", prep.module_names[8], String("Tokenize:prompt_1->tokens_1/tokens_mask_1"))
    _expect_string("prep tokenizer 2", prep.module_names[10], String("Tokenize:prompt_2->tokens_2/tokens_mask_2"))
    _expect_string("prep clip 1 encode", prep.module_names[11], String("EncodeClipText:tokens_1->text_encoder_1_hidden_state"))
    _expect_int("prep max tokens", prep.max_tokens_fallback, 77)
    _expect_string("prep vae sample mode", prep.vae_sample_mode, String("mean"))
    _expect_bool("prep clip masks unused", prep.clip_attention_masks_used, False)
    _expect_string("prep vae dtype source", prep.vae_train_dtype_source, String("model.vae_train_dtype"))
    _expect_string("prep autocast source", prep.vae_autocast_context_source, String("[model.autocast_context, model.vae_autocast_context]"))

    _expect_int("cache image split count", len(cache.image_split_names), 5)
    _expect_string("cache latent image", cache.image_split_names[0], String("latent_image"))
    _expect_string("cache latent mask", cache.image_split_names[3], String("latent_mask"))
    _expect_string("cache latent conditioning", cache.image_split_names[4], String("latent_conditioning_image"))
    _expect_int("cache aggregate count", len(cache.image_aggregate_names), 2)
    _expect_string("cache aggregate crop", cache.image_aggregate_names[0], String("crop_resolution"))
    _expect_string("cache aggregate path", cache.image_aggregate_names[1], String("image_path"))
    _expect_int("cache text split count", len(cache.text_split_names), 2)
    _expect_string("cache text tokens 1", cache.text_split_names[0], String("tokens_1"))
    _expect_string("cache text hidden 1", cache.text_split_names[1], String("text_encoder_1_hidden_state"))
    _expect_bool("cache text caching", cache.text_caching, True)
    _expect_int("cache sort names", len(cache.sort_names), 15)
    _expect_int("cache token masks not cached", len(cache.token_mask_fields_produced_but_not_cached), 2)

    _expect_int("output count", len(output.output_names), 12)
    _expect_string("output first", output.output_names[0], String("image_path"))
    _expect_string("output latent mask", output.output_names[9], String("latent_mask"))
    _expect_string("output latent conditioning", output.output_names[10], String("latent_conditioning_image"))
    _expect_string("output te1 hidden", output.output_names[11], String("text_encoder_1_hidden_state"))
    _expect_int("output module count", len(output.output_module_names), 1)
    _expect_string("output module", output.output_module_names[0], String("_output_modules_from_out_names"))
    _expect_bool("output conditioning support", output.use_conditioning_image, True)
    _expect_string("output dtype source", output.train_dtype_source, String("model.vae_train_dtype"))
    _expect_string("output vae source", output.vae_source, String("model.vae"))

    _expect_int("debug module count", len(debug), 8)
    _expect_string("debug conditioning decode", debug[2], String("DecodeVAE:latent_conditioning_image->decoded_conditioning_image"))
    _expect_string("debug mask decode", debug[4], String("ScaleImage:latent_mask->decoded_mask:factor=8"))
    _expect_string("debug prompt decode", debug[6], String("DecodeTokens:tokens_1->decoded_prompt"))

    _expect_string("dataset model type name", dataset.model_type_name, String("STABLE_DIFFUSION_XL_10_BASE"))
    _expect_int("dataset registered model types", len(dataset.registered_model_types), 2)
    _expect_int("dataset aspect quantum", dataset.aspect_bucketing_quantization, 64)

    print("SDXL SETUP DATALOADER CONTRACT OK")
