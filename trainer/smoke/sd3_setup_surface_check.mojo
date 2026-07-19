# SD3 setup/data-loader surface smoke.
#
# This is not a parity gate and does not run Serenity. It only instantiates
# typed SD3 setup and data-loader contract surfaces.

from serenity_trainer.dataLoader.StableDiffusion3BaseDataLoader import (
    StableDiffusion3BaseDataLoader,
)
from serenity_trainer.modelSetup.BaseStableDiffusion3Setup import (
    BaseStableDiffusion3Setup,
    SD3_MODEL_TYPE_NAME,
    SD35_MODEL_TYPE_NAME,
    sd3_model_t_from_timestep,
    sd3_sigma_from_timestep,
)
from serenity_trainer.modelSetup.StableDiffusion3FineTuneSetup import (
    StableDiffusion3FineTuneSetup,
)
from serenity_trainer.modelSetup.StableDiffusion3LoRASetup import (
    StableDiffusion3LoRASetup,
)
from serenity_trainer.util.enum.TrainingMethod import TM_FINE_TUNE, TM_LORA


def _abs(x: Float32) -> Float32:
    if x < Float32(0.0):
        return -x
    return x


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


def _expect_close(name: String, got: Float32, expected: Float32, tol: Float32) raises:
    var diff = _abs(got - expected)
    if diff > tol:
        raise Error(
            name + String(": got ") + String(got)
            + String(", expected ") + String(expected)
            + String(", |d| ") + String(diff)
        )


def main() raises:
    var base = BaseStableDiffusion3Setup()
    var predict = base.predict_contract()
    var opt_lora = base.optimization_contract(TM_LORA, True)
    var opt_ft = base.optimization_contract(TM_FINE_TUNE, True)
    var device_plan = base.train_device_plan(
        True, True, False, False, True, False, False, True
    )
    var text_cache = base.prepare_text_caching_plan(True, False, False)

    var lora = StableDiffusion3LoRASetup()
    var lora_params = lora.create_parameters(True, True, True, True)
    var lora_creation = lora.creation_plan(
        True, True, True, True, False, False, True, False, False, True, True
    )

    var ft = StableDiffusion3FineTuneSetup()
    var ft_params = ft.create_parameters(True, True, True, True)
    var ft_plan = ft.setup_model_plan(True)

    var loader = StableDiffusion3BaseDataLoader()
    var prep = loader._preparation_modules(
        False, False, False, True, True, True, True, True, True, False, False, False
    )
    var cache = loader._cache_modules(True, False, False, False, True, False)
    var output = loader._output_modules(True, False, False, False, True, False)
    var dataset = loader._create_dataset_options()

    _expect_string("sd3 model type", String(SD3_MODEL_TYPE_NAME), String("STABLE_DIFFUSION_3"))
    _expect_string("sd35 model type", String(SD35_MODEL_TYPE_NAME), String("STABLE_DIFFUSION_35"))
    _expect_int("base model type count", len(base.model_types), 2)
    _expect_int("predict required fields", len(predict.required_batch_fields), 1)
    _expect_string("predict latent field", predict.required_batch_fields[0], String("latent_image"))
    _expect_int("predict text fields", len(predict.text_batch_fields), 11)
    _expect_string("predict text field tokens_1", predict.text_batch_fields[0], String("tokens_1"))
    _expect_string("predict text field te3 hidden", predict.text_batch_fields[10], String("text_encoder_3_hidden_state"))
    _expect_int("predict conditional fields", len(predict.conditional_latent_fields), 2)
    _expect_int("predict output fields", len(predict.output_fields), 4)
    _expect_string("predict loss type", predict.loss_type, String("target"))
    _expect_string("predict output predicted", predict.output_fields[2], String("predicted"))
    _expect_string("predict target expression", predict.target_expression, String("latent_noise - scaled_latent_image"))
    _expect_string("predict hidden dtype expression", predict.transformer_hidden_states_expression, String("latent_input.to(dtype=model.train_dtype.torch_dtype())"))

    _expect_int("lora opt checkpoint parts", len(opt_lora.checkpoint_parts), 4)
    _expect_int("lora opt autocast parts", len(opt_lora.autocast_weight_dtype_parts), 7)
    _expect_string("lora autocast part", opt_lora.autocast_weight_dtype_parts[5], String("lora"))
    _expect_int("ft opt autocast parts", len(opt_ft.autocast_weight_dtype_parts), 6)
    _expect_bool("t5 fp16 autocast disabled", opt_lora.disables_fp16_text_encoder_3_autocast, True)

    _expect_bool("device te1 train", device_plan.text_encoder_1_on_train_device, True)
    _expect_bool("device te2 temp", device_plan.text_encoder_2_on_train_device, False)
    _expect_bool("device te3 temp", device_plan.text_encoder_3_on_train_device, False)
    _expect_bool("device vae temp", device_plan.vae_on_train_device, False)
    _expect_bool("device transformer train", device_plan.transformer_on_train_device, True)
    _expect_bool("device transformer mode", device_plan.transformer_train_mode, True)
    _expect_bool("text cache model temp", text_cache.move_model_to_temp_device, True)
    _expect_bool("text cache te1 stays", text_cache.move_text_encoder_1_to_train_device, False)
    _expect_bool("text cache te2 moves", text_cache.move_text_encoder_2_to_train_device, True)
    _expect_bool("text cache te3 moves", text_cache.move_text_encoder_3_to_train_device, True)

    _expect_int("lora param count", len(lora_params), 7)
    _expect_string("lora param te1", lora_params[0], String("text_encoder_1_lora"))
    _expect_string("lora param emb3", lora_params[5], String("embeddings_3"))
    _expect_string("lora param transformer", lora_params[6], String("transformer_lora"))
    _expect_bool("lora create te1", lora_creation.create_text_encoder_1_lora, True)
    _expect_bool("lora create te2", lora_creation.create_text_encoder_2_lora, False)
    _expect_bool("lora create te3", lora_creation.create_text_encoder_3_lora, False)
    _expect_bool("lora create transformer", lora_creation.create_transformer_lora, True)
    _expect_bool("lora pending load", lora_creation.loads_pending_state_dict, True)
    _expect_bool("lora embedding dtype move", lora_creation.moves_input_embeddings_to_embedding_weight_dtype, True)

    _expect_int("ft param count", len(ft_params), 7)
    _expect_string("ft param te1", ft_params[0], String("text_encoder_1"))
    _expect_string("ft param transformer", ft_params[6], String("transformer"))
    _expect_bool("ft embedding dtype move", ft_plan.moves_input_embeddings_to_embedding_weight_dtype, True)
    _expect_bool("ft module filter", ft_plan.uses_module_filter_for_transformer, True)

    _expect_int("prep module count", len(prep.module_names), 12)
    _expect_string("prep first module", prep.module_names[0], String("RescaleImageChannels:image->image"))
    _expect_string("prep tokenizer 3", prep.module_names[8], String("Tokenize:prompt_3->tokens_3/tokens_mask_3"))
    _expect_string("prep t5 encode", prep.module_names[11], String("EncodeT5Text:tokens_3->text_encoder_3_hidden_state"))
    _expect_int("prep max tokens", prep.max_tokens_fallback, 77)
    _expect_string("prep vae sample mode", prep.vae_sample_mode, String("mean"))

    _expect_int("cache image split count", len(cache.image_split_names), 4)
    _expect_string("cache image split mask", cache.image_split_names[3], String("latent_mask"))
    _expect_int("cache text split count", len(cache.text_split_names), 7)
    _expect_string("cache te3 hidden", cache.text_split_names[6], String("text_encoder_3_hidden_state"))
    _expect_bool("cache text caching", cache.text_caching, True)
    _expect_int("cache sort names", len(cache.sort_names), 21)

    _expect_int("output count", len(output.output_names), 18)
    _expect_string("output first", output.output_names[0], String("image_path"))
    _expect_string("output latent mask", output.output_names[14], String("latent_mask"))
    _expect_string("output te3 hidden", output.output_names[17], String("text_encoder_3_hidden_state"))
    _expect_int("output module count", len(output.output_module_names), 1)
    _expect_bool("output conditioning support", output.use_conditioning_image, True)

    _expect_string("dataset model type name", dataset.model_type_name, String("STABLE_DIFFUSION_35"))
    _expect_int("dataset aspect quantum", dataset.aspect_bucketing_quantization, 64)
    _expect_close("sigma t=250", sd3_sigma_from_timestep(250), Float32(0.251), Float32(1e-6))
    _expect_close("model_t t=250", sd3_model_t_from_timestep(250), Float32(250.0), Float32(1e-6))

    print("sd3 models =", SD3_MODEL_TYPE_NAME, "/", SD35_MODEL_TYPE_NAME)
    print("predict outputs =", len(predict.output_fields))
    print("lora opt parts =", len(opt_lora.autocast_weight_dtype_parts))
    print("ft opt parts =", len(opt_ft.autocast_weight_dtype_parts))
    print("text encoder 1 train device =", device_plan.text_encoder_1_on_train_device)
    print("text encoder 2 cache device =", text_cache.move_text_encoder_2_to_train_device)
    print("lora params =", len(lora_params), " create te1 =", lora_creation.create_text_encoder_1_lora)
    print("ft params =", len(ft_params), " module filter =", ft_plan.uses_module_filter_for_transformer)
    print("prep modules =", len(prep.module_names))
    print("cache sort names =", len(cache.sort_names))
    print("output names =", len(output.output_names))
    print("dataset model type =", dataset.model_type)
    print("model_t(t=250) =", sd3_model_t_from_timestep(250), " sigma(t=250) =", sd3_sigma_from_timestep(250))
    print("SD3 SETUP SURFACE OK")
