# SDXL LoRA/fine-tune setup contract gate.
#
# Source of truth:
#   /home/alex/Serenity/modules/modelSetup/StableDiffusionXLLoRASetup.py
#   /home/alex/Serenity/modules/modelSetup/StableDiffusionXLFineTuneSetup.py

from serenity_trainer.modelSetup.BaseStableDiffusionXLSetup import (
    SDXL_LAYER_PRESET_ATTN_MLP,
)
from serenity_trainer.modelSetup.StableDiffusionXLFineTuneSetup import (
    StableDiffusionXLFineTuneSetup,
)
from serenity_trainer.modelSetup.StableDiffusionXLLoRASetup import (
    StableDiffusionXLLoRASetup,
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
    var lora = StableDiffusionXLLoRASetup()
    var lora_params = lora.create_parameters(True, True, False)
    var lora_creation = lora.creation_plan(True, False, False, True, True, True)
    var lora_filters = lora.layer_filters(SDXL_LAYER_PRESET_ATTN_MLP)

    _expect_int("lora param count", len(lora_params), 4)
    _expect_string("lora param te1", lora_params[0], String("text_encoder_1_lora"))
    _expect_string("lora param te2", lora_params[1], String("text_encoder_2_lora"))
    _expect_string("lora param emb1", lora_params[2], String("embeddings_1"))
    _expect_string("lora param unet", lora_params[3], String("unet_lora"))
    _expect_bool("lora create te1", lora_creation.create_text_encoder_1_lora, True)
    _expect_bool("lora create te2", lora_creation.create_text_encoder_2_lora, True)
    _expect_bool("lora create unet", lora_creation.create_unet_lora, True)
    _expect_int("lora wrapper prefix count", len(lora_creation.wrapper_prefixes), 3)
    _expect_string("lora wrapper te1", lora_creation.wrapper_prefixes[0], String("lora_te1"))
    _expect_string("lora wrapper te2", lora_creation.wrapper_prefixes[1], String("lora_te2"))
    _expect_string("lora wrapper unet", lora_creation.wrapper_prefixes[2], String("lora_unet"))
    _expect_bool("lora pending state dict", lora_creation.loads_pending_state_dict, True)
    _expect_bool("lora te1 dropout", lora_creation.sets_text_encoder_1_dropout_from_config, True)
    _expect_bool("lora te2 dropout", lora_creation.sets_text_encoder_2_dropout_from_config, False)
    _expect_bool("lora unet dropout", lora_creation.sets_unet_dropout_from_config, True)
    _expect_bool("lora te1 dtype move", lora_creation.moves_text_encoder_1_lora_to_config_weight_dtype, True)
    _expect_bool("lora te2 dtype move", lora_creation.moves_text_encoder_2_lora_to_config_weight_dtype, True)
    _expect_bool("lora unet dtype move", lora_creation.moves_unet_lora_to_config_weight_dtype, True)
    _expect_bool("lora rescale zero snr", lora_creation.rescales_noise_scheduler_to_zero_terminal_snr, True)
    _expect_bool("lora force v after rescale", lora_creation.forces_v_prediction_after_rescale, True)
    _expect_bool("lora removes token embeddings", lora_creation.removes_added_embeddings_from_tokenizers, True)
    _expect_bool("lora setup embeddings", lora_creation.setups_embeddings, True)
    _expect_bool("lora setup wrappers", lora_creation.setups_embedding_wrappers, True)
    _expect_bool("lora init params", lora_creation.initializes_model_parameters, True)
    _expect_int("lora filter count", len(lora_filters), 1)
    _expect_string("lora attn-mlp filter", lora_filters[0], String("attentions"))

    var ft = StableDiffusionXLFineTuneSetup()
    var ft_params = ft.create_parameters(True, True, True)
    var ft_plan = ft.setup_model_plan(True, True, False, False)
    var ft_device = ft.train_device_plan(True, False, True, True, True)

    _expect_int("ft param count", len(ft_params), 5)
    _expect_string("ft param te1", ft_params[0], String("text_encoder_1"))
    _expect_string("ft param te2", ft_params[1], String("text_encoder_2"))
    _expect_string("ft param emb1", ft_params[2], String("embeddings_1"))
    _expect_string("ft param emb2", ft_params[3], String("embeddings_2"))
    _expect_string("ft param unet", ft_params[4], String("unet"))
    _expect_bool("ft embedding dtype move", ft_plan.moves_input_embeddings_to_embedding_weight_dtype, True)
    _expect_bool("ft rescale zero snr", ft_plan.rescales_noise_scheduler_to_zero_terminal_snr, True)
    _expect_bool("ft force v after rescale", ft_plan.forces_v_prediction_after_rescale, True)
    _expect_bool("ft force v direct disabled", ft_plan.forces_v_prediction, False)
    _expect_bool("ft force epsilon disabled", ft_plan.forces_epsilon_prediction, False)
    _expect_bool("ft removes token embeddings", ft_plan.removes_added_embeddings_from_tokenizers, True)
    _expect_bool("ft setup embeddings", ft_plan.setups_embeddings, True)
    _expect_bool("ft setup wrappers", ft_plan.setups_embedding_wrappers, True)
    _expect_bool("ft init params", ft_plan.initializes_model_parameters, True)
    _expect_bool("ft module filter", ft_plan.uses_module_filter_for_unet, True)
    _expect_bool("ft debug filter", ft_plan.uses_debug_flag_for_unet_filter, True)
    _expect_bool("ft device te1", ft_device.text_encoder_1_on_train_device, True)
    _expect_bool("ft device te2", ft_device.text_encoder_2_on_train_device, True)
    _expect_bool("ft device vae temp", ft_device.vae_on_train_device, False)
    _expect_bool("ft device unet", ft_device.unet_on_train_device, True)
    _expect_bool("ft te1 train mode", ft_device.text_encoder_1_train_mode, False)
    _expect_bool("ft te2 train mode", ft_device.text_encoder_2_train_mode, True)
    _expect_bool("ft vae train mode", ft_device.vae_train_mode, True)
    _expect_bool("ft unet train mode", ft_device.unet_train_mode, True)

    print("SDXL SETUP METHOD CONTRACT OK")
