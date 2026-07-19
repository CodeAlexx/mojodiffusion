# FLUX.1 setup/data-loader surface smoke.
#
# This is not a parity gate and does not run Serenity. It only instantiates
# typed FLUX setup and data-loader contract surfaces.

from serenity_trainer.dataLoader.FluxBaseDataLoader import FluxBaseDataLoader
from serenity_trainer.modelSetup.BaseFluxSetup import (
    BaseFluxSetup,
    FLUX_DEV_MODEL_TYPE_NAME,
    FLUX_FILL_MODEL_TYPE_NAME,
    FLUX_LAYER_PRESET_ATTN_MLP,
    flux_calculate_timestep_shift,
    flux_latent_input_channels,
    flux_model_t_from_timestep,
    flux_packed_latent_channels,
    flux_packed_latent_token_count,
    flux_sigma_from_timestep,
)
from serenity_trainer.modelSetup.FluxFineTuneSetup import FluxFineTuneSetup
from serenity_trainer.modelSetup.FluxLoRASetup import FluxLoRASetup
from serenity_trainer.util.enum.TrainingMethod import TM_FINE_TUNE, TM_LORA


def _expect(name: String, got: Int, expected: Int) raises:
    if got != expected:
        raise Error(
            name + String(" got=") + String(got)
            + String(" expected=") + String(expected)
        )


def main() raises:
    var base = BaseFluxSetup()
    var predict = base.predict_contract()
    var opt_lora = base.optimization_contract(TM_LORA, True)
    var opt_ft = base.optimization_contract(TM_FINE_TUNE, True)
    var device_plan = base.train_device_plan(True, True, False, True, False, True)
    var text_cache = base.prepare_text_caching_plan(True, False)
    var filters = base.layer_preset_filters(FLUX_LAYER_PRESET_ATTN_MLP)

    var lora = FluxLoRASetup()
    var lora_params = lora.create_parameters(True, True, True)
    var lora_creation = lora.creation_plan(
        True, True, True, False, True, False, True, True
    )

    var ft = FluxFineTuneSetup()
    var ft_params = ft.create_parameters(True, True, True)
    var ft_plan = ft.setup_model_plan(True)

    var loader = FluxBaseDataLoader()
    var prep = loader._preparation_modules(
        False, True, True, True, True, True, True, False, False
    )
    var cache = loader._cache_modules(True, True, True, False, True)
    var output = loader._output_modules(True, True, True, False, True)
    var debug = loader._debug_modules(True, True, True)
    var dataset = loader._create_dataset_options()
    var fill_mask = loader.fill_mask_plan()

    _expect("predict output fields", len(predict.output_fields), 4)
    _expect("attn-mlp filters", len(filters), 2)
    _expect("lora params", len(lora_params), 5)
    _expect("fine tune params", len(ft_params), 5)
    _expect("fill image split names", len(cache.image_split_names), 5)
    _expect("fill mask channels", fill_mask.output_channels, 64)
    _expect("fill latent input channels", flux_latent_input_channels(True, True), 96)
    _expect("base packed channels", flux_packed_latent_channels(), 64)
    _expect("packed tokens 64x64", flux_packed_latent_token_count(64, 64), 1024)

    print("flux models =", FLUX_DEV_MODEL_TYPE_NAME, "/", FLUX_FILL_MODEL_TYPE_NAME)
    print("predict outputs =", len(predict.output_fields))
    print("dtype caveats =", len(predict.dtype_boundary_caveats))
    print("lora opt parts =", len(opt_lora.autocast_weight_dtype_parts))
    print("ft opt parts =", len(opt_ft.autocast_weight_dtype_parts))
    print("text encoder 1 train device =", device_plan.text_encoder_1_on_train_device)
    print("text encoder 2 cache device =", text_cache.move_text_encoder_2_to_train_device)
    print("lora params =", len(lora_params), " create te1 =", lora_creation.create_text_encoder_1_lora)
    print("ft params =", len(ft_params), " module filter =", ft_plan.uses_module_filter_for_transformer)
    print("prep modules =", len(prep.module_names))
    print("cache sort names =", len(cache.sort_names))
    print("output names =", len(output.output_names))
    print("debug modules =", len(debug))
    print("dataset model types =", len(dataset.model_types))
    print("model_t(t=250) =", flux_model_t_from_timestep(250), " sigma(t=250) =", flux_sigma_from_timestep(250))
    print("shift(64,64) =", flux_calculate_timestep_shift(64, 64))
    print("FLUX SETUP SURFACE OK")
