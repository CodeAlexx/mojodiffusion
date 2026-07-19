# Anima setup/data-loader surface smoke.
#
# This is not a parity gate and does not run Serenity. It only instantiates
# typed Anima setup and data-loader contract surfaces.

from serenity_trainer.dataLoader.AnimaBaseDataLoader import AnimaBaseDataLoader
from serenity_trainer.modelSetup.AnimaFineTuneSetup import AnimaFineTuneSetup
from serenity_trainer.modelSetup.AnimaLoRASetup import AnimaLoRASetup
from serenity_trainer.modelSetup.BaseAnimaSetup import (
    ANIMA_MODEL_TYPE_NAME,
    anima_model_t_from_timestep,
    anima_sigma_from_timestep,
    BaseAnimaSetup,
)
from serenity_trainer.util.enum.TrainingMethod import TM_FINE_TUNE, TM_LORA


def main() raises:
    var base = BaseAnimaSetup()
    var predict = base.predict_contract()
    var opt_lora = base.optimization_contract(TM_LORA)
    var opt_ft = base.optimization_contract(TM_FINE_TUNE)
    var device_plan = base.train_device_plan(True, True)

    var lora = AnimaLoRASetup()
    var ft = AnimaFineTuneSetup()
    var loader = AnimaBaseDataLoader()
    var prep = loader._preparation_modules(False, False, False)
    var cache = loader._cache_modules(True, False, False)
    var output = loader._output_modules(True, False, False)

    print("anima model =", ANIMA_MODEL_TYPE_NAME)
    print("predict outputs =", len(predict.output_fields))
    print("lora opt parts =", len(opt_lora.autocast_weight_dtype_parts))
    print("ft opt parts =", len(opt_ft.autocast_weight_dtype_parts))
    print("text encoder train device =", device_plan.text_encoder_on_train_device)
    print("lora params =", len(lora.create_parameters()), " ft params =", len(ft.create_parameters()))
    print("prep modules =", len(prep.module_names))
    print("cache sort names =", len(cache.sort_names))
    print("output names =", len(output.output_names))
    print("model_t(t=250) =", anima_model_t_from_timestep(250), " sigma(t=250) =", anima_sigma_from_timestep(250))
    print("ANIMA SETUP SURFACE OK")
