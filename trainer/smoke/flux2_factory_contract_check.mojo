# Flux2/Klein util/create factory contract gate.
#
# Source of truth:
#   /home/alex/Serenity/modules/util/enum/ModelType.py
#   /home/alex/Serenity/modules/util/create.py
#   /home/alex/Serenity/modules/modelLoader/Flux2ModelLoader.py
#   /home/alex/Serenity/modules/modelSaver/Flux2FineTuneModelSaver.py
#   /home/alex/Serenity/modules/modelSaver/Flux2LoRAModelSaver.py
#   /home/alex/Serenity/modules/modelSetup/Flux2FineTuneSetup.py
#   /home/alex/Serenity/modules/modelSetup/Flux2LoRASetup.py
#
# This is build-only/source-contract coverage. It does not load Flux2 weights,
# run CUDA, instantiate Serenity, or make a numeric parity claim. Serenity
# registers Dev and Klein under the same ModelType.FLUX_2; Flux2ModelLoader.py
# branches after loading the transformer with num_attention_heads == 48 for Dev
# and the else branch for Klein.

from serenity_trainer.util.create import (
    create_model_loader,
    create_model_sampler,
    create_model_saver,
)
from serenity_trainer.util.enum.ModelType import (
    MODEL_TYPE_FLUX_2,
    model_type_is_flux,
    model_type_is_flux_2,
    model_type_str,
)
from serenity_trainer.util.enum.TrainingMethod import TM_EMBEDDING, TM_FINE_TUNE, TM_LORA
from serenity_trainer.util.factory import (
    FACTORY_BASE_MODEL_LOADER,
    FACTORY_BASE_MODEL_SAMPLER,
    FACTORY_BASE_MODEL_SAVER,
    FACTORY_TRAINING_METHOD_ANY,
    get_flux2_registration,
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


def _expect_raises_flux2_embedding_loader() raises:
    var raised = False
    try:
        _ = create_model_loader(MODEL_TYPE_FLUX_2, TM_EMBEDDING)
    except:
        raised = True
    _expect_bool("Flux2 embedding loader unsupported", raised, True)


def main() raises:
    _expect_bool("ModelType.is_flux", model_type_is_flux(MODEL_TYPE_FLUX_2), True)
    _expect_bool("ModelType.is_flux_2", model_type_is_flux_2(MODEL_TYPE_FLUX_2), True)
    _expect_string("ModelType.__str__", model_type_str(MODEL_TYPE_FLUX_2), String("FLUX_2"))

    var lora_loader = create_model_loader(MODEL_TYPE_FLUX_2, TM_LORA)
    _expect_int("lora loader base", lora_loader.base_kind, FACTORY_BASE_MODEL_LOADER)
    _expect_int("lora loader model", lora_loader.model_type, MODEL_TYPE_FLUX_2)
    _expect_int("lora loader method", lora_loader.training_method, TM_LORA)
    _expect_string("lora loader implementation", lora_loader.implementation, String("Flux2LoRAModelLoader"))
    _expect_string("lora loader spec", lora_loader.model_spec, String("resources/sd_model_spec/flux_2.0-lora.json"))

    var ft_loader = create_model_loader(MODEL_TYPE_FLUX_2, TM_FINE_TUNE)
    _expect_int("ft loader base", ft_loader.base_kind, FACTORY_BASE_MODEL_LOADER)
    _expect_int("ft loader model", ft_loader.model_type, MODEL_TYPE_FLUX_2)
    _expect_int("ft loader method", ft_loader.training_method, TM_FINE_TUNE)
    _expect_string("ft loader implementation", ft_loader.implementation, String("Flux2FineTuneModelLoader"))
    _expect_string("ft loader spec", ft_loader.model_spec, String("resources/sd_model_spec/flux_2.0.json"))

    var lora_saver = create_model_saver(MODEL_TYPE_FLUX_2, TM_LORA)
    _expect_int("lora saver base", lora_saver.base_kind, FACTORY_BASE_MODEL_SAVER)
    _expect_int("lora saver method", lora_saver.training_method, TM_LORA)
    _expect_string("lora saver implementation", lora_saver.implementation, String("Flux2LoRAModelSaver"))
    _expect_string("lora saver spec", lora_saver.model_spec, String())

    var ft_saver = create_model_saver(MODEL_TYPE_FLUX_2, TM_FINE_TUNE)
    _expect_int("ft saver base", ft_saver.base_kind, FACTORY_BASE_MODEL_SAVER)
    _expect_int("ft saver method", ft_saver.training_method, TM_FINE_TUNE)
    _expect_string("ft saver implementation", ft_saver.implementation, String("Flux2FineTuneModelSaver"))
    _expect_string("ft saver spec", ft_saver.model_spec, String())

    var sampler = create_model_sampler(MODEL_TYPE_FLUX_2, TM_LORA)
    _expect_int("sampler base", sampler.base_kind, FACTORY_BASE_MODEL_SAMPLER)
    _expect_int("sampler model", sampler.model_type, MODEL_TYPE_FLUX_2)
    _expect_int("sampler method fallback", sampler.training_method, FACTORY_TRAINING_METHOD_ANY)
    _expect_string("sampler implementation", sampler.implementation, String("Flux2Sampler"))
    _expect_string("sampler spec", sampler.model_spec, String())

    var direct_ft_loader = get_flux2_registration(
        FACTORY_BASE_MODEL_LOADER,
        MODEL_TYPE_FLUX_2,
        TM_FINE_TUNE,
    )
    _expect_string("direct factory ft loader", direct_ft_loader.implementation, String("Flux2FineTuneModelLoader"))

    var direct_sampler = get_flux2_registration(
        FACTORY_BASE_MODEL_SAMPLER,
        MODEL_TYPE_FLUX_2,
        TM_FINE_TUNE,
    )
    _expect_int("direct factory sampler fallback", direct_sampler.training_method, FACTORY_TRAINING_METHOD_ANY)
    _expect_string("direct factory sampler", direct_sampler.implementation, String("Flux2Sampler"))

    _expect_raises_flux2_embedding_loader()

    print("FLUX2 FACTORY CONTRACT OK")
