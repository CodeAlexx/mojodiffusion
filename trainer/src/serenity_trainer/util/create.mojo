# 1:1 surface port of Serenity modules/util/create.py
#
# Build-only dispatch. Serenity delegates to util.factory after importing
# modelSampler/modelLoader/modelSaver modules. This Mojo surface returns the
# corresponding factory registration records for product surfaces ported here.

from serenity_trainer.util.enum.ModelType import (
    MODEL_TYPE_ERNIE,
    MODEL_TYPE_FLUX_2,
    MODEL_TYPE_IDEOGRAM_4,
    MODEL_TYPE_QWEN,
    model_type_is_flux_1,
    model_type_is_stable_diffusion_3,
    model_type_is_stable_diffusion_xl,
    model_type_str,
)
from serenity_trainer.util.enum.TrainingMethod import TM_FINE_TUNE
from serenity_trainer.util.factory import (
    FACTORY_BASE_MODEL_LOADER,
    FACTORY_BASE_MODEL_SAMPLER,
    FACTORY_BASE_MODEL_SAVER,
    FactoryRegistration,
    MODEL_TYPE_ANIMA,
    get_anima_registration,
    get_ernie_registration,
    get_flux1_registration,
    get_flux2_registration,
    get_ideogram4_registration,
    get_qwen_registration,
    get_stable_diffusion3_registration,
    get_stable_diffusion_xl_registration,
)


def create_model_loader(
    model_type: Int,
    training_method: Int = TM_FINE_TUNE,
) raises -> FactoryRegistration:
    if model_type == MODEL_TYPE_QWEN:
        return get_qwen_registration(FACTORY_BASE_MODEL_LOADER, model_type, training_method)
    if model_type_is_stable_diffusion_3(model_type):
        return get_stable_diffusion3_registration(FACTORY_BASE_MODEL_LOADER, model_type, training_method)
    if model_type_is_stable_diffusion_xl(model_type):
        return get_stable_diffusion_xl_registration(FACTORY_BASE_MODEL_LOADER, model_type, training_method)
    if model_type_is_flux_1(model_type):
        return get_flux1_registration(FACTORY_BASE_MODEL_LOADER, model_type, training_method)
    if model_type == MODEL_TYPE_FLUX_2:
        return get_flux2_registration(FACTORY_BASE_MODEL_LOADER, model_type, training_method)
    if model_type == MODEL_TYPE_IDEOGRAM_4:
        return get_ideogram4_registration(FACTORY_BASE_MODEL_LOADER, model_type, training_method)
    if model_type == MODEL_TYPE_ANIMA:
        return get_anima_registration(FACTORY_BASE_MODEL_LOADER, model_type, training_method)
    if model_type == MODEL_TYPE_ERNIE:
        return get_ernie_registration(FACTORY_BASE_MODEL_LOADER, model_type, training_method)
    raise Error(String("create_model_loader: unsupported ModelType ") + model_type_str(model_type))


def create_model_saver(
    model_type: Int,
    training_method: Int = TM_FINE_TUNE,
) raises -> FactoryRegistration:
    if model_type == MODEL_TYPE_QWEN:
        return get_qwen_registration(FACTORY_BASE_MODEL_SAVER, model_type, training_method)
    if model_type_is_stable_diffusion_3(model_type):
        return get_stable_diffusion3_registration(FACTORY_BASE_MODEL_SAVER, model_type, training_method)
    if model_type_is_stable_diffusion_xl(model_type):
        return get_stable_diffusion_xl_registration(FACTORY_BASE_MODEL_SAVER, model_type, training_method)
    if model_type_is_flux_1(model_type):
        return get_flux1_registration(FACTORY_BASE_MODEL_SAVER, model_type, training_method)
    if model_type == MODEL_TYPE_FLUX_2:
        return get_flux2_registration(FACTORY_BASE_MODEL_SAVER, model_type, training_method)
    if model_type == MODEL_TYPE_IDEOGRAM_4:
        return get_ideogram4_registration(FACTORY_BASE_MODEL_SAVER, model_type, training_method)
    if model_type == MODEL_TYPE_ANIMA:
        return get_anima_registration(FACTORY_BASE_MODEL_SAVER, model_type, training_method)
    if model_type == MODEL_TYPE_ERNIE:
        return get_ernie_registration(FACTORY_BASE_MODEL_SAVER, model_type, training_method)
    raise Error(String("create_model_saver: unsupported ModelType ") + model_type_str(model_type))


def create_model_sampler(
    model_type: Int,
    training_method: Int = TM_FINE_TUNE,
) raises -> FactoryRegistration:
    if model_type == MODEL_TYPE_QWEN:
        return get_qwen_registration(FACTORY_BASE_MODEL_SAMPLER, model_type, training_method)
    if model_type_is_stable_diffusion_3(model_type):
        return get_stable_diffusion3_registration(FACTORY_BASE_MODEL_SAMPLER, model_type, training_method)
    if model_type_is_stable_diffusion_xl(model_type):
        return get_stable_diffusion_xl_registration(FACTORY_BASE_MODEL_SAMPLER, model_type, training_method)
    if model_type_is_flux_1(model_type):
        return get_flux1_registration(FACTORY_BASE_MODEL_SAMPLER, model_type, training_method)
    if model_type == MODEL_TYPE_FLUX_2:
        return get_flux2_registration(FACTORY_BASE_MODEL_SAMPLER, model_type, training_method)
    if model_type == MODEL_TYPE_IDEOGRAM_4:
        return get_ideogram4_registration(FACTORY_BASE_MODEL_SAMPLER, model_type, training_method)
    if model_type == MODEL_TYPE_ANIMA:
        return get_anima_registration(FACTORY_BASE_MODEL_SAMPLER, model_type, training_method)
    if model_type == MODEL_TYPE_ERNIE:
        return get_ernie_registration(FACTORY_BASE_MODEL_SAMPLER, model_type, training_method)
    raise Error(String("create_model_sampler: unsupported ModelType ") + model_type_str(model_type))
