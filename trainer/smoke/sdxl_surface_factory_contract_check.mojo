# SDXL util/create factory contract gate.
#
# Source of truth:
#   /home/alex/Serenity/modules/util/create.py

from serenity_trainer.util.create import (
    create_model_loader,
    create_model_sampler,
    create_model_saver,
)
from serenity_trainer.util.enum.ModelType import (
    MODEL_TYPE_STABLE_DIFFUSION_XL_10_BASE,
    MODEL_TYPE_STABLE_DIFFUSION_XL_10_BASE_INPAINTING,
)
from serenity_trainer.util.enum.TrainingMethod import TM_FINE_TUNE, TM_LORA


def _expect_int(name: String, got: Int, expected: Int) raises:
    if got != expected:
        raise Error(name + String(": got ") + String(got) + String(", expected ") + String(expected))


def _expect_string(name: String, got: String, expected: String) raises:
    if got != expected:
        raise Error(name + String(": got ") + got + String(", expected ") + expected)


def main() raises:
    var dispatch_loader = create_model_loader(MODEL_TYPE_STABLE_DIFFUSION_XL_10_BASE_INPAINTING, TM_LORA)
    var dispatch_saver = create_model_saver(MODEL_TYPE_STABLE_DIFFUSION_XL_10_BASE, TM_FINE_TUNE)
    var dispatch_sampler = create_model_sampler(MODEL_TYPE_STABLE_DIFFUSION_XL_10_BASE_INPAINTING, TM_LORA)
    _expect_string("factory lora loader", dispatch_loader.implementation, String("StableDiffusionXLLoRAModelLoader"))
    _expect_int("factory lora training method", dispatch_loader.training_method, TM_LORA)
    _expect_string("factory lora spec", dispatch_loader.model_spec, String("resources/sd_model_spec/sd_xl_base_1.0_inpainting-lora.json"))
    _expect_string("factory ft saver", dispatch_saver.implementation, String("StableDiffusionXLFineTuneModelSaver"))
    _expect_int("factory ft training method", dispatch_saver.training_method, TM_FINE_TUNE)
    _expect_string("factory sampler", dispatch_sampler.implementation, String("StableDiffusionXLSampler"))
    _expect_int("factory sampler training method", dispatch_sampler.training_method, -1)

    print("SDXL SURFACE FACTORY CONTRACT OK")
