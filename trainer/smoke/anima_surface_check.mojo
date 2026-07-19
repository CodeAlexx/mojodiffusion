# Anima loader/saver/sampler surface smoke.
#
# This is not a parity gate and does not generate an image. It only instantiates
# typed Anima plan surfaces and create/factory dispatch records.

from serenity_trainer.modelLoader.AnimaModelLoader import (
    MODEL_TYPE_ANIMA,
    AnimaFineTuneModelLoader,
    AnimaLoRAModelLoader,
    AnimaModelNames,
    AnimaQuantizationConfig,
    AnimaWeightDtypes,
)
from serenity_trainer.modelSampler.AnimaSampler import AnimaSampleConfig, AnimaSampler
from serenity_trainer.modelSaver.AnimaFineTuneModelSaver import AnimaFineTuneModelSaver
from serenity_trainer.modelSaver.AnimaLoRAModelSaver import AnimaLoRAModelSaver
from serenity_trainer.modelSaver.anima.AnimaLoRASaver import ANIMA_FMT_SAFETENSORS
from serenity_trainer.modelSaver.anima.AnimaModelSaver import anima_diffusers_to_original_key_renames
from serenity_trainer.util.create import create_model_loader, create_model_sampler, create_model_saver
from serenity_trainer.util.enum.TrainingMethod import TM_FINE_TUNE, TM_LORA


def main() raises:
    var names = AnimaModelNames(
        String("/models/anima"),
        String(),
        String(),
        String("/models/anima-lora.safetensors"),
    )
    var dtypes = AnimaWeightDtypes.bf16()
    var quantization = AnimaQuantizationConfig.default_values()

    var ft_loader = AnimaFineTuneModelLoader()
    var load_plan = ft_loader.load(MODEL_TYPE_ANIMA, names, dtypes, quantization)
    print("anima ft loader =", load_plan.model_spec, " transformer =", load_plan.transformer_class)

    var lora_loader = AnimaLoRAModelLoader()
    var lora_plan = lora_loader.load(MODEL_TYPE_ANIMA, names, dtypes, quantization)
    print("anima lora loader =", lora_plan.model_spec, " lora invoked =", lora_plan.lora_loader_invoked)

    var sampler = AnimaSampler(MODEL_TYPE_ANIMA)
    var sample_config = AnimaSampleConfig(
        String("prompt"),
        String("negative"),
        1025,
        1025,
        1,
        False,
        4,
        Float32(4.0),
        0,
    )
    var sample_plan = sampler.sample(sample_config, String("/tmp/anima.png"))
    print("anima sample =", sample_plan.height, "x", sample_plan.width, " c =", sample_plan.latent_channels)

    var ft_saver = AnimaFineTuneModelSaver()
    var save_plan = ft_saver.save_plan(
        MODEL_TYPE_ANIMA,
        ANIMA_FMT_SAFETENSORS,
        String("/tmp/anima-original.safetensors"),
        String("BF16"),
    )
    print("anima saver =", save_plan.route_name)
    var lora_saver = AnimaLoRAModelSaver()
    _ = lora_saver

    var renames = anima_diffusers_to_original_key_renames(1)
    print("anima rename entries =", len(renames))

    var reg_loader = create_model_loader(MODEL_TYPE_ANIMA, TM_LORA)
    var reg_saver = create_model_saver(MODEL_TYPE_ANIMA, TM_FINE_TUNE)
    var reg_sampler = create_model_sampler(MODEL_TYPE_ANIMA, TM_FINE_TUNE)
    print(reg_loader.implementation, reg_saver.implementation, reg_sampler.implementation)
