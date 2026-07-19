# Ernie loader/saver/sampler surface smoke.
#
# This is not a parity gate and does not generate an image. It only instantiates
# typed Ernie plan surfaces and create/factory dispatch records.

from serenity_trainer.modelLoader.ErnieModelLoader import (
    ErnieFineTuneModelLoader,
    ErnieLoRAModelLoader,
    ErnieModelNames,
    ErnieQuantizationConfig,
    ErnieWeightDtypes,
)
from serenity_trainer.modelSampler.ErnieSampler import ErnieSampleConfig, ErnieSampler
from serenity_trainer.modelSaver.ErnieFineTuneModelSaver import ErnieFineTuneModelSaver
from serenity_trainer.modelSaver.ernie.ErnieLoRASaver import ERNIE_FMT_SAFETENSORS
from serenity_trainer.util.create import create_model_loader, create_model_sampler, create_model_saver
from serenity_trainer.util.enum.ModelType import MODEL_TYPE_ERNIE
from serenity_trainer.util.enum.TrainingMethod import TM_FINE_TUNE, TM_LORA


def main() raises:
    var names = ErnieModelNames(
        String("/models/ernie"),
        String(),
        String(),
        String("/models/ernie-lora.safetensors"),
    )
    var dtypes = ErnieWeightDtypes.bf16()
    var quantization = ErnieQuantizationConfig.default_values()

    var ft_loader = ErnieFineTuneModelLoader()
    var load_plan = ft_loader.load(MODEL_TYPE_ERNIE, names, dtypes, quantization)
    print("ernie ft loader =", load_plan.model_spec, " transformer =", load_plan.transformer_class)

    var lora_loader = ErnieLoRAModelLoader()
    var lora_plan = lora_loader.load(MODEL_TYPE_ERNIE, names, dtypes, quantization)
    print("ernie lora loader =", lora_plan.model_spec, " lora invoked =", lora_plan.lora_loader_invoked)

    var sampler = ErnieSampler(MODEL_TYPE_ERNIE)
    var sample_config = ErnieSampleConfig(
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
    var sample_plan = sampler.sample(sample_config, String("/tmp/ernie.png"))
    print("ernie sample =", sample_plan.height, "x", sample_plan.width, " c =", sample_plan.latent_channels)

    var ft_saver = ErnieFineTuneModelSaver()
    var save_plan = ft_saver.save_plan(
        MODEL_TYPE_ERNIE,
        ERNIE_FMT_SAFETENSORS,
        String("/tmp/ernie-transformer.safetensors"),
        String("BF16"),
    )
    print("ernie saver =", save_plan.route_name)

    var reg_loader = create_model_loader(MODEL_TYPE_ERNIE, TM_LORA)
    var reg_saver = create_model_saver(MODEL_TYPE_ERNIE, TM_FINE_TUNE)
    var reg_sampler = create_model_sampler(MODEL_TYPE_ERNIE, TM_FINE_TUNE)
    print(reg_loader.implementation, reg_saver.implementation, reg_sampler.implementation)
