# Qwen loader/saver/sampler surface smoke.
#
# This is not a parity gate and does not generate an image. It only instantiates
# typed Qwen plan surfaces and create/factory dispatch records.

from serenity_trainer.modelLoader.qwen.QwenModelLoader import (
    QwenModelNames,
    QwenQuantizationConfig,
    QwenWeightDtypes,
)
from serenity_trainer.modelLoader.QwenFineTuneModelLoader import QwenFineTuneModelLoader
from serenity_trainer.modelLoader.QwenLoRAModelLoader import QwenLoRAModelLoader
from serenity_trainer.modelSampler.QwenSampler import QwenSampleConfig, QwenSampler
from serenity_trainer.modelSaver.QwenFineTuneModelSaver import QwenFineTuneModelSaver
from serenity_trainer.modelSaver.qwen.QwenLoRASaver import QWEN_FMT_SAFETENSORS
from serenity_trainer.util.create import create_model_loader, create_model_sampler, create_model_saver
from serenity_trainer.util.enum.ModelType import MODEL_TYPE_QWEN
from serenity_trainer.util.enum.TrainingMethod import TM_FINE_TUNE, TM_LORA


def main() raises:
    var names = QwenModelNames(
        String("/models/qwen"),
        String(),
        String(),
        String("/models/qwen-lora.safetensors"),
    )
    var dtypes = QwenWeightDtypes.bf16()
    var quantization = QwenQuantizationConfig.default_values()

    var ft_loader = QwenFineTuneModelLoader()
    var load_plan = ft_loader.load(MODEL_TYPE_QWEN, names, dtypes, quantization)
    print("qwen ft loader =", load_plan.model_spec, " transformer =", load_plan.transformer_subfolder)

    var lora_loader = QwenLoRAModelLoader()
    var lora_plan = lora_loader.load(MODEL_TYPE_QWEN, names, dtypes, quantization)
    print("qwen lora loader =", lora_plan.model_spec, " vae override =", lora_plan.override_vae_supported)

    var sampler = QwenSampler(MODEL_TYPE_QWEN)
    var sample_config = QwenSampleConfig(
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
    var sample_plan = sampler.sample(sample_config, String("/tmp/qwen.png"))
    print("qwen sample =", sample_plan.height, "x", sample_plan.width, " c =", sample_plan.latent_channels)

    var ft_saver = QwenFineTuneModelSaver()
    var save_plan = ft_saver.save_plan(
        MODEL_TYPE_QWEN,
        QWEN_FMT_SAFETENSORS,
        String("/tmp/qwen-transformer.safetensors"),
        String("BF16"),
    )
    print("qwen saver =", save_plan.route_name)

    var reg_loader = create_model_loader(MODEL_TYPE_QWEN, TM_LORA)
    var reg_saver = create_model_saver(MODEL_TYPE_QWEN, TM_FINE_TUNE)
    var reg_sampler = create_model_sampler(MODEL_TYPE_QWEN, TM_FINE_TUNE)
    print(reg_loader.implementation, reg_saver.implementation, reg_sampler.implementation)
