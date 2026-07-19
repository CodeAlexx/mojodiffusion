# FLUX.1 Dev loader/saver/sampler surface smoke.
#
# This is not a parity gate and does not generate an image. It only instantiates
# typed FLUX plan surfaces and verifies shared util/create dispatch.

from serenity_trainer.modelLoader.flux.FluxModelLoader import (
    FluxEmbeddingName,
    FluxModelNames,
    FluxQuantizationConfig,
    FluxWeightDtypes,
)
from serenity_trainer.modelLoader.flux.FluxLoRALoader import (
    flux_lora_loader_has_convert_key_sets,
)
from serenity_trainer.modelLoader.FluxEmbeddingModelLoader import FluxEmbeddingModelLoader
from serenity_trainer.modelLoader.FluxFineTuneModelLoader import FluxFineTuneModelLoader
from serenity_trainer.modelLoader.FluxLoRAModelLoader import FluxLoRAModelLoader
from serenity_trainer.modelSampler.FluxSampler import FluxSampleConfig, FluxSampler
from serenity_trainer.modelSaver.FluxEmbeddingModelSaver import FluxEmbeddingModelSaver
from serenity_trainer.modelSaver.FluxFineTuneModelSaver import FluxFineTuneModelSaver
from serenity_trainer.modelSaver.FluxLoRAModelSaver import FluxLoRAModelSaver
from serenity_trainer.modelSaver.flux.FluxEmbeddingSaver import flux_embedding_keys
from serenity_trainer.modelSaver.flux.FluxLoRASaver import (
    FLUX_FMT_INTERNAL,
    FLUX_FMT_SAFETENSORS,
    flux_lora_bundle_embedding_keys,
)
from serenity_trainer.modelSaver.flux.FluxModelSaver import flux_checkpoint_key_roots
from serenity_trainer.util.enum.ModelType import (
    MODEL_TYPE_FLUX_DEV_1,
    MODEL_TYPE_FLUX_FILL_DEV_1,
)
from serenity_trainer.util.create import (
    create_model_loader,
    create_model_sampler,
    create_model_saver,
)
from serenity_trainer.util.enum.TrainingMethod import TM_EMBEDDING, TM_FINE_TUNE, TM_LORA


def main() raises:
    var names = FluxModelNames(
        String("/models/flux-dev"),
        String("/models/flux-transformer.safetensors"),
        String("/models/flux-vae"),
        String("/models/flux-lora.safetensors"),
        FluxEmbeddingName(String("emb-uuid"), String("/models/flux-embedding.safetensors")),
        True,
        True,
    )
    var dtypes = FluxWeightDtypes.bf16()
    var quantization = FluxQuantizationConfig.default_values()

    var ft_loader = FluxFineTuneModelLoader()
    var ft_plan = ft_loader.load(MODEL_TYPE_FLUX_DEV_1, names, dtypes, quantization)
    print("flux ft loader =", ft_plan.model_spec, " transformer =", ft_plan.transformer_class)

    var lora_loader = FluxLoRAModelLoader()
    var lora_plan = lora_loader.load(MODEL_TYPE_FLUX_FILL_DEV_1, names, dtypes, quantization)
    print("flux fill lora loader =", lora_plan.model_spec, " lora invoked =", lora_plan.lora_loader_invoked)
    print("flux lora convert keys =", flux_lora_loader_has_convert_key_sets())

    var embedding_loader = FluxEmbeddingModelLoader()
    var embedding_plan = embedding_loader.load(MODEL_TYPE_FLUX_DEV_1, names, dtypes, quantization)
    print("flux embedding loader =", embedding_plan.model_spec, " embedding invoked =", embedding_plan.embedding_loader_invoked)

    var sampler = FluxSampler(MODEL_TYPE_FLUX_FILL_DEV_1)
    var sample_config = FluxSampleConfig(
        String("prompt"),
        String("negative"),
        1025,
        1025,
        7,
        False,
        4,
        Float32(3.5),
        0,
        True,
        String("/tmp/base.png"),
        String("/tmp/mask.png"),
    )
    var sample_plan = sampler.sample(sample_config, String("/tmp/flux.png"))
    print("flux sample =", sample_plan.height, "x", sample_plan.width, " packed c =", sample_plan.packed_latent_channels)
    print("flux negative prompt used =", sample_plan.negative_prompt_used, " fill =", sample_plan.inpainting_model_type)

    var ft_saver = FluxFineTuneModelSaver()
    var save_plan = ft_saver.save_plan(
        MODEL_TYPE_FLUX_DEV_1,
        FLUX_FMT_SAFETENSORS,
        String("/tmp/flux-transformer.safetensors"),
        String("BF16"),
    )
    print("flux saver =", save_plan.route_name, " converter =", save_plan.converter_name)

    var lora_saver = FluxLoRAModelSaver()
    var lora_save_plan = lora_saver.save_plan(
        MODEL_TYPE_FLUX_FILL_DEV_1,
        FLUX_FMT_INTERNAL,
        String("/tmp/flux-internal"),
    )
    print("flux lora saver =", lora_save_plan.route_name, " keys =", lora_save_plan.bundled_embedding_keys)

    var embedding_saver = FluxEmbeddingModelSaver()
    var embedding_save_plan = embedding_saver.save_multiple_plan(
        MODEL_TYPE_FLUX_DEV_1,
        FLUX_FMT_SAFETENSORS,
        String("/tmp/flux-embedding.safetensors"),
    )
    print("flux embedding saver =", embedding_save_plan.route_name, " multiple =", embedding_save_plan.is_multiple)

    var bundle_keys = flux_lora_bundle_embedding_keys(String("tok"))
    var embedding_keys = flux_embedding_keys()
    var checkpoint_roots = flux_checkpoint_key_roots()
    print("flux bundle keys =", len(bundle_keys), " embedding keys =", len(embedding_keys), " ckpt roots =", len(checkpoint_roots))
    var dispatch_loader = create_model_loader(MODEL_TYPE_FLUX_FILL_DEV_1, TM_LORA)
    var dispatch_embedding_loader = create_model_loader(MODEL_TYPE_FLUX_DEV_1, TM_EMBEDDING)
    var dispatch_saver = create_model_saver(MODEL_TYPE_FLUX_DEV_1, TM_FINE_TUNE)
    var dispatch_sampler = create_model_sampler(MODEL_TYPE_FLUX_FILL_DEV_1, TM_LORA)
    print(
        "flux dispatch =",
        dispatch_loader.implementation,
        dispatch_embedding_loader.model_spec,
        dispatch_saver.implementation,
        dispatch_sampler.implementation,
    )
    print("FLUX SURFACE OK")
