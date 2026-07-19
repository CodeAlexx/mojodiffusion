# 1:1 surface port of Serenity modules/util/factory.py
#
# Build-only product registrations. Serenity's factory stores Python classes
# keyed by base class plus criteria. Mojo cannot mirror that dynamic class
# registry directly here, so this file exposes typed registration records
# consumed by util/create.mojo.

from serenity_trainer.util.enum.ModelType import (
    MODEL_TYPE_ANIMA,
    MODEL_TYPE_ERNIE,
    MODEL_TYPE_FLUX_DEV_1,
    MODEL_TYPE_FLUX_FILL_DEV_1,
    MODEL_TYPE_FLUX_2,
    MODEL_TYPE_IDEOGRAM_4,
    MODEL_TYPE_QWEN,
    MODEL_TYPE_STABLE_DIFFUSION_35,
    MODEL_TYPE_STABLE_DIFFUSION_XL_10_BASE,
    MODEL_TYPE_STABLE_DIFFUSION_XL_10_BASE_INPAINTING,
    model_type_is_flux_1,
    model_type_is_flux_2,
    model_type_is_stable_diffusion_3,
    model_type_is_stable_diffusion_xl,
    model_type_str,
)
from serenity_trainer.util.enum.TrainingMethod import TM_EMBEDDING, TM_FINE_TUNE, TM_LORA


comptime FACTORY_BASE_MODEL_LOADER = 0
comptime FACTORY_BASE_MODEL_SAVER = 1
comptime FACTORY_BASE_MODEL_SAMPLER = 2
comptime FACTORY_TRAINING_METHOD_ANY = -1
struct FactoryRegistration(Movable):
    var base_kind: Int
    var model_type: Int
    var training_method: Int
    var implementation: String
    var model_spec: String

    def __init__(
        out self,
        base_kind: Int,
        model_type: Int,
        training_method: Int,
        var implementation: String,
        var model_spec: String,
    ):
        self.base_kind = base_kind
        self.model_type = model_type
        self.training_method = training_method
        self.implementation = implementation^
        self.model_spec = model_spec^


def qwen_model_loader_registration(training_method: Int) raises -> FactoryRegistration:
    if training_method == TM_LORA:
        return FactoryRegistration(
            FACTORY_BASE_MODEL_LOADER,
            MODEL_TYPE_QWEN,
            TM_LORA,
            String("QwenLoRAModelLoader"),
            String("resources/sd_model_spec/qwen-lora.json"),
        )
    if training_method == TM_FINE_TUNE:
        return FactoryRegistration(
            FACTORY_BASE_MODEL_LOADER,
            MODEL_TYPE_QWEN,
            TM_FINE_TUNE,
            String("QwenFineTuneModelLoader"),
            String("resources/sd_model_spec/qwen.json"),
        )
    raise Error("factory: no Qwen model loader for training method")


def qwen_model_saver_registration(training_method: Int) raises -> FactoryRegistration:
    if training_method == TM_LORA:
        return FactoryRegistration(
            FACTORY_BASE_MODEL_SAVER,
            MODEL_TYPE_QWEN,
            TM_LORA,
            String("QwenLoRAModelSaver"),
            String(),
        )
    if training_method == TM_FINE_TUNE:
        return FactoryRegistration(
            FACTORY_BASE_MODEL_SAVER,
            MODEL_TYPE_QWEN,
            TM_FINE_TUNE,
            String("QwenFineTuneModelSaver"),
            String(),
        )
    raise Error("factory: no Qwen model saver for training method")


def qwen_model_sampler_registration(training_method: Int) raises -> FactoryRegistration:
    _ = training_method
    # QwenSampler.py registers by ModelType only, not by training method. This is
    # the class Serenity create_model_sampler finds through its fallback lookup.
    return FactoryRegistration(
        FACTORY_BASE_MODEL_SAMPLER,
        MODEL_TYPE_QWEN,
        FACTORY_TRAINING_METHOD_ANY,
        String("QwenSampler"),
        String(),
    )


def qwen_model_sampler_registration_by_training_method(training_method: Int) raises -> FactoryRegistration:
    if training_method == TM_LORA or training_method == TM_FINE_TUNE:
        return FactoryRegistration(
            FACTORY_BASE_MODEL_SAMPLER,
            MODEL_TYPE_QWEN,
            training_method,
            String("QwenSampler"),
            String(),
        )
    raise Error("factory: no Qwen model sampler for training method")


def get_qwen_registration(
    base_kind: Int,
    model_type: Int,
    training_method: Int,
) raises -> FactoryRegistration:
    if model_type != MODEL_TYPE_QWEN:
        raise Error("factory: requested model is not Qwen")
    if base_kind == FACTORY_BASE_MODEL_LOADER:
        return qwen_model_loader_registration(training_method)
    if base_kind == FACTORY_BASE_MODEL_SAVER:
        return qwen_model_saver_registration(training_method)
    if base_kind == FACTORY_BASE_MODEL_SAMPLER:
        return qwen_model_sampler_registration(training_method)
    raise Error("factory: unsupported base kind")


def stable_diffusion3_model_loader_registration(
    model_type: Int,
    training_method: Int,
) raises -> FactoryRegistration:
    if not model_type_is_stable_diffusion_3(model_type):
        raise Error("factory: requested model is not StableDiffusion3")
    if training_method == TM_LORA:
        if model_type == MODEL_TYPE_STABLE_DIFFUSION_35:
            return FactoryRegistration(
                FACTORY_BASE_MODEL_LOADER,
                model_type,
                TM_LORA,
                String("StableDiffusion3LoRAModelLoader"),
                String("resources/sd_model_spec/sd_3.5_1.0-lora.json"),
            )
        return FactoryRegistration(
            FACTORY_BASE_MODEL_LOADER,
            model_type,
            TM_LORA,
            String("StableDiffusion3LoRAModelLoader"),
            String("resources/sd_model_spec/sd_3_2b_1.0-lora.json"),
        )
    if training_method == TM_EMBEDDING:
        if model_type == MODEL_TYPE_STABLE_DIFFUSION_35:
            return FactoryRegistration(
                FACTORY_BASE_MODEL_LOADER,
                model_type,
                TM_EMBEDDING,
                String("StableDiffusion3EmbeddingModelLoader"),
                String("resources/sd_model_spec/sd_3.5_1.0-embedding.json"),
            )
        return FactoryRegistration(
            FACTORY_BASE_MODEL_LOADER,
            model_type,
            TM_EMBEDDING,
            String("StableDiffusion3EmbeddingModelLoader"),
            String("resources/sd_model_spec/sd_3_2b_1.0-embedding.json"),
        )
    if training_method == TM_FINE_TUNE:
        if model_type == MODEL_TYPE_STABLE_DIFFUSION_35:
            return FactoryRegistration(
                FACTORY_BASE_MODEL_LOADER,
                model_type,
                TM_FINE_TUNE,
                String("StableDiffusion3FineTuneModelLoader"),
                String("resources/sd_model_spec/sd_3.5_1.0.json"),
            )
        return FactoryRegistration(
            FACTORY_BASE_MODEL_LOADER,
            model_type,
            TM_FINE_TUNE,
            String("StableDiffusion3FineTuneModelLoader"),
            String("resources/sd_model_spec/sd_3_2b_1.0.json"),
        )
    raise Error("factory: no StableDiffusion3 model loader for training method")


def stable_diffusion3_model_saver_registration(
    model_type: Int,
    training_method: Int,
) raises -> FactoryRegistration:
    if not model_type_is_stable_diffusion_3(model_type):
        raise Error("factory: requested model is not StableDiffusion3")
    if training_method == TM_LORA:
        return FactoryRegistration(
            FACTORY_BASE_MODEL_SAVER,
            model_type,
            TM_LORA,
            String("StableDiffusion3LoRAModelSaver"),
            String(),
        )
    if training_method == TM_EMBEDDING:
        return FactoryRegistration(
            FACTORY_BASE_MODEL_SAVER,
            model_type,
            TM_EMBEDDING,
            String("StableDiffusion3EmbeddingModelSaver"),
            String(),
        )
    if training_method == TM_FINE_TUNE:
        return FactoryRegistration(
            FACTORY_BASE_MODEL_SAVER,
            model_type,
            TM_FINE_TUNE,
            String("StableDiffusion3FineTuneModelSaver"),
            String(),
        )
    raise Error("factory: no StableDiffusion3 model saver for training method")


def stable_diffusion3_model_sampler_registration(
    model_type: Int,
    training_method: Int,
) raises -> FactoryRegistration:
    _ = training_method
    if not model_type_is_stable_diffusion_3(model_type):
        raise Error("factory: requested model is not StableDiffusion3")
    # StableDiffusion3Sampler.py registers by ModelType only. Serenity
    # create_model_sampler falls back to this registration when no
    # training-method keyed sampler exists.
    return FactoryRegistration(
        FACTORY_BASE_MODEL_SAMPLER,
        model_type,
        FACTORY_TRAINING_METHOD_ANY,
        String("StableDiffusion3Sampler"),
        String(),
    )


def get_stable_diffusion3_registration(
    base_kind: Int,
    model_type: Int,
    training_method: Int,
) raises -> FactoryRegistration:
    if not model_type_is_stable_diffusion_3(model_type):
        raise Error(String("factory: requested model is not StableDiffusion3: ") + model_type_str(model_type))
    if base_kind == FACTORY_BASE_MODEL_LOADER:
        return stable_diffusion3_model_loader_registration(model_type, training_method)
    if base_kind == FACTORY_BASE_MODEL_SAVER:
        return stable_diffusion3_model_saver_registration(model_type, training_method)
    if base_kind == FACTORY_BASE_MODEL_SAMPLER:
        return stable_diffusion3_model_sampler_registration(model_type, training_method)
    raise Error("factory: unsupported base kind")


def stable_diffusion_xl_model_loader_registration(
    model_type: Int,
    training_method: Int,
) raises -> FactoryRegistration:
    if not model_type_is_stable_diffusion_xl(model_type):
        raise Error("factory: requested model is not StableDiffusionXL")
    if training_method == TM_LORA:
        if model_type == MODEL_TYPE_STABLE_DIFFUSION_XL_10_BASE_INPAINTING:
            return FactoryRegistration(
                FACTORY_BASE_MODEL_LOADER,
                model_type,
                TM_LORA,
                String("StableDiffusionXLLoRAModelLoader"),
                String("resources/sd_model_spec/sd_xl_base_1.0_inpainting-lora.json"),
            )
        return FactoryRegistration(
            FACTORY_BASE_MODEL_LOADER,
            model_type,
            TM_LORA,
            String("StableDiffusionXLLoRAModelLoader"),
            String("resources/sd_model_spec/sd_xl_base_1.0-lora.json"),
        )
    if training_method == TM_EMBEDDING:
        if model_type == MODEL_TYPE_STABLE_DIFFUSION_XL_10_BASE_INPAINTING:
            return FactoryRegistration(
                FACTORY_BASE_MODEL_LOADER,
                model_type,
                TM_EMBEDDING,
                String("StableDiffusionXLEmbeddingModelLoader"),
                String("resources/sd_model_spec/sd_xl_base_1.0_inpainting-embedding.json"),
            )
        return FactoryRegistration(
            FACTORY_BASE_MODEL_LOADER,
            model_type,
            TM_EMBEDDING,
            String("StableDiffusionXLEmbeddingModelLoader"),
            String("resources/sd_model_spec/sd_xl_base_1.0-embedding.json"),
        )
    if training_method == TM_FINE_TUNE:
        if model_type == MODEL_TYPE_STABLE_DIFFUSION_XL_10_BASE_INPAINTING:
            return FactoryRegistration(
                FACTORY_BASE_MODEL_LOADER,
                model_type,
                TM_FINE_TUNE,
                String("StableDiffusionXLFineTuneModelLoader"),
                String("resources/sd_model_spec/sd_xl_base_1.0_inpainting.json"),
            )
        return FactoryRegistration(
            FACTORY_BASE_MODEL_LOADER,
            model_type,
            TM_FINE_TUNE,
            String("StableDiffusionXLFineTuneModelLoader"),
            String("resources/sd_model_spec/sd_xl_base_1.0.json"),
        )
    raise Error("factory: no StableDiffusionXL model loader for training method")


def stable_diffusion_xl_model_saver_registration(
    model_type: Int,
    training_method: Int,
) raises -> FactoryRegistration:
    if not model_type_is_stable_diffusion_xl(model_type):
        raise Error("factory: requested model is not StableDiffusionXL")
    if training_method == TM_LORA:
        return FactoryRegistration(
            FACTORY_BASE_MODEL_SAVER,
            model_type,
            TM_LORA,
            String("StableDiffusionXLLoRAModelSaver"),
            String(),
        )
    if training_method == TM_EMBEDDING:
        return FactoryRegistration(
            FACTORY_BASE_MODEL_SAVER,
            model_type,
            TM_EMBEDDING,
            String("StableDiffusionXLEmbeddingModelSaver"),
            String(),
        )
    if training_method == TM_FINE_TUNE:
        return FactoryRegistration(
            FACTORY_BASE_MODEL_SAVER,
            model_type,
            TM_FINE_TUNE,
            String("StableDiffusionXLFineTuneModelSaver"),
            String(),
        )
    raise Error("factory: no StableDiffusionXL model saver for training method")


def stable_diffusion_xl_model_sampler_registration(
    model_type: Int,
    training_method: Int,
) raises -> FactoryRegistration:
    _ = training_method
    if not model_type_is_stable_diffusion_xl(model_type):
        raise Error("factory: requested model is not StableDiffusionXL")
    # StableDiffusionXLSampler.py registers by ModelType only. Serenity
    # create_model_sampler falls back to this registration when no
    # training-method keyed sampler exists.
    return FactoryRegistration(
        FACTORY_BASE_MODEL_SAMPLER,
        model_type,
        FACTORY_TRAINING_METHOD_ANY,
        String("StableDiffusionXLSampler"),
        String(),
    )


def get_stable_diffusion_xl_registration(
    base_kind: Int,
    model_type: Int,
    training_method: Int,
) raises -> FactoryRegistration:
    if not model_type_is_stable_diffusion_xl(model_type):
        raise Error(String("factory: requested model is not StableDiffusionXL: ") + model_type_str(model_type))
    if base_kind == FACTORY_BASE_MODEL_LOADER:
        return stable_diffusion_xl_model_loader_registration(model_type, training_method)
    if base_kind == FACTORY_BASE_MODEL_SAVER:
        return stable_diffusion_xl_model_saver_registration(model_type, training_method)
    if base_kind == FACTORY_BASE_MODEL_SAMPLER:
        return stable_diffusion_xl_model_sampler_registration(model_type, training_method)
    raise Error("factory: unsupported base kind")


def flux1_model_loader_spec(model_type: Int, training_method: Int) raises -> String:
    if model_type == MODEL_TYPE_FLUX_FILL_DEV_1:
        if training_method == TM_LORA:
            return String("resources/sd_model_spec/flux_dev_fill_1.0-lora.json")
        if training_method == TM_EMBEDDING:
            return String("resources/sd_model_spec/flux_dev_fill_1.0-embedding.json")
        if training_method == TM_FINE_TUNE:
            return String("resources/sd_model_spec/flux_dev_fill_1.0.json")
    if model_type == MODEL_TYPE_FLUX_DEV_1:
        if training_method == TM_LORA:
            return String("resources/sd_model_spec/flux_dev_1.0-lora.json")
        if training_method == TM_EMBEDDING:
            return String("resources/sd_model_spec/flux_dev_1.0-embedding.json")
        if training_method == TM_FINE_TUNE:
            return String("resources/sd_model_spec/flux_dev_1.0.json")
    raise Error("factory: no Flux.1 model spec for model type/training method")


def flux1_model_loader_registration(
    model_type: Int,
    training_method: Int,
) raises -> FactoryRegistration:
    if not model_type_is_flux_1(model_type):
        raise Error("factory: requested model is not Flux.1")
    if training_method == TM_LORA:
        return FactoryRegistration(
            FACTORY_BASE_MODEL_LOADER,
            model_type,
            TM_LORA,
            String("FluxLoRAModelLoader"),
            flux1_model_loader_spec(model_type, training_method),
        )
    if training_method == TM_EMBEDDING:
        return FactoryRegistration(
            FACTORY_BASE_MODEL_LOADER,
            model_type,
            TM_EMBEDDING,
            String("FluxEmbeddingModelLoader"),
            flux1_model_loader_spec(model_type, training_method),
        )
    if training_method == TM_FINE_TUNE:
        return FactoryRegistration(
            FACTORY_BASE_MODEL_LOADER,
            model_type,
            TM_FINE_TUNE,
            String("FluxFineTuneModelLoader"),
            flux1_model_loader_spec(model_type, training_method),
        )
    raise Error("factory: no Flux.1 model loader for training method")


def flux1_model_saver_registration(
    model_type: Int,
    training_method: Int,
) raises -> FactoryRegistration:
    if not model_type_is_flux_1(model_type):
        raise Error("factory: requested model is not Flux.1")
    if training_method == TM_LORA:
        return FactoryRegistration(
            FACTORY_BASE_MODEL_SAVER,
            model_type,
            TM_LORA,
            String("FluxLoRAModelSaver"),
            String(),
        )
    if training_method == TM_EMBEDDING:
        return FactoryRegistration(
            FACTORY_BASE_MODEL_SAVER,
            model_type,
            TM_EMBEDDING,
            String("FluxEmbeddingModelSaver"),
            String(),
        )
    if training_method == TM_FINE_TUNE:
        return FactoryRegistration(
            FACTORY_BASE_MODEL_SAVER,
            model_type,
            TM_FINE_TUNE,
            String("FluxFineTuneModelSaver"),
            String(),
        )
    raise Error("factory: no Flux.1 model saver for training method")


def flux1_model_sampler_registration(
    model_type: Int,
    training_method: Int,
) raises -> FactoryRegistration:
    _ = training_method
    if not model_type_is_flux_1(model_type):
        raise Error("factory: requested model is not Flux.1")
    # FluxSampler.py registers by ModelType only. Serenity create_model_sampler
    # falls back to this registration when no training-method keyed sampler exists.
    return FactoryRegistration(
        FACTORY_BASE_MODEL_SAMPLER,
        model_type,
        FACTORY_TRAINING_METHOD_ANY,
        String("FluxSampler"),
        String(),
    )


def get_flux1_registration(
    base_kind: Int,
    model_type: Int,
    training_method: Int,
) raises -> FactoryRegistration:
    if not model_type_is_flux_1(model_type):
        raise Error(String("factory: requested model is not Flux.1: ") + model_type_str(model_type))
    if base_kind == FACTORY_BASE_MODEL_LOADER:
        return flux1_model_loader_registration(model_type, training_method)
    if base_kind == FACTORY_BASE_MODEL_SAVER:
        return flux1_model_saver_registration(model_type, training_method)
    if base_kind == FACTORY_BASE_MODEL_SAMPLER:
        return flux1_model_sampler_registration(model_type, training_method)
    raise Error("factory: unsupported base kind")


def flux2_model_loader_registration(training_method: Int) raises -> FactoryRegistration:
    if training_method == TM_LORA:
        return FactoryRegistration(
            FACTORY_BASE_MODEL_LOADER,
            MODEL_TYPE_FLUX_2,
            TM_LORA,
            String("Flux2LoRAModelLoader"),
            String("resources/sd_model_spec/flux_2.0-lora.json"),
        )
    if training_method == TM_FINE_TUNE:
        return FactoryRegistration(
            FACTORY_BASE_MODEL_LOADER,
            MODEL_TYPE_FLUX_2,
            TM_FINE_TUNE,
            String("Flux2FineTuneModelLoader"),
            String("resources/sd_model_spec/flux_2.0.json"),
        )
    raise Error("factory: no Flux2 model loader for training method")


def flux2_model_saver_registration(training_method: Int) raises -> FactoryRegistration:
    if training_method == TM_LORA:
        return FactoryRegistration(
            FACTORY_BASE_MODEL_SAVER,
            MODEL_TYPE_FLUX_2,
            TM_LORA,
            String("Flux2LoRAModelSaver"),
            String(),
        )
    if training_method == TM_FINE_TUNE:
        return FactoryRegistration(
            FACTORY_BASE_MODEL_SAVER,
            MODEL_TYPE_FLUX_2,
            TM_FINE_TUNE,
            String("Flux2FineTuneModelSaver"),
            String(),
        )
    raise Error("factory: no Flux2 model saver for training method")


def flux2_model_sampler_registration(training_method: Int) raises -> FactoryRegistration:
    _ = training_method
    # Flux2Sampler.py registers by ModelType only. Serenity create_model_sampler
    # falls back to this registration when no training-method keyed sampler exists.
    return FactoryRegistration(
        FACTORY_BASE_MODEL_SAMPLER,
        MODEL_TYPE_FLUX_2,
        FACTORY_TRAINING_METHOD_ANY,
        String("Flux2Sampler"),
        String(),
    )


def get_flux2_registration(
    base_kind: Int,
    model_type: Int,
    training_method: Int,
) raises -> FactoryRegistration:
    if not model_type_is_flux_2(model_type):
        raise Error(String("factory: requested model is not Flux2: ") + model_type_str(model_type))
    if base_kind == FACTORY_BASE_MODEL_LOADER:
        return flux2_model_loader_registration(training_method)
    if base_kind == FACTORY_BASE_MODEL_SAVER:
        return flux2_model_saver_registration(training_method)
    if base_kind == FACTORY_BASE_MODEL_SAMPLER:
        return flux2_model_sampler_registration(training_method)
    raise Error("factory: unsupported base kind")


def ideogram4_model_loader_registration(training_method: Int) raises -> FactoryRegistration:
    if training_method == TM_LORA:
        return FactoryRegistration(
            FACTORY_BASE_MODEL_LOADER,
            MODEL_TYPE_IDEOGRAM_4,
            TM_LORA,
            String("Ideogram4LoRAModelLoader"),
            String("resources/sd_model_spec/ideogram4-lora.json"),
        )
    if training_method == TM_FINE_TUNE:
        return FactoryRegistration(
            FACTORY_BASE_MODEL_LOADER,
            MODEL_TYPE_IDEOGRAM_4,
            TM_FINE_TUNE,
            String("Ideogram4FineTuneModelLoader"),
            String("resources/sd_model_spec/ideogram4.json"),
        )
    raise Error("factory: no Ideogram4 model loader for training method")


def ideogram4_model_saver_registration(training_method: Int) raises -> FactoryRegistration:
    if training_method == TM_LORA:
        return FactoryRegistration(
            FACTORY_BASE_MODEL_SAVER,
            MODEL_TYPE_IDEOGRAM_4,
            TM_LORA,
            String("Ideogram4LoRAModelSaver"),
            String(),
        )
    if training_method == TM_FINE_TUNE:
        return FactoryRegistration(
            FACTORY_BASE_MODEL_SAVER,
            MODEL_TYPE_IDEOGRAM_4,
            TM_FINE_TUNE,
            String("Ideogram4FineTuneModelSaver"),
            String(),
        )
    raise Error("factory: no Ideogram4 model saver for training method")


def ideogram4_model_sampler_registration(training_method: Int) raises -> FactoryRegistration:
    _ = training_method
    return FactoryRegistration(
        FACTORY_BASE_MODEL_SAMPLER,
        MODEL_TYPE_IDEOGRAM_4,
        FACTORY_TRAINING_METHOD_ANY,
        String("Ideogram4Sampler"),
        String(),
    )


def get_ideogram4_registration(
    base_kind: Int,
    model_type: Int,
    training_method: Int,
) raises -> FactoryRegistration:
    if model_type != MODEL_TYPE_IDEOGRAM_4:
        raise Error(String("factory: requested model is not Ideogram4: ") + model_type_str(model_type))
    if base_kind == FACTORY_BASE_MODEL_LOADER:
        return ideogram4_model_loader_registration(training_method)
    if base_kind == FACTORY_BASE_MODEL_SAVER:
        return ideogram4_model_saver_registration(training_method)
    if base_kind == FACTORY_BASE_MODEL_SAMPLER:
        return ideogram4_model_sampler_registration(training_method)
    raise Error("factory: unsupported base kind")


def anima_model_loader_registration(training_method: Int) raises -> FactoryRegistration:
    if training_method == TM_LORA:
        return FactoryRegistration(
            FACTORY_BASE_MODEL_LOADER,
            MODEL_TYPE_ANIMA,
            TM_LORA,
            String("AnimaLoRAModelLoader"),
            String("resources/sd_model_spec/anima-lora.json"),
        )
    if training_method == TM_FINE_TUNE:
        return FactoryRegistration(
            FACTORY_BASE_MODEL_LOADER,
            MODEL_TYPE_ANIMA,
            TM_FINE_TUNE,
            String("AnimaFineTuneModelLoader"),
            String("resources/sd_model_spec/anima.json"),
        )
    raise Error("factory: no Anima model loader for training method")


def anima_model_saver_registration(training_method: Int) raises -> FactoryRegistration:
    if training_method == TM_LORA:
        return FactoryRegistration(
            FACTORY_BASE_MODEL_SAVER,
            MODEL_TYPE_ANIMA,
            TM_LORA,
            String("AnimaLoRAModelSaver"),
            String(),
        )
    if training_method == TM_FINE_TUNE:
        return FactoryRegistration(
            FACTORY_BASE_MODEL_SAVER,
            MODEL_TYPE_ANIMA,
            TM_FINE_TUNE,
            String("AnimaFineTuneModelSaver"),
            String(),
        )
    raise Error("factory: no Anima model saver for training method")


def anima_model_sampler_registration(training_method: Int) raises -> FactoryRegistration:
    _ = training_method
    # AnimaSampler.py registers by ModelType only, not by training method. This is
    # the class Serenity create_model_sampler finds through its fallback lookup.
    return FactoryRegistration(
        FACTORY_BASE_MODEL_SAMPLER,
        MODEL_TYPE_ANIMA,
        FACTORY_TRAINING_METHOD_ANY,
        String("AnimaSampler"),
        String(),
    )


def anima_model_sampler_registration_by_training_method(training_method: Int) raises -> FactoryRegistration:
    if training_method == TM_LORA or training_method == TM_FINE_TUNE:
        return FactoryRegistration(
            FACTORY_BASE_MODEL_SAMPLER,
            MODEL_TYPE_ANIMA,
            training_method,
            String("AnimaSampler"),
            String(),
        )
    raise Error("factory: no Anima model sampler for training method")


def get_anima_registration(
    base_kind: Int,
    model_type: Int,
    training_method: Int,
) raises -> FactoryRegistration:
    if model_type != MODEL_TYPE_ANIMA:
        raise Error("factory: requested model is not Anima")
    if base_kind == FACTORY_BASE_MODEL_LOADER:
        return anima_model_loader_registration(training_method)
    if base_kind == FACTORY_BASE_MODEL_SAVER:
        return anima_model_saver_registration(training_method)
    if base_kind == FACTORY_BASE_MODEL_SAMPLER:
        return anima_model_sampler_registration(training_method)
    raise Error("factory: unsupported base kind")


def ernie_model_loader_registration(training_method: Int) raises -> FactoryRegistration:
    if training_method == TM_LORA:
        return FactoryRegistration(
            FACTORY_BASE_MODEL_LOADER,
            MODEL_TYPE_ERNIE,
            TM_LORA,
            String("ErnieLoRAModelLoader"),
            String("resources/sd_model_spec/ernie-lora.json"),
        )
    if training_method == TM_FINE_TUNE:
        return FactoryRegistration(
            FACTORY_BASE_MODEL_LOADER,
            MODEL_TYPE_ERNIE,
            TM_FINE_TUNE,
            String("ErnieFineTuneModelLoader"),
            String("resources/sd_model_spec/ernie.json"),
        )
    raise Error("factory: no Ernie model loader for training method")


def ernie_model_saver_registration(training_method: Int) raises -> FactoryRegistration:
    if training_method == TM_LORA:
        return FactoryRegistration(
            FACTORY_BASE_MODEL_SAVER,
            MODEL_TYPE_ERNIE,
            TM_LORA,
            String("ErnieLoRAModelSaver"),
            String(),
        )
    if training_method == TM_FINE_TUNE:
        return FactoryRegistration(
            FACTORY_BASE_MODEL_SAVER,
            MODEL_TYPE_ERNIE,
            TM_FINE_TUNE,
            String("ErnieFineTuneModelSaver"),
            String(),
        )
    raise Error("factory: no Ernie model saver for training method")


def ernie_model_sampler_registration(training_method: Int) raises -> FactoryRegistration:
    _ = training_method
    # ErnieSampler.py registers by ModelType only, not by training method. This is
    # the class Serenity create_model_sampler finds through its fallback lookup.
    return FactoryRegistration(
        FACTORY_BASE_MODEL_SAMPLER,
        MODEL_TYPE_ERNIE,
        FACTORY_TRAINING_METHOD_ANY,
        String("ErnieSampler"),
        String(),
    )


def ernie_model_sampler_registration_by_training_method(training_method: Int) raises -> FactoryRegistration:
    if training_method == TM_LORA or training_method == TM_FINE_TUNE:
        return FactoryRegistration(
            FACTORY_BASE_MODEL_SAMPLER,
            MODEL_TYPE_ERNIE,
            training_method,
            String("ErnieSampler"),
            String(),
        )
    raise Error("factory: no Ernie model sampler for training method")


def get_ernie_registration(
    base_kind: Int,
    model_type: Int,
    training_method: Int,
) raises -> FactoryRegistration:
    if model_type != MODEL_TYPE_ERNIE:
        raise Error("factory: requested model is not Ernie")
    if base_kind == FACTORY_BASE_MODEL_LOADER:
        return ernie_model_loader_registration(training_method)
    if base_kind == FACTORY_BASE_MODEL_SAVER:
        return ernie_model_saver_registration(training_method)
    if base_kind == FACTORY_BASE_MODEL_SAMPLER:
        return ernie_model_sampler_registration(training_method)
    raise Error("factory: unsupported base kind")
