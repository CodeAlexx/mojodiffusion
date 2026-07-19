"""Smoke tests for the Ideogram4 ai-toolkit -> Serenity Trainer bridge."""

from serenity_trainer.dataLoader.Ideogram4BaseDataLoader import (
    ideogram4_caption_looks_structured_json,
    ideogram4_data_loader_plan,
)
from serenity_trainer.model.Ideogram4Model import ideogram4_model_contract
from serenity_trainer.modelLoader.Ideogram4ModelLoader import (
    Ideogram4LoRAModelLoader,
    ideogram4_default_model_names,
    ideogram4_default_runtime_flags,
)
from serenity_trainer.modelSampler.Ideogram4Sampler import (
    ideogram4_euler_dt,
    ideogram4_guidance_for_loop_index,
    ideogram4_image_tokens,
    ideogram4_preset_default_20,
    ideogram4_flow_target_scalar,
)
from serenity_trainer.modelSaver.Ideogram4LoRAModelSaver import (
    IDEOGRAM4_FMT_SAFETENSORS,
    Ideogram4LoRAModelSaver,
)
from serenity_trainer.modelSetup.Ideogram4LoRASetup import Ideogram4LoRASetup
from serenity_trainer.modelSetup.ideogram4LoraTargets import (
    ideogram4_convert_lora_key_before_load,
    ideogram4_convert_lora_key_before_save,
    ideogram4_lora_count,
)
from serenity_trainer.util.config.AIToolkitIdeogram4Config import (
    ai_toolkit_ideogram4_to_train_config,
    ai_toolkit_ideogram4_to_trainer_ui_config,
    read_ai_toolkit_ideogram4_config,
)
from serenity_trainer.util.create import (
    create_model_loader,
    create_model_sampler,
    create_model_saver,
)
from serenity_trainer.util.enum.ModelType import (
    MODEL_TYPE_IDEOGRAM_4,
    model_type_is_flow_matching,
    model_type_str,
)
from serenity_trainer.util.enum.TrainingMethod import TM_LORA


def _expect(cond: Bool, msg: String) raises:
    if not cond:
        print("FAIL:", msg)
        raise Error(msg)


def test_model_type_and_factory() raises:
    _expect(model_type_str(MODEL_TYPE_IDEOGRAM_4) == String("IDEOGRAM_4"), "model type string")
    _expect(model_type_is_flow_matching(MODEL_TYPE_IDEOGRAM_4), "Ideogram4 is flow matching")

    var loader = create_model_loader(MODEL_TYPE_IDEOGRAM_4, TM_LORA)
    _expect(loader.implementation == String("Ideogram4LoRAModelLoader"), "loader dispatch")
    _expect(loader.model_spec == String("resources/sd_model_spec/ideogram4-lora.json"), "loader spec")

    var saver = create_model_saver(MODEL_TYPE_IDEOGRAM_4, TM_LORA)
    _expect(saver.implementation == String("Ideogram4LoRAModelSaver"), "saver dispatch")

    var sampler = create_model_sampler(MODEL_TYPE_IDEOGRAM_4, TM_LORA)
    _expect(sampler.implementation == String("Ideogram4Sampler"), "sampler dispatch")


def test_model_sampler_contract() raises:
    var model = ideogram4_model_contract()
    _expect(model.num_layers == 34, "layer count")
    _expect(model.hidden == 4608, "hidden dim")
    _expect(model.intermediate_size == 12288, "mlp dim")
    _expect(model.adaln_dim == 512, "adaLN dim")
    _expect(model.text_feature_dim == 53248, "qwen tap feature dim")
    _expect(model.sequence_padding_indicator == -1, "padding indicator")

    _expect(ideogram4_image_tokens(1024, 1024) == 4096, "1024 image tokens")
    var preset = ideogram4_preset_default_20()
    _expect(preset.num_steps == 20, "default preset steps")
    _expect(ideogram4_guidance_for_loop_index(preset, 0) == Float32(3.0), "polish guidance")
    _expect(ideogram4_guidance_for_loop_index(preset, 2) == Float32(7.0), "main guidance")
    _expect(ideogram4_euler_dt(preset, 1024, 1024, 0) > 0.0, "Euler denoise dt is positive s-t")
    _expect(ideogram4_flow_target_scalar(Float32(0.25), Float32(1.0)) == Float32(0.75), "flow target")


def test_loader_setup_saver_data_contracts() raises:
    var names = ideogram4_default_model_names()
    var flags = ideogram4_default_runtime_flags()
    var loader = Ideogram4LoRAModelLoader()
    var plan = loader.load(MODEL_TYPE_IDEOGRAM_4, names, flags)
    _expect(plan.base_model == String("/home/alex/.serenity/models/ideogram-4-fp8"), "local model root")
    _expect(plan.text_encoder_model == String("Qwen/Qwen3-VL-8B-Instruct"), "Qwen3-VL text encoder")
    _expect(plan.train_scheduler_class == String("CustomFlowMatchEulerDiscreteScheduler"), "training scheduler")
    _expect(plan.model_time_is_one_minus_training_t, "model time conversion")
    _expect(plan.model_output_is_negated_for_training_velocity, "prediction sign conversion")
    _expect(plan.lora_loader_invoked, "LoRA loader flag")
    _expect(plan.native_lora_backward_present, "native final-layer LoRA backward present")
    _expect(plan.native_lora_backward_slice == String("transformer.layers.* + transformer.final_layer.linear"), "native backward slice")

    var setup = Ideogram4LoRASetup()
    _expect(setup.registration.target_count_blocks == 204, "block target count")
    _expect(setup.registration.target_count_full == 211, "full target count")
    _expect(setup.uses_json_captions(), "json caption setup")
    _expect(setup.native_lora_backward_supported(), "setup native final-layer backward")
    _expect(setup.native_trainable_slice() == String("transformer.layers.* + transformer.final_layer.linear"), "setup native slice")

    _expect(ideogram4_lora_count() == 211, "LoRA count helper")
    var converted = ideogram4_convert_lora_key_before_save(String("transformer.layers.0.attention.qkv.lora_down.weight"))
    _expect(converted == String("diffusion_model.layers.0.attention.qkv.lora_down.weight"), "save key conversion")
    var loaded = ideogram4_convert_lora_key_before_load(converted)
    _expect(loaded == String("transformer.layers.0.attention.qkv.lora_down.weight"), "load key conversion")

    var saver = Ideogram4LoRAModelSaver()
    var save_plan = saver.save_plan(MODEL_TYPE_IDEOGRAM_4, IDEOGRAM4_FMT_SAFETENSORS, String("/tmp/ideo.safetensors"))
    _expect(save_plan.converts_transformer_prefix_to_diffusion_model, "saver key conversion")
    _expect(saver.runtime_save_supported(), "saver writes final-layer LoRA")

    var data_plan = ideogram4_data_loader_plan()
    _expect(data_plan.dataset_options.caption_ext == String("json"), "json caption extension")
    _expect(not data_plan.dataset_options.shuffle_tokens, "no token shuffle")
    _expect(data_plan.dataset_options.cache_text_embeddings, "text cache")
    _expect(ideogram4_caption_looks_structured_json(String("{\"compositional_deconstruction\":{\"elements\":[]}}")), "structured caption detector")


def test_ai_toolkit_config_bridge() raises:
    var src = read_ai_toolkit_ideogram4_config(String("/home/alex/ai-toolkit/config/gigerver3_ideogram4_lora.yaml"))
    _expect(src.name == String("gigerver3_ideogram4_lora_v1"), "recipe name")
    _expect(src.model_arch == String("ideogram4"), "recipe arch")
    _expect(src.dataset_folder_path == String("/home/alex/1/datasets/gigerver3_json"), "recipe dataset")
    _expect(src.caption_ext == String("json"), "recipe caption ext")
    _expect(not src.shuffle_tokens, "recipe disables token shuffle")
    _expect(src.cache_text_embeddings, "recipe text cache")
    _expect(src.steps == 2000, "recipe steps")
    _expect(src.gradient_accumulation == 1, "recipe grad accum")
    _expect(src.lora_rank == 16, "recipe rank")
    _expect(src.quantize and src.quantize_te and src.low_vram, "recipe quant flags")
    _expect(len(src.sample_prompts) == 5, "recipe sample prompts")

    var train_cfg = ai_toolkit_ideogram4_to_train_config(src)
    _expect(train_cfg.learning_rate == Float32(0.0001), "train cfg lr")
    _expect(train_cfg.lora_rank == 16, "train cfg rank")
    _expect(train_cfg.gradient_accumulation_steps == 1, "train cfg accum")

    var ui_cfg = ai_toolkit_ideogram4_to_trainer_ui_config(src)
    _expect(ui_cfg.backend_target == String("ideogram4"), "ui backend")
    _expect(ui_cfg.base_model_name == String("/home/alex/.serenity/models/ideogram-4-fp8"), "ui local model root")
    _expect(ui_cfg.max_train_steps == Float32(2000.0), "ui max steps")
    _expect(ui_cfg.caption_extension == String("json"), "ui caption extension")
    _expect(ui_cfg.sample_steps == Float32(20.0), "ui sample steps")
    _expect(len(ui_cfg.samples) == 5, "ui sample prompt count")


def main() raises:
    test_model_type_and_factory()
    test_model_sampler_contract()
    test_loader_setup_saver_data_contracts()
    test_ai_toolkit_config_bridge()
    print("PASS: ideogram4 port contract")
