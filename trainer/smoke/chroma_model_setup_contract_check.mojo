# Chroma model/setup contract smoke.
#
# This is not a parity gate and does not run Serenity. It verifies the
# build-only ChromaModel, BaseChromaSetup, ChromaFineTuneSetup, and
# ChromaLoRASetup surfaces
# against Serenity-visible constants, group ordering, mask behavior, and
# latent shape contracts.

from serenity_trainer.model.ChromaModel import (
    ChromaLatentShape,
    ChromaModel,
    chroma_pack_latents_shape,
    chroma_prepare_latent_image_ids_shape,
    chroma_text_encode_contract,
    chroma_unpack_latents_shape,
)
from serenity_trainer.modelSetup.BaseChromaSetup import (
    BaseChromaSetup,
    CHROMA_LAYER_PRESET_ATTN_MLP,
    chroma_deterministic_timestep_index,
    chroma_model_t_from_timestep,
    chroma_packed_latent_channels,
    chroma_packed_latent_token_count,
    chroma_sigma_from_timestep,
)
from serenity_trainer.modelSetup.ChromaFineTuneSetup import ChromaFineTuneSetup
from serenity_trainer.modelSetup.ChromaLoRASetup import ChromaLoRASetup
from serenity_trainer.util.enum.ModelType import MODEL_TYPE_CHROMA_1
from serenity_trainer.util.enum.TrainingMethod import TM_FINE_TUNE, TM_LORA


def _expect(name: String, got: Int, expected: Int) raises:
    if got != expected:
        raise Error(
            name + String(" got=") + String(got)
            + String(" expected=") + String(expected)
        )


def _expect_bool(name: String, got: Bool, expected: Bool) raises:
    if got != expected:
        raise Error(
            name + String(" got=") + String(got)
            + String(" expected=") + String(expected)
        )


def _expect_string(name: String, got: String, expected: String) raises:
    if got != expected:
        raise Error(name + String(" got=") + got + String(" expected=") + expected)


def main() raises:
    var model = ChromaModel()
    model.has_tokenizer = True
    model.has_noise_scheduler = True
    model.has_text_encoder = True
    model.has_vae = True
    model.has_transformer = True
    model.has_embedding = True
    model.additional_embedding_count = 2
    model.has_embedding_wrapper = True
    model.has_text_encoder_lora = True
    model.has_transformer_lora = True

    var adapters = model.adapters()
    _expect("adapter count", len(adapters), 2)
    _expect_string("adapter[0]", adapters[0], String("text_encoder"))
    _expect_string("adapter[1]", adapters[1], String("transformer"))
    _expect("embedding count", model.all_embeddings_count(), 3)
    _expect("text embedding count", model.all_text_encoder_embeddings_count(), 3)

    model.to(String("cuda:0"))
    _expect_string("vae device", model.vae_device, String("cuda:0"))
    _expect_string("text encoder device", model.text_encoder_device, String("cuda:0"))
    _expect_string("transformer device", model.transformer_device, String("cuda:0"))
    _expect_string("text encoder lora device", model.text_encoder_lora_device, String("cuda:0"))
    _expect_string("transformer lora device", model.transformer_lora_device, String("cuda:0"))

    model.eval()
    _expect_bool("model eval", model.eval_called, True)
    _expect_bool("vae eval", model.vae_eval_called, True)
    _expect_bool("text encoder eval", model.text_encoder_eval_called, True)
    _expect_bool("transformer eval", model.transformer_eval_called, True)

    var pipeline = model.create_pipeline()
    _expect_bool("pipeline transformer", pipeline.has_transformer, True)
    _expect_bool("pipeline scheduler", pipeline.has_scheduler, True)
    _expect_bool("pipeline vae", pipeline.has_vae, True)
    _expect_bool("pipeline text encoder", pipeline.has_text_encoder, True)
    _expect_bool("pipeline tokenizer", pipeline.has_tokenizer, True)

    var ragged_lengths = List[Int]()
    ragged_lengths.append(9)
    ragged_lengths.append(25)
    var ragged_text = chroma_text_encode_contract(ragged_lengths^, 4096)
    _expect("ragged text batch", ragged_text.batch_size, 2)
    _expect("ragged text seq", ragged_text.output_seq_length, 32)
    _expect("ragged text ids cols", ragged_text.text_ids_cols, 3)
    _expect_bool("ragged unmasks extra token", ragged_text.bool_attention_unmasks_one_extra_token, True)
    _expect_bool("ragged padded", ragged_text.pads_to_16_because_lengths_differ, True)
    _expect_bool("ragged all true", ragged_text.attention_mask_all_true, False)

    var single_lengths = List[Int]()
    single_lengths.append(31)
    var single_text = chroma_text_encode_contract(single_lengths^, 4096)
    _expect("single text seq", single_text.output_seq_length, 32)
    _expect_bool("single all true", single_text.attention_mask_all_true, True)

    var image_ids_shape = chroma_prepare_latent_image_ids_shape(128, 144)
    _expect("image ids rows", image_ids_shape[0], 4608)
    _expect("image ids cols", image_ids_shape[1], 3)

    var packed = chroma_pack_latents_shape(ChromaLatentShape(1, 16, 128, 144))
    _expect("packed batch", packed[0], 1)
    _expect("packed seq", packed[1], 4608)
    _expect("packed channels", packed[2], 64)

    var unpacked = chroma_unpack_latents_shape(1, 64, 128, 144)
    _expect("unpacked batch", unpacked[0], 1)
    _expect("unpacked channels", unpacked[1], 16)
    _expect("unpacked height", unpacked[2], 128)
    _expect("unpacked width", unpacked[3], 144)

    var base = BaseChromaSetup()
    var predict = base.predict_contract()
    var opt_lora = base.optimization_contract(TM_LORA, True, True)
    var opt_ft = base.optimization_contract(TM_FINE_TUNE, True, True)
    var filters = base.layer_preset_filters(CHROMA_LAYER_PRESET_ATTN_MLP)
    var device_plan = base.train_device_plan(True, True, True, False)
    var text_cache = base.prepare_text_caching_plan(False)

    _expect("predict outputs", len(predict.output_fields), 4)
    _expect("attn-mlp filters", len(filters), 2)
    _expect("checkpoint parts", len(opt_lora.checkpoint_parts), 2)
    _expect("checkpoint helpers", len(opt_lora.checkpoint_helpers), 2)
    _expect("quantized parts", len(opt_lora.quantized_parts), 3)
    _expect("lora opt dtype parts", len(opt_lora.autocast_weight_dtype_parts), 5)
    _expect("ft opt dtype parts", len(opt_ft.autocast_weight_dtype_parts), 4)
    _expect_bool("text encoder train device", device_plan.text_encoder_on_train_device, True)
    _expect_bool("vae train device with caching", device_plan.vae_on_train_device, False)
    _expect_bool("transformer train mode", device_plan.transformer_train_mode, False)
    _expect_bool("text cache moves model", text_cache.move_model_to_temp_device, True)
    _expect_bool("text cache moves text encoder", text_cache.move_text_encoder_to_train_device, True)

    var ft = ChromaFineTuneSetup()
    var ft_params = ft.create_parameters(True, True, True)
    var ft_params_no_text = ft.create_parameters(True, True, False)
    var ft_plan = ft.setup_model_plan(True)
    var req = ft.requires_grad_plan()
    _expect("ft model type", ft.registration.model_type, MODEL_TYPE_CHROMA_1)
    _expect("ft params", len(ft_params), 3)
    _expect("ft params no text", len(ft_params_no_text), 2)
    _expect_bool("ft move input embeddings dtype", ft_plan.moves_input_embeddings_to_embedding_weight_dtype, True)
    _expect_bool("ft module filter", ft_plan.uses_module_filter_for_transformer, True)
    _expect_bool("ft requires grad embeddings", req.setup_embeddings_requires_grad, True)
    _expect_bool("ft freezes vae", ft.freezes_vae(), True)
    _expect_bool("ft normalizes embeddings", ft.normalizes_embeddings_after_optimizer_step(), True)
    _expect_bool("ft reapplies requires grad", ft.after_optimizer_step_reapplies_requires_grad(), True)

    var lora = ChromaLoRASetup()
    var lora_params = lora.create_parameters(True, True, True)
    var lora_params_no_text = lora.create_parameters(True, True, False)
    var lora_creation = lora.creation_plan(True, False, True, True, True)
    var lora_creation_no_text = lora.creation_plan(False, True, True, False, False)
    var lora_req = lora.requires_grad_plan()
    var lora_parts = lora.trainable_model_part_names()
    var lora_base_parts = lora.base_model_part_names()
    _expect("lora model type", lora.registration.model_type, MODEL_TYPE_CHROMA_1)
    _expect("lora training method", lora.registration.training_method, TM_LORA)
    _expect("lora params", len(lora_params), 3)
    _expect("lora params no text", len(lora_params_no_text), 2)
    _expect_bool("lora create text encoder from state dict", lora_creation.create_text_encoder_lora, True)
    _expect_bool("lora create transformer", lora_creation.create_transformer_lora, True)
    _expect("lora prefixes", len(lora_creation.wrapper_prefixes), 2)
    _expect_string("lora prefix te", lora_creation.wrapper_prefixes[0], String("lora_te"))
    _expect_string("lora prefix transformer", lora_creation.wrapper_prefixes[1], String("lora_transformer"))
    _expect_bool("lora clears pending state", lora_creation.clears_pending_state_dict_after_load, True)
    _expect_bool("lora moves dtype", lora_creation.moves_lora_to_config_weight_dtype, True)
    _expect_bool("lora hooks module", lora_creation.hooks_lora_to_module, True)
    _expect_bool("lora moves embeddings dtype", lora_creation.moves_input_embeddings_to_embedding_weight_dtype, True)
    _expect_bool("lora no text encoder", lora_creation_no_text.create_text_encoder_lora, False)
    _expect_bool("lora freezes text encoder", lora_req.freezes_text_encoder_base, True)
    _expect_bool("lora freezes transformer", lora_req.freezes_transformer_base, True)
    _expect_bool("lora freezes vae", lora.freezes_vae(), True)
    _expect("lora trainable parts", len(lora_parts), 2)
    _expect("lora base parts", len(lora_base_parts), 2)
    _expect_bool("lora normalizes wrapper", lora.normalizes_embedding_wrapper_after_optimizer_step(), True)
    _expect_bool("lora reapplies requires grad", lora.after_optimizer_step_reapplies_requires_grad(), True)

    _expect("packed tokens 128x144", chroma_packed_latent_token_count(128, 144), 4608)
    _expect("packed channels", chroma_packed_latent_channels(), 64)
    _expect("deterministic t", chroma_deterministic_timestep_index(), 499)

    print("chroma model type =", MODEL_TYPE_CHROMA_1)
    print("predict outputs =", len(predict.output_fields))
    print("dtype caveats =", len(predict.dtype_boundary_caveats))
    print("lora opt parts =", len(opt_lora.autocast_weight_dtype_parts))
    print("ft params =", len(ft_params))
    print("lora params =", len(lora_params), " create te =", lora_creation.create_text_encoder_lora)
    print("text seq ragged =", ragged_text.output_seq_length)
    print("packed =", packed[1], "x", packed[2])
    print("model_t(t=500) =", chroma_model_t_from_timestep(500), " sigma(t=499) =", chroma_sigma_from_timestep(499))
    print("CHROMA MODEL/SETUP CONTRACT OK")
