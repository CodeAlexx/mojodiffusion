# BaseChromaSetup.mojo - build-only Chroma setup contract.
#
# Source of truth: /home/alex/Serenity/modules/modelSetup/BaseChromaSetup.py
# Related Serenity model helpers:
#   /home/alex/Serenity/modules/model/ChromaModel.py
#
# This records the Serenity Chroma setup/predict surface needed by later
# parity gates. It intentionally does not execute T5, ChromaTransformer2DModel,
# VAE, MGDS, or optimizer code. Tensor dtype casts in the Python reference are
# represented as contracts; persistent checkpoint/model tensor storage must
# keep the source dtype.

from serenity_trainer.util.enum.TrainingMethod import TM_LORA


comptime CHROMA_NUM_TRAIN_TIMESTEPS = 1000
comptime CHROMA_PROMPT_MASK_PAD_MULTIPLE = 16
comptime CHROMA_LATENT_PATCH_FACTOR = 2

comptime CHROMA_LAYER_PRESET_ATTN_MLP = "attn-mlp"
comptime CHROMA_LAYER_PRESET_ATTN_ONLY = "attn-only"
comptime CHROMA_LAYER_PRESET_BLOCKS = "blocks"
comptime CHROMA_LAYER_PRESET_FULL = "full"

comptime CHROMA_LOSS_TYPE_TARGET = "target"
comptime CHROMA_PREDICT_KEY_LOSS_TYPE = "loss_type"
comptime CHROMA_PREDICT_KEY_TIMESTEP = "timestep"
comptime CHROMA_PREDICT_KEY_PREDICTED = "predicted"
comptime CHROMA_PREDICT_KEY_TARGET = "target"

comptime CHROMA_PART_TRANSFORMER = "transformer"
comptime CHROMA_PART_TEXT_ENCODER = "text_encoder"
comptime CHROMA_PART_VAE = "vae"
comptime CHROMA_PART_LORA = "lora"
comptime CHROMA_PART_EMBEDDING = "embedding"
comptime CHROMA_PART_EMBEDDINGS = "embeddings"


def chroma_layer_preset_filters(preset: String) raises -> List[String]:
    """Serenity BaseChromaSetup.LAYER_PRESETS."""
    var filters = List[String]()
    if preset == CHROMA_LAYER_PRESET_ATTN_MLP:
        filters.append("attn")
        filters.append("ff.net")
    elif preset == CHROMA_LAYER_PRESET_ATTN_ONLY:
        filters.append("attn")
    elif preset == CHROMA_LAYER_PRESET_BLOCKS:
        filters.append("transformer_block")
    elif preset == CHROMA_LAYER_PRESET_FULL:
        pass
    else:
        raise Error(String("unknown Chroma layer preset: ") + preset)
    return filters^


def chroma_predict_required_batch_fields() -> List[String]:
    """Fields read unconditionally by Serenity BaseChromaSetup.predict."""
    var fields = List[String]()
    fields.append("latent_image")
    return fields^


def chroma_predict_conditioning_batch_fields() -> List[String]:
    """Optional/cache prompt fields passed through ChromaModel.encode_text."""
    var fields = List[String]()
    fields.append("tokens")
    fields.append("tokens_mask")
    fields.append("text_encoder_hidden_state")
    return fields^


def chroma_predict_output_fields() -> List[String]:
    """Fields written in Serenity model_output_data."""
    var fields = List[String]()
    fields.append(CHROMA_PREDICT_KEY_LOSS_TYPE)
    fields.append(CHROMA_PREDICT_KEY_TIMESTEP)
    fields.append(CHROMA_PREDICT_KEY_PREDICTED)
    fields.append(CHROMA_PREDICT_KEY_TARGET)
    return fields^


def chroma_setup_optimization_checkpoint_parts(
    has_text_encoder: Bool = True,
) -> List[String]:
    var parts = List[String]()
    parts.append(CHROMA_PART_TRANSFORMER)
    if has_text_encoder:
        parts.append(CHROMA_PART_TEXT_ENCODER)
    return parts^


def chroma_setup_optimization_checkpoint_helpers(
    has_text_encoder: Bool = True,
) -> List[String]:
    var helpers = List[String]()
    helpers.append("enable_checkpointing_for_chroma_transformer")
    if has_text_encoder:
        helpers.append("enable_checkpointing_for_t5_encoder_layers")
    return helpers^


def chroma_setup_optimization_quantized_parts(
    has_text_encoder: Bool = True,
) -> List[String]:
    var parts = List[String]()
    if has_text_encoder:
        parts.append(CHROMA_PART_TEXT_ENCODER)
    parts.append(CHROMA_PART_VAE)
    parts.append(CHROMA_PART_TRANSFORMER)
    return parts^


def chroma_autocast_weight_dtype_parts(
    training_method: Int, train_any_embedding: Bool
) -> List[String]:
    var parts = List[String]()
    parts.append(CHROMA_PART_TRANSFORMER)
    parts.append(CHROMA_PART_TEXT_ENCODER)
    parts.append(CHROMA_PART_VAE)
    if training_method == TM_LORA:
        parts.append(CHROMA_PART_LORA)
    if train_any_embedding:
        parts.append(CHROMA_PART_EMBEDDING)
    return parts^


def chroma_text_encoder_autocast_weight_dtype_parts(
    training_method: Int, train_any_embedding: Bool
) -> List[String]:
    var parts = List[String]()
    parts.append(CHROMA_PART_TEXT_ENCODER)
    if training_method == TM_LORA:
        parts.append(CHROMA_PART_LORA)
    if train_any_embedding:
        parts.append(CHROMA_PART_EMBEDDING)
    return parts^


def chroma_scaled_latent_expression() -> String:
    return "(latent_image - vae.config['shift_factor']) * vae.config['scaling_factor']"


def chroma_noisy_latent_expression() -> String:
    return "latent_noise * sigma + scaled_latent_image * (1 - sigma)"


def chroma_flow_target_expression() -> String:
    return "latent_noise - scaled_latent_image"


def chroma_pack_latents_expression() -> String:
    return "view(B,C,H/2,2,W/2,2) -> permute(0,2,4,1,3,5) -> reshape(B,(H/2)*(W/2),C*4)"


def chroma_unpack_latents_expression() -> String:
    return "view(B,H/2,W/2,C/4,2,2) -> permute(0,3,1,4,2,5) -> reshape(B,C/4,H,W)"


def chroma_prepare_latent_image_ids_expression() -> String:
    return "zeros(H/2,W/2,3); channel 1 = row id; channel 2 = column id; reshape((H/2)*(W/2),3)"


def chroma_text_ids_expression() -> String:
    return "zeros((text_encoder_output.shape[1], 3), device=train_device)"


def chroma_transformer_timestep_expression() -> String:
    return "timestep / 1000"


def chroma_transformer_hidden_states_expression() -> String:
    return "pack_latents(scaled_noisy_latent_image).to(dtype=model.train_dtype.torch_dtype())"


def chroma_transformer_encoder_hidden_states_expression() -> String:
    return "text_encoder_output.to(dtype=model.train_dtype.torch_dtype())"


def chroma_attention_mask_expression() -> String:
    return "cat([text_attention_mask, image_attention_mask], dim=1) if not torch.all(text_attention_mask) else None"


def chroma_predicted_expression() -> String:
    return "unpack_latents(model.transformer(..., return_dict=True).sample, latent_input.shape[2], latent_input.shape[3])"


def chroma_deterministic_timestep_index(
    num_train_timesteps: Int = CHROMA_NUM_TRAIN_TIMESTEPS,
) -> Int:
    return Int(Float64(num_train_timesteps) * Float64(0.5)) - 1


def chroma_sigma_from_timestep(
    t: Int, num_timesteps: Int = CHROMA_NUM_TRAIN_TIMESTEPS
) -> Float32:
    # ModelSetupFlowMatchingMixin._add_noise_discrete:
    # sigma[t] = arange(1, N + 1)[t] / N.
    return Float32(t + 1) / Float32(num_timesteps)


def chroma_model_t_from_timestep(
    t: Int, num_timesteps: Int = CHROMA_NUM_TRAIN_TIMESTEPS
) -> Float32:
    return Float32(t) / Float32(num_timesteps)


def chroma_packed_latent_token_count(latent_h: Int, latent_w: Int) -> Int:
    return (latent_h // CHROMA_LATENT_PATCH_FACTOR) * (
        latent_w // CHROMA_LATENT_PATCH_FACTOR
    )


def chroma_packed_latent_channels(latent_channels: Int = 16) -> Int:
    return latent_channels * CHROMA_LATENT_PATCH_FACTOR * CHROMA_LATENT_PATCH_FACTOR


def chroma_dtype_boundary_caveats() -> List[String]:
    var caveats = List[String]()
    caveats.append("Build-only surface: no T5, ChromaTransformer2DModel, VAE, tokenizer, or MGDS runtime execution is implemented here")
    caveats.append("Serenity casts transformer hidden_states, encoder_hidden_states, txt_ids, and img_ids to model.train_dtype at the transformer call")
    caveats.append("Serenity Chroma sampler creates initial inference latents as torch.float32; that does not permit persistent checkpoint/model tensors to upcast to F32")
    caveats.append("Sigma, timestep, VAE shift/scale, reductions, and scheduler scalar helpers may use F32 internally and must return the storage/runtime dtype at tensor boundaries")
    caveats.append("Serenity disables fp16 autocast for the T5 text encoder path through disable_fp16_autocast_context")
    caveats.append("RNG parity requires Serenity/PyTorch oracle noise or a proven matching generator")
    return caveats^


def chroma_unsupported_runtime_paths() -> List[String]:
    var paths = List[String]()
    paths.append("predict runtime tensor path is not implemented in this setup surface")
    paths.append("MGDS data pipeline execution is not implemented in this setup surface")
    paths.append("ChromaTransformer2DModel forward/backward is not implemented here")
    paths.append("LoRA/fine-tune optimizer deltas are represented as setup plans only")
    paths.append("Serenity execution, parity, speed, and numeric claims are owned by later runtime work")
    return paths^


struct ChromaPredictContract(Movable):
    var required_batch_fields: List[String]
    var conditioning_batch_fields: List[String]
    var output_fields: List[String]
    var loss_type: String
    var scaled_latent_expression: String
    var noisy_latent_expression: String
    var target_expression: String
    var pack_latents_expression: String
    var unpack_latents_expression: String
    var prepare_latent_image_ids_expression: String
    var text_ids_expression: String
    var transformer_timestep_expression: String
    var transformer_hidden_states_expression: String
    var transformer_encoder_hidden_states_expression: String
    var attention_mask_expression: String
    var predicted_expression: String
    var dtype_boundary_caveats: List[String]

    def __init__(out self):
        self.required_batch_fields = chroma_predict_required_batch_fields()
        self.conditioning_batch_fields = chroma_predict_conditioning_batch_fields()
        self.output_fields = chroma_predict_output_fields()
        self.loss_type = CHROMA_LOSS_TYPE_TARGET
        self.scaled_latent_expression = chroma_scaled_latent_expression()
        self.noisy_latent_expression = chroma_noisy_latent_expression()
        self.target_expression = chroma_flow_target_expression()
        self.pack_latents_expression = chroma_pack_latents_expression()
        self.unpack_latents_expression = chroma_unpack_latents_expression()
        self.prepare_latent_image_ids_expression = (
            chroma_prepare_latent_image_ids_expression()
        )
        self.text_ids_expression = chroma_text_ids_expression()
        self.transformer_timestep_expression = chroma_transformer_timestep_expression()
        self.transformer_hidden_states_expression = (
            chroma_transformer_hidden_states_expression()
        )
        self.transformer_encoder_hidden_states_expression = (
            chroma_transformer_encoder_hidden_states_expression()
        )
        self.attention_mask_expression = chroma_attention_mask_expression()
        self.predicted_expression = chroma_predicted_expression()
        self.dtype_boundary_caveats = chroma_dtype_boundary_caveats()


struct ChromaOptimizationContract(Movable):
    var checkpoint_parts: List[String]
    var checkpoint_helpers: List[String]
    var quantized_parts: List[String]
    var autocast_weight_dtype_parts: List[String]
    var text_encoder_autocast_weight_dtype_parts: List[String]
    var disables_fp16_text_encoder_autocast: Bool

    def __init__(
        out self,
        training_method: Int,
        train_any_embedding: Bool = False,
        has_text_encoder: Bool = True,
    ):
        self.checkpoint_parts = chroma_setup_optimization_checkpoint_parts(
            has_text_encoder
        )
        self.checkpoint_helpers = chroma_setup_optimization_checkpoint_helpers(
            has_text_encoder
        )
        self.quantized_parts = chroma_setup_optimization_quantized_parts(
            has_text_encoder
        )
        self.autocast_weight_dtype_parts = chroma_autocast_weight_dtype_parts(
            training_method, train_any_embedding
        )
        self.text_encoder_autocast_weight_dtype_parts = (
            chroma_text_encoder_autocast_weight_dtype_parts(
                training_method, train_any_embedding
            )
        )
        self.disables_fp16_text_encoder_autocast = True


struct ChromaTrainDevicePlan(Copyable, Movable, ImplicitlyCopyable):
    var text_encoder_on_train_device: Bool
    var vae_on_train_device: Bool
    var transformer_on_train_device: Bool
    var text_encoder_train_mode: Bool
    var vae_train_mode: Bool
    var transformer_train_mode: Bool

    def __init__(
        out self,
        latent_caching: Bool,
        train_text_encoder_or_embedding: Bool,
        text_encoder_train: Bool,
        transformer_train: Bool,
    ):
        self.text_encoder_on_train_device = (
            train_text_encoder_or_embedding or not latent_caching
        )
        self.vae_on_train_device = not latent_caching
        self.transformer_on_train_device = True
        self.text_encoder_train_mode = text_encoder_train
        self.vae_train_mode = False
        self.transformer_train_mode = transformer_train


struct ChromaTextCachingPlan(Copyable, Movable, ImplicitlyCopyable):
    var move_model_to_temp_device: Bool
    var move_text_encoder_to_train_device: Bool
    var set_eval_mode: Bool
    var run_torch_gc: Bool

    def __init__(out self, train_text_encoder_or_embedding: Bool):
        self.move_model_to_temp_device = True
        self.move_text_encoder_to_train_device = not train_text_encoder_or_embedding
        self.set_eval_mode = True
        self.run_torch_gc = True


struct ChromaUnsupportedPaths(Movable):
    var paths: List[String]

    def __init__(out self):
        self.paths = chroma_unsupported_runtime_paths()


def chroma_predict_contract() -> ChromaPredictContract:
    return ChromaPredictContract()


def chroma_optimization_contract(
    training_method: Int,
    train_any_embedding: Bool = False,
    has_text_encoder: Bool = True,
) -> ChromaOptimizationContract:
    return ChromaOptimizationContract(
        training_method, train_any_embedding, has_text_encoder
    )


def chroma_train_device_plan(
    latent_caching: Bool,
    train_text_encoder_or_embedding: Bool,
    text_encoder_train: Bool,
    transformer_train: Bool,
) -> ChromaTrainDevicePlan:
    return ChromaTrainDevicePlan(
        latent_caching,
        train_text_encoder_or_embedding,
        text_encoder_train,
        transformer_train,
    )


def chroma_text_caching_plan(
    train_text_encoder_or_embedding: Bool,
) -> ChromaTextCachingPlan:
    return ChromaTextCachingPlan(train_text_encoder_or_embedding)


struct BaseChromaSetup(Movable):
    var debug_mode: Bool

    def __init__(out self, debug_mode: Bool = False):
        self.debug_mode = debug_mode

    def layer_preset_filters(self, preset: String) raises -> List[String]:
        return chroma_layer_preset_filters(preset)

    def predict_contract(self) -> ChromaPredictContract:
        return chroma_predict_contract()

    def optimization_contract(
        self,
        training_method: Int,
        train_any_embedding: Bool = False,
        has_text_encoder: Bool = True,
    ) -> ChromaOptimizationContract:
        return chroma_optimization_contract(
            training_method, train_any_embedding, has_text_encoder
        )

    def train_device_plan(
        self,
        latent_caching: Bool,
        train_text_encoder_or_embedding: Bool,
        text_encoder_train: Bool,
        transformer_train: Bool,
    ) -> ChromaTrainDevicePlan:
        return chroma_train_device_plan(
            latent_caching,
            train_text_encoder_or_embedding,
            text_encoder_train,
            transformer_train,
        )

    def calculate_loss_consumes_sigmas(self) -> Bool:
        # calculate_loss delegates to _flow_matching_losses(..., sigmas=scheduler.sigmas).
        return True

    def calculate_loss_reduction(self) -> String:
        return "mean"

    def prepare_text_caching_plan(
        self, train_text_encoder_or_embedding: Bool
    ) -> ChromaTextCachingPlan:
        return chroma_text_caching_plan(train_text_encoder_or_embedding)

    def runtime_predict_implemented(self) -> Bool:
        return False

    def unsupported_runtime_paths(self) -> ChromaUnsupportedPaths:
        return ChromaUnsupportedPaths()
