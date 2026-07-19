# 1:1 surface port of Serenity modules/modelSampler/ChromaSampler.py
#
# Build-only sampler support. The actual Chroma text encoder, transformer,
# scheduler tensor step, VAE decode, image postprocess, save path, and generated
# image validation are outside this worker's scope, so sample() returns a plan
# and generate() is explicitly unsupported. The helpers mirror deterministic
# Serenity Chroma sampler/model behavior:
#
#   * ChromaSampler.sample: 64px resolution quantization before __sample_base.
#   * ChromaSampler.__sample_base: VAE scale 8, 16 latent channels, F32 initial
#     latent noise, prompt+negative prompt batch size 2, CFG combine, packed
#     latent/image id/text id/attention mask contracts, FlowMatch scheduler
#     copied from the model, transformer timestep /1000, VAE decode formula, and
#     PIL image output.
#   * ChromaModel.encode_text: T5 bool mask unmask-one-token behavior, optional
#     padding to a multiple of 16 only when sequence lengths differ, and pruned
#     text encoder output/mask length.
#   * ChromaModel.prepare_latent_image_ids / pack_latents / unpack_latents.

from serenity_trainer.util.enum.ModelType import (
    MODEL_TYPE_CHROMA_1,
    model_type_is_chroma,
    model_type_str,
)


comptime CHROMA_SAMPLE_FILE_TYPE_IMAGE = 0
comptime CHROMA_SAMPLE_RESOLUTION_QUANTIZATION = 64
comptime CHROMA_SAMPLE_VAE_SCALE_FACTOR = 8
comptime CHROMA_SAMPLE_LATENT_CHANNELS = 16
comptime CHROMA_SAMPLE_LATENT_PACK_SIZE = 2
comptime CHROMA_SAMPLE_PACKED_CHANNELS = CHROMA_SAMPLE_LATENT_CHANNELS * 4
comptime CHROMA_SAMPLE_CFG_BATCH_SIZE = 2
comptime CHROMA_SAMPLE_TEXT_ID_CHANNELS = 3
comptime CHROMA_SAMPLE_IMAGE_ID_CHANNELS = 3
comptime CHROMA_SAMPLE_NUM_TRAIN_TIMESTEPS = 1000
comptime CHROMA_SAMPLE_DEFAULT_WIDTH = 1024
comptime CHROMA_SAMPLE_DEFAULT_HEIGHT = 1024
comptime CHROMA_SAMPLE_DEFAULT_DIFFUSION_STEPS = 30
comptime CHROMA_SAMPLE_DEFAULT_CFG_SCALE = Float32(3.5)


@fieldwise_init
struct ChromaSamplerSchedulerConfig(Copyable, Movable, ImplicitlyCopyable):
    """FlowMatchEulerDiscreteScheduler fields used by Chroma sampler helpers.

    Serenity copies model.noise_scheduler and calls set_timesteps(n). The
    loaded model scheduler provides these values, so helper callers pass the
    concrete config instead of assuming a global default.
    """

    var num_train_timesteps: Int
    var shift: Float32
    var use_dynamic_shifting: Bool
    var invert_sigmas: Bool
    var stochastic_sampling: Bool


struct ChromaSamplerLatentContract(Copyable, Movable, ImplicitlyCopyable):
    var latent_batch_size: Int
    var model_input_batch_size: Int
    var image_height: Int
    var image_width: Int
    var latent_channels: Int
    var latent_height: Int
    var latent_width: Int
    var packed_seq_len: Int
    var packed_channels: Int
    var image_ids_rows: Int
    var image_ids_cols: Int

    def __init__(
        out self,
        image_height: Int,
        image_width: Int,
        latent_batch_size: Int = 1,
    ):
        self.latent_batch_size = latent_batch_size
        self.model_input_batch_size = latent_batch_size * CHROMA_SAMPLE_CFG_BATCH_SIZE
        self.image_height = image_height
        self.image_width = image_width
        self.latent_channels = CHROMA_SAMPLE_LATENT_CHANNELS
        self.latent_height = image_height // CHROMA_SAMPLE_VAE_SCALE_FACTOR
        self.latent_width = image_width // CHROMA_SAMPLE_VAE_SCALE_FACTOR
        self.packed_seq_len = (
            (self.latent_height // CHROMA_SAMPLE_LATENT_PACK_SIZE)
            * (self.latent_width // CHROMA_SAMPLE_LATENT_PACK_SIZE)
        )
        self.packed_channels = CHROMA_SAMPLE_PACKED_CHANNELS
        self.image_ids_rows = self.packed_seq_len
        self.image_ids_cols = CHROMA_SAMPLE_IMAGE_ID_CHANNELS


struct ChromaTextMaskContract(Copyable, Movable, ImplicitlyCopyable):
    var batch_size: Int
    var positive_input_tokens: Int
    var negative_input_tokens: Int
    var positive_bool_tokens: Int
    var negative_bool_tokens: Int
    var max_seq_length: Int
    var pads_to_16_because_lengths_differ: Bool
    var text_ids_rows: Int
    var text_ids_cols: Int

    def __init__(
        out self,
        positive_input_tokens: Int,
        negative_input_tokens: Int,
    ):
        self.batch_size = CHROMA_SAMPLE_CFG_BATCH_SIZE
        self.positive_input_tokens = positive_input_tokens
        self.negative_input_tokens = negative_input_tokens
        # ChromaModel.encode_text uses (mask_indices <= seq_lengths), explicitly
        # unmasking one token beyond tokens_mask.sum(dim=1).
        self.positive_bool_tokens = positive_input_tokens + 1
        self.negative_bool_tokens = negative_input_tokens + 1
        var max_len = self.positive_bool_tokens
        if self.negative_bool_tokens > max_len:
            max_len = self.negative_bool_tokens
        self.pads_to_16_because_lengths_differ = (
            (max_len % 16) > 0
            and (
                self.positive_bool_tokens != max_len
                or self.negative_bool_tokens != max_len
            )
        )
        if self.pads_to_16_because_lengths_differ:
            max_len += 16 - (max_len % 16)
        self.max_seq_length = max_len
        self.text_ids_rows = max_len
        self.text_ids_cols = CHROMA_SAMPLE_TEXT_ID_CHANNELS


struct ChromaAttentionMaskContract(Copyable, Movable, ImplicitlyCopyable):
    var batch_size: Int
    var text_seq_len: Int
    var image_seq_len: Int
    var attention_mask_rows: Int
    var attention_mask_cols: Int
    var image_attention_mask_all_true: Bool
    var sampler_always_passes_attention_mask: Bool
    var training_passes_attention_mask_when_text_not_all_true: Bool
    var training_omits_attention_mask_when_text_all_true: Bool

    def __init__(out self, text_seq_len: Int, image_seq_len: Int):
        self.batch_size = CHROMA_SAMPLE_CFG_BATCH_SIZE
        self.text_seq_len = text_seq_len
        self.image_seq_len = image_seq_len
        self.attention_mask_rows = CHROMA_SAMPLE_CFG_BATCH_SIZE
        self.attention_mask_cols = text_seq_len + image_seq_len
        self.image_attention_mask_all_true = True
        self.sampler_always_passes_attention_mask = True
        self.training_passes_attention_mask_when_text_not_all_true = True
        self.training_omits_attention_mask_when_text_all_true = True


@fieldwise_init
struct ChromaPackedLatentIndex(Copyable, Movable, ImplicitlyCopyable):
    var sequence_index: Int
    var packed_channel: Int


@fieldwise_init
struct ChromaUnpackedLatentIndex(Copyable, Movable, ImplicitlyCopyable):
    var channel: Int
    var latent_y: Int
    var latent_x: Int


struct ChromaSamplerSchedule(Movable):
    var sigmas: List[Float32]
    var timesteps: List[Float32]

    def __init__(
        out self,
        var sigmas: List[Float32],
        var timesteps: List[Float32],
    ):
        self.sigmas = sigmas^
        self.timesteps = timesteps^


struct ChromaSampleConfig(Movable):
    var prompt: String
    var negative_prompt: String
    var height: Int
    var width: Int
    var seed: Int
    var random_seed: Bool
    var diffusion_steps: Int
    var cfg_scale: Float32
    var noise_scheduler: Int
    var text_encoder_1_layer_skip: Int

    def __init__(
        out self,
        var prompt: String,
        var negative_prompt: String,
        height: Int,
        width: Int,
        seed: Int,
        random_seed: Bool,
        diffusion_steps: Int,
        cfg_scale: Float32,
        noise_scheduler: Int,
        text_encoder_1_layer_skip: Int = 0,
    ):
        self.prompt = prompt^
        self.negative_prompt = negative_prompt^
        self.height = height
        self.width = width
        self.seed = seed
        self.random_seed = random_seed
        self.diffusion_steps = diffusion_steps
        self.cfg_scale = cfg_scale
        self.noise_scheduler = noise_scheduler
        self.text_encoder_1_layer_skip = text_encoder_1_layer_skip


struct ChromaSamplePlan(Movable):
    var file_type: Int
    var destination: String
    var prompt: String
    var negative_prompt: String
    var height: Int
    var width: Int
    var latent_h: Int
    var latent_w: Int
    var latent_channels: Int
    var packed_seq_len: Int
    var packed_channels: Int
    var batch_size: Int
    var seed: Int
    var random_seed: Bool
    var seed_source: String
    var diffusion_steps: Int
    var cfg_scale: Float32
    var always_uses_negative_prompt: Bool
    var scheduler_copied_from_model: Bool
    var timestep_source: String
    var transformer_timestep_formula: String
    var extra_step_kwargs_may_include_generator: Bool
    var initial_noise_dtype: String
    var initial_noise_reference_reason: String
    var transformer_input_dtype: String
    var prompt_embedding_input_dtype: String
    var text_ids_dtype: String
    var image_ids_dtype: String
    var attention_mask_formula: String
    var decode_formula: String
    var postprocess_output_type: String
    var output_file_type: String
    var text_encoder_1_layer_skip: Int

    def __init__(
        out self,
        config: ChromaSampleConfig,
        var destination: String,
        height: Int,
        width: Int,
    ):
        self.file_type = CHROMA_SAMPLE_FILE_TYPE_IMAGE
        self.destination = destination^
        self.prompt = config.prompt.copy()
        self.negative_prompt = config.negative_prompt.copy()
        self.height = height
        self.width = width
        self.latent_h = height // CHROMA_SAMPLE_VAE_SCALE_FACTOR
        self.latent_w = width // CHROMA_SAMPLE_VAE_SCALE_FACTOR
        self.latent_channels = CHROMA_SAMPLE_LATENT_CHANNELS
        self.packed_seq_len = (
            (self.latent_h // CHROMA_SAMPLE_LATENT_PACK_SIZE)
            * (self.latent_w // CHROMA_SAMPLE_LATENT_PACK_SIZE)
        )
        self.packed_channels = CHROMA_SAMPLE_PACKED_CHANNELS
        self.batch_size = CHROMA_SAMPLE_CFG_BATCH_SIZE
        self.seed = config.seed
        self.random_seed = config.random_seed
        if config.random_seed:
            self.seed_source = String("torch.Generator.seed()")
        else:
            self.seed_source = String("torch.Generator.manual_seed(seed)")
        self.diffusion_steps = config.diffusion_steps
        self.cfg_scale = config.cfg_scale
        self.always_uses_negative_prompt = True
        self.scheduler_copied_from_model = True
        self.timestep_source = String("copy.deepcopy(model.noise_scheduler).set_timesteps(diffusion_steps).timesteps")
        self.transformer_timestep_formula = String("expanded_timestep / 1000")
        self.extra_step_kwargs_may_include_generator = True
        self.initial_noise_dtype = String("F32")
        self.initial_noise_reference_reason = String("Serenity torch.randn(..., dtype=torch.float32) before transformer dtype cast")
        self.transformer_input_dtype = String("model.train_dtype.torch_dtype()")
        self.prompt_embedding_input_dtype = String("model.train_dtype.torch_dtype()")
        self.text_ids_dtype = String("model.train_dtype.torch_dtype()")
        self.image_ids_dtype = String("model.train_dtype.torch_dtype()")
        self.attention_mask_formula = String("torch.cat([text_attention_mask, torch.full((2, image_seq_len), True)], dim=1)")
        self.decode_formula = String("(latent_image / vae.config.scaling_factor) + vae.config.shift_factor")
        self.postprocess_output_type = String("pil")
        self.output_file_type = String("IMAGE")
        self.text_encoder_1_layer_skip = config.text_encoder_1_layer_skip


def chroma_quantize_resolution(resolution: Int, quantization: Int) -> Int:
    # BaseModelSampler.quantize_resolution uses Python round(), which rounds
    # exact halves to even.
    var q = resolution // quantization
    var r = resolution - q * quantization
    var twice = r * 2
    if twice > quantization:
        return (q + 1) * quantization
    if twice < quantization:
        return q * quantization
    if q % 2 == 0:
        return q * quantization
    return (q + 1) * quantization


def chroma_cfg_batch_size(latent_batch_size: Int = 1) raises -> Int:
    if latent_batch_size <= 0:
        raise Error("Chroma sampler CFG batch: latent_batch_size must be positive")
    return latent_batch_size * CHROMA_SAMPLE_CFG_BATCH_SIZE


def chroma_cfg_combine_value(
    positive: Float32, negative: Float32, cfg_scale: Float32
) -> Float32:
    return negative + cfg_scale * (positive - negative)


def chroma_has_cfg_rescale() -> Bool:
    # ChromaSampler.__sample_base applies plain classifier-free guidance only.
    return False


def chroma_euler_update_value(
    sample: Float32, model_output: Float32, sigma: Float32, sigma_next: Float32
) -> Float32:
    return sample + (sigma_next - sigma) * model_output


def chroma_transformer_timestep_value(timestep: Float32) -> Float32:
    # ChromaSampler.__sample_base and BaseChromaSetup.predict pass timestep / 1000.
    return timestep / Float32(CHROMA_SAMPLE_NUM_TRAIN_TIMESTEPS)


def chroma_decode_input_value(
    latent_image: Float32,
    vae_scaling_factor: Float32,
    vae_shift_factor: Float32,
) raises -> Float32:
    if vae_scaling_factor == Float32(0.0):
        raise Error("Chroma decode helper: VAE scaling_factor must be non-zero")
    return latent_image / vae_scaling_factor + vae_shift_factor


def chroma_scale_latent_value(
    latent_image: Float32,
    vae_shift_factor: Float32,
    vae_scaling_factor: Float32,
) -> Float32:
    # BaseChromaSetup.predict: scaled_latent_image = (latent_image - shift) * scale.
    return (latent_image - vae_shift_factor) * vae_scaling_factor


def chroma_deterministic_timestep_index(num_train_timesteps: Int) raises -> Int:
    if num_train_timesteps <= 0:
        raise Error("Chroma timestep helper: num_train_timesteps must be positive")
    return Int(Float32(num_train_timesteps) * Float32(0.5)) - 1


def chroma_shift_timestep_value(
    timestep: Float32,
    num_train_timesteps: Int,
    shift: Float32,
) raises -> Float32:
    if num_train_timesteps <= 0:
        raise Error("Chroma timestep shift: num_train_timesteps must be positive")
    if shift <= Float32(0.0):
        raise Error("Chroma timestep shift: shift must be positive")
    var n_train = Float32(num_train_timesteps)
    return n_train * shift * timestep / ((shift - Float32(1.0)) * timestep + n_train)


def chroma_flow_matching_sigma_for_timestep(
    timestep_index: Int,
    num_timesteps: Int,
) raises -> Float32:
    if num_timesteps <= 0:
        raise Error("Chroma flow sigma: num_timesteps must be positive")
    if timestep_index < 0 or timestep_index >= num_timesteps:
        raise Error("Chroma flow sigma: timestep_index out of range")
    return Float32(timestep_index + 1) / Float32(num_timesteps)


def chroma_flow_matching_one_minus_sigma(
    timestep_index: Int,
    num_timesteps: Int,
) raises -> Float32:
    return Float32(1.0) - chroma_flow_matching_sigma_for_timestep(
        timestep_index, num_timesteps
    )


def chroma_add_noise_discrete_value(
    scaled_latent_image: Float32,
    latent_noise: Float32,
    timestep_index: Int,
    num_timesteps: Int,
) raises -> Float32:
    var sigma = chroma_flow_matching_sigma_for_timestep(
        timestep_index, num_timesteps
    )
    return latent_noise * sigma + scaled_latent_image * (Float32(1.0) - sigma)


def chroma_flow_target_value(latent_noise: Float32, scaled_latent_image: Float32) -> Float32:
    return latent_noise - scaled_latent_image


def chroma_predicted_scaled_latent_value(
    scaled_noisy_latent_image: Float32,
    predicted_flow: Float32,
    sigma: Float32,
) -> Float32:
    return scaled_noisy_latent_image - predicted_flow * sigma


def chroma_flow_shift_sigma(sigma: Float32, shift: Float32) raises -> Float32:
    if shift <= Float32(0.0):
        raise Error("Chroma sampler FlowMatch shift: shift must be positive")
    return shift * sigma / (Float32(1.0) + (shift - Float32(1.0)) * sigma)


def chroma_make_flow_schedule(
    diffusion_steps: Int,
    config: ChromaSamplerSchedulerConfig,
) raises -> ChromaSamplerSchedule:
    if diffusion_steps <= 0:
        raise Error("Chroma sampler schedule: diffusion_steps must be positive")
    if config.num_train_timesteps <= 0:
        raise Error("Chroma sampler schedule: num_train_timesteps must be positive")
    if config.shift <= Float32(0.0):
        raise Error("Chroma sampler schedule: shift must be positive")
    if config.use_dynamic_shifting:
        raise Error("Chroma sampler schedule: dynamic shifting is not covered by this helper")
    if config.invert_sigmas:
        raise Error("Chroma sampler schedule: invert_sigmas is not covered by this helper")
    if config.stochastic_sampling:
        raise Error("Chroma sampler schedule: stochastic sampling is not covered by this helper")

    var n = diffusion_steps
    var n_train = Float32(config.num_train_timesteps)
    var sigma_max = Float32(1.0)
    var sigma_min_base = Float32(1.0) / n_train
    var sigma_min = chroma_flow_shift_sigma(sigma_min_base, config.shift)
    var t_start = sigma_max * n_train
    var t_end = sigma_min * n_train
    var sigmas = List[Float32]()
    var timesteps = List[Float32]()

    for i in range(n):
        var timestep: Float32
        if n == 1:
            timestep = t_start
        else:
            var frac = Float32(i) / Float32(n - 1)
            timestep = t_start + frac * (t_end - t_start)
        var sigma = chroma_flow_shift_sigma(timestep / n_train, config.shift)
        sigmas.append(sigma)
        timesteps.append(sigma * n_train)

    sigmas.append(Float32(0.0))
    return ChromaSamplerSchedule(sigmas^, timesteps^)


def chroma_latent_contract_for_image(
    image_height: Int,
    image_width: Int,
    latent_batch_size: Int = 1,
) raises -> ChromaSamplerLatentContract:
    if image_height <= 0 or image_width <= 0:
        raise Error("Chroma sampler latent contract: image dimensions must be positive")
    if latent_batch_size <= 0:
        raise Error("Chroma sampler latent contract: latent_batch_size must be positive")
    if image_height % CHROMA_SAMPLE_VAE_SCALE_FACTOR != 0:
        raise Error("Chroma sampler latent contract: height must be divisible by VAE scale")
    if image_width % CHROMA_SAMPLE_VAE_SCALE_FACTOR != 0:
        raise Error("Chroma sampler latent contract: width must be divisible by VAE scale")
    var latent_h = image_height // CHROMA_SAMPLE_VAE_SCALE_FACTOR
    var latent_w = image_width // CHROMA_SAMPLE_VAE_SCALE_FACTOR
    if latent_h % CHROMA_SAMPLE_LATENT_PACK_SIZE != 0:
        raise Error("Chroma sampler latent contract: latent height must be divisible by pack size")
    if latent_w % CHROMA_SAMPLE_LATENT_PACK_SIZE != 0:
        raise Error("Chroma sampler latent contract: latent width must be divisible by pack size")
    return ChromaSamplerLatentContract(image_height, image_width, latent_batch_size)


def chroma_quantized_latent_contract(
    image_height: Int,
    image_width: Int,
    latent_batch_size: Int = 1,
) raises -> ChromaSamplerLatentContract:
    var height = chroma_quantize_resolution(
        image_height, CHROMA_SAMPLE_RESOLUTION_QUANTIZATION
    )
    var width = chroma_quantize_resolution(
        image_width, CHROMA_SAMPLE_RESOLUTION_QUANTIZATION
    )
    return chroma_latent_contract_for_image(height, width, latent_batch_size)


def chroma_text_mask_contract(
    positive_input_tokens: Int,
    negative_input_tokens: Int,
) raises -> ChromaTextMaskContract:
    if positive_input_tokens < 0 or negative_input_tokens < 0:
        raise Error("Chroma text mask contract: token counts must be non-negative")
    return ChromaTextMaskContract(positive_input_tokens, negative_input_tokens)


def chroma_attention_mask_contract(
    text_seq_len: Int,
    image_seq_len: Int,
) raises -> ChromaAttentionMaskContract:
    if text_seq_len <= 0:
        raise Error("Chroma attention mask contract: text_seq_len must be positive")
    if image_seq_len <= 0:
        raise Error("Chroma attention mask contract: image_seq_len must be positive")
    return ChromaAttentionMaskContract(text_seq_len, image_seq_len)


def chroma_training_attention_mask_is_passed(text_attention_all_true: Bool) -> Bool:
    # BaseChromaSetup.predict passes None when every text token is unmasked.
    return not text_attention_all_true


def chroma_image_id_last_row_value(latent_size: Int) raises -> Int:
    if latent_size <= 0:
        raise Error("Chroma image ids: latent dimension must be positive")
    if latent_size % CHROMA_SAMPLE_LATENT_PACK_SIZE != 0:
        raise Error("Chroma image ids: latent dimension must be divisible by pack size")
    return latent_size // CHROMA_SAMPLE_LATENT_PACK_SIZE - 1


def chroma_image_id_row_from_tile(
    tile_y: Int,
    tile_x: Int,
    latent_width: Int,
) raises -> Int:
    if tile_y < 0 or tile_x < 0:
        raise Error("Chroma image ids: tile coordinates must be non-negative")
    if latent_width <= 0:
        raise Error("Chroma image ids: latent_width must be positive")
    if latent_width % CHROMA_SAMPLE_LATENT_PACK_SIZE != 0:
        raise Error("Chroma image ids: latent_width must be divisible by pack size")
    var tile_width = latent_width // CHROMA_SAMPLE_LATENT_PACK_SIZE
    if tile_x >= tile_width:
        raise Error("Chroma image ids: tile_x out of range")
    return tile_y * tile_width + tile_x


def chroma_image_id_tile_y_from_row(row: Int, latent_width: Int) raises -> Int:
    if row < 0:
        raise Error("Chroma image ids: row must be non-negative")
    if latent_width <= 0 or latent_width % CHROMA_SAMPLE_LATENT_PACK_SIZE != 0:
        raise Error("Chroma image ids: latent_width must be positive and divisible by pack size")
    return row // (latent_width // CHROMA_SAMPLE_LATENT_PACK_SIZE)


def chroma_image_id_tile_x_from_row(row: Int, latent_width: Int) raises -> Int:
    if row < 0:
        raise Error("Chroma image ids: row must be non-negative")
    if latent_width <= 0 or latent_width % CHROMA_SAMPLE_LATENT_PACK_SIZE != 0:
        raise Error("Chroma image ids: latent_width must be positive and divisible by pack size")
    var tile_width = latent_width // CHROMA_SAMPLE_LATENT_PACK_SIZE
    return row - (row // tile_width) * tile_width


def chroma_pack_latent_index(
    channel: Int,
    latent_y: Int,
    latent_x: Int,
    latent_height: Int,
    latent_width: Int,
) raises -> ChromaPackedLatentIndex:
    if channel < 0 or channel >= CHROMA_SAMPLE_LATENT_CHANNELS:
        raise Error("Chroma pack latents: channel out of range")
    if latent_y < 0 or latent_y >= latent_height:
        raise Error("Chroma pack latents: latent_y out of range")
    if latent_x < 0 or latent_x >= latent_width:
        raise Error("Chroma pack latents: latent_x out of range")
    if latent_height % CHROMA_SAMPLE_LATENT_PACK_SIZE != 0:
        raise Error("Chroma pack latents: latent_height must be divisible by pack size")
    if latent_width % CHROMA_SAMPLE_LATENT_PACK_SIZE != 0:
        raise Error("Chroma pack latents: latent_width must be divisible by pack size")

    var tile_width = latent_width // CHROMA_SAMPLE_LATENT_PACK_SIZE
    var tile_y = latent_y // CHROMA_SAMPLE_LATENT_PACK_SIZE
    var tile_x = latent_x // CHROMA_SAMPLE_LATENT_PACK_SIZE
    var offset_y = latent_y - tile_y * CHROMA_SAMPLE_LATENT_PACK_SIZE
    var offset_x = latent_x - tile_x * CHROMA_SAMPLE_LATENT_PACK_SIZE
    return ChromaPackedLatentIndex(
        tile_y * tile_width + tile_x,
        channel * 4 + offset_y * 2 + offset_x,
    )


def chroma_unpack_latent_index(
    sequence_index: Int,
    packed_channel: Int,
    latent_height: Int,
    latent_width: Int,
) raises -> ChromaUnpackedLatentIndex:
    if sequence_index < 0:
        raise Error("Chroma unpack latents: sequence_index must be non-negative")
    if packed_channel < 0 or packed_channel >= CHROMA_SAMPLE_PACKED_CHANNELS:
        raise Error("Chroma unpack latents: packed_channel out of range")
    if latent_height <= 0 or latent_width <= 0:
        raise Error("Chroma unpack latents: latent dimensions must be positive")
    if latent_height % CHROMA_SAMPLE_LATENT_PACK_SIZE != 0:
        raise Error("Chroma unpack latents: latent_height must be divisible by pack size")
    if latent_width % CHROMA_SAMPLE_LATENT_PACK_SIZE != 0:
        raise Error("Chroma unpack latents: latent_width must be divisible by pack size")

    var tile_height = latent_height // CHROMA_SAMPLE_LATENT_PACK_SIZE
    var tile_width = latent_width // CHROMA_SAMPLE_LATENT_PACK_SIZE
    var seq_len = tile_height * tile_width
    if sequence_index >= seq_len:
        raise Error("Chroma unpack latents: sequence_index out of range")

    var tile_y = sequence_index // tile_width
    var tile_x = sequence_index - tile_y * tile_width
    var channel = packed_channel // 4
    var offset = packed_channel - channel * 4
    var offset_y = offset // 2
    var offset_x = offset - offset_y * 2
    return ChromaUnpackedLatentIndex(
        channel,
        tile_y * CHROMA_SAMPLE_LATENT_PACK_SIZE + offset_y,
        tile_x * CHROMA_SAMPLE_LATENT_PACK_SIZE + offset_x,
    )


def chroma_sample_plan(
    config: ChromaSampleConfig,
    destination: String,
) raises -> ChromaSamplePlan:
    if config.diffusion_steps <= 0:
        raise Error("ChromaSampler.sample: diffusion_steps must be positive")
    var h = chroma_quantize_resolution(
        config.height, CHROMA_SAMPLE_RESOLUTION_QUANTIZATION
    )
    var w = chroma_quantize_resolution(
        config.width, CHROMA_SAMPLE_RESOLUTION_QUANTIZATION
    )
    _ = chroma_latent_contract_for_image(h, w)
    return ChromaSamplePlan(config, destination.copy(), h, w)


struct ChromaSampler(Movable):
    var model_type: Int

    def __init__(out self, model_type: Int):
        self.model_type = model_type

    def sample(self, sample_config: ChromaSampleConfig, destination: String) raises -> ChromaSamplePlan:
        if not model_type_is_chroma(self.model_type):
            raise Error(String("ChromaSampler.sample: unsupported ModelType ") + model_type_str(self.model_type))
        return chroma_sample_plan(sample_config, destination)

    def generate(self, sample_config: ChromaSampleConfig, destination: String) raises:
        _ = sample_config
        _ = destination
        raise Error("ChromaSampler.generate: build-only surface; Chroma denoise/decode runtime is not implemented")
