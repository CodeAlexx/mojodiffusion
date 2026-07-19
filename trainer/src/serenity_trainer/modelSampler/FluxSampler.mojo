# 1:1 surface port of Serenity modules/modelSampler/FluxSampler.py
#
# Build-only sampler support. The actual FLUX transformer/VAE/text encoder
# runtime is outside this worker's scope, so sample() returns a plan and
# generate() is explicitly unsupported. The plan mirrors FluxSampler.__sample_base
# and __sample_inpainting: 64px quantization, FlowMatch scheduler copied from the
# loaded model, timestep mu=log(calculate_timestep_shift), packed 16-channel
# latents, optional fill conditioning channels, transformer guidance from
# cfg_scale, and VAE decode with scaling_factor/shift_factor.
#
# Serenity creates the initial latent image as torch.float32 before packing and
# casting the transformer input to model.train_dtype. This file records that
# reference reason as metadata; it does not create a persistent F32 tensor.

from serenity_trainer.util.enum.ModelType import (
    model_type_has_conditioning_image_input,
    model_type_is_flux_1,
    model_type_str,
)


comptime FLUX_SAMPLE_FILE_TYPE_IMAGE = 0
comptime FLUX_SAMPLE_RESOLUTION_QUANTIZATION = 64
comptime FLUX_SAMPLE_VAE_SCALE_FACTOR = 8
comptime FLUX_SAMPLE_LATENT_CHANNELS = 16
comptime FLUX_SAMPLE_PACK_FACTOR = 2
comptime FLUX_SAMPLE_PACKED_LATENT_CHANNELS = 64
comptime FLUX_SAMPLE_FILL_MASK_CHANNELS = 256
comptime FLUX_SAMPLE_BATCH_SIZE = 1


struct FluxSampleConfig(Movable):
    var prompt: String
    var negative_prompt: String
    var height: Int
    var width: Int
    var seed: Int
    var random_seed: Bool
    var diffusion_steps: Int
    var cfg_scale: Float32
    var noise_scheduler: Int
    var sample_inpainting: Bool
    var base_image_path: String
    var mask_image_path: String
    var text_encoder_1_layer_skip: Int
    var text_encoder_2_layer_skip: Int
    var text_encoder_2_sequence_length: Int
    var transformer_attention_mask: Bool

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
        sample_inpainting: Bool = False,
        var base_image_path: String = String(),
        var mask_image_path: String = String(),
        text_encoder_1_layer_skip: Int = 0,
        text_encoder_2_layer_skip: Int = 0,
        text_encoder_2_sequence_length: Int = -1,
        transformer_attention_mask: Bool = False,
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
        self.sample_inpainting = sample_inpainting
        self.base_image_path = base_image_path^
        self.mask_image_path = mask_image_path^
        self.text_encoder_1_layer_skip = text_encoder_1_layer_skip
        self.text_encoder_2_layer_skip = text_encoder_2_layer_skip
        self.text_encoder_2_sequence_length = text_encoder_2_sequence_length
        self.transformer_attention_mask = transformer_attention_mask


struct FluxSamplePlan(Movable):
    var file_type: Int
    var destination: String
    var prompt: String
    var negative_prompt: String
    var negative_prompt_used: Bool
    var height: Int
    var width: Int
    var latent_h: Int
    var latent_w: Int
    var packed_h: Int
    var packed_w: Int
    var packed_sequence_length_formula: String
    var latent_channels: Int
    var packed_latent_channels: Int
    var vae_scale_factor: Int
    var pack_factor: Int
    var batch_size: Int
    var seed: Int
    var random_seed: Bool
    var seed_source: String
    var cfg_scale: Float32
    var cfg_batch_uses_negative_prompt: Bool
    var guidance_source: String
    var guidance_runtime_depends_on_transformer_config: Bool
    var requested_noise_scheduler: Int
    var requested_noise_scheduler_ignored: Bool
    var scheduler_source: String
    var scheduler_timestep_source: String
    var timestep_shift_source: String
    var diffusion_steps: Int
    var timestep_count: Int
    var inpainting_model_type: Bool
    var sample_inpainting: Bool
    var base_image_path: String
    var mask_image_path: String
    var prepares_conditioning_image: Bool
    var erodes_mask_before_encoding: Bool
    var erode_kernel_radius: Int
    var conditioning_image_source: String
    var appends_conditioning_latents_and_mask: Bool
    var fill_conditioning_concat_dim: Int
    var fill_mask_channels: Int
    var initial_noise_dtype: String
    var initial_noise_reference_reason: String
    var latent_state_dtype: String
    var transformer_input_dtype: String
    var guidance_input_dtype: String
    var prompt_embedding_input_dtype: String
    var pooled_prompt_embedding_input_dtype: String
    var text_ids_dtype_source: String
    var image_ids_dtype_source: String
    var packs_latents_before_denoise: Bool
    var unpacks_latents_before_decode: Bool
    var decode_input_dtype: String
    var decode_formula: String
    var postprocess_output_type: String
    var do_denormalize_default_true: Bool
    var text_encoder_1_layer_skip: Int
    var text_encoder_2_layer_skip: Int
    var text_encoder_2_sequence_length: Int
    var text_encoder_2_sequence_length_is_none: Bool
    var transformer_attention_mask: Bool

    def __init__(
        out self,
        config: FluxSampleConfig,
        var destination: String,
        height: Int,
        width: Int,
        inpainting_model_type: Bool,
    ):
        self.file_type = FLUX_SAMPLE_FILE_TYPE_IMAGE
        self.destination = destination^
        self.prompt = config.prompt.copy()
        self.negative_prompt = config.negative_prompt.copy()
        self.negative_prompt_used = False
        self.height = height
        self.width = width
        self.latent_h = height // FLUX_SAMPLE_VAE_SCALE_FACTOR
        self.latent_w = width // FLUX_SAMPLE_VAE_SCALE_FACTOR
        self.packed_h = self.latent_h // FLUX_SAMPLE_PACK_FACTOR
        self.packed_w = self.latent_w // FLUX_SAMPLE_PACK_FACTOR
        self.packed_sequence_length_formula = String("(height / 8 / 2) * (width / 8 / 2)")
        self.latent_channels = FLUX_SAMPLE_LATENT_CHANNELS
        self.packed_latent_channels = FLUX_SAMPLE_PACKED_LATENT_CHANNELS
        self.vae_scale_factor = FLUX_SAMPLE_VAE_SCALE_FACTOR
        self.pack_factor = FLUX_SAMPLE_PACK_FACTOR
        self.batch_size = FLUX_SAMPLE_BATCH_SIZE
        self.seed = config.seed
        self.random_seed = config.random_seed
        if config.random_seed:
            self.seed_source = String("torch.Generator.seed()")
        else:
            self.seed_source = String("torch.Generator.manual_seed(seed)")
        self.cfg_scale = config.cfg_scale
        self.cfg_batch_uses_negative_prompt = False
        self.guidance_source = String("torch.tensor([cfg_scale]) when transformer.config.guidance_embeds")
        self.guidance_runtime_depends_on_transformer_config = True
        self.requested_noise_scheduler = config.noise_scheduler
        self.requested_noise_scheduler_ignored = True
        self.scheduler_source = String("copy.deepcopy(model.noise_scheduler)")
        self.scheduler_timestep_source = String("noise_scheduler.set_timesteps(diffusion_steps, device=train_device, mu=log(shift)).timesteps")
        self.timestep_shift_source = String("model.calculate_timestep_shift(latent_h, latent_w)")
        self.diffusion_steps = config.diffusion_steps
        self.timestep_count = config.diffusion_steps
        self.inpainting_model_type = inpainting_model_type
        self.sample_inpainting = config.sample_inpainting
        self.base_image_path = config.base_image_path.copy()
        self.mask_image_path = config.mask_image_path.copy()
        self.prepares_conditioning_image = inpainting_model_type
        self.erodes_mask_before_encoding = inpainting_model_type and config.sample_inpainting
        self.erode_kernel_radius = 2
        self.conditioning_image_source = String("none")
        if inpainting_model_type:
            if config.sample_inpainting:
                self.conditioning_image_source = String("load base RGB and L mask, resize to height/width, erode mask, encode masked image")
            else:
                self.conditioning_image_source = String("zero RGB image and all-ones latent mask")
        self.appends_conditioning_latents_and_mask = inpainting_model_type
        self.fill_conditioning_concat_dim = -1
        self.fill_mask_channels = FLUX_SAMPLE_FILL_MASK_CHANNELS
        self.initial_noise_dtype = String("F32")
        self.initial_noise_reference_reason = String("Serenity torch.randn(..., dtype=torch.float32) before pack_latents and transformer dtype cast")
        self.latent_state_dtype = String("F32 reference latent before transformer input cast")
        self.transformer_input_dtype = String("model.train_dtype.torch_dtype()")
        self.guidance_input_dtype = String("model.train_dtype.torch_dtype() when guidance_embeds")
        self.prompt_embedding_input_dtype = String("model.train_dtype.torch_dtype()")
        self.pooled_prompt_embedding_input_dtype = String("model.train_dtype.torch_dtype()")
        self.text_ids_dtype_source = String("base path torch.zeros default; fill path casts txt_ids to train_dtype")
        self.image_ids_dtype_source = String("model.prepare_latent_image_ids(..., model.train_dtype.torch_dtype())")
        self.packs_latents_before_denoise = True
        self.unpacks_latents_before_decode = True
        self.decode_input_dtype = String("VAE runtime dtype")
        self.decode_formula = String("(latent_image / vae.config.scaling_factor) + vae.config.shift_factor")
        self.postprocess_output_type = String("pil")
        self.do_denormalize_default_true = True
        self.text_encoder_1_layer_skip = config.text_encoder_1_layer_skip
        self.text_encoder_2_layer_skip = config.text_encoder_2_layer_skip
        self.text_encoder_2_sequence_length = config.text_encoder_2_sequence_length
        self.text_encoder_2_sequence_length_is_none = config.text_encoder_2_sequence_length < 0
        self.transformer_attention_mask = config.transformer_attention_mask


def flux_quantize_resolution(resolution: Int, quantization: Int) -> Int:
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


def flux_sample_plan(
    model_type: Int,
    config: FluxSampleConfig,
    destination: String,
) raises -> FluxSamplePlan:
    if config.diffusion_steps <= 0:
        raise Error("FluxSampler.sample: diffusion_steps must be positive")
    var h = flux_quantize_resolution(
        config.height, FLUX_SAMPLE_RESOLUTION_QUANTIZATION
    )
    var w = flux_quantize_resolution(
        config.width, FLUX_SAMPLE_RESOLUTION_QUANTIZATION
    )
    return FluxSamplePlan(
        config,
        destination.copy(),
        h,
        w,
        model_type_has_conditioning_image_input(model_type),
    )


struct FluxSampler(Movable):
    var model_type: Int

    def __init__(out self, model_type: Int):
        self.model_type = model_type

    def sample(
        self,
        sample_config: FluxSampleConfig,
        destination: String,
    ) raises -> FluxSamplePlan:
        if not model_type_is_flux_1(self.model_type):
            raise Error(String("FluxSampler.sample: unsupported ModelType ") + model_type_str(self.model_type))
        return flux_sample_plan(self.model_type, sample_config, destination)

    def generate(self, sample_config: FluxSampleConfig, destination: String) raises:
        _ = sample_config
        _ = destination
        raise Error("FluxSampler.generate: build-only surface; FLUX denoise/decode runtime is not implemented")
