"""Serenity trainer UI config model for the native Mojo trainer screen."""

from json.parser import loads


comptime UI_SECTION_GENERAL: Int32 = 0
comptime UI_SECTION_MODEL: Int32 = 1
comptime UI_SECTION_LORA: Int32 = 2
comptime UI_SECTION_DATASET: Int32 = 3
comptime UI_SECTION_CONCEPTS: Int32 = 4
comptime UI_SECTION_TRAINING: Int32 = 5
comptime UI_SECTION_SAMPLING: Int32 = 6
comptime UI_SECTION_BACKUP: Int32 = 7
comptime UI_SECTION_CLOUD: Int32 = 8
comptime UI_SECTION_RUNS: Int32 = 9
comptime UI_SECTION_LOGS: Int32 = 10
comptime UI_SECTION_CAPTIONER: Int32 = 11
comptime SERENITY_TRAINER_OUTPUT_DIR = "/home/alex/mojodiffusion/output"
comptime SERENITY_BOXJANA_DATASET_DIR = "/home/alex/1/datasets/boxjana"
comptime SERENITY_BOXJANA_KLEIN_CACHE = "/home/alex/flame-diffusion-archive/klein-trainer/cache/eri2_klein9b_512"  # verified 75-sample klein9b 512 cache (2026-07-03 recon); was alina_klein9b (06-11 audit)
comptime SERENITY_KLEIN9B_CHECKPOINT = "/home/alex/.serenity/models/checkpoints/flux-2-klein-base-9b.safetensors"
# Klein 9B config-driven runner (serenitymojo train_klein_real.mojo): the VAE
# and PRECACHED sample-prompt config mirror serenitymojo/configs/klein9b.json
# (the trainer's own default config; validate_klein_train_config re-asserts
# every arch dim at startup).
comptime SERENITY_KLEIN_VAE = "/home/alex/.serenity/models/vaes/flux2-vae.safetensors"
comptime SERENITY_KLEIN_SAMPLE_PROMPTS = "/home/alex/mojodiffusion/serenitymojo/configs/klein_samples.json"
comptime SERENITY_IDEOGRAM4_BASE = "/home/alex/.serenity/models/ideogram-4-fp8"
comptime SERENITY_IDEOGRAM4_CACHE = "/home/alex/serenity-trainer/output/eri2_ideogram4_cache.safetensors"  # verified 3.2GB cache (2026-07-03 recon); was ideogram4_giger_cache
# Krea-2 Raw — trainer lives in mojodiffusion because it is the shared
# model/runtime vertical; Serenity launches it as a live trainer binary.
comptime SERENITY_KREA2_CHECKPOINT = "/home/alex/.cache/huggingface/hub/models--krea--Krea-2-Raw/snapshots/4ad9f4b627a647fad78b3dfeebb09f2654aeb494/raw.safetensors"
comptime SERENITY_KREA2_VAE = "/home/alex/.cache/huggingface/hub/models--Qwen--Qwen-Image/snapshots/75e0b4be04f60ec59a75f475837eced720f823b6/vae"
comptime SERENITY_KREA2_CACHE = "/home/alex/eri2_stage_512/cache.safetensors"
comptime SERENITY_KREA2_WORKSPACE = "/home/alex/mojodiffusion/output/krea2_eri2_lora_adamw"
comptime SERENITY_KREA2_SAMPLE_PROMPTS = "/home/alex/mojodiffusion/serenitymojo/configs/krea2_samples.json"
# HiDream-O1 — serenitymojo train_hidream_o1_real (P4 of
# HIDREAM_O1_TRAINING_CAMPAIGN.md). Weights dir is COMPTIME in the trainer
# (MODEL_DIR) — base_model_name is carried for display/config only. The data
# path is a stage-A dir (images.safetensors + caption.<i>.txt from
# scripts/ideogram4_stage_images.py); giger stage = the campaign-verified one.
comptime SERENITY_HIDREAM_CHECKPOINT = "/home/alex/HiDream-O1-Image-Dev-weights"
comptime SERENITY_HIDREAM_STAGE = "/home/alex/serenity-trainer/output/eri2_ideogram4_staged"  # verified 115-sample exact stage-A format (2026-07-03 recon); was ideogram4_giger_stage

# Campaign-verified trainable verticals (serenitymojo train_<m>_real.mojo runners).
# Checkpoints / caches mirror /home/alex/mojodiffusion/serenitymojo/configs/<m>.json
# and the trainers' default cache dirs (verified on disk 2026-06-09).
comptime SERENITY_CHROMA_CHECKPOINT = "/home/alex/.serenity/models/checkpoints/chroma1_hd_bf16.safetensors"
comptime SERENITY_CHROMA_CACHE = "/home/alex/datasets/boxjana_chroma_edv2_512"
comptime SERENITY_CHROMA_VAE = "/home/alex/.cache/huggingface/hub/models--lodestones--Chroma1-HD/snapshots/0e0c60ece1e82b17cb7f77342d765ba5024c40c0/vae/diffusion_pytorch_model.safetensors"
comptime SERENITY_ERNIE_CHECKPOINT = "/home/alex/models/ERNIE-Image/transformer"
comptime SERENITY_ERNIE_CACHE = "/home/alex/EriDiffusion/EriDiffusion-v2/cache/boxjana_ernie_512_FIXED"
comptime SERENITY_ERNIE_VAE = "/home/alex/models/ERNIE-Image/vae/diffusion_pytorch_model.safetensors"
comptime SERENITY_ANIMA_CHECKPOINT = "/home/alex/.serenity/models/anima/split_files/diffusion_models/anima-base-v1.0.safetensors"
comptime SERENITY_ANIMA_CACHE = "/home/alex/EriDiffusion/EriDiffusion-v2/cache/anima_synth_smoke"
comptime SERENITY_ANIMA_VAE = "/home/alex/.serenity/models/anima/split_files/vae/qwen_image_vae.safetensors"
comptime SERENITY_SDXL_CHECKPOINT = "/home/alex/.serenity/models/checkpoints/sdxl_unet_bf16.safetensors"
comptime SERENITY_SDXL_CACHE = "/home/alex/EriDiffusion/EriDiffusion-v2/cache/eri2_sdxl_512_smoke"
comptime SERENITY_SDXL_VAE = "/home/alex/madebyollin_sdxl-vae-fp16-fix/sdxl_vae.safetensors"
comptime SERENITY_ZIMAGE_CHECKPOINT = "/home/alex/.serenity/models/zimage_base/transformer"
# train_zimage_real is comptime-shaped on production buckets 72x56/88x48
# (cap 224/256). The EriDiffusion 64x64/seq-512 caches raise "unsupported
# Z-Image production bucket" (measured 2026-06-09); this is the trainer's own
# prepare-output location — fails loud ("does not exist") until prepared.
comptime SERENITY_ZIMAGE_CACHE = "/home/alex/mojodiffusion/output/alina_zimage_cache"  # MISSING - needs restaged dataset (576x448/704x384 buckets); source AlinaAignatova lost (2026-07-03 recon)
comptime SERENITY_L2P_CHECKPOINT = "/home/alex/.serenity/models/checkpoints/L2P/model-1k-merge.safetensors"
# No prepared L2P pixel cache exists yet (2026-06-09); trainer preflight fails
# loud until one is built at this path.
comptime SERENITY_L2P_CACHE = "/home/alex/EriDiffusion/EriDiffusion-v2/cache/boxjana_l2p_512"  # was alina_l2p_cache (MISSING, 06-11 audit)
comptime SERENITY_SAMPLE_PROMPTS = "/home/alex/mojodiffusion/serenitymojo/configs/sample_prompts.example.json"
comptime SERENITY_ZIMAGE_SAMPLE_PROMPTS = "/home/alex/mojodiffusion/serenitymojo/configs/zimage_samples.json"
comptime SERENITY_ANIMA_SAMPLE_PROMPTS = "/home/alex/mojodiffusion/serenitymojo/configs/anima_samples.json"
comptime SERENITY_ERNIE_SAMPLE_PROMPTS = "/home/alex/mojodiffusion/serenitymojo/configs/ernie_samples.json"
comptime SERENITY_CHROMA_SAMPLE_PROMPTS = "/home/alex/mojodiffusion/serenitymojo/configs/chroma_samples.json"
comptime SERENITY_SDXL_SAMPLE_PROMPTS = "/home/alex/mojodiffusion/serenitymojo/configs/sdxl_samples.json"
comptime SERENITY_IDEOGRAM4_SAMPLE_PROMPTS = "/home/alex/mojodiffusion/serenitymojo/configs/ideogram4_samples.json"


struct TrainerUIConcept(Copyable, Movable):
    var name: String
    var path: String
    var trigger: String
    var image_count: Int32
    var repeats: Int32
    var concept_type: String
    var enabled: Bool

    def __init__(out self):
        self.name = String("")
        self.path = String("")
        self.trigger = String("")
        self.image_count = 0
        self.repeats = 1
        self.concept_type = String("STANDARD")
        self.enabled = True

    def __init__(
        out self,
        name: String,
        path: String,
        trigger: String,
        image_count: Int32,
        repeats: Int32,
        concept_type: String,
        enabled: Bool,
    ):
        self.name = name.copy()
        self.path = path.copy()
        self.trigger = trigger.copy()
        self.image_count = image_count
        self.repeats = repeats
        self.concept_type = concept_type.copy()
        self.enabled = enabled


struct TrainerUISample(Copyable, Movable):
    var prompt: String
    var negative_prompt: String
    var seed: Int32

    def __init__(out self):
        self.prompt = String("")
        self.negative_prompt = String("")
        self.seed = 42

    def __init__(out self, prompt: String, negative_prompt: String, seed: Int32):
        self.prompt = prompt.copy()
        self.negative_prompt = negative_prompt.copy()
        self.seed = seed


struct TrainerUIConfig(Movable):
    var section_index: Int32
    var backend_target: String
    var run_name: String

    var training_method_options: List[String]
    var training_method_index: Int32
    var training_method_open: Bool
    var model_type_options: List[String]
    var model_type_index: Int32
    var model_type_open: Bool
    var architecture_options: List[String]
    var architecture_index: Int32
    var architecture_open: Bool
    var optimizer_options: List[String]
    var optimizer_index: Int32
    var optimizer_open: Bool
    var scheduler_options: List[String]
    var scheduler_index: Int32
    var scheduler_open: Bool
    var precision_options: List[String]
    var precision_index: Int32
    var precision_open: Bool
    var cloud_type_options: List[String]
    var cloud_type_index: Int32
    var cloud_type_open: Bool
    var select_open_id: String
    var output_format_options: List[String]
    var device_options: List[String]
    var lr_scaler_options: List[String]
    var ema_options: List[String]
    var timestep_distribution_options: List[String]
    var loss_weight_options: List[String]
    var loss_scaler_options: List[String]
    var layer_filter_preset_options: List[String]
    var peft_options: List[String]
    var sample_sampler_options: List[String]
    var resolution_options: List[String]
    var captioner_model_options: List[String]
    var captioner_model_index: Int32
    var captioner_custom_model_id: String
    var captioner_quant_options: List[String]
    var captioner_quant_index: Int32
    var captioner_attention_options: List[String]
    var captioner_attention_index: Int32
    var captioner_resolution_options: List[String]
    var captioner_resolution_index: Int32
    var captioner_folder_path: String
    var captioner_prompt: String
    var captioner_skip_existing: Bool
    var captioner_summary_mode: Bool
    var captioner_one_sentence_mode: Bool
    var captioner_retain_preview: Bool
    var captioner_max_tokens: Float32

    var debug_mode: Bool
    var debug_dir: String
    var workspace_dir: String
    var cache_dir: String
    var tensorboard: Bool
    var tensorboard_expose: Bool
    var tensorboard_always_on: Bool
    var tensorboard_port: String
    var validation: Bool
    var continue_last_backup: Bool
    var prevent_overwrites: Bool
    var only_cache: Bool
    var dataloader_threads: Float32
    var train_device: String
    var temp_device: String
    var multi_gpu: Bool
    var device_indexes: String
    var fused_gradient_reduce: Bool
    var async_gradient_reduce: Bool

    var base_model_name: String
    var vae_override: String
    var output_model_destination: String
    var output_model_format: String
    var output_dtype: String
    var model_arch: String
    var model_quantize: Bool
    var model_quantize_text_encoder: Bool
    var model_low_vram: Bool
    var model_qtype_text_encoder: String
    var train_transformer: Bool
    var train_text_encoder: Bool
    var gradient_checkpointing: Bool
    var activation_offloading: Bool
    var layer_offload_fraction: Float32

    var dataset_path: String
    var sample_output_dir: String
    var concept_file_name: String
    var concepts: List[TrainerUIConcept]
    var aspect_ratio_bucketing: Bool
    var latent_caching: Bool
    var cache_text_embeddings: Bool
    var clear_cache_before_training: Bool
    var resolution: String
    var caption_extension: String
    var caption_dropout: Float32

    var learning_rate: Float32
    var text_encoder_learning_rate: Float32
    var transformer_learning_rate: Float32
    var epochs: Float32
    var max_train_steps: Float32
    var batch_size: Float32
    var gradient_accumulation_steps: Float32
    var learning_rate_warmup_steps: Float32
    var learning_rate_min_factor: Float32
    var learning_rate_cycles: Float32
    var learning_rate_scaler: String
    var weight_decay: Float32
    var train_dtype: String
    var fallback_train_dtype: String
    var ema_mode: String
    var ema_decay: Float32
    var ema_update_step_interval: Float32
    var enable_autocast_cache: Bool
    var frames: String
    var force_circular_padding: Bool
    var seed: Float32
    var clip_grad_norm: Float32
    var text_encoder_stop_after: Float32
    var text_encoder_sequence_length: String
    var transformer_stop_after: Float32
    var transformer_attention_mask: Bool
    var transformer_guidance_scale: Float32
    var loss_weight_strength: Float32
    var mse_strength: Float32
    var mae_strength: Float32
    var log_cosh_strength: Float32
    var huber_strength: Float32
    var huber_delta: Float32
    var vb_loss_strength: Float32
    var loss_weight_fn: String
    # ── T1.A loss levers (serenitymojo training/levers.mojo) ──
    # loss_fn: "mse"|"huber"|"smooth_l1" — torch-semantics selector consumed
    # by trainers that call levers_loss_grad (zimage first). SEPARATE from
    # the mse/mae/huber_strength combined-loss scheme above (klein) — do not
    # conflate. min_snr_gamma_flow is the SimpleTuner ε-style min(SNR,γ)/SNR
    # weight (0.0 = off); loss_weight_fn=MIN_SNR_GAMMA above is the klein
    # min(SNR,γ)/(SNR+1) lever. Defaults = all off (mse, γ_flow=0).
    var loss_fn: String
    var smooth_l1_beta: Float32
    var min_snr_gamma_flow: Float32
    # T2.B quantized-resident base weights (hidream emission carries it).
    # "OFF" | "fp8_e4m3". Fixed "OFF" default — TODO: dropdown widget; the
    # seam (runner JSON emission) already delivers it end-to-end.
    var quantized_resident: String
    var loss_scaler: String
    var offset_noise_weight: Float32
    var perturbation_noise_weight: Float32
    var timestep_distribution: String
    var timestep_type: String
    var noise_scheduler: String
    var min_noising_strength: Float32
    var max_noising_strength: Float32
    var noising_weight: Float32
    var noising_bias: Float32
    var timestep_shift: Float32
    var dynamic_timestep_shifting: Bool
    var masked_training: Bool
    var unmasked_probability: Float32
    var unmasked_weight: Float32
    var normalize_masked_area_loss: Bool
    var masked_prior_preservation_weight: Float32
    var custom_conditioning_image: Bool
    var layer_filter_preset: String
    var layer_filter: String
    var layer_filter_regex: Bool

    var peft_type: String
    var lora_model_name: String
    var lora_rank: Float32
    var lora_alpha: Float32
    var lora_dropout: Float32
    var lora_weight_dtype: String
    var bundle_additional_embeddings: Bool
    var oft_block_size: Float32
    var oft_coft: Bool

    var samples: List[TrainerUISample]
    var sample_after: Float32
    var sample_skip_first: Float32
    var sample_cfg: Float32
    var sample_steps: Float32
    var sample_sampler: String
    var sampler_preset: String
    var samples_to_tensorboard: Bool
    var non_ema_sampling: Bool

    var backup_after: Float32
    var rolling_backup: Bool
    var rolling_backup_count: Float32
    var backup_before_save: Bool
    var save_every: Float32
    var save_skip_first: Float32
    var save_max_keep: Float32
    var save_filename_prefix: String

    var cloud_host: String
    var cloud_port: String
    var cloud_user: String
    var cloud_workspace_dir: String
    var cloud_delete_workspace: Bool

    def __init__(out self):
        self.section_index = UI_SECTION_MODEL
        self.backend_target = String("klein")
        self.run_name = String("boxjana_klein9b_lora_v1")

        self.training_method_options = List[String]()
        self.training_method_options.append(String("LoRA"))
        self.training_method_options.append(String("Fine Tune"))
        self.training_method_options.append(String("Embedding"))
        self.training_method_index = 0
        self.training_method_open = False

        self.model_type_options = List[String]()
        self.model_type_options.append(String("IDEOGRAM_4"))
        self.model_type_options.append(String("FLUX_2"))
        self.model_type_options.append(String("STABLE_DIFFUSION_XL_10_BASE"))
        self.model_type_options.append(String("STABLE_DIFFUSION_35"))
        self.model_type_options.append(String("CHROMA_1"))
        self.model_type_options.append(String("ERNIE_IMAGE"))
        self.model_type_options.append(String("ANIMA"))
        self.model_type_options.append(String("Z_IMAGE"))
        self.model_type_options.append(String("Z_IMAGE_L2P"))
        self.model_type_options.append(String("LTX_2_VIDEO"))
        self.model_type_options.append(String("WAN_22_VIDEO"))
        # HiDream-O1 appended LAST so existing option indices stay stable.
        self.model_type_options.append(String("HIDREAM_O1"))
        # Krea-2 appended LAST so existing option indices stay stable.
        self.model_type_options.append(String("KREA_2"))
        self.model_type_index = 1
        self.model_type_open = False

        self.architecture_options = List[String]()
        self.architecture_options.append(String("Ideogram4 FP8"))
        self.architecture_options.append(String("Klein 9B"))
        self.architecture_options.append(String("Flux2 Dev"))
        self.architecture_options.append(String("SDXL 1.0"))
        self.architecture_options.append(String("Chroma1 HD"))
        self.architecture_options.append(String("Ernie Image"))
        self.architecture_options.append(String("Anima"))
        self.architecture_options.append(String("Z-Image"))
        self.architecture_options.append(String("Z-Image L2P"))
        self.architecture_options.append(String("LTX-2 AV"))
        self.architecture_options.append(String("Wan2.2 T2V 14B"))
        # HiDream-O1 appended LAST so existing option indices stay stable.
        self.architecture_options.append(String("HiDream O1"))
        # Krea-2 appended LAST so existing option indices stay stable.
        self.architecture_options.append(String("Krea 2 Raw"))
        self.architecture_index = 1
        self.architecture_open = False

        self.optimizer_options = List[String]()
        self.optimizer_options.append(String("ADAMW8BIT"))
        self.optimizer_options.append(String("ADAMW"))
        self.optimizer_options.append(String("CAME"))
        self.optimizer_options.append(String("ADAFACTOR"))
        self.optimizer_options.append(String("MUON"))
        # T1.C: serenitymojo schedule-free AdamW (training/adamw_schedulefree
        # .mojo via the levers optimizer dispatch). Appended LAST so existing
        # option indices stay stable.
        self.optimizer_options.append(String("SCHEDULE_FREE_ADAMW"))
        # ai-toolkit Automagic3 (serenitymojo training/automagic3.mojo via the
        # levers optimizer dispatch). Appended LAST so existing option indices
        # stay stable. Emits "AUTOMAGIC3" verbatim into the runner config
        # (io/train_config_reader.mojo _optimizer_int -> TRAIN_OPTIMIZER_AUTOMAGIC3).
        self.optimizer_options.append(String("AUTOMAGIC3"))
        self.optimizer_index = 1
        self.optimizer_open = False

        self.scheduler_options = List[String]()
        self.scheduler_options.append(String("COSINE"))
        self.scheduler_options.append(String("CONSTANT"))
        self.scheduler_options.append(String("LINEAR"))
        self.scheduler_index = 0
        self.scheduler_open = False

        self.precision_options = List[String]()
        self.precision_options.append(String("BFLOAT_16"))
        self.precision_options.append(String("FLOAT_16"))
        self.precision_options.append(String("FLOAT_32"))
        self.precision_index = 0
        self.precision_open = False

        self.cloud_type_options = List[String]()
        self.cloud_type_options.append(String("NONE"))
        self.cloud_type_options.append(String("RUNPOD"))
        self.cloud_type_options.append(String("LINUX"))
        self.cloud_type_index = 0
        self.cloud_type_open = False

        self.select_open_id = String("")

        self.output_format_options = List[String]()
        self.output_format_options.append(String("SAFETENSORS"))
        self.output_format_options.append(String("CKPT"))
        self.output_format_options.append(String("INTERNAL"))
        self.output_format_options.append(String("DIFFUSERS"))

        self.device_options = List[String]()
        self.device_options.append(String("cuda"))
        self.device_options.append(String("cpu"))

        self.lr_scaler_options = List[String]()
        self.lr_scaler_options.append(String("NONE"))
        self.lr_scaler_options.append(String("LINEAR"))
        self.lr_scaler_options.append(String("SQRT"))
        self.lr_scaler_options.append(String("COSINE"))

        self.ema_options = List[String]()
        self.ema_options.append(String("OFF"))
        self.ema_options.append(String("EMA"))

        self.timestep_distribution_options = List[String]()
        self.timestep_distribution_options.append(String("UNIFORM"))
        self.timestep_distribution_options.append(String("LOGIT_NORMAL"))
        self.timestep_distribution_options.append(String("MODE"))
        self.timestep_distribution_options.append(String("COSINE"))
        self.timestep_distribution_options.append(String("SIGMOID"))

        self.loss_weight_options = List[String]()
        self.loss_weight_options.append(String("MIN_SNR_GAMMA"))
        self.loss_weight_options.append(String("NONE"))
        self.loss_weight_options.append(String("P2"))
        self.loss_weight_options.append(String("DEBIASED_ESTIMATION"))

        self.loss_scaler_options = List[String]()
        self.loss_scaler_options.append(String("NONE"))
        self.loss_scaler_options.append(String("MIN_SNR_GAMMA"))
        self.loss_scaler_options.append(String("P2"))

        self.layer_filter_preset_options = List[String]()
        self.layer_filter_preset_options.append(String("full"))
        self.layer_filter_preset_options.append(String("attention"))
        self.layer_filter_preset_options.append(String("mlp"))
        self.layer_filter_preset_options.append(String("double_blocks"))
        self.layer_filter_preset_options.append(String("single_blocks"))

        self.peft_options = List[String]()
        self.peft_options.append(String("LORA"))
        self.peft_options.append(String("LOCON"))
        self.peft_options.append(String("LOKR"))
        self.peft_options.append(String("LOHA"))
        self.peft_options.append(String("DORA"))
        self.peft_options.append(String("OFT"))

        self.sample_sampler_options = List[String]()
        self.sample_sampler_options.append(String("Ideogram4 FlowMatch"))
        self.sample_sampler_options.append(String("FlowMatch Euler"))
        self.sample_sampler_options.append(String("Euler"))
        self.sample_sampler_options.append(String("DDIM"))
        self.sample_sampler_options.append(String("DPM++ 2M"))
        self.sample_sampler_options.append(String("UniPC"))

        self.resolution_options = List[String]()
        self.resolution_options.append(String("512"))
        self.resolution_options.append(String("768"))
        self.resolution_options.append(String("1024"))
        self.resolution_options.append(String("1280"))

        self.captioner_model_options = List[String]()
        self.captioner_model_options.append(String("Qwen/Qwen3.5-4B"))
        self.captioner_model_options.append(String("Qwen/Qwen3.5-9B"))
        self.captioner_model_options.append(String("Qwen/Qwen3-VL-4B-Instruct"))
        self.captioner_model_options.append(String("Qwen/Qwen3-VL-8B-Instruct"))
        self.captioner_model_options.append(String("Qwen/Qwen2.5-VL-3B-Instruct"))
        self.captioner_model_options.append(String("Qwen/Qwen2.5-VL-7B-Instruct"))
        self.captioner_model_options.append(String("Custom..."))
        self.captioner_model_index = 3
        self.captioner_custom_model_id = String("")

        self.captioner_quant_options = List[String]()
        self.captioner_quant_options.append(String("None"))
        self.captioner_quant_options.append(String("8-bit"))
        self.captioner_quant_options.append(String("4-bit"))
        self.captioner_quant_index = 1

        self.captioner_attention_options = List[String]()
        self.captioner_attention_options.append(String("flash_attention_2"))
        self.captioner_attention_options.append(String("eager"))
        self.captioner_attention_index = 0

        self.captioner_resolution_options = List[String]()
        self.captioner_resolution_options.append(String("auto"))
        self.captioner_resolution_options.append(String("auto_high"))
        self.captioner_resolution_options.append(String("fast"))
        self.captioner_resolution_options.append(String("high"))
        self.captioner_resolution_index = 0

        self.captioner_folder_path = String(SERENITY_BOXJANA_DATASET_DIR)
        self.captioner_prompt = String("Describe this media.")
        self.captioner_skip_existing = True
        self.captioner_summary_mode = False
        self.captioner_one_sentence_mode = False
        self.captioner_retain_preview = True
        self.captioner_max_tokens = 128.0

        self.debug_mode = False
        self.debug_dir = String("debug")
        self.workspace_dir = String("/home/alex/trainings/boxjana_klein9b_lora_v1")
        self.cache_dir = String(SERENITY_BOXJANA_KLEIN_CACHE)
        self.tensorboard = True
        self.tensorboard_expose = False
        self.tensorboard_always_on = False
        self.tensorboard_port = String("6006")
        self.validation = False
        self.continue_last_backup = False
        self.prevent_overwrites = True
        self.only_cache = False
        self.dataloader_threads = 8.0
        self.train_device = String("cuda")
        self.temp_device = String("cpu")
        self.multi_gpu = False
        self.device_indexes = String("0")
        self.fused_gradient_reduce = False
        self.async_gradient_reduce = False

        self.base_model_name = String(SERENITY_KLEIN9B_CHECKPOINT)
        self.vae_override = String("")
        self.output_model_destination = String(SERENITY_TRAINER_OUTPUT_DIR)
        self.output_model_format = String("SAFETENSORS")
        self.output_dtype = String("FLOAT_16")
        self.model_arch = String("klein9b")
        self.model_quantize = True
        self.model_quantize_text_encoder = True
        self.model_low_vram = True
        self.model_qtype_text_encoder = String("qfloat8")
        self.train_transformer = True
        self.train_text_encoder = False
        self.gradient_checkpointing = True
        self.activation_offloading = False
        self.layer_offload_fraction = 0.0

        self.dataset_path = String(SERENITY_BOXJANA_DATASET_DIR)
        self.sample_output_dir = String(SERENITY_TRAINER_OUTPUT_DIR)
        self.concept_file_name = String("concepts.json")
        self.concepts = List[TrainerUIConcept]()
        self.concepts.append(TrainerUIConcept(String("boxjana"), String(SERENITY_BOXJANA_DATASET_DIR), String("box1jana"), 22, 1, String("STANDARD"), True))
        self.aspect_ratio_bucketing = True
        self.latent_caching = True
        self.cache_text_embeddings = True
        self.clear_cache_before_training = False
        self.resolution = String("512")
        self.caption_extension = String("txt")
        # T1.D REQUIRED C13 companion: caption dropout now actually REACHES the
        # runners via the recipe JSON, so the UI default must be OFF (0.0) to
        # keep default runs byte-identical to the pre-T1.D baselines.
        self.caption_dropout = 0.0

        self.learning_rate = 0.0004
        self.text_encoder_learning_rate = 0.00001
        self.transformer_learning_rate = 0.0004
        self.epochs = 1.0
        self.max_train_steps = 3000.0
        self.batch_size = 1.0
        self.gradient_accumulation_steps = 1.0
        self.learning_rate_warmup_steps = 0.0
        self.learning_rate_min_factor = 0.0
        self.learning_rate_cycles = 1.0
        self.learning_rate_scaler = String("NONE")
        self.weight_decay = 0.01
        self.train_dtype = String("BFLOAT_16")
        self.fallback_train_dtype = String("BFLOAT_16")
        self.ema_mode = String("OFF")
        self.ema_decay = 0.999
        self.ema_update_step_interval = 5.0
        self.enable_autocast_cache = True
        self.frames = String("25")
        self.force_circular_padding = False
        self.seed = 42.0
        self.clip_grad_norm = 1.0
        self.text_encoder_stop_after = 30.0
        self.text_encoder_sequence_length = String("512")
        self.transformer_stop_after = 0.0
        self.transformer_attention_mask = False
        self.transformer_guidance_scale = 1.0
        self.loss_weight_strength = 5.0
        self.mse_strength = 1.0
        self.mae_strength = 0.0
        self.log_cosh_strength = 0.0
        self.huber_strength = 0.0
        self.huber_delta = 1.0
        self.vb_loss_strength = 0.0
        self.loss_weight_fn = String("MIN_SNR_GAMMA")
        self.loss_fn = String("mse")        # T1.A default-off
        self.smooth_l1_beta = 1.0
        self.min_snr_gamma_flow = 0.0       # 0.0 = off
        self.quantized_resident = String("OFF")  # T2.B default-off (C13)
        self.loss_scaler = String("NONE")
        self.offset_noise_weight = 0.0
        self.perturbation_noise_weight = 0.0
        self.timestep_distribution = String("UNIFORM")
        self.timestep_type = String("linear")
        self.noise_scheduler = String("flowmatch")
        self.min_noising_strength = 0.0
        self.max_noising_strength = 1.0
        self.noising_weight = 1.0
        self.noising_bias = 0.0
        self.timestep_shift = 1.0
        self.dynamic_timestep_shifting = False
        self.masked_training = False
        self.unmasked_probability = 0.0
        self.unmasked_weight = 0.0
        self.normalize_masked_area_loss = False
        self.masked_prior_preservation_weight = 1.0
        self.custom_conditioning_image = False
        self.layer_filter_preset = String("full")
        self.layer_filter = String("")
        self.layer_filter_regex = False

        self.peft_type = String("LORA")
        self.lora_model_name = String("boxjana_klein9b_lora_v1")
        self.lora_rank = 16.0
        self.lora_alpha = 16.0
        self.lora_dropout = 0.0
        self.lora_weight_dtype = String("FLOAT_32")
        self.bundle_additional_embeddings = True
        self.oft_block_size = 4.0
        self.oft_coft = False

        self.samples = List[TrainerUISample]()
        self.samples.append(TrainerUISample(String("box1jana, 512x512 portrait photo, confident smile, dark hair, sleek modern styling, simple studio background, natural skin detail, sharp focus."), String(""), 42))
        self.samples.append(TrainerUISample(String("box1jana, 512x512 seated portrait on an ornate chair, dark turtleneck, playful expression, soft studio light, clean white background."), String(""), 43))
        self.sample_after = 500.0
        self.sample_skip_first = 0.0
        self.sample_cfg = 7.0
        self.sample_steps = 20.0
        self.sample_sampler = String("FlowMatch Euler")
        self.sampler_preset = String("KLEIN_20")
        self.samples_to_tensorboard = True
        self.non_ema_sampling = False

        self.backup_after = 500.0
        self.rolling_backup = True
        self.rolling_backup_count = 5.0
        self.backup_before_save = True
        self.save_every = 500.0
        self.save_skip_first = 0.0
        self.save_max_keep = 4.0
        self.save_filename_prefix = String("boxjana_klein9b_lora_v1")

        self.cloud_host = String("")
        self.cloud_port = String("22")
        self.cloud_user = String("root")
        self.cloud_workspace_dir = String("/workspace/serenity")
        self.cloud_delete_workspace = False

    def section_label(self) -> String:
        if self.section_index == UI_SECTION_GENERAL:
            return String("General")
        if self.section_index == UI_SECTION_MODEL:
            return String("Model")
        if self.section_index == UI_SECTION_LORA:
            return String("LoRA / OFT")
        if self.section_index == UI_SECTION_DATASET:
            return String("Dataset")
        if self.section_index == UI_SECTION_CONCEPTS:
            return String("Validations")
        if self.section_index == UI_SECTION_TRAINING:
            return String("Training")
        if self.section_index == UI_SECTION_SAMPLING:
            return String("Sampling")
        if self.section_index == UI_SECTION_BACKUP:
            return String("Backup")
        if self.section_index == UI_SECTION_CLOUD:
            return String("Cloud")
        if self.section_index == UI_SECTION_RUNS:
            return String("Runs")
        if self.section_index == UI_SECTION_CAPTIONER:
            return String("Captioner")
        return String("Logs")

    def training_method_label(self) -> String:
        return self.training_method_options[Int(self.training_method_index)].copy()

    def model_type_label(self) -> String:
        return self.model_type_options[Int(self.model_type_index)].copy()

    def architecture_label(self) -> String:
        return self.architecture_options[Int(self.architecture_index)].copy()

    def optimizer_label(self) -> String:
        return self.optimizer_options[Int(self.optimizer_index)].copy()

    def optimizer_runner_value(self) -> String:
        """T1.C/T2.A: the optimizer enum string emitted into the RUNNER train
        config (serenitymojo io/train_config_reader.mojo _optimizer_int).
        The dropdown labels ADAMW / ADAFACTOR / SCHEDULE_FREE_ADAMW map
        verbatim; ADAMW8BIT maps to the runner enum ADAMW_8BIT (T2.A: bnb
        block-wise 8-bit AdamW, serenitymojo training/adamw8bit.mojo via the
        levers optimizer dispatch); everything else (CAME / MUON) is passed
        through UNCHANGED so the runner FAILS LOUD at config load with the
        supported list instead of silently training AdamW (the pre-T1.C
        behavior, when the runner config carried no optimizer tag at all)."""
        var label = self.optimizer_label()
        if label == String("ADAMW8BIT"):
            return String("ADAMW_8BIT")
        return label^

    def scheduler_label(self) -> String:
        return self.scheduler_options[Int(self.scheduler_index)].copy()

    def precision_label(self) -> String:
        return self.precision_options[Int(self.precision_index)].copy()

    def cloud_type_label(self) -> String:
        return self.cloud_type_options[Int(self.cloud_type_index)].copy()

    def captioner_model_label(self) -> String:
        return self.captioner_model_options[Int(self.captioner_model_index)].copy()

    def captioner_quant_label(self) -> String:
        return self.captioner_quant_options[Int(self.captioner_quant_index)].copy()

    def captioner_attention_label(self) -> String:
        return self.captioner_attention_options[Int(self.captioner_attention_index)].copy()

    def captioner_resolution_label(self) -> String:
        return self.captioner_resolution_options[Int(self.captioner_resolution_index)].copy()


def _arch_index_for_model_type(model_type_index: Int32) -> Int32:
    # model_type option index -> canonical architecture option index
    if model_type_index == 0:  # IDEOGRAM_4
        return 0
    if model_type_index == 1:  # FLUX_2
        return 1  # Klein 9B is the trainable FLUX_2 default
    if model_type_index == 2:  # STABLE_DIFFUSION_XL_10_BASE
        return 3
    if model_type_index == 4:  # CHROMA_1
        return 4
    if model_type_index == 5:  # ERNIE_IMAGE
        return 5
    if model_type_index == 6:  # ANIMA
        return 6
    if model_type_index == 7:  # Z_IMAGE
        return 7
    if model_type_index == 8:  # Z_IMAGE_L2P
        return 8
    if model_type_index == 9:  # LTX_2_VIDEO
        return 9
    if model_type_index == 10:  # WAN_22_VIDEO
        return 10
    if model_type_index == 11:  # HIDREAM_O1
        return 11
    if model_type_index == 12:  # KREA_2
        return 12
    return -1  # STABLE_DIFFUSION_35: no trainable runner yet


def _model_type_for_arch_index(architecture_index: Int32) -> Int32:
    # architecture option index -> model_type option index
    if architecture_index == 0:  # Ideogram4 FP8
        return 0
    if architecture_index == 1 or architecture_index == 2:  # Klein 9B / Flux2 Dev
        return 1
    if architecture_index == 3:  # SDXL 1.0
        return 2
    if architecture_index == 4:  # Chroma1 HD
        return 4
    if architecture_index == 5:  # Ernie Image
        return 5
    if architecture_index == 6:  # Anima
        return 6
    if architecture_index == 7:  # Z-Image
        return 7
    if architecture_index == 8:  # Z-Image L2P
        return 8
    if architecture_index == 9:  # LTX-2 AV
        return 9
    if architecture_index == 10:  # Wan2.2 T2V 14B
        return 10
    if architecture_index == 11:  # HiDream O1
        return 11
    if architecture_index == 12:  # Krea 2 Raw
        return 12
    return -1


def trainer_ui_apply_model_preset(mut cfg: TrainerUIConfig, prefer_model_type: Bool = True):
    # Resolve the canonical architecture from whichever selector the user changed.
    var arch = cfg.architecture_index
    if prefer_model_type:
        arch = _arch_index_for_model_type(cfg.model_type_index)
        if arch < 0:
            # STABLE_DIFFUSION_35 has no trainable runner (blocked refs).
            # Route to an unwired backend so launch fails loudly instead of
            # silently training the previously selected model.
            cfg.backend_target = String("sd35")
            cfg.model_arch = String("sd35")
            return
    else:
        var mt = _model_type_for_arch_index(cfg.architecture_index)
        if mt >= 0:
            cfg.model_type_index = mt

    if arch == 1 or arch == 2:
        cfg.backend_target = String("klein")
        cfg.model_type_index = 1
        cfg.architecture_index = 1
        cfg.base_model_name = String(SERENITY_KLEIN9B_CHECKPOINT)
        cfg.cache_dir = String(SERENITY_BOXJANA_KLEIN_CACHE)
        cfg.model_arch = String("klein9b")
        cfg.sample_sampler = String("FlowMatch Euler")
        cfg.sampler_preset = String("KLEIN_20")
    elif arch == 0:
        cfg.backend_target = String("ideogram4")
        cfg.model_type_index = 0
        cfg.architecture_index = 0
        cfg.base_model_name = String(SERENITY_IDEOGRAM4_BASE)
        cfg.cache_dir = String(SERENITY_IDEOGRAM4_CACHE)
        cfg.model_arch = String("ideogram4")
        cfg.sample_sampler = String("Ideogram4 FlowMatch")
        cfg.sampler_preset = String("V4_DEFAULT_20")
        cfg.samples = List[TrainerUISample]()
        cfg.samples.append(TrainerUISample(String("{\"caption\":\"gigerver3 biomechanical portrait, intricate organic machinery, sharp focus\"}"), String(""), 42))
    elif arch == 3:
        # SDXL — serenitymojo train_sdxl_real (eps-pred conv-UNet LoRA).
        # Recipe defaults mirror serenitymojo/configs/sdxl.json.
        cfg.backend_target = String("sdxl")
        cfg.model_type_index = 2
        cfg.architecture_index = 3
        cfg.base_model_name = String(SERENITY_SDXL_CHECKPOINT)
        cfg.cache_dir = String(SERENITY_SDXL_CACHE)
        cfg.model_arch = String("sdxl10")
        cfg.sample_sampler = String("Euler")
        cfg.sampler_preset = String("SDXL_20")
        cfg.learning_rate = 0.0001
        # train_sdxl_real is compiled for rank 16 (fails loud on any other).
        cfg.lora_rank = 16.0
        cfg.lora_alpha = 16.0
        cfg.timestep_shift = 1.0
    elif arch == 4:
        # Chroma1-HD — serenitymojo train_chroma_real (flow-match, block-swap).
        # Recipe defaults mirror serenitymojo/configs/chroma.json.
        cfg.backend_target = String("chroma")
        cfg.model_type_index = 4
        cfg.architecture_index = 4
        cfg.base_model_name = String(SERENITY_CHROMA_CHECKPOINT)
        cfg.cache_dir = String(SERENITY_CHROMA_CACHE)
        cfg.model_arch = String("chroma1hd")
        cfg.sample_sampler = String("FlowMatch Euler")
        cfg.sampler_preset = String("CHROMA_20")
        cfg.learning_rate = 0.0001
        # train_chroma_real is compiled for rank 16 (fails loud on any other).
        cfg.lora_rank = 16.0
        cfg.lora_alpha = 16.0
        cfg.timestep_shift = 1.15
    elif arch == 5:
        # Ernie Image — serenitymojo train_ernie_real.
        # Recipe defaults mirror serenitymojo/configs/ernie_image.json
        # (NOTE canonical lora_alpha is 1.0, not rank).
        cfg.backend_target = String("ernie")
        cfg.model_type_index = 5
        cfg.architecture_index = 5
        cfg.base_model_name = String(SERENITY_ERNIE_CHECKPOINT)
        cfg.cache_dir = String(SERENITY_ERNIE_CACHE)
        cfg.model_arch = String("ernie_image")
        cfg.sample_sampler = String("FlowMatch Euler")
        cfg.sampler_preset = String("ERNIE_20")
        cfg.learning_rate = 0.0003
        cfg.lora_alpha = 1.0
        cfg.timestep_shift = 1.0
    elif arch == 6:
        # Anima — serenitymojo train_anima_real.
        # Recipe defaults mirror serenitymojo/configs/anima.json.
        cfg.backend_target = String("anima")
        cfg.model_type_index = 6
        cfg.architecture_index = 6
        cfg.base_model_name = String(SERENITY_ANIMA_CHECKPOINT)
        cfg.cache_dir = String(SERENITY_ANIMA_CACHE)
        cfg.model_arch = String("anima")
        cfg.sample_sampler = String("FlowMatch Euler")
        cfg.sampler_preset = String("ANIMA_20")
        cfg.learning_rate = 0.0001
        cfg.lora_alpha = 16.0
        cfg.timestep_shift = 1.0
    elif arch == 7:
        # Z-Image — serenitymojo train_zimage_real.
        # train_zimage_real is compiled for rank=16, alpha=1.0, lr=3e-4
        # (fails loud on any other). Mirrors serenitymojo/configs/zimage.json.
        cfg.backend_target = String("zimage")
        cfg.model_type_index = 7
        cfg.architecture_index = 7
        cfg.base_model_name = String(SERENITY_ZIMAGE_CHECKPOINT)
        cfg.cache_dir = String(SERENITY_ZIMAGE_CACHE)
        cfg.model_arch = String("zimage")
        cfg.sample_sampler = String("FlowMatch Euler")
        cfg.sampler_preset = String("ZIMAGE_20")
        cfg.learning_rate = 0.0003
        cfg.lora_rank = 16.0
        cfg.lora_alpha = 1.0
        cfg.timestep_shift = 1.0
    elif arch == 8:
        # Z-Image L2P (pixel-space, VAE-less; Z-Image DiT body verbatim) —
        # serenitymojo train_l2p_real. Compiled for rank=16, alpha=16,
        # lr=3e-4, shift=3.0 (fails loud on any other). NOTE: no prepared
        # pixel cache exists yet — trainer preflight fails loud until
        # SERENITY_L2P_CACHE is built.
        cfg.backend_target = String("l2p")
        cfg.model_type_index = 8
        cfg.architecture_index = 8
        cfg.base_model_name = String(SERENITY_L2P_CHECKPOINT)
        cfg.cache_dir = String(SERENITY_L2P_CACHE)
        cfg.model_arch = String("zimage_l2p")
        cfg.sample_sampler = String("FlowMatch Euler")
        cfg.sampler_preset = String("L2P_20")
        cfg.learning_rate = 0.0003
        cfg.lora_rank = 16.0
        cfg.lora_alpha = 16.0
        cfg.timestep_shift = 3.0
    elif arch == 9:
        # LTX-2 AV (video+audio) — NO production trainer yet. The legacy
        # video-only train_ltx2_real is fail-closed by design; train_ltx2_av is
        # a readiness contract. Routed to an unwired backend so launch fails
        # loudly instead of silently running a legacy/unfaithful path.
        cfg.backend_target = String("ltx2")
        cfg.model_type_index = 9
        cfg.architecture_index = 9
        cfg.base_model_name = String("")
        cfg.model_arch = String("ltx2_av")
        cfg.frames = String("25")
    elif arch == 10:
        # Wan2.2-T2V 14B — train_wan22_real exists but its RoPE tables are
        # placeholders ("TODO: replace with wan22_build_rope for real
        # training") and it is config-less smoke-mode. Unwired backend:
        # fail-loud until the trainer is made faithful.
        cfg.backend_target = String("wan22")
        cfg.model_type_index = 10
        cfg.architecture_index = 10
        cfg.base_model_name = String("/home/alex/.serenity/models/checkpoints/wan2.2_t2v_low_noise_14b_fp16.safetensors")
        cfg.model_arch = String("wan22_t2v_14b")
        cfg.frames = String("1")
    elif arch == 11:
        # HiDream-O1 — serenitymojo train_hidream_o1_real (campaign-verified
        # ~1.0 s/step; flags-off 3-step anchor 0.05885428/0.33308488/
        # 0.5214583). Weights dir is comptime in the trainer; cache_dir is
        # the stage-A dir (argv 1). Recipe: lr 1e-4, rank 32, alpha=rank.
        cfg.backend_target = String("hidream")
        cfg.model_type_index = 11
        cfg.architecture_index = 11
        cfg.base_model_name = String(SERENITY_HIDREAM_CHECKPOINT)
        cfg.cache_dir = String(SERENITY_HIDREAM_STAGE)
        cfg.model_arch = String("hidream_o1")
        cfg.sample_sampler = String("FlowMatch Euler")
        cfg.sampler_preset = String("HIDREAM_20")
        cfg.learning_rate = 0.0001
        cfg.lora_rank = 32.0
        cfg.lora_alpha = 32.0
        cfg.timestep_shift = 3.0
    elif arch == 12:
        # Krea-2 Raw — mojodiffusion train_krea2.mojo (512px Eri2 cache,
        # fp8-resident base, inline sampler uses cached conditioning by default).
        cfg.backend_target = String("krea2")
        cfg.model_type_index = 12
        cfg.architecture_index = 12
        cfg.run_name = String("eri2_krea2_lora")
        cfg.base_model_name = String(SERENITY_KREA2_CHECKPOINT)
        cfg.vae_override = String(SERENITY_KREA2_VAE)
        cfg.cache_dir = String(SERENITY_KREA2_CACHE)
        cfg.workspace_dir = String(SERENITY_KREA2_WORKSPACE)
        cfg.sample_output_dir = String(SERENITY_KREA2_WORKSPACE) + String("/samples")
        cfg.model_arch = String("krea2_raw")
        cfg.sample_sampler = String("FlowMatch Euler")
        cfg.sampler_preset = String("KREA2_20")
        cfg.learning_rate = 0.0001
        cfg.lora_rank = 64.0
        cfg.lora_alpha = 64.0
        cfg.peft_type = String("LORA")
        cfg.timestep_shift = 1.0
        cfg.quantized_resident = String("fp8_e4m3")
        # Pinned to the PROVEN eri2 run recipe (user request 2026-07-04):
        # save/sample every 500, 30-step cfg-3.5 renders, sample-at-start via
        # krea2_samples.json defaults.
        cfg.sample_cfg = 3.5
        cfg.sample_steps = 30.0
        cfg.sample_after = 500.0
        cfg.max_train_steps = 2000.0
        cfg.save_every = 500.0
        cfg.save_filename_prefix = String("eri2_krea2")
        cfg.frames = String("1")
        cfg.resolution = String("512")
        cfg.text_encoder_sequence_length = String("384")


def trainer_ui_total_steps(cfg: TrainerUIConfig) -> Int32:
    if cfg.max_train_steps > 0.0:
        return Int32(cfg.max_train_steps)
    var steps = Int32(cfg.epochs * 120.0)
    if steps < 1:
        return 1
    return steps


def trainer_ui_validate(cfg: TrainerUIConfig) -> String:
    if cfg.base_model_name.byte_length() == 0:
        return String("Base model is required")
    if cfg.dataset_path.byte_length() == 0:
        return String("Dataset path is required")
    if cfg.lora_rank < 1.0:
        return String("LoRA rank must be >= 1")
    if cfg.learning_rate <= 0.0:
        return String("Learning rate must be > 0")
    if cfg.backend_target == String("ltx2") and trainer_ui_network_algorithm(cfg) != String("lora"):
        return String("LyCORIS adapters are disabled for LTX2")
    return String("Ready")


# ── UI wave 2: capability table (model -> lever keys its runner CONSUMES) ──
# ONE table drives the pre-launch "X ignores: ..." warning (TopBar validation
# line + submit log). MEASURED 2026-06-12 by grepping each runner's trainer
# source for actual consumption (not just reader parsing):
#   klein/zimage  (serenitymojo train_<m>_real): levers_loss_grad,
#     levers_optimizer_*, cfg.ema_enabled, cfg.caption_dropout_prob — ALL six.
#   hidream/ideogram4: same six (T1 fan-out + Ideogram4LiveTrainer argv 10/11).
#   krea2: loss/optimizer/network_algorithm are consumed by train_krea2; EMA
#     and caption_dropout are not consumed and should still warn if enabled.
#   chroma/ernie/anima/sdxl/l2p: ZERO hits for ema_enabled /
#     caption_dropout_prob / levers_* — the recipe-tail keys are parsed by
#     read_model_config but NEVER READ by those trainers.
#   masked_training / training_method (full-FT): consumed by NO runner.
#
# network_algorithm is emitted into runner configs. LoCon maps to the
# linear-LoRA-compatible LyCORIS path unless a model wires conv adapters;
# non-LoRA algorithms still fail loud in runners that have no real math path.


def trainer_ui_network_algorithm(cfg: TrainerUIConfig) -> String:
    if cfg.peft_type == String("LOCON"):
        return String("locon")
    if cfg.peft_type == String("LOKR"):
        return String("lokr")
    if cfg.peft_type == String("LOHA"):
        return String("loha")
    if cfg.peft_type == String("DORA"):
        return String("dora")
    if cfg.peft_type == String("OFT"):
        return String("oft")
    return String("lora")


def trainer_ui_supported_lever_keys(target: String) -> List[String]:
    var keys = List[String]()
    if (
        target == String("klein")
        or target == String("zimage")
        or target == String("hidream")
        or target == String("ideogram4")
    ):
        keys.append(String("optimizer"))
        keys.append(String("warmup"))
        keys.append(String("loss_fn"))
        keys.append(String("min_snr_gamma_flow"))
        keys.append(String("ema"))
        keys.append(String("caption_dropout"))
        keys.append(String("network_algorithm"))
    elif target == String("krea2"):
        keys.append(String("optimizer"))
        keys.append(String("warmup"))
        keys.append(String("loss_fn"))
        keys.append(String("min_snr_gamma_flow"))
        keys.append(String("network_algorithm"))
    elif (
        target == String("chroma")
        or target == String("ernie")
        or target == String("anima")
        or target == String("sdxl")
        or target == String("l2p")
        or target == String("qwen")
    ):
        keys.append(String("network_algorithm"))
    return keys^


def trainer_ui_active_lever_keys(cfg: TrainerUIConfig) -> List[String]:
    # Non-default lever/decorative widgets, stable display order. Defaults
    # produce an EMPTY list (C13: default launches warn about nothing).
    var keys = List[String]()
    if cfg.optimizer_runner_value() != String("ADAMW"):
        keys.append(String("optimizer"))
    if Int(cfg.learning_rate_warmup_steps) > 0:
        keys.append(String("warmup"))
    if cfg.loss_fn != String("mse"):
        keys.append(String("loss_fn"))
    if cfg.min_snr_gamma_flow != 0.0:
        keys.append(String("min_snr_gamma_flow"))
    if cfg.ema_mode != String("OFF"):
        keys.append(String("ema"))
    if cfg.caption_dropout > 0.0:
        keys.append(String("caption_dropout"))
    if cfg.masked_training:
        keys.append(String("masked_training"))
    if cfg.training_method_index != 0:
        keys.append(String("training_method"))
    if cfg.peft_type != String("LORA"):
        keys.append(String("network_algorithm"))
    return keys^


def trainer_ui_ignored_lever_summary(cfg: TrainerUIConfig) -> String:
    """\"<model> ignores: a, b\" for every non-default widget the selected
    model's runner does not consume; empty when nothing is ignored."""
    var active = trainer_ui_active_lever_keys(cfg)
    var supported = trainer_ui_supported_lever_keys(cfg.backend_target.copy())
    var ignored = String("")
    for i in range(len(active)):
        var hit = False
        for j in range(len(supported)):
            if supported[j] == active[i]:
                hit = True
                break
        if not hit:
            if ignored.byte_length() > 0:
                ignored = ignored + String(", ")
            ignored = ignored + active[i].copy()
    if ignored.byte_length() == 0:
        return String("")
    return cfg.backend_target.copy() + String(" ignores: ") + ignored


def _runner_recipe_json(cfg: TrainerUIConfig) -> String:
    # Shared recipe tail for the serenitymojo TrainConfig JSON schema
    # (io/train_config_reader.mojo read_model_config keys).
    var steps = Int(trainer_ui_total_steps(cfg))
    var save_every = Int(cfg.save_every)
    if save_every < 0:
        save_every = 0
    var rank = Int(cfg.lora_rank)
    if rank < 1:
        rank = 1
    return (
        String("  \"learning_rate\": ") + String(cfg.learning_rate) + String(",\n")
        + String("  \"network_algorithm\": \"") + trainer_ui_network_algorithm(cfg) + String("\",\n")
        + String("  \"adapter_algo\": \"") + trainer_ui_network_algorithm(cfg) + String("\",\n")
        + String("  \"lora_rank\": ") + String(rank) + String(",\n")
        + String("  \"lora_alpha\": ") + String(cfg.lora_alpha) + String(",\n")
        + String("  \"timestep_shift\": ") + String(cfg.timestep_shift) + String(",\n")
        + String("  \"caption_dropout_prob\": ") + String(cfg.caption_dropout) + String(",\n")
        # T1.B EMA — read_model_config maps "ema" OFF/EMA -> ema_enabled
        # (train_zimage_real consumes via training/lora_ema.mojo).
        + String("  \"ema\": \"") + cfg.ema_mode.copy() + String("\",\n")
        + String("  \"ema_decay\": ") + String(cfg.ema_decay) + String(",\n")
        + String("  \"ema_update_step_interval\": ") + String(Int(cfg.ema_update_step_interval)) + String(",\n")
        + String("  \"max_grad_norm\": ") + String(cfg.clip_grad_norm) + String(",\n")
        + String("  \"max_steps\": ") + String(steps) + String(",\n")
        + String("  \"save_every\": ") + String(save_every) + String(",\n")
        + String("  \"sample_every\": ") + String(Int(cfg.sample_after)) + String(",\n")
        + String("  \"seed\": ") + String(Int(cfg.seed)) + String(",\n")
        + String("  \"frames\": \"") + cfg.frames.copy() + String("\",\n")
        + String("  \"cache_dir\": \"") + cfg.cache_dir.copy() + String("\",\n")
        + String("  \"optimizer\": { \"eps\": 1e-8, \"weight_decay\": ")
        + String(cfg.weight_decay)
        + String(", \"beta1\": 0.9, \"beta2\": 0.999 }\n")
    )


def trainer_ui_runner_train_config_json(cfg: TrainerUIConfig) raises -> String:
    """TrainConfig-schema JSON for the serenitymojo train_<model>_real runners.

    Architecture dims are verbatim from the canonical
    /home/alex/mojodiffusion/serenitymojo/configs/<model>.json; the runner
    re-validates every dim against its comptime contract, so drift fails loud.
    Only the recipe (lr/rank/alpha/steps/save/cache/ckpt) comes from the UI.
    """
    var t = trainer_ui_json_escape(cfg.backend_target)
    if t == String("klein"):
        # Klein 9B — serenitymojo train_klein_real (config-driven; reads the
        # T1 lever keys via read_model_config since the levers fan-out commit
        # 12190f6 and CONSUMES them: levers_loss_grad, levers_optimizer_step,
        # cfg.ema_enabled, cfg.caption_dropout_prob). Arch dims verbatim from
        # serenitymojo/configs/klein9b.json — validate_klein_train_config
        # fails loud on any drift. Lever emission mirrors zimage's.
        return (
            String("{\n  \"model_type\": \"klein\",\n")
            + String("  \"checkpoint\": \"") + trainer_ui_json_escape(cfg.base_model_name) + String("\",\n")
            + String("  \"vae\": \"") + String(SERENITY_KLEIN_VAE) + String("\",\n")
            + String("  \"validation_prompts_file\": \"") + String(SERENITY_KLEIN_SAMPLE_PROMPTS) + String("\",\n")
            + String("  \"inner_dim\": 4096,\n  \"in_channels\": 128,\n")
            + String("  \"joint_attention_dim\": 12288,\n  \"out_channels\": 128,\n")
            + String("  \"num_double\": 8,\n  \"num_single\": 24,\n")
            + String("  \"num_heads\": 32,\n  \"head_dim\": 128,\n")
            + String("  \"mlp_hidden\": 12288,\n  \"timestep_dim\": 256,\n")
            + String("  \"rope_theta\": 2000,\n")
            + String("  \"loss_fn\": \"") + trainer_ui_json_escape(cfg.loss_fn) + String("\",\n")
            + String("  \"huber_delta\": ") + String(cfg.huber_delta) + String(",\n")
            + String("  \"smooth_l1_beta\": ") + String(cfg.smooth_l1_beta) + String(",\n")
            + String("  \"min_snr_gamma_flow\": ") + String(cfg.min_snr_gamma_flow) + String(",\n")
            + String("  \"optimizer\": { \"optimizer\": \"") + cfg.optimizer_runner_value() + String("\" },\n")
            + String("  \"optimizer_warmup_steps\": ") + String(Int(cfg.learning_rate_warmup_steps)) + String(",\n")
            + _runner_recipe_json(cfg)
            + String("}\n")
        )
    if t == String("krea2"):
        # Krea-2 Raw — mojodiffusion train_krea2.mojo. It has the same
        # TrainConfig reader as the config-driven runners, but the live argv is
        # `<cache.safetensors> <steps> <config.json>` because the cache is
        # positional in the Krea2 trainer.
        return (
            String("{\n  \"model_type\": \"krea2\",\n")
            + String("  \"checkpoint\": \"") + trainer_ui_json_escape(cfg.base_model_name) + String("\",\n")
            + String("  \"vae\": \"") + String(SERENITY_KREA2_VAE) + String("\",\n")
            + String("  \"validation_prompts_file\": \"") + String(SERENITY_KREA2_SAMPLE_PROMPTS) + String("\",\n")
            + String("  \"inner_dim\": 6144,\n  \"in_channels\": 64,\n")
            + String("  \"joint_attention_dim\": 2560,\n  \"out_channels\": 64,\n")
            + String("  \"num_double\": 0,\n  \"num_single\": 28,\n")
            + String("  \"num_heads\": 48,\n  \"head_dim\": 128,\n")
            + String("  \"mlp_hidden\": 16384,\n  \"timestep_dim\": 256,\n")
            + String("  \"rope_theta\": 1000,\n")
            + String("  \"quantized_resident\": \"") + trainer_ui_json_escape(cfg.quantized_resident) + String("\",\n")
            + String("  \"workspace_dir\": \"") + trainer_ui_json_escape(cfg.workspace_dir) + String("\",\n")
            + String("  \"save_filename_prefix\": \"") + trainer_ui_json_escape(cfg.save_filename_prefix) + String("\",\n")
            + String("  \"loss_fn\": \"") + trainer_ui_json_escape(cfg.loss_fn) + String("\",\n")
            + String("  \"huber_delta\": ") + String(cfg.huber_delta) + String(",\n")
            + String("  \"smooth_l1_beta\": ") + String(cfg.smooth_l1_beta) + String(",\n")
            + String("  \"min_snr_gamma_flow\": ") + String(cfg.min_snr_gamma_flow) + String(",\n")
            + String("  \"optimizer\": { \"optimizer\": \"") + cfg.optimizer_runner_value() + String("\" },\n")
            + String("  \"optimizer_warmup_steps\": ") + String(Int(cfg.learning_rate_warmup_steps)) + String(",\n")
            + _runner_recipe_json(cfg)
            + String("}\n")
        )
    if t == String("chroma"):
        return (
            String("{\n  \"model_type\": \"chroma\",\n")
            + String("  \"checkpoint\": \"") + trainer_ui_json_escape(cfg.base_model_name) + String("\",\n")
            + String("  \"vae\": \"") + String(SERENITY_CHROMA_VAE) + String("\",\n")
            + String("  \"validation_prompts_file\": \"") + String(SERENITY_CHROMA_SAMPLE_PROMPTS) + String("\",\n")
            + String("  \"inner_dim\": 3072,\n  \"in_channels\": 64,\n")
            + String("  \"joint_attention_dim\": 4096,\n  \"out_channels\": 64,\n")
            + String("  \"num_double\": 19,\n  \"num_single\": 38,\n")
            + String("  \"num_heads\": 24,\n  \"head_dim\": 128,\n")
            + String("  \"mlp_hidden\": 12288,\n  \"timestep_dim\": 256,\n")
            + String("  \"rope_theta\": 10000,\n")
            + _runner_recipe_json(cfg)
            + String("}\n")
        )
    if t == String("ernie"):
        return (
            String("{\n  \"model_type\": \"ernie_image\",\n")
            + String("  \"checkpoint\": \"") + trainer_ui_json_escape(cfg.base_model_name) + String("\",\n")
            + String("  \"vae\": \"") + String(SERENITY_ERNIE_VAE) + String("\",\n")
            + String("  \"validation_prompts_file\": \"") + String(SERENITY_ERNIE_SAMPLE_PROMPTS) + String("\",\n")
            + String("  \"inner_dim\": 4096,\n  \"in_channels\": 128,\n")
            + String("  \"joint_attention_dim\": 3072,\n  \"out_channels\": 128,\n")
            + String("  \"num_double\": 0,\n  \"num_single\": 36,\n")
            + String("  \"num_heads\": 32,\n  \"head_dim\": 128,\n")
            + String("  \"mlp_hidden\": 12288,\n  \"timestep_dim\": 4096,\n")
            + String("  \"rope_theta\": 256,\n  \"rope_axes_dim\": [32, 48, 48],\n")
            + _runner_recipe_json(cfg)
            + String("}\n")
        )
    if t == String("anima"):
        return (
            String("{\n  \"model_type\": \"anima\",\n")
            + String("  \"checkpoint\": \"") + trainer_ui_json_escape(cfg.base_model_name) + String("\",\n")
            + String("  \"vae\": \"") + String(SERENITY_ANIMA_VAE) + String("\",\n")
            + String("  \"validation_prompts_file\": \"") + String(SERENITY_ANIMA_SAMPLE_PROMPTS) + String("\",\n")
            + String("  \"inner_dim\": 2048,\n  \"in_channels\": 68,\n")
            + String("  \"joint_attention_dim\": 1024,\n  \"out_channels\": 64,\n")
            + String("  \"num_double\": 0,\n  \"num_single\": 28,\n")
            + String("  \"num_heads\": 16,\n  \"head_dim\": 128,\n")
            + String("  \"mlp_hidden\": 8192,\n  \"timestep_dim\": 2048,\n")
            + String("  \"rope_theta\": 10000,\n")
            + _runner_recipe_json(cfg)
            + String("}\n")
        )
    if t == String("zimage"):
        return (
            String("{\n  \"model_type\": \"zimage\",\n")
            + String("  \"checkpoint\": \"") + trainer_ui_json_escape(cfg.base_model_name) + String("\",\n")
            + String("  \"validation_prompts_file\": \"") + String(SERENITY_ZIMAGE_SAMPLE_PROMPTS) + String("\",\n")
            + String("  \"inner_dim\": 3840,\n  \"in_channels\": 16,\n")
            + String("  \"joint_attention_dim\": 2560,\n  \"out_channels\": 16,\n")
            + String("  \"num_double\": 0,\n  \"num_single\": 30,\n")
            + String("  \"num_noise_refiner\": 2,\n  \"num_context_refiner\": 2,\n")
            + String("  \"num_heads\": 30,\n  \"head_dim\": 128,\n")
            + String("  \"mlp_hidden\": 10240,\n  \"patch_size\": 2,\n")
            + String("  \"timestep_dim\": 1024,\n  \"min_mod\": 256,\n")
            + String("  \"rope_theta\": 256,\n  \"rope_axes_dim\": [32, 48, 48],\n")
            + String("  \"time_scale\": 1000.0,\n  \"pad_tokens_multiple\": 32,\n")
            + String("  \"norm_eps\": 1e-5,\n  \"final_norm_eps\": 1e-6,\n")
            # T1.A loss levers — zimage only this phase (train_zimage_real
            # calls serenitymojo training/levers.mojo levers_loss_grad).
            # Defaults (mse / 1.0 / 1.0 / 0.0) keep the lever OFF.
            + String("  \"loss_fn\": \"") + trainer_ui_json_escape(cfg.loss_fn) + String("\",\n")
            + String("  \"huber_delta\": ") + String(cfg.huber_delta) + String(",\n")
            + String("  \"smooth_l1_beta\": ") + String(cfg.smooth_l1_beta) + String(",\n")
            + String("  \"min_snr_gamma_flow\": ") + String(cfg.min_snr_gamma_flow) + String(",\n")
            # T1.C optimizer lever — zimage only this phase (train_zimage_real
            # wires training/levers.mojo levers_optimizer_step). Defaults
            # (ADAMW, warmup 0) keep the lever OFF; unsupported dropdown
            # values fail loud at runner config load. optimizer_warmup_steps
            # mirrors SimpleTuner's warmup_steps := args.lr_warmup_steps.
            + String("  \"optimizer\": { \"optimizer\": \"") + cfg.optimizer_runner_value() + String("\" },\n")
            + String("  \"optimizer_warmup_steps\": ") + String(Int(cfg.learning_rate_warmup_steps)) + String(",\n")
            + _runner_recipe_json(cfg)
            + String("}\n")
        )
    if t == String("l2p"):
        return (
            String("{\n  \"model_type\": \"l2p\",\n")
            + String("  \"checkpoint\": \"") + trainer_ui_json_escape(cfg.base_model_name) + String("\",\n")
            # No validation_prompts_file: L2P inline sampling is not wired, so
            # validate_l2p_train_config (train_l2p_real.mojo) fails loud on a
            # non-empty key. Use the standalone sampler for L2P validation.
            + String("  \"inner_dim\": 3840,\n  \"in_channels\": 3,\n")
            + String("  \"joint_attention_dim\": 2560,\n  \"out_channels\": 3,\n")
            + String("  \"num_double\": 0,\n  \"num_single\": 30,\n")
            + String("  \"num_noise_refiner\": 2,\n  \"num_context_refiner\": 2,\n")
            + String("  \"num_heads\": 30,\n  \"head_dim\": 128,\n")
            + String("  \"mlp_hidden\": 10240,\n  \"patch_size\": 16,\n")
            + String("  \"timestep_dim\": 1024,\n  \"min_mod\": 256,\n")
            + String("  \"rope_theta\": 256,\n  \"rope_axes_dim\": [32, 48, 48],\n")
            + String("  \"time_scale\": 1000.0,\n  \"pad_tokens_multiple\": 32,\n")
            + String("  \"norm_eps\": 1e-5,\n  \"final_norm_eps\": 1e-6,\n")
            + _runner_recipe_json(cfg)
            + String("}\n")
        )
    if t == String("sdxl"):
        return (
            String("{\n  \"model_type\": \"sdxl\",\n")
            + String("  \"checkpoint\": \"") + trainer_ui_json_escape(cfg.base_model_name) + String("\",\n")
            + String("  \"vae\": \"") + String(SERENITY_SDXL_VAE) + String("\",\n")
            + String("  \"validation_prompts_file\": \"") + String(SERENITY_SDXL_SAMPLE_PROMPTS) + String("\",\n")
            + String("  \"in_channels\": 4,\n  \"out_channels\": 4,\n")
            + String("  \"model_channels\": 320,\n  \"channel_mult\": [1, 2, 4],\n")
            + String("  \"num_res_blocks\": 2,\n  \"context_dim\": 2048,\n")
            + String("  \"head_dim\": 64,\n  \"adm_in_channels\": 2816,\n")
            + String("  \"num_groups\": 32,\n")
            + String("  \"transformer_depth_input\": [0, 0, 2, 2, 10, 10],\n")
            + String("  \"transformer_depth_middle\": 10,\n")
            + String("  \"transformer_depth_output\": [10, 10, 10, 2, 2, 2, 0, 0, 0],\n")
            + String("  \"time_embed_dim\": 1280,\n")
            + String("  \"beta_start\": 0.00085,\n  \"beta_end\": 0.012,\n")
            + String("  \"num_train_timesteps\": 1000,\n  \"prediction_type\": \"epsilon\",\n")
            + _runner_recipe_json(cfg)
            + String("}\n")
        )
    if t == String("hidream"):
        # HiDream-O1 — train_hidream_o1_real trailing argv [config.json]
        # (template: serenitymojo/configs/hidream_o1.json). No arch dims
        # required (reader keeps TrainConfig defaults for missing keys).
        # The trainer's argv keeps winning for steps/lr/rank/out_dir when
        # given; this JSON delivers the T1 levers + T2.B quantized_resident
        # + lora_rank/max_steps backstops. NOTE lora_alpha is carried but
        # the trainer is compiled alpha=rank (scale 1.0).
        return (
            String("{\n  \"model_type\": \"hidream_o1\",\n")
            + String("  \"checkpoint\": \"") + trainer_ui_json_escape(cfg.base_model_name) + String("\",\n")
            + String("  \"loss_fn\": \"") + trainer_ui_json_escape(cfg.loss_fn) + String("\",\n")
            + String("  \"huber_delta\": ") + String(cfg.huber_delta) + String(",\n")
            + String("  \"smooth_l1_beta\": ") + String(cfg.smooth_l1_beta) + String(",\n")
            + String("  \"min_snr_gamma_flow\": ") + String(cfg.min_snr_gamma_flow) + String(",\n")
            # T2.B quantized-resident base ("OFF" | "fp8_e4m3"; default OFF).
            + String("  \"quantized_resident\": \"") + trainer_ui_json_escape(cfg.quantized_resident) + String("\",\n")
            + String("  \"optimizer\": { \"optimizer\": \"") + cfg.optimizer_runner_value() + String("\" },\n")
            + String("  \"optimizer_warmup_steps\": ") + String(Int(cfg.learning_rate_warmup_steps)) + String(",\n")
            + _runner_recipe_json(cfg)
            + String("}\n")
        )
    if t == String("ideogram4"):
        # Ideogram4 — argv 11 levers JSON for serenity_ideogram4_live_trainer
        # (Ideogram4LiveTrainer.mojo header). The trainer syncs the shared
        # recipe scalars (lr/rank/alpha/steps/save) from argv 1-9 — this JSON
        # contributes ONLY the lever keys (loss_fn/min_snr_gamma_flow/ema_*/
        # optimizer*/caption_dropout_prob).
        return (
            String("{\n  \"model_type\": \"ideogram4\",\n")
            # Standard sample-prompt source: consumed by Ideogram4LiveTrainer as
            # the argv-17 fallback when the inline sampler is enabled (it reads
            # run_cfg.levers.validation_prompts_file). Timing-inert with the
            # sampler off. Delivered only when the levers JSON is passed
            # (trainer_ui_ideogram4_levers_set gates argv 11).
            + String("  \"validation_prompts_file\": \"") + String(SERENITY_IDEOGRAM4_SAMPLE_PROMPTS) + String("\",\n")
            + String("  \"loss_fn\": \"") + trainer_ui_json_escape(cfg.loss_fn) + String("\",\n")
            + String("  \"huber_delta\": ") + String(cfg.huber_delta) + String(",\n")
            + String("  \"smooth_l1_beta\": ") + String(cfg.smooth_l1_beta) + String(",\n")
            + String("  \"min_snr_gamma_flow\": ") + String(cfg.min_snr_gamma_flow) + String(",\n")
            + String("  \"optimizer\": { \"optimizer\": \"") + cfg.optimizer_runner_value() + String("\" },\n")
            + String("  \"optimizer_warmup_steps\": ") + String(Int(cfg.learning_rate_warmup_steps)) + String(",\n")
            + _runner_recipe_json(cfg)
            + String("}\n")
        )
    raise Error(String("no runner train config template for backend ") + t)


def trainer_ui_ideogram4_levers_set(cfg: TrainerUIConfig) -> Bool:
    """True when any T1 lever the ideogram4 levers JSON carries is non-default.

    Drives the argv 11 fail-safe (Ideogram4LiveTrainer contract): with every
    lever default-off the bridge passes "-" (the trainer's skip sentinel) so
    default runs stay byte-identical to the pre-bridge launches (C13).
    caption_dropout is NOT included — it travels on argv 10 regardless.
    """
    if cfg.loss_fn != String("mse"):
        return True
    if cfg.min_snr_gamma_flow != 0.0:
        return True
    if cfg.ema_mode != String("OFF"):
        return True
    if cfg.optimizer_runner_value() != String("ADAMW"):
        return True
    if Int(cfg.learning_rate_warmup_steps) > 0:
        return True
    return False


def trainer_ui_ideogram4_levers_path_or_skip(
    cfg: TrainerUIConfig, levers_json_path: String
) -> String:
    """argv 11 for serenity_ideogram4_live_trainer: the levers config JSON
    path when any lever is set, else "-" (trainer skip sentinel, C13)."""
    if trainer_ui_ideogram4_levers_set(cfg):
        return levers_json_path.copy()
    return String("-")




def trainer_ui_json_bool(v: Bool) -> String:
    """JSON booleans are lowercase; Mojo String(Bool) yields Python-style
    True/False, which made every saved snapshot malformed JSON (found live
    on disk during the 2026-07-03 UI audit - parse error at the first Bool)."""
    return String("true") if v else String("false")


def trainer_ui_json_escape(value: String) -> String:
    """Escape a string for a JSON double-quoted literal. Byte-accumulated so
    multi-byte UTF-8 survives. Fixes the UI-audit 2026-07-03 bug-2 class:
    user-typed quotes/backslashes in text fields made emitted JSON malformed."""
    var out = List[UInt8]()
    var vb = value.as_bytes()
    for i in range(value.byte_length()):
        var c = vb[i]
        if c == 0x22:
            out.append(0x5C)
            out.append(0x22)
        elif c == 0x5C:
            out.append(0x5C)
            out.append(0x5C)
        elif c == 0x0A:
            out.append(0x5C)
            out.append(0x6E)
        elif c == 0x0D:
            out.append(0x5C)
            out.append(0x72)
        elif c == 0x09:
            out.append(0x5C)
            out.append(0x74)
        elif c < 0x20:
            out.append(0x20)
        else:
            out.append(c)
    try:
        return String(from_utf8=out)
    except:
        return value.copy()


def trainer_ui_load_config_snapshot(mut cfg: TrainerUIConfig) -> Bool:
    """Load CURATED recipe fields from the last session snapshot so user-set
    values persist - notably max_train_steps, which otherwise resets to the
    struct default 3000 every launch (user report 2026-07-03). Model-agnostic
    numbers + run_name ONLY; paths and model selection stay preset-driven.
    Fail-soft: any parse problem leaves cfg untouched."""
    try:
        var f = open(String("target/serenity_trainer_ui_config.json"), "r")
        var text = f.read()
        f.close()
        var obj = loads(text)
        if obj.contains("max_train_steps"):
            var v = Float32(obj["max_train_steps"].as_float())
            if v > 0.0:
                cfg.max_train_steps = v
        if obj.contains("learning_rate"):
            var lr = Float32(obj["learning_rate"].as_float())
            if lr > 0.0:
                cfg.learning_rate = lr
        if obj.contains("lora_rank"):
            var r = Float32(obj["lora_rank"].as_float())
            if r >= 1.0:
                cfg.lora_rank = r
        if obj.contains("lora_alpha"):
            var a = Float32(obj["lora_alpha"].as_float())
            if a > 0.0:
                cfg.lora_alpha = a
        if obj.contains("save_every"):
            var se = Float32(obj["save_every"].as_float())
            if se >= 0.0:
                cfg.save_every = se
        if obj.contains("sample_after"):
            var sa = Float32(obj["sample_after"].as_float())
            if sa >= 0.0:
                cfg.sample_after = sa
        if obj.contains("sample_steps"):
            var ss = Float32(obj["sample_steps"].as_float())
            if ss > 0.0:
                cfg.sample_steps = ss
        if obj.contains("sample_cfg"):
            cfg.sample_cfg = Float32(obj["sample_cfg"].as_float())
        if obj.contains("seed"):
            cfg.seed = Float32(obj["seed"].as_float())
        if obj.contains("epochs"):
            var ep = Float32(obj["epochs"].as_float())
            if ep >= 1.0:
                cfg.epochs = ep
        if obj.contains("caption_dropout"):
            var cd = Float32(obj["caption_dropout"].as_float())
            if cd >= 0.0 and cd <= 1.0:
                cfg.caption_dropout = cd
        if obj.contains("run_name") and not obj["run_name"].is_null():
            var rn = obj["run_name"].as_string()
            if rn.byte_length() > 0:
                cfg.run_name = rn.copy()
        return True
    except:
        return False


def trainer_ui_config_json_snapshot(cfg: TrainerUIConfig) -> String:
    return (
        String("{\"schema\":\"serenity.trainer_ui.v1\",")
        + String("\"backend_target\":\"") + trainer_ui_json_escape(cfg.backend_target) + String("\",")
        + String("\"run_name\":\"") + trainer_ui_json_escape(cfg.run_name) + String("\",")
        + String("\"training_method\":\"") + trainer_ui_json_escape(cfg.training_method_label()) + String("\",")
        + String("\"model_type\":\"") + trainer_ui_json_escape(cfg.model_type_label()) + String("\",")
        + String("\"architecture\":\"") + trainer_ui_json_escape(cfg.architecture_label()) + String("\",")
        + String("\"base_model_name\":\"") + trainer_ui_json_escape(cfg.base_model_name) + String("\",")
        + String("\"vae_override\":\"") + trainer_ui_json_escape(cfg.vae_override) + String("\",")
        + String("\"output_model_destination\":\"") + trainer_ui_json_escape(cfg.output_model_destination) + String("\",")
        + String("\"output_model_format\":\"") + trainer_ui_json_escape(cfg.output_model_format) + String("\",")
        + String("\"output_dtype\":\"") + trainer_ui_json_escape(cfg.output_dtype) + String("\",")
        + String("\"model_arch\":\"") + trainer_ui_json_escape(cfg.model_arch) + String("\",")
        + String("\"model_quantize\":") + trainer_ui_json_bool(cfg.model_quantize) + String(",")
        + String("\"model_quantize_text_encoder\":") + trainer_ui_json_bool(cfg.model_quantize_text_encoder) + String(",")
        + String("\"model_low_vram\":") + trainer_ui_json_bool(cfg.model_low_vram) + String(",")
        + String("\"model_qtype_text_encoder\":\"") + trainer_ui_json_escape(cfg.model_qtype_text_encoder) + String("\",")
        + String("\"train_transformer\":") + trainer_ui_json_bool(cfg.train_transformer) + String(",")
        + String("\"train_text_encoder\":") + trainer_ui_json_bool(cfg.train_text_encoder) + String(",")
        + String("\"workspace_dir\":\"") + trainer_ui_json_escape(cfg.workspace_dir) + String("\",")
        + String("\"cache_dir\":\"") + trainer_ui_json_escape(cfg.cache_dir) + String("\",")
        + String("\"tensorboard\":") + trainer_ui_json_bool(cfg.tensorboard) + String(",")
        + String("\"validation\":") + trainer_ui_json_bool(cfg.validation) + String(",")
        + String("\"continue_last_backup\":") + trainer_ui_json_bool(cfg.continue_last_backup) + String(",")
        + String("\"prevent_overwrites\":") + trainer_ui_json_bool(cfg.prevent_overwrites) + String(",")
        + String("\"only_cache\":") + trainer_ui_json_bool(cfg.only_cache) + String(",")
        + String("\"dataloader_threads\":") + String(cfg.dataloader_threads) + String(",")
        + String("\"train_device\":\"") + trainer_ui_json_escape(cfg.train_device) + String("\",")
        + String("\"temp_device\":\"") + trainer_ui_json_escape(cfg.temp_device) + String("\",")
        + String("\"multi_gpu\":") + trainer_ui_json_bool(cfg.multi_gpu) + String(",")
        + String("\"device_indexes\":\"") + trainer_ui_json_escape(cfg.device_indexes) + String("\",")
        + String("\"dataset_path\":\"") + trainer_ui_json_escape(cfg.dataset_path) + String("\",")
        + String("\"sample_output_dir\":\"") + trainer_ui_json_escape(cfg.sample_output_dir) + String("\",")
        + String("\"concept_file_name\":\"") + trainer_ui_json_escape(cfg.concept_file_name) + String("\",")
        + String("\"aspect_ratio_bucketing\":") + trainer_ui_json_bool(cfg.aspect_ratio_bucketing) + String(",")
        + String("\"latent_caching\":") + trainer_ui_json_bool(cfg.latent_caching) + String(",")
        + String("\"cache_text_embeddings\":") + trainer_ui_json_bool(cfg.cache_text_embeddings) + String(",")
        + String("\"clear_cache_before_training\":") + trainer_ui_json_bool(cfg.clear_cache_before_training) + String(",")
        + String("\"resolution\":\"") + trainer_ui_json_escape(cfg.resolution) + String("\",")
        + String("\"caption_extension\":\"") + trainer_ui_json_escape(cfg.caption_extension) + String("\",")
        + String("\"caption_dropout\":") + String(cfg.caption_dropout) + String(",")
        + String("\"captioner_model\":\"") + trainer_ui_json_escape(cfg.captioner_model_label()) + String("\",")
        + String("\"captioner_custom_model_id\":\"") + trainer_ui_json_escape(cfg.captioner_custom_model_id) + String("\",")
        + String("\"captioner_quant\":\"") + trainer_ui_json_escape(cfg.captioner_quant_label()) + String("\",")
        + String("\"captioner_attention\":\"") + trainer_ui_json_escape(cfg.captioner_attention_label()) + String("\",")
        + String("\"captioner_resolution\":\"") + trainer_ui_json_escape(cfg.captioner_resolution_label()) + String("\",")
        + String("\"captioner_folder_path\":\"") + trainer_ui_json_escape(cfg.captioner_folder_path) + String("\",")
        + String("\"captioner_prompt\":\"") + trainer_ui_json_escape(cfg.captioner_prompt) + String("\",")
        + String("\"captioner_skip_existing\":") + trainer_ui_json_bool(cfg.captioner_skip_existing) + String(",")
        + String("\"captioner_summary_mode\":") + trainer_ui_json_bool(cfg.captioner_summary_mode) + String(",")
        + String("\"captioner_one_sentence_mode\":") + trainer_ui_json_bool(cfg.captioner_one_sentence_mode) + String(",")
        + String("\"captioner_retain_preview\":") + trainer_ui_json_bool(cfg.captioner_retain_preview) + String(",")
        + String("\"captioner_max_tokens\":") + String(cfg.captioner_max_tokens) + String(",")
        + String("\"learning_rate\":") + String(cfg.learning_rate) + String(",")
        + String("\"text_encoder_learning_rate\":") + String(cfg.text_encoder_learning_rate) + String(",")
        + String("\"transformer_learning_rate\":") + String(cfg.transformer_learning_rate) + String(",")
        + String("\"epochs\":") + String(cfg.epochs) + String(",")
        + String("\"max_train_steps\":") + String(cfg.max_train_steps) + String(",")
        + String("\"batch_size\":") + String(cfg.batch_size) + String(",")
        + String("\"gradient_accumulation_steps\":") + String(cfg.gradient_accumulation_steps) + String(",")
        + String("\"learning_rate_warmup_steps\":") + String(cfg.learning_rate_warmup_steps) + String(",")
        + String("\"learning_rate_min_factor\":") + String(cfg.learning_rate_min_factor) + String(",")
        + String("\"learning_rate_cycles\":") + String(cfg.learning_rate_cycles) + String(",")
        + String("\"learning_rate_scaler\":\"") + trainer_ui_json_escape(cfg.learning_rate_scaler) + String("\",")
        + String("\"weight_decay\":") + String(cfg.weight_decay) + String(",")
        + String("\"optimizer\":\"") + trainer_ui_json_escape(cfg.optimizer_label()) + String("\",")
        + String("\"scheduler\":\"") + trainer_ui_json_escape(cfg.scheduler_label()) + String("\",")
        + String("\"train_dtype\":\"") + trainer_ui_json_escape(cfg.train_dtype) + String("\",")
        + String("\"fallback_train_dtype\":\"") + trainer_ui_json_escape(cfg.fallback_train_dtype) + String("\",")
        + String("\"ema\":\"") + trainer_ui_json_escape(cfg.ema_mode) + String("\",")
        + String("\"ema_decay\":") + String(cfg.ema_decay) + String(",")
        + String("\"ema_update_step_interval\":") + String(cfg.ema_update_step_interval) + String(",")
        + String("\"enable_autocast_cache\":") + trainer_ui_json_bool(cfg.enable_autocast_cache) + String(",")
        + String("\"frames\":\"") + trainer_ui_json_escape(cfg.frames) + String("\",")
        + String("\"force_circular_padding\":") + trainer_ui_json_bool(cfg.force_circular_padding) + String(",")
        + String("\"seed\":") + String(cfg.seed) + String(",")
        + String("\"clip_grad_norm\":") + String(cfg.clip_grad_norm) + String(",")
        + String("\"text_encoder_stop_after\":") + String(cfg.text_encoder_stop_after) + String(",")
        + String("\"text_encoder_sequence_length\":\"") + trainer_ui_json_escape(cfg.text_encoder_sequence_length) + String("\",")
        + String("\"transformer_stop_after\":") + String(cfg.transformer_stop_after) + String(",")
        + String("\"transformer_attention_mask\":") + trainer_ui_json_bool(cfg.transformer_attention_mask) + String(",")
        + String("\"transformer_guidance_scale\":") + String(cfg.transformer_guidance_scale) + String(",")
        + String("\"loss_weight_strength\":") + String(cfg.loss_weight_strength) + String(",")
        + String("\"mse_strength\":") + String(cfg.mse_strength) + String(",")
        + String("\"mae_strength\":") + String(cfg.mae_strength) + String(",")
        + String("\"log_cosh_strength\":") + String(cfg.log_cosh_strength) + String(",")
        + String("\"huber_strength\":") + String(cfg.huber_strength) + String(",")
        + String("\"huber_delta\":") + String(cfg.huber_delta) + String(",")
        + String("\"vb_loss_strength\":") + String(cfg.vb_loss_strength) + String(",")
        + String("\"loss_weight_fn\":\"") + trainer_ui_json_escape(cfg.loss_weight_fn) + String("\",")
        + String("\"quantized_resident\":\"") + trainer_ui_json_escape(cfg.quantized_resident) + String("\",")
        + String("\"loss_scaler\":\"") + trainer_ui_json_escape(cfg.loss_scaler) + String("\",")
        + String("\"offset_noise_weight\":") + String(cfg.offset_noise_weight) + String(",")
        + String("\"perturbation_noise_weight\":") + String(cfg.perturbation_noise_weight) + String(",")
        + String("\"timestep_distribution\":\"") + trainer_ui_json_escape(cfg.timestep_distribution) + String("\",")
        + String("\"timestep_type\":\"") + trainer_ui_json_escape(cfg.timestep_type) + String("\",")
        + String("\"noise_scheduler\":\"") + trainer_ui_json_escape(cfg.noise_scheduler) + String("\",")
        + String("\"min_noising_strength\":") + String(cfg.min_noising_strength) + String(",")
        + String("\"max_noising_strength\":") + String(cfg.max_noising_strength) + String(",")
        + String("\"noising_weight\":") + String(cfg.noising_weight) + String(",")
        + String("\"noising_bias\":") + String(cfg.noising_bias) + String(",")
        + String("\"timestep_shift\":") + String(cfg.timestep_shift) + String(",")
        + String("\"dynamic_timestep_shifting\":") + trainer_ui_json_bool(cfg.dynamic_timestep_shifting) + String(",")
        + String("\"masked_training\":") + trainer_ui_json_bool(cfg.masked_training) + String(",")
        + String("\"unmasked_probability\":") + String(cfg.unmasked_probability) + String(",")
        + String("\"unmasked_weight\":") + String(cfg.unmasked_weight) + String(",")
        + String("\"normalize_masked_area_loss\":") + trainer_ui_json_bool(cfg.normalize_masked_area_loss) + String(",")
        + String("\"masked_prior_preservation_weight\":") + String(cfg.masked_prior_preservation_weight) + String(",")
        + String("\"custom_conditioning_image\":") + trainer_ui_json_bool(cfg.custom_conditioning_image) + String(",")
        + String("\"layer_filter_preset\":\"") + trainer_ui_json_escape(cfg.layer_filter_preset) + String("\",")
        + String("\"layer_filter\":\"") + trainer_ui_json_escape(cfg.layer_filter) + String("\",")
        + String("\"layer_filter_regex\":") + trainer_ui_json_bool(cfg.layer_filter_regex) + String(",")
        + String("\"peft_type\":\"") + trainer_ui_json_escape(cfg.peft_type) + String("\",")
        + String("\"network_algorithm\":\"") + trainer_ui_network_algorithm(cfg) + String("\",")
        + String("\"lora_model_name\":\"") + trainer_ui_json_escape(cfg.lora_model_name) + String("\",")
        + String("\"lora_rank\":") + String(cfg.lora_rank) + String(",")
        + String("\"lora_alpha\":") + String(cfg.lora_alpha) + String(",")
        + String("\"lora_dropout\":") + String(cfg.lora_dropout) + String(",")
        + String("\"lora_weight_dtype\":\"") + trainer_ui_json_escape(cfg.lora_weight_dtype) + String("\",")
        + String("\"oft_block_size\":") + String(cfg.oft_block_size) + String(",")
        + String("\"oft_coft\":") + trainer_ui_json_bool(cfg.oft_coft) + String(",")
        + String("\"sample_after\":") + String(cfg.sample_after) + String(",")
        + String("\"sample_skip_first\":") + String(cfg.sample_skip_first) + String(",")
        + String("\"sample_cfg\":") + String(cfg.sample_cfg) + String(",")
        + String("\"sample_steps\":") + String(cfg.sample_steps) + String(",")
        + String("\"sample_sampler\":\"") + trainer_ui_json_escape(cfg.sample_sampler) + String("\",")
        + String("\"sampler_preset\":\"") + trainer_ui_json_escape(cfg.sampler_preset) + String("\",")
        + String("\"samples_to_tensorboard\":") + trainer_ui_json_bool(cfg.samples_to_tensorboard) + String(",")
        + String("\"non_ema_sampling\":") + trainer_ui_json_bool(cfg.non_ema_sampling) + String(",")
        + String("\"backup_after\":") + String(cfg.backup_after) + String(",")
        + String("\"rolling_backup\":") + trainer_ui_json_bool(cfg.rolling_backup) + String(",")
        + String("\"rolling_backup_count\":") + String(cfg.rolling_backup_count) + String(",")
        + String("\"backup_before_save\":") + trainer_ui_json_bool(cfg.backup_before_save) + String(",")
        + String("\"save_every\":") + String(cfg.save_every) + String(",")
        + String("\"save_skip_first\":") + String(cfg.save_skip_first) + String(",")
        + String("\"save_max_keep\":") + String(cfg.save_max_keep) + String(",")
        + String("\"save_filename_prefix\":\"") + trainer_ui_json_escape(cfg.save_filename_prefix) + String("\",")
        + String("\"cloud_type\":\"") + trainer_ui_json_escape(cfg.cloud_type_label()) + String("\",")
        + String("\"cloud_host\":\"") + trainer_ui_json_escape(cfg.cloud_host) + String("\",")
        + String("\"cloud_port\":\"") + trainer_ui_json_escape(cfg.cloud_port) + String("\",")
        + String("\"cloud_user\":\"") + trainer_ui_json_escape(cfg.cloud_user) + String("\",")
        + String("\"cloud_workspace_dir\":\"") + trainer_ui_json_escape(cfg.cloud_workspace_dir) + String("\",")
        + String("\"cloud_delete_workspace\":") + trainer_ui_json_bool(cfg.cloud_delete_workspace) + String("}")
    )
