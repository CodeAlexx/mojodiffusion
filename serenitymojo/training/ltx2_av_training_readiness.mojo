# training/ltx2_av_training_readiness.mojo
#
# Executable readiness contract for production LTX-2.3 AV LoRA training.
#
# This is intentionally not a trainer. It is a fail-closed gate that records the
# minimum production surface the real trainer must satisfy before
# train_ltx2_real.mojo can be advertised as AV-capable.
#
# Current repo state:
#   - inference spine: models/dit/ltx2_dit.mojo ltx2_block_forward_av
#   - backward spine:  missing for that AV block
#   - legacy trainer: models/ltx2/ltx2_stack_lora.mojo, video attn1 only
#
# Run:
#   pixi run mojo run -I . serenitymojo/training/ltx2_av_training_readiness.mojo
# Expected current CI gate:
#   pixi run mojo run -I . serenitymojo/training/ltx2_av_training_readiness.mojo --expect-not-ready

from std.sys import argv

from serenitymojo.registry.checkpoints import path_exists
from serenitymojo.io.train_config_reader import read_model_config
from serenitymojo.training.train_config import (
    TrainConfig,
    TRAIN_MODALITY_AV,
    LORA_TARGET_LEGACY_VIDEO_ATTN1,
)


comptime LTX2_BLOCKS = 48

# musubi-tuner networks/lora_ltx2.py default LTX2_INCLUDE_PATTERNS_T2V:
#   .*\.to_k$, .*\.to_q$, .*\.to_v$, .*\.to_out\.0$
# It applies to all attention modules inside BasicAVTransformerBlock.
comptime ATTN_PROJECTIONS_PER_MODULE = 4
comptime T2V_ATTENTION_MODULES_PER_BLOCK = 6
comptime T2V_SLOTS_PER_BLOCK = T2V_ATTENTION_MODULES_PER_BLOCK * ATTN_PROJECTIONS_PER_MODULE
comptime T2V_ADAPTERS_TOTAL = LTX2_BLOCKS * T2V_SLOTS_PER_BLOCK

# musubi v2v preset adds video ff.net.{0.proj,2} and audio_ff.net.{0.proj,2}.
comptime V2V_EXTRA_FFN_SLOTS_PER_BLOCK = 4
comptime V2V_SLOTS_PER_BLOCK = T2V_SLOTS_PER_BLOCK + V2V_EXTRA_FFN_SLOTS_PER_BLOCK
comptime V2V_ADAPTERS_TOTAL = LTX2_BLOCKS * V2V_SLOTS_PER_BLOCK

# Current Mojo legacy trainer only covers attn1.{to_q,to_k,to_v,to_out.0}.
comptime LEGACY_VIDEO_SLOTS_PER_BLOCK = 4
comptime LEGACY_VIDEO_ADAPTERS_TOTAL = LTX2_BLOCKS * LEGACY_VIDEO_SLOTS_PER_BLOCK

comptime DEFAULT_TRAIN_CACHE = "/home/alex/datasets/ltx2_cache_512"
comptime LTX2_CONFIG = "/home/alex/mojodiffusion/serenitymojo/configs/ltx2.json"
comptime PROD_DIT = "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-distilled-fp8.safetensors"


def production_av_backward_ready() -> Bool:
    # There is no backward for models/dit/ltx2_dit.ltx2_block_forward_av today.
    return False


def legacy_trainer_matches_production_surface() -> Bool:
    return LEGACY_VIDEO_ADAPTERS_TOTAL >= T2V_ADAPTERS_TOTAL


def training_cache_ready() -> Bool:
    return path_exists(String(DEFAULT_TRAIN_CACHE))


def av_config_contract_ready(cfg: TrainConfig) -> Bool:
    return (
        cfg.name == String("ltx2")
        and cfg.train_modality == TRAIN_MODALITY_AV
        and cfg.lora_target_preset != LORA_TARGET_LEGACY_VIDEO_ATTN1
        and cfg.dataset_cache_dir != String("")
        and cfg.require_cached_video_latents
        and cfg.require_cached_text_embeddings
        and cfg.require_cached_audio_latents
        and cfg.hot_loop_device_only
        and cfg.video_loss_weight > Float32(0.0)
        and cfg.audio_loss_weight > Float32(0.0)
    )


def print_status() raises:
    var cfg = read_model_config(String(LTX2_CONFIG))

    print("=== LTX-2.3 AV training readiness ===")
    print("config:", LTX2_CONFIG)
    print("production DiT checkpoint:", PROD_DIT)
    print("default training cache:", DEFAULT_TRAIN_CACHE)
    print("")
    print("required production LoRA surface:")
    print("  t2v attention modules/block:", T2V_ATTENTION_MODULES_PER_BLOCK)
    print("  t2v adapters/block:", T2V_SLOTS_PER_BLOCK)
    print("  t2v adapters total:", T2V_ADAPTERS_TOTAL)
    print("  v2v adapters/block:", V2V_SLOTS_PER_BLOCK)
    print("  v2v adapters total:", V2V_ADAPTERS_TOTAL)
    print("")
    print("current legacy Mojo trainer surface:")
    print("  adapters/block:", LEGACY_VIDEO_SLOTS_PER_BLOCK)
    print("  adapters total:", LEGACY_VIDEO_ADAPTERS_TOTAL)
    print("")
    print("future AV trainer config contract:")
    print("  model_type:", cfg.name)
    print("  train_modality:", cfg.train_modality, " (need AV=", TRAIN_MODALITY_AV, ")")
    print("  lora_target_preset:", cfg.lora_target_preset, " (must not be legacy=", LORA_TARGET_LEGACY_VIDEO_ATTN1, ")")
    print("  dataset_cache_dir:", cfg.dataset_cache_dir)
    print("  require cached video/text/audio:",
          cfg.require_cached_video_latents, cfg.require_cached_text_embeddings,
          cfg.require_cached_audio_latents)
    print("  hot_loop_device_only:", cfg.hot_loop_device_only)
    print("  loss weights video/audio:", cfg.video_loss_weight, cfg.audio_loss_weight)
    print("")
    print("checks:")
    print("  checkpoint present:", path_exists(String(PROD_DIT)))
    print("  default cache present:", training_cache_ready())
    print("  AV backward implemented:", production_av_backward_ready())
    print("  legacy surface covers t2v:", legacy_trainer_matches_production_surface())
    print("  config declares AV cached-input device loop:", av_config_contract_ready(cfg))


def production_ready() raises -> Bool:
    var cfg = read_model_config(String(LTX2_CONFIG))
    return (
        path_exists(String(PROD_DIT))
        and training_cache_ready()
        and production_av_backward_ready()
        and legacy_trainer_matches_production_surface()
        and av_config_contract_ready(cfg)
    )


def main() raises:
    var a = argv()
    var expect_not_ready = False
    if len(a) >= 2 and String(a[1]) == "--expect-not-ready":
        expect_not_ready = True

    print_status()
    var ready = production_ready()

    if expect_not_ready:
        if ready:
            raise Error("ltx2_av_training_readiness: expected NOT READY, got READY")
        print("LTX-2.3 AV training readiness gate PASS: current state is correctly fail-closed")
        return

    if not ready:
        raise Error(
            "ltx2_av_training_readiness: production AV training is NOT READY; "
            "missing ltx2_dit AV backward, full AV LoRA surface, cached-input "
            "device-loop config, and/or cache"
        )

    print("LTX-2.3 AV training readiness PASS")
