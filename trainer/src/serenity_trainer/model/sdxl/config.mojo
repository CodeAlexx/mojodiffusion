# model/sdxl/config.mojo — SDXL trainer config accessor.
#
# Binding user rule (2026-05-31): NO hardcoded arch/recipe in a model file. This
# READS serenitymojo/configs/sdxl.json (the single source of truth consumed by
# the production driver serenitymojo/training/train_sdxl_real.mojo via
# read_model_config) using the same reader Ernie/Anima/Klein/Chroma use.
#
# 1:1 mirror of serenity_trainer.model.ernie.config (ernie_image()), namespaced
# for SDXL. SDXL field mapping (sdxl.json):
#   in_channels=4 ; out_channels=4 ; model_channels=320 ; channel_mult=[1,2,4] ;
#   num_res_blocks=2 ; context_dim=2048 (CLIP-L 768 + CLIP-G 1280 concat) ;
#   adm_in_channels=2816 (pooled 1280 + 6 time_ids x sin_embed_256 1536) ;
#   prediction_type="epsilon" ; scaled-linear beta 0.00085->0.012 / 1000 steps ;
#   lora_rank=16, lora_alpha=16 (scale 1.0), lr=1e-4, max_grad_norm=1.0 ;
#   AdamW beta1=0.9 beta2=0.999 eps=1e-8.

from serenitymojo.training.train_config import TrainConfig
from serenitymojo.io.train_config_reader import read_model_config


comptime SDXL_CONFIG = "/home/alex/mojodiffusion/serenitymojo/configs/sdxl.json"


def sdxl() raises -> TrainConfig:
    return read_model_config(String(SDXL_CONFIG))
