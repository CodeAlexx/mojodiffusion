# model/ernie/config.mojo — ERNIE-Image per-variant config accessor.
#
# Binding user rule (2026-05-31): NO hardcoded arch/recipe in a model file. This
# READS serenitymojo/configs/ernie_image.json (the single source of truth,
# verified against the checkpoint header by serenitymojo.models.dit.ernie_contract)
# via the same read_model_config reader Klein/Chroma use.
#
# 1:1 mirror of serenitymojo/models/ernie/config.mojo:33-34 (ernie_image()),
# namespaced into serenity_trainer. ERNIE field mapping (config.mojo:9-24):
#   inner_dim=4096 -> d_model ; in_channels=128 ; joint_attention_dim=3072
#   (Mistral hidden) ; out_channels=128 ; num_double=0 ; num_single=36 ;
#   num_heads=32 ; head_dim=128 ; mlp_hidden=12288 (GELU-gated) ;
#   timestep_dim=4096 ; rope_theta=256 ; rope_axes_dim=[32,48,48].

from serenitymojo.training.train_config import TrainConfig
from serenitymojo.io.train_config_reader import read_model_config


comptime ERNIE_IMAGE_CONFIG = "/home/alex/mojodiffusion/serenitymojo/configs/ernie_image.json"


def ernie_image() raises -> TrainConfig:
    return read_model_config(String(ERNIE_IMAGE_CONFIG))
