# model/anima/config.mojo — Anima (Cosmos-Predict2 MiniTrainDIT) config accessor.
#
# Binding user rule (2026-05-31): NO hardcoded arch/recipe in a model file. This
# READS serenitymojo/configs/anima.json (the single source of truth, verified
# against the checkpoint header by serenitymojo.models.anima.weights
# verify_anima_stack_shapes) via the same read_model_config reader Ernie/Klein/
# Chroma use.
#
# 1:1 mirror of serenitymojo/models/anima/config.mojo:34-35 (anima()), namespaced
# into serenity_trainer. Anima field mapping (anima.json):
#   inner_dim=2048 -> d_model ; in_channels=68 ((C+1)*PS*PS) ;
#   joint_attention_dim=1024 (cross-attn context) ; out_channels=64 (C*PS*PS) ;
#   num_double=0 ; num_single=28 ; num_heads=16 ; head_dim=128 ;
#   mlp_hidden=8192 (PLAIN GELU layer1/layer2, NOT SwiGLU) ; timestep_dim=2048 ;
#   rope_theta=10000.

from serenitymojo.training.train_config import TrainConfig
from serenitymojo.io.train_config_reader import read_model_config


comptime ANIMA_CONFIG = "/home/alex/mojodiffusion/serenitymojo/configs/anima.json"


def anima() raises -> TrainConfig:
    return read_model_config(String(ANIMA_CONFIG))
