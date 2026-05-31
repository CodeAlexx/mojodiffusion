# models/zimage/config.mojo — Z-Image per-model config.
#
# Dims from models/dit/zimage_dit.mojo NextDiTConfig.zimage(): dim 3840, n_heads
# 30, head_dim 128, mlp_hidden 2560(*), 30 single-stream blocks. Recipe from
# OneTrainer alina_zimage_OTpreset_2000.json: lr 3e-4, AdamW defaults, rank 16,
# alpha 1.0, base bf16. eps 1e-5.
#   (*) mlp_hidden placeholder pending confirmation from the real weight header.

from serenitymojo.training.train_config import TrainConfig


def zimage() -> TrainConfig:
    return TrainConfig(
        String("zimage"), 3840, 30, 128, 2560, 30,
        Float32(3.0e-4), Float32(1.0), 16, Float32(1.0), Float32(1.0e-5),
    )
