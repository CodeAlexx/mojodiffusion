# models/klein/config.mojo — Klein (FLUX.2) per-model config.
#
# The ONE place Klein diverges from the shared pipeline: dims + recipe values.
# Values cited from OneTrainer configs (klein9b_loss_compare.json /
# klein4b_benchmark.json) and models/dit/klein_dit.mojo KleinConfig.klein_9b().
# Dims CONFIRMED from the real safetensors headers (2026-05-30):
#   9B: inner 4096, heads 32, head_dim 128, mlp_hidden 12288, 8 double + 24
#       single = 32 blocks. lr 4e-4, shift 1.8 (project-validated), rank/alpha 16.
#   4B: inner 3072, heads 24, head_dim 128, mlp_hidden 9216, 5 double + 20
#       single = 25 blocks (flux-2-klein-base-4b.safetensors). lr 1e-4, shift 1.0.
# n_layers below is the TOTAL block count; the double/single split (8+24, 5+20)
# is needed for stacking and is recorded here until TrainConfig carries it.

from serenitymojo.training.train_config import TrainConfig


def klein_9b() -> TrainConfig:
    return TrainConfig(
        String("klein-9b"), 4096, 32, 128, 12288, 32,
        Float32(4.0e-4), Float32(1.8), 16, Float32(16.0), Float32(1.0e-6),
    )


def klein_4b() -> TrainConfig:
    # Dims confirmed from flux-2-klein-base-4b.safetensors header (G10 fixed):
    # inner 3072, heads 24, head_dim 128, mlp_hidden 9216, 25 blocks (5 dbl+20 sgl).
    return TrainConfig(
        String("klein-4b"), 3072, 24, 128, 9216, 25,
        Float32(1.0e-4), Float32(1.0), 16, Float32(16.0), Float32(1.0e-6),
    )
