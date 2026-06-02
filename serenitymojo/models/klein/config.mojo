# models/klein/config.mojo — Klein (FLUX.2) per-variant config accessors.
#
# Binding user rule (2026-05-31): NO hardcoded arch/recipe. These helpers now
# READ the variant's config file (serenitymojo/configs/klein{4b,9b}.json), which
# is the single source of truth. Pointing at a config file is not "coding params
# in" — the params live in the JSON, verified against the checkpoint header.

from serenitymojo.training.train_config import TrainConfig
from serenitymojo.io.train_config_reader import read_model_config


comptime KLEIN4B_CONFIG = "/home/alex/mojodiffusion/serenitymojo/configs/klein4b.json"
comptime KLEIN9B_CONFIG = "/home/alex/mojodiffusion/serenitymojo/configs/klein9b.json"


def klein_9b() raises -> TrainConfig:
    return read_model_config(String(KLEIN9B_CONFIG))


def klein_4b() raises -> TrainConfig:
    return read_model_config(String(KLEIN4B_CONFIG))
