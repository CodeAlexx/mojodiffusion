"""Sample parameter window surface."""

from mojoui.core.context import Context
from serenity_trainer.ui.SampleFrame import render_sample_frame
from serenity_trainer.ui.TrainerConfigModel import TrainerUIConfig


def render_sample_params_window(mut ctx: Context, mut cfg: TrainerUIConfig, content_w: Int32) raises:
    render_sample_frame(ctx, cfg, content_w)
