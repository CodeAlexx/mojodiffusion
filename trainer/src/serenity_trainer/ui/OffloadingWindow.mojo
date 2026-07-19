"""Offloading settings surface."""

from mojoui.core.context import Context
from mojoui.widgets.form import begin_form_panel, end_form_panel, field_row, slider_row, toggle_row
from serenity_trainer.ui.AuxScreenCommon import aux_label_w, aux_panel_h
from serenity_trainer.ui.TrainerConfigModel import TrainerUIConfig
from serenity_trainer.ui.UITabCommon import row1, value_w


def render_offloading_window(mut ctx: Context, mut cfg: TrainerUIConfig, content_w: Int32) raises:
    var label_w = aux_label_w(ctx, content_w)
    var val_w = value_w(content_w - ctx.theme.padding * 2, label_w)
    ctx.layout_row(row1(content_w), aux_panel_h(ctx, 6))
    begin_form_panel(ctx, String("OFFLOADING"), String("Gradient checkpointing and activation offload"))
    var async_offload = False
    _ = toggle_row(ctx, label_w, val_w, String("Gradient CKPT"), String("Enabled"), cfg.gradient_checkpointing)
    _ = toggle_row(ctx, label_w, val_w, String("Async Offload"), String("Enabled"), async_offload)
    _ = toggle_row(ctx, label_w, val_w, String("Activation Offload"), String("Enabled"), cfg.activation_offloading)
    _ = slider_row(ctx, label_w, val_w, String("Layer Fraction"), String("offload_window_fraction"), cfg.layer_offload_fraction, 0.0, 1.0)
    field_row(ctx, label_w, val_w, String("Train Device"), cfg.train_device.copy())
    field_row(ctx, label_w, val_w, String("Temp Device"), cfg.temp_device.copy())
    end_form_panel(ctx)
