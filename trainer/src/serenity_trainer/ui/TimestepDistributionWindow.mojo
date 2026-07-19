"""Timestep distribution settings surface."""

from mojoui.core.context import Context
from mojoui.widgets.form import begin_form_panel, drag_row, end_form_panel, field_row, toggle_row
from serenity_trainer.ui.AuxScreenCommon import aux_label_w, aux_panel_h
from serenity_trainer.ui.TrainerConfigModel import TrainerUIConfig
from serenity_trainer.ui.UITabCommon import row1, value_w


def render_timestep_distribution_window(mut ctx: Context, mut cfg: TrainerUIConfig, content_w: Int32) raises:
    var label_w = aux_label_w(ctx, content_w)
    var val_w = value_w(content_w - ctx.theme.padding * 2, label_w)
    ctx.layout_row(row1(content_w), aux_panel_h(ctx, 8))
    begin_form_panel(ctx, String("TIMESTEP DISTRIBUTION"), String("Noise/timestep sampling controls"))
    var dynamic_shift = False
    field_row(ctx, label_w, val_w, String("Distribution"), String("flow matching"))
    _ = drag_row(ctx, label_w, 150, String("Time Shift"), String("timestep_window_shift"), cfg.timestep_shift, 0.1)
    field_row(ctx, label_w, val_w, String("Weighting"), String("Min SNR / P2 / none"))
    _ = drag_row(ctx, label_w, 150, String("Weight Strength"), String("timestep_window_weight"), cfg.loss_weight_strength, 0.1)
    field_row(ctx, label_w, val_w, String("Noise Scheduler"), String("FlowMatch"))
    _ = toggle_row(ctx, label_w, val_w, String("Dynamic Shift"), String("Enabled"), dynamic_shift)
    field_row(ctx, label_w, val_w, String("Offset Noise"), String(cfg.offset_noise_weight))
    field_row(ctx, label_w, val_w, String("Status"), String("ready"))
    end_form_panel(ctx)
