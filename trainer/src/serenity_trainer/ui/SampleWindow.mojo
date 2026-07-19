"""Manual sampling tool surface."""

from mojoui.core.context import Context
from mojoui.widgets.form import begin_form_panel, end_form_panel, field_row
from serenity_trainer.ui.AuxScreenCommon import aux_label_w, aux_panel_h
from serenity_trainer.ui.SampleFrame import render_sample_frame
from serenity_trainer.ui.TrainerConfigModel import TrainerUIConfig
from serenity_trainer.ui.TrainerRuntimeBridge import TrainerUIRuntime
from serenity_trainer.ui.UITabCommon import row1, value_w


def render_sample_window(
    mut ctx: Context,
    mut cfg: TrainerUIConfig,
    rt: TrainerUIRuntime,
    content_w: Int32,
) raises:
    render_sample_frame(ctx, cfg, content_w)
    var label_w = aux_label_w(ctx, content_w)
    var val_w = value_w(content_w - ctx.theme.padding * 2, label_w)
    ctx.layout_row(row1(content_w), aux_panel_h(ctx, 4))
    begin_form_panel(ctx, String("LIVE SAMPLE STATUS"), String("External model sampling during training"))
    field_row(ctx, label_w, val_w, String("Training Run"), String(rt.run_id))
    field_row(ctx, label_w, val_w, String("Command"), rt.last_command.copy())
    field_row(ctx, label_w, val_w, String("Samples"), String(len(rt.samples)))
    field_row(ctx, label_w, val_w, String("Progress"), String("handled by sampler callbacks"))
    end_form_panel(ctx)
