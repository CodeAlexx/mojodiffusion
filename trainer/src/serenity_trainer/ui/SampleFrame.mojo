"""Sample frame surface shared by sample windows."""

from mojoui.core.context import Context
from mojoui.widgets.form import begin_form_panel, drag_row, end_form_panel, field_row
from serenity_trainer.ui.AuxScreenCommon import aux_label_w, aux_panel_h
from serenity_trainer.ui.TrainerConfigModel import TrainerUIConfig
from serenity_trainer.ui.UITabCommon import row2, two_col_w, value_w


def render_sample_frame(mut ctx: Context, mut cfg: TrainerUIConfig, content_w: Int32) raises:
    var cw = two_col_w(content_w)
    var label_w = aux_label_w(ctx, cw)
    var val_w = value_w(cw - ctx.theme.padding * 2, label_w)
    ctx.layout_row(row2(cw, cw), aux_panel_h(ctx, 7))
    begin_form_panel(ctx, String("SAMPLE PARAMETERS"), String("Prompt and sampler values"))
    _ = drag_row(ctx, label_w, 150, String("Steps"), String("sample_frame_steps"), cfg.sample_steps, 1.0)
    _ = drag_row(ctx, label_w, 150, String("CFG"), String("sample_frame_cfg"), cfg.sample_cfg, 0.1)
    field_row(ctx, label_w, val_w, String("Sampler"), cfg.sample_sampler.copy())
    field_row(ctx, label_w, val_w, String("Width"), String("1024"))
    field_row(ctx, label_w, val_w, String("Height"), String("1024"))
    field_row(ctx, label_w, val_w, String("Seed Mode"), String("fixed / random"))
    field_row(ctx, label_w, val_w, String("Backend"), String("Klein sampler"))
    end_form_panel(ctx)

    begin_form_panel(ctx, String("PROMPTS"), String("Configured sample prompts"))
    for i in range(len(cfg.samples)):
        field_row(ctx, label_w, val_w, String("Prompt ") + String(i + 1), cfg.samples[i].prompt.copy())
        field_row(ctx, label_w, val_w, String("Negative ") + String(i + 1), cfg.samples[i].negative_prompt.copy())
        field_row(ctx, label_w, val_w, String("Seed ") + String(i + 1), String(cfg.samples[i].seed))
    end_form_panel(ctx)
