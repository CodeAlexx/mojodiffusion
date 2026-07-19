"""Additional embeddings tab surface."""

from mojoui.core.context import Context
from mojoui.widgets.form import begin_form_panel, end_form_panel, field_row, toggle_row
from serenity_trainer.ui.AuxScreenCommon import aux_label_w, aux_panel_h
from serenity_trainer.ui.TrainerConfigModel import TrainerUIConfig
from serenity_trainer.ui.UITabCommon import row2, two_col_w, value_w


def render_additional_embeddings_tab(mut ctx: Context, mut cfg: TrainerUIConfig, content_w: Int32) raises:
    var cw = two_col_w(content_w)
    var label_w = aux_label_w(ctx, cw)
    var val_w = value_w(cw - ctx.theme.padding * 2, label_w)
    ctx.layout_row(row2(cw, cw), aux_panel_h(ctx, 5))
    begin_form_panel(ctx, String("ADDITIONAL EMBEDDINGS"), String("Embedding list and bundle behavior"))
    _ = toggle_row(ctx, label_w, val_w, String("Bundle"), String("Save with adapter"), cfg.bundle_additional_embeddings)
    field_row(ctx, label_w, val_w, String("Default Token"), String("rstprsn"))
    field_row(ctx, label_w, val_w, String("Placeholder"), String("<rstprsn>"))
    field_row(ctx, label_w, val_w, String("Initializer"), String("person"))
    field_row(ctx, label_w, val_w, String("Train"), String("enabled"))
    end_form_panel(ctx)

    begin_form_panel(ctx, String("EMBEDDING FILES"), String("Configured additional embedding artifacts"))
    field_row(ctx, label_w, val_w, String("Source"), String("training preset"))
    field_row(ctx, label_w, val_w, String("Output"), cfg.output_model_destination.copy())
    field_row(ctx, label_w, val_w, String("DType"), cfg.output_dtype.copy())
    field_row(ctx, label_w, val_w, String("Weight"), String("1.0"))
    field_row(ctx, label_w, val_w, String("Status"), String("ready"))
    end_form_panel(ctx)
