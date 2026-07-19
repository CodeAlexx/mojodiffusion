"""Caption dataset tool surface."""

from mojoui.core.context import Context
from mojoui.widgets.form import begin_form_panel, end_form_panel, field_row, toggle_row
from serenity_trainer.ui.AuxScreenCommon import aux_label_w, aux_panel_h
from serenity_trainer.ui.TrainerConfigModel import TrainerUIConfig
from serenity_trainer.ui.UITabCommon import row2, two_col_w, value_w


def render_caption_ui(mut ctx: Context, mut cfg: TrainerUIConfig, content_w: Int32) raises:
    var cw = two_col_w(content_w)
    var label_w = aux_label_w(ctx, cw)
    var val_w = value_w(cw - ctx.theme.padding * 2, label_w)
    ctx.layout_row(row2(cw, cw), aux_panel_h(ctx, 6))
    begin_form_panel(ctx, String("DATASET CAPTION TOOL"), String("Caption browse/edit surface"))
    var recursive = True
    var include_videos = True
    field_row(ctx, label_w, val_w, String("Dataset"), cfg.dataset_path.copy())
    field_row(ctx, label_w, val_w, String("Caption Ext"), String(".txt"))
    field_row(ctx, label_w, val_w, String("Backup Ext"), String(".bak"))
    _ = toggle_row(ctx, label_w, val_w, String("Recursive"), String("Scan subdirectories"), recursive)
    _ = toggle_row(ctx, label_w, val_w, String("Include Videos"), String("Show video captions"), include_videos)
    field_row(ctx, label_w, val_w, String("Status"), String("media indexed"))
    end_form_panel(ctx)

    begin_form_panel(ctx, String("CAPTION PREVIEW"), String("Selected file metadata"))
    field_row(ctx, label_w, val_w, String("File"), String("select from media grid"))
    field_row(ctx, label_w, val_w, String("Caption"), String("sidecar text"))
    field_row(ctx, label_w, val_w, String("Tokens"), String("computed by tokenizer"))
    field_row(ctx, label_w, val_w, String("Trigger"), String("rstprsn"))
    field_row(ctx, label_w, val_w, String("Save"), String("writes sidecar"))
    field_row(ctx, label_w, val_w, String("Batch"), String("generate / replace / append"))
    end_form_panel(ctx)
