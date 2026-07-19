"""Generate captions tool surface."""

from mojoui.core.context import Context
from mojoui.widgets.form import begin_form_panel, end_form_panel, field_row, progress_row, toggle_row
from serenity_trainer.ui.AuxScreenCommon import aux_label_w, aux_panel_h
from serenity_trainer.ui.TrainerConfigModel import TrainerUIConfig
from serenity_trainer.ui.UITabCommon import row1, value_w


def render_generate_captions_window(mut ctx: Context, mut cfg: TrainerUIConfig, content_w: Int32) raises:
    var label_w = aux_label_w(ctx, content_w)
    var val_w = value_w(content_w - ctx.theme.padding * 2, label_w)
    ctx.layout_row(row1(content_w), aux_panel_h(ctx, 8))
    begin_form_panel(ctx, String("GENERATE CAPTIONS"), String("Caption model batch tool"))
    var replace = False
    var append_trigger = True
    field_row(ctx, label_w, val_w, String("Dataset"), cfg.dataset_path.copy())
    field_row(ctx, label_w, val_w, String("Model"), String("BLIP / WD14 / Florence"))
    field_row(ctx, label_w, val_w, String("Prompt"), String("describe image"))
    _ = toggle_row(ctx, label_w, val_w, String("Replace"), String("Overwrite existing captions"), replace)
    _ = toggle_row(ctx, label_w, val_w, String("Append Trigger"), String("Add trigger token"), append_trigger)
    field_row(ctx, label_w, val_w, String("Threads"), String(cfg.dataloader_threads))
    progress_row(ctx, label_w, val_w, String("Progress"), 0.0)
    field_row(ctx, label_w, val_w, String("Status"), String("ready"))
    end_form_panel(ctx)
