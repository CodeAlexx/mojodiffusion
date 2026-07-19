"""Muon auxiliary Adam settings surface."""

from mojoui.core.context import Context
from mojoui.widgets.form import begin_form_panel, end_form_panel, field_row, toggle_row
from serenity_trainer.ui.AuxScreenCommon import aux_label_w, aux_panel_h
from serenity_trainer.ui.TrainerConfigModel import TrainerUIConfig
from serenity_trainer.ui.UITabCommon import row1, value_w


def render_muon_adam_window(mut ctx: Context, cfg: TrainerUIConfig, content_w: Int32) raises:
    var label_w = aux_label_w(ctx, content_w)
    var val_w = value_w(content_w - ctx.theme.padding * 2, label_w)
    ctx.layout_row(row1(content_w), aux_panel_h(ctx, 8))
    begin_form_panel(ctx, String("MUON AUX ADAM"), String("Auxiliary AdamW settings for Muon"))
    var use_regex = False
    var rms_rescale = False
    var normuon = False
    field_row(ctx, label_w, val_w, String("Main LR"), String(cfg.learning_rate))
    field_row(ctx, label_w, val_w, String("Aux Adam LR"), String(""))
    field_row(ctx, label_w, val_w, String("TE1 LR"), String(""))
    field_row(ctx, label_w, val_w, String("TE2 LR"), String(""))
    field_row(ctx, label_w, val_w, String("Hidden Layers"), String("auto"))
    _ = toggle_row(ctx, label_w, val_w, String("Use Regex"), String("Enabled"), use_regex)
    _ = toggle_row(ctx, label_w, val_w, String("RMS Rescale"), String("Enabled"), rms_rescale)
    _ = toggle_row(ctx, label_w, val_w, String("NorMuon"), String("Enabled"), normuon)
    end_form_panel(ctx)
