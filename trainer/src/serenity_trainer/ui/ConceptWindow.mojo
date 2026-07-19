"""Concept edit window surface."""

from mojoui.core.context import Context
from mojoui.widgets.form import begin_form_panel, end_form_panel, field_row, toggle_row
from serenity_trainer.ui.AuxScreenCommon import aux_label_w, aux_panel_h
from serenity_trainer.ui.TrainerConfigModel import TrainerUIConfig
from serenity_trainer.ui.UITabCommon import row2, two_col_w, value_w


def render_concept_window(mut ctx: Context, mut cfg: TrainerUIConfig, content_w: Int32) raises:
    var cw = two_col_w(content_w)
    var label_w = aux_label_w(ctx, cw)
    var val_w = value_w(cw - ctx.theme.padding * 2, label_w)
    ctx.layout_row(row2(cw, cw), aux_panel_h(ctx, 7))
    begin_form_panel(ctx, String("CONCEPT GENERAL"), String("ConceptWindow core fields"))
    if len(cfg.concepts) > 0:
        field_row(ctx, label_w, val_w, String("Name"), cfg.concepts[0].name.copy())
        field_row(ctx, label_w, val_w, String("Path"), cfg.concepts[0].path.copy())
        field_row(ctx, label_w, val_w, String("Prompt Source"), String("caption sidecars"))
        field_row(ctx, label_w, val_w, String("Trigger Token"), cfg.concepts[0].trigger.copy())
        field_row(ctx, label_w, val_w, String("Type"), cfg.concepts[0].concept_type.copy())
        field_row(ctx, label_w, val_w, String("Repeats"), String(cfg.concepts[0].repeats))
        _ = toggle_row(ctx, label_w, val_w, String("Enabled"), String("Included"), cfg.concepts[0].enabled)
    end_form_panel(ctx)

    begin_form_panel(ctx, String("IMAGE & TEXT"), String("Augmentation, captions, and balancing"))
    field_row(ctx, label_w, val_w, String("Images"), String("scan dataset path"))
    field_row(ctx, label_w, val_w, String("Masks"), String("optional sidecars"))
    field_row(ctx, label_w, val_w, String("Caption Dropout"), String(cfg.caption_dropout))
    field_row(ctx, label_w, val_w, String("Loss Weight"), String("1.0"))
    field_row(ctx, label_w, val_w, String("Balancing"), String("repeats"))
    field_row(ctx, label_w, val_w, String("Cache"), String("latents/text embeddings"))
    field_row(ctx, label_w, val_w, String("Preview"), String("dataset media grid"))
    end_form_panel(ctx)
