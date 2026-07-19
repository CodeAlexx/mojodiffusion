"""Validations tab for the native Serenity Trainer trainer UI."""

from mojoui.core.context import Context
from mojoui.widgets.form import begin_form_panel, end_form_panel, field_row, toggle_row
from serenity_trainer.ui.TrainerConfigModel import TrainerUIConfig
from serenity_trainer.ui.UITabCommon import row2, value_w


def render_concepts_tab(mut ctx: Context, mut cfg: TrainerUIConfig, content_w: Int32) raises:
    var list_w: Int32 = 300
    var detail_w = content_w - list_w - 16
    if detail_w < 420:
        detail_w = 420
    var label_w: Int32 = 168
    var val_w = value_w(detail_w - 28, label_w)
    ctx.layout_row(row2(list_w, detail_w), 430)
    begin_form_panel(ctx, String("VALIDATIONS"), String("Serenity trainer gates"))
    for i in range(len(cfg.concepts)):
        var c = cfg.concepts[i].copy()
        field_row(ctx, 120, 120, c.name.copy(), String(c.image_count) + String(" imgs x") + String(c.repeats))
    end_form_panel(ctx)

    begin_form_panel(ctx, String("VALIDATION DETAIL"), String("Current training inputs"))
    if len(cfg.concepts) > 0:
        field_row(ctx, label_w, val_w, String("Dataset"), cfg.concepts[0].path.copy())
        field_row(ctx, label_w, val_w, String("Trigger"), cfg.concepts[0].trigger.copy())
        field_row(ctx, label_w, val_w, String("Images"), String(cfg.concepts[0].image_count))
        field_row(ctx, label_w, val_w, String("Cache"), cfg.cache_dir.copy())
        field_row(ctx, label_w, val_w, String("Model"), cfg.base_model_name.copy())
        field_row(ctx, label_w, val_w, String("Steps"), String(cfg.max_train_steps))
        _ = toggle_row(ctx, label_w, val_w, String("Enabled"), String("Run validation gates"), cfg.concepts[0].enabled)
    else:
        field_row(ctx, label_w, val_w, String("Validations"), String("No validations configured"))
    end_form_panel(ctx)
