"""Learning-rate scheduler parameters surface."""

from mojoui.core.context import Context
from mojoui.widgets.form import begin_form_panel, end_form_panel, field_row
from serenity_trainer.ui.AuxScreenCommon import aux_label_w, aux_panel_h
from serenity_trainer.ui.TrainerConfigModel import TrainerUIConfig
from serenity_trainer.ui.UITabCommon import row2, two_col_w, value_w


def render_scheduler_params_window(mut ctx: Context, mut cfg: TrainerUIConfig, content_w: Int32) raises:
    var cw = two_col_w(content_w)
    var label_w = aux_label_w(ctx, cw)
    var val_w = value_w(cw - ctx.theme.padding * 2, label_w)
    ctx.layout_row(row2(cw, cw), aux_panel_h(ctx, 6))
    begin_form_panel(ctx, String("SCHEDULER SETTINGS"), String("Serenity custom scheduler parameters"))
    field_row(ctx, label_w, val_w, String("Scheduler"), cfg.scheduler_label())
    field_row(ctx, label_w, val_w, String("Warmup Steps"), String(cfg.learning_rate_warmup_steps))
    field_row(ctx, label_w, val_w, String("Cycles"), String("1"))
    field_row(ctx, label_w, val_w, String("Min Factor"), String("0.0"))
    field_row(ctx, label_w, val_w, String("Class Name"), String(""))
    field_row(ctx, label_w, val_w, String("Step Unit"), String("Serenity optimizer step"))
    end_form_panel(ctx)

    begin_form_panel(ctx, String("KEY VALUE PARAMETERS"), String("Additional scheduler kwargs"))
    field_row(ctx, label_w, val_w, String("TOTAL_STEPS"), String("resolved at train start"))
    field_row(ctx, label_w, val_w, String("STEPS_PER_EPOCH"), String("dataset approximate length"))
    field_row(ctx, label_w, val_w, String("LR"), String(cfg.learning_rate))
    field_row(ctx, label_w, val_w, String("EPOCHS"), String(cfg.epochs))
    field_row(ctx, label_w, val_w, String("SCHEDULER_STEPS"), String("optimizer updates"))
    field_row(ctx, label_w, val_w, String("Custom"), String("key/value list"))
    end_form_panel(ctx)
