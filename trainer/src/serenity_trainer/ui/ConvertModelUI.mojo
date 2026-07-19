"""Model conversion tool surface."""

from mojoui.core.context import Context
from mojoui.widgets.form import begin_form_panel, end_form_panel, field_row
from serenity_trainer.ui.AuxScreenCommon import aux_label_w, aux_panel_h
from serenity_trainer.ui.TrainerConfigModel import TrainerUIConfig
from serenity_trainer.ui.UITabCommon import row2, two_col_w, value_w


def render_convert_model_ui(mut ctx: Context, mut cfg: TrainerUIConfig, content_w: Int32) raises:
    var cw = two_col_w(content_w)
    var label_w = aux_label_w(ctx, cw)
    var val_w = value_w(cw - ctx.theme.padding * 2, label_w)
    ctx.layout_row(row2(cw, cw), aux_panel_h(ctx, 6))
    begin_form_panel(ctx, String("CONVERT MODEL"), String("Checkpoint/diffusers conversion"))
    field_row(ctx, label_w, val_w, String("Input"), cfg.base_model_name.copy())
    field_row(ctx, label_w, val_w, String("Output"), cfg.output_model_destination.copy())
    field_row(ctx, label_w, val_w, String("Model Type"), cfg.model_type_label())
    field_row(ctx, label_w, val_w, String("Format"), cfg.output_model_format.copy())
    field_row(ctx, label_w, val_w, String("DType"), cfg.output_dtype.copy())
    field_row(ctx, label_w, val_w, String("Status"), String("ready"))
    end_form_panel(ctx)

    begin_form_panel(ctx, String("CONVERSION OPTIONS"), String("Serenity conversion flags"))
    field_row(ctx, label_w, val_w, String("Prune EMA"), String("optional"))
    field_row(ctx, label_w, val_w, String("Bundle VAE"), String("optional"))
    field_row(ctx, label_w, val_w, String("SafeTensors"), String("enabled"))
    field_row(ctx, label_w, val_w, String("Metadata"), String("preserve"))
    field_row(ctx, label_w, val_w, String("Validation"), String("load after write"))
    field_row(ctx, label_w, val_w, String("Backend"), String("Mojo converter"))
    end_form_panel(ctx)
