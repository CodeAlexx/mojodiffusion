"""Model tab for the native Serenity Trainer trainer UI."""

from mojoui.core.context import Context
from mojoui.core.textedit import TextEditState
from mojoui.widgets.form import (
    begin_form_panel,
    end_form_panel,
    edit_row,
    field_row,
    select_index_row,
    select_string_row,
    toggle_row,
)
from serenity_trainer.ui.TrainerConfigModel import TrainerUIConfig, trainer_ui_apply_model_preset
from serenity_trainer.ui.UITabCommon import row2, two_col_w, value_w


def _panel_h(ctx: Context, rows: Int32) -> Int32:
    var pad = ctx.theme.padding
    var header_h = pad * 3
    var text_header_h = ctx.theme.font_size_pt * 3
    if text_header_h > header_h:
        header_h = text_header_h
    var gaps = rows - 1
    if gaps < 0:
        gaps = 0
    return header_h + pad * 2 + ctx.theme.row_height * rows + ctx.theme.spacing * gaps


def _label_w(ctx: Context, panel_w: Int32) -> Int32:
    var inner_w = panel_w - ctx.theme.padding * 2
    var w = ctx.theme.font_size_pt * 8
    if w < 178:
        w = 178
    var max_w = inner_w - 196
    if max_w < 132:
        max_w = 132
    if w > max_w:
        w = max_w
    return w


def render_model_tab(
    mut ctx: Context,
    mut cfg: TrainerUIConfig,
    content_w: Int32,
    mut base_model_edit: TextEditState,
    mut output_dir_edit: TextEditState,
) raises:
    var cw = two_col_w(content_w)
    var label_w = _label_w(ctx, cw)
    var val_w = value_w(cw - ctx.theme.padding * 2, label_w)
    ctx.layout_row(row2(cw, cw), _panel_h(ctx, 7))
    begin_form_panel(ctx, String("BASE MODEL"), String("Method, architecture, and checkpoint"), ctx.theme.padding)
    # Capability honesty (UI wave 2): every wired runner trains LoRA only —
    # "Fine Tune"/"Embedding" reach no runner (snapshot label only). The
    # capability-table warning names training_method when non-default.
    _ = select_index_row(ctx, label_w, val_w, String("Train Method [LoRA only]"), String("training_method"), cfg.training_method_options, cfg.training_method_index, cfg.select_open_id)
    var model_changed = select_index_row(ctx, label_w, val_w, String("Model Type"), String("model_type"), cfg.model_type_options, cfg.model_type_index, cfg.select_open_id)
    var arch_changed = select_index_row(ctx, label_w, val_w, String("Architecture"), String("architecture"), cfg.architecture_options, cfg.architecture_index, cfg.select_open_id)
    if model_changed or arch_changed:
        trainer_ui_apply_model_preset(cfg, model_changed)
    _ = edit_row(ctx, label_w, val_w, String("Base Model"), String("base_model_name"), cfg.base_model_name, base_model_edit)
    field_row(ctx, label_w, val_w, String("VAE"), cfg.vae_override.copy())
    _ = toggle_row(ctx, label_w, val_w, String("Transformer"), String("Train transformer"), cfg.train_transformer)
    _ = toggle_row(ctx, label_w, val_w, String("Text Encoder"), String("Train text encoder"), cfg.train_text_encoder)
    end_form_panel(ctx)

    begin_form_panel(ctx, String("OUTPUT"), String("Destination, format, and backend"), ctx.theme.padding)
    _ = edit_row(ctx, label_w, val_w, String("Destination"), String("output_model_destination"), cfg.output_model_destination, output_dir_edit)
    _ = select_string_row(ctx, label_w, val_w, String("Format"), String("output_model_format"), cfg.output_format_options, cfg.output_model_format, cfg.select_open_id)
    _ = select_string_row(ctx, label_w, val_w, String("Output DType"), String("output_dtype"), cfg.precision_options, cfg.output_dtype, cfg.select_open_id)
    field_row(ctx, label_w, val_w, String("Backend"), cfg.backend_target.copy())
    field_row(ctx, label_w, val_w, String("Checkpoint"), cfg.base_model_name.copy())
    _ = toggle_row(ctx, label_w, val_w, String("Bundle Embeds"), String("Enabled"), cfg.bundle_additional_embeddings)
    end_form_panel(ctx)
