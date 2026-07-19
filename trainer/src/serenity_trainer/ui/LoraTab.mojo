"""Adapter tab for the native Serenity Trainer trainer UI."""

from mojoui.core.context import Context
from mojoui.widgets.form import (
    begin_form_panel,
    end_form_panel,
    field_row,
    select_string_row,
    slider_row,
    toggle_row,
)
from serenity_trainer.ui.TrainerConfigModel import TrainerUIConfig
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


def render_lora_tab(mut ctx: Context, mut cfg: TrainerUIConfig, content_w: Int32) raises:
    var cw = two_col_w(content_w)
    var label_w = _label_w(ctx, cw)
    var val_w = value_w(cw - ctx.theme.padding * 2, label_w)
    ctx.layout_row(row2(cw, cw), _panel_h(ctx, 6))
    begin_form_panel(ctx, String("ADAPTER PARAMETERS"), String("Network algorithm, rank, alpha, dropout"), ctx.theme.padding)
    _ = select_string_row(ctx, label_w, val_w, String("Algorithm"), String("peft_type_lora"), cfg.peft_options, cfg.peft_type, cfg.select_open_id)
    field_row(ctx, label_w, val_w, String("LoRA Name"), cfg.lora_model_name.copy())
    _ = slider_row(ctx, label_w, val_w, String("LoRA Rank"), String("lora_rank"), cfg.lora_rank, 1.0, 256.0)
    _ = slider_row(ctx, label_w, val_w, String("LoRA Alpha"), String("lora_alpha"), cfg.lora_alpha, 1.0, 256.0)
    _ = slider_row(ctx, label_w, val_w, String("Dropout"), String("lora_dropout"), cfg.lora_dropout, 0.0, 0.5)
    _ = select_string_row(ctx, label_w, val_w, String("Weight DType"), String("lora_weight_dtype"), cfg.precision_options, cfg.lora_weight_dtype, cfg.select_open_id)
    end_form_panel(ctx)

    begin_form_panel(ctx, String("TARGETS & OFT"), String("Module target and OFT controls"), ctx.theme.padding)
    field_row(ctx, label_w, val_w, String("Transformer"), String("double/single projections"))
    field_row(ctx, label_w, val_w, String("Targets"), String("qkv, proj, mlp, modulation"))
    _ = slider_row(ctx, label_w, val_w, String("OFT Block Size"), String("oft_block_size"), cfg.oft_block_size, 2.0, 32.0)
    _ = toggle_row(ctx, label_w, val_w, String("COFT"), String("Enabled"), cfg.oft_coft)
    _ = toggle_row(ctx, label_w, val_w, String("Bundle Embeds"), String("Enabled"), cfg.bundle_additional_embeddings)
    field_row(ctx, label_w, val_w, String("Backend"), String("Ideogram4 LoRA"))
    end_form_panel(ctx)
