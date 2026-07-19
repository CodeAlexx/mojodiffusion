"""General tab for the native Serenity Trainer trainer UI."""

from mojoui.core.context import Context
from mojoui.core.textedit import TextEditState
from mojoui.widgets.form import (
    begin_form_panel,
    end_form_panel,
    drag_row,
    edit_row,
    field_row,
    select_string_row,
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


def _compact_w(ctx: Context, value_width: Int32) -> Int32:
    var w = ctx.theme.font_size_pt * 5
    if w < 150:
        w = 150
    if w > value_width:
        w = value_width
    return w


def render_general_tab(
    mut ctx: Context,
    mut cfg: TrainerUIConfig,
    content_w: Int32,
    mut workspace_edit: TextEditState,
    mut cache_edit: TextEditState,
) raises:
    var cw = two_col_w(content_w)
    var label_w = _label_w(ctx, cw)
    var val_w = value_w(cw - ctx.theme.padding * 2, label_w)
    var compact_w = _compact_w(ctx, val_w)
    ctx.layout_row(row2(cw, cw), _panel_h(ctx, 6))
    begin_form_panel(ctx, String("WORKSPACE"), String("Paths, cache policy, and safety"), ctx.theme.padding)
    _ = edit_row(ctx, label_w, val_w, String("Workspace"), String("workspace_dir"), cfg.workspace_dir, workspace_edit)
    _ = edit_row(ctx, label_w, val_w, String("Cache"), String("cache_dir"), cfg.cache_dir, cache_edit)
    _ = toggle_row(ctx, label_w, val_w, String("Continue Backup"), String("Continue from last backup"), cfg.continue_last_backup)
    _ = toggle_row(ctx, label_w, val_w, String("Only Cache"), String("Only cache"), cfg.only_cache)
    _ = toggle_row(ctx, label_w, val_w, String("Overwrite"), String("Prevent overwrites"), cfg.prevent_overwrites)
    _ = drag_row(ctx, label_w, compact_w, String("Dataloader"), String("dataloader_threads"), cfg.dataloader_threads, 1.0)
    end_form_panel(ctx)

    begin_form_panel(ctx, String("DEBUG & VALIDATION"), String("SerenityBoard, debug, and validation"), ctx.theme.padding)
    _ = toggle_row(ctx, label_w, val_w, String("Debug Mode"), String("Enabled"), cfg.debug_mode)
    field_row(ctx, label_w, val_w, String("Debug Dir"), cfg.debug_dir.copy())
    _ = toggle_row(ctx, label_w, val_w, String("SerenityBoard"), String("Enabled"), cfg.tensorboard)
    _ = toggle_row(ctx, label_w, val_w, String("Always On"), String("Keep SerenityBoard open"), cfg.tensorboard_always_on)
    field_row(ctx, label_w, val_w, String("Board Port"), cfg.tensorboard_port.copy())
    _ = toggle_row(ctx, label_w, val_w, String("Validation"), String("Enabled"), cfg.validation)
    end_form_panel(ctx)

    ctx.layout_row(row2(cw, cw), _panel_h(ctx, 5))
    begin_form_panel(ctx, String("DEVICE"), String("Placement and precision defaults"), ctx.theme.padding)
    _ = select_string_row(ctx, label_w, val_w, String("Train Device"), String("train_device"), cfg.device_options, cfg.train_device, cfg.select_open_id)
    _ = select_string_row(ctx, label_w, val_w, String("Temp Device"), String("temp_device"), cfg.device_options, cfg.temp_device, cfg.select_open_id)
    _ = select_string_row(ctx, label_w, val_w, String("Train DType"), String("precision_general"), cfg.precision_options, cfg.train_dtype, cfg.select_open_id)
    _ = toggle_row(ctx, label_w, val_w, String("Gradient CKPT"), String("Enabled"), cfg.gradient_checkpointing)
    _ = toggle_row(ctx, label_w, val_w, String("Act Offload"), String("Enabled"), cfg.activation_offloading)
    end_form_panel(ctx)

    begin_form_panel(ctx, String("MULTI-GPU"), String("Distributed trainer switches"), ctx.theme.padding)
    _ = toggle_row(ctx, label_w, val_w, String("Multi-GPU"), String("Enabled"), cfg.multi_gpu)
    field_row(ctx, label_w, val_w, String("Devices"), cfg.device_indexes.copy())
    _ = toggle_row(ctx, label_w, val_w, String("Fused Reduce"), String("Enabled"), cfg.fused_gradient_reduce)
    _ = toggle_row(ctx, label_w, val_w, String("Async Reduce"), String("Enabled"), cfg.async_gradient_reduce)
    _ = drag_row(ctx, label_w, compact_w, String("Layer Offload"), String("layer_offload_fraction"), cfg.layer_offload_fraction, 0.05)
    end_form_panel(ctx)
