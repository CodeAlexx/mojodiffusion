"""Sampling tab for the native Serenity Trainer trainer UI."""

from mojoui.core.context import Context
from mojoui.core.textedit import TextEditState
from mojoui.widgets.form import begin_form_panel, end_form_panel, drag_row, edit_row, field_row, select_string_row, toggle_row
from mojoui.widgets.image import media_virtual_grid
from serenity_trainer.ui.TrainerConfigModel import TrainerUIConfig
from serenity_trainer.ui.UITabCommon import row1, row2, two_col_w, value_w


def _video_count(flags: List[Bool]) -> Int:
    var count = 0
    for i in range(len(flags)):
        if flags[i]:
            count += 1
    return count


def render_sampling_tab(
    mut ctx: Context,
    mut cfg: TrainerUIConfig,
    content_w: Int32,
    mut sample_output_edit: TextEditState,
    preview_textures: List[UInt32],
    preview_widths: List[Int32],
    preview_heights: List[Int32],
    preview_titles: List[String],
    preview_subtitles: List[String],
    preview_is_videos: List[Bool],
    mut gallery_scroll_y: Float32,
) raises -> Int32:
    var cw = two_col_w(content_w)
    var label_w = ctx.theme.font_size_pt * 7
    if label_w < 178:
        label_w = 178
    var val_w = value_w(cw - 28, label_w)
    var settings_h = ctx.theme.row_height * 11 + ctx.theme.font_size_pt * 4
    ctx.layout_row(row2(cw, cw), settings_h)
    begin_form_panel(ctx, String("SAMPLING SETTINGS"), String("Serenity sampling cadence and sampler"))
    _ = edit_row(ctx, label_w, val_w, String("Sample Dir"), String("sample_output_dir"), cfg.sample_output_dir, sample_output_edit)
    _ = drag_row(ctx, label_w, 150, String("Sample After"), String("sample_after"), cfg.sample_after, 10.0)
    _ = drag_row(ctx, label_w, 150, String("Skip First"), String("sample_skip_first"), cfg.sample_skip_first, 1.0)
    _ = drag_row(ctx, label_w, 150, String("Steps"), String("sample_steps"), cfg.sample_steps, 1.0)
    _ = drag_row(ctx, label_w, 150, String("CFG"), String("sample_cfg"), cfg.sample_cfg, 0.1)
    _ = select_string_row(ctx, label_w, val_w, String("Sampler"), String("sample_sampler"), cfg.sample_sampler_options, cfg.sample_sampler, cfg.select_open_id)
    _ = toggle_row(ctx, label_w, val_w, String("SerenityBoard"), String("Send samples"), cfg.samples_to_tensorboard)
    _ = toggle_row(ctx, label_w, val_w, String("Non-EMA"), String("Use non-EMA model"), cfg.non_ema_sampling)
    var video_count = _video_count(preview_is_videos)
    field_row(ctx, label_w, val_w, String("Preview Count"), String(len(preview_textures)) + String(" indexed"))
    field_row(ctx, label_w, val_w, String("Videos"), String(video_count))
    end_form_panel(ctx)

    begin_form_panel(ctx, String("SAMPLE PROMPTS"), String("Prompts shown in the Rust trainer preview panel"))
    for i in range(len(cfg.samples)):
        var sample = cfg.samples[i].copy()
        field_row(ctx, label_w, val_w, String("Prompt ") + String(i + 1), sample.prompt.copy())
        field_row(ctx, label_w, val_w, String("Negative ") + String(i + 1), sample.negative_prompt.copy())
        field_row(ctx, label_w, val_w, String("Seed ") + String(i + 1), String(sample.seed))
    end_form_panel(ctx)

    var columns: Int32 = content_w // 360
    if columns < 2:
        columns = 2
    if columns > 8:
        columns = 8
    var card_w = (content_w - (columns - 1) * ctx.theme.spacing) // columns
    if card_w < 220:
        card_w = 220
    if card_w > 440:
        card_w = 440
    var card_h = card_w * 3 // 4 + ctx.theme.row_height * 2
    if card_h > 460:
        card_h = 460
    var visible_rows: Int32 = 3
    if content_w >= 1800:
        visible_rows = 4
    if content_w >= 3000:
        visible_rows = 5
    var viewport_h = card_h * visible_rows + ctx.theme.spacing * (visible_rows + 1)
    ctx.layout_row(row1(content_w), viewport_h)
    return media_virtual_grid(
        ctx,
        String("sample_virtual_grid"),
        viewport_h,
        columns,
        card_w,
        card_h,
        Int32(len(preview_textures)),
        gallery_scroll_y,
        preview_textures,
        preview_widths,
        preview_heights,
        preview_titles,
        preview_subtitles,
        preview_is_videos,
    )
