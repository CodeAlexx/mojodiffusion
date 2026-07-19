"""Dataset tab for the native Serenity Trainer trainer UI."""

from mojoui.core.context import Context
from mojoui.core.textedit import TextEditState
from mojoui.widgets.form import (
    begin_form_panel,
    end_form_panel,
    edit_row,
    field_row,
    select_string_row,
    slider_row,
    toggle_row,
)
from mojoui.widgets.image import media_virtual_grid
from serenity_trainer.ui.TrainerConfigModel import TrainerUIConfig
from serenity_trainer.ui.UITabCommon import row1, row2, two_col_w, value_w


def _video_count(flags: List[Bool]) -> Int:
    var count = 0
    for i in range(len(flags)):
        if flags[i]:
            count += 1
    return count


def render_dataset_tab(
    mut ctx: Context,
    mut cfg: TrainerUIConfig,
    content_w: Int32,
    mut dataset_path_edit: TextEditState,
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
    var settings_h = ctx.theme.row_height * 8 + ctx.theme.font_size_pt * 4
    ctx.layout_row(row2(cw, cw), settings_h)
    begin_form_panel(ctx, String("DATASET"), String("Image path, concept file, and cache policy"))
    _ = edit_row(ctx, label_w, val_w, String("Dataset Path"), String("dataset_path"), cfg.dataset_path, dataset_path_edit)
    field_row(ctx, label_w, val_w, String("Concept File"), cfg.concept_file_name.copy())
    _ = select_string_row(ctx, label_w, val_w, String("Resolution"), String("dataset_resolution"), cfg.resolution_options, cfg.resolution, cfg.select_open_id)
    # Capability honesty (UI wave 2): no runner consumes this toggle —
    # bucketing is baked into the prepared caches. Snapshot-only.
    _ = toggle_row(ctx, label_w, val_w, String("Aspect Buckets [not wired]"), String("Enabled"), cfg.aspect_ratio_bucketing)
    _ = toggle_row(ctx, label_w, val_w, String("Latent Caching"), String("Enabled"), cfg.latent_caching)
    _ = toggle_row(ctx, label_w, val_w, String("Clear Cache"), String("Before training"), cfg.clear_cache_before_training)
    end_form_panel(ctx)

    begin_form_panel(ctx, String("PREVIEW & CAPTIONS"), String("Rust snapshot style dataset summary"))
    var video_count = _video_count(preview_is_videos)
    var image_count = len(preview_textures) - video_count
    field_row(ctx, label_w, val_w, String("Media"), String(image_count) + String(" images / ") + String(video_count) + String(" videos"))
    field_row(ctx, label_w, val_w, String("Virtual Grid"), String(len(preview_textures)) + String(" files indexed"))
    field_row(ctx, label_w, val_w, String("Buckets"), String("704x1024, 1024x1024, 1024x704"))
    field_row(ctx, label_w, val_w, String("Caption Source"), String("sidecar text files"))
    _ = slider_row(ctx, label_w, val_w, String("Caption Dropout"), String("caption_dropout"), cfg.caption_dropout, 0.0, 0.5)
    var trigger = String("")
    if len(cfg.concepts) > 0:
        trigger = cfg.concepts[0].trigger.copy()
    field_row(ctx, label_w, val_w, String("Trigger"), trigger)
    field_row(ctx, label_w, val_w, String("Cache Format"), String("safetensors latent/cap"))
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
        String("dataset_virtual_grid"),
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
