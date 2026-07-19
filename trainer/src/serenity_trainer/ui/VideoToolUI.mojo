"""Video dataset tool surface."""

from mojoui.core.context import Context
from mojoui.widgets.form import begin_form_panel, end_form_panel, field_row, toggle_row
from serenity_trainer.ui.AuxScreenCommon import aux_label_w, aux_panel_h
from serenity_trainer.ui.TrainerConfigModel import TrainerUIConfig
from serenity_trainer.ui.UITabCommon import row2, two_col_w, value_w


def render_video_tool_ui(mut ctx: Context, mut cfg: TrainerUIConfig, content_w: Int32) raises:
    var cw = two_col_w(content_w)
    var label_w = aux_label_w(ctx, cw)
    var val_w = value_w(cw - ctx.theme.padding * 2, label_w)
    ctx.layout_row(row2(cw, cw), aux_panel_h(ctx, 7))
    begin_form_panel(ctx, String("VIDEO TOOL"), String("Video extraction and preview"))
    var make_thumbnails = True
    var caption_videos = True
    field_row(ctx, label_w, val_w, String("Source Dir"), cfg.dataset_path.copy())
    field_row(ctx, label_w, val_w, String("Formats"), String("mp4, mov, mkv, webm"))
    field_row(ctx, label_w, val_w, String("Frame Rate"), String("1 fps"))
    field_row(ctx, label_w, val_w, String("Max Frames"), String("auto"))
    _ = toggle_row(ctx, label_w, val_w, String("Make Thumbnails"), String("Enabled"), make_thumbnails)
    _ = toggle_row(ctx, label_w, val_w, String("Caption Videos"), String("Enabled"), caption_videos)
    field_row(ctx, label_w, val_w, String("Player"), String("MojoUI external video open"))
    end_form_panel(ctx)

    begin_form_panel(ctx, String("MOVIE DATASET"), String("Clip metadata for training"))
    field_row(ctx, label_w, val_w, String("Clip Length"), String("frames / seconds"))
    field_row(ctx, label_w, val_w, String("Stride"), String("frame interval"))
    field_row(ctx, label_w, val_w, String("Resolution"), cfg.resolution.copy())
    field_row(ctx, label_w, val_w, String("Audio"), String("ignored for image LoRA"))
    field_row(ctx, label_w, val_w, String("Thumbnail Badge"), String("VIDEO"))
    field_row(ctx, label_w, val_w, String("Preview"), String("click to play source video"))
    field_row(ctx, label_w, val_w, String("Status"), String("ready"))
    end_form_panel(ctx)
