"""Cloud tab for the native Serenity Trainer trainer UI."""

from mojoui.core.context import Context
from mojoui.widgets.form import begin_form_panel, end_form_panel, field_row, select_index_row, toggle_row
from serenity_trainer.ui.TrainerConfigModel import TrainerUIConfig
from serenity_trainer.ui.UITabCommon import row2, two_col_w, value_w


def render_cloud_tab(mut ctx: Context, mut cfg: TrainerUIConfig, content_w: Int32) raises:
    var cw = two_col_w(content_w)
    var label_w = ctx.theme.font_size_pt * 8
    if label_w < 210:
        label_w = 210
    var val_w = value_w(cw - 28, label_w)
    var panel_h = ctx.theme.row_height * 7 + ctx.theme.font_size_pt * 4
    ctx.layout_row(row2(cw, cw), panel_h)
    begin_form_panel(ctx, String("CLOUD"), String("Serenity cloud target: RunPod/Linux SSH"))
    _ = select_index_row(ctx, label_w, val_w, String("Cloud Type"), String("cloud_type"), cfg.cloud_type_options, cfg.cloud_type_index, cfg.select_open_id)
    field_row(ctx, label_w, val_w, String("Host"), cfg.cloud_host.copy())
    field_row(ctx, label_w, val_w, String("Port"), cfg.cloud_port.copy())
    field_row(ctx, label_w, val_w, String("User"), cfg.cloud_user.copy())
    field_row(ctx, label_w, val_w, String("Remote Dir"), cfg.cloud_workspace_dir.copy())
    _ = toggle_row(ctx, label_w, val_w, String("Delete Work Dir"), String("After run"), cfg.cloud_delete_workspace)
    end_form_panel(ctx)

    begin_form_panel(ctx, String("SYNC"), String("File sync and post-run behavior"))
    field_row(ctx, label_w, val_w, String("Upload"), String("config, concepts, cache metadata"))
    field_row(ctx, label_w, val_w, String("Download"), String("samples, checkpoints, logs"))
    field_row(ctx, label_w, val_w, String("File Sync"), String("Native SCP / Fabric SFTP"))
    field_row(ctx, label_w, val_w, String("On Finish"), String("None / Stop / Delete"))
    field_row(ctx, label_w, val_w, String("Status"), String("UI surface ready"))
    end_form_panel(ctx)
