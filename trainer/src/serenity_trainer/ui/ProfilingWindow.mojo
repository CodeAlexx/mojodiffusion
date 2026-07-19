"""Profiling window surface."""

from mojoui.core.context import Context
from mojoui.widgets.form import begin_form_panel, end_form_panel, field_row, toggle_row
from serenity_trainer.ui.AuxScreenCommon import aux_label_w, aux_panel_h
from serenity_trainer.ui.TrainerConfigModel import TrainerUIConfig
from serenity_trainer.ui.UITabCommon import row2, two_col_w, value_w


def render_profiling_window(mut ctx: Context, cfg: TrainerUIConfig, content_w: Int32) raises:
    var cw = two_col_w(content_w)
    var label_w = aux_label_w(ctx, cw)
    var val_w = value_w(cw - ctx.theme.padding * 2, label_w)
    ctx.layout_row(row2(cw, cw), aux_panel_h(ctx, 6))
    begin_form_panel(ctx, String("PROFILING"), String("Torch profiler and memory recorder switches"))
    var profiler = False
    var memory_trace = False
    _ = toggle_row(ctx, label_w, val_w, String("Profiler"), String("Enabled"), profiler)
    _ = toggle_row(ctx, label_w, val_w, String("Memory Trace"), String("Enabled"), memory_trace)
    field_row(ctx, label_w, val_w, String("Profile Step"), String("current global_step"))
    field_row(ctx, label_w, val_w, String("Output"), cfg.workspace_dir.copy())
    field_row(ctx, label_w, val_w, String("Trace Format"), String("json / pickle"))
    field_row(ctx, label_w, val_w, String("Status"), String("idle"))
    end_form_panel(ctx)

    begin_form_panel(ctx, String("GPU HEALTH"), String("Runtime diagnostics"))
    field_row(ctx, label_w, val_w, String("VRAM"), String("tracked by status rail"))
    field_row(ctx, label_w, val_w, String("NaN Guard"), String("loss validation"))
    field_row(ctx, label_w, val_w, String("Grad Norm"), String("pre-clip value"))
    field_row(ctx, label_w, val_w, String("Divergence"), String("multi-gpu warning"))
    field_row(ctx, label_w, val_w, String("SerenityBoard"), cfg.tensorboard_port.copy())
    field_row(ctx, label_w, val_w, String("Debug Dir"), cfg.debug_dir.copy())
    end_form_panel(ctx)
