"""Top action bar for the native Serenity Trainer trainer UI."""

from mojoui.core.context import Context
from mojoui.core.textedit import TextEditState
from mojoui.widgets.app_shell import action_button
from mojoui.widgets.basic import label
from mojoui.widgets.text_edit import text_edit
from serenity_trainer.ui.TrainerConfigModel import (
    TrainerUIConfig,
    trainer_ui_ignored_lever_summary,
    trainer_ui_validate,
)
from serenity_trainer.ui.TrainerRuntimeBridge import (
    TrainerUIRuntime,
    trainer_ui_cancel,
    trainer_ui_pause,
    trainer_ui_resume,
    trainer_ui_sample_now,
    trainer_ui_save_checkpoint_now,
    trainer_ui_submit_current,
)
from serenity_trainer.ui.UITabCommon import row4


def render_top_bar(
    mut ctx: Context,
    mut cfg: TrainerUIConfig,
    mut rt: TrainerUIRuntime,
    content_w: Int32,
    mut run_name_edit: TextEditState,
) raises:
    var section_w = ctx.theme.font_size_pt * 10
    var run_w = ctx.theme.font_size_pt * 10
    var action_w = ctx.theme.font_size_pt * 8
    var pause_w = ctx.theme.font_size_pt * 6
    var sample_w = ctx.theme.font_size_pt * 8
    var save_w = ctx.theme.font_size_pt * 10
    if section_w < 250:
        section_w = 250
    if run_w < 280:
        run_w = 280
    if action_w < 230:
        action_w = 230
    if pause_w < 132:
        pause_w = 132
    if sample_w < 170:
        sample_w = 170
    if save_w < 240:
        save_w = 240
    var spacer = content_w - section_w - run_w - action_w - 24
    if spacer < 24:
        spacer = 24

    ctx.layout_row(row4(section_w, run_w, spacer, action_w), ctx.theme.row_height)
    label(ctx, String("SECTION - ") + cfg.section_label())
    _ = text_edit(ctx, String("run_name"), cfg.run_name, run_name_edit)
    label(ctx, String("Live target - ") + rt.backend_label.copy())
    if rt.has_running:
        if action_button(ctx, String("stop_top"), String("Stop training"), True):
            trainer_ui_cancel(rt)
    else:
        if action_button(ctx, String("start"), String("Start training"), True):
            _ = trainer_ui_submit_current(cfg, rt)

    var validation_w = content_w - pause_w - sample_w - save_w - 24
    if validation_w < 360:
        validation_w = 360
    ctx.layout_row(row4(validation_w, pause_w, sample_w, save_w), ctx.theme.row_height)
    # Capability-table honesty banner: any non-default widget the selected
    # model's runner does not consume is named HERE, before launch, every
    # frame — no widget may silently lie.
    var ignored = trainer_ui_ignored_lever_summary(cfg)
    if ignored.byte_length() > 0:
        label(ctx, String("VALIDATION - ") + trainer_ui_validate(cfg) + String("  |  WARNING ") + ignored)
    else:
        label(ctx, String("VALIDATION - ") + trainer_ui_validate(cfg))
    # Pause/Sample-now append to the command file, but NO trainer reads it
    # yet (wave-3 command-file protocol). Marked so the buttons cannot lie:
    # "Paused" is UI state only; the trainer keeps running.
    if rt.has_running:
        if rt.paused:
            if action_button(ctx, String("resume"), String("Resume [not wired]"), False):
                _ = trainer_ui_resume(rt)
        else:
            if action_button(ctx, String("pause"), String("Pause [not wired]"), False):
                _ = trainer_ui_pause(rt)
    else:
        label(ctx, String(""))
    if action_button(ctx, String("sample_now"), String("Sample [not wired]"), False):
        _ = trainer_ui_sample_now(rt)
    if action_button(ctx, String("save_now"), String("Save checkpoint"), False):
        _ = trainer_ui_save_checkpoint_now(cfg, rt)
