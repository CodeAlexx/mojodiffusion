"""Native Serenity Mojo trainer UI.

Rust-snapshot layout: fixed left nav, compact top action bar, dense center
    panels, and a persistent right live-status rail. Serenity tab/config names are
    the source of truth; the runtime bridge targets the selected live trainer.
"""

from std.memory import UnsafePointer

from mojoui.app.state import retrieve_user_state, store_user_state
from mojoui.core.context import Context
from mojoui.core.textedit import TextEditState
from mojoui.core.types import Vec2, Rect, Color
from mojoui.render.backend import Backend
from mojoui.render.command_renderer import render_context_commands
from mojoui.theme.trainer_theme import apply_serenity_trainer_theme
from mojoui.widgets.app_shell import (
    TrainerShellMetrics,
    apply_shell_density,
    draw_shell_background,
    nav_row,
    trainer_shell_metrics,
)
from mojoui.widgets.basic import label, separator
from mojoui.widgets.form import begin_form_panel, console_line, end_form_panel, field_row, pill
from mojoui.widgets.progress_bar import progress_bar
from mojoui.widgets.scroll_area import begin_scroll_area, end_scroll_area
from serenity_trainer.ui.CloudTab import render_cloud_tab
from serenity_trainer.ui.CaptionerTab import CaptionerScreenState, render_captioner_tab
from serenity_trainer.ui.ConceptTab import render_concepts_tab
from serenity_trainer.ui.DatasetTab import render_dataset_tab
from serenity_trainer.ui.GeneralTab import render_general_tab
from serenity_trainer.ui.LoraTab import render_lora_tab
from serenity_trainer.ui.ModelTab import render_model_tab
from serenity_trainer.ui.SamplingTab import render_sampling_tab
from serenity_trainer.ui.TrainerMediaGallery import (
    TrainerMediaGalleryState,
    ensure_visible_dataset_textures,
    ensure_visible_sample_textures,
    load_preview_textures,
    open_dataset_preview,
    open_sample_preview,
    render_media_lightbox,
)
from serenity_trainer.ui.TopBar import render_top_bar
from serenity_trainer.ui.TrainingTab import render_training_tab
from serenity_trainer.ui.TrainerConfigModel import (
    TrainerUIConfig,
    trainer_ui_load_config_snapshot,
    UI_SECTION_BACKUP,
    UI_SECTION_CAPTIONER,
    UI_SECTION_CLOUD,
    UI_SECTION_CONCEPTS,
    UI_SECTION_DATASET,
    UI_SECTION_GENERAL,
    UI_SECTION_LOGS,
    UI_SECTION_LORA,
    UI_SECTION_MODEL,
    UI_SECTION_RUNS,
    UI_SECTION_SAMPLING,
    UI_SECTION_TRAINING,
    trainer_ui_config_json_snapshot,
    trainer_ui_validate,
)
from serenity_trainer.ui.TrainerRuntimeBridge import (
    TrainerUIRuntime,
    trainer_ui_eta_label,
    trainer_ui_progress_fraction,
    trainer_ui_tick_and_apply,
)
from serenity_trainer.ui.UITabCommon import row1, row2, row3, row4, row5, value_w


comptime INIT_W: Float32 = 1480.0
comptime INIT_H: Float32 = 920.0


struct TrainUIAppState(Movable):
    var ctx: Context
    var cfg: TrainerUIConfig
    var runtime: TrainerUIRuntime
    var font_id: UInt32
    var win_w: Float32
    var win_h: Float32
    var metrics: TrainerShellMetrics
    var main_scroll_y: Float32
    var status_scroll_y: Float32
    var media: TrainerMediaGalleryState
    var captioner: CaptionerScreenState

    var run_name_edit: TextEditState
    var workspace_edit: TextEditState
    var cache_edit: TextEditState
    var base_model_edit: TextEditState
    var output_dir_edit: TextEditState
    var dataset_path_edit: TextEditState
    var sample_output_edit: TextEditState
    var captioner_folder_edit: TextEditState
    var captioner_custom_model_edit: TextEditState
    var captioner_prompt_edit: TextEditState

    def __init__(out self):
        self.ctx = Context()
        self.cfg = TrainerUIConfig()
        # Persist user-set recipe values across sessions (max_train_steps etc.
        # otherwise reset to struct defaults every launch - the "3000 steps
        # hardcoded" report, 2026-07-03). Fail-soft when no snapshot exists.
        _ = trainer_ui_load_config_snapshot(self.cfg)
        self.runtime = TrainerUIRuntime()
        self.font_id = 0
        self.win_w = INIT_W
        self.win_h = INIT_H
        self.metrics = trainer_shell_metrics(INIT_W, INIT_H)
        self.main_scroll_y = 0.0
        self.status_scroll_y = 0.0
        self.media = TrainerMediaGalleryState()
        self.captioner = CaptionerScreenState()
        self.run_name_edit = TextEditState(single_line=True)
        self.workspace_edit = TextEditState(single_line=True)
        self.cache_edit = TextEditState(single_line=True)
        self.base_model_edit = TextEditState(single_line=True)
        self.output_dir_edit = TextEditState(single_line=True)
        self.dataset_path_edit = TextEditState(single_line=True)
        self.sample_output_edit = TextEditState(single_line=True)
        self.captioner_folder_edit = TextEditState(single_line=True)
        self.captioner_custom_model_edit = TextEditState(single_line=True)
        self.captioner_prompt_edit = TextEditState(single_line=True)


def _initial_window_size() -> Vec2:
    var display = Backend.display_size()
    if display.x <= 0.0 or display.y <= 0.0:
        return Vec2(INIT_W, INIT_H)
    var w = display.x * 0.92
    var h = display.y * 0.90
    if w < INIT_W:
        w = INIT_W
    if h < INIT_H:
        h = INIT_H
    return Vec2(w, h)


def _sync_window(mut s: TrainUIAppState):
    var win = Backend.window_size()
    if win.x <= 0.0 or win.y <= 0.0:
        win = Vec2(INIT_W, INIT_H)
    s.win_w = win.x
    s.win_h = win.y
    s.metrics = trainer_shell_metrics(win.x, win.y)
    apply_serenity_trainer_theme(s.ctx)
    s.ctx.set_default_font(s.font_id)
    apply_shell_density(s.ctx, s.metrics)


def _nav(mut s: TrainUIAppState, section: Int32, text: String):
    var m = s.metrics.copy()
    var content_w = m.nav_w - m.pad * 2
    s.ctx.layout_row(row3(m.pad, content_w, m.pad), m.row_h)
    label(s.ctx, String(""))
    if nav_row(s.ctx, String("nav_") + text.copy(), text, s.cfg.section_index == section):
        if s.cfg.section_index != section:
            s.cfg.section_index = section
            s.main_scroll_y = 0.0
    label(s.ctx, String(""))


def _sidebar(mut s: TrainUIAppState):
    var m = s.metrics.copy()
    var content_w = m.nav_w - m.pad * 2
    s.ctx.layout_row(row3(m.pad, content_w, m.pad), m.row_h + 18)
    label(s.ctx, String(""))
    label(s.ctx, String("Serenity"))
    label(s.ctx, String(""))
    s.ctx.layout_row(row3(m.pad, content_w, m.pad), 22)
    label(s.ctx, String(""))
    label(s.ctx, String("Native trainer UI"))
    label(s.ctx, String(""))
    s.ctx.layout_row(row3(m.pad, content_w, m.pad), 16)
    label(s.ctx, String(""))
    separator(s.ctx)
    label(s.ctx, String(""))
    s.ctx.layout_row(row3(m.pad, content_w, m.pad), 22)
    label(s.ctx, String(""))
    label(s.ctx, String("CONFIGURE"))
    label(s.ctx, String(""))

    _nav(s, UI_SECTION_GENERAL, String("General"))
    _nav(s, UI_SECTION_MODEL, String("Model"))
    _nav(s, UI_SECTION_LORA, String("LoRA / OFT"))
    _nav(s, UI_SECTION_DATASET, String("Dataset"))
    _nav(s, UI_SECTION_CAPTIONER, String("Captioner"))
    _nav(s, UI_SECTION_CONCEPTS, String("Validations"))
    _nav(s, UI_SECTION_TRAINING, String("Training"))
    _nav(s, UI_SECTION_SAMPLING, String("Sampling"))
    _nav(s, UI_SECTION_BACKUP, String("Backup"))
    _nav(s, UI_SECTION_CLOUD, String("Cloud"))
    _nav(s, UI_SECTION_RUNS, String("Runs"))
    _nav(s, UI_SECTION_LOGS, String("Logs"))

    s.ctx.layout_row(row3(m.pad, content_w, m.pad), 28)
    label(s.ctx, String(""))
    separator(s.ctx)
    label(s.ctx, String(""))
    s.ctx.layout_row(row3(m.pad, content_w, m.pad), 22)
    label(s.ctx, String(""))
    label(s.ctx, String("PROJECT"))
    label(s.ctx, String(""))
    s.ctx.layout_row(row3(m.pad, content_w, m.pad), m.row_h)
    label(s.ctx, String(""))
    label(s.ctx, s.cfg.run_name.copy())
    label(s.ctx, String(""))


def _backup_tab(mut s: TrainUIAppState, content_w: Int32):
    var cw = (content_w - 16) // 2
    if cw < 320:
        cw = 320
    var label_w = s.ctx.theme.font_size_pt * 8
    if label_w < 210:
        label_w = 210
    var val_w = value_w(cw - 28, label_w)
    var panel_h = s.ctx.theme.row_height * 8 + s.ctx.theme.font_size_pt * 4
    s.ctx.layout_row(row2(cw, cw), panel_h)
    begin_form_panel(s.ctx, String("BACKUP"), String("Serenity backup and save cadence"))
    field_row(s.ctx, label_w, val_w, String("Backup After"), String(s.cfg.backup_after))
    field_row(s.ctx, label_w, val_w, String("Rolling Backup"), String(s.cfg.rolling_backup))
    field_row(s.ctx, label_w, val_w, String("Rolling Count"), String(s.cfg.rolling_backup_count))
    field_row(s.ctx, label_w, val_w, String("Before Save"), String(s.cfg.backup_before_save))
    field_row(s.ctx, label_w, val_w, String("Save Every"), String(s.cfg.save_every))
    field_row(s.ctx, label_w, val_w, String("Save Prefix"), s.cfg.save_filename_prefix.copy())
    end_form_panel(s.ctx)

    begin_form_panel(s.ctx, String("PRESET SNAPSHOT"), String("Current config snapshot"))
    field_row(s.ctx, label_w, val_w, String("Schema"), String("serenity.trainer_ui.v1"))
    field_row(s.ctx, label_w, val_w, String("Backend"), s.cfg.backend_target.copy())
    field_row(s.ctx, label_w, val_w, String("Model"), s.cfg.architecture_label())
    field_row(s.ctx, label_w, val_w, String("Validation"), trainer_ui_validate(s.cfg))
    field_row(s.ctx, label_w, val_w, String("Config File"), String("target/serenity_trainer_ui_config.json"))
    end_form_panel(s.ctx)


def _runs_tab(mut s: TrainUIAppState, content_w: Int32):
    var label_w: Int32 = 120
    var val_w = value_w(content_w - 28, label_w)
    s.ctx.layout_row(row1(content_w), 520)
    begin_form_panel(s.ctx, String("RUNS"), String("Submitted trainer runs"))
    if len(s.runtime.runs) == 0:
        field_row(s.ctx, label_w, val_w, String("idle"), String("No runs yet"))
    for i in range(len(s.runtime.runs)):
        field_row(s.ctx, label_w, val_w, String("#") + String(i + 1), s.runtime.runs[i].copy())
    end_form_panel(s.ctx)


def _logs_tab(mut s: TrainUIAppState, content_w: Int32):
    var cw = (content_w - 16) // 2
    if cw < 360:
        cw = 360
    var label_w = s.ctx.theme.font_size_pt * 7
    if label_w < 160:
        label_w = 160
    var val_w = value_w(cw - 28, label_w)
    var panel_h = s.ctx.theme.row_height * 14 + s.ctx.theme.font_size_pt * 4
    s.ctx.layout_row(row2(cw, cw), panel_h)
    begin_form_panel(s.ctx, String("OUTPUT EVENTS"), String("Recent UI and trainer bridge events"))
    var n = len(s.runtime.logs)
    if n == 0:
        field_row(s.ctx, label_w, val_w, String("ready"), String("No events yet"))
    var start = 0
    if n > 10:
        start = n - 10
    for i in range(start, n):
        field_row(s.ctx, label_w, val_w, String("#") + String(i + 1), s.runtime.logs[i].copy())
    end_form_panel(s.ctx)

    begin_form_panel(s.ctx, String("ARTIFACTS"), String("Samples, checkpoints, and progress source"))
    field_row(s.ctx, label_w, val_w, String("Progress File"), s.runtime.progress_file_path.copy())
    field_row(s.ctx, label_w, val_w, String("Command File"), s.runtime.command_file_path.copy())
    if s.runtime.using_callback_progress:
        field_row(s.ctx, label_w, val_w, String("Stats Source"), String("Serenity callbacks"))
    elif s.runtime.using_live_progress:
        field_row(s.ctx, label_w, val_w, String("Stats Source"), String("Trainer progress file"))
    else:
        field_row(s.ctx, label_w, val_w, String("Stats Source"), String("Waiting for bridge"))
    field_row(s.ctx, label_w, val_w, String("Samples"), String(len(s.runtime.samples)))
    field_row(s.ctx, label_w, val_w, String("Checkpoints"), String(len(s.runtime.checkpoints)))
    var sample_start = 0
    if len(s.runtime.samples) > 4:
        sample_start = len(s.runtime.samples) - 4
    for i in range(sample_start, len(s.runtime.samples)):
        field_row(s.ctx, label_w, val_w, String("Sample ") + String(i + 1), s.runtime.samples[i].copy())
    var ckpt_start = 0
    if len(s.runtime.checkpoints) > 4:
        ckpt_start = len(s.runtime.checkpoints) - 4
    for i in range(ckpt_start, len(s.runtime.checkpoints)):
        field_row(s.ctx, label_w, val_w, String("Ckpt ") + String(i + 1), s.runtime.checkpoints[i].copy())
    end_form_panel(s.ctx)


def _main_section(mut s: TrainUIAppState, content_w: Int32) raises:
    if s.cfg.section_index == UI_SECTION_GENERAL:
        render_general_tab(s.ctx, s.cfg, content_w, s.workspace_edit, s.cache_edit)
    elif s.cfg.section_index == UI_SECTION_MODEL:
        render_model_tab(s.ctx, s.cfg, content_w, s.base_model_edit, s.output_dir_edit)
    elif s.cfg.section_index == UI_SECTION_LORA:
        render_lora_tab(s.ctx, s.cfg, content_w)
    elif s.cfg.section_index == UI_SECTION_DATASET:
        ensure_visible_dataset_textures(s.media, s.ctx, content_w)
        var clicked = render_dataset_tab(
            s.ctx,
            s.cfg,
            content_w,
            s.dataset_path_edit,
            s.media.dataset_preview_textures,
            s.media.dataset_preview_widths,
            s.media.dataset_preview_heights,
            s.media.dataset_preview_titles,
            s.media.dataset_preview_subtitles,
            s.media.dataset_preview_is_videos,
            s.media.dataset_gallery_scroll_y,
        )
        open_dataset_preview(s.media, clicked)
    elif s.cfg.section_index == UI_SECTION_CAPTIONER:
        render_captioner_tab(
            s.ctx,
            s.cfg,
            s.captioner,
            content_w,
            s.captioner_folder_edit,
            s.captioner_custom_model_edit,
            s.captioner_prompt_edit,
        )
    elif s.cfg.section_index == UI_SECTION_CONCEPTS:
        render_concepts_tab(s.ctx, s.cfg, content_w)
    elif s.cfg.section_index == UI_SECTION_TRAINING:
        render_training_tab(s.ctx, s.cfg, content_w)
    elif s.cfg.section_index == UI_SECTION_SAMPLING:
        ensure_visible_sample_textures(s.media, s.ctx, content_w)
        var clicked = render_sampling_tab(
            s.ctx,
            s.cfg,
            content_w,
            s.sample_output_edit,
            s.media.sample_preview_textures,
            s.media.sample_preview_widths,
            s.media.sample_preview_heights,
            s.media.sample_preview_titles,
            s.media.sample_preview_subtitles,
            s.media.sample_preview_is_videos,
            s.media.sample_gallery_scroll_y,
        )
        open_sample_preview(s.media, clicked)
    elif s.cfg.section_index == UI_SECTION_BACKUP:
        _backup_tab(s, content_w)
    elif s.cfg.section_index == UI_SECTION_CLOUD:
        render_cloud_tab(s.ctx, s.cfg, content_w)
    elif s.cfg.section_index == UI_SECTION_RUNS:
        _runs_tab(s, content_w)
    else:
        _logs_tab(s, content_w)


def _main_panel(mut s: TrainUIAppState) raises:
    var m = s.metrics.copy()
    var content_w = m.main_w - m.pad * 2
    if content_w < 620:
        content_w = 620
    var top_h = m.row_h * 2 + 12
    var main_h = Int32(s.win_h) - top_h - 58
    if main_h < 560:
        main_h = 560

    s.ctx.layout_row(row3(m.pad, content_w, m.pad), 14)
    label(s.ctx, String(""))
    label(s.ctx, String(""))
    label(s.ctx, String(""))

    s.ctx.layout_row(row3(m.pad, content_w, m.pad), top_h)
    label(s.ctx, String(""))
    s.ctx.begin_column()
    render_top_bar(s.ctx, s.cfg, s.runtime, content_w, s.run_name_edit)
    s.ctx.end_column()
    label(s.ctx, String(""))

    s.ctx.layout_row(row3(m.pad, content_w, m.pad), 10)
    label(s.ctx, String(""))
    separator(s.ctx)
    label(s.ctx, String(""))

    s.ctx.layout_row(row3(m.pad, content_w, m.pad), main_h)
    label(s.ctx, String(""))
    if begin_scroll_area(s.ctx, String("main_parameters"), main_h, s.main_scroll_y):
        pass
    _main_section(s, content_w)
    end_scroll_area(s.ctx)
    label(s.ctx, String(""))


def _status_rail(mut s: TrainUIAppState):
    var m = s.metrics.copy()
    var content_w = m.status_w - m.pad * 2
    var saved_font_size = s.ctx.theme.font_size_pt
    var saved_row_height = s.ctx.theme.row_height
    if s.ctx.theme.font_size_pt > 34:
        s.ctx.theme.font_size_pt = 34
    if s.ctx.theme.row_height > 60:
        s.ctx.theme.row_height = 60
    var label_w = s.ctx.theme.font_size_pt * 5
    if label_w < 126:
        label_w = 126
    var val_w = content_w - label_w - 8
    if val_w < 120:
        val_w = 120
    var rail_row_h = s.ctx.theme.row_height
    var stats_h = rail_row_h * 11 + s.ctx.theme.spacing * 10
    var artifact_h = rail_row_h * 3 + s.ctx.theme.spacing * 2
    var hardware_h = rail_row_h * 7 + s.ctx.theme.spacing * 6
    s.ctx.layout_row(row1(m.status_w), Int32(s.win_h))
    if begin_scroll_area(s.ctx, String("status_rail"), Int32(s.win_h), s.status_scroll_y):
        pass
    s.ctx.layout_row(row3(m.pad, content_w, m.pad), 24)
    label(s.ctx, String(""))
    label(s.ctx, String(""))
    label(s.ctx, String(""))
    s.ctx.layout_row(row3(m.pad, content_w, m.pad), m.row_h)
    label(s.ctx, String(""))
    label(s.ctx, String("LIVE STATUS"))
    label(s.ctx, String(""))
    s.ctx.layout_row(row3(m.pad, content_w, m.pad), 24)
    label(s.ctx, String(""))
    if s.runtime.has_running:
        if s.runtime.paused:
            pill(s.ctx, String("PAUSED"), 3)
        else:
            pill(s.ctx, String("RUNNING"), 1)
    else:
        pill(s.ctx, String("IDLE"), 0)
    label(s.ctx, String(""))
    s.ctx.layout_row(row3(m.pad, content_w, m.pad), 18)
    label(s.ctx, String(""))
    progress_bar(s.ctx, trainer_ui_progress_fraction(s.runtime))
    label(s.ctx, String(""))
    s.ctx.layout_row(row3(m.pad, content_w, m.pad), 10)
    label(s.ctx, String(""))
    separator(s.ctx)
    label(s.ctx, String(""))

    s.ctx.layout_row(row3(m.pad, content_w, m.pad), stats_h)
    label(s.ctx, String(""))
    s.ctx.begin_column()
    field_row(s.ctx, label_w, val_w, String("Step"), String(s.runtime.live.step) + String(" / ") + String(s.runtime.live.total_steps))
    field_row(s.ctx, label_w, val_w, String("Epoch"), String(s.runtime.live.epoch) + String(" / ") + String(s.runtime.live.total_epochs))
    field_row(s.ctx, label_w, val_w, String("Loss"), String(s.runtime.live.loss))
    field_row(s.ctx, label_w, val_w, String("Smooth"), String(s.runtime.live.smooth_loss))
    field_row(s.ctx, label_w, val_w, String("Grad"), String(s.runtime.live.grad_norm))
    field_row(s.ctx, label_w, val_w, String("LR"), String(s.runtime.live.learning_rate))
    field_row(s.ctx, label_w, val_w, String("Speed"), String(s.runtime.live.speed_it_s) + String(" s/step"))
    field_row(s.ctx, label_w, val_w, String("ETA"), trainer_ui_eta_label(s.runtime))
    field_row(s.ctx, label_w, val_w, String("Status"), s.runtime.status_text.copy())
    field_row(s.ctx, label_w, val_w, String("Command"), s.runtime.last_command.copy())
    if s.runtime.using_callback_progress:
        field_row(s.ctx, label_w, val_w, String("Source"), String("Serenity callbacks"))
    elif s.runtime.using_live_progress:
        field_row(s.ctx, label_w, val_w, String("Source"), String("Trainer progress file"))
    else:
        field_row(s.ctx, label_w, val_w, String("Source"), String("Waiting"))
    s.ctx.end_column()
    label(s.ctx, String(""))

    s.ctx.layout_row(row3(m.pad, content_w, m.pad), 18)
    label(s.ctx, String(""))
    separator(s.ctx)
    label(s.ctx, String(""))
    s.ctx.layout_row(row3(m.pad, content_w, m.pad), artifact_h)
    label(s.ctx, String(""))
    s.ctx.begin_column()
    field_row(s.ctx, label_w, val_w, String("Samples"), String(len(s.runtime.samples)))
    field_row(s.ctx, label_w, val_w, String("Checkpoints"), String(len(s.runtime.checkpoints)))
    field_row(s.ctx, label_w, val_w, String("Backend"), s.runtime.backend_label.copy())
    s.ctx.end_column()
    label(s.ctx, String(""))

    s.ctx.layout_row(row3(m.pad, content_w, m.pad), 18)
    label(s.ctx, String(""))
    separator(s.ctx)
    label(s.ctx, String(""))
    s.ctx.layout_row(row3(m.pad, content_w, m.pad), hardware_h)
    label(s.ctx, String(""))
    s.ctx.begin_column()
    var gpu_name = s.runtime.gpu_name.copy()
    if gpu_name.byte_length() == 0:
        gpu_name = String("unavailable")
    var gpu_driver = s.runtime.gpu_driver.copy()
    if gpu_driver.byte_length() == 0:
        gpu_driver = String("unavailable")
    var cpu_name = s.runtime.cpu_name.copy()
    if cpu_name.byte_length() == 0:
        cpu_name = String("unavailable")
    field_row(s.ctx, label_w, val_w, String("GPU Model"), gpu_name)
    field_row(s.ctx, label_w, val_w, String("Driver"), gpu_driver)
    field_row(s.ctx, label_w, val_w, String("GPU"), String(s.runtime.live.gpu_util) + String("% ") + String(s.runtime.live.temp_c) + String("C"))
    field_row(s.ctx, label_w, val_w, String("VRAM"), String(s.runtime.live.vram_gb) + String(" / ") + String(s.runtime.live.vram_total_gb) + String(" GB"))
    field_row(s.ctx, label_w, val_w, String("CPU Model"), cpu_name)
    field_row(s.ctx, label_w, val_w, String("CPU"), String(s.runtime.live.cpu_util) + String("%"))
    field_row(s.ctx, label_w, val_w, String("RAM"), String(s.runtime.live.ram_gb) + String(" / ") + String(s.runtime.ram_total_gb) + String(" GB"))
    s.ctx.end_column()
    label(s.ctx, String(""))
    end_scroll_area(s.ctx)
    s.ctx.theme.font_size_pt = saved_font_size
    s.ctx.theme.row_height = saved_row_height


def _ui(mut s: TrainUIAppState) raises:
    var m = s.metrics.copy()
    draw_shell_background(s.ctx, m, s.win_w, s.win_h)
    s.ctx.layout_row(row5(m.nav_w, m.gap, m.main_w, m.gap, m.status_w), Int32(s.win_h))
    s.ctx.begin_column()
    _sidebar(s)
    s.ctx.end_column()
    label(s.ctx, String(""))
    s.ctx.begin_column()
    _main_panel(s)
    s.ctx.end_column()
    label(s.ctx, String(""))
    s.ctx.begin_column()
    _status_rail(s)
    s.ctx.end_column()
    render_media_lightbox(s.ctx, s.runtime, s.media)


def _render(mut ctx: Context) raises:
    _ = render_context_commands(ctx, String("Serenity Trainer UI"))


def _frame() -> None:
    var sp = retrieve_user_state[TrainUIAppState]()
    if sp[].font_id == 0:
        sp[].font_id = Backend.load_font(String(""))
        sp[].ctx.set_default_font(sp[].font_id)

    load_preview_textures(sp[].media, sp[].cfg, sp[].runtime)
    _sync_window(sp[])
    trainer_ui_tick_and_apply(sp[].runtime)
    sp[].ctx.begin_frame(Vec2(sp[].win_w, sp[].win_h))
    Backend.frame_begin(sp[].ctx.theme.bg.copy())
    try:
        _ui(sp[])
    except e:
        print("Serenity Trainer UI error:", String(e))
    sp[].ctx.end_frame()
    try:
        _render(sp[].ctx)
    except e:
        print("Serenity Trainer UI render error:", String(e))
    Backend.frame_end()


def main() raises:
    var state = TrainUIAppState()
    var sp = UnsafePointer(to=state)
    store_user_state(sp)

    var initial = _initial_window_size()
    var rc = Backend.init(
        Int32(Int(initial.x)),
        Int32(Int(initial.y)),
        String("Serenity Trainer"),
    )
    if rc != 0:
        print("FAIL: Backend.init returned", rc)
        raise Error("init failed")

    print("Opening Serenity trainer UI.")
    Backend.run_blocking(_frame)
    print("PASS: Serenity trainer UI exited. runs=", len(state.runtime.runs))
