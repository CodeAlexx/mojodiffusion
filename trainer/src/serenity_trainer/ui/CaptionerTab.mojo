"""Pure-Mojo Simple Captioner screen for the native trainer UI."""

from mojoui.core.context import Context
from mojoui.core.textedit import TextEditState
from mojoui.render.backend import Backend
from mojoui.widgets.app_shell import action_button
from mojoui.widgets.basic import label
from mojoui.widgets.form import (
    begin_form_panel,
    end_form_panel,
    edit_row,
    field_row,
    select_index_row,
    slider_row,
    toggle_row,
)
from mojoui.widgets.image import media_virtual_grid
from serenity_trainer.ui.TrainerConfigModel import TrainerUIConfig
from serenity_trainer.ui.UITabCommon import row1, row2, row3, two_col_w, value_w


struct CaptionerScreenState(Movable):
    var scan_loaded: Bool
    var scan_source_path: String
    var selected_index: Int32
    var status_text: String
    var last_caption: String
    var last_media_name: String
    var total_media: Int32
    var image_count: Int32
    var video_count: Int32
    var sidecar_count: Int32
    var progress: Float32
    var gallery_scroll_y: Float32
    var preview_textures: List[UInt32]
    var preview_widths: List[Int32]
    var preview_heights: List[Int32]
    var preview_titles: List[String]
    var preview_subtitles: List[String]
    var preview_paths: List[String]
    var preview_is_videos: List[Bool]

    def __init__(out self):
        self.scan_loaded = False
        self.scan_source_path = String("")
        self.selected_index = -1
        self.status_text = String("Captioner ready. Scan a folder to index media.")
        self.last_caption = String("")
        self.last_media_name = String("")
        self.total_media = 0
        self.image_count = 0
        self.video_count = 0
        self.sidecar_count = 0
        self.progress = 0.0
        self.gallery_scroll_y = 0.0
        self.preview_textures = List[UInt32]()
        self.preview_widths = List[Int32]()
        self.preview_heights = List[Int32]()
        self.preview_titles = List[String]()
        self.preview_subtitles = List[String]()
        self.preview_paths = List[String]()
        self.preview_is_videos = List[Bool]()


def _basename(path: String) -> String:
    var last = -1
    for i in range(path.byte_length()):
        if String(path[byte=i]) == String("/"):
            last = i
    if last >= 0 and last + 1 < path.byte_length():
        return String(path[byte=last + 1:])
    return path.copy()


def _sidecar_path(path: String) -> String:
    var last_slash = -1
    var last_dot = -1
    for i in range(path.byte_length()):
        var ch = String(path[byte=i])
        if ch == String("/"):
            last_slash = i
        elif ch == String("."):
            last_dot = i
    if last_dot > last_slash:
        return String(path[byte=0:last_dot]) + String(".txt")
    return path.copy() + String(".txt")


def _file_exists(path: String) -> Bool:
    try:
        var f = open(path.copy(), "r")
        f.close()
        return True
    except:
        return False


def _read_text_file(path: String) -> String:
    try:
        var f = open(path.copy(), "r")
        var text = f.read()
        f.close()
        return text^
    except:
        return String("")


def _json_escape(text: String) -> String:
    var out = String("")
    for i in range(text.byte_length()):
        var ch = String(text[byte=i])
        if ch == String("\\"):
            out = out + String("\\\\")
        elif ch == String("\""):
            out = out + String("\\\"")
        elif ch == String("\n"):
            out = out + String("\\n")
        elif ch == String("\r"):
            out = out + String("\\r")
        elif ch == String("\t"):
            out = out + String("\\t")
        else:
            out = out + ch
    return out^


def _captioner_model_id(cfg: TrainerUIConfig) -> String:
    var selected = cfg.captioner_model_label()
    if selected == String("Custom...") and cfg.captioner_custom_model_id.byte_length() > 0:
        return cfg.captioner_custom_model_id.copy()
    return selected^


def captioner_final_prompt(cfg: TrainerUIConfig) -> String:
    var prompt = cfg.captioner_prompt.copy()
    if cfg.captioner_summary_mode and cfg.captioner_one_sentence_mode:
        prompt = prompt + String(" Give a one-sentence summary of the scene.")
    elif cfg.captioner_summary_mode:
        prompt = prompt + String(" Give a short summary of the scene.")
    elif cfg.captioner_one_sentence_mode:
        prompt = prompt + String(" Describe this image in one sentence.")
    return prompt^


def clear_captioner_scan(mut state: CaptionerScreenState):
    for i in range(len(state.preview_textures)):
        if state.preview_textures[i] != UInt32(0):
            Backend.destroy_texture(state.preview_textures[i])
    state.scan_loaded = False
    state.scan_source_path = String("")
    state.selected_index = -1
    state.last_caption = String("")
    state.last_media_name = String("")
    state.total_media = 0
    state.image_count = 0
    state.video_count = 0
    state.sidecar_count = 0
    state.progress = 0.0
    state.gallery_scroll_y = 0.0
    state.preview_textures = List[UInt32]()
    state.preview_widths = List[Int32]()
    state.preview_heights = List[Int32]()
    state.preview_titles = List[String]()
    state.preview_subtitles = List[String]()
    state.preview_paths = List[String]()
    state.preview_is_videos = List[Bool]()


def _select_captioner_media(mut state: CaptionerScreenState, idx: Int32):
    if idx < 0 or idx >= Int32(len(state.preview_paths)):
        return
    state.selected_index = idx
    var path = state.preview_paths[Int(idx)].copy()
    state.last_media_name = _basename(path.copy())
    var cap_path = _sidecar_path(path.copy())
    state.last_caption = _read_text_file(cap_path.copy())
    if state.last_caption.byte_length() == 0:
        state.last_caption = String("No sidecar caption yet: ") + cap_path


def scan_captioner_folder(mut state: CaptionerScreenState, cfg: TrainerUIConfig):
    clear_captioner_scan(state)
    if cfg.captioner_folder_path.byte_length() == 0:
        state.status_text = String("Captioner folder path is empty.")
        return
    try:
        var media = Backend.scan_media_files(cfg.captioner_folder_path.copy(), True, 4096)
        var preload = 64
        for i in range(len(media)):
            var item = media[i].copy()
            var base = _basename(item.path.copy())
            var loaded_id = UInt32(0)
            var width: Int32 = 0
            var height: Int32 = 0
            if i < preload:
                if item.is_video and Backend.is_video_file(item.path.copy()):
                    var tex = Backend.load_video_thumbnail_info(item.path.copy(), 640, 640)
                    loaded_id = tex.texture_id
                    width = tex.width
                    height = tex.height
                else:
                    var tex = Backend.load_texture_file_info(item.path.copy(), 640, 640)
                    loaded_id = tex.texture_id
                    width = tex.width
                    height = tex.height
            state.preview_textures.append(loaded_id)
            state.preview_widths.append(width)
            state.preview_heights.append(height)
            state.preview_titles.append(String("Media ") + String(i + 1))
            var sidecar = _sidecar_path(item.path.copy())
            var subtitle = base.copy()
            if _file_exists(sidecar.copy()):
                state.sidecar_count = state.sidecar_count + 1
                subtitle = subtitle + String(" - captioned")
            else:
                subtitle = subtitle + String(" - no caption")
            state.preview_subtitles.append(subtitle^)
            state.preview_paths.append(item.path.copy())
            state.preview_is_videos.append(item.is_video)
            if item.is_video:
                state.video_count = state.video_count + 1
            else:
                state.image_count = state.image_count + 1
        state.total_media = Int32(len(state.preview_paths))
        if state.total_media > 0:
            state.progress = Float32(state.sidecar_count) / Float32(state.total_media)
            _select_captioner_media(state, 0)
            state.status_text = (
                String("Indexed ")
                + String(state.total_media)
                + String(" media files from ")
                + cfg.captioner_folder_path.copy()
            )
        else:
            state.status_text = String("No captionable image/video files found.")
        state.scan_loaded = True
        state.scan_source_path = cfg.captioner_folder_path.copy()
    except e:
        state.status_text = String("Captioner scan failed: ") + String(e)


def _append_captioner_command(cfg: TrainerUIConfig, action: String) raises:
    var f = open(String("target/serenity_captioner_commands.jsonl"), "a")
    f.write(
        String("{\"schema\":\"serenity.captioner_command.v1\",")
        + String("\"action\":\"") + _json_escape(action.copy()) + String("\",")
        + String("\"folder\":\"") + _json_escape(cfg.captioner_folder_path.copy()) + String("\",")
        + String("\"model\":\"") + _json_escape(_captioner_model_id(cfg)) + String("\",")
        + String("\"quant\":\"") + _json_escape(cfg.captioner_quant_label()) + String("\",")
        + String("\"attention\":\"") + _json_escape(cfg.captioner_attention_label()) + String("\",")
        + String("\"resolution\":\"") + _json_escape(cfg.captioner_resolution_label()) + String("\",")
        + String("\"skip_existing\":") + String(cfg.captioner_skip_existing) + String(",")
        + String("\"max_tokens\":") + String(cfg.captioner_max_tokens) + String(",")
        + String("\"prompt\":\"") + _json_escape(captioner_final_prompt(cfg)) + String("\"}")
    )
    f.write("\n")
    f.close()


def queue_captioner_action(mut state: CaptionerScreenState, cfg: TrainerUIConfig, action: String):
    try:
        _append_captioner_command(cfg, action.copy())
        state.status_text = String("Queued captioner ") + action.copy() + String(" command for pure-Mojo runner.")
    except e:
        state.status_text = String("Captioner command write failed: ") + String(e)


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


def _button_total(label_w: Int32, val_w: Int32) -> Int32:
    var total = label_w + val_w
    if total < 240:
        total = 240
    return total


def _button_row2(mut ctx: Context, label_w: Int32, val_w: Int32) -> Int32:
    var total = _button_total(label_w, val_w)
    var left = total // 2
    ctx.layout_row(row2(left, total - left), ctx.theme.row_height)
    return 0


def _button_row3(mut ctx: Context, label_w: Int32, val_w: Int32) -> Int32:
    var total = _button_total(label_w, val_w)
    var left = total // 3
    var mid = total // 3
    ctx.layout_row(row3(left, mid, total - left - mid), ctx.theme.row_height)
    return 0


def render_captioner_tab(
    mut ctx: Context,
    mut cfg: TrainerUIConfig,
    mut state: CaptionerScreenState,
    content_w: Int32,
    mut folder_edit: TextEditState,
    mut custom_model_edit: TextEditState,
    mut prompt_edit: TextEditState,
) raises:
    var cw = two_col_w(content_w)
    var label_w = ctx.theme.font_size_pt * 8
    if label_w < 210:
        label_w = 210
    var val_w = value_w(cw - ctx.theme.padding * 2, label_w)

    ctx.layout_row(row2(cw, cw), _panel_h(ctx, 9))
    begin_form_panel(ctx, String("CAPTION MODEL"), String("Qwen VL caption model settings"), ctx.theme.padding)
    _ = select_index_row(ctx, label_w, val_w, String("Model"), String("captioner_model"), cfg.captioner_model_options, cfg.captioner_model_index, cfg.select_open_id)
    if cfg.captioner_model_label() == String("Custom..."):
        _ = edit_row(ctx, label_w, val_w, String("Custom ID"), String("captioner_custom_model_id"), cfg.captioner_custom_model_id, custom_model_edit)
    else:
        field_row(ctx, label_w, val_w, String("Resolved"), _captioner_model_id(cfg))
    _ = select_index_row(ctx, label_w, val_w, String("Quant"), String("captioner_quant"), cfg.captioner_quant_options, cfg.captioner_quant_index, cfg.select_open_id)
    _ = select_index_row(ctx, label_w, val_w, String("Attention"), String("captioner_attention"), cfg.captioner_attention_options, cfg.captioner_attention_index, cfg.select_open_id)
    _ = select_index_row(ctx, label_w, val_w, String("Resolution"), String("captioner_resolution"), cfg.captioner_resolution_options, cfg.captioner_resolution_index, cfg.select_open_id)
    field_row(ctx, label_w, val_w, String("Device"), String("CUDA preferred"))
    field_row(ctx, label_w, val_w, String("Backend"), String("Pure Mojo command bridge"))
    _ = _button_row2(ctx, label_w, val_w)
    if action_button(ctx, String("captioner_load_model"), String("Load / Reload Model"), True):
        queue_captioner_action(state, cfg, String("load_model"))
    if action_button(ctx, String("captioner_unload_model"), String("Unload"), False):
        queue_captioner_action(state, cfg, String("unload_model"))
    end_form_panel(ctx)

    begin_form_panel(ctx, String("FOLDER PROCESSING"), String("Recursive scan, prompt controls, sidecar output"), ctx.theme.padding)
    _ = edit_row(ctx, label_w, val_w, String("Folder"), String("captioner_folder_path"), cfg.captioner_folder_path, folder_edit)
    _ = edit_row(ctx, label_w, val_w, String("Prompt"), String("captioner_prompt"), cfg.captioner_prompt, prompt_edit)
    _ = toggle_row(ctx, label_w, val_w, String("Skip Existing"), String(".txt exists"), cfg.captioner_skip_existing)
    _ = toggle_row(ctx, label_w, val_w, String("Summary"), String("Short summary"), cfg.captioner_summary_mode)
    _ = toggle_row(ctx, label_w, val_w, String("One Sentence"), String("Constrain output"), cfg.captioner_one_sentence_mode)
    _ = toggle_row(ctx, label_w, val_w, String("Retain Preview"), String("On skipped media"), cfg.captioner_retain_preview)
    _ = slider_row(ctx, label_w, val_w, String("Max Tokens"), String("captioner_max_tokens"), cfg.captioner_max_tokens, 32.0, 512.0)
    _ = _button_row3(ctx, label_w, val_w)
    if action_button(ctx, String("captioner_scan"), String("Scan Folder"), True):
        scan_captioner_folder(state, cfg)
    if action_button(ctx, String("captioner_start"), String("Queue Caption Run"), False):
        queue_captioner_action(state, cfg, String("start"))
    if action_button(ctx, String("captioner_abort"), String("Abort"), False):
        queue_captioner_action(state, cfg, String("abort"))
    end_form_panel(ctx)

    ctx.layout_row(row2(cw, cw), _panel_h(ctx, 7))
    begin_form_panel(ctx, String("PROMPT PREVIEW"), String("Final instruction sent to caption model"), ctx.theme.padding)
    field_row(ctx, label_w, val_w, String("Final Prompt"), captioner_final_prompt(cfg))
    field_row(ctx, label_w, val_w, String("Status"), state.status_text.copy())
    field_row(ctx, label_w, val_w, String("Command File"), String("target/serenity_captioner_commands.jsonl"))
    field_row(ctx, label_w, val_w, String("Loaded"), String(state.scan_loaded))
    field_row(ctx, label_w, val_w, String("Source"), state.scan_source_path.copy())
    field_row(ctx, label_w, val_w, String("Captioned"), String(state.sidecar_count) + String(" / ") + String(state.total_media))
    end_form_panel(ctx)

    begin_form_panel(ctx, String("CURRENT MEDIA"), String("Selected preview and sidecar text"), ctx.theme.padding)
    field_row(ctx, label_w, val_w, String("File"), state.last_media_name.copy())
    if state.selected_index >= 0 and state.selected_index < Int32(len(state.preview_paths)):
        field_row(ctx, label_w, val_w, String("Path"), state.preview_paths[Int(state.selected_index)].copy())
        field_row(ctx, label_w, val_w, String("Sidecar"), _sidecar_path(state.preview_paths[Int(state.selected_index)].copy()))
    else:
        field_row(ctx, label_w, val_w, String("Path"), String("No media selected"))
        field_row(ctx, label_w, val_w, String("Sidecar"), String(""))
    field_row(ctx, label_w, val_w, String("Caption"), state.last_caption.copy())
    field_row(ctx, label_w, val_w, String("Images"), String(state.image_count))
    field_row(ctx, label_w, val_w, String("Videos"), String(state.video_count))
    field_row(ctx, label_w, val_w, String("Ready"), String(state.progress))
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
    var viewport_h = card_h * 3 + ctx.theme.spacing * 4
    ctx.layout_row(row1(content_w), viewport_h)
    var clicked = media_virtual_grid(
        ctx,
        String("captioner_media_grid"),
        viewport_h,
        columns,
        card_w,
        card_h,
        Int32(len(state.preview_textures)),
        state.gallery_scroll_y,
        state.preview_textures,
        state.preview_widths,
        state.preview_heights,
        state.preview_titles,
        state.preview_subtitles,
        state.preview_is_videos,
    )
    if clicked >= 0:
        _select_captioner_media(state, clicked)
