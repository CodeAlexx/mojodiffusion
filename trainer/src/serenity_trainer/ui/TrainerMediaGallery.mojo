"""Dataset/sample media gallery state and helpers for the Serenity trainer UI."""

from mojoui.core.context import Context
from mojoui.render.backend import Backend, LoadedTexture
from mojoui.widgets.image import (
    MEDIA_LIGHTBOX_CLOSE,
    MEDIA_LIGHTBOX_PLAY,
    media_lightbox,
)
from serenity_trainer.ui.TrainerConfigModel import TrainerUIConfig
from serenity_trainer.ui.TrainerRuntimeBridge import TrainerUIRuntime


struct TrainerMediaGalleryState(Movable):
    var preview_textures_loaded: Bool
    # Throttle for the in-training sample-gallery rescan (frames since the
    # last sample-dir scan; ~180 frames = ~3 s at 60 fps).
    var sample_refresh_frames: Int32
    var active_preview_texture: UInt32
    var active_preview_width: Int32
    var active_preview_height: Int32
    var active_preview_title: String
    var active_preview_subtitle: String
    var active_preview_path: String
    var active_preview_is_video: Bool
    var preview_source_path: String
    var dataset_gallery_scroll_y: Float32
    var sample_gallery_scroll_y: Float32
    var dataset_preview_textures: List[UInt32]
    var dataset_preview_widths: List[Int32]
    var dataset_preview_heights: List[Int32]
    var dataset_preview_titles: List[String]
    var dataset_preview_subtitles: List[String]
    var dataset_preview_paths: List[String]
    var dataset_preview_is_videos: List[Bool]
    var sample_preview_textures: List[UInt32]
    var sample_preview_widths: List[Int32]
    var sample_preview_heights: List[Int32]
    var sample_preview_titles: List[String]
    var sample_preview_subtitles: List[String]
    var sample_preview_paths: List[String]
    var sample_preview_is_videos: List[Bool]

    def __init__(out self):
        self.preview_textures_loaded = False
        self.sample_refresh_frames = 0
        self.active_preview_texture = UInt32(0)
        self.active_preview_width = 0
        self.active_preview_height = 0
        self.active_preview_title = String("")
        self.active_preview_subtitle = String("")
        self.active_preview_path = String("")
        self.active_preview_is_video = False
        self.preview_source_path = String("")
        self.dataset_gallery_scroll_y = 0.0
        self.sample_gallery_scroll_y = 0.0
        self.dataset_preview_textures = List[UInt32]()
        self.dataset_preview_widths = List[Int32]()
        self.dataset_preview_heights = List[Int32]()
        self.dataset_preview_titles = List[String]()
        self.dataset_preview_subtitles = List[String]()
        self.dataset_preview_paths = List[String]()
        self.dataset_preview_is_videos = List[Bool]()
        self.sample_preview_textures = List[UInt32]()
        self.sample_preview_widths = List[Int32]()
        self.sample_preview_heights = List[Int32]()
        self.sample_preview_titles = List[String]()
        self.sample_preview_subtitles = List[String]()
        self.sample_preview_paths = List[String]()
        self.sample_preview_is_videos = List[Bool]()


def _find_substring(text: String, token: String) -> Int:
    var text_len = text.byte_length()
    var token_len = token.byte_length()
    if token_len <= 0 or text_len < token_len:
        return -1
    var last = text_len - token_len
    var i = 0
    while i <= last:
        if String(text[byte=i:i + token_len]) == token:
            return i
        i = i + 1
    return -1


def _video_thumbnail_source_end(path: String) -> Int:
    var pos = _find_substring(path.copy(), String(".mp4-"))
    if pos >= 0:
        return pos + 4
    pos = _find_substring(path.copy(), String(".mov-"))
    if pos >= 0:
        return pos + 4
    pos = _find_substring(path.copy(), String(".m4v-"))
    if pos >= 0:
        return pos + 4
    pos = _find_substring(path.copy(), String(".mkv-"))
    if pos >= 0:
        return pos + 4
    pos = _find_substring(path.copy(), String(".avi-"))
    if pos >= 0:
        return pos + 4
    pos = _find_substring(path.copy(), String(".webm-"))
    if pos >= 0:
        return pos + 5
    return -1


def _is_video_thumbnail_path(path: String) -> Bool:
    return _video_thumbnail_source_end(path.copy()) > 0


def _video_source_path(path: String, is_video: Bool) -> String:
    if not is_video:
        return path.copy()
    var end = _video_thumbnail_source_end(path.copy())
    if end > 0:
        return String(path[byte=0:end])
    return path.copy()


def _append_preview_asset(
    path: String,
    title: String,
    subtitle: String,
    max_width: Int32,
    max_height: Int32,
    mut textures: List[UInt32],
    mut widths: List[Int32],
    mut heights: List[Int32],
    mut titles: List[String],
    mut subtitles: List[String],
    mut paths: List[String],
    mut is_videos: List[Bool],
    is_video: Bool = False,
    media_path: String = String(""),
):
    var video = is_video
    if Backend.is_video_file(path.copy()):
        video = True
    var loaded = Backend.load_texture_file_info(path.copy(), max_width, max_height)
    if video and Backend.is_video_file(path.copy()):
        loaded = Backend.load_video_thumbnail_info(path.copy(), max_width, max_height)
    textures.append(loaded.texture_id)
    widths.append(loaded.width)
    heights.append(loaded.height)
    titles.append(title.copy())
    subtitles.append(subtitle.copy())
    if media_path.byte_length() > 0:
        paths.append(media_path.copy())
    else:
        paths.append(path.copy())
    is_videos.append(video)


def _basename(path: String) -> String:
    # RAW-BYTE scan for '/': String(path[byte=i]) asserts when byte i lands
    # mid-codepoint (crashes on any non-ASCII path char). '/' is single-byte
    # ASCII so the slice at last+1 is always a codepoint boundary.
    var b = path.as_bytes()
    var last = -1
    for i in range(len(b)):
        if b[i] == 0x2F:  # '/'
            last = i
    if last >= 0 and last + 1 < path.byte_length():
        return String(path[byte=last + 1:])
    return path.copy()


def _load_media_texture(
    path: String,
    is_video: Bool,
    max_width: Int32,
    max_height: Int32,
) -> LoadedTexture:
    if is_video and Backend.is_video_file(path.copy()):
        return Backend.load_video_thumbnail_info(path.copy(), max_width, max_height)
    return Backend.load_texture_file_info(path.copy(), max_width, max_height)


def _append_scanned_media(
    path: String,
    title: String,
    subtitle: String,
    is_video: Bool,
    should_load: Bool,
    max_width: Int32,
    max_height: Int32,
    mut textures: List[UInt32],
    mut widths: List[Int32],
    mut heights: List[Int32],
    mut titles: List[String],
    mut subtitles: List[String],
    mut paths: List[String],
    mut is_videos: List[Bool],
):
    var texture_id = UInt32(0)
    var width: Int32 = 0
    var height: Int32 = 0
    if should_load:
        var loaded = _load_media_texture(path.copy(), is_video, max_width, max_height)
        texture_id = loaded.texture_id
        width = loaded.width
        height = loaded.height
    textures.append(texture_id)
    widths.append(width)
    heights.append(height)
    titles.append(title.copy())
    subtitles.append(subtitle.copy())
    paths.append(path.copy())
    is_videos.append(is_video)


def _load_dataset_gallery(mut media_state: TrainerMediaGalleryState, cfg: TrainerUIConfig) raises -> Int:
    if cfg.dataset_path.byte_length() == 0:
        return 0
    var media = Backend.scan_media_files(cfg.dataset_path.copy(), True, 4096)
    var dataset_preload = 48
    for i in range(len(media)):
        var item = media[i].copy()
        var base = _basename(item.path.copy())
        var video_like = item.is_video or _is_video_thumbnail_path(item.path.copy())
        _append_scanned_media(
            item.path.copy(),
            String("Dataset ") + String(i + 1),
            base.copy(),
            video_like,
            i < dataset_preload,
            640,
            640,
            media_state.dataset_preview_textures,
            media_state.dataset_preview_widths,
            media_state.dataset_preview_heights,
            media_state.dataset_preview_titles,
            media_state.dataset_preview_subtitles,
            media_state.dataset_preview_paths,
            media_state.dataset_preview_is_videos,
        )
    return len(media)


def _load_sample_gallery(mut media_state: TrainerMediaGalleryState, cfg: TrainerUIConfig) raises -> Int:
    if cfg.sample_output_dir.byte_length() == 0:
        return 0
    var media = Backend.scan_media_files(cfg.sample_output_dir.copy(), True, 4096)
    var sample_preload = 32
    for i in range(len(media)):
        var item = media[i].copy()
        var base = _basename(item.path.copy())
        var video_like = item.is_video or _is_video_thumbnail_path(item.path.copy())
        var subtitle = String("image")
        if video_like:
            subtitle = String("video")
        _append_scanned_media(
            item.path.copy(),
            String("Sample ") + String(i + 1),
            subtitle + String(" - ") + base.copy(),
            video_like,
            i < sample_preload,
            768,
            768,
            media_state.sample_preview_textures,
            media_state.sample_preview_widths,
            media_state.sample_preview_heights,
            media_state.sample_preview_titles,
            media_state.sample_preview_subtitles,
            media_state.sample_preview_paths,
            media_state.sample_preview_is_videos,
        )
    return len(media)


def _load_scanned_gallery(mut media_state: TrainerMediaGalleryState, cfg: TrainerUIConfig) raises -> Bool:
    var dataset_count = _load_dataset_gallery(media_state, cfg)
    var sample_count = _load_sample_gallery(media_state, cfg)
    return dataset_count > 0 or sample_count > 0


def gallery_columns(content_w: Int32) -> Int32:
    var columns: Int32 = content_w // 360
    if columns < 2:
        columns = 2
    if columns > 8:
        columns = 8
    return columns


def gallery_card_height(mut ctx: Context, content_w: Int32, columns: Int32) -> Int32:
    var card_w = (content_w - (columns - 1) * ctx.theme.spacing) // columns
    if card_w < 220:
        card_w = 220
    if card_w > 440:
        card_w = 440
    var card_h = card_w * 3 // 4 + ctx.theme.row_height * 2
    if card_h > 460:
        card_h = 460
    return card_h


def gallery_visible_rows(content_w: Int32) -> Int32:
    var visible_rows: Int32 = 3
    if content_w >= 1800:
        visible_rows = 4
    if content_w >= 3000:
        visible_rows = 5
    return visible_rows


def _ensure_media_range(
    start_index: Int,
    end_index: Int,
    max_width: Int32,
    max_height: Int32,
    active_texture: UInt32,
    mut textures: List[UInt32],
    mut widths: List[Int32],
    mut heights: List[Int32],
    paths: List[String],
    is_videos: List[Bool],
):
    if len(textures) == 0:
        return
    var start = start_index
    if start < 0:
        start = 0
    var end = end_index
    if end > len(textures):
        end = len(textures)
    var keep_start = start - 96
    if keep_start < 0:
        keep_start = 0
    var keep_end = end + 96
    if keep_end > len(textures):
        keep_end = len(textures)
    for i in range(len(textures)):
        if i >= start and i < end and textures[i] == UInt32(0):
            var loaded = _load_media_texture(
                paths[i].copy(),
                is_videos[i],
                max_width,
                max_height,
            )
            textures[i] = loaded.texture_id
            widths[i] = loaded.width
            heights[i] = loaded.height
        elif (i < keep_start or i >= keep_end) and textures[i] != UInt32(0):
            if active_texture == UInt32(0) or textures[i] != active_texture:
                Backend.destroy_texture(textures[i])
                textures[i] = UInt32(0)
                widths[i] = 0
                heights[i] = 0


def ensure_visible_dataset_textures(
    mut media_state: TrainerMediaGalleryState,
    mut ctx: Context,
    content_w: Int32,
):
    var columns = gallery_columns(content_w)
    var card_h = gallery_card_height(ctx, content_w, columns)
    var row_pitch = card_h + ctx.theme.spacing
    if row_pitch < 1:
        row_pitch = 1
    var start_row = Int(media_state.dataset_gallery_scroll_y / Float32(row_pitch))
    var rows = gallery_visible_rows(content_w) + 4
    var start = start_row * Int(columns)
    var end = (start_row + Int(rows)) * Int(columns)
    _ensure_media_range(
        start,
        end,
        640,
        640,
        media_state.active_preview_texture,
        media_state.dataset_preview_textures,
        media_state.dataset_preview_widths,
        media_state.dataset_preview_heights,
        media_state.dataset_preview_paths,
        media_state.dataset_preview_is_videos,
    )


def ensure_visible_sample_textures(
    mut media_state: TrainerMediaGalleryState,
    mut ctx: Context,
    content_w: Int32,
):
    var columns = gallery_columns(content_w)
    var card_h = gallery_card_height(ctx, content_w, columns)
    var row_pitch = card_h + ctx.theme.spacing
    if row_pitch < 1:
        row_pitch = 1
    var start_row = Int(media_state.sample_gallery_scroll_y / Float32(row_pitch))
    var rows = gallery_visible_rows(content_w) + 4
    var start = start_row * Int(columns)
    var end = (start_row + Int(rows)) * Int(columns)
    _ensure_media_range(
        start,
        end,
        768,
        768,
        media_state.active_preview_texture,
        media_state.sample_preview_textures,
        media_state.sample_preview_widths,
        media_state.sample_preview_heights,
        media_state.sample_preview_paths,
        media_state.sample_preview_is_videos,
    )


def _destroy_texture_list(textures: List[UInt32]):
    for i in range(len(textures)):
        if textures[i] != UInt32(0):
            Backend.destroy_texture(textures[i])


def clear_preview_galleries(mut media_state: TrainerMediaGalleryState):
    _destroy_texture_list(media_state.dataset_preview_textures)
    _destroy_texture_list(media_state.sample_preview_textures)
    media_state = TrainerMediaGalleryState()


def _sync_runtime_samples(media_state: TrainerMediaGalleryState, mut rt: TrainerUIRuntime):
    # Keep the status-rail / logs-tab "Samples" surface fed from the scanned
    # sample gallery (rt.samples was never populated anywhere before).
    rt.samples = List[String]()
    for i in range(len(media_state.sample_preview_paths)):
        rt.samples.append(media_state.sample_preview_paths[i].copy())


def _refresh_sample_gallery(
    mut media_state: TrainerMediaGalleryState,
    cfg: TrainerUIConfig,
    mut rt: TrainerUIRuntime,
):
    """Rescan the sample dir and rebuild the sample gallery iff the media
    file count changed (new training samples landed). The cheap rescan runs
    throttled from load_preview_textures; the texture rebuild only happens
    on an actual change."""
    if cfg.sample_output_dir.byte_length() == 0:
        return
    try:
        var media = Backend.scan_media_files(cfg.sample_output_dir.copy(), True, 4096)
        if len(media) == len(media_state.sample_preview_paths):
            return
        _destroy_texture_list(media_state.sample_preview_textures)
        media_state.sample_preview_textures = List[UInt32]()
        media_state.sample_preview_widths = List[Int32]()
        media_state.sample_preview_heights = List[Int32]()
        media_state.sample_preview_titles = List[String]()
        media_state.sample_preview_subtitles = List[String]()
        media_state.sample_preview_paths = List[String]()
        media_state.sample_preview_is_videos = List[Bool]()
        var count = _load_sample_gallery(media_state, cfg)
        _sync_runtime_samples(media_state, rt)
        rt.logs.append(
            String("sample gallery refreshed: ")
            + String(count)
            + String(" files in ")
            + cfg.sample_output_dir.copy()
        )
    except e:
        rt.logs.append(String("sample gallery refresh failed: ") + String(e))


def load_preview_textures(
    mut media_state: TrainerMediaGalleryState,
    cfg: TrainerUIConfig,
    mut rt: TrainerUIRuntime,
):
    var source_key = cfg.dataset_path.copy() + String("\n") + cfg.sample_output_dir.copy()
    if media_state.preview_textures_loaded and media_state.preview_source_path == source_key:
        # Pick up NEW sample images written during/after training: throttled
        # count-only rescan (~3 s). Skipped while a lightbox preview is open
        # so the active texture is never destroyed under it.
        media_state.sample_refresh_frames = media_state.sample_refresh_frames + 1
        if (
            media_state.sample_refresh_frames >= 180
            and media_state.active_preview_texture == UInt32(0)
        ):
            media_state.sample_refresh_frames = 0
            _refresh_sample_gallery(media_state, cfg, rt)
        return
    if media_state.preview_textures_loaded:
        clear_preview_galleries(media_state)
        media_state.preview_textures_loaded = False
    try:
        if _load_scanned_gallery(media_state, cfg):
            media_state.preview_textures_loaded = True
            media_state.preview_source_path = source_key.copy()
            _sync_runtime_samples(media_state, rt)
            rt.logs.append(
                String("loaded media galleries: ")
                + String(len(media_state.dataset_preview_textures))
                + String(" dataset files from ")
                + cfg.dataset_path.copy()
                + String(", ")
                + String(len(media_state.sample_preview_textures))
                + String(" sample files from ")
                + cfg.sample_output_dir.copy()
            )
            return
    except e:
        rt.logs.append(String("media scan failed: ") + String(e))
    media_state.preview_textures_loaded = True
    media_state.preview_source_path = source_key.copy()
    rt.logs.append(
        String("no media indexed for dataset/sample dirs: ")
        + cfg.dataset_path.copy()
        + String(" | ")
        + cfg.sample_output_dir.copy()
    )


def _open_preview(
    mut media_state: TrainerMediaGalleryState,
    texture_id: UInt32,
    width: Int32,
    height: Int32,
    title: String,
    subtitle: String,
    path: String,
    is_video: Bool,
):
    if texture_id == UInt32(0) and path.byte_length() == 0:
        return
    media_state.active_preview_texture = texture_id
    media_state.active_preview_width = width
    media_state.active_preview_height = height
    media_state.active_preview_title = title.copy()
    media_state.active_preview_subtitle = subtitle.copy()
    media_state.active_preview_path = path.copy()
    media_state.active_preview_is_video = is_video


def open_dataset_preview(mut media_state: TrainerMediaGalleryState, clicked: Int32):
    if clicked < 0 or clicked >= Int32(len(media_state.dataset_preview_textures)):
        return
    if media_state.dataset_preview_textures[Int(clicked)] == UInt32(0):
        var loaded = _load_media_texture(
            media_state.dataset_preview_paths[Int(clicked)].copy(),
            media_state.dataset_preview_is_videos[Int(clicked)],
            960,
            960,
        )
        media_state.dataset_preview_textures[Int(clicked)] = loaded.texture_id
        media_state.dataset_preview_widths[Int(clicked)] = loaded.width
        media_state.dataset_preview_heights[Int(clicked)] = loaded.height
    _open_preview(
        media_state,
        media_state.dataset_preview_textures[Int(clicked)],
        media_state.dataset_preview_widths[Int(clicked)],
        media_state.dataset_preview_heights[Int(clicked)],
        media_state.dataset_preview_titles[Int(clicked)].copy(),
        media_state.dataset_preview_subtitles[Int(clicked)].copy(),
        media_state.dataset_preview_paths[Int(clicked)].copy(),
        media_state.dataset_preview_is_videos[Int(clicked)],
    )


def open_sample_preview(mut media_state: TrainerMediaGalleryState, clicked: Int32):
    if clicked < 0 or clicked >= Int32(len(media_state.sample_preview_textures)):
        return
    if media_state.sample_preview_textures[Int(clicked)] == UInt32(0):
        var loaded = _load_media_texture(
            media_state.sample_preview_paths[Int(clicked)].copy(),
            media_state.sample_preview_is_videos[Int(clicked)],
            960,
            960,
        )
        media_state.sample_preview_textures[Int(clicked)] = loaded.texture_id
        media_state.sample_preview_widths[Int(clicked)] = loaded.width
        media_state.sample_preview_heights[Int(clicked)] = loaded.height
    _open_preview(
        media_state,
        media_state.sample_preview_textures[Int(clicked)],
        media_state.sample_preview_widths[Int(clicked)],
        media_state.sample_preview_heights[Int(clicked)],
        media_state.sample_preview_titles[Int(clicked)].copy(),
        media_state.sample_preview_subtitles[Int(clicked)].copy(),
        media_state.sample_preview_paths[Int(clicked)].copy(),
        media_state.sample_preview_is_videos[Int(clicked)],
    )


def render_media_lightbox(
    mut ctx: Context,
    mut rt: TrainerUIRuntime,
    mut media_state: TrainerMediaGalleryState,
):
    if media_state.active_preview_texture == UInt32(0):
        return
    var action = media_lightbox(
        ctx,
        String("trainer_preview_lightbox"),
        media_state.active_preview_texture,
        media_state.active_preview_title.copy(),
        media_state.active_preview_subtitle.copy(),
        media_state.active_preview_path.copy(),
        media_state.active_preview_is_video,
        media_state.active_preview_width,
        media_state.active_preview_height,
    )
    if action == MEDIA_LIGHTBOX_PLAY:
        var video_path = _video_source_path(
            media_state.active_preview_path.copy(),
            media_state.active_preview_is_video,
        )
        if Backend.open_video_file(video_path.copy()):
            rt.logs.append(String("opened video: ") + video_path.copy())
            rt.last_command = String("play video")
        else:
            rt.logs.append(String("video player launch failed: ") + video_path.copy())
            rt.last_command = String("play failed")
    elif action == MEDIA_LIGHTBOX_CLOSE:
        media_state.active_preview_texture = UInt32(0)
        media_state.active_preview_width = 0
        media_state.active_preview_height = 0
        media_state.active_preview_title = String("")
        media_state.active_preview_subtitle = String("")
        media_state.active_preview_path = String("")
        media_state.active_preview_is_video = False
