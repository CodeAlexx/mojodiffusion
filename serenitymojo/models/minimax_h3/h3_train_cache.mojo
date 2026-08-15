# serenitymojo/models/minimax_h3/h3_train_cache.mojo
#
# Musubi-NATIVE MiniMax-H3 training cache reader (mmh3_* contract).
# Oracle: musubi-tuner akane/minimax-h3 @ 04324c28 —
#   minimax_h3/cache.py (keys + validation), integration.py:1512-1585
#   (latent writer), conditioning.py:475-529 (TE writer),
#   image_video_dataset.py:193-213 (paths).
#
# Files per item (cache_directory):
#   {basename}_{W:04d}x{H:04d}_mmh3.safetensors      latent cache
#   {basename}_mmh3_te.safetensors                   conditioning cache
#
# Latent cache keys (t2va AV video item):
#   latents_{F}x{H}x{W}_{dtype}            [24, F, H, W]   video latents
#   latents_audio_2x32x{T}_{dtype}         [2, 32, T]      audio latents
#   audio_loss_mask                        [T] bool
#   video_loss_mask                        [F, H, W] bool  (optional)
#   varlen_mmh3_keyframe_video_rows_{dt}   [2R, RW]        (videos: first+last)
#   mmh3_video_geometry_int64              [2]             (audio-only caches)
#
# Conditioning cache keys:
#   varlen_mmh3_hidden_states_{dtype}      [tokens, 5120]
#   varlen_mmh3_token_tags_int64           [tokens]
#   mmh3_conditioning_task                 scalar int64 (t2va=0 i2va=1 fl2va=2
#                                          ref2va=3 ref2va_omni=4 l2va=5)
#   varlen_mmh3_empty_hidden_states_{dt} + varlen_mmh3_empty_token_tags_int64
#                                          (present iff --cache_guidance_empty)
#
# Keys carry a musubi dtype suffix (torch names: bfloat16/float32/int64/...)
# except bool masks (_mask suffix) and 0-dim scalars. `varlen_` prefixes
# variable-length tensors. Logical matching mirrors musubi's _logical_key:
# strip varlen_, strip one trailing dtype suffix.
from std.os import listdir
from max.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.tensor import Tensor

comptime TArc = ArcPointer[Tensor]
comptime H3_CACHE_ARCH = "mmh3"
comptime H3_TE_SUFFIX = "_mmh3_te.safetensors"
comptime H3_LATENT_SUFFIX = "_mmh3.safetensors"
comptime H3_TEXT_DIM = 5120
comptime H3_VIDEO_LATENT_CHANNELS = 24


def _substr(s: String, start: Int, end: Int) raises -> String:
    """Byte-range substring (cache names are ASCII)."""
    if start < 0 or end > s.byte_length() or start > end:
        raise Error("_substr: range out of bounds")
    var src = s.as_bytes()
    var out = List[UInt8]()
    for i in range(start, end):
        out.append(src[i])
    return String(unsafe_from_utf8=out)


def _known_dtype_suffixes() -> List[String]:
    # torch dtype names musubi's dtype_to_str can emit; none is a "_"+suffix
    # of another, so a single endswith pass is unambiguous.
    var s = List[String]()
    s.append(String("bfloat16"))
    s.append(String("float16"))
    s.append(String("float32"))
    s.append(String("float64"))
    s.append(String("float8_e4m3fn"))
    s.append(String("float8_e5m2"))
    s.append(String("int64"))
    s.append(String("int32"))
    s.append(String("int16"))
    s.append(String("int8"))
    s.append(String("uint8"))
    s.append(String("bool"))
    return s^


def h3_cache_logical_key(key: String) raises -> String:
    """Mirror musubi model_utils.remove_dtype_suffix + varlen_ strip."""
    var k = key
    if k.startswith("varlen_"):
        k = _substr(k, 7, k.byte_length())
    var suffixes = _known_dtype_suffixes()
    for i in range(len(suffixes)):
        var suf = String("_") + suffixes[i]
        if k.endswith(suf):
            return _substr(k, 0, k.byte_length() - suf.byte_length())
    return k


def _find_logical(st: SafeTensors, logical: String) raises -> Optional[String]:
    """The unique physical key whose logical form is `logical` (None if absent)."""
    var names = st.names()
    var found = Optional[String](None)
    for i in range(len(names)):
        if h3_cache_logical_key(names[i]) == logical:
            if found:
                raise Error(
                    String("h3 cache: duplicate logical key ") + logical
                )
            found = Optional[String](names[i])
    return found^


def _require_logical(st: SafeTensors, logical: String) raises -> String:
    var f = _find_logical(st, logical)
    if not f:
        raise Error(String("h3 cache: missing required key ") + logical)
    return f.value()


def _device_tensor(st: SafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    var info = st.tensor_info(name)
    return Tensor.from_view(
        from_parts(info.dtype, info.shape.copy(), st.tensor_bytes(name)), ctx
    )


def _host_i64(st: SafeTensors, name: String) raises -> List[Int]:
    var info = st.tensor_info(name)
    if info.dtype != STDtype.I64:
        raise Error(String("h3 cache: ") + name + " must be int64")
    var span = st.tensor_bytes(name)
    var n = len(span) // 8
    var out = List[Int]()
    for i in range(n):
        var v = 0
        for b in range(8):
            v |= Int(span[i * 8 + b]) << (8 * b)
        out.append(v)
    return out^


def _host_bool(st: SafeTensors, name: String) raises -> List[Bool]:
    var info = st.tensor_info(name)
    if info.dtype != STDtype.BOOL:
        raise Error(String("h3 cache: ") + name + " must be bool")
    var span = st.tensor_bytes(name)
    var out = List[Bool]()
    for i in range(len(span)):
        out.append(span[i] != 0)
    return out^


def _dummy(ctx: DeviceContext) raises -> Tensor:
    """Placeholder for absent optional tensors (check the has_* flag)."""
    var vals: List[Float32] = [Float32(0)]
    var sh: List[Int] = [1]
    return Tensor.from_host(vals, sh^, STDtype.BF16, ctx)


struct H3CacheItemPaths(Copyable, Movable):
    var item_key: String       # basename (no resolution, no suffix)
    var latent_path: String
    var te_path: String

    def __init__(out self, var item_key: String, var latent_path: String, var te_path: String):
        self.item_key = item_key^
        self.latent_path = latent_path^
        self.te_path = te_path^


def h3_discover_cache_items(cache_dir: String) raises -> List[H3CacheItemPaths]:
    """Enumerate paired latent+TE caches (musubi discovery: glob *_mmh3_te)."""
    var entries = listdir(cache_dir)
    var latents = List[String]()
    var tes = List[String]()
    for i in range(len(entries)):
        var name = entries[i]
        if name.endswith(H3_TE_SUFFIX):
            tes.append(name)
        elif name.endswith(H3_LATENT_SUFFIX):
            latents.append(name)
    var items = List[H3CacheItemPaths]()
    for i in range(len(tes)):
        var base = _substr(
            tes[i], 0, tes[i].byte_length() - String(H3_TE_SUFFIX).byte_length()
        )
        # latent file = {base}_{W:04d}x{H:04d}_mmh3.safetensors
        var found_latent = Optional[String](None)
        var prefix = base + "_"
        for j in range(len(latents)):
            if latents[j].startswith(prefix):
                if found_latent:
                    raise Error(
                        String("h3 cache: multiple latent caches for item ") + base
                    )
                found_latent = Optional[String](latents[j])
        if not found_latent:
            continue  # TE without latents: not trainable (musubi pairs too)
        items.append(H3CacheItemPaths(
            base^,
            cache_dir + "/" + found_latent.value(),
            cache_dir + "/" + tes[i],
        ))
    return items^


struct H3LatentCache(Movable):
    var video: TArc                 # [24, F, H, W] (dummy if has_video False)
    var has_video: Bool
    var audio: TArc                 # [2, 32, T] (dummy if has_audio False)
    var has_audio: Bool
    var audio_loss_mask: List[Bool]  # [T] (empty if no audio)
    var has_video_loss_mask: Bool
    var keyframe_rows: TArc         # [2R, RW] (dummy if has_keyframe_rows False)
    var has_keyframe_rows: Bool
    var lat_f: Int
    var lat_h: Int
    var lat_w: Int
    var audio_t: Int

    def __init__(
        out self,
        var video: TArc, has_video: Bool,
        var audio: TArc, has_audio: Bool,
        var audio_loss_mask: List[Bool],
        has_video_loss_mask: Bool,
        var keyframe_rows: TArc, has_keyframe_rows: Bool,
        lat_f: Int, lat_h: Int, lat_w: Int, audio_t: Int,
    ):
        self.video = video^
        self.has_video = has_video
        self.audio = audio^
        self.has_audio = has_audio
        self.audio_loss_mask = audio_loss_mask^
        self.has_video_loss_mask = has_video_loss_mask
        self.keyframe_rows = keyframe_rows^
        self.has_keyframe_rows = has_keyframe_rows
        self.lat_f = lat_f
        self.lat_h = lat_h
        self.lat_w = lat_w
        self.audio_t = audio_t


def h3_read_latent_cache(path: String, ctx: DeviceContext) raises -> H3LatentCache:
    var st = SafeTensors.open(path)
    var names = st.names()
    var video_key = Optional[String](None)
    var audio_key = Optional[String](None)
    for i in range(len(names)):
        var n = names[i]
        if n.startswith("latents_audio_2x32x"):
            if audio_key:
                raise Error("h3 cache: multiple audio latent tensors")
            audio_key = Optional[String](n)
        elif n.startswith("latents_") and not n.startswith("latents_audio"):
            # musubi: fullmatch latents_\d+x\d+x\d+_.+ — digit follows the '_'
            var b = n.as_bytes()
            if b[8] >= 48 and b[8] <= 57:
                if video_key:
                    raise Error("h3 cache: multiple video latent tensors")
                video_key = Optional[String](n)

    var has_video = Bool(video_key)
    var video = _dummy(ctx)
    var lat_f = 0
    var lat_h = 0
    var lat_w = 0
    if has_video:
        video = _device_tensor(st, video_key.value(), ctx)
        var sh = video.shape()
        if len(sh) != 4 or sh[0] != H3_VIDEO_LATENT_CHANNELS:
            raise Error("h3 cache: video latents must be [24, F, H, W]")
        lat_f = sh[1]
        lat_h = sh[2]
        lat_w = sh[3]

    var has_audio = Bool(audio_key)
    var audio = _dummy(ctx)
    var mask = List[Bool]()
    var audio_t = 0
    if has_audio:
        audio = _device_tensor(st, audio_key.value(), ctx)
        var sh = audio.shape()
        if len(sh) != 3 or sh[0] != 2 or sh[1] != 32:
            raise Error("h3 cache: audio latents must be [2, 32, T]")
        audio_t = sh[2]
        mask = _host_bool(st, _require_logical(st, String("audio_loss_mask")))
        if len(mask) != audio_t:
            raise Error("h3 cache: audio_loss_mask length != audio T")

    if not has_video and not has_audio:
        raise Error("h3 cache: no latent tensors in " + path)

    var kf_key = _find_logical(st, String("mmh3_keyframe_video_rows"))
    var has_kf = Bool(kf_key)
    var kf = _dummy(ctx)
    if has_kf:
        kf = _device_tensor(st, kf_key.value(), ctx)
        var sh = kf.shape()
        if len(sh) != 2 or sh[0] % 2 != 0:
            raise Error("h3 cache: keyframe rows must be [2R, RW]")

    var has_vlm = Bool(_find_logical(st, String("video_loss_mask")))
    return H3LatentCache(
        TArc(video^), has_video,
        TArc(audio^), has_audio,
        mask^, has_vlm, TArc(kf^), has_kf,
        lat_f, lat_h, lat_w, audio_t,
    )


struct H3TextCache(Movable):
    var hidden: TArc            # [tokens, 5120]
    var tags: List[Int]         # [tokens]
    var task_id: Int            # H3_CONDITIONING_TASK_IDS value
    var has_empty: Bool
    var empty_hidden: TArc      # [tokens_e, 5120] (dummy if has_empty False)
    var empty_tags: List[Int]
    var tokens: Int

    def __init__(
        out self,
        var hidden: TArc, var tags: List[Int], task_id: Int,
        has_empty: Bool, var empty_hidden: TArc, var empty_tags: List[Int],
        tokens: Int,
    ):
        self.hidden = hidden^
        self.tags = tags^
        self.task_id = task_id
        self.has_empty = has_empty
        self.empty_hidden = empty_hidden^
        self.empty_tags = empty_tags^
        self.tokens = tokens


def _hidden_pair(
    st: SafeTensors, hidden_logical: String, tags_logical: String,
    ctx: DeviceContext,
) raises -> Tuple[TArc, List[Int]]:
    var hidden = _device_tensor(st, _require_logical(st, hidden_logical), ctx)
    var sh = hidden.shape()
    if len(sh) != 2 or sh[1] != H3_TEXT_DIM:
        raise Error(
            String("h3 cache: ") + hidden_logical + " must be [tokens, 5120]"
        )
    if sh[0] == 0:
        raise Error(String("h3 cache: ") + hidden_logical + " carries no tokens")
    var tags = _host_i64(st, _require_logical(st, tags_logical))
    if len(tags) != sh[0]:
        raise Error(String("h3 cache: ") + tags_logical + " length != tokens")
    return (TArc(hidden^), tags^)


def h3_read_text_cache(path: String, ctx: DeviceContext) raises -> H3TextCache:
    var st = SafeTensors.open(path)
    var pair = _hidden_pair(
        st, String("mmh3_hidden_states"), String("mmh3_token_tags"), ctx
    )
    var hidden = pair[0]
    var tags = pair[1].copy()
    var task_vals = _host_i64(
        st, _require_logical(st, String("mmh3_conditioning_task"))
    )
    if len(task_vals) != 1:
        raise Error("h3 cache: mmh3_conditioning_task must be a scalar")
    var task_id = task_vals[0]
    if task_id < 0 or task_id > 5:
        raise Error("h3 cache: unknown conditioning task id")

    var has_empty = Bool(_find_logical(st, String("mmh3_empty_hidden_states")))
    var empty_hidden = TArc(_dummy(ctx))
    var empty_tags = List[Int]()
    if has_empty:
        var ep = _hidden_pair(
            st, String("mmh3_empty_hidden_states"),
            String("mmh3_empty_token_tags"), ctx,
        )
        empty_hidden = ep[0]
        empty_tags = ep[1].copy()

    var tokens = hidden[].shape()[0]
    return H3TextCache(
        hidden, tags^, task_id,
        has_empty, empty_hidden, empty_tags^, tokens,
    )
