"""Request-driven pure-Mojo LTX2 video CLI.

Usage:
    ltx2_request_cli <serenity.genparams.v1.json> <output_dir>

The canonical request owns prompt, conditioning artifacts, geometry, frame
count, FPS, steps, seed, sampler/scheduler, audio choice, and every LoRA row.
The compiled runner either admits those exact values or fails before loading
the model; it never substitutes a smaller smoke profile.
"""

from std.builtin.type_aliases import MutExternalOrigin
from std.ffi import external_call
from std.memory import UnsafePointer, alloc
from std.sys import argv

from json.parser import loads
from json.value import JSONValue

from serenitymojo.io.env import serenity_model_root
from serenitymojo.io.ffi import O_RDONLY, sys_close, sys_open
from serenitymojo.pipeline.ltx2_t2v_av_hq import (
    _mkdir, _write_ltx2_status, decode_request_profile, run_request_profile,
)
from serenitymojo.serve.model_scan import _read_text_file
from serenitymojo.serve.product_manifest import write_text_file


comptime _CPtr = UnsafePointer[UInt8, MutExternalOrigin]


def _setenv(name: String, value: String) raises:
    var nn = name.byte_length()
    var nb = alloc[UInt8](nn + 1)
    var nsrc = name.as_bytes()
    for i in range(nn):
        nb[i] = nsrc[i]
    nb[nn] = 0
    var vn = value.byte_length()
    var vb = alloc[UInt8](vn + 1)
    var vsrc = value.as_bytes()
    for i in range(vn):
        vb[i] = vsrc[i]
    vb[vn] = 0
    var cn = _CPtr(unsafe_from_address=Int(nb))
    var cv = _CPtr(unsafe_from_address=Int(vb))
    var rc = external_call["setenv", Int32](cn, cv, Int32(1))
    nb.free()
    vb.free()
    if rc != 0:
        raise Error(String("setenv failed for ") + name)


def _path_exists(path: String) -> Bool:
    if path.byte_length() == 0:
        return False
    var fd = sys_open(path, O_RDONLY, 0)
    if fd < 0:
        return False
    _ = sys_close(fd)
    return True


def _require_string(obj: JSONValue, key: String) raises -> String:
    if not obj.contains(key) or not obj[key].is_string():
        raise Error(String("LTX2 request: '") + key + String("' string is required"))
    return obj[key].as_string()


def _require_int(obj: JSONValue, key: String) raises -> Int:
    if not obj.contains(key) or not obj[key].is_int():
        raise Error(String("LTX2 request: '") + key + String("' integer is required"))
    return obj[key].as_int()


def _require_number(obj: JSONValue, key: String) raises -> Float64:
    if not obj.contains(key) or not obj[key].is_number():
        raise Error(String("LTX2 request: '") + key + String("' number is required"))
    return obj[key].as_float()


def _optional_string(obj: JSONValue, key: String) raises -> String:
    if not obj.contains(key):
        return String("")
    if not obj[key].is_string():
        raise Error(String("LTX2 request: '") + key + String("' must be a string"))
    return obj[key].as_string()


def _optional_bool(obj: JSONValue, key: String, default: Bool) raises -> Bool:
    if not obj.contains(key):
        return default
    if not obj[key].is_bool():
        raise Error(String("LTX2 request: '") + key + String("' must be a bool"))
    return obj[key].as_bool()


def _optional_number(
    obj: JSONValue, key: String, default: Float64
) raises -> Float64:
    if not obj.contains(key):
        return default
    if not obj[key].is_number():
        raise Error(String("LTX2 request: '") + key + String("' must be a number"))
    return obj[key].as_float()


def _resolve_lora_path(name: String) raises -> String:
    if name.byte_length() == 0:
        raise Error("LTX2 request: LoRA name cannot be empty")
    if _path_exists(name):
        return name.copy()
    if _path_exists(name + String(".safetensors")):
        return name + String(".safetensors")
    var root = serenity_model_root() + String("/loras/")
    if _path_exists(root + name):
        return root + name
    if _path_exists(root + name + String(".safetensors")):
        return root + name + String(".safetensors")
    raise Error(
        String("LTX2 request: LoRA not found: ") + name
        + String(" (tried as a path and under the Serenity LoRA root)")
    )


struct ResolvedCaps(Copyable, Movable):
    var path: String
    var has_negative: Bool
    var is_projected: Bool

    def __init__(
        out self, path: String, has_negative: Bool, is_projected: Bool = False
    ):
        self.path = path.copy()
        self.has_negative = has_negative
        self.is_projected = is_projected


def _resolve_caps_sidecar(
    authored_path: String,
    prompt: String,
    negative: String,
) raises -> ResolvedCaps:
    if authored_path.byte_length() == 0:
        raise Error("LTX2 request: caps_positive is required")
    if not _path_exists(authored_path):
        raise Error(String("LTX2 request: conditioning artifact not found: ") + authored_path)
    if not authored_path.endswith(String(".json")):
        return ResolvedCaps(authored_path, False, False)

    var meta = loads(_read_text_file(authored_path))
    if not meta.is_object():
        raise Error("LTX2 request: conditioning sidecar must be a JSON object")
    var cached_prompt = _require_string(meta, String("prompt"))
    if cached_prompt != prompt:
        raise Error(
            String("LTX2 request: prompt does not match conditioning sidecar; ")
            + String("request='") + prompt + String("' cached='")
            + cached_prompt + String("'")
        )
    var has_negative = meta.contains(String("negative_prompt")) \
        and meta[String("negative_prompt")].is_string()
    if has_negative and meta[String("negative_prompt")].as_string() != negative:
        raise Error(
            String("LTX2 request: negative prompt does not match conditioning sidecar")
        )
    var tensor_path = _require_string(meta, String("path"))
    if not _path_exists(tensor_path):
        raise Error(
            String("LTX2 request: sidecar tensor artifact not found: ")
            + tensor_path
        )
    var is_projected = False
    if meta.contains(String("conditioning_stage")):
        if not meta[String("conditioning_stage")].is_string():
            raise Error("LTX2 request: conditioning_stage must be a string")
        var stage = meta[String("conditioning_stage")].as_string()
        if stage == String("post_connector"):
            is_projected = True
        elif stage != String("pre_connector"):
            raise Error(
                "LTX2 request: conditioning_stage must be pre_connector or post_connector"
            )
    return ResolvedCaps(tensor_path, has_negative, is_projected)


def _configure_loras(obj: JSONValue) raises:
    var request_feature_id = _optional_string(
        obj, String("feature_id")
    )
    var count = 0
    if obj.contains(String("lora")):
        if not obj[String("lora")].is_array():
            raise Error("LTX2 request: 'lora' must be an array")
        count = obj[String("lora")].length()
    _setenv(String("LTX2_TRAINED_LORA_COUNT"), String(count))
    if count == 0:
        return
    var rows = obj[String("lora")]
    for i in range(count):
        var row = rows[i]
        if not row.is_object():
            raise Error(String("LTX2 request: lora[") + String(i) + String("] must be an object"))
        var name = _require_string(row, String("name"))
        var weight = Float64(1.0)
        if row.contains(String("weight")):
            if not row[String("weight")].is_number():
                raise Error(String("LTX2 request: lora[") + String(i) + String("].weight must be a number"))
            weight = row[String("weight")].as_float()
        if weight < -10.0 or weight > 10.0:
            raise Error(String("LTX2 request: lora[") + String(i) + String("].weight must be in [-10, 10]"))
        var resolved_path = _resolve_lora_path(name)
        var classified_name = resolved_path.lower()
        var row_feature_id = _optional_string(row, String("feature_id"))
        var feature_usage = _optional_string(row, String("feature_usage"))
        var is_foley = (
            classified_name.find(String("lora-foley-v2a")) >= 0
        )
        if classified_name.find(String("ic-lora")) >= 0 or (
            classified_name.find(String("ic_lora")) >= 0
        ):
            raise Error(
                String("LTX2 request: lora[") + String(i) + String("] '")
                + name
                + String("' is a feature adapter, not an ordinary LoRA overlay")
            )
        if is_foley and (
            request_feature_id != String("foley-v2a")
            or row_feature_id != String("foley-v2a")
            or feature_usage != String("v2a_feature")
        ):
            raise Error(
                String("LTX2 request: Foley may only be loaded by the ")
                + String("dedicated feature_id=foley-v2a request contract")
            )
        _setenv(
            String("LTX2_TRAINED_LORA_") + String(i),
            resolved_path,
        )
        _setenv(
            String("LTX2_TRAINED_LORA_NAME_") + String(i), name,
        )
        _setenv(
            String("LTX2_TRAINED_LORA_MULT_") + String(i), String(weight)
        )
        # Optional per-stream strengths (KJ LTX2LoraLoaderAdvanced): each in
        # [0, 1], default 1.0. Encoded for the runner as five comma-joined
        # floats `video,video_to_audio,audio,audio_to_video,other` — only set
        # when at least one differs from 1.0.
        var stream_names = [
            String("video"), String("video_to_audio"), String("audio"),
            String("audio_to_video"), String("other"),
        ]
        var stream_vals = List[Float64]()
        var any_stream = False
        for ref sname in stream_names:
            var v = Float64(1.0)
            if row.contains(sname):
                if not row[sname].is_number():
                    raise Error(
                        String("LTX2 request: lora[") + String(i)
                        + String("].") + sname + String(" must be a number")
                    )
                v = row[sname].as_float()
                if v < 0.0 or v > 1.0:
                    raise Error(
                        String("LTX2 request: lora[") + String(i)
                        + String("].") + sname + String(" must be in [0, 1]")
                    )
                if v != 1.0:
                    any_stream = True
            stream_vals.append(v)
        if is_foley and (
            stream_vals[0] != 0.0
            or stream_vals[1] != 1.0
            or stream_vals[2] != 1.0
            or stream_vals[3] != 0.0
            or stream_vals[4] != 1.0
        ):
            raise Error(
                String("LTX2 request: Foley stream contract must be ")
                + String("video=0,v2a=1,audio=1,a2v=0,other=1")
            )
        if any_stream:
            var enc = String("")
            for j in range(len(stream_vals)):
                if j > 0:
                    enc += String(",")
                enc += String(stream_vals[j])
            _setenv(
                String("LTX2_TRAINED_LORA_STREAMS_") + String(i), enc
            )


def _configure_distillation_adapter(obj: JSONValue) raises:
    # The Rust boundary resolves this exact path from the model registry.  It
    # is deliberately separate from authored overlays: a checkpoint may have
    # zero or one matching sampling/distillation adapter, and Serenity must
    # never substitute the official LTX adapter for an arbitrary finetune.
    _setenv(String("LTX2_REQUEST_DISTILLATION_LORA"), String(""))
    _setenv(String("LTX2_REQUEST_DISTILLATION_MULT"), String("1"))
    _setenv(String("LTX2_REQUEST_DISTILLATION_S1"), String(""))
    _setenv(String("LTX2_REQUEST_DISTILLATION_S2"), String(""))
    if not obj.contains(String("distillation_adapter")):
        return
    var adapter = obj[String("distillation_adapter")]
    if not adapter.is_object():
        raise Error(
            "LTX2 request: distillation_adapter must be a resolved object"
        )
    var path = _require_string(adapter, String("path"))
    if not _path_exists(path):
        raise Error(
            String("LTX2 request: distillation adapter not found: ") + path
        )
    var weight = Float64(1.0)
    if adapter.contains(String("weight")):
        if not adapter[String("weight")].is_number():
            raise Error(
                "LTX2 request: distillation_adapter.weight must be a number"
            )
        weight = adapter[String("weight")].as_float()
    if weight < -10.0 or weight > 10.0:
        raise Error(
            "LTX2 request: distillation_adapter.weight must be in [-10, 10]"
        )
    _setenv(String("LTX2_REQUEST_DISTILLATION_LORA"), path)
    _setenv(String("LTX2_REQUEST_DISTILLATION_MULT"), String(weight))
    if adapter.contains(String("stage1_weight")):
        if not adapter[String("stage1_weight")].is_number():
            raise Error(
                "LTX2 request: distillation_adapter.stage1_weight must be a number"
            )
        var stage1_weight = adapter[String("stage1_weight")].as_float()
        if stage1_weight < -10.0 or stage1_weight > 10.0:
            raise Error(
                "LTX2 request: distillation_adapter.stage1_weight must be in [-10, 10]"
            )
        _setenv(
            String("LTX2_REQUEST_DISTILLATION_S1"), String(stage1_weight)
        )
    if adapter.contains(String("stage2_weight")):
        if not adapter[String("stage2_weight")].is_number():
            raise Error(
                "LTX2 request: distillation_adapter.stage2_weight must be a number"
            )
        var stage2_weight = adapter[String("stage2_weight")].as_float()
        if stage2_weight < -10.0 or stage2_weight > 10.0:
            raise Error(
                "LTX2 request: distillation_adapter.stage2_weight must be in [-10, 10]"
            )
        _setenv(
            String("LTX2_REQUEST_DISTILLATION_S2"), String(stage2_weight)
        )


def _configure_checkpoint_workflow(obj: JSONValue) raises:
    _setenv(String("LTX2_REQUEST_WORKFLOW_PROFILE"), String(""))
    if not obj.contains(String("workflow_profile")):
        return
    if not obj[String("workflow_profile")].is_string():
        raise Error("LTX2 request: workflow_profile must be a string")
    _setenv(
        String("LTX2_REQUEST_WORKFLOW_PROFILE"),
        obj[String("workflow_profile")].as_string(),
    )


def _run_request(request_path: String, out_dir: String) raises:
    var request_text = _read_text_file(request_path)
    write_text_file(out_dir + String("/request.json"), request_text)
    var obj = loads(request_text)
    if not obj.is_object():
        raise Error("LTX2 request: root must be a JSON object")
    if obj.contains(String("schema")):
        if not obj[String("schema")].is_string() or (
            obj[String("schema")].as_string() != String("serenity.genparams.v1")
        ):
            raise Error("LTX2 request: schema must be serenity.genparams.v1")

    var prompt = _require_string(obj, String("prompt"))
    if prompt.byte_length() == 0:
        raise Error("LTX2 request: prompt cannot be empty")
    var negative = _optional_string(obj, String("negative"))
    var caps = _resolve_caps_sidecar(
        _require_string(obj, String("caps_positive")), prompt, negative
    )
    var neg_path = String("")
    var authored_neg = _optional_string(obj, String("caps_negative"))
    if authored_neg.byte_length() > 0:
        var neg_caps = _resolve_caps_sidecar(
            authored_neg, prompt, negative
        )
        neg_path = neg_caps.path.copy()
        if neg_caps.is_projected != caps.is_projected:
            raise Error(
                "LTX2 request: positive/negative conditioning stages must match"
            )
    elif caps.has_negative:
        neg_path = caps.path.copy()
    elif negative.byte_length() > 0:
        raise Error(
            "LTX2 request: negative prompt was supplied without matching negative conditioning"
        )

    var seed = _require_int(obj, String("seed"))
    if seed < 0:
        raise Error("LTX2 request: seed must be >= 0")
    var noise_fixture = _optional_string(obj, String("noise_fixture"))
    if noise_fixture.byte_length() > 0 and not _path_exists(noise_fixture):
        raise Error(
            String("LTX2 request: noise_fixture not found: ") + noise_fixture
        )
    var image_path = _optional_string(obj, String("image_path"))
    var image_strength = _optional_number(
        obj, String("image_strength"), Float64(1.0)
    )
    if image_strength < 0.0 or image_strength > 1.0:
        raise Error("LTX2 request: image_strength must be in [0, 1]")
    if image_path.byte_length() > 0 and not _path_exists(image_path):
        raise Error(
            String("LTX2 request: image_path not found: ") + image_path
        )
    if image_path.byte_length() == 0 and image_strength != 1.0:
        raise Error(
            "LTX2 request: image_strength requires a non-empty image_path"
        )
    var last_image_path = _optional_string(
        obj, String("last_image_path")
    )
    var last_image_strength = _optional_number(
        obj, String("last_image_strength"), Float64(1.0)
    )
    if last_image_strength < 0.0 or last_image_strength > 1.0:
        raise Error(
            "LTX2 request: last_image_strength must be in [0, 1]"
        )
    if (
        last_image_path.byte_length() > 0
        and not _path_exists(last_image_path)
    ):
        raise Error(
            String("LTX2 request: last_image_path not found: ")
            + last_image_path
        )
    if (
        last_image_path.byte_length() == 0
        and last_image_strength != 1.0
    ):
        raise Error(
            "LTX2 request: last_image_strength requires last_image_path"
        )
    var video_path = _optional_string(obj, String("video_path"))
    var video_strength = _optional_number(
        obj, String("video_strength"), Float64(1.0)
    )
    if video_strength < 0.0 or video_strength > 1.0:
        raise Error("LTX2 request: video_strength must be in [0, 1]")
    if video_path.byte_length() > 0 and not _path_exists(video_path):
        raise Error(
            String("LTX2 request: video_path not found: ") + video_path
        )
    if video_path.byte_length() == 0 and video_strength != 1.0:
        raise Error(
            "LTX2 request: video_strength requires a non-empty video_path"
        )
    var video_mask_path = _optional_string(
        obj, String("video_mask_path")
    )
    if (
        video_mask_path.byte_length() > 0
        and not _path_exists(video_mask_path)
    ):
        raise Error(
            String("LTX2 request: video_mask_path not found: ")
            + video_mask_path
        )
    if (
        video_mask_path.byte_length() > 0
        and video_path.byte_length() == 0
    ):
        raise Error(
            "LTX2 request: video_mask_path requires a non-empty video_path"
        )
    if image_path.byte_length() > 0 and video_path.byte_length() > 0:
        raise Error(
            "LTX2 request: image_path and video_path are mutually exclusive"
        )
    if last_image_path.byte_length() > 0 and video_path.byte_length() > 0:
        raise Error(
            "LTX2 request: last_image_path and video_path are mutually exclusive"
        )
    var video_edit_mode = _optional_string(
        obj, String("video_edit_mode")
    ).lower()
    if video_edit_mode.byte_length() == 0:
        video_edit_mode = String("standard")
    if (
        video_edit_mode != String("standard")
        and video_edit_mode != String("retake")
        and video_edit_mode != String("extend_start")
        and video_edit_mode != String("extend_end")
    ):
        raise Error(
            String("LTX2 request: unsupported video_edit_mode '")
            + video_edit_mode + String("'")
        )
    var video_edit_start = _optional_number(
        obj, String("video_edit_start"), Float64(0.0)
    )
    var video_edit_end = _optional_number(
        obj, String("video_edit_end"), Float64(0.0)
    )
    var video_source_frames = 0
    if obj.contains(String("video_source_frames")):
        video_source_frames = _require_int(
            obj, String("video_source_frames")
        )
    if video_edit_mode != String("standard"):
        if video_path.byte_length() == 0:
            raise Error(
                "LTX2 request: temporal video editing requires video_path"
            )
        if video_mask_path.byte_length() > 0:
            raise Error(
                "LTX2 request: temporal video editing cannot use video_mask_path"
            )
        if video_source_frames <= 1:
            raise Error(
                "LTX2 request: temporal video editing requires video_source_frames"
            )
        if video_edit_start < 0.0 or video_edit_end <= video_edit_start:
            raise Error(
                "LTX2 request: temporal edit window must have 0 <= start < end"
            )
        if last_image_path.byte_length() > 0:
            raise Error(
                "LTX2 keyframe interpolation cannot use Retake/Extend"
            )
    var source_audio_path = _optional_string(
        obj, String("source_audio_path")
    )
    if (
        source_audio_path.byte_length() > 0
        and not _path_exists(source_audio_path)
    ):
        raise Error(
            String("LTX2 request: source_audio_path not found: ")
            + source_audio_path
        )
    var source_audio_sample_rate = 0
    var source_audio_channels = 0
    var source_audio_samples = 0
    if obj.contains(String("source_audio_sample_rate")):
        source_audio_sample_rate = _require_int(
            obj, String("source_audio_sample_rate")
        )
    if obj.contains(String("source_audio_channels")):
        source_audio_channels = _require_int(
            obj, String("source_audio_channels")
        )
    if obj.contains(String("source_audio_samples")):
        source_audio_samples = _require_int(
            obj, String("source_audio_samples")
        )
    if source_audio_path.byte_length() > 0 and (
        source_audio_sample_rate <= 0
        or source_audio_channels != 2
        or source_audio_samples <= 0
    ):
        raise Error("LTX2 request: invalid creator source-audio metadata")
    var include_audio = _optional_bool(
        obj, String("include_audio"), False
    )
    var audio_policy = _optional_string(obj, String("audio_policy")).lower()
    if audio_policy.byte_length() == 0:
        audio_policy = String("generate") if include_audio else String("none")
    if (
        audio_policy != String("none")
        and audio_policy != String("generate")
        and audio_policy != String("preserve")
    ):
        raise Error(
            "LTX2 request: audio_policy must be none, generate, or preserve"
        )
    if (
        video_edit_mode == String("standard")
        and (audio_policy == String("generate")) != include_audio
    ):
        raise Error(
            "LTX2 request: audio_policy conflicts with include_audio"
        )
    if video_edit_mode != String("standard") and not include_audio:
        raise Error(
            "LTX2 creator Retake/Extend must decode its AV output"
        )
    if audio_policy == String("preserve") and video_path.byte_length() == 0:
        raise Error(
            "LTX2 request: audio_policy preserve requires video_path"
        )
    var regenerate_source_audio = (
        video_edit_mode == String("extend_start")
        or video_edit_mode == String("extend_end")
        or (
            video_edit_mode == String("retake")
            and audio_policy == String("generate")
        )
    )
    var feature_id = _optional_string(obj, String("feature_id"))
    if feature_id == String("foley-v2a"):
        if video_path.byte_length() == 0:
            raise Error("LTX2 request: Foley requires video_path")
        if video_mask_path.byte_length() > 0:
            raise Error("LTX2 request: Foley does not accept video_mask_path")
        if video_strength != 1.0:
            raise Error(
                "LTX2 request: Foley requires video_strength=1.0"
            )
        if audio_policy != String("generate") or not include_audio:
            raise Error(
                "LTX2 request: Foley requires generated audio"
            )
    elif feature_id == String("cinemagraph"):
        if image_path.byte_length() == 0:
            raise Error("LTX2 request: Cinemagraph requires image_path")
        if prompt.find(String("CINEMAGRAPH_MOTION")) < 0:
            raise Error(
                String("LTX2 request: Cinemagraph requires exact trigger ")
                + String("CINEMAGRAPH_MOTION")
            )
    elif feature_id.byte_length() > 0 and feature_id != String("standard"):
        raise Error(
            String("LTX2 request: unsupported admitted feature_id '")
            + feature_id + String("'")
        )
    _configure_checkpoint_workflow(obj)
    _configure_distillation_adapter(obj)
    _configure_loras(obj)
    var quant = _require_string(obj, String("quant")).lower()
    if (
        quant != String("bf16")
        and quant != String("fp8")
        and quant != String("int4")
    ):
        raise Error(
            String("LTX2 request: quant must be bf16, fp8, or int4; got '")
            + quant + String("'")
        )
    var guidance_mode = _require_string(
        obj, String("guidance_mode")
    ).lower()
    if guidance_mode != String("distilled") and guidance_mode != String("dev"):
        raise Error(
            String("LTX2 request: guidance_mode must be distilled or dev; got '")
            + guidance_mode + String("'")
        )
    _mkdir(out_dir)
    run_request_profile(
        _require_string(obj, String("checkpoint")),
        quant,
        _require_int(obj, String("width")),
        _require_int(obj, String("height")),
        _require_int(obj, String("frames")),
        _require_int(obj, String("steps")),
        UInt64(seed),
        _require_number(obj, String("fps")),
        _require_string(obj, String("sampler")),
        _require_string(obj, String("scheduler")),
        guidance_mode,
        caps.path,
        neg_path,
        caps.is_projected,
        noise_fixture,
        image_path,
        image_strength,
        last_image_path,
        last_image_strength,
        video_path,
        video_strength,
        video_mask_path,
        video_edit_mode,
        video_edit_start,
        video_edit_end,
        video_source_frames,
        source_audio_path,
        source_audio_sample_rate,
        source_audio_channels,
        source_audio_samples,
        regenerate_source_audio,
        include_audio,
        _optional_bool(obj, String("defer_decode"), False),
        out_dir,
    )


def _run_decode(request_path: String, out_dir: String) raises:
    var request = loads(_read_text_file(request_path))
    if not request.is_object():
        raise Error("LTX2 decode request must be a JSON object")
    var handoff_path = out_dir + String("/decode_handoff.json")
    if not _path_exists(handoff_path):
        raise Error(
            String("LTX2 decode handoff is missing: ") + handoff_path
        )
    var handoff = loads(_read_text_file(handoff_path))
    if not handoff.is_object() or _require_string(
        handoff, String("schema")
    ) != String("serenity.ltx2.decode_handoff.v1"):
        raise Error("LTX2 decode handoff schema mismatch")
    _configure_checkpoint_workflow(request)
    _configure_distillation_adapter(request)
    _configure_loras(request)
    var seed = _require_int(handoff, String("seed"))
    if seed < 0:
        raise Error("LTX2 decode handoff seed must be non-negative")
    decode_request_profile(
        out_dir + String("/final_latents.safetensors"),
        out_dir,
        _require_int(handoff, String("steps")),
        UInt64(seed),
        _optional_bool(handoff, String("include_audio"), False),
        _require_string(handoff, String("guidance_mode")),
        _require_string(handoff, String("quant")),
        _require_string(handoff, String("context_path")),
        _require_string(handoff, String("negative_context_path")),
        _require_int(handoff, String("request_lora_count")),
        _require_number(handoff, String("load_seconds")),
        _require_number(handoff, String("conditioning_seconds")),
        _optional_number(handoff, String("source_encode_seconds"), 0.0),
        _require_number(handoff, String("prepare_seconds")),
        _require_number(handoff, String("denoise_seconds")),
        _require_number(handoff, String("elapsed_seconds")),
        _require_int(handoff, String("total_vram_bytes")),
        _require_int(handoff, String("min_free_bytes")),
    )


def main() raises:
    var args = argv()
    var decode_mode = (
        len(args) == 4 and String(args[1]) == String("decode")
    )
    if len(args) != 3 and not decode_mode:
        raise Error(
            "usage: ltx2_request_cli <serenity.genparams.v1.json> <output_dir>"
            " | ltx2_request_cli decode <resolved-request.json> <output_dir>"
        )
    var request_path = (
        String(args[2]) if decode_mode else String(args[1])
    )
    var out_dir = (
        String(args[3]) if decode_mode else String(args[2])
    )
    _mkdir(out_dir)
    _write_ltx2_status(
        out_dir, String("running"),
        String("decode_preflight") if decode_mode else String("preflight"),
        0, 0,
        String("Validating fresh LTX2 decode")
        if decode_mode else String("Validating LTX2 request"),
    )
    try:
        if decode_mode:
            _run_decode(request_path, out_dir)
        else:
            _run_request(request_path, out_dir)
    except e:
        _write_ltx2_status(
            out_dir, String("failed"), String("failed"), 0, 0, String(e)
        )
        raise Error(String(e))
