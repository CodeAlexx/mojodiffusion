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
    _mkdir, _write_ltx2_status, run_request_profile,
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

    def __init__(out self, path: String, has_negative: Bool):
        self.path = path.copy()
        self.has_negative = has_negative


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
        return ResolvedCaps(authored_path, False)

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
    return ResolvedCaps(tensor_path, has_negative)


def _configure_loras(obj: JSONValue) raises:
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
        _setenv(
            String("LTX2_TRAINED_LORA_") + String(i),
            _resolve_lora_path(name),
        )
        _setenv(
            String("LTX2_TRAINED_LORA_MULT_") + String(i), String(weight)
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
        neg_path = _resolve_caps_sidecar(
            authored_neg, prompt, negative
        ).path
    elif caps.has_negative:
        neg_path = caps.path.copy()
    elif negative.byte_length() > 0:
        raise Error(
            "LTX2 request: negative prompt was supplied without matching negative conditioning"
        )

    var seed = _require_int(obj, String("seed"))
    if seed < 0:
        raise Error("LTX2 request: seed must be >= 0")
    _configure_loras(obj)
    _mkdir(out_dir)
    run_request_profile(
        _require_int(obj, String("width")),
        _require_int(obj, String("height")),
        _require_int(obj, String("frames")),
        _require_int(obj, String("steps")),
        UInt64(seed),
        _require_number(obj, String("fps")),
        _require_string(obj, String("sampler")),
        _require_string(obj, String("scheduler")),
        caps.path,
        neg_path,
        _optional_bool(obj, String("include_audio"), False),
        out_dir,
    )


def main() raises:
    var args = argv()
    if len(args) != 3:
        raise Error(
            "usage: ltx2_request_cli <serenity.genparams.v1.json> <output_dir>"
        )
    var request_path = String(args[1])
    var out_dir = String(args[2])
    _mkdir(out_dir)
    _write_ltx2_status(
        out_dir, String("running"), String("preflight"), 0, 0,
        String("Validating LTX2 request"),
    )
    try:
        _run_request(request_path, out_dir)
    except e:
        _write_ltx2_status(
            out_dir, String("failed"), String("failed"), 0, 0, String(e)
        )
        raise Error(String(e))
