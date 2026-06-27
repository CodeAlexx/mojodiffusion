# serenitymojo.serve.serenity_daemon — the SerenityUI generation daemon.
#
# A pure-Mojo localhost HTTP + WebSocket server (default 127.0.0.1:7801) on the
# MOJO-libs net/http stack (single-threaded epoll event loop, the
# examples/main.mojo pattern). Endpoints:
#
#   POST /v1/generate     {model,prompt,negative,width,height,steps,seed,cfg,
#                          lora:[{name,weight}]} -> {job_id, queue_position}
#   GET  /v1/jobs         JSON array of all jobs
#   GET  /v1/job/<id>     one job
#   GET  /v1/gallery      list generated PNGs + embedded genparams
#   GET  /v1/gallery/<id> read one generated PNG's embedded genparams
#   GET  /v1/gallery/read?path=<png> read/import any local PNG's genparams
#   POST /v1/reorder[/<id>] {position|before_id} reorder a queued job
#   POST /v1/remove[/<id>]  remove a queued job before it starts
#   GET/POST /v1/state      load/save last UI state
#   GET/POST /v1/presets    list/save named generation presets
#   GET  /v1/video          video readiness/status contract
#   POST /v1/video          bounded video smoke gate when runner is built
#   GET  /v1/video/probe?path=<mp4> inspect a local MP4 artifact
#   POST /v1/cancel/<id>  cancel a queued job / signal the running one
#   WS   /v1/progress     pushes {job_id,state,step,total,progress[,output_path,
#                          error,preview]} on every job-state change
#   GET  /v1/health       {status, backend, model}
#
# ONE worker, jobs run serially (single-GPU reality). Mojo gives no threads
# here, so the worker runs INSIDE the event loop: each loop tick calls
# backend.step() once (the stub sleeps ~100 ms per step inside step()), so
# HTTP latency while a job runs is bounded by one step (~100 ms). The real
# backend must keep step() similarly bounded — see backend.mojo.
#
# Every finished job (done/failed/cancelled) is appended to a pure-Mojo SQLite
# db at output/serenity_daemon/jobs.db (the gallery-index seam).

from std.ffi import external_call
from std.memory import alloc
from std.sys import argv
from std.time import perf_counter_ns

from net.poll import Epoll, EPOLLIN, EPOLLET, EVENT_SIZE, EAGAIN, rd_u64
from net.socket import Socket
from net.signals import install_signal_fd
from net.syscalls import (
    BytePtr, sys_socket, sys_bind, sys_listen, sys_accept, sys_recv, sys_send,
    sys_close, sys_fcntl, sys_setsockopt, errno, errno_str,
    AF_INET, SOCK_STREAM, SOL_SOCKET, SO_REUSEADDR,
    MSG_NOSIGNAL, F_GETFL, F_SETFL, O_NONBLOCK,
)
from http.request import (
    Request, parse_request, is_request_complete, request_consumed_len, byte_substr,
)
from http.response import Response, http_date
from http.websocket import (
    handshake_response, decode_frame, encode_text, encode_close, encode_pong,
    WsReassembler, OP_TEXT, OP_BINARY, OP_CLOSE, OP_PING,
)
from image.png import read_png_text, decode_png, encode_png
from image.studio_ops import resize_lanczos
from json.parser import loads
from json.serialize import dumps
from json.value import JSONValue
from sqlite.db import Database
from sqlite.writer import DbWriter
from sqlite.value import Value

from serenitymojo.sampling.sampler_registry import (
    default_generation_model, default_sampler_for_backend,
    default_scheduler_for_backend, sampler_backend_for_model,
    sampler_admission_for_backend, scheduler_admission_for_backend,
    swarmui_sampler_registry_json,
)
from serenitymojo.io.ffi import sys_open, sys_close, file_size, O_RDONLY
from serenitymojo.serve.backend import GenBackend, JobParams, LoraSpec, StepResult
from serenitymojo.serve.model_scan import (
    ScanEntry, scan_checkpoints, scan_loras, _read_text_file,
)
from serenitymojo.serve.video_api import (
    ltx2_staged_smoke_video_result, probe_video_file, video_readiness_doc,
)
from serenitymojo.serve.workflow_graph import apply_workflow_params as apply_workflow_graph_params
from serenitymojo.serve.stub_backend import StubBackend
from serenitymojo.serve.zimage_backend import ZImageBackend
from serenitymojo.serve.dispatch_backend import DispatchBackend
from serenitymojo.serve.process_isolated_backend import ProcessIsolatedBackend
from serenitymojo.serve.worker import run_worker

comptime DEFAULT_PORT = 7801
comptime MAX_EVENTS = 64
comptime READ_CHUNK = 65536
comptime MAX_REQUEST_BYTES = 1048576  # 1 MiB
comptime TICK_MS = 50                 # epoll wait timeout -> worker tick cadence
comptime OUT_DIR = "output/serenity_daemon"
comptime DB_PATH = "output/serenity_daemon/jobs.db"
comptime STATE_DIR = "output/serenity_daemon/state"
comptime STATE_PATH = "output/serenity_daemon/state/last_state.json"
comptime PRESETS_PATH = "output/serenity_daemon/state/presets.json"
comptime GALLERY_STATE_PATH = "output/serenity_daemon/state/gallery.json"
comptime GALLERY_THUMB_DIR = "output/serenity_daemon/thumbnails"
comptime GENPARAMS_TEXT_KEY = "serenity.genparams.v1"
comptime DB_PARAMS_MAX = 2048         # params_json cap in the db row (F1): the
                                      # pure-Mojo DbWriter has no overflow pages
                                      # (one row must fit one 4096-byte page);
                                      # the FULL params live in the PNG tEXt —
                                      # the db row is just the gallery index.
comptime SCAN_PNG_TMP = "output/serenity_daemon/state/gallery_scan.txt"


# ── small libc helpers ───────────────────────────────────────────────────────
def _system(cmd: String) -> Int:
    var n = cmd.byte_length()
    var buf = alloc[UInt8](n + 1)
    var src = cmd.as_bytes()
    for i in range(n):
        buf[i] = src[i]
    buf[n] = 0
    var status = Int(external_call["system", Int32](BytePtr(unsafe_from_address=Int(buf))))
    buf.free()
    return status


def set_nonblocking_fd(fd: Int32):
    var fl = sys_fcntl(fd, F_GETFL, Int32(0))
    _ = sys_fcntl(fd, F_SETFL, fl | O_NONBLOCK)


def listen_localhost(port: Int) raises -> Socket:
    """Bind+listen on 127.0.0.1:port (loopback ONLY — unlike net.tcp's
    TCPListener, which binds INADDR_ANY)."""
    var fd = sys_socket(AF_INET, SOCK_STREAM, Int32(0))
    if fd < 0:
        raise Error("socket() failed: " + errno_str())
    var s = Socket(fd)
    s.set_reuse()
    var addr = alloc[UInt8](16)
    var ap = BytePtr(unsafe_from_address=Int(addr))
    for i in range(16):
        ap[i] = 0
    ap[0] = 2  # sin_family = AF_INET
    ap[2] = UInt8((port >> 8) & 0xFF)
    ap[3] = UInt8(port & 0xFF)
    ap[4] = 127  # sin_addr = 127.0.0.1 (network order)
    ap[5] = 0
    ap[6] = 0
    ap[7] = 1
    var brc = sys_bind(fd, ap, Int32(16))
    addr.free()
    if brc < 0:
        raise Error("bind(127.0.0.1:" + String(port) + ") failed: " + errno_str())
    if sys_listen(fd, Int32(128)) < 0:
        raise Error("listen() failed: " + errno_str())
    return s^


def send_all_fd(fd: Int32, data: String):
    var n = data.byte_length()
    if n == 0:
        return
    var buf = alloc[UInt8](n)
    var src = data.as_bytes()
    for i in range(n):
        buf[i] = src[i]
    var p = BytePtr(unsafe_from_address=Int(buf))
    var total = 0
    while total < n:
        var sent = sys_send(fd, p + total, n - total, MSG_NOSIGNAL)
        if sent > 0:
            total += sent
        elif sent < 0 and errno() == EAGAIN:
            continue
        else:
            break
    buf.free()


# ── job records ──────────────────────────────────────────────────────────────
struct JobRecord(Copyable, Movable):
    var params: JobParams
    var created: String
    var state: String        # queued | running | done | failed | cancelled
    var progress: Int        # 0..100
    var step: Int
    var total: Int
    var output_path: String
    var error: String
    var cancel_requested: Bool

    def __init__(out self, var params: JobParams, created: String):
        self.total = params.steps
        self.params = params^
        self.created = created
        self.state = String("queued")
        self.progress = 0
        self.step = 0
        self.output_path = String("")
        self.error = String("")
        self.cancel_requested = False

    def is_terminal(self) -> Bool:
        return self.state == "done" or self.state == "failed" or self.state == "cancelled"


def _pad4(n: Int) -> String:
    var s = String(n)
    while s.byte_length() < 4:
        s = "0" + s
    return s


# ── request-body parsing / validation ────────────────────────────────────────
def _opt_int(obj: JSONValue, key: String, dflt: Int, lo: Int, hi: Int) raises -> Int:
    if not obj.contains(key) or obj[key].is_null():
        return dflt
    if not obj[key].is_int():
        raise Error("'" + key + "' must be an integer")
    var n = obj[key].as_int()
    if n < lo or n > hi:
        raise Error("'" + key + "' out of range [" + String(lo) + ".." + String(hi) + "]")
    return n


def _opt_num(obj: JSONValue, key: String, dflt: Float64, lo: Float64, hi: Float64) raises -> Float64:
    if not obj.contains(key) or obj[key].is_null():
        return dflt
    if not obj[key].is_number():
        raise Error("'" + key + "' must be a number")
    var v = obj[key].as_float()
    if v < lo or v > hi:
        raise Error("'" + key + "' out of range")
    return v


def _opt_str(obj: JSONValue, key: String, dflt: String) raises -> String:
    if not obj.contains(key) or obj[key].is_null():
        return dflt
    if not obj[key].is_string():
        raise Error("'" + key + "' must be a string")
    return obj[key].as_string()


def _opt_bool(obj: JSONValue, key: String, dflt: Bool) raises -> Bool:
    if not obj.contains(key) or obj[key].is_null():
        return dflt
    if not obj[key].is_bool():
        raise Error("'" + key + "' must be a boolean")
    return obj[key].as_bool()


struct PromptSyntaxResult(Copyable, Movable):
    var raw: String
    var resolved: String
    var loras: List[LoraSpec]
    var weighted: JSONValue
    var randoms: JSONValue
    var wildcards: JSONValue

    def __init__(out self, raw: String):
        self.raw = raw
        self.resolved = String("")
        self.loras = List[LoraSpec]()
        self.weighted = JSONValue.new_array()
        self.randoms = JSONValue.new_array()
        self.wildcards = JSONValue.new_array()


def _has_lora(loras: List[LoraSpec], name: String) -> Bool:
    for i in range(len(loras)):
        if loras[i].name == name:
            return True
    return False


def _find_next_byte(s: String, start: Int, target: Int) -> Int:
    var b = s.as_bytes()
    for i in range(start, s.byte_length()):
        if Int(b[i]) == target:
            return i
    return -1


def _find_next_double_underscore(s: String, start: Int) -> Int:
    var b = s.as_bytes()
    var i = start
    while i + 1 < s.byte_length():
        if Int(b[i]) == 95 and Int(b[i + 1]) == 95:
            return i
        i += 1
    return -1


def _choose_index(seed: Int, salt: Int, count: Int) -> Int:
    if count <= 0:
        return 0
    return (seed + salt) % count


def _choose_option_csv(options: String, seed: Int, salt: Int) -> String:
    var parts = options.split(",")
    var count = 0
    for i in range(len(parts)):
        if String(String(parts[i]).strip()) != "":
            count += 1
    if count == 0:
        return String("")
    var target = _choose_index(seed, salt, count)
    var seen = 0
    for i in range(len(parts)):
        var opt = String(String(parts[i]).strip())
        if opt == "":
            continue
        if seen == target:
            return opt^
        seen += 1
    return String("")


def _safe_wildcard_name(name: String) -> Bool:
    return name != "" and name.find("..") < 0 and name.find("/") < 0 and name.find("\\") < 0


def _choose_nonempty_line(text: String, seed: Int, salt: Int) -> String:
    var lines = text.split("\n")
    var count = 0
    for i in range(len(lines)):
        if String(String(lines[i]).strip()) != "":
            count += 1
    if count == 0:
        return String("")
    var target = _choose_index(seed, salt, count)
    var seen = 0
    for i in range(len(lines)):
        var line = String(String(lines[i]).strip())
        if line == "":
            continue
        if seen == target:
            return line^
        seen += 1
    return String("")


def _wildcard_value(name: String, seed: Int, salt: Int) -> String:
    if not _safe_wildcard_name(name):
        return String("")
    var path = String("/home/alex/.serenity/wildcards/") + name + ".txt"
    try:
        return _choose_nonempty_line(_read_text_file(path), seed, salt)
    except:
        pass
    path = String("wildcards/") + name + ".txt"
    try:
        return _choose_nonempty_line(_read_text_file(path), seed, salt)
    except:
        pass
    return String("")


def _append_prompt_weight(mut result: PromptSyntaxResult, text: String, weight: Float64):
    var ent = JSONValue.new_object()
    ent.set("text", JSONValue.from_string(text))
    ent.set("weight", JSONValue.from_float(weight))
    result.weighted.append(ent^)


def _append_prompt_random(
    mut result: PromptSyntaxResult, expr: String, selected: String, index: Int,
):
    var ent = JSONValue.new_object()
    ent.set("expr", JSONValue.from_string(expr))
    ent.set("selected", JSONValue.from_string(selected))
    ent.set("index", JSONValue.from_int(index))
    result.randoms.append(ent^)


def _append_prompt_wildcard(
    mut result: PromptSyntaxResult, name: String, selected: String, resolved: Bool,
):
    var ent = JSONValue.new_object()
    ent.set("name", JSONValue.from_string(name))
    ent.set("selected", JSONValue.from_string(selected))
    ent.set("resolved", JSONValue.from_bool(resolved))
    result.wildcards.append(ent^)


def _append_prompt_lora(mut result: PromptSyntaxResult, name: String, weight: Float64):
    if name == "":
        return
    result.loras.append(LoraSpec(name, weight))


def _prompt_syntax_json(result: PromptSyntaxResult) -> JSONValue:
    var o = JSONValue.new_object()
    o.set("schema", JSONValue.from_string(String("serenity.prompt_syntax.v1")))
    o.set("parser", JSONValue.from_string(String("daemon")))
    o.set("conditioning_weights_applied", JSONValue.from_bool(False))
    o.set("weighted", result.weighted.copy())
    o.set("random", result.randoms.copy())
    o.set("wildcards", result.wildcards.copy())
    var la = JSONValue.new_array()
    for i in range(len(result.loras)):
        var lo = JSONValue.new_object()
        lo.set("name", JSONValue.from_string(result.loras[i].name))
        lo.set("weight", JSONValue.from_float(result.loras[i].weight))
        la.append(lo^)
    o.set("lora_tags", la^)
    return o^


def _reject_unapplied_prompt_weights(result: PromptSyntaxResult) raises:
    if result.weighted.length() > 0:
        raise Error(
            "weighted prompt syntax is not supported by this product path yet; "
            + "prompt weights would be persisted with conditioning_weights_applied=false"
        )


def _parse_prompt_syntax(raw: String, seed: Int) raises -> PromptSyntaxResult:
    var result = PromptSyntaxResult(raw)
    var b = raw.as_bytes()
    var i = 0
    while i < raw.byte_length():
        var c = Int(b[i])
        if c == 60:  # '<'
            var end = _find_next_byte(raw, i + 1, 62)  # '>'
            if end > i:
                var content = String(byte_substr(raw, i + 1, end).strip())
                if content.startswith("lora:"):
                    var rest = byte_substr(content, 5, content.byte_length())
                    var colon = rest.find(":")
                    var name = rest.copy()
                    var weight = 1.0
                    if colon >= 0:
                        name = String(byte_substr(rest, 0, colon).strip())
                        weight = Float64(String(byte_substr(rest, colon + 1, rest.byte_length()).strip()))
                    else:
                        name = String(name.strip())
                    _append_prompt_lora(result, name, weight)
                    i = end + 1
                    continue
                if content.startswith("random:"):
                    var opts = byte_substr(content, 7, content.byte_length())
                    var selected = _choose_option_csv(opts, seed, i)
                    var count = 0
                    for part in opts.split(","):
                        if String(String(part).strip()) != "":
                            count += 1
                    var selected_index = _choose_index(seed, i, count)
                    result.resolved += selected
                    _append_prompt_random(result, content, selected, selected_index)
                    i = end + 1
                    continue
                if content.startswith("wildcard:"):
                    var name = String(byte_substr(content, 9, content.byte_length()).strip())
                    var selected = _wildcard_value(name, seed, i)
                    var ok = selected != ""
                    if not ok:
                        selected = name.copy()
                    result.resolved += selected
                    _append_prompt_wildcard(result, name, selected, ok)
                    i = end + 1
                    continue
        elif c == 40:  # '('
            var endp = _find_next_byte(raw, i + 1, 41)  # ')'
            if endp > i:
                var inner = byte_substr(raw, i + 1, endp)
                var colon = inner.find(":")
                if colon > 0:
                    try:
                        var text = String(byte_substr(inner, 0, colon).strip())
                        var weight = Float64(String(byte_substr(inner, colon + 1, inner.byte_length()).strip()))
                        if text != "":
                            result.resolved += text
                            _append_prompt_weight(result, text, weight)
                            i = endp + 1
                            continue
                    except:
                        pass
        elif c == 95 and i + 1 < raw.byte_length() and Int(b[i + 1]) == 95:
            var endu = _find_next_double_underscore(raw, i + 2)
            if endu > i + 2:
                var name = String(byte_substr(raw, i + 2, endu).strip())
                var selected = _wildcard_value(name, seed, i)
                var ok = selected != ""
                if not ok:
                    selected = name.copy()
                result.resolved += selected
                _append_prompt_wildcard(result, name, selected, ok)
                i = endu + 2
                continue
        result.resolved += byte_substr(raw, i, i + 1)
        i += 1
    return result^


def _json_prompt_to_string(value: JSONValue, field_name: String) raises -> String:
    if value.is_string():
        return value.as_string()
    if value.is_object() or value.is_array():
        return dumps(value)
    raise Error(field_name + " must be a string or JSON object/array")


def _set_ideogram_prompt_field(
    mut obj: JSONValue, key: String, value: String, source: String
) raises:
    if obj.contains(key) and not obj[key].is_null():
        if not obj[key].is_string():
            raise Error("ideogram4: " + key + " must be a string when " + source + " is provided")
        if obj[key].as_string() != value:
            raise Error("ideogram4: " + source + " conflicts with " + key)
        return
    obj.set(key, JSONValue.from_string(value))


def _normalize_ideogram4_structured_prompt(mut obj: JSONValue) raises:
    """Accept Ideogram structured JSON captions as the prompt string.

    Ideogram 4 consumes the JSON caption as text. Bounding boxes stay inside the
    authored JSON prompt instead of becoming flat generation controls.
    """
    var source = String("")
    var raw = String("")
    if obj.contains("prompt_json") and not obj["prompt_json"].is_null():
        source = String("prompt_json")
        raw = _json_prompt_to_string(obj["prompt_json"], source)
    elif (
        obj.contains("prompt")
        and not obj["prompt"].is_null()
        and (obj["prompt"].is_object() or obj["prompt"].is_array())
    ):
        source = String("prompt")
        raw = _json_prompt_to_string(obj["prompt"], source)
    elif (
        obj.contains("prompt_raw")
        and not obj["prompt_raw"].is_null()
        and (obj["prompt_raw"].is_object() or obj["prompt_raw"].is_array())
    ):
        source = String("prompt_raw")
        raw = _json_prompt_to_string(obj["prompt_raw"], source)

    if source == "":
        return
    if raw == "":
        raise Error("ideogram4: " + source + " must be non-empty")
    _set_ideogram_prompt_field(obj, String("prompt"), raw, source)
    _set_ideogram_prompt_field(obj, String("prompt_raw"), raw, source)


def _looks_like_ideogram4_structured_prompt(prompt: String) -> Bool:
    var s = String(prompt.strip())
    if not s.startswith("{"):
        return False
    return (
        s.find(String("\"elements\"")) >= 0
        or s.find(String("\"bbox\"")) >= 0
        or s.find(String("\"compositional_deconstruction\"")) >= 0
        or s.find(String("\"high_level_description\"")) >= 0
    )


def parse_generate(
    body: String, job_id: String, out_dir: String, default_model: String
) raises -> JobParams:
    """Validate POST /v1/generate's JSON body into JobParams (raises -> 422;
    a "[501]"-prefixed error -> 501). `default_model` is the serving backend's
    name so genparams never claim "stub" on a real-model daemon (F9)."""
    var obj = loads(body)
    if not obj.is_object():
        raise Error("body must be a JSON object")
    apply_workflow_graph_params(obj)
    var p = JobParams()
    p.job_id = job_id
    p.out_dir = out_dir
    p.model = _opt_str(obj, "model", default_generation_model(default_model))
    var sampler_backend = sampler_backend_for_model(p.model, default_model)
    if sampler_backend == "disabled":
        raise Error(
            "[501] model '" + p.model + "' is known but not runnable by this "
            + "daemon slice; use the bounded model-specific gate or keep it as "
            + "metadata/preflight-only until artifact, timing, VRAM, and sampler "
            + "evidence passes"
        )
    if sampler_backend == "ideogram4":
        _normalize_ideogram4_structured_prompt(obj)
    if not obj.contains("prompt") or not obj["prompt"].is_string():
        raise Error("'prompt' (string) is required")
    var prompt_input = obj["prompt"].as_string()
    if prompt_input == "":
        raise Error("'prompt' must be non-empty")
    p.negative = _opt_str(obj, "negative", String(""))
    var default_width = 512
    var default_height = 512
    if (
        sampler_backend == "zimage"
        or sampler_backend == "qwenimage"
        or sampler_backend == "ideogram4"
        or sampler_backend == "flux2"
        or sampler_backend == "sdxl"
        or sampler_backend == "anima"
        or sampler_backend == "sd3"
    ):
        default_width = 1024
        default_height = 1024
    p.width = _opt_int(obj, "width", default_width, 16, 2048)
    p.height = _opt_int(obj, "height", default_height, 16, 2048)
    var default_steps = 20
    if sampler_backend == "sdxl" or sampler_backend == "anima":
        default_steps = 30
    var default_seed = 0
    if sampler_backend == "sdxl":
        default_seed = 42
    var default_cfg = 4.5
    if sampler_backend == "sdxl":
        default_cfg = 7.5
    p.steps = _opt_int(obj, "steps", default_steps, 1, 500)
    p.seed = _opt_int(obj, "seed", default_seed, 0, 4294967295)
    p.cfg = _opt_num(obj, "cfg", default_cfg, 0.0, 50.0)
    p.cfg_override = _opt_num(obj, "cfg_override", -1.0, -1.0, 50.0)
    p.cfg_override_start_percent = _opt_num(obj, "cfg_override_start_percent", 0.0, 0.0, 1.0)
    p.cfg_override_end_percent = _opt_num(obj, "cfg_override_end_percent", 1.0, 0.0, 1.0)
    p.sigma_shift = _opt_num(obj, "sigma_shift", 3.0, 0.000001, 100.0)
    # P9/P10: prompt syntax is parsed in the daemon product path. Backends get
    # resolved plain text; the raw text plus parser metadata persist in PNG
    # genparams so reuse-params can restore the authored prompt.
    var prompt_raw = _opt_str(obj, "prompt_raw", prompt_input)
    var prompt_syntax = PromptSyntaxResult(prompt_raw)
    if sampler_backend == "ideogram4" and _looks_like_ideogram4_structured_prompt(prompt_raw):
        prompt_syntax.resolved = prompt_raw.copy()
        p.prompt = prompt_raw.copy()
    else:
        prompt_syntax = _parse_prompt_syntax(prompt_raw, p.seed)
        _reject_unapplied_prompt_weights(prompt_syntax)
        p.prompt = prompt_syntax.resolved
    if p.prompt == "":
        raise Error("'prompt' resolved to an empty prompt")
    # Sampler-facing fields ride the typed backend contract as well as the
    # canonical genparams JSON. Backends must execute them or fail loud.
    p.sampler = _opt_str(obj, "sampler", default_sampler_for_backend(sampler_backend))
    p.scheduler = _opt_str(obj, "scheduler", default_scheduler_for_backend(sampler_backend))
    if p.sampler == "":
        p.sampler = default_sampler_for_backend(sampler_backend)
    if p.scheduler == "":
        p.scheduler = default_scheduler_for_backend(sampler_backend)
    p.variation_seed = _opt_int(obj, "variation_seed", 0, 0, 4294967295)
    p.variation_strength = _opt_num(obj, "variation_strength", 0.0, 0.0, 1.0)
    p.images = _opt_int(obj, "images", 1, 1, 64)
    if p.seed + p.images - 1 > 4294967295:
        raise Error("'seed + images - 1' must be <= 4294967295")
    p.image_index = 0
    p.image_count = p.images
    p.workflow_save_prefix = _opt_str(obj, "workflow_save_prefix", String(""))
    # P7 img2img: init image path + creativity (0..1). The backend decides
    # whether/how to honor them (stub echoes; zimage does real img2img).
    p.init_image = _opt_str(obj, "init_image", String(""))
    p.mask_image = _opt_str(obj, "mask_image", String(""))
    p.lanpaint_mask_channel = _opt_str(obj, "lanpaint_mask_channel", String(""))
    p.inpaint_conditioning_image = _opt_str(obj, "inpaint_conditioning_image", String(""))
    p.inpaint_conditioning_mask = _opt_str(obj, "inpaint_conditioning_mask", String(""))
    p.inpaint_conditioning_noise_mask = _opt_bool(obj, "inpaint_conditioning_noise_mask", False)
    p.qwen_edit_conditioning_image = _opt_str(obj, "qwen_edit_conditioning_image", String(""))
    p.sample_caps_pos = _opt_str(obj, "sample_caps_pos", String(""))
    if p.sample_caps_pos == String(""):
        p.sample_caps_pos = _opt_str(obj, "caps_pos", String(""))
    if p.sample_caps_pos == String(""):
        p.sample_caps_pos = _opt_str(obj, "caps_positive", String(""))
    p.sample_caps_neg = _opt_str(obj, "sample_caps_neg", String(""))
    if p.sample_caps_neg == String(""):
        p.sample_caps_neg = _opt_str(obj, "caps_neg", String(""))
    if p.sample_caps_neg == String(""):
        p.sample_caps_neg = _opt_str(obj, "caps_negative", String(""))
    p.conditioning_mask_image = _opt_str(obj, "conditioning_mask_image", String(""))
    p.conditioning_mask_channel = _opt_str(obj, "conditioning_mask_channel", String(""))
    p.conditioning_mask_strength = _opt_num(obj, "conditioning_mask_strength", -1.0, -1.0, 10.0)
    p.conditioning_mask_set_area_to_bounds = _opt_bool(obj, "conditioning_mask_set_area_to_bounds", False)
    p.outpaint_left = _opt_int(obj, "outpaint_left", -1, -1, 4096)
    p.outpaint_top = _opt_int(obj, "outpaint_top", -1, -1, 4096)
    p.outpaint_right = _opt_int(obj, "outpaint_right", -1, -1, 4096)
    p.outpaint_bottom = _opt_int(obj, "outpaint_bottom", -1, -1, 4096)
    p.outpaint_feathering = _opt_int(obj, "outpaint_feathering", -1, -1, 4096)
    p.threshold_mask_value = _opt_num(obj, "threshold_mask_value", -1.0, -1.0, 1.0)
    p.threshold_mask_operator = _opt_str(obj, "threshold_mask_operator", String(""))
    p.lanpaint_mask_blend_overlap = _opt_int(obj, "lanpaint_mask_blend_overlap", -1, -1, 4096)
    p.lanpaint_num_steps = _opt_int(obj, "lanpaint_num_steps", -1, -1, 4096)
    p.lanpaint_lambda = _opt_num(obj, "lanpaint_lambda", -1.0, -1.0, 100000.0)
    p.lanpaint_step_size = _opt_num(obj, "lanpaint_step_size", -1.0, -1.0, 100000.0)
    p.lanpaint_beta = _opt_num(obj, "lanpaint_beta", -1.0, -1.0, 100000.0)
    p.lanpaint_friction = _opt_num(obj, "lanpaint_friction", -1.0, -1.0, 100000.0)
    p.lanpaint_prompt_mode = _opt_str(obj, "lanpaint_prompt_mode", String(""))
    p.lanpaint_inpainting_mode = _opt_str(obj, "lanpaint_inpainting_mode", String(""))
    p.lanpaint_add_noise = _opt_str(obj, "lanpaint_add_noise", String(""))
    p.lanpaint_noise_seed = _opt_int(obj, "lanpaint_noise_seed", -1, -1, 4294967295)
    p.lanpaint_start_at_step = _opt_int(obj, "lanpaint_start_at_step", -1, -1, 1000000)
    p.lanpaint_end_at_step = _opt_int(obj, "lanpaint_end_at_step", -1, -1, 1000000)
    p.lanpaint_return_with_leftover_noise = _opt_str(obj, "lanpaint_return_with_leftover_noise", String(""))
    p.lanpaint_early_stop = _opt_int(obj, "lanpaint_early_stop", -1, -1, 1)
    p.lanpaint_inner_threshold = _opt_num(obj, "lanpaint_inner_threshold", -1.0, -1.0, 100000.0)
    p.lanpaint_inner_patience = _opt_int(obj, "lanpaint_inner_patience", -1, -1, 1000000)
    p.reference_image = _opt_str(obj, "reference_image", String(""))
    p.reference_latent_method = _opt_str(obj, "reference_latent_method", String(""))
    p.reference_latent_count = _opt_int(obj, "reference_latent_count", 0, 0, 64)
    p.creativity = _opt_num(obj, "creativity", 0.5, 0.0, 1.0)
    if obj.contains("lora") and not obj["lora"].is_null():
        var arr = obj["lora"]
        if not arr.is_array():
            raise Error("'lora' must be an array of {name, weight}")
        for i in range(arr.length()):
            var ent = arr[i]
            if not ent.is_object():
                raise Error("'lora[" + String(i) + "]' must be an object")
            if not ent.contains("name") or not ent["name"].is_string():
                raise Error("'lora[" + String(i) + "].name' (string) is required")
            var w = _opt_num(ent, "weight", 1.0, -10.0, 10.0)
            p.loras.append(LoraSpec(ent["name"].as_string(), w))
    for i in range(len(prompt_syntax.loras)):
        if not _has_lora(p.loras, prompt_syntax.loras[i].name):
            p.loras.append(prompt_syntax.loras[i].copy())
    if sampler_backend == "ideogram4":
        var sampler_admission = sampler_admission_for_backend(String("ideogram4"), p.sampler)
        if not sampler_admission.supported:
            raise Error(
                String("ideogram4: unsupported sampler '") + p.sampler
                + String("'; ") + sampler_admission.reason
            )
        var scheduler_admission = scheduler_admission_for_backend(String("ideogram4"), p.scheduler)
        if not scheduler_admission.supported:
            raise Error(
                String("ideogram4: unsupported scheduler '") + p.scheduler
                + String("'; ") + scheduler_admission.reason
            )
        if not (p.width == 1024 and p.height == 1024):
            raise Error(
                String("ideogram4: unsupported size ") + String(p.width)
                + "x" + String(p.height)
                + " -- only 1024x1024 is served by the current fixed-shape path"
            )
        if p.negative.byte_length() > 0:
            raise Error("ideogram4: negative prompt is not supported in this bounded slice")
        if len(p.loras) > 0:
            raise Error("ideogram4: LoRA is not supported in this bounded slice")
        if p.inpaint_conditioning_image.byte_length() > 0 or p.inpaint_conditioning_mask.byte_length() > 0:
            raise Error("ideogram4: InpaintModelConditioning concat conditioning is not supported in this bounded slice")
        if p.qwen_edit_conditioning_image.byte_length() > 0:
            raise Error("ideogram4: TextEncodeQwenImageEdit image conditioning is not supported in this bounded slice")
        if (
            p.conditioning_mask_image.byte_length() > 0
            or p.conditioning_mask_channel.byte_length() > 0
            or p.conditioning_mask_strength >= 0.0
            or p.conditioning_mask_set_area_to_bounds
        ):
            raise Error("ideogram4: ConditioningSetMask/regional conditioning is not supported in this bounded slice")
        if p.init_image.byte_length() > 0:
            raise Error("ideogram4: img2img/init image is not supported in this bounded slice")
        if p.mask_image.byte_length() > 0:
            raise Error("ideogram4: SetLatentNoiseMask/inpaint mask is not supported in this bounded slice")
        if p.reference_image.byte_length() > 0 or p.reference_latent_count > 0:
            raise Error("ideogram4: ReferenceLatent/reference image conditioning is not supported in this bounded slice")
        if p.creativity != 0.5:
            raise Error("ideogram4: creativity/denoise control is not supported in this bounded txt2img slice")
        if p.variation_strength > 0.0:
            raise Error("ideogram4: variation noise is not supported in this bounded slice")
        if p.cfg <= 0.0:
            raise Error("ideogram4: cfg must be positive")
        if p.cfg_override >= 0.0:
            if p.cfg_override == 0.0:
                raise Error("ideogram4: cfg_override must be positive when set")
            if p.cfg_override_start_percent > p.cfg_override_end_percent:
                raise Error("ideogram4: cfg_override_start_percent must be <= cfg_override_end_percent")

    # canonical full-param JSON: persisted to the sidecar + jobs.db
    # (key order mirrors the UI's serenity.genparams.v1 emitter, with the
    # server-added job_id second after schema)
    var o = JSONValue.new_object()
    o.set("schema", JSONValue.from_string(String("serenity.genparams.v1")))
    o.set("job_id", JSONValue.from_string(p.job_id))
    o.set("model", JSONValue.from_string(p.model))
    o.set("prompt", JSONValue.from_string(p.prompt))
    o.set("prompt_raw", JSONValue.from_string(prompt_raw))
    o.set("prompt_syntax", _prompt_syntax_json(prompt_syntax))
    o.set("negative", JSONValue.from_string(p.negative))
    o.set("width", JSONValue.from_int(p.width))
    o.set("height", JSONValue.from_int(p.height))
    o.set("steps", JSONValue.from_int(p.steps))
    o.set("seed", JSONValue.from_int(p.seed))
    o.set("cfg", JSONValue.from_float(p.cfg))
    o.set("cfg_override", JSONValue.from_float(p.cfg_override))
    o.set("cfg_override_start_percent", JSONValue.from_float(p.cfg_override_start_percent))
    o.set("cfg_override_end_percent", JSONValue.from_float(p.cfg_override_end_percent))
    o.set("sampler", JSONValue.from_string(p.sampler))
    o.set("scheduler", JSONValue.from_string(p.scheduler))
    o.set("sigma_shift", JSONValue.from_float(p.sigma_shift))
    o.set("variation_seed", JSONValue.from_int(p.variation_seed))
    o.set("variation_strength", JSONValue.from_float(p.variation_strength))
    o.set("images", JSONValue.from_int(p.images))
    o.set("image_index", JSONValue.from_int(p.image_index))
    o.set("image_count", JSONValue.from_int(p.image_count))
    o.set("workflow_save_prefix", JSONValue.from_string(p.workflow_save_prefix))
    o.set("init_image", JSONValue.from_string(p.init_image))
    o.set("mask_image", JSONValue.from_string(p.mask_image))
    o.set("lanpaint_mask_channel", JSONValue.from_string(p.lanpaint_mask_channel))
    o.set("inpaint_conditioning_image", JSONValue.from_string(p.inpaint_conditioning_image))
    o.set("inpaint_conditioning_mask", JSONValue.from_string(p.inpaint_conditioning_mask))
    o.set("inpaint_conditioning_noise_mask", JSONValue.from_bool(p.inpaint_conditioning_noise_mask))
    o.set("qwen_edit_conditioning_image", JSONValue.from_string(p.qwen_edit_conditioning_image))
    o.set("sample_caps_pos", JSONValue.from_string(p.sample_caps_pos))
    o.set("sample_caps_neg", JSONValue.from_string(p.sample_caps_neg))
    o.set("conditioning_mask_image", JSONValue.from_string(p.conditioning_mask_image))
    o.set("conditioning_mask_channel", JSONValue.from_string(p.conditioning_mask_channel))
    o.set("conditioning_mask_strength", JSONValue.from_float(p.conditioning_mask_strength))
    o.set("conditioning_mask_set_area_to_bounds", JSONValue.from_bool(p.conditioning_mask_set_area_to_bounds))
    o.set("outpaint_left", JSONValue.from_int(p.outpaint_left))
    o.set("outpaint_top", JSONValue.from_int(p.outpaint_top))
    o.set("outpaint_right", JSONValue.from_int(p.outpaint_right))
    o.set("outpaint_bottom", JSONValue.from_int(p.outpaint_bottom))
    o.set("outpaint_feathering", JSONValue.from_int(p.outpaint_feathering))
    o.set("threshold_mask_value", JSONValue.from_float(p.threshold_mask_value))
    o.set("threshold_mask_operator", JSONValue.from_string(p.threshold_mask_operator))
    o.set("lanpaint_mask_blend_overlap", JSONValue.from_int(p.lanpaint_mask_blend_overlap))
    o.set("lanpaint_num_steps", JSONValue.from_int(p.lanpaint_num_steps))
    o.set("lanpaint_lambda", JSONValue.from_float(p.lanpaint_lambda))
    o.set("lanpaint_step_size", JSONValue.from_float(p.lanpaint_step_size))
    o.set("lanpaint_beta", JSONValue.from_float(p.lanpaint_beta))
    o.set("lanpaint_friction", JSONValue.from_float(p.lanpaint_friction))
    o.set("lanpaint_prompt_mode", JSONValue.from_string(p.lanpaint_prompt_mode))
    o.set("lanpaint_inpainting_mode", JSONValue.from_string(p.lanpaint_inpainting_mode))
    o.set("lanpaint_add_noise", JSONValue.from_string(p.lanpaint_add_noise))
    o.set("lanpaint_noise_seed", JSONValue.from_int(p.lanpaint_noise_seed))
    o.set("lanpaint_start_at_step", JSONValue.from_int(p.lanpaint_start_at_step))
    o.set("lanpaint_end_at_step", JSONValue.from_int(p.lanpaint_end_at_step))
    o.set("lanpaint_return_with_leftover_noise", JSONValue.from_string(p.lanpaint_return_with_leftover_noise))
    o.set("lanpaint_early_stop", JSONValue.from_int(p.lanpaint_early_stop))
    o.set("lanpaint_inner_threshold", JSONValue.from_float(p.lanpaint_inner_threshold))
    o.set("lanpaint_inner_patience", JSONValue.from_int(p.lanpaint_inner_patience))
    o.set("reference_image", JSONValue.from_string(p.reference_image))
    o.set("reference_latent_method", JSONValue.from_string(p.reference_latent_method))
    o.set("reference_latent_count", JSONValue.from_int(p.reference_latent_count))
    o.set("creativity", JSONValue.from_float(p.creativity))
    var la = JSONValue.new_array()
    for i in range(len(p.loras)):
        var lo = JSONValue.new_object()
        lo.set("name", JSONValue.from_string(p.loras[i].name))
        lo.set("weight", JSONValue.from_float(p.loras[i].weight))
        la.append(lo^)
    o.set("lora", la^)
    _copy_optional_string_json_field(o, obj, String("params_source"))
    _copy_optional_string_json_field(o, obj, String("params_source_hash"))
    _copy_optional_string_json_field(o, obj, String("reused_from_gallery_id"))
    _copy_optional_string_json_field(o, obj, String("reused_from_path"))
    _copy_optional_string_json_field(o, obj, String("reused_from_job_id"))
    _copy_optional_string_json_field(o, obj, String("workflow_schema"))
    _copy_optional_string_json_field(o, obj, String("workflow_executor"))
    _copy_optional_string_json_field(o, obj, String("workflow_source"))
    _copy_json_field_if_present(o, obj, String("workflow_node_count"))
    _copy_json_field_if_present(o, obj, String("workflow_edge_count"))
    p.params_json = dumps(o)
    return p^


def params_json_for_image_job(
    base_json: String, job_id: String, seed: Int, image_index: Int, image_count: Int
) raises -> String:
    var o = loads(base_json)
    if not o.is_object():
        raise Error("internal error: canonical params_json is not an object")
    o.set("job_id", JSONValue.from_string(job_id))
    o.set("seed", JSONValue.from_int(seed))
    o.set("images", JSONValue.from_int(image_count))
    o.set("image_index", JSONValue.from_int(image_index))
    o.set("image_count", JSONValue.from_int(image_count))
    return dumps(o)


# ── JSON views of a job ──────────────────────────────────────────────────────
def job_json_value(j: JobRecord) raises -> JSONValue:
    var o = JSONValue.new_object()
    o.set("id", JSONValue.from_string(j.params.job_id))
    o.set("created", JSONValue.from_string(j.created))
    o.set("model", JSONValue.from_string(j.params.model))
    o.set("state", JSONValue.from_string(j.state))
    o.set("progress", JSONValue.from_int(j.progress))
    o.set("step", JSONValue.from_int(j.step))
    o.set("total", JSONValue.from_int(j.total))
    o.set("image_index", JSONValue.from_int(j.params.image_index))
    o.set("image_count", JSONValue.from_int(j.params.image_count))
    o.set("output_path", JSONValue.from_string(j.output_path))
    o.set("error", JSONValue.from_string(j.error))
    return o^


def prior_job_json_value(row: List[String]) raises -> JSONValue:
    var o = JSONValue.new_object()
    o.set("id", JSONValue.from_string(row[0]))
    o.set("created", JSONValue.from_string(row[1]))
    o.set("model", JSONValue.from_string(row[2]))
    o.set("state", JSONValue.from_string(row[4]))
    if _terminal_state(row[4]):
        o.set("progress", JSONValue.from_int(100))
    else:
        o.set("progress", JSONValue.from_int(0))
    o.set("step", JSONValue.from_int(0))
    o.set("total", JSONValue.from_int(0))
    o.set("image_index", JSONValue.from_int(0))
    o.set("image_count", JSONValue.from_int(1))
    o.set("output_path", JSONValue.from_string(row[5]))
    o.set("error", JSONValue.from_string(String("")))
    o.set("params_json", JSONValue.from_string(row[3]))
    if row[3] != "":
        try:
            var params = loads(row[3])
            if params.is_object():
                _copy_json_field_if_present(o, params, String("image_index"))
                _copy_json_field_if_present(o, params, String("image_count"))
                _copy_json_field_if_present(o, params, String("steps"))
                o.set("params", params^)
        except:
            pass
    o.set("history", JSONValue.from_bool(True))
    return o^


def _prior_row_index(prior: List[List[String]], id: String) -> Int:
    for i in range(len(prior)):
        if len(prior[i]) >= 6 and prior[i][0] == id:
            return i
    return -1


def event_json(j: JobRecord, preview: String, phase: String) raises -> String:
    var o = JSONValue.new_object()
    o.set("job_id", JSONValue.from_string(j.params.job_id))
    o.set("state", JSONValue.from_string(j.state))
    o.set("step", JSONValue.from_int(j.step))
    o.set("total", JSONValue.from_int(j.total))
    o.set("progress", JSONValue.from_int(j.progress))
    if j.output_path != "":
        o.set("output_path", JSONValue.from_string(j.output_path))
    if j.error != "":
        o.set("error", JSONValue.from_string(j.error))
    if preview != "":
        o.set("preview", JSONValue.from_string(preview))
    if phase != "":
        # sub-state of a long non-denoise tick: loading|encoding|decoding (F6)
        o.set("phase", JSONValue.from_string(phase))
    return dumps(o)


def broadcast(mut ws: Dict[Int, Bool], msg: String):
    """Push one event frame to every connected /v1/progress client."""
    var frame = encode_text(msg)
    for e in ws.items():
        send_all_fd(Int32(e.key), frame)


# ── gallery/read-params ─────────────────────────────────────────────────────
def _png_genparams(path: String) raises -> String:
    """Read the SwarmUI-style Serenity genparams tEXt value from one PNG."""
    var keywords = List[String]()
    var values = List[String]()
    read_png_text(path, keywords, values)
    for i in range(len(keywords)):
        if keywords[i] == GENPARAMS_TEXT_KEY:
            return values[i].copy()
    return String("")


def _png_genparams_or_empty(path: String) -> String:
    try:
        return _png_genparams(path)
    except:
        return String("")


def _ensure_gallery_dirs():
    _ = _system(
        String("mkdir -p '") + STATE_DIR + String("' '") + GALLERY_THUMB_DIR + String("'")
    )


def _path_exists_file(path: String) -> Bool:
    var fd = sys_open(path, O_RDONLY)
    if fd < 0:
        return False
    _ = sys_close(fd)
    return True


def _path_size(path: String) -> Int:
    var fd = sys_open(path, O_RDONLY)
    if fd < 0:
        return 0
    var n = file_size(fd)
    _ = sys_close(fd)
    return n


def _unlink_file(path: String) -> Bool:
    var n = path.byte_length()
    var buf = alloc[UInt8](n + 1)
    var src = path.as_bytes()
    for i in range(n):
        buf[i] = src[i]
    buf[n] = 0
    var rc = Int(external_call["unlink", Int32](BytePtr(unsafe_from_address=Int(buf))))
    buf.free()
    return rc == 0


def _gallery_state_default() raises -> JSONValue:
    var o = JSONValue.new_object()
    o.set("schema", JSONValue.from_string(String("serenity.gallery_state.v1")))
    o.set("favorites", JSONValue.new_array())
    o.set("names", JSONValue.new_array())
    o.set("order", JSONValue.new_array())
    o.set("imports", JSONValue.new_array())
    return o^


def _load_gallery_state() raises -> JSONValue:
    _ensure_gallery_dirs()
    try:
        var text = _read_text_file(String(GALLERY_STATE_PATH))
        var doc = loads(text)
        if doc.is_object() and doc.contains("favorites") and doc["favorites"].is_array():
            return doc^
    except:
        pass
    return _gallery_state_default()


def _gallery_favorite(doc: JSONValue, id: String) raises -> Bool:
    if not doc.contains("favorites") or not doc["favorites"].is_array():
        return False
    var arr = doc["favorites"]
    for i in range(arr.length()):
        if arr[i].is_string() and arr[i].as_string() == id:
            return True
    return False


def _set_gallery_favorite_doc(doc: JSONValue, id: String, favorite: Bool) raises -> JSONValue:
    var out_arr = JSONValue.new_array()
    var found = False
    if doc.contains("favorites") and doc["favorites"].is_array():
        var arr = doc["favorites"]
        for i in range(arr.length()):
            if not arr[i].is_string():
                continue
            var item = arr[i].as_string()
            if item == id:
                found = True
                if favorite:
                    out_arr.append(JSONValue.from_string(id))
            else:
                out_arr.append(JSONValue.from_string(item))
    if favorite and not found:
        out_arr.append(JSONValue.from_string(id))
    var out = JSONValue.new_object()
    out.set("schema", JSONValue.from_string(String("serenity.gallery_state.v1")))
    out.set("favorites", out_arr^)
    if doc.contains("names") and doc["names"].is_array():
        out.set("names", doc["names"])
    else:
        out.set("names", JSONValue.new_array())
    if doc.contains("order") and doc["order"].is_array():
        out.set("order", doc["order"])
    else:
        out.set("order", JSONValue.new_array())
    if doc.contains("imports") and doc["imports"].is_array():
        out.set("imports", doc["imports"])
    else:
        out.set("imports", JSONValue.new_array())
    return out^


def _copy_gallery_state_arrays(mut out: JSONValue, doc: JSONValue) raises:
    if doc.contains("favorites") and doc["favorites"].is_array():
        out.set("favorites", doc["favorites"])
    else:
        out.set("favorites", JSONValue.new_array())
    if doc.contains("names") and doc["names"].is_array():
        out.set("names", doc["names"])
    else:
        out.set("names", JSONValue.new_array())
    if doc.contains("order") and doc["order"].is_array():
        out.set("order", doc["order"])
    else:
        out.set("order", JSONValue.new_array())
    if doc.contains("imports") and doc["imports"].is_array():
        out.set("imports", doc["imports"])
    else:
        out.set("imports", JSONValue.new_array())


def _gallery_name(doc: JSONValue, id: String) raises -> String:
    if doc.contains("names") and doc["names"].is_array():
        var arr = doc["names"]
        for i in range(arr.length()):
            var ent = arr[i]
            if (
                ent.is_object()
                and ent.contains("id")
                and ent["id"].is_string()
                and ent["id"].as_string() == id
                and ent.contains("name")
                and ent["name"].is_string()
            ):
                return ent["name"].as_string()
    return id.copy()


def _gallery_import_source(doc: JSONValue, id: String) raises -> String:
    if doc.contains("imports") and doc["imports"].is_array():
        var arr = doc["imports"]
        for i in range(arr.length()):
            var ent = arr[i]
            if (
                ent.is_object()
                and ent.contains("id")
                and ent["id"].is_string()
                and ent["id"].as_string() == id
                and ent.contains("source_path")
                and ent["source_path"].is_string()
            ):
                return ent["source_path"].as_string()
    return String("")


def _gallery_order_index(doc: JSONValue, id: String) raises -> Int:
    if doc.contains("order") and doc["order"].is_array():
        var arr = doc["order"]
        for i in range(arr.length()):
            if arr[i].is_string() and arr[i].as_string() == id:
                return i
    return -1


def _set_gallery_name_doc(doc: JSONValue, id: String, name: String) raises -> JSONValue:
    var out_arr = JSONValue.new_array()
    var replaced = False
    if doc.contains("names") and doc["names"].is_array():
        var arr = doc["names"]
        for i in range(arr.length()):
            var ent = arr[i]
            if ent.is_object() and ent.contains("id") and ent["id"].is_string() and ent["id"].as_string() == id:
                var next = JSONValue.new_object()
                next.set("id", JSONValue.from_string(id))
                next.set("name", JSONValue.from_string(name))
                out_arr.append(next^)
                replaced = True
            else:
                out_arr.append(ent^)
    if not replaced:
        var next = JSONValue.new_object()
        next.set("id", JSONValue.from_string(id))
        next.set("name", JSONValue.from_string(name))
        out_arr.append(next^)
    var out = JSONValue.new_object()
    out.set("schema", JSONValue.from_string(String("serenity.gallery_state.v1")))
    _copy_gallery_state_arrays(out, doc)
    out.set("names", out_arr^)
    return out^


def _set_gallery_order_doc(doc: JSONValue, ids: JSONValue) raises -> JSONValue:
    if not ids.is_array():
        raise Error("'ids' must be an array of gallery ids")
    var out_arr = JSONValue.new_array()
    for i in range(ids.length()):
        if not ids[i].is_string():
            raise Error("'ids[" + String(i) + "]' must be a string")
        var id = ids[i].as_string()
        if not _safe_gallery_id(id):
            raise Error("invalid gallery id: " + id)
        var png_path = String(OUT_DIR) + "/" + id + ".png"
        if not _path_exists_file(png_path):
            raise Error("gallery item not found: " + id)
        out_arr.append(JSONValue.from_string(id))
    var out = JSONValue.new_object()
    out.set("schema", JSONValue.from_string(String("serenity.gallery_state.v1")))
    _copy_gallery_state_arrays(out, doc)
    out.set("order", out_arr^)
    return out^


def _set_gallery_import_doc(doc: JSONValue, id: String, source_path: String) raises -> JSONValue:
    var out_arr = JSONValue.new_array()
    var replaced = False
    if doc.contains("imports") and doc["imports"].is_array():
        var arr = doc["imports"]
        for i in range(arr.length()):
            var ent = arr[i]
            if ent.is_object() and ent.contains("id") and ent["id"].is_string() and ent["id"].as_string() == id:
                var next = JSONValue.new_object()
                next.set("id", JSONValue.from_string(id))
                next.set("source_path", JSONValue.from_string(source_path))
                out_arr.append(next^)
                replaced = True
            else:
                out_arr.append(ent^)
    if not replaced:
        var next = JSONValue.new_object()
        next.set("id", JSONValue.from_string(id))
        next.set("source_path", JSONValue.from_string(source_path))
        out_arr.append(next^)
    var out = JSONValue.new_object()
    out.set("schema", JSONValue.from_string(String("serenity.gallery_state.v1")))
    _copy_gallery_state_arrays(out, doc)
    out.set("imports", out_arr^)
    return out^


def _remove_gallery_item_doc(doc: JSONValue, id: String) raises -> JSONValue:
    var fav_arr = JSONValue.new_array()
    if doc.contains("favorites") and doc["favorites"].is_array():
        var arr = doc["favorites"]
        for i in range(arr.length()):
            if arr[i].is_string() and arr[i].as_string() != id:
                fav_arr.append(JSONValue.from_string(arr[i].as_string()))
    var names_arr = JSONValue.new_array()
    if doc.contains("names") and doc["names"].is_array():
        var arr = doc["names"]
        for i in range(arr.length()):
            var ent = arr[i]
            if ent.is_object() and ent.contains("id") and ent["id"].is_string() and ent["id"].as_string() == id:
                continue
            names_arr.append(ent^)
    var order_arr = JSONValue.new_array()
    if doc.contains("order") and doc["order"].is_array():
        var arr = doc["order"]
        for i in range(arr.length()):
            if arr[i].is_string() and arr[i].as_string() != id:
                order_arr.append(JSONValue.from_string(arr[i].as_string()))
    var imports_arr = JSONValue.new_array()
    if doc.contains("imports") and doc["imports"].is_array():
        var arr = doc["imports"]
        for i in range(arr.length()):
            var ent = arr[i]
            if ent.is_object() and ent.contains("id") and ent["id"].is_string() and ent["id"].as_string() == id:
                continue
            imports_arr.append(ent^)
    var out = JSONValue.new_object()
    out.set("schema", JSONValue.from_string(String("serenity.gallery_state.v1")))
    out.set("favorites", fav_arr^)
    out.set("names", names_arr^)
    out.set("order", order_arr^)
    out.set("imports", imports_arr^)
    return out^


def _save_gallery_state(doc: JSONValue) raises:
    _ensure_gallery_dirs()
    _write_text_file(String(GALLERY_STATE_PATH), dumps(doc))


def _safe_gallery_id(id: String) -> Bool:
    return id != "" and id.startswith("job-") and id.find("/") < 0


def _gallery_thumb_path(id: String) -> String:
    return String(GALLERY_THUMB_DIR) + "/" + id + ".png"


def _ensure_gallery_thumbnail(id: String, png_path: String) raises -> String:
    _ensure_gallery_dirs()
    var thumb_path = _gallery_thumb_path(id)
    if _path_exists_file(thumb_path):
        return thumb_path^
    var img = decode_png(png_path)
    var tw = 256
    var th = 256
    if img.width >= img.height:
        th = (img.height * 256) // img.width
        if th < 1:
            th = 1
    else:
        tw = (img.width * 256) // img.height
        if tw < 1:
            tw = 1
    var thumb = resize_lanczos(img, tw, th, 3)
    encode_png(thumb, thumb_path)
    return thumb_path^


def _copy_json_field_if_present(mut out: JSONValue, src: JSONValue, key: String) raises:
    if src.is_object() and src.contains(key):
        out.set(key, src[key])


def _copy_optional_string_json_field(
    mut out: JSONValue, src: JSONValue, key: String,
) raises:
    if not src.is_object() or not src.contains(key) or src[key].is_null():
        return
    if not src[key].is_string():
        raise Error("'" + key + "' must be a string")
    out.set(key, JSONValue.from_string(src[key].as_string()))


def _gallery_item_from_png_state(
    id: String, path: String, favorite: Bool, ensure_thumb: Bool,
) raises -> JSONValue:
    var params_json = _png_genparams(path)
    var gallery_state = _load_gallery_state()
    var thumb_path = String("")
    var thumb_state = String("not_requested")
    if ensure_thumb and id != "":
        try:
            thumb_path = _ensure_gallery_thumbnail(id, path)
            thumb_state = String("cached")
        except e:
            thumb_state = String("error: ") + String(e)
    var o = JSONValue.new_object()
    o.set("id", JSONValue.from_string(id))
    o.set("name", JSONValue.from_string(_gallery_name(gallery_state, id)))
    o.set("display_name", JSONValue.from_string(_gallery_name(gallery_state, id)))
    o.set("path", JSONValue.from_string(path))
    o.set("size", JSONValue.from_int(_path_size(path)))
    o.set("favorite", JSONValue.from_bool(favorite))
    o.set("manual_order_index", JSONValue.from_int(_gallery_order_index(gallery_state, id)))
    var imported_from = _gallery_import_source(gallery_state, id)
    o.set("imported", JSONValue.from_bool(imported_from != ""))
    o.set("imported_from", JSONValue.from_string(imported_from))
    o.set("thumbnail_path", JSONValue.from_string(thumb_path))
    o.set("thumb_path", JSONValue.from_string(thumb_path))
    o.set("thumbnail", JSONValue.from_string(thumb_path))
    o.set("thumbnail_state", JSONValue.from_string(thumb_state))
    o.set("metadata_key", JSONValue.from_string(String(GENPARAMS_TEXT_KEY)))
    o.set("has_params", JSONValue.from_bool(params_json != ""))
    o.set("params_json", JSONValue.from_string(params_json))
    if params_json != "":
        try:
            var params = loads(params_json)
            _copy_json_field_if_present(o, params, String("model"))
            _copy_json_field_if_present(o, params, String("prompt"))
            _copy_json_field_if_present(o, params, String("seed"))
            _copy_json_field_if_present(o, params, String("width"))
            _copy_json_field_if_present(o, params, String("height"))
            if id != "":
                if not params.contains("params_source"):
                    params.set("params_source", JSONValue.from_string(String("gallery")))
                if not params.contains("reused_from_gallery_id"):
                    params.set("reused_from_gallery_id", JSONValue.from_string(id))
                if not params.contains("reused_from_job_id"):
                    params.set("reused_from_job_id", JSONValue.from_string(id))
                if not params.contains("reused_from_path"):
                    params.set("reused_from_path", JSONValue.from_string(path))
            o.set("params", params^)
        except e:
            o.set("params_error", JSONValue.from_string(String(e)))
    return o^


def _gallery_item_from_png(id: String, path: String) raises -> JSONValue:
    return _gallery_item_from_png_state(id, path, False, True)


def _gallery_error_item(id: String, path: String, err: String) raises -> JSONValue:
    var o = JSONValue.new_object()
    o.set("id", JSONValue.from_string(id))
    o.set("name", JSONValue.from_string(id))
    o.set("display_name", JSONValue.from_string(id))
    o.set("path", JSONValue.from_string(path))
    o.set("size", JSONValue.from_int(_path_size(path)))
    o.set("favorite", JSONValue.from_bool(False))
    o.set("thumbnail_path", JSONValue.from_string(String("")))
    o.set("thumb_path", JSONValue.from_string(String("")))
    o.set("thumbnail", JSONValue.from_string(String("")))
    o.set("thumbnail_state", JSONValue.from_string(String("error")))
    o.set("metadata_key", JSONValue.from_string(String(GENPARAMS_TEXT_KEY)))
    o.set("has_params", JSONValue.from_bool(False))
    o.set("params_json", JSONValue.from_string(String("")))
    o.set("error", JSONValue.from_string(err))
    return o^


def _scan_gallery_ids() -> List[String]:
    """Return job ids with PNG artifacts in OUT_DIR. The endpoint reads each
    PNG's tEXt chunk on demand, so this scan is only the gallery file index."""
    _ensure_gallery_dirs()
    var cmd = (
        String("find '") + OUT_DIR + "' -maxdepth 1 -type f -name 'job-*.png'"
        + " -printf '%f\\n' 2>/dev/null | sort > " + SCAN_PNG_TMP
    )
    var ids = List[String]()
    if _system(cmd) != 0:
        return ids^
    try:
        var text = _read_text_file(String(SCAN_PNG_TMP))
        for line in text.split("\n"):
            var f = String(line)
            if not f.endswith(".png"):
                continue
            ids.append(byte_substr(f, 0, f.byte_length() - 4))
    except:
        pass
    return ids^


def _string_list_contains(values: List[String], value: String) -> Bool:
    for i in range(len(values)):
        if values[i] == value:
            return True
    return False


def _gallery_id_num_local(id: String) -> Int:
    if not id.startswith("job-"):
        return 0
    try:
        return Int(byte_substr(id, 4, id.byte_length()))
    except:
        return 0


def _gallery_search_matches(id: String, path: String, params_json: String, search: String) -> Bool:
    if search == "":
        return True
    return _contains_ci(id, search) or _contains_ci(path, search) or _contains_ci(params_json, search)


def _gallery_filter_matches(
    params_json: String, favorite: Bool, filter: String, favorite_query: String,
) -> Bool:
    var favq = String(favorite_query.lower())
    if favq == "1" or favq == "true" or favq == "yes":
        if not favorite:
            return False
    elif favq == "0" or favq == "false" or favq == "no":
        if favorite:
            return False
    var f = String(filter.lower())
    if f == "" or f == "all" or f == "any":
        return True
    if f == "favorite" or f == "favorites" or f == "star" or f == "starred":
        return favorite
    if f == "has_params":
        return params_json != ""
    if f == "missing_params":
        return params_json == ""
    return _contains_ci(params_json, filter)


def _gallery_id_before(a: String, b: String, sort: String, state: JSONValue) raises -> Bool:
    var s = String(sort.lower())
    var ap = String(OUT_DIR) + "/" + a + ".png"
    var bp = String(OUT_DIR) + "/" + b + ".png"
    if s == "manual" or s == "manual_asc" or s == "order":
        var ai = _gallery_order_index(state, a)
        var bi = _gallery_order_index(state, b)
        if ai >= 0 and bi >= 0 and ai != bi:
            return ai < bi
        if ai >= 0 and bi < 0:
            return True
        if ai < 0 and bi >= 0:
            return False
    if s == "name_asc":
        return a < b
    if s == "name_desc":
        return a > b
    if s == "created_asc":
        return _gallery_id_num_local(a) < _gallery_id_num_local(b)
    if s == "size_asc":
        var az = _path_size(ap)
        var bz = _path_size(bp)
        if az != bz:
            return az < bz
    elif s == "size_desc":
        var az = _path_size(ap)
        var bz = _path_size(bp)
        if az != bz:
            return az > bz
    elif s == "favorite_desc":
        var af = _gallery_favorite(state, a)
        var bf = _gallery_favorite(state, b)
        if af != bf:
            return af
    return _gallery_id_num_local(a) > _gallery_id_num_local(b)


# ── jobs.db (the gallery-index seam) ─────────────────────────────────────────
def _db_safe_params(params_json: String) -> String:
    """Cap params_json for the db row (F1): the DbWriter cannot overflow a
    4096-byte page, and a >=4 KB prompt used to raise out of record_finished.
    Truncation backs up to a UTF-8 boundary; the FULL params stay in the PNG
    tEXt chunk (the db row is only the gallery index)."""
    if params_json.byte_length() <= DB_PARAMS_MAX:
        return params_json.copy()
    var cut = DB_PARAMS_MAX
    var b = params_json.as_bytes()
    while cut > 0 and (b[cut] & 0xC0) == 0x80:  # don't split a codepoint
        cut -= 1
    return byte_substr(params_json, 0, cut) + "...truncated"


def _terminal_state(s: String) -> Bool:
    return (
        s == "done" or s == "failed" or s == "cancelled" or s == "interrupted"
    )


def save_jobs_db(prior: List[List[String]], jobs: List[JobRecord]) raises:
    """Rewrite jobs.db = prior-session rows (F7) + every session job that has
    STARTED (state != queued). Writing the row at job START (state=running)
    makes the db crash-safe: if the daemon dies hard mid-job (the SIGTERM can
    race to an unmaskable runtime thread — see run_daemon — or SIGKILL), the
    row survives as "running" and the NEXT startup repairs it to "interrupted"
    (F10). The pure-Mojo DbWriter builds the whole file per save — fine at
    gallery scale."""
    var writer = DbWriter.create()
    var cols: List[String] = ["id", "created", "model", "params_json", "state", "output_path"]
    writer.create_table(
        "jobs",
        "CREATE TABLE jobs (id TEXT, created TEXT, model TEXT, params_json TEXT, state TEXT, output_path TEXT)",
        cols^,
    )
    for i in range(len(prior)):
        var vals = List[Value]()
        vals.append(Value.text(prior[i][0]))
        vals.append(Value.text(prior[i][1]))
        vals.append(Value.text(prior[i][2]))
        vals.append(Value.text(_db_safe_params(prior[i][3])))
        vals.append(Value.text(prior[i][4]))
        vals.append(Value.text(prior[i][5]))
        writer.insert("jobs", vals)
    for i in range(len(jobs)):
        if jobs[i].state == "queued":
            continue  # never started; nothing truthful to index yet
        var vals = List[Value]()
        vals.append(Value.text(jobs[i].params.job_id))
        vals.append(Value.text(jobs[i].created))
        vals.append(Value.text(jobs[i].params.model))
        vals.append(Value.text(_db_safe_params(jobs[i].params.params_json)))
        vals.append(Value.text(jobs[i].state))
        vals.append(Value.text(jobs[i].output_path))
        writer.insert("jobs", vals)
    writer.save(DB_PATH)


def save_jobs_db_safe(prior: List[List[String]], jobs: List[JobRecord]):
    """F1: a jobs.db failure must NEVER kill the daemon — the db is an index,
    the job's image + tEXt params are already on disk. Log and continue."""
    try:
        save_jobs_db(prior, jobs)
    except e:
        print("WARNING: jobs.db save failed (daemon continues):", e)


def _job_id_num(id: String) -> Int:
    """job-NNNN -> NNNN (0 when unparseable)."""
    if not id.startswith("job-"):
        return 0
    try:
        return Int(byte_substr(id, 4, id.byte_length()))
    except:
        return 0


def load_prior_rows() -> List[List[String]]:
    """F7: read a prior jobs.db (pure-Mojo sqlite reader) so this session's
    full-file rewrites PRESERVE history. Non-terminal states (a hard-killed
    session's "running" row) are repaired to "interrupted" here (F10).
    Missing/unreadable db -> fresh start."""
    var out = List[List[String]]()
    try:
        var db = Database.open(String(DB_PATH))
        var rows = db.read_table("jobs")
        var repaired = 0
        for i in range(len(rows)):
            var v = rows[i].values.copy()
            if len(v) < 6:
                continue
            var cols = List[String]()
            for k in range(6):
                cols.append(v[k].as_text())
            if not _terminal_state(cols[4]):
                cols[4] = String("interrupted")
                repaired += 1
            out.append(cols^)
        if len(out) > 0:
            print("jobs.db: preloaded", len(out), "prior rows (",
                  repaired, "repaired to interrupted )")
    except:
        pass  # no prior db (first run) or unreadable -> start fresh
    return out^


def max_prior_id(prior: List[List[String]]) -> Int:
    var max_id = 0
    for i in range(len(prior)):
        var n = _job_id_num(prior[i][0])
        if n > max_id:
            max_id = n
    return max_id


def max_output_png_id() -> Int:
    """F7 belt-and-braces: scan the output dir for job-*.png and return the
    highest id, so output files are never overwritten even if jobs.db was
    deleted while outputs were kept."""
    var cmd = (
        String("find '") + OUT_DIR + "' -maxdepth 1 -type f -name 'job-*.png'"
        + " -printf '%f\\n' 2>/dev/null > " + SCAN_PNG_TMP
    )
    if _system(cmd) != 0:
        return 0
    var max_id = 0
    try:
        var text = _read_text_file(String(SCAN_PNG_TMP))
        for line in text.split("\n"):
            var l = String(line)
            if not l.endswith(".png"):
                continue
            var n = _job_id_num(byte_substr(l, 0, l.byte_length() - 4))
            if n > max_id:
                max_id = n
    except:
        return 0
    return max_id


# ── the worker tick (runs INSIDE the event loop) ─────────────────────────────
def tick_worker[B: GenBackend](
    mut backend: B, mut jobs: List[JobRecord], mut running: Int,
    prior: List[List[String]], mut ws: Dict[Int, Bool],
) raises:
    """One unit of serial-worker progress per event-loop tick."""
    if running >= 0:
        if jobs[running].cancel_requested:
            backend.cancel()
        var r = backend.step()
        jobs[running].step = r.step
        if r.total > 0:
            jobs[running].total = r.total
            jobs[running].progress = (r.step * 100) // r.total
        if r.done:
            jobs[running].state = String("done")
            jobs[running].progress = 100
            jobs[running].output_path = r.output_path
        elif r.failed:
            jobs[running].state = String("failed")
            jobs[running].error = r.error
        elif r.cancelled:
            jobs[running].state = String("cancelled")
        broadcast(ws, event_json(jobs[running], r.preview, r.phase))
        if r.done or r.failed or r.cancelled:
            print("job", jobs[running].params.job_id, "->", jobs[running].state)
            save_jobs_db_safe(prior, jobs)
            # F3: reclaim this job's transient device-memory peak (per-job text
            # encoder, decode activations) back to the OS at the job boundary,
            # so idle VRAM tracks the resident footprint, not the high-water
            # mark. No-op for backends that hold no reclaimable pool memory.
            try:
                backend.between_jobs_trim()
            except e:
                print("WARNING: between_jobs_trim failed (continuing):", e)
            running = -1
        return
    # idle: promote the next queued job (skipping pre-cancelled ones)
    for i in range(len(jobs)):
        if jobs[i].state != "queued":
            continue
        if jobs[i].cancel_requested:
            jobs[i].state = String("cancelled")
            broadcast(ws, event_json(jobs[i], String(""), String("")))
            save_jobs_db_safe(prior, jobs)
            continue
        try:
            backend.start(jobs[i].params)
        except e:
            jobs[i].state = String("failed")
            jobs[i].error = String(e)
            # Symmetric with the step()-failure path (which prints "-> done/
            # failed/cancelled"): a start() failure (e.g. a model-switch load
            # that OOMs against a pinned pool — the F3 scenario) must also hit
            # stdout, not just the job-state record. "Never suppress silently."
            print("job", jobs[i].params.job_id, "-> failed (start):", jobs[i].error)
            broadcast(ws, event_json(jobs[i], String(""), String("")))
            save_jobs_db_safe(prior, jobs)
            continue
        jobs[i].state = String("running")
        running = i
        print("job", jobs[i].params.job_id, "-> running (", jobs[i].total, "steps )")
        # F10 crash-safety: persist the running row NOW, so a hard kill
        # (SIGKILL / a TERM the runtime threads dequeue) still leaves a row
        # the next startup repairs to "interrupted".
        save_jobs_db_safe(prior, jobs)
        broadcast(ws, event_json(jobs[i], String(""), String("")))
        return


# ── HTTP layer ───────────────────────────────────────────────────────────────
def json_response(status: Int, body: String) -> Response:
    var r = Response(status)
    r.set_header("Content-Type", "application/json")
    r.set_body(body)
    return r^


def error_response(status: Int, detail: String) raises -> Response:
    var o = JSONValue.new_object()
    o.set("detail", JSONValue.from_string(detail))
    return json_response(status, dumps(o))


def is_ws_upgrade(req: Request) raises -> Bool:
    var up = String(req.header("upgrade").lower())
    var conn = String(req.header("connection").lower())
    return up == "websocket" and conn.find("upgrade") >= 0


def keep_alive_wanted(req: Request) raises -> Bool:
    var connl = String(req.header("connection").lower())
    if req.version == "HTTP/1.1":
        return connl != "close"
    return connl == "keep-alive"


def _find_job(jobs: List[JobRecord], id: String) -> Int:
    for i in range(len(jobs)):
        if jobs[i].params.job_id == id:
            return i
    return -1


def _body_object_or_empty(body: String) raises -> JSONValue:
    if body == "":
        return JSONValue.new_object()
    var obj = loads(body)
    if not obj.is_object():
        raise Error("body must be a JSON object")
    return obj^


def _required_string_field(obj: JSONValue, key: String) raises -> String:
    if not obj.contains(key) or not obj[key].is_string():
        raise Error("'" + key + "' (string) is required")
    var value = obj[key].as_string()
    if value == "":
        raise Error("'" + key + "' must be non-empty")
    return value


def _route_or_body_job_id(
    path: String, exact_path: String, prefix_len: Int, body: JSONValue,
) raises -> String:
    if path == exact_path:
        return _required_string_field(body, String("id"))
    var id = byte_substr(path, prefix_len, path.byte_length())
    if id == "" or id.find("/") >= 0:
        raise Error("invalid job id in path")
    return id


def _is_active_queued(j: JobRecord) -> Bool:
    return j.state == "queued" and not j.cancel_requested


def _active_queued_count(jobs: List[JobRecord]) -> Int:
    var count = 0
    for i in range(len(jobs)):
        if _is_active_queued(jobs[i]):
            count += 1
    return count


def _queued_position_of_index(jobs: List[JobRecord], idx: Int) -> Int:
    var pos = 0
    for i in range(len(jobs)):
        if not _is_active_queued(jobs[i]):
            continue
        if i == idx:
            return pos
        pos += 1
    return -1


def _queued_index_at_position(jobs: List[JobRecord], target: Int) -> Int:
    var pos = 0
    for i in range(len(jobs)):
        if not _is_active_queued(jobs[i]):
            continue
        if pos == target:
            return i
        pos += 1
    return -1


def _queued_jobs_json(jobs: List[JobRecord]) raises -> JSONValue:
    var arr = JSONValue.new_array()
    var pos = 0
    for i in range(len(jobs)):
        if not _is_active_queued(jobs[i]):
            continue
        var o = JSONValue.new_object()
        o.set("id", JSONValue.from_string(jobs[i].params.job_id))
        o.set("position", JSONValue.from_int(pos))
        arr.append(o^)
        pos += 1
    return arr^


def _swap_jobs(mut jobs: List[JobRecord], a: Int, b: Int):
    var tmp = jobs[a].copy()
    jobs[a] = jobs[b].copy()
    jobs[b] = tmp^


def _move_queued_job_to_position(
    mut jobs: List[JobRecord], src_idx: Int, target_pos: Int,
) -> Int:
    var pos = _queued_position_of_index(jobs, src_idx)
    while pos > target_pos:
        var cur = _queued_index_at_position(jobs, pos)
        var prev = _queued_index_at_position(jobs, pos - 1)
        if cur < 0 or prev < 0:
            return pos
        _swap_jobs(jobs, cur, prev)
        pos -= 1
    while pos < target_pos:
        var cur = _queued_index_at_position(jobs, pos)
        var nxt = _queued_index_at_position(jobs, pos + 1)
        if cur < 0 or nxt < 0:
            return pos
        _swap_jobs(jobs, cur, nxt)
        pos += 1
    return pos


def _remove_job_at(mut jobs: List[JobRecord], idx: Int):
    for i in range(idx, len(jobs) - 1):
        jobs[i] = jobs[i + 1].copy()
    _ = jobs.pop()


def _reorder_target_position(
    jobs: List[JobRecord], src_idx: Int, body: JSONValue,
) raises -> Int:
    var count = _active_queued_count(jobs)
    var src_pos = _queued_position_of_index(jobs, src_idx)
    if count <= 0 or src_pos < 0:
        raise Error("job is not in the active queue")
    if body.contains("before_id") and not body["before_id"].is_null():
        if not body["before_id"].is_string():
            raise Error("'before_id' must be a string")
        var before_id = body["before_id"].as_string()
        if before_id == jobs[src_idx].params.job_id:
            return src_pos
        var before_idx = _find_job(jobs, before_id)
        if before_idx < 0:
            raise Error("no such before_id: " + before_id)
        if not _is_active_queued(jobs[before_idx]):
            raise Error("'before_id' is not an active queued job: " + before_id)
        var before_pos = _queued_position_of_index(jobs, before_idx)
        if before_pos > src_pos:
            before_pos -= 1
        return before_pos
    if not body.contains("position") or not body["position"].is_int():
        raise Error("'position' (integer) or 'before_id' (string) is required")
    var target = body["position"].as_int()
    if target < 0 or target >= count:
        raise Error("'position' out of range [0.." + String(count - 1) + "]")
    return target


# ── presets / last UI state ─────────────────────────────────────────────────
def _ensure_state_dir():
    _ = _system(String("mkdir -p '") + STATE_DIR + "'")


def _write_text_file(path: String, text: String) raises:
    _ensure_state_dir()
    with open(path, "w") as f:
        f.write(text)


# ── shell quoting ───────────────────────────────────────────────────────────
def _shell_quote(s: String) -> String:
    """Single-quote a shell argument. Paths in this project are ASCII."""
    var q = String("'")
    var bytes = s.as_bytes()
    for i in range(s.byte_length()):
        if Int(bytes[i]) == 39:
            q += String("'\\''")
        else:
            q += chr(Int(bytes[i]))
    q += String("'")
    return q^


def _default_state_doc() -> JSONValue:
    var state = JSONValue.new_object()
    var o = JSONValue.new_object()
    o.set("schema", JSONValue.from_string(String("serenity.ui_state.v1")))
    o.set("state", state^)
    return o^


def _default_presets_doc() -> JSONValue:
    var arr = JSONValue.new_array()
    var o = JSONValue.new_object()
    o.set("schema", JSONValue.from_string(String("serenity.presets.v1")))
    o.set("presets", arr^)
    return o^


def _load_state_doc() -> JSONValue:
    try:
        var text = _read_text_file(String(STATE_PATH))
        var doc = loads(text)
        if doc.is_object() and doc.contains("state") and doc["state"].is_object():
            return doc^
    except:
        pass
    return _default_state_doc()


def _load_presets_doc() -> JSONValue:
    try:
        var text = _read_text_file(String(PRESETS_PATH))
        var doc = loads(text)
        if doc.is_object() and doc.contains("presets") and doc["presets"].is_array():
            return doc^
    except:
        pass
    return _default_presets_doc()


def _state_doc_from_body(body: String) raises -> JSONValue:
    var obj = loads(body)
    if not obj.is_object():
        raise Error("state body must be a JSON object")
    var state: JSONValue
    if obj.contains("state"):
        if not obj["state"].is_object():
            raise Error("'state' must be an object")
        state = obj["state"]
    else:
        state = obj.copy()
    var out = JSONValue.new_object()
    out.set("schema", JSONValue.from_string(String("serenity.ui_state.v1")))
    out.set("state", state^)
    return out^


def _preset_name_from_path(path: String, prefix_len: Int) raises -> String:
    var name = byte_substr(path, prefix_len, path.byte_length())
    if name == "" or name.find("/") >= 0:
        raise Error("invalid preset name in path")
    return name


def _preset_params_from_body(body: JSONValue) raises -> JSONValue:
    if body.contains("params"):
        if not body["params"].is_object():
            raise Error("'params' must be an object")
        return body["params"]
    return body.copy()


def _preset_entry(name: String, params: JSONValue) -> JSONValue:
    var o = JSONValue.new_object()
    o.set("name", JSONValue.from_string(name))
    o.set("params", params.copy())
    return o^


def _preset_exists(doc: JSONValue, name: String) raises -> Bool:
    if not doc.contains("presets") or not doc["presets"].is_array():
        return False
    var arr = doc["presets"]
    for i in range(arr.length()):
        var ent = arr[i]
        if ent.is_object() and ent.contains("name") and ent["name"].as_string() == name:
            return True
    return False


def _preset_by_name(doc: JSONValue, name: String) raises -> JSONValue:
    var arr = doc["presets"]
    for i in range(arr.length()):
        var ent = arr[i]
        if ent.is_object() and ent.contains("name") and ent["name"].as_string() == name:
            return ent^
    raise Error("no such preset: " + name)


def _upsert_preset_doc(
    doc: JSONValue, name: String, params: JSONValue,
) raises -> JSONValue:
    var arr = doc["presets"]
    var out_arr = JSONValue.new_array()
    var replaced = False
    for i in range(arr.length()):
        var ent = arr[i]
        if ent.is_object() and ent.contains("name") and ent["name"].as_string() == name:
            out_arr.append(_preset_entry(name, params))
            replaced = True
        else:
            out_arr.append(ent^)
    if not replaced:
        out_arr.append(_preset_entry(name, params))
    var out = JSONValue.new_object()
    out.set("schema", JSONValue.from_string(String("serenity.presets.v1")))
    out.set("presets", out_arr^)
    return out^


def _delete_preset_doc(doc: JSONValue, name: String) raises -> JSONValue:
    var arr = doc["presets"]
    var out_arr = JSONValue.new_array()
    for i in range(arr.length()):
        var ent = arr[i]
        if ent.is_object() and ent.contains("name") and ent["name"].as_string() == name:
            continue
        out_arr.append(ent^)
    var out = JSONValue.new_object()
    out.set("schema", JSONValue.from_string(String("serenity.presets.v1")))
    out.set("presets", out_arr^)
    return out^


# ── model/LoRA browser cards ────────────────────────────────────────────────
def _contains_ci(text: String, query: String) -> Bool:
    if query == "":
        return True
    return String(text.lower()).find(String(query.lower())) >= 0


def _entry_arch(entry: ScanEntry) -> String:
    if entry.arch == "":
        return String("unknown")
    return entry.arch.copy()


def _browser_filter_matches(value: String, filter: String) -> Bool:
    if filter == "":
        return True
    var f = String(filter.lower())
    if f == "all" or f == "any":
        return True
    return String(value.lower()).find(f) >= 0


def _model_matches_browser(entry: ScanEntry, search: String, filter: String) -> Bool:
    if search != "":
        if (
            not _contains_ci(entry.name, search)
            and not _contains_ci(entry.path, search)
            and not _contains_ci(_entry_arch(entry), search)
        ):
            return False
    return _browser_filter_matches(_entry_arch(entry), filter)


def _lora_matches_browser(entry: ScanEntry, search: String, filter: String) -> Bool:
    if search != "":
        if (
            not _contains_ci(entry.name, search)
            and not _contains_ci(entry.path, search)
            and not _contains_ci(_entry_arch(entry), search)
        ):
            return False
    return _browser_filter_matches(_entry_arch(entry), filter)


def _scan_entry_before(a: ScanEntry, b: ScanEntry, sort: String) -> Bool:
    var s = String(sort.lower())
    var an = String(a.name.lower())
    var bn = String(b.name.lower())
    if s == "size" or s == "size_desc":
        if a.size != b.size:
            return a.size > b.size
    elif s == "size_asc":
        if a.size != b.size:
            return a.size < b.size
    elif s == "name_desc":
        if an != bn:
            return an > bn
    elif s == "arch" or s == "family":
        var aa = String(_entry_arch(a).lower())
        var ba = String(_entry_arch(b).lower())
        if aa != ba:
            return aa < ba
    if an != bn:
        return an < bn
    return a.path < b.path


def _int_list_contains(values: List[Int], value: Int) -> Bool:
    for i in range(len(values)):
        if values[i] == value:
            return True
    return False


def _model_lora_compatible(model_arch: String, target_arch: String) -> Bool:
    var m = String(model_arch.lower())
    var t = String(target_arch.lower())
    if t == "" or t == "unknown" or m == "" or m == "unknown":
        return False
    return m == t


def _model_arch_for(models: List[ScanEntry], model: String) -> String:
    if model == "":
        return String("")
    var needle = String(model.lower())
    for i in range(len(models)):
        if String(models[i].name.lower()) == needle or String(models[i].path.lower()) == needle:
            return _entry_arch(models[i])
    return model.copy()


def _compatible_models_json(lora: ScanEntry, models: List[ScanEntry]) raises -> JSONValue:
    var arr = JSONValue.new_array()
    var target_arch = _entry_arch(lora)
    for i in range(len(models)):
        if _model_lora_compatible(_entry_arch(models[i]), target_arch):
            arr.append(JSONValue.from_string(models[i].name))
    return arr^


def _lora_incompatible_reason(
    selected_model: String, selected_arch: String, target_arch: String, compatible: Bool,
) -> String:
    if selected_model == "":
        return String("no model selected")
    if target_arch == "unknown":
        return String("unknown LoRA target_arch")
    if selected_arch == "" or selected_arch == "unknown":
        return String("unknown selected model arch")
    if not compatible:
        return String("target_arch ") + target_arch + String(" is not compatible with model arch ") + selected_arch
    return String("")


def _model_entry_json(entry: ScanEntry, resident: String) raises -> JSONValue:
    var arch = _entry_arch(entry)
    var loaded = resident != "" and entry.name == resident
    var metadata = JSONValue.new_object()
    metadata.set("schema", JSONValue.from_string(String("serenity.model.metadata.v1")))
    metadata.set("source", JSONValue.from_string(String("disk_scan")))
    metadata.set("family", JSONValue.from_string(arch))
    metadata.set("notes", JSONValue.from_string(String("")))

    var card = JSONValue.new_object()
    card.set("schema", JSONValue.from_string(String("serenity.model.card.v1")))
    card.set("title", JSONValue.from_string(entry.name))
    card.set("subtitle", JSONValue.from_string(arch))
    card.set("path", JSONValue.from_string(entry.path))
    card.set("size", JSONValue.from_int(entry.size))
    card.set("thumbnail", JSONValue.from_string(String("")))
    card.set("preview", JSONValue.from_string(String("")))
    card.set("favorite", JSONValue.from_bool(False))
    card.set("loaded", JSONValue.from_bool(loaded))
    card.set("metadata", metadata.copy())

    var mo = JSONValue.new_object()
    mo.set("name", JSONValue.from_string(entry.name))
    mo.set("path", JSONValue.from_string(entry.path))
    mo.set("arch", JSONValue.from_string(arch))
    mo.set("size", JSONValue.from_int(entry.size))
    mo.set("loaded", JSONValue.from_bool(loaded))
    mo.set("type", JSONValue.from_string(String("checkpoint")))
    mo.set("thumbnail", JSONValue.from_string(String("")))
    mo.set("preview", JSONValue.from_string(String("")))
    mo.set("favorite", JSONValue.from_bool(False))
    mo.set("metadata", metadata^)
    mo.set("card", card^)
    return mo^


def _lora_entry_json(
    entry: ScanEntry, models: List[ScanEntry], selected_model: String, selected_arch: String,
) raises -> JSONValue:
    var target_arch = _entry_arch(entry)
    var compatible = _model_lora_compatible(selected_arch, target_arch)
    var incompatible_reason = _lora_incompatible_reason(
        selected_model, selected_arch, target_arch, compatible,
    )
    var compatibility = JSONValue.new_object()
    compatibility.set("compatible", JSONValue.from_bool(compatible))
    compatibility.set("model", JSONValue.from_string(selected_model))
    compatibility.set("model_arch", JSONValue.from_string(selected_arch))
    compatibility.set("target_arch", JSONValue.from_string(target_arch))
    compatibility.set("incompatible_reason", JSONValue.from_string(incompatible_reason))

    var metadata = JSONValue.new_object()
    metadata.set("schema", JSONValue.from_string(String("serenity.lora.metadata.v1")))
    metadata.set("source", JSONValue.from_string(String("safetensors_header_probe")))
    metadata.set("target_arch", JSONValue.from_string(target_arch))
    metadata.set("trigger", JSONValue.from_string(String("")))

    var card = JSONValue.new_object()
    card.set("schema", JSONValue.from_string(String("serenity.lora.card.v1")))
    card.set("title", JSONValue.from_string(entry.name))
    card.set("subtitle", JSONValue.from_string(target_arch))
    card.set("path", JSONValue.from_string(entry.path))
    card.set("size", JSONValue.from_int(entry.size))
    card.set("thumbnail", JSONValue.from_string(String("")))
    card.set("preview", JSONValue.from_string(String("")))
    card.set("favorite", JSONValue.from_bool(False))
    card.set("metadata", metadata.copy())
    card.set("compatibility", compatibility.copy())

    var lo = JSONValue.new_object()
    lo.set("name", JSONValue.from_string(entry.name))
    lo.set("path", JSONValue.from_string(entry.path))
    lo.set("size", JSONValue.from_int(entry.size))
    lo.set("arch", JSONValue.from_string(target_arch))
    lo.set("target_arch", JSONValue.from_string(target_arch))
    lo.set("trigger", JSONValue.from_string(String("")))
    lo.set("thumbnail", JSONValue.from_string(String("")))
    lo.set("preview", JSONValue.from_string(String("")))
    lo.set("favorite", JSONValue.from_bool(False))
    lo.set("compatible_models", _compatible_models_json(entry, models))
    lo.set("compatible", JSONValue.from_bool(compatible))
    lo.set("compatibility", compatibility^)
    lo.set("incompatible_reason", JSONValue.from_string(incompatible_reason))
    lo.set("metadata", metadata^)
    lo.set("card", card^)
    return lo^


def _models_array_json(
    models: List[ScanEntry], search: String, filter: String, sort: String, resident: String,
) raises -> JSONValue:
    var arr = JSONValue.new_array()
    var emitted = List[Int]()
    while True:
        var best = -1
        for i in range(len(models)):
            if _int_list_contains(emitted, i):
                continue
            if not _model_matches_browser(models[i], search, filter):
                continue
            if best < 0 or _scan_entry_before(models[i], models[best], sort):
                best = i
        if best < 0:
            break
        emitted.append(best)
        arr.append(_model_entry_json(models[best], resident))
    return arr^


def _loras_array_json(
    loras: List[ScanEntry], models: List[ScanEntry], search: String, filter: String,
    sort: String, selected_model: String, selected_arch: String,
) raises -> JSONValue:
    var arr = JSONValue.new_array()
    var emitted = List[Int]()
    while True:
        var best = -1
        for i in range(len(loras)):
            if _int_list_contains(emitted, i):
                continue
            if not _lora_matches_browser(loras[i], search, filter):
                continue
            if best < 0 or _scan_entry_before(loras[i], loras[best], sort):
                best = i
        if best < 0:
            break
        emitted.append(best)
        arr.append(_lora_entry_json(loras[best], models, selected_model, selected_arch))
    return arr^


def _models_query_json(
    search: String, filter: String, sort: String, lora_search: String,
    lora_filter: String, lora_sort: String, selected_model: String,
) raises -> JSONValue:
    var o = JSONValue.new_object()
    o.set("search", JSONValue.from_string(search))
    o.set("filter", JSONValue.from_string(filter))
    o.set("sort", JSONValue.from_string(sort))
    o.set("q", JSONValue.from_string(search))
    o.set("lora_search", JSONValue.from_string(lora_search))
    o.set("lora_filter", JSONValue.from_string(lora_filter))
    o.set("lora_sort", JSONValue.from_string(lora_sort))
    o.set("model", JSONValue.from_string(selected_model))
    return o^


def handle_api(
    mut jobs: List[JobRecord], mut njobs: Int, running: Int,
    prior: List[List[String]],
    backend_name: String, model_name: String, resident: String, req: Request,
) raises -> Response:
    """Route + handle one (non-WebSocket) API request."""
    var path = req.path()

    if req.method == "GET" and path == "/v1/health":
        var o = JSONValue.new_object()
        o.set("status", JSONValue.from_string(String("ok")))
        o.set("backend", JSONValue.from_string(backend_name))
        o.set("model", JSONValue.from_string(model_name))
        o.set("resident", JSONValue.from_string(resident))
        return json_response(200, dumps(o))

    if req.method == "GET" and path == "/v1/models":
        # fresh disk scan per request (header reads only — cheap, and the
        # list stays correct when files are added/removed while running)
        var models = scan_checkpoints()
        var loras = scan_loras()
        var search = req.query("search")
        var q = req.query("q")
        if search == "" and q != "":
            search = q.copy()
        var filter = req.query("filter")
        var sort = req.query("sort")
        var lora_search = req.query("lora_search")
        if lora_search == "":
            lora_search = search.copy()
        var lora_filter = req.query("lora_filter")
        var lora_sort = req.query("lora_sort")
        if lora_sort == "":
            lora_sort = sort.copy()
        var selected_model = req.query("model")
        if selected_model == "":
            selected_model = resident.copy()
        var selected_arch = _model_arch_for(models, selected_model)
        # Browser card response fields emitted by helper JSON:
        # "card" "thumbnail" "metadata" "favorite" "preview"
        # LoRA metadata/compatibility fields:
        # "target_arch" "trigger" "compatible_models" "compatible"
        # "compatibility" "incompatible_reason"
        var ma = _models_array_json(models, search, filter, sort, resident)
        var la = _loras_array_json(
            loras, models, lora_search, lora_filter, lora_sort,
            selected_model, selected_arch,
        )
        var o = JSONValue.new_object()
        o.set("schema", JSONValue.from_string(String("serenity.models.v1")))
        o.set("query", _models_query_json(
            search, filter, sort, lora_search, lora_filter, lora_sort,
            selected_model,
        ))
        o.set("models_total", JSONValue.from_int(len(models)))
        o.set("loras_total", JSONValue.from_int(len(loras)))
        o.set("model_selected", JSONValue.from_string(selected_model))
        o.set("model_selected_arch", JSONValue.from_string(selected_arch))
        o.set("models", ma^)
        o.set("loras", la^)
        return json_response(200, dumps(o))

    if req.method == "GET" and path == "/v1/samplers":
        return json_response(200, swarmui_sampler_registry_json())

    if req.method == "GET" and path == "/v1/video":
        var doc = video_readiness_doc(backend_name, model_name, resident)
        return json_response(200, dumps(doc))

    if req.method == "POST" and path == "/v1/video":
        try:
            var body = _body_object_or_empty(req.body)
            njobs += 1
            var video_id = String("video-") + _pad4(njobs)
            var doc = ltx2_staged_smoke_video_result(
                body, video_id, backend_name, model_name, resident,
            )
            var status = 200
            if doc.contains("state") and doc["state"].is_string():
                var state = doc["state"].as_string()
                if state == "failed" or state == "failed_probe":
                    status = 500
            return json_response(status, dumps(doc))
        except e:
            return error_response(422, String(e))

    if req.method == "GET" and path == "/v1/video/probe":
        var mp4_path = req.query("path")
        try:
            var doc = probe_video_file(mp4_path)
            return json_response(200, dumps(doc))
        except e:
            return error_response(422, "cannot probe MP4: " + String(e))

    if req.method == "GET" and path == "/v1/gallery":
        var search = req.query("search")
        var q = req.query("q")
        if search == "" and q != "":
            search = q.copy()
        var filter = req.query("filter")
        var sort = req.query("sort")
        var favorite_query = req.query("favorite")
        var state = _load_gallery_state()
        var ids = _scan_gallery_ids()
        var arr = JSONValue.new_array()
        var emitted = List[String]()
        while True:
            var best = String("")
            for i in range(len(ids)):
                if _string_list_contains(emitted, ids[i]):
                    continue
                var png_path = String(OUT_DIR) + "/" + ids[i] + ".png"
                var params_json = _png_genparams_or_empty(png_path)
                var fav = _gallery_favorite(state, ids[i])
                if not _gallery_search_matches(ids[i], png_path, params_json, search):
                    continue
                if not _gallery_filter_matches(params_json, fav, filter, favorite_query):
                    continue
                if best == "" or _gallery_id_before(ids[i], best, sort, state):
                    best = ids[i].copy()
            if best == "":
                break
            emitted.append(best)
            var png_path = String(OUT_DIR) + "/" + best + ".png"
            try:
                arr.append(_gallery_item_from_png_state(
                    best, png_path, _gallery_favorite(state, best), True,
                ))
            except e:
                arr.append(_gallery_error_item(best, png_path, String(e)))
        var o = JSONValue.new_object()
        o.set("schema", JSONValue.from_string(String("serenity.gallery.v1")))
        o.set("search", JSONValue.from_string(search))
        o.set("filter", JSONValue.from_string(filter))
        o.set("sort", JSONValue.from_string(sort))
        o.set("favorite", JSONValue.from_string(favorite_query))
        o.set("thumbnail_path_root", JSONValue.from_string(String(GALLERY_THUMB_DIR)))
        o.set("count", JSONValue.from_int(len(emitted)))
        o.set("total", JSONValue.from_int(len(ids)))
        o.set("items", arr^)
        return json_response(200, dumps(o))

    if req.method == "GET" and path == "/v1/gallery/read":
        var png_path = req.query("path")
        if png_path == "":
            return error_response(422, "'path' query parameter is required")
        try:
            var item = _gallery_item_from_png(String(""), png_path)
            return json_response(200, dumps(item))
        except e:
            return error_response(422, "cannot read PNG genparams: " + String(e))

    if req.method == "POST" and path == "/v1/gallery/import":
        try:
            var body = _body_object_or_empty(req.body)
            var source_path = _required_string_field(body, String("path"))
            if source_path.find("\n") >= 0 or source_path.find("\r") >= 0:
                raise Error("invalid gallery import path")
            if not _path_exists_file(source_path):
                return error_response(404, "gallery import source not found: " + source_path)
            var params_json = _png_genparams(source_path)
            if params_json == "":
                raise Error("gallery import source has no " + String(GENPARAMS_TEXT_KEY))
            njobs += 1
            var id = String("job-") + _pad4(njobs)
            var dest = String(OUT_DIR) + "/" + id + ".png"
            var rc = _system(
                String("cp ") + _shell_quote(source_path) + String(" ") + _shell_quote(dest)
            )
            if rc != 0:
                njobs -= 1
                raise Error("gallery import copy failed")
            var state = _load_gallery_state()
            var updated = _set_gallery_import_doc(state, id, source_path)
            _save_gallery_state(updated)
            var item = _gallery_item_from_png_state(id, dest, _gallery_favorite(updated, id), True)
            return json_response(200, dumps(item))
        except e:
            return error_response(422, String(e))

    if req.method == "POST" and path == "/v1/gallery/order":
        try:
            var body = _body_object_or_empty(req.body)
            if not body.contains("ids"):
                raise Error("'ids' array is required")
            var state = _load_gallery_state()
            var updated = _set_gallery_order_doc(state, body["ids"])
            _save_gallery_state(updated)
            var o = JSONValue.new_object()
            o.set("schema", JSONValue.from_string(String("serenity.gallery_order.v1")))
            o.set("order", updated["order"])
            return json_response(200, dumps(o))
        except e:
            return error_response(422, String(e))

    if req.method == "POST" and path.startswith("/v1/gallery/") and path.endswith("/rename"):
        var id = byte_substr(path, 12, path.byte_length() - 7)
        if not _safe_gallery_id(id):
            return error_response(422, "invalid gallery id: " + id)
        var png_path = String(OUT_DIR) + "/" + id + ".png"
        if not _path_exists_file(png_path):
            return error_response(404, "gallery item not found: " + id)
        try:
            var body = _body_object_or_empty(req.body)
            var name = _required_string_field(body, String("name"))
            var state = _load_gallery_state()
            var updated = _set_gallery_name_doc(state, id, name)
            _save_gallery_state(updated)
            var item = _gallery_item_from_png_state(
                id, png_path, _gallery_favorite(updated, id), True,
            )
            return json_response(200, dumps(item))
        except e:
            return error_response(422, String(e))

    if req.method == "POST" and path.startswith("/v1/gallery/") and path.endswith("/favorite"):
        var id = byte_substr(path, 12, path.byte_length() - 9)
        if not _safe_gallery_id(id):
            return error_response(422, "invalid gallery id: " + id)
        var png_path = String(OUT_DIR) + "/" + id + ".png"
        if not _path_exists_file(png_path):
            return error_response(404, "gallery item not found: " + id)
        try:
            var body = _body_object_or_empty(req.body)
            var favorite = True
            if body.contains("favorite"):
                if not body["favorite"].is_bool():
                    raise Error("'favorite' must be a boolean")
                favorite = body["favorite"].as_bool()
            var state = _load_gallery_state()
            var next = _set_gallery_favorite_doc(state, id, favorite)
            _save_gallery_state(next)
            var item = _gallery_item_from_png_state(id, png_path, favorite, True)
            return json_response(200, dumps(item))
        except e:
            return error_response(422, String(e))

    if req.method == "DELETE" and path.startswith("/v1/gallery/"):
        var id = byte_substr(path, 12, path.byte_length())
        if not _safe_gallery_id(id):
            return error_response(422, "invalid gallery id: " + id)
        var png_path = String(OUT_DIR) + "/" + id + ".png"
        if not _path_exists_file(png_path):
            return error_response(404, "gallery item not found: " + id)
        var deleted = _unlink_file(png_path)
        var thumb_path = _gallery_thumb_path(id)
        var thumb_deleted = False
        if _path_exists_file(thumb_path):
            thumb_deleted = _unlink_file(thumb_path)
        try:
            var state = _load_gallery_state()
            var next = _remove_gallery_item_doc(state, id)
            _save_gallery_state(next)
        except:
            pass
        var o = JSONValue.new_object()
        o.set("id", JSONValue.from_string(id))
        o.set("deleted", JSONValue.from_bool(deleted))
        o.set("path", JSONValue.from_string(png_path))
        o.set("thumbnail_path", JSONValue.from_string(thumb_path))
        o.set("thumbnail_deleted", JSONValue.from_bool(thumb_deleted))
        return json_response(200, dumps(o))

    if req.method == "GET" and path.startswith("/v1/gallery/"):
        var id = byte_substr(path, 12, path.byte_length())
        if not _safe_gallery_id(id):
            return error_response(422, "invalid gallery id: " + id)
        var png_path = String(OUT_DIR) + "/" + id + ".png"
        try:
            var state = _load_gallery_state()
            var item = _gallery_item_from_png_state(
                id, png_path, _gallery_favorite(state, id), True,
            )
            return json_response(200, dumps(item))
        except e:
            return error_response(404, "cannot read gallery item: " + String(e))

    if req.method == "GET" and path == "/v1/state":
        var doc = _load_state_doc()
        return json_response(200, dumps(doc))

    if req.method == "POST" and path == "/v1/state":
        try:
            var doc = _state_doc_from_body(req.body)
            _write_text_file(String(STATE_PATH), dumps(doc))
            return json_response(200, dumps(doc))
        except e:
            return error_response(422, String(e))

    if req.method == "GET" and path == "/v1/presets":
        var doc = _load_presets_doc()
        return json_response(200, dumps(doc))

    if req.method == "GET" and path.startswith("/v1/presets/"):
        try:
            var name = _preset_name_from_path(path, 12)
            var doc = _load_presets_doc()
            var preset = _preset_by_name(doc, name)
            return json_response(200, dumps(preset))
        except e:
            return error_response(404, String(e))

    if req.method == "POST" and (
        path == "/v1/presets" or path.startswith("/v1/presets/")
    ):
        try:
            var body = _body_object_or_empty(req.body)
            var name: String
            if path == "/v1/presets":
                name = _required_string_field(body, String("name"))
                if not body.contains("params"):
                    raise Error("'params' (object) is required")
            else:
                name = _preset_name_from_path(path, 12)
            var params = _preset_params_from_body(body)
            var doc = _load_presets_doc()
            var updated = _upsert_preset_doc(doc, name, params)
            _write_text_file(String(PRESETS_PATH), dumps(updated))
            var preset = _preset_by_name(updated, name)
            return json_response(200, dumps(preset))
        except e:
            return error_response(422, String(e))

    if req.method == "DELETE" and path.startswith("/v1/presets/"):
        try:
            var name = _preset_name_from_path(path, 12)
            var doc = _load_presets_doc()
            if not _preset_exists(doc, name):
                return error_response(404, "no such preset: " + name)
            var updated = _delete_preset_doc(doc, name)
            _write_text_file(String(PRESETS_PATH), dumps(updated))
            var o = JSONValue.new_object()
            o.set("name", JSONValue.from_string(name))
            o.set("deleted", JSONValue.from_bool(True))
            o.set("presets", updated["presets"])
            return json_response(200, dumps(o))
        except e:
            return error_response(422, String(e))

    if req.method == "POST" and path == "/v1/generate":
        njobs += 1
        var job_id = String("job-") + _pad4(njobs)
        var p: JobParams
        try:
            p = parse_generate(req.body, job_id, String(OUT_DIR), backend_name)
        except e:
            njobs -= 1  # id not consumed
            var msg = String(e)
            if msg.startswith("[501] "):  # reserved-feature sentinel (plan H4)
                return error_response(501, byte_substr(msg, 6, msg.byte_length()))
            return error_response(422, msg)
        var requested_images = p.images
        var base_seed = p.seed
        var base_params_json = p.params_json.copy()
        # queue_position: jobs that will run before the first generated image
        var ahead = 0
        if running >= 0:
            ahead += 1
        for i in range(len(jobs)):
            if jobs[i].state == "queued" and not jobs[i].cancel_requested:
                ahead += 1
        var ids = JSONValue.new_array()
        var created = http_date()
        for image_i in range(requested_images):
            var child = p.copy()
            if image_i > 0:
                njobs += 1
                child.job_id = String("job-") + _pad4(njobs)
            child.seed = base_seed + image_i
            child.images = requested_images
            child.image_index = image_i
            child.image_count = requested_images
            child.params_json = params_json_for_image_job(
                base_params_json, child.job_id, child.seed, image_i, requested_images
            )
            ids.append(JSONValue.from_string(child.job_id))
            print(
                "job", child.job_id, "queued (position", ahead + image_i,
                " image", image_i + 1, "/", requested_images, ")"
            )
            jobs.append(JobRecord(child^, created.copy()))
        var o = JSONValue.new_object()
        o.set("job_id", JSONValue.from_string(job_id))
        o.set("job_ids", ids^)
        o.set("images", JSONValue.from_int(requested_images))
        o.set("queue_position", JSONValue.from_int(ahead))
        return json_response(200, dumps(o))

    if req.method == "GET" and path == "/v1/jobs":
        var arr = JSONValue.new_array()
        for i in range(len(prior)):
            if len(prior[i]) >= 6:
                arr.append(prior_job_json_value(prior[i]))
        for i in range(len(jobs)):
            arr.append(job_json_value(jobs[i]))
        return json_response(200, dumps(arr))

    if req.method == "GET" and path.startswith("/v1/job/"):
        var id = byte_substr(path, 8, path.byte_length())
        var i = _find_job(jobs, id)
        if i >= 0:
            return json_response(200, dumps(job_json_value(jobs[i])))
        var pi = _prior_row_index(prior, id)
        if pi < 0:
            return error_response(404, "no such job: " + id)
        return json_response(200, dumps(prior_job_json_value(prior[pi])))

    if req.method == "POST" and (
        path == "/v1/reorder" or path.startswith("/v1/reorder/")
    ):
        var body: JSONValue
        try:
            body = _body_object_or_empty(req.body)
        except e:
            return error_response(422, String(e))
        var id: String
        try:
            id = _route_or_body_job_id(path, String("/v1/reorder"), 12, body)
        except e:
            return error_response(422, String(e))
        var i = _find_job(jobs, id)
        if i < 0:
            return error_response(404, "no such job: " + id)
        if not _is_active_queued(jobs[i]):
            return error_response(
                409,
                "only active queued jobs can be reordered: " + id,
            )
        try:
            var target = _reorder_target_position(jobs, i, body)
            var new_pos = _move_queued_job_to_position(jobs, i, target)
            var o = JSONValue.new_object()
            o.set("job_id", JSONValue.from_string(id))
            o.set("position", JSONValue.from_int(new_pos))
            o.set("queue", _queued_jobs_json(jobs))
            return json_response(200, dumps(o))
        except e:
            return error_response(422, String(e))

    if req.method == "POST" and (
        path == "/v1/remove" or path.startswith("/v1/remove/")
    ):
        var body: JSONValue
        try:
            body = _body_object_or_empty(req.body)
        except e:
            return error_response(422, String(e))
        var id: String
        try:
            id = _route_or_body_job_id(path, String("/v1/remove"), 11, body)
        except e:
            return error_response(422, String(e))
        var i = _find_job(jobs, id)
        if i < 0:
            return error_response(404, "no such job: " + id)
        if not _is_active_queued(jobs[i]):
            return error_response(
                409,
                "only active queued jobs can be removed before execution; "
                "use /v1/cancel/<id> for running jobs: " + id,
            )
        if running >= 0 and i < running:
            return error_response(
                409,
                "cannot remove queued job before the running index: " + id,
            )
        _remove_job_at(jobs, i)
        var o = JSONValue.new_object()
        o.set("job_id", JSONValue.from_string(id))
        o.set("removed", JSONValue.from_bool(True))
        o.set("queue", _queued_jobs_json(jobs))
        return json_response(200, dumps(o))

    if req.method == "POST" and path.startswith("/v1/cancel/"):
        var id = byte_substr(path, 11, path.byte_length())
        var i = _find_job(jobs, id)
        if i < 0:
            return error_response(404, "no such job: " + id)
        if jobs[i].is_terminal():
            return error_response(409, "job already " + jobs[i].state)
        jobs[i].cancel_requested = True  # tick_worker finalizes (queued or running)
        var o = JSONValue.new_object()
        o.set("job_id", JSONValue.from_string(id))
        o.set("state", JSONValue.from_string(jobs[i].state))
        o.set("cancel_requested", JSONValue.from_bool(True))
        return json_response(200, dumps(o))

    return error_response(404, "not found: " + req.method + " " + path)


def serve_request(
    mut jobs: List[JobRecord], mut njobs: Int, running: Int,
    prior: List[List[String]],
    backend_name: String, model_name: String, resident: String,
    reqbytes: String, fd: Int,
) raises -> Int:
    """One HTTP request -> response. Returns 0 keep-alive, 1 close, 2 upgraded
    to the /v1/progress WebSocket."""
    try:
        var req = parse_request(reqbytes)
        if is_ws_upgrade(req):
            if req.path() != "/v1/progress":
                send_all_fd(Int32(fd), String("HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"))
                return 1
            var key = req.header("sec-websocket-key")
            if key == "":
                send_all_fd(Int32(fd), String("HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"))
                return 1
            send_all_fd(Int32(fd), handshake_response(key))
            print("WS client connected:", fd)
            return 2
        var ka = keep_alive_wanted(req)
        var resp = handle_api(jobs, njobs, running, prior, backend_name, model_name, resident, req)
        resp.set_header("Server", "SerenityDaemon/0.1")
        resp.set_header("Date", http_date())
        resp.set_keep_alive(ka)
        print(req.method, req.target, "->", resp.status)
        if req.method == "HEAD":
            send_all_fd(Int32(fd), resp.serialize_head())
        else:
            send_all_fd(Int32(fd), resp.serialize())
        return 0 if ka else 1
    except e:
        print("bad request:", e)
        send_all_fd(Int32(fd), String("HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"))
        return 1


# ── connection plumbing (examples/main.mojo pattern) ─────────────────────────
def close_conn(
    mut ep: Epoll, mut buffers: Dict[Int, String], mut ws: Dict[Int, Bool],
    mut frag: Dict[Int, WsReassembler], fd: Int,
) raises:
    ep.remove(Int32(fd))
    _ = sys_close(Int32(fd))
    if fd in buffers:
        _ = buffers.pop(fd)
    if fd in ws:
        _ = ws.pop(fd)
    if fd in frag:
        _ = frag.pop(fd)


def accept_new(mut ep: Epoll, mut buffers: Dict[Int, String], lfd: Int32) raises:
    var null = BytePtr(unsafe_from_address=Int(0))
    while True:
        var cfd = sys_accept(lfd, null, null)
        if cfd < 0:
            break
        set_nonblocking_fd(cfd)
        ep.add(cfd, EPOLLIN | EPOLLET, UInt64(Int(cfd)))
        buffers[Int(cfd)] = String("")


def handle_ws(
    mut ep: Epoll, mut buffers: Dict[Int, String], mut ws: Dict[Int, Bool],
    mut frag: Dict[Int, WsReassembler], fd: Int,
) raises:
    """Progress sockets are server-push; from the client we only honor
    ping (pong) and close. Text/binary input is ignored."""
    var tmp = alloc[UInt8](READ_CHUNK)
    var tp = BytePtr(unsafe_from_address=Int(tmp))
    var acc = buffers[fd] if fd in buffers else String("")
    var closed = False
    while True:
        var k = sys_recv(Int32(fd), tp, READ_CHUNK, Int32(0))
        if k > 0:
            acc += String(StringSlice(ptr=tmp, length=k))
        elif k == 0:
            closed = True
            break
        else:
            if errno() == EAGAIN:
                break
            closed = True
            break
    tmp.free()

    if fd not in frag:
        frag[fd] = WsReassembler()
    var r = frag[fd].copy()
    var should_close = closed
    while True:
        var fr = decode_frame(acc)
        if fr.consumed < 0:
            break
        acc = byte_substr(acc, fr.consumed, acc.byte_length())
        var msg = r.push(fr)
        if not msg.ready:
            continue
        if msg.opcode == OP_CLOSE:
            send_all_fd(Int32(fd), encode_close())
            should_close = True
            break
        elif msg.opcode == OP_PING:
            send_all_fd(Int32(fd), encode_pong(msg.payload))
        # OP_TEXT/OP_BINARY/pong: ignored (push-only channel)
    frag[fd] = r^
    if should_close:
        print("WS client disconnected:", fd)
        close_conn(ep, buffers, ws, frag, fd)
    else:
        buffers[fd] = acc


def handle_readable(
    mut ep: Epoll, mut buffers: Dict[Int, String], mut ws: Dict[Int, Bool],
    mut frag: Dict[Int, WsReassembler], mut jobs: List[JobRecord],
    mut njobs: Int, running: Int, prior: List[List[String]], backend_name: String,
    model_name: String, resident: String, fd: Int,
) raises:
    if fd in ws:
        handle_ws(ep, buffers, ws, frag, fd)
        return
    var tmp = alloc[UInt8](READ_CHUNK)
    var tp = BytePtr(unsafe_from_address=Int(tmp))
    var acc = buffers[fd] if fd in buffers else String("")
    var closed = False
    while True:
        var k = sys_recv(Int32(fd), tp, READ_CHUNK, Int32(0))
        if k > 0:
            acc += String(StringSlice(ptr=tmp, length=k))
        elif k == 0:
            closed = True
            break
        else:
            if errno() == EAGAIN:
                break
            closed = True
            break
    tmp.free()

    if acc.byte_length() > MAX_REQUEST_BYTES:
        send_all_fd(Int32(fd), String("HTTP/1.1 413 Payload Too Large\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"))
        close_conn(ep, buffers, ws, frag, fd)
        return

    var should_close = closed
    while is_request_complete(acc):
        var consumed = request_consumed_len(acc)
        if consumed < 0:
            break
        var reqbytes = byte_substr(acc, 0, consumed)
        acc = byte_substr(acc, consumed, acc.byte_length())
        var code = serve_request(jobs, njobs, running, prior, backend_name, model_name, resident, reqbytes, fd)
        if code == 2:  # upgraded to the progress WebSocket
            ws[fd] = True
            frag[fd] = WsReassembler()
            buffers[fd] = acc
            if acc.byte_length() > 0:
                handle_ws(ep, buffers, ws, frag, fd)
            return
        if code == 1:
            should_close = True
            break
    if should_close:
        close_conn(ep, buffers, ws, frag, fd)
    else:
        buffers[fd] = acc


# ── main ─────────────────────────────────────────────────────────────────────
def run_daemon[B: GenBackend](mut backend: B, port: Int) raises:
    _ = _system(String("mkdir -p ") + OUT_DIR)

    var backend_name = backend.backend_name()
    var model_name = backend.model_name()
    var resident = backend.resident_model()

    # F7: carry prior history into this session's rewrites and seed the job
    # counter past every id ever used (db rows AND output PNGs), so ids never
    # restart at job-0001 and outputs are never overwritten across runs.
    # (load_prior_rows also repairs a hard-killed session's "running" row to
    # "interrupted" — F10 crash-safety.)
    var prior = load_prior_rows()
    var prior_max = max_prior_id(prior)
    var png_max = max_output_png_id()
    if png_max > prior_max:
        prior_max = png_max
    if len(prior) > 0:
        save_jobs_db_safe(prior, List[JobRecord]())  # persist any repairs now

    var sock = listen_localhost(port)
    sock.set_nonblocking()
    var ep = Epoll()
    ep.add(sock.fd, EPOLLIN | EPOLLET, UInt64(Int(sock.fd)))
    var sigfd = install_signal_fd()  # SIGINT/SIGTERM as a pollable fd
    ep.add(sigfd, EPOLLIN, UInt64(Int(sigfd)))
    print("serenity_daemon (backend: " + backend_name + ") on http://127.0.0.1:" + String(port))

    var buffers = Dict[Int, String]()
    var ws = Dict[Int, Bool]()
    var frag = Dict[Int, WsReassembler]()
    var jobs = List[JobRecord]()
    var njobs = prior_max  # next id = prior_max + 1 (F7)
    var running = -1  # index into jobs of the in-flight job; -1 = idle
    var evbuf = alloc[UInt8](EVENT_SIZE * MAX_EVENTS)
    var evp = BytePtr(unsafe_from_address=Int(evbuf))
    var alive = True

    while alive:
        var nev = ep.wait(evp, Int32(MAX_EVENTS), Int32(TICK_MS))
        var lfd = Int(sock.fd)
        for i in range(Int(nev)):
            var fd = Int(rd_u64(evp, i * EVENT_SIZE + 4))
            if fd == Int(sigfd):
                # GRACEFUL SHUTDOWN IS BEST-EFFORT (measured 2026-06-10): the
                # signalfd only works when the MAIN thread wins the dequeue
                # race — the Mojo runtime spawns ~12 worker threads BEFORE
                # main() with OPEN signal masks (sigprocmask can't reach
                # them, and Mojo can't register a C-ABI sigaction handler),
                # so a process-directed SIGTERM is often dequeued by a worker
                # and default-kills us instantly or mid-cleanup. That is WHY
                # jobs.db is written crash-safely (running row at job START,
                # repaired to "interrupted" at next startup) instead of only
                # on this path. SIG_IGN narrows the mid-cleanup window when
                # we DO win the race. Deterministic graceful shutdown needs
                # a C-shim signal handler (MOJO-libs gap, descoped).
                _ = external_call["signal", UInt64](Int32(15), UInt64(1))  # SIGTERM -> SIG_IGN
                _ = external_call["signal", UInt64](Int32(2), UInt64(1))   # SIGINT  -> SIG_IGN
                print("signal received -> graceful shutdown", flush=True)
                alive = False
                break
            if fd == lfd:
                accept_new(ep, buffers, sock.fd)
            else:
                handle_readable(ep, buffers, ws, frag, jobs, njobs, running,
                                prior, backend_name, model_name, resident, fd)
        if not alive:
            break
        # the serial worker runs inside the loop: one bounded step per tick
        tick_worker(backend, jobs, running, prior, ws)
        resident = backend.resident_model()  # residency can change on any tick

    # F10: a SIGINT/SIGTERM mid-job must leave a truthful db row — record the
    # in-flight job as "interrupted" before exit (db failure logged, not fatal).
    if running >= 0 and not jobs[running].is_terminal():
        jobs[running].state = String("interrupted")
        print("job", jobs[running].params.job_id, "-> interrupted (shutdown)",
              flush=True)
        try:
            broadcast(ws, event_json(jobs[running], String(""), String("")))
        except:
            pass
        save_jobs_db_safe(prior, jobs)

    ep.remove(sock.fd)
    _ = sys_close(sigfd)
    evbuf.free()
    # F10: stdout is block-buffered when piped to a log file; the zimage run's
    # shutdown line was lost without an explicit flush.
    print("serenity_daemon exited cleanly", flush=True)


def main() raises:
    """serenity_daemon [stub|zimage|dispatch] [port] — backend defaults to
    stub, port to DEFAULT_PORT; the two args are recognized by shape in any
    order.

    * stub      — no model, no GPU (the smoke backend).
    * zimage    — Z-Image only (single resident backend; legacy single-model).
    * dispatch  — multi-model residency + on-demand SWITCHING (Phase 4): the
                  resident backend is chosen per job by `model` (zimage_base /
                  qwen-image-2512), freeing + loading on a switch. /v1/health +
                  /v1/models reflect the live residency.
    * isolated  — Phase 5: like dispatch, but each resident model runs in a CHILD
                  PROCESS; a switch KILLS the child so the OS reclaims its VRAM
                  (the only reclaim that works on this runtime). Lets zimage<->qwen
                  switching fit in 24 GB.

    Internal (fork+execv target, not user-facing):
    * worker <kind> <fd> — run ONE backend (stub|zimage|qwenimage|ideogram4|klein) over the
                  inherited socket `fd` instead of HTTP. Spawned by `isolated`."""
    var args = argv()
    # Internal worker entry MUST be handled before port/mode parsing (its extra
    # args are a kind + an fd, not a port). serenity_daemon worker <kind> <fd>.
    if len(args) >= 2 and String(args[1]) == "worker":
        if len(args) < 4:
            raise Error("worker mode needs: worker <kind> <fd>")
        var wkind = String(args[2])
        var wfd = Int32(Int(String(args[3])))
        run_worker(wkind, wfd)
        return
    var port = DEFAULT_PORT
    var mode = String("stub")
    for i in range(1, len(args)):
        var a = String(args[i])
        if a == "stub" or a == "zimage" or a == "dispatch" or a == "isolated":
            mode = a^
        else:
            port = Int(a)
    if mode == "dispatch":
        var db = DispatchBackend()
        run_daemon(db, port)
    elif mode == "isolated":
        var pb = ProcessIsolatedBackend()
        run_daemon(pb, port)
    elif mode == "zimage":
        var zb = ZImageBackend()
        run_daemon(zb, port)
    else:
        var sb = StubBackend()
        run_daemon(sb, port)
