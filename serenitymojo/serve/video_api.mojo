# serenitymojo.serve.video_api — video contract helpers for serenity_daemon.
#
# This module owns the daemon-visible video readiness, MP4/A-V probing, and
# bounded LTX2 smoke runner result manifest. It is still a product wiring gate,
# not full SwarmUI video parity.

from std.ffi import external_call
from std.memory import alloc
from std.time import perf_counter_ns

from net.syscalls import BytePtr
from http.request import byte_substr
from json.parser import loads
from json.serialize import dumps
from json.value import JSONValue

from serenitymojo.serve.model_scan import _read_text_file

comptime OUT_DIR = "output/serenity_daemon"
comptime STATE_DIR = "output/serenity_daemon/state"
comptime VIDEO_PROBE_TMP = "output/serenity_daemon/state/video_probe.json"
comptime VIDEO_PROBE_ERR = "output/serenity_daemon/state/video_probe.err"
comptime LTX2_VIDEO_SMOKE_RUNNER = "output/bin/ltx2_video_smoke_runner"


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


def _ensure_state_dir():
    _ = _system(String("mkdir -p ") + _shell_quote(String(STATE_DIR)))


def _write_text_file(path: String, text: String) raises:
    _ensure_state_dir()
    with open(path, "w") as f:
        f.write(text)


def _opt_int(obj: JSONValue, key: String, dflt: Int, lo: Int, hi: Int) raises -> Int:
    if not obj.contains(key) or obj[key].is_null():
        return dflt
    if not obj[key].is_int():
        raise Error("'" + key + "' must be an integer")
    var n = obj[key].as_int()
    if n < lo or n > hi:
        raise Error("'" + key + "' out of range [" + String(lo) + ".." + String(hi) + "]")
    return n


def _opt_str(obj: JSONValue, key: String, dflt: String) raises -> String:
    if not obj.contains(key) or obj[key].is_null():
        return dflt
    if not obj[key].is_string():
        raise Error("'" + key + "' must be a string")
    return obj[key].as_string()


def _json_field_string(obj: JSONValue, key: String) raises -> String:
    if not obj.is_object() or not obj.contains(key) or obj[key].is_null():
        return String("")
    var v = obj[key]
    if v.is_string():
        return v.as_string()
    if v.is_int():
        return String(v.as_int())
    if v.is_number():
        return String(v.as_float())
    return String("")


def _json_field_float(obj: JSONValue, key: String, dflt: Float64) raises -> Float64:
    if not obj.is_object() or not obj.contains(key) or obj[key].is_null():
        return dflt
    var v = obj[key]
    if v.is_number():
        return v.as_float()
    if v.is_string():
        var s = String(v.as_string().strip())
        if s == "" or s == "N/A":
            return dflt
        try:
            return Float64(s)
        except:
            return dflt
    return dflt


def _json_field_int(obj: JSONValue, key: String, dflt: Int) raises -> Int:
    if not obj.is_object() or not obj.contains(key) or obj[key].is_null():
        return dflt
    var v = obj[key]
    if v.is_int():
        return v.as_int()
    if v.is_number():
        return Int(v.as_float())
    if v.is_string():
        var s = String(v.as_string().strip())
        if s == "" or s == "N/A":
            return dflt
        try:
            return Int(s)
        except:
            return dflt
    return dflt


def _fps_from_rate(rate: String) -> Float64:
    var r = String(rate.strip())
    if r == "" or r == "N/A":
        return 0.0
    var slash = r.find("/")
    try:
        if slash > 0:
            var num = Float64(String(byte_substr(r, 0, slash)))
            var den = Float64(String(byte_substr(r, slash + 1, r.byte_length())))
            if den == 0.0:
                return 0.0
            return num / den
        return Float64(r)
    except:
        return 0.0


def video_readiness_doc(
    backend_name: String, model_name: String, resident: String,
) raises -> JSONValue:
    var runners = JSONValue.new_array()

    var lance = JSONValue.new_object()
    lance.set("model", JSONValue.from_string(String("lance_t2v")))
    lance.set("status", JSONValue.from_string(String("smoke_only")))
    lance.set(
        "runner",
        JSONValue.from_string(
            String("serenitymojo/pipeline/lance_t2v_256_9f_dense_probe.mojo")
        ),
    )
    lance.set(
        "limit",
        JSONValue.from_string(
            String("standalone pipeline artifact gate; not daemon job-backed")
        ),
    )
    runners.append(lance^)

    var ltx2 = JSONValue.new_object()
    ltx2.set("model", JSONValue.from_string(String("ltx2_t2v_av")))
    var ltx2_ready = video_runner_available()
    ltx2.set(
        "status",
        JSONValue.from_string(
            String("bounded_smoke_ready") if ltx2_ready else String("runner_missing")
        ),
    )
    ltx2.set("runner", JSONValue.from_string(String(LTX2_VIDEO_SMOKE_RUNNER)))
    ltx2.set("mode", JSONValue.from_string(String("staged lora stream noaudio nonag")))
    ltx2.set("default_steps", JSONValue.from_int(1))
    ltx2.set("default_audio_mode", JSONValue.from_string(String("noaudio")))
    ltx2.set("target_width", JSONValue.from_int(768))
    ltx2.set("target_height", JSONValue.from_int(512))
    ltx2.set("target_frame_count", JSONValue.from_int(121))
    ltx2.set("target_fps", JSONValue.from_int(24))
    ltx2.set(
        "limit",
        JSONValue.from_string(
            String("bounded daemon smoke only; artifact acceptance requires a successful MP4/A-V probe with timings and VRAM evidence; full video parity remains separate")
        ),
    )
    runners.append(ltx2^)

    var o = JSONValue.new_object()
    o.set("schema", JSONValue.from_string(String("serenity.video_status.v1")))
    o.set("endpoint", JSONValue.from_string(String("/v1/video")))
    o.set(
        "state",
        JSONValue.from_string(
            String("bounded_smoke_ready") if ltx2_ready else String("runner_missing")
        ),
    )
    o.set(
        "readiness_label",
        JSONValue.from_string(
            String("bounded_daemon_smoke") if ltx2_ready else String("build_required")
        ),
    )
    o.set("accepted", JSONValue.from_bool(False))
    o.set("backend", JSONValue.from_string(backend_name))
    o.set("model", JSONValue.from_string(model_name))
    o.set("resident", JSONValue.from_string(resident))
    o.set("mp4", JSONValue.from_string(String("")))
    o.set("frame_count", JSONValue.from_int(0))
    o.set("duration", JSONValue.from_float(0.0))
    o.set("audio", JSONValue.from_bool(False))
    o.set(
        "non_acceptance_reason",
        JSONValue.from_string(
            String(
                "bounded smoke wiring is not full SwarmUI video parity; artifact acceptance requires frame_count, duration, muxing, audio behavior, timings, and VRAM evidence"
            )
        ),
    )
    o.set(
        "probe_endpoint",
        JSONValue.from_string(String("/v1/video/probe?path=<mp4>")),
    )
    o.set("candidate_runners", runners^)
    return o^


def video_runner_available() -> Bool:
    return _system(String("test -x ") + _shell_quote(String(LTX2_VIDEO_SMOKE_RUNNER))) == 0


def probe_video_file(mp4_path: String) raises -> JSONValue:
    if mp4_path == "":
        raise Error("'path' query parameter is required")
    if mp4_path.find("\n") >= 0 or mp4_path.find("\r") >= 0:
        raise Error("invalid video path")
    _ensure_state_dir()
    if _system(String("command -v ffprobe >/dev/null 2>&1")) != 0:
        raise Error("ffprobe is not available on PATH")

    var cmd = (
        String("ffprobe -v error -count_frames ")
        + String("-show_entries ")
        + String("stream=index,codec_type,codec_name,width,height,nb_frames,")
        + String("nb_read_frames,duration,avg_frame_rate ")
        + String("-show_entries format=duration,format_name ")
        + String("-of json ")
        + _shell_quote(mp4_path)
        + String(" > ")
        + _shell_quote(String(VIDEO_PROBE_TMP))
        + String(" 2> ")
        + _shell_quote(String(VIDEO_PROBE_ERR))
    )
    var rc = _system(cmd)
    if rc != 0:
        var err = String("")
        try:
            err = _read_text_file(String(VIDEO_PROBE_ERR))
        except:
            pass
        if err == "":
            err = String("ffprobe failed")
        raise Error(err)

    var probe = loads(_read_text_file(String(VIDEO_PROBE_TMP)))
    var format_duration = 0.0
    var format_name = String("")
    if probe.is_object() and probe.contains("format") and probe["format"].is_object():
        format_duration = _json_field_float(probe["format"], String("duration"), 0.0)
        format_name = _json_field_string(probe["format"], String("format_name"))

    var has_video = False
    var has_audio = False
    var width = 0
    var height = 0
    var frame_count = 0
    var duration = 0.0
    var fps = 0.0
    var video_codec = String("")
    var audio_codec = String("")
    var audio_duration = 0.0
    var stream_count = 0

    if probe.is_object() and probe.contains("streams") and probe["streams"].is_array():
        var streams = probe["streams"]
        stream_count = streams.length()
        for i in range(streams.length()):
            var s = streams[i]
            if not s.is_object():
                continue
            var typ = _json_field_string(s, String("codec_type"))
            if typ == "video" and not has_video:
                has_video = True
                width = _json_field_int(s, String("width"), 0)
                height = _json_field_int(s, String("height"), 0)
                video_codec = _json_field_string(s, String("codec_name"))
                duration = _json_field_float(s, String("duration"), 0.0)
                fps = _fps_from_rate(_json_field_string(s, String("avg_frame_rate")))
                frame_count = _json_field_int(s, String("nb_read_frames"), 0)
                if frame_count <= 0:
                    frame_count = _json_field_int(s, String("nb_frames"), 0)
                if frame_count <= 0 and duration > 0.0 and fps > 0.0:
                    frame_count = Int(duration * fps + 0.5)
            elif typ == "audio" and not has_audio:
                has_audio = True
                audio_codec = _json_field_string(s, String("codec_name"))
                audio_duration = _json_field_float(s, String("duration"), 0.0)

    if duration <= 0.0:
        duration = format_duration

    var o = JSONValue.new_object()
    o.set("schema", JSONValue.from_string(String("serenity.video_probe.v1")))
    o.set("mp4", JSONValue.from_string(mp4_path))
    o.set("format_name", JSONValue.from_string(format_name))
    o.set("stream_count", JSONValue.from_int(stream_count))
    o.set("has_video", JSONValue.from_bool(has_video))
    o.set("has_audio", JSONValue.from_bool(has_audio))
    o.set("audio", JSONValue.from_bool(has_audio))
    o.set("width", JSONValue.from_int(width))
    o.set("height", JSONValue.from_int(height))
    o.set("frame_count", JSONValue.from_int(frame_count))
    o.set("duration", JSONValue.from_float(duration))
    o.set("fps", JSONValue.from_float(fps))
    o.set("video_codec", JSONValue.from_string(video_codec))
    o.set("audio_codec", JSONValue.from_string(audio_codec))
    o.set("audio_duration", JSONValue.from_float(audio_duration))
    if has_video and frame_count > 0 and duration > 0.0:
        o.set("muxing", JSONValue.from_string(String("probe_ok")))
    else:
        o.set("muxing", JSONValue.from_string(String("incomplete_probe")))
    if has_audio:
        o.set("audio_behavior", JSONValue.from_string(String("audio_stream_present")))
    else:
        o.set("audio_behavior", JSONValue.from_string(String("video_only_no_audio_stream")))
    return o^


def ltx2_staged_smoke_video_result(
    body: JSONValue, video_id: String, backend_name: String,
    model_name: String, resident: String,
) raises -> JSONValue:
    var runner = _opt_str(body, "runner", String("ltx2_staged_dev_smoke"))
    if runner != "ltx2_staged_dev_smoke":
        raise Error(
            "unsupported video runner '" + runner
            + "'; supported runner is ltx2_staged_dev_smoke"
        )
    var steps = _opt_int(body, "steps", 1, 1, 3)
    var audio_mode = _opt_str(body, "audio_mode", String("noaudio"))
    if audio_mode == "video":
        audio_mode = String("noaudio")
    if audio_mode != "audio" and audio_mode != "noaudio":
        raise Error("unsupported audio_mode '" + audio_mode + "'; use audio or noaudio")
    if not video_runner_available():
        raise Error(
            String("missing executable ") + String(LTX2_VIDEO_SMOKE_RUNNER)
            + String("; run `pixi run build-video-smoke` first")
        )

    var out_dir = String(OUT_DIR) + String("/") + video_id
    var log_path = out_dir + String("/ltx2_video_runner.log")
    var mp4_path = out_dir + String("/ltx2_t2v_stage2_dev_smoke.mp4")
    var wav_path = String("")
    if audio_mode == "audio":
        mp4_path = out_dir + String("/ltx2_t2v_av_stage2_dev_smoke.mp4")
        wav_path = out_dir + String("/dev_audio.wav")
    var manifest_path = out_dir + String("/ltx2_video_result.json")
    _ = _system(String("mkdir -p ") + _shell_quote(out_dir))

    var cmd = (
        _shell_quote(String(LTX2_VIDEO_SMOKE_RUNNER))
        + String(" staged lora stream ")
        + audio_mode
        + String(" nonag ")
        + _shell_quote(out_dir)
        + String(" ")
        + String(steps)
        + String(" > ")
        + _shell_quote(log_path)
        + String(" 2>&1")
    )
    var t0 = perf_counter_ns()
    var rc = _system(cmd)
    var wall = Float64(perf_counter_ns() - t0) / 1.0e9

    var o = JSONValue.new_object()
    o.set("schema", JSONValue.from_string(String("serenity.video_result.v1")))
    o.set("video_id", JSONValue.from_string(video_id))
    o.set("runner", JSONValue.from_string(runner))
    o.set("backend", JSONValue.from_string(backend_name))
    o.set("model", JSONValue.from_string(model_name))
    o.set("resident", JSONValue.from_string(resident))
    o.set("readiness_label", JSONValue.from_string(String("bounded_daemon_smoke")))
    o.set("accepted_video_artifact", JSONValue.from_bool(False))
    o.set("accepted_av_artifact", JSONValue.from_bool(False))
    o.set("accepted_video_parity", JSONValue.from_bool(False))
    o.set("accepted_sampler_parity", JSONValue.from_bool(False))
    o.set("steps", JSONValue.from_int(steps))
    o.set("mode", JSONValue.from_string(String("staged lora stream ") + audio_mode + String(" nonag")))
    o.set("audio_mode", JSONValue.from_string(audio_mode))
    o.set("exit_code", JSONValue.from_int(rc))
    o.set("out_dir", JSONValue.from_string(out_dir))
    o.set("mp4", JSONValue.from_string(mp4_path))
    o.set("wav", JSONValue.from_string(wav_path))
    o.set("log_path", JSONValue.from_string(log_path))
    o.set("result_path", JSONValue.from_string(manifest_path))
    o.set("total_wall_seconds", JSONValue.from_float(wall))
    o.set(
        "note",
        JSONValue.from_string(
            String(
                "Daemon-backed LTX2 staged dev smoke. This proves product wiring "
                + "only when exit_code is zero and probe.muxing is probe_ok; it "
                + "does not claim full video parity."
            )
        ),
    )

    if rc != 0:
        o.set("state", JSONValue.from_string(String("failed")))
        o.set(
            "error",
            JSONValue.from_string(
                String("LTX2 staged smoke runner failed; inspect log_path")
            ),
        )
        _write_text_file(manifest_path, dumps(o))
        return o^

    try:
        var probe = probe_video_file(mp4_path)
        o.set("probe", probe.copy())
        o.set("state", JSONValue.from_string(String("done")))
        o.set("width", probe["width"])
        o.set("height", probe["height"])
        o.set("frame_count", probe["frame_count"])
        o.set("duration", probe["duration"])
        o.set("fps", probe["fps"])
        o.set("audio", probe["audio"])
        o.set("muxing", probe["muxing"])
        var has_audio = False
        if probe.contains("audio"):
            has_audio = probe["audio"].as_bool()
        o.set("accepted_video_artifact", JSONValue.from_bool(True))
        o.set(
            "accepted_av_artifact",
            JSONValue.from_bool(audio_mode == "audio" and has_audio),
        )
    except e:
        o.set("state", JSONValue.from_string(String("failed_probe")))
        o.set("error", JSONValue.from_string(String(e)))

    _write_text_file(manifest_path, dumps(o))
    return o^
