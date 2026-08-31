# Pure-Mojo SerenityFlow MiniMax-H3 endless workflow runner.
#
# Accepts the exact Comfy API-prompt JSON containing MiniMaxH3EndlessLatent
# and MiniMaxH3EndlessSampler, lowers it to bounded native H3 calls, carries a
# synchronized latent reference plus protected target boundary between calls,
# checkpoints after each completed chunk, and publishes one exact-duration MP4.
# Only the Mojo inference runtime participates in sampling this path.

from std.collections import List
from std.sys import argv

from json.parser import loads
from json.value import JSONValue

from serenitymojo.components.artifacts import shell_quote
from serenitymojo.io.ffi import sys_system
from serenitymojo.serve.model_scan import _read_text_file
from serenitymojo.serve.product_manifest import json_bool, json_escape, write_text_file
from serenitymojo.models.minimax_h3.endless import (
    MiniMaxH3EndlessChunk,
    MINIMAX_H3_ENDLESS_FPS,
    minimax_h3_endless_snap_frames_up,
    minimax_h3_plan_endless_chunks,
)


comptime DEFAULT_RUNNER = "output/bin/minimax_h3_serenity_runtime"


@fieldwise_init
struct MiniMaxH3EndlessWorkflow(Copyable, Movable):
    var opening_prompt: String
    var continuation_prompt: String
    var width: Int
    var height: Int
    var duration_seconds: Int
    var output_frames: Int
    var internal_frames: Int
    var seed: Int
    var steps: Int
    var chunk_frames: Int
    var reference_frames: Int
    var boundary_frames: Int
    var attention_backend: String
    var checkpoint_name: String
    var resume: Bool


def _input(node: JSONValue, key: String) raises -> JSONValue:
    if not node.is_object() or not node.contains(String("inputs")):
        raise Error("MiniMax-H3 SerenityFlow node has no inputs")
    var inputs = node[String("inputs")]
    if not inputs.is_object() or not inputs.contains(key):
        raise Error(String("MiniMax-H3 SerenityFlow node is missing input '") + key + String("'"))
    return inputs[key].copy()


def _input_int(node: JSONValue, key: String) raises -> Int:
    var value = _input(node, key)
    if value.is_int():
        return value.as_int()
    if value.is_number():
        return Int(value.as_float())
    raise Error(String("MiniMax-H3 SerenityFlow input '") + key + String("' must be numeric"))


def _input_string(node: JSONValue, key: String) raises -> String:
    var value = _input(node, key)
    if not value.is_string():
        raise Error(String("MiniMax-H3 SerenityFlow input '") + key + String("' must be a string"))
    return value.as_string()


def _input_bool(node: JSONValue, key: String) raises -> Bool:
    var value = _input(node, key)
    if not value.is_bool():
        raise Error(String("MiniMax-H3 SerenityFlow input '") + key + String("' must be boolean"))
    return value.as_bool()


def _find_nth_node(
    graph: JSONValue, class_name: String, wanted: Int
) raises -> JSONValue:
    var seen = 0
    for index in range(1, 4097):
        var key = String(index)
        if not graph.contains(key):
            continue
        var node = graph[key]
        if not node.is_object() or not node.contains(String("class_type")):
            continue
        var typ = node[String("class_type")]
        if typ.is_string() and typ.as_string() == class_name:
            if seen == wanted:
                return node.copy()
            seen += 1
    raise Error(String("SerenityFlow workflow is missing ") + class_name)


def _parse_workflow(text: String) raises -> MiniMaxH3EndlessWorkflow:
    var graph = loads(text)
    if not graph.is_object():
        raise Error("SerenityFlow MiniMax-H3 workflow must be a Comfy API prompt object")
    var opening = _find_nth_node(graph, String("MiniMaxH3TextEncode"), 0)
    var continuation = _find_nth_node(graph, String("MiniMaxH3TextEncode"), 1)
    var loader = _find_nth_node(graph, String("MiniMaxH3Loader"), 0)
    var latent = _find_nth_node(graph, String("MiniMaxH3EndlessLatent"), 0)
    var sampler = _find_nth_node(graph, String("MiniMaxH3EndlessSampler"), 0)
    _ = _find_nth_node(graph, String("MiniMaxH3DecodeRelease"), 0)
    _ = _find_nth_node(graph, String("CreateVideo"), 0)
    _ = _find_nth_node(graph, String("SaveVideo"), 0)

    var duration = _input_int(latent, String("duration"))
    if duration < 5 or duration > 3600:
        raise Error("MiniMax-H3 endless duration must be from 5 to 3600 seconds")
    var output_frames = duration * MINIMAX_H3_ENDLESS_FPS
    var internal_frames = minimax_h3_endless_snap_frames_up(output_frames)
    var sampler_name = _input_string(sampler, String("sampler_name"))
    var scheduler = _input_string(sampler, String("scheduler"))
    if sampler_name != String("euler") or scheduler != String("normal"):
        raise Error("MiniMax-H3 endless native path requires euler + normal")
    return MiniMaxH3EndlessWorkflow(
        _input_string(opening, String("prompt")),
        _input_string(continuation, String("prompt")),
        _input_int(latent, String("width")),
        _input_int(latent, String("height")),
        duration,
        output_frames,
        internal_frames,
        _input_int(sampler, String("seed")),
        _input_int(sampler, String("steps")),
        _input_int(sampler, String("chunk_frames")),
        _input_int(sampler, String("reference_frames")),
        _input_int(sampler, String("boundary_frames")),
        _input_string(loader, String("attention_backend")),
        _input_string(sampler, String("checkpoint_name")),
        _input_bool(sampler, String("resume")),
    )


def _fingerprint(
    workflow: MiniMaxH3EndlessWorkflow,
    quant: String,
    attention: String,
) raises -> String:
    return (
        String("serenity.mojo.minimax_h3.endless.v1|")
        + String(workflow.width) + String("x") + String(workflow.height)
        + String("|") + String(workflow.duration_seconds)
        + String("|") + String(workflow.seed)
        + String("|") + String(workflow.steps)
        + String("|") + String(workflow.chunk_frames)
        + String("|") + String(workflow.reference_frames)
        + String("|") + String(workflow.boundary_frames)
        + String("|") + quant
        + String("|") + attention
        + String("|") + workflow.opening_prompt
        + String("|") + workflow.continuation_prompt
    )


def _checkpoint_json(
    workflow: MiniMaxH3EndlessWorkflow,
    fingerprint: String,
    completed: Int,
    total: Int,
) raises -> String:
    return (
        String("{\n")
        + String("  \"schema\":\"serenity.mojo.minimax_h3.endless.checkpoint.v1\",\n")
        + String("  \"fingerprint\":\"") + json_escape(fingerprint) + String("\",\n")
        + String("  \"checkpoint_name\":\"")
        + json_escape(workflow.checkpoint_name) + String("\",\n")
        + String("  \"completed_chunks\":") + String(completed) + String(",\n")
        + String("  \"total_chunks\":") + String(total) + String(",\n")
        + String("  \"complete\":") + json_bool(completed == total) + String("\n")
        + String("}\n")
    )


def _completed_from_checkpoint(
    path: String, fingerprint: String, total: Int
) raises -> Int:
    if sys_system(String("test -f ") + shell_quote(path)) != 0:
        return 0
    var state = loads(_read_text_file(path))
    if not state.is_object() \
            or not state.contains(String("schema")) \
            or not state[String("schema")].is_string() \
            or state[String("schema")].as_string() \
                != String("serenity.mojo.minimax_h3.endless.checkpoint.v1"):
        raise Error("MiniMax-H3 endless checkpoint schema is incompatible")
    if not state.contains(String("fingerprint")) \
            or not state[String("fingerprint")].is_string() \
            or state[String("fingerprint")].as_string() != fingerprint:
        raise Error("MiniMax-H3 endless checkpoint does not match this workflow")
    if not state.contains(String("completed_chunks")) \
            or not state[String("completed_chunks")].is_int():
        raise Error("MiniMax-H3 endless checkpoint has no valid chunk count")
    var completed = state[String("completed_chunks")].as_int()
    if completed < 0 or completed > total:
        raise Error("MiniMax-H3 endless checkpoint has an invalid chunk count")
    return completed


def _write_checkpoint_atomic(path: String, contents: String) raises:
    var temporary = path + String(".tmp")
    write_text_file(temporary, contents)
    if sys_system(
        String("mv -f -- ") + shell_quote(temporary) + String(" ")
        + shell_quote(path)
    ) != 0:
        raise Error("MiniMax-H3 endless checkpoint could not be published")


def _verify_completed_chunks(root: String, completed: Int) raises:
    for index in range(completed):
        var chunk = _chunk_dir(root, index)
        if sys_system(
            String("test -s ") + shell_quote(chunk + String("/video.mp4"))
        ) != 0 or sys_system(
            String("test -s ")
            + shell_quote(chunk + String("/motion_context.safetensors"))
        ) != 0:
            raise Error(
                String("MiniMax-H3 endless checkpoint references incomplete chunk ")
                + String(index)
            )


def _chunk_manifest(plan: List[MiniMaxH3EndlessChunk]) -> String:
    var out = String("[\n")
    for i in range(len(plan)):
        var c = plan[i].copy()
        out += String("    {")
        out += String("\"index\":") + String(c.index)
        out += String(",\"frame_start\":") + String(c.frame_start)
        out += String(",\"frame_end\":") + String(c.frame_end)
        out += String(",\"sample_frames\":") + String(c.sample_frames)
        out += String(",\"new_frames\":") + String(c.new_frames)
        out += String(",\"output_trim_frames\":") + String(c.output_trim_frames)
        out += String(",\"sample_video_t\":") + String(c.sample_video_t)
        out += String(",\"boundary_video_t\":") + String(c.boundary_video_t)
        out += String(",\"sample_audio_t\":") + String(c.sample_audio_t)
        out += String(",\"context_audio_t\":") + String(c.context_audio_t)
        out += String(",\"new_audio_t\":") + String(c.new_audio_t)
        out += String(",\"reference_video_t\":") + String(c.reference_video_t)
        out += String(",\"reference_audio_t\":") + String(c.reference_audio_t)
        out += String("}")
        if i + 1 < len(plan):
            out += String(",")
        out += String("\n")
    out += String("  ]")
    return out^


def _chunk_dir(root: String, index: Int) -> String:
    return root + String("/chunk-") + String(index)


def _runtime_command(
    runner: String,
    prompt: String,
    root: String,
    out_dir: String,
    workflow: MiniMaxH3EndlessWorkflow,
    chunk: MiniMaxH3EndlessChunk,
    quant: String,
    attention: String,
    decode_mode: String,
) -> String:
    var decode = decode_mode != String("")
    var command = String("")
    if decode_mode == String("decode_video_only"):
        # The optimized H3 VAE needs Mojo GPU allocator headroom on 16 GiB
        # cards. This is the established fresh video-decode process policy.
        command += String(
            "MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT=55 "
            "MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_CHUNK_PERCENT=100 "
        )
    command += shell_quote(runner)
    if decode:
        command += String(" decode ") + shell_quote(out_dir)
    else:
        command += String(" ") + shell_quote(prompt) + String(" ") + shell_quote(out_dir)
    command += String(" ") + String(workflow.steps)
    command += String(" ") + String(workflow.seed + chunk.index)
    command += String(" 50")
    if decode:
        command += String(" ") + decode_mode
    command += String(" --width=") + String(workflow.width)
    command += String(" --height=") + String(workflow.height)
    command += String(" --frames=") + String(chunk.sample_frames)
    command += String(" --output-frames=") + String(chunk.new_frames)
    command += String(" --fps=24 --output-fps=24")
    command += String(" --quant=") + quant
    command += String(" --attention-backend=") + attention
    command += String(" --step-cache=exact --resident-blocks=0")
    command += String(" --encoder-storage=int8")
    if not decode:
        command += String(" --defer-video-decode")
    if chunk.index > 0 and not decode:
        var previous = _chunk_dir(root, chunk.index - 1)
        command += String(" --motion-context=") \
            + shell_quote(previous + String("/motion_context.safetensors"))
        command += String(" --motion-context-frames=") \
            + String(workflow.reference_frames)
        command += String(" --trim-start-frames=") \
            + String(workflow.boundary_frames)
        command += String(" --endless-boundary-frames=") \
            + String(workflow.boundary_frames)
        command += String(" --endless-boundary-audio-latents=") \
            + String(chunk.context_audio_t)
    return command^


def _mux_chunk(
    workflow: MiniMaxH3EndlessWorkflow,
    chunk: MiniMaxH3EndlessChunk,
    out_dir: String,
) raises -> String:
    var output_seconds = Float64(chunk.new_frames) / 24.0
    var output = out_dir + String("/video.mp4")
    var command = String("ffmpeg -v error -y -f rawvideo -pixel_format rgb24")
    command += String(" -video_size ") + String(workflow.width) \
        + String("x") + String(workflow.height)
    command += String(" -framerate 24")
    if chunk.output_trim_frames > 0:
        command += String(" -skip_initial_bytes ") + String(
            chunk.output_trim_frames * workflow.width * workflow.height * 3
        )
    command += String(" -i ") + shell_quote(out_dir + String("/frames.rgb"))
    if chunk.output_trim_frames > 0:
        command += String(" -ss ") + String(
            Float64(chunk.output_trim_frames) / 24.0
        )
    command += String(" -i ") + shell_quote(out_dir + String("/audio.wav"))
    command += String(" -frames:v ") + String(chunk.new_frames)
    var tail = String(" -pix_fmt yuv420p -af apad,atrim=duration=")
    tail += String(output_seconds) + String(" -c:a aac -t ")
    tail += String(output_seconds) + String(" -movflags +faststart ")
    tail += shell_quote(output)
    var nvenc = command + String(
        " -c:v h264_nvenc -preset p7 -tune hq -rc vbr -cq 18 -b:v 0"
    ) + tail
    if sys_system(nvenc) != 0:
        var portable = command + String(
            " -c:v libx264 -preset medium -crf 18"
        ) + tail
        if sys_system(portable) != 0:
            raise Error("MiniMax-H3 endless chunk A/V mux failed")
    return output^


def _assemble(
    root: String,
    chunks: Int,
    output_frames: Int,
) raises -> String:
    var concat_path = root + String("/segments.ffconcat")
    var concat_text = String("ffconcat version 1.0\n")
    for index in range(chunks):
        concat_text += String("file '") + _chunk_dir(root, index) \
            + String("/video.mp4'\n")
    write_text_file(concat_path, concat_text)
    var output = root + String("/video.mp4")
    var duration = Float64(output_frames) / 24.0
    var base = (
        String("ffmpeg -y -hide_banner -loglevel error -f concat -safe 0 -i ")
        + shell_quote(concat_path)
        + String(" -vf fps=24 -frames:v ") + String(output_frames)
        + String(" -t ") + String(duration)
        + String(" -c:a aac -ar 32000 -ac 2 -shortest ")
    )
    var nvenc = base + String(" -c:v h264_nvenc -preset p4 ") + shell_quote(output)
    if sys_system(nvenc) != 0:
        var portable = base + String(" -c:v libx264 -preset medium -crf 18 ") \
            + shell_quote(output)
        if sys_system(portable) != 0:
            raise Error("MiniMax-H3 endless final A/V assembly failed")
    return output^


def main() raises:
    var args = argv()
    if len(args) < 3:
        print(
            "usage: minimax_h3_endless <serenityflow_workflow.json> <out_dir>"
            " [--runner=PATH] [--quant=bf16|int8|int8-fast]"
            " [--attention-backend=cudnn|ck-int8|sage-int8] [--plan-only]"
        )
        return
    var workflow_path = String(args[1])
    var out_dir = String(args[2])
    if out_dir.find(String("\n")) >= 0 or out_dir.find(String("'")) >= 0:
        raise Error(
            "MiniMax-H3 endless output path cannot contain a newline or apostrophe"
        )
    var runner = String(DEFAULT_RUNNER)
    var quant = String("bf16")
    var attention_override = String("")
    var plan_only = False
    var force_no_resume = False
    for i in range(3, len(args)):
        var arg = String(args[i])
        if arg.startswith("--runner="):
            runner = String(arg.split("=")[1])
        elif arg.startswith("--quant="):
            quant = String(arg.split("=")[1])
        elif arg.startswith("--attention-backend="):
            attention_override = String(arg.split("=")[1])
        elif arg == String("--plan-only"):
            plan_only = True
        elif arg == String("--no-resume"):
            force_no_resume = True
        else:
            raise Error(String("unknown MiniMax-H3 endless option: ") + arg)
    if quant != String("bf16") and quant != String("int8") \
            and quant != String("int8-fast"):
        raise Error("MiniMax-H3 endless quant must be bf16, int8, or int8-fast")

    var workflow_text = _read_text_file(workflow_path)
    var workflow = _parse_workflow(workflow_text)
    var plan = minimax_h3_plan_endless_chunks(
        workflow.internal_frames,
        workflow.chunk_frames,
        workflow.reference_frames,
        workflow.boundary_frames,
    )
    var attention = (
        attention_override
        if attention_override != String("") else workflow.attention_backend
    )
    if attention != String("cudnn") and attention != String("ck-int8") \
            and attention != String("sage-int8"):
        raise Error(
            "MiniMax-H3 endless attention backend must be cudnn, ck-int8,"
            " or sage-int8"
        )
    var fingerprint = _fingerprint(workflow, quant, attention)
    var manifest = (
        String("{\n")
        + String("  \"schema\":\"serenity.mojo.minimax_h3.endless.plan.v1\",\n")
        + String("  \"source\":\"SerenityFlow/MiniMaxH3EndlessSampler\",\n")
        + String("  \"backend\":\"mojo\",\n")
        + String("  \"width\":") + String(workflow.width) + String(",\n")
        + String("  \"height\":") + String(workflow.height) + String(",\n")
        + String("  \"output_frames\":") + String(workflow.output_frames) + String(",\n")
        + String("  \"internal_frames\":") + String(workflow.internal_frames) + String(",\n")
        + String("  \"fps\":24,\n")
        + String("  \"chunk_frames\":") + String(workflow.chunk_frames) + String(",\n")
        + String("  \"reference_frames\":") + String(workflow.reference_frames) + String(",\n")
        + String("  \"boundary_frames\":") + String(workflow.boundary_frames) + String(",\n")
        + String("  \"quant\":\"") + json_escape(quant) + String("\",\n")
        + String("  \"attention_backend\":\"")
        + json_escape(attention) + String("\",\n")
        + String("  \"fingerprint\":\"") + json_escape(fingerprint) + String("\",\n")
        + String("  \"chunks\":") + _chunk_manifest(plan) + String("\n}\n")
    )
    print(manifest)
    if plan_only:
        return
    if sys_system(String("test -x ") + shell_quote(runner)) != 0:
        raise Error(String("MiniMax-H3 endless runner is missing: ") + runner)
    if sys_system(String("mkdir -p ") + shell_quote(out_dir)) != 0:
        raise Error("MiniMax-H3 endless output directory could not be created")
    write_text_file(out_dir + String("/plan.json"), manifest)
    write_text_file(out_dir + String("/workflow.json"), workflow_text)
    var checkpoint_path = out_dir + String("/checkpoint.json")
    var completed = 0
    if workflow.resume and not force_no_resume:
        completed = _completed_from_checkpoint(
            checkpoint_path, fingerprint, len(plan)
        )
        _verify_completed_chunks(out_dir, completed)
    for index in range(completed, len(plan)):
        var chunk = plan[index].copy()
        var chunk_out = _chunk_dir(out_dir, index)
        if sys_system(String("mkdir -p ") + shell_quote(chunk_out)) != 0:
            raise Error("MiniMax-H3 endless chunk directory could not be created")
        var prompt = (
            workflow.opening_prompt if index == 0 else workflow.continuation_prompt
        )
        var denoise = _runtime_command(
            runner, prompt, out_dir, chunk_out, workflow, chunk,
            quant, attention, String("")
        )
        print("[endless] denoise chunk", index + 1, "of", len(plan))
        if sys_system(denoise) != 0:
            raise Error(String("MiniMax-H3 endless denoise failed at chunk ") + String(index))
        var decode_video = _runtime_command(
            runner, String("decode"), out_dir, chunk_out, workflow, chunk,
            quant, attention, String("decode_video_only"),
        )
        print("[endless] decode video chunk", index + 1, "of", len(plan))
        if sys_system(decode_video) != 0:
            raise Error(
                String("MiniMax-H3 endless video decode failed at chunk ")
                + String(index)
            )
        _ = _mux_chunk(workflow, chunk, chunk_out)
        if sys_system(String("test -s ") + shell_quote(chunk_out + String("/video.mp4"))) != 0 \
                or sys_system(String("test -s ") + shell_quote(chunk_out + String("/motion_context.safetensors"))) != 0:
            raise Error("MiniMax-H3 endless chunk did not publish its A/V and continuation artifacts")
        completed = index + 1
        _write_checkpoint_atomic(
            checkpoint_path,
            _checkpoint_json(workflow, fingerprint, completed, len(plan)),
        )

    var artifact = _assemble(out_dir, len(plan), workflow.output_frames)
    var result = (
        String("{\n")
        + String("  \"schema\":\"serenity.mojo.minimax_h3.endless.result.v1\",\n")
        + String("  \"state\":\"done\",\n")
        + String("  \"backend\":\"mojo\",\n")
        + String("  \"artifact_path\":\"") + json_escape(artifact) + String("\",\n")
        + String("  \"frame_count\":") + String(workflow.output_frames) + String(",\n")
        + String("  \"fps\":24,\n")
        + String("  \"duration\":") + String(Float64(workflow.output_frames) / 24.0) + String(",\n")
        + String("  \"completed_chunks\":") + String(len(plan)) + String("\n")
        + String("}\n")
    )
    write_text_file(out_dir + String("/result.json"), result)
    print("[endless] complete ->", artifact)
