# serenitymojo.serve.ipc_codec — Phase-5 worker IPC message (de)serialization.
#
# The transport for process isolation: newline-framed JSON objects between the
# parent ProcessIsolatedBackend and a child worker. This module owns the wire
# schema so neither side re-implements it. Reuses the same json lib + field set
# as the daemon's parse_generate / job serializers (no NEW contract — process
# isolation is a transport change). See PHASE5_PROCESS_ISOLATION_DESIGN.md.
#
# Messages (one JSON object per line):
#   parent->child : {"cmd":"start", <all JobParams fields>}  |  {"cmd":"cancel"}
#   child->parent : {"ev":"ready"}
#                   {"ev":"progress","step":N,"total":M,"phase":"..."}
#                   {"ev":"done","output_path":"..."}
#                   {"ev":"failed","error":"..."}
#                   {"ev":"cancelled"}

from json.parser import loads
from json.serialize import dumps
from json.value import JSONValue

from serenitymojo.serve.backend import JobParams, LoraSpec, StepResult


# ── parent -> child ──────────────────────────────────────────────────────────

def encode_start(p: JobParams) raises -> String:
    """Serialize a fully-built JobParams (as the daemon already has it, with
    params_json populated) into a "start" command line. ALL fields ride,
    including job_id / out_dir / loras / params_json, so the child reconstructs
    an identical JobParams without re-running HTTP validation."""
    var o = JSONValue.new_object()
    o.set("cmd", JSONValue.from_string(String("start")))
    o.set("job_id", JSONValue.from_string(p.job_id))
    o.set("model", JSONValue.from_string(p.model))
    o.set("prompt", JSONValue.from_string(p.prompt))
    o.set("negative", JSONValue.from_string(p.negative))
    o.set("width", JSONValue.from_int(p.width))
    o.set("height", JSONValue.from_int(p.height))
    o.set("steps", JSONValue.from_int(p.steps))
    o.set("seed", JSONValue.from_int(p.seed))
    o.set("cfg", JSONValue.from_float(p.cfg))
    o.set("init_image", JSONValue.from_string(p.init_image))
    o.set("creativity", JSONValue.from_float(p.creativity))
    o.set("out_dir", JSONValue.from_string(p.out_dir))
    o.set("params_json", JSONValue.from_string(p.params_json))
    var la = JSONValue.new_array()
    for i in range(len(p.loras)):
        var lo = JSONValue.new_object()
        lo.set("name", JSONValue.from_string(p.loras[i].name))
        lo.set("weight", JSONValue.from_float(p.loras[i].weight))
        la.append(lo^)
    o.set("lora", la^)
    return dumps(o)


def decode_start(obj: JSONValue) raises -> JobParams:
    """Reconstruct JobParams from a parsed "start" object (inverse of
    encode_start). Reads fields directly — no defaulting/validation, the parent
    already validated at HTTP time."""
    var p = JobParams()
    p.job_id = obj["job_id"].as_string()
    p.model = obj["model"].as_string()
    p.prompt = obj["prompt"].as_string()
    p.negative = obj["negative"].as_string()
    p.width = Int(obj["width"].as_float())
    p.height = Int(obj["height"].as_float())
    p.steps = Int(obj["steps"].as_float())
    p.seed = Int(obj["seed"].as_float())
    p.cfg = obj["cfg"].as_float()
    p.init_image = obj["init_image"].as_string()
    p.creativity = obj["creativity"].as_float()
    p.out_dir = obj["out_dir"].as_string()
    p.params_json = obj["params_json"].as_string()
    if obj.contains("lora") and obj["lora"].is_array():
        var arr = obj["lora"]
        for i in range(arr.length()):
            var ent = arr[i]
            p.loras.append(LoraSpec(ent["name"].as_string(), ent["weight"].as_float()))
    return p^


def encode_cancel() raises -> String:
    var o = JSONValue.new_object()
    o.set("cmd", JSONValue.from_string(String("cancel")))
    return dumps(o)


# ── child -> parent ──────────────────────────────────────────────────────────

def encode_ready() raises -> String:
    var o = JSONValue.new_object()
    o.set("ev", JSONValue.from_string(String("ready")))
    return dumps(o)


def encode_ev(r: StepResult) raises -> String:
    """Serialize a StepResult into a child->parent event line."""
    var o = JSONValue.new_object()
    if r.done:
        o.set("ev", JSONValue.from_string(String("done")))
        o.set("output_path", JSONValue.from_string(r.output_path))
    elif r.failed:
        o.set("ev", JSONValue.from_string(String("failed")))
        o.set("error", JSONValue.from_string(r.error))
    elif r.cancelled:
        o.set("ev", JSONValue.from_string(String("cancelled")))
    else:
        o.set("ev", JSONValue.from_string(String("progress")))
        o.set("step", JSONValue.from_int(r.step))
        o.set("total", JSONValue.from_int(r.total))
        o.set("phase", JSONValue.from_string(r.phase))
        o.set("preview", JSONValue.from_string(r.preview))
    return dumps(o)


def decode_ev(obj: JSONValue) raises -> StepResult:
    """Inverse of encode_ev: a child->parent event line into a StepResult.
    A "ready" event maps to a plain no-progress StepResult (step=0)."""
    var r = StepResult()
    var ev = obj["ev"].as_string()
    if ev == "done":
        r.done = True
        r.output_path = obj["output_path"].as_string()
    elif ev == "failed":
        r.failed = True
        r.error = obj["error"].as_string()
    elif ev == "cancelled":
        r.cancelled = True
    elif ev == "progress":
        r.step = Int(obj["step"].as_float())
        r.total = Int(obj["total"].as_float())
        r.phase = obj["phase"].as_string()
        r.preview = obj["preview"].as_string()
    # "ready" -> empty StepResult (no-op tick)
    return r^
