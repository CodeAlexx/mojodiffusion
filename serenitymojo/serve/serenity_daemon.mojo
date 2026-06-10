# serenitymojo.serve.serenity_daemon — the SerenityUI generation daemon (skeleton).
#
# A pure-Mojo localhost HTTP + WebSocket server (default 127.0.0.1:7801) on the
# MOJO-libs net/http stack (single-threaded epoll event loop, the
# examples/main.mojo pattern). Endpoints:
#
#   POST /v1/generate     {model,prompt,negative,width,height,steps,seed,cfg,
#                          lora:[{name,weight}]} -> {job_id, queue_position}
#   GET  /v1/jobs         JSON array of all jobs
#   GET  /v1/job/<id>     one job
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
from json.parser import loads
from json.serialize import dumps
from json.value import JSONValue
from sqlite.db import Database
from sqlite.writer import DbWriter
from sqlite.value import Value

from serenitymojo.serve.backend import GenBackend, JobParams, LoraSpec, StepResult
from serenitymojo.serve.model_scan import (
    ScanEntry, scan_checkpoints, scan_loras, _read_text_file,
)
from serenitymojo.serve.stub_backend import StubBackend
from serenitymojo.serve.zimage_backend import ZImageBackend
from serenitymojo.serve.dispatch_backend import DispatchBackend

comptime DEFAULT_PORT = 7801
comptime MAX_EVENTS = 64
comptime READ_CHUNK = 65536
comptime MAX_REQUEST_BYTES = 1048576  # 1 MiB
comptime TICK_MS = 50                 # epoll wait timeout -> worker tick cadence
comptime OUT_DIR = "output/serenity_daemon"
comptime DB_PATH = "output/serenity_daemon/jobs.db"
comptime DB_PARAMS_MAX = 2048         # params_json cap in the db row (F1): the
                                      # pure-Mojo DbWriter has no overflow pages
                                      # (one row must fit one 4096-byte page);
                                      # the FULL params live in the PNG tEXt —
                                      # the db row is just the gallery index.
comptime SCAN_PNG_TMP = "/tmp/serenity_daemon_pngscan.txt"


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


def parse_generate(
    body: String, job_id: String, out_dir: String, default_model: String
) raises -> JobParams:
    """Validate POST /v1/generate's JSON body into JobParams (raises -> 422;
    a "[501]"-prefixed error -> 501). `default_model` is the serving backend's
    name so genparams never claim "stub" on a real-model daemon (F9)."""
    var obj = loads(body)
    if not obj.is_object():
        raise Error("body must be a JSON object")
    if obj.contains("workflow"):
        # plan H4: graph bodies are reserved in the schema, not implemented.
        raise Error(
            "[501] 'workflow' (graph body) is reserved and not implemented"
            " yet (plan H4); send flat genparams"
        )
    var p = JobParams()
    p.job_id = job_id
    p.out_dir = out_dir
    if not obj.contains("prompt") or not obj["prompt"].is_string():
        raise Error("'prompt' (string) is required")
    p.prompt = obj["prompt"].as_string()
    if p.prompt == "":
        raise Error("'prompt' must be non-empty")
    p.model = _opt_str(obj, "model", default_model)
    # P9/P10: the UI resolves prompt syntax at submit; `prompt` is already the
    # RESOLVED text and `prompt_raw` (the original, with syntax) is a
    # passthrough field persisted for reuse-params.
    var prompt_raw = _opt_str(obj, "prompt_raw", String(""))
    p.negative = _opt_str(obj, "negative", String(""))
    p.width = _opt_int(obj, "width", 512, 16, 2048)
    p.height = _opt_int(obj, "height", 512, 16, 2048)
    p.steps = _opt_int(obj, "steps", 20, 1, 500)
    p.seed = _opt_int(obj, "seed", 0, 0, 4294967295)
    p.cfg = _opt_num(obj, "cfg", 4.5, 0.0, 50.0)
    # Phase-2 genparams passthrough (plan H1): the backend ignores these today,
    # but they MUST survive into params_json (PNG tEXt + jobs.db) so the UI's
    # reuse-params restores ALL fields (P15) and the UI-state JSON == the
    # daemon-recorded JSON except server-added job_id (gate G2f).
    var sampler = _opt_str(obj, "sampler", String(""))
    var scheduler = _opt_str(obj, "scheduler", String(""))
    var variation_seed = _opt_int(obj, "variation_seed", 0, 0, 4294967295)
    var variation_strength = _opt_num(obj, "variation_strength", 0.0, 0.0, 1.0)
    var images = _opt_int(obj, "images", 1, 1, 64)
    # P7 img2img: init image path + creativity (0..1). The backend decides
    # whether/how to honor them (stub echoes; zimage does real img2img).
    p.init_image = _opt_str(obj, "init_image", String(""))
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

    # canonical full-param JSON: persisted to the sidecar + jobs.db
    # (key order mirrors the UI's serenity.genparams.v1 emitter, with the
    # server-added job_id second after schema)
    var o = JSONValue.new_object()
    o.set("schema", JSONValue.from_string(String("serenity.genparams.v1")))
    o.set("job_id", JSONValue.from_string(p.job_id))
    o.set("model", JSONValue.from_string(p.model))
    o.set("prompt", JSONValue.from_string(p.prompt))
    o.set("prompt_raw", JSONValue.from_string(prompt_raw))
    o.set("negative", JSONValue.from_string(p.negative))
    o.set("width", JSONValue.from_int(p.width))
    o.set("height", JSONValue.from_int(p.height))
    o.set("steps", JSONValue.from_int(p.steps))
    o.set("seed", JSONValue.from_int(p.seed))
    o.set("cfg", JSONValue.from_float(p.cfg))
    o.set("sampler", JSONValue.from_string(sampler))
    o.set("scheduler", JSONValue.from_string(scheduler))
    o.set("variation_seed", JSONValue.from_int(variation_seed))
    o.set("variation_strength", JSONValue.from_float(variation_strength))
    o.set("images", JSONValue.from_int(images))
    o.set("init_image", JSONValue.from_string(p.init_image))
    o.set("creativity", JSONValue.from_float(p.creativity))
    var la = JSONValue.new_array()
    for i in range(len(p.loras)):
        var lo = JSONValue.new_object()
        lo.set("name", JSONValue.from_string(p.loras[i].name))
        lo.set("weight", JSONValue.from_float(p.loras[i].weight))
        la.append(lo^)
    o.set("lora", la^)
    p.params_json = dumps(o)
    return p^


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
    o.set("output_path", JSONValue.from_string(j.output_path))
    o.set("error", JSONValue.from_string(j.error))
    return o^


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
        for k in range(6):
            vals.append(Value.text(prior[i][k]))
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


def handle_api(
    mut jobs: List[JobRecord], mut njobs: Int, running: Int,
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
        var ma = JSONValue.new_array()
        for i in range(len(models)):
            var mo = JSONValue.new_object()
            mo.set("name", JSONValue.from_string(models[i].name))
            mo.set("path", JSONValue.from_string(models[i].path))
            mo.set("arch", JSONValue.from_string(models[i].arch))
            mo.set("size", JSONValue.from_int(models[i].size))
            mo.set("loaded", JSONValue.from_bool(
                resident != "" and models[i].name == resident
            ))
            ma.append(mo^)
        var la = JSONValue.new_array()
        for i in range(len(loras)):
            var lo = JSONValue.new_object()
            lo.set("name", JSONValue.from_string(loras[i].name))
            lo.set("path", JSONValue.from_string(loras[i].path))
            lo.set("size", JSONValue.from_int(loras[i].size))
            la.append(lo^)
        var o = JSONValue.new_object()
        o.set("models", ma^)
        o.set("loras", la^)
        return json_response(200, dumps(o))

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
        # queue_position: jobs that will run before this one
        var ahead = 0
        if running >= 0:
            ahead += 1
        for i in range(len(jobs)):
            if jobs[i].state == "queued" and not jobs[i].cancel_requested:
                ahead += 1
        jobs.append(JobRecord(p^, http_date()))
        print("job", job_id, "queued (position", ahead, ")")
        var o = JSONValue.new_object()
        o.set("job_id", JSONValue.from_string(job_id))
        o.set("queue_position", JSONValue.from_int(ahead))
        return json_response(200, dumps(o))

    if req.method == "GET" and path == "/v1/jobs":
        var arr = JSONValue.new_array()
        for i in range(len(jobs)):
            arr.append(job_json_value(jobs[i]))
        return json_response(200, dumps(arr))

    if req.method == "GET" and path.startswith("/v1/job/"):
        var id = byte_substr(path, 8, path.byte_length())
        var i = _find_job(jobs, id)
        if i < 0:
            return error_response(404, "no such job: " + id)
        return json_response(200, dumps(job_json_value(jobs[i])))

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
        var resp = handle_api(jobs, njobs, running, backend_name, model_name, resident, req)
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
    mut njobs: Int, running: Int, backend_name: String, model_name: String,
    resident: String, fd: Int,
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
        var code = serve_request(jobs, njobs, running, backend_name, model_name, resident, reqbytes, fd)
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
                                backend_name, model_name, resident, fd)
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
                  /v1/models reflect the live residency."""
    var port = DEFAULT_PORT
    var mode = String("stub")
    var args = argv()
    for i in range(1, len(args)):
        var a = String(args[i])
        if a == "stub" or a == "zimage" or a == "dispatch":
            mode = a^
        else:
            port = Int(a)
    if mode == "dispatch":
        var db = DispatchBackend()
        run_daemon(db, port)
    elif mode == "zimage":
        var zb = ZImageBackend()
        run_daemon(zb, port)
    else:
        var sb = StubBackend()
        run_daemon(sb, port)
