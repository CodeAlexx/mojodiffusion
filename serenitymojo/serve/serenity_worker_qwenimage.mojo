# serenitymojo.serve.serenity_worker_qwenimage — STANDALONE per-kind Qwen-Image worker.
#
# Per-kind sibling of serenity_worker_zimage.mojo / serenity_worker_ideogram4.mojo: the
# SAME AF_UNIX newline-JSON IPC loop, but driving the REAL QwenImageBackend (GPU/MAX
# Qwen-Image pipeline — Qwen2.5-VL text encode + QwenImage DiT + VAE decode). The Rust
# control plane (serenity-server) spawns one of these per resident model, identically to
# the zimage/ideogram4 workers: `serenity_worker_qwenimage <fd>`.
#
# !!! DO NOT BUILD THIS FILE INTERACTIVELY !!!
# Like the other per-kind workers, this entrypoint imports the full Qwen-Image GPU+MAX
# stack; a bare `mojo build` (default -O3) pulls in heavy graph/kernel compilation that
# has OOM-killed the desktop. The ORCHESTRATOR capped-builds it via
# `pixi run build-worker-qwenimage-raw` (--optimization-level 2); never build it from an
# interactive session.
#
# Invoked:  serenity_worker_qwenimage <fd>
#   <fd> = inherited AF_UNIX socket fd (decimal), exactly as the zimage/ideogram4 workers.
#
# Wire (unchanged, ipc_codec.mojo):
#   parent->child : {"cmd":"start", <JobParams fields>} | {"cmd":"cancel"}
#   child->parent : {"ev":"ready"} | progress | done | failed | cancelled

from std.sys import argv
from std.time import sleep
from json.parser import loads

from serenitymojo.serve.backend import StepResult
from serenitymojo.serve.qwenimage_backend import QwenImageBackend
from serenitymojo.serve.proc_ipc import LineReader, write_msg, set_nonblock
from serenitymojo.serve.ipc_codec import decode_start, encode_ev, encode_ready

comptime WORKER_IDLE_SLEEP_S = 0.02  # poll cadence while waiting for a command


def _fail_line(msg: String) raises -> String:
    var r = StepResult()
    r.failed = True
    r.error = msg
    return encode_ev(r)


def _qwenimage_worker_loop(mut backend: QwenImageBackend, fd: Int32) raises:
    """Identical control flow to the zimage/ideogram4 worker loops, specialized to
    QwenImageBackend. One job at a time (single-GPU contract)."""
    set_nonblock(fd)
    write_msg(fd, encode_ready())
    var reader = LineReader(fd)
    var still_open = True
    while True:
        var line = reader.next_line(still_open)
        if not still_open:
            return                        # parent closed the socket -> exit
        if line == "":
            sleep(WORKER_IDLE_SLEEP_S)     # idle: no command yet
            continue
        var obj = loads(line)
        var cmd = obj["cmd"].as_string() if obj.contains("cmd") else String("")
        if cmd != "start":
            continue                      # ignore stray/cancel-when-idle
        var p = decode_start(obj)
        try:
            backend.start(p)
        except e:
            write_msg(fd, _fail_line(String("worker start failed: ") + String(e)))
            continue
        while True:
            var cl = reader.next_line(still_open)
            if not still_open:
                return
            if cl != "":
                var co = loads(cl)
                if co.contains("cmd") and co["cmd"].as_string() == "cancel":
                    backend.cancel()
            var r = backend.step()
            var terminal = r.is_terminal()
            write_msg(fd, encode_ev(r))
            if terminal:
                # Reclaim the per-job transient peak back to the OS between jobs
                # (mirrors the zimage / ideogram4 worker trim).
                try:
                    backend.between_jobs_trim()
                except e:
                    print("[worker] between_jobs_trim failed (continuing):", e)
                break


def main() raises:
    var args = argv()
    if len(args) < 2:
        print("usage: serenity_worker_qwenimage <fd>")
        return
    var fd = Int32(Int(String(args[1])))
    var b = QwenImageBackend()
    _qwenimage_worker_loop(b, fd)
