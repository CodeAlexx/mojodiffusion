# serenitymojo.serve.serenity_worker_stub — minimal STANDALONE stub worker binary.
#
# Phase-A of the Rust-control-plane migration. Proves a non-Mojo (Rust) parent can
# spawn and drive a Mojo worker over the existing AF_UNIX newline-JSON IPC WITHOUT
# building the 57 MB serenity_daemon monolith. Imports ONLY the CPU stub backend +
# the IPC codec — NO GPU/MAX, NO server (serenity_daemon.mojo), NO graph executor
# (workflow_graph.mojo), the two measured OOM offenders. Builds tiny.
#
# Invoked:  serenity_worker_stub <fd>
#   <fd> = inherited AF_UNIX socket fd (decimal), same contract as the daemon's
#          internal `serenity_daemon worker stub <fd>` mode (worker.mojo).
#
# Wire (unchanged, ipc_codec.mojo):
#   parent->child : {"cmd":"start", <JobParams fields>} | {"cmd":"cancel"}
#   child->parent : {"ev":"ready"} | progress | done | failed | cancelled

from std.sys import argv
from std.time import sleep
from json.parser import loads

from serenitymojo.serve.backend import StepResult
from serenitymojo.serve.stub_backend import StubBackend
from serenitymojo.serve.proc_ipc import LineReader, write_msg, set_nonblock
from serenitymojo.serve.ipc_codec import decode_start, encode_ev, encode_ready

comptime WORKER_IDLE_SLEEP_S = 0.02  # poll cadence while waiting for a command


def _fail_line(msg: String) raises -> String:
    var r = StepResult()
    r.failed = True
    r.error = msg
    return encode_ev(r)


def _stub_worker_loop(mut backend: StubBackend, fd: Int32) raises:
    """Identical control flow to worker._worker_loop, specialized to StubBackend
    so this binary does not import the GPU backends. One job at a time."""
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
                # Per-job VRAM reclaim (no-op for the CPU stub; real for GPU kinds) —
                # mirrors the daemon loop (serenity_daemon.mojo:2276).
                try:
                    backend.between_jobs_trim()
                except e:
                    print("[worker] between_jobs_trim failed (continuing):", e)
                break


def main() raises:
    var args = argv()
    if len(args) < 2:
        print("usage: serenity_worker_stub <fd>")
        return
    var fd = Int32(Int(String(args[1])))
    var b = StubBackend()
    _stub_worker_loop(b, fd)
