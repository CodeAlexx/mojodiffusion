# serenitymojo.serve.worker — Phase-5 child-process worker.
#
# Reached via `serenity_daemon worker <kind> <fd>` (fork+execv'd by the parent
# ProcessIsolatedBackend). Runs ONE real backend (stub | zimage | qwenimage |
# ideogram4 | klein | sdxl | anima)
# driven by newline-framed JSON over the inherited socket `fd` instead of HTTP.
# A fresh process => fresh Mojo runtime + fresh CUDA context; the parent kills
# this process to reclaim VRAM on a model switch. See
# PHASE5_PROCESS_ISOLATION_DESIGN.md.

from std.time import sleep
from json.parser import loads

from serenitymojo.serve.backend import GenBackend, StepResult
from serenitymojo.serve.stub_backend import StubBackend
from serenitymojo.serve.zimage_backend import ZImageBackend
from serenitymojo.serve.qwenimage_backend import QwenImageBackend
from serenitymojo.serve.ideogram4_backend import Ideogram4Backend
from serenitymojo.serve.klein_backend import KleinBackend
from serenitymojo.serve.sample_cli_backend import SampleCliBackend
from serenitymojo.serve.proc_ipc import LineReader, write_msg, set_nonblock
from serenitymojo.serve.ipc_codec import (
    decode_start, encode_ev, encode_ready,
)

comptime WORKER_IDLE_SLEEP_S = 0.02  # poll cadence while waiting for a command


def _fail_line(msg: String) raises -> String:
    var r = StepResult()
    r.failed = True
    r.error = msg
    return encode_ev(r)


def _worker_loop[B: GenBackend](mut backend: B, fd: Int32) raises:
    """Serve commands over `fd` until the parent closes it. One job at a time
    (matches the single-GPU, single-in-flight daemon contract)."""
    set_nonblock(fd)
    write_msg(fd, encode_ready())
    var reader = LineReader(fd)
    var still_open = True
    while True:
        var line = reader.next_line(still_open)
        if not still_open:
            return                       # parent closed the socket -> exit
        if line == "":
            sleep(WORKER_IDLE_SLEEP_S)    # idle: no command yet
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
        # run this job to a terminal StepResult, streaming progress
        while True:
            # peek (non-blocking) for an interleaved cancel before stepping
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
                # Reclaim the per-job transient VRAM (e.g. Z-Image's ~7.5 GB Qwen3
                # encoder + decode activations) back to the OS — mirrors the daemon
                # job loop (serenity_daemon.mojo:2276). Without it, worker-mode parks
                # at the per-job VRAM peak instead of the resident baseline.
                try:
                    backend.between_jobs_trim()
                except e:
                    print("[worker] between_jobs_trim failed (continuing):", e)
                break


def run_worker(kind: String, fd: Int32) raises:
    """Construct the kind's backend and serve it over `fd`. `kind` is the parent's
    _kind_name string: "stub" (CPU) | "zimage" | "qwenimage" | "ideogram4" |
    "klein" | "sdxl" | "anima"."""
    if kind == "stub" or kind == "stub2":
        var b = StubBackend()      # stub2 = a 2nd distinct stub kind (CPU switch test)
        _worker_loop(b, fd)
    elif kind == "zimage":
        var b = ZImageBackend()
        _worker_loop(b, fd)
    elif kind == "qwenimage":
        var b = QwenImageBackend()
        _worker_loop(b, fd)
    elif kind == "ideogram4":
        var b = Ideogram4Backend()
        _worker_loop(b, fd)
    elif kind == "klein":
        var b = KleinBackend()
        _worker_loop(b, fd)
    elif kind == "sdxl":
        var b = SampleCliBackend(String("sdxl"))
        _worker_loop(b, fd)
    elif kind == "anima":
        var b = SampleCliBackend(String("anima"))
        _worker_loop(b, fd)
    else:
        raise Error("worker: unknown kind '" + kind + "' (want stub|zimage|qwenimage|ideogram4|klein|sdxl|anima)")
