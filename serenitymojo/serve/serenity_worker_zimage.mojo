# serenitymojo.serve.serenity_worker_zimage — STANDALONE per-kind Z-Image worker.
#
# Per-kind sibling of serenity_worker_stub.mojo: the SAME AF_UNIX newline-JSON IPC
# loop, but driving the REAL ZImageBackend (GPU/MAX Z-Image pipeline) instead of
# the CPU StubBackend. The Rust control plane (serenity-server) spawns one of these
# per resident model, identically to the stub: `serenity_worker_zimage <fd>`.
#
# !!! DO NOT BUILD THIS FILE HERE !!!
# Unlike serenity_worker_stub (tiny, CPU-only, desktop-safe), this entrypoint
# imports zimage_backend -> the full Z-Image DiT/VAE/encoder GPU+MAX stack. A
# `mojo build` of it pulls in the heavy graph/kernel compilation that has OOM-killed
# the GNOME desktop. The ORCHESTRATOR capped-builds it (e.g. a memory-limited
# `pixi run build-worker-zimage-safe`); never `mojo build` / `pixi run build-*` it
# from an interactive session.
#
# Invoked:  serenity_worker_zimage <fd>
#   <fd> = inherited AF_UNIX socket fd (decimal), exactly as the stub worker and
#          the daemon's internal `worker <kind> <fd>` mode (worker.mojo).
#
# Wire (unchanged, ipc_codec.mojo):
#   parent->child : {"cmd":"start", <JobParams fields>} | {"cmd":"cancel"}
#   child->parent : {"ev":"ready"} | progress | done | failed | cancelled

from std.sys import argv
from std.time import sleep
from json.parser import loads

from serenitymojo.serve.backend import StepResult
from serenitymojo.serve.zimage_backend import ZImageBackend
from serenitymojo.serve.proc_ipc import LineReader, write_msg, set_nonblock
from serenitymojo.serve.ipc_codec import decode_start, encode_ev, encode_ready
from serenitymojo.serve.zimage_encode_subprocess import encode_child_run

comptime WORKER_IDLE_SLEEP_S = 0.02  # poll cadence while waiting for a command


def _fail_line(msg: String) raises -> String:
    var r = StepResult()
    r.failed = True
    r.error = msg
    return encode_ev(r)


def _zimage_worker_loop(mut backend: ZImageBackend, fd: Int32) raises:
    """Identical control flow to worker._worker_loop and the stub worker loop,
    specialized to ZImageBackend so this binary's backend is the real Z-Image
    pipeline. One job at a time (single-GPU contract)."""
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
                # Reclaim the per-job ~7.5 GB Qwen3 encoder + decode activations back
                # to the OS (cuMemPoolTrimTo) — mirrors serenity_daemon.mojo:2276 so
                # worker-mode drops to the ~13 GB resident baseline between jobs
                # instead of parking at the ~21.5 GB per-job peak.
                try:
                    backend.between_jobs_trim()
                except e:
                    print("[worker] between_jobs_trim failed (continuing):", e)
                break


def main() raises:
    var args = argv()
    if len(args) < 2:
        print("usage: serenity_worker_zimage <fd> | serenity_worker_zimage encode-child <prefix> <prompt> <negative>")
        return
    # P4: encoder-in-a-child-process. The parent zimage worker fork+execv's THIS
    # same binary as `encode-child` so the ~7.5 GB Qwen3 encoder runs in a separate
    # process whose death reclaims the encoder VRAM (the resident DiT stays in the
    # parent). See serve/zimage_encode_subprocess.mojo. Routed BEFORE the fd parse
    # because Int("encode-child") would raise.
    if String(args[1]) == "encode-child":
        if len(args) < 5:
            print("usage: serenity_worker_zimage encode-child <prefix> <prompt> <negative>")
            return
        encode_child_run(String(args[2]), String(args[3]), String(args[4]))
        return                            # process exits → encoder VRAM reclaimed
    var fd = Int32(Int(String(args[1])))
    var b = ZImageBackend()
    _zimage_worker_loop(b, fd)
