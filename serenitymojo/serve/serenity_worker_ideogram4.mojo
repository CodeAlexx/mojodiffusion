# serenitymojo.serve.serenity_worker_ideogram4 — STANDALONE per-kind Ideogram-4 worker.
#
# Per-kind sibling of serenity_worker_zimage.mojo: the SAME AF_UNIX newline-JSON IPC
# loop, but driving the REAL Ideogram4Backend (GPU/MAX Ideogram-4 fp8 pipeline). The
# Rust control plane (serenity-server) spawns one of these per resident model,
# identically to the zimage worker: `serenity_worker_ideogram4 <fd>`.
#
# !!! DO NOT BUILD THIS FILE INTERACTIVELY !!!
# Like the zimage worker, this entrypoint imports the full Ideogram-4 GPU+MAX stack;
# a bare `mojo build` pulls in heavy graph/kernel compilation that has OOM-killed the
# desktop. The ORCHESTRATOR capped-builds it (e.g. a memory-limited
# `pixi run build-worker-ideogram4-raw` inside an outer systemd scope); never build it
# from an interactive session.
#
# Invoked:  serenity_worker_ideogram4 <fd>
#   <fd> = inherited AF_UNIX socket fd (decimal), exactly as the stub/zimage workers
#          and the daemon's internal `worker <kind> <fd>` mode (worker.mojo:98).
#
# Wire (unchanged, ipc_codec.mojo):
#   parent->child : {"cmd":"start", <JobParams fields>} | {"cmd":"cancel"}
#   child->parent : {"ev":"ready"} | progress | done | failed | cancelled

from std.sys import argv
from std.memory import alloc, UnsafePointer
from std.ffi import external_call
from std.time import sleep
from json.parser import loads

from serenitymojo.serve.backend import StepResult
from serenitymojo.serve.ideogram4_backend import Ideogram4Backend
from serenitymojo.serve.proc_ipc import LineReader, write_msg, set_nonblock
from serenitymojo.serve.ipc_codec import decode_start, encode_ev, encode_ready

comptime WORKER_IDLE_SLEEP_S = 0.02  # poll cadence while waiting for a command


def _fail_line(msg: String) raises -> String:
    var r = StepResult()
    r.failed = True
    r.error = msg
    return encode_ev(r)


def _ideogram4_worker_loop(mut backend: Ideogram4Backend, fd: Int32) raises:
    """Identical control flow to worker._worker_loop and the zimage worker loop,
    specialized to Ideogram4Backend. One job at a time (single-GPU contract)."""
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
                # (mirrors the zimage worker / serenity_daemon trim).
                try:
                    backend.between_jobs_trim()
                except e:
                    print("[worker] between_jobs_trim failed (continuing):", e)
                break


comptime _CStr = UnsafePointer[UInt8, MutExternalOrigin]


def _cstr(s: String) -> _CStr:
    var n = s.byte_length()
    var buf = alloc[UInt8](n + 1)
    var src = s.as_bytes()
    for i in range(n):
        buf[i] = src[i]
    buf[n] = 0
    return buf


def main() raises:
    # MJ-1142: ideogram4's dual-trunk CFG runs two transient size families
    # (cond S=TOTAL, uncond S=NIMG) at 1024^2 — the default async device
    # allocator grows to the cumulative transient peak and OOMs a direct
    # cuBLAS workspace alloc with ~12GB free (measured: variable death step
    # 3-13, flat 11.9GB used). The runtime's SYNCHRONOUS allocator frees
    # back to the driver each op; house precedent: nldiffusion
    # tiled_decode_probe. setenv(overwrite=0) respects an operator override;
    # must run BEFORE the first DeviceContext().
    _ = external_call["setenv", Int32](
        _cstr(String("MODULAR_DEVICE_CONTEXT_SYNC_MODE")),
        _cstr(String("true")),
        Int32(0),
    )
    var args = argv()
    if len(args) < 2:
        print("usage: serenity_worker_ideogram4 <fd>")
        return
    var fd = Int32(Int(String(args[1])))
    var b = Ideogram4Backend()
    _ideogram4_worker_loop(b, fd)
