# serenitymojo.serve.serenity_worker_klein — STANDALONE per-kind Klein worker.
#
# Per-kind sibling of serenity_worker_sdxl.mojo / serenity_worker_qwenimage.mojo /
# serenity_worker_zimage.mojo / serenity_worker_ideogram4.mojo: the SAME AF_UNIX
# newline-JSON IPC loop, but driving the REAL in-process KleinRuntimeBackend
# (runtime Qwen3 tokenize+encode of params.prompt/negative -> Klein Euler
# flow-match denoise + Klein VAE decode + PNG save, all in this process — NO
# precache sidecar files, NO external sampler process). The Rust control plane
# (serenity-server) spawns one of these per resident model, identically to the
# zimage/qwenimage/sdxl/ideogram4 workers: `serenity_worker_klein <fd>`.
#
# !!! DO NOT BUILD THIS FILE INTERACTIVELY !!!
# Like the other per-kind workers, this entrypoint imports the full Klein GPU+MAX
# stack (Qwen3 encoder + Klein DiT + VAE); a bare `mojo build` (default -O3) pulls
# in heavy graph/kernel compilation that has OOM-killed the desktop. The
# ORCHESTRATOR capped-builds it via the pinned -O2 worker recipe; never build it
# from an interactive session.
#
# Invoked:  serenity_worker_klein <fd>
#   <fd> = inherited AF_UNIX socket fd (decimal), exactly as the other workers.
#
# Wire (unchanged, ipc_codec.mojo):
#   parent->child : {"cmd":"start", <JobParams fields>} | {"cmd":"cancel"}
#   child->parent : {"ev":"ready"} | progress | done | failed | cancelled

from std.sys import argv
from std.time import sleep
from json.parser import loads

from serenitymojo.serve.backend import StepResult
from serenitymojo.serve.klein_runtime_backend import (
    KleinRuntimeBackend, klein_encode_child_run,
)
from serenitymojo.serve.proc_ipc import LineReader, write_msg, set_nonblock
from serenitymojo.serve.ipc_codec import decode_start, encode_ev, encode_ready
from serenitymojo.serve.klein_decode_subprocess import (
    decode_child_run as klein_decode_child_run,
)

comptime WORKER_IDLE_SLEEP_S = 0.02  # poll cadence while waiting for a command


def _fail_line(msg: String) raises -> String:
    var r = StepResult()
    r.failed = True
    r.error = msg
    return encode_ev(r)


def _klein_worker_loop(mut backend: KleinRuntimeBackend, fd: Int32) raises:
    """Identical control flow to the zimage/qwenimage/sdxl/ideogram4 worker loops,
    specialized to KleinRuntimeBackend. One job at a time (single-GPU contract)."""
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
                # (mirrors the zimage / qwenimage / sdxl / ideogram4 worker trim).
                try:
                    backend.between_jobs_trim()
                except e:
                    print("[worker] between_jobs_trim failed (continuing):", e)
                break


def main() raises:
    var args = argv()
    if len(args) < 2:
        print("usage: serenity_worker_klein <fd> | serenity_worker_klein encode-child <prefix> <variant> <prompt> <negative> <joint> | serenity_worker_klein decode-child <latent> <rgb_out> <vae> <lh> <lw>")
        return
    if String(args[1]) == "encode-child":
        if len(args) < 7:
            print("usage: serenity_worker_klein encode-child <prefix> <variant> <prompt> <negative> <joint>")
            return
        klein_encode_child_run(
            String(args[2]), String(args[3]), String(args[4]), String(args[5]),
            Int(String(args[6])),
        )
        return                            # process exits -> encoder VRAM reclaimed
    # WHOLE-image VAE decode-in-a-child-process. klein_sample (allow_child_decode)
    # fork+execv's THIS same binary as `decode-child` so the whole-image decode runs
    # in a fresh CUDA context with its own pool (tiled decode degrades output —
    # MJ-1054), then this process exits and the OS reclaims the decode VRAM. See
    # serve/klein_decode_subprocess.mojo. Routed BEFORE the fd parse because
    # Int("decode-child") would raise.
    if String(args[1]) == "decode-child":
        if len(args) < 7:
            print("usage: serenity_worker_klein decode-child <latent> <rgb_out> <vae> <lh> <lw>")
            return
        klein_decode_child_run(
            String(args[2]), String(args[3]), String(args[4]),
            Int(String(args[5])), Int(String(args[6])),
        )
        return                            # process exits → decode VRAM reclaimed
    var fd = Int32(Int(String(args[1])))
    var b = KleinRuntimeBackend()
    b.set_progress_fd(fd)
    _klein_worker_loop(b, fd)
