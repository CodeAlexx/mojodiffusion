# serenitymojo.serve.serenity_worker_krea2 — STANDALONE per-kind Krea-2 worker.
#
# Per-kind sibling of serenity_worker_zimage.mojo: the SAME AF_UNIX newline-JSON IPC
# loop, driving the REAL Krea2Backend (fp8-resident DiT + Qwen3-VL-4B TE + VAE)
# instead of the CPU stub. serenity-server spawns one per resident model as
# `serenity_worker_krea2 <fd>`.
#
# !!! DO NOT BUILD THIS FILE HERE (interactively) !!!
# It imports krea2_backend → the full krea2 DiT/VAE/encoder GPU+MAX stack. A
# `mojo build` at -O3 has OOM-killed the desktop; build with -O2 to a scratch name,
# verify, then swap into output/bin/ (see KREA2_WIRE_STATUS.md build flags).
#
# Invoked:  serenity_worker_krea2 <fd>
#   <fd> = inherited AF_UNIX socket fd (decimal), exactly like the stub / zimage.
# Self-exec encode child (VRAM isolation, krea2_encode_subprocess.mojo):
#   serenity_worker_krea2 encode-child <prompt> <negative> <pos_bin> <neg_bin>
#
# Wire (unchanged, ipc_codec.mojo):
#   parent->child : {"cmd":"start", <JobParams fields>} | {"cmd":"cancel"}
#   child->parent : {"ev":"ready"} | progress | done | failed | cancelled

from std.sys import argv
from std.time import sleep
from json.parser import loads

from serenitymojo.serve.backend import StepResult
from serenitymojo.serve.krea2_backend import Krea2Backend
from serenitymojo.serve.proc_ipc import LineReader, write_msg, set_nonblock
from serenitymojo.serve.ipc_codec import decode_start, encode_ev, encode_ready
from serenitymojo.serve.krea2_encode_subprocess import krea2_encode_child_run

comptime WORKER_IDLE_SLEEP_S = 0.02  # poll cadence while waiting for a command


def _fail_line(msg: String) raises -> String:
    var r = StepResult()
    r.failed = True
    r.error = msg
    return encode_ev(r)


def _krea2_worker_loop(mut backend: Krea2Backend, fd: Int32) raises:
    """Identical control flow to the stub / zimage worker loop, specialized to
    Krea2Backend. One job at a time (single-GPU contract). Ready is written FIRST,
    before any heavy load, so the server's 15 s Ready handshake never trips."""
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
                # Reclaim the per-job transient device memory back to the OS between
                # jobs (the fp8-resident base is rebuilt per job in v1).
                try:
                    backend.between_jobs_trim()
                except e:
                    print("[worker] between_jobs_trim failed (continuing):", e)
                break


def main() raises:
    var args = argv()
    if len(args) < 2:
        print("usage: serenity_worker_krea2 <fd> | serenity_worker_krea2 encode-child <prompt> <negative> <pos_bin> <neg_bin>")
        return
    # Self-exec encode child: the parent worker fork+execv's THIS binary as
    # `encode-child` so the ~9.6 GB Qwen3-VL-4B TE runs in a separate process whose
    # death reclaims the encoder VRAM before the DiT resident base is built. Routed
    # BEFORE the fd parse because Int("encode-child") would raise.
    if String(args[1]) == "encode-child":
        if len(args) < 6:
            print("usage: serenity_worker_krea2 encode-child <prompt> <negative> <pos_bin> <neg_bin>")
            return
        krea2_encode_child_run(
            String(args[2]), String(args[3]), String(args[4]), String(args[5]),
        )
        return                            # process exits → encoder VRAM reclaimed
    var fd = Int32(Int(String(args[1])))
    var b = Krea2Backend()
    b.set_progress_fd(fd)
    _krea2_worker_loop(b, fd)
