# serenitymojo.serve.serenity_worker_lens — STANDALONE per-kind MS-Lens worker.
#
# Per-kind sibling of serenity_worker_sdxl.mojo / serenity_worker_qwenimage.mojo /
# serenity_worker_zimage.mojo / serenity_worker_ideogram4.mojo: the SAME AF_UNIX
# newline-JSON IPC loop, but driving the REAL LensBackend (GPU/MAX Microsoft-Lens
# pipeline — runtime GPT-OSS text encode + 48-block Lens DiT flow-match Euler CFG
# denoise + FLUX.2 VAE decode). The Rust control plane (serenity-server) spawns
# one of these per resident model, identically to the other workers:
# `serenity_worker_lens <fd>`.
#
# !!! DO NOT BUILD THIS FILE INTERACTIVELY !!!
# Like the other per-kind workers, this entrypoint imports the full Lens GPU+MAX
# stack (the streamed GPT-OSS encoder with on-GPU MXFP4 dequant, the streamed Lens
# DiT, the Flux2 VAE); a bare `mojo build` (default -O3) pulls in heavy
# graph/kernel compilation that has OOM-killed the desktop. The ORCHESTRATOR
# capped-builds it via a -O2 `pixi run build-worker-lens-raw`; never build it from
# an interactive session.
#
# Invoked:  serenity_worker_lens <fd>
#   <fd> = inherited AF_UNIX socket fd (decimal), exactly as the zimage/qwenimage/
#          sdxl/ideogram4 workers.
#
# Wire (unchanged, ipc_codec.mojo):
#   parent->child : {"cmd":"start", <JobParams fields>} | {"cmd":"cancel"}
#   child->parent : {"ev":"ready"} | progress | done | failed | cancelled
#
# NOTE (carried from lens_backend.mojo): the REAL text encode depends on
# models/text_encoder/gpt_oss_encoder.mojo, whose parity gate is BLOCKED by an
# open BF16→F32 cast bug in _moe (gated_scatter_add requires F32). Until that
# one-line fix lands, the FIRST job will fail-loud at the encode tick with
# "gated_scatter_add: expert_out and accum must be F32" (surfaced as a `failed`
# event, not a silent zeroed-text fallback). The worker loop itself is unaffected
# and stays alive for the next command.

from std.sys import argv
from std.time import sleep
from json.parser import loads

from serenitymojo.serve.backend import StepResult
from serenitymojo.serve.lens_backend import LensBackend
from serenitymojo.serve.proc_ipc import LineReader, write_msg, set_nonblock
from serenitymojo.serve.ipc_codec import decode_start, encode_ev, encode_ready

comptime WORKER_IDLE_SLEEP_S = 0.02  # poll cadence while waiting for a command


def _fail_line(msg: String) raises -> String:
    var r = StepResult()
    r.failed = True
    r.error = msg
    return encode_ev(r)


def _lens_worker_loop(mut backend: LensBackend, fd: Int32) raises:
    """Identical control flow to the sdxl/qwenimage/ideogram4 worker loops,
    specialized to LensBackend. One job at a time (single-GPU contract)."""
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
                # (mirrors the sdxl / qwenimage / ideogram4 worker trim).
                try:
                    backend.between_jobs_trim()
                except e:
                    print("[worker] between_jobs_trim failed (continuing):", e)
                break


def main() raises:
    var args = argv()
    if len(args) < 2:
        print("usage: serenity_worker_lens <fd>")
        return
    var fd = Int32(Int(String(args[1])))
    var b = LensBackend()
    _lens_worker_loop(b, fd)
