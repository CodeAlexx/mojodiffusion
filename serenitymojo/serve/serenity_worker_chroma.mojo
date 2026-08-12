# serenitymojo.serve.serenity_worker_chroma — STANDALONE per-kind Chroma worker.
#
# Per-kind sibling of serenity_worker_flux.mojo / serenity_worker_qwenimage.mojo /
# serenity_worker_sdxl.mojo / serenity_worker_zimage.mojo: the SAME AF_UNIX
# newline-JSON IPC loop, but driving the REAL ChromaBackend (GPU/MAX Chroma1-HD
# pipeline — runtime T5-XXL text encode (cond + uncond), block-streamed Chroma
# DiT with distilled-guidance approximator, real-CFG flow-match Euler denoise +
# FLUX VAE decode). The Rust control plane (serenity-server) spawns one of these
# per resident model, identically to the other workers: `serenity_worker_chroma <fd>`.
#
# !!! DO NOT BUILD THIS FILE INTERACTIVELY !!!
# Like the other per-kind workers, this entrypoint imports the full Chroma
# GPU+MAX stack; a bare `mojo build` (default -O3) pulls in heavy graph/kernel
# compilation that has OOM-killed the desktop. The ORCHESTRATOR capped-builds it
# via `pixi run build-worker-chroma-raw` (--optimization-level 2); never build it
# from an interactive session.
#
# Invoked:  serenity_worker_chroma <fd>
#   <fd> = inherited AF_UNIX socket fd (decimal), exactly as the flux/zimage/
#          qwenimage/sdxl workers.
#
# Wire (unchanged, ipc_codec.mojo):
#   parent->child : {"cmd":"start", <JobParams fields>} | {"cmd":"cancel"}
#   child->parent : {"ev":"ready"} | progress | done | failed | cancelled

from std.sys import argv
from std.time import sleep
from json.parser import loads

from serenitymojo.serve.backend import StepResult
from serenitymojo.serve.chroma_backend import ChromaBackend
from serenitymojo.serve.proc_ipc import LineReader, write_msg, set_nonblock
from serenitymojo.serve.ipc_codec import decode_start, encode_ev, encode_ready
from serenitymojo.serve.chroma_encode_subprocess import encode_child_run
from serenitymojo.serve.chroma_decode_subprocess import (
    decode_child_run as chroma_decode_child_run,
    decode_tiled_child_run as chroma_decode_tiled_child_run,
)

comptime WORKER_IDLE_SLEEP_S = 0.02  # poll cadence while waiting for a command


def _fail_line(msg: String) raises -> String:
    var r = StepResult()
    r.failed = True
    r.error = msg
    return encode_ev(r)


def _chroma_worker_loop(mut backend: ChromaBackend, fd: Int32) raises:
    """Identical control flow to the flux/zimage/qwenimage/sdxl worker loops,
    specialized to ChromaBackend. One job at a time (single-GPU contract)."""
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
                # (mirrors the flux / zimage / qwenimage / sdxl worker trim).
                try:
                    backend.between_jobs_trim()
                except e:
                    print("[worker] between_jobs_trim failed (continuing):", e)
                break


def main() raises:
    var args = argv()
    if len(args) < 2:
        print("usage: serenity_worker_chroma <fd> | serenity_worker_chroma encode-child <prefix> <prompt> <negative> | serenity_worker_chroma decode-child <latent_path> <rgb_out_path> <latent_h> <latent_w> | serenity_worker_chroma decode-tiled-child <latent_path> <rgb_out_path> <latent_h> <latent_w>")
        return
    # T5-encoder-in-a-child-process. The parent chroma worker fork+execv's THIS
    # same binary as `encode-child` so the ~9.5 GB T5-XXL encoder runs in a
    # separate process whose death reclaims the encoder VRAM (in-process free +
    # trim measured to reclaim ~0 — jobs 0075/0076). See
    # serve/chroma_encode_subprocess.mojo. Routed BEFORE the fd parse because
    # Int("encode-child") would raise.
    if String(args[1]) == "encode-child":
        if len(args) < 5:
            print("usage: serenity_worker_chroma encode-child <prefix> <prompt> <negative>")
            return
        encode_child_run(String(args[2]), String(args[3]), String(args[4]))
        return                            # process exits → encoder VRAM reclaimed
    # WHOLE-image VAE decode-in-a-child-process. The parent chroma worker
    # fork+execv's THIS same binary as `decode-child` so the 1024² whole decode
    # runs in a fresh CUDA context (measured job-0078: the ~11.7 GB decode peak
    # on the parent's ~3 GB floor OOMs the 16 GB card; a clean-context child
    # fits). See serve/chroma_decode_subprocess.mojo. Routed BEFORE the fd
    # parse because Int("decode-child") would raise.
    if String(args[1]) == "decode-child":
        if len(args) < 6:
            print("usage: serenity_worker_chroma decode-child <latent_path> <rgb_out_path> <latent_h> <latent_w>")
            return
        chroma_decode_child_run(
            String(args[2]), String(args[3]),
            Int(String(args[4])), Int(String(args[5])),
        )
        return                            # process exits → decode VRAM reclaimed
    if String(args[1]) == "decode-tiled-child":
        if len(args) < 6:
            print("usage: serenity_worker_chroma decode-tiled-child <latent_path> <rgb_out_path> <latent_h> <latent_w>")
            return
        chroma_decode_tiled_child_run(
            String(args[2]), String(args[3]),
            Int(String(args[4])), Int(String(args[5])),
        )
        return                            # process exits → tiled VAE VRAM reclaimed
    var fd = Int32(Int(String(args[1])))
    var b = ChromaBackend()
    _chroma_worker_loop(b, fd)
