# serenitymojo.serve.flux_encode_subprocess — the FLUX.1-dev text encoders
# (CLIP-L + T5-XXL) in a fork+execv CHILD process, so their VRAM is reclaimed by
# process death.
#
# WHY (MEASURED, flux job-0077 — the same live-leak class fixed for chroma in
# chroma_encode_subprocess.mojo, jobs 0075/0076): the per-job CLIP-L + T5-XXL
# encode IN the long-lived flux worker grows the process's CUDA pool from
# 1062 MiB to 13830 MiB and it NEVER comes back — "after text encode (encoders
# freed)" still reads 13830 MiB, the offloaded DiT then streams on top and the
# job OOMs mid-denoise at 14698 MiB on the 16 GB card. In-process free +
# cuMemPoolTrimTo reclaim ~0 (measured repeatedly on this card). The ONLY
# reliable reclaim is process exit: fork+execv a FRESH child that runs the
# VERIFIED flux_sample_cli.encode_text (CLIP-L pooled + T5-XXL hidden,
# byte-identical math — imported, not re-derived), writes both cap tensors to
# /tmp as bit-identical raw bytes (io.cap_cache), and EXITS. The parent
# waitpid's (blocking-reap → VRAM released before we touch the GPU again),
# reads the caps back (~8 MB H2D), and loads the DiT offloader from a ~1.3 GB
# floor.
#
# FLUX is guidance-distilled: only the POSITIVE prompt is encoded (the negative
# is discarded upstream — no CFG path), so the child takes just <prefix> and
# <prompt>. FLUX also has no T5 pad-row mask contract (the verified
# flux_sample_cli path attends all 512 rows), so the 24-byte meta sidecar
# carries the magic + two reserved zeros — its presence/magic is purely the
# "child fully succeeded" signal, mirroring the chroma/zimage protocol shape.
#
# SELF-EXEC (one binary, one build): the encode child IS serenity_worker_flux
# re-exec'd with argv ["encode-child", <prefix>, <prompt>];
# serenity_worker_flux.main routes that to encode_child_run(). No separate
# encoder binary.
#
# SAFETY / FALLBACK: for ANY subprocess failure (fork error, timeout, abnormal
# exit, unreadable caps) — or a host binary that does not route "encode-child" —
# encode_text_subprocess transparently falls back to the in-process
# encode_text. Correctness is never sacrificed for the VRAM win.

from std.ffi import external_call
from std.memory import alloc, UnsafePointer
from max.gpu.host import DeviceContext
from std.time import sleep

from serenitymojo.tensor import Tensor
from serenitymojo.io.cap_cache import save_tensor_bin, load_tensor_bin
from serenitymojo.io.ffi import (
    BytePtr, sys_open, sys_pwrite, sys_pread, sys_close,
    O_WRONLY, O_CREAT, O_TRUNC, O_RDONLY,
)
from serenitymojo.pipeline.flux_sample_cli import FluxCaps, encode_text
from serenitymojo.serve.proc_ipc import (
    build_argv, cstr, sys_execv, sys__exit, sys_waitpid, proc_kill_wait,
    SELF_EXE, SIGKILL, WNOHANG,
)
from serenitymojo.offload.vmm_cuda import cu_mem_get_info
from net.syscalls import sys_fork, errno_str


# encoder load (CLIP-L ~250 MB + T5-XXL ~9.5 GB F16 from disk) + two forwards is
# tens of seconds → 300 s is a pure hang backstop, after which we SIGKILL + fall
# back.
comptime _ENCODE_CHILD_TIMEOUT_S = 300.0
comptime _ENCODE_POLL_S = 0.05
# Pre-flight guard: the encoder child is a separate process whose peak (~9.5 GB
# F16 T5 weights + CLIP-L + cast copies + forward activations + its own CUDA
# context) must fit in the GPU's CURRENT free memory alongside this resident
# parent. Require ~11 GB free before forking, else go straight to in-process
# (the parent's grown pool can't be trimmed back to make room — measured
# cuMemPoolTrim reclaims 0). The flux parent sits at ~1 GB when the encode runs
# (the DiT offloader loads AFTER the encode and is freed before every decode),
# so a clean-card job always forks. Same threshold as chroma_encode_subprocess.
comptime _ENCODE_CHILD_MIN_FREE_BYTES = Int(11264) * 1024 * 1024  # ~11 GiB
# 24-byte binary sidecar: [magic][reserved 0][reserved 0] as 3x Int64 LE.
# (FLUX has no real-length/pad-mask contract — see module header.)
comptime _META_MAGIC = Int64(0x464C584341505631)  # "FLXCAPV1"


def _getpid() -> Int:
    return Int(external_call["getpid", Int32]())


def _unlink_file(path: String):
    """Best-effort delete of a /tmp cap sidecar (mirrors serenity_daemon._unlink_file)."""
    _ = external_call["unlink", Int32](cstr(path))


def _write_meta(path: String) raises:
    var fd = sys_open(path, O_WRONLY | O_CREAT | O_TRUNC, 0o644)
    if fd < 0:
        raise Error(String("flux_encode_subprocess: meta open failed: ") + path)
    var tmp = alloc[Int64](3)
    tmp[0] = _META_MAGIC
    tmp[1] = Int64(0)  # reserved (no pad-mask contract in the FLUX path)
    tmp[2] = Int64(0)  # reserved
    var p = BytePtr(unsafe_from_address=Int(tmp))
    var w = sys_pwrite(fd, p, 24, 0)
    tmp.free()
    _ = sys_close(fd)
    if w != 24:
        raise Error("flux_encode_subprocess: short meta write")


def _read_meta(path: String) raises:
    var fd = sys_open(path, O_RDONLY, 0)
    if fd < 0:
        raise Error(String("flux_encode_subprocess: meta open failed: ") + path)
    var tmp = alloc[Int64](3)
    var p = BytePtr(unsafe_from_address=Int(tmp))
    var r = sys_pread(fd, p, 24, 0)
    var magic = tmp[0]
    tmp.free()
    _ = sys_close(fd)
    if r != 24:
        raise Error("flux_encode_subprocess: short meta read")
    if magic != _META_MAGIC:
        raise Error("flux_encode_subprocess: bad meta magic")


def encode_child_run(prefix: String, prompt: String) raises:
    """CHILD body (after execv into `serenity_worker_flux encode-child`). Fresh
    process image → fresh CUDA context. Run the VERIFIED
    flux_sample_cli.encode_text (CLIP-L BPE + T5-XXL Unigram tokenize → CLIP-L
    pooled [1,768] BF16 + T5-XXL hidden [1,512,4096] BF16, encoders loaded +
    freed inside), write both caps to `<prefix>.{t5,clip}.bin` and the meta
    sidecar, then RETURN so the process exits and the OS reclaims ALL encoder
    VRAM. The `.meta` file is written LAST: its presence + magic is the
    parent's "child fully succeeded" signal."""
    var ctx = DeviceContext()
    var caps = encode_text(prompt, ctx)
    save_tensor_bin(caps.txt, prefix + String(".t5.bin"), ctx)
    save_tensor_bin(caps.vector, prefix + String(".clip.bin"), ctx)
    _write_meta(prefix + String(".meta"))
    print("[flux-encode-child] wrote caps", prefix)


def encode_text_subprocess(prompt: String, ctx: DeviceContext) raises -> FluxCaps:
    """PARENT body (flux worker ENCODE phase). fork+execv a fresh
    `serenity_worker_flux encode-child` to run the ~10 GB CLIP-L + T5-XXL
    encode in ITS OWN process, blocking-reap it (VRAM released by process
    death), then read back the BF16 caps it wrote and delete the /tmp sidecars.
    Nothing big is resident in THIS process during the encode (the DiT
    offloader loads after), so this CUDA context is never used in the child:
    fork copies the fd table and the child execv's immediately. Falls back to
    the in-process `encode_text` on any failure or on a host binary that does
    not route `encode-child` (see module header)."""
    var prefix = String("/tmp/serenity_flux_caps_") + String(_getpid())
    var t5_path = prefix + String(".t5.bin")
    var clip_path = prefix + String(".clip.bin")
    var meta_path = prefix + String(".meta")

    # Pre-flight guard: skip a doomed fork when the GPU's current free memory
    # can't hold the ~11 GB encoder child. cu_mem_get_info reports device-global
    # free, which is exactly what the child's separate CUDA context will see.
    # Below threshold → in-process encode (correct, just no reclaim win).
    var free_bytes = cu_mem_get_info().free_bytes
    if free_bytes < _ENCODE_CHILD_MIN_FREE_BYTES:
        print("[flux] free VRAM", free_bytes // (1024 * 1024),
              "MiB < encoder-child need", _ENCODE_CHILD_MIN_FREE_BYTES // (1024 * 1024),
              "MiB → in-process encode (no fork)")
        return encode_text(prompt, ctx)

    # argv + execv path built BEFORE fork (no allocation between fork and execv).
    var args = List[String]()
    args.append(SELF_EXE)                  # argv[0]
    args.append(String("encode-child"))
    args.append(prefix)
    args.append(prompt)
    var argv = build_argv(args)
    var path = cstr(SELF_EXE)

    print("[flux] fork encoder child (parent pid", _getpid(), ")")
    var pid = sys_fork()
    if pid == 0:
        # CHILD: async-signal-safe only, then execv into a fresh image.
        _ = sys_execv(path, argv)
        sys__exit(127)                     # execv failed
    if pid < 0:
        print("[flux] fork failed (", errno_str(), ") → in-process encode")
        return encode_text(prompt, ctx)

    # PARENT: bounded WNOHANG reap (hang backstop). Blocking-reap once it exits
    # so the OS has released the child's VRAM before we load the caps onto the
    # GPU.
    var st = alloc[Int32](1)
    var stp = rebind[UnsafePointer[Int32, MutExternalOrigin]](st)
    var waited = 0.0
    var reaped = Int32(0)
    while waited < _ENCODE_CHILD_TIMEOUT_S:
        reaped = sys_waitpid(pid, stp, WNOHANG)
        if reaped == pid:
            break
        if reaped < 0:
            break
        sleep(_ENCODE_POLL_S)
        waited += _ENCODE_POLL_S
    var status = Int(st[0])
    st.free()

    if reaped != pid:
        proc_kill_wait(pid, SIGKILL)
        print("[flux] encoder child timed out/errored → in-process encode")
        return encode_text(prompt, ctx)

    var exited_ok = (status & 0x7F) == 0 and ((status >> 8) & 0xFF) == 0
    if not exited_ok:
        print("[flux] encoder child abnormal exit (status", status,
              ") → in-process encode")
        return encode_text(prompt, ctx)

    # Success path: read the caps the child wrote, then delete the sidecars.
    # Any read failure (e.g. a host that exits 0 but never wrote the sidecar)
    # → in-process fallback.
    try:
        _read_meta(meta_path)
        var txt = load_tensor_bin(t5_path, ctx)
        var vector = load_tensor_bin(clip_path, ctx)
        _unlink_file(t5_path)
        _unlink_file(clip_path)
        _unlink_file(meta_path)
        print("[flux] encoder child reaped → caps loaded (encoder VRAM reclaimed)")
        return FluxCaps(vector^, txt^)
    except e:
        print("[flux] caps read-back failed (", e, ") → in-process encode")
        return encode_text(prompt, ctx)
