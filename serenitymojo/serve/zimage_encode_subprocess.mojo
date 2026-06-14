# serenitymojo.serve.zimage_encode_subprocess — P4: the Qwen3 text encoder in a
# fork+execv CHILD process, so its VRAM is reclaimed by process death.
#
# WHY (MEASURED): the Qwen3-4B encoder (~7.5 GB) loaded IN the long-lived zimage
# worker gets stuck in that process's CUDA memory pool, fragmented around the
# resident ~11.5 GB DiT — cuMemPoolTrimTo(0) reclaims ~0 (between_jobs_trim
# measured "reclaimed 0 MiB"). The ONLY reliable reclaim is process exit. So on a
# conditioning-cache MISS the worker fork+execv's a FRESH child that loads the
# encoder, encodes (prompt, negative), writes the BF16 caps to disk (bit-identical
# raw bytes via io.cap_cache), and EXITS — the OS frees every byte of the encoder
# VRAM. The parent waitpid's (blocking-reap → VRAM released before we touch the GPU
# again), reads the caps back (a one-time ~2.6 MB H2D, NOT per-step weight paging),
# and denoises at the ~13 GB resident baseline.
#
# This is the SAME split the Klein-9B pipeline already uses (io/cap_cache.mojo
# header: "process death frees every byte of encoder GPU memory before the DiT
# process ever starts"), under the Phase-5 process-isolation contract
# (proc_ipc.mojo: argv built BEFORE fork, async-signal-safe calls only between
# fork and execv, execv into a fresh image — fork()ing a CUDA/AsyncRT process is
# only safe if the child does nothing but execv).
#
# SELF-EXEC (one binary, one build): the encode child IS serenity_worker_zimage
# re-exec'd with argv ["encode-child", <prefix>, <prompt>, <negative>];
# serenity_worker_zimage.main routes that to encode_child_run(). No separate
# encoder binary.
#
# SAFETY / FALLBACK: ZImageBackend is also constructed by hosts whose main() does
# NOT route "encode-child" (serenity_daemon, worker.mojo, dispatch_backend). For
# those — and for ANY subprocess failure (fork error, timeout, abnormal exit,
# unreadable caps) — encode_captions_subprocess transparently falls back to the
# in-process encode_captions_fixed. Correctness is never sacrificed for the VRAM
# win; only the live serenity_worker_zimage path gets the reclaim.

from std.ffi import external_call
from std.memory import alloc, UnsafePointer
from std.builtin.type_aliases import MutExternalOrigin
from std.gpu.host import DeviceContext
from std.time import sleep

from serenitymojo.pipeline.zimage_generate import (
    encode_captions_fixed, CapFeatsFixed,
)
from serenitymojo.io.cap_cache import save_tensor_bin, load_tensor_bin
from serenitymojo.io.ffi import (
    BytePtr, sys_open, sys_pwrite, sys_pread, sys_close,
    O_WRONLY, O_CREAT, O_TRUNC, O_RDONLY,
)
from serenitymojo.serve.proc_ipc import (
    build_argv, cstr, sys_execv, sys__exit, sys_waitpid, proc_kill_wait,
    SELF_EXE, SIGKILL, WNOHANG,
)
from serenitymojo.offload.vmm_cuda import cu_mem_get_info
from net.syscalls import sys_fork, errno_str


# encoder load (7.5 GB) + cond+uncond forward is a few seconds → ~30 s; this is a
# pure hang backstop, far above any real encode, after which we SIGKILL + fall back.
comptime _ENCODE_CHILD_TIMEOUT_S = 300.0
comptime _ENCODE_POLL_S = 0.05
# Pre-flight guard: the encoder child is a separate process whose ~10 GB peak
# (7.5 GB trimmed Qwen3 weights + forward activations + its own CUDA context) must
# fit in the GPU's CURRENT free memory alongside this resident parent. MEASURED
# child footprint ≈ 10.1 GB; require this much free before forking, else go
# straight to in-process (the parent's grown pool can't be trimmed back to make
# room — cuMemPoolTrim reclaims 0). Tuned so a clean-parent job (free ≈ 11.4 GB)
# forks while a grown-parent job (free ≈ 8.5 GB) does not.
comptime _ENCODE_CHILD_MIN_FREE_BYTES = Int(10800) * 1024 * 1024  # ~10.8 GiB
# 24-byte binary sidecar: [magic][real_cond][real_uncond] as 3x Int64 LE.
comptime _META_MAGIC = Int64(0x5A494D4341505631)  # "ZIMCAPV1"


def _getpid() -> Int:
    return Int(external_call["getpid", Int32]())


def _write_meta(path: String, real_cond: Int, real_uncond: Int) raises:
    var fd = sys_open(path, O_WRONLY | O_CREAT | O_TRUNC, 0o644)
    if fd < 0:
        raise Error(String("zimage_encode_subprocess: meta open failed: ") + path)
    var tmp = alloc[Int64](3)
    tmp[0] = _META_MAGIC
    tmp[1] = Int64(real_cond)
    tmp[2] = Int64(real_uncond)
    var p = BytePtr(unsafe_from_address=Int(tmp))
    var w = sys_pwrite(fd, p, 24, 0)
    tmp.free()
    _ = sys_close(fd)
    if w != 24:
        raise Error("zimage_encode_subprocess: short meta write")


def _read_meta(path: String) raises -> List[Int]:
    var fd = sys_open(path, O_RDONLY, 0)
    if fd < 0:
        raise Error(String("zimage_encode_subprocess: meta open failed: ") + path)
    var tmp = alloc[Int64](3)
    var p = BytePtr(unsafe_from_address=Int(tmp))
    var r = sys_pread(fd, p, 24, 0)
    var magic = tmp[0]
    var rc = Int(tmp[1])
    var ru = Int(tmp[2])
    tmp.free()
    _ = sys_close(fd)
    if r != 24:
        raise Error("zimage_encode_subprocess: short meta read")
    if magic != _META_MAGIC:
        raise Error("zimage_encode_subprocess: bad meta magic")
    var out = List[Int]()
    out.append(rc)
    out.append(ru)
    return out^


def encode_child_run(prefix: String, prompt: String, negative: String) raises:
    """CHILD body (after execv into `serenity_worker_zimage encode-child`). Fresh
    process image → fresh CUDA context. Load the encoder, encode cond+uncond, write
    the BF16 caps + token counts to `<prefix>.{cond,uncond}.bin` / `<prefix>.meta`,
    then RETURN so the process exits and the OS reclaims ALL encoder VRAM. The
    `.meta` file is written LAST: its presence + magic is the parent's
    "child fully succeeded" signal."""
    var ctx = DeviceContext()
    var caps = encode_captions_fixed(prompt, negative, ctx)
    save_tensor_bin(caps.cond, prefix + String(".cond.bin"), ctx)
    save_tensor_bin(caps.uncond, prefix + String(".uncond.bin"), ctx)
    _write_meta(prefix + String(".meta"), caps.real_cond, caps.real_uncond)
    print("[zimage-encode-child] wrote caps", prefix,
          "real_cond=", caps.real_cond, "real_uncond=", caps.real_uncond)


def encode_captions_subprocess(
    prompt: String, negative: String, ctx: DeviceContext
) raises -> CapFeatsFixed:
    """PARENT body (zimage worker, conditioning-cache MISS). fork+execv a fresh
    `serenity_worker_zimage encode-child` to run the 7.5 GB Qwen3 encoder in ITS
    OWN process, blocking-reap it (VRAM released), then read back the BF16 caps it
    wrote. The resident DiT in THIS process is untouched: fork copies the fd table
    and the child execv's immediately, so this CUDA context is never used in the
    child. Falls back to in-process `encode_captions_fixed` on any failure or on a
    host binary that does not route `encode-child` (see module header)."""
    var prefix = String("/tmp/serenity_zimage_caps_") + String(_getpid())
    var cond_path = prefix + String(".cond.bin")
    var uncond_path = prefix + String(".uncond.bin")
    var meta_path = prefix + String(".meta")

    # Pre-flight guard: skip a doomed fork (and its transient >24 GB spike) when the
    # GPU's current free memory can't hold the ~10 GB encoder child. cu_mem_get_info
    # reports device-global free, which is exactly what the child's separate CUDA
    # context will see. Below threshold → in-process encode (correct, just no win).
    var free_bytes = cu_mem_get_info().free_bytes
    if free_bytes < _ENCODE_CHILD_MIN_FREE_BYTES:
        print("[zimage] free VRAM", free_bytes // (1024 * 1024),
              "MiB < encoder-child need", _ENCODE_CHILD_MIN_FREE_BYTES // (1024 * 1024),
              "MiB → in-process encode (no fork)")
        return encode_captions_fixed(prompt, negative, ctx)

    # argv + execv path built BEFORE fork (no allocation between fork and execv).
    var args = List[String]()
    args.append(SELF_EXE)                  # argv[0]
    args.append(String("encode-child"))
    args.append(prefix)
    args.append(prompt)
    args.append(negative)
    var argv = build_argv(args)
    var path = cstr(SELF_EXE)

    print("[zimage] cache MISS → fork encoder child (parent pid", _getpid(), ")")
    var pid = sys_fork()
    if pid == 0:
        # CHILD: async-signal-safe only, then execv into a fresh image.
        _ = sys_execv(path, argv)
        sys__exit(127)                     # execv failed
    if pid < 0:
        print("[zimage] fork failed (", errno_str(), ") → in-process encode")
        return encode_captions_fixed(prompt, negative, ctx)

    # PARENT: bounded WNOHANG reap (hang backstop). Blocking-reap once it exits so
    # the OS has released the child's VRAM before we load the caps onto the GPU.
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
        print("[zimage] encoder child timed out/errored → in-process encode")
        return encode_captions_fixed(prompt, negative, ctx)

    var exited_ok = (status & 0x7F) == 0 and ((status >> 8) & 0xFF) == 0
    if not exited_ok:
        print("[zimage] encoder child abnormal exit (status", status,
              ") → in-process encode")
        return encode_captions_fixed(prompt, negative, ctx)

    # Success path: read the caps the child wrote. Any read failure (e.g. a host
    # that exits 0 but never wrote the sidecar) → in-process fallback.
    try:
        var meta = _read_meta(meta_path)
        var cond = load_tensor_bin(cond_path, ctx)
        var uncond = load_tensor_bin(uncond_path, ctx)
        print("[zimage] encoder child reaped → caps loaded (encoder VRAM reclaimed)")
        return CapFeatsFixed(cond^, uncond^, meta[0], meta[1])
    except e:
        print("[zimage] caps read-back failed (", e, ") → in-process encode")
        return encode_captions_fixed(prompt, negative, ctx)
