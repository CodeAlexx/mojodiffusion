# serenitymojo.serve.qwenimage_encode_subprocess — the Qwen2.5-VL text encoder in
# a fork+execv CHILD process, so its VRAM is reclaimed by process death.
#
# WHY (MEASURED): the Qwen-Image worker is resident at ~22 GB (vs flux 12.7 GB)
# because the ~16 GB Qwen2.5-VL encoder VRAM is NOT reclaimed before the DiT
# streams — cu_mempool_trim reclaims ~0 (between_jobs_trim measured "reclaimed
# 0 MiB"): MAX holds the pool, the freed encoder blocks are fragmented in the
# same segments as the resident DiT offloader buffers, and the GPU is left with
# only ~2 GB headroom so the (already-present) block prefetch can't overlap. The
# ONLY reliable reclaim is process exit. So on a conditioning-cache MISS the
# worker fork+execv's a FRESH child that loads the encoder, encodes
# (prompt, negative), writes the BF16 caps to disk (bit-identical raw bytes via
# io.cap_cache), and EXITS — the OS frees every byte of the encoder VRAM. The
# parent waitpid's (blocking-reap → VRAM released before we touch the GPU again),
# reads the caps back (a one-time ~7 MB H2D for the two [1,512,3584] BF16 caps,
# NOT per-step weight paging), and denoises at the resident DiT baseline.
#
# This is the SAME split zimage already uses (serve/zimage_encode_subprocess.mojo)
# and the SAME split the Klein-9B pipeline uses (io/cap_cache.mojo header: "process
# death frees every byte of encoder GPU memory before the DiT process ever
# starts"), under the Phase-5 process-isolation contract (proc_ipc.mojo: argv
# built BEFORE fork, async-signal-safe calls only between fork and execv, execv
# into a fresh image — fork()ing a CUDA/AsyncRT process is only safe if the child
# does nothing but execv).
#
# SELF-EXEC (one binary, one build): the encode child IS serenity_worker_qwenimage
# re-exec'd with argv ["encode-child", <prefix>, <prompt>, <negative>];
# serenity_worker_qwenimage.main routes that to encode_child_run(). No separate
# encoder binary.
#
# BYTE-IDENTITY: the produced conditioning is byte-identical to the in-process
# encode_captions_from_strings path. The caps tensors (pos/neg) are serialized as
# raw device bytes by save_tensor_bin (no dtype cast) and reloaded by
# load_tensor_bin, so the BF16 buffers the DiT consumes are bit-for-bit the same
# whether the encode ran in-process or in the child. real_pos/real_neg ride the
# 24-byte meta sidecar.
#
# SAFETY / STRICTNESS: the server path raises on fork, timeout, abnormal exit, or
# unreadable caps instead of falling back to in-process encode. Loading the
# encoder in the parent defeats the VRAM/offload strategy, so production must
# fail loud rather than silently fragment the worker.

from std.memory import alloc, UnsafePointer
from std.builtin.type_aliases import MutExternalOrigin
from std.ffi import external_call
from std.gpu.host import DeviceContext
from std.time import sleep

from serenitymojo.pipeline.qwenimage_sample_cli import (
    encode_captions_from_strings, QwenCaps,
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


# encoder load (~16 GB across 4 shards) + cond+uncond forward is several seconds →
# ~300 s is a pure hang backstop, far above any real encode, after which we
# SIGKILL + fall back.
comptime _ENCODE_CHILD_TIMEOUT_S = 300.0
comptime _ENCODE_POLL_S = 0.05
# Pre-flight guard: the encoder child is a separate process whose peak (~16 GB
# Qwen2.5-VL weights + forward activations + its own CUDA context) must fit in the
# GPU's CURRENT free memory alongside this resident parent. The parent's grown
# pool can't be trimmed back to make room (cu_mempool_trim reclaims 0). On the
# clean-parent first job the offloader handle is not yet loaded, so free VRAM is
# high and the fork proceeds; once the DiT is resident with grown buffers the
# guard sends the encode in-process (correct, just no win). MEASURED encoder
# child footprint ≈ 17 GB; require this much free before forking.
comptime _ENCODE_CHILD_MIN_FREE_BYTES = Int(17400) * 1024 * 1024  # ~17 GiB
# 24-byte binary sidecar: [magic][real_pos][real_neg] as 3x Int64 LE.
comptime _META_MAGIC = Int64(0x51494D4341505631)  # "QIMCAPV1"


def _getpid() -> Int:
    return Int(external_call["getpid", Int32]())


def _write_meta(path: String, real_pos: Int, real_neg: Int) raises:
    var fd = sys_open(path, O_WRONLY | O_CREAT | O_TRUNC, 0o644)
    if fd < 0:
        raise Error(String("qwenimage_encode_subprocess: meta open failed: ") + path)
    var tmp = alloc[Int64](3)
    tmp[0] = _META_MAGIC
    tmp[1] = Int64(real_pos)
    tmp[2] = Int64(real_neg)
    var p = BytePtr(unsafe_from_address=Int(tmp))
    var w = sys_pwrite(fd, p, 24, 0)
    tmp.free()
    _ = sys_close(fd)
    if w != 24:
        raise Error("qwenimage_encode_subprocess: short meta write")


def _read_meta(path: String) raises -> List[Int]:
    var fd = sys_open(path, O_RDONLY, 0)
    if fd < 0:
        raise Error(String("qwenimage_encode_subprocess: meta open failed: ") + path)
    var tmp = alloc[Int64](3)
    var p = BytePtr(unsafe_from_address=Int(tmp))
    var r = sys_pread(fd, p, 24, 0)
    var magic = tmp[0]
    var rp = Int(tmp[1])
    var rn = Int(tmp[2])
    tmp.free()
    _ = sys_close(fd)
    if r != 24:
        raise Error("qwenimage_encode_subprocess: short meta read")
    if magic != _META_MAGIC:
        raise Error("qwenimage_encode_subprocess: bad meta magic")
    var out = List[Int]()
    out.append(rp)
    out.append(rn)
    return out^


def encode_child_run(prefix: String, prompt: String, negative: String) raises:
    """CHILD body (after execv into `serenity_worker_qwenimage encode-child`).
    Fresh process image → fresh CUDA context. Load the ~16 GB Qwen2.5-VL encoder,
    encode pos+neg, write the BF16 caps + real token counts to
    `<prefix>.{pos,neg}.bin` / `<prefix>.meta`, then RETURN so the process exits
    and the OS reclaims ALL encoder VRAM. The `.meta` file is written LAST: its
    presence + magic is the parent's "child fully succeeded" signal."""
    var ctx = DeviceContext()
    var caps = encode_captions_from_strings(prompt, negative, ctx)
    save_tensor_bin(caps.pos, prefix + String(".pos.bin"), ctx)
    save_tensor_bin(caps.neg, prefix + String(".neg.bin"), ctx)
    _write_meta(prefix + String(".meta"), caps.real_pos, caps.real_neg)
    print("[qwenimage-encode-child] wrote caps", prefix,
          "real_pos=", caps.real_pos, "real_neg=", caps.real_neg)


def encode_captions_subprocess(
    prompt: String, negative: String, ctx: DeviceContext
) raises -> QwenCaps:
    """PARENT body (qwenimage worker, conditioning-cache MISS). fork+execv a fresh
    `serenity_worker_qwenimage encode-child` to run the ~16 GB Qwen2.5-VL encoder
    in ITS OWN process, blocking-reap it (VRAM released), then read back the BF16
    caps it wrote. The resident DiT offloader in THIS process is untouched: fork
    copies the fd table and the child execv's immediately, so this CUDA context is
    never used in the child. Server inference is strict: child failure raises
    instead of falling back to in-process encode, because parent-side encoder
    load fragments the Qwen worker's CUDA pool around the resident offloader."""
    var prefix = String("/tmp/serenity_qwenimage_caps_") + String(_getpid())
    var pos_path = prefix + String(".pos.bin")
    var neg_path = prefix + String(".neg.bin")
    var meta_path = prefix + String(".meta")

    # Pre-flight guard: skip a doomed fork (and its transient VRAM spike) when the
    # GPU's current free memory can't hold the ~16 GB encoder child. cu_mem_get_info
    # reports device-global free, which is exactly what the child's separate CUDA
    # context will see. Below threshold raises so production does not silently
    # fragment the parent with an in-process encoder load.
    var free_bytes = cu_mem_get_info().free_bytes
    if free_bytes < _ENCODE_CHILD_MIN_FREE_BYTES:
        raise Error(
            String("qwenimage encoder child preflight failed: free VRAM ")
            + String(free_bytes // (1024 * 1024))
            + String(" MiB < required ")
            + String(_ENCODE_CHILD_MIN_FREE_BYTES // (1024 * 1024))
            + String(" MiB")
        )

    # argv + execv path built BEFORE fork (no allocation between fork and execv).
    var args = List[String]()
    args.append(SELF_EXE)                  # argv[0]
    args.append(String("encode-child"))
    args.append(prefix)
    args.append(prompt)
    args.append(negative)
    var argv = build_argv(args)
    var path = cstr(SELF_EXE)

    print("[qwenimage] cache MISS → fork encoder child (parent pid", _getpid(), ")")
    var pid = sys_fork()
    if pid == 0:
        # CHILD: async-signal-safe only, then execv into a fresh image.
        _ = sys_execv(path, argv)
        sys__exit(127)                     # execv failed
    if pid < 0:
        raise Error(String("qwenimage encoder child fork failed: ") + errno_str())

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
        raise Error("qwenimage encoder child timed out or waitpid failed")

    var exited_ok = (status & 0x7F) == 0 and ((status >> 8) & 0xFF) == 0
    if not exited_ok:
        raise Error(
            String("qwenimage encoder child abnormal exit status ")
            + String(status)
        )

    # Success path: read the caps the child wrote. Any read failure (e.g. a host
    # that exits 0 but never wrote the sidecar) fails loud.
    try:
        var meta = _read_meta(meta_path)
        var pos = load_tensor_bin(pos_path, ctx)
        var neg = load_tensor_bin(neg_path, ctx)
        print("[qwenimage] encoder child reaped → caps loaded (encoder VRAM reclaimed)")
        return QwenCaps(pos^, neg^, meta[0], meta[1])
    except e:
        raise Error(String("qwenimage encoder caps read-back failed: ") + String(e))
