# serenitymojo.serve.zimage_decode_subprocess — whole-image Z-Image VAE decode
# for the finite 1MP product ladder in a fork+execv CHILD process, so it runs in
# a CLEAN CUDA context instead of tiling.
#
# WHY (MEASURED): tiled VAE decode measurably degrades output (MJ-1054: for person
# content 37–85% of pixels differ vs a whole-image decode). The monolithic
# ZImageDecoder[128,128].decode of a [1,16,128,128] latent is VRAM-tight IN the
# resident zimage worker: that process holds the ~13 GB DiT (kept resident BY
# DESIGN — freeing it would force a per-job reload) plus a grown pool that
# cuMemPoolTrimTo can NOT return (fragmented around the live DiT — MEASURED
# reclaimed 0), so a whole 1024² decode on top of it risks OOM and the worker falls
# back to the 3×3 tiled crop. Running the whole decode in a SEPARATE process gives
# it its own CUDA context and pool: the child sees the device-global free memory,
# decodes the whole latent, writes the RGB tensor to disk, and EXITS (OS reclaims
# every byte). The resident DiT in the parent is untouched — fork copies the fd
# table and the child execv's immediately, so this CUDA context is never used in
# the child.
#
# WHY an RGB TENSOR (not the final PNG): the parent's _decode_and_save still owns
# two post-decode steps that MUST run on the rgb — the LanPaint mask blend (for
# 1024 inpaint jobs) and the genparams tEXt PNG write. So the child does ONLY the
# GPU-heavy VAE decode and hands back the [1,3,1024,1024] rgb tensor (bit-identical
# raw bytes via io.cap_cache); the parent applies the blend and writes the PNG
# exactly as it does for the tiled path. Round-trip is ~6 MB — negligible.
#
# SELF-EXEC (one binary): the decode child IS serenity_worker_zimage re-exec'd with
# argv ["decode-child", <latent_path>, <rgb_out_path>, <latent_h>, <latent_w>];
# serenity_worker_zimage.main routes that to decode_child_run(). No new binary.
#
# SAFETY / FALLBACK: decode_whole_subprocess RAISES on a pre-flight guard failure,
# a host binary that does not route `decode-child`, or any subprocess failure
# (fork error, timeout, abnormal exit, unreadable rgb). The caller
# (zimage_backend._decode_and_save) catches and falls back to the resident-VAE
# tiled decode — correctness is never sacrificed for the whole-decode quality win.
#
# Mirrors serve/zimage_encode_subprocess.mojo (same Phase-5 process-isolation
# contract: argv built BEFORE fork, async-signal-safe calls only between fork and
# execv, execv into a fresh image).

from std.memory import alloc, UnsafePointer
from std.builtin.type_aliases import MutExternalOrigin
from std.ffi import external_call
from std.gpu.host import DeviceContext
from std.time import sleep

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.cap_cache import save_tensor_bin, load_tensor_bin
from serenitymojo.models.vae.zimage_decoder import ZImageDecoder
from serenitymojo.pipeline.zimage_generate import VAE_DIR
from serenitymojo.training.aspect_buckets import (
    DEFAULT_ASPECT_LADDER_LEN, DEFAULT_ASPECT_LADDER_X100,
    aspect_lat_h_units, aspect_lat_w_units,
)
from serenitymojo.serve.proc_ipc import (
    build_argv, cstr, sys_execv, sys__exit, sys_waitpid, proc_kill_wait,
    SELF_EXE, SIGKILL, WNOHANG,
)
from serenitymojo.offload.vmm_cuda import cu_mem_get_info, cu_mempool_trim_current
from net.syscalls import sys_fork, errno_str


# Whole 1024² decode = weight load (~0.5 s) + a few seconds of conv GPU work → a
# 120 s cap is a pure hang backstop, far above any real decode, after which we
# SIGKILL + fall back to tiled.
comptime _DECODE_CHILD_TIMEOUT_S = 120.0
comptime _DECODE_POLL_S = 0.05
# Pre-flight guard: the decode child is a SEPARATE process whose peak (the
# ZImageDecoder[128,128] weights + the 1024² conv activations/cuDNN workspaces +
# its own CUDA context) must fit in the GPU's CURRENT free memory alongside the
# resident parent DiT. The parent does NOT trim here — its grown pool is fragmented
# around the live DiT (cuMemPoolTrim reclaims 0), so we simply require enough
# device-global free memory. ESTIMATE ≈ 6 GiB for the whole 1024² decode child
# (activations dominated by the 256/128-ch @ 512²/1024² up-block feature maps plus
# conv workspaces; NOT independently measured — the orchestrator validates). On the
# 24 GB card with the ~13 GB DiT resident free is typically ~11 GB, so the fork
# proceeds; if this underestimates, the child OOMs → nonzero exit → tiled fallback.
comptime _DECODE_CHILD_MIN_FREE_BYTES = Int(6144) * 1024 * 1024  # ~6 GiB
comptime _ZIMAGE_PRODUCT_EDGE_UNITS = 16


def _getpid() -> Int:
    return Int(external_call["getpid", Int32]())


def _decode_child_shape[LH_: Int, LW_: Int](
    latent: Tensor, rgb_out_path: String, ctx: DeviceContext
) raises:
    var dec = ZImageDecoder[LH_, LW_].load(VAE_DIR, ctx)
    var rgb = dec.decode(latent, ctx)
    save_tensor_bin(rgb, rgb_out_path, ctx)


def decode_child_run(latent_path: String, rgb_out_path: String, lh: Int, lw: Int) raises:
    """CHILD body (after execv into `serenity_worker_zimage decode-child`). Fresh
    process image → fresh CUDA context. Load the BF16 [1,16,lh,lw] latent, load
    the resident-VAE-dir ZImageDecoder at that grid, decode the WHOLE latent,
    write the [1,3,8*lh,8*lw] rgb tensor to `rgb_out_path` (raw bytes via
    io.cap_cache), then RETURN so the process exits and the OS reclaims ALL decode
    VRAM. The rgb file's presence is the parent's "child succeeded" signal.

    Only the finite seven-shape 1MP product ladder is wired. Other grids raise
    so the parent can fail loudly into the bounded tiled fallback."""
    var ctx = DeviceContext()
    var latent = load_tensor_bin(latent_path, ctx)
    comptime for bi in range(DEFAULT_ASPECT_LADDER_LEN):
        comptime X100_BI = DEFAULT_ASPECT_LADDER_X100[bi]
        comptime LH_BI = aspect_lat_h_units(X100_BI, _ZIMAGE_PRODUCT_EDGE_UNITS)
        comptime LW_BI = aspect_lat_w_units(X100_BI, _ZIMAGE_PRODUCT_EDGE_UNITS)
        if lh == LH_BI and lw == LW_BI:
            _decode_child_shape[LH_BI, LW_BI](latent, rgb_out_path, ctx)
            print("[zimage-decode-child] wrote rgb", rgb_out_path,
                  "grid=", lh, "x", lw)
            return
    raise Error(
        String("zimage decode-child: unsupported latent grid ")
        + String(lh) + String("x") + String(lw)
    )


def decode_whole_subprocess(
    latent: Tensor, lh: Int, lw: Int, ctx: DeviceContext
) raises -> Tensor:
    """PARENT body (zimage worker, finite 1MP decode). Serialize the BF16 latent, fork+
    execv a fresh `serenity_worker_zimage decode-child` to run the WHOLE-image VAE
    decode in ITS OWN clean CUDA context, blocking-reap it (VRAM released), then read
    back the rectangular rgb it wrote. RAISES on guard failure, a host that does
    not route `decode-child`, or any subprocess failure — the caller falls back to
    the resident-VAE tiled decode."""
    var prefix = String("/tmp/serenity_zimage_decode_") + String(_getpid())
    var latent_path = prefix + String(".latent.bin")
    var rgb_path = prefix + String(".rgb.bin")

    # Best-effort trim BEFORE the guard: _decode_and_save frees the per-job denoise
    # working set (rope/cap_seq/lora/latent) just before calling us, so trimming the
    # default pool returns those freed buffers to the OS and lifts device-global free
    # for the child. The resident DiT is untouched — its ArcPointers keep those
    # buffers LIVE, so trim cannot return them (this is why the worker stays resident
    # across jobs). Unlike the encoder-fragmentation case (reclaims ~0 around the
    # live DiT), the freed denoise buffers are typically reclaimable here.
    try:
        cu_mempool_trim_current(0)
    except e:
        print("[zimage] pre-decode pool trim failed (continuing):", e)

    # Pre-flight guard: device-global free memory is exactly what the child's
    # separate CUDA context will see. Below threshold → raise (caller tiles), no
    # doomed fork, no transient spike.
    var free_bytes = cu_mem_get_info().free_bytes
    if free_bytes < _DECODE_CHILD_MIN_FREE_BYTES:
        raise Error(
            String("zimage decode child preflight failed: free VRAM ")
            + String(free_bytes // (1024 * 1024))
            + String(" MiB < required ")
            + String(_DECODE_CHILD_MIN_FREE_BYTES // (1024 * 1024))
            + String(" MiB")
        )

    # Write the latent BEFORE the fork so the child only reads it (no shared GPU
    # state crosses fork — the child gets a fresh context on execv).
    save_tensor_bin(latent, latent_path, ctx)

    # argv + execv path built BEFORE fork (no allocation between fork and execv).
    var args = List[String]()
    args.append(SELF_EXE)                  # argv[0]
    args.append(String("decode-child"))
    args.append(latent_path)
    args.append(rgb_path)
    args.append(String(lh))
    args.append(String(lw))
    var argv = build_argv(args)
    var path = cstr(SELF_EXE)

    print("[zimage] whole-image decode → fork VAE child (parent pid", _getpid(), ")")
    var pid = sys_fork()
    if pid == 0:
        # CHILD: async-signal-safe only, then execv into a fresh image.
        _ = sys_execv(path, argv)
        sys__exit(127)                     # execv failed
    if pid < 0:
        raise Error(String("zimage decode child fork failed: ") + errno_str())

    # PARENT: bounded WNOHANG reap (hang backstop). Blocking-reap once it exits so
    # the OS has released the child's VRAM before we touch the GPU again.
    var st = alloc[Int32](1)
    var stp = rebind[UnsafePointer[Int32, MutExternalOrigin]](st)
    var waited = 0.0
    var reaped = Int32(0)
    while waited < _DECODE_CHILD_TIMEOUT_S:
        reaped = sys_waitpid(pid, stp, WNOHANG)
        if reaped == pid:
            break
        if reaped < 0:
            break
        sleep(_DECODE_POLL_S)
        waited += _DECODE_POLL_S
    var status = Int(st[0])
    st.free()

    if reaped != pid:
        proc_kill_wait(pid, SIGKILL)
        raise Error("zimage decode child timed out or waitpid failed")

    var exited_ok = (status & 0x7F) == 0 and ((status >> 8) & 0xFF) == 0
    if not exited_ok:
        raise Error(
            String("zimage decode child abnormal exit status ") + String(status)
        )

    # Success path: read the rgb the child wrote. A read failure (e.g. a host that
    # exits 0 but never wrote the rgb) raises → caller tiles.
    var rgb = load_tensor_bin(rgb_path, ctx)
    print("[zimage] decode child reaped → whole-image rgb loaded (decode VRAM reclaimed)")
    return rgb^
