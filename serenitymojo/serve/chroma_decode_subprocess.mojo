# serenitymojo.serve.chroma_decode_subprocess — the WHOLE-IMAGE Chroma (FLUX VAE)
# decode in a fork+execv CHILD process, so it runs in a CLEAN CUDA context.
#
# WHY (MEASURED, job-0078): with the encode already subprocessed and the DiT
# released + trimmed, the chroma parent sits at ~3 GB when the decode starts —
# but the whole-image FLUX VAE decode of the [1,16,128,128] latent allocates
# ~11.7 GB of activations on top, peaking at 14686 MiB and OOMing within ~1 GB
# of fitting the 16 GB card. A CHILD process starts from a fresh CUDA context
# (~1 GB), so the same ~11.7 GB decode peaks ≈ 12.7 GB device-global and fits.
# Tiled decode is NOT the answer (measured to degrade output — MJ-1054); the
# whole-image path is the pipeline's proven decode.
#
# The child keeps the EXACT chroma_pipeline Stage-8 math at each admitted
# compile-time grid: unpack packed BF16 → F32 cast (the Flux VAE decoder path
# is F32-only) → matching load_flux1_ldm_decoder[LH,LW] (ae.safetensors) →
# whole rectangular decode → [1,3,8*LH,8*LW] SIGNED rgb.
# The parent hands the packed latent over as bit-identical raw bytes
# (io.cap_cache) and gets the rgb tensor back the same way; it then runs the
# identical PNG-save (genparams tEXt) + manifest path as before.
#
# SELF-EXEC (one binary): the decode child IS serenity_worker_chroma re-exec'd
# with argv ["decode-child", <latent_path>, <rgb_out_path>, <lh>, <lw>];
# serenity_worker_chroma.main routes that to decode_child_run(). The VAE path
# stays fixed; geometry is finite-dispatched over the seven product shapes.
#
# SAFETY / FALLBACK: decode_whole_subprocess RAISES on preflight failure, a host
# that does not route `decode-child`, or any subprocess failure. The caller
# (chroma_backend._decode_and_save) catches and falls back to the bounded
# in-process 3x3 tiled decode, printing LOUDLY which path ran.
#
# Mirrors serve/klein_decode_subprocess.mojo / zimage_decode_subprocess.mojo and
# the Phase-5 process-isolation contract (proc_ipc.mojo): argv built BEFORE
# fork, async-signal-safe calls only between fork and execv, execv into a fresh
# image.

from std.memory import alloc, UnsafePointer
from std.builtin.type_aliases import MutExternalOrigin
from std.ffi import external_call
from std.gpu.host import DeviceContext
from std.time import sleep

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.cap_cache import save_tensor_bin, load_tensor_bin
from serenitymojo.models.vae.ldm_decoder import load_flux1_ldm_decoder
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.pipeline.flux_tiled_decode import flux_tiled_decode
from serenitymojo.pipeline.chroma_pipeline_1024_multistep import (
    _unpack_latent_shape, VAE_PATH,
)
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


# Whole decode = VAE load (~330 MB) + a few seconds of conv GPU work → 120 s is
# a pure hang backstop, after which we SIGKILL + fall back in-process.
comptime _DECODE_CHILD_TIMEOUT_S = 120.0
comptime _DECODE_POLL_S = 0.05
# Pre-flight guard for the WHOLE decoder. Host-verified on the 24 GB product GPU:
# the child still OOMs with 20.7 GiB device-global free after VMM denoiser
# eviction. Require 22 GiB so 24 GB cards skip the doomed ~2.6 s attempt and go
# directly to the bounded tiled child; larger cards retain the quality-first
# whole path. This is measured admission, not a model or UI quality gate.
comptime _DECODE_CHILD_MIN_FREE_BYTES = Int(22528) * 1024 * 1024  # 22 GiB
comptime _TILED_CHILD_MIN_FREE_BYTES = Int(9216) * 1024 * 1024
comptime _CHROMA_PRODUCT_EDGE_UNITS = 16


def _getpid() -> Int:
    return Int(external_call["getpid", Int32]())


def _unlink_file(path: String):
    """Best-effort delete of a /tmp decode sidecar (mirrors serenity_daemon._unlink_file)."""
    _ = external_call["unlink", Int32](cstr(path))


def _decode_child_shape[LH_: Int, LW_: Int](
    packed: Tensor, rgb_out_path: String, ctx: DeviceContext
) raises:
    var latent = _unpack_latent_shape[LH_, LW_](packed, ctx)
    var latent_f32 = cast_tensor(latent, STDtype.F32, ctx)
    var dec = load_flux1_ldm_decoder[LH_, LW_](String(VAE_PATH), ctx)
    var rgb = dec.decode(latent_f32, ctx)
    save_tensor_bin(rgb, rgb_out_path, ctx)


def _decode_tiled_child_shape[LH_: Int, LW_: Int](
    packed: Tensor, rgb_out_path: String, ctx: DeviceContext
) raises:
    var latent = _unpack_latent_shape[LH_, LW_](packed, ctx)
    var latent_f32 = cast_tensor(latent, STDtype.F32, ctx)
    var rgb = flux_tiled_decode[LH_, LW_](
        latent_f32, String(VAE_PATH), ctx
    )
    save_tensor_bin(rgb, rgb_out_path, ctx)


def decode_child_run(
    latent_path: String, rgb_out_path: String, lh: Int, lw: Int
) raises:
    """CHILD body (after execv into `serenity_worker_chroma decode-child`).
    Fresh process image → fresh CUDA context. Load the packed BF16 latent, run
    the pipeline's Stage-8 math (unpack → F32 cast → whole FLUX VAE decode),
    write the matching rectangular rgb tensor to `rgb_out_path` (raw bytes
    via io.cap_cache), then RETURN so the process exits and the OS reclaims ALL
    decode VRAM."""
    var ctx = DeviceContext()
    var packed = load_tensor_bin(latent_path, ctx)
    comptime for bi in range(DEFAULT_ASPECT_LADDER_LEN):
        comptime X100_BI = DEFAULT_ASPECT_LADDER_X100[bi]
        comptime LH_BI = aspect_lat_h_units(X100_BI, _CHROMA_PRODUCT_EDGE_UNITS)
        comptime LW_BI = aspect_lat_w_units(X100_BI, _CHROMA_PRODUCT_EDGE_UNITS)
        if lh == LH_BI and lw == LW_BI:
            _decode_child_shape[LH_BI, LW_BI](packed, rgb_out_path, ctx)
            print("[chroma-decode-child] wrote rgb", rgb_out_path,
                  "grid=", lh, "x", lw)
            return
    raise Error(
        String("chroma decode-child: unsupported latent grid ")
        + String(lh) + String("x") + String(lw)
    )


def decode_tiled_child_run(
    latent_path: String, rgb_out_path: String, lh: Int, lw: Int
) raises:
    """Fresh-process bounded fallback used when whole decode cannot coexist
    with the resident parent. Process exit reclaims every tiled-VAE allocation
    while the parent's denoiser residency remains intact."""
    var ctx = DeviceContext()
    var packed = load_tensor_bin(latent_path, ctx)
    comptime for bi in range(DEFAULT_ASPECT_LADDER_LEN):
        comptime X100_BI = DEFAULT_ASPECT_LADDER_X100[bi]
        comptime LH_BI = aspect_lat_h_units(X100_BI, _CHROMA_PRODUCT_EDGE_UNITS)
        comptime LW_BI = aspect_lat_w_units(X100_BI, _CHROMA_PRODUCT_EDGE_UNITS)
        if lh == LH_BI and lw == LW_BI:
            _decode_tiled_child_shape[LH_BI, LW_BI](
                packed, rgb_out_path, ctx
            )
            print("[chroma-decode-tiled-child] wrote rgb", rgb_out_path,
                  "grid=", lh, "x", lw)
            return
    raise Error(
        String("chroma decode-tiled-child: unsupported latent grid ")
        + String(lh) + String("x") + String(lw)
    )


def decode_whole_subprocess(
    packed_latent: Tensor, lh: Int, lw: Int, ctx: DeviceContext
) raises -> Tensor:
    """PARENT body (chroma worker DECODE phase, after the DiT release + trim).
    Serialize the packed latent, fork+execv a fresh `serenity_worker_chroma
    decode-child` to run the WHOLE-image VAE decode in ITS OWN clean pool,
    blocking-reap it (VRAM released by process death), read back the
    rectangular rgb, and delete the /tmp sidecars. RAISES on preflight
    failure, a host that does not route `decode-child`, or any subprocess
    failure — the caller falls back to the bounded in-process 3x3 tiled decode
    (loudly)."""
    var prefix = String("/tmp/serenity_chroma_decode_") + String(_getpid())
    var latent_path = prefix + String(".latent.bin")
    var rgb_path = prefix + String(".rgb.bin")

    # The caller already released the DiT and trimmed, but trim again cheaply so
    # the preflight sees the true post-release device-global free (klein does
    # the same before its fork).
    try:
        cu_mempool_trim_current(0)
    except e:
        print("[chroma] pre-decode pool trim failed (continuing):", e)

    var free_bytes = cu_mem_get_info().free_bytes
    if free_bytes < _DECODE_CHILD_MIN_FREE_BYTES:
        raise Error(
            String("chroma decode child preflight failed: free VRAM ")
            + String(free_bytes // (1024 * 1024))
            + String(" MiB < required ")
            + String(_DECODE_CHILD_MIN_FREE_BYTES // (1024 * 1024))
            + String(" MiB")
        )

    # Write the latent BEFORE the fork so the child only reads it.
    save_tensor_bin(packed_latent, latent_path, ctx)

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

    print("[chroma] whole-image decode → fork VAE child (parent pid", _getpid(), ")")
    var pid = sys_fork()
    if pid == 0:
        # CHILD: async-signal-safe only, then execv into a fresh image.
        _ = sys_execv(path, argv)
        sys__exit(127)                     # execv failed
    if pid < 0:
        raise Error(String("chroma decode child fork failed: ") + errno_str())

    # PARENT: bounded WNOHANG reap (hang backstop). Blocking-reap once it exits
    # so the OS has released the child's VRAM before we touch the GPU again.
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
        _unlink_file(latent_path)
        raise Error("chroma decode child timed out or waitpid failed")

    var exited_ok = (status & 0x7F) == 0 and ((status >> 8) & 0xFF) == 0
    if not exited_ok:
        _unlink_file(latent_path)
        raise Error(
            String("chroma decode child abnormal exit status ") + String(status)
        )

    # Success path: read the rgb the child wrote, then delete the sidecars.
    # A read failure raises → caller falls back in-process.
    var rgb = load_tensor_bin(rgb_path, ctx)
    _unlink_file(latent_path)
    _unlink_file(rgb_path)
    print("[chroma] decode child reaped → whole-image rgb loaded (decode VRAM reclaimed)")
    return rgb^


def decode_tiled_subprocess(
    packed_latent: Tensor, lh: Int, lw: Int, ctx: DeviceContext
) raises -> Tensor:
    """Run the existing gated 3x3 fallback in a fresh child process.

    This path is selected only after the preferred whole child cannot fit next
    to the resident parent. Unlike the old in-process fallback, child exit
    cannot strand VAE allocations in the parent's CUDA pool."""
    var prefix = String("/tmp/serenity_chroma_decode_tiled_") + String(_getpid())
    var latent_path = prefix + String(".latent.bin")
    var rgb_path = prefix + String(".rgb.bin")

    try:
        cu_mempool_trim_current(0)
    except e:
        print("[chroma] pre-tiled-decode pool trim failed (continuing):", e)
    var free_bytes = cu_mem_get_info().free_bytes
    if free_bytes < _TILED_CHILD_MIN_FREE_BYTES:
        raise Error(
            String("chroma tiled decode child preflight failed: free VRAM ")
            + String(free_bytes // (1024 * 1024))
            + String(" MiB < required ")
            + String(_TILED_CHILD_MIN_FREE_BYTES // (1024 * 1024))
            + String(" MiB")
        )

    save_tensor_bin(packed_latent, latent_path, ctx)
    var args = List[String]()
    args.append(SELF_EXE)
    args.append(String("decode-tiled-child"))
    args.append(latent_path)
    args.append(rgb_path)
    args.append(String(lh))
    args.append(String(lw))
    var argv = build_argv(args)
    var path = cstr(SELF_EXE)

    print("[chroma] tiled decode → fork VAE child; resident denoiser retained (parent pid",
          _getpid(), ")")
    var pid = sys_fork()
    if pid == 0:
        _ = sys_execv(path, argv)
        sys__exit(127)
    if pid < 0:
        _unlink_file(latent_path)
        raise Error(String("chroma tiled decode child fork failed: ") + errno_str())

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
        _unlink_file(latent_path)
        _unlink_file(rgb_path)
        raise Error("chroma tiled decode child timed out or waitpid failed")
    var exited_ok = (status & 0x7F) == 0 and ((status >> 8) & 0xFF) == 0
    if not exited_ok:
        _unlink_file(latent_path)
        _unlink_file(rgb_path)
        raise Error(
            String("chroma tiled decode child abnormal exit status ")
            + String(status)
        )

    var rgb = load_tensor_bin(rgb_path, ctx)
    _unlink_file(latent_path)
    _unlink_file(rgb_path)
    print("[chroma] tiled decode child reaped → rgb loaded; denoiser still resident")
    return rgb^
