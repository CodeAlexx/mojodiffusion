# serenitymojo.serve.sdxl_decode_subprocess — SDXL's existing parity-preserving
# 3x3 tiled VAE decode in a fork+execv child process.
#
# The parent keeps its ~6.6 GiB UNet resident across jobs. The child receives the
# exact F32 latent through io.cap_cache, executes sdxl_tiled_decode at one of the
# seven compiled product grids, writes the RGB tensor, and exits. Process death
# reclaims all VAE allocations without destroying/reloading the parent's UNet.

from std.memory import alloc, UnsafePointer
from std.ffi import external_call
from max.gpu.host import DeviceContext
from std.time import sleep

from serenitymojo.tensor import Tensor
from serenitymojo.io.cap_cache import save_tensor_bin, load_tensor_bin
from serenitymojo.models.vae.sdxl_tiled_decode import sdxl_tiled_decode
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


comptime _DECODE_CHILD_TIMEOUT_S = 120.0
comptime _DECODE_POLL_S = 0.05
# Measured in-process tiled decode adds about 6.6 GiB above the live worker
# floor. Require 9 GiB device-global free for the fresh child context, decoder,
# activations, and workspace margin; failure falls back to the old release+decode.
comptime _DECODE_CHILD_MIN_FREE_BYTES = Int(9216) * 1024 * 1024
comptime _SDXL_PRODUCT_EDGE_UNITS = 16


def _getpid() -> Int:
    return Int(external_call["getpid", Int32]())


def _unlink_file(path: String):
    _ = external_call["unlink", Int32](cstr(path))


def _decode_child_shape[LH_: Int, LW_: Int](
    latent: Tensor, rgb_path: String, vae_path: String, ctx: DeviceContext
) raises:
    var rgb = sdxl_tiled_decode[LH_, LW_](latent, vae_path, ctx)
    save_tensor_bin(rgb, rgb_path, ctx)


def decode_child_run(
    latent_path: String,
    rgb_path: String,
    vae_path: String,
    lh: Int,
    lw: Int,
) raises:
    var ctx = DeviceContext()
    var latent = load_tensor_bin(latent_path, ctx)
    comptime for bi in range(DEFAULT_ASPECT_LADDER_LEN):
        comptime X100_BI = DEFAULT_ASPECT_LADDER_X100[bi]
        comptime LH_BI = aspect_lat_h_units(X100_BI, _SDXL_PRODUCT_EDGE_UNITS)
        comptime LW_BI = aspect_lat_w_units(X100_BI, _SDXL_PRODUCT_EDGE_UNITS)
        if lh == LH_BI and lw == LW_BI:
            _decode_child_shape[LH_BI, LW_BI](latent, rgb_path, vae_path, ctx)
            print("[sdxl-decode-child] wrote tiled rgb", rgb_path,
                  "grid=", lh, "x", lw)
            return
    raise Error(
        String("sdxl decode-child: unsupported latent grid ")
        + String(lh) + String("x") + String(lw)
    )


def decode_tiled_subprocess(
    latent: Tensor,
    vae_path: String,
    lh: Int,
    lw: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    var prefix = String("/tmp/serenity_sdxl_decode_") + String(_getpid())
    var latent_path = prefix + String(".latent.bin")
    var rgb_path = prefix + String(".rgb.bin")

    try:
        cu_mempool_trim_current(0)
    except e:
        print("[sdxl] pre-decode pool trim failed (continuing):", e)
    var free_bytes = cu_mem_get_info().free_bytes
    if free_bytes < _DECODE_CHILD_MIN_FREE_BYTES:
        raise Error(
            String("sdxl decode child preflight failed: free VRAM ")
            + String(free_bytes // (1024 * 1024))
            + String(" MiB < required ")
            + String(_DECODE_CHILD_MIN_FREE_BYTES // (1024 * 1024))
            + String(" MiB")
        )

    save_tensor_bin(latent, latent_path, ctx)
    var args = List[String]()
    args.append(SELF_EXE)
    args.append(String("decode-child"))
    args.append(latent_path)
    args.append(rgb_path)
    args.append(vae_path)
    args.append(String(lh))
    args.append(String(lw))
    var child_argv = build_argv(args)
    var path = cstr(SELF_EXE)

    print("[sdxl] tiled decode -> fork VAE child; UNet remains resident (parent pid",
          _getpid(), ")")
    var pid = sys_fork()
    if pid == 0:
        _ = sys_execv(path, child_argv)
        sys__exit(127)
    if pid < 0:
        _unlink_file(latent_path)
        raise Error(String("sdxl decode child fork failed: ") + errno_str())

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
        raise Error("sdxl decode child timed out or waitpid failed")
    var exited_ok = (status & 0x7F) == 0 and ((status >> 8) & 0xFF) == 0
    if not exited_ok:
        _unlink_file(latent_path)
        _unlink_file(rgb_path)
        raise Error(
            String("sdxl decode child abnormal exit status ") + String(status)
        )

    var rgb = load_tensor_bin(rgb_path, ctx)
    _unlink_file(latent_path)
    _unlink_file(rgb_path)
    print("[sdxl] decode child reaped; tiled RGB loaded and UNet retained")
    return rgb^
