# serenitymojo.serve.sd3_decode_subprocess — process-isolated SD3 VAE decode.
#
# The parent SD3 worker retains its complete pinned-host FP8 denoiser store. A
# self-exec child receives only the final latent, performs whole-image decode (or
# the existing 5x5 fallback), writes RGB, and exits. Process death is the
# reliable CUDA allocation-reclaim boundary on this Mojo runtime.

from std.memory import alloc, UnsafePointer
from std.builtin.type_aliases import MutExternalOrigin
from std.ffi import external_call
from std.gpu.host import DeviceContext
from std.time import sleep

from serenitymojo.tensor import Tensor
from serenitymojo.io.cap_cache import save_tensor_bin, load_tensor_bin
from serenitymojo.models.vae.ldm_decoder import load_sd3_embedded_ldm_decoder
from serenitymojo.pipeline.sd3_tiled_decode import sd3_tiled_decode_5x5_lowmem
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


comptime _DECODE_CHILD_TIMEOUT_S = 180.0
comptime _DECODE_POLL_S = 0.05
comptime _WHOLE_CHILD_MIN_FREE_BYTES = Int(18432) * 1024 * 1024
comptime _TILED_CHILD_MIN_FREE_BYTES = Int(10240) * 1024 * 1024
comptime _SD3_PRODUCT_EDGE_UNITS = 16


def _getpid() -> Int:
    return Int(external_call["getpid", Int32]())


def _unlink_file(path: String):
    _ = external_call["unlink", Int32](cstr(path))


def _decode_child_shape[LH_: Int, LW_: Int](
    latent: Tensor,
    rgb_path: String,
    checkpoint_path: String,
    tiled: Bool,
    ctx: DeviceContext,
) raises:
    if tiled:
        var rgb = sd3_tiled_decode_5x5_lowmem[LH_, LW_](
            latent, checkpoint_path, ctx
        )
        save_tensor_bin(rgb, rgb_path, ctx)
        return
    var decoder = load_sd3_embedded_ldm_decoder[LH_, LW_](checkpoint_path, ctx)
    var rgb = decoder.decode(latent, ctx)
    save_tensor_bin(rgb, rgb_path, ctx)


def decode_child_run(
    latent_path: String,
    rgb_path: String,
    checkpoint_path: String,
    lh: Int,
    lw: Int,
    tiled: Bool,
) raises:
    var ctx = DeviceContext()
    var latent = load_tensor_bin(latent_path, ctx)
    comptime for bi in range(DEFAULT_ASPECT_LADDER_LEN):
        comptime X100_BI = DEFAULT_ASPECT_LADDER_X100[bi]
        comptime LH_BI = aspect_lat_h_units(X100_BI, _SD3_PRODUCT_EDGE_UNITS)
        comptime LW_BI = aspect_lat_w_units(X100_BI, _SD3_PRODUCT_EDGE_UNITS)
        if lh == LH_BI and lw == LW_BI:
            _decode_child_shape[LH_BI, LW_BI](
                latent, rgb_path, checkpoint_path, tiled, ctx
            )
            print(
                "[sd3-decode-child] wrote",
                "tiled" if tiled else "whole",
                "rgb", rgb_path, "grid=", lh, "x", lw,
            )
            return
    raise Error(
        String("sd3 decode child: unsupported latent grid ")
        + String(lh) + String("x") + String(lw)
    )


def decode_subprocess(
    latent: Tensor,
    checkpoint_path: String,
    lh: Int,
    lw: Int,
    tiled: Bool,
    ctx: DeviceContext,
) raises -> Tensor:
    var prefix = String("/tmp/serenity_sd3_decode_") + String(_getpid())
    var latent_path = prefix + String(".latent.bin")
    var rgb_path = prefix + String(".rgb.bin")

    try:
        cu_mempool_trim_current(0)
    except e:
        print("[sd3] pre-decode pool trim failed (continuing):", e)
    var required = (
        _TILED_CHILD_MIN_FREE_BYTES if tiled else _WHOLE_CHILD_MIN_FREE_BYTES
    )
    var free_bytes = cu_mem_get_info().free_bytes
    if free_bytes < required:
        raise Error(
            String("sd3 decode child preflight failed: free VRAM ")
            + String(free_bytes // (1024 * 1024))
            + String(" MiB < required ")
            + String(required // (1024 * 1024)) + String(" MiB")
        )

    save_tensor_bin(latent, latent_path, ctx)
    var mode = String("decode-tiled-child") if tiled else String("decode-whole-child")
    var args = List[String]()
    args.append(SELF_EXE)
    args.append(mode)
    args.append(latent_path)
    args.append(rgb_path)
    args.append(checkpoint_path)
    args.append(String(lh))
    args.append(String(lw))
    var child_argv = build_argv(args)
    var path = cstr(SELF_EXE)

    print(
        "[sd3]", "tiled" if tiled else "whole",
        "decode -> fork VAE child; host denoiser store remains warm (parent pid",
        _getpid(), ")",
    )
    var pid = sys_fork()
    if pid == 0:
        _ = sys_execv(path, child_argv)
        sys__exit(127)
    if pid < 0:
        _unlink_file(latent_path)
        raise Error(String("sd3 decode child fork failed: ") + errno_str())

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
        raise Error("sd3 decode child timed out or waitpid failed")
    var exited_ok = (status & 0x7F) == 0 and ((status >> 8) & 0xFF) == 0
    if not exited_ok:
        _unlink_file(latent_path)
        _unlink_file(rgb_path)
        raise Error(
            String("sd3 decode child abnormal exit status ") + String(status)
        )

    var rgb = load_tensor_bin(rgb_path, ctx)
    _unlink_file(latent_path)
    _unlink_file(rgb_path)
    print("[sd3] decode child reaped; RGB loaded and VAE VRAM returned")
    return rgb^
