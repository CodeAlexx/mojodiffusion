# serenitymojo.serve.klein_decode_subprocess — the WHOLE-IMAGE Klein (FLUX.2) VAE
# decode in a fork+execv CHILD process, so it runs in a CLEAN CUDA context instead
# of tiling.
#
# WHY (MEASURED): tiled VAE decode measurably degrades output (MJ-1054). The
# monolithic KleinVaeDecoder[64,64].decode of a packed [1,128,64,64] latent peaks
# ~22 GB IN the sampler process — but that peak is dominated by the ~13–16 GB Klein
# DiT+LoRA stack pool that was freed at STAGE 2 yet NOT returned to the OS (the MAX
# caching allocator retains it). The whole-decode's own activations are only a few
# GB; they simply don't FIT on top of the retained stack pool, so klein_sampler
# falls back to a 3×3 tiled crop. Running the decode in a SEPARATE process gives it
# its own pool with no retained stack: the child decodes the whole latent, writes
# the RGB tensor to disk, and EXITS.
#
# TWO steps make the child fit the 24 GB card:
#   1) the PARENT trims its default pool to 0 before forking. Unlike the zimage
#      encoder case (encoder VRAM fragmented around a LIVE resident DiT → trim
#      reclaims 0), here the whole stack+LoRA is already freed with NOTHING live, so
#      cuMemPoolTrimTo(0) returns the full retained stack to the OS → free jumps to
#      ~device-total.
#   2) a pre-flight free-VRAM guard then confirms the child's estimated peak fits.
#
# WHY an RGB TENSOR (not the final PNG): klein_sample must RETURN the decoded image
# tensor (its contract) and the runtime backend re-opens the PNG afterward to embed
# genparams tEXt. So the child hands back the [1,3,16*LH,16*LW] rgb tensor
# (bit-identical raw bytes via io.cap_cache); the parent writes the PNG (save_image)
# and returns the tensor exactly as the in-process path does.
#
# SELF-EXEC (one binary): the decode child IS serenity_worker_klein re-exec'd with
# argv ["decode-child", <latent_path>, <rgb_out_path>, <vae_path>, <lh>, <lw>];
# serenity_worker_klein.main routes that to decode_child_run(). The comptime
# KleinVaeDecoder[LH,LW] specialization is chosen from the runtime lh/lw (the finite
# 512-square plus seven-shape 1MP set), mirroring how klein_runtime_backend
# dispatches klein_sample.
#
# SAFETY / FALLBACK: decode_whole_subprocess RAISES on guard failure, a host that
# does not route `decode-child`, or any subprocess failure. The caller
# (klein_sampler.klein_sample, only when allow_child_decode=True) catches and falls
# back to klein_tiled_decode. Standalone CLIs pass allow_child_decode=False and keep
# the tiled path untouched — no fork from a non-worker binary.
#
# Mirrors serve/zimage_decode_subprocess.mojo and the Phase-5 process-isolation
# contract (proc_ipc.mojo): argv built BEFORE fork, async-signal-safe calls only
# between fork and execv, execv into a fresh image.

from std.memory import alloc, UnsafePointer
from std.builtin.type_aliases import MutExternalOrigin
from std.ffi import external_call
from std.gpu.host import DeviceContext
from std.time import sleep

from serenitymojo.tensor import Tensor
from serenitymojo.io.cap_cache import save_tensor_bin, load_tensor_bin
from serenitymojo.models.vae.klein_decoder import KleinVaeDecoder
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


# Whole decode = VAE load + a few seconds of conv GPU work → 120 s is a pure hang
# backstop, after which we SIGKILL + fall back to tiled.
comptime _DECODE_CHILD_TIMEOUT_S = 120.0
comptime _DECODE_POLL_S = 0.05
# Pre-flight guard: the child is a SEPARATE process whose peak (KleinVaeDecoder
# weights + the whole-decode conv activations/cuDNN workspaces + its own CUDA
# context) must fit in the CURRENT free memory. The parent trims its pool to 0
# first (nothing live at STAGE 3), so free is ~device-total when the fork proceeds.
# ESTIMATE ≈ 10 GiB for the 1024² (LH=64) child (the ~22 GB in-process figure was
# dominated by the retained stack pool, NOT the decode itself; the decode's own
# activations are a few GB — NOT independently measured, the orchestrator
# validates). The 512² (LH=32) child is far smaller and clears this bar trivially.
# If this underestimates, the child OOMs → nonzero exit → tiled fallback.
comptime _DECODE_CHILD_MIN_FREE_BYTES = Int(10240) * 1024 * 1024  # ~10 GiB
comptime _KLEIN_PRODUCT_EDGE_UNITS = 16


def _getpid() -> Int:
    return Int(external_call["getpid", Int32]())


def _decode_child_shape[LH_: Int, LW_: Int](
    packed: Tensor, rgb_out_path: String, vae_path: String, ctx: DeviceContext
) raises:
    var dec = KleinVaeDecoder[LH_, LW_].load(vae_path, ctx)
    var rgb = dec.decode(packed, ctx)
    save_tensor_bin(rgb, rgb_out_path, ctx)


def decode_child_run(
    latent_path: String, rgb_out_path: String, vae_path: String, lh: Int, lw: Int
) raises:
    """CHILD body (after execv into `serenity_worker_klein decode-child`). Fresh
    process image → fresh CUDA context. Load the packed [1,128,lh,lw] latent, build
    the KleinVaeDecoder at the WHOLE (lh,lw) grid, decode the WHOLE latent, write the
    [1,3,16*lh,16*lw] rgb tensor to `rgb_out_path` (raw bytes via io.cap_cache), then
    RETURN so the process exits and the OS reclaims ALL decode VRAM.

    lh/lw select the comptime KleinVaeDecoder specialization from the finite Klein
    resolution set (512 square plus the seven-shape 1MP product ladder).
    Unsupported grids raise (parent → tiled fallback)."""
    var ctx = DeviceContext()
    var packed = load_tensor_bin(latent_path, ctx)
    if lh == 32 and lw == 32:
        _decode_child_shape[32, 32](packed, rgb_out_path, vae_path, ctx)
        print("[klein-decode-child] wrote rgb", rgb_out_path, "grid=", lh, "x", lw)
        return
    comptime for bi in range(DEFAULT_ASPECT_LADDER_LEN):
        comptime X100_BI = DEFAULT_ASPECT_LADDER_X100[bi]
        comptime LH_BI = aspect_lat_h_units(X100_BI, _KLEIN_PRODUCT_EDGE_UNITS) // 2
        comptime LW_BI = aspect_lat_w_units(X100_BI, _KLEIN_PRODUCT_EDGE_UNITS) // 2
        if lh == LH_BI and lw == LW_BI:
            _decode_child_shape[LH_BI, LW_BI](packed, rgb_out_path, vae_path, ctx)
            print("[klein-decode-child] wrote rgb", rgb_out_path,
                  "grid=", lh, "x", lw)
            return
    raise Error(
        String("klein decode-child: unsupported latent grid ")
        + String(lh) + String("x") + String(lw)
    )


def decode_whole_subprocess(
    packed_latent: Tensor, vae_path: String, lh: Int, lw: Int, ctx: DeviceContext
) raises -> Tensor:
    """PARENT body (klein worker STAGE 3, allow_child_decode=True). Trim the freed
    stack pool back to the OS, serialize the packed latent, fork+execv a fresh
    `serenity_worker_klein decode-child` to run the WHOLE-image VAE decode in ITS OWN
    clean pool, blocking-reap it, then read back the [1,3,16*lh,16*lw] rgb. RAISES on
    guard failure, a host that does not route `decode-child`, or any subprocess
    failure — the caller falls back to klein_tiled_decode."""
    var prefix = String("/tmp/serenity_klein_decode_") + String(_getpid())
    var latent_path = prefix + String(".latent.bin")
    var rgb_path = prefix + String(".rgb.bin")

    # STAGE 3 has freed the whole stack+LoRA (nothing live), so trimming the default
    # pool to 0 returns the retained stack bytes to the OS — this is what makes the
    # device-global free jump high enough for the child. (Contrast the encoder case,
    # where trim reclaims 0 because the encoder is fragmented around a live DiT.)
    try:
        cu_mempool_trim_current(0)
    except e:
        print("[klein] pre-decode pool trim failed (continuing):", e)

    # Pre-flight guard on the post-trim free memory — device-global free is exactly
    # what the child's separate CUDA context will see. Below threshold → raise
    # (caller tiles).
    var free_bytes = cu_mem_get_info().free_bytes
    if free_bytes < _DECODE_CHILD_MIN_FREE_BYTES:
        raise Error(
            String("klein decode child preflight failed: free VRAM ")
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
    args.append(vae_path)
    args.append(String(lh))
    args.append(String(lw))
    var argv = build_argv(args)
    var path = cstr(SELF_EXE)

    print("[klein] whole-image decode → fork VAE child (parent pid", _getpid(), ")")
    var pid = sys_fork()
    if pid == 0:
        # CHILD: async-signal-safe only, then execv into a fresh image.
        _ = sys_execv(path, argv)
        sys__exit(127)                     # execv failed
    if pid < 0:
        raise Error(String("klein decode child fork failed: ") + errno_str())

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
        raise Error("klein decode child timed out or waitpid failed")

    var exited_ok = (status & 0x7F) == 0 and ((status >> 8) & 0xFF) == 0
    if not exited_ok:
        raise Error(
            String("klein decode child abnormal exit status ") + String(status)
        )

    # Success path: read the rgb the child wrote. A read failure raises → caller
    # tiles.
    var rgb = load_tensor_bin(rgb_path, ctx)
    print("[klein] decode child reaped → whole-image rgb loaded (decode VRAM reclaimed)")
    return rgb^
