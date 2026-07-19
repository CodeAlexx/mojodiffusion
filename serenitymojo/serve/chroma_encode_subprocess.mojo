# serenitymojo.serve.chroma_encode_subprocess — the T5-XXL text encoder in a
# fork+execv CHILD process, so its VRAM is reclaimed by process death.
#
# WHY (MEASURED, jobs 0075/0076): the T5-XXL encoder (~9.5 GB F16) loaded IN the
# long-lived chroma worker grows the process's CUDA memory pool by ~12.8 GB and
# NOTHING gets it back: after the encoder was dropped — and even after bouncing
# the surviving cap tensors to host so the T5 segments held zero live
# suballocations — cuMemPoolTrimTo(0) reclaimed ~0 ("after T5 segment trim:
# 13841 MiB", was 1073 before the load) and the whole-image VAE decode then
# OOM'd the 16 GB card. The ONLY reliable reclaim is process exit. So the worker
# fork+execv's a FRESH child that loads the encoder, encodes (prompt, negative),
# writes the BF16 caps to disk (bit-identical raw bytes via io.cap_cache), and
# EXITS — the OS frees every byte of the encoder VRAM. The parent waitpid's
# (blocking-reap → VRAM released before we touch the GPU again), reads the caps
# back (a one-time ~8 MB H2D), and streams the DiT from a clean card.
#
# This is the SAME split PROVEN on this card by zimage_encode_subprocess.mojo
# (this file is its port), under the Phase-5 process-isolation contract
# (proc_ipc.mojo: argv built BEFORE fork, async-signal-safe calls only between
# fork and execv, execv into a fresh image — fork()ing a CUDA/AsyncRT process is
# only safe if the child does nothing but execv).
#
# SELF-EXEC (one binary, one build): the encode child IS serenity_worker_chroma
# re-exec'd with argv ["encode-child", <prefix>, <prompt>, <negative>];
# serenity_worker_chroma.main routes that to encode_child_run(). No separate
# encoder binary.
#
# SAFETY / FALLBACK: for ANY subprocess failure (fork error, timeout, abnormal
# exit, unreadable caps) — or a host binary that does not route "encode-child" —
# encode_captions_subprocess transparently falls back to the in-process
# encode_captions_inprocess. Correctness is never sacrificed for the VRAM win.
#
# ENCODE MATH (byte-identical to the flux_sample_cli T5 contract): T5-XXL
# Unigram tokenizer (bit-exact vs HF) → _fit to 512 (pad 0, eos 1) →
# T5Encoder[512] forward → BF16 cast. Caps are [1,512,4096] BF16 cond + uncond;
# the unpadded token counts ride a 24-byte meta sidecar and feed the Chroma DiT
# T5 pad-row attention mask (MJ-1048).

from std.ffi import external_call
from std.memory import alloc, UnsafePointer
from std.builtin.type_aliases import MutExternalOrigin
from std.gpu.host import DeviceContext
from std.time import sleep

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.cap_cache import save_tensor_bin, load_tensor_bin
from serenitymojo.io.ffi import (
    BytePtr, sys_open, sys_pwrite, sys_pread, sys_close,
    O_WRONLY, O_CREAT, O_TRUNC, O_RDONLY,
)
from serenitymojo.models.text_encoder.t5_encoder import T5Encoder, T5Config
from serenitymojo.tokenizer.t5_tokenizer import T5Tokenizer
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.serve.proc_ipc import (
    build_argv, cstr, sys_execv, sys__exit, sys_waitpid, proc_kill_wait,
    SELF_EXE, SIGKILL, WNOHANG,
)
from serenitymojo.offload.vmm_cuda import cu_mem_get_info
from net.syscalls import sys_fork, errno_str


# T5-XXL encoder artifacts — the SAME files flux_backend's encode_text uses
# (staged under the installer's shared models/text-encoders directory).
comptime T5_PATH = "models/text-encoders/t5xxl_fp16.safetensors"
comptime T5_TOK_JSON = "models/text-encoders/t5xxl_fp16.tokenizer.json"
comptime T5_LEN = 512  # == CHROMA_T5_SEQ_LEN

# encoder load (~9.5 GB F16 from disk) + cond+uncond forward is tens of seconds
# → 300 s is a pure hang backstop, after which we SIGKILL + fall back.
comptime _ENCODE_CHILD_TIMEOUT_S = 300.0
comptime _ENCODE_POLL_S = 0.05
# Pre-flight guard: the encoder child is a separate process whose peak
# (~9.5 GB F16 T5 weights + F32/BF16 cast copies + forward activations + its own
# CUDA context) must fit in the GPU's CURRENT free memory alongside this
# resident parent. Require ~11 GB free before forking, else go straight to
# in-process (the parent's grown pool can't be trimmed back to make room —
# measured cuMemPoolTrim reclaims 0). The chroma parent sits at ~1 GB when the
# encode runs (nothing resident before LOAD), so a clean-card job always forks.
comptime _ENCODE_CHILD_MIN_FREE_BYTES = Int(11264) * 1024 * 1024  # ~11 GiB
# 24-byte binary sidecar: [magic][real_cond][real_uncond] as 3x Int64 LE.
comptime _META_MAGIC = Int64(0x4348524341505631)  # "CHRCAPV1"


# ── one encoded caption pair (fixed [1,512,4096] BF16 buffers + true counts) ──
@fieldwise_init
struct ChromaCaps(Movable):
    var cond: Tensor      # [1, 512, 4096] BF16 T5 hidden (prompt)
    var uncond: Tensor    # [1, 512, 4096] BF16 T5 hidden (negative)
    var real_cond: Int    # true cond token count (≤ 512), incl. EOS
    var real_uncond: Int  # true uncond token count (≤ 512), incl. EOS


# ── pad/truncate token ids to a fixed length, keeping EOS at the tail ─────────
# Byte-copy of flux_sample_cli._fit (HF T5: max 512, pad 0, eos 1; encode()
# already includes the special tokens).
def _fit(var ids: List[Int], target: Int, pad_id: Int, eos_id: Int) -> List[Int]:
    var out = List[Int]()
    if len(ids) >= target:
        for i in range(target - 1):
            out.append(ids[i])
        out.append(eos_id)
    else:
        for i in range(len(ids)):
            out.append(ids[i])
        while len(out) < target:
            out.append(pad_id)
    return out^


# Byte-copy of flux_sample_cli._to_bf16 (F16 → F32 → BF16 two-hop cast).
def _to_bf16(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    if x.dtype() == STDtype.BF16:
        return cast_tensor(x, STDtype.BF16, ctx)
    if x.dtype() == STDtype.F16:
        var x_f32 = cast_tensor(x, STDtype.F32, ctx)
        return cast_tensor(x_f32, STDtype.BF16, ctx)
    return cast_tensor(x, STDtype.BF16, ctx)


def encode_captions_inprocess(
    prompt: String, negative: String, ctx: DeviceContext
) raises -> ChromaCaps:
    """The Chroma T5-XXL encode (cond AND uncond — Chroma runs real CFG, so the
    negative IS encoded). Loads the encoder, uses it for both forwards, frees it
    at scope exit. This is the CHILD body's math and the parent's fallback —
    identical either way. Also returns the unpadded token counts (clamped to
    [1, 512]), the live analog of the chroma_encode.rs sidecar's
    cond_real_len/uncond_real_len keys (MJ-1048 pad-row mask)."""
    var t5_tok = T5Tokenizer(T5_TOK_JSON)
    # T5-XXL: encode() adds EOS(1); fit to 512 (pad 0) — flux _fit contract.
    var cond_raw = t5_tok.encode(prompt)
    var real_cond = len(cond_raw)
    if real_cond > T5_LEN:
        real_cond = T5_LEN
    if real_cond < 1:
        real_cond = 1
    var cond_ids = _fit(cond_raw^, T5_LEN, 0, 1)
    var uncond_raw = t5_tok.encode(negative)
    var real_uncond = len(uncond_raw)
    if real_uncond > T5_LEN:
        real_uncond = T5_LEN
    if real_uncond < 1:
        real_uncond = 1
    var uncond_ids = _fit(uncond_raw^, T5_LEN, 0, 1)
    print("[chroma][text] T5-XXL encode (cond", real_cond, "ids, uncond",
          real_uncond, "ids of", T5_LEN, ")")
    var t5 = T5Encoder[T5_LEN].load(T5_PATH, T5Config.t5_xxl(), ctx)
    var cond_hidden = _to_bf16(t5.encode(cond_ids^, ctx), ctx)
    var uncond_hidden = _to_bf16(t5.encode(uncond_ids^, ctx), ctx)
    return ChromaCaps(cond_hidden^, uncond_hidden^, real_cond, real_uncond)


def _getpid() -> Int:
    return Int(external_call["getpid", Int32]())


def _unlink_file(path: String):
    """Best-effort delete of a /tmp cap sidecar (mirrors serenity_daemon._unlink_file)."""
    _ = external_call["unlink", Int32](cstr(path))


def _write_meta(path: String, real_cond: Int, real_uncond: Int) raises:
    var fd = sys_open(path, O_WRONLY | O_CREAT | O_TRUNC, 0o644)
    if fd < 0:
        raise Error(String("chroma_encode_subprocess: meta open failed: ") + path)
    var tmp = alloc[Int64](3)
    tmp[0] = _META_MAGIC
    tmp[1] = Int64(real_cond)
    tmp[2] = Int64(real_uncond)
    var p = BytePtr(unsafe_from_address=Int(tmp))
    var w = sys_pwrite(fd, p, 24, 0)
    tmp.free()
    _ = sys_close(fd)
    if w != 24:
        raise Error("chroma_encode_subprocess: short meta write")


def _read_meta(path: String) raises -> List[Int]:
    var fd = sys_open(path, O_RDONLY, 0)
    if fd < 0:
        raise Error(String("chroma_encode_subprocess: meta open failed: ") + path)
    var tmp = alloc[Int64](3)
    var p = BytePtr(unsafe_from_address=Int(tmp))
    var r = sys_pread(fd, p, 24, 0)
    var magic = tmp[0]
    var rc = Int(tmp[1])
    var ru = Int(tmp[2])
    tmp.free()
    _ = sys_close(fd)
    if r != 24:
        raise Error("chroma_encode_subprocess: short meta read")
    if magic != _META_MAGIC:
        raise Error("chroma_encode_subprocess: bad meta magic")
    var out = List[Int]()
    out.append(rc)
    out.append(ru)
    return out^


def encode_child_run(prefix: String, prompt: String, negative: String) raises:
    """CHILD body (after execv into `serenity_worker_chroma encode-child`).
    Fresh process image → fresh CUDA context. Load the T5-XXL encoder, encode
    cond+uncond, write the BF16 caps + token counts to
    `<prefix>.{cond,uncond}.bin` / `<prefix>.meta`, then RETURN so the process
    exits and the OS reclaims ALL encoder VRAM. The `.meta` file is written
    LAST: its presence + magic is the parent's "child fully succeeded" signal."""
    var ctx = DeviceContext()
    var caps = encode_captions_inprocess(prompt, negative, ctx)
    save_tensor_bin(caps.cond, prefix + String(".cond.bin"), ctx)
    save_tensor_bin(caps.uncond, prefix + String(".uncond.bin"), ctx)
    _write_meta(prefix + String(".meta"), caps.real_cond, caps.real_uncond)
    print("[chroma-encode-child] wrote caps", prefix,
          "real_cond=", caps.real_cond, "real_uncond=", caps.real_uncond)


def encode_captions_subprocess(
    prompt: String, negative: String, ctx: DeviceContext
) raises -> ChromaCaps:
    """PARENT body (chroma worker ENCODE phase). fork+execv a fresh
    `serenity_worker_chroma encode-child` to run the ~9.5 GB T5-XXL encoder in
    ITS OWN process, blocking-reap it (VRAM released by process death), then
    read back the BF16 caps it wrote and delete the /tmp sidecars. Nothing is
    resident in THIS process during the encode (the chroma DiT loads after),
    so this CUDA context is never used in the child: fork copies the fd table
    and the child execv's immediately. Falls back to in-process
    `encode_captions_inprocess` on any failure or on a host binary that does
    not route `encode-child` (see module header)."""
    var prefix = String("/tmp/serenity_chroma_caps_") + String(_getpid())
    var cond_path = prefix + String(".cond.bin")
    var uncond_path = prefix + String(".uncond.bin")
    var meta_path = prefix + String(".meta")

    # Pre-flight guard: skip a doomed fork when the GPU's current free memory
    # can't hold the ~11 GB encoder child. cu_mem_get_info reports device-global
    # free, which is exactly what the child's separate CUDA context will see.
    # Below threshold → in-process encode (correct, just no reclaim win).
    var free_bytes = cu_mem_get_info().free_bytes
    if free_bytes < _ENCODE_CHILD_MIN_FREE_BYTES:
        print("[chroma] free VRAM", free_bytes // (1024 * 1024),
              "MiB < encoder-child need", _ENCODE_CHILD_MIN_FREE_BYTES // (1024 * 1024),
              "MiB → in-process encode (no fork)")
        return encode_captions_inprocess(prompt, negative, ctx)

    # argv + execv path built BEFORE fork (no allocation between fork and execv).
    var args = List[String]()
    args.append(SELF_EXE)                  # argv[0]
    args.append(String("encode-child"))
    args.append(prefix)
    args.append(prompt)
    args.append(negative)
    var argv = build_argv(args)
    var path = cstr(SELF_EXE)

    print("[chroma] fork encoder child (parent pid", _getpid(), ")")
    var pid = sys_fork()
    if pid == 0:
        # CHILD: async-signal-safe only, then execv into a fresh image.
        _ = sys_execv(path, argv)
        sys__exit(127)                     # execv failed
    if pid < 0:
        print("[chroma] fork failed (", errno_str(), ") → in-process encode")
        return encode_captions_inprocess(prompt, negative, ctx)

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
        print("[chroma] encoder child timed out/errored → in-process encode")
        return encode_captions_inprocess(prompt, negative, ctx)

    var exited_ok = (status & 0x7F) == 0 and ((status >> 8) & 0xFF) == 0
    if not exited_ok:
        print("[chroma] encoder child abnormal exit (status", status,
              ") → in-process encode")
        return encode_captions_inprocess(prompt, negative, ctx)

    # Success path: read the caps the child wrote, then delete the sidecars.
    # Any read failure (e.g. a host that exits 0 but never wrote the sidecar)
    # → in-process fallback.
    try:
        var meta = _read_meta(meta_path)
        var cond = load_tensor_bin(cond_path, ctx)
        var uncond = load_tensor_bin(uncond_path, ctx)
        _unlink_file(cond_path)
        _unlink_file(uncond_path)
        _unlink_file(meta_path)
        print("[chroma] encoder child reaped → caps loaded (encoder VRAM reclaimed)")
        return ChromaCaps(cond^, uncond^, meta[0], meta[1])
    except e:
        print("[chroma] caps read-back failed (", e, ") → in-process encode")
        return encode_captions_inprocess(prompt, negative, ctx)
