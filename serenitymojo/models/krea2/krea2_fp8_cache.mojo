# krea2_fp8_cache.mojo — DISK SIDECAR for the krea2 fp8-quantized-resident base.
#
# PROBLEM (MJ: "compile every time? looks frozen"): every krea2 LoRA launch that
# selects `quantized_resident=fp8_e4m3` re-quantizes the 28 frozen blocks' 8
# matmul weights (E4M3 bytes + per-row F32 scale) from the 26GB bf16 checkpoint,
# ~3-4 minutes of GPU work, BEFORE the first training step. The quantized bytes
# are a pure function of the frozen checkpoint, so this is wasted every launch.
#
# FIX: after the first quantize, WRITE the resident store to ONE safetensors
# sidecar next to the checkpoint (`<ckpt>.fp8cache.safetensors`, ~12GB). Every
# later launch that finds a fresh sidecar RELOADS it H2D (~tens of seconds, a
# straight mmap→pinned-host→device copy) instead of re-quantizing.
#
# BIT-EXACTNESS is the bar: the sidecar stores the EXACT device bytes the
# quantizer produced (D2H raw copy through save_safetensors), and the load path
# copies those exact bytes back H2D (verbatim, dtype-preserving). So a store
# loaded-from-sidecar is BYTE-IDENTICAL to one freshly quantized → identical
# training math. Gated by krea2_fp8_cache_gate.mojo (blocks 0 and 27, every fp8
# byte + scale byte, byte-for-byte).
#
# STALENESS: the sidecar embeds `__meta__.*` tensors — the source checkpoint's
# path + size + mtime (stat(2), which follows the HF-cache symlink to the blob) +
# nblocks + a format version. A launch loads the sidecar ONLY if all match the
# current checkpoint; any mismatch (checkpoint re-downloaded, nblocks changed,
# format bumped) falls back to a fresh quantize (+rewrite). The cache is
# WEIGHTS-ONLY — it does NOT depend on LTMAX / bucket / LoRA config, so one
# sidecar is valid across every krea2 run against the same checkpoint.
#
# Mojo 1.0.0b1, Linux x86-64, NVIDIA GPU.

from std.memory import alloc, UnsafePointer, ArcPointer
from std.ffi import external_call
from max.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.io.ffi import BytePtr, sys_memcpy
from serenitymojo.models.dit.krea2_dit import (
    Krea2ResidentFp8,
    Krea2BlockResidentFp8,
)


# Sidecar format version — bump if the tensor naming / layout below changes so
# an old sidecar is treated stale instead of silently mis-loaded.
comptime KREA2_FP8CACHE_VERSION = 1

# 8 matmul-weight slots per block (Krea2BlockWeights field order:
# wq wk wv gate wo mlp_gate mlp_up mlp_down) — the tensors that get fp8'd.
comptime _FP8_KEYS = 8


def krea2_fp8_cache_path(checkpoint: String) -> String:
    """The sidecar path for a given base checkpoint: `<ckpt>.fp8cache.safetensors`
    right next to the checkpoint (deterministic; no config needed)."""
    return checkpoint + String(".fp8cache.safetensors")


# ── stat(2): source checkpoint size + mtime for the staleness guard ───────────
# Linux x86-64 `struct stat` is ABI-stable: st_size at byte offset 48 (off_t),
# st_mtim.tv_sec at offset 88 (time_t) — both 8-byte, 8-aligned. `stat` (not
# `lstat`) follows the HF-cache symlink to the blob, so we key on the real file.
# glibc >= 2.33 exports `stat` as a real symbol (this box is glibc >> 2.33).
def _stat_size_mtime(path: String) -> List[Int]:
    """[size, mtime_sec] of `path`, or [-1, -1] if stat fails."""
    var n = path.byte_length()
    var cbuf = alloc[UInt8](n + 1)
    var src = path.as_bytes()
    for i in range(n):
        cbuf[i] = src[i]
    cbuf[n] = 0
    var statbuf = alloc[UInt8](160)   # struct stat is 144 bytes on x86-64
    var rc = Int(
        external_call["stat", Int32](
            BytePtr(unsafe_from_address=Int(cbuf)),
            BytePtr(unsafe_from_address=Int(statbuf)),
        )
    )
    var size = -1
    var mtime = -1
    if rc == 0:
        var q = statbuf.bitcast[Int64]()
        size = Int(q[6])    # offset 48 / 8 == st_size
        mtime = Int(q[11])  # offset 88 / 8 == st_mtim.tv_sec
    cbuf.free()
    statbuf.free()
    var out: List[Int] = [size, mtime]
    return out^


# ── tiny device tensors that carry the __meta__ staleness fields ──────────────
def _i64_tensor(v: Int, ctx: DeviceContext) raises -> Tensor:
    """A [1] I64 device tensor holding `v` (little-endian, native x86-64)."""
    var host = ctx.enqueue_create_host_buffer[DType.uint8](8)
    host.unsafe_ptr().bitcast[Int64]()[0] = Int64(v)
    var dev = ctx.enqueue_create_buffer[DType.uint8](8)
    ctx.enqueue_copy(dst_buf=dev, src_buf=host)
    ctx.synchronize()
    var shape: List[Int] = [1]
    return Tensor(dev^, shape^, STDtype.I64)


def _u8_tensor(s: String, ctx: DeviceContext) raises -> Tensor:
    """A [len] U8 device tensor holding the UTF-8 bytes of `s` (the src path)."""
    var n = s.byte_length()
    if n == 0:
        raise Error("_u8_tensor: empty string")
    var host = ctx.enqueue_create_host_buffer[DType.uint8](n)
    var src = s.as_bytes()
    var hp = host.unsafe_ptr()
    for i in range(n):
        hp[i] = src[i]
    var dev = ctx.enqueue_create_buffer[DType.uint8](n)
    ctx.enqueue_copy(dst_buf=dev, src_buf=host)
    ctx.synchronize()
    var shape: List[Int] = [n]
    return Tensor(dev^, shape^, STDtype.U8)


def _meta_i64(st: SafeTensors, name: String, default: Int) raises -> Int:
    """Read a [1] I64 meta tensor's value from the mmap'd bytes; `default` if the
    tensor is absent. Assembled little-endian from the 8 raw bytes."""
    if name not in st.tensors:
        return default
    var b = st.tensor_bytes(name)
    if len(b) < 8:
        return default
    var v = 0
    for i in range(8):
        v = v | (Int(b[i]) << (8 * i))
    return v


def _meta_str(st: SafeTensors, name: String) raises -> String:
    """Read a U8 meta tensor's bytes back into a String; empty if absent."""
    if name not in st.tensors:
        return String("")
    var b = st.tensor_bytes(name)
    var s = String("")
    for i in range(len(b)):
        s += chr(Int(b[i]))
    return s^


def krea2_fp8_cache_valid(
    cache_path: String, checkpoint: String, nblocks: Int
) raises -> Bool:
    """True iff a sidecar at `cache_path` exists AND its embedded staleness meta
    matches the current `checkpoint` (path+size+mtime), `nblocks`, and the format
    version. Any failure to open/parse → False (fall back to fresh quantize)."""
    var sm = _stat_size_mtime(checkpoint)
    if sm[0] < 0:
        return False   # can't stat the source → can't validate → re-quantize
    var st: SafeTensors
    try:
        st = SafeTensors.open(cache_path)
    except:
        return False   # missing / unreadable / malformed sidecar
    if _meta_i64(st, String("__meta__.version"), -1) != KREA2_FP8CACHE_VERSION:
        return False
    if _meta_i64(st, String("__meta__.nblocks"), -1) != nblocks:
        return False
    if _meta_i64(st, String("__meta__.src_size"), -1) != sm[0]:
        return False
    if _meta_i64(st, String("__meta__.src_mtime"), -1) != sm[1]:
        return False
    if _meta_str(st, String("__meta__.src_path")) != checkpoint:
        return False
    return True


# ── fast verbatim H2D load (dtype-preserving) ─────────────────────────────────
# Tensor.from_view rejects non-compute dtypes (F8_E4M3) and Tensor.from_view_raw
# copies byte-by-byte in a host loop (fatal for 12GB). This mirrors from_view's
# fast sys_memcpy → enqueue_copy, but preserves the on-disk STDtype verbatim
# (F8_E4M3 / F32 / BF16), so the device bytes are byte-identical to the sidecar.
def _load_raw_h2d(st: SafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    var info = st.tensor_info(name)
    var bytes = st.tensor_bytes(name)
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    var nbytes = tv.nbytes()
    var host = ctx.enqueue_create_host_buffer[DType.uint8](nbytes)
    var dst = BytePtr(unsafe_from_address=Int(host.unsafe_ptr()))
    var srcp = BytePtr(unsafe_from_address=Int(tv.data.unsafe_ptr()))
    _ = sys_memcpy(dst, srcp, nbytes)
    var dev = ctx.enqueue_create_buffer[DType.uint8](nbytes)
    ctx.enqueue_copy(dst_buf=dev, src_buf=host)
    ctx.synchronize()
    return Tensor(dev^, tv.shape.copy(), tv.dtype)


def _block_prefix(bi: Int) -> String:
    return String("b") + String(bi) + String(".")


def load_krea2_fp8_cache(
    cache_path: String, nblocks: Int, ctx: DeviceContext,
    resident_blocks: Int = -1,
) raises -> Krea2ResidentFp8:
    """Rebuild the fp8-resident store from a sidecar (assumes caller already
    validated staleness). Verbatim H2D copy per tensor → byte-identical to the
    store the quantizer originally produced. Progress every 7 blocks (the UI
    tails these lines, same cadence as the fresh-quantize path).

    `resident_blocks` mirrors build_krea2_resident_fp8's parameter of the same
    name: load only the FIRST N blocks to device and leave [N:nblocks] to the
    caller's per-step stream (the consumers all dispatch on
    `bi < len(resident.blocks)`, so a short store degrades to streaming rather
    than misindexing). -1 (default) = all nblocks, the prior behavior for every
    existing caller.

    This parameter exists because the sidecar path used to ignore the resident
    budget entirely: a build with -D KREA2_RESIDENT_BLOCKS=4 still loaded all 28
    blocks (~12GB) whenever a sidecar was present, so the 16GB refit silently did
    nothing and 1024px OOM'd at identical peak for every flag value (measured
    2026-07-26 on the 5080: 14.4GB peak at both 12 and 4)."""
    var st = SafeTensors.open(cache_path)
    var n_res = nblocks if resident_blocks < 0 else (
        resident_blocks if resident_blocks < nblocks else nblocks
    )
    var blocks = List[Krea2BlockResidentFp8]()
    for bi in range(n_res):
        var p = _block_prefix(bi)
        var fp8 = List[ArcPointer[Tensor]]()
        var scale = List[ArcPointer[Tensor]]()
        for ki in range(_FP8_KEYS):
            fp8.append(ArcPointer(_load_raw_h2d(st, p + String("fp8.") + String(ki), ctx)))
            scale.append(ArcPointer(_load_raw_h2d(st, p + String("scale.") + String(ki), ctx)))
        blocks.append(Krea2BlockResidentFp8(
            fp8^, scale^,
            ArcPointer(_load_raw_h2d(st, p + String("qnorm_scale"), ctx)),
            ArcPointer(_load_raw_h2d(st, p + String("knorm_scale"), ctx)),
            ArcPointer(_load_raw_h2d(st, p + String("prenorm_scale"), ctx)),
            ArcPointer(_load_raw_h2d(st, p + String("postnorm_scale"), ctx)),
            ArcPointer(_load_raw_h2d(st, p + String("mod_lin"), ctx)),
        ))
        if (bi + 1) % 7 == 0 or bi + 1 == n_res:
            print("[fp8cache] loaded", bi + 1, "/", n_res, "blocks from sidecar")
    return Krea2ResidentFp8(blocks^)


def save_krea2_fp8_cache(
    store: Krea2ResidentFp8,
    checkpoint: String,
    cache_path: String,
    nblocks: Int,
    ctx: DeviceContext,
) raises:
    """Write the freshly-quantized `store` to the sidecar (one atomic safetensors,
    tmp+rename via save_safetensors). Embeds the __meta__ staleness fields keyed
    on the current checkpoint's stat(2). Raises if the checkpoint cannot be
    stat'd (caller treats a write failure as non-fatal: just no cache)."""
    var sm = _stat_size_mtime(checkpoint)
    if sm[0] < 0:
        raise Error(
            String("[fp8cache] cannot stat checkpoint for staleness meta: ")
            + checkpoint
        )
    var names = List[String]()
    var tensors = List[ArcPointer[Tensor]]()

    # __meta__ staleness guard (weights-only: no LTMAX/bucket/LoRA dependence).
    names.append(String("__meta__.version"))
    tensors.append(ArcPointer(_i64_tensor(KREA2_FP8CACHE_VERSION, ctx)))
    names.append(String("__meta__.nblocks"))
    tensors.append(ArcPointer(_i64_tensor(nblocks, ctx)))
    names.append(String("__meta__.src_size"))
    tensors.append(ArcPointer(_i64_tensor(sm[0], ctx)))
    names.append(String("__meta__.src_mtime"))
    tensors.append(ArcPointer(_i64_tensor(sm[1], ctx)))
    names.append(String("__meta__.src_path"))
    tensors.append(ArcPointer(_u8_tensor(checkpoint, ctx)))

    if len(store.blocks) != nblocks:
        raise Error(
            String("save_krea2_fp8_cache: store has ")
            + String(len(store.blocks)) + " blocks, expected " + String(nblocks)
        )
    for bi in range(nblocks):
        var p = _block_prefix(bi)
        ref b = store.blocks[bi]
        if len(b.fp8) != _FP8_KEYS or len(b.scale) != _FP8_KEYS:
            raise Error(
                String("save_krea2_fp8_cache: block ") + String(bi)
                + " has " + String(len(b.fp8)) + " fp8 / "
                + String(len(b.scale)) + " scale, expected " + String(_FP8_KEYS)
            )
        for ki in range(_FP8_KEYS):
            names.append(p + String("fp8.") + String(ki))
            tensors.append(b.fp8[ki].copy())
            names.append(p + String("scale.") + String(ki))
            tensors.append(b.scale[ki].copy())
        names.append(p + String("qnorm_scale"))
        tensors.append(b.qnorm_scale.copy())
        names.append(p + String("knorm_scale"))
        tensors.append(b.knorm_scale.copy())
        names.append(p + String("prenorm_scale"))
        tensors.append(b.prenorm_scale.copy())
        names.append(p + String("postnorm_scale"))
        tensors.append(b.postnorm_scale.copy())
        names.append(p + String("mod_lin"))
        tensors.append(b.mod_lin.copy())

    save_safetensors(names, tensors, cache_path, ctx)
