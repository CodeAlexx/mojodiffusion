# krea2_int8_cache.mojo — DISK SIDECAR for the krea2 int8 W8A8 hybrid base
# (task #13). Mirror of krea2_fp8_cache.mojo for the int8 inference base.
#
# PROBLEM: every krea2 EDIT-inference launch (default base = int8 W8A8 hybrid,
# pipeline/krea2_edit_infer.mojo) pays ~35s to load-once-quantize the 28 frozen
# blocks' 8 matmul weights (tensorwise int8 encode) from the bf16 checkpoint —
# blocks [0:K) via build_krea2_resident_int8 (device-resident) and [K:28) via
# build_krea2_host_int8_inf (pinned-host). The int8 bytes + scalar scales are a
# pure function of the FROZEN checkpoint, so this is wasted every launch.
#
# FIX: after the first quantize, WRITE all 28 blocks to ONE safetensors sidecar
# next to the checkpoint (`<ckpt>.int8cache.safetensors`, ~12GB). Later launches
# that find a fresh sidecar RELOAD it — the resident split [0:K) straight
# mmap→pinned-host→device H2D, the host split [K:28) mmap→pinned-host memcpy
# with NO device round-trip — instead of touching the bf16 checkpoint blocks.
#
# BIT-EXACTNESS is the bar (same as fp8cache): the sidecar stores the EXACT
# int8/scale/small-tensor bytes the quantizer produced (resident tensors D2H raw;
# pinned-host int8 bytes written directly from host RAM), and the load path
# copies those exact bytes back verbatim (dtype-preserving). A store loaded from
# the sidecar is BYTE-IDENTICAL to one freshly quantized → identical math.
#
# STALENESS: identical scheme to krea2_fp8_cache — `__meta__.*` tensors embed
# the source checkpoint's path + size + mtime (stat(2) follows the HF-cache
# symlink to the blob) + nblocks + a format version. Any mismatch → fresh
# quantize (+rewrite). The cache is WEIGHTS-ONLY and SPLIT-INDEPENDENT: all 28
# blocks are stored the same way, so one sidecar serves ANY --i8-resident-blocks
# K against the same checkpoint (the loader applies the split at load time).
#
# WRITER: save_safetensors would need every payload as a DEVICE tensor, but the
# host split's int8 bytes live in pinned HOST RAM (~433MB/block) — H2D'ing them
# just to D2H them back would add a multi-GB transient device spike next to the
# ~9GB resident store on the 16GB card. So this module carries its own streaming
# writer (same byte-exact format + tmp+rename atomicity as save_safetensors):
# device tensors are D2H'd one at a time, pinned-host payloads pwritten direct.
#
# Mojo 1.0.0b1, Linux x86-64, NVIDIA GPU.

from std.collections import Optional
from std.memory import alloc, UnsafePointer, ArcPointer
from std.ffi import external_call
from max.gpu.host import DeviceContext, HostBuffer

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.ffi import (
    BytePtr, sys_memcpy, sys_open, sys_close, sys_rename, sys_remove,
    O_WRONLY, O_CREAT, O_TRUNC,
)
from serenitymojo.io.safetensors_writer import _write_all
from serenitymojo.models.dit.krea2_dit import (
    Krea2ResidentInt8, Krea2BlockResidentInt8,
    Krea2HostInt8Inf, Krea2BlockHostInt8Inf,
    Krea2SharedResident, Krea2TextFusionWeights,
)
# Shared with the fp8 sidecar: stat(2) staleness key, meta readers, and the
# verbatim dtype-preserving H2D loader (I8/F32/BF16 pass through untouched).
from serenitymojo.models.krea2.krea2_fp8_cache import (
    _stat_size_mtime, _meta_i64, _meta_str, _load_raw_h2d,
)


# Sidecar format version — bump if the tensor naming / layout below changes so
# an old sidecar is treated stale instead of silently mis-loaded.
# v2: + the SHARED (non-block) resident weights (shared.t.* / shared.txtf.*) —
# the ~1.3GB first/tmlp/tproj/txtfusion/txtmlp/last set, so a sidecar launch
# never opens the bf16 checkpoint's big tensors at all (~2.5s saved).
comptime KREA2_INT8CACHE_VERSION = 2

# 8 matmul-weight slots per block (Krea2BlockWeights field order:
# wq wk wv gate wo mlp_gate mlp_up mlp_down) — the tensors that get int8'd.
comptime _I8_KEYS = 8

# Shared store layout (build_krea2_shared_resident): 18 flat tensor slots + 4
# TextFusion bundles of 12 tensors each (fixed field order, see _txtf save/load).
comptime _SHARED_T = 18
comptime _SHARED_TXTF = 4

comptime _HArc = ArcPointer[HostBuffer[DType.uint8]]


def krea2_int8_cache_path(checkpoint: String) -> String:
    """The sidecar path for a given base checkpoint: `<ckpt>.int8cache.safetensors`
    right next to the checkpoint (deterministic; no config needed)."""
    return checkpoint + String(".int8cache.safetensors")


def _stat_dev_ino(path: String) -> List[Int]:
    """Return Linux stat(2) [st_dev, st_ino], or [-1, -1] on failure."""
    var n = path.byte_length()
    var cbuf = alloc[UInt8](n + 1)
    var src = path.as_bytes()
    for i in range(n):
        cbuf[i] = src[i]
    cbuf[n] = 0
    var statbuf = alloc[UInt8](160)
    var rc = Int(
        external_call["stat", Int32](
            BytePtr(unsafe_from_address=Int(cbuf)),
            BytePtr(unsafe_from_address=Int(statbuf)),
        )
    )
    var dev = -1
    var ino = -1
    if rc == 0:
        var q = statbuf.bitcast[Int64]()
        dev = Int(q[0])
        ino = Int(q[1])
    cbuf.free()
    statbuf.free()
    var out: List[Int] = [dev, ino]
    return out^


def krea2_same_checkpoint_file(left: String, right: String) -> Bool:
    """True only when both paths resolve through stat(2) to one exact file."""
    var a = _stat_dev_ino(left)
    var b = _stat_dev_ino(right)
    return a[0] >= 0 and a[0] == b[0] and a[1] == b[1]


def krea2_int8_cache_valid(
    cache_path: String, checkpoint: String, nblocks: Int
) raises -> Bool:
    """True iff a sidecar at `cache_path` exists AND its embedded staleness meta
    matches the current `checkpoint` (path+size+mtime), `nblocks`, and the format
    version. Any failure to open/parse → False (fall back to fresh quantize).
    NOTE: deliberately does NOT key on the resident/host split K — the sidecar
    stores all `nblocks` blocks uniformly and the loader splits at load time."""
    var sm = _stat_size_mtime(checkpoint)
    if sm[0] < 0:
        return False   # can't stat the source → can't validate → re-quantize
    var st: SafeTensors
    try:
        st = SafeTensors.open(cache_path)
    except:
        return False   # missing / unreadable / malformed sidecar
    if _meta_i64(st, String("__meta__.version"), -1) != KREA2_INT8CACHE_VERSION:
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


def _block_prefix(bi: Int) -> String:
    return String("b") + String(bi) + String(".")


# ── LOADERS ───────────────────────────────────────────────────────────────────

def load_krea2_int8_cache_resident(
    cache_path: String, resident_blocks: Int, ctx: DeviceContext
) raises -> Krea2ResidentInt8:
    """Rebuild the DEVICE-resident int8 store (blocks [0:resident_blocks)) from a
    sidecar (caller already validated staleness). Verbatim H2D copy per tensor →
    byte-identical to the store build_krea2_resident_int8 produced."""
    var st = SafeTensors.open(cache_path)
    var blocks = List[Krea2BlockResidentInt8]()
    for bi in range(resident_blocks):
        var p = _block_prefix(bi)
        var w8 = List[ArcPointer[Tensor]]()
        var scale = List[ArcPointer[Tensor]]()
        for ki in range(_I8_KEYS):
            w8.append(ArcPointer(_load_raw_h2d(st, p + String("w8.") + String(ki), ctx)))
            scale.append(ArcPointer(_load_raw_h2d(st, p + String("scale.") + String(ki), ctx)))
        blocks.append(Krea2BlockResidentInt8(
            w8^, scale^,
            ArcPointer(_load_raw_h2d(st, p + String("qnorm_scale"), ctx)),
            ArcPointer(_load_raw_h2d(st, p + String("knorm_scale"), ctx)),
            ArcPointer(_load_raw_h2d(st, p + String("prenorm_scale"), ctx)),
            ArcPointer(_load_raw_h2d(st, p + String("postnorm_scale"), ctx)),
            ArcPointer(_load_raw_h2d(st, p + String("mod_lin"), ctx)),
        ))
        if (bi + 1) % 7 == 0 or bi + 1 == resident_blocks:
            print("[int8cache] loaded", bi + 1, "/", resident_blocks,
                  "resident blocks from sidecar")
    return Krea2ResidentInt8(blocks^)


def load_krea2_int8_cache_host(
    cache_path: String, nblocks: Int, resident_blocks: Int, ctx: DeviceContext
) raises -> Krea2HostInt8Inf:
    """Rebuild the PINNED-HOST int8 store (blocks [resident_blocks:nblocks)) from
    a sidecar. The int8 payloads go mmap→pinned-host by straight memcpy — NO
    device round-trip (device cost = the scalar scales + 5 small tensors only),
    exactly the residency shape build_krea2_host_int8_inf produces."""
    var st = SafeTensors.open(cache_path)
    var blocks = List[Krea2BlockHostInt8Inf]()
    for bi in range(resident_blocks, nblocks):
        var p = _block_prefix(bi)
        var w8_h = List[_HArc]()
        var w8_nbytes = List[Int]()
        var w8_shape = List[List[Int]]()
        var scale = List[ArcPointer[Tensor]]()
        for ki in range(_I8_KEYS):
            var name = p + String("w8.") + String(ki)
            var info = st.tensor_info(name)
            var bytes = st.tensor_bytes(name)
            var n = len(bytes)
            var bh = ctx.enqueue_create_host_buffer[DType.uint8](n)
            var dst = BytePtr(unsafe_from_address=Int(bh.unsafe_ptr()))
            var srcp = BytePtr(unsafe_from_address=Int(bytes.unsafe_ptr()))
            _ = sys_memcpy(dst, srcp, n)
            w8_h.append(_HArc(bh^))
            w8_nbytes.append(n)
            w8_shape.append(info.shape.copy())
            scale.append(ArcPointer(_load_raw_h2d(st, p + String("scale.") + String(ki), ctx)))
        blocks.append(Krea2BlockHostInt8Inf(
            w8_h^, w8_nbytes^, w8_shape^, scale^,
            ArcPointer(_load_raw_h2d(st, p + String("qnorm_scale"), ctx)),
            ArcPointer(_load_raw_h2d(st, p + String("knorm_scale"), ctx)),
            ArcPointer(_load_raw_h2d(st, p + String("prenorm_scale"), ctx)),
            ArcPointer(_load_raw_h2d(st, p + String("postnorm_scale"), ctx)),
            ArcPointer(_load_raw_h2d(st, p + String("mod_lin"), ctx)),
        ))
        if (bi + 1 - resident_blocks) % 7 == 0 or bi + 1 == nblocks:
            print("[int8cache] loaded", bi + 1 - resident_blocks, "/",
                  nblocks - resident_blocks, "pinned-host blocks from sidecar")
    return Krea2HostInt8Inf(resident_blocks, blocks^)


def load_krea2_int8_cache_shared(
    cache_path: String, ctx: DeviceContext
) raises -> Krea2SharedResident:
    """Rebuild the SHARED (non-block) resident store from a sidecar — verbatim
    H2D per tensor, byte-identical to build_krea2_shared_resident's output,
    without opening the bf16 checkpoint (whose tproj alone is 906MB F32)."""
    var st = SafeTensors.open(cache_path)
    var t = List[ArcPointer[Tensor]]()
    for i in range(_SHARED_T):
        t.append(ArcPointer(
            _load_raw_h2d(st, String("shared.t.") + String(i), ctx)
        ))
    var txtf = List[Krea2TextFusionWeights]()
    for bj in range(_SHARED_TXTF):
        var p = String("shared.txtf.") + String(bj) + String(".")
        txtf.append(Krea2TextFusionWeights(
            ArcPointer(_load_raw_h2d(st, p + String("prenorm"), ctx)),
            ArcPointer(_load_raw_h2d(st, p + String("postnorm"), ctx)),
            ArcPointer(_load_raw_h2d(st, p + String("wq"), ctx)),
            ArcPointer(_load_raw_h2d(st, p + String("wk"), ctx)),
            ArcPointer(_load_raw_h2d(st, p + String("wv"), ctx)),
            ArcPointer(_load_raw_h2d(st, p + String("gate_w"), ctx)),
            ArcPointer(_load_raw_h2d(st, p + String("wo"), ctx)),
            ArcPointer(_load_raw_h2d(st, p + String("qnorm"), ctx)),
            ArcPointer(_load_raw_h2d(st, p + String("knorm"), ctx)),
            ArcPointer(_load_raw_h2d(st, p + String("mlp_gate"), ctx)),
            ArcPointer(_load_raw_h2d(st, p + String("mlp_up"), ctx)),
            ArcPointer(_load_raw_h2d(st, p + String("mlp_down"), ctx)),
        ))
    print("[int8cache] loaded shared (non-block) weights from sidecar")
    return Krea2SharedResident(t^, txtf^)


# ── SAVER ─────────────────────────────────────────────────────────────────────
# Streaming mixed-source safetensors writer: payloads come from device tensors
# (D2H'd one at a time), pinned host buffers (pwritten direct), or inline host
# bytes (the __meta__ fields). Same on-disk format as io/safetensors_writer
# (8-byte LE header_len + compact JSON header + contiguous payloads) and the
# same tmp+rename atomicity — a crash mid-write never leaves a corrupt sidecar.

struct _SaveSet(Movable):
    var names: List[String]
    var dts: List[String]        # STDtype.name() strings ("I8","F32","BF16",...)
    var shapes: List[List[Int]]
    var nbytes: List[Int]
    var kinds: List[Int]         # 0 = device tensor, 1 = pinned host, 2 = inline
    var idxs: List[Int]          # index into devs / hosts / inln per kind
    var devs: List[ArcPointer[Tensor]]
    var hosts: List[_HArc]
    var inln: List[List[UInt8]]

    def __init__(out self):
        self.names = List[String]()
        self.dts = List[String]()
        self.shapes = List[List[Int]]()
        self.nbytes = List[Int]()
        self.kinds = List[Int]()
        self.idxs = List[Int]()
        self.devs = List[ArcPointer[Tensor]]()
        self.hosts = List[_HArc]()
        self.inln = List[List[UInt8]]()

    def add_dev(mut self, name: String, var t: ArcPointer[Tensor]):
        self.names.append(name)
        self.dts.append(t[].dtype().name())
        self.shapes.append(t[].shape())
        self.nbytes.append(t[].nbytes())
        self.kinds.append(0)
        self.idxs.append(len(self.devs))
        self.devs.append(t^)

    def add_host(mut self, name: String, var h: _HArc, nbytes: Int, var shape: List[Int]):
        self.names.append(name)
        self.dts.append(String("I8"))
        self.shapes.append(shape^)
        self.nbytes.append(nbytes)
        self.kinds.append(1)
        self.idxs.append(len(self.hosts))
        self.hosts.append(h^)

    def add_inline(mut self, name: String, dt: String, var shape: List[Int], var payload: List[UInt8]):
        self.names.append(name)
        self.dts.append(dt)
        self.shapes.append(shape^)
        self.nbytes.append(len(payload))
        self.kinds.append(2)
        self.idxs.append(len(self.inln))
        self.inln.append(payload^)


def _i64_bytes(v: Int) -> List[UInt8]:
    """8 little-endian bytes of `v` (the [1] I64 meta payload)."""
    var out = List[UInt8]()
    var x = v
    for _ in range(8):
        out.append(UInt8(x & 0xFF))
        x >>= 8
    return out^


def _str_bytes(s: String) -> List[UInt8]:
    var out = List[UInt8]()
    var src = s.as_bytes()
    for i in range(len(src)):
        out.append(src[i])
    return out^


def _write_sidecar(s: _SaveSet, cache_path: String, ctx: DeviceContext) raises:
    """Write the collected items as one safetensors file (atomic tmp+rename).
    Sidecar names contain only [A-Za-z0-9._] so no JSON escaping is needed."""
    var n = len(s.names)
    var offsets = List[Int]()
    var running = 0
    for i in range(n):
        offsets.append(running)
        running += s.nbytes[i]
    offsets.append(running)

    # Compact JSON header, insertion order — same schema io/safetensors.mojo reads.
    var hdr = String("{")
    for i in range(n):
        if i > 0:
            hdr += ","
        hdr += String('"') + s.names[i] + String('":{"dtype":"') + s.dts[i]
        hdr += String('","shape":[')
        for d in range(len(s.shapes[i])):
            if d > 0:
                hdr += ","
            hdr += String(s.shapes[i][d])
        hdr += String('],"data_offsets":[') + String(offsets[i]) + String(",")
        hdr += String(offsets[i + 1]) + String("]}")
    hdr += "}"
    var hb = hdr.as_bytes()
    var header_len = len(hb)

    var tmp_path = cache_path + String(".tmp")
    var fd = sys_open(tmp_path, O_WRONLY | O_CREAT | O_TRUNC, Int32(0o644))
    if fd < 0:
        raise Error(String("[int8cache] failed to open for write: ") + tmp_path)
    try:
        # 1) 8-byte little-endian header length at offset 0.
        var lenbuf = alloc[UInt8](8)
        var hl = header_len
        for i in range(8):
            lenbuf[i] = UInt8(hl & 0xFF)
            hl >>= 8
        _write_all(fd, BytePtr(unsafe_from_address=Int(lenbuf)), 8, 0)
        lenbuf.free()
        # 2) header bytes at offset 8.
        var hbuf = alloc[UInt8](header_len)
        for i in range(header_len):
            hbuf[i] = hb[i]
        _write_all(fd, BytePtr(unsafe_from_address=Int(hbuf)), header_len, 8)
        hbuf.free()
        # 3) payloads at 8 + header_len + offsets[i], one at a time (the device
        #    D2H staging buffer is per-tensor transient — no multi-GB spike).
        var data_offset = 8 + header_len
        for i in range(n):
            var off = data_offset + offsets[i]
            var nb = s.nbytes[i]
            if s.kinds[i] == 0:
                ref t = s.devs[s.idxs[i]]
                var host = ctx.enqueue_create_host_buffer[DType.uint8](nb)
                ctx.enqueue_copy(dst_buf=host, src_buf=t[].buf)
                ctx.synchronize()
                _write_all(fd, BytePtr(unsafe_from_address=Int(host.unsafe_ptr())), nb, off)
            elif s.kinds[i] == 1:
                ref h = s.hosts[s.idxs[i]]
                _write_all(fd, BytePtr(unsafe_from_address=Int(h[].unsafe_ptr())), nb, off)
            else:
                ref pb = s.inln[s.idxs[i]]
                var buf = alloc[UInt8](nb)
                for j in range(nb):
                    buf[j] = pb[j]
                _write_all(fd, BytePtr(unsafe_from_address=Int(buf)), nb, off)
                buf.free()
    except e:
        _ = sys_close(fd)
        _ = sys_remove(tmp_path)
        raise Error(String("[int8cache] sidecar write failed: ") + String(e))

    _ = sys_close(fd)
    # Atomic publish: only after rename does `cache_path` reflect the sidecar.
    if sys_rename(tmp_path, cache_path) != 0:
        _ = sys_remove(tmp_path)
        raise Error(
            String("[int8cache] atomic rename ") + tmp_path + " -> " + cache_path
            + " failed (different filesystem? out of space?)"
        )


def save_krea2_int8_cache(
    resident: Optional[Krea2ResidentInt8],
    host: Optional[Krea2HostInt8Inf],
    shared: Krea2SharedResident,
    checkpoint: String,
    cache_path: String,
    nblocks: Int,
    ctx: DeviceContext,
) raises:
    """Write the freshly-quantized hybrid store (resident blocks [0:K) + pinned-
    host blocks [K:nblocks)) + the shared (non-block) resident weights to the
    sidecar. Embeds the __meta__ staleness fields keyed on the current
    checkpoint's stat(2). Raises if the split doesn't cover exactly [0:nblocks)
    or the checkpoint can't be stat'd (caller treats a write failure as
    non-fatal: just no cache this launch)."""
    var sm = _stat_size_mtime(checkpoint)
    if sm[0] < 0:
        raise Error(
            String("[int8cache] cannot stat checkpoint for staleness meta: ")
            + checkpoint
        )
    var n_res = 0
    if Bool(resident):
        n_res = len(resident.value().blocks)
    var n_host = 0
    if Bool(host):
        n_host = len(host.value().blocks)
        if host.value().first != n_res:
            raise Error(
                String("save_krea2_int8_cache: host store first=")
                + String(host.value().first) + " != resident count " + String(n_res)
            )
    if n_res + n_host != nblocks:
        raise Error(
            String("save_krea2_int8_cache: resident ") + String(n_res)
            + " + host " + String(n_host) + " != nblocks " + String(nblocks)
        )

    var s = _SaveSet()
    # __meta__ staleness guard (weights-only: no split/LoRA/steps dependence).
    var sh1: List[Int] = [1]
    s.add_inline(String("__meta__.version"), String("I64"), sh1.copy(),
                 _i64_bytes(KREA2_INT8CACHE_VERSION))
    s.add_inline(String("__meta__.nblocks"), String("I64"), sh1.copy(),
                 _i64_bytes(nblocks))
    s.add_inline(String("__meta__.src_size"), String("I64"), sh1.copy(),
                 _i64_bytes(sm[0]))
    s.add_inline(String("__meta__.src_mtime"), String("I64"), sh1.copy(),
                 _i64_bytes(sm[1]))
    var pshape: List[Int] = [checkpoint.byte_length()]
    s.add_inline(String("__meta__.src_path"), String("U8"), pshape^,
                 _str_bytes(checkpoint))

    for bi in range(nblocks):
        var p = _block_prefix(bi)
        if bi < n_res:
            ref b = resident.value().blocks[bi]
            if len(b.w8) != _I8_KEYS or len(b.scale) != _I8_KEYS:
                raise Error(
                    String("save_krea2_int8_cache: resident block ") + String(bi)
                    + " has " + String(len(b.w8)) + " w8 / "
                    + String(len(b.scale)) + " scale, expected " + String(_I8_KEYS)
                )
            for ki in range(_I8_KEYS):
                s.add_dev(p + String("w8.") + String(ki), b.w8[ki].copy())
                s.add_dev(p + String("scale.") + String(ki), b.scale[ki].copy())
            s.add_dev(p + String("qnorm_scale"), b.qnorm_scale.copy())
            s.add_dev(p + String("knorm_scale"), b.knorm_scale.copy())
            s.add_dev(p + String("prenorm_scale"), b.prenorm_scale.copy())
            s.add_dev(p + String("postnorm_scale"), b.postnorm_scale.copy())
            s.add_dev(p + String("mod_lin"), b.mod_lin.copy())
        else:
            ref hb = host.value().blocks[bi - n_res]
            if len(hb.w8_h) != _I8_KEYS or len(hb.scale) != _I8_KEYS:
                raise Error(
                    String("save_krea2_int8_cache: host block ") + String(bi)
                    + " has " + String(len(hb.w8_h)) + " w8 / "
                    + String(len(hb.scale)) + " scale, expected " + String(_I8_KEYS)
                )
            for ki in range(_I8_KEYS):
                s.add_host(p + String("w8.") + String(ki), hb.w8_h[ki].copy(),
                           hb.w8_nbytes[ki], hb.w8_shape[ki].copy())
                s.add_dev(p + String("scale.") + String(ki), hb.scale[ki].copy())
            s.add_dev(p + String("qnorm_scale"), hb.qnorm_scale.copy())
            s.add_dev(p + String("knorm_scale"), hb.knorm_scale.copy())
            s.add_dev(p + String("prenorm_scale"), hb.prenorm_scale.copy())
            s.add_dev(p + String("postnorm_scale"), hb.postnorm_scale.copy())
            s.add_dev(p + String("mod_lin"), hb.mod_lin.copy())

    # SHARED (non-block) weights — the exact device bytes the per-forward path
    # would have produced (build_krea2_shared_resident's _wb/_scale loaders).
    if len(shared.t) != _SHARED_T or len(shared.txtf) != _SHARED_TXTF:
        raise Error(
            String("save_krea2_int8_cache: shared store has ")
            + String(len(shared.t)) + " t slots / " + String(len(shared.txtf))
            + " txtf bundles, expected " + String(_SHARED_T) + " / "
            + String(_SHARED_TXTF)
        )
    for i in range(_SHARED_T):
        s.add_dev(String("shared.t.") + String(i), shared.t[i].copy())
    for bj in range(_SHARED_TXTF):
        var tp = String("shared.txtf.") + String(bj) + String(".")
        ref tw = shared.txtf[bj]
        s.add_dev(tp + String("prenorm"), tw.prenorm.copy())
        s.add_dev(tp + String("postnorm"), tw.postnorm.copy())
        s.add_dev(tp + String("wq"), tw.wq.copy())
        s.add_dev(tp + String("wk"), tw.wk.copy())
        s.add_dev(tp + String("wv"), tw.wv.copy())
        s.add_dev(tp + String("gate_w"), tw.gate_w.copy())
        s.add_dev(tp + String("wo"), tw.wo.copy())
        s.add_dev(tp + String("qnorm"), tw.qnorm.copy())
        s.add_dev(tp + String("knorm"), tw.knorm.copy())
        s.add_dev(tp + String("mlp_gate"), tw.mlp_gate.copy())
        s.add_dev(tp + String("mlp_up"), tw.mlp_up.copy())
        s.add_dev(tp + String("mlp_down"), tw.mlp_down.copy())

    _write_sidecar(s, cache_path, ctx)
