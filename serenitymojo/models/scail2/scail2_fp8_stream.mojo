# Persistent, fail-closed SCAIL-2 14B per-block E4M3 stream.
#
# Input is the canonical safetensors schema produced from the pinned official
# checkpoint mapping. Each of 40 blocks has exactly 32 source tensors: 12 large
# 2-D matrices become E4M3 plus F32 row scales, and 20 tensors remain exact
# BF16. The 27 shared BF16 tensors are published last as the completion marker.

from std.collections import Dict, List
from max.gpu.host import DeviceContext
from std.memory import ArcPointer, alloc

from serenitymojo.io.dtype import STDtype
from serenitymojo.components.artifacts import shell_quote
from serenitymojo.io.ffi import (
    O_RDONLY,
    BytePtr,
    file_size,
    sys_close,
    sys_memcpy,
    sys_mkdirs,
    sys_open,
    sys_pread,
    sys_system,
)
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.tensor_view import TensorView, from_parts
from serenitymojo.ops.fp8 import fp8_e4m3_dequant_perrow_to_bf16
from serenitymojo.ops.fp8_quant import fp8_e4m3_encode_perrow, fp8_e4m3_rowscale
from serenitymojo.tensor import Tensor


comptime TArc = ArcPointer[Tensor]
comptime SCAIL2_BLOCKS = 40
comptime SCAIL2_BLOCK_SOURCE_TENSORS = 32
comptime SCAIL2_BLOCK_FP8_MATRICES = 12
comptime SCAIL2_BLOCK_EXACT_TENSORS = 20
comptime SCAIL2_SHARED_TENSORS = 27
comptime SCAIL2_FP8_SCALE_SUFFIX = ".__fp8_scale"
comptime SCAIL2_SOURCE_PROVENANCE = (
    "checkpoint_sha256="
    "d6c73e94c57eb36e6351c800d1228e41ed7e45db1ccf410dd875bcfdd2945e7f\n"
    "model_revision=150cc0ca4e98e50e60b9295dacde39442fdccab2\n"
    "source_commit=5cfe1b8daac8bcb22ee19794e6c04f1bf5de6ac5\n"
)


def _path_exists(path: String) -> Bool:
    var fd = sys_open(path, O_RDONLY)
    if fd < 0:
        return False
    _ = sys_close(fd)
    return True


def _valid_pinned_provenance(path: String) -> Bool:
    var fd = sys_open(path, O_RDONLY)
    if fd < 0:
        return False
    var expected = SCAIL2_SOURCE_PROVENANCE
    var nbytes = file_size(fd)
    if nbytes != expected.byte_length():
        _ = sys_close(fd)
        return False
    var data = alloc[UInt8](nbytes)
    var done = 0
    while done < nbytes:
        var got = sys_pread(fd, data + done, nbytes - done, done)
        if got <= 0:
            break
        done += got
    _ = sys_close(fd)
    if done != nbytes:
        data.free()
        return False
    var expected_bytes = expected.as_bytes()
    for i in range(nbytes):
        if data[i] != expected_bytes[i]:
            data.free()
            return False
    data.free()
    return True


def _bind_cache_provenance(source_path: String, cache_dir: String) raises:
    var source = source_path + String("/source_provenance.sha256")
    var cache = cache_dir + String("/source_provenance.sha256")
    if not _valid_pinned_provenance(source):
        raise Error("SCAIL-2 canonical source provenance is missing or unpinned")
    if _path_exists(cache):
        if not _valid_pinned_provenance(cache):
            raise Error("SCAIL-2 FP8 cache provenance is not the pinned model")
        if sys_system(
            String("cmp -s -- ") + shell_quote(source) + String(" ")
            + shell_quote(cache)
        ) != 0:
            raise Error("refusing to mix SCAIL-2 FP8 blocks from another source")
        return
    var has_cache_artifacts = _path_exists(cache_dir + String("/shared.safetensors"))
    for bi in range(SCAIL2_BLOCKS):
        if _path_exists(scail2_block_cache_path(cache_dir, bi)):
            has_cache_artifacts = True
    if has_cache_artifacts:
        raise Error(
            "refusing to resume unbound SCAIL-2 FP8 cache; use a fresh cache directory"
        )
    var tmp = cache + String(".tmp")
    if sys_system(
        String("cp -- ") + shell_quote(source) + String(" ") + shell_quote(tmp)
    ) != 0:
        raise Error("failed to stage SCAIL-2 cache provenance")
    if sys_system(
        String("mv -- ") + shell_quote(tmp) + String(" ") + shell_quote(cache)
    ) != 0:
        raise Error("failed to publish SCAIL-2 cache provenance")


def _substr(s: String, start: Int, end: Int) -> String:
    var out = String("")
    var i = 0
    for ch in s.codepoint_slices():
        if i >= start and i < end:
            out += String(ch)
        i += 1
    return out^


def scail2_block_prefix(index: Int) -> String:
    return String("blocks.") + String(index) + String(".")


def scail2_block_cache_path(cache_dir: String, index: Int) -> String:
    var digits = String(index)
    if index < 10:
        digits = String("0") + digits
    return cache_dir + String("/block_") + digits + String(".safetensors")


def _is_large_block_suffix(s: String) -> Bool:
    return (
        s == "self_attn.q.weight" or s == "self_attn.k.weight"
        or s == "self_attn.v.weight" or s == "self_attn.o.weight"
        or s == "cross_attn.q.weight" or s == "cross_attn.k.weight"
        or s == "cross_attn.v.weight" or s == "cross_attn.o.weight"
        or s == "cross_attn.k_img.weight" or s == "cross_attn.v_img.weight"
        or s == "ffn.0.weight" or s == "ffn.2.weight"
    )


def _is_exact_block_suffix(s: String) -> Bool:
    return (
        s == "modulation"
        or s == "self_attn.q.bias" or s == "self_attn.k.bias"
        or s == "self_attn.v.bias" or s == "self_attn.o.bias"
        or s == "self_attn.norm_q.weight" or s == "self_attn.norm_k.weight"
        or s == "cross_attn.q.bias" or s == "cross_attn.k.bias"
        or s == "cross_attn.v.bias" or s == "cross_attn.o.bias"
        or s == "cross_attn.k_img.bias" or s == "cross_attn.v_img.bias"
        or s == "cross_attn.norm_q.weight"
        or s == "cross_attn.norm_k.weight"
        or s == "cross_attn.norm_k_img.weight"
        or s == "norm3.weight" or s == "norm3.bias"
        or s == "ffn.0.bias" or s == "ffn.2.bias"
    )


def _is_shared_name(s: String) -> Bool:
    return (
        s == "patch_embedding.weight" or s == "patch_embedding.bias"
        or s == "patch_embedding_pose.weight" or s == "patch_embedding_pose.bias"
        or s == "patch_embedding_mask.weight" or s == "patch_embedding_mask.bias"
        or s == "text_embedding.0.weight" or s == "text_embedding.0.bias"
        or s == "text_embedding.2.weight" or s == "text_embedding.2.bias"
        or s == "time_embedding.0.weight" or s == "time_embedding.0.bias"
        or s == "time_embedding.2.weight" or s == "time_embedding.2.bias"
        or s == "time_projection.1.weight" or s == "time_projection.1.bias"
        or s == "head.modulation" or s == "head.head.weight"
        or s == "head.head.bias"
        or s == "img_emb.proj.0.weight" or s == "img_emb.proj.0.bias"
        or s == "img_emb.proj.1.weight" or s == "img_emb.proj.1.bias"
        or s == "img_emb.proj.3.weight" or s == "img_emb.proj.3.bias"
        or s == "img_emb.proj.4.weight" or s == "img_emb.proj.4.bias"
    )


def _shape1(shape: List[Int], a: Int) -> Bool:
    return len(shape) == 1 and shape[0] == a


def _shape2(shape: List[Int], a: Int, b: Int) -> Bool:
    return len(shape) == 2 and shape[0] == a and shape[1] == b


def _shape5(
    shape: List[Int], a: Int, b: Int, c: Int, d: Int, e: Int
) -> Bool:
    return (
        len(shape) == 5 and shape[0] == a and shape[1] == b
        and shape[2] == c and shape[3] == d and shape[4] == e
    )


def _valid_block_shape(suffix: String, shape: List[Int]) -> Bool:
    var base = suffix
    var scale = False
    if suffix.endswith(SCAIL2_FP8_SCALE_SUFFIX):
        base = _substr(
            suffix, 0, len(suffix) - len(SCAIL2_FP8_SCALE_SUFFIX)
        )
        scale = True
    if base == String("ffn.0.weight"):
        return _shape1(shape, 13824) if scale else _shape2(shape, 13824, 5120)
    if base == String("ffn.2.weight"):
        return _shape1(shape, 5120) if scale else _shape2(shape, 5120, 13824)
    if _is_large_block_suffix(base):
        return _shape1(shape, 5120) if scale else _shape2(shape, 5120, 5120)
    if base == String("modulation"):
        return len(shape) == 3 and shape[0] == 1 and shape[1] == 6 and shape[2] == 5120
    if base == String("ffn.0.bias"):
        return _shape1(shape, 13824)
    if _is_exact_block_suffix(base):
        return _shape1(shape, 5120)
    return False


def _valid_shared_shape(name: String, shape: List[Int]) -> Bool:
    if name == String("patch_embedding.weight") or name == String("patch_embedding_pose.weight"):
        return _shape5(shape, 5120, 20, 1, 2, 2)
    if name == String("patch_embedding_mask.weight"):
        return _shape5(shape, 5120, 28, 1, 2, 2)
    if name == String("text_embedding.0.weight"):
        return _shape2(shape, 5120, 4096)
    if name == String("text_embedding.2.weight"):
        return _shape2(shape, 5120, 5120)
    if name == String("time_embedding.0.weight"):
        return _shape2(shape, 5120, 256)
    if name == String("time_embedding.2.weight"):
        return _shape2(shape, 5120, 5120)
    if name == String("time_projection.1.weight"):
        return _shape2(shape, 30720, 5120)
    if name == String("time_projection.1.bias"):
        return _shape1(shape, 30720)
    if name == String("head.modulation"):
        return len(shape) == 3 and shape[0] == 1 and shape[1] == 2 and shape[2] == 5120
    if name == String("head.head.weight"):
        return _shape2(shape, 64, 5120)
    if name == String("head.head.bias"):
        return _shape1(shape, 64)
    if name == String("img_emb.proj.0.weight") or name == String("img_emb.proj.0.bias"):
        return _shape1(shape, 1280)
    if name == String("img_emb.proj.1.weight"):
        return _shape2(shape, 1280, 1280)
    if name == String("img_emb.proj.1.bias"):
        return _shape1(shape, 1280)
    if name == String("img_emb.proj.3.weight"):
        return _shape2(shape, 5120, 1280)
    if (
        name == String("img_emb.proj.3.bias")
        or name == String("img_emb.proj.4.weight")
        or name == String("img_emb.proj.4.bias")
    ):
        return _shape1(shape, 5120)
    if _is_shared_name(name):
        return _shape1(shape, 5120)
    return False


def _checksum_path(path: String) -> String:
    return path + String(".sha256")


def validate_scail2_artifact_checksum(path: String) -> Bool:
    var checksum = _checksum_path(path)
    if not _path_exists(path) or not _path_exists(checksum):
        return False
    return sys_system(
        String("expected=$(cut -d ' ' -f 1 -- ") + shell_quote(checksum)
        + String(") && actual=$(sha256sum -- ") + shell_quote(path)
        + String(" | cut -d ' ' -f 1) && test \"$actual\" = \"$expected\"")
    ) == 0


def validate_scail2_artifact_checksum_metadata(path: String) -> Bool:
    """Runtime check for an installer-verified immutable cache artifact.

    Cache creation/resume uses `validate_scail2_artifact_checksum` and hashes
    the payload. Generation validates that both files exist, that the checksum
    sidecar contains exactly one SHA-256 digest, and that the safetensors schema
    below still matches. Re-hashing every 400 MB block on every launch made the
    nominally metadata-only preflight scan the full cache twice.
    """
    var checksum = _checksum_path(path)
    if not _path_exists(path) or not _path_exists(checksum):
        return False
    return sys_system(
        String("test $(wc -l < ") + shell_quote(checksum)
        + String(") -eq 1 && grep -Eq '^[0-9a-fA-F]{64}([[:space:]].*)?$' -- ")
        + shell_quote(checksum)
    ) == 0


def _publish_artifact_checksum(path: String) raises:
    var checksum = _checksum_path(path)
    var tmp = checksum + String(".tmp")
    if sys_system(
        String("sha256sum -- ") + shell_quote(path)
        + String(" | cut -d ' ' -f 1 > ") + shell_quote(tmp)
        + String(" && mv -- ") + shell_quote(tmp) + String(" ")
        + shell_quote(checksum)
    ) != 0:
        raise Error(String("failed to publish SCAIL-2 artifact checksum: ") + path)


def _valid_block_cache_file(path: String, index: Int) raises -> Bool:
    if not validate_scail2_artifact_checksum(path):
        return False
    var st: ShardedSafeTensors
    try:
        st = ShardedSafeTensors.open(path)
    except:
        return False
    var prefix = scail2_block_prefix(index)
    var fp8 = 0
    var scales = 0
    var exact = 0
    for ref raw_name in st.names():
        var name = String(raw_name)
        if not name.startswith(prefix):
            return False
        var suffix = _substr(name, len(prefix), len(name))
        var tv = st.tensor_view(name)
        if not _valid_block_shape(suffix, tv.shape):
            return False
        if suffix.endswith(SCAIL2_FP8_SCALE_SUFFIX):
            var base = _substr(
                suffix, 0, len(suffix) - len(SCAIL2_FP8_SCALE_SUFFIX)
            )
            if not _is_large_block_suffix(base) or tv.dtype != STDtype.F32:
                return False
            scales += 1
        elif _is_large_block_suffix(suffix):
            if tv.dtype != STDtype.F8_E4M3:
                return False
            fp8 += 1
        elif _is_exact_block_suffix(suffix):
            if tv.dtype != STDtype.BF16:
                return False
            exact += 1
        else:
            return False
    return (
        fp8 == SCAIL2_BLOCK_FP8_MATRICES
        and scales == SCAIL2_BLOCK_FP8_MATRICES
        and exact == SCAIL2_BLOCK_EXACT_TENSORS
    )


def _valid_block_cache_metadata(path: String, index: Int) raises -> Bool:
    if not validate_scail2_artifact_checksum_metadata(path):
        return False
    var st: ShardedSafeTensors
    try:
        st = ShardedSafeTensors.open(path)
    except:
        return False
    var prefix = scail2_block_prefix(index)
    var fp8 = 0
    var scales = 0
    var exact = 0
    for ref raw_name in st.names():
        var name = String(raw_name)
        if not name.startswith(prefix):
            return False
        var suffix = _substr(name, len(prefix), len(name))
        var tv = st.tensor_view(name)
        if not _valid_block_shape(suffix, tv.shape):
            return False
        if suffix.endswith(SCAIL2_FP8_SCALE_SUFFIX):
            var base = _substr(
                suffix, 0, len(suffix) - len(SCAIL2_FP8_SCALE_SUFFIX)
            )
            if not _is_large_block_suffix(base) or tv.dtype != STDtype.F32:
                return False
            scales += 1
        elif _is_large_block_suffix(suffix):
            if tv.dtype != STDtype.F8_E4M3:
                return False
            fp8 += 1
        elif _is_exact_block_suffix(suffix):
            if tv.dtype != STDtype.BF16:
                return False
            exact += 1
        else:
            return False
    return (
        fp8 == SCAIL2_BLOCK_FP8_MATRICES
        and scales == SCAIL2_BLOCK_FP8_MATRICES
        and exact == SCAIL2_BLOCK_EXACT_TENSORS
    )


def prepare_scail2_14b_fp8_cache(
    source_path: String, cache_dir: String, ctx: DeviceContext
) raises:
    """Create/resume the bounded SCAIL-2 stream from canonical safetensors."""
    if sys_mkdirs(cache_dir) != 0:
        pass
    _bind_cache_provenance(source_path, cache_dir)
    var source = ShardedSafeTensors.open(source_path)
    var source_names = source.names()

    for bi in range(SCAIL2_BLOCKS):
        var block_path = scail2_block_cache_path(cache_dir, bi)
        var block_valid = False
        try:
            block_valid = _valid_block_cache_file(block_path, bi)
        except:
            block_valid = False
        if block_valid:
            print("[scail2-cache] resume block", bi + 1, "/", SCAIL2_BLOCKS)
            continue
        var prefix = scail2_block_prefix(bi)
        var names = List[String]()
        var tensors = List[TArc]()
        var source_count = 0
        var fp8_count = 0
        var exact_count = 0
        for ref raw_name in source_names:
            var name = String(raw_name)
            if not name.startswith(prefix):
                continue
            var suffix = _substr(name, len(prefix), len(name))
            var tv = source.tensor_view(name)
            source_count += 1
            if _is_large_block_suffix(suffix):
                if len(tv.shape) != 2:
                    raise Error(String("SCAIL-2 large tensor is not rank 2: ") + name)
                var bf16 = Tensor.from_view_as_bf16(tv, ctx)
                var scale = fp8_e4m3_rowscale(bf16, ctx)
                var bytes = fp8_e4m3_encode_perrow(bf16, scale, ctx)
                ctx.synchronize()
                names.append(name)
                tensors.append(TArc(bytes^))
                names.append(name + String(SCAIL2_FP8_SCALE_SUFFIX))
                tensors.append(TArc(scale^))
                fp8_count += 1
            elif _is_exact_block_suffix(suffix):
                names.append(name)
                tensors.append(TArc(Tensor.from_view_as_bf16(tv, ctx)))
                exact_count += 1
            else:
                raise Error(String("SCAIL-2 unsupported block tensor: ") + name)
        if source_count != SCAIL2_BLOCK_SOURCE_TENSORS:
            raise Error(
                String("SCAIL-2 block ") + String(bi) + String(" source count ")
                + String(source_count) + String(" != 32")
            )
        if fp8_count != 12 or exact_count != 20:
            raise Error(String("SCAIL-2 block cache schema mismatch at ") + String(bi))
        save_safetensors(names, tensors, block_path, ctx)
        _publish_artifact_checksum(block_path)
        print(
            "[scail2-cache] wrote block", bi + 1, "/", SCAIL2_BLOCKS,
            "fp8=", fp8_count, "exact_bf16=", exact_count,
        )

    # Publish the complete shared schema last. Unknown or missing names fail.
    var shared_names = List[String]()
    var shared_tensors = List[TArc]()
    for ref raw_name in source_names:
        var name = String(raw_name)
        if name.startswith("blocks."):
            continue
        if not _is_shared_name(name):
            raise Error(String("SCAIL-2 unsupported shared tensor: ") + name)
        shared_names.append(name)
        shared_tensors.append(TArc(Tensor.from_view_as_bf16(source.tensor_view(name), ctx)))
    if len(shared_names) != SCAIL2_SHARED_TENSORS:
        raise Error(
            String("SCAIL-2 shared tensor count ") + String(len(shared_names))
            + String(" != 27")
        )
    save_safetensors(
        shared_names,
        shared_tensors,
        cache_dir + String("/shared.safetensors"),
        ctx,
    )
    _publish_artifact_checksum(cache_dir + String("/shared.safetensors"))
    print("[scail2-cache] GATE cache complete:", cache_dir)


def validate_scail2_14b_fp8_cache(cache_dir: String) raises -> Bool:
    if not _valid_pinned_provenance(
        cache_dir + String("/source_provenance.sha256")
    ):
        return False
    for bi in range(SCAIL2_BLOCKS):
        if not _valid_block_cache_metadata(
            scail2_block_cache_path(cache_dir, bi), bi
        ):
            return False
    var shared_path = cache_dir + String("/shared.safetensors")
    if not validate_scail2_artifact_checksum_metadata(shared_path):
        return False
    var st: ShardedSafeTensors
    try:
        st = ShardedSafeTensors.open(shared_path)
    except:
        return False
    var shared = 0
    for ref raw_name in st.names():
        var name = String(raw_name)
        var tv = st.tensor_view(name)
        if (
            not _is_shared_name(name) or tv.dtype != STDtype.BF16
            or not _valid_shared_shape(name, tv.shape)
        ):
            return False
        shared += 1
    return shared == SCAIL2_SHARED_TENSORS


# ═══════════════════════════════════════════════════════════════════════════════
# HOST-RESIDENT FP8 BASE (speed): copy each block's raw cache bytes into OWNED
# host RAM ONCE (preload), then serve `load_block_bf16` from that resident store
# with NO per-step `ShardedSafeTensors.open` / mmap fault / disk read. The 40
# blocks are 15.4 GB of fp8 (E4M3 matrices + F32 row-scale sidecars + the 20
# exact-BF16 per-block tensors) — fits comfortably in the 52 GB free host RAM.
#
# CORRECTNESS: the resident bytes are a verbatim memcpy of the SAME cache bytes
# the disk path reads, reconstructed into an identical `TensorView` via
# `from_parts` and fed through the SAME `Tensor.from_view_raw` / `Tensor.from_view`
# + `fp8_e4m3_dequant_perrow_to_bf16`. The produced BF16 weights are therefore
# BIT-IDENTICAL to the disk path (same bytes → same H2D copy → same dequant).
# ═══════════════════════════════════════════════════════════════════════════════
def _own_bytes[
    mut: Bool, //, origin: Origin[mut=mut]
](tv: TensorView[origin]) raises -> List[UInt8]:
    """Verbatim owned-host-RAM copy of a TensorView's raw bytes (bulk memcpy).

    Copying OUT of the mmap into anonymous heap RAM is what makes the per-step
    load deterministic: subsequent H2D uploads read from resident RAM, never
    fault the file back from disk. Uses the same fast `sys_memcpy` path as the
    Tensor H2D staging (a per-byte scalar loop over a 385 MB block would cost
    seconds)."""
    var n = tv.nbytes()
    var out = List[UInt8]()
    out.resize(n, UInt8(0))
    if n > 0:
        var dst = BytePtr(unsafe_from_address=Int(out.unsafe_ptr()))
        var src = BytePtr(unsafe_from_address=Int(tv.data.unsafe_ptr()))
        _ = sys_memcpy(dst, src, n)
    return out^


struct _ResidentBlock(Movable):
    """One block's cache tensors held in owned host RAM (44 entries: 12 fp8
    matrices + 12 F32 row-scale sidecars + 20 exact BF16). Short names (the
    `blocks.{i}.` prefix stripped) with an index for the fp8→scale sidecar
    lookup. Held behind an ArcPointer so the resident List never copies the
    385 MB payloads."""

    var names: List[String]
    var dtypes: List[STDtype]
    var shapes: List[List[Int]]
    var datas: List[List[UInt8]]
    var index: Dict[String, Int]

    def __init__(out self):
        self.names = List[String]()
        self.dtypes = List[STDtype]()
        self.shapes = List[List[Int]]()
        self.datas = List[List[UInt8]]()
        self.index = Dict[String, Int]()

    def add(
        mut self, var name: String, dtype: STDtype,
        var shape: List[Int], var data: List[UInt8],
    ):
        self.index[name] = len(self.names)
        self.names.append(name^)
        self.dtypes.append(dtype)
        self.shapes.append(shape^)
        self.datas.append(data^)

    def nbytes(self) -> Int:
        var total = 0
        for i in range(len(self.datas)):
            total += len(self.datas[i])
        return total


comptime _RArc = ArcPointer[_ResidentBlock]


struct Scail2A14BFP8Stream(Movable):
    var cache_dir: String
    var resident: Bool
    var blocks: List[_RArc]

    @staticmethod
    def open(cache_dir: String) raises -> Scail2A14BFP8Stream:
        if not validate_scail2_14b_fp8_cache(cache_dir):
            raise Error(String("invalid/incomplete SCAIL-2 FP8 cache: ") + cache_dir)
        return Scail2A14BFP8Stream(cache_dir)

    def __init__(out self, var cache_dir: String):
        self.cache_dir = cache_dir^
        self.resident = False
        self.blocks = List[_RArc]()

    # ── HOST-RESIDENT preload (call ONCE before the training step loop) ──────
    def preload_resident(mut self, ctx: DeviceContext) raises:
        """Read all 40 blocks' fp8 cache tensors into owned host RAM once.

        After this, `load_block_bf16` serves from the resident store with no
        disk read / mmap fault per step. Idempotent (a second call is a no-op).
        The exact-bf16 tensors and the fp8 E4M3 matrices + F32 scale sidecars
        are all copied verbatim, so the served BF16 weights are byte-identical
        to the disk path. `ctx` is unused here (host-only copy) but kept for a
        uniform preload signature and a possible future device-resident tier."""
        if self.resident:
            return
        var blocks = List[_RArc]()
        var total_bytes = 0
        for bi in range(SCAIL2_BLOCKS):
            var st = ShardedSafeTensors.open(
                scail2_block_cache_path(self.cache_dir, bi)
            )
            var prefix = scail2_block_prefix(bi)
            var rb = _ResidentBlock()
            for ref raw_name in st.names():
                var name = String(raw_name)
                if not name.startswith(prefix):
                    raise Error(
                        String("SCAIL-2 resident: unexpected tensor ") + name
                    )
                var short_name = _substr(name, len(prefix), len(name))
                var tv = st.tensor_view(name)
                rb.add(short_name, tv.dtype, tv.shape.copy(), _own_bytes(tv))
            total_bytes += rb.nbytes()
            blocks.append(_RArc(rb^))
        self.blocks = blocks^
        self.resident = True
        _ = ctx
        print(
            "[scail2-resident] preloaded", SCAIL2_BLOCKS, "fp8 blocks into host RAM =",
            total_bytes, "bytes (", Float64(total_bytes) / Float64(1 << 30), "GiB)",
        )

    def _load_block_bf16_resident(
        self, index: Int, ctx: DeviceContext
    ) raises -> Dict[String, TArc]:
        """Serve `load_block_bf16` from the resident host store (no disk I/O).
        Same dequant math as the disk path; bit-identical weights."""
        ref rb = self.blocks[index][]
        var out = Dict[String, TArc]()
        for i in range(len(rb.names)):
            var short_name = rb.names[i]
            if short_name.endswith(SCAIL2_FP8_SCALE_SUFFIX):
                continue
            var dtype = rb.dtypes[i]
            if dtype == STDtype.F8_E4M3:
                var w_view = from_parts(dtype, rb.shapes[i].copy(), Span(rb.datas[i]))
                var raw = Tensor.from_view_raw(w_view, ctx)
                var si = rb.index[short_name + String(SCAIL2_FP8_SCALE_SUFFIX)]
                var s_view = from_parts(
                    rb.dtypes[si], rb.shapes[si].copy(), Span(rb.datas[si])
                )
                var scale = Tensor.from_view(s_view, ctx)
                var bf16 = fp8_e4m3_dequant_perrow_to_bf16(raw, scale, ctx)
                out[short_name] = TArc(bf16^)
            elif dtype == STDtype.BF16:
                var t_view = from_parts(dtype, rb.shapes[i].copy(), Span(rb.datas[i]))
                out[short_name] = TArc(Tensor.from_view(t_view, ctx))
            else:
                raise Error(
                    String("unsupported SCAIL-2 resident dtype: ") + dtype.name()
                )
        ctx.synchronize()
        if len(out) != SCAIL2_BLOCK_SOURCE_TENSORS:
            raise Error("SCAIL-2 resident block tensor count mismatch")
        return out^

    def load_shared(self, ctx: DeviceContext) raises -> Dict[String, TArc]:
        var st = ShardedSafeTensors.open(
            self.cache_dir + String("/shared.safetensors")
        )
        var out = Dict[String, TArc]()
        for ref raw_name in st.names():
            var name = String(raw_name)
            var tv = st.tensor_view(name)
            if not _is_shared_name(name) or tv.dtype != STDtype.BF16:
                raise Error(String("invalid SCAIL-2 shared cache tensor: ") + name)
            out[name] = TArc(Tensor.from_view(tv, ctx))
        if len(out) != SCAIL2_SHARED_TENSORS:
            raise Error("SCAIL-2 shared cache tensor count mismatch")
        return out^

    def load_block_bf16(
        self, index: Int, ctx: DeviceContext
    ) raises -> Dict[String, TArc]:
        if index < 0 or index >= SCAIL2_BLOCKS:
            raise Error("SCAIL-2 block index out of range")
        # Resident fast path: serve from owned host RAM (no disk read). Falls
        # through to the disk/mmap path when the base was not preloaded.
        if self.resident:
            return self._load_block_bf16_resident(index, ctx)
        var prefix = scail2_block_prefix(index)
        var st = ShardedSafeTensors.open(
            scail2_block_cache_path(self.cache_dir, index)
        )
        var out = Dict[String, TArc]()
        for ref raw_name in st.names():
            var name = String(raw_name)
            if not name.startswith(prefix) or name.endswith(SCAIL2_FP8_SCALE_SUFFIX):
                continue
            var short_name = _substr(name, len(prefix), len(name))
            var tv = st.tensor_view(name)
            if tv.dtype == STDtype.F8_E4M3:
                var scale_tv = st.tensor_view(name + String(SCAIL2_FP8_SCALE_SUFFIX))
                var raw = Tensor.from_view_raw(tv, ctx)
                var scale = Tensor.from_view(scale_tv, ctx)
                var bf16 = fp8_e4m3_dequant_perrow_to_bf16(raw, scale, ctx)
                out[short_name] = TArc(bf16^)
            elif tv.dtype == STDtype.BF16:
                out[short_name] = TArc(Tensor.from_view(tv, ctx))
            else:
                raise Error(String("unsupported SCAIL-2 cache dtype: ") + tv.dtype.name())
        ctx.synchronize()
        if len(out) != SCAIL2_BLOCK_SOURCE_TENSORS:
            raise Error("SCAIL-2 cached block tensor count mismatch")
        return out^

    def load_block_fp8(
        self, index: Int, ctx: DeviceContext
    ) raises -> Dict[String, TArc]:
        """Load raw E4M3 matrices and their row scales for native FP8 GEMM."""
        if index < 0 or index >= SCAIL2_BLOCKS:
            raise Error("SCAIL-2 block index out of range")
        var prefix = scail2_block_prefix(index)
        var st = ShardedSafeTensors.open(
            scail2_block_cache_path(self.cache_dir, index)
        )
        var out = Dict[String, TArc]()
        var matrix_count = 0
        for ref raw_name in st.names():
            var name = String(raw_name)
            if not name.startswith(prefix) or name.endswith(SCAIL2_FP8_SCALE_SUFFIX):
                continue
            var short_name = _substr(name, len(prefix), len(name))
            var tv = st.tensor_view(name)
            if tv.dtype == STDtype.F8_E4M3:
                var scale_tv = st.tensor_view(name + String(SCAIL2_FP8_SCALE_SUFFIX))
                if scale_tv.dtype != STDtype.F32:
                    raise Error("SCAIL-2 FP8 scale must be F32")
                out[short_name] = TArc(Tensor.from_view_raw(tv, ctx))
                out[short_name + String(SCAIL2_FP8_SCALE_SUFFIX)] = TArc(
                    Tensor.from_view(scale_tv, ctx)
                )
                matrix_count += 1
            elif tv.dtype == STDtype.BF16:
                out[short_name] = TArc(Tensor.from_view(tv, ctx))
            else:
                raise Error(String("unsupported SCAIL-2 cache dtype: ") + tv.dtype.name())
        ctx.synchronize()
        if matrix_count != SCAIL2_BLOCK_FP8_MATRICES:
            raise Error("SCAIL-2 FP8 block matrix count mismatch")
        if len(out) != SCAIL2_BLOCK_SOURCE_TENSORS + matrix_count:
            raise Error("SCAIL-2 raw FP8 block tensor count mismatch")
        return out^
