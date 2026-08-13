# Persistent, per-block E4M3 stream for Wan2.2 A14B experts.
#
# Bernini-R ships two F32 14B experts.  Neither a whole F32/BF16 expert nor both
# expert caches may be device-resident on a 16 GiB card.  This module reuses the
# proven Wan key adapter, Mojo per-row E4M3 quantizer, safetensors writer, and
# one-block-at-a-time offload discipline:
#
#   official F32 shards -> BF16 -> per-row E4M3 (once, Mojo) -> 40 block files
#   cache mmap -> active block E4M3 H2D -> BF16 dequant -> existing Wan block
#
# The cache is persistent and bounded: one block is quantized/written at a time.
# Exact small tensors stay BF16.  A shared.safetensors file contains the 15
# non-block tensors and is written last, so a partial preparation cannot pass
# validate_wan22_a14b_fp8_cache.

from std.collections import Dict, List
from max.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.ffi import sys_mkdirs
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.models.dit.wan22_dit import wan22_canonical_weight_key
from serenitymojo.ops.fp8 import fp8_e4m3_dequant_perrow_to_bf16
from serenitymojo.ops.fp8_quant import fp8_e4m3_encode_perrow, fp8_e4m3_rowscale
from serenitymojo.tensor import Tensor


comptime TArc = ArcPointer[Tensor]
comptime WAN22_A14B_BLOCKS = 40
comptime WAN22_BLOCK_SOURCE_TENSORS = 27
comptime WAN22_BLOCK_FP8_MATRICES = 10
comptime WAN22_BLOCK_EXACT_TENSORS = 17
comptime WAN22_SHARED_TENSORS = 15
comptime WAN22_FP8_SCALE_SUFFIX = ".__fp8_scale"


def _substr(s: String, start: Int, end: Int) -> String:
    var out = String("")
    var i = 0
    for ch in s.codepoint_slices():
        if i >= start and i < end:
            out += String(ch)
        i += 1
    return out^


def _block_prefix(index: Int) -> String:
    return String("blocks.") + String(index) + String(".")


def _block_cache_path(cache_dir: String, index: Int) -> String:
    var digits = String(index)
    if index < 10:
        digits = String("0") + digits
    return cache_dir + String("/block_") + digits + String(".safetensors")


def _valid_block_cache_file(path: String, index: Int) raises -> Bool:
    """Validate one atomically-published block so interrupted builds resume."""
    var st: ShardedSafeTensors
    try:
        st = ShardedSafeTensors.open(path)
    except:
        return False
    var prefix = _block_prefix(index)
    var fp8 = 0
    var scales = 0
    var exact = 0
    for ref raw_name in st.names():
        var name = String(raw_name)
        if not name.startswith(prefix):
            return False
        var tv = st.tensor_view(name)
        if name.endswith(WAN22_FP8_SCALE_SUFFIX):
            if tv.dtype != STDtype.F32:
                return False
            scales += 1
        elif tv.dtype == STDtype.F8_E4M3:
            fp8 += 1
        elif tv.dtype == STDtype.BF16:
            exact += 1
        else:
            return False
    return (
        fp8 == WAN22_BLOCK_FP8_MATRICES
        and scales == WAN22_BLOCK_FP8_MATRICES
        and exact == WAN22_BLOCK_EXACT_TENSORS
    )


def prepare_wan22_a14b_fp8_cache(
    source_dir: String, cache_dir: String, ctx: DeviceContext,
) raises:
    """Build one immutable expert cache with bounded host/device memory."""
    if sys_mkdirs(cache_dir) != 0:
        # sys_mkdirs returns the final mkdir result; EEXIST can be nonzero.
        pass
    var source = ShardedSafeTensors.open(source_dir)
    var source_names = source.names()

    for bi in range(WAN22_A14B_BLOCKS):
        var block_path = _block_cache_path(cache_dir, bi)
        var block_valid = False
        try:
            block_valid = _valid_block_cache_file(block_path, bi)
        except:
            block_valid = False
        if block_valid:
            print(
                "[wan-a14b-cache] resume block", bi + 1, "/",
                WAN22_A14B_BLOCKS, "already valid",
            )
            continue
        var prefix = _block_prefix(bi)
        var names = List[String]()
        var tensors = List[TArc]()
        var source_count = 0
        var fp8_count = 0
        var exact_count = 0
        for ref raw_name in source_names:
            var source_name = String(raw_name)
            if not source_name.startswith(prefix):
                continue
            source_count += 1
            var canonical = wan22_canonical_weight_key(source_name)
            var tv = source.tensor_view(source_name)
            var large_matrix = len(tv.shape) == 2 and tv.numel() >= (1 << 16)
            if large_matrix:
                var bf16 = Tensor.from_view_as_bf16(tv, ctx)
                var scale = fp8_e4m3_rowscale(bf16, ctx)
                var bytes = fp8_e4m3_encode_perrow(bf16, scale, ctx)
                ctx.synchronize()
                names.append(canonical)
                tensors.append(TArc(bytes^))
                names.append(canonical + String(WAN22_FP8_SCALE_SUFFIX))
                tensors.append(TArc(scale^))
                fp8_count += 1
            else:
                names.append(canonical)
                tensors.append(TArc(Tensor.from_view_as_bf16(tv, ctx)))
                exact_count += 1
        if source_count != WAN22_BLOCK_SOURCE_TENSORS:
            raise Error(
                String("Wan A14B cache block ") + String(bi)
                + String(" source count ") + String(source_count)
                + String(" != ") + String(WAN22_BLOCK_SOURCE_TENSORS)
            )
        if (
            fp8_count != WAN22_BLOCK_FP8_MATRICES
            or exact_count != WAN22_BLOCK_EXACT_TENSORS
        ):
            raise Error(
                String("Wan A14B cache block layout mismatch at ") + String(bi)
                + String(": fp8=") + String(fp8_count)
                + String(" exact=") + String(exact_count)
            )
        save_safetensors(names, tensors, block_path, ctx)
        print(
            "[wan-a14b-cache] wrote block", bi + 1, "/", WAN22_A14B_BLOCKS,
            "fp8=", fp8_count, "exact_bf16=", exact_count,
        )

    # Shared file is the completion marker and is written only after all blocks.
    var shared_names = List[String]()
    var shared_tensors = List[TArc]()
    for ref raw_name in source_names:
        var source_name = String(raw_name)
        if source_name.startswith("blocks."):
            continue
        var canonical = wan22_canonical_weight_key(source_name)
        var tv = source.tensor_view(source_name)
        shared_names.append(canonical)
        shared_tensors.append(TArc(Tensor.from_view_as_bf16(tv, ctx)))
    if len(shared_names) != WAN22_SHARED_TENSORS:
        raise Error(
            String("Wan A14B shared tensor count ") + String(len(shared_names))
            + String(" != ") + String(WAN22_SHARED_TENSORS)
        )
    save_safetensors(
        shared_names, shared_tensors, cache_dir + String("/shared.safetensors"), ctx
    )
    print("[wan-a14b-cache] GATE cache complete:", cache_dir)


def validate_wan22_a14b_fp8_cache(cache_dir: String) raises -> Bool:
    """Fail-closed tensor-count/dtype/schema validation for one expert cache.

    The cache intentionally consists of one file per block plus one shared
    file.  Open those files directly: ShardedSafeTensors directory mode only
    recognizes Diffusers index files and must not be used as an implicit
    directory scanner.
    """
    for bi in range(WAN22_A14B_BLOCKS):
        if not _valid_block_cache_file(_block_cache_path(cache_dir, bi), bi):
            return False

    var st: ShardedSafeTensors
    try:
        st = ShardedSafeTensors.open(cache_dir + String("/shared.safetensors"))
    except:
        return False
    var shared = 0
    for ref raw_name in st.names():
        var name = String(raw_name)
        var tv = st.tensor_view(name)
        if name.startswith("blocks.") or tv.dtype != STDtype.BF16:
            return False
        shared += 1
    return shared == WAN22_SHARED_TENSORS


struct Wan22A14BFP8Stream(Movable):
    var cache_dir: String

    @staticmethod
    def open(cache_dir: String) raises -> Wan22A14BFP8Stream:
        if not validate_wan22_a14b_fp8_cache(cache_dir):
            raise Error(String("invalid/incomplete Wan A14B FP8 cache: ") + cache_dir)
        return Wan22A14BFP8Stream(cache_dir)

    def __init__(out self, var cache_dir: String):
        self.cache_dir = cache_dir^

    def load_shared(self, ctx: DeviceContext) raises -> Dict[String, TArc]:
        var cache = ShardedSafeTensors.open(
            self.cache_dir + String("/shared.safetensors")
        )
        var out = Dict[String, TArc]()
        for ref raw_name in cache.names():
            var name = String(raw_name)
            if name.startswith("blocks."):
                continue
            var tv = cache.tensor_view(name)
            if tv.dtype != STDtype.BF16:
                raise Error(String("Wan A14B shared cache tensor not BF16: ") + name)
            out[name] = TArc(Tensor.from_view(tv, ctx))
        if len(out) != WAN22_SHARED_TENSORS:
            raise Error("Wan A14B shared cache tensor count mismatch")
        return out^

    def load_block_bf16(
        self, index: Int, ctx: DeviceContext,
    ) raises -> Dict[String, TArc]:
        if index < 0 or index >= WAN22_A14B_BLOCKS:
            raise Error("Wan A14B block index out of range")
        var prefix = _block_prefix(index)
        var cache = ShardedSafeTensors.open(_block_cache_path(self.cache_dir, index))
        var out = Dict[String, TArc]()
        for ref raw_name in cache.names():
            var name = String(raw_name)
            if not name.startswith(prefix) or name.endswith(WAN22_FP8_SCALE_SUFFIX):
                continue
            var short_name = _substr(name, len(prefix), len(name))
            var tv = cache.tensor_view(name)
            if tv.dtype == STDtype.F8_E4M3:
                var scale_name = name + String(WAN22_FP8_SCALE_SUFFIX)
                var scale_tv = cache.tensor_view(scale_name)
                var raw = Tensor.from_view_raw(tv, ctx)
                var scale = Tensor.from_view(scale_tv, ctx)
                var bf16 = fp8_e4m3_dequant_perrow_to_bf16(raw, scale, ctx)
                out[short_name] = TArc(bf16^)
            elif tv.dtype == STDtype.BF16:
                out[short_name] = TArc(Tensor.from_view(tv, ctx))
            else:
                raise Error(String("unsupported Wan A14B cache dtype: ") + tv.dtype.name())
        ctx.synchronize()
        if len(out) != WAN22_BLOCK_SOURCE_TENSORS:
            raise Error(
                String("Wan A14B cached block ") + String(index)
                + String(" tensor count ") + String(len(out))
                + String(" != ") + String(WAN22_BLOCK_SOURCE_TENSORS)
            )
        return out^
