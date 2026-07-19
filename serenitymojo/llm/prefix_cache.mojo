# serenitymojo/llm/prefix_cache.mojo — persist/load a primed KVCache prefix.
#
# The magic-prompt captioner's system prompt is FIXED (~6.4k tokens) and, under
# causal attention, its per-layer K/V depend only on the prefix itself. So the
# prefix is primed ONCE, saved to a safetensors file, and every later run loads
# it (~1GB read) instead of re-priming 6.4k sequential decode_steps (MEASURED
# 263s of the captioner's ~4:45 wall). Bytes round-trip bit-exact (bf16 raw
# D2H on save, raw H2D on load), so generation from a loaded prefix is
# token-for-token identical to a fresh full prime — gated by
# llm/tests/prefix_cache_gate.mojo.
#
# File layout (single safetensors):
#   prefix_ids  F32 [n]                 the EXACT token ids primed (token ids
#                                       < 152k are exactly representable in F32)
#   k.{layer}   BF16 [1, n, H_kv, dh]   per-layer key cache
#   v.{layer}   BF16 [1, n, H_kv, dh]   per-layer value cache
# The loader FAILS LOUD on any missing/mis-shaped tensor; callers treat any
# load error as "rebuild the cache" (self-healing).

from std.gpu.host import DeviceContext
from std.memory import ArcPointer
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.io.ffi import BytePtr, sys_memcpy
from serenitymojo.llm.decoder import KVCache


def save_prefix_cache(
    cache: KVCache, prefix_ids: List[Int], path: String, ctx: DeviceContext
) raises:
    """Write a primed KVCache (+ the ids it covers) to `path` (atomic tmp+rename
    via save_safetensors). Every layer must be populated and cover exactly
    len(prefix_ids) positions."""
    var n = len(prefix_ids)
    if n == 0:
        raise Error("save_prefix_cache: empty prefix")
    var names = List[String]()
    var tensors = List[ArcPointer[Tensor]]()

    # prefix_ids as F32 (exact for ids < 2^24; Qwen3 vocab is 151936).
    var idsf = List[Float32]()
    for i in range(n):
        idsf.append(Float32(prefix_ids[i]))
    names.append(String("prefix_ids"))
    tensors.append(ArcPointer(Tensor.from_host(idsf, [n], STDtype.F32, ctx)))

    for layer in range(len(cache.k)):
        if not cache.has[layer]:
            raise Error(
                String("save_prefix_cache: layer ") + String(layer)
                + " has no cache (prime the full prefix first)"
            )
        var ksh = cache.k[layer][].shape()
        if ksh[1] != n:
            raise Error(
                String("save_prefix_cache: layer ") + String(layer)
                + " cache depth " + String(ksh[1])
                + " != len(prefix_ids) " + String(n)
            )
        names.append(String("k.") + String(layer))
        tensors.append(cache.k[layer].copy())
        names.append(String("v.") + String(layer))
        tensors.append(cache.v[layer].copy())

    save_safetensors(names, tensors, path, ctx)


def _load_tensor(st: SafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    """Raw H2D copy of one tensor from the mmap'd file (bit-exact; mirrors
    Tensor.from_view's staging: mmap bytes -> pinned host -> device)."""
    var info = st.tensor_info(name)
    var bytes = st.tensor_bytes(name)
    var nbytes = len(bytes)
    var host = ctx.enqueue_create_host_buffer[DType.uint8](nbytes)
    var dst = BytePtr(unsafe_from_address=Int(host.unsafe_ptr()))
    var src = BytePtr(unsafe_from_address=Int(bytes.unsafe_ptr()))
    _ = sys_memcpy(dst, src, nbytes)
    var dev = ctx.enqueue_create_buffer[DType.uint8](nbytes)
    ctx.enqueue_copy(dst_buf=dev, src_buf=host)
    ctx.synchronize()
    return Tensor(dev^, info.shape.copy(), info.dtype)


def load_prefix_cache(
    path: String, num_layers: Int, ctx: DeviceContext, mut ids_out: List[Int]
) raises -> KVCache:
    """Load a saved prefix cache; fills `ids_out` with the EXACT token ids the
    cache covers (callers verify against their current tokenization). Validates
    layer count, dtypes, and that every layer covers the same n positions as
    prefix_ids. Raises on ANY mismatch — callers rebuild on failure."""
    var st = SafeTensors.open(path)

    var ids_t = _load_tensor(st, String("prefix_ids"), ctx)
    if ids_t.dtype() != STDtype.F32:
        raise Error("load_prefix_cache: prefix_ids must be F32")
    var idsf = ids_t.to_host(ctx)
    ids_out.clear()
    for i in range(len(idsf)):
        ids_out.append(Int(idsf[i]))
    var n = len(ids_out)
    if n == 0:
        raise Error("load_prefix_cache: empty prefix_ids")

    var cache = KVCache(num_layers)
    for layer in range(num_layers):
        var k = _load_tensor(st, String("k.") + String(layer), ctx)
        var v = _load_tensor(st, String("v.") + String(layer), ctx)
        if k.dtype() != STDtype.BF16 or v.dtype() != STDtype.BF16:
            raise Error(
                String("load_prefix_cache: layer ") + String(layer)
                + " k/v must be BF16"
            )
        var ksh = k.shape()
        if len(ksh) != 4 or ksh[0] != 1 or ksh[1] != n:
            raise Error(
                String("load_prefix_cache: layer ") + String(layer)
                + " k shape mismatch (depth " + String(ksh[1])
                + " vs ids " + String(n) + ")"
            )
        cache.k.append(ArcPointer(k^))
        cache.v.append(ArcPointer(v^))
        cache.has[layer] = True
    cache.length = n
    return cache^
