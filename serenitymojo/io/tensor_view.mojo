# tensor_view.mojo — TensorView: a tensor's typed metadata bundled with an
# ORIGIN-BOUND view of its raw bytes.
#
# Chunk 2 of the serenity-safetensors -> Mojo port (BUILD_PLAN.md row
# "tensor_view.mojo"). Builds on chunk-1 `SafeTensors` (safetensors.mojo) and
# its origin-bound `tensor_bytes(self,name) -> Span[UInt8, origin_of(self)]`.
#
# The whole point of this type is the LIFETIME CONTRACT inherited from chunk-1's
# F1 fix: the bytes are a `Span[UInt8, origin]` where `origin` is a STRUCT
# PARAMETER tied to the source `SafeTensors`. The compiler therefore keeps the
# source (and its mmap'd region) alive for exactly as long as any TensorView
# over it is in use. No bare/untracked pointer is ever stored.
#
# Scope of the compile-time guarantee (chunk-2 skeptic F2 — accurate claim):
# the origin binding compile-rejects escape-returning a view past its source and
# using a view past an explicit `src^.__del__()`. It does NOT catch
# *reassigning/overwriting* the source binding while a view is live
# (`var v = ...; src = other` compiles and yields a use-after-free) — a Mojo
# 1.0.0b1 origin-tracking limitation. Not reassigning the source while a view is
# live is the caller's contract.
#
# Mojo 1.0.0b1, Linux x86-64.

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors


struct TensorView[mut: Bool, //, origin: Origin[mut=mut]](Movable):
    """A tensor's typed metadata + an origin-bound view of its raw bytes.

    Parametric on `origin`: the byte view (`data`) carries `Self.origin`, which
    is tied to the source `SafeTensors`. Because the origin is part of the type,
    escape-returning a `TensorView` past its source, or using one past an
    explicit `src^.__del__()`, is a compile error — the same use-after-munmap
    protection chunk-1 established for `tensor_bytes`, now carried in a value you
    can pass around and store *within the source's scope*. Reassigning the
    source binding while a view is live is NOT caught (see module header F2);
    avoiding that is the caller's contract.

    Fields:
      * `dtype` — the safetensors dtype (STDtype; carries label + byte size).
      * `shape` — the tensor's dimensions (row-major, as stored).
      * `data`  — `Span[UInt8, Self.origin]` over the exact tensor bytes in the
                  mmap'd data segment (length == nbytes)."""

    var dtype: STDtype
    var shape: List[Int]
    var data: Span[UInt8, Self.origin]

    def __init__(
        out self,
        dtype: STDtype,
        var shape: List[Int],
        data: Span[UInt8, Self.origin],
    ):
        self.dtype = dtype
        self.shape = shape^
        self.data = data

    def nbytes(self) -> Int:
        """Byte length of the view == the tensor's stored size."""
        return len(self.data)

    def numel(self) -> Int:
        """Number of elements = product of shape dims (1 for a scalar/empty
        shape)."""
        var n = 1
        for i in range(len(self.shape)):
            n *= self.shape[i]
        return n

    # ── Future H2D upload — STUB ONLY (read pilot; do NOT implement now) ───────
    #
    # When the compute path lands, host->device upload will copy `self.data`
    # (host pages) to a device buffer. The intended primitive is
    # `gpu.host.DeviceContext.enqueue_copy`, e.g. (illustrative signature):
    #
    #   from gpu.host import DeviceContext, DeviceBuffer
    #   def to_device(self, ctx: DeviceContext) raises -> DeviceBuffer[DType...]:
    #       var buf = ctx.enqueue_create_buffer[dt](self.numel())
    #       ctx.enqueue_copy(buf, self.data.unsafe_ptr())  # H2D
    #       ctx.synchronize()
    #       return buf^
    #
    # Deferred per BUILD_PLAN row + PLAN Decision #3 (full Tensor deferred). The
    # weight-loading path is intentionally host-side I/O (mmap -> host pages);
    # the device upload is a separate, small step that belongs with the compute
    # port, not the reader. Left unimplemented on purpose.


def from_parts[
    mut: Bool, //, origin: Origin[mut=mut]
](dtype: STDtype, var shape: List[Int], data: Span[UInt8, origin]) -> TensorView[
    origin
]:
    """Assemble a `TensorView`, inferring the view's `origin` FROM the `data`
    span argument. Inferring the origin from the span (rather than naming it
    independently, e.g. `origin_of(st)`) is what lets the type unify cleanly —
    Mojo 1.0.0b1 treats an independently-named `origin_of(x)` as a distinct
    symbol from the span's own inferred origin even when they print identically.
    See SKEPTIC-BAIT in the chunk-2 report."""
    return TensorView[origin](dtype=dtype, shape=shape^, data=data)


# NOTE on the public builder shape (chunk-2 finding):
#
# A one-call `view_of(st, name) -> TensorView[origin_of(st)]` does NOT compile
# in Mojo 1.0.0b1: naming the return origin `origin_of(st)` makes a symbol that
# the compiler refuses to unify with the *inferred* origin of the span returned
# by `st.tensor_bytes(name)` (both print as `origin_of(st)` — a real Mojo
# origin-identity quirk, reproduced in parity/probe_view_origin.mojo). The
# clean, working idiom infers the origin FROM the span at the call site:
#
#     var info  = st.tensor_info(name)
#     var bytes = st.tensor_bytes(name)        # Span[UInt8, <st's origin>]
#     var view  = from_parts(info.dtype, info.shape.copy(), bytes)
#
# `from_parts` infers `origin` from `bytes`, so the TensorView is correctly
# lifetime-bound to `st` with no named-origin mismatch and no unsafe cast.
# `ShardedSafeTensors.tensor_view` uses exactly this idiom over its owned shard.


# ── Test (task-required): build a TensorView from the VAE ─────────────────────
def test_tensor_view_vae() raises:
    comptime VAE_PATH = (
        "/home/alex/.cache/huggingface/hub/models--Tongyi-MAI--Z-Image/"
        "snapshots/04cc4abb7c5069926f75c9bfde9ef43d49423021/vae/"
        "diffusion_pytorch_model.safetensors"
    )
    var st = SafeTensors.open(String(VAE_PATH))
    # Build the view via the call-site idiom (origin inferred from the span).
    var info = st.tensor_info(String("decoder.conv_in.bias"))
    var bytes = st.tensor_bytes(String("decoder.conv_in.bias"))
    var v = from_parts(info.dtype, info.shape.copy(), bytes)
    print("TensorView(decoder.conv_in.bias):")
    print("  dtype =", v.dtype.name())
    var shape_str = String("[")
    for i in range(len(v.shape)):
        if i > 0:
            shape_str += ", "
        shape_str += String(v.shape[i])
    shape_str += "]"
    print("  shape =", shape_str)
    print("  nbytes =", v.nbytes())
    print("  numel =", v.numel())
    print("  first_byte =", Int(v.data[0]))
    # Sanity: nbytes == numel * dtype.byte_size() for this BF16 tensor.
    var expect = v.numel() * v.dtype.byte_size()
    if v.nbytes() != expect:
        raise Error(
            String("nbytes mismatch: ")
            + String(v.nbytes())
            + " != "
            + String(expect)
        )
    print("  nbytes == numel*byte_size OK")
    _ = st.count()


def main() raises:
    test_tensor_view_vae()
