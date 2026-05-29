# tensor.mojo — Tensor: an on-GPU tensor for the Serenitymojo inference library.
#
# Phase A foundation (PHASE_AB_PLAN.md, Decision #3). This is the type every
# later forward-pass module (encoders / VAEs / offloading) builds on. It owns a
# `gpu.host.DeviceBuffer` (device memory) plus host-side `shape` + `dtype`
# metadata. Inference-only: no autograd, no training, no in-place mutation API.
#
# Design notes / contracts:
#   * BF16-first. The stored device dtype is whatever `STDtype.to_mojo_dtype()`
#     yields (bfloat16 / float16 / float32). Ops that need F32 accumulation
#     (matmul, rms_norm) cast internally — the Tensor itself just holds storage.
#   * Construction is via H2D upload: `from_host` (host List) or `from_view`
#     (a loader `io.TensorView` — the weight-load path). Both copy the bytes to
#     a fresh `DeviceBuffer` so the Tensor's lifetime is independent of the
#     source mmap (no origin entanglement with the loader's Span — the data is
#     copied, not aliased).
#   * `to_host` copies device -> host for parity / inspection.
#   * The `DeviceBuffer` carries its own `dtype` parameter, but we erase it
#     behind a runtime `STDtype` so a single `Tensor` type can hold any of the
#     three compute dtypes. We store ALL device data as raw bytes in a
#     `DeviceBuffer[DType.uint8]` and reinterpret (`.bitcast`) at the op
#     boundary. This keeps `Tensor` monomorphic (no dtype type-parameter to
#     thread through every model module) while still being byte-exact.
#
# Mojo 1.0.0b1, Linux x86-64, NVIDIA. GPU-only compute.

from std.gpu.host import DeviceContext, DeviceBuffer
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.tensor_view import TensorView


struct Tensor(Movable):
    """An on-GPU tensor: device-resident bytes + shape + dtype.

    Storage is a `DeviceBuffer[DType.uint8]` holding the raw element bytes
    (numel * dtype.byte_size() bytes). The runtime `dtype` (STDtype) records how
    to interpret them; ops `.bitcast` the byte buffer to the concrete element
    DType at the call boundary. Not Copyable — uniquely owns its device buffer
    (mirrors the loader's `SafeTensors`/`MmapRegion` ownership discipline)."""

    var buf: DeviceBuffer[DType.uint8]
    var _shape: List[Int]
    var _dtype: STDtype

    def __init__(
        out self,
        var buf: DeviceBuffer[DType.uint8],
        var shape: List[Int],
        dtype: STDtype,
    ):
        self.buf = buf^
        self._shape = shape^
        self._dtype = dtype

    # ── Metadata ──────────────────────────────────────────────────────────────
    def shape(self) -> List[Int]:
        """Tensor dimensions (row-major), copied out."""
        return self._shape.copy()

    def dtype(self) -> STDtype:
        """The element dtype (STDtype: BF16 / F16 / F32 for compute)."""
        return self._dtype

    def numel(self) -> Int:
        """Number of elements = product of shape dims (1 for scalar)."""
        var n = 1
        for i in range(len(self._shape)):
            n *= self._shape[i]
        return n

    def nbytes(self) -> Int:
        """Device byte length == numel * dtype.byte_size()."""
        return self.numel() * self._dtype.byte_size()

    # ── Constructors ────────────────────────────────────────────────────────
    @staticmethod
    def from_host(
        values: List[Float32],
        var shape: List[Int],
        dtype: STDtype,
        ctx: DeviceContext,
    ) raises -> Tensor:
        """Upload host F32 values to a fresh device buffer, casting to `dtype`.

        `values` are always provided as F32 (the convenient host representation
        for tests / numpy oracles); we cast each element down to the requested
        compute dtype as we pack the byte buffer. numel(shape) must equal
        len(values)."""
        var n = 1
        for i in range(len(shape)):
            n *= shape[i]
        if n != len(values):
            raise Error(
                String("from_host: numel(shape)=")
                + String(n)
                + " != len(values)="
                + String(len(values))
            )
        var bsz = dtype.byte_size()
        var nbytes = n * bsz
        # Stage in a host byte buffer, casting F32 -> compute dtype.
        var host = ctx.enqueue_create_host_buffer[DType.uint8](nbytes)
        var dt = dtype.to_mojo_dtype()
        if dt == DType.float32:
            var fp = host.unsafe_ptr().bitcast[Float32]()
            for i in range(n):
                fp[i] = values[i]
        elif dt == DType.bfloat16:
            var bp = host.unsafe_ptr().bitcast[BFloat16]()
            for i in range(n):
                bp[i] = values[i].cast[DType.bfloat16]()
        else:  # float16
            var hp = host.unsafe_ptr().bitcast[Float16]()
            for i in range(n):
                hp[i] = values[i].cast[DType.float16]()
        var dev = ctx.enqueue_create_buffer[DType.uint8](nbytes)
        ctx.enqueue_copy(dst_buf=dev, src_buf=host)
        ctx.synchronize()
        return Tensor(dev^, shape^, dtype)

    @staticmethod
    def from_view[
        mut: Bool, //, origin: Origin[mut=mut]
    ](tv: TensorView[origin], ctx: DeviceContext) raises -> Tensor:
        """Build a Tensor from a loader `TensorView` (weight-load path) by H2D
        copy of its raw bytes. The bytes are COPIED into a fresh device buffer,
        so the resulting Tensor does not alias (and does not keep alive) the
        source mmap region. Only BF16/F16/F32 views are accepted (compute
        dtypes); others raise via `to_mojo_dtype()`."""
        _ = tv.dtype.to_mojo_dtype()  # validate it's a supported compute dtype
        var nbytes = tv.nbytes()
        var expect = tv.numel() * tv.dtype.byte_size()
        if nbytes != expect:
            raise Error(
                String("from_view: view nbytes=")
                + String(nbytes)
                + " != numel*byte_size="
                + String(expect)
            )
        # Stage the host bytes (tv.data is host-resident mmap pages) into a
        # pinned host buffer, then H2D.
        var host = ctx.enqueue_create_host_buffer[DType.uint8](nbytes)
        var hp = host.unsafe_ptr()
        for i in range(nbytes):
            hp[i] = tv.data[i]
        var dev = ctx.enqueue_create_buffer[DType.uint8](nbytes)
        ctx.enqueue_copy(dst_buf=dev, src_buf=host)
        ctx.synchronize()
        return Tensor(dev^, tv.shape.copy(), tv.dtype)

    @staticmethod
    def from_view_raw[
        mut: Bool, //, origin: Origin[mut=mut]
    ](tv: TensorView[origin], ctx: DeviceContext) raises -> Tensor:
        """Build a Tensor by a verbatim H2D byte copy, PRESERVING the on-disk
        dtype (no compute-dtype validation). Use this for non-compute dtypes
        such as FP8 (F8_E4M3) where the bytes are loaded raw and later
        dequantized on the GPU (see ops/fp8.mojo). The result's `dtype()`
        reports the original STDtype (e.g. F8_E4M3); ops that need a compute
        dtype must dequantize first."""
        var nbytes = tv.nbytes()
        var expect = tv.numel() * tv.dtype.byte_size()
        if nbytes != expect:
            raise Error(
                String("from_view_raw: view nbytes=")
                + String(nbytes)
                + " != numel*byte_size="
                + String(expect)
            )
        var host = ctx.enqueue_create_host_buffer[DType.uint8](nbytes)
        var hp = host.unsafe_ptr()
        for i in range(nbytes):
            hp[i] = tv.data[i]
        var dev = ctx.enqueue_create_buffer[DType.uint8](nbytes)
        ctx.enqueue_copy(dst_buf=dev, src_buf=host)
        ctx.synchronize()
        return Tensor(dev^, tv.shape.copy(), tv.dtype)

    @staticmethod
    def from_view_as_bf16[
        mut: Bool, //, origin: Origin[mut=mut]
    ](tv: TensorView[origin], ctx: DeviceContext) raises -> Tensor:
        """Build a BF16 Tensor from a safetensors view, converting F32/F16 on
        the host before H2D upload. This is the production loader path for
        F32-on-disk 8B checkpoints such as HiDream-O1: the transient F32 copy
        never lives on the GPU."""
        _ = tv.dtype.to_mojo_dtype()
        var n = tv.numel()
        var nbytes_in = tv.nbytes()
        var expect = n * tv.dtype.byte_size()
        if nbytes_in != expect:
            raise Error(
                String("from_view_as_bf16: view nbytes=")
                + String(nbytes_in)
                + " != numel*byte_size="
                + String(expect)
            )

        var nbytes_out = n * STDtype.BF16.byte_size()
        var host_out = ctx.enqueue_create_host_buffer[DType.uint8](nbytes_out)
        var bp = host_out.unsafe_ptr().bitcast[BFloat16]()

        if tv.dtype == STDtype.BF16:
            var outp = host_out.unsafe_ptr()
            for i in range(nbytes_out):
                outp[i] = tv.data[i]
        else:
            # Stage the source bytes in a typed host buffer so Mojo can read
            # F32/F16 scalars safely before casting to BF16.
            var host_in = ctx.enqueue_create_host_buffer[DType.uint8](nbytes_in)
            var hp = host_in.unsafe_ptr()
            for i in range(nbytes_in):
                hp[i] = tv.data[i]
            if tv.dtype == STDtype.F32:
                var fp = host_in.unsafe_ptr().bitcast[Float32]()
                for i in range(n):
                    bp[i] = fp[i].cast[DType.bfloat16]()
            elif tv.dtype == STDtype.F16:
                var hp16 = host_in.unsafe_ptr().bitcast[Float16]()
                for i in range(n):
                    bp[i] = hp16[i].cast[DType.float32]().cast[DType.bfloat16]()
            else:
                raise Error(
                    String("from_view_as_bf16: unsupported source dtype: ")
                    + tv.dtype.name()
                )

        var dev = ctx.enqueue_create_buffer[DType.uint8](nbytes_out)
        ctx.enqueue_copy(dst_buf=dev, src_buf=host_out)
        ctx.synchronize()
        return Tensor(dev^, tv.shape.copy(), STDtype.BF16)

    # ── Readback ──────────────────────────────────────────────────────────────
    def to_host(self, ctx: DeviceContext) raises -> List[Float32]:
        """Copy device data back to host as F32 (upcasting from the stored
        compute dtype). For parity / inspection only — never in the hot path."""
        var n = self.numel()
        var nbytes = self.nbytes()
        var host = ctx.enqueue_create_host_buffer[DType.uint8](nbytes)
        ctx.enqueue_copy(dst_buf=host, src_buf=self.buf)
        ctx.synchronize()
        var out = List[Float32]()
        var dt = self._dtype.to_mojo_dtype()
        if dt == DType.float32:
            var fp = host.unsafe_ptr().bitcast[Float32]()
            for i in range(n):
                out.append(fp[i])
        elif dt == DType.bfloat16:
            var bp = host.unsafe_ptr().bitcast[BFloat16]()
            for i in range(n):
                out.append(bp[i].cast[DType.float32]())
        else:  # float16
            var hp = host.unsafe_ptr().bitcast[Float16]()
            for i in range(n):
                out.append(hp[i].cast[DType.float32]())
        return out^
