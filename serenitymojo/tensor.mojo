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
    # Autograd identity. 0 = untracked (the inference default — every existing
    # 3-arg caller stays untracked, byte-identical). The training `Tape`
    # stamps a fresh nonzero id when a tensor enters the graph (see
    # serenitymojo/autograd.mojo). This is the only training-related field on
    # the otherwise inference-only Tensor; it is inert unless a Tape uses it.
    var id: Int

    def __init__(
        out self,
        var buf: DeviceBuffer[DType.uint8],
        var shape: List[Int],
        dtype: STDtype,
        id: Int = 0,
    ):
        self.buf = buf^
        self._shape = shape^
        self._dtype = dtype
        self.id = id

    # ── Autograd identity (training path; inert for inference) ──────────────
    def set_id(mut self, id: Int):
        """Stamp the autograd id (called by `Tape` when this tensor enters the
        graph). Inference never calls this; id stays 0 (untracked)."""
        self.id = id

    def clone(self, ctx: DeviceContext) raises -> Tensor:
        """Device→device copy into a fresh buffer (same dtype/shape, id=0).

        Needed by the training tape to SAVE activations for backward (the
        forward output may be moved/freed by the caller). Pure d2d copy — the
        same `enqueue_copy(dst_buf=dev, src_buf=self.buf)` pattern the VAE
        decoders already use. NOT in any inference hot path."""
        var nbytes = self.nbytes()
        var dev = ctx.enqueue_create_buffer[DType.uint8](nbytes)
        ctx.enqueue_copy(dst_buf=dev, src_buf=self.buf)
        ctx.synchronize()
        return Tensor(dev^, self._shape.copy(), self._dtype, 0)

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

    @staticmethod
    def from_view_as_f32[
        mut: Bool, //, origin: Origin[mut=mut]
    ](tv: TensorView[origin], ctx: DeviceContext) raises -> Tensor:
        """Build an F32 Tensor from a safetensors view, upcasting BF16/F16 on
        the host before H2D upload. Used by the LTX-2 vocoder parity path: the
        Mojo conv1d accumulates in F32, so keeping every weight/activation in
        F32 makes the chain match the F32 oracle (no BF16 round-trip jitter)."""
        _ = tv.dtype.to_mojo_dtype()
        var n = tv.numel()
        var nbytes_in = tv.nbytes()
        var expect = n * tv.dtype.byte_size()
        if nbytes_in != expect:
            raise Error(
                String("from_view_as_f32: view nbytes=")
                + String(nbytes_in)
                + " != numel*byte_size="
                + String(expect)
            )

        var nbytes_out = n * STDtype.F32.byte_size()
        var host_out = ctx.enqueue_create_host_buffer[DType.uint8](nbytes_out)
        var fp = host_out.unsafe_ptr().bitcast[Float32]()

        if tv.dtype == STDtype.F32:
            var outp = host_out.unsafe_ptr()
            for i in range(nbytes_out):
                outp[i] = tv.data[i]
        else:
            var host_in = ctx.enqueue_create_host_buffer[DType.uint8](nbytes_in)
            var hp = host_in.unsafe_ptr()
            for i in range(nbytes_in):
                hp[i] = tv.data[i]
            if tv.dtype == STDtype.BF16:
                var bp = host_in.unsafe_ptr().bitcast[BFloat16]()
                for i in range(n):
                    fp[i] = bp[i].cast[DType.float32]()
            elif tv.dtype == STDtype.F16:
                var hp16 = host_in.unsafe_ptr().bitcast[Float16]()
                for i in range(n):
                    fp[i] = hp16[i].cast[DType.float32]()
            else:
                raise Error(
                    String("from_view_as_f32: unsupported source dtype: ")
                    + tv.dtype.name()
                )

        var dev = ctx.enqueue_create_buffer[DType.uint8](nbytes_out)
        ctx.enqueue_copy(dst_buf=dev, src_buf=host_out)
        ctx.synchronize()
        return Tensor(dev^, tv.shape.copy(), STDtype.F32)

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

    def to_host_bf16(self, ctx: DeviceContext) raises -> List[BFloat16]:
        """Copy device data back to host as raw BF16 (HALF the bytes of
        `to_host`'s F32 list). This is the TRAINING hot-path activation-save
        carrier: flame-core keeps saved activations in BF16, and a BF16 host
        list (`List[BFloat16]`) is the faithful, memory-correct store — using
        `to_host` here doubled the resident activation set (the offload-trainer
        OOM). The stored compute dtype is cast DOWN to BF16 as we pack; a BF16
        source is a verbatim copy (no precision change)."""
        var n = self.numel()
        var nbytes = self.nbytes()
        var host = ctx.enqueue_create_host_buffer[DType.uint8](nbytes)
        ctx.enqueue_copy(dst_buf=host, src_buf=self.buf)
        ctx.synchronize()
        var out = List[BFloat16]()
        var dt = self._dtype.to_mojo_dtype()
        if dt == DType.bfloat16:
            var bp = host.unsafe_ptr().bitcast[BFloat16]()
            for i in range(n):
                out.append(bp[i])
        elif dt == DType.float32:
            var fp = host.unsafe_ptr().bitcast[Float32]()
            for i in range(n):
                out.append(fp[i].cast[DType.bfloat16]())
        else:  # float16
            var hp = host.unsafe_ptr().bitcast[Float16]()
            for i in range(n):
                out.append(hp[i].cast[DType.float32]().cast[DType.bfloat16]())
        return out^

    @staticmethod
    def from_host_bf16(
        values: List[BFloat16],
        var shape: List[Int],
        ctx: DeviceContext,
    ) raises -> Tensor:
        """Upload a host BF16 list to a fresh BF16 device buffer (verbatim, no
        F32 detour). The re-upload counterpart of `to_host_bf16` for the
        training activation-save path: `saved = t.to_host_bf16(ctx)` then in
        backward `Tensor.from_host_bf16(saved, shape, ctx)`. numel(shape) must
        equal len(values)."""
        var n = 1
        for i in range(len(shape)):
            n *= shape[i]
        if n != len(values):
            raise Error(
                String("from_host_bf16: numel(shape)=")
                + String(n)
                + " != len(values)="
                + String(len(values))
            )
        var nbytes = n * STDtype.BF16.byte_size()
        var host = ctx.enqueue_create_host_buffer[DType.uint8](nbytes)
        var bp = host.unsafe_ptr().bitcast[BFloat16]()
        for i in range(n):
            bp[i] = values[i]
        var dev = ctx.enqueue_create_buffer[DType.uint8](nbytes)
        ctx.enqueue_copy(dst_buf=dev, src_buf=host)
        ctx.synchronize()
        return Tensor(dev^, shape^, STDtype.BF16)
