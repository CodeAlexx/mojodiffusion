from std.builtin.dtype import DType
from std.collections import List
from std.gpu.host import DeviceBuffer, DeviceContext
from std.memory import ArcPointer

from serenitymojo.io.dtype import STDtype
from serenitymojo.tensor import Tensor


def _align_up(n: Int, alignment: Int) -> Int:
    if alignment <= 1:
        return n
    var rem = n % alignment
    if rem == 0:
        return n
    return n + (alignment - rem)


@fieldwise_init
struct ScratchRingMark(Copyable, Movable):
    var slab_index: Int
    var offset: Int
    var back_slab_index: Int
    var back_offset: Int


struct ScratchRingAllocator(Movable):
    """Frame-scoped GPU scratch allocator for temporary Tensor storage.

    Returned tensors are views into allocator-owned slabs. Callers must only
    reset or rewind after every tensor allocated since that mark is dead or no
    longer read by queued device work.
    """

    var slabs: List[ArcPointer[Tensor]]
    var slab_bytes: Int
    var slab_index: Int
    var offset: Int
    var back_slab_index: Int
    var back_offset: Int
    var alignment: Int
    var peak_bytes: Int

    def __init__(
        out self,
        ctx: DeviceContext,
        slab_bytes: Int,
        num_slabs: Int = 1,
        alignment: Int = 16,
    ) raises:
        if slab_bytes <= 0:
            raise Error("ScratchRingAllocator slab_bytes must be positive")
        if num_slabs <= 0:
            raise Error("ScratchRingAllocator num_slabs must be positive")

        self.slabs = List[ArcPointer[Tensor]]()
        self.slab_bytes = slab_bytes
        self.slab_index = 0
        self.offset = 0
        self.back_slab_index = num_slabs - 1
        self.back_offset = slab_bytes
        self.alignment = alignment
        self.peak_bytes = 0

        for _ in range(num_slabs):
            var dev = ctx.enqueue_create_buffer[DType.uint8](slab_bytes)
            var shape = List[Int]()
            shape.append(slab_bytes)
            self.slabs.append(ArcPointer[Tensor](Tensor(dev^, shape^, STDtype.U8)))

    def capacity_bytes(self) -> Int:
        return len(self.slabs) * self.slab_bytes

    def used_bytes(self) -> Int:
        var front_used = self.slab_index * self.slab_bytes + self.offset
        var back_used = (
            (len(self.slabs) - 1 - self.back_slab_index) * self.slab_bytes
            + (self.slab_bytes - self.back_offset)
        )
        return front_used + back_used

    def mark(self) -> ScratchRingMark:
        return ScratchRingMark(
            slab_index=self.slab_index,
            offset=self.offset,
            back_slab_index=self.back_slab_index,
            back_offset=self.back_offset,
        )

    def reset(mut self):
        self.slab_index = 0
        self.offset = 0
        self.back_slab_index = len(self.slabs) - 1
        self.back_offset = self.slab_bytes

    def rewind(mut self, mark: ScratchRingMark) raises:
        if mark.slab_index < 0 or mark.slab_index >= len(self.slabs):
            raise Error("ScratchRingAllocator rewind mark out of range")
        if mark.offset < 0 or mark.offset > self.slab_bytes:
            raise Error("ScratchRingAllocator rewind offset out of range")
        if mark.back_slab_index < 0 or mark.back_slab_index >= len(self.slabs):
            raise Error("ScratchRingAllocator rewind back mark out of range")
        if mark.back_offset < 0 or mark.back_offset > self.slab_bytes:
            raise Error("ScratchRingAllocator rewind back offset out of range")
        self.slab_index = mark.slab_index
        self.offset = mark.offset
        self.back_slab_index = mark.back_slab_index
        self.back_offset = mark.back_offset

    def _alloc_buffer(mut self, nbytes: Int) raises -> DeviceBuffer[DType.uint8]:
        if nbytes < 0:
            raise Error("ScratchRingAllocator cannot allocate negative bytes")
        var aligned_nbytes = _align_up(nbytes, self.alignment)
        if aligned_nbytes > self.slab_bytes:
            raise Error("ScratchRingAllocator allocation larger than slab")

        if self.slab_index >= len(self.slabs):
            raise Error("ScratchRingAllocator exhausted")

        if (
            self.offset + aligned_nbytes > self.slab_bytes
            or (
                self.slab_index == self.back_slab_index
                and self.offset + aligned_nbytes > self.back_offset
            )
        ):
            self.slab_index += 1
            self.offset = 0
            if self.slab_index >= len(self.slabs) or self.slab_index > self.back_slab_index:
                raise Error("ScratchRingAllocator exhausted")

        var alloc_offset = self.offset
        self.offset += aligned_nbytes
        if self.slab_index == self.back_slab_index and self.offset > self.back_offset:
            raise Error("ScratchRingAllocator exhausted")
        var used = self.used_bytes()
        if used > self.peak_bytes:
            self.peak_bytes = used

        return self.slabs[self.slab_index][].buf.create_sub_buffer[DType.uint8](
            alloc_offset, nbytes
        )

    def _alloc_buffer_reverse(
        mut self, nbytes: Int
    ) raises -> DeviceBuffer[DType.uint8]:
        if nbytes < 0:
            raise Error("ScratchRingAllocator cannot allocate negative bytes")
        var aligned_nbytes = _align_up(nbytes, self.alignment)
        if aligned_nbytes > self.slab_bytes:
            raise Error("ScratchRingAllocator allocation larger than slab")

        if self.back_slab_index < 0:
            raise Error("ScratchRingAllocator exhausted")

        var new_offset = self.back_offset - aligned_nbytes
        if (
            new_offset < 0
            or (
                self.back_slab_index == self.slab_index
                and new_offset < self.offset
            )
        ):
            self.back_slab_index -= 1
            self.back_offset = self.slab_bytes
            if self.back_slab_index < 0 or self.back_slab_index < self.slab_index:
                raise Error("ScratchRingAllocator exhausted")
            new_offset = self.back_offset - aligned_nbytes

        if self.back_slab_index == self.slab_index and new_offset < self.offset:
            raise Error("ScratchRingAllocator exhausted")
        self.back_offset = new_offset
        var used = self.used_bytes()
        if used > self.peak_bytes:
            self.peak_bytes = used

        return self.slabs[self.back_slab_index][].buf.create_sub_buffer[DType.uint8](
            self.back_offset, nbytes
        )

    def alloc_tensor(
        mut self, var shape: List[Int], dtype: STDtype
    ) raises -> Tensor:
        var elems = 1
        for i in range(len(shape)):
            elems *= shape[i]
        var buf = self._alloc_buffer(elems * dtype.byte_size())
        return Tensor(buf^, shape^, dtype)

    def alloc_tensor_reverse(
        mut self, var shape: List[Int], dtype: STDtype
    ) raises -> Tensor:
        """Allocate from the tail of the ring for backward/recompute frames."""
        var elems = 1
        for i in range(len(shape)):
            elems *= shape[i]
        var buf = self._alloc_buffer_reverse(elems * dtype.byte_size())
        return Tensor(buf^, shape^, dtype)

    def empty_like(mut self, x: Tensor) raises -> Tensor:
        return self.alloc_tensor(x.shape(), x.dtype())

    def empty_like_reverse(mut self, x: Tensor) raises -> Tensor:
        return self.alloc_tensor_reverse(x.shape(), x.dtype())

    def clone_tensor(mut self, x: Tensor, ctx: DeviceContext) raises -> Tensor:
        var out = self.empty_like(x)
        ctx.enqueue_copy(dst_buf=out.buf, src_buf=x.buf)
        return out^

    def clone_tensor_reverse(
        mut self, x: Tensor, ctx: DeviceContext
    ) raises -> Tensor:
        var out = self.empty_like_reverse(x)
        ctx.enqueue_copy(dst_buf=out.buf, src_buf=x.buf)
        return out^
