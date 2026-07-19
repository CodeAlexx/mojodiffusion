# offload/vmm_manager.mojo - model/block owner for CUDA VMM slabs.
#
# This is the Mojo-owned equivalent of Stagehand's VmmModelHandle layer: one
# model maps its transformer blocks to reserved virtual-address regions, while
# VmmSlabAllocator remains the low-level CUDA map/unmap/refcount primitive.
# Population from safetensors/mmap and background prefetch workers belong in the
# Turbo loader above this layer.

from std.collections import List

from serenitymojo.io.dtype import STDtype
from serenitymojo.offload.vmm_cuda import cu_mem_get_allocation_granularity
from serenitymojo.offload.vmm_slab import VmmSlabAllocator


def _round_up(n: Int, alignment: Int) -> Int:
    if alignment <= 1:
        return n
    var rem = n % alignment
    if rem == 0:
        return n
    return n + alignment - rem


@fieldwise_init
struct VmmBlockRecord(Copyable, Movable, ImplicitlyCopyable):
    var region_id: Int
    var requested_bytes: Int
    var reserved_bytes: Int
    var populated: Bool
    var last_touch: Int


struct VmmModelHandle(Movable):
    var model_id: String
    var dtype: STDtype
    var slab: VmmSlabAllocator
    var blocks: List[VmmBlockRecord]
    var touch_counter: Int
    var destroyed: Bool

    @staticmethod
    def create(
        var model_id: String,
        var block_sizes: List[Int],
        dtype: STDtype,
        device_ordinal: Int32 = 0,
    ) raises -> VmmModelHandle:
        if len(block_sizes) == 0:
            raise Error("VmmModelHandle.create: no blocks")

        var granularity = cu_mem_get_allocation_granularity(device_ordinal)
        var total_reserved = 0
        for i in range(len(block_sizes)):
            if block_sizes[i] <= 0:
                raise Error("VmmModelHandle.create: block size must be positive")
            total_reserved += _round_up(block_sizes[i], granularity)

        var slab = VmmSlabAllocator.create(total_reserved, device_ordinal)
        var blocks = List[VmmBlockRecord]()
        for i in range(len(block_sizes)):
            var region_id = slab.define_region(block_sizes[i])
            var region = slab.regions[region_id]
            blocks.append(VmmBlockRecord(
                region_id,
                block_sizes[i],
                region.size,
                False,
                -1,
            ))

        return VmmModelHandle(model_id^, dtype, slab^, blocks^)

    def __init__(
        out self,
        var model_id: String,
        dtype: STDtype,
        var slab: VmmSlabAllocator,
        var blocks: List[VmmBlockRecord],
    ):
        self.model_id = model_id^
        self.dtype = dtype
        self.slab = slab^
        self.blocks = blocks^
        self.touch_counter = 0
        self.destroyed = False

    def block_count(self) -> Int:
        return len(self.blocks)

    def requested_bytes(self) -> Int:
        var total = 0
        for i in range(len(self.blocks)):
            total += self.blocks[i].requested_bytes
        return total

    def reserved_bytes(self) raises -> Int:
        self._check_alive()
        return self.slab.total_size

    def resident_bytes(self) raises -> Int:
        self._check_alive()
        return self.slab.mapped_bytes

    def block_region_id(self, block_index: Int) raises -> Int:
        self._check_block(block_index)
        return self.blocks[block_index].region_id

    def block_requested_bytes(self, block_index: Int) raises -> Int:
        self._check_block(block_index)
        return self.blocks[block_index].requested_bytes

    def block_reserved_bytes(self, block_index: Int) raises -> Int:
        self._check_block(block_index)
        return self.blocks[block_index].reserved_bytes

    def block_refcount(self, block_index: Int) raises -> Int:
        self._check_block(block_index)
        var region_id = self.blocks[block_index].region_id
        return self.slab.regions[region_id].refcount

    def is_block_resident(self, block_index: Int) raises -> Bool:
        self._check_block(block_index)
        var region_id = self.blocks[block_index].region_id
        return self.slab.regions[region_id].resident

    def is_block_populated(self, block_index: Int) raises -> Bool:
        self._check_block(block_index)
        return self.blocks[block_index].populated

    def ensure_block_resident(mut self, block_index: Int) raises -> UInt64:
        self._check_block(block_index)
        var region_id = self.blocks[block_index].region_id
        var ptr = self.slab.ensure_resident(region_id)
        self._touch(block_index)
        return ptr

    def reserved_block_ptr(mut self, block_index: Int) raises -> UInt64:
        self._check_block(block_index)
        return self.slab.region_ptr(self.blocks[block_index].region_id)

    def mark_block_populated(mut self, block_index: Int) raises:
        self._check_block(block_index)
        var rec = self.blocks[block_index]
        rec.populated = True
        self.blocks[block_index] = rec

    def release_block(mut self, block_index: Int) raises:
        self._check_block(block_index)
        self.slab.release(self.blocks[block_index].region_id)

    def evict_block(mut self, block_index: Int) raises:
        self._check_block(block_index)
        self.slab.evict(self.blocks[block_index].region_id)
        var rec = self.blocks[block_index]
        rec.populated = False
        self.blocks[block_index] = rec

    def destroy(mut self) raises:
        if self.destroyed:
            return
        self.slab.destroy()
        self.destroyed = True

    def _touch(mut self, block_index: Int) raises:
        self._check_block(block_index)
        var rec = self.blocks[block_index]
        rec.last_touch = self.touch_counter
        self.touch_counter += 1
        self.blocks[block_index] = rec

    def _check_alive(self) raises:
        if self.destroyed:
            raise Error("VmmModelHandle: handle destroyed")

    def _check_block(self, block_index: Int) raises:
        self._check_alive()
        if block_index < 0 or block_index >= len(self.blocks):
            raise Error("VmmModelHandle: invalid block index")


struct VmmModelManager(Movable):
    var device_ordinal: Int32
    var registered_models: Int
    var registered_reserved_bytes: Int

    def __init__(out self, device_ordinal: Int32 = 0):
        self.device_ordinal = device_ordinal
        self.registered_models = 0
        self.registered_reserved_bytes = 0

    def register_model(
        mut self,
        var model_id: String,
        var block_sizes: List[Int],
        dtype: STDtype,
    ) raises -> VmmModelHandle:
        var handle = VmmModelHandle.create(
            model_id^,
            block_sizes^,
            dtype,
            self.device_ordinal,
        )
        self.registered_models += 1
        self.registered_reserved_bytes += handle.reserved_bytes()
        return handle^
