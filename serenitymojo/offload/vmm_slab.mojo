# offload/vmm_slab.mojo - first physical CUDA VMM slab primitive.
#
# This is the small, testable base for Stagehand parity:
# reserve one VA slab, define granularity-aligned regions, map/unmap physical
# memory on demand, and keep simple refcounts. Policy, background prefetch,
# last-use CUDA events, and LRU eviction belong above this layer.

from std.collections import List

from serenitymojo.offload.vmm_cuda import (
    make_allocation_prop,
    make_access_desc,
    free_packed,
    cu_mem_get_allocation_granularity,
    cu_mem_address_reserve,
    cu_mem_address_free,
    cu_mem_create,
    cu_mem_release,
    cu_mem_map,
    cu_mem_unmap,
    cu_mem_set_access,
)
from serenitymojo.io.ffi import BytePtr


def _round_up(n: Int, alignment: Int) -> Int:
    if alignment <= 1:
        return n
    var rem = n % alignment
    if rem == 0:
        return n
    return n + alignment - rem


@fieldwise_init
struct VmmRegion(Copyable, Movable, ImplicitlyCopyable):
    var offset: Int
    var size: Int
    var resident: Bool
    var phys_handle: UInt64
    var refcount: Int


struct VmmSlabAllocator(Movable):
    var device_ordinal: Int32
    var granularity: Int
    var base_ptr: UInt64
    var total_size: Int
    var next_offset: Int
    var mapped_bytes: Int
    var alloc_prop: BytePtr
    var access_desc: BytePtr
    var regions: List[VmmRegion]

    @staticmethod
    def create(total_size: Int, device_ordinal: Int32 = 0) raises -> VmmSlabAllocator:
        var granularity = cu_mem_get_allocation_granularity(device_ordinal)
        var rounded = _round_up(total_size, granularity)
        var base_ptr = cu_mem_address_reserve(rounded, granularity)
        var prop = make_allocation_prop(device_ordinal)
        var desc = make_access_desc(device_ordinal)
        return VmmSlabAllocator(
            device_ordinal, granularity, base_ptr, rounded, prop, desc,
        )

    def __init__(
        out self,
        device_ordinal: Int32,
        granularity: Int,
        base_ptr: UInt64,
        total_size: Int,
        alloc_prop: BytePtr,
        access_desc: BytePtr,
    ):
        self.device_ordinal = device_ordinal
        self.granularity = granularity
        self.base_ptr = base_ptr
        self.total_size = total_size
        self.next_offset = 0
        self.mapped_bytes = 0
        self.alloc_prop = alloc_prop
        self.access_desc = access_desc
        self.regions = List[VmmRegion]()

    def define_region(mut self, size: Int) raises -> Int:
        var rounded = _round_up(size, self.granularity)
        if self.next_offset + rounded > self.total_size:
            raise Error("VmmSlabAllocator.define_region: slab exhausted")
        var idx = len(self.regions)
        self.regions.append(VmmRegion(
            self.next_offset,
            rounded,
            False,
            UInt64(0),
            0,
        ))
        self.next_offset += rounded
        return idx

    def region_ptr(self, region_id: Int) raises -> UInt64:
        self._check_region(region_id)
        return self.base_ptr + UInt64(self.regions[region_id].offset)

    def ensure_resident(mut self, region_id: Int) raises -> UInt64:
        self._check_region(region_id)
        var r = self.regions[region_id]
        if not r.resident:
            var ptr = self.base_ptr + UInt64(r.offset)
            var handle = cu_mem_create(r.size, self.alloc_prop)
            try:
                cu_mem_map(ptr, r.size, handle)
                cu_mem_set_access(ptr, r.size, self.access_desc)
            except e:
                try:
                    cu_mem_release(handle)
                except:
                    pass
                raise e^
            r.resident = True
            r.phys_handle = handle
            self.mapped_bytes += r.size
        r.refcount += 1
        self.regions[region_id] = r
        return self.base_ptr + UInt64(r.offset)

    def release(mut self, region_id: Int) raises:
        self._check_region(region_id)
        var r = self.regions[region_id]
        if r.refcount <= 0:
            raise Error("VmmSlabAllocator.release: refcount already zero")
        r.refcount -= 1
        self.regions[region_id] = r

    def evict(mut self, region_id: Int) raises:
        self._check_region(region_id)
        var r = self.regions[region_id]
        if r.refcount != 0:
            raise Error("VmmSlabAllocator.evict: region still referenced")
        if r.resident:
            var ptr = self.base_ptr + UInt64(r.offset)
            cu_mem_unmap(ptr, r.size)
            cu_mem_release(r.phys_handle)
            self.mapped_bytes -= r.size
            if self.mapped_bytes < 0:
                self.mapped_bytes = 0
            r.resident = False
            r.phys_handle = UInt64(0)
            self.regions[region_id] = r

    def destroy(mut self) raises:
        for i in range(len(self.regions)):
            var r = self.regions[i]
            if r.refcount != 0:
                raise Error("VmmSlabAllocator.destroy: live region refcount")
            if r.resident:
                var ptr = self.base_ptr + UInt64(r.offset)
                cu_mem_unmap(ptr, r.size)
                cu_mem_release(r.phys_handle)
                r.resident = False
                r.phys_handle = UInt64(0)
                self.regions[i] = r
        if self.base_ptr != UInt64(0):
            cu_mem_address_free(self.base_ptr, self.total_size)
            self.base_ptr = UInt64(0)
        free_packed(self.access_desc)
        free_packed(self.alloc_prop)
        self.mapped_bytes = 0

    def _check_region(self, region_id: Int) raises:
        if region_id < 0 or region_id >= len(self.regions):
            raise Error("VmmSlabAllocator: invalid region id")
