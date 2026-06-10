# offload/vmm_cuda.mojo - CUDA VMM driver ABI shared by runtime offload code.
#
# This is the Mojo-side foundation for Stagehand/Flame VMM parity. It exposes
# only low-level driver calls and byte-packed CUDA structs; higher-level slab,
# region, refcount, and eviction policy belong in separate runtime modules.

from std.ffi import external_call
from std.memory import alloc

from serenitymojo.io.ffi import BytePtr


comptime CUDA_SUCCESS = 0
comptime CU_MEM_ALLOCATION_TYPE_PINNED = UInt32(1)
comptime CU_MEM_LOCATION_TYPE_DEVICE = UInt32(1)
comptime CU_MEM_ACCESS_FLAGS_PROT_READWRITE = UInt32(3)
comptime CU_MEM_ALLOC_GRANULARITY_RECOMMENDED = UInt32(1)
comptime CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED = Int32(102)
comptime CU_EVENT_DISABLE_TIMING = UInt32(2)


def _ptr[pointee: AnyType](p: UnsafePointer[pointee, MutAnyOrigin]) -> BytePtr:
    return BytePtr(unsafe_from_address=Int(p))


def _write_u8(buf: BytePtr, offset: Int, value: UInt8):
    buf[offset] = value


def _write_u16_le(buf: BytePtr, offset: Int, value: UInt16):
    buf[offset] = UInt8(value & UInt16(0x00FF))
    buf[offset + 1] = UInt8((value >> 8) & UInt16(0x00FF))


def _write_u32_le(buf: BytePtr, offset: Int, value: UInt32):
    buf[offset] = UInt8(value & UInt32(0x000000FF))
    buf[offset + 1] = UInt8((value >> 8) & UInt32(0x000000FF))
    buf[offset + 2] = UInt8((value >> 16) & UInt32(0x000000FF))
    buf[offset + 3] = UInt8((value >> 24) & UInt32(0x000000FF))


def _write_i32_le(buf: BytePtr, offset: Int, value: Int32):
    _write_u32_le(buf, offset, UInt32(value))


def _write_u64_le(buf: BytePtr, offset: Int, value: UInt64):
    buf[offset] = UInt8(value & UInt64(0x00000000000000FF))
    buf[offset + 1] = UInt8((value >> 8) & UInt64(0x00000000000000FF))
    buf[offset + 2] = UInt8((value >> 16) & UInt64(0x00000000000000FF))
    buf[offset + 3] = UInt8((value >> 24) & UInt64(0x00000000000000FF))
    buf[offset + 4] = UInt8((value >> 32) & UInt64(0x00000000000000FF))
    buf[offset + 5] = UInt8((value >> 40) & UInt64(0x00000000000000FF))
    buf[offset + 6] = UInt8((value >> 48) & UInt64(0x00000000000000FF))
    buf[offset + 7] = UInt8((value >> 56) & UInt64(0x00000000000000FF))


def _zero(buf: BytePtr, n: Int):
    for i in range(n):
        buf[i] = 0


def check_cuda(rc: Int, op: String) raises:
    if rc != CUDA_SUCCESS:
        raise Error(op + String(" failed CUresult=") + String(rc))


def cu_device_get(ordinal: Int32 = 0) raises -> Int32:
    var out = alloc[Int32](1)
    out[0] = 0
    var rc = Int(external_call["cuDeviceGet", Int32](_ptr(out), ordinal))
    var dev = out[0]
    out.free()
    check_cuda(rc, String("cuDeviceGet"))
    return dev


def cu_device_get_attribute(device: Int32, attribute: Int32) raises -> Int32:
    var out = alloc[Int32](1)
    out[0] = 0
    var rc = Int(external_call["cuDeviceGetAttribute", Int32](
        _ptr(out), attribute, device
    ))
    var value = out[0]
    out.free()
    check_cuda(rc, String("cuDeviceGetAttribute"))
    return value


def cu_device_total_mem(device: Int32) raises -> Int:
    var out = alloc[Int](1)
    out[0] = 0
    var rc = Int(external_call["cuDeviceTotalMem_v2", Int32](_ptr(out), device))
    var value = out[0]
    out.free()
    check_cuda(rc, String("cuDeviceTotalMem_v2"))
    return value


def cu_ctx_get_current() raises -> UInt64:
    var out = alloc[UInt64](1)
    out[0] = 0
    var rc = Int(external_call["cuCtxGetCurrent", Int32](_ptr(out)))
    var value = out[0]
    out.free()
    check_cuda(rc, String("cuCtxGetCurrent"))
    return value


@fieldwise_init
struct CuMemInfo(Copyable, Movable):
    """Free / total device memory in bytes (cuMemGetInfo_v2)."""
    var free_bytes: Int
    var total_bytes: Int

    def used_bytes(self) -> Int:
        return self.total_bytes - self.free_bytes


def cu_mem_get_info() raises -> CuMemInfo:
    """Query free + total device memory of the CURRENT context (cuMemGetInfo_v2).

    Used by offload memory smokes to report peak resident GPU bytes
    (peak = total - min(free) observed across the run). Requires a current
    CUDA context (DeviceContext construction establishes one)."""
    var free = alloc[Int](1)
    var total = alloc[Int](1)
    free[0] = 0
    total[0] = 0
    var rc = Int(external_call["cuMemGetInfo_v2", Int32](_ptr(free), _ptr(total)))
    var f = free[0]
    var t = total[0]
    free.free()
    total.free()
    check_cuda(rc, String("cuMemGetInfo_v2"))
    return CuMemInfo(f, t)


def cu_device_get_mempool(device: Int32 = 0) raises -> UInt64:
    """The device's DEFAULT stream-ordered memory pool handle
    (cuDeviceGetMemPool). MAX's DeviceContext caching allocator frees device
    buffers back to THIS pool (cuMemAllocFromPoolAsync / cuMemFreeAsync), and
    the pool keeps the bytes reserved from the OS until trimmed — this is the
    Phase-4 F3 "~GBs never returned between jobs" retention. The handle is a
    CUmemoryPool (an opaque pointer); we carry it as UInt64."""
    var out = alloc[UInt64](1)
    out[0] = 0
    var rc = Int(external_call["cuDeviceGetMemPool", Int32](_ptr(out), device))
    var pool = out[0]
    out.free()
    check_cuda(rc, String("cuDeviceGetMemPool"))
    return pool


def cu_mempool_trim_to(pool: UInt64, min_bytes_to_keep: Int) raises:
    """Release reserved-but-unused pool memory back to the OS, keeping at most
    `min_bytes_to_keep` bytes reserved (cuMemPoolTrimTo). Call this at a JOB
    BOUNDARY (no in-flight allocations) — it only reclaims chunks with no live
    suballocations, so live resident weights are untouched; the idle/freed
    high-water-mark bytes are what come back. This is the F3 pool-trim hook the
    daemon calls between jobs / on a model switch."""
    var rc = Int(external_call["cuMemPoolTrimTo", Int32](
        pool, UInt64(min_bytes_to_keep)
    ))
    check_cuda(rc, String("cuMemPoolTrimTo"))


def cu_mempool_trim_current(min_bytes_to_keep: Int = 0) raises:
    """Trim the current device's default pool to `min_bytes_to_keep` (default 0
    = release everything not live). Convenience wrapper used by the daemon's
    between-jobs / switch path."""
    var dev = cu_device_get(0)
    var pool = cu_device_get_mempool(dev)
    cu_mempool_trim_to(pool, min_bytes_to_keep)


def vmm_supported(device_ordinal: Int32 = 0) raises -> Bool:
    var dev = cu_device_get(device_ordinal)
    var supported = cu_device_get_attribute(
        dev, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED
    )
    return supported != 0


def make_allocation_prop(device_ordinal: Int32) -> BytePtr:
    """Allocate and fill CUDA 12.x CUmemAllocationProp (32 bytes).

    Caller owns the returned buffer and must `free()` it.
    Layout copied from Flame's `cuda_ffi.rs`:
      type:u32, requestedHandleTypes:u32, location{type:u32,id:i32},
      win32HandleMetaData:void*, allocFlags{u8,u8,u16,u8[4]}.
    """
    var buf = alloc[UInt8](32)
    var p = _ptr(buf)
    _zero(p, 32)
    _write_u32_le(p, 0, CU_MEM_ALLOCATION_TYPE_PINNED)
    _write_u32_le(p, 4, UInt32(0))
    _write_u32_le(p, 8, CU_MEM_LOCATION_TYPE_DEVICE)
    _write_i32_le(p, 12, device_ordinal)
    _write_u64_le(p, 16, UInt64(0))
    _write_u8(p, 24, UInt8(0))       # compressionType
    _write_u8(p, 25, UInt8(0))       # gpuDirectRDMACapable
    _write_u16_le(p, 26, UInt16(0))  # usage
    return p


def free_packed(buf: BytePtr):
    buf.free()


def make_access_desc(device_ordinal: Int32) -> BytePtr:
    """Allocate and fill CUDA CUmemAccessDesc (12 bytes).

    Caller owns the returned buffer and must `free()` it.
    """
    var buf = alloc[UInt8](12)
    var p = _ptr(buf)
    _zero(p, 12)
    _write_u32_le(p, 0, CU_MEM_LOCATION_TYPE_DEVICE)
    _write_i32_le(p, 4, device_ordinal)
    _write_u32_le(p, 8, CU_MEM_ACCESS_FLAGS_PROT_READWRITE)
    return p


def cu_mem_get_allocation_granularity(device_ordinal: Int32 = 0) raises -> Int:
    var gran = alloc[Int](1)
    gran[0] = 0
    var prop = make_allocation_prop(device_ordinal)
    var rc = Int(external_call["cuMemGetAllocationGranularity", Int32](
        _ptr(gran), prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED
    ))
    var value = gran[0]
    gran.free()
    free_packed(prop)
    check_cuda(rc, String("cuMemGetAllocationGranularity"))
    return value


def cu_mem_address_reserve(size: Int, alignment: Int) raises -> UInt64:
    var ptr_out = alloc[UInt64](1)
    ptr_out[0] = 0
    var rc = Int(external_call["cuMemAddressReserve", Int32](
        _ptr(ptr_out), size, alignment, UInt64(0), UInt64(0)
    ))
    var ptr = ptr_out[0]
    ptr_out.free()
    check_cuda(rc, String("cuMemAddressReserve"))
    return ptr


def cu_mem_address_free(ptr: UInt64, size: Int) raises:
    var rc = Int(external_call["cuMemAddressFree", Int32](ptr, size))
    check_cuda(rc, String("cuMemAddressFree"))


def cu_mem_create(size: Int, prop: BytePtr) raises -> UInt64:
    var handle_out = alloc[UInt64](1)
    handle_out[0] = 0
    var rc = Int(external_call["cuMemCreate", Int32](
        _ptr(handle_out), size, prop, UInt64(0)
    ))
    var handle = handle_out[0]
    handle_out.free()
    check_cuda(rc, String("cuMemCreate"))
    return handle


def cu_mem_release(handle: UInt64) raises:
    var rc = Int(external_call["cuMemRelease", Int32](handle))
    check_cuda(rc, String("cuMemRelease"))


def cu_mem_map(ptr: UInt64, size: Int, handle: UInt64) raises:
    var rc = Int(external_call["cuMemMap", Int32](
        ptr, size, Int(0), handle, UInt64(0)
    ))
    check_cuda(rc, String("cuMemMap"))


def cu_mem_unmap(ptr: UInt64, size: Int) raises:
    var rc = Int(external_call["cuMemUnmap", Int32](ptr, size))
    check_cuda(rc, String("cuMemUnmap"))


def cu_mem_set_access(ptr: UInt64, size: Int, desc: BytePtr) raises:
    var rc = Int(external_call["cuMemSetAccess", Int32](
        ptr, size, desc, Int(1)
    ))
    check_cuda(rc, String("cuMemSetAccess"))


def cu_event_create_disable_timing() raises -> UInt64:
    var event_out = alloc[UInt64](1)
    event_out[0] = 0
    var rc = Int(external_call["cuEventCreate", Int32](
        _ptr(event_out), CU_EVENT_DISABLE_TIMING
    ))
    var event = event_out[0]
    event_out.free()
    check_cuda(rc, String("cuEventCreate"))
    return event


def cu_event_record(event: UInt64, stream: UInt64 = UInt64(0)) raises:
    var rc = Int(external_call["cuEventRecord", Int32](event, stream))
    check_cuda(rc, String("cuEventRecord"))


def cu_event_synchronize(event: UInt64) raises:
    var rc = Int(external_call["cuEventSynchronize", Int32](event))
    check_cuda(rc, String("cuEventSynchronize"))


def cu_event_destroy(event: UInt64) raises:
    var rc = Int(external_call["cuEventDestroy_v2", Int32](event))
    check_cuda(rc, String("cuEventDestroy_v2"))


def cu_stream_wait_event(stream: UInt64, event: UInt64) raises:
    var rc = Int(external_call["cuStreamWaitEvent", Int32](
        stream, event, UInt32(0)
    ))
    check_cuda(rc, String("cuStreamWaitEvent"))
