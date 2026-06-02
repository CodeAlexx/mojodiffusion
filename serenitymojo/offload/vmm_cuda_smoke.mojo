# vmm_cuda_smoke.mojo - CUDA VMM ABI probe.
#
# This is intentionally small: it initializes a DeviceContext, probes VMM
# support/granularity, reserves and frees one VA range, and maps/unmaps a tiny
# physical allocation when supported. It does not implement residency policy.

from std.gpu.host import DeviceContext

from serenitymojo.offload.vmm_cuda import (
    vmm_supported,
    cu_device_get,
    cu_device_total_mem,
    cu_ctx_get_current,
    cu_mem_get_allocation_granularity,
    cu_mem_address_reserve,
    cu_mem_address_free,
    cu_mem_create,
    cu_mem_release,
    cu_mem_map,
    cu_mem_unmap,
    cu_mem_set_access,
    cu_event_create_disable_timing,
    cu_event_record,
    cu_event_synchronize,
    cu_event_destroy,
    make_allocation_prop,
    make_access_desc,
    free_packed,
)


def _round_up(n: Int, alignment: Int) -> Int:
    var r = n % alignment
    if r == 0:
        return n
    return n + alignment - r


def main() raises:
    var ctx = DeviceContext()
    var dev = cu_device_get(0)
    var total = cu_device_total_mem(dev)
    var current_ctx = cu_ctx_get_current()
    var supported = vmm_supported(0)
    print("[vmm-cuda] device", dev, "total_mib", Float64(total) / 1048576.0)
    print("[vmm-cuda] current_ctx", current_ctx, "supported", supported)
    if not supported:
        print("[vmm-cuda] VMM not supported on this device/driver")
        return

    var gran = cu_mem_get_allocation_granularity(0)
    var size = _round_up(4 * 1024 * 1024, gran)
    print("[vmm-cuda] granularity", gran, "probe_size", size)

    var va = cu_mem_address_reserve(size, gran)
    print("[vmm-cuda] reserved_va", va)
    var prop = make_allocation_prop(0)
    var desc = make_access_desc(0)
    var handle = cu_mem_create(size, prop)
    cu_mem_map(va, size, handle)
    cu_mem_set_access(va, size, desc)
    cu_mem_unmap(va, size)
    cu_mem_release(handle)
    cu_mem_address_free(va, size)
    free_packed(desc)
    free_packed(prop)
    var ev = cu_event_create_disable_timing()
    cu_event_record(ev)
    cu_event_synchronize(ev)
    cu_event_destroy(ev)
    print("[vmm-cuda] event lifecycle PASS")
    print("[vmm-cuda] PASS")
