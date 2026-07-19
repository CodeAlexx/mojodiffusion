# vmm_slab_smoke.mojo - physical VMM slab primitive smoke.

from std.gpu.host import DeviceContext

from serenitymojo.offload.vmm_cuda import vmm_supported
from serenitymojo.offload.vmm_slab import VmmSlabAllocator


def _check(name: String, got: Int, expected: Int) raises:
    print("[vmm-slab]", name, "got=", got, "expected=", expected)
    if got != expected:
        raise Error(String("vmm slab mismatch: ") + name)


def _check_bool(name: String, got: Bool, expected: Bool) raises:
    print("[vmm-slab]", name, "got=", got, "expected=", expected)
    if got != expected:
        raise Error(String("vmm slab mismatch: ") + name)


def main() raises:
    var ctx = DeviceContext()
    if not vmm_supported(0):
        print("[vmm-slab] SKIP: CUDA VMM not supported")
        return

    var slab = VmmSlabAllocator.create(8 * 1024 * 1024, 0)
    print("[vmm-slab] base", slab.base_ptr, "size", slab.total_size, "gran", slab.granularity)
    var r0 = slab.define_region(1024 * 1024)
    var r1 = slab.define_region(1024 * 1024)
    _check(String("region0 id"), r0, 0)
    _check(String("region1 id"), r1, 1)
    _check(String("mapped initially"), slab.mapped_bytes, 0)

    var p0 = slab.ensure_resident(r0)
    _check_bool(String("region0 resident"), slab.regions[r0].resident, True)
    _check(String("region0 refcount"), slab.regions[r0].refcount, 1)
    _check(String("mapped after r0"), slab.mapped_bytes, slab.regions[r0].size)
    var p0_again = slab.ensure_resident(r0)
    if p0_again != p0:
        raise Error("vmm slab: resident pointer changed")
    _check(String("region0 refcount x2"), slab.regions[r0].refcount, 2)
    slab.release(r0)
    slab.release(r0)
    slab.evict(r0)
    _check_bool(String("region0 evicted"), slab.regions[r0].resident, False)
    _check(String("mapped after r0 evict"), slab.mapped_bytes, 0)

    var p1 = slab.ensure_resident(r1)
    if p1 == UInt64(0):
        raise Error("vmm slab: null region pointer")
    slab.release(r1)
    slab.destroy()
    _check(String("mapped after destroy"), slab.mapped_bytes, 0)
    print("[vmm-slab] PASS")
