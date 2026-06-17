# vmm_manager_smoke.mojo - VMM model/block handle smoke.

from std.collections import List
from std.gpu.host import DeviceContext

from serenitymojo.io.dtype import STDtype
from serenitymojo.offload.vmm_cuda import vmm_supported
from serenitymojo.offload.vmm_manager import VmmModelManager


def _check(name: String, got: Int, expected: Int) raises:
    print("[vmm-manager]", name, "got=", got, "expected=", expected)
    if got != expected:
        raise Error(String("vmm manager mismatch: ") + name)


def _check_bool(name: String, got: Bool, expected: Bool) raises:
    print("[vmm-manager]", name, "got=", got, "expected=", expected)
    if got != expected:
        raise Error(String("vmm manager mismatch: ") + name)


def main() raises:
    var ctx = DeviceContext()
    if not vmm_supported(0):
        print("[vmm-manager] SKIP: CUDA VMM not supported")
        return

    var sizes = List[Int]()
    sizes.append(1024 * 1024)
    sizes.append(2 * 1024 * 1024)
    sizes.append(1024 * 1024)

    var manager = VmmModelManager(0)
    var handle = manager.register_model(String("vmm-manager-smoke"), sizes^, STDtype.BF16)
    _check(String("registered models"), manager.registered_models, 1)
    _check(String("block count"), handle.block_count(), 3)
    _check(String("requested bytes"), handle.requested_bytes(), 4 * 1024 * 1024)
    _check(String("resident initially"), handle.resident_bytes(), 0)
    _check_bool(String("block0 populated initially"), handle.is_block_populated(0), False)

    var p0 = handle.ensure_block_resident(0)
    if p0 == UInt64(0):
        raise Error("vmm manager: null resident pointer")
    _check_bool(String("block0 resident"), handle.is_block_resident(0), True)
    _check(String("block0 refcount"), handle.block_refcount(0), 1)
    _check(String("resident after block0"), handle.resident_bytes(), handle.block_reserved_bytes(0))

    var p0_again = handle.ensure_block_resident(0)
    if p0_again != p0:
        raise Error("vmm manager: resident pointer changed")
    _check(String("block0 refcount x2"), handle.block_refcount(0), 2)
    handle.mark_block_populated(0)
    _check_bool(String("block0 populated"), handle.is_block_populated(0), True)
    handle.release_block(0)
    handle.release_block(0)
    handle.evict_block(0)
    _check_bool(String("block0 evicted"), handle.is_block_resident(0), False)
    _check_bool(String("block0 evict clears populated"), handle.is_block_populated(0), False)
    _check(String("resident after block0 evict"), handle.resident_bytes(), 0)

    var p1 = handle.ensure_block_resident(1)
    if p1 == UInt64(0):
        raise Error("vmm manager: null block1 pointer")
    handle.mark_block_populated(1)
    _check_bool(String("block1 populated"), handle.is_block_populated(1), True)
    handle.release_block(1)
    handle.destroy()
    _check_bool(String("destroyed"), handle.destroyed, True)
    print("[vmm-manager] PASS")
