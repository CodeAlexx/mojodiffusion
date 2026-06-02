# offload_smoke.mojo — correctness smoke test for the BlockLoader.
#
# Not numerical parity — CORRECTNESS of the streaming mechanism:
#   1. open the Z-Image *transformer* dir (521 tensors, 2 shards via index).
#   2. load_block("layers.0.") -> print the block's tensor count + names.
#   3. verify each loaded Tensor's bytes match a DIRECT
#      ShardedSafeTensors.tensor_bytes(name) (same source -> must match exactly).
#   4. unload (drop the block).
#   5. load_block("layers.1."), then load_block("layers.0.") AGAIN — proves the
#      re-open works (depends on the io F1 NUL-term fix).
#   6. (optional) print free VRAM before/after unload to show unload frees VRAM.
#
# Mojo 1.0.0b1, Linux x86-64, NVIDIA. Run:
#   pixi run mojo run -I . serenitymojo/offload/offload_smoke.mojo

from std.gpu.host import DeviceContext
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.offload.block_loader import BlockLoader, Block, unload_block


comptime TRANSFORMER_DIR = (
    "/home/alex/.cache/huggingface/hub/models--Tongyi-MAI--Z-Image/"
    "snapshots/04cc4abb7c5069926f75c9bfde9ef43d49423021/transformer"
)


def _sorted_names(var names: List[String]) -> List[String]:
    """Insertion sort (small N) so block name listing is deterministic."""
    for i in range(1, len(names)):
        var key = names[i]
        var j = i - 1
        while j >= 0 and names[j] > key:
            names[j + 1] = names[j]
            j -= 1
        names[j + 1] = key
    return names^


def _verify_block_bytes_match(
    ref block: Block,
    ref sharded: ShardedSafeTensors,
    ctx: DeviceContext,
) raises -> Bool:
    """For every tensor in the block, read it back from the device and compare
    every byte to a DIRECT ShardedSafeTensors.tensor_bytes(name). Same source
    bytes -> must be identical. Returns True iff all match."""
    var all_ok = True
    for ref e in block.items():
        var name = e.key
        # Device -> host readback as the stored compute dtype, re-packed to bytes.
        # Simpler: compare via raw bytes. Tensor stores raw element bytes, so we
        # round-trip to host F32 only if needed; instead compare the on-disk
        # bytes against a host copy of the device buffer.
        var direct = sharded.tensor_bytes(name)  # Span[UInt8] over the mmap
        var host = ctx.enqueue_create_host_buffer[DType.uint8](e.value[].nbytes())
        ctx.enqueue_copy(dst_buf=host, src_buf=e.value[].buf)
        ctx.synchronize()
        var hp = host.unsafe_ptr()
        var nb = e.value[].nbytes()
        if nb != len(direct):
            print("  BYTE-LEN MISMATCH", name, "dev", nb, "disk", len(direct))
            all_ok = False
            continue
        var mismatch = False
        for i in range(nb):
            if hp[i] != direct[i]:
                mismatch = True
                break
        if mismatch:
            print("  BYTE MISMATCH in", name)
            all_ok = False
    return all_ok


def main() raises:
    var ctx = DeviceContext()
    print("=== BlockLoader smoke: Z-Image transformer ===")
    var loader = BlockLoader.open(String(TRANSFORMER_DIR))

    # Sanity on the underlying sharded loader.
    var st = ShardedSafeTensors.open(String(TRANSFORMER_DIR))
    print("num_shards =", st.num_shards(), " num_tensors =", st.num_tensors())

    # --- VRAM before ---
    var mem_before = ctx.get_memory_info()
    print("free VRAM before load (bytes):", mem_before[0])

    # --- prefetch_block (warm page cache; MADV_WILLNEED) ---
    loader.prefetch_block(String("layers.0."))
    print("prefetch_block('layers.0.') OK")

    # --- load_block("layers.0.") ---
    var b0 = loader.load_block(String("layers.0."), ctx)
    print("load_block('layers.0.') -> tensor count:", len(b0))
    var names0 = List[String]()
    for ref e in b0.items():
        names0.append(e.key)
    names0 = _sorted_names(names0^)
    print("  names:")
    for ref nm in names0:
        print("    ", nm)

    # --- byte-match vs direct load ---
    var ok0 = _verify_block_bytes_match(b0, st, ctx)
    print("  byte-match vs direct ShardedSafeTensors.tensor_bytes:", ok0)

    var mem_loaded = ctx.get_memory_info()
    print("free VRAM after load (bytes):", mem_loaded[0])
    print("  delta (load) =", mem_before[0] - mem_loaded[0], "bytes")

    # --- unload (explicit drop) ---
    var n0 = len(b0)
    unload_block(b0^)
    ctx.synchronize()
    var mem_unloaded = ctx.get_memory_info()
    print("free VRAM after unload (bytes):", mem_unloaded[0])
    print("  reclaimed =", mem_unloaded[0] - mem_loaded[0], "bytes")

    # --- load_block("layers.1.") then load_block("layers.0.") AGAIN ---
    # Proves the re-open / re-enumeration works (io F1 NUL-term dependency).
    var b1 = loader.load_block(String("layers.1."), ctx)
    var n1 = len(b1)
    print("load_block('layers.1.') -> tensor count:", n1)
    var ok1 = _verify_block_bytes_match(b1, st, ctx)
    print("  byte-match vs direct:", ok1)
    unload_block(b1^)

    var b0_again = loader.load_block(String("layers.0."), ctx)
    print("load_block('layers.0.') AGAIN -> tensor count:", len(b0_again))
    var ok0b = _verify_block_bytes_match(b0_again, st, ctx)
    print("  byte-match vs direct (reload):", ok0b)
    var n0_again = len(b0_again)
    unload_block(b0_again^)

    # --- summary ---
    print("=== SUMMARY ===")
    var first_reload_match = n0 == n0_again
    print("layers.0. count first/reload:", n0, "/", n0_again, " match:", first_reload_match)
    print("layers.1. count:", n1)
    var all_pass = ok0 and ok1 and ok0b and first_reload_match
    print("ALL CORRECTNESS CHECKS PASS:", all_pass)
    if not all_pass:
        raise Error("offload_smoke: a correctness check FAILED")
    _ = st.num_shards()
