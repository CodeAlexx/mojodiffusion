# skeptic_probe2.mojo — tighter checks: held-tensor BYTE correctness after unload,
# double-unload safety. NOT scope code.
# Run: cd /home/alex/mojodiffusion && pixi run mojo run -I . serenitymojo/offload/skeptic_probe2.mojo

from std.gpu.host import DeviceContext
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.offload.block_loader import BlockLoader, Block, unload_block


comptime TRANSFORMER_DIR = (
    "/home/alex/.cache/huggingface/hub/models--Tongyi-MAI--Z-Image/"
    "snapshots/04cc4abb7c5069926f75c9bfde9ef43d49423021/transformer"
)


def main() raises:
    var ctx = DeviceContext()
    var loader = BlockLoader.open(String(TRANSFORMER_DIR))
    var st = ShardedSafeTensors.open(String(TRANSFORMER_DIR))

    # Hold one tensor, capture its bytes BEFORE unload; unload block; re-read AFTER
    # unload; compare to disk. A silent use-after-free could keep nbytes but corrupt data.
    var b = loader.load_block(String("layers.0."), ctx)
    var held = Block()
    var picked = String("")
    for ref e in b.items():
        picked = e.key
        held[e.key] = e.value
        break

    # disk ground truth
    var disk = st.tensor_bytes(picked)
    var dn = len(disk)

    # device readback BEFORE unloading the block
    var nb = held[picked][].nbytes()
    var host_before = ctx.enqueue_create_host_buffer[DType.uint8](nb)
    ctx.enqueue_copy(dst_buf=host_before, src_buf=held[picked][].buf)
    ctx.synchronize()
    var hb = host_before.unsafe_ptr()
    var match_before = (nb == dn)
    if match_before:
        for i in range(nb):
            if hb[i] != disk[i]:
                match_before = False
                break
    print("held tensor:", picked, " nbytes:", nb, " disk:", dn)
    print("  bytes match disk BEFORE block unload:", match_before)

    # now UNLOAD the block; held arc should keep the tensor alive & intact
    unload_block(b^)
    ctx.synchronize()

    var host_after = ctx.enqueue_create_host_buffer[DType.uint8](nb)
    ctx.enqueue_copy(dst_buf=host_after, src_buf=held[picked][].buf)
    ctx.synchronize()
    var ha = host_after.unsafe_ptr()
    var disk2 = st.tensor_bytes(picked)
    var match_after = (nb == len(disk2))
    if match_after:
        for i in range(nb):
            if ha[i] != disk2[i]:
                match_after = False
                break
    print("  bytes match disk AFTER block unload (held arc alive):", match_after)
    print("  HELD-TENSOR-INTACT-AFTER-UNLOAD:", match_before and match_after)
    unload_block(held^)
    ctx.synchronize()

    # Double-unload: load, unload, then attempt to unload again. The second unload
    # operates on a SEPARATE empty Block (we cannot call unload_block(b^) twice —
    # the compiler consumes b). Simulate "double free" intent: rebind then drop.
    var b2 = loader.load_block(String("layers.2."), ctx)
    unload_block(b2^)
    ctx.synchronize()
    # b2 is consumed; re-load into the same name and unload again — exercises
    # repeated free of the loader's resources without crash.
    var b2b = loader.load_block(String("layers.2."), ctx)
    print("re-load layers.2. after unload -> count:", len(b2b), "(no double-free crash)")
    unload_block(b2b^)
    ctx.synchronize()

    # Empty-block unload (no-match prefix): drop an empty Block — must not crash.
    var be = loader.load_block(String("layers.999."), ctx)
    print("empty block count:", len(be))
    unload_block(be^)
    ctx.synchronize()
    print("empty-block unload: OK (no crash)")

    _ = st.num_shards()
    print("=== PROBE2 COMPLETE ===")
