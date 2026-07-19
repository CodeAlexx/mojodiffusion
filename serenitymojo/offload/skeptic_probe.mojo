# skeptic_probe.mojo — adversarial probe for BlockLoader (SKEPTIC review 2026-05-25).
#
# NOT scope code. Exercises the scenarios the smoke test skipped:
#   P1. VRAM leak across ALL 30 transformer layers (load->unload each), trace free VRAM.
#   P2. Prefix edge cases: layers.1. vs layers.1{0..9}., no-match prefix, count-per-prefix.
#   P3. ArcPointer keep-alive: hold one tensor out of a block, unload the block,
#       is the held tensor still valid? double-unload? double-load (two live blocks)?
#   P4. Reload stress: many load/unload cycles on the io re-open path.
#
# Run: cd /home/alex/mojodiffusion && pixi run mojo run -I . serenitymojo/offload/skeptic_probe.mojo

from std.memory import ArcPointer
from std.gpu.host import DeviceContext
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.offload.block_loader import BlockLoader, Block, unload_block
from serenitymojo.tensor import Tensor


comptime TRANSFORMER_DIR = (
    "/home/alex/.cache/huggingface/hub/models--Tongyi-MAI--Z-Image/"
    "snapshots/04cc4abb7c5069926f75c9bfde9ef43d49423021/transformer"
)

comptime MB = 1024 * 1024


def _free_mb(ctx: DeviceContext) raises -> Int:
    var mi = ctx.get_memory_info()
    return Int(mi[0] // MB)


def p1_leak_trace(ctx: DeviceContext) raises:
    print("=== P1: VRAM leak trace across 30 layers (load->unload each) ===")
    var loader = BlockLoader.open(String(TRANSFORMER_DIR))
    var baseline = _free_mb(ctx)
    print("baseline free MB (no block loaded):", baseline)
    print("idx, free_after_load_MB, loaded_delta_MB, free_after_unload_MB, residual_vs_baseline_MB, tensors")
    var worst_residual = 0
    for i in range(30):
        var prefix = String("layers.") + String(i) + String(".")
        var b = loader.load_block(prefix, ctx)
        var n = len(b)
        var free_loaded = _free_mb(ctx)
        var loaded_delta = baseline - free_loaded
        unload_block(b^)
        ctx.synchronize()
        var free_unloaded = _free_mb(ctx)
        var residual = baseline - free_unloaded
        if residual > worst_residual:
            worst_residual = residual
        print(i, ",", free_loaded, ",", loaded_delta, ",", free_unloaded, ",", residual, ",", n)
    var final_free = _free_mb(ctx)
    print("final free MB:", final_free, " baseline:", baseline,
          " net drift MB:", baseline - final_free, " worst residual MB:", worst_residual)
    print("LEAK VERDICT (net drift should be ~0; small constant = allocator, growth = leak)")


def p2_prefix_edges(ctx: DeviceContext) raises:
    print("\n=== P2: prefix edge cases ===")
    var loader = BlockLoader.open(String(TRANSFORMER_DIR))

    # count-per-prefix should all be 15 (Python ground truth)
    var all15 = True
    for i in range(30):
        var prefix = String("layers.") + String(i) + String(".")
        var c = loader.block_count_for(prefix)
        if c != 15:
            print("  layer", i, "count", c, "!= 15")
            all15 = False
    print("all 30 layer prefixes count == 15:", all15)

    # trailing-dot: layers.1. must capture ONLY layer 1 (not 10..19)
    var b1 = loader.load_block(String("layers.1."), ctx)
    var only_layer1 = True
    for ref e in b1.items():
        var nm = e.key
        if not nm.startswith(String("layers.1.")):
            only_layer1 = False
        # any name like layers.1X. would still startswith layers.1. only if X.. — check it isn't a 2-digit
        # extract char after 'layers.1' (index 8); if it's a digit, it's 1X -> leak
        if nm.byte_length() > 8:
            var c8 = nm[byte = 8]
            if c8 >= "0" and c8 <= "9":
                print("  LEAK: layers.1. captured a 2-digit-layer name:", nm)
                only_layer1 = False
    print("layers.1. count:", len(b1), " captured only layer 1:", only_layer1)
    unload_block(b1^)

    # no-match prefix: clean empty Dict or crash?
    var bz = loader.load_block(String("layers.999."), ctx)
    print("layers.999. (no match) -> count:", len(bz), "(expect 0, no crash)")
    unload_block(bz^)

    var bnone = loader.load_block(String("zzznotaprefix"), ctx)
    print("'zzznotaprefix' (no match) -> count:", len(bnone), "(expect 0)")
    unload_block(bnone^)

    # substring-not-from-start: 'attn' appears inside names but should match 0 at start
    var bsub = loader.load_block(String("attn"), ctx)
    print("'attn' (substring not at start) -> count:", len(bsub), "(expect 0; startswith only)")
    unload_block(bsub^)

    # empty prefix: matches EVERYTHING (Rust: starts_with("") is true for all)
    var call_count = loader.block_count_for(String(""))
    print("empty-prefix block_count_for(''):", call_count, "(expect 521 = ALL tensors)")


def p3_arc_lifetime(ctx: DeviceContext) raises:
    print("\n=== P3: ArcPointer lifetime / double-unload / double-load ===")
    var loader = BlockLoader.open(String(TRANSFORMER_DIR))

    # Hold one tensor out of the block, then unload the block. Is it still valid?
    var b = loader.load_block(String("layers.0."), ctx)
    # grab a copy of one ArcPointer (refcount bump) for a known name
    var held = Block()  # stash the single held arc so it outlives b
    var picked = String("")
    for ref e in b.items():
        picked = e.key
        held[e.key] = e.value  # ArcPointer copy == refcount bump
        break
    print("held tensor:", picked)
    var held_nbytes_before = held[picked][].nbytes()
    unload_block(b^)  # drop the block; held arc keeps that one tensor alive
    ctx.synchronize()
    # read back the held tensor — if it were freed, this would corrupt/crash
    var hostv = held[picked][].to_host(ctx)
    var nb_after = held[picked][].nbytes()
    print("  after unloading block, held tensor nbytes:", nb_after,
          " (before:", held_nbytes_before, ") to_host len:", len(hostv))
    print("  held-tensor-survives-block-unload:", nb_after == held_nbytes_before and len(hostv) > 0)
    # now drop held
    unload_block(held^)
    ctx.synchronize()

    # double-load same prefix without unloading: two independent blocks?
    var free0 = _free_mb(ctx)
    var ba = loader.load_block(String("layers.0."), ctx)
    var free1 = _free_mb(ctx)
    var bb = loader.load_block(String("layers.0."), ctx)
    var free2 = _free_mb(ctx)
    var d1 = free0 - free1
    var d2 = free1 - free2
    print("double-load layers.0.: first delta MB:", d1, " second delta MB:", d2,
          " (independent copies -> ~equal, VRAM doubles)")
    print("  block A count:", len(ba), " block B count:", len(bb))
    unload_block(ba^)
    unload_block(bb^)
    ctx.synchronize()
    var free3 = _free_mb(ctx)
    print("  after both unloaded, free MB:", free3, " vs start", free0,
          " net:", free0 - free3)


def p4_reload_stress(ctx: DeviceContext) raises:
    print("\n=== P4: reload stress (50 cycles on io re-open path) ===")
    var loader = BlockLoader.open(String(TRANSFORMER_DIR))
    var baseline = _free_mb(ctx)
    var ok = True
    var last_n = -1
    for cyc in range(50):
        var i = cyc % 30
        var prefix = String("layers.") + String(i) + String(".")
        var b = loader.load_block(prefix, ctx)
        var n = len(b)
        if n != 15:
            print("  cycle", cyc, "prefix", prefix, "count", n, "!= 15")
            ok = False
        last_n = n
        unload_block(b^)
        ctx.synchronize()
    var final_free = _free_mb(ctx)
    print("50 cycles done. all counts==15:", ok, " net drift MB:", baseline - final_free,
          " last_n:", last_n)


def main() raises:
    var ctx = DeviceContext()
    p1_leak_trace(ctx)
    p2_prefix_edges(ctx)
    p3_arc_lifetime(ctx)
    p4_reload_stress(ctx)
    print("\n=== PROBE COMPLETE ===")
