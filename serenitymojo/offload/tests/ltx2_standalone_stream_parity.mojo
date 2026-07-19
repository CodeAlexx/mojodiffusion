# offload/tests/ltx2_standalone_stream_parity.mojo — STANDALONE-buffer loader gate.
#
# Same coverage as ltx2_fixed_stream_parity but for the capture-mode STANDALONE
# weight stage (one device buffer PER tensor, offset 0), which is the topology a
# captured CUDA graph reliably re-reads under an external per-block refill (see
# LTX2StandaloneStage + ltx2_capture_external_refill_constraint). Prove
# load_block_bf16_standalone lands byte-identical weights to the DEFAULT
# load_block_bf16, with per-tensor addresses stable across visits.
#
# Blocks 0, 24, 47: load DEFAULT and STANDALONE, byte-compare every tensor
# (n_mismatch==0, block NONZERO), back-to-back through the SAME stage (content
# signatures prove the blocks differ so a stale-buffer bug can't pass), each
# tensor's device address recorded on the first load and asserted stable.
#
#   rm -f serenitymojo.mojopkg && pixi run mojo build -O2 -I . \
#     -Xlinker -lm -Xlinker -lcuda -Xlinker -L.pixi/envs/default/lib \
#     -Xlinker -lsqlite3 -Xlinker -Lserenitymojo/ops/cshim/lib \
#     -Xlinker -lserenity_cudnn_sdpa \
#     serenitymojo/offload/tests/ltx2_standalone_stream_parity.mojo -o /tmp/ltx2_standalone_stream_parity
#   env LD_LIBRARY_PATH=.pixi/envs/default/lib:serenitymojo/ops/cshim/lib:\
#     $HOME/.local/lib/python3.12/site-packages/nvidia/cudnn/lib \
#     /tmp/ltx2_standalone_stream_parity
#   # Optional >24 GB residency arm (not part of the 16 GB product gate):
#   # /tmp/ltx2_standalone_stream_parity resident

from std.gpu.host import DeviceContext
from sys import argv
from serenitymojo.offload.ltx2_block_stream import (
    LTX2BlockStream, LTX2StandaloneStage, FP8Block,
)

comptime CKPT = "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-distilled-fp8.safetensors"


def _cmp(a: List[Float32], b: List[Float32]) -> Tuple[Int, Bool]:
    if len(a) != len(b):
        return (-1, False)
    var nm = 0
    var nz = False
    for i in range(len(a)):
        if a[i] != b[i]:
            nm += 1
        if a[i] != Float32(0.0):
            nz = True
    return (nm, nz)


def _compare_block(
    label: String,
    default_blk: FP8Block,
    sa_blk: FP8Block,
    ctx: DeviceContext,
    mut allok: Bool,
    mut sig_out: Float64,
) raises -> Dict[String, Int]:
    """Byte-compare every tensor of DEFAULT vs STANDALONE and return name ->
    device address (for the cross-visit stability check). Folds a weighted content
    signature of DEFAULT into sig_out."""
    var addrs = Dict[String, Int]()
    var n_tensors = 0
    var n_mismatch_total = 0
    var n_nonzero = 0
    var sig = Float64(0.0)
    if len(sa_blk) != len(default_blk):
        print("   FAIL:", label, "tensor-count mismatch default=",
              len(default_blk), "standalone=", len(sa_blk))
        allok = False
    for ref e in default_blk.items():
        var key = e.key
        n_tensors += 1
        if key not in sa_blk:
            print("   FAIL:", label, "standalone block missing tensor", key)
            allok = False
            continue
        var da = default_blk[key][].to_host(ctx)
        var fa = sa_blk[key][].to_host(ctx)
        var r = _cmp(da, fa)
        var nm = r[0]
        var nz = r[1]
        if nm != 0:
            n_mismatch_total += nm
            allok = False
            print("   FAIL:", label, key, " n_mismatch=", nm, " n=", len(da))
        if nz:
            n_nonzero += 1
        var step = 1 + (len(da) // 64)
        var si = 0
        while si < len(da):
            sig += Float64(da[si]) * Float64(si + 1)
            si += step
        addrs[key] = Int(sa_blk[key][].buf.unsafe_ptr())
    sig_out = sig
    var ok = (n_mismatch_total == 0) and (n_nonzero > 0) and (n_tensors > 0)
    var verdict = "PASS" if ok else "FAIL"
    print("  ", verdict, label, " tensors=", n_tensors,
          " n_mismatch_total=", n_mismatch_total, " nonzero_tensors=", n_nonzero)
    if n_nonzero == 0:
        print("   FAIL:", label, "block is all-zero — degenerate compare")
        allok = False
    return addrs^


def main() raises:
    print("=== LTX2 STANDALONE-buffer block loader BIT gate (standalone vs default streamed) ===")
    var ctx = DeviceContext()
    var stream = LTX2BlockStream.open(CKPT)
    var nb = stream.block_count()
    print("  block_count =", nb)
    if nb < 48:
        raise Error(
            String("ltx2_standalone_stream_parity: expected >=48 video blocks, got ")
            + String(nb)
        )

    var stage = stream.build_standalone_stage(ctx)
    print("  [byte-math] standalone stage: entries=", stage.n_entries(),
          " per-block bf16 footprint=", stage.total_bytes(), "bytes (",
          stage.total_bytes() // (1024 * 1024), "MiB, ", stage.n_entries(), "standalone buffers )")

    var allok = True
    var blocks = [0, 24, 47]
    var ref_addrs = Dict[String, Int]()
    var sig0 = Float64(0.0)
    for bi in range(len(blocks)):
        var blk = blocks[bi]
        print(" -- block", blk, "--")
        var d = stream.load_block_bf16(blk, ctx)
        var f = stream.load_block_bf16_standalone(blk, stage, ctx)
        var sig = Float64(0.0)
        var addrs = _compare_block(String("blk") + String(blk), d, f, ctx, allok, sig)
        if bi == 0:
            ref_addrs = addrs^
            sig0 = sig
        else:
            if sig == sig0:
                print("   FAIL: blk", blk, "content signature == block 0 (", sig,
                      ") — stale-buffer catch would pass silently")
                allok = False
            else:
                print("   PASS blk", blk, "content differs from block 0 (sig", sig,
                      "vs", sig0, ") — stale-buffer catch is meaningful")
            var drift = 0
            var missing = 0
            for ref e in addrs.items():
                if e.key not in ref_addrs:
                    missing += 1
                    allok = False
                elif ref_addrs[e.key] != e.value:
                    drift += 1
                    allok = False
            if drift != 0 or missing != 0:
                print("   FAIL: blk", blk, "address stability: drift=", drift,
                      " missing_from_ref=", missing)
            else:
                print("   PASS blk", blk, "addresses identical to block 0 (",
                      len(addrs), "standalone buffers )")

    var args = argv()
    if len(args) < 2 or args[1] != String("resident"):
        if allok:
            print("GATE ltx2_standalone_stream_parity: ALL PASS",
                  "(standalone == default streamed, byte-exact;",
                  "addresses stable streamed 0->24->47; resident arm skipped",
                  "because the product target is 16 GB; pass 'resident' on >24 GB)")
            return
        print("GATE ltx2_standalone_stream_parity: FAIL")
        raise Error("ltx2_standalone_stream_parity gate FAILED")

    # ── OPTIONAL RESIDENT arm: enable the fp8 pool so
    #    load_block_bf16_standalone dispatches
    #    to _load_block_bf16_standalone_resident (device-only dequant, no H2D). Reuse
    #    the SAME stage: the resident refill must land byte-identical to the default
    #    (also resident) AND at the SAME device addresses the streamed phase recorded
    #    (a fresh-buffer re-point would break capture). This is the branch the speed
    #    fix added; it MUST be exercised and byte-exact. ─────────────────────────
    print(" == RESIDENT arm (enable_fp8_resident_range 0..", nb - 1, ") ==")
    stream.enable_fp8_resident_range(0, nb - 1, ctx)
    for bi in range(len(blocks)):
        var blk = blocks[bi]
        print(" -- resident block", blk, "--")
        var d = stream.load_block_bf16(blk, ctx)                    # resident oracle
        var f = stream.load_block_bf16_standalone(blk, stage, ctx)  # resident NEW branch
        var sig = Float64(0.0)
        var addrs = _compare_block(String("res_blk") + String(blk), d, f, ctx, allok, sig)
        var drift = 0
        var missing = 0
        for ref e in addrs.items():
            if e.key not in ref_addrs:
                missing += 1
                allok = False
            elif ref_addrs[e.key] != e.value:
                drift += 1
                allok = False
        if drift != 0 or missing != 0:
            print("   FAIL: resident blk", blk, "address drift=", drift,
                  " missing_from_ref=", missing)
        else:
            print("   PASS resident blk", blk, "addresses identical to the streamed",
                  "phase (", len(addrs), "standalone buffers — resident refill in place )")

    if allok:
        print("GATE ltx2_standalone_stream_parity: ALL PASS",
              "(standalone == default streamed AND resident, byte-exact;",
              "addresses stable streamed 0->24->47 and through the resident refill)")
    else:
        print("GATE ltx2_standalone_stream_parity: FAIL")
        raise Error("ltx2_standalone_stream_parity gate FAILED")
