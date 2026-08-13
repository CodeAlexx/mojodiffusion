# offload/tests/ltx2_fixed_stream_parity.mojo — FIXED-buffer block loader BIT gate.
#
# Stage (b) of the LTX2 capture leg: prove LTX2BlockStream.load_block_bf16_fixed
# (stream weights into ONE fixed device buffer, return create_sub_buffer views)
# lands byte-identical weights to the DEFAULT load_block_bf16 (fresh per-tensor
# allocs), AND that the fixed buffer's addresses are stable across visits — the
# hard precondition for CUDA-graph replay (contract C8, AUTOGRAD_V2_MOJO_DESIGN.md:
# replay re-runs recorded kernels against recorded pointers).
#
# For blocks 0, 24, 47 (all 48 video blocks shape-identical): load DEFAULT and
# FIXED, byte-compare every tensor (n_mismatch==0, block NONZERO so a degenerate
# all-zero compare fails). The three fixed loads run BACK-TO-BACK through the SAME
# stage (0 → 24 → 47), comparing AFTER each refill — this is what catches a
# stale-buffer bug (block 24's compare must see 24's bytes, not 0's). Each
# tensor's device address is recorded on the first fixed load and asserted equal
# on every later refill.
#
#   rm -f serenitymojo.mojopkg && pixi run mojo build -O2 -I . \
#     -Xlinker -lm -Xlinker -lcuda -Xlinker -L.pixi/envs/default/lib \
#     -Xlinker -lsqlite3 -Xlinker -Lserenitymojo/ops/cshim/lib \
#     -Xlinker -lserenity_cudnn_sdpa \
#     serenitymojo/offload/tests/ltx2_fixed_stream_parity.mojo -o /tmp/ltx2_fixed_stream_parity
#   env LD_LIBRARY_PATH=.pixi/envs/default/lib:serenitymojo/ops/cshim/lib:\
#     $HOME/.local/lib/python3.12/site-packages/nvidia/cudnn/lib \
#     /tmp/ltx2_fixed_stream_parity

from max.gpu.host import DeviceContext
from serenitymojo.offload.ltx2_block_stream import (
    LTX2BlockStream, LTX2FixedStage, FP8Block,
)

comptime CKPT = "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-distilled-fp8.safetensors"


def _cmp(a: List[Float32], b: List[Float32]) -> Tuple[Int, Bool]:
    """(n_mismatch, nonzero). -1 mismatch marks a length mismatch."""
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
    fixed_blk: FP8Block,
    ctx: DeviceContext,
    mut allok: Bool,
    mut sig_out: Float64,
) raises -> Dict[String, Int]:
    """Byte-compare every tensor of the DEFAULT block vs the FIXED block and
    return name -> device address of the fixed view (for the cross-visit
    stability check). FAILs on any n_mismatch, a missing/extra tensor, or a
    degenerate all-zero block. Also folds a weighted content signature of the
    DEFAULT block into `sig_out` — main compares signatures across blocks to
    PROVE their content differs, so stale-buffer reuse can't pass silently."""
    var addrs = Dict[String, Int]()
    var n_tensors = 0
    var n_mismatch_total = 0
    var n_nonzero = 0
    var sig = Float64(0.0)
    if len(fixed_blk) != len(default_blk):
        print("   FAIL:", label, "tensor-count mismatch default=",
              len(default_blk), "fixed=", len(fixed_blk))
        allok = False
    for ref e in default_blk.items():
        var key = e.key
        n_tensors += 1
        if key not in fixed_blk:
            print("   FAIL:", label, "fixed block missing tensor", key)
            allok = False
            continue
        var da = default_blk[key][].to_host(ctx)
        var fa = fixed_blk[key][].to_host(ctx)
        var r = _cmp(da, fa)
        var nm = r[0]
        var nz = r[1]
        if nm != 0:
            n_mismatch_total += nm
            allok = False
            print("   FAIL:", label, key, " n_mismatch=", nm, " n=", len(da))
        if nz:
            n_nonzero += 1
        # Weighted fold of a deterministic sample of this tensor's content.
        var step = 1 + (len(da) // 64)
        var si = 0
        while si < len(da):
            sig += Float64(da[si]) * Float64(si + 1)
            si += step
        addrs[key] = Int(fixed_blk[key][].buf.unsafe_ptr())
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
    print("=== LTX2 FIXED-buffer block loader BIT gate (fixed vs default streamed) ===")
    var ctx = DeviceContext()
    var stream = LTX2BlockStream.open(CKPT)
    var nb = stream.block_count()
    print("  block_count =", nb)
    if nb < 48:
        raise Error(
            String("ltx2_fixed_stream_parity: expected >=48 video blocks, got ")
            + String(nb)
        )

    # Build ONE fixed stage (layout from block 0; reused for every block).
    var stage = stream.build_fixed_stage(ctx)
    var footprint = 0
    for i in range(stage.n_entries()):
        footprint += stage.layout[i].nbytes
    print("  [byte-math] fixed stage: entries=", stage.n_entries(),
          " per-block bf16 footprint (unaligned)=", footprint, "bytes (",
          footprint // (1024 * 1024), "MiB )",
          " buffer(256-aligned)=", stage.total_bytes, "bytes")

    var allok = True
    var blocks = [0, 24, 47]
    var ref_addrs = Dict[String, Int]()
    var sig0 = Float64(0.0)
    for bi in range(len(blocks)):
        var blk = blocks[bi]
        print(" -- block", blk, "--")
        var d = stream.load_block_bf16(blk, ctx)
        # FIXED loads run back-to-back through the SAME stage (0 -> 24 -> 47);
        # compare AFTER the refill to catch a stale buffer.
        var f = stream.load_block_bf16_fixed(blk, stage, ctx)
        var sig = Float64(0.0)
        var addrs = _compare_block(String("blk") + String(blk), d, f, ctx, allok, sig)
        if bi == 0:
            ref_addrs = addrs^
            sig0 = sig
        else:
            # (i) Stale-buffer test VALIDITY: this block's DEFAULT content must
            #     differ from block 0's, else a stale buffer (still holding 0's
            #     bytes) would compare equal to this block's default and pass
            #     silently. Prove content differs before trusting the byte match.
            if sig == sig0:
                print("   FAIL: blk", blk, "content signature == block 0 (", sig,
                      ") — stale-buffer catch would pass silently; pick another block")
                allok = False
            else:
                print("   PASS blk", blk, "content differs from block 0 (sig", sig,
                      "vs", sig0, ") — stale-buffer catch is meaningful")
            # (ii) Address stability across the SAME stage.
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
                      len(addrs), "tensors )")
        # d and f drop here — block 0's fixed views are stale after the next
        # refill by design; the compare above already captured their bytes.

    if allok:
        print("GATE ltx2_fixed_stream_parity: ALL PASS",
              "(fixed == default streamed, byte-exact; addresses stable 0->24->47)")
    else:
        print("GATE ltx2_fixed_stream_parity: FAIL")
        raise Error("ltx2_fixed_stream_parity gate FAILED")
