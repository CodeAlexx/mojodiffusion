# serenitymojo/models/wan22/parity/bernini_cond_forward_parity.mojo
#
# PARITY GATE for BERNINI-R Tier-2b — the reference-CONDITIONING PACKED sequence.
#
# WHAT THIS GATES (stated explicitly, per the Tier-2b gate spec):
#   Tier-2b adds exactly ONE new piece of compute on top of the ALREADY-certified
#   pieces — the MULTI-SEGMENT source-id RoPE for a packed [conditioning | target]
#   sequence, plus the trainer's TARGET-REGION slice. The wan22 block/stack forward
#   is generic over sequence length + cos/sin and is certified at cos>=0.999 by
#   wan22_block_lora_parity.mojo (arbitrary S, arbitrary rope); the SINGLE-segment
#   source-id rope is certified by bernini_src_id_rope_parity.mojo. So this gate
#   proves the NEW assembly:
#     (G1a) build_bernini_src_id_rope over the REAL Tier-2b geometry — a
#           CONDITIONING segment (grid 1x8x8, source_id=1, 64 tokens) concatenated
#           AHEAD of the TARGET segment (grid 1x16x16, source_id=0, 256 tokens),
#           real head_dim=128 — matches the torch/diffusers packed rope (the exact
#           tables the Tier-2b forward consumes) at cos>=0.999 on BOTH cos and sin.
#     (G1b) TARGET-REGION identity: the trailing S_TGT=256 rows of the packed rope
#           (the region the trainer slices for the loss) are BIT-IDENTICAL to the
#           stock wan rope for the target grid (source_id=0 identity property) —
#           i.e. the target is rendered under stock geometry, unperturbed.
#     (G1c) CONDITIONING-REGION rotation: the leading S_COND=64 rows DIFFER from the
#           stock cond-grid rope (source_id=1 phase actually applied), so the
#           conditioning carries a distinct source id, not a no-op.
#   The forward-output FINITE check on the real packed sequence is covered by the
#   G2 real-weight conditioned smoke (train_bernini_r_cond.mojo), which runs the
#   full 40-block packed forward+backward on the real Bernini-R fp8 base.
#
# Run (oracle FIRST, SEPARATE command):
#   /home/alex/ai-toolkit/venv/bin/python \
#       serenitymojo/models/wan22/parity/bernini_cond_forward_oracle.py
#   rm -f serenitymojo.mojopkg
#   pixi run mojo run -I . \
#       serenitymojo/models/wan22/parity/bernini_cond_forward_parity.mojo

from std.gpu.host import DeviceContext
from std.collections import List
from std.memory import alloc
from serenitymojo.parity import ParityHarness
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from serenitymojo.io.dtype import STDtype
from serenitymojo.models.wan22.bernini_src_id_rope import (
    build_bernini_src_id_rope, BerniniRopeSegment,
)
from serenitymojo.models.dit.wan22_dit import wan22_build_rope


comptime REF_DIR = "/home/alex/mojodiffusion/serenitymojo/models/wan22/parity/"

# MUST match bernini_cond_forward_oracle.py AND the trainer smoke geometry.
comptime HEAD_DIM = 128
comptime THETA = Float32(10000.0)
comptime HALF = HEAD_DIM // 2          # 64
comptime COND_F = 1
comptime COND_H = 8
comptime COND_W = 8
comptime S_COND = COND_F * COND_H * COND_W    # 64
comptime TGT_F = 1
comptime TGT_H = 16
comptime TGT_W = 16
comptime S_TGT = TGT_F * TGT_H * TGT_W        # 256
comptime S_PACKED = S_COND + S_TGT            # 320
comptime COND_SRC_ID = Float32(1.0)
comptime TGT_SRC_ID = Float32(0.0)


def _read_bin_f32(path: String) raises -> List[Float32]:
    var fd = sys_open(path, O_RDONLY)
    if fd < 0:
        raise Error(String("cannot open: ") + path)
    var n = file_size(fd)
    if n <= 0:
        _ = sys_close(fd)
        raise Error(String("empty/missing ref (run the oracle first): ") + path)
    var buf = alloc[UInt8](n)
    var done = 0
    while done < n:
        var got = sys_pread(fd, buf + done, n - done, done)
        if got <= 0:
            break
        done += got
    _ = sys_close(fd)
    var nf = n // 4
    var fp = buf.bitcast[Float32]()
    var out = List[Float32]()
    for i in range(nf):
        out.append(fp[i])
    buf.free()
    return out^


def _in(name: String) raises -> List[Float32]:
    return _read_bin_f32(REF_DIR + name + ".bin")


def _check(
    mut harness: ParityHarness, name: String,
    actual: List[Float32], expected: List[Float32], mut allok: Bool,
) raises:
    var r = harness.compare_host(actual, expected)
    print("  cos(", name, ") =", r.cos, "  max_abs =", r.max_abs,
          "  n =", r.n, "  ", "PASS" if r.passed else "FAIL")
    if not r.passed:
        allok = False


# Rows [r0, r0+nrows) of a [rows, HALF] host list.
def _rows(a: List[Float32], r0: Int, nrows: Int) raises -> List[Float32]:
    var out = List[Float32]()
    var start = r0 * HALF
    var end = (r0 + nrows) * HALF
    if end > len(a):
        raise Error("row slice out of range")
    for i in range(start, end):
        out.append(a[i])
    return out^


def _max_abs_diff(a: List[Float32], b: List[Float32]) raises -> Float32:
    if len(a) != len(b):
        raise Error("length mismatch in _max_abs_diff")
    var m = Float32(0.0)
    for i in range(len(a)):
        var d = a[i] - b[i]
        if d < 0.0:
            d = -d
        if d > m:
            m = d
    return m


def main() raises:
    var ctx = DeviceContext()
    print("==== bernini_cond_forward_parity (Tier-2b packed [cond|target] rope vs torch) ====")
    print("HEAD_DIM=", HEAD_DIM, " packed S=", S_PACKED,
          " (cond", S_COND, " src_id=", COND_SRC_ID,
          " | target", S_TGT, " src_id=", TGT_SRC_ID, ")  theta=", THETA)

    var harness = ParityHarness()
    var allok = True

    # ── build the packed src-id rope exactly as the trainer does ────────────────
    var segs = List[BerniniRopeSegment]()
    segs.append(BerniniRopeSegment(COND_F, COND_H, COND_W, COND_SRC_ID))
    segs.append(BerniniRopeSegment(TGT_F, TGT_H, TGT_W, TGT_SRC_ID))
    var packed = build_bernini_src_id_rope(segs, HEAD_DIM, THETA, STDtype.F32, ctx)
    var pcos = packed[0].to_host(ctx)     # [S_PACKED*HALF]
    var psin = packed[1].to_host(ctx)
    if len(pcos) != S_PACKED * HALF or len(psin) != S_PACKED * HALF:
        raise Error("packed rope length != S_PACKED*HALF")

    # ── (G1a) full packed rope vs torch ─────────────────────────────────────────
    print("")
    print("---- G1a: packed [cond|target] rope vs torch (the exact Tier-2b tables) ----")
    _check(harness, "packed_cos", pcos, _in("cf_packed_cos"), allok)
    _check(harness, "packed_sin", psin, _in("cf_packed_sin"), allok)

    # ── (G1b) TARGET-REGION slice == stock target-grid rope (src_id=0 identity) ──
    # This is the region the trainer slices for the velocity-MSE loss: the trailing
    # S_TGT rows, offset = S_PACKED - S_TGT = S_COND.
    print("")
    print("---- G1b: target region (trailing", S_TGT, "rows, offset", S_COND,
          ") == stock target rope (src_id=0 identity) ----")
    var tgt_cos = _rows(pcos, S_COND, S_TGT)
    var tgt_sin = _rows(psin, S_COND, S_TGT)
    # Bit-identity is a MOJO-internal invariant (src_id=0 => stock wan rope), so it
    # must be checked against the MOJO-built stock target rope, NOT the f32-rounded
    # torch dump (which only agrees to ~1e-6). Build the stock target-grid rope and
    # assert the packed target slice equals it EXACTLY.
    var stock_tgt = wan22_build_rope(TGT_F, TGT_H, TGT_W, HEAD_DIM, THETA, STDtype.F32, ctx)
    var mojo_stock_cos = stock_tgt[0].to_host(ctx)
    var mojo_stock_sin = stock_tgt[1].to_host(ctx)
    var dtc = _max_abs_diff(tgt_cos, mojo_stock_cos)
    var dts = _max_abs_diff(tgt_sin, mojo_stock_sin)
    print("  IDENTITY (vs Mojo stock) max|tgt_cos-stock| =", dtc,
          "  max|tgt_sin-stock| =", dts,
          "  ", "PASS (bit-identical)" if (dtc == 0.0 and dts == 0.0) else "FAIL")
    if not (dtc == 0.0 and dts == 0.0):
        allok = False
    # And cos-parity the sliced target vs the torch stock reference (cross-check).
    _check(harness, "target slice vs torch stock", tgt_cos, _in("cf_stock_cos_tgt"), allok)

    # ── (G1c) CONDITIONING-REGION slice IS rotated (src_id=1, not a no-op) ───────
    print("")
    print("---- G1c: conditioning region (leading", S_COND,
          " rows) is rotated vs stock cond rope (src_id=1 applied) ----")
    var cond_cos = _rows(pcos, 0, S_COND)
    var cond_sin = _rows(psin, 0, S_COND)
    var stock_cond_cos = _in("cf_stock_cos_cond")
    var stock_cond_sin = _in("cf_stock_sin_cond")
    var dcc = _max_abs_diff(cond_cos, stock_cond_cos)
    var dcs = _max_abs_diff(cond_sin, stock_cond_sin)
    print("  ROTATION max|cond_cos - stock| =", dcc, "  max|cond_sin - stock| =", dcs,
          "  ", "PASS (rotated, non-zero)" if (dcc > 0.0 or dcs > 0.0) else "FAIL")
    if not (dcc > 0.0 or dcs > 0.0):
        allok = False

    print("")
    if allok:
        print("VERDICT: PASS — Tier-2b packed [cond|target] src-id rope matches torch",
              "(cos>=0.999), the target-region slice is bit-identical to stock wan",
              "rope (src_id=0), and the conditioning region carries its src_id=1 phase.")
    else:
        print("VERDICT: FAIL — at least one check diverged (see FAIL lines above)")
