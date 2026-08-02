# serenitymojo/models/minimax_h3/parity/minimax_h3_fp8_policy_parity.mojo
#
# MiniMax-H3 unit 14 parity gate: the fp8 residency policy and the host-side
# E4M3 encoder.
#
# Two claims are under test, and they are the two halves of "can this run on
# one 3090 Ti in fp8":
#
#   [1-4] FOOTPRINT — classify all 638 planned tensors and account the bytes.
#         Checked against an independent Python implementation over the same
#         key list, which is itself anchored to the converter's own printed
#         totals (638 keys / 12 fp32 / 66280430080 bytes). Every quantity is
#         an exact integer: byte counts have no tolerance.
#
#   [5-8] E4M3 BYTES — encode must be BIT-EXACT with torch.float8_e4m3fn, on
#         values chosen to break it: exact ties at every mantissa step, the
#         subnormal boundary and its m==8 promotion, the 16->8 carry, negative
#         zero, and saturation past 448. A "close enough" encoder here would
#         write a checkpoint that silently disagrees with every other runtime.
#
# The dequant tolerance in [8] is 0 as well — decode is exact arithmetic on a
# power-of-two grid, so the round trip reproduces torch's float32 result
# exactly, not approximately. Only the QUANTIZATION error is lossy, and that
# is a property of the format, reported by the oracle rather than gated here.
#
# Oracle: python3 scripts/minimax_h3_fp8_policy_oracle.py
# Run:
#   pixi run mojo build -O2 -j 1 -I . -I vendor/mojo-libs -Xlinker -lm \
#     serenitymojo/models/minimax_h3/parity/minimax_h3_fp8_policy_parity.mojo \
#     -o output/checks/minimax_h3_fp8_policy_parity \
#   && output/checks/minimax_h3_fp8_policy_parity

from std.collections import List
from std.pathlib import Path

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.models.minimax_h3.fp8_policy import (
    H3_FP8_ADALN,
    H3_FP8_BF16_KEEP,
    H3_FP8_F32_KEEP,
    H3_FP8_ROW,
    MiniMaxH3Fp8Budget,
    minimax_h3_e4m3_decode,
    minimax_h3_e4m3_encode,
    minimax_h3_e4m3_encode_row,
    minimax_h3_e4m3_row_scale,
)

comptime REF = "/home/alex/mojodiffusion/output/minimax_h3_fp8/fp8_policy_ref.safetensors"
comptime CSV = "/home/alex/mojodiffusion/output/minimax_h3_fp8/key_plan.csv"
comptime GIB = Float64(1024.0 * 1024.0 * 1024.0)


def _load_f32(ref st: SafeTensors, name: String) raises -> List[Float32]:
    var info = st.tensor_info(name)
    var bytes = st.tensor_bytes(name)
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    if tv.dtype != STDtype.F32:
        raise Error(String("_load_f32: unexpected dtype for ") + name)
    var p = tv.data.unsafe_ptr().bitcast[Float32]()
    var out = List[Float32]()
    for i in range(tv.numel()):
        out.append(p[i])
    return out^


def _load_i64(ref st: SafeTensors, name: String) raises -> List[Int]:
    var info = st.tensor_info(name)
    var bytes = st.tensor_bytes(name)
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    if tv.dtype != STDtype.I64:
        raise Error(String("_load_i64: unexpected dtype for ") + name)
    var p = tv.data.unsafe_ptr().bitcast[Int64]()
    var out = List[Int]()
    for i in range(tv.numel()):
        out.append(Int(p[i]))
    return out^


def main() raises:
    print("MiniMax-H3 fp8 policy + E4M3 encoder parity gate")
    print("  reference:", REF)
    var st = SafeTensors.open(String(REF))
    var checks = 0
    var failures = 0

    # ── build the budget from the real 638-key plan ───────────────────────────
    var budget = MiniMaxH3Fp8Budget()
    var lines = Path(String(CSV)).read_text().split("\n")
    var parsed = 0
    for li in range(len(lines)):
        ref line = lines[li]
        if line.byte_length() == 0:
            continue
        var parts = line.split(",")
        if len(parts) != 3:
            raise Error(String("bad csv row: ") + line)
        budget.add(String(parts[0]), Int(String(parts[1])), Int(String(parts[2])))
        parsed += 1

    var want_keys = _load_i64(st, "want.keys")
    var want_params = _load_i64(st, "want.params")
    var want_bytes = _load_i64(st, "want.bytes")

    print("")
    print("[1] key counts by class  (parsed", parsed, "rows)")
    var names = [
        String("F32_KEEP"), String("BF16_KEEP"), String("FP8_ROW"), String("ADALN")
    ]
    for c in range(4):
        checks += 1
        if budget.keys[c] == want_keys[c]:
            print("  ok  ", names[c], budget.keys[c], "keys")
        else:
            failures += 1
            print("  FAIL", names[c], budget.keys[c], "!=", want_keys[c])

    # 12 fp32 tensors is the converter's own number, and the five-prefix rule in
    # the intake is only credible if the name test reproduces it exactly.
    checks += 1
    if budget.keys[H3_FP8_F32_KEEP] == 12:
        print("  ok   exactly 12 fp32 tensors — the five-prefix rule holds")
    else:
        failures += 1
        print("  FAIL fp32 tensor count", budget.keys[H3_FP8_F32_KEEP], "!= 12")

    print("")
    print("[2] parameters by class")
    for c in range(4):
        checks += 1
        if budget.params[c] == want_params[c]:
            print(
                "  ok  ", names[c],
                Float64(budget.params[c]) / 1.0e9, "B params",
            )
        else:
            failures += 1
            print("  FAIL", names[c], budget.params[c], "!=", want_params[c])

    checks += 1
    var total = budget.total_params()
    if total == 33_123_004_320 or total > 33_000_000_000 and total < 33_200_000_000:
        print("  ok   total", Float64(total) / 1.0e9, "B params (33.1 B model)")
    else:
        failures += 1
        print("  FAIL total params", total)

    print("")
    print("[3] policy bytes by class")
    for c in range(4):
        checks += 1
        if budget.bytes[c] == want_bytes[c]:
            print("  ok  ", names[c], Float64(budget.bytes[c]) / GIB, "GiB")
        else:
            failures += 1
            print("  FAIL", names[c], budget.bytes[c], "!=", want_bytes[c])

    print("")
    print("[4] the verdict — does it fit 24 GiB")
    var want_bf16 = _load_i64(st, "want.bf16_all_bytes")[0]
    var want_r25 = _load_i64(st, "want.resident_25")[0]
    var want_r50 = _load_i64(st, "want.resident_50")[0]
    var want_ada = _load_i64(st, "want.adaln_resident")[0]
    var want_rows = _load_i64(st, "want.adaln_out_rows")[0]

    checks += 1
    if budget.adaln_out_rows == want_rows:
        print("  ok   adaLN output rows", budget.adaln_out_rows)
    else:
        failures += 1
        print("  FAIL adaLN output rows", budget.adaln_out_rows, "!=", want_rows)

    var got_bf16 = budget.bf16_all_bytes()
    checks += 1
    if got_bf16 == want_bf16:
        print("  ok   bf16 everything      ", Float64(got_bf16) / GIB, "GiB")
    else:
        failures += 1
        print("  FAIL bf16 total", got_bf16, "!=", want_bf16)

    var got_ada = budget.adaln_resident_bytes()
    checks += 1
    if got_ada == want_ada:
        print("  ok   fp8, adaLN resident  ", Float64(got_ada) / GIB, "GiB")
    else:
        failures += 1
        print("  FAIL adaLN-resident", got_ada, "!=", want_ada)

    var got_25 = budget.resident_bytes(25)
    var got_50 = budget.resident_bytes(50)
    checks += 1
    if got_25 == want_r25 and got_50 == want_r50:
        print("  ok   fp8, adaLN evicted   ", Float64(got_25) / GIB, "GiB @25 steps")
        print("  ok   fp8, adaLN evicted   ", Float64(got_50) / GIB, "GiB @50 steps")
    else:
        failures += 1
        print("  FAIL resident", got_25, got_50, "!=", want_r25, want_r50)

    # The claim the whole unit exists to make. 24 GiB card, weights only.
    comptime CARD_BYTES = 24 * 1024 * 1024 * 1024
    checks += 1
    if got_50 < CARD_BYTES and got_ada > CARD_BYTES and got_bf16 > CARD_BYTES:
        print(
            "  ok   VERDICT: fp8 + adaLN eviction is the ONLY one of the three",
            "that fits a 24 GiB card",
        )
        print(
            "       spare after weights:",
            Float64(CARD_BYTES - got_50) / GIB, "GiB for activations",
        )
    else:
        failures += 1
        print("  FAIL fit verdict changed — re-read the policy before trusting it")

    # ── E4M3 encoder ─────────────────────────────────────────────────────────
    print("")
    print("[5] E4M3 scalars vs torch.float8_e4m3fn")
    #
    # ONE DELIBERATE DIVERGENCE, and it is worth being precise about.
    #
    # E4M3-fn reserves the top mantissa code (exp 15, mant 7) for NaN, so 448 is
    # the largest finite value. torch's cast rounds to nearest and, when that
    # lands on the reserved code, RETURNS NaN — measured here: 464 ties down to
    # 448 and stays finite, 480 hits the code exactly and becomes NaN.
    #
    # Our encoder clamps to +-448 instead. That is not sloppiness, it is the
    # decision ops/fp8_quant.mojo already made and it is the right one for a
    # weight codec: a single NaN weight silently destroys every output that
    # touches it, while a saturated weight is off by at most one ULP of a value
    # that was already the largest in its row.
    #
    # It is also UNREACHABLE in this pipeline, which is what makes it safe
    # rather than merely preferable — check [5c] proves it: the per-row scale is
    # absmax/448, so max|w/scale| is exactly 448 and nothing can round past it.
    # Deleting the clamp would be equally correct today and catastrophic the
    # first time a scale arrives from somewhere else.
    var scalars = _load_f32(st, "in.scalars")
    var want_sb = _load_i64(st, "want.scalar_bytes")

    var bad = 0
    var first_bad = -1
    var finite_n = 0
    var overflow_n = 0
    var overflow_wrong = 0
    for i in range(len(scalars)):
        var got = Int(minimax_h3_e4m3_encode(scalars[i]))
        var want = want_sb[i]
        if (want & 0x7F) == 0x7F:
            # torch produced NaN; we must produce the saturated value, same sign
            overflow_n += 1
            var want_sat = 0xFE if (want & 0x80) != 0 else 0x7E
            if got != want_sat:
                overflow_wrong += 1
                print(
                    "  FAIL overflow", scalars[i], "got", got, "want saturated",
                    want_sat,
                )
        else:
            finite_n += 1
            if got != want:
                bad += 1
                if first_bad < 0:
                    first_bad = i

    print("  [5a] where torch stays finite — must be bit-identical")
    checks += 1
    if bad == 0:
        print("    ok  ", finite_n, "values, every byte identical")
    else:
        failures += 1
        print(
            "    FAIL", bad, "of", finite_n, "differ; first at", first_bad,
            "value", scalars[first_bad],
            "got", Int(minimax_h3_e4m3_encode(scalars[first_bad])),
            "want", want_sb[first_bad],
        )

    print("  [5b] where torch overflows to NaN — must saturate, never NaN")
    checks += 1
    if overflow_n > 0 and overflow_wrong == 0:
        print(
            "    ok  ", overflow_n, "overflowing values all clamp to +-448",
            "(0x7E / 0xFE)",
        )
    else:
        failures += 1
        print("    FAIL", overflow_wrong, "of", overflow_n, "overflow cases wrong")

    print("  [5c] the divergence is unreachable under per-row absmax scaling")
    checks += 1
    var mat_probe = _load_f32(st, "in.matrix")
    var worst_scaled = Float32(0.0)
    for r in range(6):
        var row = List[Float32]()
        for c in range(512):
            row.append(mat_probe[r * 512 + c])
        var sc = minimax_h3_e4m3_row_scale(row)
        for c in range(512):
            var a = row[c] / sc
            if a < 0:
                a = -a
            if a > worst_scaled:
                worst_scaled = a
    if worst_scaled <= Float32(448.0):
        print(
            "    ok   max |w/scale| =", worst_scaled,
            "<= 448 — no weight can reach the NaN region",
        )
    else:
        failures += 1
        print("    FAIL max |w/scale| =", worst_scaled, "> 448")

    # Negative zero is its own check: it is the one input where a sign-magnitude
    # encoder can silently lose information, and 0x00 vs 0x80 never shows up in
    # a value comparison because both decode to zero.
    print("")
    print("[6] signed zero")
    # Its own check because a value comparison can never see it: 0x00 and 0x80
    # both decode to a zero, so only a byte comparison catches a dropped sign.
    # The first version of this check accepted either answer and duly passed
    # while the encoder was wrong.
    checks += 1
    var pz = Int(minimax_h3_e4m3_encode(Float32(0.0)))
    var nz = Int(minimax_h3_e4m3_encode(Float32(-0.0)))
    if pz == 0x00 and nz == 0x80:
        print("  ok   +0 -> 0x00, -0 -> 0x80, sign preserved as torch does")
    else:
        failures += 1
        print("  FAIL +0 ->", pz, "(want 0)  -0 ->", nz, "(want 128)")

    print("")
    print("[7] per-row scale and bytes on a bf16 matrix")
    var mat = _load_f32(st, "in.matrix")
    var want_scale = _load_f32(st, "want.row_scale")
    var want_mb = _load_i64(st, "want.matrix_bytes")
    comptime ROWS = 6
    comptime COLS = 512
    var scale_bad = 0
    var byte_bad = 0
    var got_scales = List[Float32]()
    for r in range(ROWS):
        var row = List[Float32]()
        for c in range(COLS):
            row.append(mat[r * COLS + c])
        var sc = minimax_h3_e4m3_row_scale(row)
        got_scales.append(sc)
        if sc != want_scale[r]:
            scale_bad += 1
        var bytes = minimax_h3_e4m3_encode_row(row, sc)
        for c in range(COLS):
            if Int(bytes[c]) != want_mb[r * COLS + c]:
                byte_bad += 1
    checks += 1
    if scale_bad == 0:
        print("  ok   all", ROWS, "row scales bit-identical (zero row ->",
              got_scales[3], ")")
    else:
        failures += 1
        print("  FAIL", scale_bad, "row scales differ")
    checks += 1
    if byte_bad == 0:
        print("  ok   all", ROWS * COLS, "encoded bytes identical to torch")
    else:
        failures += 1
        print("  FAIL", byte_bad, "of", ROWS * COLS, "bytes differ")

    print("")
    print("[8] decode: round trip reproduces torch exactly")
    var want_deq = _load_f32(st, "want.matrix_dequant")
    checks += 1
    var deq_bad = 0
    var worst = Float32(0.0)
    for r in range(ROWS):
        var row = List[Float32]()
        for c in range(COLS):
            row.append(mat[r * COLS + c])
        var sc = minimax_h3_e4m3_row_scale(row)
        var bytes = minimax_h3_e4m3_encode_row(row, sc)
        for c in range(COLS):
            var got = minimax_h3_e4m3_decode(bytes[c]) * sc
            var want = want_deq[r * COLS + c]
            if got != want:
                deq_bad += 1
                var d = got - want
                var mag = -d if d < 0 else d
                if mag > worst:
                    worst = mag
    if deq_bad == 0:
        print("  ok  ", ROWS * COLS, "dequantized values exact (no tolerance)")
    else:
        failures += 1
        print("  FAIL", deq_bad, "dequantized values differ, worst", worst)

    print("")
    if failures == 0:
        print("PASS:", checks, "checks")
    else:
        print("FAIL:", failures, "of", checks, "checks")
        raise Error("MiniMax-H3 fp8 policy parity gate failed")
