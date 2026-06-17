# boogu_c2_parity.mojo — C2 (3-axis RoPE) parity gate vs torch oracle.
#
# Builds the Mojo joint RoPE cos/sin tables (build_boogu_rope_tables) for the
# T2I no-ref case cap_len=16, h_tok=16, w_tok=16 (seq=272) and compares to the
# torch BooguImageDoubleStreamRotaryPosEmbed freqs_cis dumped by boogu_c2_oracle.py
# (complex e^{iθ}: real=cos θ, imag=sin θ). Gate cos↔real, sin↔imag (cos >= 0.999).
#
# Run (oracle FIRST):
#   /home/alex/serenityflow-v2/.venv/bin/python serenitymojo/models/dit/parity/boogu_c2_oracle.py
#   rm -f serenitymojo.mojopkg
#   pixi run mojo run -I . serenitymojo/models/dit/parity/boogu_c2_parity.mojo
from std.gpu.host import DeviceContext
from std.collections import List
from std.memory import alloc
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from serenitymojo.parity import ParityHarness
from serenitymojo.models.dit.boogu_dit import build_boogu_rope_tables

comptime DUMP = "/home/alex/mojodiffusion/serenitymojo/models/dit/parity/boogu_dumps/"
comptime CAP = 16
comptime HTOK = 16
comptime WTOK = 16
comptime SEQ = CAP + HTOK * WTOK   # 272
comptime HALF = 60


def _read_bin_f32(path: String) raises -> List[Float32]:
    var fd = sys_open(path, O_RDONLY)
    if fd < 0:
        raise Error(String("cannot open (run boogu_c2_oracle.py first): ") + path)
    var n = file_size(fd)
    if n <= 0:
        _ = sys_close(fd)
        raise Error(String("empty/missing oracle dump: ") + path)
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


def main() raises:
    var ctx = DeviceContext()
    print("=== C2 (Boogu 3-axis RoPE) parity vs torch ===")
    print("  cap_len", CAP, "h_tok", HTOK, "w_tok", WTOK, "-> seq", SEQ, "half", HALF)

    var ref_real = _read_bin_f32(DUMP + "c2_joint_real.bin")
    var ref_imag = _read_bin_f32(DUMP + "c2_joint_imag.bin")
    if len(ref_real) != SEQ * HALF or len(ref_imag) != SEQ * HALF:
        raise Error("oracle joint bin size wrong: " + String(len(ref_real)))

    var tables = build_boogu_rope_tables(CAP, HTOK, WTOK, ctx)  # (cos[272,60], sin[272,60])

    var h = ParityHarness()
    var r_cos = h.compare(tables[0], ref_real, ctx)   # cos  <-> real(freqs_cis)
    var r_sin = h.compare(tables[1], ref_imag, ctx)   # sin  <-> imag(freqs_cis)
    print("  cos<->real ", r_cos)
    print("  sin<->imag ", r_sin)

    if not (r_cos.passed and r_sin.passed):
        raise Error("C2 RoPE parity FAIL (cos < 0.999 on cos or sin table)")
    print("VERDICT: C2 PASS — 3-axis RoPE tables match torch (cos >= 0.999)")
