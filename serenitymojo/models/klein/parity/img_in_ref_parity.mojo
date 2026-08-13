# serenitymojo/models/klein/parity/img_in_ref_parity.mojo
#
# PARITY GATE for the Klein image-EDIT reference-conditioning unit
# (models/klein/img_in_ref.mojo). Loads the EXACT inputs + torch-autograd
# reference grads dumped by img_in_ref_oracle.py, runs img_in_ref_forward +
# img_in_ref_backward, and asserts:
#   * d_img_in_ref_w vs torch     cos >= 0.999   (the load-bearing NEW param grad)
#   * d_ref_tokens   vs torch     cos >= 0.999   (self-contained extra)
#   * img_act        vs torch     cos >= 0.999   (summed forward, sanity)
#   * FLAGS-OFF IDENTITY: with img_in_ref = 0, forward EQUALS linear(noise, img_in)
#     at max_abs 0 (byte-identical) — the C13 flags-off contract.
#
# REAL Klein input dims D=4096, in_ch=128; N_IMG=256 (small for oracle speed).
# NON-DEGENERATE inputs (sinusoidal/randn — no modular aliasing).
#
# Run (oracle FIRST, as a SEPARATE command — never chained after a mojo build):
#   cd /home/alex/mojodiffusion
#   /home/alex/serenityflow-v2/.venv/bin/python \
#       serenitymojo/models/klein/parity/img_in_ref_oracle.py
#   rm -f serenitymojo.mojopkg
#   pixi run mojo run -I . serenitymojo/models/klein/parity/img_in_ref_parity.mojo

from max.gpu.host import DeviceContext
from std.collections import List
from std.memory import alloc
from serenitymojo.parity import ParityHarness, ParityResult
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from serenitymojo.models.klein.img_in_ref import (
    img_in_ref_forward, img_in_ref_backward, ImgInRefForward, ImgInRefGrads,
)
# reuse the SAME host linear the unit + stack use (identity reference)
from serenitymojo.models.klein.klein_stack import _linear_fwd


comptime REF_DIR = "/home/alex/mojodiffusion/serenitymojo/models/klein/parity/"

# dims MUST match img_in_ref_oracle.py
comptime D = 4096
comptime IN_CH = 128
comptime N_IMG = 256


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


def _zeros(d: Int) -> List[Float32]:
    var o = List[Float32]()
    for _ in range(d):
        o.append(0.0)
    return o^


def _check(
    mut harness: ParityHarness, name: String,
    actual: List[Float32], expected: List[Float32], mut allok: Bool,
) raises:
    var r = harness.compare_host(actual, expected)
    print("  cos(", name, ") =", r.cos, "  max_abs =", r.max_abs,
          "  n =", r.n, "  ", "PASS" if r.passed else "FAIL")
    if not r.passed:
        allok = False


def main() raises:
    var ctx = DeviceContext()
    print("==== img_in_ref_parity (Klein image-edit ref projection vs torch) ====")
    print("D=", D, " IN_CH=", IN_CH, " N_IMG=", N_IMG)

    # ── load inputs (byte-identical to the oracle) ──
    var noise = _in("iir_in_noise")
    var ref_tok = _in("iir_in_ref")
    var img_in = _in("iir_in_img_in")
    var img_in_ref = _in("iir_in_img_in_ref")
    var d_img = _in("iir_in_d_img")

    var harness = ParityHarness()
    var allok = True

    # ── forward (non-zero img_in_ref) ──
    var fwd = img_in_ref_forward(
        noise, img_in, ref_tok, img_in_ref, N_IMG, IN_CH, D, ctx,
    )
    print("")
    print("---- forward (summed projection) vs torch ----")
    _check(harness, "img_act", fwd.img_act, _in("iir_ref_img_out"), allok)

    # ── backward ──
    var g = img_in_ref_backward(
        d_img, fwd, img_in_ref, N_IMG, IN_CH, D, ctx,
    )
    print("")
    print("---- grads vs torch ----")
    _check(harness, "d_img_in_ref_w", g.d_img_in_ref_w, _in("iir_ref_d_img_in_ref"), allok)
    _check(harness, "d_ref_tokens  ", g.d_ref_tokens, _in("iir_ref_d_ref"), allok)

    # ── FLAGS-OFF IDENTITY: img_in_ref = 0 -> forward == linear(noise, img_in) ──
    print("")
    print("---- flags-off identity (img_in_ref = 0) ----")
    var zero_ref_w = _zeros(D * IN_CH)
    var fwd0 = img_in_ref_forward(
        noise, img_in, ref_tok, zero_ref_w, N_IMG, IN_CH, D, ctx,
    )
    var base_only = _linear_fwd(noise, img_in, N_IMG, IN_CH, D, ctx)
    var rid = harness.compare_host(fwd0.img_act, base_only)
    print("  identity: cos =", rid.cos, "  max_abs =", rid.max_abs,
          "  n =", rid.n, "  ", "PASS" if rid.max_abs == 0.0 else "FAIL")
    if rid.max_abs != 0.0:
        allok = False

    print("")
    if allok:
        print("VERDICT: PASS — img_in_ref fwd+bwd matches torch (cos>=0.999) and the img_in_ref=0 identity is max_abs 0")
    else:
        print("VERDICT: FAIL — see FAIL lines above")
