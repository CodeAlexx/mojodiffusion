# pipeline/ltx2_stft_smoke.mojo — GPU numeric gate for P-stft (forward-STFT-as-
# conv log-mel), the LTX-2 audio de-risk #1a.
#
# LTX2_PORT_PLAN_2026-05-28 §P-stft gate (PARITY): load the REAL checkpoint
# bases (vocoder.mel_stft.{mel_basis, stft_fn.forward_basis}) and a fixed 16 kHz
# stereo sine, run Mojo compute_mel, and require cos >= 0.999 (and a small
# max_abs in log-mel space) against the dumped reference mel produced by the
# exact Rust `compute_mel` algorithm (ltx2_vocoder.rs:1018-1047).
#
# This single test proves, end to end on the REAL weights:
#   * forward_basis [514,1,512] maps to conv1d weight convention (NOT transposed)
#   * 514 = 257×2 real-FIRST / imag-second channel split
#   * hop=80 (the BWE STFT hop, NOT mel_hop=160 which is the audio VAE)
#   * win=512, left zero-pad = win - hop = 432
#   * magnitude = sqrt(re² + im²); mel = magnitude @ mel_basisᵀ; clamp(1e-5,1e10).log()
#
# Oracle: serenitymojo/ops/parity/stft_mel_oracle.py writes
#   serenitymojo/ops/parity/stft_mel_oracle.safetensors with keys
#   {forward_basis[514,1,512], mel_basis[64,257], audio[1,2,T], mel_ref[1,2,64,Tf]}
# all as f32 (Mojo loads them as BF16 to match the Rust bf16 compute path).
#
# Build:
#   pixi run mojo build -I . -Xlinker -lm \
#       serenitymojo/pipeline/ltx2_stft_smoke.mojo -o /tmp/ltx2_stft_smoke
# Run:
#   /tmp/ltx2_stft_smoke

from std.math import sqrt
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.models.vocoder.ltx2_stft import compute_mel


comptime _ORACLE = (
    "/home/alex/mojodiffusion/serenitymojo/ops/parity/stft_mel_oracle.safetensors"
)
comptime _HOP = 80
comptime _COS_GATE = Float64(0.999)
# log-mel values span ~[-10.3, 0.4] (oracle stats). BF16 storage rounds to
# ~2^-8·|v| ≈ 0.04 near |v|≈10, plus the conv/matmul accumulation in bf16. A
# max_abs gate of 0.25 in log space is comfortably above the bf16 floor yet
# tight enough that any structural bug (wrong split, wrong hop, transposed
# basis, missing clamp/log) blows it far past — and cos would collapse.
comptime _MAXABS_GATE = Float64(0.25)


def _load_bf16(ref st: SafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    var info = st.tensor_info(name)
    var bytes = st.tensor_bytes(name)
    var view = from_parts(info.dtype, info.shape.copy(), bytes)
    return Tensor.from_view_as_bf16(view, ctx)


def main() raises:
    var ctx = DeviceContext()
    print("=== P-stft forward-STFT-as-conv log-mel GPU smoke (BF16) ===")

    var st = SafeTensors.open(String(_ORACLE))
    var forward_basis = _load_bf16(st, String("forward_basis"), ctx)
    var mel_basis = _load_bf16(st, String("mel_basis"), ctx)
    var audio = _load_bf16(st, String("audio"), ctx)
    var mel_ref = _load_bf16(st, String("mel_ref"), ctx)

    var fbs = forward_basis.shape()
    var mbs = mel_basis.shape()
    var ash = audio.shape()
    print(
        "  forward_basis ["
        + String(fbs[0]) + "," + String(fbs[1]) + "," + String(fbs[2])
        + "]  mel_basis [" + String(mbs[0]) + "," + String(mbs[1])
        + "]  audio [" + String(ash[0]) + "," + String(ash[1]) + "," + String(ash[2]) + "]"
    )

    # ── Run Mojo compute_mel ──────────────────────────────────────────────────
    var mel = compute_mel(audio, forward_basis, mel_basis, _HOP, ctx)
    var msh = mel.shape()
    var rsh = mel_ref.shape()
    print(
        "  mel out [" + String(msh[0]) + "," + String(msh[1]) + ","
        + String(msh[2]) + "," + String(msh[3]) + "]  ref ["
        + String(rsh[0]) + "," + String(rsh[1]) + "," + String(rsh[2]) + ","
        + String(rsh[3]) + "]"
    )
    if len(msh) != len(rsh):
        raise Error("compute_mel: rank mismatch vs reference")
    for i in range(len(msh)):
        if msh[i] != rsh[i]:
            raise Error(
                String("compute_mel: shape mismatch at dim ") + String(i)
                + " got " + String(msh[i]) + " expect " + String(rsh[i])
            )

    # ── Gate: cos + max_abs vs the dumped reference mel ───────────────────────
    var got = mel.to_host(ctx)
    var refv = mel_ref.to_host(ctx)
    var n = len(got)
    if n != len(refv):
        raise Error("compute_mel: numel mismatch vs reference")

    var dot = Float64(0.0)
    var ng = Float64(0.0)
    var nr = Float64(0.0)
    var maxabs = Float64(0.0)
    var has_bad = False
    for i in range(n):
        var g = got[i].cast[DType.float64]()
        var r = refv[i].cast[DType.float64]()
        if g != g:
            has_bad = True
        dot += g * r
        ng += g * g
        nr += r * r
        var d = (g - r).__abs__()
        if d > maxabs:
            maxabs = d
    var cos = Float64(0.0)
    if ng > 0.0 and nr > 0.0:
        cos = dot / (sqrt(ng) * sqrt(nr))
    if has_bad:
        cos = Float64(-1.0)

    var ok = (cos >= _COS_GATE) and (maxabs < _MAXABS_GATE)
    print(
        "  [" + ("PASS" if ok else "FAIL") + "] compute_mel: cos="
        + String(cos) + " max_abs=" + String(maxabs)
        + " (gate cos>=" + String(_COS_GATE) + ", max_abs<" + String(_MAXABS_GATE) + ")"
    )

    print("=== " + ("ALL PASS" if ok else "FAILED") + " ===")
    if not ok:
        raise Error("p_stft smoke FAILED numeric gate")
