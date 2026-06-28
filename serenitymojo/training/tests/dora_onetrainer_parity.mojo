# tests/dora_onetrainer_parity.mojo — Mojo DoRA (wd_on_out=False, per-INPUT axis)
# vs OneTrainer's OWN DoRAModule.forward (MJ-1023 re-target).
#
# Oracle: /tmp/dora_ot_oracle.safetensors from
#   python3 serenitymojo/training/tests/gen_dora_onetrainer_oracle.py
# (OneTrainer DoRAModule default decompose_output_axis=False, eps=0, alpha=2 rank=4).
#
# Build/run:
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#   pixi run mojo build --optimization-level 2 -I . -Xlinker -lm -Xlinker -lcuda \
#     -Xlinker -L.pixi/envs/default/lib -Xlinker -lsqlite3 \
#     serenitymojo/training/tests/dora_onetrainer_parity.mojo -o /tmp/dora_ot_parity \
#   && /tmp/dora_ot_parity
from std.collections import List
from std.math import sqrt
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.dtype import STDtype
from serenitymojo.training.dora_adapter import DoRAAdapter, dora_forward, dora_backward

comptime ORACLE = "/tmp/dora_ot_oracle.safetensors"
comptime COS_BAR = 0.999
comptime NREL_BAR = 8.0e-3


def _zeros(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for _ in range(n):
        out.append(Float32(0.0))
    return out^


def _read_f32(st: SafeTensors, name: String) raises -> List[Float32]:
    var info = st.tensor_info(name)
    if info.dtype != STDtype.F32:
        raise Error(String("oracle tensor not F32: ") + name)
    var bytes = st.tensor_bytes(name)
    var fp = bytes.unsafe_ptr().bitcast[Float32]()
    var out = List[Float32]()
    for i in range(info.size // 4):
        out.append(fp[i])
    return out^


def _cos(a: List[Float32], b: List[Float32]) raises -> Float64:
    if len(a) != len(b):
        raise Error("cos: len mismatch")
    var dot = 0.0; var na = 0.0; var nb = 0.0
    for i in range(len(a)):
        dot += Float64(a[i]) * Float64(b[i])
        na += Float64(a[i]) * Float64(a[i])
        nb += Float64(b[i]) * Float64(b[i])
    if na == 0.0 or nb == 0.0:
        raise Error("cos: zero vector")
    return dot / (sqrt(na) * sqrt(nb))


def _nrel(a: List[Float32], b: List[Float32]) raises -> Float64:
    var d = 0.0; var n = 0.0
    for i in range(len(a)):
        var dd = Float64(a[i]) - Float64(b[i])
        d += dd * dd
        n += Float64(b[i]) * Float64(b[i])
    if n == 0.0:
        return sqrt(d)
    return sqrt(d / n)


def main() raises:
    var st = SafeTensors.open(ORACLE)
    var dims = _read_f32(st, "dora.dims")
    var IN = Int(dims[0]); var OUT = Int(dims[1]); var R = Int(dims[2]); var M = Int(dims[3])
    print("[dora OneTrainer parity] IN=", IN, " OUT=", OUT, " R=", R, " M=", M, " (per-INPUT axis)")

    var A = _read_f32(st, "dora.A")        # [R,IN]
    var B = _read_f32(st, "dora.B")        # [OUT,R]
    var m = _read_f32(st, "dora.m_in")     # [IN]  per-input magnitude
    var W = _read_f32(st, "dora.W")        # [OUT,IN]
    var x = _read_f32(st, "dora.x")        # [M,IN]
    var y_ref = _read_f32(st, "dora.y")    # [M,OUT]

    # Build the Mojo DoRA with the EXACT oracle A/B/m, wd_on_out=False, eps=0.
    var d = DoRAAdapter(
        A.copy(), B.copy(), m.copy(),
        R, IN, OUT, Float32(2.0), Float32(0.0),
        _zeros(R * IN), _zeros(R * IN), _zeros(OUT * R), _zeros(OUT * R),
        _zeros(IN), _zeros(IN),
        False,                              # wd_on_out=False → per-INPUT (OneTrainer)
    )
    var y_got = dora_forward(x.copy(), W.copy(), d, M)

    var c = _cos(y_got, y_ref)
    var nr = _nrel(y_got, y_ref)
    print("  forward cos=", c, " nrel=", nr)
    var fwd_ok = c >= COS_BAR and nr <= NREL_BAR

    # ── BACKWARD vs OneTrainer autograd (detached norm) ──
    var dy = _read_f32(st, "dora.dy")        # [M,OUT]
    var dA = _read_f32(st, "dora.dA")        # [R,IN]
    var dB = _read_f32(st, "dora.dB")        # [OUT,R]
    var dm = _read_f32(st, "dora.dm_in")     # [IN]
    var g = dora_backward(dy.copy(), x.copy(), W.copy(), d, M)
    var ca = _cos(g.d_a, dA); var na = _nrel(g.d_a, dA)
    var cb = _cos(g.d_b, dB); var nb = _nrel(g.d_b, dB)
    var cm = _cos(g.d_m, dm); var nm = _nrel(g.d_m, dm)
    print("  d_A cos=", ca, " nrel=", na)
    print("  d_B cos=", cb, " nrel=", nb)
    print("  d_m cos=", cm, " nrel=", nm)
    var bwd_ok = (ca >= COS_BAR and na <= NREL_BAR and cb >= COS_BAR and nb <= NREL_BAR
                  and cm >= COS_BAR and nm <= NREL_BAR)

    if fwd_ok and bwd_ok:
        print("PASS — Mojo DoRA (per-INPUT, wd_on_out=False) reproduces OneTrainer DoRAModule fwd + autograd bwd")
    else:
        raise Error("FAIL: Mojo per-input DoRA does not match OneTrainer (see cos/nrel above)")
