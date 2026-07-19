# tests/oft_serenity_trainer_parity.mojo — Mojo SerenityTrainer-OFT forward (5-term Neumann,
# input-side) vs SerenityTrainer's OWN OFTModule.forward (MJ-1024).
#
# Oracle: /tmp/oft_serenity_oracle.safetensors from
#   python3 serenitymojo/training/tests/gen_oft_serenity_trainer_oracle.py
#
# Build/run:
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#   pixi run mojo build --optimization-level 2 -I . -Xlinker -lm -Xlinker -lcuda \
#     -Xlinker -L.pixi/envs/default/lib -Xlinker -lsqlite3 \
#     serenitymojo/training/tests/oft_serenity_trainer_parity.mojo -o /tmp/oft_serenity_parity \
#   && /tmp/oft_serenity_parity
from std.collections import List
from std.math import sqrt
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.dtype import STDtype
from serenitymojo.training.oft_serenity_trainer import oft_serenity_forward

comptime ORACLE = "/tmp/oft_serenity_oracle.safetensors"
comptime COS_BAR = 0.999
comptime NREL_BAR = 8.0e-3


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
    var dims = _read_f32(st, "oft.dims")
    var IN = Int(dims[0]); var OUT = Int(dims[1]); var M = Int(dims[2])
    var b = Int(dims[3]); var r = Int(dims[4]); var ne = Int(dims[5])
    print("[oft SerenityTrainer parity] IN=", IN, " OUT=", OUT, " M=", M,
          " block_size=", b, " r=", r, " n_elements=", ne, " (5-term Neumann, input-side)")

    var vec = _read_f32(st, "oft.weight_vec")   # [r,ne]
    var W = _read_f32(st, "oft.W")              # [OUT,IN]
    var x = _read_f32(st, "oft.x")              # [M,IN]
    var y_ref = _read_f32(st, "oft.y")          # [M,OUT]

    var y_got = oft_serenity_forward(x.copy(), vec.copy(), W.copy(), M, IN, OUT, b, r)

    var c = _cos(y_got, y_ref)
    var nr = _nrel(y_got, y_ref)
    print("  forward cos=", c, " nrel=", nr)
    if c >= COS_BAR and nr <= NREL_BAR:
        print("PASS — Mojo SerenityTrainer-OFT (5-term Neumann, input-side) reproduces SerenityTrainer OFTModule.forward")
    else:
        raise Error("FAIL: Mojo OFT-Neumann does not match SerenityTrainer (cos/nrel above)")
