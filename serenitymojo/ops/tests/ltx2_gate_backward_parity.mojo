# ops/tests/ltx2_gate_backward_parity.mojo — bit-level parity gate for the device
# gate-grad op (ltx2_gate_dgates) vs the HOST oracle (the exact loop replaced,
# ltx2_av_backward.mojo:359-367), both bf16 AND F32 storage, real shapes
# (SQ=256, H=32, DH=128). Bar: n_mismatch=0.
#
#   pixi run mojo build -I . -Xlinker -lm -Xlinker -lcuda \
#     -Xlinker -L.pixi/envs/default/lib -Xlinker -lsqlite3 \
#     serenitymojo/ops/tests/ltx2_gate_backward_parity.mojo -o /tmp/ltx2_gate_parity
#   env LD_LIBRARY_PATH=.pixi/envs/default/lib /tmp/ltx2_gate_parity

from std.math import sin, cos
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.autograd_v2.step_slab import StepSlab
from serenitymojo.ops.ltx2_gate_backward import ltx2_gate_dgates, ltx2_gate_dgates_slab

comptime SQ = 256
comptime H = 32
comptime DH = 128
comptime INNER = H * DH


def _fill(n: Int, phase: Float32) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        var fi = Float32(i)
        out.append(Float32(0.05) * (sin(Float32(0.9) * fi + phase)
                   + Float32(0.5) * cos(Float32(1.7) * fi)))
    return out^


# HOST oracle — the EXACT loop from ltx2_av_backward.mojo:359-367.
def _host_dgates(dag_h: List[Float32], af_h: List[Float32]) -> List[Float32]:
    var dg = List[Float32]()
    for s in range(SQ):
        for h in range(H):
            var acc = Float32(0.0)
            var base = s * INNER + h * DH
            for d in range(DH):
                acc += dag_h[base + d] * af_h[base + d]
            dg.append(acc)
    return dg^


def _run(dt: STDtype, name: String, ctx: DeviceContext) raises -> Bool:
    var dag = Tensor.from_host(_fill(SQ * INNER, Float32(0.13)), [SQ, INNER], dt, ctx)
    var af = Tensor.from_host(_fill(SQ * INNER, Float32(0.29)), [SQ, INNER], dt, ctx)
    # host oracle: upcast (to_host = F32), F32 dot, then STORE as dt (from_host cast)
    var dag_h = dag.to_host(ctx)
    var af_h = af.to_host(ctx)
    var dg_host_f32 = _host_dgates(dag_h, af_h)
    var d_gates_host = Tensor.from_host(dg_host_f32, [SQ, H], dt, ctx)  # cast to dt
    # device kernel (non-slab) + slab variant
    var d_gates_dev = ltx2_gate_dgates(dag, af, SQ, H, DH, ctx)
    var slab = StepSlab(ctx, SQ * H * 4 + 4096)
    var d_gates_slab = ltx2_gate_dgates_slab(dag, af, SQ, H, DH, ctx, slab)
    # bit-compare the STORED values (upcast to F32 is exact for bf16/F32)
    var hh = d_gates_host.to_host(ctx)
    var dd = d_gates_dev.to_host(ctx)
    var ss = d_gates_slab.to_host(ctx)
    var nm = 0
    var nm_slab = 0
    var nz = False
    for i in range(len(hh)):
        if hh[i] != dd[i]:
            nm += 1
        if dd[i] != ss[i]:
            nm_slab += 1
        if hh[i] != Float32(0.0):
            nz = True
    var ok = (nm == 0 and nm_slab == 0 and nz)
    var verdict = "PASS" if ok else "FAIL"
    print("  ", verdict, name, " n_mismatch(vs host)=", nm, " n_mismatch(slab vs dev)=",
          nm_slab, " nonzero=", nz, " n=", len(hh))
    return ok


def main() raises:
    print("=== ltx2 gate-grad device vs host BIT parity (real shapes) ===")
    var ctx = DeviceContext()
    var ok_bf16 = _run(STDtype.BF16, "bf16", ctx)
    var ok_f32 = _run(STDtype.F32, "f32 ", ctx)
    if ok_bf16 and ok_f32:
        print("GATE ltx2_gate_backward_parity: ALL PASS (bit-exact, both storage classes)")
    else:
        print("GATE ltx2_gate_backward_parity: not bit-exact (see lines) — needs the flag path")
        raise Error("ltx2_gate_backward_parity FAILED bit-exactness")
