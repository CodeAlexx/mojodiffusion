# sampling/parity/cosmos_sampler_parity.mojo — Cosmos RF + UniPC step-parity gate.
#
# NUMERIC STEP-PARITY vs the CANONICAL Cosmos sampler. The oracle arrays below
# are produced by running the canonical Python sampler
#   /home/alex/refs/cosmos-predict2.5/cosmos_predict2/_src/predict2/models/
#   fm_solvers_unipc.py  (FlowUniPCMultistepScheduler)
# via gen_cosmos_sampler_reference.py — NOT a transcription. Both samplers are
# driven with a FIXED model-output (velocity) per step + FIXED sigma schedule
# (shift=5, num_train=1000), so each step is gated as pure numeric step-parity
# with no model in the loop.
#
# Two gates, cos >= 0.999 per step AND over the full trajectory:
#   RF    : CosmosRectifiedFlowSampler.step (FlowMatch Euler)
#   UniPC : UniPcMultistepScheduler.step    (bh2 multistep predictor/corrector)
#
# Run:
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#     pixi run mojo run -I . serenitymojo/sampling/parity/cosmos_sampler_parity.mojo

from collections import List
from std.gpu.host import DeviceContext

from serenitymojo.io.dtype import STDtype
from serenitymojo.tensor import Tensor
from serenitymojo.parity import ParityHarness, ParityResult
from serenitymojo.sampling.cosmos_rf import CosmosRectifiedFlowSampler
from serenitymojo.sampling.cosmos_unipc import build_cosmos_unipc_sampler


comptime N_INF = 6
comptime DIM = 8


def _dim_shape() -> List[Int]:
    var sh = List[Int]()
    sh.append(DIM)
    return sh^


# ── Oracle arrays (generated from the canonical Cosmos sampler) ──────────────

def _x_init() -> List[Float32]:
    return [-1.60383681, 0.06409991, 0.74089130, 0.15261919, 0.86374389, 2.91309922, -1.47882336, 0.94547297]


def _velocity(i: Int) -> List[Float32]:
    if i == 0:
        return [-1.66613546, 0.34374458, -0.51244371, 1.32375896, -0.86028019, 0.51949320, -1.26514372, -2.15913901]
    if i == 1:
        return [0.43473395, 1.73328932, 0.52013416, -1.00216579, 0.26834554, 0.76717470, 1.19127203, -1.15741081]
    if i == 2:
        return [0.69627940, 0.35138369, -0.03241508, 0.01318158, -0.67924997, -0.62053203, 1.33121422, 0.25883851]
    if i == 3:
        return [-0.48148392, -2.49178962, -0.87656377, -0.50550913, -1.28312917, -1.33032842, 0.82599258, -0.24721501]
    if i == 4:
        return [-1.69970612, -1.33515287, -0.29963889, 1.11480690, -1.50640885, 1.59011208, -0.48732518, -1.71110219]
    return [0.51309021, 1.43709192, -0.22180411, 0.64881650, -0.31789119, -0.01097826, 1.66541650, 0.89578643]


def _rf_oracle(i: Int) -> List[Float32]:
    if i == 0:
        return [-1.53974738, 0.05087746, 0.76060291, 0.10169959, 0.89683536, 2.89311644, -1.43015845, 1.02852624]
    if i == 1:
        return [-1.56255037, -0.04003832, 0.73332045, 0.15426594, 0.88275990, 2.85287602, -1.49264393, 1.08923561]
    if i == 2:
        return [-1.61530299, -0.06666040, 0.73577633, 0.15326725, 0.93422231, 2.89988975, -1.59350149, 1.06962508]
    if i == 3:
        return [-1.55798074, 0.22999538, 0.84013415, 0.21344978, 1.08698308, 3.05826975, -1.69183864, 1.09905684]
    if i == 4:
        return [-1.19376010, 0.51609792, 0.90434211, -0.02543603, 1.40978311, 2.71753341, -1.58741242, 1.46571948]
    return [-1.45027438, -0.20236169, 1.01523084, -0.34980530, 1.56870961, 2.72302188, -2.42002061, 1.01788008]


def _unipc_oracle(i: Int) -> List[Float32]:
    if i == 0:
        return [-1.53974736, 0.05087746, 0.76060290, 0.10169959, 0.89683535, 2.89311641, -1.43015843, 1.02852622]
    if i == 1:
        return [-1.60658649, -0.06916446, 0.71167665, 0.20301945, 0.85910285, 2.84768433, -1.54413264, 1.06823846]
    if i == 2:
        return [-1.66962303, -0.01871328, 0.74581825, 0.14301600, 0.96373568, 2.97000550, -1.64806726, 0.97405521]
    if i == 3:
        return [-1.50266541, 0.50921478, 0.91707995, 0.26308632, 1.15614763, 3.17005571, -1.69789248, 1.07066809]
    if i == 4:
        return [-0.93817914, 0.54700845, 0.86577090, -0.26109897, 1.50466572, 2.29905650, -1.36843861, 1.69633434]
    return [-1.19469342, -0.17145117, 0.97665963, -0.58546824, 1.66359222, 2.30454497, -2.20104680, 1.24849494]


def _extend(mut acc: List[Float32], src: List[Float32]):
    for i in range(len(src)):
        acc.append(src[i])


def main() raises:
    var ctx = DeviceContext()
    var harness = ParityHarness(0.999)
    print("=== Cosmos-Predict2.5 sampler step-parity gate (cos>=0.999) ===")
    print("    N_INF=", N_INF, " DIM=", DIM, " shift=5.0  (F32, no model in loop)")

    # ── RF (FlowMatch Euler) gate ────────────────────────────────────────────
    var rf = CosmosRectifiedFlowSampler(N_INF, 5.0)
    var rf_x = Tensor.from_host(_x_init(), _dim_shape(), STDtype.F32, ctx)
    var rf_all_actual = List[Float32]()
    var rf_all_ref = List[Float32]()
    var rf_min_cos: Float64 = 1.0
    print("--- RF rectified-flow Euler ---")
    for i in range(N_INF):
        var v = Tensor.from_host(_velocity(i), _dim_shape(), STDtype.F32, ctx)
        var nxt = rf.step(rf_x, v, i, ctx)
        var ref_vals = _rf_oracle(i)
        var r = harness.compare(nxt, ref_vals, ctx)
        print("  step", i, ":", r)
        if r.cos < rf_min_cos:
            rf_min_cos = r.cos
        var act = nxt.to_host(ctx)
        _extend(rf_all_actual, act)
        _extend(rf_all_ref, ref_vals)
        rf_x = nxt^

    var rf_multi = harness.compare_host(rf_all_actual, rf_all_ref)
    print("  RF multi-step (concatenated):", rf_multi)
    print("  RF min per-step cos =", rf_min_cos)

    # ── UniPC (bh2 multistep) gate ───────────────────────────────────────────
    var sch = build_cosmos_unipc_sampler(N_INF, 5.0)
    var up_x = Tensor.from_host(_x_init(), _dim_shape(), STDtype.F32, ctx)
    var up_all_actual = List[Float32]()
    var up_all_ref = List[Float32]()
    var up_min_cos: Float64 = 1.0
    print("--- UniPC bh2 multistep predictor/corrector ---")
    for i in range(N_INF):
        var v = Tensor.from_host(_velocity(i), _dim_shape(), STDtype.F32, ctx)
        var nxt = sch.step(v, up_x, ctx)
        var ref_vals = _unipc_oracle(i)
        var r = harness.compare(nxt, ref_vals, ctx)
        print("  step", i, ":", r)
        if r.cos < up_min_cos:
            up_min_cos = r.cos
        var act = nxt.to_host(ctx)
        _extend(up_all_actual, act)
        _extend(up_all_ref, ref_vals)
        up_x = nxt^

    var up_multi = harness.compare_host(up_all_actual, up_all_ref)
    print("  UniPC multi-step (concatenated):", up_multi)
    print("  UniPC min per-step cos =", up_min_cos)

    # ── Gate ─────────────────────────────────────────────────────────────────
    var ok = (
        rf_min_cos >= 0.999
        and rf_multi.cos >= 0.999
        and up_min_cos >= 0.999
        and up_multi.cos >= 0.999
    )
    print("=================================================================")
    print("RESULT rfCos=", rf_multi.cos, " unipcCos=", up_multi.cos)
    if not ok:
        raise Error(
            String("cosmos sampler parity FAIL: rf_min=")
            + String(rf_min_cos)
            + " rf_multi="
            + String(rf_multi.cos)
            + " unipc_min="
            + String(up_min_cos)
            + " unipc_multi="
            + String(up_multi.cos)
        )
    print("PASS: Cosmos RF + UniPC step-parity vs canonical >= 0.999")
