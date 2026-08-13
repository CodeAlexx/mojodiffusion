# Numeric schedule and trajectory parity against pinned SCAIL-2
# wan/utils/fm_solvers_unipc.py at source commit 5cfe1b8.

from std.collections import List
from max.gpu.host import DeviceContext

from serenitymojo.io.dtype import STDtype
from serenitymojo.sampling.scail2_unipc import (
    build_scail2_unipc_sigma_schedule,
    build_scail2_unipc_timesteps,
)
from serenitymojo.sampling.unipc import UniPcMultistepScheduler
from serenitymojo.tensor import Tensor


comptime N = 6
comptime DIM = 8


def _abs(x: Float64) -> Float64:
    return x if x >= 0.0 else -x


def _sigmas40() -> List[Float32]:
    return [
        0.9996664524, 0.9911891222, 0.9824194908, 0.9733423591,
        0.9639410973, 0.9541981220, 0.9440944195, 0.9336096048,
        0.9227216840, 0.9114069343, 0.8996397853, 0.8873925209,
        0.8746352196, 0.8613352180, 0.8474572897, 0.8329627514,
        0.8178097010, 0.8019521832, 0.7853399515, 0.7679176927,
        0.7496247888, 0.7303943038, 0.7101522088, 0.6888164878,
        0.6662961245, 0.6424896717, 0.6172835827, 0.5905508399,
        0.5621483326, 0.5319145322, 0.4996665716, 0.4651961029,
        0.4282652140, 0.3886007667, 0.3458875120, 0.2997599542,
        0.2497916371, 0.1954820156, 0.1362396628, 0.0713605434, 0.0,
    ]


def _timesteps40() -> List[Float32]:
    return [
        999.0, 991.0, 982.0, 973.0, 963.0, 954.0, 944.0, 933.0,
        922.0, 911.0, 899.0, 887.0, 874.0, 861.0, 847.0, 832.0,
        817.0, 801.0, 785.0, 767.0, 749.0, 730.0, 710.0, 688.0,
        666.0, 642.0, 617.0, 590.0, 562.0, 531.0, 499.0, 465.0,
        428.0, 388.0, 345.0, 299.0, 249.0, 195.0, 136.0, 71.0,
    ]


def _x() -> List[Float32]:
    return [-0.75, -0.55, -0.35, -0.15, 0.0499999672, 0.2499999702, 0.45, 0.65]


def _velocity(i: Int) -> List[Float32]:
    if i == 0:
        return [0.1391987354, 0.2138095647, 0.2634242773, 0.2808277607, 0.2629986107, 0.2115501463, 0.1325506568, 0.0357452258]
    if i == 1:
        return [0.2158740908, 0.2884698212, 0.2265758663, 0.0575578399, -0.1364150792, -0.2606766820, -0.2568469048, -0.1326542795]
    if i == 2:
        return [0.2691622674, 0.2309989184, -0.0344148949, -0.2501865625, -0.1967350245, 0.0481492318, 0.1992137134, 0.0830943435]
    if i == 3:
        return [0.2915964425, 0.0678288713, -0.2447387725, -0.1182384491, 0.1784619093, 0.0801830664, -0.2391526848, -0.1814068258]
    if i == 4:
        return [0.2798184156, -0.1200502589, -0.1872039139, 0.1808361262, -0.0059016170, -0.2828073800, 0.0889592469, 0.2239439785]
    return [0.2350368947, -0.2392431647, 0.0583051443, 0.0796518624, -0.2866080999, 0.1664595753, 0.0731403381, -0.1719842404]


def _trajectory(i: Int) -> List[Float32]:
    if i == 0:
        return [-0.7587024570, -0.5633670092, -0.3664688468, -0.1675568521, 0.0335577577, 0.2367742211, 0.4417131543, 0.6477652192]
    if i == 1:
        return [-0.7786377668, -0.5890690684, -0.3834371567, -0.1646561772, 0.0579867139, 0.2736450732, 0.4754839242, 0.6641040444]
    if i == 2:
        return [-0.8113172054, -0.6092447639, -0.3600109518, -0.1150723100, 0.0825456008, 0.2436998636, 0.4183887243, 0.6383322477]
    if i == 3:
        return [-0.8565562963, -0.6020646095, -0.3042279780, -0.1174762547, 0.0123489201, 0.2324986458, 0.5100874901, 0.6983441710]
    if i == 4:
        return [-0.9167065620, -0.5417278409, -0.2761253119, -0.2154847383, 0.0488100871, 0.3658097088, 0.4314752221, 0.5738777518]
    return [-1.0047793388, -0.4520789385, -0.2979733348, -0.2453317791, 0.1562075019, 0.3034341931, 0.4040681720, 0.6383234859]


def main() raises:
    var sigmas = build_scail2_unipc_sigma_schedule(40, 3.0, 1000)
    var timesteps = build_scail2_unipc_timesteps(40, 3.0, 1000)
    var ref_sigmas = _sigmas40()
    var ref_timesteps = _timesteps40()
    var schedule_max = 0.0
    var timestep_max = 0.0
    for i in range(41):
        schedule_max = max(schedule_max, _abs(sigmas[i] - Float64(ref_sigmas[i])))
    for i in range(40):
        timestep_max = max(
            timestep_max, _abs(Float64(timesteps[i] - ref_timesteps[i]))
        )

    var six = build_scail2_unipc_sigma_schedule(N, 3.0, 1000)
    var scheduler = UniPcMultistepScheduler.from_sigmas(six^, 2)
    var ctx = DeviceContext()
    var x = Tensor.from_host(_x(), [1, 1, DIM], STDtype.F32, ctx)
    var trajectory_max = 0.0
    for step in range(N):
        var velocity = Tensor.from_host(
            _velocity(step), [1, 1, DIM], STDtype.F32, ctx
        )
        x = scheduler.step(velocity, x, ctx)
        var actual = x.to_host(ctx)
        var expected = _trajectory(step)
        for lane in range(DIM):
            trajectory_max = max(
                trajectory_max,
                _abs(Float64(actual[lane] - expected[lane])),
            )
    print("SCAIL-2 UniPC schedule max_abs=", schedule_max)
    print("SCAIL-2 UniPC timestep max_abs=", timestep_max)
    print("SCAIL-2 UniPC trajectory max_abs=", trajectory_max)
    if schedule_max > 1.0e-8 or timestep_max != 0.0 or trajectory_max > 5.0e-5:
        raise Error("SCAIL-2 UniPC parity FAIL")
    print("GATE PASS SCAIL-2 UniPC schedule and trajectory match creator")
