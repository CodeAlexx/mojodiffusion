# Anima cached-conditioning tensor smoke.
#
# Loads the Rust-captured Anima `context_cond`/`context_uncond` sidecar and the
# cached-context latent oracle into Mojo tensors, then exercises the Anima
# linear FlowMatch CFG/Euler runtime surface without porting MiniTrainDIT yet.

from std.gpu.host import DeviceContext

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.models.dit.anima_contract import (
    ANIMA_ADAPTER_DIM,
    ANIMA_LATENT_CHANNELS,
    ANIMA_LATENT_H,
    ANIMA_LATENT_T,
    ANIMA_LATENT_W,
    ANIMA_MAX_SEQ_LEN,
    ANIMA_NUM_STEPS,
    anima_default_conditioning_path,
    anima_default_rust_latent_path,
    validate_anima_conditioning_header,
    validate_anima_rust_latent_header,
)
from serenitymojo.sampling.anima_sampling import (
    AnimaLinearFlowScheduler,
    anima_cfg,
    anima_euler_step,
)
from serenitymojo.tensor import Tensor


def _abs(v: Float32) -> Float32:
    if v < 0.0:
        return -v
    return v


def _check_close(name: String, got: Float32, expected: Float32, tol: Float32) raises:
    if _abs(got - expected) > tol:
        raise Error(
            String("Anima tensor smoke mismatch: ")
            + name
            + String(" got=")
            + String(got)
            + String(" expected=")
            + String(expected)
        )


def _load_named(
    st: ShardedSafeTensors, name: String, ctx: DeviceContext
) raises -> Tensor:
    var tv = st.tensor_view(name)
    return Tensor.from_view(tv, ctx)


def _check_shape(name: String, t: Tensor, expected: List[Int]) raises:
    var shape = t.shape()
    if len(shape) != len(expected):
        raise Error(String("Anima rank mismatch for ") + name)
    for i in range(len(expected)):
        if shape[i] != expected[i]:
            raise Error(String("Anima shape mismatch for ") + name)


def _max_abs_diff(a: Tensor, b: Tensor, ctx: DeviceContext) raises -> Float32:
    var ah = a.to_host(ctx)
    var bh = b.to_host(ctx)
    if len(ah) != len(bh):
        raise Error("Anima diff length mismatch")
    var max_diff: Float32 = 0.0
    for i in range(len(ah)):
        var d = _abs(ah[i] - bh[i])
        if d > max_diff:
            max_diff = d
    return max_diff


def _mean_abs(t: Tensor, ctx: DeviceContext) raises -> Float64:
    var h = t.to_host(ctx)
    if len(h) == 0:
        raise Error("Anima empty tensor")
    var total: Float64 = 0.0
    for i in range(len(h)):
        var v = h[i]
        if v < 0.0:
            total += Float64(-v)
        else:
            total += Float64(v)
    return total / Float64(len(h))


def main() raises:
    var ctx = DeviceContext()
    print("=== Anima cached-conditioning tensor smoke ===")

    var embeddings_path = anima_default_conditioning_path()
    var cond_contract = validate_anima_conditioning_header(embeddings_path)
    var emb = ShardedSafeTensors.open(embeddings_path)
    var context_cond = _load_named(emb, String("context_cond"), ctx)
    var context_uncond = _load_named(emb, String("context_uncond"), ctx)
    var context_shape = List[Int]()
    context_shape.append(1)
    context_shape.append(ANIMA_MAX_SEQ_LEN)
    context_shape.append(ANIMA_ADAPTER_DIM)
    _check_shape(String("context_cond"), context_cond, context_shape.copy())
    _check_shape(String("context_uncond"), context_uncond, context_shape.copy())

    if context_cond.dtype() != context_uncond.dtype():
        raise Error("Anima conditioning dtype mismatch after H2D")
    if context_cond.dtype() != STDtype.BF16 and context_cond.dtype() != STDtype.F32:
        raise Error("Anima conditioning H2D dtype must be BF16 or F32")

    # guidance=1.0 should reduce to the conditional branch across the full
    # cached sidecar tensor; this exercises the real sidecar shape on GPU.
    var context_guided = anima_cfg(context_cond, context_uncond, 1.0, ctx)
    var context_diff = _max_abs_diff(context_guided, context_cond, ctx)
    if context_diff > 0.0001:
        raise Error("Anima sidecar CFG(guidance=1) should equal context_cond")

    var latent_path = anima_default_rust_latent_path()
    _ = validate_anima_rust_latent_header(latent_path)
    var latent_st = ShardedSafeTensors.open(latent_path)
    var latent = _load_named(latent_st, String("latent"), ctx)
    var latent_shape = List[Int]()
    latent_shape.append(1)
    latent_shape.append(ANIMA_LATENT_CHANNELS)
    latent_shape.append(ANIMA_LATENT_T)
    latent_shape.append(ANIMA_LATENT_H)
    latent_shape.append(ANIMA_LATENT_W)
    _check_shape(String("latent"), latent, latent_shape.copy())
    if latent.dtype() != STDtype.F32:
        raise Error("Anima Rust latent oracle should load as F32")

    var zeros = List[Float32]()
    for _ in range(latent.numel()):
        zeros.append(0.0)
    var zero_velocity = Tensor.from_host(zeros^, latent_shape.copy(), STDtype.F32, ctx)

    var sched = AnimaLinearFlowScheduler.default_30()
    var sigmas = sched.sigmas()
    if len(sigmas) != ANIMA_NUM_STEPS + 1:
        raise Error("Anima sigma schedule length mismatch")
    _check_close(String("sigma[0]"), sigmas[0], 1.0, 0.000001)
    _check_close(String("sigma[15]"), sigmas[15], 0.5, 0.000001)
    _check_close(String("sigma[30]"), sigmas[30], 0.0, 0.000001)
    _check_close(String("model_timestep[0]"), sched.model_timestep(0), 1.0, 0.000001)
    for i in range(ANIMA_NUM_STEPS):
        if sched.dt(i) >= 0.0:
            raise Error("Anima Euler delta must be negative")

    var stepped_latent = sched.step(latent, zero_velocity, 0, ctx)
    var latent_diff = _max_abs_diff(stepped_latent, latent, ctx)
    if latent_diff != 0.0:
        raise Error("Anima zero-velocity latent step should be exact")

    var sh = List[Int]()
    sh.append(2)
    sh.append(2)
    var uncond = Tensor.from_host([1.0, 2.0, 3.0, 4.0], sh.copy(), STDtype.F32, ctx)
    var cond = Tensor.from_host([2.0, 4.0, 6.0, 8.0], sh.copy(), STDtype.F32, ctx)
    var guided = anima_cfg(cond, uncond, 4.5, ctx)
    var gv = guided.to_host(ctx)
    _check_close(String("cfg[0]"), gv[0], 5.5, 0.000001)
    _check_close(String("cfg[3]"), gv[3], 22.0, 0.000001)

    var velocity = Tensor.from_host([0.5, 1.0, -0.5, -1.0], sh.copy(), STDtype.F32, ctx)
    var stepped = anima_euler_step(uncond, velocity, sched.dt(0), ctx)
    var sv = stepped.to_host(ctx)
    _check_close(String("step[0]"), sv[0], 1.0 + 0.5 * sched.dt(0), 0.000001)
    _check_close(String("step[3]"), sv[3], 4.0 - 1.0 * sched.dt(0), 0.000001)

    print(
        "[anima-context] cond tokens/hidden/dtype=",
        cond_contract.text_tokens,
        cond_contract.hidden,
        context_cond.dtype().name(),
    )
    print("[anima-context] context mean_abs=", _mean_abs(context_cond, ctx))
    print("[anima-context] latent mean_abs=", _mean_abs(latent, ctx))
    print("[anima-context] sidecar_cfg_max_diff=", context_diff)
    print("Anima cached-conditioning tensor smoke PASS")
