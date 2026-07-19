# Ideogram4StackTrain.mojo — optimizer bridge for Ideogram4 block-stack LoRA.
#
# The model module computes d_A/d_B for every repeated transformer block. This
# trainer module owns AdamW state and applies those gradients to the LoRA set.

from std.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.tensor_algebra import zeros_device

from serenity_trainer.model.Ideogram4LoRABlock import (
    I4_SLOTS_PER_BLOCK,
    Ideogram4LoraSet,
    Ideogram4StackLoraGrads,
    ideogram4_stack_lora_backward,
    ideogram4_stack_lora_backward_graph,
    ideogram4_stack_lora_forward,
)
from serenity_trainer.util.config.TrainConfig import TrainConfig
from serenity_trainer.util.optimizer.adamw_extensions import adamw_step

# Shared device train-step ABI (MJ-1038 fused optimizer swap): the fused
# multitensor AdamW packs every adapter a/b into ONE kernel launch through
# serenitymojo's DeviceTrainableSet/DeviceGradSet/DeviceAdamWState contract
# (training/device_train_step.mojo:408, training/fused_adamw_multitensor.mojo).
from serenitymojo.training.device_train_step import (
    DeviceAdamWState,
    DeviceGradSet,
    DeviceTrainableSet,
    device_adamw_train_step_update,
)

comptime TArc = ArcPointer[Tensor]

# ── autograd_v2 GRAPH SWAP (P7 rollout, klein KLEIN_V2_GRAPH precedent) ───────
# When IDEOGRAM4_V2_GRAPH_PATH is on, the per-block recompute + hand-chain
# backward pair in the stack backward is driven by the autograd_v2 graph engine
# (ideogram4_stack_lora_backward_graph — same conductor loop, same slot fan-in,
# per-block coarse mini-graphs whose backward calls the hand-chain oracle;
# SAME-PROCESS bit gate: autograd_v2/tests/ideogram4_block_parity.mojo).
# ideogram4 is COARSE stage-1 = engine only, NO slab/capture (like Klein P6).
# False = the previous hand-chain path (C13 gate-don't-delete: the hand-chain in
# block.mojo stays compiled + reachable; flag default-OFF is byte-identical to
# today).
comptime IDEOGRAM4_V2_ENGINE = True
# DEFAULT-ON (2026-06-25): all trainers must dispatch backward through autograd_v2
# (Alex mandate). Gated by the per-block bit gate autograd_v2/tests/ideogram4_block_parity
# (engine grad == hand-chain grad BIT-EQUAL, lead-verified). Hand-chain stays reachable
# in the else branch (C13). End-to-end trainer N-step anchor pending the eri2 cache.
comptime IDEOGRAM4_V2_GRAPH = True
comptime IDEOGRAM4_V2_GRAPH_PATH = IDEOGRAM4_V2_ENGINE and IDEOGRAM4_V2_GRAPH

# ── FUSED MULTITENSOR ADAMW (MJ-1038, TRAINING_SPEED_AUDIT_2026-07-01.md) ─────
# True  = ONE fused kernel launch updates all 408 tensors (204 adapters × a/b)
#         through the shared device train-step ABI
#         (device_adamw_train_step_update -> fused_adamw_step). Replaces the
#         old per-tensor loop of ~408 adamw_step launches per optimizer step.
# False = the previous per-tensor adamw_step loop (C13 gate-don't-delete: kept
#         compiled + reachable in the comptime else branch below).
#
# NUMERICS DELTAS on the fused path (documented, NOT bit-identical — the
# fixture smoke smoke/ideogram4_fused_adamw_parity_smoke.mojo bounds them at
# ulp-class):
#   * m/v moment storage is F32 (DeviceAdamWState contract) instead of
#     Serenity's BF16-quantized moments — strictly higher moment precision.
#   * param write-back is the fused kernel's plain RNE cast; the fallback's
#     torch-RNE helper AND cfg.stochastic_rounding are NOT applied. With the
#     recipe default stochastic_rounding=True this is a semantic change
#     (bf16 SR exists because RNE can stall sub-ULP updates); flip this flag
#     to False to restore the exact Serenity-parity update.
#   * nonfinite grads FAIL LOUD (device_grad_stats) instead of being consumed.
# Optimizer-state checkpoints save whatever moment dtype is live, so a resume
# ACROSS a flag flip fails loud in either direction (DeviceAdamWState
# validate_for on F32-expected, adamw_step's BF16 check on the fallback).
comptime IDEOGRAM4_FUSED_ADAMW = True

# On-device global-norm grad clip (ai-toolkit trains ideogram4 with
# max_grad_norm=1.0; this trainer historically applied NO clip — a parity
# gap). False (default) preserves current numerics: clip_scale is forced to
# 1.0 by passing max_grad_norm=0.0 into the shared ABI. True folds
# cfg.clip_grad_norm (config default 1.0, matching ai-toolkit) into the fused
# update via the device global-norm fold — flipping it is the user's parity
# decision, not this campaign's. Only meaningful when IDEOGRAM4_FUSED_ADAMW
# is True (the fallback loop has no clip seam).
comptime IDEOGRAM4_GRAD_CLIP = False

# Progress-line telemetry cadence (MJ-1038 item 3): the full-tensor D2H L1
# readbacks (all B grads + all B params) run only every this-many optimizer
# steps (plus first/final step) instead of EVERY step. The fused path also
# returns a free per-step device grad_norm scalar (4-byte readback) in
# Ideogram4StackTrainResult.grad_norm.
comptime IDEOGRAM4_TELEMETRY_EVERY_STEPS = 10


struct Ideogram4LoraAdamState(Movable):
    var m_a: List[TArc]
    var v_a: List[TArc]
    var m_b: List[TArc]
    var v_b: List[TArc]

    def __init__(
        out self,
        var m_a: List[TArc],
        var v_a: List[TArc],
        var m_b: List[TArc],
        var v_b: List[TArc],
    ):
        self.m_a = m_a^
        self.v_a = v_a^
        self.m_b = m_b^
        self.v_b = v_b^


@fieldwise_init
struct Ideogram4StackTrainResult(Copyable, Movable):
    var adapters: Int
    var slots_per_block: Int
    var grad_b_l1: Float32
    var adapter_b_l1: Float32
    # Fused-path device scalars (MJ-1038): global L2 grad norm over ALL a/b
    # grads and the clip fold actually applied (1.0 = no clip). Both 0.0/1.0
    # on the per-tensor fallback path, which computes neither.
    var grad_norm: Float32
    var clip_scale: Float32


def make_ideogram4_lora_adam_state(
    loras: Ideogram4LoraSet, ctx: DeviceContext
) raises -> Ideogram4LoraAdamState:
    # Fused shared-ABI AdamW requires F32 m/v moment storage (DeviceAdamWState
    # contract, device_train_step.mojo:195); the per-tensor fallback keeps
    # Serenity's BF16-moment policy (adamw_extensions.mojo header). Checkpoints
    # save the live dtype, so resume across an IDEOGRAM4_FUSED_ADAMW flip
    # fails loud in either direction.
    var state_dtype = STDtype.BF16
    comptime if IDEOGRAM4_FUSED_ADAMW:
        state_dtype = STDtype.F32
    var m_a = List[TArc]()
    var v_a = List[TArc]()
    var m_b = List[TArc]()
    var v_b = List[TArc]()
    for i in range(len(loras.ad)):
        m_a.append(TArc(zeros_device(loras.ad[i][].a.shape(), state_dtype, ctx)))
        v_a.append(TArc(zeros_device(loras.ad[i][].a.shape(), state_dtype, ctx)))
        m_b.append(TArc(zeros_device(loras.ad[i][].b.shape(), state_dtype, ctx)))
        v_b.append(TArc(zeros_device(loras.ad[i][].b.shape(), state_dtype, ctx)))
    return Ideogram4LoraAdamState(m_a^, v_a^, m_b^, v_b^)


def apply_ideogram4_lora_grads(
    loras: Ideogram4LoraSet,
    mut state: Ideogram4LoraAdamState,
    grads: Ideogram4StackLoraGrads,
    optimizer_step: Int,
    cfg: TrainConfig,
    ctx: DeviceContext,
    want_l1_telemetry: Bool = True,
) raises -> Ideogram4StackTrainResult:
    if len(loras.ad) != len(grads.d_a) or len(loras.ad) != len(grads.d_b):
        raise Error("apply_ideogram4_lora_grads: adapter/grad count mismatch")
    if len(loras.ad) != len(state.m_a) or len(loras.ad) != len(state.m_b):
        raise Error("apply_ideogram4_lora_grads: optimizer state count mismatch")

    # Real LoRA-B gradient L1 (the 10-step gate reads result.grad_b_l1). Sum
    # |d_b| across every adapter BEFORE the optimizer step, so it reflects the
    # gradients that drove this step — matches the live trainer's grad_norm and
    # the levers path. Replaces the old grad_b_l1=0.0 / adapter_b_l1=0.0 stubs
    # that made the gate fail and the trainer look dead while it was learning.
    # MJ-1038: this is a FULL D2H of every B grad, so it is now cadence-gated —
    # want_l1_telemetry defaults True (gates/tests unchanged) and the live
    # trainer passes it every IDEOGRAM4_TELEMETRY_EVERY_STEPS steps only.
    var grad_b_l1 = Float32(0.0)
    if want_l1_telemetry:
        for i in range(len(loras.ad)):
            grad_b_l1 += _l1_sum(grads.d_b[i][], ctx)

    var grad_norm = Float32(0.0)
    var clip_scale = Float32(1.0)
    comptime if IDEOGRAM4_FUSED_ADAMW:
        # ONE fused multitensor update (MJ-1038): alias every live adapter a/b
        # (DeviceBuffer.copy() is a refcounted handle copy — same allocation,
        # the lora_adamw_plain_fused.mojo:918 packing precedent) plus the
        # already-boxed device grads into the shared ABI, then a single
        # device_adamw_train_step_update = one grad-stats launch + one fused
        # AdamW launch instead of ~408 per-tensor launches. Params/m/v update
        # IN PLACE through the shared buffers.
        var trainables = DeviceTrainableSet()
        var grad_set = DeviceGradSet()
        var adamw_state = DeviceAdamWState()
        for i in range(len(loras.ad)):
            var key_a = String("adapter.") + String(i) + String(".a")
            var key_b = String("adapter.") + String(i) + String(".b")
            trainables.append(
                key_a,
                TArc(Tensor(
                    loras.ad[i][].a.buf.copy(),
                    loras.ad[i][].a.shape(),
                    loras.ad[i][].a.dtype(),
                )),
                String("ideogram4-lora-a"),
            )
            trainables.append(
                key_b,
                TArc(Tensor(
                    loras.ad[i][].b.buf.copy(),
                    loras.ad[i][].b.shape(),
                    loras.ad[i][].b.dtype(),
                )),
                String("ideogram4-lora-b"),
            )
            grad_set.append(key_a, grads.d_a[i], String("ideogram4-lora-a"))
            grad_set.append(key_b, grads.d_b[i], String("ideogram4-lora-b"))
            adamw_state.append(state.m_a[i], state.v_a[i])
            adamw_state.append(state.m_b[i], state.v_b[i])
        # IDEOGRAM4_GRAD_CLIP=False keeps today's no-clip numerics: max_norm
        # 0.0 makes device_clip_scale return 1.0 (device_train_step.mojo:378).
        var max_grad_norm = Float32(0.0)
        comptime if IDEOGRAM4_GRAD_CLIP:
            max_grad_norm = cfg.clip_grad_norm
        var dres = device_adamw_train_step_update(
            trainables,
            grad_set,
            adamw_state,
            Float32(0.0),          # loss: threaded-through telemetry only
            optimizer_step,
            cfg.learning_rate,
            cfg.beta1,
            cfg.beta2,
            cfg.eps,
            cfg.weight_decay,
            max_grad_norm,
            ctx,
        )
        grad_norm = dres.grad_norm
        clip_scale = dres.clip_scale
    else:
        # C13 gate-don't-delete fallback: the exact pre-MJ-1038 per-tensor
        # loop (Serenity BF16 moments + torch-RNE/stochastic-rounding param
        # write-back), ~408 launches per step.
        for i in range(len(loras.ad)):
            adamw_step(
                loras.ad[i][].a,
                state.m_a[i][],
                state.v_a[i][],
                grads.d_a[i][],
                optimizer_step,
                cfg.learning_rate,
                cfg.beta1,
                cfg.beta2,
                cfg.eps,
                cfg.weight_decay,
                cfg.stochastic_rounding,
                cfg.seed + UInt32(optimizer_step + i),
                ctx,
            )
            adamw_step(
                loras.ad[i][].b,
                state.m_b[i][],
                state.v_b[i][],
                grads.d_b[i][],
                optimizer_step,
                cfg.learning_rate,
                cfg.beta1,
                cfg.beta2,
                cfg.eps,
                cfg.weight_decay,
                cfg.stochastic_rounding,
                cfg.seed + UInt32(optimizer_step + 100000 + i),
                ctx,
            )

    # Post-step LoRA-B param L1 (the live trainer reads result.adapter_b_l1 for
    # its last_b summary; mirrors the levers path's returned b_l1). Cadence-
    # gated like grad_b_l1 above (full D2H of every B param).
    var adapter_b_l1 = Float32(0.0)
    if want_l1_telemetry:
        for i in range(len(loras.ad)):
            adapter_b_l1 += _l1_sum(loras.ad[i][].b, ctx)

    return Ideogram4StackTrainResult(
        len(loras.ad), I4_SLOTS_PER_BLOCK, grad_b_l1, adapter_b_l1,
        grad_norm, clip_scale,
    )


def train_ideogram4_stack_lora_step[
    S: Int, Hidden: Int, Heads: Int, Dh: Int, FF: Int, Adaln: Int,
](
    x_in: Tensor,
    adaln_input: Tensor,
    cosf: Tensor,
    sinf: Tensor,
    transformer_weights: ShardedSafeTensors,
    loras: Ideogram4LoraSet,
    mut opt_state: Ideogram4LoraAdamState,
    d_stack_out: Tensor,
    optimizer_step: Int,
    cfg: TrainConfig,
    ctx: DeviceContext,
) raises -> Ideogram4StackTrainResult:
    var fwd = ideogram4_stack_lora_forward[S, Hidden, Heads, Dh, FF, Adaln](
        x_in, adaln_input, cosf, sinf, transformer_weights, loras, ctx
    )
    comptime if IDEOGRAM4_V2_GRAPH_PATH:
        # P7: per-block graph-engine backward (same conductor loop, same slot
        # fan-in, same arg list — drop-in for the hand-chain call below; bit
        # gate = ideogram4_block_parity).
        var grads = ideogram4_stack_lora_backward_graph[S, Hidden, Heads, Dh, FF, Adaln](
            d_stack_out, adaln_input, cosf, sinf, transformer_weights, loras, fwd^, ctx
        )
        return apply_ideogram4_lora_grads(
            loras, opt_state, grads^, optimizer_step, cfg, ctx
        )
    else:
        var grads = ideogram4_stack_lora_backward[S, Hidden, Heads, Dh, FF, Adaln](
            d_stack_out, adaln_input, cosf, sinf, transformer_weights, loras, fwd^, ctx
        )
        return apply_ideogram4_lora_grads(
            loras, opt_state, grads^, optimizer_step, cfg, ctx
        )


def _l1_sum(x: Tensor, ctx: DeviceContext) raises -> Float32:
    var host = x.to_host(ctx)
    var s = Float32(0.0)
    for i in range(len(host)):
        var v = host[i]
        if v < Float32(0.0):
            s -= v
        else:
            s += v
    return s


# ══════════════════════════════════════════════════════════════════════════════
# T1.C OPTIMIZER LEVERS BRIDGE (TIER1_PARITY_CAMPAIGN_2026-06-11.md).
#
# serenitymojo's levers optimizers (adafactor / schedule-free AdamW,
# training/levers.mojo levers_optimizer_step_host) step HOST
# serenitymojo.training.train_step.LoraAdapter mirrors (List[BFloat16] a/b),
# while ideogram4's adapters are DEVICE Tensors (module/LoRAModule.LoraAdapter
# inside Ideogram4LoraSet) stepped in place by the fused adamw_step above. The
# structs do not match, so this bridge implements the documented
# adapter-mirror host path:
#   lazy init (step 1): download every device a/b into a host mirror set
#   per step:           grads D2H -> levers_optimizer_step_host on the mirrors
#                       -> RNE-bf16 mirrors re-uploaded into the LoraSet's
#                          device tensors (Tensor.from_host_bf16, the same
#                          host->device construction the LoRA loader uses)
# The mirrors stay authoritative for params while the lever is active (the
# device AdamW never runs), so no per-step download is needed after init.
# Correctness over speed: ~2 H2D/D2H sweeps of the 204-adapter set per step.
#
# DEFAULT-OFF CONTRACT (C13): the trainer seam is
#   `if levers_optimizer_active(lcfg): ideogram4_levers_optimizer_step(...)
#    else: <the existing literal apply_ideogram4_lora_grads call>`
# — optimizer=ADAMW (the config default) never enters this section.
#
# RESUME: levers optimizer state has no save/resume sidecar (levers.mojo T1.C
# header) — levers_optimizer_step_host fails loud when its first call arrives
# at k != 1, which covers ideogram4 resume runs.
# ══════════════════════════════════════════════════════════════════════════════

from serenitymojo.training.train_step import LoraAdapter as SmLoraAdapter
from serenitymojo.training.levers import (
    LeversOptimizerState,
    levers_optimizer_active,
    levers_optimizer_step_host,
)
from serenitymojo.training.train_config import TrainConfig as LeversConfig


struct Ideogram4LeversBridge(Movable):
    """Host mirror set + levers optimizer state for one train run. Also the
    EMA tracking target (training/lora_ema.mojo wants the same host-mirror
    shape): with the default AdamW the driver refreshes the mirrors from
    device after each step; with a levers optimizer they are already live."""

    var mirrors: List[SmLoraAdapter]
    var mirrors_inited: Bool
    var opt_st: LeversOptimizerState

    def __init__(out self):
        self.mirrors = List[SmLoraAdapter]()
        self.mirrors_inited = False
        self.opt_st = LeversOptimizerState()


def ideogram4_levers_mirrors_init(
    mut bridge: Ideogram4LeversBridge,
    loras: Ideogram4LoraSet,
    ctx: DeviceContext,
) raises:
    """Download the device LoRA set into fresh host mirrors (bf16-exact:
    device bf16 -> F32 upcast -> the SmLoraAdapter ctor's RNE bf16 re-round is
    the identity on bf16-representable values). Adam moment lists are left
    EMPTY — the levers optimizers and lora_ema only touch a/b/shape fields."""
    if bridge.mirrors_inited:
        return
    for i in range(len(loras.ad)):
        var a_sh = loras.ad[i][].a.shape()   # [rank, in]
        var b_sh = loras.ad[i][].b.shape()   # [out, rank]
        var rank = loras.ad[i][].rank
        if len(a_sh) != 2 or len(b_sh) != 2 or a_sh[0] != rank or b_sh[1] != rank:
            raise Error(
                String("ideogram4_levers_mirrors_init: adapter ") + String(i)
                + String(" has unexpected a/b shape")
            )
        bridge.mirrors.append(
            SmLoraAdapter(
                loras.ad[i][].a.to_host(ctx),
                loras.ad[i][].b.to_host(ctx),
                rank, a_sh[1], b_sh[0],
                loras.ad[i][].scale(),
                List[Float32](), List[Float32](),
                List[Float32](), List[Float32](),
            )
        )
    bridge.mirrors_inited = True


def ideogram4_levers_refresh_mirrors(
    mut bridge: Ideogram4LeversBridge,
    loras: Ideogram4LoraSet,
    ctx: DeviceContext,
) raises:
    """Re-pull the live device a/b into the host mirrors (verbatim bf16).
    Needed each step ONLY on the default fused-AdamW path when EMA is on —
    the levers optimizer path keeps the mirrors authoritative itself."""
    if not bridge.mirrors_inited or len(bridge.mirrors) != len(loras.ad):
        raise Error("ideogram4_levers_refresh_mirrors: mirrors not initialized")
    for i in range(len(loras.ad)):
        bridge.mirrors[i].a = loras.ad[i][].a.to_host_bf16(ctx)
        bridge.mirrors[i].b = loras.ad[i][].b.to_host_bf16(ctx)


def ideogram4_levers_optimizer_step(
    lcfg: LeversConfig,
    loras: Ideogram4LoraSet,
    mut bridge: Ideogram4LeversBridge,
    grads: Ideogram4StackLoraGrads,
    k: Int,
    step_lr: Float32,
    ctx: DeviceContext,
) raises -> Float32:
    """The ONE ideogram4 call for the T1.C optimizer lever: host
    adafactor/schedule-free step over ALL adapters' mirrors, then re-upload
    into the LoraSet's live device tensors. step_lr = the trainer-scheduled lr
    (ideogram4 has no LR scheduler -> the constant cfg.learning_rate; the
    schedule-free path ignores it and uses lcfg.lr per the levers contract).
    Returns the post-step LoRA-B |.|_1 (host, for the progress summary).
    Call ONLY when levers_optimizer_active(lcfg) (C13)."""
    ideogram4_levers_mirrors_init(bridge, loras, ctx)
    if len(grads.d_a) != len(bridge.mirrors) or len(grads.d_b) != len(bridge.mirrors):
        raise Error("ideogram4_levers_optimizer_step: grad/mirror count mismatch")

    var d_a_h = List[List[Float32]]()
    var d_b_h = List[List[Float32]]()
    for i in range(len(bridge.mirrors)):
        d_a_h.append(grads.d_a[i][].to_host(ctx))
        d_b_h.append(grads.d_b[i][].to_host(ctx))

    levers_optimizer_step_host(
        lcfg, bridge.mirrors, d_a_h, d_b_h, k, step_lr,
        0, len(bridge.mirrors), bridge.opt_st,
    )

    var b_l1 = Float32(0.0)
    for i in range(len(bridge.mirrors)):
        loras.ad[i][].a = Tensor.from_host_bf16(
            bridge.mirrors[i].a, loras.ad[i][].a.shape(), ctx
        )
        loras.ad[i][].b = Tensor.from_host_bf16(
            bridge.mirrors[i].b, loras.ad[i][].b.shape(), ctx
        )
        for j in range(len(bridge.mirrors[i].b)):
            var v = bridge.mirrors[i].b[j].cast[DType.float32]()
            if v < Float32(0.0):
                b_l1 -= v
            else:
                b_l1 += v
    return b_l1
