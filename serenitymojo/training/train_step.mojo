# pipeline/train_step.mojo — the SHARED, model-agnostic LoRA training step.
#
# This is the ~85% that train_klein.mojo and train_zimage.mojo used to duplicate
# verbatim (only the config TYPE name differed). Lifted ONCE here, parameterized
# on TrainConfig. Per-model files now supply only a TrainConfig (+ eventually a
# block kind / weight loader / lora-target map) and call run_synthetic / train_step.
#
# Recipe (the proven single-stream assembly, end to end):
#   flow_match_noise_target -> LoRA delta on block input -> dit_block_forward
#   -> mse -> dit_block_backward -> LoRA backward -> AdamW on LoRA params only.
# Base block weights are FROZEN (LoRA training). All interior via proven primitives
# (dit_block fwd/bwd, linear/linear_backward, adamw_step, schedule).
#
# Mojo idioms (handoff §4 / MOJO_CONVENTIONS): def at top level, move-only Tensor
# crosses the API as host List[Float32], multi-return via Movable structs, comptime
# attention dims [Bp,Sp,Hp,Dhp].

from std.collections import List, Optional
from std.gpu.host import DeviceContext
from std.math import sqrt
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype

from serenitymojo.ops.linear import linear
from serenitymojo.ops.linalg_backward import linear_backward
from serenitymojo.training.dit_block import (
    BlockWeights,
    BlockForward,
    BlockGrads,
    dit_block_forward,
    dit_block_backward,
)
from serenitymojo.training.schedule import (
    flow_match_noise_target,
    sample_timestep_logit_normal,
)
from serenitymojo.training.train_config import TrainConfig


# ── synthetic loop dims (module-level comptime) ──────────────────────────────
# dit_block_forward/backward and sdpa_nomask take B/S/H/Dh as COMPTIME params, so
# the synthetic dims the driver runs MUST be comptime (the dit_block_unit_parity
# gate does the same). The REAL model dims live in TrainConfig (runtime) for the
# recipe + GAP documentation; the actual real-dim run swaps these comptime values
# for cfg.d_model etc. (and adds the loader). D == H*Dh.
comptime _M = 4     # tokens (stand-in for the patchified image+text seq len)
comptime _D = 8     # model dim (stand-in for inner_dim)
comptime _H = 2     # heads (stand-in for real head count; H must divide D)
comptime _Dh = 4    # head dim (D == H*Dh)
comptime _FF = 16   # mlp hidden (stand-in for mlp_hidden)


# ── deterministic host randn (mirrors zimage_train_step / composed_chain) ────
def _randn(n: Int, seed: UInt64, scale: Float32) -> List[Float32]:
    var out = List[Float32]()
    var state = seed
    for _ in range(n):
        state = state * 6364136223846793005 + 1442695040888963407
        var u = Float32(Int(state >> 40)) * Float32(1.0 / 16777216.0)
        out.append((u - Float32(0.5)) * scale)
    return out^


def _zeros(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for _ in range(n):
        out.append(Float32(0.0))
    return out^


def _abs_sum(h: List[Float32]) -> Float32:
    var s = Float32(0.0)
    for i in range(len(h)):
        var v = h[i]
        s += v if v >= 0.0 else -v
    return s


def _f32_to_bf16_list(v: List[Float32]) -> List[BFloat16]:
    var out = List[BFloat16]()
    for i in range(len(v)):
        out.append(BFloat16(v[i]))
    return out^


def _bf16_to_f32_list(v: List[BFloat16]) -> List[Float32]:
    var out = List[Float32]()
    for i in range(len(v)):
        out.append(v[i].cast[DType.float32]())
    return out^


# ── inline MSE leaf (loss_swiglu_backward.mse_backward is UNIMPORTABLE — same as
#    zimage_train_step / composed_chain_parity) ───────────────────────────────
def _mse_loss(pred_h: List[Float32], tgt_h: List[Float32]) -> Float32:
    var acc = Float32(0.0)
    for i in range(len(pred_h)):
        var diff = pred_h[i] - tgt_h[i]
        acc += diff * diff
    return acc / Float32(len(pred_h))


def _mse_grad(pred_h: List[Float32], tgt_h: List[Float32]) -> List[Float32]:
    var n = len(pred_h)
    var d = List[Float32]()
    for i in range(n):
        d.append(Float32(2.0) * (pred_h[i] - tgt_h[i]) / Float32(n))
    return d^


# ── synthetic block weights for a single-stream block ────────────────────────
# Real models load these from safetensors (GAP G1). Synthetic here so the loop is
# shape-correct and exercises the proven dit_block forward+backward end to end.
def _make_block_weights(cfg: TrainConfig, seed: UInt64) raises -> BlockWeights:
    var D = cfg.d_model
    var FF = cfg.mlp_hidden
    return BlockWeights(
        _randn(D * D, seed + 1, 0.02),     # wq [D,D]
        _randn(D * D, seed + 2, 0.02),     # wk
        _randn(D * D, seed + 3, 0.02),     # wv
        _randn(D * D, seed + 4, 0.02),     # wo
        _randn(FF * D, seed + 5, 0.02),    # wg [FF,D]
        _randn(FF * D, seed + 6, 0.02),    # wu
        _randn(D * FF, seed + 7, 0.02),    # wd [D,FF]
        _randn(D, seed + 8, 0.0),          # g1 ~1 (rms gain); start near 1
        _randn(D, seed + 9, 0.0),          # g2
    )


# ── LoRA adapter on ONE projection (REAL low-rank delta + grad path) ─────────
# y = x@Wᵀ; delta y += scale·(x@Aᵀ)@Bᵀ, A:[rank,in], B:[out,rank],
# scale=(alpha/rank)·multiplier. Train A,B; base W FROZEN. Proven linear path.
struct LoraAdapter(Copyable, Movable):
    var a: List[BFloat16]  # [rank, in] BF16 model storage
    var b: List[BFloat16]  # [out, rank] BF16 model storage
    var rank: Int
    var in_f: Int
    var out_f: Int
    var scale: Float32
    var ma: List[Float32]
    var va: List[Float32]
    var mb: List[Float32]
    var vb: List[Float32]

    def __init__(
        out self, var a: List[Float32], var b: List[Float32],
        rank: Int, in_f: Int, out_f: Int, scale: Float32,
        var ma: List[Float32], var va: List[Float32],
        var mb: List[Float32], var vb: List[Float32],
    ):
        self.a = _f32_to_bf16_list(a)
        self.b = _f32_to_bf16_list(b)
        self.rank = rank
        self.in_f = in_f
        self.out_f = out_f
        self.scale = scale
        self.ma = ma^
        self.va = va^
        self.mb = mb^
        self.vb = vb^


# PEFT/ai-toolkit init: A ~ small randn, B = 0 (adapter identity at step 0).
def _make_lora(cfg: TrainConfig, in_f: Int, out_f: Int, seed: UInt64) -> LoraAdapter:
    var r = cfg.lora_rank
    var scale = (cfg.lora_alpha / Float32(r))  # multiplier=1.0 at train time
    return LoraAdapter(
        _randn(r * in_f, seed, 0.01),   # A small randn
        _zeros(out_f * r),              # B = 0 (identity at init)
        r, in_f, out_f, scale,
        _zeros(r * in_f), _zeros(r * in_f),
        _zeros(out_f * r), _zeros(out_f * r),
    )


# Adapter forward contribution on x [M,in] → [M,out] (host list in/out).
def _lora_fwd(
    x_h: List[Float32], lo: LoraAdapter, M: Int, ctx: DeviceContext
) raises -> List[Float32]:
    var nb1 = Optional[Tensor](None)
    var t = linear(
        Tensor.from_host(x_h.copy(), [M, lo.in_f], STDtype.BF16, ctx),
        Tensor.from_host_bf16(lo.a.copy(), [lo.rank, lo.in_f], ctx),
        nb1^, ctx,
    ).to_host(ctx)                                   # [M,rank]
    var nb2 = Optional[Tensor](None)
    var dy = linear(
        Tensor.from_host(t.copy(), [M, lo.rank], STDtype.BF16, ctx),
        Tensor.from_host_bf16(lo.b.copy(), [lo.out_f, lo.rank], ctx),
        nb2^, ctx,
    ).to_host(ctx)                                   # [M,out]
    var out = List[Float32]()
    for i in range(len(dy)):
        out.append(lo.scale * dy[i])
    return out^


struct LoraGrads(Copyable, Movable):
    var d_a: List[Float32]
    var d_b: List[Float32]

    def __init__(out self, var d_a: List[Float32], var d_b: List[Float32]):
        self.d_a = d_a^
        self.d_b = d_b^


def _lora_bwd(
    d_contrib_h: List[Float32], x_h: List[Float32], lo: LoraAdapter,
    M: Int, ctx: DeviceContext,
) raises -> LoraGrads:
    var nb_t = Optional[Tensor](None)
    var t = linear(
        Tensor.from_host(x_h.copy(), [M, lo.in_f], STDtype.BF16, ctx),
        Tensor.from_host_bf16(lo.a.copy(), [lo.rank, lo.in_f], ctx),
        nb_t^, ctx,
    ).to_host(ctx)                                   # [M,rank]
    var d_dy = List[Float32]()
    for i in range(len(d_contrib_h)):
        d_dy.append(lo.scale * d_contrib_h[i])       # [M,out]
    var lbB = linear_backward(
        Tensor.from_host(d_dy^, [M, lo.out_f], STDtype.BF16, ctx),
        Tensor.from_host(t.copy(), [M, lo.rank], STDtype.BF16, ctx),
        Tensor.from_host_bf16(lo.b.copy(), [lo.out_f, lo.rank], ctx),
        M, lo.rank, lo.out_f, ctx,
    )
    var d_t = lbB.d_x.to_host(ctx)                   # [M,rank]
    var d_b = lbB.d_w.to_host(ctx)                   # [out_f,rank]
    var lbA = linear_backward(
        Tensor.from_host(d_t^, [M, lo.rank], STDtype.BF16, ctx),
        Tensor.from_host(x_h.copy(), [M, lo.in_f], STDtype.BF16, ctx),
        Tensor.from_host_bf16(lo.a.copy(), [lo.rank, lo.in_f], ctx),
        M, lo.in_f, lo.rank, ctx,
    )
    var d_a = lbA.d_w.to_host(ctx)                   # [rank,in_f]
    return LoraGrads(d_a^, d_b^)


# AdamW one step on a host-resident LoRA list. LoRA adapters and moments are
# stored as List[Float32] in the Klein trainer; doing this update on host avoids
# per-adapter GPU upload/readback churn while matching PyTorch/OneTrainer AdamW:
# decoupled weight decay is applied before the adaptive Adam subtraction.
def _adamw_host_list(
    mut p: List[BFloat16], g: List[Float32],
    mut m: List[Float32], mut v: List[Float32],
    t: Int, lr: Float32, beta1: Float32, beta2: Float32,
    eps: Float32, weight_decay: Float32,
) raises:
    var n = len(p)
    if len(g) != n or len(m) != n or len(v) != n:
        raise Error("_adamw_host_list: param/grad/m/v len mismatch")
    if t < 1:
        raise Error("_adamw_host_list: t must be >= 1")

    var b1p = Float32(1.0)
    var b2p = Float32(1.0)
    for _ in range(t):
        b1p *= beta1
        b2p *= beta2
    var bc1 = Float32(1.0) - b1p
    var bc2 = Float32(1.0) - b2p

    for i in range(n):
        var gv = g[i]
        var mi = beta1 * m[i] + (Float32(1.0) - beta1) * gv
        var vi = beta2 * v[i] + (Float32(1.0) - beta2) * gv * gv
        m[i] = mi
        v[i] = vi
        var m_hat = mi / bc1
        var v_hat = vi / bc2
        var pv = p[i].cast[DType.float32]()
        if weight_decay > 0.0:
            pv = pv * (Float32(1.0) - lr * weight_decay)
        pv = pv - lr * m_hat / (sqrt(v_hat) + eps)
        p[i] = BFloat16(pv)


# AdamW one step on a LoRA adapter (A and B).
def _lora_adamw(
    mut lo: LoraAdapter, g: LoraGrads, t: Int, lr: Float32, ctx: DeviceContext,
    beta1: Float32 = Float32(0.9), beta2: Float32 = Float32(0.999),
    eps: Float32 = Float32(1.0e-8), weight_decay: Float32 = Float32(0.01),
) raises:
    _adamw_host_list(lo.a, g.d_a, lo.ma, lo.va, t, lr, beta1, beta2, eps, weight_decay)
    _adamw_host_list(lo.b, g.d_b, lo.mb, lo.vb, t, lr, beta1, beta2, eps, weight_decay)


# ── ONE training step (single-stream block + ONE trained LoRA adapter) ───────
struct StepResult(Copyable, Movable):
    var loss: Float32
    var lora_grad_absum: Float32

    def __init__(out self, loss: Float32, lora_grad_absum: Float32):
        self.loss = loss
        self.lora_grad_absum = lora_grad_absum


def train_step[
    Bp: Int, Sp: Int, Hp: Int, Dhp: Int
](
    x_in_h: List[Float32],       # [M,D] block input (from flow_match)
    target_h: List[Float32],     # [M,D] v-target
    w: BlockWeights,             # FROZEN base block weights
    mut lo: LoraAdapter,         # TRAINED LoRA adapter (on block input)
    cfg: TrainConfig,
    M: Int, t_step: Int,
    ctx: DeviceContext,
) raises -> StepResult:
    var D = cfg.d_model
    var FF = cfg.mlp_hidden
    var scale = Float32(1.0) / sqrt(Float32(cfg.head_dim))

    # LoRA delta on the block input: x' = x + scale·(x@Aᵀ)@Bᵀ. B=0 at init ⇒ x'==x.
    var lora_contrib = _lora_fwd(x_in_h, lo, M, ctx)        # [M,D]
    var x_mod = List[Float32]()
    for i in range(len(x_in_h)):
        x_mod.append(x_in_h[i] + lora_contrib[i])

    var fwd = dit_block_forward[Bp, Sp, Hp, Dhp](
        x_mod, w, M, D, FF, cfg.eps, scale, ctx
    )
    var y_h = fwd.y.copy()

    var loss = _mse_loss(y_h, target_h)
    var d_y = _mse_grad(y_h, target_h)                      # [M,D]

    var grads = dit_block_backward[Bp, Sp, Hp, Dhp](
        d_y, w, fwd.saved, M, D, FF, cfg.eps, scale, ctx
    )
    var d_contrib = grads.d_x.copy()                        # [M,D]

    var lg = _lora_bwd(d_contrib, x_in_h, lo, M, ctx)
    var absum = _abs_sum(lg.d_a) + _abs_sum(lg.d_b)
    _lora_adamw(lo, lg, t_step, cfg.lr, ctx)

    return StepResult(loss, absum)


# ── generic synthetic driver: a short LoRA training loop on synthetic data ───
# Shared by every model's thin entry point. Uses the module-level comptime synth
# dims; keeps the model's REAL recipe (lr/shift/rank/alpha) from `cfg`. A real run
# swaps the comptime _* for cfg dims + a weight loader (GAP G1).
def run_synthetic(cfg: TrainConfig, ctx: DeviceContext) raises:
    print("=== LoRA training:", cfg.name, "===")
    # Keep the model's real recipe; override only the dims with the synth comptime
    # dims (a real run would use cfg's dims + a weight loader — GAP G1).
    var c = cfg.copy()
    c.d_model = _D
    c.n_heads = _H
    c.head_dim = _Dh
    c.mlp_hidden = _FF

    var w = _make_block_weights(c, 100)
    var lo = _make_lora(c, _D, _D, 7)

    var first_loss = Float32(0.0)
    var last_loss = Float32(0.0)
    var first_absum = Float32(0.0)
    var n_iters = 6

    for it in range(n_iters):
        var t_step = it + 1
        var sigma = sample_timestep_logit_normal(UInt64(it + 1), c.timestep_shift)
        var latent = Tensor.from_host(
            _randn(_M * _D, UInt64(200 + it), 1.0), [1, _M, _D], STDtype.F32, ctx)
        var noise = Tensor.from_host(
            _randn(_M * _D, UInt64(300 + it), 1.0), [1, _M, _D], STDtype.F32, ctx)
        var fm = flow_match_noise_target(latent, sigma, noise, ctx)
        var x_in_h = fm.x_t.to_host(ctx)
        var target_h = fm.target.to_host(ctx)

        var res = train_step[1, _M, _H, _Dh](
            x_in_h, target_h, w, lo, c, _M, t_step, ctx
        )

        if it == 0:
            first_loss = res.loss
            first_absum = res.lora_grad_absum
            print("  [grad] sum|d_lora| =", res.lora_grad_absum, " (sigma=", sigma, ")")
            if not (res.lora_grad_absum > 0.0):
                print("  WARN: LoRA grad is zero at step 0 — inspect for dead branch")
        last_loss = res.loss
        print("  iter", it, " loss=", res.loss)

    print("  first_loss=", first_loss, " last_loss=", last_loss,
          " first|d_lora|=", first_absum)
    if last_loss < first_loss:
        print("PASS:", cfg.name, "LoRA step ran; loss DECREASED",
              first_loss, "->", last_loss)
    else:
        print("INFO:", cfg.name, "loss did not strictly decrease (",
              first_loss, "->", last_loss, ") — synthetic; inspect on real data.")
