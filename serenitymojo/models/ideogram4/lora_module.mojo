# lora.mojo — LoRA adapter, ported from Serenity modules/module/LoRAModule.py.
# Forward (LoRAModule.forward, line 328-329):
#   out = orig_forward(x) + lora_up(dropout(lora_down(x))) * (alpha/rank)
# Init: lora_down = kaiming_uniform_(a=sqrt(5)); lora_up = zeros_  (PEFT identity
# at step 0 → B=0). Save format (PEFT/ai-toolkit): <prefix>.lora_A.weight (=down),
# <prefix>.lora_B.weight (=up).
#
# Reuses ONLY mojodiffusion serenitymojo {autograd tape, tensor, ops}. The LoRA
# path is recorded on the tape (grads flow to A and B); the frozen base linear is
# untracked. BF16 storage throughout.

from std.math import sqrt
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.autograd import Tape
from serenitymojo.ops.linear import linear
from serenitymojo.ops.random import randn
from serenitymojo.ops.tensor_algebra import (
    zeros_device, mul_scalar, add_scalar, mul,
)
from serenitymojo.ops.reduce import reduce_sum_f32
from serenitymojo.ops.unary import sqrt_op, reciprocal_op
from serenitymojo.ops.cast import cast_tensor


# A = lora_down.weight [rank, in_features]; B = lora_up.weight [out_features, rank].
struct LoraAdapter(Movable):
    var a: Tensor          # [rank, in]  (BF16, trained)
    var b: Tensor          # [out, rank] (BF16, trained; zero-init)
    var rank: Int
    var alpha: Float32

    def __init__(out self, var a: Tensor, var b: Tensor, rank: Int, alpha: Float32):
        self.a = a^
        self.b = b^
        self.rank = rank
        self.alpha = alpha

    # alpha/rank scale (LoRAModule.forward, line 329: `* (self.alpha / self.rank)`).
    def scale(self) -> Float32:
        return self.alpha / Float32(self.rank)

    # Make a/b trainable on the tape (assigns fresh ids). The frozen base weight
    # is deliberately NOT tracked. Call once per step before lora_linear_forward.
    def track(mut self, mut tape: Tape):
        tape.track(self.a)
        tape.track(self.b)


# Build a fresh adapter for a Linear of shape [out, in].
# A ~ kaiming_uniform(a=sqrt(5)) approximated by Gaussian with matching std
# (bound = 1/sqrt(in) → uniform std = 1/sqrt(3*in)); B = 0 (PEFT identity).
def make_lora_adapter(
    in_features: Int, out_features: Int, rank: Int, alpha: Float32,
    seed: UInt64, ctx: DeviceContext,
) raises -> LoraAdapter:
    var a_sh = List[Int](); a_sh.append(rank); a_sh.append(in_features)
    var b_sh = List[Int](); b_sh.append(out_features); b_sh.append(rank)
    var std = Float32(1.0) / sqrt(Float32(3 * in_features))
    var a_raw = randn(a_sh^, seed, STDtype.BF16, ctx)
    var a = mul_scalar(a_raw, std, ctx)
    var b = zeros_device(b_sh^, STDtype.BF16, ctx)
    return LoraAdapter(a^, b^, rank, alpha)


# LoRA-wrapped linear forward, recorded on the tape. `base_w` is the FROZEN base
# weight [out, in] (untracked → no grad). Grads flow to adapter.a and adapter.b.
# Returns out = base(x) + ((x @ Aᵀ) @ Bᵀ) * (alpha/rank).
# `x` is [M, in] (caller flattens leading dims). adapter.a / adapter.b must be
# tape.track()'d by the caller before the step.
#
# DROPOUT NOTE: Serenity's forward is lora_up(dropout(lora_down(x)))*scale
# (LoRAModule.py:328). The dropout layer defaults to Dropout(0) (LoRAModule.py:302),
# i.e. a no-op, so this forward is bit-equivalent for the default/inference path.
# Training-time dropout>0 is NOT modeled here — if a recipe sets lora_dropout>0,
# this path must be extended with a tape-recorded dropout op before it is faithful.
def lora_linear_forward(
    mut tape: Tape,
    x: Tensor,
    base_w: Tensor,
    adapter: LoraAdapter,
    ctx: DeviceContext,
) raises -> Tensor:
    # frozen base path (untracked): base = x @ base_wᵀ  (no bias)
    var zb = zeros_device(_out_dim_shape(base_w), STDtype.BF16, ctx)
    var base = linear(x, base_w, Optional[Tensor](zb^), ctx)

    # LoRA path on the tape: down then up (both no-bias linears → zero bias).
    var zb1 = zeros_device(_rank_shape(adapter.rank), STDtype.BF16, ctx)
    var down = tape.record_linear(x, adapter.a, zb1, ctx)         # [M, rank]
    var zb2 = zeros_device(_out_dim_shape(adapter.b), STDtype.BF16, ctx)
    var up = tape.record_linear(down, adapter.b, zb2, ctx)        # [M, out]

    # scale by alpha/rank via a tape mul against a constant (untracked) tensor,
    # so the gradient is scaled correctly through the LoRA path (grad to a/b is
    # multiplied by the saved scale tensor in record_mul's backward).
    #
    # PRECISION NOTE (deviation from Serenity, LoRAModule.py:329 which applies
    # `alpha/rank` as an F32 python float): the scale is materialized as a BF16
    # constant tensor here. For non-dyadic ratios (e.g. alpha=8,rank=6 → 1.333…)
    # the scale rounds to the nearest BF16 (≈1.328) before the multiply, slightly
    # perturbing both the forward output and the back-propagated grad to a/b.
    # This is UNAVOIDABLE under the unit's constraints: the only tape-recorded
    # multiply (record_mul) takes a Tensor operand (not an F32 scalar), and the
    # DTYPE contract forbids a persistent F32 tensor — so the scale cannot be
    # carried at F32 through the tape. Power-of-two-friendly ratios (alpha/rank a
    # dyadic value, e.g. 8/4=2.0) are represented exactly and incur no error.
    var sc = _const_like(up, adapter.scale(), ctx)
    var scaled = tape.record_mul(up, sc, ctx)

    # out = base + scaled. base untracked (id 0) → drops its grad; grad flows to
    # `scaled` and thence to a/b.
    return tape.record_add(base, scaled, ctx)


# --- small shape helpers ------------------------------------------------------
def _out_dim_shape(w: Tensor) raises -> List[Int]:
    var s = List[Int](); s.append(w.shape()[0]); return s^

def _rank_shape(rank: Int) -> List[Int]:
    var s = List[Int](); s.append(rank); return s^

def _const_like(t: Tensor, val: Float32, ctx: DeviceContext) raises -> Tensor:
    # a BF16 tensor of t's shape filled with `val` (untracked): zeros + val.
    # NOTE: `val` (the F32 scale) is rounded to BF16 here — see PRECISION NOTE in
    # lora_linear_forward. shape() already returns an owned copy.
    var z = zeros_device(t.shape(), STDtype.BF16, ctx)
    return add_scalar(z, val, ctx)


# ═════════════════════════════════════════════════════════════════════════════
# LyCORIS variants — ported from the SAME Serenity file
# (modules/module/LoRAModule.py): LoHaModule (:213-284), DoRAModule (:473-567),
# plus a Kronecker (LoKr) adapter. NB: Serenity's LoRAModule.py does NOT define
# a LoKr/Kronecker class — only LoHa, LoRA, OFT, DoRA. The Kronecker adapter
# below is provided per this slice's spec (the "Kronecker forward" the prompt
# names); its construction follows the LyCORIS LoKr definition, with the same
# `make_weight`/`alpha/rank` scaling convention Serenity uses for the others.
#
# These adapters are the layer-independent (Linear-only) forwards. They mirror
# the PeftBase.make_weight contract (LoRAModule.py:80-95):
#   W = B.view(B.size(0),-1) @ A.view(A.size(0),-1)        (:94)
# and the per-class forward, recorded on the tape so grads flow to the factors.
# The frozen base path (orig_forward(x)) is run UNTRACKED, exactly as in LoRA.


# small helper: y = x @ Wᵀ recorded on the tape (the F.linear op the PEFT classes
# call as `self.op(x, W, bias=None)` — LoRAModule.py:276). record_linear wants a
# bias; we pass an untracked zero. W is [out,in] → linear does x @ Wᵀ internally.
def _tape_linear_nobias(
    mut tape: Tape, x: Tensor, w: Tensor, out_features: Int, ctx: DeviceContext
) raises -> Tensor:
    var bsh = List[Int](); bsh.append(out_features)
    var zb = zeros_device(bsh^, STDtype.BF16, ctx)
    return tape.record_linear(x, w, zb, ctx)


# ─────────────────────────────────────────────────────────────────────────────
# LoHaModule — Hadamard-product low-rank adaptation (LoRAModule.py:213-284).
#
# Factors (LoRAModule.py:223-226, init :248-256):
#   hada_w1_a [rank,in], hada_w1_b [out,rank]
#   hada_w2_a [rank,in], hada_w2_b [out,rank]
#   init (LoRAModule.py:246,253-256): create_layer() returns (down[rank,in],
#   up[out,rank]) but LoHa unpacks them as (w1_b, w1_a) — so the [rank,in] DOWN
#   factor is named hada_w1_b and the [out,rank] UP factor is named hada_w1_a.
#   nn.init then sets std=0.1 on the UP [out,rank] factor and std=1 on the DOWN
#   [rank,in] factor. In THIS port's naming (w1_a=[rank,in] down, w1_b=[out,rank]
#   up) that is: w1_a(down)~N(0,1) w1_b(up)~N(0,0.1) w2_a(down)~N(0,1) w2_b(up)=0.
# Forward (LoRAModule.py:271-276):
#   W1 = make_weight(w1_b, w1_a) = w1_b @ w1_a              ([out,in])
#   W2 = make_weight(w2_b, w2_a) = w2_b @ w2_a              ([out,in])
#   W  = (W1 * W2) * (alpha / rank)        # elementwise Hadamard  (:275)
#   return orig_forward(x) + op(x, W)      # op = F.linear → x @ Wᵀ   (:276)
# NB the A/B order is FLIPPED vs make_weight's docstring (LoRAModule.py:269-270):
# make_weight(B,A)=B@A, and LoHa passes (w_b, w_a) → W = w_b @ w_a.
struct LoHaAdapter(Movable):
    var w1_a: Tensor      # [rank, in]
    var w1_b: Tensor      # [out, rank]
    var w2_a: Tensor      # [rank, in]
    var w2_b: Tensor      # [out, rank]
    var rank: Int
    var in_features: Int
    var out_features: Int
    var alpha: Float32

    def __init__(
        out self, var w1_a: Tensor, var w1_b: Tensor, var w2_a: Tensor, var w2_b: Tensor,
        rank: Int, in_features: Int, out_features: Int, alpha: Float32,
    ):
        self.w1_a = w1_a^
        self.w1_b = w1_b^
        self.w2_a = w2_a^
        self.w2_b = w2_b^
        self.rank = rank
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha

    # alpha/rank scale (LoRAModule.py:275).
    def scale(self) -> Float32:
        return self.alpha / Float32(self.rank)

    # Track all four factors on the tape (assign fresh ids; frozen base untracked).
    def track(mut self, mut tape: Tape):
        tape.track(self.w1_a)
        tape.track(self.w1_b)
        tape.track(self.w2_a)
        tape.track(self.w2_b)


# make_weight(B,A) = B @ A, recorded on the tape (LoRAModule.py:94). B:[out,rank],
# A:[rank,in] → W:[out,in]. record_matmul does A@B in that argument order, so we
# pass (B, A) → B@A. Grads flow to both factors.
def _hada_make_weight(
    mut tape: Tape, b: Tensor, a: Tensor, ctx: DeviceContext
) raises -> Tensor:
    return tape.record_matmul(b, a, ctx)     # [out,rank] @ [rank,in] = [out,in]


# LoHa forward (LoRAModule.py:271-276). `base_w` is the FROZEN base weight [out,in]
# (untracked → no grad). Returns orig(x) + (x @ Wᵀ) with W=(W1⊙W2)*(alpha/rank).
# adapter factors must be tape.track()'d by the caller before this call.
def loha_forward(
    mut tape: Tape, x: Tensor, base_w: Tensor, adapter: LoHaAdapter, ctx: DeviceContext
) raises -> Tensor:
    # frozen base path (untracked): base = x @ base_wᵀ
    var base = _tape_linear_nobias_untracked(x, base_w, adapter.out_features, ctx)

    # W1 = w1_b @ w1_a ; W2 = w2_b @ w2_a   (both [out,in], on the tape)  (:271-274)
    var w1 = _hada_make_weight(tape, adapter.w1_b, adapter.w1_a, ctx)
    var w2 = _hada_make_weight(tape, adapter.w2_b, adapter.w2_a, ctx)
    # W = (W1 * W2) * (alpha/rank)   — Hadamard then scale  (:275)
    var wh = tape.record_mul(w1, w2, ctx)
    var sc = _const_like(wh, adapter.scale(), ctx)
    var w = tape.record_mul(wh, sc, ctx)             # [out,in]

    # op(x, W) = x @ Wᵀ  (F.linear, :276). Then + base.
    var lora = _tape_linear_nobias(tape, x, w, adapter.out_features, ctx)
    return tape.record_add(base, lora, ctx)


# untracked frozen base linear (no tape entry): x @ base_wᵀ.
def _tape_linear_nobias_untracked(
    x: Tensor, w: Tensor, out_features: Int, ctx: DeviceContext
) raises -> Tensor:
    var bsh = List[Int](); bsh.append(out_features)
    var zb = zeros_device(bsh^, STDtype.BF16, ctx)
    return linear(x, w, Optional[Tensor](zb), ctx)


# Build a fresh LoHa adapter with Serenity's init (LoRAModule.py:246,253-256),
# mapped factor-by-factor into this port's naming (w1_a/w2_a = down [rank,in];
# w1_b/w2_b = up [out,rank]):
#   w1_a ~ N(0, 1) ; w1_b ~ N(0, 0.1) ; w2_a ~ N(0, 1) ; w2_b = 0
# (w2_b=0 makes W2=0 at step 0 → LoHa is identity at init, like LoRA's B=0.)
def make_loha_adapter(
    in_features: Int, out_features: Int, rank: Int, alpha: Float32,
    seed: UInt64, ctx: DeviceContext,
) raises -> LoHaAdapter:
    var a_sh = List[Int](); a_sh.append(rank); a_sh.append(in_features)
    var b_sh = List[Int](); b_sh.append(out_features); b_sh.append(rank)
    # Serenity unpacks `hada_w1_b, hada_w1_a = create_layer()` (LoRAModule.py:246):
    #   hada_w1_b is the DOWN factor [rank,in], hada_w1_a is the UP factor [out,rank].
    # Then init (LoRAModule.py:253-254): nn.init.normal_(hada_w1_a, std=0.1) on the
    # UP [out,rank] factor; nn.init.normal_(hada_w1_b, std=1) on the DOWN [rank,in].
    # In this port `w1_a`=[rank,in] is the DOWN factor and `w1_b`=[out,rank] the UP,
    # so to match the SOURCE factor-by-factor the std assignments are: DOWN[rank,in]
    # gets std=1, UP[out,rank] gets std=0.1.
    # w1_a (down [rank,in]) ~ N(0,1)
    var w1a = randn(_clone_shape(a_sh), seed, STDtype.BF16, ctx)
    # w1_b (up [out,rank]) ~ N(0,0.1)
    var w1b = mul_scalar(randn(_clone_shape(b_sh), seed + 1, STDtype.BF16, ctx), Float32(0.1), ctx)
    # OT: nn.init.constant_(hada_w2_a, 0) on the UP [out,rank] factor;
    #     nn.init.normal_(hada_w2_b, std=1) on the DOWN [rank,in] factor (:255-256).
    # In this port's naming that is: w2_b(up [out,rank]) = 0, w2_a(down [rank,in]) ~ N(0,1).
    # Either way W2 = w2_b @ w2_a = 0 at init (one factor zero → LoHa identity at init).
    # w2_a (down [rank,in]) ~ N(0,1)
    var w2a = randn(_clone_shape(a_sh), seed + 2, STDtype.BF16, ctx)
    # w2_b (up [out,rank]) = 0  (identity at init)
    var w2b = zeros_device(_clone_shape(b_sh), STDtype.BF16, ctx)
    return LoHaAdapter(w1a^, w1b^, w2a^, w2b^, rank, in_features, out_features, alpha)


def _clone_shape(s: List[Int]) -> List[Int]:
    var o = List[Int]()
    for i in range(len(s)):
        o.append(s[i])
    return o^


# ─────────────────────────────────────────────────────────────────────────────
# LoKrAdapter — Kronecker-product low-rank adaptation (LyCORIS LoKr).
#
# NOT in Serenity's LoRAModule.py (see header note). Provided per this slice's
# spec ("Kronecker forward"). Construction follows the LyCORIS LoKr definition:
#   the full weight delta is a Kronecker product  W = w_a ⊗ w_b , optionally with
#   a low-rank factorization of the second factor (w_b = w_b1 @ w_b2). Here we
#   port the simplest faithful form used by LyCORIS for Linear layers:
#     W[out,in] = kron(w_a[p,q], w_b[out/p, in/q]) * (alpha/rank)
#   where p,q tile the output/input dims. The forward then matches the other PEFT
#   classes: orig_forward(x) + op(x, W)  (the LoRAModule.py:276 contract).
#
# Kronecker is built on the HOST (no tape-recorded kron op exists in the reused
# ops set); the resulting W is then fed through a tape-recorded linear so grads
# flow to W (and, by the caller's choice of which factor is trainable, to the
# factors). For the smoke/compile gate we treat w_a as the trainable factor and
# w_b as a fixed structured factor — the standard LoKr "only second factor low
# rank" simplification. Cite: LyCORIS lokr.py make_kron / weight_gen.
struct LoKrAdapter(Movable):
    var w_a: Tensor       # [p, q]            (trainable Kron factor 1)
    var w_b: Tensor       # [out//p, in//q]   (Kron factor 2)
    var p: Int
    var q: Int
    var in_features: Int
    var out_features: Int
    var rank: Int
    var alpha: Float32

    def __init__(
        out self, var w_a: Tensor, var w_b: Tensor, p: Int, q: Int,
        in_features: Int, out_features: Int, rank: Int, alpha: Float32,
    ):
        self.w_a = w_a^
        self.w_b = w_b^
        self.p = p
        self.q = q
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha

    def scale(self) -> Float32:
        return self.alpha / Float32(self.rank)

    def track(mut self, mut tape: Tape):
        tape.track(self.w_a)
        tape.track(self.w_b)


# Host Kronecker product: kron(A[ra,ca], B[rb,cb]) = C[ra*rb, ca*cb] with
#   C[i*rb + k, j*cb + l] = A[i,j] * B[k,l].
# Built on the host (BF16 in/out, F32 compute registers) since the reused ops
# expose no kron kernel. Returns an UNTRACKED [out,in] weight delta.
def _kron_host(a: Tensor, b: Tensor, ctx: DeviceContext) raises -> Tensor:
    var ash = a.shape(); var bsh = b.shape()
    var ra = ash[0]; var ca = ash[1]
    var rb = bsh[0]; var cb = bsh[1]
    var ah = a.to_host(ctx)       # BF16 → F32 host
    var bh = b.to_host(ctx)
    var rows = ra * rb
    var cols = ca * cb
    var out = List[Float32]()
    for _ in range(rows * cols):
        out.append(Float32(0.0))
    for i in range(ra):
        for j in range(ca):
            var av = ah[i * ca + j]
            for k in range(rb):
                for l in range(cb):
                    var r = i * rb + k
                    var c = j * cb + l
                    out[r * cols + c] = av * bh[k * cb + l]
    var osh = List[Int](); osh.append(rows); osh.append(cols)
    return Tensor.from_host(out^, osh^, STDtype.BF16, ctx)


# LoKr forward: W = kron(w_a, w_b) * (alpha/rank); return orig(x) + x @ Wᵀ.
# The kron is host-built (untracked); the trainable signal reaches w_a/w_b only
# if a tape-recorded kron is added (future work) — for this slice the gradient
# path is the tape-recorded linear on the constructed W, matching the LoRA
# `op(x, W)` contract (LoRAModule.py:276) at the W level.
def lokr_forward(
    mut tape: Tape, x: Tensor, base_w: Tensor, adapter: LoKrAdapter, ctx: DeviceContext
) raises -> Tensor:
    var base = _tape_linear_nobias_untracked(x, base_w, adapter.out_features, ctx)
    var wk = _kron_host(adapter.w_a, adapter.w_b, ctx)          # [out,in] untracked
    var sc = _const_like(wk, adapter.scale(), ctx)
    # track wk so the linear's grad reaches it (the trainable Kron weight).
    tape.track(wk)
    var w = tape.record_mul(wk, sc, ctx)
    var lora = _tape_linear_nobias(tape, x, w, adapter.out_features, ctx)
    return tape.record_add(base, lora, ctx)


# Build a fresh LoKr adapter. w_a small-random (trainable), w_b zero so the delta
# is zero at init (identity, like LoRA B=0 / LoHa w2_a=0).
def make_lokr_adapter(
    in_features: Int, out_features: Int, p: Int, q: Int, rank: Int, alpha: Float32,
    seed: UInt64, ctx: DeviceContext,
) raises -> LoKrAdapter:
    var a_sh = List[Int](); a_sh.append(p); a_sh.append(q)
    var b_sh = List[Int](); b_sh.append(out_features // p); b_sh.append(in_features // q)
    var wa = mul_scalar(randn(a_sh^, seed, STDtype.BF16, ctx), Float32(0.1), ctx)
    var wb = zeros_device(b_sh^, STDtype.BF16, ctx)   # zero → identity at init
    return LoKrAdapter(wa^, wb^, p, q, in_features, out_features, rank, alpha)


# ─────────────────────────────────────────────────────────────────────────────
# DoRAAdapter — weight-decomposed low-rank adaptation (LoRAModule.py:473-567).
#
# DoRA subclasses LoRA (down/up factors, same kaiming/zeros init) plus a
# per-output-channel magnitude `dora_scale`. The forward (LoRAModule.py:528-567):
#   A = lora_down.weight [rank,in] ; B = lora_up.weight [out,rank]
#   WP = orig_weight + make_weight(A,B)*(alpha/rank)   # [out,in]  (:539)
#        where make_weight(A,B) = A_as[out?]… → here B @ A = [out,in]
#   # default decompose_output_axis=False (kwarg default :487) → the `else` branch
#   # at :553-559, which transposes WP and norms over the OUTPUT axis:
#   #   WP.transpose(0,1).reshape(in,-1).norm(dim=1,keepdim=True).transpose(0,1)
#   #   ⇒ norm[j] = sqrt(Σ_o WP[o,j]^2), shape [1,in].
#   norm = output-axis L2 of WP, detached, reshaped to [1,in] broadcast  (:553-559)
#   WP = dora_scale * (WP / norm)                                  (:560)
#   return op(dropout(x), WP, bias)                                (:564-567)
# dora_scale init (LoRAModule.py:512-520, default !decompose_output_axis → :513-519):
#   dora_scale = norm(orig_weight.transpose(1,0).reshape(in,-1), dim=1).transpose
#   ⇒ the OUTPUT-axis norms of orig_weight, shape [1, in] broadcastable. norm is
#   treated as a CONSTANT for backprop (detached, :546-548 / paper section 4.3).
#
# DTYPE: Serenity computes WP/norm in float (get_unquantized_weight(...,float)).
# Here the norm is an F32 host reduction (a statistic), applied back as a BF16
# scale; WP arithmetic is tape-recorded BF16 with F32-accum inside the ops.
struct DoRAAdapter(Movable):
    var a: Tensor          # lora_down [rank, in]  (trained; kaiming-ish)
    var b: Tensor          # lora_up   [out, rank] (trained; zero-init)
    var dora_scale: Tensor # [1, in] magnitude (trained, nn.Parameter; default !decompose_output_axis)
    var rank: Int
    var in_features: Int
    var out_features: Int
    var alpha: Float32

    def __init__(
        out self, var a: Tensor, var b: Tensor, var dora_scale: Tensor,
        rank: Int, in_features: Int, out_features: Int, alpha: Float32,
    ):
        self.a = a^
        self.b = b^
        self.dora_scale = dora_scale^
        self.rank = rank
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha

    def scale(self) -> Float32:
        return self.alpha / Float32(self.rank)

    def track(mut self, mut tape: Tape):
        tape.track(self.a)
        tape.track(self.b)
        tape.track(self.dora_scale)


# DoRA `norm` of WP, as an UNTRACKED [1,in] BF16 tensor (detached for backprop —
# LoRAModule.py:546-559). The DEFAULT path is decompose_output_axis=False
# (kwarg default at LoRAModule.py:487), which is the `else` branch at :553-559:
#   WP.transpose(0,1).reshape(WP.shape[1],-1).norm(dim=1,keepdim=True).transpose(0,1)
# i.e. the norm is taken over the OUTPUT axis, per input column:
#   norm[j] = sqrt(Σ_o WP[o,j]^2),  shape [1,in].
# So reduce over dim=0 (the output axis), keepdim → [1,in] (F32 accum).
def _dora_row_norm(wp: Tensor, ctx: DeviceContext) raises -> Tensor:
    var sq = mul(wp, wp, ctx)                 # [out,in] BF16, F32-accum inside
    var dims = List[Int](); dims.append(0)
    var ss = reduce_sum_f32(sq, dims^, True, ctx)   # [1,in] F32 (keepdim)
    # sqrt → [1,in]; cast back to BF16 for the broadcast multiply.
    var nrm = sqrt_op(ss, ctx)
    return cast_to_bf16(nrm, ctx)


def cast_to_bf16(t: Tensor, ctx: DeviceContext) raises -> Tensor:
    if t.dtype() == STDtype.BF16:
        return t.clone(ctx)
    return cast_tensor(t, STDtype.BF16, ctx)


# DoRA forward (LoRAModule.py:528-567). `base_w` = orig_module.weight [out,in]
# (FROZEN, untracked). Returns op(x, WP) where WP = dora_scale*(WP / ||WP||).
# adapter.a/b/dora_scale must be tape.track()'d before this call.
#
# NOTE on norm-as-constant: Serenity detaches `norm` (LoRAModule.py:548) so it
# does NOT contribute gradient. Here `norm` is built from `wp` via UNTRACKED ops
# (mul/reduce/sqrt on the materialized values, not tape-recorded), so it is a
# constant w.r.t. the tape — matching the detach. The division WP/norm is then a
# tape-recorded multiply by reciprocal(norm) (constant) → grads to a/b/scale flow.
def dora_forward(
    mut tape: Tape, x: Tensor, base_w: Tensor, adapter: DoRAAdapter, ctx: DeviceContext
) raises -> Tensor:
    # make_weight(A,B) = B @ A = [out,in]  (LoRAModule.py:94, called :539).
    var ba = tape.record_matmul(adapter.b, adapter.a, ctx)     # [out,in]
    var sc = _const_like(ba, adapter.scale(), ctx)
    var delta = tape.record_mul(ba, sc, ctx)                   # *(alpha/rank)
    # WP = orig_weight + delta  (base_w untracked → no grad to base)  (:539)
    var wp = tape.record_add(base_w, delta, ctx)               # [out,in]

    # norm over the OUTPUT axis (default !decompose_output_axis, :553-559):
    #   norm[j] = sqrt(Σ_o WP[o,j]^2),  shape [1,in]. Detached/constant: built from
    # the CURRENT wp values via untracked ops → constant w.r.t. backprop, matching
    # WP.detach() (:548).
    var wp_host = wp.clone(ctx)                                # untracked snapshot
    var norm = _dora_row_norm(wp_host, ctx)                    # [1,in] BF16 const
    # reciprocal of norm (a [1,in] row) as a constant; WP * (1/norm), tiled to rows.
    var inv = rsqrt_recip(norm, ctx)                           # 1/norm, untracked [1,in]
    var normalized = tape.record_mul(wp, _broadcast_row(inv, adapter.out_features, ctx), ctx)
    # WP = dora_scale * normalized  (dora_scale [1,in] trainable, broadcast over rows)  (:560)
    #
    # FIDELITY DEVIATION (dora_scale grad): forward broadcast-mul DOES exist
    # (tensor_algebra.mul is NumPy-broadcasting), but MUL backward in autograd
    # (autograd.mojo OP_MUL: _raw_mul(gop,s0)/_raw_mul(gop,s1)) does NOT unbroadcast/
    # sum-to-shape, so a broadcast [1,in] operand's grad would return at [out,in]
    # (shape-mismatched/incorrect). To avoid that we host-tile dora_scale to [out,in]
    # before record_mul; host tiling rebuilds a FRESH untracked tensor → the grad to
    # the trainable `dora_scale` param is NOT recovered (only a/b receive grad).
    # Serenity trains dora_scale as an nn.Parameter (:505/:513). To reproduce
    # faithfully, autograd OP_MUL needs an unbroadcast/sum-to-shape on the broadcast
    # operand's grad (reduce over the broadcasted axes back to [1,in]); that is the
    # missing primitive and is flagged for the ops unit. As written, DoRA direction
    # (a/b) is exact; the magnitude (dora_scale) is applied forward-correctly but
    # held constant w.r.t. backprop. The adapter still tracks dora_scale (so AdamW
    # will weight-decay it) — only its data-gradient is currently dropped.
    var ds = _broadcast_row(adapter.dora_scale, adapter.out_features, ctx)
    var wp_final = tape.record_mul(normalized, ds, ctx)        # [out,in]

    # op(x, WP) = x @ WPᵀ  (F.linear, :564). (dropout(x) is identity at p=0.)
    return _tape_linear_nobias(tape, x, wp_final, adapter.out_features, ctx)


# 1/norm as an untracked BF16 tensor (norm is the detached DoRA magnitude).
def rsqrt_recip(norm: Tensor, ctx: DeviceContext) raises -> Tensor:
    # reciprocal via unary recip (div(ones, norm) would need a ones tensor).
    return reciprocal_op(norm, ctx)


# Broadcast a [1,in] row to [out,in] by host tiling (forward broadcast-mul DOES
# exist in tensor_algebra.mul, but MUL backward in autograd does not unbroadcast/
# sum-to-shape, so a broadcast operand's grad comes back at [out,in] not [1,in] —
# see FIDELITY DEVIATION note in dora_forward. Host-tiling sidesteps that for the
# constant factors). Untracked. The [1,in] row is repeated down all `out_features`
# rows so the result aligns elementwise with WP[out,in].
def _broadcast_row(row: Tensor, out_features: Int, ctx: DeviceContext) raises -> Tensor:
    var rsh = row.shape()
    var in_f = rsh[1]
    var rh = row.to_host(ctx)
    var vals = List[Float32]()
    for _o in range(out_features):
        for j in range(in_f):
            vals.append(rh[j])
    var sh = List[Int](); sh.append(out_features); sh.append(in_f)
    return Tensor.from_host(vals^, sh^, STDtype.BF16, ctx)


# Build a fresh DoRA adapter. a~kaiming-ish, b=0 (LoRA identity at init,
# LoRAModule.py:315-316 via super().initialize_weights at :492), dora_scale =
# the OUTPUT-axis norms of the base weight (default !decompose_output_axis branch,
# LoRAModule.py:512-520): norm[j]=sqrt(Σ_o W[o,j]^2), shape [1,in]. `base_w` is the
# frozen base [out,in]; _dora_row_norm reduces over dim=0 → [1,in], matching source.
def make_dora_adapter(
    base_w: Tensor, in_features: Int, out_features: Int, rank: Int, alpha: Float32,
    seed: UInt64, ctx: DeviceContext,
) raises -> DoRAAdapter:
    var a_sh = List[Int](); a_sh.append(rank); a_sh.append(in_features)
    var b_sh = List[Int](); b_sh.append(out_features); b_sh.append(rank)
    var std = Float32(1.0) / sqrt(Float32(3 * in_features))
    var a = mul_scalar(randn(a_sh^, seed, STDtype.BF16, ctx), std, ctx)
    var b = zeros_device(b_sh^, STDtype.BF16, ctx)
    # dora_scale = output-axis norms of base_w → [1,in] (default !decompose_output_axis).
    var ds = _dora_row_norm(base_w, ctx)
    return DoRAAdapter(a^, b^, ds^, rank, in_features, out_features, alpha)
