# decoder2d.mojo — shared 2D-VAE-decoder kit + the Z-Image (ldm_decoder) config.
#
# Pure-Mojo, GPU-compute, inference-only. Builds on the Serenitymojo foundation:
#   * Tensor (serenitymojo/tensor.mojo) — on-GPU bytes + shape + dtype.
#   * ops.conv.conv2d        — NHWC input, RSCF [Kh,Kw,Cin,Cout] filter.
#   * ops.norm.group_norm    — NHWC input, [C] gamma/beta.
#   * ops.activations.silu   — elementwise.
#   * ops.linear.linear      — x @ wᵀ + b, weight [out,in] (PyTorch row-major).
#   * ops.attention.sdpa     — BSHD flash attention (comptime B/S/H/Dh).
#   * io.ShardedSafeTensors  — single-file VAE weight loader.
# Kit-local glue (foundation does not provide): vae_ops.{clone,add,reshape},
# upsample.upsample_nearest2x_nhwc.
#
# === Reference: inference-flame/src/vae/ldm_decoder.rs (read FULL, 802 L) ===
# Architecture (block_out_channels = (128, 256, 512, 512), layers_per_block=2 →
# num_resnets = layers_per_block + 1 = 3):
#   conv_in (latent_ch -> 512)
#   mid: ResBlock(512) + AttnBlock(512) + ResBlock(512)
#   up blocks: PROCESSED top-down. ldm_decoder.rs labels them up.3 -> up.0
#     (up.3 first, up.0 last); each up.{n>0} (ldm label) has an upsample.
#     Z-Image ships DIFFUSERS-format keys (decoder.up_blocks.0..3, native
#     processing order), so this kit works natively in diffusers order — process
#     up_blocks.0 first; NO LDM relabel/remap. Per up_block:
#       up_blocks.0: 512->512, 3 resnets, upsample.    (no conv_shortcut)
#       up_blocks.1: 512->512, 3 resnets, upsample.    (no conv_shortcut)
#       up_blocks.2: 512->256, 3 resnets, upsample.    (resnet0 conv_shortcut)
#       up_blocks.3: 256->128, 3 resnets, NO upsample. (resnet0 conv_shortcut)
#   conv_norm_out (GroupNorm 32) -> silu -> conv_out (128 -> 3)
#
# LAYOUT RULE (ldm_decoder.rs:37-76): the Rust path keeps activations in NCHW for
# cuDNN Conv2d, flipping to NHWC only for GroupNorm. OUR foundation conv2d AND
# group_norm are BOTH NHWC-native, so this kit stays NHWC end-to-end: NCHW->NHWC
# once at decode() entry, NHWC->NCHW once at exit. No per-op ping-pong. (The
# Rust comment's GPU-permute pain does not apply.)
#
# GroupNorm eps = 1e-6, num_groups = 32 (ldm_decoder.rs:224 etc).
#
# Conv2d shapes are COMPILE-TIME params in the foundation op, and VAE spatial
# size changes at each upsample. So the kit is COMPTIME-PARAMETERIZED on the
# latent (LH, LW): every intermediate H/W is comptime-derivable. The Z-Image
# config is exposed as `ZImageDecoder[LH, LW]`.

from std.math import sqrt
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.conv import conv2d
from serenitymojo.ops.norm import group_norm
from serenitymojo.ops.activations import silu
from serenitymojo.ops.linear import linear
from serenitymojo.ops.attention import sdpa
from serenitymojo.ops.tensor_algebra import permute
from serenitymojo.models.vae.vae_ops import clone, add, reshape
from serenitymojo.models.vae.upsample import upsample_nearest2x_nhwc


comptime GN_GROUPS = 32
comptime GN_EPS = Float32(1e-6)


# ── weight-loading helpers ────────────────────────────────────────────────────


def _load_weight(
    st: ShardedSafeTensors, name: String, ctx: DeviceContext
) raises -> Tensor:
    """Load a tensor by name from the (single-file) VAE shards as a GPU Tensor,
    preserving its stored shape/dtype (BF16). H2D copy via Tensor.from_view."""
    var tv = st.tensor_view(name)
    return Tensor.from_view(tv, ctx)


def _load_conv_weight_rscf(
    st: ShardedSafeTensors, name: String, ctx: DeviceContext
) raises -> Tensor:
    """Load a PyTorch conv weight (OIHW = [Cout,Cin,Kh,Kw]) and transpose it on
    the host to RSCF [Kh,Kw,Cin,Cout] (the foundation conv2d filter layout).
    Pure index remap on host F32 values; re-uploaded as BF16. Verified in
    conv_probe.mojo:
      OIHW idx = ((o*Cin + ci)*Kh + r)*Kw + s
      RSCF idx = ((r*Kw + s)*Cin + ci)*Cout + o
    """
    var w = _load_weight(st, name, ctx)
    var sh = w.shape()
    if len(sh) != 4:
        raise Error(String("conv weight ") + name + " is not rank-4 OIHW")
    var cout = sh[0]
    var cin = sh[1]
    var kh = sh[2]
    var kw = sh[3]
    var host = w.to_host(ctx)  # F32, OIHW order
    var rscf = List[Float32]()
    var total = kh * kw * cin * cout
    for _ in range(total):
        rscf.append(0.0)
    for o in range(cout):
        for ci in range(cin):
            for r in range(kh):
                for s in range(kw):
                    var src = ((o * cin + ci) * kh + r) * kw + s
                    var dst = ((r * kw + s) * cin + ci) * cout + o
                    rscf[dst] = host[src]
    var rshape = List[Int]()
    rshape.append(kh)
    rshape.append(kw)
    rshape.append(cin)
    rshape.append(cout)
    return Tensor.from_host(rscf, rshape^, w.dtype(), ctx)


# ── NCHW <-> NHWC GPU permute (entry/exit only) ───────────────────────────────


def nchw_to_nhwc(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    var sh = x.shape()
    if len(sh) != 4:
        raise Error("nchw_to_nhwc: need rank-4")
    var p = List[Int]()
    p.append(0)
    p.append(2)
    p.append(3)
    p.append(1)
    return permute(x, p^, ctx)


def nhwc_to_nchw(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    var sh = x.shape()
    if len(sh) != 4:
        raise Error("nhwc_to_nchw: need rank-4")
    var p = List[Int]()
    p.append(0)
    p.append(3)
    p.append(1)
    p.append(2)
    return permute(x, p^, ctx)


# ── ResnetBlock (ldm_decoder.rs ResBlock, NHWC) ───────────────────────────────
#
# forward (ldm_decoder.rs:223-237):
#   h = group_norm(x); h = silu(h); h = conv1(h)
#   h = group_norm(h); h = silu(h); h = conv2(h)
#   residual = shortcut(x) if in!=out else x;  out = residual + h
# conv1 Cin->Cout 3x3 pad1; conv2 Cout->Cout 3x3 pad1; shortcut 1x1 Cin->Cout.
# All preserve H/W. H/W/Cin/Cout are comptime so conv2d can be called.


@fieldwise_init
struct ResnetBlock[N: Int, H: Int, W: Int, Cin: Int, Cout: Int](Movable):
    var norm1_w: Tensor
    var norm1_b: Tensor
    var conv1_w: Tensor
    var conv1_b: Tensor
    var norm2_w: Tensor
    var norm2_b: Tensor
    var conv2_w: Tensor
    var conv2_b: Tensor
    var has_shortcut: Bool
    var sc_w: Tensor
    var sc_b: Tensor

    @staticmethod
    def load(
        st: ShardedSafeTensors, prefix: String, ctx: DeviceContext
    ) raises -> ResnetBlock[Self.N, Self.H, Self.W, Self.Cin, Self.Cout]:
        var n1w = _load_weight(st, prefix + ".norm1.weight", ctx)
        var n1b = _load_weight(st, prefix + ".norm1.bias", ctx)
        var c1w = _load_conv_weight_rscf(st, prefix + ".conv1.weight", ctx)
        var c1b = _load_weight(st, prefix + ".conv1.bias", ctx)
        var n2w = _load_weight(st, prefix + ".norm2.weight", ctx)
        var n2b = _load_weight(st, prefix + ".norm2.bias", ctx)
        var c2w = _load_conv_weight_rscf(st, prefix + ".conv2.weight", ctx)
        var c2b = _load_weight(st, prefix + ".conv2.bias", ctx)

        var has_sc = Self.Cin != Self.Cout
        var scw: Tensor
        var scb: Tensor
        if has_sc:
            scw = _load_conv_weight_rscf(
                st, prefix + ".conv_shortcut.weight", ctx
            )
            scb = _load_weight(st, prefix + ".conv_shortcut.bias", ctx)
        else:
            var d = List[Float32]()
            d.append(0.0)
            var ds = List[Int]()
            ds.append(1)
            scw = Tensor.from_host(d.copy(), ds.copy(), STDtype.BF16, ctx)
            scb = Tensor.from_host(d, ds^, STDtype.BF16, ctx)
        return ResnetBlock[Self.N, Self.H, Self.W, Self.Cin, Self.Cout](
            n1w^, n1b^, c1w^, c1b^, n2w^, n2b^, c2w^, c2b^, has_sc, scw^, scb^
        )

    def forward(self, x: Tensor, ctx: DeviceContext) raises -> Tensor:
        var h = group_norm(
            x, self.norm1_w, self.norm1_b, GN_GROUPS, GN_EPS, ctx
        )
        h = silu(h, ctx)
        h = conv2d[Self.N, Self.H, Self.W, Self.Cin, 3, 3, Self.Cout, 1, 1, 1, 1](
            h, clone(self.conv1_w, ctx),
            Optional[Tensor](clone(self.conv1_b, ctx)), ctx
        )
        h = group_norm(
            h, self.norm2_w, self.norm2_b, GN_GROUPS, GN_EPS, ctx
        )
        h = silu(h, ctx)
        h = conv2d[Self.N, Self.H, Self.W, Self.Cout, 3, 3, Self.Cout, 1, 1, 1, 1](
            h, clone(self.conv2_w, ctx),
            Optional[Tensor](clone(self.conv2_b, ctx)), ctx
        )

        var residual: Tensor
        if self.has_shortcut:
            residual = conv2d[Self.N, Self.H, Self.W, Self.Cin, 1, 1, Self.Cout, 1, 1, 0, 0](
                x, clone(self.sc_w, ctx),
                Optional[Tensor](clone(self.sc_b, ctx)), ctx
            )
        else:
            residual = clone(x, ctx)
        return add(residual, h, ctx)


# ── AttnBlock (diffusers VAE mid-block self-attention, single head, NHWC) ─────
#
# diffusers Attention (heads=1, residual_connection=True, group_norm 32 eps1e-6,
# scale=1/sqrt(C), to_q/k/v/to_out.0 are Linear [C,C]):
#   residual = x
#   h = group_norm(x)                         (NHWC -> NHWC)
#   flatten NHWC [N,H,W,C] -> tokens [N,HW,C]
#   q = to_q(h); k = to_k(h); v = to_v(h)     (Linear, weight [C,C])
#   sdpa over [N, HW, 1, C], scale=1/sqrt(C)  -> [N,HW,1,C]
#   out = to_out.0(out)                        (Linear [C,C])
#   reshape to NHWC [N,H,W,C]; return residual + out
#
# NOTE on QKV format (SKEPTIC-BAIT): ldm_decoder.rs squeezes Conv2d-1x1 QKV
# [C,C,1,1] -> [C,C] for a matmul. The Z-Image VAE on disk ships DIFFUSERS-format
# attention weights that are ALREADY Linear [C,C] (to_q/to_k/to_v/to_out.0) — no
# Conv2d-1x1, no squeeze. We use the foundation `linear` directly. Math is
# identical (1x1 conv over HW == per-token linear).


@fieldwise_init
struct AttnBlock[N: Int, H: Int, W: Int, C: Int](Movable):
    var norm_w: Tensor
    var norm_b: Tensor
    var q_w: Tensor
    var q_b: Tensor
    var k_w: Tensor
    var k_b: Tensor
    var v_w: Tensor
    var v_b: Tensor
    var o_w: Tensor
    var o_b: Tensor

    @staticmethod
    def load(
        st: ShardedSafeTensors, prefix: String, ctx: DeviceContext
    ) raises -> AttnBlock[Self.N, Self.H, Self.W, Self.C]:
        return AttnBlock[Self.N, Self.H, Self.W, Self.C](
            _load_weight(st, prefix + ".group_norm.weight", ctx),
            _load_weight(st, prefix + ".group_norm.bias", ctx),
            _load_weight(st, prefix + ".to_q.weight", ctx),
            _load_weight(st, prefix + ".to_q.bias", ctx),
            _load_weight(st, prefix + ".to_k.weight", ctx),
            _load_weight(st, prefix + ".to_k.bias", ctx),
            _load_weight(st, prefix + ".to_v.weight", ctx),
            _load_weight(st, prefix + ".to_v.bias", ctx),
            _load_weight(st, prefix + ".to_out.0.weight", ctx),
            _load_weight(st, prefix + ".to_out.0.bias", ctx),
        )

    def forward(self, x: Tensor, ctx: DeviceContext) raises -> Tensor:
        comptime S = Self.H * Self.W
        var residual = clone(x, ctx)
        var h = group_norm(x, self.norm_w, self.norm_b, GN_GROUPS, GN_EPS, ctx)
        # NHWC [Self.N,Self.H,Self.W,Self.C] is already token-contiguous; treat as [Self.N*S, Self.C] for
        # linear (linear flattens leading dims). Reshape to [Self.N, S, Self.C] view.
        var sh = List[Int]()
        sh.append(Self.N)
        sh.append(S)
        sh.append(Self.C)
        var h_tok = reshape(h, sh.copy(), ctx)  # [Self.N,S,Self.C]

        var q = linear(h_tok, clone(self.q_w, ctx),
                       Optional[Tensor](clone(self.q_b, ctx)), ctx)
        var k = linear(h_tok, clone(self.k_w, ctx),
                       Optional[Tensor](clone(self.k_b, ctx)), ctx)
        var v = linear(h_tok, clone(self.v_w, ctx),
                       Optional[Tensor](clone(self.v_b, ctx)), ctx)

        # [Self.N,S,Self.C] -> [Self.N,S,1,Self.C] (BSHD with Self.H=1 head, Dh=Self.C).
        var bshd = List[Int]()
        bshd.append(Self.N)
        bshd.append(S)
        bshd.append(1)
        bshd.append(Self.C)
        var qh = reshape(q, bshd.copy(), ctx)
        var kh = reshape(k, bshd.copy(), ctx)
        var vh = reshape(v, bshd.copy(), ctx)

        # zero mask [Self.N,1,S,S]. Keep this on device; at 1024 output the
        # VAE mid-attn mask is large enough that a host zero list is not viable.
        var mn = Self.N * 1 * S * S
        var mbuf = ctx.enqueue_create_buffer[DType.uint8](mn * x.dtype().byte_size())
        ctx.enqueue_memset[DType.uint8](mbuf, 0)
        ctx.synchronize()
        var ms = List[Int]()
        ms.append(Self.N)
        ms.append(1)
        ms.append(S)
        ms.append(S)
        var mask = Tensor(mbuf^, ms^, x.dtype())

        var scale = Float32(1.0) / sqrt(Float32(Self.C))
        var att = sdpa[Self.N, S, 1, Self.C](qh, kh, vh, mask, scale, ctx)  # [Self.N,S,1,Self.C]

        # back to [Self.N,S,Self.C] for the output projection
        var attsh = List[Int]()
        attsh.append(Self.N)
        attsh.append(S)
        attsh.append(Self.C)
        var att_tok = reshape(att, attsh^, ctx)
        var out = linear(att_tok, clone(self.o_w, ctx),
                         Optional[Tensor](clone(self.o_b, ctx)), ctx)

        # [Self.N,S,Self.C] -> NHWC [Self.N,Self.H,Self.W,Self.C], then residual add.
        var nhwc = List[Int]()
        nhwc.append(Self.N)
        nhwc.append(Self.H)
        nhwc.append(Self.W)
        nhwc.append(Self.C)
        var out_nhwc = reshape(out, nhwc^, ctx)
        return add(residual, out_nhwc, ctx)


# ── Upsample (nearest 2x + conv 3x3 pad1), NHWC ───────────────────────────────
#
# diffusers Upsample2D: interpolate(nearest, 2x) then conv(C->C, 3x3 pad1).
# H/W double; channels unchanged.


@fieldwise_init
struct Upsample[N: Int, H: Int, W: Int, C: Int](Movable):
    var conv_w: Tensor
    var conv_b: Tensor

    @staticmethod
    def load(
        st: ShardedSafeTensors, prefix: String, ctx: DeviceContext
    ) raises -> Upsample[Self.N, Self.H, Self.W, Self.C]:
        return Upsample[Self.N, Self.H, Self.W, Self.C](
            _load_conv_weight_rscf(st, prefix + ".conv.weight", ctx),
            _load_weight(st, prefix + ".conv.bias", ctx),
        )

    def forward(self, x: Tensor, ctx: DeviceContext) raises -> Tensor:
        var up = upsample_nearest2x_nhwc(x, ctx)  # [Self.N,2H,2W,Self.C]
        return conv2d[Self.N, Self.H * 2, Self.W * 2, Self.C, 3, 3, Self.C, 1, 1, 1, 1](
            up, clone(self.conv_w, ctx),
            Optional[Tensor](clone(self.conv_b, ctx)), ctx
        )
