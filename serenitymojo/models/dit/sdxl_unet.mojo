# models/dit/sdxl_unet.mojo — SDXL UNet (LDM-format), GPU, inference-only.
#
# Pure-Mojo port of inference-flame/src/models/sdxl_unet.rs (read FULL, 1296 L)
# and the end-to-end input contract in inference-flame/src/bin/sdxl_infer.rs.
#
# Architecture (SDXLConfig::default(), sdxl_unet.rs:185-202):
#   in/out_channels = 4, model_channels = 320, channel_mult = (1,2,4),
#   num_res_blocks = 2, context_dim = 2048, adm_in_channels = 2816,
#   head_dim = 64 (num_heads = channels / 64), use_linear_in_transformer = true,
#   transformer_depth_input  = [0,0,2,2,10,10]   (per input res-block)
#   transformer_depth_middle = 10
#   transformer_depth_output = [10,10,10,2,2,2,0,0,0]  (per output res-block)
#  (NOTE: the .rs file's doc-comment header line 8 says output [0,0,0,2,2,2,10,
#   10,10]; the actual Default impl on line 198 is [10,10,10,2,2,2,0,0,0]. The
#   Default impl WINS — it matches the task constants and the LDM checkpoint.)
#
# Forward (sdxl_unet.rs:851 forward):
#   emb = time_embed(t) + label_embed(y)              -> [1, 1280]
#   h = conv_in(x)                                     (input_blocks.0.0)
#   input_blocks 1..8: ResBlock(+SpatialTransformer)  | Downsample (3,6)
#     store every block output for skip connections
#   middle: ResBlock + SpatialTransformer + ResBlock
#   output_blocks 0..8: cat(h, skip) -> ResBlock(+ST)(+Upsample)
#   out: GroupNorm(32) -> SiLU -> conv_out             (out.0/out.2)
#
# Flat input_blocks (9): 0 conv_in; 1,2 Res(320); 3 Down; 4,5 Res(640)+ST(d2);
#   6 Down; 7,8 Res(1280)+ST(d10). Output_blocks (9): mirror with skip concat
#   (out.0/1 Res(2560->1280)+ST10; out.2 Res(1920->1280)+ST10+Up; out.3 Res
#   (1920->640)+ST2; out.4 Res(1280->640)+ST2; out.5 Res(960->640)+ST2+Up; out.6
#   Res(960->320); out.7,8 Res(640->320)).  (skip-channel arithmetic verified
#   against sdxl_unet.rs test_output_block_in_channels.)
#
# ResBlock (sdxl_unet.rs:593): in_layers GN32(eps1e-5)->SiLU->Conv3x3;
#   emb_layers SiLU->Linear(1280,out_ch) broadcast-added (NHWC: per-channel);
#   out_layers GN32->SiLU->Conv3x3; skip 1x1 conv if Cin!=Cout; residual add.
# SpatialTransformer (sdxl_unet.rs:761): GN32(eps1e-6)->proj_in(Linear)->
#   N x BasicTransformerBlock->proj_out(Linear); residual add. Linear proj
#   (use_linear_in_transformer=true): reshape NHWC->[1,HW,C] before proj_in.
# BasicTransformerBlock (sdxl_unet.rs:702): LN->self-attn(attn1)->res;
#   LN->cross-attn(attn2, K/V from context)->res; LN->GEGLU FF->res.
#   attn1/attn2 use NO bias on q/k/v; to_out.0 has bias.
# GEGLU (sdxl_unet.rs:638): proj(Linear+bias) to 2*ff -> split -> x*gelu(gate).
# Timestep embedding (sdxl_unet.rs:482): sinusoidal dim=320, COS-first SIN-second,
#   freq exp(-ln(1e4)*i/half) — IDENTICAL to ops/embeddings.timestep_embedding,
#   which is reused directly.
#
# LAYOUT: NHWC end-to-end (foundation conv2d/group_norm are NHWC-native). Latent
# arrives NCHW [1,4,LH,LW]; convert to NHWC once at entry, back to NCHW at exit.
# Cross-attention is rectangular (q-seq HW != kv-seq 77) and UNMASKED -> uses the
# SDXL-local sdxl_sdpa (models/dit/sdxl_attention.mojo), NOT the foundation
# square-mask sdpa. Self-attention routes through the same helper (Sq==Skv).
#
# Comptime-parameterized on the latent spatial size (LH, LW): SDXL 1024² ->
# LH=LW=128. conv2d/sdxl_sdpa need static shapes; every spatial size + seq len
# is comptime-derivable from (LH, LW).
#
# Mojo 1.0.0b1, NVIDIA GPU. BF16 storage (the extracted sdxl_unet_bf16 ckpt),
# F32 accumulation in ops.

from std.memory import ArcPointer
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu import global_idx
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.conv import conv2d
from serenitymojo.ops.norm import group_norm, layer_norm
from serenitymojo.ops.activations import silu, gelu
from serenitymojo.ops.linear import linear
from serenitymojo.ops.embeddings import timestep_embedding
from serenitymojo.ops.tensor_algebra import add, mul, concat, reshape, slice
from serenitymojo.models.dit.sdxl_attention import sdxl_sdpa
from serenitymojo.models.vae.decoder2d import nchw_to_nhwc, nhwc_to_nchw
from serenitymojo.models.vae.upsample import upsample_nearest2x_nhwc


comptime _DYN1 = Layout.row_major(-1)
comptime _BLOCK = 256
comptime MC = 320          # model_channels
comptime EMB_CH = 1280     # time/label embed dim
comptime CTX_DIM = 2048    # cross-attention context dim
comptime CTX_SEQ = 77      # text token count
comptime HEAD_DIM = 64
comptime GN_EPS_RES = Float32(1e-5)
comptime GN_EPS_ST = Float32(1e-6)


# ── conv-weight RSCF host loader (OIHW [Co,Ci,Kh,Kw] -> RSCF [Kh,Kw,Ci,Co]) ──
# Same remap as decoder2d._load_conv_weight_rscf; duplicated here so the UNet
# does not depend on a VAE-private helper. Pure host index remap, re-upload BF16.
def _to_rscf(w: Tensor, ctx: DeviceContext) raises -> Tensor:
    var sh = w.shape()
    if len(sh) != 4:
        raise Error("conv weight not rank-4 OIHW")
    var cout = sh[0]
    var cin = sh[1]
    var kh = sh[2]
    var kw = sh[3]
    var host = w.to_host(ctx)
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


# Per-channel broadcast add: o[.., c] = x[.., c] + b[c], over NHWC flat [rows, C].
# Used for the ResBlock time-embed injection (emb_out [1,Cout] added to NHWC h).
def _bcast_add_c_kernel_bf16(
    x: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    b: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    rows: Int, c: Int,
):
    var idx = Int(global_idx.x)
    var total = rows * c
    if idx < total:
        var ch = idx % c
        var xv = rebind[Scalar[DType.bfloat16]](x[idx]).cast[DType.float32]()
        var bv = rebind[Scalar[DType.bfloat16]](b[ch]).cast[DType.float32]()
        o[idx] = rebind[o.element_type]((xv + bv).cast[DType.bfloat16]())


def _bcast_add_c_kernel_f32(
    x: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    b: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    rows: Int, c: Int,
):
    var idx = Int(global_idx.x)
    var total = rows * c
    if idx < total:
        var ch = idx % c
        var xv = rebind[Scalar[DType.float32]](x[idx])
        var bv = rebind[Scalar[DType.float32]](b[ch])
        o[idx] = rebind[o.element_type](xv + bv)


def _bcast_add_channel(x: Tensor, b: Tensor, rows: Int, c: Int, ctx: DeviceContext) raises -> Tensor:
    """NHWC x [rows, c] (flattened) + per-channel b [c]."""
    var dt = x.dtype().to_mojo_dtype()
    var n = x.numel()
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
    var rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var brl = RuntimeLayout[_DYN1].row_major(IndexList[1](c))
    var grid = (n + _BLOCK - 1) // _BLOCK
    if dt == DType.float32:
        var X = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float32](), rl
        )
        var B = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            b.buf.unsafe_ptr().bitcast[Float32](), brl
        )
        var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), rl
        )
        ctx.enqueue_function[_bcast_add_c_kernel_f32, _bcast_add_c_kernel_f32](
            X, B, O, rows, c, grid_dim=grid, block_dim=_BLOCK
        )
    elif dt == DType.bfloat16:
        var X = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[BFloat16](), rl
        )
        var B = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            b.buf.unsafe_ptr().bitcast[BFloat16](), brl
        )
        var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), rl
        )
        ctx.enqueue_function[_bcast_add_c_kernel_bf16, _bcast_add_c_kernel_bf16](
            X, B, O, rows, c, grid_dim=grid, block_dim=_BLOCK
        )
    else:
        raise Error("_bcast_add_channel: only F32/BF16")
    ctx.synchronize()
    return Tensor(out_buf^, x.shape(), x.dtype())


# ── SDXLUNet ──────────────────────────────────────────────────────────────────
struct SDXLUNet[LH: Int, LW: Int](Movable):
    """All-resident SDXL UNet. Owns every weight (ArcPointer; Tensor is
    Movable-not-Copyable). Conv weights are converted to RSCF at load. comptime
    (LH, LW) = latent spatial size; SDXL 1024² -> [128,128].

    Movable so it can be held in `List[ArcPointer[SDXLUNet[LH,LW]]]` for resident
    caching in serve/sdxl_backend.mojo (matches QwenImageDitOffloaded(Movable));
    all fields are already movable so the synthesized __moveinit__ is valid."""

    var weights: List[ArcPointer[Tensor]]
    var name_to_idx: Dict[String, Int]

    def __init__(
        out self,
        var weights: List[ArcPointer[Tensor]],
        var name_to_idx: Dict[String, Int],
    ):
        self.weights = weights^
        self.name_to_idx = name_to_idx^

    @staticmethod
    def load(
        path_or_dir: String, ctx: DeviceContext
    ) raises -> SDXLUNet[Self.LH, Self.LW]:
        """Load all UNet tensors. Expects keys stripped of any
        'model.diffusion_model.' prefix (the extracted sdxl_unet_bf16.safetensors
        ships them stripped). Conv weights (rank-4) are converted OIHW->RSCF."""
        var st = ShardedSafeTensors.open(path_or_dir)
        var weights = List[ArcPointer[Tensor]]()
        var name_to_idx = Dict[String, Int]()
        for ref nm in st.names():
            var tv = st.tensor_view(nm)
            var t = Tensor.from_view(tv, ctx)
            # Conv weights are rank-4 and end in ".weight"; convert to RSCF.
            var sh = t.shape()
            if len(sh) == 4 and nm.endswith(".weight"):
                var rscf = _to_rscf(t, ctx)
                t = rscf^
            var idx = len(weights)
            weights.append(ArcPointer(t^))
            name_to_idx[nm] = idx
        return SDXLUNet[Self.LH, Self.LW](weights^, name_to_idx^)

    def _w(self, name: String) raises -> ref [self.weights] Tensor:
        if name not in self.name_to_idx:
            raise Error(String("missing UNet weight: ") + name)
        var idx = self.name_to_idx[name]
        return self.weights[idx][]

    def _has(self, name: String) -> Bool:
        return name in self.name_to_idx

    # ── linear helpers ──────────────────────────────────────────────────────
    def _lin_nb(self, x: Tensor, wkey: String, ctx: DeviceContext) raises -> Tensor:
        """x @ W^T, no bias."""
        ref w = self._w(wkey)
        return linear(x, w, None, ctx)

    def _lin_b(self, x: Tensor, wkey: String, bkey: String, ctx: DeviceContext) raises -> Tensor:
        """x @ W^T + bias."""
        ref w = self._w(wkey)
        ref b = self._w(bkey)
        return linear(x, w, Optional[Tensor](_clone(b, ctx)), ctx)

    # ── conv helper (NHWC; comptime spatial + channel + kernel + stride/pad) ──
    def _conv[
        H: Int, W: Int, Cin: Int, Kh: Int, Kw: Int, Cout: Int, Sh: Int, Sw: Int, Ph: Int, Pw: Int
    ](self, x: Tensor, prefix: String, ctx: DeviceContext) raises -> Tensor:
        ref w = self._w(prefix + ".weight")  # already RSCF
        var has_bias = self._has(prefix + ".bias")
        if has_bias:
            ref b = self._w(prefix + ".bias")
            return conv2d[1, H, W, Cin, Kh, Kw, Cout, Sh, Sw, Ph, Pw](
                x, _clone(w, ctx), Optional[Tensor](_clone(b, ctx)), ctx
            )
        return conv2d[1, H, W, Cin, Kh, Kw, Cout, Sh, Sw, Ph, Pw](
            x, _clone(w, ctx), None, ctx
        )

    # ── timestep + label embedding ──────────────────────────────────────────
    def _time_embed(self, t_val: Float32, ctx: DeviceContext) raises -> Tensor:
        # sinusoidal dim=320 (COS-first, matches LDM), then MLP.
        var th = List[Float32]()
        th.append(t_val)
        var tsh = List[Int]()
        tsh.append(1)
        var t = Tensor.from_host(th, tsh^, STDtype.F32, ctx)
        var dtype = self._w(String("time_embed.0.weight")).dtype()
        var emb = timestep_embedding(t, MC, ctx, Float32(10000.0), dtype)
        var h = self._lin_b(emb, String("time_embed.0.weight"), String("time_embed.0.bias"), ctx)
        h = silu(h, ctx)
        return self._lin_b(h, String("time_embed.2.weight"), String("time_embed.2.bias"), ctx)

    def _label_embed(self, y: Tensor, ctx: DeviceContext) raises -> Tensor:
        # y: [1, 2816] -> Linear(2816,1280) -> SiLU -> Linear(1280,1280).
        var h = self._lin_b(y, String("label_emb.0.0.weight"), String("label_emb.0.0.bias"), ctx)
        h = silu(h, ctx)
        return self._lin_b(h, String("label_emb.0.2.weight"), String("label_emb.0.2.bias"), ctx)

    # ── ResBlock (NHWC) ───────────────────────────────────────────────────────
    # prefix e.g. "input_blocks.1.0" or "middle_block.0".
    def _resblock[
        H: Int, W: Int, Cin: Int, Cout: Int
    ](self, x: Tensor, emb: Tensor, prefix: String, ctx: DeviceContext) raises -> Tensor:
        # in_layers: GN32 -> SiLU -> Conv3x3.
        ref gn1w = self._w(prefix + ".in_layers.0.weight")
        ref gn1b = self._w(prefix + ".in_layers.0.bias")
        var h = group_norm(x, gn1w, gn1b, 32, GN_EPS_RES, ctx)
        h = silu(h, ctx)
        h = self._conv[H, W, Cin, 3, 3, Cout, 1, 1, 1, 1](h, prefix + ".in_layers.2", ctx)

        # emb_layers: SiLU(emb) -> Linear(EMB_CH, Cout); broadcast over HW.
        var emb_h = silu(emb, ctx)
        var emb_out = self._lin_b(
            emb_h, prefix + ".emb_layers.1.weight", prefix + ".emb_layers.1.bias", ctx
        )  # [1, Cout]
        h = _bcast_add_channel(h, emb_out, H * W, Cout, ctx)

        # out_layers: GN32 -> SiLU -> Conv3x3 (index 3 due to Dropout at 2).
        ref gn2w = self._w(prefix + ".out_layers.0.weight")
        ref gn2b = self._w(prefix + ".out_layers.0.bias")
        h = group_norm(h, gn2w, gn2b, 32, GN_EPS_RES, ctx)
        h = silu(h, ctx)
        h = self._conv[H, W, Cout, 3, 3, Cout, 1, 1, 1, 1](h, prefix + ".out_layers.3", ctx)

        # skip connection: 1x1 conv if channels mismatch.
        var residual: Tensor
        if self._has(prefix + ".skip_connection.weight"):
            residual = self._conv[H, W, Cin, 1, 1, Cout, 1, 1, 0, 0](
                x, prefix + ".skip_connection", ctx
            )
        else:
            residual = _clone(x, ctx)
        return add(residual, h, ctx)

    # ── BasicTransformerBlock ───────────────────────────────────────────────
    # x: [1, S, C] tokens. self-attn (Sq=Skv=S), cross-attn (Sq=S, Skv=77).
    def _basic_block[
        S: Int, C: Int, Heads: Int
    ](self, x: Tensor, context: Tensor, prefix: String, ctx: DeviceContext) raises -> Tensor:
        # self-attn: norm1 -> attn1(Q/K/V from x, no bias) -> res.
        ref n1w = self._w(prefix + ".norm1.weight")
        ref n1b = self._w(prefix + ".norm1.bias")
        var xn1 = layer_norm(x, n1w, n1b, Float32(1e-5), ctx)
        var attn1 = self._attn[S, S, C, Heads](xn1, xn1, prefix + ".attn1", ctx)
        var x1 = add(x, attn1, ctx)

        # cross-attn: norm2 -> attn2(Q from x1, K/V from context) -> res.
        ref n2w = self._w(prefix + ".norm2.weight")
        ref n2b = self._w(prefix + ".norm2.bias")
        var xn2 = layer_norm(x1, n2w, n2b, Float32(1e-5), ctx)
        var attn2 = self._attn[S, CTX_SEQ, C, Heads](xn2, context, prefix + ".attn2", ctx)
        var x2 = add(x1, attn2, ctx)

        # FF: norm3 -> GEGLU -> Linear -> res.
        ref n3w = self._w(prefix + ".norm3.weight")
        ref n3b = self._w(prefix + ".norm3.bias")
        var xn3 = layer_norm(x2, n3w, n3b, Float32(1e-5), ctx)
        var ff = self._geglu[S, C](xn3, prefix + ".ff.net.0.proj", ctx)
        var ff_out = self._lin_b(
            ff, prefix + ".ff.net.2.weight", prefix + ".ff.net.2.bias", ctx
        )
        return add(x2, ff_out, ctx)

    # cross/self attention. q from x_q [1,Sq,C]; k/v from x_kv [1,Skv,*]. q/k/v
    # Linears are NO-bias; to_out.0 has bias. C = Heads*64.
    def _attn[
        Sq: Int, Skv: Int, C: Int, Heads: Int
    ](self, x_q: Tensor, x_kv: Tensor, prefix: String, ctx: DeviceContext) raises -> Tensor:
        var q = self._lin_nb(x_q, prefix + ".to_q.weight", ctx)   # [1,Sq,C]
        var k = self._lin_nb(x_kv, prefix + ".to_k.weight", ctx)  # [1,Skv,C]
        var v = self._lin_nb(x_kv, prefix + ".to_v.weight", ctx)  # [1,Skv,C]

        # reshape to BSHD.
        var qsh = List[Int]()
        qsh.append(1); qsh.append(Sq); qsh.append(Heads); qsh.append(HEAD_DIM)
        q = reshape(q, qsh^, ctx)
        var ksh = List[Int]()
        ksh.append(1); ksh.append(Skv); ksh.append(Heads); ksh.append(HEAD_DIM)
        k = reshape(k, ksh.copy(), ctx)
        v = reshape(v, ksh^, ctx)

        var scale = Float32(1.0) / Float32(8.0)
        var att = sdxl_sdpa[1, Sq, Skv, Heads, HEAD_DIM](q, k, v, scale, ctx)

        # [1,Sq,Heads,64] -> [1,Sq,C].
        var osh = List[Int]()
        osh.append(1); osh.append(Sq); osh.append(C)
        att = reshape(att, osh^, ctx)
        return self._lin_b(att, prefix + ".to_out.0.weight", prefix + ".to_out.0.bias", ctx)

    # GEGLU: proj(Linear+bias) to 2*ff -> split halves -> x * gelu(gate).
    def _geglu[
        S: Int, C: Int
    ](self, x: Tensor, prefix: String, ctx: DeviceContext) raises -> Tensor:
        var proj = self._lin_b(x, prefix + ".weight", prefix + ".bias", ctx)  # [1,S,2*ff]
        var last = proj.shape()[2]
        var half = last // 2
        # split on last dim: x_part = [..,0:half], gate = [..,half:2half].
        var xp = slice(proj, 2, 0, half, ctx)
        var gate = slice(proj, 2, half, half, ctx)
        var g = gelu(gate, ctx)
        return mul(xp, g, ctx)

    # ── SpatialTransformer (NHWC in/out) ────────────────────────────────────
    # x: NHWC [1,H,W,C]. GN32(eps1e-6) -> proj_in(Linear) -> N BasicBlocks ->
    # proj_out(Linear) -> residual add. (use_linear_in_transformer=true.)
    def _spatial_transformer[
        H: Int, W: Int, C: Int, Heads: Int, Depth: Int
    ](self, x: Tensor, context: Tensor, prefix: String, ctx: DeviceContext) raises -> Tensor:
        comptime S = H * W
        var residual = _clone(x, ctx)
        ref gnw = self._w(prefix + ".norm.weight")
        ref gnb = self._w(prefix + ".norm.bias")
        var xn = group_norm(x, gnw, gnb, 32, GN_EPS_ST, ctx)
        # NHWC [1,H,W,C] is already token-contiguous -> view as [1,S,C].
        var tsh = List[Int]()
        tsh.append(1); tsh.append(S); tsh.append(C)
        var tok = reshape(xn, tsh^, ctx)
        var hid = self._lin_b(tok, prefix + ".proj_in.weight", prefix + ".proj_in.bias", ctx)

        for j in range(Depth):
            hid = self._basic_block[S, C, Heads](
                hid, context, prefix + ".transformer_blocks." + String(j), ctx
            )

        var outp = self._lin_b(hid, prefix + ".proj_out.weight", prefix + ".proj_out.bias", ctx)
        # back to NHWC [1,H,W,C].
        var nhwc = List[Int]()
        nhwc.append(1); nhwc.append(H); nhwc.append(W); nhwc.append(C)
        outp = reshape(outp, nhwc^, ctx)
        return add(residual, outp, ctx)

    # ── Downsample / Upsample (NHWC) ─────────────────────────────────────────
    def _downsample[H: Int, W: Int, C: Int](self, x: Tensor, prefix: String, ctx: DeviceContext) raises -> Tensor:
        # Conv2d stride2 kernel3 pad1; key {prefix}.op.
        return self._conv[H, W, C, 3, 3, C, 2, 2, 1, 1](x, prefix + ".op", ctx)

    def _upsample[H: Int, W: Int, C: Int](self, x: Tensor, prefix: String, ctx: DeviceContext) raises -> Tensor:
        # nearest 2x -> conv3x3 pad1; key {prefix}.conv.
        var up = upsample_nearest2x_nhwc(x, ctx)  # [1,2H,2W,C]
        return self._conv[2 * H, 2 * W, C, 3, 3, C, 1, 1, 1, 1](up, prefix + ".conv", ctx)

    # ── full forward ──────────────────────────────────────────────────────────
    # x: latent NCHW [1,4,LH,LW]; t_val: scalar timestep; context: [1,77,2048];
    # y: [1,2816]. Returns eps NCHW [1,4,LH,LW].
    #
    # Spatial pyramid: L0 = LH×LW (320), L1 = LH/2 (640), L2 = LH/4 (1280).
    def forward(
        self, x_nchw: Tensor, t_val: Float32, context: Tensor, y: Tensor, ctx: DeviceContext
    ) raises -> Tensor:
        comptime H0 = Self.LH
        comptime W0 = Self.LW
        comptime H1 = Self.LH // 2
        comptime W1 = Self.LW // 2
        comptime H2 = Self.LH // 4
        comptime W2 = Self.LW // 4

        var emb_t = self._time_embed(t_val, ctx)
        var emb_y = self._label_embed(y, ctx)
        var emb = add(emb_t, emb_y, ctx)  # [1,1280]

        # NCHW -> NHWC once.
        var h = nchw_to_nhwc(x_nchw, ctx)  # [1,LH,LW,4]

        # --- input blocks ---
        var skips = List[ArcPointer[Tensor]]()

        # 0: conv_in (4 -> 320) @ H0.
        h = self._conv[H0, W0, 4, 3, 3, MC, 1, 1, 1, 1](h, String("input_blocks.0.0"), ctx)
        skips.append(ArcPointer(_clone(h, ctx)))

        # 1,2: Res(320->320) @ H0, no transformer.
        h = self._resblock[H0, W0, MC, MC](h, emb, String("input_blocks.1.0"), ctx)
        skips.append(ArcPointer(_clone(h, ctx)))
        h = self._resblock[H0, W0, MC, MC](h, emb, String("input_blocks.2.0"), ctx)
        skips.append(ArcPointer(_clone(h, ctx)))

        # 3: Downsample(320) @ H0 -> H1.
        h = self._downsample[H0, W0, MC](h, String("input_blocks.3.0"), ctx)
        skips.append(ArcPointer(_clone(h, ctx)))

        # 4,5: Res(320->640 / 640->640) + ST(depth2, 10 heads) @ H1.
        h = self._resblock[H1, W1, MC, 640](h, emb, String("input_blocks.4.0"), ctx)
        h = self._spatial_transformer[H1, W1, 640, 10, 2](h, context, String("input_blocks.4.1"), ctx)
        skips.append(ArcPointer(_clone(h, ctx)))
        h = self._resblock[H1, W1, 640, 640](h, emb, String("input_blocks.5.0"), ctx)
        h = self._spatial_transformer[H1, W1, 640, 10, 2](h, context, String("input_blocks.5.1"), ctx)
        skips.append(ArcPointer(_clone(h, ctx)))

        # 6: Downsample(640) @ H1 -> H2.
        h = self._downsample[H1, W1, 640](h, String("input_blocks.6.0"), ctx)
        skips.append(ArcPointer(_clone(h, ctx)))

        # 7,8: Res(640->1280 / 1280->1280) + ST(depth10, 20 heads) @ H2.
        h = self._resblock[H2, W2, 640, 1280](h, emb, String("input_blocks.7.0"), ctx)
        h = self._spatial_transformer[H2, W2, 1280, 20, 10](h, context, String("input_blocks.7.1"), ctx)
        skips.append(ArcPointer(_clone(h, ctx)))
        h = self._resblock[H2, W2, 1280, 1280](h, emb, String("input_blocks.8.0"), ctx)
        h = self._spatial_transformer[H2, W2, 1280, 20, 10](h, context, String("input_blocks.8.1"), ctx)
        skips.append(ArcPointer(_clone(h, ctx)))

        # --- middle block: Res + ST(depth10) + Res @ H2 (1280) ---
        h = self._resblock[H2, W2, 1280, 1280](h, emb, String("middle_block.0"), ctx)
        h = self._spatial_transformer[H2, W2, 1280, 20, 10](h, context, String("middle_block.1"), ctx)
        h = self._resblock[H2, W2, 1280, 1280](h, emb, String("middle_block.2"), ctx)

        # --- output blocks (pop skips, concat on channel dim = NHWC axis 3) ---
        # out0: cat(1280,1280)=2560 -> Res ->1280 + ST10 @ H2.
        h = concat(3, ctx, h, _pop(skips, ctx))
        h = self._resblock[H2, W2, 2560, 1280](h, emb, String("output_blocks.0.0"), ctx)
        h = self._spatial_transformer[H2, W2, 1280, 20, 10](h, context, String("output_blocks.0.1"), ctx)
        # out1: cat(1280,1280)=2560 -> Res ->1280 + ST10 @ H2.
        h = concat(3, ctx, h, _pop(skips, ctx))
        h = self._resblock[H2, W2, 2560, 1280](h, emb, String("output_blocks.1.0"), ctx)
        h = self._spatial_transformer[H2, W2, 1280, 20, 10](h, context, String("output_blocks.1.1"), ctx)
        # out2: cat(1280,640)=1920 -> Res ->1280 + ST10 + Upsample @ H2 -> H1.
        h = concat(3, ctx, h, _pop(skips, ctx))
        h = self._resblock[H2, W2, 1920, 1280](h, emb, String("output_blocks.2.0"), ctx)
        h = self._spatial_transformer[H2, W2, 1280, 20, 10](h, context, String("output_blocks.2.1"), ctx)
        h = self._upsample[H2, W2, 1280](h, String("output_blocks.2.2"), ctx)  # -> H1

        # out3: cat(1280,640)=1920 -> Res ->640 + ST2 @ H1.
        h = concat(3, ctx, h, _pop(skips, ctx))
        h = self._resblock[H1, W1, 1920, 640](h, emb, String("output_blocks.3.0"), ctx)
        h = self._spatial_transformer[H1, W1, 640, 10, 2](h, context, String("output_blocks.3.1"), ctx)
        # out4: cat(640,640)=1280 -> Res ->640 + ST2 @ H1.
        h = concat(3, ctx, h, _pop(skips, ctx))
        h = self._resblock[H1, W1, 1280, 640](h, emb, String("output_blocks.4.0"), ctx)
        h = self._spatial_transformer[H1, W1, 640, 10, 2](h, context, String("output_blocks.4.1"), ctx)
        # out5: cat(640,320)=960 -> Res ->640 + ST2 + Upsample @ H1 -> H0.
        h = concat(3, ctx, h, _pop(skips, ctx))
        h = self._resblock[H1, W1, 960, 640](h, emb, String("output_blocks.5.0"), ctx)
        h = self._spatial_transformer[H1, W1, 640, 10, 2](h, context, String("output_blocks.5.1"), ctx)
        h = self._upsample[H1, W1, 640](h, String("output_blocks.5.2"), ctx)  # -> H0

        # out6: cat(640,320)=960 -> Res ->320, no transformer @ H0.
        h = concat(3, ctx, h, _pop(skips, ctx))
        h = self._resblock[H0, W0, 960, MC](h, emb, String("output_blocks.6.0"), ctx)
        # out7: cat(320,320)=640 -> Res ->320 @ H0.
        h = concat(3, ctx, h, _pop(skips, ctx))
        h = self._resblock[H0, W0, 640, MC](h, emb, String("output_blocks.7.0"), ctx)
        # out8: cat(320,320)=640 -> Res ->320 @ H0.
        h = concat(3, ctx, h, _pop(skips, ctx))
        h = self._resblock[H0, W0, 640, MC](h, emb, String("output_blocks.8.0"), ctx)

        # --- final: GN32 -> SiLU -> conv_out (320 -> 4) @ H0 ---
        ref ow = self._w(String("out.0.weight"))
        ref ob = self._w(String("out.0.bias"))
        h = group_norm(h, ow, ob, 32, GN_EPS_RES, ctx)
        h = silu(h, ctx)
        h = self._conv[H0, W0, MC, 3, 3, 4, 1, 1, 1, 1](h, String("out.2"), ctx)

        # NHWC -> NCHW for the scheduler.
        return nhwc_to_nchw(h, ctx)


# ── module-level helpers ──────────────────────────────────────────────────────
def _clone(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    var dev = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
    ctx.enqueue_copy(dst_buf=dev, src_buf=x.buf)
    ctx.synchronize()
    return Tensor(dev^, x.shape(), x.dtype())


# Pop the last skip (LIFO) and return a fresh Tensor clone (the ArcPointer keeps
# the stored copy alive; we clone so the concat input is an owned Tensor).
def _pop(mut skips: List[ArcPointer[Tensor]], ctx: DeviceContext) raises -> Tensor:
    if len(skips) == 0:
        raise Error("output_blocks: ran out of skip connections")
    var top = skips[len(skips) - 1]
    var out = _clone(top[], ctx)
    _ = skips.pop()
    return out^


# GPU dtype cast for the timestep F32 embedding -> weight dtype. Tiny local
# helper (foundation ops/cast exists but we keep the UNet import surface small).
from std.gpu import global_idx as _gidx


def _cast_kernel_f32_to_bf16(
    x: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    n: Int,
):
    var i = Int(_gidx.x)
    if i < n:
        var v = rebind[Scalar[DType.float32]](x[i])
        o[i] = rebind[o.element_type](v.cast[DType.bfloat16]())


def _cast_to(x: Tensor, dtype: STDtype, ctx: DeviceContext) raises -> Tensor:
    if x.dtype() == dtype:
        return _clone(x, ctx)
    if x.dtype() != STDtype.F32 or dtype != STDtype.BF16:
        raise Error("_cast_to: only F32->BF16 supported in UNet")
    var n = x.numel()
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](n * 2)
    var rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var X = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        x.buf.unsafe_ptr().bitcast[Float32](), rl
    )
    var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        out_buf.unsafe_ptr().bitcast[BFloat16](), rl
    )
    var grid = (n + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[_cast_kernel_f32_to_bf16, _cast_kernel_f32_to_bf16](
        X, O, n, grid_dim=grid, block_dim=_BLOCK
    )
    ctx.synchronize()
    return Tensor(out_buf^, x.shape(), dtype)
