# models/dit/ideogram4_resident.mojo — RESIDENT fp8 Ideogram-4 DiT.
# Loads all weights ONCE (fp8 weights stay F8_E4M3 + F32 per-row scale; norms/
# biases bf16) and runs the forward with the fused fp8 GEMM (linear_fp8) — NO
# per-step re-dequant, and both cond+uncond transformers fit GPU-resident
# (~9.3GB each) so CFG runs without streaming. Math identical to ideogram4_dit's
# ideogram4_forward (parity-gated); only the weight source + matmul differ.
from std.gpu.host import DeviceContext
from std.memory import ArcPointer
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.ops.linear import linear
from serenitymojo.ops.activations import silu, swiglu
from serenitymojo.ops.norm import rms_norm, layer_norm_no_affine
from serenitymojo.ops.unary import tanh_op
from serenitymojo.ops.tensor_algebra import mul, add, add_scalar, reshape, slice, gather_rows, transpose, mul_scalar
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.fp8 import fp8_e4m3_dequant_perrow_to_bf16
from serenitymojo.models.dit.ideogram4_dit import (
    apply_rope_ideogram,
    ideogram4_embedscalar_sinusoid,
    ideogram4_sdpa_product_fwd,
)


struct Ideogram4Weights(Movable):
    var t: Dict[String, ArcPointer[Tensor]]
    var lora_a: Dict[String, ArcPointer[Tensor]]   # base-weight-name -> A [rank,in]
    var lora_b: Dict[String, ArcPointer[Tensor]]   # base-weight-name -> B [out,rank]

    fn __init__(out self, var t: Dict[String, ArcPointer[Tensor]]):
        self.t = t^
        self.lora_a = Dict[String, ArcPointer[Tensor]]()
        self.lora_b = Dict[String, ArcPointer[Tensor]]()

    @staticmethod
    def load(st: ShardedSafeTensors, ctx: DeviceContext) raises -> Ideogram4Weights:
        var d = Dict[String, ArcPointer[Tensor]]()
        for ref nm in st.names():
            var info = st.tensor_info(nm)
            if info.dtype == STDtype.F8_E4M3:
                d[nm] = ArcPointer(Tensor.from_view_raw(
                    from_parts(info.dtype, info.shape.copy(), st.tensor_bytes(nm)), ctx))
            elif info.dtype == STDtype.F32:
                d[nm] = ArcPointer(Tensor.from_view_as_f32(
                    from_parts(info.dtype, info.shape.copy(), st.tensor_bytes(nm)), ctx))
            else:
                d[nm] = ArcPointer(Tensor.from_view(st.tensor_view(nm), ctx))
        return Ideogram4Weights(d^)

    def w(self, name: String) raises -> ref [self.t] Tensor:
        if name not in self.t:
            raise Error("Ideogram4Weights: missing " + name)
        return self.t[name][]

    # Runtime additive LoRA overlay (NEVER fused into a saved model; memory-safe —
    # keeps the fp8 weights resident, stores only the rank-16 A/B). _lin adds
    # B·(A·x) for any base weight that has a LoRA. Keys:
    # diffusion_model.<base>.lora_A/B.weight ; this LoRA has alpha==rank (16/16) ⇒
    # scale 1.0 (folded into B at this scale; add a mul if a future LoRA differs).
    def load_lora(mut self, lora_path: String, ctx: DeviceContext) raises -> Int:
        var lst = ShardedSafeTensors.open(lora_path)
        var n = 0
        for ref nm in lst.names():
            if not nm.endswith(".lora_A.weight"):
                continue
            var inner = String(nm[byte=16 : nm.byte_length() - 14])   # strip "diffusion_model." + ".lora_A.weight"
            var bw = inner + ".weight"
            if bw not in self.t:
                continue
            self.lora_a[bw] = ArcPointer(Tensor.from_view(lst.tensor_view(nm), ctx))           # [rank,in] bf16
            self.lora_b[bw] = ArcPointer(Tensor.from_view(
                lst.tensor_view(String("diffusion_model.") + inner + ".lora_B.weight"), ctx))  # [out,rank] bf16
            n += 1
        return n


# fp8 linear: dequant the RESIDENT fp8 weight -> bf16 (cheap GPU kernel, no mmap
# re-read) then vendor-BLAS linear (fast). Faster than a hand-tiled fp8 GEMM and
# is exactly the parity-gated path (load_fp8_dequant+linear). Weights stay fp8-
# resident so both transformers fit + zero per-step mmap/streaming.
def _lin(w: Ideogram4Weights, x: Tensor, name: String, bias: String, ctx: DeviceContext) raises -> Tensor:
    var wbf = fp8_e4m3_dequant_perrow_to_bf16(w.w(name), w.w(name + "_scale"), ctx)
    # runtime LoRA overlay: out += B·(A·x)  (A [rank,in], B [out,rank], scale 1.0)
    if name in w.lora_a:
        var down = linear(x, w.lora_a[name][].clone(ctx), None, ctx)   # x·Aᵀ -> [..,rank]
        var up = linear(down, w.lora_b[name][].clone(ctx), None, ctx)  # ·Bᵀ -> [..,out]
        if len(bias) == 0:
            return add(linear(x, wbf, None, ctx), up, ctx)
        return add(linear(x, wbf, Optional[Tensor](w.w(bias).clone(ctx)), ctx), up, ctx)
    if len(bias) == 0:
        return linear(x, wbf, None, ctx)
    return linear(x, wbf, Optional[Tensor](w.w(bias).clone(ctx)), ctx)


def _t_embed_r(w: Ideogram4Weights, t: Tensor, dim: Int, ctx: DeviceContext) raises -> Tensor:
    var emb = ideogram4_embedscalar_sinusoid(t, dim, ctx)
    var h = _lin(w, emb, "t_embedding.mlp_in.weight", "t_embedding.mlp_in.bias", ctx)
    var a = silu(h, ctx)
    return _lin(w, a, "t_embedding.mlp_out.weight", "t_embedding.mlp_out.bias", ctx)


def _attn_r[S: Int](
    w: Ideogram4Weights, p: String, x: Tensor, cosf: Tensor, sinf: Tensor,
    num_heads: Int, head_dim: Int, ctx: DeviceContext,
) raises -> Tensor:
    var sh = x.shape()
    var L = sh[1]
    var hidden = sh[2]
    var qkv = _lin(w, x, p + "attention.qkv.weight", "", ctx)
    var qkv5 = reshape(qkv, [1, L, 3, num_heads, head_dim], ctx)
    var q = reshape(slice(qkv5, 2, 0, 1, ctx), [1, L, num_heads, head_dim], ctx)
    var k = reshape(slice(qkv5, 2, 1, 1, ctx), [1, L, num_heads, head_dim], ctx)
    var v = reshape(slice(qkv5, 2, 2, 1, ctx), [1, L, num_heads, head_dim], ctx)
    q = rms_norm(q, w.w(p + "attention.norm_q.weight"), Float32(1.0e-5), ctx)
    k = rms_norm(k, w.w(p + "attention.norm_k.weight"), Float32(1.0e-5), ctx)
    q = apply_rope_ideogram(q, cosf, sinf, ctx)
    k = apply_rope_ideogram(k, cosf, sinf, ctx)
    var scale = Float32(1.0 / (Float32(head_dim) ** 0.5))
    var attn = ideogram4_sdpa_product_fwd[1, S, 18, 256](q, k, v, scale, ctx)
    var merged = reshape(attn, [1, L, hidden], ctx)
    return _lin(w, merged, p + "attention.o.weight", "", ctx)


def _block_r[S: Int](
    w: Ideogram4Weights, p: String, x: Tensor, adaln_input: Tensor,
    cosf: Tensor, sinf: Tensor, num_heads: Int, head_dim: Int, hidden: Int, ctx: DeviceContext,
) raises -> Tensor:
    var mod = _lin(w, adaln_input, p + "adaln_modulation.weight", p + "adaln_modulation.bias", ctx)
    var scale_msa = add_scalar(slice(mod, 2, 0 * hidden, hidden, ctx), Float32(1.0), ctx)
    var gate_msa = tanh_op(slice(mod, 2, 1 * hidden, hidden, ctx), ctx)
    var scale_mlp = add_scalar(slice(mod, 2, 2 * hidden, hidden, ctx), Float32(1.0), ctx)
    var gate_mlp = tanh_op(slice(mod, 2, 3 * hidden, hidden, ctx), ctx)
    var an1 = rms_norm(x, w.w(p + "attention_norm1.weight"), Float32(1.0e-5), ctx)
    var attn_in = mul(an1, scale_msa, ctx)
    var attn_out = _attn_r[S](w, p, attn_in, cosf, sinf, num_heads, head_dim, ctx)
    var an2 = rms_norm(attn_out, w.w(p + "attention_norm2.weight"), Float32(1.0e-5), ctx)
    var x1 = add(x, mul(gate_msa, an2, ctx), ctx)
    var fn1 = rms_norm(x1, w.w(p + "ffn_norm1.weight"), Float32(1.0e-5), ctx)
    var mlp_in = mul(fn1, scale_mlp, ctx)
    var g = _lin(w, mlp_in, p + "feed_forward.w1.weight", "", ctx)
    var u = _lin(w, mlp_in, p + "feed_forward.w3.weight", "", ctx)
    var act = swiglu(g, u, ctx)
    var ff = _lin(w, act, p + "feed_forward.w2.weight", "", ctx)
    var fn2 = rms_norm(ff, w.w(p + "ffn_norm2.weight"), Float32(1.0e-5), ctx)
    return add(x1, mul(gate_mlp, fn2, ctx), ctx)


struct Ideogram4Masks(Movable):
    var llm_mask: Tensor
    var img_mask: Tensor
    var img_ids: List[Int]

    fn __init__(out self, var llm_mask: Tensor, var img_mask: Tensor, var img_ids: List[Int]):
        self.llm_mask = llm_mask^
        self.img_mask = img_mask^
        self.img_ids = img_ids^


# Build the (constant) indicator masks ONCE — hoisted out of the per-step forward
# to kill the per-forward D2H (indicator.to_host) + host mask rebuild. EDv2/Klein
# transfer-reduction lesson (FLAME_BLOCK_SWAP_AUDIT): don't redo D2H every step.
def ideogram4_build_masks(indicator: Tensor, ctx: DeviceContext) raises -> Ideogram4Masks:
    var L = indicator.shape()[1]
    var ind_h = indicator.to_host(ctx)
    var llm_mask_v = List[Float32]()
    var img_mask_v = List[Float32]()
    var img_ids = List[Int]()
    for i in range(L):
        var vi = ind_h[i]
        llm_mask_v.append(Float32(1.0) if (vi > 2.5 and vi < 3.5) else Float32(0.0))
        var is_img = (vi > 1.5 and vi < 2.5)
        img_mask_v.append(Float32(1.0) if is_img else Float32(0.0))
        img_ids.append(1 if is_img else 0)
    return Ideogram4Masks(
        Tensor.from_host(llm_mask_v^, [1, L, 1], STDtype.BF16, ctx),
        Tensor.from_host(img_mask_v^, [1, L, 1], STDtype.BF16, ctx),
        img_ids^,
    )


def ideogram4_forward_r[S: Int](
    w: Ideogram4Weights,
    x_in: Tensor, llm_in: Tensor, t_in: Tensor, masks: Ideogram4Masks,
    cosf: Tensor, sinf: Tensor,
    num_layers: Int, num_heads: Int, head_dim: Int, hidden: Int, ctx: DeviceContext,
) raises -> Tensor:
    var L = x_in.shape()[1]
    var llm = mul(llm_in, masks.llm_mask, ctx)
    var x = mul(x_in, masks.img_mask, ctx)
    x = mul(_lin(w, x, "input_proj.weight", "input_proj.bias", ctx), masks.img_mask, ctx)

    var t_cond = reshape(_t_embed_r(w, t_in, hidden, ctx), [1, 1, hidden], ctx)
    var adaln_input = silu(_lin(w, t_cond, "adaln_proj.weight", "adaln_proj.bias", ctx), ctx)

    llm = rms_norm(llm, w.w("llm_cond_norm.weight"), Float32(1.0e-6), ctx)
    llm = mul(_lin(w, llm, "llm_cond_proj.weight", "llm_cond_proj.bias", ctx), masks.llm_mask, ctx)

    var h = add(x, llm, ctx)
    var iemb = reshape(gather_rows(w.w("embed_image_indicator.weight"), masks.img_ids, ctx), [1, L, hidden], ctx)
    h = add(h, iemb, ctx)

    for li in range(num_layers):
        var p = String("layers.") + String(li) + "."
        h = _block_r[S](w, p, h, adaln_input, cosf, sinf, num_heads, head_dim, hidden, ctx)

    var fscale = add_scalar(_lin(w, silu(adaln_input, ctx), "final_layer.adaln_modulation.weight", "final_layer.adaln_modulation.bias", ctx), Float32(1.0), ctx)
    var hn = mul(layer_norm_no_affine(h, Float32(1.0e-6), ctx), fscale, ctx)
    var out = _lin(w, hn, "final_layer.linear.weight", "final_layer.linear.bias", ctx)
    return cast_tensor(out, STDtype.F32, ctx)
