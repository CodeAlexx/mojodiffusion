# models/dit/ideogram4_resident.mojo — RESIDENT fp8 Ideogram-4 DiT.
# Loads all weights ONCE (fp8 weights stay F8_E4M3 + F32 per-row scale; norms/
# biases bf16) and runs the forward with the fused fp8 GEMM (linear_fp8) — NO
# per-step re-dequant, and both cond+uncond transformers fit GPU-resident
# (~9.3GB each) so CFG runs without streaming. Math identical to ideogram4_dit's
# ideogram4_forward (parity-gated); only the weight source + matmul differ.
from max.gpu.host import DeviceContext, HostBuffer, DeviceBuffer
from std.memory import ArcPointer
from serenitymojo.io.ffi import sys_memcpy, BytePtr
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
from serenitymojo.ops.attention import sdpa_qwen_flash_padmask
from serenitymojo.models.dit.ideogram4_dit import (
    apply_rope_ideogram,
    ideogram4_embedscalar_sinusoid,
    ideogram4_sdpa_product_fwd,
)


struct Ideogram4Weights(Movable):
    var t: Dict[String, ArcPointer[Tensor]]
    var lora_a: Dict[String, ArcPointer[Tensor]]   # base-weight-name -> A [rank,in]
    var lora_b: Dict[String, ArcPointer[Tensor]]   # base-weight-name -> B [out,rank]
    var lora_scale: Dict[String, Float32]

    def __init__(out self, var t: Dict[String, ArcPointer[Tensor]]):
        self.t = t^
        self.lora_a = Dict[String, ArcPointer[Tensor]]()
        self.lora_b = Dict[String, ArcPointer[Tensor]]()
        self.lora_scale = Dict[String, Float32]()

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

    def w(self, name: String) raises -> ref [self.t[String("")]] Tensor:
        if name not in self.t:
            raise Error("Ideogram4Weights: missing " + name)
        return self.t[name][]

    # Runtime additive LoRA overlay (NEVER fused into a saved model; memory-safe —
    # keeps the fp8 weights resident, stores only the rank-16 A/B). _lin adds
    # B·(A·x) for any base weight that has a LoRA. Accept canonical PEFT
    # lora_A/B and Serenity lora_down/up spellings, with file alpha/rank
    # composed with the UI multiplier.
    def load_lora(
        mut self,
        lora_path: String,
        ctx: DeviceContext,
        multiplier: Float32 = Float32(1.0),
    ) raises -> Int:
        var lst = ShardedSafeTensors.open(lora_path)
        var n = 0
        for ref nm in lst.names():
            var suffix_a = String("")
            var suffix_b = String("")
            if nm.endswith(".lora_A.weight"):
                suffix_a = String(".lora_A.weight")
                suffix_b = String(".lora_B.weight")
            elif nm.endswith(".lora_down.weight"):
                suffix_a = String(".lora_down.weight")
                suffix_b = String(".lora_up.weight")
            else:
                continue
            var prefix = String(
                nm[byte=0 : nm.byte_length() - suffix_a.byte_length()]
            )
            var inner = prefix.copy()
            if inner.startswith("diffusion_model."):
                var _tmp_inner = String(inner[byte=16 :])
                inner = _tmp_inner^
            elif inner.startswith("transformer."):
                var _tmp_inner = String(inner[byte=12 :])
                inner = _tmp_inner^
            var bw = inner + ".weight"
            if bw not in self.t:
                continue
            var b_key = prefix + suffix_b
            if b_key not in lst.names():
                raise Error(String("Ideogram4 LoRA missing paired tensor ") + b_key)
            var a = Tensor.from_view(lst.tensor_view(nm), ctx)
            var b = Tensor.from_view(lst.tensor_view(b_key), ctx)
            var ash = a.shape()
            var bsh = b.shape()
            var wsh = self.w(bw).shape()
            if (
                len(ash) != 2
                or len(bsh) != 2
                or len(wsh) != 2
                or ash[1] != wsh[1]
                or bsh[0] != wsh[0]
                or bsh[1] != ash[0]
            ):
                raise Error(String("Ideogram4 LoRA shape mismatch for ") + prefix)
            var scale = multiplier
            var alpha_key = prefix + String(".alpha")
            if alpha_key in lst.names():
                var alpha = Tensor.from_view_as_f32(
                    lst.tensor_view(alpha_key), ctx
                ).to_host(ctx)
                if len(alpha) != 1:
                    raise Error(
                        String("Ideogram4 LoRA alpha must be scalar for ") + prefix
                    )
                scale *= alpha[0] / Float32(ash[0])
            self.lora_a[bw] = ArcPointer(a^)
            self.lora_b[bw] = ArcPointer(b^)
            self.lora_scale[bw] = scale
            n += 1
        if n == 0:
            raise Error(
                "Ideogram4 LoRA contained no compatible transformer projection pairs"
            )
        return n


# fp8 linear: dequant the RESIDENT fp8 weight -> bf16 (cheap GPU kernel, no mmap
# re-read) then vendor-BLAS linear (fast). Faster than a hand-tiled fp8 GEMM and
# is exactly the parity-gated path (load_fp8_dequant+linear). Weights stay fp8-
# resident so both transformers fit + zero per-step mmap/streaming.
def _lin(w: Ideogram4Weights, x: Tensor, name: String, bias: String, ctx: DeviceContext) raises -> Tensor:
    var wbf = fp8_e4m3_dequant_perrow_to_bf16(w.w(name), w.w(name + "_scale"), ctx)
    # runtime LoRA overlay: out += scale·B·(A·x)
    if name in w.lora_a:
        var down = linear(x, w.lora_a[name][].clone(ctx), None, ctx)   # x·Aᵀ -> [..,rank]
        var up = linear(down, w.lora_b[name][].clone(ctx), None, ctx)  # ·Bᵀ -> [..,out]
        up = mul_scalar(up, w.lora_scale[name], ctx)
        if bias.byte_length() == 0:
            return add(linear(x, wbf, None, ctx), up, ctx)
        return add(linear(x, wbf, Optional[Tensor](w.w(bias).clone(ctx)), ctx), up, ctx)
    if bias.byte_length() == 0:
        return linear(x, wbf, None, ctx)
    return linear(x, wbf, Optional[Tensor](w.w(bias).clone(ctx)), ctx)


def _t_embed_r(w: Ideogram4Weights, t: Tensor, dim: Int, ctx: DeviceContext) raises -> Tensor:
    var emb = ideogram4_embedscalar_sinusoid(t, dim, ctx)
    var h = _lin(w, emb, "t_embedding.mlp_in.weight", "t_embedding.mlp_in.bias", ctx)
    var a = silu(h, ctx)
    return _lin(w, a, "t_embedding.mlp_out.weight", "t_embedding.mlp_out.bias", ctx)


def _attn_r_masked[S: Int, N_TXT: Int](
    w: Ideogram4Weights, p: String, x: Tensor, cosf: Tensor, sinf: Tensor,
    real_txt_len: Int, num_heads: Int, head_dim: Int, ctx: DeviceContext,
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
    var scale = Float32(1.0 / (Float32(head_dim) ** Float32(0.5)))
    var attn: Tensor
    comptime if N_TXT > 0:
        attn = sdpa_qwen_flash_padmask[1, S, 18, 256, N_TXT](
            q, k, v, real_txt_len, scale, ctx
        )
    else:
        attn = ideogram4_sdpa_product_fwd[1, S, 18, 256](q, k, v, scale, ctx)
    var merged = reshape(attn, [1, L, hidden], ctx)
    return _lin(w, merged, p + "attention.o.weight", "", ctx)


def _attn_r[S: Int](
    w: Ideogram4Weights, p: String, x: Tensor, cosf: Tensor, sinf: Tensor,
    num_heads: Int, head_dim: Int, ctx: DeviceContext,
) raises -> Tensor:
    return _attn_r_masked[S, 0](
        w, p, x, cosf, sinf, 0, num_heads, head_dim, ctx
    )


def _block_r_masked[S: Int, N_TXT: Int](
    w: Ideogram4Weights, p: String, x: Tensor, adaln_input: Tensor,
    cosf: Tensor, sinf: Tensor, real_txt_len: Int,
    num_heads: Int, head_dim: Int, hidden: Int, ctx: DeviceContext,
) raises -> Tensor:
    var mod = _lin(w, adaln_input, p + "adaln_modulation.weight", p + "adaln_modulation.bias", ctx)
    var scale_msa = add_scalar(slice(mod, 2, 0 * hidden, hidden, ctx), Float32(1.0), ctx)
    var gate_msa = tanh_op(slice(mod, 2, 1 * hidden, hidden, ctx), ctx)
    var scale_mlp = add_scalar(slice(mod, 2, 2 * hidden, hidden, ctx), Float32(1.0), ctx)
    var gate_mlp = tanh_op(slice(mod, 2, 3 * hidden, hidden, ctx), ctx)
    var an1 = rms_norm(x, w.w(p + "attention_norm1.weight"), Float32(1.0e-5), ctx)
    var attn_in = mul(an1, scale_msa, ctx)
    var attn_out = _attn_r_masked[S, N_TXT](
        w, p, attn_in, cosf, sinf, real_txt_len, num_heads, head_dim, ctx
    )
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


def _block_r[S: Int](
    w: Ideogram4Weights, p: String, x: Tensor, adaln_input: Tensor,
    cosf: Tensor, sinf: Tensor, num_heads: Int, head_dim: Int, hidden: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    return _block_r_masked[S, 0](
        w, p, x, adaln_input, cosf, sinf, 0,
        num_heads, head_dim, hidden, ctx
    )


struct Ideogram4Masks(Movable):
    var llm_mask: Tensor
    var img_mask: Tensor
    var img_ids: List[Int]

    def __init__(out self, var llm_mask: Tensor, var img_mask: Tensor, var img_ids: List[Int]):
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


def ideogram4_forward_r_masked[S: Int, N_TXT: Int](
    w: Ideogram4Weights,
    x_in: Tensor, llm_in: Tensor, t_in: Tensor, masks: Ideogram4Masks,
    cosf: Tensor, sinf: Tensor,
    real_txt_len: Int,
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
        h = _block_r_masked[S, N_TXT](
            w, p, h, adaln_input, cosf, sinf, real_txt_len,
            num_heads, head_dim, hidden, ctx
        )

    var fscale = add_scalar(_lin(w, silu(adaln_input, ctx), "final_layer.adaln_modulation.weight", "final_layer.adaln_modulation.bias", ctx), Float32(1.0), ctx)
    var hn = mul(layer_norm_no_affine(h, Float32(1.0e-6), ctx), fscale, ctx)
    var out = _lin(w, hn, "final_layer.linear.weight", "final_layer.linear.bias", ctx)
    return cast_tensor(out, STDtype.F32, ctx)


def ideogram4_forward_r[S: Int](
    w: Ideogram4Weights,
    x_in: Tensor, llm_in: Tensor, t_in: Tensor, masks: Ideogram4Masks,
    cosf: Tensor, sinf: Tensor,
    num_layers: Int, num_heads: Int, head_dim: Int, hidden: Int, ctx: DeviceContext,
) raises -> Tensor:
    return ideogram4_forward_r_masked[S, 0](
        w, x_in, llm_in, t_in, masks, cosf, sinf, 0,
        num_layers, num_heads, head_dim, hidden, ctx
    )


struct Ideogram4UncondStream(Movable):
    """TRUE dual-trunk uncond on cards that cannot hold BOTH fp8 trunks
    resident (MJ-1142: 2x9.3GB + 1024^2 activations OOMs the 24GB 3090 Ti).

    The uncond trunk's per-layer tensors live VERBATIM in one pinned host
    slab; non-layer tensors (embedders, projections, final layer) stay
    device-resident in `active`. During the uncond pass each layer is staged
    into `active` right before its block runs and the previous layer's
    device buffers are Arc-dropped — ~one layer (~275MB) resident instead of
    9.3GB. Copies and kernels share the context stream, so no fences are
    needed and the math/bytes are identical to the resident trunk (same
    tensors, same kernels, same order)."""

    var active: Ideogram4Weights
    var host: HostBuffer[DType.uint8]
    var slab: DeviceBuffer[DType.uint8]
    var e_name: List[String]
    var e_layer: List[Int]
    var e_off: List[Int]
    var e_rel: List[Int]
    var e_nbytes: List[Int]
    var e_shape: List[ArcPointer[List[Int]]]
    var e_dtype: List[STDtype]
    var staged: List[String]
    var staged_layer: Int

    def __init__(
        out self,
        var active: Ideogram4Weights,
        var host: HostBuffer[DType.uint8],
        var slab: DeviceBuffer[DType.uint8],
        var e_name: List[String],
        var e_layer: List[Int],
        var e_off: List[Int],
        var e_rel: List[Int],
        var e_nbytes: List[Int],
        var e_shape: List[ArcPointer[List[Int]]],
        var e_dtype: List[STDtype],
    ):
        self.active = active^
        self.host = host^
        self.slab = slab^
        self.e_name = e_name^
        self.e_layer = e_layer^
        self.e_off = e_off^
        self.e_rel = e_rel^
        self.e_nbytes = e_nbytes^
        self.e_shape = e_shape^
        self.e_dtype = e_dtype^
        self.staged = List[String]()
        self.staged_layer = -1

    @staticmethod
    def load_stream(
        st: ShardedSafeTensors, ctx: DeviceContext
    ) raises -> Ideogram4UncondStream:
        """Split the checkpoint: non-layer tensors -> device (exactly like
        Ideogram4Weights.load); `layers.<i>.*` tensors -> verbatim bytes in a
        pinned host slab with an entry table. F8/F32/BF16 all copy verbatim
        (dtype conversions in the resident load are identity for this
        checkpoint), so staged device tensors are byte-identical to resident
        ones."""
        var d = Dict[String, ArcPointer[Tensor]]()
        var names = List[String]()
        var layers = List[Int]()
        var offs = List[Int]()
        var rels = List[Int]()
        var nbs = List[Int]()
        var shapes = List[ArcPointer[List[Int]]]()
        var dts = List[STDtype]()
        var total = 0
        var layer_cursor = Dict[Int, Int]()
        var max_layer_bytes = 0
        var lp = String("layers.")
        for ref nm in st.names():
            if nm.find(lp) != 0:
                continue
            var info = st.tensor_info(nm)
            names.append(nm.copy())
            var nb = info.shape.copy()
            var numel = 1
            for i in range(len(nb)):
                numel *= nb[i]
            var b = numel * info.dtype.byte_size()
            var ci = lp.byte_length()
            var raw = nm.as_bytes()
            var li = 0
            while ci < len(raw) and Int(raw[ci]) >= 48 and Int(raw[ci]) <= 57:
                li = li * 10 + (Int(raw[ci]) - 48)
                ci += 1
            layers.append(li)
            offs.append(total)
            var cur = 0
            if li in layer_cursor:
                cur = layer_cursor[li]
            # 256-byte alignment inside the slab keeps every staged tensor
            # naturally aligned for the fp8 GEMM loads.
            cur = (cur + 255) & ~255
            rels.append(cur)
            layer_cursor[li] = cur + b
            if cur + b > max_layer_bytes:
                max_layer_bytes = cur + b
            nbs.append(b)
            shapes.append(ArcPointer(info.shape.copy()))
            dts.append(info.dtype)
            total += b
        var host = ctx.enqueue_create_host_buffer[DType.uint8](total)
        ctx.synchronize()
        for i in range(len(names)):
            var span = st.tensor_bytes(names[i])
            var dst = BytePtr(
                unsafe_from_address=Int(host.unsafe_ptr()) + offs[i]
            )
            var src = BytePtr(unsafe_from_address=Int(span.unsafe_ptr()))
            _ = sys_memcpy(dst, src, nbs[i])
        st.release_to_os()
        for ref nm in st.names():
            if nm.find(lp) == 0:
                continue
            var info = st.tensor_info(nm)
            if info.dtype == STDtype.F8_E4M3:
                d[nm] = ArcPointer(Tensor.from_view_raw(
                    from_parts(info.dtype, info.shape.copy(), st.tensor_bytes(nm)), ctx))
            elif info.dtype == STDtype.F32:
                d[nm] = ArcPointer(Tensor.from_view_as_f32(
                    from_parts(info.dtype, info.shape.copy(), st.tensor_bytes(nm)), ctx))
            else:
                d[nm] = ArcPointer(Tensor.from_view(st.tensor_view(nm), ctx))
        # ONE fixed staging slab, reused for every layer: per-layer
        # create/free churn makes the caching arena hoard the whole card and
        # starves direct driver allocs (the 45MB cuBLAS workspace OOM).
        var slab = ctx.enqueue_create_buffer[DType.uint8](max_layer_bytes)
        ctx.synchronize()
        return Ideogram4UncondStream(
            Ideogram4Weights(d^), host^, slab^, names^, layers^, offs^,
            rels^, nbs^, shapes^, dts^,
        )

    def stage_layer(mut self, li: Int, ctx: DeviceContext) raises:
        """Drop the previously staged layer's device tensors and upload layer
        `li` from the pinned slab. Stream-ordered with subsequent kernels."""
        if self.staged_layer == li:
            return
        for ref nm in self.staged:
            if nm in self.active.t:
                _ = self.active.t.pop(nm)
        self.staged = List[String]()
        for i in range(len(self.e_name)):
            if self.e_layer[i] != li:
                continue
            var dsub = self.slab.create_sub_buffer[DType.uint8](
                self.e_rel[i], self.e_nbytes[i]
            )
            var hv = self.host.create_sub_buffer[DType.uint8](
                self.e_off[i], self.e_nbytes[i]
            )
            ctx.enqueue_copy(dst_buf=dsub, src_buf=hv)
            var view = self.slab.create_sub_buffer[DType.uint8](
                self.e_rel[i], self.e_nbytes[i]
            )
            self.active.t[self.e_name[i]] = ArcPointer(
                Tensor(view^, self.e_shape[i][].copy(), self.e_dtype[i])
            )
            self.staged.append(self.e_name[i].copy())
        self.staged_layer = li

    def forward_streamed[S: Int](
        mut self,
        x_in: Tensor, llm_in: Tensor, t_in: Tensor, masks: Ideogram4Masks,
        cosf: Tensor, sinf: Tensor,
        num_layers: Int, num_heads: Int, head_dim: Int, hidden: Int,
        ctx: DeviceContext,
    ) raises -> Tensor:
        """Byte-identical twin of `ideogram4_forward_r` (the N_TXT=0 uncond
        path) that stages each layer just-in-time from the pinned slab."""
        var L = x_in.shape()[1]
        var llm = mul(llm_in, masks.llm_mask, ctx)
        var x = mul(x_in, masks.img_mask, ctx)
        x = mul(_lin(self.active, x, "input_proj.weight", "input_proj.bias", ctx), masks.img_mask, ctx)

        var t_cond = reshape(_t_embed_r(self.active, t_in, hidden, ctx), [1, 1, hidden], ctx)
        var adaln_input = silu(_lin(self.active, t_cond, "adaln_proj.weight", "adaln_proj.bias", ctx), ctx)

        llm = rms_norm(llm, self.active.w("llm_cond_norm.weight"), Float32(1.0e-6), ctx)
        llm = mul(_lin(self.active, llm, "llm_cond_proj.weight", "llm_cond_proj.bias", ctx), masks.llm_mask, ctx)

        var h = add(x, llm, ctx)
        var iemb = reshape(gather_rows(self.active.w("embed_image_indicator.weight"), masks.img_ids, ctx), [1, L, hidden], ctx)
        h = add(h, iemb, ctx)

        for li in range(num_layers):
            self.stage_layer(li, ctx)
            var p = String("layers.") + String(li) + "."
            h = _block_r_masked[S, 0](
                self.active, p, h, adaln_input, cosf, sinf, 0,
                num_heads, head_dim, hidden, ctx
            )
            # Fence each layer: without it the pass's dequant/activation
            # transients stay live until the next sync (~4.5GB per uncond
            # pass, measured job-0207/0208) and the arena hoards the card
            # until a direct cuBLAS workspace alloc OOMs. ~ms per fence vs
            # tens-of-seconds steps.
            ctx.synchronize()

        var fscale = add_scalar(_lin(self.active, silu(adaln_input, ctx), "final_layer.adaln_modulation.weight", "final_layer.adaln_modulation.bias", ctx), Float32(1.0), ctx)
        var hn = mul(layer_norm_no_affine(h, Float32(1.0e-6), ctx), fscale, ctx)
        var out = _lin(self.active, hn, "final_layer.linear.weight", "final_layer.linear.bias", ctx)
        return cast_tensor(out, STDtype.F32, ctx)
