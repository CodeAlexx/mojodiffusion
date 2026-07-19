# models/chroma/weights.mojo — REAL Chroma1-HD transformer safetensors -> the
# Flux-shaped block weight structs the proven block fwd/bwd consume.
#
# THE GENUINE CHROMA-SPECIFIC COMPUTE: Chroma stores SEPARATE attn projections;
# the proven block expects FUSED ones (see models/chroma/chroma_block.mojo). This
# loader does the row-stack:
#
#   DOUBLE block bi, img stream:
#     transformer_blocks.{bi}.attn.to_q.weight   [D,D]   ┐ row-stack (q;k;v)
#     transformer_blocks.{bi}.attn.to_k.weight   [D,D]   ├-> wqkv  [3D,D]
#     transformer_blocks.{bi}.attn.to_v.weight   [D,D]   ┘
#     transformer_blocks.{bi}.attn.to_q/k/v.bias [D]     -> bqkv  [3D]
#     transformer_blocks.{bi}.attn.to_out.0.weight [D,D] -> wproj [D,D]   + .bias
#     transformer_blocks.{bi}.ff.net.0.proj.weight [Fmlp,D] -> wmlp0      + .bias
#     transformer_blocks.{bi}.ff.net.2.weight      [D,Fmlp] -> wmlp2      + .bias
#     transformer_blocks.{bi}.attn.norm_q.weight   [Dh]   -> q_norm
#     transformer_blocks.{bi}.attn.norm_k.weight   [Dh]   -> k_norm
#   DOUBLE block bi, txt stream (add_q_proj/add_k_proj/add_v_proj, to_add_out,
#     ff_context.net.0.proj / ff_context.net.2, norm_added_q / norm_added_k).
#
#   SINGLE block bi:
#     single_transformer_blocks.{bi}.attn.to_q/.to_k/.to_v.weight [D,D]  ┐
#     single_transformer_blocks.{bi}.proj_mlp.weight              [Fmlp,D] ┘
#       row-stack (to_q;to_k;to_v;proj_mlp) -> w1 [3D+Fmlp, D]   (+ biases -> b1)
#     single_transformer_blocks.{bi}.proj_out.weight [D, D+Fmlp] -> w2    + .bias
#     single_transformer_blocks.{bi}.attn.norm_q/.norm_k.weight  [Dh] -> q/k_norm
#
# Dims CONFIRMED from the real safetensors header (this session): D=3072, H=24,
# Dh=128, Fmlp=12288, 19 double + 38 single. The checkpoint is BF16; production
# loaders keep checkpoint tensors device-resident in their stored dtype.
#
# Mojo 0.26.x+: def not fn; move-only Tensor; reuses io.safetensors +
# io.tensor_view.from_parts (the Flux loader pattern).

from std.memory import ArcPointer
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.ops.tensor_algebra import concat
from serenity_trainer.model.chroma.chroma_block import (
    ChromaStreamWeights, ChromaDoubleBlockWeights, ChromaSingleBlockWeights,
)


comptime TArc = ArcPointer[Tensor]


def _row_stack3_tensors(var a: Tensor, var b: Tensor, var c: Tensor, ctx: DeviceContext) raises -> TArc:
    return TArc(concat(0, ctx, a, b, c))


def _row_stack4_tensors(
    var a: Tensor, var b: Tensor, var c: Tensor, var d: Tensor, ctx: DeviceContext
) raises -> TArc:
    return TArc(concat(0, ctx, a, b, c, d))


# ── DOUBLE block: per-stream loader (img: to_q/k/v/to_out.0/ff.net.0.proj/ff.net.2;
#    txt: add_q/k/v_proj/to_add_out/ff_context.net.0.proj/ff_context.net.2). ──
def _load_double_stream(
    st: SafeTensors, bp: String,
    qkey: String, kkey: String, vkey: String, outkey: String,
    mlp0key: String, mlp2key: String, nqkey: String, nkkey: String,
    ctx: DeviceContext,
) raises -> ChromaStreamWeights:
    var wq = _load_dev_preserve(st, bp + qkey + String(".weight"), ctx)
    var wk = _load_dev_preserve(st, bp + kkey + String(".weight"), ctx)
    var wv = _load_dev_preserve(st, bp + vkey + String(".weight"), ctx)
    var wqkv = _row_stack3_tensors(wq^, wk^, wv^, ctx)      # [3D, D]

    var bq = _load_dev_preserve(st, bp + qkey + String(".bias"), ctx)
    var bk = _load_dev_preserve(st, bp + kkey + String(".bias"), ctx)
    var bv = _load_dev_preserve(st, bp + vkey + String(".bias"), ctx)
    var bqkv = _row_stack3_tensors(bq^, bk^, bv^, ctx)      # [3D]

    return ChromaStreamWeights(
        wqkv^, bqkv^,
        TArc(_load_dev_preserve(st, bp + outkey + String(".weight"), ctx)),   # wproj [D,D]
        TArc(_load_dev_preserve(st, bp + outkey + String(".bias"), ctx)),     # bproj [D]
        TArc(_load_dev_preserve(st, bp + mlp0key + String(".weight"), ctx)),  # wmlp0 [Fmlp,D]
        TArc(_load_dev_preserve(st, bp + mlp0key + String(".bias"), ctx)),    # bmlp0 [Fmlp]
        TArc(_load_dev_preserve(st, bp + mlp2key + String(".weight"), ctx)),  # wmlp2 [D,Fmlp]
        TArc(_load_dev_preserve(st, bp + mlp2key + String(".bias"), ctx)),    # bmlp2 [D]
        TArc(_load_dev_preserve(st, bp + nqkey + String(".weight"), ctx)),    # q_norm [Dh]
        TArc(_load_dev_preserve(st, bp + nkkey + String(".weight"), ctx)),    # k_norm [Dh]
    )


def load_double_block_weights(
    st: SafeTensors, block_idx: Int, ctx: DeviceContext
) raises -> ChromaDoubleBlockWeights:
    var bp = String("transformer_blocks.") + String(block_idx) + String(".")
    var img = _load_double_stream(
        st, bp,
        String("attn.to_q"), String("attn.to_k"), String("attn.to_v"),
        String("attn.to_out.0"), String("ff.net.0.proj"), String("ff.net.2"),
        String("attn.norm_q"), String("attn.norm_k"), ctx,
    )
    var txt = _load_double_stream(
        st, bp,
        String("attn.add_q_proj"), String("attn.add_k_proj"), String("attn.add_v_proj"),
        String("attn.to_add_out"), String("ff_context.net.0.proj"), String("ff_context.net.2"),
        String("attn.norm_added_q"), String("attn.norm_added_k"), ctx,
    )
    return ChromaDoubleBlockWeights(img^, txt^)


# ── SINGLE block loader: row-stack to_q/to_k/to_v/proj_mlp -> w1 [3D+Fmlp, D]. ──
def load_single_block_weights(
    st: SafeTensors, block_idx: Int, ctx: DeviceContext
) raises -> ChromaSingleBlockWeights:
    var sp = String("single_transformer_blocks.") + String(block_idx) + String(".")

    var wq = _load_dev_preserve(st, sp + String("attn.to_q.weight"), ctx)
    var wk = _load_dev_preserve(st, sp + String("attn.to_k.weight"), ctx)
    var wv = _load_dev_preserve(st, sp + String("attn.to_v.weight"), ctx)
    var wm = _load_dev_preserve(st, sp + String("proj_mlp.weight"), ctx)
    var w1 = _row_stack4_tensors(wq^, wk^, wv^, wm^, ctx)        # [3D+Fmlp, D]

    var bq = _load_dev_preserve(st, sp + String("attn.to_q.bias"), ctx)
    var bk = _load_dev_preserve(st, sp + String("attn.to_k.bias"), ctx)
    var bv = _load_dev_preserve(st, sp + String("attn.to_v.bias"), ctx)
    var bm = _load_dev_preserve(st, sp + String("proj_mlp.bias"), ctx)
    var b1 = _row_stack4_tensors(bq^, bk^, bv^, bm^, ctx)        # [3D+Fmlp]

    return ChromaSingleBlockWeights(
        w1^, b1^,
        TArc(_load_dev_preserve(st, sp + String("proj_out.weight"), ctx)),   # w2 [D, D+Fmlp]
        TArc(_load_dev_preserve(st, sp + String("proj_out.bias"), ctx)),     # b2 [D]
        TArc(_load_dev_preserve(st, sp + String("attn.norm_q.weight"), ctx)),# q_norm [Dh]
        TArc(_load_dev_preserve(st, sp + String("attn.norm_k.weight"), ctx)),# k_norm [Dh]
    )


# ── stack-level base loader: x_embedder / context_embedder / proj_out ────────
# These frozen base linears live OUTSIDE the streamed blocks; the offload stack
# (chroma_stack_lora.mojo) holds them resident. The approximator
# (distilled_guidance_layer) is loaded separately via models/dit/chroma_dit.mojo
# ChromaDitCache (it produces the per-step pooled_temb modulation table).
def _load_dev_preserve(st: SafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    var info = st.tensor_info(name)
    var bytes = st.tensor_bytes(name)
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    return Tensor.from_view(tv, ctx)


from serenity_trainer.model.chroma.chroma_stack_lora import ChromaStackBase


def load_chroma_stack_base(
    st: SafeTensors, num_double: Int, num_single: Int, ctx: DeviceContext
) raises -> ChromaStackBase:
    return ChromaStackBase(
        ArcPointer(_load_dev_preserve(st, String("x_embedder.weight"), ctx)),
        ArcPointer(_load_dev_preserve(st, String("x_embedder.bias"), ctx)),
        ArcPointer(_load_dev_preserve(st, String("context_embedder.weight"), ctx)),
        ArcPointer(_load_dev_preserve(st, String("context_embedder.bias"), ctx)),
        ArcPointer(_load_dev_preserve(st, String("proj_out.weight"), ctx)),
        ArcPointer(_load_dev_preserve(st, String("proj_out.bias"), ctx)),
        num_double, num_single,
    )
