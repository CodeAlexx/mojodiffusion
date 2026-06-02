# models/sdxl/real_weights.mojo — REAL-WEIGHT loaders assembling the gated
# fwd+bwd unit weight structs (EmbWeights, ResBlockWeights, SpatialTransformerWeights)
# from the extracted `sdxl_unet_bf16.safetensors` (LDM key layout, prefix-stripped).
#
# This is the missing piece between the parity-gated block units (which used
# SYNTHETIC weights) and a REAL run: it builds the SAME structs the gated
# forward/backward consume, but from real disk tensors. NO new ops/ primitive
# (Tenet 1) — pure key→tensor assembly reusing weights._load_f32 / _load_conv_rscf
# and the proven load_resblock_weights.
#
# Keys verified vs the live checkpoint (1680 tensors, prefix-stripped):
#   time_embed.0/2.weight|bias              [1280,320]/[1280] , [1280,1280]/[1280]
#   label_emb.0.0/0.2.weight|bias           [1280,2816]/[1280], [1280,1280]/[1280]
#   <res>.in_layers.0/.2 ; emb_layers.1 ; out_layers.0/.3 ; skip_connection (load_resblock_weights)
#   <st>.norm.weight|bias                   [C]
#   <st>.proj_in/proj_out.weight|bias       [C,C]/[C]   (use_linear_in_transformer)
#   <st>.transformer_blocks.<j>.norm1/2/3.weight|bias            [C]
#   <st>.transformer_blocks.<j>.attn1/2.{to_q,to_k,to_v}.weight  (no bias)
#   <st>.transformer_blocks.<j>.attn1/2.to_out.0.weight|bias
#   <st>.transformer_blocks.<j>.ff.net.0.proj.weight|bias        [2*Cff,C]/[2*Cff]
#   <st>.transformer_blocks.<j>.ff.net.2.weight|bias             [C,Cff]/[C]
#
# All returned F32 (training compute dtype); conv filters RSCF [Kh,Kw,Cin,Cout].

from std.gpu.host import DeviceContext
from std.memory import ArcPointer
from std.collections import List
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.ops.cast import cast_tensor

from serenitymojo.models.sdxl.embed import EmbWeights
from serenitymojo.models.sdxl.weights import ResBlockWeights, load_resblock_weights
from serenitymojo.models.sdxl.spatial_transformer import (
    AttnWeights, BasicTransformerBlockWeights, SpatialTransformerWeights,
)

comptime TArc = ArcPointer[Tensor]


# ── one named tensor as an F32 device Tensor (mirrors weights._load_f32) ──────
def _f32(st: SafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    var info = st.tensor_info(name)
    var bytes = st.tensor_bytes(name)
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    var t = Tensor.from_view(tv, ctx)
    return cast_tensor(t, STDtype.F32, ctx)


def _arc(st: SafeTensors, name: String, ctx: DeviceContext) raises -> TArc:
    return TArc(_f32(st, name, ctx))


# ── EmbWeights (time + label MLPs) ────────────────────────────────────────────
def load_emb_weights(st: SafeTensors, ctx: DeviceContext) raises -> EmbWeights:
    return EmbWeights(
        _f32(st, String("time_embed.0.weight"), ctx), _f32(st, String("time_embed.0.bias"), ctx),
        _f32(st, String("time_embed.2.weight"), ctx), _f32(st, String("time_embed.2.bias"), ctx),
        _f32(st, String("label_emb.0.0.weight"), ctx), _f32(st, String("label_emb.0.0.bias"), ctx),
        _f32(st, String("label_emb.0.2.weight"), ctx), _f32(st, String("label_emb.0.2.bias"), ctx),
    )


# ── one BasicTransformerBlock's attn weights ──────────────────────────────────
def _load_attn(st: SafeTensors, prefix: String, ctx: DeviceContext) raises -> AttnWeights:
    return AttnWeights(
        _arc(st, prefix + String(".to_q.weight"), ctx),
        _arc(st, prefix + String(".to_k.weight"), ctx),
        _arc(st, prefix + String(".to_v.weight"), ctx),
        _arc(st, prefix + String(".to_out.0.weight"), ctx),
        _arc(st, prefix + String(".to_out.0.bias"), ctx),
    )


def _load_btb(st: SafeTensors, prefix: String, ctx: DeviceContext) raises -> BasicTransformerBlockWeights:
    return BasicTransformerBlockWeights(
        _arc(st, prefix + String(".norm1.weight"), ctx), _arc(st, prefix + String(".norm1.bias"), ctx),
        _load_attn(st, prefix + String(".attn1"), ctx),
        _arc(st, prefix + String(".norm2.weight"), ctx), _arc(st, prefix + String(".norm2.bias"), ctx),
        _load_attn(st, prefix + String(".attn2"), ctx),
        _arc(st, prefix + String(".norm3.weight"), ctx), _arc(st, prefix + String(".norm3.bias"), ctx),
        _arc(st, prefix + String(".ff.net.0.proj.weight"), ctx), _arc(st, prefix + String(".ff.net.0.proj.bias"), ctx),
        _arc(st, prefix + String(".ff.net.2.weight"), ctx), _arc(st, prefix + String(".ff.net.2.bias"), ctx),
    )


# ── one SpatialTransformer's weights (depth BTBs) ─────────────────────────────
def load_st_weights(
    st: SafeTensors, prefix: String, depth: Int, ctx: DeviceContext
) raises -> SpatialTransformerWeights:
    var blocks = List[BasicTransformerBlockWeights]()
    for j in range(depth):
        blocks.append(_load_btb(st, prefix + String(".transformer_blocks.") + String(j), ctx))
    return SpatialTransformerWeights(
        _arc(st, prefix + String(".norm.weight"), ctx), _arc(st, prefix + String(".norm.bias"), ctx),
        _arc(st, prefix + String(".proj_in.weight"), ctx), _arc(st, prefix + String(".proj_in.bias"), ctx),
        blocks^,
        _arc(st, prefix + String(".proj_out.weight"), ctx), _arc(st, prefix + String(".proj_out.bias"), ctx),
    )


# ── conv_in / conv_out / final-GN (loose tensors the stack threads directly) ──
def load_conv_rscf(st: SafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    # OIHW [Cout,Cin,Kh,Kw] -> RSCF [Kh,Kw,Cin,Cout]; same remap as weights._load_conv_rscf.
    var w = _f32(st, name, ctx)
    var sh = w.shape()
    if len(sh) != 4:
        raise Error(String("conv weight ") + name + " not rank-4 OIHW")
    var cout = sh[0]; var cin = sh[1]; var kh = sh[2]; var kw = sh[3]
    var host = w.to_host(ctx)
    var rscf = List[Float32]()
    for _ in range(kh * kw * cin * cout):
        rscf.append(0.0)
    for o in range(cout):
        for ci in range(cin):
            for r in range(kh):
                for s in range(kw):
                    var src = ((o * cin + ci) * kh + r) * kw + s
                    var dst = ((r * kw + s) * cin + ci) * cout + o
                    rscf[dst] = host[src]
    var rshape = List[Int]()
    rshape.append(kh); rshape.append(kw); rshape.append(cin); rshape.append(cout)
    return Tensor.from_host(rscf, rshape^, STDtype.F32, ctx)


def load_bias(st: SafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    return _f32(st, name, ctx)


# ═══════════════════════════════════════════════════════════════════════════════
# FULL real-weight builder: assemble SdxlRealWeights from the checkpoint, in the
# exact run-order the real-dims stack consumes (sdxl_real_train.mojo indexing).
# ═══════════════════════════════════════════════════════════════════════════════
from serenitymojo.models.sdxl.sdxl_real_train import SdxlRealWeights
from serenitymojo.models.sdxl.weights import load_resblock_weights


def build_sdxl_real_weights(st: SafeTensors, ctx: DeviceContext) raises -> SdxlRealWeights:
    var emb = load_emb_weights(st, ctx)
    var conv_in_w = load_conv_rscf(st, String("input_blocks.0.0.weight"), ctx)
    var conv_in_b = load_bias(st, String("input_blocks.0.0.bias"), ctx)
    var out_gn_w = _f32(st, String("out.0.weight"), ctx)
    var out_gn_b = _f32(st, String("out.0.bias"), ctx)
    var conv_out_w = load_conv_rscf(st, String("out.2.weight"), ctx)
    var conv_out_b = load_bias(st, String("out.2.bias"), ctx)

    # ── ResBlocks, run order (matches R_* indices) ──
    var res = List[ArcPointer[ResBlockWeights]]()
    var res_prefixes = List[String]()
    res_prefixes.append(String("input_blocks.1.0"))   # R_IN1
    res_prefixes.append(String("input_blocks.2.0"))   # R_IN2
    res_prefixes.append(String("input_blocks.4.0"))   # R_IN4
    res_prefixes.append(String("input_blocks.5.0"))   # R_IN5
    res_prefixes.append(String("input_blocks.7.0"))   # R_IN7
    res_prefixes.append(String("input_blocks.8.0"))   # R_IN8
    res_prefixes.append(String("middle_block.0"))     # R_MID0
    res_prefixes.append(String("middle_block.2"))     # R_MID2
    res_prefixes.append(String("output_blocks.0.0"))  # R_OUT0
    res_prefixes.append(String("output_blocks.1.0"))  # R_OUT1
    res_prefixes.append(String("output_blocks.2.0"))  # R_OUT2
    res_prefixes.append(String("output_blocks.3.0"))  # R_OUT3
    res_prefixes.append(String("output_blocks.4.0"))  # R_OUT4
    res_prefixes.append(String("output_blocks.5.0"))  # R_OUT5
    res_prefixes.append(String("output_blocks.6.0"))  # R_OUT6
    res_prefixes.append(String("output_blocks.7.0"))  # R_OUT7
    res_prefixes.append(String("output_blocks.8.0"))  # R_OUT8
    for i in range(len(res_prefixes)):
        res.append(ArcPointer(load_resblock_weights(st, res_prefixes[i], ctx)))

    # ── downsamples (in3, in6) — the conv lives at <prefix>.op ──
    var down_w = List[ArcPointer[Tensor]]()
    var down_b = List[ArcPointer[Tensor]]()
    down_w.append(ArcPointer(load_conv_rscf(st, String("input_blocks.3.0.op.weight"), ctx)))
    down_b.append(ArcPointer(load_bias(st, String("input_blocks.3.0.op.bias"), ctx)))
    down_w.append(ArcPointer(load_conv_rscf(st, String("input_blocks.6.0.op.weight"), ctx)))
    down_b.append(ArcPointer(load_bias(st, String("input_blocks.6.0.op.bias"), ctx)))

    # ── upsamples (out2, out5) — the conv lives at <prefix>.conv ──
    var up_w = List[ArcPointer[Tensor]]()
    var up_b = List[ArcPointer[Tensor]]()
    up_w.append(ArcPointer(load_conv_rscf(st, String("output_blocks.2.2.conv.weight"), ctx)))
    up_b.append(ArcPointer(load_bias(st, String("output_blocks.2.2.conv.bias"), ctx)))
    up_w.append(ArcPointer(load_conv_rscf(st, String("output_blocks.5.2.conv.weight"), ctx)))
    up_b.append(ArcPointer(load_bias(st, String("output_blocks.5.2.conv.bias"), ctx)))

    # ── STs, run order (matches ST_* indices). depth: level-1 STs=2, level-2 STs=10 ──
    var sts = List[SpatialTransformerWeights]()
    sts.append(load_st_weights(st, String("input_blocks.4.1"), 2, ctx))    # ST_IN4
    sts.append(load_st_weights(st, String("input_blocks.5.1"), 2, ctx))    # ST_IN5
    sts.append(load_st_weights(st, String("input_blocks.7.1"), 10, ctx))   # ST_IN7
    sts.append(load_st_weights(st, String("input_blocks.8.1"), 10, ctx))   # ST_IN8
    sts.append(load_st_weights(st, String("middle_block.1"), 10, ctx))     # ST_MID
    sts.append(load_st_weights(st, String("output_blocks.0.1"), 10, ctx))  # ST_OUT0
    sts.append(load_st_weights(st, String("output_blocks.1.1"), 10, ctx))  # ST_OUT1
    sts.append(load_st_weights(st, String("output_blocks.2.1"), 10, ctx))  # ST_OUT2
    sts.append(load_st_weights(st, String("output_blocks.3.1"), 2, ctx))   # ST_OUT3
    sts.append(load_st_weights(st, String("output_blocks.4.1"), 2, ctx))   # ST_OUT4
    sts.append(load_st_weights(st, String("output_blocks.5.1"), 2, ctx))   # ST_OUT5

    return SdxlRealWeights(
        emb^, conv_in_w^, conv_in_b^, out_gn_w^, out_gn_b^, conv_out_w^, conv_out_b^,
        res^, down_w^, down_b^, up_w^, up_b^, sts^,
    )


# ── the 11 ST prefixes (run order) for LoRA save key assembly ─────────────────
def sdxl_st_prefixes() -> List[String]:
    var p = List[String]()
    p.append(String("input_blocks.4.1")); p.append(String("input_blocks.5.1"))
    p.append(String("input_blocks.7.1")); p.append(String("input_blocks.8.1"))
    p.append(String("middle_block.1"))
    p.append(String("output_blocks.0.1")); p.append(String("output_blocks.1.1"))
    p.append(String("output_blocks.2.1")); p.append(String("output_blocks.3.1"))
    p.append(String("output_blocks.4.1")); p.append(String("output_blocks.5.1"))
    return p^


# ── per-ST (C, Cff, depth) for building the LoRA sets ─────────────────────────
def sdxl_st_C(i: Int) -> Int:
    # level-1 STs (idx 0,1,8,9,10) -> 640; level-2 STs -> 1280
    if i == 0 or i == 1 or i == 8 or i == 9 or i == 10:
        return 640
    return 1280


def sdxl_st_Cff(i: Int) -> Int:
    # GEGLU inner half Cff = ff.net.2 in_features. SDXL ff mult=4 -> proj out=2*Cff=8*?
    # checkpoint: C=640 -> proj [5120,640] (2*Cff=5120 -> Cff=2560=4*C);
    #             C=1280 -> Cff=5120=4*C. So Cff = 4*C.
    return 4 * sdxl_st_C(i)


def sdxl_st_depth(i: Int) -> Int:
    if i == 0 or i == 1 or i == 8 or i == 9 or i == 10:
        return 2
    return 10
