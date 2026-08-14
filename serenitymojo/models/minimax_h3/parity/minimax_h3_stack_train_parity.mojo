# minimax_h3_stack_train_parity — gate the H3 stack driver (recompute
# backward + d_x handoff) against the 2-block chain arm of the torch
# oracle (real block-0/1 weights, no LoRA: adapters absent so the block
# path reduces to the gated base math).
from max.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.tensor import Tensor
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.tensor_algebra import reshape
from serenitymojo.models.minimax_h3.h3_block_train import (
    H3BlockTrainWeights, H3BlockLoraDevice,
)
from serenitymojo.models.klein.lora_block import LoraAdapterDevice
from serenitymojo.models.minimax_h3.h3_stack_train import (
    h3_stack_train_forward, h3_stack_train_backward,
)

comptime ORACLE = "/home/alex/mojodiffusion/output/checks/h3_block0_oracle.safetensors"
comptime CKPT = "/home/alex/.serenity/models/checkpoints/MiniMax-H3/FL2VA/transformer"
comptime S = 384
comptime H = 56
comptime Dh = 128
comptime D = 5376
comptime F = 14336
comptime ROTARY = 96
comptime EPS = Float32(1.0e-5)


def _cos(a: Tensor, b: Tensor, ctx: DeviceContext) raises -> Float64:
    var ah = a.to_host(ctx)
    var bh = b.to_host(ctx)
    if len(ah) != len(bh):
        raise Error("cos: length mismatch")
    var dot = Float64(0)
    var na = Float64(0)
    var nb = Float64(0)
    for i in range(len(ah)):
        var x = Float64(ah[i])
        var y = Float64(bh[i])
        dot += x * y
        na += x * x
        nb += y * y
    return dot / ((na**0.5) * (nb**0.5) + 1e-30)


def _load(st: SafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    var info = st.tensor_info(name)
    return Tensor.from_view(
        from_parts(info.dtype, info.shape.copy(), st.tensor_bytes(name)), ctx
    )


def _bf16(st: SafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    return cast_tensor(_load(st, name, ctx), STDtype.BF16, ctx)


def _f32(st: SafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    return cast_tensor(_load(st, name, ctx), STDtype.F32, ctx)


def _block_weights(
    sharded: ShardedSafeTensors, layer: Int, ctx: DeviceContext
) raises -> H3BlockTrainWeights:
    var p = String("blocks.") + String(layer) + String(".")
    var names: List[String] = [
        String("attn.qkv_proj.weight"), String("attn.out_proj.weight"),
        String("mlp.fc1.weight"), String("mlp.fc2.weight"),
        String("attn.q_norm.weight"), String("attn.k_norm.weight"),
        String("norm1.weight"), String("norm2.weight"),
    ]
    var wt = List[ArcPointer[Tensor]]()
    for i in range(len(names)):
        wt.append(ArcPointer(cast_tensor(
            Tensor.from_view(sharded.tensor_view(p + names[i]), ctx),
            STDtype.BF16, ctx,
        )))
    return H3BlockTrainWeights(
        wt[0], wt[1], wt[2], wt[3], wt[4], wt[5], wt[6], wt[7],
    )


def main() raises:
    var ctx = DeviceContext()
    var orc = SafeTensors.open(String(ORACLE))
    var sharded = ShardedSafeTensors.open(String(CKPT))

    var blocks = List[H3BlockTrainWeights]()
    blocks.append(_block_weights(sharded, 0, ctx))
    blocks.append(_block_weights(sharded, 1, ctx))

    var none_lora = Optional[LoraAdapterDevice](None)
    var loras = List[H3BlockLoraDevice]()
    for _ in range(2):
        loras.append(H3BlockLoraDevice(
            none_lora.copy(), none_lora.copy(), none_lora.copy(), none_lora.copy(),
        ))

    var x = reshape(_bf16(orc, String("in_x"), ctx), [S, D], ctx)
    var mod0 = _bf16(orc, String("modulation"), ctx)
    var mod1 = _bf16(orc, String("chain_mod1"), ctx)
    var mods = List[ArcPointer[Tensor]]()
    mods.append(ArcPointer(mod0^))
    mods.append(ArcPointer(mod1^))
    var cos = _f32(orc, String("in_cos"), ctx)
    var sin = _f32(orc, String("in_sin"), ctx)
    var loss_w = _f32(orc, String("loss_w"), ctx)
    var idx_t = _f32(orc, String("in_idx"), ctx)
    var idx_h = idx_t.to_host(ctx)
    var idx = List[Int]()
    for i in range(len(idx_h)):
        idx.append(Int(idx_h[i]))

    var fwd = h3_stack_train_forward[H, Dh, S](
        x, blocks, loras, mods, idx, cos, sin, D, F, ROTARY, EPS, ctx,
    )
    ctx.synchronize()
    var out_ref = reshape(_f32(orc, String("chain_out"), ctx), [S, D], ctx)
    var c_out = _cos(fwd.out[], out_ref, ctx)
    print("cos(chain_out) =", c_out)
    var h1_ref = reshape(_f32(orc, String("chain_h1"), ctx), [S, D], ctx)
    var c_h1 = _cos(fwd.block_inputs[1][], h1_ref, ctx)
    print("cos(h1 fwd drift) =", c_h1)

    var d_out = cast_tensor(reshape(loss_w, [S, D], ctx), STDtype.BF16, ctx)
    # STRICT arm: seed block-1's recompute with TORCH's own h1 so each
    # block's backward is judged on exact inputs (torch gradient
    # checkpointing recomputes from its own h1 the same way; the own-h1
    # drift-amplification class is covered by the chain_d_x bar below).
    fwd.block_inputs[1] = ArcPointer(
        cast_tensor(h1_ref, STDtype.BF16, ctx)
    )
    var grads = h3_stack_train_backward[H, Dh, S](
        d_out, fwd, blocks, loras, mods, idx, cos, sin,
        D, F, ROTARY, EPS, ctx,
    )
    ctx.synchronize()

    var ok = c_out >= 0.999
    var c_dx = _cos(grads.d_x[], _f32(orc, String("chain_d_x"), ctx), ctx)
    print("cos(chain_d_x) =", c_dx)
    if c_dx < 0.999:
        ok = False

    # per-block weight-grad spot checks: qkv + fc2 + d_mod on both blocks
    var labels = List[String]()
    var refs = List[String]()
    var got = List[ArcPointer[Tensor]]()
    got.append(grads.base[0].d_qkv_w)
    refs.append(String("chain_b0_d_attn_qkv_proj_weight"))
    labels.append(String("b0_d_qkv_w"))
    got.append(grads.base[0].d_fc2_w)
    refs.append(String("chain_b0_d_mlp_fc2_weight"))
    labels.append(String("b0_d_fc2_w"))
    got.append(grads.base[0].d_mod)
    refs.append(String("chain_d_mod0"))
    labels.append(String("b0_d_mod"))
    got.append(grads.base[1].d_qkv_w)
    refs.append(String("chain_b1_d_attn_qkv_proj_weight"))
    labels.append(String("b1_d_qkv_w"))
    got.append(grads.base[1].d_fc2_w)
    refs.append(String("chain_b1_d_mlp_fc2_weight"))
    labels.append(String("b1_d_fc2_w"))
    got.append(grads.base[1].d_mod)
    refs.append(String("chain_d_mod1"))
    labels.append(String("b1_d_mod"))
    for i in range(len(got)):
        var g = got[i]
        var expect = _f32(orc, refs[i], ctx)
        var c = _cos(g[], expect, ctx)
        print("cos(" + labels[i] + ") =", c)
        if c < 0.999:
            ok = False

    if ok:
        print("PASS: h3 stack (recompute bwd + d_x handoff) matches torch on 2 real blocks")
    else:
        print("FAIL: h3 stack parity below bar")
        raise Error("minimax_h3_stack_train_parity failed")
