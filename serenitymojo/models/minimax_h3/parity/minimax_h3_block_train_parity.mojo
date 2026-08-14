# minimax_h3_block_train_parity — gate the H3 training block fwd+bwd against
# torch autograd on REAL block-0 weights (parity/h3_block_oracle.py bundle).
# PASS bar: cos >= 0.999 on out, d_x, d_mod, and every weight grad
# (numeric-parity-testing house bar for bf16 hand-chain vs torch-bf16).
from max.gpu.host import DeviceContext

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.tensor import Tensor
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.tensor_algebra import reshape
from serenitymojo.models.minimax_h3.h3_block_train import (
    H3BlockTrainWeights, h3_block_train_forward, h3_block_train_backward,
    H3BlockLoraDevice, h3_block_train_forward_lora, h3_block_train_backward_lora,
)
from serenitymojo.models.klein.lora_block import (
    LoraAdapterDevice, KleinLoraDeviceGradTensors,
)
from std.memory import ArcPointer

comptime ORACLE = "/home/alex/mojodiffusion/output/checks/h3_block0_oracle.safetensors"
comptime CKPT = "/home/alex/.serenity/models/checkpoints/MiniMax-H3/FL2VA/transformer"
comptime S = 384
comptime H = 56
comptime Dh = 128
comptime D = 5376
comptime F = 14336
comptime ROTARY = 96
comptime N_TS = 3
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


def main() raises:
    var ctx = DeviceContext()
    var orc = SafeTensors.open(String(ORACLE))

    # ── block-0 weights, RAW layout, bf16 (matches the oracle's block dtype) ──
    var sharded = ShardedSafeTensors.open(String(CKPT))
    var p = String("blocks.0.")
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
    var w = H3BlockTrainWeights(
        wt[0], wt[1], wt[2], wt[3], wt[4], wt[5], wt[6], wt[7],
    )

    # ── oracle inputs ──
    var x = _bf16(orc, String("in_x"), ctx)          # [1,S,D] f32 -> bf16
    var mod = _bf16(orc, String("modulation"), ctx)  # [N_TS, 6D]
    var cos = _f32(orc, String("in_cos"), ctx)       # [S, 96]
    var sin = _f32(orc, String("in_sin"), ctx)
    var loss_w = _f32(orc, String("loss_w"), ctx)    # [1,S,D]
    var idx_t = _f32(orc, String("in_idx"), ctx)
    var idx_h = idx_t.to_host(ctx)
    var idx = List[Int]()
    for i in range(len(idx_h)):
        idx.append(Int(idx_h[i]))

    # x arrives [1,S,D]; the block takes [S,D]
    var x_sd = reshape(x, [S, D], ctx)

    var fwd = h3_block_train_forward[H, Dh, S](
        x_sd, w, mod, idx, cos, sin, D, F, ROTARY, EPS, ctx,
    )
    ctx.synchronize()

    var out_ref = reshape(_f32(orc, String("out"), ctx), [S, D], ctx)
    var c_out = _cos(fwd.out[], out_ref, ctx)
    print("cos(out) =", c_out)

    # upstream grad: d_out = loss_w (loss = sum(out*w))
    var d_out = cast_tensor(reshape(loss_w, [S, D], ctx), STDtype.BF16, ctx)
    var mod_table_rows = mod.shape()[0]
    var grads = h3_block_train_backward[H, Dh, S](
        d_out, w, fwd.saved, idx, cos, sin, mod_table_rows, D, F, ROTARY, EPS, ctx,
    )
    ctx.synchronize()

    var ok = c_out >= 0.999

    var got = List[ArcPointer[Tensor]]()
    var refs = List[String]()
    var labels = List[String]()
    got.append(grads.d_x)
    refs.append(String("d_x"))
    labels.append(String("d_x"))
    got.append(grads.d_mod)
    refs.append(String("d_modulation"))
    labels.append(String("d_mod"))
    got.append(grads.d_qkv_w)
    refs.append(String("d_attn_qkv_proj_weight"))
    labels.append(String("d_qkv_w"))
    got.append(grads.d_out_w)
    refs.append(String("d_attn_out_proj_weight"))
    labels.append(String("d_out_w"))
    got.append(grads.d_fc1_w)
    refs.append(String("d_mlp_fc1_weight"))
    labels.append(String("d_fc1_w"))
    got.append(grads.d_fc2_w)
    refs.append(String("d_mlp_fc2_weight"))
    labels.append(String("d_fc2_w"))
    got.append(grads.d_q_norm)
    refs.append(String("d_attn_q_norm_weight"))
    labels.append(String("d_q_norm"))
    got.append(grads.d_k_norm)
    refs.append(String("d_attn_k_norm_weight"))
    labels.append(String("d_k_norm"))
    got.append(grads.d_norm1_w)
    refs.append(String("d_norm1_weight"))
    labels.append(String("d_norm1"))
    got.append(grads.d_norm2_w)
    refs.append(String("d_norm2_weight"))
    labels.append(String("d_norm2"))
    for i in range(len(got)):
        var expect = _f32(orc, refs[i], ctx)
        var g = got[i]
        var c = _cos(g[], expect, ctx)
        print("cos(" + labels[i] + ") =", c)
        if c < 0.999:
            ok = False

    # ── LoRA arm ──────────────────────────────────────────────────────────
    comptime RANK = 16
    comptime INNER = H * Dh
    var la_names: List[String] = [
        String("qkv"), String("out"), String("fc1"), String("fc2"),
    ]
    var la_in: List[Int] = [D, INNER, D, F]
    var la_out: List[Int] = [3 * INNER, D, 2 * F, D]
    var adapters = List[LoraAdapterDevice]()
    for i in range(4):
        var a = _bf16(orc, String("lora_") + la_names[i] + String("_a"), ctx)
        var b = _bf16(orc, String("lora_") + la_names[i] + String("_b"), ctx)
        adapters.append(LoraAdapterDevice(
            ArcPointer(a^), ArcPointer(b^), RANK, la_in[i], la_out[i], Float32(1.0),
        ))
    var lora = H3BlockLoraDevice(
        Optional[LoraAdapterDevice](adapters[0].copy()),
        Optional[LoraAdapterDevice](adapters[1].copy()),
        Optional[LoraAdapterDevice](adapters[2].copy()),
        Optional[LoraAdapterDevice](adapters[3].copy()),
    )

    var fwd_l = h3_block_train_forward_lora[H, Dh, S](
        x_sd, w, lora, mod, idx, cos, sin, D, F, ROTARY, EPS, ctx,
    )
    ctx.synchronize()
    var lora_out_ref = reshape(_f32(orc, String("lora_out"), ctx), [S, D], ctx)
    var c_lout = _cos(fwd_l.out[], lora_out_ref, ctx)
    print("cos(lora_out) =", c_lout)
    if c_lout < 0.999:
        ok = False

    var bwd_l = h3_block_train_backward_lora[H, Dh, S](
        d_out, w, lora, fwd_l.saved, idx, cos, sin, mod_table_rows,
        D, F, ROTARY, EPS, ctx,
    )
    ctx.synchronize()
    for i in range(4):
        var g: KleinLoraDeviceGradTensors
        if i == 0:
            g = bwd_l.lora.qkv.value().copy()
        elif i == 1:
            g = bwd_l.lora.out.value().copy()
        elif i == 2:
            g = bwd_l.lora.fc1.value().copy()
        else:
            g = bwd_l.lora.fc2.value().copy()
        var ra = _f32(orc, String("d_lora_") + la_names[i] + String("_a"), ctx)
        var rb = _f32(orc, String("d_lora_") + la_names[i] + String("_b"), ctx)
        var ca = _cos(g.d_a[], ra, ctx)
        var cb = _cos(g.d_b[], rb, ctx)
        print("cos(d_lora_" + la_names[i] + "_a) =", ca)
        print("cos(d_lora_" + la_names[i] + "_b) =", cb)
        if ca < 0.999 or cb < 0.999:
            ok = False

    if ok:
        print("PASS: h3 training block fwd+bwd (+LoRA) matches torch autograd (cos>=0.999)")
    else:
        print("FAIL: h3 training block parity below bar")
        raise Error("minimax_h3_block_train_parity failed")
