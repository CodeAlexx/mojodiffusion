# models/dit/tests/hidream_o1_block_parity.mojo — campaign P1 gate:
# hidream_o1_train_block fwd+bwd vs the torch-autograd oracle
# (/tmp/hidream_block_oracle.safetensors, generated from the DiffSynth
# Qwen3VLTextDecoderLayer math — rms->qkv(+lora)->qk-norm->halfsplit rope->
# GQA->prefix-causal sdpa->o(+lora)->residual->rms->swiglu mlp(+lora)->
# residual; reduced-but-faithful dims D=256 H=8 HKV=2 Dh=32 F=512 S=96
# ar_len=32, rank-4 LoRA scale 0.5 on all 7 slots, F32).
#
# Acceptance: out + d_hidden + all 14 adapter grads cosine >= 0.99999
# (F32 end-to-end; order-of-ops differences only).
#
# Build:
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#   pixi run mojo build -I . -Xlinker -lm -Xlinker -lcuda \
#     -Xlinker -L.pixi/envs/default/lib -Xlinker -lsqlite3 \
#     serenitymojo/models/dit/tests/hidream_o1_block_parity.mojo \
#     -o /tmp/hidream_block_par
# Run: LD_LIBRARY_PATH=.pixi/envs/default/lib /tmp/hidream_block_par

from std.gpu.host import DeviceContext
from std.math import sqrt
from std.memory import ArcPointer
from std.collections import Optional
from serenitymojo.tensor import Tensor
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.dtype import STDtype
from serenitymojo.models.zimage.lora_block import ZImageLoraAdapterDevice
from serenitymojo.models.dit.hidream_o1_train_block import (
    HiDreamO1BlockWeights,
    HiDreamO1BlockLora,
    hidream_o1_block_lora_forward,
    hidream_o1_block_lora_backward,
)

comptime TArc = ArcPointer[Tensor]
comptime D = 256
comptime H = 8
comptime HKV = 2
comptime Dh = 32
comptime F = 512
comptime S = 96
comptime RANK = 4
comptime LSCALE = Float32(0.5)
comptime EPS = Float32(1.0e-6)
comptime ORACLE = "/tmp/hidream_block_oracle.safetensors"


def _cos(a: List[Float32], b: List[Float32]) -> Float64:
    var dot = 0.0
    var na = 0.0
    var nb = 0.0
    for i in range(len(a)):
        dot += Float64(a[i]) * Float64(b[i])
        na += Float64(a[i]) * Float64(a[i])
        nb += Float64(b[i]) * Float64(b[i])
    if na == 0.0 or nb == 0.0:
        return -2.0  # degenerate -> guaranteed FAIL
    return dot / (sqrt(na) * sqrt(nb))


def _gate(name: String, got: List[Float32], exp: List[Float32]) raises -> Bool:
    var c = _cos(got, exp)
    var ok = c >= 0.99999
    print("GATE hidream_block " + name + " cos=", c, " PASS" if ok else " FAIL")
    return ok


def _replicate_heads_host(
    half_tab: List[Float32], heads: Int
) raises -> List[Float32]:
    """[S, half] per-position -> [S*heads, half] rows (row = s*heads + h),
    matching x flattened [1,S,h,Dh] row order."""
    comptime half = Dh // 2
    var out = List[Float32]()
    for s in range(S):
        for _h in range(heads):
            for c in range(half):
                out.append(half_tab[s * half + c])
    return out^


def _t(fx: ShardedSafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    return Tensor.from_view(fx.tensor_view(name), ctx)


def _ta(fx: ShardedSafeTensors, name: String, ctx: DeviceContext) raises -> TArc:
    return TArc(Tensor.from_view(fx.tensor_view(name), ctx))


def _adapter(
    fx: ShardedSafeTensors, name: String, in_f: Int, out_f: Int,
    ctx: DeviceContext,
) raises -> Optional[ZImageLoraAdapterDevice]:
    return Optional[ZImageLoraAdapterDevice](ZImageLoraAdapterDevice(
        _ta(fx, "lora_" + name + "_a", ctx), _ta(fx, "lora_" + name + "_b", ctx),
        RANK, in_f, out_f, LSCALE,
    ))


def main() raises:
    var ctx = DeviceContext()
    var fx = ShardedSafeTensors.open(String(ORACLE))

    var w = HiDreamO1BlockWeights(
        _ta(fx, "w_in_ln", ctx), _ta(fx, "w_qw", ctx), _ta(fx, "w_kw", ctx),
        _ta(fx, "w_vw", ctx), _ta(fx, "w_q_norm", ctx), _ta(fx, "w_k_norm", ctx),
        _ta(fx, "w_ow", ctx), _ta(fx, "w_post_ln", ctx), _ta(fx, "w_gw", ctx),
        _ta(fx, "w_uw", ctx), _ta(fx, "w_dw", ctx),
    )

    var lora = HiDreamO1BlockLora(
        _adapter(fx, "q", D, H * Dh, ctx), _adapter(fx, "k", D, HKV * Dh, ctx),
        _adapter(fx, "v", D, HKV * Dh, ctx), _adapter(fx, "o", H * Dh, D, ctx),
        _adapter(fx, "gate", D, F, ctx), _adapter(fx, "up", D, F, ctx),
        _adapter(fx, "down", F, D, ctx),
    )

    var cos_half = fx.tensor_view("cos_half")
    var sin_half = fx.tensor_view("sin_half")
    var cos_h_host = Tensor.from_view(cos_half, ctx).to_host(ctx)
    var sin_h_host = Tensor.from_view(sin_half, ctx).to_host(ctx)
    comptime half = Dh // 2
    var cq: List[Int] = [S * H * half]
    var ck: List[Int] = [S * HKV * half]
    var cos_q = Tensor.from_host(_replicate_heads_host(cos_h_host, H), cq.copy(), STDtype.F32, ctx)
    var sin_q = Tensor.from_host(_replicate_heads_host(sin_h_host, H), cq^, STDtype.F32, ctx)
    var cos_k = Tensor.from_host(_replicate_heads_host(cos_h_host, HKV), ck.copy(), STDtype.F32, ctx)
    var sin_k = Tensor.from_host(_replicate_heads_host(sin_h_host, HKV), ck^, STDtype.F32, ctx)

    var mask_hss = _t(fx, "mask_hss", ctx)                       # [H*S, S] F32
    var mask_h = mask_hss.to_host(ctx)
    var m4_sh: List[Int] = [1, H, S, S]
    var mask4 = Tensor.from_host(mask_h.copy(), m4_sh^, STDtype.F32, ctx)
    var mhs_sh: List[Int] = [H * S, S]
    var mask_f32 = Tensor.from_host(mask_h^, mhs_sh^, STDtype.F32, ctx)

    var hidden = _ta(fx, "hidden", ctx)
    var d_out = _t(fx, "d_out", ctx)

    print("[fwd] hidream_o1_block_lora_forward ...")
    var fwd = hidream_o1_block_lora_forward[S, H, HKV, Dh](
        hidden, w, lora, cos_q, sin_q, cos_k, sin_k, mask4,
        D, F, EPS, ctx,
    )
    ctx.synchronize()
    var ok = _gate(String("out"), fwd.out[].to_host(ctx), _t(fx, "out", ctx).to_host(ctx))

    print("[bwd] hidream_o1_block_lora_backward ...")
    var g = hidream_o1_block_lora_backward[S, H, HKV, Dh](
        d_out, w, lora, fwd.saved, cos_q, sin_q, cos_k, sin_k, mask_f32,
        D, F, EPS, ctx,
    )
    ctx.synchronize()
    ok = _gate(String("d_hidden"), g.d_hidden[].to_host(ctx), _t(fx, "d_hidden", ctx).to_host(ctx)) and ok

    var names: List[String] = [
        String("q"), String("k"), String("v"), String("o"),
        String("gate"), String("up"), String("down"),
    ]
    for i in range(7):
        if not g.d_a[i]:
            print("GATE hidream_block d_" + names[i] + "_a FAIL (missing)")
            ok = False
            continue
        ok = _gate("d_" + names[i] + "_a", g.d_a[i].value()[].to_host(ctx), _t(fx, "d_" + names[i] + "_a", ctx).to_host(ctx)) and ok
        ok = _gate("d_" + names[i] + "_b", g.d_b[i].value()[].to_host(ctx), _t(fx, "d_" + names[i] + "_b", ctx).to_host(ctx)) and ok

    if ok:
        print("=== hidream_o1_block_parity: ALL GATES PASS ===")
    else:
        print("=== hidream_o1_block_parity: FAIL ===")
        raise Error("hidream block parity failed")
