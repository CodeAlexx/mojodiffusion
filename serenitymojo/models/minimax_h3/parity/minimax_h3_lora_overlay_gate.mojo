# minimax_h3_lora_overlay_gate — the inference LoRA overlay must reproduce
# W' = W + mult·(alpha/rank)·(up@down) on a REAL raw block weight, and its
# file parser must round-trip the trainer's save format.
#
# Uses the trainer's own smoke/sanity LoRA file (trainer key format) plus
# real block-0 raw weights. Reference: host F64 recompute of sampled entries.
from max.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.tensor import Tensor
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.models.minimax_h3.h3_lora_overlay import (
    H3LoraOverlay, h3_overlay_slot_of,
)

comptime CKPT = "/home/alex/.serenity/models/checkpoints/MiniMax-H3/FL2VA/transformer"
comptime LORA = "/tmp/claude-1000/-home-alex-mojodiffusion/c2c1504f-695c-4e34-9036-ad52dfe27b5a/scratchpad/h3_real_sanity/sanity_step10.safetensors"
comptime MULT = Float32(1.0)


def main() raises:
    var ctx = DeviceContext()
    var overlay = H3LoraOverlay.load(String(LORA), MULT, ctx)
    print("adapters:", overlay.adapters)
    if overlay.adapters != 200:
        raise Error("expected 200 adapters (50 blocks x 4 slots)")

    var sharded = ShardedSafeTensors.open(String(CKPT))
    var name = String("blocks.0.attn.out_proj.weight")  # [5376, 7168]
    var w_raw = cast_tensor(
        Tensor.from_view(sharded.tensor_view(name), ctx), STDtype.BF16, ctx
    )
    var slot = h3_overlay_slot_of(String("attn.out_proj.weight"))
    if not overlay.has(0, slot):
        raise Error("block-0 out_proj adapter missing")
    var w_new = overlay.apply_raw(0, slot, w_raw, ctx)
    ctx.synchronize()

    # host reference on sampled entries: W' == bf16(f32(W) + s*(up@down))
    var lf = SafeTensors.open(String(LORA))
    var base = String("lora_unet_blocks_0_attn_out_proj")
    var d_info = lf.tensor_info(base + ".lora_down.weight")
    var d_h = Tensor.from_view(
        from_parts(d_info.dtype, d_info.shape.copy(), lf.tensor_bytes(base + ".lora_down.weight")), ctx
    ).to_host(ctx)
    var u_info = lf.tensor_info(base + ".lora_up.weight")
    var u_h = Tensor.from_view(
        from_parts(u_info.dtype, u_info.shape.copy(), lf.tensor_bytes(base + ".lora_up.weight")), ctx
    ).to_host(ctx)
    var rank = d_info.shape[0]
    var in_f = d_info.shape[1]
    var a_h = lf.tensor_bytes(base + ".alpha")
    # alpha is f32 [1]
    var alpha_bits = (
        Int(a_h[0]) | (Int(a_h[1]) << 8) | (Int(a_h[2]) << 16) | (Int(a_h[3]) << 24)
    )
    var alpha = _bits_to_f32(alpha_bits)
    var s = Float64(MULT) * Float64(alpha) / Float64(rank)

    var w_old_h = w_raw.to_host(ctx)
    var w_new_h = w_new.to_host(ctx)
    var checked = 0
    var max_rel = Float64(0)
    for probe in range(64):
        var r = (probe * 379) % 5376
        var c = (probe * 733) % in_f
        var delta = Float64(0)
        for k in range(rank):
            delta += Float64(u_h[r * rank + k]) * Float64(d_h[k * in_f + c])
        var expect = Float64(w_old_h[r * in_f + c]) + s * delta
        var got = Float64(w_new_h[r * in_f + c])
        var denom = abs(expect)
        if denom < 1e-3:
            denom = 1e-3
        var rel = abs(expect - got) / denom
        if rel > max_rel:
            max_rel = rel
        checked += 1
    print("checked", checked, "entries; max rel", max_rel)
    # bf16 storage rounding bounds the error: one quantum ~ 2^-8 rel
    if max_rel > 0.006:
        raise Error("overlay math off beyond bf16 rounding")
    print("PASS: h3 lora overlay parses trainer format + applies the exact overlay add")


def _bits_to_f32(bits: Int) -> Float32:
    var u = UInt32(bits)
    return UnsafePointer(to=u).bitcast[Float32]()[]
