# Real-data loss parity vs Serenity (SERENITY_loss=0.469153 on real sample, t=499 σ=0.5).
# Feed IDENTICAL scaled_noisy/target/cap (Serenity's own) → Mojo loss must match.
from std.math import sqrt
from max.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.tensor_algebra import sub, mul, mul_scalar
from serenitymojo.ops.reduce import reduce_mean_f32
from serenity_trainer.modelLoader.ZImageModelLoader import ZImageWeights
from serenity_trainer.model.ZImageModel import build_zimage_lora_set
from serenity_trainer.model.ZImageDiT import zimage_forward_full_lora

comptime HL = 72
comptime WL = 56
comptime CAPLEN = 224

def main() raises:
    var ctx = DeviceContext()
    var st = ShardedSafeTensors.open(String("trainer/parity/zi_realdata.safetensors"))
    var sn = cast_tensor(Tensor.from_view(st.tensor_view(String("scaled_noisy")), ctx), STDtype.BF16, ctx)  # [1,16,72,56]
    var tg = cast_tensor(Tensor.from_view(st.tensor_view(String("target")), ctx), STDtype.BF16, ctx)        # [1,16,72,56]
    var cap = cast_tensor(Tensor.from_view(st.tensor_view(String("cap")), ctx), STDtype.BF16, ctx)          # [224,2560]
    var vel_ref = Tensor.from_view(st.tensor_view(String("velocity")), ctx)                                  # f32 reference trainer velocity
    print("loading zimage_base ...")
    var w = ZImageWeights.load(String("/home/alex/.serenity/models/zimage_base/transformer"), ctx)
    var loras = build_zimage_lora_set(8, Float32(8.0), ctx)   # B=0 → base forward
    print("forward [72,56,224] ...")
    var fo = zimage_forward_full_lora[HL, WL, CAPLEN](sn, Float32(0.501), cap, w, loras, ctx)  # t_model=(1000-499)/1000
    var predicted = mul_scalar(fo.velocity, Float32(-1.0), ctx)   # predicted = -velocity
    var diff = sub(predicted, tg, ctx)
    var sq = mul(diff, diff, ctx)
    var dims = List[Int]()
    for i in range(len(sq.shape())): dims.append(i)
    var lt = reduce_mean_f32(sq, dims^, False, ctx).to_host(ctx)
    # velocity cos vs reference trainer
    var m = fo.velocity.to_host(ctx); var r = vel_ref.to_host(ctx)
    var dot=Float64(0); var na=Float64(0); var nb=Float64(0)
    for i in range(len(m)):
        dot+=Float64(m[i])*Float64(r[i]); na+=Float64(m[i])*Float64(m[i]); nb+=Float64(r[i])*Float64(r[i])
    print("=== REAL-DATA LOSS PARITY vs Serenity ===")
    print("  Mojo loss =", lt[0], "   SERENITY_loss = 0.469153")
    print("  velocity cos =", dot/(sqrt(na)*sqrt(nb)))
