# Forward-parity vs Serenity: same fixed latent+cap → compare Mojo velocity to
# Serenity's ZImageTransformer2DModel velocity (parity/zi_fwd.safetensors).
# LoRA B=0 → forward == base, directly comparable to reference trainer base transformer.
from std.math import sqrt
from max.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.cast import cast_tensor
from serenity_trainer.modelLoader.ZImageModelLoader import ZImageWeights
from serenity_trainer.model.ZImageModel import build_zimage_lora_set
from serenity_trainer.model.ZImageDiT import zimage_forward_full_lora

comptime HL = 16
comptime WL = 16
comptime CAPLEN = 64

def main() raises:
    var ctx = DeviceContext()
    # fixed inputs + reference trainer reference velocity
    var st = ShardedSafeTensors.open(String("trainer/parity/zi_fwd.safetensors"))
    var lat_f32 = Tensor.from_view(st.tensor_view(String("latent")), ctx)      # [1,16,16,16] f32
    var cap_f32 = Tensor.from_view(st.tensor_view(String("cap")), ctx)         # [64,2560] f32
    var vel_ref = Tensor.from_view(st.tensor_view(String("velocity")), ctx)    # [1,16,16,16] f32
    var lat = cast_tensor(lat_f32, STDtype.BF16, ctx)
    var cap = cast_tensor(cap_f32, STDtype.BF16, ctx)

    print("loading zimage_base ...")
    var w = ZImageWeights.load(String("/home/alex/.serenity/models/zimage_base/transformer"), ctx)
    var loras = build_zimage_lora_set(8, Float32(8.0), ctx)   # B=0 → identity (forward==base)
    print("running Mojo forward (B=0) ...")
    var fo = zimage_forward_full_lora[HL, WL, CAPLEN](lat, Float32(0.75), cap, w, loras, ctx)

    var m = fo.velocity.to_host(ctx)       # bf16→f32 host
    var r = vel_ref.to_host(ctx)
    var n = len(m)
    if len(r) != n: raise Error("len mismatch")
    var dot = Float64(0.0); var na = Float64(0.0); var nb = Float64(0.0); var mx = Float32(0.0)
    for i in range(n):
        var a = m[i]; var b = r[i]
        dot += Float64(a)*Float64(b); na += Float64(a)*Float64(a); nb += Float64(b)*Float64(b)
        var d = a-b; var ad = d if d>=0.0 else -d
        if ad > mx: mx = ad
    var cos = dot / (sqrt(na)*sqrt(nb))
    print("=== FORWARD PARITY vs Serenity ===")
    print("  n =", n, " cos =", cos, " max_abs_diff =", mx)
    print("  Mojo[0:3] =", m[0], m[1], m[2], "  reference trainer[0:3] =", r[0], r[1], r[2])
