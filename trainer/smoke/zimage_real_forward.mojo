# Real-weights Z-Image full forward (load zimage_base, LoRA B=0 → forward==base).
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.random import randn
from serenity_trainer.modelLoader.ZImageModelLoader import ZImageWeights
from serenity_trainer.model.ZImageModel import build_zimage_lora_set
from serenity_trainer.model.ZImageDiT import zimage_forward_full_lora
from serenity_trainer.modelSetup.BaseZImageSetup import model_t_from_timestep

comptime HL = 16
comptime WL = 16
comptime CAPLEN = 64

def main() raises:
    var ctx = DeviceContext()
    print("loading zimage_base weights ...")
    var w = ZImageWeights.load(String("/home/alex/.serenity/models/zimage_base/transformer"), ctx)
    print("  loaded.")
    var lat = randn([1, 16, HL, WL], UInt64(11), STDtype.BF16, ctx)
    var cap = randn([CAPLEN, 2560], UInt64(22), STDtype.BF16, ctx)   # cap_feat_dim=2560
    var loras = build_zimage_lora_set(8, Float32(8.0), ctx)           # B=0 → identity
    var t_model = model_t_from_timestep(250)                          # 0.75
    print("running full forward [HL=16,WL=16,CAPLEN=64] ...")
    var fo = zimage_forward_full_lora[HL, WL, CAPLEN](lat, t_model, cap, w, loras, ctx)
    var v = fo.velocity.to_host(ctx)
    var n = len(v)
    var s = Float32(0.0); var s2 = Float32(0.0); var nf = 0
    for i in range(n):
        var x = v[i]
        if x != x: nf += 1
        s += x; s2 += x*x
    var mean = s / Float32(n)
    print("velocity: n =", n, " mean =", mean, " var =", s2/Float32(n) - mean*mean, " nonfinite =", nf)
    print("ZIMAGE REAL FORWARD OK" if nf == 0 else "ZIMAGE REAL FORWARD HAS NONFINITE")
