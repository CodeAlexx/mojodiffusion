# Sampler trajectory parity vs Serenity (parity/zi_sampler_ref.safetensors).
# Verify (1) scheduler sigmas/timesteps, (2) denoise latent_final from identical latent0+cap.
from std.math import sqrt
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.cast import cast_tensor
from serenity_trainer.modelLoader.ZImageModelLoader import ZImageWeights
from serenity_trainer.model.ZImageModel import build_zimage_inactive_lora_set
from serenity_trainer.modelSampler.FlowMatchEulerDiscreteScheduler import make_zimage_scheduler
from serenity_trainer.modelSampler.ZImageSampler import _predict_noise_cached
from serenity_trainer.model.ZImageDiT import prepare_zimage_infer_cache

comptime HL = 8
comptime WL = 8
comptime CAPLEN = 224

def main() raises:
    var ctx = DeviceContext()
    var st = ShardedSafeTensors.open(String("trainer/parity/zi_sampler_ref.safetensors"))
    var lat = Tensor.from_view(st.tensor_view(String("latent0")), ctx)        # [1,16,8,8] f32
    var cap = cast_tensor(Tensor.from_view(st.tensor_view(String("cap")), ctx), STDtype.BF16, ctx)
    var sig_ref = Tensor.from_view(st.tensor_view(String("sigmas")), ctx).to_host(ctx)
    var ts_ref = Tensor.from_view(st.tensor_view(String("timesteps")), ctx).to_host(ctx)
    var vfin = Tensor.from_view(st.tensor_view(String("latent_final")), ctx).to_host(ctx)

    var sch = make_zimage_scheduler(8)
    # 1) sigmas/timesteps parity
    print("=== scheduler set_timesteps(8) ===")
    var smax = Float32(0.0)
    for i in range(len(sig_ref)):
        var d = sch.sigmas[i] - sig_ref[i]; var ad = d if d>=0.0 else -d
        if ad > smax: smax = ad
    var tmax = Float32(0.0)
    for i in range(len(ts_ref)):
        var d = sch.timesteps[i] - ts_ref[i]; var ad = d if d>=0.0 else -d
        if ad > tmax: tmax = ad
    print("  sigma max|d| =", smax, "  timestep max|d| =", tmax)

    print("loading zimage_base ...")
    var w = ZImageWeights.load(String("/home/alex/.serenity/models/zimage_base/transformer"), ctx)
    var loras = build_zimage_inactive_lora_set(8)   # base model, no adapter GEMMs
    var cache = prepare_zimage_infer_cache[HL, WL, CAPLEN](cap, w, ctx)
    print("denoise 8 steps ...")
    for i in range(8):
        var t = sch.timesteps[i]
        var t_model = (Float32(1000.0) - t) / Float32(1000.0)
        var np = _predict_noise_cached[HL, WL, CAPLEN](lat, t_model, cache, w, loras, ctx)
        lat = sch.step(np, lat, i, ctx)
    var m = lat.to_host(ctx)
    var dot=Float64(0); var na=Float64(0); var nb=Float64(0); var s=Float32(0); var s2=Float32(0)
    for i in range(len(m)):
        dot+=Float64(m[i])*Float64(vfin[i]); na+=Float64(m[i])*Float64(m[i]); nb+=Float64(vfin[i])*Float64(vfin[i])
        s+=m[i]; s2+=m[i]*m[i]
    var n=Float32(len(m)); var mean=s/n
    print("=== DENOISE TRAJECTORY PARITY vs Serenity ===")
    print("  latent_final cos =", dot/(sqrt(na)*sqrt(nb)))
    print("  Mojo mean=", mean, " std=", sqrt(s2/n - mean*mean), "  reference trainer mean=-0.26172 std=1.07031")
