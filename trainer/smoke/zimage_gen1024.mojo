# Mojo Z-Image generate @1024 — ports mojodiffusion zimage_pipeline.mojo structure:
# denoise() loads the transformer INSIDE its scope and returns the latent, so the
# weights are FREED before main() loads the VAE (zimage_pipeline.mojo:189-237,240-254).
# Never transformer + VAE resident together → no OOM at 1024.
from std.math import sqrt
from std.memory import ArcPointer
from std.gpu.host import DeviceContext
from std.time import perf_counter_ns
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.ops.cast import cast_tensor
from serenity_trainer.modelLoader.ZImageModelLoader import ZImageWeights
from serenity_trainer.model.ZImageModel import build_zimage_inactive_lora_set
from serenity_trainer.modelSampler.FlowMatchEulerDiscreteScheduler import make_zimage_scheduler
from serenity_trainer.modelSampler.ZImageSampler import _predict_noise_cached
from serenity_trainer.model.ZImageDiT import prepare_zimage_infer_cache

comptime LH = 128
comptime LW = 128
comptime CAPLEN = 224
comptime STEPS = 30


# Loads the transformer INSIDE; weights freed on return (mojodiffusion denoise()).
def denoise(var lat0: Tensor, cap: Tensor, ctx: DeviceContext) raises -> Tensor:
    print("[denoise] loading transformer", LH, "x", LW)
    var t_load0 = perf_counter_ns()
    var w = ZImageWeights.load(String("/home/alex/.serenity/models/zimage_base/transformer"), ctx)
    var loras = build_zimage_inactive_lora_set(8)   # base model, no adapter GEMMs
    var sch = make_zimage_scheduler(STEPS, Float32(6.0))
    var t_cache0 = perf_counter_ns()
    var cache = prepare_zimage_infer_cache[LH, LW, CAPLEN](cap, w, ctx)
    var t_cache1 = perf_counter_ns()
    var lat = lat0^
    var step_total_ns = t_cache1 - t_cache1
    for i in range(STEPS):
        var t_step0 = perf_counter_ns()
        print("  sampling step", i + 1, "/", STEPS)
        var t = sch.timesteps[i]
        var t_model = (Float32(1000.0) - t) / Float32(1000.0)
        var np = _predict_noise_cached[LH, LW, CAPLEN](lat, t_model, cache, w, loras, ctx)
        lat = sch.step(np, lat, i, ctx)
        var t_step1 = perf_counter_ns()
        var step_ns = t_step1 - t_step0
        step_total_ns += step_ns
        print("    step_s =", Float64(step_ns) / Float64(1000000000))
    var load_s = Float32(Float64(t_cache0 - t_load0) / 1.0e9)
    var infer_cache_s = Float32(Float64(t_cache1 - t_cache0) / 1.0e9)
    var denoise_avg_step_s = Float32(Float64(step_total_ns) / (Float64(STEPS) * 1.0e9))
    var denoise_total_s = Float32(Float64(step_total_ns) / 1.0e9)
    print("speed: transformer_load_s =", load_s,
          " infer_cache_s =", infer_cache_s,
          " denoise_avg_step_s =", denoise_avg_step_s,
          " denoise_total_s =", denoise_total_s)
    return lat^     # w, loras destroyed here → transformer freed before VAE load


def main() raises:
    var ctx = DeviceContext()
    var st = ShardedSafeTensors.open(String("trainer/parity/zi_gen1024_ref.safetensors"))
    var lat0 = Tensor.from_view(st.tensor_view(String("latent0")), ctx)          # [1,16,128,128] f32 SAME as reference trainer
    var cap = cast_tensor(Tensor.from_view(st.tensor_view(String("cap")), ctx), STDtype.BF16, ctx)
    var latfin_ref = Tensor.from_view(st.tensor_view(String("latent_final")), ctx).to_host(ctx)

    var lat = denoise(lat0^, cap, ctx)     # transformer freed on return

    # latent parity vs Serenity
    var lf = lat.to_host(ctx)
    var dot=Float64(0); var na=Float64(0); var nb=Float64(0)
    for i in range(len(lf)):
        dot+=Float64(lf[i])*Float64(latfin_ref[i]); na+=Float64(lf[i])*Float64(lf[i]); nb+=Float64(latfin_ref[i])*Float64(latfin_ref[i])
    print("latent_final cos vs Serenity =", dot/(sqrt(na)*sqrt(nb)))

    var names = List[String]()
    var tensors = List[ArcPointer[Tensor]]()
    names.append(String("latent_final"))
    tensors.append(ArcPointer(lat^))
    save_safetensors(
        names^, tensors^,
        String("trainer/parity/zi_MOJO_1024_latent.safetensors"),
        ctx,
    )
    print("WROTE trainer/parity/zi_MOJO_1024_latent.safetensors")
