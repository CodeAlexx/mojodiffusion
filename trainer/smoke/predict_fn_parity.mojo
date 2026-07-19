# Per-function predict parity vs Serenity (parity/predict_fn_ref.json).
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenity_trainer.model.ZImageVAE import scale_latents
from serenity_trainer.modelSetup.BaseZImageSetup import sigma_from_timestep, calculate_timestep_shift
from serenity_trainer.modelSetup.mixin.ModelSetupFlowMatchingMixin import FlowMatchSigmaTable

def chk(name: String, got: Float64, want: Float64) raises:
    var d = got - want
    var ad = d if d >= 0.0 else -d
    var ok = ad < 1e-4
    print(("PASS " if ok else "FAIL ") + name, " got=", got, " reference trainer=", want, " |d|=", ad)
    if not ok: raise Error("PARITY FAIL: " + name)

def main() raises:
    var ctx = DeviceContext()
    # 1) _add_noise_discrete sigma table  sigma[t]=(t+1)/N
    var tab = FlowMatchSigmaTable(1000)
    chk("sigma[499]", Float64(tab.sigma[499]), 0.5)
    chk("sigma[250]", Float64(tab.sigma[250]), 0.251)
    chk("sigma[0]",   Float64(tab.sigma[0]),   0.001)
    # sigma_from_timestep (BaseZImageSetup)
    chk("sigma_from_timestep(499)", Float64(sigma_from_timestep(499)), 0.5)
    chk("sigma_from_timestep(250)", Float64(sigma_from_timestep(250)), 0.251)
    # 2) calculate_timestep_shift(h,w)  56x72 -> latent_h=72,latent_w=56
    chk("calc_shift(72,56)", Float64(calculate_timestep_shift(72, 56)), 1.872532)
    chk("calc_shift(64,64)", Float64(calculate_timestep_shift(64, 64)), 1.877611)
    # 3) scale_latents(1.0) = (1-0.1159)*0.3611
    var one = Tensor.from_host([Float32(1.0)], [1], STDtype.BF16, ctx)
    var sl = scale_latents(one, ctx).to_host(ctx)
    chk("scale_latents(1.0)", Float64(sl[0]), 0.318359375)
    print("=== ALL PREDICT-FN PARITY PASS vs Serenity ===")
