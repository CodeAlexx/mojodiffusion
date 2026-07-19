# Z-Image setup compile+run: predict path (synthetic weights) via ZImageLoRASetup.
from std.gpu.host import DeviceContext
from serenity_trainer.modelSetup.ZImageLoRASetup import zimage_lora_predict_smoke
from serenity_trainer.modelSetup.BaseZImageSetup import sigma_from_timestep, model_t_from_timestep

def main() raises:
    var ctx = DeviceContext()
    # sigma/timestep mapping sanity (1:1 BaseZImageSetup): model_t = (1000-t)/1000
    print("model_t(t=250) =", model_t_from_timestep(250), " sigma(t=250) =", sigma_from_timestep(250))
    var loss = zimage_lora_predict_smoke[16](8, ctx)   # tiny S=16, rank=8
    print("zimage predict smoke loss =", loss)
    print("ZIMAGE SETUP SMOKE OK")
