# Full Z-Image generate: denoise (base, no LoRA) → unscale → VAE decode → PNG.
from max.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.random import randn
from serenitymojo.image.png import save_png, ValueRange
from serenity_trainer.modelLoader.ZImageModelLoader import ZImageWeights
from serenity_trainer.model.ZImageModel import build_zimage_lora_set
from serenity_trainer.modelSampler.FlowMatchEulerDiscreteScheduler import make_zimage_scheduler
from serenity_trainer.modelSampler.ZImageSampler import _predict_noise
from serenity_trainer.model.ZImageVAE import ZImageDecoder, decode_latent, unscale_latents

comptime LH = 32     # latent → 256x256 image
comptime LW = 32
comptime CAPLEN = 224
comptime STEPS = 12

def main() raises:
    var ctx = DeviceContext()
    # real caption features (from Serenity's cached sample)
    var st = ShardedSafeTensors.open(String("trainer/parity/zi_realclean.safetensors"))
    var cap = cast_tensor(Tensor.from_view(st.tensor_view(String("cap")), ctx), STDtype.BF16, ctx)  # [224,2560]
    print("loading transformer ...")
    var w = ZImageWeights.load(String("/home/alex/.serenity/models/zimage_base/transformer"), ctx)
    var loras = build_zimage_lora_set(8, Float32(8.0), ctx)   # B=0 → base model
    print("loading VAE ...")
    var dec = ZImageDecoder[LH, LW].load(String("/home/alex/.serenity/models/zimage_base/vae"), ctx)

    var lat = randn([1, 16, LH, LW], UInt64(777), STDtype.F32, ctx)
    var sch = make_zimage_scheduler(STEPS, Float32(6.0))
    print("denoising", STEPS, "steps ...")
    for i in range(STEPS):
        var t = sch.timesteps[i]
        var t_model = (Float32(1000.0) - t) / Float32(1000.0)
        var np = _predict_noise[LH, LW, CAPLEN](lat, t_model, cap, w, loras, ctx)
        lat = sch.step(np, lat, i, ctx)

    print("decoding ...")
    var unscaled = unscale_latents(cast_tensor(lat, STDtype.BF16, ctx), ctx)
    var img = decode_latent[LH, LW](dec, unscaled, ctx)        # [1,3,256,256]
    print("image shape =", img.shape()[0], img.shape()[1], img.shape()[2], img.shape()[3])
    save_png(img, String("/tmp/zimage_sample.png"), ctx, ValueRange.SIGNED)
    print("WROTE /tmp/zimage_sample.png")
