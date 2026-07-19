# Save a Z-Image LoRA set; gate verifies keys match a real Serenity LoRA.
from std.gpu.host import DeviceContext
from serenity_trainer.model.ZImageModel import build_zimage_lora_set
from serenity_trainer.modelSaver.zImage.ZImageLoRASaver import save_zimage_lora

def main() raises:
    var ctx = DeviceContext()
    var lset = build_zimage_lora_set(16, Float32(16.0), ctx)   # rank 16 like the real preset
    save_zimage_lora(lset, String("/tmp/zi_mojo_lora.safetensors"), ctx)
    print("WROTE /tmp/zi_mojo_lora.safetensors")
