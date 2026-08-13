# ideogram4_encoders_smoke.mojo — end-to-end gate for the two Ideogram-4
# training-data encoders, through the serenity_trainer wrapper surface.
#
# Run (GPU free):
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#     pixi run mojo run -I . -I trainer/src \
#       trainer/smoke/ideogram4_encoders_smoke.mojo
#
# Gates BOTH wrappers vs the torch oracles:
#   - Ideogram4VaeEncoder.encode  vs ideogram4_fx_vae_encode.safetensors (latents)
#   - Ideogram4TextEncoder.encode vs ideogram4_fx_qwen.safetensors (chunk7 13-tap)
from max.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.tensor_algebra import slice
from serenity_trainer.model.Ideogram4VAE import Ideogram4VaeEncoder
from serenity_trainer.model.Ideogram4TextEncoder import (
    ideogram4_load_text_encoder_default,
    ideogram4_encode_text,
)

comptime VAE_FX = "/home/alex/mojodiffusion/serenitymojo/models/vae/parity/ideogram4_fx_vae_encode.safetensors"
comptime QWEN_FX = "/home/alex/mojodiffusion/serenitymojo/models/dit/parity/ideogram4_fx_qwen.safetensors"


def main() raises:
    var ctx = DeviceContext()

    # ── VAE encoder (image 256 -> packed normalized latent [1,128,16,16]) ──
    var venc = Ideogram4VaeEncoder[32, 32].load_default(ctx)
    var vfx = ShardedSafeTensors.open(VAE_FX)
    var img = cast_tensor(Tensor.from_view(vfx.tensor_view("image"), ctx), STDtype.BF16, ctx)
    var lat = venc.encode(img, ctx)
    var lat_exp = Tensor.from_view(vfx.tensor_view("latents"), ctx).to_host(ctx)
    print("VAE encoder latents parity:", ParityHarness(0.999).compare(lat, lat_exp, ctx))

    # ── Text encoder (Qwen3-VL 13-tap [1,L,53248]) ──
    var tenc = ideogram4_load_text_encoder_default(ctx)
    var ids = [151644, 872, 198, 64, 2518, 23739, 389, 264, 4158, 1965, 151645, 198, 151644, 77091, 198, 151643]
    var feats16 = ideogram4_encode_text(tenc, ids, ctx)
    var feats = slice(feats16, 1, 0, 15, ctx)
    var qfx = ShardedSafeTensors.open(QWEN_FX)
    var feats_exp = Tensor.from_view(qfx.tensor_view("chunk7.llm_features"), ctx).to_host(ctx)
    print("Text encoder 13-tap parity:", ParityHarness(0.999).compare(feats, feats_exp, ctx))
