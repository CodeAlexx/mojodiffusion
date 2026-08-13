# qwen_model_compile_check.mojo - compile gate for QwenModel build-only surface.
#
# Compile only after the build lock is clear:
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#     pixi run mojo build -I . -I trainer/src \
#       trainer/smoke/qwen_model_compile_check.mojo \
#       -o /tmp/qwen_model_compile_check
#
# This is not a parity gate. It uses tiny synthetic tensors only to instantiate
# Qwen pack/unpack and VAE scale/unscale helpers.

from max.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenity_trainer.model.QwenModel import (
    QwenModel,
    QwenSchedulerShiftConfig,
    QWEN_PROMPT_MAX_LENGTH,
    QWEN_TOKENIZER_MAX_LENGTH,
    QWEN_DEFAULT_PROMPT_TEMPLATE_CROP_START,
    qwen_apply_default_prompt_template,
    qwen_text_encode_contract,
    calculate_timestep_shift,
    pack_latents,
    unpack_latents,
    scale_latents,
    unscale_latents,
)


def main() raises:
    var model = QwenModel()
    model.has_tokenizer = True
    model.has_noise_scheduler = True
    model.has_text_encoder = True
    model.has_vae = True
    model.has_transformer = True
    model.has_text_encoder_lora = True
    model.has_transformer_lora = True
    model.to(String("compile-device"))
    model.eval()

    var adapters = model.adapters()
    var pipe = model.create_pipeline()
    print("qwen adapters =", len(adapters), " pipeline has vae =", pipe.has_vae)
    print(
        "qwen prompt max =", QWEN_PROMPT_MAX_LENGTH,
        " tokenizer max =", QWEN_TOKENIZER_MAX_LENGTH,
        " crop =", QWEN_DEFAULT_PROMPT_TEMPLATE_CROP_START,
    )
    print(qwen_apply_default_prompt_template(String("a test prompt")).byte_length())

    var lengths = List[Int]()
    lengths.append(31)
    lengths.append(47)
    var contract = qwen_text_encode_contract(lengths^, 3584)
    print(
        "qwen text batch =", contract.batch_size,
        " out seq =", contract.output_seq_length,
        " hidden =", contract.hidden_size,
    )

    var shift_cfg = QwenSchedulerShiftConfig(
        256, 4096, Float32(0.5), Float32(1.15)
    )
    print("qwen shift =", calculate_timestep_shift(4, 4, shift_cfg))

    var ctx = DeviceContext()
    var vals = List[Float32]()
    for i in range(16 * 4 * 4):
        vals.append(Float32(i % 17) * 0.01)
    var latent_shape = List[Int]()
    latent_shape.append(1)
    latent_shape.append(16)
    latent_shape.append(1)
    latent_shape.append(4)
    latent_shape.append(4)
    var latent = Tensor.from_host(vals^, latent_shape^, STDtype.BF16, ctx)

    var packed = pack_latents(latent, ctx)
    var unpacked = unpack_latents(packed, 4, 4, ctx)

    var mean = List[Float32]()
    var std_values = List[Float32]()
    for i in range(16):
        mean.append(Float32(i) * 0.001)
        std_values.append(Float32(1.0) + Float32(i) * 0.01)
    var scaled = scale_latents(unpacked, mean^, std_values^, ctx)

    var mean2 = List[Float32]()
    var std_values2 = List[Float32]()
    for i in range(16):
        mean2.append(Float32(i) * 0.001)
        std_values2.append(Float32(1.0) + Float32(i) * 0.01)
    var unscaled = unscale_latents(scaled, mean2^, std_values2^, ctx)
    print("qwen latent dtype =", unscaled.dtype().name(), " rank =", len(unscaled.shape()))
    print("QWEN MODEL COMPILE-CHECK OK")
