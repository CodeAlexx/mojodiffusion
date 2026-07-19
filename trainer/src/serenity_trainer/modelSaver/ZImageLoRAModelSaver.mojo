# ZImageLoRAModelSaver.mojo — 1:1 port of Serenity
#   modules/modelSaver/ZImageLoRAModelSaver.py
#     ZImageLoRAModelSaver = make_lora_model_saver(
#         ModelType.Z_IMAGE, model_class=ZImageModel,
#         lora_saver_class=ZImageLoRASaver, embedding_saver_class=None)
#
# make_lora_model_saver (modules/modelSaver/GenericLoRAModelSaver.py) builds a
# BaseModelSaver subclass whose .save() delegates to the lora_saver_class's
# ZImageLoRASaver.save (which is LoRASaverMixin._save). embedding_saver_class is
# None for Z-Image → no embedding state is saved.
#
# In the Mojo port the saver is data-oriented (no class hierarchy / no PyTorch
# nn.Module): the trained adapters live in a `ZImageLoraSet`, and the actual save
# mechanics are in zImage/ZImageLoRASaver.mojo::save_zimage_lora. This wrapper is
# the thin, named entry point the trainer/orchestrator calls — matching the role
# of Serenity's ZImageLoRAModelSaver factory product.
#
# ModelFormat (Serenity modules/util/enum/ModelFormat.py): SAFETENSORS /
# LEGACY_SAFETENSORS both route to a single-file write (key_sets is None for
# Z-Image, so they are byte-identical); INTERNAL writes to
# "<dest>/lora/lora.safetensors" (LoRASaverMixin.__save_internal :70-77).

from std.gpu.host import DeviceContext
from serenitymojo.io.dtype import STDtype

from serenity_trainer.model.ZImageModel import ZImageLoraSet
from serenity_trainer.modelSaver.zImage.ZImageLoRASaver import save_zimage_lora


# ModelFormat enum kinds (subset Serenity ModelFormat the LoRA saver honours).
comptime FMT_SAFETENSORS = 0
comptime FMT_LEGACY_SAFETENSORS = 1
comptime FMT_INTERNAL = 2


# ZImageLoRAModelSaver.save (the factory product's .save). Mirrors
# LoRASaverMixin._save (LoRASaverMixin.py:80-96) format dispatch.
#   output_model_format : FMT_* (above)
#   output_model_destination : file path (SAFETENSORS) or dir (INTERNAL)
#   dtype : save-dtype override (BF16 by default, matching the trained storage)
def zimage_lora_model_saver_save(
    set: ZImageLoraSet,
    output_model_format: Int,
    output_model_destination: String,
    ctx: DeviceContext,
    dtype: STDtype = STDtype.BF16,
) raises:
    if output_model_format == FMT_SAFETENSORS \
            or output_model_format == FMT_LEGACY_SAFETENSORS:
        # key_sets is None for Z-Image → legacy == raw == plain write.
        save_zimage_lora(set, output_model_destination, ctx, dtype)
    elif output_model_format == FMT_INTERNAL:
        # __save_internal: "<dest>/lora/lora.safetensors", dtype=None (no cast →
        # write at the trained BF16 storage). (LoRASaverMixin.py:70-77)
        var path = output_model_destination + String("/lora/lora.safetensors")
        save_zimage_lora(set, path, ctx, STDtype.BF16)
    else:
        raise Error(
            String("zimage_lora_model_saver_save: unsupported ModelFormat ")
            + String(output_model_format)
        )
