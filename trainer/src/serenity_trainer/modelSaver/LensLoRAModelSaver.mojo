# LensLoRAModelSaver.mojo — 1:1 port of Serenity
#   modules/modelSaver/LensLoRAModelSaver.py  (pr-1510)
#     LensLoRAModelSaver = make_lora_model_saver(
#         ModelType.LENS, model_class=LensModel,
#         lora_saver_class=LensLoRASaver, embedding_saver_class=None)
# structurally mirrored on modelSaver/ZImageLoRAModelSaver.mojo.
#
# make_lora_model_saver (modules/modelSaver/GenericLoRAModelSaver.py) builds a
# BaseModelSaver subclass whose .save() delegates to LensLoRASaver.save (which is
# LoRASaverMixin._save). embedding_saver_class is None for Lens → no embedding
# state is saved.
#
# In the Mojo port the saver is data-oriented (no class hierarchy / no PyTorch
# nn.Module): the trained adapters live in a `LensLoraSet`, and the save mechanics
# are in lens/LensLoRASaver.mojo::save_lens_lora. This wrapper is the thin, named
# entry the trainer/cadence calls — matching Serenity's LensLoRAModelSaver
# factory product.
#
# ModelFormat: SAFETENSORS / LEGACY_SAFETENSORS both route to a single-file write
# (key_sets is None for Lens → byte-identical); INTERNAL writes to
# "<dest>/lora/lora.safetensors" (LoRASaverMixin.__save_internal).

from std.gpu.host import DeviceContext
from serenitymojo.io.dtype import STDtype

from serenity_trainer.model.LensModel import LensLoraSet
from serenity_trainer.modelSaver.lens.LensLoRASaver import save_lens_lora


# ModelFormat enum kinds (subset Serenity ModelFormat the LoRA saver honours).
comptime FMT_SAFETENSORS        = 0
comptime FMT_LEGACY_SAFETENSORS = 1
comptime FMT_INTERNAL           = 2


# LensLoRAModelSaver.save (the factory product's .save). Mirrors
# LoRASaverMixin._save format dispatch.
#   output_model_format : FMT_* (above)
#   output_model_destination : file path (SAFETENSORS) or dir (INTERNAL)
#   dtype : save-dtype override (BF16 by default, matching the trained storage)
def lens_lora_model_saver_save(
    set: LensLoraSet,
    output_model_format: Int,
    output_model_destination: String,
    ctx: DeviceContext,
    dtype: STDtype = STDtype.BF16,
) raises:
    if output_model_format == FMT_SAFETENSORS \
            or output_model_format == FMT_LEGACY_SAFETENSORS:
        # key_sets is None for Lens → legacy == raw == plain write.
        save_lens_lora(set, output_model_destination, ctx, dtype)
    elif output_model_format == FMT_INTERNAL:
        # __save_internal: "<dest>/lora/lora.safetensors", dtype=None (no cast →
        # write at the trained BF16 storage).
        var path = output_model_destination + String("/lora/lora.safetensors")
        save_lens_lora(set, path, ctx, STDtype.BF16)
    else:
        raise Error(
            String("lens_lora_model_saver_save: unsupported ModelFormat ")
            + String(output_model_format)
        )
