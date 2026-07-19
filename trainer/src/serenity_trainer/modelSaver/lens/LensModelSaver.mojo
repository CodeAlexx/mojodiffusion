# LensModelSaver.mojo — 1:1 port of Serenity
#   modules/modelSaver/lens/LensModelSaver.py  (pr-1510)
# structurally mirrored on modelSaver/zImage/ZImageModelSaver.mojo (the
# fine-tune / full-transformer saver).
#
# Serenity SOURCE (LensModelSaver.py):
#   def __save_safetensors(model, destination, dtype):
#       # Lens transformer uses diffusers-format keys; no key conversion needed.
#       state_dict = model.transformer.state_dict()
#       save_state_dict = self._convert_state_dict_dtype(state_dict, dtype)
#       self._convert_state_dict_to_contiguous(save_state_dict)
#       os.makedirs(Path(destination).parent.absolute(), exist_ok=True)
#       save_file(save_state_dict, destination, self._create_safetensors_header(...))
#   def __save_diffusers(model, destination, dtype):
#       model.materialize_text_encoder_for_save()
#       pipeline = model.create_pipeline(); pipeline.to("cpu"); ...
#       save_pipeline.save_pretrained(destination)
#   def __save_internal(model, destination): self.__save_diffusers(model, destination, None)
#   save(...): match output_model_format:
#       DIFFUSERS    → __save_diffusers
#       SAFETENSORS  → __save_safetensors
#       INTERNAL     → __save_internal
#
# The Mojo port is data-oriented: the FROZEN transformer weights live in
# LensWeights (modelLoader/LensModelLoader.mojo). __save_safetensors writes that
# store verbatim (diffusers keys, BF16 storage), which is the full fine-tune
# checkpoint. The DIFFUSERS pipeline export (re-saving tokenizer/scheduler/vae/
# text_encoder folders) has no Mojo nn.Module pipeline analogue and is NOT ported
# here — the safetensors path is the faithful subset (full transformer weights);
# the source pipeline export is preserved in this comment for fidelity review.
#
# Reuses ONLY serenitymojo {tensor, io, ops}.

from std.memory import ArcPointer
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.ops.cast import cast_tensor

from serenity_trainer.modelLoader.LensModelLoader import LensWeights


comptime TArc = ArcPointer[Tensor]

# ModelFormat enum kinds (subset Serenity ModelFormat the saver honours).
comptime FMT_DIFFUSERS   = 3
comptime FMT_SAFETENSORS = 0
comptime FMT_INTERNAL    = 2


# __save_safetensors: write the full transformer weight store to a single-file
# safetensors at `destination` (diffusers keys, no conversion). `dtype` casts each
# tensor to the save dtype (_convert_state_dict_dtype); BF16 is the trained store.
def save_lens_transformer_safetensors(
    w: LensWeights,
    destination: String,
    ctx: DeviceContext,
    dtype: STDtype = STDtype.BF16,
) raises:
    var names = List[String]()
    var tensors = List[TArc]()
    # Iterate the store in load order (state_dict iteration order is irrelevant to
    # safetensors, which is keyed by name). w.name_to_idx maps name → index.
    for ref kv in w.name_to_idx.items():
        var nm = kv.key
        var idx = kv.value
        names.append(nm)
        tensors.append(TArc(cast_tensor(w.weights[idx][], dtype, ctx)))
    save_safetensors(names, tensors, destination, ctx)


# LensModelSaver.save format dispatch (LensModelSaver.py:save).
def lens_model_saver_save(
    w: LensWeights,
    output_model_format: Int,
    output_model_destination: String,
    ctx: DeviceContext,
    dtype: STDtype = STDtype.BF16,
) raises:
    if output_model_format == FMT_SAFETENSORS:
        save_lens_transformer_safetensors(w, output_model_destination, ctx, dtype)
    elif output_model_format == FMT_DIFFUSERS or output_model_format == FMT_INTERNAL:
        # __save_diffusers / __save_internal re-export the full DiffusionPipeline
        # (transformer + tokenizer + scheduler + vae + text_encoder folders). No
        # Mojo nn.Module pipeline analogue exists; the faithful subset is the
        # transformer safetensors write. Callers that need the full diffusers
        # export must run it through Serenity's Python saver.
        raise Error(
            "lens_model_saver_save: DIFFUSERS/INTERNAL pipeline export is not "
            "ported (no Mojo DiffusionPipeline); use FMT_SAFETENSORS for the "
            "full transformer checkpoint."
        )
    else:
        raise Error(
            String("lens_model_saver_save: unsupported ModelFormat ")
            + String(output_model_format)
        )
