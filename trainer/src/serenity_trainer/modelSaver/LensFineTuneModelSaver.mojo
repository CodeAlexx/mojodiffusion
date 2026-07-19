# LensFineTuneModelSaver.mojo — 1:1 port of Serenity
#   modules/modelSaver/LensFineTuneModelSaver.py  (pr-1510)
#     LensFineTuneModelSaver = make_fine_tune_model_saver(
#         ModelType.LENS, model_class=LensModel,
#         model_saver_class=LensModelSaver, embedding_saver_class=None)
# structurally mirrored on modelSaver/ZImageFineTuneModelSaver.mojo.
#
# make_fine_tune_model_saver (modules/modelSaver/GenericFineTuneModelSaver.py)
# builds a BaseModelSaver whose .save() delegates to LensModelSaver.save (the full
# transformer / pipeline saver). embedding_saver_class is None for Lens.
#
# In the Mojo port the FROZEN/fine-tuned transformer weights live in LensWeights;
# the save mechanics are in lens/LensModelSaver.mojo::lens_model_saver_save. This
# wrapper is the thin, named entry the trainer/cadence calls for a full fine-tune
# checkpoint — matching Serenity's LensFineTuneModelSaver factory product.

from std.gpu.host import DeviceContext
from serenitymojo.io.dtype import STDtype

from serenity_trainer.modelLoader.LensModelLoader import LensWeights
from serenity_trainer.modelSaver.lens.LensModelSaver import lens_model_saver_save


# LensFineTuneModelSaver.save → LensModelSaver.save (format dispatch).
def lens_fine_tune_model_saver_save(
    w: LensWeights,
    output_model_format: Int,
    output_model_destination: String,
    ctx: DeviceContext,
    dtype: STDtype = STDtype.BF16,
) raises:
    lens_model_saver_save(w, output_model_format, output_model_destination, ctx, dtype)
