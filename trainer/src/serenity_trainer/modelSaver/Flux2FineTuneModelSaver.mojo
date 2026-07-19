# 1:1 surface port of Serenity
#   modules/modelSaver/Flux2FineTuneModelSaver.py
#
# Serenity:
#   Flux2FineTuneModelSaver = make_fine_tune_model_saver(
#       ModelType.FLUX_2,
#       model_class=Flux2Model,
#       model_saver_class=Flux2ModelSaver,
#       embedding_saver_class=None,
#   )
#
# Build-only wrapper contract mirror. The full-model leaf saver exposes route
# plans only; no Mojo runtime save or numeric parity claim is made here.

from serenity_trainer.modelSaver.flux2.Flux2ModelSaver import (
    Flux2ModelSavePlan,
    Flux2ModelSaver,
)
from serenity_trainer.util.enum.ModelType import MODEL_TYPE_FLUX_2, model_type_str


comptime FLUX2_FINE_TUNE_MODEL_SAVER_MODEL_TYPE = MODEL_TYPE_FLUX_2
comptime FLUX2_FINE_TUNE_MODEL_SAVER_FACTORY = "make_fine_tune_model_saver"
comptime FLUX2_FINE_TUNE_MODEL_SAVER_MODEL_CLASS = "Flux2Model"
comptime FLUX2_FINE_TUNE_MODEL_SAVER_MODEL_SAVER_CLASS = "Flux2ModelSaver"
comptime FLUX2_FINE_TUNE_MODEL_SAVER_EMBEDDING_SAVER_CLASS = "None"


struct Flux2FineTuneModelSaverContract(Movable):
    var model_type: Int
    var factory_name: String
    var model_class_name: String
    var model_saver_class_name: String
    var embedding_saver_class_name: String
    var has_embedding_saver: Bool
    var leaf_model_saver_invoked: Bool
    var internal_save_data_after_leaf_save: Bool
    var runtime_save_implemented: Bool

    def __init__(out self):
        self.model_type = FLUX2_FINE_TUNE_MODEL_SAVER_MODEL_TYPE
        self.factory_name = String(FLUX2_FINE_TUNE_MODEL_SAVER_FACTORY)
        self.model_class_name = String(FLUX2_FINE_TUNE_MODEL_SAVER_MODEL_CLASS)
        self.model_saver_class_name = String(FLUX2_FINE_TUNE_MODEL_SAVER_MODEL_SAVER_CLASS)
        self.embedding_saver_class_name = String(FLUX2_FINE_TUNE_MODEL_SAVER_EMBEDDING_SAVER_CLASS)
        self.has_embedding_saver = False
        self.leaf_model_saver_invoked = True
        self.internal_save_data_after_leaf_save = True
        self.runtime_save_implemented = False


def flux2_fine_tune_model_saver_contract() -> Flux2FineTuneModelSaverContract:
    return Flux2FineTuneModelSaverContract()


struct Flux2FineTuneModelSaver(Movable):
    def __init__(out self):
        pass

    def validate_model_type(self, model_type: Int) raises:
        if model_type != MODEL_TYPE_FLUX_2:
            raise Error(String("Flux2FineTuneModelSaver: unsupported ModelType ") + model_type_str(model_type))

    def contract(self, model_type: Int) raises -> Flux2FineTuneModelSaverContract:
        self.validate_model_type(model_type)
        return flux2_fine_tune_model_saver_contract()

    def save_plan(
        self,
        model_type: Int,
        output_model_format: Int,
        output_model_destination: String,
        dtype_override: String,
    ) raises -> Flux2ModelSavePlan:
        self.validate_model_type(model_type)
        var saver = Flux2ModelSaver()
        return saver.save_plan(output_model_format, output_model_destination, dtype_override)

    def runtime_save_supported(self) -> Bool:
        return False
