# Ideogram4FineTuneModelSaver.mojo - full transformer save contract.
#
# ai-toolkit's full save unwraps the Ideogram4 transformer, dequantizes any QTensor,
# casts to the requested dtype, and writes a safetensors file with ideogram4 meta.

from serenity_trainer.util.enum.ModelType import MODEL_TYPE_IDEOGRAM_4, model_type_str


struct Ideogram4FineTuneSavePlan(Copyable, Movable, ImplicitlyCopyable):
    var output_model_destination: String
    var save_dtype: String
    var writes_safetensors: Bool
    var metadata_name: String
    var saves_transformer_only: Bool
    var dequantizes_qtensor_before_write: Bool
    var runtime_write_implemented: Bool

    def __init__(
        out self,
        var output_model_destination: String,
        var save_dtype: String,
    ):
        self.output_model_destination = output_model_destination^
        self.save_dtype = save_dtype^
        self.writes_safetensors = True
        self.metadata_name = String("ideogram4")
        self.saves_transformer_only = True
        self.dequantizes_qtensor_before_write = True
        self.runtime_write_implemented = False


struct Ideogram4FineTuneModelSaverContract(Copyable, Movable, ImplicitlyCopyable):
    var model_type: Int
    var factory_name: String
    var model_class_name: String
    var runtime_save_implemented: Bool

    def __init__(out self):
        self.model_type = MODEL_TYPE_IDEOGRAM_4
        self.factory_name = String("make_model_saver")
        self.model_class_name = String("Ideogram4Model")
        self.runtime_save_implemented = False


def ideogram4_fine_tune_model_saver_contract() -> Ideogram4FineTuneModelSaverContract:
    return Ideogram4FineTuneModelSaverContract()


struct Ideogram4FineTuneModelSaver(Movable):
    def __init__(out self):
        pass

    def validate_model_type(self, model_type: Int) raises:
        if model_type != MODEL_TYPE_IDEOGRAM_4:
            raise Error(String("Ideogram4FineTuneModelSaver: unsupported ModelType ") + model_type_str(model_type))

    def contract(self, model_type: Int) raises -> Ideogram4FineTuneModelSaverContract:
        self.validate_model_type(model_type)
        return ideogram4_fine_tune_model_saver_contract()

    def save_plan(
        self,
        model_type: Int,
        output_model_destination: String,
        save_dtype: String,
    ) raises -> Ideogram4FineTuneSavePlan:
        self.validate_model_type(model_type)
        return Ideogram4FineTuneSavePlan(output_model_destination, save_dtype)

    def runtime_save_supported(self) -> Bool:
        return False

