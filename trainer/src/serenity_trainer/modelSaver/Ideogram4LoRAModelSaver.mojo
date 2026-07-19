# Ideogram4LoRAModelSaver.mojo - Ideogram4 LoRA save contract.
#
# ai-toolkit conversion:
#   save: key.replace("transformer.", "diffusion_model.")
#   load: key.replace("diffusion_model.", "transformer.")

from serenity_trainer.modelSetup.ideogram4LoraTargets import (
    ideogram4_block_lora_save_prefixes,
    ideogram4_convert_lora_key_before_save,
    ideogram4_full_lora_save_prefixes,
)
from std.gpu.host import DeviceContext
from serenity_trainer.module.LoRAModule import LoraAdapter
from serenity_trainer.model.Ideogram4LoRABlock import Ideogram4LoraSet
from serenity_trainer.modelSaver.GenericLoRAModelSaver import save_lora, save_lora_one
from serenity_trainer.trainer.Ideogram4TrainCore import (
    IDEOGRAM4_FINAL_LINEAR_PREFIX,
)
from serenity_trainer.util.enum.ModelType import MODEL_TYPE_IDEOGRAM_4, model_type_str


comptime IDEOGRAM4_FMT_DIFFUSERS = 0
comptime IDEOGRAM4_FMT_CKPT = 1
comptime IDEOGRAM4_FMT_SAFETENSORS = 2
comptime IDEOGRAM4_FMT_LEGACY_SAFETENSORS = 3
comptime IDEOGRAM4_FMT_COMFY_LORA = 4
comptime IDEOGRAM4_FMT_INTERNAL = 5


struct Ideogram4LoraStateDictContract(Movable):
    var includes_transformer_lora: Bool
    var includes_preloaded_lora_state_dict: Bool
    var includes_text_encoder_lora: Bool
    var has_convert_key_sets: Bool
    var key_prefix_before_save: String
    var key_prefix_after_save: String
    var lora_down_suffix: String
    var lora_up_suffix: String
    var alpha_suffix: String
    var sample_target_prefixes: List[String]
    var runtime_write_implemented: Bool

    def __init__(out self):
        self.includes_transformer_lora = True
        self.includes_preloaded_lora_state_dict = True
        self.includes_text_encoder_lora = False
        self.has_convert_key_sets = True
        self.key_prefix_before_save = String("transformer.")
        self.key_prefix_after_save = String("diffusion_model.")
        self.lora_down_suffix = String(".lora_A.weight")
        self.lora_up_suffix = String(".lora_B.weight")
        self.alpha_suffix = String(".alpha")
        self.sample_target_prefixes = ideogram4_full_lora_save_prefixes()
        self.runtime_write_implemented = True


struct Ideogram4LoraSavePlan(Movable):
    var output_model_format: Int
    var output_model_destination: String
    var route_name: String
    var target_key_namespace: String
    var writes_safetensors: Bool
    var internal_destination: String
    var converts_transformer_prefix_to_diffusion_model: Bool
    var state_dict_contract: Ideogram4LoraStateDictContract
    var runtime_write_implemented_for_contract: Bool

    def __init__(
        out self,
        output_model_format: Int,
        var output_model_destination: String,
        var route_name: String,
        var target_key_namespace: String,
        var internal_destination: String,
        writes_safetensors: Bool,
    ):
        self.output_model_format = output_model_format
        self.output_model_destination = output_model_destination^
        self.route_name = route_name^
        self.target_key_namespace = target_key_namespace^
        self.writes_safetensors = writes_safetensors
        self.internal_destination = internal_destination^
        self.converts_transformer_prefix_to_diffusion_model = True
        self.state_dict_contract = Ideogram4LoraStateDictContract()
        self.runtime_write_implemented_for_contract = True


def ideogram4_lora_state_dict_contract() -> Ideogram4LoraStateDictContract:
    return Ideogram4LoraStateDictContract()


def ideogram4_lora_save_plan(
    output_model_format: Int,
    output_model_destination: String,
) raises -> Ideogram4LoraSavePlan:
    if output_model_format == IDEOGRAM4_FMT_SAFETENSORS:
        return Ideogram4LoraSavePlan(
            output_model_format,
            output_model_destination.copy(),
            String("safetensors"),
            String("diffusion_model.<native module>.{lora_A.weight,lora_B.weight,alpha}"),
            String(),
            True,
        )
    if output_model_format == IDEOGRAM4_FMT_LEGACY_SAFETENSORS:
        return Ideogram4LoraSavePlan(
            output_model_format,
            output_model_destination.copy(),
            String("legacy_safetensors"),
            String("diffusion_model.<native module>.{lora_A.weight,lora_B.weight,alpha}"),
            String(),
            True,
        )
    if output_model_format == IDEOGRAM4_FMT_INTERNAL:
        return Ideogram4LoraSavePlan(
            output_model_format,
            output_model_destination.copy(),
            String("internal_lora"),
            String("diffusion_model.<native module>.{lora_A.weight,lora_B.weight,alpha}"),
            output_model_destination + String("/lora/lora.safetensors"),
            True,
        )
    if output_model_format == IDEOGRAM4_FMT_DIFFUSERS:
        raise Error("Ideogram4LoRAModelSaver: DIFFUSERS LoRA output is not implemented")
    raise Error("Ideogram4LoRAModelSaver: unsupported ModelFormat")


struct Ideogram4LoRAModelSaverContract(Movable):
    var model_type: Int
    var factory_name: String
    var model_class_name: String
    var lora_saver_class_name: String
    var embedding_saver_class_name: String
    var has_embedding_saver: Bool
    var converts_keys_before_save: Bool
    var runtime_save_implemented: Bool

    def __init__(out self):
        self.model_type = MODEL_TYPE_IDEOGRAM_4
        self.factory_name = String("make_lora_model_saver")
        self.model_class_name = String("Ideogram4Model")
        self.lora_saver_class_name = String("Ideogram4LoRASaver")
        self.embedding_saver_class_name = String("None")
        self.has_embedding_saver = False
        self.converts_keys_before_save = True
        self.runtime_save_implemented = True


def ideogram4_lora_model_saver_contract() -> Ideogram4LoRAModelSaverContract:
    return Ideogram4LoRAModelSaverContract()


struct Ideogram4LoRAModelSaver(Movable):
    def __init__(out self):
        pass

    def validate_model_type(self, model_type: Int) raises:
        if model_type != MODEL_TYPE_IDEOGRAM_4:
            raise Error(String("Ideogram4LoRAModelSaver: unsupported ModelType ") + model_type_str(model_type))

    def contract(self, model_type: Int) raises -> Ideogram4LoRAModelSaverContract:
        self.validate_model_type(model_type)
        return ideogram4_lora_model_saver_contract()

    def save_plan(
        self,
        model_type: Int,
        output_model_format: Int,
        output_model_destination: String,
    ) raises -> Ideogram4LoraSavePlan:
        self.validate_model_type(model_type)
        return ideogram4_lora_save_plan(output_model_format, output_model_destination)

    def convert_key_before_save(self, key: String) -> String:
        return ideogram4_convert_lora_key_before_save(key)

    def runtime_save_supported(self) -> Bool:
        return True

    def save_final_linear_lora(
        self,
        var adapter: LoraAdapter,
        output_model_destination: String,
        ctx: DeviceContext,
    ) raises:
        save_lora_one(
            self.convert_key_before_save(String(IDEOGRAM4_FINAL_LINEAR_PREFIX)),
            adapter^,
            output_model_destination,
            ctx,
        )

    def save_block_stack_lora(
        self,
        loras: Ideogram4LoraSet,
        output_model_destination: String,
        ctx: DeviceContext,
    ) raises:
        var src_prefixes = ideogram4_block_lora_save_prefixes(loras.n_layers)
        if len(src_prefixes) != len(loras.ad):
            raise Error("Ideogram4LoRAModelSaver.save_block_stack_lora: prefix/adapter count mismatch")
        var prefixes = List[String]()
        for i in range(len(src_prefixes)):
            prefixes.append(self.convert_key_before_save(src_prefixes[i]))
        save_lora(prefixes^, loras.ad.copy(), output_model_destination, ctx)
