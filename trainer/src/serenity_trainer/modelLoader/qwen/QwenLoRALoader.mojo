# 1:1 surface port of Serenity modules/modelLoader/qwen/QwenLoRALoader.py
#
# QwenLoRALoader._get_convert_key_sets returns None in Serenity, so LoRA files
# are loaded with raw diffusers/PEFT keys. This build-only Mojo port reads
# safetensors into a state-dict carrier and marks the model handle as having
# loaded LoRA data. CKPT loading remains outside Mojo product runtime.

from std.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.dtype import STDtype
from serenitymojo.tensor import Tensor

from serenity_trainer.modelLoader.qwen.QwenModelLoader import QwenModelHandle, QwenModelNames
from serenity_trainer.modelSetup.qwenLoraTargets import (
    QwenLoraTargetSpecs,
    qwen_lora_down_key,
    qwen_lora_up_key,
    qwen_lora_alpha_key,
)


comptime TArc = ArcPointer[Tensor]
comptime QWEN_LORA_NONE = 0
comptime QWEN_LORA_SAFETENSORS = 1
comptime QWEN_LORA_INTERNAL = 2


struct QwenLoraStateDict(Movable):
    var names: List[String]
    var tensors: List[TArc]
    var route: Int

    def __init__(out self, var names: List[String], var tensors: List[TArc], route: Int):
        self.names = names^
        self.tensors = tensors^
        self.route = route

    @staticmethod
    def empty() -> QwenLoraStateDict:
        var names = List[String]()
        var tensors = List[TArc]()
        return QwenLoraStateDict(names^, tensors^, QWEN_LORA_NONE)


struct QwenLoraReload(Movable):
    var a: List[TArc]
    var b: List[TArc]
    var alpha: List[Float32]
    var rank: Int

    def __init__(out self, var a: List[TArc], var b: List[TArc], var alpha: List[Float32], rank: Int):
        self.a = a^
        self.b = b^
        self.alpha = alpha^
        self.rank = rank


def qwen_lora_has_convert_key_sets() -> Bool:
    return False


def load_qwen_lora_safetensors(path: String, ctx: DeviceContext) raises -> QwenLoraStateDict:
    var sharded = ShardedSafeTensors.open(path)
    var names = List[String]()
    var tensors = List[TArc]()
    for ref nm in sharded.names():
        var tv = sharded.tensor_view(nm)
        var t = Tensor.from_view(tv, ctx)
        names.append(nm.copy())
        tensors.append(TArc(t^))
    return QwenLoraStateDict(names^, tensors^, QWEN_LORA_SAFETENSORS)


def load_qwen_lora_internal(path: String, ctx: DeviceContext) raises -> QwenLoraStateDict:
    var lora_path = path + String("/lora/lora.safetensors")
    var state = load_qwen_lora_safetensors(lora_path, ctx)
    state.route = QWEN_LORA_INTERNAL
    return state^


def load_qwen_lora_targets(
    path: String,
    targets: QwenLoraTargetSpecs,
    ctx: DeviceContext,
    expected_dtype: STDtype = STDtype.BF16,
) raises -> QwenLoraReload:
    var sharded = ShardedSafeTensors.open(path)
    var have = Dict[String, Int]()
    for ref nm in sharded.names():
        have[nm] = 1

    var a = List[TArc]()
    var b = List[TArc]()
    var alpha = List[Float32]()
    var rank = -1
    for i in range(targets.len()):
        var prefix = targets.prefixes[i]
        var in_features = targets.in_features[i]
        var out_features = targets.out_features[i]
        var ak = qwen_lora_down_key(prefix)
        var bk = qwen_lora_up_key(prefix)
        if not (ak in have):
            raise Error(String("load_qwen_lora_targets: missing ") + ak)
        if not (bk in have):
            raise Error(String("load_qwen_lora_targets: missing ") + bk)

        var at = Tensor.from_view(sharded.tensor_view(ak), ctx)
        var bt = Tensor.from_view(sharded.tensor_view(bk), ctx)
        _expect_lora_tensor(ak, at, expected_dtype)
        _expect_lora_tensor(bk, bt, expected_dtype)

        var ash = at.shape()
        var bsh = bt.shape()
        _expect_int(ak + String(".rank"), len(ash), 2)
        _expect_int(bk + String(".rank"), len(bsh), 2)
        _expect_int(ak + String(".in"), ash[1], in_features)
        _expect_int(bk + String(".out"), bsh[0], out_features)
        if rank < 0:
            rank = ash[0]
        _expect_int(ak + String(".rank_dim"), ash[0], rank)
        _expect_int(bk + String(".rank_dim"), bsh[1], rank)

        var al = Float32(rank)
        var ah = qwen_lora_alpha_key(prefix)
        if ah in have:
            var alt = Tensor.from_view(sharded.tensor_view(ah), ctx)
            _expect_lora_tensor(ah, alt, expected_dtype)
            var alsh = alt.shape()
            if len(alsh) != 0:
                if len(alsh) != 1 or alsh[0] != 1:
                    raise Error(ah + String(": alpha must be scalar or [1]"))
            var host = alt.to_host(ctx)
            if len(host) > 0:
                al = host[0]

        a.append(TArc(at^))
        b.append(TArc(bt^))
        alpha.append(al)

    return QwenLoraReload(a^, b^, alpha^, rank)


def _expect_lora_tensor(name: String, tensor: Tensor, expected_dtype: STDtype) raises:
    if tensor.dtype() != expected_dtype:
        raise Error(
            name + String(": dtype got ") + tensor.dtype().name()
            + String(", expected ") + expected_dtype.name()
        )


def _expect_int(name: String, got: Int, expected: Int) raises:
    if got != expected:
        raise Error(
            name + String(": got ") + String(got)
            + String(", expected ") + String(expected)
        )


struct QwenLoRALoader(Movable):
    def __init__(out self):
        pass

    def _get_convert_key_sets(self, model: QwenModelHandle) -> Bool:
        _ = model
        return qwen_lora_has_convert_key_sets()

    def load(
        self,
        mut model: QwenModelHandle,
        model_names: QwenModelNames,
        ctx: DeviceContext,
    ) raises -> QwenLoraStateDict:
        if model_names.lora == String():
            return QwenLoraStateDict.empty()

        var state = load_qwen_lora_safetensors(model_names.lora, ctx)
        model.lora_loaded = True
        return state^

    def load_internal(
        self,
        mut model: QwenModelHandle,
        lora_dir: String,
        ctx: DeviceContext,
    ) raises -> QwenLoraStateDict:
        var state = load_qwen_lora_internal(lora_dir, ctx)
        model.lora_loaded = True
        return state^
