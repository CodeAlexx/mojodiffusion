# weights.mojo — base model weight loader.
#
# Loads frozen base-model weights from a safetensors file or sharded directory
# using serenitymojo's verified io layer. The dtype of each stored tensor is
# PRESERVED (BF16 stays BF16) via Tensor.from_view — matching Serenity's
# practice of loading the base model in its checkpoint dtype and freezing it
# (LoRA only trains the adapters; cf. LoRAModuleWrapper.requires_grad_, which is
# applied only to the lora_modules, never the orig base weights).
#
# Reuses ONLY serenitymojo {io.sharded, tensor}. No Python, no MGDS.
#
# ShardedSafeTensors.open(dir) transparently handles three cases:
#   - a direct *.safetensors file path,
#   - a directory with a *.index.json shard map (multi-shard checkpoints),
#   - a directory with a single *.safetensors fallback.
# So this one entry point covers the Serenity base-model load surface.

from std.gpu.host import DeviceContext
from std.memory import ArcPointer
from serenitymojo.tensor import Tensor
from serenitymojo.io.sharded import ShardedSafeTensors

comptime TArc = ArcPointer[Tensor]


# A loaded, named base-weight collection. Tensors are MOVE-ONLY, so each is
# boxed in an ArcPointer and addressed by name (the safetensors key, e.g.
# "transformer.layers.0.attn.to_q.weight").
struct BaseWeights(Movable):
    var names: List[String]
    var tensors: List[TArc]

    def __init__(out self, var names: List[String], var tensors: List[TArc]):
        self.names = names^
        self.tensors = tensors^

    def __len__(self) -> Int:
        return len(self.names)

    # Index of `name`, or -1 if absent.
    def _index_of(self, name: String) -> Int:
        for i in range(len(self.names)):
            if self.names[i] == name:
                return i
        return -1

    def has(self, name: String) -> Bool:
        return self._index_of(name) != -1

    # Borrow the boxed tensor for `name` (raises if missing). Clone via
    # `get(name)[].clone(ctx)` when an owned copy is needed.
    def get(self, name: String) raises -> TArc:
        var idx = self._index_of(name)
        if idx == -1:
            raise Error(String("BaseWeights: tensor '") + name + "' not found")
        return self.tensors[idx]


# Load every tensor from `path` (file or sharded dir), dtype-preserving.
# H2D copy per tensor through Tensor.from_view (the source mmap is not aliased).
def load_base_weights(path: String, ctx: DeviceContext) raises -> BaseWeights:
    var src = ShardedSafeTensors.open(path)
    var keys = src.names()
    var out_names = List[String]()
    var out_tensors = List[TArc]()
    for ref nm in keys:
        var tv = src.tensor_view(nm)
        var t = Tensor.from_view(tv, ctx)
        out_names.append(nm)
        out_tensors.append(TArc(t^))
    return BaseWeights(out_names^, out_tensors^)


# Load only the tensors whose names are listed in `want` (e.g. just the Linear
# weights a LoRA wraps), dtype-preserving. Missing names raise.
def load_named_weights(
    path: String, want: List[String], ctx: DeviceContext
) raises -> BaseWeights:
    var src = ShardedSafeTensors.open(path)
    var out_names = List[String]()
    var out_tensors = List[TArc]()
    for i in range(len(want)):
        var nm = want[i]
        var tv = src.tensor_view(nm)
        var t = Tensor.from_view(tv, ctx)
        out_names.append(nm)
        out_tensors.append(TArc(t^))
    return BaseWeights(out_names^, out_tensors^)
