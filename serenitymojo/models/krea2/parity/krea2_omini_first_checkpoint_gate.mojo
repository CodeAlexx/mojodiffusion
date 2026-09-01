# CUDA-oracle numeric gate plus checkpoint structure gate for the corrected
# Omini condition-only `first` LoRA.
#
# Numeric run:
#   ${KREA2_ORACLE_PYTHON} \
#       scripts/krea2_omini_first_torch_oracle.py
#   pixi run mojo run -I . \
#       serenitymojo/models/krea2/parity/krea2_omini_first_checkpoint_gate.mojo \
#       serenitymojo/models/krea2/parity/krea2_omini_first_oracle.safetensors
#
# Checkpoint run (second optional argument):
#   ... krea2_omini_first_oracle.safetensors <trained_lora.safetensors>

from max.gpu.host import DeviceContext
from std.sys import argv
from std.collections import List
from std.memory import ArcPointer
from serenitymojo.tensor import Tensor
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.parity import ParityHarness
from serenitymojo.models.klein.lora_block import (
    LoraAdapterDevice,
    klein_lora_fwd_device_resident_unfused,
    klein_lora_bwd_device_resident_tensors,
)


def _check_shape(got: List[Int], a: Int, b: Int) raises:
    if len(got) != 2 or got[0] != a or got[1] != b:
        raise Error("first checkpoint tensor shape mismatch")


def _load(st: ShardedSafeTensors, key: String, ctx: DeviceContext) raises -> Tensor:
    if key not in st.name_to_shard:
        raise Error("first oracle is missing " + key)
    return Tensor.from_view(st.tensor_view(key), ctx)


def _gate(
    name: String, got: Tensor, expected: Tensor, threshold: Float64,
    ctx: DeviceContext,
) raises:
    var ref_h = expected.to_host(ctx)
    var result = ParityHarness(threshold).compare(got, ref_h, ctx)
    print(name, " cos=", result.cos, " max_abs=", result.max_abs,
          " pass=", result.passed)
    if not result.passed:
        raise Error("first CUDA parity failed: " + name)


def _numeric_gate(path: String, ctx: DeviceContext) raises:
    var st = ShardedSafeTensors.open(path)
    var rank = 4
    var in_f = 64
    var out_f = 6144
    var rows = 1024
    var scale = Float32(1.0)
    _check_shape(st.tensor_info("kin_first_a").shape.copy(), rank, in_f)
    _check_shape(st.tensor_info("kin_first_b").shape.copy(), out_f, rank)
    var x = _load(st, "kin_first_x", ctx)
    var a = _load(st, "kin_first_a", ctx)
    var b = _load(st, "kin_first_b", ctx)
    var d_out = _load(st, "kin_first_d_out", ctx)
    var lo = LoraAdapterDevice(
        ArcPointer[Tensor](a^), ArcPointer[Tensor](b^),
        rank, in_f, out_f, scale,
    )

    var delta = klein_lora_fwd_device_resident_unfused(x, lo, rows, ctx)
    var grads = klein_lora_bwd_device_resident_tensors(
        d_out, x, lo, rows, ctx
    )
    _gate("first forward", delta, _load(st, "kref_first_delta", ctx),
          0.999999, ctx)
    _gate("first d_A", grads.d_a[], _load(st, "kref_first_d_a", ctx),
          0.999, ctx)
    _gate("first d_B", grads.d_b[], _load(st, "kref_first_d_b", ctx),
          0.999, ctx)
    _gate("first d_X", grads.d_x[], _load(st, "kref_first_d_x", ctx),
          0.999, ctx)
    print("PASS first LoRA CUDA forward/backward parity")


def _checkpoint_gate(path: String, ctx: DeviceContext) raises:
    var st = ShardedSafeTensors.open(path)
    var ak = String("diffusion_model.first.lora_A.weight")
    var bk = String("diffusion_model.first.lora_B.weight")
    if ak not in st.name_to_shard or bk not in st.name_to_shard:
        raise Error("corrected edit checkpoint is missing diffusion_model.first")
    var ash = st.tensor_info(ak).shape.copy()
    var bsh = st.tensor_info(bk).shape.copy()
    if len(ash) != 2 or len(bsh) != 2 or ash[1] != 64 or bsh[0] != 6144 \
        or ash[0] != bsh[1]:
        raise Error("first checkpoint tensor shape mismatch")
    var b = Tensor.from_view(st.tensor_view(bk), ctx).to_host(ctx)
    var max_abs = Float32(0.0)
    for x in b:
        var ax = -x if x < Float32(0.0) else x
        if ax > max_abs:
            max_abs = ax
    if max_abs == Float32(0.0):
        raise Error("first LoRA B remained exactly zero: adapter did not update")
    print("PASS first LoRA keys/shapes; B max_abs=", max_abs, " path=", path)


def main() raises:
    var args = argv()
    if len(args) < 2 or len(args) > 3:
        raise Error(
            "usage: krea2_omini_first_checkpoint_gate <oracle.safetensors>"
            " [trained_lora.safetensors]"
        )
    var ctx = DeviceContext()
    _numeric_gate(String(args[1]), ctx)
    if len(args) == 3:
        _checkpoint_gate(String(args[2]), ctx)
