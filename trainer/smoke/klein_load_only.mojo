# Load-only: measure the Klein 9B weight-load peak GPU memory (isolate load vs forward).
from std.collections import List
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenity_trainer.model.klein.double_block import DoubleBlockWeights
from serenity_trainer.model.klein.single_block import SingleBlockWeights
from serenity_trainer.model.klein.weights import (
    load_double_block_weights, load_single_block_weights,
    load_klein_stack_base, load_klein_step_mod_weights, build_klein_vec_silu,
)
from serenity_trainer.model.KleinModel import KDIM, KNUM_DOUBLE, KNUM_SINGLE, KTIMESTEP_DIM

comptime CKPT = "/home/alex/.serenity/models/checkpoints/flux-2-klein-base-9b.safetensors"

def main() raises:
    var ctx = DeviceContext()
    print("[load] open", CKPT)
    var ckpt = SafeTensors.open(String(CKPT))
    var ts_dev = Tensor.from_host([Float32(250.0)], [1], STDtype.F32, ctx)
    var vec_silu = build_klein_vec_silu(ckpt, ts_dev, KTIMESTEP_DIM, KDIM, ctx)
    var base = load_klein_stack_base(ckpt, vec_silu, KDIM, ctx)
    var step_mod_w = load_klein_step_mod_weights(ckpt, KDIM, ctx)
    print("[load] base done")
    var dbw = List[DoubleBlockWeights]()
    for bi in range(KNUM_DOUBLE):
        dbw.append(load_double_block_weights(ckpt, bi, ctx))
        print("[load] double", bi, "done")
    var sbw = List[SingleBlockWeights]()
    for bi in range(KNUM_SINGLE):
        sbw.append(load_single_block_weights(ckpt, bi, ctx))
        print("[load] single", bi, "done")
    print("[load] ALL DONE:", len(dbw), "double +", len(sbw), "single")
    ctx.synchronize()
