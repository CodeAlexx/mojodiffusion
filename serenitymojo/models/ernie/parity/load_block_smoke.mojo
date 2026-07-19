# serenitymojo/models/ernie/parity/load_block_smoke.mojo
#
# Verifies models/ernie/weights.mojo can open the REAL ERNIE transformer sharded
# safetensors and load block-0's weights into ErnieBlockWeights — a real H2D
# round-trip with a shape assert on to_q. Mirrors klein/parity/load_*_smoke.mojo.
#
# Run:
#   cd /home/alex/mojodiffusion
#   pixi run mojo build -I . -Xlinker -lm \
#       serenitymojo/models/ernie/parity/load_block_smoke.mojo -o /tmp/ernie_load_block_smoke
#   /tmp/ernie_load_block_smoke

from std.gpu.host import DeviceContext
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.models.dit.ernie_contract import (
    ERNIE_TRANSFORMER_DIR, ERNIE_DIT_HIDDEN, ERNIE_DIT_HEAD_DIM, ERNIE_DIT_FFN_HIDDEN,
)
from serenitymojo.models.ernie.weights import (
    load_ernie_block_weights, verify_block_to_q_shape,
)


def main() raises:
    var ctx = DeviceContext()
    print("==== ernie load_block_smoke (real shard 0 block-0 H2D) ====")
    var st = ShardedSafeTensors.open(String(ERNIE_TRANSFORMER_DIR))
    print("transformer tensors:", st.num_tensors())

    var hidden = verify_block_to_q_shape(st, 0, ERNIE_DIT_HIDDEN)
    print("block-0 to_q dim0 (hidden) =", hidden, " expected", ERNIE_DIT_HIDDEN)

    var w = load_ernie_block_weights(st, 0, ctx)
    # touch a few resident tensors to confirm the upload happened at real shapes
    var wq_sh = w.wq[].shape()
    var wgate_sh = w.wgate[].shape()
    var qn_sh = w.q_norm[].shape()
    print("wq shape    [", wq_sh[0], ",", wq_sh[1], "]")
    print("wgate shape [", wgate_sh[0], ",", wgate_sh[1], "]")
    print("q_norm shape [", qn_sh[0], "]")

    if wq_sh[0] != ERNIE_DIT_HIDDEN or wq_sh[1] != ERNIE_DIT_HIDDEN:
        raise Error("ernie load_block_smoke: wq shape mismatch")
    if wgate_sh[0] != ERNIE_DIT_FFN_HIDDEN or wgate_sh[1] != ERNIE_DIT_HIDDEN:
        raise Error("ernie load_block_smoke: wgate shape mismatch")
    if qn_sh[0] != ERNIE_DIT_HEAD_DIM:
        raise Error("ernie load_block_smoke: q_norm shape mismatch")

    print("VERDICT: PASS — ERNIE block-0 weights loaded from real shards at correct shapes")
