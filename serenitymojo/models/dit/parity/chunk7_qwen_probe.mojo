# chunk7 parity: Ideogram Qwen3-VL 13-tap features vs Wave-0 fixture.
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.parity import ParityHarness
from serenitymojo.ops.tensor_algebra import slice
from serenitymojo.models.text_encoder.ideogram_qwen3vl import (
    load_ideogram_qwen3vl, encode_ideogram_taps,
)

comptime TE = "/home/alex/.serenity/models/ideogram-4-fp8/text_encoder/model.safetensors"
comptime FX = "/home/alex/mojodiffusion/serenitymojo/models/dit/parity/ideogram4_fx_qwen.safetensors"


def main() raises:
    var ctx = DeviceContext()
    var enc = load_ideogram_qwen3vl(TE, ctx)
    # 15 real tokens + 1 pad (151643) -> seq 16 (a supported comptime case);
    # causal + right-pad masking leaves the 15 real-token outputs unchanged.
    var ids = [151644, 872, 198, 64, 2518, 23739, 389, 264, 4158, 1965, 151645, 198, 151644, 77091, 198, 151643]
    var feats16 = encode_ideogram_taps(enc, ids, ctx)
    var feats = slice(feats16, 1, 0, 15, ctx)   # back to 15 real tokens
    print("feats:", feats.shape()[0], feats.shape()[1], feats.shape()[2], "dtype", feats.dtype().name())
    var fx = ShardedSafeTensors.open(FX)
    var exp_host = Tensor.from_view(fx.tensor_view("chunk7.llm_features"), ctx).to_host(ctx)
    print("chunk7 qwen 13-tap parity:", ParityHarness(0.999).compare(feats, exp_host, ctx))
