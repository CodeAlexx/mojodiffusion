# Numeric block-0 parity gate for cosmos_predict25_dit.
# Loads cosmos_block0_fixture.safetensors (built by gen_cosmos_block0_oracle.py
# from the REAL post-trained checkpoint + the prior Rust port's captured
# activations; oracle self-check vs captured block_0_output = cos 0.999987),
# runs cosmos_block_forward, compares to `expected`. Gate cos >= 0.999.

from std.gpu.host import DeviceContext
from std.memory import ArcPointer
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.parity import ParityHarness
from serenitymojo.models.dit.cosmos_predict25_dit import (
    CosmosConfig, cosmos_block_forward,
)

comptime FIX = "/home/alex/mojodiffusion/serenitymojo/models/dit/parity/cosmos_block0_fixture.safetensors"
comptime N = 128       # cropped grid (Tp=2,Hp=8,Wp=8); real-checkpoint weights
comptime TP = 2
comptime HPWP = 64     # 8*8
comptime TXT = 512
comptime H = 16
comptime DH = 128


def _load(st: ShardedSafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    var tv = st.tensor_view(name)
    return Tensor.from_view(tv, ctx)


def main() raises:
    var ctx = DeviceContext()
    var cfg = CosmosConfig.v2_2b_production()
    var st = ShardedSafeTensors.open(FIX)

    var x_in = _load(st, "input", ctx)        # [N,D] F32
    var emb = cast_tensor(_load(st, "emb", ctx), STDtype.BF16, ctx)         # [Tp,D]
    var adaln = cast_tensor(_load(st, "adaln_lora", ctx), STDtype.BF16, ctx)  # [Tp,3D]
    var text = cast_tensor(_load(st, "text_ctx", ctx), STDtype.BF16, ctx)    # [TXT,1024]
    var cos_b = cast_tensor(_load(st, "cos", ctx), STDtype.BF16, ctx)        # [N,64]
    var sin_b = cast_tensor(_load(st, "sin", ctx), STDtype.BF16, ctx)
    var expected = _load(st, "expected", ctx)  # [N,D] F32

    var bw = Dict[String, ArcPointer[Tensor]]()
    var suffixes = [
        "adaln_modulation_self_attn.1.weight", "adaln_modulation_self_attn.2.weight",
        "adaln_modulation_cross_attn.1.weight", "adaln_modulation_cross_attn.2.weight",
        "adaln_modulation_mlp.1.weight", "adaln_modulation_mlp.2.weight",
        "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
        "self_attn.output_proj.weight", "self_attn.q_norm.weight", "self_attn.k_norm.weight",
        "cross_attn.q_proj.weight", "cross_attn.k_proj.weight", "cross_attn.v_proj.weight",
        "cross_attn.output_proj.weight", "cross_attn.q_norm.weight", "cross_attn.k_norm.weight",
        "mlp.layer1.weight", "mlp.layer2.weight",
    ]
    for sfx in suffixes:
        var s = String(sfx)
        var t = cast_tensor(_load(st, String("w_") + s, ctx), STDtype.BF16, ctx)
        bw[s] = ArcPointer(t^)

    var out = cosmos_block_forward[N, TXT, H, DH](
        x_in^, emb, adaln, text, cos_b, sin_b, bw, cfg, TP, HPWP, ctx
    )

    var ph = ParityHarness(0.999)
    var ref_host = expected.to_host(ctx)
    var res = ph.compare(out, ref_host, ctx)
    print("cosmos block-0 parity:", res)
    if res.passed:
        print("GATE: PASS")
    else:
        print("GATE: FAIL")
