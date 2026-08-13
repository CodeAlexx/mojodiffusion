# Isolated per-layer parity gate for gemma4_ltx_streamed.mojo.
#
# The 49-state end-to-end gate measures ACCUMULATED error. This driver instead
# feeds ONE layer the oracle's own captured input (forward_pre_hook) and
# compares against the oracle's captured output for that same layer, which
# separates "this layer's math is wrong" from "48 layers of bf16 drift".
#
# Layer 0 is `sliding_attention` (head_dim 256, 8 kv heads, own v_proj).
# Layer 5 is `full_attention` (head_dim 512, 1 kv head, V = raw k_proj view).
#
# Padding convention: the oracle ran the full 1024 window with the real tokens
# LEFT-padded into the tail, so its valid rows are [1024-real_len, 1024). The
# Mojo layer wants the real rows at the FRONT of a compact bucket with RoPE
# started at offset (1024-real_len). Both therefore see identical absolute
# positions, and only the valid prefix is compared.
#
# Build: same cuDNN-shim linker flag set as gemma4_states_parity.mojo.

from max.gpu.host import DeviceContext

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.models.text_encoder.gemma4_ltx_streamed import (
    GEMMA4_HIDDEN,
    GEMMA4_MAX_TOKENS,
    _build_rope,
    _layer_forward,
    _load_layer,
    detect_gemma4_prefix,
)
from serenitymojo.parity import ParityHarness
from serenitymojo.tensor import Tensor


comptime CKPT = String(
    "/home/alex/.serenity/models/text_encoders/gemma-4-12b-it-standalone"
)
comptime REFS = String(
    "/home/alex/mojodiffusion/serenitymojo/models/text_encoder/parity/gemma4_refs"
)
comptime BUCKET = 128


def _rms(v: List[Float32]) -> Float64:
    var s: Float64 = 0.0
    for i in range(len(v)):
        s += Float64(v[i]) * Float64(v[i])
    return (s / Float64(len(v))) ** 0.5


def _gate_layer(
    layer_idx: Int, is_global: Bool, ref_file: String, ctx: DeviceContext
) raises:
    var st = ShardedSafeTensors.open(CKPT)
    var refs = ShardedSafeTensors.open(REFS + "/" + ref_file)
    var meta = ShardedSafeTensors.open(
        REFS + "/gemma4_oracle_meta_f32.safetensors"
    )
    var rl = Tensor.from_view_as_f32(
        meta.tensor_view("real_len_f32"), ctx
    ).to_host(ctx)
    var real_len = Int(rl[0])
    var offset = GEMMA4_MAX_TOKENS - real_len

    var in_full = Tensor.from_view_as_bf16(
        refs.tensor_view("in"), ctx
    ).to_host(ctx)
    var out_full = Tensor.from_view_as_bf16(
        refs.tensor_view("out"), ctx
    ).to_host(ctx)

    # Valid rows of the oracle input -> front of a compact bucket.
    var packed = List[Float32]()
    for i in range(offset * GEMMA4_HIDDEN, GEMMA4_MAX_TOKENS * GEMMA4_HIDDEN):
        packed.append(in_full[i])
    while len(packed) < BUCKET * GEMMA4_HIDDEN:
        packed.append(Float32(0.0))
    var shape: List[Int] = [1, BUCKET, GEMMA4_HIDDEN]
    var hidden = Tensor.from_host(packed, shape^, STDtype.BF16, ctx)

    var rope = _build_rope(BUCKET, real_len, ctx)
    var w = _load_layer(st, layer_idx, is_global, detect_gemma4_prefix(st), ctx)
    var got = _layer_forward(w, hidden, rope, real_len, ctx)
    var got_host = got.to_host(ctx)

    var valid = real_len * GEMMA4_HIDDEN
    var mine = List[Float32]()
    for i in range(valid):
        mine.append(got_host[i])
    var want = List[Float32]()
    for i in range(offset * GEMMA4_HIDDEN, GEMMA4_MAX_TOKENS * GEMMA4_HIDDEN):
        want.append(out_full[i])

    var harness = ParityHarness(0.999)
    var r = harness.compare_host(mine, want)
    var rm = _rms(mine)
    var rr = _rms(want)
    print(
        "layer", layer_idx,
        " global=", is_global,
        " cos=", r.cos,
        " max_abs=", r.max_abs,
        " rms_mine=", rm,
        " rms_ref=", rr,
        " rms_ratio=", (rm / rr) if rr != 0.0 else 0.0,
        " PASS" if r.passed else " FAIL",
    )


def main() raises:
    var ctx = DeviceContext()
    print("== gemma4 ISOLATED per-layer parity (oracle-hook inputs) ==")
    _gate_layer(0, False, String("gemma4_oracle_layer0.safetensors"), ctx)
    _gate_layer(5, True, String("gemma4_oracle_layer5.safetensors"), ctx)
