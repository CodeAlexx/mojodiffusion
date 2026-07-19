# models/nldiffusion/parity/backbone_layer_probe.mojo — CHUNK A gate.
#
# Loads `encoder.layers.0.*` (bf16) from the 6-shard NL-Diffusion-Image model,
# feeds the captured oracle input `layer0.in` + the captured YaRN rope cos/sin,
# runs ONE `nldiff_decoder_layer`, and compares the output to `layer0.out` with
# ParityHarness (gate cos >= 0.999).
#
# Run (JIT):
#   cd /home/alex/mojodiffusion && \
#     pixi run mojo run -I . serenitymojo/models/nldiffusion/parity/backbone_layer_probe.mojo

from std.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.parity import ParityHarness
from serenitymojo.models.nldiffusion.backbone import (
    NLDiffConfig,
    NLDiffLayerWeights,
    nldiff_decoder_layer,
)

comptime MODEL_DIR = "/mnt/disk1/models/NL-Diffusion-Image"
comptime PARITY_DIR = "/home/alex/mojodiffusion/serenitymojo/models/nldiffusion/parity"
comptime SEQ = 1141
comptime HEAD_DIM = 128
comptime HALF = 64  # HEAD_DIM // 2


def _load_bf16(
    st: ShardedSafeTensors, name: String, ctx: DeviceContext
) raises -> ArcPointer[Tensor]:
    var tv = st.tensor_view(name)
    var t = Tensor.from_view(tv, ctx)
    return ArcPointer(t^)


def _load_oracle_f32(
    st: ShardedSafeTensors, name: String, ctx: DeviceContext
) raises -> Tensor:
    var tv = st.tensor_view(name)
    return Tensor.from_view(tv, ctx)


def _build_rope_perhead(
    cos_raw: List[Float32], heads: Int
) raises -> List[Float32]:
    """Reshape captured rope emb [1,SEQ,HEAD_DIM] into rope_halfsplit's per-head
    half-split layout [SEQ*heads, HALF].

    The reference builds `emb = cat(freqs, freqs)` then cos/sin over it, so the
    first HALF columns per position ARE cos/sin(freqs) (cols HALF..HEAD_DIM are
    an identical copy). rope_halfsplit wants cos[row, i] with row = position*head
    and i in [0,HALF); the angle depends only on position, repeated across heads.
    """
    var out = List[Float32]()
    for t in range(SEQ):
        for _hh in range(heads):
            for i in range(HALF):
                out.append(cos_raw[t * HEAD_DIM + i])
    return out^


def main() raises:
    var ctx = DeviceContext()
    var cfg = NLDiffConfig.default()

    print("[chunk A] loading encoder.layers.0.* (bf16) from", MODEL_DIR)
    var model = ShardedSafeTensors.open(MODEL_DIR)
    var p = String("encoder.layers.0.")
    var w = NLDiffLayerWeights(
        _load_bf16(model, p + "input_layernorm.weight", ctx),
        _load_bf16(model, p + "post_attention_layernorm.weight", ctx),
        _load_bf16(model, p + "self_attn.q_proj.weight", ctx),
        _load_bf16(model, p + "self_attn.k_proj.weight", ctx),
        _load_bf16(model, p + "self_attn.v_proj.weight", ctx),
        _load_bf16(model, p + "self_attn.o_proj.weight", ctx),
        _load_bf16(model, p + "mlp.gate_proj.weight", ctx),
        _load_bf16(model, p + "mlp.up_proj.weight", ctx),
        _load_bf16(model, p + "mlp.down_proj.weight", ctx),
    )

    # ── oracle inputs ────────────────────────────────────────────────────────
    print("[chunk A] loading oracle inputs")
    var oracle = ShardedSafeTensors.open(
        String(PARITY_DIR) + "/oracle1.safetensors"
    )
    var rope = ShardedSafeTensors.open(
        String(PARITY_DIR) + "/rope_oracle.safetensors"
    )

    # layer0.in is F32 in-file (losslessly upcast from the reference bf16). Cast
    # back to bf16 to mirror the bf16 reference compute.
    var x_f32 = _load_oracle_f32(oracle, "layer0.in", ctx)
    var x = cast_tensor(x_f32, STDtype.BF16, ctx)

    # rope cos/sin: read the captured emb [1,SEQ,HEAD_DIM] to host F32, then
    # build the per-head half-split tables. Kept F32 (rope_halfsplit accepts F32
    # tables with a bf16 x).
    var cos_t = _load_oracle_f32(rope, "rope.cos", ctx)
    var sin_t = _load_oracle_f32(rope, "rope.sin", ctx)
    var cos_raw = cos_t.to_host(ctx)
    var sin_raw = sin_t.to_host(ctx)

    var cos_q_h = _build_rope_perhead(cos_raw, cfg.num_heads)
    var sin_q_h = _build_rope_perhead(sin_raw, cfg.num_heads)
    var cos_k_h = _build_rope_perhead(cos_raw, cfg.num_kv_heads)
    var sin_k_h = _build_rope_perhead(sin_raw, cfg.num_kv_heads)

    var q_sh = [SEQ * cfg.num_heads, HALF]
    var k_sh = [SEQ * cfg.num_kv_heads, HALF]
    var cos_q = Tensor.from_host(cos_q_h, q_sh.copy(), STDtype.F32, ctx)
    var sin_q = Tensor.from_host(sin_q_h, q_sh.copy(), STDtype.F32, ctx)
    var cos_k = Tensor.from_host(cos_k_h, k_sh.copy(), STDtype.F32, ctx)
    var sin_k = Tensor.from_host(sin_k_h, k_sh.copy(), STDtype.F32, ctx)

    # ── run one decoder layer ────────────────────────────────────────────────
    print("[chunk A] running nldiff_decoder_layer[S=", SEQ, "]")
    var out = nldiff_decoder_layer[SEQ](
        x, w, cfg, cos_q, sin_q, cos_k, sin_k, ctx
    )

    # ── parity vs layer0.out ─────────────────────────────────────────────────
    var reference = _load_oracle_f32(oracle, "layer0.out", ctx)
    var ref_host = reference.to_host(ctx)
    var harness = ParityHarness(0.999)
    var res = harness.compare(out, ref_host, ctx)
    print("[chunk A] layer0 parity:", res)
    print("[chunk A] cos =", res.cos, " max_abs_diff =", res.max_abs)
    if res.passed:
        print("[chunk A] GATE PASS (cos >= 0.999)")
    else:
        print("[chunk A] GATE FAIL (cos < 0.999)")
