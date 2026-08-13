# Checkpoint-free shape/geometry probe for gemma4_ltx_streamed.mojo.
#
# Covers, without touching the 10.6 GB Gemma-4 checkpoint:
#   1. the global "proportional" partial-rotary RoPE table — the first 64 of
#      256 pairs must rotate, pairs 64..255 must be exactly cos 1 / sin 0;
#   2. both Gemma-4 RMSNorm kernels (weighted and the weight-free v_norm);
#   3. a full layer forward at bucket 128 for ONE sliding layer (head_dim 256,
#      8 kv heads, separate v_proj) and ONE global layer (head_dim 512, 1 kv
#      head, V taken from the raw k_proj view), on synthetic BF16 weights.
#
# Passing a checkpoint path as argv[1] adds an OPTIONAL real-weight stage that
# loads layer 0 (sliding) and layer 5 (global) through `_load_layer` to prove
# the `model.language_model.` key names and the [1] layer_scalar view. The
# default, argument-free run stays checkpoint-free.
#
# Run (the cuDNN SDPA shim must be linked; a bare `mojo run -I .` fails at JIT
# time with "Symbols not found: [ flame_cudnn_sdpa_bf16_train_fwd ]"):
#
#   cd /home/alex/mojodiffusion && pixi run mojo run -I . \
#     -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa \
#     -Xlinker -Lserenitymojo/ops/cshim/lib/cudnn_stubs -Xlinker -lcudnn \
#     -Xlinker -rpath -Xlinker \
#       /home/alex/mojodiffusion/serenitymojo/ops/cshim/lib \
#     serenitymojo/models/text_encoder/parity/gemma4_streamed_probe.mojo

from max.gpu.host import DeviceContext
from std.sys import argv

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.models.text_encoder.gemma4_ltx_streamed import (
    GEMMA4_GLOBAL_HEAD_DIM,
    GEMMA4_GLOBAL_KV_HEADS,
    GEMMA4_GLOBAL_ROPE_ANGLES,
    GEMMA4_HEAD_DIM,
    GEMMA4_HEADS,
    GEMMA4_HIDDEN,
    GEMMA4_KV_HEADS,
    GEMMA4_LAYERS,
    GEMMA4_MAX_TOKENS,
    Gemma4HiddenBatch,
    Gemma4LayerWeights,
    _build_rope,
    _layer_forward,
    _load_layer,
    detect_gemma4_prefix,
    encode_gemma4_hidden_states_streamed,
    gemma4_rms_norm,
    gemma4_rms_norm_noscale,
)
from serenitymojo.ops.random import randn
from serenitymojo.ops.tensor_algebra import add_scalar, mul_scalar
from serenitymojo.tensor import Tensor


comptime BUCKET = 128
comptime REAL_LEN = 100
comptime INTERMEDIATE = 15360


def _stats(name: String, t: Tensor, ctx: DeviceContext) raises:
    var h = t.to_host(ctx)
    var n = len(h)
    var mn = h[0]
    var mx = h[0]
    var acc = Float64(0.0)
    var sq = Float64(0.0)
    var nonfinite = 0
    for i in range(n):
        var v = h[i]
        if not (v == v) or v > Float32(3.0e38) or v < Float32(-3.0e38):
            nonfinite += 1
        if v < mn:
            mn = v
        if v > mx:
            mx = v
        acc += Float64(v)
        sq += Float64(v) * Float64(v)
    var mean = acc / Float64(n)
    var rms = (sq / Float64(n)) ** 0.5
    print(
        name, "n=", n, "min=", mn, "max=", mx,
        "mean=", Float32(mean), "rms=", Float32(rms),
        "nonfinite=", nonfinite,
    )


def _w(var shape: List[Int], seed: UInt64, scale: Float32,
       ctx: DeviceContext) raises -> Tensor:
    """Synthetic BF16 weight ~ N(0, scale)."""
    return mul_scalar(randn(shape^, seed, STDtype.BF16, ctx), scale, ctx)


def _norm_w(var shape: List[Int], seed: UInt64,
            ctx: DeviceContext) raises -> Tensor:
    """Synthetic RMSNorm weight ~ 1 + N(0, 0.05), like the real tower."""
    return add_scalar(
        mul_scalar(randn(shape^, seed, STDtype.BF16, ctx), Float32(0.05), ctx),
        Float32(1.0), ctx,
    )


def _fake_layer(is_global: Bool, ctx: DeviceContext) raises -> Gemma4LayerWeights:
    var head_dim = GEMMA4_GLOBAL_HEAD_DIM if is_global else GEMMA4_HEAD_DIM
    var kv_heads = GEMMA4_GLOBAL_KV_HEADS if is_global else GEMMA4_KV_HEADS
    var q_out = GEMMA4_HEADS * head_dim
    var kv_out = kv_heads * head_dim
    var ws = Float32(0.02)

    var v_proj: Tensor
    if is_global:
        # Global layers carry no v_proj key at all.
        var dummy: List[Float32] = [Float32(0.0)]
        var dummy_shape: List[Int] = [1]
        v_proj = Tensor.from_host(dummy, dummy_shape^, STDtype.BF16, ctx)
    else:
        var vshape: List[Int] = [kv_out, GEMMA4_HIDDEN]
        v_proj = _w(vshape^, 907, ws, ctx)

    var ln1: List[Int] = [GEMMA4_HIDDEN]
    var ln2: List[Int] = [GEMMA4_HIDDEN]
    var ln3: List[Int] = [GEMMA4_HIDDEN]
    var ln4: List[Int] = [GEMMA4_HIDDEN]
    var qn: List[Int] = [head_dim]
    var kn: List[Int] = [head_dim]
    var qs: List[Int] = [q_out, GEMMA4_HIDDEN]
    var ks: List[Int] = [kv_out, GEMMA4_HIDDEN]
    var os_: List[Int] = [GEMMA4_HIDDEN, q_out]
    var gs: List[Int] = [INTERMEDIATE, GEMMA4_HIDDEN]
    var us: List[Int] = [INTERMEDIATE, GEMMA4_HIDDEN]
    var ds: List[Int] = [GEMMA4_HIDDEN, INTERMEDIATE]

    return Gemma4LayerWeights(
        _norm_w(ln1^, 101, ctx),
        _norm_w(ln2^, 102, ctx),
        _norm_w(ln3^, 103, ctx),
        _norm_w(ln4^, 104, ctx),
        _norm_w(qn^, 105, ctx),
        _norm_w(kn^, 106, ctx),
        _w(qs^, 901, ws, ctx),
        _w(ks^, 903, ws, ctx),
        v_proj^,
        _w(os_^, 909, ws, ctx),
        _w(gs^, 911, ws, ctx),
        _w(us^, 913, ws, ctx),
        _w(ds^, 915, ws, ctx),
        Float32(0.37),  # measured layer_scalar values live in 0.0045..0.887
        is_global,
    )


def _real_layer_stage(
    path: String, with_global: Bool, ctx: DeviceContext
) raises:
    """Optional: load a real sliding (and, on request, global) layer to prove
    the resolved key names (google `model.language_model.*` vs LTX-2.5 `model.*`)
    against a shipped checkpoint.

    `with_global` is opt-in because layer 5's tensors sit near the END of the
    single 23.9 GB safetensors file: requesting them against a still-downloading
    checkpoint would read past the mmap'd extent.
    """
    var st = ShardedSafeTensors.open(path)
    var prefix = detect_gemma4_prefix(st)
    print("detected key prefix '", prefix, "'")
    var l0 = _load_layer(st, 0, False, prefix, ctx)
    print(
        "real layer0 (sliding) q_proj=[", l0.q_proj.shape()[0], ",",
        l0.q_proj.shape()[1], "] k_proj=[", l0.k_proj.shape()[0], ",",
        l0.k_proj.shape()[1], "] v_proj=[", l0.v_proj.shape()[0], ",",
        l0.v_proj.shape()[1], "] q_norm=", l0.q_norm.numel(),
        " layer_scalar=", l0.layer_scalar,
    )
    st.release_to_os()
    if not with_global:
        print("real layer5 (global)  SKIPPED (pass 'global' as argv[2])")
        return
    var l5 = _load_layer(st, 5, True, prefix, ctx)
    print(
        "real layer5 (global)  q_proj=[", l5.q_proj.shape()[0], ",",
        l5.q_proj.shape()[1], "] k_proj=[", l5.k_proj.shape()[0], ",",
        l5.k_proj.shape()[1], "] q_norm=", l5.q_norm.numel(),
        " o_proj=[", l5.o_proj.shape()[0], ",", l5.o_proj.shape()[1], "]",
        " layer_scalar=", l5.layer_scalar,
    )
    st.release_to_os()


def _entrypoint_compile_touch(ctx: DeviceContext) raises:
    """Never executed — forces elaboration of the streamed entrypoint so the
    probe's exit code covers it too."""
    var ids: List[List[Int]] = [[1, 2, 3]]
    var batch: Gemma4HiddenBatch = encode_gemma4_hidden_states_streamed(
        String("/nonexistent"), ids, ctx
    )
    print(len(batch.states), batch.bucket, len(batch.lengths), GEMMA4_LAYERS)


def main() raises:
    var ctx = DeviceContext()
    var args = argv()
    print("== gemma4 streamed probe ==")
    print(
        "config hidden=", GEMMA4_HIDDEN, " heads=", GEMMA4_HEADS,
        " kv=", GEMMA4_KV_HEADS, " head_dim=", GEMMA4_HEAD_DIM,
        " global_head_dim=", GEMMA4_GLOBAL_HEAD_DIM,
        " global_kv=", GEMMA4_GLOBAL_KV_HEADS,
        " rope_angles=", GEMMA4_GLOBAL_ROPE_ANGLES,
        " max_tokens=", GEMMA4_MAX_TOKENS,
    )

    # ── 1) global partial-rotary RoPE table ──────────────────────────────────
    var rope = _build_rope(BUCKET, REAL_LEN, ctx)
    var gcos = rope.global_cos_q[].to_host(ctx)
    var gsin = rope.global_sin_q[].to_host(ctx)
    var half = GEMMA4_GLOBAL_HEAD_DIM // 2
    print("global rope table numel=", len(gcos), " expected=",
          BUCKET * GEMMA4_HEADS * half)
    # Row 0 is (position=1024-100=924, head 0).
    for col in [0, 1, 63, 64, 65, 255]:
        print(
            "  global row0 col", col, " cos=", gcos[col], " sin=", gsin[col],
        )
    var bad = 0
    for col in range(GEMMA4_GLOBAL_ROPE_ANGLES, half):
        if gcos[col] != Float32(1.0) or gsin[col] != Float32(0.0):
            bad += 1
    print("  non-identity NOPE pairs (must be 0):", bad)
    var lcos = rope.local_cos_q[].to_host(ctx)
    var lsin = rope.local_sin_q[].to_host(ctx)
    print(
        "  sliding row0 col0 cos=", lcos[0], " sin=", lsin[0],
        " col127 cos=", lcos[127], " sin=", lsin[127],
    )

    # ── 2) both RMSNorm kernels ──────────────────────────────────────────────
    var xshape: List[Int] = [4, GEMMA4_HIDDEN]
    var x = mul_scalar(
        randn(xshape^, 7, STDtype.BF16, ctx), Float32(3.0), ctx
    )
    var wshape: List[Int] = [GEMMA4_HIDDEN]
    var w = _norm_w(wshape^, 8, ctx)
    _stats("rms input      ", x, ctx)
    _stats("rms weighted   ", gemma4_rms_norm(x, w, ctx), ctx)
    _stats("rms noscale    ", gemma4_rms_norm_noscale(x, ctx), ctx)

    # ── 3) one sliding + one global layer forward at bucket 128 ──────────────
    var hshape: List[Int] = [1, BUCKET, GEMMA4_HIDDEN]
    var hidden = mul_scalar(
        randn(hshape^, 21, STDtype.BF16, ctx), Float32(2.0), ctx
    )
    _stats("layer input    ", hidden, ctx)

    var sliding = _fake_layer(False, ctx)
    var out_s = _layer_forward(sliding, hidden, rope, REAL_LEN, ctx)
    ctx.synchronize()
    print(
        "sliding out shape=[", out_s.shape()[0], ",", out_s.shape()[1],
        ",", out_s.shape()[2], "]",
    )
    _stats("sliding output ", out_s, ctx)

    var glob = _fake_layer(True, ctx)
    var out_g = _layer_forward(glob, hidden, rope, REAL_LEN, ctx)
    ctx.synchronize()
    print(
        "global out shape=[", out_g.shape()[0], ",", out_g.shape()[1],
        ",", out_g.shape()[2], "]",
    )
    _stats("global output  ", out_g, ctx)

    if len(args) > 1:
        var with_global = len(args) > 2 and String(args[2]) == String("global")
        _real_layer_stage(String(args[1]), with_global, ctx)
    if len(args) > 99:
        _entrypoint_compile_touch(ctx)

    print("PROBE OK")
