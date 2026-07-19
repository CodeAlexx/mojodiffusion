# Full-forward parity gate for magihuman_dit (CHUNK B).
#
# Streams the REAL distill bf16 checkpoint layer-by-layer (the 30.6 GB ckpt does
# not fit GPU resident; each layer's weights ~0.7-1.5 GB fit in free VRAM). Runs
# the full MagiHuman forward — adapter embed + Fourier RoPE (REAL adapter.rope.bands)
# + 40 layers (MM 0..3 GELU7 / shared 4..35 SwiGLU7 / MM 36..39 SwiGLU7) + final
# heads — via the CHUNK-B ops in models/dit/magihuman_dit.mojo, and compares to
# the canonical Python full-forward oracle (magihuman_full_fixture.safetensors).
#
# Self-attn uses sdpa_nomask_tiled (online softmax, no [S,S]) → no Dh=128 OOM.
# Gate cos >= 0.99 (deep 40-layer chain).

from std.gpu.host import DeviceContext
from std.memory import ArcPointer
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.tensor_algebra import add_scalar, slice
from serenitymojo.parity import ParityHarness
from serenitymojo.models.dit.magihuman_dit import (
    MagiHumanConfig,
    magihuman_shared_block_forward,
    magihuman_mm_block_forward,
    magihuman_adapter_embed,
    magihuman_rope_from_coords,
    magihuman_final_heads,
    _is_mm_layer,
    _is_gelu7_layer,
)

comptime CKPT = "/home/alex/.serenity/models/dits/magihuman_distill_bf16.safetensors"
comptime FIX = "/home/alex/mojodiffusion/serenitymojo/models/dit/parity/magihuman_full_fixture.safetensors"
comptime L = 128          # V=64 + A=32 + T=32 (matches oracle)
comptime H = 40
comptime HKV = 8
comptime DH = 128


def _load(st: ShardedSafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    var tv = st.tensor_view(name)
    return Tensor.from_view(tv, ctx)


# Load a layer's weights into a dict. Big weights -> BF16 (already bf16 in ckpt).
# Norm gains get the (weight+1) variant under ".p1".
def _load_layer(
    ck: ShardedSafeTensors, prefix: String, ctx: DeviceContext
) raises -> Dict[String, ArcPointer[Tensor]]:
    var w = Dict[String, ArcPointer[Tensor]]()
    var big = [
        "attention.linear_qkv.weight", "attention.linear_proj.weight",
        "mlp.up_gate_proj.weight", "mlp.down_proj.weight",
    ]
    for sfx in big:
        var s = String(sfx)
        var t = _load(ck, prefix + s, ctx)
        w[s] = ArcPointer(t^)
    var norms = [
        "attention.pre_norm.weight", "attention.q_norm.weight",
        "attention.k_norm.weight", "mlp.pre_norm.weight",
    ]
    for sfx in norms:
        var s = String(sfx)
        var t = _load(ck, prefix + s, ctx)
        var p1 = add_scalar(t, 1.0, ctx)
        w[s + ".p1"] = ArcPointer(p1^)
    return w^


def main() raises:
    var ctx = DeviceContext()
    var cfg = MagiHumanConfig.magihuman_15b()

    var fx = ShardedSafeTensors.open(FIX)
    var ck = ShardedSafeTensors.open(CKPT)

    # Group sizes from the fixture (V,A,T) — must match comptime L.
    var V = 64
    var A = 32
    var T = 32
    var gs = List[Int]()
    gs.append(V)
    gs.append(A)
    gs.append(T)

    # ----- Adapter inputs + Fourier RoPE (REAL bands) -----
    var xv = _load(fx, "xv", ctx)        # [V,192] F32
    var xa = _load(fx, "xa", ctx)        # [A,64]  F32
    var xt = _load(fx, "xt", ctx)        # [T,3584] F32
    var expected = _load(fx, "expected", ctx)  # [L,192] F32

    # bands: read the REAL adapter.rope.bands from the checkpoint (skeptic FRAGILE
    # note — do NOT fabricate). coords from the fixture.
    var bands_t = _load(ck, "adapter.rope.bands", ctx)  # [16] BF16
    var bands_host = bands_t.to_host(ctx)
    var bands64 = List[Float64]()
    for i in range(len(bands_host)):
        bands64.append(Float64(bands_host[i]))

    var coords_t = _load(fx, "coords", ctx)  # [L,9] F32
    var coords_host = coords_t.to_host(ctx)
    var coords32 = List[Float32]()
    for i in range(len(coords_host)):
        coords32.append(Float32(coords_host[i]))

    var rope = magihuman_rope_from_coords(coords32, bands64, L, ctx)

    # ----- Adapter embed -----
    var aw_video_w = _load(ck, "adapter.video_embedder.weight", ctx)
    var aw_video_b = _load(ck, "adapter.video_embedder.bias", ctx)
    var aw_audio_w = _load(ck, "adapter.audio_embedder.weight", ctx)
    var aw_audio_b = _load(ck, "adapter.audio_embedder.bias", ctx)
    var aw_text_w = _load(ck, "adapter.text_embedder.weight", ctx)
    var aw_text_b = _load(ck, "adapter.text_embedder.bias", ctx)
    # adapter inputs are F32; cast embedders to F32 for the matmul.
    var h = magihuman_adapter_embed(
        xv, xa, xt, gs,
        cast_tensor(aw_video_w, STDtype.F32, ctx), cast_tensor(aw_video_b, STDtype.F32, ctx),
        cast_tensor(aw_audio_w, STDtype.F32, ctx), cast_tensor(aw_audio_b, STDtype.F32, ctx),
        cast_tensor(aw_text_w, STDtype.F32, ctx), cast_tensor(aw_text_b, STDtype.F32, ctx),
        cfg.hidden_size, ctx,
    )  # [L, hidden] F32

    # ----- 40-layer stack (streamed per layer) -----
    var layers_done = 0
    for i in range(cfg.num_layers):
        var prefix = String("block.layers.") + String(i) + "."
        var w = _load_layer(ck, prefix, ctx)
        if _is_mm_layer(i):
            var use_swiglu = not _is_gelu7_layer(i)
            h = magihuman_mm_block_forward[L, H, HKV, DH](
                h, rope.cos_e, rope.sin_e, w, cfg, gs, use_swiglu, ctx
            )
        else:
            h = magihuman_shared_block_forward[L, H, HKV, DH](
                h, rope.cos_e, rope.sin_e, w, cfg, ctx
            )
        layers_done += 1
        # w drops here -> per-layer VRAM freed.
    print("layers run:", layers_done)

    # ----- Final heads -----
    var fnv = _load(ck, "final_norm_video.weight", ctx)
    var fnv_p1 = add_scalar(fnv, 1.0, ctx)
    var flv = _load(ck, "final_linear_video.weight", ctx)
    var fna = _load(ck, "final_norm_audio.weight", ctx)
    var fna_p1 = add_scalar(fna, 1.0, ctx)
    var fla = _load(ck, "final_linear_audio.weight", ctx)
    var out = magihuman_final_heads(
        h, gs, fnv_p1, flv, fna_p1, fla, cfg.rms_eps, ctx
    )  # [L,192] F32

    var expected_h = expected.to_host(ctx)
    var harness = ParityHarness(0.99)
    var res = harness.compare(out, expected_h, ctx)
    print("MagiHuman FULL forward gate:")
    print("  cos     =", res.cos)
    print("  max_abs =", res.max_abs)
    print("  n       =", res.n)
    if res.passed:
        print("  GATE: PASS (cos >= 0.99)")
    else:
        print("  GATE: FAIL (cos < 0.99)")
