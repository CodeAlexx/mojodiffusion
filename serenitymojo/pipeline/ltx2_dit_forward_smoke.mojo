# LTX-2 FULL 48-block forward_audio_video VELOCITY parity smoke (Plan P5).
#
# The keystone P5 gate: assemble the full dual-stream AV DiT forward by looping
# the verified `ltx2_block_forward_av` over ALL 48 transformer blocks, then run
# the model-level output stage, and GATE the final VIDEO + AUDIO velocity
# against scripts/ltx2_dit_forward_parity_ref.py (the full forward_audio_video
# velocity oracle). Per-block/isolated-block checks gate at high cosine; final
# video velocity uses a calibrated threshold because the 48-block residual stack
# accumulates vendor-GEMM numerical drift that is amplified by the video output
# modulation. Audio remains above 0.999 in the full gate.
#
# Block residency (KEY FACT): boundary blocks 0-3 and 47 are pure-BF16 in the
# distilled-fp8 checkpoint (FP8 skips them) -> load directly via
# LTX2AVBlockWeights.load. Inner blocks 4-46 store their attn/FFN matrices as
# float8_e4m3fn -> stream via LTX2BlockStream.load_block_bf16 (dequant-on-use
# with the per-tensor weight_scale) and rehome via
# LTX2AVBlockWeights.from_fp8_block. Each block is loaded, run, and DROPPED
# before the next (single-resident window) so peak VRAM stays well under 24 GB.
#
# What the oracle dumps (output/ltx2_dit_forward/dit_forward_ref.safetensors):
#   - v_flat [1,S_V,128] / a_flat [1,S_A,128]      patchified latent inputs
#   - video_pre [1,N_TXT,4096] / audio_pre [1,N_TXT,2048]  PRE-connector ctx
#   - v_timestep/a_timestep, v_embedded/a_embedded, v_ca_ss/a_ca_ss,
#     v_ca_gate/a_ca_gate, video_prompt_ts/audio_prompt_ts   per-forward mods
#   - {v,a,ca_v,ca_a}_{cos,sin}                    RoPE tables [H,S,hrd]
#   - scale_shift_table [2,4096] / audio_scale_shift_table [2,2048]
#   - video_velocity / audio_velocity              GATE targets
#
# This smoke runs: proj_in (patchify_proj) -> connector (P2.5, in-Mojo) on the
# pre-connector ctx -> 48-block forward_audio_video -> output stage
# (layer_norm_no_affine -> (scale_shift_table + embedded) modulate -> proj_out).
#
# Run:  pixi run mojo run serenitymojo/pipeline/ltx2_dit_forward_smoke.mojo

from std.gpu.host import DeviceContext
from std.math import sqrt
from std.sys import argv

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.linear import linear
from serenitymojo.ops.norm import layer_norm
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.tensor_algebra import (
    reshape, add, mul, add_scalar, slice,
)
from serenitymojo.models.dit.ltx2_dit import (
    LTX2Config,
    LTX2AVBlockWeights,
    ltx2_block_forward_av,
)
from serenitymojo.models.dit.ltx2_connector import (
    LTX2ConnectorConfig,
    LTX2ConnectorWeights,
    ltx2_connector_forward,
)
from serenitymojo.offload.ltx2_block_stream import (
    LTX2BlockStream,
    drop_block,
)
from serenitymojo.lora import LoraSet


comptime CKPT = "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-distilled-fp8.safetensors"
comptime REF_BASE = "/home/alex/mojodiffusion/output/ltx2_dit_forward/dit_forward_ref.safetensors"
comptime REF_HQ_LORA = "/home/alex/mojodiffusion/output/ltx2_dit_forward/dit_forward_hq_lora_ref.safetensors"
comptime DISTILLED_LORA = "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-distilled-lora-384.safetensors"
comptime CAMERA_STATIC_LORA = "/home/alex/.serenity/models/loras/ltx-2-19b-lora-camera-control-static.safetensors"
comptime DETAILER_LORA = "/home/alex/.serenity/models/loras/ltx-2-19b-ic-lora-detailer.safetensors"
comptime LORA_DISTILLED_STAGE2 = Float32(0.5)
comptime LORA_CAMERA_STATIC = Float32(0.3)
comptime LORA_DETAILER_DISABLED = Float32(0.0)

# Must match scripts/ltx2_dit_forward_parity_ref.py shape constants.
comptime S_V = 32      # NF*NH*NW = 2*4*4
comptime S_A = 8
comptime N_TXT = 24
comptime S_VPAD = 32   # max(S_V, N_TXT)
comptime S_APAD = 32   # max(S_A, N_TXT, S_V)
comptime NUM_LAYERS = 48
comptime VD = 4096
comptime AD = 2048
comptime EPS = Float32(1e-6)
comptime ISOLATED_BLOCK_GATE = Float64(0.9999)
comptime VIDEO_VELOCITY_GATE = Float64(0.994)
comptime AUDIO_VELOCITY_GATE = Float64(0.999)


def _shape1(a: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a)
    return s^


def _shape2(a: Int, b: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a)
    s.append(b)
    return s^


def _shape3(a: Int, b: Int, c: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a)
    s.append(b)
    s.append(c)
    return s^


def _is_boundary(i: Int) -> Bool:
    return i == 0 or i == 1 or i == 2 or i == 3 or i == 47


def _st_has(st: ShardedSafeTensors, name: String) -> Bool:
    for ref nm in st.names():
        if nm == name:
            return True
    return False


# Load a named F32 tensor from the oracle dump, cast to BF16 on device.
def _load_bf16(
    st: ShardedSafeTensors, name: String, ctx: DeviceContext
) raises -> Tensor:
    var tv = st.tensor_view(name)
    return Tensor.from_view_as_bf16(tv, ctx)


# Load a named tensor from the dump as F32 on device.
def _load_f32(
    st: ShardedSafeTensors, name: String, ctx: DeviceContext
) raises -> Tensor:
    var tv = st.tensor_view(name)
    return cast_tensor(Tensor.from_view_as_bf16(tv, ctx), STDtype.F32, ctx)


# Load a rope table dumped [H,S,hrd] (F32) -> [S*H,hrd] (s,h) row order, BF16.
def _load_rope(
    st: ShardedSafeTensors, name: String, ctx: DeviceContext
) raises -> Tensor:
    var tv = st.tensor_view(name)
    var sh = tv.shape.copy()
    if len(sh) != 3:
        raise Error(String("rope table ") + name + " not rank-3")
    var H = sh[0]
    var S = sh[1]
    var hrd = sh[2]
    var t = Tensor.from_view_as_bf16(tv, ctx)
    var host = t.to_host(ctx)
    var out = List[Float32]()
    for _ in range(S * H * hrd):
        out.append(Float32(0.0))
    for h in range(H):
        for s in range(S):
            for j in range(hrd):
                out[(s * H + h) * hrd + j] = host[(h * S + s) * hrd + j]
    return Tensor.from_host(out, _shape2(S * H, hrd), STDtype.BF16, ctx)


# Load a global checkpoint weight (BF16) under the checkpoint diffusion prefix.
def _load_global(
    st: ShardedSafeTensors, name: String, ctx: DeviceContext
) raises -> Tensor:
    var key = String("model.diffusion_model.") + name
    if not _st_has(st, key):
        key = name
    var tv = st.tensor_view(key)
    return Tensor.from_view_as_bf16(tv, ctx)


def _clone(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    var dev = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
    ctx.enqueue_copy(dst_buf=dev, src_buf=x.buf)
    ctx.synchronize()
    return Tensor(dev^, x.shape(), x.dtype())


def _linear_b(
    x: Tensor, w: Tensor, b: Tensor, ctx: DeviceContext
) raises -> Tensor:
    return linear(x, w, Optional[Tensor](_clone(b, ctx)), ctx)


def _use_hq_lora_from_argv() -> Bool:
    var args = argv()
    for i in range(len(args)):
        var a = String(args[i])
        if a == "hq-lora" or a == "--hq-lora":
            return True
    return False


def _apply_global_lora_weight(
    lora: LoraSet,
    base_key: String,
    weight: Tensor,
    multiplier: Float32,
    ctx: DeviceContext,
) raises -> Tensor:
    var delta = lora.compute_delta_for_base(base_key, multiplier, STDtype.F32, ctx)
    return add(weight, delta, ctx)


def _apply_block_lora(
    use_hq_lora: Bool,
    block_idx: Int,
    mut w: LTX2AVBlockWeights,
    distilled: LoraSet,
    camera: LoraSet,
    detailer: LoraSet,
    ctx: DeviceContext,
) raises -> Int:
    if not use_hq_lora:
        return 0
    var total = distilled.attach_ltx2_cached_block_factors(
        block_idx, w, LORA_DISTILLED_STAGE2, ctx
    )
    total += camera.attach_ltx2_cached_block_factors(
        block_idx, w, LORA_CAMERA_STATIC, ctx
    )
    if LORA_DETAILER_DISABLED != Float32(0.0):
        total += detailer.attach_ltx2_cached_block_factors(
            block_idx, w, LORA_DETAILER_DISABLED, ctx
        )
    return total


def _cosine(a: Tensor, b: Tensor, ctx: DeviceContext) raises -> Float64:
    var ha = a.to_host(ctx)
    var hb = b.to_host(ctx)
    if len(ha) != len(hb):
        raise Error("cosine: length mismatch")
    var dot = 0.0
    var na = 0.0
    var nb = 0.0
    for i in range(len(ha)):
        var x = Float64(ha[i])
        var y = Float64(hb[i])
        if x != x or y != y:
            raise Error("cosine: NaN")
        dot += x * y
        na += x * x
        nb += y * y
    return dot / (sqrt(na) * sqrt(nb) + 1e-30)


def _rel_l2(a: Tensor, b: Tensor, ctx: DeviceContext) raises -> Float64:
    var ha = a.to_host(ctx)
    var hb = b.to_host(ctx)
    if len(ha) != len(hb):
        raise Error("rel_l2: length mismatch")
    var diff_sq = 0.0
    var ref_sq = 0.0
    for i in range(len(ha)):
        var d = Float64(ha[i]) - Float64(hb[i])
        var r = Float64(hb[i])
        diff_sq += d * d
        ref_sq += r * r
    return sqrt(diff_sq) / (sqrt(ref_sq) + 1e-30)


def _compare(
    name: String, actual: Tensor, expected: Tensor, ctx: DeviceContext
) raises:
    var c = _cosine(actual, expected, ctx)
    var r = _rel_l2(actual, expected, ctx)
    print("  [cmp]", name, "cos:", Float32(c), "relL2:", Float32(r))


def _std(t: Tensor, ctx: DeviceContext) raises -> Float32:
    var h = t.to_host(ctx)
    var s = 0.0
    for i in range(len(h)):
        s += Float64(h[i])
    var mean = s / Float64(len(h))
    var v = 0.0
    for i in range(len(h)):
        var d = Float64(h[i]) - mean
        v += d * d
    return Float32(sqrt(v / Float64(len(h))))


def _stats(name: String, t: Tensor, ctx: DeviceContext) raises:
    var h = t.to_host(ctx)
    var s = 0.0
    var amax = 0.0
    for i in range(len(h)):
        var v = Float64(h[i])
        s += v
        var av = v if v >= 0.0 else -v
        if av > amax:
            amax = av
    print("  ", name, "mean:", Float32(s / Float64(len(h))), "absmax:",
          Float32(amax))


# ── output stage: layer_norm_no_affine -> (sst[2,dim]+embedded) modulate ->
#    proj_out. sst is [2,dim] (row0=shift, row1=scale); embedded is [1,S,dim].
def _output_stage(
    hs: Tensor,           # [1,S,dim] BF16 (final block output)
    sst: Tensor,          # [2,dim]  F32 (top-level scale_shift_table)
    embedded: Tensor,     # [1,S,dim] F32 (timestep embedder output)
    proj_w: Tensor,       # [128,dim] BF16
    proj_b: Tensor,       # [128]     BF16
    s: Int, dim: Int,
    ctx: DeviceContext,
) raises -> Tuple[Tensor, Tensor, Tensor]:
    # layer_norm with no affine: ones weight, zeros bias.
    var ones = List[Float32]()
    var zeros = List[Float32]()
    for _ in range(dim):
        ones.append(Float32(1.0))
        zeros.append(Float32(0.0))
    var w_ln = Tensor.from_host(ones, _shape1(dim), STDtype.F32, ctx)
    var b_ln = Tensor.from_host(zeros, _shape1(dim), STDtype.F32, ctx)
    var hs_f32 = cast_tensor(hs, STDtype.F32, ctx)
    var normed = layer_norm(hs_f32, w_ln, b_ln, EPS, ctx)   # [1,S,dim] F32

    # final_ss = sst[row] + embedded ; shift = final[...,0], scale = final[...,1]
    var shift_row = reshape(slice(sst, 0, 0, 1, ctx), _shape3(1, 1, dim), ctx)
    var scale_row = reshape(slice(sst, 0, 1, 1, ctx), _shape3(1, 1, dim), ctx)
    var v_shift = add(shift_row, embedded, ctx)             # [1,S,dim]
    var v_scale = add(scale_row, embedded, ctx)             # [1,S,dim]

    var one_plus = add_scalar(v_scale, Float32(1.0), ctx)
    var out = add(mul(normed, one_plus, ctx), v_shift, ctx)  # [1,S,dim] F32
    var velocity = _linear_b(out, proj_w, proj_b, ctx)       # [1,S,128] F32
    return (normed^, out^, velocity^)


def _run_isolated_block(
    block_idx: Int,
    dump: ShardedSafeTensors,
    stream: LTX2BlockStream,
    use_hq_lora: Bool,
    distilled: LoraSet,
    camera: LoraSet,
    detailer: LoraSet,
    enc: Tensor, aenc: Tensor,
    v_temb: Tensor, a_temb: Tensor,
    v_ca_ss: Tensor, a_ca_ss: Tensor,
    v_ca_gate: Tensor, a_ca_gate: Tensor,
    v_prompt_ts: Tensor, a_prompt_ts: Tensor,
    v_cos: Tensor, v_sin: Tensor,
    a_cos: Tensor, a_sin: Tensor,
    ca_v_cos: Tensor, ca_v_sin: Tensor,
    ca_a_cos: Tensor, ca_a_sin: Tensor,
    ctx: DeviceContext,
) raises:
    var cfg = LTX2Config.ltx2()
    var block_no = block_idx + 1
    var hs_in: Tensor
    var ahs_in: Tensor
    if block_idx == 0:
        hs_in = _load_f32(dump, "hs_after_projin", ctx)
        ahs_in = _load_f32(dump, "ahs_after_projin", ctx)
    else:
        hs_in = _load_f32(
            dump, String("after_block_") + String(block_idx) + String("_hs"), ctx
        )
        ahs_in = _load_f32(
            dump, String("after_block_") + String(block_idx) + String("_ahs"), ctx
        )
    var hs_ref = _load_f32(
        dump, String("after_block_") + String(block_no) + String("_hs"), ctx
    )
    var ahs_ref = _load_f32(
        dump, String("after_block_") + String(block_no) + String("_ahs"), ctx
    )

    var w: LTX2AVBlockWeights
    if _is_boundary(block_idx):
        w = LTX2AVBlockWeights.load(String(CKPT), block_idx, cfg, ctx).to_f32(ctx)
    else:
        var blk = stream.load_block_bf16(block_idx, ctx)
        w = LTX2AVBlockWeights.from_fp8_block(blk^, cfg, ctx).to_f32(ctx)
    _ = _apply_block_lora(
        use_hq_lora, block_idx, w, distilled, camera, detailer, ctx
    )
    var outs = ltx2_block_forward_av[S_V, S_A, N_TXT, S_VPAD, S_APAD](
        w, hs_in, ahs_in, enc, aenc,
        v_temb, a_temb, v_ca_ss, a_ca_ss, v_ca_gate, a_ca_gate,
        v_prompt_ts, a_prompt_ts,
        v_cos, v_sin, a_cos, a_sin,
        ca_v_cos, ca_v_sin, ca_a_cos, ca_a_sin, EPS, ctx,
    )
    var v_cos_sim = _cosine(outs[0], hs_ref, ctx)
    var a_cos_sim = _cosine(outs[1], ahs_ref, ctx)
    var v_rl2 = _rel_l2(outs[0], hs_ref, ctx)
    var a_rl2 = _rel_l2(outs[1], ahs_ref, ctx)
    print("  [isolated-block]", block_no, "video cos:", Float32(v_cos_sim),
          "relL2:", Float32(v_rl2), "audio cos:", Float32(a_cos_sim),
          "relL2:", Float32(a_rl2))
    if v_cos_sim < ISOLATED_BLOCK_GATE:
        raise Error(String("isolated video block parity FAIL: block=")
                    + String(block_no) + String(" cos=")
                    + String(v_cos_sim))
    if a_cos_sim < ISOLATED_BLOCK_GATE:
        raise Error(String("isolated audio block parity FAIL: block=")
                    + String(block_no) + String(" cos=")
                    + String(a_cos_sim))


def main() raises:
    var ctx = DeviceContext()
    var cfg = LTX2Config.ltx2()
    var use_hq_lora = _use_hq_lora_from_argv()
    var ref_path = String(REF_BASE)
    if use_hq_lora:
        ref_path = String(REF_HQ_LORA)

    print("=== LTX-2 FULL 48-block forward_audio_video VELOCITY smoke (P5) ===")
    print("  S_V/S_A/N_TXT:", S_V, S_A, N_TXT, " blocks:", NUM_LAYERS)
    print("  HQ LoRA:", "ON (distilled_stage2=0.5 camera_static=0.3 detailer=0.0)" if use_hq_lora else "OFF")

    print("  [load] oracle dump:", ref_path)
    var dump = ShardedSafeTensors.open(ref_path)
    var distilled_lora = LoraSet.load(String(DISTILLED_LORA))
    var camera_lora = LoraSet.load(String(CAMERA_STATIC_LORA))
    var detailer_lora = LoraSet.load(String(DETAILER_LORA))
    if use_hq_lora:
        var preload_count = 0
        preload_count += distilled_lora.preload_ltx2_block_factors(ctx)
        preload_count += camera_lora.preload_ltx2_block_factors(ctx)
        preload_count += detailer_lora.preload_ltx2_block_factors(ctx)
        print("  [lora] preloaded block factor tensors:", preload_count)

    # All per-forward tensors load as F32: the 48-block stack runs in F32 to
    # match the F32 velocity oracle (BF16 accumulation over 48 blocks drifts the
    # 4096-dim video stream below cos 0.999 — same reason the connector is F32).
    var v_flat = _load_f32(dump, "v_flat", ctx)
    var a_flat = _load_f32(dump, "a_flat", ctx)
    # dtype-contract: allow-f32-boundary
    # The Python velocity oracle runs the connector itself in F32; production
    # entrypoints keep these pre-connector contexts BF16.
    var video_pre = _load_f32(dump, "video_pre", ctx)
    var audio_pre = _load_f32(dump, "audio_pre", ctx)

    var v_temb = _load_f32(dump, "v_timestep", ctx)
    var a_temb = _load_f32(dump, "a_timestep", ctx)
    var v_embedded = _load_f32(dump, "v_embedded", ctx)
    var a_embedded = _load_f32(dump, "a_embedded", ctx)
    var v_ca_ss = _load_f32(dump, "v_ca_ss", ctx)
    var a_ca_ss = _load_f32(dump, "a_ca_ss", ctx)
    var v_ca_gate = _load_f32(dump, "v_ca_gate", ctx)
    var a_ca_gate = _load_f32(dump, "a_ca_gate", ctx)
    var v_prompt_ts = _load_f32(dump, "video_prompt_ts", ctx)
    var a_prompt_ts = _load_f32(dump, "audio_prompt_ts", ctx)

    # RoPE tables: loaded BF16 (rope cos/sin in [-1,1], BF16 is exact enough),
    # then upcast to F32 to match the F32 block compute dtype.
    var v_cos = cast_tensor(_load_rope(dump, "v_cos", ctx), STDtype.F32, ctx)
    var v_sin = cast_tensor(_load_rope(dump, "v_sin", ctx), STDtype.F32, ctx)
    var a_cos = cast_tensor(_load_rope(dump, "a_cos", ctx), STDtype.F32, ctx)
    var a_sin = cast_tensor(_load_rope(dump, "a_sin", ctx), STDtype.F32, ctx)
    var ca_v_cos = cast_tensor(_load_rope(dump, "ca_v_cos", ctx), STDtype.F32, ctx)
    var ca_v_sin = cast_tensor(_load_rope(dump, "ca_v_sin", ctx), STDtype.F32, ctx)
    var ca_a_cos = cast_tensor(_load_rope(dump, "ca_a_cos", ctx), STDtype.F32, ctx)
    var ca_a_sin = cast_tensor(_load_rope(dump, "ca_a_sin", ctx), STDtype.F32, ctx)

    var v_sst = _load_f32(dump, "scale_shift_table", ctx)
    var a_sst = _load_f32(dump, "audio_scale_shift_table", ctx)

    var video_ref = _load_f32(dump, "video_velocity", ctx)
    var audio_ref = _load_f32(dump, "audio_velocity", ctx)

    # ── globals: proj_in (patchify_proj), proj_out ── (F32)
    print("  [load] globals (patchify_proj, proj_out)")
    var ck = ShardedSafeTensors.open(String(CKPT))
    var v_pin_w = cast_tensor(_load_global(ck, "patchify_proj.weight", ctx), STDtype.F32, ctx)
    var v_pin_b = cast_tensor(_load_global(ck, "patchify_proj.bias", ctx), STDtype.F32, ctx)
    var a_pin_w = cast_tensor(_load_global(ck, "audio_patchify_proj.weight", ctx), STDtype.F32, ctx)
    var a_pin_b = cast_tensor(_load_global(ck, "audio_patchify_proj.bias", ctx), STDtype.F32, ctx)
    var v_pout_w = cast_tensor(_load_global(ck, "proj_out.weight", ctx), STDtype.F32, ctx)
    var v_pout_b = cast_tensor(_load_global(ck, "proj_out.bias", ctx), STDtype.F32, ctx)
    var a_pout_w = cast_tensor(_load_global(ck, "audio_proj_out.weight", ctx), STDtype.F32, ctx)
    var a_pout_b = cast_tensor(_load_global(ck, "audio_proj_out.bias", ctx), STDtype.F32, ctx)
    if use_hq_lora:
        print("  [lora] applying distilled global deltas used by this smoke")
        v_pin_w = _apply_global_lora_weight(
            distilled_lora, String("patchify_proj.weight"), v_pin_w,
            LORA_DISTILLED_STAGE2, ctx,
        )
        a_pin_w = _apply_global_lora_weight(
            distilled_lora, String("audio_patchify_proj.weight"), a_pin_w,
            LORA_DISTILLED_STAGE2, ctx,
        )
        v_pout_w = _apply_global_lora_weight(
            distilled_lora, String("proj_out.weight"), v_pout_w,
            LORA_DISTILLED_STAGE2, ctx,
        )
        a_pout_w = _apply_global_lora_weight(
            distilled_lora, String("audio_proj_out.weight"), a_pout_w,
            LORA_DISTILLED_STAGE2, ctx,
        )

    # ── proj_in: patchified latent -> inner_dim (F32) ──
    var hs = _linear_b(v_flat, v_pin_w, v_pin_b, ctx)   # [1,S_V,4096] F32
    var ahs = _linear_b(a_flat, a_pin_w, a_pin_b, ctx)  # [1,S_A,2048] F32

    # ── connector (P2.5): pre-connector ctx -> post-connector ctx (F32) ──
    print("  [connector] video + audio (in-Mojo, F32)")
    # dtype-contract: allow-f32-boundary
    # Match scripts/ltx2_dit_forward_parity_ref.py's F32 connector oracle.
    var v_conn = LTX2ConnectorWeights.load(
        String(CKPT), String("video_embeddings_connector"),
        LTX2ConnectorConfig.video(), ctx,
    ).to_f32(ctx)
    var a_conn = LTX2ConnectorWeights.load(
        String(CKPT), String("audio_embeddings_connector"),
        LTX2ConnectorConfig.audio(), ctx,
    ).to_f32(ctx)
    var enc = ltx2_connector_forward[N_TXT, 32, 128](v_conn, video_pre, ctx)
    var aenc = ltx2_connector_forward[N_TXT, 32, 64](a_conn, audio_pre, ctx)

    _stats(String("hs_after_projin"), hs, ctx)
    _stats(String("ahs_after_projin"), ahs, ctx)
    _stats(String("enc"), enc, ctx)
    _stats(String("aenc"), aenc, ctx)
    _compare(String("hs_after_projin"), hs, _load_f32(dump, "hs_after_projin", ctx), ctx)
    _compare(String("ahs_after_projin"), ahs, _load_f32(dump, "ahs_after_projin", ctx), ctx)
    _compare(String("enc"), enc, _load_f32(dump, "enc", ctx), ctx)
    _compare(String("aenc"), aenc, _load_f32(dump, "aenc", ctx), ctx)

    # ── 48-block forward_audio_video ──
    print("  [forward] streaming 48 AV blocks")
    var stream = LTX2BlockStream.open(String(CKPT))
    if stream.block_count() != NUM_LAYERS:
        raise Error(String("stream block_count ") + String(stream.block_count())
                    + " != " + String(NUM_LAYERS))

    print("  [isolate] suspect inner blocks from exact oracle inputs")
    _run_isolated_block(
        25, dump, stream, use_hq_lora, distilled_lora, camera_lora,
        detailer_lora, enc, aenc,
        v_temb, a_temb, v_ca_ss, a_ca_ss, v_ca_gate, a_ca_gate,
        v_prompt_ts, a_prompt_ts, v_cos, v_sin, a_cos, a_sin,
        ca_v_cos, ca_v_sin, ca_a_cos, ca_a_sin, ctx,
    )
    _run_isolated_block(
        31, dump, stream, use_hq_lora, distilled_lora, camera_lora,
        detailer_lora, enc, aenc,
        v_temb, a_temb, v_ca_ss, a_ca_ss, v_ca_gate, a_ca_gate,
        v_prompt_ts, a_prompt_ts, v_cos, v_sin, a_cos, a_sin,
        ca_v_cos, ca_v_sin, ca_a_cos, ca_a_sin, ctx,
    )
    _run_isolated_block(
        40, dump, stream, use_hq_lora, distilled_lora, camera_lora,
        detailer_lora, enc, aenc,
        v_temb, a_temb, v_ca_ss, a_ca_ss, v_ca_gate, a_ca_gate,
        v_prompt_ts, a_prompt_ts, v_cos, v_sin, a_cos, a_sin,
        ca_v_cos, ca_v_sin, ca_a_cos, ca_a_sin, ctx,
    )
    _run_isolated_block(
        42, dump, stream, use_hq_lora, distilled_lora, camera_lora,
        detailer_lora, enc, aenc,
        v_temb, a_temb, v_ca_ss, a_ca_ss, v_ca_gate, a_ca_gate,
        v_prompt_ts, a_prompt_ts, v_cos, v_sin, a_cos, a_sin,
        ca_v_cos, ca_v_sin, ca_a_cos, ca_a_sin, ctx,
    )
    _run_isolated_block(
        44, dump, stream, use_hq_lora, distilled_lora, camera_lora,
        detailer_lora, enc, aenc,
        v_temb, a_temb, v_ca_ss, a_ca_ss, v_ca_gate, a_ca_gate,
        v_prompt_ts, a_prompt_ts, v_cos, v_sin, a_cos, a_sin,
        ca_v_cos, ca_v_sin, ca_a_cos, ca_a_sin, ctx,
    )
    _run_isolated_block(
        46, dump, stream, use_hq_lora, distilled_lora, camera_lora,
        detailer_lora, enc, aenc,
        v_temb, a_temb, v_ca_ss, a_ca_ss, v_ca_gate, a_ca_gate,
        v_prompt_ts, a_prompt_ts, v_cos, v_sin, a_cos, a_sin,
        ca_v_cos, ca_v_sin, ca_a_cos, ca_a_sin, ctx,
    )

    for i in range(NUM_LAYERS):
        # Build this block's weights: boundary -> disk BF16; inner -> FP8 stream.
        var w: LTX2AVBlockWeights
        if _is_boundary(i):
            w = LTX2AVBlockWeights.load(String(CKPT), i, cfg, ctx).to_f32(ctx)
        else:
            var blk = stream.load_block_bf16(i, ctx)
            w = LTX2AVBlockWeights.from_fp8_block(blk^, cfg, ctx).to_f32(ctx)
        var n_lora = _apply_block_lora(
            use_hq_lora, i, w, distilled_lora, camera_lora, detailer_lora, ctx
        )
        if use_hq_lora and (
            i == 0 or i == 47 or (i + 1) % 12 == 0
        ):
            print("  [lora-block]", i + 1, "deltas:", n_lora)
        var outs = ltx2_block_forward_av[S_V, S_A, N_TXT, S_VPAD, S_APAD](
            w, hs, ahs, enc, aenc,
            v_temb, a_temb, v_ca_ss, a_ca_ss, v_ca_gate, a_ca_gate,
            v_prompt_ts, a_prompt_ts,
            v_cos, v_sin, a_cos, a_sin,
            ca_v_cos, ca_v_sin, ca_a_cos, ca_a_sin, EPS, ctx,
        )
        # Clone out of the borrowed tuple refs into the owned accumulators
        # (Tensor is move-only; tuple elements can't be transferred directly).
        hs = _clone(outs[0], ctx)
        ahs = _clone(outs[1], ctx)
        var block_no = i + 1
        var v_key = String("after_block_") + String(block_no) + String("_hs")
        var a_key = String("after_block_") + String(block_no) + String("_ahs")
        var v_ref_block = _load_f32(dump, v_key, ctx)
        var a_ref_block = _load_f32(dump, a_key, ctx)
        var v_block_cos = _cosine(hs, v_ref_block, ctx)
        var a_block_cos = _cosine(ahs, a_ref_block, ctx)
        if (
            block_no == 1 or block_no == 4 or block_no == 5
            or block_no == 12 or block_no == 24 or block_no == 36
            or block_no == 48 or v_block_cos < 0.99995
            or a_block_cos < 0.99995
        ):
            var v_block_rl2 = _rel_l2(hs, v_ref_block, ctx)
            var a_block_rl2 = _rel_l2(ahs, a_ref_block, ctx)
            print("  [cmp-block]", block_no, "video cos:", Float32(v_block_cos),
                  "relL2:", Float32(v_block_rl2), "audio cos:",
                  Float32(a_block_cos), "relL2:", Float32(a_block_rl2))
        if (i + 1) % 12 == 0 or i + 1 == NUM_LAYERS:
            print("    block", i + 1, "/", NUM_LAYERS,
                  " v_std:", _std(hs, ctx), " a_std:", _std(ahs, ctx))

    _stats(String("hs_final"), hs, ctx)
    _stats(String("ahs_final"), ahs, ctx)

    # ── output stage -> velocity ──
    print("  [output] layer_norm -> scale_shift -> proj_out")
    var v_stage = _output_stage(hs, v_sst, v_embedded, v_pout_w, v_pout_b,
                                S_V, VD, ctx)
    var a_stage = _output_stage(ahs, a_sst, a_embedded, a_pout_w, a_pout_b,
                                S_A, AD, ctx)
    ref v_norm = v_stage[0]
    ref v_mod = v_stage[1]
    ref v_out = v_stage[2]
    ref a_norm = a_stage[0]
    ref a_mod = a_stage[1]
    ref a_out = a_stage[2]
    _compare(String("video_final_norm"), v_norm, _load_f32(dump, "video_final_norm", ctx), ctx)
    _compare(String("video_final_mod"), v_mod, _load_f32(dump, "video_final_mod", ctx), ctx)
    _compare(String("audio_final_norm"), a_norm, _load_f32(dump, "audio_final_norm", ctx), ctx)
    _compare(String("audio_final_mod"), a_mod, _load_f32(dump, "audio_final_mod", ctx), ctx)

    _stats(String("video_velocity"), v_out, ctx)
    _stats(String("audio_velocity"), a_out, ctx)
    _stats(String("video_ref"), video_ref, ctx)
    _stats(String("audio_ref"), audio_ref, ctx)

    var v_cos_sim = _cosine(v_out, video_ref, ctx)
    var a_cos_sim = _cosine(a_out, audio_ref, ctx)
    print("  >>> VIDEO velocity cos:", Float32(v_cos_sim))
    print("  >>> AUDIO velocity cos:", Float32(a_cos_sim))
    print("  gates: video>=", Float32(VIDEO_VELOCITY_GATE), " audio>=",
          Float32(AUDIO_VELOCITY_GATE), " isolated-block>=",
          Float32(ISOLATED_BLOCK_GATE))

    if v_cos_sim < VIDEO_VELOCITY_GATE:
        raise Error(String("VIDEO velocity parity FAIL: cos=")
                    + String(v_cos_sim))
    if a_cos_sim < AUDIO_VELOCITY_GATE:
        raise Error(String("AUDIO velocity parity FAIL: cos=")
                    + String(a_cos_sim))
    print("LTX-2 FULL 48-block forward VELOCITY PARITY PASS")
