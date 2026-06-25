# models/dit/parity/krea2_block_tf_real_probe.mojo — CHUNK 7b bug-localization gate.
# TEACHER-FORCED per-block parity with REAL raw.safetensors weights, RESIDENT
# (one block's weights at a time, NO offload). Reuses the EriTrainer per-block taps.
#
# For each chosen block i (0, 13, 19): run krea2_single_stream_block on the
# reference's BYTE-IDENTICAL x_in[i] (padded to 256 + the pad-to-256 mask, the
# reference's own forward conditions), slice the 23 real-token output, compare vs
# the reference x_out[i]. METRIC: cos is MAGNITUDE-BLIND (it hid the Rust outlier-
# channel bug), so we ALSO report per-channel rel-diff on ch 2569 & 3389 (the Rust
# divergence channels) + the count of channels with rel-diff > 1%. FAIL-LOUD if
# any block's ch-2569/3389 rel-diff exceeds 1% OR overall cos < 0.999.
#
# Oracle: krea2_block_tf_real_oracle.safetensors — per block i: w.blocks.i.* (13
# REAL weights), x_in.i / x_out.i [1,23,6144] f32; shared tvec [1,1,36864] f32,
# cos/sin [256,64] f32 (extracted from the reference freqs tap).
#
# Run: cd /home/alex/mojodiffusion && \
#   pixi run mojo run -I . serenitymojo/models/dit/parity/krea2_block_tf_real_probe.mojo
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.math import sqrt
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness
from serenitymojo.ops.tensor_algebra import reshape, concat, zeros_device, slice
from serenitymojo.models.dit.krea2_dit import (
    krea2_single_stream_block,
    build_krea2_text_mask,
)


comptime ORACLE = "/home/alex/mojodiffusion/serenitymojo/models/dit/parity/krea2_block_tf_real_oracle.safetensors"

comptime FEATURES = 6144
comptime HEADS = 48
comptime KVHEADS = 12
comptime HEADDIM = 128
comptime LREAL = 23
comptime LPAD = 256
comptime CH_A = 2569   # Rust outlier channel
comptime CH_B = 3389   # Rust outlier channel


def _pad_to_256(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    """Pad x [1, LREAL, F] -> [1, LPAD, F] with zeros on the seq axis."""
    var pshape = List[Int]()
    pshape.append(1); pshape.append(LPAD - LREAL); pshape.append(FEATURES)
    var pad = zeros_device(pshape^, x.dtype(), ctx)
    return concat(1, ctx, x, pad)


def _w(st: ShardedSafeTensors, blk: Int, suf: String, ctx: DeviceContext) raises -> Tensor:
    """Load a block weight, preserving stored dtype (bf16 proj / f32 scale)."""
    return Tensor.from_view(st.tensor_view("w.blocks." + String(blk) + "." + suf), ctx)


def _wf32(st: ShardedSafeTensors, blk: Int, suf: String, ctx: DeviceContext) raises -> Tensor:
    """Load a block scale weight as F32."""
    return Tensor.from_view_as_f32(st.tensor_view("w.blocks." + String(blk) + "." + suf), ctx)


def _per_channel_report(
    name: String,
    got: List[Float32],     # [LREAL*FEATURES] my block output (sliced to 23)
    xref: List[Float32],    # [LREAL*FEATURES] reference x_out (23)
) raises -> Bool:
    """Per-channel rel-diff report. Returns True if PASS (ch A/B rel < 1% AND
    overall cos >= 0.999). Channel c rel = max_t|got[t,c]-xref[t,c]| / (max_t|xref[t,c]| + eps)."""
    # overall cosine (magnitude-blind — reported but not the only bar)
    var dot: Float64 = 0.0
    var na: Float64 = 0.0
    var nb: Float64 = 0.0
    for i in range(LREAL * FEATURES):
        var a = Float64(got[i])
        var b = Float64(xref[i])
        dot += a * b
        na += a * a
        nb += b * b
    var cos = dot / (sqrt(na) * sqrt(nb) + 1e-30)

    # per-channel max-abs-diff and reference channel magnitude
    var eps = Float32(1e-6)
    var rel_a: Float32 = 0.0
    var rel_b: Float32 = 0.0
    var n_over_1pct = 0
    var max_rel: Float32 = 0.0
    var max_rel_ch = 0
    for c in range(FEATURES):
        var maxdiff: Float32 = 0.0
        var maxref: Float32 = 0.0
        for tk in range(LREAL):
            var idx = tk * FEATURES + c
            var d = abs(got[idx] - xref[idx])
            if d > maxdiff:
                maxdiff = d
            var r = abs(xref[idx])
            if r > maxref:
                maxref = r
        var rel = maxdiff / (maxref + eps)
        if c == CH_A:
            rel_a = rel
        if c == CH_B:
            rel_b = rel
        if rel > Float32(0.01):
            n_over_1pct += 1
        if rel > max_rel:
            max_rel = rel
            max_rel_ch = c

    print(name, "cos=", cos,
          " ch", CH_A, "rel=", rel_a, " ch", CH_B, "rel=", rel_b,
          " #ch_rel>1%=", n_over_1pct, "/", FEATURES,
          " worst_ch=", max_rel_ch, " worst_rel=", max_rel)
    var pass_cos = cos >= 0.999
    var pass_a = rel_a <= Float32(0.01)
    var pass_b = rel_b <= Float32(0.01)
    return pass_cos and pass_a and pass_b


def _run_block(st: ShardedSafeTensors, blk: Int, ctx: DeviceContext) raises -> Bool:
    # teacher-forced input + reference output (23 real tokens). The reference
    # x_out is the MATH-backend block output (xout_math.N): the production taps
    # were generated with cuDNN, which == MATH at the block level (cos 0.99999),
    # and serenity's masked sdpa faithfully computes the MATH (additive) result —
    # so MATH-xout is the math-correct teacher-forced target.
    var x_in = Tensor.from_view_as_bf16(st.tensor_view("x_in." + String(blk)), ctx)  # bf16 [1,23,6144]
    var x_ref = Tensor.from_view_as_f32(st.tensor_view("xout_math." + String(blk)), ctx).to_host(ctx)

    # shared modulation vec + rope cos/sin (256) — loaded once per block (cheap).
    var tvec = Tensor.from_view_as_bf16(st.tensor_view("tvec"), ctx)  # bf16 [1,1,36864]
    var vshape = List[Int]()
    vshape.append(1); vshape.append(6 * FEATURES)
    var vec = reshape(tvec, vshape^, ctx)                            # [1, 36864]
    var cos = Tensor.from_view_as_f32(st.tensor_view("cos"), ctx)    # f32 [256,64]
    var sin = Tensor.from_view_as_f32(st.tensor_view("sin"), ctx)

    # pad x_in to 256 + build the pad-to-256 mask (keep = ones[0:23], zeros[23:256]).
    var x_pad = _pad_to_256(x_in, ctx)                               # [1,256,6144]
    var keep_host = List[Float32]()
    for i in range(LPAD):
        keep_host.append(Float32(1.0) if i < LREAL else Float32(0.0))
    var kshape = List[Int]()
    kshape.append(LPAD)
    var keep = Tensor.from_host(keep_host^, kshape^, STDtype.F32, ctx)
    var mask = build_krea2_text_mask(keep, HEADS, LPAD, ctx, STDtype.BF16)  # [1,48,256,256]

    # run the block at S=256 with the mask (the reference's own forward conditions).
    # mod.lin is F32 in raw.safetensors but bf16 in the reference's bf16 model
    # (model.to(bf16) casts it); the modulation add runs bf16 -> load it bf16.
    var mod_lin = Tensor.from_view_as_bf16(
        st.tensor_view("w.blocks." + String(blk) + ".mod.lin"), ctx
    )
    var y = krea2_single_stream_block[LPAD, HEADS, KVHEADS, HEADDIM](
        x_pad,
        vec,
        mod_lin,
        _wf32(st, blk, "prenorm.scale", ctx),
        _wf32(st, blk, "postnorm.scale", ctx),
        _w(st, blk, "attn.wq.weight", ctx),
        _w(st, blk, "attn.wk.weight", ctx),
        _w(st, blk, "attn.wv.weight", ctx),
        _w(st, blk, "attn.gate.weight", ctx),
        _w(st, blk, "attn.wo.weight", ctx),
        _wf32(st, blk, "attn.qknorm.qnorm.scale", ctx),
        _wf32(st, blk, "attn.qknorm.knorm.scale", ctx),
        _w(st, blk, "mlp.gate.weight", ctx),
        _w(st, blk, "mlp.up.weight", ctx),
        _w(st, blk, "mlp.down.weight", ctx),
        cos, sin,
        Optional[Tensor](mask^),
        ctx,
    )                                                               # [1,256,6144]

    # slice the 23 real tokens, compare per-channel.
    var y23 = slice(y, 1, 0, LREAL, ctx)                            # [1,23,6144]
    var got = y23.to_host(ctx)
    return _per_channel_report(String("block ") + String(blk), got, x_ref)


def main() raises:
    var ctx = DeviceContext()
    var st = ShardedSafeTensors.open(ORACLE)

    var all_pass = True
    var blocks = [0, 13, 19]
    for bi in range(len(blocks)):
        var ok = _run_block(st, blocks[bi], ctx)
        if not ok:
            all_pass = False

    if not all_pass:
        raise Error(
            "krea2 teacher-forced per-block FAILED: a block diverges from torch "
            "(overall cos < 0.999 OR ch 2569/3389 rel-diff > 1%) — Mojo bug localized above."
        )
    print("krea2 teacher-forced per-block: ALL BLOCKS MATCH torch (per-channel) — forward is faithful")
