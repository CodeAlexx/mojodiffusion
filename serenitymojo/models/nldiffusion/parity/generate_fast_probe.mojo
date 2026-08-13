# models/nldiffusion/parity/generate_fast_probe.mojo — SPEED variant of
# generate_probe.mojo (CHUNK F). Identical math; the ONLY change is the backbone
# weight source: instead of re-parsing all 34 layers (~15GB bf16) from the 6
# shards on EVERY step (`nldiff_backbone`), it parses them ONCE into a persistent
# pinned-host bf16 store (`build_nldiff_host_store`) and streams per-layer H2D
# with double-buffered prefetch (`nldiff_backbone_resident`).
#
# PROFILE-FIRST: runs 2 "before" steps with the original per-step-reload backbone
# (timed load-vs-compute split), then builds the store and runs the full 64-step
# loop with the resident backbone (timed split). Reports both.
#
# GATE: the final token grid MUST be BYTE-IDENTICAL to the committed
# parity/mojo_tokens.safetensors (key x0_img) — int-exact, 4096/4096, 0 diffs.
# A single differing token => the optimization changed the math => FAIL.
#
# Run (JIT; parses ~15GB once at start, then fast steps):
#   cd /home/alex/mojodiffusion && \
#     pixi run mojo run -I . serenitymojo/models/nldiffusion/parity/generate_fast_probe.mojo

from max.gpu.host import DeviceContext
from std.memory import ArcPointer
from std.time import perf_counter_ns

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.tensor_algebra import reshape, gather_rows
from serenitymojo.ops.norm import rms_norm
from serenitymojo.models.nldiffusion.backbone import (
    NLDiffConfig,
    nldiff_decoder_layer,
)
from serenitymojo.models.nldiffusion.backbone_stack import (
    nldiff_yarn_perhead,
    nldiff_backbone_resident,
    build_nldiff_host_store,
    NLDiffHostStore,
    _load_layer_weights,
    _load_bf16,
)
from serenitymojo.models.text_encoder.qwen3_encoder import _clone
from serenitymojo.models.nldiffusion.image_head import (
    nldiff_gen_embedding,
    nldiff_downsample_gen,
    nldiff_upsample_gen,
    nldiff_gen_predictor,
)
from serenitymojo.models.nldiffusion.sampler import (
    nldiff_num_transfer_tokens,
    nldiff_argmax_lastdim,
    nldiff_stratified_select,
)

comptime MODEL_DIR = "/mnt/disk1/models/NL-Diffusion-Image"
comptime PARITY_DIR = "/home/alex/mojodiffusion/serenitymojo/models/nldiffusion/parity"

comptime SEQ = 1141
comptime N_TOKENS = 4096
comptime N_SLOTS = 1024
comptime HID = 4096
comptime N_STEPS = 64
comptime SHIFT = 5
comptime RESERVE_ID = 18
comptime IMG_MASK_ID = 131073


def _load_f32(st: ShardedSafeTensors, name: String, ctx: DeviceContext) raises -> List[Float32]:
    var tv = st.tensor_view(name)
    return Tensor.from_view_as_f32(tv, ctx).to_host(ctx)


def _round_ints(xs: List[Float32]) -> List[Int]:
    var out = List[Int]()
    for i in range(len(xs)):
        var v = xs[i]
        if v >= 0.0:
            out.append(Int(v + 0.5))
        else:
            out.append(Int(v - 0.5))
    return out^


# ── streamed backbone with per-phase timing (the BEFORE reference) ────────────
def _stream_backbone_timed[
    S: Int
](
    seq: Tensor,
    model: ShardedSafeTensors,
    cfg: NLDiffConfig,
    cos_q: Tensor, sin_q: Tensor, cos_k: Tensor, sin_k: Tensor,
    ctx: DeviceContext,
    mut load_ns: Int,
    mut comp_ns: Int,
) raises -> Tensor:
    var hidden = _clone(seq, ctx)
    for i in range(cfg.num_layers):
        var a = perf_counter_ns()
        var w = _load_layer_weights(model, i, ctx)
        ctx.synchronize()
        var b = perf_counter_ns()
        var nh = nldiff_decoder_layer[S](
            hidden, w, cfg, cos_q, sin_q, cos_k, sin_k, ctx)
        ctx.synchronize()
        var c = perf_counter_ns()
        hidden = nh^
        load_ns += Int(b - a)
        comp_ns += Int(c - b)
        _ = w^
    var norm_w = _load_bf16(model, String("encoder.norm.weight"), ctx)
    return rms_norm(hidden, norm_w[], cfg.rms_norm_eps, ctx)


# ── head-pre: gen_embedding -> downsample -> splice into base seq ─────────────
def _head_pre(
    xt: List[Int],
    model: ShardedSafeTensors,
    ds_cw: Tensor, ds_nw: Tensor,
    base_host: List[Float32],
    is_gen: List[Int],
    ctx: DeviceContext,
) raises -> Tensor:
    var gen_emb = nldiff_gen_embedding(xt, model, ctx)
    var gen_ds = nldiff_downsample_gen[64, 64, HID](gen_emb, ds_cw, ds_nw, ctx)
    var gen_host = gen_ds.to_host(ctx)
    _ = gen_emb^
    _ = gen_ds^
    var seq_host = base_host.copy()
    for i in range(N_SLOTS):
        var pos = is_gen[i]
        var dst = pos * HID
        var src = i * HID
        for c in range(HID):
            seq_host[dst + c] = gen_host[src + c]
    return Tensor.from_host(seq_host, [1, SEQ, HID], STDtype.BF16, ctx)


# ── head-post: gather reserve rows -> upsample -> gen_predictor -> logits ─────
def _head_post_logits(
    hidden: Tensor,
    is_gen: List[Int],
    us_cw: Tensor, us_nw: Tensor, pred_w: Tensor,
    ctx: DeviceContext,
) raises -> Tensor:
    var hidden2d = reshape(hidden, [SEQ, HID], ctx)
    var gen_hid = gather_rows(hidden2d, is_gen, ctx)
    var gen_hid3 = reshape(gen_hid, [1, N_SLOTS, HID], ctx)
    _ = hidden2d^
    _ = gen_hid^
    var up = nldiff_upsample_gen[32, 32](gen_hid3, us_cw, us_nw, ctx)
    _ = gen_hid3^
    var up_flat = reshape(up, [N_TOKENS, HID], ctx)
    _ = up^
    var logits = nldiff_gen_predictor(up_flat, pred_w, ctx)
    _ = up_flat^
    return logits^


def main() raises:
    var ctx = DeviceContext()
    var cfg = NLDiffConfig.default()
    print("============= SPEED: NL-Diffusion resident-store generation loop =============")

    var model = ShardedSafeTensors.open(MODEL_DIR)
    var oracle1 = ShardedSafeTensors.open(String(PARITY_DIR) + "/oracle1.safetensors")
    var oracle2 = ShardedSafeTensors.open(String(PARITY_DIR) + "/oracle2_tokens.safetensors")
    var mojo_ref = ShardedSafeTensors.open(String(PARITY_DIR) + "/mojo_tokens.safetensors")

    var input_ids = _round_ints(_load_f32(oracle1, "embed_tokens.in", ctx))
    if len(input_ids) != SEQ:
        raise Error("input_ids len != 1141")
    var is_gen = List[Int]()
    for i in range(SEQ):
        if input_ids[i] == RESERVE_ID:
            is_gen.append(i)
    if len(is_gen) != N_SLOTS:
        raise Error("is_gen count != 1024")

    var embed_tv = model.tensor_view("encoder.embed_tokens.weight")
    var embed_w = Tensor.from_view(embed_tv, ctx)
    var base_emb = gather_rows(embed_w, input_ids, ctx)
    var base_host = base_emb.to_host(ctx)
    _ = base_emb^
    _ = embed_w^

    var ds_cw = Tensor.from_view(
        model.tensor_view("encoder.downsample_gen.downsample.conv.weight"), ctx)
    var ds_nw = Tensor.from_view(
        model.tensor_view("encoder.downsample_gen.downsample.norm.weight"), ctx)
    var us_cw = Tensor.from_view(
        model.tensor_view("encoder.upsample_gen.upsample.conv.weight"), ctx)
    var us_nw = Tensor.from_view(
        model.tensor_view("encoder.upsample_gen.upsample.norm.weight"), ctx)
    var pred_w = Tensor.from_view(
        model.tensor_view("encoder.gen_predictor.weight"), ctx)

    var qh = nldiff_yarn_perhead[SEQ](cfg.num_heads, ctx)
    var kh = nldiff_yarn_perhead[SEQ](cfg.num_kv_heads, ctx)

    var num_transfer = nldiff_num_transfer_tokens(N_TOKENS, N_STEPS, SHIFT)
    var order = _round_ints(_load_f32(oracle2, "unmask_order", ctx))

    # ═══════════════ BEFORE profile: 2 steps, per-step reload backbone ══════════
    print("\n[BEFORE] profiling 2 steps with per-step shard-reload backbone ...")
    var bx = List[Int]()
    for _i in range(N_TOKENS):
        bx.append(IMG_MASK_ID)
    var b_committed = 0
    var b_pre_ns = 0
    var b_load_ns = 0
    var b_comp_ns = 0
    var b_post_ns = 0
    var b_arg_ns = 0
    var b_step_ns = 0
    comptime N_BEFORE = 2
    for step in range(N_BEFORE):
        var s0 = perf_counter_ns()
        var seq = _head_pre(bx, model, ds_cw, ds_nw, base_host, is_gen, ctx)
        ctx.synchronize()
        var s1 = perf_counter_ns()
        var ld = 0
        var cp = 0
        var hidden = _stream_backbone_timed[SEQ](
            seq, model, cfg, qh[0], qh[1], kh[0], kh[1], ctx, ld, cp)
        _ = seq^
        ctx.synchronize()
        var s2 = perf_counter_ns()
        var logits = _head_post_logits(hidden, is_gen, us_cw, us_nw, pred_w, ctx)
        _ = hidden^
        ctx.synchronize()
        var s3 = perf_counter_ns()
        var x0 = nldiff_argmax_lastdim(logits, ctx)
        _ = logits^
        var s4 = perf_counter_ns()
        var nt = num_transfer[step]
        var sel = nldiff_stratified_select(order, b_committed, nt)
        for k in range(len(sel)):
            bx[sel[k]] = x0[sel[k]]
        b_committed += nt
        b_pre_ns += Int(s1 - s0)
        b_load_ns += ld
        b_comp_ns += cp
        b_post_ns += Int(s3 - s2)
        b_arg_ns += Int(s4 - s3)
        b_step_ns += Int(s4 - s0)
        print("  [BEFORE] step", step, "(", Float64(s4 - s0) / 1.0e6, "ms )")
    var bd = Float64(N_BEFORE)
    print("[BEFORE] avg per-step split (ms):")
    print("   head_pre (gen_emb+downsample+splice):", Float64(b_pre_ns) / bd / 1.0e6)
    print("   backbone WEIGHT-LOAD (H2D+parse):    ", Float64(b_load_ns) / bd / 1.0e6)
    print("   backbone COMPUTE:                    ", Float64(b_comp_ns) / bd / 1.0e6)
    print("   head_post (gather+upsample+predictor):", Float64(b_post_ns) / bd / 1.0e6)
    print("   argmax/sampler:                      ", Float64(b_arg_ns) / bd / 1.0e6)
    print("   TOTAL:                               ", Float64(b_step_ns) / bd / 1.0e6, "ms/step")

    # ═══════════════ Build the persistent pinned-host store ONCE ════════════════
    print("\n[STORE] building persistent pinned-host bf16 weight store (once) ...")
    var store_t0 = perf_counter_ns()
    var store = build_nldiff_host_store(model, cfg, ctx)
    var store_t1 = perf_counter_ns()
    print("[STORE] build took", Float64(store_t1 - store_t0) / 1.0e9, "s")

    # ═══════════════ AFTER: full 64-step loop with resident backbone ════════════
    print("\n[AFTER] running full", N_STEPS, "steps with resident backbone ...")
    var xt = List[Int]()
    for _i in range(N_TOKENS):
        xt.append(IMG_MASK_ID)
    var committed = 0
    var a_pre_ns = 0
    var a_bb_ns = 0
    var a_post_ns = 0
    var a_arg_ns = 0
    var loop_t0 = perf_counter_ns()
    for step in range(N_STEPS):
        var s0 = perf_counter_ns()
        var seq = _head_pre(xt, model, ds_cw, ds_nw, base_host, is_gen, ctx)
        ctx.synchronize()
        var s1 = perf_counter_ns()
        var hidden = nldiff_backbone_resident[SEQ](
            seq, store, cfg, qh[0], qh[1], kh[0], kh[1], ctx)
        _ = seq^
        ctx.synchronize()
        var s2 = perf_counter_ns()
        var logits = _head_post_logits(hidden, is_gen, us_cw, us_nw, pred_w, ctx)
        _ = hidden^
        ctx.synchronize()
        var s3 = perf_counter_ns()
        var x0 = nldiff_argmax_lastdim(logits, ctx)
        _ = logits^
        var s4 = perf_counter_ns()
        var nt = num_transfer[step]
        var sel = nldiff_stratified_select(order, committed, nt)
        for k in range(len(sel)):
            xt[sel[k]] = x0[sel[k]]
        committed += nt
        # skip step-0 (autotune) from the phase averages
        if step >= 1:
            a_pre_ns += Int(s1 - s0)
            a_bb_ns += Int(s2 - s1)
            a_post_ns += Int(s3 - s2)
            a_arg_ns += Int(s4 - s3)
        if step < 3 or step % 16 == 0 or step == N_STEPS - 1:
            print("  [AFTER] step", step, "committed", committed,
                  "(", Float64(s4 - s0) / 1.0e6, "ms )")
    var loop_t1 = perf_counter_ns()
    if committed != N_TOKENS:
        raise Error("committed != 4096 after loop")
    var total_s = Float64(loop_t1 - loop_t0) / 1.0e9
    var ad = Float64(N_STEPS - 1)
    print("\n[AFTER] avg per-step split (steps 1..63, ms):")
    print("   head_pre:                ", Float64(a_pre_ns) / ad / 1.0e6)
    print("   backbone (resident H2D+compute):", Float64(a_bb_ns) / ad / 1.0e6)
    print("   head_post:               ", Float64(a_post_ns) / ad / 1.0e6)
    print("   argmax/sampler:          ", Float64(a_arg_ns) / ad / 1.0e6)
    print("[AFTER] total loop:", total_s, "s (", total_s / Float64(N_STEPS), "s/step )")

    var before_step_s = Float64(b_step_ns) / bd / 1.0e9
    var after_step_s = total_s / Float64(N_STEPS)
    print("\n[SPEEDUP] before", before_step_s, "s/step  ->  after", after_step_s,
          "s/step  =", before_step_s / after_step_s, "x")

    # ═══════════════ BYTE-IDENTICAL gate vs committed mojo_tokens ═══════════════
    var reference = _round_ints(_load_f32(mojo_ref, "x0_img", ctx))
    var n_match = 0
    var n_diff = 0
    for i in range(N_TOKENS):
        if xt[i] == reference[i]:
            n_match += 1
        else:
            n_diff += 1
    print("\n[GATE] BYTE-IDENTICAL token check vs parity/mojo_tokens.safetensors:")
    print("   n_match", n_match, "/", N_TOKENS, "  n_diff", n_diff)
    if n_match == N_TOKENS:
        print("[GATE] PASS — 4096/4096 IDENTICAL. Optimization preserved numerics.")
    else:
        print("[GATE] FAIL — token grid CHANGED. The optimization altered the math.")
        raise Error("byte-identity gate failed: n_diff=" + String(n_diff))

    print("============= SPEED probe complete =============")
