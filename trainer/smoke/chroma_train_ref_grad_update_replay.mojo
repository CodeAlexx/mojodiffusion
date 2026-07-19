# Replay the real Serenity Chroma train-step dump through Mojo
# forward -> loss -> LoRA backward -> AdamW, then compare the resulting adapter
# delta against Serenity's dumped adapter_after.*.
#
# Mirrors smoke/klein_train_ref_grad_update_replay.mojo, adapted to Chroma's
# single-step dump (chroma_train_ref_step000_adapters.safetensors). Klein used a
# two-step /tmp dump; the Chroma dump is ONE optimizer step with:
#     adapter_before.*  (LoRA-up == 0, LoRA-down == Serenity random init)
#     adapter_after.*   (post step-0 LoRA tensors)
# Because LoRA-up (B) starts at 0, at step 0:  grad_A == 0 (so A only decays by
# weight_decay) and grad_B == scale * dY @ (A@x)^T uses Serenity's adapter_before
# A. So we LOAD adapter_before into the Mojo carrier (exactly Klein's contract of
# initialising from the dumped before-state) and the post-step adapter IS the
# update we reproduce.
#
# Target inventory (verified from the 304-adapter dump): the "attn,ff.net"
# layer_filter set — double img {to_q,to_k,to_v,to_out.0,ff.net.0.proj,ff.net.2},
# double txt {add_q_proj,add_k_proj,add_v_proj,to_add_out} (NO ff_context MLP),
# single {to_q,to_k,to_v} (NO proj_mlp / proj_out). Untrained carrier slots are
# zeroed (A=0,B=0) so they contribute nothing to forward/backward/clip.
#
# Recipe (parity/chroma_train_ref_meta.json): lr=3e-4, AdamW(0.9,0.999,1e-8),
# weight_decay=0.01 (confirmed: down-delta/|A| == lr*wd), clip_grad_norm=1.0,
# rank=16, alpha=1.0 (scale=1/16). B=2 samples (timesteps 0.907 / 0.709); loss is
# the MSE mean over both unpacked samples (= the loss the loss-replay gate checks,
# 0.29571867), so d_loss = (2/N_total)(pred - target) in packed space.
#
# The MAIN LOOP owns the final parity bar; this smoke prints grad/update stats
# (Klein-style) for re-verification. Storage note: Mojo LoRA tensors are BF16, so
# the reproduced adapter_after carries ~bf16 rounding vs Serenity's F32 dump.

from std.collections import List, Optional
from std.gpu.host import DeviceContext
from std.math import sqrt
from std.time import perf_counter_ns

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.tensor import Tensor

from serenitymojo.models.dit.flux1_dit import build_flux1_rope_tables
from serenitymojo.models.dit.chroma_dit import ChromaDitCache
from serenitymojo.offload.plan import build_chroma1_hd_block_plan, OffloadConfig
from serenitymojo.offload.turbo_planned_loader import TurboPlannedLoader
from serenitymojo.models.flux.flux_stack_lora import (
    FluxLoraSet, FluxLoraGradSet, build_flux_lora_set, total_adapters,
    flux_lora_adamw_step,
)
from serenitymojo.training.train_step import LoraAdapter

from serenity_trainer.model.chroma.weights import load_chroma_stack_base
from serenity_trainer.model.chroma.chroma_stack_lora import (
    chroma_stack_lora_forward_offload, chroma_stack_lora_backward_offload,
)


# ── arch ─────────────────────────────────────────────────────────────────────
comptime H = 24
comptime Dh = 128
comptime D = H * Dh            # 3072
comptime FMLP = 12288
comptime IN_CH = 64
comptime TXT_CH = 4096
comptime OUT_CH = 64
comptime NUM_DOUBLE = 19
comptime NUM_SINGLE = 38
comptime MOD_INDEX = 3 * NUM_SINGLE + 2 * 6 * NUM_DOUBLE + 2   # 344
comptime EPS = Float32(1e-06)

comptime HT = 32
comptime WT = 32
comptime N_IMG = HT * WT       # 1024
comptime N_TXT = 224
comptime S = N_TXT + N_IMG     # 1248
comptime BATCH = 2

comptime LAT_C = 16
comptime LAT_H = 64
comptime LAT_W = 64
comptime PATCH = 2

# ── recipe ───────────────────────────────────────────────────────────────────
comptime RANK = 16
comptime ALPHA = Float32(1.0)      # dump lora_alpha=1.0 -> scale 1/16
comptime LR = Float32(3.0e-4)
comptime BETA1 = Float32(0.9)
comptime BETA2 = Float32(0.999)
comptime ADAM_EPS = Float32(1.0e-8)
comptime WEIGHT_DECAY = Float32(0.01)
comptime CLIP_GRAD_NORM = Float32(1.0)

comptime DBL_SLOTS = 12             # 2 streams * 6 slots
comptime SGL_SLOTS_FULL = 5

comptime PARITY = "trainer/parity/chroma_train_ref_step000.safetensors"
comptime ADAPTERS = "trainer/parity/chroma_train_ref_step000_adapters.safetensors"
comptime CKPT = "/home/alex/.serenity/models/checkpoints/chroma1_hd_bf16.safetensors"


def _sec(ns0: UInt, ns1: UInt) -> Float64:
    return Float64(ns1 - ns0) / Float64(1000000000.0)


def _zeros(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for _ in range(n):
        out.append(Float32(0.0))
    return out^


def _dump_f32(st: ShardedSafeTensors, key: String, ctx: DeviceContext) raises -> List[Float32]:
    var t = Tensor.from_view(st.tensor_view(key), ctx)
    if t.dtype() == STDtype.BF16:
        var bf = t.to_host_bf16(ctx)
        var out = List[Float32]()
        for i in range(len(bf)):
            out.append(bf[i].cast[DType.float32]())
        return out^
    return t.to_host(ctx)


def _slice_sample(flat: List[Float32], b: Int, per: Int) -> List[Float32]:
    var out = List[Float32]()
    var base = b * per
    for i in range(per):
        out.append(flat[base + i])
    return out^


# pack_latents: [16,64,64] flat -> [N_IMG,64] channel-major patchify (train_chroma_real).
def _pack_latents(lat: List[Float32]) -> List[Float32]:
    var out = List[Float32]()
    for ih in range(HT):
        for iw in range(WT):
            for c in range(LAT_C):
                for ph in range(PATCH):
                    for pw in range(PATCH):
                        var hh = ih * PATCH + ph
                        var ww = iw * PATCH + pw
                        var idx = c * LAT_H * LAT_W + hh * LAT_W + ww
                        out.append(lat[idx])
    return out^


def _pooled_modulation(approx: ChromaDitCache, t_model: Float32, ctx: DeviceContext) raises -> List[Float32]:
    var approx_in = approx._approximator_input(t_model, ctx)
    var pooled_t = approx.approximator_forward(approx_in, ctx)
    var bf = pooled_t.to_host_bf16(ctx)
    var out = List[Float32]()
    for i in range(len(bf)):
        out.append(bf[i].cast[DType.float32]())
    return out^


# ── trained-slot inventory -> (carrier flat index, dump diffusers module name) ──
def _dbl_base(bi: Int) -> Int:
    return bi * DBL_SLOTS


def _sgl_base(bi: Int) -> Int:
    return NUM_DOUBLE * DBL_SLOTS + bi * SGL_SLOTS_FULL


# Returns parallel lists: carrier flat index, dump key suffix (block-qualified).
def _trained_targets() -> Tuple[List[Int], List[String]]:
    var idxs = List[Int]()
    var keys = List[String]()
    for bi in range(NUM_DOUBLE):
        var base = _dbl_base(bi)
        var bp = String("transformer_blocks.") + String(bi) + String(".")
        # img stream slots 0..5
        idxs.append(base + 0); keys.append(bp + String("attn.to_q"))
        idxs.append(base + 1); keys.append(bp + String("attn.to_k"))
        idxs.append(base + 2); keys.append(bp + String("attn.to_v"))
        idxs.append(base + 3); keys.append(bp + String("attn.to_out.0"))
        idxs.append(base + 4); keys.append(bp + String("ff.net.0.proj"))
        idxs.append(base + 5); keys.append(bp + String("ff.net.2"))
        # txt stream slots 6..9 (NO ff_context MLP)
        idxs.append(base + 6); keys.append(bp + String("attn.add_q_proj"))
        idxs.append(base + 7); keys.append(bp + String("attn.add_k_proj"))
        idxs.append(base + 8); keys.append(bp + String("attn.add_v_proj"))
        idxs.append(base + 9); keys.append(bp + String("attn.to_add_out"))
    for bi in range(NUM_SINGLE):
        var sbase = _sgl_base(bi)
        var sp = String("single_transformer_blocks.") + String(bi) + String(".")
        idxs.append(sbase + 0); keys.append(sp + String("attn.to_q"))
        idxs.append(sbase + 1); keys.append(sp + String("attn.to_k"))
        idxs.append(sbase + 2); keys.append(sp + String("attn.to_v"))
    return (idxs^, keys^)


def _adapter_key(phase: String, module: String, kind: String) -> String:
    return phase + String(".lora_transformer.") + module + String(".") + kind + String(".weight")


def _read_adapter(ad: ShardedSafeTensors, phase: String, module: String, kind: String, ctx: DeviceContext) raises -> List[Float32]:
    return Tensor.from_view(ad.tensor_view(_adapter_key(phase, module, kind)), ctx).to_host(ctx)


# ── zero one carrier adapter (A=0,B=0, optimizer state 0) preserving shape/scale ─
def _zero_adapter(set: FluxLoraSet, idx: Int) -> LoraAdapter:
    ref a = set.ad[idx]
    var inf = a.in_f
    var outf = a.out_f
    var sc = a.scale
    return LoraAdapter(
        _zeros(RANK * inf), _zeros(outf * RANK), RANK, inf, outf, sc,
        _zeros(RANK * inf), _zeros(RANK * inf), _zeros(outf * RANK), _zeros(outf * RANK),
    )


def _set_adapter(set: FluxLoraSet, idx: Int, var a_vals: List[Float32], var b_vals: List[Float32]) -> LoraAdapter:
    ref a = set.ad[idx]
    var inf = a.in_f
    var outf = a.out_f
    var sc = a.scale
    return LoraAdapter(
        a_vals^, b_vals^, RANK, inf, outf, sc,
        _zeros(RANK * inf), _zeros(RANK * inf), _zeros(outf * RANK), _zeros(outf * RANK),
    )


@fieldwise_init
struct Stats(Copyable, Movable, ImplicitlyCopyable):
    var elems: Int
    var nonzero: Int
    var nonfinite: Int
    var sumsq: Float64
    var max_abs: Float32


def _empty_stats() -> Stats:
    return Stats(0, 0, 0, Float64(0.0), Float32(0.0))


def _is_nonfinite(x: Float32) -> Bool:
    if x != x:
        return True
    return (x - x) != Float32(0.0)


def _scan(var s: Stats, x: Float32) -> Stats:
    s.elems += 1
    if x != Float32(0.0):
        s.nonzero += 1
    if _is_nonfinite(x):
        s.nonfinite += 1
        return s^
    var ax = x if x >= Float32(0.0) else -x
    s.sumsq += Float64(x) * Float64(x)
    if ax > s.max_abs:
        s.max_abs = ax
    return s^


def _l2(s: Stats) -> Float64:
    return sqrt(s.sumsq)


def _b_host(set: FluxLoraSet, idx: Int) -> List[Float32]:
    ref a = set.ad[idx]
    var out = List[Float32]()
    for i in range(len(a.b)):
        out.append(a.b[i].cast[DType.float32]())
    return out^


def _a_host(set: FluxLoraSet, idx: Int) -> List[Float32]:
    ref a = set.ad[idx]
    var out = List[Float32]()
    for i in range(len(a.a)):
        out.append(a.a[i].cast[DType.float32]())
    return out^


# ── accumulate two grad sets (element-wise) ──────────────────────────────────
def _accumulate(mut acc: FluxLoraGradSet, add: FluxLoraGradSet) raises:
    for i in range(len(acc.d_a)):
        for j in range(len(add.d_a[i])):
            if j < len(acc.d_a[i]):
                acc.d_a[i][j] = acc.d_a[i][j] + add.d_a[i][j]
        for j in range(len(add.d_b[i])):
            if j < len(acc.d_b[i]):
                acc.d_b[i][j] = acc.d_b[i][j] + add.d_b[i][j]


def _global_norm(grads: FluxLoraGradSet) -> Float64:
    var ss = 0.0
    for i in range(len(grads.d_a)):
        for j in range(len(grads.d_a[i])):
            ss += Float64(grads.d_a[i][j]) * Float64(grads.d_a[i][j])
        for j in range(len(grads.d_b[i])):
            ss += Float64(grads.d_b[i][j]) * Float64(grads.d_b[i][j])
    return sqrt(ss)


def _clip(mut grads: FluxLoraGradSet, max_norm: Float32) -> Float64:
    var gn = _global_norm(grads)
    if gn <= Float64(max_norm) or gn == 0.0:
        return gn
    var s = Float32(Float64(max_norm) / gn)
    for i in range(len(grads.d_a)):
        for j in range(len(grads.d_a[i])):
            grads.d_a[i][j] = grads.d_a[i][j] * s
        for j in range(len(grads.d_b[i])):
            grads.d_b[i][j] = grads.d_b[i][j] * s
    return gn


def main() raises:
    var ctx = DeviceContext()
    var all0 = perf_counter_ns()

    print("=== Chroma train-ref grad/update replay ===")
    print("[parity]  ", PARITY)
    print("[adapters]", ADAPTERS)
    print("[ckpt]    ", CKPT)
    print("[shape] N_IMG =", N_IMG, "N_TXT =", N_TXT, "S =", S, "BATCH =", BATCH)

    var st = ShardedSafeTensors.open(String(PARITY))
    var img_all = _dump_f32(st, String("trace.packed_latent_input"), ctx)
    var txt_all = _dump_f32(st, String("trace.encoder_hidden_states"), ctx)
    var tgt_all = _dump_f32(st, String("output.target"), ctx)            # [B,16,64,64]
    var ts_all = _dump_f32(st, String("trace.transformer_timestep"), ctx)
    print("[dump] timesteps =", ts_all[0], ts_all[1])

    var adp = ShardedSafeTensors.open(String(ADAPTERS))

    var load0 = perf_counter_ns()
    var base_st = SafeTensors.open(String(CKPT))
    var base = load_chroma_stack_base(base_st, NUM_DOUBLE, NUM_SINGLE, ctx)
    var approx = ChromaDitCache.load(String(CKPT), ctx)
    var plan = build_chroma1_hd_block_plan()
    var cfg = OffloadConfig.synchronous_single()
    var loader = TurboPlannedLoader.open(String(CKPT), plan^, cfg, ctx)
    var load1 = perf_counter_ns()
    print("[load] base + approximator + offload loader (", loader.block_count(), "blocks)")

    var rope = build_flux1_rope_tables[N_IMG, N_TXT, H, Dh](HT, WT, ctx, STDtype.BF16)
    var cos = rope[0].to_host(ctx)
    var sin = rope[1].to_host(ctx)

    # ── carrier: zero every slot, then load adapter_before into trained slots ──
    var lora = build_flux_lora_set(NUM_DOUBLE, NUM_SINGLE, D, FMLP, RANK, ALPHA)
    var n_adapters = total_adapters(lora)
    for idx in range(n_adapters):
        lora.ad[idx] = _zero_adapter(lora, idx)
    var tt = _trained_targets()
    var t_idx = tt[0].copy()
    var t_key = tt[1].copy()
    var n_trained = len(t_idx)
    print("[lora] adapters:", n_adapters, " trained slots:", n_trained)
    for j in range(n_trained):
        var a_vals = _read_adapter(adp, String("adapter_before"), t_key[j], String("lora_down"), ctx)
        var b_vals = _read_adapter(adp, String("adapter_before"), t_key[j], String("lora_up"), ctx)
        lora.ad[t_idx[j]] = _set_adapter(lora, t_idx[j], a_vals^, b_vals^)

    var img_per = N_IMG * IN_CH
    var txt_per = N_TXT * TXT_CH
    var tgt_per = LAT_C * LAT_H * LAT_W   # 65536

    var n_total = BATCH * N_IMG * OUT_CH
    var inv_n = Float32(2.0) / Float32(n_total)

    # ── per-sample forward -> d_loss -> backward, accumulate grads ──
    var grads_acc = Optional[FluxLoraGradSet](None)
    var loss_sum = Float64(0.0)
    var fwd_total = Float64(0.0)
    var bwd_total = Float64(0.0)
    for b in range(BATCH):
        var img_tokens = _slice_sample(img_all, b, img_per)
        var txt_tokens = _slice_sample(txt_all, b, txt_per)
        var tgt_unpacked = _slice_sample(tgt_all, b, tgt_per)
        var target_packed = _pack_latents(tgt_unpacked)
        var pooled = _pooled_modulation(approx, ts_all[b], ctx)

        var f0 = perf_counter_ns()
        var fwd = chroma_stack_lora_forward_offload[H, Dh, N_IMG, N_TXT, S](
            img_tokens.copy(), txt_tokens.copy(), pooled^, MOD_INDEX,
            base, loader, lora, cos.copy(), sin.copy(),
            D, FMLP, IN_CH, TXT_CH, OUT_CH, EPS, ctx,
        )
        var f1 = perf_counter_ns()
        fwd_total += _sec(f0, f1)

        var d_loss = List[Float32]()
        var nout = len(fwd.out)
        for i in range(nout):
            var diff = fwd.out[i] - target_packed[i]
            loss_sum += Float64(diff) * Float64(diff)
            d_loss.append(inv_n * diff)

        var w0 = perf_counter_ns()
        var grads_b = chroma_stack_lora_backward_offload[H, Dh, N_IMG, N_TXT, S](
            d_loss, img_tokens.copy(), txt_tokens.copy(), base, loader, lora,
            cos.copy(), sin.copy(), fwd,
            D, FMLP, IN_CH, TXT_CH, OUT_CH, EPS, ctx,
        )
        var w1 = perf_counter_ns()
        bwd_total += _sec(w0, w1)
        print("[sample", b, "] forward+backward done; nonfinite_lora_grads =",
              grads_b.nonfinite_lora_grads)

        if grads_acc:
            var acc = grads_acc.take()
            _accumulate(acc, grads_b)
            grads_acc = Optional[FluxLoraGradSet](acc^)
        else:
            grads_acc = Optional[FluxLoraGradSet](grads_b^)

    var loss = Float32(loss_sum / Float64(n_total))
    print("[loss] mojo =", loss, " (dump output.loss_for_backward = 0.29571867)")

    var grads = grads_acc.take()
    # ── B=0 invariant gate (noise-immune, structural): B is zero-init, so
    # d_A = (alpha/r) B^T d_y x^T == 0 EXACTLY. Any material d_A is a real bug
    # in the A-gradient arm. d_B carries all the real backward signal here.
    var da_max = Float64(0.0)
    var da_l2 = Float64(0.0)
    var db_max = Float64(0.0)
    var db_l2 = Float64(0.0)
    for i in range(len(grads.d_a)):
        for j in range(len(grads.d_a[i])):
            var v = Float64(grads.d_a[i][j])
            var av = v if v >= 0.0 else -v
            if av > da_max: da_max = av
            da_l2 += v * v
        for j in range(len(grads.d_b[i])):
            var v = Float64(grads.d_b[i][j])
            var av = v if v >= 0.0 else -v
            if av > db_max: db_max = av
            db_l2 += v * v
    print("[raw grad] d_A: max =", da_max, " l2 =", da_l2 ** 0.5,
          "  d_B: max =", db_max, " l2 =", db_l2 ** 0.5,
          "  (B=0 invariant ⇒ d_A must be ~0)")

    var gn_before = _clip(grads, CLIP_GRAD_NORM)
    print("[clip] global_grad_norm before clip =", gn_before, " (max_norm =", CLIP_GRAD_NORM, ")")

    flux_lora_adamw_step(lora, grads, 1, LR, ctx, BETA1, BETA2, ADAM_EPS, WEIGHT_DECAY)

    # ── compare trained-slot adapters vs adapter_after.* ──
    var b_ref = _empty_stats()
    var b_mojo = _empty_stats()
    var b_err = _empty_stats()
    var a_ref = _empty_stats()
    var a_err = _empty_stats()
    for j in range(n_trained):
        var idx = t_idx[j]
        var b_after_ref = _read_adapter(adp, String("adapter_after"), t_key[j], String("lora_up"), ctx)
        var a_after_ref = _read_adapter(adp, String("adapter_after"), t_key[j], String("lora_down"), ctx)
        var b_after_mojo = _b_host(lora, idx)
        var a_after_mojo = _a_host(lora, idx)
        for i in range(len(b_after_ref)):
            b_ref = _scan(b_ref, b_after_ref[i])
            b_mojo = _scan(b_mojo, b_after_mojo[i])
            b_err = _scan(b_err, b_after_mojo[i] - b_after_ref[i])
        for i in range(len(a_after_ref)):
            a_ref = _scan(a_ref, a_after_ref[i])
            a_err = _scan(a_err, a_after_mojo[i] - a_after_ref[i])
        if j == 0 or j == NUM_DOUBLE * 10:  # first double + first single probe
            var bl = _empty_stats()
            var bm = _empty_stats()
            var be = _empty_stats()
            for i in range(len(b_after_ref)):
                bl = _scan(bl, b_after_ref[i])
                bm = _scan(bm, b_after_mojo[i])
                be = _scan(be, b_after_mojo[i] - b_after_ref[i])
            print("  probe", t_key[j], "B: ref_l2 =", _l2(bl), " mojo_l2 =", _l2(bm),
                  " err_l2 =", _l2(be), " ref_max =", bl.max_abs, " err_max =", be.max_abs)

    var b_ref_l2 = _l2(b_ref)
    var b_err_l2 = _l2(b_err)
    var a_ref_l2 = _l2(a_ref)
    var a_err_l2 = _l2(a_err)
    print("[B(lora_up) update]  ref_l2 =", b_ref_l2, " mojo_l2 =", _l2(b_mojo),
          " err_l2 =", b_err_l2,
          " rel =", (b_err_l2 / b_ref_l2 if b_ref_l2 > 0.0 else 0.0),
          " ref_max =", b_ref.max_abs, " err_max =", b_err.max_abs,
          " nonfinite =", b_mojo.nonfinite)
    print("[A(lora_down) decay] ref_l2 =", a_ref_l2, " err_l2 =", a_err_l2,
          " rel =", (a_err_l2 / a_ref_l2 if a_ref_l2 > 0.0 else 0.0),
          " err_max =", a_err.max_abs)

    # ── masked directional gate: cos + sign-agreement on signal-bearing B ──
    # 75% of ref B elements are at the bf16-zero floor (tiny grads round to 0);
    # only |ref| >> floor carries sign-robust signal. Gate on that subset.
    var bmax = b_ref.max_abs
    for thr_frac in [Float32(0.5), Float32(0.1)]:
        var thr = thr_frac * bmax
        var dot = Float64(0.0)
        var nm = Float64(0.0)
        var nr = Float64(0.0)
        var nsel = 0
        var sign_ok = 0
        for j in range(n_trained):
            var idx = t_idx[j]
            var br = _read_adapter(adp, String("adapter_after"), t_key[j], String("lora_up"), ctx)
            var bm = _b_host(lora, idx)
            for i in range(len(br)):
                var r = br[i]
                if (r if r >= 0.0 else -r) > thr:
                    var m = bm[i]
                    dot += Float64(m) * Float64(r)
                    nm += Float64(m) * Float64(m)
                    nr += Float64(r) * Float64(r)
                    nsel += 1
                    if (m > 0.0) == (r > 0.0):
                        sign_ok += 1
        var denom = (nm * nr) ** 0.5
        var cosv = (dot / denom) if denom > 0.0 else 0.0
        var signfrac = (Float64(sign_ok) / Float64(nsel)) if nsel > 0 else 0.0
        print("[B masked |ref|>", thr_frac, "*max] n =", nsel,
              " cos =", cosv, " sign_agree =", signfrac)

    var all1 = perf_counter_ns()
    print("time_s: load =", _sec(load0, load1), " forward =", fwd_total,
          " backward =", bwd_total, " total =", _sec(all0, all1))
    if b_mojo.nonfinite != 0:
        raise Error("Chroma grad/update replay produced nonfinite adapter values")
    print("CHROMA TRAIN REF GRAD/UPDATE REPLAY DONE (main loop owns the bar)")
