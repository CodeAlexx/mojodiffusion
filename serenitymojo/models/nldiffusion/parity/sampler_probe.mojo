# models/nldiffusion/parity/sampler_probe.mojo — CHUNK E gate.
#
# Gates the DETERMINISTIC unmask-sampler ops against sampler_oracle.safetensors
# (F32 in-file) + the captured gen_predictor logits in oracle1.safetensors.
#
#   num_transfer          [64] : nldiff_num_transfer_tokens(4096,64,shift=5)
#                                 -> EXACT int equality (max abs int diff == 0).
#   sch_temps_linear      [64] : nldiff_temp_schedule_linear(64,0.01)
#                                 -> cos >= 0.9999 / tiny max_abs.
#   cfg_combined         [4,16]: nldiff_cfg_combine(cfg_cond,cfg_uncond,gs=5)
#                                 -> cos >= 0.99999 (near-exact elementwise).
#   gen_predictor_argmax [128] : nldiff_argmax_lastdim(oracle1 gen_predictor.out
#                                 [128,131072]) -> EXACT int equality.
#   stratified                 : nldiff_stratified_select over unmask_order[4096]
#                                 with the per-step num_transfer budget; the
#                                 concatenation of all selections must reproduce
#                                 unmask_order exactly (self-consistency).
#
# Run (JIT):
#   cd /home/alex/mojodiffusion && \
#     pixi run mojo run -I . serenitymojo/models/nldiffusion/parity/sampler_probe.mojo

from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.parity import ParityHarness
from serenitymojo.models.nldiffusion.sampler import (
    nldiff_num_transfer_tokens,
    nldiff_temp_schedule_linear,
    nldiff_cfg_combine,
    nldiff_argmax_lastdim,
    nldiff_stratified_select,
)

comptime PARITY_DIR = "/home/alex/mojodiffusion/serenitymojo/models/nldiffusion/parity"


def _load_f32(st: ShardedSafeTensors, name: String, ctx: DeviceContext) raises -> List[Float32]:
    var tv = st.tensor_view(name)
    return Tensor.from_view_as_f32(tv, ctx).to_host(ctx)


def _load_tensor_f32(st: ShardedSafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    var tv = st.tensor_view(name)
    return Tensor.from_view_as_f32(tv, ctx)


def _round_ints(xs: List[Float32]) -> List[Int]:
    var out = List[Int]()
    for i in range(len(xs)):
        var v = xs[i]
        if v >= 0.0:
            out.append(Int(v + 0.5))
        else:
            out.append(Int(v - 0.5))
    return out^


def main() raises:
    var ctx = DeviceContext()
    var oracle = ShardedSafeTensors.open(String(PARITY_DIR) + "/sampler_oracle.safetensors")

    print("==================== CHUNK E: deterministic sampler gate ====================")

    # ── 1. num_transfer (shift schedule) — EXACT int equality ────────────────
    var ntt_ref = _round_ints(_load_f32(oracle, "num_transfer", ctx))
    var ntt = nldiff_num_transfer_tokens(4096, 64, 5)
    var max_int_diff = 0
    var mine_sum = 0
    var ref_sum = 0
    for i in range(len(ntt_ref)):
        var d = ntt[i] - ntt_ref[i]
        if d < 0:
            d = -d
        if d > max_int_diff:
            max_int_diff = d
        mine_sum += ntt[i]
        ref_sum += ntt_ref[i]
    print("[E] num_transfer: len", len(ntt), "(ref", len(ntt_ref), ")",
          " sum(mine)=", mine_sum, " sum(ref)=", ref_sum,
          " max_abs_int_diff=", max_int_diff)
    print("    mine[:6] =", ntt[0], ntt[1], ntt[2], ntt[3], ntt[4], ntt[5])
    print("    ref [:6] =", ntt_ref[0], ntt_ref[1], ntt_ref[2], ntt_ref[3], ntt_ref[4], ntt_ref[5])
    if max_int_diff == 0 and mine_sum == 4096:
        print("[E] num_transfer GATE PASS (EXACT int equality, sum==4096)")
    else:
        print("[E] num_transfer GATE FAIL")

    # ── 2. sch_temps_linear — cos >= 0.9999 ──────────────────────────────────
    var st_ref = _load_f32(oracle, "sch_temps_linear", ctx)
    var st_mine = nldiff_temp_schedule_linear(64, 0.01)
    var harness = ParityHarness(0.9999)
    var st_res = harness.compare_host(st_mine, st_ref)
    print("[E] sch_temps_linear: cos =", st_res.cos, " max_abs =", st_res.max_abs)
    if st_res.cos >= 0.9999:
        print("[E] sch_temps_linear GATE PASS (cos >= 0.9999)")
    else:
        print("[E] sch_temps_linear GATE FAIL")

    # ── 3. cfg_combine — cos >= 0.99999 ──────────────────────────────────────
    var cond = _load_tensor_f32(oracle, "cfg_cond", ctx)
    var uncond = _load_tensor_f32(oracle, "cfg_uncond", ctx)
    var cfg_out = nldiff_cfg_combine(cond, uncond, 5.0, ctx)
    var cfg_host = cfg_out.to_host(ctx)
    var cfg_ref = _load_f32(oracle, "cfg_combined", ctx)
    var cfg_h = ParityHarness(0.99999)
    var cfg_res = cfg_h.compare_host(cfg_host, cfg_ref)
    print("[E] cfg_combined: cos =", cfg_res.cos, " max_abs =", cfg_res.max_abs)
    if cfg_res.cos >= 0.99999:
        print("[E] cfg_combined GATE PASS (cos >= 0.99999)")
    else:
        print("[E] cfg_combined GATE FAIL")
    _ = cfg_out^
    _ = cond^
    _ = uncond^

    # ── 4. gen_predictor_argmax — EXACT int equality over 131072 classes ─────
    var o1 = ShardedSafeTensors.open(String(PARITY_DIR) + "/oracle1.safetensors")
    var gp = _load_tensor_f32(o1, "gen_predictor.out", ctx)  # [128,131072] f32
    print("[E] gen_predictor.out shape =", gp.shape()[0], "x", gp.shape()[1])
    var am_mine = nldiff_argmax_lastdim(gp, ctx)  # [128]
    var am_ref = _round_ints(_load_f32(oracle, "gen_predictor_argmax", ctx))
    var am_mismatch = 0
    for i in range(len(am_ref)):
        if am_mine[i] != am_ref[i]:
            am_mismatch += 1
    print("[E] gen_predictor_argmax: rows =", len(am_mine),
          " mismatches =", am_mismatch)
    print("    mine[:6] =", am_mine[0], am_mine[1], am_mine[2], am_mine[3], am_mine[4], am_mine[5])
    print("    ref [:6] =", am_ref[0], am_ref[1], am_ref[2], am_ref[3], am_ref[4], am_ref[5])
    if am_mismatch == 0:
        print("[E] gen_predictor_argmax GATE PASS (EXACT int equality over 131072 classes)")
    else:
        print("[E] gen_predictor_argmax GATE FAIL")
    _ = gp^

    # ── 5. stratified select — self-consistency vs unmask_order ──────────────
    var order = _round_ints(_load_f32(oracle, "unmask_order", ctx))  # [4096]
    # Walk the per-step budget; each step selects order[start:start+num_transfer]
    # with start = tokens committed so far. Concatenation must == order exactly.
    var recon = List[Int]()
    var committed = 0
    for step in range(len(ntt)):
        var sel = nldiff_stratified_select(order, committed, ntt[step])
        for k in range(len(sel)):
            recon.append(sel[k])
        committed += ntt[step]
    var strat_mismatch = 0
    if len(recon) != len(order):
        strat_mismatch = -1
    else:
        for i in range(len(order)):
            if recon[i] != order[i]:
                strat_mismatch += 1
    print("[E] stratified: committed =", committed, " recon_len =", len(recon),
          " order_len =", len(order), " mismatches =", strat_mismatch)
    if strat_mismatch == 0 and committed == 4096:
        print("[E] stratified GATE PASS (per-step slices reconstruct unmask_order exactly)")
    else:
        print("[E] stratified GATE FAIL")

    print("==================== CHUNK E gate complete ====================")
