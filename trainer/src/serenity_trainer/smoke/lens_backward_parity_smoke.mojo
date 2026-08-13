# lens_backward_parity_smoke.mojo — the load-bearing Lens LoRA BACKWARD parity gate.
#
# Proves the hand-chained Mojo LoRA backward (lens_backward_full_lora) matches a
# torch-autograd oracle per adapter. The oracle (parity/lens/lens_backward_oracle.py,
# Serenity-only: lens/transformer.py) runs the REAL-weights base transformer on
# the SAME fixed forward inputs as the forward-parity smoke, captures each wrapped
# Linear's input x_l and output-grad g_l via hooks, and dumps the analytic
#     d_B_l = scale * g_l^T @ (x_l @ A_l^T)        (B=0 → only d_B is nonzero)
# plus the exact per-adapter A_l and the fixed MSE target target_v.
#
# This smoke loads those SAME A_l into the 480-adapter LensLoraSet (B=0), runs the
# LoRA-overlaid TRAINING forward (lens_forward_full_lora; B=0 ⇒ velocity == base),
# computes the loss DIRECTLY on the packed velocity (NOT the predict/scale/noise
# path), takes d_velocity = 2(v - target)/numel, runs the hand-chained backward,
# and compares each adapter's d_B to the oracle d_B — except the 4 architecturally-
# zero adapters (last block txt-post; the img-only head reads no txt grad) which
# must have |d_B| absmax < 1e-4.
#
# DTYPE-AWARE GATE (corrects the dtype bad-reference trap):
#   The Mojo backward runs BF16; the oracle dB_{idx} is F32 truth. A flat
#   cos>=0.999-vs-F32 bar is UNACHIEVABLE for the low-magnitude txt gradient even
#   by torch itself — torch's own BF16 d_B vs its F32 d_B drops txt_w1 to mean
#   0.957 / min 0.897 (parity/lens/lens_backward_bf16_ceiling.py). So the bar is
#   not "match F32"; it is "be AT LEAST AS ACCURATE as torch's own BF16".
#   For each nonzero adapter idx, with ceiling = cos(torch_bf16_dB, torch_f32_dB)
#   from backward_grad_bf16_meta.json["ceiling_cos"][idx]:
#       PASS iff mojo_cos >= min(0.999, ceiling - 0.01)
#   i.e. Mojo must be within 0.01 of torch's BF16 ceiling, OR clear 0.999 outright.
#
# Reference policy: Serenity ONLY (lens/transformer.py via the oracle). No Rust.
#
# Oracle fixtures (parity/lens/):
#   backward_grad_ref.safetensors  keys: target_v [1,64,128]; per idx 0..479:
#       A_{idx}  [rank=16, in_f]    (the A loaded here — byte-identical to oracle)
#       dB_{idx} [out_f, 16]        (the autograd-derived d_B reference)
#   dit_fwd_in_hidden.safetensors  [1,64,128]    packed image latent
#   dit_fwd_in_txt_{0..3}.safetensors [1,16,2880] per-layer GPT-OSS feats (raw)
#   dit_fwd_in_timestep.safetensors [1]           timestep (0.5; → t_model 0.5)
#
# DTYPE: BF16 storage in/out (the trained boundary); the oracle is f32. The mojo
# adapter A is BF16-rounded on load (storage policy) while the oracle keeps f32 A,
# so a small BF16↔f32 gap is expected — the 0.999 cosine bar accounts for it.

from std.collections import List, Optional
from max.gpu.host import DeviceContext
from std.math import sqrt, isfinite

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.tensor_algebra import concat

from serenity_trainer.modelLoader.LensModelLoader import LensWeights, LENS_TRANSFORMER_DIR
from serenity_trainer.module.LensLoRAModule import LoraAdapter
from serenity_trainer.model.lens.lens_backward import (
    LensLoraSet, lens_backward_full_lora, LensStackLoraGrads,
)
from serenity_trainer.model.lens.lens_stack_lora import lens_forward_full_lora
from serenity_trainer.modelSetup.lensLoraTargets import (
    lora_module_prefix, LORA_SLOTS_PER_BLOCK, LENS_N_BLOCKS,
)
from serenitymojo.io.json_header import _Cursor, _parse_string, _skip_value
from serenity_trainer.util.config.TrainConfigReader import (
    _read_file_bytes, _parse_number,
)


comptime PARITY_DIR = "trainer/parity/lens"
comptime REF_PATH   = "trainer/parity/lens/backward_grad_ref.safetensors"
comptime HLp        = 8       # patchified img height (meta img_shapes (1,8,8))
comptime WLp        = 8
comptime N_IMG      = 64      # HLp*WLp
comptime CAPLEN     = 16      # s_txt
comptime IN_CH      = 128     # packed channels
comptime ENC        = 2880    # per-layer GPT-OSS feature dim
comptime RANK       = 16
comptime ALPHA      = Float32(16.0)
comptime N_ADAPTERS = LENS_N_BLOCKS * LORA_SLOTS_PER_BLOCK   # 480
comptime COS_BAR    = Float32(0.999)    # hard cap: clearing this always passes
comptime ZERO_BAR   = Float32(1.0e-4)
comptime CEIL_MARGIN = Float32(0.01)    # allowed slack below torch's BF16 ceiling
comptime BF16_META_PATH = "trainer/parity/lens/backward_grad_bf16_meta.json"


def _load_x(name: String, ctx: DeviceContext) raises -> Tensor:
    var st = ShardedSafeTensors.open(String(PARITY_DIR) + String("/") + name)
    return Tensor.from_view(st.tensor_view(String("x")), ctx)


def _zeros(n: Int) -> List[Float32]:
    var v = List[Float32]()
    for _ in range(n):
        v.append(Float32(0.0))
    return v^


def _cosine(a: List[Float32], b: List[Float32]) raises -> Float32:
    if len(a) != len(b):
        raise Error(String("cosine: length mismatch ") + String(len(a))
                    + String(" vs ") + String(len(b)))
    var dot = Float64(0.0)
    var na = Float64(0.0)
    var nb = Float64(0.0)
    for i in range(len(a)):
        dot += Float64(a[i]) * Float64(b[i])
        na += Float64(a[i]) * Float64(a[i])
        nb += Float64(b[i]) * Float64(b[i])
    if na <= 0.0 or nb <= 0.0:
        raise Error("cosine: zero-norm vector")
    return Float32(dot / (na ** 0.5 * nb ** 0.5))


def _absmax(a: List[Float32]) -> Float32:
    var m = Float32(0.0)
    for i in range(len(a)):
        var v = a[i]
        if v < 0.0:
            v = -v
        if v > m:
            m = v
    return m


# Read the per-adapter BF16 accuracy ceiling — cos(torch_bf16_dB, torch_f32_dB) —
# from backward_grad_bf16_meta.json["ceiling_cos"] (480 floats, same (block,slot)
# order as the oracle dB_{idx}). Uses the same pure-Mojo JSON machinery the port's
# config reader uses (_Cursor / _parse_string / _parse_number / _skip_value), no
# Python. Top-level object: skip every key except "ceiling_cos", whose value is a
# flat JSON array of numbers.
def _load_ceiling(path: String) raises -> List[Float32]:
    var bytes = _read_file_bytes(path)
    var cur = _Cursor(bytes^)
    var out = List[Float32]()
    cur.expect(0x7B)  # top-level '{'
    cur.skip_ws()
    if cur.peek() == 0x7D:  # empty object
        cur.advance()
        return out^
    while True:
        var key = _parse_string(cur)
        cur.expect(0x3A)  # ':'
        if key == "ceiling_cos":
            cur.expect(0x5B)  # '['
            cur.skip_ws()
            if cur.peek() == 0x5D:  # empty array
                cur.advance()
            else:
                while True:
                    var v = _parse_number(cur)
                    out.append(Float32(v))
                    cur.skip_ws()
                    var ch = cur.peek()
                    if ch == 0x2C:  # ','
                        cur.advance()
                        continue
                    if ch == 0x5D:  # ']'
                        cur.advance()
                        break
                    raise Error(String("ceiling_cos: expected ',' or ']' at byte ")
                                + String(cur.pos))
        else:
            _skip_value(cur)
        cur.skip_ws()
        var c = cur.peek()
        if c == 0x2C:  # ','
            cur.advance()
            continue
        if c == 0x7D:  # '}'
            cur.advance()
            break
        raise Error(String("bf16 meta: expected ',' or '}' at byte ") + String(cur.pos))
    return out^


# Build the 480-adapter LensLoraSet with A LOADED from the oracle ref (B=0). A is
# stored BF16 (LoraAdapter storage policy); in_f/out_f are read from the A/dB
# tensor shapes so the (block,slot) dims come straight from the oracle's Linears.
def build_loaded_lora_set(st: ShardedSafeTensors, ctx: DeviceContext) raises -> LensLoraSet:
    var block = List[LoraAdapter]()
    for b in range(LENS_N_BLOCKS):
        for slot in range(LORA_SLOTS_PER_BLOCK):
            var idx = b * LORA_SLOTS_PER_BLOCK + slot
            var a_t = Tensor.from_view(
                st.tensor_view(String("A_") + String(idx)), ctx)   # [rank, in_f] f32
            var a_sh = a_t.shape()
            var in_f = a_sh[1]
            var out_f = st.tensor_info(String("dB_") + String(idx)).shape[0]  # [out_f, rank]
            var a_host = a_t.to_host(ctx)                          # rank*in_f F32
            var b_list = _zeros(out_f * RANK)                      # B = 0
            block.append(
                LoraAdapter(
                    a_host^, b_list^, RANK, in_f, out_f, ALPHA / Float32(RANK),
                    _zeros(RANK * in_f), _zeros(RANK * in_f),
                    _zeros(out_f * RANK), _zeros(out_f * RANK),
                )
            )
    return LensLoraSet(block^, RANK)


def main() raises:
    var ctx = DeviceContext()
    print("=== Lens LoRA backward parity smoke (dtype-aware BF16-ceiling gate) ===")
    print("  PASS rule (nonzero adapter): mojo_cos >= min(", COS_BAR,
          ", ceiling - ", CEIL_MARGIN, ")  [ceiling = cos(torch_bf16_dB, torch_f32_dB)]")

    # ── per-adapter BF16 accuracy ceiling (cos torch_bf16 vs torch_f32) ────────
    print("[ceiling] reading", String(BF16_META_PATH))
    var ceiling = _load_ceiling(String(BF16_META_PATH))
    if len(ceiling) != N_ADAPTERS:
        raise Error(String("ceiling_cos has ") + String(len(ceiling))
                    + String(" entries, expected ") + String(N_ADAPTERS))
    print("  loaded", len(ceiling), "ceiling cos values")

    # ── frozen transformer weights (real checkpoint) ──────────────────────────
    print("[weights] loading Lens transformer:", String(LENS_TRANSFORMER_DIR))
    var weights = LensWeights.load(String(LENS_TRANSFORMER_DIR), ctx)
    print("  loaded", weights.count(), "tensors")

    # ── oracle ref (A_{idx}, dB_{idx}, target_v) ──────────────────────────────
    var st = ShardedSafeTensors.open(String(REF_PATH))

    # ── 480 adapters with A from the oracle, B=0 ──────────────────────────────
    print("[lora] building", N_ADAPTERS, "adapters with oracle A (B=0)")
    var loras = build_loaded_lora_set(st, ctx)
    print("  built", len(loras.block), "adapters, rank", loras.rank)

    # ── fixed forward inputs (the SAME the oracle used), cast to BF16 ──────────
    var packed = cast_tensor(_load_x(String("dit_fwd_in_hidden.safetensors"), ctx), STDtype.BF16, ctx)
    var t0 = _load_x(String("dit_fwd_in_txt_0.safetensors"), ctx)   # [1,16,2880] f32
    var t1 = _load_x(String("dit_fwd_in_txt_1.safetensors"), ctx)
    var t2 = _load_x(String("dit_fwd_in_txt_2.safetensors"), ctx)
    var t3 = _load_x(String("dit_fwd_in_txt_3.safetensors"), ctx)
    var cap_f32 = concat(2, ctx, t0, t1, t2, t3)                    # [1,16,11520] f32
    var cap_feats = cast_tensor(cap_f32, STDtype.BF16, ctx)         # raw concat (norm inside fwd)
    var ts_h = _load_x(String("dit_fwd_in_timestep.safetensors"), ctx).to_host(ctx)
    var t_model = ts_h[0]                                           # 0.5 → t_model*1000 = 500

    # ── LoRA-overlaid TRAINING forward (B=0 ⇒ velocity == base) ───────────────
    print("[forward] lens_forward_full_lora (B=0)  t_model =", t_model)
    var fo = lens_forward_full_lora[HLp, WLp, CAPLEN](
        packed, t_model, cap_feats, weights, loras, ctx
    )
    var vel_h = fo.velocity.to_host(ctx)                            # [1,64,128]
    for i in range(len(vel_h)):
        if not isfinite(vel_h[i]):
            raise Error(String("velocity non-finite at i=") + String(i))

    # ── loss DIRECTLY on the packed velocity (NOT the predict/scale/noise path) ─
    var target_h = Tensor.from_view(st.tensor_view(String("target_v")), ctx).to_host(ctx)
    var numel = N_IMG * IN_CH                                       # 64*128 = 8192
    if len(vel_h) != numel or len(target_h) != numel:
        raise Error(String("velocity/target numel mismatch: ") + String(len(vel_h))
                    + String(" / ") + String(len(target_h)) + String(" vs ") + String(numel))
    var loss = Float64(0.0)
    var dvel_host = List[Float32]()
    for i in range(numel):
        var d = vel_h[i] - target_h[i]
        loss += Float64(d) * Float64(d)
        dvel_host.append(Float32(2.0) * d / Float32(numel))
    loss /= Float64(numel)
    print("  loss(direct MSE) =", Float32(loss), " (oracle loss 4.027209)")

    var dsh = List[Int](); dsh.append(1); dsh.append(N_IMG); dsh.append(IN_CH)
    var d_velocity = Tensor.from_host(dvel_host, dsh^, STDtype.BF16, ctx)

    # ── hand-chained backward → 480 d_a/d_b ───────────────────────────────────
    print("[backward] lens_backward_full_lora")
    var grads = lens_backward_full_lora[HLp, WLp, CAPLEN](
        d_velocity, fo.saved, loras, ctx
    )
    if len(grads.block) != N_ADAPTERS:
        raise Error(String("backward returned ") + String(len(grads.block))
                    + String(" adapters, expected ") + String(N_ADAPTERS))

    # ── per-adapter compare d_B vs oracle dB_{idx} (dtype-aware ceiling gate) ──
    var n_pass = 0
    var n_nonzero = 0
    var n_zero = 0
    var n_zero_fail = 0
    var min_cos = Float32(2.0)
    var min_cos_idx = -1
    # "worst" = most-negative (mojo_cos - ceiling) gap among nonzero adapters
    var worst_gap = Float32(1.0e9)
    var worst_idx = -1
    var worst_cos = Float32(0.0)
    var worst_ceil = Float32(0.0)
    var zero_absmax = List[Float32]()
    var zero_names = List[String]()
    var slot_cos_sum = _zeros(LORA_SLOTS_PER_BLOCK)
    var slot_gap_sum = _zeros(LORA_SLOTS_PER_BLOCK)   # sum of (mojo_cos - ceiling)
    var slot_cnt = _zeros(LORA_SLOTS_PER_BLOCK)

    for b in range(LENS_N_BLOCKS):
        for slot in range(LORA_SLOTS_PER_BLOCK):
            var idx = b * LORA_SLOTS_PER_BLOCK + slot
            var oracle_db = Tensor.from_view(
                st.tensor_view(String("dB_") + String(idx)), ctx).to_host(ctx)
            var mojo_db = grads.block[idx].d_b.copy()
            if len(mojo_db) != len(oracle_db):
                raise Error(String("dB len mismatch idx ") + String(idx) + String(": ")
                            + String(len(mojo_db)) + String(" vs ") + String(len(oracle_db)))
            var oracle_absmax = _absmax(oracle_db)
            if oracle_absmax == Float32(0.0):
                # architecturally-zero adapter (oracle dB exactly 0) — unchanged bar
                n_zero += 1
                var mam = _absmax(mojo_db)
                zero_absmax.append(mam)
                zero_names.append(lora_module_prefix(b, slot))
                if mam >= ZERO_BAR:
                    n_zero_fail += 1
                    print("  ZERO-ADAPTER FAIL idx", idx, lora_module_prefix(b, slot),
                          "|d_B| absmax =", mam, ">=", ZERO_BAR)
            else:
                n_nonzero += 1
                var cos = _cosine(mojo_db, oracle_db)
                var ceil = ceiling[idx]
                # dtype-aware bar = min(0.999, ceiling - 0.01)
                var bar = COS_BAR
                var ceil_bar = ceil - CEIL_MARGIN
                if ceil_bar < bar:
                    bar = ceil_bar
                var gap = cos - ceil
                slot_cos_sum[slot] += cos
                slot_gap_sum[slot] += gap
                slot_cnt[slot] += Float32(1.0)
                if cos < min_cos:
                    min_cos = cos
                    min_cos_idx = idx
                if gap < worst_gap:
                    worst_gap = gap
                    worst_idx = idx
                    worst_cos = cos
                    worst_ceil = ceil
                if cos >= bar:
                    n_pass += 1
                else:
                    print("  ADAPTER FAIL idx", idx, lora_module_prefix(b, slot),
                          " mojo_cos =", cos, " ceiling =", ceil,
                          " bar =", bar, " gap =", gap)

    # ── report ────────────────────────────────────────────────────────────────
    print("")
    print("=== RESULTS ===")
    print("  nonzero adapters:", n_nonzero, " zero adapters:", n_zero)
    print("  PASS (dtype-aware ceiling gate):", n_pass, "of", n_nonzero, "nonzero adapters")
    if worst_idx >= 0:
        var wb = worst_idx // LORA_SLOTS_PER_BLOCK
        var ws = worst_idx % LORA_SLOTS_PER_BLOCK
        print("  WORST adapter by (mojo_cos - ceiling) gap: idx", worst_idx,
              "(" + lora_module_prefix(wb, ws) + ")")
        print("    mojo_cos =", worst_cos, " ceiling =", worst_ceil, " gap =", worst_gap)
    if min_cos_idx >= 0:
        var mb = min_cos_idx // LORA_SLOTS_PER_BLOCK
        var ms = min_cos_idx % LORA_SLOTS_PER_BLOCK
        print("  (lowest absolute mojo_cos =", min_cos, " at idx", min_cos_idx,
              "(" + lora_module_prefix(mb, ms) + "))")
    print("  zero-adapter |d_B| absmax (bar <", ZERO_BAR, "):")
    for i in range(len(zero_absmax)):
        print("    ", zero_names[i], "=", zero_absmax[i])
    print("  per-slot mean (mojo_cos - ceiling) gap (0..9 = img_qkv,txt_qkv,to_out,"
          "to_add_out,img_w1,img_w2,img_w3,txt_w1,txt_w2,txt_w3):")
    for slot in range(LORA_SLOTS_PER_BLOCK):
        if slot_cnt[slot] > 0.0:
            print("    slot", slot, "mean gap =", slot_gap_sum[slot] / slot_cnt[slot],
                  " mean cos =", slot_cos_sum[slot] / slot_cnt[slot],
                  " n =", Int(slot_cnt[slot]))

    var gate_ok = (n_pass == n_nonzero) and (n_zero_fail == 0)
    if not gate_ok:
        print("")
        print("  GATE FAIL:", n_nonzero - n_pass, "nonzero adapter(s) below the BF16",
              "ceiling bar and", n_zero_fail, "zero adapter(s) >=", ZERO_BAR)
        raise Error("BACKWARD PARITY FAIL")
    print("")
    print("  GATE OK: all", n_nonzero, "nonzero adapters within",
          CEIL_MARGIN, "of torch's BF16 ceiling (or >=", COS_BAR,
          ") and all", n_zero, "zero adapters |d_B| <", ZERO_BAR)
    print("=== smoke complete ===")
