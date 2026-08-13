# train_acestep.mojo — ACE-Step-1.5 xl-base LoRA trainer driver (Tier-3 T3.C, #14).
#
# Wires the training step (acestep_train_step) to the resident fused plain-AdamW
# over the 512 LoRA params, with global-norm grad clip, warmup+cosine LR, and
# LoRA-only PEFT checkpoint save. bs=1, the recipe optimizer (AdamW lr1e-4 wd0.01,
# max_grad_norm 1.0, cosine+warmup100). LoRA r8/α16 q/k/v/o self+cross ×32L.
#
# v1 dataset = the oracle batch (target_latents/context/encoder_hidden_states from
# the dump) — proves the driver loop end-to-end (loss + B-growth + saved LoRA). The
# .pt→Mojo cache converter (#10) generalizes the data source; this driver's loop is
# unchanged. Output under /home/alex/mojodiffusion/output/<run_name>/.
#
# The 256 adapters are train_step.LoraAdapter (host a/b + F32 moments); grads come
# out of acestep_train_step as device tensors → to_host → fused_lora_adamw_plain_step
# (folds the clip_scale, mutates a/b/moments in place) → re-upload a/b to the device
# `full` dict for the next forward. Faithful host-list AdamW (parity-proven); the
# device-resident-devgrads fused path is a speed follow-on (zimage precedent).
#
# Mojo 1.0.0b1, NVIDIA.

from max.gpu.host import DeviceContext
from std.memory import ArcPointer, alloc, UnsafePointer
from std.builtin.dtype import DType
from std.collections import Optional
from std.ffi import external_call
from std.sys import argv
from std.math import sqrt, cos as _cos
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.ffi import sys_system
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.parity import ParityHarness
from serenitymojo.models.dit.acestep_dit import AceStepDiTConfig
from serenitymojo.models.acestep.acestep_train_step import (
    acestep_train_step, acestep_apply_cfg_dropout,
)
from serenitymojo.models.acestep.acestep_cache_reader import (
    acestep_load_sample, acestep_read_manifest,
)
from serenitymojo.io.env import env_or, env_int
from serenitymojo.training.train_step import LoraAdapter, _randn, _zeros
from serenitymojo.training.lora_adamw_plain_fused import (
    fused_lora_adamw_plain_step_resident_preloaded_grads,
    lora_adamw_plain_device_state_copy_device_grad_pair,
    lora_adamw_plain_device_state_init,
    LoraAdamWPlainDeviceState,
)
from serenitymojo.training.lora_save import NamedLora, save_lora_peft
from serenitymojo.ops.tensor_algebra import add as _dev_add, mul_scalar as _dev_muls

comptime SP = 64
comptime L = 64
comptime NH = 32
comptime LAYERS = 32
comptime PI = Float32(3.14159265358979)
comptime TArc = ArcPointer[Tensor]
comptime _CPtr = UnsafePointer[UInt8, MutExternalOrigin]


# ── argv helpers (UI config-runner: positional args, "-"/"" = keep default) ───
def _atof(s: String) -> Float64:
    var n = s.byte_length()
    var buf = alloc[UInt8](n + 1)
    var src = s.as_bytes()
    for i in range(n):
        buf[i] = src[i]
    buf[n] = 0
    var v = external_call["atof", Float64](_CPtr(unsafe_from_address=Int(buf)))
    buf.free()
    return v


def _arg(args: List[String], i: Int, d: String) -> String:
    if i >= len(args):
        return d
    var s = args[i]
    if s == "-" or len(s) == 0:
        return d
    return s


def _argi(args: List[String], i: Int, d: Int) -> Int:
    var s = _arg(args, i, String("-"))
    if s == "-":
        return d
    var v = 0
    var bytes = s.as_bytes()
    for j in range(len(bytes)):
        var c = Int(bytes[j])
        if c < 48 or c > 57:
            return d
        v = v * 10 + (c - 48)
    return v


def _argf(args: List[String], i: Int, d: Float32) -> Float32:
    var s = _arg(args, i, String("-"))
    if s == "-":
        return d
    return Float32(_atof(s))


@fieldwise_init
struct AceStepTrainConfig(Copyable, Movable):
    var checkpoint_dir: String   # xl-base decoder ckpt dir
    var output_dir: String       # base output root
    var run_name: String
    var steps: Int
    var lr: Float32
    var weight_decay: Float32
    var lora_rank: Int
    var lora_alpha: Float32
    var cfg_ratio: Float32
    var max_grad_norm: Float32
    var warmup: Int
    var save_every: Int
    var seed: UInt64
    var grad_accum: Int          # micro-steps per optimizer step (recipe: 4)

    @staticmethod
    def default() -> AceStepTrainConfig:
        # steps = OPTIMIZER steps; total micro-steps = steps * grad_accum. Smoke
        # defaults (4 opt x 4 accum = 16 micro); a real run sets steps higher.
        return AceStepTrainConfig(
            "/home/alex/ACE-Step-1.5/checkpoints/acestep-v15-xl-base",
            "/home/alex/mojodiffusion/output", "acestep_xlbase_lora",
            4, Float32(1.0e-4), Float32(0.01), 8, Float32(16.0),
            Float32(0.0), Float32(1.0), 100, 2, 42, 4,
        )


# ── the 8 LoRA slots (attn, proj, in_f, out_f), slot order == the backward's ──
def _slot_attn(s: Int) -> String:
    return String("self_attn") if s < 4 else String("cross_attn")


def _slot_proj(s: Int) -> String:
    var m = s % 4
    if m == 0: return String("q_proj")
    if m == 1: return String("k_proj")
    if m == 2: return String("v_proj")
    return String("o_proj")


def _slot_in(s: Int) -> Int:
    # q/k/v read H=2560; o reads nh*dh=4096.
    return 4096 if (s % 4) == 3 else 2560


def _slot_out(s: Int) -> Int:
    var m = s % 4
    if m == 0: return 4096   # q: nh*dh
    if m == 3: return 2560   # o: H
    return 1024              # k/v: nkv*dh


def _bf16(st: ShardedSafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    return cast_tensor(Tensor.from_view(st.tensor_view(name), ctx), STDtype.BF16, ctx)


# CFG dropout: load the model's `null_condition_emb` [1,1,2048] (a TOP-LEVEL
# Parameter, not decoder.* — present in the checkpoint index) and expand it to
# [1,L,2048] == ehs shape (the upstream `null.expand_as(ehs)`; every L position
# gets the same null vector).
def _load_null_cond(ckpt_dir: String, seq_len: Int, ctx: DeviceContext) raises -> Tensor:
    var st = ShardedSafeTensors.open(ckpt_dir)
    var null_t = Tensor.from_view(st.tensor_view("null_condition_emb"), ctx)  # [1,1,D]
    var nv = null_t.to_host(ctx)          # F32, D values
    var d = null_t.numel()                # 2048
    var expanded = List[Float32]()
    for _ in range(seq_len):
        for j in range(d):
            expanded.append(nv[j])
    return Tensor.from_host(expanded^, [1, seq_len, d], STDtype.BF16, ctx)


def _load_base(ckpt_dir: String, ctx: DeviceContext) raises -> Dict[String, ArcPointer[Tensor]]:
    var xl = ShardedSafeTensors.open(ckpt_dir)
    var full = Dict[String, ArcPointer[Tensor]]()
    for nm in xl.names():
        var n = String(nm)
        if n.startswith("decoder."):
            full[n] = ArcPointer(_bf16(xl, n, ctx))
    return full^


# ── build 256 LoRA adapters (A~randn·0.01, B=0) in slot order + their keys ────
def _build_adapters(
    rank: Int, alpha: Float32, seed: UInt64,
    mut adapters: List[LoraAdapter], mut keys: List[String],
) raises:
    var scale = alpha / Float32(rank)
    for li in range(LAYERS):
        for s in range(8):
            var in_f = _slot_in(s)
            var out_f = _slot_out(s)
            var sd = seed + UInt64(li * 8 + s) + 1
            adapters.append(
                LoraAdapter(
                    _randn(rank * in_f, sd, Float32(0.01)),   # A
                    _zeros(out_f * rank),                     # B = 0
                    rank, in_f, out_f, scale,
                    _zeros(rank * in_f), _zeros(rank * in_f),
                    _zeros(out_f * rank), _zeros(out_f * rank),
                )
            )
            keys.append(String("decoder.layers.") + String(li) + "." + _slot_attn(s) + "." + _slot_proj(s))


# RESIDENT: point the `full` dict's LoRA tensors at sub-buffer VIEWS of the
# optimizer's dev_p — the in-place AdamW update IS the next step's weights (no
# per-step re-upload; the forward's `_layer_bw` clones the live view). Krea2's
# proven `_resident_lora_adapter_a3` pattern (dev_p.create_sub_buffer, ×2 bytes/bf16).
def _acestep_devp_views(
    adapters: List[LoraAdapter], keys: List[String],
    mut state: LoraAdamWPlainDeviceState,
    mut full: Dict[String, ArcPointer[Tensor]], ctx: DeviceContext,
) raises:
    for i in range(len(adapters)):
        var n_a = len(adapters[i].a)
        var n_b = len(adapters[i].b)
        var a_off = state.elem_offset(i, False)
        var b_off = state.elem_offset(i, True)
        var a_view = Tensor(
            state.dev_p.create_sub_buffer[DType.uint8](a_off * 2, n_a * 2),
            [adapters[i].rank, adapters[i].in_f], STDtype.BF16)
        var b_view = Tensor(
            state.dev_p.create_sub_buffer[DType.uint8](b_off * 2, n_b * 2),
            [adapters[i].out_f, adapters[i].rank], STDtype.BF16)
        full[keys[i] + ".lora_A"] = ArcPointer(a_view^)
        full[keys[i] + ".lora_B"] = ArcPointer(b_view^)




def _lr_at(step: Int, base_lr: Float32, warmup: Int, total: Int) -> Float32:
    if step <= warmup:
        return base_lr * Float32(step) / Float32(warmup)
    var denom = total - warmup
    if denom < 1:
        denom = 1
    var prog = Float32(step - warmup) / Float32(denom)
    return base_lr * Float32(0.5) * (Float32(1.0) + _cos(PI * prog))


def _sum_abs_b(adapters: List[LoraAdapter]) -> Float32:
    var s = Float32(0.0)
    for i in range(len(adapters)):
        for v in adapters[i].b:
            var f = Float32(v)
            if f < 0.0:
                f = -f
            s += f
    return s


def _save(adapters: List[LoraAdapter], keys: List[String], dir: String, run: String, step: Int, ctx: DeviceContext) raises -> String:
    var named = List[NamedLora]()
    for i in range(len(adapters)):
        named.append(NamedLora(keys[i], adapters[i].copy()))
    var path = dir + "/" + run + "_step" + String(step) + ".safetensors"
    var n = save_lora_peft(named, path, ctx)
    print("  saved", n, "LoRA pairs →", path)
    return path


# ── the training loop: returns (losses, sum|B|_final, final_ckpt_path) ────────
struct AceTrainResult(Movable):
    var losses: List[Float32]
    var sum_abs_b: Float32
    var ckpt: String

    def __init__(out self, var losses: List[Float32], sum_abs_b: Float32, var ckpt: String):
        self.losses = losses^
        self.sum_abs_b = sum_abs_b
        self.ckpt = ckpt^


def acestep_train(
    cfg: AceStepTrainConfig,
    sample_paths: List[String],   # cache sample safetensors (STREAMED per step)
    ctx: DeviceContext,
    null_cond: Optional[Tensor] = None,   # CFG-dropout null emb [1,L,2048] (if cfg_ratio>0)
) raises -> AceTrainResult:
    if len(sample_paths) == 0:
        raise Error("acestep_train: no samples")
    var work = cfg.output_dir + "/" + cfg.run_name
    _ = sys_system(String("mkdir -p ") + work)
    var cfgm = AceStepDiTConfig.xl_base()
    var lora_scale = cfg.lora_alpha / Float32(cfg.lora_rank)

    print("loading xl-base decoder…")
    var full = _load_base(cfg.checkpoint_dir, ctx)
    var adapters = List[LoraAdapter]()
    var keys = List[String]()
    _build_adapters(cfg.lora_rank, cfg.lora_alpha, cfg.seed, adapters, keys)
    # resident AdamW: params + moments live on device (dev_p); `full` LoRA = dev_p
    # views (no per-step re-upload). host adapters mirror dev_p (synced each step
    # for save). state must outlive the views + the loop.
    var state = lora_adamw_plain_device_state_init(adapters, 0, len(adapters), ctx)
    _acestep_devp_views(adapters, keys, state, full, ctx)
    var accum = cfg.grad_accum if cfg.grad_accum > 0 else 1
    var micro_total = cfg.steps * accum
    print("adapters:", len(adapters), " | samples:", len(sample_paths),
          " |", cfg.steps, "opt steps x", accum, "accum =", micro_total, "micro-steps")

    # FULLY DEVICE-RESIDENT grad path (sync-skip): accumulate `accum` micro-step
    # grads ON DEVICE in F32 (cast bf16→F32, add — same values/order as the host
    # F32 accumulation), MEAN (÷accum), stage into state.dev_g via
    # copy_device_grad_pair, and step with preloaded_grads (norm+clip computed
    # ON DEVICE via on_device_grad_stats; sync_params_to_host only at save/final).
    # No grad to_host, no per-step param readback. Matches the upstream mean
    # convention (loss/accum → mean grad).
    var acc_a = List[TArc]()
    var acc_b = List[TArc]()
    var losses = List[Float32]()
    var opt_step = 0
    var win_loss = Float32(0.0)
    var win_micro = 0
    for micro in range(1, micro_total + 1):
        var si = (micro - 1) % len(sample_paths)            # cycle through the dataset
        var smp = acestep_load_sample(sample_paths[si], ctx)  # stream from disk
        var bg = acestep_train_step[SP, L, NH, LAYERS](
            smp.target_latents, smp.context, smp.ehs,
            full, cfgm, lora_scale, cfg.seed + UInt64(micro) * 1000,
            ctx, Float32(-0.4), Float32(1.0), cfg.cfg_ratio, null_cond,
        )
        if win_micro == 0:
            acc_a = List[TArc]()
            acc_b = List[TArc]()
            for i in range(len(adapters)):
                acc_a.append(TArc(cast_tensor(bg.d_a[i][], STDtype.F32, ctx)))
                acc_b.append(TArc(cast_tensor(bg.d_b[i][], STDtype.F32, ctx)))
        else:
            for i in range(len(adapters)):
                acc_a[i] = TArc(_dev_add(acc_a[i][], cast_tensor(bg.d_a[i][], STDtype.F32, ctx), ctx))
                acc_b[i] = TArc(_dev_add(acc_b[i][], cast_tensor(bg.d_b[i][], STDtype.F32, ctx), ctx))
        win_loss += bg.loss
        win_micro += 1
        if win_micro < accum and micro != micro_total:
            continue

        # ── optimizer step: mean → stage device grads → resident preloaded step ──
        opt_step += 1
        var inv = Float32(1.0) / Float32(win_micro)
        for i in range(len(adapters)):
            var ga = TArc(_dev_muls(acc_a[i][], inv, ctx))   # F32 mean, numel = rank*in
            var gb = TArc(_dev_muls(acc_b[i][], inv, ctx))
            lora_adamw_plain_device_state_copy_device_grad_pair(state, i, ga, gb, ctx)
        var lr = _lr_at(opt_step, cfg.lr, cfg.warmup, cfg.steps)
        var do_sync = (opt_step % cfg.save_every == 0) or (micro == micro_total)
        var norm = fused_lora_adamw_plain_step_resident_preloaded_grads(
            state, adapters, opt_step, lr,
            Float32(0.9), Float32(0.999), Float32(1.0e-8), cfg.weight_decay, ctx,
            Float32(1.0), do_sync, cfg.max_grad_norm,   # clip folded on device
        )
        var mean_loss = win_loss / Float32(win_micro)
        losses.append(mean_loss)
        print("opt", opt_step, " loss", mean_loss, " lr", lr, " norm", norm, " sync", do_sync)
        win_loss = Float32(0.0)
        win_micro = 0
        if opt_step % cfg.save_every == 0:
            _ = _save(adapters, keys, work, cfg.run_name, opt_step, ctx)

    var ckpt = _save(adapters, keys, work, cfg.run_name, opt_step, ctx)
    return AceTrainResult(losses^, _sum_abs_b(adapters), ckpt^)


# ── entry: UI config-runner (positional argv) OR CLI smoke (env vars) ─────────
# serenity-trainer's `acestep` argv shape (webui/src/main.rs launch match):
#   argv[1]=checkpoint_dir  [2]=cache_dir(-=oracle)  [3]=output_dir  [4]=run_name
#   [5]=steps  [6]=grad_accum  [7]=lr  [8]=rank  [9]=alpha  [10]=save_every
#   [11]=cfg_ratio_pct  [12]=seed   ("-"/"" keeps the recipe default). No argv →
#   env-var CLI smoke (unchanged: $ACESTEP_STEPS/ACCUM/CFG_PCT/CACHE).
def main() raises:
    var ctx = DeviceContext()
    var cfg = AceStepTrainConfig.default()

    var raw = argv()
    var args = List[String]()
    for i in range(len(raw)):
        args.append(String(raw[i]))
    var cache_dir = String("")

    if len(args) > 1:
        print("UI config-runner: parsing argv (", len(args) - 1, "args)")
        cfg.checkpoint_dir = _arg(args, 1, cfg.checkpoint_dir)
        cache_dir = _arg(args, 2, String(""))          # "-" → oracle dump
        cfg.output_dir = _arg(args, 3, cfg.output_dir)
        cfg.run_name = _arg(args, 4, cfg.run_name)
        cfg.steps = _argi(args, 5, cfg.steps)
        cfg.grad_accum = _argi(args, 6, cfg.grad_accum)
        cfg.lr = _argf(args, 7, cfg.lr)
        cfg.lora_rank = _argi(args, 8, cfg.lora_rank)
        cfg.lora_alpha = _argf(args, 9, cfg.lora_alpha)
        cfg.save_every = _argi(args, 10, cfg.save_every)
        cfg.cfg_ratio = Float32(_argi(args, 11, 0)) / Float32(100.0)
        cfg.seed = UInt64(_argi(args, 12, Int(cfg.seed)))
    else:
        cfg.steps = env_int("ACESTEP_STEPS", cfg.steps)          # optimizer steps
        cfg.grad_accum = env_int("ACESTEP_ACCUM", cfg.grad_accum)  # micro-steps/opt-step
        cfg.cfg_ratio = Float32(env_int("ACESTEP_CFG_PCT", 0)) / Float32(100.0)
        cache_dir = env_or("ACESTEP_CACHE", "")
    var paths = List[String]()
    if len(cache_dir) > 0:
        print("cache:", cache_dir)
        paths = acestep_read_manifest(cache_dir)
    else:
        print("no $ACESTEP_CACHE — training on the oracle dump (v1 smoke)")
        paths.append("/home/alex/mojodiffusion/serenitymojo/models/acestep/parity/acestep_train_ref/acestep_train_ref.safetensors")

    # CFG dropout: load the real null_condition_emb + verify the mechanism.
    var null_opt = Optional[Tensor](None)
    if cfg.cfg_ratio > Float32(0.0):
        print("CFG dropout ON: ratio", cfg.cfg_ratio, "— loading null_condition_emb")
        var null_cond = _load_null_cond(cfg.checkpoint_dir, L, ctx)   # [1,L,2048]
        # direct mechanism check vs the first sample's ehs (real null).
        var ehs0 = _bf16(ShardedSafeTensors.open(paths[0]), "encoder_hidden_states", ctx)
        var kept = acestep_apply_cfg_dropout(ehs0, null_cond, Float32(0.0), 5, ctx)   # keep
        var dropped = acestep_apply_cfg_dropout(ehs0, null_cond, Float32(1.0), 5, ctx)  # → null
        var ph = ParityHarness(0.999)
        var rk = ph.compare(kept, ehs0.to_host(ctx), ctx)
        var rd = ph.compare(dropped, null_cond.to_host(ctx), ctx)
        print("  CFG check: ratio0 keeps ehs cos=", rk.cos, " ratio1→null cos=", rd.cos)
        null_opt = Optional[Tensor](null_cond^)

    var res = acestep_train(cfg, paths, ctx, null_opt)

    # ── smoke asserts ──
    var all_finite = True
    var in_range = True
    for i in range(len(res.losses)):
        var lv = res.losses[i]
        if lv != lv:   # NaN
            all_finite = False
        if lv < 0.5 or lv > 5.0:
            in_range = False
    var first = res.losses[0]
    var last = res.losses[len(res.losses) - 1]
    print("--- SMOKE SUMMARY ---")
    print("loss first =", first, " last =", last, " (stochastic; B-growth is the signal)")
    print("sum|B| after training =", res.sum_abs_b, " (init 0 → must be > 0)")
    print("final ckpt:", res.ckpt)
    var b_grew = res.sum_abs_b > Float32(0.0)
    if all_finite and in_range and b_grew:
        print("GATE: PASS")
    else:
        print("GATE: FAIL  (finite=", all_finite, " in_range=", in_range, " b_grew=", b_grew, ")")
