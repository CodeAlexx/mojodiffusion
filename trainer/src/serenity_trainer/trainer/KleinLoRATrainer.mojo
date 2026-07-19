# KleinLoRATrainer.mojo — cached Klein/Flux2 LoRA train driver.
#
# This runner consumes precomputed Klein cache tensors. It intentionally does not
# keep raw images, text encoders, or VAE weights resident during LoRA training.
# Raw JPEG/TXT datasets must first be cached into latent/text tensors.

from std.collections import Optional
from std.gpu.host import DeviceContext
from std.math import sqrt
from std.os import makedirs
from std.time import perf_counter, perf_counter_ns

from serenitymojo.autograd import Tape
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.loss_swiglu_backward import mse_backward
from serenitymojo.ops.tensor_algebra import mul, reshape_owned as _reshape_owned, sub
from serenitymojo.tensor import Tensor

from serenity_trainer.model.klein.double_block import DoubleBlockWeights
from serenity_trainer.model.klein.single_block import SingleBlockWeights
from serenity_trainer.model.klein.klein_stack_lora import (
    KleinLoraGrads,
    KleinLoraSet,
    klein_lora_adamw_step,
    klein_lora_set_to_device,
)
from serenity_trainer.model.klein.weights import (
    KleinStepModWeights,
    build_klein_vec_silu,
    load_double_block_weights,
    load_klein_stack_base,
    load_klein_step_mod_weights,
    load_single_block_weights,
)
from serenity_trainer.model.KleinModel import (
    KDIM,
    KH,
    KDh,
    KNUM_DOUBLE,
    KNUM_SINGLE,
    KTIMESTEP_DIM,
    build_klein9b_lora_set,
    build_klein_rope_tables_port,
)
from serenity_trainer.model.KleinVAE import KLEIN_BN_EPS
from serenity_trainer.modelLoader.Flux2RuntimeLoader import load_flux2_lora_fused
from serenity_trainer.modelSaver.flux2.Flux2LoRASaver import save_flux2_lora
from serenity_trainer.modelSetup.Flux2LoRASetup import make_flux2_lora_spec
from serenity_trainer.trainer.TrainState import TrainProgress
from serenity_trainer.util.config.TrainConfig import TrainConfig


comptime DEFAULT_KLEIN_VAE = "/home/alex/.serenity/models/vaes/flux2-vae.safetensors"
comptime _STATELESS_LORA_FILE = "klein_lora_last.safetensors"


@fieldwise_init
struct KleinLoRATrainRunConfig(Copyable, Movable):
    var checkpoint_path: String
    var cache_path: String
    var dataset_path: String
    var output_dir: String
    var vae_path: String
    var resume_lora_path: String
    var steps: Int
    var default_timestep: Int
    var save_every_steps: Int
    var progress_file_path: String

    @staticmethod
    def defaults(
        checkpoint_path: String,
        cache_path: String,
        dataset_path: String,
        output_dir: String,
    ) -> KleinLoRATrainRunConfig:
        return KleinLoRATrainRunConfig(
            checkpoint_path=checkpoint_path,
            cache_path=cache_path,
            dataset_path=dataset_path,
            output_dir=output_dir,
            vae_path=String(DEFAULT_KLEIN_VAE),
            resume_lora_path=String(""),
            steps=10,
            default_timestep=250,
            save_every_steps=0,
            progress_file_path=String(""),
        )


@fieldwise_init
struct KleinLoRATrainSummary(Copyable, Movable):
    var steps_ran: Int
    var optimizer_steps: Int
    var last_loss: Float32
    var grad_norm: Float64
    var elapsed_seconds: Float64
    var seconds_per_step: Float64
    var lora_path: String
    var progress: TrainProgress


@fieldwise_init
struct _GradStats(Copyable, Movable):
    var elems: Int
    var nonfinite: Int
    var abs_sum: Float64
    var sumsq: Float64


def train_klein_lora_auto(
    cfg: TrainConfig,
    run_cfg: KleinLoRATrainRunConfig,
    ctx: DeviceContext,
) raises -> KleinLoRATrainSummary:
    var src = ShardedSafeTensors.open(run_cfg.cache_path)
    if _has_tensor(src, String("latent")) or _has_tensor(src, String("trace.latent")):
        var latent_key = String("latent")
        if not _has_tensor(src, latent_key):
            latent_key = String("trace.latent")
        var lat_shape = src.tensor_info(latent_key).shape.copy()
        var txt_key = _text_key(src)
        var txt_shape = src.tensor_info(txt_key).shape.copy()
        var ntxt: Int
        if len(txt_shape) == 3:
            ntxt = txt_shape[1]
        else:
            ntxt = txt_shape[0]
        if len(lat_shape) == 4 and lat_shape[2] == 32 and lat_shape[3] == 32 and ntxt == 48:
            return train_klein_lora_from_cache[16, 16, 48](cfg, run_cfg, ctx)
        if len(lat_shape) == 4 and lat_shape[2] == 64 and lat_shape[3] == 64 and ntxt == 512:
            return train_klein_lora_from_cache[32, 32, 512](cfg, run_cfg, ctx)
        raise Error(
            String("Klein cache shape unsupported: latent=")
            + String(lat_shape)
            + String(" text=")
            + String(txt_shape)
            + String(". Supported: 512 cache [1,32,64,64]+[512,*] or parity [1,32,32,32]+[48,*].")
        )
    raise Error(
        String("Klein cached trainer needs a .safetensors cache with keys ")
        + String("'latent' and 'txt'/'encoder_hidden_states'. Raw JPEG datasets ")
        + String("must be cached before training.")
    )


def train_klein_lora_from_cache[HL: Int, WL: Int, NTXT: Int](
    cfg_in: TrainConfig,
    run_cfg: KleinLoRATrainRunConfig,
    ctx: DeviceContext,
) raises -> KleinLoRATrainSummary:
    if run_cfg.steps < 1:
        raise Error("train_klein_lora_from_cache: steps must be >= 1")
    makedirs(run_cfg.output_dir, exist_ok=True)

    var cfg = cfg_in.copy()
    _force_constant_timestep(cfg, run_cfg.default_timestep)

    var cache = ShardedSafeTensors.open(run_cfg.cache_path)
    var latent = cast_tensor(
        Tensor.from_view(cache.tensor_view(_latent_key(cache)), ctx),
        STDtype.BF16,
        ctx,
    )
    var txt = cast_tensor(
        Tensor.from_view(cache.tensor_view(_text_key(cache)), ctx),
        STDtype.BF16,
        ctx,
    )
    txt = _text_2d[NTXT](txt^, ctx)
    var bn_inv_scale = _load_bn_inv_scale(cache, run_cfg.vae_path, ctx)
    var bn_mean = _load_bn_mean(cache, run_cfg.vae_path, ctx)

    var load0 = perf_counter_ns()
    var ckpt = SafeTensors.open(run_cfg.checkpoint_path)
    var ts = Tensor.from_host([Float32(run_cfg.default_timestep)], [1], STDtype.F32, ctx)
    var vec_silu = build_klein_vec_silu(ckpt, ts, KTIMESTEP_DIM, KDIM, ctx)
    var base = load_klein_stack_base(ckpt, vec_silu, KDIM, ctx)
    var step_mod_w = load_klein_step_mod_weights(ckpt, KDIM, ctx)
    var dbw = List[DoubleBlockWeights]()
    for bi in range(KNUM_DOUBLE):
        dbw.append(load_double_block_weights(ckpt, bi, ctx))
    var sbw = List[SingleBlockWeights]()
    for bi in range(KNUM_SINGLE):
        sbw.append(load_single_block_weights(ckpt, bi, ctx))
    var load1 = perf_counter_ns()

    var rank = Int(cfg.lora_rank)
    if rank < 1:
        rank = 1
    var alpha = cfg.lora_alpha
    if alpha <= Float32(0.0):
        alpha = Float32(rank)

    var lora_host: KleinLoraSet
    if run_cfg.resume_lora_path.byte_length() > 0:
        lora_host = load_flux2_lora_fused(
            run_cfg.resume_lora_path, KNUM_DOUBLE, KNUM_SINGLE, ctx
        )
    else:
        lora_host = build_klein9b_lora_set(rank, alpha)

    var rope = build_klein_rope_tables_port[HL * WL, NTXT, KH, KDh](ctx, STDtype.BF16)
    var cos_t = rope[0].clone(ctx)
    var sin_t = rope[1].clone(ctx)

    var progress = TrainProgress()
    var opt_step = 0
    var last_loss = Float32(0.0)
    var last_grad_norm = Float64(0.0)
    var smooth_loss = Float32(0.0)
    var smooth_ready = False
    var train0 = perf_counter()

    _append_klein_status(
        run_cfg.progress_file_path,
        String("Staged Klein 9B base in ") + String(_sec_from_ns(load0, load1)) + String("s"),
        0,
        run_cfg.steps,
        Float32(0.0),
        Float32(0.0),
        Float32(cfg.learning_rate),
    )

    for local_step in range(run_cfg.steps):
        var step0 = perf_counter()
        var lora_dev = klein_lora_set_to_device(lora_host, ctx)
        var spec = make_flux2_lora_spec[HL, WL, NTXT](
            base.copy(),
            dbw.copy(),
            sbw.copy(),
            _clone_step_mod_weights(step_mod_w, ctx),
            lora_dev^,
            cos_t.clone(ctx),
            sin_t.clone(ctx),
            bn_inv_scale.clone(ctx),
            bn_mean.clone(ctx),
            txt.clone(ctx),
            latent.clone(ctx),
            UInt64(0x4B1E_0000 + local_step),
            Float32(1.0),
            False,
        )
        var tape = Tape()
        var out = spec.predict(tape, cfg, local_step, ctx)
        last_loss = _mse_host(out.predicted, out.target, ctx)
        var d_flow = mse_backward(out.predicted, out.target, ctx)
        var grads = spec.backward_lora(d_flow, ctx)
        var stats = _grad_stats(grads)
        last_grad_norm = sqrt(stats.sumsq)
        if stats.nonfinite != 0:
            raise Error("Klein LoRA gradient contains nonfinite values")
        opt_step += 1
        klein_lora_adamw_step(
            lora_host,
            grads,
            opt_step,
            cfg.learning_rate,
            ctx,
            cfg.beta1,
            cfg.beta2,
            cfg.eps,
            cfg.weight_decay,
            cfg.stochastic_rounding,
        )
        progress.next_step(cfg.batch_size)
        if not smooth_ready:
            smooth_loss = last_loss
            smooth_ready = True
        else:
            smooth_loss = smooth_loss * Float32(0.99) + last_loss * Float32(0.01)

        if run_cfg.save_every_steps > 0 and opt_step % run_cfg.save_every_steps == 0:
            save_flux2_lora(
                lora_host,
                KDIM,
                _step_lora_path(run_cfg.output_dir, opt_step),
                ctx,
                STDtype.BF16,
            )

        var elapsed = perf_counter() - train0
        var speed = perf_counter() - step0
        _append_klein_progress(
            run_cfg.progress_file_path,
            run_cfg,
            progress,
            cfg,
            last_loss,
            smooth_loss,
            Float32(last_grad_norm),
            Float32(speed),
            elapsed,
        )

    var elapsed_total = perf_counter() - train0
    var final_lora = _final_lora_path(run_cfg.output_dir)
    save_flux2_lora(lora_host, KDIM, final_lora, ctx, STDtype.BF16)
    _append_klein_status(
        run_cfg.progress_file_path,
        String("Finished Klein LoRA: saved ") + final_lora.copy(),
        run_cfg.steps,
        run_cfg.steps,
        last_loss,
        Float32(last_grad_norm),
        Float32(cfg.learning_rate),
    )
    return KleinLoRATrainSummary(
        run_cfg.steps,
        opt_step,
        last_loss,
        last_grad_norm,
        elapsed_total,
        elapsed_total / Float64(run_cfg.steps),
        final_lora,
        progress.copy(),
    )


def _force_constant_timestep(mut cfg: TrainConfig, timestep: Int):
    cfg.dynamic_timestep_shifting = False
    cfg.timestep_shift = Float32(1.0)
    var lo = Float32(timestep) / Float32(1000.0)
    var hi = Float32(timestep + 1) / Float32(1000.0)
    cfg.min_noising_strength = lo
    cfg.max_noising_strength = hi


def _has_tensor(st: ShardedSafeTensors, key: String) -> Bool:
    return key in st.name_to_shard


def _latent_key(st: ShardedSafeTensors) raises -> String:
    if _has_tensor(st, String("latent")):
        return String("latent")
    if _has_tensor(st, String("trace.latent")):
        return String("trace.latent")
    raise Error("Klein cache missing latent tensor")


def _text_key(st: ShardedSafeTensors) raises -> String:
    if _has_tensor(st, String("txt")):
        return String("txt")
    if _has_tensor(st, String("encoder_hidden_states")):
        return String("encoder_hidden_states")
    if _has_tensor(st, String("trace.encoder_hidden_states")):
        return String("trace.encoder_hidden_states")
    raise Error("Klein cache missing text tensor")


def _text_2d[NTXT: Int](var txt: Tensor, ctx: DeviceContext) raises -> Tensor:
    var sh = txt.shape()
    if len(sh) == 3 and sh[0] == 1 and sh[1] == NTXT:
        return _reshape_owned(txt^, [NTXT, sh[2]])
    if len(sh) == 2 and sh[0] == NTXT:
        return txt^
    raise Error(String("Klein cache text shape mismatch: ") + String(sh))


def _load_bn_inv_scale(
    cache: ShardedSafeTensors,
    vae_path: String,
    ctx: DeviceContext,
) raises -> Tensor:
    if _has_tensor(cache, String("bn_inv_scale")):
        return Tensor.from_view(cache.tensor_view(String("bn_inv_scale")), ctx)
    if _has_tensor(cache, String("bn_var")):
        var bn_var = Tensor.from_view(cache.tensor_view(String("bn_var")), ctx)
        var host = bn_var.to_host(ctx)
        var vals = List[Float32]()
        for i in range(len(host)):
            vals.append(Float32(1.0) / sqrt(host[i] + KLEIN_BN_EPS))
        return Tensor.from_host(vals^, [len(host)], STDtype.F32, ctx)
    var source = ShardedSafeTensors.open(vae_path)
    var bn_var = Tensor.from_view(source.tensor_view(String("bn.running_var")), ctx)
    var host = bn_var.to_host(ctx)
    var vals = List[Float32]()
    for i in range(len(host)):
        vals.append(Float32(1.0) / sqrt(host[i] + KLEIN_BN_EPS))
    return Tensor.from_host(vals^, [len(host)], STDtype.F32, ctx)


def _load_bn_mean(
    cache: ShardedSafeTensors,
    vae_path: String,
    ctx: DeviceContext,
) raises -> Tensor:
    if _has_tensor(cache, String("bn_mean")):
        return Tensor.from_view(cache.tensor_view(String("bn_mean")), ctx)
    if _has_tensor(cache, String("bn.running_mean")):
        return Tensor.from_view(cache.tensor_view(String("bn.running_mean")), ctx)
    var vae = ShardedSafeTensors.open(vae_path)
    return Tensor.from_view(vae.tensor_view(String("bn.running_mean")), ctx)


def _clone_opt_tensor(x: Optional[Tensor], ctx: DeviceContext) raises -> Optional[Tensor]:
    if x:
        return Optional[Tensor](x.value().clone(ctx))
    return Optional[Tensor](None)


def _clone_step_mod_weights(w: KleinStepModWeights, ctx: DeviceContext) raises -> KleinStepModWeights:
    return KleinStepModWeights(
        w.t_in.clone(ctx),
        w.t_out.clone(ctx),
        _clone_opt_tensor(w.g_in, ctx),
        _clone_opt_tensor(w.g_out, ctx),
        w.img_mod.clone(ctx),
        w.txt_mod.clone(ctx),
        w.single_mod.clone(ctx),
        w.final_mod.clone(ctx),
    )


def _mse_host(a: Tensor, b: Tensor, ctx: DeviceContext) raises -> Float32:
    var d = sub(a, b, ctx)
    var sq = mul(d, d, ctx)
    var h = sq.to_host(ctx)
    var s = Float32(0.0)
    for i in range(len(h)):
        s += h[i]
    return s / Float32(len(h))


def _is_nonfinite(x: Float32) -> Bool:
    if x != x:
        return True
    return (x - x) != Float32(0.0)


def _scan_values(values: List[Float32], var stats: _GradStats) -> _GradStats:
    for i in range(len(values)):
        var x = values[i]
        stats.elems += 1
        if _is_nonfinite(x):
            stats.nonfinite += 1
        else:
            var xf = Float64(x)
            stats.sumsq += xf * xf
            if x < Float32(0.0):
                stats.abs_sum -= Float64(x)
            else:
                stats.abs_sum += Float64(x)
    return stats^


def _grad_stats(grads: KleinLoraGrads) -> _GradStats:
    var stats = _GradStats(0, 0, Float64(0.0), Float64(0.0))
    for i in range(len(grads.dbl_d_a)):
        stats = _scan_values(grads.dbl_d_a[i], stats^)
    for i in range(len(grads.dbl_d_b)):
        stats = _scan_values(grads.dbl_d_b[i], stats^)
    for i in range(len(grads.sgl_d_a)):
        stats = _scan_values(grads.sgl_d_a[i], stats^)
    for i in range(len(grads.sgl_d_b)):
        stats = _scan_values(grads.sgl_d_b[i], stats^)
    return stats^


def _append_klein_status(
    path: String,
    status: String,
    step: Int,
    max_steps: Int,
    loss: Float32,
    grad_norm: Float32,
    lr: Float32,
) raises:
    if path.byte_length() == 0:
        return
    var f = open(path, "a")
    f.write(
        String("[Serenity-callback] progress epoch 1/1 | step ")
        + String(step)
        + String("/")
        + String(max_steps)
        + String(" | global_step ")
        + String(step)
        + String(" | loss ")
        + String(loss)
        + String(" | smooth_loss ")
        + String(loss)
        + String(" | grad_norm ")
        + String(grad_norm)
        + String(" | lr ")
        + String(lr)
        + String(" | status ")
        + status
    )
    f.write("\n")
    f.close()


def _append_klein_progress(
    path: String,
    run_cfg: KleinLoRATrainRunConfig,
    progress: TrainProgress,
    cfg: TrainConfig,
    loss: Float32,
    smooth_loss: Float32,
    grad_norm: Float32,
    speed_seconds: Float32,
    elapsed_seconds: Float64,
) raises:
    var remaining = run_cfg.steps - progress.epoch_step
    if remaining < 0:
        remaining = 0
    var eta_seconds = Float64(remaining) * Float64(speed_seconds)
    var line = (
        String("[Klein-lora] model FLUX_2 | type LoRA | dataset ")
        + run_cfg.dataset_path.copy()
        + String(" | step ")
        + String(progress.epoch_step)
        + String("/")
        + String(run_cfg.steps)
        + String(" | epoch 1/1 | loss ")
        + String(loss)
        + String(" | smooth_loss ")
        + String(smooth_loss)
        + String(" | grad_norm ")
        + String(grad_norm)
        + String(" | lr ")
        + String(cfg.learning_rate)
        + String(" | ")
        + String(speed_seconds)
        + String("s/step | elapsed ")
        + _format_hms(elapsed_seconds)
        + String(" | ETA ")
        + _format_hms(eta_seconds)
    )
    if path.byte_length() > 0:
        var f = open(path, "a")
        f.write(line)
        f.write("\n")
        f.close()
    print(line)


def _sec_from_ns(ns0: UInt, ns1: UInt) -> Float64:
    return Float64(ns1 - ns0) / Float64(1000000000.0)


def _format_hms(seconds_f: Float64) -> String:
    var total = Int(seconds_f)
    if total < 0:
        total = 0
    var hours = total // 3600
    var rem = total - hours * 3600
    var mins = rem // 60
    var secs = rem - mins * 60
    return String(hours) + String(":") + _two_digits(mins) + String(":") + _two_digits(secs)


def _two_digits(v: Int) -> String:
    if v < 10:
        return String("0") + String(v)
    return String(v)


def _final_lora_path(output_dir: String) -> String:
    return output_dir + String("/") + String(_STATELESS_LORA_FILE)


def _step_lora_path(output_dir: String, step: Int) -> String:
    return output_dir + String("/klein_lora_step_") + String(step) + String(".safetensors")
