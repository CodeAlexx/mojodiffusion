# Ideogram4LoRATrainer.mojo — staged full-block LoRA train driver.
#
# This is trainer-owned orchestration around the verified one-step math:
#   stage cache metadata -> load ONE transformer weight set -> stream one sample
#   -> ideogram4_lora_train_step -> save LoRA + Adam state.
#
# It intentionally does not accept List[Tensor] batches. Samples are materialised
# one at a time from Ideogram4TrainCache so the activation-heavy train step does
# not compete with a resident dataset tensor pile.

from max.gpu.host import DeviceContext
from std.memory import ArcPointer
from std.os import makedirs
from std.time import perf_counter

from serenitymojo.tensor import Tensor
from serenitymojo.training.levers import (
    caption_dropout_pick,
    levers_optimizer_active,
    levers_optimizer_validate,
    levers_optimizer_eval_for_save,
    levers_optimizer_train_after_save,
)
from serenitymojo.training.lora_ema import (
    LoraEmaState,
    lora_ema_track,
    ema_update,
    ema_shadow_a_bf16,
    ema_shadow_b_bf16,
    ema_path_for_lora,
)
from serenitymojo.training.train_config import TrainConfig as LeversConfig
from serenitymojo.training.train_config import (
    TRAIN_ADAPTER_ALGO_LOKR, TRAIN_ADAPTER_ALGO_LOHA,
)
from serenitymojo.training.adapter_algo_policy import (
    require_lora_or_locon_linear, adapter_algo_name,
)
from serenitymojo.training.lokr_stack import LOKR_CARRIER_MAX_DEVICE_BYTES
from serenitymojo.models.ideogram4.config import (
    IDEOGRAM4_ADALN_DIM, IDEOGRAM4_HIDDEN, IDEOGRAM4_INTERMEDIATE_SIZE,
    IDEOGRAM4_NUM_LAYERS,
)
from serenitymojo.models.ideogram4.ideogram4_lokr_stack import (
    Ideogram4LoKrSet, empty_ideogram4_lokr_set, build_ideogram4_lokr_set,
    ideogram4_lokr_carrier_total_bytes, ideogram4_lokr_carrier_device_set,
    ideogram4_lokr_chain_from_device, ideogram4_lokr_grad_norm,
    ideogram4_lokr_clip_grads, ideogram4_lokr_adamw_step,
    ideogram4_lokr_zero_leg_l1, save_ideogram4_lokr,
)
from serenitymojo.models.ideogram4.ideogram4_loha_stack import (
    Ideogram4LoHaSet, empty_ideogram4_loha_set, build_ideogram4_loha_set,
    ideogram4_loha_carrier_total_bytes, ideogram4_loha_carrier_device_set,
    ideogram4_loha_chain_from_device, ideogram4_loha_grad_norm,
    ideogram4_loha_clip_grads, ideogram4_loha_adamw_step,
    ideogram4_loha_zero_leg_l1, save_ideogram4_loha,
)
from serenitymojo.training.schedule import sample_timestep_logit_normal_scaled
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.tensor_algebra import zeros_device
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.models.dit.ideogram4_resident import Ideogram4Weights

from serenity_trainer.dataLoader.Ideogram4CacheReader import Ideogram4TrainCache
from serenity_trainer.model.Ideogram4LoRABlock import (
    Ideogram4LoraSet,
    Ideogram4StackLoraGrads,
    build_ideogram4_native_lora_set,
)
from serenity_trainer.modelLoader.Ideogram4LoRALoader import (
    load_ideogram4_block_stack_lora,
)
from serenity_trainer.modelSaver.Ideogram4LoRAModelSaver import (
    Ideogram4LoRAModelSaver,
)
from serenity_trainer.module.LoRAModule import LoraAdapter
from serenity_trainer.trainer.Ideogram4LoRATrainStep import (
    Ideogram4LoRATrainResult,
    ideogram4_lora_train_step_resident,
    ideogram4_lora_train_compute_resident,
    ideogram4_lora_train_compute_resident_handchain,
    ideogram4_lora_train_compute_resident_b2,
)
from serenity_trainer.trainer.Ideogram4StackTrain import (
    IDEOGRAM4_TELEMETRY_EVERY_STEPS,
    Ideogram4LoraAdamState,
    Ideogram4LeversBridge,
    apply_ideogram4_lora_grads,
    ideogram4_levers_mirrors_init,
    ideogram4_levers_refresh_mirrors,
    ideogram4_levers_optimizer_step,
    make_ideogram4_lora_adam_state,
)
from serenity_trainer.trainer.TrainState import TrainProgress
from serenity_trainer.trainer.cadence.SampleCadence import SampleCadence
from serenity_trainer.trainer.Ideogram4SampleResident import (
    ideogram4_sample_resident,
    ideogram4_decode_latent_to_png,
    ideogram4_encode_sample_prompt,
)
from serenity_trainer.util.enum.TimeUnit import TU_STEP
from serenity_trainer.util.config.TrainConfig import TrainConfig

from serenitymojo.ops.random import randn


comptime TArc = ArcPointer[Tensor]
comptime _I4_STATE_FILE = "ideogram4_train_state.safetensors"
comptime I4_SAMPLE_PROMPT_TOKENS = 1024


@fieldwise_init
struct Ideogram4LoRATrainRunConfig(Copyable, Movable):
    var transformer_path: String
    var cache_path: String
    var output_dir: String
    var resume_lora_path: String
    var resume_state_dir: String
    var steps: Int
    var default_t_flow: Float32
    var save_every_steps: Int
    var checkpoint_every_steps: Int
    var noise_seed: UInt64
    var lora_seed: UInt64
    var progress_file_path: String
    # T1.D caption dropout probability (default-off 0.0; serenity-trainer's
    # TrainConfig does not carry caption_dropout_prob, so the run config owns
    # it). When the pick fires, the step trains on cache.uncond[NT] (the
    # llm_uncond empty-caption features) instead of the sample's llm features.
    # 0.0 here falls back to levers.caption_dropout_prob (one knob wins).
    var caption_dropout_prob: Float32
    # T1 levers carrier (serenitymojo TrainConfig): loss_fn/huber_delta/
    # smooth_l1_beta/min_snr_gamma_flow (T1.A), ema_* (T1.B), optimizer/
    # optimizer_* (T1.C), caption_dropout_prob fallback (T1.D). Defaults are
    # ALL default-off (C13) == the pre-lever trainer byte-for-byte. The shared
    # recipe scalars (lr/beta1/beta2/eps/weight_decay/rank/alpha) are SYNCED
    # from the serenity-trainer TrainConfig at run start — that struct stays
    # the single source of truth for them.
    var levers: LeversConfig
    # ── sample-during-training (SampleCadence-wired) ──────────────────────────
    # sample_every_steps: TU_STEP cadence interval (0 = disabled, default-off so
    #   the pre-sampling trainer is byte-for-byte unchanged). On fire, for each
    #   prompt index the trainer denoises a sample from the CURRENT resident base
    #   + live LoRA and writes <output_dir>/samples/step_<N>_<promptidx>.png.
    # sample_steps / sample_cfg: the denoise loop length + CFG scale (inference
    #   defaults 8 / 7.0 — ideogram4_pipeline.mojo:35-36).
    # sample_seed: base RNG seed for the t=1 init noise (per-prompt offset added).
    # sample_resolution: generated image square size for inline sampling. This
    #   is independent of the train cache bucket; the train step can stay at the
    #   512px cached GH/GW while the sample instantiates a larger denoise grid.
    # sample_prompts: JSON prompt strings. They are encoded once before the
    #   resident transformer/optimizer load; the sample path then uses those
    #   features directly instead of cached dataset captions.
    var sample_every_steps: Int
    var sample_steps: Int
    var sample_cfg: Float32
    var sample_seed: UInt64
    var sample_resolution: Int
    var sample_prompts: List[String]

    @staticmethod
    def defaults(
        transformer_path: String,
        cache_path: String,
        output_dir: String,
    ) -> Ideogram4LoRATrainRunConfig:
        return Ideogram4LoRATrainRunConfig(
            transformer_path=transformer_path,
            cache_path=cache_path,
            output_dir=output_dir,
            resume_lora_path=String(""),
            resume_state_dir=String(""),
            steps=10,
            default_t_flow=Float32(0.7),
            save_every_steps=0,
            checkpoint_every_steps=0,
            noise_seed=UInt64(0x1D3A_4A11),
            lora_seed=UInt64(0x1D3A_4000),
            progress_file_path=String(""),
            caption_dropout_prob=Float32(0.0),
            levers=LeversConfig.default(),
            sample_every_steps=0,          # default-off: no sampling-in-training
            sample_steps=8,                # inference default (pipeline STEPS)
            sample_cfg=Float32(7.0),       # inference default (pipeline CFG)
            sample_seed=UInt64(0x1D3A_5A91),
            sample_resolution=512,
            sample_prompts=List[String](),
        )


@fieldwise_init
struct Ideogram4LoRATrainSummary(Copyable, Movable):
    var steps_ran: Int
    var cache_samples: Int
    var optimizer_steps: Int
    var last_loss: Float32
    var adapter_b_l1: Float32
    var elapsed_seconds: Float64
    var seconds_per_step: Float64
    var lora_path: String
    var state_dir: String
    var loaded_weight_sets: Int
    var progress: TrainProgress


@fieldwise_init
struct Ideogram4LoadedLoRAState(Copyable, Movable):
    var progress: TrainProgress
    var opt_step: Int


def train_ideogram4_lora_from_cache[NT: Int, GH: Int, GW: Int](
    cfg: TrainConfig,
    run_cfg: Ideogram4LoRATrainRunConfig,
    ctx: DeviceContext,
) raises -> Ideogram4LoRATrainSummary:
    if run_cfg.steps < 1:
        raise Error("train_ideogram4_lora_from_cache: steps must be >= 1")

    # ── T1 levers config (TIER1_PARITY_CAMPAIGN_2026-06-11.md) ────────────────
    # run_cfg.levers carries the LEVER keys; the shared optimizer/LoRA recipe
    # scalars are synced FROM the serenity-trainer cfg (the struct argv/gates
    # already control) so there is exactly one knob per scalar.
    var lcfg = run_cfg.levers.copy()
    lcfg.lr = cfg.learning_rate
    lcfg.beta1 = cfg.beta1
    lcfg.beta2 = cfg.beta2
    lcfg.eps = cfg.eps
    lcfg.weight_decay = cfg.weight_decay
    lcfg.lora_rank = cfg.lora_rank
    lcfg.lora_alpha = cfg.lora_alpha
    if lcfg.masked_training:
        raise Error(
            "train_ideogram4_lora_from_cache: masked_training is set but the"
            " ideogram4 stager emits no masks — masked loss (T1.E) is not"
            " wired for this trainer"
        )
    levers_optimizer_validate(lcfg, String("Ideogram4 trainer"))
    var levers_opt = levers_optimizer_active(lcfg)
    if levers_opt:
        print(
            "[Ideogram4-lora] T1.C levers optimizer active: tag=",
            lcfg.optimizer,
            " (2=ADAFACTOR, 7=SCHEDULE_FREE_ADAMW) warmup=",
            lcfg.optimizer_warmup_steps,
        )
    # effective caption-dropout p: run_cfg owns it (T1.D precedent); 0.0 falls
    # back to the levers key so a config-file value still reaches the pick.
    var drop_p = run_cfg.caption_dropout_prob
    if drop_p <= Float32(0.0):
        drop_p = lcfg.caption_dropout_prob

    # ── LyCORIS carrier dispatch (adapter_algo comes from the levers config JSON:
    # network_algorithm/algo/adapter_algo -> lcfg.adapter_algo, parsed by
    # serenitymojo read_model_config). LOKR/LOHA take the additive (a,b)-carrier
    # path; DORA/OFT/BOFT/FULL fail loud HERE (before the resident transformer
    # load) via adapter_algo_policy.require_lora_or_locon_linear. LORA/LOCON fall
    # through to the plain device-LoRA loop below. ──────────────────────────────
    if (
        lcfg.adapter_algo == TRAIN_ADAPTER_ALGO_LOKR
        or lcfg.adapter_algo == TRAIN_ADAPTER_ALGO_LOHA
    ):
        return _train_ideogram4_lycoris_from_cache[NT, GH, GW](cfg, run_cfg, lcfg, ctx)
    require_lora_or_locon_linear(lcfg, String("Ideogram4"))

    # ── gradient accumulation: DEVICE-GRAD ARM (fail-loud; not wired this wave) ──
    # The plain-LoRA path produces DEVICE grad tensors (grads.d_a/d_b). BOTH
    # optimizer routes consume them on-device: the default fused-multitensor AdamW
    # (IDEOGRAM4_FUSED_ADAMW=True) and the retained per-tensor fallback
    # (IDEOGRAM4_FUSED_ADAMW=False, ~408 device adamw_step launches) — the latter
    # still steps `grads.d_a[i][]`/`grads.d_b[i][]` DEVICE tensors, NOT a host
    # List[List[Float32]] grad loop. So NO host-grad arm survives to hang host
    # micro-step accumulation on (that would reintroduce the D2H removed in
    # MJ-1038, and flipping IDEOGRAM4_FUSED_ADAMW would NOT expose a host route).
    # Real grad-accum here needs DEVICE-side accumulation into the optimizer
    # (schedule.grad_accumulate), a separate follow-up. Fail loud rather than
    # silently ignore N>1 (mirrors the LyCORIS carrier fence).
    if cfg.gradient_accumulation_steps > 1:
        raise Error(
            "train_ideogram4_lora_from_cache: grad_accum_steps>1 is not wired for"
            " the plain-LoRA arm — both the fused and per-tensor AdamW routes are"
            " DEVICE-grad (no host-grad arm survives); device-side accumulation is"
            " a separate follow-up"
        )

    # ── TRUE batch-2 (row-stacked) fences ───────────────────────────────────────
    # batch_size==2 → the device-grad b2 path (ideogram4_lora_train_compute_resident_b2:
    # per-sample frozen embed/final, batched trainable stack, joint 2N-mean loss,
    # grads SUM both samples = batch gradient into the SAME device optimizer). Only
    # 1 or 2 are wired. accum>1 already fenced above (so accum+b2 is fenced together).
    if cfg.batch_size < 1 or cfg.batch_size > 2:
        raise Error(
            "train_ideogram4_lora_from_cache: only batch_size 1 or 2 is wired"
            " (b2 = TRUE row-stacked device-grad path); got "
            + String(cfg.batch_size)
        )
    if cfg.batch_size == 2 and levers_opt:
        raise Error(
            "train_ideogram4_lora_from_cache: batch_size=2 supports only the default"
            " device fused-AdamW optimizer this wave — the T1.C levers host optimizer"
            " (adafactor/schedule-free) is not wired for b2"
        )

    makedirs(run_cfg.output_dir, exist_ok=True)
    var final_lora_path = _final_lora_path(run_cfg.output_dir)
    var final_state_dir = _final_state_dir(run_cfg.output_dir)

    # Stage 1: cache metadata only. Samples are loaded inside the loop one at a
    # time; no dataset tensor list is kept resident.
    var cache = Ideogram4TrainCache.open(run_cfg.cache_path)

    # Inline sample prompt encoding happens before the resident transformer and
    # optimizer state are loaded, so Qwen3-VL is not resident during training.
    var sample_enabled = run_cfg.sample_every_steps > 0
    var samples_dir = run_cfg.output_dir + String("/samples")
    if sample_enabled:
        makedirs(samples_dir, exist_ok=True)
    var sample_prompt_list = run_cfg.sample_prompts.copy()
    var sample_prompt_llms = List[TArc]()
    var sample_prompt_text_lens = List[Int]()
    if sample_enabled:
        if len(sample_prompt_list) == 0:
            raise Error(
                "train_ideogram4_lora_from_cache: inline sampling requires at"
                " least one JSON sample prompt"
            )
        print(
            "[Ideogram4-lora] encoding inline sampler JSON prompts before"
            " transformer load count=",
            len(sample_prompt_list),
        )
        for pi in range(len(sample_prompt_list)):
            var encoded_llm = ideogram4_encode_sample_prompt[I4_SAMPLE_PROMPT_TOKENS](
                sample_prompt_list[pi], ctx, sample_prompt_text_lens
            )
            sample_prompt_llms.append(TArc(encoded_llm^))
    var cadence = SampleCadence(
        Float64(run_cfg.sample_every_steps),
        TU_STEP,
        Float64(0.0),                      # sample_skip_first: no delay
        sample_prompt_list^,
    )

    # Stage 2: exactly one resident FP8 transformer weight set for training.
    # This matches the fast Ideogram inference path: no per-step safetensors
    # reads, and no per-step H2D transfer of the frozen trunk.
    var weights = Ideogram4Weights.load(
        ShardedSafeTensors.open(run_cfg.transformer_path), ctx
    )

    # Stage 3: LoRA + optimizer state. Resume loads adapter weights first, then
    # rebases Adam moments/progress if a state dir was provided.
    var loras = _load_or_build_loras(
        run_cfg.resume_lora_path,
        cfg.lora_rank,
        cfg.lora_alpha,
        run_cfg.lora_seed,
        ctx,
    )
    var opt = make_ideogram4_lora_adam_state(loras, ctx)
    var progress = TrainProgress()
    var opt_step = 0
    if run_cfg.resume_state_dir.byte_length() > 0:
        var loaded = load_ideogram4_lora_train_state(
            run_cfg.resume_state_dir, opt, ctx
        )
        progress = loaded.progress.copy()
        opt_step = loaded.opt_step

    # T1.C levers bridge (host mirrors + optimizer state; nothing allocated on
    # the default AdamW path unless EMA needs the mirrors) + T1.B EMA shadows.
    # EMA tracks AFTER any resume load (shadow init = clone of current params,
    # SimpleTuner ema.py:123 semantics via training/lora_ema.mojo).
    var bridge = Ideogram4LeversBridge()
    var ema = LoraEmaState(
        lcfg.ema_decay, lcfg.ema_min_decay,
        lcfg.ema_update_after_step, lcfg.ema_update_step_interval,
    )
    if lcfg.ema_enabled or levers_opt:
        ideogram4_levers_mirrors_init(bridge, loras, ctx)
    if lcfg.ema_enabled:
        var ema_base = lora_ema_track(ema, bridge.mirrors, 0, len(bridge.mirrors))
        if ema_base != 0:
            raise Error("train_ideogram4_lora_from_cache: ema shadow base must be 0")
        print(
            "[Ideogram4-lora] T1.B EMA tracking", len(bridge.mirrors),
            "adapters decay=", lcfg.ema_decay,
            " min_decay=", lcfg.ema_min_decay,
            " update_after_step=", lcfg.ema_update_after_step,
            " interval=", lcfg.ema_update_step_interval,
        )

    var last_loss = Float32(0.0)
    var last_b = Float32(0.0)
    # Last-known telemetry L1s (MJ-1038): the full-tensor D2H readbacks behind
    # them are cadence-gated; the progress line holds these between refreshes.
    var last_grad_l1 = Float32(0.0)
    var train_start = perf_counter()
    var smooth_loss = Float32(0.0)
    var smooth_inited = False

    # ── sample-during-training cadence (default-off when sample_every_steps==0) ─
    # SampleCadence drives "should we sample now?" on the SAME TrainProgress that
    # threads through the loop. TU_STEP + start_at_zero=True: fires when
    # global_step % sample_every_steps == 0. We check it AFTER progress.next_step
    # so the just-completed optimizer step is reflected in the sampled LoRA state.

    for local_step in range(run_cfg.steps):
        var sample_index = progress.global_step % cache.len()
        var seed = run_cfg.noise_seed + UInt64(opt_step + local_step)
        # Per-step flow time ~ logit-normal(0, 1.0) = t = sigmoid(N(0,1)),
        # matching BOTH references: SerenityTrainer (ModelSetupNoiseMixin LOGIT_NORMAL,
        # scale = noising_weight+1.0 = 1.0 with the ideogram preset's defaults) and
        # the Rust train_ideogram (t = sigmoid(u)). Was 1.5 (DiffSynth INFERENCE
        # set_timesteps_ideogram4) — wrong vs both training oracles.
        # The old fixed default_t_flow=0.7 trained EVERY step at one timestep
        # (measured: loss collapsed to 1.3e-4, grad_norm 0 — learned nothing).
        # Separate RNG stream from the noise draw (the zimage *7919 idiom).
        var t_step = sample_timestep_logit_normal_scaled(
            run_cfg.noise_seed * UInt64(7919) + UInt64(opt_step + local_step),
            Float32(1.0),
        )
        var sample = cache.sample[NT, GH, GW](
            sample_index, t_step, seed, ctx
        )

        # T1.D caption dropout (default-off p<=0 never draws): shared levers
        # pick on the noise_seed stream; when it fires, train this step on the
        # cached empty-caption llm_uncond features (fail-loud if the cache
        # predates the --uncond stager).
        var llm_in = sample.llm_features.copy()
        # natural caption length for the DiT pad indicator; uncond substitutes
        # its own length when caption dropout fires.
        var text_len_in = sample.text_len
        if caption_dropout_pick(
            UInt64(opt_step + local_step),
            run_cfg.noise_seed,
            drop_p,
        ):
            llm_in = ArcPointer[Tensor](cache.uncond[NT](ctx))
            text_len_in = cache.uncond_text_len[NT](ctx)

        # forward + loss (T1.A lever seam inside) + backward — NO optimizer.
        var step_loss = Float32(0.0)
        var grads: Ideogram4StackLoraGrads
        if cfg.batch_size == 2:
            # ── TRUE batch-2: pair a second cache sample (independent t/seed/dropout
            # draw). Both frozen embeds carry their own text_len; the trainable stack
            # runs batched → summed grads = batch gradient into the device optimizer.
            var sample_index1 = (progress.global_step + 1) % cache.len()
            var seed1 = (
                run_cfg.noise_seed + UInt64(opt_step + local_step)
                + UInt64(0x5EED_00B2)
            )
            var t_step1 = sample_timestep_logit_normal_scaled(
                run_cfg.noise_seed * UInt64(7919)
                + UInt64(opt_step + local_step) + UInt64(0x00B2_7919),
                Float32(1.0),
            )
            var sample1 = cache.sample[NT, GH, GW](
                sample_index1, t_step1, seed1, ctx
            )
            var llm_in1 = sample1.llm_features.copy()
            var text_len_in1 = sample1.text_len
            if caption_dropout_pick(
                UInt64(opt_step + local_step) + UInt64(0x00B2),
                run_cfg.noise_seed,
                drop_p,
            ):
                llm_in1 = ArcPointer[Tensor](cache.uncond[NT](ctx))
                text_len_in1 = cache.uncond_text_len[NT](ctx)
            var loss0 = Float32(0.0)
            var loss1 = Float32(0.0)
            grads = ideogram4_lora_train_compute_resident_b2[NT, GH, GW](
                weights,
                sample.noisy[], sample.clean[], sample.noise[], sample.t_flow,
                llm_in[], text_len_in,
                sample1.noisy[], sample1.clean[], sample1.noise[], sample1.t_flow,
                llm_in1[], text_len_in1,
                loras,
                lcfg,
                step_loss,
                loss0,
                loss1,
                ctx,
            )
        else:
            grads = ideogram4_lora_train_compute_resident[NT, GH, GW](
                weights,
                sample.noisy[],
                sample.clean[],
                sample.noise[],
                sample.t_flow,
                llm_in[],
                loras,
                lcfg,
                step_loss,
                ctx,
                text_len_in,
            )
        # Real gradient L1 for the progress line. The apply/levers telemetry
        # returns were stubbed (apply_ideogram4_lora_grads returned grad_b_l1=0.0,
        # adapter_b_l1=0.0) and the progress line hardcoded "grad_norm 0.0000" —
        # so the trainer looked dead even though LoRA-B is learning. grads.d_b
        # holds the per-adapter LoRA-B gradients; to_host upcasts to Float32.
        # Computed BEFORE the optimizer consumes `grads` (grads^), once for both
        # the default-AdamW and levers paths.
        # MJ-1038: this was a FULL D2H of every B grad EVERY step (and apply
        # re-read B grads + B params again). Now the L1 refresh runs only every
        # IDEOGRAM4_TELEMETRY_EVERY_STEPS steps (plus first and final step so
        # the summary/save gates always see current values); the progress line
        # holds the last refresh in between.
        var telemetry_step = (
            opt_step % IDEOGRAM4_TELEMETRY_EVERY_STEPS == 0
            or local_step + 1 == run_cfg.steps
        )
        if telemetry_step:
            var l1 = Float32(0.0)
            for gi in range(len(grads.d_b)):
                var gh = grads.d_b[gi][].to_host(ctx)
                for gj in range(len(gh)):
                    var gv = gh[gj]
                    if gv < Float32(0.0):
                        l1 -= gv
                    else:
                        l1 += gv
            last_grad_l1 = l1
        var step_grad_l1 = last_grad_l1
        # T1.C optimizer seam: levers host path vs the existing literal fused
        # AdamW call (C13: optimizer=ADAMW routes around the levers entirely).
        var k = opt_step + 1
        var step_b_l1 = Float32(0.0)
        if levers_opt:
            step_b_l1 = ideogram4_levers_optimizer_step(
                lcfg, loras, bridge, grads, k,
                cfg.learning_rate, ctx,
            )
            _ = grads^
        else:
            var res = apply_ideogram4_lora_grads(
                loras, opt, grads^, k, cfg, ctx,
                want_l1_telemetry=telemetry_step,
            )
            if telemetry_step:
                step_b_l1 = res.adapter_b_l1
            else:
                step_b_l1 = last_b
        # T1.B EMA, post-optimizer: the default AdamW stepped the params on
        # device, so refresh the host mirrors first; the levers optimizer
        # keeps the mirrors authoritative already.
        if lcfg.ema_enabled:
            if not levers_opt:
                ideogram4_levers_refresh_mirrors(bridge, loras, ctx)
            ema_update(ema, bridge.mirrors, k)
        last_loss = step_loss
        last_b = step_b_l1
        opt_step += 1
        progress.next_step(cfg.batch_size)
        if not smooth_inited:
            smooth_loss = step_loss
            smooth_inited = True
        else:
            smooth_loss = smooth_loss * Float32(0.99) + step_loss * Float32(0.01)

        if run_cfg.progress_file_path.byte_length() > 0:
            var elapsed = perf_counter() - train_start
            var speed = elapsed / Float64(local_step + 1)
            _append_ideogram4_live_progress(
                run_cfg.progress_file_path,
                progress,
                run_cfg.steps,
                cfg,
                last_loss,
                smooth_loss,
                step_grad_l1,
                Float32(speed),
                elapsed,
            )

        if run_cfg.save_every_steps > 0 and opt_step % run_cfg.save_every_steps == 0:
            # schedule-free save bracket (no-op for every other optimizer —
            # levers.mojo SAVE CONTRACT) around the product save + EMA sibling.
            levers_optimizer_eval_for_save(lcfg, bridge.opt_st)
            var step_path = _step_lora_path(run_cfg.output_dir, opt_step)
            _save_lora(loras, step_path, ctx)
            if lcfg.ema_enabled:
                _save_lora_ema(ema, loras, step_path, ctx)
            levers_optimizer_train_after_save(lcfg, bridge.opt_st)

        if (
            run_cfg.checkpoint_every_steps > 0
            and opt_step % run_cfg.checkpoint_every_steps == 0
        ):
            levers_optimizer_eval_for_save(lcfg, bridge.opt_st)
            var ckpt_dir = _step_state_dir(run_cfg.output_dir, opt_step)
            save_ideogram4_lora_train_state(ckpt_dir, opt, progress, opt_step, ctx)
            levers_optimizer_train_after_save(lcfg, bridge.opt_st)

        # ── sample-during-training (fail-loud; default-off) ───────────────────
        # Checked AFTER next_step so global_step reflects the just-completed step.
        # cadence.should_sample mutates the cadence clock; only fires when
        # sample_every_steps>0 (sample_enabled) AND the TU_STEP interval is hit.
        if sample_enabled and cadence.should_sample(progress):
            for pi in range(cadence.num_prompts()):
                if run_cfg.sample_resolution == 2048:
                    _ideogram4_run_sample[NT, 128, 128](
                        weights, loras, run_cfg,
                        sample_prompt_llms, sample_prompt_text_lens, samples_dir,
                        progress.global_step, pi, ctx,
                    )
                elif run_cfg.sample_resolution == 1024:
                    _ideogram4_run_sample[NT, 64, 64](
                        weights, loras, run_cfg,
                        sample_prompt_llms, sample_prompt_text_lens, samples_dir,
                        progress.global_step, pi, ctx,
                    )
                elif run_cfg.sample_resolution == 512:
                    _ideogram4_run_sample[NT, 32, 32](
                        weights, loras, run_cfg,
                        sample_prompt_llms, sample_prompt_text_lens, samples_dir,
                        progress.global_step, pi, ctx,
                    )
                else:
                    raise Error(
                        "train_ideogram4_lora_from_cache: sample_resolution"
                        " must be 512, 1024, or 2048"
                    )

    var train_elapsed = perf_counter() - train_start
    var seconds_per_step = train_elapsed / Float64(run_cfg.steps)

    levers_optimizer_eval_for_save(lcfg, bridge.opt_st)
    _save_lora(loras, final_lora_path, ctx)
    if lcfg.ema_enabled:
        _save_lora_ema(ema, loras, final_lora_path, ctx)
    save_ideogram4_lora_train_state(final_state_dir, opt, progress, opt_step, ctx)
    levers_optimizer_train_after_save(lcfg, bridge.opt_st)

    return Ideogram4LoRATrainSummary(
        run_cfg.steps,
        cache.len(),
        opt_step,
        last_loss,
        last_b,
        train_elapsed,
        seconds_per_step,
        final_lora_path,
        final_state_dir,
        1,
        progress.copy(),
    )


# ──────────────────────────────────────────────────────────────────────────────
# LyCORIS (LoKr / LoHa) carrier training driver. ADDITIVE (a,b)-carrier families:
# each step synthesizes per-slot (a,b) matrices from the family masters into a
# DEVICE Ideogram4LoraSet, runs the EXISTING resident forward/backward
# (ideogram4_lora_train_compute_resident), pulls the device (a,b)-grads to host,
# chain-rules them to the master factors, and takes a HOST AdamW step on the
# masters (lokr_adamw/loha_adamw — the ideogram4 fused device AdamW drives only
# plain LoRA a/b tensors, not carrier masters, so the carrier path uses the same
# host optimizer arm zimage's carrier uses). Re-synthesized next step.
#
# Fail-loud (klein train_klein_real.mojo:1202-1318 posture) on combos this wave
# does not wire: grad-accum>1, EMA, levers optimizers, resume, inline sampling,
# masked training, init_lokr_norm (needs a base SafeTensors handle the resident
# fp8 trunk doesn't expose here).
def _train_ideogram4_lycoris_from_cache[NT: Int, GH: Int, GW: Int](
    cfg: TrainConfig,
    run_cfg: Ideogram4LoRATrainRunConfig,
    lcfg: LeversConfig,
    ctx: DeviceContext,
) raises -> Ideogram4LoRATrainSummary:
    var is_lokr = lcfg.adapter_algo == TRAIN_ADAPTER_ALGO_LOKR
    var algo_name = adapter_algo_name(lcfg.adapter_algo)
    var tag = String("[Ideogram4-") + algo_name + String("]")

    if run_cfg.steps < 1:
        raise Error(tag + " steps must be >= 1")
    if cfg.gradient_accumulation_steps > 1:
        raise Error(tag + " gradient accumulation is not wired for the carrier path")
    if lcfg.ema_enabled:
        raise Error(tag + " EMA shadows are not wired for the carrier path")
    if levers_optimizer_active(lcfg):
        raise Error(tag + " levers optimizers are not wired for the carrier path")
    if run_cfg.resume_lora_path.byte_length() > 0 or run_cfg.resume_state_dir.byte_length() > 0:
        raise Error(tag + " resume is not wired for the carrier path")
    if run_cfg.sample_every_steps > 0:
        raise Error(tag + " inline sampling is not wired for the carrier path")
    if lcfg.masked_training:
        raise Error(tag + " masked training is not wired for this trainer")

    var H = IDEOGRAM4_HIDDEN
    var F = IDEOGRAM4_INTERMEDIATE_SIZE
    var A = IDEOGRAM4_ADALN_DIM
    var NL = IDEOGRAM4_NUM_LAYERS
    var targets = 1 if lcfg.lokr_targets == 1 else 2
    var rank = cfg.lora_rank
    var alpha = cfg.lora_alpha

    makedirs(run_cfg.output_dir, exist_ok=True)
    var final_lora_path = _final_lycoris_path(run_cfg.output_dir)

    var cache = Ideogram4TrainCache.open(run_cfg.cache_path)

    # ── build masters + device-byte preflight (fail loud over the shared budget) ─
    var lokr_masters = empty_ideogram4_lokr_set()
    var loha_masters = empty_ideogram4_loha_set()
    var carrier_bytes = 0
    if is_lokr:
        lokr_masters = build_ideogram4_lokr_set(
            NL, H, F, A, rank, alpha, lcfg.lokr_factor,
            lcfg.lokr_decompose_both, lcfg.lokr_full_matrix, targets, run_cfg.lora_seed,
        )
        if lcfg.init_lokr_norm > 0.0:
            raise Error(
                tag + " init_lokr_norm perturbed init needs a base SafeTensors"
                + " handle; not wired for the resident fp8 trunk"
            )
        carrier_bytes = ideogram4_lokr_carrier_total_bytes(lokr_masters, H, F, A)
    else:
        loha_masters = build_ideogram4_loha_set(NL, H, F, A, rank, alpha, targets, run_cfg.lora_seed)
        carrier_bytes = ideogram4_loha_carrier_total_bytes(loha_masters, H, F, A)

    print(
        tag, " model IDEOGRAM_4 | targets ", targets, " | rank ", rank,
        " | carrier_device_bytes ", carrier_bytes,
        " | budget ", LOKR_CARRIER_MAX_DEVICE_BYTES,
    )
    if carrier_bytes > LOKR_CARRIER_MAX_DEVICE_BYTES:
        raise Error(
            tag + " carrier set needs " + String(carrier_bytes)
            + " device bytes (> budget " + String(LOKR_CARRIER_MAX_DEVICE_BYTES)
            + "); use a smaller lokr_factor or targets=attn"
        )

    # ── one resident fp8 transformer weight set (same as the plain LoRA path) ────
    var weights = Ideogram4Weights.load(
        ShardedSafeTensors.open(run_cfg.transformer_path), ctx
    )

    var drop_p = run_cfg.caption_dropout_prob
    if drop_p <= Float32(0.0):
        drop_p = lcfg.caption_dropout_prob

    var progress = TrainProgress()
    var last_loss = Float32(0.0)
    var last_master_norm = Float64(0.0)
    var train_start = perf_counter()
    var smooth_loss = Float32(0.0)
    var smooth_inited = False

    for local_step in range(run_cfg.steps):
        var sample_index = progress.global_step % cache.len()
        var seed = run_cfg.noise_seed + UInt64(local_step)
        var t_step = sample_timestep_logit_normal_scaled(
            run_cfg.noise_seed * UInt64(7919) + UInt64(local_step), Float32(1.0)
        )
        var sample = cache.sample[NT, GH, GW](sample_index, t_step, seed, ctx)

        var llm_in = sample.llm_features.copy()
        var text_len_in = sample.text_len
        if caption_dropout_pick(UInt64(local_step), run_cfg.noise_seed, drop_p):
            llm_in = ArcPointer[Tensor](cache.uncond[NT](ctx))
            text_len_in = cache.uncond_text_len[NT](ctx)

        var step_loss = Float32(0.0)
        var k = local_step + 1
        var master_norm = Float64(0.0)
        var zero_leg = Float64(0.0)

        if is_lokr:
            var device = ideogram4_lokr_carrier_device_set(lokr_masters, H, F, A, ctx)
            var grads = ideogram4_lora_train_compute_resident_handchain[NT, GH, GW](
                weights, sample.noisy[], sample.clean[], sample.noise[],
                sample.t_flow, llm_in[], device, lcfg, step_loss, ctx, text_len_in,
            )
            var mg = ideogram4_lokr_chain_from_device(lokr_masters, grads, ctx)
            master_norm = ideogram4_lokr_grad_norm(mg)
            if cfg.clip_grad_norm > Float32(0.0) and master_norm > Float64(cfg.clip_grad_norm):
                var cs = Float32(Float64(cfg.clip_grad_norm) / (master_norm + Float64(1.0e-6)))
                ideogram4_lokr_clip_grads(mg, cs)
            ideogram4_lokr_adamw_step(
                lokr_masters, mg, k, cfg.learning_rate,
                cfg.beta1, cfg.beta2, cfg.eps, cfg.weight_decay,
            )
            zero_leg = ideogram4_lokr_zero_leg_l1(lokr_masters)
            _ = device^
        else:
            var device = ideogram4_loha_carrier_device_set(loha_masters, H, F, A, ctx)
            var grads = ideogram4_lora_train_compute_resident_handchain[NT, GH, GW](
                weights, sample.noisy[], sample.clean[], sample.noise[],
                sample.t_flow, llm_in[], device, lcfg, step_loss, ctx, text_len_in,
            )
            var mg = ideogram4_loha_chain_from_device(loha_masters, grads, ctx)
            master_norm = ideogram4_loha_grad_norm(mg)
            if cfg.clip_grad_norm > Float32(0.0) and master_norm > Float64(cfg.clip_grad_norm):
                var cs = Float32(Float64(cfg.clip_grad_norm) / (master_norm + Float64(1.0e-6)))
                ideogram4_loha_clip_grads(mg, cs)
            ideogram4_loha_adamw_step(
                loha_masters, mg, k, cfg.learning_rate,
                cfg.beta1, cfg.beta2, cfg.eps, cfg.weight_decay,
            )
            zero_leg = ideogram4_loha_zero_leg_l1(loha_masters)
            _ = device^

        last_loss = step_loss
        last_master_norm = master_norm
        progress.next_step(cfg.batch_size)
        if not smooth_inited:
            smooth_loss = step_loss
            smooth_inited = True
        else:
            smooth_loss = smooth_loss * Float32(0.99) + step_loss * Float32(0.01)

        print(
            tag, " step ", k, "/", run_cfg.steps, " | loss ", step_loss,
            " | master_grad_norm ", Float32(master_norm),
            " | zero_leg_l1 ", zero_leg,
        )

        if run_cfg.progress_file_path.byte_length() > 0:
            var elapsed = perf_counter() - train_start
            var speed = elapsed / Float64(local_step + 1)
            _append_ideogram4_live_progress(
                run_cfg.progress_file_path, progress, run_cfg.steps, cfg,
                last_loss, smooth_loss, Float32(master_norm),
                Float32(speed), elapsed,
            )

        if run_cfg.save_every_steps > 0 and k % run_cfg.save_every_steps == 0:
            var step_path = _step_lycoris_path(run_cfg.output_dir, k)
            _ = _save_ideogram4_lycoris(lokr_masters, loha_masters, is_lokr, step_path, ctx)

    var train_elapsed = perf_counter() - train_start
    var seconds_per_step = train_elapsed / Float64(run_cfg.steps)
    var n_written = _save_ideogram4_lycoris(lokr_masters, loha_masters, is_lokr, final_lora_path, ctx)
    print(tag, " saved ", n_written, " adapters -> ", final_lora_path)

    return Ideogram4LoRATrainSummary(
        run_cfg.steps,
        cache.len(),
        run_cfg.steps,
        last_loss,
        Float32(last_master_norm),
        train_elapsed,
        seconds_per_step,
        final_lora_path,
        String(""),
        1,
        progress.copy(),
    )


def _final_lycoris_path(output_dir: String) -> String:
    return output_dir + String("/lycoris_last.safetensors")


def _step_lycoris_path(output_dir: String, step: Int) -> String:
    return output_dir + String("/lycoris_step_") + String(step) + String(".safetensors")


def _save_ideogram4_lycoris(
    lokr: Ideogram4LoKrSet, loha: Ideogram4LoHaSet, is_lokr: Bool,
    path: String, ctx: DeviceContext,
) raises -> Int:
    if is_lokr:
        return save_ideogram4_lokr(lokr, path, ctx)
    return save_ideogram4_loha(loha, path, ctx)


def save_ideogram4_lora_train_state(
    dir: String,
    opt: Ideogram4LoraAdamState,
    progress: TrainProgress,
    opt_step: Int,
    ctx: DeviceContext,
) raises:
    makedirs(dir, exist_ok=True)
    var names = List[String]()
    var tensors = List[TArc]()

    if (
        len(opt.m_a) != len(opt.v_a)
        or len(opt.m_a) != len(opt.m_b)
        or len(opt.m_a) != len(opt.v_b)
    ):
        raise Error("save_ideogram4_lora_train_state: Adam state list mismatch")

    for i in range(len(opt.m_a)):
        names.append(_state_key(i, String("m_a")))
        tensors.append(TArc(opt.m_a[i][].clone(ctx)))
        names.append(_state_key(i, String("v_a")))
        tensors.append(TArc(opt.v_a[i][].clone(ctx)))
        names.append(_state_key(i, String("m_b")))
        tensors.append(TArc(opt.m_b[i][].clone(ctx)))
        names.append(_state_key(i, String("v_b")))
        tensors.append(TArc(opt.v_b[i][].clone(ctx)))

    var meta_vals = List[Float32]()
    meta_vals.append(Float32(progress.epoch))
    meta_vals.append(Float32(progress.epoch_step))
    meta_vals.append(Float32(progress.epoch_sample))
    meta_vals.append(Float32(progress.global_step))
    meta_vals.append(Float32(opt_step))
    var meta = Tensor.from_host(meta_vals^, [5], STDtype.F32, ctx)
    names.append(String("train_progress"))
    tensors.append(TArc(meta^))

    save_safetensors(names^, tensors^, _state_file(dir), ctx)


def load_ideogram4_lora_train_state(
    dir: String,
    mut opt: Ideogram4LoraAdamState,
    ctx: DeviceContext,
) raises -> Ideogram4LoadedLoRAState:
    var src = ShardedSafeTensors.open(_state_file(dir))
    var meta = Tensor.from_view(src.tensor_view(String("train_progress")), ctx).to_host(ctx)
    if len(meta) != 5:
        raise Error("load_ideogram4_lora_train_state: malformed train_progress meta")

    var expected = len(opt.m_a)
    var n = 0
    while _has_state_slot(src, n):
        n += 1
    if n != expected:
        raise Error(
            String("load_ideogram4_lora_train_state: slot mismatch have ")
            + String(expected) + String(" checkpoint ") + String(n)
        )

    for i in range(expected):
        opt.m_a[i] = TArc(Tensor.from_view(src.tensor_view(_state_key(i, String("m_a"))), ctx))
        opt.v_a[i] = TArc(Tensor.from_view(src.tensor_view(_state_key(i, String("v_a"))), ctx))
        opt.m_b[i] = TArc(Tensor.from_view(src.tensor_view(_state_key(i, String("m_b"))), ctx))
        opt.v_b[i] = TArc(Tensor.from_view(src.tensor_view(_state_key(i, String("v_b"))), ctx))

    var progress = TrainProgress(
        Int(meta[0]), Int(meta[1]), Int(meta[2]), Int(meta[3])
    )
    return Ideogram4LoadedLoRAState(progress.copy(), Int(meta[4]))


# ──────────────────────────────────────────────────────────────────────────────
# _ideogram4_run_sample — one sample-during-training image.
#   cond conditioning : pre-encoded JSON sample prompt llm_features.
#   uncond conditioning: reuses cond features with text_len=0; the split embed
#                       creates zero text hidden rows without a 53248-wide tensor.
#   init noise        : randn [1,128,GH,GW] F32, seed = sample_seed + step*1000 +
#                       prompt_idx (deterministic per (step,prompt)).
#   denoise           : ideogram4_sample_resident (resident base + live LoRA).
#   decode + write    : ideogram4_decode_latent_to_png ->
#                       <samples_dir>/step_<N>_<promptidx>.png.
# Fail-loud: any raise propagates (no silent skip), per the build request.
def _ideogram4_build_sample_latent[NT: Int, GH: Int, GW: Int](
    weights: Ideogram4Weights,
    loras: Ideogram4LoraSet,
    run_cfg: Ideogram4LoRATrainRunConfig,
    sample_prompt_llms: List[TArc],
    sample_prompt_text_lens: List[Int],
    step: Int,
    prompt_idx: Int,
    prompt_feature_index: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    comptime PACKED_CH = 128

    var cond_llm = sample_prompt_llms[prompt_feature_index][].clone(ctx)
    var cond_shape = cond_llm.shape()
    if len(cond_shape) != 3 or cond_shape[1] != I4_SAMPLE_PROMPT_TOKENS:
        raise Error("_ideogram4_run_sample: encoded prompt has wrong shape")

    # t=1 init noise [1,128,GH,GW] F32, deterministic per (step, prompt).
    var seed = run_cfg.sample_seed + UInt64(step * 1000 + prompt_idx)
    print(
        "[Ideogram4-lora] sample build latent GH=", GH, " GW=", GW,
        " steps=", run_cfg.sample_steps, " cfg=", run_cfg.sample_cfg,
        " seed=", seed,
    )
    var init_noise = randn([1, PACKED_CH, GH, GW], seed, STDtype.F32, ctx)

    var latent = ideogram4_sample_resident[I4_SAMPLE_PROMPT_TOKENS, GH, GW](
        weights, loras, cond_llm, cond_llm, init_noise,
        run_cfg.sample_steps, run_cfg.sample_cfg, GH, GW, ctx,
        sample_prompt_text_lens[prompt_feature_index], 0,
    )
    return latent^


def _save_ideogram4_sample_latent(
    latent: Tensor,
    latent_path: String,
    ctx: DeviceContext,
) raises:
    var names = List[String]()
    names.append(String("latent"))
    var tensors = List[TArc]()
    tensors.append(TArc(latent.clone(ctx)))
    save_safetensors(names^, tensors^, latent_path, ctx)
    print("[Ideogram4-lora] sample latent -> ", latent_path)


def _ideogram4_run_sample[NT: Int, GH: Int, GW: Int](
    weights: Ideogram4Weights,
    loras: Ideogram4LoraSet,
    run_cfg: Ideogram4LoRATrainRunConfig,
    sample_prompt_llms: List[TArc],
    sample_prompt_text_lens: List[Int],
    samples_dir: String,
    step: Int,
    prompt_idx: Int,
    ctx: DeviceContext,
) raises:
    if len(sample_prompt_llms) == 0:
        raise Error("_ideogram4_run_sample: no encoded JSON sample prompts")
    if len(sample_prompt_llms) != len(sample_prompt_text_lens):
        raise Error("_ideogram4_run_sample: prompt feature/text_len list mismatch")

    var prompt_feature_index = prompt_idx % len(sample_prompt_llms)
    var latent = _ideogram4_build_sample_latent[NT, GH, GW](
        weights, loras, run_cfg, sample_prompt_llms, sample_prompt_text_lens,
        step, prompt_idx, prompt_feature_index, ctx,
    )

    var out_path = (
        samples_dir + String("/step_") + String(step)
        + String("_") + String(prompt_idx) + String(".png")
    )
    var latent_path = (
        samples_dir + String("/step_") + String(step)
        + String("_") + String(prompt_idx) + String("_latent.safetensors")
    )
    _save_ideogram4_sample_latent(latent, latent_path, ctx)
    ideogram4_decode_latent_to_png[GH, GW](latent, out_path, ctx)
    print(
        "[Ideogram4-lora] sample step=", step, " prompt=", prompt_idx,
        " prompt_json_idx=", prompt_feature_index,
        " tokens=", sample_prompt_text_lens[prompt_feature_index],
        " -> ", out_path,
    )


def _load_or_build_loras(
    resume_lora_path: String,
    rank: Int,
    alpha: Float32,
    seed: UInt64,
    ctx: DeviceContext,
) raises -> Ideogram4LoraSet:
    if resume_lora_path.byte_length() > 0:
        return load_ideogram4_block_stack_lora(resume_lora_path, ctx)
    return build_ideogram4_native_lora_set(rank, alpha, ctx, seed=seed)


def _save_lora(loras: Ideogram4LoraSet, path: String, ctx: DeviceContext) raises:
    var saver = Ideogram4LoRAModelSaver()
    saver.save_block_stack_lora(loras, path, ctx)


# T1.B: save the EMA shadow set as the *_ema.safetensors sibling of a
# just-saved LoRA, through the SAME product writer (zimage's
# _save_zimage_lora_ema precedent). Shadows are flat-indexed 1:1 with
# loras.ad (lora_ema_track tracked the full mirror set, base 0); the bf16
# export round is lora_ema.mojo's copy_to cast (ema.py:454).
def _save_lora_ema(
    ema: LoraEmaState, loras: Ideogram4LoraSet, lora_path: String,
    ctx: DeviceContext,
) raises:
    var ad = List[ArcPointer[LoraAdapter]]()
    for i in range(len(loras.ad)):
        var a_t = Tensor.from_host_bf16(
            ema_shadow_a_bf16(ema, i), loras.ad[i][].a.shape(), ctx
        )
        var b_t = Tensor.from_host_bf16(
            ema_shadow_b_bf16(ema, i), loras.ad[i][].b.shape(), ctx
        )
        ad.append(
            ArcPointer[LoraAdapter](
                LoraAdapter(
                    a_t^, b_t^, loras.ad[i][].rank, loras.ad[i][].alpha
                )
            )
        )
    var ema_set = Ideogram4LoraSet(ad^, loras.n_layers, loras.rank)
    var ema_path = ema_path_for_lora(lora_path)
    _save_lora(ema_set, ema_path, ctx)
    print("[Ideogram4-lora] save_ema path=", ema_path)


def _append_ideogram4_live_progress(
    path: String,
    progress: TrainProgress,
    max_steps: Int,
    cfg: TrainConfig,
    loss: Float32,
    smooth_loss: Float32,
    grad_norm: Float32,
    speed_seconds: Float32,
    elapsed_seconds: Float64,
) raises:
    var eta_seconds = Float64(max_steps - progress.epoch_step) * Float64(speed_seconds)
    if eta_seconds < 0.0:
        eta_seconds = 0.0
    var line = (
        String("[Ideogram4-lora] model IDEOGRAM_4 | type LoRA | step ")
        + String(progress.epoch_step)
        + String("/")
        + String(max_steps)
        + String(" | epoch ")
        + String(progress.epoch + 1)
        + String("/1 | loss ")
        + String(loss)
        + String(" | smooth_loss ")
        + String(smooth_loss)
        + String(" | grad_norm ")
        + String(grad_norm)
        + String(" | ")
        + String(speed_seconds)
        + String("s/step | elapsed ")
        + _format_hms(elapsed_seconds)
        + String(" | ETA ")
        + _format_hms(eta_seconds)
    )
    var f = open(path, "a")
    f.write(line)
    f.write("\n")
    f.close()
    print(line)


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


def _state_key(i: Int, part: String) -> String:
    return String("adapter.") + String(i) + String(".") + part


def _has_state_slot(src: ShardedSafeTensors, i: Int) -> Bool:
    return _state_key(i, String("m_a")) in src.name_to_shard


def _state_file(dir: String) -> String:
    return dir + String("/") + String(_I4_STATE_FILE)


def _final_lora_path(output_dir: String) -> String:
    return output_dir + String("/lora_last.safetensors")


def _final_state_dir(output_dir: String) -> String:
    return output_dir + String("/state_last")


def _step_lora_path(output_dir: String, step: Int) -> String:
    return output_dir + String("/lora_step_") + String(step) + String(".safetensors")


def _step_state_dir(output_dir: String, step: Int) -> String:
    return output_dir + String("/state_step_") + String(step)
