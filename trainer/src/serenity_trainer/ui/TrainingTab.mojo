"""Training tab for the native Serenity Mojo trainer UI."""

from mojoui.core.context import Context
from mojoui.widgets.form import (
    begin_form_panel,
    end_form_panel,
    drag_row,
    field_row,
    select_index_row,
    select_string_row,
    slider_row,
    toggle_row,
)
from serenity_trainer.ui.TrainerConfigModel import TrainerUIConfig
from serenity_trainer.ui.UITabCommon import row2, two_col_w, value_w


def _panel_h(ctx: Context, rows: Int32) -> Int32:
    var pad = ctx.theme.padding
    var header_h = pad * 3
    var text_header_h = ctx.theme.font_size_pt * 3
    if text_header_h > header_h:
        header_h = text_header_h
    var gaps = rows - 1
    if gaps < 0:
        gaps = 0
    return header_h + pad * 2 + ctx.theme.row_height * rows + ctx.theme.spacing * gaps


def _label_w(ctx: Context, panel_w: Int32) -> Int32:
    var inner_w = panel_w - ctx.theme.padding * 2
    var w = ctx.theme.font_size_pt * 8
    if w < 178:
        w = 178
    var max_w = inner_w - 196
    if max_w < 132:
        max_w = 132
    if w > max_w:
        w = max_w
    return w


def _compact_w(ctx: Context, value_width: Int32) -> Int32:
    var w = ctx.theme.font_size_pt * 5
    if w < 150:
        w = 150
    if w > value_width:
        w = value_width
    return w


def render_training_tab(mut ctx: Context, mut cfg: TrainerUIConfig, content_w: Int32) raises:
    var cw = two_col_w(content_w)
    var label_w = _label_w(ctx, cw)
    var val_w = value_w(cw - ctx.theme.padding * 2, label_w)
    var compact_w = _compact_w(ctx, val_w)
    ctx.layout_row(row2(cw, cw), _panel_h(ctx, 9))
    begin_form_panel(ctx, String("BASE SCHEDULE"), String("Serenity epoch, batch, and LR cycle fields"), ctx.theme.padding)
    _ = slider_row(ctx, label_w, val_w, String("Epochs"), String("epochs"), cfg.epochs, 1.0, 50.0)
    _ = slider_row(ctx, label_w, val_w, String("Local Batch"), String("batch_size"), cfg.batch_size, 1.0, 16.0)
    _ = slider_row(ctx, label_w, val_w, String("Accum Steps"), String("gradient_accumulation_steps"), cfg.gradient_accumulation_steps, 1.0, 16.0)
    _ = drag_row(ctx, label_w, compact_w, String("Warmup"), String("warmup_steps"), cfg.learning_rate_warmup_steps, 10.0)
    if cfg.learning_rate_warmup_steps < 0.0:
        cfg.learning_rate_warmup_steps = 0.0
    _ = drag_row(ctx, label_w, compact_w, String("LR Min Factor"), String("learning_rate_min_factor"), cfg.learning_rate_min_factor, 0.01)
    _ = drag_row(ctx, label_w, compact_w, String("LR Cycles"), String("learning_rate_cycles"), cfg.learning_rate_cycles, 0.1)
    _ = select_string_row(ctx, label_w, val_w, String("LR Scaler"), String("learning_rate_scaler"), cfg.lr_scaler_options, cfg.learning_rate_scaler, cfg.select_open_id)
    _ = drag_row(ctx, label_w, compact_w, String("Seed"), String("seed"), cfg.seed, 1.0)
    if cfg.seed < 0.0:
        cfg.seed = 0.0
    # max_train_steps was a display-only row ("max train steps" literal) —
    # now the real editable field. Every runner takes steps from this value
    # (config runners argv 2, hidream argv 2, ideogram4 argv 5) plus the
    # max_steps key in the runner config JSON.
    _ = drag_row(ctx, label_w, compact_w, String("Max Steps"), String("max_train_steps"), cfg.max_train_steps, 50.0)
    if cfg.max_train_steps < 1.0:
        cfg.max_train_steps = 1.0
    end_form_panel(ctx)

    begin_form_panel(ctx, String("OPTIMIZER"), String("Optimizer, scheduler, learning rates"), ctx.theme.padding)
    _ = select_index_row(ctx, label_w, val_w, String("Optimizer"), String("optimizer"), cfg.optimizer_options, cfg.optimizer_index, cfg.select_open_id)
    _ = select_index_row(ctx, label_w, val_w, String("Scheduler"), String("scheduler"), cfg.scheduler_options, cfg.scheduler_index, cfg.select_open_id)
    _ = drag_row(ctx, label_w, compact_w, String("Learning Rate"), String("learning_rate"), cfg.learning_rate, 0.00001)
    # drag_value has no clamping — a scrub past zero must not emit lr <= 0.
    if cfg.learning_rate < 0.000001:
        cfg.learning_rate = 0.000001
    _ = drag_row(ctx, label_w, compact_w, String("Text Enc LR"), String("text_encoder_learning_rate"), cfg.text_encoder_learning_rate, 0.00001)
    if cfg.text_encoder_learning_rate < 0.0:
        cfg.text_encoder_learning_rate = 0.0
    _ = drag_row(ctx, label_w, compact_w, String("Transformer LR"), String("transformer_learning_rate"), cfg.transformer_learning_rate, 0.00001)
    if cfg.transformer_learning_rate < 0.0:
        cfg.transformer_learning_rate = 0.0
    _ = drag_row(ctx, label_w, compact_w, String("Weight Decay"), String("weight_decay"), cfg.weight_decay, 0.001)
    _ = drag_row(ctx, label_w, compact_w, String("Clip Grad"), String("clip_grad_norm"), cfg.clip_grad_norm, 0.1)
    end_form_panel(ctx)

    ctx.layout_row(row2(cw, cw), _panel_h(ctx, 10))
    begin_form_panel(ctx, String("PRECISION & MEMORY"), String("Precision and VRAM switches"), ctx.theme.padding)
    _ = select_string_row(ctx, label_w, val_w, String("Train DType"), String("train_precision"), cfg.precision_options, cfg.train_dtype, cfg.select_open_id)
    _ = select_string_row(ctx, label_w, val_w, String("Fallback DType"), String("fallback_train_dtype"), cfg.precision_options, cfg.fallback_train_dtype, cfg.select_open_id)
    _ = toggle_row(ctx, label_w, val_w, String("Gradient CKPT"), String("Enabled"), cfg.gradient_checkpointing)
    _ = toggle_row(ctx, label_w, val_w, String("Act Offload"), String("Enabled"), cfg.activation_offloading)
    _ = slider_row(ctx, label_w, val_w, String("Offload Fraction"), String("offload_fraction_train"), cfg.layer_offload_fraction, 0.0, 1.0)
    _ = toggle_row(ctx, label_w, val_w, String("Autocast Cache"), String("Enabled"), cfg.enable_autocast_cache)
    _ = select_string_row(ctx, label_w, val_w, String("Resolution"), String("train_resolution"), cfg.resolution_options, cfg.resolution, cfg.select_open_id)
    field_row(ctx, label_w, val_w, String("Frames"), cfg.frames.copy())
    _ = toggle_row(ctx, label_w, val_w, String("Circular Pad"), String("Force"), cfg.force_circular_padding)
    _ = select_string_row(ctx, label_w, val_w, String("Train Device"), String("train_device_training"), cfg.device_options, cfg.train_device, cfg.select_open_id)
    end_form_panel(ctx)

    begin_form_panel(ctx, String("EMA & TARGETS"), String("EMA and trainable model sections"), ctx.theme.padding)
    _ = select_string_row(ctx, label_w, val_w, String("EMA"), String("ema_mode"), cfg.ema_options, cfg.ema_mode, cfg.select_open_id)
    _ = drag_row(ctx, label_w, compact_w, String("EMA Decay"), String("ema_decay"), cfg.ema_decay, 0.001)
    _ = drag_row(ctx, label_w, compact_w, String("EMA Update"), String("ema_update_step_interval"), cfg.ema_update_step_interval, 1.0)
    _ = toggle_row(ctx, label_w, val_w, String("Transformer"), String("Train"), cfg.train_transformer)
    _ = toggle_row(ctx, label_w, val_w, String("Text Encoder"), String("Train"), cfg.train_text_encoder)
    _ = drag_row(ctx, label_w, compact_w, String("TE Stop After"), String("text_encoder_stop_after"), cfg.text_encoder_stop_after, 1.0)
    _ = drag_row(ctx, label_w, compact_w, String("Tr Stop After"), String("transformer_stop_after"), cfg.transformer_stop_after, 1.0)
    field_row(ctx, label_w, val_w, String("Backend"), String("Ideogram4 LoRA"))
    end_form_panel(ctx)

    ctx.layout_row(row2(cw, cw), _panel_h(ctx, 6))
    begin_form_panel(ctx, String("TEXT ENCODER"), String("Serenity text encoder controls"), ctx.theme.padding)
    _ = toggle_row(ctx, label_w, val_w, String("Train"), String("Enabled"), cfg.train_text_encoder)
    _ = slider_row(ctx, label_w, val_w, String("Caption Dropout"), String("caption_dropout_train"), cfg.caption_dropout, 0.0, 0.5)
    _ = drag_row(ctx, label_w, compact_w, String("Stop After"), String("text_encoder_stop_after_2"), cfg.text_encoder_stop_after, 1.0)
    _ = drag_row(ctx, label_w, compact_w, String("Learning Rate"), String("text_encoder_learning_rate_2"), cfg.text_encoder_learning_rate, 0.00001)
    field_row(ctx, label_w, val_w, String("Sequence Len"), cfg.text_encoder_sequence_length.copy())
    field_row(ctx, label_w, val_w, String("Clip Skip"), String("not used by Flux2"))
    end_form_panel(ctx)

    begin_form_panel(ctx, String("TRANSFORMER"), String("Serenity transformer controls"), ctx.theme.padding)
    _ = toggle_row(ctx, label_w, val_w, String("Train"), String("Enabled"), cfg.train_transformer)
    _ = drag_row(ctx, label_w, compact_w, String("Stop After"), String("transformer_stop_after_2"), cfg.transformer_stop_after, 1.0)
    _ = drag_row(ctx, label_w, compact_w, String("Learning Rate"), String("transformer_learning_rate_2"), cfg.transformer_learning_rate, 0.00001)
    _ = toggle_row(ctx, label_w, val_w, String("Attention Mask"), String("Force"), cfg.transformer_attention_mask)
    _ = drag_row(ctx, label_w, compact_w, String("Guidance"), String("transformer_guidance_scale"), cfg.transformer_guidance_scale, 0.1)
    field_row(ctx, label_w, val_w, String("Target"), String("Flux2 transformer"))
    end_form_panel(ctx)

    ctx.layout_row(row2(cw, cw), _panel_h(ctx, 10))
    begin_form_panel(ctx, String("NOISE & TIMESTEPS"), String("Flow matching noising and timestep controls"), ctx.theme.padding)
    _ = slider_row(ctx, label_w, val_w, String("Offset Noise"), String("offset_noise_weight"), cfg.offset_noise_weight, 0.0, 1.0)
    _ = slider_row(ctx, label_w, val_w, String("Perturb Noise"), String("perturbation_noise_weight"), cfg.perturbation_noise_weight, 0.0, 1.0)
    _ = select_string_row(ctx, label_w, val_w, String("Distribution"), String("timestep_distribution"), cfg.timestep_distribution_options, cfg.timestep_distribution, cfg.select_open_id)
    _ = slider_row(ctx, label_w, val_w, String("Min Strength"), String("min_noising_strength"), cfg.min_noising_strength, 0.0, 1.0)
    _ = slider_row(ctx, label_w, val_w, String("Max Strength"), String("max_noising_strength"), cfg.max_noising_strength, 0.0, 1.0)
    _ = drag_row(ctx, label_w, compact_w, String("Noising Weight"), String("noising_weight"), cfg.noising_weight, 0.1)
    _ = drag_row(ctx, label_w, compact_w, String("Noising Bias"), String("noising_bias"), cfg.noising_bias, 0.1)
    _ = drag_row(ctx, label_w, compact_w, String("Time Shift"), String("timestep_shift"), cfg.timestep_shift, 0.1)
    _ = toggle_row(ctx, label_w, val_w, String("Dynamic Shift"), String("Enabled"), cfg.dynamic_timestep_shifting)
    field_row(ctx, label_w, val_w, String("Prediction"), String("flow matching"))
    end_form_panel(ctx)

    # Capability honesty (UI wave 2): NO runner consumes masked-training —
    # the widgets are snapshot-only and excluded from every runner emission.
    # The [not wired] suffix + capability-table warning keep them from lying.
    begin_form_panel(ctx, String("MASKED TRAINING [not wired]"), String("No runner consumes these yet - excluded from launch config"), ctx.theme.padding)
    _ = toggle_row(ctx, label_w, val_w, String("Masked"), String("Enabled"), cfg.masked_training)
    _ = slider_row(ctx, label_w, val_w, String("Unmasked Prob"), String("unmasked_probability"), cfg.unmasked_probability, 0.0, 1.0)
    _ = slider_row(ctx, label_w, val_w, String("Unmasked Weight"), String("unmasked_weight"), cfg.unmasked_weight, 0.0, 2.0)
    _ = toggle_row(ctx, label_w, val_w, String("Normalize Loss"), String("Enabled"), cfg.normalize_masked_area_loss)
    _ = slider_row(ctx, label_w, val_w, String("Prior Weight"), String("masked_prior_preservation_weight"), cfg.masked_prior_preservation_weight, 0.0, 2.0)
    _ = toggle_row(ctx, label_w, val_w, String("Custom Cond"), String("Image"), cfg.custom_conditioning_image)
    end_form_panel(ctx)

    ctx.layout_row(row2(cw, cw), _panel_h(ctx, 13))
    begin_form_panel(ctx, String("LOSS"), String("Serenity loss mix and scaling"), ctx.theme.padding)
    # T1.A loss-fn selector (levers_loss_grad trainers, zimage first). SEPARATE
    # from the mse/mae/huber strength combined-loss rows below — do not conflate.
    var loss_fn_options = List[String]()
    loss_fn_options.append(String("mse"))
    loss_fn_options.append(String("huber"))
    loss_fn_options.append(String("smooth_l1"))
    _ = select_string_row(ctx, label_w, val_w, String("Loss Fn"), String("loss_fn"), loss_fn_options, cfg.loss_fn, cfg.select_open_id)
    _ = drag_row(ctx, label_w, compact_w, String("MSE"), String("mse_strength"), cfg.mse_strength, 0.1)
    _ = drag_row(ctx, label_w, compact_w, String("MAE"), String("mae_strength"), cfg.mae_strength, 0.1)
    _ = drag_row(ctx, label_w, compact_w, String("log-cosh"), String("log_cosh_strength"), cfg.log_cosh_strength, 0.1)
    _ = drag_row(ctx, label_w, compact_w, String("Huber"), String("huber_strength"), cfg.huber_strength, 0.1)
    _ = drag_row(ctx, label_w, compact_w, String("Huber Delta"), String("huber_delta"), cfg.huber_delta, 0.1)
    _ = drag_row(ctx, label_w, compact_w, String("SmoothL1 Beta"), String("smooth_l1_beta"), cfg.smooth_l1_beta, 0.1)
    _ = drag_row(ctx, label_w, compact_w, String("VB"), String("vb_loss_strength"), cfg.vb_loss_strength, 0.1)
    _ = select_string_row(ctx, label_w, val_w, String("Weight Fn"), String("loss_weight_fn"), cfg.loss_weight_options, cfg.loss_weight_fn, cfg.select_open_id)
    _ = drag_row(ctx, label_w, compact_w, String("Gamma"), String("loss_weight_strength"), cfg.loss_weight_strength, 0.1)
    # Min-SNR gamma (flow): SimpleTuner ε-style min(SNR,γ)/SNR weight, 0 = off.
    # NOTE divisor: this is /SNR, NOT the klein loss_weight_fn=MIN_SNR_GAMMA
    # lever above which divides by (SNR+1).
    _ = drag_row(ctx, label_w, compact_w, String("Min-SNR Flow"), String("min_snr_gamma_flow"), cfg.min_snr_gamma_flow, 0.1)
    _ = select_string_row(ctx, label_w, val_w, String("Loss Scaler"), String("loss_scaler"), cfg.loss_scaler_options, cfg.loss_scaler, cfg.select_open_id)
    field_row(ctx, label_w, val_w, String("Backend"), String("Serenity loss mix"))
    end_form_panel(ctx)

    begin_form_panel(ctx, String("LAYER FILTER"), String("Target layer preset and explicit filter"), ctx.theme.padding)
    _ = select_string_row(ctx, label_w, val_w, String("Preset"), String("layer_filter_preset"), cfg.layer_filter_preset_options, cfg.layer_filter_preset, cfg.select_open_id)
    field_row(ctx, label_w, val_w, String("Filter"), cfg.layer_filter.copy())
    _ = toggle_row(ctx, label_w, val_w, String("Regex"), String("Enabled"), cfg.layer_filter_regex)
    # This selector is emitted as network_algorithm/adapter_algo in runner JSON.
    _ = select_string_row(ctx, label_w, val_w, String("Algorithm"), String("peft_type_training"), cfg.peft_options, cfg.peft_type, cfg.select_open_id)
    field_row(ctx, label_w, val_w, String("Target"), String("transformer blocks"))
    end_form_panel(ctx)
