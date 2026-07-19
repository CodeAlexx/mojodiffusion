"""Optimizer parameters surface for Serenity trainer."""

from mojoui.core.context import Context
from mojoui.widgets.form import begin_form_panel, end_form_panel, field_row, toggle_row
from serenity_trainer.ui.AuxScreenCommon import aux_label_w, aux_panel_h
from serenity_trainer.ui.TrainerConfigModel import TrainerUIConfig
from serenity_trainer.ui.UITabCommon import row2, two_col_w, value_w


def render_optimizer_params_window(mut ctx: Context, mut cfg: TrainerUIConfig, content_w: Int32) raises:
    var cw = two_col_w(content_w)
    var label_w = aux_label_w(ctx, cw)
    var val_w = value_w(cw - ctx.theme.padding * 2, label_w)
    ctx.layout_row(row2(cw, cw), aux_panel_h(ctx, 7))
    begin_form_panel(ctx, String("OPTIMIZER SETTINGS"), String("Dynamic parameters used by the selected optimizer"))
    var fused = False
    field_row(ctx, label_w, val_w, String("Optimizer"), cfg.optimizer_label())
    field_row(ctx, label_w, val_w, String("Learning Rate"), String(cfg.learning_rate))
    field_row(ctx, label_w, val_w, String("Weight Decay"), String(cfg.weight_decay))
    field_row(ctx, label_w, val_w, String("Beta1"), String("0.9"))
    field_row(ctx, label_w, val_w, String("Beta2"), String("0.999"))
    field_row(ctx, label_w, val_w, String("EPS"), String("1e-8"))
    _ = toggle_row(ctx, label_w, val_w, String("Fused"), String("Use fused optimizer path"), fused)
    end_form_panel(ctx)

    begin_form_panel(ctx, String("ADVANCED KEYS"), String("Known Serenity optimizer extension keys"))
    field_row(ctx, label_w, val_w, String("Stability"), String("amsgrad, capturable, foreach, differentiable"))
    field_row(ctx, label_w, val_w, String("Paged/8bit"), String("is_paged, optim_bits, min_8bit_size"))
    field_row(ctx, label_w, val_w, String("D Adapt"), String("d0, d_coef, growth_rate, safeguard_warmup"))
    field_row(ctx, label_w, val_w, String("Prodigy"), String("beta3, use_bias_correction, slice_p"))
    field_row(ctx, label_w, val_w, String("CAME"), String("clip_threshold, scale_parameter, relative_step"))
    field_row(ctx, label_w, val_w, String("Muon"), String("MuonWithAuxAdam, ns_steps, rms_rescaling"))
    field_row(ctx, label_w, val_w, String("Schedulefree"), String("use_schedulefree, schedulefree_c"))
    end_form_panel(ctx)
