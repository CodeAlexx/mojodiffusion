"""Compile/render smoke for auxiliary Serenity UI surfaces."""

from mojoui.core.context import Context
from mojoui.core.textedit import TextEditState
from mojoui.core.types import Vec2
from serenity_trainer.ui.AdditionalEmbeddingsTab import render_additional_embeddings_tab
from serenity_trainer.ui.CaptionerTab import CaptionerScreenState, render_captioner_tab
from serenity_trainer.ui.CaptionUI import render_caption_ui
from serenity_trainer.ui.ConceptWindow import render_concept_window
from serenity_trainer.ui.ConfigList import render_config_list
from serenity_trainer.ui.ConvertModelUI import render_convert_model_ui
from serenity_trainer.ui.GenerateCaptionsWindow import render_generate_captions_window
from serenity_trainer.ui.GenerateMasksWindow import render_generate_masks_window
from serenity_trainer.ui.MuonAdamWindow import render_muon_adam_window
from serenity_trainer.ui.OffloadingWindow import render_offloading_window
from serenity_trainer.ui.OptimizerParamsWindow import render_optimizer_params_window
from serenity_trainer.ui.ProfilingWindow import render_profiling_window
from serenity_trainer.ui.SampleFrame import render_sample_frame
from serenity_trainer.ui.SampleParamsWindow import render_sample_params_window
from serenity_trainer.ui.SampleWindow import render_sample_window
from serenity_trainer.ui.SchedulerParamsWindow import render_scheduler_params_window
from serenity_trainer.ui.TimestepDistributionWindow import render_timestep_distribution_window
from serenity_trainer.ui.TrainerConfigModel import TrainerUIConfig
from serenity_trainer.ui.TrainerRuntimeBridge import TrainerUIRuntime
from serenity_trainer.ui.VideoToolUI import render_video_tool_ui


def main() raises:
    var ctx = Context()
    var cfg = TrainerUIConfig()
    var rt = TrainerUIRuntime()
    var captioner = CaptionerScreenState()
    var captioner_folder_edit = TextEditState(single_line=True)
    var captioner_custom_model_edit = TextEditState(single_line=True)
    var captioner_prompt_edit = TextEditState(single_line=True)
    var rows = List[String]()
    rows.append(String("key=value"))
    rows.append(String("another=value"))
    ctx.begin_frame_no_input(Vec2(1800.0, 24000.0), Vec2(-100.0, -100.0), False, False)
    render_additional_embeddings_tab(ctx, cfg, 1200)
    render_captioner_tab(
        ctx,
        cfg,
        captioner,
        1200,
        captioner_folder_edit,
        captioner_custom_model_edit,
        captioner_prompt_edit,
    )
    render_caption_ui(ctx, cfg, 1200)
    render_concept_window(ctx, cfg, 1200)
    render_config_list(ctx, String("CONFIG LIST"), 1200, rows)
    render_convert_model_ui(ctx, cfg, 1200)
    render_generate_captions_window(ctx, cfg, 1200)
    render_generate_masks_window(ctx, cfg, 1200)
    render_muon_adam_window(ctx, cfg, 1200)
    render_offloading_window(ctx, cfg, 1200)
    render_optimizer_params_window(ctx, cfg, 1200)
    render_profiling_window(ctx, cfg, 1200)
    render_sample_frame(ctx, cfg, 1200)
    render_sample_params_window(ctx, cfg, 1200)
    render_sample_window(ctx, cfg, rt, 1200)
    render_scheduler_params_window(ctx, cfg, 1200)
    render_timestep_distribution_window(ctx, cfg, 1200)
    render_video_tool_ui(ctx, cfg, 1200)
    ctx.end_frame()
    print("PASS: auxiliary UI surfaces")
