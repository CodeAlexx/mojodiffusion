# serenitymojo/training/bernini_tasks.mojo
#
# BERNINI-R per-task recipe table — the exact training recipe for each of the
# renderer task family, verified against
#   /home/alex/Bernini/configs/bernini_renderer_train/train_cfg/bernini_renderer_high.yaml
#   (noise_scheduler_config: shift_config + weighting_scheme_config)
#   /home/alex/Bernini/bernini/training/data.py  (SYSTEM_PROMPTS :32-46,
#                                                 encode/pack_vae_latents ordering)
#
# For each task this returns:
#   shift          — discrete-flow shift fed to the FlowMatchScheduler sigma map.
#   is_mode        — weighting scheme: True = "mode" (video), False =
#                    "logit_normal" (image). Drives which density the timestep
#                    sampler draws (training/schedule.mojo bernini_sample_sigma).
#   n_cond         — number of CLEAN conditioning source-id segments packed AHEAD
#                    of the noised target (each -> source_id = idx+1; target=0).
#   cond_kind      — human-readable conditioning shape (documentation).
#   system_prompt  — the task's SYSTEM_PROMPTS entry (feeds the umt5 planner text;
#                    the Tier-2b renderer trainer conditions on the packed VAE
#                    latents and keeps text synthetic, so this is threaded as
#                    metadata now and wired to a real tokenizer as a follow-up).
#
# Fallback rule (data.py NoiseScheduler.get_noise_sigma): a task NOT present in
# the config maps weighting by suffix — "*2i" -> image(logit_normal),
# "*2v" -> video(mode) — and shift -> default 5.0. Only "ads2v" uses this path.
#
# Mojo 1.0.0b1. Pure host config; no device work.

from std.collections import List


@fieldwise_init
struct BerniniTaskRecipe(Copyable, Movable):
    var task: String
    var shift: Float64
    var is_mode: Bool          # True = mode (video), False = logit_normal (image)
    var n_cond: Int            # number of conditioning source-id segments
    var cond_kind: String      # conditioning shape (doc)
    var system_prompt: String


# The 12 renderer tasks. shift/weighting are EXACT from bernini_renderer_high.yaml;
# n_cond/cond_kind from the task family definition; system_prompt from data.py.
def bernini_recipe_for(task: String) raises -> BerniniTaskRecipe:
    if task == String("t2i"):
        return BerniniTaskRecipe(String("t2i"), 3.0, False, 0, String("none"),
            String("You are a helpful assistant specialized in text-to-image generation."))
    if task == String("t2v"):
        return BerniniTaskRecipe(String("t2v"), 3.0, True, 0, String("none"),
            String("You are a helpful assistant specialized in text-to-video generation."))
    if task == String("i2i"):
        return BerniniTaskRecipe(String("i2i"), 4.0, False, 1, String("image"),
            String("You are a helpful assistant specialized in image editing."))
    if task == String("r2i"):
        return BerniniTaskRecipe(String("r2i"), 4.0, False, 1, String("reference"),
            String("You are a helpful assistant specialized in subject-to-image generation."))
    if task == String("r2v"):
        return BerniniTaskRecipe(String("r2v"), 4.0, True, 1, String("reference"),
            String("You are a helpful assistant specialized in subject-to-video generation."))
    if task == String("v2v"):
        return BerniniTaskRecipe(String("v2v"), 5.0, True, 1, String("video"),
            String("You are a helpful assistant specialized in video editing."))
    if task == String("i2v"):
        return BerniniTaskRecipe(String("i2v"), 5.0, True, 1, String("image"),
            String("You are a helpful assistant specialized in image-to-video generation."))
    if task == String("vi2v"):
        return BerniniTaskRecipe(String("vi2v"), 5.0, True, 2, String("video+image"),
            String("You are a helpful assistant specialized in video editing on content propagation."))
    if task == String("vr2v"):
        return BerniniTaskRecipe(String("vr2v"), 5.0, True, 2, String("video+reference"),
            String("You are a helpful assistant specialized in video editing with reference."))
    if task == String("vrc2v"):
        return BerniniTaskRecipe(String("vrc2v"), 5.0, True, 3, String("video+reference+control"),
            String("You are a helpful assistant for editing. You may need to adjust the subject's action or position."))
    if task == String("mv2v"):
        # multiple videos: N variable. Representative smoke uses N=2 (>=2).
        return BerniniTaskRecipe(String("mv2v"), 5.0, True, 2, String("multi-video(N>=2)"),
            String("You are a helpful assistant for editing. You might need to adjust the video's style, lighting, colors, textures, and the subject's pose or action."))
    if task == String("ads2v"):
        # NOT in shift/weighting config -> fallback: ends in 'v' => mode, shift=default 5.0.
        return BerniniTaskRecipe(String("ads2v"), 5.0, True, 2, String("video+image(ads)"),
            String("You are a helpful assistant specialized in ads insertion."))
    raise Error(String("bernini_recipe_for: unknown task '") + task + String("'"))


# Per-conditioning-segment token-grid geometry for the representative smoke.
# Each segment is a CLEAN VAE latent patchified (1,2,2). The grid (f,h,w) is the
# post-patchify token grid; token count = f*h*w. Real caches override these with
# the true cached latent grids.
@fieldwise_init
struct BerniniCondSeg(Copyable, Movable):
    var f: Int
    var h: Int
    var w: Int
    var kind: String          # "image" | "video" | "reference" | "control"


# Build the smoke conditioning-segment list for a task (kind-by-kind).
# image/reference/control -> single-frame 8x8 grid (64 tokens);
# video -> 2-frame 8x8 grid (128 tokens).
def bernini_smoke_cond_segments(task: String) raises -> List[BerniniCondSeg]:
    var segs = List[BerniniCondSeg]()
    if task == String("t2i") or task == String("t2v"):
        return segs^
    if task == String("i2i") or task == String("i2v"):
        segs.append(BerniniCondSeg(1, 8, 8, String("image")))
    elif task == String("r2i") or task == String("r2v"):
        segs.append(BerniniCondSeg(1, 8, 8, String("reference")))
    elif task == String("v2v"):
        segs.append(BerniniCondSeg(2, 8, 8, String("video")))
    elif task == String("vi2v"):
        segs.append(BerniniCondSeg(2, 8, 8, String("video")))
        segs.append(BerniniCondSeg(1, 8, 8, String("image")))
    elif task == String("vr2v"):
        segs.append(BerniniCondSeg(2, 8, 8, String("video")))
        segs.append(BerniniCondSeg(1, 8, 8, String("reference")))
    elif task == String("ads2v"):
        segs.append(BerniniCondSeg(2, 8, 8, String("video")))
        segs.append(BerniniCondSeg(1, 8, 8, String("image")))
    elif task == String("vrc2v"):
        segs.append(BerniniCondSeg(2, 8, 8, String("video")))
        segs.append(BerniniCondSeg(1, 8, 8, String("reference")))
        segs.append(BerniniCondSeg(1, 8, 8, String("control")))
    elif task == String("mv2v"):
        segs.append(BerniniCondSeg(2, 8, 8, String("video")))
        segs.append(BerniniCondSeg(2, 8, 8, String("video")))
    else:
        raise Error(String("bernini_smoke_cond_segments: unknown task '") + task + String("'"))
    return segs^
