# conditioning.mojo -- text/audio/video conditioning selection contract.

from serenitymojo.training.ltx2.config import MODE_AUDIO, MODE_AV, MODE_VIDEO


comptime TEXT_SOURCE_VIDEO = 0
comptime TEXT_SOURCE_AUDIO = 1
comptime TEXT_SOURCE_COMBINED_AV = 2
comptime TEXT_SOURCE_SPLIT_VIDEO = 3
comptime TEXT_SOURCE_SPLIT_AUDIO = 4


@fieldwise_init
struct ConditioningSelection(Copyable, Movable):
    var text_source: Int
    var video_enabled: Bool
    var audio_enabled: Bool
    var requires_audio_latents: Bool
    var split_combined_context: Bool
    var video_context_dim: Int
    var audio_context_dim: Int
    var coupled_timesteps: Bool

    def uses_audio_branch(self) -> Bool:
        return self.audio_enabled

    def uses_video_branch(self) -> Bool:
        return self.video_enabled


def text_source_name(source: Int) -> String:
    if source == TEXT_SOURCE_VIDEO:
        return String("video_prompt_embeds")
    if source == TEXT_SOURCE_AUDIO:
        return String("audio_prompt_embeds")
    if source == TEXT_SOURCE_COMBINED_AV:
        return String("prompt_embeds[video|audio]")
    if source == TEXT_SOURCE_SPLIT_VIDEO:
        return String("split(prompt_embeds).video")
    if source == TEXT_SOURCE_SPLIT_AUDIO:
        return String("split(prompt_embeds).audio")
    return String("unknown")


def can_split_combined_context(total_dim: Int, expected_video_dim: Int, expected_audio_dim: Int) -> Bool:
    if expected_video_dim > 0 and expected_audio_dim > 0:
        return total_dim == expected_video_dim + expected_audio_dim
    return total_dim > 0 and (total_dim % 2) == 0


def combined_context_video_dim(total_dim: Int, expected_video_dim: Int, expected_audio_dim: Int) -> Int:
    if expected_video_dim > 0 and expected_audio_dim > 0 and total_dim == expected_video_dim + expected_audio_dim:
        return expected_video_dim
    if total_dim % 2 == 0:
        return total_dim // 2
    return total_dim


def combined_context_audio_dim(total_dim: Int, expected_video_dim: Int, expected_audio_dim: Int) -> Int:
    if expected_video_dim > 0 and expected_audio_dim > 0 and total_dim == expected_video_dim + expected_audio_dim:
        return expected_audio_dim
    if total_dim % 2 == 0:
        return total_dim // 2
    return total_dim


def select_conditioning(
    mode: Int,
    has_audio_latents: Bool,
    has_video_prompt_embeds: Bool,
    has_audio_prompt_embeds: Bool,
    prompt_embed_dim: Int,
    expected_video_dim: Int,
    expected_audio_dim: Int,
    independent_audio_timestep: Bool,
) -> ConditioningSelection:
    if mode == MODE_VIDEO:
        var source = TEXT_SOURCE_VIDEO if has_video_prompt_embeds else TEXT_SOURCE_SPLIT_VIDEO
        return ConditioningSelection(
            source, True, False, False,
            not has_video_prompt_embeds and can_split_combined_context(prompt_embed_dim, expected_video_dim, expected_audio_dim),
            expected_video_dim, 0, True,
        )
    if mode == MODE_AUDIO:
        var asource = TEXT_SOURCE_AUDIO if has_audio_prompt_embeds else TEXT_SOURCE_SPLIT_AUDIO
        return ConditioningSelection(
            asource, False, True, True,
            not has_audio_prompt_embeds and can_split_combined_context(prompt_embed_dim, expected_video_dim, expected_audio_dim),
            0, expected_audio_dim, not independent_audio_timestep,
        )
    if has_audio_latents:
        var split = can_split_combined_context(prompt_embed_dim, expected_video_dim, expected_audio_dim)
        return ConditioningSelection(
            TEXT_SOURCE_COMBINED_AV,
            True,
            True,
            False,
            split,
            combined_context_video_dim(prompt_embed_dim, expected_video_dim, expected_audio_dim),
            combined_context_audio_dim(prompt_embed_dim, expected_video_dim, expected_audio_dim),
            not independent_audio_timestep,
        )
    var vsource = TEXT_SOURCE_VIDEO if has_video_prompt_embeds else TEXT_SOURCE_SPLIT_VIDEO
    return ConditioningSelection(
        vsource,
        True,
        False,
        False,
        not has_video_prompt_embeds and can_split_combined_context(prompt_embed_dim, expected_video_dim, expected_audio_dim),
        expected_video_dim,
        0,
        True,
    )
