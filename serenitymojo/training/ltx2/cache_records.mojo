# cache_records.mojo -- LTX-2 cache record and safetensors key contract.
#
# Mirrors musubi-tuner's ltx2 cache names. These are records and names only; a
# future data loader should use serenitymojo.io.safetensors to read the tensors.

comptime ARCHITECTURE_LTX2 = "ltx2"
comptime ARCHITECTURE_LTX2_FULL = "ltx2_v1"
comptime FORMAT_VERSION = "1.0.1"

comptime VIDEO_CACHE_SUFFIX = "_ltx2.safetensors"
comptime AUDIO_CACHE_SUFFIX = "_ltx2_audio.safetensors"
comptime TEXT_CACHE_SUFFIX = "_ltx2_te.safetensors"
comptime DINO_CACHE_SUFFIX = "_ltx2_dino.safetensors"


@fieldwise_init
struct LTX2CacheRecord(Copyable, Movable):
    var item_key: String
    var latent_cache_path: String
    var text_encoder_cache_path: String
    var audio_latent_cache_path: String
    var width: Int
    var height: Int
    var frame_count: Int
    var has_video_latents: Bool
    var has_audio_latents: Bool
    var has_text_cache: Bool

    def valid_geometry(self) -> Bool:
        return self.width > 0 and self.height > 0 and self.frame_count > 0

    def cache_ready(self) -> Bool:
        return self.valid_geometry() and self.has_video_latents and self.has_text_cache

    def av_ready(self) -> Bool:
        return self.cache_ready() and self.has_audio_latents


def dtype_suffix(dtype_name: String) -> String:
    if dtype_name == "BF16":
        return String("bf16")
    if dtype_name == "F16":
        return String("fp16")
    if dtype_name == "F32":
        return String("fp32")
    return dtype_name


def video_latents_key(frames: Int, height: Int, width: Int, dtype: String) -> String:
    return (
        String("latents_") + String(frames) + String("x") + String(height)
        + String("x") + String(width) + String("_") + dtype
    )

def video_clean_latents_key(frames: Int, height: Int, width: Int, dtype: String) -> String:
    return (
        String("latents_clean_") + String(frames) + String("x") + String(height)
        + String("x") + String(width) + String("_") + dtype
    )


def audio_latents_key(time_steps: Int, mel_bins: Int, channels: Int, dtype: String) -> String:
    return (
        String("audio_latents_") + String(time_steps) + String("x") + String(mel_bins)
        + String("x") + String(channels) + String("_") + dtype
    )


def audio_lengths_key(dtype: String = "int32") -> String:
    return String("audio_lengths_") + dtype


def video_prompt_embeds_key(dtype: String) -> String:
    return String("video_prompt_embeds_") + dtype


def audio_prompt_embeds_key(dtype: String) -> String:
    return String("audio_prompt_embeds_") + dtype


def legacy_text_key(dtype: String) -> String:
    return String("text_") + dtype


def prompt_attention_mask_key() -> String:
    return String("prompt_attention_mask")


def legacy_text_mask_key() -> String:
    return String("text_mask")


def latent_metadata_contract() -> String:
    return String("architecture,width,height,format_version,frame_count")


def text_metadata_contract() -> String:
    return String("architecture,caption1,format_version")


def default_record(item_key: String) -> LTX2CacheRecord:
    return LTX2CacheRecord(
        item_key,
        item_key + String(VIDEO_CACHE_SUFFIX),
        item_key + String(TEXT_CACHE_SUFFIX),
        item_key + String(AUDIO_CACHE_SUFFIX),
        0,
        0,
        0,
        False,
        False,
        False,
    )
