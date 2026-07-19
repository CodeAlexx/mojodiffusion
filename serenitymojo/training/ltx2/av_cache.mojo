# av_cache.mojo -- LTX-2 AV tri-pair cache pairing (P6.2).
#
# Host-side plumbing that pairs, with each video latent cache file
# `.../{stem}_ltx2.safetensors`, its two siblings in the SAME dir (musubi
# ltx2_cache_latents.py:260 basename route):
#   * `{stem}_ltx2_audio.safetensors`  audio_latents_{T}x{mel}x{C}_{dtype}
#                                      ([C,T,mel]) + audio_lengths_int32 (scalar)
#   * `{stem}_ltx2_te.safetensors`     video_prompt_embeds_* + audio_prompt_embeds_*
#                                      + prompt_attention_mask
# Audio latent stored [C,T,mel]; the audio patchify in_features = C*mel and the
# audio stream is S_A=T tokens (ltx2_cache_latents.py:473-490,
# ltx2_train_network.py:208). Path construction + key discovery only (no
# DeviceContext); the GPU loads stay in the trainer / gate.

from serenitymojo.io.sharded import ShardedSafeTensors


def _replace_ext(path: String, suffix: String) -> String:
    # path ends with ".safetensors" (12 bytes) -> strip and append suffix.
    var keep = path.byte_length() - 12
    var b = path.as_bytes()
    var out = String("")
    for i in range(keep):
        out += chr(Int(b[i]))
    return out + suffix


def audio_cache_path(video_path: String) -> String:
    """`.../{stem}_ltx2.safetensors` -> `{stem}_ltx2_audio.safetensors`."""
    return _replace_ext(video_path, String("_audio.safetensors"))


def te_cache_path(video_path: String) -> String:
    """`.../{stem}_ltx2.safetensors` -> `{stem}_ltx2_te.safetensors`."""
    return _replace_ext(video_path, String("_te.safetensors"))


def discover_audio_latents_key(st: ShardedSafeTensors) raises -> String:
    """The single `audio_latents_*` key. Fail loud on zero/ambiguous."""
    var found = String("")
    var count = 0
    var names = st.names()
    for ref n in names:
        if n.startswith("audio_latents_"):
            found = n.copy()
            count += 1
    if count == 0:
        raise Error("av audio cache has no audio_latents_* key")
    if count > 1:
        raise Error(String("av audio cache has ") + String(count) + " audio_latents_* keys (ambiguous)")
    return found^


def discover_audio_lengths_key(st: ShardedSafeTensors) raises -> String:
    """The single `audio_lengths_*` key (int32)."""
    var names = st.names()
    for ref n in names:
        if n.startswith("audio_lengths_"):
            return n.copy()
    raise Error("av audio cache has no audio_lengths_* key")
