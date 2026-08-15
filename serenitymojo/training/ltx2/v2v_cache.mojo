# v2v_cache.mojo -- LTX-2 IC-LoRA / V2V reference-latent cache pairing (P5 u1).
#
# Host-side plumbing that pairs a downscaled REFERENCE latent with each target
# latent, mirroring torchref's TRAINING route
# (src/torchref/dataset/image_video_dataset.py):
#   * the reference latent lives in a SEPARATE reference_cache_directory, under
#     the SAME basename as the target cache file
#     (image_video_dataset.py:2780-2781
#        ref_cache_path = os.path.join(reference_cache_directory,
#                                      os.path.basename(cache_file))),
#   * it is a standard ltx2 latent cache written by save_latent_cache_ltx2
#     (ltx2_cache_latents.py:687, from encode_and_save_reference_latents), keyed
#     `latents_{F}x{H}x{W}_{dtype}` on its OWN spatially-downscaled grid
#     (ref_w = max((w//ds//32)*32, 32); ltx2_cache_latents.py:565-566,
#     single image repeated to --reference_frames),
#   * the target's forward reads it as `ref_latents` and PREPENDS it on the seq
#     axis (ltx2_train_network.py:3345-3356).
#
# ALL functions here are host-side (NO DeviceContext): path construction, key
# discovery, and fail-loud rank/channel validation. The GPU load
# (Tensor.from_view_as_bf16) stays in the trainer -> the v2v forward wiring is
# P5 unit 2. torchref's `v2v_ref_latent` name (the sample-prompt route,
# ltx2_cache_latents.py:1019) is the SEMANTIC name for this stream; the
# dataset/training file itself uses the standard latents_* key.

from serenitymojo.io.sharded import ShardedSafeTensors


def reference_cache_path(lat_path: String, ref_cache_dir: String) -> String:
    """ref_cache_dir + "/" + basename(lat_path) -- torchref basename route
    (image_video_dataset.py:2781)."""
    var b = lat_path.as_bytes()
    var start = 0
    for i in range(len(b)):
        if b[i] == UInt8(47):  # '/'
            start = i + 1
    var base = String("")
    for i in range(start, len(b)):
        base += chr(Int(b[i]))
    var d = ref_cache_dir
    var dlen = d.byte_length()
    if dlen > 0 and d.as_bytes()[dlen - 1] == UInt8(47):
        return d + base
    return d + "/" + base


def discover_ref_latent_key(st: ShardedSafeTensors) raises -> String:
    """The single `latents_*` key (excluding `latents_clean_*`) in a reference
    cache file. Fail loud on zero or more than one -- an ambiguous ref cache is
    a contract error, not a silent pick."""
    var found = String("")
    var count = 0
    var names = st.names()
    for ref n in names:
        if n.startswith("latents_") and not n.startswith("latents_clean_"):
            found = n.copy()
            count += 1
    if count == 0:
        raise Error("v2v ref cache has no latents_* key")
    if count > 1:
        raise Error(
            String("v2v ref cache has ") + String(count)
            + " latents_* keys (ambiguous)")
    return found^


def assert_reference_latent(
    st: ShardedSafeTensors, ref_key: String, channels: Int
) raises:
    """torchref asserts 5D + channel match (ltx2_train_network.py path); the stored
    ltx2 latent is [C,F,H,W] (batch squeezed at ltx2_cache_latents.py:678
    `ref_latent = latent[0]`), so accept rank 4 [C,F,H,W] or rank 5 [1,C,F,H,W].
    Fail loud on wrong rank, channel mismatch, or a non-positive spatial dim."""
    var info = st.tensor_info(ref_key)
    ref shp = info.shape
    var rank = len(shp)
    if rank != 4 and rank != 5:
        raise Error(
            String("v2v ref latent '") + ref_key + "' rank " + String(rank)
            + " (expected 4 [C,F,H,W] or 5 [1,C,F,H,W])")
    var ch = shp[0] if rank == 4 else shp[1]
    if ch != channels:
        raise Error(
            String("v2v ref latent channels ") + String(ch)
            + " != target channels " + String(channels))
    var first_spatial = 1 if rank == 4 else 2
    for i in range(first_spatial, rank):
        if shp[i] <= 0:
            raise Error(
                String("v2v ref latent '") + ref_key
                + "' has a non-positive spatial dim")


def pair_reference(
    lat_path: String, ref_cache_dir: String, channels: Int
) raises -> String:
    """Full host-side pairing for one target: locate the reference cache file,
    open it (fail loud if missing/unreadable -- open() puts the path in its
    message), discover its latents_* key, validate rank/channels. Returns the
    reference cache PATH; the trainer stores it on the CacheItem and the forward
    (unit 2) re-discovers the key via discover_ref_latent_key."""
    var ref_path = reference_cache_path(lat_path, ref_cache_dir)
    var st = ShardedSafeTensors.open(ref_path)
    var key = discover_ref_latent_key(st)
    assert_reference_latent(st, key, channels)
    return ref_path^
