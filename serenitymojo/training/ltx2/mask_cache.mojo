# mask_cache.mojo -- LTX2 intrinsic INPAINT mask-cache loader (P5.5 unit 2).
#
# Reads a per-sample binary latent mask from a mask cache directory and returns
# the [seq] {0,1} token mask (frame->h->w order) for the union, mirroring the
# official LTX-2 MASK condition:
#   flexible.py:503-506  mask = (batch[mask_dir]["mask"].reshape(B, seq) > 0.5).float()
# The cache files are produced by the preprocessing pass (avg_pool32 spatial +
# amax over VAE_TEMPORAL_FACTOR frames + binarize; process_videos.py:905-988),
# stored beside the latent cache under the SAME basename in a mask_cache_directory
# (the v2v reference-cache route), key "mask" shape [NF,NH,NW] (or [1,NF,NH,NW]).
#
# HOST-only (no DeviceContext): reads the raw bytes via ShardedSafeTensors and
# thresholds >0.5. F32 and BF16 storage supported (the mask is binary either way).

from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.dtype import STDtype


def mask_cache_path(lat_path: String, mask_cache_dir: String) -> String:
    """mask_cache_dir + "/" + basename(lat_path) (same-basename route)."""
    var b = lat_path.as_bytes()
    var start = 0
    for i in range(len(b)):
        if b[i] == UInt8(47):  # '/'
            start = i + 1
    var base = String("")
    for i in range(start, len(b)):
        base += chr(Int(b[i]))
    var d = mask_cache_dir
    var dlen = d.byte_length()
    if dlen > 0 and d.as_bytes()[dlen - 1] == UInt8(47):
        return d + base
    return d + "/" + base


# mask values are NONNEGATIVE [0,1] (avgpool / binarize), and for nonnegative
# IEEE floats the raw bit pattern is monotonic in value, so `value > 0.5` <=>
# `bits > bits(0.5)` — no bitcast needed. 0.5: F32 0x3F000000, BF16 0x3F00.
def _f32_gt_half(bytes: List[UInt8], i: Int) -> Bool:
    var o = i * 4
    var bits = (
        UInt32(bytes[o]) | (UInt32(bytes[o + 1]) << 8)
        | (UInt32(bytes[o + 2]) << 16) | (UInt32(bytes[o + 3]) << 24)
    )
    return bits > UInt32(0x3F000000)


def _bf16_gt_half(bytes: List[UInt8], i: Int) -> Bool:
    var o = i * 2
    var half = UInt32(bytes[o]) | (UInt32(bytes[o + 1]) << 8)
    return half > UInt32(0x3F00)


# Load the per-sample inpaint mask -> [seq] {0,1} token mask (frame->h->w),
# thresholded >0.5 (official binarize-at-load). Fail loud on missing key or a
# token-count mismatch (the mask must be the target latent's own [NF,NH,NW]).
def load_mask_tokens(mask_path: String, nf: Int, nh: Int, nw: Int) raises -> List[Bool]:
    var seq = nf * nh * nw
    var st = ShardedSafeTensors.open(mask_path)
    if not st.has_tensor("mask"):
        raise Error(String("inpaint mask cache missing 'mask' key: ") + mask_path)
    var info = st.tensor_info("mask")
    var numel = 1
    for i in range(len(info.shape)):
        numel *= info.shape[i]
    if numel != seq:
        raise Error(
            String("inpaint mask numel ") + String(numel) + " != seq " + String(seq)
            + " (mask must be the target [NF,NH,NW]): " + mask_path)
    var span = st.tensor_bytes("mask")
    var bytes = List[UInt8]()
    for i in range(len(span)):
        bytes.append(span[i])
    var out = List[Bool]()
    if info.dtype == STDtype.F32:
        for i in range(seq):
            out.append(_f32_gt_half(bytes, i))
    elif info.dtype == STDtype.BF16:
        for i in range(seq):
            out.append(_bf16_gt_half(bytes, i))
    else:
        raise Error("inpaint mask cache dtype must be F32 or BF16")
    return out^
