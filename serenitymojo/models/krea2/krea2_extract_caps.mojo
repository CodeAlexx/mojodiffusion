# models/krea2/krea2_extract_caps.mojo — extract per-prompt context tensors from
# a krea2 training cache into raw .bin files for the inline sampler's caps path.
#
# WHY: the trainer's inline sampler has NO text encoder; custom sample prompts
# need PRECACHED conditioning (`caps: {positive: <bin>, negative: <bin>}` in the
# sample-prompts JSON → _krea2_inline_cond_from_bin → load_tensor_bin). The
# cache prep (krea2_prepare_cache) already encodes prompt.<i>.txt → context.<i>
# and uncond.txt → context_uncond; this tool just re-exports them as bins.
#
# usage: krea2_extract_caps <cache.safetensors> <out_dir> <n>
#   writes <out_dir>/cap_pos_<i>.bin for context.<i>, i in [0,n)
#   and    <out_dir>/cap_neg.bin     for context_uncond (if present)
#
# Mojo 1.0.0b1, NVIDIA GPU (device round-trip via save_tensor_bin).

from std.sys import argv
from max.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.io.cap_cache import save_tensor_bin


def _load(st: SafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    # SafeTensors exposes tensor_info + tensor_bytes (no tensor_view); build the
    # view via from_parts (== krea2_fp8_cache._load_raw_h2d) and H2D it.
    var info = st.tensor_info(name)
    var bytes = st.tensor_bytes(name)
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    return Tensor.from_view(tv, ctx)


def main() raises:
    var args = argv()
    if len(args) < 4:
        raise Error("usage: krea2_extract_caps <cache.safetensors> <out_dir> <n>")
    var cache_path = String(args[1])
    var out_dir = String(args[2])
    var n = Int(String(args[3]))

    var ctx = DeviceContext()
    var st = SafeTensors.open(cache_path)

    for i in range(n):
        var key = String("context.") + String(i)
        var t = _load(st, key, ctx)
        var out = out_dir + String("/cap_pos_") + String(i) + String(".bin")
        save_tensor_bin(t, out, ctx)
        print("wrote", out, " shape[1]=", t.shape()[1])

    # uncond (present only when the cache was prepared with uncond.txt)
    try:
        var u = _load(st, String("context_uncond"), ctx)
        var uout = out_dir + String("/cap_neg.bin")
        save_tensor_bin(u, uout, ctx)
        print("wrote", uout, " shape[1]=", u.shape()[1])
    except e:
        print("no context_uncond in cache (cfg>1 sampling needs one):", String(e))
