# sharded_vae_parity.mojo — SKEPTIC chunk-2: full byte-parity for the single-file
# fallback path. Opens the Z-Image VAE *directory* via ShardedSafeTensors (no
# index → single-file fallback) and the resolved .safetensors directly via
# chunk-1 SafeTensors, then full-FNV compares ALL 244 tensors.
#
# Run: pixi run mojo run -I . serenitymojo/io/parity/sharded_vae_parity.mojo

from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.safetensors import SafeTensors


comptime VAE_DIR = (
    "/home/alex/.cache/huggingface/hub/models--Tongyi-MAI--Z-Image/"
    "snapshots/04cc4abb7c5069926f75c9bfde9ef43d49423021/vae"
)
comptime VAE_FILE = String(VAE_DIR) + "/diffusion_pytorch_model.safetensors"


def fnv1a(span: Span[UInt8, _]) -> UInt64:
    var h: UInt64 = 0xCBF29CE484222325
    for i in range(len(span)):
        h = h ^ UInt64(span[i])
        h = h * 0x00000100000001B3
    return h


def main() raises:
    var sh = ShardedSafeTensors.open(String(VAE_DIR))
    var direct = SafeTensors.open(String(VAE_FILE))
    print("VAE single-file fallback: num_shards =", sh.num_shards(),
          " num_tensors =", sh.num_tensors())
    print("direct chunk-1 count =", direct.count())

    var total = 0
    var matched = 0
    var mism = List[String]()
    for ref nm in sh.names():
        total += 1
        var a_info = sh.tensor_info(nm)
        var a = sh.tensor_bytes(nm)
        var b_info = direct.tensor_info(nm)
        var b = direct.tensor_bytes(nm)
        var ok = True
        if len(a) != len(b):
            ok = False
        if fnv1a(a) != fnv1a(b):
            ok = False
        if a_info.dtype.name() != b_info.dtype.name():
            ok = False
        if len(a_info.shape) != len(b_info.shape):
            ok = False
        else:
            for k in range(len(a_info.shape)):
                if a_info.shape[k] != b_info.shape[k]:
                    ok = False
        if ok:
            matched += 1
        else:
            mism.append(nm)

    print("VAE TOTAL:", total, " MATCHED:", matched, " MISMATCH:", len(mism))
    for ref m in mism:
        print("  MISMATCH:", m)
    if matched == total and total == 244:
        print("VAE SINGLE-FILE PARITY PASSED (244/244)")
    else:
        raise Error("VAE parity failed")
