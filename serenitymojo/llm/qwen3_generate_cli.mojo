from std.gpu.host import DeviceContext
from serenitymojo.models.text_encoder.qwen3_encoder import Qwen3Encoder, Qwen3Config
from serenitymojo.models.text_encoder.qwen3_magic import generate_greedy

comptime SNAP = "/home/alex/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"

def main() raises:
    var ctx = DeviceContext()
    var enc = Qwen3Encoder.load(SNAP, Qwen3Config.qwen3_06b(), ctx)
    var ids = List[Int]()
    for t in [785, 6722, 315, 9625, 374]:
        ids.append(t)
    var gen = generate_greedy(enc, ids, 8, 151645, 0, 512, ctx)
    print("mojo greedy gen ids:")
    for i in range(len(gen)):
        print("  ", gen[i])
