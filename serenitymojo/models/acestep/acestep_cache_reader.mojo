# acestep_cache_reader.mojo — ACE-Step-1.5 train-cache reader (Tier-3 T3.C, #10).
#
# Reads the per-sample BF16 safetensors cache that scripts/acestep_pt_to_cache.py
# writes (from the upstream OFFLINE precompute .pt files, or the parity oracle
# dump). A cache dir has one `sample_NNNNN.safetensors` per sample + `manifest.txt`
# (one filename per line). Each sample carries the 5 recipe tensors:
#   target_latents          [1,T,64]     (Oobleck VAE latent — data x0)
#   context_latents         [1,T,128]    (64 src_lat + 64 chunk_mask)
#   encoder_hidden_states   [1,L,2048]   (Qwen3-Embedding cond, RAW)
#   attention_mask          [1,T]        (stored; unused at SP<=window — no-op mask)
#   encoder_attention_mask  [1,L]        (stored; unused at L==SP no-pad cross)
# The trainer (train_acestep.mojo) reads samples[step % N]; the loop is unchanged
# whether the batch comes from the cache or the oracle dump.
#
# Mojo 1.0.0b1, NVIDIA.

from max.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.disk_check import _read_text
from serenitymojo.ops.cast import cast_tensor


struct AcestepTrainSample(Movable):
    var target_latents: Tensor   # [1,T,64]  data x0
    var context: Tensor          # [1,T,128]
    var ehs: Tensor              # [1,L,2048] RAW encoder_hidden_states
    var attn_mask: Tensor        # [1,T]  (stored; driver assumes full attn)
    var enc_mask: Tensor         # [1,L]

    def __init__(
        out self, var target_latents: Tensor, var context: Tensor,
        var ehs: Tensor, var attn_mask: Tensor, var enc_mask: Tensor,
    ):
        self.target_latents = target_latents^
        self.context = context^
        self.ehs = ehs^
        self.attn_mask = attn_mask^
        self.enc_mask = enc_mask^


def _read_bf16(st: ShardedSafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    return cast_tensor(Tensor.from_view(st.tensor_view(name), ctx), STDtype.BF16, ctx)


def acestep_load_sample(path: String, ctx: DeviceContext) raises -> AcestepTrainSample:
    """Load one cache sample safetensors → the 5 BF16 tensors."""
    var st = ShardedSafeTensors.open(path)
    return AcestepTrainSample(
        _read_bf16(st, "target_latents", ctx),
        _read_bf16(st, "context_latents", ctx),
        _read_bf16(st, "encoder_hidden_states", ctx),
        _read_bf16(st, "attention_mask", ctx),
        _read_bf16(st, "encoder_attention_mask", ctx),
    )


def acestep_read_manifest(cache_dir: String) raises -> List[String]:
    """Read `<cache_dir>/manifest.txt` → the list of sample paths (one per line)."""
    var text = _read_text(cache_dir + "/manifest.txt")
    var paths = List[String]()
    var cur = String("")
    var bytes = text.as_bytes()
    for i in range(len(bytes)):
        if bytes[i] == UInt8(10):        # '\n'
            if cur.byte_length() > 0:
                paths.append(cache_dir + "/" + cur)
            cur = String("")
        else:
            cur += chr(Int(bytes[i]))
    if cur.byte_length() > 0:
        paths.append(cache_dir + "/" + cur)
    if len(paths) == 0:
        raise Error("acestep_read_manifest: empty manifest at " + cache_dir)
    return paths^

# NOTE: samples are STREAMED by path (acestep_load_sample per step) — the trainer
# holds only one resident at a time (Tensor is not Copyable, and streaming scales
# to large datasets). `acestep_load_sample` reads the 5 keys by name, so it also
# works directly on the parity oracle dump (which carries those keys + extras).
