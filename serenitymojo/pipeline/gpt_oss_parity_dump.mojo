# serenitymojo/pipeline/gpt_oss_parity_dump.mojo
#
# PARITY GATE driver: run the streamed pure-Mojo GPT-OSS (Lens) text encoder on a
# fixed prompt's token ids and dump the captured hidden states at layers
# [5,11,17,23] to a safetensors file for cos-similarity comparison against the
# HF-transformers oracle.
#
# Token ids are HARD-CODED here (the raw tokenizer.json encode of
# "a photo of a cat" -> [64, 8767, 328, 261, 9059], no BOS/EOS, matching how the
# Rust GptOssEncoder::encode is fed input_ids directly). This keeps the Mojo
# driver tokenizer-free; the oracle uses the SAME ids.
#
# Output: serenitymojo/models/text_encoder/parity/mine_captures.safetensors
#   keys "l5","l11","l17","l23" each [1, S, 2880] BF16 (byte-exact storage).

from std.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.tensor import Tensor
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.models.text_encoder.gpt_oss_encoder import (
    GptOssEncoder,
    GptOssConfig,
    lens_extract_layers,
)


def main() raises:
    var te_dir = String(
        "/home/alex/.serenity/models/microsoft_lens/text_encoder"
    )
    var out_path = String(
        "/home/alex/mojodiffusion/serenitymojo/models/text_encoder/parity/mine_captures.safetensors"
    )

    # "a photo of a cat" per tokenizer.json (raw, no specials).
    var token_ids = List[Int]()
    token_ids.append(64)
    token_ids.append(8767)
    token_ids.append(328)
    token_ids.append(261)
    token_ids.append(9059)

    print("[parity-dump] prompt tokens:", len(token_ids))
    for i in range(len(token_ids)):
        print("  id[", i, "] =", token_ids[i])

    var ctx = DeviceContext()
    var cfg = GptOssConfig.lens_default()
    print(
        "[parity-dump] cfg hidden/layers/heads/kv/experts:",
        cfg.hidden_size,
        cfg.num_layers,
        cfg.num_heads,
        cfg.num_kv_heads,
        cfg.num_experts,
    )

    var enc = GptOssEncoder.load(te_dir, cfg, ctx)
    # BISECT: capture 5,11,17 + every layer 18..23 to localize the l23 blowup.
    var extract = List[Int]()
    extract.append(5)
    extract.append(11)
    extract.append(17)
    extract.append(18)
    extract.append(19)
    extract.append(20)
    extract.append(21)
    extract.append(22)
    extract.append(23)
    print("[parity-dump] extract layers count:", len(extract))

    var caps = enc.encode(token_ids, extract, ctx)
    print("[parity-dump] captures returned:", len(caps))

    # encode() returns captures sorted ASCENDING by layer index, deduped.
    var names = List[String]()
    names.append("l5")
    names.append("l11")
    names.append("l17")
    names.append("l18")
    names.append("l19")
    names.append("l20")
    names.append("l21")
    names.append("l22")
    names.append("l23")

    # Report shape + basic stats per capture (read what we actually produced).
    for i in range(len(caps)):
        ref t = caps[i][]
        var sh = t.shape()
        print(
            "[parity-dump]",
            names[i],
            "shape:",
            sh[0],
            sh[1],
            sh[2],
            "numel:",
            t.numel(),
        )
        var host = t.to_host(ctx)
        var n = len(host)
        var s = Float32(0.0)
        var ss = Float32(0.0)
        var amax = Float32(0.0)
        for j in range(n):
            var v = host[j]
            s += v
            ss += v * v
            var av = v if v >= 0.0 else -v
            if av > amax:
                amax = av
        var mean = s / Float32(n)
        var var_ = ss / Float32(n) - mean * mean
        print(
            "[parity-dump]",
            names[i],
            "mean/var/absmax:",
            mean,
            var_,
            amax,
        )

    save_safetensors(names, caps, out_path, ctx)
    print("[parity-dump] wrote:", out_path)
    print("[parity-dump] DONE")
