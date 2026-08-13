# ideogram4_prepare_cache.mojo — stage B of the Ideogram-4 cache prepare:
# the staged images + rendered prompts (scripts/ideogram4_stage_images.py)
# go through the GATED Mojo encoders (Ideogram4VaeEncoder, Qwen3Tokenizer +
# ideogram4_encode_text 13-tap) into the indexed safetensors training cache
# the Ideogram4CacheReader streams (clean.<i> [1,128,GH,GW] F32 +
# llm.<i> [1,NT,53248] BF16).
#
# WHY THIS EXISTS (measured 2026-06-11): the UI's "cache" default was the
# ideogram4_fx_predict PARITY FIXTURE (one fixed sample) — a 3000-step run
# completed with loss ~1.3e-4 / grad_norm 0.0000 = trained on garbage.
# This tool builds the REAL multi-sample cache.
#
# Text padding: token ids padded (or truncated) to NT with 151643 (the
# eos/pad id the tokenizer-probe oracle sequence ends with) BEFORE encoding,
# so llm features are the encoder's real output at fixed NT.
# HYPOTHESIS to cross-check vs DiffSynth-Studio: their pad strategy.
#
# Run (after stage A; GPU free; ~70 samples => one Qwen3-VL + VAE pass each):
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#   pixi run mojo build -I . -I trainer/src \
#     -Xlinker -lm -Xlinker -lcuda -Xlinker -L.pixi/envs/default/lib \
#     -Xlinker -lsqlite3 \
#     trainer/smoke/ideogram4_prepare_cache.mojo \
#     -o /tmp/ideogram4_prepare
#   LD_LIBRARY_PATH=/home/alex/mojodiffusion/.pixi/envs/default/lib \
#     /tmp/ideogram4_prepare <stage_dir> <out_cache.safetensors> <n>

from max.gpu.host import DeviceContext
from std.memory import ArcPointer
from std.sys import argv

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.tensor_algebra import mul
from serenitymojo.registry.checkpoints import path_exists
from serenitymojo.tokenizer.tokenizer import Qwen3Tokenizer
from serenity_trainer.model.Ideogram4VAE import Ideogram4VaeEncoder
from serenity_trainer.model.Ideogram4TextEncoder import (
    ideogram4_load_text_encoder_default,
    ideogram4_encode_text,
)

comptime TArc = ArcPointer[Tensor]
comptime TOK = "/home/alex/.serenity/models/ideogram-4-fp8/tokenizer/tokenizer.json"
comptime NT = 256          # text bucket: ids padded with PAD_ID, then encoded
comptime PAD_ID = 151643   # Qwen3 eos/pad (the tokenizer-probe oracle tail)
comptime LH = 64           # 512px -> VAE latent 64x64 -> packed [1,128,32,32]
comptime LW = 64


def _read_text(path: String) raises -> String:
    var f = open(path, "r")
    var s = f.read()
    f.close()
    return s^


def main() raises:
    var args = argv()
    if len(args) < 4:
        raise Error("usage: ideogram4_prepare <stage_dir> <out.safetensors> <n>")
    var stage_dir = String(args[1])
    var out_path = String(args[2])
    var n = Int(String(args[3]))

    var ctx = DeviceContext()
    print("[prepare] loading VAE encoder (512px bucket)")
    var venc = Ideogram4VaeEncoder[LH, LW].load_default(ctx)
    print("[prepare] loading Qwen3-VL text encoder")
    var tenc = ideogram4_load_text_encoder_default(ctx)
    var tok = Qwen3Tokenizer(String(TOK))

    var imgs = ShardedSafeTensors.open(stage_dir + "/images.safetensors")

    var names = List[String]()
    var tensors = List[TArc]()
    for i in range(n):
        # ── image -> packed normalized latent [1,128,32,32] F32 ──
        var img = cast_tensor(
            Tensor.from_view(imgs.tensor_view(String("image.") + String(i)), ctx),
            STDtype.BF16, ctx,
        )
        var lat = venc.encode(img, ctx)
        var clean = cast_tensor(lat, STDtype.F32, ctx)

        # ── caption -> ids (pad/truncate NT) -> 13-tap features BF16 ──
        var prompt = _read_text(stage_dir + "/prompt." + String(i) + ".txt")
        var ids = tok.encode(prompt)
        # natural_len = real token count BEFORE padding (capped at NT for truncation).
        # ai-toolkit encodes at the caption's natural length then masks/pads later;
        # we pad-to-NT before encode, so we record the real length and zero the pad
        # feature rows below (matching ai-toolkit get_qwen3_vl_features stacked*text_mask).
        var natural_len = len(ids)
        if natural_len > NT:
            natural_len = NT
        if len(ids) > NT:
            var cut = List[Int]()
            for j in range(NT):
                cut.append(ids[j])
            ids = cut^
        while len(ids) < NT:
            ids.append(PAD_ID)
        var feats = ideogram4_encode_text(tenc, ids, ctx)
        # Zero the encoder features at pad positions [natural_len, NT): build a
        # [1,NT,1] mask (1.0 real / 0.0 pad) and broadcast-multiply. This is the
        # ai-toolkit pipeline.py:156-157 `stacked * text_mask` step (the encoder
        # itself never masked — ideogram_qwen3vl.mojo). Skip when no padding.
        var feats_masked: Tensor
        if natural_len < NT:
            var mask_host = List[Float32]()
            for j in range(NT):
                if j < natural_len:
                    mask_host.append(Float32(1.0))
                else:
                    mask_host.append(Float32(0.0))
            var mask_f32 = Tensor.from_host(mask_host^, [1, NT, 1], STDtype.F32, ctx)
            # mask must match feats dtype (BF16) — mul does NOT auto-cast; 1.0/0.0 exact in bf16
            var mask = cast_tensor(mask_f32, STDtype.BF16, ctx)
            feats_masked = mul(feats, mask, ctx)
        else:
            feats_masked = feats^
        var llm = cast_tensor(feats_masked, STDtype.BF16, ctx)

        names.append(String("clean.") + String(i))
        tensors.append(TArc(clean^))
        names.append(String("llm.") + String(i))
        tensors.append(TArc(llm^))
        # text_len.<i> scalar: the natural (pre-pad) token count. The cache reader
        # threads it to ideogram4_build_packed_inputs so the DiT indicator is 0 at
        # pad positions (ai-toolkit pipeline.py:249) and pad position_ids hold at
        # real_len-1. Absent => callers default text_len=NT (all-real).
        var tl_host = List[Float32]()
        tl_host.append(Float32(natural_len))
        var tl = Tensor.from_host(tl_host^, [1], STDtype.F32, ctx)
        names.append(String("text_len.") + String(i))
        tensors.append(TArc(tl^))
        print("[prepare] sample", i, "done (ids", len(ids), "real", natural_len, ")")

    # ── T1.D caption dropout: optional uncond features ──
    # Stage A `--uncond` writes <stage_dir>/uncond.txt (the empty-caption
    # render through the SAME schema). Encode it through the SAME tokenize/
    # pad/encode path as the samples into llm_uncond [1,NT,53248] BF16;
    # Ideogram4TrainCache.uncond[NT] reads it when caption_dropout fires.
    var uncond_path = stage_dir + "/uncond.txt"
    if path_exists(uncond_path):
        var uprompt = _read_text(uncond_path)
        var uids = tok.encode(uprompt)
        var u_natural = len(uids)
        if u_natural > NT:
            u_natural = NT
        if len(uids) > NT:
            var ucut = List[Int]()
            for j in range(NT):
                ucut.append(uids[j])
            uids = ucut^
        while len(uids) < NT:
            uids.append(PAD_ID)
        var ufeats = ideogram4_encode_text(tenc, uids, ctx)
        # Zero pad feature rows for the uncond render too (same masking).
        var ufeats_masked: Tensor
        if u_natural < NT:
            var umask_host = List[Float32]()
            for j in range(NT):
                if j < u_natural:
                    umask_host.append(Float32(1.0))
                else:
                    umask_host.append(Float32(0.0))
            var umask_f32 = Tensor.from_host(umask_host^, [1, NT, 1], STDtype.F32, ctx)
            var umask = cast_tensor(umask_f32, STDtype.BF16, ctx)
            ufeats_masked = mul(ufeats, umask, ctx)
        else:
            ufeats_masked = ufeats^
        var ullm = cast_tensor(ufeats_masked, STDtype.BF16, ctx)
        names.append(String("llm_uncond"))
        tensors.append(TArc(ullm^))
        # text_len for the uncond render (read symmetrically by the dropout path).
        var utl_host = List[Float32]()
        utl_host.append(Float32(u_natural))
        var utl = Tensor.from_host(utl_host^, [1], STDtype.F32, ctx)
        names.append(String("text_len_uncond"))
        tensors.append(TArc(utl^))
        print("[prepare] llm_uncond done (ids", len(uids), "real", u_natural, ")")

    save_safetensors(names, tensors, out_path, ctx)
    print("[prepare] WROTE", out_path, "samples=", n)
