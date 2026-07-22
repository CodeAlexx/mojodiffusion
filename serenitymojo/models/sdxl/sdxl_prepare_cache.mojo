# Product-owned SDXL training-cache producer (P6 wave-1, 2026-07-22).
#
# Input is the shared image/caption stage (sample_NNNNN.safetensors with key
# "image" [1,3,512,512] F32 in [-1,1] + sample_NNNNN.txt caption). Output is one
# atomic safetensors record per sample containing every frozen input the SDXL
# trainer (serenity-trainer train_sdxl_real.mojo) reads:
#
#   latent         [1,4,64,64]   BF16  (SDXL VAE encode_mean * 0.13025 scaling)
#   text_embedding [1,77,2048]   BF16  (CLIP-L penult [768] | CLIP-G penult [1280])
#   pooled         [1,1280]      BF16  (CLIP-G pooled @ text_projection^T)
#   time_ids       [1,6]         F32   ([512,512,0,0,512,512] orig/crop/target)
#
# Conditioning mirrors serve/sdxl_backend.mojo _encode_one EXACTLY (MJ-1061
# penultimate/diffusers convention, encode_sdxl[77](ids, ctx, True)); the VAE is
# the shared LdmVaeEncoder SDXL factory (models/vae/ldm_encoder.mojo) with the
# diffusers post-encode scaling z = 0.13025 * z applied here (the encoder
# returns the raw distribution mean).
#
# Every model asset is an explicit argument. The caller owns path resolution;
# this executable contains no workstation-specific fallback.

from std.collections import List, Optional
from std.gpu.host import DeviceContext
from std.memory import ArcPointer
from std.os import makedirs
from std.sys import argv

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.linear import linear
from serenitymojo.ops.tensor_algebra import concat, mul_scalar
from serenitymojo.tokenizer.clip_tokenizer import ClipTokenizer
from serenitymojo.models.text_encoder.clip_encoder import ClipEncoder, ClipConfig
from serenitymojo.models.vae.ldm_encoder import load_sdxl_ldm_encoder

comptime IH = 512
comptime IW = 512
comptime LH = 64
comptime LW = 64
comptime CLIP_LEN = 77
comptime CLIP_PAD_ID = 49407   # CLIP eos == pad
comptime CLIP_EOS_ID = 49407
comptime CLIP_G_TEXT_PROJ = "text_projection.weight"
comptime SDXL_LATENT_SCALE = Float32(0.13025)
comptime TArc = ArcPointer[Tensor]


def _sample_name(i: Int) -> String:
    var value = String(i)
    var padding = String("")
    for _ in range(5 - value.byte_length() if value.byte_length() < 5 else 0):
        padding += String("0")
    return String("sample_") + padding + value


def _read_text(path: String) raises -> String:
    var file = open(path, "r")
    var value = file.read()
    file.close()
    return value^


# Pad/truncate CLIP ids to 77 keeping a real EOS at the tail (HF CLIP pad==eos).
# Verbatim serve/sdxl_backend.mojo _fit_clip_ids.
def _fit_clip_ids(var ids: List[Int]) -> List[Int]:
    if len(ids) > CLIP_LEN:
        var trimmed = List[Int]()
        for i in range(CLIP_LEN):
            trimmed.append(ids[i])
        trimmed[CLIP_LEN - 1] = CLIP_EOS_ID
        return trimmed^
    while len(ids) < CLIP_LEN:
        ids.append(CLIP_PAD_ID)
    return ids^


def main() raises:
    var args = argv()
    if len(args) != 9:
        raise Error(
            "usage: sdxl_prepare_cache <stage_dir> <out_dir> <n>"
            " <sdxl_vae.safetensors> <clip_l.safetensors> <clip_g.safetensors>"
            " <clip_l.tokenizer.json> <clip_g.tokenizer.json>"
        )

    var stage_dir = String(args[1])
    var out_dir = String(args[2])
    var count = Int(String(args[3]))
    var vae_path = String(args[4])
    var clip_l_path = String(args[5])
    var clip_g_path = String(args[6])
    var clip_l_tok_path = String(args[7])
    var clip_g_tok_path = String(args[8])
    if count <= 0:
        raise Error("sdxl_prepare_cache: n must be positive")

    makedirs(out_dir, exist_ok=True)
    var ctx = DeviceContext()

    var clip_l = ClipEncoder.load(clip_l_path, ClipConfig.clip_l(), ctx)
    var clip_g = ClipEncoder.load(clip_g_path, ClipConfig.clip_g(), ctx)
    # text_projection.weight lives OUTSIDE text_model.* so ClipEncoder.load
    # skips it; load it directly from the CLIP-G safetensors ([1280,1280]).
    var g_st = ShardedSafeTensors.open(clip_g_path)
    var text_proj = Tensor.from_view(g_st.tensor_view(String(CLIP_G_TEXT_PROJ)), ctx)
    var clip_l_tok = ClipTokenizer(clip_l_tok_path)
    var clip_g_tok = ClipTokenizer(clip_g_tok_path)
    var vae = load_sdxl_ldm_encoder[LH, LW](vae_path, ctx)

    # SDXL micro-conditioning for a 512x512 native/uncropped stage:
    # (orig_h, orig_w, crop_top, crop_left, target_h, target_w).
    var tid_host = List[Float32]()
    tid_host.append(Float32(IH))
    tid_host.append(Float32(IW))
    tid_host.append(Float32(0.0))
    tid_host.append(Float32(0.0))
    tid_host.append(Float32(IH))
    tid_host.append(Float32(IW))

    for i in range(count):
        var stem = _sample_name(i)
        var staged_path = stage_dir + String("/") + stem + String(".safetensors")
        var caption_path = stage_dir + String("/") + stem + String(".txt")
        var staged = ShardedSafeTensors.open(staged_path)
        var image_info = staged.tensor_info(String("image"))
        if (
            len(image_info.shape) != 4 or image_info.shape[0] != 1
            or image_info.shape[1] != 3 or image_info.shape[2] != IH
            or image_info.shape[3] != IW
        ):
            raise Error(
                String("sdxl_prepare_cache: ") + staged_path
                + String(" image must be [1,3,512,512]")
            )

        var image = Tensor.from_view(staged.tensor_view(String("image")), ctx)
        var latent_raw = vae.encode_mean(image, ctx)          # [1,4,64,64] raw mean
        var latent_f32 = mul_scalar(latent_raw, SDXL_LATENT_SCALE, ctx)
        var latent = cast_tensor(latent_f32, STDtype.BF16, ctx)

        var caption = _read_text(caption_path)
        var l_ids = _fit_clip_ids(clip_l_tok.encode(caption))
        var g_ids = _fit_clip_ids(clip_g_tok.encode(caption))
        # (hidden [1,77,H], pooled [1,H]); True = penultimate/diffusers (MJ-1061)
        var l_out = clip_l.encode_sdxl[CLIP_LEN](l_ids^, ctx, True)
        var g_out = clip_g.encode_sdxl[CLIP_LEN](g_ids^, ctx, True)
        var context = concat(2, ctx, l_out[0], g_out[0])      # [1,77,2048]
        var context_shape = context.shape()
        if (
            len(context_shape) != 3 or context_shape[0] != 1
            or context_shape[1] != CLIP_LEN or context_shape[2] != 2048
        ):
            raise Error("sdxl_prepare_cache: context shape contract failed")
        var text_embedding = cast_tensor(context, STDtype.BF16, ctx)
        # clip_g_text_embeds = clip_g_pool_raw @ text_projection^T -> [1,1280]
        var pooled_f = linear(g_out[1], text_proj, Optional[Tensor](None), ctx)
        var pooled = cast_tensor(pooled_f, STDtype.BF16, ctx)

        var tid_shape = List[Int]()
        tid_shape.append(1)
        tid_shape.append(6)
        var time_ids = Tensor.from_host(tid_host.copy(), tid_shape^, STDtype.F32, ctx)

        var names = List[String]()
        names.append(String("latent"))
        names.append(String("text_embedding"))
        names.append(String("pooled"))
        names.append(String("time_ids"))
        var tensors = List[TArc]()
        tensors.append(TArc(latent^))
        tensors.append(TArc(text_embedding^))
        tensors.append(TArc(pooled^))
        tensors.append(TArc(time_ids^))
        var out_path = out_dir + String("/") + stem + String(".safetensors")
        save_safetensors(names, tensors, out_path, ctx)
        print("[sdxl-prepare] wrote", out_path)

    print("[sdxl-prepare] complete samples=", count, " cache=", out_dir)
