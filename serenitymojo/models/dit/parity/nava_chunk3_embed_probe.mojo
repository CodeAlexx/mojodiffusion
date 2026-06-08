# NAVA chunk 3: pre-embedding parity probe.
# Runs all 4 sub-paths (video patch, audio patch, text embed, time embed)
# against fixtures from nava_fx_stage0.safetensors (inputs) and
# nava_fx_chunk5_block0.safetensors (targets). Gate: cos >= 0.999.
#
# Fixture input keys (nava_fx_stage0.safetensors, all F32):
#   in_lat_vid [1280,48], in_lat_aud [34,128], in_text [42,4096], in_t [1]
#
# Fixture target keys (nava_fx_chunk5_block0.safetensors, all F32):
#   x [1,354,3072]          → x[:,:320,:] is x_vid target, x[:,320:,:] is x_audio
#   context [1,512,3072]    → text embed target
#   e_vid [1,320,6,3072]    → time embed target (broadcast; compare token 0)
#
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.parity import ParityHarness
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.tensor_algebra import reshape, slice, concat
from serenitymojo.models.dit.nava_embed import (
    load_nava_embed_weights,
    nava_video_patch_embed,
    nava_audio_patch_embed,
    nava_text_embed,
    nava_time_embed,
)

comptime NAVA = "/home/alex/.serenity/models/checkpoints/NAVA/NAVA_fp8.safetensors"
comptime FX0  = "/home/alex/mojodiffusion/serenitymojo/models/dit/parity/nava_fx_stage0.safetensors"
comptime FX5  = "/home/alex/mojodiffusion/serenitymojo/models/dit/parity/nava_fx_chunk5_block0.safetensors"


def main() raises:
    var ctx = DeviceContext()
    print("=== NAVA chunk3 embed parity ===")

    # ── Load input fixtures (F32 → cast to BF16 for latents/text; in_t stays F32) ──
    var fx0 = ShardedSafeTensors.open(FX0)

    var in_lat_vid_f32 = Tensor.from_view(fx0.tensor_view("in_lat_vid"), ctx)  # [1280,48]
    var in_lat_aud_f32 = Tensor.from_view(fx0.tensor_view("in_lat_aud"), ctx)  # [34,128]
    var in_text_f32    = Tensor.from_view(fx0.tensor_view("in_text"),    ctx)  # [42,4096]
    var in_t           = Tensor.from_view(fx0.tensor_view("in_t"),       ctx)  # [1] F32

    var in_lat_vid = cast_tensor(in_lat_vid_f32, STDtype.BF16, ctx)  # [1280,48] BF16
    var in_lat_aud = cast_tensor(in_lat_aud_f32, STDtype.BF16, ctx)  # [34,128]  BF16
    var in_text    = cast_tensor(in_text_f32,    STDtype.BF16, ctx)  # [42,4096] BF16

    # ── Load target fixtures (F32) ──────────────────────────────────────────────
    var fx5 = ShardedSafeTensors.open(FX5)

    # x [1,354,3072]: slice dim-1 to get vid [1,320,3072] and audio [1,34,3072]
    var x_all_f32 = Tensor.from_view(fx5.tensor_view("x"), ctx)  # [1,354,3072]

    # Targets kept as F32 on host for comparison
    var x_vid_ref_host   = slice(x_all_f32, 1,   0, 320, ctx).to_host(ctx)  # [1,320,3072]
    var x_aud_ref_host   = slice(x_all_f32, 1, 320,  34, ctx).to_host(ctx)  # [1,34,3072]
    var ctx_ref_host     = Tensor.from_view(fx5.tensor_view("context"), ctx).to_host(ctx)  # [1,512,3072]

    # e_vid [1,320,6,3072]: slice dim-1 to get token 0 → [1,1,6,3072]
    var e_vid_all_f32    = Tensor.from_view(fx5.tensor_view("e_vid"), ctx)  # [1,320,6,3072]
    var e_vid_tok0_host  = slice(e_vid_all_f32, 1, 0, 1, ctx).to_host(ctx)  # [1,1,6,3072]

    # ── Load weights ─────────────────────────────────────────────────────────────
    var st = ShardedSafeTensors.open(NAVA)
    var w  = load_nava_embed_weights(st, "backbone.", ctx)

    # ── (A) Video patch-embed ────────────────────────────────────────────────────
    var x_vid  = nava_video_patch_embed(in_lat_vid, w, ctx)   # [1,320,3072] BF16
    var x_vid_f32 = cast_tensor(x_vid, STDtype.F32, ctx)

    var harness  = ParityHarness(0.999)
    var res_vid  = harness.compare(x_vid_f32, x_vid_ref_host, ctx)
    print("(A) video patch-embed vs torch:", res_vid)

    # ── (B) Audio patch-embed ────────────────────────────────────────────────────
    var x_audio  = nava_audio_patch_embed(in_lat_aud, w, ctx)  # [1,34,3072] BF16
    var x_aud_f32 = cast_tensor(x_audio, STDtype.F32, ctx)

    var res_aud  = harness.compare(x_aud_f32, x_aud_ref_host, ctx)
    print("(B) audio patch-embed vs torch:", res_aud)

    # ── (C) Text embed ───────────────────────────────────────────────────────────
    var context  = nava_text_embed(in_text, w, ctx)   # [1,512,3072] BF16
    var ctx_f32  = cast_tensor(context, STDtype.F32, ctx)

    var res_ctx  = harness.compare(ctx_f32, ctx_ref_host, ctx)
    print("(C) text embed vs torch:", res_ctx)

    # ── (D) Time embed ───────────────────────────────────────────────────────────
    # e0 is [1,1,6,3072]; compare against first token of e_vid fixture
    var e0      = nava_time_embed(in_t, w, ctx)  # [1,1,6,3072] BF16
    var e0_f32  = cast_tensor(e0, STDtype.F32, ctx)

    var res_t   = harness.compare(e0_f32, e_vid_tok0_host, ctx)
    print("(D) time embed vs torch:", res_t)
