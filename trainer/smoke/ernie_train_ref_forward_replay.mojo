# Replay the real Serenity ERNIE-Image train-step dump through the Mojo forward.
#
# Mirrors smoke/chroma_train_ref_forward_replay.mojo, adapted to ERNIE's single-
# stream DiT and the resident-device forward the production driver uses
# (serenitymojo/training/train_ernie_real.mojo:729). It consumes
# parity/ernie_train_ref_step000.safetensors and compares the Mojo B=0 LoRA
# forward against Serenity's dumped trace.packed_predicted_flow (per batch sample).
#
# ERNIE pipeline (mirrors train_ernie_real.mojo predict, :702-734):
#   - trace.transformer_hidden_states [B,128,40,28] (the NOISY packed latent fed
#     to the DiT) -> img_tokens [N_IMG,128] via NCHW->NHWC pack
#     (_latent_to_img_tokens; train_ernie_real.mojo:186-192).
#   - trace.encoder_hidden_states [B,201,3072] -> txt_tokens [N_TXT,3072] (rows).
#   - trace.transformer_timestep [B] i32 -> sigma_idx (integer) feeds the shared-
#     AdaLN source (train_ernie_real.mojo:319-348 _shared_adaln_source), built
#     ONCE from the resident base + the timestep: mv (6D modvecs), f_scale,
#     f_shift. This source is the SAME for every block (ERNIE shared AdaLN).
#   - ernie_stack_lora_forward_resident_device -> pred [N_IMG,128].
#   - compare pred vs trace.packed_predicted_flow[b] (packed the SAME NCHW->NHWC
#     way so element order aligns with pred's token-major [N_IMG, out_ch]).
#
# B=0 LoRA init -> the adapter is identity, so the forward is the base-DiT flow
# (alpha-independent). This is NOT the full train parity gate; loss/backward/
# AdamW are checked by ernie_train_ref_grad_update_replay.mojo. The MAIN LOOP
# owns the final parity bar; this smoke prints cos / max_abs for re-verification.

from std.collections import List, Optional
from std.gpu.host import DeviceContext
from std.math import sqrt

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.tensor import Tensor

from serenitymojo.ops.linear import linear
from serenitymojo.ops.activations import silu
from serenitymojo.ops.embeddings import timestep_embedding_sin_first
from serenitymojo.models.dit.ernie_image import build_ernie_rope_tables

from serenity_trainer.model.ernie.weights import (
    ErnieStackBase, ErnieBlockWeights,
    load_ernie_stack_base, load_ernie_all_blocks_bf16_normf32,
)
from serenity_trainer.model.ernie.ernie_block import ErnieModVecs, ERNIE_SLOTS
from serenity_trainer.model.ernie.ernie_stack_lora import (
    ErnieLoraSet, ErnieLoraDeviceSet, build_ernie_lora_set, ernie_lora_set_to_device,
    ernie_stack_lora_forward_resident_device,
)


# ── arch (ernie_image.json; verified vs the checkpoint header) ───────────────
comptime H = 32
comptime Dh = 128
comptime D = H * Dh            # 4096
comptime F = 12288
comptime IN_CH = 128
comptime TEXT_IN = 3072
comptime OUT_CH = 128
comptime NUM_LAYERS = 36
comptime EPS = Float32(1e-06)

# ── dump shape: latent [B,128,40,28] -> N_IMG=40*28=1120; text 201 rows; B=2 ──
comptime IMG_H = 40
comptime IMG_W = 28
comptime N_IMG = IMG_H * IMG_W   # 1120
comptime N_TXT = 201             # dump's padded text length (Tmax)
comptime S = N_IMG + N_TXT       # 1321
comptime BATCH = 2

# Per-sample VALID text length (encode_text.tokens_mask true-counts = [201, 157]).
# The reference masks text PADDING keys out of attention and sets the image-token
# RoPE axis-0 position to text_lens (transformer_ernie_image.py:387, :392-400).
# For the image-token output, masking out the (Tmax - text_lens) padding KEYS is
# bit-equivalent to DROPPING them from the sequence: the softmax key-set is the
# same {N_IMG image + text_lens valid text}, the surviving text RoPE positions
# arange(text_lens) == arange(Tmax)[:text_lens], image axis-0 == text_lens, and
# the dropped padding rows only fed their own (discarded) query outputs. So we
# build each sample with its own valid text length (no [B,H,S,S] mask needed).
comptime N_TXT_S0 = 201
comptime S_S0 = N_IMG + N_TXT_S0
comptime N_TXT_S1 = 157
comptime S_S1 = N_IMG + N_TXT_S1

# ── recipe (LoRA carrier; B=0 init so forward is alpha-independent) ──────────
comptime RANK = 16
comptime ALPHA = Float32(1.0)

comptime PARITY = "trainer/parity/ernie_train_ref_step000.safetensors"
comptime CKPT = "/home/alex/models/ERNIE-Image/transformer"

comptime MIN_FORWARD_COS = Float64(0.999)


def _dump_f32(st: ShardedSafeTensors, key: String, ctx: DeviceContext) raises -> List[Float32]:
    var t = Tensor.from_view(st.tensor_view(key), ctx)
    if t.dtype() == STDtype.BF16:
        var bf = t.to_host_bf16(ctx)
        var out = List[Float32]()
        for i in range(len(bf)):
            out.append(bf[i].cast[DType.float32]())
        return out^
    if t.dtype() == STDtype.F32:
        return t.to_host(ctx)
    return cast_tensor(t, STDtype.F32, ctx).to_host(ctx)


def _slice_sample(flat: List[Float32], b: Int, per: Int) -> List[Float32]:
    var out = List[Float32]()
    var base = b * per
    for i in range(per):
        out.append(flat[base + i])
    return out^


# Read an I32 dump tensor (e.g. trace.transformer_timestep) directly from raw
# bytes — cast_tensor has no I32 compute path. Little-endian i32, host-side.
def _dump_i32(st: ShardedSafeTensors, key: String) raises -> List[Int]:
    var tv = st.tensor_view(key)
    var n = 1
    for i in range(len(tv.shape)):
        n *= tv.shape[i]
    var out = List[Int]()
    for i in range(n):
        var v = (
            Int(tv.data[i * 4 + 0])
            | (Int(tv.data[i * 4 + 1]) << 8)
            | (Int(tv.data[i * 4 + 2]) << 16)
            | (Int(tv.data[i * 4 + 3]) << 24)
        )
        out.append(v)
    return out^


# Row-sum of an I64 mask dump tensor [rows, cols] (e.g. tokens_mask). Values are
# 0/1, so the per-row sum is the valid (true) token count. Little-endian i64.
def _dump_i64_rowsum(
    st: ShardedSafeTensors, key: String, rows: Int, cols: Int
) raises -> List[Int]:
    var tv = st.tensor_view(key)
    var out = List[Int]()
    for r in range(rows):
        var s = 0
        for c in range(cols):
            var idx = r * cols + c
            var v = 0
            for byte in range(8):
                v |= Int(tv.data[idx * 8 + byte]) << (8 * byte)
            s += v
        out.append(s)
    return out^


# [128,40,28] flat (NCHW; one batch sample) -> [N_IMG,128] token-major (NHWC).
# token t (= r*W + c), channel ch -> src[ch*H*W + t]. Mirrors
# train_ernie_real.mojo:186-192 _latent_to_img_tokens; used for BOTH the DiT
# input AND the reference packing so pred/ref element order aligns.
def _pack_nchw_to_tokens(src: List[Float32]) -> List[Float32]:
    var out = List[Float32]()
    var hw = IMG_H * IMG_W
    for t in range(hw):
        for ch in range(IN_CH):
            out.append(src[ch * hw + t])
    return out^


def _ones(n: Int) -> List[Float32]:
    var o = List[Float32]()
    for _ in range(n):
        o.append(Float32(1.0))
    return o^


def _chunk(src: List[Float32], idx: Int, width: Int) -> List[Float32]:
    var o = List[Float32]()
    var off = idx * width
    for i in range(width):
        o.append(src[off + i])
    return o^


# Shared-AdaLN SOURCE (train_ernie_real.mojo:319-348). Built ONCE from the
# resident base + sigma_idx (integer timestep). Returns (mv, f_scale, f_shift).
#   c   = linear2(silu(linear1(timestep_embedding_sin_first(idx))))
#   mv  = chunk6(silu(c) @ adaln_w + adaln_b)
#   fs  = chunk2(c @ final_norm_w + final_norm_b) -> [f_scale, f_shift]
def _shared_adaln_source(
    base: ErnieStackBase, sigma_idx: Int, ctx: DeviceContext
) raises -> Tuple[ErnieModVecs, List[Float32], List[Float32]]:
    var ts = List[Float32]()
    ts.append(Float32(sigma_idx))
    var ts_t = Tensor.from_host(ts, [1], STDtype.F32, ctx)
    var emb_in = timestep_embedding_sin_first(ts_t, D, ctx, 10000.0, base.te_w1[].dtype())
    var h1 = linear(emb_in, base.te_w1[], Optional[Tensor](base.te_b1[].clone(ctx)), ctx)
    h1 = silu(h1, ctx)
    var c = linear(h1, base.te_w2[], Optional[Tensor](base.te_b2[].clone(ctx)), ctx)

    var sc = silu(c, ctx)
    var adaln = linear(sc, base.adaln_w[], Optional[Tensor](base.adaln_b[].clone(ctx)), ctx)
    var adaln_h = adaln.to_host(ctx)
    var fmod = linear(c, base.final_norm_w[], Optional[Tensor](base.final_norm_b[].clone(ctx)), ctx)
    var fmod_h = fmod.to_host(ctx)

    var mv = ErnieModVecs(
        _chunk(adaln_h, 0, D), _chunk(adaln_h, 1, D), _chunk(adaln_h, 2, D),
        _chunk(adaln_h, 3, D), _chunk(adaln_h, 4, D), _chunk(adaln_h, 5, D),
    )
    var f_scale = _chunk(fmod_h, 0, D)
    var f_shift = _chunk(fmod_h, 1, D)
    return (mv^, f_scale^, f_shift^)


def _compare(
    label: String, got: List[Float32], expected: List[Float32],
) raises -> Float64:
    if len(got) != len(expected):
        raise Error(label + String(": len mismatch got ") + String(len(got))
                    + String(" expected ") + String(len(expected)))
    var dot = Float64(0.0)
    var na = Float64(0.0)
    var nb = Float64(0.0)
    var max_abs = Float32(0.0)
    var nonfinite = 0
    for i in range(len(got)):
        var a = got[i]
        var bexp = expected[i]
        if a != a or bexp != bexp or (a - a) != Float32(0.0) or (bexp - bexp) != Float32(0.0):
            nonfinite += 1
            continue
        dot += Float64(a) * Float64(bexp)
        na += Float64(a) * Float64(a)
        nb += Float64(bexp) * Float64(bexp)
        var d = a - bexp
        var ad = d if d >= Float32(0.0) else -d
        if ad > max_abs:
            max_abs = ad
    var cos = dot / (sqrt(na) * sqrt(nb))
    print(label, "n =", len(got), "cos =", cos, "max_abs_diff =", max_abs, "nonfinite =", nonfinite)
    print(label, "got[0:3] =", got[0], got[1], got[2],
          "ref[0:3] =", expected[0], expected[1], expected[2])
    return cos


# Run ONE batch sample through the Mojo forward at its OWN valid text length.
# N_TXT_B = valid text tokens for this sample (== text_lens); S_B = N_IMG+N_TXT_B.
# The dump stores all N_TXT (=201) padded text rows; we keep the FIRST N_TXT_B
# (valid rows are left-aligned: reference valid_text = arange(Tmax) < text_lens).
def _run_sample[
    N_TXT_B: Int, S_B: Int
](
    b: Int,
    img_all: List[Float32], txt_all: List[Float32], ref_all: List[Float32],
    ts_all: List[Int],
    base: ErnieStackBase, blocks: List[ErnieBlockWeights],
    lora_dev: ErnieLoraDeviceSet,
    ctx: DeviceContext,
) raises -> Float64:
    var img_per = IN_CH * N_IMG
    var ref_per = IN_CH * N_IMG

    var img_nchw = _slice_sample(img_all, b, img_per)
    var img_tokens = _pack_nchw_to_tokens(img_nchw)

    # text: take this sample's full 201-row block, keep the first N_TXT_B rows.
    var txt_full = _slice_sample(txt_all, b, N_TXT * TEXT_IN)
    var txt_tokens = List[Float32]()
    for i in range(N_TXT_B * TEXT_IN):
        txt_tokens.append(txt_full[i])

    var ref_nchw = _slice_sample(ref_all, b, ref_per)
    var ref_tokens = _pack_nchw_to_tokens(ref_nchw)
    var sigma_idx = ts_all[b]

    var src = _shared_adaln_source(base, sigma_idx, ctx)
    var mv = src[0].copy()
    var f_scale = src[1].copy()
    var f_shift = src[2].copy()
    print("[sample", b, "] sigma_idx =", sigma_idx, " text_len =", N_TXT_B)

    # Per-sample RoPE: image-token axis-0 position == text_lens (== N_TXT_B).
    var rope = build_ernie_rope_tables[N_IMG, N_TXT_B, H, Dh](
        IMG_H, IMG_W, N_TXT_B, ctx, STDtype.F32
    )

    var fwd = ernie_stack_lora_forward_resident_device[H, Dh, N_IMG, N_TXT_B, S_B](
        img_tokens^, txt_tokens^, base, blocks, lora_dev, mv,
        f_scale.copy(), f_shift.copy(), rope[0], rope[1],
        D, F, IN_CH, TEXT_IN, OUT_CH, EPS, ctx,
    )
    return _compare(String("packed_flow[") + String(b) + String("]"),
                    fwd.out.copy(), ref_tokens)


def main() raises:
    var ctx = DeviceContext()

    print("=== Ernie train-ref forward replay ===")
    print("[parity]", PARITY)
    print("[ckpt]  ", CKPT)
    print("[shape] N_IMG =", N_IMG, "N_TXT =", N_TXT, "S =", S, "BATCH =", BATCH)
    print("[arch]  D =", D, "H =", H, "Dh =", Dh, "F =", F, "layers =", NUM_LAYERS)

    var st = ShardedSafeTensors.open(String(PARITY))
    var img_all = _dump_f32(st, String("trace.transformer_hidden_states"), ctx)  # [B,128,40,28]
    var txt_all = _dump_f32(st, String("trace.encoder_hidden_states"), ctx)      # [B,201,3072]
    var ref_all = _dump_f32(st, String("trace.packed_predicted_flow"), ctx)      # [B,128,40,28]
    var ts_all = _dump_i32(st, String("trace.transformer_timestep"))             # [B] i32
    print("[dump] img n =", len(img_all), " txt n =", len(txt_all),
          " ref n =", len(ref_all), " timesteps =", ts_all[0], ts_all[1])

    # ── resident base + 36 BF16 blocks (no offload; resident-device path) ──
    var ckpt_st = ShardedSafeTensors.open(String(CKPT))
    print("  opened transformer: num_shards =", ckpt_st.num_shards())
    var base = load_ernie_stack_base(ckpt_st, D, IN_CH, ctx)
    var blocks = load_ernie_all_blocks_bf16_normf32(ckpt_st, NUM_LAYERS, ctx)
    print("[load] base + blocks resident:", len(blocks))

    # ── per-sample valid text length (tokens_mask true-count) ──
    var text_lens = _dump_i64_rowsum(
        st, String("trace.encode_text.tokens_mask"), BATCH, 512
    )
    print("[mask] text_lens =", text_lens[0], text_lens[1])
    if text_lens[0] != N_TXT_S0 or text_lens[1] != N_TXT_S1:
        raise Error("ERNIE smoke: tokens_mask counts changed; update N_TXT_S0/S1")

    # ── LoRA carrier (B=0 init -> identity at step 0) + device staging ──
    var lora = build_ernie_lora_set(NUM_LAYERS, D, F, RANK, ALPHA)
    var lora_dev = ernie_lora_set_to_device(lora, STDtype.BF16, ctx)
    print("[lora] adapters:", NUM_LAYERS * ERNIE_SLOTS)

    # Each sample runs at its OWN valid text length (drops padding text == the
    # reference's attention mask, for the image-token output). Unrolled because
    # N_TXT/S are compile-time params.
    var cos0 = _run_sample[N_TXT_S0, S_S0](
        0, img_all, txt_all, ref_all, ts_all, base, blocks, lora_dev, ctx
    )
    var cos1 = _run_sample[N_TXT_S1, S_S1](
        1, img_all, txt_all, ref_all, ts_all, base, blocks, lora_dev, ctx
    )
    var min_cos = cos0 if cos0 < cos1 else cos1

    print("[gate] min forward cos =", min_cos, " bar =", MIN_FORWARD_COS)
    if min_cos < MIN_FORWARD_COS:
        raise Error("ERNIE forward replay below cos gate (main loop owns the bar)")
    print("ERNIE TRAIN REF FORWARD REPLAY PASS")
