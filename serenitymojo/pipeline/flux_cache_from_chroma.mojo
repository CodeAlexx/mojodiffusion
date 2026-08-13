# pipeline/flux_cache_from_chroma.mojo — derive the FLUX.1 boxjana training
# cache from the on-box CHROMA boxjana cache + REAL Mojo CLIP-L caption encodes.
#
# WHY THIS IS VALID (the flux full-FT rollout, phase (c) data gap):
#   * latent: chroma's VAE IS the flux VAE (ae.safetensors — chroma is
#     flux-derived; train_chroma_real.mojo:187 decodes with the FLUX ae).
#     BOTH trainers store the latent RAW and apply the IDENTICAL shift/scale
#     at train time: chroma VAE_SHIFT=0.1159 VAE_SCALE=0.3611
#     (train_chroma_real.mojo:158-159, applied :586) == flux VAE_SHIFT/SCALE
#     (train_flux_real.mojo:156-157, applied :583-584). The chroma cache
#     latent [1,16,64,64] F32 is therefore byte-reusable as a flux cache
#     latent (flux_prepare.mojo schema line 7: RAW, no shift/scale, no pack).
#   * t5_embed: chroma reuses flux's T5-XXL text stack (train_chroma_real
#     TXT_CH=4096 == train_flux_real TXT_CH=4096); both trainers pad/trim
#     the cached [1,seq,4096] to N_TXT=512 rows at train time
#     (train_flux_real.mojo:568-580 == train_chroma_real.mojo:570-582), so
#     the variable-seq chroma t5_embed is byte-reusable.
#   * clip_pool [1,768]: chroma has NO CLIP conditioning, so this is built
#     HERE with the REAL pure-Mojo CLIP-L encoder (models/text_encoder/
#     clip_encoder.mojo, pooled gated cos 1.000 vs HF) + the REAL pure-Mojo
#     CLIP BPE tokenizer (tokenizer/clip_tokenizer.mojo, bit-exact vs HF),
#     fed the REAL per-image boxjana captions. NO constant/zero conditioning.
#
# CAPTION MAPPING: the chroma cache filenames are md5(<abs image path>) of the
# /home/alex/datasets/boxjana images; every one of the 22 cache samples maps to
# an <n>.jpg with an <n>.txt caption (verified 22/22). The mapping is staged as
# a TSV manifest ("<cache-file>\t<caption>") because Mojo has no md5 in-tree:
#   python3: md5(path).hexdigest()+".safetensors" -> caption text
#   -> output/cache/boxjana_flux_512_captions.tsv
#
# OUTPUT (flux_prepare.mojo sample schema, one safetensors per sample):
#   latent    F32 [1,16,64,64]   (copied verbatim from the chroma cache)
#   t5_embed  F32 [1,seq,4096]   (copied verbatim from the chroma cache)
#   clip_pool F32 [1,768]        (REAL Mojo CLIP-L pooled caption encode)
#
# CLIP-L assets (staged 2026-07-08): weights /mnt/disk1/models/checkpoints/
# flux1-aux/text_encoder/model.safetensors (196 text_model.* keys, BF16);
# tokenizer.json from Serenity's text-encoder assets
# (openai/clip-vit-large-patch14 BPE, vocab 49408,
# end_of_word_suffix "</w>").
#
# Run:
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#     pixi run mojo build -I . -Xlinker -lm -Xlinker -lcuda \
#       serenitymojo/pipeline/flux_cache_from_chroma.mojo -o /tmp/flux_cache_from_chroma \
#     && /tmp/flux_cache_from_chroma
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.collections import List
from max.gpu.host import DeviceContext
from std.memory import ArcPointer
from std.os import makedirs

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.io.ffi import (
    BytePtr, sys_open, sys_close, sys_pread, O_RDONLY,
)
from std.memory import alloc
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.models.text_encoder.clip_encoder import ClipEncoder, ClipConfig
from serenitymojo.tokenizer.clip_tokenizer import ClipTokenizer


comptime SRC_DIR = "/home/alex/datasets/boxjana_chroma_edv2_512"
comptime OUT_DIR = "/home/alex/mojodiffusion/output/cache/boxjana_flux_512"
comptime MANIFEST = "/home/alex/mojodiffusion/output/cache/boxjana_flux_512_captions.tsv"
comptime CLIP_PATH = "/mnt/disk1/models/checkpoints/flux1-aux/text_encoder/model.safetensors"
comptime CLIP_TOK_JSON = "/home/alex/.serenity/models/text_encoders/clip_l.tokenizer.json"

comptime CLIP_LEN = 77
comptime CLIP_BOS = 49406
comptime CLIP_EOS = 49407
comptime VEC_DIM = 768
comptime LAT_C = 16
comptime LAT_H = 64
comptime LAT_W = 64
comptime TXT_CH = 4096


def _read_file_bytes(path: String) raises -> List[UInt8]:
    var fd = sys_open(path, O_RDONLY, Int32(0))
    if fd < 0:
        raise Error(String("flux_cache_from_chroma: cannot open ") + path)
    var out = List[UInt8]()
    comptime CHUNK = 65536
    var buf = alloc[UInt8](CHUNK)
    var offset = 0
    while True:
        var n = sys_pread(fd, BytePtr(unsafe_from_address=Int(buf)), CHUNK, offset)
        if n < 0:
            buf.free()
            _ = sys_close(fd)
            raise Error("flux_cache_from_chroma: read error")
        if n == 0:
            break
        for i in range(n):
            out.append(buf[i])
        offset += n
        if n < CHUNK:
            break
    buf.free()
    _ = sys_close(fd)
    return out^


# manifest line format: "<cache-file>\t<caption>\n"
struct _ManifestRow(Copyable, Movable):
    var file: String
    var caption: String

    def __init__(out self, var file: String, var caption: String):
        self.file = file^
        self.caption = caption^


def _read_manifest(path: String) raises -> List[_ManifestRow]:
    var bytes = _read_file_bytes(path)
    var rows = List[_ManifestRow]()
    var field = String("")
    var file = String("")
    var in_caption = False
    for i in range(len(bytes)):
        var b = bytes[i]
        if b == 0x09 and not in_caption:      # TAB
            file = field.copy()
            field = String("")
            in_caption = True
        elif b == 0x0A:                        # NEWLINE
            if in_caption and file.byte_length() > 0:
                rows.append(_ManifestRow(file.copy(), field.copy()))
            field = String("")
            file = String("")
            in_caption = False
        else:
            # captions are verified pure-ASCII (manifest generation step); a
            # per-byte chr() append is exact for ASCII (the in-repo
            # clip_tokenizer's own byte->String pattern).
            if b > 0x7F:
                raise Error("flux_cache_from_chroma: non-ASCII caption byte (manifest contract)")
            field += String(chr(Int(b)))
    if in_caption and file.byte_length() > 0 and field.byte_length() > 0:
        rows.append(_ManifestRow(file^, field^))
    if len(rows) == 0:
        raise Error(String("flux_cache_from_chroma: empty caption manifest ") + path)
    return rows^


def _load_named(st: SafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    var info = st.tensor_info(name)
    var bytes = st.tensor_bytes(name)
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    return Tensor.from_view(tv, ctx)


# pad/truncate CLIP ids to 77 keeping EOS at the tail (flux_sample_cli._fit).
def _fit_clip(var ids: List[Int]) -> List[Int]:
    var out = List[Int]()
    if len(ids) >= CLIP_LEN:
        for i in range(CLIP_LEN - 1):
            out.append(ids[i])
        out.append(CLIP_EOS)
    else:
        for i in range(len(ids)):
            out.append(ids[i])
        while len(out) < CLIP_LEN:
            out.append(CLIP_EOS)
    return out^


def _std_host(v: List[Float32]) -> Float32:
    var n = len(v)
    var s = 0.0
    var s2 = 0.0
    for i in range(n):
        var x = Float64(v[i])
        s += x
        s2 += x * x
    var m = s / Float64(n)
    var vv = s2 / Float64(n) - m * m
    if vv < 0.0:
        vv = 0.0
    from std.math import sqrt
    return Float32(sqrt(vv))


def main() raises:
    var ctx = DeviceContext()
    print("=== flux boxjana cache derive: chroma latent/t5 + Mojo CLIP-L clip_pool ===")
    print("  src  :", SRC_DIR, "(chroma cache: latent F32 [1,16,64,64] + t5_embed F32 [1,seq,4096])")
    print("  out  :", OUT_DIR, "(flux_prepare schema: + clip_pool [1,768])")
    print("  clip :", CLIP_PATH)
    makedirs(String(OUT_DIR), exist_ok=True)

    var rows = _read_manifest(String(MANIFEST))
    print("[manifest]", len(rows), "caption-mapped samples")

    print("[load] ClipTokenizer:", CLIP_TOK_JSON)
    var tok = ClipTokenizer(String(CLIP_TOK_JSON))
    print("[load] ClipEncoder (CLIP-L):", CLIP_PATH)
    var clip = ClipEncoder.load(String(CLIP_PATH), ClipConfig.clip_l(), ctx)

    for ri in range(len(rows)):
        ref row = rows[ri]
        var src_path = String(SRC_DIR) + String("/") + row.file
        var st = SafeTensors.open(src_path)

        # latent + t5_embed copied VERBATIM (shape/dtype-checked, fail-loud).
        var latent = _load_named(st, String("latent"), ctx)
        var lsh = latent.shape()
        if len(lsh) != 4 or lsh[1] != LAT_C or lsh[2] != LAT_H or lsh[3] != LAT_W:
            raise Error(String("bad latent shape in ") + row.file)
        if latent.dtype() != STDtype.F32:
            raise Error(String("latent not F32 in ") + row.file)
        var t5 = _load_named(st, String("t5_embed"), ctx)
        var tsh = t5.shape()
        if len(tsh) != 3 or tsh[0] != 1 or tsh[2] != TXT_CH:
            raise Error(String("bad t5_embed shape in ") + row.file)

        # REAL caption -> CLIP BPE ids -> CLIP-L pooled [1,768].
        var ids = _fit_clip(tok.encode(row.caption))
        var enc = clip.encode_sdxl[CLIP_LEN](ids^, ctx)
        var pooled_f32 = cast_tensor(enc[1], STDtype.F32, ctx)   # [1,768]
        var psh = pooled_f32.shape()
        if len(psh) != 2 or psh[0] != 1 or psh[1] != VEC_DIM:
            raise Error(String("bad clip_pool shape for ") + row.file)
        var pool_host = pooled_f32.to_host(ctx)
        var pstd = _std_host(pool_host)
        if pstd <= Float32(0.0):
            raise Error(
                String("degenerate clip_pool (std=0) for ") + row.file
                + String(" — constant conditioning is forbidden")
            )

        var names = List[String]()
        names.append(String("latent"))
        names.append(String("t5_embed"))
        names.append(String("clip_pool"))
        var tensors = List[ArcPointer[Tensor]]()
        tensors.append(ArcPointer[Tensor](latent^))
        tensors.append(ArcPointer[Tensor](t5^))
        tensors.append(ArcPointer[Tensor](pooled_f32^))
        var out_path = String(OUT_DIR) + String("/") + row.file
        save_safetensors(names, tensors, out_path, ctx)
        print(
            "  [", ri + 1, "/", len(rows), "]", row.file,
            " t5_seq=", tsh[1], " clip_pool std=", pstd,
        )

    print("")
    print("PASS: wrote", len(rows), "flux cache samples to", OUT_DIR)
    print("      latent/t5_embed = chroma boxjana cache VERBATIM (shared flux VAE+T5)")
    print("      clip_pool = REAL Mojo CLIP-L encode of the per-image boxjana caption")
