# klein9b_encode_smoke.mojo - Klein 9B caption-encode step (separate process).
#
# Loads Qwen3-8B, runs encode_klein for the positive + negative prompts, writes
# both [1,512,12288] embeddings to disk via io/ffi (cap_cache.save_tensor_bin),
# then main() returns and the process exits. Process death frees ALL ~16 GB of
# encoder GPU memory — the hardest possible separation from the Klein 9B DiT,
# which runs in a SEPARATE process (klein9b_pipeline_multistep_smoke.mojo) that
# never imports Qwen3Encoder.
#
# The prompts / template / tokenization MUST match the denoise smoke so the
# cached embeddings are the exact numbers the pre-split in-process run produced.

from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.tokenizer.tokenizer import Qwen3Tokenizer
from serenitymojo.models.text_encoder.qwen3_encoder import Qwen3Encoder, Qwen3Config
from serenitymojo.io.cap_cache import save_tensor_bin


comptime QWEN8_DIR = (
    "/home/alex/.cache/huggingface/hub/models--Qwen--Qwen3-8B/"
    "snapshots/b968826d9c46dd6066d109eabc6255188de91218"
)
comptime TOK_JSON = QWEN8_DIR + "/tokenizer.json"
comptime CAPS_POS = "/home/alex/mojodiffusion/output/klein9b_caps_pos.bin"
comptime CAPS_NEG = "/home/alex/mojodiffusion/output/klein9b_caps_neg.bin"
comptime PAD_ID = 151643
comptime SEQ = 512

# Must match klein9b_pipeline_multistep_smoke.mojo exactly.
comptime PROMPT = (
    "Illustrate a surrealist, macro-level close-up of a human eye with black "
    "lashes and blue iris, a perfectly formed honeycomb pattern fused into the "
    "iris, each chamber subtly pulsing with golden illumination. The pupil rests "
    "at the center of this hive tunnel, a dark portal surrounded by symmetry and "
    "structure, suggesting balance between organic instinct and intelligent "
    "design. The skin surrounding the eye is covered in a biomimetic mesh "
    "resembling beehive architecture across the brow and upper cheek, blending "
    "seamlessly with natural skin tones. A single honeybee lands delicately on "
    "the lower eyelid, tiny wings catching ambient light, fine legs touching the "
    "boundary between hive eye and human skin as though delivering a secret. "
    "The bee is captured in perfect anatomical realism, its eyes reflecting the "
    "honeycomb pattern. The eyelashes curve like antennae, slightly iridescent "
    "at the tips. Dramatic intimate lighting, wet iris highlights, subtle lens "
    "flare in the reflection of the bee's eye, union of nature and consciousness, "
    "reverence for bees, vision, and hidden geometries of nature woven into "
    "human form, photorealistic, ultra detailed, natural skin texture, realistic "
    "lighting, soft shadows, cinematic lighting, 50mm lens, shallow depth of "
    "field, high dynamic range, subsurface scattering, film grain, RAW photo, "
    "fine detail, professional photography"
)
comptime NEGATIVE = "low quality, blurry, watermark, jpeg artifacts"


def _klein_template(prompt: String) -> String:
    return (
        String("<|im_start|>user\n")
        + prompt
        + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def _tokenize_512(tok: Qwen3Tokenizer, prompt: String) raises -> List[Int]:
    var ids_full = tok.encode(_klein_template(prompt))
    if len(ids_full) > SEQ:
        raise Error("Klein prompt too long for 512 tokens")
    var ids = List[Int](capacity=SEQ)
    for i in range(len(ids_full)):
        ids.append(ids_full[i])
    for _ in range(SEQ - len(ids_full)):
        ids.append(PAD_ID)
    print("  tokens:", len(ids_full), "->", SEQ)
    return ids^


def main() raises:
    var ctx = DeviceContext()
    print("=== Klein 9B caption encode (separate process) ===")
    print("[text] Qwen3-8B Klein conditioning")
    var tok = Qwen3Tokenizer(TOK_JSON)
    var pos_ids = _tokenize_512(tok, PROMPT)
    var neg_ids = _tokenize_512(tok, NEGATIVE)
    var enc = Qwen3Encoder.load(QWEN8_DIR, Qwen3Config.klein_9b(), ctx)
    var pos = enc.encode_klein(pos_ids, ctx)
    var neg = enc.encode_klein(neg_ids, ctx)

    var ps = pos.shape()
    var ns = neg.shape()
    print(
        "  pos shape:", ps[0], ps[1], ps[2],
        "dtype.tag", pos.dtype().tag, "nbytes", pos.nbytes(),
    )
    print(
        "  neg shape:", ns[0], ns[1], ns[2],
        "dtype.tag", neg.dtype().tag, "nbytes", neg.nbytes(),
    )

    save_tensor_bin(pos, CAPS_POS, ctx)
    save_tensor_bin(neg, CAPS_NEG, ctx)
    print("[cache] wrote", CAPS_POS)
    print("[cache] wrote", CAPS_NEG)
    print("[done] encoder GPU memory freed on process exit")
