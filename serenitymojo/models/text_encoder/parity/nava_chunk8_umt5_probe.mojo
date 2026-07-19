# NAVA chunk 8: umt5-xxl text encoder parity probe.
#
# Loads umt5_xxl_enc.safetensors, reads the 42 token ids from
# nava_fx_stage0.safetensors, encodes them, and gates the output
# cosine similarity against the `in_text` [42,4096] F32 reference.
#
# Gate: cos >= 0.999.
#
# Fixture keys used from nava_fx_stage0.safetensors:
#   umt5_ids  [42]       int32  — the 42 token ids
#   in_text   [42,4096]  F32    — oracle encoder output
#
# Token ids are read by staging the I32 view bytes into a host buffer and
# bitcasting to Int32, then collected into List[Int] for encode().

from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.parity import ParityHarness
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.tensor_algebra import reshape
from serenitymojo.models.text_encoder.umt5_encoder import Umt5Encoder, Umt5Config

comptime UMT5_PATH = "/home/alex/.serenity/models/checkpoints/NAVA/umt5_xxl_enc.safetensors"
comptime FX_PATH   = "/home/alex/mojodiffusion/serenitymojo/models/dit/parity/nava_fx_stage0.safetensors"

# S=42: NAVA passes exactly 42 valid token ids to the text encoder.
comptime S = 42


def main() raises:
    var ctx = DeviceContext()
    print("=== NAVA chunk8: umt5-xxl text encoder parity ===")

    # ── Load fixture file ─────────────────────────────────────────────────────
    var fx = ShardedSafeTensors.open(FX_PATH)

    # Read umt5_ids [42] int32 from fixture.
    # Strategy: get the tensor_view (dtype=I32, shape=[42]), stage its raw bytes
    # into a pinned host buffer, bitcast to Int32, collect into List[Int].
    var ids_view = fx.tensor_view(String("umt5_ids"))
    var n_ids = ids_view.numel()  # 42
    var ids_nbytes = ids_view.nbytes()  # 42 * 4 = 168 bytes
    # Stage bytes into a pinned host buffer so we can safely bitcast.
    var ids_host = ctx.enqueue_create_host_buffer[DType.uint8](ids_nbytes)
    var ids_hp = ids_host.unsafe_ptr()
    for bi in range(ids_nbytes):
        ids_hp[bi] = ids_view.data[bi]
    # Bitcast to Int32 and collect.
    var ids_i32_ptr = ids_host.unsafe_ptr().bitcast[Int32]()
    var token_ids = List[Int]()
    for i in range(n_ids):
        token_ids.append(Int(ids_i32_ptr[i]))
    print("Loaded", n_ids, "token ids; first=", token_ids[0], "last=", token_ids[n_ids - 1])

    # Load in_text [42,4096] F32 reference (as a flat list for ParityHarness).
    var in_text_tensor = Tensor.from_view(fx.tensor_view(String("in_text")), ctx)  # [42,4096] F32
    var in_text_host = in_text_tensor.to_host(ctx)  # List[Float32], 42*4096 elems
    print("Reference in_text loaded: numel=", len(in_text_host))

    # ── Load umt5 encoder weights ─────────────────────────────────────────────
    print("Loading umt5-xxl weights (242 keys, ~11GB BF16) ...")
    var config = Umt5Config.umt5_xxl()
    var encoder = Umt5Encoder[S].load(UMT5_PATH, config, ctx)
    print("Weights loaded.")

    # ── Encode ────────────────────────────────────────────────────────────────
    print("Encoding 42 tokens through 24 umt5 layers ...")
    var out = encoder.encode(token_ids, ctx)  # [1, 42, 4096] BF16

    # Cast output to F32 for parity comparison.
    var out_f32 = cast_tensor(out, STDtype.F32, ctx)   # [1, 42, 4096]
    # Reshape to [42, 4096] so element count matches reference (42*4096).
    var flat_sh = List[Int]()
    flat_sh.append(S)
    flat_sh.append(config.d_model)
    var out_flat = reshape(out_f32, flat_sh^, ctx)     # [42, 4096]

    # ── Parity gate ───────────────────────────────────────────────────────────
    var harness = ParityHarness(0.999)
    var result = harness.compare(out_flat, in_text_host, ctx)
    print("umt5 encode vs torch in_text:", result)

    if result.passed:
        print("=== GATE PASS: cos >= 0.999 ===")
        print("  cos=", result.cos, "max_abs=", result.max_abs, "n=", result.n)
    else:
        print("=== GATE FAIL ===")
        print("  cos=", result.cos, "max_abs=", result.max_abs, "n=", result.n)
        print("  Hypothesis: check per-layer bias permute [2,0,1], or gate/fc1 swap in FFN.")
