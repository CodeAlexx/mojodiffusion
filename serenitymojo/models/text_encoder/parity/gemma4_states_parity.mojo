# Real-weight parity gate: gemma4_ltx_streamed.mojo vs the GPU-bf16 HF oracle.
#
# Oracle: `gemma4_oracle_dump.py` (transformers 5.15.0 `gemma4_unified`, bf16,
# GPU math via accelerate cpu_offload). It runs the FULL 1024 window with the
# real tokens LEFT-padded into the tail, so the real token j sits at absolute
# position (1024 - real_len) + j.
#
# The Mojo encoder instead packs the real tokens at the FRONT of a compact
# 128-step bucket and starts RoPE at offset (1024 - real_len), which reproduces
# exactly those absolute positions while the causal pad-mask keeps the padding
# out of attention. The two therefore line up as:
#
#     oracle[0, 1024 - real_len : 1024, :]   ==   mojo[0, 0 : real_len, :]
#
# Only that valid prefix is compared; the bucket's trailing pad rows are not
# part of the contract (they are masked out of every downstream consumer).
#
# Build (the cuDNN SDPA shim MUST be linked — a bare `mojo run -I .` dies at
# JIT time with "Symbols not found: [ flame_cudnn_sdpa_bf16_train_fwd ]"):
#
#   pixi run mojo build -O0 -j 1 -I . -I vendor/mojo-libs \
#     -Xlinker -lm -Xlinker -lcuda \
#     -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa \
#     -Xlinker -Lserenitymojo/ops/cshim/lib/cudnn_stubs -Xlinker -lcudnn \
#     -Xlinker -rpath -Xlinker '$ORIGIN/../../serenitymojo/ops/cshim/lib' \
#     -Xlinker -rpath -Xlinker '$ORIGIN/../../.pixi/envs/default/lib' \
#     serenitymojo/models/text_encoder/parity/gemma4_states_parity.mojo \
#     -o output/checks/gemma4_states_parity

from max.gpu.host import DeviceContext

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.models.text_encoder.gemma4_ltx_streamed import (
    GEMMA4_HIDDEN,
    GEMMA4_LAYERS,
    GEMMA4_MAX_TOKENS,
    encode_gemma4_hidden_states_streamed,
)
from serenitymojo.parity import ParityHarness
from serenitymojo.tensor import Tensor


comptime CKPT = String(
    "/home/alex/.serenity/models/text_encoders/gemma-4-12b-it-standalone"
)
comptime REFS = String(
    "/home/alex/mojodiffusion/serenitymojo/models/text_encoder/parity/gemma4_refs"
)


def _host_of(st: ShardedSafeTensors, name: String, ctx: DeviceContext) raises -> List[Float32]:
    """Read an oracle tensor to host F32 (the refs are bf16 on disk)."""
    var tv = st.tensor_view(name)
    var t = Tensor.from_view_as_bf16(tv, ctx)
    return t.to_host(ctx)


def main() raises:
    var ctx = DeviceContext()
    print("== gemma4 real-weight states parity ==")

    # ---- oracle meta: the exact token ids the reference consumed ----
    # `gemma4_oracle_meta_f32.safetensors` carries the same ids as F32 (every
    # id < 2^24, so the cast is exact) because Tensor has no int64 host read.
    var meta = ShardedSafeTensors.open(
        REFS + "/gemma4_oracle_meta_f32.safetensors"
    )
    var ids_raw = Tensor.from_view_as_f32(
        meta.tensor_view("input_ids_f32"), ctx
    ).to_host(ctx)
    var rl_raw = Tensor.from_view_as_f32(
        meta.tensor_view("real_len_f32"), ctx
    ).to_host(ctx)
    var real_len = Int(rl_raw[0])
    print("oracle real_len=", real_len, " padded window=", len(ids_raw))
    if len(ids_raw) != GEMMA4_MAX_TOKENS:
        raise Error("oracle meta: expected a 1024-token padded window")

    # The real ids are the TAIL of the left-padded oracle window.
    var ids = List[Int]()
    for i in range(GEMMA4_MAX_TOKENS - real_len, GEMMA4_MAX_TOKENS):
        ids.append(Int(ids_raw[i]))
    print("first ids:", ids[0], ids[1], ids[2], " last id:", ids[real_len - 1])

    var prompts = List[List[Int]]()
    prompts.append(ids^)

    # ---- run the Mojo encoder on the real checkpoint ----
    var batch = encode_gemma4_hidden_states_streamed(CKPT, prompts, ctx)
    print("mojo bucket=", batch.bucket, " states=", len(batch.states[0]))
    if len(batch.states[0]) != GEMMA4_LAYERS + 1:
        raise Error("mojo encoder did not retain 49 states")

    # ---- compare each of the 49 states over the valid prefix ----
    var states_ref = ShardedSafeTensors.open(
        REFS + "/gemma4_oracle_states.safetensors"
    )
    var harness = ParityHarness(0.999)
    var worst_cos: Float64 = 2.0
    var worst_idx = -1
    var n_fail = 0
    var offset = GEMMA4_MAX_TOKENS - real_len
    var valid = real_len * GEMMA4_HIDDEN

    for si in range(GEMMA4_LAYERS + 1):
        var name = String("state_")
        if si < 10:
            name += "0"
        name += String(si)
        var ref_full = _host_of(states_ref, name, ctx)
        if len(ref_full) != GEMMA4_MAX_TOKENS * GEMMA4_HIDDEN:
            raise Error("oracle state size mismatch for " + name)
        # oracle valid prefix = rows [offset, 1024)
        var ref_valid = List[Float32]()
        for i in range(offset * GEMMA4_HIDDEN, GEMMA4_MAX_TOKENS * GEMMA4_HIDDEN):
            ref_valid.append(ref_full[i])

        # mojo valid prefix = rows [0, real_len)
        var mine_full = batch.states[0][si][].to_host(ctx)
        var mine_valid = List[Float32]()
        for i in range(valid):
            mine_valid.append(mine_full[i])

        var r = harness.compare_host(mine_valid, ref_valid)
        if not r.passed:
            n_fail += 1
        if r.cos < worst_cos:
            worst_cos = r.cos
            worst_idx = si
        # Print EVERY state: the shape of the curve separates smooth
        # accumulation from a per-layer-type defect (a global layer every 6th
        # would show up as a sawtooth, not a monotone slide).
        var is_global_out = (si > 0) and (si % 6) == 0
        print(
            name, " cos=", r.cos, " max_abs=", r.max_abs,
            " [after-global]" if is_global_out else "",
            " PASS" if r.passed else " FAIL",
        )

    print("---")
    print("worst state=", worst_idx, " cos=", worst_cos, " failures=", n_fail)
    if n_fail == 0:
        print("GATE PASS (all 49 states cos >= 0.999)")
    else:
        print("GATE FAIL")
