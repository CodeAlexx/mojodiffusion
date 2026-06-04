# GPT-OSS (Lens) text encoder — PARITY GATE run, 2026-06-03

Component: `gpt_oss_encoder` (serenitymojo/models/text_encoder/gpt_oss_encoder.mojo)
Oracle: HF transformers 4.57.6, GptOssForCausalLM, bf16, MXFP4→bf16 dequant.
Prompt: "a photo of a cat" → raw tokenizer.json ids [64, 8767, 328, 261, 9059] (5 tokens, no BOS/EOS).
Capture layers (Lens): [5, 11, 17, 23]; HF map = hidden_states[6,12,18,24].

## RESULT: BLOCKED (gate did NOT produce Mojo numbers)

The HF oracle ran cleanly and dumped reference captures. The pure-Mojo streamed
forward FAILED at runtime in layer 0's MoE, before producing any capture, so NO
per-layer cosine could be measured. This is a never-before-exercised runtime bug
(the skeptic explicitly noted "the runtime behaviour of the streamed forward is
NOT exercised … no parity oracle was run").

## What ran

### Oracle — PASS (dumped oracle_captures.safetensors, bf16)
- 25 hidden_states (24 layers + embeddings), all [1,5,2880].
- Forward run on CPU bf16: the fully-dequantized bf16 GPT-OSS-20B is ~60GB
  resident (MoE experts expand ~4× off the 13.7GB MXFP4 on disk) and does NOT
  fit as one resident model on the 24GB RTX 3090 Ti. On-GPU MXFP4 dequant also
  OOMs (the dequant scratch tops 24GB on top of the model). The Mojo port avoids
  this by STREAMING one layer at a time; HF has no streamed forward, so the
  oracle forward is on CPU. Still bf16 storage with fp32 accumulation — a
  faithful bf16 oracle, same dtype contract as the GPU path.
- Per-layer stats (mean / var / absmax):
  - l5  : 0.057620 / 74.98 / 180.0
  - l11 : -2.827229 / 114667.6 / 40448.0   (large activation outliers — known GPT-OSS behavior)
  - l17 : -0.171777 / 11350.06 / 2768.0
  - l23 : 0.029108 / 22.81 / 178.0
  - final_normed : 0.080153 / 137.57 / 508.0

### Mojo — FAILED at runtime (no captures produced)
Tokenized + config + load OK, embedding OK, layer 0 attention OK, then in
layer-0 MoE the run aborted:

    Unhandled exception caught during execution:
    gated_scatter_add: expert_out and accum must be F32

## BLOCKER (report-only; bugfix is a separate agent)

### BUG-1 — MoE down-projection output is BF16, but gated_scatter_add requires F32
- Location: `gpt_oss_encoder.mojo` `_moe`, line 903:
  `gated_scatter_add(down_out, gating_e, idx_e, accum, ctx)`
- `down_out` is BF16: it comes from `linear(act, down_w, None, ctx)` (BF16 in →
  BF16 out) then `_add_row_bias(...)` (BF16). `ops/moe.gated_scatter_add` hard-
  requires `expert_out.dtype()==F32 and accum.dtype()==F32`
  (ops/moe.mojo:393). So the very first expert scatter in layer 0 raises.
- This is NOT a numerical/parity bug — it is a hard runtime dtype-contract bug
  that aborts the forward. It was never caught because the encoder forward was
  never run end-to-end (no prior oracle).
- Minimal fix for the bugfix agent (one line, no math change; F32 accumulation
  is the intended/standard MoE accumulation): cast before the scatter, e.g.
  insert before line 903:
      `down_out = cast_tensor(down_out, STDtype.F32, ctx)`
  (`cast_tensor` is already imported at line 61.) The accumulator `accum` is
  already F32 and the result is read back F32→BF16 at line 906-907, so only the
  expert output needs the F32 cast.

### BUG-0 (already fixed by me to enable the build) — module did not compile
The file as delivered did NOT compile under the current Mojo toolchain
(1.0.0b1). The skeptic's note claims "module now compiles" after fixing ONE site
(`pair[0].copy()` at line 204), but that was incomplete: 22 further sites failed
with "value of type 'Tensor'/'List[Float32]' cannot be implicitly copied". These
were all weight-binding lines `var w = self._bw(block, name)` (which copies a
non-ImplicitlyCopyable Tensor returned by `ref`) and one `var use_mask =
sliding_mask if … else full_mask` (copies a List). I changed these `var`→`ref`
bindings (lines 757-822, 926, 933) and split the mask ternary into an if/else
(line 1011), passing the selected mask by borrow. ALL consumers (`linear`,
`rms_norm`, `slice`, `t_add`, `_sdpa_with_sinks`, `_layer`) take Tensor/List by
borrow, so these are pure binding fixes — zero math change, identical bytes flow
to every op. After this, the module compiles and links; the run then reaches
BUG-1.

## To complete the gate
After BUG-1 is fixed, re-run:
1. `cd /home/alex/mojodiffusion && pixi run mojo build -I . -Xlinker -lm -Xlinker -lcuda serenitymojo/pipeline/gpt_oss_parity_dump.mojo -o /tmp/gptoss_parity && /tmp/gptoss_parity`
   → writes mine_captures.safetensors (keys l5,l11,l17,l23).
2. `compare.py` (below): cos + |mine|/|ref| per layer vs oracle_captures.safetensors.
Files in this dir: oracle_gpt_oss.py, oracle_captures.safetensors,
gpt_oss_parity_dump.mojo (in pipeline/), compare_gpt_oss.py.
