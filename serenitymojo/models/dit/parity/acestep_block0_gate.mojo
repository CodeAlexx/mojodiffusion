# Numeric block-0 parity gate for acestep_dit (ACE-Step-1.5 turbo DiT).
#
# Loads acestep_block0_fixture.safetensors (built by gen_acestep_block0_oracle.py
# from the REAL acestep-v15-turbo checkpoint: canonical AceStepDiTLayer(0) run
# eager-mode bf16 on GPU at S=64<=sliding_window(128) so the self-attn mask is
# all-zeros -> sdpa_nomask is exact). Runs acestep_block0_forward, compares to
# `expected`. Gate cos >= 0.999.
#
# Block math (decoder.layers.0), matching modeling_acestep_v15_turbo.AceStepDiTLayer:
#   mod = scale_shift_table[1,6,H] + temb[1,6,H]; chunk6 ->
#         shift_msa,scale_msa,gate_msa,c_shift,c_scale,c_gate  each [H]
#   1) n = rms_norm(x, self_attn_norm); n = (1+scale_msa)*n + shift_msa
#      self-attn: q/k/v proj (no bias) -> [S,heads/kv,Dh]; per-head rms_norm q,k
#      (Qwen3RMSNorm on Dh); rope_halfsplit on q,k; GQA repeat_kv; sdpa_nomask;
#      o_proj.  x = x + gate_msa * attn
#   2) n = rms_norm(x, cross_attn_norm)
#      cross-attn: q from n; k,v from enc; per-head rms_norm q,k; NO rope;
#      GQA; sdpa_nomask (enc mask all-zero); o_proj.  x = x + cross
#   3) n = rms_norm(x, mlp_norm); n = (1+c_scale)*n + c_shift
#      swiglu MLP: down(silu(gate(n))*up(n)).  x = x + c_gate * mlp

from std.gpu.host import DeviceContext
from std.memory import ArcPointer
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.linear import linear
from serenitymojo.ops.norm import rms_norm
from serenitymojo.ops.activations import silu
from serenitymojo.ops.attention import sdpa_nomask
from serenitymojo.ops.rope import rope_halfsplit
from serenitymojo.ops.tensor_algebra import reshape, slice, transpose
from serenitymojo.ops.elementwise import modulate, residual_gate
from serenitymojo.parity import ParityHarness
from serenitymojo.models.dit.acestep_dit import (
    AceStepDiTConfig, acestep_block0_forward,
)

comptime FIX = "/home/alex/mojodiffusion/serenitymojo/models/dit/parity/acestep_block0_fixture.safetensors"
comptime S = 64
comptime L = 48


def _load_bf16(st: ShardedSafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    var tv = st.tensor_view(name)
    return cast_tensor(Tensor.from_view(tv, ctx), STDtype.BF16, ctx)


def main() raises:
    var ctx = DeviceContext()
    var cfg = AceStepDiTConfig.turbo()
    var st = ShardedSafeTensors.open(FIX)

    var hidden = _load_bf16(st, "hidden", ctx)   # [1,S,H]
    var enc    = _load_bf16(st, "enc", ctx)      # [1,L,H]
    var temb   = _load_bf16(st, "temb", ctx)     # [1,6,H]
    var cos_b  = _load_bf16(st, "cos", ctx)      # [S,Dh/2]
    var sin_b  = _load_bf16(st, "sin", ctx)      # [S,Dh/2]

    var bw = Dict[String, ArcPointer[Tensor]]()
    var suffixes = [
        "scale_shift_table",
        "self_attn_norm.weight", "cross_attn_norm.weight", "mlp_norm.weight",
        "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
        "self_attn.o_proj.weight", "self_attn.q_norm.weight", "self_attn.k_norm.weight",
        "cross_attn.q_proj.weight", "cross_attn.k_proj.weight", "cross_attn.v_proj.weight",
        "cross_attn.o_proj.weight", "cross_attn.q_norm.weight", "cross_attn.k_norm.weight",
        "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
    ]
    for sfx in suffixes:
        var s = String(sfx)
        bw[s] = ArcPointer(_load_bf16(st, String("w_") + s, ctx)^)

    # Layer 0 is "sliding_attention" ((0+1)%2==1); at S=64 <= window(128) the
    # |i-j|<=window mask is all-zeros so sdpa_nomask stays exact (gate unchanged).
    var out = acestep_block0_forward[S, L](
        hidden, temb, enc, cos_b, sin_b, bw, cfg, 1, 128, ctx
    )

    var ph = ParityHarness(0.999)
    var expected = Tensor.from_view(st.tensor_view("expected"), ctx)  # F32
    var ref_host = expected.to_host(ctx)
    var res = ph.compare(out, ref_host, ctx)
    print("acestep block-0 parity:", res)
    if res.passed:
        print("GATE: PASS")
    else:
        print("GATE: FAIL")
