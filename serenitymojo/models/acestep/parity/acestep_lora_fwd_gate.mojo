# xl-base forward-with-LoRA parity gate for acestep_dit (ACE-Step-1.5 xl-base, 4B).
#
# Loads the REAL xl-base decoder weights (sharded, 628 decoder.* tensors) + the
# oracle-dumped LoRA A/B (512) + inputs (xt/context/enc/t) + expected out0, then
# runs the LoRA-aware acestep_forward[SP=64, L=64, NH=32] with lora_scale=2.0
# (alpha/r = 16/8) and compares to out0. Double duty: validates the LoRA overlay
# math AND proves acestep_forward is config-parametric at 4B (32L/2560/32h).
#
# Oracle: scratchpad/acestep_train_ref_dump.py -> acestep_train_ref/. LoRA dump
# keys are "w_base_model__model__layers__N__<attn>__<proj>_proj__lora_{A,B}__default__weight",
# remapped here to full key "decoder.layers.N.<attn>.<proj>_proj.lora_{A,B}"
# (which _layer_bw slices to the per-layer relative "<attn>.<proj>_proj.lora_{A,B}").

from max.gpu.host import DeviceContext
from std.memory import ArcPointer
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.parity import ParityHarness
from serenitymojo.models.dit.acestep_dit import AceStepDiTConfig, acestep_forward

comptime XL_DIR = "/home/alex/ACE-Step-1.5/checkpoints/acestep-v15-xl-base"
comptime REF = "/home/alex/mojodiffusion/serenitymojo/models/acestep/parity/acestep_train_ref/acestep_train_ref.safetensors"
comptime SP = 64        # T=128 / patch 2
comptime L = 64
comptime NH = 32        # xl-base heads
comptime LAYERS = 32
comptime LORA_SCALE = Float32(2.0)   # alpha/r = 16/8


def _bf16(st: ShardedSafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    return cast_tensor(Tensor.from_view(st.tensor_view(name), ctx), STDtype.BF16, ctx)


def main() raises:
    var ctx = DeviceContext()
    var cfg = AceStepDiTConfig.xl_base()

    # --- base decoder weights (sharded xl-base) into `full` keyed decoder.* ----
    var xl = ShardedSafeTensors.open(XL_DIR)
    var full = Dict[String, ArcPointer[Tensor]]()
    var nbase = 0
    for nm in xl.names():
        var n = String(nm)
        if n.startswith("decoder."):
            full[n] = ArcPointer(_bf16(xl, n, ctx))
            nbase += 1
    print("base decoder tensors:", nbase)

    # --- oracle dump: inputs + expected + LoRA A/B ----------------------------
    var dump = ShardedSafeTensors.open(REF)
    var x_t = _bf16(dump, "xt", ctx)                    # [1,128,64]
    var context = _bf16(dump, "context_latents", ctx)   # [1,128,128]
    var enc = _bf16(dump, "encoder_hidden_states", ctx) # [1,64,2048]
    var t_host = Tensor.from_view(dump.tensor_view("t"), ctx).to_host(ctx)
    var t = t_host[0]

    # LoRA A/B: construct dump key + full key programmatically (no string parse).
    var attns = [String("self_attn"), String("cross_attn")]
    var projs = [String("q_proj"), String("k_proj"), String("v_proj"), String("o_proj")]
    var abs = [String("A"), String("B")]
    var nlora = 0
    for li in range(LAYERS):
        var ln = String(li)
        for a in attns:
            for p in projs:
                for ab in abs:
                    var dk = String("w_base_model__model__layers__") + ln + "__" + a + "__" + p + "__lora_" + ab + "__default__weight"
                    var fk = String("decoder.layers.") + ln + "." + a + "." + p + ".lora_" + ab
                    full[fk] = ArcPointer(_bf16(dump, dk, ctx))
                    nlora += 1
    print("LoRA A/B tensors loaded:", nlora, "(expect 512)")

    # --- LoRA-aware forward ---------------------------------------------------
    var out = acestep_forward[SP, L, NH](
        x_t, context, enc, t, t, full, cfg, ctx, LORA_SCALE
    )

    # --- compare to out0 ------------------------------------------------------
    var expected = Tensor.from_view(dump.tensor_view("out0"), ctx)   # F32
    var ph = ParityHarness(0.99)
    var res = ph.compare(out, expected.to_host(ctx), ctx)
    print("acestep xl-base LoRA-forward parity:", res)
    if res.passed:
        print("GATE: PASS")
    else:
        print("GATE: FAIL")
