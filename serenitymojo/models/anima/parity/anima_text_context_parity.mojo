# serenitymojo/models/anima/parity/anima_text_context_parity.mojo
#
# PARITY GATE for the Anima net.llm_adapter forward
# (models/anima/anima_text_context.mojo: anima_llm_adapter_forward).
# Loads the EXACT FIXED inputs (T5 ids [1,512], Qwen3-hidden [1,512,1024]) and
# every real adapter weight dumped by anima_text_context_oracle.py, runs the Mojo
# adapter forward (F32), and compares the FROZEN context output [1,512,1024] vs
# the torch reference at cos >= 0.999 (reports max_abs).
#
# The oracle reimplements diffusers' AnimaTextConditioner from the authoritative
# Anima-Standalone-Trainer LLMAdapter source (== the class OneTrainer's
# AnimaModel instantiates) — a real adversarial cross-check, not a self-port.
#
# Run (oracle FIRST, SEPARATE command — never chained after a mojo build):
#   cd /home/alex/mojodiffusion
#   /home/alex/serenityflow-v2/.venv/bin/python \
#       serenitymojo/models/anima/parity/anima_text_context_oracle.py
#   rm -f serenitymojo.mojopkg
#   pixi run mojo build -I . -Xlinker -lm \
#       serenitymojo/models/anima/parity/anima_text_context_parity.mojo -o /tmp/anima_txt_ctx
#   /tmp/anima_txt_ctx

from std.gpu.host import DeviceContext
from std.collections import List, Dict
from std.memory import alloc, ArcPointer
from serenitymojo.parity import ParityHarness, ParityResult
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.models.anima.anima_text_context import (
    AnimaAdapterWeights, anima_llm_adapter_forward,
)

comptime REF_DIR = "/home/alex/mojodiffusion/serenitymojo/models/anima/parity/"

# MUST match anima_text_context_oracle.py
comptime S_TXT = 512
comptime S_LLM = 512
comptime DIM = 1024
comptime N_HEADS = 16
comptime HEAD_DIM = 64
comptime N_BLOCKS = 6
comptime MLP = 4096


def _read_bin_f32(path: String) raises -> List[Float32]:
    var fd = sys_open(path, O_RDONLY)
    if fd < 0:
        raise Error(String("cannot open: ") + path)
    var n = file_size(fd)
    if n <= 0:
        _ = sys_close(fd)
        raise Error(String("empty/missing ref (run the oracle first): ") + path)
    var buf = alloc[UInt8](n)
    var done = 0
    while done < n:
        var got = sys_pread(fd, buf + done, n - done, done)
        if got <= 0:
            break
        done += got
    _ = sys_close(fd)
    var nf = n // 4
    var fp = buf.bitcast[Float32]()
    var out = List[Float32]()
    for i in range(nf):
        out.append(fp[i])
    buf.free()
    return out^


def _in(name: String) raises -> List[Float32]:
    return _read_bin_f32(REF_DIR + name + ".bin")


def _t2(name: String, d0: Int, d1: Int, ctx: DeviceContext) raises -> Tensor:
    var sh = List[Int]()
    sh.append(d0)
    sh.append(d1)
    return Tensor.from_host(_in(name), sh^, STDtype.F32, ctx)


def _t1(name: String, d0: Int, ctx: DeviceContext) raises -> Tensor:
    var sh = List[Int]()
    sh.append(d0)
    return Tensor.from_host(_in(name), sh^, STDtype.F32, ctx)


def _put(
    mut w: Dict[String, ArcPointer[Tensor]], key: String, var t: Tensor
):
    w[key] = ArcPointer(t^)


def _load_weights(ctx: DeviceContext) raises -> AnimaAdapterWeights:
    var w = Dict[String, ArcPointer[Tensor]]()
    # embed / out_proj / norm
    _put(w, String("embed.weight"), _t2("in_w_embed_weight", 32128, DIM, ctx))
    _put(w, String("out_proj.weight"), _t2("in_w_out_proj_weight", DIM, DIM, ctx))
    _put(w, String("out_proj.bias"), _t1("in_w_out_proj_bias", DIM, ctx))
    _put(w, String("norm.weight"), _t1("in_w_norm_weight", DIM, ctx))

    for j in range(N_BLOCKS):
        var bp = String("blocks.") + String(j)
        var fp = String("in_w_blocks_") + String(j) + "_"
        # norms
        _put(w, bp + ".norm_self_attn.weight", _t1(fp + "norm_self_attn_weight", DIM, ctx))
        _put(w, bp + ".norm_cross_attn.weight", _t1(fp + "norm_cross_attn_weight", DIM, ctx))
        _put(w, bp + ".norm_mlp.weight", _t1(fp + "norm_mlp_weight", DIM, ctx))
        # self_attn
        _put(w, bp + ".self_attn.q_proj.weight", _t2(fp + "self_attn_q_proj_weight", DIM, DIM, ctx))
        _put(w, bp + ".self_attn.k_proj.weight", _t2(fp + "self_attn_k_proj_weight", DIM, DIM, ctx))
        _put(w, bp + ".self_attn.v_proj.weight", _t2(fp + "self_attn_v_proj_weight", DIM, DIM, ctx))
        _put(w, bp + ".self_attn.o_proj.weight", _t2(fp + "self_attn_o_proj_weight", DIM, DIM, ctx))
        _put(w, bp + ".self_attn.q_norm.weight", _t1(fp + "self_attn_q_norm_weight", HEAD_DIM, ctx))
        _put(w, bp + ".self_attn.k_norm.weight", _t1(fp + "self_attn_k_norm_weight", HEAD_DIM, ctx))
        # cross_attn
        _put(w, bp + ".cross_attn.q_proj.weight", _t2(fp + "cross_attn_q_proj_weight", DIM, DIM, ctx))
        _put(w, bp + ".cross_attn.k_proj.weight", _t2(fp + "cross_attn_k_proj_weight", DIM, DIM, ctx))
        _put(w, bp + ".cross_attn.v_proj.weight", _t2(fp + "cross_attn_v_proj_weight", DIM, DIM, ctx))
        _put(w, bp + ".cross_attn.o_proj.weight", _t2(fp + "cross_attn_o_proj_weight", DIM, DIM, ctx))
        _put(w, bp + ".cross_attn.q_norm.weight", _t1(fp + "cross_attn_q_norm_weight", HEAD_DIM, ctx))
        _put(w, bp + ".cross_attn.k_norm.weight", _t1(fp + "cross_attn_k_norm_weight", HEAD_DIM, ctx))
        # mlp (bias)
        _put(w, bp + ".mlp.0.weight", _t2(fp + "mlp_0_weight", MLP, DIM, ctx))
        _put(w, bp + ".mlp.0.bias", _t1(fp + "mlp_0_bias", MLP, ctx))
        _put(w, bp + ".mlp.2.weight", _t2(fp + "mlp_2_weight", DIM, MLP, ctx))
        _put(w, bp + ".mlp.2.bias", _t1(fp + "mlp_2_bias", DIM, ctx))

    return AnimaAdapterWeights(w^)


def main() raises:
    var ctx = DeviceContext()
    print("[anima-text-context-parity] loading oracle fixtures...")
    var wts = _load_weights(ctx)

    # T5 ids (stored as F32-encoded ints)
    var ids_f = _in("in_t5_ids")
    if len(ids_f) != S_TXT:
        raise Error("in_t5_ids length != 512")
    var t5_ids = List[Int]()
    for i in range(S_TXT):
        t5_ids.append(Int(ids_f[i]))

    # Qwen3 hidden [1,512,1024]
    var qh_sh = List[Int]()
    qh_sh.append(1)
    qh_sh.append(S_LLM)
    qh_sh.append(DIM)
    var qwen_hidden = Tensor.from_host(_in("in_qwen_hidden"), qh_sh^, STDtype.F32, ctx)

    print("[anima-text-context-parity] running Mojo adapter forward (F32)...")
    var context = anima_llm_adapter_forward(t5_ids, qwen_hidden, wts, ctx)

    var cs = context.shape()
    print("[anima-text-context-parity] context shape = [",
          cs[0], ",", cs[1], ",", cs[2], "]")
    if len(cs) != 3 or cs[0] != 1 or cs[1] != S_TXT or cs[2] != DIM:
        raise Error("context shape != [1,512,1024]")

    var ref_ctx = _in("ref_context")
    var harness = ParityHarness(0.999)
    var res = harness.compare(context, ref_ctx, ctx)
    print("[anima-text-context-parity] ", res)
    if res.passed:
        print("[anima-text-context-parity] PASS cos=", res.cos, " max_abs=", res.max_abs)
    else:
        print("[anima-text-context-parity] FAIL cos=", res.cos, " max_abs=", res.max_abs)
        raise Error("adapter parity FAILED (cos < 0.999)")
