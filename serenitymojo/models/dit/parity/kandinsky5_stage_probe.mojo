# kandinsky5_stage_probe.mojo — localize the block-0 mismatch by gating each
# sub-stage (after self-attn, after cross-attn) against the oracle dumps.

from std.gpu.host import DeviceContext
from std.memory import alloc, ArcPointer
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.models.dit.kandinsky5_dit import (
    Kandinsky5Config, kandinsky5_build_visual_rope, _expand_rope_per_head,
    kandinsky5_modulation, _mod_chunk, kandinsky5_mod_pre, kandinsky5_gate_sum,
    _self_attention, _cross_attention,
)

comptime DIR = "/home/alex/mojodiffusion/serenitymojo/models/dit/parity/"
comptime S = 4
comptime TXT = 8
comptime DIM = 1792
comptime TIME = 512
comptime NH = 28
comptime HD = 64


def _read(name: String) raises -> List[Float32]:
    var path = String(DIR) + name + ".bin"
    var fd = sys_open(path, O_RDONLY)
    if fd < 0:
        raise Error(String("cannot open ") + path)
    var nbytes = file_size(fd)
    var buf = alloc[UInt8](nbytes)
    var done = 0
    while done < nbytes:
        var got = sys_pread(fd, buf + done, nbytes - done, done)
        if got <= 0:
            break
        done += got
    _ = sys_close(fd)
    var out = List[Float32]()
    var fptr = buf.bitcast[Float32]()
    for i in range(done // 4):
        out.append(fptr[i])
    buf.free()
    return out^


def _w2(name: String, r: Int, c: Int, dt: STDtype, ctx: DeviceContext) raises -> Tensor:
    var h = _read(name)
    var f = Tensor.from_host(h^, [r, c], STDtype.F32, ctx)
    if dt == STDtype.F32:
        return f^
    return cast_tensor(f, dt, ctx)


def _w1(name: String, n: Int, dt: STDtype, ctx: DeviceContext) raises -> Tensor:
    var h = _read(name)
    var f = Tensor.from_host(h^, [n], STDtype.F32, ctx)
    if dt == STDtype.F32:
        return f^
    return cast_tensor(f, dt, ctx)


def main() raises:
    var ctx = DeviceContext()
    var cfg = Kandinsky5Config.t2v_lite_5s()

    var visual = cast_tensor(Tensor.from_host(_read("k5_block0_visual_in"), [1, S, DIM], STDtype.F32, ctx), STDtype.BF16, ctx)
    var text = cast_tensor(Tensor.from_host(_read("k5_block0_text_in"), [1, TXT, DIM], STDtype.F32, ctx), STDtype.BF16, ctx)
    var time_embed = Tensor.from_host(_read("k5_block0_time_in"), [1, TIME], STDtype.F32, ctx)

    var vcs = kandinsky5_build_visual_rope(1, 2, 2, cfg, cfg.max_period, 1.0, 1.0, 1.0, STDtype.BF16, ctx)
    var cos_e = _expand_rope_per_head(vcs[0], S, NH, HD // 2, ctx)
    var sin_e = _expand_rope_per_head(vcs[1], S, NH, HD // 2, ctx)

    var w = Dict[String, ArcPointer[Tensor]]()
    w["visual_modulation.out_layer.weight"] = ArcPointer(_w2("w_visual_modulation_out_layer_weight", 9 * DIM, TIME, STDtype.F32, ctx))
    w["visual_modulation.out_layer.bias"] = ArcPointer(_w1("w_visual_modulation_out_layer_bias", 9 * DIM, STDtype.F32, ctx))
    var aks = ["self_attention", "cross_attention"]
    for ak in aks:
        var p = String(ak)
        w[p + ".to_query.weight"] = ArcPointer(_w2("w_" + p + "_to_query_weight", DIM, DIM, STDtype.BF16, ctx))
        w[p + ".to_query.bias"] = ArcPointer(_w1("w_" + p + "_to_query_bias", DIM, STDtype.BF16, ctx))
        w[p + ".to_key.weight"] = ArcPointer(_w2("w_" + p + "_to_key_weight", DIM, DIM, STDtype.BF16, ctx))
        w[p + ".to_key.bias"] = ArcPointer(_w1("w_" + p + "_to_key_bias", DIM, STDtype.BF16, ctx))
        w[p + ".to_value.weight"] = ArcPointer(_w2("w_" + p + "_to_value_weight", DIM, DIM, STDtype.BF16, ctx))
        w[p + ".to_value.bias"] = ArcPointer(_w1("w_" + p + "_to_value_bias", DIM, STDtype.BF16, ctx))
        w[p + ".query_norm.weight"] = ArcPointer(_w1("w_" + p + "_query_norm_weight", HD, STDtype.BF16, ctx))
        w[p + ".key_norm.weight"] = ArcPointer(_w1("w_" + p + "_key_norm_weight", HD, STDtype.BF16, ctx))
        w[p + ".out_layer.weight"] = ArcPointer(_w2("w_" + p + "_out_layer_weight", DIM, DIM, STDtype.BF16, ctx))
        w[p + ".out_layer.bias"] = ArcPointer(_w1("w_" + p + "_out_layer_bias", DIM, STDtype.BF16, ctx))

    var dim = cfg.model_dim
    var eps = cfg.eps
    var mp = kandinsky5_modulation(time_embed, w, "visual_modulation.out_layer.weight", "visual_modulation.out_layer.bias", ctx)
    var sa_shift = _mod_chunk(mp, 0, dim, ctx)
    var sa_scale = _mod_chunk(mp, 1, dim, ctx)
    var sa_gate = _mod_chunk(mp, 2, dim, ctx)
    var ca_shift = _mod_chunk(mp, 3, dim, ctx)
    var ca_scale = _mod_chunk(mp, 4, dim, ctx)
    var ca_gate = _mod_chunk(mp, 5, dim, ctx)

    # stage 1: after self-attn
    var sa_in = cast_tensor(kandinsky5_mod_pre(visual, sa_scale, sa_shift, eps, ctx), visual.dtype(), ctx)
    var sa_out = _self_attention[S, NH, HD, False](sa_in, cos_e, sin_e, w, "self_attention.", eps, ctx)
    var v_sa = kandinsky5_gate_sum(visual, sa_out, sa_gate, ctx)
    var h1 = ParityHarness(0.999)
    var r1 = h1.compare(cast_tensor(v_sa, STDtype.F32, ctx), _read("k5_block0_after_sa"), ctx)
    print("after_sa:", r1)

    # stage 2: after cross-attn
    var ca_in = cast_tensor(kandinsky5_mod_pre(v_sa, ca_scale, ca_shift, eps, ctx), visual.dtype(), ctx)
    var ca_out = _cross_attention[S, TXT, NH, HD](ca_in, text, w, "cross_attention.", eps, ctx)
    var v_ca = kandinsky5_gate_sum(v_sa, ca_out, ca_gate, ctx)
    var h2 = ParityHarness(0.999)
    var r2 = h2.compare(cast_tensor(v_ca, STDtype.F32, ctx), _read("k5_block0_after_ca"), ctx)
    print("after_ca:", r2)
