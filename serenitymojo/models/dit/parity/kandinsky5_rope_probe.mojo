# kandinsky5_rope_probe.mojo — isolate self-attn: compare q pre/post RoPE and the
# raw self-attn output against oracle dumps. Pinpoints whether RoPE or attn diverges.

from std.gpu.host import DeviceContext
from std.memory import alloc, ArcPointer
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.linear import linear
from serenitymojo.ops.norm import rms_norm
from serenitymojo.ops.rope import rope_interleaved
from serenitymojo.ops.tensor_algebra import reshape
from serenitymojo.models.dit.kandinsky5_dit import (
    Kandinsky5Config, kandinsky5_build_visual_rope, _expand_rope_per_head,
)

comptime DIR = "/home/alex/mojodiffusion/serenitymojo/models/dit/parity/"
comptime S = 4
comptime TXT = 8
comptime DIM = 1792
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


def _w2(name: String, r: Int, c: Int, ctx: DeviceContext) raises -> Tensor:
    return cast_tensor(Tensor.from_host(_read(name), [r, c], STDtype.F32, ctx), STDtype.BF16, ctx)


def _w1(name: String, n: Int, ctx: DeviceContext) raises -> Tensor:
    return cast_tensor(Tensor.from_host(_read(name), [n], STDtype.F32, ctx), STDtype.BF16, ctx)


def main() raises:
    var ctx = DeviceContext()
    var cfg = Kandinsky5Config.t2v_lite_5s()

    var sa_in = cast_tensor(Tensor.from_host(_read("k5_sa_in"), [1, S, DIM], STDtype.F32, ctx), STDtype.BF16, ctx)
    var qw = _w2("w_self_attention_to_query_weight", DIM, DIM, ctx)
    var qb = _w1("w_self_attention_to_query_bias", DIM, ctx)
    var qn = _w1("w_self_attention_query_norm_weight", HD, ctx)

    var q = linear(sa_in, qw, Optional(qb.clone(ctx)), ctx)  # [1,S,dim]
    var q3 = reshape(q, [S, NH, HD], ctx)
    q3 = rms_norm(q3, qn, cfg.eps, ctx)
    # compare pre-rope q3 vs oracle [S,H,DH]
    var hp = ParityHarness(0.999)
    var rp = hp.compare(cast_tensor(q3, STDtype.F32, ctx), _read("k5_sa_q_prerope"), ctx)
    print("q_prerope:", rp)

    var vcs = kandinsky5_build_visual_rope(1, 2, 2, cfg, cfg.max_period, 1.0, 1.0, 1.0, STDtype.BF16, ctx)
    var cos_e = _expand_rope_per_head(vcs[0], S, NH, HD // 2, ctx)
    var sin_e = _expand_rope_per_head(vcs[1], S, NH, HD // 2, ctx)
    var q4 = reshape(q3, [1, S, NH, HD], ctx)
    q4 = rope_interleaved(q4, cos_e, sin_e, ctx)
    var hpr = ParityHarness(0.999)
    var rpr = hpr.compare(cast_tensor(q4, STDtype.F32, ctx), _read("k5_sa_q_postrope"), ctx)
    print("q_postrope:", rpr)

    # full self-attn via the model helper -> compare raw out
    from serenitymojo.models.dit.kandinsky5_dit import _self_attention
    var w = Dict[String, ArcPointer[Tensor]]()
    var p = String("self_attention")
    w[p + ".to_query.weight"] = ArcPointer(_w2("w_self_attention_to_query_weight", DIM, DIM, ctx))
    w[p + ".to_query.bias"] = ArcPointer(_w1("w_self_attention_to_query_bias", DIM, ctx))
    w[p + ".to_key.weight"] = ArcPointer(_w2("w_self_attention_to_key_weight", DIM, DIM, ctx))
    w[p + ".to_key.bias"] = ArcPointer(_w1("w_self_attention_to_key_bias", DIM, ctx))
    w[p + ".to_value.weight"] = ArcPointer(_w2("w_self_attention_to_value_weight", DIM, DIM, ctx))
    w[p + ".to_value.bias"] = ArcPointer(_w1("w_self_attention_to_value_bias", DIM, ctx))
    w[p + ".query_norm.weight"] = ArcPointer(_w1("w_self_attention_query_norm_weight", HD, ctx))
    w[p + ".key_norm.weight"] = ArcPointer(_w1("w_self_attention_key_norm_weight", HD, ctx))
    w[p + ".out_layer.weight"] = ArcPointer(_w2("w_self_attention_out_layer_weight", DIM, DIM, ctx))
    w[p + ".out_layer.bias"] = ArcPointer(_w1("w_self_attention_out_layer_bias", DIM, ctx))
    var sa_raw = _self_attention[S, NH, HD, True](sa_in, cos_e, sin_e, w, "self_attention.", cfg.eps, ctx)
    var hr = ParityHarness(0.999)
    var rr = hr.compare(cast_tensor(sa_raw, STDtype.F32, ctx), _read("k5_sa_raw_out"), ctx)
    print("sa_raw_out:", rr)
