# kandinsky5_block_parity.mojo — GPU bf16 parity for ONE Kandinsky-5 visual
# (decoder) transformer block: self-attn + cross-attn + FFN, with 9-param AdaLN.
#
# Drives serenitymojo/models/dit/kandinsky5_dit.kandinsky5_decoder_block against
# the canonical TransformerDecoderBlock oracle (kandinsky5_gen_oracle.py). Both
# sides are fed byte-identical f32 inputs (visual, text, time) and load the EXACT
# same weights; the Mojo path runs bf16 on GPU. Gate: cos >= 0.999.
#
# Run the oracle first, then this probe:
#   cd /home/alex/mojodiffusion
#   /home/alex/musubi-tuner/.venv/bin/python serenitymojo/models/dit/parity/kandinsky5_gen_oracle.py
#   pixi run mojo run -I . serenitymojo/models/dit/parity/kandinsky5_block_parity.mojo

from std.gpu.host import DeviceContext
from std.memory import alloc, ArcPointer
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.models.dit.kandinsky5_dit import (
    Kandinsky5Config, kandinsky5_build_visual_rope, kandinsky5_decoder_block,
    _expand_rope_per_head,
)


comptime DIR = "/home/alex/mojodiffusion/serenitymojo/models/dit/parity/"

# Grid from k5_grid.txt: D=1,H=2,W=2 -> S=4; TXT=8; dim=1792; head=64; heads=28.
comptime S = 4
comptime TXT = 8
comptime DIM = 1792
comptime TIME = 512
comptime NH = 28
comptime HD = 64
comptime D_OUT = 1
comptime H_OUT = 2
comptime W_OUT = 2


def _read_f32_bin(name: String) raises -> List[Float32]:
    var path = String(DIR) + name + ".bin"
    var fd = sys_open(path, O_RDONLY)
    if fd < 0:
        raise Error(String("cannot open ") + path + " (run the oracle first)")
    var nbytes = file_size(fd)
    if nbytes <= 0:
        _ = sys_close(fd)
        raise Error(String("empty bin: ") + path)
    var buf = alloc[UInt8](nbytes)
    var done = 0
    while done < nbytes:
        var got = sys_pread(fd, buf + done, nbytes - done, done)
        if got <= 0:
            break
        done += got
    _ = sys_close(fd)
    var nfloats = done // 4
    var out = List[Float32]()
    var fptr = buf.bitcast[Float32]()
    for i in range(nfloats):
        out.append(fptr[i])
    buf.free()
    return out^


# Load a 2D weight bin [r,c] -> device Tensor of dtype dt.
def _load_w2(name: String, r: Int, c: Int, dt: STDtype, ctx: DeviceContext) raises -> Tensor:
    var h = _read_f32_bin(name)
    if len(h) != r * c:
        raise Error(String("size mismatch ") + name + " got " + String(len(h)) + " want " + String(r * c))
    var f32 = Tensor.from_host(h^, [r, c], STDtype.F32, ctx)
    if dt == STDtype.F32:
        return f32^
    return cast_tensor(f32, dt, ctx)


# Load a 1D weight bin [n] -> device Tensor of dtype dt.
def _load_w1(name: String, n: Int, dt: STDtype, ctx: DeviceContext) raises -> Tensor:
    var h = _read_f32_bin(name)
    if len(h) != n:
        raise Error(String("size mismatch ") + name + " got " + String(len(h)) + " want " + String(n))
    var f32 = Tensor.from_host(h^, [n], STDtype.F32, ctx)
    if dt == STDtype.F32:
        return f32^
    return cast_tensor(f32, dt, ctx)


def main() raises:
    var ctx = DeviceContext()
    print("=== Kandinsky-5 visual block-0 parity (S=", S, " TXT=", TXT, ", bf16 GPU) ===")

    var cfg = Kandinsky5Config.t2v_lite_5s()

    # ── inputs (f32 bytes) -> device ──
    var vin_h = _read_f32_bin("k5_block0_visual_in")   # [S,DIM]
    var tin_h = _read_f32_bin("k5_block0_text_in")     # [TXT,DIM]
    var time_h = _read_f32_bin("k5_block0_time_in")    # [TIME]
    var ref_h = _read_f32_bin("k5_block0_out")         # [S,DIM]

    if len(vin_h) != S * DIM:
        raise Error("visual_in size mismatch")
    if len(tin_h) != TXT * DIM:
        raise Error("text_in size mismatch")
    if len(time_h) != TIME:
        raise Error("time_in size mismatch")
    if len(ref_h) != S * DIM:
        raise Error("out size mismatch")

    var v_f32 = Tensor.from_host(vin_h.copy(), [1, S, DIM], STDtype.F32, ctx)
    var visual = cast_tensor(v_f32, STDtype.BF16, ctx)
    var t_f32 = Tensor.from_host(tin_h.copy(), [1, TXT, DIM], STDtype.F32, ctx)
    var text = cast_tensor(t_f32, STDtype.BF16, ctx)
    # time embed stays F32 (modulation math is F32).
    var time_embed = Tensor.from_host(time_h.copy(), [1, TIME], STDtype.F32, ctx)

    # ── 3D RoPE tables (bf16, interleaved) for the grid, expanded per head ──
    var vcs = kandinsky5_build_visual_rope(
        D_OUT, H_OUT, W_OUT, cfg, cfg.max_period, 1.0, 1.0, 1.0, STDtype.BF16, ctx
    )
    var cos_e = _expand_rope_per_head(vcs[0], S, NH, HD // 2, ctx)
    var sin_e = _expand_rope_per_head(vcs[1], S, NH, HD // 2, ctx)

    # ── load block weights (exact oracle weights) into the dotted-key dict ──
    var w = Dict[String, ArcPointer[Tensor]]()
    # modulation (F32, [9*DIM, TIME] and [9*DIM])
    w["visual_modulation.out_layer.weight"] = ArcPointer(
        _load_w2("w_visual_modulation_out_layer_weight", 9 * DIM, TIME, STDtype.F32, ctx))
    w["visual_modulation.out_layer.bias"] = ArcPointer(
        _load_w1("w_visual_modulation_out_layer_bias", 9 * DIM, STDtype.F32, ctx))
    # self + cross attention QKV/out (bf16, [DIM,DIM] + [DIM] bias)
    var attn_kinds = ["self_attention", "cross_attention"]
    for ak in attn_kinds:
        var p = String(ak)
        w[p + ".to_query.weight"] = ArcPointer(_load_w2("w_" + p + "_to_query_weight", DIM, DIM, STDtype.BF16, ctx))
        w[p + ".to_query.bias"] = ArcPointer(_load_w1("w_" + p + "_to_query_bias", DIM, STDtype.BF16, ctx))
        w[p + ".to_key.weight"] = ArcPointer(_load_w2("w_" + p + "_to_key_weight", DIM, DIM, STDtype.BF16, ctx))
        w[p + ".to_key.bias"] = ArcPointer(_load_w1("w_" + p + "_to_key_bias", DIM, STDtype.BF16, ctx))
        w[p + ".to_value.weight"] = ArcPointer(_load_w2("w_" + p + "_to_value_weight", DIM, DIM, STDtype.BF16, ctx))
        w[p + ".to_value.bias"] = ArcPointer(_load_w1("w_" + p + "_to_value_bias", DIM, STDtype.BF16, ctx))
        w[p + ".query_norm.weight"] = ArcPointer(_load_w1("w_" + p + "_query_norm_weight", HD, STDtype.BF16, ctx))
        w[p + ".key_norm.weight"] = ArcPointer(_load_w1("w_" + p + "_key_norm_weight", HD, STDtype.BF16, ctx))
        w[p + ".out_layer.weight"] = ArcPointer(_load_w2("w_" + p + "_out_layer_weight", DIM, DIM, STDtype.BF16, ctx))
        w[p + ".out_layer.bias"] = ArcPointer(_load_w1("w_" + p + "_out_layer_bias", DIM, STDtype.BF16, ctx))
    # FFN (bf16, no bias)
    w["feed_forward.in_layer.weight"] = ArcPointer(_load_w2("w_feed_forward_in_layer_weight", cfg.ff_dim, DIM, STDtype.BF16, ctx))
    w["feed_forward.out_layer.weight"] = ArcPointer(_load_w2("w_feed_forward_out_layer_weight", DIM, cfg.ff_dim, STDtype.BF16, ctx))

    # ── forward ──
    var out_bf16 = kandinsky5_decoder_block[S, TXT, NH, HD](
        visual, text, time_embed, cos_e, sin_e, w, cfg, ctx
    )
    var out_f32 = cast_tensor(out_bf16, STDtype.F32, ctx)

    var harness = ParityHarness(0.999)
    var r = harness.compare(out_f32, ref_h, ctx)
    print("    kandinsky5 visual block0 (bf16):", r)
    if r.passed:
        print("GATE PASS blockGateCos=", r.cos)
    else:
        print("GATE FAIL blockGateCos=", r.cos)
