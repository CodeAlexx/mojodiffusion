# pipeline/qwen3vl_caption.mojo — pure-Mojo IMAGE CAPTIONER (Qwen3-VL-4B-Instruct).
#
# image file -> caption text, no Python/llama.cpp at runtime. Composes four
# already-gated pieces + the fast cached decode:
#   1. lingbot_vision_preprocess  (bicubic + patchify, parity vs Qwen processor)
#   2. Qwen3VLVisionModel         (24-block tower, parity cos>=0.999 vs torch)
#   3. the fuse math ROW-WISE: masked-scatter pooler rows into the token stream,
#      3D interleaved M-RoPE positions (fuse._build_positions/_build_mrope_tables,
#      parity-gated in qwen3vl_fuse_parity), deepstack adds after layers 0/1/2 —
#      primed incrementally through llm/vl_decode.vl_decode_step_embed into a
#      KVCache (per-position math identical to fuse_core's batched form).
#   4. greedy generation via the same cached step (~O(1)/token). Generated TEXT
#      positions have t=h=w=cursor+i, where the equal-axis M-RoPE row equals a
#      plain-rope row — built with the SAME mrope table builder for exactness.
# Logits via the TIED embedding table (Qwen3-VL-4B has no lm_head key).
#
# V1 GEOMETRY: comptime S=1024 vision patches = 512x512 processed size (grid
# 32x32, 256 merged vision tokens). Images whose smart_resize grid != 32x32
# FAIL LOUD with the required pre-resize. More comptime buckets are mechanical.
#
# Run:
#   pixi run mojo build --optimization-level 2 -I . -I /home/alex/MOJO-libs \
#     -Xlinker -lm -Xlinker -lcuda serenitymojo/pipeline/qwen3vl_caption.mojo \
#     -o output/bin/qwen3vl_caption
#   LD_LIBRARY_PATH=.pixi/envs/default/lib output/bin/qwen3vl_caption <img.png> \
#     ["prompt"] [max_new]
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.sys import argv
from std.math import cos as fcos, sin as fsin, exp as fexp, log as flog
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext
from std.collections import Optional
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.tokenizer.tokenizer import Qwen3Tokenizer
from serenitymojo.models.text_encoder.qwen3_encoder import Qwen3Encoder, _reshape
from serenitymojo.models.text_encoder.krea2_qwen3vl_4b import load_krea2_qwen3vl_4b
from serenitymojo.models.text_encoder.lingbot_qwen3vl_vision import (
    Qwen3VLVisionModel, VisionOutput,
)
from serenitymojo.models.text_encoder.lingbot_qwen3vl_fuse import (
    _build_positions, _build_mrope_tables,
)
from serenitymojo.models.lingbotvideo.lingbot_vision_preprocess import (
    lingbot_vision_preprocess_bucketed, VisionPreproc,
)
from serenitymojo.ops.tensor_algebra import slice
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.llm.decoder import KVCache
from serenitymojo.llm.vl_decode import vl_decode_step_embed

comptime SNAP = "/home/alex/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17"
comptime TOKJSON = SNAP + "/tokenizer.json"
comptime EOS = 151645          # <|im_end|>
comptime IMG_PAD = 151655      # <|image_pad|> (fuse.IMAGE_TOKEN_ID)
# Comptime vision-seq buckets (see lingbot_vision_preprocess_bucketed):
#   S=1024 (512²) | 1536 (512x768 mix) | 4096 (1024²) | 6144 (1024x1536 mix)
#   | 9216 (1536²) | 16384 (2048² — tower O(S²) attention; may OOM on 24GB, measured)


def _row(t: Tensor, i: Int, hidden: Int, ctx: DeviceContext) raises -> Tensor:
    """row i of a [n, hidden] tensor -> [1,1,hidden] bf16."""
    var r = slice(t, 0, i, 1, ctx)                # [1, hidden]
    var r3 = _reshape(r, [1, 1, hidden], ctx)
    if r3.dtype() != STDtype.BF16:
        return cast_tensor(r3, STDtype.BF16, ctx)
    return r3^


struct RopeRow(Movable):
    """cos/sin rope rows for one position (move-only Tensors can't cross as a
    tuple — same workaround as ops FwhtQuantOut)."""
    var c: Tensor
    var s: Tensor

    def __init__(out self, var c: Tensor, var s: Tensor):
        self.c = c^
        self.s = s^


def _rope_row(
    tab_cos: List[Float32], tab_sin: List[Float32], i: Int, heads: Int, half: Int,
    ctx: DeviceContext,
) raises -> RopeRow:
    """Slice position i's (heads*half) cos/sin rows out of a full mrope table."""
    var off = i * heads * half
    var c = List[Float32]()
    var s = List[Float32]()
    for j in range(heads * half):
        c.append(tab_cos[off + j])
        s.append(tab_sin[off + j])
    var ct = Tensor.from_host(c, [heads * half], STDtype.BF16, ctx)
    var st = Tensor.from_host(s, [heads * half], STDtype.BF16, ctx)
    return RopeRow(ct^, st^)


def _plain_rope_row(
    pos: Int, heads: Int, dh: Int, theta: Float64, ctx: DeviceContext,
) raises -> RopeRow:
    """Equal-axis (t=h=w=pos) M-RoPE row == plain rope row: axis selection is
    irrelevant when all three positions match, so angle_k = pos * theta^(-2k/dh)."""
    var half = dh // 2
    var log_theta = flog(Float32(theta))
    var c = List[Float32]()
    var sn = List[Float32]()
    for _h in range(heads):
        for k in range(half):
            var inv = fexp(-log_theta * Float32(2 * k) / Float32(dh))
            var ang = Float32(pos) * inv
            c.append(fcos(ang))
            sn.append(fsin(ang))
    var ct = Tensor.from_host(c, [heads * half], STDtype.BF16, ctx)
    var st = Tensor.from_host(sn, [heads * half], STDtype.BF16, ctx)
    return RopeRow(ct^, st^)


def _argmax(logits: Tensor, ctx: DeviceContext) raises -> Int:
    var h = logits.to_host(ctx)
    var best = 0
    var bv = h[0]
    for i in range(1, len(h)):
        if h[i] > bv:
            bv = h[i]
            best = i
    return best


def _s(a: Int, b: Int) -> Float64:
    return Float64(Int(b) - Int(a)) / 1.0e9


def _read_text_file(path: String) raises -> String:
    with open(path, "r") as f:
        return f.read()


def main() raises:
    var a = argv()
    # Two CLI forms:
    #   legacy:   qwen3vl_caption <image> [prompt] [max_new]
    #   director: qwen3vl_caption --prompt-file=P [--system-file=S] [--image=I]
    #             [--max-new=N]   (text-only when --image is absent; the H3
    #             Director pass uses this with the serenity.h3.caption.v2
    #             system prompt and a JSON response contract)
    var img_path = String("")
    var prompt = String("Describe this image in detail.")
    var system_prompt = String("")
    var max_new = 200
    var flag_mode = len(a) >= 2 and String(a[1]).startswith("--")
    if flag_mode:
        var prompt_given = False
        for i in range(1, len(a)):
            var arg = String(a[i])
            var fields = arg.split("=")
            if len(fields) < 2:
                raise Error(String("flag needs a value: ") + arg)
            var value = String(fields[1])
            for k in range(2, len(fields)):
                value += "=" + String(fields[k])
            if arg.startswith("--prompt-file="):
                prompt = _read_text_file(value)
                prompt_given = True
            elif arg.startswith("--system-file="):
                system_prompt = _read_text_file(value)
            elif arg.startswith("--image="):
                img_path = value
            elif arg.startswith("--max-new="):
                max_new = atol(value)
            else:
                raise Error(String("unknown flag: ") + arg)
        if not prompt_given:
            raise Error("--prompt-file is required in flag mode")
    else:
        if len(a) < 2:
            raise Error("usage: qwen3vl_caption <image> [prompt] [max_new] | "
                        "--prompt-file=P [--system-file=S] [--image=I] [--max-new=N]")
        img_path = String(a[1])
        if len(a) >= 3:
            prompt = String(a[2])
        if len(a) >= 4:
            max_new = Int(String(a[3]))
    if max_new < 1 or max_new > 4096:
        raise Error("max_new must be 1..4096")
    var has_image = img_path != String("")
    var ctx = DeviceContext()
    var t0 = perf_counter_ns()
    # 1) vision preprocess + 2) vision tower (only when an image is supplied)
    var grid_h = 0
    var grid_w = 0
    var nvis = 0
    var vout: Optional[VisionOutput] = None
    if has_image:
        var vp = lingbot_vision_preprocess_bucketed(img_path, ctx)
        print("[caption] bucket grid:", vp.grid_h, "x", vp.grid_w, " (S =", vp.seq, ")")
        var vision = Qwen3VLVisionModel.load(SNAP, ctx)
        if vp.seq == 1024:
            vout = Optional[VisionOutput](vision.forward[1024](vp.pixel_values, vp.grid_t, vp.grid_h, vp.grid_w, ctx))
        elif vp.seq == 1536:
            vout = Optional[VisionOutput](vision.forward[1536](vp.pixel_values, vp.grid_t, vp.grid_h, vp.grid_w, ctx))
        elif vp.seq == 4096:
            vout = Optional[VisionOutput](vision.forward[4096](vp.pixel_values, vp.grid_t, vp.grid_h, vp.grid_w, ctx))
        elif vp.seq == 6144:
            vout = Optional[VisionOutput](vision.forward[6144](vp.pixel_values, vp.grid_t, vp.grid_h, vp.grid_w, ctx))
        elif vp.seq == 9216:
            vout = Optional[VisionOutput](vision.forward[9216](vp.pixel_values, vp.grid_t, vp.grid_h, vp.grid_w, ctx))
        elif vp.seq == 16384:
            vout = Optional[VisionOutput](vision.forward[16384](vp.pixel_values, vp.grid_t, vp.grid_h, vp.grid_w, ctx))
        else:
            raise Error(String("unsupported vision bucket S=") + String(vp.seq))
        grid_h = vp.grid_h
        grid_w = vp.grid_w
        nvis = (vp.grid_h // 2) * (vp.grid_w // 2)   # merged vision tokens
    var t_vis = perf_counter_ns()
    # 3) LM + tokenizer + fused token stream (chat template: optional system
    #    turn, user turn with optional vision block, assistant header)
    var enc = load_krea2_qwen3vl_4b(String(SNAP), ctx)
    var tok = Qwen3Tokenizer(TOKJSON)
    var ids = List[Int]()
    var has_system = system_prompt != String("")
    if has_system:
        ids = tok.encode(String("<|im_start|>system\n") + system_prompt
                         + "<|im_end|>\n<|im_start|>user\n")
    else:
        ids = tok.encode(String("<|im_start|>user\n"))
    if has_image:
        var vs_ids = tok.encode(String("<|vision_start|>"))
        for i in range(len(vs_ids)):
            ids.append(vs_ids[i])
        for _i in range(nvis):
            ids.append(IMG_PAD)
        var ve_ids = tok.encode(String("<|vision_end|>"))
        for i in range(len(ve_ids)):
            ids.append(ve_ids[i])
    var tail_ids = tok.encode(prompt + "<|im_end|>\n<|im_start|>assistant\n")
    for i in range(len(tail_ids)):
        ids.append(tail_ids[i])
    var L = len(ids)
    var npad = 0
    for i in range(L):
        if ids[i] == IMG_PAD:
            npad += 1
    print("[caption] prefix tokens:", L, " image_pad:", npad, " nvis:", nvis,
          " system:", has_system)
    if npad != nvis:
        raise Error(String("token stream image_pad count ") + String(npad)
                    + " != nvis " + String(nvis))
    var t_load = perf_counter_ns()
    var cfg = enc.config
    var half = cfg.head_dim // 2
    var hidden_sz = cfg.hidden_size
    var cache = KVCache(cfg.num_layers)
    var last_logits = Tensor.from_host([Float32(0.0)], [1], STDtype.BF16, ctx)
    var next_pos = L
    if has_image:
        # 3D interleaved M-RoPE tables over the whole prefix (parity-gated builder).
        var pos = _build_positions(ids, L, L, grid_h, grid_w)
        var q_tab = _build_mrope_tables(pos, L, cfg.num_heads, cfg.head_dim, cfg.rope_theta)
        var k_tab = _build_mrope_tables(pos, L, cfg.num_kv_heads, cfg.head_dim, cfg.rope_theta)
        var vis_start = pos.vis_start
        ref vo = vout.value()
        for i in range(L):
            var qr = _rope_row(q_tab[0], q_tab[1], i, cfg.num_heads, half, ctx)
            var kr = _rope_row(k_tab[0], k_tab[1], i, cfg.num_kv_heads, half, ctx)
            var want = i == L - 1
            if ids[i] == IMG_PAD:
                var vi = i - vis_start
                var emb = _row(vo.pooler, vi, hidden_sz, ctx)
                var d0 = Optional[Tensor](_row(vo.ds0, vi, hidden_sz, ctx))
                var d1 = Optional[Tensor](_row(vo.ds1, vi, hidden_sz, ctx))
                var d2 = Optional[Tensor](_row(vo.ds2, vi, hidden_sz, ctx))
                last_logits = vl_decode_step_embed(
                    enc, cache, emb^, qr.c, qr.s, kr.c, kr.s,
                    d0^, d1^, d2^, ctx, want_logits=want,
                )
            else:
                var toks = List[Int]()
                toks.append(ids[i])
                var emb = enc._embed(toks, ctx)
                last_logits = vl_decode_step_embed(
                    enc, cache, emb^, qr.c, qr.s, kr.c, kr.s,
                    Optional[Tensor](None), Optional[Tensor](None), Optional[Tensor](None),
                    ctx, want_logits=want,
                )
        next_pos = pos.pt[L - 1] + 1
    else:
        # Text-only: Qwen3-VL positions are equal-axis (t=h=w=i) -> plain RoPE.
        for i in range(L):
            var qr = _plain_rope_row(i, cfg.num_heads, cfg.head_dim, cfg.rope_theta, ctx)
            var kr = _plain_rope_row(i, cfg.num_kv_heads, cfg.head_dim, cfg.rope_theta, ctx)
            var toks = List[Int]()
            toks.append(ids[i])
            var emb = enc._embed(toks, ctx)
            last_logits = vl_decode_step_embed(
                enc, cache, emb^, qr.c, qr.s, kr.c, kr.s,
                Optional[Tensor](None), Optional[Tensor](None), Optional[Tensor](None),
                ctx, want_logits=(i == L - 1),
            )
        next_pos = L
    var t_prime = perf_counter_ns()
    # 5) greedy generation: text positions t=h=w=next_pos (equal-axis mrope ==
    #    plain rope — built via the SAME table builder for exactness).
    var gen = List[Int]()
    for _g in range(max_new):
        var best = _argmax(last_logits, ctx)
        if best == EOS:
            break
        gen.append(best)
        var qr = _plain_rope_row(next_pos, cfg.num_heads, cfg.head_dim, cfg.rope_theta, ctx)
        var kr = _plain_rope_row(next_pos, cfg.num_kv_heads, cfg.head_dim, cfg.rope_theta, ctx)
        var toks = List[Int]()
        toks.append(best)
        var emb = enc._embed(toks, ctx)
        last_logits = vl_decode_step_embed(
            enc, cache, emb^, qr.c, qr.s, kr.c, kr.s,
            Optional[Tensor](None), Optional[Tensor](None), Optional[Tensor](None),
            ctx, want_logits=True,
        )
        next_pos += 1
    var t_gen = perf_counter_ns()
    print("=== CAPTION ===")
    print(tok.decode(gen))
    print("=== END ===")
    print("timing: vision", _s(t0, t_vis), "s | LM load", _s(t_vis, t_load),
          "s | prime(", L, "rows)", _s(t_load, t_prime),
          "s | generate(", len(gen), "tok)", _s(t_prime, t_gen), "s")
