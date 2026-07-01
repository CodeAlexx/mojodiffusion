# ZImage OneTrainer train-ref selected-gradient replay preflight.
#
# This consumes the real OneTrainer step0 and adapter dumps, verifies the selected
# layer gradient target tensors, and computes the exact padded sequence geometry
# produced by diffusers ZImageTransformer2DModel for the dumped batch.
#
# It intentionally does not compare adapter gradients yet. The dumped batch has
# per-sample caption lengths 145 and 127, which become 160 and 128 after the
# model's pad-to-32 step. Diffusers then pads the batch to the max sequence length
# and passes an attention mask. The non-graph Mojo B=2 stack now has the required
# cap/unified key-tail mask plumbing; this gate remains a preflight until a
# BF16-boundary Mojo replay is wired through full forward/backward and compared
# against the adapter-dump host comparison tensors.

from std.builtin.dtype import DType
from std.collections import List
from std.gpu.host import DeviceContext
from std.pathlib import Path
from std.sys.defines import get_defined_int

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.ops.tensor_algebra import reshape
from serenitymojo.models.zimage.real_weights import (
    ZImageRealAux,
    load_zimage_real_aux,
    build_x_seq,
    build_cap_seq,
    build_positions,
    build_rope,
    build_adaln,
    build_block_modvecs,
    build_f_scale,
)
from serenitymojo.models.zimage.weights import (
    ZImageBlockWeights, load_zimage_block_weights_prefixed_mixed,
)
from serenitymojo.models.zimage.block import ZImageModVecs
from serenitymojo.models.zimage.lora_block import (
    ZImageBlockLoraDevice, ZImageLoraAdapterDevice,
    ZIMAGE_SLOTS,
    zimage_lora_adapter_to_device,
    zimage_modvecs_pack2_to_device,
)
from serenitymojo.models.zimage.zimage_stack_lora import (
    ZImageLoraDeviceSet,
    ZImageModVecsDevice,
    zimage_stack_lora_forward_main_device_b2_masked_streamed,
    zimage_stack_lora_backward_main_device_b2_masked_streamed,
)
from serenitymojo.tensor import Tensor
from serenitymojo.training.train_step import LoraAdapter


comptime META_JSON = "/home/alex/serenity-trainer/parity/zimage_train_ref_meta.json"
comptime STEP_DUMP = "/home/alex/serenity-trainer/parity/zimage_train_ref_step000.safetensors"
comptime STEP0_ADAPTERS = "/home/alex/serenity-trainer/parity/zimage_train_ref_step000_adapters.safetensors"
comptime TRANSFORMER_DIR = "/home/alex/.serenity/models/zimage_base/transformer"
comptime FORWARD_PHASE_PREFIX = "adapter_before."
comptime GRAD_PHASE_PREFIX = "adapter_post_clip_grad."
comptime SELECTED_LAYER = 0
comptime BATCH = 2
comptime LAT_C = 16
comptime LAT_H = 64
comptime LAT_W = 64
comptime PATCH = 2
comptime CAP_DIM = 2560
comptime OUT_CH = LAT_C * PATCH * PATCH
comptime H = 30
comptime DH = 128
comptime D_MODEL = 3840
comptime F_MODEL = 10240
comptime RANK = 16
comptime NUM_NR = 2
comptime NUM_CR = 2
comptime MAIN_DEPTH = 30
comptime ADALN_DIM = 256
comptime T_SCALE = Float32(1000.0)
comptime ROPE_THETA = Float32(256.0)
comptime AXIS0 = 32
comptime AXIS1 = 48
comptime AXIS2 = 48
comptime EPS = Float32(1.0e-5)
comptime FINAL_EPS = Float32(1.0e-6)
comptime VALID_CAP0 = 145
comptime VALID_CAP1 = 127
comptime IMG_ROWS = (LAT_H // PATCH) * (LAT_W // PATCH)
comptime CAP_BUCKET = 160
comptime SEQ_BUCKET = IMG_ROWS + CAP_BUCKET
comptime EXPECTED_STEP_TENSORS = 42
comptime EXPECTED_ADAPTER_TENSORS = 3360
comptime EXPECTED_SELECTED_TENSORS = 14
comptime EXPECTED_SELECTED_NUMEL = 1167360
comptime EXPECTED_ALL_GRAD_TENSORS = MAIN_DEPTH * EXPECTED_SELECTED_TENSORS
comptime EXPECTED_ALL_GRAD_NUMEL = MAIN_DEPTH * EXPECTED_SELECTED_NUMEL
comptime SELECTED_REPLAY_SCALE = 1.0 / 16.0
comptime SELECTED_GRAD_MAX_ABS_TOL = Float32(1.0e-5)
comptime SELECTED_REPLAY_MAIN_DEPTH = get_defined_int["ZIMAGE_SELECTED_REPLAY_MAIN_DEPTH", 0]()


def _require(ok: Bool, msg: String) raises:
    if not ok:
        raise Error(msg)


def _shape0() -> List[Int]:
    return List[Int]()


def _shape1(a: Int) -> List[Int]:
    var out = List[Int]()
    out.append(a)
    return out^


def _shape2(a: Int, b: Int) -> List[Int]:
    var out = List[Int]()
    out.append(a)
    out.append(b)
    return out^


def _shape4(a: Int, b: Int, c: Int, d: Int) -> List[Int]:
    var out = List[Int]()
    out.append(a)
    out.append(b)
    out.append(c)
    out.append(d)
    return out^


def _shape5(a: Int, b: Int, c: Int, d: Int, e: Int) -> List[Int]:
    var out = List[Int]()
    out.append(a)
    out.append(b)
    out.append(c)
    out.append(d)
    out.append(e)
    return out^


def _same_shape(a: List[Int], b: List[Int]) -> Bool:
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if a[i] != b[i]:
            return False
    return True


def _shape_text(shape: List[Int]) -> String:
    var out = String("[")
    for i in range(len(shape)):
        if i > 0:
            out += String(",")
        out += String(shape[i])
    out += String("]")
    return out^


def _numel(shape: List[Int]) -> Int:
    if len(shape) == 0:
        return 1
    var n = 1
    for i in range(len(shape)):
        n *= shape[i]
    return n


def _abs(x: Float32) -> Float32:
    if x < Float32(0.0):
        return -x
    return x


def _require_tensor(
    st: SafeTensors, key: String, dtype: STDtype, expected_shape: List[Int]
) raises:
    _require(key in st.tensors, String("missing ZImage train-ref tensor ") + key)
    var info = st.tensor_info(key)
    _require(
        info.dtype == dtype,
        String("dtype mismatch for ")
        + key
        + String(" got=")
        + info.dtype.name()
        + String(" expected=")
        + dtype.name(),
    )
    _require(
        _same_shape(info.shape, expected_shape),
        String("shape mismatch for ")
        + key
        + String(" got=")
        + _shape_text(info.shape)
        + String(" expected=")
        + _shape_text(expected_shape),
    )
    _require(
        info.size == _numel(expected_shape) * dtype.byte_size(),
        String("byte-size mismatch for ") + key,
    )


def _load_step_tensor_preserve(
    st: SafeTensors, key: String, expected_dtype: STDtype, ctx: DeviceContext
) raises -> Tensor:
    var info = st.tensor_info(key)
    _require(
        info.dtype == expected_dtype,
        String("device ingest dtype mismatch for ")
        + key
        + String(" got=")
        + info.dtype.name()
        + String(" expected=")
        + expected_dtype.name(),
    )
    var bytes = st.tensor_bytes(key)
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    var t = Tensor.from_view(tv, ctx)
    _require(
        t.dtype() == expected_dtype,
        String("device tensor dtype boundary changed for ") + key,
    )
    return t^


def _zeros(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for _ in range(n):
        out.append(Float32(0.0))
    return out^


def _read_f32_matrix(
    st: SafeTensors, key: String, rows: Int, cols: Int
) raises -> List[Float32]:
    _ = _require_adapter_matrix(st, key, rows, cols)
    var info = st.tensor_info(key)
    var bytes = st.tensor_bytes(key)
    var fp = bytes.unsafe_ptr().bitcast[Float32]()
    var out = List[Float32]()
    for i in range(info.size // 4):
        out.append(fp[i])
    return out^


def _read_f32_value(st: SafeTensors, key: String, index: Int) raises -> Float32:
    var info = st.tensor_info(key)
    _require(info.dtype == STDtype.F32, String("expected F32 scalar/vector tensor ") + key)
    _require(index >= 0 and index < info.size // 4, String("F32 index out of bounds for ") + key)
    var bytes = st.tensor_bytes(key)
    var fp = bytes.unsafe_ptr().bitcast[Float32]()
    return fp[index]


def _read_bf16_value(st: SafeTensors, key: String, index: Int) raises -> Float32:
    var info = st.tensor_info(key)
    _require(info.dtype == STDtype.BF16, String("expected BF16 tensor ") + key)
    _require(index >= 0 and index < info.size // 2, String("BF16 index out of bounds for ") + key)
    var bytes = st.tensor_bytes(key)
    var bp = bytes.unsafe_ptr().bitcast[BFloat16]()
    return bp[index].cast[DType.float32]()


def _patchified_value(
    st: SafeTensors, key: String, sample: Int, row: Int, col: Int
) raises -> Float32:
    var ih = row // (LAT_W // PATCH)
    var iw = row - ih * (LAT_W // PATCH)
    var phpw = col // LAT_C
    var c = col - phpw * LAT_C
    var ph = phpw // PATCH
    var pw = phpw - ph * PATCH
    var h = ih * PATCH + ph
    var w = iw * PATCH + pw
    var idx = (
        sample * LAT_C * LAT_H * LAT_W
        + c * LAT_H * LAT_W
        + h * LAT_W
        + w
    )
    return _read_bf16_value(st, key, idx)


def _flow_d_out(st: SafeTensors, sample: Int) raises -> List[Float32]:
    # Loss is MSE(-raw_velocity, flow_target). The stack backward consumes
    # dL/d(raw_velocity), so d_out = (target - predicted_flow) * 2 / batch_numel.
    var out = List[Float32]()
    var scale = Float32(2.0) / Float32(BATCH * IMG_ROWS * OUT_CH)
    for row in range(IMG_ROWS):
        for col in range(OUT_CH):
            var target = _patchified_value(st, String("flow_target"), sample, row, col)
            var pred = _patchified_value(st, String("predicted_flow"), sample, row, col)
            out.append((target - pred) * scale)
    return out^


def _compare_vec_max_abs(
    got: List[Float32], expected: List[Float32], label: String
) raises -> Float32:
    _require(len(got) == len(expected), String("grad compare length mismatch for ") + label)
    var max_abs = Float32(0.0)
    for i in range(len(got)):
        var ae = _abs(got[i] - expected[i])
        if ae > max_abs:
            max_abs = ae
    return max_abs


def _compare_selected_layer0_grads(
    adapters: SafeTensors, grads_a: List[List[Float32]], grads_b: List[List[Float32]]
) raises -> Float32:
    _require(len(grads_a) >= 7 and len(grads_b) >= 7, String("full selected replay missing layer0 grads"))
    var max_abs = Float32(0.0)
    for slot in range(7):
        var in_f = _slot_in_f(slot)
        var out_f = _slot_out_f(slot)
        var exp_a = _read_f32_matrix(
            adapters,
            _adapter_key_for_layer(String(GRAD_PHASE_PREFIX), SELECTED_LAYER, slot, True),
            RANK,
            in_f,
        )
        var exp_b = _read_f32_matrix(
            adapters,
            _adapter_key_for_layer(String(GRAD_PHASE_PREFIX), SELECTED_LAYER, slot, False),
            out_f,
            RANK,
        )
        var a_err = _compare_vec_max_abs(grads_a[slot], exp_a, _adapter_key(String(GRAD_PHASE_PREFIX), slot, True))
        var b_err = _compare_vec_max_abs(grads_b[slot], exp_b, _adapter_key(String(GRAD_PHASE_PREFIX), slot, False))
        if a_err > max_abs:
            max_abs = a_err
        if b_err > max_abs:
            max_abs = b_err
    return max_abs


def _compare_all_main_grads(
    adapters: SafeTensors, grads_a: List[List[Float32]], grads_b: List[List[Float32]]
) raises -> Float32:
    _require(
        len(grads_a) == MAIN_DEPTH * ZIMAGE_SLOTS and len(grads_b) == MAIN_DEPTH * ZIMAGE_SLOTS,
        String("full selected replay missing all main grads"),
    )
    var max_abs = Float32(0.0)
    var compared_tensors = 0
    var compared_numel = 0
    for layer in range(MAIN_DEPTH):
        for slot in range(ZIMAGE_SLOTS):
            var flat = layer * ZIMAGE_SLOTS + slot
            var in_f = _slot_in_f(slot)
            var out_f = _slot_out_f(slot)
            var exp_a = _read_f32_matrix(
                adapters,
                _adapter_key_for_layer(String(GRAD_PHASE_PREFIX), layer, slot, True),
                RANK,
                in_f,
            )
            var exp_b = _read_f32_matrix(
                adapters,
                _adapter_key_for_layer(String(GRAD_PHASE_PREFIX), layer, slot, False),
                out_f,
                RANK,
            )
            var a_err = _compare_vec_max_abs(
                grads_a[flat],
                exp_a,
                _adapter_key_for_layer(String(GRAD_PHASE_PREFIX), layer, slot, True),
            )
            var b_err = _compare_vec_max_abs(
                grads_b[flat],
                exp_b,
                _adapter_key_for_layer(String(GRAD_PHASE_PREFIX), layer, slot, False),
            )
            if a_err > max_abs:
                max_abs = a_err
            if b_err > max_abs:
                max_abs = b_err
            compared_tensors += 2
            compared_numel += len(exp_a) + len(exp_b)
    _require(
        compared_tensors == EXPECTED_ALL_GRAD_TENSORS,
        String("full selected replay compared tensor count mismatch"),
    )
    _require(
        compared_numel == EXPECTED_ALL_GRAD_NUMEL,
        String("full selected replay compared numel mismatch"),
    )
    return max_abs


def _slot_suffix(slot: Int) raises -> String:
    if slot == 0:
        return String("attention.to_q")
    if slot == 1:
        return String("attention.to_k")
    if slot == 2:
        return String("attention.to_v")
    if slot == 3:
        return String("attention.to_out.0")
    if slot == 4:
        return String("feed_forward.w1")
    if slot == 5:
        return String("feed_forward.w3")
    if slot == 6:
        return String("feed_forward.w2")
    raise Error(String("bad ZImage LoRA slot ") + String(slot))


def _slot_in_f(slot: Int) raises -> Int:
    if slot == 6:
        return F_MODEL
    if slot >= 0 and slot <= 5:
        return D_MODEL
    raise Error(String("bad ZImage LoRA slot ") + String(slot))


def _slot_out_f(slot: Int) raises -> Int:
    if slot == 4 or slot == 5:
        return F_MODEL
    if slot >= 0 and slot <= 3:
        return D_MODEL
    if slot == 6:
        return D_MODEL
    raise Error(String("bad ZImage LoRA slot ") + String(slot))


def _adapter_key_for_layer(
    phase_prefix: String, layer: Int, slot: Int, down: Bool
) raises -> String:
    var key = phase_prefix
    key += String("transformer.layers.")
    key += String(layer)
    key += String(".")
    key += _slot_suffix(slot)
    if down:
        key += String(".lora_down.weight")
    else:
        key += String(".lora_up.weight")
    return key^


def _adapter_key(phase_prefix: String, slot: Int, down: Bool) raises -> String:
    return _adapter_key_for_layer(phase_prefix, SELECTED_LAYER, slot, down)


def _require_adapter_matrix(
    st: SafeTensors, key: String, rows: Int, cols: Int
) raises -> Int:
    _require(key in st.tensors, String("missing OT adapter tensor ") + key)
    var info = st.tensor_info(key)
    _require(info.dtype == STDtype.F32, String("expected F32 adapter tensor ") + key)
    _require(
        _same_shape(info.shape, _shape2(rows, cols)),
        String("shape mismatch for ")
        + key
        + String(" got=")
        + _shape_text(info.shape)
        + String(" expected=")
        + _shape_text(_shape2(rows, cols)),
    )
    _require(info.size == rows * cols * 4, String("byte-size mismatch for ") + key)
    return rows * cols


def _check_selected_layer_targets(st: SafeTensors) raises -> Int:
    var total = 0
    for slot in range(7):
        var in_f = _slot_in_f(slot)
        var out_f = _slot_out_f(slot)
        total += _require_adapter_matrix(
            st, _adapter_key(String(FORWARD_PHASE_PREFIX), slot, True), RANK, in_f
        )
        total += _require_adapter_matrix(
            st, _adapter_key(String(FORWARD_PHASE_PREFIX), slot, False), out_f, RANK
        )
        total += _require_adapter_matrix(
            st, _adapter_key(String(GRAD_PHASE_PREFIX), slot, True), RANK, in_f
        )
        total += _require_adapter_matrix(
            st, _adapter_key(String(GRAD_PHASE_PREFIX), slot, False), out_f, RANK
        )
    # `total` counts forward and grad phase matrices. The selected layer's trainable
    # gradient target surface is half of this: 14 matrices / 1,167,360 elements.
    return total // 2


def _adapter_from_dump(st: SafeTensors, slot: Int) raises -> LoraAdapter:
    return _adapter_from_dump_layer(st, SELECTED_LAYER, slot)


def _adapter_from_dump_layer(
    st: SafeTensors, layer: Int, slot: Int
) raises -> LoraAdapter:
    var in_f = _slot_in_f(slot)
    var out_f = _slot_out_f(slot)
    var a = _read_f32_matrix(
        st, _adapter_key_for_layer(String(FORWARD_PHASE_PREFIX), layer, slot, True), RANK, in_f
    )
    var b = _read_f32_matrix(
        st, _adapter_key_for_layer(String(FORWARD_PHASE_PREFIX), layer, slot, False), out_f, RANK
    )
    # The OneTrainer dump stores live LoRA tensors as F32 host comparison data.
    # LoraAdapter converts A/B to BF16 model storage before the device upload.
    return LoraAdapter(
        a^, b^, RANK, in_f, out_f, Float32(SELECTED_REPLAY_SCALE),
        _zeros(RANK * in_f), _zeros(RANK * in_f),
        _zeros(out_f * RANK), _zeros(out_f * RANK),
    )


def _selected_layer_lora_device(
    st: SafeTensors, ctx: DeviceContext
) raises -> ZImageBlockLoraDevice:
    return ZImageBlockLoraDevice(
        zimage_lora_adapter_to_device(_adapter_from_dump(st, 0), ctx),
        zimage_lora_adapter_to_device(_adapter_from_dump(st, 1), ctx),
        zimage_lora_adapter_to_device(_adapter_from_dump(st, 2), ctx),
        zimage_lora_adapter_to_device(_adapter_from_dump(st, 3), ctx),
        zimage_lora_adapter_to_device(_adapter_from_dump(st, 4), ctx),
        zimage_lora_adapter_to_device(_adapter_from_dump(st, 5), ctx),
        zimage_lora_adapter_to_device(_adapter_from_dump(st, 6), ctx),
    )


def _main_lora_device_set_from_dump(
    st: SafeTensors, depth: Int, ctx: DeviceContext
) raises -> ZImageLoraDeviceSet:
    var ad = List[ZImageLoraAdapterDevice]()
    for layer in range(depth):
        for slot in range(7):
            ad.append(zimage_lora_adapter_to_device(_adapter_from_dump_layer(st, layer, slot), ctx))
    return ZImageLoraDeviceSet(ad^, 0, 0, depth, RANK)


def _require_device_adapter_bf16(
    ad: ZImageLoraAdapterDevice, slot_name: String
) raises:
    _require(
        ad.a[].dtype() == STDtype.BF16,
        String("selected replay device LoRA A must be BF16 for ") + slot_name,
    )
    _require(
        ad.b[].dtype() == STDtype.BF16,
        String("selected replay device LoRA B must be BF16 for ") + slot_name,
    )


def _require_selected_layer_device_bf16(block: ZImageBlockLoraDevice) raises:
    _require_device_adapter_bf16(block.to_q, String("attention.to_q"))
    _require_device_adapter_bf16(block.to_k, String("attention.to_k"))
    _require_device_adapter_bf16(block.to_v, String("attention.to_v"))
    _require_device_adapter_bf16(block.to_out, String("attention.to_out.0"))
    _require_device_adapter_bf16(block.w1, String("feed_forward.w1"))
    _require_device_adapter_bf16(block.w3, String("feed_forward.w3"))
    _require_device_adapter_bf16(block.w2, String("feed_forward.w2"))


def _require_device_tensor_bf16(t: Tensor, name: String) raises:
    _require(
        t.dtype() == STDtype.BF16,
        String("selected replay base weight must load as BF16 for ") + name,
    )


def _require_selected_base_block_bf16(w: ZImageBlockWeights) raises:
    _require_device_tensor_bf16(w.n1[], String("attention_norm1"))
    _require_device_tensor_bf16(w.wq[], String("attention.to_q"))
    _require_device_tensor_bf16(w.wk[], String("attention.to_k"))
    _require_device_tensor_bf16(w.wv[], String("attention.to_v"))
    _require_device_tensor_bf16(w.wo[], String("attention.to_out.0"))
    _require_device_tensor_bf16(w.q_norm[], String("attention.norm_q"))
    _require_device_tensor_bf16(w.k_norm[], String("attention.norm_k"))
    _require_device_tensor_bf16(w.n2[], String("attention_norm2"))
    _require_device_tensor_bf16(w.fn1[], String("ffn_norm1"))
    _require_device_tensor_bf16(w.w1[], String("feed_forward.w1"))
    _require_device_tensor_bf16(w.w3[], String("feed_forward.w3"))
    _require_device_tensor_bf16(w.w2[], String("feed_forward.w2"))
    _require_device_tensor_bf16(w.fn2[], String("ffn_norm2"))


def _cap_padded(valid_cap: Int) -> Int:
    return ((valid_cap + 31) // 32) * 32


def _unified_positions(
    x_pos: List[List[Int]], cap_pos: List[List[Int]]
) -> List[List[Int]]:
    var out = List[List[Int]]()
    for i in range(len(x_pos)):
        out.append(x_pos[i].copy())
    for i in range(len(cap_pos)):
        out.append(cap_pos[i].copy())
    return out^


def _cap_seq_with_pad(
    aux: ZImageRealAux, cap_feats: Tensor, valid_cap: Int, cap_len: Int,
    cap_pad_h: List[Float32], ctx: DeviceContext,
) raises -> List[Float32]:
    var seq = build_cap_seq(aux, cap_feats, EPS, ctx)
    _require(len(seq) == valid_cap * D_MODEL, String("real cap embed length mismatch"))
    _require(len(cap_pad_h) == D_MODEL, String("cap_pad_token host width mismatch"))
    for _r in range(valid_cap, cap_len):
        for c in range(D_MODEL):
            seq.append(cap_pad_h[c])
    _require(len(seq) == cap_len * D_MODEL, String("cap_seq padded length mismatch"))
    return seq^


def _load_trace_latent_as_nchw(
    st: SafeTensors, key: String, ctx: DeviceContext
) raises -> Tensor:
    var latent_cf = _load_step_tensor_preserve(st, key, STDtype.BF16, ctx)
    var shaped = reshape(latent_cf, _shape4(1, LAT_C, LAT_H, LAT_W), ctx)
    _require(shaped.dtype() == STDtype.BF16, String("latent reshape widened dtype for ") + key)
    return shaped^


def _update_min_free(ctx: DeviceContext, min_free: Int) raises -> Int:
    var mem = ctx.get_memory_info()
    var free_now = Int(mem[0])
    if free_now < min_free:
        return free_now
    return min_free


def _peak_vram_mib(total_vram: Int, min_free: Int) -> Float64:
    return Float64(total_vram - min_free) / 1048576.0


def _run_real_streamed_input_smoke(
    step: SafeTensors, adapters: SafeTensors, transformer: ShardedSafeTensors,
    ctx: DeviceContext,
) raises -> List[Float64]:
    if SELECTED_REPLAY_MAIN_DEPTH != 0 and SELECTED_REPLAY_MAIN_DEPTH != MAIN_DEPTH:
        raise Error("ZIMAGE_SELECTED_REPLAY_MAIN_DEPTH must be 0 or 30")
    var mem0 = ctx.get_memory_info()
    var min_free = Int(mem0[0])
    var total_vram = Int(mem0[1])

    var aux = load_zimage_real_aux(transformer, NUM_NR, MAIN_DEPTH, ctx)
    min_free = _update_min_free(ctx, min_free)
    var cap_pad_h = aux.cap_pad_token[].to_host(ctx)

    var latent0 = _load_trace_latent_as_nchw(step, String("trace.latent_input_list.0"), ctx)
    var latent1 = _load_trace_latent_as_nchw(step, String("trace.latent_input_list.1"), ctx)
    var x_seq0 = build_x_seq(aux, latent0, LAT_C, LAT_H, LAT_W, PATCH, ctx)
    var x_seq1 = build_x_seq(aux, latent1, LAT_C, LAT_H, LAT_W, PATCH, ctx)
    _require(len(x_seq0) == IMG_ROWS * D_MODEL, String("x_seq0 length mismatch"))
    _require(len(x_seq1) == IMG_ROWS * D_MODEL, String("x_seq1 length mismatch"))
    min_free = _update_min_free(ctx, min_free)

    var cap0_t = _load_step_tensor_preserve(step, String("text_encoder_output_0"), STDtype.BF16, ctx)
    var cap1_t = _load_step_tensor_preserve(step, String("text_encoder_output_1"), STDtype.BF16, ctx)
    var cap0 = _cap_padded(VALID_CAP0)
    var cap1 = _cap_padded(VALID_CAP1)
    var max_cap = cap0 if cap0 >= cap1 else cap1
    _require(max_cap == CAP_BUCKET, String("selected replay cap bucket changed"))
    var cap_seq0 = _cap_seq_with_pad(aux, cap0_t, VALID_CAP0, max_cap, cap_pad_h, ctx)
    var cap_seq1 = _cap_seq_with_pad(aux, cap1_t, VALID_CAP1, max_cap, cap_pad_h, ctx)
    min_free = _update_min_free(ctx, min_free)

    var pos0 = build_positions(IMG_ROWS, LAT_H // PATCH, LAT_W // PATCH, max_cap, VALID_CAP0)
    var pos1 = build_positions(IMG_ROWS, LAT_H // PATCH, LAT_W // PATCH, max_cap, VALID_CAP1)
    var x_pos0 = pos0[0].copy()
    var cap_pos0 = pos0[1].copy()
    var x_pos1 = pos1[0].copy()
    var cap_pos1 = pos1[1].copy()
    var uni2_pos = _unified_positions(x_pos0, cap_pos0)
    var uni1 = _unified_positions(x_pos1, cap_pos1)
    for i in range(len(uni1)):
        uni2_pos.append(uni1[i].copy())

    var xr0 = build_rope(x_pos0, H, DH, ROPE_THETA, AXIS0, AXIS1, AXIS2, ctx)
    var xr1 = build_rope(x_pos1, H, DH, ROPE_THETA, AXIS0, AXIS1, AXIS2, ctx)
    var cr0 = build_rope(cap_pos0, H, DH, ROPE_THETA, AXIS0, AXIS1, AXIS2, ctx)
    var cr1 = build_rope(cap_pos1, H, DH, ROPE_THETA, AXIS0, AXIS1, AXIS2, ctx)
    var ur2 = build_rope(uni2_pos, H, DH, ROPE_THETA, AXIS0, AXIS1, AXIS2, ctx)
    min_free = _update_min_free(ctx, min_free)

    var t0 = _read_f32_value(step, String("trace.transformer_timestep"), 0)
    var t1 = _read_f32_value(step, String("trace.transformer_timestep"), 1)
    var adaln0 = build_adaln(aux, t0, ADALN_DIM, T_SCALE, ctx)
    var adaln1 = build_adaln(aux, t1, ADALN_DIM, T_SCALE, ctx)
    var nr_mod0 = List[ZImageModVecs]()
    var nr_mod1 = List[ZImageModVecs]()
    for i in range(NUM_NR):
        nr_mod0.append(build_block_modvecs(aux.nr_mod_w[i][], aux.nr_mod_b[i][], adaln0, D_MODEL, ctx))
        nr_mod1.append(build_block_modvecs(aux.nr_mod_w[i][], aux.nr_mod_b[i][], adaln1, D_MODEL, ctx))
    var main_mod_b2 = List[ZImageModVecsDevice]()
    for i in range(MAIN_DEPTH):
        var m0 = build_block_modvecs(aux.main_mod_w[i][], aux.main_mod_b[i][], adaln0, D_MODEL, ctx)
        var m1 = build_block_modvecs(aux.main_mod_w[i][], aux.main_mod_b[i][], adaln1, D_MODEL, ctx)
        main_mod_b2.append(zimage_modvecs_pack2_to_device(m0, m1, D_MODEL, ctx))
    var f_scale0 = build_f_scale(aux, adaln0, D_MODEL, ctx)
    var f_scale2 = f_scale0.copy()
    var f_scale1 = build_f_scale(aux, adaln1, D_MODEL, ctx)
    for i in range(D_MODEL):
        f_scale2.append(f_scale1[i])
    min_free = _update_min_free(ctx, min_free)

    var lora_dev = _main_lora_device_set_from_dump(adapters, SELECTED_REPLAY_MAIN_DEPTH, ctx)
    var fwd = zimage_stack_lora_forward_main_device_b2_masked_streamed[
        H, DH, IMG_ROWS, CAP_BUCKET, SEQ_BUCKET,
    ](
        transformer,
        x_seq0.copy(), cap_seq0.copy(),
        x_seq1.copy(), cap_seq1.copy(),
        cap0,
        cap1,
        IMG_ROWS + cap0,
        IMG_ROWS + cap1,
        NUM_NR,
        NUM_CR,
        SELECTED_REPLAY_MAIN_DEPTH,
        nr_mod0,
        nr_mod1,
        main_mod_b2,
        lora_dev,
        f_scale2.copy(),
        aux.final_lin_w[],
        aux.final_lin_b[],
        xr0[0][],
        xr0[1][],
        xr1[0][],
        xr1[1][],
        cr0[0][],
        cr0[1][],
        cr1[0][],
        cr1[1][],
        ur2[0][],
        ur2[1][],
        D_MODEL,
        F_MODEL,
        OUT_CH,
        EPS,
        FINAL_EPS,
        ctx,
    )
    _require(len(fwd.out0) == IMG_ROWS * OUT_CH, String("real streamed smoke out0 length mismatch"))
    _require(len(fwd.out1) == IMG_ROWS * OUT_CH, String("real streamed smoke out1 length mismatch"))
    min_free = _update_min_free(ctx, min_free)

    var d_out0 = _flow_d_out(step, 0)
    var d_out1 = _flow_d_out(step, 1)
    var grads = zimage_stack_lora_backward_main_device_b2_masked_streamed[
        H, DH, IMG_ROWS, CAP_BUCKET, SEQ_BUCKET,
    ](
        transformer,
        d_out0,
        d_out1,
        IMG_ROWS + cap0,
        IMG_ROWS + cap1,
        SELECTED_REPLAY_MAIN_DEPTH,
        main_mod_b2,
        lora_dev,
        f_scale2.copy(),
        aux.final_lin_w[],
        ur2[0][],
        ur2[1][],
        fwd,
        D_MODEL,
        F_MODEL,
        OUT_CH,
        EPS,
        FINAL_EPS,
        ctx,
    )
    _require(grads.nonfinite_lora_grads == 0, String("real streamed smoke nonfinite grads"))
    var selected_layer0_grad_max_abs = Float32(-1.0)
    var all_grad_max_abs = Float32(-1.0)
    var all_grad_tensor_count = 0
    var all_grad_numel = 0
    if SELECTED_REPLAY_MAIN_DEPTH == 0:
        _require(len(grads.d_a) == 0 and len(grads.d_b) == 0, String("real streamed smoke expected zero main grads"))
    else:
        selected_layer0_grad_max_abs = _compare_selected_layer0_grads(adapters, grads.d_a, grads.d_b)
        all_grad_max_abs = _compare_all_main_grads(adapters, grads.d_a, grads.d_b)
        all_grad_tensor_count = EXPECTED_ALL_GRAD_TENSORS
        all_grad_numel = EXPECTED_ALL_GRAD_NUMEL
        _require(
            all_grad_max_abs <= SELECTED_GRAD_MAX_ABS_TOL,
            String("full selected all-main grad max abs exceeds tolerance"),
        )
    min_free = _update_min_free(ctx, min_free)
    var out = List[Float64]()
    out.append(_peak_vram_mib(total_vram, min_free))
    out.append(Float64(selected_layer0_grad_max_abs))
    out.append(Float64(all_grad_max_abs))
    out.append(Float64(all_grad_tensor_count))
    out.append(Float64(all_grad_numel))
    return out^


def _check_meta_json() raises:
    var meta = Path(String(META_JSON)).read_text()
    _require(meta.byte_length() > 0, String("empty ZImage OneTrainer metadata JSON"))
    _require(meta.find(String("\"producer\": \"scripts/zimage_dump_train_ref.py\"")) >= 0, String("metadata producer mismatch"))
    _require(meta.find(String("\"model_type\": \"Z_IMAGE\"")) >= 0, String("metadata model_type mismatch"))
    _require(meta.find(String("\"training_method\": \"LORA\"")) >= 0, String("metadata training_method mismatch"))
    _require(meta.find(String("\"train_dtype\": \"BFLOAT_16\"")) >= 0, String("metadata train dtype mismatch"))
    _require(meta.find(String("\"parameter_entries\": 420")) >= 0, String("metadata trainable count mismatch"))
    _require(meta.find(String(STEP_DUMP)) >= 0, String("metadata step0 path mismatch"))
    _require(meta.find(String(STEP0_ADAPTERS)) >= 0, String("metadata step0 adapter path mismatch"))


def main() raises:
    _check_meta_json()

    var step = SafeTensors.open(String(STEP_DUMP))
    _require(step.count() == EXPECTED_STEP_TENSORS, String("expected 42 ZImage step0 tensors"))
    _require_tensor(step, String("scaled_noisy_latent_image"), STDtype.BF16, _shape4(BATCH, LAT_C, LAT_H, LAT_W))
    _require_tensor(step, String("latent_input"), STDtype.BF16, _shape5(BATCH, LAT_C, 1, LAT_H, LAT_W))
    _require_tensor(step, String("trace.latent_input_list.0"), STDtype.BF16, _shape4(LAT_C, 1, LAT_H, LAT_W))
    _require_tensor(step, String("trace.latent_input_list.1"), STDtype.BF16, _shape4(LAT_C, 1, LAT_H, LAT_W))
    _require_tensor(step, String("predicted_flow"), STDtype.BF16, _shape4(BATCH, LAT_C, LAT_H, LAT_W))
    _require_tensor(step, String("flow_target"), STDtype.BF16, _shape4(BATCH, LAT_C, LAT_H, LAT_W))
    _require_tensor(step, String("output.loss_for_backward"), STDtype.F32, _shape0())
    _require_tensor(step, String("trace.transformer_timestep"), STDtype.F32, _shape1(BATCH))
    _require_tensor(step, String("timestep"), STDtype.F32, _shape1(BATCH))
    _require_tensor(step, String("sigma"), STDtype.F32, _shape4(BATCH, 1, 1, 1))
    _require_tensor(step, String("text_encoder_output_0"), STDtype.BF16, _shape2(VALID_CAP0, CAP_DIM))
    _require_tensor(step, String("text_encoder_output_1"), STDtype.BF16, _shape2(VALID_CAP1, CAP_DIM))
    _require_tensor(step, String("text_encoder_output_batch_size"), STDtype.I64, _shape0())

    var adapters = SafeTensors.open(String(STEP0_ADAPTERS))
    _require(adapters.count() == EXPECTED_ADAPTER_TENSORS, String("expected 3360 adapter phase tensors"))
    var selected_numel = _check_selected_layer_targets(adapters)
    _require(selected_numel == EXPECTED_SELECTED_NUMEL, String("selected layer adapter numel mismatch"))

    var ctx = DeviceContext()
    _ = _load_step_tensor_preserve(step, String("latent_input"), STDtype.BF16, ctx)
    _ = _load_step_tensor_preserve(step, String("predicted_flow"), STDtype.BF16, ctx)
    _ = _load_step_tensor_preserve(step, String("flow_target"), STDtype.BF16, ctx)
    _ = _load_step_tensor_preserve(step, String("text_encoder_output_0"), STDtype.BF16, ctx)
    _ = _load_step_tensor_preserve(step, String("text_encoder_output_1"), STDtype.BF16, ctx)

    var selected_lora = _selected_layer_lora_device(adapters, ctx)
    _require_selected_layer_device_bf16(selected_lora)

    var transformer = ShardedSafeTensors.open(String(TRANSFORMER_DIR))
    var selected_block = load_zimage_block_weights_prefixed_mixed(
        transformer, String("layers.") + String(SELECTED_LAYER), ctx,
    )
    _require_selected_base_block_bf16(selected_block)

    var cap0 = _cap_padded(VALID_CAP0)
    var cap1 = _cap_padded(VALID_CAP1)
    var max_cap = cap0 if cap0 >= cap1 else cap1
    var seq0 = IMG_ROWS + cap0
    var seq1 = IMG_ROWS + cap1
    var max_seq = seq0 if seq0 >= seq1 else seq1
    var masked_cap_rows1 = max_cap - cap1
    var masked_unified_rows1 = max_seq - seq1
    _require(cap0 == 160, String("sample0 pad-to-32 caption length changed"))
    _require(cap1 == 128, String("sample1 pad-to-32 caption length changed"))
    _require(masked_cap_rows1 == 32, String("sample1 cap batch-pad rows changed"))
    _require(masked_unified_rows1 == 32, String("sample1 unified batch-pad rows changed"))

    print(
        "[zimage-selected-grad-replay] preflight PASS step_tensors=",
        EXPECTED_STEP_TENSORS,
        " adapter_tensors=",
        EXPECTED_ADAPTER_TENSORS,
        " selected_layer=",
        SELECTED_LAYER,
        " selected_tensors=",
        EXPECTED_SELECTED_TENSORS,
        " selected_numel=",
        selected_numel,
    )
    print(
        "[zimage-selected-grad-replay] exact_ot_geometry img_rows=",
        IMG_ROWS,
        " cap_valid=(",
        VALID_CAP0,
        ",",
        VALID_CAP1,
        ") cap_padded=(",
        cap0,
        ",",
        cap1,
        ") max_cap=",
        max_cap,
        " seq=(",
        seq0,
        ",",
        seq1,
        ") max_seq=",
        max_seq,
        " sample1_masked_cap_rows=",
        masked_cap_rows1,
        " sample1_masked_unified_rows=",
        masked_unified_rows1,
    )
    print(
        "[zimage-selected-grad-replay] bf16_ingest PASS step_boundary=BF16",
        " adapter_dump_dtype=F32 adapter_device_boundary=BF16",
        " selected_layer=",
        SELECTED_LAYER,
        " adapter_device_tensors=",
        EXPECTED_SELECTED_TENSORS,
        " replay_scale=",
        Float32(SELECTED_REPLAY_SCALE),
    )
    print(
        "[zimage-selected-grad-replay] base_block_ingest PASS checkpoint_boundary=BF16",
        " transformer_tensors=",
        transformer.num_tensors(),
        " selected_layer=",
        SELECTED_LAYER,
        " base_block_device_tensors=13",
        " stream_prereq=single_block_load",
    )
    var real_smoke = _run_real_streamed_input_smoke(step, adapters, transformer, ctx)
    if SELECTED_REPLAY_MAIN_DEPTH == 0:
        print(
            "[zimage-selected-grad-replay] real_streamed_input_smoke PASS",
            " evidence=real-input-bounded-smoke",
            " step_boundary=BF16 checkpoint_boundary=BF16",
            " streamed_refiner_blocks=4 streamed_main_blocks=",
            SELECTED_REPLAY_MAIN_DEPTH,
            " prepared_main_mod_b2=",
            MAIN_DEPTH,
            " depth_define=ZIMAGE_SELECTED_REPLAY_MAIN_DEPTH",
            " cap_attn_len=(",
            cap0,
            ",",
            cap1,
            ") main_attn_len=(",
            IMG_ROWS + cap0,
            ",",
            IMG_ROWS + cap1,
            ") x_rope_per_sample=true",
            " activation_carrier_dtype=F32 not_strict_BF16_step_storage",
            " observed_vram_mib_lower_bound=",
            real_smoke[0],
            " selected_layer0_grad_max_abs=",
            real_smoke[1],
            " peak_vram_bytes_missing=true",
        )
    else:
        print(
            "[zimage-selected-grad-replay] full_selected_grad_replay PASS",
            " evidence=full-depth-all-trainable-grad-replay",
            " step_boundary=BF16 checkpoint_boundary=BF16",
            " streamed_refiner_blocks=4 streamed_main_blocks=",
            SELECTED_REPLAY_MAIN_DEPTH,
            " streamed_b2_selected_replay_blocks=",
            SELECTED_REPLAY_MAIN_DEPTH,
            " streamed_b2_selected_replay_no_resident_main_blocks=true",
            " prepared_main_mod_b2=",
            MAIN_DEPTH,
            " depth_define=ZIMAGE_SELECTED_REPLAY_MAIN_DEPTH",
            " cap_attn_len=(",
            cap0,
            ",",
            cap1,
            ") main_attn_len=(",
            IMG_ROWS + cap0,
            ",",
            IMG_ROWS + cap1,
            ") x_rope_per_sample=true",
            " all_trainable_grad_tensors=",
            Int(real_smoke[3]),
            " all_trainable_grad_numel=",
            Int(real_smoke[4]),
            " all_trainable_grad_max_abs=",
            real_smoke[2],
            " selected_layer0_grad_max_abs=",
            real_smoke[1],
            " all_trainable_grad_tol=",
            SELECTED_GRAD_MAX_ABS_TOL,
            " activation_carrier_dtype=F32 not_strict_BF16_step_storage",
            " observed_vram_mib_lower_bound=",
            real_smoke[0],
            " streamed_b2_selected_replay_peak_vram_bytes_missing=true",
        )
    print(
        "[zimage-selected-grad-replay] streamed_bridge_required",
        " forward=zimage_stack_lora_forward_main_device_b2_masked_streamed",
        " backward=zimage_stack_lora_backward_main_device_b2_masked_streamed",
        " resident_masked_b2_accepted=false",
        " activation_carrier_dtype=F32 not_strict_BF16_step_storage",
        " adapter_param_boundary=BF16 adapter_grad_boundary=F32_optimizer_or_host_compare_only",
    )
    if SELECTED_REPLAY_MAIN_DEPTH == 0:
        print(
            "[zimage-selected-grad-replay] BLOCKED missing=masked_b2_streamed_forward_backward_replay_integration; non-graph masked B2 stack wiring exists, and selected step/adapters/base block now ingest with BF16 device boundaries, but the full streamed replay/comparison against adapter dump host tensors is not yet wired, so strict adapter gradient comparison is intentionally not run"
        )
    else:
        print(
            "[zimage-selected-grad-replay] BLOCKED missing=streamed_b2_selected_replay_peak_vram_bytes; full streamed all-trainable grad comparison passed, but VRAM evidence is only an in-process lower-bound sample, not a true peak monitor"
        )
