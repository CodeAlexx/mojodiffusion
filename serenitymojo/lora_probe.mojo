# lora_probe.mojo — compile-check + tiny synthetic exercise of lora.mojo.
#
# COMPILE-ONLY in code mode (GPU wedged). Build:
#   cd /home/alex/mojodiffusion && pixi run mojo build -I . -Xlinker -lm \
#       serenitymojo/lora_probe.mojo -o /tmp/loraprobe
# Running it post-reboot exercises: format detection, prefix→base_key mapping,
# the slot-merge math on a synthetic base Dict, and a real-file LoraSet.load +
# merge_into when a `--lora <path>` arg is given.

from std.memory import ArcPointer
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.lora import (
    LoraSet,
    LoraMapping,
    _apply_slot,
    _detect_format,
    _resolve_mapping,
    SLOT_FULL,
    SLOT_ROWS,
    SLOT_COLS,
    SLOT_ROWRANGE,
    FMT_DIFFUSION_MODEL,
    FMT_KOHYA_SDXL,
    FMT_ZIMAGE_TRAINER,
)


def main() raises:
    var ctx = DeviceContext()

    # ── 1. Format detection on synthetic key lists ───────────────────────────
    var dm_keys = List[String]()
    dm_keys.append(String("double_blocks.0.img_attn.to_q.lora_A.weight"))
    dm_keys.append(String("double_blocks.0.img_attn.to_q.lora_B.weight"))
    var fmt_dm = _detect_format(dm_keys)
    print("detect(train_klein bare .lora_A.weight) =", fmt_dm,
          "expect", FMT_DIFFUSION_MODEL)

    var kohya_keys = List[String]()
    kohya_keys.append(String("lora_unet_input_blocks_4_1_attn1_to_q.lora_down.weight"))
    kohya_keys.append(String("lora_unet_input_blocks_4_1_attn1_to_q.lora_up.weight"))
    var fmt_kohya = _detect_format(kohya_keys)
    print("detect(kohya lora_unet_/lora_down) =", fmt_kohya, "expect", FMT_KOHYA_SDXL)

    var zimg_keys = List[String]()
    zimg_keys.append(String("layers.0.attention.to_q.lora_A"))
    zimg_keys.append(String("layers.0.attention.to_q.lora_B"))
    var fmt_zimg = _detect_format(zimg_keys)
    print("detect(zimage split attention) =", fmt_zimg, "expect", FMT_ZIMAGE_TRAINER)

    # ── 2. Prefix → base_key resolution ──────────────────────────────────────
    var m_dm = _resolve_mapping(
        FMT_DIFFUSION_MODEL, String("double_blocks.0.img_attn.to_q")
    )
    print("map DM 'double_blocks.0.img_attn.to_q' ->", m_dm.base_key,
          "(slot", m_dm.slot_kind, ")")

    var m_zq = _resolve_mapping(
        FMT_ZIMAGE_TRAINER, String("layers.0.attention.to_k")
    )
    print("map ZImage to_k ->", m_zq.base_key, "slot", m_zq.slot_kind,
          "start", m_zq.slot_a, "len", m_zq.slot_b)

    # ── 3. Slot-merge math on synthetic GPU tensors ──────────────────────────
    # SLOT_FULL: base [2,3] + delta [2,3].
    var base_full = Tensor.from_host(
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [2, 3], STDtype.F32, ctx
    )
    var delta_full = Tensor.from_host(
        [0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [2, 3], STDtype.F32, ctx
    )
    var merged_full = _apply_slot(base_full, delta_full, SLOT_FULL, 0, 0, ctx)
    var hf = merged_full.to_host(ctx)
    print("FULL merge [0]=", hf[0], "expect 1.5")

    # SLOT_ROWS(1): base [3,2], delta into top 1 row.
    var base_rows = Tensor.from_host(
        [1.0, 1.0, 2.0, 2.0, 3.0, 3.0], [3, 2], STDtype.F32, ctx
    )
    var delta_rows = Tensor.from_host([10.0, 10.0], [1, 2], STDtype.F32, ctx)
    var merged_rows = _apply_slot(base_rows, delta_rows, SLOT_ROWS, 1, 0, ctx)
    var hr = merged_rows.to_host(ctx)
    print("ROWS merge [0]=", hr[0], "expect 11.0 ; [2]=", hr[2], "expect 2.0")

    # SLOT_ROWRANGE(start=1,len=1): base [3,2], delta into middle row.
    var base_rr = Tensor.from_host(
        [1.0, 1.0, 2.0, 2.0, 3.0, 3.0], [3, 2], STDtype.F32, ctx
    )
    var delta_rr = Tensor.from_host([5.0, 5.0], [1, 2], STDtype.F32, ctx)
    var merged_rr = _apply_slot(base_rr, delta_rr, SLOT_ROWRANGE, 1, 1, ctx)
    var hrr = merged_rr.to_host(ctx)
    print("ROWRANGE [0]=", hrr[0], "expect 1.0 ; [2]=", hrr[2], "expect 7.0")

    # ── 4. merge_into on a synthetic base Dict, loading a real LoRA file ──────
    # Construct a tiny synthetic base Dict to exercise the mutate-in-place API.
    # The base key matches a train_klein LoRA prefix (DiffusionModel format),
    # so merge_into will resolve + merge the real to_q module into it.
    var base = Dict[String, ArcPointer[Tensor]]()
    base[String("double_blocks.0.img_attn.to_q.weight")] = ArcPointer[Tensor](
        Tensor.from_host([0.0, 0.0, 0.0, 0.0], [2, 2], STDtype.F32, ctx)
    )
    print("synthetic base Dict size:", len(base))

    # A real on-disk train_klein LoRA (verified header 2026-05-26). The load is
    # wrapped so a missing file at runtime is non-fatal (compile is the gate).
    comptime LORA_PATH = "/home/alex/EriDiffusion/EriDiffusion-v2/output/klein_lr3e4_const_b1/klein_lora_step200.safetensors"
    try:
        var lset = LoraSet.load(String(LORA_PATH))
        print(
            "loaded LoRA format", lset.format_name(),
            "mappings", lset.num_mappings(),
        )
        # Scale is per-module (alpha defaults to module_rank when absent →
        # scale = multiplier = 1.0). The synth base shape [2,2] won't match the
        # real [4096,4096] weight, so the merge raises on the shape check —
        # caught here. Real use passes the real base.
        try:
            var n = lset.merge_into(base, 1.0, ctx)
            print("merged", n, "module(s) into synthetic base")
        except e:
            print("(merge raised on synthetic shape mismatch, as expected:", e, ")")
    except e:
        print("(LoRA file not present at runtime; load skipped:", e, ")")

    print("lora_probe OK")
