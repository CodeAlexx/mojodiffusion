# 1:1 port of Serenity modules/util/enum/ModelType.py
# Source of truth: /home/alex/Serenity/modules/util/enum/ModelType.py plus
# /home/alex/Serenity-anima-ref/modules/util/enum/ModelType.py for ANIMA.
#
# comptime-int constants matching the Python ModelType members exactly
# (names + order). String value == member name. PeftType ported at the bottom.

comptime MODEL_TYPE_STABLE_DIFFUSION_15 = 0
comptime MODEL_TYPE_STABLE_DIFFUSION_15_INPAINTING = 1
comptime MODEL_TYPE_STABLE_DIFFUSION_20 = 2
comptime MODEL_TYPE_STABLE_DIFFUSION_20_BASE = 3
comptime MODEL_TYPE_STABLE_DIFFUSION_20_INPAINTING = 4
comptime MODEL_TYPE_STABLE_DIFFUSION_20_DEPTH = 5
comptime MODEL_TYPE_STABLE_DIFFUSION_21 = 6
comptime MODEL_TYPE_STABLE_DIFFUSION_21_BASE = 7
comptime MODEL_TYPE_STABLE_DIFFUSION_3 = 8
comptime MODEL_TYPE_STABLE_DIFFUSION_35 = 9
comptime MODEL_TYPE_STABLE_DIFFUSION_XL_10_BASE = 10
comptime MODEL_TYPE_STABLE_DIFFUSION_XL_10_BASE_INPAINTING = 11
comptime MODEL_TYPE_WUERSTCHEN_2 = 12
comptime MODEL_TYPE_STABLE_CASCADE_1 = 13
comptime MODEL_TYPE_PIXART_ALPHA = 14
comptime MODEL_TYPE_PIXART_SIGMA = 15
comptime MODEL_TYPE_FLUX_DEV_1 = 16
comptime MODEL_TYPE_FLUX_FILL_DEV_1 = 17
comptime MODEL_TYPE_FLUX_2 = 18
comptime MODEL_TYPE_SANA = 19
comptime MODEL_TYPE_HUNYUAN_VIDEO = 20
comptime MODEL_TYPE_HI_DREAM_FULL = 21
comptime MODEL_TYPE_CHROMA_1 = 22
comptime MODEL_TYPE_QWEN = 23
comptime MODEL_TYPE_ANIMA = 24
comptime MODEL_TYPE_Z_IMAGE = 25
comptime MODEL_TYPE_ERNIE = 26
comptime MODEL_TYPE_LENS = 27
comptime MODEL_TYPE_IDEOGRAM_4 = 28


# is_stable_diffusion  (ModelType.py:47-55)
def model_type_is_stable_diffusion(kind: Int) -> Bool:
    return (
        kind == MODEL_TYPE_STABLE_DIFFUSION_15
        or kind == MODEL_TYPE_STABLE_DIFFUSION_15_INPAINTING
        or kind == MODEL_TYPE_STABLE_DIFFUSION_20
        or kind == MODEL_TYPE_STABLE_DIFFUSION_20_BASE
        or kind == MODEL_TYPE_STABLE_DIFFUSION_20_INPAINTING
        or kind == MODEL_TYPE_STABLE_DIFFUSION_20_DEPTH
        or kind == MODEL_TYPE_STABLE_DIFFUSION_21
        or kind == MODEL_TYPE_STABLE_DIFFUSION_21_BASE
    )


# is_stable_diffusion_xl  (ModelType.py:57-59)
def model_type_is_stable_diffusion_xl(kind: Int) -> Bool:
    return (
        kind == MODEL_TYPE_STABLE_DIFFUSION_XL_10_BASE
        or kind == MODEL_TYPE_STABLE_DIFFUSION_XL_10_BASE_INPAINTING
    )


# is_stable_diffusion_3  (ModelType.py:61-63)
def model_type_is_stable_diffusion_3(kind: Int) -> Bool:
    return (
        kind == MODEL_TYPE_STABLE_DIFFUSION_3
        or kind == MODEL_TYPE_STABLE_DIFFUSION_35
    )


# is_stable_diffusion_3_5  (ModelType.py:65-66)
def model_type_is_stable_diffusion_3_5(kind: Int) -> Bool:
    return kind == MODEL_TYPE_STABLE_DIFFUSION_35


# is_wuerstchen  (ModelType.py:68-70)
def model_type_is_wuerstchen(kind: Int) -> Bool:
    return (
        kind == MODEL_TYPE_WUERSTCHEN_2
        or kind == MODEL_TYPE_STABLE_CASCADE_1
    )


# is_pixart  (ModelType.py:72-74)
def model_type_is_pixart(kind: Int) -> Bool:
    return (
        kind == MODEL_TYPE_PIXART_ALPHA
        or kind == MODEL_TYPE_PIXART_SIGMA
    )


# is_pixart_alpha  (ModelType.py:76-77)
def model_type_is_pixart_alpha(kind: Int) -> Bool:
    return kind == MODEL_TYPE_PIXART_ALPHA


# is_pixart_sigma  (ModelType.py:79-80)
def model_type_is_pixart_sigma(kind: Int) -> Bool:
    return kind == MODEL_TYPE_PIXART_SIGMA


# is_flux  (ModelType.py:82-85)
def model_type_is_flux(kind: Int) -> Bool:
    return (
        kind == MODEL_TYPE_FLUX_DEV_1
        or kind == MODEL_TYPE_FLUX_FILL_DEV_1
        or kind == MODEL_TYPE_FLUX_2
    )


# is_flux_1  (ModelType.py:87-89)
def model_type_is_flux_1(kind: Int) -> Bool:
    return (
        kind == MODEL_TYPE_FLUX_DEV_1
        or kind == MODEL_TYPE_FLUX_FILL_DEV_1
    )


# is_flux_2  (ModelType.py:91-92)
def model_type_is_flux_2(kind: Int) -> Bool:
    return kind == MODEL_TYPE_FLUX_2


# is_chroma  (ModelType.py:94-95)
def model_type_is_chroma(kind: Int) -> Bool:
    return kind == MODEL_TYPE_CHROMA_1


# is_qwen  (ModelType.py:97-98)
def model_type_is_qwen(kind: Int) -> Bool:
    return kind == MODEL_TYPE_QWEN


# is_anima  (Serenity-anima-ref ModelType.py:102-103)
def model_type_is_anima(kind: Int) -> Bool:
    return kind == MODEL_TYPE_ANIMA


# is_sana  (ModelType.py:100-101)
def model_type_is_sana(kind: Int) -> Bool:
    return kind == MODEL_TYPE_SANA


# is_hunyuan_video  (ModelType.py:103-104)
def model_type_is_hunyuan_video(kind: Int) -> Bool:
    return kind == MODEL_TYPE_HUNYUAN_VIDEO


# is_hi_dream  (ModelType.py:106-107)
def model_type_is_hi_dream(kind: Int) -> Bool:
    return kind == MODEL_TYPE_HI_DREAM_FULL


# is_z_image  (ModelType.py:109-110)
def model_type_is_z_image(kind: Int) -> Bool:
    return kind == MODEL_TYPE_Z_IMAGE


# is_ernie  (ModelType.py:112-113)
def model_type_is_ernie(kind: Int) -> Bool:
    return kind == MODEL_TYPE_ERNIE


# is_lens  (ModelType.py:119-120)
def model_type_is_lens(kind: Int) -> Bool:
    return kind == MODEL_TYPE_LENS


def model_type_is_ideogram_4(kind: Int) -> Bool:
    return kind == MODEL_TYPE_IDEOGRAM_4


# has_mask_input  (ModelType.py:115-119)
def model_type_has_mask_input(kind: Int) -> Bool:
    return (
        kind == MODEL_TYPE_STABLE_DIFFUSION_15_INPAINTING
        or kind == MODEL_TYPE_STABLE_DIFFUSION_20_INPAINTING
        or kind == MODEL_TYPE_STABLE_DIFFUSION_XL_10_BASE_INPAINTING
        or kind == MODEL_TYPE_FLUX_FILL_DEV_1
    )


# has_conditioning_image_input  (ModelType.py:121-125)
def model_type_has_conditioning_image_input(kind: Int) -> Bool:
    return (
        kind == MODEL_TYPE_STABLE_DIFFUSION_15_INPAINTING
        or kind == MODEL_TYPE_STABLE_DIFFUSION_20_INPAINTING
        or kind == MODEL_TYPE_STABLE_DIFFUSION_XL_10_BASE_INPAINTING
        or kind == MODEL_TYPE_FLUX_FILL_DEV_1
    )


# has_depth_input  (ModelType.py:127-128)
def model_type_has_depth_input(kind: Int) -> Bool:
    return kind == MODEL_TYPE_STABLE_DIFFUSION_20_DEPTH


# has_multiple_text_encoders  (ModelType.py:130-135)
def model_type_has_multiple_text_encoders(kind: Int) -> Bool:
    return (
        model_type_is_stable_diffusion_3(kind)
        or model_type_is_stable_diffusion_xl(kind)
        or model_type_is_flux_1(kind)
        or model_type_is_hunyuan_video(kind)
        or model_type_is_hi_dream(kind)
    )


# is_sd_v1  (ModelType.py:137-139)
def model_type_is_sd_v1(kind: Int) -> Bool:
    return (
        kind == MODEL_TYPE_STABLE_DIFFUSION_15
        or kind == MODEL_TYPE_STABLE_DIFFUSION_15_INPAINTING
    )


# is_sd_v2  (ModelType.py:141-147)
def model_type_is_sd_v2(kind: Int) -> Bool:
    return (
        kind == MODEL_TYPE_STABLE_DIFFUSION_20
        or kind == MODEL_TYPE_STABLE_DIFFUSION_20_BASE
        or kind == MODEL_TYPE_STABLE_DIFFUSION_20_INPAINTING
        or kind == MODEL_TYPE_STABLE_DIFFUSION_20_DEPTH
        or kind == MODEL_TYPE_STABLE_DIFFUSION_21
        or kind == MODEL_TYPE_STABLE_DIFFUSION_21_BASE
    )


# is_wuerstchen_v2  (ModelType.py:149-150)
def model_type_is_wuerstchen_v2(kind: Int) -> Bool:
    return kind == MODEL_TYPE_WUERSTCHEN_2


# is_stable_cascade  (ModelType.py:152-153)
def model_type_is_stable_cascade(kind: Int) -> Bool:
    return kind == MODEL_TYPE_STABLE_CASCADE_1


# is_flow_matching  (ModelType.py:162-172) — Serenity's is_flow_matching ends
# with `or self.is_lens()`; Lens uses FlowMatchEulerDiscreteScheduler and the
# flow-match predict path (BaseLensSetup.predict → _add_noise_discrete /
# _get_timestep_discrete), so LENS MUST be included here.
def model_type_is_flow_matching(kind: Int) -> Bool:
    return (
        model_type_is_stable_diffusion_3(kind)
        or model_type_is_flux(kind)
        or model_type_is_chroma(kind)
        or model_type_is_qwen(kind)
        or model_type_is_sana(kind)
        or model_type_is_hunyuan_video(kind)
        or model_type_is_hi_dream(kind)
        or model_type_is_z_image(kind)
        or model_type_is_ernie(kind)
        or model_type_is_lens(kind)
        or model_type_is_ideogram_4(kind)
    )


# is_video_model  (ModelType.py:166-167)
def model_type_is_video_model(kind: Int) -> Bool:
    return model_type_is_hunyuan_video(kind)


def model_type_str(kind: Int) -> String:
    if kind == MODEL_TYPE_STABLE_DIFFUSION_15:
        return "STABLE_DIFFUSION_15"
    elif kind == MODEL_TYPE_STABLE_DIFFUSION_15_INPAINTING:
        return "STABLE_DIFFUSION_15_INPAINTING"
    elif kind == MODEL_TYPE_STABLE_DIFFUSION_20:
        return "STABLE_DIFFUSION_20"
    elif kind == MODEL_TYPE_STABLE_DIFFUSION_20_BASE:
        return "STABLE_DIFFUSION_20_BASE"
    elif kind == MODEL_TYPE_STABLE_DIFFUSION_20_INPAINTING:
        return "STABLE_DIFFUSION_20_INPAINTING"
    elif kind == MODEL_TYPE_STABLE_DIFFUSION_20_DEPTH:
        return "STABLE_DIFFUSION_20_DEPTH"
    elif kind == MODEL_TYPE_STABLE_DIFFUSION_21:
        return "STABLE_DIFFUSION_21"
    elif kind == MODEL_TYPE_STABLE_DIFFUSION_21_BASE:
        return "STABLE_DIFFUSION_21_BASE"
    elif kind == MODEL_TYPE_STABLE_DIFFUSION_3:
        return "STABLE_DIFFUSION_3"
    elif kind == MODEL_TYPE_STABLE_DIFFUSION_35:
        return "STABLE_DIFFUSION_35"
    elif kind == MODEL_TYPE_STABLE_DIFFUSION_XL_10_BASE:
        return "STABLE_DIFFUSION_XL_10_BASE"
    elif kind == MODEL_TYPE_STABLE_DIFFUSION_XL_10_BASE_INPAINTING:
        return "STABLE_DIFFUSION_XL_10_BASE_INPAINTING"
    elif kind == MODEL_TYPE_WUERSTCHEN_2:
        return "WUERSTCHEN_2"
    elif kind == MODEL_TYPE_STABLE_CASCADE_1:
        return "STABLE_CASCADE_1"
    elif kind == MODEL_TYPE_PIXART_ALPHA:
        return "PIXART_ALPHA"
    elif kind == MODEL_TYPE_PIXART_SIGMA:
        return "PIXART_SIGMA"
    elif kind == MODEL_TYPE_FLUX_DEV_1:
        return "FLUX_DEV_1"
    elif kind == MODEL_TYPE_FLUX_FILL_DEV_1:
        return "FLUX_FILL_DEV_1"
    elif kind == MODEL_TYPE_FLUX_2:
        return "FLUX_2"
    elif kind == MODEL_TYPE_SANA:
        return "SANA"
    elif kind == MODEL_TYPE_HUNYUAN_VIDEO:
        return "HUNYUAN_VIDEO"
    elif kind == MODEL_TYPE_HI_DREAM_FULL:
        return "HI_DREAM_FULL"
    elif kind == MODEL_TYPE_CHROMA_1:
        return "CHROMA_1"
    elif kind == MODEL_TYPE_QWEN:
        return "QWEN"
    elif kind == MODEL_TYPE_ANIMA:
        return "ANIMA"
    elif kind == MODEL_TYPE_Z_IMAGE:
        return "Z_IMAGE"
    elif kind == MODEL_TYPE_ERNIE:
        return "ERNIE"
    elif kind == MODEL_TYPE_LENS:
        return "LENS"
    elif kind == MODEL_TYPE_IDEOGRAM_4:
        return "IDEOGRAM_4"
    else:
        return "UNKNOWN"


# PeftType  (ModelType.py:170-176)
comptime PEFT_TYPE_LORA = 0   # PeftType.LORA
comptime PEFT_TYPE_LOHA = 1   # PeftType.LOHA
comptime PEFT_TYPE_OFT_2 = 2  # PeftType.OFT_2


def peft_type_str(kind: Int) -> String:
    if kind == PEFT_TYPE_LORA:
        return "LORA"
    elif kind == PEFT_TYPE_LOHA:
        return "LOHA"
    else:
        return "OFT_2"
