# SDXL base setup contract gate.
#
# Source of truth:
#   /home/alex/Serenity/modules/modelSetup/BaseStableDiffusionXLSetup.py

from serenity_trainer.modelSetup.BaseStableDiffusionXLSetup import (
    BaseStableDiffusionXLSetup,
    SDXL_INPAINT_MODEL_TYPE_NAME,
    SDXL_MODEL_TYPE_NAME,
    sdxl_model_t_from_timestep,
    sdxl_noise_sigma_from_betas,
    sdxl_scheduler_sigma_from_betas,
    sdxl_snr_from_betas,
)
from serenity_trainer.util.enum.TrainingMethod import TM_FINE_TUNE, TM_LORA


def _abs(x: Float32) -> Float32:
    if x < Float32(0.0):
        return -x
    return x


def _expect_int(name: String, got: Int, expected: Int) raises:
    if got != expected:
        raise Error(name + String(": got ") + String(got) + String(", expected ") + String(expected))


def _expect_bool(name: String, got: Bool, expected: Bool) raises:
    if got != expected:
        raise Error(name + String(": got ") + String(got) + String(", expected ") + String(expected))


def _expect_string(name: String, got: String, expected: String) raises:
    if got != expected:
        raise Error(name + String(": got ") + got + String(", expected ") + expected)


def _expect_close(name: String, got: Float32, expected: Float32, tol: Float32) raises:
    var diff = _abs(got - expected)
    if diff > tol:
        raise Error(name + String(": got ") + String(got) + String(", expected ") + String(expected) + String(", |d| ") + String(diff))


def main() raises:
    var base = BaseStableDiffusionXLSetup()
    var predict = base.predict_contract()
    var opt_lora = base.optimization_contract(TM_LORA, True, True)
    var opt_ft = base.optimization_contract(TM_FINE_TUNE, True, False)
    var device_plan = base.train_device_plan(True, True, False, True, False, True, False)
    var text_cache = base.prepare_text_caching_plan(True, False)

    _expect_string("sdxl model type", String(SDXL_MODEL_TYPE_NAME), String("STABLE_DIFFUSION_XL_10_BASE"))
    _expect_string("sdxl inpaint model type", String(SDXL_INPAINT_MODEL_TYPE_NAME), String("STABLE_DIFFUSION_XL_10_BASE_INPAINTING"))
    _expect_int("base model type count", len(base.model_types), 2)

    _expect_int("predict required fields", len(predict.required_batch_fields), 6)
    _expect_string("predict latent field", predict.required_batch_fields[0], String("latent_image"))
    _expect_string("predict tokens 1", predict.required_batch_fields[1], String("tokens_1"))
    _expect_string("predict tokens 2", predict.required_batch_fields[2], String("tokens_2"))
    _expect_string("predict crop resolution", predict.required_batch_fields[5], String("crop_resolution"))
    _expect_int("predict cached text fields", len(predict.cached_text_batch_fields), 3)
    _expect_string("predict cached te2 pooled", predict.cached_text_batch_fields[2], String("text_encoder_2_pooled_state"))
    _expect_int("predict token masks not consumed", len(predict.token_mask_fields_not_consumed), 2)
    _expect_string("predict latent mask", predict.conditional_latent_fields[0], String("latent_mask"))
    _expect_string("predict latent conditioning", predict.conditional_latent_fields[1], String("latent_conditioning_image"))
    _expect_int("predict output fields", len(predict.output_fields), 5)
    _expect_string("predict loss type", predict.loss_type, String("target"))
    _expect_string("predict output predicted", predict.output_fields[2], String("predicted"))
    _expect_string("predict output prediction type", predict.output_fields[4], String("prediction_type"))
    _expect_int("predict prediction type count", len(predict.prediction_types), 2)
    _expect_string("predict epsilon", predict.prediction_types[0], String("epsilon"))
    _expect_string("predict v prediction", predict.prediction_types[1], String("v_prediction"))
    _expect_string("predict latent scale expression", predict.scale_latents_expression, String("latent_image * vae.config['scaling_factor']"))
    _expect_string("predict add time ids expression", predict.add_time_ids_expression, String("stack([original_height, original_width, crop_top, crop_left, target_height, target_width], dim=1).to(dtype=scaled_noisy_latent_image.dtype)"))
    _expect_string("predict UNet sample expression", predict.unet_sample_expression, String("latent_input.to(dtype=model.train_dtype.torch_dtype())"))
    _expect_string("predict epsilon target", predict.epsilon_target_expression, String("latent_noise"))

    _expect_int("lora opt checkpoint parts", len(opt_lora.checkpoint_parts), 3)
    _expect_string("lora checkpoint unet", opt_lora.checkpoint_parts[0], String("unet"))
    _expect_string("lora checkpoint te2", opt_lora.checkpoint_parts[2], String("text_encoder_2"))
    _expect_int("lora checkpoint helpers", len(opt_lora.checkpoint_helpers), 4)
    _expect_int("lora circular padding parts", len(opt_lora.force_circular_padding_parts), 3)
    _expect_string("lora circular unet lora", opt_lora.force_circular_padding_parts[2], String("unet_lora"))
    _expect_int("lora quantized parts", len(opt_lora.quantized_parts), 4)
    _expect_int("lora autocast parts", len(opt_lora.autocast_weight_dtype_parts), 6)
    _expect_string("lora autocast embedding", opt_lora.autocast_weight_dtype_parts[5], String("embedding"))
    _expect_bool("lora disables fp16 vae autocast", opt_lora.disables_fp16_vae_autocast, True)
    _expect_int("lora dtype caveats", len(opt_lora.dtype_boundary_caveats), 7)
    _expect_int("ft opt autocast parts", len(opt_ft.autocast_weight_dtype_parts), 5)

    _expect_bool("device te1 train", device_plan.text_encoder_1_on_train_device, True)
    _expect_bool("device te2 temp", device_plan.text_encoder_2_on_train_device, False)
    _expect_bool("device vae temp", device_plan.vae_on_train_device, False)
    _expect_bool("device unet train", device_plan.unet_on_train_device, True)
    _expect_bool("device te1 train mode", device_plan.text_encoder_1_train_mode, True)
    _expect_bool("device te2 train mode", device_plan.text_encoder_2_train_mode, False)
    _expect_bool("device vae train mode", device_plan.vae_train_mode, False)
    _expect_bool("device unet train mode", device_plan.unet_train_mode, True)
    _expect_bool("text cache model temp", text_cache.move_model_to_temp_device, True)
    _expect_bool("text cache te1 stays", text_cache.move_text_encoder_1_to_train_device, False)
    _expect_bool("text cache te2 moves", text_cache.move_text_encoder_2_to_train_device, True)

    var betas = List[Float32]()
    betas.append(Float32(0.0001))
    betas.append(Float32(0.0002))
    betas.append(Float32(0.0003))
    _expect_close("model_t t=250", sdxl_model_t_from_timestep(250), Float32(250.0), Float32(1e-6))
    _expect_close("noise sigma t=2", sdxl_noise_sigma_from_betas(betas, 2), Float32(0.02449265), Float32(1e-4))
    _expect_close("scheduler sigma t=2", sdxl_scheduler_sigma_from_betas(betas, 2), Float32(0.0245), Float32(1e-4))
    _expect_close("snr t=2", sdxl_snr_from_betas(betas, 2), Float32(1665.9723), Float32(1.0))

    print("SDXL SETUP BASE CONTRACT OK")
