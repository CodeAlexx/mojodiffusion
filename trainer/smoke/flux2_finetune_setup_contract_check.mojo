# Flux2/Klein fine-tune setup contract smoke.
#
# This is not a train parity gate and does not run Serenity. It verifies the
# build-only Flux2FineTuneSetup surface against Serenity-visible setup
# behavior: transformer-only full finetune, frozen text/VAE, train-device
# placement, ModuleFilter use, text-caching plan, and dtype caveats.

from serenity_trainer.modelSetup.Flux2FineTuneSetup import Flux2FineTuneSetup
from serenity_trainer.util.enum.ModelType import MODEL_TYPE_FLUX_2
from serenity_trainer.util.enum.TrainingMethod import TM_FINE_TUNE


def _expect(name: String, got: Int, expected: Int) raises:
    if got != expected:
        raise Error(
            name + String(" got=") + String(got)
            + String(" expected=") + String(expected)
        )


def _expect_bool(name: String, got: Bool, expected: Bool) raises:
    if got != expected:
        raise Error(
            name + String(" got=") + String(got)
            + String(" expected=") + String(expected)
        )


def _expect_string(name: String, got: String, expected: String) raises:
    if got != expected:
        raise Error(name + String(" got=") + got + String(" expected=") + expected)


def main() raises:
    var setup = Flux2FineTuneSetup()
    _expect("model type", setup.registration.model_type, MODEL_TYPE_FLUX_2)
    _expect("training method", setup.registration.training_method, TM_FINE_TUNE)
    _expect_bool("trains text encoder", setup.registration.trains_text_encoder, False)
    _expect_bool("trains transformer", setup.registration.trains_transformer, True)
    _expect_bool("trains vae", setup.registration.trains_vae, False)
    _expect_bool("embedding training", setup.registration.supports_embedding_training, False)
    _expect_bool("output embedding training", setup.registration.supports_output_embedding_training, False)

    var params = setup.create_parameters()
    _expect("param groups", len(params), 1)
    _expect_string("param group 0", params[0], String("transformer"))

    var setup_plan = setup.setup_model_plan()
    _expect_bool("init params", setup_plan.initializes_model_parameters, True)
    _expect_bool("module filter", setup_plan.uses_module_filter_for_transformer, True)
    _expect_bool("debug filter", setup_plan.uses_debug_flag_for_transformer_filter, True)
    _expect_bool("setup embeddings", setup_plan.setups_embeddings, False)

    var req = setup.requires_grad_plan()
    _expect_bool("requires transformer config", req.applies_transformer_config, True)
    _expect_bool("freezes text encoder", req.freezes_text_encoder, True)
    _expect_bool("freezes vae", req.freezes_vae, True)
    _expect_bool("requires module filter", req.transformer_uses_module_filter, True)

    var cached_device = setup.train_device_plan(True, False)
    _expect_bool("cached text device", cached_device.text_encoder_on_train_device, False)
    _expect_bool("cached vae device", cached_device.vae_on_train_device, False)
    _expect_bool("cached transformer device", cached_device.transformer_on_train_device, True)
    _expect_bool("cached text eval", cached_device.text_encoder_train_mode, False)
    _expect_bool("cached vae eval", cached_device.vae_train_mode, False)
    _expect_bool("cached transformer eval", cached_device.transformer_train_mode, False)

    var uncached_device = setup.train_device_plan(False, True)
    _expect_bool("uncached text device", uncached_device.text_encoder_on_train_device, True)
    _expect_bool("uncached vae device", uncached_device.vae_on_train_device, True)
    _expect_bool("uncached transformer train", uncached_device.transformer_train_mode, True)

    var text_cache = setup.prepare_text_caching_plan()
    _expect_bool("text cache model temp", text_cache.move_model_to_temp_device, True)
    _expect_bool("text cache text encoder train", text_cache.move_text_encoder_to_train_device, True)
    _expect_bool("text cache eval", text_cache.set_eval_mode, True)
    _expect_bool("text cache gc", text_cache.run_torch_gc, True)

    var trainable = setup.trainable_model_part_names()
    var frozen = setup.frozen_model_part_names()
    _expect("trainable parts", len(trainable), 1)
    _expect("frozen parts", len(frozen), 2)
    _expect_string("trainable transformer", trainable[0], String("transformer"))
    _expect_string("frozen text", frozen[0], String("text_encoder"))
    _expect_string("frozen vae", frozen[1], String("vae"))
    _expect_bool("method freezes text", setup.freezes_text_encoder(), True)
    _expect_bool("method freezes vae", setup.freezes_vae(), True)
    _expect_bool("after step reset", setup.after_optimizer_step_reapplies_requires_grad(), True)

    var caveats = setup.dtype_caveats()
    _expect("dtype caveats", len(caveats), 4)

    print("flux2 ft model type =", MODEL_TYPE_FLUX_2)
    print("param groups =", len(params))
    print("frozen parts =", len(frozen))
    print("cached text on train =", cached_device.text_encoder_on_train_device)
    print("uncached text on train =", uncached_device.text_encoder_on_train_device)
    print("dtype caveats =", len(caveats))
    print("FLUX2 FINE-TUNE SETUP CONTRACT OK")
