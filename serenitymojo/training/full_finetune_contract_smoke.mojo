# training/full_finetune_contract_smoke.mojo
#
# No-CUDA contract smoke:
#   pixi run mojo run -I . serenitymojo/training/full_finetune_contract_smoke.mojo

from serenitymojo.training.full_finetune_contract import (
    FULL_FINETUNE_ADAM_M_PREFIX,
    FULL_FINETUNE_ADAM_V_PREFIX,
    FULL_FINETUNE_META_ACCUM_COUNT_INDEX,
    FULL_FINETUNE_META_FIELD_COUNT,
    FULL_FINETUNE_META_KEY,
    FULL_FINETUNE_META_T_STEP_INDEX,
    FULL_FINETUNE_OPTIMIZER_DTYPE,
    FULL_FINETUNE_PARAM_MASTER_PREFIX,
    FULL_FINETUNE_TENSOR_NAME_MANIFEST_KEY,
    OT_FULL_FT_LOOP_UNSUPPORTED,
    create_full_finetune_product_run_plan,
    default_full_finetune_resume_sidecar_spec,
    full_finetune_product_run_blocker,
    full_finetune_loop_status_name,
    full_finetune_model_type_has_onetrainer_registration,
    full_finetune_optimizer_adam_m_key,
    full_finetune_optimizer_adam_v_key,
    full_finetune_optimizer_param_key,
    full_finetune_target_for_key,
    full_finetune_target_for_model_type,
    full_finetune_target_for_product_model_type,
    full_finetune_targets,
    validate_full_finetune_contract,
    validate_full_finetune_resume_sidecar_spec,
    validate_full_finetune_product_run_plan,
)


def _check(cond: Bool, msg: String) raises:
    if not cond:
        raise Error(String("full_finetune_contract_smoke FAILED: ") + msg)


def _check_unsupported_target(key: String, expected_runner: String) raises:
    var target = full_finetune_target_for_key(key)
    _check(target.onetrainer_fine_tune_registered, key + String(" registered"))
    _check(
        target.real_train_loop_name == expected_runner,
        key + String(" runner mismatch"),
    )
    _check(
        target.product_loop_status == OT_FULL_FT_LOOP_UNSUPPORTED,
        key + String(" real full-finetune loop must remain unsupported"),
    )
    _check(
        target.shared_tensor_save_load_supported,
        key + String(" shared tensor save/load support missing"),
    )
    _check(
        target.shared_optimizer_resume_sidecar_supported,
        key + String(" shared optimizer sidecar support missing"),
    )
    _check(
        not target.can_run_full_finetune(),
        key + String(" must not claim runnable full-finetune"),
    )


def _gate_targets() raises:
    print("--- full-finetune target map ---")
    var targets = full_finetune_targets()
    _check(len(targets) == 9, "target family count")

    _check_unsupported_target(String("qwen"), String("train_qwenimage_real"))
    _check_unsupported_target(String("flux1"), String("train_flux_real"))
    _check_unsupported_target(String("flux2"), String("train_klein_real"))
    _check_unsupported_target(String("zimage"), String("train_zimage_real"))
    _check_unsupported_target(String("chroma"), String("train_chroma_real"))
    _check_unsupported_target(String("sdxl"), String("train_sdxl_real"))
    _check_unsupported_target(String("sd35"), String("train_sd35_real"))
    _check_unsupported_target(String("ernie"), String("train_ernie_real"))
    _check_unsupported_target(String("anima"), String("train_anima_real"))

    var anima = full_finetune_target_for_key(String("anima"))
    _check(
        anima.reference_root == String("/home/alex/OneTrainer-anima-ref"),
        "Anima must use OneTrainer-anima-ref",
    )
    var qwen = full_finetune_target_for_model_type(String("QWEN"))
    _check(qwen.key == String("qwen"), "QWEN model type lookup")
    var qwen_product = full_finetune_target_for_product_model_type(String("qwenimage"))
    _check(qwen_product.key == String("qwen"), "qwenimage product model type lookup")
    var qwen_product_plan = create_full_finetune_product_run_plan(String("qwenimage"), 2)
    validate_full_finetune_product_run_plan(qwen_product_plan)
    _check(
        qwen_product_plan.runner_name == String("train_qwenimage_real"),
        "qwenimage product full-finetune runner reference",
    )
    _check(
        qwen_product_plan.requires_model_save_hook,
        "qwenimage product full-finetune save requirement",
    )
    _check(
        qwen_product_plan.requires_model_load_rebind,
        "qwenimage product full-finetune load/rebind requirement",
    )
    _check(
        qwen_product_plan.requires_tensor_name_manifest,
        "qwenimage product full-finetune manifest requirement",
    )
    _check(
        not qwen_product_plan.can_run_full_finetune(),
        "qwenimage product full-finetune must remain unsupported",
    )
    _check(
        full_finetune_product_run_blocker(qwen_product_plan).byte_length() > 0,
        "qwenimage product full-finetune blocker text",
    )
    var sdxl = full_finetune_target_for_model_type(
        String("STABLE_DIFFUSION_XL_10_BASE_INPAINTING")
    )
    _check(sdxl.key == String("sdxl"), "SDXL inpainting model type lookup")
    var sd35 = full_finetune_target_for_model_type(String("STABLE_DIFFUSION_35"))
    _check(sd35.key == String("sd35"), "SD35 model type lookup")
    _check(
        not full_finetune_model_type_has_onetrainer_registration(String("STABLE_DIFFUSION_3")),
        "plain SD3 is outside this full-finetune target contract",
    )
    var anima_by_model = full_finetune_target_for_model_type(String("ANIMA"))
    _check(anima_by_model.key == String("anima"), "ANIMA model type lookup")

    _check(
        full_finetune_model_type_has_onetrainer_registration(String("FLUX_FILL_DEV_1")),
        "FLUX_FILL_DEV_1 registration",
    )
    _check(
        not full_finetune_model_type_has_onetrainer_registration(String("LTX_2")),
        "LTX_2 is outside this full-finetune target contract",
    )
    _check(
        full_finetune_loop_status_name(OT_FULL_FT_LOOP_UNSUPPORTED)
        == String("unsupported"),
        "status name",
    )
    print("  target map PASS")


def _gate_sidecar() raises:
    print("--- full-finetune sidecar schema ---")
    _check(
        String(FULL_FINETUNE_TENSOR_NAME_MANIFEST_KEY)
        == String("__full_finetune_tensor_names_utf8__"),
        "tensor-name manifest key",
    )
    _check(String(FULL_FINETUNE_PARAM_MASTER_PREFIX) == String("param."), "param prefix")
    _check(String(FULL_FINETUNE_ADAM_M_PREFIX) == String("adam_m."), "adam_m prefix")
    _check(String(FULL_FINETUNE_ADAM_V_PREFIX) == String("adam_v."), "adam_v prefix")
    _check(String(FULL_FINETUNE_META_KEY) == String("__meta__"), "meta key")
    _check(FULL_FINETUNE_META_T_STEP_INDEX == 0, "meta t index")
    _check(FULL_FINETUNE_META_ACCUM_COUNT_INDEX == 1, "meta accum_count index")
    _check(FULL_FINETUNE_META_FIELD_COUNT == 2, "meta field count")
    _check(String(FULL_FINETUNE_OPTIMIZER_DTYPE) == String("F32"), "optimizer dtype")

    _check(full_finetune_optimizer_param_key(7) == String("param.7"), "param key")
    _check(full_finetune_optimizer_adam_m_key(7) == String("adam_m.7"), "adam_m key")
    _check(full_finetune_optimizer_adam_v_key(7) == String("adam_v.7"), "adam_v key")

    var spec = default_full_finetune_resume_sidecar_spec()
    validate_full_finetune_resume_sidecar_spec(spec)

    var missing_manifest = spec.copy()
    missing_manifest.has_tensor_name_manifest = False
    var missing_manifest_raised = False
    try:
        validate_full_finetune_resume_sidecar_spec(missing_manifest)
    except e:
        missing_manifest_raised = True
        print("  raised as expected [missing manifest]:", String(e))
    _check(missing_manifest_raised, "missing manifest must fail")

    var wrong_meta = spec.copy()
    wrong_meta.meta_field_count = 1
    var wrong_meta_raised = False
    try:
        validate_full_finetune_resume_sidecar_spec(wrong_meta)
    except e:
        wrong_meta_raised = True
        print("  raised as expected [wrong meta field count]:", String(e))
    _check(wrong_meta_raised, "wrong meta field count must fail")

    var wrong_order = spec.copy()
    wrong_order.param_order_bound_to_manifest = False
    var wrong_order_raised = False
    try:
        validate_full_finetune_resume_sidecar_spec(wrong_order)
    except e:
        wrong_order_raised = True
        print("  raised as expected [unbound param order]:", String(e))
    _check(wrong_order_raised, "unbound param order must fail")

    var wrong_dtype = spec.copy()
    wrong_dtype.optimizer_master_dtype = String("BF16")
    var wrong_dtype_raised = False
    try:
        validate_full_finetune_resume_sidecar_spec(wrong_dtype)
    except e:
        wrong_dtype_raised = True
        print("  raised as expected [wrong optimizer master dtype]:", String(e))
    _check(wrong_dtype_raised, "wrong optimizer dtype must fail")
    print("  sidecar schema PASS")


def main() raises:
    print("==== full_finetune_contract no-CUDA smoke ====")
    validate_full_finetune_contract()
    _gate_targets()
    _gate_sidecar()
    print("full_finetune_contract_smoke PASS")
