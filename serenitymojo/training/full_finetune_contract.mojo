# training/full_finetune_contract.mojo -- OneTrainer full-finetune readiness map.
#
# This is a no-CUDA metadata contract. It does not import DeviceContext, does not
# load model weights, and does not claim a real full-weight training loop exists.
#
# Reference sources:
#   /home/alex/OneTrainer/modules/modelSetup/*FineTuneSetup.py
#   /home/alex/OneTrainer-anima-ref/modules/modelSetup/AnimaFineTuneSetup.py
#
# Scope:
#   * maps target OneTrainer model families that register TrainingMethod.FINE_TUNE
#   * separates shared full-weight save/load scaffolding from unsupported real loops
#   * pins the TrainState optimizer/master resume sidecar keys from loop.mojo
#
# Dtype contract:
#   Model tensor payloads stay in checkpoint/model storage dtype. Optimizer
#   masters and Adam moments are intentionally F32 sidecar tensors.

from std.collections import List


comptime OT_FULL_FT_LOOP_UNSUPPORTED = 0
comptime OT_FULL_FT_LOOP_FAIL_LOUD_ONLY = 1
comptime OT_FULL_FT_LOOP_WIRED = 2

comptime FULL_FINETUNE_TENSOR_NAME_MANIFEST_KEY = "__full_finetune_tensor_names_utf8__"
comptime FULL_FINETUNE_PARAM_MASTER_PREFIX = "param."
comptime FULL_FINETUNE_ADAM_M_PREFIX = "adam_m."
comptime FULL_FINETUNE_ADAM_V_PREFIX = "adam_v."
comptime FULL_FINETUNE_META_KEY = "__meta__"
comptime FULL_FINETUNE_META_T_STEP_INDEX = 0
comptime FULL_FINETUNE_META_ACCUM_COUNT_INDEX = 1
comptime FULL_FINETUNE_META_FIELD_COUNT = 2
comptime FULL_FINETUNE_OPTIMIZER_DTYPE = "F32"


@fieldwise_init
struct OTFullFinetuneTarget(Copyable, Movable):
    var key: String
    var label: String
    var reference_root: String
    var setup_file: String
    var model_types_csv: String
    var onetrainer_fine_tune_registered: Bool
    var real_train_loop_name: String
    var product_loop_status: Int
    var shared_tensor_save_load_supported: Bool
    var shared_optimizer_resume_sidecar_supported: Bool
    var note: String

    def has_shared_resume_scaffolding(self) -> Bool:
        return (
            self.shared_tensor_save_load_supported
            and self.shared_optimizer_resume_sidecar_supported
        )

    def has_product_full_finetune_loop(self) -> Bool:
        return self.product_loop_status == OT_FULL_FT_LOOP_WIRED

    def can_run_full_finetune(self) -> Bool:
        return (
            self.onetrainer_fine_tune_registered
            and self.has_shared_resume_scaffolding()
            and self.has_product_full_finetune_loop()
        )


@fieldwise_init
struct OTFullFinetuneResumeSidecarSpec(Copyable, Movable):
    var has_model_tensor_payload: Bool
    var has_tensor_name_manifest: Bool
    var has_param_master_prefix: Bool
    var has_adam_m_prefix: Bool
    var has_adam_v_prefix: Bool
    var has_meta_tensor: Bool
    var meta_field_count: Int
    var param_order_bound_to_manifest: Bool
    var optimizer_master_dtype: String
    var optimizer_moment_dtype: String


@fieldwise_init
struct OTFullFinetuneProductRunPlan(Copyable, Movable):
    var requested_model_type: String
    var optimizer: Int
    var target: OTFullFinetuneTarget
    var runner_name: String
    var product_loop_status: Int
    var requires_model_tensor_inventory: Bool
    var requires_model_save_hook: Bool
    var requires_model_load_rebind: Bool
    var requires_tensor_name_manifest: Bool
    var requires_optimizer_sidecar_binding: Bool
    var requires_resume_manifest_mapping: Bool
    var requires_positive_product_smoke: Bool

    def can_run_full_finetune(self) -> Bool:
        return self.target.can_run_full_finetune()


def full_finetune_loop_status_name(status: Int) -> String:
    if status == OT_FULL_FT_LOOP_UNSUPPORTED:
        return String("unsupported")
    if status == OT_FULL_FT_LOOP_FAIL_LOUD_ONLY:
        return String("fail_loud_only")
    if status == OT_FULL_FT_LOOP_WIRED:
        return String("wired")
    return String("unknown")


def _target(
    key: String,
    label: String,
    reference_root: String,
    setup_file: String,
    model_types_csv: String,
    runner: String,
    note: String,
) -> OTFullFinetuneTarget:
    return OTFullFinetuneTarget(
        key,
        label,
        reference_root,
        setup_file,
        model_types_csv,
        True,
        runner,
        OT_FULL_FT_LOOP_UNSUPPORTED,
        True,
        True,
        note,
    )


def full_finetune_targets() -> List[OTFullFinetuneTarget]:
    var out = List[OTFullFinetuneTarget]()
    var root = String("/home/alex/OneTrainer")
    var anima_root = String("/home/alex/OneTrainer-anima-ref")
    var note = String("OneTrainer FINE_TUNE is registered; Mojo product loop is still LoRA-only/full-weight unsupported")

    out.append(_target(
        String("qwen"),
        String("Qwen"),
        root,
        String("QwenFineTuneSetup.py"),
        String("QWEN"),
        String("train_qwenimage_real"),
        note,
    ))
    out.append(_target(
        String("flux1"),
        String("Flux.1 dev/fill"),
        root,
        String("FluxFineTuneSetup.py"),
        String("FLUX_DEV_1,FLUX_FILL_DEV_1"),
        String("train_flux_real"),
        note,
    ))
    out.append(_target(
        String("flux2"),
        String("Flux2/Klein"),
        root,
        String("Flux2FineTuneSetup.py"),
        String("FLUX_2"),
        String("train_klein_real"),
        note,
    ))
    out.append(_target(
        String("zimage"),
        String("Z-Image"),
        root,
        String("ZImageFineTuneSetup.py"),
        String("Z_IMAGE"),
        String("train_zimage_real"),
        note,
    ))
    out.append(_target(
        String("chroma"),
        String("Chroma"),
        root,
        String("ChromaFineTuneSetup.py"),
        String("CHROMA_1"),
        String("train_chroma_real"),
        note,
    ))
    out.append(_target(
        String("sdxl"),
        String("SDXL"),
        root,
        String("StableDiffusionXLFineTuneSetup.py"),
        String("STABLE_DIFFUSION_XL_10_BASE,STABLE_DIFFUSION_XL_10_BASE_INPAINTING"),
        String("train_sdxl_real"),
        note,
    ))
    out.append(_target(
        String("sd35"),
        String("SD3.5"),
        root,
        String("StableDiffusion3FineTuneSetup.py"),
        String("STABLE_DIFFUSION_35"),
        String("train_sd35_real"),
        note,
    ))
    out.append(_target(
        String("ernie"),
        String("Ernie"),
        root,
        String("ErnieFineTuneSetup.py"),
        String("ERNIE"),
        String("train_ernie_real"),
        note,
    ))
    out.append(_target(
        String("anima"),
        String("Anima"),
        anima_root,
        String("AnimaFineTuneSetup.py"),
        String("ANIMA"),
        String("train_anima_real"),
        String("Anima FINE_TUNE reference is local OneTrainer-anima-ref; Mojo product loop is still LoRA-only/full-weight unsupported"),
    ))
    return out^


def full_finetune_target_for_key(key: String) raises -> OTFullFinetuneTarget:
    var targets = full_finetune_targets()
    for i in range(len(targets)):
        if targets[i].key == key:
            return targets[i].copy()
    raise Error(String("full-finetune contract: unknown target key ") + key)


def full_finetune_target_for_model_type(model_type: String) raises -> OTFullFinetuneTarget:
    if model_type == String("QWEN"):
        return full_finetune_target_for_key(String("qwen"))
    if model_type == String("FLUX_DEV_1") or model_type == String("FLUX_FILL_DEV_1"):
        return full_finetune_target_for_key(String("flux1"))
    if model_type == String("FLUX_2"):
        return full_finetune_target_for_key(String("flux2"))
    if model_type == String("Z_IMAGE"):
        return full_finetune_target_for_key(String("zimage"))
    if model_type == String("CHROMA_1"):
        return full_finetune_target_for_key(String("chroma"))
    if (
        model_type == String("STABLE_DIFFUSION_XL_10_BASE")
        or model_type == String("STABLE_DIFFUSION_XL_10_BASE_INPAINTING")
    ):
        return full_finetune_target_for_key(String("sdxl"))
    if model_type == String("STABLE_DIFFUSION_35"):
        return full_finetune_target_for_key(String("sd35"))
    if model_type == String("ERNIE"):
        return full_finetune_target_for_key(String("ernie"))
    if model_type == String("ANIMA"):
        return full_finetune_target_for_key(String("anima"))
    raise Error(String("full-finetune contract: unsupported ModelType ") + model_type)


def full_finetune_target_for_product_model_type(model_type: String) raises -> OTFullFinetuneTarget:
    if (
        model_type == String("qwenimage")
        or model_type == String("qwen")
        or model_type == String("QWEN")
    ):
        return full_finetune_target_for_key(String("qwen"))
    if (
        model_type == String("flux")
        or model_type == String("flux1")
        or model_type == String("FLUX_DEV_1")
        or model_type == String("FLUX_FILL_DEV_1")
    ):
        return full_finetune_target_for_key(String("flux1"))
    if (
        model_type == String("klein")
        or model_type == String("klein4b")
        or model_type == String("klein9b")
        or model_type == String("flux2")
        or model_type == String("FLUX_2")
    ):
        return full_finetune_target_for_key(String("flux2"))
    if model_type == String("zimage") or model_type == String("Z_IMAGE"):
        return full_finetune_target_for_key(String("zimage"))
    if model_type == String("chroma") or model_type == String("CHROMA_1"):
        return full_finetune_target_for_key(String("chroma"))
    if (
        model_type == String("sdxl")
        or model_type == String("STABLE_DIFFUSION_XL_10_BASE")
        or model_type == String("STABLE_DIFFUSION_XL_10_BASE_INPAINTING")
    ):
        return full_finetune_target_for_key(String("sdxl"))
    if (
        model_type == String("sd35")
        or model_type == String("sd3.5")
        or model_type == String("sd3-5")
        or model_type == String("STABLE_DIFFUSION_35")
    ):
        return full_finetune_target_for_key(String("sd35"))
    if (
        model_type == String("ernie_image")
        or model_type == String("ernie")
        or model_type == String("ERNIE")
    ):
        return full_finetune_target_for_key(String("ernie"))
    if model_type == String("anima") or model_type == String("ANIMA"):
        return full_finetune_target_for_key(String("anima"))
    raise Error(String("full-finetune contract: unsupported product model_type ") + model_type)


def full_finetune_model_type_has_onetrainer_registration(model_type: String) -> Bool:
    try:
        var target = full_finetune_target_for_model_type(model_type)
        return target.onetrainer_fine_tune_registered
    except e:
        return False


def full_finetune_optimizer_param_key(i: Int) -> String:
    return String(FULL_FINETUNE_PARAM_MASTER_PREFIX) + String(i)


def full_finetune_optimizer_adam_m_key(i: Int) -> String:
    return String(FULL_FINETUNE_ADAM_M_PREFIX) + String(i)


def full_finetune_optimizer_adam_v_key(i: Int) -> String:
    return String(FULL_FINETUNE_ADAM_V_PREFIX) + String(i)


def default_full_finetune_resume_sidecar_spec() -> OTFullFinetuneResumeSidecarSpec:
    return OTFullFinetuneResumeSidecarSpec(
        True,
        True,
        True,
        True,
        True,
        True,
        FULL_FINETUNE_META_FIELD_COUNT,
        True,
        String(FULL_FINETUNE_OPTIMIZER_DTYPE),
        String(FULL_FINETUNE_OPTIMIZER_DTYPE),
    )


def create_full_finetune_product_run_plan(
    requested_model_type: String, optimizer: Int
) raises -> OTFullFinetuneProductRunPlan:
    var target = full_finetune_target_for_product_model_type(requested_model_type)
    return OTFullFinetuneProductRunPlan(
        requested_model_type.copy(),
        optimizer,
        target.copy(),
        target.real_train_loop_name.copy(),
        target.product_loop_status,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
    )


def validate_full_finetune_resume_sidecar_spec(
    spec: OTFullFinetuneResumeSidecarSpec,
) raises:
    if not spec.has_model_tensor_payload:
        raise Error("full-finetune resume requires the full model tensor payload")
    if not spec.has_tensor_name_manifest:
        raise Error(
            String("full-finetune resume requires tensor-name manifest ")
            + String(FULL_FINETUNE_TENSOR_NAME_MANIFEST_KEY)
        )
    if not spec.has_param_master_prefix:
        raise Error(
            String("full-finetune resume sidecar requires ")
            + String(FULL_FINETUNE_PARAM_MASTER_PREFIX)
            + String("N F32 master tensors")
        )
    if not spec.has_adam_m_prefix:
        raise Error(
            String("full-finetune resume sidecar requires ")
            + String(FULL_FINETUNE_ADAM_M_PREFIX)
            + String("N F32 Adam first-moment tensors")
        )
    if not spec.has_adam_v_prefix:
        raise Error(
            String("full-finetune resume sidecar requires ")
            + String(FULL_FINETUNE_ADAM_V_PREFIX)
            + String("N F32 Adam second-moment tensors")
        )
    if not spec.has_meta_tensor:
        raise Error(
            String("full-finetune resume sidecar requires ")
            + String(FULL_FINETUNE_META_KEY)
            + String(" metadata tensor")
        )
    if spec.meta_field_count != FULL_FINETUNE_META_FIELD_COUNT:
        raise Error("full-finetune resume metadata must be F32 [t_step, accum_count]")
    if not spec.param_order_bound_to_manifest:
        raise Error("full-finetune resume param.N order must match the tensor-name manifest")
    if spec.optimizer_master_dtype != String(FULL_FINETUNE_OPTIMIZER_DTYPE):
        raise Error("full-finetune optimizer masters must be F32 sidecar tensors")
    if spec.optimizer_moment_dtype != String(FULL_FINETUNE_OPTIMIZER_DTYPE):
        raise Error("full-finetune Adam moments must be F32 sidecar tensors")


def validate_full_finetune_product_run_plan(
    plan: OTFullFinetuneProductRunPlan,
) raises:
    validate_full_finetune_target_contract(plan.target)
    if plan.requested_model_type == String(""):
        raise Error("full-finetune product run plan: requested model_type is empty")
    if plan.runner_name != plan.target.real_train_loop_name:
        raise Error("full-finetune product run plan: runner name drift")
    if plan.product_loop_status != plan.target.product_loop_status:
        raise Error("full-finetune product run plan: loop status drift")
    if plan.product_loop_status != OT_FULL_FT_LOOP_UNSUPPORTED:
        raise Error("full-finetune product run plan: unsupported preflight must not claim support")
    if not plan.requires_model_tensor_inventory:
        raise Error("full-finetune product run plan: missing model tensor inventory requirement")
    if not plan.requires_model_save_hook:
        raise Error("full-finetune product run plan: missing model save hook requirement")
    if not plan.requires_model_load_rebind:
        raise Error("full-finetune product run plan: missing model load/rebind requirement")
    if not plan.requires_tensor_name_manifest:
        raise Error("full-finetune product run plan: missing tensor-name manifest requirement")
    if not plan.requires_optimizer_sidecar_binding:
        raise Error("full-finetune product run plan: missing optimizer sidecar binding requirement")
    if not plan.requires_resume_manifest_mapping:
        raise Error("full-finetune product run plan: missing resume manifest mapping requirement")
    if not plan.requires_positive_product_smoke:
        raise Error("full-finetune product run plan: missing positive product smoke requirement")
    validate_full_finetune_resume_sidecar_spec(
        default_full_finetune_resume_sidecar_spec()
    )


def full_finetune_product_run_blocker(plan: OTFullFinetuneProductRunPlan) -> String:
    return (
        String("model_type=")
        + plan.requested_model_type
        + String(" target=")
        + plan.target.key
        + String(" setup=")
        + plan.target.reference_root
        + String("/modules/modelSetup/")
        + plan.target.setup_file
        + String(" runner=")
        + plan.runner_name
        + String(" optimizer_tag=")
        + String(plan.optimizer)
        + String(" product_loop_status=")
        + full_finetune_loop_status_name(plan.product_loop_status)
        + String("; required before support: exact model tensor inventory, model-specific save_full_finetune_model_tensors hook, load_full_finetune_model_tensors rebind path, tensor-name manifest ")
        + String(FULL_FINETUNE_TENSOR_NAME_MANIFEST_KEY)
        + String(", TrainState sidecar binding for ")
        + String(FULL_FINETUNE_PARAM_MASTER_PREFIX)
        + String("N/")
        + String(FULL_FINETUNE_ADAM_M_PREFIX)
        + String("N/")
        + String(FULL_FINETUNE_ADAM_V_PREFIX)
        + String("N plus ")
        + String(FULL_FINETUNE_META_KEY)
        + String(", resume manifest/load mapping, and positive product smoke/parity")
    )


def validate_full_finetune_target_contract(target: OTFullFinetuneTarget) raises:
    if not target.onetrainer_fine_tune_registered:
        raise Error(
            String("full-finetune contract: target has no OneTrainer FINE_TUNE registration: ")
            + target.key
        )
    if not target.shared_tensor_save_load_supported:
        raise Error("full-finetune contract: shared tensor save/load support missing")
    if not target.shared_optimizer_resume_sidecar_supported:
        raise Error("full-finetune contract: shared optimizer resume sidecar support missing")
    if target.product_loop_status == OT_FULL_FT_LOOP_WIRED:
        raise Error(
            String("full-finetune contract: product loop is marked wired without a model-specific proof: ")
            + target.key
        )


def validate_full_finetune_contract() raises:
    var targets = full_finetune_targets()
    if len(targets) != 9:
        raise Error("full-finetune contract: expected 9 target model families")
    for i in range(len(targets)):
        validate_full_finetune_target_contract(targets[i])
    validate_full_finetune_resume_sidecar_spec(
        default_full_finetune_resume_sidecar_spec()
    )
