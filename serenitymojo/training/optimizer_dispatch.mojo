# training/optimizer_dispatch.mojo — OneTrainer optimizer name -> Mojo backend.
#
# Groundwork only: this does not wire trainers yet. The config reader sidecar can
# feed the raw OneTrainer optimizer string here once its owned files are ready.
# Local target preset audit (2026-06-05): explicit optimizer names in
# /home/alex/OneTrainer/{training_presets,configs} and /home/alex/training_presets
# are ADAMW and ADAFACTOR; missing/null optimizer objects inherit OneTrainer's
# TrainOptimizerConfig.default_values() default of ADAMW.


comptime OPT_STATUS_SUPPORTED = 0
comptime OPT_STATUS_ALIASED = 1
comptime OPT_STATUS_UNSUPPORTED = 2

comptime OPT_BACKEND_UNSUPPORTED = 0
comptime OPT_BACKEND_FUSED_ADAMW = 1
comptime OPT_BACKEND_SCALAR_ADAMW = 2
comptime OPT_BACKEND_ADAFACTOR_HOST = 3
comptime OPT_BACKEND_SGD_MOMENTUM = 4
comptime OPT_BACKEND_LION = 5
comptime OPT_BACKEND_STABLE_ADAMW = 6
comptime OPT_BACKEND_PRODIGY_HOST_SINGLE_PARAM = 7
comptime OPT_BACKEND_RADAM_SCHEDULE_FREE_HOST = 8

comptime OPT_STATE_UNSUPPORTED = 0
comptime OPT_STATE_ADAM_M_V_F32 = 1
comptime OPT_STATE_SGD_MOMENTUM_F32 = 2
comptime OPT_STATE_LION_M_F32 = 3
comptime OPT_STATE_ADAFACTOR_FACTORED_HOST_F32 = 4
comptime OPT_STATE_PRODIGY_HOST_F32 = 5
comptime OPT_STATE_SCHEDULE_FREE_HOST_F32 = 6


@fieldwise_init
struct OptimizerDispatch(Movable, Writable):
    var requested: String
    var canonical: String
    var backend: Int
    var state_kind: Int
    var status: Int
    var note: String

    def is_supported(self) -> Bool:
        return self.status == OPT_STATUS_SUPPORTED

    def is_aliased(self) -> Bool:
        return self.status == OPT_STATUS_ALIASED

    def is_unsupported(self) -> Bool:
        return self.status == OPT_STATUS_UNSUPPORTED

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "OptimizerDispatch(requested=",
            self.requested,
            ", canonical=",
            self.canonical,
            ", backend=",
            optimizer_backend_name(self.backend),
            ", state=",
            optimizer_state_name(self.state_kind),
            ", status=",
            optimizer_status_name(self.status),
            ", note=",
            self.note,
            ")",
        )


def optimizer_status_name(status: Int) -> String:
    if status == OPT_STATUS_SUPPORTED:
        return "SUPPORTED"
    elif status == OPT_STATUS_ALIASED:
        return "ALIASED"
    return "UNSUPPORTED"


def optimizer_backend_name(backend: Int) -> String:
    if backend == OPT_BACKEND_FUSED_ADAMW:
        return "fused_adamw_step"
    elif backend == OPT_BACKEND_SCALAR_ADAMW:
        return "adamw_step"
    elif backend == OPT_BACKEND_ADAFACTOR_HOST:
        return "adafactor_host_step"
    elif backend == OPT_BACKEND_SGD_MOMENTUM:
        return "sgd_step"
    elif backend == OPT_BACKEND_LION:
        return "lion_step"
    elif backend == OPT_BACKEND_STABLE_ADAMW:
        return "stableadamw_step"
    elif backend == OPT_BACKEND_PRODIGY_HOST_SINGLE_PARAM:
        return "prodigy_host_single_param"
    elif backend == OPT_BACKEND_RADAM_SCHEDULE_FREE_HOST:
        return "radam_schedulefree_host"
    return "unsupported"


def optimizer_state_name(state_kind: Int) -> String:
    if state_kind == OPT_STATE_ADAM_M_V_F32:
        return "adam_m_v_f32"
    elif state_kind == OPT_STATE_SGD_MOMENTUM_F32:
        return "sgd_momentum_f32"
    elif state_kind == OPT_STATE_LION_M_F32:
        return "lion_m_f32"
    elif state_kind == OPT_STATE_ADAFACTOR_FACTORED_HOST_F32:
        return "adafactor_factored_host_f32"
    elif state_kind == OPT_STATE_PRODIGY_HOST_F32:
        return "prodigy_host_f32"
    elif state_kind == OPT_STATE_SCHEDULE_FREE_HOST_F32:
        return "schedule_free_host_f32"
    return "unsupported"


def optimizer_identifier_is_default(identifier: String) -> Bool:
    return (
        identifier == ""
        or identifier == "<missing>"
        or identifier == "None"
        or identifier == "null"
        or identifier == "NULL"
    )


def optimizer_dispatch_for_identifier(identifier: String) -> OptimizerDispatch:
    """Resolve a OneTrainer optimizer string to the current Mojo backend.

    Missing/null maps to ADAMW because OneTrainer's TrainOptimizerConfig default
    is Optimizer.ADAMW. Unsupported advanced/8-bit identifiers are explicit so a
    caller can fail closed instead of silently using the wrong optimizer.
    """
    if optimizer_identifier_is_default(identifier):
        return OptimizerDispatch(
            requested=identifier,
            canonical="ADAMW",
            backend=OPT_BACKEND_FUSED_ADAMW,
            state_kind=OPT_STATE_ADAM_M_V_F32,
            status=OPT_STATUS_SUPPORTED,
            note="OneTrainer default optimizer; product path uses fused AdamW",
        )
    elif identifier == "ADAMW":
        return OptimizerDispatch(
            requested=identifier,
            canonical="ADAMW",
            backend=OPT_BACKEND_FUSED_ADAMW,
            state_kind=OPT_STATE_ADAM_M_V_F32,
            status=OPT_STATUS_SUPPORTED,
            note="torch.optim.AdamW equivalent; scalar adamw_step remains the reference sibling",
        )
    elif identifier == "ADAFACTOR":
        return OptimizerDispatch(
            requested=identifier,
            canonical="ADAFACTOR",
            backend=OPT_BACKEND_ADAFACTOR_HOST,
            state_kind=OPT_STATE_ADAFACTOR_FACTORED_HOST_F32,
            status=OPT_STATUS_SUPPORTED,
            note="transformers.Adafactor equivalent; factored/elementwise host-F32 state",
        )
    elif identifier == "LION":
        return OptimizerDispatch(
            requested=identifier,
            canonical="LION",
            backend=OPT_BACKEND_LION,
            state_kind=OPT_STATE_LION_M_F32,
            status=OPT_STATUS_SUPPORTED,
            note="available but not used by the audited target presets",
        )
    elif identifier == "STABLEADAMW":
        return OptimizerDispatch(
            requested=identifier,
            canonical="STABLEADAMW",
            backend=OPT_BACKEND_STABLE_ADAMW,
            state_kind=OPT_STATE_ADAM_M_V_F32,
            status=OPT_STATUS_SUPPORTED,
            note="available local backend; not a current OneTrainer enum member",
        )
    elif identifier == "PRODIGY":
        return OptimizerDispatch(
            requested=identifier,
            canonical="PRODIGY",
            backend=OPT_BACKEND_PRODIGY_HOST_SINGLE_PARAM,
            state_kind=OPT_STATE_PRODIGY_HOST_F32,
            status=OPT_STATUS_ALIASED,
            note="single-parameter host backend only; not yet a multi-parameter product optimizer",
        )
    elif identifier == "SGD":
        return OptimizerDispatch(
            requested=identifier,
            canonical="SGD",
            backend=OPT_BACKEND_SGD_MOMENTUM,
            state_kind=OPT_STATE_SGD_MOMENTUM_F32,
            status=OPT_STATUS_ALIASED,
            note="matches OneTrainer SGD only for weight_decay=0; Mojo helper uses decoupled WD otherwise",
        )
    elif identifier == "RADAM_SCHEDULE_FREE":
        return OptimizerDispatch(
            requested=identifier,
            canonical="RADAM_SCHEDULE_FREE",
            backend=OPT_BACKEND_RADAM_SCHEDULE_FREE_HOST,
            state_kind=OPT_STATE_SCHEDULE_FREE_HOST_F32,
            status=OPT_STATUS_SUPPORTED,
            note="local EDv2 RAdamScheduleFree backend; not a OneTrainer preset identifier",
        )
    elif identifier == "SCHEDULE_FREE_ADAMW" or identifier == "SCHEDULE_FREE_SGD":
        return OptimizerDispatch(
            requested=identifier,
            canonical=identifier,
            backend=OPT_BACKEND_UNSUPPORTED,
            state_kind=OPT_STATE_UNSUPPORTED,
            status=OPT_STATUS_UNSUPPORTED,
            note="OneTrainer schedule-free AdamW/SGD backends are not implemented",
        )
    elif identifier == "ADAMW_8BIT" or identifier == "ADAM_8BIT" or identifier == "LION_8BIT":
        return OptimizerDispatch(
            requested=identifier,
            canonical=identifier,
            backend=OPT_BACKEND_UNSUPPORTED,
            state_kind=OPT_STATE_UNSUPPORTED,
            status=OPT_STATUS_UNSUPPORTED,
            note="8-bit optimizer-state backend is not implemented",
        )
    elif identifier == "ADAMW_ADV" or identifier == "LION_ADV" or identifier == "PRODIGY_ADV":
        return OptimizerDispatch(
            requested=identifier,
            canonical=identifier,
            backend=OPT_BACKEND_UNSUPPORTED,
            state_kind=OPT_STATE_UNSUPPORTED,
            status=OPT_STATUS_UNSUPPORTED,
            note="advanced OneTrainer modifier stack is not implemented",
        )
    else:
        return OptimizerDispatch(
            requested=identifier,
            canonical=identifier,
            backend=OPT_BACKEND_UNSUPPORTED,
            state_kind=OPT_STATE_UNSUPPORTED,
            status=OPT_STATUS_UNSUPPORTED,
            note="no Mojo product backend mapped for this OneTrainer optimizer",
        )
